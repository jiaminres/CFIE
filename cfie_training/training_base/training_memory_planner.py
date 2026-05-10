"""训练显存/内存规划——根据 GPU/CPU/NVMe 容量生成 MemoryPlan（设计文档 Section 5-7）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from cfie_training.training_base.peak_monitor import (
    TrainingResourcePeaks,
    TrainingResourceSnapshot,
)

FP32_BYTES = 4
GPTQ_INT4_OVERHEAD_FACTOR = 0.55


def _require_non_empty_string(name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_non_negative_int(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


def _require_positive_int(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be >= 1")


def _require_positive_float(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0")


def _require_non_negative_float(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


@dataclass(frozen=True, slots=True)
class MemoryProfile:
    total_vram_bytes: int
    total_cpu_ram_bytes: int
    nvme_available_bytes: int = 0
    pinned_memory_cap_bytes: int = 0
    max_gpu_reserved_ratio: float = 0.88
    emergency_reserve_bytes: int = 1 << 30
    forward_dtype_bytes: int = 2

    def __post_init__(self) -> None:
        _require_positive_int("total_vram_bytes", self.total_vram_bytes)
        _require_positive_int("total_cpu_ram_bytes", self.total_cpu_ram_bytes)
        _require_non_negative_int("nvme_available_bytes", self.nvme_available_bytes)
        _require_non_negative_int(
            "pinned_memory_cap_bytes",
            self.pinned_memory_cap_bytes,
        )
        if not 0.0 < self.max_gpu_reserved_ratio <= 1.0:
            raise ValueError("max_gpu_reserved_ratio must be in (0, 1]")
        _require_non_negative_int(
            "emergency_reserve_bytes",
            self.emergency_reserve_bytes,
        )
        _require_positive_int("forward_dtype_bytes", self.forward_dtype_bytes)

    @property
    def usable_vram_bytes(self) -> int:
        return max(
            0,
            int(self.total_vram_bytes * self.max_gpu_reserved_ratio)
            - self.emergency_reserve_bytes,
        )

    @property
    def pinned_memory_limit_bytes(self) -> int:
        default = min(int(self.total_cpu_ram_bytes * 0.12), 32 << 30)
        if self.pinned_memory_cap_bytes > 0:
            return self.pinned_memory_cap_bytes
        return default


@dataclass(frozen=True, slots=True)
class ModelDimensions:
    num_layers: int
    num_experts: int
    hidden_size: int
    intermediate_size: int
    tp_size: int = 1
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    vocab_size: int = 0
    max_seq_len: int = 8192

    def __post_init__(self) -> None:
        _require_positive_int("num_layers", self.num_layers)
        _require_positive_int("num_experts", self.num_experts)
        _require_positive_int("hidden_size", self.hidden_size)
        _require_positive_int("intermediate_size", self.intermediate_size)
        _require_positive_int("tp_size", self.tp_size)
        if self.intermediate_size % self.tp_size:
            raise ValueError("intermediate_size must be divisible by tp_size")

    @property
    def per_rank_intermediate(self) -> int:
        return self.intermediate_size // self.tp_size

    @property
    def expert_w13_elements(self) -> int:
        return 2 * self.per_rank_intermediate * self.hidden_size

    @property
    def expert_w2_elements(self) -> int:
        return self.hidden_size * self.per_rank_intermediate

    @property
    def total_expert_fp32_bytes(self) -> int:
        w13 = self.expert_w13_elements
        w2 = self.expert_w2_elements
        return self.num_layers * self.num_experts * (w13 + w2) * FP32_BYTES


@dataclass(frozen=True, slots=True)
class MemoryPlan:
    vram_budget_bytes: int
    cpu_hot_budget_bytes: int
    bucket_count: int
    bucket_size_bytes: int
    expert_cache_bytes: int
    hot_shadow_bytes: int
    activation_workspace_bytes: int
    kernel_workspace_bytes: int
    fragmentation_reserve_bytes: int
    emergency_reserve_bytes: int
    dense_forward_bytes: int = 0
    communication_bytes: int = 0

    def __post_init__(self) -> None:
        _require_non_negative_int("vram_budget_bytes", self.vram_budget_bytes)
        _require_non_negative_int("cpu_hot_budget_bytes", self.cpu_hot_budget_bytes)
        _require_positive_int("bucket_count", self.bucket_count)
        _require_positive_int("bucket_size_bytes", self.bucket_size_bytes)
        _require_non_negative_int("expert_cache_bytes", self.expert_cache_bytes)
        _require_positive_int("hot_shadow_bytes", self.hot_shadow_bytes)
        _require_non_negative_int(
            "activation_workspace_bytes",
            self.activation_workspace_bytes,
        )
        _require_non_negative_int(
            "kernel_workspace_bytes",
            self.kernel_workspace_bytes,
        )
        _require_non_negative_int(
            "fragmentation_reserve_bytes",
            self.fragmentation_reserve_bytes,
        )
        _require_non_negative_int(
            "emergency_reserve_bytes",
            self.emergency_reserve_bytes,
        )

    @property
    def grad_ring_bytes(self) -> int:
        return self.bucket_count * self.bucket_size_bytes

    @property
    def total_vram_planned_bytes(self) -> int:
        return (
            self.dense_forward_bytes
            + self.hot_shadow_bytes
            + self.expert_cache_bytes
            + self.grad_ring_bytes
            + self.activation_workspace_bytes
            + self.kernel_workspace_bytes
            + self.communication_bytes
            + self.fragmentation_reserve_bytes
            + self.emergency_reserve_bytes
        )


@dataclass(slots=True)
class TrainingMemoryPlanner:
    profile: MemoryProfile
    model_dims: ModelDimensions

    def build_initial_plan(self) -> MemoryPlan:
        usable = self.profile.usable_vram_bytes
        if usable < (1 << 30):
            raise ValueError("VRAM too small for training")

        if usable <= 34_359_738_368:
            defaults = _plan_for_32gib(
                usable,
                self.model_dims,
                self.profile.forward_dtype_bytes,
            )
        elif usable <= 52_000_000_000:
            defaults = _plan_for_48gib(
                usable,
                self.model_dims,
                self.profile.forward_dtype_bytes,
            )
        else:
            defaults = _plan_for_80gib(
                usable,
                self.model_dims,
                self.profile.forward_dtype_bytes,
            )

        return MemoryPlan(
            vram_budget_bytes=usable,
            cpu_hot_budget_bytes=_cpu_hot_budget(
                self.profile.total_cpu_ram_bytes
            ),
            **defaults,
        )

    def replan(
        self,
        peak_report: TrainingResourcePeaks,
        current_plan: MemoryPlan,
    ) -> MemoryPlan:
        from dataclasses import replace

        hot_shadow = current_plan.hot_shadow_bytes
        activation_ws = current_plan.activation_workspace_bytes
        cpu_budget = current_plan.cpu_hot_budget_bytes

        if (
            peak_report.max_gpu_reserved_bytes
            > current_plan.vram_budget_bytes * 0.95
        ):
            hot_shadow = max(
                int(current_plan.hot_shadow_bytes * 0.7),
                current_plan.bucket_size_bytes,
            )
            activation_ws = max(
                int(current_plan.activation_workspace_bytes * 0.8),
                current_plan.bucket_size_bytes,
            )

        if peak_report.max_pinned_bytes > self.profile.pinned_memory_limit_bytes:
            cpu_budget = max(
                int(current_plan.cpu_hot_budget_bytes * 0.8),
                current_plan.bucket_size_bytes * current_plan.bucket_count,
            )

        plan = replace(
            current_plan,
            hot_shadow_bytes=hot_shadow,
            activation_workspace_bytes=activation_ws,
            cpu_hot_budget_bytes=cpu_budget,
        )
        self.validate_plan(plan, self.profile)
        return plan

    @staticmethod
    def validate_plan(plan: MemoryPlan, profile: MemoryProfile) -> None:
        if plan.total_vram_planned_bytes > profile.total_vram_bytes:
            raise ValueError(
                f"plan requires {plan.total_vram_planned_bytes} vram bytes, "
                f"but only {profile.total_vram_bytes} are available"
            )
        if plan.vram_budget_bytes > profile.usable_vram_bytes:
            raise ValueError(
                f"plan vram budget {plan.vram_budget_bytes} exceeds "
                f"usable vram {profile.usable_vram_bytes}"
            )


def _plan_for_32gib(
    usable: int,
    dims: ModelDimensions,
    fp_dtype_bytes: int,
) -> dict[str, int]:
    fragmentation_reserve = 2_684_354_560
    emergency_reserve = max(1_610_612_736, int(usable * 0.06))
    gb = 1 << 30
    return {
        "bucket_count": 4,
        "bucket_size_bytes": 512 * 1 << 20,
        "expert_cache_bytes": min(3 * gb, usable // 8),
        "hot_shadow_bytes": min(4 * gb, usable // 4),
        "activation_workspace_bytes": min(
            int(4.5 * gb),
            usable // 5,
        ),
        "kernel_workspace_bytes": min(2 * gb, usable // 8),
        "dense_forward_bytes": min(10 * gb, usable // 2),
        "communication_bytes": 0,
        "fragmentation_reserve_bytes": fragmentation_reserve,
        "emergency_reserve_bytes": emergency_reserve,
    }


def _plan_for_48gib(
    usable: int,
    dims: ModelDimensions,
    fp_dtype_bytes: int,
) -> dict[str, int]:
    gb = 1 << 30
    return {
        "bucket_count": 4,
        "bucket_size_bytes": 1 * gb,
        "expert_cache_bytes": min(6 * gb, usable // 4),
        "hot_shadow_bytes": min(6 * gb, usable // 3),
        "activation_workspace_bytes": min(8 * gb, usable // 4),
        "kernel_workspace_bytes": min(3 * gb, usable // 8),
        "dense_forward_bytes": min(12 * gb, usable // 2),
        "communication_bytes": 0,
        "fragmentation_reserve_bytes": 3 * gb,
        "emergency_reserve_bytes": max(int(1.5 * gb), int(usable * 0.06)),
    }


def _plan_for_80gib(
    usable: int,
    dims: ModelDimensions,
    fp_dtype_bytes: int,
) -> dict[str, int]:
    gb = 1 << 30
    return {
        "bucket_count": 4,
        "bucket_size_bytes": 1 * gb,
        "expert_cache_bytes": min(10 * gb, usable // 3),
        "hot_shadow_bytes": min(12 * gb, usable // 3),
        "activation_workspace_bytes": min(16 * gb, usable // 4),
        "kernel_workspace_bytes": min(4 * gb, usable // 10),
        "dense_forward_bytes": min(16 * gb, usable // 3),
        "communication_bytes": 0,
        "fragmentation_reserve_bytes": 4 * gb,
        "emergency_reserve_bytes": max(int(2 * gb), int(usable * 0.06)),
    }


def _cpu_hot_budget(total_cpu_ram_bytes: int) -> int:
    return int(total_cpu_ram_bytes * 0.40)
