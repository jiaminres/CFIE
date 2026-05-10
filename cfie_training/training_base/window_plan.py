"""训练窗口规划——定义 HotSetPlan、TrainableParamSpec、TrainingWindowBudget。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

from cfie_training.training_base.progress_state import digest_hot_set

ParamKind = Literal["dense", "moe", "other"]


def _require_non_negative_int(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


def _require_positive_int(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be >= 1")


def _require_non_empty_string(name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class TrainableParamSpec:
    """单个可训练参数的规格描述——供 HotSetScheduler 和 TrainingWindowPlanner 使用。"""
    param_id: str              # 参数唯一标识
    kind: ParamKind = "moe"    # 参数类型: dense / moe / other
    fp32_bytes: int = 0        # FP32 master 占用的 CPU/NVMe 字节数
    gpu_shadow_bytes: int = 0  # GPU shadow 占用字节数
    adam_bytes: int = 0        # Adam FP8 状态占用字节数
    gptq_requant_bytes: int = 0  # 训练后重新量化为 GPTQ 的字节数
    priority: float = 0.0      # 选择优先级（越高越优先训练）

    def __post_init__(self) -> None:
        _require_non_empty_string("param_id", self.param_id)
        if self.kind not in {"dense", "moe", "other"}: raise ValueError("kind must be dense, moe, or other")
        _require_non_negative_int("fp32_bytes", self.fp32_bytes)
        _require_non_negative_int("gpu_shadow_bytes", self.gpu_shadow_bytes)
        _require_non_negative_int("adam_bytes", self.adam_bytes)
        _require_non_negative_int("gptq_requant_bytes", self.gptq_requant_bytes)


@dataclass(frozen=True, slots=True)
class TrainingWindowBudget:
    window_steps: int = 50
    max_fp32_hot_bytes: int = 0
    max_gpu_shadow_bytes: int = 0
    max_adam_bytes: int = 0
    max_gptq_requant_bytes: int = 0
    min_params_per_window: int = 1

    def __post_init__(self) -> None:
        _require_positive_int("window_steps", self.window_steps)
        _require_non_negative_int("max_fp32_hot_bytes", self.max_fp32_hot_bytes)
        _require_non_negative_int(
            "max_gpu_shadow_bytes",
            self.max_gpu_shadow_bytes,
        )
        _require_non_negative_int("max_adam_bytes", self.max_adam_bytes)
        _require_non_negative_int(
            "max_gptq_requant_bytes",
            self.max_gptq_requant_bytes,
        )
        _require_non_negative_int("min_params_per_window", self.min_params_per_window)

    @classmethod
    def from_memory_plan(
        cls,
        *,
        hot_shadow_bytes: int,
        cpu_hot_budget_bytes: int = 0,
        bucket_count: int = 4,
        bucket_size_bytes: int = 1 << 20,
        window_steps: int = 50,
        min_params_per_window: int = 1,
        adam_budget_fraction: float = 0.5,
    ) -> "TrainingWindowBudget":
        return cls(
            window_steps=window_steps,
            max_fp32_hot_bytes=cpu_hot_budget_bytes,
            max_gpu_shadow_bytes=hot_shadow_bytes,
            max_adam_bytes=int(cpu_hot_budget_bytes * adam_budget_fraction),
            max_gptq_requant_bytes=int(hot_shadow_bytes * 2),
            min_params_per_window=min_params_per_window,
        )


@dataclass(frozen=True, slots=True)
class HotSetPlan:
    window_index: int
    start_step: int
    end_step_exclusive: int
    selected_params: tuple[TrainableParamSpec, ...]
    skipped_param_ids: tuple[str, ...]
    total_fp32_hot_bytes: int
    total_gpu_shadow_bytes: int
    total_adam_bytes: int
    total_gptq_requant_bytes: int

    @property
    def param_ids(self) -> tuple[str, ...]:
        return tuple(param.param_id for param in self.selected_params)

    @property
    def hot_set_digest(self) -> str:
        return digest_hot_set(self.param_ids)

    def touched_summary(
        self,
        touched_param_ids: Iterable[str],
    ) -> "TouchedParamSummary":
        touched_set = set(touched_param_ids)
        touched = tuple(
            param_id
            for param_id in self.param_ids
            if param_id in touched_set
        )
        return TouchedParamSummary(
            window_index=self.window_index,
            touched_param_ids=touched,
            touched_digest=digest_hot_set(touched),
        )


@dataclass(frozen=True, slots=True)
class TouchedParamSummary:
    window_index: int
    touched_param_ids: tuple[str, ...]
    touched_digest: str


@dataclass(slots=True)
class TrainingWindowPlanner:
    budget: TrainingWindowBudget

    def plan_window(
        self,
        candidates: Iterable[TrainableParamSpec],
        *,
        global_step: int,
    ) -> HotSetPlan:
        _require_non_negative_int("global_step", global_step)
        ordered_candidates = sorted(
            candidates,
            key=lambda item: (-item.priority, item.param_id),
        )

        selected: list[TrainableParamSpec] = []
        skipped: list[str] = []
        total_fp32 = 0
        total_gpu = 0
        total_adam = 0
        total_gptq = 0

        for candidate in ordered_candidates:
            next_fp32 = total_fp32 + candidate.fp32_bytes
            next_gpu = total_gpu + candidate.gpu_shadow_bytes
            next_adam = total_adam + candidate.adam_bytes
            next_gptq = total_gptq + candidate.gptq_requant_bytes

            if (
                self._exceeds(self.budget.max_fp32_hot_bytes, next_fp32)
                or self._exceeds(self.budget.max_gpu_shadow_bytes, next_gpu)
                or self._exceeds(self.budget.max_adam_bytes, next_adam)
                or self._exceeds(self.budget.max_gptq_requant_bytes, next_gptq)
            ):
                skipped.append(candidate.param_id)
                continue

            selected.append(candidate)
            total_fp32 = next_fp32
            total_gpu = next_gpu
            total_adam = next_adam
            total_gptq = next_gptq

        if len(selected) < self.budget.min_params_per_window:
            raise ValueError(
                "not enough trainable params fit in the current window budget"
            )

        window_index = global_step // self.budget.window_steps
        start_step = window_index * self.budget.window_steps
        end_step = start_step + self.budget.window_steps

        return HotSetPlan(
            window_index=window_index,
            start_step=start_step,
            end_step_exclusive=end_step,
            selected_params=tuple(selected),
            skipped_param_ids=tuple(skipped),
            total_fp32_hot_bytes=total_fp32,
            total_gpu_shadow_bytes=total_gpu,
            total_adam_bytes=total_adam,
            total_gptq_requant_bytes=total_gptq,
        )

    @staticmethod
    def _exceeds(limit: int, value: int) -> bool:
        return limit > 0 and value > limit
