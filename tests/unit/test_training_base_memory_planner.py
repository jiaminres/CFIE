"""Tests for the GPU/CPU memory planner."""

from __future__ import annotations

import pytest

from cfie_training.training_base.peak_monitor import TrainingResourcePeaks
from cfie_training.training_base.training_memory_planner import (
    MemoryPlan,
    MemoryProfile,
    ModelDimensions,
    TrainingMemoryPlanner,
)
from cfie_training.training_base.window_plan import TrainingWindowBudget

GB = 1 << 30


def _standard_profile(
    *,
    vram_gib: int = 32,
    cpu_ram_gib: int = 256,
) -> MemoryProfile:
    return MemoryProfile(
        total_vram_bytes=vram_gib * GB,
        total_cpu_ram_bytes=cpu_ram_gib * GB,
    )


def _122b_dims() -> ModelDimensions:
    return ModelDimensions(
        num_layers=60,
        num_experts=160,
        hidden_size=4096,
        intermediate_size=1024,
        tp_size=1,
    )


class TestMemoryProfile:
    def test_usable_vram_respects_ratio_and_reserve(self) -> None:
        profile = MemoryProfile(
            total_vram_bytes=32 * GB,
            total_cpu_ram_bytes=256 * GB,
            max_gpu_reserved_ratio=0.88,
            emergency_reserve_bytes=GB,
        )
        expected = int(32 * GB * 0.88) - GB
        assert abs(profile.usable_vram_bytes - expected) < 100_000_000

    def test_pinned_memory_limit_default(self) -> None:
        profile = MemoryProfile(
            total_vram_bytes=32 * GB,
            total_cpu_ram_bytes=256 * GB,
        )
        limit = profile.pinned_memory_limit_bytes
        expected = min(int(256 * GB * 0.12), 32 * GB)
        assert limit == expected

    def test_rejects_invalid_ratio(self) -> None:
        with pytest.raises(ValueError, match="ratio"):
            MemoryProfile(
                total_vram_bytes=32 * GB,
                total_cpu_ram_bytes=256 * GB,
                max_gpu_reserved_ratio=1.5,
            )


class TestModelDimensions:
    def test_per_rank_intermediate_divides_by_tp(self) -> None:
        dims = ModelDimensions(
            num_layers=10,
            num_experts=8,
            hidden_size=512,
            intermediate_size=256,
            tp_size=4,
        )
        assert dims.per_rank_intermediate == 64

    def test_rejects_indivisible_tp(self) -> None:
        with pytest.raises(ValueError, match="divisible"):
            ModelDimensions(
                num_layers=10,
                num_experts=8,
                hidden_size=512,
                intermediate_size=257,
                tp_size=4,
            )

    def test_total_expert_fp32_bytes_computes_correctly(self) -> None:
        dims = ModelDimensions(
            num_layers=2,
            num_experts=3,
            hidden_size=64,
            intermediate_size=32,
            tp_size=1,
        )
        w13_elements = dims.expert_w13_elements
        w2_elements = dims.expert_w2_elements
        expected = 2 * 3 * (w13_elements + w2_elements) * 4
        assert dims.total_expert_fp32_bytes == expected


class TestTrainingMemoryPlanner:
    def test_builds_initial_plan_for_32gib(self) -> None:
        planner = TrainingMemoryPlanner(
            _standard_profile(vram_gib=32),
            _122b_dims(),
        )
        plan = planner.build_initial_plan()
        assert plan.bucket_count == 4
        assert plan.bucket_size_bytes > 0
        assert plan.hot_shadow_bytes > 0
        assert plan.expert_cache_bytes > 0
        assert plan.emergency_reserve_bytes > 0

    def test_plan_total_vram_does_not_exceed_budget(self) -> None:
        profile = _standard_profile(vram_gib=32)
        planner = TrainingMemoryPlanner(profile, _122b_dims())
        plan = planner.build_initial_plan()
        assert plan.total_vram_planned_bytes <= profile.total_vram_bytes

    def test_replan_reduces_hot_shadow_on_gpu_pressure(self) -> None:
        profile = _standard_profile(vram_gib=32)
        planner = TrainingMemoryPlanner(profile, _122b_dims())
        plan = planner.build_initial_plan()

        peaks = TrainingResourcePeaks(
            max_gpu_reserved_bytes=int(plan.vram_budget_bytes * 0.97),
            snapshots_seen=10,
        )
        new_plan = planner.replan(peaks, plan)
        assert new_plan.hot_shadow_bytes < plan.hot_shadow_bytes

    def test_replan_reduces_cpu_budget_on_pinned_pressure(self) -> None:
        profile = _standard_profile(vram_gib=32)
        planner = TrainingMemoryPlanner(profile, _122b_dims())
        plan = planner.build_initial_plan()

        peaks = TrainingResourcePeaks(
            max_pinned_bytes=profile.pinned_memory_limit_bytes + 1,
            snapshots_seen=10,
        )
        new_plan = planner.replan(peaks, plan)
        assert new_plan.cpu_hot_budget_bytes < plan.cpu_hot_budget_bytes

    def test_validate_rejects_over_budget(self) -> None:
        plan = MemoryPlan(
            vram_budget_bytes=30 * GB,
            cpu_hot_budget_bytes=100 * GB,
            bucket_count=4,
            bucket_size_bytes=512 << 20,
            expert_cache_bytes=20 * GB,
            hot_shadow_bytes=4 * GB,
            activation_workspace_bytes=4 * GB,
            kernel_workspace_bytes=2 * GB,
            fragmentation_reserve_bytes=2 * GB,
            emergency_reserve_bytes=1 * GB,
        )
        profile = MemoryProfile(
            total_vram_bytes=10 * GB,
            total_cpu_ram_bytes=256 * GB,
        )
        with pytest.raises(ValueError, match="vram"):
            TrainingMemoryPlanner.validate_plan(plan, profile)

    def test_training_window_budget_from_memory_plan(self) -> None:
        budget = TrainingWindowBudget.from_memory_plan(
            hot_shadow_bytes=4 * GB,
            cpu_hot_budget_bytes=100 * GB,
            bucket_size_bytes=512 << 20,
            window_steps=50,
        )
        assert budget.max_fp32_hot_bytes == 100 * GB
        assert budget.max_gpu_shadow_bytes == 4 * GB
        assert budget.max_adam_bytes == int(100 * GB * 0.5)
        assert budget.window_steps == 50

    def test_rejects_vram_too_small(self) -> None:
        profile = MemoryProfile(
            total_vram_bytes=100_000_000,
            total_cpu_ram_bytes=256 * GB,
        )
        planner = TrainingMemoryPlanner(profile, _122b_dims())
        with pytest.raises(ValueError, match="small"):
            planner.build_initial_plan()
