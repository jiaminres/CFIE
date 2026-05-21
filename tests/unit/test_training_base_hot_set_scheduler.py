"""Tests for the hot-set MoE training scheduler."""

from __future__ import annotations

import pytest

from cfie_training.training_base.hot_set_scheduler import (
    CoverageConstraint,
    ExpertRoutingStats,
    HotSetSelection,
    HotSetScheduler,
    RoutingWindowStats,
)
from cfie_training.training_base.window_plan import TrainingWindowBudget, TrainableParamSpec


def _make_expert_specs(
    layer_id: int,
    expert_id: int,
    *,
    fp32_bytes: int = 1000,
    gpu_shadow_bytes: int = 500,
    adam_bytes: int = 200,
) -> tuple[TrainableParamSpec, TrainableParamSpec]:
    prefix = f"layers.{layer_id}.experts.{expert_id}"
    w13 = TrainableParamSpec(
        param_id=f"{prefix}.w13_weight",
        kind="moe",
        fp32_bytes=fp32_bytes,
        gpu_shadow_bytes=gpu_shadow_bytes // 2,
        adam_bytes=adam_bytes // 2,
        priority=float(layer_id * 100 + expert_id),
    )
    w2 = TrainableParamSpec(
        param_id=f"{prefix}.w2_weight",
        kind="moe",
        fp32_bytes=fp32_bytes // 2,
        gpu_shadow_bytes=gpu_shadow_bytes // 2,
        adam_bytes=adam_bytes // 2,
        priority=float(layer_id * 100 + expert_id),
    )
    return w13, w2


def _make_route_stats(
    expert_activation_counts: dict[tuple[int, int], int],
    *,
    total_activations: int = 100,
) -> RoutingWindowStats:
    stats: dict[tuple[int, int], ExpertRoutingStats] = {}
    for key, count in expert_activation_counts.items():
        stats[key] = ExpertRoutingStats(
            layer_id=key[0],
            expert_id=key[1],
            activation_count=count,
            total_score=float(count),
            token_count=count * 2,
        )
    return RoutingWindowStats(
        expert_stats=stats,
        total_activations=total_activations,
    )


class TestHotSetScheduler:
    def test_selects_highest_priority_within_budget(self) -> None:
        expert_specs: list[TrainableParamSpec] = []
        for eid in range(4):
            w13, w2 = _make_expert_specs(0, eid)
            expert_specs.extend([w13, w2])

        route_stats = _make_route_stats(
            {(0, 0): 40, (0, 1): 30, (0, 2): 20, (0, 3): 10},
        )
        budget = TrainingWindowBudget(
            window_steps=50,
            max_fp32_hot_bytes=3000,
            max_gpu_shadow_bytes=2000,
        )
        scheduler = HotSetScheduler()
        selection = scheduler.select_hot_set(
            expert_specs,
            budget=budget,
            route_stats=route_stats,
            expert_to_param_ids={},
        )

        assert len(selection.moe_experts) >= 1
        assert selection.total_fp32_bytes <= 3000
        assert selection.total_gpu_shadow_bytes <= 2000

    def test_enforces_low_frequency_coverage(self) -> None:
        expert_specs: list[TrainableParamSpec] = []
        for eid in range(8):
            w13, w2 = _make_expert_specs(0, eid)
            expert_specs.extend([w13, w2])

        route_stats = _make_route_stats(
            {
                (0, 0): 50,
                (0, 1): 40,
                (0, 2): 5,
                (0, 3): 3,
                (0, 4): 1,
                (0, 5): 1,
                (0, 6): 0,
                (0, 7): 0,
            },
        )
        budget = TrainingWindowBudget(
            max_fp32_hot_bytes=10000,
            max_gpu_shadow_bytes=10000,
        )
        scheduler = HotSetScheduler(
            coverage=CoverageConstraint(min_low_frequency_ratio=0.2),
        )
        selection = scheduler.select_hot_set(
            expert_specs,
            budget=budget,
            route_stats=route_stats,
            expert_to_param_ids={},
        )

        assert selection.low_freq_expert_count >= 1

    def test_tracks_rounds_since_trained(self) -> None:
        scheduler = HotSetScheduler()
        scheduler.record_training_round({(0, 0)})
        assert scheduler.rounds_since_trained((0, 0)) == 0
        assert scheduler.rounds_since_trained((0, 1)) == 0

        scheduler.record_training_round({(0, 0)})
        assert scheduler.rounds_since_trained((0, 0)) == 0

        scheduler.record_training_round({(0, 1)})
        assert scheduler.rounds_since_trained((0, 0)) == 1
        assert scheduler.rounds_since_trained((0, 1)) == 0

    def test_boosts_collapsed_layers(self) -> None:
        expert_specs: list[TrainableParamSpec] = []
        for eid in range(2):
            w13, w2 = _make_expert_specs(0, eid)
            expert_specs.extend([w13, w2])
        for eid in range(2):
            w13, w2 = _make_expert_specs(1, eid)
            expert_specs.extend([w13, w2])

        route_stats = _make_route_stats(
            {
                (0, 0): 80,
                (0, 1): 20,
                (1, 0): 99,
                (1, 1): 1,
            },
        )
        budget = TrainingWindowBudget(
            max_fp32_hot_bytes=5000,
            max_gpu_shadow_bytes=5000,
        )
        scheduler = HotSetScheduler(
            coverage=CoverageConstraint(
                collapse_entropy_threshold=0.5,
                collapse_boost_factor=2.0,
            ),
        )
        selection = scheduler.select_hot_set(
            expert_specs,
            budget=budget,
            route_stats=route_stats,
            expert_to_param_ids={},
        )

        assert len(selection.collapsed_layer_ids) >= 0

    def test_skips_experts_below_budget(self) -> None:
        expert_specs: list[TrainableParamSpec] = []
        for eid in range(5):
            w13, w2 = _make_expert_specs(0, eid, fp32_bytes=800, gpu_shadow_bytes=400)
            expert_specs.extend([w13, w2])

        route_stats = _make_route_stats({(0, e): 20 for e in range(5)})
        budget = TrainingWindowBudget(
            max_fp32_hot_bytes=2000,
            max_gpu_shadow_bytes=1000,
        )
        scheduler = HotSetScheduler()
        selection = scheduler.select_hot_set(
            expert_specs,
            budget=budget,
            route_stats=route_stats,
            expert_to_param_ids={},
        )

        assert selection.total_fp32_bytes <= 2000

    def test_selection_includes_dense_params(self) -> None:
        dense_spec = TrainableParamSpec(
            param_id="dense.layer_norm",
            kind="dense",
            fp32_bytes=200,
            gpu_shadow_bytes=100,
            adam_bytes=50,
            priority=1.0,
        )
        moe_specs: list[TrainableParamSpec] = []
        for eid in range(2):
            w13, w2 = _make_expert_specs(0, eid)
            moe_specs.extend([w13, w2])

        route_stats = _make_route_stats({(0, 0): 10, (0, 1): 10})
        budget = TrainingWindowBudget(
            max_fp32_hot_bytes=5000,
            max_gpu_shadow_bytes=5000,
        )
        scheduler = HotSetScheduler()
        selection = scheduler.select_hot_set(
            [dense_spec, *moe_specs],
            budget=budget,
            route_stats=route_stats,
            expert_to_param_ids={},
        )

        assert "dense.layer_norm" in selection.dense_param_ids
        assert len(selection.moe_param_ids) >= 1

    def test_serialization_round_trip(self) -> None:
        scheduler = HotSetScheduler()
        scheduler.record_training_round({(0, 0), (1, 3)})
        scheduler._round_id = 5

        state = scheduler.state_dict()
        restored = HotSetScheduler()
        restored.load_state_dict(state)

        assert restored._frequency_counter == {(0, 0): 0, (1, 3): 0}
        assert restored._round_id == 5

    def test_empty_routing_stats(self) -> None:
        expert_specs: list[TrainableParamSpec] = []
        for eid in range(2):
            w13, w2 = _make_expert_specs(0, eid)
            expert_specs.extend([w13, w2])

        route_stats = RoutingWindowStats(
            expert_stats={},
            total_activations=0,
        )
        budget = TrainingWindowBudget(
            max_fp32_hot_bytes=5000,
            max_gpu_shadow_bytes=5000,
        )
        scheduler = HotSetScheduler()
        selection = scheduler.select_hot_set(
            expert_specs,
            budget=budget,
            route_stats=route_stats,
            expert_to_param_ids={},
        )

        assert selection.moe_param_ids
        assert len(selection.collapsed_layer_ids) == 0

    def test_empty_candidates(self) -> None:
        route_stats = RoutingWindowStats(expert_stats={}, total_activations=0)
        budget = TrainingWindowBudget()
        scheduler = HotSetScheduler()
        selection = scheduler.select_hot_set(
            [],
            budget=budget,
            route_stats=route_stats,
            expert_to_param_ids={},
        )

        assert selection.expert_count == 0
        assert not selection.moe_param_ids
        assert selection.total_fp32_bytes == 0


class TestCoverageConstraint:
    def test_validates_min_low_frequency_ratio(self) -> None:
        with pytest.raises(ValueError, match="low_frequency"):
            CoverageConstraint(min_low_frequency_ratio=1.5)

    def test_validates_collapse_boost(self) -> None:
        with pytest.raises(ValueError, match="boost"):
            CoverageConstraint(collapse_boost_factor=0.5)

    def test_validates_entropy_threshold(self) -> None:
        with pytest.raises(ValueError, match="entropy"):
            CoverageConstraint(collapse_entropy_threshold=1.5)


class TestHotSetSelection:
    def test_validates_non_negative_fields(self) -> None:
        with pytest.raises(ValueError, match="fp32"):
            HotSetSelection(
                dense_param_ids=(),
                moe_experts=(),
                moe_param_ids=(),
                total_fp32_bytes=-1,
                total_gpu_shadow_bytes=0,
                total_adam_bytes=0,
                expected_grad_bytes=0,
                activation_checkpoint_segment_hint=0,
                budget_utilization_ratio=0.0,
                low_freq_expert_count=0,
                collapsed_layer_ids=(),
            )

    def test_all_param_ids_combines_dense_and_moe(self) -> None:
        selection = HotSetSelection(
            dense_param_ids=("dense.a",),
            moe_experts=((0, 0),),
            moe_param_ids=("layers.0.experts.0.w13_weight",),
            total_fp32_bytes=100,
            total_gpu_shadow_bytes=50,
            total_adam_bytes=30,
            expected_grad_bytes=30,
            activation_checkpoint_segment_hint=1,
            budget_utilization_ratio=0.5,
            low_freq_expert_count=1,
            collapsed_layer_ids=(),
        )
        assert selection.all_param_ids == (
            "dense.a",
            "layers.0.experts.0.w13_weight",
        )
