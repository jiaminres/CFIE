"""Tests for activation checkpoint segment planning."""

from __future__ import annotations

import pytest

from cfie_training.training_base.activation_checkpoint import (
    ActivationCheckpointPlanner,
    ActivationPeakEstimator,
    CheckpointSegment,
    SegmentGradEstimator,
)
from cfie_training.training_base.window_plan import TrainableParamSpec


def _make_param_specs(
    num_layers: int,
    params_per_layer: int = 2,
    *,
    fp32_bytes: int = 1000,
) -> tuple[TrainableParamSpec, ...]:
    specs: list[TrainableParamSpec] = []
    for layer_id in range(num_layers):
        for pid in range(params_per_layer):
            specs.append(
                TrainableParamSpec(
                    param_id=f"layers.{layer_id}.experts.0.w{pid + 1}",
                    kind="moe",
                    fp32_bytes=fp32_bytes,
                    gpu_shadow_bytes=fp32_bytes // 2,
                    adam_bytes=fp32_bytes // 4,
                )
            )
    return tuple(specs)


class TestSegmentGradEstimator:
    def test_estimates_total_fp32_bytes(self) -> None:
        estimator = SegmentGradEstimator(grad_dtype_bytes=4)
        specs = _make_param_specs(3, params_per_layer=2, fp32_bytes=100)
        result = estimator.estimate_grad_bytes(specs)
        assert result == 3 * 2 * 100

    def test_requires_positive_dtype_bytes(self) -> None:
        with pytest.raises(ValueError, match="dtype"):
            SegmentGradEstimator(grad_dtype_bytes=0)


class TestActivationPeakEstimator:
    def test_estimates_peak_scales_with_tokens(self) -> None:
        est_small = ActivationPeakEstimator(
            max_tokens=128,
            hidden_size=4096,
        )
        est_large = ActivationPeakEstimator(
            max_tokens=256,
            hidden_size=4096,
        )
        small_peak = est_small.estimate_peak_bytes()
        large_peak = est_large.estimate_peak_bytes()
        assert large_peak > small_peak * 1.5

    def test_includes_attention_and_moe_overhead(self) -> None:
        est = ActivationPeakEstimator(
            max_tokens=128,
            hidden_size=4096,
            num_attention_heads=32,
            moe_topk=8,
        )
        peak = est.estimate_peak_bytes()
        assert peak > 0


class TestActivationCheckpointPlanner:
    def test_planner_segments_fit_within_bucket_bounds(self) -> None:
        specs = _make_param_specs(6, params_per_layer=1, fp32_bytes=600)
        planner = ActivationCheckpointPlanner()
        segments = planner.plan_segments(
            specs,
            bucket_size_bytes=1000,
            num_buckets=4,
            vram_budget_bytes=1 << 30,
            static_vram_bytes=10 << 20,
        )

        min_seg = int(1000 * 0.8)
        max_seg = (4 - 1) * 1000
        for seg in segments:
            assert seg.grad_bytes <= max_seg

        assert len(segments) >= 1

    def test_planner_splits_large_params(self) -> None:
        large_spec = TrainableParamSpec(
            param_id="layers.0.experts.0.w13_weight",
            kind="moe",
            fp32_bytes=10000,
        )
        planner = ActivationCheckpointPlanner()
        segments = planner.plan_segments(
            [large_spec],
            bucket_size_bytes=2000,
            num_buckets=4,
            vram_budget_bytes=10 << 30,
        )
        assert len(segments) >= 1
        for seg in segments:
            assert seg.grad_bytes <= (4 - 1) * 2000

    def test_planner_rejects_over_budget(self) -> None:
        specs = _make_param_specs(2, fp32_bytes=10_000_000)
        planner = ActivationCheckpointPlanner()
        with pytest.raises(ValueError, match="exceeds"):
            planner.plan_segments(
                specs,
                bucket_size_bytes=100_000_000,
                num_buckets=4,
                vram_budget_bytes=1_000_000,
                static_vram_bytes=0,
            )

    def test_empty_specs(self) -> None:
        planner = ActivationCheckpointPlanner()
        segments = planner.plan_segments(
            [],
            bucket_size_bytes=1000,
            num_buckets=4,
            vram_budget_bytes=1 << 30,
        )
        assert segments == ()

    def test_total_workspace_is_max_peak(self) -> None:
        planner = ActivationCheckpointPlanner()
        seg1 = CheckpointSegment(
            layer_ids=(0,),
            grad_bytes=100,
            activation_peak_bytes=2000,
            total_peak_bytes=5000,
            trainable_param_ids=("a",),
        )
        seg2 = CheckpointSegment(
            layer_ids=(1,),
            grad_bytes=150,
            activation_peak_bytes=500,
            total_peak_bytes=3000,
            trainable_param_ids=("b",),
        )
        assert planner.estimate_total_workspace_bytes([seg1, seg2]) == 2000


class TestCheckpointSegment:
    def test_requires_non_empty_layers(self) -> None:
        with pytest.raises(ValueError, match="layer_ids"):
            CheckpointSegment(
                layer_ids=(),
                grad_bytes=0,
                activation_peak_bytes=0,
                total_peak_bytes=0,
                trainable_param_ids=(),
            )
