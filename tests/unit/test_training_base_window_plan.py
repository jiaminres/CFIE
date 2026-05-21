"""Unit tests for training-window hot-set planning."""

from __future__ import annotations

import pytest

from cfie_training.training_base import (
    TrainableParamSpec,
    TrainingWindowBudget,
    TrainingWindowPlanner,
)


def test_plan_window_selects_by_priority_with_stable_tie_break() -> None:
    planner = TrainingWindowPlanner(
        TrainingWindowBudget(
            window_steps=50,
            max_fp32_hot_bytes=100,
            max_gpu_shadow_bytes=100,
            max_adam_bytes=100,
            max_gptq_requant_bytes=100,
        )
    )

    plan = planner.plan_window(
        [
            TrainableParamSpec("b", fp32_bytes=10, priority=1.0),
            TrainableParamSpec("c", fp32_bytes=10, priority=2.0),
            TrainableParamSpec("a", fp32_bytes=10, priority=1.0),
        ],
        global_step=125,
    )

    assert plan.window_index == 2
    assert plan.start_step == 100
    assert plan.end_step_exclusive == 150
    assert plan.param_ids == ("c", "a", "b")
    assert plan.total_fp32_hot_bytes == 30


def test_plan_window_respects_each_budget_dimension() -> None:
    planner = TrainingWindowPlanner(
        TrainingWindowBudget(
            window_steps=50,
            max_fp32_hot_bytes=30,
            max_gpu_shadow_bytes=30,
            max_adam_bytes=30,
            max_gptq_requant_bytes=30,
        )
    )

    plan = planner.plan_window(
        [
            TrainableParamSpec(
                "fits",
                fp32_bytes=10,
                gpu_shadow_bytes=10,
                adam_bytes=10,
                gptq_requant_bytes=10,
                priority=10.0,
            ),
            TrainableParamSpec(
                "too_much_gpu",
                fp32_bytes=1,
                gpu_shadow_bytes=25,
                adam_bytes=1,
                gptq_requant_bytes=1,
                priority=9.0,
            ),
            TrainableParamSpec(
                "still_fits",
                fp32_bytes=10,
                gpu_shadow_bytes=10,
                adam_bytes=10,
                gptq_requant_bytes=10,
                priority=8.0,
            ),
        ],
        global_step=0,
    )

    assert plan.param_ids == ("fits", "still_fits")
    assert plan.skipped_param_ids == ("too_much_gpu",)
    assert plan.total_gpu_shadow_bytes == 20


def test_zero_budget_means_unlimited_for_that_dimension() -> None:
    planner = TrainingWindowPlanner(
        TrainingWindowBudget(
            window_steps=10,
            max_fp32_hot_bytes=0,
            max_gpu_shadow_bytes=0,
        )
    )

    plan = planner.plan_window(
        [
            TrainableParamSpec("a", fp32_bytes=10_000, gpu_shadow_bytes=10_000),
            TrainableParamSpec("b", fp32_bytes=20_000, gpu_shadow_bytes=20_000),
        ],
        global_step=19,
    )

    assert plan.window_index == 1
    assert plan.param_ids == ("a", "b")
    assert plan.total_fp32_hot_bytes == 30_000


def test_plan_window_raises_when_no_param_fits_minimum() -> None:
    planner = TrainingWindowPlanner(
        TrainingWindowBudget(
            max_fp32_hot_bytes=1,
            min_params_per_window=1,
        )
    )

    with pytest.raises(ValueError, match="not enough trainable params"):
        planner.plan_window(
            [TrainableParamSpec("too_large", fp32_bytes=2)],
            global_step=0,
        )


def test_touched_summary_preserves_plan_order_and_digest() -> None:
    planner = TrainingWindowPlanner(TrainingWindowBudget())
    plan = planner.plan_window(
        [
            TrainableParamSpec("a", priority=3),
            TrainableParamSpec("b", priority=2),
            TrainableParamSpec("c", priority=1),
        ],
        global_step=0,
    )

    summary = plan.touched_summary(["c", "a", "not_selected"])

    assert summary.window_index == 0
    assert summary.touched_param_ids == ("a", "c")
    assert summary.touched_digest


def test_budget_rejects_invalid_window_steps() -> None:
    with pytest.raises(ValueError, match="window_steps"):
        TrainingWindowBudget(window_steps=0)


def test_param_spec_rejects_empty_id() -> None:
    with pytest.raises(ValueError, match="param_id"):
        TrainableParamSpec("")
