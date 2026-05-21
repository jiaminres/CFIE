"""Unit tests for training-window runtime orchestration."""

from __future__ import annotations

import struct

import pytest

from cfie_training.training_base import (
    AdamStateShardRecord,
    CompositeTrainingWindowHooks,
    CpuAdamFp8StateStore,
    FP32ShardStore,
    GptqCacheRecord,
    GptqCacheStore,
    LoggingTrainingWindowHooks,
    ParamShardRecord,
    ProgressStateWriter,
    TrainableParamSpec,
    TrainingProgressState,
    TrainingWindowBudget,
    TrainingWindowCommitter,
    TrainingWindowPlanner,
    TrainingWindowRuntime,
    ValidatingTrainingWindowHooks,
    WindowCommitPayload,
    digest_hot_set,
    state_key,
)
from cfie_training.training_base.window_plan import HotSetPlan


def _fp32_bytes(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def _fp32_values(payload: bytes) -> tuple[float, ...]:
    return struct.unpack(f"<{len(payload) // 4}f", payload)


class RecordingHooks:
    def __init__(self) -> None:
        self.events: list[tuple[str, int, int]] = []

    def prepare_window(
        self,
        plan: HotSetPlan,
        progress: TrainingProgressState,
    ) -> None:
        self.events.append(("prepare", plan.window_index, progress.global_step))

    def drain_before_commit(
        self,
        plan: HotSetPlan,
        payload: WindowCommitPayload,
    ) -> None:
        self.events.append(("drain", plan.window_index, payload.global_step))

    def after_commit(
        self,
        plan: HotSetPlan,
        state: TrainingProgressState,
    ) -> None:
        self.events.append(("after", plan.window_index, state.global_step))


def test_runtime_begins_and_commits_window_in_order(tmp_path) -> None:
    fp32_store = FP32ShardStore.create(
        tmp_path / "fp32",
        {
            "param_a": ParamShardRecord("param_a", "shard_0000.bin", 0, 1),
        },
    )
    adam_store = CpuAdamFp8StateStore.create(
        tmp_path / "adam",
        {
            state_key("param_a", "m"): AdamStateShardRecord(
                "param_a",
                "m",
                "adam_0000.bin",
                0,
                2,
            ),
        },
    )
    gptq_store = GptqCacheStore.create(
        tmp_path / "gptq",
        {
            "bundle_a": GptqCacheRecord("bundle_a", "gptq_0000.bin", 0, 3),
        },
    )
    progress_writer = ProgressStateWriter.in_dir(tmp_path / "state")
    hooks = RecordingHooks()
    runtime = TrainingWindowRuntime(
        planner=TrainingWindowPlanner(TrainingWindowBudget(window_steps=50)),
        committer=TrainingWindowCommitter(
            fp32_store,
            progress_writer,
            adam_store=adam_store,
            gptq_store=gptq_store,
        ),
        candidates=(
            TrainableParamSpec("param_a", fp32_bytes=4, priority=1.0),
        ),
        hooks=hooks,
    )

    plan = runtime.begin_window()
    state = runtime.commit_window(
        plan,
        WindowCommitPayload(
            fp32_updates={"param_a": _fp32_bytes([3.0])},
            adam_updates={"param_a": {"m": b"aa"}},
            gptq_updates={"bundle_a": b"xyz"},
            global_step=50,
            epoch=1,
            dataset_cursor="dataset:50",
            touched_param_ids=["param_a"],
            consumed_samples=8,
            consumed_tokens=1024,
        ),
    )

    assert plan.param_ids == ("param_a",)
    assert _fp32_values(fp32_store.read_param("param_a")) == (3.0,)
    assert adam_store.read_state("param_a", "m") == b"aa"
    assert gptq_store.read_bundle("bundle_a") == b"xyz"
    assert state.hot_set_digest == digest_hot_set(plan.param_ids)
    assert state.consumed_samples == 8
    assert state.consumed_tokens == 1024
    assert hooks.events == [
        ("prepare", 0, 0),
        ("drain", 0, 50),
        ("after", 0, 50),
    ]


def test_runtime_plans_next_window_from_committed_progress(tmp_path) -> None:
    fp32_store = FP32ShardStore.create(
        tmp_path / "fp32",
        {
            "param_a": ParamShardRecord("param_a", "shard_0000.bin", 0, 1),
        },
    )
    runtime = TrainingWindowRuntime(
        planner=TrainingWindowPlanner(TrainingWindowBudget(window_steps=50)),
        committer=TrainingWindowCommitter(
            fp32_store,
            ProgressStateWriter.in_dir(tmp_path / "state"),
        ),
        candidates=(TrainableParamSpec("param_a"),),
    )

    first_plan = runtime.begin_window()
    runtime.commit_window(
        first_plan,
        WindowCommitPayload(
            fp32_updates={"param_a": _fp32_bytes([1.0])},
            global_step=50,
            epoch=0,
            dataset_cursor="dataset:50",
        ),
    )

    second_plan = runtime.begin_window()

    assert second_plan.window_index == 1
    assert second_plan.start_step == 50
    assert second_plan.end_step_exclusive == 100


def test_runtime_rejects_updates_outside_hot_set_before_drain(tmp_path) -> None:
    fp32_store = FP32ShardStore.create(
        tmp_path / "fp32",
        {
            "param_a": ParamShardRecord("param_a", "shard_0000.bin", 0, 1),
            "param_b": ParamShardRecord("param_b", "shard_0000.bin", 1, 1),
        },
    )
    progress_writer = ProgressStateWriter.in_dir(tmp_path / "state")
    hooks = RecordingHooks()
    runtime = TrainingWindowRuntime(
        planner=TrainingWindowPlanner(TrainingWindowBudget(window_steps=50)),
        committer=TrainingWindowCommitter(fp32_store, progress_writer),
        candidates=(TrainableParamSpec("param_a"),),
        hooks=hooks,
    )
    plan = runtime.begin_window()

    with pytest.raises(ValueError, match="outside the hot set"):
        runtime.commit_window(
            plan,
            WindowCommitPayload(
                fp32_updates={"param_b": _fp32_bytes([2.0])},
                global_step=50,
                epoch=0,
                dataset_cursor="dataset:50",
            ),
        )

    assert hooks.events == [("prepare", 0, 0)]
    assert not progress_writer.path.exists()


def test_runtime_rejects_empty_candidate_pool(tmp_path) -> None:
    fp32_store = FP32ShardStore.create(tmp_path / "fp32", {})
    runtime = TrainingWindowRuntime(
        planner=TrainingWindowPlanner(TrainingWindowBudget()),
        committer=TrainingWindowCommitter(
            fp32_store,
            ProgressStateWriter.in_dir(tmp_path / "state"),
        ),
    )

    with pytest.raises(ValueError, match="candidates"):
        runtime.begin_window()


def test_logging_hooks_emits_events(tmp_path) -> None:
    import logging

    fp32_store = FP32ShardStore.create(
        tmp_path / "fp32",
        {"param_a": ParamShardRecord("param_a", "shard_0000.bin", 0, 1)},
    )
    progress_writer = ProgressStateWriter.in_dir(tmp_path / "state")

    log_stream = logging.getLogger("cfie.training_base.window")
    log_stream.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    log_stream.addHandler(handler)

    hooks = LoggingTrainingWindowHooks(
        logger_name="cfie.training_base.window",
        log_level=logging.DEBUG,
    )
    runtime = TrainingWindowRuntime(
        planner=TrainingWindowPlanner(TrainingWindowBudget(window_steps=50)),
        committer=TrainingWindowCommitter(fp32_store, progress_writer),
        candidates=(TrainableParamSpec("param_a"),),
        hooks=hooks,
    )

    plan = runtime.begin_window()
    state = runtime.commit_window(
        plan,
        WindowCommitPayload(
            fp32_updates={"param_a": _fp32_bytes([2.0])},
            global_step=50,
            epoch=0,
            dataset_cursor="dataset:50",
        ),
    )
    assert state.global_step == 50


def test_validating_hooks_rejects_empty_plan(tmp_path) -> None:
    fp32_store = FP32ShardStore.create(tmp_path / "fp32", {})
    hooks = ValidatingTrainingWindowHooks(strict=True)
    runtime = TrainingWindowRuntime(
        planner=TrainingWindowPlanner(TrainingWindowBudget()),
        committer=TrainingWindowCommitter(
            fp32_store,
            ProgressStateWriter.in_dir(tmp_path / "state"),
        ),
        hooks=hooks,
    )
    with pytest.raises(ValueError, match="candidates"):
        runtime.begin_window()


def test_validating_hooks_rejects_fp32_missing_param(tmp_path) -> None:
    fp32_store = FP32ShardStore.create(
        tmp_path / "fp32",
        {
            "param_a": ParamShardRecord("param_a", "shard_0000.bin", 0, 1),
            "param_b": ParamShardRecord("param_b", "shard_0000.bin", 1, 1),
        },
    )
    progress_writer = ProgressStateWriter.in_dir(tmp_path / "state")
    hooks = ValidatingTrainingWindowHooks(strict=True)
    runtime = TrainingWindowRuntime(
        planner=TrainingWindowPlanner(TrainingWindowBudget()),
        committer=TrainingWindowCommitter(fp32_store, progress_writer),
        candidates=(
            TrainableParamSpec("param_a"),
            TrainableParamSpec("param_b"),
        ),
        hooks=hooks,
    )
    plan = runtime.begin_window()
    with pytest.raises(ValueError, match="missing"):
        runtime.commit_window(
            plan,
            WindowCommitPayload(
                fp32_updates={"param_a": _fp32_bytes([2.0])},
                global_step=50,
                epoch=0,
                dataset_cursor="dataset:50",
            ),
        )


def test_validating_hooks_non_strict_accepts_partial(tmp_path) -> None:
    fp32_store = FP32ShardStore.create(
        tmp_path / "fp32",
        {
            "param_a": ParamShardRecord("param_a", "shard_0000.bin", 0, 1),
            "param_b": ParamShardRecord("param_b", "shard_0000.bin", 1, 1),
        },
    )
    progress_writer = ProgressStateWriter.in_dir(tmp_path / "state")
    hooks = ValidatingTrainingWindowHooks(strict=False)
    runtime = TrainingWindowRuntime(
        planner=TrainingWindowPlanner(TrainingWindowBudget()),
        committer=TrainingWindowCommitter(fp32_store, progress_writer),
        candidates=(
            TrainableParamSpec("param_a"),
            TrainableParamSpec("param_b"),
        ),
        hooks=hooks,
    )
    plan = runtime.begin_window()
    state = runtime.commit_window(
        plan,
        WindowCommitPayload(
            fp32_updates={"param_a": _fp32_bytes([3.0])},
            global_step=50,
            epoch=0,
            dataset_cursor="dataset:50",
        ),
    )
    assert state.global_step == 50


def test_composite_hooks_executes_all_in_order(tmp_path) -> None:
    fp32_store = FP32ShardStore.create(
        tmp_path / "fp32",
        {"param_a": ParamShardRecord("param_a", "shard_0000.bin", 0, 1)},
    )
    progress_writer = ProgressStateWriter.in_dir(tmp_path / "state")

    recording_a = RecordingHooks()
    recording_b = RecordingHooks()
    composite = CompositeTrainingWindowHooks(
        hooks=[recording_a, recording_b],
    )
    runtime = TrainingWindowRuntime(
        planner=TrainingWindowPlanner(TrainingWindowBudget(window_steps=50)),
        committer=TrainingWindowCommitter(fp32_store, progress_writer),
        candidates=(TrainableParamSpec("param_a"),),
        hooks=composite,
    )

    plan = runtime.begin_window()
    runtime.commit_window(
        plan,
        WindowCommitPayload(
            fp32_updates={"param_a": _fp32_bytes([5.0])},
            global_step=50,
            epoch=0,
            dataset_cursor="dataset:50",
        ),
    )

    assert recording_a.events == recording_b.events
    assert recording_a.events == [
        ("prepare", 0, 0),
        ("drain", 0, 50),
        ("after", 0, 50),
    ]


def test_validating_hooks_rejects_step_before_start(tmp_path) -> None:
    fp32_store = FP32ShardStore.create(
        tmp_path / "fp32",
        {"param_a": ParamShardRecord("param_a", "shard_0000.bin", 0, 1)},
    )
    progress_writer = ProgressStateWriter.in_dir(tmp_path / "state")
    hooks = ValidatingTrainingWindowHooks()
    runtime = TrainingWindowRuntime(
        planner=TrainingWindowPlanner(TrainingWindowBudget(window_steps=50)),
        committer=TrainingWindowCommitter(fp32_store, progress_writer),
        candidates=(TrainableParamSpec("param_a"),),
        hooks=hooks,
    )
    plan = runtime.begin_window()
    with pytest.raises(ValueError, match="step"):
        runtime.commit_window(
            plan,
            WindowCommitPayload(
                fp32_updates={"param_a": _fp32_bytes([2.0])},
                global_step=10,
                epoch=0,
                dataset_cursor="dataset:10",
            ),
        )
