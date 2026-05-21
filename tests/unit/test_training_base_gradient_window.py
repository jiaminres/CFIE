"""Unit tests for in-memory gradient window training runtime."""

from __future__ import annotations

import struct

import pytest
import torch

from cfie_training.training_base import (
    AdamStateShardRecord,
    AdamWConfig,
    CpuAdamFp8StateStore,
    CpuAdamFp8Updater,
    FP32ShardStore,
    ForwardShadowStore,
    GptqCacheRecord,
    GptqCacheRequantizer,
    GptqCacheStore,
    GradientBucketRing,
    HotParamTrainingWindow,
    ParamShardRecord,
    ProgressStateWriter,
    SymmetricInt4GptqCodec,
    SymmetricInt4GptqLayout,
    TrainableParamSpec,
    TrainingWindowBudget,
    TrainingWindowCommitter,
    TrainingWindowPlanner,
    TrainingWindowRuntime,
    WindowCommitPayload,
    adam_state_num_bytes,
    gptq_bundle_num_bytes,
    state_key,
)


def _fp32_bytes(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def _fp32_values(payload: bytes) -> tuple[float, ...]:
    return struct.unpack(f"<{len(payload) // 4}f", payload)


def test_gradient_bucket_ring_seals_by_capacity_and_accumulates() -> None:
    ring = GradientBucketRing(bucket_capacity_bytes=16)

    assert ring.add_gradient("param_a", torch.tensor([1.0, 2.0])) == ()
    assert ring.active_bytes == 8
    assert ring.add_gradient("param_a", torch.tensor([3.0, 4.0])) == ()
    assert ring.active_bytes == 16

    sealed = ring.add_gradient("param_b", torch.tensor([5.0]))

    assert len(sealed) == 1
    assert sealed[0].bucket_id == 0
    assert sealed[0].num_bytes == 16
    assert torch.equal(sealed[0].grads["param_a"], torch.tensor([4.0, 6.0]))
    assert ring.active_bytes == 4


def test_forward_shadow_store_refreshes_cast_view() -> None:
    shadow_store = ForwardShadowStore(dtype=torch.float16)

    shadow = shadow_store.refresh("param_a", torch.tensor([1.25, -2.5]))

    assert shadow.dtype == torch.float16
    assert shadow_store.get("param_a").dtype == torch.float16
    assert torch.allclose(
        shadow_store.get("param_a").to(torch.float32),
        torch.tensor([1.25, -2.5]),
    )


def test_hot_param_window_updates_master_and_shadow_before_commit(tmp_path) -> None:
    fp32_store = FP32ShardStore.create(
        tmp_path / "fp32",
        {
            "param_a": ParamShardRecord("param_a", "fp32_0000.bin", 0, 2),
            "param_b": ParamShardRecord("param_b", "fp32_0000.bin", 2, 2),
        },
    )
    fp32_store.flush_touched(
        {
            "param_a": _fp32_bytes([1.0, -2.0]),
            "param_b": _fp32_bytes([3.0, -4.0]),
        },
        generation=1,
    )

    state_bytes = adam_state_num_bytes(2)
    adam_store = CpuAdamFp8StateStore.create(
        tmp_path / "adam",
        {
            state_key("param_a", "m"): AdamStateShardRecord(
                "param_a",
                "m",
                "adam_0000.bin",
                0,
                state_bytes,
            ),
            state_key("param_a", "v"): AdamStateShardRecord(
                "param_a",
                "v",
                "adam_0000.bin",
                state_bytes,
                state_bytes,
            ),
            state_key("param_b", "m"): AdamStateShardRecord(
                "param_b",
                "m",
                "adam_0000.bin",
                state_bytes * 2,
                state_bytes,
            ),
            state_key("param_b", "v"): AdamStateShardRecord(
                "param_b",
                "v",
                "adam_0000.bin",
                state_bytes * 3,
                state_bytes,
            ),
        },
    )
    updater = CpuAdamFp8Updater(AdamWConfig(lr=0.1))
    gptq_codec = SymmetricInt4GptqCodec(
        SymmetricInt4GptqLayout(group_size=2)
    )
    gptq_store = GptqCacheStore.create(
        tmp_path / "gptq",
        {
            "bundle_param_a": GptqCacheRecord(
                "bundle_param_a",
                "gptq_0000.bin",
                0,
                gptq_bundle_num_bytes(2, group_size=2),
                quant_layout_hash=gptq_codec.layout_hash,
            ),
        },
    )
    gptq_requantizer = GptqCacheRequantizer(
        store=gptq_store,
        param_to_bundle={"param_a": "bundle_param_a"},
        codec=gptq_codec,
    )
    hot_window = HotParamTrainingWindow.load_from_stores(
        fp32_store=fp32_store,
        adam_store=adam_store,
        updater=updater,
        hot_param_ids=("param_a", "param_b"),
        bucket_capacity_bytes=12,
        shadow_dtype=torch.float16,
    )

    sealed = hot_window.add_gradient("param_a", torch.tensor([0.25, -0.5]))
    assert sealed == ()
    sealed = hot_window.add_gradient("param_b", torch.tensor([1.0, -1.0]))
    summary = hot_window.apply_buckets(sealed, optimizer_step=1)
    final_summary = hot_window.drain_all(optimizer_step=1)

    assert summary.touched_param_ids == ("param_a",)
    assert final_summary.touched_param_ids == ("param_b",)
    assert hot_window.touched_param_ids == ["param_a", "param_b"]
    assert torch.allclose(
        hot_window.masters["param_a"],
        torch.tensor([0.9, -1.9]),
        atol=1e-6,
    )
    assert torch.allclose(
        hot_window.masters["param_b"],
        torch.tensor([2.9, -3.9]),
        atol=1e-6,
    )
    assert torch.allclose(
        hot_window.shadow_store.get("param_a").to(torch.float32),
        torch.tensor([0.89990234375, -1.900390625]),
        atol=1e-6,
    )

    progress_writer = ProgressStateWriter.in_dir(tmp_path / "state")
    progress_writer.write_after_flush(
        global_step=0,
        epoch=0,
        dataset_cursor="dataset:0",
        round_id=0,
        fp32_master_generation=1,
        optimizer_generation=0,
        gptq_cache_generation=0,
    )
    runtime = TrainingWindowRuntime(
        planner=TrainingWindowPlanner(TrainingWindowBudget(window_steps=50)),
        committer=TrainingWindowCommitter(
            fp32_store,
            progress_writer,
            adam_store=adam_store,
            gptq_store=gptq_store,
        ),
        candidates=(
            TrainableParamSpec("param_a", priority=2.0),
            TrainableParamSpec("param_b", priority=1.0),
        ),
    )
    plan = runtime.begin_window()
    payload = hot_window.make_commit_payload(
        global_step=2,
        epoch=0,
        dataset_cursor="dataset:2",
        consumed_samples=4,
        consumed_tokens=128,
        gptq_update_builder=gptq_requantizer,
    )
    state = runtime.commit_window(plan, payload)

    assert _fp32_values(fp32_store.read_param("param_a")) == pytest.approx(
        (0.9, -1.9),
        abs=1e-6,
    )
    assert _fp32_values(fp32_store.read_param("param_b")) == pytest.approx(
        (2.9, -3.9),
        abs=1e-6,
    )
    assert len(adam_store.read_state("param_a", "m")) == state_bytes
    assert len(adam_store.read_state("param_b", "v")) == state_bytes
    assert len(gptq_store.read_bundle("bundle_param_a")) == (
        gptq_bundle_num_bytes(2, group_size=2)
    )
    assert state.global_step == 2
    assert state.consumed_tokens == 128
    assert state.gptq_cache_generation == 2


def test_hot_param_window_rejects_cold_param_gradient(tmp_path) -> None:
    fp32_store = FP32ShardStore.create(
        tmp_path / "fp32",
        {
            "param_a": ParamShardRecord("param_a", "fp32_0000.bin", 0, 1),
        },
    )
    fp32_store.flush_touched(
        {"param_a": _fp32_bytes([1.0])},
        generation=1,
    )
    state_bytes = adam_state_num_bytes(1)
    adam_store = CpuAdamFp8StateStore.create(
        tmp_path / "adam",
        {
            state_key("param_a", "m"): AdamStateShardRecord(
                "param_a",
                "m",
                "adam_0000.bin",
                0,
                state_bytes,
            ),
            state_key("param_a", "v"): AdamStateShardRecord(
                "param_a",
                "v",
                "adam_0000.bin",
                state_bytes,
                state_bytes,
            ),
        },
    )
    hot_window = HotParamTrainingWindow.load_from_stores(
        fp32_store=fp32_store,
        adam_store=adam_store,
        updater=CpuAdamFp8Updater(AdamWConfig(lr=0.1)),
        hot_param_ids=("param_a",),
        bucket_capacity_bytes=16,
    )

    with pytest.raises(KeyError, match="hot set"):
        hot_window.add_gradient("param_b", torch.ones(1))


def test_gradient_bucket_ring_gradient_accumulation_same_param() -> None:
    ring = GradientBucketRing(bucket_capacity_bytes=1024)
    grad_a = torch.tensor([1.0, 2.0], dtype=torch.float32)
    grad_b = torch.tensor([0.5, 0.5], dtype=torch.float32)

    ring.add_gradient("a", grad_a)
    ring.add_gradient("a", grad_b)

    buckets = ring.drain_all()
    assert len(buckets) == 1
    result = buckets[0].grads["a"]
    assert torch.allclose(result, torch.tensor([1.5, 2.5], dtype=torch.float32))


def test_forward_shadow_store_get_unknown_param() -> None:
    store = ForwardShadowStore()
    with pytest.raises(KeyError, match="unknown"):
        store.get("nonexistent")


def test_hot_param_window_switch_rejects_pending_gradients(tmp_path) -> None:
    fp32_store = FP32ShardStore.create(
        tmp_path / "fp32",
        {
            "param_a": ParamShardRecord("param_a", "fp32_0000.bin", 0, 1),
            "param_b": ParamShardRecord("param_b", "fp32_0000.bin", 1, 1),
        },
    )
    fp32_store.flush_touched(
        {"param_a": _fp32_bytes([1.0]), "param_b": _fp32_bytes([2.0])},
        generation=1,
    )
    state_bytes = adam_state_num_bytes(1)
    adam_store = CpuAdamFp8StateStore.create(
        tmp_path / "adam",
        {
            state_key("param_a", "m"): AdamStateShardRecord(
                "param_a", "m", "adam_0000.bin", 0, state_bytes,
            ),
            state_key("param_a", "v"): AdamStateShardRecord(
                "param_a", "v", "adam_0000.bin", state_bytes, state_bytes,
            ),
            state_key("param_b", "m"): AdamStateShardRecord(
                "param_b", "m", "adam_0000.bin", state_bytes * 2, state_bytes,
            ),
            state_key("param_b", "v"): AdamStateShardRecord(
                "param_b", "v", "adam_0000.bin", state_bytes * 3, state_bytes,
            ),
        },
    )
    hot_window = HotParamTrainingWindow.load_from_stores(
        fp32_store=fp32_store,
        adam_store=adam_store,
        updater=CpuAdamFp8Updater(AdamWConfig(lr=0.1)),
        hot_param_ids=("param_a",),
        bucket_capacity_bytes=16,
    )
    hot_window.add_gradient("param_a", torch.ones(1))
    with pytest.raises(RuntimeError, match="pending"):
        hot_window.switch_hot_params(("param_b",))


def test_router_prefetch_plan_empty_no_matching_bundles() -> None:
    from cfie_training.training_base.router_prefetch import (
        ExpertBundleIds,
        RouterGptqPrefetchPlanner,
        RoutedExpert,
    )
    planner = RouterGptqPrefetchPlanner(
        layer_expert_to_bundles={
            (0, 0): ExpertBundleIds(
                w13_bundle_id="bundle_w13",
                w2_bundle_id="bundle_w2",
            ),
        },
        prefetch_depth=16,
    )
    plan = planner.plan(
        current_experts=(RoutedExpert(0, 1, score=1.0),),
    )
    assert plan.prefetch_bundle_ids == ()
    assert plan.skipped_experts == ((0, 1),)


def test_runtime_rejects_commit_before_window_start(tmp_path) -> None:
    fp32_store = FP32ShardStore.create(
        tmp_path / "fp32",
        {"param_a": ParamShardRecord("param_a", "shard_0000.bin", 0, 1)},
        generation=100,
    )
    progress_writer = ProgressStateWriter.in_dir(tmp_path / "state")
    progress_writer.write_after_flush(
        global_step=100,
        epoch=2,
        dataset_cursor="dataset:100",
        round_id=2,
        fp32_master_generation=100,
        optimizer_generation=100,
        gptq_cache_generation=100,
    )
    runtime = TrainingWindowRuntime(
        planner=TrainingWindowPlanner(TrainingWindowBudget(window_steps=50)),
        committer=TrainingWindowCommitter(fp32_store, progress_writer),
        candidates=(TrainableParamSpec("param_a"),),
    )
    plan = runtime.begin_window()
    with pytest.raises(ValueError, match="advance beyond"):
        runtime.commit_window(
            plan,
            WindowCommitPayload(
                fp32_updates={"param_a": _fp32_bytes([1.0])},
                global_step=50,
                epoch=0,
                dataset_cursor="dataset:50",
            ),
        )


def test_runtime_rejects_commit_after_window_end(tmp_path) -> None:
    fp32_store = FP32ShardStore.create(
        tmp_path / "fp32",
        {"param_a": ParamShardRecord("param_a", "shard_0000.bin", 0, 1)},
    )
    runtime = TrainingWindowRuntime(
        planner=TrainingWindowPlanner(TrainingWindowBudget(window_steps=50)),
        committer=TrainingWindowCommitter(
            fp32_store,
            ProgressStateWriter.in_dir(tmp_path / "state"),
        ),
        candidates=(TrainableParamSpec("param_a"),),
    )
    plan = runtime.begin_window()
    with pytest.raises(ValueError, match="exceed"):
        runtime.commit_window(
            plan,
            WindowCommitPayload(
                fp32_updates={"param_a": _fp32_bytes([1.0])},
                global_step=100,
                epoch=0,
                dataset_cursor="dataset:100",
            ),
        )


def test_training_window_budget_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="window_steps"):
        TrainingWindowBudget(window_steps=0)
    with pytest.raises(ValueError, match="fp32"):
        TrainingWindowBudget(max_fp32_hot_bytes=-1)
    with pytest.raises(ValueError, match="adam"):
        TrainingWindowBudget(max_adam_bytes=-1)


def test_trainable_param_spec_rejects_invalid_kind() -> None:
    with pytest.raises(ValueError, match="kind"):
        TrainableParamSpec(param_id="x", kind="invalid")
