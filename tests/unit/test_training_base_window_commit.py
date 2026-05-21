"""Unit tests for training-window commit ordering."""

from __future__ import annotations

import struct

import pytest

from cfie_training.training_base import (
    AdamStateShardRecord,
    CpuAdamFp8StateStore,
    FP32ShardStore,
    GptqCacheRecord,
    GptqCacheStore,
    ParamShardRecord,
    ProgressStateWriter,
    TrainingWindowCommitter,
    state_key,
)


def _fp32_bytes(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def _fp32_values(payload: bytes) -> tuple[float, ...]:
    return struct.unpack(f"<{len(payload) // 4}f", payload)


def test_commit_fp32_window_flushes_store_then_writes_progress(tmp_path) -> None:
    fp32_store = FP32ShardStore.create(
        tmp_path / "fp32",
        {
            "param_a": ParamShardRecord("param_a", "shard_0000.bin", 0, 2),
        },
    )
    progress_writer = ProgressStateWriter.in_dir(tmp_path / "state")
    committer = TrainingWindowCommitter(fp32_store, progress_writer)

    state = committer.commit_fp32_window(
        fp32_updates={"param_a": _fp32_bytes([1.0, 2.0])},
        global_step=50,
        epoch=1,
        dataset_cursor="dataset:50",
        round_id=3,
        hot_set=["param_a"],
        consumed_samples=16,
        consumed_tokens=4096,
    )

    assert _fp32_values(fp32_store.read_param("param_a")) == (1.0, 2.0)
    assert state.global_step == 50
    assert state.fp32_master_generation == 50
    assert state.optimizer_generation == 50
    assert state.gptq_cache_generation == 50
    assert progress_writer.load_latest_or_init() == state
    assert committer.load_committed_progress() == state


def test_commit_fp32_window_uses_explicit_external_generations(tmp_path) -> None:
    fp32_store = FP32ShardStore.create(
        tmp_path / "fp32",
        {
            "param_a": ParamShardRecord("param_a", "shard_0000.bin", 0, 1),
        },
    )
    committer = TrainingWindowCommitter(
        fp32_store,
        ProgressStateWriter.in_dir(tmp_path / "state"),
    )

    state = committer.commit_fp32_window(
        fp32_updates={"param_a": _fp32_bytes([5.0])},
        global_step=100,
        epoch=2,
        dataset_cursor="cursor",
        round_id=4,
        flush_generation=17,
        optimizer_generation=18,
        gptq_cache_generation=19,
    )

    assert state.fp32_master_generation == 17
    assert state.optimizer_generation == 18
    assert state.gptq_cache_generation == 19


def test_commit_fp32_window_flushes_adam_before_progress(tmp_path) -> None:
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
    progress_writer = ProgressStateWriter.in_dir(tmp_path / "state")
    committer = TrainingWindowCommitter(
        fp32_store,
        progress_writer,
        adam_store=adam_store,
    )

    state = committer.commit_fp32_window(
        fp32_updates={"param_a": _fp32_bytes([7.0])},
        adam_updates={"param_a": {"m": b"aa"}},
        global_step=25,
        epoch=1,
        dataset_cursor="cursor",
        round_id=2,
    )

    assert _fp32_values(fp32_store.read_param("param_a")) == (7.0,)
    assert adam_store.read_state("param_a", "m") == b"aa"
    assert state.fp32_master_generation == 25
    assert state.optimizer_generation == 25
    assert progress_writer.load_latest_or_init() == state


def test_commit_fp32_window_does_not_write_progress_when_adam_flush_fails(
    tmp_path,
) -> None:
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
                4,
            ),
        },
    )
    progress_writer = ProgressStateWriter.in_dir(tmp_path / "state")
    committer = TrainingWindowCommitter(
        fp32_store,
        progress_writer,
        adam_store=adam_store,
    )

    with pytest.raises(ValueError, match="expected 4 bytes"):
        committer.commit_fp32_window(
            fp32_updates={"param_a": _fp32_bytes([3.0])},
            adam_updates={"param_a": {"m": b"aa"}},
            global_step=25,
            epoch=1,
            dataset_cursor="cursor",
            round_id=2,
        )

    assert not progress_writer.path.exists()


def test_commit_fp32_window_flushes_gptq_before_progress(tmp_path) -> None:
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
    committer = TrainingWindowCommitter(
        fp32_store,
        progress_writer,
        adam_store=adam_store,
        gptq_store=gptq_store,
    )

    state = committer.commit_fp32_window(
        fp32_updates={"param_a": _fp32_bytes([8.0])},
        adam_updates={"param_a": {"m": b"aa"}},
        gptq_updates={"bundle_a": b"xyz"},
        global_step=30,
        epoch=1,
        dataset_cursor="cursor",
        round_id=2,
    )

    assert _fp32_values(fp32_store.read_param("param_a")) == (8.0,)
    assert adam_store.read_state("param_a", "m") == b"aa"
    assert gptq_store.read_bundle("bundle_a") == b"xyz"
    assert state.fp32_master_generation == 30
    assert state.optimizer_generation == 30
    assert state.gptq_cache_generation == 30
    assert progress_writer.load_latest_or_init() == state


def test_commit_fp32_window_does_not_write_progress_when_gptq_flush_fails(
    tmp_path,
) -> None:
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
    committer = TrainingWindowCommitter(
        fp32_store,
        progress_writer,
        adam_store=adam_store,
        gptq_store=gptq_store,
    )

    with pytest.raises(ValueError, match="expected 3 bytes"):
        committer.commit_fp32_window(
            fp32_updates={"param_a": _fp32_bytes([8.0])},
            adam_updates={"param_a": {"m": b"aa"}},
            gptq_updates={"bundle_a": b"xy"},
            global_step=30,
            epoch=1,
            dataset_cursor="cursor",
            round_id=2,
        )

    assert not progress_writer.path.exists()


def test_load_committed_progress_detects_generation_mismatch(tmp_path) -> None:
    fp32_store = FP32ShardStore.create(
        tmp_path / "fp32",
        {
            "param_a": ParamShardRecord("param_a", "shard_0000.bin", 0, 1),
        },
    )
    progress_writer = ProgressStateWriter.in_dir(tmp_path / "state")
    committer = TrainingWindowCommitter(fp32_store, progress_writer)
    committer.commit_window(
        fp32_updates={"param_a": _fp32_bytes([1.0])},
        global_step=10,
        epoch=1,
        dataset_cursor="cursor",
        round_id=1,
    )
    fp32_store.flush_touched(
        {"param_a": _fp32_bytes([2.0])},
        generation=11,
    )

    with pytest.raises(ValueError, match="fp32_master_generation"):
        committer.load_committed_progress()


def test_commit_fp32_window_does_not_write_progress_when_flush_fails(
    tmp_path,
) -> None:
    fp32_store = FP32ShardStore.create(
        tmp_path / "fp32",
        {
            "param_a": ParamShardRecord("param_a", "shard_0000.bin", 0, 2),
        },
    )
    progress_writer = ProgressStateWriter.in_dir(tmp_path / "state")
    committer = TrainingWindowCommitter(fp32_store, progress_writer)

    with pytest.raises(ValueError, match="expected 8 bytes"):
        committer.commit_fp32_window(
            fp32_updates={"param_a": _fp32_bytes([1.0])},
            global_step=50,
            epoch=1,
            dataset_cursor="dataset:50",
            round_id=3,
        )

    assert not progress_writer.path.exists()
