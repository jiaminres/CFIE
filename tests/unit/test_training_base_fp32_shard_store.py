"""Unit tests for FP32 master shard persistence."""

from __future__ import annotations

import struct

import pytest

from cfie_training.training_base import FP32ShardStore, ParamShardRecord


def _fp32_bytes(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def _fp32_values(payload: bytes) -> tuple[float, ...]:
    return struct.unpack(f"<{len(payload) // 4}f", payload)


def test_flush_touched_creates_and_reads_shard(tmp_path) -> None:
    store = FP32ShardStore.create(
        tmp_path,
        {
            "param_a": ParamShardRecord(
                param_id="param_a",
                shard_name="shard_0000.bin",
                offset_elements=0,
                num_elements=3,
            ),
        },
    )

    written = store.flush_touched(
        {"param_a": _fp32_bytes([1.0, 2.0, 3.0])},
        generation=1,
    )

    assert written == 1
    assert store.generation == 1
    assert _fp32_values(store.read_param("param_a")) == (1.0, 2.0, 3.0)
    assert FP32ShardStore.load(tmp_path).generation == 1
    assert not list(tmp_path.glob(".shard_0000.bin.*.tmp"))


def test_flush_touched_preserves_other_params_in_same_shard(tmp_path) -> None:
    store = FP32ShardStore.create(
        tmp_path,
        {
            "param_a": ParamShardRecord("param_a", "shard_0000.bin", 0, 2),
            "param_b": ParamShardRecord("param_b", "shard_0000.bin", 2, 2),
        },
    )
    store.flush_touched(
        {
            "param_a": _fp32_bytes([1.0, 2.0]),
            "param_b": _fp32_bytes([3.0, 4.0]),
        },
        generation=1,
    )

    store.flush_touched(
        {"param_a": _fp32_bytes([9.0, 10.0])},
        generation=2,
    )

    assert _fp32_values(store.read_param("param_a")) == (9.0, 10.0)
    assert _fp32_values(store.read_param("param_b")) == (3.0, 4.0)


def test_flush_touched_updates_multiple_shards(tmp_path) -> None:
    store = FP32ShardStore.create(
        tmp_path,
        {
            "param_a": ParamShardRecord("param_a", "shard_0000.bin", 0, 1),
            "param_b": ParamShardRecord("param_b", "shard_0001.bin", 0, 1),
        },
    )

    written = store.flush_touched(
        {
            "param_a": _fp32_bytes([5.0]),
            "param_b": _fp32_bytes([6.0]),
        },
        generation=7,
    )

    assert written == 2
    assert _fp32_values(store.read_param("param_a")) == (5.0,)
    assert _fp32_values(store.read_param("param_b")) == (6.0,)
    assert FP32ShardStore.load(tmp_path).generation == 7


def test_flush_touched_rejects_unknown_param(tmp_path) -> None:
    store = FP32ShardStore.create(tmp_path, {})

    with pytest.raises(KeyError, match="unknown FP32 param"):
        store.flush_touched({"missing": _fp32_bytes([1.0])}, generation=1)


def test_flush_touched_rejects_wrong_payload_size(tmp_path) -> None:
    store = FP32ShardStore.create(
        tmp_path,
        {
            "param_a": ParamShardRecord("param_a", "shard_0000.bin", 0, 2),
        },
    )

    with pytest.raises(ValueError, match="expected 8 bytes"):
        store.flush_touched({"param_a": _fp32_bytes([1.0])}, generation=1)


def test_flush_touched_rejects_generation_regression(tmp_path) -> None:
    store = FP32ShardStore.create(
        tmp_path,
        {
            "param_a": ParamShardRecord("param_a", "shard_0000.bin", 0, 1),
        },
        generation=3,
    )

    with pytest.raises(ValueError, match="generation must be"):
        store.flush_touched({"param_a": _fp32_bytes([1.0])}, generation=2)


def test_record_rejects_path_like_shard_name() -> None:
    with pytest.raises(ValueError, match="plain file name"):
        ParamShardRecord("param_a", "nested/shard.bin", 0, 1)
