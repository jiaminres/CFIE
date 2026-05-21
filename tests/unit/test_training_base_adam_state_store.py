"""Unit tests for CPU Adam FP8 state shard persistence."""

from __future__ import annotations

import pytest

from cfie_training.training_base import (
    AdamStateShardRecord,
    CpuAdamFp8StateStore,
    state_key,
)


def test_flush_touched_creates_and_reads_adam_components(tmp_path) -> None:
    store = CpuAdamFp8StateStore.create(
        tmp_path,
        {
            state_key("param_a", "m"): AdamStateShardRecord(
                param_id="param_a",
                component="m",
                shard_name="adam_0000.bin",
                offset_bytes=0,
                num_bytes=4,
            ),
            state_key("param_a", "v"): AdamStateShardRecord(
                param_id="param_a",
                component="v",
                shard_name="adam_0000.bin",
                offset_bytes=4,
                num_bytes=4,
            ),
        },
    )

    written = store.flush_touched(
        {
            "param_a": {
                "m": b"\x01\x02\x03\x04",
                "v": b"\x05\x06\x07\x08",
            },
        },
        generation=1,
    )

    assert written == 2
    assert store.generation == 1
    assert store.read_state("param_a", "m") == b"\x01\x02\x03\x04"
    assert store.read_state("param_a", "v") == b"\x05\x06\x07\x08"
    assert CpuAdamFp8StateStore.load(tmp_path).generation == 1
    assert not list(tmp_path.glob(".adam_0000.bin.*.tmp"))


def test_flush_touched_preserves_other_components(tmp_path) -> None:
    store = CpuAdamFp8StateStore.create(
        tmp_path,
        {
            state_key("param_a", "m"): AdamStateShardRecord(
                "param_a",
                "m",
                "adam_0000.bin",
                0,
                2,
            ),
            state_key("param_a", "v"): AdamStateShardRecord(
                "param_a",
                "v",
                "adam_0000.bin",
                2,
                2,
            ),
        },
    )
    store.flush_touched(
        {"param_a": {"m": b"aa", "v": b"bb"}},
        generation=1,
    )

    store.flush_touched({"param_a": {"m": b"cc"}}, generation=2)

    assert store.read_state("param_a", "m") == b"cc"
    assert store.read_state("param_a", "v") == b"bb"


def test_flush_touched_rejects_unknown_component(tmp_path) -> None:
    store = CpuAdamFp8StateStore.create(tmp_path, {})

    with pytest.raises(KeyError, match="unknown Adam state"):
        store.flush_touched({"param_a": {"m": b"aa"}}, generation=1)


def test_flush_touched_rejects_wrong_payload_size(tmp_path) -> None:
    store = CpuAdamFp8StateStore.create(
        tmp_path,
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

    with pytest.raises(ValueError, match="expected 4 bytes"):
        store.flush_touched({"param_a": {"m": b"aa"}}, generation=1)


def test_adam_state_record_rejects_path_like_shard_name() -> None:
    with pytest.raises(ValueError, match="plain file name"):
        AdamStateShardRecord("param_a", "m", "nested/adam.bin", 0, 1)
