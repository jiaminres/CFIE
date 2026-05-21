"""Unit tests for GPTQ cache shard persistence."""

from __future__ import annotations

import pytest

from cfie_training.training_base import GptqCacheRecord, GptqCacheStore


def test_flush_touched_creates_and_reads_gptq_bundle(tmp_path) -> None:
    store = GptqCacheStore.create(
        tmp_path,
        {
            "layer1.expert3.w13": GptqCacheRecord(
                bundle_id="layer1.expert3.w13",
                shard_name="gptq_0000.bin",
                offset_bytes=0,
                num_bytes=4,
                quant_layout_hash="layout-a",
            ),
        },
    )

    written = store.flush_touched(
        {"layer1.expert3.w13": b"\x01\x02\x03\x04"},
        generation=1,
    )

    assert written == 1
    assert store.generation == 1
    assert store.read_bundle("layer1.expert3.w13") == b"\x01\x02\x03\x04"
    assert GptqCacheStore.load(tmp_path).generation == 1
    assert not list(tmp_path.glob(".gptq_0000.bin.*.tmp"))


def test_flush_touched_preserves_other_bundles_in_same_shard(tmp_path) -> None:
    store = GptqCacheStore.create(
        tmp_path,
        {
            "bundle_a": GptqCacheRecord("bundle_a", "gptq_0000.bin", 0, 2),
            "bundle_b": GptqCacheRecord("bundle_b", "gptq_0000.bin", 2, 2),
        },
    )
    store.flush_touched(
        {
            "bundle_a": b"aa",
            "bundle_b": b"bb",
        },
        generation=1,
    )

    store.flush_touched({"bundle_a": b"cc"}, generation=2)

    assert store.read_bundle("bundle_a") == b"cc"
    assert store.read_bundle("bundle_b") == b"bb"


def test_flush_touched_updates_multiple_gptq_shards(tmp_path) -> None:
    store = GptqCacheStore.create(
        tmp_path,
        {
            "bundle_a": GptqCacheRecord("bundle_a", "gptq_0000.bin", 0, 1),
            "bundle_b": GptqCacheRecord("bundle_b", "gptq_0001.bin", 0, 1),
        },
    )

    written = store.flush_touched(
        {
            "bundle_a": b"a",
            "bundle_b": b"b",
        },
        generation=4,
    )

    assert written == 2
    assert store.read_bundle("bundle_a") == b"a"
    assert store.read_bundle("bundle_b") == b"b"
    assert GptqCacheStore.load(tmp_path).generation == 4


def test_flush_touched_rejects_unknown_gptq_bundle(tmp_path) -> None:
    store = GptqCacheStore.create(tmp_path, {})

    with pytest.raises(KeyError, match="unknown GPTQ cache bundle"):
        store.flush_touched({"missing": b"a"}, generation=1)


def test_flush_touched_rejects_wrong_gptq_payload_size(tmp_path) -> None:
    store = GptqCacheStore.create(
        tmp_path,
        {
            "bundle_a": GptqCacheRecord("bundle_a", "gptq_0000.bin", 0, 4),
        },
    )

    with pytest.raises(ValueError, match="expected 4 bytes"):
        store.flush_touched({"bundle_a": b"aa"}, generation=1)


def test_gptq_record_rejects_path_like_shard_name() -> None:
    with pytest.raises(ValueError, match="plain file name"):
        GptqCacheRecord("bundle_a", "nested/gptq.bin", 0, 1)
