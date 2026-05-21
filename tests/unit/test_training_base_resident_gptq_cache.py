"""Unit tests for resident GPTQ cache behavior."""

from __future__ import annotations

import pytest
import torch

from cfie_training.training_base import (
    GptqCacheRecord,
    GptqCacheStore,
    ResidentGptqCache,
    TorchTensorResidentGptqBackend,
)


class _TrackingBackend:
    def __init__(self) -> None:
        self.released: list[tuple[str, bytes]] = []

    def load(self, bundle_id: str, payload: bytes) -> bytearray:
        _ = bundle_id
        return bytearray(payload)

    def release(self, bundle_id: str, payload: bytearray) -> None:
        self.released.append((bundle_id, bytes(payload)))

    def num_bytes(self, payload: bytearray) -> int:
        return len(payload)


class _SynchronizingBackend:
    def __init__(self) -> None:
        self.synchronized: list[tuple[str, bytes]] = []

    def load(self, bundle_id: str, payload: bytes) -> bytearray:
        _ = bundle_id
        return bytearray(payload)

    def release(self, bundle_id: str, payload: bytearray) -> None:
        _ = bundle_id, payload

    def synchronize(self, bundle_id: str, payload: bytearray) -> None:
        self.synchronized.append((bundle_id, bytes(payload)))

    def num_bytes(self, payload: bytearray) -> int:
        return len(payload)


def _store_with_bundles(tmp_path) -> GptqCacheStore:
    store = GptqCacheStore.create(
        tmp_path,
        {
            "bundle_a": GptqCacheRecord("bundle_a", "gptq_0000.bin", 0, 4),
            "bundle_b": GptqCacheRecord("bundle_b", "gptq_0000.bin", 4, 4),
            "bundle_c": GptqCacheRecord("bundle_c", "gptq_0000.bin", 8, 4),
        },
    )
    store.flush_touched(
        {
            "bundle_a": b"aaaa",
            "bundle_b": b"bbbb",
            "bundle_c": b"cccc",
        },
        generation=1,
    )
    return store


def test_resident_gptq_cache_tracks_hits_misses_and_loads(tmp_path) -> None:
    cache = ResidentGptqCache(_store_with_bundles(tmp_path), capacity_bytes=8)

    assert cache.get("bundle_a") == b"aaaa"
    assert cache.get("bundle_a") == b"aaaa"

    assert cache.resident_bundle_ids == ("bundle_a",)
    assert cache.used_bytes == 4
    assert cache.stats.misses == 1
    assert cache.stats.hits == 1
    assert cache.stats.loads == 1
    assert cache.stats.miss_rate == pytest.approx(0.5)


def test_resident_gptq_cache_evicts_lru_unlocked_bundle(tmp_path) -> None:
    cache = ResidentGptqCache(_store_with_bundles(tmp_path), capacity_bytes=8)

    cache.get("bundle_a")
    cache.get("bundle_b")
    cache.get("bundle_a")
    cache.get("bundle_c")

    assert cache.resident_bundle_ids == ("bundle_a", "bundle_c")
    assert cache.stats.evictions == 1
    assert cache.stats.loads == 3


def test_resident_gptq_cache_preserves_locked_bundles(tmp_path) -> None:
    cache = ResidentGptqCache(_store_with_bundles(tmp_path), capacity_bytes=8)

    cache.lock(["bundle_a"])
    cache.get("bundle_b")
    cache.get("bundle_c")

    assert cache.locked_bundle_ids == ("bundle_a",)
    assert cache.resident_bundle_ids == ("bundle_a", "bundle_c")
    assert cache.stats.evictions == 1


def test_resident_gptq_cache_raises_when_all_resident_bundles_locked(
    tmp_path,
) -> None:
    cache = ResidentGptqCache(_store_with_bundles(tmp_path), capacity_bytes=8)
    cache.lock(["bundle_a", "bundle_b"])

    with pytest.raises(RuntimeError, match="cannot evict"):
        cache.get("bundle_c")


def test_resident_gptq_cache_rejects_bundle_larger_than_capacity(tmp_path) -> None:
    store = GptqCacheStore.create(
        tmp_path,
        {
            "bundle_a": GptqCacheRecord("bundle_a", "gptq_0000.bin", 0, 9),
        },
    )
    store.flush_touched({"bundle_a": b"123456789"}, generation=1)
    cache = ResidentGptqCache(store, capacity_bytes=8)

    with pytest.raises(ValueError, match="exceeds cache capacity"):
        cache.get("bundle_a")


def test_resident_gptq_cache_unlock_and_clear_unlocked(tmp_path) -> None:
    cache = ResidentGptqCache(_store_with_bundles(tmp_path), capacity_bytes=12)
    cache.lock(["bundle_a", "bundle_b"])
    cache.unlock(["bundle_b"])
    cache.prefetch("bundle_c")

    removed = cache.clear_unlocked()

    assert removed == ("bundle_b", "bundle_c")
    assert cache.resident_bundle_ids == ("bundle_a",)
    assert cache.used_bytes == 4


def test_resident_gptq_cache_prefetch_many_reports_new_loads(tmp_path) -> None:
    cache = ResidentGptqCache(_store_with_bundles(tmp_path), capacity_bytes=12)

    assert cache.prefetch_many(["bundle_a", "bundle_b"]) == (
        "bundle_a",
        "bundle_b",
    )
    assert cache.prefetch_many(["bundle_a", "bundle_c"]) == ("bundle_c",)
    cache.reset_stats()

    assert cache.stats.loads == 0
    assert cache.stats.requests == 0


def test_resident_gptq_cache_can_reside_payloads_as_torch_tensors(
    tmp_path,
) -> None:
    cache = ResidentGptqCache(
        _store_with_bundles(tmp_path),
        capacity_bytes=8,
        backend=TorchTensorResidentGptqBackend(device=torch.device("cpu")),
    )

    payload = cache.get("bundle_a")

    assert isinstance(payload, torch.Tensor)
    assert payload.dtype == torch.uint8
    assert payload.device.type == "cpu"
    assert bytes(payload.tolist()) == b"aaaa"
    assert cache.used_bytes == 4


def test_resident_gptq_cache_releases_backend_payloads(
    tmp_path,
) -> None:
    backend = _TrackingBackend()
    cache = ResidentGptqCache(
        _store_with_bundles(tmp_path),
        capacity_bytes=8,
        backend=backend,
    )

    cache.get("bundle_a")
    cache.get("bundle_b")
    cache.get("bundle_c")
    cache.invalidate(["bundle_b"])
    cache.clear_unlocked()

    assert backend.released == [
        ("bundle_a", b"aaaa"),
        ("bundle_b", b"bbbb"),
        ("bundle_c", b"cccc"),
    ]


def test_resident_gptq_cache_only_waits_when_payload_is_consumed(
    tmp_path,
) -> None:
    backend = _SynchronizingBackend()
    cache = ResidentGptqCache(
        _store_with_bundles(tmp_path),
        capacity_bytes=8,
        backend=backend,
    )

    cache.prefetch("bundle_a")
    assert backend.synchronized == []

    assert cache.wait_ready(["bundle_a", "bundle_missing"]) == ("bundle_a",)
    assert cache.get("bundle_a") == bytearray(b"aaaa")
    assert cache.get("bundle_b") == bytearray(b"bbbb")

    assert backend.synchronized == [
        ("bundle_a", b"aaaa"),
        ("bundle_a", b"aaaa"),
        ("bundle_b", b"bbbb"),
    ]
