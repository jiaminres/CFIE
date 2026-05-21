"""Unit tests for router-driven GPTQ prefetch planning."""

from __future__ import annotations

import pytest

from cfie_training.training_base import (
    GptqCacheRecord,
    GptqCacheStore,
    ResidentGptqCache,
    RoutedExpert,
    RouterPrefetchDepthTuner,
    RouterPrefetchDepthTuningConfig,
    RouterGptqPrefetchPlanner,
)


class _SynchronizingBackend:
    def __init__(self) -> None:
        self.synchronized: list[str] = []

    def load(self, bundle_id: str, payload: bytes) -> bytes:
        _ = bundle_id
        return payload

    def release(self, bundle_id: str, payload: bytes) -> None:
        _ = bundle_id, payload

    def synchronize(self, bundle_id: str, payload: bytes) -> None:
        _ = payload
        self.synchronized.append(bundle_id)

    def num_bytes(self, payload: bytes) -> int:
        return len(payload)


def _make_store(tmp_path) -> GptqCacheStore:
    records = {}
    updates = {}
    offset = 0
    for bundle_id in (
        "layers.0.experts.0.w13_weight",
        "layers.0.experts.0.w2_weight",
        "layers.0.experts.1.w13_weight",
        "layers.0.experts.1.w2_weight",
        "layers.1.experts.0.w13_weight",
        "layers.1.experts.0.w2_weight",
    ):
        records[bundle_id] = GptqCacheRecord(
            bundle_id,
            "gptq_0000.bin",
            offset,
            4,
        )
        updates[bundle_id] = bundle_id.encode("utf-8")[:4].ljust(4, b"_")
        offset += 4
    store = GptqCacheStore.create(tmp_path, records)
    store.flush_touched(updates, generation=1)
    return store


def _param_to_bundle() -> dict[str, str]:
    return {
        bundle_id: bundle_id
        for bundle_id in (
            "layers.0.experts.0.w13_weight",
            "layers.0.experts.0.w2_weight",
            "layers.0.experts.1.w13_weight",
            "layers.0.experts.1.w2_weight",
            "layers.1.experts.0.w13_weight",
            "layers.1.experts.0.w2_weight",
        )
    }


def test_router_prefetch_plan_orders_by_priority_and_locks_current() -> None:
    planner = RouterGptqPrefetchPlanner.from_param_to_bundle(
        _param_to_bundle(),
        prefetch_depth=4,
    )

    plan = planner.plan(
        current_experts=[
            RoutedExpert(0, 0, score=0.2, token_count=1),
            RoutedExpert(0, 1, score=0.9, token_count=2),
        ],
        predicted_experts=[
            RoutedExpert(1, 0, score=0.8, token_count=1),
        ],
    )

    assert plan.prefetch_bundle_ids == (
        "layers.0.experts.1.w13_weight",
        "layers.0.experts.1.w2_weight",
        "layers.0.experts.0.w13_weight",
        "layers.0.experts.0.w2_weight",
    )
    assert plan.locked_bundle_ids == plan.prefetch_bundle_ids
    assert plan.skipped_experts == ()


def test_router_prefetch_plan_skips_hot_and_missing_experts() -> None:
    planner = RouterGptqPrefetchPlanner.from_param_to_bundle(
        _param_to_bundle(),
        prefetch_depth=8,
    )
    planner.set_hot_experts([(0, 1)])

    plan = planner.plan(
        current_experts=[
            RoutedExpert(0, 1, score=1.0),
            RoutedExpert(9, 9, score=0.9),
            RoutedExpert(0, 0, score=0.8),
        ],
    )

    assert plan.prefetch_bundle_ids == (
        "layers.0.experts.0.w13_weight",
        "layers.0.experts.0.w2_weight",
    )
    assert plan.locked_bundle_ids == plan.prefetch_bundle_ids
    assert plan.skipped_experts == ((0, 1), (9, 9))


def test_router_prefetch_execute_loads_and_locks_cache(tmp_path) -> None:
    cache = ResidentGptqCache(_make_store(tmp_path), capacity_bytes=16)
    planner = RouterGptqPrefetchPlanner.from_param_to_bundle(
        _param_to_bundle(),
        prefetch_depth=4,
    )

    result = planner.execute(
        cache,
        current_experts=[RoutedExpert(0, 0, score=1.0)],
        predicted_experts=[RoutedExpert(1, 0, score=0.5)],
    )

    assert result.loaded_bundle_ids == (
        "layers.0.experts.0.w13_weight",
        "layers.0.experts.0.w2_weight",
        "layers.1.experts.0.w13_weight",
        "layers.1.experts.0.w2_weight",
    )
    assert result.ready_bundle_ids == (
        "layers.0.experts.0.w13_weight",
        "layers.0.experts.0.w2_weight",
    )
    assert result.locked_bundle_ids == (
        "layers.0.experts.0.w13_weight",
        "layers.0.experts.0.w2_weight",
    )
    assert result.resident_bundle_ids == result.loaded_bundle_ids


def test_router_prefetch_execute_reports_existing_resident_as_not_loaded(
    tmp_path,
) -> None:
    cache = ResidentGptqCache(_make_store(tmp_path), capacity_bytes=16)
    planner = RouterGptqPrefetchPlanner.from_param_to_bundle(
        _param_to_bundle(),
        prefetch_depth=2,
    )
    planner.execute(cache, current_experts=[RoutedExpert(0, 0)])

    result = planner.execute(cache, current_experts=[RoutedExpert(0, 0)])

    assert result.loaded_bundle_ids == ()
    assert result.ready_bundle_ids == (
        "layers.0.experts.0.w13_weight",
        "layers.0.experts.0.w2_weight",
    )
    assert result.locked_bundle_ids == (
        "layers.0.experts.0.w13_weight",
        "layers.0.experts.0.w2_weight",
    )


def test_router_prefetch_depth_can_split_expert_bundle_pair() -> None:
    planner = RouterGptqPrefetchPlanner.from_param_to_bundle(
        _param_to_bundle(),
        prefetch_depth=1,
    )

    plan = planner.plan(current_experts=[RoutedExpert(0, 0)])

    assert plan.prefetch_bundle_ids == ("layers.0.experts.0.w13_weight",)
    assert plan.locked_bundle_ids == ("layers.0.experts.0.w13_weight",)


def test_router_prefetch_can_include_hot_experts_when_configured() -> None:
    planner = RouterGptqPrefetchPlanner.from_param_to_bundle(
        _param_to_bundle(),
        prefetch_depth=2,
        skip_hot_experts=False,
    )
    planner.set_hot_experts([(0, 0)])

    plan = planner.plan(current_experts=[RoutedExpert(0, 0)])

    assert plan.prefetch_bundle_ids == (
        "layers.0.experts.0.w13_weight",
        "layers.0.experts.0.w2_weight",
    )
    assert plan.skipped_experts == ()


def test_router_prefetch_execute_waits_current_not_predicted_experts(
    tmp_path,
) -> None:
    backend = _SynchronizingBackend()
    cache = ResidentGptqCache(
        _make_store(tmp_path),
        capacity_bytes=16,
        backend=backend,
    )
    planner = RouterGptqPrefetchPlanner.from_param_to_bundle(
        _param_to_bundle(),
        prefetch_depth=4,
    )

    result = planner.execute(
        cache,
        current_experts=[RoutedExpert(0, 0, score=1.0)],
        predicted_experts=[RoutedExpert(1, 0, score=0.5)],
    )

    assert result.ready_bundle_ids == (
        "layers.0.experts.0.w13_weight",
        "layers.0.experts.0.w2_weight",
    )
    assert backend.synchronized == list(result.ready_bundle_ids)


def test_router_prefetch_execute_records_capacity_pressure_when_partial_allowed(
    tmp_path,
) -> None:
    cache = ResidentGptqCache(_make_store(tmp_path), capacity_bytes=8)
    planner = RouterGptqPrefetchPlanner.from_param_to_bundle(
        _param_to_bundle(),
        prefetch_depth=4,
    )
    planner.execute(cache, current_experts=[RoutedExpert(0, 0)])

    result = planner.execute(
        cache,
        current_experts=[],
        predicted_experts=[RoutedExpert(0, 1)],
        allow_partial=True,
    )

    assert result.loaded_bundle_ids == ()
    assert result.failed_bundle_ids == (
        "layers.0.experts.1.w13_weight",
        "layers.0.experts.1.w2_weight",
    )
    assert result.has_capacity_pressure
    assert result.resident_bundle_ids == (
        "layers.0.experts.0.w13_weight",
        "layers.0.experts.0.w2_weight",
    )
    assert result.locked_bundle_ids == result.resident_bundle_ids


def test_router_prefetch_execute_raises_capacity_pressure_by_default(
    tmp_path,
) -> None:
    cache = ResidentGptqCache(_make_store(tmp_path), capacity_bytes=8)
    planner = RouterGptqPrefetchPlanner.from_param_to_bundle(
        _param_to_bundle(),
        prefetch_depth=4,
    )
    planner.execute(cache, current_experts=[RoutedExpert(0, 0)])

    with pytest.raises(RuntimeError, match="cannot evict"):
        planner.execute(
            cache,
            current_experts=[],
            predicted_experts=[RoutedExpert(0, 1)],
        )


def test_router_prefetch_depth_tuner_decreases_on_capacity_pressure() -> None:
    planner = RouterGptqPrefetchPlanner.from_param_to_bundle(
        _param_to_bundle(),
        prefetch_depth=6,
    )
    tuner = RouterPrefetchDepthTuner(
        RouterPrefetchDepthTuningConfig(
            min_prefetch_depth=2,
            max_prefetch_depth=8,
            decrease_step=3,
            capacity_pressure_steps=1,
        )
    )

    decision = tuner.update(
        planner,
        miss_rate=0.0,
        capacity_pressure_rate=0.5,
    )

    assert decision.changed
    assert decision.old_depth == 6
    assert decision.new_depth == 3
    assert decision.reason == "capacity_pressure"
    assert planner.prefetch_depth == 3


def test_router_prefetch_depth_tuner_increases_on_sustained_miss_rate() -> None:
    planner = RouterGptqPrefetchPlanner.from_param_to_bundle(
        _param_to_bundle(),
        prefetch_depth=2,
    )
    tuner = RouterPrefetchDepthTuner(
        RouterPrefetchDepthTuningConfig(
            min_prefetch_depth=1,
            max_prefetch_depth=6,
            increase_step=2,
            miss_rate_threshold=0.2,
            miss_rate_steps=2,
        )
    )

    first = tuner.update(
        planner,
        miss_rate=0.5,
        capacity_pressure_rate=0.0,
    )
    second = tuner.update(
        planner,
        miss_rate=0.5,
        capacity_pressure_rate=0.0,
    )

    assert not first.changed
    assert first.miss_rate_streak == 1
    assert second.changed
    assert second.old_depth == 2
    assert second.new_depth == 4
    assert second.reason == "cache_miss"
    assert planner.prefetch_depth == 4


def test_router_prefetch_depth_tuner_prefers_pressure_over_miss_rate() -> None:
    planner = RouterGptqPrefetchPlanner.from_param_to_bundle(
        _param_to_bundle(),
        prefetch_depth=4,
    )
    tuner = RouterPrefetchDepthTuner(
        RouterPrefetchDepthTuningConfig(
            min_prefetch_depth=1,
            max_prefetch_depth=8,
            increase_step=2,
            decrease_step=1,
            miss_rate_threshold=0.1,
            miss_rate_steps=1,
            capacity_pressure_steps=1,
        )
    )

    decision = tuner.update(
        planner,
        miss_rate=1.0,
        capacity_pressure_rate=1.0,
    )

    assert decision.reason == "capacity_pressure"
    assert decision.new_depth == 3
    assert planner.prefetch_depth == 3
