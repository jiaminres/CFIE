"""Unit tests for training resource peak monitoring."""

from __future__ import annotations

import pytest

from cfie_training.training_base import (
    PeakTelemetryRecorder,
    PeakMonitor,
    TrainingResourceSnapshot,
    TrainingResourceThresholds,
    TrainingResourceUsage,
)


class FixedResourceSampler:
    def sample(self) -> TrainingResourceUsage:
        return TrainingResourceUsage(
            gpu_allocated_bytes=10,
            gpu_reserved_bytes=20,
            gpu_nvml_used_bytes=30,
            cpu_rss_bytes=40,
            pinned_bytes=5,
        )


def test_peak_monitor_tracks_resource_peaks() -> None:
    monitor = PeakMonitor()

    assert monitor.record(
        TrainingResourceSnapshot(
            step=1,
            gpu_allocated_bytes=10,
            gpu_reserved_bytes=12,
            gpu_nvml_used_bytes=14,
            cpu_rss_bytes=20,
            pinned_bytes=4,
        )
    ) == ()
    assert monitor.record(
        TrainingResourceSnapshot(
            step=2,
            gpu_allocated_bytes=8,
            gpu_reserved_bytes=16,
            gpu_nvml_used_bytes=13,
            cpu_rss_bytes=25,
            pinned_bytes=3,
        )
    ) == ()

    peaks = monitor.peaks
    assert peaks.snapshots_seen == 2
    assert peaks.last_step == 2
    assert peaks.max_gpu_allocated_bytes == 10
    assert peaks.max_gpu_reserved_bytes == 16
    assert peaks.max_gpu_nvml_used_bytes == 14
    assert peaks.max_cpu_rss_bytes == 25
    assert peaks.max_pinned_bytes == 4
    assert peaks.max_expert_cache_capacity_pressure_rate == 0.0


def test_grad_bucket_wait_event_requires_consecutive_steps() -> None:
    monitor = PeakMonitor(
        TrainingResourceThresholds(
            grad_bucket_wait_ratio=0.08,
            grad_bucket_wait_steps=3,
        )
    )

    assert monitor.record(
        TrainingResourceSnapshot(step=1, grad_bucket_wait_ratio=0.09)
    ) == ()
    assert monitor.record(
        TrainingResourceSnapshot(step=2, grad_bucket_wait_ratio=0.09)
    ) == ()
    events = monitor.record(
        TrainingResourceSnapshot(step=3, grad_bucket_wait_ratio=0.09)
    )

    assert len(events) == 1
    assert events[0].name == "grad_bucket_wait"
    assert events[0].consecutive_steps == 3
    assert "grad bucket" in events[0].action


def test_expert_cache_miss_streak_resets_below_threshold() -> None:
    monitor = PeakMonitor(
        TrainingResourceThresholds(
            expert_cache_miss_rate=0.05,
            expert_cache_miss_steps=2,
        )
    )

    assert monitor.record(
        TrainingResourceSnapshot(step=1, expert_cache_miss_rate=0.06)
    ) == ()
    assert monitor.record(
        TrainingResourceSnapshot(step=2, expert_cache_miss_rate=0.01)
    ) == ()
    assert monitor.record(
        TrainingResourceSnapshot(step=3, expert_cache_miss_rate=0.06)
    ) == ()
    events = monitor.record(
        TrainingResourceSnapshot(step=4, expert_cache_miss_rate=0.06)
    )

    assert len(events) == 1
    assert events[0].name == "expert_cache_miss"
    assert "GPTQ resident cache" in events[0].action


def test_expert_cache_capacity_pressure_event_requires_consecutive_steps() -> None:
    monitor = PeakMonitor(
        TrainingResourceThresholds(
            expert_cache_capacity_pressure_rate=0.0,
            expert_cache_capacity_pressure_steps=2,
        )
    )

    assert monitor.record(
        TrainingResourceSnapshot(
            step=1,
            expert_cache_capacity_pressure_rate=0.5,
        )
    ) == ()
    events = monitor.record(
        TrainingResourceSnapshot(
            step=2,
            expert_cache_capacity_pressure_rate=0.5,
        )
    )

    assert len(events) == 1
    assert events[0].name == "expert_cache_capacity_pressure"
    assert events[0].consecutive_steps == 2
    assert "resident cache" in events[0].action
    assert monitor.peaks.max_expert_cache_capacity_pressure_rate == 0.5


def test_immediate_memory_and_flush_events() -> None:
    monitor = PeakMonitor(
        TrainingResourceThresholds(
            cpu_rss_budget_bytes=100,
            cpu_rss_budget_ratio=0.9,
            pinned_budget_bytes=10,
            expected_flush_seconds=5.0,
            flush_slowdown_factor=2.0,
        )
    )

    events = monitor.record(
        TrainingResourceSnapshot(
            step=7,
            cpu_rss_bytes=91,
            pinned_bytes=11,
            flush_seconds=11.0,
        )
    )

    assert [event.name for event in events] == [
        "cpu_rss",
        "pinned_memory",
        "flush_latency",
    ]


def test_window_counter_reset_does_not_clear_peaks() -> None:
    monitor = PeakMonitor(
        TrainingResourceThresholds(
            grad_bucket_wait_ratio=0.08,
            grad_bucket_wait_steps=2,
        )
    )

    monitor.record(
        TrainingResourceSnapshot(
            step=1,
            gpu_allocated_bytes=32,
            grad_bucket_wait_ratio=0.09,
        )
    )
    monitor.reset_window_counters()
    events = monitor.record(
        TrainingResourceSnapshot(step=2, grad_bucket_wait_ratio=0.09)
    )

    assert events == ()
    assert monitor.peaks.max_gpu_allocated_bytes == 32


def test_snapshot_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="cpu_rss_bytes"):
        TrainingResourceSnapshot(step=1, cpu_rss_bytes=-1)


def test_telemetry_recorder_merges_runtime_metrics_with_sample() -> None:
    recorder = PeakTelemetryRecorder(
        monitor=PeakMonitor(
            TrainingResourceThresholds(
                expert_cache_miss_rate=0.0,
                expert_cache_miss_steps=1,
            )
        ),
        sampler=FixedResourceSampler(),
    )

    events = recorder.record_step(
        step=3,
        expert_cache_miss_rate=1.0,
        pinned_bytes=7,
    )

    assert [event.name for event in events] == ["expert_cache_miss"]
    assert recorder.peaks.snapshots_seen == 1
    assert recorder.peaks.max_gpu_allocated_bytes == 10
    assert recorder.peaks.max_cpu_rss_bytes == 40
    assert recorder.peaks.max_pinned_bytes == 7


def test_telemetry_recorder_records_capacity_pressure_metric() -> None:
    recorder = PeakTelemetryRecorder(
        monitor=PeakMonitor(
            TrainingResourceThresholds(
                expert_cache_capacity_pressure_rate=0.0,
                expert_cache_capacity_pressure_steps=1,
            )
        ),
        sampler=FixedResourceSampler(),
    )

    events = recorder.record_step(
        step=5,
        expert_cache_capacity_pressure_rate=0.25,
    )

    assert [event.name for event in events] == [
        "expert_cache_capacity_pressure"
    ]
    assert recorder.peaks.max_expert_cache_capacity_pressure_rate == 0.25
