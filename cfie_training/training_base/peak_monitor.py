"""峰值监控与自动降级——采集 GPU/CPU/NVMe 峰值，触发降级链（设计文档 Section 12）。"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol


def _require_non_negative_int(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


def _require_positive_int(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be >= 1")


def _require_non_negative_float(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


def _require_positive_float(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0")


@dataclass(frozen=True, slots=True)
class TrainingResourceSnapshot:
    step: int
    gpu_allocated_bytes: int = 0
    gpu_reserved_bytes: int = 0
    gpu_nvml_used_bytes: int = 0
    cpu_rss_bytes: int = 0
    pinned_bytes: int = 0
    grad_bucket_wait_ratio: float = 0.0
    expert_cache_miss_rate: float = 0.0
    expert_cache_capacity_pressure_rate: float = 0.0
    flush_seconds: float = 0.0
    expected_flush_seconds: float = 0.0

    def __post_init__(self) -> None:
        _require_non_negative_int("step", self.step)
        _require_non_negative_int("gpu_allocated_bytes", self.gpu_allocated_bytes)
        _require_non_negative_int("gpu_reserved_bytes", self.gpu_reserved_bytes)
        _require_non_negative_int("gpu_nvml_used_bytes", self.gpu_nvml_used_bytes)
        _require_non_negative_int("cpu_rss_bytes", self.cpu_rss_bytes)
        _require_non_negative_int("pinned_bytes", self.pinned_bytes)
        _require_non_negative_float(
            "grad_bucket_wait_ratio",
            self.grad_bucket_wait_ratio,
        )
        _require_non_negative_float(
            "expert_cache_miss_rate",
            self.expert_cache_miss_rate,
        )
        _require_non_negative_float(
            "expert_cache_capacity_pressure_rate",
            self.expert_cache_capacity_pressure_rate,
        )
        _require_non_negative_float("flush_seconds", self.flush_seconds)
        _require_non_negative_float(
            "expected_flush_seconds",
            self.expected_flush_seconds,
        )


@dataclass(frozen=True, slots=True)
class TrainingResourceUsage:
    gpu_allocated_bytes: int = 0
    gpu_reserved_bytes: int = 0
    gpu_nvml_used_bytes: int = 0
    cpu_rss_bytes: int = 0
    pinned_bytes: int = 0

    def __post_init__(self) -> None:
        _require_non_negative_int("gpu_allocated_bytes", self.gpu_allocated_bytes)
        _require_non_negative_int("gpu_reserved_bytes", self.gpu_reserved_bytes)
        _require_non_negative_int("gpu_nvml_used_bytes", self.gpu_nvml_used_bytes)
        _require_non_negative_int("cpu_rss_bytes", self.cpu_rss_bytes)
        _require_non_negative_int("pinned_bytes", self.pinned_bytes)


class TrainingResourceSampler(Protocol):
    def sample(self) -> TrainingResourceUsage:
        ...


@dataclass(frozen=True, slots=True)
class TorchProcessResourceSampler:
    device: object | None = None

    def sample(self) -> TrainingResourceUsage:
        gpu_allocated_bytes = 0
        gpu_reserved_bytes = 0
        gpu_used_bytes = 0

        try:
            import torch

            if torch.cuda.is_available():
                device = torch.device(self.device or torch.cuda.current_device())
                if device.type == "cuda":
                    device_index = (
                        torch.cuda.current_device()
                        if device.index is None
                        else device.index
                    )
                    gpu_allocated_bytes = torch.cuda.memory_allocated(device_index)
                    gpu_reserved_bytes = torch.cuda.memory_reserved(device_index)
                    try:
                        free_bytes, total_bytes = torch.cuda.mem_get_info(
                            device_index
                        )
                        gpu_used_bytes = total_bytes - free_bytes
                    except Exception:
                        gpu_used_bytes = 0
        except Exception:
            pass

        return TrainingResourceUsage(
            gpu_allocated_bytes=gpu_allocated_bytes,
            gpu_reserved_bytes=gpu_reserved_bytes,
            gpu_nvml_used_bytes=gpu_used_bytes,
            cpu_rss_bytes=_process_rss_bytes(),
        )


@dataclass(frozen=True, slots=True)
class TrainingResourceThresholds:
    grad_bucket_wait_ratio: float = 0.08
    grad_bucket_wait_steps: int = 20
    expert_cache_miss_rate: float = 0.05
    expert_cache_miss_steps: int = 50
    expert_cache_capacity_pressure_rate: float = 0.0
    expert_cache_capacity_pressure_steps: int = 1
    cpu_rss_budget_bytes: int = 0
    cpu_rss_budget_ratio: float = 0.9
    pinned_budget_bytes: int = 0
    expected_flush_seconds: float = 0.0
    flush_slowdown_factor: float = 2.0

    def __post_init__(self) -> None:
        _require_non_negative_float(
            "grad_bucket_wait_ratio",
            self.grad_bucket_wait_ratio,
        )
        _require_positive_int("grad_bucket_wait_steps", self.grad_bucket_wait_steps)
        _require_non_negative_float(
            "expert_cache_miss_rate",
            self.expert_cache_miss_rate,
        )
        _require_positive_int("expert_cache_miss_steps", self.expert_cache_miss_steps)
        _require_non_negative_float(
            "expert_cache_capacity_pressure_rate",
            self.expert_cache_capacity_pressure_rate,
        )
        _require_positive_int(
            "expert_cache_capacity_pressure_steps",
            self.expert_cache_capacity_pressure_steps,
        )
        _require_non_negative_int("cpu_rss_budget_bytes", self.cpu_rss_budget_bytes)
        _require_non_negative_float("cpu_rss_budget_ratio", self.cpu_rss_budget_ratio)
        _require_non_negative_int("pinned_budget_bytes", self.pinned_budget_bytes)
        _require_non_negative_float(
            "expected_flush_seconds",
            self.expected_flush_seconds,
        )
        _require_positive_float("flush_slowdown_factor", self.flush_slowdown_factor)


@dataclass(frozen=True, slots=True)
class TrainingResourcePeaks:
    snapshots_seen: int = 0
    last_step: int = 0
    max_gpu_allocated_bytes: int = 0
    max_gpu_reserved_bytes: int = 0
    max_gpu_nvml_used_bytes: int = 0
    max_cpu_rss_bytes: int = 0
    max_pinned_bytes: int = 0
    max_expert_cache_capacity_pressure_rate: float = 0.0


@dataclass(frozen=True, slots=True)
class ThresholdEvent:
    name: str
    step: int
    value: float
    limit: float
    consecutive_steps: int
    action: str


@dataclass(slots=True)
class PeakTelemetryRecorder:
    monitor: PeakMonitor = field(default_factory=lambda: PeakMonitor())
    sampler: TrainingResourceSampler = field(
        default_factory=TorchProcessResourceSampler
    )

    @property
    def peaks(self) -> TrainingResourcePeaks:
        return self.monitor.peaks

    def record_step(
        self,
        *,
        step: int,
        grad_bucket_wait_ratio: float = 0.0,
        expert_cache_miss_rate: float = 0.0,
        expert_cache_capacity_pressure_rate: float = 0.0,
        flush_seconds: float = 0.0,
        expected_flush_seconds: float = 0.0,
        pinned_bytes: int | None = None,
    ) -> tuple[ThresholdEvent, ...]:
        usage = self.sampler.sample()
        snapshot = TrainingResourceSnapshot(
            step=step,
            gpu_allocated_bytes=usage.gpu_allocated_bytes,
            gpu_reserved_bytes=usage.gpu_reserved_bytes,
            gpu_nvml_used_bytes=usage.gpu_nvml_used_bytes,
            cpu_rss_bytes=usage.cpu_rss_bytes,
            pinned_bytes=(
                usage.pinned_bytes if pinned_bytes is None else pinned_bytes
            ),
            grad_bucket_wait_ratio=grad_bucket_wait_ratio,
            expert_cache_miss_rate=expert_cache_miss_rate,
            expert_cache_capacity_pressure_rate=(
                expert_cache_capacity_pressure_rate
            ),
            flush_seconds=flush_seconds,
            expected_flush_seconds=expected_flush_seconds,
        )
        return self.monitor.record(snapshot)


@dataclass(slots=True)
# ────── PeakMonitor — 峰值检测与自动降级触发器（设计文档 Section 12）──────
class PeakMonitor:
    thresholds: TrainingResourceThresholds = field(
        default_factory=TrainingResourceThresholds
    )
    _peaks: TrainingResourcePeaks = field(default_factory=TrainingResourcePeaks)
    _grad_bucket_wait_streak: int = 0
    _expert_cache_miss_streak: int = 0
    _expert_cache_capacity_pressure_streak: int = 0
    _last_events: tuple[ThresholdEvent, ...] = ()

    @property
    def peaks(self) -> TrainingResourcePeaks:
        return self._peaks

    @property
    def last_events(self) -> tuple[ThresholdEvent, ...]:
        return self._last_events

    def requires_replan(self) -> bool:
        """是否有阈值事件触发 → 需要降级？"""
        return bool(self._last_events)                            # 任何事件都需处理

    def suggested_actions(self) -> tuple[str, ...]:
        """收集所有事件的降级动作（去重）。"""
        return tuple(dict.fromkeys(event.action for event in self._last_events))

    def record(
        self,
        snapshot: TrainingResourceSnapshot,
    ) -> tuple[ThresholdEvent, ...]:
        self._update_peaks(snapshot)
        events: list[ThresholdEvent] = []

        events.extend(self._check_grad_bucket_wait(snapshot))
        events.extend(self._check_expert_cache_miss(snapshot))
        events.extend(self._check_expert_cache_capacity_pressure(snapshot))
        events.extend(self._check_cpu_rss(snapshot))
        events.extend(self._check_pinned_memory(snapshot))
        events.extend(self._check_flush_latency(snapshot))

        self._last_events = tuple(events)
        return self._last_events

    def reset_window_counters(self) -> None:
        self._grad_bucket_wait_streak = 0
        self._expert_cache_miss_streak = 0
        self._expert_cache_capacity_pressure_streak = 0

    def reset_peaks(self) -> None:
        self._peaks = TrainingResourcePeaks()

    def _update_peaks(self, snapshot: TrainingResourceSnapshot) -> None:
        self._peaks = replace(
            self._peaks,
            snapshots_seen=self._peaks.snapshots_seen + 1,
            last_step=snapshot.step,
            max_gpu_allocated_bytes=max(
                self._peaks.max_gpu_allocated_bytes,
                snapshot.gpu_allocated_bytes,
            ),
            max_gpu_reserved_bytes=max(
                self._peaks.max_gpu_reserved_bytes,
                snapshot.gpu_reserved_bytes,
            ),
            max_gpu_nvml_used_bytes=max(
                self._peaks.max_gpu_nvml_used_bytes,
                snapshot.gpu_nvml_used_bytes,
            ),
            max_cpu_rss_bytes=max(
                self._peaks.max_cpu_rss_bytes,
                snapshot.cpu_rss_bytes,
            ),
            max_pinned_bytes=max(
                self._peaks.max_pinned_bytes,
                snapshot.pinned_bytes,
            ),
            max_expert_cache_capacity_pressure_rate=max(
                self._peaks.max_expert_cache_capacity_pressure_rate,
                snapshot.expert_cache_capacity_pressure_rate,
            ),
        )

    def _check_grad_bucket_wait(
        self,
        snapshot: TrainingResourceSnapshot,
    ) -> tuple[ThresholdEvent, ...]:
        limit = self.thresholds.grad_bucket_wait_ratio
        if snapshot.grad_bucket_wait_ratio > limit:
            self._grad_bucket_wait_streak += 1
        else:
            self._grad_bucket_wait_streak = 0
        if self._grad_bucket_wait_streak < self.thresholds.grad_bucket_wait_steps:
            return ()
        return (
            ThresholdEvent(
                name="grad_bucket_wait",
                step=snapshot.step,
                value=snapshot.grad_bucket_wait_ratio,
                limit=limit,
                consecutive_steps=self._grad_bucket_wait_streak,
                action="increase grad bucket size or CPU optimizer workers",
            ),
        )

    def _check_expert_cache_miss(
        self,
        snapshot: TrainingResourceSnapshot,
    ) -> tuple[ThresholdEvent, ...]:
        limit = self.thresholds.expert_cache_miss_rate
        if snapshot.expert_cache_miss_rate > limit:
            self._expert_cache_miss_streak += 1
        else:
            self._expert_cache_miss_streak = 0
        if self._expert_cache_miss_streak < self.thresholds.expert_cache_miss_steps:
            return ()
        return (
            ThresholdEvent(
                name="expert_cache_miss",
                step=snapshot.step,
                value=snapshot.expert_cache_miss_rate,
                limit=limit,
                consecutive_steps=self._expert_cache_miss_streak,
                action="increase GPTQ resident cache or prefetch horizon",
            ),
        )

    def _check_expert_cache_capacity_pressure(
        self,
        snapshot: TrainingResourceSnapshot,
    ) -> tuple[ThresholdEvent, ...]:
        limit = self.thresholds.expert_cache_capacity_pressure_rate
        if snapshot.expert_cache_capacity_pressure_rate > limit:
            self._expert_cache_capacity_pressure_streak += 1
        else:
            self._expert_cache_capacity_pressure_streak = 0
        if (
            self._expert_cache_capacity_pressure_streak
            < self.thresholds.expert_cache_capacity_pressure_steps
        ):
            return ()
        return (
            ThresholdEvent(
                name="expert_cache_capacity_pressure",
                step=snapshot.step,
                value=snapshot.expert_cache_capacity_pressure_rate,
                limit=limit,
                consecutive_steps=self._expert_cache_capacity_pressure_streak,
                action="increase GPTQ resident cache or reduce prefetch depth",
            ),
        )

    def _check_cpu_rss(
        self,
        snapshot: TrainingResourceSnapshot,
    ) -> tuple[ThresholdEvent, ...]:
        if self.thresholds.cpu_rss_budget_bytes == 0:
            return ()
        limit = (
            self.thresholds.cpu_rss_budget_bytes
            * self.thresholds.cpu_rss_budget_ratio
        )
        if snapshot.cpu_rss_bytes <= limit:
            return ()
        return (
            ThresholdEvent(
                name="cpu_rss",
                step=snapshot.step,
                value=float(snapshot.cpu_rss_bytes),
                limit=float(limit),
                consecutive_steps=1,
                action="mmap GPTQ cache or reduce hot set",
            ),
        )

    def _check_pinned_memory(
        self,
        snapshot: TrainingResourceSnapshot,
    ) -> tuple[ThresholdEvent, ...]:
        if self.thresholds.pinned_budget_bytes == 0:
            return ()
        limit = self.thresholds.pinned_budget_bytes
        if snapshot.pinned_bytes <= limit:
            return ()
        return (
            ThresholdEvent(
                name="pinned_memory",
                step=snapshot.step,
                value=float(snapshot.pinned_bytes),
                limit=float(limit),
                consecutive_steps=1,
                action="reduce staging buffers or transfer prefetch depth",
            ),
        )

    def _check_flush_latency(
        self,
        snapshot: TrainingResourceSnapshot,
    ) -> tuple[ThresholdEvent, ...]:
        expected = (
            snapshot.expected_flush_seconds
            if snapshot.expected_flush_seconds > 0
            else self.thresholds.expected_flush_seconds
        )
        if expected == 0:
            return ()
        limit = expected * self.thresholds.flush_slowdown_factor
        if snapshot.flush_seconds <= limit:
            return ()
        return (
            ThresholdEvent(
                name="flush_latency",
                step=snapshot.step,
                value=snapshot.flush_seconds,
                limit=limit,
                consecutive_steps=1,
                action="extend flush window or reduce hot set size",
            ),
        )


def _process_rss_bytes() -> int:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        return 0
