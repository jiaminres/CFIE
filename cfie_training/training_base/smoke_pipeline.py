"""End-to-end smoke pipeline for the training-base prototype."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from cfie_training.training_base.adam_update import AdamWConfig, CpuAdamFp8Updater
from cfie_training.training_base.checkpoint_import import (
    CheckpointStoreInitResult,
    Qwen35MoeCheckpointImportConfig,
    import_qwen35_moe_checkpoint_to_fp32_store,
)
from cfie_training.training_base.gptq_requant import (
    GptqCacheRequantizer,
    GptqMarlinBundleRequantizer,
    SymmetricInt4GptqCodec,
    SymmetricInt4GptqLayout,
)
from cfie_training.training_base.gradient_window import (
    HotParamTrainingWindow,
    HotParamWindowUpdateSummary,
)
from cfie_training.training_base.hot_set_scheduler import (
    CoverageConstraint,
    ExpertRoutingStats,
    HotSetScheduler,
    RoutingWindowStats,
)
from cfie_training.training_base.manifest_builder import ManifestShardConfig
from cfie_training.training_base.peak_monitor import (
    PeakMonitor,
    PeakTelemetryRecorder,
    ThresholdEvent,
    TorchProcessResourceSampler,
    TrainingResourcePeaks,
    TrainingResourceSampler,
    TrainingResourceThresholds,
)
from cfie_training.training_base.progress_state import (
    ProgressStateWriter,
    TrainingProgressState,
)
from cfie_training.training_base.resident_gptq_cache import (
    ResidentGptqCache,
    ResidentGptqPayloadBackend,
)
from cfie_training.training_base.router_prefetch import (
    RoutedExpert,
    RouterPrefetchDepthDecision,
    RouterPrefetchDepthTuner,
    RouterPrefetchDepthTuningConfig,
    RouterGptqPrefetchPlanner,
    RouterPrefetchResult,
)
from cfie_training.training_base.window_commit import TrainingWindowCommitter
from cfie_training.training_base.window_plan import (
    TrainableParamSpec,
    TrainingWindowBudget,
    TrainingWindowPlanner,
)
from cfie_training.training_base.window_runtime import TrainingWindowRuntime


def _require_non_empty_string(name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_positive_int(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be >= 1")


def _require_non_negative_int(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


def _require_non_negative_float(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


@dataclass(frozen=True, slots=True)
class TrainingSmokePipelineConfig:
    root: Path
    hot_param_ids: tuple[str, ...]
    manifest_config: ManifestShardConfig
    import_config: Qwen35MoeCheckpointImportConfig = (
        Qwen35MoeCheckpointImportConfig()
    )
    adam_config: AdamWConfig = AdamWConfig(lr=0.1)
    window_budget: TrainingWindowBudget = TrainingWindowBudget()
    bucket_capacity_bytes: int = 1 << 20
    resident_cache_capacity_bytes: int = 1 << 20
    resident_cache_backend: ResidentGptqPayloadBackend | None = None
    router_prefetch_depth: int = 16
    prefetch_depth_tuning_config: RouterPrefetchDepthTuningConfig | None = None
    dynamic_hotset_max_experts: int = 0
    dynamic_hotset_min_priority: float = 0.0
    shadow_dtype: torch.dtype = torch.float16
    shadow_device: torch.device | str = "cpu"
    enable_peak_monitor: bool = True
    resource_thresholds: TrainingResourceThresholds = field(
        default_factory=TrainingResourceThresholds
    )
    resource_sampler: TrainingResourceSampler | None = None
    coverage_constraint: CoverageConstraint | None = None
    expected_flush_seconds: float = 0.0

    def __post_init__(self) -> None:
        _require_positive_int("bucket_capacity_bytes", self.bucket_capacity_bytes)
        _require_positive_int(
            "resident_cache_capacity_bytes",
            self.resident_cache_capacity_bytes,
        )
        _require_positive_int("router_prefetch_depth", self.router_prefetch_depth)
        _require_non_negative_int(
            "dynamic_hotset_max_experts",
            self.dynamic_hotset_max_experts,
        )
        _require_non_negative_float(
            "dynamic_hotset_min_priority",
            self.dynamic_hotset_min_priority,
        )
        if not self.hot_param_ids:
            raise ValueError("hot_param_ids must not be empty")
        for param_id in self.hot_param_ids:
            _require_non_empty_string("hot_param_id", param_id)


@dataclass(frozen=True, slots=True)
class HotSetSwitchDecision:
    old_hot_param_ids: tuple[str, ...]
    new_hot_param_ids: tuple[str, ...]
    promoted_param_ids: tuple[str, ...]
    demoted_param_ids: tuple[str, ...]
    selected_experts: tuple[tuple[int, int], ...]
    reason: str

    @property
    def changed(self) -> bool:
        return self.old_hot_param_ids != self.new_hot_param_ids


@dataclass(frozen=True, slots=True)
class TrainingSmokeStepInput:
    gradients: Mapping[str, Any]
    global_step: int
    epoch: int
    dataset_cursor: str
    current_experts: tuple[RoutedExpert, ...] = ()
    predicted_experts: tuple[RoutedExpert, ...] = ()
    consumed_samples: int = 0
    consumed_tokens: int = 0
    grad_bucket_wait_ratio: float = 0.0
    pinned_bytes: int = 0


@dataclass(frozen=True, slots=True)
class TrainingSmokeStepResult:
    progress_state: TrainingProgressState
    update_summary: HotParamWindowUpdateSummary
    prefetch_result: RouterPrefetchResult
    next_prefetch_result: RouterPrefetchResult
    touched_param_ids: tuple[str, ...]
    resident_bundle_ids: tuple[str, ...]
    locked_bundle_ids: tuple[str, ...]
    threshold_events: tuple[ThresholdEvent, ...] = ()
    resource_peaks: TrainingResourcePeaks | None = None
    flush_seconds: float = 0.0
    expert_cache_miss_rate: float = 0.0
    expert_cache_capacity_pressure_rate: float = 0.0
    prefetch_depth_decision: RouterPrefetchDepthDecision | None = None
    router_prefetch_depth: int = 0
    hotset_switch_decision: HotSetSwitchDecision | None = None
    hot_param_ids: tuple[str, ...] = ()


@dataclass(slots=True)
class TrainingSmokePipeline:
    init_result: CheckpointStoreInitResult
    runtime: TrainingWindowRuntime
    hot_window: HotParamTrainingWindow
    gptq_requantizer: GptqCacheRequantizer | GptqMarlinBundleRequantizer
    resident_cache: ResidentGptqCache
    router_prefetch_planner: RouterGptqPrefetchPlanner
    prefetch_depth_tuner: RouterPrefetchDepthTuner | None = None
    dynamic_hotset_max_experts: int = 0
    dynamic_hotset_min_priority: float = 0.0
    hot_set_scheduler: HotSetScheduler | None = None
    telemetry: PeakTelemetryRecorder | None = None
    expected_flush_seconds: float = 0.0

    @classmethod
    def from_qwen35_moe_checkpoint(
        cls,
        weights: Iterable[tuple[str, Any]] | Mapping[str, Any],
        config: TrainingSmokePipelineConfig,
    ) -> "TrainingSmokePipeline":
        init_result = import_qwen35_moe_checkpoint_to_fp32_store(
            weights,
            root=config.root,
            import_config=config.import_config,
            manifest_config=config.manifest_config,
            generation=0,
        )
        if init_result.import_plan.gptq_cache_updates:
            gptq_requantizer: GptqCacheRequantizer | GptqMarlinBundleRequantizer = (
                GptqMarlinBundleRequantizer(
                    store=init_result.gptq_store,
                    param_to_bundle=init_result.manifest.param_to_gptq_bundle,
                )
            )
        else:
            codec = SymmetricInt4GptqCodec(
                SymmetricInt4GptqLayout(
                    group_size=config.manifest_config.gptq_group_size,
                )
            )
            gptq_requantizer = GptqCacheRequantizer(
                store=init_result.gptq_store,
                param_to_bundle=init_result.manifest.param_to_gptq_bundle,
                codec=codec,
            )
            initial_gptq_updates = gptq_requantizer.requantize_touched(
                init_result.import_plan.fp32_updates,
                init_result.manifest.param_to_gptq_bundle,
            )
            init_result.gptq_store.flush_touched(initial_gptq_updates, generation=0)

        committer = TrainingWindowCommitter(
            init_result.fp32_store,
            ProgressStateWriter.in_dir(config.root / "state"),
            adam_store=init_result.adam_store,
            gptq_store=init_result.gptq_store,
        )
        runtime = TrainingWindowRuntime(
            planner=TrainingWindowPlanner(config.window_budget),
            committer=committer,
            candidates=tuple(
                TrainableParamSpec(param_id=param_id)
                for param_id in config.hot_param_ids
            ),
        )
        hot_window = HotParamTrainingWindow.load_from_stores(
            fp32_store=init_result.fp32_store,
            adam_store=init_result.adam_store,
            updater=CpuAdamFp8Updater(config.adam_config),
            hot_param_ids=config.hot_param_ids,
            bucket_capacity_bytes=config.bucket_capacity_bytes,
            shadow_dtype=config.shadow_dtype,
            shadow_device=config.shadow_device,
        )
        if config.resident_cache_backend is None:
            resident_cache = ResidentGptqCache(
                init_result.gptq_store,
                capacity_bytes=config.resident_cache_capacity_bytes,
            )
        else:
            resident_cache = ResidentGptqCache(
                init_result.gptq_store,
                capacity_bytes=config.resident_cache_capacity_bytes,
                backend=config.resident_cache_backend,
            )
        router_prefetch_planner = RouterGptqPrefetchPlanner.from_param_to_bundle(
            init_result.manifest.param_to_gptq_bundle,
            prefetch_depth=config.router_prefetch_depth,
        )
        router_prefetch_planner.set_hot_experts(
            _expert_keys_from_param_ids(config.hot_param_ids)
        )
        prefetch_depth_tuner = (
            RouterPrefetchDepthTuner(config.prefetch_depth_tuning_config)
            if config.prefetch_depth_tuning_config is not None
            else None
        )
        scheduler = (
            HotSetScheduler(
                coverage=config.coverage_constraint or CoverageConstraint(),
            )
            if config.dynamic_hotset_max_experts > 0
            else None
        )
        telemetry = (
            PeakTelemetryRecorder(
                monitor=PeakMonitor(config.resource_thresholds),
                sampler=config.resource_sampler or TorchProcessResourceSampler(),
            )
            if config.enable_peak_monitor
            else None
        )
        return cls(
            init_result=init_result,
            runtime=runtime,
            hot_window=hot_window,
            gptq_requantizer=gptq_requantizer,
            resident_cache=resident_cache,
            router_prefetch_planner=router_prefetch_planner,
            prefetch_depth_tuner=prefetch_depth_tuner,
            dynamic_hotset_max_experts=config.dynamic_hotset_max_experts,
            dynamic_hotset_min_priority=config.dynamic_hotset_min_priority,
            hot_set_scheduler=scheduler,
            telemetry=telemetry,
            expected_flush_seconds=config.expected_flush_seconds,
        )

    def run_step(self, step_input: TrainingSmokeStepInput) -> TrainingSmokeStepResult:
        current_prefetch_result = self.router_prefetch_planner.execute(
            self.resident_cache,
            current_experts=step_input.current_experts,
            predicted_experts=(),
        )
        next_prefetch_result = self.router_prefetch_planner.execute(
            self.resident_cache,
            current_experts=(),
            predicted_experts=step_input.predicted_experts,
            allow_partial=True,
        )
        plan = self.runtime.begin_window(
            tuple(
                TrainableParamSpec(param_id=param_id)
                for param_id in self.hot_window.hot_param_ids
            )
        )
        summaries: list[HotParamWindowUpdateSummary] = []
        for param_id, grad in step_input.gradients.items():
            buckets = self.hot_window.add_gradient(param_id, grad)
            if buckets:
                summaries.append(
                    self.hot_window.apply_buckets(
                        buckets,
                        optimizer_step=step_input.global_step,
                    )
                )
        summaries.append(
            self.hot_window.drain_all(optimizer_step=step_input.global_step)
        )
        update_summary = _merge_update_summaries(summaries)
        payload = self.hot_window.make_commit_payload(
            global_step=step_input.global_step,
            epoch=step_input.epoch,
            dataset_cursor=step_input.dataset_cursor,
            consumed_samples=step_input.consumed_samples,
            consumed_tokens=step_input.consumed_tokens,
            gptq_update_builder=self.gptq_requantizer,
        )
        updated_bundle_ids = tuple((payload.gptq_updates or {}).keys())
        flush_start = time.perf_counter()
        progress_state = self.runtime.commit_window(plan, payload)
        flush_seconds = time.perf_counter() - flush_start
        touched_param_ids = payload.touched_ids()
        self.resident_cache.invalidate(updated_bundle_ids)
        self.resident_cache.unlock(current_prefetch_result.locked_bundle_ids)
        self.hot_window.mark_committed()
        expert_cache_miss_rate = _effective_prefetch_miss_rate(
            current_prefetch_result
        )
        expert_cache_capacity_pressure_rate = _prefetch_capacity_pressure_rate(
            next_prefetch_result
        )
        threshold_events: tuple[ThresholdEvent, ...] = ()
        resource_peaks: TrainingResourcePeaks | None = None
        if self.telemetry is not None:
            threshold_events = self.telemetry.record_step(
                step=step_input.global_step,
                grad_bucket_wait_ratio=step_input.grad_bucket_wait_ratio,
                expert_cache_miss_rate=expert_cache_miss_rate,
                expert_cache_capacity_pressure_rate=(
                    expert_cache_capacity_pressure_rate
                ),
                flush_seconds=flush_seconds,
                expected_flush_seconds=self.expected_flush_seconds,
                pinned_bytes=step_input.pinned_bytes,
            )
            resource_peaks = self.telemetry.peaks
        prefetch_depth_decision: RouterPrefetchDepthDecision | None = None
        if self.prefetch_depth_tuner is not None:
            prefetch_depth_decision = self.prefetch_depth_tuner.update(
                self.router_prefetch_planner,
                miss_rate=expert_cache_miss_rate,
                capacity_pressure_rate=expert_cache_capacity_pressure_rate,
            )
        hotset_switch_decision = self._switch_hotset_for_next_step(
            current_experts=step_input.current_experts,
            predicted_experts=step_input.predicted_experts,
        )
        return TrainingSmokeStepResult(
            progress_state=progress_state,
            update_summary=update_summary,
            prefetch_result=current_prefetch_result,
            next_prefetch_result=next_prefetch_result,
            touched_param_ids=touched_param_ids,
            resident_bundle_ids=self.resident_cache.resident_bundle_ids,
            locked_bundle_ids=self.resident_cache.locked_bundle_ids,
            threshold_events=threshold_events,
            resource_peaks=resource_peaks,
            flush_seconds=flush_seconds,
            expert_cache_miss_rate=expert_cache_miss_rate,
            expert_cache_capacity_pressure_rate=(
                expert_cache_capacity_pressure_rate
            ),
            prefetch_depth_decision=prefetch_depth_decision,
            router_prefetch_depth=self.router_prefetch_planner.prefetch_depth,
            hotset_switch_decision=hotset_switch_decision,
            hot_param_ids=self.hot_window.hot_param_ids,
        )

    def _switch_hotset_for_next_step(
        self,
        *,
        current_experts: Iterable[RoutedExpert],
        predicted_experts: Iterable[RoutedExpert],
    ) -> HotSetSwitchDecision | None:
        if self.dynamic_hotset_max_experts == 0:
            return None

        old_hot_param_ids = self.hot_window.hot_param_ids
        decision = self._plan_hotset_switch(
            old_hot_param_ids=old_hot_param_ids,
            experts=(*tuple(predicted_experts), *tuple(current_experts)),
        )
        if not decision.changed:
            return decision

        self.hot_window.switch_hot_params(decision.new_hot_param_ids)
        self.router_prefetch_planner.set_hot_experts(
            _expert_keys_from_param_ids(decision.new_hot_param_ids)
        )
        self.resident_cache.invalidate(
            self._bundle_ids_for_params(decision.promoted_param_ids)
        )
        return decision

    def _plan_hotset_switch(
        self,
        *,
        old_hot_param_ids: tuple[str, ...],
        experts: Iterable[RoutedExpert],
    ) -> HotSetSwitchDecision:
        if self.hot_set_scheduler is not None:
            return self._scheduled_hotset_switch(
                old_hot_param_ids=old_hot_param_ids,
                experts=experts,
            )

        selected_experts: list[tuple[int, int]] = []
        selected_param_ids: list[str] = []
        ordered_experts = RouterGptqPrefetchPlanner._ordered_experts(experts)
        for expert in ordered_experts:
            if expert.priority < self.dynamic_hotset_min_priority:
                continue
            key = (expert.layer_id, expert.expert_id)
            if key in selected_experts:
                continue
            expert_param_ids = _param_ids_for_expert(key)
            if not all(
                param_id in self.init_result.fp32_store.records
                for param_id in expert_param_ids
            ):
                continue
            selected_experts.append(key)
            selected_param_ids.extend(expert_param_ids)
            if len(selected_experts) >= self.dynamic_hotset_max_experts:
                break

        if not selected_param_ids:
            return HotSetSwitchDecision(
                old_hot_param_ids=old_hot_param_ids,
                new_hot_param_ids=old_hot_param_ids,
                promoted_param_ids=(),
                demoted_param_ids=(),
                selected_experts=(),
                reason="no_router_candidates",
            )

        new_hot_param_ids = tuple(selected_param_ids)
        old_set = set(old_hot_param_ids)
        new_set = set(new_hot_param_ids)
        return HotSetSwitchDecision(
            old_hot_param_ids=old_hot_param_ids,
            new_hot_param_ids=new_hot_param_ids,
            promoted_param_ids=tuple(
                param_id for param_id in new_hot_param_ids if param_id not in old_set
            ),
            demoted_param_ids=tuple(
                param_id for param_id in old_hot_param_ids if param_id not in new_set
            ),
            selected_experts=tuple(selected_experts),
            reason="router_priority",
        )

    def _scheduled_hotset_switch(
        self,
        *,
        old_hot_param_ids: tuple[str, ...],
        experts: Iterable[RoutedExpert],
    ) -> HotSetSwitchDecision:
        assert self.hot_set_scheduler is not None
        routed_experts = tuple(experts)
        route_stats_dict: dict[tuple[int, int], ExpertRoutingStats] = {}
        for expert in routed_experts:
            key = (expert.layer_id, expert.expert_id)
            existing = route_stats_dict.get(key)
            if existing is None:
                route_stats_dict[key] = ExpertRoutingStats(
                    layer_id=expert.layer_id,
                    expert_id=expert.expert_id,
                    activation_count=1,
                    total_score=expert.score,
                    token_count=expert.token_count,
                )
            else:
                route_stats_dict[key] = ExpertRoutingStats(
                    layer_id=existing.layer_id,
                    expert_id=existing.expert_id,
                    activation_count=existing.activation_count + 1,
                    total_score=existing.total_score + expert.score,
                    token_count=existing.token_count + expert.token_count,
                )
        route_stats = RoutingWindowStats(
            expert_stats=route_stats_dict,
            total_activations=len(routed_experts),
        )

        param_specs: list[TrainableParamSpec] = []
        for key, stats in route_stats_dict.items():
            expert_param_ids = _param_ids_for_expert(key)
            for param_id in expert_param_ids:
                record = self.init_result.fp32_store.records.get(param_id)
                if record is not None:
                    param_specs.append(
                        TrainableParamSpec(
                            param_id=param_id,
                            kind="moe",
                            fp32_bytes=record.num_bytes,
                            priority=stats.avg_score * 10.0,
                        )
                    )

        if not param_specs:
            return HotSetSwitchDecision(
                old_hot_param_ids=old_hot_param_ids,
                new_hot_param_ids=old_hot_param_ids,
                promoted_param_ids=(),
                demoted_param_ids=(),
                selected_experts=(),
                reason="no_router_candidates",
            )

        selection = self.hot_set_scheduler.select_hot_set(
            param_specs,
            budget=TrainingWindowBudget(
                max_fp32_hot_bytes=0,
                max_gpu_shadow_bytes=0,
                max_adam_bytes=0,
            ),
            route_stats=route_stats,
            expert_to_param_ids={},
        )

        new_hot_param_ids = selection.all_param_ids
        if not new_hot_param_ids:
            return HotSetSwitchDecision(
                old_hot_param_ids=old_hot_param_ids,
                new_hot_param_ids=old_hot_param_ids,
                promoted_param_ids=(),
                demoted_param_ids=(),
                selected_experts=(),
                reason="scheduler_no_selection",
            )

        selected_experts = list(selection.moe_experts)
        old_set = set(old_hot_param_ids)
        new_set = set(new_hot_param_ids)

        self.hot_set_scheduler.record_training_round(
            set(
                key
                for key in route_stats_dict
                if any(
                    pid in new_set for pid in _param_ids_for_expert(key)
                )
            )
        )

        return HotSetSwitchDecision(
            old_hot_param_ids=old_hot_param_ids,
            new_hot_param_ids=new_hot_param_ids,
            promoted_param_ids=tuple(
                param_id for param_id in new_hot_param_ids if param_id not in old_set
            ),
            demoted_param_ids=tuple(
                param_id for param_id in old_hot_param_ids if param_id not in new_set
            ),
            selected_experts=tuple(selected_experts),
            reason="hot_set_scheduler",
        )

    def _bundle_ids_for_params(
        self,
        param_ids: Iterable[str],
    ) -> tuple[str, ...]:
        bundle_ids: list[str] = []
        for param_id in param_ids:
            bundle_id = self.init_result.manifest.param_to_gptq_bundle.get(param_id)
            if bundle_id is not None and bundle_id not in bundle_ids:
                bundle_ids.append(bundle_id)
        return tuple(bundle_ids)


def _expert_keys_from_param_ids(
    param_ids: Iterable[str],
) -> tuple[tuple[int, int], ...]:
    keys: set[tuple[int, int]] = set()
    for param_id in param_ids:
        parts = param_id.split(".")
        if len(parts) != 5 or parts[0] != "layers" or parts[2] != "experts":
            continue
        try:
            keys.add((int(parts[1]), int(parts[3])))
        except ValueError:
            continue
    return tuple(sorted(keys))


def _param_ids_for_expert(key: tuple[int, int]) -> tuple[str, str]:
    layer_id, expert_id = key
    prefix = f"layers.{layer_id}.experts.{expert_id}"
    return (f"{prefix}.w13_weight", f"{prefix}.w2_weight")


def _merge_update_summaries(
    summaries: Iterable[HotParamWindowUpdateSummary],
) -> HotParamWindowUpdateSummary:
    touched_param_ids: list[str] = []
    grad_norms: dict[str, float] = {}
    update_norms: dict[str, float] = {}
    drained_bucket_ids: list[int] = []
    for summary in summaries:
        drained_bucket_ids.extend(summary.drained_bucket_ids)
        for param_id in summary.touched_param_ids:
            if param_id not in touched_param_ids:
                touched_param_ids.append(param_id)
            grad_norms[param_id] = summary.grad_norms[param_id]
            update_norms[param_id] = summary.update_norms[param_id]
    return HotParamWindowUpdateSummary(
        touched_param_ids=tuple(touched_param_ids),
        grad_norms=grad_norms,
        update_norms=update_norms,
        drained_bucket_ids=tuple(drained_bucket_ids),
    )


def _effective_prefetch_miss_rate(
    result: RouterPrefetchResult,
) -> float:
    requested = len(result.plan.prefetch_bundle_ids)
    if requested == 0:
        return result.miss_rate
    return len(result.loaded_bundle_ids) / requested


def _prefetch_capacity_pressure_rate(
    result: RouterPrefetchResult,
) -> float:
    requested = len(result.plan.prefetch_bundle_ids)
    if requested == 0:
        return 0.0
    return len(result.failed_prefetches) / requested
