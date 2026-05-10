"""训练主循环——设计文档 Section 16 伪代码的落地实现。

编排完整训练周期:
  checkpoint 导入 → hot 参数加载 → forward → backward → 梯度 bucket
  → CPU Adam 更新 → GPU shadow 刷新 → 窗口提交 → progress state

窗口 = hot set 切换边界（设计文档 Section 13.1）。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Protocol
import torch

from cfie_training.training_base.adam_update import AdamWConfig, CpuAdamFp8Updater
from cfie_training.training_base.gptq_requant import (
    GptqCacheRequantizer, GptqMarlinBundleRequantizer,
)
from cfie_training.training_base.gradient_window import (
    ForwardShadowStore, GradientBucket,
    HotParamTrainingWindow, HotParamWindowUpdateSummary,
)
from cfie_training.training_base.hot_set_scheduler import HotSetScheduler, HotSetSelection
from cfie_training.training_base.peak_monitor import (
    PeakMonitor, PeakTelemetryRecorder, ThresholdEvent,
    TorchProcessResourceSampler, TrainingResourcePeaks,
    TrainingResourceSampler, TrainingResourceThresholds,
)
from cfie_training.training_base.progress_state import ProgressStateWriter, TrainingProgressState
from cfie_training.training_base.resident_gptq_cache import ResidentGptqCache, ResidentGptqPayloadBackend
from cfie_training.training_base.router_prefetch import RoutedExpert, RouterGptqPrefetchPlanner
from cfie_training.training_base.training_memory_planner import (
    MemoryPlan, MemoryProfile, ModelDimensions, TrainingMemoryPlanner,
)
from cfie_training.training_base.window_commit import TrainingWindowCommitter
from cfie_training.training_base.window_plan import TrainableParamSpec, TrainingWindowBudget


def _require_non_empty_string(name: str, value: str) -> None:
    if not value.strip(): raise ValueError(f"{name} must be a non-empty string")

def _require_positive_int(name: str, value: int) -> None:
    if value < 1: raise ValueError(f"{name} must be >= 1")


# ──────────────────── 协议定义 ────────────────────

class TrainingDataBatch(Protocol):
    """训练数据 batch 接口。"""
    @property
    def input_ids(self) -> torch.Tensor: ...
    def labels(self) -> torch.Tensor | None: ...


class TrainingModelWrapper(Protocol):
    """训练模型接口——Qwen35ForTraining 或其他模型需满足此协议。"""
    def train(self, mode: bool = True) -> None: ...
    def forward(self, input_ids: torch.Tensor, *, hot_param_ids: Iterable[str] | None = None) -> Any: ...
    def backward(self, loss: Any) -> None: ...
    def collect_gradients(self, param_ids: Iterable[str]) -> dict[str, torch.Tensor]: ...
    def zero_grad(self) -> None: ...


# ──────────────────── 测试桩 ────────────────────

@dataclass(slots=True)
class NoOpTrainingModel:
    """测试用空操作模型——返回伪梯度和零 loss。仅用于单元测试。"""
    param_shapes: dict[str, torch.Size] = field(default_factory=dict)
    param_sizes: dict[str, int] = field(default_factory=dict)
    def train(self, mode: bool = True) -> None: pass
    def forward(self, input_ids: torch.Tensor, *, hot_param_ids: Iterable[str] | None = None) -> torch.Tensor:
        return torch.tensor(0.0, requires_grad=True)
    def backward(self, loss: Any) -> None:
        if hasattr(loss, "backward"): loss.backward()
    def collect_gradients(self, param_ids: Iterable[str]) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        for param_id in param_ids:
            num_elements = self.param_sizes.get(param_id, 0)
            shape = self.param_shapes.get(param_id, None)
            if shape is not None: tensor = torch.randn(shape, dtype=torch.float32)
            elif num_elements > 0: tensor = torch.randn(num_elements, dtype=torch.float32)
            else: tensor = torch.randn(2, dtype=torch.float32)
            result[param_id] = tensor
        return result
    def zero_grad(self) -> None: pass


# ──────────────────── 训练循环数据结构 ────────────────────

@dataclass(slots=True)
class TrainingDataBatchInput:
    """单步训练的输入数据。"""
    input_ids: torch.Tensor                                    # [B, T] token IDs
    _labels: torch.Tensor | None = None                        # [B, T] label IDs
    global_step: int = 0
    epoch: int = 0
    dataset_cursor: str = ""                                   # 数据集游标（用于恢复）
    router_plan: tuple[RoutedExpert, ...] = ()
    consumed_samples: int = 0
    consumed_tokens: int = 0
    def labels(self) -> torch.Tensor | None: return self._labels


@dataclass(slots=True)
class TrainingLoopConfig:
    """训练循环配置——所有参数可通过 CLI 传入。"""
    adam_config: AdamWConfig = field(default_factory=lambda: AdamWConfig(lr=0.1))
    shadow_dtype: torch.dtype = torch.float16                  # GPU shadow 精度
    shadow_device: torch.device | str = "cpu"                  # GPU shadow 设备
    bucket_capacity_bytes: int = 1 << 20                       # 梯度 bucket 容量（待调优）
    max_sealed_buckets: int = 4                                # 最大 seal bucket 数（待调优）
    window_steps: int = 50                                     # 兜底窗口步数
    window_budget: TrainingWindowBudget = field(default_factory=TrainingWindowBudget)
    memory_profile: MemoryProfile | None = None
    model_dims: ModelDimensions | None = None
    router_prefetch_depth: int = 16
    resident_cache_capacity_bytes: int = 1 << 20
    resident_cache_backend: ResidentGptqPayloadBackend | None = None
    enable_peak_monitor: bool = True
    resource_thresholds: TrainingResourceThresholds = field(default_factory=TrainingResourceThresholds)
    resource_sampler: TrainingResourceSampler | None = None
    dynamic_hotset_max_experts: int = 0                        # 动态 hot set 大小（0=禁用）
    dynamic_hotset_min_priority: float = 0.0
    predictor_checkpoint: str = ""                             # predictor checkpoint 路径
    predictor_hidden_size: int = 3072
    predictor_num_layers: int = 48
    predictor_num_experts: int = 256


@dataclass(frozen=True, slots=True)
class TrainingLoopStepResult:
    """单步训练结果。"""
    global_step: int; epoch: int; loss: float
    grad_norms: dict[str, float]; update_norms: dict[str, float]
    threshold_events: tuple[ThresholdEvent, ...]
    resource_peaks: TrainingResourcePeaks | None
    flush_completed: bool; window_committed: bool


@dataclass(frozen=True, slots=True)
class _HotSetSwitchDecision:
    """Hot set 切换决策。"""
    old_hot_param_ids: tuple[str, ...]; new_hot_param_ids: tuple[str, ...]
    @property
    def changed(self) -> bool: return self.old_hot_param_ids != self.new_hot_param_ids


# ──────────────────── 训练主循环 ────────────────────

@dataclass(slots=True)
class TrainingLoop:
    """训练主循环——设计文档 Section 16。

    持有所有训练状态：stores、gradient window、scheduler、predictor、telemetry。
    通过 attach_model/attach_dataloader 注入外部组件后调用 run() 或 run_step()。
    """
    # ── 存储层 ──
    fp32_store: Any                                             # NVMe FP32 主参数存储
    adam_store: Any                                             # NVMe Adam 状态存储
    gptq_store: Any                                             # NVMe GPTQ 缓存存储
    manifest: Any
    progress_writer: ProgressStateWriter                        # 训练进度原子写入
    committer: TrainingWindowCommitter                          # 窗口提交（NVMe + CPU 缓存）

    # ── 训练窗口 ──
    hot_window: HotParamTrainingWindow                          # FP32 master + Adam + shadow + bucket ring
    gptq_requantizer: GptqCacheRequantizer | GptqMarlinBundleRequantizer  # 训练后重量化

    # ── 专家缓存与预取 ──
    resident_cache: ResidentGptqCache                           # GPU resident cold expert cache
    router_prefetch_planner: RouterGptqPrefetchPlanner          # predictor-guided prefetch 计划

    # ── 调度与监控 ──
    hot_set_scheduler: HotSetScheduler | None                   # hot set 选择
    memory_planner: TrainingMemoryPlanner | None                # 显存规划
    telemetry: PeakTelemetryRecorder | None                     # 峰值监控
    window_budget: TrainingWindowBudget
    window_steps: int
    current_hot_param_ids: tuple[str, ...]
    dynamic_hotset_max_experts: int = 0
    dynamic_hotset_min_priority: float = 0.0
    predictor_guide: Any = None                                # PredictorHotSetGuide 实例

    # ── 运行时状态 ──
    _global_step: int = 0
    _steps_in_window: int = 0
    _model: TrainingModelWrapper | None = None
    _dataloader: Iterator[TrainingDataBatchInput] | None = None

    # ── 外部注入 ──
    def attach_model(self, model: TrainingModelWrapper) -> None:
        self._model = model

    def attach_dataloader(self, dataloader: Iterator[TrainingDataBatchInput]) -> None:
        self._dataloader = dataloader

    # ──────────────────── 单步训练 ────────────────────

    def run_step(
        self, batch: TrainingDataBatchInput, *,
        current_experts: Iterable[RoutedExpert] = (),
        predicted_experts: Iterable[RoutedExpert] = (),
    ) -> TrainingLoopStepResult:
        """执行一步完整训练: prefetch → forward → lock → backward → unlock → Adam → commit。

        流程（设计文档 Section 16）:
          1. Predictor prefetch（不锁，router 确认后再 lock）
          2. Forward（router 选专家，记录激活的冷专家 ID）
          3. Lock 冷专家（forward 后、backward 前，防淘汰）
          4. Backward（冷专家权重参与 dInput 计算）
          5. Unlock 冷专家（backward 完成）
          6. 收集梯度 → bucket → Adam 更新
          7. 窗口提交（hot set 切换 或 window_steps 兜底）
        """
        # ── Step 1: Predictor prefetch ──
        prefetch_result = self.router_prefetch_planner.execute(
            self.resident_cache,
            current_experts=current_experts, predicted_experts=(),
            lock=False,  # 不锁——router 确认激活后才 lock
        )
        self.router_prefetch_planner.execute(
            self.resident_cache,
            current_experts=(), predicted_experts=predicted_experts,
            allow_partial=True, lock=False,
        )

        # ── Step 2: Forward ──
        self._model.train(True); self._model.zero_grad()
        if self.memory_planner is not None:
            self._setup_checkpoint_policy()  # activation checkpoint segment 规划
        loss_tensor = self._model.forward(batch.input_ids, hot_param_ids=self.hot_window.hot_param_ids)

        # ── Step 3-5: Lock → Backward → Unlock ──
        self._lock_active_cold_experts()          # router 确认 → lock
        self._model.backward(loss_tensor)
        loss_value = float(loss_tensor.detach().item())
        self._unlock_active_cold_experts()        # backward 完成 → unlock

        # ── Step 6: 梯度收集 → bucket → Adam ──
        grads = self._model.collect_gradients(param_ids=self.hot_window.hot_param_ids)
        summaries: list[HotParamWindowUpdateSummary] = []
        for param_id, grad in grads.items():
            try:
                buckets = self.hot_window.add_gradient(param_id, grad)
            except KeyError:
                continue
            if buckets:
                summaries.append(self.hot_window.apply_buckets(buckets, optimizer_step=self._global_step + 1))
        final_summary = self.hot_window.drain_ready(optimizer_step=self._global_step + 1)
        if final_summary.touched_param_ids or grads:
            summaries.append(final_summary)
        update_summary = _merge_summaries(summaries)
        self._global_step += 1; self._steps_in_window += 1

        # ── 遥测 ──
        expert_cache_miss_rate = (0.0 if self.resident_cache.stats.requests == 0
                                  else self.resident_cache.stats.miss_rate)
        bucket_wait = (self.hot_window.bucket_ring.sealed_count
                       / max(self.hot_window.bucket_ring.max_sealed_buckets, 1)
                       if hasattr(self.hot_window, "bucket_ring") else 0.0)
        threshold_events: tuple[ThresholdEvent, ...] = ()
        peaks: TrainingResourcePeaks | None = None
        if self.telemetry is not None:
            threshold_events = self.telemetry.record_step(
                step=self._global_step, grad_bucket_wait_ratio=bucket_wait,
                expert_cache_miss_rate=expert_cache_miss_rate,
                expert_cache_capacity_pressure_rate=0.0,
                flush_seconds=0.0, expected_flush_seconds=0.0, pinned_bytes=0,
            )
            peaks = self.telemetry.peaks
        if self.telemetry is not None and self.telemetry.monitor.requires_replan():
            self._apply_degradation()

        # ── Step 7: 窗口提交 ──
        flush_completed = False; window_committed = False
        if self.hot_set_scheduler is not None:
            switch = self._commit_and_switch_hot_set(
                batch=batch,
                current_experts=current_experts, predicted_experts=predicted_experts,
            )
            if switch is not None and switch.changed:
                flush_completed = True; window_committed = True
                self.current_hot_param_ids = switch.new_hot_param_ids
                self._steps_in_window = 0
        # 兜底: window_steps 到但未切换 → 强制提交
        if not window_committed and self._steps_in_window >= self.window_steps:
            flush_completed, window_committed = self._commit_current_window(batch)

        self.resident_cache.unlock(prefetch_result.locked_bundle_ids)
        return TrainingLoopStepResult(
            global_step=self._global_step, epoch=batch.epoch, loss=loss_value,
            grad_norms=update_summary.grad_norms, update_norms=update_summary.update_norms,
            threshold_events=threshold_events, resource_peaks=peaks,
            flush_completed=flush_completed, window_committed=window_committed,
        )

    # ──────────────────── 窗口提交 + Hot Set 切换 ────────────────────

    def _commit_and_switch_hot_set(
        self, *, batch: TrainingDataBatchInput,
        current_experts: Iterable[RoutedExpert], predicted_experts: Iterable[RoutedExpert],
    ) -> _HotSetSwitchDecision | None:
        """窗口提交 + hot set 切换合并（设计文档 Section 13.2）。

        1. drain 所有 bucket → Adam 更新
        2. 构建 commit payload → 写入 NVMe + CPU GPTQ 缓存
        3. Scheduler 选择新 hot set（predictor 候选优先）
        4. 从 NVMe 加载新 hot set → CPU master
        """
        if self.dynamic_hotset_max_experts == 0 or self.hot_set_scheduler is None:
            return None

        # 提交当前窗口: drain → payload → commit → mark_committed
        self.hot_window.drain_all(optimizer_step=self._global_step)
        payload = self.hot_window.make_commit_payload(
            global_step=self._global_step, epoch=batch.epoch, dataset_cursor=batch.dataset_cursor,
            consumed_samples=batch.consumed_samples, consumed_tokens=batch.consumed_tokens,
            gptq_update_builder=self.gptq_requantizer,
        )
        self.committer.commit_window(
            fp32_updates=payload.fp32_updates, adam_updates=payload.adam_updates,
            gptq_updates=payload.gptq_updates,
            global_step=payload.global_step, epoch=payload.epoch,
            dataset_cursor=payload.dataset_cursor, round_id=self._global_step,
        )
        self.resident_cache.invalidate(tuple((payload.gptq_updates or {}).keys()))
        self.hot_window.mark_committed()

        # 选择新 hot set: predictor 收集 → 合并 → 优先级排序
        old_ids = self.current_hot_param_ids
        predictor_experts = self._collect_predictor_experts()
        all_experts = tuple(predicted_experts) + tuple(current_experts) + predictor_experts
        selected_ids, _ = self._select_hot_params_by_priority(all_experts, predictor_candidates=predictor_experts)
        if not selected_ids or set(selected_ids) == set(old_ids):
            return None

        new_ids = tuple(selected_ids)
        self.hot_window.switch_hot_params(new_ids)  # 从 NVMe 加载新 hot set
        self.router_prefetch_planner.set_hot_experts(_expert_keys_from_param_ids(new_ids))
        self.current_hot_param_ids = new_ids
        return _HotSetSwitchDecision(old_hot_param_ids=old_ids, new_hot_param_ids=new_ids)

    def _commit_current_window(self, batch: TrainingDataBatchInput) -> tuple[bool, bool]:
        """兜底提交: window_steps 到但 hot set 未切换时调用。"""
        self.hot_window.drain_all(optimizer_step=self._global_step)
        payload = self.hot_window.make_commit_payload(
            global_step=self._global_step, epoch=batch.epoch, dataset_cursor=batch.dataset_cursor,
            consumed_samples=batch.consumed_samples, consumed_tokens=batch.consumed_tokens,
            gptq_update_builder=self.gptq_requantizer,
        )
        self.committer.commit_window(
            fp32_updates=payload.fp32_updates, adam_updates=payload.adam_updates,
            gptq_updates=payload.gptq_updates,
            global_step=payload.global_step, epoch=payload.epoch,
            dataset_cursor=payload.dataset_cursor, round_id=self._global_step,
        )
        self.resident_cache.invalidate(tuple((payload.gptq_updates or {}).keys()))
        self.hot_window.mark_committed()
        self._steps_in_window = 0
        return True, True

    # ──────────────────── Activation Checkpoint ────────────────────

    def _setup_checkpoint_policy(self) -> None:
        """使用 memory_planner 生成 activation checkpoint policy 并设置到模型。"""
        if self.memory_planner is None or self._model is None: return
        from cfie_training.training_base.activation_checkpoint import (
            ActivationCheckpointPlanner, ActivationCheckpointPolicy,
        )
        specs = []
        for pid in self.hot_window.hot_param_ids:
            record = self.fp32_store.records.get(pid)
            if record is not None:
                specs.append(TrainableParamSpec(param_id=pid, kind="moe", fp32_bytes=record.num_bytes))
        if not specs: return
        planner = ActivationCheckpointPlanner()
        segments = planner.plan_segments(
            specs,
            bucket_size_bytes=self.hot_window.bucket_ring.bucket_capacity_bytes,
            num_buckets=self.hot_window.bucket_ring.max_sealed_buckets,
            vram_budget_bytes=self.memory_planner.plan.vram_budget if self.memory_planner.plan else 28 << 30,
        )
        if segments:
            self._model.checkpoint_policy = ActivationCheckpointPolicy(segments)

    # ──────────────────── GPU Cache Lock/Unlock ────────────────────

    def _lock_active_cold_experts(self) -> None:
        """Router 确认激活后，lock 当前 step 需要的冷专家（forward→backward 期间防淘汰）。"""
        if self._model is None: return
        active = getattr(self._model, "_active_expert_ids", set())
        if not active: return
        bundle_ids: list[str] = []
        for layer_id, expert_id in active:
            bundles = self.router_prefetch_planner.layer_expert_to_bundles.get((layer_id, expert_id))
            if bundles is None: continue
            for bid in bundles.bundle_ids:
                if bid in self.resident_cache.resident_bundle_ids:
                    bundle_ids.append(bid)
        if bundle_ids: self.resident_cache.lock(tuple(bundle_ids))

    def _unlock_active_cold_experts(self) -> None:
        """Backward 完成后 unlock 本 step 的冷专家。"""
        if self._model is None: return
        active = getattr(self._model, "_active_expert_ids", set())
        if not active: return
        bundle_ids: list[str] = []
        for layer_id, expert_id in active:
            bundles = self.router_prefetch_planner.layer_expert_to_bundles.get((layer_id, expert_id))
            if bundles is None: continue
            bundle_ids.extend(bundles.bundle_ids)
        if bundle_ids: self.resident_cache.unlock(tuple(bundle_ids))

    # ──────────────────── Predictor 集成 ────────────────────

    def _collect_predictor_experts(self) -> tuple[tuple[int, int], ...]:
        """从模型 forward 中收集 predictor 预测的专家候选（stride 层对齐）。"""
        if self.predictor_guide is None or not self.predictor_guide.loaded or self._model is None:
            return ()
        try:
            layer_states = getattr(self._model, "_layer_hidden_states", {})
            stride = self.predictor_guide.stride_layers
            for layer_idx in sorted(layer_states, reverse=True):
                if layer_idx % stride == 0:
                    return self.predictor_guide.predict_next_experts(
                        hidden_state=layer_states[layer_idx], insertion_layer_index=layer_idx,
                    )
        except Exception: pass
        return ()

    # ──────────────────── Hot Set 选择 ────────────────────

    def _select_hot_params_by_priority(
        self, experts: tuple[RoutedExpert, ...], *,
        predictor_candidates: tuple[tuple[int, int], ...] = (),
    ) -> tuple[list[str], list[tuple[int, int]]]:
        """按 priority + predictor bonus 排序选择 top experts。"""
        predictor_set = set(predictor_candidates)
        selected_experts: list[tuple[int, int]] = []; selected_param_ids: list[str] = []

        def _sort_key(expert: RoutedExpert) -> float:
            bonus = 10.0 if (expert.layer_id, expert.expert_id) in predictor_set else 0.0
            return -(expert.priority + bonus)

        for expert in sorted(experts, key=_sort_key):
            if expert.priority < self.dynamic_hotset_min_priority and (expert.layer_id, expert.expert_id) not in predictor_set:
                continue
            key = (expert.layer_id, expert.expert_id)
            if key in selected_experts: continue
            param_ids = _param_ids_for_expert(key)
            if not all(hasattr(self.fp32_store, "records") and pid in self.fp32_store.records for pid in param_ids):
                continue
            selected_experts.append(key); selected_param_ids.extend(param_ids)
            if len(selected_experts) >= self.dynamic_hotset_max_experts: break
        return selected_param_ids, selected_experts

    # ──────────────────── 降级 ────────────────────

    def _apply_degradation(self) -> None:
        """根据 PeakMonitor 阈值事件执行降级（设计文档 Section 12.3）。"""
        if self.telemetry is None: return
        for action in self.telemetry.monitor.suggested_actions():
            if "resident cache" in action: self.resident_cache.clear_unlocked()
            elif "prefetch horizon" in action or "reduce prefetch depth" in action:
                if self.router_prefetch_planner.prefetch_depth > 2:
                    self.router_prefetch_planner.prefetch_depth -= 2
            elif "reduce hot set" in action:
                if self.dynamic_hotset_max_experts > 1:
                    self.dynamic_hotset_max_experts = max(1, self.dynamic_hotset_max_experts // 2)
        self.telemetry.monitor.reset_window_counters()

    # ──────────────────── 批量训练 ────────────────────

    def run(self, *, num_steps: int | None = None) -> list[TrainingLoopStepResult]:
        """运行 num_steps 步训练。"""
        if self._dataloader is None:
            raise RuntimeError("No dataloader attached to TrainingLoop")
        results: list[TrainingLoopStepResult] = []
        for step_idx, batch in enumerate(self._dataloader):
            if num_steps is not None and step_idx >= num_steps: break
            results.append(self.run_step(batch))
        return results

    # ──────────────────── 工厂方法 ────────────────────

    @classmethod
    def from_stores(
        cls, *, fp32_store: Any, adam_store: Any, gptq_store: Any,
        manifest: Any, progress_writer: ProgressStateWriter,
        hot_param_ids: tuple[str, ...], config: TrainingLoopConfig | None = None,
    ) -> "TrainingLoop":
        """从 NVMe stores 构造完整的 TrainingLoop。

        创建 HotParamTrainingWindow → TrainingWindowCommitter → GptqRequantizer
        → ResidentGptqCache → RouterPrefetchPlanner → HotSetScheduler → Telemetry
        """
        cfg = config or TrainingLoopConfig()
        # 训练窗口
        hot_window = HotParamTrainingWindow.load_from_stores(
            fp32_store=fp32_store, adam_store=adam_store,
            updater=CpuAdamFp8Updater(cfg.adam_config),
            hot_param_ids=hot_param_ids, bucket_capacity_bytes=cfg.bucket_capacity_bytes,
            shadow_dtype=cfg.shadow_dtype, shadow_device=cfg.shadow_device,
            max_sealed_buckets=cfg.max_sealed_buckets,
        )
        # 窗口提交器
        committer = TrainingWindowCommitter(fp32_store, progress_writer, adam_store=adam_store, gptq_store=gptq_store)
        # GPTQ 重量化器
        param_to_bundle = getattr(manifest, "param_to_gptq_bundle", {}) if manifest else {}
        gptq_codec = getattr(manifest, "gptq_codec", None) if manifest else None
        if gptq_codec is not None:
            gptq_requantizer = GptqMarlinBundleRequantizer(store=gptq_store, param_to_bundle=param_to_bundle)
        else:
            from cfie_training.training_base.gptq_requant import SymmetricInt4GptqCodec, SymmetricInt4GptqLayout, GptqCacheRequantizer as R
            gptq_requantizer = R(store=gptq_store, param_to_bundle=param_to_bundle, codec=SymmetricInt4GptqCodec(SymmetricInt4GptqLayout()))
        # GPU resident cache
        resident_cache = ResidentGptqCache(gptq_store, capacity_bytes=cfg.resident_cache_capacity_bytes, backend=cfg.resident_cache_backend) \
            if cfg.resident_cache_backend else ResidentGptqCache(gptq_store, capacity_bytes=cfg.resident_cache_capacity_bytes)
        # Router prefetch planner
        router_planner = RouterGptqPrefetchPlanner.from_param_to_bundle(param_to_bundle, prefetch_depth=cfg.router_prefetch_depth)
        router_planner.set_hot_experts(_expert_keys_from_param_ids(hot_param_ids))
        # Scheduler + MemoryPlanner + Telemetry
        scheduler = HotSetScheduler()
        memory_planner = TrainingMemoryPlanner(cfg.memory_profile, cfg.model_dims) if cfg.memory_profile and cfg.model_dims else None
        telemetry = PeakTelemetryRecorder(monitor=PeakMonitor(cfg.resource_thresholds), sampler=cfg.resource_sampler or TorchProcessResourceSampler()) if cfg.enable_peak_monitor else None

        return cls(
            fp32_store=fp32_store, adam_store=adam_store, gptq_store=gptq_store,
            manifest=manifest, progress_writer=progress_writer, committer=committer,
            hot_window=hot_window, gptq_requantizer=gptq_requantizer,
            resident_cache=resident_cache, router_prefetch_planner=router_planner,
            hot_set_scheduler=scheduler, memory_planner=memory_planner, telemetry=telemetry,
            window_budget=cfg.window_budget, window_steps=cfg.window_steps,
            current_hot_param_ids=hot_param_ids,
            dynamic_hotset_max_experts=cfg.dynamic_hotset_max_experts,
            dynamic_hotset_min_priority=cfg.dynamic_hotset_min_priority,
            predictor_guide=_maybe_load_predictor(cfg),
        )


# ──────────────────── 模块级工具函数 ────────────────────

def _merge_summaries(summaries: list[HotParamWindowUpdateSummary]) -> HotParamWindowUpdateSummary:
    """合并多个 apply_buckets 的摘要为一个。"""
    touched_ids: list[str] = []; grad_norms: dict[str, float] = {}; update_norms: dict[str, float] = {}; drained: list[int] = []
    for s in summaries:
        drained.extend(s.drained_bucket_ids)
        for pid in s.touched_param_ids:
            if pid not in touched_ids: touched_ids.append(pid)
            grad_norms[pid] = s.grad_norms[pid]; update_norms[pid] = s.update_norms[pid]
    return HotParamWindowUpdateSummary(
        touched_param_ids=tuple(touched_ids), grad_norms=grad_norms,
        update_norms=update_norms, drained_bucket_ids=tuple(drained),
    )

def _expert_keys_from_param_ids(param_ids: Iterable[str]) -> tuple[tuple[int, int], ...]:
    """从 param_id 列表提取 (layer_id, expert_id) 集合。"""
    keys: set[tuple[int, int]] = set()
    for pid in param_ids:
        parts = pid.split(".")
        if len(parts) != 5 or parts[0] != "layers" or parts[2] != "experts": continue
        try: keys.add((int(parts[1]), int(parts[3])))
        except ValueError: continue
    return tuple(sorted(keys))

def _param_ids_for_expert(key: tuple[int, int]) -> tuple[str, str]:
    """(layer_id, expert_id) → (w13_param_id, w2_param_id)。"""
    lid, eid = key; p = f"layers.{lid}.experts.{eid}"
    return (f"{p}.w13_weight", f"{p}.w2_weight")

def _maybe_load_predictor(cfg: Any) -> Any:
    """如果配置了 predictor checkpoint，则加载并返回 PredictorHotSetGuide。"""
    if not cfg.predictor_checkpoint: return None
    try:
        from cfie_training.training_base.predictor_integration import PredictorHotSetGuide
        return PredictorHotSetGuide.from_checkpoint(
            checkpoint_path=cfg.predictor_checkpoint,
            hidden_size=cfg.predictor_hidden_size,
            num_layers=cfg.predictor_num_layers,
            num_experts=cfg.predictor_num_experts,
        )
    except Exception: return None
