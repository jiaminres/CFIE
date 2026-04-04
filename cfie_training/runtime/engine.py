"""First executable training engine for the standalone CFIE training package."""

from __future__ import annotations

from dataclasses import dataclass, field

from cfie_training.config import TrainingProjectConfig
from cfie_training.runtime.executor import (
    RepresentativeBucketExecutor,
    RepresentativeExecutionResult,
)
from cfie_training.runtime.memory import TrainingMemoryPlan, TrainingMemoryPlanner
from cfie_training.runtime.optimizer import CPUOptimizerRuntime, OptimizerStepResult
from cfie_training.runtime.planner import ExpertRotationScheduler, LayerBucketPlanner
from cfie_training.runtime.residency import (
    ParameterResidencyController,
    ResidencyPlanResult,
)
from cfie_training.runtime.timeline import MicroBatchPlanner, StreamTimelinePlanner
from cfie_training.runtime.transport import WeightTransportPlanner, WeightTransportRuntime
from cfie_training.runtime.types import (
    BatchShape,
    BucketStreamTrace,
    ExpertWindowPlan,
    LayerBucketPlan,
    ParameterLoadSummary,
    ParameterPrefetchSummary,
    ParameterShardSnapshot,
    ParameterSourceSummary,
    ParameterStoreSummary,
    RepresentativeBucketRecord,
    RepresentativeExecutionSummary,
    RuntimeAction,
    StreamOperationTrace,
    StreamOverlapSummary,
    TransportExecutionSummary,
    TransportPlanSummary,
    TrainingRunTrace,
    TrainingRuntimeSnapshot,
    TrainingStepTrace,
)
from cfie_training.runtime.store import ParameterShardStore
from cfie_training.runtime.warehouse import ParameterWarehouse, WarehouseStepResult


@dataclass(slots=True)
class _TrainingRuntimeState:
    next_step_index: int = 0
    static_modules_staged: bool = False
    cumulative_samples_processed: int = 0
    cumulative_tokens_processed: int = 0
    retired_window_group_ids: tuple[str, ...] = ()


@dataclass(slots=True)
class FirstVersionTrainingEngine:
    config: TrainingProjectConfig
    _bucket_planner: LayerBucketPlanner = field(init=False)
    _memory_planner: TrainingMemoryPlanner = field(init=False)
    _residency_controller: ParameterResidencyController = field(init=False)
    _rotation: ExpertRotationScheduler = field(init=False)
    _warehouse: ParameterWarehouse = field(init=False)
    _parameter_store: ParameterShardStore = field(init=False)
    _transport_planner: WeightTransportPlanner = field(init=False)
    _transport_runtime: WeightTransportRuntime = field(init=False)
    _executor: RepresentativeBucketExecutor = field(init=False)
    _optimizer: CPUOptimizerRuntime = field(init=False)
    _micro_batch_planner: MicroBatchPlanner = field(init=False)
    _timeline_planner: StreamTimelinePlanner = field(init=False)
    _layer_buckets: tuple[LayerBucketPlan, ...] = field(init=False)
    _state: _TrainingRuntimeState = field(init=False)

    def __post_init__(self) -> None:
        # 先校验整份训练项目配置。
        self.config.validate()
        # 初始化层 bucket 规划器。
        self._bucket_planner = LayerBucketPlanner(self.config)
        # 初始化显存 / 内存规划器。
        self._memory_planner = TrainingMemoryPlanner(self.config)
        # 初始化驻留状态控制器。
        self._residency_controller = ParameterResidencyController(self.config)
        # 初始化 expert 轮转调度器。
        self._rotation = ExpertRotationScheduler(self.config)
        # 初始化参数仓库。
        self._warehouse = ParameterWarehouse(self.config)
        # 初始化参数存储运行时。
        self._parameter_store = ParameterShardStore(self.config)
        # 初始化权重传输规划器。
        self._transport_planner = WeightTransportPlanner(self.config)
        # 初始化权重传输运行时。
        self._transport_runtime = WeightTransportRuntime(self.config)
        # 初始化代表性 bucket 执行器。
        self._executor = RepresentativeBucketExecutor(self.config)
        # 初始化 CPU 优化器运行时。
        self._optimizer = CPUOptimizerRuntime(self.config)
        # 初始化 micro-batch 规划器。
        self._micro_batch_planner = MicroBatchPlanner(self.config)
        # 初始化流时间线规划器。
        self._timeline_planner = StreamTimelinePlanner(self.config)
        # 预构建全局层 bucket 计划。
        self._layer_buckets = self._bucket_planner.build()
        # 初始化引擎内部运行时状态。
        self._state = _TrainingRuntimeState()

    @property
    def layer_buckets(self) -> tuple[LayerBucketPlan, ...]:
        # 对外暴露当前引擎使用的层 bucket 规划。
        return self._layer_buckets

    @property
    def next_step_index(self) -> int:
        # 返回下一个待执行 step 的序号。
        return self._state.next_step_index

    def build_memory_plan(self, batch: BatchShape) -> TrainingMemoryPlan:
        # 根据当前 batch 规模构建一份内存预算规划。
        return self._memory_planner.build(batch)

    # 对 group_id 序列做稳定去重，避免重复回收同一组分片。
    def _dedupe_group_ids(
        self,
        group_ids: list[str] | tuple[str, ...],
    ) -> tuple[str, ...]:
        # 直接用 dict 保留首次出现顺序并完成去重。
        return tuple(dict.fromkeys(group_ids))

    # 对给定 group_id 集合执行参数与优化器双侧 offload。
    def _offload_window_groups(
        self,
        *,
        group_ids: tuple[str, ...],
        parameter_store: ParameterShardStore,
        optimizer: CPUOptimizerRuntime,
    ) -> bool:
        # 空集合时无需执行任何回收动作。
        if not group_ids:
            return False
        # 先把优化器状态落冷。
        optimizer.offload_group_ids(group_ids=group_ids)
        # 再同步参数主副本并把参数分片落冷。
        parameter_store.offload_group_ids(
            group_ids=group_ids,
            sync_fp32_to_nvme=True,
        )
        return True

    # 判断当前窗口级常驻是否已经触达 CPU / GPU 热预算。
    def _window_pressure_reclaim_required(
        self,
        *,
        memory_plan: TrainingMemoryPlan,
        parameter_store: ParameterShardStore,
    ) -> bool:
        # 先按现有接口检查 CPU hot 常驻量是否超过可用预算。
        cpu_hot_over_budget = (
            parameter_store.cpu_hot_resident_bytes()
            > memory_plan.cpu_hot.available_bytes
        )
        # 再读取一次参数存储汇总，检查 GPU packed cache 是否超过热预算。
        parameter_store_summary = parameter_store.summary()
        gpu_hot_over_budget = (
            parameter_store_summary.gpu_quantized_bytes
            > memory_plan.gpu_hot.available_bytes
        )
        # 任一热层超预算，都需要触发窗口回收。
        return cpu_hot_over_budget or gpu_hot_over_budget

    # 在 step 结束时决定当前窗口保留、退休或立即回收的 routed group 集合。
    def _finalize_window_retention(
        self,
        *,
        current_window_group_ids: tuple[str, ...],
        retired_window_group_ids: tuple[str, ...],
        retain_routed_window_after_step: bool,
        allow_boundary_window_retention: bool,
        memory_plan: TrainingMemoryPlan,
        parameter_store: ParameterShardStore,
        optimizer: CPUOptimizerRuntime,
    ) -> tuple[tuple[str, ...], bool]:
        # -----------------
        # 先标准化当前窗口与历史退休窗口的 group_id 集合。
        offload_performed = False
        retired_window_group_ids = self._dedupe_group_ids(retired_window_group_ids)
        current_window_group_ids = self._dedupe_group_ids(current_window_group_ids)
        retained_group_ids = tuple(
            group_id
            for group_id in retired_window_group_ids
            if group_id not in current_window_group_ids
        )

        # -----------------
        # 未启用 routed window 保留，或更新后本就不 offload 时，直接清空历史退休窗口。
        if (
            not self.config.expert_rotation.retain_active_window_state_in_memory
            or not self.config.optimizer.offload_state_after_update
        ):
            offload_performed = self._offload_window_groups(
                group_ids=retained_group_ids,
                parameter_store=parameter_store,
                optimizer=optimizer,
            ) or offload_performed
            return (), offload_performed
        # 当前 step 没有 routed group 被更新时，也无需继续处理。
        if not current_window_group_ids:
            if allow_boundary_window_retention:
                return retained_group_ids, offload_performed
            offload_performed = self._offload_window_groups(
                group_ids=retained_group_ids,
                parameter_store=parameter_store,
                optimizer=optimizer,
            ) or offload_performed
            return (), offload_performed

        # -----------------
        # 若当前热层已超预算，则无论是否同窗口都优先回收全部退休窗口与当前窗口。
        pressure_reclaim_required = self._window_pressure_reclaim_required(
            memory_plan=memory_plan,
            parameter_store=parameter_store,
        )
        if pressure_reclaim_required:
            offload_performed = self._offload_window_groups(
                group_ids=self._dedupe_group_ids(
                    retained_group_ids + current_window_group_ids
                ),
                parameter_store=parameter_store,
                optimizer=optimizer,
            ) or offload_performed
            return (), offload_performed

        # -----------------
        # 下一步仍复用当前窗口时，仅把“已退休且未被重激活”的历史窗口继续保留。
        if retain_routed_window_after_step and not pressure_reclaim_required:
            return retained_group_ids, offload_performed

        # -----------------
        # 跨窗口边界且已知下一步时，把当前窗口并入退休热集，预算内尽量延长热驻留。
        if (
            not retain_routed_window_after_step
            and allow_boundary_window_retention
        ):
            return self._dedupe_group_ids(
                retained_group_ids + current_window_group_ids
            ), offload_performed

        # -----------------
        # 其余情况都把历史退休窗口与当前 routed window 一并落冷。
        offload_performed = self._offload_window_groups(
            group_ids=self._dedupe_group_ids(
                retained_group_ids + current_window_group_ids
            ),
            parameter_store=parameter_store,
            optimizer=optimizer,
        ) or offload_performed
        return (), offload_performed

    def _run_bucket_stream(
        self,
        *,
        step_index: int,
        batch: BatchShape,
        memory_plan: TrainingMemoryPlan,
        active_expert_ids: tuple[int, ...],
        prefetched_expert_ids: tuple[int, ...],
        parameter_shards: tuple[ParameterShardSnapshot, ...],
        parameter_store: ParameterShardStore,
        executor: RepresentativeBucketExecutor,
        optimizer: CPUOptimizerRuntime,
        transport_runtime: WeightTransportRuntime,
        retain_routed_window_after_step: bool,
        retired_window_group_ids: tuple[str, ...],
        allow_boundary_window_retention: bool,
    ) -> tuple[
        RepresentativeExecutionResult,
        OptimizerStepResult,
        tuple[BucketStreamTrace, ...],
        tuple[str, ...],
    ]:
        # -----------------
        # 初始化 bucket 级聚合容器与本步公共上下文。
        bucket_records = []
        optimizer_updates = []
        bucket_stream_traces = []
        optimizer_summary = None
        gradient_payload_count = 0
        window_hot_routed_group_ids: list[str] = []
        # 先把当前 batch 切成 micro-batch。
        micro_batches = self._micro_batch_planner.plan(batch)
        # 识别“当前 active experts 对应的预取窗口”分片，避免重复保留其计算视图。
        consumed_prefetch_group_ids = tuple(
            shard.group_id
            for shard in parameter_shards
            if shard.component == "expert_window_prefetch"
            and shard.expert_ids == active_expert_ids
        )
        # 命中已消费的预取窗口时，先释放其 GPU 计算视图。
        if consumed_prefetch_group_ids:
            parameter_store.release_compute_views(group_ids=consumed_prefetch_group_ids)
        # 若存在下一窗口预取集且显存预算允许，则提前把它们 stage 到计算设备。
        if prefetched_expert_ids and memory_plan.gpu_hot.within_budget:
            prefetch_window_shards = tuple(
                shard
                for shard in parameter_shards
                if shard.component == "expert_window_prefetch"
                and shard.expert_ids == prefetched_expert_ids
            )
            parameter_store.stage_compute_views(
                step_index=step_index,
                parameter_shards=prefetch_window_shards,
                device=executor.compute_device,
            )
        # 预先为每个 bucket 建好“bucket_id -> shard 列表”映射，避免循环里重复筛选。
        bucket_shard_map = {
            bucket.bucket_id: executor.select_bucket_shards(
                bucket_id=bucket.bucket_id,
                parameter_shards=parameter_shards,
            )
            for bucket in self._layer_buckets
        }
        # 读取 bucket 级 lookahead 深度配置。
        lookahead_depth = max(0, self.config.bucket_schedule.prefetch_buckets)

        # -----------------
        # 逐个 bucket 执行 prefetch、forward/backward、optimizer update 与 trace 聚合。
        for bucket_index, bucket in enumerate(self._layer_buckets):
            # 先取出当前 bucket 关联的参数分片。
            bucket_shards = bucket_shard_map[bucket.bucket_id]
            # 记录当前 bucket 对应的 group_id 集合。
            bucket_group_ids = tuple(shard.group_id for shard in bucket_shards)
            # 统计 prefetch 前仍处于 CPU hot 的分片数量。
            cpu_hot_before_prefetch, _ = parameter_store.resident_tier_counts_for_groups(
                bucket_group_ids
            )
            # 构造 lookahead bucket 列表，供预取阶段复用。
            lookahead_buckets = tuple(
                future_bucket
                for future_bucket in self._layer_buckets[
                    bucket_index + 1 : bucket_index + 1 + lookahead_depth
                ]
            )
            # 预取候选默认包含当前 bucket 自己的分片。
            prefetch_shards = list(bucket_shards)
            for future_bucket in lookahead_buckets:
                # 再把 lookahead bucket 的分片并进预取集合。
                prefetch_shards.extend(bucket_shard_map[future_bucket.bucket_id])
            # 执行当前 bucket + lookahead bucket 的参数预取。
            parameter_store.prefetch_shards(
                step_index=step_index,
                parameter_shards=tuple(prefetch_shards),
            )
            # 抽取仅属于当前 bucket 的 prefetch 汇总。
            bucket_prefetch_summary = parameter_store.prefetch_summary_for_groups(
                bucket_group_ids
            )
            # 为当前 bucket 的权重访问申请 / 复用传输缓冲区。
            transport_runtime.stage_weight_buffers(
                step_index=step_index,
                bucket_id=bucket.bucket_id,
                parameter_shards=bucket_shards,
            )
            # 把当前 bucket 的计算视图提前搬到目标设备。
            parameter_store.stage_compute_views(
                step_index=step_index,
                parameter_shards=bucket_shards,
                device=executor.compute_device,
            )
            # 用于聚合当前 bucket 内所有 micro-batch 的执行结果。
            micro_bucket_records = []
            bucket_host_gradient_bytes = 0
            bucket_gradient_payloads = 0
            bucket_update_group_ids: list[str] = []
            keep_resident_group_ids = ()
            # 若配置要求保留 active routed window，则记录当前 bucket 的 routed group。
            if self.config.expert_rotation.retain_active_window_state_in_memory:
                keep_resident_group_ids = tuple(
                    shard.group_id
                    for shard in bucket_shards
                    if shard.component == "bucket_active_experts"
                )
                window_hot_routed_group_ids.extend(keep_resident_group_ids)
            # 逐个 micro-batch 执行当前 bucket。
            for micro_batch_id, micro_batch in enumerate(micro_batches):
                # 执行 bucket 前后向并收集梯度。
                bucket_result = executor.execute_bucket(
                    step_index=step_index,
                    batch=micro_batch,
                    bucket=bucket,
                    parameter_shards=parameter_shards,
                    parameter_store=parameter_store,
                )
                # 统计当前 micro-batch 产出的梯度 payload 数量。
                gradient_count = len(bucket_result.gradients)
                gradient_payload_count += gradient_count
                bucket_gradient_payloads += gradient_count
                # 记录当前 micro-batch 的 bucket 级执行结果。
                micro_bucket_records.append(bucket_result.bucket_record)
                # 为当前 micro-batch 的梯度回传申请 / 复用梯度缓冲区。
                transport_runtime.stage_gradient_buffers(
                    step_index=step_index,
                    bucket_id=bucket.bucket_id,
                    micro_batch_id=micro_batch_id,
                    gradient_payloads=bucket_result.gradients,
                )
                # 执行优化器更新，并按配置决定哪些 group 保持驻留。
                bucket_optimizer_result = optimizer.apply_gradients(
                    step_index=step_index,
                    parameter_shards=bucket_shards,
                    parameter_store=parameter_store,
                    gradient_payloads=bucket_result.gradients,
                    keep_resident_group_ids=keep_resident_group_ids,
                )
                # 并入当前 step 的优化器更新记录。
                optimizer_updates.extend(bucket_optimizer_result.updates)
                # 记录被当前 bucket 更新过的 group_id。
                bucket_update_group_ids.extend(
                    update.group_id for update in bucket_optimizer_result.updates
                )
                # 保存最近一次优化器汇总，最后会作为 step 级 summary 使用。
                optimizer_summary = bucket_optimizer_result.optimizer_summary
                # 记录当前 bucket 需要的宿主侧梯度缓冲区峰值。
                bucket_host_gradient_bytes = max(
                    bucket_host_gradient_bytes,
                    bucket_optimizer_result.optimizer_summary.last_bucket_staged_gradient_bytes,
                )
                # 当前 micro-batch 更新完成后立即释放梯度传输缓冲区。
                transport_runtime.release_buffers(
                    buffer_kind="gradient_stage",
                    owner_group_ids=tuple(
                        payload.group_id for payload in bucket_result.gradients
                    ),
                )
            # 统计当前 bucket 在 load 路径上的访问汇总。
            bucket_load_summary = parameter_store.load_summary_for_groups(
                bucket_group_ids
            )
            # bucket 结束后释放其权重传输缓冲区。
            transport_runtime.release_buffers(
                buffer_kind="weight_stage",
                owner_group_ids=bucket_group_ids,
            )
            # 每个 bucket 至少应该产生一条 micro-batch 记录。
            assert micro_bucket_records
            # 把多个 micro-batch 的 bucket 结果聚合成单个 bucket 级记录。
            aggregated_bucket_record = RepresentativeBucketRecord(
                bucket_id=bucket.bucket_id,
                attention_types=bucket.attention_types,
                contains_full_attention=bucket.contains_full_attention,
                active_expert_ids=micro_bucket_records[0].active_expert_ids,
                semantic_layout_used=all(
                    record.semantic_layout_used for record in micro_bucket_records
                ),
                semantic_roles=tuple(
                    sorted(
                        {
                            role
                            for record in micro_bucket_records
                            for role in record.semantic_roles
                        }
                    )
                ),
                execution_mode=(
                    micro_bucket_records[0].execution_mode
                    if len(
                        {record.execution_mode for record in micro_bucket_records}
                    )
                    == 1
                    else "mixed_bucket_execution"
                ),
                loss_value=sum(
                    record.loss_value for record in micro_bucket_records
                ) / len(micro_bucket_records),
                non_routed_gradient_l2_norm=max(
                    record.non_routed_gradient_l2_norm
                    for record in micro_bucket_records
                ),
                expert_gradient_l2_norm=max(
                    record.expert_gradient_l2_norm
                    for record in micro_bucket_records
                ),
                peak_activation_bytes=max(
                    record.peak_activation_bytes for record in micro_bucket_records
                ),
            )
            # 保存聚合后的 bucket 记录。
            bucket_records.append(aggregated_bucket_record)
            # 统计当前 bucket 更新结束后，相关分片还剩多少 CPU hot / 已落冷。
            cpu_hot_after_update, offloaded_after_update = (
                parameter_store.resident_tier_counts_for_groups(bucket_group_ids)
            )
            # 生成当前 bucket 的双流 trace 记录。
            bucket_stream_traces.append(
                BucketStreamTrace(
                    bucket_id=bucket.bucket_id,
                    layer_indices=bucket.layer_indices,
                    attention_types=bucket.attention_types,
                    micro_batch_count=len(micro_batches),
                    cpu_hot_shards_before_prefetch=cpu_hot_before_prefetch,
                    lookahead_prefetched_bucket_ids=tuple(
                        future_bucket.bucket_id for future_bucket in lookahead_buckets
                    ),
                    prefetch_summary=bucket_prefetch_summary,
                    load_summary=bucket_load_summary,
                    bucket_record=aggregated_bucket_record,
                    optimizer_update_count=len(bucket_update_group_ids),
                    optimizer_updated_groups=tuple(bucket_update_group_ids),
                    gradient_release_count=bucket_gradient_payloads,
                    gradients_released_immediately=(
                        self.config.bucket_schedule.release_gradients_immediately
                    ),
                    host_gradient_buffer_bytes=bucket_host_gradient_bytes,
                    host_gradient_buffer_storage_dtype=(
                        "fp32"
                        if optimizer_summary is None
                        else optimizer_summary.gradient_buffer_storage_dtype
                    ),
                    offloaded_shards_after_update=offloaded_after_update,
                    cpu_hot_shards_after_update=cpu_hot_after_update,
                )
            )

        # -----------------
        # 在 step 结束时统一结算 routed window 的保留 / 退休 / 回收策略。
        next_retired_window_group_ids, offload_performed = (
            self._finalize_window_retention(
                current_window_group_ids=tuple(window_hot_routed_group_ids),
                retired_window_group_ids=retired_window_group_ids,
                retain_routed_window_after_step=retain_routed_window_after_step,
                allow_boundary_window_retention=allow_boundary_window_retention,
                memory_plan=memory_plan,
                parameter_store=parameter_store,
                optimizer=optimizer,
            )
        )
        if offload_performed:
            # 有显式回收动作时，重新生成 offload 后的优化器汇总。
            optimizer_summary = optimizer.summary()

        # 当前 step 至少应有一份优化器汇总。
        assert optimizer_summary is not None
        # 聚合本 step 的整体执行统计。
        execution_summary = RepresentativeExecutionSummary(
            executed_buckets=len(bucket_records),
            gradient_shards=gradient_payload_count,
            total_loss=sum(record.loss_value for record in bucket_records),
            max_gradient_l2_norm=max(
                (
                    max(
                        record.non_routed_gradient_l2_norm,
                        record.expert_gradient_l2_norm,
                    )
                    for record in bucket_records
                ),
                default=0.0,
            ),
            peak_activation_bytes=max(
                (record.peak_activation_bytes for record in bucket_records),
                default=0,
            ),
            peak_host_gradient_buffer_bytes=max(
                (
                    trace.host_gradient_buffer_bytes
                    for trace in bucket_stream_traces
                ),
                default=0,
            ),
            gradient_buffer_storage_dtype=(
                optimizer_summary.gradient_buffer_storage_dtype
            ),
            bucket_records=tuple(bucket_records),
        )
        # 返回执行结果占位对象、优化器汇总以及 bucket 级 trace。
        return (
            RepresentativeExecutionResult(
                gradients=(),
                execution_summary=execution_summary,
            ),
            OptimizerStepResult(
                updates=tuple(optimizer_updates),
                optimizer_summary=optimizer_summary,
            ),
            tuple(bucket_stream_traces),
            next_retired_window_group_ids,
        )

    def _predict_step_state(
        self,
        *,
        step_index: int,
        memory_plan: TrainingMemoryPlan,
        batch: BatchShape,
        next_batch: BatchShape | None,
        stage_static_modules: bool,
    ) -> tuple[
        ResidencyPlanResult,
        WarehouseStepResult,
        ParameterStoreSummary,
        ParameterSourceSummary,
        ParameterPrefetchSummary,
        ParameterLoadSummary,
        TransportPlanSummary,
        TransportExecutionSummary,
        RepresentativeExecutionResult,
        OptimizerStepResult,
        tuple[BucketStreamTrace, ...],
    ]:
        # -----------------
        # 为纯预测路径重新构造一套独立的运行时组件，避免污染真实状态。
        controller = ParameterResidencyController(self.config)
        warehouse = ParameterWarehouse(self.config)
        parameter_store = ParameterShardStore(self.config)
        transport_planner = WeightTransportPlanner(self.config)
        transport_runtime = WeightTransportRuntime(self.config)
        executor = RepresentativeBucketExecutor(self.config)
        optimizer = CPUOptimizerRuntime(self.config)
        # 初始化本函数需要回传的各类结果占位。
        residency_plan: ResidencyPlanResult | None = None
        warehouse_result: WarehouseStepResult | None = None
        transport_summary: TransportPlanSummary | None = None
        transport_execution_summary: TransportExecutionSummary | None = None
        parameter_prefetch_summary: ParameterPrefetchSummary | None = None
        execution_result: RepresentativeExecutionResult | None = None
        optimizer_result: OptimizerStepResult | None = None
        bucket_stream_traces: tuple[BucketStreamTrace, ...] = ()
        predicted_retired_window_group_ids: tuple[str, ...] = ()
        # 从 step 0 预测到目标 step_index，逐步重放状态演进。
        for predicted_step in range(step_index + 1):
            # 只有最后一个预测步才使用调用方传入的 next_batch。
            predicted_next_batch = batch if predicted_step < step_index else next_batch
            # 推导当前预测步之前累计处理的样本数。
            predicted_cumulative_samples = predicted_step * batch.samples
            # 推导当前预测步之前累计处理的 token 数。
            predicted_cumulative_tokens = predicted_step * batch.total_tokens
            # 基于累计进度预测当前步的 expert window。
            predicted_window = self._rotation.plan_window(
                step_index=predicted_step,
                batch=batch,
                layer_buckets=self._layer_buckets,
                next_batch=predicted_next_batch,
                cumulative_samples_processed=predicted_cumulative_samples,
                cumulative_tokens_processed=predicted_cumulative_tokens,
            )
            # 取当前预测步的 active / prefetch expert 集。
            predicted_active = predicted_window.active_expert_ids
            predicted_prefetch = predicted_window.prefetched_expert_ids
            predicted_next_active: tuple[int, ...] = ()
            # 需要保留 active window 时，额外预测下一步 active experts。
            if (
                self.config.expert_rotation.retain_active_window_state_in_memory
                and predicted_next_batch is not None
            ):
                predicted_next_window = self._rotation.plan_window(
                    step_index=predicted_step + 1,
                    batch=predicted_next_batch,
                    layer_buckets=self._layer_buckets,
                    next_batch=predicted_next_batch,
                    cumulative_samples_processed=(
                        predicted_cumulative_samples + batch.samples
                    ),
                    cumulative_tokens_processed=(
                        predicted_cumulative_tokens + batch.total_tokens
                    ),
                )
                predicted_next_active = predicted_next_window.active_expert_ids
            # 默认只有 step 0 需要首次 stage static modules。
            predicted_stage_static = predicted_step == 0
            # 命中目标预测步时，以调用方显式传入的值为准。
            if predicted_step == step_index:
                predicted_stage_static = stage_static_modules
            # 基于预测得到的 active/prefetch 集生成驻留迁移计划。
            residency_plan = controller.plan_step(
                step_index=predicted_step,
                layer_buckets=self._layer_buckets,
                active_expert_ids=predicted_active,
                prefetched_expert_ids=predicted_prefetch,
                memory_plan=memory_plan,
                stage_static_modules=predicted_stage_static,
                update_state=True,
            )
            # 把驻留计划应用到参数仓库。
            warehouse_result = warehouse.apply_residency_plan(
                step_index=predicted_step,
                transitions=residency_plan.transitions,
                memory_plan=memory_plan,
            )
            # 基于当前触达分片生成传输规划。
            transport_summary = transport_planner.plan_step(
                warehouse_result.touched_shards
            )
            # 执行本预测步的文件缓存与传输 stage。
            transport_execution_summary = transport_runtime.execute_step(
                step_index=predicted_step,
                transport_plan=transport_summary,
            )
            # 将 transport cache 命中上下文同步到参数存储。
            parameter_store.set_transport_cache_context(
                step_index=predicted_step,
                cached_file_names=transport_runtime.cached_file_names(),
            )
            # 执行 bucket 流并得到执行、优化器与 trace 结果。
            (
                execution_result,
                optimizer_result,
                bucket_stream_traces,
                predicted_retired_window_group_ids,
            ) = self._run_bucket_stream(
                step_index=predicted_step,
                batch=batch,
                memory_plan=memory_plan,
                active_expert_ids=predicted_active,
                prefetched_expert_ids=predicted_prefetch,
                parameter_shards=warehouse_result.touched_shards,
                parameter_store=parameter_store,
                executor=executor,
                optimizer=optimizer,
                transport_runtime=transport_runtime,
                retain_routed_window_after_step=(
                    bool(predicted_next_active)
                    and predicted_next_active == predicted_active
                ),
                retired_window_group_ids=predicted_retired_window_group_ids,
                allow_boundary_window_retention=predicted_next_batch is not None,
            )
            # 读取本预测步最终 transport 执行摘要。
            transport_execution_summary = transport_runtime.step_summary()
            # 读取本预测步最终 prefetch 摘要。
            parameter_prefetch_summary = parameter_store.step_prefetch_summary()
        # 预测结束后，上述结果都应已被填充。
        assert residency_plan is not None
        assert warehouse_result is not None
        assert transport_summary is not None
        assert transport_execution_summary is not None
        assert parameter_prefetch_summary is not None
        assert execution_result is not None
        assert optimizer_result is not None
        return (
            residency_plan,
            warehouse_result,
            parameter_store.summary(),
            parameter_store.source_summary(warehouse_result.touched_shards),
            parameter_prefetch_summary,
            parameter_store.step_load_summary(),
            transport_summary,
            transport_execution_summary,
            execution_result,
            optimizer_result,
            bucket_stream_traces,
        )

    def _build_stream_schedule(
        self,
        *,
        batch: BatchShape,
        bucket_stream_traces: tuple[BucketStreamTrace, ...],
    ) -> tuple[
        tuple[BatchShape, ...],
        StreamOverlapSummary,
        tuple[StreamOperationTrace, ...],
    ]:
        # 先把 batch 规划成 micro-batch 序列。
        micro_batches = self._micro_batch_planner.plan(batch)
        # 再结合 bucket trace 生成双流时间线和重叠统计。
        overlap_summary, stream_operations = self._timeline_planner.plan(
            micro_batches=micro_batches,
            bucket_stream_traces=bucket_stream_traces,
        )
        # 返回调度后的 micro-batch、重叠汇总和流操作明细。
        return micro_batches, overlap_summary, stream_operations

    def _build_trace(
        self,
        *,
        step_index: int,
        batch: BatchShape,
        expert_window_plan: ExpertWindowPlan,
        active_experts: tuple[int, ...],
        prefetched_experts: tuple[int, ...],
        stage_static_modules: bool,
        residency_plan: ResidencyPlanResult,
        warehouse_result: WarehouseStepResult,
        parameter_store_summary: ParameterStoreSummary,
        parameter_source_summary: ParameterSourceSummary,
        parameter_prefetch_summary: ParameterPrefetchSummary,
        parameter_load_summary: ParameterLoadSummary,
        transport_summary: TransportPlanSummary,
        transport_execution_summary: TransportExecutionSummary,
        execution_result: RepresentativeExecutionResult,
        optimizer_result: OptimizerStepResult,
        bucket_stream_traces: tuple[BucketStreamTrace, ...],
        actions: tuple[RuntimeAction, ...],
    ) -> TrainingStepTrace:
        # 先基于 bucket trace 构建 micro-batch 调度和流时间线信息。
        (
            scheduled_micro_batches,
            stream_overlap_summary,
            stream_operations,
        ) = self._build_stream_schedule(
            batch=batch,
            bucket_stream_traces=bucket_stream_traces,
        )
        # 把本 step 的全部执行快照拼装成 TrainingStepTrace。
        return TrainingStepTrace(
            step_index=step_index,
            batch=batch,
            active_expert_ids=active_experts,
            prefetched_expert_ids=prefetched_experts,
            released_expert_ids=active_experts,
            expert_window_plan=expert_window_plan,
            layer_buckets=self._layer_buckets,
            scheduled_micro_batches=scheduled_micro_batches,
            bucket_stream_traces=bucket_stream_traces,
            stream_operations=stream_operations,
            actions=actions,
            static_modules_staged=stage_static_modules,
            residency_transitions=residency_plan.transitions,
            residency_ending_states=residency_plan.ending_states,
            parameter_shards=warehouse_result.touched_shards,
            warehouse_summary=warehouse_result.warehouse_summary,
            parameter_store_summary=parameter_store_summary,
            parameter_source_summary=parameter_source_summary,
            parameter_prefetch_summary=parameter_prefetch_summary,
            parameter_load_summary=parameter_load_summary,
            transport_summary=transport_summary,
            transport_execution_summary=transport_execution_summary,
            stream_overlap_summary=stream_overlap_summary,
            optimizer_updates=optimizer_result.updates,
            optimizer_summary=optimizer_result.optimizer_summary,
            execution_summary=execution_result.execution_summary,
        )

    def _build_actions(
        self,
        step_index: int,
        expert_window_plan: ExpertWindowPlan,
        active_experts: tuple[int, ...],
        prefetched_experts: tuple[int, ...],
        stage_static_modules: bool,
        batch: BatchShape,
    ) -> tuple[RuntimeAction, ...]:
        # 用列表累积当前 step 会发生的高层运行时动作。
        actions: list[RuntimeAction] = []
        # 首次执行时需要先 stage 静态模块。
        if stage_static_modules:
            actions.append(
                RuntimeAction(
                    name="stage_static_modules",
                    owner="host_io",
                    description=(
                        "Stage embeddings, router, norms, and shared expert state "
                        "before routed-expert rotation begins."
                    ),
                    metadata={
                        "batch_total_tokens": batch.total_tokens,
                        "weight_offload_backend": self.config.resource_policy.weight_offload_backend,
                    },
                )
            )

        # 无论是否首次执行，都要为 routed experts 做 prefetch / active window 切换。
        actions.append(
            RuntimeAction(
                name="prefetch_routed_experts",
                owner="host_io",
                description=(
                    "Stage the active routed experts and the next prefetch window "
                    "without materializing the full 256-expert pool on device."
                ),
                metadata={
                    "step_index": step_index,
                    "compute_device": self.config.execution.compute_device,
                    "selection_strategy": expert_window_plan.selection_strategy,
                    "router_score_source": expert_window_plan.router_score_source,
                    "active_expert_ids": list(active_experts),
                    "prefetched_expert_ids": list(prefetched_experts),
                    "hot_expert_ids": list(expert_window_plan.hot_expert_ids),
                    "prefetch_priority_expert_ids": list(
                        expert_window_plan.prefetch_priority_expert_ids
                    ),
                },
            )
        )

        # 逐个 bucket 追加 forward / backward / update-release 三类动作。
        for bucket in self._layer_buckets:
            # 先抽出 bucket 级公共 metadata。
            bucket_metadata = {
                "bucket_id": bucket.bucket_id,
                "layer_indices": list(bucket.layer_indices),
                "attention_types": list(bucket.attention_types),
                "active_expert_ids": list(active_experts),
                "compute_device": self.config.execution.compute_device,
            }
            actions.append(
                RuntimeAction(
                    name="forward_bucket",
                    owner=self.config.execution.compute_stream_name,
                    description=(
                        "Run the forward path for the active layer bucket with "
                        "minimal activation residency."
                    ),
                    metadata=bucket_metadata,
                )
            )
            actions.append(
                RuntimeAction(
                    name="backward_bucket",
                    owner=self.config.execution.compute_stream_name,
                    description=(
                        "Recompute the bucket as needed and emit gradients as "
                        "soon as this bucket finishes backward."
                    ),
                    metadata=bucket_metadata,
                )
            )
            actions.append(
                RuntimeAction(
                    name="cpu_update_release_bucket",
                    owner=self.config.execution.transfer_stream_name,
                    description=(
                        "Ship gradients to the CPU update path, update parameters, "
                        "and release bucket-local state immediately."
                    ),
                    metadata={
                        **bucket_metadata,
                        "optimizer_device": self.config.execution.optimizer_device,
                        "gradient_device": self.config.execution.gradient_device,
                        "release_gradients_immediately": (
                            self.config.bucket_schedule.release_gradients_immediately
                        ),
                    },
                )
            )

        # 最后追加 expert window 轮转动作。
        actions.append(
            RuntimeAction(
                name="rotate_expert_window",
                owner="scheduler",
                description=(
                    "Advance the active expert window and keep the shared expert "
                    "outside routed-expert rotation."
                ),
                metadata={
                    "next_step_index": step_index + 1,
                    "rotate_every_steps": self.config.expert_rotation.rotate_every_steps,
                },
            )
        )
        # 返回当前 step 的动作清单。
        return tuple(actions)

    def plan_step(
        self,
        *,
        step_index: int,
        batch: BatchShape,
        next_batch: BatchShape | None = None,
        stage_static_modules: bool | None = None,
    ) -> TrainingStepTrace:
        # step 序号不能为负。
        if step_index < 0:
            raise ValueError("step_index must be >= 0")
        # 未提供 next_batch 时默认按同 batch 继续预测。
        if next_batch is None:
            next_batch = batch
        # 先预测当前 step 的 expert window。
        expert_window_plan = self._rotation.plan_window(
            step_index=step_index,
            batch=batch,
            layer_buckets=self._layer_buckets,
            next_batch=next_batch,
            cumulative_samples_processed=step_index * batch.samples,
            cumulative_tokens_processed=step_index * batch.total_tokens,
        )
        # 取出当前 step 的 active / prefetched experts。
        active_experts = expert_window_plan.active_expert_ids
        prefetched_experts = expert_window_plan.prefetched_expert_ids
        # 未显式指定时，仅 step 0 会 stage static modules。
        if stage_static_modules is None:
            stage_static_modules = step_index == 0
        # 构建当前 batch 对应的内存预算规划。
        memory_plan = self.build_memory_plan(batch)
        # 走预测路径，生成不污染真实状态的完整 step 结果。
        (
            residency_plan,
            warehouse_result,
            parameter_store_summary,
            parameter_source_summary,
            parameter_prefetch_summary,
            parameter_load_summary,
            transport_summary,
            transport_execution_summary,
            execution_result,
            optimizer_result,
            bucket_stream_traces,
        ) = self._predict_step_state(
            step_index=step_index,
            memory_plan=memory_plan,
            batch=batch,
            next_batch=next_batch,
            stage_static_modules=stage_static_modules,
        )
        # 构建高层动作清单。
        actions = self._build_actions(
            step_index=step_index,
            expert_window_plan=expert_window_plan,
            active_experts=active_experts,
            prefetched_experts=prefetched_experts,
            stage_static_modules=stage_static_modules,
            batch=batch,
        )
        # 拼装并返回预测得到的 step trace。
        return self._build_trace(
            step_index=step_index,
            batch=batch,
            expert_window_plan=expert_window_plan,
            active_experts=active_experts,
            prefetched_experts=prefetched_experts,
            stage_static_modules=stage_static_modules,
            residency_plan=residency_plan,
            warehouse_result=warehouse_result,
            parameter_store_summary=parameter_store_summary,
            parameter_source_summary=parameter_source_summary,
            parameter_prefetch_summary=parameter_prefetch_summary,
            parameter_load_summary=parameter_load_summary,
            transport_summary=transport_summary,
            transport_execution_summary=transport_execution_summary,
            execution_result=execution_result,
            optimizer_result=optimizer_result,
            bucket_stream_traces=bucket_stream_traces,
            actions=actions,
        )

    def run_step(
        self,
        batch: BatchShape,
        next_batch: BatchShape | None = None,
    ) -> TrainingStepTrace:
        # -----------------
        # 先取出真实执行路径下的 step 序号和内存预算。
        step_index = self._state.next_step_index
        memory_plan = self.build_memory_plan(batch)
        # 保存调用方是否显式提供了 next_batch，后面决定是否预判下一窗口。
        provided_next_batch = next_batch
        # 未提供 next_batch 时默认按当前 batch 继续。
        if next_batch is None:
            next_batch = batch
        # 规划当前 step 的 expert window。
        expert_window_plan = self._rotation.plan_window(
            step_index=step_index,
            batch=batch,
            layer_buckets=self._layer_buckets,
            next_batch=next_batch,
            cumulative_samples_processed=self._state.cumulative_samples_processed,
            cumulative_tokens_processed=self._state.cumulative_tokens_processed,
        )
        next_active_experts: tuple[int, ...] = ()
        # 若启用 active window 保留，则额外预测下一步 active experts。
        if (
            self.config.expert_rotation.retain_active_window_state_in_memory
            and provided_next_batch is not None
        ):
            next_expert_window_plan = self._rotation.plan_window(
                step_index=step_index + 1,
                batch=next_batch,
                layer_buckets=self._layer_buckets,
                next_batch=next_batch,
                cumulative_samples_processed=(
                    self._state.cumulative_samples_processed + batch.samples
                ),
                cumulative_tokens_processed=(
                    self._state.cumulative_tokens_processed + batch.total_tokens
                ),
            )
            next_active_experts = next_expert_window_plan.active_expert_ids
        # 取出本步的 active / prefetched experts。
        active_experts = expert_window_plan.active_expert_ids
        prefetched_experts = expert_window_plan.prefetched_expert_ids
        # static modules 只在首次真实执行时 stage 一次。
        stage_static_modules = not self._state.static_modules_staged
        # 先由驻留控制器生成本步迁移计划。
        residency_plan = self._residency_controller.plan_step(
            step_index=step_index,
            layer_buckets=self._layer_buckets,
            active_expert_ids=active_experts,
            prefetched_expert_ids=prefetched_experts,
            memory_plan=memory_plan,
            stage_static_modules=stage_static_modules,
            update_state=True,
        )
        # 将驻留迁移应用到真实参数仓库。
        warehouse_result = self._warehouse.apply_residency_plan(
            step_index=step_index,
            transitions=residency_plan.transitions,
            memory_plan=memory_plan,
        )
        # 基于本步触达分片生成传输规划。
        transport_summary = self._transport_planner.plan_step(
            warehouse_result.touched_shards
        )
        # 执行真实传输步骤。
        transport_execution_summary = self._transport_runtime.execute_step(
            step_index=step_index,
            transport_plan=transport_summary,
        )
        # 同步 transport cache 命中上下文到真实参数存储。
        self._parameter_store.set_transport_cache_context(
            step_index=step_index,
            cached_file_names=self._transport_runtime.cached_file_names(),
        )
        # 执行 bucket 流、优化器更新和 trace 聚合。
        (
            execution_result,
            optimizer_result,
            bucket_stream_traces,
            self._state.retired_window_group_ids,
        ) = self._run_bucket_stream(
            step_index=step_index,
            batch=batch,
            memory_plan=memory_plan,
            active_expert_ids=active_experts,
            prefetched_expert_ids=prefetched_experts,
            parameter_shards=warehouse_result.touched_shards,
            parameter_store=self._parameter_store,
            executor=self._executor,
            optimizer=self._optimizer,
            transport_runtime=self._transport_runtime,
            retain_routed_window_after_step=(
                bool(next_active_experts)
                and next_active_experts == active_experts
            ),
            retired_window_group_ids=self._state.retired_window_group_ids,
            allow_boundary_window_retention=provided_next_batch is not None,
        )
        # 读取真实执行后的 transport 执行汇总。
        transport_execution_summary = self._transport_runtime.step_summary()
        # 读取真实执行后的参数预取汇总。
        parameter_prefetch_summary = self._parameter_store.step_prefetch_summary()
        # 构建本步高层动作清单。
        actions = self._build_actions(
            batch=batch,
            step_index=step_index,
            expert_window_plan=expert_window_plan,
            active_experts=active_experts,
            prefetched_experts=prefetched_experts,
            stage_static_modules=stage_static_modules,
        )
        # 拼装训练 step trace。
        trace = self._build_trace(
            step_index=step_index,
            batch=batch,
            expert_window_plan=expert_window_plan,
            active_experts=active_experts,
            prefetched_experts=prefetched_experts,
            stage_static_modules=stage_static_modules,
            residency_plan=residency_plan,
            warehouse_result=warehouse_result,
            parameter_store_summary=self._parameter_store.summary(),
            parameter_source_summary=self._parameter_store.source_summary(
                warehouse_result.touched_shards
            ),
            parameter_prefetch_summary=parameter_prefetch_summary,
            parameter_load_summary=self._parameter_store.step_load_summary(),
            transport_summary=transport_summary,
            transport_execution_summary=transport_execution_summary,
            execution_result=execution_result,
            optimizer_result=optimizer_result,
            bucket_stream_traces=bucket_stream_traces,
            actions=actions,
        )
        # 真实执行完成后，静态模块已完成首次 stage。
        self._state.static_modules_staged = True
        # 推进下一步 step 序号。
        self._state.next_step_index += 1
        # 累计已处理样本数。
        self._state.cumulative_samples_processed += batch.samples
        # 累计已处理 token 数。
        self._state.cumulative_tokens_processed += batch.total_tokens
        # 返回真实执行得到的 trace。
        return trace

    def snapshot_state(self) -> TrainingRuntimeSnapshot:
        # 先取一次参数存储汇总，后面要把累计量化 / 同步计数写进快照。
        parameter_store_summary = self._parameter_store.summary()
        # 组装当前引擎全部关键运行时状态的快照对象。
        return TrainingRuntimeSnapshot(
            profile_name=self.config.profile_name,
            next_step_index=self._state.next_step_index,
            static_modules_staged=self._state.static_modules_staged,
            runtime_quantization_session_id=(
                self.config.runtime_quantization.session_id
            ),
            residency_static_modules_state=self._residency_controller.static_modules_state,
            residency_staged_expert_ids=self._residency_controller.staged_expert_ids,
            warehouse_shards=self._warehouse.snapshot(),
            parameter_store_shards=self._parameter_store.snapshot(),
            transport_cached_files=self._transport_runtime.snapshot(),
            transport_buffers=self._transport_runtime.buffer_snapshot(),
            optimizer_shards=self._optimizer.snapshot(),
            cumulative_samples_processed=self._state.cumulative_samples_processed,
            cumulative_tokens_processed=self._state.cumulative_tokens_processed,
            retired_window_group_ids=self._state.retired_window_group_ids,
            rotation_window_cache=self._rotation.window_cache_snapshot(),
            parameter_store_cumulative_quantize_ops=(
                parameter_store_summary.cumulative_quantize_ops
            ),
            parameter_store_cumulative_nvme_sync_ops=(
                parameter_store_summary.cumulative_nvme_sync_ops
            ),
        )

    def load_state(self, snapshot: TrainingRuntimeSnapshot) -> None:
        # 快照的 profile_name 必须和当前引擎配置一致。
        if snapshot.profile_name != self.config.profile_name:
            raise ValueError(
                "snapshot.profile_name must match engine config.profile_name"
            )
        # 若快照里的量化会话 id 与当前配置不同，则切换会话并重建相关运行时。
        if (
            snapshot.runtime_quantization_session_id
            and snapshot.runtime_quantization_session_id
            != self.config.runtime_quantization.session_id
        ):
            self.config.runtime_quantization.session_id = (
                snapshot.runtime_quantization_session_id
            )
            self._parameter_store = ParameterShardStore(self.config)
            self._optimizer = CPUOptimizerRuntime(self.config)
        # 恢复引擎级运行时计数器。
        self._state = _TrainingRuntimeState(
            next_step_index=snapshot.next_step_index,
            static_modules_staged=snapshot.static_modules_staged,
            cumulative_samples_processed=snapshot.cumulative_samples_processed,
            cumulative_tokens_processed=snapshot.cumulative_tokens_processed,
            retired_window_group_ids=snapshot.retired_window_group_ids,
        )
        # 恢复驻留控制器内部状态。
        self._residency_controller.load_state(
            static_modules_state=snapshot.residency_static_modules_state,
            staged_expert_ids=snapshot.residency_staged_expert_ids,
        )
        # 恢复 expert window 轮换缓存。
        self._rotation.load_window_cache(snapshot.rotation_window_cache)
        # 恢复参数仓库快照。
        self._warehouse.load_snapshot(snapshot.warehouse_shards)
        # 恢复参数存储快照及其累计计数。
        self._parameter_store.load_snapshot(
            snapshot.parameter_store_shards,
            cumulative_quantize_ops=(
                snapshot.parameter_store_cumulative_quantize_ops
            ),
            cumulative_nvme_sync_ops=(
                snapshot.parameter_store_cumulative_nvme_sync_ops
            ),
        )
        # 恢复 transport 文件缓存与缓冲区池。
        self._transport_runtime.load_snapshot(snapshot.transport_cached_files)
        self._transport_runtime.load_buffer_snapshot(snapshot.transport_buffers)
        # 恢复优化器分片状态。
        self._optimizer.load_snapshot(snapshot.optimizer_shards)

    def simulate(
        self,
        *,
        steps: int,
        batch: BatchShape,
    ) -> TrainingRunTrace:
        # 至少需要模拟 1 步。
        if steps < 1:
            raise ValueError("steps must be >= 1")
        # 顺序执行指定步数，并把最后一步的 next_batch 置空。
        traces = tuple(
            self.run_step(batch, next_batch=batch if index + 1 < steps else None)
            for index in range(steps)
        )
        # 返回整段 run 的聚合 trace。
        return TrainingRunTrace(
            profile_name=self.config.profile_name,
            batch=batch,
            step_count=steps,
            steps=traces,
            resource_plan=self.build_memory_plan(batch).to_dict(),
        )
