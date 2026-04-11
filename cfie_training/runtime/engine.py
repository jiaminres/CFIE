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
    # 当前训练引擎绑定的训练项目配置对象。
    config: TrainingProjectConfig

    # 按层构造 bucket 计划的规划器。
    _bucket_planner: LayerBucketPlanner = field(init=False)

    # 基于 batch 形状构造多级内存预算的规划器。
    _memory_planner: TrainingMemoryPlanner = field(init=False)

    # 管理参数驻留状态迁移的控制器。
    _residency_controller: ParameterResidencyController = field(init=False)

    # 负责 active expert window 轮换的调度器。
    _rotation: ExpertRotationScheduler = field(init=False)

    # 维护参数分片仓库视图的运行时组件。
    _warehouse: ParameterWarehouse = field(init=False)

    # 维护参数分片冷热层状态与加载路径的参数存储运行时。
    _parameter_store: ParameterShardStore = field(init=False)

    # 负责生成权重传输计划的规划器。
    _transport_planner: WeightTransportPlanner = field(init=False)

    # 负责执行权重文件缓存、缓冲区申请与传输阶段动作的运行时。
    _transport_runtime: WeightTransportRuntime = field(init=False)

    # 负责执行 bucket 级前向、反向与梯度收集的执行器。
    _executor: RepresentativeBucketExecutor = field(init=False)

    # 负责 CPU 侧优化器更新与状态管理的运行时。
    _optimizer: CPUOptimizerRuntime = field(init=False)

    # 负责将 batch 切分为 micro-batch 序列的规划器。
    _micro_batch_planner: MicroBatchPlanner = field(init=False)

    # 负责根据 bucket trace 构造双流时间线的规划器。
    _timeline_planner: StreamTimelinePlanner = field(init=False)

    # 当前引擎持有的全局层 bucket 规划结果。
    _layer_buckets: tuple[LayerBucketPlan, ...] = field(init=False)

    # 当前引擎的内部运行时状态。
    _state: _TrainingRuntimeState = field(init=False)

    def __post_init__(self) -> None:
        # ------------------------------- 校验训练配置并初始化核心运行时组件 -------------------------------
        # 对当前训练配置执行统一合法性校验。
        self.config.validate()

        # 创建层 bucket 规划器。
        self._bucket_planner = LayerBucketPlanner(self.config)

        # 创建训练内存规划器。
        self._memory_planner = TrainingMemoryPlanner(self.config)

        # 创建参数驻留状态控制器。
        self._residency_controller = ParameterResidencyController(self.config)

        # 创建 expert window 轮换调度器。
        self._rotation = ExpertRotationScheduler(self.config)

        # 创建参数仓库运行时。
        self._warehouse = ParameterWarehouse(self.config)

        # 创建参数分片存储运行时。
        self._parameter_store = ParameterShardStore(self.config)

        # 创建权重传输规划器。
        self._transport_planner = WeightTransportPlanner(self.config)

        # 创建权重传输执行运行时。
        self._transport_runtime = WeightTransportRuntime(self.config)

        # 创建 bucket 级执行器。
        self._executor = RepresentativeBucketExecutor(self.config)

        # 创建 CPU 优化器运行时。
        self._optimizer = CPUOptimizerRuntime(self.config)

        # 创建 micro-batch 规划器。
        self._micro_batch_planner = MicroBatchPlanner(self.config)

        # 创建双流时间线规划器。
        self._timeline_planner = StreamTimelinePlanner(self.config)

        # ------------------------------- 预构建全局层 bucket 规划并初始化内部状态 -------------------------------
        # 预先构造当前配置对应的全局层 bucket 计划。
        self._layer_buckets = self._bucket_planner.build()

        # 初始化训练引擎内部运行时状态。
        self._state = _TrainingRuntimeState()

    @property
    def layer_buckets(self) -> tuple[LayerBucketPlan, ...]:
        # ------------------------------- 对外暴露当前引擎使用的层 bucket 规划 -------------------------------
        # 返回当前训练引擎内部维护的层 bucket 计划。
        return self._layer_buckets

    @property
    def next_step_index(self) -> int:
        # ------------------------------- 返回下一个待执行 step 的序号 -------------------------------
        # 返回当前运行时状态中记录的下一个 step 编号。
        return self._state.next_step_index

    def build_memory_plan(self, batch: BatchShape) -> TrainingMemoryPlan:
        # ------------------------------- 基于给定 batch 形状构造内存规划结果 -------------------------------
        # 调用内存规划器，根据当前 batch 构造训练内存预算结果。
        return self._memory_planner.build(batch)

    def _dedupe_group_ids(
        self,
        group_ids: list[str] | tuple[str, ...],
    ) -> tuple[str, ...]:
        # ------------------------------- 对 group_id 序列执行稳定去重 -------------------------------
        # 使用 dict 保留首次出现顺序，并生成去重后的 group_id 元组。
        return tuple(dict.fromkeys(group_ids))

    def _offload_window_groups(
        self,
        *,
        group_ids: tuple[str, ...],
        parameter_store: ParameterShardStore,
        optimizer: CPUOptimizerRuntime,
    ) -> bool:
        # ------------------------------- 对指定 routed window group 执行统一落冷 -------------------------------
        # 当 group_id 集合为空时，无需执行任何回收操作。
        if not group_ids:
            # 返回 False，表示本次没有发生实际 offload。
            return False

        # 先将这些 group 对应的优化器状态执行落冷。
        optimizer.offload_group_ids(group_ids=group_ids)

        # 再将这些 group 对应的参数主副本同步并落冷。
        parameter_store.offload_group_ids(
            group_ids=group_ids,
            sync_fp32_to_nvme=True,
        )

        # 返回 True，表示本次确实执行了 offload。
        return True

    def _window_pressure_reclaim_required(
        self,
        *,
        memory_plan: TrainingMemoryPlan,
        parameter_store: ParameterShardStore,
    ) -> bool:
        # ------------------------------- 判断当前窗口级热驻留是否已经触发预算压力 -------------------------------
        # 计算当前 CPU hot 常驻字节数是否已经超过可用 CPU hot 预算。
        cpu_hot_over_budget = (
            parameter_store.cpu_hot_resident_bytes()
            > memory_plan.cpu_hot.available_bytes
        )

        # 读取一次参数存储汇总结果，用于获取 GPU packed cache 的占用情况。
        parameter_store_summary = parameter_store.summary()

        # 计算当前 GPU 量化权重热驻留是否已经超过可用 GPU hot 预算。
        gpu_hot_over_budget = (
            parameter_store_summary.gpu_quantized_bytes
            > memory_plan.gpu_hot.available_bytes
        )

        # 只要 CPU hot 或 GPU hot 任意一侧超预算，就需要触发窗口级回收。
        return cpu_hot_over_budget or gpu_hot_over_budget

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
        # ------------------------------- 标准化当前窗口与历史退休窗口的 group_id 集合 -------------------------------
        # 初始化本轮是否执行过 offload 的标志位。
        offload_performed = False

        # 对历史退休窗口 group_id 集合执行稳定去重。
        retired_window_group_ids = self._dedupe_group_ids(retired_window_group_ids)

        # 对当前窗口 group_id 集合执行稳定去重。
        current_window_group_ids = self._dedupe_group_ids(current_window_group_ids)

        # 只保留历史退休窗口中那些不属于当前窗口的 group_id。
        retained_group_ids = tuple(
            group_id
            for group_id in retired_window_group_ids
            if group_id not in current_window_group_ids
        )

        # ------------------------------- 在不允许热保留时直接回收历史退休窗口 -------------------------------
        # 当未启用 routed window 热保留，或优化器更新后本来就不保留状态时，直接回收历史退休窗口。
        if (
            not self.config.expert_rotation.retain_active_window_state_in_memory
            or not self.config.optimizer.offload_state_after_update
        ):
            # 对历史退休窗口执行 offload，并累积回收标志。
            offload_performed = self._offload_window_groups(
                group_ids=retained_group_ids,
                parameter_store=parameter_store,
                optimizer=optimizer,
            ) or offload_performed

            # 返回空的退休窗口集合与是否执行过 offload 的标志。
            return (), offload_performed

        # ------------------------------- 当前 step 未更新 routed group 时处理历史窗口 -------------------------------
        # 当当前 step 根本没有 routed group 被更新时，说明无需继续处理当前窗口。
        if not current_window_group_ids:
            # 如果允许跨窗口边界保留历史热驻留，则直接返回现有历史退休窗口集合。
            if allow_boundary_window_retention:
                # 返回保留后的历史退休窗口集合。
                return retained_group_ids, offload_performed

            # 否则将历史退休窗口整体落冷。
            offload_performed = self._offload_window_groups(
                group_ids=retained_group_ids,
                parameter_store=parameter_store,
                optimizer=optimizer,
            ) or offload_performed

            # 返回空的退休窗口集合与是否执行过 offload 的标志。
            return (), offload_performed

        # ------------------------------- 在热层超预算时强制回收历史窗口与当前窗口 -------------------------------
        # 判断当前窗口级热驻留是否已经触发 CPU 或 GPU 热预算压力。
        pressure_reclaim_required = self._window_pressure_reclaim_required(
            memory_plan=memory_plan,
            parameter_store=parameter_store,
        )

        # 当预算压力成立时，优先回收历史退休窗口与当前窗口的全部 group。
        if pressure_reclaim_required:
            # 合并历史退休窗口与当前窗口 group_id 后执行统一 offload。
            offload_performed = self._offload_window_groups(
                group_ids=self._dedupe_group_ids(
                    retained_group_ids + current_window_group_ids
                ),
                parameter_store=parameter_store,
                optimizer=optimizer,
            ) or offload_performed

            # 返回空的退休窗口集合与是否执行过 offload 的标志。
            return (), offload_performed

        # ------------------------------- 下一步仍复用当前窗口时仅保留历史退休窗口 -------------------------------
        # 当下一步仍会复用当前窗口且当前没有预算压力时，不将当前窗口并入退休集合。
        if retain_routed_window_after_step and not pressure_reclaim_required:
            # 返回仅保留历史退休窗口的集合。
            return retained_group_ids, offload_performed

        # ------------------------------- 跨窗口边界时在预算允许下延长当前窗口热驻留 -------------------------------
        # 当下一步不再复用当前窗口，但允许跨边界保留时，将当前窗口并入退休热集。
        if (
            not retain_routed_window_after_step
            and allow_boundary_window_retention
        ):
            # 返回历史退休窗口与当前窗口合并后的去重集合。
            return self._dedupe_group_ids(
                retained_group_ids + current_window_group_ids
            ), offload_performed

        # ------------------------------- 其余情况统一回收历史窗口与当前窗口 -------------------------------
        # 将历史退休窗口与当前窗口合并后执行统一落冷。
        offload_performed = self._offload_window_groups(
            group_ids=self._dedupe_group_ids(
                retained_group_ids + current_window_group_ids
            ),
            parameter_store=parameter_store,
            optimizer=optimizer,
        ) or offload_performed

        # 返回空的退休窗口集合与是否执行过 offload 的标志。
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
        # ------------------------------- 初始化 bucket 级聚合容器与当前 step 的公共上下文 -------------------------------
        # 用于收集聚合后的 bucket 级执行记录。
        bucket_records = []

        # 用于收集当前 step 内全部优化器更新记录。
        optimizer_updates = []

        # 用于收集每个 bucket 对应的流级 trace。
        bucket_stream_traces = []

        # 当前 step 的优化器汇总结果，初始为空。
        optimizer_summary = None

        # 统计当前 step 内总共产生的梯度 payload 数量。
        gradient_payload_count = 0

        # 记录当前 step 内 routed window 中保持热驻留的 group_id。
        window_hot_routed_group_ids: list[str] = []

        # 根据当前 batch 形状先生成 micro-batch 序列。
        micro_batches = self._micro_batch_planner.plan(batch)

        # ------------------------------- 处理已消费的预取窗口并按需预取下一窗口计算视图 -------------------------------
        # 抽取属于当前 active experts 预取窗口的 group_id，用于消费后释放其计算视图。
        consumed_prefetch_group_ids = tuple(
            shard.group_id
            for shard in parameter_shards
            if shard.component == "expert_window_prefetch"
            and shard.expert_ids == active_expert_ids
        )

        # 当存在已消费的预取窗口时，先释放其 GPU 计算视图。
        if consumed_prefetch_group_ids:
            # 释放已命中的预取窗口对应的计算视图。
            parameter_store.release_compute_views(group_ids=consumed_prefetch_group_ids)

        # 当存在下一窗口预取专家集合且当前 GPU 热预算允许时，提前 stage 下一窗口计算视图。
        if prefetched_expert_ids and memory_plan.gpu_hot.within_budget:
            # 选取当前参数分片中属于下一预取窗口的分片集合。
            prefetch_window_shards = tuple(
                shard
                for shard in parameter_shards
                if shard.component == "expert_window_prefetch"
                and shard.expert_ids == prefetched_expert_ids
            )

            # 将下一窗口的计算视图预先 stage 到执行设备。
            parameter_store.stage_compute_views(
                step_index=step_index,
                parameter_shards=prefetch_window_shards,
                device=executor.compute_device,
            )

        # ------------------------------- 预构建 bucket 到参数分片的映射并读取 lookahead 配置 -------------------------------
        # 为每个 bucket 建立 bucket_id 到其参数分片列表的映射，避免循环内部重复筛选。
        bucket_shard_map = {
            bucket.bucket_id: executor.select_bucket_shards(
                bucket_id=bucket.bucket_id,
                parameter_shards=parameter_shards,
            )
            for bucket in self._layer_buckets
        }

        # 从配置中读取 bucket 级 lookahead 预取深度。
        lookahead_depth = max(0, self.config.bucket_schedule.prefetch_buckets)

        # ------------------------------- 逐个 bucket 执行预取、前反向、更新与 trace 聚合 -------------------------------
        # 按 layer bucket 顺序逐个执行当前 step 的 bucket 流。
        for bucket_index, bucket in enumerate(self._layer_buckets):
            # 取出当前 bucket 对应的参数分片集合。
            bucket_shards = bucket_shard_map[bucket.bucket_id]

            # 抽取当前 bucket 对应的全部 group_id。
            bucket_group_ids = tuple(shard.group_id for shard in bucket_shards)

            # 统计预取前当前 bucket 仍位于 CPU hot 的分片数量。
            cpu_hot_before_prefetch, _ = parameter_store.resident_tier_counts_for_groups(
                bucket_group_ids
            )

            # 根据 lookahead 深度构造后续 bucket 列表。
            lookahead_buckets = tuple(
                future_bucket
                for future_bucket in self._layer_buckets[
                    bucket_index + 1 : bucket_index + 1 + lookahead_depth
                ]
            )

            # 预取列表先从当前 bucket 自己的参数分片开始。
            prefetch_shards = list(bucket_shards)

            # 将 lookahead bucket 的参数分片并入预取列表。
            for future_bucket in lookahead_buckets:
                # 追加未来 bucket 对应的参数分片。
                prefetch_shards.extend(bucket_shard_map[future_bucket.bucket_id])

            # 对当前 bucket 以及 lookahead bucket 的参数分片执行统一预取。
            parameter_store.prefetch_shards(
                step_index=step_index,
                parameter_shards=tuple(prefetch_shards),
            )

            # 抽取仅属于当前 bucket 的预取摘要结果。
            bucket_prefetch_summary = parameter_store.prefetch_summary_for_groups(
                bucket_group_ids
            )

            # 为当前 bucket 的权重访问申请或复用权重传输缓冲区。
            transport_runtime.stage_weight_buffers(
                step_index=step_index,
                bucket_id=bucket.bucket_id,
                parameter_shards=bucket_shards,
            )

            # 将当前 bucket 的计算视图提前 stage 到执行设备。
            parameter_store.stage_compute_views(
                step_index=step_index,
                parameter_shards=bucket_shards,
                device=executor.compute_device,
            )

            # ------------------------------- 初始化当前 bucket 的 micro-batch 聚合容器 -------------------------------
            # 收集当前 bucket 内各 micro-batch 执行结果的列表。
            micro_bucket_records = []

            # 记录当前 bucket 在 host 侧梯度缓冲的峰值字节数。
            bucket_host_gradient_bytes = 0

            # 记录当前 bucket 内产生的梯度 payload 数量。
            bucket_gradient_payloads = 0

            # 记录当前 bucket 内实际被更新的 group_id。
            bucket_update_group_ids: list[str] = []

            # 当前 bucket 需要在优化后继续保留驻留的 group_id，初始为空。
            keep_resident_group_ids = ()

            # 当启用 active routed window 热保留时，提取当前 bucket 的 active routed group。
            if self.config.expert_rotation.retain_active_window_state_in_memory:
                # 抽取属于 bucket_active_experts 组件的 group_id，作为热驻留保留集合。
                keep_resident_group_ids = tuple(
                    shard.group_id
                    for shard in bucket_shards
                    if shard.component == "bucket_active_experts"
                )

                # 将当前 bucket 的保留 group 并入窗口级热驻留集合。
                window_hot_routed_group_ids.extend(keep_resident_group_ids)

            # ------------------------------- 逐个 micro-batch 执行当前 bucket 的前反向与优化更新 -------------------------------
            # 遍历当前 bucket 的 micro-batch 序列。
            for micro_batch_id, micro_batch in enumerate(micro_batches):
                # 执行当前 micro-batch 对应的 bucket 前向、反向与梯度生成。
                bucket_result = executor.execute_bucket(
                    step_index=step_index,
                    batch=micro_batch,
                    bucket=bucket,
                    parameter_shards=parameter_shards,
                    parameter_store=parameter_store,
                )

                # 统计当前 micro-batch 生成的梯度 payload 数量。
                gradient_count = len(bucket_result.gradients)

                # 将当前 micro-batch 的梯度 payload 数量累加到 step 级计数。
                gradient_payload_count += gradient_count

                # 将当前 micro-batch 的梯度 payload 数量累加到 bucket 级计数。
                bucket_gradient_payloads += gradient_count

                # 保存当前 micro-batch 的 bucket 执行记录。
                micro_bucket_records.append(bucket_result.bucket_record)

                # 为当前 micro-batch 的梯度传输申请或复用梯度缓冲区。
                transport_runtime.stage_gradient_buffers(
                    step_index=step_index,
                    bucket_id=bucket.bucket_id,
                    micro_batch_id=micro_batch_id,
                    gradient_payloads=bucket_result.gradients,
                )

                # 将当前 micro-batch 的梯度应用到优化器更新路径中。
                bucket_optimizer_result = optimizer.apply_gradients(
                    step_index=step_index,
                    parameter_shards=bucket_shards,
                    parameter_store=parameter_store,
                    gradient_payloads=bucket_result.gradients,
                    keep_resident_group_ids=keep_resident_group_ids,
                )

                # 将当前 micro-batch 的优化器更新记录并入 step 级更新列表。
                optimizer_updates.extend(bucket_optimizer_result.updates)

                # 记录当前 micro-batch 实际更新过的 group_id。
                bucket_update_group_ids.extend(
                    update.group_id for update in bucket_optimizer_result.updates
                )

                # 记录最近一次优化器汇总结果，后续作为 step 级汇总的来源。
                optimizer_summary = bucket_optimizer_result.optimizer_summary

                # 维护当前 bucket 所需 host 侧梯度缓冲区的峰值字节数。
                bucket_host_gradient_bytes = max(
                    bucket_host_gradient_bytes,
                    bucket_optimizer_result.optimizer_summary.last_bucket_staged_gradient_bytes,
                )

                # 当前 micro-batch 更新完成后，立即释放对应的梯度传输缓冲区。
                transport_runtime.release_buffers(
                    buffer_kind="gradient_stage",
                    owner_group_ids=tuple(
                        payload.group_id for payload in bucket_result.gradients
                    ),
                )

            # ------------------------------- 汇总当前 bucket 的加载结果并释放权重缓冲区 -------------------------------
            # 读取当前 bucket 对应 group 的加载路径汇总结果。
            bucket_load_summary = parameter_store.load_summary_for_groups(
                bucket_group_ids
            )

            # 当前 bucket 执行结束后，释放其对应的权重传输缓冲区。
            transport_runtime.release_buffers(
                buffer_kind="weight_stage",
                owner_group_ids=bucket_group_ids,
            )

            # 当前 bucket 至少应产生一条 micro-batch 执行记录。
            assert micro_bucket_records

            # ------------------------------- 将多个 micro-batch 结果聚合为单个 bucket 级执行记录 -------------------------------
            # 构造当前 bucket 聚合后的代表性执行记录。
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

            # 将聚合后的 bucket 执行记录保存到 step 级列表中。
            bucket_records.append(aggregated_bucket_record)

            # 统计当前 bucket 更新结束后，相关 group 仍位于 CPU hot 与已落冷的分片数量。
            cpu_hot_after_update, offloaded_after_update = (
                parameter_store.resident_tier_counts_for_groups(bucket_group_ids)
            )

            # ------------------------------- 生成当前 bucket 的流级 trace 记录 -------------------------------
            # 构造当前 bucket 的双流执行轨迹摘要。
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

        # ------------------------------- 在 step 结束时统一结算 routed window 的保留与回收策略 -------------------------------
        # 根据当前窗口、历史退休窗口以及下一步复用关系，结算 routed window 的保留与回收结果。
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

        # 当本轮确实执行了显式回收动作时，重新读取 offload 后的优化器汇总结果。
        if offload_performed:
            # 重新生成 offload 后的优化器汇总。
            optimizer_summary = optimizer.summary()

        # 当前 step 至少应产生一份优化器汇总结果。
        assert optimizer_summary is not None

        # ------------------------------- 聚合当前 step 的整体执行统计结果 -------------------------------
        # 构造当前 step 的整体执行摘要。
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

        # ------------------------------- 返回当前 step 的执行结果、优化器结果、bucket trace 与退休窗口集合 -------------------------------
        # 返回占位形式的执行结果、优化器结果、bucket 级 trace 以及下一轮退休窗口 group 集合。
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
        # ------------------------------- 为纯预测路径构造一套独立运行时组件 -------------------------------
        # 创建独立的参数驻留控制器，避免污染真实执行状态。
        controller = ParameterResidencyController(self.config)

        # 创建独立的参数仓库运行时。
        warehouse = ParameterWarehouse(self.config)

        # 创建独立的参数分片存储运行时。
        parameter_store = ParameterShardStore(self.config)

        # 创建独立的权重传输规划器。
        transport_planner = WeightTransportPlanner(self.config)

        # 创建独立的权重传输执行运行时。
        transport_runtime = WeightTransportRuntime(self.config)

        # 创建独立的 bucket 执行器。
        executor = RepresentativeBucketExecutor(self.config)

        # 创建独立的 CPU 优化器运行时。
        optimizer = CPUOptimizerRuntime(self.config)

        # ------------------------------- 初始化预测路径需要回传的结果占位变量 -------------------------------
        # 当前预测步的驻留规划结果，占位为空。
        residency_plan: ResidencyPlanResult | None = None

        # 当前预测步的参数仓库执行结果，占位为空。
        warehouse_result: WarehouseStepResult | None = None

        # 当前预测步的传输规划汇总，占位为空。
        transport_summary: TransportPlanSummary | None = None

        # 当前预测步的传输执行汇总，占位为空。
        transport_execution_summary: TransportExecutionSummary | None = None

        # 当前预测步的参数预取摘要，占位为空。
        parameter_prefetch_summary: ParameterPrefetchSummary | None = None

        # 当前预测步的执行结果，占位为空。
        execution_result: RepresentativeExecutionResult | None = None

        # 当前预测步的优化器结果，占位为空。
        optimizer_result: OptimizerStepResult | None = None

        # 当前预测步的 bucket 流 trace，初始为空元组。
        bucket_stream_traces: tuple[BucketStreamTrace, ...] = ()

        # 当前预测路径内累积的退休窗口 group_id 集合，初始为空。
        predicted_retired_window_group_ids: tuple[str, ...] = ()

        # ------------------------------- 从 step 0 顺序重放到目标 step_index 以预测状态演进 -------------------------------
        # 从第 0 步开始顺序重放到目标 step，逐步推演预测路径的状态。
        for predicted_step in range(step_index + 1):
            # 当未到目标预测步时，默认将当前 batch 作为下一步 batch；仅在最后一步使用调用方传入的 next_batch。
            predicted_next_batch = batch if predicted_step < step_index else next_batch

            # 计算在当前预测步之前累计已经处理的样本数。
            predicted_cumulative_samples = predicted_step * batch.samples

            # 计算在当前预测步之前累计已经处理的 token 数。
            predicted_cumulative_tokens = predicted_step * batch.total_tokens

            # 基于累计进度与 batch 信息规划当前预测步的 expert window。
            predicted_window = self._rotation.plan_window(
                step_index=predicted_step,
                batch=batch,
                layer_buckets=self._layer_buckets,
                next_batch=predicted_next_batch,
                cumulative_samples_processed=predicted_cumulative_samples,
                cumulative_tokens_processed=predicted_cumulative_tokens,
            )

            # 提取当前预测步的 active expert 集合。
            predicted_active = predicted_window.active_expert_ids

            # 提取当前预测步的预取 expert 集合。
            predicted_prefetch = predicted_window.prefetched_expert_ids

            # 默认下一步 active expert 集合为空。
            predicted_next_active: tuple[int, ...] = ()

            # ------------------------------- 在需要保留 active window 时额外预测下一步窗口 -------------------------------
            # 当启用 active routed window 热保留且存在下一步 batch 时，额外预测下一步 active experts。
            if (
                self.config.expert_rotation.retain_active_window_state_in_memory
                and predicted_next_batch is not None
            ):
                # 规划下一预测步的 expert window。
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

                # 提取下一预测步的 active expert 集合。
                predicted_next_active = predicted_next_window.active_expert_ids

            # ------------------------------- 决定当前预测步是否需要 stage static modules -------------------------------
            # 默认仅在第 0 个预测步自动 stage 静态模块。
            predicted_stage_static = predicted_step == 0

            # 当命中目标预测步时，优先采用调用方显式传入的 stage_static_modules 取值。
            if predicted_step == step_index:
                # 覆盖目标预测步的静态模块 stage 决策。
                predicted_stage_static = stage_static_modules

            # ------------------------------- 生成驻留计划并将其应用到参数仓库 -------------------------------
            # 为当前预测步生成参数驻留迁移计划，并同步更新预测控制器内部状态。
            residency_plan = controller.plan_step(
                step_index=predicted_step,
                layer_buckets=self._layer_buckets,
                active_expert_ids=predicted_active,
                prefetched_expert_ids=predicted_prefetch,
                memory_plan=memory_plan,
                stage_static_modules=predicted_stage_static,
                update_state=True,
            )

            # 将当前预测步的驻留迁移应用到预测参数仓库中。
            warehouse_result = warehouse.apply_residency_plan(
                step_index=predicted_step,
                transitions=residency_plan.transitions,
                memory_plan=memory_plan,
            )

            # ------------------------------- 生成并执行当前预测步的传输计划 -------------------------------
            # 基于当前预测步触达的参数分片生成传输规划。
            transport_summary = transport_planner.plan_step(
                warehouse_result.touched_shards
            )

            # 执行当前预测步的文件缓存与传输 stage。
            transport_execution_summary = transport_runtime.execute_step(
                step_index=predicted_step,
                transport_plan=transport_summary,
            )

            # 将 transport cache 的文件命中上下文同步到参数存储。
            parameter_store.set_transport_cache_context(
                step_index=predicted_step,
                cached_file_names=transport_runtime.cached_file_names(),
            )

            # ------------------------------- 执行当前预测步的 bucket 流 -------------------------------
            # 执行当前预测步的 bucket 流，并得到执行结果、优化器结果、bucket trace 与下一轮退休窗口集合。
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

            # 读取当前预测步最终的 transport 执行摘要。
            transport_execution_summary = transport_runtime.step_summary()

            # 读取当前预测步最终的参数预取摘要。
            parameter_prefetch_summary = parameter_store.step_prefetch_summary()

        # ------------------------------- 断言预测路径所需结果均已填充 -------------------------------
        # 断言最终驻留计划结果已经生成。
        assert residency_plan is not None

        # 断言最终参数仓库执行结果已经生成。
        assert warehouse_result is not None

        # 断言最终传输规划摘要已经生成。
        assert transport_summary is not None

        # 断言最终传输执行摘要已经生成。
        assert transport_execution_summary is not None

        # 断言最终参数预取摘要已经生成。
        assert parameter_prefetch_summary is not None

        # 断言最终执行结果已经生成。
        assert execution_result is not None

        # 断言最终优化器结果已经生成。
        assert optimizer_result is not None

        # ------------------------------- 返回目标预测步对应的完整预测结果 -------------------------------
        # 返回预测步的驻留规划、仓库结果、参数存储摘要、来源摘要、预取摘要、加载摘要、传输摘要、执行摘要与 bucket trace。
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
        # ------------------------------- 基于 batch 与 bucket trace 构造双流时间线 -------------------------------
        # 先将当前 batch 切分为 micro-batch 序列。
        micro_batches = self._micro_batch_planner.plan(batch)

        # 基于 micro-batch 序列与 bucket 流 trace 规划双流时间线与重叠统计。
        overlap_summary, stream_operations = self._timeline_planner.plan(
            micro_batches=micro_batches,
            bucket_stream_traces=bucket_stream_traces,
        )

        # 返回调度后的 micro-batch 序列、重叠汇总与流操作明细。
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
        # ------------------------------- 先构造 micro-batch 调度与双流时间线信息 -------------------------------
        # 基于当前 batch 与 bucket 流 trace 生成 micro-batch 调度、双流重叠汇总与流操作明细。
        (
            scheduled_micro_batches,
            stream_overlap_summary,
            stream_operations,
        ) = self._build_stream_schedule(
            batch=batch,
            bucket_stream_traces=bucket_stream_traces,
        )

        # ------------------------------- 组装当前 step 的完整 TrainingStepTrace -------------------------------
        # 将当前 step 的全部执行快照拼装为 TrainingStepTrace 对象并返回。
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
        # ------------------------------- 初始化当前 step 的高层运行时动作列表 -------------------------------
        # 用于累积当前 step 所对应的高层运行时动作。
        actions: list[RuntimeAction] = []

        # ------------------------------- 在首次执行时追加静态模块 stage 动作 -------------------------------
        # 当当前 step 需要首次 stage 静态模块时，追加对应的高层动作描述。
        if stage_static_modules:
            # 追加静态模块 stage 动作。
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

        # ------------------------------- 追加 routed expert 预取与 active window 切换动作 -------------------------------
        # 无论是否首次执行，都需要为 routed experts 追加预取与 active window 切换动作。
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

        # ------------------------------- 为每个 bucket 追加 forward、backward 与 update-release 动作 -------------------------------
        # 遍历所有 layer bucket，为每个 bucket 构造三类高层动作。
        for bucket in self._layer_buckets:
            # 构造当前 bucket 的公共 metadata。
            bucket_metadata = {
                "bucket_id": bucket.bucket_id,
                "layer_indices": list(bucket.layer_indices),
                "attention_types": list(bucket.attention_types),
                "active_expert_ids": list(active_experts),
                "compute_device": self.config.execution.compute_device,
            }

            # 追加当前 bucket 的 forward 动作。
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

            # 追加当前 bucket 的 backward 动作。
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

            # 追加当前 bucket 的 CPU 更新与即时释放动作。
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

        # ------------------------------- 追加 expert window 轮转动作 -------------------------------
        # 在当前 step 的末尾追加 expert window 轮转动作。
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

        # ------------------------------- 返回当前 step 的动作清单 -------------------------------
        # 返回当前 step 构造出的高层动作元组。
        return tuple(actions)

    def plan_step(
        self,
        *,
        step_index: int,
        batch: BatchShape,
        next_batch: BatchShape | None = None,
        stage_static_modules: bool | None = None,
    ) -> TrainingStepTrace:
        # ------------------------------- 校验 step 参数并补全默认 next_batch -------------------------------
        # step 序号不允许为负数。
        if step_index < 0:
            # 抛出非法 step 序号异常。
            raise ValueError("step_index must be >= 0")

        # 当未提供 next_batch 时，默认使用当前 batch 作为下一步的预测 batch。
        if next_batch is None:
            # 将当前 batch 作为默认 next_batch。
            next_batch = batch

        # ------------------------------- 规划当前 step 的 expert window 与 batch 级预算 -------------------------------
        # 基于 step 序号、batch 信息与累计进度预测当前 step 的 expert window。
        expert_window_plan = self._rotation.plan_window(
            step_index=step_index,
            batch=batch,
            layer_buckets=self._layer_buckets,
            next_batch=next_batch,
            cumulative_samples_processed=step_index * batch.samples,
            cumulative_tokens_processed=step_index * batch.total_tokens,
        )

        # 提取当前 step 的 active expert 集合。
        active_experts = expert_window_plan.active_expert_ids

        # 提取当前 step 的预取 expert 集合。
        prefetched_experts = expert_window_plan.prefetched_expert_ids

        # 当未显式指定时，仅在 step 0 执行静态模块 stage。
        if stage_static_modules is None:
            # 默认仅第 0 步需要 stage 静态模块。
            stage_static_modules = step_index == 0

        # 基于当前 batch 构造内存预算规划结果。
        memory_plan = self.build_memory_plan(batch)

        # ------------------------------- 走纯预测路径生成不污染真实状态的 step 结果 -------------------------------
        # 调用预测路径，生成完整的 step 级结果而不修改真实引擎状态。
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

        # ------------------------------- 构造高层动作清单并返回 step trace -------------------------------
        # 构造当前 step 的高层运行时动作列表。
        actions = self._build_actions(
            step_index=step_index,
            expert_window_plan=expert_window_plan,
            active_experts=active_experts,
            prefetched_experts=prefetched_experts,
            stage_static_modules=stage_static_modules,
            batch=batch,
        )

        # 将当前预测结果拼装成完整的 step trace 并返回。
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
        # ------------------------------- 读取真实执行路径下的 step 序号与 batch 预算 -------------------------------
        # 读取当前真实执行路径下的 step 编号。
        step_index = self._state.next_step_index

        # 基于当前 batch 生成内存预算规划结果。
        memory_plan = self.build_memory_plan(batch)

        # 保存调用方原始传入的 next_batch，用于后续判断是否需要显式预测下一窗口。
        provided_next_batch = next_batch

        # 当未显式提供 next_batch 时，默认按当前 batch 继续执行。
        if next_batch is None:
            # 将当前 batch 作为默认 next_batch。
            next_batch = batch

        # ------------------------------- 规划当前 step 与可能的下一步 expert window -------------------------------
        # 基于真实累计进度规划当前 step 的 expert window。
        expert_window_plan = self._rotation.plan_window(
            step_index=step_index,
            batch=batch,
            layer_buckets=self._layer_buckets,
            next_batch=next_batch,
            cumulative_samples_processed=self._state.cumulative_samples_processed,
            cumulative_tokens_processed=self._state.cumulative_tokens_processed,
        )

        # 默认下一步 active expert 集合为空。
        next_active_experts: tuple[int, ...] = ()

        # 当启用 active window 热保留且调用方显式提供了 next_batch 时，额外预测下一步 active experts。
        if (
            self.config.expert_rotation.retain_active_window_state_in_memory
            and provided_next_batch is not None
        ):
            # 规划下一 step 的 expert window。
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

            # 提取下一 step 的 active expert 集合。
            next_active_experts = next_expert_window_plan.active_expert_ids

        # 提取本 step 的 active expert 集合。
        active_experts = expert_window_plan.active_expert_ids

        # 提取本 step 的预取 expert 集合。
        prefetched_experts = expert_window_plan.prefetched_expert_ids

        # 仅当静态模块尚未 stage 过时，本 step 需要执行首次 stage。
        stage_static_modules = not self._state.static_modules_staged

        # ------------------------------- 生成驻留迁移计划并将其应用到真实仓库 -------------------------------
        # 基于当前 step 的 expert window 与内存预算生成驻留迁移计划。
        residency_plan = self._residency_controller.plan_step(
            step_index=step_index,
            layer_buckets=self._layer_buckets,
            active_expert_ids=active_experts,
            prefetched_expert_ids=prefetched_experts,
            memory_plan=memory_plan,
            stage_static_modules=stage_static_modules,
            update_state=True,
        )

        # 将驻留迁移计划应用到真实参数仓库中。
        warehouse_result = self._warehouse.apply_residency_plan(
            step_index=step_index,
            transitions=residency_plan.transitions,
            memory_plan=memory_plan,
        )

        # ------------------------------- 生成并执行真实传输计划 -------------------------------
        # 基于本 step 触达的参数分片生成传输计划。
        transport_summary = self._transport_planner.plan_step(
            warehouse_result.touched_shards
        )

        # 执行本 step 的真实传输过程。
        transport_execution_summary = self._transport_runtime.execute_step(
            step_index=step_index,
            transport_plan=transport_summary,
        )

        # 将 transport cache 的命中文件上下文同步到真实参数存储中。
        self._parameter_store.set_transport_cache_context(
            step_index=step_index,
            cached_file_names=self._transport_runtime.cached_file_names(),
        )

        # ------------------------------- 执行真实 bucket 流并完成优化器更新 -------------------------------
        # 执行 bucket 流、优化器更新以及 bucket 级 trace 聚合。
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

        # 读取真实传输执行完成后的最终 transport 执行摘要。
        transport_execution_summary = self._transport_runtime.step_summary()

        # 读取真实参数存储执行完成后的最终参数预取摘要。
        parameter_prefetch_summary = self._parameter_store.step_prefetch_summary()

        # ------------------------------- 构造高层动作清单并拼装真实执行 trace -------------------------------
        # 构造本 step 的高层动作清单。
        actions = self._build_actions(
            batch=batch,
            step_index=step_index,
            expert_window_plan=expert_window_plan,
            active_experts=active_experts,
            prefetched_experts=prefetched_experts,
            stage_static_modules=stage_static_modules,
        )

        # 将当前真实执行结果拼装为 TrainingStepTrace。
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

        # ------------------------------- 推进真实运行时状态计数器 -------------------------------
        # 标记静态模块已经完成首次 stage。
        self._state.static_modules_staged = True

        # 推进下一个待执行 step 序号。
        self._state.next_step_index += 1

        # 累加已经处理的样本数。
        self._state.cumulative_samples_processed += batch.samples

        # 累加已经处理的 token 数。
        self._state.cumulative_tokens_processed += batch.total_tokens

        # ------------------------------- 返回真实执行得到的 step trace -------------------------------
        # 返回当前 step 的真实执行轨迹。
        return trace

    def snapshot_state(self) -> TrainingRuntimeSnapshot:
        # ------------------------------- 读取参数存储汇总并组装当前引擎运行时快照 -------------------------------
        # 读取一次参数存储汇总，用于写入累计量化与累计 NVMe 同步计数。
        parameter_store_summary = self._parameter_store.summary()

        # 返回当前引擎关键运行时状态组成的快照对象。
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
        # ------------------------------- 校验快照所属 profile 与当前引擎配置一致 -------------------------------
        # 当快照中的 profile 名称与当前引擎配置不一致时，拒绝加载该快照。
        if snapshot.profile_name != self.config.profile_name:
            # 抛出 profile 不匹配异常。
            raise ValueError(
                "snapshot.profile_name must match engine config.profile_name"
            )

        # ------------------------------- 在量化会话切换时重建相关运行时组件 -------------------------------
        # 当快照中的量化会话 id 与当前配置不同且有效时，切换配置会话并重建依赖该会话的运行时。
        if (
            snapshot.runtime_quantization_session_id
            and snapshot.runtime_quantization_session_id
            != self.config.runtime_quantization.session_id
        ):
            # 将当前配置中的量化会话 id 更新为快照中的会话 id。
            self.config.runtime_quantization.session_id = (
                snapshot.runtime_quantization_session_id
            )

            # 基于新量化会话重建参数分片存储运行时。
            self._parameter_store = ParameterShardStore(self.config)

            # 基于新量化会话重建 CPU 优化器运行时。
            self._optimizer = CPUOptimizerRuntime(self.config)

        # ------------------------------- 恢复引擎级运行时计数器与窗口状态 -------------------------------
        # 使用快照中的计数器与窗口状态重建内部运行时状态对象。
        self._state = _TrainingRuntimeState(
            next_step_index=snapshot.next_step_index,
            static_modules_staged=snapshot.static_modules_staged,
            cumulative_samples_processed=snapshot.cumulative_samples_processed,
            cumulative_tokens_processed=snapshot.cumulative_tokens_processed,
            retired_window_group_ids=snapshot.retired_window_group_ids,
        )

        # 恢复参数驻留控制器内部状态。
        self._residency_controller.load_state(
            static_modules_state=snapshot.residency_static_modules_state,
            staged_expert_ids=snapshot.residency_staged_expert_ids,
        )

        # 恢复 expert window 轮换缓存。
        self._rotation.load_window_cache(snapshot.rotation_window_cache)

        # ------------------------------- 恢复仓库、参数存储、传输缓存与优化器状态 -------------------------------
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

        # 恢复 transport 文件缓存快照。
        self._transport_runtime.load_snapshot(snapshot.transport_cached_files)

        # 恢复 transport 缓冲区池快照。
        self._transport_runtime.load_buffer_snapshot(snapshot.transport_buffers)

        # 恢复优化器分片状态。
        self._optimizer.load_snapshot(snapshot.optimizer_shards)

    def simulate(
        self,
        *,
        steps: int,
        batch: BatchShape,
    ) -> TrainingRunTrace:
        # ------------------------------- 校验模拟步数参数 -------------------------------
        # 模拟步数至少必须为 1。
        if steps < 1:
            # 抛出非法模拟步数异常。
            raise ValueError("steps must be >= 1")

        # ------------------------------- 顺序执行指定步数并构造整段运行轨迹 -------------------------------
        # 顺序执行指定数量的 step，并仅在最后一步将 next_batch 置为空。
        traces = tuple(
            self.run_step(batch, next_batch=batch if index + 1 < steps else None)
            for index in range(steps)
        )

        # ------------------------------- 返回整段模拟运行的聚合 trace -------------------------------
        # 返回当前 profile、batch 形状、步数、逐步 trace 与资源规划摘要组成的运行轨迹。
        return TrainingRunTrace(
            profile_name=self.config.profile_name,
            batch=batch,
            step_count=steps,
            steps=traces,
            resource_plan=self.build_memory_plan(batch).to_dict(),
        )
