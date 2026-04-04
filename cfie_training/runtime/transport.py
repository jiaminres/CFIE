"""Weight transport planning and execution backed by the local safetensors manifest."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING

import torch

from cfie_training.config import TrainingProjectConfig
from cfie_training.runtime.quantization import runtime_device_weight_bytes_per_param
from cfie_training.runtime.source import LocalWeightManifest
from cfie_training.runtime.types import (
    ParameterShardSnapshot,
    TransportBufferSnapshot,
    TransportCachedFileSnapshot,
    TransportExecutionSummary,
    TransportPlanSummary,
    TransportShardPlan,
)

if TYPE_CHECKING:
    from cfie_training.runtime.executor import GradientPayload

_GIB = 1024**3
_TRAINABLE_COMPONENTS = frozenset({"bucket_non_routed", "bucket_active_experts"})


# 将 dtype 名称映射为传输缓冲区每元素字节数。
def _dtype_bytes(dtype_name: str) -> int:
    # FP32 每个元素占 4 字节。
    if dtype_name == "fp32":
        return 4
    # FP16 / BF16 每个元素占 2 字节。
    if dtype_name in {"fp16", "bf16"}:
        return 2
    # FP8 每个元素占 1 字节。
    if dtype_name in {"fp8_e4m3fn", "fp8_e5m2"}:
        return 1
    # 其他 dtype 当前不支持用于传输 staging。
    raise ValueError(f"unsupported dtype for transfer staging: {dtype_name}")


@dataclass(slots=True)
class WeightTransportPlanner:
    config: TrainingProjectConfig
    _manifest: LocalWeightManifest = field(init=False)

    # 初始化传输规划器并构造本地权重清单。
    def __post_init__(self) -> None:
        # 为后续分片传输规划加载本地权重清单。
        self._manifest = LocalWeightManifest(self.config)

    # 返回当前绑定的本地权重清单对象。
    @property
    def manifest(self) -> LocalWeightManifest:
        # 对外暴露规划器内部持有的 manifest 实例。
        return self._manifest

    # 为一组参数分片生成权重传输规划。
    def plan_step(
        self,
        parameter_shards: tuple[ParameterShardSnapshot, ...],
    ) -> TransportPlanSummary:
        # -----------------
        # 逐个分片匹配本地 manifest，并统计文件与张量需求。
        # 保存每个分片自己的传输计划。
        shard_plans: list[TransportShardPlan] = []
        # 统计成功命中 manifest 的分片数量。
        matched_shards = 0
        # 统计未命中 manifest 的分片数量。
        unmatched_shards = 0
        # 收集本 step 涉及到的去重文件名。
        unique_files: set[str] = set()
        # 累计本 step 需要访问的张量数量。
        tensor_count = 0
        # 累计估算的 stage 字节数。
        estimated_stage_bytes = 0
        # 读取设备侧每参数权重字节数估算规则。
        device_weight_bytes_per_param = runtime_device_weight_bytes_per_param(
            self.config
        )

        for shard in parameter_shards:
            # 先解析当前分片对应的权重来源。
            source = self._manifest.source_for_shard(shard)
            # 没命中任何来源时，记为 unmatched 并跳过。
            if not source.matched:
                unmatched_shards += 1
                continue
            # 代表性规划阶段最多只看 128 个参数。
            representative_params = max(1, min(shard.logical_params, 128))
            # 根据代表性参数规模规划参数来源布局。
            source_plan = self._manifest.plan_parameter_buffer_sources(
                shard,
                representative_params=representative_params,
            )
            # 当前分片已成功命中 manifest。
            matched_shards += 1
            # 取出当前分片实际触达的文件名集合。
            file_names = source_plan.used_file_names
            # 并入全局文件集合。
            unique_files.update(file_names)
            # 累加当前分片涉及的张量数。
            tensor_count += source_plan.used_tensor_count
            # 按逻辑参数量估算当前分片的 stage 字节数。
            shard_stage_bytes = math.ceil(
                shard.logical_params * device_weight_bytes_per_param
            )
            # 并入总的 stage 字节数。
            estimated_stage_bytes += shard_stage_bytes
            # 为当前分片写入独立的传输计划记录。
            shard_plans.append(
                TransportShardPlan(
                    group_id=shard.group_id,
                    component=shard.component,
                    file_names=file_names,
                    tensor_count=source_plan.used_tensor_count,
                    estimated_stage_bytes=shard_stage_bytes,
                )
            )

        # -----------------
        # 汇总当前 step 的整体传输规划结果。
        return TransportPlanSummary(
            manifest_available=self._manifest.available,
            matched_shards=matched_shards,
            unmatched_shards=unmatched_shards,
            file_count=len(unique_files),
            tensor_count=tensor_count,
            estimated_stage_bytes=estimated_stage_bytes,
            model_path=self._manifest.model_path,
            shard_plans=tuple(shard_plans),
        )


@dataclass(slots=True)
class _CachedTransportFileState:
    file_name: str
    file_size_bytes: int
    stage_count: int = 0
    reuse_count: int = 0
    last_used_step: int = -1

    # 将缓存文件状态转换为可持久化快照。
    def to_snapshot(self) -> TransportCachedFileSnapshot:
        # 直接把当前缓存文件状态打包成快照对象。
        return TransportCachedFileSnapshot(
            file_name=self.file_name,
            file_size_bytes=self.file_size_bytes,
            stage_count=self.stage_count,
            reuse_count=self.reuse_count,
            last_used_step=self.last_used_step,
        )


@dataclass(slots=True)
class _TransferBufferState:
    buffer_id: str
    buffer_kind: str
    owner_group_id: str
    capacity_bytes: int
    pinned: bool
    stage_count: int = 0
    reuse_count: int = 0
    last_used_step: int = -1
    last_bucket_id: int | None = None
    last_micro_batch_id: int | None = None
    active: bool = False

    # 将传输缓冲区状态转换为可持久化快照。
    def to_snapshot(self) -> TransportBufferSnapshot:
        # 直接把当前缓冲区状态打包成快照对象。
        return TransportBufferSnapshot(
            buffer_id=self.buffer_id,
            buffer_kind=self.buffer_kind,
            owner_group_id=self.owner_group_id,
            capacity_bytes=self.capacity_bytes,
            pinned=self.pinned,
            stage_count=self.stage_count,
            reuse_count=self.reuse_count,
            last_used_step=self.last_used_step,
            last_bucket_id=self.last_bucket_id,
            last_micro_batch_id=self.last_micro_batch_id,
            active=self.active,
        )


@dataclass(slots=True)
class WeightTransportRuntime:
    config: TrainingProjectConfig
    _planner: WeightTransportPlanner = field(init=False)
    _cached_files: dict[str, _CachedTransportFileState] = field(default_factory=dict)
    _buffer_pool: dict[str, _TransferBufferState] = field(default_factory=dict)
    _resident_bytes: int = 0
    _pinned_memory_supported: bool = field(init=False)
    _current_step_index: int = -1
    _current_file_summary: TransportExecutionSummary | None = None
    _step_h2d_transfer_bytes: int = 0
    _step_d2h_transfer_bytes: int = 0
    _step_overlap_eligible_bytes: int = 0
    _step_peak_weight_stage_bytes: int = 0
    _step_peak_gradient_stage_bytes: int = 0
    _step_released_buffer_count: int = 0

    # 初始化运行时传输器，并探测 pinned memory 能力。
    def __post_init__(self) -> None:
        # 先创建运行时使用的传输规划器。
        self._planner = WeightTransportPlanner(self.config)
        # 再探测当前环境是否支持 pinned host memory。
        self._pinned_memory_supported = self._detect_pinned_memory_support()

    # 返回文件缓存允许占用的最大字节数。
    @property
    def max_cache_bytes(self) -> int:
        # 把配置里的 GiB 预算换算成字节数。
        return int(self.config.transport.max_staged_file_cache_gb * _GIB)

    # 探测当前环境是否支持 pinned host memory。
    def _detect_pinned_memory_support(self) -> bool:
        try:
            # 直接尝试分配一小块 pinned memory。
            candidate = torch.empty(1, dtype=torch.uint8, pin_memory=True)
        except Exception:
            # 任意异常都视为当前环境不支持 pinned memory。
            return False
        # 根据张量是否被标记为 pinned 来确认结果。
        return bool(candidate.is_pinned())

    # 返回当前缓存中的文件名列表。
    def cached_file_names(self) -> tuple[str, ...]:
        # 按文件名字典序返回，保证输出稳定。
        return tuple(sorted(self._cached_files))

    # 在每个 step 开始前重置本步统计与缓冲区活跃标记。
    def _reset_step_metrics(self, step_index: int) -> None:
        # 记录当前正在执行的 step 序号。
        self._current_step_index = step_index
        # 清零本步 H2D 传输字节数。
        self._step_h2d_transfer_bytes = 0
        # 清零本步 D2H 传输字节数。
        self._step_d2h_transfer_bytes = 0
        # 清零本步可重叠的传输字节数。
        self._step_overlap_eligible_bytes = 0
        # 清零本步权重缓冲区峰值。
        self._step_peak_weight_stage_bytes = 0
        # 清零本步梯度缓冲区峰值。
        self._step_peak_gradient_stage_bytes = 0
        # 清零本步释放缓冲区次数。
        self._step_released_buffer_count = 0
        # 顺便把上一步遗留的文件摘要清空，后面 execute_step 会重建它。
        self._current_file_summary = None
        for state in self._buffer_pool.values():
            # 每步开始时，先把全部缓冲区标记为非活跃。
            state.active = False

    # 申请或复用一个传输缓冲区状态对象。
    def _acquire_buffer(
        self,
        *,
        buffer_kind: str,
        owner_group_id: str,
        capacity_bytes: int,
        step_index: int,
        bucket_id: int | None,
        micro_batch_id: int | None,
    ) -> _TransferBufferState:
        # -----------------
        # 先根据 buffer 类型和 owner 定位唯一缓冲区槽位。
        # 用缓冲区类型和 owner group 生成稳定 buffer id。
        buffer_id = f"{buffer_kind}:{owner_group_id}"
        # 优先查找已有缓冲区状态。
        state = self._buffer_pool.get(buffer_id)
        if state is None:
            # 首次使用时创建新缓冲区状态。
            state = _TransferBufferState(
                buffer_id=buffer_id,
                buffer_kind=buffer_kind,
                owner_group_id=owner_group_id,
                capacity_bytes=max(capacity_bytes, 1),
                pinned=self._pinned_memory_supported,
                stage_count=1,
            )
            # 把新缓冲区挂到缓冲池里。
            self._buffer_pool[buffer_id] = state
        else:
            # 命中已有缓冲区时累计 stage 次数。
            state.stage_count += 1
            # 本次需求更大时更新缓冲区容量。
            if state.capacity_bytes < capacity_bytes:
                state.capacity_bytes = capacity_bytes
            else:
                # 容量足够时记一次复用。
                state.reuse_count += 1
        # -----------------
        # 刷新当前缓冲区的最近使用元信息。
        # 记录最近使用的 step。
        state.last_used_step = step_index
        # 记录最近使用的 bucket。
        state.last_bucket_id = bucket_id
        # 记录最近使用的 micro-batch。
        state.last_micro_batch_id = micro_batch_id
        # 标记当前缓冲区已经被激活。
        state.active = True
        return state

    # 统计当前仍处于激活状态的某类缓冲区字节数。
    def _active_buffer_bytes(self, buffer_kind: str) -> int:
        # 只统计目标类型且 active=True 的缓冲区容量。
        return sum(
            state.capacity_bytes
            for state in self._buffer_pool.values()
            if state.buffer_kind == buffer_kind and state.active
        )

    # 为当前 bucket 的权重 stage 分配 / 复用缓冲区。
    def stage_weight_buffers(
        self,
        *,
        step_index: int,
        bucket_id: int,
        parameter_shards: tuple[ParameterShardSnapshot, ...],
    ) -> None:
        # 逐个参数分片申请或复用权重传输缓冲区。
        for shard in parameter_shards:
            # 只有可训练分片才需要为权重传输申请缓冲区。
            if shard.component not in _TRAINABLE_COMPONENTS:
                continue
            # 按逻辑参数规模估算当前分片需要的权重缓冲区字节数。
            bytes_required = math.ceil(
                shard.logical_params * runtime_device_weight_bytes_per_param(self.config)
            )
            # 为当前分片申请或复用权重 stage 缓冲区。
            self._acquire_buffer(
                buffer_kind="weight_stage",
                owner_group_id=shard.group_id,
                capacity_bytes=bytes_required,
                step_index=step_index,
                bucket_id=bucket_id,
                micro_batch_id=None,
            )
            # 本次权重传输计入 H2D 字节数。
            self._step_h2d_transfer_bytes += bytes_required
            # 允许重叠时，这部分字节也计入 overlap eligible。
            if self.config.execution.overlap_backward_and_update:
                self._step_overlap_eligible_bytes += bytes_required
        # 用当前活跃权重缓冲区总量更新峰值。
        self._step_peak_weight_stage_bytes = max(
            self._step_peak_weight_stage_bytes,
            self._active_buffer_bytes("weight_stage"),
        )

    # 为当前 micro-batch 的梯度回传分配 / 复用缓冲区。
    def stage_gradient_buffers(
        self,
        *,
        step_index: int,
        bucket_id: int,
        micro_batch_id: int,
        gradient_payloads: tuple["GradientPayload", ...],
    ) -> None:
        # 根据梯度缓冲区存储 dtype 计算每元素字节数。
        bytes_per_element = _dtype_bytes(
            self.config.optimizer.gradient_buffer_storage_dtype
        )
        # 逐个梯度 payload 申请或复用梯度回传缓冲区。
        for payload in gradient_payloads:
            # 当前梯度 payload 需要的缓冲区字节数由逻辑参数量决定。
            bytes_required = payload.logical_params * bytes_per_element
            # 为当前 payload 申请或复用梯度 stage 缓冲区。
            self._acquire_buffer(
                buffer_kind="gradient_stage",
                owner_group_id=payload.group_id,
                capacity_bytes=bytes_required,
                step_index=step_index,
                bucket_id=bucket_id,
                micro_batch_id=micro_batch_id,
            )
            # 本次梯度回传计入 D2H 字节数。
            self._step_d2h_transfer_bytes += bytes_required
            # 允许重叠时，这部分也算作可重叠传输。
            if self.config.execution.overlap_backward_and_update:
                self._step_overlap_eligible_bytes += bytes_required
        # 用当前活跃梯度缓冲区总量更新峰值。
        self._step_peak_gradient_stage_bytes = max(
            self._step_peak_gradient_stage_bytes,
            self._active_buffer_bytes("gradient_stage"),
        )

    # 释放指定 owner 对应的传输缓冲区活跃状态。
    def release_buffers(
        self,
        *,
        buffer_kind: str,
        owner_group_ids: tuple[str, ...],
    ) -> None:
        # 逐个 owner group 释放对应的缓冲区活跃标记。
        for group_id in owner_group_ids:
            # 用缓冲区类型和 group_id 组装 buffer id。
            buffer_id = f"{buffer_kind}:{group_id}"
            # 查询当前缓冲区状态。
            state = self._buffer_pool.get(buffer_id)
            # 不存在或已经非活跃时直接跳过。
            if state is None or not state.active:
                continue
            # 把缓冲区标记为已释放。
            state.active = False
            # 累计本步释放次数。
            self._step_released_buffer_count += 1

    # 汇总当前 step 的传输执行结果。
    def step_summary(self) -> TransportExecutionSummary:
        # 还没有文件缓存执行摘要时，返回空摘要对象。
        if self._current_file_summary is None:
            return TransportExecutionSummary(
                manifest_available=False,
                requested_file_count=0,
                staged_file_count=0,
                reused_file_count=0,
                evicted_file_count=0,
                cache_hit_shards=0,
                cache_miss_shards=0,
                staged_bytes=0,
                reused_bytes=0,
                evicted_bytes=0,
                cache_file_count=0,
                cache_resident_bytes=0,
                max_cache_bytes=self.max_cache_bytes,
                pinned_memory_supported=self._pinned_memory_supported,
            )
        # -----------------
        # 先汇总缓冲区池级别的统计数据。
        # 统计缓冲池中缓冲区总数。
        pooled_buffer_count = len(self._buffer_pool)
        # 统计缓冲池总字节数。
        pooled_buffer_bytes = sum(
            state.capacity_bytes for state in self._buffer_pool.values()
        )
        # 统计 pinned 缓冲区数量。
        pinned_buffer_count = sum(
            1 for state in self._buffer_pool.values() if state.pinned
        )
        # 统计 pinned 缓冲区总字节数。
        pinned_buffer_bytes = sum(
            state.capacity_bytes
            for state in self._buffer_pool.values()
            if state.pinned
        )
        # 统计当前仍然活跃的缓冲区数量。
        active_buffer_count = sum(
            1 for state in self._buffer_pool.values() if state.active
        )
        # 在文件缓存摘要的基础上，补上缓冲区相关统计。
        return TransportExecutionSummary(
            manifest_available=self._current_file_summary.manifest_available,
            requested_file_count=self._current_file_summary.requested_file_count,
            staged_file_count=self._current_file_summary.staged_file_count,
            reused_file_count=self._current_file_summary.reused_file_count,
            evicted_file_count=self._current_file_summary.evicted_file_count,
            cache_hit_shards=self._current_file_summary.cache_hit_shards,
            cache_miss_shards=self._current_file_summary.cache_miss_shards,
            staged_bytes=self._current_file_summary.staged_bytes,
            reused_bytes=self._current_file_summary.reused_bytes,
            evicted_bytes=self._current_file_summary.evicted_bytes,
            cache_file_count=self._current_file_summary.cache_file_count,
            cache_resident_bytes=self._current_file_summary.cache_resident_bytes,
            max_cache_bytes=self._current_file_summary.max_cache_bytes,
            pinned_memory_supported=self._pinned_memory_supported,
            active_buffer_count=active_buffer_count,
            pooled_buffer_count=pooled_buffer_count,
            pooled_buffer_bytes=pooled_buffer_bytes,
            pinned_buffer_count=pinned_buffer_count,
            pinned_buffer_bytes=pinned_buffer_bytes,
            weight_stage_buffer_bytes=self._step_peak_weight_stage_bytes,
            gradient_stage_buffer_bytes=self._step_peak_gradient_stage_bytes,
            h2d_transfer_bytes=self._step_h2d_transfer_bytes,
            d2h_transfer_bytes=self._step_d2h_transfer_bytes,
            overlap_eligible_bytes=self._step_overlap_eligible_bytes,
            released_buffer_count=self._step_released_buffer_count,
            staged_files=self._current_file_summary.staged_files,
            reused_files=self._current_file_summary.reused_files,
            evicted_files=self._current_file_summary.evicted_files,
        )

    # 在缓存预算内持续淘汰旧文件，直到为新文件腾出空间。
    def _evict_until_fit(
        self,
        *,
        required_bytes: int,
        protected_files: set[str],
        evicted_files: list[str],
    ) -> int:
        # 记录本轮为了腾挪空间总共驱逐了多少字节。
        evicted_bytes = 0
        # 只要“当前已驻留字节 + 本次需求”仍超预算，就持续驱逐。
        while self._resident_bytes + required_bytes > self.max_cache_bytes:
            # 只能驱逐不在 protected_files 里的文件。
            eviction_candidates = [
                state
                for state in self._cached_files.values()
                if state.file_name not in protected_files
            ]
            # 没有可驱逐候选时结束循环。
            if not eviction_candidates:
                break
            # 选择“最久未使用 + 文件名字典序”最小的候选作为受害者。
            victim = min(
                eviction_candidates,
                key=lambda state: (state.last_used_step, state.file_name),
            )
            # 把该文件字节数计入驱逐总量。
            evicted_bytes += victim.file_size_bytes
            # 同步扣减缓存已占用字节数。
            self._resident_bytes -= victim.file_size_bytes
            # 记录本次被驱逐的文件名。
            evicted_files.append(victim.file_name)
            # 从缓存表中删除该文件。
            del self._cached_files[victim.file_name]
        # 返回本轮为腾挪空间累计驱逐的字节数。
        return evicted_bytes

    # 按规划结果执行一次权重文件缓存与 stage 过程。
    def execute_step(
        self,
        *,
        step_index: int,
        transport_plan: TransportPlanSummary,
    ) -> TransportExecutionSummary:
        # -----------------
        # 先重置本步统计；manifest 不可用时直接返回空执行摘要。
        # 每步开始前先清零所有本步统计字段。
        self._reset_step_metrics(step_index)
        if not transport_plan.manifest_available:
            # manifest 不可用时，直接构造空缓存摘要并返回。
            self._current_file_summary = TransportExecutionSummary(
                manifest_available=transport_plan.manifest_available,
                requested_file_count=transport_plan.file_count,
                staged_file_count=0,
                reused_file_count=0,
                evicted_file_count=0,
                cache_hit_shards=0,
                cache_miss_shards=transport_plan.matched_shards,
                staged_bytes=0,
                reused_bytes=0,
                evicted_bytes=0,
                cache_file_count=0,
                cache_resident_bytes=0,
                max_cache_bytes=self.max_cache_bytes,
            )
            return self.step_summary()

        # -----------------
        # 如禁用跨步复用，则每步开始前清空文件缓存。
        if not self.config.transport.reuse_staged_files_across_steps:
            # 非复用模式下，从空缓存重新开始。
            self._cached_files = {}
            self._resident_bytes = 0

        # -----------------
        # 统计请求文件集合、需求热度以及分片级 cache hit / miss。
        preexisting_files = set(self._cached_files)
        requested_files: set[str] = set()
        file_demand_counts: dict[str, int] = {}
        cache_hit_shards = 0
        cache_miss_shards = 0
        for shard_plan in transport_plan.shard_plans:
            # 当前分片涉及的文件集合。
            shard_files = set(shard_plan.file_names)
            # 把这些文件并入当前 step 的请求集合。
            requested_files.update(shard_files)
            for file_name in shard_files:
                # 记录每个文件被多少个分片请求，用于排序。
                file_demand_counts[file_name] = file_demand_counts.get(file_name, 0) + 1
            # 若当前分片所有文件都已缓存，则记为 cache hit。
            if shard_files and shard_files.issubset(preexisting_files):
                cache_hit_shards += 1
            else:
                # 否则记为 cache miss。
                cache_miss_shards += 1

        # -----------------
        # 初始化 staged / reused / evicted 文件集合与字节计数。
        staged_files: list[str] = []
        reused_files: list[str] = []
        evicted_files: list[str] = []
        staged_bytes = 0
        reused_bytes = 0
        evicted_bytes = 0

        requested_files_in_order = sorted(
            requested_files,
            key=lambda file_name: (
                # 优先处理需求次数更高的文件。
                -file_demand_counts.get(file_name, 0),
                # 热度相同时优先小文件。
                self._planner.manifest.file_size_bytes(file_name),
                # 最后按文件名字典序稳定排序。
                file_name,
            ),
        )

        # -----------------
        # 按需求优先级处理每个文件：优先复用，必要时淘汰，再执行 stage。
        for file_name in requested_files_in_order:
            # 优先尝试直接命中文件缓存。
            cached = self._cached_files.get(file_name)
            if cached is not None:
                # 命中时累计复用次数。
                cached.reuse_count += 1
                # 更新最近使用 step。
                cached.last_used_step = step_index
                # 把文件名记到 reused 列表。
                reused_files.append(file_name)
                # 累加复用字节数。
                reused_bytes += cached.file_size_bytes
                continue

            # 读取当前文件的实际大小。
            file_size_bytes = self._planner.manifest.file_size_bytes(file_name)
            # 单文件超过缓存上限时无法缓存，直接跳过。
            if file_size_bytes > self.max_cache_bytes:
                continue
            # 先驱逐旧文件，尽量为当前文件腾出空间。
            evicted_bytes += self._evict_until_fit(
                required_bytes=file_size_bytes,
                protected_files=requested_files,
                evicted_files=evicted_files,
            )
            # 即使驱逐后仍放不下时，也只能跳过当前文件。
            if self._resident_bytes + file_size_bytes > self.max_cache_bytes:
                continue
            # 为当前文件创建新的缓存状态。
            cached = _CachedTransportFileState(
                file_name=file_name,
                file_size_bytes=file_size_bytes,
                stage_count=1,
                reuse_count=0,
                last_used_step=step_index,
            )
            # 把文件加入缓存表。
            self._cached_files[file_name] = cached
            # 累加当前缓存总占用字节数。
            self._resident_bytes += file_size_bytes
            # 把文件记到 staged 列表。
            staged_files.append(file_name)
            # 累加 stage 字节数。
            staged_bytes += file_size_bytes

        # -----------------
        # 生成文件缓存执行摘要，并在非复用模式下回收临时缓存。
        self._current_file_summary = TransportExecutionSummary(
            manifest_available=transport_plan.manifest_available,
            requested_file_count=len(requested_files),
            staged_file_count=len(staged_files),
            reused_file_count=len(reused_files),
            evicted_file_count=len(evicted_files),
            cache_hit_shards=cache_hit_shards,
            cache_miss_shards=cache_miss_shards,
            staged_bytes=staged_bytes,
            reused_bytes=reused_bytes,
            evicted_bytes=evicted_bytes,
            cache_file_count=len(self._cached_files),
            cache_resident_bytes=self._resident_bytes,
            max_cache_bytes=self.max_cache_bytes,
            staged_files=tuple(staged_files),
            reused_files=tuple(reused_files),
            evicted_files=tuple(evicted_files),
        )
        if not self.config.transport.reuse_staged_files_across_steps:
            # 非复用模式下，step 结束后立刻清空缓存。
            self._cached_files = {}
            self._resident_bytes = 0
        # 返回补齐缓冲区统计后的完整 step 摘要。
        return self.step_summary()

    # 导出当前文件缓存快照。
    def snapshot(self) -> tuple[TransportCachedFileSnapshot, ...]:
        # 按文件名排序导出缓存快照，保证输出稳定。
        return tuple(
            state.to_snapshot()
            for _, state in sorted(self._cached_files.items(), key=lambda item: item[0])
        )

    # 导出当前传输缓冲区池快照。
    def buffer_snapshot(self) -> tuple[TransportBufferSnapshot, ...]:
        # 按 buffer_id 排序导出缓冲区池快照。
        return tuple(
            state.to_snapshot()
            for _, state in sorted(self._buffer_pool.items(), key=lambda item: item[0])
        )

    # 从外部快照恢复文件缓存状态。
    def load_snapshot(
        self,
        snapshots: tuple[TransportCachedFileSnapshot, ...],
    ) -> None:
        # 用快照列表重建缓存文件状态表。
        self._cached_files = {
            snapshot.file_name: _CachedTransportFileState(
                file_name=snapshot.file_name,
                file_size_bytes=snapshot.file_size_bytes,
                stage_count=snapshot.stage_count,
                reuse_count=snapshot.reuse_count,
                last_used_step=snapshot.last_used_step,
            )
            for snapshot in snapshots
        }
        # 重新累计缓存文件总占用字节数。
        self._resident_bytes = sum(
            snapshot.file_size_bytes for snapshot in snapshots
        )
        # 快照恢复只重建文件缓存，不保留上一 step 的执行摘要。
        self._current_file_summary = None

    # 从外部快照恢复传输缓冲区池状态。
    def load_buffer_snapshot(
        self,
        snapshots: tuple[TransportBufferSnapshot, ...],
    ) -> None:
        # 用快照列表重建缓冲区池状态表。
        self._buffer_pool = {
            snapshot.buffer_id: _TransferBufferState(
                buffer_id=snapshot.buffer_id,
                buffer_kind=snapshot.buffer_kind,
                owner_group_id=snapshot.owner_group_id,
                capacity_bytes=snapshot.capacity_bytes,
                pinned=snapshot.pinned,
                stage_count=snapshot.stage_count,
                reuse_count=snapshot.reuse_count,
                last_used_step=snapshot.last_used_step,
                last_bucket_id=snapshot.last_bucket_id,
                last_micro_batch_id=snapshot.last_micro_batch_id,
                active=snapshot.active,
            )
            for snapshot in snapshots
        }
        # 缓冲区池恢复后，保留 active 标记，由下一步 reset 再统一清零。
