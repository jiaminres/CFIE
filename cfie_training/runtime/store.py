"""Representative parameter store for the CFIE training runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
import gc
import hashlib
import ctypes

import torch

from cfie_training.config import TrainingProjectConfig
from cfie_training.runtime.quantization import (
    FP32MasterMirror,
    GPTQTrainingQuantizer,
    PackedQuantizedTensor,
)
from cfie_training.runtime.source import LocalWeightManifest
from cfie_training.runtime.types import (
    ParameterLoadRecord,
    ParameterPrefetchSummary,
    ParameterLoadSummary,
    ParameterShardSnapshot,
    ParameterSourceSlice,
    ParameterSourceRecord,
    ParameterSourceSummary,
    ParameterStoreShardSnapshot,
    ParameterStoreSummary,
)

_MAX_REPRESENTATIVE_PARAMS = 128
_PREFETCHABLE_COMPONENTS = frozenset({"bucket_non_routed", "bucket_active_experts"})
_FULL_MATERIALIZATION_COMPONENTS = frozenset(
    {"bucket_non_routed", "bucket_active_experts"}
)


def _stable_seed(label: str) -> int:
    # 先对 group_id 等稳定标签做 SHA1，确保不同进程也能得到相同种子。
    digest = hashlib.sha1(label.encode("utf-8")).digest()
    # 取前 8 字节转成无符号整数，作为可复现的伪随机种子。
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _initial_parameter_buffer(group_id: str, size: int) -> tuple[float, ...]:
    # 先根据 group_id 生成稳定种子。
    seed = _stable_seed(group_id)
    # 把种子压到一个较小基值范围，避免初始化值过大。
    base = ((seed % 127) + 1) * 1e-3
    # 在基值上附加递增微扰，构造稳定且可区分的默认参数向量。
    return tuple(base + 1e-5 * index for index in range(size))


def _device_matches(lhs: torch.device, rhs: torch.device) -> bool:
    # 设备类型不同则一定不匹配。
    if lhs.type != rhs.type:
        return False
    # 非 CUDA 设备只要类型相同就视为匹配。
    if lhs.type != "cuda":
        return True
    # CUDA 设备还要检查 index；rhs 未指定 index 时视为兼容。
    return rhs.index is None or lhs.index == rhs.index


def _trim_host_memory() -> None:
    # 先触发 Python 层 GC，尽量回收无引用对象。
    gc.collect()
    try:
        # 通过 libc 的 malloc_trim 把空闲堆内存还给系统。
        libc = ctypes.CDLL("libc.so.6")
        # 某些环境可能没有导出 malloc_trim，需安全获取。
        trim = getattr(libc, "malloc_trim", None)
        # 能拿到 trim 时执行一次内存裁剪。
        if trim is not None:
            trim(0)
    except OSError:
        # 找不到 libc 时忽略裁剪失败，避免影响主流程。
        return


@dataclass(slots=True)
class _ParameterStoreShardState:
    group_id: str
    component: str
    logical_params: int
    representative_params: int
    bucket_id: int | None = None
    expert_ids: tuple[int, ...] = ()
    resident_tier: str = "nvme_cold"
    source_kind: str = "synthetic_seed"
    source_file_names: tuple[str, ...] = ()
    source_tensor_count: int = 0
    source_layout: tuple[ParameterSourceSlice, ...] = ()
    transport_backed: bool = False
    stage_count: int = 0
    offload_count: int = 0
    last_touched_step: int = -1
    parameter_buffer: tuple[float, ...] = ()
    parameter_buffer_tensor: torch.Tensor | None = None
    parameter_tensor: torch.Tensor | None = None
    quantized_parameter: PackedQuantizedTensor | None = None
    gpu_quantized_parameter: PackedQuantizedTensor | None = None
    gpu_parameter_tensor: torch.Tensor | None = None
    gpu_stage_count: int = 0
    gpu_release_count: int = 0

    def to_snapshot(
        self,
        parameter_values_override: tuple[float, ...] | None = None,
        *,
        omit_parameter_values: bool = False,
    ) -> ParameterStoreShardSnapshot:
        # -----------------
        # 先决定快照里是否携带参数值。
        # 调用方要求省略参数值时，直接写空元组。
        if omit_parameter_values:
            parameter_values = ()
        else:
            # 默认优先使用调用方提供的 override；否则回落到当前 buffer。
            parameter_values = (
                self.parameter_buffer
                if parameter_values_override is None
                else parameter_values_override
            )
            # 若当前已有可变 CPU 参数张量，则以它为准导出快照值。
            if self.parameter_tensor is not None:
                parameter_values = tuple(
                    float(value) for value in self.parameter_tensor.tolist()
                )
            # 否则若存在 CPU buffer tensor，也把它转成 float 元组落盘。
            elif self.parameter_buffer_tensor is not None:
                parameter_values = tuple(
                    float(value)
                    for value in self.parameter_buffer_tensor.to(dtype=torch.float32).tolist()
                )
        # -----------------
        # 把内部状态字段封装成对外快照对象。
        return ParameterStoreShardSnapshot(
            group_id=self.group_id,
            component=self.component,
            logical_params=self.logical_params,
            representative_params=self.representative_params,
            resident_tier=self.resident_tier,
            source_kind=self.source_kind,
            source_file_names=self.source_file_names,
            source_tensor_count=self.source_tensor_count,
            source_layout=self.source_layout,
            stage_count=self.stage_count,
            offload_count=self.offload_count,
            last_touched_step=self.last_touched_step,
            parameter_values=parameter_values,
            gpu_stage_count=self.gpu_stage_count,
            gpu_release_count=self.gpu_release_count,
        )


@dataclass(slots=True)
class ParameterShardStore:
    config: TrainingProjectConfig
    _states: dict[str, _ParameterStoreShardState] = field(default_factory=dict)
    _cumulative_stage_ops: int = 0
    _cumulative_offload_ops: int = 0
    _manifest: LocalWeightManifest = field(init=False)
    _transport_cache_context_step: int = -1
    _transport_cached_files: frozenset[str] = frozenset()
    _step_prefetch_step: int = -1
    _step_prefetch_records: dict[str, ParameterLoadRecord] = field(default_factory=dict)
    _step_load_step: int = -1
    _step_load_records: dict[str, ParameterLoadRecord] = field(default_factory=dict)
    _cumulative_gpu_stage_ops: int = 0
    _cumulative_gpu_release_ops: int = 0
    _quantizer: GPTQTrainingQuantizer = field(init=False)
    _fp32_mirror: FP32MasterMirror = field(init=False)
    _cumulative_quantize_ops: int = 0
    _cumulative_nvme_sync_ops: int = 0

    def __post_init__(self) -> None:
        # 初始化本地权重 manifest，用于按 shard 定位 safetensors 来源。
        self._manifest = LocalWeightManifest(self.config)
        # 初始化训练量化器，后面 GPU 计算视图会复用它。
        self._quantizer = GPTQTrainingQuantizer(self.config)
        # 初始化 FP32 主副本镜像，用于 NVMe 持久化与恢复。
        self._fp32_mirror = FP32MasterMirror(self.config)

    def _representative_params(self, logical_params: int) -> int:
        # 代表性参数个数至少为 1，且不会超过逻辑参数量和全局上限。
        return max(1, min(logical_params, _MAX_REPRESENTATIVE_PARAMS))

    def _logical_materialization_enabled(self) -> bool:
        # 只有 execution 配置显式为 logical 时，才启用逻辑物化模式。
        return self.config.execution.trainable_shard_materialization == "logical"

    def _full_bucket_logical_cuda_enabled(self) -> bool:
        # 仅在逻辑物化 + GPU 计算 + full_bucket 模式下返回 True。
        return (
            self._logical_materialization_enabled()
            and self.config.execution.compute_device == "gpu"
            and self.config.execution.logical_cuda_execution_mode == "full_bucket"
        )

    def _materialized_params(
        self,
        shard: ParameterShardSnapshot,
    ) -> int:
        # 逻辑物化模式下，指定组件直接按完整 logical_params 物化。
        if (
            self.config.execution.trainable_shard_materialization == "logical"
            and shard.component in _FULL_MATERIALIZATION_COMPONENTS
        ):
            return max(1, shard.logical_params)
        # 其余情况都退回代表性参数长度。
        return self._representative_params(shard.logical_params)

    def _get_or_create_state(
        self,
        shard: ParameterShardSnapshot,
    ) -> _ParameterStoreShardState:
        # 先尝试命中已有 shard 状态。
        state = self._states.get(shard.group_id)
        # 命中失败时按当前 shard 元信息创建新状态。
        if state is None:
            # 先计算该 shard 需要实际物化多少参数。
            representative_params = self._materialized_params(shard)
            # 查询 manifest，看当前 shard 是否能映射到真实权重。
            source = self._manifest.source_for_shard(shard)
            # 尝试从 FP32 NVMe 主副本中直接恢复参数。
            mirrored_values = self._fp32_mirror.load(shard.group_id)
            # source_plan 用于“只规划来源布局但不立刻搬数据”的场景。
            source_plan = None
            # 满足 logical + 目标组件时，当前 shard 会走全量物化路径。
            full_materialization = (
                self.config.execution.trainable_shard_materialization == "logical"
                and shard.component in _FULL_MATERIALIZATION_COMPONENTS
            )
            # 如果已有 NVMe 主副本，则允许 manifest 保持惰性，只保留布局信息。
            lazy_manifest_materialization = (
                mirrored_values is None and full_materialization and source.matched
            )
            # 已有镜像值时，只需规划来源，不必立刻读取 manifest 数据。
            if mirrored_values is not None:
                source_plan = (
                    self._manifest.plan_full_parameter_buffer_sources(shard)
                    if full_materialization
                    else self._manifest.plan_parameter_buffer_sources(
                        shard,
                        representative_params=representative_params,
                    )
                )
            # load_result 保存真正把数据从 manifest 读出来后的结果。
            load_result = None
            # 尚未生成 source_plan 但 manifest 可用时，补做一次纯规划。
            if source_plan is None and source.matched:
                source_plan = (
                    self._manifest.plan_full_parameter_buffer_sources(shard)
                    if full_materialization
                    else self._manifest.plan_parameter_buffer_sources(
                        shard,
                        representative_params=representative_params,
                    )
                )
            # 没有镜像值、manifest 可用且不走惰性模式时，直接把参数读出来。
            if (
                mirrored_values is None
                and source.matched
                and not lazy_manifest_materialization
            ):
                load_result = (
                    self._manifest.build_full_parameter_buffer(shard)
                    if full_materialization
                    else self._manifest.build_parameter_buffer(
                        shard,
                        representative_params=representative_params,
                    )
                )
            # preloaded_tensor 用于保存已经具备张量形态的预加载结果。
            preloaded_tensor = None
            # 走 NVMe 主副本全量恢复时，直接把镜像值转成 CPU float32 张量。
            if mirrored_values is not None and full_materialization:
                preloaded_tensor = mirrored_values.to(
                    dtype=torch.float32,
                    device="cpu",
                ).reshape(-1).contiguous()
            # manifest 直接读出的全量结果若已经是 tensor，也直接缓存起来。
            elif (
                load_result is not None
                and isinstance(load_result.values, torch.Tensor)
            ):
                preloaded_tensor = load_result.values.to(
                    dtype=torch.float32,
                    device="cpu",
                ).reshape(-1).contiguous()
            # materialized_params 记录当前状态中真实可用的参数长度。
            materialized_params = representative_params
            # 有预加载 tensor 时，物化长度直接等于 tensor 元素数。
            if preloaded_tensor is not None:
                materialized_params = max(1, int(preloaded_tensor.numel()))
            # 仅有 source_plan 且走全量物化时，用布局总长度估算参数量。
            elif source_plan is not None and full_materialization:
                materialized_params = max(
                    1,
                    sum(entry.length for entry in source_plan.source_layout),
                )
            # 代表性读取结果是 tuple 时，用其实际长度作为物化参数数。
            elif load_result is not None and not isinstance(load_result.values, torch.Tensor):
                materialized_params = max(1, len(load_result.values))
            # 基于上面得到的信息构造新的内部状态对象。
            state = _ParameterStoreShardState(
                group_id=shard.group_id,
                component=shard.component,
                logical_params=shard.logical_params,
                representative_params=materialized_params,
                bucket_id=shard.bucket_id,
                expert_ids=shard.expert_ids,
                source_kind=(
                    "nvme_fp32_mirror"
                    if mirrored_values is not None
                    else (
                        "local_manifest"
                        if source_plan is not None or load_result is not None
                        else "synthetic_seed"
                    )
                ),
                source_file_names=(
                    (self._fp32_mirror.path_for_group(shard.group_id).name,)
                    if mirrored_values is not None
                    else (
                        source_plan.used_file_names
                        if source_plan is not None
                        else (load_result.used_file_names if load_result is not None else ())
                    )
                ),
                source_tensor_count=(
                    1
                    if mirrored_values is not None
                    else (
                        source_plan.used_tensor_count
                        if source_plan is not None
                        else (load_result.used_tensor_count if load_result is not None else 0)
                    )
                ),
                source_layout=(
                    source_plan.source_layout
                    if source_plan is not None
                    else (load_result.source_layout if load_result is not None else ())
                ),
                parameter_buffer=(
                    ()
                    if preloaded_tensor is not None
                    else (
                        tuple(float(value) for value in mirrored_values.tolist())
                        if mirrored_values is not None
                        else (
                            ()
                            if lazy_manifest_materialization
                            else (
                                load_result.values
                                if load_result is not None
                                else _initial_parameter_buffer(
                                    shard.group_id,
                                    materialized_params,
                                )
                            )
                        )
                    )
                ),
                parameter_buffer_tensor=preloaded_tensor,
            )
            # 预加载张量已经在 CPU hot 态，因此补记一次 stage 操作。
            if preloaded_tensor is not None:
                state.resident_tier = "cpu_hot"
                state.stage_count = 1
                self._cumulative_stage_ops += 1
            # 把新状态注册进全局状态表。
            self._states[shard.group_id] = state
        # 返回命中或新建后的状态。
        return state

    def _refresh_transport_backing(
        self,
        state: _ParameterStoreShardState,
        *,
        step_index: int | None = None,
    ) -> None:
        # 只有 manifest 来源、上下文 step 匹配且存在来源文件时，才可能命中 transport cache。
        if (
            state.source_kind == "local_manifest"
            and (
                step_index is None
                or step_index == self._transport_cache_context_step
            )
            and state.source_file_names
        ):
            # 任一来源文件在缓存集合中命中，就把当前 shard 视为 transport_backed。
            state.transport_backed = any(
                file_name in self._transport_cached_files
                for file_name in state.source_file_names
            )
        else:
            # 其余情况统一标记为不依赖 transport cache。
            state.transport_backed = False

    def _ensure_prefetch_tracking(self, step_index: int) -> None:
        # 进入新 step 时重置 prefetch 统计容器。
        if self._step_prefetch_step != step_index:
            self._step_prefetch_step = step_index
            self._step_prefetch_records = {}

    def _ensure_step_tracking(self, step_index: int) -> None:
        # 进入新 step 时重置 load 统计容器。
        if self._step_load_step != step_index:
            self._step_load_step = step_index
            self._step_load_records = {}

    def _creation_load_path(self, state: _ParameterStoreShardState) -> str:
        # synthetic_seed 状态首次创建时记录为 synthetic_seed 路径。
        if state.source_kind == "synthetic_seed":
            return "synthetic_seed"
        # NVMe 主副本恢复的状态记录为 nvme_fp32_mirror 路径。
        if state.source_kind == "nvme_fp32_mirror":
            return "nvme_fp32_mirror"
        # 若当前来源文件已由 transport 预热缓存，则记为 transport_cache。
        if state.transport_backed:
            return "transport_cache"
        # 剩余 manifest 直读路径统一记为 direct_manifest。
        return "direct_manifest"

    def _record_step_access(
        self,
        *,
        step_index: int,
        state: _ParameterStoreShardState,
        load_path: str,
    ) -> None:
        # 先确保当前 step 的 load 统计表已初始化。
        self._ensure_step_tracking(step_index)
        # 同一 step 内同一个 group 只记录一次访问。
        if state.group_id in self._step_load_records:
            return
        # 追加一条参数加载记录，供后续 step summary 使用。
        self._step_load_records[state.group_id] = ParameterLoadRecord(
            group_id=state.group_id,
            component=state.component,
            source_kind=state.source_kind,
            load_path=load_path,
            resident_tier_after_access=state.resident_tier,
        )

    def _record_prefetch_access(
        self,
        *,
        step_index: int,
        state: _ParameterStoreShardState,
        load_path: str,
    ) -> None:
        # 先确保当前 step 的 prefetch 统计表已初始化。
        self._ensure_prefetch_tracking(step_index)
        # 同一 step 内同一个 group 只记录一次 prefetch。
        if state.group_id in self._step_prefetch_records:
            return
        # 追加一条 prefetch 访问记录。
        self._step_prefetch_records[state.group_id] = ParameterLoadRecord(
            group_id=state.group_id,
            component=state.component,
            source_kind=state.source_kind,
            load_path=load_path,
            resident_tier_after_access=state.resident_tier,
        )

    def _materialize(self, state: _ParameterStoreShardState) -> None:
        # -----------------
        # 已经有 CPU 可变张量时无需重复物化。
        if state.parameter_tensor is not None:
            return
        # 若已有 CPU buffer tensor，直接升级成 parameter_tensor。
        if state.parameter_buffer_tensor is not None:
            state.parameter_tensor = state.parameter_buffer_tensor.to(
                dtype=torch.float32,
                device="cpu",
            ).reshape(-1).contiguous()
            # 升级完成后清空 buffer tensor 持有。
            state.parameter_buffer_tensor = None
            # 当前 shard 已处于 CPU hot。
            state.resident_tier = "cpu_hot"
            # 累计一次 stage 操作。
            state.stage_count += 1
            self._cumulative_stage_ops += 1
            return
        # 先尝试从轻量 tuple buffer 中恢复。
        parameter_values = state.parameter_buffer
        # 没有本地 buffer 且启用 FP32 mirror 时，尝试从 NVMe 主副本恢复。
        if not parameter_values and self._fp32_mirror.enabled:
            mirrored_values = self._fp32_mirror.load(state.group_id)
            if mirrored_values is not None:
                # 全量物化 shard 直接把镜像值转成 CPU 张量。
                if state.representative_params == state.logical_params:
                    state.parameter_tensor = mirrored_values.to(
                        dtype=torch.float32,
                        device="cpu",
                    ).reshape(-1).contiguous()
                    state.resident_tier = "cpu_hot"
                    state.stage_count += 1
                    self._cumulative_stage_ops += 1
                    return
                # 非全量物化时，先把镜像值转成 Python tuple 缓冲区。
                parameter_values = tuple(
                    float(value) for value in mirrored_values.tolist()
                )
        # 仍无参数值且来源是 manifest 时，再回源读取权重。
        if not parameter_values and state.source_kind == "local_manifest":
            # 先恢复出 build_parameter_buffer 所需的 ParameterShardSnapshot。
            shard = ParameterShardSnapshot(
                group_id=state.group_id,
                component=state.component,
                residency_state="nvme_cold",
                committed_version=0,
                pending_version=None,
                logical_params=state.logical_params,
                bucket_id=state.bucket_id,
                expert_ids=state.expert_ids,
                last_touched_step=state.last_touched_step,
            )
            # 全量物化组件优先尝试读取完整参数缓冲区。
            load_result = (
                self._manifest.build_full_parameter_buffer(shard)
                if (
                    state.representative_params == state.logical_params
                    and state.component in _FULL_MATERIALIZATION_COMPONENTS
                )
                else None
            )
            # 全量路径未命中且当前是代表性模式时，退回代表性读取。
            if load_result is None and state.representative_params != state.logical_params:
                load_result = self._manifest.build_parameter_buffer(
                    shard,
                    representative_params=state.representative_params,
                )
            # manifest 成功读取到数据后，按返回形态分别处理。
            if load_result is not None:
                # 张量结果直接落成 CPU 参数张量。
                if isinstance(load_result.values, torch.Tensor):
                    state.parameter_tensor = load_result.values.to(
                        dtype=torch.float32,
                        device="cpu",
                    ).reshape(-1).contiguous()
                    state.resident_tier = "cpu_hot"
                    state.stage_count += 1
                    self._cumulative_stage_ops += 1
                    return
                # tuple 结果则先暂存为 parameter_values。
                parameter_values = load_result.values
        # 仍然没有任何真实数据时，用稳定种子生成兜底缓冲区。
        if not parameter_values:
            parameter_values = _initial_parameter_buffer(
                state.group_id,
                state.representative_params,
            )
        # 最终统一把 parameter_values 物化成 CPU float32 张量。
        state.parameter_tensor = torch.tensor(
            parameter_values,
            dtype=torch.float32,
            device="cpu",
        )
        # 物化完成后更新驻留层级与统计计数。
        state.resident_tier = "cpu_hot"
        state.stage_count += 1
        self._cumulative_stage_ops += 1

    def _quantize(self, state: _ParameterStoreShardState) -> None:
        # 未启用量化时直接返回。
        if not self._quantizer.enabled:
            return
        # 优先使用当前 CPU 参数张量作为量化源。
        source_tensor = state.parameter_tensor
        # 没有 parameter_tensor 时，继续向更轻的缓存层回退。
        if source_tensor is None:
            if state.parameter_buffer_tensor is not None:
                source_tensor = state.parameter_buffer_tensor
            elif state.parameter_buffer:
                source_tensor = torch.tensor(
                    state.parameter_buffer,
                    dtype=torch.float32,
                    device="cpu",
                )
            elif self._fp32_mirror.enabled:
                source_tensor = self._fp32_mirror.load(state.group_id)
        # 最终仍没有源张量时无法量化。
        if source_tensor is None:
            return
        # 执行量化并缓存 CPU 端量化结果。
        state.quantized_parameter = self._quantizer.quantize(source_tensor)
        # CPU 量化结果更新后，旧的 GPU 量化缓存必须失效。
        state.gpu_quantized_parameter = None
        # 记录一次量化操作。
        self._cumulative_quantize_ops += 1

    def _clear_quantized_cache(
        self,
        state: _ParameterStoreShardState,
    ) -> None:
        # 清空 CPU 侧量化缓存。
        state.quantized_parameter = None
        # 同时清空 GPU 侧量化缓存。
        state.gpu_quantized_parameter = None

    def _sync_fp32_master(self, state: _ParameterStoreShardState) -> None:
        # 未启用 FP32 mirror 时无需同步。
        if not self._fp32_mirror.enabled:
            return
        # 优先把当前 CPU 参数张量写回镜像。
        source_tensor = state.parameter_tensor
        # 没有 parameter_tensor 时，回退到 buffer tensor 或 tuple buffer。
        if source_tensor is None:
            if state.parameter_buffer_tensor is not None:
                source_tensor = state.parameter_buffer_tensor
            else:
                source_tensor = torch.tensor(
                    state.parameter_buffer,
                    dtype=torch.float32,
                    device="cpu",
                )
        # 把当前可得的 FP32 数据保存到 NVMe 主副本。
        self._fp32_mirror.save(state.group_id, source_tensor)
        # 累计一次 NVMe 同步操作。
        self._cumulative_nvme_sync_ops += 1

    def _offload(self, state: _ParameterStoreShardState) -> None:
        # 逻辑物化 + FP32 mirror 模式下，offload 前先丢掉量化缓存。
        if self._logical_materialization_enabled() and self._fp32_mirror.enabled:
            self._clear_quantized_cache(state)
        # 当前没有 CPU 参数张量时无需 offload。
        if state.parameter_tensor is None:
            return
        # 启用 FP32 mirror 时，offload 后不再保留本地参数缓冲。
        if self._fp32_mirror.enabled:
            state.parameter_buffer = ()
            state.parameter_buffer_tensor = None
        # 全量物化 shard 在无 mirror 场景下保留一份 CPU tensor clone 作为 buffer。
        elif state.representative_params == state.logical_params:
            state.parameter_buffer = ()
            state.parameter_buffer_tensor = state.parameter_tensor.to(
                dtype=torch.float32,
                device="cpu",
            ).clone()
        else:
            # 代表性模式则把张量回写成轻量 tuple buffer。
            state.parameter_buffer = tuple(
                float(value) for value in state.parameter_tensor.tolist()
            )
            state.parameter_buffer_tensor = None
        # 释放 CPU 参数张量本体。
        state.parameter_tensor = None
        # 更新驻留层级与 offload 计数。
        state.resident_tier = "nvme_cold"
        state.offload_count += 1
        self._cumulative_offload_ops += 1
        # 逻辑物化模式下额外尝试裁剪主机内存。
        if self._logical_materialization_enabled():
            _trim_host_memory()

    def _materialize_compute_view(
        self,
        state: _ParameterStoreShardState,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        # 先确保 CPU 侧参数已经物化完成。
        self._materialize(state)
        # 非 CUDA 请求直接返回 CPU 参数张量。
        if device.type != "cuda":
            assert state.parameter_tensor is not None
            return state.parameter_tensor
        # -----------------
        # 先准备 GPU 侧量化缓存（若启用了量化）。
        if self._quantizer.enabled:
            # CPU 侧量化缓存不存在时先现算一次。
            if state.quantized_parameter is None:
                self._quantize(state)
            # GPU 侧量化缓存不存在或设备不匹配时，重新拷到目标设备。
            if (
                state.gpu_quantized_parameter is None
                or not _device_matches(
                    state.gpu_quantized_parameter.qweight.device,
                    device,
                )
            ):
                assert state.quantized_parameter is not None
                state.gpu_quantized_parameter = state.quantized_parameter.to_device(
                    device
                )
        # -----------------
        # 再准备 GPU 侧可直接参与计算的 FP32 视图。
        if (
            state.gpu_parameter_tensor is None
            or not _device_matches(state.gpu_parameter_tensor.device, device)
        ):
            # 启用量化时，通过 GPU 量化缓存解量化得到计算视图。
            if self._quantizer.enabled:
                assert state.gpu_quantized_parameter is not None
                state.gpu_parameter_tensor = self._quantizer.dequantize(
                    state.gpu_quantized_parameter,
                    device=device,
                )
            else:
                # 未启用量化时，直接把 CPU 参数张量拷到目标设备。
                assert state.parameter_tensor is not None
                state.gpu_parameter_tensor = state.parameter_tensor.detach().to(
                    device=device
                )
            # 记录一次 GPU stage 操作。
            state.gpu_stage_count += 1
            self._cumulative_gpu_stage_ops += 1
        # 返回当前设备上的 GPU 计算视图。
        return state.gpu_parameter_tensor

    def _materialize_quantized_view(
        self,
        state: _ParameterStoreShardState,
        *,
        device: torch.device,
    ) -> PackedQuantizedTensor | None:
        # 未启用量化时没有量化视图可返回。
        if not self._quantizer.enabled:
            return None
        # 先确保 CPU 侧参数已经物化。
        self._materialize(state)
        # CPU 侧量化缓存不存在时先计算。
        if state.quantized_parameter is None:
            self._quantize(state)
        # 非 CUDA 设备直接返回 CPU 侧量化结果。
        if device.type != "cuda":
            return state.quantized_parameter
        # GPU 量化缓存不存在或设备不匹配时，重新迁移到目标设备。
        if (
            state.gpu_quantized_parameter is None
            or not _device_matches(
                state.gpu_quantized_parameter.qweight.device,
                device,
            )
        ):
            assert state.quantized_parameter is not None
            state.gpu_quantized_parameter = state.quantized_parameter.to_device(device)
        # 返回目标设备上的量化视图。
        return state.gpu_quantized_parameter

    def _release_compute_view(
        self,
        state: _ParameterStoreShardState,
    ) -> None:
        # CPU 外的两类 GPU 缓存都为空时无需释放。
        if state.gpu_parameter_tensor is None and state.gpu_quantized_parameter is None:
            return
        # 清空 GPU 侧 FP32 计算视图。
        state.gpu_parameter_tensor = None
        # 同步清空 GPU 侧量化缓存。
        state.gpu_quantized_parameter = None
        # 记录一次 GPU cache release。
        state.gpu_release_count += 1
        self._cumulative_gpu_release_ops += 1

    def _release_gpu_parameter_tensor(
        self,
        state: _ParameterStoreShardState,
    ) -> None:
        # 没有 GPU FP32 计算视图时无需单独释放。
        if state.gpu_parameter_tensor is None:
            return
        # 仅释放 GPU FP32 张量，保留 GPU 量化缓存。
        state.gpu_parameter_tensor = None
        # 累计一次 GPU release 操作。
        state.gpu_release_count += 1
        self._cumulative_gpu_release_ops += 1

    def parameter_view(
        self,
        shard: ParameterShardSnapshot,
        *,
        step_index: int,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        # -----------------
        # 先判断这次访问是首次创建、CPU 热复用还是 buffer / NVMe 复用。
        self._ensure_step_tracking(step_index)
        existing_state = self._states.get(shard.group_id)
        existing_tensor = (
            None if existing_state is None else existing_state.parameter_tensor
        )
        # 命中现有状态或创建新状态。
        state = self._get_or_create_state(shard)
        # 结合当前 step 的 transport cache 上下文刷新标记。
        self._refresh_transport_backing(state, step_index=step_index)
        # 根据进入 materialize 前的状态推断 load_path。
        if existing_state is None:
            load_path = self._creation_load_path(state)
        elif existing_tensor is not None:
            load_path = "cpu_hot_reuse"
        elif not state.parameter_buffer and self._fp32_mirror.enabled:
            load_path = "nvme_fp32_mirror"
        else:
            load_path = "buffer_reuse"
        # -----------------
        # 执行物化并记录本次 step 访问。
        self._materialize(state)
        state.last_touched_step = step_index
        self._record_step_access(
            step_index=step_index,
            state=state,
            load_path=load_path,
        )
        # 未指定 device 时返回 CPU 侧张量副本，避免外部直接改内部缓存。
        if device is None:
            assert state.parameter_tensor is not None
            return state.parameter_tensor.detach().clone()
        # 把 device 参数规范化成 torch.device。
        view_device = torch.device(device)
        # CUDA 请求直接走 GPU 计算视图路径。
        if view_device.type == "cuda":
            tensor = self._materialize_compute_view(
                state,
                device=view_device,
            )
            return tensor.detach()
        # 其余设备则从 CPU 参数张量 clone 后再转移。
        assert state.parameter_tensor is not None
        return state.parameter_tensor.detach().clone().to(device=view_device)

    def stage_compute_views(
        self,
        *,
        step_index: int,
        parameter_shards: tuple[ParameterShardSnapshot, ...],
        device: str | torch.device | None,
    ) -> None:
        # 没有目标设备时无需预先 stage。
        if device is None:
            return
        # 统一把 device 转成 torch.device。
        view_device = torch.device(device)
        # 非 CUDA 设备不需要 GPU stage。
        if view_device.type != "cuda":
            return
        # 逻辑物化 + 量化模式下，仅 full_bucket CUDA 路径才允许预先 stage。
        if (
            self._quantizer.enabled
            and self.config.execution.trainable_shard_materialization == "logical"
        ):
            if not self._full_bucket_logical_cuda_enabled():
                return
        # 准备本 step 的加载统计容器。
        self._ensure_step_tracking(step_index)
        # 逐个 shard 构建或命中状态，并把计算视图提前搬到 GPU。
        for shard in parameter_shards:
            state = self._get_or_create_state(shard)
            self._refresh_transport_backing(state, step_index=step_index)
            state.last_touched_step = step_index
            self._materialize_compute_view(
                state,
                device=view_device,
            )

    def quantized_parameter_view(
        self,
        shard: ParameterShardSnapshot,
        *,
        step_index: int,
        device: str | torch.device | None = None,
    ) -> PackedQuantizedTensor | None:
        # 获取或创建当前 shard 状态。
        state = self._get_or_create_state(shard)
        # 刷新 transport backing 标记。
        self._refresh_transport_backing(state, step_index=step_index)
        # 记录最近触达 step。
        state.last_touched_step = step_index
        # 未显式指定设备时默认返回 CPU 量化视图。
        view_device = torch.device("cpu") if device is None else torch.device(device)
        # 返回目标设备上的量化参数视图。
        return self._materialize_quantized_view(
            state,
            device=view_device,
        )

    def release_compute_views(
        self,
        *,
        group_ids: tuple[str, ...],
    ) -> None:
        # 逐个 group 释放 GPU 侧缓存视图。
        for group_id in group_ids:
            state = self._states.get(group_id)
            if state is None:
                continue
            self._release_compute_view(state)

    def mutable_parameter(
        self,
        shard: ParameterShardSnapshot,
        *,
        step_index: int,
    ) -> torch.Tensor:
        # -----------------
        # 先确定这次写访问的加载路径类型。
        self._ensure_step_tracking(step_index)
        existing_state = self._states.get(shard.group_id)
        existing_tensor = (
            None if existing_state is None else existing_state.parameter_tensor
        )
        # 获取或创建当前 shard 状态。
        state = self._get_or_create_state(shard)
        # 刷新 transport cache 命中标记。
        self._refresh_transport_backing(state, step_index=step_index)
        # 依据进入物化前的状态推断 load_path。
        if existing_state is None:
            load_path = self._creation_load_path(state)
        elif existing_tensor is not None:
            load_path = "cpu_hot_reuse"
        elif not state.parameter_buffer and self._fp32_mirror.enabled:
            load_path = "nvme_fp32_mirror"
        else:
            load_path = "buffer_reuse"
        # -----------------
        # 物化出可写 CPU 参数张量并登记访问。
        self._materialize(state)
        state.last_touched_step = step_index
        self._record_step_access(
            step_index=step_index,
            state=state,
            load_path=load_path,
        )
        # 返回内部可变张量本体，供优化器原地更新。
        assert state.parameter_tensor is not None
        return state.parameter_tensor

    def finalize_update(
        self,
        shard: ParameterShardSnapshot,
        *,
        step_index: int,
        offload_after_update: bool,
        sync_fp32_to_nvme: bool = True,
        retain_gpu_quantized_cache: bool = False,
    ) -> None:
        # 先获取当前 shard 状态并刷新缓存命中信息。
        state = self._get_or_create_state(shard)
        self._refresh_transport_backing(state, step_index=step_index)
        # 更新最近触达 step。
        state.last_touched_step = step_index
        # 需要时先把最新 FP32 参数同步回 NVMe 主副本。
        if sync_fp32_to_nvme:
            self._sync_fp32_master(state)
        # 非“立即 offload + logical mirror”场景下，保留或刷新量化缓存。
        if not (
            offload_after_update
            and self._logical_materialization_enabled()
            and self._fp32_mirror.enabled
        ):
            self._quantize(state)
        else:
            # 准备 offload 时，logical mirror 路径直接清空量化缓存。
            self._clear_quantized_cache(state)
        # 用户要求保留 GPU 量化缓存且当前不会 offload 时，只释放 GPU FP32 视图。
        if (
            retain_gpu_quantized_cache
            and not offload_after_update
            and self._quantizer.enabled
            and self.config.execution.compute_device == "gpu"
        ):
            self._release_gpu_parameter_tensor(state)
            self._materialize_quantized_view(
                state,
                device=torch.device("cuda"),
            )
        else:
            # 其余情况直接释放所有 GPU 相关缓存。
            self._release_compute_view(state)
        # 若配置要求更新后立刻 offload，则执行落冷。
        if offload_after_update:
            self._offload(state)

    def offload_group_ids(
        self,
        *,
        group_ids: tuple[str, ...],
        sync_fp32_to_nvme: bool = True,
    ) -> None:
        # 逐个 group 执行显式 offload。
        for group_id in group_ids:
            state = self._states.get(group_id)
            if state is None:
                continue
            # 需要时先把当前 FP32 数据同步回 NVMe。
            if sync_fp32_to_nvme:
                self._sync_fp32_master(state)
            # 非 logical mirror 场景下，offload 前保留 CPU 量化缓存。
            if not (
                self._logical_materialization_enabled()
                and self._fp32_mirror.enabled
            ):
                self._quantize(state)
            else:
                # logical mirror 场景则直接清空量化缓存。
                self._clear_quantized_cache(state)
            # 释放 GPU 视图并把 CPU 参数落冷。
            self._release_compute_view(state)
            self._offload(state)

    def set_transport_cache_context(
        self,
        *,
        step_index: int,
        cached_file_names: tuple[str, ...],
    ) -> None:
        # 当前 step 的 prefetch / load 统计都需要切到同一个上下文。
        self._ensure_prefetch_tracking(step_index)
        self._ensure_step_tracking(step_index)
        # 记录 transport cache 所属的 step。
        self._transport_cache_context_step = step_index
        # 保存当前 step 已缓存的文件集合。
        self._transport_cached_files = frozenset(cached_file_names)
        # 让所有已跟踪状态重新判断自己是否被 transport cache 覆盖。
        for state in self._states.values():
            self._refresh_transport_backing(state)

    def prefetch_shards(
        self,
        *,
        step_index: int,
        parameter_shards: tuple[ParameterShardSnapshot, ...],
    ) -> ParameterPrefetchSummary:
        # 先准备本 step 的 prefetch 统计容器。
        self._ensure_prefetch_tracking(step_index)
        # 逐个 shard 执行预取。
        for shard in parameter_shards:
            # 仅特定组件允许被 prefetch。
            if shard.component not in _PREFETCHABLE_COMPONENTS:
                continue
            # 先查看当前 shard 是否已有状态 / 已有 CPU 热张量。
            existing_state = self._states.get(shard.group_id)
            existing_tensor = (
                None if existing_state is None else existing_state.parameter_tensor
            )
            # 获取或创建内部状态。
            state = self._get_or_create_state(shard)
            # 结合当前 step 刷新 transport backing 标记。
            self._refresh_transport_backing(state, step_index=step_index)
            # 更新最近触达 step。
            state.last_touched_step = step_index
            # 根据预取前状态判定本次 load_path。
            if existing_state is None:
                load_path = self._creation_load_path(state)
            elif existing_tensor is not None:
                load_path = "cpu_hot_reuse"
            elif not state.parameter_buffer and self._fp32_mirror.enabled:
                load_path = "nvme_fp32_mirror"
            else:
                load_path = "buffer_reuse"
            # logical + mirror 模式下允许只登记来源，不强制把参数物化到 CPU。
            if not (self._logical_materialization_enabled() and self._fp32_mirror.enabled):
                self._materialize(state)
            # 把本次预取记录写入 step 统计。
            self._record_prefetch_access(
                step_index=step_index,
                state=state,
                load_path=load_path,
            )
        # 返回当前 step 的 prefetch 汇总。
        return self.step_prefetch_summary()

    def _snapshot_parameter_values(
        self,
        state: _ParameterStoreShardState,
    ) -> tuple[float, ...] | None:
        # 已有 parameter_tensor 或 tuple buffer 时，让 to_snapshot 自己处理，不在这里覆盖。
        if state.parameter_tensor is not None or state.parameter_buffer:
            return None
        # 若存在 buffer tensor，则把它导出成 tuple 供快照使用。
        if state.parameter_buffer_tensor is not None:
            return tuple(
                float(value)
                for value in state.parameter_buffer_tensor.to(dtype=torch.float32).tolist()
            )
        # 未启用 FP32 mirror 时没有更多兜底来源。
        if not self._fp32_mirror.enabled:
            return None
        # 再尝试从 NVMe 主副本读取一份快照值。
        mirrored_values = self._fp32_mirror.load(state.group_id)
        # 主副本也不存在时返回 None。
        if mirrored_values is None:
            return None
        # 把镜像值转成 tuple，供快照序列化使用。
        return tuple(float(value) for value in mirrored_values.tolist())

    def _sync_fp32_master_for_snapshot(
        self,
        state: _ParameterStoreShardState,
    ) -> bool:
        # 未启用 mirror 或当前参数量很小时，不需要为快照额外刷 NVMe。
        if (
            not self._fp32_mirror.enabled
            or state.representative_params <= _MAX_REPRESENTATIVE_PARAMS
        ):
            return False
        # 优先使用当前 CPU 参数张量。
        source_tensor = state.parameter_tensor
        # 没有 parameter_tensor 时，继续从 buffer tensor / tuple buffer / 已存在 mirror 兜底。
        if source_tensor is None:
            if state.parameter_buffer_tensor is not None:
                source_tensor = state.parameter_buffer_tensor
            elif state.parameter_buffer:
                source_tensor = torch.tensor(
                    state.parameter_buffer,
                    dtype=torch.float32,
                    device="cpu",
                )
            elif self._fp32_mirror.path_for_group(state.group_id).exists():
                # 已经有 mirror 文件时，说明快照可直接省略参数值。
                return True
        # 仍没有任何可同步源时返回 False。
        if source_tensor is None:
            return False
        # 把当前 FP32 数据补写到 NVMe 主副本。
        self._fp32_mirror.save(state.group_id, source_tensor)
        # 返回 True，表示快照可以省略具体参数值。
        return True

    def snapshot(self) -> tuple[ParameterStoreShardSnapshot, ...]:
        # 按 group_id 排序导出全部 shard 状态快照。
        return tuple(
            state.to_snapshot(
                parameter_values_override=self._snapshot_parameter_values(state),
                omit_parameter_values=self._sync_fp32_master_for_snapshot(state),
            )
            for _, state in sorted(self._states.items(), key=lambda item: item[0])
        )

    def load_snapshot(
        self,
        snapshots: tuple[ParameterStoreShardSnapshot, ...],
        *,
        cumulative_quantize_ops: int | None = None,
        cumulative_nvme_sync_ops: int | None = None,
    ) -> None:
        # -----------------
        # 先清空当前内存中的所有 shard 状态。
        self._states = {}
        # 逐条快照恢复内部状态对象。
        for snapshot in snapshots:
            state = _ParameterStoreShardState(
                group_id=snapshot.group_id,
                component=snapshot.component,
                logical_params=snapshot.logical_params,
                representative_params=max(snapshot.representative_params, 1),
                bucket_id=None,
                expert_ids=(),
                resident_tier=snapshot.resident_tier,
                source_kind=snapshot.source_kind,
                source_file_names=snapshot.source_file_names,
                source_tensor_count=snapshot.source_tensor_count,
                source_layout=snapshot.source_layout,
                stage_count=snapshot.stage_count,
                offload_count=snapshot.offload_count,
                last_touched_step=snapshot.last_touched_step,
                parameter_buffer=(
                    snapshot.parameter_values
                    if snapshot.parameter_values
                    else ()
                ),
                parameter_buffer_tensor=None,
                parameter_tensor=None,
                quantized_parameter=None,
                gpu_quantized_parameter=None,
                gpu_parameter_tensor=None,
                gpu_stage_count=snapshot.gpu_stage_count,
                gpu_release_count=snapshot.gpu_release_count,
            )
            # 启用 FP32 mirror 且快照自带参数值时，同时恢复 NVMe 主副本文件。
            if self._fp32_mirror.enabled and snapshot.parameter_values:
                self._fp32_mirror.save(
                    snapshot.group_id,
                    torch.tensor(snapshot.parameter_values, dtype=torch.float32),
                )
                # 参数值已经落到 NVMe 后，本地 tuple buffer 可清空。
                state.parameter_buffer = ()
            # 把恢复后的状态重新挂回状态表。
            self._states[snapshot.group_id] = state
        # 根据快照内容重建累计 stage/offload 计数。
        self._cumulative_stage_ops = sum(
            snapshot.stage_count for snapshot in snapshots
        )
        self._cumulative_offload_ops = sum(
            snapshot.offload_count for snapshot in snapshots
        )
        self._cumulative_gpu_stage_ops = sum(
            snapshot.gpu_stage_count for snapshot in snapshots
        )
        self._cumulative_gpu_release_ops = sum(
            snapshot.gpu_release_count for snapshot in snapshots
        )
        # 快照恢复后，step 级统计需要重新开始计数。
        self._step_prefetch_step = -1
        self._step_prefetch_records = {}
        self._step_load_step = -1
        self._step_load_records = {}
        # 量化累计计数先清零，后面可能重新推导。
        self._cumulative_quantize_ops = 0
        # 若当前启用了量化，则为所有恢复状态重新构建量化缓存。
        if self._quantizer.enabled:
            for state in self._states.values():
                self._quantize(state)
        # 调用方没提供累计量化次数时，使用刚刚推导出的结果。
        if cumulative_quantize_ops is None:
            cumulative_quantize_ops = self._cumulative_quantize_ops
        self._cumulative_quantize_ops = cumulative_quantize_ops
        # 未显式给出 NVMe sync 次数时，按 offload 次数推断一个默认值。
        inferred_nvme_sync_ops = (
            sum(snapshot.offload_count for snapshot in snapshots)
            if self._fp32_mirror.enabled
            else 0
        )
        # 记录最终采用的 NVMe sync 累计次数。
        self._cumulative_nvme_sync_ops = (
            inferred_nvme_sync_ops
            if cumulative_nvme_sync_ops is None
            else cumulative_nvme_sync_ops
        )

    def summary(self) -> ParameterStoreSummary:
        # 先初始化各类驻留与来源统计计数器。
        cpu_hot = 0
        cpu_hot_resident_bytes = 0
        nvme_cold = 0
        gpu_cached = 0
        gpu_cached_bytes = 0
        quantized = 0
        quantized_bytes = 0
        gpu_quantized = 0
        gpu_quantized_bytes = 0
        manifest_backed = 0
        synthetic_seeded = 0
        nvme_fp32_mirror = 0
        transport_backed = 0
        unique_files: set[str] = set()
        source_tensor_count = 0
        # 遍历全部 shard 状态，累计 summary 所需指标。
        for state in self._states.values():
            # 统计 CPU hot / NVMe cold 数量与 CPU 常驻字节数。
            if state.resident_tier == "cpu_hot":
                cpu_hot += 1
                cpu_hot_resident_bytes += self._resident_cpu_hot_bytes(state)
            else:
                nvme_cold += 1
            # 统计 GPU FP32 计算缓存数量和字节数。
            if state.gpu_parameter_tensor is not None:
                gpu_cached += 1
                gpu_cached_bytes += (
                    state.gpu_parameter_tensor.numel()
                    * state.gpu_parameter_tensor.element_size()
                )
            # 统计 CPU 侧量化缓存。
            if state.quantized_parameter is not None:
                quantized += 1
                quantized_bytes += state.quantized_parameter.resident_bytes
            # 统计 GPU 侧量化缓存。
            if state.gpu_quantized_parameter is not None:
                gpu_quantized += 1
                gpu_quantized_bytes += state.gpu_quantized_parameter.resident_bytes
            # 按来源种类分别累计 shard 数。
            if state.source_kind == "local_manifest":
                manifest_backed += 1
            elif state.source_kind == "synthetic_seed":
                synthetic_seeded += 1
            else:
                nvme_fp32_mirror += 1
            # 统计 transport cache 覆盖数量。
            if state.transport_backed:
                transport_backed += 1
            # 累加来源文件去重集合与来源张量计数。
            unique_files.update(state.source_file_names)
            source_tensor_count += state.source_tensor_count
        # 把累计结果封装成 ParameterStoreSummary 返回。
        return ParameterStoreSummary(
            tracked_shards=len(self._states),
            cpu_hot_shards=cpu_hot,
            nvme_cold_shards=nvme_cold,
            cpu_hot_resident_bytes=cpu_hot_resident_bytes,
            gpu_cached_shards=gpu_cached,
            gpu_cached_bytes=gpu_cached_bytes,
            quantized_shards=quantized,
            quantized_bytes=quantized_bytes,
            gpu_quantized_shards=gpu_quantized,
            gpu_quantized_bytes=gpu_quantized_bytes,
            manifest_backed_shards=manifest_backed,
            synthetic_seeded_shards=synthetic_seeded,
            nvme_fp32_mirror_shards=nvme_fp32_mirror,
            transport_backed_shards=transport_backed,
            source_file_count=len(unique_files),
            source_tensor_count=source_tensor_count,
            cumulative_stage_ops=self._cumulative_stage_ops,
            cumulative_offload_ops=self._cumulative_offload_ops,
            cumulative_gpu_stage_ops=self._cumulative_gpu_stage_ops,
            cumulative_gpu_release_ops=self._cumulative_gpu_release_ops,
            cumulative_quantize_ops=self._cumulative_quantize_ops,
            cumulative_nvme_sync_ops=self._cumulative_nvme_sync_ops,
        )

    def source_summary(
        self,
        parameter_shards: tuple[ParameterShardSnapshot, ...],
    ) -> ParameterSourceSummary:
        # 先准备逐 shard 来源记录与汇总计数器。
        shard_sources: list[ParameterSourceRecord] = []
        unique_files: set[str] = set()
        manifest_backed = 0
        synthetic_seeded = 0
        nvme_fp32_mirror = 0
        transport_backed = 0
        tensor_count = 0
        # 仅对调用方传入的 parameter_shards 做来源统计。
        for shard in parameter_shards:
            state = self._states.get(shard.group_id)
            if state is None:
                continue
            # 按来源类型累计 shard 数。
            if state.source_kind == "local_manifest":
                manifest_backed += 1
            elif state.source_kind == "synthetic_seed":
                synthetic_seeded += 1
            else:
                nvme_fp32_mirror += 1
            # 累计 transport cache 覆盖情况。
            if state.transport_backed:
                transport_backed += 1
            # 累计去重文件数与来源张量数。
            unique_files.update(state.source_file_names)
            tensor_count += state.source_tensor_count
            # 从 source_layout 中提取当前 shard 涉及的层索引集合。
            layer_indices = tuple(
                sorted(
                    {
                        entry.layer_index
                        for entry in state.source_layout
                        if entry.layer_index is not None
                    }
                )
            )
            # 从 source_layout 中提取当前 shard 涉及的语义角色集合。
            semantic_roles = tuple(
                sorted({entry.semantic_role for entry in state.source_layout})
            )
            # 追加一条逐 shard 的来源记录。
            shard_sources.append(
                ParameterSourceRecord(
                    group_id=state.group_id,
                    component=state.component,
                    source_kind=state.source_kind,
                    file_names=state.source_file_names,
                    tensor_count=state.source_tensor_count,
                    resident_tier=state.resident_tier,
                    transport_backed=state.transport_backed,
                    layer_indices=layer_indices,
                    semantic_roles=semantic_roles,
                )
            )
        # 返回来源层面的汇总对象。
        return ParameterSourceSummary(
            touched_shards=len(shard_sources),
            manifest_backed_shards=manifest_backed,
            synthetic_seeded_shards=synthetic_seeded,
            nvme_fp32_mirror_shards=nvme_fp32_mirror,
            transport_backed_shards=transport_backed,
            file_count=len(unique_files),
            tensor_count=tensor_count,
            shard_sources=tuple(shard_sources),
        )

    def _build_load_summary(
        self,
        records: tuple[ParameterLoadRecord, ...],
    ) -> ParameterLoadSummary:
        # 初始化不同 load_path 的计数器。
        transport_cache_loads = 0
        direct_manifest_loads = 0
        synthetic_seed_loads = 0
        nvme_fp32_mirror_loads = 0
        buffer_reuses = 0
        cpu_hot_reuses = 0
        # 逐条记录统计各类加载路径命中次数。
        for record in records:
            if record.load_path == "transport_cache":
                transport_cache_loads += 1
            elif record.load_path == "direct_manifest":
                direct_manifest_loads += 1
            elif record.load_path == "synthetic_seed":
                synthetic_seed_loads += 1
            elif record.load_path == "nvme_fp32_mirror":
                nvme_fp32_mirror_loads += 1
            elif record.load_path == "buffer_reuse":
                buffer_reuses += 1
            elif record.load_path == "cpu_hot_reuse":
                cpu_hot_reuses += 1
        # 返回 step 级加载汇总。
        return ParameterLoadSummary(
            touched_shards=len(records),
            transport_cache_loads=transport_cache_loads,
            direct_manifest_loads=direct_manifest_loads,
            synthetic_seed_loads=synthetic_seed_loads,
            nvme_fp32_mirror_loads=nvme_fp32_mirror_loads,
            buffer_reuses=buffer_reuses,
            cpu_hot_reuses=cpu_hot_reuses,
            records=records,
        )

    def _build_prefetch_summary(
        self,
        records: tuple[ParameterLoadRecord, ...],
    ) -> ParameterPrefetchSummary:
        # 初始化不同 prefetch 路径的计数器。
        transport_cache_prefetches = 0
        direct_manifest_prefetches = 0
        synthetic_seed_prefetches = 0
        nvme_fp32_mirror_prefetches = 0
        buffer_reuses = 0
        cpu_hot_reuses = 0
        # 逐条 prefetch 记录统计路径命中次数。
        for record in records:
            if record.load_path == "transport_cache":
                transport_cache_prefetches += 1
            elif record.load_path == "direct_manifest":
                direct_manifest_prefetches += 1
            elif record.load_path == "synthetic_seed":
                synthetic_seed_prefetches += 1
            elif record.load_path == "nvme_fp32_mirror":
                nvme_fp32_mirror_prefetches += 1
            elif record.load_path == "buffer_reuse":
                buffer_reuses += 1
            elif record.load_path == "cpu_hot_reuse":
                cpu_hot_reuses += 1
        # 返回 step 级 prefetch 汇总。
        return ParameterPrefetchSummary(
            touched_shards=len(records),
            transport_cache_prefetches=transport_cache_prefetches,
            direct_manifest_prefetches=direct_manifest_prefetches,
            synthetic_seed_prefetches=synthetic_seed_prefetches,
            nvme_fp32_mirror_prefetches=nvme_fp32_mirror_prefetches,
            buffer_reuses=buffer_reuses,
            cpu_hot_reuses=cpu_hot_reuses,
            records=records,
        )

    def step_load_summary(self) -> ParameterLoadSummary:
        # 按 group_id 排序后汇总当前 step 的 load 记录。
        return self._build_load_summary(
            tuple(
                record
                for _, record in sorted(
                    self._step_load_records.items(),
                    key=lambda item: item[0],
                )
            )
        )

    def load_summary_for_groups(
        self,
        group_ids: tuple[str, ...],
    ) -> ParameterLoadSummary:
        # 只筛选指定 group_ids 的 load 记录并汇总。
        return self._build_load_summary(
            tuple(
                self._step_load_records[group_id]
                for group_id in sorted(group_ids)
                if group_id in self._step_load_records
            )
        )

    def step_prefetch_summary(self) -> ParameterPrefetchSummary:
        # 按 group_id 排序后汇总当前 step 的 prefetch 记录。
        return self._build_prefetch_summary(
            tuple(
                record
                for _, record in sorted(
                    self._step_prefetch_records.items(),
                    key=lambda item: item[0],
                )
            )
        )

    def prefetch_summary_for_groups(
        self,
        group_ids: tuple[str, ...],
    ) -> ParameterPrefetchSummary:
        # 只筛选指定 group_ids 的 prefetch 记录并汇总。
        return self._build_prefetch_summary(
            tuple(
                self._step_prefetch_records[group_id]
                for group_id in sorted(group_ids)
                if group_id in self._step_prefetch_records
            )
        )

    def resident_tier_counts_for_groups(
        self,
        group_ids: tuple[str, ...],
    ) -> tuple[int, int]:
        # 初始化 CPU hot / NVMe cold 计数器。
        cpu_hot = 0
        nvme_cold = 0
        # 仅统计指定 group_ids 的驻留层级。
        for group_id in group_ids:
            state = self._states.get(group_id)
            if state is None:
                continue
            if state.resident_tier == "cpu_hot":
                cpu_hot += 1
            else:
                nvme_cold += 1
        # 返回 CPU hot 与 NVMe cold 的数量对。
        return cpu_hot, nvme_cold

    def cpu_hot_resident_bytes(self) -> int:
        # 汇总所有 CPU hot shard 当前常驻的 CPU 内存字节数。
        return sum(
            self._resident_cpu_hot_bytes(state)
            for state in self._states.values()
            if state.resident_tier == "cpu_hot"
        )

    def _resident_cpu_hot_bytes(
        self,
        state: _ParameterStoreShardState,
    ) -> int:
        # 从 0 开始累计当前 shard 的 CPU 常驻字节数。
        resident_bytes = 0
        # parameter_tensor 存在时，计入其张量字节数。
        if state.parameter_tensor is not None:
            resident_bytes += (
                state.parameter_tensor.numel() * state.parameter_tensor.element_size()
            )
        # buffer tensor 存在时，计入其张量字节数。
        if state.parameter_buffer_tensor is not None:
            resident_bytes += (
                state.parameter_buffer_tensor.numel()
                * state.parameter_buffer_tensor.element_size()
            )
        elif state.parameter_buffer:
            # tuple buffer 按 float32 近似估算，每个元素记 4 字节。
            resident_bytes += len(state.parameter_buffer) * 4
        # CPU 侧量化缓存也属于 CPU hot 常驻内存的一部分。
        if state.quantized_parameter is not None:
            resident_bytes += state.quantized_parameter.resident_bytes
        # 返回当前 shard 的累计常驻字节数。
        return resident_bytes

    def source_layout(
        self,
        shard: ParameterShardSnapshot,
    ) -> tuple[ParameterSourceSlice, ...]:
        # 直接返回当前 shard 状态里记录的来源布局。
        return self._get_or_create_state(shard).source_layout
