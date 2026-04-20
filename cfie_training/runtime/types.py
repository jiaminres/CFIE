"""Serializable runtime types for the first CFIE training engine."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Literal


# 校验整数必须为正数。
def _require_positive_int(name: str, value: int) -> None:
    # 小于 1 时说明该字段不满足“正整数”约束。
    if value < 1:
        # 直接抛出带字段名的错误，方便上层定位配置问题。
        raise ValueError(f"{name} must be >= 1")


# 校验整数必须为非负数。
def _require_non_negative_int(name: str, value: int) -> None:
    # 小于 0 时说明该字段不满足“非负整数”约束。
    if value < 0:
        # 直接抛出带字段名的错误，方便上层定位配置问题。
        raise ValueError(f"{name} must be >= 0")


ResidencyState = Literal["nvme_cold", "cpu_staged", "gpu_active", "cpu_dirty"]
OptimizerStateTier = Literal["cpu_hot", "nvme_cold"]
ParameterStoreTier = Literal["cpu_hot", "nvme_cold"]
ParameterSourceKind = Literal[
    "local_manifest",
    "synthetic_seed",
    "nvme_fp32_mirror",
]
ParameterLoadPath = Literal[
    "transport_cache",
    "direct_manifest",
    "synthetic_seed",
    "nvme_fp32_mirror",
    "buffer_reuse",
    "cpu_hot_reuse",
]
# 上述 Literal 类型统一约束训练运行时里各类状态字段的可选值范围。


@dataclass(slots=True, frozen=True)
class BatchShape:
    samples: int
    tokens_per_sample: int
    source_kind: str = "synthetic_shape"
    dataset_name: str | None = None
    sample_indices: tuple[int, ...] = ()
    loss_token_count: int = 0
    token_rows: tuple[tuple[int, ...], ...] = ()
    target_rows: tuple[tuple[int, ...], ...] = ()
    attention_mask_rows: tuple[tuple[int, ...], ...] = ()
    target_attention_mask_rows: tuple[tuple[int, ...], ...] = ()

    # 校验 batch 形状与显式 token 行是否一致。
    def __post_init__(self) -> None:
        # 先校验样本数必须为正。
        _require_positive_int("samples", self.samples)
        # 再校验每个样本的 token 数必须为正。
        _require_positive_int("tokens_per_sample", self.tokens_per_sample)
        # loss token 计数允许为 0，但不能为负。
        _require_non_negative_int("loss_token_count", self.loss_token_count)
        # 如果显式给了 token_rows，则其行数必须和样本数一致。
        if self.token_rows and len(self.token_rows) != self.samples:
            raise ValueError("token_rows length must match samples when provided")
        # 如果显式给了 target_rows，则其行数同样必须和样本数一致。
        if self.target_rows and len(self.target_rows) != self.samples:
            raise ValueError("target_rows length must match samples when provided")
        # 如果显式给了 attention_mask_rows，则其行数同样必须和样本数一致。
        if self.attention_mask_rows and len(self.attention_mask_rows) != self.samples:
            raise ValueError(
                "attention_mask_rows length must match samples when provided"
            )
        # 如果显式给了 target_attention_mask_rows，则其行数同样必须和样本数一致。
        if (
            self.target_attention_mask_rows
            and len(self.target_attention_mask_rows) != self.samples
        ):
            raise ValueError(
                "target_attention_mask_rows length must match samples when provided"
            )
        # 逐行校验 token_rows 中每一行的长度。
        for row in self.token_rows:
            # 每个 token 行都必须覆盖完整的 tokens_per_sample。
            if len(row) != self.tokens_per_sample:
                raise ValueError(
                    "each token_rows entry must match tokens_per_sample"
                )
        # 逐行校验 target_rows 中每一行的长度。
        for row in self.target_rows:
            # 每个 target 行也必须覆盖完整的 tokens_per_sample。
            if len(row) != self.tokens_per_sample:
                raise ValueError(
                    "each target_rows entry must match tokens_per_sample"
                )
        # 逐行校验 attention_mask_rows 中每一行的长度和取值。
        for row in self.attention_mask_rows:
            # 每个 attention mask 行都必须覆盖完整的 tokens_per_sample。
            if len(row) != self.tokens_per_sample:
                raise ValueError(
                    "each attention_mask_rows entry must match tokens_per_sample"
                )
            # mask 只允许 0/1，避免上层把 token id 或权重误传进来。
            if any(value not in (0, 1, False, True) for value in row):
                raise ValueError("attention_mask_rows entries must be 0 or 1")
        # 逐行校验 target_attention_mask_rows 中每一行的长度和取值。
        for row in self.target_attention_mask_rows:
            # target mask 与 target_rows 一样按 tokens_per_sample 对齐。
            if len(row) != self.tokens_per_sample:
                raise ValueError(
                    "each target_attention_mask_rows entry must match "
                    "tokens_per_sample"
                )
            # mask 只允许 0/1，保持 loss token 统计口径清晰。
            if any(value not in (0, 1, False, True) for value in row):
                raise ValueError("target_attention_mask_rows entries must be 0 or 1")
        # attention mask 只能依附于显式 token_rows 使用。
        if self.attention_mask_rows and not self.token_rows:
            raise ValueError("attention_mask_rows requires token_rows")
        # target attention mask 只能依附于显式 target_rows 使用。
        if self.target_attention_mask_rows and not self.target_rows:
            raise ValueError("target_attention_mask_rows requires target_rows")
        # 当前数据规划器只生成尾部 padding，因此 mask 中不能出现 0 后又出现 1。
        for row in self.attention_mask_rows:
            if tuple(row) != tuple(sorted(row, reverse=True)):
                raise ValueError("attention_mask_rows must use tail padding")
        # target mask 同样保持尾部 padding 约束，便于 loss 侧按前缀有效长度处理。
        for row in self.target_attention_mask_rows:
            if tuple(row) != tuple(sorted(row, reverse=True)):
                raise ValueError("target_attention_mask_rows must use tail padding")
        # 如果显式给了 sample_indices，则其长度也必须和样本数一致。
        if self.sample_indices and len(self.sample_indices) != self.samples:
            raise ValueError("sample_indices length must match samples when provided")

    # 返回当前 batch 的总 token 数。
    @property
    def total_tokens(self) -> int:
        # 总 token 数等于样本数乘以每样本 token 数。
        return self.samples * self.tokens_per_sample

    # 判断 batch 是否携带了显式 token 行数据。
    @property
    def has_token_rows(self) -> bool:
        # 只要 token_rows 非空，就视为携带了显式 token 行。
        return bool(self.token_rows)

    # 判断 batch 是否携带输入 attention mask。
    @property
    def has_attention_mask_rows(self) -> bool:
        # 只要 attention_mask_rows 非空，就说明 token_rows 中存在显式 padding 语义。
        return bool(self.attention_mask_rows)

    # 判断 batch 是否携带目标 attention mask。
    @property
    def has_target_attention_mask_rows(self) -> bool:
        # 只要 target_attention_mask_rows 非空，就说明 target_rows 中存在显式 loss mask。
        return bool(self.target_attention_mask_rows)

    # 返回输入侧真实 token 数量。
    @property
    def valid_token_count(self) -> int:
        # 有 attention mask 时按 mask 求和；否则所有形状 token 都视为有效。
        if self.attention_mask_rows:
            return sum(int(value) for row in self.attention_mask_rows for value in row)
        return self.total_tokens

    # 返回目标侧真实 loss token 数量。
    @property
    def valid_loss_token_count(self) -> int:
        # 有 target mask 时按 mask 求和；否则回退到显式 loss_token_count 或总 token 数。
        if self.target_attention_mask_rows:
            return sum(
                int(value)
                for row in self.target_attention_mask_rows
                for value in row
            )
        return self.loss_token_count or self.total_tokens

    # 将 batch 形状序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 统一把 batch 的核心统计字段序列化成 JSON 友好的结构。
        return {
            "samples": self.samples,
            "tokens_per_sample": self.tokens_per_sample,
            "total_tokens": self.total_tokens,
            "source_kind": self.source_kind,
            "dataset_name": self.dataset_name,
            "sample_indices": list(self.sample_indices),
            "loss_token_count": self.loss_token_count,
            "has_token_rows": self.has_token_rows,
            "has_attention_mask_rows": self.has_attention_mask_rows,
            "has_target_attention_mask_rows": self.has_target_attention_mask_rows,
            "valid_token_count": self.valid_token_count,
            "valid_loss_token_count": self.valid_loss_token_count,
        }


@dataclass(slots=True, frozen=True)
class LayerBucketPlan:
    bucket_id: int
    layer_indices: tuple[int, ...]
    attention_types: tuple[str, ...]

    # 判断当前 bucket 是否包含 full attention 层。
    @property
    def contains_full_attention(self) -> bool:
        # 只要 attention_types 中出现 full_attention，就说明该 bucket 含全注意力层。
        return "full_attention" in self.attention_types

    # 将层 bucket 规划序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 把 bucket 的层索引和注意力类型列表转成可序列化结构。
        return {
            "bucket_id": self.bucket_id,
            "layer_indices": list(self.layer_indices),
            "attention_types": list(self.attention_types),
            "contains_full_attention": self.contains_full_attention,
        }


@dataclass(slots=True, frozen=True)
class ExpertWindowPlan:
    selection_strategy: str
    router_score_source: str
    active_expert_ids: tuple[int, ...]
    prefetched_expert_ids: tuple[int, ...]
    hot_expert_ids: tuple[int, ...] = ()
    prefetch_priority_expert_ids: tuple[int, ...] = ()

    # 将 expert window 规划序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 把 expert window 各类 expert 集合统一转成列表形式输出。
        return {
            "selection_strategy": self.selection_strategy,
            "router_score_source": self.router_score_source,
            "active_expert_ids": list(self.active_expert_ids),
            "prefetched_expert_ids": list(self.prefetched_expert_ids),
            "hot_expert_ids": list(self.hot_expert_ids),
            "prefetch_priority_expert_ids": list(
                self.prefetch_priority_expert_ids
            ),
        }


@dataclass(slots=True, frozen=True)
class RuntimeAction:
    name: str
    owner: str
    description: str
    metadata: dict[str, Any] = field(default_factory=dict)

    # 将运行时动作记录序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 直接输出动作名、owner、描述和附加 metadata。
        return {
            "name": self.name,
            "owner": self.owner,
            "description": self.description,
            "metadata": self.metadata,
        }


@dataclass(slots=True, frozen=True)
class ResidencyTransition:
    group_id: str
    component: str
    from_state: ResidencyState
    to_state: ResidencyState
    trigger: str
    bucket_id: int | None = None
    expert_ids: tuple[int, ...] = ()

    # 将驻留状态迁移记录序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 把状态迁移的前后状态、触发原因和附加定位信息统一导出。
        return {
            "group_id": self.group_id,
            "component": self.component,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "trigger": self.trigger,
            "bucket_id": self.bucket_id,
            "expert_ids": list(self.expert_ids),
        }


@dataclass(slots=True, frozen=True)
class ParameterShardSnapshot:
    group_id: str
    component: str
    residency_state: ResidencyState
    committed_version: int
    pending_version: int | None
    logical_params: int
    bucket_id: int | None = None
    expert_ids: tuple[int, ...] = ()
    last_touched_step: int = -1

    # 将参数分片快照序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 输出参数分片的标识、版本、逻辑规模以及最近触达信息。
        return {
            "group_id": self.group_id,
            "component": self.component,
            "residency_state": self.residency_state,
            "committed_version": self.committed_version,
            "pending_version": self.pending_version,
            "logical_params": self.logical_params,
            "bucket_id": self.bucket_id,
            "expert_ids": list(self.expert_ids),
            "last_touched_step": self.last_touched_step,
        }


@dataclass(slots=True, frozen=True)
class ParameterSourceSlice:
    tensor_name: str
    file_name: str
    layer_index: int | None
    semantic_role: str
    start_offset: int
    length: int
    tensor_shape: tuple[int, ...] = ()
    slice_shape: tuple[int, ...] = ()

    # 将参数来源切片描述序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 把来源切片的张量名、文件名、线性偏移和形状信息转成字典。
        return {
            "tensor_name": self.tensor_name,
            "file_name": self.file_name,
            "layer_index": self.layer_index,
            "semantic_role": self.semantic_role,
            "start_offset": self.start_offset,
            "length": self.length,
            "tensor_shape": list(self.tensor_shape),
            "slice_shape": list(self.slice_shape),
        }


@dataclass(slots=True, frozen=True)
class ParameterWarehouseSummary:
    total_shards: int
    nvme_cold: int
    cpu_staged: int
    gpu_active: int
    cpu_dirty: int
    dirty_shards: int

    # 将参数仓库汇总序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 输出参数仓库在不同驻留层级下的 shard 统计。
        return {
            "total_shards": self.total_shards,
            "nvme_cold": self.nvme_cold,
            "cpu_staged": self.cpu_staged,
            "gpu_active": self.gpu_active,
            "cpu_dirty": self.cpu_dirty,
            "dirty_shards": self.dirty_shards,
        }


@dataclass(slots=True, frozen=True)
class OptimizerUpdateRecord:
    group_id: str
    component: str
    step_index: int
    target_version: int
    logical_params: int
    representative_params: int
    algorithm: str
    learning_rate: float
    weight_decay: float
    state_tier: OptimizerStateTier
    offloaded_after_update: bool
    shard_update_count: int
    gradient_l2_norm: float
    parameter_l2_norm_before: float
    parameter_l2_norm_after: float

    # 将优化器更新记录序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 输出一次优化器更新涉及的版本、超参和范数统计。
        return {
            "group_id": self.group_id,
            "component": self.component,
            "step_index": self.step_index,
            "target_version": self.target_version,
            "logical_params": self.logical_params,
            "representative_params": self.representative_params,
            "algorithm": self.algorithm,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "state_tier": self.state_tier,
            "offloaded_after_update": self.offloaded_after_update,
            "shard_update_count": self.shard_update_count,
            "gradient_l2_norm": self.gradient_l2_norm,
            "parameter_l2_norm_before": self.parameter_l2_norm_before,
            "parameter_l2_norm_after": self.parameter_l2_norm_after,
        }


@dataclass(slots=True, frozen=True)
class OptimizerShardStateSnapshot:
    group_id: str
    component: str
    logical_params: int
    representative_params: int
    update_count: int
    last_committed_version: int
    last_updated_step: int
    state_tier: OptimizerStateTier
    parameter_values: tuple[float, ...] = ()
    exp_avg_values: tuple[float, ...] = ()
    exp_avg_sq_values: tuple[float, ...] = ()

    # 将优化器分片状态快照序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 把优化器分片状态及三组状态向量统一转成可序列化结构。
        return {
            "group_id": self.group_id,
            "component": self.component,
            "logical_params": self.logical_params,
            "representative_params": self.representative_params,
            "update_count": self.update_count,
            "last_committed_version": self.last_committed_version,
            "last_updated_step": self.last_updated_step,
            "state_tier": self.state_tier,
            "parameter_values": list(self.parameter_values),
            "exp_avg_values": list(self.exp_avg_values),
            "exp_avg_sq_values": list(self.exp_avg_sq_values),
        }


@dataclass(slots=True, frozen=True)
class OptimizerSummary:
    tracked_shards: int
    cpu_hot_shards: int
    nvme_cold_shards: int
    cumulative_updates_applied: int
    state_storage_dtype: str
    gradient_buffer_storage_dtype: str
    gradient_buffer_scope: str
    last_bucket_staged_gradient_bytes: int
    peak_bucket_staged_gradient_bytes: int

    # 将优化器汇总信息序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 输出优化器跟踪规模、驻留层级和梯度缓冲区统计。
        return {
            "tracked_shards": self.tracked_shards,
            "cpu_hot_shards": self.cpu_hot_shards,
            "nvme_cold_shards": self.nvme_cold_shards,
            "cumulative_updates_applied": self.cumulative_updates_applied,
            "state_storage_dtype": self.state_storage_dtype,
            "gradient_buffer_storage_dtype": self.gradient_buffer_storage_dtype,
            "gradient_buffer_scope": self.gradient_buffer_scope,
            "last_bucket_staged_gradient_bytes": self.last_bucket_staged_gradient_bytes,
            "peak_bucket_staged_gradient_bytes": self.peak_bucket_staged_gradient_bytes,
        }


@dataclass(slots=True, frozen=True)
class ParameterStoreShardSnapshot:
    group_id: str
    component: str
    logical_params: int
    representative_params: int
    resident_tier: ParameterStoreTier
    source_kind: ParameterSourceKind
    stage_count: int
    offload_count: int
    last_touched_step: int
    source_file_names: tuple[str, ...] = ()
    source_tensor_count: int = 0
    source_layout: tuple[ParameterSourceSlice, ...] = ()
    parameter_values: tuple[float, ...] = ()
    gpu_stage_count: int = 0
    gpu_release_count: int = 0

    # 将参数存储分片快照序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 把参数存储分片的来源、布局、驻留层级和参数值统一导出。
        return {
            "group_id": self.group_id,
            "component": self.component,
            "logical_params": self.logical_params,
            "representative_params": self.representative_params,
            "resident_tier": self.resident_tier,
            "source_kind": self.source_kind,
            "source_file_names": list(self.source_file_names),
            "source_tensor_count": self.source_tensor_count,
            "source_layout": [entry.to_dict() for entry in self.source_layout],
            "stage_count": self.stage_count,
            "offload_count": self.offload_count,
            "last_touched_step": self.last_touched_step,
            "parameter_values": list(self.parameter_values),
            "gpu_stage_count": self.gpu_stage_count,
            "gpu_release_count": self.gpu_release_count,
        }


@dataclass(slots=True, frozen=True)
class ParameterStoreSummary:
    tracked_shards: int
    cpu_hot_shards: int
    nvme_cold_shards: int
    cpu_hot_resident_bytes: int
    gpu_cached_shards: int
    gpu_cached_bytes: int
    quantized_shards: int
    quantized_bytes: int
    gpu_quantized_shards: int
    gpu_quantized_bytes: int
    manifest_backed_shards: int
    synthetic_seeded_shards: int
    nvme_fp32_mirror_shards: int
    transport_backed_shards: int
    source_file_count: int
    source_tensor_count: int
    cumulative_stage_ops: int
    cumulative_offload_ops: int
    cumulative_gpu_stage_ops: int
    cumulative_gpu_release_ops: int
    cumulative_quantize_ops: int
    cumulative_nvme_sync_ops: int

    # 将参数存储汇总信息序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 输出参数存储在 CPU / GPU / 量化缓存 / 来源层面的总体统计。
        return {
            "tracked_shards": self.tracked_shards,
            "cpu_hot_shards": self.cpu_hot_shards,
            "nvme_cold_shards": self.nvme_cold_shards,
            "cpu_hot_resident_bytes": self.cpu_hot_resident_bytes,
            "gpu_cached_shards": self.gpu_cached_shards,
            "gpu_cached_bytes": self.gpu_cached_bytes,
            "quantized_shards": self.quantized_shards,
            "quantized_bytes": self.quantized_bytes,
            "gpu_quantized_shards": self.gpu_quantized_shards,
            "gpu_quantized_bytes": self.gpu_quantized_bytes,
            "manifest_backed_shards": self.manifest_backed_shards,
            "synthetic_seeded_shards": self.synthetic_seeded_shards,
            "nvme_fp32_mirror_shards": self.nvme_fp32_mirror_shards,
            "transport_backed_shards": self.transport_backed_shards,
            "source_file_count": self.source_file_count,
            "source_tensor_count": self.source_tensor_count,
            "cumulative_stage_ops": self.cumulative_stage_ops,
            "cumulative_offload_ops": self.cumulative_offload_ops,
            "cumulative_gpu_stage_ops": self.cumulative_gpu_stage_ops,
            "cumulative_gpu_release_ops": self.cumulative_gpu_release_ops,
            "cumulative_quantize_ops": self.cumulative_quantize_ops,
            "cumulative_nvme_sync_ops": self.cumulative_nvme_sync_ops,
        }


@dataclass(slots=True, frozen=True)
class ParameterSourceRecord:
    group_id: str
    component: str
    source_kind: ParameterSourceKind
    file_names: tuple[str, ...]
    tensor_count: int
    resident_tier: ParameterStoreTier
    transport_backed: bool
    layer_indices: tuple[int, ...] = ()
    semantic_roles: tuple[str, ...] = ()

    # 将参数来源记录序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 输出单个 shard 的来源文件、来源类型与角色覆盖情况。
        return {
            "group_id": self.group_id,
            "component": self.component,
            "source_kind": self.source_kind,
            "file_names": list(self.file_names),
            "tensor_count": self.tensor_count,
            "resident_tier": self.resident_tier,
            "transport_backed": self.transport_backed,
            "layer_indices": list(self.layer_indices),
            "semantic_roles": list(self.semantic_roles),
        }


@dataclass(slots=True, frozen=True)
class ParameterSourceSummary:
    touched_shards: int
    manifest_backed_shards: int
    synthetic_seeded_shards: int
    nvme_fp32_mirror_shards: int
    transport_backed_shards: int
    file_count: int
    tensor_count: int
    shard_sources: tuple[ParameterSourceRecord, ...] = ()

    # 将参数来源汇总序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 输出被触达 shard 的来源聚合统计及逐 shard 详情。
        return {
            "touched_shards": self.touched_shards,
            "manifest_backed_shards": self.manifest_backed_shards,
            "synthetic_seeded_shards": self.synthetic_seeded_shards,
            "nvme_fp32_mirror_shards": self.nvme_fp32_mirror_shards,
            "transport_backed_shards": self.transport_backed_shards,
            "file_count": self.file_count,
            "tensor_count": self.tensor_count,
            "shard_sources": [record.to_dict() for record in self.shard_sources],
        }


@dataclass(slots=True, frozen=True)
class ParameterLoadRecord:
    group_id: str
    component: str
    source_kind: ParameterSourceKind
    load_path: ParameterLoadPath
    resident_tier_after_access: ParameterStoreTier

    # 将参数加载记录序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 输出一次参数访问的来源类型、加载路径和访问后驻留层级。
        return {
            "group_id": self.group_id,
            "component": self.component,
            "source_kind": self.source_kind,
            "load_path": self.load_path,
            "resident_tier_after_access": self.resident_tier_after_access,
        }


@dataclass(slots=True, frozen=True)
class ParameterLoadSummary:
    touched_shards: int
    transport_cache_loads: int
    direct_manifest_loads: int
    synthetic_seed_loads: int
    nvme_fp32_mirror_loads: int
    buffer_reuses: int
    cpu_hot_reuses: int
    records: tuple[ParameterLoadRecord, ...] = ()

    # 将参数加载汇总序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 输出本 step 参数加载在各条路径上的命中次数和明细记录。
        return {
            "touched_shards": self.touched_shards,
            "transport_cache_loads": self.transport_cache_loads,
            "direct_manifest_loads": self.direct_manifest_loads,
            "synthetic_seed_loads": self.synthetic_seed_loads,
            "nvme_fp32_mirror_loads": self.nvme_fp32_mirror_loads,
            "buffer_reuses": self.buffer_reuses,
            "cpu_hot_reuses": self.cpu_hot_reuses,
            "records": [record.to_dict() for record in self.records],
        }


@dataclass(slots=True, frozen=True)
class ParameterPrefetchSummary:
    touched_shards: int
    transport_cache_prefetches: int
    direct_manifest_prefetches: int
    synthetic_seed_prefetches: int
    nvme_fp32_mirror_prefetches: int
    buffer_reuses: int
    cpu_hot_reuses: int
    records: tuple[ParameterLoadRecord, ...] = ()

    # 将参数预取汇总序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 输出本 step 参数预取在各条路径上的命中次数和明细记录。
        return {
            "touched_shards": self.touched_shards,
            "transport_cache_prefetches": self.transport_cache_prefetches,
            "direct_manifest_prefetches": self.direct_manifest_prefetches,
            "synthetic_seed_prefetches": self.synthetic_seed_prefetches,
            "nvme_fp32_mirror_prefetches": self.nvme_fp32_mirror_prefetches,
            "buffer_reuses": self.buffer_reuses,
            "cpu_hot_reuses": self.cpu_hot_reuses,
            "records": [record.to_dict() for record in self.records],
        }


@dataclass(slots=True, frozen=True)
class TransportShardPlan:
    group_id: str
    component: str
    file_names: tuple[str, ...]
    tensor_count: int
    estimated_stage_bytes: int

    # 将传输分片规划序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 输出单个 shard 需要访问的文件集合、张量数和 stage 字节估算。
        return {
            "group_id": self.group_id,
            "component": self.component,
            "file_names": list(self.file_names),
            "tensor_count": self.tensor_count,
            "estimated_stage_bytes": self.estimated_stage_bytes,
        }


@dataclass(slots=True, frozen=True)
class TransportPlanSummary:
    manifest_available: bool
    matched_shards: int
    unmatched_shards: int
    file_count: int
    tensor_count: int
    estimated_stage_bytes: int
    model_path: str
    shard_plans: tuple[TransportShardPlan, ...] = ()

    # 将传输规划汇总序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 输出 transport 规划是否可用、命中规模和逐 shard 计划。
        return {
            "manifest_available": self.manifest_available,
            "matched_shards": self.matched_shards,
            "unmatched_shards": self.unmatched_shards,
            "file_count": self.file_count,
            "tensor_count": self.tensor_count,
            "estimated_stage_bytes": self.estimated_stage_bytes,
            "model_path": self.model_path,
            "shard_plans": [plan.to_dict() for plan in self.shard_plans],
        }


@dataclass(slots=True, frozen=True)
class TransportCachedFileSnapshot:
    file_name: str
    file_size_bytes: int
    stage_count: int
    reuse_count: int
    last_used_step: int

    # 将缓存文件快照序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 输出单个缓存文件的大小、stage / reuse 计数和最近使用步号。
        return {
            "file_name": self.file_name,
            "file_size_bytes": self.file_size_bytes,
            "stage_count": self.stage_count,
            "reuse_count": self.reuse_count,
            "last_used_step": self.last_used_step,
        }


@dataclass(slots=True, frozen=True)
class TransportBufferSnapshot:
    buffer_id: str
    buffer_kind: str
    owner_group_id: str
    capacity_bytes: int
    pinned: bool
    stage_count: int
    reuse_count: int
    last_used_step: int
    last_bucket_id: int | None = None
    last_micro_batch_id: int | None = None
    active: bool = False

    # 将传输缓冲区快照序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 输出单个传输缓冲区的容量、pin 状态和最近使用元信息。
        return {
            "buffer_id": self.buffer_id,
            "buffer_kind": self.buffer_kind,
            "owner_group_id": self.owner_group_id,
            "capacity_bytes": self.capacity_bytes,
            "pinned": self.pinned,
            "stage_count": self.stage_count,
            "reuse_count": self.reuse_count,
            "last_used_step": self.last_used_step,
            "last_bucket_id": self.last_bucket_id,
            "last_micro_batch_id": self.last_micro_batch_id,
            "active": self.active,
        }


@dataclass(slots=True, frozen=True)
class TransportExecutionSummary:
    manifest_available: bool
    requested_file_count: int
    staged_file_count: int
    reused_file_count: int
    evicted_file_count: int
    cache_hit_shards: int
    cache_miss_shards: int
    staged_bytes: int
    reused_bytes: int
    evicted_bytes: int
    cache_file_count: int
    cache_resident_bytes: int
    max_cache_bytes: int
    pinned_memory_supported: bool = False
    active_buffer_count: int = 0
    pooled_buffer_count: int = 0
    pooled_buffer_bytes: int = 0
    pinned_buffer_count: int = 0
    pinned_buffer_bytes: int = 0
    weight_stage_buffer_bytes: int = 0
    gradient_stage_buffer_bytes: int = 0
    h2d_transfer_bytes: int = 0
    d2h_transfer_bytes: int = 0
    overlap_eligible_bytes: int = 0
    released_buffer_count: int = 0
    staged_files: tuple[str, ...] = ()
    reused_files: tuple[str, ...] = ()
    evicted_files: tuple[str, ...] = ()

    # 将传输执行汇总序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 输出文件缓存、缓冲区池以及 H2D/D2H 传输层面的综合统计。
        return {
            "manifest_available": self.manifest_available,
            "requested_file_count": self.requested_file_count,
            "staged_file_count": self.staged_file_count,
            "reused_file_count": self.reused_file_count,
            "evicted_file_count": self.evicted_file_count,
            "cache_hit_shards": self.cache_hit_shards,
            "cache_miss_shards": self.cache_miss_shards,
            "staged_bytes": self.staged_bytes,
            "reused_bytes": self.reused_bytes,
            "evicted_bytes": self.evicted_bytes,
            "cache_file_count": self.cache_file_count,
            "cache_resident_bytes": self.cache_resident_bytes,
            "max_cache_bytes": self.max_cache_bytes,
            "pinned_memory_supported": self.pinned_memory_supported,
            "active_buffer_count": self.active_buffer_count,
            "pooled_buffer_count": self.pooled_buffer_count,
            "pooled_buffer_bytes": self.pooled_buffer_bytes,
            "pinned_buffer_count": self.pinned_buffer_count,
            "pinned_buffer_bytes": self.pinned_buffer_bytes,
            "weight_stage_buffer_bytes": self.weight_stage_buffer_bytes,
            "gradient_stage_buffer_bytes": self.gradient_stage_buffer_bytes,
            "h2d_transfer_bytes": self.h2d_transfer_bytes,
            "d2h_transfer_bytes": self.d2h_transfer_bytes,
            "overlap_eligible_bytes": self.overlap_eligible_bytes,
            "released_buffer_count": self.released_buffer_count,
            "staged_files": list(self.staged_files),
            "reused_files": list(self.reused_files),
            "evicted_files": list(self.evicted_files),
        }


@dataclass(slots=True, frozen=True)
class RepresentativeBucketRecord:
    bucket_id: int
    attention_types: tuple[str, ...]
    contains_full_attention: bool
    active_expert_ids: tuple[int, ...]
    semantic_layout_used: bool
    semantic_roles: tuple[str, ...]
    execution_mode: str
    loss_value: float
    non_routed_gradient_l2_norm: float
    expert_gradient_l2_norm: float
    peak_activation_bytes: int

    # 将单个 bucket 的代表性执行结果序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 输出 bucket 级执行涉及的注意力模式、expert 集合和梯度统计。
        return {
            "bucket_id": self.bucket_id,
            "attention_types": list(self.attention_types),
            "contains_full_attention": self.contains_full_attention,
            "active_expert_ids": list(self.active_expert_ids),
            "semantic_layout_used": self.semantic_layout_used,
            "semantic_roles": list(self.semantic_roles),
            "execution_mode": self.execution_mode,
            "loss_value": self.loss_value,
            "non_routed_gradient_l2_norm": self.non_routed_gradient_l2_norm,
            "expert_gradient_l2_norm": self.expert_gradient_l2_norm,
            "peak_activation_bytes": self.peak_activation_bytes,
        }


@dataclass(slots=True, frozen=True)
class RepresentativeExecutionSummary:
    executed_buckets: int
    gradient_shards: int
    total_loss: float
    max_gradient_l2_norm: float
    peak_activation_bytes: int
    peak_host_gradient_buffer_bytes: int
    gradient_buffer_storage_dtype: str
    bucket_records: tuple[RepresentativeBucketRecord, ...] = ()

    # 将代表性执行汇总序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 输出整个代表性执行过程的 loss、梯度峰值和逐 bucket 记录。
        return {
            "executed_buckets": self.executed_buckets,
            "gradient_shards": self.gradient_shards,
            "total_loss": self.total_loss,
            "max_gradient_l2_norm": self.max_gradient_l2_norm,
            "peak_activation_bytes": self.peak_activation_bytes,
            "peak_host_gradient_buffer_bytes": self.peak_host_gradient_buffer_bytes,
            "gradient_buffer_storage_dtype": self.gradient_buffer_storage_dtype,
            "bucket_records": [
                record.to_dict() for record in self.bucket_records
            ],
        }


@dataclass(slots=True, frozen=True)
class BucketStreamTrace:
    bucket_id: int
    layer_indices: tuple[int, ...]
    attention_types: tuple[str, ...]
    micro_batch_count: int
    cpu_hot_shards_before_prefetch: int
    lookahead_prefetched_bucket_ids: tuple[int, ...]
    prefetch_summary: ParameterPrefetchSummary
    load_summary: ParameterLoadSummary
    bucket_record: RepresentativeBucketRecord
    optimizer_update_count: int
    optimizer_updated_groups: tuple[str, ...]
    gradient_release_count: int
    gradients_released_immediately: bool
    host_gradient_buffer_bytes: int
    host_gradient_buffer_storage_dtype: str
    offloaded_shards_after_update: int
    cpu_hot_shards_after_update: int

    # 将 bucket 级双流轨迹序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 输出 bucket 级调度、prefetch/load、优化器更新和释放统计。
        return {
            "bucket_id": self.bucket_id,
            "layer_indices": list(self.layer_indices),
            "attention_types": list(self.attention_types),
            "micro_batch_count": self.micro_batch_count,
            "cpu_hot_shards_before_prefetch": self.cpu_hot_shards_before_prefetch,
            "lookahead_prefetched_bucket_ids": list(
                self.lookahead_prefetched_bucket_ids
            ),
            "prefetch_summary": self.prefetch_summary.to_dict(),
            "load_summary": self.load_summary.to_dict(),
            "bucket_record": self.bucket_record.to_dict(),
            "optimizer_update_count": self.optimizer_update_count,
            "optimizer_updated_groups": list(self.optimizer_updated_groups),
            "gradient_release_count": self.gradient_release_count,
            "gradients_released_immediately": self.gradients_released_immediately,
            "host_gradient_buffer_bytes": self.host_gradient_buffer_bytes,
            "host_gradient_buffer_storage_dtype": self.host_gradient_buffer_storage_dtype,
            "offloaded_shards_after_update": self.offloaded_shards_after_update,
            "cpu_hot_shards_after_update": self.cpu_hot_shards_after_update,
        }


@dataclass(slots=True, frozen=True)
class StreamOperationTrace:
    stream_name: str
    operation: str
    micro_batch_id: int
    batch: BatchShape
    bucket_id: int
    start_time_us: int
    end_time_us: int
    duration_us: int
    token_count: int
    activation_bytes: int
    update_group_count: int = 0

    # 将单条流操作轨迹序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 输出一条 compute/transfer 流操作的时序和负载统计。
        return {
            "stream_name": self.stream_name,
            "operation": self.operation,
            "micro_batch_id": self.micro_batch_id,
            "batch": self.batch.to_dict(),
            "bucket_id": self.bucket_id,
            "start_time_us": self.start_time_us,
            "end_time_us": self.end_time_us,
            "duration_us": self.duration_us,
            "token_count": self.token_count,
            "activation_bytes": self.activation_bytes,
            "update_group_count": self.update_group_count,
        }


@dataclass(slots=True, frozen=True)
class StreamOverlapSummary:
    micro_batch_count: int
    scheduled_samples: int
    scheduled_tokens: int
    compute_operation_count: int
    transfer_operation_count: int
    compute_stream_span_us: int
    transfer_stream_span_us: int
    estimated_step_makespan_us: int
    compute_wait_us: int
    transfer_idle_us: int
    max_update_lag_us: int
    overlap_ratio: float

    # 将流重叠汇总序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 输出双流重叠在时间跨度、等待时间和重叠率上的统计。
        return {
            "micro_batch_count": self.micro_batch_count,
            "scheduled_samples": self.scheduled_samples,
            "scheduled_tokens": self.scheduled_tokens,
            "compute_operation_count": self.compute_operation_count,
            "transfer_operation_count": self.transfer_operation_count,
            "compute_stream_span_us": self.compute_stream_span_us,
            "transfer_stream_span_us": self.transfer_stream_span_us,
            "estimated_step_makespan_us": self.estimated_step_makespan_us,
            "compute_wait_us": self.compute_wait_us,
            "transfer_idle_us": self.transfer_idle_us,
            "max_update_lag_us": self.max_update_lag_us,
            "overlap_ratio": self.overlap_ratio,
        }


@dataclass(slots=True, frozen=True)
class TrainingStepTrace:
    step_index: int
    batch: BatchShape
    active_expert_ids: tuple[int, ...]
    prefetched_expert_ids: tuple[int, ...]
    released_expert_ids: tuple[int, ...]
    layer_buckets: tuple[LayerBucketPlan, ...]
    scheduled_micro_batches: tuple[BatchShape, ...]
    bucket_stream_traces: tuple[BucketStreamTrace, ...]
    stream_operations: tuple[StreamOperationTrace, ...]
    actions: tuple[RuntimeAction, ...]
    static_modules_staged: bool
    expert_window_plan: ExpertWindowPlan | None = None
    residency_transitions: tuple[ResidencyTransition, ...] = ()
    residency_ending_states: dict[str, str] = field(default_factory=dict)
    parameter_shards: tuple[ParameterShardSnapshot, ...] = ()
    warehouse_summary: ParameterWarehouseSummary | None = None
    parameter_store_summary: ParameterStoreSummary | None = None
    parameter_source_summary: ParameterSourceSummary | None = None
    parameter_prefetch_summary: ParameterPrefetchSummary | None = None
    parameter_load_summary: ParameterLoadSummary | None = None
    transport_summary: TransportPlanSummary | None = None
    transport_execution_summary: TransportExecutionSummary | None = None
    stream_overlap_summary: StreamOverlapSummary | None = None
    optimizer_updates: tuple[OptimizerUpdateRecord, ...] = ()
    optimizer_summary: OptimizerSummary | None = None
    execution_summary: RepresentativeExecutionSummary | None = None

    # 将单步训练轨迹完整序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # -----------------
        # 先序列化基础 batch、expert window 和 bucket 调度信息。
        return {
            "step_index": self.step_index,
            "batch": self.batch.to_dict(),
            "active_expert_ids": list(self.active_expert_ids),
            "prefetched_expert_ids": list(self.prefetched_expert_ids),
            "released_expert_ids": list(self.released_expert_ids),
            "expert_window_plan": (
                None
                if self.expert_window_plan is None
                else self.expert_window_plan.to_dict()
            ),
            "layer_buckets": [bucket.to_dict() for bucket in self.layer_buckets],
            "scheduled_micro_batches": [
                micro_batch.to_dict()
                for micro_batch in self.scheduled_micro_batches
            ],
            "bucket_stream_traces": [
                trace.to_dict() for trace in self.bucket_stream_traces
            ],
            "stream_operations": [
                operation.to_dict() for operation in self.stream_operations
            ],
            "actions": [action.to_dict() for action in self.actions],
            "static_modules_staged": self.static_modules_staged,

            # -----------------
            # 再附加驻留、参数仓库、传输和优化器等运行时快照。
            "residency_transitions": [
                transition.to_dict() for transition in self.residency_transitions
            ],
            "residency_ending_states": self.residency_ending_states,
            "parameter_shards": [
                shard.to_dict() for shard in self.parameter_shards
            ],
            "warehouse_summary": (
                None if self.warehouse_summary is None else self.warehouse_summary.to_dict()
            ),
            "parameter_store_summary": (
                None
                if self.parameter_store_summary is None
                else self.parameter_store_summary.to_dict()
            ),
            "parameter_source_summary": (
                None
                if self.parameter_source_summary is None
                else self.parameter_source_summary.to_dict()
            ),
            "parameter_prefetch_summary": (
                None
                if self.parameter_prefetch_summary is None
                else self.parameter_prefetch_summary.to_dict()
            ),
            "parameter_load_summary": (
                None
                if self.parameter_load_summary is None
                else self.parameter_load_summary.to_dict()
            ),
            "transport_summary": (
                None if self.transport_summary is None else self.transport_summary.to_dict()
            ),
            "transport_execution_summary": (
                None
                if self.transport_execution_summary is None
                else self.transport_execution_summary.to_dict()
            ),
            "stream_overlap_summary": (
                None
                if self.stream_overlap_summary is None
                else self.stream_overlap_summary.to_dict()
            ),
            "optimizer_updates": [
                record.to_dict() for record in self.optimizer_updates
            ],
            "optimizer_summary": (
                None if self.optimizer_summary is None else self.optimizer_summary.to_dict()
            ),

            # -----------------
            # 最后挂载执行层的代表性统计。
            "execution_summary": (
                None if self.execution_summary is None else self.execution_summary.to_dict()
            ),
        }


@dataclass(slots=True, frozen=True)
class TrainingRunTrace:
    profile_name: str
    batch: BatchShape
    step_count: int
    steps: tuple[TrainingStepTrace, ...]
    resource_plan: dict[str, Any]

    # 将整段训练 run 轨迹序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 输出 profile、batch、step 列表和资源规划信息。
        return {
            "profile_name": self.profile_name,
            "batch": self.batch.to_dict(),
            "step_count": self.step_count,
            "steps": [step.to_dict() for step in self.steps],
            "resource_plan": self.resource_plan,
        }


@dataclass(slots=True, frozen=True)
class TrainingSessionTrace:
    profile_name: str
    total_steps: int
    steps: tuple[TrainingStepTrace, ...]
    average_loss: float
    max_loss: float
    peak_activation_bytes: int
    checkpoint_format: str = "training_session_checkpoint"
    checkpoint_paths: tuple[str, ...] = ()

    # 将训练会话轨迹序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 输出训练会话的总体 loss 统计、峰值激活和 checkpoint 列表。
        return {
            "profile_name": self.profile_name,
            "total_steps": self.total_steps,
            "steps": [step.to_dict() for step in self.steps],
            "average_loss": self.average_loss,
            "max_loss": self.max_loss,
            "peak_activation_bytes": self.peak_activation_bytes,
            "checkpoint_format": self.checkpoint_format,
            "checkpoint_paths": list(self.checkpoint_paths),
        }


@dataclass(slots=True, frozen=True)
class BatchPlannerCheckpoint:
    planner_kind: str
    base_samples: int
    tokens_per_sample: int
    dataset_path: str | None = None
    tokenizer_path: str | None = None
    dataset_format: str = "auto"
    dataset_text_key: str = "text"

    # 将 batch 规划器 checkpoint 序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 输出 batch 规划器的类型、基础 batch 配置和数据集定位信息。
        return {
            "planner_kind": self.planner_kind,
            "base_samples": self.base_samples,
            "tokens_per_sample": self.tokens_per_sample,
            "dataset_path": self.dataset_path,
            "tokenizer_path": self.tokenizer_path,
            "dataset_format": self.dataset_format,
            "dataset_text_key": self.dataset_text_key,
        }

    @classmethod
    # 从字典恢复 batch 规划器 checkpoint。
    def from_dict(cls, payload: dict[str, Any]) -> "BatchPlannerCheckpoint":
        # 从 payload 中逐字段恢复 batch 规划器 checkpoint，并补默认值。
        return cls(
            planner_kind=payload["planner_kind"],
            base_samples=payload["base_samples"],
            tokens_per_sample=payload["tokens_per_sample"],
            dataset_path=payload.get("dataset_path"),
            tokenizer_path=payload.get("tokenizer_path"),
            dataset_format=payload.get("dataset_format", "auto"),
            dataset_text_key=payload.get("dataset_text_key", "text"),
        )


@dataclass(slots=True, frozen=True)
class TrainingRuntimeSnapshot:
    profile_name: str
    next_step_index: int
    static_modules_staged: bool
    runtime_quantization_session_id: str
    residency_static_modules_state: ResidencyState
    residency_staged_expert_ids: tuple[int, ...]
    warehouse_shards: tuple[ParameterShardSnapshot, ...]
    parameter_store_shards: tuple[ParameterStoreShardSnapshot, ...]
    transport_cached_files: tuple[TransportCachedFileSnapshot, ...]
    transport_buffers: tuple[TransportBufferSnapshot, ...]
    optimizer_shards: tuple[OptimizerShardStateSnapshot, ...]
    cumulative_samples_processed: int
    cumulative_tokens_processed: int
    retired_window_group_ids: tuple[str, ...] = ()
    rotation_window_cache: tuple[tuple[int, tuple[int, ...]], ...] = ()
    parameter_store_cumulative_quantize_ops: int = 0
    parameter_store_cumulative_nvme_sync_ops: int = 0

    # 将训练运行时快照序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # -----------------
        # 先写入顶层运行时状态和累计计数器。
        return {
            "profile_name": self.profile_name,
            "next_step_index": self.next_step_index,
            "static_modules_staged": self.static_modules_staged,
            "runtime_quantization_session_id": self.runtime_quantization_session_id,
            "cumulative_samples_processed": self.cumulative_samples_processed,
            "cumulative_tokens_processed": self.cumulative_tokens_processed,
            "retired_window_group_ids": list(self.retired_window_group_ids),
            "residency_static_modules_state": self.residency_static_modules_state,
            "residency_staged_expert_ids": list(self.residency_staged_expert_ids),

            # -----------------
            # 再序列化仓库、参数存储、传输与优化器快照。
            "warehouse_shards": [
                shard.to_dict() for shard in self.warehouse_shards
            ],
            "parameter_store_shards": [
                shard.to_dict() for shard in self.parameter_store_shards
            ],
            "transport_cached_files": [
                file.to_dict() for file in self.transport_cached_files
            ],
            "transport_buffers": [
                buffer.to_dict() for buffer in self.transport_buffers
            ],
            "optimizer_shards": [
                shard.to_dict() for shard in self.optimizer_shards
            ],

            # -----------------
            # 最后记录轮换窗口缓存和参数存储累计指标。
            "rotation_window_cache": [
                {
                    "rotation_index": rotation_index,
                    "active_expert_ids": list(active_expert_ids),
                }
                for rotation_index, active_expert_ids in self.rotation_window_cache
            ],
            "parameter_store_cumulative_quantize_ops": (
                self.parameter_store_cumulative_quantize_ops
            ),
            "parameter_store_cumulative_nvme_sync_ops": (
                self.parameter_store_cumulative_nvme_sync_ops
            ),
        }

    # 将训练运行时快照导出为 JSON 文本。
    def to_json(self, *, indent: int = 2) -> str:
        # 统一使用排序键和可配置缩进导出稳定 JSON。
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    # 从字典恢复训练运行时快照。
    def from_dict(cls, payload: dict[str, Any]) -> "TrainingRuntimeSnapshot":
        # -----------------
        # 先恢复参数仓库、传输缓存和优化器快照对象。
        # 逐条恢复参数仓库分片快照。
        warehouse_shards = tuple(
            ParameterShardSnapshot(
                group_id=shard["group_id"],
                component=shard["component"],
                residency_state=shard["residency_state"],
                committed_version=shard["committed_version"],
                pending_version=shard["pending_version"],
                logical_params=shard["logical_params"],
                bucket_id=shard.get("bucket_id"),
                expert_ids=tuple(shard.get("expert_ids", [])),
                last_touched_step=shard.get("last_touched_step", -1),
            )
            for shard in payload.get("warehouse_shards", [])
        )
        # 先取出 parameter_store_shards 原始载荷，后面要做新旧格式兼容。
        parameter_store_raw = payload.get("parameter_store_shards", [])
        # 恢复传输缓存中的文件快照列表。
        transport_cached_files = tuple(
            TransportCachedFileSnapshot(
                file_name=file_payload["file_name"],
                file_size_bytes=file_payload["file_size_bytes"],
                stage_count=file_payload.get("stage_count", 0),
                reuse_count=file_payload.get("reuse_count", 0),
                last_used_step=file_payload.get("last_used_step", -1),
            )
            for file_payload in payload.get("transport_cached_files", [])
        )
        # 恢复传输缓冲区池快照列表。
        transport_buffers = tuple(
            TransportBufferSnapshot(
                buffer_id=buffer_payload["buffer_id"],
                buffer_kind=buffer_payload["buffer_kind"],
                owner_group_id=buffer_payload["owner_group_id"],
                capacity_bytes=buffer_payload["capacity_bytes"],
                pinned=buffer_payload.get("pinned", False),
                stage_count=buffer_payload.get("stage_count", 0),
                reuse_count=buffer_payload.get("reuse_count", 0),
                last_used_step=buffer_payload.get("last_used_step", -1),
                last_bucket_id=buffer_payload.get("last_bucket_id"),
                last_micro_batch_id=buffer_payload.get("last_micro_batch_id"),
                active=buffer_payload.get("active", False),
            )
            for buffer_payload in payload.get("transport_buffers", [])
        )
        # 恢复优化器分片状态快照列表。
        optimizer_shards = tuple(
            OptimizerShardStateSnapshot(
                group_id=shard["group_id"],
                component=shard["component"],
                logical_params=shard["logical_params"],
                representative_params=shard.get("representative_params", 0),
                update_count=shard["update_count"],
                last_committed_version=shard["last_committed_version"],
                last_updated_step=shard["last_updated_step"],
                state_tier=shard["state_tier"],
                parameter_values=tuple(shard.get("parameter_values", [])),
                exp_avg_values=tuple(shard.get("exp_avg_values", [])),
                exp_avg_sq_values=tuple(shard.get("exp_avg_sq_values", [])),
            )
            for shard in payload.get("optimizer_shards", [])
        )

        # -----------------
        # 优先从显式 parameter_store_shards 恢复；旧快照则从优化器状态兼容构造。
        if parameter_store_raw:
            # 新格式快照里直接包含 parameter_store_shards，按新格式逐条恢复。
            parameter_store_shards = tuple(
                ParameterStoreShardSnapshot(
                    group_id=shard["group_id"],
                    component=shard["component"],
                    logical_params=shard["logical_params"],
                    representative_params=shard.get("representative_params", 0),
                    resident_tier=shard.get("resident_tier", "nvme_cold"),
                    source_kind=shard.get("source_kind", "synthetic_seed"),
                    source_file_names=tuple(shard.get("source_file_names", [])),
                    source_tensor_count=shard.get("source_tensor_count", 0),
                    source_layout=tuple(
                        ParameterSourceSlice(
                            tensor_name=entry["tensor_name"],
                            file_name=entry["file_name"],
                            layer_index=entry.get("layer_index"),
                            semantic_role=entry.get("semantic_role", "unknown"),
                            start_offset=entry.get("start_offset", 0),
                            length=entry.get("length", 0),
                            tensor_shape=tuple(entry.get("tensor_shape", [])),
                            slice_shape=tuple(entry.get("slice_shape", [])),
                        )
                        for entry in shard.get("source_layout", [])
                    ),
                    stage_count=shard.get("stage_count", 0),
                    offload_count=shard.get("offload_count", 0),
                    last_touched_step=shard.get("last_touched_step", -1),
                    parameter_values=tuple(shard.get("parameter_values", [])),
                    gpu_stage_count=shard.get("gpu_stage_count", 0),
                    gpu_release_count=shard.get("gpu_release_count", 0),
                )
                for shard in parameter_store_raw
            )
        else:
            # 旧格式没有 parameter_store_shards 时，退回用优化器状态兼容构造。
            parameter_store_shards = tuple(
                ParameterStoreShardSnapshot(
                    group_id=shard.group_id,
                    component=shard.component,
                    logical_params=shard.logical_params,
                    representative_params=shard.representative_params,
                    resident_tier=shard.state_tier,
                    source_kind="synthetic_seed",
                    source_file_names=(),
                    source_tensor_count=0,
                    source_layout=(),
                    stage_count=0,
                    offload_count=0,
                    last_touched_step=shard.last_updated_step,
                    parameter_values=shard.parameter_values,
                    gpu_stage_count=0,
                    gpu_release_count=0,
                )
                for shard in optimizer_shards
                if shard.parameter_values
            )

        # -----------------
        # 组装最终运行时快照对象并返回。
        return cls(
            profile_name=payload["profile_name"],
            next_step_index=payload["next_step_index"],
            static_modules_staged=payload["static_modules_staged"],
            runtime_quantization_session_id=payload.get(
                "runtime_quantization_session_id",
                "",
            ),
            cumulative_samples_processed=payload.get(
                "cumulative_samples_processed",
                0,
            ),
            cumulative_tokens_processed=payload.get(
                "cumulative_tokens_processed",
                0,
            ),
            retired_window_group_ids=tuple(
                payload.get("retired_window_group_ids", [])
            ),
            residency_static_modules_state=payload["residency_static_modules_state"],
            residency_staged_expert_ids=tuple(
                payload.get("residency_staged_expert_ids", [])
            ),
            warehouse_shards=warehouse_shards,
            parameter_store_shards=parameter_store_shards,
            transport_cached_files=transport_cached_files,
            transport_buffers=transport_buffers,
            optimizer_shards=optimizer_shards,
            rotation_window_cache=tuple(
                (
                    int(entry["rotation_index"]),
                    tuple(entry.get("active_expert_ids", [])),
                )
                for entry in payload.get("rotation_window_cache", [])
            ),
            parameter_store_cumulative_quantize_ops=payload.get(
                "parameter_store_cumulative_quantize_ops",
                0,
            ),
            parameter_store_cumulative_nvme_sync_ops=payload.get(
                "parameter_store_cumulative_nvme_sync_ops",
                0,
            ),
        )


@dataclass(slots=True, frozen=True)
class TrainingSessionCheckpoint:
    checkpoint_kind: str
    profile_name: str
    planner: BatchPlannerCheckpoint
    runtime_snapshot: TrainingRuntimeSnapshot

    # 将训练会话 checkpoint 序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 输出 checkpoint 类型、profile 名称以及两块核心内容。
        return {
            "checkpoint_kind": self.checkpoint_kind,
            "profile_name": self.profile_name,
            "planner": self.planner.to_dict(),
            "runtime_snapshot": self.runtime_snapshot.to_dict(),
        }

    # 将训练会话 checkpoint 导出为 JSON 文本。
    def to_json(self, *, indent: int = 2) -> str:
        # 使用稳定排序键和指定缩进导出 JSON 文本。
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    # 从字典恢复训练会话 checkpoint。
    def from_dict(cls, payload: dict[str, Any]) -> "TrainingSessionCheckpoint":
        # -----------------
        # 先恢复顶层 checkpoint 类型与 profile 名称。
        # checkpoint_kind 兼容旧格式，缺省时退回标准名字。
        checkpoint_kind = payload.get(
            "checkpoint_kind",
            "training_session_checkpoint",
        )
        # 当前 checkpoint 绑定的 profile 名称必须显式存在。
        profile_name = payload["profile_name"]

        # -----------------
        # 再恢复 planner 与 runtime_snapshot 两块核心对象。
        # 先恢复 batch planner checkpoint。
        planner = BatchPlannerCheckpoint.from_dict(payload["planner"])
        # 再恢复训练运行时快照。
        runtime_snapshot = TrainingRuntimeSnapshot.from_dict(
            payload["runtime_snapshot"]
        )

        # -----------------
        # 最后组装完整的训练会话 checkpoint。
        # 从 payload 中恢复 checkpoint 元信息、planner 和 runtime_snapshot。
        return cls(
            checkpoint_kind=checkpoint_kind,
            profile_name=profile_name,
            planner=planner,
            runtime_snapshot=runtime_snapshot,
        )
