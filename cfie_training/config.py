"""Configuration primitives for the standalone CFIE training package."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Literal
import uuid

ProjectStage = Literal["development", "target"]
BucketUnit = Literal["expert", "layer", "hybrid"]
ActivationPolicy = Literal["minimal_cache", "recompute"]
Placement = Literal["cpu", "gpu"]
WeightOffloadBackend = Literal["cpu", "cpu+nvme"]
QuantizationMethod = Literal["none", "gptq"]
OptimizerAlgorithm = Literal["adamw"]
CPUStorageDType = Literal["fp32", "fp16", "bf16", "fp8_e4m3fn", "fp8_e5m2"]
GradientBufferScope = Literal["current_bucket_only", "full_model"]
PredictorSelectionMode = Literal["shadow_exact", "masked_candidate_topk"]
OnlineExpertSource = Literal["cpu_hot_only", "cpu_or_nvme"]
ExpertSelectionStrategy = Literal["round_robin", "router_hotness"]
PackDType = Literal["int32"]
QuantizedComputeViewDType = Literal["fp16", "fp32"]
ShardMaterializationMode = Literal["representative", "logical"]
LogicalCudaExecutionMode = Literal["compact_layer", "full_bucket"]


# 校验字符串字段不能为空。
def _require_non_empty(name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


# 校验整数必须为正数。
def _require_positive_int(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be >= 1")


# 校验整数必须为非负数。
def _require_non_negative_int(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


# 校验浮点数必须大于零。
def _require_positive_float(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0")


# 校验浮点数必须为非负数。
def _require_non_negative_float(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


# 校验概率参数必须落在 [0, 1) 区间。
def _require_probability(name: str, value: float) -> None:
    if value < 0 or value >= 1:
        raise ValueError(f"{name} must be in [0, 1)")


# 校验放置位置只能是 CPU 或 GPU。
def _require_placement(name: str, value: str) -> None:
    if value not in {"cpu", "gpu"}:
        raise ValueError(f"{name} must be cpu or gpu")


@dataclass(slots=True)
class ModelSpecConfig:
    architecture: str = ""
    text_model_type: str = ""
    hidden_size: int = 0
    num_hidden_layers: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    shared_expert_intermediate_size: int = 0
    full_attention_interval: int = 0
    max_position_embeddings: int = 0
    mtp_num_hidden_layers: int = 0
    attention_pattern: tuple[str, ...] = ()
    quantization: QuantizationMethod = "none"
    quant_bits: int = 0
    quant_group_size: int = 0
    quant_sym: bool = False
    quant_dynamic_exclusions: tuple[str, ...] = ()
    total_params_billion: float = 0.0

    # 判断模型结构字段是否已经被显式定义。
    def is_defined(self) -> bool:
        return bool(self.architecture or self.text_model_type or self.num_hidden_layers)

    # 校验模型结构相关字段的一致性。
    def validate(self) -> "ModelSpecConfig":
        # -----------------
        # 未定义模型结构时直接返回，允许 generic 配置懒加载。
        if not self.is_defined():
            return self

        # -----------------
        # 已定义模型结构时，逐项校验几何、注意力周期与量化摘要字段。
        _require_non_empty("architecture", self.architecture)
        _require_non_empty("text_model_type", self.text_model_type)
        _require_positive_int("hidden_size", self.hidden_size)
        _require_positive_int("num_hidden_layers", self.num_hidden_layers)
        _require_positive_int("num_attention_heads", self.num_attention_heads)
        _require_positive_int("num_key_value_heads", self.num_key_value_heads)
        _require_positive_int("num_experts", self.num_experts)
        _require_positive_int("num_experts_per_tok", self.num_experts_per_tok)
        _require_positive_int("moe_intermediate_size", self.moe_intermediate_size)
        _require_positive_int(
            "shared_expert_intermediate_size",
            self.shared_expert_intermediate_size,
        )
        _require_positive_int(
            "full_attention_interval",
            self.full_attention_interval,
        )
        _require_positive_int(
            "max_position_embeddings",
            self.max_position_embeddings,
        )
        _require_positive_int("mtp_num_hidden_layers", self.mtp_num_hidden_layers)
        if not self.attention_pattern:
            raise ValueError("attention_pattern must not be empty when model_spec is defined")
        if len(self.attention_pattern) != self.full_attention_interval:
            raise ValueError(
                "attention_pattern length must match full_attention_interval "
                "when model_spec is defined"
            )
        _require_non_negative_float("total_params_billion", self.total_params_billion)
        return self

    # dataclass 初始化后立即校验模型结构配置。
    def __post_init__(self) -> None:
        self.validate()


@dataclass(slots=True)
class ModelTargets:
    development_model: str = "Qwen3.5-35B-A3B"
    target_model: str = "Qwen3.5-122B-class-MoE"
    family: str = "qwen3.5_moe"
    stage: ProjectStage = "development"

    # 校验项目目标模型与家族标识。
    def validate(self) -> "ModelTargets":
        _require_non_empty("development_model", self.development_model)
        _require_non_empty("target_model", self.target_model)
        _require_non_empty("family", self.family)
        return self

    # dataclass 初始化后立即校验目标配置。
    def __post_init__(self) -> None:
        self.validate()


@dataclass(slots=True)
class ExpertRotationConfig:
    enabled: bool = True
    active_experts_per_step: int = 8
    rotate_every_steps: int = 1
    rotate_every_samples: int = 0
    rotate_every_tokens: int = 0
    train_shared_expert_every_step: bool = True
    retain_active_window_state_in_memory: bool = True
    selection_strategy: ExpertSelectionStrategy = "round_robin"
    cross_step_hotness_decay: float = 0.35
    next_step_score_weight: float = 0.7
    prefetch_active_overlap: int = 2

    # 校验 routed expert 轮换与预取配置。
    def validate(self) -> "ExpertRotationConfig":
        if self.enabled:
            _require_positive_int(
                "active_experts_per_step",
                self.active_experts_per_step,
            )
            _require_positive_int("rotate_every_steps", self.rotate_every_steps)
            _require_non_negative_int(
                "rotate_every_samples",
                self.rotate_every_samples,
            )
            _require_non_negative_int(
                "rotate_every_tokens",
                self.rotate_every_tokens,
            )
            if self.rotate_every_samples > 0 and self.rotate_every_tokens > 0:
                raise ValueError(
                    "rotate_every_samples and rotate_every_tokens are mutually exclusive"
                )
            if self.selection_strategy not in {
                "round_robin",
                "router_hotness",
            }:
                raise ValueError(
                    "selection_strategy must be round_robin or router_hotness"
                )
            _require_probability(
                "cross_step_hotness_decay",
                self.cross_step_hotness_decay,
            )
            if not 0.0 <= self.next_step_score_weight <= 1.0:
                raise ValueError("next_step_score_weight must be in [0, 1]")
            _require_non_negative_int(
                "prefetch_active_overlap",
                self.prefetch_active_overlap,
            )
            if self.prefetch_active_overlap > self.active_experts_per_step:
                raise ValueError(
                    "prefetch_active_overlap must be <= active_experts_per_step"
                )
        return self

    # dataclass 初始化后立即校验 expert 轮换配置。
    def __post_init__(self) -> None:
        self.validate()


@dataclass(slots=True)
class BucketScheduleConfig:
    unit: BucketUnit = "hybrid"
    max_live_buckets: int = 2
    prefetch_buckets: int = 1
    host_gradient_buffer_scope: GradientBufferScope = "current_bucket_only"
    release_gradients_immediately: bool = True
    update_immediately_after_backward: bool = True
    include_mtp_dedicated_bucket: bool = False

    # 校验 bucket 划分与预取配置。
    def validate(self) -> "BucketScheduleConfig":
        _require_positive_int("max_live_buckets", self.max_live_buckets)
        _require_non_negative_int("prefetch_buckets", self.prefetch_buckets)
        return self

    # dataclass 初始化后立即校验 bucket 调度配置。
    def __post_init__(self) -> None:
        self.validate()


@dataclass(slots=True)
class ExecutionConfig:
    compute_device: Placement = "cpu"
    optimizer_device: Placement = "cpu"
    gradient_device: Placement = "cpu"
    trainable_shard_materialization: ShardMaterializationMode = "representative"
    logical_cuda_execution_mode: LogicalCudaExecutionMode = "full_bucket"
    deterministic_cuda_execution: bool = True
    activation_policy: ActivationPolicy = "minimal_cache"
    overlap_backward_and_update: bool = True
    compute_stream_name: str = "compute"
    transfer_stream_name: str = "transfer_update"
    sample_parallelism: int = 2
    max_tokens_per_micro_batch: int = 256

    # 校验执行设备、流名称与物化模式配置。
    def validate(self) -> "ExecutionConfig":
        _require_placement("compute_device", self.compute_device)
        _require_placement("optimizer_device", self.optimizer_device)
        _require_placement("gradient_device", self.gradient_device)
        if self.trainable_shard_materialization not in {
            "representative",
            "logical",
        }:
            raise ValueError(
                "trainable_shard_materialization must be representative or logical"
            )
        if self.logical_cuda_execution_mode not in {
            "compact_layer",
            "full_bucket",
        }:
            raise ValueError(
                "logical_cuda_execution_mode must be compact_layer or full_bucket"
            )
        _require_non_empty("compute_stream_name", self.compute_stream_name)
        _require_non_empty("transfer_stream_name", self.transfer_stream_name)
        _require_positive_int("sample_parallelism", self.sample_parallelism)
        _require_positive_int(
            "max_tokens_per_micro_batch",
            self.max_tokens_per_micro_batch,
        )
        return self

    # dataclass 初始化后立即校验执行配置。
    def __post_init__(self) -> None:
        self.validate()


@dataclass(slots=True)
class ModelSourceConfig:
    model_path: str = ""
    index_filename: str = "model.safetensors.index.json"
    use_local_weight_manifest: bool = True

    # 校验模型源路径相关配置。
    def validate(self) -> "ModelSourceConfig":
        _require_non_empty("index_filename", self.index_filename)
        return self

    # dataclass 初始化后立即校验模型源配置。
    def __post_init__(self) -> None:
        self.validate()


@dataclass(slots=True)
class OptimizerConfig:
    algorithm: OptimizerAlgorithm = "adamw"
    learning_rate: float = 1e-5
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    weight_decay: float = 0.0
    cpu_state_storage_dtype: CPUStorageDType = "fp8_e4m3fn"
    gradient_buffer_storage_dtype: CPUStorageDType = "fp8_e4m3fn"
    offload_state_after_update: bool = True

    # 校验优化器超参数与状态存储配置。
    def validate(self) -> "OptimizerConfig":
        _require_positive_float("learning_rate", self.learning_rate)
        _require_probability("beta1", self.beta1)
        _require_probability("beta2", self.beta2)
        _require_positive_float("epsilon", self.epsilon)
        _require_non_negative_float("weight_decay", self.weight_decay)
        return self

    # dataclass 初始化后立即校验优化器配置。
    def __post_init__(self) -> None:
        self.validate()


@dataclass(slots=True)
class ResourcePolicyConfig:
    gpu_is_scarcest_resource: bool = True
    prioritize_memory_over_throughput: bool = True
    allow_cpu_participation: bool = True
    weight_offload_backend: WeightOffloadBackend = "cpu+nvme"

    # 校验资源策略配置。
    def validate(self) -> "ResourcePolicyConfig":
        return self

    # dataclass 初始化后立即校验资源策略配置。
    def __post_init__(self) -> None:
        self.validate()


@dataclass(slots=True)
class TransportConfig:
    max_staged_file_cache_gb: float = 2.0
    reuse_staged_files_across_steps: bool = True
    eviction_policy: str = "lru"

    # 校验传输层缓存与淘汰策略配置。
    def validate(self) -> "TransportConfig":
        _require_positive_float(
            "max_staged_file_cache_gb",
            self.max_staged_file_cache_gb,
        )
        _require_non_empty("eviction_policy", self.eviction_policy)
        return self

    # dataclass 初始化后立即校验传输配置。
    def __post_init__(self) -> None:
        self.validate()


@dataclass(slots=True)
class RuntimeQuantizationConfig:
    enabled: bool = False
    method: QuantizationMethod = "gptq"
    bits: int = 4
    group_size: int = 128
    sym: bool = True
    pack_dtype: PackDType = "int32"
    compute_view_dtype: QuantizedComputeViewDType = "fp32"
    persist_fp32_to_nvme: bool = True
    nvme_staging_dir: str = ""
    session_id: str = ""

    # 校验运行时量化配置是否满足当前实现约束。
    def validate(self) -> "RuntimeQuantizationConfig":
        if not self.enabled:
            return self
        if self.method != "gptq":
            raise ValueError("runtime quantization currently supports only gptq")
        if self.bits != 4:
            raise ValueError("runtime GPTQ quantization currently supports only 4-bit")
        _require_positive_int("group_size", self.group_size)
        if self.group_size % (32 // self.bits) != 0:
            raise ValueError("group_size must be divisible by 32 // bits")
        if self.pack_dtype != "int32":
            raise ValueError("pack_dtype must be int32")
        if self.compute_view_dtype not in {"fp16", "fp32"}:
            raise ValueError("compute_view_dtype must be fp16 or fp32")
        return self

    # 初始化运行时量化配置，并为会话补充默认 session id。
    def __post_init__(self) -> None:
        if not self.session_id:
            self.session_id = uuid.uuid4().hex
        self.validate()


@dataclass(slots=True)
class PredictorRoutingConfig:
    enabled: bool = True
    window_layers: int = 8
    stride_layers: int = 8
    shared_gpu_candidate_slots: int = 256
    executed_experts_per_layer: int = 8
    candidate_experts_per_layer: int = 40
    online_expert_source: OnlineExpertSource = "cpu_hot_only"
    selection_mode: PredictorSelectionMode = "masked_candidate_topk"
    allow_candidate_mismatch: bool = True
    teacher_metric: str = "recall@40"

    # 返回均分窗口后每层可用的 speculative expert 数。
    @property
    def speculative_experts_per_layer(self) -> int:
        return self.shared_gpu_candidate_slots // self.window_layers

    # 校验 predictor routing 的窗口与候选池配置。
    def validate(self) -> "PredictorRoutingConfig":
        if not self.enabled:
            return self
        _require_positive_int("window_layers", self.window_layers)
        _require_positive_int("stride_layers", self.stride_layers)
        _require_positive_int(
            "shared_gpu_candidate_slots",
            self.shared_gpu_candidate_slots,
        )
        _require_positive_int(
            "executed_experts_per_layer",
            self.executed_experts_per_layer,
        )
        _require_positive_int(
            "candidate_experts_per_layer",
            self.candidate_experts_per_layer,
        )
        _require_non_empty("teacher_metric", self.teacher_metric)
        if self.shared_gpu_candidate_slots % self.window_layers != 0:
            raise ValueError(
                "shared_gpu_candidate_slots must divide evenly across window_layers"
            )
        if self.candidate_experts_per_layer < self.executed_experts_per_layer:
            raise ValueError(
                "candidate_experts_per_layer must be >= executed_experts_per_layer"
            )
        expected_candidates = (
            self.executed_experts_per_layer + self.speculative_experts_per_layer
        )
        if self.candidate_experts_per_layer != expected_candidates:
            raise ValueError(
                "candidate_experts_per_layer must equal executed_experts_per_layer "
                "+ shared_gpu_candidate_slots / window_layers for the current "
                "even-split design"
            )
        return self

    # dataclass 初始化后立即校验 predictor routing 配置。
    def __post_init__(self) -> None:
        self.validate()


@dataclass(slots=True)
class PredictorTrainerConfig:
    input_summary_dim: int = 64
    hidden_dim: int = 128
    batch_size: int = 8
    epochs: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    examples_per_step: int = 4
    synthetic_trace_noise_scale: float = 0.05
    seed: int = 0

    # 校验 predictor trainer 的训练超参数。
    def validate(self) -> "PredictorTrainerConfig":
        _require_positive_int("input_summary_dim", self.input_summary_dim)
        _require_positive_int("hidden_dim", self.hidden_dim)
        _require_positive_int("batch_size", self.batch_size)
        _require_positive_int("epochs", self.epochs)
        _require_positive_float("learning_rate", self.learning_rate)
        _require_non_negative_float("weight_decay", self.weight_decay)
        _require_positive_int("examples_per_step", self.examples_per_step)
        _require_non_negative_float(
            "synthetic_trace_noise_scale",
            self.synthetic_trace_noise_scale,
        )
        _require_non_negative_int("seed", self.seed)
        return self

    # dataclass 初始化后立即校验 predictor trainer 配置。
    def __post_init__(self) -> None:
        self.validate()


@dataclass(slots=True)
class MemoryBudgetConfig:
    gpu_hot_budget_gb: float = 8.0
    cpu_hot_budget_gb: float = 32.0
    nvme_cold_budget_gb: float = 512.0
    gpu_safety_margin_gb: float = 1.0
    cpu_safety_margin_gb: float = 4.0
    nvme_safety_margin_gb: float = 16.0

    # 校验 GPU / CPU / NVMe 预算与安全边界。
    def validate(self) -> "MemoryBudgetConfig":
        _require_positive_float("gpu_hot_budget_gb", self.gpu_hot_budget_gb)
        _require_positive_float("cpu_hot_budget_gb", self.cpu_hot_budget_gb)
        _require_positive_float("nvme_cold_budget_gb", self.nvme_cold_budget_gb)
        _require_non_negative_float("gpu_safety_margin_gb", self.gpu_safety_margin_gb)
        _require_non_negative_float("cpu_safety_margin_gb", self.cpu_safety_margin_gb)
        _require_non_negative_float(
            "nvme_safety_margin_gb",
            self.nvme_safety_margin_gb,
        )
        if self.gpu_safety_margin_gb >= self.gpu_hot_budget_gb:
            raise ValueError("gpu_safety_margin_gb must be smaller than gpu_hot_budget_gb")
        if self.cpu_safety_margin_gb >= self.cpu_hot_budget_gb:
            raise ValueError("cpu_safety_margin_gb must be smaller than cpu_hot_budget_gb")
        if self.nvme_safety_margin_gb >= self.nvme_cold_budget_gb:
            raise ValueError("nvme_safety_margin_gb must be smaller than nvme_cold_budget_gb")
        return self

    # dataclass 初始化后立即校验内存预算配置。
    def __post_init__(self) -> None:
        self.validate()


@dataclass(slots=True)
class StateBytesConfig:
    device_weight_bytes_per_param: int = 2
    gradient_bytes_per_param: int = 2
    master_weight_bytes_per_param: int = 4
    optimizer_state_bytes_per_param: int = 8
    activation_bytes_per_element: int = 2
    activation_residency_multiplier: float = 2.0

    # 校验各类训练状态的单位字节配置。
    def validate(self) -> "StateBytesConfig":
        _require_positive_int(
            "device_weight_bytes_per_param",
            self.device_weight_bytes_per_param,
        )
        _require_positive_int(
            "gradient_bytes_per_param",
            self.gradient_bytes_per_param,
        )
        _require_positive_int(
            "master_weight_bytes_per_param",
            self.master_weight_bytes_per_param,
        )
        _require_positive_int(
            "optimizer_state_bytes_per_param",
            self.optimizer_state_bytes_per_param,
        )
        _require_positive_int(
            "activation_bytes_per_element",
            self.activation_bytes_per_element,
        )
        _require_positive_float(
            "activation_residency_multiplier",
            self.activation_residency_multiplier,
        )
        return self

    # dataclass 初始化后立即校验状态字节配置。
    def __post_init__(self) -> None:
        self.validate()


@dataclass(slots=True)
class TrainingProjectConfig:
    package_name: str = "cfie_training"
    profile_name: str = "generic"
    model_targets: ModelTargets = field(default_factory=ModelTargets)
    model_spec: ModelSpecConfig = field(default_factory=ModelSpecConfig)
    model_source: ModelSourceConfig = field(default_factory=ModelSourceConfig)
    expert_rotation: ExpertRotationConfig = field(default_factory=ExpertRotationConfig)
    bucket_schedule: BucketScheduleConfig = field(default_factory=BucketScheduleConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    resource_policy: ResourcePolicyConfig = field(default_factory=ResourcePolicyConfig)
    transport: TransportConfig = field(default_factory=TransportConfig)
    runtime_quantization: RuntimeQuantizationConfig = field(
        default_factory=RuntimeQuantizationConfig
    )
    predictor_routing: PredictorRoutingConfig = field(
        default_factory=PredictorRoutingConfig
    )
    predictor_trainer: PredictorTrainerConfig = field(
        default_factory=PredictorTrainerConfig
    )
    memory_budget: MemoryBudgetConfig = field(default_factory=MemoryBudgetConfig)
    state_bytes: StateBytesConfig = field(default_factory=StateBytesConfig)
    notes: tuple[str, ...] = (
        "Keep training code separate from the inference-only cfie package.",
        "Treat bucket, expert, and phase boundaries as first-class scheduling units.",
        "Prefer GPU/host memory savings over throughput or implementation convenience.",
    )

    # 校验整套训练项目配置的跨字段约束。
    def validate(self) -> "TrainingProjectConfig":
        # -----------------
        # 先逐个校验子配置对象自身是否合法。
        _require_non_empty("package_name", self.package_name)
        _require_non_empty("profile_name", self.profile_name)
        self.model_targets.validate()
        self.model_spec.validate()
        self.model_source.validate()
        self.expert_rotation.validate()
        self.bucket_schedule.validate()
        self.execution.validate()
        self.optimizer.validate()
        self.resource_policy.validate()
        self.transport.validate()
        self.runtime_quantization.validate()
        self.predictor_routing.validate()
        self.predictor_trainer.validate()
        self.memory_budget.validate()
        self.state_bytes.validate()

        # -----------------
        # 再校验跨模块的预算、expert 数量与量化一致性约束。
        if self.transport.max_staged_file_cache_gb >= self.memory_budget.cpu_hot_budget_gb:
            raise ValueError(
                "transport.max_staged_file_cache_gb must stay below cpu_hot_budget_gb"
            )
        if self.model_spec.is_defined() and self.expert_rotation.enabled:
            if self.expert_rotation.active_experts_per_step > self.model_spec.num_experts:
                raise ValueError(
                    "active_experts_per_step must not exceed model_spec.num_experts"
                )
            if (
                self.expert_rotation.active_experts_per_step
                < self.model_spec.num_experts_per_tok
            ):
                raise ValueError(
                    "active_experts_per_step must be >= model_spec.num_experts_per_tok"
                )
        if self.runtime_quantization.enabled and self.model_spec.is_defined():
            if self.model_spec.quantization != "gptq":
                raise ValueError(
                    "runtime quantization requires model_spec.quantization to be gptq"
                )
            if self.model_spec.quant_bits and (
                self.runtime_quantization.bits != self.model_spec.quant_bits
            ):
                raise ValueError(
                    "runtime quantization bits must match model_spec.quant_bits"
                )
            if self.model_spec.quant_group_size and (
                self.runtime_quantization.group_size
                != self.model_spec.quant_group_size
            ):
                raise ValueError(
                    "runtime quantization group_size must match model_spec.quant_group_size"
                )
            if self.runtime_quantization.sym != self.model_spec.quant_sym:
                raise ValueError(
                    "runtime quantization sym must match model_spec.quant_sym"
                )
        return self

    # dataclass 初始化后立即校验整套项目配置。
    def __post_init__(self) -> None:
        self.validate()

    # 将训练项目配置导出为原生字典。
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # 将训练项目配置导出为稳定排序的 JSON 文本。
    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    # 从原始字典递归构造训练项目配置对象。
    def from_dict(cls, raw: dict[str, Any]) -> "TrainingProjectConfig":
        # -----------------
        # 先解包各个子配置对象，并为运行时量化补默认值。
        data = dict(raw)
        model_targets = ModelTargets(**data.pop("model_targets", {}))
        model_spec = ModelSpecConfig(**data.pop("model_spec", {}))
        model_source = ModelSourceConfig(**data.pop("model_source", {}))
        expert_rotation = ExpertRotationConfig(**data.pop("expert_rotation", {}))
        bucket_schedule = BucketScheduleConfig(**data.pop("bucket_schedule", {}))
        execution = ExecutionConfig(**data.pop("execution", {}))
        optimizer = OptimizerConfig(**data.pop("optimizer", {}))
        resource_policy = ResourcePolicyConfig(**data.pop("resource_policy", {}))
        transport = TransportConfig(**data.pop("transport", {}))
        runtime_quantization_raw = data.pop("runtime_quantization", None)
        if runtime_quantization_raw is None:
            runtime_quantization = RuntimeQuantizationConfig(
                enabled=(
                    model_spec.is_defined()
                    and model_spec.quantization == "gptq"
                ),
                bits=model_spec.quant_bits or 4,
                group_size=model_spec.quant_group_size or 128,
                sym=model_spec.quant_sym,
            )
        else:
            runtime_quantization = RuntimeQuantizationConfig(
                **runtime_quantization_raw
            )
        predictor_routing = PredictorRoutingConfig(**data.pop("predictor_routing", {}))
        predictor_trainer = PredictorTrainerConfig(**data.pop("predictor_trainer", {}))
        memory_budget = MemoryBudgetConfig(**data.pop("memory_budget", {}))
        state_bytes = StateBytesConfig(**data.pop("state_bytes", {}))
        notes_raw = data.pop("notes", None)
        kwargs: dict[str, Any] = {}
        if notes_raw is not None:
            kwargs["notes"] = tuple(notes_raw)

        # -----------------
        # 回填剩余顶层字段，并构造最终配置对象。
        return cls(
            model_targets=model_targets,
            model_spec=model_spec,
            model_source=model_source,
            expert_rotation=expert_rotation,
            bucket_schedule=bucket_schedule,
            execution=execution,
            optimizer=optimizer,
            resource_policy=resource_policy,
            transport=transport,
            runtime_quantization=runtime_quantization,
            predictor_routing=predictor_routing,
            predictor_trainer=predictor_trainer,
            memory_budget=memory_budget,
            state_bytes=state_bytes,
            **kwargs,
            **data,
        )

    @classmethod
    # 从 JSON 文件读取并构造训练项目配置对象。
    def from_json_file(cls, path: str | Path) -> "TrainingProjectConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("training config JSON must decode to an object")
        return cls.from_dict(payload)
