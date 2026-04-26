"""Configuration primitives for the standalone CFIE training package."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Literal
import uuid

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
TeacherOutputKind = Literal["final_only", "cumulative", "delta"]
TeacherEngineOffloadBackend = Literal["auto", "uva", "prefetch"]
PredictorModelArchitecture = Literal[
    "mlp",
    "residual_mlp",
    "query_transformer",
    "factorized",
]


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
class TeacherSamplingConfig:
    """
    描述 predictor teacher 前向捕获时使用的采样参数来源。

    这里的配置只控制 teacher 引擎为了完成一次真实 generate 请求所需的采样行为。
    若某个采样字段保持为 None，则运行时会优先从模型目录下的
    generation_config.json 读取对应默认值；若文件缺失或字段不存在，
    再退回到 vLLM/CFIE 侧的安全默认值。

    max_tokens 默认仍固定为 1，因为 teacher trace 只需要最小生成步来驱动
    请求完成与 routed experts 回传，不应该默认继承模型聊天配置里的长生成上限。
    """

    # 模型目录下用于读取默认采样参数的配置文件名。
    generation_config_filename: str = "generation_config.json"

    # teacher 采样温度；为 None 时继承 generation_config.json 中的 temperature。
    temperature: float | None = None

    # nucleus sampling 阈值；为 None 时继承 generation_config.json 中的 top_p。
    top_p: float | None = None

    # top-k 候选数量；为 None 时继承 generation_config.json 中的 top_k。
    top_k: int | None = None

    # min-p 阈值；为 None 时继承 generation_config.json 中的 min_p。
    min_p: float | None = None

    # 重复惩罚；为 None 时继承 generation_config.json 中的 repetition_penalty。
    repetition_penalty: float | None = None

    # teacher trace 每条请求只需要生成的 token 数，默认保持最小生成 1。
    max_tokens: int = 1

    # teacher capture 只消费最终输出对象，因此默认使用 final-only 输出。
    output_kind: TeacherOutputKind = "final_only"

    # 校验 teacher 采样配置的取值范围。
    def validate(self) -> "TeacherSamplingConfig":
        _require_non_empty(
            "generation_config_filename",
            self.generation_config_filename,
        )
        if self.temperature is not None:
            _require_non_negative_float("temperature", self.temperature)
        if self.top_p is not None and not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k is not None and self.top_k < -1:
            raise ValueError("top_k must be -1, 0, or a positive integer")
        if self.min_p is not None and not 0.0 <= self.min_p <= 1.0:
            raise ValueError("min_p must be in [0, 1]")
        if self.repetition_penalty is not None:
            _require_positive_float("repetition_penalty", self.repetition_penalty)
        _require_positive_int("max_tokens", self.max_tokens)
        if self.output_kind not in {"final_only", "cumulative", "delta"}:
            raise ValueError("output_kind must be final_only, cumulative, or delta")
        return self

    # dataclass 初始化后立即校验 teacher 采样配置。
    def __post_init__(self) -> None:
        self.validate()


@dataclass(slots=True)
class PredictorTeacherEngineConfig:
    gpu_memory_utilization: float = 0.6
    offload_backend: TeacherEngineOffloadBackend = "auto"
    cpu_offload_gb: float = 0.0
    moe_cpu_budget_gb: float = 0.0
    moe_cpu_min_free_gb: float = 0.0
    log_runtime_moe_cache_events: bool = False

    def validate(self) -> "PredictorTeacherEngineConfig":
        if not 0.0 < self.gpu_memory_utilization < 1.0:
            raise ValueError("gpu_memory_utilization must be in (0, 1)")
        _require_non_negative_float("cpu_offload_gb", self.cpu_offload_gb)
        _require_non_negative_float(
            "moe_cpu_budget_gb",
            self.moe_cpu_budget_gb,
        )
        _require_non_negative_float(
            "moe_cpu_min_free_gb",
            self.moe_cpu_min_free_gb,
        )
        return self

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
    """
    描述 predictor 在推理侧如何把未来专家预测结果映射成候选池与执行池的配置。

    这组配置不直接决定 predictor 网络的结构，而是规定 predictor 产出的
    future expert 预测结果要按多大的窗口解释、给每层预留多少共享候选槽位、
    最终允许多少 expert 进入执行集合，以及这些候选 expert 在运行时从哪里取。
    训练、评估、checkpoint 校验和后续部署都需要复用这一套口径，因此必须集中定义。
    """

    # 是否启用 predictor 路由链路；关闭后，后续窗口预算和候选池约束都不再生效。
    enabled: bool = True
    # predictor 一次要覆盖多少个未来层；它决定单条样本要输出多长的 future window。
    window_layers: int = 8
    # 相邻训练样本在层维度上向前推进多少层；它决定 future window 的滑动步长。
    stride_layers: int = 8
    # 整个 future window 共享的 GPU 候选 expert 槽位总数；后续会按窗口层数均分到每一层。
    shared_gpu_candidate_slots: int = 256
    # 每个未来层最终允许真正执行的 expert 数量；它对应推理侧的硬执行预算。
    executed_experts_per_layer: int = 8
    # 每个未来层允许进入候选集合的 expert 数量；它必须覆盖执行预算和额外 speculative 预算。
    candidate_experts_per_layer: int = 40
    # 候选 expert 在线补齐时优先从哪里取；它约束推理期可以使用的在线专家来源。
    online_expert_source: OnlineExpertSource = "cpu_hot_only"
    # predictor 输出的候选集合如何裁剪成最终可执行集合；它定义推理侧选择策略的解释方式。
    selection_mode: PredictorSelectionMode = "masked_candidate_topk"
    # 是否允许候选集合与 teacher 标签存在一定不完全一致；它用于放宽训练到部署之间的严格对齐要求。
    allow_candidate_mismatch: bool = True
    # teacher 监督链路默认关注的核心指标名称；它通常会和候选预算口径保持一致。
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
    model_architecture: PredictorModelArchitecture = "mlp"
    model_depth: int = 2
    model_dropout: float = 0.0
    model_num_heads: int = 8
    model_memory_tokens: int = 8
    model_ffn_multiplier: int = 4
    batch_size: int = 8
    epochs: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    examples_per_step: int = 4
    seed: int = 0

    # 校验 predictor trainer 的训练超参数。
    def validate(self) -> "PredictorTrainerConfig":
        _require_positive_int("input_summary_dim", self.input_summary_dim)
        _require_positive_int("hidden_dim", self.hidden_dim)
        _require_positive_int("model_depth", self.model_depth)
        _require_non_negative_float("model_dropout", self.model_dropout)
        if self.model_dropout >= 1.0:
            raise ValueError("model_dropout must be in [0, 1)")
        _require_positive_int("model_num_heads", self.model_num_heads)
        _require_positive_int("model_memory_tokens", self.model_memory_tokens)
        _require_positive_int("model_ffn_multiplier", self.model_ffn_multiplier)
        _require_positive_int("batch_size", self.batch_size)
        _require_positive_int("epochs", self.epochs)
        _require_positive_float("learning_rate", self.learning_rate)
        _require_non_negative_float("weight_decay", self.weight_decay)
        _require_positive_int("examples_per_step", self.examples_per_step)
        _require_non_negative_int("seed", self.seed)
        if (
                self.model_architecture == "query_transformer"
                and self.hidden_dim % self.model_num_heads != 0
        ):
            raise ValueError(
                "hidden_dim must be divisible by model_num_heads for query_transformer"
            )
        return self

    # dataclass 初始化后立即校验 predictor trainer 配置。
    def __post_init__(self) -> None:
        self.validate()


@dataclass(slots=True)
class MemoryBudgetConfig:
    """
    描述训练侧分层参数驻留的容量预算。

    这个配置只表达规划阶段可使用的资源上限，不代表运行时实时剩余容量。
    memory planner 会先从每个层级预算中扣除对应 safety margin，
    再决定参数 bucket 应该放在 GPU hot、CPU hot 还是 NVMe cold。
    """

    # GPU hot tier 的总预算，主要用于放置当前训练 step 最频繁访问的热参数。
    gpu_hot_budget_gb: float = 8.0

    # CPU hot tier 的总预算，用于承接 GPU 放不下但仍希望保持较快访问的参数。
    cpu_hot_budget_gb: float = 32.0

    # NVMe cold tier 的总预算，用于存放低频访问或可延迟加载的冷参数。
    nvme_cold_budget_gb: float = 512.0

    # GPU 侧保留的安全余量，避免 planner 把显存预算打满导致运行期 OOM。
    gpu_safety_margin_gb: float = 1.0

    # CPU 侧保留的安全余量，避免热参数缓存挤占系统运行所需内存。
    cpu_safety_margin_gb: float = 4.0

    # NVMe 侧保留的安全余量，避免冷参数缓存占满磁盘影响写入和临时文件。
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
    # ------------------------------- 项目身份与档位 -------------------------------
    # 训练包名称，用于序列化、日志与导出信息中的项目标识。
    package_name: str = "cfie_training"
    # 当前配置所属的训练档位名称，用于区分不同 profile 的默认参数集合。
    profile_name: str = "generic"

    # ------------------------------- 模型结构与模型来源 -------------------------------
    # 模型结构规格配置，定义层数、hidden 维度、专家数等训练核心形状参数。
    model_spec: ModelSpecConfig = field(default_factory=ModelSpecConfig)
    # 模型来源配置，定义模型权重目录与加载来源信息。
    model_source: ModelSourceConfig = field(default_factory=ModelSourceConfig)

    # ------------------------------- 调度与执行主线配置 -------------------------------
    # 专家轮换配置，控制每步激活专家集合与轮换策略。
    expert_rotation: ExpertRotationConfig = field(default_factory=ExpertRotationConfig)
    # bucket 调度配置，控制 micro-bucket 切分、并发与预取行为。
    bucket_schedule: BucketScheduleConfig = field(default_factory=BucketScheduleConfig)
    # 执行配置，定义设备类型、执行模式及运行期调度开关。
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    # 优化器配置，定义学习率、权重衰减等训练超参数。
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    # 资源策略配置，定义资源分层、放置与回收策略。
    resource_policy: ResourcePolicyConfig = field(default_factory=ResourcePolicyConfig)
    # 传输配置，定义分片搬运、缓存与文件分发参数。
    transport: TransportConfig = field(default_factory=TransportConfig)

    # ------------------------------- 量化与 predictor 配置 -------------------------------
    # 运行时量化配置，定义训练运行过程中可用的量化策略与位宽约束。
    runtime_quantization: RuntimeQuantizationConfig = field(
        default_factory=RuntimeQuantizationConfig
    )
    # predictor 路由配置，定义窗口层数、候选预算与执行预算等路由参数。
    predictor_routing: PredictorRoutingConfig = field(
        default_factory=PredictorRoutingConfig
    )
    # predictor 训练配置，定义 predictor 网络宽度、轮数与优化超参。
    predictor_trainer: PredictorTrainerConfig = field(
        default_factory=PredictorTrainerConfig
    )
    # teacher 采样配置，定义 trace 捕获时如何继承模型默认 generation_config。
    teacher_sampling: TeacherSamplingConfig = field(
        default_factory=TeacherSamplingConfig
    )
    teacher_engine: PredictorTeacherEngineConfig = field(
        default_factory=PredictorTeacherEngineConfig
    )

    # ------------------------------- 显存预算与状态字节配置 -------------------------------
    # 分层内存预算配置，定义 GPU-hot/CPU-hot/NVMe-cold 各层可用容量。
    memory_budget: MemoryBudgetConfig = field(default_factory=MemoryBudgetConfig)
    # 状态字节配置，定义权重、梯度、优化器状态与激活的字节口径。
    state_bytes: StateBytesConfig = field(default_factory=StateBytesConfig)

    # ------------------------------- 设计说明备注 -------------------------------
    # 训练侧架构约束说明，用于强调训练与推理解耦、调度优先级与资源取舍原则。
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
        self.teacher_sampling.validate()
        self.teacher_engine.validate()
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
        data.pop("model_targets", None)
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
        teacher_sampling = TeacherSamplingConfig(**data.pop("teacher_sampling", {}))
        teacher_engine = PredictorTeacherEngineConfig(
            **data.pop("teacher_engine", {})
        )
        memory_budget = MemoryBudgetConfig(**data.pop("memory_budget", {}))
        state_bytes = StateBytesConfig(**data.pop("state_bytes", {}))
        notes_raw = data.pop("notes", None)
        kwargs: dict[str, Any] = {}
        if notes_raw is not None:
            kwargs["notes"] = tuple(notes_raw)

        # -----------------
        # 回填剩余顶层字段，并构造最终配置对象。
        return cls(
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
            teacher_sampling=teacher_sampling,
            teacher_engine=teacher_engine,
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
