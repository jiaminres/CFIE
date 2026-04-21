# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import copy
import dataclasses
import functools
import json
import sys
from collections.abc import Callable
from dataclasses import MISSING, dataclass, fields, is_dataclass
from itertools import permutations
from types import UnionType
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    TypeAlias,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
)

import huggingface_hub
import regex as re
import torch
from pydantic import TypeAdapter, ValidationError
from pydantic.fields import FieldInfo
from typing_extensions import TypeIs

import cfie.envs as envs
from cfie.config import (
    AttentionConfig,
    CacheConfig,
    CompilationConfig,
    ConfigType,
    DeviceConfig,
    ECTransferConfig,
    EPLBConfig,
    KernelConfig,
    KVEventsConfig,
    KVTransferConfig,
    LoadConfig,
    LoRAConfig,
    ModelConfig,
    MultiModalConfig,
    ObservabilityConfig,
    OffloadConfig,
    ParallelConfig,
    PoolerConfig,
    PrefetchOffloadConfig,
    ProfilerConfig,
    SchedulerConfig,
    SpeculativeConfig,
    StructuredOutputsConfig,
    UVAOffloadConfig,
    CfieConfig,
    WeightTransferConfig,
    get_attr_docs,
)
from cfie.config.cache import (
    CacheDType,
    KVOffloadingBackend,
    MambaCacheMode,
    MambaDType,
    PrefixCachingHashAlgo,
)
from cfie.config.device import Device
from cfie.config.kernel import MoEBackend
from cfie.config.lora import MaxLoRARanks
from cfie.config.model import (
    ConvertOption,
    HfOverrides,
    LogprobsMode,
    ModelDType,
    RunnerOption,
    TokenizerMode,
)
from cfie.config.multimodal import MMCacheType, MMEncoderTPMode
from cfie.config.observability import DetailedTraceModules
from cfie.config.parallel import (
    All2AllBackend,
    DataParallelBackend,
    DCPCommBackend,
    DistributedExecutorBackend,
    ExpertPlacementStrategy,
)
from cfie.config.scheduler import SchedulerPolicy
from cfie.config.utils import get_field
from cfie.config.cfie import OptimizationLevel, PerformanceMode
from cfie.logger import init_logger, suppress_logging
from cfie.platforms import CpuArchEnum, current_platform
from cfie.plugins import load_general_plugins
from cfie.ray.lazy_utils import is_in_ray_actor, is_ray_initialized
from cfie.transformers_utils.config import (
    is_interleaved,
    maybe_override_with_speculators,
)
from cfie.transformers_utils.gguf_utils import is_gguf
from cfie.transformers_utils.repo_utils import get_model_path
from cfie.transformers_utils.utils import is_cloud_storage
from cfie.utils.argparse_utils import FlexibleArgumentParser
from cfie.utils.mem_constants import GiB_bytes
from cfie.utils.network_utils import get_ip
from cfie.utils.torch_utils import resolve_kv_cache_dtype_string
from cfie.v1.attention.backends.registry import AttentionBackendEnum
from cfie.v1.sample.logits_processor import LogitsProcessor

if TYPE_CHECKING:
    from cfie.model_executor.layers.quantization import QuantizationMethods
    from cfie.model_executor.model_loader import LoadFormats
    from cfie.usage.usage_lib import UsageContext
    from cfie.v1.executor import Executor
else:
    Executor = Any
    QuantizationMethods = Any
    LoadFormats = Any
    UsageContext = Any

logger = init_logger(__name__)

# object is used to allow for special typing forms
T = TypeVar("T")
TypeHint: TypeAlias = type[Any] | object
TypeHintT: TypeAlias = type[T] | object


def parse_type(return_type: Callable[[str], T]) -> Callable[[str], T]:
    def _parse_type(val: str) -> T:
        try:
            return return_type(val)
        except ValueError as e:
            raise argparse.ArgumentTypeError(
                f"Value {val} cannot be converted to {return_type}."
            ) from e

    return _parse_type


def optional_type(return_type: Callable[[str], T]) -> Callable[[str], T | None]:
    def _optional_type(val: str) -> T | None:
        if val == "" or val == "None":
            return None
        return parse_type(return_type)(val)

    return _optional_type


def union_dict_and_str(val: str) -> str | dict[str, str] | None:
    if not re.match(r"(?s)^\s*{.*}\s*$", val):
        return str(val)
    return optional_type(json.loads)(val)


def is_type(type_hint: TypeHint, type: TypeHintT) -> TypeIs[TypeHintT]:
    """Check if the type hint is a specific type."""
    return type_hint is type or get_origin(type_hint) is type


def contains_type(type_hints: set[TypeHint], type: TypeHintT) -> bool:
    """Check if the type hints contain a specific type."""
    return any(is_type(type_hint, type) for type_hint in type_hints)


def get_type(type_hints: set[TypeHint], type: TypeHintT) -> TypeHintT:
    """Get the specific type from the type hints."""
    return next((th for th in type_hints if is_type(th, type)), None)


def literal_to_kwargs(type_hints: set[TypeHint]) -> dict[str, Any]:
    """Get the `type` and `choices` from a `Literal` type hint in `type_hints`.

    If `type_hints` also contains `str`, we use `metavar` instead of `choices`.
    """
    type_hint = get_type(type_hints, Literal)
    options = get_args(type_hint)
    option_type = type(options[0])
    if not all(isinstance(option, option_type) for option in options):
        raise ValueError(
            "All options must be of the same type. "
            f"Got {options} with types {[type(c) for c in options]}"
        )
    kwarg = "metavar" if contains_type(type_hints, str) else "choices"
    return {"type": option_type, kwarg: sorted(options)}


def collection_to_kwargs(type_hints: set[TypeHint], type: TypeHint) -> dict[str, Any]:
    type_hint = get_type(type_hints, type)
    types = get_args(type_hint)
    elem_type = types[0]

    # Handle Ellipsis
    assert all(t is elem_type for t in types if t is not Ellipsis), (
        f"All non-Ellipsis elements must be of the same type. Got {types}."
    )

    # Handle Union types
    if get_origin(elem_type) in {Union, UnionType}:
        # Union for Union[X, Y] and UnionType for X | Y
        assert str in get_args(elem_type), (
            "If element can have multiple types, one must be 'str' "
            f"(i.e. 'list[int | str]'). Got {elem_type}."
        )
        elem_type = str

    return {
        "type": elem_type,
        "nargs": "+" if type is not tuple or Ellipsis in types else len(types),
    }


def is_not_builtin(type_hint: TypeHint) -> bool:
    """Check if the class is not a built-in type."""
    return type_hint.__module__ != "builtins"


def get_type_hints(type_hint: TypeHint) -> set[TypeHint]:
    """Extract type hints from Annotated or Union type hints."""
    type_hints: set[TypeHint] = set()
    origin = get_origin(type_hint)
    args = get_args(type_hint)

    if origin is Annotated:
        type_hints.update(get_type_hints(args[0]))
    elif origin in {Union, UnionType}:
        # Union for Union[X, Y] and UnionType for X | Y
        for arg in args:
            type_hints.update(get_type_hints(arg))
    else:
        type_hints.add(type_hint)

    return type_hints


NEEDS_HELP = (
        any("--help" in arg for arg in sys.argv)  # cfie SUBCOMMAND --help
        or (argv0 := sys.argv[0]).endswith("mkdocs")  # mkdocs SUBCOMMAND
        or argv0.endswith("mkdocs/__main__.py")  # python -m mkdocs SUBCOMMAND
)


@functools.lru_cache(maxsize=30)
def _compute_kwargs(cls: ConfigType) -> dict[str, dict[str, Any]]:
    # Save time only getting attr docs if we're generating help text
    cls_docs = get_attr_docs(cls) if NEEDS_HELP else {}
    kwargs = {}
    for field in fields(cls):
        # Get the set of possible types for the field
        type_hints: set[TypeHint] = get_type_hints(field.type)

        # If the field is a dataclass, we can use the model_validate_json
        generator = (th for th in type_hints if is_dataclass(th))
        dataclass_cls = next(generator, None)

        # Get the default value of the field
        if field.default is not MISSING:
            default = field.default
            # Handle pydantic.Field defaults
            if isinstance(default, FieldInfo):
                if default.default_factory is None:
                    default = default.default
                else:
                    # CfieConfig's Fields have default_factory set to config classes.
                    # These could emit logs on init, which would be confusing.
                    with suppress_logging():
                        default = default.default_factory()  # type: ignore[call-arg]
        elif field.default_factory is not MISSING:
            default = field.default_factory()

        # Get the help text for the field
        name = field.name
        help = cls_docs.get(name, "").strip()
        # Escape % for argparse
        help = help.replace("%", "%%")

        # Initialise the kwargs dictionary for the field
        kwargs[name] = {"default": default, "help": help}

        # Set other kwargs based on the type hints
        json_tip = (
            "Should either be a valid JSON string or JSON keys passed individually."
        )
        if dataclass_cls is not None:

            def parse_dataclass(val: str, cls=dataclass_cls) -> Any:
                try:
                    return TypeAdapter(cls).validate_json(val)
                except ValidationError as e:
                    raise argparse.ArgumentTypeError(repr(e)) from e

            kwargs[name]["type"] = parse_dataclass
            kwargs[name]["help"] += f"\n\n{json_tip}"
        elif contains_type(type_hints, bool):
            # Creates --no-<name> and --<name> flags
            kwargs[name]["action"] = argparse.BooleanOptionalAction
        elif contains_type(type_hints, Literal):
            kwargs[name].update(literal_to_kwargs(type_hints))
        elif contains_type(type_hints, tuple):
            kwargs[name].update(collection_to_kwargs(type_hints, tuple))
        elif contains_type(type_hints, list):
            kwargs[name].update(collection_to_kwargs(type_hints, list))
        elif contains_type(type_hints, set):
            kwargs[name].update(collection_to_kwargs(type_hints, set))
        elif contains_type(type_hints, int):
            if name == "max_model_len":
                kwargs[name]["type"] = human_readable_int_or_auto
                kwargs[name]["help"] += f"\n\n{human_readable_int_or_auto.__doc__}"
            elif name in ("max_num_batched_tokens", "kv_cache_memory_bytes"):
                kwargs[name]["type"] = human_readable_int
                kwargs[name]["help"] += f"\n\n{human_readable_int.__doc__}"
            else:
                kwargs[name]["type"] = int
        elif contains_type(type_hints, float):
            kwargs[name]["type"] = float
        elif contains_type(type_hints, dict) and (
                contains_type(type_hints, str)
                or any(is_not_builtin(th) for th in type_hints)
        ):
            kwargs[name]["type"] = union_dict_and_str
        elif contains_type(type_hints, dict):
            kwargs[name]["type"] = parse_type(json.loads)
            kwargs[name]["help"] += f"\n\n{json_tip}"
        elif contains_type(type_hints, str) or any(
                is_not_builtin(th) for th in type_hints
        ):
            kwargs[name]["type"] = str
        else:
            raise ValueError(f"Unsupported type {type_hints} for argument {name}.")

        # If the type hint was a sequence of literals, use the helper function
        # to update the type and choices
        if get_origin(kwargs[name].get("type")) is Literal:
            kwargs[name].update(literal_to_kwargs({kwargs[name]["type"]}))

        # If None is in type_hints, make the argument optional.
        # But not if it's a bool, argparse will handle this better.
        if type(None) in type_hints and not contains_type(type_hints, bool):
            kwargs[name]["type"] = optional_type(kwargs[name]["type"])
            if kwargs[name].get("choices"):
                kwargs[name]["choices"].append("None")
    return kwargs


def get_kwargs(cls: ConfigType) -> dict[str, dict[str, Any]]:
    """Return argparse kwargs for the given Config dataclass.

    If `--help` or `mkdocs` are not present in the command line command, the
    attribute documentation will not be included in the help output.

    The heavy computation is cached via functools.lru_cache, and a deep copy
    is returned so callers can mutate the dictionary without affecting the
    cached version.
    """
    return copy.deepcopy(_compute_kwargs(cls))


@dataclass
class EngineArgs:
    """Arguments for vLLM engine."""

    # 模型标识、权重来源与 tokenizer 基础配置。
    # 模型名称、本地路径或仓库标识。
    model: str = ModelConfig.model
    # 是否在输出中返回每层路由到的专家信息。
    enable_return_routed_experts: bool = ModelConfig.enable_return_routed_experts
    # 模型权重路径，可与 model 分离指定。
    model_weights: str = ModelConfig.model_weights
    # 对外暴露的服务模型名，可为单个或多个别名。
    served_model_name: str | list[str] | None = ModelConfig.served_model_name
    # tokenizer 名称或路径。
    tokenizer: str | None = ModelConfig.tokenizer
    # 显式指定 HuggingFace config 路径。
    hf_config_path: str | None = ModelConfig.hf_config_path
    # 选择运行任务类型，如 generate、pooling 等。
    runner: RunnerOption = ModelConfig.runner
    # 模型转换策略，如是否转换为特定实现。
    convert: ConvertOption = ModelConfig.convert
    # 是否跳过 tokenizer 初始化。
    skip_tokenizer_init: bool = ModelConfig.skip_tokenizer_init
    # 是否允许直接输入 prompt embeddings。
    enable_prompt_embeds: bool = ModelConfig.enable_prompt_embeds
    # tokenizer 运行模式。
    tokenizer_mode: TokenizerMode | str = ModelConfig.tokenizer_mode
    # 是否信任远端自定义代码。
    trust_remote_code: bool = ModelConfig.trust_remote_code
    # 允许本地媒体文件读取的根路径。
    allowed_local_media_path: str = ModelConfig.allowed_local_media_path
    # 允许拉取媒体资源的域名白名单。
    allowed_media_domains: list[str] | None = ModelConfig.allowed_media_domains
    # 下载模型与权重的目标目录。
    download_dir: str | None = LoadConfig.download_dir
    # safetensors 权重的加载策略。
    safetensors_load_strategy: str = LoadConfig.safetensors_load_strategy
    # 权重加载格式，如 auto、safetensors、gguf 等。
    load_format: str | LoadFormats = LoadConfig.load_format
    # 模型配置文件格式。
    config_format: str = ModelConfig.config_format
    # 模型权重与计算使用的数据类型。
    dtype: ModelDType = ModelConfig.dtype
    # KV cache 使用的数据类型。
    kv_cache_dtype: CacheDType = CacheConfig.cache_dtype
    # 全局随机种子。
    seed: int = ModelConfig.seed
    # 允许的最大上下文长度。
    max_model_len: int = ModelConfig.max_model_len

    # CUDAGraph 与执行后端相关配置。
    # 需要捕获为 CUDAGraph 的 batch size 列表。
    cudagraph_capture_sizes: list[int] | None = (
        CompilationConfig.cudagraph_capture_sizes
    )
    # 允许捕获的最大 CUDAGraph batch size。
    max_cudagraph_capture_size: int | None = get_field(
        CompilationConfig, "max_cudagraph_capture_size"
    )
    # Note: Specifying a custom executor backend by passing a class
    # is intended for expert use only. The API may change without
    # notice.
    # 分布式执行后端或自定义 Executor 类型。
    distributed_executor_backend: (
            str | DistributedExecutorBackend | type[Executor] | None
    ) = ParallelConfig.distributed_executor_backend

    # 并行拓扑、分布式通信与数据并行配置。
    # number of P/D disaggregation (or other disaggregation) workers
    # pipeline parallel 的阶段数。
    pipeline_parallel_size: int = ParallelConfig.pipeline_parallel_size
    # 分布式主节点地址。
    master_addr: str = ParallelConfig.master_addr
    # 分布式主节点端口。
    master_port: int = ParallelConfig.master_port
    # 参与分布式的节点总数。
    nnodes: int = ParallelConfig.nnodes
    # 当前节点在多节点集群中的编号。
    node_rank: int = ParallelConfig.node_rank
    # 分布式初始化超时时间。
    distributed_timeout_seconds: int | None = ParallelConfig.distributed_timeout_seconds
    # tensor parallel 切分份数。
    tensor_parallel_size: int = ParallelConfig.tensor_parallel_size
    # prefill 阶段的 context parallel 切分份数。
    prefill_context_parallel_size: int = ParallelConfig.prefill_context_parallel_size
    # decode 阶段的 context parallel 切分份数。
    decode_context_parallel_size: int = ParallelConfig.decode_context_parallel_size
    # decode context parallel 使用的通信后端。
    dcp_comm_backend: DCPCommBackend = ParallelConfig.dcp_comm_backend
    # decode context parallel 的 KV cache 交错粒度。
    dcp_kv_cache_interleave_size: int = ParallelConfig.dcp_kv_cache_interleave_size
    # context parallel 的 KV cache 交错粒度。
    cp_kv_cache_interleave_size: int = ParallelConfig.cp_kv_cache_interleave_size
    # data parallel 副本总数。
    data_parallel_size: int = ParallelConfig.data_parallel_size
    # 当前实例所属的 data parallel rank。
    data_parallel_rank: int | None = None
    # 当前节点负责的起始 data parallel rank。
    data_parallel_start_rank: int | None = None
    # 当前节点本地运行的 data parallel 副本数。
    data_parallel_size_local: int | None = None
    # data parallel 集群头节点地址。
    data_parallel_address: str | None = None
    # data parallel RPC 通信端口。
    data_parallel_rpc_port: int | None = None
    # 是否启用 hybrid load balancing 的 data parallel 模式。
    data_parallel_hybrid_lb: bool = False
    # 是否启用外部负载均衡的 data parallel 模式。
    data_parallel_external_lb: bool = False
    # data parallel 使用的后端类型，如 mp 或 ray。
    data_parallel_backend: DataParallelBackend = ParallelConfig.data_parallel_backend
    # 是否启用 expert parallel。
    enable_expert_parallel: bool = ParallelConfig.enable_expert_parallel
    # MoE 算子实现后端。
    moe_backend: MoEBackend = KernelConfig.moe_backend
    # expert parallel 的 all-to-all 通信后端。
    all2all_backend: All2AllBackend = ParallelConfig.all2all_backend
    # 是否启用 elastic expert parallel。
    enable_elastic_ep: bool = ParallelConfig.enable_elastic_ep
    # 是否启用动态 batch 优化。
    enable_dbo: bool = ParallelConfig.enable_dbo
    # micro-batch 或 ubatch 的大小。
    ubatch_size: int = ParallelConfig.ubatch_size
    # decode 阶段触发 DBO 的 token 阈值。
    dbo_decode_token_threshold: int = ParallelConfig.dbo_decode_token_threshold
    # prefill 阶段触发 DBO 的 token 阈值。
    dbo_prefill_token_threshold: int = ParallelConfig.dbo_prefill_token_threshold
    # 是否禁用 DP 同步时的 NCCL。
    disable_nccl_for_dp_synchronization: bool | None = (
        ParallelConfig.disable_nccl_for_dp_synchronization
    )
    # EPLB 的详细配置对象。
    eplb_config: EPLBConfig = get_field(ParallelConfig, "eplb_config")
    # 是否启用 EPLB。
    enable_eplb: bool = ParallelConfig.enable_eplb
    # 专家在设备上的放置策略。
    expert_placement_strategy: ExpertPlacementStrategy = (
        ParallelConfig.expert_placement_strategy
    )
    # API 进程总数，仅内部使用。
    _api_process_count: int = ParallelConfig._api_process_count
    # 当前 API 进程编号，仅内部使用。
    _api_process_rank: int = ParallelConfig._api_process_rank
    # 并行加载权重时允许的最大 worker 数。
    max_parallel_loading_workers: int | None = (
        ParallelConfig.max_parallel_loading_workers
    )

    # KV cache、prefix cache、offload 与显存预算配置。
    block_size: int | None = None
    enable_prefix_caching: bool | None = None
    prefix_caching_hash_algo: PrefixCachingHashAlgo = (
        CacheConfig.prefix_caching_hash_algo
    )
    disable_sliding_window: bool = ModelConfig.disable_sliding_window
    disable_cascade_attn: bool = ModelConfig.disable_cascade_attn
    offload_backend: str = OffloadConfig.offload_backend
    moe_cpu_budget_gb: float = OffloadConfig.moe_cpu_budget_gb
    moe_cpu_min_free_gb: float = OffloadConfig.moe_cpu_min_free_gb
    cpu_offload_gb: float = UVAOffloadConfig.cpu_offload_gb
    cpu_offload_params: set[str] = get_field(UVAOffloadConfig, "cpu_offload_params")
    offload_group_size: int = PrefetchOffloadConfig.offload_group_size
    offload_num_in_group: int = PrefetchOffloadConfig.offload_num_in_group
    offload_prefetch_step: int = PrefetchOffloadConfig.offload_prefetch_step
    offload_params: set[str] = get_field(PrefetchOffloadConfig, "offload_params")
    gpu_memory_utilization: float = CacheConfig.gpu_memory_utilization
    kv_cache_memory_bytes: int | None = CacheConfig.kv_cache_memory_bytes

    # 调度、partial prefill、采样日志与请求并发控制配置。
    max_num_batched_tokens: int | None = None
    max_num_partial_prefills: int = SchedulerConfig.max_num_partial_prefills
    max_long_partial_prefills: int = SchedulerConfig.max_long_partial_prefills
    long_prefill_token_threshold: int = SchedulerConfig.long_prefill_token_threshold
    max_num_seqs: int | None = None
    max_logprobs: int = ModelConfig.max_logprobs
    logprobs_mode: LogprobsMode = ModelConfig.logprobs_mode
    disable_log_stats: bool = False
    aggregate_engine_logging: bool = False

    # HuggingFace revision、量化与执行细节配置。
    revision: str | None = ModelConfig.revision
    code_revision: str | None = ModelConfig.code_revision
    hf_token: bool | str | None = ModelConfig.hf_token
    hf_overrides: HfOverrides = get_field(ModelConfig, "hf_overrides")
    tokenizer_revision: str | None = ModelConfig.tokenizer_revision
    quantization: QuantizationMethods | str | None = ModelConfig.quantization
    allow_deprecated_quantization: bool = ModelConfig.allow_deprecated_quantization
    enforce_eager: bool = ModelConfig.enforce_eager
    disable_custom_all_reduce: bool = ParallelConfig.disable_custom_all_reduce

    # 多模态输入、媒体 IO 与处理器缓存配置。
    language_model_only: bool = MultiModalConfig.language_model_only
    limit_mm_per_prompt: dict[str, int | dict[str, int]] = get_field(
        MultiModalConfig, "limit_per_prompt"
    )
    enable_mm_embeds: bool = MultiModalConfig.enable_mm_embeds
    interleave_mm_strings: bool = MultiModalConfig.interleave_mm_strings
    media_io_kwargs: dict[str, dict[str, Any]] = get_field(
        MultiModalConfig, "media_io_kwargs"
    )
    mm_processor_kwargs: dict[str, Any] | None = MultiModalConfig.mm_processor_kwargs
    mm_processor_cache_gb: float = MultiModalConfig.mm_processor_cache_gb
    mm_processor_cache_type: MMCacheType | None = (
        MultiModalConfig.mm_processor_cache_type
    )
    mm_shm_cache_max_object_size_mb: int = (
        MultiModalConfig.mm_shm_cache_max_object_size_mb
    )
    mm_encoder_only: bool = MultiModalConfig.mm_encoder_only
    mm_encoder_tp_mode: MMEncoderTPMode = MultiModalConfig.mm_encoder_tp_mode
    mm_encoder_attn_backend: AttentionBackendEnum | str | None = (
        MultiModalConfig.mm_encoder_attn_backend
    )
    io_processor_plugin: str | None = None
    skip_mm_profiling: bool = MultiModalConfig.skip_mm_profiling
    video_pruning_rate: float | None = MultiModalConfig.video_pruning_rate

    # LoRA 适配器相关配置。
    # LoRA fields
    enable_lora: bool = False
    max_loras: int = LoRAConfig.max_loras
    max_lora_rank: MaxLoRARanks = LoRAConfig.max_lora_rank
    default_mm_loras: dict[str, str] | None = LoRAConfig.default_mm_loras
    fully_sharded_loras: bool = LoRAConfig.fully_sharded_loras
    max_cpu_loras: int | None = LoRAConfig.max_cpu_loras
    lora_dtype: str | torch.dtype | None = LoRAConfig.lora_dtype
    enable_tower_connector_lora: bool = LoRAConfig.enable_tower_connector_lora
    specialize_active_lora: bool = LoRAConfig.specialize_active_lora

    # 权重加载与运行时杂项配置。
    ray_workers_use_nsight: bool = ParallelConfig.ray_workers_use_nsight
    num_gpu_blocks_override: int | None = CacheConfig.num_gpu_blocks_override
    model_loader_extra_config: dict = get_field(LoadConfig, "model_loader_extra_config")
    ignore_patterns: str | list[str] = get_field(LoadConfig, "ignore_patterns")

    enable_chunked_prefill: bool | None = None
    disable_chunked_mm_input: bool = SchedulerConfig.disable_chunked_mm_input

    disable_hybrid_kv_cache_manager: bool | None = (
        SchedulerConfig.disable_hybrid_kv_cache_manager
    )

    # 结构化输出与 reasoning parser 配置。
    structured_outputs_config: StructuredOutputsConfig = get_field(
        CfieConfig, "structured_outputs_config"
    )
    reasoning_parser: str = StructuredOutputsConfig.reasoning_parser
    reasoning_parser_plugin: str | None = None

    # speculative decoding 入口配置。
    speculative_config: dict[str, Any] | None = None

    # 可观测性与 tracing 配置。
    show_hidden_metrics_for_version: str | None = (
        ObservabilityConfig.show_hidden_metrics_for_version
    )
    otlp_traces_endpoint: str | None = ObservabilityConfig.otlp_traces_endpoint
    collect_detailed_traces: list[DetailedTraceModules] | None = (
        ObservabilityConfig.collect_detailed_traces
    )
    kv_cache_metrics: bool = ObservabilityConfig.kv_cache_metrics
    kv_cache_metrics_sample: float = get_field(
        ObservabilityConfig, "kv_cache_metrics_sample"
    )
    cudagraph_metrics: bool = ObservabilityConfig.cudagraph_metrics
    enable_layerwise_nvtx_tracing: bool = (
        ObservabilityConfig.enable_layerwise_nvtx_tracing
    )
    enable_mfu_metrics: bool = ObservabilityConfig.enable_mfu_metrics
    enable_logging_iteration_details: bool = (
        ObservabilityConfig.enable_logging_iteration_details
    )
    enable_mm_processor_stats: bool = ObservabilityConfig.enable_mm_processor_stats
    scheduling_policy: SchedulerPolicy = SchedulerConfig.policy
    scheduler_cls: str | type[object] | None = SchedulerConfig.scheduler_cls

    # pooler、编译、attention/kernel 与 worker 配置。
    pooler_config: PoolerConfig | None = ModelConfig.pooler_config
    compilation_config: CompilationConfig = get_field(CfieConfig, "compilation_config")
    attention_config: AttentionConfig = get_field(CfieConfig, "attention_config")
    kernel_config: KernelConfig = get_field(CfieConfig, "kernel_config")
    enable_flashinfer_autotune: bool = get_field(
        KernelConfig, "enable_flashinfer_autotune"
    )
    worker_cls: str = ParallelConfig.worker_cls
    worker_extension_cls: str = ParallelConfig.worker_extension_cls

    # profiler、KV 传输与编码缓存相关配置。
    profiler_config: ProfilerConfig = get_field(CfieConfig, "profiler_config")

    kv_transfer_config: KVTransferConfig | None = None
    kv_events_config: KVEventsConfig | None = None

    ec_transfer_config: ECTransferConfig | None = None

    # generation config、attention backend 与模型实现细节。
    generation_config: str = ModelConfig.generation_config
    enable_sleep_mode: bool = ModelConfig.enable_sleep_mode
    override_generation_config: dict[str, Any] = get_field(
        ModelConfig, "override_generation_config"
    )
    model_impl: str = ModelConfig.model_impl
    override_attention_dtype: str | None = ModelConfig.override_attention_dtype
    attention_backend: AttentionBackendEnum | None = AttentionConfig.backend

    # Mamba/KV cache 与附加配置。
    calculate_kv_scales: bool = CacheConfig.calculate_kv_scales
    mamba_cache_dtype: MambaDType = CacheConfig.mamba_cache_dtype
    mamba_ssm_cache_dtype: MambaDType = CacheConfig.mamba_ssm_cache_dtype
    mamba_block_size: int | None = get_field(CacheConfig, "mamba_block_size")
    mamba_cache_mode: MambaCacheMode = CacheConfig.mamba_cache_mode

    additional_config: dict[str, Any] = get_field(CfieConfig, "additional_config")

    use_tqdm_on_load: bool = LoadConfig.use_tqdm_on_load
    pt_load_map_location: str | dict[str, str] = LoadConfig.pt_load_map_location

    # logits processor、异步调度与输出流控配置。
    logits_processors: list[str | type[LogitsProcessor]] | None = (
        ModelConfig.logits_processors
    )
    """Custom logitproc types"""

    async_scheduling: bool | None = SchedulerConfig.async_scheduling

    stream_interval: int = SchedulerConfig.stream_interval

    kv_sharing_fast_prefill: bool = CacheConfig.kv_sharing_fast_prefill
    optimization_level: OptimizationLevel = CfieConfig.optimization_level
    performance_mode: PerformanceMode = CfieConfig.performance_mode

    kv_offloading_size: float | None = CacheConfig.kv_offloading_size
    kv_offloading_backend: KVOffloadingBackend = CacheConfig.kv_offloading_backend
    tokens_only: bool = False

    # 关闭行为、权重传输与环境校验配置。
    shutdown_timeout: int = 0

    weight_transfer_config: WeightTransferConfig | None = get_field(
        CfieConfig,
        "weight_transfer_config",
    )

    fail_on_environ_validation: bool = False

    def __post_init__(self):
        # support `EngineArgs(compilation_config={...})`
        # without having to manually construct a
        # CompilationConfig object
        # 允许直接传入 dict 形式的 compilation_config。
        if isinstance(self.compilation_config, dict):
            self.compilation_config = CompilationConfig(**self.compilation_config)
        # 允许直接传入 dict 形式的 attention_config。
        if isinstance(self.attention_config, dict):
            self.attention_config = AttentionConfig(**self.attention_config)
        # 允许直接传入 dict 形式的 kernel_config。
        if isinstance(self.kernel_config, dict):
            self.kernel_config = KernelConfig(**self.kernel_config)
        # 允许直接传入 dict 形式的 eplb_config。
        if isinstance(self.eplb_config, dict):
            self.eplb_config = EPLBConfig(**self.eplb_config)
        # 允许直接传入 dict 形式的 weight_transfer_config。
        if isinstance(self.weight_transfer_config, dict):
            self.weight_transfer_config = WeightTransferConfig(
                **self.weight_transfer_config
            )
        # Setup plugins
        from cfie.plugins import load_general_plugins

        # 初始化通用插件，确保后续配置和注册逻辑一致。
        load_general_plugins()
        # when use hf offline,replace model and tokenizer id to local model path
        # HuggingFace 离线模式下把 model/tokenizer id 映射到本地缓存路径。
        if huggingface_hub.constants.HF_HUB_OFFLINE:
            # 先保存原始 model id 便于日志对比。
            model_id = self.model
            # 将 model id 解析成本地路径。
            self.model = get_model_path(self.model, self.revision)
            if model_id is not self.model:
                logger.info(
                    "HF_HUB_OFFLINE is True, replace model_id [%s] to model_path [%s]",
                    model_id,
                    self.model,
                )
            if self.tokenizer is not None:
                # 先保存原始 tokenizer id。
                tokenizer_id = self.tokenizer
                # 将 tokenizer id 解析成本地路径。
                self.tokenizer = get_model_path(self.tokenizer, self.tokenizer_revision)
                if tokenizer_id is not self.tokenizer:
                    logger.info(
                        "HF_HUB_OFFLINE is True, replace tokenizer_id [%s] "
                        "to tokenizer_path [%s]",
                        tokenizer_id,
                        self.tokenizer,
                    )

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        """Shared CLI arguments for vLLM engine."""

        # Model arguments
        model_kwargs = get_kwargs(ModelConfig)
        model_group = parser.add_argument_group(
            title="ModelConfig",
            description=ModelConfig.__doc__,
        )
        if not ("serve" in sys.argv[1:] and "--help" in sys.argv[1:]):
            model_group.add_argument("--model", **model_kwargs["model"])
        model_group.add_argument("--runner", **model_kwargs["runner"])
        model_group.add_argument("--convert", **model_kwargs["convert"])
        model_group.add_argument("--tokenizer", **model_kwargs["tokenizer"])
        model_group.add_argument("--tokenizer-mode", **model_kwargs["tokenizer_mode"])
        model_group.add_argument(
            "--trust-remote-code", **model_kwargs["trust_remote_code"]
        )
        model_group.add_argument("--dtype", **model_kwargs["dtype"])
        model_group.add_argument("--seed", **model_kwargs["seed"])
        model_group.add_argument("--hf-config-path", **model_kwargs["hf_config_path"])
        model_group.add_argument(
            "--allowed-local-media-path", **model_kwargs["allowed_local_media_path"]
        )
        model_group.add_argument(
            "--allowed-media-domains", **model_kwargs["allowed_media_domains"]
        )
        model_group.add_argument("--revision", **model_kwargs["revision"])
        model_group.add_argument("--code-revision", **model_kwargs["code_revision"])
        model_group.add_argument(
            "--tokenizer-revision", **model_kwargs["tokenizer_revision"]
        )
        model_group.add_argument("--max-model-len", **model_kwargs["max_model_len"])
        model_group.add_argument("--quantization", "-q", **model_kwargs["quantization"])
        model_group.add_argument(
            "--allow-deprecated-quantization",
            **model_kwargs["allow_deprecated_quantization"],
        )
        model_group.add_argument("--enforce-eager", **model_kwargs["enforce_eager"])
        model_group.add_argument(
            "--enable-return-routed-experts",
            **model_kwargs["enable_return_routed_experts"],
        )
        model_group.add_argument("--max-logprobs", **model_kwargs["max_logprobs"])
        model_group.add_argument("--logprobs-mode", **model_kwargs["logprobs_mode"])
        model_group.add_argument(
            "--disable-sliding-window", **model_kwargs["disable_sliding_window"]
        )
        model_group.add_argument(
            "--disable-cascade-attn", **model_kwargs["disable_cascade_attn"]
        )
        model_group.add_argument(
            "--skip-tokenizer-init", **model_kwargs["skip_tokenizer_init"]
        )
        model_group.add_argument(
            "--enable-prompt-embeds", **model_kwargs["enable_prompt_embeds"]
        )
        model_group.add_argument(
            "--served-model-name", **model_kwargs["served_model_name"]
        )
        model_group.add_argument("--config-format", **model_kwargs["config_format"])
        # This one is a special case because it can bool
        # or str. TODO: Handle this in get_kwargs
        model_group.add_argument(
            "--hf-token",
            type=str,
            nargs="?",
            const=True,
            default=model_kwargs["hf_token"]["default"],
            help=model_kwargs["hf_token"]["help"],
        )
        model_group.add_argument("--hf-overrides", **model_kwargs["hf_overrides"])
        model_group.add_argument("--pooler-config", **model_kwargs["pooler_config"])
        model_group.add_argument(
            "--generation-config", **model_kwargs["generation_config"]
        )
        model_group.add_argument(
            "--override-generation-config", **model_kwargs["override_generation_config"]
        )
        model_group.add_argument(
            "--enable-sleep-mode", **model_kwargs["enable_sleep_mode"]
        )
        model_group.add_argument("--model-impl", **model_kwargs["model_impl"])
        model_group.add_argument(
            "--override-attention-dtype", **model_kwargs["override_attention_dtype"]
        )
        model_group.add_argument(
            "--logits-processors", **model_kwargs["logits_processors"]
        )
        model_group.add_argument(
            "--io-processor-plugin", **model_kwargs["io_processor_plugin"]
        )

        # Model loading arguments
        load_kwargs = get_kwargs(LoadConfig)
        load_group = parser.add_argument_group(
            title="LoadConfig",
            description=LoadConfig.__doc__,
        )
        load_group.add_argument("--load-format", **load_kwargs["load_format"])
        load_group.add_argument("--download-dir", **load_kwargs["download_dir"])
        load_group.add_argument(
            "--safetensors-load-strategy", **load_kwargs["safetensors_load_strategy"]
        )
        load_group.add_argument(
            "--model-loader-extra-config", **load_kwargs["model_loader_extra_config"]
        )
        load_group.add_argument("--ignore-patterns", **load_kwargs["ignore_patterns"])
        load_group.add_argument("--use-tqdm-on-load", **load_kwargs["use_tqdm_on_load"])
        load_group.add_argument(
            "--pt-load-map-location", **load_kwargs["pt_load_map_location"]
        )

        # Attention arguments
        attention_kwargs = get_kwargs(AttentionConfig)
        attention_group = parser.add_argument_group(
            title="AttentionConfig",
            description=AttentionConfig.__doc__,
        )
        attention_group.add_argument(
            "--attention-backend", **attention_kwargs["backend"]
        )

        # Structured outputs arguments
        structured_outputs_kwargs = get_kwargs(StructuredOutputsConfig)
        structured_outputs_group = parser.add_argument_group(
            title="StructuredOutputsConfig",
            description=StructuredOutputsConfig.__doc__,
        )
        structured_outputs_group.add_argument(
            "--reasoning-parser",
            # Choices need to be validated after parsing to include plugins
            **structured_outputs_kwargs["reasoning_parser"],
        )
        structured_outputs_group.add_argument(
            "--reasoning-parser-plugin",
            **structured_outputs_kwargs["reasoning_parser_plugin"],
        )

        # Parallel arguments
        parallel_kwargs = get_kwargs(ParallelConfig)
        parallel_group = parser.add_argument_group(
            title="ParallelConfig",
            description=ParallelConfig.__doc__,
        )
        parallel_group.add_argument(
            "--distributed-executor-backend",
            **parallel_kwargs["distributed_executor_backend"],
        )
        parallel_group.add_argument(
            "--pipeline-parallel-size",
            "-pp",
            **parallel_kwargs["pipeline_parallel_size"],
        )
        parallel_group.add_argument("--master-addr", **parallel_kwargs["master_addr"])
        parallel_group.add_argument("--master-port", **parallel_kwargs["master_port"])
        parallel_group.add_argument("--nnodes", "-n", **parallel_kwargs["nnodes"])
        parallel_group.add_argument("--node-rank", "-r", **parallel_kwargs["node_rank"])
        parallel_group.add_argument(
            "--distributed-timeout-seconds",
            **parallel_kwargs["distributed_timeout_seconds"],
        )
        parallel_group.add_argument(
            "--tensor-parallel-size", "-tp", **parallel_kwargs["tensor_parallel_size"]
        )
        parallel_group.add_argument(
            "--decode-context-parallel-size",
            "-dcp",
            **parallel_kwargs["decode_context_parallel_size"],
        )
        parallel_group.add_argument(
            "--dcp-comm-backend",
            **parallel_kwargs["dcp_comm_backend"],
        )
        parallel_group.add_argument(
            "--dcp-kv-cache-interleave-size",
            **parallel_kwargs["dcp_kv_cache_interleave_size"],
        )
        parallel_group.add_argument(
            "--cp-kv-cache-interleave-size",
            **parallel_kwargs["cp_kv_cache_interleave_size"],
        )
        parallel_group.add_argument(
            "--prefill-context-parallel-size",
            "-pcp",
            **parallel_kwargs["prefill_context_parallel_size"],
        )
        parallel_group.add_argument(
            "--data-parallel-size", "-dp", **parallel_kwargs["data_parallel_size"]
        )
        parallel_group.add_argument(
            "--data-parallel-rank",
            "-dpn",
            type=int,
            help="Data parallel rank of this instance. "
                 "When set, enables external load balancer mode.",
        )
        parallel_group.add_argument(
            "--data-parallel-start-rank",
            "-dpr",
            type=int,
            help="Starting data parallel rank for secondary nodes.",
        )
        parallel_group.add_argument(
            "--data-parallel-size-local",
            "-dpl",
            type=int,
            help="Number of data parallel replicas to run on this node.",
        )
        parallel_group.add_argument(
            "--data-parallel-address",
            "-dpa",
            type=str,
            help="Address of data parallel cluster head-node.",
        )
        parallel_group.add_argument(
            "--data-parallel-rpc-port",
            "-dpp",
            type=int,
            help="Port for data parallel RPC communication.",
        )
        parallel_group.add_argument(
            "--data-parallel-backend",
            "-dpb",
            type=str,
            default="mp",
            help='Backend for data parallel, either "mp" or "ray".',
        )
        parallel_group.add_argument(
            "--data-parallel-hybrid-lb",
            "-dph",
            **parallel_kwargs["data_parallel_hybrid_lb"],
        )
        parallel_group.add_argument(
            "--data-parallel-external-lb",
            "-dpe",
            **parallel_kwargs["data_parallel_external_lb"],
        )
        parallel_group.add_argument(
            "--enable-expert-parallel",
            "-ep",
            **parallel_kwargs["enable_expert_parallel"],
        )
        parallel_group.add_argument(
            "--all2all-backend", **parallel_kwargs["all2all_backend"]
        )
        parallel_group.add_argument("--enable-dbo", **parallel_kwargs["enable_dbo"])
        parallel_group.add_argument(
            "--ubatch-size",
            **parallel_kwargs["ubatch_size"],
        )
        parallel_group.add_argument(
            "--enable-elastic-ep", **parallel_kwargs["enable_elastic_ep"]
        )
        parallel_group.add_argument(
            "--dbo-decode-token-threshold",
            **parallel_kwargs["dbo_decode_token_threshold"],
        )
        parallel_group.add_argument(
            "--dbo-prefill-token-threshold",
            **parallel_kwargs["dbo_prefill_token_threshold"],
        )
        parallel_group.add_argument(
            "--disable-nccl-for-dp-synchronization",
            **parallel_kwargs["disable_nccl_for_dp_synchronization"],
        )
        parallel_group.add_argument("--enable-eplb", **parallel_kwargs["enable_eplb"])
        parallel_group.add_argument("--eplb-config", **parallel_kwargs["eplb_config"])
        parallel_group.add_argument(
            "--expert-placement-strategy",
            **parallel_kwargs["expert_placement_strategy"],
        )

        parallel_group.add_argument(
            "--max-parallel-loading-workers",
            **parallel_kwargs["max_parallel_loading_workers"],
        )
        parallel_group.add_argument(
            "--ray-workers-use-nsight", **parallel_kwargs["ray_workers_use_nsight"]
        )
        parallel_group.add_argument(
            "--disable-custom-all-reduce",
            **parallel_kwargs["disable_custom_all_reduce"],
        )
        parallel_group.add_argument("--worker-cls", **parallel_kwargs["worker_cls"])
        parallel_group.add_argument(
            "--worker-extension-cls", **parallel_kwargs["worker_extension_cls"]
        )

        # KV cache arguments
        cache_kwargs = get_kwargs(CacheConfig)
        cache_group = parser.add_argument_group(
            title="CacheConfig",
            description=CacheConfig.__doc__,
        )
        cache_group.add_argument("--block-size", **cache_kwargs["block_size"])
        cache_group.add_argument(
            "--gpu-memory-utilization", **cache_kwargs["gpu_memory_utilization"]
        )
        cache_group.add_argument(
            "--kv-cache-memory-bytes", **cache_kwargs["kv_cache_memory_bytes"]
        )
        cache_group.add_argument("--kv-cache-dtype", **cache_kwargs["cache_dtype"])
        cache_group.add_argument(
            "--num-gpu-blocks-override", **cache_kwargs["num_gpu_blocks_override"]
        )
        cache_group.add_argument(
            "--enable-prefix-caching",
            **{
                **cache_kwargs["enable_prefix_caching"],
                "default": None,
            },
        )
        cache_group.add_argument(
            "--prefix-caching-hash-algo", **cache_kwargs["prefix_caching_hash_algo"]
        )
        cache_group.add_argument(
            "--calculate-kv-scales", **cache_kwargs["calculate_kv_scales"]
        )
        cache_group.add_argument(
            "--kv-sharing-fast-prefill", **cache_kwargs["kv_sharing_fast_prefill"]
        )
        cache_group.add_argument(
            "--mamba-cache-dtype", **cache_kwargs["mamba_cache_dtype"]
        )
        cache_group.add_argument(
            "--mamba-ssm-cache-dtype", **cache_kwargs["mamba_ssm_cache_dtype"]
        )
        cache_group.add_argument(
            "--mamba-block-size", **cache_kwargs["mamba_block_size"]
        )
        cache_group.add_argument(
            "--mamba-cache-mode", **cache_kwargs["mamba_cache_mode"]
        )
        cache_group.add_argument(
            "--kv-offloading-size", **cache_kwargs["kv_offloading_size"]
        )
        cache_group.add_argument(
            "--kv-offloading-backend", **cache_kwargs["kv_offloading_backend"]
        )

        # Model weight offload related configs
        offload_kwargs = get_kwargs(OffloadConfig)
        uva_kwargs = get_kwargs(UVAOffloadConfig)
        prefetch_kwargs = get_kwargs(PrefetchOffloadConfig)
        offload_group = parser.add_argument_group(
            title="OffloadConfig",
            description=OffloadConfig.__doc__,
        )
        offload_group.add_argument(
            "--offload-backend", **offload_kwargs["offload_backend"]
        )
        offload_group.add_argument(
            "--moe-cpu-budget-gb", **offload_kwargs["moe_cpu_budget_gb"]
        )
        offload_group.add_argument(
            "--moe-cpu-min-free-gb", **offload_kwargs["moe_cpu_min_free_gb"]
        )
        offload_group.add_argument("--cpu-offload-gb", **uva_kwargs["cpu_offload_gb"])
        offload_group.add_argument(
            "--cpu-offload-params", **uva_kwargs["cpu_offload_params"]
        )
        offload_group.add_argument(
            "--offload-group-size",
            **prefetch_kwargs["offload_group_size"],
        )
        offload_group.add_argument(
            "--offload-num-in-group",
            **prefetch_kwargs["offload_num_in_group"],
        )
        offload_group.add_argument(
            "--offload-prefetch-step",
            **prefetch_kwargs["offload_prefetch_step"],
        )
        offload_group.add_argument(
            "--offload-params", **prefetch_kwargs["offload_params"]
        )

        # Multimodal related configs
        multimodal_kwargs = get_kwargs(MultiModalConfig)
        multimodal_group = parser.add_argument_group(
            title="MultiModalConfig",
            description=MultiModalConfig.__doc__,
        )
        multimodal_group.add_argument(
            "--language-model-only", **multimodal_kwargs["language_model_only"]
        )
        multimodal_group.add_argument(
            "--limit-mm-per-prompt", **multimodal_kwargs["limit_per_prompt"]
        )
        multimodal_group.add_argument(
            "--enable-mm-embeds", **multimodal_kwargs["enable_mm_embeds"]
        )
        multimodal_group.add_argument(
            "--media-io-kwargs", **multimodal_kwargs["media_io_kwargs"]
        )
        multimodal_group.add_argument(
            "--mm-processor-kwargs", **multimodal_kwargs["mm_processor_kwargs"]
        )
        multimodal_group.add_argument(
            "--mm-processor-cache-gb", **multimodal_kwargs["mm_processor_cache_gb"]
        )
        multimodal_group.add_argument(
            "--mm-processor-cache-type", **multimodal_kwargs["mm_processor_cache_type"]
        )
        multimodal_group.add_argument(
            "--mm-shm-cache-max-object-size-mb",
            **multimodal_kwargs["mm_shm_cache_max_object_size_mb"],
        )
        multimodal_group.add_argument(
            "--mm-encoder-only", **multimodal_kwargs["mm_encoder_only"]
        )
        multimodal_group.add_argument(
            "--mm-encoder-tp-mode", **multimodal_kwargs["mm_encoder_tp_mode"]
        )
        multimodal_group.add_argument(
            "--mm-encoder-attn-backend",
            **multimodal_kwargs["mm_encoder_attn_backend"],
        )
        multimodal_group.add_argument(
            "--interleave-mm-strings", **multimodal_kwargs["interleave_mm_strings"]
        )
        multimodal_group.add_argument(
            "--skip-mm-profiling", **multimodal_kwargs["skip_mm_profiling"]
        )

        multimodal_group.add_argument(
            "--video-pruning-rate", **multimodal_kwargs["video_pruning_rate"]
        )

        # LoRA related configs
        lora_kwargs = get_kwargs(LoRAConfig)
        lora_group = parser.add_argument_group(
            title="LoRAConfig",
            description=LoRAConfig.__doc__,
        )
        lora_group.add_argument(
            "--enable-lora",
            action=argparse.BooleanOptionalAction,
            help="If True, enable handling of LoRA adapters.",
        )
        lora_group.add_argument("--max-loras", **lora_kwargs["max_loras"])
        lora_group.add_argument("--max-lora-rank", **lora_kwargs["max_lora_rank"])
        lora_group.add_argument(
            "--lora-dtype",
            **lora_kwargs["lora_dtype"],
        )
        lora_group.add_argument(
            "--enable-tower-connector-lora",
            **lora_kwargs["enable_tower_connector_lora"],
        )
        lora_group.add_argument("--max-cpu-loras", **lora_kwargs["max_cpu_loras"])
        lora_group.add_argument(
            "--fully-sharded-loras", **lora_kwargs["fully_sharded_loras"]
        )
        lora_group.add_argument("--default-mm-loras", **lora_kwargs["default_mm_loras"])
        lora_group.add_argument(
            "--specialize-active-lora", **lora_kwargs["specialize_active_lora"]
        )

        # Observability arguments
        observability_kwargs = get_kwargs(ObservabilityConfig)
        observability_group = parser.add_argument_group(
            title="ObservabilityConfig",
            description=ObservabilityConfig.__doc__,
        )
        observability_group.add_argument(
            "--show-hidden-metrics-for-version",
            **observability_kwargs["show_hidden_metrics_for_version"],
        )
        observability_group.add_argument(
            "--otlp-traces-endpoint", **observability_kwargs["otlp_traces_endpoint"]
        )
        # TODO: generalise this special case
        choices = observability_kwargs["collect_detailed_traces"]["choices"]
        metavar = f"{{{','.join(choices)}}}"
        observability_kwargs["collect_detailed_traces"]["metavar"] = metavar
        observability_kwargs["collect_detailed_traces"]["choices"] += [
            ",".join(p) for p in permutations(get_args(DetailedTraceModules), r=2)
        ]
        observability_group.add_argument(
            "--collect-detailed-traces",
            **observability_kwargs["collect_detailed_traces"],
        )
        observability_group.add_argument(
            "--kv-cache-metrics", **observability_kwargs["kv_cache_metrics"]
        )
        observability_group.add_argument(
            "--kv-cache-metrics-sample",
            **observability_kwargs["kv_cache_metrics_sample"],
        )
        observability_group.add_argument(
            "--cudagraph-metrics",
            **observability_kwargs["cudagraph_metrics"],
        )
        observability_group.add_argument(
            "--enable-layerwise-nvtx-tracing",
            **observability_kwargs["enable_layerwise_nvtx_tracing"],
        )
        observability_group.add_argument(
            "--enable-mfu-metrics",
            **observability_kwargs["enable_mfu_metrics"],
        )
        observability_group.add_argument(
            "--enable-logging-iteration-details",
            **observability_kwargs["enable_logging_iteration_details"],
        )

        # SchedulerConfig 参数统一在这里注册，供 CLI 显式覆盖调度行为。
        scheduler_kwargs = get_kwargs(SchedulerConfig)
        scheduler_group = parser.add_argument_group(
            title="SchedulerConfig",
            description=SchedulerConfig.__doc__,
        )
        scheduler_group.add_argument(
            "--max-num-batched-tokens",
            **{
                **scheduler_kwargs["max_num_batched_tokens"],
                "default": None,
            },
        )
        scheduler_group.add_argument(
            "--max-num-seqs",
            **{
                **scheduler_kwargs["max_num_seqs"],
                "default": None,
            },
        )
        scheduler_group.add_argument(
            "--max-num-partial-prefills", **scheduler_kwargs["max_num_partial_prefills"]
        )
        scheduler_group.add_argument(
            "--max-long-partial-prefills",
            **scheduler_kwargs["max_long_partial_prefills"],
        )
        scheduler_group.add_argument(
            "--long-prefill-token-threshold",
            **scheduler_kwargs["long_prefill_token_threshold"],
        )
        # 多步调度参数已删除，当前只保留统一 step 调度模式。
        scheduler_group.add_argument(
            "--scheduling-policy", **scheduler_kwargs["policy"]
        )
        scheduler_group.add_argument(
            "--enable-chunked-prefill",
            **{
                **scheduler_kwargs["enable_chunked_prefill"],
                "default": None,
            },
        )
        scheduler_group.add_argument(
            "--disable-chunked-mm-input", **scheduler_kwargs["disable_chunked_mm_input"]
        )
        scheduler_group.add_argument(
            "--scheduler-cls", **scheduler_kwargs["scheduler_cls"]
        )
        scheduler_group.add_argument(
            "--disable-hybrid-kv-cache-manager",
            **scheduler_kwargs["disable_hybrid_kv_cache_manager"],
        )
        scheduler_group.add_argument(
            "--async-scheduling", **scheduler_kwargs["async_scheduling"]
        )
        scheduler_group.add_argument(
            "--stream-interval", **scheduler_kwargs["stream_interval"]
        )

        # Compilation arguments
        compilation_kwargs = get_kwargs(CompilationConfig)
        compilation_group = parser.add_argument_group(
            title="CompilationConfig",
            description=CompilationConfig.__doc__,
        )
        compilation_group.add_argument(
            "--cudagraph-capture-sizes", **compilation_kwargs["cudagraph_capture_sizes"]
        )
        compilation_group.add_argument(
            "--max-cudagraph-capture-size",
            **compilation_kwargs["max_cudagraph_capture_size"],
        )

        # Kernel arguments
        kernel_kwargs = get_kwargs(KernelConfig)
        kernel_group = parser.add_argument_group(
            title="KernelConfig",
            description=KernelConfig.__doc__,
        )
        kernel_group.add_argument(
            "--enable-flashinfer-autotune",
            **kernel_kwargs["enable_flashinfer_autotune"],
        )
        moe_backend_kwargs = kernel_kwargs["moe_backend"]
        moe_backend_kwargs["type"] = lambda s: s.lower().replace("-", "_")
        kernel_group.add_argument("--moe-backend", **moe_backend_kwargs)

        # vLLM arguments
        cfie_kwargs = get_kwargs(CfieConfig)
        cfie_group = parser.add_argument_group(
            title="CfieConfig",
            description=CfieConfig.__doc__,
        )
        # We construct SpeculativeConfig using fields from other configs in
        # create_engine_config. So we set the type to a JSON string here to
        # delay the Pydantic validation that comes with SpeculativeConfig.
        cfie_kwargs["speculative_config"]["type"] = optional_type(json.loads)
        cfie_group.add_argument(
            "--speculative-config", **cfie_kwargs["speculative_config"]
        )
        cfie_group.add_argument(
            "--kv-transfer-config", **cfie_kwargs["kv_transfer_config"]
        )
        cfie_group.add_argument("--kv-events-config", **cfie_kwargs["kv_events_config"])
        cfie_group.add_argument(
            "--ec-transfer-config", **cfie_kwargs["ec_transfer_config"]
        )
        cfie_group.add_argument(
            "--compilation-config", "-cc", **cfie_kwargs["compilation_config"]
        )
        cfie_group.add_argument(
            "--attention-config", "-ac", **cfie_kwargs["attention_config"]
        )
        cfie_group.add_argument("--kernel-config", **cfie_kwargs["kernel_config"])
        cfie_group.add_argument(
            "--additional-config", **cfie_kwargs["additional_config"]
        )
        cfie_group.add_argument(
            "--structured-outputs-config", **cfie_kwargs["structured_outputs_config"]
        )
        cfie_group.add_argument("--profiler-config", **cfie_kwargs["profiler_config"])
        cfie_group.add_argument(
            "--optimization-level", **cfie_kwargs["optimization_level"]
        )
        cfie_group.add_argument("--performance-mode", **cfie_kwargs["performance_mode"])
        cfie_group.add_argument(
            "--weight-transfer-config", **cfie_kwargs["weight_transfer_config"]
        )

        # Other arguments
        parser.add_argument(
            "--disable-log-stats",
            action="store_true",
            help="Disable logging statistics.",
        )

        parser.add_argument(
            "--aggregate-engine-logging",
            action="store_true",
            help="Log aggregate rather than per-engine statistics "
                 "when using data parallelism.",
        )

        parser.add_argument(
            "--fail-on-environ-validation",
            help="If set, the engine will raise an error if "
                 "environment validation fails.",
            default=False,
            action=argparse.BooleanOptionalAction,
        )

        parser.add_argument(
            "--shutdown-timeout",
            type=int,
            default=0,
            help="Shutdown timeout in seconds. 0 = abort, >0 = wait.",
        )

        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # Get the list of attributes of this dataclass.
        # 取出 EngineArgs dataclass 的全部字段名。
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        # Set the attributes from the parsed arguments.
        # 仅把 argparse 中真实存在的字段写入 EngineArgs。
        engine_args = cls(
            **{attr: getattr(args, attr) for attr in attrs if hasattr(args, attr)}
        )
        # 返回组装完成的 EngineArgs。
        return engine_args

    def create_model_config(self) -> ModelConfig:
        # gguf file needs a specific model loader
        # gguf 模型强制切换到 gguf 加载模式。
        if is_gguf(self.model):
            self.quantization = self.load_format = "gguf"

        # 非多进程模式下提示随机种子会影响当前 Python 进程状态。
        if not envs.VLLM_ENABLE_V1_MULTIPROCESSING:
            logger.warning(
                "The global random seed is set to %d. Since "
                "VLLM_ENABLE_V1_MULTIPROCESSING is set to False, this may "
                "affect the random state of the Python process that "
                "launched vLLM.",
                self.seed,
            )

        # 将 EngineArgs 中的模型相关字段收拢成 ModelConfig。
        return ModelConfig(
            model=self.model,
            model_weights=self.model_weights,
            hf_config_path=self.hf_config_path,
            runner=self.runner,
            convert=self.convert,
            tokenizer=self.tokenizer,  # type: ignore[arg-type]
            tokenizer_mode=self.tokenizer_mode,
            trust_remote_code=self.trust_remote_code,
            allowed_local_media_path=self.allowed_local_media_path,
            allowed_media_domains=self.allowed_media_domains,
            dtype=self.dtype,
            seed=self.seed,
            revision=self.revision,
            code_revision=self.code_revision,
            hf_token=self.hf_token,
            hf_overrides=self.hf_overrides,
            tokenizer_revision=self.tokenizer_revision,
            max_model_len=self.max_model_len,
            quantization=self.quantization,
            allow_deprecated_quantization=self.allow_deprecated_quantization,
            enforce_eager=self.enforce_eager,
            enable_return_routed_experts=self.enable_return_routed_experts,
            max_logprobs=self.max_logprobs,
            logprobs_mode=self.logprobs_mode,
            disable_sliding_window=self.disable_sliding_window,
            disable_cascade_attn=self.disable_cascade_attn,
            skip_tokenizer_init=self.skip_tokenizer_init,
            enable_prompt_embeds=self.enable_prompt_embeds,
            served_model_name=self.served_model_name,
            language_model_only=self.language_model_only,
            limit_mm_per_prompt=self.limit_mm_per_prompt,
            enable_mm_embeds=self.enable_mm_embeds,
            interleave_mm_strings=self.interleave_mm_strings,
            media_io_kwargs=self.media_io_kwargs,
            skip_mm_profiling=self.skip_mm_profiling,
            config_format=self.config_format,
            mm_processor_kwargs=self.mm_processor_kwargs,
            mm_processor_cache_gb=self.mm_processor_cache_gb,
            mm_processor_cache_type=self.mm_processor_cache_type,
            mm_shm_cache_max_object_size_mb=self.mm_shm_cache_max_object_size_mb,
            mm_encoder_only=self.mm_encoder_only,
            mm_encoder_tp_mode=self.mm_encoder_tp_mode,
            mm_encoder_attn_backend=self.mm_encoder_attn_backend,
            pooler_config=self.pooler_config,
            generation_config=self.generation_config,
            override_generation_config=self.override_generation_config,
            enable_sleep_mode=self.enable_sleep_mode,
            model_impl=self.model_impl,
            override_attention_dtype=self.override_attention_dtype,
            logits_processors=self.logits_processors,
            video_pruning_rate=self.video_pruning_rate,
            io_processor_plugin=self.io_processor_plugin,
        )

    def validate_tensorizer_args(self):
        from cfie.model_executor.model_loader.tensorizer import TensorizerConfig

        # 把平铺在 extra config 中的 tensorizer 字段归拢到 tensorizer_config。
        for key in self.model_loader_extra_config:
            if key in TensorizerConfig._fields:
                self.model_loader_extra_config["tensorizer_config"][key] = (
                    self.model_loader_extra_config[key]
                )

    def create_load_config(self) -> LoadConfig:
        # bitsandbytes 量化时强制对应的 load_format。
        if self.quantization == "bitsandbytes":
            self.load_format = "bitsandbytes"

        # tensorizer 模式下补齐 tensorizer 的专用加载参数。
        if self.load_format == "tensorizer":
            if hasattr(self.model_loader_extra_config, "to_serializable"):
                # 先把额外配置转成可序列化字典。
                self.model_loader_extra_config = (
                    self.model_loader_extra_config.to_serializable()
                )
            # 初始化 tensorizer_config 子字典。
            self.model_loader_extra_config["tensorizer_config"] = {}
            # 把 model 路径写入 tensorizer_dir。
            self.model_loader_extra_config["tensorizer_config"]["tensorizer_dir"] = (
                self.model
            )
            # 校验并补齐 tensorizer 专属参数。
            self.validate_tensorizer_args()

        # 将加载相关字段收拢成 LoadConfig。
        return LoadConfig(
            load_format=self.load_format,
            download_dir=self.download_dir,
            safetensors_load_strategy=self.safetensors_load_strategy,
            model_loader_extra_config=self.model_loader_extra_config,
            ignore_patterns=self.ignore_patterns,
            use_tqdm_on_load=self.use_tqdm_on_load,
            pt_load_map_location=self.pt_load_map_location,
        )

    # 根据 CLI/调用侧给出的 spec 参数，构造 target+draft 共用的投机解码配置。
    def create_speculative_config(
            self,
            target_model_config: ModelConfig,
            target_parallel_config: ParallelConfig,
    ) -> SpeculativeConfig | None:
        """Initializes and returns a SpeculativeConfig object based on
        `speculative_config`.

        This function utilizes `speculative_config` to create a
        SpeculativeConfig object. The `speculative_config` can either be
        provided as a JSON string input via CLI arguments or directly as a
        dictionary from the engine.
        """
        # 未启用 speculative decoding 时直接返回空配置。
        if self.speculative_config is None:
            return None

        # Note(Shangming): These parameters are not obtained from the cli arg
        # '--speculative-config' and must be passed in when creating the engine
        # config.
        # 把目标模型与并行配置补进 speculative_config，供 drafter 侧复用。
        self.speculative_config.update(
            {
                "target_model_config": target_model_config,
                "target_parallel_config": target_parallel_config,
            }
        )
        # 将配置字典实例化为 SpeculativeConfig。
        return SpeculativeConfig(**self.speculative_config)

    # 把 `EngineArgs` 展开成完整 `CfieConfig`，这是 chat 进入 v1 引擎前的总装步骤。
    def create_engine_config(
            self,
            usage_context: UsageContext | None = None,
            headless: bool = False,
    ) -> CfieConfig:
        """
        Create the CfieConfig.

        NOTE: If CfieConfig is incompatible, we raise an error.
        """
        # 先让平台层完成设备预注册与配置更新。
        current_platform.pre_register_and_update()

        # 根据当前平台设备类型创建 DeviceConfig。
        device_config = DeviceConfig(device=cast(Device, current_platform.device_type))

        # 校验环境变量配置是否合法。
        envs.validate_environ(self.fail_on_environ_validation)

        # 非云存储路径下，优先探测并替换 speculator 相关配置。
        if not is_cloud_storage(self.model):
            (self.model, self.tokenizer, self.speculative_config) = (
                maybe_override_with_speculators(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    revision=self.revision,
                    trust_remote_code=self.trust_remote_code,
                    cfie_speculative_config=self.speculative_config,
                )
            )

        # 创建 ModelConfig，并同步回写标准化后的 model/tokenizer 路径。
        model_config = self.create_model_config()
        self.model = model_config.model
        self.model_weights = model_config.model_weights
        self.tokenizer = model_config.tokenizer

        # 校验当前特性组合是否被支持。
        self._check_feature_supported()

        # 按模型能力补齐 chunked prefill / prefix caching 默认值。
        self._set_default_chunked_prefill_and_prefix_caching_args(model_config)

        # 按使用场景补齐 max_num_seqs 与 batched tokens 默认值。
        self._set_default_max_num_seqs_and_batched_tokens_args(
            usage_context,
            model_config
        )

        # 默认为无滑窗模型。
        sliding_window: int | None = None
        if not is_interleaved(model_config.hf_text_config):
            # 对纯滑窗模型，读取其滑窗大小写入 cache 配置。
            sliding_window = model_config.get_sliding_window()

        # 将 auto 类型的 kv cache dtype 解析成模型实际使用值。
        resolved_cache_dtype = resolve_kv_cache_dtype_string(
            self.kv_cache_dtype, model_config
        )

        # 到这里 prefix caching 开关必须已经被默认逻辑补齐。
        assert self.enable_prefix_caching is not None, (
            "enable_prefix_caching must be set by this point"
        )

        # 组装 CacheConfig，统一描述 KV cache 与缓存行为。
        cache_config = CacheConfig(
            block_size=self.block_size,  # type: ignore[arg-type]
            gpu_memory_utilization=self.gpu_memory_utilization,
            kv_cache_memory_bytes=self.kv_cache_memory_bytes,
            cache_dtype=resolved_cache_dtype,  # type: ignore[arg-type]
            is_attention_free=model_config.is_attention_free,
            num_gpu_blocks_override=self.num_gpu_blocks_override,
            sliding_window=sliding_window,
            enable_prefix_caching=self.enable_prefix_caching,
            prefix_caching_hash_algo=self.prefix_caching_hash_algo,
            calculate_kv_scales=self.calculate_kv_scales,
            kv_sharing_fast_prefill=self.kv_sharing_fast_prefill,
            mamba_cache_dtype=self.mamba_cache_dtype,
            mamba_ssm_cache_dtype=self.mamba_ssm_cache_dtype,
            mamba_block_size=self.mamba_block_size,
            mamba_cache_mode=self.mamba_cache_mode,
            kv_offloading_size=self.kv_offloading_size,
            kv_offloading_backend=self.kv_offloading_backend,
        )

        # 默认不携带 Ray runtime env。
        ray_runtime_env = None
        if is_ray_initialized():
            # Ray Serve LLM calls `create_engine_config` in the context
            # of a Ray task, therefore we check is_ray_initialized()
            # as opposed to is_in_ray_actor().
            import ray

            ray_runtime_env = ray.get_runtime_context().runtime_env
            # Avoid logging sensitive environment variables
            sanitized_env = ray_runtime_env.to_dict() if ray_runtime_env else {}
            if "env_vars" in sanitized_env:
                sanitized_env["env_vars"] = {
                    k: "***" for k in sanitized_env["env_vars"]
                }
            # 打印脱敏后的 Ray runtime 环境信息。
            logger.info("Using ray runtime env (env vars redacted): %s", sanitized_env)

        # 默认不绑定 placement group。
        placement_group = None
        if is_in_ray_actor():
            import ray

            # This call initializes Ray automatically if it is not initialized,
            # but we should not do this here.
            # 仅在 Ray actor 内获取当前 placement group。
            placement_group = ray.util.get_current_placement_group()

        # headless 模式下不允许 hybrid LB。
        assert not headless or not self.data_parallel_hybrid_lb, (
            "data_parallel_hybrid_lb is not applicable in headless mode"
        )

        # external_lb 与 hybrid_lb 不能同时开启。
        assert not (self.data_parallel_hybrid_lb and self.data_parallel_external_lb), (
            "data_parallel_hybrid_lb and data_parallel_external_lb cannot both be True."
        )

        # 多节点模式目前只支持 mp 数据并行后端。
        assert self.data_parallel_backend == "mp" or self.nnodes == 1, (
            "nnodes > 1 is only supported with data_parallel_backend=mp"
        )

        # 默认推断出的 data parallel rank 为 0。
        inferred_data_parallel_rank = 0

        if self.nnodes > 1:
            # 计算跨节点的总 world size。
            world_size = (
                    self.data_parallel_size
                    * self.pipeline_parallel_size
                    * self.tensor_parallel_size
            )

            # 计算单个 DP 组内部的 world size。
            world_size_within_dp = (
                    self.pipeline_parallel_size * self.tensor_parallel_size
            )

            # 计算每个节点上的 local world size。
            local_world_size = world_size // self.nnodes

            # 要求 world size 能被节点数整除，否则无法平均切分到各节点。
            assert world_size % self.nnodes == 0, (
                f"world_size={world_size} must be divisible by nnodes={self.nnodes}."
            )
            # 当前节点编号必须落在 `[0, nnodes)` 范围内。
            assert self.node_rank < self.nnodes, (
                f"node_rank={self.node_rank} must be less than nnodes={self.nnodes}."
            )

            # 推断当前节点对应的 data parallel rank 起点。
            inferred_data_parallel_rank = (
                                                  self.node_rank * local_world_size
                                          ) // world_size_within_dp

            # 外部 LB 模式下，直接把推断出的 DP rank 写回当前配置。
            if self.data_parallel_size > 1 and self.data_parallel_external_lb:
                # 记录当前节点负责的全局 data parallel rank。
                self.data_parallel_rank = inferred_data_parallel_rank
                # 打印由 node_rank 推断出的 DP rank，便于排查多机拓扑。
                logger.info(
                    "Inferred data_parallel_rank %d from node_rank %d for external lb",
                    self.data_parallel_rank,
                    self.node_rank,
                )
            elif self.data_parallel_size_local is None:
                # 内部 DPLB 模式下，如果没显式给本地 DP 大小，就按节点切分结果推断。
                self.data_parallel_size_local = max(
                    local_world_size // world_size_within_dp, 1
                )

        # 只要显式给了 data_parallel_rank，就视为 external LB 模式。
        data_parallel_external_lb = (
                self.data_parallel_external_lb or self.data_parallel_rank is not None
        )

        # local DP 数为 1 时，走纯 external LB 路线。
        if data_parallel_external_lb:
            # external LB 模式必须能确定当前进程所属的 data parallel rank。
            assert self.data_parallel_rank is not None, (
                "data_parallel_rank or node_rank must be specified if "
                "data_parallel_external_lb is enable."
            )
            # external LB 模式下，每个节点只能对应一个本地 DP rank。
            assert self.data_parallel_size_local in (1, None), (
                "data_parallel_size_local must be 1 or None when data_parallel_rank "
                "is set"
            )

            # 纯 external LB 下，本地 DP 大小固定为 1。
            data_parallel_size_local = 1

            # 既然已经退化成纯 external LB，就不再开启 hybrid LB。
            self.data_parallel_hybrid_lb = False

        elif self.data_parallel_size_local is not None:

            # 用户显式给了本地 DP 大小，直接采用。
            data_parallel_size_local = self.data_parallel_size_local

            # 只要设置了 data_parallel_start_rank 且不是 headless，就推断为 hybrid LB。
            if self.data_parallel_start_rank and not headless:
                # 自动打开 hybrid LB 模式。
                self.data_parallel_hybrid_lb = True

            # hybrid LB 但本地只有 1 个 DP rank 时，实际没有意义，转成 external LB。
            if self.data_parallel_hybrid_lb and data_parallel_size_local == 1:
                # 打印自动切换日志，说明 hybrid LB 被降级。
                logger.warning(
                    "data_parallel_hybrid_lb is not eligible when "
                    "data_parallel_size_local = 1, autoswitch to "
                    "data_parallel_external_lb."
                )
                # 改为 external LB。
                data_parallel_external_lb = True
                # 同时关闭 hybrid LB 标记。
                self.data_parallel_hybrid_lb = False

            # 如果本地 DP 大小已经等于全局 DP 大小，说明其实还是单节点，不需要 hybrid LB。
            if data_parallel_size_local == self.data_parallel_size:
                # 单节点场景关闭 hybrid LB。
                self.data_parallel_hybrid_lb = False

            # 给当前节点确定最终的 data parallel rank。
            self.data_parallel_rank = (
                    self.data_parallel_start_rank or inferred_data_parallel_rank
            )

            # 多节点时打印推断出的 DP rank，便于核对部署拓扑。
            if self.nnodes > 1:
                logger.info(
                    "Inferred data_parallel_rank %d from node_rank %d",
                    self.data_parallel_rank,
                    self.node_rank,
                )
        else:
            # 没有本地 DP 大小时，不允许直接开 hybrid LB。
            assert not self.data_parallel_hybrid_lb, (
                "data_parallel_size_local must be set to use data_parallel_hybrid_lb."
            )

            # Ray 的 span 打包策略允许一个 DP rank 横跨多个节点，此时本地 DP 大小默认记为 1。
            if self.data_parallel_backend == "ray" and (
                    envs.VLLM_RAY_DP_PACK_STRATEGY == "span"
            ):
                # DP rank 横跨多节点时，本地 DP 数默认按 1 处理。
                data_parallel_size_local = 1
            else:
                # 其他情况下，若未显式设置，本地 DP 大小默认等于全局 DP 大小。
                data_parallel_size_local = self.data_parallel_size

        # 确定 data parallel 使用的通信地址，供 torch distributed 与 ZMQ 使用。
        if self.data_parallel_address is None:
            # Ray 后端没有显式地址时，直接使用当前主机 IP。
            if self.data_parallel_backend == "ray":
                # 自动探测当前主机 IP。
                host_ip = get_ip()
                # 记录当前采用的 Ray DP 地址。
                logger.info(
                    "Using host IP %s as ray-based data parallel address", host_ip
                )
                # 把主机 IP 作为 DP 地址。
                data_parallel_address = host_ip
            else:
                # 非 Ray 路线当前只支持 mp 数据并行后端。
                assert self.data_parallel_backend == "mp", (
                    "data_parallel_backend can only be ray or mp, got %s",
                    self.data_parallel_backend,
                )
                # mp 路线优先使用显式 master_addr，否则退回 ParallelConfig 默认地址。
                data_parallel_address = (
                        self.master_addr or ParallelConfig.data_parallel_master_ip
                )
        else:
            # 如果用户显式给了 data_parallel_address，就直接使用。
            data_parallel_address = self.data_parallel_address

        # 远端 DP engine 需要一个 RPC 端口；纯本地场景则会改用本地 IPC 传输。
        data_parallel_rpc_port = (
            self.data_parallel_rpc_port
            if (self.data_parallel_rpc_port is not None)
            else ParallelConfig.data_parallel_rpc_port
        )

        # tokens-only 模式不需要 tokenizer，若模型尚未跳过初始化则在这里强制打开。
        if self.tokens_only and not model_config.skip_tokenizer_init:
            # 直接跳过 tokenizer 初始化，减少无用开销。
            model_config.skip_tokenizer_init = True
            # 打印模式切换日志。
            logger.info("Skipping tokenizer initialization for tokens-only mode.")

        # 组装 ParallelConfig，把并行拓扑、DP/LB、Ray 与 worker 参数统一收口。
        parallel_config = ParallelConfig(
            pipeline_parallel_size=self.pipeline_parallel_size,
            tensor_parallel_size=self.tensor_parallel_size,
            prefill_context_parallel_size=self.prefill_context_parallel_size,
            data_parallel_size=self.data_parallel_size,
            data_parallel_rank=self.data_parallel_rank or 0,
            data_parallel_external_lb=data_parallel_external_lb,
            data_parallel_size_local=data_parallel_size_local,
            master_addr=self.master_addr,
            master_port=self.master_port,
            nnodes=self.nnodes,
            node_rank=self.node_rank,
            distributed_timeout_seconds=self.distributed_timeout_seconds,
            data_parallel_master_ip=data_parallel_address,
            data_parallel_rpc_port=data_parallel_rpc_port,
            data_parallel_backend=self.data_parallel_backend,
            data_parallel_hybrid_lb=self.data_parallel_hybrid_lb,
            is_moe_model=model_config.is_moe,
            enable_expert_parallel=self.enable_expert_parallel,
            all2all_backend=self.all2all_backend,
            enable_elastic_ep=self.enable_elastic_ep,
            enable_dbo=self.enable_dbo,
            ubatch_size=self.ubatch_size,
            dbo_decode_token_threshold=self.dbo_decode_token_threshold,
            dbo_prefill_token_threshold=self.dbo_prefill_token_threshold,
            disable_nccl_for_dp_synchronization=self.disable_nccl_for_dp_synchronization,
            enable_eplb=self.enable_eplb,
            eplb_config=self.eplb_config,
            expert_placement_strategy=self.expert_placement_strategy,
            max_parallel_loading_workers=self.max_parallel_loading_workers,
            disable_custom_all_reduce=self.disable_custom_all_reduce,
            ray_workers_use_nsight=self.ray_workers_use_nsight,
            ray_runtime_env=ray_runtime_env,
            placement_group=placement_group,
            distributed_executor_backend=self.distributed_executor_backend,
            worker_cls=self.worker_cls,
            worker_extension_cls=self.worker_extension_cls,
            decode_context_parallel_size=self.decode_context_parallel_size,
            dcp_comm_backend=self.dcp_comm_backend,
            dcp_kv_cache_interleave_size=self.dcp_kv_cache_interleave_size,
            cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
            _api_process_count=self._api_process_count,
            _api_process_rank=self._api_process_rank,
        )

        # 根据 target model 与并行配置推导 speculative decoding 的最终配置。
        speculative_config = self.create_speculative_config(
            target_model_config=model_config,
            target_parallel_config=parallel_config,
        )

        # 到这里 batched token 上限必须已经被默认逻辑补齐。
        assert self.max_num_batched_tokens is not None, (
            "max_num_batched_tokens must be set by this point"
        )
        # 到这里最大序列数必须已经被默认逻辑补齐。
        assert self.max_num_seqs is not None, "max_num_seqs must be set by this point"
        # 到这里 chunked prefill 开关必须已经确定。
        assert self.enable_chunked_prefill is not None, (
            "enable_chunked_prefill must be set by this point"
        )
        # ModelConfig 在这个阶段必须已经拿到最终 max_model_len。
        assert model_config.max_model_len is not None, (
            "max_model_len must be set by this point"
        )

        # ----------------- 折叠成最终 SchedulerConfig -----------------
        # 组装 SchedulerConfig，统一描述调度上限、prefill 策略和异步调度行为。
        scheduler_config = SchedulerConfig(
            runner_type=model_config.runner_type,
            max_num_batched_tokens=self.max_num_batched_tokens,
            max_num_seqs=self.max_num_seqs,
            max_model_len=model_config.max_model_len,
            enable_chunked_prefill=self.enable_chunked_prefill,
            disable_chunked_mm_input=self.disable_chunked_mm_input,
            is_multimodal_model=model_config.is_multimodal_model,
            is_encoder_decoder=model_config.is_encoder_decoder,
            policy=self.scheduling_policy,
            scheduler_cls=self.scheduler_cls,
            max_num_partial_prefills=self.max_num_partial_prefills,
            max_long_partial_prefills=self.max_long_partial_prefills,
            long_prefill_token_threshold=self.long_prefill_token_threshold,
            disable_hybrid_kv_cache_manager=self.disable_hybrid_kv_cache_manager,
            async_scheduling=self.async_scheduling,
            stream_interval=self.stream_interval,
        )

        # 非多模态模型不允许配置默认的 modality-specific LoRA。
        if not model_config.is_multimodal_model and self.default_mm_loras:
            raise ValueError(
                "Default modality-specific LoRA(s) were provided for a "
                "non multimodal model"
            )

        # 仅在 enable_lora=True 时才真正创建 LoRAConfig。
        lora_config = (
            LoRAConfig(
                max_lora_rank=self.max_lora_rank,
                max_loras=self.max_loras,
                default_mm_loras=self.default_mm_loras,
                fully_sharded_loras=self.fully_sharded_loras,
                lora_dtype=self.lora_dtype,
                enable_tower_connector_lora=self.enable_tower_connector_lora,
                specialize_active_lora=self.specialize_active_lora,
                max_cpu_loras=self.max_cpu_loras
                if self.max_cpu_loras and self.max_cpu_loras > 0
                else None,
            )
            if self.enable_lora
            else None
        )

        # LoRA 与 speculative decoding 同开时，需要保证 batched token 预算足够覆盖一轮草稿展开。
        if (
                lora_config is not None
                and speculative_config is not None
                and scheduler_config.max_num_batched_tokens
                < (
                scheduler_config.max_num_seqs
                * (speculative_config.num_speculative_tokens + 1)
        )
        ):
            raise ValueError(
                "Consider increasing max_num_batched_tokens or "
                "decreasing num_speculative_tokens"
            )

        # bitsandbytes 预量化模型必须改用对应的专用加载器。
        if model_config.quantization == "bitsandbytes":
            # 同时把 quantization 和 load_format 都切到 bitsandbytes。
            self.quantization = self.load_format = "bitsandbytes"

        # 准备 attention 配置的副本，后续只在副本上应用 CLI 覆写。
        attention_config = copy.deepcopy(self.attention_config)
        # 如果 CLI 显式指定了 attention_backend，则在这里覆盖默认配置。
        if self.attention_backend is not None:
            # 不能同时在 attention_config.backend 和 CLI 上重复指定。
            if attention_config.backend is not None:
                raise ValueError(
                    "attention_backend and attention_config.backend "
                    "are mutually exclusive"
                )
            # 复用校验器，统一处理 "auto" 与字符串到枚举的转换。
            attention_config.backend = AttentionConfig.validate_backend_before(
                self.attention_backend
            )

        # 准备 kernel 配置副本，避免直接修改原始 EngineArgs 上的对象。
        kernel_config = copy.deepcopy(self.kernel_config)
        # 如果 CLI 指定了 flashinfer autotune，就在副本上应用。
        if self.enable_flashinfer_autotune is not None:
            # 不允许同时在 kernel_config 和 CLI 两边都给这个选项。
            if kernel_config.enable_flashinfer_autotune is not None:
                raise ValueError(
                    "enable_flashinfer_autotune and "
                    "kernel_config.enable_flashinfer_autotune "
                    "are mutually exclusive"
                )
            # 覆盖 flashinfer autotune 开关。
            kernel_config.enable_flashinfer_autotune = self.enable_flashinfer_autotune
        # 只要 moe_backend 不是 auto，就显式覆写 kernel 配置中的 MoE 后端。
        if self.moe_backend != "auto":
            kernel_config.moe_backend = self.moe_backend

        # 创建权重加载相关的 LoadConfig。
        load_config = self.create_load_config()

        # 如果配置了 reasoning parser，就把它写入结构化输出配置。
        if self.reasoning_parser:
            self.structured_outputs_config.reasoning_parser = self.reasoning_parser

        # reasoning parser plugin 也在这里写入结构化输出配置。
        if self.reasoning_parser_plugin:
            self.structured_outputs_config.reasoning_parser_plugin = (
                self.reasoning_parser_plugin
            )

        # 组装可观测性配置，统一描述 tracing、metrics 与调试统计项。
        observability_config = ObservabilityConfig(
            show_hidden_metrics_for_version=self.show_hidden_metrics_for_version,
            otlp_traces_endpoint=self.otlp_traces_endpoint,
            collect_detailed_traces=self.collect_detailed_traces,
            kv_cache_metrics=self.kv_cache_metrics,
            kv_cache_metrics_sample=self.kv_cache_metrics_sample,
            cudagraph_metrics=self.cudagraph_metrics,
            enable_layerwise_nvtx_tracing=self.enable_layerwise_nvtx_tracing,
            enable_mfu_metrics=self.enable_mfu_metrics,
            enable_mm_processor_stats=self.enable_mm_processor_stats,
            enable_logging_iteration_details=self.enable_logging_iteration_details,
        )

        # 准备 compilation 配置副本，后续把 CLI 覆写项合并进去。
        compilation_config = copy.deepcopy(self.compilation_config)
        # 如果 CLI 给了 cudagraph_capture_sizes，就覆盖副本中的对应字段。
        if self.cudagraph_capture_sizes is not None:
            # 不允许同时在 compilation_config 和 CLI 两边都指定 capture sizes。
            if compilation_config.cudagraph_capture_sizes is not None:
                raise ValueError(
                    "cudagraph_capture_sizes and compilation_config."
                    "cudagraph_capture_sizes are mutually exclusive"
                )
            # 写入最终的 cudagraph capture sizes。
            compilation_config.cudagraph_capture_sizes = self.cudagraph_capture_sizes
        # 如果 CLI 给了 max_cudagraph_capture_size，就继续覆盖副本。
        if self.max_cudagraph_capture_size is not None:
            # 同样不允许与 compilation_config 中的同名字段重复指定。
            if compilation_config.max_cudagraph_capture_size is not None:
                raise ValueError(
                    "max_cudagraph_capture_size and compilation_config."
                    "max_cudagraph_capture_size are mutually exclusive"
                )
            # 写入最终的最大 cudagraph capture size。
            compilation_config.max_cudagraph_capture_size = (
                self.max_cudagraph_capture_size
            )

        # 组装 offload 配置，把 CPU/UVA/prefetch 相关参数统一收口。
        offload_config = OffloadConfig(
            offload_backend=self.offload_backend,
            moe_cpu_budget_gb=self.moe_cpu_budget_gb,
            moe_cpu_min_free_gb=self.moe_cpu_min_free_gb,
            uva=UVAOffloadConfig(
                cpu_offload_gb=self.cpu_offload_gb,
                cpu_offload_params=self.cpu_offload_params,
            ),
            prefetch=PrefetchOffloadConfig(
                offload_group_size=self.offload_group_size,
                offload_num_in_group=self.offload_num_in_group,
                offload_prefetch_step=self.offload_prefetch_step,
                offload_params=self.offload_params,
            ),
        )

        # 最后组装总配置对象 CfieConfig，供 engine / worker / scheduler 共同使用。
        config = CfieConfig(
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=parallel_config,
            scheduler_config=scheduler_config,
            device_config=device_config,
            load_config=load_config,
            offload_config=offload_config,
            attention_config=attention_config,
            kernel_config=kernel_config,
            lora_config=lora_config,
            speculative_config=speculative_config,
            structured_outputs_config=self.structured_outputs_config,
            observability_config=observability_config,
            compilation_config=compilation_config,
            kv_transfer_config=self.kv_transfer_config,
            kv_events_config=self.kv_events_config,
            ec_transfer_config=self.ec_transfer_config,
            profiler_config=self.profiler_config,
            additional_config=self.additional_config,
            optimization_level=self.optimization_level,
            performance_mode=self.performance_mode,
            weight_transfer_config=self.weight_transfer_config,
            shutdown_timeout=self.shutdown_timeout,
        )

        # 返回最终完整可运行的总配置。
        return config

    def _check_feature_supported(self):

        # 当前实现还不支持并发 partial prefill，自定义该参数时直接报错。
        if (
                self.max_num_partial_prefills != SchedulerConfig.max_num_partial_prefills
                or self.max_long_partial_prefills != SchedulerConfig.max_long_partial_prefills
        ):
            _raise_unsupported_error(feature_name="Concurrent Partial Prefill")

        # pipeline parallel 大于 1 时，需要执行后端显式支持 PP。
        if self.pipeline_parallel_size > 1:
            # 从后端对象上读取是否支持 pipeline parallel 的能力标记。
            supports_pp = getattr(
                self.distributed_executor_backend, "supports_pp", False
            )
            # 既不是内建支持的后端，也没有显式 supports_pp 时，拒绝启动。
            if not supports_pp and self.distributed_executor_backend not in (
                    ParallelConfig.distributed_executor_backend,
                    "ray",
                    "mp",
                    "external_launcher",
            ):
                # 组织更明确的报错文案，指出当前不支持的组合。
                name = (
                    "Pipeline Parallelism without Ray distributed "
                    "executor or multiprocessing executor or external "
                    "launcher"
                )
                _raise_unsupported_error(feature_name=name)

    @classmethod
    def get_batch_defaults(
            cls,
            world_size: int,
    ) -> tuple[dict[UsageContext | None, int], dict[UsageContext | None, int]]:
        from cfie.usage.usage_lib import UsageContext

        default_max_num_batched_tokens: dict[UsageContext | None, int]
        default_max_num_seqs: dict[UsageContext | None, int]

        # ----------------- 按硬件和 usage_context 选默认预算 -----------------
        # 这里返回默认值表，后续再由调用方决定是否采用。

        # 先探测设备显存和设备名；失败时回退到保守档位。
        try:
            device_memory = current_platform.get_device_total_memory()
            device_name = current_platform.get_device_name().lower()
        except Exception:
            # 这里只影响默认预算选择，不影响用户显式配置。
            device_memory = 0
            device_name = ""

        # 大显存 GPU 默认给更高预算，但单独排除 A100。
        if device_memory >= 70 * GiB_bytes and "a100" not in device_name:
            # H100/H200/MI300X 一类设备走高档默认值。
            default_max_num_batched_tokens = {
                UsageContext.LLM_CLASS: 16384,
                UsageContext.OPENAI_API_SERVER: 8192,
            }
            default_max_num_seqs = {
                UsageContext.LLM_CLASS: 1024,
                UsageContext.OPENAI_API_SERVER: 1024,
            }
        else:
            # 其余 GPU 统一走保守档默认值。
            default_max_num_batched_tokens = {
                UsageContext.LLM_CLASS: 8192,
                UsageContext.OPENAI_API_SERVER: 2048,
            }
            default_max_num_seqs = {
                UsageContext.LLM_CLASS: 256,
                UsageContext.OPENAI_API_SERVER: 256,
            }

        # TPU 按芯片型号覆写默认 token 预算。
        if current_platform.is_tpu():
            chip_name = current_platform.get_device_name()

            if chip_name == "V6E":
                # V6E 走最高一档 TPU 默认值。
                default_max_num_batched_tokens = {
                    UsageContext.LLM_CLASS: 2048,
                    UsageContext.OPENAI_API_SERVER: 1024,
                }
            elif chip_name == "V5E":
                # V5E 次一档。
                default_max_num_batched_tokens = {
                    UsageContext.LLM_CLASS: 1024,
                    UsageContext.OPENAI_API_SERVER: 512,
                }
            elif chip_name == "V5P":
                # V5P 再保守一档。
                default_max_num_batched_tokens = {
                    UsageContext.LLM_CLASS: 512,
                    UsageContext.OPENAI_API_SERVER: 256,
                }

        # CPU 默认值按 world_size 线性放大。
        if current_platform.is_cpu():
            default_max_num_batched_tokens = {
                UsageContext.LLM_CLASS: 4096 * world_size,
                UsageContext.OPENAI_API_SERVER: 2048 * world_size,
            }
            default_max_num_seqs = {
                UsageContext.LLM_CLASS: 256 * world_size,
                UsageContext.OPENAI_API_SERVER: 128 * world_size,
            }

        # 返回默认值表，后续再折叠成最终配置。
        return default_max_num_batched_tokens, default_max_num_seqs

    def _set_default_chunked_prefill_and_prefix_caching_args(
            self, model_config: ModelConfig
    ) -> None:
        # ----------------- 先按模型能力补默认开关 -----------------
        # 这一层只负责“开关补默认”，不负责最终预算校验。
        # 读取该模型是否官方支持 chunked prefill 的默认能力。
        default_chunked_prefill = model_config.is_chunked_prefill_supported
        # 读取该模型是否官方支持 prefix caching 的默认能力。
        default_prefix_caching = model_config.is_prefix_caching_supported

        # 用户未显式指定时，沿用模型能力给出的默认 chunked prefill 开关。
        if self.enable_chunked_prefill is None:
            self.enable_chunked_prefill = default_chunked_prefill

            # 打印最终采用的默认 chunked prefill 决策。
            logger.debug(
                "%s chunked prefill by default",
                "Enabling" if default_chunked_prefill else "Disabling",
            )
        elif (
                model_config.runner_type == "generate"
                and not self.enable_chunked_prefill
                and default_chunked_prefill
        ):
            # 生成模型若手动关闭官方建议开启的 chunked prefill，则给出风险警告。
            logger.warning_once(
                "This model does not officially support disabling chunked prefill. "
                "Disabling this manually may cause the engine to crash "
                "or produce incorrect outputs.",
                scope="local",
            )
        elif (
                model_config.runner_type == "pooling"
                and self.enable_chunked_prefill
                and not default_chunked_prefill
        ):
            # pooling 模型若手动开启官方不支持的 chunked prefill，也给出风险警告。
            logger.warning_once(
                "This model does not officially support chunked prefill. "
                "Enabling this manually may cause the engine to crash "
                "or produce incorrect outputs.",
                scope="local",
            )

        # 用户未显式指定时，沿用模型能力给出的默认 prefix caching 开关。
        if self.enable_prefix_caching is None:
            self.enable_prefix_caching = default_prefix_caching

            # 打印最终采用的默认 prefix caching 决策。
            logger.debug(
                "%s prefix caching by default",
                "Enabling" if default_prefix_caching else "Disabling",
            )
        elif (
                model_config.runner_type == "pooling"
                and self.enable_prefix_caching
                and not default_prefix_caching
        ):
            # pooling 模型手动开启官方不支持的 prefix caching 时给出风险警告。
            logger.warning_once(
                "This model does not officially support prefix caching. "
                "Enabling this manually may cause the engine to crash "
                "or produce incorrect outputs.",
                scope="local",
            )

        # V1 后端在 RISC-V CPU 上统一禁用这两个特性。
        if current_platform.is_cpu() and current_platform.get_cpu_architecture() in (
                CpuArchEnum.RISCV,
        ):
            logger.info(
                "Chunked prefill is not supported for"
                "RISC-V CPUs; "
                "disabling it for V1 backend."
            )
            self.enable_chunked_prefill = False
            logger.info(
                "Prefix caching is not supported for "
                "RISC-V CPUs; "
                "disabling it for V1 backend."
            )
            self.enable_prefix_caching = False

    def _set_default_max_num_seqs_and_batched_tokens_args(
            self,
            usage_context: UsageContext | None,
            model_config: ModelConfig,
    ):
        # ----------------- 再按运行场景补调度预算 -----------------
        # 这里基于前面已确定的开关，补齐 batched tokens 和 seqs 两个预算。

        # 并行规模会影响默认预算档位。
        world_size = self.pipeline_parallel_size * self.tensor_parallel_size

        (
            default_max_num_batched_tokens,
            default_max_num_seqs,
        ) = self.get_batch_defaults(world_size)

        # 保存原始输入，后面用来区分默认值和用户显式值。
        orig_max_num_batched_tokens = self.max_num_batched_tokens
        orig_max_num_seqs = self.max_num_seqs

        # 先按 usage_context 补齐默认 token 预算。
        if self.max_num_batched_tokens is None:
            self.max_num_batched_tokens = default_max_num_batched_tokens.get(
                usage_context,
                SchedulerConfig.DEFAULT_MAX_NUM_BATCHED_TOKENS,
            )

        # 再按 usage_context 补齐默认请求数预算。
        if self.max_num_seqs is None:
            self.max_num_seqs = default_max_num_seqs.get(
                usage_context,
                SchedulerConfig.DEFAULT_MAX_NUM_SEQS,
            )

        # throughput 模式只放大默认值，不改用户显式值。
        if self.performance_mode == "throughput":
            if orig_max_num_batched_tokens is None:
                self.max_num_batched_tokens *= 2
            if orig_max_num_seqs is None:
                self.max_num_seqs *= 2

        # 只有默认 token 预算才继续自动收口。
        if orig_max_num_batched_tokens is None:
            assert model_config.max_model_len is not None, (
                "max_model_len must be set by this point"
            )

            # 关闭 chunked prefill 时，默认 token 预算至少要容纳完整上下文。
            if not self.enable_chunked_prefill:
                self.max_num_batched_tokens = max(
                    model_config.max_model_len,
                    self.max_num_batched_tokens,
                )

            # 再用并发上限和上下文长度把默认 token 预算裁到合理范围内。
            self.max_num_batched_tokens = min(
                self.max_num_seqs * model_config.max_model_len,
                self.max_num_batched_tokens,
            )

            # 只有默认值路径才打印 defaulting 日志。
            logger.debug(
                "Defaulting max_num_batched_tokens to %d for %s usage context.",
                self.max_num_batched_tokens,
                usage_context.value if usage_context else None,
            )

        # 只有默认请求数预算才继续自动收口。
        if orig_max_num_seqs is None:
            assert self.max_num_batched_tokens is not None  # For type checking

            # 默认请求数不应超过最终 token 预算。
            self.max_num_seqs = min(self.max_num_seqs, self.max_num_batched_tokens)

            # 同样只在默认值路径记录日志。
            logger.debug(
                "Defaulting max_num_seqs to %d for %s usage context.",
                self.max_num_seqs,
                usage_context.value if usage_context else None,
            )


@dataclass
class AsyncEngineArgs(EngineArgs):
    """Arguments for asynchronous vLLM engine."""

    enable_log_requests: bool = False

    @staticmethod
    def add_cli_args(
            parser: FlexibleArgumentParser, async_args_only: bool = False
    ) -> FlexibleArgumentParser:
        # Initialize plugin to update the parser, for example, The plugin may
        # add a new kind of quantization method to --quantization argument or
        # a new device to --device argument.
        load_general_plugins()
        if not async_args_only:
            parser = EngineArgs.add_cli_args(parser)
        parser.add_argument(
            "--enable-log-requests",
            action=argparse.BooleanOptionalAction,
            default=AsyncEngineArgs.enable_log_requests,
            help="Enable logging request information, dependent on log level:\n"
                 "- INFO: Request ID, parameters and LoRA request.\n"
                 "- DEBUG: Prompt inputs (e.g: text, token IDs).\n"
                 "You can set the minimum log level via `VLLM_LOGGING_LEVEL`.",
        )
        current_platform.pre_register_and_update(parser)
        return parser


def _raise_unsupported_error(feature_name: str):
    msg = (
        f"{feature_name} is not supported. We recommend to "
        f"remove {feature_name} from your config."
    )
    raise NotImplementedError(msg)


def human_readable_int(value: str) -> int:
    """Parse human-readable integers like '1k', '2M', etc.
    Including decimal values with decimal multipliers.

    Examples:
    - '1k' -> 1,000
    - '1K' -> 1,024
    - '25.6k' -> 25,600
    """
    value = value.strip()

    match = re.fullmatch(r"(\d+(?:\.\d+)?)([kKmMgGtT])", value)
    if match:
        decimal_multiplier = {
            "k": 10 ** 3,
            "m": 10 ** 6,
            "g": 10 ** 9,
            "t": 10 ** 12,
        }
        binary_multiplier = {
            "K": 2 ** 10,
            "M": 2 ** 20,
            "G": 2 ** 30,
            "T": 2 ** 40,
        }

        number, suffix = match.groups()
        if suffix in decimal_multiplier:
            mult = decimal_multiplier[suffix]
            return int(float(number) * mult)
        elif suffix in binary_multiplier:
            mult = binary_multiplier[suffix]
            # Do not allow decimals with binary multipliers
            try:
                return int(number) * mult
            except ValueError as e:
                raise argparse.ArgumentTypeError(
                    "Decimals are not allowed "
                    f"with binary suffixes like {suffix}. Did you mean to use "
                    f"{number}{suffix.lower()} instead?"
                ) from e

    # Regular plain number.
    return int(value)


def human_readable_int_or_auto(value: str) -> int:
    """Parse human-readable integers like '1k', '2M', etc.
    Including decimal values with decimal multipliers.
    Also accepts -1 or 'auto' as a special value for auto-detection.

    Examples:
    - '1k' -> 1,000
    - '1K' -> 1,024
    - '25.6k' -> 25,600
    - '-1' or 'auto' -> -1 (special value for auto-detection)
    """
    value = value.strip()

    if value == "-1" or value.lower() == "auto":
        return -1

    return human_readable_int(value)
