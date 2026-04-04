# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the cfie project

# 统一导出 attention 子配置。
from cfie.config.attention import AttentionConfig
# 统一导出 cache 子配置。
from cfie.config.cache import CacheConfig
# 统一导出 compilation 相关配置与枚举。
from cfie.config.compilation import (
    CompilationConfig,
    CompilationMode,
    CUDAGraphMode,
    PassConfig,
)
# 统一导出 device 子配置。
from cfie.config.device import DeviceConfig
# 统一导出 EC transfer 子配置。
from cfie.config.ec_transfer import ECTransferConfig
# 统一导出 kernel 子配置。
from cfie.config.kernel import KernelConfig
# 统一导出 KV events 子配置。
from cfie.config.kv_events import KVEventsConfig
# 统一导出 KV transfer 子配置。
from cfie.config.kv_transfer import KVTransferConfig
# 统一导出模型加载配置。
from cfie.config.load import LoadConfig
# 统一导出 LoRA 配置。
from cfie.config.lora import LoRAConfig
# 统一导出模型配置及模型架构辅助函数。
from cfie.config.model import (
    ModelConfig,
    iter_architecture_defaults,
    str_dtype_to_torch_dtype,
    try_match_architecture_defaults,
)
# 统一导出多模态配置。
from cfie.config.multimodal import MultiModalConfig
# 统一导出可观测性配置。
from cfie.config.observability import ObservabilityConfig
# 统一导出 offload 相关配置。
from cfie.config.offload import (
    OffloadBackend,
    OffloadConfig,
    PrefetchOffloadConfig,
    UVAOffloadConfig,
)
# 统一导出并行配置。
from cfie.config.parallel import EPLBConfig, ParallelConfig
# 统一导出 pooling 配置。
from cfie.config.pooler import PoolerConfig
# 统一导出 profiler 配置。
from cfie.config.profiler import ProfilerConfig
# 统一导出调度器配置。
from cfie.config.scheduler import SchedulerConfig
# 统一导出 speculative decoding 配置。
from cfie.config.speculative import SpeculativeConfig
# 统一导出语音转文本配置。
from cfie.config.speech_to_text import SpeechToTextConfig
# 统一导出结构化输出配置。
from cfie.config.structured_outputs import StructuredOutputsConfig
# 统一导出配置工具函数。
from cfie.config.utils import (
    ConfigType,
    SupportsMetricsInfo,
    config,
    get_attr_docs,
    is_init_field,
    replace,
    update_config,
)
# 统一导出顶层 CfieConfig 与上下文访问函数。
from cfie.config.cfie import (
    CfieConfig,
    get_cached_compilation_config,
    get_current_cfie_config,
    get_current_cfie_config_or_none,
    get_layers_from_cfie_config,
    set_current_cfie_config,
)
# 统一导出权重传输配置。
from cfie.config.weight_transfer import WeightTransferConfig

# Keep upstream config names as aliases while CFIE still contains partially
# migrated call sites that expect the vLLM naming surface.
# 保留一组兼容 vLLM 命名的别名，便于旧调用点逐步迁移。
VllmConfig = CfieConfig
get_current_vllm_config = get_current_cfie_config
get_current_vllm_config_or_none = get_current_cfie_config_or_none
set_current_vllm_config = set_current_cfie_config
get_layers_from_vllm_config = get_layers_from_cfie_config

# __all__ should only contain classes and functions.
# Types and globals should be imported from their respective modules.
# __all__ 只导出公共 API，避免把内部类型细节暴露出去。
__all__ = [
    # From cfie.config.attention
    "AttentionConfig",
    # From cfie.config.cache
    "CacheConfig",
    # From cfie.config.compilation
    "CompilationConfig",
    "CompilationMode",
    "CUDAGraphMode",
    "PassConfig",
    # From cfie.config.device
    "DeviceConfig",
    # From cfie.config.ec_transfer
    "ECTransferConfig",
    # From cfie.config.kernel
    "KernelConfig",
    # From cfie.config.kv_events
    "KVEventsConfig",
    # From cfie.config.kv_transfer
    "KVTransferConfig",
    # From cfie.config.load
    "LoadConfig",
    # From cfie.config.lora
    "LoRAConfig",
    # From cfie.config.model
    "ModelConfig",
    "iter_architecture_defaults",
    "str_dtype_to_torch_dtype",
    "try_match_architecture_defaults",
    # From cfie.config.multimodal
    "MultiModalConfig",
    # From cfie.config.observability
    "ObservabilityConfig",
    # From cfie.config.offload
    "OffloadBackend",
    "OffloadConfig",
    "PrefetchOffloadConfig",
    "UVAOffloadConfig",
    # From cfie.config.parallel
    "EPLBConfig",
    "ParallelConfig",
    # From cfie.config.pooler
    "PoolerConfig",
    # From cfie.config.scheduler
    "SchedulerConfig",
    # From cfie.config.speculative
    "SpeculativeConfig",
    # From cfie.config.speech_to_text
    "SpeechToTextConfig",
    # From cfie.config.structured_outputs
    "StructuredOutputsConfig",
    # From cfie.config.profiler
    "ProfilerConfig",
    # From cfie.config.utils
    "ConfigType",
    "SupportsMetricsInfo",
    "config",
    "get_attr_docs",
    "is_init_field",
    "replace",
    "update_config",
    # From cfie.config.cfie
    "CfieConfig",
    "VllmConfig",
    "get_cached_compilation_config",
    "get_current_cfie_config",
    "get_current_cfie_config_or_none",
    "get_current_vllm_config",
    "get_current_vllm_config_or_none",
    "set_current_cfie_config",
    "set_current_vllm_config",
    "get_layers_from_cfie_config",
    "get_layers_from_vllm_config",
    "WeightTransferConfig",
]
