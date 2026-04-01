# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the cfie project

from cfie.config.attention import AttentionConfig
from cfie.config.cache import CacheConfig
from cfie.config.compilation import (
    CompilationConfig,
    CompilationMode,
    CUDAGraphMode,
    PassConfig,
)
from cfie.config.device import DeviceConfig
from cfie.config.ec_transfer import ECTransferConfig
from cfie.config.kernel import KernelConfig
from cfie.config.kv_events import KVEventsConfig
from cfie.config.kv_transfer import KVTransferConfig
from cfie.config.load import LoadConfig
from cfie.config.lora import LoRAConfig
from cfie.config.model import (
    ModelConfig,
    iter_architecture_defaults,
    str_dtype_to_torch_dtype,
    try_match_architecture_defaults,
)
from cfie.config.multimodal import MultiModalConfig
from cfie.config.observability import ObservabilityConfig
from cfie.config.offload import (
    OffloadBackend,
    OffloadConfig,
    PrefetchOffloadConfig,
    UVAOffloadConfig,
)
from cfie.config.parallel import EPLBConfig, ParallelConfig
from cfie.config.pooler import PoolerConfig
from cfie.config.profiler import ProfilerConfig
from cfie.config.scheduler import SchedulerConfig
from cfie.config.speculative import SpeculativeConfig
from cfie.config.speech_to_text import SpeechToTextConfig
from cfie.config.structured_outputs import StructuredOutputsConfig
from cfie.config.utils import (
    ConfigType,
    SupportsMetricsInfo,
    config,
    get_attr_docs,
    is_init_field,
    replace,
    update_config,
)
from cfie.config.cfie import (
    CfieConfig,
    get_cached_compilation_config,
    get_current_cfie_config,
    get_current_cfie_config_or_none,
    get_layers_from_cfie_config,
    set_current_cfie_config,
)
from cfie.config.weight_transfer import WeightTransferConfig

# Keep upstream config names as aliases while CFIE still contains partially
# migrated call sites that expect the vLLM naming surface.
VllmConfig = CfieConfig
get_current_vllm_config = get_current_cfie_config
get_current_vllm_config_or_none = get_current_cfie_config_or_none
set_current_vllm_config = set_current_cfie_config
get_layers_from_vllm_config = get_layers_from_cfie_config

# __all__ should only contain classes and functions.
# Types and globals should be imported from their respective modules.
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
