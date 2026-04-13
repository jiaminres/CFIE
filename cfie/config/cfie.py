# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import getpass
import json
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import is_dataclass
from datetime import datetime
from enum import IntEnum
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar, get_args

import torch
from pydantic import ConfigDict, Field, model_validator

import cfie.envs as envs
from cfie.logger import enable_trace_function_call, init_logger
from cfie.transformers_utils.runai_utils import is_runai_obj_uri
from cfie.utils import random_uuid
from cfie.utils.hashing import safe_hash

from .attention import AttentionConfig
from .cache import CacheConfig
from .compilation import CompilationConfig, CompilationMode, CUDAGraphMode
from .device import DeviceConfig
from .ec_transfer import ECTransferConfig
from .kernel import KernelConfig
from .kv_events import KVEventsConfig
from .kv_transfer import KVTransferConfig
from .load import LoadConfig
from .lora import LoRAConfig
from .model import ModelConfig
from .observability import ObservabilityConfig
from .offload import OffloadConfig
from .parallel import ParallelConfig
from .profiler import ProfilerConfig
from .scheduler import SchedulerConfig
from .speculative import EagleModelTypes, NgramGPUTypes, SpeculativeConfig
from .structured_outputs import StructuredOutputsConfig
from .utils import SupportsHash, config, replace
from .weight_transfer import WeightTransferConfig

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from cfie.model_executor.layers.quantization.base_config import QuantizationConfig
    from cfie.v1.kv_cache_interface import KVCacheConfig
else:
    PretrainedConfig = Any

    QuantizationConfig = Any

    KVCacheConfig = Any

logger = init_logger(__name__)


class OptimizationLevel(IntEnum):
    """优化级别枚举。"""

    O0 = 0
    """O0：不做优化。不开启编译、不使用 cudagraph，也不做其他优化，
    直接立即启动。"""
    O1 = 1
    """O1：快速优化。启用 Dynamo+Inductor 编译，以及分段（Piecewise）
    cudagraph。"""
    O2 = 2
    """O2：完整优化。在 -O1 基础上同时启用完整与分段 cudagraph。"""
    O3 = 3
    """O3：当前与 -O2 相同。"""


PerformanceMode = Literal["balanced", "interactivity", "throughput"]

# 这些占位变量原本打算依赖 model_config 动态决定，目前统一关闭。
IS_QUANTIZED = False
IS_DENSE = False


# 依赖这些属性的优化目前在所有情况下都被设为 False。
# if model_config is not None:
#     IS_QUANTIZED = lambda c: c.model_config.is_quantized()
#     IS_DENSE = lambda c: not c.model_config.is_model_moe()
# 参见：https://github.com/cfie-project/cfie/issues/25689。


def enable_norm_fusion(cfg: "CfieConfig") -> bool:
    """当 RMSNorm 或 quant FP8 自定义算子启用时开启；
    否则交由 Inductor 处理融合。"""

    # 只要 RMSNorm 或 quant_fp8 任一自定义算子打开，就启用该融合。
    return cfg.compilation_config.is_custom_op_enabled(
        "rms_norm"
    ) or cfg.compilation_config.is_custom_op_enabled("quant_fp8")


def enable_act_fusion(cfg: "CfieConfig") -> bool:
    """
    当 SiLU+Mul 或 quant FP8 自定义算子启用时开启；
    否则交由 Inductor 处理融合。
    对 FP4 模型也会启用，因为 FP4 量化总是走自定义算子，Inductor 无法融合。
    """
    # SiLU+Mul / quant_fp8 / NVFP4 任一条件满足时，都需要走激活融合路径。
    return (
            cfg.compilation_config.is_custom_op_enabled("silu_and_mul")
            or cfg.compilation_config.is_custom_op_enabled("quant_fp8")
            or (cfg.model_config is not None and cfg.model_config.is_nvfp4_quantized())
    )


def enable_allreduce_rms_fusion(cfg: "CfieConfig") -> bool:
    """当 TP > 1 且为 Hopper/Blackwell 架构并已安装 flashinfer 时启用。"""
    from cfie.platforms import current_platform
    from cfie.utils.flashinfer import has_flashinfer

    # 仅在 TP>1、CUDA Hopper/Blackwell 且 flashinfer 可用时打开该融合。
    return (
            cfg.parallel_config.tensor_parallel_size > 1
            and current_platform.is_cuda()
            and has_flashinfer()
            and (
                    current_platform.is_device_capability(100)
                    or current_platform.is_device_capability(90)
            )
            # tp-dp 组合当前存在问题：
            # https://github.com/cfie-project/cfie/issues/34458
            and cfg.parallel_config.data_parallel_size == 1
            # tp-pp 组合当前存在问题：
            # https://github.com/cfie-project/cfie/issues/35426
            and cfg.parallel_config.pipeline_parallel_size == 1
    )


def enable_rope_kvcache_fusion(cfg: "CfieConfig") -> bool:
    """当 rotary embedding 自定义算子启用且
    use_inductor_graph_partition 开启时启用。
    """
    from cfie._aiter_ops import rocm_aiter_ops

    # 当前只在 ROCm AITER + rotary_embedding custom op + graph partition 场景下启用。
    return (
            rocm_aiter_ops.is_enabled()
            and cfg.compilation_config.is_custom_op_enabled("rotary_embedding")
            and cfg.compilation_config.use_inductor_graph_partition
    )


def enable_norm_pad_fusion(cfg: "CfieConfig") -> bool:
    """当使用 AITER RMSNorm 与 AITER Triton GEMM 且
    hidden size 为 2880（即 gpt-oss）时启用；否则交由 Inductor 处理融合。"""
    from cfie._aiter_ops import rocm_aiter_ops

    # 目前这个融合只针对 gpt-oss 的 hidden_size=2880 特判。
    return (
            rocm_aiter_ops.is_rmsnorm_enabled()
            and not rocm_aiter_ops.is_triton_gemm_enabled()
            and cfg.model_config is not None
            and cfg.model_config.get_hidden_size() == 2880
    )


    # O0：全部保守关闭，优先启动速度与稳定性。
OPTIMIZATION_LEVEL_00 = {
    "compilation_config": {
        "pass_config": {
            "fuse_norm_quant": False,
            "fuse_act_quant": False,
            "fuse_allreduce_rms": False,
            "fuse_attn_quant": False,
            "enable_sp": False,
            "fuse_gemm_comms": False,
            "fuse_act_padding": False,
            "fuse_rope_kvcache": False,
        },
        "cudagraph_mode": CUDAGraphMode.NONE,
        "use_inductor_graph_partition": False,
    },
    "kernel_config": {
        "enable_flashinfer_autotune": False,
    },
}
# O1：开启基础编译与分段 cudagraph。
OPTIMIZATION_LEVEL_01 = {
    "compilation_config": {
        "pass_config": {
            "fuse_norm_quant": enable_norm_fusion,
            "fuse_act_quant": enable_act_fusion,
            "fuse_allreduce_rms": False,
            "fuse_attn_quant": False,
            "enable_sp": False,
            "fuse_gemm_comms": False,
            "fuse_act_padding": enable_norm_pad_fusion,
            "fuse_rope_kvcache": enable_rope_kvcache_fusion,
        },
        "cudagraph_mode": CUDAGraphMode.PIECEWISE,
        "use_inductor_graph_partition": False,
    },
    "kernel_config": {
        "enable_flashinfer_autotune": True,
    },
}
# O2：在 O1 基础上进一步开启更多融合与完整/分段 cudagraph。
OPTIMIZATION_LEVEL_02 = {
    "compilation_config": {
        "pass_config": {
            "fuse_norm_quant": enable_norm_fusion,
            "fuse_act_quant": enable_act_fusion,
            "fuse_allreduce_rms": enable_allreduce_rms_fusion,
            "fuse_attn_quant": IS_QUANTIZED,
            "enable_sp": IS_DENSE,
            "fuse_gemm_comms": IS_DENSE,
            "fuse_act_padding": enable_norm_pad_fusion,
            "fuse_rope_kvcache": enable_rope_kvcache_fusion,
        },
        "cudagraph_mode": CUDAGraphMode.FULL_AND_PIECEWISE,
        "use_inductor_graph_partition": False,
    },
    "kernel_config": {
        "enable_flashinfer_autotune": True,
    },
}
# O3：当前实现与 O2 等价，作为未来更激进优化预留档位。
OPTIMIZATION_LEVEL_03 = {
    "compilation_config": {
        "pass_config": {
            "fuse_norm_quant": enable_norm_fusion,
            "fuse_act_quant": enable_act_fusion,
            "fuse_allreduce_rms": enable_allreduce_rms_fusion,
            "fuse_attn_quant": IS_QUANTIZED,
            "enable_sp": IS_DENSE,
            "fuse_gemm_comms": IS_DENSE,
            "fuse_act_padding": enable_norm_pad_fusion,
            "fuse_rope_kvcache": enable_rope_kvcache_fusion,
        },
        "cudagraph_mode": CUDAGraphMode.FULL_AND_PIECEWISE,
        "use_inductor_graph_partition": False,
    },
    "kernel_config": {
        "enable_flashinfer_autotune": True,
    },
}

# 优化级别到默认配置模板的映射表。
OPTIMIZATION_LEVEL_TO_CONFIG = {
    OptimizationLevel.O0: OPTIMIZATION_LEVEL_00,
    OptimizationLevel.O1: OPTIMIZATION_LEVEL_01,
    OptimizationLevel.O2: OPTIMIZATION_LEVEL_02,
    OptimizationLevel.O3: OPTIMIZATION_LEVEL_03,
}


@config(config=ConfigDict(arbitrary_types_allowed=True))
class CfieConfig:
    """包含所有 cfie 相关配置的数据类。
    这可以简化在代码库中传递各类配置对象的过程。
    """

    # ----------------- 顶层子配置对象 -----------------
    # TODO：当 ModelConfig 的默认构造不再尝试下载模型时，改用 default_factory
    model_config: ModelConfig = Field(default=None)
    """模型配置。"""
    # cache_config 负责 KV cache / prefix cache / mamba cache 等运行期缓存设置。
    cache_config: CacheConfig = Field(default_factory=CacheConfig)
    """缓存配置。"""
    # parallel_config 负责 TP/DP/EP/EPLB/DCP 等并行拓扑。
    parallel_config: ParallelConfig = Field(default_factory=ParallelConfig)
    """并行配置。"""
    # scheduler_config 负责请求调度与 token budget。
    scheduler_config: SchedulerConfig = Field(
        default_factory=SchedulerConfig.default_factory,
    )
    """调度器配置。"""
    # device_config 负责设备类型与平台推断。
    device_config: DeviceConfig = Field(default_factory=DeviceConfig)
    """设备配置。"""
    # load_config 负责 checkpoint 加载方式。
    load_config: LoadConfig = Field(default_factory=LoadConfig)
    """加载配置。"""
    # offload_config 负责通用权重 offload 与 MoE tiered cache 预算上限。
    offload_config: OffloadConfig = Field(default_factory=OffloadConfig)
    """模型权重卸载配置。"""
    # attention_config 负责 attention backend 相关选项。
    attention_config: AttentionConfig = Field(default_factory=AttentionConfig)
    """注意力配置。"""
    # kernel_config 负责底层 kernel 与 autotune 选择。
    kernel_config: KernelConfig = Field(default_factory=KernelConfig)
    """内核配置。"""
    # LoRA 子配置；未启用时为 None。
    lora_config: LoRAConfig | None = None
    """LoRA 配置。"""
    # speculative decoding 子配置；未启用时为 None。
    speculative_config: SpeculativeConfig | None = None
    """投机解码配置。"""
    # 结构化输出配置。
    structured_outputs_config: StructuredOutputsConfig = Field(
        default_factory=StructuredOutputsConfig
    )
    """结构化输出配置。"""
    # tracing / metrics / OTEL 等可观测性配置。
    observability_config: ObservabilityConfig = Field(
        default_factory=ObservabilityConfig
    )
    """可观测性配置。"""
    # 已解析好的量化配置对象；通常在 __post_init__ 中补出。
    quant_config: QuantizationConfig | None = None
    """量化配置。"""
    # compile / cudagraph / pass 开关的总配置。
    compilation_config: CompilationConfig = Field(default_factory=CompilationConfig)
    """模型的 `torch.compile` 与 cudagraph 捕获配置。

    简写方式可通过 -cc.parameter=argument 追加编译参数，
    例如 `-cc.mode=3`（等价于 `-cc='{"mode":3}'`）。

    也可以像下面这样指定完整编译配置：
    `{"mode": 3, "cudagraph_capture_sizes": [1, 2, 4, 8]}`
    """
    # profiler 配置。
    profiler_config: ProfilerConfig = Field(default_factory=ProfilerConfig)
    """性能分析配置。"""
    # KV transfer 配置。
    kv_transfer_config: KVTransferConfig | None = None
    """分布式 KV 缓存传输配置。"""
    # KV 事件配置。
    kv_events_config: KVEventsConfig | None = None
    """事件发布配置。"""
    # EC transfer 配置。
    ec_transfer_config: ECTransferConfig | None = None
    """分布式 EC 缓存传输配置。"""
    # 用于挂接树外扩展信息的附加配置容器。
    # 一些不透明配置，仅用于为哈希计算提供附加信息，
    # 主要用于测试、调试或树外配置注册。
    additional_config: dict | SupportsHash = Field(default_factory=dict)
    """特定平台的附加配置。不同平台支持的配置可能不同。
    请确保配置对你使用的平台有效。内容必须可哈希。"""
    # 当前实例的唯一 ID。
    instance_id: str = ""
    """vLLM 实例 ID。"""
    # 用于统一控制 compile / cudagraph 默认档位的优化级别。
    optimization_level: OptimizationLevel = OptimizationLevel.O2
    """优化级别。这些级别在启动耗时与性能之间做权衡：
    -O0 启动最快，-O3 性能最佳。默认使用 -O2。
    完整说明见 OptimizationLevel。"""

    # 运行时偏交互还是偏吞吐的性能模式。
    performance_mode: PerformanceMode = "balanced"
    """运行时性能模式，默认是 'balanced'。
    'interactivity' 在小 batch 下更偏向更低的端到端单请求时延
    （更细粒度 CUDA 图、偏时延的内核）。
    'throughput' 在高并发下更偏向更高总体 tokens/sec
    （更大的 CUDA 图、更激进的批处理、偏吞吐的内核）。"""

    # RL 训练期间使用的权重传输配置。
    weight_transfer_config: WeightTransferConfig | None = None
    """RL 训练期间的权重传输配置。"""

    # 优雅关闭等待时间。
    shutdown_timeout: int = Field(default=0, ge=0)
    """在途请求的优雅关闭宽限期。关闭会最多延迟这么久，
    以便已在运行的请求完成。超时后剩余请求会被中止。
    """

    def compute_hash(self) -> str:
        """
        警告：每当为该配置新增字段时，
        如果它会影响计算图，请确保将其纳入 factors 列表。

        生成一个哈希，唯一标识所有会影响计算图结构的配置。
        范围为从输入 ids/embeddings 到最终 hidden states 的计算图，
        不包括输入 ids/embeddings 之前及最终 hidden states 之后的部分。
        """
        # factors 最外层仍保持列表，便于延续历史哈希结构。
        factors: list[Any] = []

        # ----------------- 逐个汇总所有会影响计算图与执行形态的子配置哈希 -----------------
        # 汇总 cfie 配置
        cfie_factors: list[Any] = []
        from cfie import __version__

        # 把 cfie 自身版本号纳入哈希，避免跨版本误复用缓存。
        cfie_factors.append(__version__)
        if self.model_config:
            # model_config 是最核心的图结构因子。
            cfie_factors.append(self.model_config.compute_hash())
            if (
                    self.compilation_config
                    and getattr(self.compilation_config, "compile_mm_encoder", False)
                    and self.model_config.multimodal_config
            ):
                # 若编译多模态 encoder，也把 multimodal_config 的哈希一并纳入。
                cfie_factors.append(self.model_config.multimodal_config.compute_hash())
        else:
            cfie_factors.append("None")
        if self.cache_config:
            cfie_factors.append(self.cache_config.compute_hash())
        else:
            cfie_factors.append("None")
        if self.parallel_config:
            cfie_factors.append(self.parallel_config.compute_hash())
        else:
            cfie_factors.append("None")
        if self.scheduler_config:
            cfie_factors.append(self.scheduler_config.compute_hash())
        else:
            cfie_factors.append("None")
        if self.device_config:
            cfie_factors.append(self.device_config.compute_hash())
        else:
            cfie_factors.append("None")
        if self.load_config:
            cfie_factors.append(self.load_config.compute_hash())
        else:
            cfie_factors.append("None")
        if self.offload_config:
            cfie_factors.append(self.offload_config.compute_hash())
        else:
            cfie_factors.append("None")
        if self.attention_config:
            cfie_factors.append(self.attention_config.compute_hash())
        else:
            cfie_factors.append("None")
        if self.lora_config:
            cfie_factors.append(self.lora_config.compute_hash())
        else:
            cfie_factors.append("None")
        if self.speculative_config:
            cfie_factors.append(self.speculative_config.compute_hash())
        else:
            cfie_factors.append("None")
        if self.structured_outputs_config:
            cfie_factors.append(self.structured_outputs_config.compute_hash())
        if self.profiler_config:
            cfie_factors.append(self.profiler_config.compute_hash())
        else:
            cfie_factors.append("None")
        cfie_factors.append(self.observability_config.compute_hash())
        if self.quant_config:
            pass  # should be captured by model_config.quantization
        if self.compilation_config:
            cfie_factors.append(self.compilation_config.compute_hash())
        else:
            cfie_factors.append("None")
        if self.kv_transfer_config:
            cfie_factors.append(self.kv_transfer_config.compute_hash())
        else:
            cfie_factors.append("None")
        if self.ec_transfer_config:
            cfie_factors.append(self.ec_transfer_config.compute_hash())
        else:
            cfie_factors.append("None")
        if self.additional_config:
            if isinstance(additional_config := self.additional_config, dict):
                # dict 形态的 additional_config 直接按 JSON 排序序列化后求哈希。
                additional_config_hash = safe_hash(
                    json.dumps(additional_config, sort_keys=True).encode(),
                    usedforsecurity=False,
                ).hexdigest()
            else:
                # 否则要求该对象自己实现 SupportsHash。
                additional_config_hash = additional_config.compute_hash()
            cfie_factors.append(additional_config_hash)
        else:
            cfie_factors.append("None")
        factors.append(cfie_factors)

        # 最终仍沿用 safe_hash，并截断成 10 位短哈希。
        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()[
            :10
        ]
        return hash_str

    @property
    def num_speculative_tokens(self) -> int:
        """返回投机解码 token 数；若未配置则返回 0。"""
        # 仅当 speculative_config 存在且显式设置了 token 数时返回该值。
        if (
                self.speculative_config is not None
                and self.speculative_config.num_speculative_tokens is not None
        ):
            return self.speculative_config.num_speculative_tokens
        # 未启用投机解码时返回 0。
        return 0

    @property
    def needs_dp_coordinator(self) -> bool:
        """
        判断是否需要 DPCoordinator 进程。

        DPCoordinator 在以下两种场景需要启用：
        1. DP > 1 的 MoE 模型：用于处理 wave 协调
           （即使是外部 LB 模式也需要，因为 wave 协调在 coordinator 中执行）
        2. 非 MoE 模型且使用 internal/hybrid LB 模式：用于采集并发布
           队列统计信息，以便在各 DP rank 间做负载均衡

        返回：
            若需要 DPCoordinator 进程则为 True，否则为 False。
        """

        # 对非 MoE 模型，只在 internal/hybrid LB 模式下需要 coordinator（用于统计采集）。
        return self.parallel_config.data_parallel_size > 1 and (
                self.model_config is None
                or self.model_config.is_moe
                or not self.parallel_config.data_parallel_external_lb
        )

    def enable_trace_function_call_for_thread(self) -> None:
        """
        若通过环境变量 `VLLM_TRACE_FUNCTION` 启用，
        则为当前线程开启函数调用追踪。
        """
        # 只有开启环境变量时，才为当前线程生成 trace 文件。
        if envs.VLLM_TRACE_FUNCTION:
            tmp_dir = tempfile.gettempdir()
            # 在 tmp_dir 中加入用户名，避免权限问题
            tmp_dir = os.path.join(tmp_dir, getpass.getuser())
            # 文件名里带上进程、线程和时间戳，方便区分多线程 trace。
            filename = (
                f"VLLM_TRACE_FUNCTION_for_process_{os.getpid()}"
                f"_thread_{threading.get_ident()}_at_{datetime.now()}.log"
            ).replace(" ", "_")
            log_path = os.path.join(
                tmp_dir,
                "cfie",
                f"cfie-instance-{self.instance_id}",
                filename,
            )
            # 提前创建目录，再真正打开 trace。
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            enable_trace_function_call(log_path)

    @staticmethod
    def _get_quantization_config(
            model_config: ModelConfig, load_config: LoadConfig
    ) -> QuantizationConfig | None:
        """获取量化配置。"""
        # 延迟导入当前平台对象，用于读取 GPU 能力信息。
        from cfie.platforms import current_platform

        # ----------------- 仅当模型显式声明了 quantization 时才解析量化配置 -----------------
        # 只有在模型显式声明 quantization 方法时，才需要构建量化配置。
        if model_config.quantization is not None:
            # 延迟导入量化配置解析函数，避免模块级循环依赖与不必要开销。
            from cfie.model_executor.model_loader.weight_utils import get_quant_config

            # 根据模型配置与加载配置，解析出具体的量化配置对象。
            quant_config = get_quant_config(model_config, load_config)
            # 读取当前设备的计算能力（compute capability）。
            capability_tuple = current_platform.get_device_capability()

            # ----------------- 校验当前设备是否支持该量化方法 -----------------
            # 若当前平台能返回设备能力，则继续做“量化方法是否支持该 GPU”的校验。
            if capability_tuple is not None:
                # 把 capability 从平台对象转换成整数形式，便于比较。
                capability = capability_tuple.to_int()
                # 若当前 GPU 能力低于该量化方法要求的最小能力，则直接报错。
                if capability < quant_config.get_min_capability():
                    raise ValueError(
                        f"The quantization method {model_config.quantization} "
                        "is not supported for the current GPU. Minimum "
                        f"capability: {quant_config.get_min_capability()}. "
                        f"Current capability: {capability}."
                    )
            # ----------------- 再校验当前模型 dtype 是否被该量化方法支持 -----------------
            # 读取该量化方法允许使用的激活 dtype 集合。
            supported_dtypes = quant_config.get_supported_act_dtypes()
            # 若当前模型 dtype 不在支持集合里，则直接报错。
            if model_config.dtype not in supported_dtypes:
                raise ValueError(
                    f"{model_config.dtype} is not supported for quantization "
                    f"method {model_config.quantization}. Supported dtypes: "
                    f"{supported_dtypes}"
                )
            # ----------------- 最后让量化配置根据模型目录补全细节 -----------------
            # 某些量化配置还需要基于模型目录内容进一步修正自身参数，这里执行一次补全。
            quant_config.maybe_update_config(model_config.model)
            # 返回最终可用的量化配置对象。
            return quant_config
        # 若模型根本未启用量化，则返回 None。
        return None

    @staticmethod
    def get_quantization_config(
            model_config: ModelConfig, load_config: LoadConfig
    ) -> QuantizationConfig | None:
        import copy

        # _get_quantization_config 内部会修改 model_config，因此这里先 deepcopy 一份。
        # 出于某些原因，带下划线的版本会修改 model_config 对象，
        # 因此这里用 deepcopy 避免该问题。
        return CfieConfig._get_quantization_config(
            copy.deepcopy(model_config), load_config
        )

    def with_hf_config(
            self,
            hf_config: PretrainedConfig,
            architectures: list[str] | None = None,
    ) -> "CfieConfig":
        """返回一个替换了 hf_config（可选替换 architectures）的新配置对象。"""
        # 若调用方传了新的 architectures，则先复制 hf_config 再覆写。
        if architectures is not None:
            hf_config = copy.deepcopy(hf_config)
            hf_config.architectures = architectures

        # 复制一份 model_config，避免直接污染原配置对象。
        model_config = copy.deepcopy(self.model_config)

        if (
                model_config.is_multimodal_model
                and hasattr(model_config.hf_config, "tie_word_embeddings")
                and not hasattr(hf_config.get_text_config(), "tie_word_embeddings")
        ):
            # ----------------- 多模态模型下，手工把 tie_word_embeddings 从顶层 config 透传到 text_config -----------------
            # 在 Transformers v5 中，tie_word_embeddings 属于能同时看到
            # 两个待绑定层的那个类的配置。例如：
            #
            # SomeVLModel:
            #   self.language_model = SomeLanguageModel()
            #   self.vision_model = SomeVisionModel()
            #
            # SomeVLModelForMultimodalLM:
            #   self.model = SomeVLModel()
            #   self.lm_head = nn.Linear()
            #
            # 因此，tie_word_embeddings 定义在 SomeVLModelForMultimodalLM 的
            # config 中，而不在 SomeVLModel 的 config 中。在 vLLM 里，lm_head
            # 属于 language_model，所以必须确保 language_model 的 config 中
            # 正确设置了 tie_word_embeddings。
            tie_word_embeddings = model_config.hf_config.tie_word_embeddings
            hf_config.get_text_config().tie_word_embeddings = tie_word_embeddings

        # 用新的 hf_config 刷新 model_config 及其派生的 model_arch_config。
        model_config.hf_config = hf_config
        model_config.model_arch_config = model_config.get_model_arch_config()

        # 返回一个仅替换了 model_config 的新 CfieConfig。
        return replace(self, model_config=model_config)

    def _set_config_default(self, config_obj: Any, key: str, value: Any) -> None:
        """当用户未设置时，将配置属性设为默认值。

        参数：
            config_obj：要更新的配置对象。
            key：属性名。
            value：默认值（静态值或可调用对象）。
        """
        # 只有在字段仍为 None 时，才把优化级别/模式推导出的默认值写进去。
        if getattr(config_obj, key) is None:
            # 有些配置值在初始化前就已确定，属于硬编码默认值。
            # 其他值依赖用户配置，故使用 lambda 在运行时决定。
            setattr(config_obj, key, value(self) if callable(value) else value)

    def _apply_optimization_level_defaults(self, defaults: dict[str, Any]) -> None:
        """以 self 为根对象应用优化级别默认值。

        递归地把 defaults 中的值写入嵌套配置对象。
        仅处理 defaults 中存在的字段。

        如果用户配置未显式指定某个默认字段，
        且应用完用户选择后该字段仍为 None，
        则会应用默认值。用户显式设置的字段不会被默认值覆盖。

        参数：
            defaults：要应用的默认值字典。
        """

        def apply_recursive(config_obj: Any, config_defaults: dict[str, Any]) -> None:
            """以 self 为根对象，递归地向 config_obj 应用默认值。"""
            for key, value in config_defaults.items():
                # defaults 中不存在于目标配置对象上的字段直接跳过。
                if not hasattr(config_obj, key):
                    continue

                current = getattr(config_obj, key)
                # 若当前字段还是 dataclass 且默认值也是嵌套字典，则递归下钻。
                if isinstance(value, dict) and is_dataclass(current):
                    apply_recursive(current, value)
                else:
                    self._set_config_default(config_obj, key, value)

        apply_recursive(self, defaults)

    def _post_init_kv_transfer_config(self) -> None:
        """基于 CfieConfig 顶层配置更新 KVTransferConfig。

        当前该函数会读取 CacheConfig 中的 offloading 设置，
        并据此配置 KVTransferConfig。
        """
        # 只有设置了 kv_offloading_size，才需要自动补 KV transfer 配置。
        # 仅在设置了 kv_offloading_size 时启用 KV offloading。
        if (kv_offloading_size := self.cache_config.kv_offloading_size) is None:
            return

        # 读取当前选用的 KV offloading backend。
        kv_offloading_backend = self.cache_config.kv_offloading_backend

        # 若用户没有单独提供 KVTransferConfig，则先创建一个默认实例。
        # 若未提供 KVTransferConfig，则创建一个默认实例。
        if self.kv_transfer_config is None:
            self.kv_transfer_config = KVTransferConfig()
        # KV rank 总数按 TP * PP 计算。
        num_kv_ranks = (
                self.parallel_config.tensor_parallel_size
                * self.parallel_config.pipeline_parallel_size
        )

        # native backend 直接走 OffloadingConnector，并把总 CPU 字节预算写进去。
        if kv_offloading_backend == "native":
            self.kv_transfer_config.kv_connector = "OffloadingConnector"
            self.kv_transfer_config.kv_connector_extra_config.update(
                {"cpu_bytes_to_use": kv_offloading_size * (1 << 30)}
            )
        # lmcache backend 则按 KV ranks 均分总预算。
        elif kv_offloading_backend == "lmcache":
            self.kv_transfer_config.kv_connector = "LMCacheConnectorV1"
            kv_gb_per_rank = kv_offloading_size / num_kv_ranks
            self.kv_transfer_config.kv_connector_extra_config = {
                "lmcache.local_cpu": True,
                "lmcache.max_local_cpu_size": kv_gb_per_rank,
            }

        # 当前所有 backend 最终都把本实例视为既可发送又可接收 KV 的节点。
        # 所有后端该设置一致
        self.kv_transfer_config.kv_role = "kv_both"

    def __post_init__(self):
        """校验各配置是否合法且彼此一致。"""

        # ----------------- 基础实例初始化与预校验 -----------------
        # --------------- 基础实例初始化与预校验 ---------------
        # 为当前这份配置生成一个唯一实例 ID。
        # 这个 ID 常用于 profile / debug / 日志关联。
        self.instance_id = f"{time.time_ns()}"

        # 若用户选择的性能模式不是默认 balanced，则打印一次提示日志。
        if self.performance_mode != "balanced":
            logger.info_once(
                "Performance mode set to '%s'.", self.performance_mode, scope="local"
            )

        # 先执行一轮“基础配置修正 + 合法性检查”。
        # 这里会处理很多跨子配置的默认值与兼容性细节。
        self.try_verify_and_update_config()

        # --------------- 模型配置与并行配置对齐 ---------------
        # 仅当 model_config 已经存在时，继续执行模型相关校验。
        if self.model_config is not None:
            # 检查模型配置与并行配置是否匹配。
            self.model_config.verify_with_parallel_config(self.parallel_config)
            # 检查 dual chunk attention 相关配置是否合法。
            self.model_config.verify_dual_chunk_attention_config(self.load_config)

            # 把“当前模型是否为 MoE”这一事实回写到 parallel_config，
            # 供后续调度器、执行器、offload planner 统一使用。
            self.parallel_config.is_moe_model = self.model_config.is_moe

        # 若启用了 LoRA，则检查 LoRA 配置是否与目标模型兼容。
        if self.lora_config is not None:
            self.lora_config.verify_with_model_config(self.model_config)

        # ----------------- target / draft 最大长度对齐 -----------------
        # --------------- target / draft 最大长度对齐 ---------------
        # 当存在 speculative decoding 且带 draft model 时，
        # 需要保证 draft 的最大长度不超过 target model。
        if (
            self.model_config is not None
            and self.speculative_config is not None
            and self.speculative_config.draft_model_config is not None
        ):
            # 取出 draft model 配置对象。
            draft_model_config = self.speculative_config.draft_model_config
            # 目标模型的最终最大上下文长度。
            target_max_model_len = int(self.model_config.max_model_len)
            # 读取 draft 当前配置的最大长度；如果没有则返回 None。
            draft_max_model_len = getattr(draft_model_config, "max_model_len", None)
            # 若 draft 未显式声明最大长度，则直接继承 target 的长度。
            if draft_max_model_len is None:
                draft_model_config.max_model_len = target_max_model_len
            # 若 draft 长度比 target 更大，则强制裁剪到 target 上限。
            elif int(draft_max_model_len) > target_max_model_len:
                draft_model_config.max_model_len = target_max_model_len

        # 在日志中输出最终生效的 max_model_len，便于排查 target/draft 不一致问题。
        if self.model_config is not None:
            # 读取 target model 最终长度。
            target_max_model_len = int(self.model_config.max_model_len)
            # 若同时存在 draft model，则一起打印 target/draft 的最终结果。
            if (
                self.speculative_config is not None
                and self.speculative_config.draft_model_config is not None
            ):
                # 取出 draft model 配置。
                draft_model_config = self.speculative_config.draft_model_config
                # 读取 draft 最终长度；若字段缺失则退回 target 长度。
                draft_max_model_len = getattr(
                    draft_model_config, "max_model_len", target_max_model_len
                )
                # 记录 target / draft 的最终长度。
                logger.info(
                    "Resolved final max model len: target=%d draft=%d",
                    target_max_model_len,
                    int(draft_max_model_len),
                )
            else:
                # 若没有 draft，则只记录 target 的最终长度。
                logger.info(
                    "Resolved final max model len: target=%d",
                    target_max_model_len,
                )

        # ----------------- 量化配置补全与 MoE offload plan 注入 -----------------
        # --------------- 量化配置补全与 MoE offload plan 注入 ---------------
        # 若尚未显式生成 quant_config，则基于 model/load 配置自动推导。
        if self.quant_config is None and self.model_config is not None:
            self.quant_config = CfieConfig._get_quantization_config(
                self.model_config,
                self.load_config
            )

        # 当模型配置与量化配置都齐备后，尝试为当前这份 CfieConfig 注入 MoE tiered cache plan。
        # 对 target 来说，这里就是主入口：
        # 1. 先构建 target 自己的 plan
        # 2. 若 spec_method=mtp，会在 plan 内部递归预估一份 reserve-only 的 draft plan
        # 3. 真正加载 draft/MTP 模型时，再由 draft 自己那份 CfieConfig 单独重建 actual plan
        if self.model_config is not None and self.quant_config is not None:
            from cfie.offload.policy import maybe_inject_moe_tiered_cache_plan

            maybe_inject_moe_tiered_cache_plan(self)

        # ----------------- 异步调度能力判定 -----------------
        # --------------- 异步调度能力判定 ---------------
        # 读取当前分布式执行后端。
        executor_backend = self.parallel_config.distributed_executor_backend
        # 判断该执行后端是否支持 async scheduling。
        executor_supports_async_sched = executor_backend in (
            "mp",
            "uni",
            "external_launcher",
        )

        # 若用户显式开启了 async scheduling，则对所有不兼容项做“硬校验”。
        if self.scheduler_config.async_scheduling:
            # 已显式启用异步调度，若存在不兼容项则直接报错。
            # 当前异步调度仅支持 eagle 类投机解码。
            if self.speculative_config is not None:
                # 当前只有 EAGLE / MTP / Draft Model / NGram GPU 路径支持异步调度。
                if (
                        self.speculative_config.method not in get_args(EagleModelTypes)
                        and self.speculative_config.method not in get_args(NgramGPUTypes)
                        and self.speculative_config.method != "draft_model"
                ):
                    raise ValueError(
                        "Currently, async scheduling is only supported "
                        "with EAGLE/MTP/Draft Model/NGram GPU kind of "
                        "speculative decoding"
                    )
                if self.speculative_config.disable_padded_drafter_batch:
                    raise ValueError(
                        "Async scheduling is not compatible with "
                        "disable_padded_drafter_batch=True."
                    )
            # 若执行器后端本身不支持异步调度，则直接报错。
            if not executor_supports_async_sched:
                raise ValueError(
                    "Currently, async scheduling only supports `mp`, `uni`, or "
                    "`external_launcher` distributed executor backend, but you chose "
                    f"`{executor_backend}`."
                )
        # 若用户没有显式指定 async_scheduling，则由系统自动决定开关状态。
        elif self.scheduler_config.async_scheduling is None:
            # 若无不兼容项，则启用异步调度。
            # 若 speculative decoding 方法不在支持列表中，则自动关闭并给出告警。
            if (
                    self.speculative_config is not None
                    and self.speculative_config.method not in get_args(EagleModelTypes)
                    and self.speculative_config.method not in get_args(NgramGPUTypes)
            ):
                logger.warning_once(
                    "Async scheduling not supported with %s-based "
                    "speculative decoding and will be disabled.",
                    self.speculative_config.method,
                    scope="local",
                )
                self.scheduler_config.async_scheduling = False
            # disable_padded_drafter_batch 与异步调度不兼容，自动关闭。
            elif (
                    self.speculative_config is not None
                    and self.speculative_config.disable_padded_drafter_batch
            ):
                logger.warning_once(
                    "Async scheduling is not compatible with "
                    "disable_padded_drafter_batch=True and will be disabled.",
                    scope="local",
                )
                self.scheduler_config.async_scheduling = False
            # 执行器后端不支持异步调度时，也自动关闭。
            elif not executor_supports_async_sched:
                logger.warning_once(
                    "Async scheduling will be disabled because it is not supported "
                    "with the `%s` distributed executor backend (only `mp`, `uni`, and "
                    "`external_launcher` are supported).",
                    executor_backend,
                    scope="local",
                )
                self.scheduler_config.async_scheduling = False
            else:
                # 当没有检测到不兼容项时，默认启用异步调度。
                self.scheduler_config.async_scheduling = True

        # 输出一次异步调度最终启用状态。
        logger.info_once(
            "Asynchronous scheduling is %s.",
            "enabled" if self.scheduler_config.async_scheduling else "disabled",
        )

        # ----------------- DP 同步后端的默认策略 -----------------
        # --------------- DP 同步后端的默认策略 ---------------
        # 若用户没有显式指定 DP 同步是否禁用 NCCL，则根据 async scheduling 自动决定。
        if self.parallel_config.disable_nccl_for_dp_synchronization is None:
            # 异步调度下默认关闭 NCCL 做 DP 同步。
            if self.scheduler_config.async_scheduling:
                # 在 DP>1 且模型为 MoE（或未知）时，给出更明确的说明日志。
                if self.parallel_config.data_parallel_size > 1 and (
                        self.model_config is None or self.model_config.is_moe
                ):
                    logger.info_once(
                        "Disabling NCCL for DP synchronization "
                        "when using async scheduling.",
                        scope="local",
                    )
                self.parallel_config.disable_nccl_for_dp_synchronization = True
            else:
                # 非异步调度下，默认保留 NCCL 做 DP 同步。
                self.parallel_config.disable_nccl_for_dp_synchronization = False

        # ----------------- 平台相关特殊兼容处理 -----------------
        # --------------- 平台相关特殊兼容处理 ---------------
        # 延迟导入当前平台对象，避免模块级循环依赖。
        from cfie.platforms import current_platform

        # Turing + fp32 + chunked prefill 是一个已知性能/数值兼容边界，提前告警。
        if (
                self.model_config is not None
                and self.scheduler_config.enable_chunked_prefill
                and self.model_config.dtype == torch.float32
                and current_platform.get_device_capability() == (7, 5)
        ):
            logger.warning_once(
                "Turing devices tensor cores do not support float32 matmul. "
                "To workaround this limitation, vLLM will set 'ieee' input "
                "precision for chunked prefill triton kernels."
            )

        # 若模型要求强制 eager，则显式关闭 compile 与 cudagraph。
        if self.model_config is not None and self.model_config.enforce_eager:
            logger.warning(
                "Enforce eager set, disabling torch.compile and CUDAGraphs. "
                "This is equivalent to setting -cc.mode=none -cc.cudagraph_mode=none"
            )
            self.compilation_config.mode = CompilationMode.NONE
            self.compilation_config.cudagraph_mode = CUDAGraphMode.NONE

        # 若最终不是 inductor + vLLM compile 路径，则提示相关优化项会被忽略。
        if self.compilation_config.backend == "eager" or (
                self.compilation_config.mode is not None
                and self.compilation_config.mode != CompilationMode.VLLM_COMPILE
        ):
            logger.warning(
                "Inductor compilation was disabled by user settings, "
                "optimizations settings that are only active during "
                "inductor compilation will be ignored."
            )

        # --------------- 阻塞式权重(blocked weights)检测与 quant_fp8 内核开关 ---------------
        # 定义一个局部辅助函数，用于判断当前量化权重是否是 blocked layout。
        def has_blocked_weights():
            # 判断是否启用了按 block 存储的权重。
            if self.quant_config is not None:
                # 新式接口：直接检查 weight_block_size 字段。
                if hasattr(self.quant_config, "weight_block_size"):
                    return self.quant_config.weight_block_size is not None
                # 旧式接口：通过 has_blocked_weights() 方法判断。
                elif hasattr(self.quant_config, "has_blocked_weights"):
                    return self.quant_config.has_blocked_weights()
            return False

        # 启用 quant_fp8 CUDA 算子（TODO：后续再考虑关闭）
        # 在 H100 上 CUDA 内核比
        # 原生实现更快
        # https://github.com/cfie-project/cfie/issues/25094
        # 若检测到 blocked weights，则确保自定义算子列表中包含 quant_fp8。
        if has_blocked_weights():
            custom_ops = self.compilation_config.custom_ops
            if "-quant_fp8" not in custom_ops:
                custom_ops.append("+quant_fp8")

        # 让平台按自身能力补齐默认值或修正某些配置开关。
        # 根据平台特性补齐平台默认配置。
        current_platform.apply_config_platform_defaults(self)

        # ----------------- 编译模式与优化等级默认值补全 -----------------
        # --------------- 编译模式与优化等级默认值补全 ---------------
        # 若 compilation mode 还未定下来，则按 optimization_level 补默认值。
        if self.compilation_config.mode is None:
            if self.optimization_level > OptimizationLevel.O0:
                self.compilation_config.mode = CompilationMode.VLLM_COMPILE
            else:
                self.compilation_config.mode = CompilationMode.NONE

        # 若 custom_ops 里既没有 all 也没有 none，则补一个默认策略。
        if all(s not in self.compilation_config.custom_ops for s in ("all", "none")):
            if (
                    self.compilation_config.backend == "inductor"
                    and self.compilation_config.mode != CompilationMode.NONE
            ):
                # 走 inductor 且启用了编译时，默认让 pass 自己选择 custom op。
                self.compilation_config.custom_ops.append("none")
            else:
                # 其他情况默认开启全部 custom op。
                self.compilation_config.custom_ops.append("all")

        # 读取当前 optimization level 对应的默认参数模板。
        default_config = OPTIMIZATION_LEVEL_TO_CONFIG[self.optimization_level]
        # 把优化等级默认值落到各个子配置上，但不覆盖用户显式指定值。
        self._apply_optimization_level_defaults(default_config)
        # 到这一步后，flashinfer autotune 必须已经被填充。
        if self.kernel_config.enable_flashinfer_autotune is None:
            raise ValueError(
                "KernelConfig.enable_flashinfer_autotune must be set after applying "
                "optimization level defaults."
            )

        # 若 cudagraph mode 需要 piecewise compile，但 compilation mode 不支持，
        # 则将 cudagraph mode 回退为 NONE。
        if (
                self.compilation_config.cudagraph_mode.requires_piecewise_compilation()
                and self.compilation_config.mode != CompilationMode.VLLM_COMPILE
        ):
            logger.info(
                "Cudagraph mode %s is not compatible with compilation mode %s."
                "Overriding to NONE.",
                self.compilation_config.cudagraph_mode,
                self.compilation_config.mode,
            )
            # 不兼容时把 cudagraph mode 直接回退为 NONE。
            self.compilation_config.cudagraph_mode = CUDAGraphMode.NONE

        # ----------------- 序列并行 / 通信融合相关设置 -----------------
        # --------------- 序列并行 / 通信融合相关设置 ---------------
        # async TP 构建在序列并行之上，
        # 因此需要启用序列并行。
        # 若启用了 gemm 通信融合，则强制启用序列并行。
        if self.compilation_config.pass_config.fuse_gemm_comms:
            self.compilation_config.pass_config.enable_sp = True
        # 若序列并行已启用，则继续校验其前提条件。
        if self.compilation_config.pass_config.enable_sp:
            # TP=1 时序列并行无意义，直接关闭。
            if self.parallel_config.tensor_parallel_size == 1:
                logger.warning("Sequence Parallelism requires TP>1, disabling")
                self.compilation_config.pass_config.enable_sp = False
                self.compilation_config.pass_config.fuse_gemm_comms = False
            else:
                # 提前计算 SP 阈值；若为 None 则禁用（模型太小，SP 无收益）。
                pass_config = self.compilation_config.pass_config
                # 若用户未显式指定 SP 最小 token 阈值，则使用启发式计算。
                if pass_config.sp_min_token_num is None:
                    from cfie.compilation.passes.fusion.sequence_parallelism import (
                        get_sequence_parallelism_threshold,
                    )

                    # 读取启发式所需的 TP 大小、hidden size 与元素字节数。
                    tp_size = self.parallel_config.tensor_parallel_size
                    hidden_size = self.model_config.get_hidden_size()
                    element_size = self.model_config.dtype.itemsize
                    # 计算建议的 SP 触发阈值。
                    pass_config.sp_min_token_num = get_sequence_parallelism_threshold(
                        hidden_size, tp_size, element_size
                    )

                # 若阈值启发式返回 None，说明模型太小，不值得开启 SP。
                if pass_config.sp_min_token_num is None:
                    logger.warning(
                        "Model hidden_size too small for the SP "
                        "threshold heuristic, disabling. To force SP, "
                        "set pass_config.sp_min_token_num manually."
                    )
                    self.compilation_config.pass_config.enable_sp = False
                    self.compilation_config.pass_config.fuse_gemm_comms = False

        # ----------------- fast_moe_cold_start 默认值决议 -----------------
        # --------------- fast_moe_cold_start 默认值决议 ---------------
        # 延迟导入 torch 能力标记。
        from cfie.utils.torch_utils import HAS_OPAQUE_TYPE

        # torch >= 2.11 时使用 OpaqueObject 路径，不再依赖 fast_moe_cold_start。
        if HAS_OPAQUE_TYPE:
            # 在 torch >= 2.11 上，提升后的 OpaqueObject 方案取代了
            # fast_moe_cold_start，因此强制关闭后者。
            self.compilation_config.fast_moe_cold_start = False
        # 仅当用户未显式指定时，才推导 fast_moe_cold_start 的默认值。
        elif self.compilation_config.fast_moe_cold_start is None:
            # 解析默认行为：尽量选择更安全的配置。
            # 若任一投机解码 draft model 含 MOE，该配置并不安全。
            # 因此只要检测到投机解码就保守地关闭。
            self.compilation_config.fast_moe_cold_start = (
                    self.speculative_config is None
            )

        # 按 speculative decoding 额外占位情况，修正调度器可调度 token 上限。
        self._set_max_num_scheduled_tokens()

        # ----------------- CUDA Graph / Static Graph 最终裁决 -----------------
        # --------------- CUDA Graph / Static Graph 最终裁决 ---------------
        # 若当前平台支持 static graph，再继续做 cudagraph 相关检查。
        if current_platform.support_static_graph_mode():
            # 若 cudagraph_mode 含完整 cudagraph，需要进一步检查是否支持
            if model_config := self.model_config:
                # pooling 模型不支持 full cudagraph，必要时降级为 PIECEWISE。
                if (
                        self.compilation_config.cudagraph_mode.has_full_cudagraphs()
                        and model_config.pooler_config is not None
                ):
                    logger.warning_once(
                        "Pooling models do not support full cudagraphs. "
                        "Overriding cudagraph_mode to PIECEWISE."
                    )
                    self.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
                # encoder-decoder 仅支持 NONE / FULL_DECODE_ONLY。
                elif (
                        model_config.is_encoder_decoder
                        and self.compilation_config.cudagraph_mode
                        not in (CUDAGraphMode.NONE, CUDAGraphMode.FULL_DECODE_ONLY)
                ):
                    logger.info_once(
                        "Encoder-decoder models do not support %s. "
                        "Overriding cudagraph_mode to FULL_DECODE_ONLY.",
                        self.compilation_config.cudagraph_mode.name,
                    )
                    self.compilation_config.cudagraph_mode = (
                        CUDAGraphMode.FULL_DECODE_ONLY
                    )

            # 检查 KV connector 是否要求 CUDA graph 使用 PIECEWISE 模式
            # 若 KV connector 内部存在无法被完整捕获的 layerwise async 操作，
            # 则需要把 full cudagraph 降级成 piecewise。
            if (
                    self.kv_transfer_config is not None
                    and self.kv_transfer_config.is_kv_transfer_instance
                    and self.compilation_config.cudagraph_mode.has_full_cudagraphs()
            ):
                # 延迟导入以避免循环依赖
                from cfie.distributed.kv_transfer.kv_connector.factory import (
                    KVConnectorFactory,
                )

                connector_cls = KVConnectorFactory.get_connector_class(
                    self.kv_transfer_config
                )
                # connector 若声明“full cudagraph 不安全”，则强制降级。
                if connector_cls.requires_piecewise_for_cudagraph(
                        self.kv_transfer_config.kv_connector_extra_config
                ):
                    logger.warning_once(
                        "KV connector %s requires PIECEWISE CUDA graph mode "
                        "due to layerwise async operations that cannot be "
                        "captured in CUDA graphs. "
                        "Overriding cudagraph_mode from %s to PIECEWISE.",
                        connector_cls.__name__,
                        self.compilation_config.cudagraph_mode.name,
                    )
                    self.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE

            # 在强制 eager 执行时禁用 cudagraph
            # enforce_eager 优先级最高，最终直接把 cudagraph 全部关闭。
            if self.model_config is not None and self.model_config.enforce_eager:
                logger.info("Cudagraph is disabled under eager mode")
                self.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
                # 在强制 eager 时覆盖相关设置
                self.compilation_config.max_cudagraph_capture_size = 0
                self.compilation_config.cudagraph_capture_sizes = []
            else:
                # 非 eager 模式下，默认预留 1 次 warmup 给 cudagraph。
                self.compilation_config.cudagraph_num_of_warmups = 1

            # 根据当前配置生成最终 cudagraph capture size 列表。
            self._set_cudagraph_sizes()
        else:
            # 平台不支持 static graph，则彻底关闭 cudagraph。
            self.compilation_config.cudagraph_mode = CUDAGraphMode.NONE

        # ----------------- KV sharing fast prefill 与编译范围设置 -----------------
        # --------------- KV sharing fast prefill 与编译范围设置 ---------------
        # 若启用了 KV sharing fast prefill，则先检查和 EAGLE 的兼容性。
        if self.cache_config.kv_sharing_fast_prefill:
            if (
                    self.speculative_config is not None
                    and self.speculative_config.use_eagle()
            ):
                raise ValueError(
                    "Fast prefill optimization for KV sharing is not "
                    "compatible with EAGLE as EAGLE requires correct logits "
                    "for all tokens while fast prefill gives incorrect logits "
                    "for prompt tokens."
                )

            logger.warning_once(
                "--kv-sharing-fast-prefill requires changes on model side for "
                "correctness and to realize prefill savings."
            )
        # TODO：在 https://github.com/cfie-project/cfie/pull/26847 合入后再移动
        # 根据当前 compile / cudagraph / backend 状态生成 compile ranges。
        self._set_compile_ranges()

        # Whisper 在 fork worker 场景下已知容易卡启动，提前给出诊断提示。
        if (
                self.model_config
                and self.model_config.architecture == "WhisperForConditionalGeneration"
                and os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") != "spawn"
        ):
            logger.warning(
                "Whisper is known to have issues with "
                "forked workers. If startup is hanging, "
                "try setting 'VLLM_WORKER_MULTIPROC_METHOD' "
                "to 'spawn'."
            )

        # --------------- KV events 与 prefix caching 关系校验 ---------------
        # 打开 KV cache events 但没开 prefix caching 时，提醒用户。
        if (
                self.kv_events_config is not None
                and self.kv_events_config.enable_kv_cache_events
                and not self.cache_config.enable_prefix_caching
        ):
            logger.warning(
                "KV cache events are on, but prefix caching is not enabled. "
                "Use --enable-prefix-caching to enable."
            )
        # 若 publisher 配了非 null，但 KV cache events 本身没打开，也提醒用户。
        if (
                self.kv_events_config is not None
                and self.kv_events_config.publisher != "null"
                and not self.kv_events_config.enable_kv_cache_events
        ):
            logger.warning(
                "KV cache events are disabled, "
                "but the scheduler is configured to publish them. "
                "Modify KVEventsConfig.enable_kv_cache_events "
                "to True to enable."
            )
        # 让平台执行最后一轮 platform-specific config 修正。
        current_platform.check_and_update_config(self)

        # ----------------- V1 splitting ops 与 SP 特殊处理 -----------------
        # --------------- V1 splitting ops 与 SP 特殊处理 ---------------
        # 这一步需放在 compilation_config.mode 完成所有更新之后
        # dense 模型在 DP 场景下不需要按真实 data_parallel_size 参与这里的切分决策。
        effective_dp_size = (
            self.parallel_config.data_parallel_size
            if self.model_config is None or self.model_config.is_moe
            else 1
        )
        # 根据 all2all backend 与 effective DP size 生成 v1 splitting ops 配置。
        self.compilation_config.set_splitting_ops_for_v1(
            all2all_backend=self.parallel_config.all2all_backend,
            data_parallel_size=effective_dp_size,
        )

        # 若启用了序列并行，还需要进一步处理 rms_norm 相关兼容项。
        if self.compilation_config.pass_config.enable_sp:
            # 在流水线并行或 dynamo 分区下，
            # 原生 rms norm 追踪会因 residual 形状不正确而报错。
            # 这里使用自定义 rms norm 规避。未来该 pass
            # 会基于更高层 IR 运行，以避免该问题。
            # TODO: https://github.com/cfie-project/cfie/issues/27894
            # 若当前并非 vLLM compile 模式，则先给出不匹配告警。
            if self.compilation_config.mode != CompilationMode.VLLM_COMPILE:
                logger.warning(
                    "Sequence parallelism is enabled, but running in wrong "
                    "cfie compile mode: %s.",
                    self.compilation_config.mode,
                )

            # 判定当前执行图是否可视为 fullgraph。
            is_fullgraph = (
                    self.compilation_config.use_inductor_graph_partition
                    or len(self.compilation_config.splitting_ops) == 0
            )
            # 只要开启了 PP 或不是 fullgraph，就需要强制启用自定义 rms_norm。
            if self.parallel_config.pipeline_parallel_size > 1 or not is_fullgraph:
                if "-rms_norm" not in self.compilation_config.custom_ops:
                    self.compilation_config.custom_ops.append("+rms_norm")
                else:
                    # 若用户显式关掉了自定义 rms_norm，则给出更强的风险告警。
                    regime = (
                        "Dynamo partition"
                        if not is_fullgraph
                        else "pipeline parallelism"
                    )
                    logger.warning_once(
                        "Sequence parallelism not supported with "
                        "native rms_norm when using %s, "
                        "this will likely lead to an error.",
                        regime,
                    )

        # ----------------- CUDA 平台下的最终 cudagraph 断言与告警 -----------------
        # --------------- CUDA 平台下的最终 cudagraph 断言与告警 ---------------
        # 在所有可能更新完成后，对 cudagraph mode 做最终检查
        if current_platform.is_cuda_alike():
            # 若只启用了 full cudagraph，且可能遇到 cascade attention，则提示会回退 eager。
            if (
                    self.compilation_config.cudagraph_mode.has_full_cudagraphs()
                    and self.model_config is not None
                    and not self.model_config.disable_cascade_attn
                    and not self.compilation_config.cudagraph_mode.has_piecewise_cudagraphs()  # noqa: E501
            ):
                logger.warning_once(
                    "No piecewise cudagraph for executing cascade attention."
                    " Will fall back to eager execution if a batch runs "
                    "into cascade attentions."
                )

            # 使用 piecewise cudagraph 时，必须同时启用 VLLM_COMPILE。
            if self.compilation_config.cudagraph_mode.requires_piecewise_compilation():
                assert self.compilation_config.mode == CompilationMode.VLLM_COMPILE, (
                    "Compilation mode should be CompilationMode.VLLM_COMPILE "
                    "when cudagraph_mode piecewise cudagraphs is used, "
                    f"cudagraph_mode={self.compilation_config.cudagraph_mode}"
                )
        # 延迟导入 batch invariant 检测函数，避免不必要的模块开销。
        from cfie.model_executor.layers.batch_invariant import cfie_is_batch_invariant

        # 若全局开启了 batch invariant，则禁止 cascade attention。
        if (
                self.model_config
                and cfie_is_batch_invariant()
                and not self.model_config.disable_cascade_attn
        ):
            self.model_config.disable_cascade_attn = True
            logger.warning_once(
                "Disabling cascade attention when VLLM_BATCH_INVARIANT is enabled.",
                scope="local",
            )

        # ----------------- UBatching / DBO 兼容性处理 -----------------
        # --------------- UBatching / DBO 兼容性处理 ---------------
        # 若启用了 ubatching，则 all2all backend 必须落在 DeepEP 支持集合中。
        if self.parallel_config.use_ubatching:
            a2a_backend = self.parallel_config.all2all_backend
            assert a2a_backend in [
                "deepep_low_latency",
                "deepep_high_throughput",
            ], (
                "Microbatching currently only supports the deepep_low_latency and "
                f"deepep_high_throughput all2all backend. {a2a_backend} is not "
                "supported. To fix use --all2all-backend=deepep_low_latency or "
                "--all2all-backend=deepep_high_throughput and install the DeepEP"
                " kernels."
            )

            # DBO 与 cascade attention 不兼容，因此在此强制关闭。
            if not self.model_config.disable_cascade_attn:
                self.model_config.disable_cascade_attn = True
                logger.warning_once("Disabling cascade attention when DBO is enabled.")

        # 若前面由于某种原因把 instance_id 清空了，这里补一个短 UUID 兜底。
        if not self.instance_id:
            self.instance_id = random_uuid()[:5]

        # ----------------- Hybrid KV Cache Manager(HMA) 开关决议 -----------------
        # --------------- Hybrid KV Cache Manager(HMA) 开关决议 ---------------
        # Hybrid KV cache manager（HMA）运行时规则：
        # - 显式启用（--no-disable-kv-cache-manager）：若运行时要禁用则报错
        # - 无偏好：对不支持特性（如 kv connector）自动禁用
        # - 显式禁用（--disable-kv-cache-manager）：始终遵从
        # 初始化“是否需要禁用 HMA”的累积标记。
        need_disable_hybrid_kv_cache_manager = False
        # logger 仅应对 hybrid 模型打印告警。由于此处还无法确定模型是否 hybrid，
        # 因此这里不打告警，后续再记录。
        # 非支持平台时，HMA 直接不可用。
        if not current_platform.support_hybrid_kv_cache():
            # 非 GPU 平台不支持 Hybrid KV cache manager。
            need_disable_hybrid_kv_cache_manager = True
        # KV events 与 HMA 不兼容。
        if self.kv_events_config is not None:
            # Hybrid KV cache manager 与 KV events 不兼容。
            need_disable_hybrid_kv_cache_manager = True
        # 若模型开启了 chunked local attention，还需继续判断是否允许 HMA。
        if (
                self.model_config is not None
                and self.model_config.attention_chunk_size is not None
        ):
            # chunked local attention + EAGLE 目前不支持 HMA。
            if (
                    self.speculative_config is not None
                    and self.speculative_config.use_eagle()
            ):
                # Hybrid KV cache manager 暂不支持 chunked local attention + eagle。
                need_disable_hybrid_kv_cache_manager = True
            # 若环境变量未显式允许，则对 chunked local attention 默认关闭 HMA。
            elif not envs.VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE:
                logger.warning(
                    "There is a latency regression when using chunked local"
                    " attention with the hybrid KV cache manager. Disabling"
                    " it, by default. To enable it, set the environment "
                    "VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE=1."
                )
                # Hybrid KV cache manager 暂不支持 chunked local attention。
                need_disable_hybrid_kv_cache_manager = True

        # 若用户未显式指定是否禁用 HMA，则由系统自动决定。
        if self.scheduler_config.disable_hybrid_kv_cache_manager is None:
            # 默认禁用 HMA，但仅在用户未表达偏好时生效。
            if self.kv_transfer_config is not None:
                # NOTE(Kuntai)：对于 connector 默认关闭 HMA，除非显式启用。
                need_disable_hybrid_kv_cache_manager = True
                logger.warning(
                    "Turning off hybrid kv cache manager because "
                    "`--kv-transfer-config` is set. This will reduce the "
                    "performance of vLLM on LLMs with sliding window attention "
                    "or Mamba attention. If you are a developer of kv connector"
                    ", please consider supporting hybrid kv cache manager for "
                    "your connector by making sure your connector is a subclass"
                    " of `SupportsHMA` defined in kv_connector/v1/base.py and"
                    " use --no-disable-hybrid-kv-cache-manager to start vLLM."
                )
            self.scheduler_config.disable_hybrid_kv_cache_manager = (
                need_disable_hybrid_kv_cache_manager
            )
        # 若用户显式要求启用 HMA，但当前环境判定必须禁用，则直接报错。
        elif (
                self.scheduler_config.disable_hybrid_kv_cache_manager is False
                and need_disable_hybrid_kv_cache_manager
        ):
            raise ValueError(
                "Hybrid KV cache manager was explicitly enabled but is not "
                "supported in this configuration. Consider omitting the "
                "--no-disable-hybrid-kv-cache-manager flag to let vLLM decide"
                " automatically."
            )

        # 若到现在依然是 None，说明没有任何地方显式禁用，则默认启用 HMA。
        if self.scheduler_config.disable_hybrid_kv_cache_manager is None:
            # 若用户或上述逻辑未显式禁用，则默认启用 HMA。
            self.scheduler_config.disable_hybrid_kv_cache_manager = False

        # --------------- debug dump 路径标准化 ---------------
        # 若配置对象里自带 debug dump 路径，则先转成绝对路径并展开用户目录。
        if self.compilation_config.debug_dump_path:
            self.compilation_config.debug_dump_path = (
                self.compilation_config.debug_dump_path.absolute().expanduser()
            )
        # 若环境变量里显式给了 debug dump 路径，则用环境变量覆盖配置文件中的值。
        if envs.VLLM_DEBUG_DUMP_PATH is not None:
            # 先把环境变量路径规范化成绝对路径。
            env_path = Path(envs.VLLM_DEBUG_DUMP_PATH).absolute().expanduser()
            # 若配置中原本也设置过路径，则打印覆盖告警。
            if self.compilation_config.debug_dump_path:
                logger.warning(
                    "Config-specified debug dump path is overridden"
                    " by VLLM_DEBUG_DUMP_PATH to %s",
                    env_path,
                )
            # 用环境变量中的路径覆盖最终值。
            self.compilation_config.debug_dump_path = env_path

        # --------------- 收尾：再次处理 blocked weights / KV connector / pass 日志 ---------------
        # 这里再次定义同名辅助函数，是为了复用当前作用域内的量化配置状态。
        # 保留现有执行顺序，不改变函数逻辑。
        def has_blocked_weights():
            # 判断是否启用了按 block 存储的权重。
            if self.quant_config is not None:
                # 新式接口：直接检查 weight_block_size 字段。
                if hasattr(self.quant_config, "weight_block_size"):
                    return self.quant_config.weight_block_size is not None
                # 旧式接口：通过 has_blocked_weights() 方法判断。
                elif hasattr(self.quant_config, "has_blocked_weights"):
                    return self.quant_config.has_blocked_weights()
            return False

        # 启用 quant_fp8 CUDA 算子（TODO：后续再考虑关闭）
        # 在 H100 上 CUDA 内核比
        # 原生实现更快
        # https://github.com/cfie-project/cfie/issues/25094
        # 若检测到 blocked weights，则确保 quant_fp8 custom op 已经开启。
        if has_blocked_weights():
            custom_ops = self.compilation_config.custom_ops
            if "-quant_fp8" not in custom_ops:
                custom_ops.append("+quant_fp8")

        # 处理 KV connector 相关配置
        # 这一阶段会根据 connector 类型与角色，补齐其额外配置字段。
        self._post_init_kv_transfer_config()

        # 记录已启用的自定义 pass
        # 便于在启动日志中快速看到本次真正生效的 pass 集合。
        self.compilation_config.pass_config.log_enabled_passes()

    def update_sizes_for_sequence_parallelism(self, possible_sizes: list) -> list:
        """按序列并行约束过滤 batch size，仅保留可被 TP 大小整除的值。"""
        # ----------------- 先找出所有不满足 TP 整除约束的候选值 -----------------
        # 启用序列并行时，移除不能被 tp_size 整除的 size
        removed_sizes = [
            # 遍历每个候选 batch size。
            size
            # 候选列表来自上游预设的 cudagraph capture sizes。
            for size in possible_sizes
            # 不能被 TP 大小整除的 size 在 SP 下无效。
            if size % self.parallel_config.tensor_parallel_size != 0
        ]
        # 若确实移除了某些值，则打印一条告警帮助排查性能差异。
        if removed_sizes:
            logger.warning(
                "Batch sizes %s are removed because they are not "
                "multiple of tp_size %d when "
                "sequence parallelism is enabled",
                removed_sizes,
                self.parallel_config.tensor_parallel_size,
            )

        # 返回过滤后的合法 size 列表。
        return [
            # 再次遍历候选列表。
            size
            # 保留所有满足 TP 整除约束的 size。
            for size in possible_sizes
            # 这些 size 才能用于序列并行场景下的编译/捕获。
            if size % self.parallel_config.tensor_parallel_size == 0
        ]

    def _set_max_num_scheduled_tokens(self):
        """
        在大多数情况下，调度器可调度的 batch token 数量可达到 worker 配置上限。
        但对某些投机解码方法，drafter 模型在草拟时会向 batch 插入额外槽位。
        因此需要把 max_num_scheduled_tokens 下调一个上界，
        以覆盖可能新增的槽位数量。
        """
        # ----------------- 针对 speculative decoding 预留 drafting 槽位 -----------------
        # 仅在启用 speculative decoding 时才需要调整该上限。
        if self.speculative_config is not None:
            # 额外槽位数 = 每条序列可能新增的 drafting slots * 最大序列数。
            scheduled_token_delta = (
                    self.speculative_config.max_num_new_slots_for_drafting
                    * self.scheduler_config.max_num_seqs
            )
            # 读取调度器允许的总 batch token 上限。
            max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
            # 若用户未显式给出 max_num_scheduled_tokens，则自动留出 drafting 余量。
            if self.scheduler_config.max_num_scheduled_tokens is None:
                self.scheduler_config.max_num_scheduled_tokens = (
                        max_num_batched_tokens - scheduled_token_delta
                )

            # 读取最终生效的 max_num_scheduled_tokens。
            max_num_scheduled_tokens = self.scheduler_config.max_num_scheduled_tokens
            # 若总槽位不够容纳“已调度 token + drafting 预留”，则直接报错。
            if max_num_batched_tokens < max_num_scheduled_tokens + (
                    self.speculative_config.max_num_new_slots_for_drafting
                    * self.scheduler_config.max_num_seqs
            ):
                raise ValueError(
                    f"CfieConfig received max_num_scheduled_tokens but it does not have"
                    " enough slots to support the speculative decoding settings."
                    f" It should be greater by at least {scheduled_token_delta}, but"
                    f" got {max_num_batched_tokens=} and {max_num_scheduled_tokens=}."
                )

    def _set_cudagraph_sizes(self):
        """
        vLLM 默认将 CUDA graph 捕获的 batch size 候选列表定义为：

        ```python
        max_graph_size = min(max_num_seqs * 2, 512)
        # 1、2、4，然后是到 256（不含）前步长为 8 的倍数，
        # 再然后是到 max_graph_size 的步长为 16 的倍数
        cudagraph_capture_sizes = [1, 2, 4] + list(range(8, 256, 8)) + list(
            range(256, max_graph_size + 1, 16))

        最终 `cfie_config.compilation_config.cudagraph_capture_sizes`
        会成为实际用于捕获 cudagraph 的最终 size（升序）。

        这些 size 用于在性能关键路径（如解码）捕获并复用 CUDA graph。
        捕获后可绕开 Python 开销，从而显著加快 kernel 调度。随后该列表会依据
        `max_num_batched_tokens`（例如多数 GPU 上是 8192）过滤，该参数控制
        一个 batch 允许的总 token 数。由于每条序列 token 数可变，
        实际可用的最大 batch size 取决于具体序列长度。

        示例：
            当 `max_num_batched_tokens = 8192`，且典型序列平均约 32 token 时，
            大多数可用 batch size 会小于 256。
            但若 shape 与显存允许，系统仍可支持最高到 512 的捕获 size。

        注意：
            若用户在 compilation config 中显式指定了 cudagraph_capture_sizes，
            会覆盖这里的默认逻辑。
            在运行时：

            - 若 batch size <= 某个 `cudagraph_capture_sizes` 值，
            则使用最接近的填充后 CUDA graph。
            - 若 batch size > 最大的 `cudagraph_capture_sizes`，
            则不使用 cudagraph。
        """

        # ----------------- 仅在真正使用 cudagraph 时才计算捕获 size -----------------
        if (
                self.model_config is not None
                and not self.model_config.enforce_eager
                and self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        ):
            # 确定初始的 max_cudagraph_capture_size
            max_cudagraph_capture_size = (
                self.compilation_config.max_cudagraph_capture_size
            )
            # 若用户没有显式给出上限，则按 max_num_seqs 和 speculative query 长度推导。
            if max_cudagraph_capture_size is None:
                # 普通 decode query_len 默认为 1。
                decode_query_len = 1
                # speculative decoding 会把一次 decode 的 query_len 拉长。
                if (
                        self.speculative_config
                        and self.speculative_config.num_speculative_tokens
                ):
                    decode_query_len += self.speculative_config.num_speculative_tokens
                # 默认上限仍保守截断在 512。
                max_cudagraph_capture_size = min(
                    self.scheduler_config.max_num_seqs * decode_query_len * 2, 512
                )
            # 再用 max_num_batched_tokens 对捕获上限做最终裁剪。
            max_num_tokens = self.scheduler_config.max_num_batched_tokens
            max_cudagraph_capture_size = min(max_num_tokens, max_cudagraph_capture_size)

            # 使用 cudagraph 时，最终 capture size 上限至少要 >= 1。
            assert max_cudagraph_capture_size >= 1, (
                "Maximum cudagraph size should be greater than or equal to 1 "
                "when using cuda graph."
            )

            # ----------------- 生成最终的 cudagraph_capture_sizes 列表 -----------------
            # 确定 cudagraph_capture_sizes
            if self.compilation_config.cudagraph_capture_sizes is not None:
                # 用户手工指定列表时，要求列表非空。
                assert len(self.compilation_config.cudagraph_capture_sizes) > 0, (
                    "cudagraph_capture_sizes should contain at least one element "
                    "when using cuda graph."
                )
                # 对配置中给出的 size 去重
                dedup_sizes = list(set(self.compilation_config.cudagraph_capture_sizes))
                # 过滤掉超过 max_num_tokens 的非法 size。
                cudagraph_capture_sizes = [
                    i for i in dedup_sizes if i <= max_num_tokens
                ]
                # 排序，确保 size 按升序
                cudagraph_capture_sizes.sort()
            else:
                # interactivity 模式优先生成更细粒度的小 batch 捕获列表。
                if self.performance_mode == "interactivity":
                    # 小 batch 下使用细粒度 CUDA graph，
                    # 以最小化 padding 开销
                    interactivity_max = min(max_cudagraph_capture_size, 32)
                    cudagraph_capture_sizes = list(range(1, interactivity_max + 1))
                else:
                    # balanced/throughput 模式先保留几个最常见的小 batch 点。
                    cudagraph_capture_sizes = [
                        i for i in [1, 2, 4] if i <= max_cudagraph_capture_size
                    ]
                if max_cudagraph_capture_size >= 8:
                    # 小 batch 区间使用步长 8，直到 256（不含）
                    cudagraph_capture_sizes += list(
                        range(8, min(max_cudagraph_capture_size + 1, 256), 8)
                    )
                if max_cudagraph_capture_size >= 256:
                    # 大 batch 区间使用步长 16
                    cudagraph_capture_sizes += list(
                        range(256, max_cudagraph_capture_size + 1, 16)
                    )
                # 去重并排序
                cudagraph_capture_sizes = sorted(set(cudagraph_capture_sizes))

            # 若开启了序列并行，则继续删掉无法被 TP 整除的 capture size。
            if (
                    self.parallel_config.tensor_parallel_size > 1
                    and self.compilation_config.pass_config.enable_sp
            ):
                cudagraph_capture_sizes = self.update_sizes_for_sequence_parallelism(
                    cudagraph_capture_sizes
                )

            # 当用户指定的 compilation_config.max_cudagraph_capture_size
            # 与实际可用值不一致时，会被截断到 valid_max_size。
            valid_max_size = (
                cudagraph_capture_sizes[-1] if cudagraph_capture_sizes else 0
            )
            # 若用户显式给出的 max 值与最终列表最大值不一致，则要么报错要么截断。
            if (
                    self.compilation_config.max_cudagraph_capture_size is not None
                    and self.compilation_config.max_cudagraph_capture_size != valid_max_size
            ):
                # 仅当两个参数都由用户显式指定且互相不一致时抛错
                if self.compilation_config.cudagraph_capture_sizes is not None:
                    raise ValueError(
                        "customized max_cudagraph_capture_size"
                        f"(={self.compilation_config.max_cudagraph_capture_size}) "
                        "should be consistent with the max value of "
                        f"cudagraph_capture_sizes(={valid_max_size})"
                    )

                logger.warning(
                    "Truncating max_cudagraph_capture_size to %d",
                    valid_max_size,
                )
            # 始终写回最终的 max_cudagraph_capture_size
            self.compilation_config.max_cudagraph_capture_size = valid_max_size

            # 若用户显式指定过 capture sizes，但经过过滤后变短，则打印覆盖告警。
            if self.compilation_config.cudagraph_capture_sizes is not None and len(
                    cudagraph_capture_sizes
            ) < len(self.compilation_config.cudagraph_capture_sizes):
                # 如果用户指定了 capture sizes，只需比较修改前后长度；
                # 因为修改后的列表仅可能是原列表的子集。
                logger.warning(
                    (
                        "cudagraph_capture_sizes specified in compilation_config"
                        " %s is overridden by config %s"
                    ),
                    self.compilation_config.cudagraph_capture_sizes,
                    cudagraph_capture_sizes,
                )
            # 始终写回最终 size 列表
            self.compilation_config.cudagraph_capture_sizes = cudagraph_capture_sizes

        else:
            # 当前不使用 cudagraph
            self.compilation_config.max_cudagraph_capture_size = 0
            self.compilation_config.cudagraph_capture_sizes = []

        # 完成剩余流程
        self.compilation_config.post_init_cudagraph_sizes()

    def _set_compile_ranges(self):
        """
        为 compilation config 设置编译区间。
        """
        # ----------------- 汇总所有会改变图优化策略的 compile range 端点 -----------------
        # 取局部别名，便于后续多次访问 compilation_config。
        compilation_config = self.compilation_config
        # 用列表收集所有候选端点，最后再统一排序。
        computed_compile_ranges_endpoints = []

        # 编译区间上界是 max_num_batched_tokens。
        compile_range_end = self.scheduler_config.max_num_batched_tokens
        # max_num_batched_tokens 若存在，一定是最外层的终止端点。
        if compile_range_end is not None:
            computed_compile_ranges_endpoints.append(compile_range_end)

        # ----------------- 按 allreduce-rms 融合阈值补充端点 -----------------
        # 添加 flashinfer 的编译区间
        if compilation_config.pass_config.fuse_allreduce_rms:
            # TP 大小会影响 flashinfer 融合的阈值。
            tp_size = self.parallel_config.tensor_parallel_size
            # 读取 flashinfer allreduce-rms 对应的最大启用 size。
            max_size = compilation_config.pass_config.flashinfer_max_size(tp_size)
            if max_size is not None:
                # 把字节阈值换算成 token 数阈值。
                max_token_num = max_size // (
                        self.model_config.get_hidden_size()
                        * self.model_config.dtype.itemsize
                )
                # 只有阈值落在编译上界内时才需要新增一个分段端点。
                if compile_range_end is not None and max_token_num < compile_range_end:
                    computed_compile_ranges_endpoints.append(max_token_num)
                else:
                    logger.debug(
                        "Max num batched tokens below allreduce-rms fusion threshold, "
                        "allreduce-rms fusion will be enabled for all num_tokens."
                    )

        # ----------------- 按序列并行阈值补充端点 -----------------
        # 添加序列并行的编译区间
        if compilation_config.pass_config.enable_sp:
            # 取局部别名，避免多次深层属性访问。
            pass_config = compilation_config.pass_config

            # 若未显式提供，则计算 min_token_num
            # 用户覆盖值始终生效，不受 hidden_size 影响
            if pass_config.sp_min_token_num is None:
                from cfie.compilation.passes.fusion.sequence_parallelism import (
                    get_sequence_parallelism_threshold,
                )

                tp_size = self.parallel_config.tensor_parallel_size
                hidden_size = self.model_config.get_hidden_size()
                element_size = self.model_config.dtype.itemsize
                # 根据模型隐藏维、TP 和 dtype 估算 SP 何时值得打开。
                pass_config.sp_min_token_num = get_sequence_parallelism_threshold(
                    hidden_size, tp_size, element_size
                )

            # 读取最终 SP 启用阈值。
            min_token_num = pass_config.sp_min_token_num
            # 读取 batch token 总上限。
            max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
            if min_token_num is not None and (
                    max_num_batched_tokens is not None
                    and min_token_num < max_num_batched_tokens
                    and min_token_num > 1
            ):
                # 在 min_token_num - 1 处添加端点，确保 SP 从 min_token_num 开始生效
                # 由此形成区间：[1, min-1]（无 SP），[min, max]（启用 SP）
                computed_compile_ranges_endpoints.append(min_token_num - 1)

        # ----------------- 按 rope+kvcache 融合阈值补充端点 -----------------
        if compilation_config.pass_config.fuse_rope_kvcache:
            # 读取 rope+kvcache 融合可用的最大 token 数阈值。
            max_token_num = (
                compilation_config.pass_config.rope_kvcache_fusion_max_token_num
            )
            if max_token_num is not None:
                # 阈值落在总上界内时才追加一段新的编译范围。
                if compile_range_end is not None and max_token_num < compile_range_end:
                    computed_compile_ranges_endpoints.append(max_token_num)
                else:
                    logger.debug(
                        "Max num batched tokens below rope+kvcache fusion threshold, "
                        "rope+kvcache fusion enabled for num_tokens <= %d.",
                        compile_range_end,
                    )

        # ----------------- 合并用户自定义的 compile_ranges_endpoints -----------------
        if compilation_config.compile_ranges_endpoints is not None:
            # 用户提供的每个端点都要先经过合法性检查。
            for x in compilation_config.compile_ranges_endpoints:
                # 端点必须是正整数。
                assert isinstance(x, int)
                assert x > 0, f"Invalid compile range endpoint: {x}"
                # 只保留落在总上界内部且有分段意义的端点。
                if compile_range_end is not None and x < compile_range_end and x > 1:
                    computed_compile_ranges_endpoints.append(x)
        # 最终将所有端点排序后写回 compilation_config。
        compilation_config.compile_ranges_endpoints = sorted(
            computed_compile_ranges_endpoints
        )

    def try_verify_and_update_config(self):
        """按模型架构尝试校验并更新配置，仅执行一次。"""
        # ----------------- 无 model_config 时直接跳过 -----------------
        if self.model_config is None:
            return

        # 避免重复执行 try_verify_and_update_config
        if getattr(self.model_config, "config_updated", False):
            return
        # 标记本轮已经执行过 verify/update。
        self.model_config.config_updated = True

        # 读取当前模型架构名。
        architecture = self.model_config.architecture
        if architecture is None:
            return

        from cfie.model_executor.models.config import (
            MODELS_CONFIG_MAP,
            HybridAttentionMambaModelConfig,
        )

        # 先按 architecture 找到专用的配置修正类。
        cls = MODELS_CONFIG_MAP.get(architecture, None)
        if cls is not None:
            # 让具体模型类按自身规则修正 CfieConfig。
            cls.verify_and_update_config(self)

        # hybrid 模型还需要额外走一轮 HybridAttentionMambaModelConfig 校验。
        if self.model_config.is_hybrid:
            HybridAttentionMambaModelConfig.verify_and_update_config(self)

        # classify 转换路径需要额外把 CausalLM 配置修正为分类模型配置。
        if self.model_config.convert_type == "classify":
            # 可能将 ForCausalLM 转换为 ForSequenceClassification 模型。
            from cfie.model_executor.models.adapters import SequenceClassificationConfig

            SequenceClassificationConfig.verify_and_update_config(self)

        # 若检测到 Run:ai 对象存储 URI，则自动调整 load_format。
        if hasattr(self.model_config, "model_weights") and is_runai_obj_uri(
                self.model_config.model_weights
        ):
            if self.load_config.load_format == "auto":
                logger.info(
                    "Detected Run:ai model config. "
                    "Overriding `load_format` to 'runai_streamer'"
                )
                self.load_config.load_format = "runai_streamer"
            elif self.load_config.load_format not in (
                    "runai_streamer",
                    "runai_streamer_sharded",
            ):
                raise ValueError(
                    f"To load a model from S3, 'load_format' "
                    f"must be 'runai_streamer' or 'runai_streamer_sharded', "
                    f"but got '{self.load_config.load_format}'. "
                    f"Model: {self.model_config.model}"
                )

    def compile_debug_dump_path(self) -> Path | None:
        """返回带 rank 信息的路径，
        用于导出 torch.compile 调试信息。
        """
        # 未启用 debug dump 时返回 None。
        if self.compilation_config.debug_dump_path is None:
            return None
        # 读取当前 TP rank。
        tp_rank = self.parallel_config.rank
        # 读取当前 DP rank。
        dp_rank = self.parallel_config.data_parallel_index
        # 为每个 rank 构造独立子目录。
        append_path = f"rank_{tp_rank}_dp_{dp_rank}"
        # 把 rank 信息拼到基础 debug_dump_path 后面。
        path = self.compilation_config.debug_dump_path / append_path
        # 返回最终导出目录。
        return path

    def __str__(self):
        """返回用于日志展示的关键配置摘要字符串。"""
        # 拼出一条单行摘要字符串，方便在日志中快速查看主配置。
        return (
            f"model={self.model_config.model!r}, "
            f"speculative_config={self.speculative_config!r}, "
            f"tokenizer={self.model_config.tokenizer!r}, "
            f"skip_tokenizer_init={self.model_config.skip_tokenizer_init}, "
            f"tokenizer_mode={self.model_config.tokenizer_mode}, "
            f"revision={self.model_config.revision}, "
            f"tokenizer_revision={self.model_config.tokenizer_revision}, "
            f"trust_remote_code={self.model_config.trust_remote_code}, "
            f"dtype={self.model_config.dtype}, "
            f"max_seq_len={self.model_config.max_model_len}, "
            f"download_dir={self.load_config.download_dir!r}, "
            f"load_format={self.load_config.load_format}, "
            f"tensor_parallel_size={self.parallel_config.tensor_parallel_size}, "  # noqa
            f"pipeline_parallel_size={self.parallel_config.pipeline_parallel_size}, "  # noqa
            f"data_parallel_size={self.parallel_config.data_parallel_size}, "  # noqa
            f"decode_context_parallel_size={self.parallel_config.decode_context_parallel_size}, "  # noqa
            f"dcp_comm_backend={self.parallel_config.dcp_comm_backend}, "  # noqa
            f"disable_custom_all_reduce={self.parallel_config.disable_custom_all_reduce}, "  # noqa
            f"quantization={self.model_config.quantization}, "
            f"enforce_eager={self.model_config.enforce_eager}, "
            f"enable_return_routed_experts={self.model_config.enable_return_routed_experts}, "  # noqa
            f"kv_cache_dtype={self.cache_config.cache_dtype}, "
            f"device_config={self.device_config.device}, "
            f"structured_outputs_config={self.structured_outputs_config!r}, "
            f"observability_config={self.observability_config!r}, "
            f"seed={self.model_config.seed}, "
            f"served_model_name={self.model_config.served_model_name}, "
            f"enable_prefix_caching={self.cache_config.enable_prefix_caching}, "
            f"enable_chunked_prefill={self.scheduler_config.enable_chunked_prefill}, "  # noqa
            f"pooler_config={self.model_config.pooler_config!r}, "
            f"compilation_config={self.compilation_config!r}"
        )

    def validate_block_size(self) -> None:
        """Validate final block size against DCP and Mamba scheduling limits."""
        block_size = self.cache_config.block_size

        if self.parallel_config.decode_context_parallel_size > 1:
            if self.parallel_config.dcp_kv_cache_interleave_size > 1 and (
                self.parallel_config.cp_kv_cache_interleave_size
                != self.parallel_config.dcp_kv_cache_interleave_size
            ):
                self.parallel_config.cp_kv_cache_interleave_size = (
                    self.parallel_config.dcp_kv_cache_interleave_size
                )
                logger.warning_once(
                    "cp_kv_cache_interleave_size is overridden by "
                    "dcp_kv_cache_interleave_size. And "
                    "dcp-kv-cache-interleave-size will be deprecated when "
                    "PCP is fully supported."
                )
            assert (
                self.parallel_config.cp_kv_cache_interleave_size <= block_size
                and block_size
                % self.parallel_config.cp_kv_cache_interleave_size
                == 0
            ), (
                f"Block_size({block_size}) should be greater than or equal to "
                "and divisible by cp_kv_cache_interleave_size "
                f"({self.parallel_config.cp_kv_cache_interleave_size})."
            )

        if self.cache_config.mamba_cache_mode == "align":
            if block_size > self.scheduler_config.max_num_batched_tokens:
                logger.warning_once(
                    "Increasing max_num_batched_tokens from %d to %d because "
                    "Mamba cache align mode requires the scheduler budget to "
                    "cover at least one full block.",
                    self.scheduler_config.max_num_batched_tokens,
                    block_size,
                )
                self.scheduler_config.max_num_batched_tokens = block_size
            if self.scheduler_config.long_prefill_token_threshold > 0:
                assert self.scheduler_config.long_prefill_token_threshold >= block_size
            assert not self.scheduler_config.disable_chunked_mm_input, (
                "Chunked MM input is required because we need the flexibility "
                "to schedule a multiple of block_size tokens even if they are "
                "in the middle of a mm input"
            )

    @model_validator(mode="after")
    def validate_mamba_block_size(self) -> "CfieConfig":
        """校验 mamba_block_size 与前缀缓存设置的组合是否合法。"""
        # 没有 model_config 时无需校验，直接返回当前对象。
        if self.model_config is None:
            return self
        # 仅当用户显式设置了 mamba_block_size 且它不同于 max_model_len 时才算“启用”。
        mamba_block_size_is_set = (
                self.cache_config.mamba_block_size is not None
                and self.cache_config.mamba_block_size != self.model_config.max_model_len
        )
        # 自定义 mamba_block_size 依赖 prefix caching，因此两者必须一起开启。
        if mamba_block_size_is_set and not self.cache_config.enable_prefix_caching:
            raise ValueError(
                "--mamba-block-size can only be set with --enable-prefix-caching"
            )
        # 通过校验后返回当前对象，符合 pydantic model_validator 约定。
        return self


_current_cfie_config: CfieConfig | None = None
_current_prefix: str | None = None


@contextmanager
def set_current_cfie_config(
        cfie_config: CfieConfig, check_compile=False, prefix: str | None = None
):
    """
    临时设置当前 vLLM 配置。
    用于模型初始化期间。
    我们将当前 vLLM 配置保存到全局变量中，
    以便所有模块都能访问，例如 custom op
    可据此决定分发逻辑。
    """
    # 需要修改模块级全局状态，因此显式声明 global。
    global _current_cfie_config, _current_prefix
    # 先保存旧的配置上下文，方便 finally 恢复。
    old_cfie_config = _current_cfie_config
    # 同时保存旧的层名前缀上下文。
    old_prefix = _current_prefix
    # 延迟导入 compilation_counter，避免常规路径的额外开销。
    from cfie.compilation.counter import compilation_counter

    # 记录进入上下文前，已经注册过多少个可编译模型。
    num_models_seen = compilation_counter.num_models_seen
    try:
        # 上下文变化时清理 compilation config 缓存。
        # 因为旧配置可能在新配置设置前已被访问并缓存。
        get_cached_compilation_config.cache_clear()

        # 写入当前上下文的 CfieConfig。
        _current_cfie_config = cfie_config
        # 写入当前层名前缀。
        _current_prefix = prefix
        # 把控制权交给 with 语句块内部代码。
        yield
    except Exception:
        # 异常路径不做吞掉处理，原样向上抛出。
        raise
    else:
        # 若要求检查 compile/custom op 日志，则在正常退出后执行。
        if check_compile:
            cfie_config.compilation_config.custom_op_log_check()

        # 若启用了 compile 但模型计数未增加，说明模型未真正支持 torch.compile。
        if (
                check_compile
                and cfie_config.compilation_config.mode == CompilationMode.VLLM_COMPILE
                and compilation_counter.num_models_seen == num_models_seen
        ):
            # 若模型支持编译，
            # compilation_counter.num_models_seen 至少应增加 1。
            # 若未增加，说明模型不支持编译
            # （没有 @support_torch_compile 装饰器）。
            logger.warning(
                "`torch.compile` is turned on, but the model %s"
                " does not support it. Please open an issue on GitHub"
                " if you want it to be supported.",
                cfie_config.model_config.model,
            )
    finally:
        # 恢复进入上下文前的 CfieConfig。
        _current_cfie_config = old_cfie_config
        # 恢复进入上下文前的前缀。
        _current_prefix = old_prefix
        # 上下文变化时清理 compilation config 缓存
        get_cached_compilation_config.cache_clear()


@lru_cache(maxsize=1)
def get_cached_compilation_config():
    """缓存配置，避免重复调用 get_current_cfie_config()。"""
    # 当前上下文不变时，直接复用同一份 compilation_config。
    return get_current_cfie_config().compilation_config


def get_current_cfie_config() -> CfieConfig:
    """获取当前上下文中的 vLLM 配置；若未设置则抛出异常。"""
    # 未进入 set_current_cfie_config 上下文时，这里视为编程错误。
    if _current_cfie_config is None:
        raise AssertionError(
            "Current vLLM config is not set. This typically means "
            "get_current_cfie_config() was called outside of a "
            "set_current_cfie_config() context, or a CustomOp was instantiated "
            "at module import time or model forward time when config is not set. "
            "For tests that directly test custom ops/modules, use the "
            "'default_cfie_config' pytest fixture from tests/conftest.py."
        )
    # 返回当前线程共享的全局 CfieConfig。
    return _current_cfie_config


def get_current_cfie_config_or_none() -> CfieConfig | None:
    """获取当前 vLLM 配置；若不存在则返回 None。"""
    # 允许调用方显式处理“当前没有配置上下文”的情况。
    return _current_cfie_config


T = TypeVar("T")


def get_layers_from_cfie_config(
        cfie_config: CfieConfig,
        layer_type: type[T],
        layer_names: list[str] | None = None,
) -> dict[str, T]:
    """
    从 vLLM 配置中获取层对象。

    参数：
        cfie_config：vLLM 配置对象。
        layer_type：要获取的层类型。
        layer_names：要获取的层名列表。若为 None，则返回所有层。
    """

    # 未指定 layer_names 时，默认遍历 static_forward_context 中的所有层名。
    if layer_names is None:
        layer_names = list(cfie_config.compilation_config.static_forward_context.keys())

    # 取出静态 forward 上下文，里面缓存了层名到层对象的映射。
    forward_context = cfie_config.compilation_config.static_forward_context

    # 返回“名字存在且类型匹配”的层对象字典。
    return {
        # 用层名作为返回字典的 key。
        layer_name: forward_context[layer_name]
        # 遍历调用方指定或默认展开得到的层名列表。
        for layer_name in layer_names
        # 同时要求该层已存在于上下文，且实例类型满足 layer_type。
        if layer_name in forward_context
           and isinstance(forward_context[layer_name], layer_type)
    }
