# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
from collections import Counter
from collections.abc import Callable
from dataclasses import field, fields
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import Field, TypeAdapter, field_validator

import cfie.envs as envs
from cfie.compilation.passes.inductor_pass import CallableInductorPass, InductorPass
from cfie.config.utils import (
    Range,
    config,
    get_hash_factors,
    hash_factors,
)
from cfie.logger import init_logger
from cfie.platforms import current_platform
from cfie.utils.import_utils import resolve_obj_by_qualname
from cfie.utils.math_utils import round_up
from cfie.utils.torch_utils import is_torch_equal_or_newer

if TYPE_CHECKING:
    from cfie.config import CfieConfig
else:
    CfieConfig = object

logger = init_logger(__name__)


def _has_triton_runtime() -> bool:
    return find_spec("triton") is not None


class CompilationMode(enum.IntEnum):
    """The compilation approach used for torch.compile-based compilation of the
    model."""

    NONE = 0
    """No torch.compile compilation is applied, model runs in fully eager pytorch mode.
    The model runs as-is."""
    STOCK_TORCH_COMPILE = 1
    """The standard `torch.compile` compilation pipeline."""
    DYNAMO_TRACE_ONCE = 2
    """Single Dynamo trace through the model, avoiding recompilation."""
    VLLM_COMPILE = 3
    """Custom vLLM Inductor-based backend with caching, piecewise compilation,
    shape specialization, and custom passes."""


class CUDAGraphMode(enum.Enum):
    """Constants for the cudagraph mode in CompilationConfig.
    Meanwhile, the subset enum `NONE`, `PIECEWISE` and `FULL` are also
    treated as concrete runtime mode for cudagraph runtime dispatching.
    """

    NONE = 0
    PIECEWISE = 1
    FULL = 2
    FULL_DECODE_ONLY = (FULL, NONE)
    FULL_AND_PIECEWISE = (FULL, PIECEWISE)

    def decode_mode(self) -> "CUDAGraphMode":
        return CUDAGraphMode(self.value[0]) if self.separate_routine() else self

    def mixed_mode(self) -> "CUDAGraphMode":
        return CUDAGraphMode(self.value[1]) if self.separate_routine() else self

    def has_mode(self, mode: "CUDAGraphMode") -> bool:
        assert not mode.separate_routine()
        if self.separate_routine():
            return mode.value in self.value
        return self == mode

    def requires_piecewise_compilation(self) -> bool:
        return self.has_mode(CUDAGraphMode.PIECEWISE)

    def max_cudagraph_mode(self) -> "CUDAGraphMode":
        return CUDAGraphMode(max(self.value)) if self.separate_routine() else self

    def has_full_cudagraphs(self) -> bool:
        return self.max_cudagraph_mode() == CUDAGraphMode.FULL

    def has_piecewise_cudagraphs(self) -> bool:
        return self.requires_piecewise_compilation()

    def separate_routine(self) -> bool:
        return isinstance(self.value, tuple)

    @classmethod
    def valid_runtime_modes(cls) -> frozenset["CUDAGraphMode"]:
        return frozenset({cls.NONE, cls.PIECEWISE, cls.FULL})

    def is_valid_runtime_mode(self) -> bool:
        return self in CUDAGraphMode.valid_runtime_modes()

    def __str__(self) -> str:
        return self.name

    def __bool__(self) -> bool:
        return self != CUDAGraphMode.NONE


@config
class PassConfig:
    """Configuration for custom Inductor passes.

    This is separate from general `CompilationConfig` so that inductor passes
    don't all have access to full configuration - that would create a cycle as
    the `PassManager` is set as a property of config.

    You must pass PassConfig to VLLMConfig constructor via the CompilationConfig
    constructor. VLLMConfig's post_init does further initialization.
    If used outside of the VLLMConfig, some fields may be left in an
    improper state.
    """

    # New flags
    fuse_norm_quant: bool = Field(default=None)
    """Fuse the custom RMSNorm + quant ops."""
    fuse_act_quant: bool = Field(default=None)
    """Fuse the custom SiluMul + quant ops."""
    fuse_attn_quant: bool = Field(default=None)
    """Fuse the custom attention + quant ops."""
    eliminate_noops: bool = Field(default=True)
    """Eliminate no-op ops."""
    enable_sp: bool = Field(default=None)
    """Enable sequence parallelism. Requires TP>1. Automatically disabled
    if the model's hidden_size is too small for SP to be beneficial
    (threshold is device-capability dependent)."""
    fuse_gemm_comms: bool = Field(default=None)
    """Enable async TP."""
    fuse_allreduce_rms: bool = Field(default=None)
    """Enable flashinfer allreduce fusion."""
    enable_qk_norm_rope_fusion: bool = False
    """Enable fused Q/K RMSNorm + RoPE pass."""

    # ROCm/AITER specific fusions
    fuse_act_padding: bool = Field(default=None)
    """Fuse the custom RMSNorm + padding ops."""
    fuse_rope_kvcache: bool = Field(default=None)
    """Fuse the QK rope + KV cache ops."""

    rope_kvcache_fusion_max_token_num: int = 256
    """The threshold for ROCm AITER RoPE+KVCache fusion e.g. for small batch decode.
    Larger batch sizes e.g. during prefill will use the unfused kernels.
    """

    fi_allreduce_fusion_max_size_mb: float | None = None
    """The threshold of the communicated tensor sizes under which
    cfie should use flashinfer fused allreduce. Specified as a
    float in MB.
    Unspecified will fallback to default values
    which are compute capability and world size dependent.
        FI_ALLREDUCE_FUSION_MAX_SIZE_MB = {
            90: {
                2: 64,  # 64MB
                4: 2,  # 2MB
                8: 1,  # 1MB
            },
            100: {
                2: 64,  # 64MB
                4: 32,  # 32MB
                8: 1,  # 1MB
            },
        }, where key is the device capability"""
    sp_min_token_num: int | None = None
    """The minimum number of tokens above which cfie should use
    sequence parallelism. Specified as an integer token count.
    Unspecified will fallback to default values which are compute
    capability and world size dependent."""

    # TODO(luka) better pass enabling system.

    def flashinfer_max_size(self, world_size: int) -> int | None:
        """
        Returns the max communication size in bytes for flashinfer
        allreduce fusion for the given world size. Returns None if world size
        is not supported by configs as it's not supported by flashinfer.
        """

        MiB = 1024 * 1024
        FI_SUPPORTED_WORLD_SIZES = [2, 4, 8]
        if world_size not in FI_SUPPORTED_WORLD_SIZES:
            return None
        max_size_mb = self.fi_allreduce_fusion_max_size_mb
        if max_size_mb is None:
            max_size_mb = self.default_fi_allreduce_fusion_max_size_mb().get(world_size)

        return int(max_size_mb * MiB) if max_size_mb is not None else None

    @staticmethod
    def default_fi_allreduce_fusion_max_size_mb() -> dict[int, float]:
        from cfie.compilation.passes.fusion.allreduce_rms_fusion import (
            FI_ALLREDUCE_FUSION_MAX_SIZE_MB,
        )
        from cfie.platforms import current_platform

        if not current_platform.is_cuda():
            return {}
        return FI_ALLREDUCE_FUSION_MAX_SIZE_MB.get(
            current_platform.get_device_capability().to_int(), {}
        )

    def compute_hash(self) -> str:
        """
        Produces a hash unique to the pass configuration.
        Any new fields that affect compilation should be added to the hash.
        Any future fields that don't affect compilation should be excluded.
        """

        return hash_factors(get_hash_factors(self, set()))

    @field_validator(
        "fuse_norm_quant",
        "fuse_act_quant",
        "fuse_attn_quant",
        "enable_sp",
        "fuse_gemm_comms",
        "fuse_allreduce_rms",
        "fuse_act_padding",
        "fuse_rope_kvcache",
        mode="wrap",
    )
    @classmethod
    def _skip_none_validation(cls, value: Any, handler: Callable) -> Any:
        """Skip validation if the value is `None` when initialisation is delayed."""
        if value is None:
            return value
        return handler(value)

    def __post_init__(self) -> None:
        # Handle deprecation and defaults

        if not self.eliminate_noops:
            if self.fuse_norm_quant or self.fuse_act_quant:
                logger.warning_once(
                    "Fusion enabled but reshape elimination disabled. "
                    "RMSNorm/SiluMul + quant (fp8) fusion might not work"
                )
            if self.fuse_attn_quant:
                logger.warning_once(
                    "Fusion enabled but reshape elimination disabled. "
                    "Attention + quant (fp8) fusion might not work"
                )
            if self.fuse_allreduce_rms:
                logger.warning_once(
                    "Fusion enabled but reshape elimination disabled. "
                    "Allreduce + rms norm + quant (fp8) fusion might not work"
                )
            if self.fuse_act_padding:
                logger.warning_once(
                    "Fusion enabled but reshape elimination disabled. "
                    "RMSNorm + padding fusion might not work"
                )
        if self.enable_qk_norm_rope_fusion and not current_platform.is_cuda_alike():
            logger.warning_once(
                "QK Norm + RoPE fusion enabled but the current platform is not "
                "CUDA or ROCm. The fusion will be disabled."
            )
            self.enable_qk_norm_rope_fusion = False
        if self.fuse_act_padding and not current_platform.is_rocm():
            logger.warning_once(
                "Padding fusion enabled but the current platform is not ROCm. "
                "The fusion will be disabled."
            )
            self.fuse_act_padding = False
        if self.fuse_rope_kvcache and not current_platform.is_rocm():
            logger.warning_once(
                "KV cache fusion currently only enabled on ROCm. "
                "The fusion will be disabled."
            )
            self.fuse_rope_kvcache = False

    def log_enabled_passes(self) -> None:
        """
        Log the enabled custom fusion passes.
        This is called at the end of VLLMConfig post_init,
        after all defaults are finalized.
        TODO also log the compile ranges for which this is enabled.
        """
        enabled_fusions = [
            f.name[len("fuse_"):]
            for f in fields(self)
            if getattr(self, f.name) and f.name.startswith("fuse_")
        ]

        if enabled_fusions:
            logger.info_once(
                "Enabled custom fusions: %s", ", ".join(enabled_fusions), scope="global"
            )


class DynamicShapesType(str, enum.Enum):
    """Types of dynamic shapes handling in torch.compile().
    see  Dynamic shapes and cfie guard dropping in torch_compile.md
    for more details."""

    BACKED = "backed"
    """Use backed dynamic shapes. torch.compile() guards on backed dynamic
    shapes and may add guards. Symbols are specialized to 0, 1, or >=2 even
    without encountering branching on those ranges."""

    UNBACKED = "unbacked"
    """Use unbacked dynamic shapes. Guaranteed not to be guarded on and not
    0/1 specialized, but may throw data dependent errors when branches require
    their value without explicit unbacked handling."""

    BACKED_SIZE_OBLIVIOUS = "backed_size_oblivious"
    """Experimental flag that treats backed symbols as unbacked when explicit
    unbacked handling is defined."""


@config
class DynamicShapesConfig:
    """Configuration to control/debug torch compile dynamic shapes."""

    type: DynamicShapesType = DynamicShapesType.BACKED
    """Controls the type of dynamic shapes handling to use with torch.compile().

    - BACKED: Default PyTorch behavior with potential guards ignored.
    - UNBACKED: No guards guaranteed (most sound) but may throw
      data dependent errors.
    - BACKED_SIZE_OBLIVIOUS: Experimental safer alternative to
      backed/unbacked.
    """

    evaluate_guards: bool = False
    """
    A debug mode to detect and fail if Dynamo ever specializes a dynamic shape by
    guarding on it. When True, dynamic shape guards are not dropped from dynamo.
    And a failure will be triggered if a recompilation ever happens due to that.
    This mode requires VLLM_USE_BYTECODE_HOOK to be 0.
    Enabling this allow observing the dynamic shapes guards in the tlparse
    artifacts also.
    When type is backed, aot_compile must be disabled for this mode to work.
    until this change picked up https://github.com/pytorch/pytorch/pull/169239.
    """

    assume_32_bit_indexing: bool = False
    """
    whether all tensor sizes can use 32 bit indexing.
    `True` requires PyTorch 2.10+
    """

    def compute_hash(self) -> str:
        """
        Provide a hash for DynamicShapesConfig
        """

        from cfie.config.utils import get_hash_factors, hash_factors

        factors = get_hash_factors(self, {})
        return hash_factors(factors)


@config
class CompilationConfig:
    """编译相关配置。

    这个配置对象必须作为 `VLLMConfig` 的一部分传入构造函数。
    `VLLMConfig.post_init()` 会继续补全这里的若干字段；如果脱离
    `VLLMConfig` 单独使用，本类中的部分字段可能处于未完成初始化的状态。

    本类内部还包含 `PassConfig`，用于控制自定义 fusion / graph
    transformation pass。除此之外，整体配置大致分为三部分：

    - 顶层编译控制：
      - [`mode`][cfie.config.CompilationConfig.mode]
      - [`debug_dump_path`][cfie.config.CompilationConfig.debug_dump_path]
      - [`cache_dir`][cfie.config.CompilationConfig.cache_dir]
      - [`backend`][cfie.config.CompilationConfig.backend]
      - [`custom_ops`][cfie.config.CompilationConfig.custom_ops]
      - [`splitting_ops`][cfie.config.CompilationConfig.splitting_ops]
      - [`compile_mm_encoder`][cfie.config.CompilationConfig.compile_mm_encoder]
    - CUDAGraph 捕获相关：
      - [`cudagraph_mode`][cfie.config.CompilationConfig.cudagraph_mode]
      - [`cudagraph_capture_sizes`]
        [cfie.config.CompilationConfig.cudagraph_capture_sizes]
      - [`max_cudagraph_capture_size`]
        [cfie.config.CompilationConfig.max_cudagraph_capture_size]
      - [`cudagraph_num_of_warmups`]
        [cfie.config.CompilationConfig.cudagraph_num_of_warmups]
      - [`cudagraph_copy_inputs`]
        [cfie.config.CompilationConfig.cudagraph_copy_inputs]
    - Inductor 编译相关：
      - [`compile_sizes`][cfie.config.CompilationConfig.compile_sizes]
      - [`compile_ranges_endpoints`]
        [cfie.config.CompilationConfig.compile_ranges_endpoints]
      - [`inductor_compile_config`]
        [cfie.config.CompilationConfig.inductor_compile_config]
      - [`inductor_passes`][cfie.config.CompilationConfig.inductor_passes]
      - 自定义 inductor pass

    之所以 `cudagraph` 和 `inductor` 分别维护不同的 size 配置，是因为：

    - `cudagraph`：针对某个固定 size 捕获得到的图，只能复用于同样的 size，
      因而需要把想支持的所有 size 都逐一捕获出来。
    - `inductor`：针对一般形状编译出来的图，往往可以覆盖一段 shape 范围。
      它当然也可以针对某个固定 size 做更激进的静态优化，但在大多数场景下，
      通用 shape 编译已经足够；只有部分较小 batch size，定制编译可能更划算。
    """

    # 顶层编译控制
    mode: CompilationMode = Field(default=None)
    """控制模型采用哪种基于 `torch.compile` 的编译方式。

    - `None`：自动选择默认模式；对于 V1 engine，默认是 `3`
    - `0 / NONE`：不做 `torch.compile`，模型完全以 eager PyTorch 方式运行
    - `1 / STOCK_TORCH_COMPILE`：使用标准的 `torch.compile` 编译流水线
    - `2 / DYNAMO_TRACE_ONCE`：只做一次 Dynamo trace，并通过去掉 guard
      避免重复编译；要求模型里不能有依赖动态 shape 的控制流
    - `3 / VLLM_COMPILE`：使用 vLLM 定制的 Inductor 后端，支持缓存、
      分段编译、shape 特化和自定义 pass
    """
    debug_dump_path: Path | None = None
    """调试信息导出目录。"""
    cache_dir: str = ""
    """编译图缓存目录。

    该目录用于保存已编译的图，以加速后续 Inductor 编译。
    默认会根据模型相关信息自动生成一个缓存目录。
    """
    compile_cache_save_format: Literal["binary", "unpacked"] = field(
        default_factory=lambda: envs.VLLM_COMPILE_CACHE_SAVE_FORMAT
    )
    """`torch.compile` 缓存的保存格式。

    - `"binary"`：保存为单个二进制文件，支持多进程安全访问
    - `"unpacked"`：保存为目录结构，便于排查和调试，但不具备多进程安全性

    若未显式指定，则默认读取环境变量 `VLLM_COMPILE_CACHE_SAVE_FORMAT`。
    """
    backend: str = ""
    """编译后端名称，必须以字符串形式给出。

    可选写法包括：

    - `""`（空字符串）：使用默认后端；在 CUDA 类平台上通常是 `"inductor"`
    - `"eager"` / `"openxla"` / ...：使用 PyTorch 中已注册的后端
    - `"full.module.name"`：使用可导入的完整限定名来定位某个后端函数

    这里使用字符串而不是直接传函数对象，主要是为了避免分布式场景下的序列化问题。

    - 当 `mode` 为 `1` 或 `2` 时，`backend` 直接接收整张图并负责编译
    - 当 `mode` 为 `3` 时，后端既可以支持整图编译，也可以支持分段编译；
      可用后端包括 `eager`、`inductor` 以及通过 `get_compile_backend`
      定义的自定义后端

    另外，只有当 `splitting_ops` 配置为相应值、且 `use_inductor_graph_partition`
    关闭时，才会真正启用分段编译。通常默认的 `splitting_ops` 设置已经足够支撑
    piecewise compilation。
    """
    custom_ops: list[str] = field(default_factory=list)
    """对启用/禁用哪些自定义算子（custom ops）进行细粒度控制。

    使用 'all' 表示启用全部自定义算子，
    使用 'none' 表示禁用全部自定义算子。

    也可以额外给出一个自定义算子名字列表：
    - 以 '+' 前缀表示显式启用某个算子
    - 以 '-' 前缀表示显式禁用某个算子

    示例：

    - 'all,-op1'
      表示启用所有自定义算子，但禁用 op1

    - 'none,+op1,+op2'
      表示默认禁用所有自定义算子，只启用 op1 和 op2

    默认行为如下：

    - 当不使用 Inductor 运行时，默认启用所有自定义算子
    - 当使用 Inductor 运行时，默认禁用所有自定义算子
      条件是：mode > CompilationMode.NONE 且 backend="inductor"

    对于被禁用的自定义算子，Inductor 会为它们生成（可能融合后的）Triton kernel。
    """
    splitting_ops: list[str] | None = None
    """在 piecewise compilation 中，用于把哪些算子排除出 cudagraph。

    它的行为取决于 `use_inductor_graph_partition`：

    - 当 `use_inductor_graph_partition=False`（默认）时：
      这些算子会被用作 Dynamo FX 层面的图切分点。Inductor 编译之前，
      整张图会先在这些算子处切开，从而生成多个可用于 cudagraph 捕获的子图。

    - 当 `use_inductor_graph_partition=True` 时：
      这些算子会被注册成 Inductor 的 partition rule。真正的切分发生在
      Inductor codegen 阶段、也就是所有 pass 和 fusion 都跑完之后；
      这样编译和自定义 pass 仍然能基于完整图工作，同时这些算子依然可以被排除在
      cudagraph 之外。

    - 若为 `None`：默认使用 attention 相关算子，适配 piecewise cudagraph
    - 若为空列表 `[]`：表示不排除任何算子，更适合 full cudagraph
    """
    compile_mm_encoder: bool = False
    """是否编译多模态编码器。

    当前仅在部分平台上的 `Qwen2_5_vl` 和 `mLLaMa4` 模型中可用。
    在更多模型经过验证之前，默认保持关闭。
    """

    # Inductor 编译 size 相关配置
    compile_sizes: list[int | str] | None = None
    """指定要为 Inductor 编译哪些 size。

    除了整数外，也支持使用 `"cudagraph_capture_sizes"`，
    表示直接复用 cudagraph 的 capture size 列表。
    """

    compile_ranges_endpoints: list[int] | None = None
    """Inductor 编译区间的端点列表。

    这些端点会形成如下编译区间：

    - `[1, endpoints[0]]`
    - `[endpoints[0] + 1, endpoints[1]]`
    - ...
    - `[endpoints[-1] + 1, max_num_batched_tokens]`

    `compile_sizes` 也会被视作单点区间，即：
    `[compile_sizes[i], compile_sizes[i]]`

    如果某个区间与某个 `compile_size` 重叠，则优先使用该 `compile_size`
    对应的编译图。比如区间是 `[1, 8]`，而 `compile_size=4`，
    那么 size=4 时优先使用为 4 单独编译的图，而不是 `[1, 8]` 区间图。
    """

    inductor_compile_config: dict = field(default_factory=dict)
    """Inductor 的额外配置项。

    - `None`：表示使用默认配置
    """

    inductor_passes: dict[str, str] = field(default_factory=dict)
    """Inductor 的额外 pass 配置。

    该字段是一个字典，键是 pass 名，值是 pass 函数的完整限定名。
    之所以保存函数名而不是直接保存函数对象，是因为配置本身需要支持 JSON 形式。

    如果是直接在 Python 里构造配置对象，也可以直接把函数对象传进来，例如：

    `CompilationConfig(inductor_passes={"a": func})`
    """

    # CUDAGraph 相关配置
    cudagraph_mode: CUDAGraphMode = Field(default=None)
    """CUDAGraph 模式。

    可选模式包括：

    - `NONE`：不做 cudagraph 捕获
    - `PIECEWISE`
    - `FULL`
    - `FULL_DECODE_ONLY`
    - `FULL_AND_PIECEWISE`（V1 默认）

    各模式语义如下：

    - `PIECEWISE`：
      只构建分段 cudagraph，把不兼容 cudagraph 的算子（例如部分 attention）
      保留在图外，以换取更好的通用性。

    - `FULL`：
      对所有 batch 直接捕获整张 cudagraph。它在小模型或小 prompt 工作负载下可能有利，
      但许多 backend 并不支持；从整体性能看，通常 `FULL_AND_PIECEWISE` 更好。

    - `FULL_DECODE_ONLY`：
      只为 decode batch 捕获整图 cudagraph；mixed prefill-decode batch 不走
      cudagraph。这个模式适合 P/D 分离场景下的 decode 实例：prefill 性能不那么关键，
      但可以节省一部分显存。

    - `FULL_AND_PIECEWISE`：
      decode batch 走 full cudagraph；prefill 和 mixed prefill-decode batch
      走 piecewise cudagraph。对大多数模型来说，这是当前默认且整体性能最好的模式。

    目前 `cudagraph_mode` 只用于 V1 engine。

    需要注意的是，cudagraph 逻辑与编译逻辑大体正交：

    - piecewise cudagraph 需要 piecewise compilation
      （即 `mode=VLLM_COMPILE` 且 `splitting_ops` 非空）
    - full cudagraph 则既可以与编译结合，也可以独立使用

    这是一个仍在演进中的新配置项，后续可能继续调整并增加新模式。
    """
    cudagraph_num_of_warmups: int = 0
    """cudagraph 预热次数。

    前若干次执行会被视为 warmup，不参与正式录制。
    只有 warmup 结束后，系统才会开始录制并在后续复用录制好的 cudagraph。
    """
    cudagraph_capture_sizes: list[int] | None = None
    """要捕获 cudagraph 的 size 列表。

    - `None`（默认）：根据 cfie 配置自动推导 capture size
    - `list[int]`：显式指定要捕获哪些 size
    """
    cudagraph_copy_inputs: bool = False
    """是否为 cudagraph 复制输入张量。

    如果调用方能够保证每次都复用同一块输入 buffer，可以设为 `False`；
    否则应设为 `True`，这样编译器会先把输入拷贝到自己管理的内部 buffer。

    默认值为 `False`。
    该选项仅在 `cudagraph_mode=PIECEWISE` 时生效。
    """
    cudagraph_specialize_lora: bool = True
    """是否为“启用 LoRA”和“未启用 LoRA”分别创建独立的 cuda graph。

    若设为 `False`，则即使当前没有激活任何 LoRA adapter，也会统一复用
    带 LoRA 的那份 cuda graph，从而额外承担执行 LoRA 相关算子的开销。

    若设为 `True`，则可以消除这部分额外开销，但代价是启动时间更长、
    显存占用也会略有增加。

    当 `enable_lora=False` 时，该选项无效。
    """

    use_inductor_graph_partition: bool = Field(default=None)
    """是否使用 inductor graph partition 在 `cudagraph_unsafe` 算子处切图。

    这种切分发生在 Inductor codegen 阶段，也就是所有 pass 和 fusion 都执行完之后。
    它会生成一个统一的 `call` 函数：把 cudagraph-safe 的算子包进若干 partition
    function，而把 cudagraph-unsafe 的算子留在 partition 外面。

    如果一张图里有 `N` 个 cudagraph-unsafe 算子（例如 Attention），
    最终就会得到 `N+1` 个 partition。若要把某个自定义算子标记为
    cudagraph unsafe，可以在注册时添加：

    `tags=(torch._C.Tag.cudagraph_unsafe)`

    这个配置的好处是：无需编译两次，就能同时支持 full cudagraph 和
    piecewise cudagraph。

    - 对 piecewise cudagraph：
      会给每个 partition 套一层 vLLM CUDAGraph wrapper；
      如果有 `N+1` 个 partition，就会对应创建 `N+1` 个 wrapper 实例

    - 对 full cudagraph：
      始终是在 model runner 里，把一层总的 CUDAGraph wrapper 放在 inductor
      `call` 函数外层；顶层 full cudagraph 捕获不会关心内部 partition
    """

    pass_config: PassConfig = field(default_factory=PassConfig)
    """自定义 inductor pass 配置；更详细说明见 `PassConfig`。"""

    max_cudagraph_capture_size: int = field(default=None)
    """允许捕获的最大 cudagraph size。

    - 如果显式给了 `cudagraph_capture_sizes`，这里会被设置为其中最大的 size
      （或者在用户也手动设置了本字段时，检查两者是否一致）
    - 如果没有给 `cudagraph_capture_sizes`，系统会按以下模式自动生成 size 列表：

      `[1, 2, 4] + list(range(8, 256, 8)) + list(range(256, max_size + 1, 16))`

    若本字段也未指定，则默认取：

    `min(max_num_seqs * 2, 512)`

    这样做的目的是：

    - 在显存紧张、`max_num_seqs` 又较小的情况下，避免 OOM
    - 避免去捕获大量非常大的图（尤其是大于 512 的图），因为它们会显著拉长启动时间，
      但收益通常有限
    """

    dynamic_shapes_config: DynamicShapesConfig = field(
        default_factory=DynamicShapesConfig
    )
    """动态 shape 相关配置。"""

    local_cache_dir: str = field(default=None, init=False)  # type: ignore
    """每个 rank 自己的本地缓存目录。"""

    fast_moe_cold_start: bool | None = None
    """加速 MoE 冷启动的优化开关。

    这是一个带有前提假设的优化，默认假设：

    1. 当前正在执行的 decoder forward 只属于当前模型
    2. decoder forward 会按初始化顺序依次跑过所有 MoE 层

    只有在这两个条件都成立时，该优化才能显著降低 MoE 模型的冷启动时间。

    可选值含义如下：

    - `True`：始终开启
    - `False`：始终关闭
    - `None`：通常开启，但在 speculative decoding 时关闭

    如果上述两个前提不成立，这个优化会产生静默错误。
    当前最典型的不满足场景就是 speculative decoding，因为此时还会有一个
    draft model，它本身也可能包含 MoE 层。

    目前这只是一个过渡性方案；后续会有不依赖这些前提假设的长期解法。
    """

    # 记录哪些自定义算子最终被启用 / 禁用
    enabled_custom_ops: Counter[str] = field(default_factory=Counter, init=False)
    """实际被启用的自定义算子计数。"""
    disabled_custom_ops: Counter[str] = field(default_factory=Counter, init=False)
    """实际被禁用的自定义算子计数。"""
    traced_files: set[str] = field(default_factory=set, init=False)
    """参与编译 trace 的文件集合。"""
    compilation_time: float = field(default=0.0, init=False)
    """累计编译耗时。"""

    static_forward_context: dict[str, Any] = field(default_factory=dict, init=False)
    """每个模型实例对应的静态 forward 上下文。

    该字典把“层名”映射到“层对象本身”，用于在模型定义代码之外
    访问这些层。例如在 dp_size > 1 等场景下，外部运行时可能需要
    直接拿到 Attention、FusedMoE 等层对象。"""

    static_all_moe_layers: list[str] = field(default_factory=list, init=False)
    """模型中所有 MoE 层的层名列表。"""

    # attention 相关算子；用于 piecewise cudagraph 切图
    # 这里使用 PyTorch 运算符全名格式：`namespace::name`
    _attention_ops: ClassVar[list[str]] = [
        "cfie::unified_attention",
        "cfie::unified_attention_with_output",
        "cfie::unified_mla_attention",
        "cfie::unified_mla_attention_with_output",
        "cfie::mamba_mixer2",
        "cfie::mamba_mixer",
        "cfie::short_conv",
        "cfie::linear_attention",
        "cfie::plamo2_mamba_mixer",
        "cfie::gdn_attention_core",
        "cfie::olmo_hybrid_gdn_full_forward",
        "cfie::kda_attention",
        "cfie::sparse_attn_indexer",
        "cfie::rocm_aiter_sparse_attn_indexer",
    ]

    def compute_hash(self) -> str:
        """
        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        # Opt-out: default-include declared fields; keep a tiny exclude set;
        # normalize types; keep SHA-256. For nested opaque configs, include a
        # stable identifier (e.g., pass_config.compute_hash()) instead of object id.

        ignored_factors = {
            # Paths/dirs and runtime/metrics that don’t affect compiled graph
            "debug_dump_path",
            "cache_dir",
            "local_cache_dir",
            "traced_files",
            "compilation_time",
            "static_forward_context",
            "pass_config",  # handled separately below
            "dynamic_shapes_config",  # handled separately below
        }

        from cfie.config.utils import get_hash_factors, hash_factors

        factors = get_hash_factors(self, ignored_factors)

        factors["pass_config"] = self.pass_config.compute_hash()
        factors["dynamic_shapes_config"] = self.dynamic_shapes_config.compute_hash()
        return hash_factors(factors)

    def __repr__(self) -> str:
        exclude = {
            "static_forward_context": True,
            "enabled_custom_ops": True,
            "disabled_custom_ops": True,
            "compilation_time": True,
            "traced_files": True,
            "inductor_compile_config": {
                "post_grad_custom_post_pass": True,
            },
        }

        # exclude default attr in pass_config
        pass_config_exclude = {}
        for attr, default_val in vars(PassConfig()).items():
            if getattr(self.pass_config, attr) == default_val:
                pass_config_exclude[attr] = True
        if pass_config_exclude:
            exclude["pass_config"] = pass_config_exclude

        config = TypeAdapter(CompilationConfig).dump_python(
            self, exclude=exclude, exclude_unset=True
        )

        return str(config)

    __str__ = __repr__

    @field_validator("mode", mode="before")
    @classmethod
    def validate_mode_before(cls, value: Any) -> Any:
        """
        Enable parsing the `mode` field from string mode names.
        Accepts both integers (0-3) and string names, like NONE, STOCK_TORCH_COMPILE,
        DYNAMO_TRACE_ONCE, VLLM_COMPILE.
        """
        if isinstance(value, str):
            # Convert string mode name to integer value
            mode_name = value.upper()

            if mode_name not in CompilationMode.__members__:
                raise ValueError(
                    f"Invalid compilation mode: {value}. "
                    f"Valid modes are: {', '.join(CompilationMode.__members__.keys())}"
                )

            return CompilationMode[mode_name]
        return value

    @field_validator("cudagraph_mode", mode="before")
    @classmethod
    def validate_cudagraph_mode_before(cls, value: Any) -> Any:
        """Enable parsing of the `cudagraph_mode` enum type from string."""
        if isinstance(value, str):
            return CUDAGraphMode[value.upper()]
        return value

    @field_validator("pass_config", mode="before")
    @classmethod
    def validate_pass_config_before(cls, value: Any) -> Any:
        """Enable parsing of the `pass_config` field from a dictionary."""
        if isinstance(value, dict):
            return PassConfig(**value)
        return value

    @field_validator("compile_cache_save_format")
    @classmethod
    def validate_compile_cache_save_format(cls, value: str) -> str:
        if value not in ("binary", "unpacked"):
            raise ValueError(
                f"compile_cache_save_format must be 'binary' or 'unpacked', "
                f"got: {value}"
            )
        return value

    @field_validator(
        "level",
        "mode",
        "cudagraph_mode",
        "max_cudagraph_capture_size",
        "use_inductor_graph_partition",
        mode="wrap",
    )
    @classmethod
    def _skip_none_validation(cls, value: Any, handler: Callable) -> Any:
        """Skip validation if the value is `None` when initialisation is delayed."""
        if value is None:
            return value
        return handler(value)

    def __post_init__(self) -> None:
        count_none = self.custom_ops.count("none")
        count_all = self.custom_ops.count("all")
        assert count_none + count_all <= 1, "Can only specify 'none' or 'all'"

        # TODO(zou3519/luka): There are 2 issues with auto-functionalization V2:
        # 1. A bug in PyTorch, fixed in 2.7:
        #    https://github.com/pytorch/pytorch/issues/147924
        # 2. Custom passes (fusion) rely on auto-functionalization V1 and don't
        #    work with V2. Addressing this will take extra engineering effort
        #    and it is not yet a priority. RFC here:
        #    https://github.com/cfie-project/cfie/issues/14703

        KEY = "enable_auto_functionalized_v2"
        if KEY not in self.inductor_compile_config:
            self.inductor_compile_config[KEY] = False

        for k, v in self.inductor_passes.items():
            if not isinstance(v, str):
                assert callable(v), f"pass {k} should be callable or a qualified name"
                self.inductor_compile_config[k] = (
                    v if isinstance(v, InductorPass) else CallableInductorPass(v)
                )
                continue

            # resolve function from qualified name
            names = v.split(".")
            module = ".".join(names[:-1])
            func_name = names[-1]
            func = __import__(module).__dict__[func_name]
            self.inductor_compile_config[k] = (
                func if isinstance(func, InductorPass) else CallableInductorPass(func)
            )

        if (
                self.pass_config.enable_qk_norm_rope_fusion
                and "+rotary_embedding" not in self.custom_ops
        ):
            # TODO(zhuhaoran): support rope native forward match and remove this.
            # Linked issue: https://github.com/cfie-project/cfie/issues/28042
            self.custom_ops.append("+rotary_embedding")
        if (
                self.pass_config.fuse_rope_kvcache
                and "+rotary_embedding" not in self.custom_ops
        ):
            # TODO(Rohan138): support rope native forward match and remove this.
            # Linked issue: https://github.com/cfie-project/cfie/issues/28042
            self.custom_ops.append("+rotary_embedding")

        if (
                is_torch_equal_or_newer("2.9.0.dev")
                and "combo_kernels" not in self.inductor_compile_config
                and "benchmark_combo_kernel" not in self.inductor_compile_config
                # (fixme @boyuan) combo kernel does not support cpu yet.
                and not current_platform.is_cpu()
        ):
            # use horizontal fusion, which is useful for fusing qk-norm and
            # qk-rope when query and key have different shapes.
            self.inductor_compile_config["combo_kernels"] = True
            self.inductor_compile_config["benchmark_combo_kernel"] = True

        if self.use_inductor_graph_partition and not is_torch_equal_or_newer(
                "2.9.0.dev"
        ):
            raise ValueError(
                "use_inductor_graph_partition is only "
                "supported with torch>=2.9.0.dev. Set "
                "use_inductor_graph_partition=False instead."
            )

        for op in self.custom_ops:
            if op[0] not in {"+", "-"} and op not in {"all", "none"}:
                raise ValueError(
                    f"Invalid syntax '{op}' for custom op, "
                    "must be 'all', 'none', '+op' or '-op' "
                    "(where 'op' is the registered op name)"
                )

        # Currently only eager and inductor backend are supported.
        # for piecewise compilation. Custom backends are not supported for
        # piecewise compilation. Update when more backends are supported.
        if self.mode == CompilationMode.VLLM_COMPILE and self.backend not in [
            "",
            "eager",
            "inductor",
        ]:
            raise ValueError(
                f"Invalid backend for piecewise compilation: {self.backend}"
            )

        if self.backend == "":
            self.backend = current_platform.get_compile_backend()

        if (
                self.mode != CompilationMode.NONE
                and self.backend == "inductor"
                and current_platform.is_cuda()
                and not _has_triton_runtime()
        ):
            logger.warning_once(
                "Triton is not available in the current CUDA runtime. "
                "Falling back from the inductor compilation backend to eager."
            )
            self.backend = "eager"

    def init_backend(self, cfie_config: "CfieConfig") -> str | Callable:
        """
        Initialize the backend for the compilation config from a cfie config.
        Arguments:
            cfie_config: The cfie config to initialize the backend from.
        Returns:
            The backend for the compilation config.
        """
        if self.mode is None:
            raise ValueError(
                "No compilation mode is set. This method should only be "
                "called via cfie config where the level is set if none is "
                "provided."
            )
        if self.mode == CompilationMode.NONE:
            raise ValueError("No compilation mode is set.")

        from torch._dynamo.backends.registry import list_backends

        torch_backends = list_backends(exclude_tags=tuple())
        if self.mode in [
            CompilationMode.STOCK_TORCH_COMPILE,
            CompilationMode.DYNAMO_TRACE_ONCE,
        ]:
            if self.backend in torch_backends:
                return self.backend
            return resolve_obj_by_qualname(self.backend)

        assert self.mode == CompilationMode.VLLM_COMPILE
        if self.backend not in ["eager", "inductor"]:
            logger.info("Using OOT custom backend for compilation.")

        from cfie.compilation.backends import CfieBackend

        # TODO[@lucaskabela]: See if we can forward prefix
        # https://github.com/cfie-project/cfie/issues/27045
        return CfieBackend(cfie_config)

    def post_init_cudagraph_sizes(self) -> None:
        """To complete the initialization after cudagraph related
        configs are set. This includes:
        - initialize compile_sizes
        """

        computed_compile_sizes = []
        if self.compile_sizes is not None:
            # de-duplicate the sizes provided by the config
            self.compile_sizes = list(set(self.compile_sizes))
            for x in self.compile_sizes:
                if isinstance(x, str):
                    assert x == "cudagraph_capture_sizes", (
                        "Unrecognized size type in compile_sizes, "
                        f"expect 'cudagraph_capture_sizes', got {x}"
                    )
                    computed_compile_sizes.extend(self.cudagraph_capture_sizes)
                else:
                    assert isinstance(x, int)
                    computed_compile_sizes.append(x)
        self.compile_sizes = computed_compile_sizes  # type: ignore

        # make sure the sizes are in ascending order
        self.cudagraph_capture_sizes.sort()
        if self.cudagraph_capture_sizes:
            assert self.cudagraph_capture_sizes[-1] == self.max_cudagraph_capture_size

    def set_splitting_ops_for_v1(
            self, all2all_backend: str, data_parallel_size: int = 1
    ):
        # To compatible with OOT hardware plugin platform (for example cfie-ascend)
        # which currently only supports sequence parallelism in eager mode.
        if self.mode != CompilationMode.VLLM_COMPILE:
            if self.splitting_ops is None:
                self.splitting_ops = []
            return

        # NOTE: this function needs to be called only when mode is
        # CompilationMode.VLLM_COMPILE
        assert self.mode == CompilationMode.VLLM_COMPILE, (
            "set_splitting_ops_for_v1 should only be called when "
            "mode is CompilationMode.VLLM_COMPILE"
        )

        if self.pass_config.fuse_attn_quant and not self.use_inductor_graph_partition:
            self.set_splitting_ops_for_attn_fusion()
        else:
            if self.splitting_ops is None:
                # NOTE: When using full cudagraph, instead of setting an empty
                # list and capture the full cudagraph inside the flattened fx
                # graph, we keep the piecewise fx graph structure but capture
                # the full cudagraph outside the fx graph. This reduces some
                # cpu overhead when the runtime batch_size is not cudagraph
                # captured. see https://github.com/cfie-project/cfie/pull/20059
                # for details. Make a copy to avoid mutating the class-level
                # list via reference.
                self.splitting_ops = list(self._attention_ops)

                # unified_kv_cache_update has a string param that prevents Inductor
                # from reusing piecewise graphs. Remove it from the compiled graph.
                # This has the side-effect of excluding cache from cudagraphs but
                # that doesn't seem to affect performance.
                # https://github.com/cfie-project/cfie/issues/33267
                if not self.use_inductor_graph_partition:
                    self.splitting_ops.append("cfie::unified_kv_cache_update")
                    self.splitting_ops.append("cfie::unified_mla_kv_cache_update")

            elif len(self.splitting_ops) == 0:
                if (
                        self.cudagraph_mode == CUDAGraphMode.PIECEWISE
                        or self.cudagraph_mode == CUDAGraphMode.FULL_AND_PIECEWISE
                ):
                    logger.warning_once(
                        "Using piecewise cudagraph with empty splitting_ops"
                    )
                if self.cudagraph_mode == CUDAGraphMode.PIECEWISE:
                    logger.warning_once(
                        "Piecewise compilation with empty splitting_ops does not "
                        "contain piecewise cudagraph. Setting cudagraph_"
                        "mode to NONE. Hint: If you are using attention "
                        "backends that support cudagraph, consider manually "
                        "setting cudagraph_mode to FULL or FULL_DECODE_ONLY "
                        "to enable full cudagraphs."
                    )
                    self.cudagraph_mode = CUDAGraphMode.NONE
                elif self.cudagraph_mode == CUDAGraphMode.FULL_AND_PIECEWISE:
                    logger.warning_once(
                        "Piecewise compilation with empty splitting_ops does "
                        "not contain piecewise cudagraph. Setting "
                        "cudagraph_mode to FULL."
                    )
                    self.cudagraph_mode = CUDAGraphMode.FULL
                self.splitting_ops = []

        # Disable CUDA graphs for DeepEP high-throughput since its not CG compatible
        if (
                all2all_backend == "deepep_high_throughput"
                and data_parallel_size > 1
                and self.cudagraph_mode != CUDAGraphMode.NONE
        ):
            # TODO: Piecewise Cuda graph might be enabled
            # if torch compile cache key issue fixed
            # See https://github.com/cfie-project/cfie/pull/25093
            logger.info(
                "DeepEP: Disabling CUDA Graphs since DeepEP high-throughput kernels "
                "are optimized for prefill and are incompatible with CUDA Graphs. "
                "In order to use CUDA Graphs for decode-optimized workloads, "
                "use --all2all-backend with another option, such as "
                "deepep_low_latency or allgather_reducescatter."
            )
            self.cudagraph_mode = CUDAGraphMode.NONE

    def set_splitting_ops_for_attn_fusion(self):
        assert self.pass_config.fuse_attn_quant
        if self.splitting_ops is None:
            self.splitting_ops = []
            if self.cudagraph_mode.has_piecewise_cudagraphs():
                logger.warning_once(
                    "fuse_attn_quant is incompatible with piecewise "
                    "cudagraph when use_inductor_graph_partition is off. "
                    "In this case, splitting_ops will be set to empty "
                    "list, and cudagraph_mode will be set to FULL. "
                    "Please ensure you are using attention backends that "
                    "support cudagraph or set cudagraph_mode to NONE "
                    "explicitly if encountering any problems."
                )
                self.cudagraph_mode = CUDAGraphMode.FULL

        assert not self.splitting_ops_contain_attention(), (
            "attention ops should not be in splitting_ops when fuse_attn_quant is True"
        )

    def splitting_ops_contain_attention(self) -> bool:
        return self.splitting_ops is not None and all(
            op in self.splitting_ops for op in self._attention_ops
        )

    def is_attention_compiled_piecewise(self) -> bool:
        if not self.splitting_ops_contain_attention():
            return False

        if not self.use_inductor_graph_partition:
            # Dynamo-level FX split case
            return self.mode == CompilationMode.VLLM_COMPILE

        # Inductor partition case
        return self.backend == "inductor" and self.mode != CompilationMode.NONE

    def custom_op_log_check(self):
        """
        This method logs the enabled/disabled custom ops and checks that the
        passed custom_ops field only contains relevant ops.
        It is called at the end of set_current_cfie_config,
        after the custom ops have been instantiated.
        """

        if len(self.enabled_custom_ops) + len(self.disabled_custom_ops) == 0:
            logger.debug("No custom ops found in model.")
            return

        logger.debug("enabled custom ops: %s", self.enabled_custom_ops)
        logger.debug("disabled custom ops: %s", self.disabled_custom_ops)

        all_ops_in_model = self.enabled_custom_ops | self.disabled_custom_ops
        for op in self.custom_ops:
            if op in {"all", "none"}:
                continue

            assert op[0] in {"+", "-"}, (
                "Invalid custom op syntax (should be checked during init)"
            )

            # check if op name exists in model
            op_name = op[1:]
            if op_name not in all_ops_in_model:
                from cfie.model_executor.custom_op import op_registry

                # Does op exist at all or is it just not present in this model?
                # Note: Only imported op classes appear in the registry.
                missing_str = (
                    "doesn't exist (or wasn't imported/registered)"
                    if op_name not in op_registry
                    else "not present in model"
                )

                enable_str = "enabling" if op[0] == "+" else "disabling"
                logger.warning_once(
                    "Op '%s' %s, %s with '%s' has no effect",
                    op_name,
                    missing_str,
                    enable_str,
                    op,
                )

    def is_custom_op_enabled(self, op: str) -> bool:
        if "all" in self.custom_ops:
            return f"-{op}" not in self.custom_ops

        assert "none" in self.custom_ops
        return f"+{op}" in self.custom_ops

    def adjust_cudagraph_sizes_for_spec_decode(
            self, uniform_decode_query_len: int, tensor_parallel_size: int
    ):
        multiple_of = uniform_decode_query_len
        if tensor_parallel_size > 1 and self.pass_config.enable_sp:
            multiple_of = max(uniform_decode_query_len, tensor_parallel_size)
            if (
                    multiple_of % uniform_decode_query_len != 0
                    or multiple_of % tensor_parallel_size != 0
            ):
                raise ValueError(
                    f"Can't determine cudagraph shapes that are both a "
                    f"multiple of {uniform_decode_query_len} "
                    f"(num_speculative_tokens + 1) required by spec-decode "
                    f"and {tensor_parallel_size} (tensor_parallel_size) "
                    f"required by sequence parallelism please adjust "
                    f"num_speculative_tokens or disable sequence parallelism"
                )

        if not self.cudagraph_capture_sizes or multiple_of <= 1:
            return

        assert self.max_cudagraph_capture_size is not None
        rounded_sizes = sorted(
            set(
                round_up(size, multiple_of)
                for size in self.cudagraph_capture_sizes
                if round_up(size, multiple_of) <= self.max_cudagraph_capture_size
            )
        )

        if len(rounded_sizes) == 0 and multiple_of <= self.max_cudagraph_capture_size:
            # if one valid but would be round_down use that
            rounded_sizes = [multiple_of]

        if len(rounded_sizes) == 0:
            raise ValueError(
                f"No valid cudagraph sizes after rounding to multiple of {multiple_of} "
                f"(num_speculative_tokens + 1 or tp if sequence parallelism is enabled)"
                f" please adjust num_speculative_tokens ({uniform_decode_query_len - 1}"
                f") or max_cudagraph_capture_size ({self.max_cudagraph_capture_size})"
                f" or cudagraph_capture_sizes ({self.cudagraph_capture_sizes})"
            )

        self.max_cudagraph_capture_size = rounded_sizes[-1]
        self.cudagraph_capture_sizes = rounded_sizes

    def adjust_cudagraph_sizes_for_mamba_cache(
            self, num_mamba_cache_blocks: int
    ) -> None:
        """Cap cudagraph capture sizes to available Mamba cache blocks.

        For hybrid Mamba/attention models, the Mamba conv_state and
        ssm_state tensors have their first dimension equal to num_blocks
        (from KVCacheConfig). During CUDA graph capture the decode batch
        size equals num_tokens, so capture sizes exceeding num_blocks
        would cause out-of-bounds access in Mamba kernels.

        See: https://github.com/cfie-project/cfie/issues/34094
        """
        if not self.cudagraph_capture_sizes or num_mamba_cache_blocks <= 0:
            return

        assert self.max_cudagraph_capture_size is not None

        if num_mamba_cache_blocks >= self.max_cudagraph_capture_size:
            return

        capped_sizes = [
            s for s in self.cudagraph_capture_sizes if s <= num_mamba_cache_blocks
        ]

        if len(capped_sizes) == 0:
            logger.warning(
                "No valid cudagraph capture sizes remain after capping "
                "to Mamba cache blocks (%d). The smallest capture size "
                "was %d. Disabling cudagraph capture. Consider reducing "
                "max_num_seqs or increasing available GPU memory.",
                num_mamba_cache_blocks,
                self.cudagraph_capture_sizes[0],
            )
            self.cudagraph_capture_sizes = []
            self.max_cudagraph_capture_size = 0
            return

        logger.warning(
            "Capping cudagraph capture sizes from max %d to %d to fit "
            "Mamba cache blocks (%d blocks available). This limits the "
            "maximum batch size that can use CUDA graphs. To increase "
            "this limit, reduce max_num_seqs or increase available GPU "
            "memory.",
            self.max_cudagraph_capture_size,
            capped_sizes[-1],
            num_mamba_cache_blocks,
        )

        self.max_cudagraph_capture_size = capped_sizes[-1]
        self.cudagraph_capture_sizes = capped_sizes

    def get_compile_ranges(self) -> list[Range]:
        """Get the compile ranges for the compilation config."""
        if self.compile_ranges_endpoints is None:
            return []
        endpoints = sorted(set(self.compile_ranges_endpoints))
        return [
            Range(start=s + 1, end=e) for s, e in zip([0] + endpoints[:-1], endpoints)
        ]
