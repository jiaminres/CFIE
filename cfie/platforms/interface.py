# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import enum
import os
import platform
import sys
from datetime import timedelta
from typing import TYPE_CHECKING, Any, NamedTuple

import torch

from cfie.logger import init_logger
from cfie.v1.attention.backends.registry import AttentionBackendEnum

if TYPE_CHECKING:
    from torch.distributed import PrefixStore, ProcessGroup

    from cfie.config import CfieConfig
    from cfie.inputs import ProcessorInputs
    from cfie.pooling_params import PoolingParams
    from cfie.sampling_params import SamplingParams
    from cfie.utils.argparse_utils import FlexibleArgumentParser
    from cfie.v1.attention.selector import AttentionSelectorConfig
else:
    FlexibleArgumentParser = object

logger = init_logger(__name__)


def in_wsl() -> bool:
    # Reference: https://github.com/microsoft/WSL/issues/4071
    return "microsoft" in " ".join(platform.uname()).lower()


class PlatformEnum(enum.Enum):
    """Enumeration of supported hardware platforms."""

    CUDA = enum.auto()
    ROCM = enum.auto()
    TPU = enum.auto()
    XPU = enum.auto()
    CPU = enum.auto()
    OOT = enum.auto()
    UNSPECIFIED = enum.auto()


class CpuArchEnum(enum.Enum):
    X86 = enum.auto()
    ARM = enum.auto()
    POWERPC = enum.auto()
    S390X = enum.auto()
    RISCV = enum.auto()
    OTHER = enum.auto()
    UNKNOWN = enum.auto()


class DeviceCapability(NamedTuple):
    major: int
    minor: int

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, DeviceCapability):
            return NotImplemented
        return (self.major, self.minor) < (other.major, other.minor)

    def __le__(self, other: Any) -> bool:
        if not isinstance(other, DeviceCapability):
            return NotImplemented
        return (self.major, self.minor) <= (other.major, other.minor)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DeviceCapability):
            return NotImplemented
        return (self.major, self.minor) == (other.major, other.minor)

    def __ge__(self, other: Any) -> bool:
        if not isinstance(other, DeviceCapability):
            return NotImplemented
        return (self.major, self.minor) >= (other.major, other.minor)

    def __gt__(self, other: Any) -> bool:
        if not isinstance(other, DeviceCapability):
            return NotImplemented
        return (self.major, self.minor) > (other.major, other.minor)

    def as_version_str(self) -> str:
        return f"{self.major}.{self.minor}"

    def to_int(self) -> int:
        """
        Express device capability as an integer `<major><minor>`.

        It is assumed that the minor version is always a single digit.
        """
        assert 0 <= self.minor < 10
        return self.major * 10 + self.minor


class Platform:
    _enum: PlatformEnum
    device_name: str
    device_type: str

    # ------------------------------- 定义当前平台在 PyTorch 中使用的 dispatch key -------------------------------
    # 设置当前平台默认使用的 PyTorch dispatch key；若平台未在 PyTorch 中注册，则退回到 CPU。
    dispatch_key: str = "CPU"

    # ------------------------------- 定义当前平台在 Ray 中使用的设备键 -------------------------------
    # 设置当前平台在 Ray 中对应的设备键；空字符串表示当前设备类型不支持 Ray 设备调度。
    ray_device_key: str = ""

    # ------------------------------- 定义当前平台控制可见设备的环境变量名 -------------------------------
    # 记录当前平台用于控制可见设备的环境变量名称，例如 CUDA_VISIBLE_DEVICES。
    device_control_env_var: str = "VLLM_DEVICE_CONTROL_ENV_VAR_PLACEHOLDER"

    # ------------------------------- 定义需要阻止 Ray 设置可见设备的环境变量列表 -------------------------------
    # 保存一组环境变量名；当这些变量被设置为 1 时，可阻止 Ray 自动改写可见设备。
    ray_noset_device_env_vars: list[str] = []

    # ------------------------------- 定义简单函数默认使用的 torch.compile backend -------------------------------
    # 为当前平台设置简单函数与独立函数编译时默认使用的 backend，默认采用 inductor。
    simple_compile_backend: str = "inductor"

    # ------------------------------- 定义当前平台默认的分布式通信 backend -------------------------------
    # 保存当前平台用于分布式通信的 backend 名称；空字符串表示尚未指定。
    dist_backend: str = ""

    # ------------------------------- 定义当前平台支持的量化类型列表 -------------------------------
    # 保存当前平台显式支持的量化方法名称列表；空列表表示不做额外限制。
    supported_quantization: list[str] = []

    # ------------------------------- 定义当前平台附加需要设置的环境变量列表 -------------------------------
    # 保存当前平台初始化或运行时可能需要附加设置的环境变量名列表。
    additional_env_vars: list[str] = []

    # ------------------------------- 定义当前平台级别的全局 graph pool 缓存 -------------------------------
    # 保存当前平台共享的全局 graph pool 句柄；首次访问时懒加载初始化。
    _global_graph_pool: Any | None = None

    @property
    def pass_key(self) -> str:
        # ------------------------------- 返回当前平台在 inductor 中注册自定义 pass 的配置键 -------------------------------
        # 返回用于注册 PassManager 自定义 pass 的 inductor 配置键。
        return "post_grad_custom_post_pass"

    @property
    def supported_dtypes(self) -> list[torch.dtype]:
        # ------------------------------- 返回当前平台支持的数据类型列表 -------------------------------
        # 返回当前平台支持的 dtype 列表，并约定列表中的第一个 dtype 作为 auto 模式下的默认回退类型。
        return [torch.bfloat16, torch.float16, torch.float32]

    def is_cuda(self) -> bool:
        # ------------------------------- 判断当前平台是否为 CUDA -------------------------------
        # 当平台枚举值为 CUDA 时返回 True。
        return self._enum == PlatformEnum.CUDA

    def is_rocm(self) -> bool:
        # ------------------------------- 判断当前平台是否为 ROCm -------------------------------
        # 当平台枚举值为 ROCM 时返回 True。
        return self._enum == PlatformEnum.ROCM

    def is_tpu(self) -> bool:
        # ------------------------------- 判断当前平台是否为 TPU -------------------------------
        # 当平台枚举值为 TPU 时返回 True。
        return self._enum == PlatformEnum.TPU

    def is_xpu(self) -> bool:
        # ------------------------------- 判断当前平台是否为 XPU -------------------------------
        # 当平台枚举值为 XPU 时返回 True。
        return self._enum == PlatformEnum.XPU

    def is_cpu(self) -> bool:
        # ------------------------------- 判断当前平台是否为 CPU -------------------------------
        # 当平台枚举值为 CPU 时返回 True。
        return self._enum == PlatformEnum.CPU

    def is_out_of_tree(self) -> bool:
        # ------------------------------- 判断当前平台是否为树外平台 -------------------------------
        # 当平台枚举值为 OOT 时返回 True。
        return self._enum == PlatformEnum.OOT

    def is_unspecified(self) -> bool:
        # ------------------------------- 判断当前平台是否未指定 -------------------------------
        # 当平台枚举值为 UNSPECIFIED 时返回 True。
        return self._enum == PlatformEnum.UNSPECIFIED

    def get_max_output_tokens(self, prompt_len: int) -> int:
        # ------------------------------- 返回当前平台允许的最大输出 token 数 -------------------------------
        # 默认不对输出 token 数施加平台级限制，因此返回系统允许的最大整数值。
        return sys.maxsize

    def is_cuda_alike(self) -> bool:
        # ------------------------------- 判断当前平台是否属于 CUDA 类平台 -------------------------------
        # 当平台属于 CUDA 或 ROCm 时返回 True，用作无状态版的 cuda 可用性判断。
        return self._enum in (PlatformEnum.CUDA, PlatformEnum.ROCM)

    def is_sleep_mode_available(self) -> bool:
        # ------------------------------- 判断当前平台是否支持 sleep mode -------------------------------
        # 当前默认将 CUDA 与 ROCm 都视为支持 sleep mode 的平台。
        return self._enum in (PlatformEnum.CUDA, PlatformEnum.ROCM)

    @classmethod
    def get_pass_manager_cls(cls) -> str:
        # ------------------------------- 返回当前平台使用的 PassManager 类路径 -------------------------------
        # 返回当前平台用于注册编译 pass 的 PassManager 类完整导入路径。
        return "cfie.compilation.passes.pass_manager.PostGradPassManager"

    @classmethod
    def get_compile_backend(cls) -> str:
        # ------------------------------- 返回当前平台简单编译路径使用的 backend -------------------------------
        # 返回当前平台在简单函数编译路径中使用的 compile backend。
        return cls.simple_compile_backend

    @classmethod
    def device_id_to_physical_device_id(cls, device_id: int):
        # ------------------------------- 将逻辑设备编号映射到物理设备编号 -------------------------------
        # 当平台配置了控制可见设备的环境变量且其值非空时，从该环境变量中解析物理设备映射关系。
        if (
            cls.device_control_env_var in os.environ
            and os.environ[cls.device_control_env_var] != ""
        ):
            # 读取环境变量中的可见设备编号列表。
            device_ids = os.environ[cls.device_control_env_var].split(",")

            # 根据逻辑设备编号取出对应的物理设备编号。
            physical_device_id = device_ids[device_id]

            # 返回转换为整数的物理设备编号。
            return int(physical_device_id)
        else:
            # 当未配置环境变量映射时，直接把逻辑设备编号视为物理设备编号。
            return device_id

    @classmethod
    def import_kernels(cls) -> None:
        # ------------------------------- 导入当前平台相关的 C/CUDA 扩展内核 -------------------------------
        # 优先尝试导入主扩展模块 cfie._C；若失败则仅记录告警而不中断流程。
        try:
            import cfie._C  # noqa: F401
        except ImportError as e:
            logger.warning("Failed to import from cfie._C: %r", e)

        # 在忽略 ImportError 的前提下尝试导入 MoE 相关扩展模块。
        with contextlib.suppress(ImportError):
            import cfie._moe_C  # noqa: F401

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: "AttentionBackendEnum",
        attn_selector_config: "AttentionSelectorConfig",
        num_heads: int | None = None,
    ) -> str:
        # ------------------------------- 返回当前设备平台对应的 attention backend 类 -------------------------------
        # 默认实现返回空字符串，具体平台应在子类中覆盖该方法。
        return ""

    @classmethod
    def get_supported_vit_attn_backends(cls) -> list["AttentionBackendEnum"]:
        # ------------------------------- 返回当前平台支持的 ViT attention backend 列表 -------------------------------
        # 默认仅声明支持 PyTorch 的 SDPA backend。
        return [
            AttentionBackendEnum.TORCH_SDPA,
        ]

    @classmethod
    def get_vit_attn_backend(
            cls,
            head_size: int,
            dtype: torch.dtype,
            backend: "AttentionBackendEnum | None" = None,
    ) -> "AttentionBackendEnum":
        # ------------------------------- 选择当前平台上的 ViT attention backend -------------------------------
        # 当调用方显式指定 backend 时，优先校验并使用该 backend。
        if backend is not None:
            # 校验显式指定的 backend 是否属于当前平台支持的 ViT backend 集合。
            assert backend in cls.get_supported_vit_attn_backends(), (
                f"Backend {backend} is not supported for vit attention"
                f"Supported backends are: {cls.get_supported_vit_attn_backends()}"
            )

            # 记录一次日志，说明当前使用的是显式指定的 backend。
            logger.info_once(f"Using backend {backend} for vit attention")

            # 返回显式指定的 backend。
            return backend

        # ------------------------------- 在未显式指定 backend 时返回默认 ViT backend -------------------------------
        # 记录一次日志，说明当前退回到默认的 TORCH_SDPA backend。
        logger.info_once(
            f"Using default backend {AttentionBackendEnum.TORCH_SDPA} for vit attention"
        )

        # 返回默认的 TORCH_SDPA backend。
        return AttentionBackendEnum.TORCH_SDPA

    @classmethod
    def get_device_capability(
        cls,
        device_id: int = 0,
    ) -> DeviceCapability | None:
        # ------------------------------- 返回当前设备的 capability 信息 -------------------------------
        # 默认实现不提供 capability 信息，具体平台应在子类中覆盖。
        return None

    @classmethod
    def has_device_capability(
        cls,
        capability: tuple[int, int] | int,
        device_id: int = 0,
    ) -> bool:
        # ------------------------------- 判断当前设备 capability 是否至少达到目标值 -------------------------------
        # 先读取当前设备的 capability 信息。
        current_capability = cls.get_device_capability(device_id=device_id)

        # 当当前平台无法提供 capability 信息时，直接返回 False。
        if current_capability is None:
            return False

        # 当目标 capability 以二元组形式给出时，直接按元组大小比较。
        if isinstance(capability, tuple):
            return current_capability >= capability

        # 当目标 capability 以整数形式给出时，比较 capability 的整数编码。
        return current_capability.to_int() >= capability

    @classmethod
    def is_device_capability(
        cls,
        capability: tuple[int, int] | int,
        device_id: int = 0,
    ) -> bool:
        # ------------------------------- 判断当前设备 capability 是否与目标值完全相等 -------------------------------
        # 先读取当前设备的 capability 信息。
        current_capability = cls.get_device_capability(device_id=device_id)

        # 当当前平台无法提供 capability 信息时，直接返回 False。
        if current_capability is None:
            return False

        # 当目标 capability 以二元组形式给出时，直接按元组相等比较。
        if isinstance(capability, tuple):
            return current_capability == capability

        # 当目标 capability 以整数形式给出时，比较 capability 的整数编码是否完全相等。
        return current_capability.to_int() == capability

    @classmethod
    def is_device_capability_family(
        cls,
        capability: int,
        device_id: int = 0,
    ) -> bool:
        # ------------------------------- 判断当前设备 capability 是否属于指定大版本 family -------------------------------
        # 先读取当前设备的 capability 信息。
        current_capability = cls.get_device_capability(device_id=device_id)

        # 当当前平台无法提供 capability 信息时，直接返回 False。
        if current_capability is None:
            return False

        # 仅比较 capability 的十位及以上部分，以判断是否属于同一大版本 family。
        return (current_capability.to_int() // 10) == (capability // 10)

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        # ------------------------------- 返回指定设备的名称 -------------------------------
        # 默认实现要求子类覆盖，否则抛出未实现异常。
        raise NotImplementedError

    @classmethod
    def get_device_uuid(cls, device_id: int = 0) -> str:
        # ------------------------------- 返回指定设备的唯一标识 -------------------------------
        # 默认实现要求子类覆盖，否则抛出未实现异常。
        raise NotImplementedError

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        # ------------------------------- 返回指定设备的总显存或总设备内存 -------------------------------
        # 默认实现要求子类覆盖，否则抛出未实现异常。
        raise NotImplementedError

    @classmethod
    def inference_mode(cls):
        # ------------------------------- 返回当前平台推荐使用的 inference_mode 上下文 -------------------------------
        # 默认返回 torch.inference_mode；某些平台可在子类中改写为 no_grad 等替代形式。
        return torch.inference_mode(mode=True)

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        # ------------------------------- 设置当前平台使用的设备 -------------------------------
        # 默认实现要求子类覆盖，否则抛出未实现异常。
        raise NotImplementedError

    @classmethod
    def pre_register_and_update(
        cls, parser: FlexibleArgumentParser | None = None
    ) -> None:
        # ------------------------------- 在配置初始化前执行平台相关的预注册或预更新动作 -------------------------------
        # 默认实现不执行任何操作，树外平台可在子类中覆盖该方法。
        pass

    @classmethod
    def apply_config_platform_defaults(cls, cfie_config: "CfieConfig") -> None:
        # ------------------------------- 将平台相关默认值应用到配置对象 -------------------------------
        # 默认实现不修改配置对象，具体平台可在子类中原地更新配置默认值。
        pass

    @classmethod
    def check_and_update_config(cls, cfie_config: "CfieConfig") -> None:
        # ------------------------------- 校验并修正当前平台下的配置兼容性 -------------------------------
        # 默认实现不做任何处理，具体平台可在子类中检查兼容性并原地修正配置。
        pass

    @classmethod
    def update_block_size_for_backend(cls, cfie_config: "CfieConfig") -> None:
        # ------------------------------- 根据 attention backend 自动调整 kv cache block_size -------------------------------
        from cfie.config.cache import CacheConfig

        # 读取配置中的 cache_config 对象。
        cache_config = cfie_config.cache_config

        # 当用户已经显式指定 block_size 时，保留用户配置，不再自动修改。
        if cache_config.user_specified_block_size:
            return

        # 读取模型配置对象。
        model_config = cfie_config.model_config

        # 当模型配置不存在或属于 hybrid 模型时，直接退回到默认 block_size。
        if model_config is None or model_config.is_hybrid:
            cache_config.block_size = CacheConfig.DEFAULT_BLOCK_SIZE
            return

        from cfie.config.cfie import (
            get_layers_from_cfie_config,
            set_current_cfie_config,
        )
        from cfie.model_executor.layers.attention_layer_base import (
            AttentionLayerBase,
        )

        # 从当前配置中收集全部 attention 层。
        attn_layers = get_layers_from_cfie_config(
            cfie_config,
            AttentionLayerBase,
        )

        # 当没有 attention 层时，退回到默认 block_size。
        if not attn_layers:
            cache_config.block_size = CacheConfig.DEFAULT_BLOCK_SIZE
            return

        # 取第一个 attention 层作为 backend 选择参考。
        first_layer = next(iter(attn_layers.values()))

        # 获取该 attention 层实际使用的 backend 类。
        backend_cls = first_layer.get_attn_backend()

        # 在当前配置上下文中查询该 backend 的首选 block_size。
        with set_current_cfie_config(cfie_config):
            preferred = backend_cls.get_preferred_block_size(
                CacheConfig.DEFAULT_BLOCK_SIZE
            )

        # 当首选 block_size 与默认值不一致时，打印一次日志。
        if preferred != CacheConfig.DEFAULT_BLOCK_SIZE:
            logger.info(
                "Setting kv cache block size to %d for %s backend.",
                preferred,
                backend_cls.get_name(),
            )

        # 将 cache_config 的 block_size 更新为 backend 首选值。
        cache_config.block_size = preferred

    @classmethod
    def verify_model_arch(cls, model_arch: str) -> None:
        # ------------------------------- 校验当前平台是否支持指定模型架构 -------------------------------
        # 默认实现视为全部模型架构均可支持，不执行额外校验。
        pass

    @classmethod
    def verify_quantization(cls, quant: str) -> None:
        # ------------------------------- 校验当前平台是否支持指定量化方式 -------------------------------
        # 当平台显式声明了支持的量化列表，且目标量化方式不在其中时，抛出异常。
        if cls.supported_quantization and quant not in cls.supported_quantization:
            raise ValueError(
                f"{quant} quantization is currently not supported in {cls.device_name}."
            )

    @classmethod
    def get_cpu_architecture(cls) -> CpuArchEnum:
        # ------------------------------- 检测当前系统的 CPU 架构 -------------------------------
        # 读取当前系统的 machine 字段并统一转为小写。
        machine = platform.machine().lower()

        # 当 machine 落在 x86 常见取值集合中时，返回 X86 架构。
        if machine in ("x86_64", "amd64", "i386", "i686"):
            return CpuArchEnum.X86

        # 当 machine 以 arm 或 aarch 开头时，返回 ARM 架构。
        elif machine.startswith("arm") or machine.startswith("aarch"):
            return CpuArchEnum.ARM

        # 当 machine 以 ppc 开头时，返回 POWERPC 架构。
        elif machine.startswith("ppc"):
            return CpuArchEnum.POWERPC

        # 当 machine 等于 s390x 时，返回 S390X 架构。
        elif machine == "s390x":
            return CpuArchEnum.S390X

        # 当 machine 以 riscv 开头时，返回 RISCV 架构。
        elif machine.startswith("riscv"):
            return CpuArchEnum.RISCV

        # 当 machine 非空但不属于已知架构时，返回 OTHER；否则返回 UNKNOWN。
        return CpuArchEnum.OTHER if machine else CpuArchEnum.UNKNOWN

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        # ------------------------------- 判断当前平台是否可用 pinned memory -------------------------------
        # 当检测到当前运行环境位于 WSL 中时，记录告警并返回 False。
        if in_wsl():
            logger.warning(
                "Using 'pin_memory=False' as WSL is detected. "
                "This may slow down the performance."
            )
            return False

        # 当不在 WSL 中时，默认认为 pinned memory 可用。
        return True

    @classmethod
    def get_current_memory_usage(
        cls, device: torch.types.Device | None = None
    ) -> float:
        # ------------------------------- 返回当前平台的内存占用 -------------------------------
        # 默认实现要求子类覆盖，否则抛出未实现异常。
        raise NotImplementedError

    @classmethod
    def get_punica_wrapper(cls) -> str:
        # ------------------------------- 返回当前平台使用的 punica wrapper 类路径 -------------------------------
        # 默认实现要求子类覆盖，否则抛出未实现异常。
        raise NotImplementedError

    @classmethod
    def get_infinity_values(cls, dtype: torch.dtype) -> tuple[float, float]:
        # ------------------------------- 返回当前平台使用的负无穷与正无穷值 -------------------------------
        # 默认直接返回 Python 浮点语义下的负无穷和正无穷。
        return float("-inf"), float("inf")

    @classmethod
    def can_update_inplace(cls) -> bool:
        # ------------------------------- 判断当前平台是否允许原地更新内存 -------------------------------
        # 默认认为当前平台允许 inplace 更新。
        return True

    @classmethod
    def get_lora_vocab_padding_size(cls) -> int:
        # ------------------------------- 返回 LoRA logits 所需的词表 padding 大小 -------------------------------
        # 默认返回 256，供 LoRA 相关 kernel 使用。
        return 256

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        # ------------------------------- 返回当前平台使用的设备通信器类路径 -------------------------------
        # 默认返回基础设备通信器类的完整导入路径。
        return "cfie.distributed.device_communicators.base_device_communicator.DeviceCommunicatorBase"

    @classmethod
    def supports_mx(cls) -> bool:
        # ------------------------------- 判断当前平台是否支持 MX 类型 -------------------------------
        # 默认返回 False。
        return False

    @classmethod
    def supports_fp8(cls) -> bool:
        # ------------------------------- 判断当前平台是否支持 FP8 类型 -------------------------------
        # 默认返回 False。
        return False

    @classmethod
    def is_fp8_fnuz(cls) -> bool:
        # ------------------------------- 判断当前平台首选的 FP8 表示是否为 FNUZ -------------------------------
        # 默认返回 False，表示平台不优先采用 FNUZ 表示。
        return False

    @classmethod
    def fp8_dtype(cls) -> torch.dtype:
        # ------------------------------- 返回当前平台首选的 FP8 dtype -------------------------------
        # 默认返回 torch.float8_e4m3fn。
        return torch.float8_e4m3fn

    @classmethod
    def use_all_gather(cls) -> bool:
        # ------------------------------- 判断当前平台是否在 LogitsProcessor 中使用 all_gather -------------------------------
        # 默认返回 True。
        return True

    @classmethod
    def use_custom_allreduce(cls) -> bool:
        # ------------------------------- 判断当前平台是否支持自定义 allreduce -------------------------------
        # 默认返回 False。
        return False

    @classmethod
    def opaque_attention_op(cls) -> bool:
        # ------------------------------- 判断当前平台是否将 attention 注册为单个不透明自定义算子 -------------------------------
        # 默认返回 False。
        return False

    @classmethod
    def validate_request(
        cls,
        processed_inputs: "ProcessorInputs",
        params: "SamplingParams | PoolingParams",
    ) -> None:
        # ------------------------------- 校验当前请求在当前平台上是否受支持 -------------------------------
        # 默认实现不执行任何检查，具体平台可在子类中覆盖。
        return None

    def __getattr__(self, key: str):
        # ------------------------------- 为未显式定义的属性提供设备模块级别的回退查找 -------------------------------
        # 当访问的是双下划线方法时，直接抛出 AttributeError，避免 pickle 等逻辑误判。
        if key.startswith("__") and key.endswith("__"):
            raise AttributeError(key)

        # 从 torch.<device_type> 模块中尝试获取同名属性。
        device = getattr(torch, self.device_type, None)
        if device is not None and hasattr(device, key):
            attr = getattr(device, key)

            # 当该属性实际存在且不为 None 时，直接返回。
            if attr is not None:
                return attr

        # 当当前平台及其 torch 设备模块上都不存在该属性时，打印告警并返回 None。
        logger.warning(
            "Current platform %s does not have '%s' attribute.",
            self.device_type,
            key,
        )
        return None

    def get_global_graph_pool(self) -> Any:
        # ------------------------------- 返回当前平台共享的全局 graph pool -------------------------------
        # 获取当前实例所属的类对象。
        cls = self.__class__

        # 当类级别 graph pool 尚未初始化时，调用 graph_pool_handle 进行懒加载初始化。
        if cls._global_graph_pool is None:
            cls._global_graph_pool = self.graph_pool_handle()

        # 返回当前平台共享的全局 graph pool 句柄。
        return cls._global_graph_pool

    @classmethod
    def get_static_graph_wrapper_cls(cls) -> str:
        # ------------------------------- 返回当前平台使用的 static graph wrapper 类路径 -------------------------------
        # 默认返回抽象静态图包装器类的完整导入路径。
        return "cfie.compilation.base_static_graph.AbstractStaticGraphWrapper"

    @classmethod
    def stateless_init_device_torch_dist_pg(
        cls,
        backend: str,
        prefix_store: "PrefixStore",
        group_rank: int,
        group_size: int,
        timeout: timedelta,
    ) -> "ProcessGroup":
        # ------------------------------- 初始化当前平台的无状态设备侧 torch 分布式进程组 -------------------------------
        # 默认实现要求子类覆盖，否则抛出未实现异常。
        raise NotImplementedError

    @classmethod
    def check_if_supports_dtype(cls, dtype: torch.dtype):
        # ------------------------------- 校验当前平台是否支持指定 dtype -------------------------------
        # 默认实现要求子类覆盖，否则抛出未实现异常。
        raise NotImplementedError

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        # ------------------------------- 判断当前平台是否支持 hybrid kv cache -------------------------------
        # 默认返回 False。
        return False

    @classmethod
    def support_static_graph_mode(cls) -> bool:
        # ------------------------------- 判断当前平台是否支持静态图模式 -------------------------------
        # 默认返回 False。
        return False

    @classmethod
    def use_custom_op_collectives(cls) -> bool:
        # ------------------------------- 判断当前平台是否使用 torch.ops.cfie.* 自定义 collective 算子 -------------------------------
        # 默认返回 False，只有显式选择的平台才会开启。
        return False

    @classmethod
    def use_sync_weight_loader(cls) -> bool:
        # ------------------------------- 判断当前平台是否需要同步权重加载器 -------------------------------
        # 默认返回 False。
        return False

    @classmethod
    def make_synced_weight_loader(cls, original_weight_loader):
        # ------------------------------- 根据平台需求为权重加载器包一层同步逻辑 -------------------------------
        # 当当前平台不需要同步权重加载时，直接返回原始加载器。
        if not cls.use_sync_weight_loader():
            return original_weight_loader

        # 定义带同步逻辑的包装权重加载器。
        def _synced_weight_loader(param, *args, **kwargs):
            # 先调用原始权重加载器完成权重写入。
            out = original_weight_loader(param, *args, **kwargs)

            # 当参数不在 CPU 上时，调用 torch._sync 强制同步。
            if param.device != torch.device("cpu"):
                torch._sync(param)

            # 返回原始加载器的返回值。
            return out

        # 返回同步版权重加载器。
        return _synced_weight_loader

    @classmethod
    def get_nixl_supported_devices(cls) -> dict[str, tuple[str, ...]]:
        # ------------------------------- 返回当前平台在 nixl 中支持的设备映射 -------------------------------
        # 默认返回空映射。
        return {}

    @classmethod
    def get_nixl_memory_type(cls) -> str | None:
        # ------------------------------- 返回当前平台在 nixl 中使用的内存类型 -------------------------------
        # 默认返回 None。
        return None

    @classmethod
    def check_max_model_len(cls, max_model_len: int) -> int:
        # ------------------------------- 校验并返回当前平台允许的最大模型长度 -------------------------------
        # 默认不修改输入值，直接返回原始 max_model_len。
        return max_model_len

    @classmethod
    def set_additional_forward_context(cls, *args, **kwargs) -> dict[str, Any]:
        # ------------------------------- 为当前平台设置额外的前向上下文字段 -------------------------------
        # 默认不设置额外上下文，返回空字典。
        return {}

    @classmethod
    def num_compute_units(cls, device_id: int = 0) -> int:
        # ------------------------------- 返回当前平台设备的计算单元数量 -------------------------------
        # 默认实现要求子类覆盖，否则抛出未实现异常。
        raise NotImplementedError(
            "num_compute_units is not implemented for the current platform."
        )


class UnspecifiedPlatform(Platform):
    _enum = PlatformEnum.UNSPECIFIED
    device_type = ""
