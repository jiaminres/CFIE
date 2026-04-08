# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
This module re-exports linear kernel implementations to provide a
stable import interface during an ongoing reorganization. Upcoming
PRs will remove the scaled_mm and mixed_precision subdirectories
and reorganize kernels by provider (aiter, cutlass, flashinfer, etc.)
rather than by precision type. By centralizing exports here, we
minimize the need to update imports across other modules when the
internal structure changes. If you are adding a new kernel selector
or kernel implementation, add it to this __init__.py to maintain
import stability.
"""

from typing import TypeVar

import torch

import cfie.envs as envs
from cfie.logger import init_logger
from cfie.model_executor.kernels.linear.mixed_precision import (
    MPLinearKernel,
    MPLinearLayerConfig,
)
from cfie.model_executor.kernels.linear.mixed_precision.allspark import (
    AllSparkLinearKernel,
)
from cfie.model_executor.kernels.linear.mixed_precision.conch import (
    ConchLinearKernel,
)
from cfie.model_executor.kernels.linear.mixed_precision.cpu import (
    CPUWNA16LinearKernel,
)
from cfie.model_executor.kernels.linear.mixed_precision.cutlass import (
    CutlassW4A8LinearKernel,
)
from cfie.model_executor.kernels.linear.mixed_precision.dynamic_4bit import (
    Dynamic4bitLinearKernel,
)
from cfie.model_executor.kernels.linear.mixed_precision.exllama import (
    ExllamaLinearKernel,
)
from cfie.model_executor.kernels.linear.mixed_precision.machete import (
    MacheteLinearKernel,
)
from cfie.model_executor.kernels.linear.mixed_precision.marlin import (
    MarlinLinearKernel,
)
from cfie.model_executor.kernels.linear.mixed_precision.xpu import (
    XPUwNa16LinearKernel,
)
from cfie.model_executor.kernels.linear.scaled_mm import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
    Int8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
    ScaledMMLinearKernel,
    ScaledMMLinearLayerConfig,
)
from cfie.model_executor.kernels.linear.scaled_mm.aiter import (
    AiterInt8ScaledMMLinearKernel,
)
from cfie.model_executor.kernels.linear.scaled_mm.cpu import (
    CPUInt8ScaledMMLinearKernel,
)
from cfie.model_executor.kernels.linear.scaled_mm.cutlass import (
    CutlassFP8ScaledMMLinearKernel,
    CutlassInt8ScaledMMLinearKernel,
)
from cfie.model_executor.kernels.linear.scaled_mm.flashinfer import (
    FlashInferFP8ScaledMMLinearKernel,
)
from cfie.model_executor.kernels.linear.scaled_mm.pytorch import (
    ChannelWiseTorchFP8ScaledMMLinearKernel,
    PerTensorTorchFP8ScaledMMLinearKernel,
    RowWiseTorchFP8ScaledMMLinearKernel,
)
from cfie.model_executor.kernels.linear.scaled_mm.rocm import (
    ROCmFP8ScaledMMLinearKernel,
)
from cfie.model_executor.kernels.linear.scaled_mm.triton import (
    TritonInt8ScaledMMLinearKernel,
)
from cfie.model_executor.kernels.linear.scaled_mm.xpu import (
    XPUFP8ScaledMMLinearKernel,
)
from cfie.model_executor.layers.quantization.utils.quant_utils import QuantKey
from cfie.platforms import PlatformEnum, current_platform

logger = init_logger(__name__)

# in priority/performance order (when available)
_POSSIBLE_INT8_KERNELS: dict[PlatformEnum, list[type[Int8ScaledMMLinearKernel]]] = {
    PlatformEnum.CPU: [CPUInt8ScaledMMLinearKernel],
    PlatformEnum.CUDA: [
        CutlassInt8ScaledMMLinearKernel,
        TritonInt8ScaledMMLinearKernel,
    ],
    PlatformEnum.ROCM: [AiterInt8ScaledMMLinearKernel, TritonInt8ScaledMMLinearKernel],
}

# in priority/performance order (when available)
_POSSIBLE_FP8_KERNELS: dict[PlatformEnum, list[type[FP8ScaledMMLinearKernel]]] = {
    PlatformEnum.CUDA: [
        FlashInferFP8ScaledMMLinearKernel,
        CutlassFP8ScaledMMLinearKernel,
        PerTensorTorchFP8ScaledMMLinearKernel,
        ChannelWiseTorchFP8ScaledMMLinearKernel,
    ],
    PlatformEnum.ROCM: [
        ROCmFP8ScaledMMLinearKernel,
        PerTensorTorchFP8ScaledMMLinearKernel,
        RowWiseTorchFP8ScaledMMLinearKernel,
        ChannelWiseTorchFP8ScaledMMLinearKernel,
    ],
    PlatformEnum.CPU: [
        PerTensorTorchFP8ScaledMMLinearKernel,
        ChannelWiseTorchFP8ScaledMMLinearKernel,
    ],
    PlatformEnum.XPU: [
        XPUFP8ScaledMMLinearKernel,
    ],
}

# in priority/performance order (when available)
_POSSIBLE_KERNELS: dict[PlatformEnum, list[type[MPLinearKernel]]] = {
    PlatformEnum.CUDA: [
        CutlassW4A8LinearKernel,
        MacheteLinearKernel,
        AllSparkLinearKernel,
        MarlinLinearKernel,
        ConchLinearKernel,
        ExllamaLinearKernel,
    ],
    PlatformEnum.ROCM: [
        ConchLinearKernel,
        ExllamaLinearKernel,
    ],
    PlatformEnum.XPU: [
        XPUwNa16LinearKernel,
    ],
    PlatformEnum.CPU: [
        Dynamic4bitLinearKernel,
        CPUWNA16LinearKernel,
    ],
}

_KernelT = TypeVar("_KernelT", bound=ScaledMMLinearKernel)
_KernelConfigT = TypeVar("_KernelConfigT", bound=ScaledMMLinearLayerConfig)


def is_supported_and_can_implement_kernel(
    kernel: type[_KernelT], config: _KernelConfigT, compute_capability: int | None
) -> tuple[bool, str]:
    if kernel.__name__ in envs.VLLM_DISABLED_KERNELS:
        return False, f" {kernel.__name__} is disabled by environment variable"

    if compute_capability is None:
        _cc = current_platform.get_device_capability()
        if _cc is not None:
            compute_capability = _cc[0] * 10 + _cc[1]

    is_supported, failure_reason = kernel.is_supported(compute_capability)
    if not is_supported:
        return False, f"{kernel.__name__} {failure_reason}."

    can_implement, failure_reason = kernel.can_implement(config)
    if not can_implement:
        return (
            False,
            f"{kernel.__name__} {failure_reason}.",
        )

    return True, ""


def choose_scaled_mm_linear_kernel(
    config: _KernelConfigT,
    possible_kernels: dict[PlatformEnum, list[type[_KernelT]]],
    compute_capability: int | None = None,
    force_kernel: type[_KernelT] | None = None,
) -> type[_KernelT]:
    """
    Choose a _KernelT that can implement the given config for the
    given compute capability. Attempts to choose the best kernel in terms of
    performance.

    Args:
        config (_KernelConfigT): Description of the linear layer
            to be implemented.
        possible_kernels (dict[PlatformEnum, list[_KernelT]]): A
            dictionary of platforms and their list of possible kernels.
        compute_capability (Optional[int], optional): The compute capability of
            the target device, if None uses `current_platform` to get the
            compute capability. Defaults to None.
        force_kernel (Optional[type[_KernelT]]): An Optional forced kernel to override
            the possible_kernels if it can be implemented. If None, it will only try the
            possible kernels.

    Raises:
        ValueError: If no kernel can implement the given config.

    Returns:
        _KernelT: Chosen kernel.
    """

    failure_reason_list = []

    if force_kernel is not None:
        can_implement, failure_reason = is_supported_and_can_implement_kernel(
            force_kernel, config, compute_capability
        )
        if can_implement:
            return force_kernel

        logger.info_once(
            "Tried to force %s, but the kernel couldn't be implemented",
            force_kernel.__name__,
            scope="global",
        )

    for kernel in possible_kernels[current_platform._enum]:
        is_supported_and_can_implement, failure_reason = (
            is_supported_and_can_implement_kernel(kernel, config, compute_capability)
        )
        if is_supported_and_can_implement:
            return kernel
        failure_reason_list.append(failure_reason)

    raise ValueError(
        "Failed to find a kernel that can implement the "
        "ScaledMM linear layer. Reasons: \n" + "\n".join(failure_reason_list)
    )


def init_fp8_linear_kernel(
    activation_quant_key: QuantKey,
    weight_quant_key: QuantKey,
    out_dtype: torch.dtype,
    force_kernel: type[FP8ScaledMMLinearKernel] | None = None,
    module_name: str | None = None,
) -> FP8ScaledMMLinearKernel:
    scaled_mm_linear_kernel_config = FP8ScaledMMLinearLayerConfig(
        weight_quant_key=weight_quant_key,
        activation_quant_key=activation_quant_key,
        out_dtype=out_dtype,
    )

    kernel_type = choose_scaled_mm_linear_kernel(
        scaled_mm_linear_kernel_config, _POSSIBLE_FP8_KERNELS, force_kernel=force_kernel
    )

    if module_name:
        logger.info_once(
            "Selected %s for %s",
            kernel_type.__name__,
            module_name,
            scope="global",
        )

    return kernel_type(
        scaled_mm_linear_kernel_config,
        layer_param_names=["weight", "weight_scale", "input_scale", "input_scale_ub"],
    )


def init_int8_linear_kernel(
    is_channelwise: bool,
    is_static_input_scheme: bool,
    input_symmetric: bool,
    module_name: str,
) -> Int8ScaledMMLinearKernel:
    config = Int8ScaledMMLinearLayerConfig(
        is_channelwise=is_channelwise,
        is_static_input_scheme=is_static_input_scheme,
        input_symmetric=input_symmetric,
    )

    kernel_type = choose_scaled_mm_linear_kernel(
        config,
        _POSSIBLE_INT8_KERNELS,
    )

    logger.info_once(
        "Selected %s for %s",
        kernel_type.__name__,
        module_name,
        scope="global",
    )

    return kernel_type(
        config,
        layer_param_names=[
            "weight",
            "weight_scale",
            "input_scale",
            "input_zero_point",
            "azp_adj",
        ],
    )


def choose_mp_linear_kernel(
    config: MPLinearLayerConfig,                 # 当前线性层的实现需求描述，包含 shape / quant_type / act_type / group_size / 是否有 g_idx 等
    compute_capability: int | None = None       # 目标设备算力，格式通常是 80 / 86 / 89 / 90 这种；若不传则自动探测
) -> type[MPLinearKernel]:
    """
    选择一个能够实现给定 config 的 MPLinearKernel。
    选择目标不只是“能跑”，还尽量偏向性能更好的 kernel。

    返回值不是 kernel 实例，而是 kernel“类”：
        type[MPLinearKernel]

    后续上层代码通常会再做：
        kernel_type(...)
    去实例化它。
    """

    # -------------------------------------------------------------
    # Step 1. 若外部没传 compute_capability，则自动从当前平台探测
    # -------------------------------------------------------------
    if compute_capability is None:
        # 如果当前平台对象都没有，就无法知道设备算力
        # 这时没法做 kernel 能力筛选，直接报错
        if current_platform is None:
            raise ValueError("Cannot determine compute capability")

        # 从当前平台获取设备 capability
        # 常见返回形式可能是:
        #   (8, 0)  -> A100 风格 SM80
        #   (8, 6)  -> RTX30 系 SM86
        #   (8, 9)  -> Ada 某些卡 SM89
        #   (9, 0)  -> Hopper SM90
        _cc = current_platform.get_device_capability()

        if _cc is not None:
            # 把 (major, minor) 转成整数形式
            # 例如:
            #   (8, 0) -> 80
            #   (8, 6) -> 86
            #   (9, 0) -> 90
            compute_capability = _cc[0] * 10 + _cc[1]

    # -------------------------------------------------------------
    # Step 2. 准备记录“为什么某些 kernel 不可用”
    # -------------------------------------------------------------
    # 如果最后一个都选不出来，会把这些失败原因拼起来抛出
    failure_reasons = []

    # -------------------------------------------------------------
    # Step 3. 遍历当前平台上的所有候选 kernel
    # -------------------------------------------------------------
    # _POSSIBLE_KERNELS 按平台枚举维护候选 kernel 列表
    #
    # 例如逻辑上可能像：
    #   _POSSIBLE_KERNELS[CUDA] = [KernelA, KernelB, KernelC, ...]
    #
    # 注意：
    # - 顺序很重要
    # - 一般默认候选列表已经按“优先级 / 性能偏好”排好
    # - 因此函数会返回“第一个能实现 config 的 kernel”
    for kernel in _POSSIBLE_KERNELS[current_platform._enum]:

        # ---------------------------------------------------------
        # Step 3.1 如果该 kernel 被环境变量禁用了，则直接跳过
        # ---------------------------------------------------------
        # envs.VLLM_DISABLED_KERNELS 通常是一个名字集合
        # 用户可通过环境变量手动屏蔽某些 kernel
        #
        # 例如调试时可能会禁用某些 backend，以强制走别的路径
        if kernel.__name__ in envs.VLLM_DISABLED_KERNELS:
            failure_reasons.append(
                f" {kernel.__name__} disabled by environment variable"
            )
            continue

        # ---------------------------------------------------------
        # Step 3.2 检查当前设备算力是否满足该 kernel 的最低要求
        # ---------------------------------------------------------
        # kernel.get_min_capability() 返回这个 kernel 至少要求的 SM 能力
        #
        # 例如：
        #   某 kernel 要求 >= 80
        #   当前设备若是 75，则不能用
        if (
            compute_capability is not None
            and kernel.get_min_capability() > compute_capability
        ):
            failure_reasons.append(
                f"{kernel.__name__} requires capability "
                f"{kernel.get_min_capability()}, current compute "
                f" capability is {compute_capability}"
            )
            continue

        # ---------------------------------------------------------
        # Step 3.3 让 kernel 自己判断：能不能实现当前这个 config
        # ---------------------------------------------------------
        # can_implement(config) 一般会检查：
        # - quant_type 是否支持（如 int4 / int8 / fp8）
        # - act_type 是否支持（如 fp16 / bf16 / int8 / fp8）
        # - K/N 形状是否满足 tile 对齐要求
        # - group_size 是否支持
        # - 是否支持 zero-points / g_idx / desc_act
        # - 是否支持 row/column parallel 对应的 partition shape
        #
        # 返回：
        #   can_implement: bool
        #   failure_reason: str
        can_implement, failure_reason = kernel.can_implement(config)

        # ---------------------------------------------------------
        # Step 3.4 若当前 kernel 可以实现，则立即返回它
        # ---------------------------------------------------------
        # 注意这里返回的是 kernel 类，不是实例
        #
        # 由于候选列表通常已按“性能优先级”排序，
        # 所以第一个可用的 kernel 就被视为最优选择
        if can_implement:
            return kernel

        # ---------------------------------------------------------
        # Step 3.5 否则记录失败原因，继续尝试下一个 kernel
        # ---------------------------------------------------------
        else:
            failure_reasons.append(
                f" {kernel.__name__} cannot implement due to: {failure_reason}"
            )

    # -------------------------------------------------------------
    # Step 4. 如果所有候选 kernel 都失败，则抛出汇总错误
    # -------------------------------------------------------------
    # 报错里会包含所有失败原因，便于定位：
    # - 是设备算力不够
    # - 是被环境变量禁用了
    # - 还是 shape / group_size / dtype / g_idx 不满足约束
    raise ValueError(
        "Failed to find a kernel that can implement the "
        "WNA16 linear layer. Reasons: \n" + "\n".join(failure_reasons)
    )


__all__ = [
    "init_fp8_linear_kernel",
    "init_int8_linear_kernel",
    "choose_mp_linear_kernel",
    "FP8ScaledMMLinearKernel",
    "Int8ScaledMMLinearKernel",
    "ScaledMMLinearKernel",
    "FP8ScaledMMLinearLayerConfig",
    "Int8ScaledMMLinearLayerConfig",
    "ScaledMMLinearLayerConfig",
    "AiterInt8ScaledMMLinearKernel",
    "CPUInt8ScaledMMLinearKernel",
    "CutlassFP8ScaledMMLinearKernel",
    "CutlassInt8ScaledMMLinearKernel",
    "FlashInferFP8ScaledMMLinearKernel",
    "ChannelWiseTorchFP8ScaledMMLinearKernel",
    "PerTensorTorchFP8ScaledMMLinearKernel",
    "RowWiseTorchFP8ScaledMMLinearKernel",
    "ROCmFP8ScaledMMLinearKernel",
    "TritonInt8ScaledMMLinearKernel",
    "MPLinearKernel",
    "MPLinearLayerConfig",
    "AllSparkLinearKernel",
    "ConchLinearKernel",
    "CPUWNA16LinearKernel",
    "CutlassW4A8LinearKernel",
    "Dynamic4bitLinearKernel",
    "ExllamaLinearKernel",
    "MacheteLinearKernel",
    "MarlinLinearKernel",
    "XPUwNa16LinearKernel",
]
