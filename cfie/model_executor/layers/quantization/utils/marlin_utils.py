# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import numpy
import torch

import cfie.envs as envs
from cfie import _custom_ops as ops
from cfie.logger import init_logger
from cfie.model_executor.layers.linear import LinearBase
from cfie.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from cfie.model_executor.layers.quantization.utils.int8_utils import (
    per_token_quant_int8,
)
from cfie.model_executor.layers.quantization.utils.quant_utils import GroupShape
from cfie.platforms import current_platform
from cfie.scalar_type import ScalarType, scalar_types
from cfie.utils.platform_utils import num_compute_units

from .quant_utils import pack_cols, unpack_cols

logger = init_logger(__name__)

GPTQ_MARLIN_TILE = 16
GPTQ_MARLIN_MIN_THREAD_N = 64
GPTQ_MARLIN_MIN_THREAD_K = 128
GPTQ_MARLIN_MAX_PARALLEL = 16

MARLIN_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]

# In case there is a performance issue with Marlin, the variable below can be
# changed to False, which allows Marlin to perform global reductions in fp16
# precision (instead of fp32), and therefore, save on some memory movements.
USE_FP32_REDUCE_DEFAULT = True


# For binary size and compile time, we don't support the same types for with and
#  without runtime zero-point. We support common cases, i.e. AWQ and GPTQ.
#  TODO: we may want to move this into the C++ so its closer to the actual impl
def query_marlin_supported_quant_types(
        has_zp: bool | None = None,
        include_fp_type: bool = True,
        device_capability: int | None = None,
):
    if current_platform.is_cpu():
        return _query_cpu_marlin_supported_quant_types(has_zp, include_fp_type)

    if device_capability is None:
        capability_tuple = current_platform.get_device_capability()
        device_capability = (
            -1 if capability_tuple is None else capability_tuple.to_int()
        )

    if device_capability < 75:
        return []

    # - has_zp is True: return quant_types that has zero points
    # - has_zp is False: return quant_types that has not zero points
    # - has_zp is None: both
    if has_zp is None:
        types0 = query_marlin_supported_quant_types(
            False, include_fp_type, device_capability
        )
        types1 = query_marlin_supported_quant_types(
            True, include_fp_type, device_capability
        )
        return types0 + types1

    if has_zp:
        # AWQ style, unsigned + runtime zero-point
        return [scalar_types.uint4]
    else:
        # GPTQ style, unsigned + symmetric bias
        res = [scalar_types.uint4b8, scalar_types.uint8b128]
        if include_fp_type:
            res += [scalar_types.float8_e4m3fn, scalar_types.float4_e2m1f]
        return res


def _query_cpu_marlin_supported_quant_types(
        has_zp: bool | None = None,
        include_fp_type: bool = True,
):
    # - has_zp is True: return quant_types that has zero points
    # - has_zp is False: return quant_types that has not zero points
    # - has_zp is None: both
    if has_zp is None:
        types0 = _query_cpu_marlin_supported_quant_types(
            False,
            include_fp_type,
        )
        types1 = _query_cpu_marlin_supported_quant_types(
            True,
            include_fp_type,
        )
        return types0 + types1

    if has_zp:
        # AWQ style, unsigned + runtime zero-point
        return [scalar_types.uint4]
    else:
        # GPTQ style, unsigned + symmetric bias, only supports 4-bits for now
        res = [scalar_types.uint4b8]
        return res


def _check_marlin_supported(
        quant_type: ScalarType,
        group_size: int | None,
        has_zp: bool,
        device_capability: int | None = None,
) -> tuple[bool, str | None]:
    if device_capability is None:
        capability_tuple = current_platform.get_device_capability()
        device_capability = (
            -1 if capability_tuple is None else capability_tuple.to_int()
        )

    supported_types = query_marlin_supported_quant_types(
        has_zp, True, device_capability
    )

    if quant_type not in supported_types:
        return (
            False,
            f"Marlin does not support weight_bits = {quant_type}. "
            f"Only types = {supported_types} "
            f"are supported (for group_size = {group_size}, "
            f"device_capability = {device_capability}, zp = {has_zp}).",
        )
    if group_size is None or group_size not in MARLIN_SUPPORTED_GROUP_SIZES:
        return (
            False,
            f"Marlin does not support group_size = {group_size}. "
            f"Only group_sizes = {MARLIN_SUPPORTED_GROUP_SIZES} "
            "are supported.",
        )

    return True, None


def check_marlin_supported(
        quant_type: ScalarType,
        group_size: int,
        has_zp: bool = False,
        device_capability: int | None = None,
) -> bool:
    cond, _ = _check_marlin_supported(quant_type, group_size, has_zp, device_capability)
    return cond


def verify_marlin_supported(
        quant_type: ScalarType, group_size: int, has_zp: bool = False
) -> None:
    cond, err_msg = _check_marlin_supported(quant_type, group_size, has_zp)
    if not cond:
        assert err_msg is not None
        raise ValueError(err_msg)


def verify_marlin_supports_shape(
        output_size_per_partition: int,
        input_size_per_partition: int,
        input_size: int,
        group_size: int,
) -> None:
    # Validate output_size_per_partition
    if output_size_per_partition % GPTQ_MARLIN_MIN_THREAD_N != 0:
        raise ValueError(
            f"Weight output_size_per_partition = "
            f"{output_size_per_partition} is not divisible by "
            f" min_thread_n = {GPTQ_MARLIN_MIN_THREAD_N}. "
            "Consider reducing tensor_parallel_size or running "
            "with --quantization gptq."
        )

    # Validate input_size_per_partition
    if input_size_per_partition % GPTQ_MARLIN_MIN_THREAD_K != 0:
        raise ValueError(
            f"Weight input_size_per_partition = "
            f"{input_size_per_partition} is not divisible "
            f"by min_thread_k = {GPTQ_MARLIN_MIN_THREAD_K}. "
            "Consider reducing tensor_parallel_size or running "
            "with --quantization gptq."
        )

    if group_size < input_size and input_size_per_partition % group_size != 0:
        raise ValueError(
            f"Weight input_size_per_partition = {input_size_per_partition}"
            f" is not divisible by group_size = {group_size}. "
            "Consider reducing tensor_parallel_size or running "
            "with --quantization gptq."
        )


def check_marlin_supports_shape(
        output_size_per_partition: int,
        input_size_per_partition: int,
        input_size: int,
        group_size: int,
) -> tuple[bool, str | None]:
    try:
        verify_marlin_supports_shape(
            output_size_per_partition, input_size_per_partition, input_size, group_size
        )
    except ValueError as e:
        return False, e.__str__()
    return True, None


def check_marlin_supports_layer(layer: LinearBase, group_size: int) -> bool:
    if current_platform.is_rocm():
        return False
    output_size_per_partition = (
            getattr(layer, "output_size_per_partition", None) or layer.output_size
    )
    input_size_per_partition = (
            getattr(layer, "input_size_per_partition", None) or layer.input_size
    )

    return check_marlin_supports_shape(
        output_size_per_partition=output_size_per_partition,
        input_size_per_partition=input_size_per_partition,
        input_size=layer.input_size,
        group_size=group_size,
    )[0]


def check_moe_marlin_supports_layer(layer: LinearBase, group_size: int) -> bool:
    # ROCm 路径当前不支持 MoE Marlin kernel，因此直接返回 False。
    if current_platform.is_rocm():
        return False

    # 读取当前 MoE 层的 hidden size 与每个分片上的 intermediate size。
    hidden_size = layer.hidden_size
    intermediate_size_per_partition = layer.intermediate_size_per_partition

    # MoE Marlin 当前不支持把 router weight 直接乘到输入上的执行路径。
    # 一旦该层启用了这类行为，就不能复用当前 MoE Marlin kernel。
    supports_router_weight = not layer.apply_router_weight_on_input

    # -------------------- 检查 MoE Marlin 的 shape 对齐约束 --------------------
    # 对 MoE 来说主要有两类线性变换：
    # - gate-up: (n, k) = (intermediate_size_per_partition * 2, hidden_size)
    # - down:    (n, k) = (hidden_size, intermediate_size_per_partition)
    #
    # 当前 MoE Marlin kernel 要求：
    # - n 维满足 128 对齐
    # - k 维至少满足 64 对齐，同时还要兼容量化 group_size 的整除要求
    #
    # 这里用：
    # - hidden_size % 128 == 0
    # - intermediate_size_per_partition % max(64, group_size) == 0
    # 来统一覆盖这组约束。
    supports_shape = (
            hidden_size % 128 == 0
            and intermediate_size_per_partition % max(64, group_size) == 0
    )

    # -------------------- 检查 group_size 是否落在支持集合内 --------------------
    # 当前 MoE Marlin 只支持固定的少数组大小。
    supports_group_size = group_size in [-1, 32, 64, 128]

    # 只有同时满足 shape、group_size 和执行语义约束时，当前层才可走 MoE Marlin。
    return supports_shape and supports_group_size and supports_router_weight


def marlin_moe_intermediate_size(w1_packed: torch.Tensor, w2_packed: torch.Tensor):
    """
    Given Marlin packed weight matrices w1_packed, and w2_packed,
    return the MoE intermediate size N
    """
    marlin_tile_size = 16
    return w2_packed.size(1) * marlin_tile_size


def marlin_make_workspace_new(
    device: torch.device, max_blocks_per_sm: int = 1
) -> torch.Tensor:
    """Allocate workspace for Marlin kernels.

    The workspace length equals the number of launched thread blocks:
    ``num_sms * max_blocks_per_sm``.
    """
    sms = num_compute_units(device.index)
    return torch.zeros(
        sms * max_blocks_per_sm, dtype=torch.int, device=device, requires_grad=False
    )


def marlin_is_k_full(act_order: bool, is_row_parallel: bool) -> bool:
    return (not act_order) or (act_order and not is_row_parallel)


def marlin_repeat_scales_on_all_ranks(
        act_order: bool,  # 是否启用 activation-order / desc_act
        group_size: int,  # 量化 group size；-1 通常表示 channelwise
        is_row_parallel: bool  # 当前层是否是 RowParallelLinear（输入维被切分）
) -> bool:
    # 如果 group_size == -1，表示不是普通 group-wise quant，
    # 而是 channelwise 这种特殊情况
    is_channelwise = group_size == -1

    # 返回 True 的两种情况：
    #
    # 1) act_order=True
    #    只要启用了 activation-order / desc_act，
    #    就要求 scales 在每个 rank 上都重复保存一份
    #
    # 2) channelwise + row parallel
    #    当是 channelwise 量化，并且当前层还是 RowParallelLinear 时，
    #    也要求每个 rank 上都有完整 scales
    return act_order or (is_channelwise and is_row_parallel)


def marlin_make_empty_g_idx(device: torch.device) -> torch.Tensor:
    return torch.nn.Parameter(
        torch.empty(0, dtype=torch.int, device=device), requires_grad=False
    )


def marlin_sort_g_idx(g_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort act-order group ids and return the sort permutation.

    Args:
        g_idx: Group id per local K position, shape ``[K]``.

    Returns:
        A tuple ``(sorted_g_idx, sort_indices)`` where
        ``sorted_g_idx = g_idx[sort_indices]``.
    """
    g_idx_sort_indices = torch.argsort(g_idx).to(torch.int)
    return g_idx[g_idx_sort_indices], g_idx_sort_indices


def get_scale_perms():
    """Build scale permutations used by Marlin layouts.

    Returns:
        A tuple ``(scale_perm, scale_perm_single)`` where:
        - ``scale_perm`` is used for grouped scales.
        - ``scale_perm_single`` is used for single-group/channelwise layouts
          and 8-bit-activation paths.
    """
    scale_perm: list[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single: list[int] = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


def marlin_permute_scales(
        s: torch.Tensor,
        size_k: int,
        size_n: int,
        group_size: int,
        is_a_8bit: bool = False
) -> torch.Tensor:
    """Permute scales into the memory layout expected by Marlin kernels.

    For grouped quantization (``group_size < size_k`` and ``group_size != -1``)
    with non-8-bit activations, use the grouped permutation. Otherwise use the
    single-group/channelwise permutation.
    """
    scale_perm, scale_perm_single = get_scale_perms()
    if group_size < size_k and group_size != -1 and not is_a_8bit:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    s = s.reshape((-1, size_n)).contiguous()
    return s


def marlin_permute_bias(s: torch.Tensor) -> torch.Tensor:
    origin_shape = s.shape
    _, scale_perm_single = get_scale_perms()
    s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    return s.reshape(*origin_shape).contiguous()


def marlin_act_int8_process_scales(s: torch.Tensor):
    a_scales_scale_factor = 1 / 4096 * s.max().float()
    s = s / s.max() * 4096
    s = s.round().to(torch.int16).view(s.dtype)
    return s, a_scales_scale_factor


def marlin_moe_permute_scales(
        s: torch.Tensor,
        size_k: int,
        size_n: int,
        group_size: int,
        is_a_8bit: bool = False
):
    # -------------------------------------------------------------
    # 1) MoE 版本的输入语义
    # -------------------------------------------------------------
    # s 的形状通常为：
    #   [num_experts, num_groups_or_rows, size_n]
    #
    # 与普通线性层相比，只是在最前面多了一维 expert 维。
    # 每个 expert 的 scales 都需要各自独立地转成 Marlin 布局。
    num_experts = s.shape[0]

    # -------------------------------------------------------------
    # 2) 预分配输出
    # -------------------------------------------------------------
    # scale permutation 不改变每个 expert 的张量 shape，
    # 因此这里直接按输入形状创建输出缓冲区。
    output = torch.empty(
        (num_experts, s.shape[1], s.shape[2]),
        device=s.device,
        dtype=s.dtype,
    )

    # -------------------------------------------------------------
    # 3) 逐 expert 调用普通的 scale 重排逻辑
    # -------------------------------------------------------------
    # 这里没有做跨 expert 的混排；
    # 它只是一个“沿 expert 维循环调用 marlin_permute_scales(...)”的薄封装。
    for e in range(num_experts):
        output[e] = marlin_permute_scales(s[e], size_k, size_n, group_size, is_a_8bit)
    return output


def marlin_zero_points(
        zp: torch.Tensor,
        size_k: int,
        size_n: int,
        num_bits: int,
        is_a_8bit: bool = False
) -> torch.Tensor:
    # -------------------------------------------------------------
    # 1) 先按和 scales 类似的方式，对 zero-points 做一轮 permutation
    # -------------------------------------------------------------
    # 注释里这句很关键：
    #   "Permute zero-points in a similar way to scales, but do not use the
    #    'single' permutation, since zero-points are applied on every MMA"
    #
    # 含义：
    # - zp 的布局也要和 Marlin 的 MMA / tile 访问方式对齐
    # - 但它不能完全照搬 scales 的所有 permutation 逻辑
    # - 因为 zero-point 在运行时是“每次 MMA 都要用”的，
    #   所以它采用的是更适合内核每次取 zp 的排列方式
    #
    # get_scale_perms() 返回两套 permutation；
    # 这里取的是第一套 scale_perm，而不是第二套。
    scale_perm, _ = get_scale_perms()

    # 这里：
    #   zp.reshape((-1, len(scale_perm)))
    # 是把 zp 拉成很多行、每行长度等于 len(scale_perm) 的块
    #
    # 然后：
    #   [:, scale_perm]
    # 就是在每个小块内部，按 scale_perm 重新排列列顺序
    #
    # 也就是说：
    # - 不是对整个大矩阵随便 permute
    # - 而是按一个固定 block 大小，对 block 内部做重排
    zp = zp.reshape((-1, len(scale_perm)))[:, scale_perm]

    # -------------------------------------------------------------
    # 2) 为后续 dequant 代码做“列交错(interleave)”
    # -------------------------------------------------------------
    # 这一步是 Marlin 特有的物理布局要求。
    #
    # 目的：
    # - 让 zero-points 的列顺序更适合后续 dequant / MMA kernel 读取
    # - 然后再重新 pack 回 int32
    #
    # 不同 bit-width 的 interleave 顺序不同：
    #
    # num_bits == 4:
    #   一组 8 个元素，重排为 [0,2,4,6,1,3,5,7]
    #
    # num_bits == 8:
    #   一组 4 个元素，重排为 [0,2,1,3]
    #
    # 这和前面你在 gptq_marlin_repack(...) 里看到的 pack_idx 风格很像，
    # 本质上都是为了匹配 Marlin 内核的读取/拼 fragment 方式。
    if num_bits == 4:
        interleave = numpy.array([0, 2, 4, 6, 1, 3, 5, 7])
    elif num_bits == 8:
        interleave = numpy.array([0, 2, 1, 3])
    else:
        raise Exception("num_bits must be 4 or 8, got {}".format(num_bits))

    # -------------------------------------------------------------
    # 3) 对非 8bit A 路径，再做一轮列交错
    # -------------------------------------------------------------
    # 如果不是 A=8bit 的特殊路径，
    # 那么 zp 还需要进一步按 interleave 顺序重排。
    #
    # 这里的处理逻辑是：
    #   - 先按 interleave 的长度切块
    #   - 每个块内部按 interleave 重排
    #   - 再 flatten 回一维
    #
    # 例如 num_bits=4 时，interleave 长度是 8：
    #   每 8 个 zp 值作为一个小块重排
    #
    # 为什么 A=8bit 时跳过？
    # 因为 A=8bit（比如 fp8/int8）路径下，Marlin 对 B/zp 的组织方式不同，
    # 这条普通 interleave 规则不适用。
    if not is_a_8bit:
        zp = zp.reshape((-1, len(interleave)))[:, interleave].ravel()

    # -------------------------------------------------------------
    # 4) 重新整理回二维形状
    # -------------------------------------------------------------
    # 最终重新拉回成：
    #   [size_k, size_n]
    #
    # 在这里：
    # - size_k 往往是 grouped_k（也就是 K 维 group 数）
    # - size_n 是当前本地输出列 N_local
    #
    # 也就是说，这一步之后 zp 逻辑上还是“每个 group × 每个输出列”一张表，
    # 只是内部元素顺序已经被重新组织过了。
    zp = zp.reshape((-1, size_n)).contiguous()

    # -------------------------------------------------------------
    # 5) 再按列方向重新 pack 回 int32
    # -------------------------------------------------------------
    # 前面 zp 是“解包后的普通 int32 矩阵”：
    #   shape = [size_k, size_n]
    #
    # 这里 pack_cols(...) 会把它重新打包成：
    #   [size_k, size_n / pack_factor]
    #
    # 其中：
    # - 4bit -> pack_factor = 8
    # - 8bit -> pack_factor = 4
    #
    # 这一步完成后，zp 就变成 Marlin kernel 真正运行时想吃的 packed 格式了。
    zp = pack_cols(zp, num_bits, size_k, size_n)

    return zp


def awq_to_marlin_zero_points(
        q_zp_packed: torch.Tensor,
        size_k: int,
        size_n: int,
        num_bits: int,
        is_a_8bit: bool = False,
) -> torch.Tensor:
    # AWQ zero-points are quantized and packed on the column dim.
    # In addition, the values are permuted based on dequantizer.
    # Here we undo both of these, and then apply marlin permutation
    # and pack it back.
    q_zp = unpack_cols(q_zp_packed, num_bits, size_k, size_n)

    # Undo interleaving (use argsort(..) to get inverse perm)
    if num_bits == 4:
        undo_interleave = numpy.argsort(numpy.array([0, 2, 4, 6, 1, 3, 5, 7]))
    elif num_bits == 8:
        undo_interleave = numpy.argsort(numpy.array([0, 2, 1, 3]))
    else:
        raise Exception("num_bits must be 4 or 8, got {}".format(num_bits))

    q_zp = q_zp.reshape((-1, len(undo_interleave)))[:, undo_interleave].ravel()
    q_zp = q_zp.reshape((-1, size_n)).contiguous()

    marlin_zp = marlin_zero_points(q_zp, size_k, size_n, num_bits, is_a_8bit)
    return marlin_zp


def moe_awq_to_marlin_zero_points(
        q_zp_packed: torch.Tensor,
        size_k: int,
        size_n: int,
        num_bits: int,
        is_a_8bit: bool = False,
):
    num_experts = q_zp_packed.shape[0]
    output = torch.empty(
        (num_experts, q_zp_packed.shape[1], q_zp_packed.shape[2]),
        device=q_zp_packed.device,
        dtype=q_zp_packed.dtype,
    )
    for e in range(num_experts):
        output[e] = awq_to_marlin_zero_points(
            q_zp_packed[e], size_k, size_n, num_bits, is_a_8bit
        )
    return output


def maybe_warn_marlin_atomic_add(device, dtype):
    if torch.compiler.is_dynamo_compiling():
        return
    device_capability = torch.cuda.get_device_capability(device)
    if device_capability[0] < 9 and dtype == torch.bfloat16:
        logger.info_once(
            "You are running Marlin kernel with bf16 on GPUs before SM90. "
            "You can consider change to fp16 to achieve better performance "
            "if possible."
        )


def maybe_warn_marlin_atomic_add_env():
    if torch.compiler.is_dynamo_compiling():
        return
    if envs.VLLM_MARLIN_USE_ATOMIC_ADD:
        return
    logger.info_once(
        "Marlin kernel can achieve better performance for small size_n "
        "with experimental use_atomic_add feature. "
        "You can consider set environment variable "
        "VLLM_MARLIN_USE_ATOMIC_ADD to 1 if possible."
    )


def should_use_atomic_add_reduce(
        m: int, n: int, k: int, device: torch.device, dtype: torch.dtype
) -> bool:
    # atomicAdd reduce 只在输出列较小且 K 维较大时可能快于全局 reduce。
    if n >= 2048 or k < 2048 or device.type != "cuda":
        # 非 CUDA、输出列过大或 K 维不足时，直接保持默认全局 reduce 路径。
        return False

    # atomicAdd reduce 默认关闭，需要显式设置 VLLM_MARLIN_USE_ATOMIC_ADD=1。
    if not envs.VLLM_MARLIN_USE_ATOMIC_ADD:
        # 未开启实验开关时输出一次提示，并避免改变默认执行路径。
        maybe_warn_marlin_atomic_add_env()
        # 返回 False 让调用方继续使用普通 Marlin reduce。
        return False

    # 读取当前 GPU 架构，用于判断 BF16 atomicAdd 是否有原生支持。
    device_capability = torch.cuda.get_device_capability(device)
    # SM90 之前没有原生 BF16 atomicAdd，BF16 输入不能启用该优化。
    if device_capability[0] < 9 and dtype == torch.bfloat16:
        # 对不支持的架构给出性能提示，避免用户误以为开关未生效。
        maybe_warn_marlin_atomic_add(device, dtype)
        # 回退到全局 reduce，保证 BF16 路径兼容 SM8x。
        return False

    # 通过规模、环境变量和架构检查后，允许 Marlin 使用 atomicAdd reduce。
    return True


_quant_fp8_method: QuantFP8 | None = None


def get__quant_fp8_method() -> QuantFP8:
    global _quant_fp8_method
    if _quant_fp8_method is None:
        _quant_fp8_method = QuantFP8(False, GroupShape.PER_TOKEN)
    return _quant_fp8_method


def get_marlin_input_dtype(prefix: str | None = None):
    if envs.VLLM_MARLIN_INPUT_DTYPE is None:
        return
    elif envs.VLLM_MARLIN_INPUT_DTYPE.lower() == "int8":
        return torch.int8
    elif envs.VLLM_MARLIN_INPUT_DTYPE.lower() == "fp8":
        if not current_platform.is_device_capability(
                89
        ) and not current_platform.is_device_capability(120):
            raise ValueError(
                "Marlin W4A8-FP8 only support SM89 or SM120 device "
                "(It is slower than Marlin W4A16 on other devices). "
                "You can consider using W4A8-INT8 instead"
                "(set VLLM_MARLIN_INPUT_DTYPE=int8)."
            )

        _ = get__quant_fp8_method()
        return torch.float8_e4m3fn
    else:
        return


def marlin_quant_input(x: torch.Tensor, quant_dtype: torch.dtype):
    x = x.reshape(-1, x.shape[-1])
    if quant_dtype == torch.int8:
        return per_token_quant_int8(x)
    elif quant_dtype == torch.float8_e4m3fn:
        return get__quant_fp8_method()(x)
    else:
        raise ValueError(f"unsupported quant_dtype {quant_dtype}")


def apply_gptq_marlin_linear(
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_zp: torch.Tensor,
        g_idx: torch.Tensor,
        g_idx_sort_indices: torch.Tensor,
        workspace: torch.Tensor,
        wtype: ScalarType,
        output_size_per_partition: int,
        input_size_per_partition: int,
        is_k_full: bool,
        input_global_scale: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT,
        input_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    reshaped_x = input.reshape(-1, input.shape[-1])
    out_shape = input.shape[:-1] + (output_size_per_partition,)

    use_atomic_add = should_use_atomic_add_reduce(
        m=reshaped_x.size(0),
        n=output_size_per_partition,
        k=reshaped_x.size(1),
        device=input.device,
        dtype=input.dtype,
    )

    a_scales = None
    if input_dtype == torch.int8:
        assert wtype == scalar_types.uint4b8, (
            "W8A8-INT8 is not supported by marlin kernel."
        )
        reshaped_x, a_scales = marlin_quant_input(reshaped_x, input_dtype)
        a_scales = a_scales * input_global_scale
    elif input_dtype == torch.float8_e4m3fn:
        assert wtype == scalar_types.uint4b8, (
            "INT8 weight + FP8 activation is not supported."
        )

        reshaped_x, a_scales = marlin_quant_input(reshaped_x, input_dtype)

    output = ops.marlin_gemm(
        reshaped_x,
        None,
        weight,
        bias,
        weight_scale,
        a_scales,
        None,
        weight_zp,
        g_idx,
        g_idx_sort_indices,
        workspace,
        wtype,
        size_m=reshaped_x.shape[0],
        size_n=output_size_per_partition,
        size_k=input_size_per_partition,
        is_k_full=is_k_full,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=use_fp32_reduce,
        is_zp_float=False,
    )

    return output.reshape(out_shape)


def apply_awq_marlin_linear(
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_zp: torch.Tensor,
        g_idx: torch.Tensor,
        g_idx_sort_indices: torch.Tensor,
        workspace: torch.Tensor,
        quant_type: ScalarType,
        output_size_per_partition: int,
        input_size_per_partition: int,
        input_global_scale: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT,
        input_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    reshaped_x = input.reshape(-1, input.shape[-1])
    out_shape = input.shape[:-1] + (output_size_per_partition,)

    use_atomic_add = should_use_atomic_add_reduce(
        m=reshaped_x.size(0),
        n=output_size_per_partition,
        k=reshaped_x.size(1),
        device=input.device,
        dtype=input.dtype,
    )

    a_scales = None
    if input_dtype == torch.int8:
        assert quant_type == scalar_types.uint4, (
            "W8A8-INT8 is not supported by marlin kernel."
        )
        reshaped_x, a_scales = marlin_quant_input(reshaped_x, input_dtype)
        a_scales = a_scales * input_global_scale
    elif input_dtype == torch.float8_e4m3fn:
        assert quant_type == scalar_types.uint4, (
            "INT8 weight + FP8 activation is not supported."
        )
        reshaped_x, a_scales = marlin_quant_input(reshaped_x, input_dtype)

    output = ops.marlin_gemm(
        reshaped_x,
        None,
        weight,
        bias,
        weight_scale,
        a_scales,
        None,
        weight_zp,
        g_idx,
        g_idx_sort_indices,
        workspace,
        quant_type,
        size_m=reshaped_x.shape[0],
        size_n=output_size_per_partition,
        size_k=input_size_per_partition,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=use_fp32_reduce,
        is_zp_float=False,
    )

    return output.reshape(out_shape)
