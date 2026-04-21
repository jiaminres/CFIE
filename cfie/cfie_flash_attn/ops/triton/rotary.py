# Copy from https://github.com/cfie-project/flash-attention/blob/main/flash_attn/ops/triton/rotary.py
# Copyright (c) 2023, Tri Dao.

from typing import Optional, Union

import torch

from cfie.logger import init_logger
from cfie.triton_utils import HAS_TRITON, tl, triton

logger = init_logger(__name__)


@triton.jit
def rotary_kernel(
        OUT,  # 输出张量指针
        X,  # 输入张量指针
        COS,  # cos 表指针
        SIN,  # sin 表指针
        CU_SEQLENS,  # 变长序列边界指针；若不是 varlen，可忽略
        SEQLEN_OFFSETS,  # 序列位置偏移；可能是一个整数，也可能是一个张量指针

        # ----------------- 运行时尺寸参数 -----------------
        seqlen,  # 非 varlen 时的序列长度；varlen 时会在 kernel 内被当前 batch 的真实长度覆盖
        rotary_dim,  # 实际参与 rotary 的维度数
        seqlen_ro,  # cos/sin 表可支持的位置长度

        # ----------------- OUT 的 stride -----------------
        stride_out_batch,  # OUT 沿 batch 维移动一步跨多少元素
        stride_out_seqlen,  # OUT 沿 seqlen 维移动一步跨多少元素
        stride_out_nheads,  # OUT 沿 nheads 维移动一步跨多少元素
        stride_out_headdim,  # OUT 沿 headdim 维移动一步跨多少元素

        # ----------------- X 的 stride -----------------
        stride_x_batch,  # X 沿 batch 维移动一步跨多少元素
        stride_x_seqlen,  # X 沿 seqlen 维移动一步跨多少元素
        stride_x_nheads,  # X 沿 nheads 维移动一步跨多少元素
        stride_x_headdim,  # X 沿 headdim 维移动一步跨多少元素

        # ----------------- Triton 编译期元参数 -----------------
        BLOCK_K: tl.constexpr,  # 每个程序实例在 head_dim 方向处理多少列
        IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr,  # seqlen_offsets 是否是张量
        IS_VARLEN: tl.constexpr,  # 是否是变长输入
        INTERLEAVED: tl.constexpr,  # True: GPT-J 风格；False: NeoX 风格
        CONJUGATE: tl.constexpr,  # 是否做共轭旋转；backward 时会用到
        BLOCK_M: tl.constexpr,  # 每个程序实例在 seqlen 方向处理多少个 token
):
    # Triton grid 的三个 program_id：
    # axis=0 -> 当前处理的是第几个 seqlen block
    # axis=1 -> 当前处理的是第几个 head
    # axis=2 -> 当前处理的是第几个 batch 样本
    pid_m = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)
    pid_batch = tl.program_id(axis=2)

    # rotary_dim 的一半
    # 因为 RoPE 的 cos/sin 表只存 half dim，真正旋转时两两配对
    rotary_dim_half = rotary_dim // 2

    # ----------------- 先把 X / OUT 指针移动到当前 batch、当前 head 的起点 -----------------
    if not IS_VARLEN:
        # 非变长：
        # x 形状逻辑上是 [batch, seqlen, nheads, headdim]
        #
        # 先跳到当前 batch 的起始位置，再跳到当前 head 的起始位置
        X = X + pid_batch * stride_x_batch + pid_head * stride_x_nheads
        OUT = OUT + pid_batch * stride_out_batch + pid_head * stride_out_nheads
    else:
        # 变长：
        # x 逻辑上是 [total_seqlen, nheads, headdim]
        # 需要先通过 cu_seqlens 找到当前 batch 样本在拼接大张量中的起点和终点
        start_idx = tl.load(CU_SEQLENS + pid_batch)
        # 当前序列真实长度 = 下一个边界 - 当前边界
        seqlen = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx

        # X / OUT 跳到当前 batch 对应序列在 total_seqlen 中的起点，再跳到当前 head
        X = X + start_idx * stride_x_seqlen + pid_head * stride_x_nheads
        OUT = OUT + start_idx * stride_out_seqlen + pid_head * stride_out_nheads

    # 如果当前这个 seqlen block 完全落在序列长度之外，直接返回
    if pid_m * BLOCK_M >= seqlen:
        return

    # 当前程序实例负责的 token 行号
    # 形状: [BLOCK_M]
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # 当前 token 在 cos/sin 表中的真实位置索引
    if not IS_SEQLEN_OFFSETS_TENSOR:
        # seqlen_offsets 是单个整数，所有 batch 共用同一个偏移
        rm_cs = rm + SEQLEN_OFFSETS
    else:
        # seqlen_offsets 是 [batch] 张量，每条序列有自己的偏移
        rm_cs = rm + tl.load(SEQLEN_OFFSETS + pid_batch)

    # 当前程序实例在 head_dim 方向负责的列号
    # 形状: [BLOCK_K]
    rk = tl.arange(0, BLOCK_K)

    # 非 interleaved 情况下，实际只需要处理 half dim
    # 形状: [BLOCK_K // 2]
    rk_half = tl.arange(0, BLOCK_K // 2)

    if not INTERLEAVED:
        # ============================================================
        # NeoX 风格：
        # 把前半维和后半维配对
        # 例如:
        # [x0, x1, x2, x3, x4, x5, x6, x7]
        # 配对为:
        # (x0, x4), (x1, x5), (x2, x6), (x3, x7)
        # ============================================================

        # 让 X 指针移动到当前 token block、当前 head 的“前半维”起点
        # 最终 X 对应的访问网格形状是 [BLOCK_M, BLOCK_K // 2]
        X = X + (rm[:, None] * stride_x_seqlen + rk_half[None, :] * stride_x_headdim)

        # COS / SIN 指针移动到当前 token block、当前 rotary half dim 的位置
        # COS / SIN 的逻辑形状是 [seqlen_ro, rotary_dim_half]
        COS = COS + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])
        SIN = SIN + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])

        # 读取 cos
        # mask 条件：
        # 1. 真实位置 rm_cs 不能越过 cos/sin 表长度 seqlen_ro
        # 2. rk_half 不能越过 rotary_dim_half
        # 越界时 cos 默认填 1.0（相当于不旋转）
        cos = tl.load(
            COS,
            mask=(rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < rotary_dim_half),
            other=1.0,
        ).to(tl.float32)
        # cos 形状: [BLOCK_M, BLOCK_K // 2]

        # 读取 sin
        # 越界时 sin 默认填 0.0
        sin = tl.load(
            SIN,
            mask=(rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < rotary_dim_half),
            other=0.0,
        ).to(tl.float32)
        # sin 形状: [BLOCK_M, BLOCK_K // 2]

        # 读取前半部分 x0
        x0 = tl.load(
            X,
            mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
            other=0.0,
        ).to(tl.float32)
        # x0 形状: [BLOCK_M, BLOCK_K // 2]

        # 读取后半部分 x1
        # 相对于前半部分，在 headdim 方向整体偏移 rotary_dim_half
        x1 = tl.load(
            X + rotary_dim_half * stride_x_headdim,
            mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
            other=0.0,
        ).to(tl.float32)
        # x1 形状: [BLOCK_M, BLOCK_K // 2]

        # backward 时做逆旋转，所以把 sin 取反
        if CONJUGATE:
            sin = -sin

        # 旋转公式
        # o0 = x0 * cos - x1 * sin
        # o1 = x0 * sin + x1 * cos
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos
        # o0 / o1 形状: [BLOCK_M, BLOCK_K // 2]

        # 把 OUT 指针移动到当前 token block、前半维起点
        OUT = OUT + (rm[:, None] * stride_out_seqlen + rk_half[None, :] * stride_out_headdim)

        # 写回前半部分
        tl.store(
            OUT,
            o0,
            mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
        )

        # 写回后半部分
        tl.store(
            OUT + rotary_dim_half * stride_out_headdim,
            o1,
            mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
        )
    else:
        # ============================================================
        # GPT-J 风格：
        # 按偶数维 / 奇数维交错配对
        # 例如:
        # [x0, x1, x2, x3, x4, x5, x6, x7]
        # 配对为:
        # (x0, x1), (x2, x3), (x4, x5), (x6, x7)
        # ============================================================

        # 这里作者的技巧是：
        # 不直接分别取偶数列和奇数列，因为那样两个都可能访问不连续，效率差
        # 而是：
        # x0 读取自然顺序 [0,1,2,3,4,5,...]
        # x1 读取交换顺序 [1,0,3,2,5,4,...]
        # 再配合 cos/sin 的重复索引与 tl.where 组合得到最终结果

        # 构造交换后的列索引：
        # 0->1, 1->0, 2->3, 3->2, ...
        rk_swap = rk + ((rk + 1) % 2) * 2 - 1
        # 例如:
        # rk      = [0,1,2,3,4,5]
        # rk_swap = [1,0,3,2,5,4]

        # 构造重复索引：
        # 0,0,1,1,2,2,...
        rk_repeat = tl.arange(0, BLOCK_K) // 2

        # X0 读取原始顺序列
        X0 = X + (rm[:, None] * stride_x_seqlen + rk[None, :] * stride_x_headdim)

        # X1 读取交换顺序列
        X1 = X + (rm[:, None] * stride_x_seqlen + rk_swap[None, :] * stride_x_headdim)

        # COS / SIN 对每对维度复用同一组 cos/sin，因此用 rk_repeat
        COS = COS + (rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :])
        SIN = SIN + (rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :])

        # 读取 cos
        cos = tl.load(
            COS,
            mask=(rm_cs[:, None] < seqlen_ro) & (rk_repeat[None, :] < rotary_dim_half),
            other=1.0,
        ).to(tl.float32)
        # cos 形状: [BLOCK_M, BLOCK_K]

        # 读取 sin
        sin = tl.load(
            SIN,
            mask=(rm_cs[:, None] < seqlen_ro) & (rk_repeat[None, :] < rotary_dim_half),
            other=0.0,
        ).to(tl.float32)
        # sin 形状: [BLOCK_M, BLOCK_K]

        # 读取原顺序 x0
        x0 = tl.load(
            X0,
            mask=(rm[:, None] < seqlen) & (rk[None, :] < rotary_dim),
            other=0.0,
        ).to(tl.float32)
        # x0 形状: [BLOCK_M, BLOCK_K]

        # 读取交换顺序 x1
        x1 = tl.load(
            X1,
            mask=(rm[:, None] < seqlen) & (rk_swap[None, :] < rotary_dim),
            other=0.0,
        ).to(tl.float32)
        # x1 形状: [BLOCK_M, BLOCK_K]

        # backward 时做逆旋转
        if CONJUGATE:
            sin = -sin

        # 先分别算 x0*cos 和 x1*sin
        x0_cos = x0 * cos
        x1_sin = x1 * sin

        # 对偶数位置和奇数位置分别选不同公式：
        # 偶数位: x_even * cos - x_odd * sin
        # 奇数位: x_odd * cos + x_even * sin
        out = tl.where(
            rk[None, :] % 2 == 0,
            x0_cos - x1_sin,
            x0_cos + x1_sin,
        )
        # out 形状: [BLOCK_M, BLOCK_K]

        # 把 OUT 指针移动到当前 token block、当前 rotary dim 起点
        OUT = OUT + (rm[:, None] * stride_out_seqlen + rk[None, :] * stride_out_headdim)

        # 写回结果
        tl.store(
            OUT,
            out,
            mask=(rm[:, None] < seqlen) & (rk[None, :] < rotary_dim),
        )


def _rotate_half_reference(x: torch.Tensor, interleaved: bool) -> torch.Tensor:
    if interleaved:
        x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
        rotated = torch.stack(
            (-x_reshaped[..., 1], x_reshaped[..., 0]),
            dim=-1,
        )
        return rotated.reshape_as(x)

    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _build_rotary_positions(
        x: torch.Tensor,
        *,
        batch: int,
        seqlen: int,
        is_varlen: bool,
        seqlen_offsets: Union[int, torch.Tensor],
        cu_seqlens: Optional[torch.Tensor],
) -> torch.Tensor:
    if not is_varlen:
        base_positions = torch.arange(seqlen, device=x.device, dtype=torch.long)
        if isinstance(seqlen_offsets, torch.Tensor):
            return base_positions.unsqueeze(0) + seqlen_offsets.to(
                device=x.device,
                dtype=torch.long,
            ).unsqueeze(1)
        return base_positions.unsqueeze(0) + int(seqlen_offsets)

    assert cu_seqlens is not None
    positions = torch.empty((x.shape[0],), device=x.device, dtype=torch.long)
    cu_seqlens_cpu = cu_seqlens.to(device="cpu", dtype=torch.long)
    if isinstance(seqlen_offsets, torch.Tensor):
        offsets_cpu = seqlen_offsets.to(device="cpu", dtype=torch.long)
    else:
        offsets_cpu = None

    for batch_idx in range(batch):
        start = int(cu_seqlens_cpu[batch_idx].item())
        end = int(cu_seqlens_cpu[batch_idx + 1].item())
        seq_offset = (
            int(offsets_cpu[batch_idx].item())
            if offsets_cpu is not None
            else int(seqlen_offsets)
        )
        positions[start:end] = torch.arange(
            end - start,
            device=x.device,
            dtype=torch.long,
        ) + seq_offset
    return positions


def _apply_rotary_reference(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        seqlen_offsets: Union[int, torch.Tensor],
        cu_seqlens: Optional[torch.Tensor],
        max_seqlen: Optional[int],
        *,
        interleaved: bool,
        inplace: bool,
        conjugate: bool,
) -> torch.Tensor:
    is_varlen = cu_seqlens is not None
    if not is_varlen:
        batch, seqlen, _, headdim = x.shape
    else:
        assert max_seqlen is not None
        total_seqlen, _, headdim = x.shape
        batch = cu_seqlens.shape[0] - 1
        seqlen = max_seqlen

    seqlen_ro, rotary_half_dim = cos.shape
    rotary_dim = rotary_half_dim * 2
    assert rotary_dim <= headdim
    assert seqlen_ro >= seqlen

    output = x if inplace else x.clone()
    x_rot = x[..., :rotary_dim]

    positions = _build_rotary_positions(
        x,
        batch=batch,
        seqlen=seqlen,
        is_varlen=is_varlen,
        seqlen_offsets=seqlen_offsets,
        cu_seqlens=cu_seqlens,
    )
    flat_positions = positions.reshape(-1)
    cos_pos = cos.to(device=x.device).index_select(0, flat_positions)
    sin_pos = sin.to(device=x.device).index_select(0, flat_positions)
    if conjugate:
        sin_pos = -sin_pos

    if not is_varlen:
        cos_pos = cos_pos.view(batch, seqlen, 1, rotary_half_dim)
        sin_pos = sin_pos.view(batch, seqlen, 1, rotary_half_dim)
    else:
        cos_pos = cos_pos.view(total_seqlen, 1, rotary_half_dim)
        sin_pos = sin_pos.view(total_seqlen, 1, rotary_half_dim)

    if interleaved:
        cos_full = torch.repeat_interleave(cos_pos, 2, dim=-1)
        sin_full = torch.repeat_interleave(sin_pos, 2, dim=-1)
    else:
        cos_full = torch.cat((cos_pos, cos_pos), dim=-1)
        sin_full = torch.cat((sin_pos, sin_pos), dim=-1)

    rotated = (
        x_rot.to(torch.float32) * cos_full.to(torch.float32)
        + _rotate_half_reference(x_rot.to(torch.float32), interleaved)
        * sin_full.to(torch.float32)
    ).to(dtype=x.dtype)
    output[..., :rotary_dim] = rotated
    return output


def apply_rotary(
        x: torch.Tensor,  # 输入张量
        # 若 cu_seqlens is None:
        #   形状: [batch, seqlen, nheads, headdim]
        # 若 cu_seqlens is not None:
        #   形状: [total_seqlen, nheads, headdim]
        cos: torch.Tensor,  # cos 表，形状: [seqlen_ro, rotary_dim / 2]
        sin: torch.Tensor,  # sin 表，形状: [seqlen_ro, rotary_dim / 2]
        seqlen_offsets: Union[int, torch.Tensor] = 0,  # 每条序列的位置偏移
        # 可以是:
        # - 一个 int：所有序列共用同一个偏移
        # - 一个张量，形状: [batch]，每条序列各自一个偏移
        cu_seqlens: Optional[torch.Tensor] = None,  # 变长序列边界，形状: [batch + 1]
        max_seqlen: Optional[int] = None,  # 变长 batch 中的最大序列长度
        interleaved=False,  # True: GPT-J 风格；False: NeoX 风格
        inplace=False,  # 是否原地修改 x
        conjugate=False,  # 是否做共轭旋转；反向传播时会用到
) -> torch.Tensor:
    """
    参数:
        x:
            若 cu_seqlens is None:
                [batch, seqlen, nheads, headdim]
            否则:
                [total_seqlen, nheads, headdim]

        cos:
            [seqlen_ro, rotary_dim / 2]

        sin:
            [seqlen_ro, rotary_dim / 2]

        seqlen_offsets:
            一个整数，或形状为 [batch] 的整数张量

        cu_seqlens:
            [batch + 1]，或 None

        max_seqlen:
            一个整数；变长输入时必须提供

    返回:
        y:
            若非变长:
                [batch, seqlen, nheads, headdim]
            若变长:
                [total_seqlen, nheads, headdim]
    """

    # 是否是变长输入
    # 变长输入时，x 的第 0 维是 total_seqlen，而不是 batch
    is_varlen = cu_seqlens is not None

    if not is_varlen:
        # 非变长情况:
        # x 形状: [batch, seqlen, nheads, headdim]
        batch, seqlen, nheads, headdim = x.shape
    else:
        # 变长情况:
        # x 形状: [total_seqlen, nheads, headdim]
        assert max_seqlen is not None, (
            "If cu_seqlens is passed in, then max_seqlen must be passed"
        )

        total_seqlen, nheads, headdim = x.shape

        # cu_seqlens 形状是 [batch + 1]
        batch_p_1 = cu_seqlens.shape[0]
        batch = batch_p_1 - 1

        # 对 kernel 来说，序列长度用“最大序列长度”来驱动
        seqlen = max_seqlen

    # cos 的形状是 [seqlen_ro, rotary_dim / 2]
    seqlen_ro, rotary_dim = cos.shape

    # sin 必须与 cos 完全同形状
    assert sin.shape == cos.shape

    # 因为 cos.shape[1] 实际存的是 rotary_dim / 2，
    # 所以这里乘 2 还原真正的 rotary_dim
    rotary_dim *= 2

    # 旋转维度不能超过 head 总维度
    assert rotary_dim <= headdim, "rotary_dim must be <= headdim"

    # 当前 kernel 只支持 headdim <= 256
    assert headdim <= 256, "Only support headdim <= 256"

    # cos/sin 的可支持位置长度，必须覆盖当前序列长度
    assert seqlen_ro >= seqlen, "seqlen_ro must be >= seqlen"

    # cos 和 sin 的 dtype 必须一致
    assert (
            cos.dtype == sin.dtype
    ), f"cos and sin must have the same dtype, got {cos.dtype} and {sin.dtype}"

    # x 与 cos/sin 也必须同 dtype
    assert (
            x.dtype == cos.dtype
    ), f"Input and cos/sin must have the same dtype, got {x.dtype} and {cos.dtype}"

    # 确保 cos / sin 内存连续，方便 Triton kernel 访问
    cos, sin = cos.contiguous(), sin.contiguous()

    # 处理 seqlen_offsets
    if isinstance(seqlen_offsets, torch.Tensor):
        # 若是张量，则必须是一维 [batch]
        assert seqlen_offsets.shape == (batch,)
        # dtype 必须是整型
        assert seqlen_offsets.dtype in [torch.int32, torch.int64]
        # 保证连续
        seqlen_offsets = seqlen_offsets.contiguous()
    else:
        # 若是单个整数偏移，则要求：
        # 偏移 + 当前序列长度 <= cos/sin 支持的位置长度
        assert seqlen_offsets + seqlen <= seqlen_ro

    if not HAS_TRITON:
        logger.warning_once(
            "cfie_flash_attn rotary is unavailable because Triton runtime is "
            "not present; falling back to the shared torch rotary path."
        )
        return _apply_rotary_reference(
            x,
            cos,
            sin,
            seqlen_offsets,
            cu_seqlens,
            max_seqlen,
            interleaved=interleaved,
            inplace=inplace,
            conjugate=conjugate,
        )

    # 准备输出张量
    # 若不是原地模式，创建一个和 x 同形状同 dtype 的新张量
    # 若是原地模式，直接复用 x
    output = torch.empty_like(x) if not inplace else x

    # 若 rotary_dim < headdim，说明只有前 rotary_dim 维要旋转
    # 后面的维度要原样拷贝
    # 这里仅在“非原地模式”下需要手动拷贝
    if rotary_dim < headdim and not inplace:
        output[..., rotary_dim:].copy_(x[..., rotary_dim:])

    # 为 Triton kernel 选择 BLOCK_K
    # 实际就是根据 rotary_dim 选择更合适的 tile 大小
    BLOCK_K = (
        32
        if rotary_dim <= 32
        else (64 if rotary_dim <= 64 else (128 if rotary_dim <= 128 else 256))
    )

    # Triton grid 定义
    # 三维 launch：
    # - 第 0 维：沿 seqlen 分块
    # - 第 1 维：按 nheads
    # - 第 2 维：按 batch

    # triton.cdiv(seqlen, META["BLOCK_M"])
    # 把序列长度 seqlen 按 BLOCK_M 大小分块后，需要多少块
    # cdiv 是 ceil division，也就是向上取整除法。
    grid = lambda META: (triton.cdiv(seqlen, META["BLOCK_M"]), nheads, batch)

    # BLOCK_M 也是 kernel tile 参数
    # interleaved 时取 4
    # 非 interleaved 时，根据 rotary_dim 决定取 8 或 4
    BLOCK_M = 4 if interleaved else (8 if rotary_dim <= 128 else 4)

    # 必须显式切到 x 所在 CUDA 设备
    # 否则 Triton 可能默认在 cuda:0 发 kernel，导致设备不匹配错误
    with torch.cuda.device(x.device.index):
        rotary_kernel[grid](
            output,  # 输出指针
            x,  # 输入指针
            cos,  # cos 表
            sin,  # sin 表
            cu_seqlens,  # 变长边界；非变长时为 None
            seqlen_offsets,  # 位置偏移；可以是 int 或 [batch]

            # -------- 形状参数 --------
            seqlen,  # 当前序列长度（或 max_seqlen）
            rotary_dim,  # 实际参与旋转的维度数
            seqlen_ro,  # cos/sin 表可支持的位置长度

            # -------- output 的 stride --------
            output.stride(0) if not is_varlen else 0,  # 非变长时是 batch stride；变长时无 batch 维，传 0
            output.stride(-3),  # 非变长时是 seqlen stride；变长时是 total_seqlen stride
            output.stride(-2),  # nheads stride
            output.stride(-1),  # headdim stride

            # -------- x 的 stride --------
            x.stride(0) if not is_varlen else 0,  # 非变长时是 batch stride；变长时传 0
            x.stride(-3),  # 非变长时是 seqlen stride；变长时是 total_seqlen stride
            x.stride(-2),  # nheads stride
            x.stride(-1),  # headdim stride

            # -------- kernel 配置 --------
            BLOCK_K,  # rotary_dim 方向 tile 大小
            isinstance(seqlen_offsets, torch.Tensor),  # seqlen_offsets 是否是张量
            is_varlen,  # 是否是变长输入
            interleaved,  # GPT-J / NeoX 风格
            conjugate,  # 是否做共轭旋转（backward 用）
            BLOCK_M,  # seqlen 方向 tile 大小
            num_warps=2 if rotary_dim <= 64 else 4,  # Triton kernel 使用的 warp 数
        )

    return output
