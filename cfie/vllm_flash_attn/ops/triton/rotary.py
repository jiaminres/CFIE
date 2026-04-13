# Copy from https://github.com/vllm-project/flash-attention/blob/main/flash_attn/ops/triton/rotary.py
# Copyright (c) 2023, Tri Dao.

from typing import Optional, Union

import torch

from cfie.logger import init_logger
from cfie.triton_utils import HAS_TRITON, tl, triton

logger = init_logger(__name__)


@triton.jit
def rotary_kernel(
    OUT,  # Pointers to matrices
    X,
    COS,
    SIN,
    CU_SEQLENS,
    SEQLEN_OFFSETS,  # this could be int or a pointer
    # Matrix dimensions
    seqlen,
    rotary_dim,
    seqlen_ro,
    # strides
    stride_out_batch,
    stride_out_seqlen,
    stride_out_nheads,
    stride_out_headdim,
    stride_x_batch,
    stride_x_seqlen,
    stride_x_nheads,
    stride_x_headdim,
    # Meta-parameters
    BLOCK_K: tl.constexpr,
    IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    CONJUGATE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)
    pid_batch = tl.program_id(axis=2)
    rotary_dim_half = rotary_dim // 2

    if not IS_VARLEN:
        X = X + pid_batch * stride_x_batch + pid_head * stride_x_nheads
        OUT = OUT + pid_batch * stride_out_batch + pid_head * stride_out_nheads
    else:
        start_idx = tl.load(CU_SEQLENS + pid_batch)
        seqlen = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx
        X = X + start_idx * stride_x_seqlen + pid_head * stride_x_nheads
        OUT = OUT + start_idx * stride_out_seqlen + pid_head * stride_out_nheads

    if pid_m * BLOCK_M >= seqlen:
        return
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    if not IS_SEQLEN_OFFSETS_TENSOR:
        rm_cs = rm + SEQLEN_OFFSETS
    else:
        rm_cs = rm + tl.load(SEQLEN_OFFSETS + pid_batch)
    rk = tl.arange(0, BLOCK_K)
    rk_half = tl.arange(0, BLOCK_K // 2)

    if not INTERLEAVED:
        # Load the 1st and 2nd halves of X, do calculation, then store to 1st and 2nd halves of OUT
        X = X + (rm[:, None] * stride_x_seqlen + rk_half[None, :] * stride_x_headdim)
        COS = COS + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])
        SIN = SIN + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])
        cos = tl.load(
            COS, mask=(rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < rotary_dim_half), other=1.0
        ).to(tl.float32)
        sin = tl.load(
            SIN, mask=(rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < rotary_dim_half), other=0.0
        ).to(tl.float32)
        x0 = tl.load(
            X, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half), other=0.0
        ).to(tl.float32)
        x1 = tl.load(
            X + rotary_dim_half * stride_x_headdim,
            mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
            other=0.0,
        ).to(tl.float32)
        if CONJUGATE:
            sin = -sin
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos
        # write back result
        OUT = OUT + (rm[:, None] * stride_out_seqlen + rk_half[None, :] * stride_out_headdim)
        tl.store(OUT, o0, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half))
        tl.store(
            OUT + rotary_dim_half * stride_out_headdim,
            o1,
            mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
        )
    else:
        # We don't want to load X[0, 2, 4, ...] and X[1, 3, 5, ...] separately since both are slow.
        # Instead, we load x0 = X[0, 1, 2, 3, ...] and x1 = X[1, 0, 3, 2, ...].
        # Loading x0 will be fast but x1 will be slow.
        # Then we load cos = COS[0, 0, 1, 1, ...] and sin = SIN[0, 0, 1, 1, ...].
        # Then we do the calculation and use tl.where to pick put the right outputs for the even
        # and for the odd indices.
        rk_swap = rk + ((rk + 1) % 2) * 2 - 1  # 1, 0, 3, 2, 5, 4, ...
        rk_repeat = tl.arange(0, BLOCK_K) // 2
        X0 = X + (rm[:, None] * stride_x_seqlen + rk[None, :] * stride_x_headdim)
        X1 = X + (rm[:, None] * stride_x_seqlen + rk_swap[None, :] * stride_x_headdim)
        COS = COS + (rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :])
        SIN = SIN + (rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :])
        cos = tl.load(
            COS,
            mask=(rm_cs[:, None] < seqlen_ro) & (rk_repeat[None, :] < rotary_dim_half),
            other=1.0,
        ).to(tl.float32)
        sin = tl.load(
            SIN,
            mask=(rm_cs[:, None] < seqlen_ro) & (rk_repeat[None, :] < rotary_dim_half),
            other=0.0,
        ).to(tl.float32)
        x0 = tl.load(X0, mask=(rm[:, None] < seqlen) & (rk[None, :] < rotary_dim), other=0.0).to(
            tl.float32
        )
        x1 = tl.load(
            X1, mask=(rm[:, None] < seqlen) & (rk_swap[None, :] < rotary_dim), other=0.0
        ).to(tl.float32)
        if CONJUGATE:
            sin = -sin
        x0_cos = x0 * cos
        x1_sin = x1 * sin
        out = tl.where(rk[None, :] % 2 == 0, x0_cos - x1_sin, x0_cos + x1_sin)
        OUT = OUT + (rm[:, None] * stride_out_seqlen + rk[None, :] * stride_out_headdim)
        tl.store(OUT, out, mask=(rm[:, None] < seqlen) & (rk[None, :] < rotary_dim))


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
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    interleaved=False,
    inplace=False,
    conjugate=False,
) -> torch.Tensor:
    """
    Arguments:
        x: (batch, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim).
        cos: (seqlen_ro, rotary_dim / 2)
        sin: (seqlen_ro, rotary_dim / 2)
        seqlen_offsets: integer or integer tensor of size (batch,)
        cu_seqlens: (batch + 1,) or None
        max_seqlen: int
    Returns:
        y: (batch, seqlen, nheads, headdim)
    """
    is_varlen = cu_seqlens is not None
    if not is_varlen:
        batch, seqlen, nheads, headdim = x.shape
    else:
        assert max_seqlen is not None, "If cu_seqlens is passed in, then max_seqlen must be passed"
        total_seqlen, nheads, headdim = x.shape
        batch_p_1 = cu_seqlens.shape[0]
        batch = batch_p_1 - 1
        seqlen = max_seqlen
    seqlen_ro, rotary_dim = cos.shape
    assert sin.shape == cos.shape
    rotary_dim *= 2
    assert rotary_dim <= headdim, "rotary_dim must be <= headdim"
    assert headdim <= 256, "Only support headdim <= 256"
    assert seqlen_ro >= seqlen, "seqlen_ro must be >= seqlen"

    assert (
        cos.dtype == sin.dtype
    ), f"cos and sin must have the same dtype, got {cos.dtype} and {sin.dtype}"
    assert (
        x.dtype == cos.dtype
    ), f"Input and cos/sin must have the same dtype, got {x.dtype} and {cos.dtype}"

    cos, sin = cos.contiguous(), sin.contiguous()
    if isinstance(seqlen_offsets, torch.Tensor):
        assert seqlen_offsets.shape == (batch,)
        assert seqlen_offsets.dtype in [torch.int32, torch.int64]
        seqlen_offsets = seqlen_offsets.contiguous()
    else:
        assert seqlen_offsets + seqlen <= seqlen_ro

    if not HAS_TRITON:
        logger.warning_once(
            "vllm_flash_attn rotary is unavailable because Triton runtime is "
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

    output = torch.empty_like(x) if not inplace else x
    if rotary_dim < headdim and not inplace:
        output[..., rotary_dim:].copy_(x[..., rotary_dim:])

    BLOCK_K = (
        32
        if rotary_dim <= 32
        else (64 if rotary_dim <= 64 else (128 if rotary_dim <= 128 else 256))
    )
    grid = lambda META: (triton.cdiv(seqlen, META["BLOCK_M"]), nheads, batch)  # noqa
    BLOCK_M = 4 if interleaved else (8 if rotary_dim <= 128 else 4)

    # Need this, otherwise Triton tries to launch from cuda:0 and we get
    # ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
    with torch.cuda.device(x.device.index):
        rotary_kernel[grid](
            output,  # data ptrs
            x,
            cos,
            sin,
            cu_seqlens,
            seqlen_offsets,
            seqlen,  # shapes
            rotary_dim,
            seqlen_ro,
            output.stride(0) if not is_varlen else 0,  # batch_strides if not varlen else 0
            output.stride(-3),  # seqlen_stride or total_seqlen_stride
            output.stride(-2),  # nheads_stride
            output.stride(-1),  # headdim_stride
            x.stride(0) if not is_varlen else 0,  # batch_strides if not varlen else 0
            x.stride(-3),  # seqlen stride or total_seqlen_stride
            x.stride(-2),  # nheads stride
            x.stride(-1),  # headdim stride
            BLOCK_K,
            isinstance(seqlen_offsets, torch.Tensor),
            is_varlen,
            interleaved,
            conjugate,
            BLOCK_M,
            num_warps=2 if rotary_dim <= 64 else 4,
        )
    return output
