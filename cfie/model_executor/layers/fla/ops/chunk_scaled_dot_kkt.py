# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501

import torch
import cfie._custom_ops as ops

from cfie.logger import init_logger
from cfie.triton_utils import HAS_TRITON, tl, triton

from .index import prepare_chunk_indices
from .op import exp

logger = init_logger(__name__)


def _chunk_scaled_dot_kkt_fwd_reference(
    k: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    B, T, Hg, K = k.shape
    H = beta.shape[-1]
    if H % Hg != 0:
        raise ValueError(f"Expected H ({H}) to be divisible by Hg ({Hg}).")
    heads_per_group = H // Hg

    k_work = k.float()
    beta_work = beta.float()
    g_work = g.float() if g is not None else None
    out = torch.zeros(B, T, H, chunk_size, device=k.device, dtype=output_dtype)

    if cu_seqlens is None:
        chunk_iter = [(batch_idx, chunk_idx) for batch_idx in range(B) for chunk_idx in range((T + chunk_size - 1) // chunk_size)]

        def _token_range(_batch_idx: int, chunk_idx: int) -> tuple[int, int]:
            start = chunk_idx * chunk_size
            return start, min(start + chunk_size, T)
    else:
        if B != 1:
            raise ValueError("Only batch size 1 is supported with cu_seqlens.")
        chunk_iter = []
        for seq_idx in range(cu_seqlens.numel() - 1):
            seq_start = int(cu_seqlens[seq_idx].item())
            seq_end = int(cu_seqlens[seq_idx + 1].item())
            seq_len = seq_end - seq_start
            local_chunks = (seq_len + chunk_size - 1) // chunk_size
            for local_chunk_idx in range(local_chunks):
                chunk_iter.append((0, (seq_idx, local_chunk_idx)))

        def _token_range(_batch_idx: int, seq_chunk: tuple[int, int]) -> tuple[int, int]:
            seq_idx, chunk_idx = seq_chunk
            seq_start = int(cu_seqlens[seq_idx].item())
            seq_end = int(cu_seqlens[seq_idx + 1].item())
            start = seq_start + chunk_idx * chunk_size
            return start, min(start + chunk_size, seq_end)

    for batch_idx, chunk_idx in chunk_iter:
        start, end = _token_range(batch_idx, chunk_idx)
        if end <= start:
            continue
        k_chunk_all = k_work[batch_idx, start:end]
        beta_chunk_all = beta_work[batch_idx, start:end]
        g_chunk_all = g_work[batch_idx, start:end] if g_work is not None else None
        chunk_len = end - start

        for head_idx in range(H):
            group_idx = head_idx // heads_per_group
            k_chunk = k_chunk_all[:, group_idx, :]
            beta_chunk = beta_chunk_all[:, head_idx]
            attn_chunk = torch.matmul(k_chunk * beta_chunk.unsqueeze(-1), k_chunk.transpose(0, 1))
            if g_chunk_all is not None:
                g_chunk = g_chunk_all[:, head_idx]
                attn_chunk = attn_chunk * torch.exp(
                    g_chunk.unsqueeze(-1) - g_chunk.unsqueeze(0)
                )
            attn_chunk = torch.tril(attn_chunk, diagonal=-1)
            out[batch_idx, start:end, head_idx, :chunk_len].copy_(
                attn_chunk.to(output_dtype)
            )
    return out


def _try_precompiled_chunk_scaled_dot_kkt_fwd(
    k: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor | None:
    if not k.is_cuda:
        return None
    if not ops.has_precompiled_chunk_scaled_dot_kkt_fwd():
        return None
    return ops.chunk_scaled_dot_kkt_fwd_precompiled(
        k,
        g,
        beta,
        cu_seqlens,
        chunk_size,
        output_dtype,
    )


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BK": BK}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64, 128]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "BT", "IS_VARLEN"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_scaled_dot_kkt_fwd_kernel(
    k,
    beta,
    g,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    p_beta = tl.make_block_ptr(
        beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    b_beta = tl.load(p_beta, boundary_check=(0,))

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * Hg + i_h // (H // Hg)) * K,
            (T, K),
            (Hg * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = b_k * b_beta[:, None]
        b_A += tl.dot(b_kb.to(b_k.dtype), tl.trans(b_k))

    if USE_G:
        p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_g_diff = b_g[:, None] - b_g[None, :]
        b_A = b_A * exp(b_g_diff)

    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)
    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT, (T, BT), (BT * H, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))


def chunk_scaled_dot_kkt_fwd(
    k: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    r"""
    Compute beta * K * K^T.

    Args:
        k (torch.Tensor):
            The key tensor of shape `[B, T, H, K]`.
        beta (torch.Tensor):
            The beta tensor of shape `[B, T, H]`.
        g (torch.Tensor):
            The cumulative sum of the gate tensor of shape `[B, T, H]`. Default: `None`.
        cu_seqlens (torch.LongTensor):
            The cumulative sequence lengths of the input tensor.
            Default: None
        chunk_size (int):
            The chunk size. Default: 64.
        output_dtype (torch.dtype):
            The dtype of the output tensor. Default: `torch.float32`

    Returns:
        beta * K * K^T of shape `[B, T, H, BT]` where `BT` is the chunk size.
    """
    # This kernel is slightly different from fla to support Q/K with different head numbers.
    # In fla, Q/K always have the same head number, so Hg is always equal to H.
    B, T, Hg, K = k.shape
    H = beta.shape[-1]
    BT = chunk_size
    if not HAS_TRITON:
        precompiled = _try_precompiled_chunk_scaled_dot_kkt_fwd(
            k,
            g,
            beta,
            cu_seqlens,
            chunk_size,
            output_dtype,
        )
        if precompiled is not None:
            return precompiled
        logger.warning_once(
            "FLA chunk_scaled_dot_kkt is falling back to the PyTorch "
            "reference path because Triton runtime is unavailable."
        )
        return _chunk_scaled_dot_kkt_fwd_reference(
            k,
            g,
            beta,
            cu_seqlens,
            chunk_size,
            output_dtype,
        )
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    A = torch.empty(B, T, H, BT, device=k.device, dtype=output_dtype)
    chunk_scaled_dot_kkt_fwd_kernel[(NT, B * H)](
        k=k,
        g=g,
        beta=beta,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        BT=BT,
    )
    return A
