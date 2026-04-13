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

logger = init_logger(__name__)


def _recompute_w_u_fwd_reference(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: torch.LongTensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, Hg, K = k.shape
    H = v.shape[2]
    V = v.shape[3]
    BT = A.shape[-1]
    if Hg <= 0 or H % Hg != 0:
        raise ValueError(f"Expected H ({H}) to be divisible by Hg ({Hg}).")
    heads_per_group = H // Hg

    w = k.new_empty(B, T, H, K)
    u = torch.empty_like(v)

    k_work = k.float()
    v_work = v.float()
    beta_work = beta.float()
    g_work = torch.exp(g_cumsum.float())
    A_work = A.float()

    def _write_chunk(batch_idx: int, start: int, end: int) -> None:
        if end <= start:
            return
        chunk_len = end - start
        A_chunk_all = A_work[batch_idx, start:end, :, :chunk_len]
        for head_idx in range(H):
            group_idx = head_idx // heads_per_group
            beta_chunk = beta_work[batch_idx, start:end, head_idx].unsqueeze(-1)
            g_chunk = g_work[batch_idx, start:end, head_idx].unsqueeze(-1)
            A_chunk = A_chunk_all[:, head_idx, :]
            v_chunk = v_work[batch_idx, start:end, head_idx, :]
            k_chunk = k_work[batch_idx, start:end, group_idx, :]
            u_chunk = torch.matmul(A_chunk, v_chunk * beta_chunk)
            w_chunk = torch.matmul(A_chunk, k_chunk * beta_chunk * g_chunk)
            u[batch_idx, start:end, head_idx].copy_(u_chunk.to(u.dtype))
            w[batch_idx, start:end, head_idx].copy_(w_chunk.to(w.dtype))

    if cu_seqlens is None:
        total_chunks = (T + BT - 1) // BT
        for batch_idx in range(B):
            for chunk_idx in range(total_chunks):
                start = chunk_idx * BT
                end = min(start + BT, T)
                _write_chunk(batch_idx, start, end)
    else:
        if B != 1:
            raise ValueError("Only batch size 1 is supported with cu_seqlens.")
        for seq_idx in range(cu_seqlens.numel() - 1):
            seq_start = int(cu_seqlens[seq_idx].item())
            seq_end = int(cu_seqlens[seq_idx + 1].item())
            seq_len = seq_end - seq_start
            local_chunks = (seq_len + BT - 1) // BT
            for chunk_idx in range(local_chunks):
                start = seq_start + chunk_idx * BT
                end = min(start + BT, seq_end)
                _write_chunk(0, start, end)

    return w, u


def _try_precompiled_recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: torch.LongTensor | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not k.is_cuda:
        return None
    if not ops.has_precompiled_recompute_w_u_fwd():
        return None
    return ops.recompute_w_u_fwd_precompiled(
        k,
        v,
        beta,
        g_cumsum,
        A,
        cu_seqlens,
    )


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "V", "BT", "BK", "BV", "IS_VARLEN"],
)
@triton.jit(do_not_specialize=["T"])
def recompute_w_u_fwd_kernel(
    k,
    v,
    beta,
    w,
    u,
    A,
    g,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
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
    p_beta = tl.make_block_ptr(
        beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    p_g = tl.make_block_ptr(g + (bos * H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_g = tl.exp(tl.load(p_g, boundary_check=(0,)))

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        p_u = tl.make_block_ptr(
            u + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * Hg + i_h // (H // Hg)) * K,
            (T, K),
            (Hg * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_w = tl.make_block_ptr(
            w + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = (b_k * b_beta[:, None] * b_g[:, None]).to(b_k.dtype)
        b_w = tl.dot(b_A, b_kb)
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: torch.LongTensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, Hg, K, V = *k.shape, v.shape[-1]
    H = v.shape[-2]
    BT = A.shape[-1]

    if not HAS_TRITON:
        precompiled = _try_precompiled_recompute_w_u_fwd(
            k,
            v,
            beta,
            g_cumsum,
            A,
            cu_seqlens,
        )
        if precompiled is not None:
            return precompiled
        logger.warning_once(
            "FLA recompute_w_u_fwd is falling back to the PyTorch reference "
            "path because Triton runtime is unavailable."
        )
        return _recompute_w_u_fwd_reference(
            k,
            v,
            beta,
            g_cumsum,
            A,
            cu_seqlens,
        )

    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BK = 64
    BV = 64
    u = torch.empty_like(v)
    w = k.new_empty(B, T, H, K)
    recompute_w_u_fwd_kernel[(NT, B * H)](
        k=k,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        g=g_cumsum,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    return w, u
