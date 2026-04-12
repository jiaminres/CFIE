# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch

from cfie import _custom_ops as ops
from cfie.logger import init_logger
from cfie.triton_utils import HAS_TRITON, tl, triton

logger = init_logger(__name__)


def _try_precompiled_fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: float,
    threshold: float,
    scale: float,
    initial_state: torch.Tensor | None,
    inplace_final_state: bool,
    cu_seqlens: torch.LongTensor | None,
    ssm_state_indices: torch.Tensor | None,
    num_accepted_tokens: torch.Tensor | None,
    use_qk_l2norm_in_kernel: bool,
    is_kda: bool,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if initial_state is None or not q.is_cuda:
        return None
    if not ops.has_precompiled_fused_sigmoid_gating_delta_rule_update():
        return None
    return ops.fused_sigmoid_gating_delta_rule_update_precompiled(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        beta=beta,
        threshold=threshold,
        scale=scale,
        initial_state=initial_state,
        inplace_final_state=inplace_final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        is_kda=is_kda,
    )


def _resolve_state_index(
    ssm_state_indices: torch.Tensor | None,
    seq_idx: int,
    token_offset: int,
) -> int | None:
    if ssm_state_indices is None:
        return None
    if ssm_state_indices.ndim == 1:
        flat_index = seq_idx + token_offset
        if flat_index >= ssm_state_indices.numel():
            return -1
        return int(ssm_state_indices[flat_index].item())
    return int(ssm_state_indices[seq_idx, token_offset].item())


def _softplus_with_threshold(
    x: torch.Tensor,
    beta: float,
    threshold: float,
) -> torch.Tensor:
    beta_x = beta * x
    return torch.where(
        beta_x <= threshold,
        torch.log1p(torch.exp(beta_x)) / beta,
        x,
    )


def _fused_sigmoid_gating_delta_rule_update_ref(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: float,
    threshold: float,
    scale: float,
    initial_state: torch.Tensor | None,
    final_state: torch.Tensor,
    inplace_final_state: bool,
    cu_seqlens: torch.LongTensor | None,
    ssm_state_indices: torch.Tensor | None,
    num_accepted_tokens: torch.Tensor | None,
    use_qk_l2norm_in_kernel: bool,
    is_kda: bool,
    out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    hv_per_h = HV // H
    head_index = (
        torch.arange(HV, device=q.device, dtype=torch.long) // hv_per_h
    )

    if cu_seqlens is None:
        q_tokens = q.reshape(B * T, H, K)
        k_tokens = k.reshape(B * T, H, K)
        v_tokens = v.reshape(B * T, HV, V)
        if not is_kda:
            a_tokens = a.reshape(B * T, HV)
        else:
            a_tokens = a.reshape(B * T, HV, K)
        b_tokens = b.reshape(B * T, HV)
        seq_ranges = [(seq_idx * T, (seq_idx + 1) * T) for seq_idx in range(B)]
    else:
        q_tokens = q[0]
        k_tokens = k[0]
        v_tokens = v[0]
        if not is_kda:
            a_tokens = a.reshape(-1, HV)
        else:
            a_tokens = a.reshape(-1, HV, K)
        b_tokens = b.reshape(-1, HV)
        seq_ranges = [
            (int(cu_seqlens[seq_idx].item()), int(cu_seqlens[seq_idx + 1].item()))
            for seq_idx in range(len(cu_seqlens) - 1)
        ]

    out_tokens = out.reshape(-1, HV, V)
    zeros_output = torch.zeros(HV, V, device=q.device, dtype=torch.float32)
    A_log_f32 = A_log.to(torch.float32)
    dt_bias_f32 = dt_bias.to(torch.float32)

    for seq_idx, (bos, eos) in enumerate(seq_ranges):
        seq_len = eos - bos
        if seq_len <= 0:
            continue

        if initial_state is None:
            h = torch.zeros(HV, V, K, device=q.device, dtype=torch.float32)
        elif ssm_state_indices is not None:
            accepted_offset = 0
            if num_accepted_tokens is not None:
                accepted_offset = int(num_accepted_tokens[seq_idx].item()) - 1
            state_idx = _resolve_state_index(ssm_state_indices, seq_idx, accepted_offset)
            if state_idx is None:
                h = torch.zeros(HV, V, K, device=q.device, dtype=torch.float32)
            elif state_idx < 0:
                out_tokens[bos:eos].zero_()
                continue
            else:
                h = initial_state[state_idx].to(torch.float32).clone()
        else:
            state_base = seq_idx
            if initial_state.shape[0] > bos:
                state_base = bos
            h = initial_state[state_base].to(torch.float32).clone()

        for token_offset, token_idx in enumerate(range(bos, eos)):
            q_token = q_tokens[token_idx].to(torch.float32)[head_index]
            k_token = k_tokens[token_idx].to(torch.float32)[head_index]
            v_token = v_tokens[token_idx].to(torch.float32)
            beta_token = torch.sigmoid(b_tokens[token_idx].to(torch.float32))

            if not is_kda:
                x = a_tokens[token_idx].to(torch.float32) + dt_bias_f32
                g = -torch.exp(A_log_f32) * _softplus_with_threshold(
                    x,
                    beta=beta,
                    threshold=threshold,
                )
            else:
                x = a_tokens[token_idx].to(torch.float32) + dt_bias_f32[:, None]
                g = -torch.exp(A_log_f32).unsqueeze(1) * _softplus_with_threshold(
                    x,
                    beta=beta,
                    threshold=threshold,
                )

            if use_qk_l2norm_in_kernel:
                q_token = q_token / torch.sqrt((q_token * q_token).sum(dim=-1, keepdim=True) + 1e-6)
                k_token = k_token / torch.sqrt((k_token * k_token).sum(dim=-1, keepdim=True) + 1e-6)

            q_token = q_token * scale
            if not is_kda:
                h = h * torch.exp(g).view(HV, 1, 1)
            else:
                h = h * torch.exp(g)[:, None, :]

            updated_v = v_token - (h * k_token[:, None, :]).sum(dim=-1)
            updated_v = updated_v * beta_token[:, None]
            h = h + updated_v[:, :, None] * k_token[:, None, :]
            out_tokens[token_idx] = (h * q_token[:, None, :]).sum(dim=-1).to(torch.float32)

            if inplace_final_state:
                final_state_idx = _resolve_state_index(
                    ssm_state_indices,
                    seq_idx,
                    token_offset,
                )
                if final_state_idx is not None and final_state_idx >= 0:
                    final_state[final_state_idx] = h.to(final_state.dtype)
            else:
                final_state[token_idx] = h.to(final_state.dtype)

    if ssm_state_indices is not None:
        valid_mask = ssm_state_indices.reshape(-1) >= 0
        if out_tokens.shape[0] == valid_mask.shape[0]:
            out_tokens.copy_(
                torch.where(
                    valid_mask.view(-1, 1, 1),
                    out_tokens,
                    zeros_output.to(out_tokens.dtype).unsqueeze(0).expand_as(out_tokens),
                )
            )

    return out, final_state


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "IS_CONTINUOUS_BATCHING": lambda args: args["ssm_state_indices"] is not None,
        "IS_SPEC_DECODING": lambda args: args["num_accepted_tokens"] is not None,
    }
)
@triton.jit(do_not_specialize=["N", "T"])
def fused_sigmoid_gating_delta_rule_update_kernel(
    A_log,
    a,
    b,
    dt_bias,
    beta,
    threshold,
    q,
    k,
    v,
    o,
    h0,
    ht,
    cu_seqlens,
    ssm_state_indices,
    num_accepted_tokens,
    scale,
    N: tl.int64,  # num of sequences
    T: tl.int64,  # num of tokens
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
    INPLACE_FINAL_STATE: tl.constexpr,  # whether to store final state inplace
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    IS_KDA: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    if T == 0:
        # no tokens to process for this sequence
        return

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v

    p_A_log = A_log + i_hv
    if not IS_KDA:
        p_a = a + bos * HV + i_hv
        p_dt_bias = dt_bias + i_hv
    else:
        p_a = a + (bos * HV + i_hv) * K + o_k
        p_dt_bias = dt_bias + i_hv * K + o_k

    p_b = b + bos * HV + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if IS_CONTINUOUS_BATCHING:
            if IS_SPEC_DECODING:
                i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
            else:
                i_t = 0
            # Load state index and check for PAD_SLOT_ID (-1)
            state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t).to(
                tl.int64
            )
            # Skip if state index is invalid (PAD_SLOT_ID = -1)
            if state_idx < 0:
                return
            p_h0 = h0 + state_idx * stride_init_state_token
        else:
            p_h0 = h0 + bos * HV * V * K
        p_h0 = p_h0 + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for i_t in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(p_b).to(tl.float32)

        # If the model is loaded in fp16, without the .float() here, A might be -inf
        x = tl.load(p_a).to(tl.float32) + tl.load(p_dt_bias).to(tl.float32)
        softplus_x = tl.where(
            beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
        )
        b_g = -tl.exp(tl.load(p_A_log).to(tl.float32)) * softplus_x

        # compute beta_output = sigmoid(b)
        b_beta = tl.sigmoid(b_b.to(tl.float32))

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q * (tl.rsqrt(tl.sum(b_q * b_q) + 1e-6))
            b_k = b_k * (tl.rsqrt(tl.sum(b_k * b_k) + 1e-6))
        b_q = b_q * scale
        # [BV, BK]
        if not IS_KDA:
            b_h *= tl.exp(b_g)
        else:
            b_h *= tl.exp(b_g[None, :])
        # [BV]
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        b_v *= b_beta
        # [BV, BK]
        b_h += b_v[:, None] * b_k[None, :]
        # [BV]
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # keep the states for multi-query tokens
        if INPLACE_FINAL_STATE:
            # Load state index and check for PAD_SLOT_ID (-1)
            final_state_idx = tl.load(
                ssm_state_indices + i_n * stride_indices_seq + i_t
            ).to(tl.int64)
            # Only store if state index is valid (not PAD_SLOT_ID)
            if final_state_idx >= 0:
                p_ht = ht + final_state_idx * stride_final_state_token
                p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
                tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
        else:
            p_ht = ht + (bos + i_t) * stride_final_state_token
            p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

        # Update pointers for next timestep
        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        p_b += HV
        p_a += HV


def fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
    scale: float = None,
    initial_state: torch.Tensor = None,
    inplace_final_state: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    is_kda: bool = False,
):
    """
    Fused triton implementation of sigmoid gating delta rule update.
    This function uses a single fused kernel that combines both sigmoid gating
    computation and the recurrent delta rule update for better performance.
    """
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 4

    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(
            f"The batch size is expected to be 1 rather than {q.shape[0]}"
            f" when using `cu_seqlens`. Please flatten variable-length"
            f" inputs before processing."
        )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"

    o = q.new_empty(NK, *v.shape)
    if inplace_final_state:
        final_state = initial_state
    else:
        final_state = q.new_empty(T, HV, V, K, dtype=initial_state.dtype)

    if not HAS_TRITON:
        precompiled = _try_precompiled_fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            a=a,
            b=b,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            beta=beta,
            threshold=threshold,
            scale=scale,
            initial_state=initial_state,
            inplace_final_state=inplace_final_state,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=ssm_state_indices,
            num_accepted_tokens=num_accepted_tokens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            is_kda=is_kda,
        )
        if precompiled is not None:
            return precompiled
        logger.warning_once(
            "Qwen3Next fused GDN gating is falling back to the PyTorch "
            "reference path because Triton runtime is unavailable."
        )
        return _fused_sigmoid_gating_delta_rule_update_ref(
            A_log=A_log,
            a=a,
            b=b,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            beta=beta,
            threshold=threshold,
            scale=scale,
            initial_state=initial_state,
            final_state=final_state,
            inplace_final_state=inplace_final_state,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=ssm_state_indices,
            num_accepted_tokens=num_accepted_tokens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            is_kda=is_kda,
            out=o.squeeze(0),
        )

    stride_init_state_token = initial_state.stride(0)
    stride_final_state_token = final_state.stride(0)

    if ssm_state_indices is None:
        stride_indices_seq, stride_indices_tok = 1, 1
    elif ssm_state_indices.ndim == 1:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride(0), 1
    else:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()

    grid = (NK, NV, N * HV)
    fused_sigmoid_gating_delta_rule_update_kernel[grid](
        A_log=A_log,
        a=a.contiguous(),
        b=b.contiguous(),
        dt_bias=dt_bias,
        beta=beta,
        threshold=threshold,
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        scale=scale,
        N=N,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        stride_init_state_token=stride_init_state_token,
        stride_final_state_token=stride_final_state_token,
        stride_indices_seq=stride_indices_seq,
        stride_indices_tok=stride_indices_tok,
        INPLACE_FINAL_STATE=inplace_final_state,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_KDA=is_kda,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    o = o.squeeze(0)
    return o, final_state
