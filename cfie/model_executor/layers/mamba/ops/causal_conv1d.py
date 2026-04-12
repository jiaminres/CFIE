# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024, Tri Dao.
# Adapted from https://github.com/Dao-AILab/causal-conv1d/blob/main/causal_conv1d/causal_conv1d_interface.py


import numpy as np
import torch

import cfie._custom_ops as ops
from cfie.logger import init_logger
from cfie.triton_utils import HAS_TRITON, tl, triton
from cfie.v1.attention.backends.utils import PAD_SLOT_ID

logger = init_logger(__name__)


def _activation_to_kernel_arg(activation: str | None) -> str:
    return "" if activation is None else activation


def _try_precompiled_causal_conv1d_fn(
    *,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor | None,
    has_initial_state: torch.Tensor | None,
    activation: str | None,
    pad_slot_id: int,
) -> torch.Tensor | None:
    if not x.is_cuda or not conv_states.is_cuda:
        return None
    if not ops.has_precompiled_causal_conv1d_fn():
        return None
    return ops.causal_conv1d_fn_precompiled(
        x=x,
        weight=weight,
        bias=bias,
        conv_states=conv_states,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        activation=_activation_to_kernel_arg(activation),
        pad_slot_id=pad_slot_id,
    )


def _try_precompiled_causal_conv1d_update(
    *,
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
    conv_state_indices: torch.Tensor | None,
    num_accepted_tokens: torch.Tensor | None,
    query_start_loc: torch.Tensor | None,
    pad_slot_id: int,
    block_idx_last_scheduled_token: torch.Tensor | None,
    initial_state_idx: torch.Tensor | None,
) -> torch.Tensor | None:
    if not x.is_cuda or not conv_state.is_cuda:
        return None
    if not ops.has_precompiled_causal_conv1d_update():
        return None
    return ops.causal_conv1d_update_precompiled(
        x=x,
        conv_state=conv_state,
        weight=weight,
        bias=bias,
        activation=_activation_to_kernel_arg(activation),
        conv_state_indices=conv_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        query_start_loc=query_start_loc,
        pad_slot_id=pad_slot_id,
        block_idx_last_scheduled_token=block_idx_last_scheduled_token,
        initial_state_idx=initial_state_idx,
    )


def _apply_activation_ref(x: torch.Tensor, activation: str | None) -> torch.Tensor:
    if activation in ["silu", "swish"]:
        return x * torch.sigmoid(x)
    return x


def _resolve_cache_line(
        cache_indices: torch.Tensor | None,
        seq_idx: int,
        *,
        block_offset: int = 0,
) -> int:
    if cache_indices is None:
        return seq_idx

    if cache_indices.dim() == 1:
        return int(cache_indices[seq_idx].item())

    return int(cache_indices[seq_idx, block_offset].item())


def _load_initial_history(
        state: torch.Tensor | None,
        history_len: int,
        *,
        load_initial_state: bool,
        start_offset: int = 0,
) -> torch.Tensor:
    if state is None or history_len <= 0 or not load_initial_state:
        dim = 0 if state is None else state.shape[0]
        dtype = torch.float32 if state is None else state.dtype
        device = None if state is None else state.device
        return torch.zeros((dim, history_len), dtype=dtype, device=device)

    history = state[:, start_offset: start_offset + history_len]
    if history.shape[-1] == history_len:
        return history

    padded = torch.zeros(
        (state.shape[0], history_len),
        dtype=state.dtype,
        device=state.device,
    )
    if history.numel() > 0:
        padded[:, : history.shape[-1]] = history
    return padded


def _update_conv_state_ref(
        state: torch.Tensor,
        seq_tokens: torch.Tensor,
        *,
        shift_tokens: int,
) -> torch.Tensor:
    state_len = state.shape[-1]
    if state_len == 0:
        return state

    shift_tokens = max(shift_tokens, 0)
    retained_state = state[:, min(shift_tokens, state_len):]
    updated = torch.cat([retained_state, seq_tokens], dim=-1)

    if updated.shape[-1] >= state_len:
        return updated[:, -state_len:]

    padded = torch.zeros_like(state)
    padded[:, -updated.shape[-1]:] = updated
    return padded


def _causal_conv1d_sequence_ref(
        seq_tokens: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        history: torch.Tensor,
        *,
        activation: str | None,
) -> torch.Tensor:
    _, width = weight.shape
    combined = torch.cat([history, seq_tokens], dim=-1)
    windows = combined.unfold(dimension=-1, size=width, step=1)
    output = (windows * weight[:, None, :]).sum(dim=-1)
    if bias is not None:
        output = output + bias[:, None]
    return _apply_activation_ref(output, activation)


def _causal_conv1d_fn_ref(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        conv_states: torch.Tensor | None,
        query_start_loc: torch.Tensor,
        *,
        cache_indices: torch.Tensor | None = None,
        has_initial_state: torch.Tensor | None = None,
        activation: str | None = "silu",
) -> torch.Tensor:
    out = torch.empty_like(x)
    _, width = weight.shape
    history_len = width - 1
    batch = query_start_loc.numel() - 1

    for seq_idx in range(batch):
        seq_start = int(query_start_loc[seq_idx].item())
        seq_end = int(query_start_loc[seq_idx + 1].item())
        if seq_end <= seq_start:
            continue

        cache_line = _resolve_cache_line(cache_indices, seq_idx)
        if cache_line == PAD_SLOT_ID:
            continue

        state = None if conv_states is None else conv_states[cache_line]
        load_initial_state = (
            has_initial_state is None or bool(has_initial_state[seq_idx].item())
        )
        history = _load_initial_history(
            state,
            history_len,
            load_initial_state=load_initial_state,
        )
        seq_tokens = x[:, seq_start:seq_end]
        out[:, seq_start:seq_end] = _causal_conv1d_sequence_ref(
            seq_tokens,
            weight,
            bias,
            history,
            activation=activation,
        )

        if conv_states is not None:
            conv_states[cache_line] = _update_conv_state_ref(
                conv_states[cache_line],
                seq_tokens,
                shift_tokens=seq_tokens.shape[-1],
            )

    return out


def _causal_conv1d_update_ref(
        x: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        *,
        activation: str | None,
        conv_state_indices: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
        pad_slot_id: int = PAD_SLOT_ID,
        block_idx_last_scheduled_token: torch.Tensor | None = None,
        initial_state_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    if (
            query_start_loc is None
            and num_accepted_tokens is None
            and block_idx_last_scheduled_token is None
            and initial_state_idx is None
            and (conv_state_indices is None or conv_state_indices.ndim == 1)
    ):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        batch, dim, seqlen = x.shape
        history_len = weight.shape[1] - 1
        safe_indices = (
            torch.arange(batch, device=x.device, dtype=torch.long)
            if conv_state_indices is None
            else conv_state_indices.to(device=x.device, dtype=torch.long)
        )
        valid_mask = safe_indices >= 0
        gather_indices = safe_indices.clamp_min(0)
        state = conv_state.index_select(0, gather_indices)
        history = state[:, :, :history_len]
        combined = torch.cat([history, x], dim=-1)
        windows = combined.unfold(dimension=-1, size=weight.shape[1], step=1)
        out = (windows * weight.view(1, dim, 1, weight.shape[1])).sum(dim=-1)
        if bias is not None:
            out = out + bias.view(1, dim, 1)
        out = _apply_activation_ref(out, activation)
        out = out * valid_mask.view(batch, 1, 1)

        updated_state = _update_conv_state_ref(
            state.reshape(batch * dim, state.shape[-1]),
            x.reshape(batch * dim, seqlen),
            shift_tokens=seqlen,
        ).reshape(batch, dim, state.shape[-1])
        conv_state.index_copy_(0, gather_indices, updated_state)
        return out

    out = x.clone()
    _, width = weight.shape
    history_len = width - 1

    if query_start_loc is None:
        batch, _, _ = x.shape
    else:
        batch = query_start_loc.numel() - 1

    for seq_idx in range(batch):
        if query_start_loc is None:
            seq_tokens = x[seq_idx]
            seq_len = seq_tokens.shape[-1]
        else:
            seq_start = int(query_start_loc[seq_idx].item())
            seq_end = int(query_start_loc[seq_idx + 1].item())
            if seq_end <= seq_start:
                continue
            seq_tokens = x[seq_start:seq_end].transpose(0, 1).contiguous()
            seq_len = seq_end - seq_start

        input_block_offset = 0
        if initial_state_idx is not None:
            input_block_offset = int(initial_state_idx[seq_idx].item())
        input_cache_line = _resolve_cache_line(
            conv_state_indices,
            seq_idx,
            block_offset=input_block_offset,
        )
        if input_cache_line == pad_slot_id:
            continue

        output_block_offset = 0
        if block_idx_last_scheduled_token is not None:
            output_block_offset = int(block_idx_last_scheduled_token[seq_idx].item())
        output_cache_line = _resolve_cache_line(
            conv_state_indices,
            seq_idx,
            block_offset=output_block_offset,
        )

        state = conv_state[input_cache_line]
        history_offset = 0
        shift_tokens = seq_len
        if num_accepted_tokens is not None:
            history_offset = max(int(num_accepted_tokens[seq_idx].item()) - 1, 0)
            shift_tokens = 1

        history = _load_initial_history(
            state,
            history_len,
            load_initial_state=True,
            start_offset=history_offset,
        )
        seq_output = _causal_conv1d_sequence_ref(
            seq_tokens,
            weight,
            bias,
            history,
            activation=activation,
        )

        if query_start_loc is None:
            out[seq_idx, :, :seq_len] = seq_output
        else:
            out[seq_start:seq_end] = seq_output.transpose(0, 1)

        if output_cache_line != pad_slot_id:
            conv_state[output_cache_line] = _update_conv_state_ref(
                conv_state[input_cache_line],
                seq_tokens,
                shift_tokens=shift_tokens,
            )

    return out


@triton.jit()
def _causal_conv1d_fwd_kernel(  # continuous batching
        # Pointers to matrices
        x_ptr,  # (dim, cu_seqlen) holding `batch` of actual sequences + padded sequences
        w_ptr,  # (dim, width)
        bias_ptr,
        initial_states_ptr,  # conv_states_ptr
        cache_indices_ptr,  # (batch, n_blocks + padding) The second dimension contains
        # the block indices relevant for each sequence
        # plus potential 0-padding at the beginning and at the end
        has_initial_states_ptr,
        query_start_loc_ptr,
        batch_ptr,
        token_chunk_offset_ptr,
        block_idx_first_scheduled_token,  # (batch,)
        block_idx_last_scheduled_token,  # (batch,)
        initial_state_idx,  # (batch,)
        num_computed_tokens,  # (batch,)
        o_ptr,  # (dim, seqlen) - actually pointing to x_ptr
        # Matrix dimensions
        dim: tl.constexpr,
        seqlen: tl.int32,  # cu_seqlen
        num_cache_lines: tl.constexpr,  # added to support vLLM larger cache lines
        # Strides
        stride_x_dim: tl.constexpr,  # stride to get to next feature-value,
        stride_x_token: tl.constexpr,  # stride to get to next token (same feature-index, same sequence-index)
        stride_w_dim: tl.constexpr,  # stride to get to next dim-axis value
        stride_w_width: tl.constexpr,  # stride to get to next width-axis value
        stride_istate_seq: tl.constexpr,
        stride_istate_dim: tl.constexpr,
        stride_istate_token: tl.constexpr,
        stride_cache_indices: tl.constexpr,
        stride_o_dim: tl.constexpr,
        stride_o_token: tl.constexpr,
        stride_block_m: tl.constexpr,  # Stride block to align divided by BLOCK_M
        # others
        pad_slot_id: tl.constexpr,
        # Meta-parameters
        HAS_BIAS: tl.constexpr,
        KERNEL_WIDTH: tl.constexpr,
        SILU_ACTIVATION: tl.constexpr,
        IS_APC_ENABLED: tl.constexpr,
        USE_PAD_SLOT: tl.constexpr,
        NP2_STATELEN: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
):
    conv_states_ptr = initial_states_ptr
    conv_state_indices_ptr = cache_indices_ptr
    stride_conv_state_seq = stride_istate_seq
    stride_conv_state_dim = stride_istate_dim
    stride_conv_state_tok = stride_istate_token
    state_len = (
            KERNEL_WIDTH - 1
    )  # can be passed via argument if it's not the same as this value

    # one program handles one chunk in a single sequence
    # rather than mixing sequences - to make updating initial_states across sequences efficiently

    # single-sequence id
    idx_seq = tl.load(batch_ptr + tl.program_id(0)).to(tl.int64)
    chunk_offset = tl.load(token_chunk_offset_ptr + tl.program_id(0))

    # BLOCK_N elements along the feature-dimension (channel)
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    if idx_seq == pad_slot_id:
        return

    sequence_start_index = tl.load(query_start_loc_ptr + idx_seq)
    sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1)
    # find the actual sequence length
    seqlen = sequence_end_index - sequence_start_index

    B_size: tl.constexpr = stride_block_m * BLOCK_M

    if IS_APC_ENABLED:
        # Handle the case if prefix caching is enabled.
        # In particular, if prefix caching is enabled, the program write additional cache states to "cache_indices_ptr"

        # Get the length of the completed sequence so far and compute the offset.
        current_first_index = tl.load(block_idx_first_scheduled_token + idx_seq)
        current_last_index = tl.load(block_idx_last_scheduled_token + idx_seq)
        sequence_completed_index = tl.load(num_computed_tokens + idx_seq)

        # Compute the offset where the first stride_block_m-aligned first full block is
        # Value in "token-space"
        sequence_completed_offset_token = sequence_completed_index % B_size
        seq_completed_offset = B_size - sequence_completed_offset_token
        seq_end_offset = (seqlen - seq_completed_offset) % B_size
        last_full_block_token_index = sequence_end_index - seq_end_offset
        # If the sequence without the sequence_offset_index is stride_cache_chunk-aligned, then the last full chunk is the second-to-last one
        if seq_end_offset == 0:
            last_full_block_token_index = last_full_block_token_index - B_size

        # Get the number of blocks to be filled for the current sequence
        # If n_block_to_fill = 0, then only the state at the sequence end is stored
        n_block_to_fill = current_last_index - current_first_index

        # Get the index of the init block
        conv_state_init_index = tl.load(initial_state_idx + idx_seq)
    else:
        n_block_to_fill = 0
        current_last_index = 0
        conv_state_init_index = 0
        current_first_index = 0
        last_full_block_token_index = 0

    token_offset = BLOCK_M * chunk_offset
    segment_len = min(BLOCK_M, seqlen - token_offset)

    # base of the sequence
    x_base = (
            x_ptr + sequence_start_index * stride_x_token + idx_feats * stride_x_dim
    )  # [BLOCK_N,]

    # cache_idx
    conv_states_input_coord = tl.load(
        conv_state_indices_ptr + idx_seq * stride_cache_indices + conv_state_init_index
    ).to(tl.int64)

    if USE_PAD_SLOT:  # noqa
        if conv_states_input_coord == pad_slot_id:
            # not processing as this is not the actual sequence
            return
    conv_states_base = (
            conv_states_ptr
            + (conv_states_input_coord * stride_conv_state_seq)
            + (idx_feats * stride_conv_state_dim)
    )  # [BLOCK_N,]

    w_base = w_ptr + (idx_feats * stride_w_dim)  # [BLOCK_N,]

    # Does 2 things:
    # 1. READ prior-block init-state data - [done by every Triton programs]
    # 2. update conv_state with new data [only by the Triton program handles chunk_offset=0]
    if chunk_offset == 0:
        # read from conv_states
        load_init_state = tl.load(has_initial_states_ptr + idx_seq).to(tl.int1)
        if load_init_state:
            # load from conv_states
            prior_tokens = conv_states_base + (state_len - 1) * stride_conv_state_tok
            mask_w = idx_feats < dim
            if KERNEL_WIDTH == 2:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
            if KERNEL_WIDTH == 3:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
            if KERNEL_WIDTH == 4:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N]
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 2 * stride_conv_state_tok  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
            if KERNEL_WIDTH == 5:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col3 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N]
                col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 2 * stride_conv_state_tok  # [BLOCK_N]
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 3 * stride_conv_state_tok  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
        else:
            # prior-tokens are zeros
            if KERNEL_WIDTH >= 2:  # STRATEGY1
                # first chunk and does not have prior-token, so just set to 0
                col0 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 3:  # STRATEGY1
                col1 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 4:  # STRATEGY1
                col2 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 5:  # STRATEGY1
                col3 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)

        # STEP 2:
        # here prepare data for updating conv_state
        if (
                state_len <= seqlen
        ):  # SMALL_CACHE=True (only move part of 'x' into conv_state cache)
            # just read from 'x'
            # copy 'x' data to conv_state
            # load only 'x' data (and set 0 before 'x' if seqlen < state_len)
            idx_tokens_last = (seqlen - state_len) + tl.arange(
                0, NP2_STATELEN
            )  # [BLOCK_M]
            x_ptrs = (
                    x_ptr
                    + ((sequence_start_index + idx_tokens_last) * stride_x_token)[:, None]
                    + (idx_feats * stride_x_dim)[None, :]
            )  # [BLOCK_M,BLOCK_N,]
            mask_x = (
                    (idx_tokens_last >= 0)[:, None]
                    & (idx_tokens_last < seqlen)[:, None]
                    & (idx_feats < dim)[None, :]
            )  # token-index  # token-index  # feature-index
            loaded_x = tl.load(x_ptrs, mask_x, 0.0)
            idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

            # Compute the offset where the last block should be written in the conv_states
            conv_states_output_coord = tl.load(
                conv_state_indices_ptr
                + idx_seq * stride_cache_indices
                + current_last_index
            ).to(tl.int64)

            conv_states_ptrs_target = (
                                              conv_states_ptr
                                              + (conv_states_output_coord * stride_conv_state_seq)  # Offset from seq
                                              + (idx_feats * stride_conv_state_dim)
                                      )[None, :] + (  # [BLOCK_N,]
                                              idx_tokens_conv * stride_conv_state_tok
                                      )[:, None]

            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
            tl.debug_barrier()  # NOTE: use this due to bug in Triton compiler
            tl.store(conv_states_ptrs_target, loaded_x, mask)

        else:
            if load_init_state:
                # update conv_state by shifting left, i.e. take last few cols from conv_state + cols from 'x'
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

                conv_states_ptrs_source = (
                        conv_states_ptr
                        + (conv_states_input_coord * stride_conv_state_seq)
                        + (idx_feats * stride_conv_state_dim)[None, :]
                        + ((idx_tokens_conv + seqlen) * stride_conv_state_tok)[:, None]
                )  # [BLOCK_M, BLOCK_N]
                mask = (
                        (conv_states_input_coord < num_cache_lines)
                        & ((idx_tokens_conv + seqlen) < state_len)[:, None]
                        & (idx_feats < dim)[None, :]
                )
                conv_state = tl.load(conv_states_ptrs_source, mask, other=0.0)

                VAL = state_len - seqlen

                x_ptrs = (
                        x_base[None, :]
                        + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                )  # [BLOCK_M, BLOCK_N]

                mask_x = (
                        (idx_tokens_conv - VAL >= 0)[:, None]
                        & (idx_tokens_conv - VAL < seqlen)[:, None]
                        & (idx_feats < dim)[None, :]
                )  # token-index  # token-index  # feature-index
                loaded_x = tl.load(x_ptrs, mask_x, 0.0)

                tl.debug_barrier()  # need this due to the bug in tl.where not enforcing this when data is the result of another tl.load
                new_conv_state = tl.where(
                    mask, conv_state, loaded_x
                )  # BUG in 'tl.where'  which requires a barrier before this
                conv_states_ptrs_target = (
                        conv_states_base
                        + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                )  # [BLOCK_M, BLOCK_N]
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[
                    None, :
                ]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)
            else:  # load_init_state == False
                # update conv_state by shifting left, BUT
                # set cols prior to 'x' as zeros + cols from 'x'
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

                VAL = state_len - seqlen

                x_ptrs = (
                        x_base[None, :]
                        + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                )  # [BLOCK_M, BLOCK_N]

                mask_x = (
                        (idx_tokens_conv - VAL >= 0)[:, None]
                        & (idx_tokens_conv - VAL < seqlen)[:, None]
                        & (idx_feats < dim)[None, :]
                )  # token-index  # token-index  # feature-index
                new_conv_state = tl.load(x_ptrs, mask_x, 0.0)

                conv_states_ptrs_target = (
                        conv_states_base
                        + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                )  # [BLOCK_M, BLOCK_N]
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[
                    None, :
                ]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)

    else:  # chunk_offset > 0
        # read prior-token data from `x`
        load_init_state = True
        prior_tokens = x_base + (token_offset - 1) * stride_x_token
        mask_w = idx_feats < dim
        if KERNEL_WIDTH == 2:
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 3:
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 4:
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col2 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N]
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 2 * stride_x_token  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 5:
            # ruff: noqa: F841
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col3 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N]
            col2 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 2 * stride_x_token  # [BLOCK_N]
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 3 * stride_x_token  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")

        # Store intermediate states aligned with stride_block_m
        # The additional states are cached starting from the last stride_block_m.
        # For example:
        # If n_block_to_fill = 0, then only the state at the sequence end is cached and the process below is not involved.
        # If n_block_to_fill > 0, then the states at the sequence end and at the n_block_to_fill-last
        # stride_block_m are cached.
        # For example chunk_offset = n_block_to_fill stores the state at last_full_block
        if (chunk_offset - 1) < n_block_to_fill:
            # Store the states at the chunk boundaries from the start of the sequence
            idx_tokens_last = (
                                      last_full_block_token_index
                                      - (n_block_to_fill - chunk_offset) * B_size
                                      - state_len
                              ) + tl.arange(0, NP2_STATELEN)  # [BLOCK_M]
            x_ptrs = (
                    x_ptr
                    + (idx_tokens_last * stride_x_token)[:, None]
                    + (idx_feats * stride_x_dim)[None, :]
            )  # [BLOCK_M,BLOCK_N,]

            mask_x = (idx_tokens_last >= 0)[:, None] & (idx_feats < dim)[
                None, :
            ]  # token-index  # token-index  # feature-index
            loaded_x = tl.load(x_ptrs, mask_x, 0.0)
            idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

            # cache_idx
            conv_states_output_coord = tl.load(
                conv_state_indices_ptr
                + idx_seq * stride_cache_indices
                + current_first_index
                + (chunk_offset - 1)
            ).to(tl.int64)

            conv_states_ptrs_target = (
                                              conv_states_ptr
                                              + (conv_states_output_coord * stride_conv_state_seq)  # Offset from seq
                                              + (idx_feats * stride_conv_state_dim)
                                      )[None, :] + (  # [BLOCK_N,]
                                              idx_tokens_conv * stride_conv_state_tok
                                      )[:, None]

            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
            tl.debug_barrier()  # NOTE: use this due to bug in Triton compiler
            tl.store(conv_states_ptrs_target, loaded_x, mask)

    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(
            tl.float32
        )  # [BLOCK_N]
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    x_base_1d = x_base + token_offset * stride_x_token  # starting of chunk

    # PRE-LOAD WEIGHTS
    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_ptrs = w_base + (0 * stride_w_width)  # [BLOCK_N] tensor
        w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
        w_ptrs = w_base + (1 * stride_w_width)  # [BLOCK_N] tensor
        w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_ptrs = w_base + (2 * stride_w_width)  # [BLOCK_N] tensor
        w_col2 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_ptrs = w_base + (3 * stride_w_width)  # [BLOCK_N] tensor
        w_col3 = tl.load(w_ptrs, mask_w, other=0.0)
    mask_x_1d = idx_feats < dim
    for idx_token in range(segment_len):
        acc = acc_preload

        matrix_w = w_col0
        matrix_x = col0
        for j in tl.static_range(KERNEL_WIDTH):
            if KERNEL_WIDTH == 2:
                if j == 1:  # KERNEL_WIDTH-1:
                    matrix_w = w_col1
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 3:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 4:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)

            acc += matrix_x * matrix_w  # [BLOCK_N]

        if KERNEL_WIDTH == 2:
            col0 = matrix_x
        elif KERNEL_WIDTH == 3:
            col0 = col1
            col1 = matrix_x
        elif KERNEL_WIDTH == 4:
            col0 = col1
            col1 = col2
            col2 = matrix_x

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < segment_len) & (
                idx_feats < dim
        )  # token-index  # feature-index
        o_ptrs = (
                o_ptr
                + (sequence_start_index + token_offset + idx_token) * stride_o_token
                + (idx_feats * stride_o_dim)
        )

        tl.store(o_ptrs, acc, mask=mask_1d)


def causal_conv1d_fn(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        conv_states: torch.Tensor,
        query_start_loc: torch.Tensor,
        cache_indices: torch.Tensor | None = None,
        has_initial_state: torch.Tensor | None = None,
        activation: str | None = "silu",
        pad_slot_id: int = PAD_SLOT_ID,
        block_idx_first_scheduled_token: torch.Tensor | None = None,
        block_idx_last_scheduled_token: torch.Tensor | None = None,
        initial_state_idx: torch.Tensor | None = None,
        num_computed_tokens: torch.Tensor | None = None,
        block_size_to_align=0,
        metadata=None,
        validate_data=False,
):
    """support varlen + continuous batching when x is 2D tensor

    x: (dim,cu_seq_len)
        cu_seq_len = total tokens of all seqs in that batch
        sequences are concatenated from left to right for varlen
    weight: (dim, width)
    conv_states: (...,dim,width - 1) itype
        updated inplace if cache_indices are not provided
        [it use `cache_indices` to get the index to the cache of conv_state for that sequence

        conv_state[cache_indices[i]] for seq-i - to be used as initial_state when has_initial_state[i] = True
             and after that conv_state[cache_indices[i]] need to be shift-left and updated with values from 'x'
        ]
    query_start_loc: (batch + 1) int32
        The cumulative sequence lengths of the sequences in
        the batch, used to index into sequence. prepended by 0.
        if
        x = [5, 1, 1, 1] <- continuous batching (batch=4)
        then
        query_start_loc = [0, 5, 6, 7, 8] <- the starting index of the next sequence; while the last value is
           the ending index of the last sequence
        [length(query_start_loc)-1 == batch]
        for example: query_start_loc = torch.Tensor([0,10,16,17]),
        x.shape=(dim,17)
    cache_indices: (batch)  int32
        indicates the corresponding state index,
        like so: conv_state = conv_states[cache_indices[batch_id]]
    has_initial_state: (batch) bool
        indicates whether should the kernel take the current state as initial
        state for the calculations
        [single boolean for each sequence in the batch: True or False]
    bias: (dim,)
    activation: either None or "silu" or "swish" or True
    pad_slot_id: int
        if cache_indices is passed, lets the kernel identify padded
        entries that will not be processed,
        for example: cache_indices = [pad_slot_id, 1, 20, pad_slot_id]
        in this case, the kernel will not process entries at
        indices 0 and 3
    block_idx_first_scheduled_token: (batch,), dtype int32
        The pointer into cache_indices, where the first cache block to be filled is located.
    block_idx_last_scheduled_token: (batch,), dtype int32
        The pointer into cache_indices, where the last cache block to be filled is located.
    initial_state_idx: (batch,), dtype int32
        The pointer into cache_indices, where the cache block containing the initial state is located.
    num_computed_tokens: (batch,), dtype int32
        The number of tokens already completed for each sequence
    block_size_to_align: int
        The block size to align the cached states to
    out: same shape as `x`
    """
    if isinstance(activation, bool) and activation:
        activation = "silu"

    args = None
    # Store original dtype to cast back at the end
    original_x_dtype = x.dtype
    x = x.to(conv_states.dtype)
    if not HAS_TRITON:
        precompiled = _try_precompiled_causal_conv1d_fn(
            x=x,
            weight=weight,
            bias=bias,
            conv_states=conv_states,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            activation=activation,
            pad_slot_id=pad_slot_id,
        )
        if precompiled is not None:
            return precompiled.to(original_x_dtype)
        logger.warning_once(
            "Mamba causal_conv1d prefill is falling back to the PyTorch "
            "reference path because Triton runtime is unavailable."
        )
        return _causal_conv1d_fn_ref(
            x,
            weight,
            bias,
            conv_states,
            query_start_loc,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            activation=activation,
        ).to(original_x_dtype)
    out = torch.empty_like(x)
    if metadata is not None:
        nums_dict = metadata.nums_dict
        args = nums_dict
        batch_ptr = metadata.batch_ptr
        token_chunk_offset_ptr = metadata.token_chunk_offset_ptr
    else:
        seqlens = query_start_loc.diff().to("cpu")
        args = seqlens
        MAX_NUM_PROGRAMS = 1024

        batch_ptr = torch.full(
            (MAX_NUM_PROGRAMS,), PAD_SLOT_ID, dtype=torch.int32, device=x.device
        )  # tracking which seq-idx the Triton program is handling
        token_chunk_offset_ptr = torch.full(
            (MAX_NUM_PROGRAMS,), PAD_SLOT_ID, dtype=torch.int32, device=x.device
        )  # tracking BLOCK_M-based index in the sequence the Triton program is handling

    is_channel_last = (x.stride(0) == 1) & (x.stride(1) > 1)
    dim, cu_seqlen = x.shape
    _, width = weight.shape
    state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)

    padded_batch = query_start_loc.size(0) - 1
    stride_x_dim = x.stride(0)
    stride_x_token = x.stride(1)
    stride_w_dim = weight.stride(0)
    stride_w_width = weight.stride(1)
    stride_istate_seq = 0
    stride_istate_dim = 0
    stride_istate_token = 0
    num_cache_lines = 0
    BLOCK_M = 8
    if conv_states is not None:
        # extensions to support vLLM:
        # 1. conv_states is used to replaced initial_states
        # 2. conv_states serve as a cache with num cache lines can be larger than batch size
        # 3. mapping from sequence x[idx] to a cache line at index as specified via cache_indices[idx]
        # 4. computation can be skipped if cache_indices[idx] == pad_slot_id
        num_cache_lines = conv_states.size(0)
        assert (
                num_cache_lines == conv_states.shape[0]
                and dim == conv_states.shape[1]
                and width - 1 <= conv_states.shape[2]
        )
        stride_istate_seq = conv_states.stride(0)
        stride_istate_dim = conv_states.stride(1)
        stride_istate_token = conv_states.stride(2)
        assert stride_istate_dim == 1
    if out.dim() == 2:
        stride_o_dim = out.stride(0)
        stride_o_token = out.stride(1)
    else:
        stride_o_dim = out.stride(1)
        stride_o_token = out.stride(2)
    stride_cache_indices = cache_indices.stride(0) if cache_indices is not None else 0

    if validate_data:
        assert x.dim() == 2
        assert query_start_loc is not None
        assert query_start_loc.dim() == 1
        assert x.stride(0) == 1 or x.stride(1) == 1
        if bias is not None:
            assert bias.dim() == 1
            assert dim == bias.size(0)
        if cache_indices is not None:
            assert cache_indices.dim() == 1
            assert padded_batch == cache_indices.size(0)
        if has_initial_state is not None:
            assert has_initial_state.size() == (padded_batch,)
            assert conv_states is not None, (
                "ERROR: `has_initial_state` is used, which needs also `conv_states`"
            )
        assert weight.stride(1) == 1
        assert (dim, width) == weight.shape
        assert is_channel_last, "Need to run in channel-last layout"
        if block_size_to_align is not None and block_size_to_align > 0:
            assert (block_size_to_align % BLOCK_M) == 0, (
                "The mamba block size needs to be divisible by the BLOCK_M"
            )
        else:
            block_size_to_align = BLOCK_M

    if metadata is None:

        def num_program(META, seqlens):
            tot = 0

            mlist = []
            offsetlist = []  # type: ignore

            nums = -(-seqlens // META["BLOCK_M"])

            tot = nums.sum().item()
            mlist = np.repeat(np.arange(len(nums)), nums)
            for idx, num in enumerate(nums):
                offsetlist.extend(
                    range(num)
                )  # chunk-idx if a sequence is split into multiple chunks

            if META["batch_ptr"].nelement() < len(mlist):
                newlen = len(mlist) + 1
                META["batch_ptr"].resize_(newlen).fill_(PAD_SLOT_ID)
                META["token_chunk_offset_ptr"].resize_(newlen).fill_(PAD_SLOT_ID)

            if META["batch_ptr"].nelement() >= len(mlist):
                META["batch_ptr"][0: len(mlist)].copy_(
                    torch.from_numpy(np.array(mlist))
                )
                META["token_chunk_offset_ptr"][0: len(mlist)].copy_(
                    torch.from_numpy(np.array(offsetlist))
                )

            META["batch_ptr"] = META["batch_ptr"].to(META["x_ptr"].device)
            META["token_chunk_offset_ptr"] = META["token_chunk_offset_ptr"].to(
                META["x_ptr"].device
            )
            return tot
    else:

        def num_program(META, nums_dict):
            tot = nums_dict[META["BLOCK_M"]]["tot"]

            mlist = nums_dict[META["BLOCK_M"]]["mlist"]
            mlist_len = nums_dict[META["BLOCK_M"]]["mlist_len"]

            offsetlist = nums_dict[META["BLOCK_M"]]["offsetlist"]

            if nums_dict[META["BLOCK_M"]]["batch_ptr"] is not None:
                META["batch_ptr"] = nums_dict[META["BLOCK_M"]]["batch_ptr"]
                META["token_chunk_offset_ptr"] = nums_dict[META["BLOCK_M"]][
                    "token_chunk_offset_ptr"
                ]
            else:
                if META["batch_ptr"].nelement() < mlist_len:
                    newlen = mlist_len + 1
                    META["batch_ptr"].resize_(newlen).fill_(PAD_SLOT_ID)
                    META["token_chunk_offset_ptr"].resize_(newlen).fill_(PAD_SLOT_ID)

                if META["batch_ptr"].nelement() >= mlist_len:
                    META["batch_ptr"][0:mlist_len].copy_(mlist)
                    META["token_chunk_offset_ptr"][0:mlist_len].copy_(offsetlist)
            return tot

    def grid(META):
        return (
            num_program(META, args),
            triton.cdiv(dim, META["BLOCK_N"]),
        )

    if batch_ptr.device != x.device:
        batch_ptr = batch_ptr.to(x.device)
        token_chunk_offset_ptr = token_chunk_offset_ptr.to(x.device)

    _causal_conv1d_fwd_kernel[grid](
        # Pointers to matrices
        x,
        weight,
        bias,
        conv_states,
        cache_indices,
        has_initial_state,
        query_start_loc,
        batch_ptr,
        token_chunk_offset_ptr,
        block_idx_first_scheduled_token,
        block_idx_last_scheduled_token,
        initial_state_idx,
        num_computed_tokens,
        out,
        # Matrix dimensions
        dim,
        cu_seqlen,
        num_cache_lines,
        # stride
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_cache_indices,
        stride_o_dim,
        stride_o_token,
        block_size_to_align // BLOCK_M,
        # others
        pad_slot_id,
        # META
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_APC_ENABLED=block_idx_last_scheduled_token is not None,
        USE_PAD_SLOT=pad_slot_id is not None,
        NP2_STATELEN=np2_statelen,
        # launch_cooperative_grid=True
        BLOCK_M=BLOCK_M,
        BLOCK_N=256,
        num_stages=2,
    )
    return out.to(original_x_dtype)


@triton.jit()
def _causal_conv1d_update_kernel(
        # Pointers to matrices
        x_ptr,  # (batch, dim, seqlen)
        w_ptr,  # (dim, width)
        bias_ptr,
        conv_state_ptr,
        conv_state_indices_ptr,
        num_accepted_tokens_ptr,
        query_start_loc_ptr,  # (batch + 1)
        block_idx_last_scheduled_token,  # (batch,)
        initial_state_idx,  # (batch,)
        o_ptr,  # (batch, dim, seqlen)
        # Matrix dimensions
        batch: int,
        dim: tl.constexpr,
        seqlen: tl.constexpr,
        state_len: tl.constexpr,
        num_cache_lines: tl.constexpr,  # added to support vLLM larger cache lines
        # Strides
        stride_x_seq: tl.constexpr,
        stride_x_dim: tl.constexpr,
        stride_x_token: tl.constexpr,
        stride_w_dim: tl.constexpr,
        stride_w_width: tl.constexpr,
        stride_conv_state_seq: tl.constexpr,
        stride_conv_state_dim: tl.constexpr,
        stride_conv_state_tok: tl.constexpr,
        stride_state_indices: tl.constexpr,
        stride_o_seq: tl.constexpr,
        stride_o_dim: tl.constexpr,
        stride_o_token: tl.constexpr,
        # others
        pad_slot_id: tl.constexpr,
        # Meta-parameters
        HAS_BIAS: tl.constexpr,
        KERNEL_WIDTH: tl.constexpr,
        SILU_ACTIVATION: tl.constexpr,
        IS_VARLEN: tl.constexpr,
        IS_APC_ENABLED: tl.constexpr,
        IS_SPEC_DECODING: tl.constexpr,
        NP2_STATELEN: tl.constexpr,
        USE_PAD_SLOT: tl.constexpr,
        BLOCK_N: tl.constexpr,
):
    # ruff: noqa: E501
    idx_seq = tl.program_id(0)
    if idx_seq >= batch:
        return

    # [BLOCK_N,] elements along the feature-dimension (channel)
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    if IS_APC_ENABLED:
        # Get the state from the initial_state_idx
        conv_state_init = tl.load(initial_state_idx + idx_seq)
        current_last_index = tl.load(block_idx_last_scheduled_token + idx_seq)
    else:
        conv_state_init = 0
        current_last_index = 0

    # cache_idx
    conv_states_input_coord = tl.load(
        conv_state_indices_ptr + idx_seq * stride_state_indices + conv_state_init
    ).to(tl.int64)

    if USE_PAD_SLOT:  # noqa
        if conv_states_input_coord == pad_slot_id:
            # not processing as this is not the actual sequence
            return

    if IS_VARLEN:
        query_start_index = tl.load(query_start_loc_ptr + idx_seq).to(tl.int64)
        query_end_index = tl.load(query_start_loc_ptr + (idx_seq + 1)).to(tl.int64)
        # revise state_len and seqlen
        state_len = state_len - (seqlen - (query_end_index - query_start_index))
        seqlen = query_end_index - query_start_index
        x_offset = query_start_index * stride_x_token
        o_offset = query_start_index * stride_o_token
    else:
        query_start_index = idx_seq * seqlen
        query_end_index = query_start_index + seqlen
        x_offset = idx_seq * stride_x_seq
        o_offset = idx_seq * stride_o_seq

    if query_start_index == query_end_index:
        return

    if IS_SPEC_DECODING:
        # The rolling of conv state:
        #
        # Before forward, the conv_state is:
        # [history1, history2, ..., historyM].
        #
        # After forward, the conv_state becomes:
        # [history2, ..., historyM, draft1, draft2, ..., draftN].
        #
        # After acceptance, it becomes:
        #
        # - accept 1 tokens: [history2, ..., historyM, draft1]
        # - accept 2 tokens: [history3, ..., historyM, draft1, draft2]
        # - and so on.
        conv_state_token_offset = (
                tl.load(num_accepted_tokens_ptr + idx_seq).to(tl.int64) - 1
        )
    else:
        conv_state_token_offset = 0

    # STEP 1: READ init_state data
    conv_states_base = (
            conv_state_ptr
            + (conv_states_input_coord * stride_conv_state_seq)
            + (idx_feats * stride_conv_state_dim)
    )
    mask_w = idx_feats < dim

    prior_tokens = conv_states_base + conv_state_token_offset * stride_conv_state_tok
    if KERNEL_WIDTH >= 2:
        conv_states_ptrs = prior_tokens  # [BLOCK_N]
        col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 3:
        conv_states_ptrs = prior_tokens + 1 * stride_conv_state_tok  # [BLOCK_N]
        col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 4:
        conv_states_ptrs = prior_tokens + 2 * stride_conv_state_tok  # [BLOCK_N]
        col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 5:
        conv_states_ptrs = prior_tokens + 3 * stride_conv_state_tok  # [BLOCK_N]
        col3 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 6:
        conv_states_ptrs = prior_tokens + 4 * stride_conv_state_tok  # [BLOCK_N]
        col4 = tl.load(conv_states_ptrs, mask_w, 0.0)

    # STEP 2: assume state_len > seqlen
    idx_tokens = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

    # With speculative decoding, the conv_state updates works in a sliding
    # window manner, at each forward pass, the tokens are shift by 1, so we
    # load since idx_tokens + 1.
    conv_state_ptrs_source = (
            conv_state_ptr
            + (conv_states_input_coord * stride_conv_state_seq)
            + conv_state_token_offset * stride_conv_state_tok
            + (idx_feats * stride_conv_state_dim)[None, :]
            + ((idx_tokens + (1 if IS_SPEC_DECODING else seqlen)) * stride_conv_state_tok)[
                :, None
            ]
    )  # [BLOCK_M, BLOCK_N]
    mask = (
            (conv_states_input_coord < num_cache_lines)
            & ((idx_tokens + seqlen) < state_len)[:, None]
            & (idx_feats < dim)[None, :]
    )
    conv_state = tl.load(conv_state_ptrs_source, mask, other=0.0)

    VAL = state_len - seqlen
    x_base = x_ptr + x_offset + (idx_feats * stride_x_dim)  # [BLOCK_N]

    x_ptrs = (
            x_base[None, :] + ((idx_tokens - VAL) * stride_x_token)[:, None]
    )  # [BLOCK_M, BLOCK_N]

    mask_x = (
            (idx_tokens - VAL >= 0)[:, None]
            & (idx_tokens - VAL < seqlen)[:, None]
            & (idx_feats < dim)[None, :]
    )  # token-index  # token-index  # feature-index
    loaded_x = tl.load(x_ptrs, mask_x, 0.0)
    tl.debug_barrier()

    new_conv_state = tl.where(mask, conv_state, loaded_x)

    # Get the state from the initial_state_idx
    # cache_idx
    conv_states_offset = tl.load(
        conv_state_indices_ptr + idx_seq * stride_state_indices + current_last_index
    ).to(tl.int64)
    conv_state_ptrs_target = (
                                     conv_state_ptr
                                     + (conv_states_offset * stride_conv_state_seq)  # Offset from seq
                                     + (idx_feats * stride_conv_state_dim)
                             )[None, :] + (  # [BLOCK_N,]
                                     idx_tokens * stride_conv_state_tok
                             )[:, None]
    mask = (idx_tokens < state_len)[:, None] & (idx_feats < dim)[None, :]
    tl.store(conv_state_ptrs_target, new_conv_state, mask)

    # STEP 3: init accumulator
    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(
            tl.float32
        )  # [BLOCK_N]
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # STEP 4:
    # PRE-LOAD WEIGHTS
    # first kernel column, configured for weights to handle BLOCK_N features in range
    w_base = w_ptr + (idx_feats * stride_w_dim)  # [BLOCK_N,]
    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_ptrs = w_base + (0 * stride_w_width)  # [BLOCK_N] tensor
        w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
        w_ptrs = w_base + (1 * stride_w_width)  # [BLOCK_N] tensor
        w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_ptrs = w_base + (2 * stride_w_width)  # [BLOCK_N] tensor
        w_col2 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_ptrs = w_base + (3 * stride_w_width)  # [BLOCK_N] tensor
        w_col3 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 5:
        w_ptrs = w_base + (4 * stride_w_width)  # [BLOCK_N] tensor
        w_col4 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 6:
        w_ptrs = w_base + (5 * stride_w_width)  # [BLOCK_N] tensor
        w_col5 = tl.load(w_ptrs, mask_w, other=0.0)

    x_base_1d = x_base  # starting of chunk [BLOCK_N]
    mask_x_1d = idx_feats < dim

    # STEP 5: compute each token
    for idx_token in tl.range(seqlen):
        acc = acc_preload

        matrix_w = w_col0
        matrix_x = col0
        for j in tl.static_range(KERNEL_WIDTH):
            if KERNEL_WIDTH == 2:
                if j == 1:  # KERNEL_WIDTH-1:
                    matrix_w = w_col1
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 3:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 4:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 5:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    matrix_x = col3
                elif j == 4:
                    matrix_w = w_col4
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 6:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    matrix_x = col3
                elif j == 4:
                    matrix_w = w_col4
                    matrix_x = col4
                elif j == 5:
                    matrix_w = w_col5
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)

            acc += matrix_x * matrix_w  # [BLOCK_N]

        if KERNEL_WIDTH == 2:
            col0 = matrix_x
        elif KERNEL_WIDTH == 3:
            col0 = col1
            col1 = matrix_x
        elif KERNEL_WIDTH == 4:
            col0 = col1
            col1 = col2
            col2 = matrix_x
        elif KERNEL_WIDTH == 5:
            col0 = col1
            col1 = col2
            col2 = col3
            col3 = matrix_x
        elif KERNEL_WIDTH == 6:
            col0 = col1
            col1 = col2
            col2 = col3
            col3 = col4
            col4 = matrix_x

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < seqlen) & (
                idx_feats < dim
        )  # token-index  # feature-index
        o_ptrs = (
                o_ptr + o_offset + idx_token * stride_o_token + (idx_feats * stride_o_dim)
        )

        tl.store(o_ptrs, acc, mask=mask_1d)


def causal_conv1d_update(
        x: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        activation: bool | str | None = None,
        conv_state_indices: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
        max_query_len: int = -1,
        pad_slot_id: int = PAD_SLOT_ID,
        block_idx_last_scheduled_token: torch.Tensor | None = None,
        initial_state_idx: torch.Tensor | None = None,
        validate_data=False,
):
    """
    对因果 1D 卷积做“增量更新”。

    它既可以处理：
    1. 普通 decode：一次一个 token
    2. 一次多个 token
    3. continuous batching / varlen batching
    4. speculative decoding 下的滑窗状态更新

    参数说明
    ----------


    conv_state:
        卷积缓存状态，形状:
        [..., dim, state_len]
        其中 state_len >= width - 1

        这里最后一维保存“历史 token”的卷积上下文，
        相当于因果卷积需要保留的左侧窗口。

    weight:
        卷积核权重，形状:
        [dim, width]

        可以理解为：每个通道各自有一个长度为 width 的 1D 卷积核。

    bias:
        偏置，形状:
        [dim]

    conv_state_indices:
        状态缓存索引，形状:
        [batch]，dtype=int32

        如果不为 None，表示 conv_state 第 0 维不是和 batch 一一对应，
        而是一个更大的状态池；当前 batch 需要用哪些状态槽位，
        由 conv_state_indices 指定。

        这在 continuous batching 中很常见。

    block_idx_last_scheduled_token:
        形状:
        [batch]，dtype=int32

        指向 conv_state_indices 中“最后一个已调度 token 所在 cache block”的位置。

    initial_state_idx:
        形状:
        [batch]，dtype=int32

        指向 conv_state_indices 中“初始状态所在 cache block”的位置。

    num_accepted_tokens:
        形状:
        [batch]，dtype=int32

        若不为 None，表示 speculative decoding 中每条序列被接受的 token 数。

        此时 conv_state 会以滑动窗口方式更新。

    query_start_loc:
        形状:
        [batch + 1]，dtype=int32

        若不为 None，说明输入 x 采用变长拼接形式，
        该张量保存每条序列在拼接大张量中的起始位置（前缀和格式）。

    max_query_len:
        当 query_start_loc 不为 None 时，表示 batch 中最大序列长度。

    pad_slot_id:
        如果 conv_state_indices 传入，则可用这个特殊值标记“填充槽位”，
        kernel 会跳过这些位置。

        例如:
            conv_state_indices = [pad_slot_id, 1, 20, pad_slot_id]

        则索引 0 和 3 对应的项不会真正参与计算。

    返回
    ----------
    out:
        输出形状与输入 x 相同：

        - [batch, dim]
        - [batch, dim, seqlen]
        - [num_tokens, dim]
    """

    # -------------------------
    # 1. 可选的数据合法性检查
    # -------------------------
    if validate_data:
        # pad_slot_id 必须存在
        assert pad_slot_id is not None

        # 要求 x 在 dim 这一维上是连续的
        assert x.stride(1) == 1

    # -------------------------
    # 2. 统一 activation 参数
    # -------------------------
    if isinstance(activation, bool):
        # True -> "silu"
        # False -> None
        activation = "silu" if activation is True else None
    elif activation is not None:
        # 若传字符串，只允许 "silu" 或 "swish"
        assert activation in ["silu", "swish"]

    # 保存原始 dtype，最后输出时还原
    original_x_dtype = x.dtype

    # 为了和 conv_state 计算兼容，先把 x 转到 conv_state.dtype
    x = x.to(conv_state.dtype)

    # -------------------------
    # 3. 统一输入形状
    # -------------------------

    # x:
    #     输入张量，可有三种形状：
    #
    #     - [batch, dim]
    #       表示每个样本当前只有 1 个 token
    #
    #     - [batch, dim, seqlen]
    #       表示每个样本当前有 seqlen 个 token
    #
    #     - [num_tokens, dim]
    #       表示变长拼接形式，把一个 batch 中所有序列的 token 沿 token 维拼起来
    # 若 x 是 [batch, dim]，且不是 varlen 形式

    # 则把它补成 [batch, dim, 1]
    unsqueeze = query_start_loc is None and x.dim() == 2
    if unsqueeze:
        # 变成单 token 的三维表示
        x = x.unsqueeze(-1)

    # -------------------------
    # 4. 根据输入形式解析 batch / dim / seqlen
    # -------------------------
    if query_start_loc is None:
        # 普通形式：
        # x 形状为 [batch, dim, seqlen]
        batch, dim, seqlen = x.shape
    else:
        # varlen 形式：
        # x 形状为 [num_tokens, dim]
        # batch 由 conv_state_indices 决定
        assert conv_state_indices is not None
        batch = conv_state_indices.size(0)
        dim = x.size(1)
        seqlen = max_query_len

    # 卷积核形状 [dim, width]
    _, width = weight.shape

    # conv_state 形状 [..., dim, state_len]
    num_cache_lines, _, state_len = conv_state.size()

    # -------------------------
    # 5. 更严格的 shape / stride 校验
    # -------------------------
    if validate_data:
        # x 的通道维必须和 weight 的第 0 维一致
        assert dim == weight.size(0)

        # conv_state 在 dim 维上要求连续
        assert conv_state.stride(-2) == 1, (
            f"ERROR: expect contiguous along feat-dim of conv_state (currently stride={conv_state.stride()})"
        )

        # 状态缓存长度至少要够卷积核左窗口
        assert state_len >= width - 1

        # conv_state 的通道维也必须匹配 dim
        assert dim == conv_state.size(1)

        if conv_state_indices is None:
            # 若没有显式索引，则 conv_state 第一维必须覆盖 batch
            assert conv_state.size(0) >= batch
        else:
            # 若有显式索引，则其长度必须等于 batch
            assert batch == conv_state_indices.shape[0], (
                f"ERROR: conv_state_indices should have shape ({batch},*) but got {conv_state_indices.shape}"
            )

        # cache line 数必须足够
        assert num_cache_lines >= batch

        # weight 在 width 维上要求连续
        assert weight.stride(1) == 1

    if not HAS_TRITON:
        precompiled = _try_precompiled_causal_conv1d_update(
            x=x,
            conv_state=conv_state,
            weight=weight,
            bias=bias,
            activation=activation,
            conv_state_indices=conv_state_indices,
            num_accepted_tokens=num_accepted_tokens,
            query_start_loc=query_start_loc,
            pad_slot_id=pad_slot_id,
            block_idx_last_scheduled_token=block_idx_last_scheduled_token,
            initial_state_idx=initial_state_idx,
        )
        if precompiled is not None:
            out = precompiled
            if unsqueeze:
                out = out.squeeze(-1)
            return out.to(original_x_dtype)
        logger.warning_once(
            "Mamba causal_conv1d decode is falling back to the PyTorch "
            "reference path because Triton runtime is unavailable."
        )
        out = _causal_conv1d_update_ref(
            x,
            conv_state,
            weight,
            bias,
            activation=activation,
            conv_state_indices=conv_state_indices,
            num_accepted_tokens=num_accepted_tokens,
            query_start_loc=query_start_loc,
            pad_slot_id=pad_slot_id,
            block_idx_last_scheduled_token=block_idx_last_scheduled_token,
            initial_state_idx=initial_state_idx,
        )
        if unsqueeze:
            out = out.squeeze(-1)
        return out.to(original_x_dtype)

    # -------------------------
    # 6. 输出直接复用 x
    # -------------------------
    # 这里采用“原地覆盖 x”的策略，
    # 而不是额外新建输出张量
    out = x

    # 取 weight 的 stride
    stride_w_dim, stride_w_width = weight.stride()

    # -------------------------
    # 7. 解析 x / out 的 stride
    # -------------------------
    if query_start_loc is None:
        # 普通形式：x 形状 [batch, dim, seqlen]
        stride_x_seq, stride_x_dim, stride_x_token = x.stride()
        stride_o_seq, stride_o_dim, stride_o_token = out.stride()
    else:
        # varlen 形式：x 形状 [num_tokens, dim]
        # 这里“序列维”并不存在，所以 stride_x_seq / stride_o_seq 记为 0
        stride_x_token, stride_x_dim = x.stride()
        stride_x_seq = 0

        stride_o_token, stride_o_dim = out.stride()
        stride_o_seq = 0

    # conv_state 的 stride
    stride_istate_seq, stride_istate_dim, stride_istate_token = conv_state.stride()

    # conv_state_indices 的 stride
    stride_state_indices = (
        conv_state_indices.stride(0) if conv_state_indices is not None else 0
    )

    # -------------------------
    # 8. 计算实际需要的 state_len
    # -------------------------
    if num_accepted_tokens is not None:
        # speculative decoding 下，
        # 需要考虑滑窗更新时的有效状态长度
        state_len = width - 1 + (seqlen - 1)
    else:
        # 普通情况下，只需要保留卷积左窗口
        state_len = width - 1

    # Triton kernel 常需要 2 的幂长度
    np2_statelen = triton.next_power_of_2(state_len)

    # -------------------------
    # 9. 定义 Triton grid
    # -------------------------
    def grid(META):
        return (
            batch,  # 第 0 维：batch 方向
            triton.cdiv(dim, META["BLOCK_N"]),  # 第 1 维：dim 方向分块
        )

    # -------------------------
    # 10. 调 Triton kernel
    # -------------------------
    _causal_conv1d_update_kernel[grid](
        # ---- 输入/输出指针 ----
        x,  # 输入
        weight,  # 卷积核
        bias,  # bias
        conv_state,  # 卷积状态缓存
        conv_state_indices,  # 状态索引
        num_accepted_tokens,  # speculative decoding 接受 token 数
        query_start_loc,  # varlen 边界
        block_idx_last_scheduled_token,
        initial_state_idx,
        out,  # 输出（这里与 x 复用）

        # ---- 尺寸信息 ----
        batch,
        dim,
        seqlen,
        state_len,
        num_cache_lines,

        # ---- x 的 stride ----
        stride_x_seq,
        stride_x_dim,
        stride_x_token,

        # ---- weight 的 stride ----
        stride_w_dim,
        stride_w_width,

        # ---- conv_state 的 stride ----
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,

        # ---- conv_state_indices 的 stride ----
        stride_state_indices,

        # ---- out 的 stride ----
        stride_o_seq,
        stride_o_dim,
        stride_o_token,

        # ---- 其他参数 ----
        pad_slot_id,

        # ---- Triton META 参数 ----
        HAS_BIAS=bias is not None,  # 是否有 bias
        KERNEL_WIDTH=width,  # 卷积核宽度
        SILU_ACTIVATION=activation in ["silu", "swish"],  # 是否启用 silu/swish
        IS_VARLEN=query_start_loc is not None,  # 是否是 varlen 输入
        IS_APC_ENABLED=block_idx_last_scheduled_token is not None,  # 是否启用 APC
        IS_SPEC_DECODING=num_accepted_tokens is not None,  # 是否 speculative decoding
        NP2_STATELEN=np2_statelen,  # state_len 向上取 2 的幂
        USE_PAD_SLOT=pad_slot_id is not None,  # 是否启用 pad slot
        BLOCK_N=256,  # dim 方向的 block 大小
    )

    # -------------------------
    # 11. 若一开始做过 unsqueeze，则恢复形状
    # -------------------------
    if unsqueeze:
        out = out.squeeze(-1)

    # 最后把 dtype 还原成输入原始 dtype
    return out.to(original_x_dtype)
