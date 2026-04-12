# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from cfie.config import CfieConfig, replace
from cfie.offload.policy import PLAN_KEY
from cfie.triton_utils import HAS_TRITON, tl, triton
from cfie.v1.attention.backends.utils import (
    CommonAttentionMetadata,
)

PADDING_SLOT_ID = -1


@triton.jit
def eagle_step_slot_mapping_metadata_kernel(
    positions_ptr,  # [batch_size] - current positions (1D view for M-RoPE)
    block_table_ptr,  # [batch_size, n_blocks_per_req]
    block_table_stride,  # stride for block_table dim 1
    seq_lens_ptr,  # [batch_size] - read and write
    out_clamped_positions_ptr,  # [batch_size] (output)
    out_slot_mapping_ptr,  # [input_batch_size] (output)
    block_size: tl.constexpr,
    max_model_len: tl.constexpr,
    n_blocks_per_req: tl.constexpr,
    PAD_ID: tl.constexpr,
    batch_size,
):
    """
    Fused kernel for EAGLE autoregressive step: updates positions, slot mapping,
    and sequence lengths in a single kernel to reduce launch overhead.

    Launched with input_batch_size threads. Threads with req_idx >= batch_size
    are cudagraph padding slots and only write PADDING_SLOT_ID.

    Each real thread handles one request in the batch. Computes:
    - new_position = position + 1, clamped if exceeds max_model_len
    - slot_mapping from block table lookup
    - seq_lens += 1, or 1 if position exceeds max
    """
    req_idx = tl.program_id(0)

    if req_idx >= batch_size:
        tl.store(out_slot_mapping_ptr + req_idx, PAD_ID)
        return

    # Load current position and increment
    position = tl.load(positions_ptr + req_idx)
    new_position = position + 1

    # Check bounds and compute clamped position
    exceeds_max = new_position >= max_model_len
    clamped_position = tl.where(exceeds_max, 0, new_position)

    # Block table lookup: block_number = position // block_size
    # Clamp block_number to avoid OOB when position is at max
    block_number = clamped_position // block_size
    block_number = tl.minimum(block_number, n_blocks_per_req - 1)

    block_id = tl.load(block_table_ptr + req_idx * block_table_stride + block_number)
    slot_id = block_id * block_size + (clamped_position % block_size)
    slot_id = tl.where(exceeds_max, PAD_ID, slot_id)

    # Update seq_lens: +1 normally, or 1 if exceeded
    seq_len = tl.load(seq_lens_ptr + req_idx)
    new_seq_len = tl.where(exceeds_max, 1, seq_len + 1)
    new_seq_len = tl.minimum(new_seq_len, max_model_len)

    # Store outputs
    tl.store(out_clamped_positions_ptr + req_idx, clamped_position)
    tl.store(out_slot_mapping_ptr + req_idx, slot_id)
    tl.store(seq_lens_ptr + req_idx, new_seq_len)


def eagle_step_update_slot_mapping_and_metadata(
    positions_1d: torch.Tensor,
    block_table_tensor: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_model_len: int,
    out_clamped_positions: torch.Tensor,
    out_slot_mapping: torch.Tensor,
    input_batch_size: int | None = None,
) -> None:
    """
    Fused update of slot mapping and metadata for one EAGLE autoregressive step.
    Updates seq_lens in place. Writes to out_clamped_positions and out_slot_mapping.

    When input_batch_size > batch_size, threads beyond batch_size write
    PADDING_SLOT_ID to out_slot_mapping for cudagraph padding.

    Args:
        positions_1d: [batch_size] current positions (use positions[0] for M-RoPE)
        block_table_tensor: [batch_size, n_blocks_per_req]
        seq_lens: [batch_size] updated in place
        block_size: KV cache block size
        max_model_len: max model length for clamping
        out_clamped_positions: [batch_size] output buffer for clamped positions
        out_slot_mapping: [input_batch_size] output buffer for slot mapping
        input_batch_size: total batch size including cudagraph padding;
            defaults to batch_size (no padding)
    """
    batch_size = positions_1d.shape[0]
    if input_batch_size is None:
        input_batch_size = batch_size
    n_blocks_per_req = block_table_tensor.shape[1]

    if not HAS_TRITON:
        new_position = positions_1d + 1
        exceeds_max = new_position >= max_model_len
        clamped_position = torch.where(
            exceeds_max,
            torch.zeros_like(new_position),
            new_position,
        )
        block_number = clamped_position.div(block_size, rounding_mode="floor")
        block_number = torch.clamp(block_number, max=n_blocks_per_req - 1)
        req_idx = torch.arange(batch_size, device=positions_1d.device)
        block_id = block_table_tensor[req_idx, block_number]
        slot_id = block_id * block_size + torch.remainder(
            clamped_position, block_size
        )
        padding_slot_ids = torch.full_like(slot_id, PADDING_SLOT_ID)
        slot_id = torch.where(exceeds_max, padding_slot_ids, slot_id)

        new_seq_len = torch.where(exceeds_max, torch.ones_like(seq_lens), seq_lens + 1)
        new_seq_len.clamp_(max=max_model_len)

        out_clamped_positions[:batch_size].copy_(clamped_position)
        out_slot_mapping[:batch_size].copy_(slot_id)
        if input_batch_size > batch_size:
            out_slot_mapping[batch_size:input_batch_size].fill_(PADDING_SLOT_ID)
        seq_lens.copy_(new_seq_len)
        return

    eagle_step_slot_mapping_metadata_kernel[(input_batch_size,)](
        positions_1d,
        block_table_tensor,
        block_table_tensor.stride(0),
        seq_lens,
        out_clamped_positions,
        out_slot_mapping,
        block_size=block_size,
        max_model_len=max_model_len,
        n_blocks_per_req=n_blocks_per_req,
        PAD_ID=PADDING_SLOT_ID,
        batch_size=batch_size,
    )


@triton.jit
def eagle_prepare_inputs_padded_kernel(
    cu_num_draft_tokens_ptr,  # [num_reqs]
    valid_sampled_tokens_count_ptr,  # [num_reqs]
    query_start_loc_gpu_ptr,  # [num_reqs + 1]
    token_indices_to_sample_ptr,  # [num_reqs] (output)
    num_rejected_tokens_gpu_ptr,  # [num_reqs] (output)
    num_reqs,  # tl.int32
):
    """
    Fused kernel for Eagle prepare_input_padded. This kernel computes the
    token index to sample for each request, taking into account the number
    of draft tokens and the number of valid sampled tokens (which is one more than
    the number of accepted tokens).
    """
    req_idx = tl.program_id(axis=0)
    if req_idx >= num_reqs:
        return

    # Calculate num_draft_tokens from cu_num_draft_tokens, which is an inclusive
    # cumulative sum (first entry is the first value, not zero).
    cu_draft_curr = tl.load(cu_num_draft_tokens_ptr + req_idx)

    num_draft_tokens = 0
    if req_idx == 0:
        num_draft_tokens = cu_draft_curr
    else:
        cu_draft_prev = tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
        num_draft_tokens = cu_draft_curr - cu_draft_prev

    valid_count = tl.load(valid_sampled_tokens_count_ptr + req_idx)
    num_rejected_tokens = num_draft_tokens + 1 - valid_count
    num_rejected_tokens = tl.where(num_draft_tokens > 0, num_rejected_tokens, 0)

    # query_start_loc[req_idx + 1] is the start position of the next request,
    # which is one past the last token of this request.
    q_last_tok_idx = tl.load(query_start_loc_gpu_ptr + req_idx + 1) - 1

    index_to_sample = q_last_tok_idx - num_rejected_tokens
    tl.store(token_indices_to_sample_ptr + req_idx, index_to_sample)
    tl.store(num_rejected_tokens_gpu_ptr + req_idx, num_rejected_tokens)


@triton.jit
def eagle_prepare_next_token_padded_kernel(
    sampled_token_ids_ptr,  # [num_reqs, num_sampled_tokens_per_req]
    discard_request_mask_ptr,  # [num_reqs]
    backup_next_token_ids_ptr,  # [num_reqs]
    next_token_ids_ptr,  # [num_reqs] (output)
    valid_sampled_tokens_count_ptr,  # [num_reqs] (output)
    vocab_size,  # tl.int32
    num_sampled_tokens_per_req,  # tl.int32 (num_spec_tokens + 1)
    num_reqs,  # tl.int32
    stride_sampled_token_ids,  # tl.int32 (stride for dim 0)
    BLOCK_SIZE_TOKENS: tl.constexpr,  # Power-of-2 >= num_sampled_tokens_per_req
):
    """
    Fused kernel for Eagle prepare_next_token_ids_padded. This kernel computes the
    number of valid (1 + accepted) tokens for each request, and the corresponding
    "next" token id to sample from during speculative decoding. This is the
    "last accepted token" from the sampled tokens, or the backup token if no
    tokens were accepted or if the request is marked as discarded.
    """
    req_idx = tl.program_id(axis=0)
    if req_idx >= num_reqs:
        return

    # Check if this request is discarded.
    is_discarded = tl.load(discard_request_mask_ptr + req_idx)

    if is_discarded:
        backup_token = tl.load(backup_next_token_ids_ptr + req_idx)
        valid_count = tl.full((), 0, dtype=tl.uint32)
        tl.store(next_token_ids_ptr + req_idx, backup_token)
        tl.store(valid_sampled_tokens_count_ptr + req_idx, valid_count)
    else:
        # Count the number of valid tokens among the sampled tokens.
        token_offs = tl.arange(0, BLOCK_SIZE_TOKENS)
        token_mask = token_offs < num_sampled_tokens_per_req

        row_ptr = sampled_token_ids_ptr + req_idx * stride_sampled_token_ids
        token_ids = tl.load(row_ptr + token_offs, mask=token_mask, other=-1)

        # Rejected tokens are -1, valid tokens are in [0, vocab_size)
        is_valid_mask = (token_ids != -1) & (token_ids < vocab_size) & token_mask
        valid_count = tl.sum(is_valid_mask)

        if valid_count > 0:
            # Guaranteed to be well-defined since
            # valid_count > 0 implies is_valid_mask is not empty
            last_valid_index = tl.max(tl.where(is_valid_mask, token_offs, -1))

            # Select the token at that index, using a sum trick since
            # we don't want to load again to access token_ids[last_valid_index].
            last_valid_token = tl.sum(
                tl.where(token_offs == last_valid_index, token_ids, 0)
            )
            tl.store(next_token_ids_ptr + req_idx, last_valid_token)
        else:
            # No valid tokens found, use backup token
            backup_token = tl.load(backup_next_token_ids_ptr + req_idx)
            tl.store(next_token_ids_ptr + req_idx, backup_token)

        tl.store(valid_sampled_tokens_count_ptr + req_idx, valid_count)


def eagle_prepare_inputs_padded(
    cu_num_draft_tokens: torch.Tensor,
    valid_sampled_tokens_count: torch.Tensor,
    query_start_loc_gpu: torch.Tensor,
    token_indices_to_sample: torch.Tensor,
    num_rejected_tokens_gpu: torch.Tensor,
) -> None:
    num_reqs = valid_sampled_tokens_count.shape[0]

    if not HAS_TRITON:
        num_draft_tokens = torch.empty_like(valid_sampled_tokens_count)
        if num_reqs > 0:
            num_draft_tokens[0] = cu_num_draft_tokens[0]
        if num_reqs > 1:
            num_draft_tokens[1:] = cu_num_draft_tokens[1:] - cu_num_draft_tokens[:-1]

        num_rejected_tokens = (
            num_draft_tokens + 1 - valid_sampled_tokens_count
        ).to(num_rejected_tokens_gpu.dtype)
        num_rejected_tokens = torch.where(
            num_draft_tokens > 0,
            num_rejected_tokens,
            torch.zeros_like(num_rejected_tokens),
        )
        index_to_sample = (query_start_loc_gpu[1:] - 1 - num_rejected_tokens).to(
            token_indices_to_sample.dtype
        )

        token_indices_to_sample.copy_(index_to_sample)
        num_rejected_tokens_gpu.copy_(num_rejected_tokens)
        return

    grid = (num_reqs,)
    eagle_prepare_inputs_padded_kernel[grid](
        cu_num_draft_tokens,
        valid_sampled_tokens_count,
        query_start_loc_gpu,
        token_indices_to_sample,
        num_rejected_tokens_gpu,
        num_reqs,
    )


def eagle_prepare_next_token_padded(
    sampled_token_ids: torch.Tensor,
    discard_request_mask: torch.Tensor,
    backup_next_token_ids: torch.Tensor,
    next_token_ids: torch.Tensor,
    valid_sampled_tokens_count: torch.Tensor,
    vocab_size: int,
) -> None:
    batch_size, num_sampled_tokens_per_req = sampled_token_ids.shape

    if not HAS_TRITON:
        valid_mask = (sampled_token_ids != -1) & (sampled_token_ids < vocab_size)
        valid_count = valid_mask.sum(dim=1).to(valid_sampled_tokens_count.dtype)

        token_offsets = torch.arange(
            num_sampled_tokens_per_req,
            device=sampled_token_ids.device,
            dtype=torch.int64,
        ).expand(batch_size, -1)
        last_valid_index = torch.where(
            valid_mask,
            token_offsets,
            token_offsets.new_full(token_offsets.shape, -1),
        ).amax(dim=1)
        gather_index = last_valid_index.clamp_min(0).unsqueeze(1)
        last_valid_token = sampled_token_ids.gather(1, gather_index).squeeze(1)
        next_token = torch.where(
            valid_count > 0,
            last_valid_token.to(backup_next_token_ids.dtype),
            backup_next_token_ids,
        )

        next_token = torch.where(discard_request_mask, backup_next_token_ids, next_token)
        valid_count = torch.where(
            discard_request_mask,
            torch.zeros_like(valid_count),
            valid_count,
        )

        next_token_ids.copy_(next_token.to(next_token_ids.dtype))
        valid_sampled_tokens_count.copy_(valid_count)
        return

    grid = (batch_size,)
    block_size_tokens = triton.next_power_of_2(num_sampled_tokens_per_req)
    eagle_prepare_next_token_padded_kernel[grid](
        sampled_token_ids,
        discard_request_mask,
        backup_next_token_ids,
        next_token_ids,
        valid_sampled_tokens_count,
        vocab_size,
        num_sampled_tokens_per_req,
        batch_size,
        sampled_token_ids.stride(0),
        BLOCK_SIZE_TOKENS=block_size_tokens,
    )


def compute_new_slot_mapping(
    cad: CommonAttentionMetadata,
    new_positions: torch.Tensor,
    is_rejected_token_mask: torch.Tensor,
    block_size: int,
    num_new_tokens: int,
    max_model_len: int,
):
    batch_size, n_blocks_per_req = cad.block_table_tensor.shape
    req_indices = torch.arange(batch_size, device=cad.query_start_loc.device)
    req_indices = torch.repeat_interleave(
        req_indices,
        cad.naive_query_lens() + num_new_tokens,
        output_size=len(new_positions),
    )
    # Clamp the positions to prevent an out-of-bounds error when indexing
    # into block_table_tensor.
    clamped_positions = torch.clamp(new_positions, max=max_model_len - 1)
    block_table_indices = (
        req_indices * n_blocks_per_req + clamped_positions // block_size
    )
    block_nums = cad.block_table_tensor.view(-1)[block_table_indices]
    block_offsets = clamped_positions % block_size
    new_slot_mapping = block_nums * block_size + block_offsets
    # Mask out the position ids that exceed the max model length.
    exceeds_max_model_len = new_positions >= max_model_len
    new_slot_mapping.masked_fill_(exceeds_max_model_len, PADDING_SLOT_ID)
    # Mask out rejected tokens to prevent saves to the KV cache.
    new_slot_mapping.masked_fill_(is_rejected_token_mask, PADDING_SLOT_ID)
    return new_slot_mapping


def create_vllm_config_for_draft_model(
    target_model_vllm_config: CfieConfig,
    additional_config_updates: dict[str, object] | None = None,
) -> CfieConfig:
    """
    当前传入的 cfie_config 是面向 target 模型构造的，
    其中已经包含 target 模型对应的量化配置和并行配置等信息。

    但 draft 模型可能采用不同的量化方式，
    也可能具有不同的张量并行规模。

    因此，这个函数会基于 target 模型配置派生出一份新的 cfie_config，
    并将其调整为适用于 draft 模型的配置。

    这份新的 cfie_config 主要用于后续通过 get_model() 加载 draft 模型。
    """
    # ------------------------------- 基于 target 配置派生 draft 模型配置 -------------------------------
    # 将当前传入的 target 模型配置保存为旧配置对象，后续所有 draft 配置都从它派生。
    old = target_model_vllm_config

    # 断言当前 target 配置已经构造出 speculative_config，否则无法继续派生 draft 配置。
    assert old.speculative_config is not None, "speculative_config is not set"

    # 读取 target 配置上挂载的 speculative_config，其中包含 draft 模型与并行相关配置。
    old_spec_config = old.speculative_config

    # ------------------------------- 派生 draft 专属的并行配置 -------------------------------
    # 基于 speculative_config 中的 draft 并行配置复制出一份新配置，并强制沿用 target 当前进程的 rank。
    new_parallel_config = replace(
        old_spec_config.draft_parallel_config,
        rank=old.parallel_config.rank,
    )

    # ------------------------------- 复制并清理 additional_config -------------------------------
    # 先读取旧配置中的 additional_config，后续按需复制并清理其中不应继承到 draft 的字段。
    new_additional_config = getattr(old, "additional_config", None)

    # 仅当 additional_config 实际为字典时，才执行浅拷贝与字段清理。
    if isinstance(new_additional_config, dict):
        # 复制一份字典，避免后续修改 draft 配置时反向污染 target 配置。
        new_additional_config = dict(new_additional_config)

        # 删除 target 配置中已有的计划字段，避免 draft 误继承 target 的 MoE 驻留计划。
        new_additional_config.pop(PLAN_KEY, None)

        # 当调用方额外提供配置更新项时，将这些更新项写入新的 additional_config。
        if additional_config_updates:
            # 将调用方提供的附加配置更新合并到新的 additional_config 中。
            new_additional_config.update(additional_config_updates)

    # ------------------------------- 构造 draft 专属的 CfieConfig -------------------------------
    # 基于旧配置复制出一份新的 CfieConfig，并替换成 draft 模型所需的关键子配置。
    new: CfieConfig = replace(
        old,
        # 清空旧的 quant_config，使 draft 配置在后续初始化过程中按自身模型重新解析量化配置。
        quant_config=None,

        # 使用为 draft 模型派生出的并行配置。
        parallel_config=new_parallel_config,

        # 将 model_config 切换为 speculative_config 中保存的 draft 模型配置。
        model_config=old_spec_config.draft_model_config,

        # 使用清理后的 additional_config，避免 target 的旧计划污染 draft 配置。
        additional_config=new_additional_config,
    )

    # ------------------------------- 返回 draft 模型配置对象 -------------------------------
    # 返回面向 draft 模型构造完成的新 CfieConfig。
    return new


def extend_all_queries_by_N(
    common_attn_metadata: CommonAttentionMetadata,
    N: int,
    arange: torch.Tensor,
    new_slot_mapping: torch.Tensor,
) -> CommonAttentionMetadata:
    """
    Creates a new CommonAttentionMetadata with all query lengths increased by N.
    Also all seq lens are increased by N.
    This is useful e.g. in speculative decoding with parallel drafting, where we
    extend each sequence by N tokens and predict all tokens in one pass.
    The slot mapping is computed externally, as it requires more information.
    """
    cad = common_attn_metadata
    # query start loc must be increased by [+0, +N, +2N, ..., +batch_size * N]
    new_query_start_loc = cad.query_start_loc + N * arange[: len(cad.query_start_loc)]
    new_query_start_loc_cpu = cad.query_start_loc_cpu + N * torch.arange(
        len(cad.query_start_loc_cpu), dtype=torch.int32
    )
    new_cad = cad.replace(
        query_start_loc=new_query_start_loc,
        query_start_loc_cpu=new_query_start_loc_cpu,
        seq_lens=cad.seq_lens + N,
        # each request is extended by N tokens -> batch_size * N tokens are added
        num_actual_tokens=cad.num_actual_tokens + cad.batch_size() * N,
        # All query lens increase by N, so max query len increases by N
        max_query_len=cad.max_query_len + N,
        max_seq_len=cad.max_seq_len + N,
        slot_mapping=new_slot_mapping,
    )
    return new_cad


def copy_and_expand_eagle_inputs(
    target_token_ids_ptr: torch.Tensor,
    target_positions_ptr: torch.Tensor,
    next_token_ids_ptr: torch.Tensor,
    out_input_ids_ptr: torch.Tensor,
    out_positions_ptr: torch.Tensor,
    out_is_rejected_token_mask_ptr: torch.Tensor,
    out_is_masked_token_mask_ptr: torch.Tensor,
    out_new_token_indices_ptr: torch.Tensor,
    out_hidden_state_mapping_ptr: torch.Tensor,
    query_start_loc_ptr: torch.Tensor,
    query_end_loc_ptr: torch.Tensor,
    padding_token_id: int,
    parallel_drafting_token_id: int,
    total_input_tokens: int,
    num_padding_slots_per_request: int,
    shift_input_ids: bool,
) -> None:
    batch_size = query_end_loc_ptr.shape[0]

    if not HAS_TRITON:
        last_input_index = total_input_tokens - 1
        for request_idx in range(batch_size):
            query_start_loc = int(query_start_loc_ptr[request_idx].item())
            next_query_start_loc = int(query_start_loc_ptr[request_idx + 1].item())
            query_end_loc = int(query_end_loc_ptr[request_idx].item())

            if shift_input_ids:
                num_valid_tokens = query_end_loc - query_start_loc
                input_offset = 1
                output_start = query_start_loc + request_idx * (
                    num_padding_slots_per_request - 1
                )
            else:
                num_valid_tokens = query_end_loc - query_start_loc + 1
                input_offset = 0
                output_start = query_start_loc + request_idx * (
                    num_padding_slots_per_request
                )

            num_rejected = next_query_start_loc - query_end_loc - 1
            total_output_tokens = (
                num_valid_tokens + num_padding_slots_per_request + num_rejected
            )
            j = torch.arange(
                total_output_tokens,
                device=target_token_ids_ptr.device,
                dtype=torch.int64,
            )
            out_idx = output_start + j

            is_valid_region = j < num_valid_tokens
            is_bonus_region = j == num_valid_tokens
            is_parallel_draft_region = (j > num_valid_tokens) & (
                j < num_valid_tokens + num_padding_slots_per_request
            )
            is_rejected_region = j >= (
                num_valid_tokens + num_padding_slots_per_request
            )

            in_idx = query_start_loc + input_offset + j
            in_idx_clamped = torch.clamp(in_idx, max=last_input_index)
            token_ids = torch.where(
                is_valid_region,
                target_token_ids_ptr[in_idx_clamped],
                torch.zeros_like(in_idx_clamped, dtype=target_token_ids_ptr.dtype),
            )

            if target_positions_ptr.ndim == 1:
                start_pos = target_positions_ptr[query_start_loc]
            else:
                start_pos = target_positions_ptr[0, query_start_loc]
            positions = start_pos + j.to(out_positions_ptr.dtype)

            bonus_token = next_token_ids_ptr[request_idx]
            token_ids = torch.where(
                is_bonus_region,
                torch.full_like(token_ids, bonus_token),
                token_ids,
            )
            token_ids = torch.where(
                is_parallel_draft_region,
                torch.full_like(token_ids, parallel_drafting_token_id),
                token_ids,
            )
            token_ids = torch.where(
                is_rejected_region,
                torch.full_like(token_ids, padding_token_id),
                token_ids,
            )
            positions = torch.where(
                is_rejected_region,
                torch.zeros_like(positions),
                positions,
            )

            out_input_ids_ptr[out_idx] = token_ids.to(out_input_ids_ptr.dtype)
            out_positions_ptr[out_idx] = positions.to(out_positions_ptr.dtype)
            out_is_rejected_token_mask_ptr[out_idx] = is_rejected_region.to(
                out_is_rejected_token_mask_ptr.dtype
            )
            out_is_masked_token_mask_ptr[out_idx] = is_parallel_draft_region.to(
                out_is_masked_token_mask_ptr.dtype
            )

            is_new_token_region = (j >= num_valid_tokens) & (
                j < num_valid_tokens + num_padding_slots_per_request
            )
            if is_new_token_region.any():
                new_token_local_idx = j[is_new_token_region] - num_valid_tokens
                new_token_out_idx = (
                    request_idx * num_padding_slots_per_request + new_token_local_idx
                )
                out_new_token_indices_ptr[new_token_out_idx] = out_idx[
                    is_new_token_region
                ].to(out_new_token_indices_ptr.dtype)

            if shift_input_ids:
                num_input_tokens_this_request = next_query_start_loc - query_start_loc
                src_idx = torch.arange(
                    query_start_loc,
                    next_query_start_loc,
                    device=target_token_ids_ptr.device,
                    dtype=torch.int64,
                )
                mapped_out_idx = out_idx[:num_input_tokens_this_request]
                out_hidden_state_mapping_ptr[src_idx] = mapped_out_idx.to(
                    out_hidden_state_mapping_ptr.dtype
                )
        return

    max_num_tokens_per_request = 0
    if batch_size > 0:
        max_num_tokens_per_request = int(
            (query_start_loc_ptr[1:] - query_start_loc_ptr[:-1]).max().item()
        ) + num_padding_slots_per_request
    block_size_tokens = min(256, triton.next_power_of_2(max_num_tokens_per_request))
    num_blocks = (
        max_num_tokens_per_request + block_size_tokens - 1
    ) // block_size_tokens
    grid = (batch_size, num_blocks)
    copy_and_expand_eagle_inputs_kernel[grid](
        target_token_ids_ptr=target_token_ids_ptr,
        target_positions_ptr=target_positions_ptr,
        next_token_ids_ptr=next_token_ids_ptr,
        out_input_ids_ptr=out_input_ids_ptr,
        out_positions_ptr=out_positions_ptr,
        out_is_rejected_token_mask_ptr=out_is_rejected_token_mask_ptr,
        out_is_masked_token_mask_ptr=out_is_masked_token_mask_ptr,
        out_new_token_indices_ptr=out_new_token_indices_ptr,
        out_hidden_state_mapping_ptr=out_hidden_state_mapping_ptr,
        query_start_loc_ptr=query_start_loc_ptr,
        query_end_loc_ptr=query_end_loc_ptr,
        padding_token_id=padding_token_id,
        parallel_drafting_token_id=parallel_drafting_token_id,
        total_input_tokens=total_input_tokens,
        num_padding_slots_per_request=num_padding_slots_per_request,
        shift_input_ids=shift_input_ids,
        BLOCK_SIZE_TOKENS=block_size_tokens,
    )


# Unified copy/expand kernel
@triton.jit
def copy_and_expand_eagle_inputs_kernel(
    # (Padded) Inputs from the target model
    target_token_ids_ptr,  # [total_tokens_in_batch]
    target_positions_ptr,  # [total_tokens_in_batch]
    next_token_ids_ptr,  # [num_reqs]
    # Outputs to the drafting buffers
    out_input_ids_ptr,  # [total_draft_tokens_in_batch] (output)
    out_positions_ptr,  # [total_draft_tokens_in_batch] (output)
    out_is_rejected_token_mask_ptr,  # [total_draft_tokens_in_batch] (output)
    out_is_masked_token_mask_ptr,  # [total_draft_tokens_in_batch] (output)
    out_new_token_indices_ptr,  # [num_padding_slots_per_request * num_reqs] (output)
    out_hidden_state_mapping_ptr,  # [total_tokens_in_batch]
    # Input metadata
    query_start_loc_ptr,  # [num_reqs + 1], last value is the total num input tokens
    query_end_loc_ptr,  # [num_reqs]
    padding_token_id,  # tl.int32
    parallel_drafting_token_id,  # tl.int32
    # Sizing info
    total_input_tokens,  # tl.int32
    num_padding_slots_per_request,  # tl.int32
    shift_input_ids,  # tl.bool
    BLOCK_SIZE_TOKENS: tl.constexpr,  # Blocks along token dim to handle prefills
):
    """
    Copy and expand inputs from the target model to the drafting buffers for Eagle
    speculative decoding. This kernel handles padding slots and parallel drafting
    tokens, if enabled.
    """
    request_idx = tl.program_id(axis=0)
    token_batch_idx = tl.program_id(axis=1)

    # Load query locations
    query_start_loc = tl.load(query_start_loc_ptr + request_idx)
    next_query_start_loc = tl.load(query_start_loc_ptr + request_idx + 1)
    query_end_loc = tl.load(query_end_loc_ptr + request_idx)

    # Calculate number of valid tokens to copy and input offset
    # With shift_input_ids=True, we skip the first token
    # Output layout: each request gets (input_len + num_padding_slots_per_request) slots
    # But with shift, we lose one token per request
    if shift_input_ids:
        num_valid_tokens = query_end_loc - query_start_loc
        input_offset = 1
        output_start = query_start_loc + request_idx * (
            num_padding_slots_per_request - 1
        )
    else:
        num_valid_tokens = query_end_loc - query_start_loc + 1
        input_offset = 0
        output_start = query_start_loc + request_idx * num_padding_slots_per_request

    # Number of rejected tokens from previous speculation
    num_rejected = next_query_start_loc - query_end_loc - 1

    # Total output tokens for this request
    total_output_tokens = (
        num_valid_tokens + num_padding_slots_per_request + num_rejected
    )

    # Process tokens in this block
    j = token_batch_idx * BLOCK_SIZE_TOKENS + tl.arange(0, BLOCK_SIZE_TOKENS)

    # Compute masks for different output regions:
    # [0, num_valid_tokens): valid tokens copied from input
    # [num_valid_tokens]: bonus token from next_token_ids
    # (num_valid_tokens, num_valid_tokens + num_padding_slots_per_request):
    #     parallel drafting slots
    # [num_valid_tokens + num_padding_slots_per_request, total_output_tokens):
    #     rejected slots
    in_bounds = j < total_output_tokens
    is_valid_region = j < num_valid_tokens
    is_bonus_region = j == num_valid_tokens
    is_parallel_draft_region = (j > num_valid_tokens) & (
        j < num_valid_tokens + num_padding_slots_per_request
    )
    is_rejected_region = j >= num_valid_tokens + num_padding_slots_per_request

    # Compute output indices
    out_idx = output_start + j

    # For valid tokens, compute input index
    in_idx = query_start_loc + input_offset + j
    # Clamp to avoid out-of-bounds access (masked loads still need valid addresses)
    in_idx_clamped = tl.minimum(in_idx, total_input_tokens - 1)

    # Load input tokens (masked to valid region)
    token_ids = tl.load(
        target_token_ids_ptr + in_idx_clamped, mask=is_valid_region & in_bounds, other=0
    )

    # Load the starting position for this request (first position in the sequence)
    start_pos = tl.load(target_positions_ptr + query_start_loc)

    # Load bonus token for this request
    bonus_token = tl.load(next_token_ids_ptr + request_idx)

    # Build final token_ids based on region
    token_ids = tl.where(is_bonus_region, bonus_token, token_ids)
    token_ids = tl.where(
        is_parallel_draft_region, parallel_drafting_token_id, token_ids
    )
    token_ids = tl.where(is_rejected_region, padding_token_id, token_ids)

    # Build final positions:
    # Positions are NOT shifted - they start from the first input position and increment
    # Output position j gets start_pos + j
    # (e.g., input positions [5,6,7] -> output [5,6,7,8,9,...])
    positions = start_pos + j
    # Rejected positions are don't-care, set to 0
    positions = tl.where(is_rejected_region, 0, positions)

    # Compute output masks
    is_rejected_out = is_rejected_region & in_bounds
    is_masked_out = is_parallel_draft_region & in_bounds

    # Compute indices of new tokens (bonus + parallel drafting) for sampling
    # New tokens are at positions
    #     [num_valid_tokens, num_valid_tokens + num_padding_slots_per_request)
    is_new_token_region = (j >= num_valid_tokens) & (
        j < num_valid_tokens + num_padding_slots_per_request
    )
    new_token_local_idx = (
        j - num_valid_tokens
    )  # 0 for bonus, 1, 2, ... for parallel drafting
    new_token_out_idx = (
        request_idx * num_padding_slots_per_request + new_token_local_idx
    )

    # Compute hidden state mapping (source index -> destination index)
    # This maps each input position to its corresponding output position
    # Hidden states don't get shifted, so we map all input tokens (including rejected)
    if shift_input_ids:
        num_input_tokens_this_request = next_query_start_loc - query_start_loc
        is_input_region = j < num_input_tokens_this_request
        src_idx = query_start_loc + j
        tl.store(out_hidden_state_mapping_ptr + src_idx, out_idx, mask=is_input_region)

    # Store outputs
    tl.store(out_input_ids_ptr + out_idx, token_ids, mask=in_bounds)
    tl.store(out_positions_ptr + out_idx, positions, mask=in_bounds)
    tl.store(out_is_rejected_token_mask_ptr + out_idx, is_rejected_out, mask=in_bounds)
    tl.store(out_is_masked_token_mask_ptr + out_idx, is_masked_out, mask=in_bounds)
    tl.store(
        out_new_token_indices_ptr + new_token_out_idx,
        out_idx,
        mask=is_new_token_region & in_bounds,
    )
