# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from cfie import _custom_ops
from cfie.v1.sample.rejection_sampler import (
    PLACEHOLDER_TOKEN_ID,
    _rejection_greedy_sample_torch,
    _rejection_random_sample_torch,
)


def _require_cuda_precompiled(op_available: bool) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if not op_available:
        pytest.skip("Precompiled op is not available in this build")


def _sample_recovered_reference(
    cu_num_draft_tokens: torch.Tensor,
    draft_token_ids: torch.Tensor,
    draft_probs: torch.Tensor | None,
    target_probs: torch.Tensor,
    inv_q: torch.Tensor,
) -> torch.Tensor:
    result: list[int] = []
    start_idx = 0
    cu_cpu = cu_num_draft_tokens.cpu().tolist()
    draft_ids_cpu = draft_token_ids.cpu()
    target_probs_cpu = target_probs.cpu()
    draft_probs_cpu = None if draft_probs is None else draft_probs.cpu()
    inv_q_cpu = inv_q.cpu()

    for req_idx, end_idx in enumerate(cu_cpu):
        req_inv_q = inv_q_cpu[req_idx]
        for token_idx in range(start_idx, end_idx):
            if draft_probs_cpu is None:
                probs = target_probs_cpu[token_idx].clone()
                probs[draft_ids_cpu[token_idx].to(torch.long)] = 0
            else:
                probs = torch.clamp(
                    target_probs_cpu[token_idx] - draft_probs_cpu[token_idx], min=0
                )
            result.append(int(torch.argmax(probs * req_inv_q).item()))
        start_idx = end_idx

    return torch.tensor(result, device=target_probs.device, dtype=draft_token_ids.dtype)


def test_expand_batch_to_tokens_precompiled_matches_reference() -> None:
    _require_cuda_precompiled(_custom_ops.has_precompiled_expand_batch_to_tokens())

    x = torch.tensor([0, 3, 5], device="cuda", dtype=torch.int32)
    cu_num_tokens = torch.tensor([2, 5, 6], device="cuda", dtype=torch.int32)

    actual = _custom_ops.expand_batch_to_tokens_precompiled(
        x,
        cu_num_tokens,
        replace_from=0,
        replace_to=7,
    )

    counts = torch.tensor([2, 3, 1], device="cuda", dtype=torch.int32)
    expected = torch.repeat_interleave(x, counts)
    expected = torch.where(expected == 0, expected.new_full((), 7), expected)

    assert torch.equal(actual.cpu(), expected.cpu())


@pytest.mark.parametrize("with_draft_probs", [False, True])
def test_sample_recovered_tokens_precompiled_matches_reference(
    with_draft_probs: bool,
) -> None:
    _require_cuda_precompiled(_custom_ops.has_precompiled_sample_recovered_tokens())

    cu_num_draft_tokens = torch.tensor([2, 3], device="cuda", dtype=torch.int32)
    draft_token_ids = torch.tensor([1, 4, 2], device="cuda", dtype=torch.int32)
    target_probs = torch.tensor(
        [
            [0.10, 0.20, 0.30, 0.05, 0.15, 0.20],
            [0.05, 0.10, 0.05, 0.30, 0.20, 0.30],
            [0.25, 0.05, 0.10, 0.15, 0.20, 0.25],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    draft_probs = (
        torch.tensor(
            [
                [0.02, 0.12, 0.05, 0.01, 0.04, 0.01],
                [0.01, 0.02, 0.01, 0.10, 0.08, 0.06],
                [0.10, 0.01, 0.02, 0.03, 0.04, 0.05],
            ],
            device="cuda",
            dtype=torch.float32,
        )
        if with_draft_probs
        else None
    )
    inv_q = torch.tensor(
        [
            [1.0, 2.0, 0.5, 1.5, 0.8, 1.2],
            [0.7, 1.1, 2.3, 0.6, 1.7, 0.9],
        ],
        device="cuda",
        dtype=torch.float32,
    )

    actual = _custom_ops.sample_recovered_tokens_precompiled(
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        inv_q,
    )
    expected = _sample_recovered_reference(
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        inv_q,
    )

    assert torch.equal(actual.cpu(), expected.cpu())


def test_rejection_greedy_sample_precompiled_matches_reference() -> None:
    _require_cuda_precompiled(_custom_ops.has_precompiled_rejection_greedy_sample())

    max_spec_len = 3
    output_actual = torch.full(
        (2, max_spec_len + 1),
        PLACEHOLDER_TOKEN_ID,
        device="cuda",
        dtype=torch.int32,
    )
    output_expected = output_actual.clone()

    cu_num_draft_tokens = torch.tensor([2, 4], device="cuda", dtype=torch.int32)
    draft_token_ids = torch.tensor([11, 12, 21, 22], device="cuda", dtype=torch.int32)
    target_argmax = torch.tensor([11, 99, 21, 22], device="cuda", dtype=torch.int32)
    bonus_token_ids = torch.tensor([7, 8], device="cuda", dtype=torch.int32)
    is_greedy = torch.tensor([True, False], device="cuda", dtype=torch.bool)

    _custom_ops.rejection_greedy_sample_precompiled(
        output_actual,
        cu_num_draft_tokens,
        draft_token_ids,
        target_argmax,
        bonus_token_ids,
        is_greedy,
        max_spec_len,
    )
    _rejection_greedy_sample_torch(
        output_expected,
        cu_num_draft_tokens,
        draft_token_ids,
        target_argmax,
        bonus_token_ids,
        is_greedy,
        max_spec_len,
    )

    assert torch.equal(output_actual.cpu(), output_expected.cpu())


def test_rejection_random_sample_precompiled_matches_reference() -> None:
    _require_cuda_precompiled(_custom_ops.has_precompiled_rejection_random_sample())

    max_spec_len = 3
    output_actual = torch.full(
        (2, max_spec_len + 1),
        PLACEHOLDER_TOKEN_ID,
        device="cuda",
        dtype=torch.int32,
    )
    output_expected = output_actual.clone()

    cu_num_draft_tokens = torch.tensor([2, 4], device="cuda", dtype=torch.int32)
    draft_token_ids = torch.tensor([1, 3, 2, 4], device="cuda", dtype=torch.int32)
    draft_probs = torch.tensor(
        [
            [0.10, 0.50, 0.20, 0.20, 0.00],
            [0.05, 0.05, 0.20, 0.60, 0.10],
            [0.10, 0.15, 0.45, 0.10, 0.20],
            [0.10, 0.05, 0.05, 0.10, 0.70],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    target_probs = torch.tensor(
        [
            [0.10, 0.60, 0.10, 0.20, 0.00],
            [0.10, 0.10, 0.15, 0.45, 0.20],
            [0.05, 0.20, 0.20, 0.15, 0.40],
            [0.10, 0.20, 0.05, 0.15, 0.50],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    bonus_token_ids = torch.tensor([8, 9], device="cuda", dtype=torch.int32)
    recovered_token_ids = torch.tensor([4, 2, 1, 0], device="cuda", dtype=torch.int32)
    uniform_probs = torch.tensor(
        [0.1, 0.95, 0.2, 0.3],
        device="cuda",
        dtype=torch.float64,
    )
    is_greedy = torch.tensor([False, True], device="cuda", dtype=torch.bool)

    _custom_ops.rejection_random_sample_precompiled(
        output_actual,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        bonus_token_ids,
        recovered_token_ids,
        uniform_probs,
        is_greedy,
        max_spec_len,
    )
    _rejection_random_sample_torch(
        output_expected,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        bonus_token_ids,
        recovered_token_ids,
        uniform_probs,
        is_greedy,
        max_spec_len,
    )

    assert torch.equal(output_actual.cpu(), output_expected.cpu())


def test_eagle_step_update_slot_mapping_precompiled_matches_reference() -> None:
    _require_cuda_precompiled(
        _custom_ops.has_precompiled_eagle_step_update_slot_mapping_and_metadata()
    )

    positions = torch.tensor([0, 4], device="cuda", dtype=torch.int64)
    block_table = torch.tensor([[10, 11], [20, 21]], device="cuda", dtype=torch.int32)
    seq_lens = torch.tensor([1, 5], device="cuda", dtype=torch.int32)
    out_clamped_positions = torch.empty_like(positions)
    out_slot_mapping = torch.empty((4,), device="cuda", dtype=torch.int64)

    _custom_ops.eagle_step_update_slot_mapping_and_metadata_precompiled(
        positions,
        block_table,
        seq_lens,
        block_size=4,
        max_model_len=5,
        out_clamped_positions=out_clamped_positions,
        out_slot_mapping=out_slot_mapping,
        input_batch_size=4,
    )

    assert torch.equal(
        out_clamped_positions.cpu(),
        torch.tensor([1, 0], dtype=torch.int64),
    )
    assert torch.equal(
        out_slot_mapping.cpu(),
        torch.tensor([41, -1, -1, -1], dtype=torch.int64),
    )
    assert torch.equal(seq_lens.cpu(), torch.tensor([2, 1], dtype=torch.int32))


def test_eagle_prepare_inputs_padded_precompiled_matches_reference() -> None:
    _require_cuda_precompiled(_custom_ops.has_precompiled_eagle_prepare_inputs_padded())

    token_indices_to_sample = torch.empty((2,), device="cuda", dtype=torch.int64)
    num_rejected_tokens = torch.empty((2,), device="cuda", dtype=torch.int32)

    _custom_ops.eagle_prepare_inputs_padded_precompiled(
        cu_num_draft_tokens=torch.tensor([2, 5], device="cuda", dtype=torch.int32),
        valid_sampled_tokens_count=torch.tensor(
            [2, 4],
            device="cuda",
            dtype=torch.int32,
        ),
        query_start_loc_gpu=torch.tensor([0, 4, 9], device="cuda", dtype=torch.int32),
        token_indices_to_sample=token_indices_to_sample,
        num_rejected_tokens_gpu=num_rejected_tokens,
    )

    assert torch.equal(
        token_indices_to_sample.cpu(),
        torch.tensor([2, 8], dtype=torch.int64),
    )
    assert torch.equal(
        num_rejected_tokens.cpu(),
        torch.tensor([1, 0], dtype=torch.int32),
    )


def test_eagle_prepare_next_token_padded_precompiled_matches_reference() -> None:
    _require_cuda_precompiled(
        _custom_ops.has_precompiled_eagle_prepare_next_token_padded()
    )

    next_token_ids = torch.empty((2,), device="cuda", dtype=torch.int64)
    valid_sampled_tokens_count = torch.empty((2,), device="cuda", dtype=torch.int32)

    _custom_ops.eagle_prepare_next_token_padded_precompiled(
        sampled_token_ids=torch.tensor(
            [[10, 11, -1], [7, 99, 5]],
            device="cuda",
            dtype=torch.int64,
        ),
        discard_request_mask=torch.tensor([False, True], device="cuda"),
        backup_next_token_ids=torch.tensor([42, 43], device="cuda", dtype=torch.int64),
        next_token_ids=next_token_ids,
        valid_sampled_tokens_count=valid_sampled_tokens_count,
        vocab_size=50,
    )

    assert torch.equal(next_token_ids.cpu(), torch.tensor([11, 43], dtype=torch.int64))
    assert torch.equal(
        valid_sampled_tokens_count.cpu(),
        torch.tensor([2, 0], dtype=torch.int32),
    )


def test_copy_and_expand_eagle_inputs_precompiled_matches_reference() -> None:
    _require_cuda_precompiled(_custom_ops.has_precompiled_copy_and_expand_eagle_inputs())

    out_input_ids = torch.full((8,), -99, device="cuda", dtype=torch.int32)
    out_positions = torch.full((8,), -99, device="cuda", dtype=torch.int64)
    out_is_rejected = torch.ones((8,), device="cuda", dtype=torch.bool)
    out_is_masked = torch.ones((8,), device="cuda", dtype=torch.bool)
    out_new_token_indices = torch.full((4,), -1, device="cuda", dtype=torch.int64)
    out_hidden_state_mapping = torch.full((4,), -1, device="cuda", dtype=torch.int32)

    _custom_ops.copy_and_expand_eagle_inputs_precompiled(
        target_token_ids=torch.tensor([10, 11, 20, 21], device="cuda", dtype=torch.int32),
        target_positions=torch.tensor([0, 1, 5, 6], device="cuda", dtype=torch.int64),
        next_token_ids=torch.tensor([100, 200], device="cuda", dtype=torch.int32),
        out_input_ids=out_input_ids,
        out_positions=out_positions,
        out_is_rejected_token_mask=out_is_rejected,
        out_is_masked_token_mask=out_is_masked,
        out_new_token_indices=out_new_token_indices,
        out_hidden_state_mapping=out_hidden_state_mapping,
        query_start_loc=torch.tensor([0, 2, 4], device="cuda", dtype=torch.int32),
        query_end_loc=torch.tensor([1, 3], device="cuda", dtype=torch.int32),
        padding_token_id=-1,
        parallel_drafting_token_id=-2,
        total_input_tokens=4,
        num_padding_slots_per_request=2,
        shift_input_ids=False,
    )

    assert torch.equal(
        out_input_ids.cpu(),
        torch.tensor([10, 11, 100, -2, 20, 21, 200, -2], dtype=torch.int32),
    )
    assert torch.equal(
        out_positions.cpu(),
        torch.tensor([0, 1, 2, 3, 5, 6, 7, 8], dtype=torch.int64),
    )
    assert torch.equal(out_is_rejected.cpu(), torch.zeros((8,), dtype=torch.bool))
    assert torch.equal(
        out_is_masked.cpu(),
        torch.tensor([False, False, False, True, False, False, False, True]),
    )
    assert torch.equal(
        out_new_token_indices.cpu(),
        torch.tensor([2, 3, 6, 7], dtype=torch.int64),
    )
    assert torch.equal(
        out_hidden_state_mapping.cpu(),
        torch.tensor([-1, -1, -1, -1], dtype=torch.int32),
    )


def test_prepare_eagle_inputs_precompiled_matches_reference() -> None:
    _require_cuda_precompiled(_custom_ops.has_precompiled_prepare_eagle_inputs())

    last_token_indices = torch.empty((2,), device="cuda", dtype=torch.int64)
    eagle_input_ids = torch.zeros((8,), device="cuda", dtype=torch.int32)
    eagle_positions = torch.zeros((8,), device="cuda", dtype=torch.int64)

    _custom_ops.prepare_eagle_inputs_precompiled(
        last_token_indices=last_token_indices,
        eagle_input_ids=eagle_input_ids,
        eagle_positions=eagle_positions,
        target_input_ids=torch.tensor([10, 11, 12, 20, 21], device="cuda", dtype=torch.int32),
        target_positions=torch.tensor([0, 1, 2, 0, 1], device="cuda", dtype=torch.int64),
        idx_mapping=torch.tensor([2, 0], device="cuda", dtype=torch.int32),
        last_sampled=torch.tensor([100, 101, 102], device="cuda", dtype=torch.int32),
        next_prefill_tokens=torch.tensor([200, 201, 202], device="cuda", dtype=torch.int32),
        num_sampled=torch.tensor([1, 0], device="cuda", dtype=torch.int32),
        num_rejected=torch.tensor([1, 0], device="cuda", dtype=torch.int32),
        query_start_loc=torch.tensor([0, 3, 5], device="cuda", dtype=torch.int32),
    )

    assert torch.equal(last_token_indices.cpu(), torch.tensor([1, 4], dtype=torch.int64))
    assert torch.equal(
        eagle_input_ids[:5].cpu(),
        torch.tensor([11, 102, 0, 21, 200], dtype=torch.int32),
    )
    assert torch.equal(
        eagle_positions[:5].cpu(),
        torch.tensor([0, 1, 0, 0, 1], dtype=torch.int64),
    )


def test_prepare_eagle_decode_precompiled_matches_reference() -> None:
    _require_cuda_precompiled(_custom_ops.has_precompiled_prepare_eagle_decode())

    input_ids = torch.zeros((8,), device="cuda", dtype=torch.int32)
    positions = torch.tensor([5, 3, 0, 0], device="cuda", dtype=torch.int64)
    query_start_loc = torch.empty((5,), device="cuda", dtype=torch.int32)
    seq_lens = torch.empty((4,), device="cuda", dtype=torch.int32)
    input_hidden_states = torch.zeros((8, 4), device="cuda", dtype=torch.float32)
    output_hidden_states = torch.arange(20, device="cuda", dtype=torch.float32).reshape(5, 4)

    _custom_ops.prepare_eagle_decode_precompiled(
        draft_tokens=torch.tensor([7, 8], device="cuda", dtype=torch.int64),
        output_hidden_states=output_hidden_states,
        last_token_indices=torch.tensor([1, 4], device="cuda", dtype=torch.int64),
        target_seq_lens=torch.tensor([6, 4], device="cuda", dtype=torch.int32),
        num_rejected=torch.tensor([1, 0], device="cuda", dtype=torch.int32),
        input_ids=input_ids,
        positions=positions,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        input_hidden_states=input_hidden_states,
        max_model_len=16,
        max_num_reqs=4,
    )

    assert torch.equal(query_start_loc.cpu(), torch.tensor([0, 1, 2, 2, 2], dtype=torch.int32))
    assert torch.equal(seq_lens.cpu(), torch.tensor([6, 5, 0, 0], dtype=torch.int32))
    assert torch.equal(input_ids[:2].cpu(), torch.tensor([7, 8], dtype=torch.int32))
    assert torch.equal(positions[:2].cpu(), torch.tensor([6, 4], dtype=torch.int64))
    assert torch.equal(input_hidden_states[0].cpu(), output_hidden_states[1].cpu())
    assert torch.equal(input_hidden_states[1].cpu(), output_hidden_states[4].cpu())


def test_update_eagle_inputs_precompiled_matches_reference() -> None:
    _require_cuda_precompiled(_custom_ops.has_precompiled_update_eagle_inputs())

    input_ids = torch.zeros((4,), device="cuda", dtype=torch.int32)
    positions = torch.tensor([15, 3, 0, 0], device="cuda", dtype=torch.int64)
    seq_lens = torch.tensor([16, 5, 0, 0], device="cuda", dtype=torch.int32)
    hidden_states = torch.zeros((4, 2), device="cuda", dtype=torch.float32)

    _custom_ops.update_eagle_inputs_precompiled(
        draft_tokens=torch.tensor([9, 10], device="cuda", dtype=torch.int64),
        output_hidden_states=torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            device="cuda",
            dtype=torch.float32,
        ),
        input_ids=input_ids,
        positions=positions,
        seq_lens=seq_lens,
        hidden_states=hidden_states,
        max_model_len=16,
    )

    assert torch.equal(input_ids[:2].cpu(), torch.tensor([9, 10], dtype=torch.int32))
    assert torch.equal(positions[:2].cpu(), torch.tensor([15, 4], dtype=torch.int64))
    assert torch.equal(seq_lens[:2].cpu(), torch.tensor([16, 6], dtype=torch.int32))
    assert torch.equal(
        hidden_states[:2].cpu(),
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
    )
