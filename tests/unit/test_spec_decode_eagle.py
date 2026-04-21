from __future__ import annotations

import torch

from cfie.model_executor.layers.fla.ops.fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update,
)
from cfie.v1.spec_decode.utils import (
    PADDING_SLOT_ID,
    copy_and_expand_eagle_inputs,
    eagle_prepare_inputs_padded,
    eagle_prepare_next_token_padded,
    eagle_step_update_slot_mapping_and_metadata,
)


def test_eagle_prepare_next_token_padded_torch_fallback(monkeypatch) -> None:
    monkeypatch.setattr("cfie.v1.spec_decode.utils.HAS_TRITON", False)

    sampled_token_ids = torch.tensor(
        [[11, 22, -1], [-1, -1, -1], [7, 8, 9]],
        dtype=torch.int32,
    )
    discard_request_mask = torch.tensor([False, False, True], dtype=torch.bool)
    backup_next_token_ids = torch.tensor([101, 102, 103], dtype=torch.int32)
    next_token_ids = torch.empty(3, dtype=torch.int32)
    valid_sampled_tokens_count = torch.empty(3, dtype=torch.int32)

    eagle_prepare_next_token_padded(
        sampled_token_ids,
        discard_request_mask,
        backup_next_token_ids,
        next_token_ids,
        valid_sampled_tokens_count,
        vocab_size=100,
    )

    assert next_token_ids.tolist() == [22, 102, 103]
    assert valid_sampled_tokens_count.tolist() == [2, 0, 0]


def test_eagle_prepare_inputs_padded_torch_fallback(monkeypatch) -> None:
    monkeypatch.setattr("cfie.v1.spec_decode.utils.HAS_TRITON", False)

    cu_num_draft_tokens = torch.tensor([2, 5], dtype=torch.int32)
    valid_sampled_tokens_count = torch.tensor([2, 1], dtype=torch.int32)
    query_start_loc_gpu = torch.tensor([0, 4, 9], dtype=torch.int32)
    token_indices_to_sample = torch.empty(2, dtype=torch.int32)
    num_rejected_tokens_gpu = torch.empty(2, dtype=torch.int32)

    eagle_prepare_inputs_padded(
        cu_num_draft_tokens,
        valid_sampled_tokens_count,
        query_start_loc_gpu,
        token_indices_to_sample,
        num_rejected_tokens_gpu,
    )

    assert token_indices_to_sample.tolist() == [2, 5]
    assert num_rejected_tokens_gpu.tolist() == [1, 3]


def test_eagle_step_update_slot_mapping_and_metadata_torch_fallback(
    monkeypatch,
) -> None:
    monkeypatch.setattr("cfie.v1.spec_decode.utils.HAS_TRITON", False)

    positions_1d = torch.tensor([2, 7], dtype=torch.int32)
    block_table_tensor = torch.tensor([[10, 11], [20, 21]], dtype=torch.int32)
    seq_lens = torch.tensor([3, 7], dtype=torch.int32)
    out_clamped_positions = torch.empty(2, dtype=torch.int32)
    out_slot_mapping = torch.empty(4, dtype=torch.int32)

    eagle_step_update_slot_mapping_and_metadata(
        positions_1d=positions_1d,
        block_table_tensor=block_table_tensor,
        seq_lens=seq_lens,
        block_size=4,
        max_model_len=8,
        out_clamped_positions=out_clamped_positions,
        out_slot_mapping=out_slot_mapping,
        input_batch_size=4,
    )

    assert out_clamped_positions.tolist() == [3, 0]
    assert out_slot_mapping.tolist() == [43, PADDING_SLOT_ID, PADDING_SLOT_ID, PADDING_SLOT_ID]
    assert seq_lens.tolist() == [4, 1]


def test_copy_and_expand_eagle_inputs_torch_fallback_shifted(monkeypatch) -> None:
    monkeypatch.setattr("cfie.v1.spec_decode.utils.HAS_TRITON", False)

    target_token_ids = torch.tensor([10, 11, 20, 21, 22], dtype=torch.int32)
    target_positions = torch.tensor([100, 101, 200, 201, 202], dtype=torch.int32)
    next_token_ids = torch.tensor([90, 91], dtype=torch.int32)
    out_input_ids = torch.empty(7, dtype=torch.int32)
    out_positions = torch.empty(7, dtype=torch.int32)
    out_is_rejected_token_mask = torch.empty(7, dtype=torch.bool)
    out_is_masked_token_mask = torch.empty(7, dtype=torch.bool)
    out_new_token_indices = torch.empty(4, dtype=torch.int32)
    out_hidden_state_mapping = torch.empty(5, dtype=torch.int32)
    query_start_loc = torch.tensor([0, 2, 5], dtype=torch.int32)
    query_end_loc = torch.tensor([1, 4], dtype=torch.int32)

    copy_and_expand_eagle_inputs(
        target_token_ids_ptr=target_token_ids,
        target_positions_ptr=target_positions,
        next_token_ids_ptr=next_token_ids,
        out_input_ids_ptr=out_input_ids,
        out_positions_ptr=out_positions,
        out_is_rejected_token_mask_ptr=out_is_rejected_token_mask,
        out_is_masked_token_mask_ptr=out_is_masked_token_mask,
        out_new_token_indices_ptr=out_new_token_indices,
        out_hidden_state_mapping_ptr=out_hidden_state_mapping,
        query_start_loc_ptr=query_start_loc,
        query_end_loc_ptr=query_end_loc,
        padding_token_id=0,
        parallel_drafting_token_id=77,
        total_input_tokens=5,
        num_padding_slots_per_request=2,
        shift_input_ids=True,
    )

    assert out_input_ids.tolist() == [11, 90, 77, 21, 22, 91, 77]
    assert out_positions.tolist() == [100, 101, 102, 200, 201, 202, 203]
    assert out_is_rejected_token_mask.tolist() == [False] * 7
    assert out_is_masked_token_mask.tolist() == [False, False, True, False, False, False, True]
    assert out_new_token_indices.tolist() == [1, 2, 5, 6]
    assert out_hidden_state_mapping.tolist() == [0, 1, 3, 4, 5]


def test_fused_sigmoid_gating_delta_rule_update_torch_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        "cfie.model_executor.layers.fla.ops.fused_sigmoid_gating.HAS_TRITON",
        False,
    )

    A_log = torch.tensor([0.0], dtype=torch.float32)
    a = torch.tensor([[0.0]], dtype=torch.float32)
    b = torch.tensor([[0.0]], dtype=torch.float32)
    dt_bias = torch.tensor([0.0], dtype=torch.float32)
    q = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float32)
    k = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float32)
    v = torch.tensor([[[[2.0]]]], dtype=torch.float32)
    initial_state = torch.zeros(1, 1, 1, 2, dtype=torch.float32)
    cu_seqlens = torch.tensor([0, 1], dtype=torch.long)
    ssm_state_indices = torch.tensor([0], dtype=torch.int32)

    out, final_state = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        beta=1.0,
        threshold=20.0,
        scale=1.0,
        initial_state=initial_state,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=False,
    )

    assert out.shape == (1, 1, 1, 1)
    assert torch.allclose(out, torch.tensor([[[[1.0]]]], dtype=torch.float32))
    assert torch.allclose(final_state, torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float32))


def test_fused_sigmoid_gating_delta_rule_update_uses_precompiled_path(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "cfie.model_executor.layers.fla.ops.fused_sigmoid_gating.HAS_TRITON",
        False,
    )

    expected_out = torch.full((1, 1, 1, 1), 5.0, dtype=torch.float32)
    expected_state = torch.full((1, 1, 1, 2), -1.0, dtype=torch.float32)
    monkeypatch.setattr(
        "cfie.model_executor.layers.fla.ops.fused_sigmoid_gating._try_precompiled_fused_sigmoid_gating_delta_rule_update",
        lambda **_: (expected_out, expected_state),
    )

    actual_out, actual_state = fused_sigmoid_gating_delta_rule_update(
        A_log=torch.tensor([0.0], dtype=torch.float32),
        a=torch.tensor([[0.0]], dtype=torch.float32),
        b=torch.tensor([[0.0]], dtype=torch.float32),
        dt_bias=torch.tensor([0.0], dtype=torch.float32),
        q=torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float32),
        k=torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float32),
        v=torch.tensor([[[[2.0]]]], dtype=torch.float32),
        beta=1.0,
        threshold=20.0,
        scale=1.0,
        initial_state=torch.zeros(1, 1, 1, 2, dtype=torch.float32),
        inplace_final_state=True,
        cu_seqlens=torch.tensor([0, 1], dtype=torch.long),
        ssm_state_indices=torch.tensor([0], dtype=torch.int32),
        use_qk_l2norm_in_kernel=False,
    )

    assert torch.equal(actual_out, expected_out)
    assert torch.equal(actual_state, expected_state)
