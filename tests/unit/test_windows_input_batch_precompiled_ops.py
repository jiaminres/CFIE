# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from cfie import _custom_ops
from cfie.v1.worker.gpu import input_batch


def _require_cuda_precompiled() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    required_ops = [
        _custom_ops.has_precompiled_input_batch_prepare_prefill_inputs,
        _custom_ops.has_precompiled_input_batch_prepare_pos_seq_lens,
        _custom_ops.has_precompiled_input_batch_combine_sampled_and_draft_tokens,
        _custom_ops.has_precompiled_input_batch_get_num_sampled_and_rejected,
        _custom_ops.has_precompiled_input_batch_post_update,
        _custom_ops.has_precompiled_input_batch_post_update_pool,
        _custom_ops.has_precompiled_input_batch_expand_idx_mapping,
    ]
    if not all(has_op() for has_op in required_ops):
        pytest.skip("Input batch precompiled ops are not available in this build")


def test_prepare_prefill_and_positions_precompiled_match_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_cuda_precompiled()
    monkeypatch.setattr(input_batch, "HAS_TRITON", False)

    def run(
        device: str,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor],
    ]:
        input_ids = torch.full((5,), -1, device=device, dtype=torch.int32)
        next_prefill_tokens = torch.full((3,), -1, device=device, dtype=torch.int32)
        idx_mapping = torch.tensor([2, 0], device=device, dtype=torch.int32)
        query_start_loc = torch.tensor([0, 3, 5], device=device, dtype=torch.int32)
        all_token_ids = torch.tensor(
            [
                [10, 11, 12, 13, 14, 15],
                [20, 21, 22, 23, 24, 25],
                [30, 31, 32, 33, 34, 35],
            ],
            device=device,
            dtype=torch.int32,
        )
        prefill_len = torch.tensor([4, 5, 6], device=device, dtype=torch.int32)
        num_computed_tokens = torch.tensor(
            [1, 5, 2], device=device, dtype=torch.int32
        )

        input_batch.prepare_prefill_inputs(
            input_ids,
            next_prefill_tokens,
            idx_mapping,
            query_start_loc,
            all_token_ids,
            prefill_len,
            num_computed_tokens,
        )

        positions = torch.full((5,), -1, device=device, dtype=torch.int64)
        seq_lens = torch.full((4,), 99, device=device, dtype=torch.int32)
        input_batch.prepare_pos_seq_lens(
            idx_mapping,
            query_start_loc,
            num_computed_tokens,
            positions,
            seq_lens,
        )
        return input_ids, next_prefill_tokens, positions, seq_lens

    actual = run("cuda")
    expected = run("cpu")

    for actual_tensor, expected_tensor in zip(actual, expected):
        assert torch.equal(actual_tensor.cpu(), expected_tensor)


def test_spec_decode_metadata_precompiled_match_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_cuda_precompiled()
    monkeypatch.setattr(input_batch, "HAS_TRITON", False)

    def run(device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = torch.full((7,), -1, device=device, dtype=torch.int32)
        idx_mapping = torch.tensor([2, 0], device=device, dtype=torch.int32)
        last_sampled_tokens = torch.tensor(
            [100, 110, 120], device=device, dtype=torch.int32
        )
        query_start_loc = torch.tensor([0, 4, 7], device=device, dtype=torch.int32)
        seq_lens = torch.tensor([7, 3], device=device, dtype=torch.int32)
        prefill_len = torch.tensor([4, 5, 6], device=device, dtype=torch.int32)
        draft_tokens = torch.tensor(
            [[11, 12], [21, 22], [31, 32]], device=device, dtype=torch.int32
        )
        cu_num_logits = torch.tensor([0, 3, 4], device=device, dtype=torch.int32)

        logits_indices = input_batch.combine_sampled_and_draft_tokens(
            input_ids,
            idx_mapping,
            last_sampled_tokens,
            query_start_loc,
            seq_lens,
            prefill_len,
            draft_tokens,
            cu_num_logits,
            num_logits=4,
        )

        num_sampled = torch.tensor([2, 1], device=device, dtype=torch.int32)
        _, num_rejected = input_batch.get_num_sampled_and_rejected(
            num_sampled,
            seq_lens,
            cu_num_logits,
            idx_mapping,
            prefill_len,
        )

        expanded_idx_mapping, expanded_local_pos = input_batch.expand_idx_mapping(
            idx_mapping,
            total_num_logits=4,
            cu_num_logits=cu_num_logits,
            max_expand_len=3,
        )
        return logits_indices, input_ids, num_sampled, num_rejected, (
            expanded_idx_mapping,
            expanded_local_pos,
        )

    actual_logits, actual_input_ids, actual_sampled, actual_rejected, actual_expand = (
        run("cuda")
    )
    (
        expected_logits,
        expected_input_ids,
        expected_sampled,
        expected_rejected,
        expected_expand,
    ) = run("cpu")

    assert torch.equal(actual_logits.cpu(), expected_logits)
    assert torch.equal(actual_input_ids.cpu(), expected_input_ids)
    assert torch.equal(actual_sampled.cpu(), expected_sampled)
    assert torch.equal(actual_rejected.cpu(), expected_rejected)
    assert torch.equal(actual_expand[0].cpu(), expected_expand[0])
    assert torch.equal(actual_expand[1].cpu(), expected_expand[1])


def test_post_update_precompiled_matches_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_cuda_precompiled()
    monkeypatch.setattr(input_batch, "HAS_TRITON", False)

    def run(
        device: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        idx_mapping = torch.tensor([2, 0], device=device, dtype=torch.int32)
        num_computed_tokens = torch.tensor(
            [1, 10, 2], device=device, dtype=torch.int32
        )
        last_sampled_tokens = torch.tensor(
            [100, 110, 120], device=device, dtype=torch.int32
        )
        output_bin_counts = torch.zeros((3, 256), device=device, dtype=torch.int32)
        sampled_tokens = torch.tensor(
            [[101, 102, 103], [201, 202, 203]],
            device=device,
            dtype=torch.int32,
        )
        num_sampled = torch.tensor([2, 0], device=device, dtype=torch.int32)
        num_rejected = torch.tensor([1, 0], device=device, dtype=torch.int32)
        query_start_loc = torch.tensor([0, 4, 7], device=device, dtype=torch.int32)
        all_token_ids = torch.zeros((3, 12), device=device, dtype=torch.int32)
        total_len = torch.tensor([3, 4, 3], device=device, dtype=torch.int32)

        input_batch.post_update(
            idx_mapping,
            num_computed_tokens,
            last_sampled_tokens,
            output_bin_counts,
            sampled_tokens,
            num_sampled,
            num_rejected,
            query_start_loc,
            all_token_ids,
            total_len,
        )
        return num_computed_tokens, last_sampled_tokens, output_bin_counts, (
            all_token_ids,
            total_len,
        )

    actual = run("cuda")
    expected = run("cpu")

    assert torch.equal(actual[0].cpu(), expected[0])
    assert torch.equal(actual[1].cpu(), expected[1])
    assert torch.equal(actual[2].cpu(), expected[2])
    assert torch.equal(actual[3][0].cpu(), expected[3][0])
    assert torch.equal(actual[3][1].cpu(), expected[3][1])


def test_post_update_pool_precompiled_matches_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_cuda_precompiled()
    monkeypatch.setattr(input_batch, "HAS_TRITON", False)

    def run(device: str) -> torch.Tensor:
        idx_mapping = torch.tensor([2, 0], device=device, dtype=torch.int32)
        num_computed_tokens = torch.tensor(
            [1, 10, 2], device=device, dtype=torch.int32
        )
        query_start_loc = torch.tensor([0, 4, 7], device=device, dtype=torch.int32)

        input_batch.post_update_pool(
            idx_mapping,
            num_computed_tokens,
            query_start_loc,
        )
        return num_computed_tokens

    actual = run("cuda")
    expected = run("cpu")

    assert torch.equal(actual.cpu(), expected)
