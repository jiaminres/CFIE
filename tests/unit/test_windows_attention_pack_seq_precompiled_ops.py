# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from cfie import _custom_ops
from cfie.v1.attention.ops import common


def _require_cuda_pack_ops() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if not _custom_ops.has_precompiled_pack_seq():
        pytest.skip("Precompiled pack op is not available in this build")
    if not _custom_ops.has_precompiled_unpack_seq():
        pytest.skip("Precompiled unpack op is not available in this build")


def _reference_pack_seq(
    x: torch.Tensor,
    lengths: torch.Tensor,
    pad_value: float,
) -> torch.Tensor:
    original_shape = x.shape
    if len(original_shape) > 2:
        x = x.reshape(original_shape[0], -1)

    lengths_cpu = lengths.to(dtype=torch.long, device="cpu").tolist()
    batch = len(lengths_cpu)
    max_len = max(lengths_cpu, default=0)
    out = torch.full((batch, max_len, x.shape[1]), pad_value, device=x.device, dtype=x.dtype)

    start = 0
    for batch_idx, seq_len in enumerate(lengths_cpu):
        if seq_len > 0:
            out[batch_idx, :seq_len].copy_(x[start : start + seq_len])
        start += seq_len

    if len(original_shape) > 2:
        out = out.reshape((batch, max_len) + original_shape[1:])
    return out


def _reference_unpack_seq(
    packed_tensor: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    original_shape = packed_tensor.shape
    if len(original_shape) > 3:
        packed_tensor = packed_tensor.reshape(original_shape[0], original_shape[1], -1)

    lengths_cpu = lengths.to(dtype=torch.long, device="cpu").tolist()
    total_tokens = sum(lengths_cpu)
    out = torch.empty(
        (total_tokens, packed_tensor.shape[2]),
        device=packed_tensor.device,
        dtype=packed_tensor.dtype,
    )

    start = 0
    for batch_idx, seq_len in enumerate(lengths_cpu):
        if seq_len > 0:
            out[start : start + seq_len].copy_(packed_tensor[batch_idx, :seq_len])
        start += seq_len

    if len(original_shape) > 3:
        out = out.reshape((total_tokens,) + original_shape[2:])
    return out


@pytest.mark.parametrize("pad_value", [-float("inf"), -7.5])
@pytest.mark.parametrize("use_multidim", [False, True])
def test_pack_unpack_precompiled_matches_reference(
    monkeypatch: pytest.MonkeyPatch,
    pad_value: float,
    use_multidim: bool,
) -> None:
    _require_cuda_pack_ops()
    monkeypatch.setattr(common, "HAS_TRITON", False)

    lengths = torch.tensor([2, 1, 3], device="cuda", dtype=torch.int32)
    if use_multidim:
        x = torch.arange(6 * 2 * 3, device="cuda", dtype=torch.float16).reshape(6, 2, 3)
    else:
        x = torch.arange(6 * 4, device="cuda", dtype=torch.float16).reshape(6, 4)

    packed = common.pack_seq_triton(x, lengths, pad_value=pad_value)
    expected_packed = _reference_pack_seq(x, lengths, pad_value=pad_value)
    torch.testing.assert_close(packed.cpu(), expected_packed.cpu(), rtol=0.0, atol=0.0)

    unpacked = common.unpack_seq_triton(packed, lengths)
    expected_unpacked = _reference_unpack_seq(expected_packed, lengths)
    torch.testing.assert_close(unpacked.cpu(), expected_unpacked.cpu(), rtol=0.0, atol=0.0)
    torch.testing.assert_close(unpacked.cpu(), x.cpu(), rtol=0.0, atol=0.0)


def test_pack_unpack_torch_fallback_matches_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(common, "HAS_TRITON", False)
    monkeypatch.setattr(_custom_ops, "has_precompiled_pack_seq", lambda: False)
    monkeypatch.setattr(_custom_ops, "has_precompiled_unpack_seq", lambda: False)

    lengths = torch.tensor([1, 3], dtype=torch.int32)
    x = torch.arange(4 * 2, dtype=torch.float32).reshape(4, 2)

    packed = common.pack_seq_triton(x, lengths, pad_value=-9.0)
    expected_packed = _reference_pack_seq(x, lengths, pad_value=-9.0)
    torch.testing.assert_close(packed, expected_packed, rtol=0.0, atol=0.0)

    unpacked = common.unpack_seq_triton(packed, lengths)
    torch.testing.assert_close(unpacked, x, rtol=0.0, atol=0.0)
