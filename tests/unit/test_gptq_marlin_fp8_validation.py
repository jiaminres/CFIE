from __future__ import annotations

from pathlib import Path

import pytest
import torch

from cfie import _custom_ops as ops
from cfie.model_executor.layers.quantization.utils.quant_utils import gptq_pack
from cfie.op_validation.gptq_marlin_fp8 import (
    GPTQMarlinFP8Linear,
    load_opcheck_library,
)

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)


def _dequantize_reference(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    *,
    size_k: int,
    size_n: int,
    group_size: int,
) -> torch.Tensor:
    unpacked = torch.empty((size_k, size_n), device=qweight.device, dtype=torch.int32)
    for offset in range(8):
        unpacked[offset::8] = ((qweight >> (offset * 4)) & 0xF).to(torch.int32)
    unpacked = unpacked - 8
    dense = torch.empty((size_k, size_n), device=qweight.device, dtype=torch.float32)
    num_groups = 1 if group_size == -1 else size_k // group_size
    rows_per_group = size_k if group_size == -1 else group_size
    for group_index in range(num_groups):
        start = group_index * rows_per_group
        end = min(start + rows_per_group, size_k)
        dense[start:end] = (
            unpacked[start:end].to(torch.float32)
            * scales[group_index].to(torch.float32)
        )
    return dense


def test_gptq_marlin_fp8_validation_forward_and_backward() -> None:
    try:
        library_path = load_opcheck_library()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))

    assert library_path

    device = torch.device("cuda")
    size_k = 128
    size_n = 128
    group_size = 32
    num_groups = size_k // group_size
    batch = 8

    logical_qweight = torch.randint(
        -8, 8, (size_k, size_n), device=device, dtype=torch.int32
    )
    packed_qweight = gptq_pack(
        (logical_qweight + 8).to(torch.int32),
        4,
        size_k,
        size_n,
    ).contiguous()
    scales = (
        torch.rand((num_groups, size_n), device=device, dtype=torch.float16) * 0.05
        + 0.01
    ).contiguous()

    dense_ref = _dequantize_reference(
        packed_qweight,
        scales,
        size_k=size_k,
        size_n=size_n,
        group_size=group_size,
    )
    dense_from_op = torch.ops.opcheck_C.gptq_marlin_fp8_dequantize(
        packed_qweight,
        scales,
        size_k,
        size_n,
        group_size,
        False,
    )
    dense_t_from_op = torch.ops.opcheck_C.gptq_marlin_fp8_dequantize(
        packed_qweight,
        scales,
        size_k,
        size_n,
        group_size,
        True,
    )
    dense_ref_cast = dense_ref.to(dense_from_op.dtype)
    torch.testing.assert_close(dense_from_op, dense_ref_cast, atol=0, rtol=0)
    torch.testing.assert_close(
        dense_t_from_op,
        dense_ref_cast.transpose(0, 1),
        atol=0,
        rtol=0,
    )

    module = GPTQMarlinFP8Linear.from_raw_gptq(
        packed_qweight,
        scales,
        size_k=size_k,
        size_n=size_n,
        group_size=group_size,
    )
    x = (
        torch.randn((batch, size_k), device=device, dtype=torch.float16) * 0.5
    ).requires_grad_(True)
    output = module(x)

    output_ref = x.detach().to(torch.float32) @ dense_ref
    torch.testing.assert_close(output.float(), output_ref, atol=2e-1, rtol=2e-1)

    grad_output = torch.randn_like(output) * 0.5
    output.backward(grad_output)
    grad_input_ref = grad_output.to(torch.float32) @ dense_ref.transpose(0, 1)
    torch.testing.assert_close(
        x.grad.float(),
        grad_input_ref,
        atol=2e-1,
        rtol=2e-1,
    )
