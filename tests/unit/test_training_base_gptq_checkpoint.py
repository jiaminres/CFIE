"""Unit tests for GPTQ checkpoint tensor decoding."""

from __future__ import annotations

import pytest
import torch

from cfie_training.training_base import (
    GptqInt4CheckpointDecoder,
    GptqInt4CheckpointTensors,
    pack_gptq_int4_qweight,
    pack_gptq_int4_qzeros,
    unpack_gptq_int4_qweight,
    unpack_gptq_int4_qzeros,
)


def test_gptq_qweight_pack_and_unpack_round_trips_int4_values() -> None:
    values = torch.arange(32, dtype=torch.int32).reshape(16, 2) % 16

    packed = pack_gptq_int4_qweight(values)
    unpacked = unpack_gptq_int4_qweight(packed, size_k=16, size_n=2)

    assert packed.shape == (2, 2)
    assert torch.equal(unpacked, values.to(torch.uint8))


def test_gptq_qzeros_pack_and_unpack_round_trips_int4_values() -> None:
    values = torch.arange(16, dtype=torch.int32).reshape(2, 8)

    packed = pack_gptq_int4_qzeros(values)
    unpacked = unpack_gptq_int4_qzeros(packed, num_groups=2, size_n=8)

    assert packed.shape == (2, 1)
    assert torch.equal(unpacked, values.to(torch.uint8))


def test_gptq_decoder_uses_scales_qzeros_and_g_idx() -> None:
    quantized = torch.tensor(
        [
            [8, 9],
            [10, 7],
            [6, 11],
            [12, 5],
            [9, 8],
            [7, 10],
            [11, 6],
            [5, 12],
        ],
        dtype=torch.int32,
    )
    qzeros = torch.full((2, 2), 7, dtype=torch.int32)
    decoder = GptqInt4CheckpointDecoder(group_size=4, decoded_layout="k_n")

    decoded = decoder.decode(
        GptqInt4CheckpointTensors(
            qweight=pack_gptq_int4_qweight(quantized),
            scales=torch.tensor([[1.0, 2.0], [0.5, 4.0]]),
            qzeros=pack_gptq_int4_qzeros(qzeros),
            g_idx=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
        )
    )

    expected = torch.tensor(
        [
            [0.0, 2.0],
            [2.0, -2.0],
            [-2.0, 6.0],
            [4.0, -6.0],
            [0.5, 0.0],
            [-0.5, 8.0],
            [1.5, -8.0],
            [-1.5, 16.0],
        ]
    )
    assert torch.equal(decoded, expected)


def test_gptq_decoder_can_return_output_input_layout() -> None:
    quantized = torch.tensor([[8, 9], [10, 7]], dtype=torch.int32)
    decoder = GptqInt4CheckpointDecoder(group_size=8, decoded_layout="n_k")

    decoded = decoder.decode(
        GptqInt4CheckpointTensors(
            qweight=pack_gptq_int4_qweight(quantized),
            scales=torch.tensor([[1.0, 2.0]]),
            g_idx=torch.tensor([0, 0]),
        )
    )

    assert torch.equal(decoded, torch.tensor([[0.0, 2.0], [2.0, -2.0]]))


def test_gptq_decoder_rejects_scale_group_mismatch() -> None:
    quantized = torch.full((8, 2), 8, dtype=torch.int32)
    decoder = GptqInt4CheckpointDecoder(group_size=4)

    with pytest.raises(ValueError, match="outside scales"):
        decoder.decode(
            GptqInt4CheckpointTensors(
                qweight=pack_gptq_int4_qweight(quantized),
                scales=torch.ones(1, 2),
                g_idx=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
            )
        )
