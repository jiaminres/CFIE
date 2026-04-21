# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from cfie.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
    triton_reshape_and_cache_flash_diffkv,
)


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")


def _make_hnd_backed_view(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3)


def _manual_fill_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    expected_key = torch.zeros_like(key_cache)
    expected_value = torch.zeros_like(value_cache)
    block_size = key_cache.shape[1]

    for token_idx, slot in enumerate(slot_mapping.cpu().tolist()):
        if slot < 0:
            continue
        block_idx = slot // block_size
        block_offset = slot % block_size
        expected_key[block_idx, block_offset].copy_(key[token_idx])
        expected_value[block_idx, block_offset].copy_(value[token_idx])

    return expected_key, expected_value


def _manual_fill_head_major_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    expected_key = torch.zeros_like(key_cache)
    expected_value = torch.zeros_like(value_cache)
    block_size = key_cache.shape[3]
    pack_x = key_cache.shape[4]

    for token_idx, slot in enumerate(slot_mapping.cpu().tolist()):
        if slot < 0:
            continue
        block_idx = slot // block_size
        block_offset = slot % block_size
        for head_idx in range(key.shape[1]):
            for dim_idx in range(key.shape[2]):
                expected_key[
                    block_idx,
                    head_idx,
                    dim_idx // pack_x,
                    block_offset,
                    dim_idx % pack_x,
                ] = key[token_idx, head_idx, dim_idx]
                expected_value[
                    block_idx,
                    head_idx,
                    dim_idx,
                    block_offset,
                ] = value[token_idx, head_idx, dim_idx]

    return expected_key, expected_value


@pytest.mark.parametrize("use_hnd_layout", [False, True])
def test_triton_reshape_and_cache_flash_prefers_precompiled_path(
    use_hnd_layout: bool,
) -> None:
    _require_cuda()

    key = torch.arange(3 * 2 * 4, device="cuda", dtype=torch.float16).view(3, 2, 4)
    value = (100 + torch.arange(3 * 2 * 4, device="cuda", dtype=torch.float16)).view(
        3, 2, 4
    )
    key_cache = torch.zeros((2, 4, 2, 4), device="cuda", dtype=torch.float16)
    value_cache = torch.zeros((2, 4, 2, 4), device="cuda", dtype=torch.float16)
    if use_hnd_layout:
        key_cache = _make_hnd_backed_view(key_cache)
        value_cache = _make_hnd_backed_view(value_cache)

    slot_mapping = torch.tensor([0, 5, -1], device="cuda", dtype=torch.long)
    scales = torch.ones(1, device="cuda", dtype=torch.float32)

    triton_reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        "auto",
        scales,
        scales,
    )

    expected_key, expected_value = _manual_fill_cache(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
    )

    torch.testing.assert_close(key_cache.cpu(), expected_key.cpu(), rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        value_cache.cpu(),
        expected_value.cpu(),
        rtol=0.0,
        atol=0.0,
    )


def test_triton_reshape_and_cache_flash_prefers_precompiled_head_major_path() -> None:
    _require_cuda()

    key = torch.arange(3 * 2 * 8, device="cuda", dtype=torch.float16).view(3, 2, 8)
    value = (200 + torch.arange(3 * 2 * 8, device="cuda", dtype=torch.float16)).view(
        3, 2, 8
    )
    key_cache = torch.zeros((2, 2, 1, 4, 8), device="cuda", dtype=torch.float16)
    value_cache = torch.zeros((2, 2, 8, 4), device="cuda", dtype=torch.float16)
    slot_mapping = torch.tensor([0, 5, -1], device="cuda", dtype=torch.long)
    scales = torch.ones(1, device="cuda", dtype=torch.float32)

    triton_reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        "auto",
        scales,
        scales,
    )

    expected_key, expected_value = _manual_fill_head_major_cache(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
    )

    torch.testing.assert_close(key_cache.cpu(), expected_key.cpu(), rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        value_cache.cpu(),
        expected_value.cpu(),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("use_hnd_layout", [False, True])
def test_triton_reshape_and_cache_flash_diffkv_prefers_precompiled_path(
    use_hnd_layout: bool,
) -> None:
    _require_cuda()

    key = torch.arange(3 * 2 * 4, device="cuda", dtype=torch.float16).view(3, 2, 4)
    value = (50 + torch.arange(3 * 2 * 3, device="cuda", dtype=torch.float16)).view(
        3, 2, 3
    )
    kv_cache = torch.zeros((2, 4, 2, 7), device="cuda", dtype=torch.float16)
    if use_hnd_layout:
        kv_cache = _make_hnd_backed_view(kv_cache)

    slot_mapping = torch.tensor([1, -1, 6], device="cuda", dtype=torch.long)
    scales = torch.ones(1, device="cuda", dtype=torch.float32)

    triton_reshape_and_cache_flash_diffkv(
        key,
        value,
        kv_cache,
        slot_mapping,
        "auto",
        scales,
        scales,
    )

    key_cache = kv_cache[..., : key.shape[2]]
    value_cache = kv_cache[..., key.shape[2] :]
    expected_key, expected_value = _manual_fill_cache(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
    )

    torch.testing.assert_close(key_cache.cpu(), expected_key.cpu(), rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        value_cache.cpu(),
        expected_value.cpu(),
        rtol=0.0,
        atol=0.0,
    )
