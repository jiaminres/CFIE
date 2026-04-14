# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from cfie import envs
from cfie.logger import init_logger
from cfie.model_executor.layers.batch_invariant import cfie_is_batch_invariant
from cfie.platforms import current_platform
from cfie.utils.math_utils import next_power_of_2

logger = init_logger(__name__)
is_batch_invariant = cfie_is_batch_invariant()

_LORA_A_PTR_DICT: dict[tuple[int, ...], tuple[torch.tensor, ...]] = {}
_LORA_B_PTR_DICT: dict[tuple[int, ...], tuple[torch.tensor, ...]] = {}


def normalize_lora_weight_dims(lora_weight: torch.Tensor) -> torch.Tensor:
    if lora_weight.ndim == 4:
        assert lora_weight.size(1) == 1
        return lora_weight.squeeze(dim=1)
    assert lora_weight.ndim == 3
    return lora_weight


def normalize_lora_scale_dims(lora_scale: torch.Tensor) -> torch.Tensor:
    if lora_scale.ndim == 4:
        assert lora_scale.size(1) == 1
        return lora_scale.squeeze(dim=1)
    return lora_scale


def iter_lora_token_indices(
    token_indices_sorted_by_lora_ids: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    num_active_loras: torch.Tensor | int | None = None,
):
    if isinstance(num_active_loras, int):
        active_loras = num_active_loras
    elif num_active_loras is not None:
        active_loras = int(num_active_loras.item())
    else:
        active_loras = lora_ids.size(0)
    active_loras = min(active_loras, lora_ids.size(0))

    for lora_idx in range(active_loras):
        lora_id = int(lora_ids[lora_idx].item())
        if lora_id < 0:
            continue

        start = int(lora_token_start_loc[lora_idx].item())
        token_count = int(num_tokens_per_lora[lora_idx].item())
        if token_count <= 0:
            continue

        yield lora_id, token_indices_sorted_by_lora_ids.narrow(0, start, token_count)


def iter_fused_moe_lora_assignments(
    token_lora_mapping: torch.Tensor,
    expert_ids: torch.Tensor,
    top_k_num: int,
    *,
    sorted_token_ids: torch.Tensor | None = None,
    num_tokens_post_padded: torch.Tensor | None = None,
    lora_ids: torch.Tensor | None = None,
    num_active_loras: torch.Tensor | int | None = None,
):
    if sorted_token_ids is None:
        flat_expert_ids = expert_ids.reshape(-1)
        total_assignments = flat_expert_ids.numel()
        for flat_idx in range(total_assignments):
            token_idx = flat_idx // top_k_num
            if token_idx >= token_lora_mapping.size(0):
                break
            expert_id = int(flat_expert_ids[flat_idx].item())
            if expert_id < 0:
                continue
            lora_id = int(token_lora_mapping[token_idx].item())
            if lora_id < 0:
                continue
            yield lora_id, token_idx, flat_idx % top_k_num, expert_id
        return

    assert lora_ids is not None
    active_loras = (
        num_active_loras
        if isinstance(num_active_loras, int)
        else int(num_active_loras.item()) if num_active_loras is not None else lora_ids.size(0)
    )
    active_loras = min(active_loras, lora_ids.size(0), sorted_token_ids.size(0))
    max_token_slots = token_lora_mapping.size(0) * top_k_num

    for lora_slot in range(active_loras):
        lora_id = int(lora_ids[lora_slot].item())
        if lora_id < 0:
            continue
        padded_count = (
            int(num_tokens_post_padded[lora_slot].item())
            if num_tokens_post_padded is not None
            else sorted_token_ids.size(1)
        )
        padded_count = min(padded_count, sorted_token_ids.size(1), expert_ids.size(1))
        for col_idx in range(padded_count):
            flat_idx = int(sorted_token_ids[lora_slot, col_idx].item())
            expert_id = int(expert_ids[lora_slot, col_idx].item())
            if flat_idx < 0 or flat_idx >= max_token_slots or expert_id < 0:
                continue
            token_idx = flat_idx // top_k_num
            yield lora_id, token_idx, flat_idx % top_k_num, expert_id


def materialize_lora_token_scales(
    a_scale: torch.Tensor | None,
    token_indices: torch.Tensor,
    width: int,
    *,
    group_k: int = 0,
) -> torch.Tensor:
    if a_scale is None:
        return torch.ones((token_indices.numel(), width), dtype=torch.float32)

    scale = normalize_lora_scale_dims(a_scale).to(torch.float32)
    if scale.ndim == 0 or scale.numel() == 1:
        return scale.reshape(1, 1).expand(token_indices.numel(), width)

    select_indices = token_indices.to(device=scale.device, dtype=torch.long)
    if scale.ndim == 1:
        selected = (
            scale.index_select(0, select_indices)
            if scale.size(0) > int(select_indices.max().item())
            else scale
        )
        if selected.numel() == 1:
            return selected.reshape(1, 1).expand(token_indices.numel(), width)
        return selected.reshape(-1, 1).expand(-1, width)

    selected = (
        scale.index_select(0, select_indices)
        if scale.size(0) > int(select_indices.max().item())
        else scale
    )
    if selected.ndim == 1:
        return selected.reshape(-1, 1).expand(-1, width)
    if group_k > 0:
        k_groups = torch.div(
            torch.arange(width, device=selected.device),
            group_k,
            rounding_mode="floor",
        ).clamp_max(selected.size(1) - 1)
        return selected.index_select(1, k_groups)
    if selected.size(1) == 1:
        return selected.expand(-1, width)
    if selected.size(1) >= width:
        return selected[:, :width]
    return selected[:, -1:].expand(-1, width)


def materialize_lora_weight_scales(
    weight_scale: torch.Tensor | None,
    lora_id: int,
    output_width: int,
    input_width: int,
    *,
    group_n: int = 0,
    group_k: int = 0,
    per_channel_quant: bool = False,
) -> torch.Tensor:
    if weight_scale is None:
        return torch.ones((output_width, input_width), dtype=torch.float32)

    scale = normalize_lora_scale_dims(weight_scale).to(torch.float32)
    if scale.ndim > 0 and scale.size(0) > lora_id:
        scale = scale[lora_id]

    if scale.ndim == 0 or scale.numel() == 1:
        return scale.reshape(1, 1).expand(output_width, input_width)

    if group_k > 0 and group_n > 0:
        if scale.ndim == 1:
            if scale.numel() == (output_width + group_n - 1) // group_n:
                n_groups = torch.div(
                    torch.arange(output_width, device=scale.device),
                    group_n,
                    rounding_mode="floor",
                ).clamp_max(scale.numel() - 1)
                return scale.index_select(0, n_groups).reshape(-1, 1).expand(
                    -1, input_width
                )
            if scale.numel() == (input_width + group_k - 1) // group_k:
                k_groups = torch.div(
                    torch.arange(input_width, device=scale.device),
                    group_k,
                    rounding_mode="floor",
                ).clamp_max(scale.numel() - 1)
                return scale.index_select(0, k_groups).reshape(1, -1).expand(
                    output_width, -1
                )
        if scale.ndim == 1:
            scale = scale.reshape(-1, 1)
        n_groups = torch.div(
            torch.arange(output_width, device=scale.device),
            group_n,
            rounding_mode="floor",
        ).clamp_max(scale.size(0) - 1)
        k_groups = torch.div(
            torch.arange(input_width, device=scale.device),
            group_k,
            rounding_mode="floor",
        ).clamp_max(scale.size(1) - 1)
        return scale.index_select(0, n_groups).index_select(1, k_groups)

    if per_channel_quant or (scale.ndim == 1 and scale.numel() >= output_width):
        scale_vec = scale.reshape(-1)
        n_groups = torch.arange(output_width, device=scale.device).clamp_max(
            scale_vec.numel() - 1
        )
        return scale_vec.index_select(0, n_groups).reshape(-1, 1).expand(-1, input_width)

    return scale.reshape(1, 1).expand(output_width, input_width)


def _get_lora_a_ptr(lora_a_weights: list[torch.Tensor], device: torch.device):
    """
    `_LORA_A_PTR_DICT` collects the required information during `profile_run`,
    After this, it remains constant and subsequent usage is through LUT.
    Refer to:
    https://github.com/triton-lang/triton/blob/release/3.1.x/python/tutorials/08-grouped-gemm.py
    """
    key = tuple(lora_weight.data_ptr() for lora_weight in lora_a_weights)

    if values := _LORA_A_PTR_DICT.get(key):
        return values

    lora_strides_d0 = []
    lora_strides_d1 = []
    lora_strides_d2 = []
    tensor_ptrs = []
    for lora_a_weight in lora_a_weights:
        lora_a_weight = normalize_lora_weight_dims(lora_a_weight)
        assert lora_a_weight.is_contiguous()
        tensor_ptrs.append(lora_a_weight.data_ptr())
        lora_strides_d0.append(lora_a_weight.stride(0))
        lora_strides_d1.append(lora_a_weight.stride(1))
        lora_strides_d2.append(lora_a_weight.stride(2))
    if len(lora_a_weights) > 1:
        lora_ptr_tensor = torch.tensor(tensor_ptrs, device=device, dtype=torch.uint64)
    else:
        lora_ptr_tensor = lora_a_weights[0]

    if (
        len(set(lora_strides_d0)) > 1
        or len(set(lora_strides_d1)) > 1
        or len(set(lora_strides_d2)) > 1
    ):
        raise ValueError("All LoRA weights must have the same stride.")

    _LORA_A_PTR_DICT[key] = (
        lora_ptr_tensor,
        lora_strides_d0[0],
        lora_strides_d1[0],
        lora_strides_d2[0],
    )
    return _LORA_A_PTR_DICT.get(key)


def _get_lora_b_ptr(
    lora_weights: list[torch.Tensor], offset_start: int, device: torch.device
):
    """
     `_LORA_B_PTR_DICT` collects the required information during `profile_run`,
    After this, it remains constant and subsequent usage is through LUT.
    Refer to:
    https://github.com/triton-lang/triton/blob/release/3.1.x/python/tutorials/08-grouped-gemm.py

    """

    key = tuple(lora_weight.data_ptr() for lora_weight in lora_weights)
    if values := _LORA_B_PTR_DICT.get(key):
        return values
    slice_offset_lst = []
    tensor_ptrs = []
    lora_strides_d0 = []
    lora_strides_d1 = []
    lora_strides_d2 = []
    hidden_sizes = []
    slice_offset = offset_start
    for lora_b_weight in lora_weights:
        lora_b_weight = normalize_lora_weight_dims(lora_b_weight)
        assert lora_b_weight.is_contiguous()
        tensor_ptrs.append(lora_b_weight.data_ptr())
        lora_strides_d0.append(lora_b_weight.stride(0))
        lora_strides_d1.append(lora_b_weight.stride(1))
        lora_strides_d2.append(lora_b_weight.stride(2))
        slice_offset_lst.append(slice_offset)
        slice_offset += lora_b_weight.size(1)
        hidden_sizes.append(lora_b_weight.size(1))

    if len(lora_weights) > 1:
        # note these are device tensors
        lora_ptr_tensor = torch.tensor(tensor_ptrs, device=device, dtype=torch.uint64)
        slice_start_tensor = torch.tensor(
            slice_offset_lst, device=device, dtype=torch.uint64
        )
    else:
        slice_start_tensor = slice_offset_lst[0]
        lora_ptr_tensor = lora_b_weight[0]

    # If each lora has the same stride, there's no need to use a
    # tensor for storage.
    if (
        len(set(lora_strides_d0)) == 1
        and len(set(lora_strides_d1)) == 1
        and len(set(lora_strides_d2)) == 1
    ) and len(set(hidden_sizes)) == 1:
        lora_strides_d0_tensor = lora_strides_d0[0]
        lora_strides_d1_tensor = lora_strides_d1[0]
        lora_strides_d2_tensor = lora_strides_d2[0]
        hidden_sizes_tensor = hidden_sizes[0]
        same_stride = True

    else:
        lora_strides_d0_tensor = torch.tensor(lora_strides_d0, device=device)
        lora_strides_d1_tensor = torch.tensor(lora_strides_d1, device=device)
        lora_strides_d2_tensor = torch.tensor(lora_strides_d2, device=device)
        hidden_sizes_tensor = torch.tensor(hidden_sizes, device=device)
        same_stride = False
    # MAX_N is the maximum hidden size among all the lora_b weights
    MAX_N = max(hidden_sizes)
    _LORA_B_PTR_DICT[key] = (
        slice_start_tensor,
        lora_ptr_tensor,
        lora_strides_d0_tensor,
        lora_strides_d1_tensor,
        lora_strides_d2_tensor,
        hidden_sizes_tensor,
        same_stride,
        MAX_N,
    )
    return _LORA_B_PTR_DICT.get(key)


@functools.lru_cache
def load_lora_op_config(op_type: str, add_inputs: bool | None) -> dict | None:
    user_defined_config_folder = envs.VLLM_TUNED_CONFIG_FOLDER
    # Avoid optimizing for the batch invariant case. Use default config
    if user_defined_config_folder is not None and not is_batch_invariant:
        gpu_name = torch.cuda.get_device_name()
        gpu_name = gpu_name.replace(" ", "_")
        gpu_name = gpu_name.replace("-", "_")

        config_fname = None
        # only expand op needs to consider add_inputs
        if op_type == "expand":
            config_fname = (
                f"{gpu_name}_{op_type.upper()}_{str(add_inputs).upper()}.json"
            )
        else:
            config_fname = f"{gpu_name}_{op_type.upper()}.json"

        config_path = Path(f"{user_defined_config_folder}/{config_fname}")
        if not config_path.exists():
            logger.warning_once(f"No LoRA kernel configs found in {config_path}")
            return None

        # Load json
        logger.info_once(f"Using tuned LoRA kernel configs from {config_path}.")
        with open(str(config_path)) as f:
            config_data = json.load(f)
    else:
        config_data = None

    return config_data


@functools.lru_cache
def get_lora_op_configs(
    op_type: str,
    max_loras: int,
    batch: int,
    hidden_size: int,
    rank: int,
    num_slices: int,
    add_inputs: bool | None = None,
    moe_intermediate_size: int | None = None,
) -> dict[str, int | None]:
    # Add support for fused_moe_lora ops
    assert op_type in [
        "shrink",
        "expand",
        "fused_moe_lora_w13_shrink",
        "fused_moe_lora_w13_expand",
        "fused_moe_lora_w2_shrink",
        "fused_moe_lora_w2_expand",
    ]

    # default config
    default = {}
    if op_type == "shrink":
        split_k = 64 if batch < 128 else 8
        if is_batch_invariant:
            split_k = 1
        default = {
            "block_m": 32,
            "block_n": 16,
            "block_k": 256 if batch < 128 else 32,
            "split_k": split_k,
            "num_warps": 4,
            "num_ctas": 1,
            "group_size_m": 8,
            "num_stages": 2,
            "max_nreg": None,
        }
    # The default config for fused_moe_lora ops
    elif op_type in [
        "fused_moe_lora_w13_shrink",
        "fused_moe_lora_w2_shrink",
    ]:
        default = {
            "block_m": 64,
            "block_n": min(64, next_power_of_2(rank)),
            "block_k": 32,
            "num_warps": 4,
            "num_stages": 3,
            "group_size_m": 8,
            "split_k": 1,
        }
    elif op_type in [
        "fused_moe_lora_w13_expand",
        "fused_moe_lora_w2_expand",
    ]:
        default = {
            "block_m": 64,
            "block_n": 64,
            "block_k": max(16, min(32, next_power_of_2(rank))),
            "num_warps": 4,
            "num_stages": 3,
            "group_size_m": 8,
            "split_k": 1,
        }
    else:
        default = {
            "block_m": 64,
            "block_n": 64 if num_slices > 1 else 128,
            "block_k": 32,
            "num_warps": 4,
            "num_ctas": 1,
            "num_stages": 2,
            "max_nreg": None,
        }
    m = batch

    k, n = (hidden_size, rank) if op_type == "shrink" else (rank, hidden_size)

    config_data: Any
    config_data = load_lora_op_config(op_type, add_inputs)
    if not config_data:
        logger.warning_once("Using default LoRA kernel configs")
        return default

    # config is structured as config_data[max_loras][num_slices][m][k][n] = {}
    # slice by max_loras
    config_data = (
        config_data.get(str(max_loras))
        or config_data[min(config_data.keys(), key=lambda x: abs(int(x) - max_loras))]
    )
    # slice by num_slices
    config_data = config_data[str(num_slices)]
    # slice by m
    config_data = (
        config_data.get(str(m))
        or config_data[min(config_data.keys(), key=lambda x: abs(int(x) - m))]
    )
    # slice by k
    config_data = (
        config_data.get(str(k))
        or config_data[min(config_data.keys(), key=lambda x: abs(int(x) - k))]
    )
    # slice by n
    config_data = (
        config_data.get(str(n))
        or config_data[min(config_data.keys(), key=lambda x: abs(int(x) - n))]
    )

    # slice by moe-intermediate-size if applicable
    if moe_intermediate_size is not None:
        i = moe_intermediate_size
        config_data = (
            config_data.get(str(i))
            or config_data[min(config_data.keys(), key=lambda x: abs(int(x) - i))]
        )

    assert config_data is not None
    return config_data


@lru_cache
def supports_pdl(device: torch.device | None = None) -> bool:
    """
    Refer to: https://github.com/triton-lang/triton/blob/v3.5.0/python/tutorials/11-programmatic-dependent-launch.py
    """
    # PDL requires compute capability SM90 or above

    return (
        current_platform.is_cuda()
        and current_platform.has_device_capability(90)
        and not envs.VLLM_LORA_DISABLE_PDL
    )


@lru_cache
def supports_tma(device: torch.device | None = None) -> bool:
    # TMA requires compute capability SM90 or above
    return current_platform.is_cuda() and current_platform.has_device_capability(90)
