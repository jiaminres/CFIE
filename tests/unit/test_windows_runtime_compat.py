"""Regression tests for Windows runtime compatibility branches."""

from __future__ import annotations

import importlib
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import cfie._custom_ops as custom_ops
import cfie.compilation.wrapper as compilation_wrapper
import cfie.config.cfie as cfie_config_mod
import cfie.config.compilation as compilation_config
import cfie.lora.ops.triton_ops.fused_moe_lora_op as fused_moe_lora_op_mod
import cfie.lora.ops.triton_ops.fused_moe_lora_fp8_op as fused_moe_lora_fp8_op_mod
import cfie.lora.ops.triton_ops.fp8_kernel_utils as lora_fp8_kernel_utils_mod
import cfie.lora.ops.triton_ops.kernel_utils as lora_kernel_utils_mod
import cfie.lora.ops.triton_ops.lora_expand_op as lora_expand_mod
import cfie.lora.ops.triton_ops.lora_expand_fp8_op as lora_expand_fp8_mod
import cfie.lora.ops.triton_ops.lora_shrink_op as lora_shrink_mod
import cfie.lora.ops.triton_ops.lora_shrink_fp8_op as lora_shrink_fp8_mod
import cfie.model_executor.layers.fused_moe as fused_moe_pkg
import cfie.model_executor.layers.activation as activation_mod
import cfie.model_executor.layers.batch_invariant as batch_invariant
import cfie.model_executor.layers.fused_moe.activation as fused_moe_activation
import cfie.model_executor.layers.fused_moe.batched_deep_gemm_moe as batched_deep_gemm_moe_mod
import cfie.model_executor.layers.fused_moe.deep_gemm_utils as deep_gemm_utils
import cfie.model_executor.layers.fused_moe.fused_batched_moe as fused_batched_moe_mod
import cfie.model_executor.layers.fused_moe.fused_moe as fused_moe_mod
import cfie.model_executor.layers.fused_moe.gpt_oss_triton_kernels_moe as gpt_oss_triton_moe_mod
import cfie.model_executor.layers.fused_moe.utils as fused_moe_utils
import cfie.model_executor.layers.fused_moe.oracle.unquantized as unquantized_moe_oracle
import cfie.model_executor.layers.fla.ops.chunk as fla_chunk
import cfie.model_executor.layers.fla.ops.chunk_delta_h as fla_chunk_delta_h
import cfie.model_executor.layers.fla.ops.cumsum as fla_cumsum
import cfie.model_executor.layers.fla.ops.chunk_o as fla_chunk_o
import cfie.model_executor.layers.fla.ops.chunk_scaled_dot_kkt as fla_chunk_scaled_dot_kkt
import cfie.model_executor.layers.fla.ops.fused_sigmoid_gating as fused_sigmoid_gating_mod
import cfie.model_executor.layers.fla.ops.fused_recurrent as fused_recurrent
import cfie.model_executor.layers.fla.ops.l2norm as fla_l2norm
import cfie.model_executor.layers.fla.ops.layernorm_guard as fla_layernorm_guard
import cfie.model_executor.layers.fla.ops.op as fla_op
import cfie.model_executor.layers.fla.ops.solve_tril as fla_solve_tril
import cfie.model_executor.layers.fla.ops.wy_fast as fla_wy_fast
import cfie.model_executor.layers.lightning_attn as lightning_attn_mod
import cfie.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm as compressed_tensors_scaled_mm_mod
import cfie.model_executor.layers.quantization.utils.fp8_utils as fp8_utils_mod
import cfie.model_executor.layers.quantization.utils.int8_utils as int8_utils_mod
import cfie.model_executor.layers.quantization.mxfp4 as mxfp4_mod
import cfie.model_executor.layers.quantization.qutlass_utils as qutlass_utils_mod
import cfie.model_executor.layers.mamba.ops.causal_conv1d as causal_conv1d
import cfie.model_executor.layers.mamba.mamba_utils as model_mamba_utils
import cfie.model_executor.layers.mamba.ops.layernorm_gated as mamba_layernorm_gated
import cfie.model_executor.layers.mamba.ops.mamba_ssm as mamba_ssm
import cfie.model_executor.layers.mamba.ops.ssd_bmm as mamba_ssd_bmm
import cfie.model_executor.layers.mamba.ops.ssd_chunk_scan as mamba_ssd_chunk_scan
import cfie.model_executor.layers.mamba.ops.ssd_chunk_state as mamba_ssd_chunk_state
import cfie.model_executor.layers.mamba.ops.ssd_state_passing as mamba_ssd_state_passing
import cfie.model_executor.layers.mamba.ops.triton_helpers as mamba_triton_helpers
import cfie.model_executor.layers.quantization.awq_triton as awq_triton_mod
import cfie.model_executor.models.qwen3_next as qwen3_next
import cfie.v1.worker.utils as worker_utils
import cfie.model_executor.layers.rotary_embedding.mrope as mrope
import cfie.model_executor.layers.rotary_embedding.common as rotary_common
import cfie.cfie_flash_attn.ops.triton.rotary as cfie_flash_attn_rotary_mod
import cfie.v1.worker.gpu_model_runner as gpu_model_runner
import cfie.platforms.cuda as cuda_platform
import cfie.triton_utils.allocation as triton_allocation
import cfie.triton_utils.importing as triton_importing
import cfie.utils.network_utils as network_utils
import cfie.utils.runtime_fallback_trace as runtime_fallback_trace_mod
import cfie.utils.system_utils as system_utils
import cfie.vllm_flash_attn.ops.triton.rotary as vllm_flash_attn_rotary_mod
import cfie.v1.attention.backends.mla.triton_mla as triton_mla
import cfie.v1.attention.backends.mla.sparse_utils as mla_sparse_utils
import cfie.v1.attention.backends.mla.rocm_aiter_mla as rocm_aiter_mla_mod
import cfie.v1.attention.backends.mla.rocm_aiter_mla_sparse as rocm_aiter_mla_sparse_mod
import cfie.v1.attention.backends.rocm_aiter_fa as rocm_aiter_fa_mod
import cfie.v1.attention.backends.flashinfer as flashinfer_mod
import cfie.v1.attention.backends.triton_attn as triton_attn
import cfie.v1.attention.ops.chunked_prefill_paged_decode as chunked_prefill_mod
import cfie.v1.attention.ops.common as attention_common_mod
import cfie.v1.attention.ops.dcp_alltoall as dcp_alltoall_mod
import cfie.v1.attention.ops.xpu_mla_sparse as xpu_mla_sparse_mod
import cfie.v1.attention.ops.rocm_aiter_mla_sparse as rocm_aiter_mla_sparse_ops_mod
import cfie.v1.attention.ops.triton_decode_attention as triton_decode_attention_mod
import cfie.v1.attention.ops.triton_merge_attn_states as triton_merge_attn_states_mod
import cfie.v1.attention.ops.prefix_prefill as prefix_prefill_mod
import cfie.v1.attention.ops.triton_prefill_attention as triton_prefill_attention_mod
import cfie.v1.attention.ops.triton_reshape_and_cache_flash as triton_reshape_cache_flash_mod
import cfie.v1.attention.ops.triton_unified_attention as triton_unified_attention_mod
import cfie.v1.sample.ops.topk_topp_sampler as topk_topp_sampler_mod
import cfie.v1.sample.rejection_sampler as sample_rejection_sampler_mod
import cfie.v1.spec_decode.utils as spec_decode_utils
import cfie.v1.worker.gpu.block_table as block_table_mod
import cfie.v1.worker.gpu.input_batch as input_batch_mod
import cfie.v1.worker.gpu.sample.bad_words as bad_words_mod
import cfie.v1.worker.gpu.metrics.logits as logits_metrics
import cfie.v1.worker.gpu.sample.gumbel as gumbel_mod
import cfie.v1.worker.gpu.sample.logit_bias as logit_bias_mod
import cfie.v1.worker.gpu.sample.logprob as logprob_mod
import cfie.v1.worker.gpu.sample.min_p as min_p_mod
import cfie.v1.worker.gpu.sample.penalties as penalties_mod
import cfie.v1.worker.gpu.sample.prompt_logprob as prompt_logprob_mod
import cfie.v1.worker.gpu.structured_outputs as structured_outputs_mod
import cfie.v1.worker.gpu.buffer_utils as buffer_utils_mod
import cfie.v1.worker.gpu.cp_utils as cp_utils_mod
import cfie.v1.worker.gpu.mm.rope as rope_state_mod
import cfie.v1.worker.gpu.spec_decode.rejection_sampler as spec_rejection_sampler
import cfie.v1.worker.gpu.spec_decode.eagle.speculator as eagle_speculator
import cfie.v1.worker.mamba_utils as worker_mamba_utils
from cfie.distributed.parallel_state import GroupCoordinator
from cfie.platforms import current_platform
from cfie.platforms.cuda import CudaPlatformBase
from cfie.platforms.interface import DeviceCapability
from cfie.config.compilation import CompilationMode
from cfie.v1.attention.backend import AttentionType
from cfie.v1.attention.backends.registry import AttentionBackendEnum
from cfie.v1.kv_cache_interface import MambaSpec
from cfie.v1.attention.selector import AttentionSelectorConfig
from cfie.v1.engine.llm_engine import _should_enable_v1_multiprocessing
from cfie.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    int4_w4a16_moe_quant_config,
    int8_w8a16_moe_quant_config,
)


def _make_cfie_config(
    *,
    world_size: int = 1,
    data_parallel_size: int = 1,
):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            world_size=world_size,
            data_parallel_size=data_parallel_size,
        )
    )


def _pack_rows_for_test(
    logical_qweight: torch.Tensor,
    num_bits: int,
) -> torch.Tensor:
    pack_factor = 32 // num_bits
    size_k, size_n = logical_qweight.shape
    shifts = (
        torch.arange(pack_factor, device=logical_qweight.device, dtype=torch.int64)
        * num_bits
    )
    packed = logical_qweight.to(torch.int64).reshape(
        size_k // pack_factor,
        pack_factor,
        size_n,
    )
    packed = (packed << shifts.view(1, pack_factor, 1)).sum(dim=1)
    return packed.to(torch.int32)


def _pack_cols_for_test(
    logical_qweight: torch.Tensor,
    num_bits: int,
) -> torch.Tensor:
    pack_factor = 32 // num_bits
    size_k, size_n = logical_qweight.shape
    shifts = (
        torch.arange(pack_factor, device=logical_qweight.device, dtype=torch.int64)
        * num_bits
    )
    packed = logical_qweight.to(torch.int64).reshape(
        size_k,
        size_n // pack_factor,
        pack_factor,
    )
    packed = (packed << shifts.view(1, 1, pack_factor)).sum(dim=-1)
    return packed.to(torch.int32)


def _pack_int4_pairs_on_last_dim_for_test(logical_qweight: torch.Tensor) -> torch.Tensor:
    assert logical_qweight.shape[-1] % 2 == 0
    low = logical_qweight[..., 0::2].to(torch.int32)
    high = logical_qweight[..., 1::2].to(torch.int32)
    return (low | (high << 4)).to(torch.int8)


def _pack_int4_pairs_on_first_dim_for_test(logical_qweight: torch.Tensor) -> torch.Tensor:
    assert logical_qweight.shape[0] % 2 == 0
    low = logical_qweight[0::2, :].to(torch.int32)
    high = logical_qweight[1::2, :].to(torch.int32)
    return (low | (high << 4)).to(torch.int8)


def _reference_marlin_repack_from_logical(
    logical_qweight: torch.Tensor,
    num_bits: int,
    is_a_8bit: bool,
) -> torch.Tensor:
    size_k, size_n = logical_qweight.shape
    pack_factor = 32 // num_bits
    out = torch.empty(
        (size_k // 16, size_n * 16 // pack_factor),
        dtype=torch.int32,
        device=logical_qweight.device,
    )
    out_view = out.view(-1)
    target_tile_k = 32 if is_a_8bit else 16
    target_tile_n = 32 if is_a_8bit else 64
    offsets = [0, 1, 8, 9]
    cursor = 0

    for k_tile in range(size_k // target_tile_k):
        for n_tile in range(size_n // target_tile_n):
            tile = logical_qweight[
                k_tile * target_tile_k : (k_tile + 1) * target_tile_k,
                n_tile * target_tile_n : (n_tile + 1) * target_tile_n,
            ]
            for thread_id in range(32):
                tc_col = thread_id // 4
                tc_row = (thread_id % 4) * (4 if is_a_8bit else 2)
                for warp_id in range(4):
                    if is_a_8bit:
                        cur_n = (warp_id // 2) * 16 + tc_col + (warp_id % 2) * 8
                        vals = [int(tile[tc_row + i, cur_n]) for i in range(4)]
                        vals.extend(int(tile[tc_row + 16 + i, cur_n]) for i in range(4))
                    else:
                        cur_n = warp_id * 16 + tc_col
                        rows = [tc_row + offset for offset in offsets]
                        vals = [int(tile[row, cur_n]) for row in rows]
                        vals.extend(int(tile[row, cur_n + 8]) for row in rows)

                    if num_bits == 4:
                        order = (
                            [0, 4, 1, 5, 2, 6, 3, 7]
                            if is_a_8bit
                            else [0, 2, 4, 6, 1, 3, 5, 7]
                        )
                        packed = 0
                        for idx, src in enumerate(order):
                            packed |= (vals[src] & 0xF) << (idx * 4)
                        out_view[cursor] = (
                            packed if packed < 2**31 else packed - 2**32
                        )
                        cursor += 1
                    else:
                        order = [0, 1, 2, 3] if is_a_8bit else [0, 2, 1, 3]
                        packed0 = 0
                        packed1 = 0
                        for idx, src in enumerate(order):
                            packed0 |= (vals[src] & 0xFF) << (idx * 8)
                            packed1 |= (vals[4 + src] & 0xFF) << (idx * 8)
                        out_view[cursor] = (
                            packed0 if packed0 < 2**31 else packed0 - 2**32
                        )
                        out_view[cursor + 1] = (
                            packed1 if packed1 < 2**31 else packed1 - 2**32
                        )
                        cursor += 2

    return out


def _make_hnd_backed_view(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3)


class _FakeCudaTensor(torch.Tensor):
    @property
    def is_cuda(self) -> bool:
        return True


def _as_fake_cuda(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.as_subclass(_FakeCudaTensor)


class _FakeCudaDeviceTensor(_FakeCudaTensor):
    @property
    def device(self) -> torch.device:
        return torch.device("cuda")


def _as_fake_cuda_device(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.as_subclass(_FakeCudaDeviceTensor)


def _cpu_alloc_ignoring_device(factory):
    def _wrapper(*args, **kwargs):
        kwargs = dict(kwargs)
        kwargs.pop("device", None)
        return factory(*args, **kwargs)

    return _wrapper


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

    for token_idx, slot in enumerate(slot_mapping.tolist()):
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

    for token_idx, slot in enumerate(slot_mapping.tolist()):
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


def _iter_lora_segments_for_test(
    token_indices_sorted_by_lora_ids: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    num_active_loras: torch.Tensor,
):
    for lora_idx in range(int(num_active_loras.item())):
        start = int(lora_token_start_loc[lora_idx].item())
        count = int(num_tokens_per_lora[lora_idx].item())
        yield int(lora_ids[lora_idx].item()), token_indices_sorted_by_lora_ids[start : start + count]


def _reference_lora_shrink(
    inputs: torch.Tensor,
    lora_a_weights: list[torch.Tensor],
    token_indices_sorted_by_lora_ids: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    num_active_loras: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    outputs = []
    for lora_a_weight in lora_a_weights:
        slice_output = torch.zeros(
            (inputs.size(0), lora_a_weight.size(1)),
            dtype=inputs.dtype,
        )
        for lora_id, token_indices in _iter_lora_segments_for_test(
            token_indices_sorted_by_lora_ids,
            num_tokens_per_lora,
            lora_token_start_loc,
            lora_ids,
            num_active_loras,
        ):
            slice_output[token_indices] = (
                inputs[token_indices].to(torch.float32)
                @ lora_a_weight[lora_id].to(torch.float32).transpose(0, 1)
                * scaling
            ).to(inputs.dtype)
        outputs.append(slice_output)
    return torch.stack(outputs, dim=0)


def _reference_lora_expand(
    inputs: torch.Tensor,
    lora_b_weights: list[torch.Tensor],
    base_output: torch.Tensor,
    token_indices_sorted_by_lora_ids: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    num_active_loras: torch.Tensor,
    offset_start: int,
    add_inputs: bool,
) -> torch.Tensor:
    output = base_output.clone()
    current_offset = offset_start

    for slice_idx, lora_b_weight in enumerate(lora_b_weights):
        hidden_size = lora_b_weight.size(1)
        slice_output = output[:, current_offset : current_offset + hidden_size]
        for lora_id, token_indices in _iter_lora_segments_for_test(
            token_indices_sorted_by_lora_ids,
            num_tokens_per_lora,
            lora_token_start_loc,
            lora_ids,
            num_active_loras,
        ):
            expanded = (
                inputs[slice_idx, token_indices].to(torch.float32)
                @ lora_b_weight[lora_id].to(torch.float32).transpose(0, 1)
            ).to(base_output.dtype)
            if add_inputs:
                expanded = expanded + slice_output[token_indices]
            slice_output[token_indices] = expanded
        current_offset += hidden_size

    return output


def test_windows_single_rank_can_stay_inproc() -> None:
    cfie_config = _make_cfie_config()

    assert _should_enable_v1_multiprocessing(
        cfie_config,
        requested=False,
    ) is False


def test_windows_multi_rank_still_requires_multiprocessing() -> None:
    assert _should_enable_v1_multiprocessing(
        _make_cfie_config(world_size=2),
        requested=False,
    ) is True
    assert _should_enable_v1_multiprocessing(
        _make_cfie_config(data_parallel_size=2),
        requested=False,
    ) is True
    assert _should_enable_v1_multiprocessing(
        _make_cfie_config(),
        requested=True,
    ) is True


def test_windows_zmq_ipc_path_falls_back_to_loopback_tcp(monkeypatch) -> None:
    monkeypatch.setattr(network_utils.os, "name", "nt", raising=False)
    monkeypatch.setattr(network_utils, "get_loopback_ip", lambda: "127.0.0.1")
    monkeypatch.setattr(network_utils, "get_open_port", lambda: 47001)

    assert network_utils.get_open_zmq_ipc_path() == "tcp://127.0.0.1:47001"


def test_windows_forces_spawn_multiprocessing(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    monkeypatch.setattr(system_utils.os, "name", "nt", raising=False)
    monkeypatch.setattr(system_utils, "is_in_ray_actor", lambda: False)
    monkeypatch.setattr(system_utils, "cuda_is_initialized", lambda: False)
    monkeypatch.setattr(system_utils, "xpu_is_initialized", lambda: False)
    monkeypatch.setattr(system_utils, "in_wsl", lambda: False)

    system_utils._maybe_force_spawn()

    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"


def test_single_rank_group_coordinator_works_without_distributed(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "cfie.distributed.parallel_state.torch.distributed.is_initialized",
        lambda: False,
    )
    monkeypatch.setattr(current_platform, "is_cuda_alike", lambda: False)
    monkeypatch.setattr(current_platform, "is_xpu", lambda: False)
    monkeypatch.setattr(current_platform, "is_out_of_tree", lambda: False)

    group = GroupCoordinator(
        group_ranks=[[0]],
        local_rank=0,
        torch_distributed_backend="gloo",
        use_device_communicator=False,
        group_name="test_local_world",
    )

    assert group.rank == 0
    assert group.local_rank == 0
    assert group.world_size == 1
    assert group.rank_in_group == 0
    assert group.cpu_group is None
    assert group.device_group is None
    assert group.device.type == "cpu"
    assert group.backend == "gloo"
    group.destroy()


def test_reshape_and_cache_flash_torch_fallback_handles_hnd_layout(
    monkeypatch,
) -> None:
    monkeypatch.setattr(custom_ops, "_get_torch_op", lambda namespace, name: None)

    key = torch.arange(3 * 2 * 4, dtype=torch.float16).view(3, 2, 4)
    value = (50 + torch.arange(3 * 2 * 4, dtype=torch.float16)).view(3, 2, 4)
    key_cache = _make_hnd_backed_view(torch.zeros((2, 4, 2, 4), dtype=torch.float16))
    value_cache = _make_hnd_backed_view(
        torch.zeros((2, 4, 2, 4), dtype=torch.float16)
    )
    slot_mapping = torch.tensor([0, 5, -1], dtype=torch.long)
    scales = torch.ones(1, dtype=torch.float32)

    custom_ops.reshape_and_cache_flash(
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
    torch.testing.assert_close(key_cache, expected_key, rtol=0.0, atol=0.0)
    torch.testing.assert_close(value_cache, expected_value, rtol=0.0, atol=0.0)


def test_reshape_and_cache_torch_fallback_handles_head_major_layout(
    monkeypatch,
) -> None:
    monkeypatch.setattr(custom_ops, "_get_torch_op", lambda namespace, name: None)

    key = torch.arange(3 * 2 * 8, dtype=torch.float16).view(3, 2, 8)
    value = (100 + torch.arange(3 * 2 * 8, dtype=torch.float16)).view(3, 2, 8)
    key_cache = torch.zeros((2, 2, 1, 4, 8), dtype=torch.float16)
    value_cache = torch.zeros((2, 2, 8, 4), dtype=torch.float16)
    slot_mapping = torch.tensor([0, 5, -1], dtype=torch.long)
    scales = torch.ones(1, dtype=torch.float32)

    custom_ops.reshape_and_cache(
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
    torch.testing.assert_close(key_cache, expected_key, rtol=0.0, atol=0.0)
    torch.testing.assert_close(value_cache, expected_value, rtol=0.0, atol=0.0)


def test_reshape_and_cache_flash_diffkv_torch_fallback_handles_hnd_layout(
    monkeypatch,
) -> None:
    monkeypatch.setattr(custom_ops, "_get_torch_op", lambda namespace, name: None)

    key = torch.arange(3 * 2 * 4, dtype=torch.float16).view(3, 2, 4)
    value = (200 + torch.arange(3 * 2 * 3, dtype=torch.float16)).view(3, 2, 3)
    kv_cache = _make_hnd_backed_view(torch.zeros((2, 4, 2, 7), dtype=torch.float16))
    slot_mapping = torch.tensor([1, -1, 6], dtype=torch.long)
    scales = torch.ones(1, dtype=torch.float32)

    custom_ops.reshape_and_cache_flash_diffkv(
        key,
        value,
        kv_cache,
        slot_mapping,
        "auto",
        scales,
        scales,
    )

    key_cache = kv_cache[..., : key.shape[-1]]
    value_cache = kv_cache[..., key.shape[-1] :]
    expected_key, expected_value = _manual_fill_cache(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
    )
    torch.testing.assert_close(key_cache, expected_key, rtol=0.0, atol=0.0)
    torch.testing.assert_close(value_cache, expected_value, rtol=0.0, atol=0.0)


def test_single_rank_group_coordinator_rejects_multi_rank_shortcut(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "cfie.distributed.parallel_state.torch.distributed.is_initialized",
        lambda: False,
    )

    with pytest.raises(RuntimeError, match="single-rank local group"):
        GroupCoordinator(
            group_ranks=[[0, 1]],
            local_rank=0,
            torch_distributed_backend="gloo",
            use_device_communicator=False,
            group_name="test_invalid_local_world",
        )


def test_validate_block_size_auto_expands_mamba_align_scheduler_budget() -> None:
    cfg = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=2096, mamba_cache_mode="align"),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=2048,
            long_prefill_token_threshold=0,
            disable_chunked_mm_input=False,
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            dcp_kv_cache_interleave_size=1,
            cp_kv_cache_interleave_size=1,
        ),
    )

    cfie_config_mod.CfieConfig.validate_block_size(cfg)

    assert cfg.scheduler_config.max_num_batched_tokens == 2096


def test_triton_attention_backend_accepts_reference_runtime_compat(
    monkeypatch,
) -> None:
    monkeypatch.setattr(triton_attn, "HAS_TRITON", False)
    monkeypatch.setattr(triton_attn.current_platform, "is_cuda", lambda: True)

    reasons = triton_attn.TritonAttentionBackend.validate_configuration(
        head_size=64,
        dtype=torch.float16,
        kv_cache_dtype="auto",
        block_size=16,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=False,
        use_per_head_quant_scales=False,
        device_capability=DeviceCapability(9, 0),
        attn_type=AttentionType.DECODER,
    )

    assert reasons == []


def test_triton_attention_backend_accepts_flash_attn_runtime_compat(
    monkeypatch,
) -> None:
    monkeypatch.setattr(triton_attn, "HAS_TRITON", False)
    monkeypatch.setattr(triton_attn.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(triton_attn, "is_flash_attn_varlen_func_available", lambda: True)
    monkeypatch.setattr(triton_attn, "get_flash_attn_version", lambda **_: 2)
    monkeypatch.setattr(triton_attn, "flash_attn_supports_sinks", lambda: False)

    reasons = triton_attn.TritonAttentionBackend.validate_configuration(
        head_size=64,
        dtype=torch.float16,
        kv_cache_dtype="auto",
        block_size=16,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=False,
        use_per_head_quant_scales=False,
        device_capability=DeviceCapability(9, 0),
        attn_type=AttentionType.DECODER,
    )

    assert reasons == []


def test_triton_attention_backend_keeps_mm_prefix_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(triton_attn, "HAS_TRITON", False)
    monkeypatch.setattr(triton_attn.current_platform, "is_cuda", lambda: True)

    reasons = triton_attn.TritonAttentionBackend.validate_configuration(
        head_size=64,
        dtype=torch.float16,
        kv_cache_dtype="auto",
        block_size=16,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=True,
        use_per_head_quant_scales=False,
        device_capability=DeviceCapability(9, 0),
        attn_type=AttentionType.DECODER,
    )

    assert reasons == []


def test_triton_mla_backend_accepts_reference_runtime_compat(monkeypatch) -> None:
    monkeypatch.setattr(triton_mla, "HAS_TRITON", False)

    reasons = triton_mla.TritonMLABackend.validate_configuration(
        head_size=576,
        dtype=torch.bfloat16,
        kv_cache_dtype="auto",
        block_size=16,
        use_mla=True,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=False,
        use_per_head_quant_scales=False,
        device_capability=DeviceCapability(9, 0),
        attn_type=AttentionType.DECODER,
    )

    assert reasons == []


def test_flashinfer_backend_rejects_missing_python_package(monkeypatch) -> None:
    monkeypatch.setattr(flashinfer_mod, "HAS_FLASHINFER", False)

    reasons = flashinfer_mod.FlashInferBackend.validate_configuration(
        head_size=64,
        dtype=torch.float16,
        kv_cache_dtype="auto",
        block_size=16,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=False,
        use_per_head_quant_scales=False,
        device_capability=DeviceCapability(9, 0),
        attn_type=AttentionType.DECODER,
    )

    assert reasons == ["flashinfer python package not installed"]


def test_flashinfer_kvfp8_dequant_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(flashinfer_mod, "HAS_TRITON", False)

    def _raise_if_triton_path_runs(*_args, **_kwargs):
        raise AssertionError("FlashInfer Triton KV dequant path should not run")

    monkeypatch.setattr(
        flashinfer_mod,
        "_trtllm_prefill_attn_kvfp8_dequant",
        _raise_if_triton_path_runs,
    )

    kv_cache = torch.tensor(
        [
            [
                [[1.0, 2.0]],
                [[10.0, 20.0]],
            ],
            [
                [[3.0, 4.0]],
                [[30.0, 40.0]],
            ],
            [
                [[5.0, 6.0]],
                [[50.0, 60.0]],
            ],
        ],
        dtype=torch.float16,
    ).reshape(3, 2, 1, 1, 2)
    block_tables_prefill = torch.tensor([[2, 0], [1, 2]], dtype=torch.int32)
    k_scale = torch.tensor(0.5, dtype=torch.float32)
    v_scale = torch.tensor(0.25, dtype=torch.float32)

    mock_kv_cache, mock_block_table = flashinfer_mod.trtllm_prefill_attn_kvfp8_dequant(
        kv_cache,
        block_tables_prefill,
        k_scale,
        v_scale,
        dequant_dtype=torch.float16,
    )

    expected_mock_block_table = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    torch.testing.assert_close(mock_block_table.cpu(), expected_mock_block_table)

    expected_slot_1 = torch.tensor(
        [[[[2.5, 3.0]]], [[[12.5, 15.0]]]],
        dtype=torch.float16,
    )
    expected_slot_3 = torch.tensor(
        [[[[1.5, 2.0]]], [[[7.5, 10.0]]]],
        dtype=torch.float16,
    )
    expected_slot_4 = expected_slot_1

    torch.testing.assert_close(mock_kv_cache[1].cpu(), expected_slot_1)
    torch.testing.assert_close(mock_kv_cache[3].cpu(), expected_slot_3)
    torch.testing.assert_close(mock_kv_cache[4].cpu(), expected_slot_4)


def test_cuda_backend_selection_keeps_triton_mla_without_runtime_when_reference_fallback_available(
    monkeypatch,
) -> None:
    monkeypatch.setattr(triton_mla, "HAS_TRITON", False)
    monkeypatch.setattr(
        cuda_platform,
        "_get_backend_priorities",
        lambda use_mla, device_capability, num_heads: [
            AttentionBackendEnum.TRITON_MLA
        ],
    )

    valid_backends, invalid_reasons = CudaPlatformBase.get_valid_backends(
        device_capability=DeviceCapability(9, 0),
        attn_selector_config=AttentionSelectorConfig(
            head_size=576,
            dtype=torch.bfloat16,
            kv_cache_dtype="auto",
            block_size=16,
            use_mla=True,
            has_sink=False,
            use_sparse=False,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            attn_type=AttentionType.DECODER,
        ),
        num_heads=128,
    )

    assert valid_backends == [(AttentionBackendEnum.TRITON_MLA, 0)]
    assert invalid_reasons == {}


def test_triton_decode_attention_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(triton_decode_attention_mod, "HAS_TRITON", False)

    q = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]]],
        dtype=torch.float32,
    )
    k_buffer = torch.tensor(
        [
            [[[1.0, 0.0]], [[0.0, 1.0]]],
            [[[1.0, 1.0]], [[0.0, 0.0]]],
        ],
        dtype=torch.float32,
    )
    v_buffer = torch.tensor(
        [
            [[[10.0]], [[20.0]]],
            [[[30.0]], [[0.0]]],
        ],
        dtype=torch.float32,
    )
    o = torch.empty((1, 2, 1), dtype=torch.float32)
    lse = torch.empty((1, 2), dtype=torch.float32)
    req_to_token = torch.tensor([[0, 1]], dtype=torch.int32)
    b_seq_len = torch.tensor([3], dtype=torch.int32)
    attn_logits = torch.empty((1, 2, 4, 2), dtype=torch.float32)

    triton_decode_attention_mod.decode_attention_fwd(
        q=q,
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        o=o,
        lse=lse,
        req_to_token=req_to_token,
        b_seq_len=b_seq_len,
        attn_logits=attn_logits,
        num_kv_splits=4,
        sm_scale=1.0,
        page_size=2,
    )

    scores_h0 = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    scores_h1 = torch.tensor([0.0, 1.0, 1.0], dtype=torch.float32)
    probs_h0 = torch.softmax(scores_h0, dim=0)
    probs_h1 = torch.softmax(scores_h1, dim=0)
    values = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32)
    expected_o = torch.tensor(
        [[
            [(probs_h0 * values).sum().item()],
            [(probs_h1 * values).sum().item()],
        ]],
        dtype=torch.float32,
    )
    expected_lse = torch.stack(
        [torch.logsumexp(scores_h0, dim=0), torch.logsumexp(scores_h1, dim=0)]
    ).unsqueeze(0)

    assert torch.allclose(o, expected_o, atol=1e-5)
    assert torch.allclose(lse, expected_lse, atol=1e-5)
    assert torch.count_nonzero(attn_logits) == 0


def test_triton_decode_attention_normal_direct_falls_back_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(triton_decode_attention_mod, "HAS_TRITON", False)

    called = {"value": False}

    def _fake_reference(**kwargs) -> None:
        called["value"] = True
        kwargs["o"].fill_(1.5)
        kwargs["lse"].fill_(2.5)
        kwargs["attn_logits"].zero_()

    monkeypatch.setattr(
        triton_decode_attention_mod,
        "_reference_decode_attention_fwd",
        _fake_reference,
    )

    o = torch.empty((1, 1, 1), dtype=torch.float32)
    lse = torch.empty((1, 1), dtype=torch.float32)
    attn_logits = torch.empty((1, 1, 2, 2), dtype=torch.float32)

    triton_decode_attention_mod.decode_attention_fwd_normal(
        q=torch.tensor([[[1.0, 0.0]]], dtype=torch.float32),
        k_buffer=torch.tensor([[[[1.0, 0.0]]], [[[0.0, 1.0]]]], dtype=torch.float32),
        v_buffer=torch.tensor([[[[10.0]]], [[[20.0]]]], dtype=torch.float32),
        o=o,
        lse=lse,
        req_to_token=torch.tensor([[0, 1]], dtype=torch.int32),
        b_seq_len=torch.tensor([2], dtype=torch.int32),
        attn_logits=attn_logits,
        num_kv_splits=2,
        sm_scale=1.0,
        page_size=1,
    )

    assert called["value"] is True
    torch.testing.assert_close(o, torch.full_like(o, 1.5))
    torch.testing.assert_close(lse, torch.full_like(lse, 2.5))
    torch.testing.assert_close(attn_logits, torch.zeros_like(attn_logits))


def test_triton_decode_attention_grouped_direct_falls_back_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(triton_decode_attention_mod, "HAS_TRITON", False)

    called = {"value": False}

    def _fake_reference(**kwargs) -> None:
        called["value"] = True
        kwargs["o"].fill_(3.0)
        kwargs["lse"].fill_(4.0)
        kwargs["attn_logits"].zero_()

    monkeypatch.setattr(
        triton_decode_attention_mod,
        "_reference_decode_attention_fwd",
        _fake_reference,
    )

    o = torch.empty((1, 2, 1), dtype=torch.float32)
    lse = torch.empty((1, 2), dtype=torch.float32)
    attn_logits = torch.empty((1, 2, 2, 2), dtype=torch.float32)

    triton_decode_attention_mod.decode_attention_fwd_grouped(
        q=torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32),
        k_buffer=torch.tensor([[[[1.0, 0.0]]], [[[0.0, 1.0]]]], dtype=torch.float32),
        v_buffer=torch.tensor([[[[10.0]]], [[[20.0]]]], dtype=torch.float32),
        o=o,
        lse=lse,
        req_to_token=torch.tensor([[0, 1]], dtype=torch.int32),
        b_seq_len=torch.tensor([2], dtype=torch.int32),
        attn_logits=attn_logits,
        num_kv_splits=2,
        sm_scale=1.0,
        page_size=1,
    )

    assert called["value"] is True
    torch.testing.assert_close(o, torch.full_like(o, 3.0))
    torch.testing.assert_close(lse, torch.full_like(lse, 4.0))
    torch.testing.assert_close(attn_logits, torch.zeros_like(attn_logits))


def test_triton_unified_attention_prefers_paged_attention_v1_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(triton_unified_attention_mod, "HAS_TRITON", False)
    monkeypatch.setattr(
        triton_unified_attention_mod,
        "_get_flash_attn_unified_attention_version",
        lambda *args, **kwargs: None,
    )

    def _raise_if_reference_path_runs(**_kwargs):
        raise AssertionError("Unified attention reference path should not run")

    calls: dict[str, object] = {}

    def _fake_paged_attention_v1(**kwargs):
        calls["num_kv_heads"] = kwargs["num_kv_heads"]
        calls["block_size"] = kwargs["block_size"]
        kwargs["out"].fill_(3.0)

    monkeypatch.setattr(
        triton_unified_attention_mod,
        "_reference_unified_attention",
        _raise_if_reference_path_runs,
    )
    monkeypatch.setattr(
        triton_unified_attention_mod.ops,
        "paged_attention_v1",
        _fake_paged_attention_v1,
        raising=False,
    )

    q = torch.randn((2, 2, 4), dtype=torch.float32)
    k = torch.randn((1, 8, 1, 4), dtype=torch.float32)
    v = torch.randn((1, 8, 1, 4), dtype=torch.float32)
    out = torch.empty_like(q)
    cu_seqlens_q = torch.tensor([0, 1, 2], dtype=torch.int32)
    seqused_k = torch.tensor([1, 1], dtype=torch.int32)
    block_table = torch.tensor([[0], [0]], dtype=torch.int32)
    descale = torch.tensor([1.0], dtype=torch.float32)

    triton_unified_attention_mod.unified_attention(
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        seqused_k=seqused_k,
        max_seqlen_k=1,
        softmax_scale=1.0,
        causal=True,
        window_size=None,
        block_table=block_table,
        softcap=0.0,
        q_descale=None,
        k_descale=descale,
        v_descale=descale,
    )

    assert calls == {"num_kv_heads": 1, "block_size": 8}
    torch.testing.assert_close(out, torch.full_like(out, 3.0))


def test_triton_unified_attention_falls_back_to_reference_when_fastpath_fails(
    monkeypatch,
) -> None:
    monkeypatch.setattr(triton_unified_attention_mod, "HAS_TRITON", False)
    monkeypatch.setattr(
        triton_unified_attention_mod,
        "_get_flash_attn_unified_attention_version",
        lambda *args, **kwargs: None,
    )

    def _raise_fastpath(*_args, **_kwargs):
        raise RuntimeError("paged_attention_v1 unavailable")

    calls = {"reference": False}

    def _fake_reference_unified_attention(**kwargs):
        calls["reference"] = True
        kwargs["out"].fill_(5.0)

    monkeypatch.setattr(
        triton_unified_attention_mod.ops,
        "paged_attention_v1",
        _raise_fastpath,
        raising=False,
    )
    monkeypatch.setattr(
        triton_unified_attention_mod,
        "_reference_unified_attention",
        _fake_reference_unified_attention,
    )

    q = torch.randn((2, 2, 4), dtype=torch.float32)
    k = torch.randn((1, 8, 1, 4), dtype=torch.float32)
    v = torch.randn((1, 8, 1, 4), dtype=torch.float32)
    out = torch.empty_like(q)
    cu_seqlens_q = torch.tensor([0, 1, 2], dtype=torch.int32)
    seqused_k = torch.tensor([1, 1], dtype=torch.int32)
    block_table = torch.tensor([[0], [0]], dtype=torch.int32)
    descale = torch.tensor([1.0], dtype=torch.float32)

    triton_unified_attention_mod.unified_attention(
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        seqused_k=seqused_k,
        max_seqlen_k=1,
        softmax_scale=1.0,
        causal=True,
        window_size=None,
        block_table=block_table,
        softcap=0.0,
        q_descale=None,
        k_descale=descale,
        v_descale=descale,
    )

    assert calls["reference"] is True
    torch.testing.assert_close(out, torch.full_like(out, 5.0))


def test_triton_prefill_attention_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(triton_prefill_attention_mod, "HAS_TRITON", False)

    q = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
            [[1.0, 1.0], [1.0, -1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    k = torch.tensor(
        [
            [[1.0, 0.0]],
            [[0.0, 1.0]],
            [[1.0, 1.0]],
            [[1.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    v = torch.tensor(
        [
            [[10.0, 1.0]],
            [[20.0, 2.0]],
            [[30.0, 3.0]],
            [[40.0, 4.0]],
        ],
        dtype=torch.float32,
    )
    o = torch.empty_like(q)
    b_start_loc = torch.tensor([0, 3], dtype=torch.int32)
    b_seq_len = torch.tensor([3, 1], dtype=torch.int32)
    softmax_scale = 1.0

    triton_prefill_attention_mod.context_attention_fwd(
        q=q,
        k=k,
        v=v,
        o=o,
        b_start_loc=b_start_loc,
        b_seq_len=b_seq_len,
        max_input_len=3,
        is_causal=True,
        softmax_scale=softmax_scale,
        sliding_window_q=1,
        sliding_window_k=None,
    )

    expected = torch.empty_like(o)
    for seq_start, seq_len in zip(b_start_loc.tolist(), b_seq_len.tolist(), strict=True):
        seq_stop = seq_start + seq_len
        q_seq = q[seq_start:seq_stop].to(torch.float32).permute(1, 0, 2)
        k_seq = k[seq_start:seq_stop].to(torch.float32).permute(1, 0, 2)
        v_seq = v[seq_start:seq_stop].to(torch.float32).permute(1, 0, 2)
        k_seq = k_seq.repeat_interleave(q.shape[1] // k.shape[1], dim=0)
        v_seq = v_seq.repeat_interleave(q.shape[1] // v.shape[1], dim=0)
        scores = torch.matmul(q_seq, k_seq.transpose(-1, -2)) * softmax_scale

        positions = torch.arange(seq_len, dtype=torch.int64)
        q_pos = positions[:, None]
        k_pos = positions[None, :]
        mask = (q_pos >= k_pos) & ((q_pos - k_pos) <= 1)
        mask = mask.unsqueeze(0)

        masked_scores = scores.masked_fill(~mask, float("-inf"))
        row_max = masked_scores.amax(dim=-1, keepdim=True)
        row_max = torch.where(
            torch.isfinite(row_max), row_max, torch.zeros_like(row_max)
        )
        exp_scores = torch.exp(masked_scores - row_max) * mask.to(torch.float32)
        probs = exp_scores / exp_scores.sum(dim=-1, keepdim=True)
        expected[seq_start:seq_stop] = torch.matmul(probs, v_seq).permute(1, 0, 2)

    assert torch.allclose(o, expected, atol=1e-5)


def test_triton_prefill_attention_prefers_precompiled_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(triton_prefill_attention_mod, "HAS_TRITON", False)

    calls: dict[str, object] = {}

    def _fake_precompiled(**kwargs):
        calls["is_causal"] = kwargs["is_causal"]
        calls["softmax_scale"] = kwargs["softmax_scale"]
        calls["sliding_window_q"] = kwargs["sliding_window_q"]
        calls["sliding_window_k"] = kwargs["sliding_window_k"]
        kwargs["output"].fill_(11.0)

    monkeypatch.setattr(
        triton_prefill_attention_mod,
        "_supports_precompiled_prefill_attention",
        lambda **_: True,
    )
    monkeypatch.setattr(
        triton_prefill_attention_mod.ops,
        "prefill_attention_precompiled",
        _fake_precompiled,
    )
    monkeypatch.setattr(
        triton_prefill_attention_mod,
        "_reference_context_attention_fwd",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("reference fallback should not run")
        ),
    )

    q = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
            [[1.0, 1.0], [1.0, -1.0]],
        ],
        dtype=torch.float32,
    )
    k = torch.tensor(
        [
            [[1.0, 0.0]],
            [[0.0, 1.0]],
            [[1.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    v = torch.tensor(
        [
            [[10.0, 1.0]],
            [[20.0, 2.0]],
            [[30.0, 3.0]],
        ],
        dtype=torch.float32,
    )
    o = torch.empty_like(q)

    triton_prefill_attention_mod.context_attention_fwd(
        q=q,
        k=k,
        v=v,
        o=o,
        b_start_loc=torch.tensor([0], dtype=torch.int32),
        b_seq_len=torch.tensor([3], dtype=torch.int32),
        max_input_len=3,
        is_causal=False,
        softmax_scale=0.5,
        sliding_window_q=None,
        sliding_window_k=2,
    )

    assert calls == {
        "is_causal": False,
        "softmax_scale": 0.5,
        "sliding_window_q": 0,
        "sliding_window_k": 2,
    }
    torch.testing.assert_close(o, torch.full_like(o, 11.0))


def test_prefix_prefill_attention_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(prefix_prefill_mod, "HAS_TRITON", False)

    q = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    k = torch.tensor(
        [
            [[1.0, 1.0]],
            [[1.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    v = torch.tensor(
        [
            [[30.0, 3.0]],
            [[40.0, 4.0]],
        ],
        dtype=torch.float32,
    )
    key_cache = torch.tensor(
        [[[[[1.0], [0.0]], [[0.0], [1.0]]]]],
        dtype=torch.float32,
    )
    value_cache = torch.tensor(
        [[[[10.0, 20.0], [1.0, 2.0]]]],
        dtype=torch.float32,
    )
    o = torch.empty_like(q)
    b_loc = torch.tensor([[0]], dtype=torch.int32)
    b_start_loc = torch.tensor([0, 2], dtype=torch.int32)
    b_seq_len = torch.tensor([4], dtype=torch.int32)

    prefix_prefill_mod.context_attention_fwd(
        q=q,
        k=k,
        v=v,
        o=o,
        kv_cache_dtype="auto",
        k_cache=key_cache,
        v_cache=value_cache,
        b_loc=b_loc,
        b_start_loc=b_start_loc,
        b_seq_len=b_seq_len,
        max_seq_len=4,
        max_input_len=2,
        k_scale=torch.tensor([1.0]),
        v_scale=torch.tensor([1.0]),
        sliding_window=0,
        sm_scale=1.0,
        skip_decode=False,
    )

    context_k = torch.tensor(
        [
            [[1.0, 0.0]],
            [[0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    context_v = torch.tensor(
        [
            [[10.0, 1.0]],
            [[20.0, 2.0]],
        ],
        dtype=torch.float32,
    )
    all_k = torch.cat([context_k, k], dim=0)
    all_v = torch.cat([context_v, v], dim=0)
    expected = torch.empty_like(o)
    q_heads = q.permute(1, 0, 2)
    k_heads = all_k.permute(1, 0, 2).repeat_interleave(q.shape[1], dim=0)
    v_heads = all_v.permute(1, 0, 2).repeat_interleave(q.shape[1], dim=0)
    query_positions = torch.tensor([2, 3], dtype=torch.int64)
    key_positions = torch.arange(4, dtype=torch.int64)
    mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    scores = torch.matmul(q_heads, k_heads.transpose(-1, -2))
    scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    expected.copy_(torch.matmul(probs, v_heads).permute(1, 0, 2))

    assert torch.allclose(o, expected, atol=1e-5)


def test_prefix_prefill_attention_prefers_precompiled_gather_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(prefix_prefill_mod, "HAS_TRITON", False)

    calls: dict[str, object] = {}

    def _fake_gather(**kwargs):
        calls["batch_size"] = kwargs["batch_size"]
        kwargs["gathered_key"].copy_(
            torch.tensor(
                [
                    [[1.0, 0.0]],
                    [[0.0, 1.0]],
                ],
                dtype=kwargs["gathered_key"].dtype,
                device=kwargs["gathered_key"].device,
            )
        )
        kwargs["gathered_value"].copy_(
            torch.tensor(
                [
                    [[10.0, 1.0]],
                    [[20.0, 2.0]],
                ],
                dtype=kwargs["gathered_value"].dtype,
                device=kwargs["gathered_value"].device,
            )
        )

    monkeypatch.setattr(prefix_prefill_mod.ops, "gather_paged_kv_cache", _fake_gather)
    monkeypatch.setattr(
        prefix_prefill_mod,
        "_supports_precompiled_paged_kv_gather",
        lambda **_: True,
    )
    monkeypatch.setattr(
        prefix_prefill_mod,
        "_reshape_key_cache_for_reference",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("reference cache reshape should not run")
        ),
    )
    monkeypatch.setattr(
        prefix_prefill_mod,
        "_reshape_value_cache_for_reference",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("reference cache reshape should not run")
        ),
    )

    q = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    k = torch.tensor(
        [
            [[1.0, 1.0]],
            [[1.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    v = torch.tensor(
        [
            [[30.0, 3.0]],
            [[40.0, 4.0]],
        ],
        dtype=torch.float32,
    )
    o = torch.empty_like(q)

    prefix_prefill_mod.context_attention_fwd(
        q=q,
        k=k,
        v=v,
        o=o,
        kv_cache_dtype="auto",
        k_cache=torch.empty((1, 1, 2, 2, 1), dtype=torch.float32),
        v_cache=torch.empty((1, 1, 2, 2), dtype=torch.float32),
        b_loc=torch.tensor([[0]], dtype=torch.int32),
        b_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        b_seq_len=torch.tensor([4], dtype=torch.int32),
        max_seq_len=4,
        max_input_len=2,
        k_scale=torch.tensor([1.0]),
        v_scale=torch.tensor([1.0]),
        sliding_window=0,
        sm_scale=1.0,
        skip_decode=False,
    )

    assert calls["batch_size"] == 1
    assert torch.isfinite(o).all()


def test_prefix_prefill_attention_prefers_sdpa_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(prefix_prefill_mod, "HAS_TRITON", False)

    calls: dict[str, object] = {}

    def _fake_sdpa(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
    ) -> torch.Tensor:
        calls["q_shape"] = tuple(q.shape)
        calls["mask_shape"] = tuple(attn_mask.shape) if attn_mask is not None else None
        calls["scale"] = scale
        return torch.full_like(q, 7.0)

    monkeypatch.setattr(
        prefix_prefill_mod,
        "_supports_reference_sdpa_fastpath",
        lambda **_: True,
    )
    monkeypatch.setattr(prefix_prefill_mod.F, "scaled_dot_product_attention", _fake_sdpa)

    q = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    k = torch.tensor(
        [
            [[1.0, 1.0]],
            [[1.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    v = torch.tensor(
        [
            [[30.0, 3.0]],
            [[40.0, 4.0]],
        ],
        dtype=torch.float32,
    )
    key_cache = torch.tensor(
        [[[[[1.0], [0.0]], [[0.0], [1.0]]]]],
        dtype=torch.float32,
    )
    value_cache = torch.tensor(
        [[[[10.0, 20.0], [1.0, 2.0]]]],
        dtype=torch.float32,
    )
    o = torch.empty_like(q)

    prefix_prefill_mod.context_attention_fwd(
        q=q,
        k=k,
        v=v,
        o=o,
        kv_cache_dtype="auto",
        k_cache=key_cache,
        v_cache=value_cache,
        b_loc=torch.tensor([[0]], dtype=torch.int32),
        b_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        b_seq_len=torch.tensor([4], dtype=torch.int32),
        max_seq_len=4,
        max_input_len=2,
        k_scale=torch.tensor([1.0]),
        v_scale=torch.tensor([1.0]),
        sliding_window=0,
        sm_scale=1.0,
        skip_decode=False,
    )

    assert calls == {
        "q_shape": (1, 2, 2, 2),
        "mask_shape": (1, 1, 2, 4),
        "scale": 1.0,
    }
    torch.testing.assert_close(o, torch.full_like(o, 7.0))


def test_prefix_prefill_attention_prefers_precompiled_compute_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(prefix_prefill_mod, "HAS_TRITON", False)

    calls: dict[str, object] = {}

    def _fake_gather(**kwargs):
        kwargs["gathered_key"].copy_(
            torch.tensor(
                [
                    [[1.0, 0.0]],
                    [[0.0, 1.0]],
                ],
                dtype=kwargs["gathered_key"].dtype,
                device=kwargs["gathered_key"].device,
            )
        )
        kwargs["gathered_value"].copy_(
            torch.tensor(
                [
                    [[10.0, 1.0]],
                    [[20.0, 2.0]],
                ],
                dtype=kwargs["gathered_value"].dtype,
                device=kwargs["gathered_value"].device,
            )
        )

    def _fake_compute(**kwargs):
        calls["q_shape"] = tuple(kwargs["q"].shape)
        calls["cu_ctx_lens"] = kwargs["cu_ctx_lens"].tolist()
        calls["sliding_window"] = kwargs["sliding_window"]
        kwargs["output"].fill_(9.0)

    monkeypatch.setattr(prefix_prefill_mod.ops, "gather_paged_kv_cache", _fake_gather)
    monkeypatch.setattr(
        prefix_prefill_mod,
        "_supports_precompiled_paged_kv_gather",
        lambda **_: True,
    )
    monkeypatch.setattr(
        prefix_prefill_mod,
        "_supports_precompiled_prefix_prefill_attention",
        lambda **_: True,
    )
    monkeypatch.setattr(
        prefix_prefill_mod.ops,
        "prefix_prefill_attention_precompiled",
        _fake_compute,
    )
    monkeypatch.setattr(
        prefix_prefill_mod.F,
        "scaled_dot_product_attention",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("SDPA fallback should not run")
        ),
    )

    q = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    k = torch.tensor(
        [
            [[1.0, 1.0]],
            [[1.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    v = torch.tensor(
        [
            [[30.0, 3.0]],
            [[40.0, 4.0]],
        ],
        dtype=torch.float32,
    )
    o = torch.empty_like(q)

    prefix_prefill_mod.context_attention_fwd(
        q=q,
        k=k,
        v=v,
        o=o,
        kv_cache_dtype="auto",
        k_cache=torch.empty((1, 1, 2, 2, 1), dtype=torch.float32),
        v_cache=torch.empty((1, 1, 2, 2), dtype=torch.float32),
        b_loc=torch.tensor([[0]], dtype=torch.int32),
        b_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        b_seq_len=torch.tensor([4], dtype=torch.int32),
        max_seq_len=4,
        max_input_len=2,
        k_scale=torch.tensor([1.0]),
        v_scale=torch.tensor([1.0]),
        sliding_window=0,
        sm_scale=1.0,
        skip_decode=False,
    )

    assert calls == {
        "q_shape": (2, 2, 2),
        "cu_ctx_lens": [0, 2],
        "sliding_window": 0,
    }
    torch.testing.assert_close(o, torch.full_like(o, 9.0))


def test_chunked_prefill_paged_decode_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(chunked_prefill_mod, "HAS_TRITON", False)
    monkeypatch.setattr(prefix_prefill_mod, "HAS_TRITON", False)

    query = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    key = torch.tensor(
        [
            [[1.0, 1.0]],
            [[1.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    value = torch.tensor(
        [
            [[30.0, 3.0]],
            [[40.0, 4.0]],
        ],
        dtype=torch.float32,
    )
    key_cache = torch.tensor(
        [[[[[1.0], [0.0]], [[0.0], [1.0]]]]],
        dtype=torch.float32,
    )
    value_cache = torch.tensor(
        [[[[10.0, 20.0], [1.0, 2.0]]]],
        dtype=torch.float32,
    )
    output = torch.empty_like(query)
    block_table = torch.tensor([[0]], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 2], dtype=torch.int32)
    seq_lens = torch.tensor([4], dtype=torch.int32)

    chunked_prefill_mod.chunked_prefill_paged_decode(
        query=query,
        key=key,
        value=value,
        output=output,
        kv_cache_dtype="auto",
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        max_seq_len=4,
        max_query_len=2,
        k_scale=torch.tensor([1.0]),
        v_scale=torch.tensor([1.0]),
        sliding_window=0,
        sm_scale=1.0,
    )

    expected = torch.empty_like(output)
    prefix_prefill_mod.context_attention_fwd(
        q=query,
        k=key,
        v=value,
        o=expected,
        kv_cache_dtype="auto",
        k_cache=key_cache,
        v_cache=value_cache,
        b_loc=block_table,
        b_start_loc=query_start_loc,
        b_seq_len=seq_lens,
        max_seq_len=4,
        max_input_len=2,
        k_scale=torch.tensor([1.0]),
        v_scale=torch.tensor([1.0]),
        sliding_window=0,
        sm_scale=1.0,
        skip_decode=False,
    )

    assert torch.allclose(output, expected, atol=1e-5)


def test_chunked_prefill_decode_prefers_paged_attention_v1_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(chunked_prefill_mod, "HAS_TRITON", False)

    calls: dict[str, object] = {}

    def _fake_paged_attention_v1(**kwargs):
        calls["paged_attention_v1"] = kwargs["seq_lens"].clone()
        kwargs["out"].copy_(torch.full_like(kwargs["out"], 123.0))

    monkeypatch.setattr(chunked_prefill_mod.ops, "paged_attention_v1", _fake_paged_attention_v1)
    monkeypatch.setattr(
        chunked_prefill_mod,
        "context_attention_fwd",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("prefix fallback should not run for pure decode batch")
        ),
    )

    query = torch.zeros((1, 2, 32), dtype=torch.float16)
    output = torch.empty_like(query)
    key_cache = torch.zeros((1, 1, 32, 16, 1), dtype=torch.float16)
    value_cache = torch.zeros((1, 1, 32, 16), dtype=torch.float16)
    block_table = torch.tensor([[0]], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32)
    seq_lens = torch.tensor([1], dtype=torch.int32)

    chunked_prefill_mod.chunked_prefill_paged_decode(
        query=query,
        key=torch.zeros((1, 1, 32), dtype=torch.float16),
        value=torch.zeros((1, 1, 32), dtype=torch.float16),
        output=output,
        kv_cache_dtype="auto",
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        max_seq_len=1,
        max_query_len=1,
        k_scale=torch.tensor([1.0], dtype=torch.float32),
        v_scale=torch.tensor([1.0], dtype=torch.float32),
        sliding_window=0,
        sm_scale=1.0,
    )

    assert "paged_attention_v1" in calls
    assert torch.allclose(output, torch.full_like(output, 123.0))


def test_chunked_prefill_decode_without_triton_allows_missing_key_value_for_decode_fastpath(
    monkeypatch,
) -> None:
    monkeypatch.setattr(chunked_prefill_mod, "HAS_TRITON", False)

    calls: dict[str, object] = {}

    def _fake_paged_attention_v1(**kwargs):
        calls["seq_lens"] = kwargs["seq_lens"].clone()
        calls["num_kv_heads"] = kwargs["num_kv_heads"]
        kwargs["out"].copy_(torch.full_like(kwargs["out"], 55.0))

    monkeypatch.setattr(chunked_prefill_mod.ops, "paged_attention_v1", _fake_paged_attention_v1)
    monkeypatch.setattr(
        chunked_prefill_mod,
        "context_attention_fwd",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("prefix fallback should not run for pure decode batch")
        ),
    )

    query = torch.zeros((1, 2, 32), dtype=torch.float16)
    output = torch.empty_like(query)
    key_cache = torch.zeros((1, 1, 32, 16, 1), dtype=torch.float16)
    value_cache = torch.zeros((1, 1, 32, 16), dtype=torch.float16)
    block_table = torch.tensor([[0]], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32)
    seq_lens = torch.tensor([1], dtype=torch.int32)

    chunked_prefill_mod.chunked_prefill_paged_decode(
        query=query,
        key=None,
        value=None,
        output=output,
        kv_cache_dtype="auto",
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        max_seq_len=1,
        max_query_len=1,
        k_scale=torch.tensor([1.0], dtype=torch.float32),
        v_scale=torch.tensor([1.0], dtype=torch.float32),
        sliding_window=0,
        sm_scale=1.0,
    )

    assert torch.equal(calls["seq_lens"], seq_lens)
    assert calls["num_kv_heads"] == key_cache.shape[1]
    assert torch.allclose(output, torch.full_like(output, 55.0))


def test_chunked_prefill_mixed_batch_uses_decode_fastpath_and_prefix_fallback(
    monkeypatch,
) -> None:
    monkeypatch.setattr(chunked_prefill_mod, "HAS_TRITON", False)
    monkeypatch.setattr(prefix_prefill_mod, "HAS_TRITON", False)

    calls: dict[str, object] = {}

    def _fake_paged_attention_v1(**kwargs):
        calls["paged_attention_v1"] = kwargs["seq_lens"].clone()
        kwargs["out"].copy_(torch.full_like(kwargs["out"], 77.0))

    def _fake_prefix_context_attention_fwd(**kwargs):
        calls["prefix_skip_decode"] = kwargs["skip_decode"]
        kwargs["o"].fill_(5.0)

    monkeypatch.setattr(chunked_prefill_mod.ops, "paged_attention_v1", _fake_paged_attention_v1)
    monkeypatch.setattr(prefix_prefill_mod, "context_attention_fwd", _fake_prefix_context_attention_fwd)
    monkeypatch.setattr(chunked_prefill_mod, "context_attention_fwd", _fake_prefix_context_attention_fwd)

    query = torch.zeros((3, 2, 32), dtype=torch.float16)
    output = torch.empty_like(query)
    key_cache = torch.zeros((1, 1, 32, 16, 1), dtype=torch.float16)
    value_cache = torch.zeros((1, 1, 32, 16), dtype=torch.float16)
    block_table = torch.tensor([[0], [0]], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 2, 3], dtype=torch.int32)
    seq_lens = torch.tensor([4, 1], dtype=torch.int32)

    chunked_prefill_mod.chunked_prefill_paged_decode(
        query=query,
        key=torch.zeros((3, 1, 32), dtype=torch.float16),
        value=torch.zeros((3, 1, 32), dtype=torch.float16),
        output=output,
        kv_cache_dtype="auto",
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        max_seq_len=4,
        max_query_len=2,
        k_scale=torch.tensor([1.0], dtype=torch.float32),
        v_scale=torch.tensor([1.0], dtype=torch.float32),
        sliding_window=0,
        sm_scale=1.0,
    )

    assert calls["prefix_skip_decode"] is True
    assert torch.equal(calls["paged_attention_v1"], torch.tensor([1], dtype=torch.int32))
    assert torch.allclose(output[:2], torch.full_like(output[:2], 5.0))
    assert torch.allclose(output[2:], torch.full_like(output[2:], 77.0))


def test_chunked_prefill_without_triton_still_requires_key_value_for_prefix_queries(
    monkeypatch,
) -> None:
    monkeypatch.setattr(chunked_prefill_mod, "HAS_TRITON", False)

    query = torch.zeros((2, 2, 32), dtype=torch.float16)
    output = torch.empty_like(query)
    key_cache = torch.zeros((1, 1, 32, 16, 1), dtype=torch.float16)
    value_cache = torch.zeros((1, 1, 32, 16), dtype=torch.float16)
    block_table = torch.tensor([[0]], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 2], dtype=torch.int32)
    seq_lens = torch.tensor([2], dtype=torch.int32)

    with pytest.raises(
        NotImplementedError,
        match="requires materialized key/value tensors for prefix-prefill handling",
    ):
        chunked_prefill_mod.chunked_prefill_paged_decode(
            query=query,
            key=None,
            value=None,
            output=output,
            kv_cache_dtype="auto",
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            max_seq_len=2,
            max_query_len=2,
            k_scale=torch.tensor([1.0], dtype=torch.float32),
            v_scale=torch.tensor([1.0], dtype=torch.float32),
            sliding_window=0,
            sm_scale=1.0,
        )


def test_cuda_backend_selection_keeps_triton_without_runtime_when_reference_fallback_available(
    monkeypatch,
) -> None:
    monkeypatch.setattr(triton_attn, "HAS_TRITON", False)
    monkeypatch.setattr(triton_attn.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        cuda_platform,
        "_get_backend_priorities",
        lambda use_mla, device_capability, num_heads: [
            AttentionBackendEnum.TRITON_ATTN
        ],
    )

    valid_backends, invalid_reasons = CudaPlatformBase.get_valid_backends(
        device_capability=DeviceCapability(9, 0),
        attn_selector_config=AttentionSelectorConfig(
            head_size=64,
            dtype=torch.float16,
            kv_cache_dtype="auto",
            block_size=16,
            use_mla=False,
            has_sink=False,
            use_sparse=False,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            attn_type=AttentionType.DECODER,
        ),
        num_heads=32,
    )

    assert valid_backends == [(AttentionBackendEnum.TRITON_ATTN, 0)]
    assert invalid_reasons == {}


def test_cuda_backend_selection_keeps_triton_with_flash_attn_runtime_compat(
    monkeypatch,
) -> None:
    monkeypatch.setattr(triton_attn, "HAS_TRITON", False)
    monkeypatch.setattr(triton_attn.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(triton_attn, "is_flash_attn_varlen_func_available", lambda: True)
    monkeypatch.setattr(triton_attn, "get_flash_attn_version", lambda **_: 2)
    monkeypatch.setattr(triton_attn, "flash_attn_supports_sinks", lambda: False)
    monkeypatch.setattr(
        cuda_platform,
        "_get_backend_priorities",
        lambda use_mla, device_capability, num_heads: [
            AttentionBackendEnum.TRITON_ATTN
        ],
    )

    valid_backends, invalid_reasons = CudaPlatformBase.get_valid_backends(
        device_capability=DeviceCapability(9, 0),
        attn_selector_config=AttentionSelectorConfig(
            head_size=64,
            dtype=torch.float16,
            kv_cache_dtype="auto",
            block_size=16,
            use_mla=False,
            has_sink=False,
            use_sparse=False,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            attn_type=AttentionType.DECODER,
        ),
        num_heads=32,
    )

    assert valid_backends == [(AttentionBackendEnum.TRITON_ATTN, 0)]
    assert invalid_reasons == {}


def _reference_flash_attn_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    seqlen_offsets: int | torch.Tensor = 0,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: int | None = None,
    interleaved: bool = False,
    conjugate: bool = False,
) -> torch.Tensor:
    is_varlen = cu_seqlens is not None
    rotary_half_dim = cos.shape[1]
    rotary_dim = rotary_half_dim * 2
    output = x.clone()

    if not is_varlen:
        batch, seqlen, _, _ = x.shape
        base_positions = torch.arange(seqlen, device=x.device, dtype=torch.long)
        if isinstance(seqlen_offsets, torch.Tensor):
            positions = base_positions.unsqueeze(0) + seqlen_offsets.to(
                device=x.device,
                dtype=torch.long,
            ).unsqueeze(1)
        else:
            positions = base_positions.unsqueeze(0) + int(seqlen_offsets)
    else:
        assert cu_seqlens is not None
        assert max_seqlen is not None
        total_seqlen, _, _ = x.shape
        batch = cu_seqlens.shape[0] - 1
        positions = torch.empty((total_seqlen,), device=x.device, dtype=torch.long)
        cu_seqlens_cpu = cu_seqlens.to(device="cpu", dtype=torch.long)
        if isinstance(seqlen_offsets, torch.Tensor):
            offsets_cpu = seqlen_offsets.to(device="cpu", dtype=torch.long)
        else:
            offsets_cpu = None
        for batch_idx in range(batch):
            start = int(cu_seqlens_cpu[batch_idx].item())
            end = int(cu_seqlens_cpu[batch_idx + 1].item())
            seq_offset = (
                int(offsets_cpu[batch_idx].item())
                if offsets_cpu is not None
                else int(seqlen_offsets)
            )
            positions[start:end] = torch.arange(
                end - start,
                device=x.device,
                dtype=torch.long,
            ) + seq_offset

    cos_pos = cos.to(device=x.device).index_select(0, positions.reshape(-1))
    sin_pos = sin.to(device=x.device).index_select(0, positions.reshape(-1))
    if conjugate:
        sin_pos = -sin_pos

    if not is_varlen:
        cos_pos = cos_pos.view(batch, seqlen, 1, rotary_half_dim)
        sin_pos = sin_pos.view(batch, seqlen, 1, rotary_half_dim)
        x_rot = x[..., :rotary_dim].to(torch.float32)
    else:
        cos_pos = cos_pos.view(x.shape[0], 1, rotary_half_dim)
        sin_pos = sin_pos.view(x.shape[0], 1, rotary_half_dim)
        x_rot = x[..., :rotary_dim].to(torch.float32)

    if interleaved:
        cos_full = torch.repeat_interleave(cos_pos, 2, dim=-1).to(torch.float32)
        sin_full = torch.repeat_interleave(sin_pos, 2, dim=-1).to(torch.float32)
        x_pairs = x_rot.reshape(*x_rot.shape[:-1], -1, 2)
        rotated_half = torch.stack((-x_pairs[..., 1], x_pairs[..., 0]), dim=-1)
        rotated_half = rotated_half.reshape_as(x_rot)
    else:
        cos_full = torch.cat((cos_pos, cos_pos), dim=-1).to(torch.float32)
        sin_full = torch.cat((sin_pos, sin_pos), dim=-1).to(torch.float32)
        half = x_rot.shape[-1] // 2
        rotated_half = torch.cat((-x_rot[..., half:], x_rot[..., :half]), dim=-1)

    output[..., :rotary_dim] = (x_rot * cos_full + rotated_half * sin_full).to(x.dtype)
    return output


def test_cfie_flash_attn_rotary_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(cfie_flash_attn_rotary_mod, "HAS_TRITON", False)

    def _raise_if_triton_path_runs(*_args, **_kwargs):
        raise AssertionError("cfie flash-attn Triton rotary path should not run")

    monkeypatch.setattr(
        cfie_flash_attn_rotary_mod,
        "rotary_kernel",
        _raise_if_triton_path_runs,
    )

    x = torch.tensor(
        [
            [[[1.0, 2.0, 3.0, 4.0, 9.0, 10.0]], [[5.0, 6.0, 7.0, 8.0, 11.0, 12.0]]],
            [[[2.0, 1.0, 4.0, 3.0, 13.0, 14.0]], [[6.0, 5.0, 8.0, 7.0, 15.0, 16.0]]],
        ],
        dtype=torch.float32,
    )
    cos = torch.tensor(
        [
            [1.0, 0.5],
            [0.75, 1.25],
            [1.5, -0.5],
            [0.25, 2.0],
        ],
        dtype=torch.float32,
    )
    sin = torch.tensor(
        [
            [0.0, 0.25],
            [0.5, -0.75],
            [1.0, 0.5],
            [-0.5, 1.0],
        ],
        dtype=torch.float32,
    )
    seqlen_offsets = torch.tensor([0, 1], dtype=torch.int32)

    actual = cfie_flash_attn_rotary_mod.apply_rotary(
        x.clone(),
        cos,
        sin,
        seqlen_offsets=seqlen_offsets,
        interleaved=False,
        inplace=False,
    )
    expected = _reference_flash_attn_rotary(
        x,
        cos,
        sin,
        seqlen_offsets=seqlen_offsets,
        interleaved=False,
    )

    torch.testing.assert_close(actual, expected)


def test_vllm_flash_attn_rotary_varlen_falls_back_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(vllm_flash_attn_rotary_mod, "HAS_TRITON", False)

    def _raise_if_triton_path_runs(*_args, **_kwargs):
        raise AssertionError("vllm flash-attn Triton rotary path should not run")

    monkeypatch.setattr(
        vllm_flash_attn_rotary_mod,
        "rotary_kernel",
        _raise_if_triton_path_runs,
    )

    x = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0]],
            [[5.0, 6.0, 7.0, 8.0]],
            [[2.0, 4.0, 6.0, 8.0]],
            [[1.0, 3.0, 5.0, 7.0]],
            [[0.5, 1.5, 2.5, 3.5]],
        ],
        dtype=torch.float32,
    )
    cos = torch.tensor(
        [
            [1.0, 0.5],
            [0.75, 1.25],
            [1.5, -0.5],
            [0.25, 2.0],
            [0.5, 1.5],
        ],
        dtype=torch.float32,
    )
    sin = torch.tensor(
        [
            [0.0, 0.25],
            [0.5, -0.75],
            [1.0, 0.5],
            [-0.5, 1.0],
            [0.25, -0.25],
        ],
        dtype=torch.float32,
    )
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)
    seqlen_offsets = torch.tensor([1, 0], dtype=torch.int32)

    actual = vllm_flash_attn_rotary_mod.apply_rotary(
        x.clone(),
        cos,
        sin,
        seqlen_offsets=seqlen_offsets,
        cu_seqlens=cu_seqlens,
        max_seqlen=3,
        interleaved=True,
        inplace=False,
        conjugate=True,
    )
    expected = _reference_flash_attn_rotary(
        x,
        cos,
        sin,
        seqlen_offsets=seqlen_offsets,
        cu_seqlens=cu_seqlens,
        max_seqlen=3,
        interleaved=True,
        conjugate=True,
    )

    torch.testing.assert_close(actual, expected)


def test_rotary_cuda_path_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(rotary_common, "_cfie_flash_attn_apply_rotary_emb", None)
    monkeypatch.setattr(rotary_common, "_cfie_flash_attn_rotary_unavailable", False)
    monkeypatch.setattr(rotary_common, "find_spec", lambda _: None)

    rotary = object.__new__(rotary_common.ApplyRotaryEmb)
    torch.nn.Module.__init__(rotary)
    rotary.is_neox_style = True
    rotary.enable_fp32_compute = False
    x = torch.randn(2, 3, 4, 8)
    cos = torch.randn(3, 4)
    sin = torch.randn(3, 4)

    expected = rotary.forward_native(x.clone(), cos, sin)
    actual = rotary.forward_cuda(x.clone(), cos, sin)

    assert torch.allclose(actual, expected)


def test_rotary_cuda_path_uses_precompiled_op_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(
        rotary_common,
        "_try_precompiled_apply_rotary_emb",
        lambda **_: torch.full((2, 3, 4, 8), 6.0),
    )

    rotary = object.__new__(rotary_common.ApplyRotaryEmb)
    torch.nn.Module.__init__(rotary)
    rotary.is_neox_style = True
    rotary.enable_fp32_compute = False

    actual = rotary.forward_cuda(
        torch.randn(2, 3, 4, 8),
        torch.randn(3, 4),
        torch.randn(3, 4),
    )

    assert torch.equal(actual, torch.full((2, 3, 4, 8), 6.0))


def test_mrope_cuda_path_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(mrope, "HAS_TRITON", False)

    rotary = object.__new__(mrope.MRotaryEmbedding)
    torch.nn.Module.__init__(rotary)
    rotary.head_size = 8
    rotary.rotary_dim = 8
    rotary.is_neox_style = True
    rotary.mrope_section = [1, 1, 2]
    rotary.mrope_interleaved = False
    rotary.cos_sin_cache = torch.randn(16, 8)
    rotary._match_cos_sin_cache_dtype = lambda query: rotary.cos_sin_cache.to(query.dtype)

    apply_rotary = object.__new__(rotary_common.ApplyRotaryEmb)
    torch.nn.Module.__init__(apply_rotary)
    apply_rotary.is_neox_style = True
    apply_rotary.enable_fp32_compute = False
    rotary.apply_rotary_emb = apply_rotary

    positions = torch.tensor(
        [[0, 1, 2], [0, 1, 2], [0, 1, 2]],
        dtype=torch.long,
    )
    query = torch.randn(3, 16)
    key = torch.randn(3, 8)

    expected_q, expected_k = rotary.forward_native(positions, query.clone(), key.clone())
    actual_q, actual_k = rotary.forward_cuda(positions, query.clone(), key.clone())

    assert actual_k is not None
    assert torch.allclose(actual_q, expected_q)
    assert torch.allclose(actual_k, expected_k)


def test_mrope_cuda_path_uses_precompiled_op_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(mrope, "HAS_TRITON", False)

    rotary = object.__new__(mrope.MRotaryEmbedding)
    torch.nn.Module.__init__(rotary)
    rotary.head_size = 8
    rotary.rotary_dim = 8
    rotary.is_neox_style = True
    rotary.mrope_section = [1, 1, 2]
    rotary.mrope_interleaved = False
    rotary.cos_sin_cache = torch.randn(16, 8)
    rotary._match_cos_sin_cache_dtype = lambda query: rotary.cos_sin_cache.to(query.dtype)
    rotary.apply_rotary_emb = object()

    positions = torch.tensor(
        [[0, 1, 2], [0, 1, 2], [0, 1, 2]],
        dtype=torch.long,
    )
    query = torch.randn(3, 16)
    key = torch.randn(3, 8)
    expected_q = torch.full_like(query, 1.25)
    expected_k = torch.full_like(key, -2.5)

    monkeypatch.setattr(
        mrope,
        "_try_precompiled_mrope",
        lambda **_: (expected_q, expected_k),
    )

    actual_q, actual_k = rotary.forward_cuda(positions, query, key)

    assert torch.equal(actual_q, expected_q)
    assert torch.equal(actual_k, expected_k)


def test_inductor_backend_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(compilation_config.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(compilation_config, "find_spec", lambda _: None)

    cfg = compilation_config.CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        backend="inductor",
    )

    assert cfg.backend == "eager"


def test_eager_compile_backend_skips_torch_compile(monkeypatch) -> None:
    monkeypatch.setattr(compilation_wrapper.envs, "VLLM_USE_BYTECODE_HOOK", False)
    monkeypatch.setattr(
        compilation_wrapper,
        "get_current_cfie_config",
        lambda: SimpleNamespace(
            compilation_config=SimpleNamespace(
                mode=CompilationMode.VLLM_COMPILE,
                backend="eager",
                init_backend=lambda _: object(),
                dynamic_shapes_config=SimpleNamespace(
                    evaluate_guards=False,
                    type=compilation_config.DynamicShapesType.BACKED,
                ),
                inductor_compile_config={},
            ),
            observability_config=SimpleNamespace(
                enable_layerwise_nvtx_tracing=False,
            ),
        ),
    )

    def _torch_compile_should_not_run(*args, **kwargs):
        raise AssertionError("torch.compile should not be called for eager backend")

    monkeypatch.setattr(compilation_wrapper.torch, "compile", _torch_compile_should_not_run)

    class Dummy(compilation_wrapper.TorchCompileWithNoGuardsWrapper):
        def forward(self, x):
            return x + 1

    dummy = Dummy()

    assert dummy(torch.tensor(1)).item() == 2


def test_set_triton_allocator_is_noop_without_allocator_api(monkeypatch) -> None:
    monkeypatch.setattr(triton_allocation, "HAS_TRITON", True)
    monkeypatch.setattr(triton_allocation, "triton", SimpleNamespace())

    triton_allocation.set_triton_allocator(torch.device("cpu"))


def test_triton_placeholder_exposes_basic_runtime_helpers() -> None:
    placeholder = triton_importing.TritonPlaceholder()

    placeholder.set_allocator(lambda *args, **kwargs: None)

    assert placeholder.cdiv(10, 4) == 3
    assert placeholder.next_power_of_2(70) == 128
    assert placeholder.reinterpret(torch.tensor([1.0])).shape == (1,)


def test_fla_op_resolve_helper_falls_back_to_runtime_placeholder() -> None:
    helper = fla_op._resolve_triton_helper(None, None, name="exp")

    assert callable(helper)
    with pytest.raises(RuntimeError, match="exp requires a Triton runtime"):
        helper(torch.tensor([1.0]))


def test_fla_op_resolve_helper_prefers_first_callable() -> None:
    first = lambda x: ("first", x)
    second = lambda x: ("second", x)

    resolved = fla_op._resolve_triton_helper(first, second, name="exp")

    assert resolved is first
    assert callable(fla_op.exp)
    assert callable(fla_op.log)
    assert callable(fla_op.log2)
    assert callable(fla_op.gather)
    assert callable(fla_op.make_tensor_descriptor)


def test_mamba_fast_exp_reference_matches_torch_exp() -> None:
    x = torch.tensor([-2.0, 0.0, 1.5], dtype=torch.float32)

    actual = mamba_triton_helpers._fast_exp_reference(x)

    torch.testing.assert_close(actual, torch.exp(x))


def test_mamba_fast_exp_helper_stays_callable_without_triton() -> None:
    assert callable(mamba_triton_helpers.fast_exp)

    if not triton_importing.HAS_TRITON:
        x = torch.tensor([-1.0, 0.0, 2.0], dtype=torch.float32)
        torch.testing.assert_close(mamba_triton_helpers.fast_exp(x), torch.exp(x))


def test_mamba_ssd_state_passing_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(mamba_ssd_state_passing, "HAS_TRITON", False)

    states = torch.tensor(
        [
            [[1.0, 10.0]],
            [[2.0, 20.0]],
            [[3.0, 30.0]],
        ],
        dtype=torch.float16,
    )
    dA_cumsum = torch.zeros((1, 3, 2), dtype=torch.float16)
    dA_cumsum[0, 0, 1] = torch.log(torch.tensor(2.0)).to(torch.float16)
    dA_cumsum[0, 1, 1] = torch.log(torch.tensor(3.0)).to(torch.float16)
    dA_cumsum[0, 2, 1] = torch.log(torch.tensor(4.0)).to(torch.float16)
    last_chunk_indices = torch.tensor([1, 2], dtype=torch.long)
    initial_states = torch.tensor(
        [
            [[0.5, 1.0]],
            [[1.0, 1.0]],
        ],
        dtype=torch.float16,
    )

    actual = mamba_ssd_state_passing._state_passing_fwd(
        states=states,
        dA_cumsum=dA_cumsum,
        last_chunk_indices=last_chunk_indices,
        initial_states=initial_states,
        out_dtype=torch.float32,
    )

    expected = torch.tensor(
        [
            [[2.0, 12.0]],
            [[8.0, 56.0]],
            [[7.0, 34.0]],
        ],
        dtype=torch.float32,
    )

    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


def test_mamba_ssd_state_passing_without_initial_states_resets_per_sequence(
    monkeypatch,
) -> None:
    monkeypatch.setattr(mamba_ssd_state_passing, "HAS_TRITON", False)

    states = torch.tensor(
        [
            [[1.0, 2.0]],
            [[3.0, 4.0]],
            [[5.0, 6.0]],
        ],
        dtype=torch.float16,
    )
    dA_cumsum = torch.zeros((1, 3, 2), dtype=torch.float16)
    dA_cumsum[0, 0, 1] = torch.log(torch.tensor(2.0)).to(torch.float16)
    dA_cumsum[0, 1, 1] = torch.log(torch.tensor(3.0)).to(torch.float16)
    dA_cumsum[0, 2, 1] = torch.log(torch.tensor(4.0)).to(torch.float16)
    last_chunk_indices = torch.tensor([0, 2], dtype=torch.long)

    actual = mamba_ssd_state_passing._state_passing_fwd(
        states=states,
        dA_cumsum=dA_cumsum,
        last_chunk_indices=last_chunk_indices,
    )

    expected = torch.tensor(
        [
            [[1.0, 2.0]],
            [[3.0, 4.0]],
            [[17.0, 22.0]],
        ],
        dtype=torch.float16,
    )

    assert actual.dtype == torch.float16
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


def test_mamba_ssd_bmm_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(mamba_ssd_bmm, "HAS_TRITON", False)

    a = torch.tensor(
        [
            [[1.0, 2.0]],
            [[3.0, 4.0]],
            [[5.0, 6.0]],
            [[7.0, 8.0]],
            [[9.0, 10.0]],
        ],
        dtype=torch.float16,
    )
    b = torch.tensor(
        [
            [[0.0, 1.0]],
            [[1.0, 0.0]],
            [[1.0, 1.0]],
            [[2.0, 0.0]],
            [[0.0, 2.0]],
        ],
        dtype=torch.float16,
    )
    cu_chunk_seqlens = torch.tensor([0, 3, 5], dtype=torch.long)

    actual = mamba_ssd_bmm._bmm_chunk_fwd(
        a=a,
        b=b,
        chunk_size=3,
        cu_chunk_seqlens=cu_chunk_seqlens,
        output_dtype=torch.float32,
    )

    expected = torch.zeros((2, 1, 3, 3), dtype=torch.float32)
    expected[0, 0, :3, :3] = torch.tensor(
        [
            [2.0, 1.0, 3.0],
            [4.0, 3.0, 7.0],
            [6.0, 5.0, 11.0],
        ],
        dtype=torch.float32,
    )
    expected[1, 0, :2, :2] = torch.tensor(
        [
            [14.0, 16.0],
            [18.0, 20.0],
        ],
        dtype=torch.float32,
    )

    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


def test_mamba_ssd_bmm_causal_preserves_default_dtype_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(mamba_ssd_bmm, "HAS_TRITON", False)

    a = torch.tensor(
        [
            [[1.0, 2.0]],
            [[3.0, 4.0]],
            [[5.0, 6.0]],
            [[7.0, 8.0]],
        ],
        dtype=torch.float16,
    )
    b = torch.tensor(
        [
            [[2.0, 0.0]],
            [[1.0, 1.0]],
            [[0.0, 2.0]],
            [[3.0, 1.0]],
        ],
        dtype=torch.float16,
    )
    cu_chunk_seqlens = torch.tensor([0, 3, 4], dtype=torch.long)

    actual = mamba_ssd_bmm._bmm_chunk_fwd(
        a=a,
        b=b,
        chunk_size=3,
        cu_chunk_seqlens=cu_chunk_seqlens,
        causal=True,
    )

    expected = torch.zeros((2, 1, 3, 3), dtype=torch.float16)
    chunk0 = torch.matmul(
        a[:3, 0].to(torch.float32),
        b[:3, 0].to(torch.float32).transpose(0, 1),
    )
    expected[0, 0, :3, :3] = torch.triu(chunk0).to(torch.float16)
    chunk1 = torch.matmul(
        a[3:, 0].to(torch.float32),
        b[3:, 0].to(torch.float32).transpose(0, 1),
    )
    expected[1, 0, :1, :1] = torch.triu(chunk1).to(torch.float16)

    assert actual.dtype == torch.float16
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


def test_mamba_ssd_chunk_scan_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(mamba_ssd_chunk_scan, "HAS_TRITON", False)

    cb = torch.tensor(
        [
            [
                [
                    [1.0, 0.5],
                    [0.25, 2.0],
                ]
            ],
            [
                [
                    [2.0, 0.0],
                    [0.0, 0.0],
                ]
            ],
        ],
        dtype=torch.float32,
    )
    x = torch.tensor(
        [
            [[1.0, 2.0]],
            [[3.0, 4.0]],
            [[5.0, 6.0]],
        ],
        dtype=torch.float32,
    )
    dt = torch.ones((1, 2, 2), dtype=torch.float32)
    dA_cumsum = torch.zeros((1, 2, 2), dtype=torch.float32)
    C = torch.tensor(
        [
            [[1.0, 2.0]],
            [[3.0, 4.0]],
            [[5.0, 6.0]],
        ],
        dtype=torch.float32,
    )
    states = torch.zeros((2, 1, 2, 2), dtype=torch.float32)
    states[0, 0] = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    cu_chunk_seqlens = torch.tensor([0, 2, 3], dtype=torch.long)
    out = torch.empty_like(x)
    seq_idx = torch.tensor([0, 0], dtype=torch.long)
    D = torch.tensor([[0.1, 0.2]], dtype=torch.float32)
    initial_states = torch.tensor(
        [
            [
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ]
            ]
        ],
        dtype=torch.float32,
    )

    actual = mamba_ssd_chunk_scan._chunk_scan_fwd(
        cb=cb,
        x=x,
        dt=dt,
        dA_cumsum=dA_cumsum,
        C=C,
        states=states,
        cu_chunk_seqlens=cu_chunk_seqlens,
        out=out,
        seq_idx=seq_idx,
        D=D,
        initial_states=initial_states,
    )

    expected = torch.tensor(
        [
            [[2.1, 4.4]],
            [[9.55, 13.3]],
            [[27.5, 52.2]],
        ],
        dtype=torch.float32,
    )

    assert actual is None
    torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)


def test_mamba_ssd_chunk_scan_without_optional_terms_uses_prior_state(
    monkeypatch,
) -> None:
    monkeypatch.setattr(mamba_ssd_chunk_scan, "HAS_TRITON", False)

    cb = torch.zeros((2, 1, 1, 1), dtype=torch.float32)
    x = torch.tensor(
        [
            [[1.0, 2.0]],
            [[3.0, 4.0]],
        ],
        dtype=torch.float32,
    )
    dt = torch.tensor([[[0.5], [0.25]]], dtype=torch.float32)
    dA_cumsum = torch.tensor([[[0.0], [0.2]]], dtype=torch.float32)
    C = torch.tensor(
        [
            [[1.0, 0.0]],
            [[0.5, 1.0]],
        ],
        dtype=torch.float32,
    )
    states = torch.zeros((2, 1, 2, 2), dtype=torch.float32)
    states[0, 0] = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    cu_chunk_seqlens = torch.tensor([0, 1, 2], dtype=torch.long)
    out = torch.empty_like(x)
    seq_idx = torch.tensor([0, 0], dtype=torch.long)

    actual = mamba_ssd_chunk_scan._chunk_scan_fwd(
        cb=cb,
        x=x,
        dt=dt,
        dA_cumsum=dA_cumsum,
        C=C,
        states=states,
        cu_chunk_seqlens=cu_chunk_seqlens,
        out=out,
        seq_idx=seq_idx,
    )

    expected = torch.zeros_like(out)
    previous_state = states[0, 0]
    prev_term = torch.matmul(C[1, 0], previous_state.transpose(0, 1))
    expected[1, 0] = prev_term * torch.exp(dA_cumsum[0, 1, 0])

    assert actual is None
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_mamba_ssd_chunk_scan_with_z_vector_d_and_sequence_reset(
    monkeypatch,
) -> None:
    monkeypatch.setattr(mamba_ssd_chunk_scan, "HAS_TRITON", False)

    cb = torch.tensor(
        [
            [[[2.0]]],
            [[[1.5]]],
            [[[0.5]]],
        ],
        dtype=torch.float32,
    )
    x = torch.tensor(
        [
            [[1.0, 2.0]],
            [[2.0, 1.0]],
            [[3.0, 4.0]],
        ],
        dtype=torch.float32,
    )
    dt = torch.tensor([[[0.5], [0.25], [0.75]]], dtype=torch.float32)
    dA_cumsum = torch.tensor([[[0.0], [0.2], [0.1]]], dtype=torch.float32)
    C = torch.tensor(
        [
            [[1.0, 0.5]],
            [[0.25, 1.0]],
            [[2.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    states = torch.zeros((3, 1, 2, 2), dtype=torch.float32)
    states[0, 0] = torch.tensor([[1.0, 2.0], [0.5, 1.5]], dtype=torch.float32)
    cu_chunk_seqlens = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    out = torch.empty_like(x)
    seq_idx = torch.tensor([0, 0, 1], dtype=torch.long)
    D = torch.tensor([0.1], dtype=torch.float32)
    z = torch.tensor(
        [
            [[1.0, 0.0]],
            [[-1.0, 2.0]],
            [[0.5, -0.5]],
        ],
        dtype=torch.float32,
    )

    actual = mamba_ssd_chunk_scan._chunk_scan_fwd(
        cb=cb,
        x=x,
        dt=dt,
        dA_cumsum=dA_cumsum,
        C=C,
        states=states,
        cu_chunk_seqlens=cu_chunk_seqlens,
        out=out,
        seq_idx=seq_idx,
        D=D,
        z=z,
    )

    expected = torch.zeros_like(out)
    for chunk_idx in range(3):
        chunk_start = int(cu_chunk_seqlens[chunk_idx].item())
        current_seq_idx = int(seq_idx[chunk_idx].item())
        previous_seq_idx = int(seq_idx[chunk_idx - 1].item()) if chunk_idx > 0 else -1
        if chunk_idx == 0 or current_seq_idx != previous_seq_idx:
            previous_state = torch.zeros((2, 2), dtype=torch.float32)
        else:
            previous_state = states[chunk_idx - 1, 0].to(torch.float32)

        C_chunk = C[chunk_start, 0].to(torch.float32)
        dA_chunk = dA_cumsum[0, chunk_idx, 0]
        acc = torch.matmul(C_chunk, previous_state.transpose(0, 1))
        acc *= torch.exp(dA_chunk)
        acc += cb[chunk_idx, 0, 0, 0] * dt[0, chunk_idx, 0] * x[chunk_start, 0]
        acc += x[chunk_start, 0] * D[0]
        z_chunk = z[chunk_start, 0]
        acc *= z_chunk * torch.sigmoid(z_chunk)
        expected[chunk_start, 0] = acc

    assert actual is None
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_mamba_ssd_chunk_state_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(mamba_ssd_chunk_state, "HAS_TRITON", False)

    dt = torch.tensor(
        [
            [0.1],
            [0.2],
            [0.3],
        ],
        dtype=torch.float32,
    )
    A = torch.tensor([2.0], dtype=torch.float32)
    cu_chunk_seqlens = torch.tensor([0, 2, 3], dtype=torch.long)

    dA_cumsum, dt_out = mamba_ssd_chunk_state._chunk_cumsum_fwd(
        dt=dt,
        A=A,
        chunk_size=2,
        cu_chunk_seqlens=cu_chunk_seqlens,
    )

    expected_dt_out = torch.tensor(
        [
            [
                [0.1, 0.2],
                [0.3, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    expected_dA_cumsum = torch.tensor(
        [
            [
                [0.2, 0.6],
                [0.6, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(dt_out, expected_dt_out, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        dA_cumsum,
        expected_dA_cumsum,
        atol=1e-6,
        rtol=1e-6,
    )

    x = torch.tensor(
        [
            [[1.0, 2.0]],
            [[3.0, 4.0]],
            [[5.0, 6.0]],
        ],
        dtype=torch.float32,
    )
    B = torch.tensor(
        [
            [[1.0, 2.0]],
            [[3.0, 4.0]],
            [[5.0, 6.0]],
        ],
        dtype=torch.float32,
    )

    actual_states = mamba_ssd_chunk_state._chunk_state_fwd(
        B=B,
        x=x,
        dt=dt_out,
        dA_cumsum=dA_cumsum,
        cu_chunk_seqlens=cu_chunk_seqlens,
        states_in_fp32=True,
    )

    chunk0_scale = torch.tensor(
        [torch.exp(torch.tensor(0.4)) * 0.1, 0.2],
        dtype=torch.float32,
    )
    expected_chunk0 = x[:2, 0].transpose(0, 1) @ (
        B[:2, 0] * chunk0_scale.unsqueeze(-1)
    )
    expected_chunk1 = x[2:, 0].transpose(0, 1) @ (B[2:, 0] * 0.3)
    expected_states = torch.stack(
        [expected_chunk0, expected_chunk1],
        dim=0,
    ).unsqueeze(1)

    torch.testing.assert_close(
        actual_states,
        expected_states,
        atol=1e-5,
        rtol=1e-5,
    )


def test_mamba_ssd_chunk_cumsum_with_bias_softplus_limit_and_empty_chunk(
    monkeypatch,
) -> None:
    monkeypatch.setattr(mamba_ssd_chunk_state, "HAS_TRITON", False)

    dt = torch.tensor(
        [
            [-1.0, 0.0],
            [0.5, 25.0],
            [10.0, -5.0],
        ],
        dtype=torch.float32,
    )
    A = torch.tensor([2.0, -1.5], dtype=torch.float32)
    dt_bias = torch.tensor([0.25, -0.25], dtype=torch.float32)
    cu_chunk_seqlens = torch.tensor([0, 2, 2, 3], dtype=torch.long)

    dA_cumsum, dt_out = mamba_ssd_chunk_state._chunk_cumsum_fwd(
        dt=dt,
        A=A,
        chunk_size=2,
        cu_chunk_seqlens=cu_chunk_seqlens,
        dt_bias=dt_bias,
        dt_softplus=True,
        dt_limit=(0.2, 0.8),
    )

    expected_dt_out = torch.zeros((2, 3, 2), dtype=torch.float32)
    expected_dA_cumsum = torch.zeros_like(expected_dt_out)
    for chunk_idx in range(cu_chunk_seqlens.numel() - 1):
        chunk_start = int(cu_chunk_seqlens[chunk_idx].item())
        chunk_end = int(cu_chunk_seqlens[chunk_idx + 1].item())
        chunk_len = chunk_end - chunk_start
        if chunk_len <= 0:
            continue
        dt_chunk = dt[chunk_start:chunk_end].transpose(0, 1) + dt_bias.unsqueeze(-1)
        dt_chunk = torch.where(
            dt_chunk <= 20.0,
            torch.nn.functional.softplus(dt_chunk),
            dt_chunk,
        )
        dt_chunk = torch.clamp(dt_chunk, min=0.2, max=0.8)
        expected_dt_out[:, chunk_idx, :chunk_len] = dt_chunk
        expected_dA_cumsum[:, chunk_idx, :chunk_len] = torch.cumsum(
            dt_chunk * A.unsqueeze(-1),
            dim=-1,
        )

    torch.testing.assert_close(dt_out, expected_dt_out, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        dA_cumsum,
        expected_dA_cumsum,
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(dt_out[:, 1], torch.zeros((2, 2), dtype=torch.float32))
    torch.testing.assert_close(
        dA_cumsum[:, 1],
        torch.zeros((2, 2), dtype=torch.float32),
    )


def test_mamba_ssd_chunk_state_handles_preallocated_buffer_and_empty_chunk(
    monkeypatch,
) -> None:
    monkeypatch.setattr(mamba_ssd_chunk_state, "HAS_TRITON", False)

    x = torch.tensor(
        [
            [[1.0, 2.0], [2.0, 1.0]],
            [[3.0, 4.0], [0.5, 1.5]],
            [[5.0, 6.0], [1.0, 2.0]],
        ],
        dtype=torch.float16,
    )
    B = torch.tensor(
        [
            [[1.0, 0.5]],
            [[2.0, 1.0]],
            [[3.0, 1.5]],
        ],
        dtype=torch.float16,
    )
    dt = torch.tensor(
        [
            [
                [0.1, 0.2],
                [0.0, 0.0],
                [0.4, 0.0],
            ],
            [
                [0.05, 0.15],
                [0.0, 0.0],
                [0.3, 0.0],
            ],
        ],
        dtype=torch.float32,
    )
    dA_cumsum = torch.tensor(
        [
            [
                [0.1, 0.3],
                [0.0, 0.0],
                [0.4, 0.0],
            ],
            [
                [0.05, 0.2],
                [0.0, 0.0],
                [0.3, 0.0],
            ],
        ],
        dtype=torch.float32,
    )
    cu_chunk_seqlens = torch.tensor([0, 2, 2, 3], dtype=torch.long)
    states = torch.full((3, 2, 2, 2), -1.0, dtype=torch.float16)

    actual_states = mamba_ssd_chunk_state._chunk_state_fwd(
        B=B,
        x=x,
        dt=dt,
        dA_cumsum=dA_cumsum,
        cu_chunk_seqlens=cu_chunk_seqlens,
        states=states,
    )

    expected_states = torch.zeros((3, 2, 2, 2), dtype=torch.float16)
    for chunk_idx, (chunk_start, chunk_end) in enumerate(
        zip(cu_chunk_seqlens[:-1], cu_chunk_seqlens[1:])
    ):
        chunk_start_int = int(chunk_start.item())
        chunk_end_int = int(chunk_end.item())
        chunk_len = chunk_end_int - chunk_start_int
        if chunk_len <= 0:
            continue
        B_chunk = B[chunk_start_int:chunk_end_int, 0].to(torch.float32)
        for head_idx in range(2):
            x_chunk = x[chunk_start_int:chunk_end_int, head_idx].to(torch.float32)
            dA_chunk = dA_cumsum[head_idx, chunk_idx, :chunk_len]
            dt_chunk = dt[head_idx, chunk_idx, :chunk_len]
            scale = torch.exp(dA_chunk[-1] - dA_chunk) * dt_chunk
            chunk_state = x_chunk.transpose(0, 1) @ (B_chunk * scale.unsqueeze(-1))
            expected_states[chunk_idx, head_idx] = chunk_state.to(torch.float16)

    assert actual_states.data_ptr() == states.data_ptr()
    assert actual_states.dtype == torch.float16
    torch.testing.assert_close(actual_states, expected_states, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(
        actual_states[1],
        torch.zeros((2, 2, 2), dtype=torch.float16),
    )

    allocated_states = mamba_ssd_chunk_state._chunk_state_fwd(
        B=B,
        x=x,
        dt=dt,
        dA_cumsum=dA_cumsum,
        cu_chunk_seqlens=cu_chunk_seqlens,
        states=None,
        states_in_fp32=False,
    )

    assert allocated_states.dtype == torch.float16


def test_mamba_do_copy_block_falls_back_without_triton_for_overlapping_copy(
    monkeypatch,
) -> None:
    monkeypatch.setattr(worker_mamba_utils, "HAS_TRITON", False)

    def _raise_if_triton_path_runs(*_args, **_kwargs):
        raise AssertionError("Mamba Triton batch_memcpy path should not run")

    monkeypatch.setattr(worker_mamba_utils, "batch_memcpy", _raise_if_triton_path_runs)

    def _make_cpu_gpu_buffer(n: int, dtype: torch.dtype):
        return worker_mamba_utils.CpuGpuBuffer(
            n,
            dtype=dtype,
            device=torch.device("cpu"),
            pin_memory=False,
        )

    copy_bufs = worker_mamba_utils.MambaCopyBuffers(
        src_ptrs=_make_cpu_gpu_buffer(1, torch.int64),
        dst_ptrs=_make_cpu_gpu_buffer(1, torch.int64),
        sizes=_make_cpu_gpu_buffer(1, torch.int32),
        offset=1,
        python_copies=[],
    )

    state = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    copy_bufs.python_copies.append((state[0, 1:], state[0], 3))

    worker_mamba_utils.do_mamba_copy_block(copy_bufs)

    torch.testing.assert_close(
        state,
        torch.tensor([[2.0, 3.0, 4.0, 4.0]], dtype=torch.float32),
    )
    assert copy_bufs.python_copies == []


def test_preprocess_mamba_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(worker_mamba_utils, "HAS_TRITON", False)

    def _raise_if_triton_path_runs(*_args, **_kwargs):
        raise AssertionError("Mamba Triton batch_memcpy path should not run")

    monkeypatch.setattr(worker_mamba_utils, "batch_memcpy", _raise_if_triton_path_runs)

    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                kv_cache_spec=MambaSpec(
                    block_size=4,
                    shapes=((4,), (2,)),
                    dtypes=(torch.float32, torch.float32),
                    mamba_cache_mode="align",
                ),
                layer_names=["layer0"],
            )
        ]
    )

    def _make_cpu_gpu_buffer(n: int, dtype: torch.dtype):
        return worker_mamba_utils.CpuGpuBuffer(
            n,
            dtype=dtype,
            device=torch.device("cpu"),
            pin_memory=False,
        )

    copy_bufs = worker_mamba_utils.MambaCopyBuffers.create(
        max_num_reqs=1,
        kv_cache_config=kv_cache_config,
        copy_funcs=(
            model_mamba_utils.get_conv_copy_spec,
            model_mamba_utils.get_temporal_copy_spec,
        ),
        make_buffer=_make_cpu_gpu_buffer,
    )

    conv_state = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [9.0, 9.0, 9.0, 9.0],
        ],
        dtype=torch.float32,
    )
    temporal_state = torch.tensor(
        [
            [10.0, 20.0],
            [30.0, 40.0],
        ],
        dtype=torch.float32,
    )
    forward_context = {
        "layer0": SimpleNamespace(
            kv_cache=([conv_state, temporal_state],),
        )
    }
    scheduler_output = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=None,
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
        num_scheduled_tokens={"req0": 1},
    )
    cache_config = SimpleNamespace(enable_prefix_caching=True)
    mamba_state_idx: dict[str, int] = {}
    input_batch = SimpleNamespace(
        req_ids=["req0"],
        num_accepted_tokens_cpu=np.array([1], dtype=np.int32),
    )
    requests = {
        "req0": SimpleNamespace(
            block_ids=([0, 1],),
            num_computed_tokens=4,
        )
    }

    worker_mamba_utils.preprocess_mamba(
        scheduler_output=scheduler_output,
        kv_cache_config=kv_cache_config,
        cache_config=cache_config,
        mamba_state_idx=mamba_state_idx,
        input_batch=input_batch,
        requests=requests,
        forward_context=forward_context,
        mamba_state_copy_funcs=(
            model_mamba_utils.get_conv_copy_spec,
            model_mamba_utils.get_temporal_copy_spec,
        ),
        copy_bufs=copy_bufs,
    )

    torch.testing.assert_close(
        conv_state,
        torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
            ],
            dtype=torch.float32,
        ),
    )
    torch.testing.assert_close(
        temporal_state,
        torch.tensor(
            [
                [10.0, 20.0],
                [10.0, 20.0],
            ],
            dtype=torch.float32,
        ),
    )
    assert mamba_state_idx == {"req0": 1}
    assert int(input_batch.num_accepted_tokens_cpu[0]) == 1


def test_fla_layernorm_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(fla_layernorm_guard, "HAS_TRITON", False)

    x = torch.randn(2, 8, dtype=torch.float16)
    weight = torch.randn(8, dtype=torch.float16)
    bias = torch.randn(8, dtype=torch.float16)
    z = torch.randn(2, 8, dtype=torch.float16)

    expected = fla_layernorm_guard.layer_norm_ref(
        x,
        weight,
        bias,
        z=z,
        eps=1e-5,
        norm_before_gate=False,
        activation="silu",
    )
    actual = fla_layernorm_guard.layernorm_fn(
        x,
        weight,
        bias,
        z=z,
        eps=1e-5,
        norm_before_gate=False,
        activation="silu",
    )

    assert torch.allclose(actual, expected, atol=1e-3, rtol=1e-3)


def test_fla_layernorm_uses_precompiled_op_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(fla_layernorm_guard, "HAS_TRITON", False)
    monkeypatch.setattr(
        fla_layernorm_guard,
        "_try_precompiled_layer_norm",
        lambda **_: torch.full((2, 8), 3.0, dtype=torch.float16),
    )

    x = torch.randn(2, 8, dtype=torch.float16)
    weight = torch.randn(8, dtype=torch.float16)
    bias = torch.randn(8, dtype=torch.float16)

    actual = fla_layernorm_guard.layernorm_fn(
        x,
        weight,
        bias,
        eps=1e-5,
        activation="silu",
    )

    assert torch.equal(actual, torch.full((2, 8), 3.0, dtype=torch.float16))


def test_fla_rmsnorm_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(fla_layernorm_guard, "HAS_TRITON", False)

    x = torch.randn(2, 8, dtype=torch.float16)
    weight = torch.randn(8, dtype=torch.float16)
    z = torch.randn(2, 8, dtype=torch.float16)

    expected = fla_layernorm_guard.rms_norm_ref(
        x,
        weight,
        None,
        z=z,
        eps=1e-5,
        norm_before_gate=True,
        activation="sigmoid",
    )
    actual = fla_layernorm_guard.rmsnorm_fn(
        x,
        weight,
        None,
        z=z,
        eps=1e-5,
        norm_before_gate=True,
        activation="sigmoid",
    )

    assert torch.allclose(actual, expected, atol=1e-3, rtol=1e-3)


def test_fused_sigmoid_gating_delta_rule_update_falls_back_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fused_sigmoid_gating_mod, "HAS_TRITON", False)

    A_log = torch.tensor([0.1], dtype=torch.float32)
    a = torch.tensor([[[0.2], [0.4]]], dtype=torch.float32)
    b = torch.tensor([[[0.3], [0.5]]], dtype=torch.float32)
    dt_bias = torch.tensor([0.05], dtype=torch.float32)
    q = torch.tensor([[[[1.0, 0.0]], [[0.5, 0.5]]]], dtype=torch.float32)
    k = torch.tensor([[[[0.2, 0.8]], [[0.6, 0.4]]]], dtype=torch.float32)
    v = torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]], dtype=torch.float32)
    scale = k.shape[-1] ** -0.5
    initial_state = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    ref_final_state = torch.empty((2, 1, 2, 2), dtype=torch.float32)
    ref_out = torch.empty((1, 2, 1, 2), dtype=torch.float32)

    expected_out, expected_state = (
        fused_sigmoid_gating_mod._fused_sigmoid_gating_delta_rule_update_ref(
            A_log=A_log,
            a=a,
            b=b,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            beta=1.0,
            threshold=20.0,
            scale=scale,
            initial_state=initial_state,
            final_state=ref_final_state,
            inplace_final_state=False,
            cu_seqlens=None,
            ssm_state_indices=None,
            num_accepted_tokens=None,
            use_qk_l2norm_in_kernel=False,
            is_kda=False,
            out=ref_out,
        )
    )

    actual_out, actual_state = (
        fused_sigmoid_gating_mod.fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            a=a,
            b=b,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            scale=scale,
            initial_state=initial_state,
            inplace_final_state=False,
        )
    )

    torch.testing.assert_close(actual_out, expected_out)
    torch.testing.assert_close(actual_state, expected_state)


def test_fused_sigmoid_gating_delta_rule_update_uses_precompiled_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fused_sigmoid_gating_mod, "HAS_TRITON", False)
    monkeypatch.setattr(
        fused_sigmoid_gating_mod.ops,
        "has_precompiled_fused_sigmoid_gating_delta_rule_update",
        lambda: True,
    )

    expected_out = torch.full((1, 2, 1, 2), 3.0, dtype=torch.float32)
    expected_state = torch.full((1, 1, 2, 2), 4.0, dtype=torch.float32)
    monkeypatch.setattr(
        fused_sigmoid_gating_mod.ops,
        "fused_sigmoid_gating_delta_rule_update_precompiled",
        lambda **kwargs: (expected_out, expected_state),
    )

    actual_out, actual_state = (
        fused_sigmoid_gating_mod.fused_sigmoid_gating_delta_rule_update(
            A_log=torch.tensor([0.1], dtype=torch.float32),
            a=torch.tensor([[[0.2], [0.4]]], dtype=torch.float32),
            b=torch.tensor([[[0.3], [0.5]]], dtype=torch.float32),
            dt_bias=torch.tensor([0.05], dtype=torch.float32),
            q=_as_fake_cuda(
                torch.tensor([[[[1.0, 0.0]], [[0.5, 0.5]]]], dtype=torch.float32)
            ),
            k=torch.tensor([[[[0.2, 0.8]], [[0.6, 0.4]]]], dtype=torch.float32),
            v=torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]], dtype=torch.float32),
            initial_state=torch.zeros((1, 1, 2, 2), dtype=torch.float32),
            inplace_final_state=True,
        )
    )

    assert actual_out is expected_out
    assert actual_state is expected_state


def test_fla_l2norm_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(fla_l2norm, "HAS_TRITON", False)

    x = torch.randn(2, 3, 4, dtype=torch.bfloat16)
    expected = (x.float() * torch.rsqrt(x.float().square().sum(dim=-1, keepdim=True) + 1e-6)).to(
        torch.float16
    )
    actual = fla_l2norm.l2norm_fwd(x, output_dtype=torch.float16)

    assert actual.dtype == torch.float16
    assert torch.allclose(actual, expected, atol=2e-3, rtol=2e-3)


def test_fla_l2norm_prefers_precompiled_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(fla_l2norm, "HAS_TRITON", False)

    expected = torch.full((2, 3, 4), 1.25, dtype=torch.float16)
    monkeypatch.setattr(
        fla_l2norm,
        "_try_precompiled_l2norm",
        lambda x, eps, output_dtype: expected,
    )

    actual = fla_l2norm.l2norm_fwd(
        torch.randn(2, 3, 4, dtype=torch.bfloat16),
        output_dtype=torch.float16,
    )

    assert torch.equal(actual, expected)


def test_fla_chunk_local_cumsum_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(fla_cumsum, "HAS_TRITON", False)
    monkeypatch.setattr(
        fla_cumsum,
        "_try_precompiled_chunk_local_cumsum",
        lambda *args, **kwargs: None,
    )

    g = torch.tensor(
        [
            [
                [[1.0, 10.0], [2.0, 20.0]],
                [[3.0, 30.0], [4.0, 40.0]],
                [[5.0, 50.0], [6.0, 60.0]],
                [[7.0, 70.0], [8.0, 80.0]],
                [[9.0, 90.0], [10.0, 100.0]],
            ]
        ],
        dtype=torch.float16,
    )
    cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.long)

    actual = fla_cumsum.chunk_local_cumsum(
        g,
        chunk_size=2,
        reverse=True,
        cu_seqlens=cu_seqlens,
        output_dtype=torch.float32,
    )

    expected = torch.tensor(
        [
            [
                [[4.0, 40.0], [6.0, 60.0]],
                [[3.0, 30.0], [4.0, 40.0]],
                [[5.0, 50.0], [6.0, 60.0]],
                [[16.0, 160.0], [18.0, 180.0]],
                [[9.0, 90.0], [10.0, 100.0]],
            ]
        ],
        dtype=torch.float32,
    )

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)


def test_fla_chunk_local_cumsum_prefers_precompiled_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fla_cumsum, "HAS_TRITON", False)

    expected = torch.full((1, 5, 2, 2), 7.0, dtype=torch.float16)
    monkeypatch.setattr(
        fla_cumsum,
        "_try_precompiled_chunk_local_cumsum",
        lambda *args, **kwargs: expected,
    )
    monkeypatch.setattr(
        fla_cumsum,
        "_chunk_local_cumsum_reference",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("reference fallback should not run")
        ),
    )

    actual = fla_cumsum.chunk_local_cumsum(
        torch.randn(1, 5, 2, 2, dtype=torch.float16),
        chunk_size=2,
    )

    assert torch.equal(actual, expected)


def test_fla_chunk_o_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(fla_chunk_o, "HAS_TRITON", False)
    monkeypatch.setattr(
        fla_chunk_o,
        "_try_precompiled_chunk_fwd_o",
        lambda *args, **kwargs: None,
    )

    q = torch.tensor(
        [[[[1.0, 0.0]], [[0.0, 1.0]]]],
        dtype=torch.float16,
    )
    k = torch.tensor(
        [[[[2.0, 0.0]], [[0.0, 3.0]]]],
        dtype=torch.float16,
    )
    v = torch.tensor(
        [
            [
                [[10.0, 20.0], [1.0, 2.0]],
                [[30.0, 40.0], [3.0, 4.0]],
            ]
        ],
        dtype=torch.float16,
    )
    h = torch.tensor(
        [
            [
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                ]
            ]
        ],
        dtype=torch.float16,
    )

    actual = fla_chunk_o.chunk_fwd_o(
        q=q,
        k=k,
        v=v,
        h=h,
        scale=1.0,
        chunk_size=2,
    )

    expected = torch.tensor(
        [
            [
                [[21.0, 43.0], [7.0, 11.0]],
                [[92.0, 124.0], [15.0, 20.0]],
            ]
        ],
        dtype=torch.float16,
    )

    torch.testing.assert_close(actual, expected)


def test_fla_chunk_o_prefers_precompiled_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(fla_chunk_o, "HAS_TRITON", False)
    expected = torch.full((1, 2, 2, 2), 6.0, dtype=torch.float16)
    monkeypatch.setattr(
        fla_chunk_o,
        "_try_precompiled_chunk_fwd_o",
        lambda *args, **kwargs: expected,
    )
    monkeypatch.setattr(
        fla_chunk_o,
        "_chunk_fwd_o_reference",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("reference fallback should not run")
        ),
    )

    actual = fla_chunk_o.chunk_fwd_o(
        q=torch.randn(1, 2, 1, 2, dtype=torch.float16),
        k=torch.randn(1, 2, 1, 2, dtype=torch.float16),
        v=torch.randn(1, 2, 2, 2, dtype=torch.float16),
        h=torch.randn(1, 1, 2, 2, 2, dtype=torch.float16),
        chunk_size=2,
    )

    assert torch.equal(actual, expected)


def test_fla_chunk_scaled_dot_kkt_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(fla_chunk_scaled_dot_kkt, "HAS_TRITON", False)
    monkeypatch.setattr(
        fla_chunk_scaled_dot_kkt,
        "_try_precompiled_chunk_scaled_dot_kkt_fwd",
        lambda *args, **kwargs: None,
    )

    k = torch.tensor(
        [[[[1.0, 0.0]], [[1.0, 2.0]]]],
        dtype=torch.float16,
    )
    beta = torch.tensor([[[2.0], [3.0]]], dtype=torch.float16)

    actual = fla_chunk_scaled_dot_kkt.chunk_scaled_dot_kkt_fwd(
        k=k,
        beta=beta,
        chunk_size=2,
        output_dtype=torch.float32,
    )

    expected = torch.tensor(
        [[[[0.0, 0.0]], [[3.0, 0.0]]]],
        dtype=torch.float32,
    )

    torch.testing.assert_close(actual, expected)


def test_fla_chunk_scaled_dot_kkt_prefers_precompiled_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fla_chunk_scaled_dot_kkt, "HAS_TRITON", False)
    expected = torch.full((1, 2, 1, 2), 4.0, dtype=torch.float32)
    monkeypatch.setattr(
        fla_chunk_scaled_dot_kkt,
        "_try_precompiled_chunk_scaled_dot_kkt_fwd",
        lambda *args, **kwargs: expected,
    )
    monkeypatch.setattr(
        fla_chunk_scaled_dot_kkt,
        "_chunk_scaled_dot_kkt_fwd_reference",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("reference fallback should not run")
        ),
    )

    actual = fla_chunk_scaled_dot_kkt.chunk_scaled_dot_kkt_fwd(
        k=torch.randn(1, 2, 1, 2, dtype=torch.float16),
        beta=torch.randn(1, 2, 1, dtype=torch.float16),
        chunk_size=2,
        output_dtype=torch.float32,
    )

    assert torch.equal(actual, expected)


def test_fla_chunk_delta_h_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(fla_chunk_delta_h, "HAS_TRITON", False)
    monkeypatch.setattr(
        fla_chunk_delta_h,
        "_try_precompiled_chunk_gated_delta_rule_fwd_h",
        lambda *args, **kwargs: None,
    )

    k = torch.tensor(
        [[[[1.0, 0.0]], [[0.0, 1.0]]]],
        dtype=torch.float16,
    )
    w = torch.zeros((1, 2, 1, 2), dtype=torch.float16)
    u = torch.tensor(
        [[[[10.0, 20.0]], [[30.0, 40.0]]]],
        dtype=torch.float16,
    )

    h, v_new, final_state = fla_chunk_delta_h.chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        initial_state=torch.zeros((1, 1, 2, 2), dtype=torch.float32),
        output_final_state=True,
        chunk_size=2,
        save_new_value=True,
    )

    expected_h = torch.zeros((1, 1, 1, 2, 2), dtype=torch.float16)
    expected_v_new = u
    expected_final_state = torch.tensor(
        [[[[10.0, 30.0], [20.0, 40.0]]]],
        dtype=torch.float32,
    )

    torch.testing.assert_close(h, expected_h)
    assert v_new is not None
    torch.testing.assert_close(v_new, expected_v_new)
    assert final_state is not None
    torch.testing.assert_close(final_state, expected_final_state)


def test_fla_chunk_delta_h_prefers_precompiled_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(fla_chunk_delta_h, "HAS_TRITON", False)
    expected_h = torch.full((1, 1, 1, 2, 2), 3.0, dtype=torch.float16)
    expected_v_new = torch.full((1, 2, 1, 2), 4.0, dtype=torch.float16)
    expected_final_state = torch.full((1, 1, 2, 2), 5.0, dtype=torch.float32)
    monkeypatch.setattr(
        fla_chunk_delta_h,
        "_try_precompiled_chunk_gated_delta_rule_fwd_h",
        lambda *args, **kwargs: (expected_h, expected_v_new, expected_final_state),
    )
    monkeypatch.setattr(
        fla_chunk_delta_h,
        "_chunk_gated_delta_rule_fwd_h_reference",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("reference fallback should not run")
        ),
    )

    h, v_new, final_state = fla_chunk_delta_h.chunk_gated_delta_rule_fwd_h(
        k=torch.randn(1, 2, 1, 2, dtype=torch.float16),
        w=torch.randn(1, 2, 1, 2, dtype=torch.float16),
        u=torch.randn(1, 2, 1, 2, dtype=torch.float16),
        initial_state=torch.zeros((1, 1, 2, 2), dtype=torch.float32),
        output_final_state=True,
        chunk_size=2,
    )

    assert torch.equal(h, expected_h)
    assert v_new is not None and torch.equal(v_new, expected_v_new)
    assert final_state is not None and torch.equal(final_state, expected_final_state)


def test_fla_solve_tril_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(fla_solve_tril, "HAS_TRITON", False)
    monkeypatch.setattr(
        fla_solve_tril,
        "_try_precompiled_solve_tril",
        lambda *args, **kwargs: None,
    )

    A = torch.zeros((1, 16, 1, 16), dtype=torch.float16)
    A[0, 1, 0, 0] = 2.0
    A[0, 2, 0, 0] = 1.0
    A[0, 2, 0, 1] = 3.0

    actual = fla_solve_tril.solve_tril(
        A=A,
        output_dtype=torch.float32,
    )

    expected = torch.zeros((1, 16, 1, 16), dtype=torch.float32)
    expected[0, :, 0, :] = torch.eye(16, dtype=torch.float32)
    expected[0, 1, 0, 0] = -2.0
    expected[0, 2, 0, 0] = 5.0
    expected[0, 2, 0, 1] = -3.0

    torch.testing.assert_close(actual, expected)


def test_fla_solve_tril_prefers_precompiled_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(fla_solve_tril, "HAS_TRITON", False)
    expected = torch.full((1, 16, 1, 16), 7.0, dtype=torch.float32)
    monkeypatch.setattr(
        fla_solve_tril,
        "_try_precompiled_solve_tril",
        lambda *args, **kwargs: expected,
    )
    monkeypatch.setattr(
        fla_solve_tril,
        "_solve_tril_reference",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("reference fallback should not run")
        ),
    )

    actual = fla_solve_tril.solve_tril(
        A=torch.zeros((1, 16, 1, 16), dtype=torch.float16),
        output_dtype=torch.float32,
    )

    assert torch.equal(actual, expected)


def test_fla_recompute_w_u_fwd_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(fla_wy_fast, "HAS_TRITON", False)
    monkeypatch.setattr(
        fla_wy_fast,
        "_try_precompiled_recompute_w_u_fwd",
        lambda *args, **kwargs: None,
    )

    k = torch.tensor(
        [[[[1.0, 2.0]], [[3.0, 4.0]]]],
        dtype=torch.float16,
    )
    v = torch.tensor(
        [[[[10.0, 20.0]], [[30.0, 40.0]]]],
        dtype=torch.float16,
    )
    beta = torch.tensor([[[2.0], [3.0]]], dtype=torch.float16)
    g_cumsum = torch.zeros((1, 2, 1), dtype=torch.float16)
    A = torch.tensor(
        [[[[1.0, 0.0]], [[5.0, 6.0]]]],
        dtype=torch.float16,
    )

    w, u = fla_wy_fast.recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        g_cumsum=g_cumsum,
        A=A,
        cu_seqlens=None,
    )

    expected_w = torch.tensor(
        [[[[2.0, 4.0]], [[64.0, 92.0]]]],
        dtype=torch.float16,
    )
    expected_u = torch.tensor(
        [[[[20.0, 40.0]], [[640.0, 920.0]]]],
        dtype=torch.float16,
    )

    torch.testing.assert_close(w, expected_w)
    torch.testing.assert_close(u, expected_u)


def test_fla_recompute_w_u_fwd_prefers_precompiled_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fla_wy_fast, "HAS_TRITON", False)
    expected_w = torch.full((1, 2, 1, 2), 8.0, dtype=torch.float16)
    expected_u = torch.full((1, 2, 1, 2), 9.0, dtype=torch.float16)
    monkeypatch.setattr(
        fla_wy_fast,
        "_try_precompiled_recompute_w_u_fwd",
        lambda *args, **kwargs: (expected_w, expected_u),
    )
    monkeypatch.setattr(
        fla_wy_fast,
        "_recompute_w_u_fwd_reference",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("reference fallback should not run")
        ),
    )

    w, u = fla_wy_fast.recompute_w_u_fwd(
        k=torch.randn(1, 2, 1, 2, dtype=torch.float16),
        v=torch.randn(1, 2, 1, 2, dtype=torch.float16),
        beta=torch.randn(1, 2, 1, dtype=torch.float16),
        g_cumsum=torch.randn(1, 2, 1, dtype=torch.float16),
        A=torch.randn(1, 2, 1, 2, dtype=torch.float16),
        cu_seqlens=None,
    )

    assert torch.equal(w, expected_w)
    assert torch.equal(u, expected_u)


def test_fla_chunk_gated_delta_rule_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(fla_chunk, "HAS_TRITON", False)

    q = torch.zeros(1, 4, 2, 3, dtype=torch.float16)
    k = torch.zeros(1, 4, 2, 3, dtype=torch.float16)
    v = torch.randn(1, 4, 2, 5, dtype=torch.float16)
    g = torch.zeros(1, 4, 2, dtype=torch.float16)
    beta = torch.ones(1, 4, 2, dtype=torch.float16)
    initial_state = torch.randn(1, 2, 5, 3, dtype=torch.float32)
    cu_seqlens = torch.tensor([0, 4], dtype=torch.long)

    output, final_state = fla_chunk.chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )

    assert torch.count_nonzero(output) == 0
    assert final_state is not None
    assert torch.allclose(final_state, initial_state)


def test_fla_chunk_gated_delta_rule_uses_precompiled_op_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fla_chunk, "HAS_TRITON", False)
    expected_output = torch.full((1, 4, 2, 5), 1.5, dtype=torch.float16)
    expected_final_state = torch.full((1, 2, 5, 3), 2.5, dtype=torch.float32)
    monkeypatch.setattr(
        fla_chunk,
        "_try_precompiled_chunk_gated_delta_rule",
        lambda **_: (expected_output, expected_final_state),
    )

    output, final_state = fla_chunk.chunk_gated_delta_rule(
        q=torch.zeros(1, 4, 2, 3, dtype=torch.float16),
        k=torch.zeros(1, 4, 2, 3, dtype=torch.float16),
        v=torch.randn(1, 4, 2, 5, dtype=torch.float16),
        g=torch.zeros(1, 4, 2, dtype=torch.float16),
        beta=torch.ones(1, 4, 2, dtype=torch.float16),
        initial_state=torch.randn(1, 2, 5, 3, dtype=torch.float32),
        output_final_state=True,
        cu_seqlens=torch.tensor([0, 4], dtype=torch.long),
    )

    assert torch.equal(output, expected_output)
    assert torch.equal(final_state, expected_final_state)


def test_causal_conv1d_fn_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(causal_conv1d, "HAS_TRITON", False)

    x = torch.tensor(
        [[3.0, 4.0, 5.0]],
        dtype=torch.float32,
    )
    weight = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
    bias = torch.tensor([0.5], dtype=torch.float32)
    conv_states = torch.tensor([[[1.0, 2.0]]], dtype=torch.float32)
    query_start_loc = torch.tensor([0, 3], dtype=torch.int32)
    cache_indices = torch.tensor([0], dtype=torch.int32)
    has_initial_state = torch.tensor([True], dtype=torch.bool)

    expected = torch.tensor([[1.9, 2.5, 3.1]], dtype=torch.float32)
    actual = causal_conv1d.causal_conv1d_fn(
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        activation=None,
    )

    assert torch.allclose(actual, expected)
    assert torch.allclose(conv_states[0], torch.tensor([[4.0, 5.0]]))


def test_mamba_gated_rms_norm_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(mamba_layernorm_gated, "HAS_TRITON", False)

    def _raise_if_triton_path_runs(*_args, **_kwargs):
        raise AssertionError("Triton gated RMSNorm path should not run")

    monkeypatch.setattr(
        mamba_layernorm_gated,
        "_layer_norm_fwd",
        _raise_if_triton_path_runs,
    )

    x = torch.randn(2, 3, 4, dtype=torch.float32)
    z = torch.randn(2, 3, 4, dtype=torch.float32)
    weight = torch.randn(4, dtype=torch.float32)
    group_size = 2
    eps = 1e-6

    actual = mamba_layernorm_gated.rms_norm_gated(
        x,
        weight,
        bias=None,
        z=z,
        eps=eps,
        group_size=group_size,
        norm_before_gate=False,
    )

    gated_x = x * (z * torch.sigmoid(z))
    grouped = gated_x.reshape(-1, 4).view(-1, 2, group_size)
    expected = grouped * torch.rsqrt(grouped.pow(2).mean(dim=-1, keepdim=True) + eps)
    expected = expected.view(-1, 4) * weight.view(1, 4)
    expected = expected.reshape_as(x)

    assert torch.allclose(actual, expected, atol=1e-6)


def test_mamba_gated_rms_norm_norm_before_gate_with_bias_falls_back_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(mamba_layernorm_gated, "HAS_TRITON", False)

    def _raise_if_triton_path_runs(*_args, **_kwargs):
        raise AssertionError("Triton gated RMSNorm path should not run")

    monkeypatch.setattr(
        mamba_layernorm_gated,
        "_layer_norm_fwd",
        _raise_if_triton_path_runs,
    )

    x = torch.randn(2, 3, 4, dtype=torch.float32)
    z = torch.randn(2, 3, 4, dtype=torch.float32)
    weight = torch.randn(4, dtype=torch.float32)
    bias = torch.randn(4, dtype=torch.float32)
    group_size = 2
    eps = 1e-6

    actual = mamba_layernorm_gated.rms_norm_gated(
        x,
        weight,
        bias=bias,
        z=z,
        eps=eps,
        group_size=group_size,
        norm_before_gate=True,
    )

    x_grouped = x.reshape(-1, 4).view(-1, 2, group_size)
    variance = x_grouped.pow(2).mean(dim=-1, keepdim=True)
    expected = x_grouped * torch.rsqrt(variance + eps)
    expected = expected.view(-1, 4) * weight.view(1, 4) + bias.view(1, 4)
    expected = expected * (z.reshape(-1, 4) * torch.sigmoid(z.reshape(-1, 4)))
    expected = expected.reshape_as(x)

    assert torch.allclose(actual, expected, atol=1e-6)


def test_mamba_gated_rms_norm_without_z_uses_full_group_fallback_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(mamba_layernorm_gated, "HAS_TRITON", False)

    def _raise_if_triton_path_runs(*_args, **_kwargs):
        raise AssertionError("Triton gated RMSNorm path should not run")

    monkeypatch.setattr(
        mamba_layernorm_gated,
        "_layer_norm_fwd",
        _raise_if_triton_path_runs,
    )

    x = torch.randn(2, 3, 4, dtype=torch.float32)
    weight = torch.randn(4, dtype=torch.float32)
    bias = torch.randn(4, dtype=torch.float32)
    eps = 1e-6

    actual = mamba_layernorm_gated.rms_norm_gated(
        x,
        weight,
        bias=bias,
        z=None,
        eps=eps,
        group_size=None,
        norm_before_gate=True,
    )

    x_work = x.reshape(-1, 4)
    variance = x_work.pow(2).mean(dim=-1, keepdim=True)
    expected = x_work * torch.rsqrt(variance + eps)
    expected = expected * weight.view(1, 4) + bias.view(1, 4)
    expected = expected.reshape_as(x)

    assert torch.allclose(actual, expected, atol=1e-6)


def test_mamba_selective_state_update_falls_back_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(mamba_ssm, "HAS_TRITON", False)

    state = torch.zeros(1, 2, 2, dtype=torch.float32)
    x = torch.tensor([[2.0, 3.0]], dtype=torch.float32)
    dt = torch.tensor([[0.1, 0.2]], dtype=torch.float32)
    A = torch.zeros(2, 2, dtype=torch.float32)
    B = torch.tensor([[0.5, 1.0]], dtype=torch.float32)
    C = torch.ones(1, 2, dtype=torch.float32)
    out = torch.empty_like(x)

    mamba_ssm.selective_state_update(
        state,
        x,
        dt,
        A,
        B,
        C,
        out=out,
    )

    expected_state = torch.tensor(
        [
            [
                [0.1, 0.2],
                [0.3, 0.6],
            ]
        ],
        dtype=torch.float32,
    )
    expected_out = torch.tensor([[0.3, 0.9]], dtype=torch.float32)

    assert torch.allclose(state, expected_state)
    assert torch.allclose(out, expected_out)


def test_mamba_selective_state_update_varlen_with_bias_softplus_d_and_z_falls_back_without_triton(  # noqa: E501
    monkeypatch,
) -> None:
    monkeypatch.setattr(mamba_ssm, "HAS_TRITON", False)

    state = torch.tensor(
        [
            [[0.5, -0.5]],
            [[1.0, 0.25]],
        ],
        dtype=torch.float32,
    )
    x = torch.tensor([[2.0], [-1.0], [3.0]], dtype=torch.float32)
    dt = torch.tensor([[0.0], [0.5], [-0.25]], dtype=torch.float32)
    A = torch.tensor([[0.2, -0.3]], dtype=torch.float32)
    B = torch.tensor(
        [[0.5, 1.0], [1.5, -0.5], [0.25, 0.75]],
        dtype=torch.float32,
    )
    C = torch.tensor(
        [[1.0, 0.5], [-1.0, 0.25], [0.5, -0.5]],
        dtype=torch.float32,
    )
    D = torch.tensor([0.2], dtype=torch.float32)
    z = torch.tensor([[0.5], [-1.0], [1.5]], dtype=torch.float32)
    dt_bias = torch.tensor([0.1], dtype=torch.float32)
    cu_seqlens = torch.tensor([0, 2, 3], dtype=torch.int32)
    out = torch.zeros_like(x)
    initial_state = state.clone()

    mamba_ssm.selective_state_update(
        state,
        x,
        dt,
        A,
        B,
        C,
        D=D,
        z=z,
        dt_bias=dt_bias,
        dt_softplus=True,
        out=out,
        cu_seqlens=cu_seqlens,
    )

    expected_state = initial_state.clone()
    expected_out = torch.zeros_like(out)

    for seq_idx, (token_start, token_end) in enumerate(((0, 2), (2, 3))):
        state_work = expected_state[seq_idx, 0].to(torch.float32).clone()
        for token_idx in range(token_start, token_end):
            dt_eff = torch.nn.functional.softplus(
                dt[token_idx, 0].to(torch.float32) + dt_bias[0]
            )
            dA = torch.exp(A[0].to(torch.float32) * dt_eff)
            dB = B[token_idx].to(torch.float32) * dt_eff
            state_work = state_work * dA + dB * x[token_idx, 0].to(torch.float32)

            out_t = (state_work * C[token_idx].to(torch.float32)).sum()
            out_t = out_t + x[token_idx, 0].to(torch.float32) * D[0]
            z_t = z[token_idx, 0].to(torch.float32)
            out_t = out_t * (z_t * torch.sigmoid(z_t))
            expected_out[token_idx, 0] = out_t.to(expected_out.dtype)

        expected_state[seq_idx, 0] = state_work.to(expected_state.dtype)

    assert torch.allclose(state, expected_state, atol=1e-6)
    assert torch.allclose(out, expected_out, atol=1e-6)


def test_mamba_selective_state_update_with_state_indices_and_pad_slot_falls_back_without_triton(  # noqa: E501
    monkeypatch,
) -> None:
    monkeypatch.setattr(mamba_ssm, "HAS_TRITON", False)

    pad_slot_id = 99
    state = torch.tensor(
        [
            [[0.0, 0.0]],
            [[1.0, 2.0]],
            [[-3.0, 1.0]],
            [[0.5, -0.5]],
        ],
        dtype=torch.float32,
    )
    x = torch.tensor([[2.0], [4.0]], dtype=torch.float32)
    dt = torch.tensor([[0.5], [0.25]], dtype=torch.float32)
    A = torch.zeros((1, 2), dtype=torch.float32)
    B = torch.tensor([[1.0, 0.5], [0.25, 0.75]], dtype=torch.float32)
    C = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    state_batch_indices = torch.tensor(
        [[0, 1], [pad_slot_id, 2]],
        dtype=torch.int32,
    )
    dst_state_batch_indices = torch.tensor(
        [[3, 0], [pad_slot_id, 2]],
        dtype=torch.int32,
    )
    num_accepted_tokens = torch.tensor([2, 1], dtype=torch.int32)
    out = torch.zeros_like(x)

    expected_state = state.clone()
    state_work = expected_state[1, 0].to(torch.float32).clone()
    dt_eff = dt[0, 0].to(torch.float32)
    state_work = state_work + B[0].to(torch.float32) * dt_eff * x[0, 0].to(torch.float32)
    expected_state[3, 0] = state_work.to(expected_state.dtype)
    expected_out = torch.zeros_like(out)
    expected_out[0, 0] = (state_work * C[0].to(torch.float32)).sum().to(expected_out.dtype)

    mamba_ssm.selective_state_update(
        state,
        x,
        dt,
        A,
        B,
        C,
        out=out,
        state_batch_indices=state_batch_indices,
        dst_state_batch_indices=dst_state_batch_indices,
        pad_slot_id=pad_slot_id,
        num_accepted_tokens=num_accepted_tokens,
    )

    assert torch.allclose(state, expected_state, atol=1e-6)
    assert torch.allclose(out, expected_out, atol=1e-6)


def test_causal_conv1d_fn_uses_precompiled_op_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(causal_conv1d, "HAS_TRITON", False)

    expected = torch.tensor([[9.0, 8.0, 7.0]], dtype=torch.float32)
    monkeypatch.setattr(
        causal_conv1d,
        "_try_precompiled_causal_conv1d_fn",
        lambda **_: expected,
    )

    actual = causal_conv1d.causal_conv1d_fn(
        torch.tensor([[3.0, 4.0, 5.0]], dtype=torch.float32),
        torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32),
        torch.tensor([0.5], dtype=torch.float32),
        torch.tensor([[[1.0, 2.0]]], dtype=torch.float32),
        torch.tensor([0, 3], dtype=torch.int32),
        cache_indices=torch.tensor([0], dtype=torch.int32),
        has_initial_state=torch.tensor([True], dtype=torch.bool),
        activation=None,
    )

    assert torch.equal(actual, expected)


def test_causal_conv1d_fn_handles_pad_slot_and_activation_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(causal_conv1d, "HAS_TRITON", False)
    monkeypatch.setattr(
        causal_conv1d,
        "_try_precompiled_causal_conv1d_fn",
        lambda **_: None,
    )

    x = torch.tensor(
        [[1.0, 2.0, 3.0], [0.5, -1.0, 4.0]],
        dtype=torch.float32,
    )
    weight = torch.tensor(
        [[0.1, 0.2, 0.3], [0.4, -0.1, 0.2]],
        dtype=torch.float32,
    )
    bias = torch.tensor([0.5, -0.25], dtype=torch.float32)
    conv_states = torch.tensor(
        [
            [[9.0, 9.0], [9.0, 9.0]],
            [[1.0, 2.0], [0.5, -0.5]],
        ],
        dtype=torch.float32,
    )
    query_start_loc = torch.tensor([0, 2, 3], dtype=torch.int32)
    cache_indices = torch.tensor([causal_conv1d.PAD_SLOT_ID, 1], dtype=torch.int32)
    has_initial_state = torch.tensor([True, False], dtype=torch.bool)

    expected_states = conv_states.clone()
    expected = causal_conv1d._causal_conv1d_fn_ref(
        x,
        weight,
        bias,
        expected_states,
        query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        activation="silu",
    )

    actual = causal_conv1d.causal_conv1d_fn(
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        activation=True,
    )

    active_start = int(query_start_loc[1].item())
    active_end = int(query_start_loc[2].item())
    torch.testing.assert_close(
        actual[:, active_start:active_end],
        expected[:, active_start:active_end],
    )
    torch.testing.assert_close(conv_states, expected_states)


def test_causal_conv1d_update_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(causal_conv1d, "HAS_TRITON", False)

    x = torch.tensor([[3.0], [6.0]], dtype=torch.float32)
    conv_state = torch.tensor(
        [
            [[1.0, 2.0]],
            [[4.0, 5.0]],
        ],
        dtype=torch.float32,
    )
    weight = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
    bias = torch.tensor([0.5], dtype=torch.float32)
    conv_state_indices = torch.tensor([0, 1], dtype=torch.int32)

    actual = causal_conv1d.causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias,
        activation=None,
        conv_state_indices=conv_state_indices,
    )

    expected = torch.tensor([[1.9], [3.7]], dtype=torch.float32)
    assert torch.allclose(actual, expected)
    assert torch.allclose(conv_state[0], torch.tensor([[2.0, 3.0]]))
    assert torch.allclose(conv_state[1], torch.tensor([[5.0, 6.0]]))


def test_causal_conv1d_update_handles_apc_spec_varlen_and_pad_slot_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(causal_conv1d, "HAS_TRITON", False)
    monkeypatch.setattr(
        causal_conv1d,
        "_try_precompiled_causal_conv1d_update",
        lambda **_: None,
    )

    x = torch.tensor(
        [[1.0], [2.0], [3.0]],
        dtype=torch.float32,
    )
    conv_state = torch.tensor(
        [
            [[10.0, 11.0]],
            [[1.0, 2.0]],
            [[7.0, 8.0]],
        ],
        dtype=torch.float32,
    )
    weight = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
    bias = torch.tensor([0.5], dtype=torch.float32)
    conv_state_indices = torch.tensor(
        [
            [1, 0],
            [causal_conv1d.PAD_SLOT_ID, 2],
        ],
        dtype=torch.int32,
    )
    query_start_loc = torch.tensor([0, 2, 3], dtype=torch.int32)
    block_idx_last_scheduled_token = torch.tensor([1, 1], dtype=torch.int32)
    initial_state_idx = torch.tensor([0, 0], dtype=torch.int32)
    num_accepted_tokens = torch.tensor([2, 1], dtype=torch.int32)

    expected_state = conv_state.clone()
    expected = causal_conv1d._causal_conv1d_update_ref(
        x,
        expected_state,
        weight,
        bias,
        activation="silu",
        conv_state_indices=conv_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        query_start_loc=query_start_loc,
        pad_slot_id=causal_conv1d.PAD_SLOT_ID,
        block_idx_last_scheduled_token=block_idx_last_scheduled_token,
        initial_state_idx=initial_state_idx,
    )

    actual = causal_conv1d.causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias,
        activation=True,
        conv_state_indices=conv_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        query_start_loc=query_start_loc,
        max_query_len=2,
        pad_slot_id=causal_conv1d.PAD_SLOT_ID,
        block_idx_last_scheduled_token=block_idx_last_scheduled_token,
        initial_state_idx=initial_state_idx,
    )

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(conv_state, expected_state)


def test_causal_conv1d_update_uses_precompiled_op_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(causal_conv1d, "HAS_TRITON", False)

    expected = torch.tensor([[4.0], [3.0]], dtype=torch.float32)
    monkeypatch.setattr(
        causal_conv1d,
        "_try_precompiled_causal_conv1d_update",
        lambda **_: expected.unsqueeze(-1),
    )

    actual = causal_conv1d.causal_conv1d_update(
        torch.tensor([[3.0], [6.0]], dtype=torch.float32),
        torch.tensor(
            [
                [[1.0, 2.0]],
                [[4.0, 5.0]],
            ],
            dtype=torch.float32,
        ),
        torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32),
        torch.tensor([0.5], dtype=torch.float32),
        activation=None,
        conv_state_indices=torch.tensor([0, 1], dtype=torch.int32),
    )

    assert torch.equal(actual, expected)


def test_fused_recurrent_packed_decode_falls_back_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fused_recurrent, "HAS_TRITON", False)

    mixed_qkv = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=torch.float32)
    a = torch.zeros((1, 1), dtype=torch.float32)
    b = torch.zeros((1, 1), dtype=torch.float32)
    A_log = torch.zeros(1, dtype=torch.float32)
    dt_bias = torch.zeros(1, dtype=torch.float32)
    initial_state = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    out = torch.zeros((1, 1, 1, 2), dtype=torch.float32)
    ssm_state_indices = torch.tensor([0], dtype=torch.int32)

    actual_out, actual_state = fused_recurrent.fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=1.0,
        initial_state=initial_state,
        out=out,
        ssm_state_indices=ssm_state_indices,
    )

    expected_out = torch.tensor([[[[27.5, 33.0]]]], dtype=torch.float32)
    expected_state = torch.tensor([[[[7.5, 10.0], [9.0, 12.0]]]], dtype=torch.float32)

    assert torch.allclose(actual_out, expected_out)
    assert torch.allclose(actual_state, expected_state)


def test_fused_recurrent_packed_decode_prefers_precompiled_op_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fused_recurrent, "HAS_TRITON", False)
    expected_out = torch.full((1, 1, 1, 2), 8.0, dtype=torch.float32)
    expected_state = torch.full((1, 1, 2, 2), 9.0, dtype=torch.float32)
    monkeypatch.setattr(
        fused_recurrent,
        "_try_precompiled_fused_recurrent_gated_delta_rule_packed_decode",
        lambda **_: (expected_out, expected_state),
    )
    monkeypatch.setattr(
        fused_recurrent,
        "_fused_recurrent_gated_delta_rule_packed_decode_ref",
        lambda **_: (_ for _ in ()).throw(
            AssertionError("reference fallback should not run")
        ),
    )

    actual_out, actual_state = fused_recurrent.fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=torch.float32),
        a=torch.zeros((1, 1), dtype=torch.float32),
        b=torch.zeros((1, 1), dtype=torch.float32),
        A_log=torch.zeros(1, dtype=torch.float32),
        dt_bias=torch.zeros(1, dtype=torch.float32),
        scale=1.0,
        initial_state=torch.zeros((1, 1, 2, 2), dtype=torch.float32),
        out=torch.zeros((1, 1, 1, 2), dtype=torch.float32),
        ssm_state_indices=torch.tensor([0], dtype=torch.int32),
    )

    assert torch.equal(actual_out, expected_out)
    assert torch.equal(actual_state, expected_state)


def test_fused_recurrent_packed_decode_skips_invalid_state_updates(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fused_recurrent, "HAS_TRITON", False)
    monkeypatch.setattr(
        fused_recurrent,
        "_try_precompiled_fused_recurrent_gated_delta_rule_packed_decode",
        lambda **_: None,
    )

    mixed_qkv = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2.0, 1.0, 4.0, 3.0, 6.0, 5.0],
        ],
        dtype=torch.float32,
    )
    a = torch.zeros((2, 1), dtype=torch.float32)
    b = torch.zeros((2, 1), dtype=torch.float32)
    A_log = torch.zeros(1, dtype=torch.float32)
    dt_bias = torch.zeros(1, dtype=torch.float32)
    initial_state = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    out = torch.zeros((2, 1, 1, 2), dtype=torch.float32)
    ssm_state_indices = torch.tensor([0, -1], dtype=torch.int32)

    actual_out, actual_state = fused_recurrent.fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=1.0,
        initial_state=initial_state,
        out=out,
        ssm_state_indices=ssm_state_indices,
    )

    expected_valid_out = torch.tensor([[[[27.5, 33.0]]]], dtype=torch.float32)
    expected_valid_state = torch.tensor(
        [[[[7.5, 10.0], [9.0, 12.0]]]],
        dtype=torch.float32,
    )

    torch.testing.assert_close(actual_out[:1], expected_valid_out)
    torch.testing.assert_close(actual_out[1:], torch.zeros_like(actual_out[1:]))
    torch.testing.assert_close(actual_state, expected_valid_state)


def test_fused_recurrent_fwd_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(fused_recurrent, "HAS_TRITON", False)
    monkeypatch.setattr(
        fused_recurrent,
        "_try_precompiled_fused_recurrent_gated_delta_rule_fwd",
        lambda **_: None,
    )

    q = torch.tensor(
        [[[[1.0, 0.0]], [[0.0, 1.0]]]],
        dtype=torch.float16,
    )
    k = torch.tensor(
        [[[[1.0, 0.0]], [[0.0, 1.0]]]],
        dtype=torch.float16,
    )
    v = torch.tensor(
        [[[[10.0, 20.0]], [[30.0, 40.0]]]],
        dtype=torch.float16,
    )
    g = torch.zeros((1, 2, 1), dtype=torch.float16)
    beta = torch.ones((1, 2, 1), dtype=torch.float16)
    initial_state = torch.zeros((1, 1, 2, 2), dtype=torch.float32)

    output, final_state = fused_recurrent.fused_recurrent_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=1.0,
        initial_state=initial_state,
        inplace_final_state=False,
    )

    expected_output = torch.tensor(
        [[[[10.0, 20.0]], [[30.0, 40.0]]]],
        dtype=torch.float16,
    )
    expected_state = torch.tensor(
        [
            [[[10.0, 0.0], [20.0, 0.0]]],
            [[[10.0, 30.0], [20.0, 40.0]]],
        ],
        dtype=torch.float32,
    )

    torch.testing.assert_close(output, expected_output)
    torch.testing.assert_close(final_state, expected_state)


def test_fused_recurrent_fwd_prefers_precompiled_op_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fused_recurrent, "HAS_TRITON", False)
    expected_output = torch.full((1, 2, 1, 2), 3.0, dtype=torch.float16)
    expected_state = torch.full((1, 1, 2, 2), 4.0, dtype=torch.float32)
    monkeypatch.setattr(
        fused_recurrent,
        "_try_precompiled_fused_recurrent_gated_delta_rule_fwd",
        lambda **_: (expected_output, expected_state),
    )
    monkeypatch.setattr(
        fused_recurrent,
        "_fused_recurrent_gated_delta_rule_fwd_ref",
        lambda **_: (_ for _ in ()).throw(
            AssertionError("reference fallback should not run")
        ),
    )

    output, final_state = fused_recurrent.fused_recurrent_gated_delta_rule_fwd(
        q=torch.zeros((1, 2, 1, 2), dtype=torch.float16),
        k=torch.zeros((1, 2, 1, 2), dtype=torch.float16),
        v=torch.zeros((1, 2, 1, 2), dtype=torch.float16),
        g=torch.zeros((1, 2, 1), dtype=torch.float16),
        beta=torch.ones((1, 2, 1), dtype=torch.float16),
        scale=1.0,
        initial_state=torch.zeros((1, 1, 2, 2), dtype=torch.float32),
        inplace_final_state=True,
    )

    assert torch.equal(output, expected_output)
    assert torch.equal(final_state, expected_state)


def test_fused_recurrent_fwd_skips_invalid_continuous_batching_slots(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fused_recurrent, "HAS_TRITON", False)
    monkeypatch.setattr(
        fused_recurrent,
        "_try_precompiled_fused_recurrent_gated_delta_rule_fwd",
        lambda **_: None,
    )

    q = torch.tensor(
        [
            [[[1.0, 0.0]]],
            [[[1.0, 0.0]]],
        ],
        dtype=torch.float16,
    )
    k = torch.tensor(
        [
            [[[1.0, 0.0]]],
            [[[1.0, 0.0]]],
        ],
        dtype=torch.float16,
    )
    v = torch.tensor(
        [
            [[[10.0, 20.0]]],
            [[[30.0, 40.0]]],
        ],
        dtype=torch.float16,
    )
    g = torch.zeros((2, 1, 1), dtype=torch.float16)
    beta = torch.ones((2, 1, 1), dtype=torch.float16)
    initial_state = torch.zeros((1, 1, 2, 2), dtype=torch.float32)

    output, final_state = fused_recurrent.fused_recurrent_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=1.0,
        initial_state=initial_state,
        inplace_final_state=True,
        ssm_state_indices=torch.tensor([0, -1], dtype=torch.int32),
    )

    expected_output = torch.tensor(
        [
            [[[10.0, 20.0]]],
            [[[0.0, 0.0]]],
        ],
        dtype=torch.float16,
    )
    expected_state = torch.tensor(
        [[[[10.0, 0.0], [20.0, 0.0]]]],
        dtype=torch.float32,
    )

    torch.testing.assert_close(output, expected_output)
    torch.testing.assert_close(final_state, expected_state)


def test_kv_block_zeroer_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(worker_utils, "HAS_TRITON", False)

    zeroer = worker_utils.KVBlockZeroer(device=torch.device("cpu"), pin_memory=False)
    zeroer._meta = (torch.empty(0, dtype=torch.int64), 4, 4, 1)
    kv = torch.arange(8, dtype=torch.int32).reshape(2, 4)
    zeroer._fallback_blocks = [(kv, 0, 1)]

    zeroer.zero_block_ids([1])

    assert torch.equal(
        kv,
        torch.tensor([[0, 1, 2, 3], [0, 0, 0, 0]], dtype=torch.int32),
    )


def test_kv_block_zeroer_falls_back_without_triton_for_kv_outer_layout(
    monkeypatch,
) -> None:
    monkeypatch.setattr(worker_utils, "HAS_TRITON", False)

    zeroer = worker_utils.KVBlockZeroer(device=torch.device("cpu"), pin_memory=False)
    zeroer._meta = (torch.empty(0, dtype=torch.int64), 4, 4, 1)
    kv = torch.arange(12, dtype=torch.int32).reshape(2, 3, 2)
    zeroer._fallback_blocks = [(kv, 1, 1)]

    zeroer.zero_block_ids([1])

    assert torch.equal(
        kv,
        torch.tensor(
            [
                [[0, 1], [0, 0], [4, 5]],
                [[6, 7], [0, 0], [10, 11]],
            ],
            dtype=torch.int32,
        ),
    )


def test_kv_block_zeroer_uses_precompiled_op_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(worker_utils, "HAS_TRITON", False)

    zeroer = worker_utils.KVBlockZeroer(device=torch.device("cpu"), pin_memory=False)
    zeroer._meta = (torch.empty(0, dtype=torch.int64), 4, 4, 1)
    zeroer._fallback_blocks = []

    called = {"value": False}

    def _fake_try_precompiled(zeroer_obj, block_ids):
        called["value"] = True
        assert block_ids == [1]
        return True

    monkeypatch.setattr(worker_utils, "_try_precompiled_zero_kv_blocks", _fake_try_precompiled)

    zeroer.zero_block_ids([1])

    assert called["value"] is True


def test_qwen3_next_fused_gdn_gating_falls_back_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(qwen3_next, "HAS_TRITON", False)

    A_log = torch.tensor([0.0, 0.5], dtype=torch.float16)
    a = torch.tensor([[1.0, -0.5]], dtype=torch.float16)
    b = torch.tensor([[0.0, 2.0]], dtype=torch.float16)
    dt_bias = torch.tensor([0.25, -0.25], dtype=torch.float16)

    actual_g, actual_beta = qwen3_next.fused_gdn_gating(A_log, a, b, dt_bias)

    expected_g = (
        -torch.exp(A_log.to(torch.float32))
        * torch.nn.functional.softplus(
            a.to(torch.float32) + dt_bias.to(torch.float32),
            beta=1.0,
            threshold=20.0,
        )
    ).unsqueeze(0)
    expected_beta = torch.sigmoid(b.to(torch.float32)).to(b.dtype).unsqueeze(0)

    assert torch.allclose(actual_g, expected_g)
    assert torch.allclose(actual_beta, expected_beta)


def test_qwen3_next_fused_gdn_gating_uses_precompiled_op_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(qwen3_next, "HAS_TRITON", False)
    expected_g = torch.full((1, 1, 2), 3.0, dtype=torch.float32)
    expected_beta = torch.full((1, 1, 2), 4.0, dtype=torch.float16)
    monkeypatch.setattr(
        qwen3_next,
        "_try_precompiled_fused_gdn_gating",
        lambda **_: (expected_g, expected_beta),
    )

    actual_g, actual_beta = qwen3_next.fused_gdn_gating(
        torch.tensor([0.0, 0.5], dtype=torch.float16),
        torch.tensor([[1.0, -0.5]], dtype=torch.float16),
        torch.tensor([[0.0, 2.0]], dtype=torch.float16),
        torch.tensor([0.25, -0.25], dtype=torch.float16),
    )

    assert torch.equal(actual_g, expected_g)
    assert torch.equal(actual_beta, expected_beta)


def test_profile_cudagraph_memory_skips_with_tiered_moe_cache() -> None:
    dummy_runner = SimpleNamespace(
        _has_capture_unsafe_tiered_moe_cache=lambda: True,
    )

    assert gpu_model_runner.GPUModelRunner.profile_cudagraph_memory(dummy_runner) == 0


def test_capture_model_skips_with_tiered_moe_cache() -> None:
    dummy_runner = SimpleNamespace(
        _has_capture_unsafe_tiered_moe_cache=lambda: True,
    )

    assert gpu_model_runner.GPUModelRunner.capture_model(dummy_runner) == 0


def test_profile_cudagraph_memory_skips_with_cpu_offload() -> None:
    dummy_runner = SimpleNamespace(
        _has_capture_unsafe_tiered_moe_cache=lambda: False,
        offload_config=SimpleNamespace(
            uva=SimpleNamespace(cpu_offload_gb=8),
        )
    )

    assert gpu_model_runner.GPUModelRunner.profile_cudagraph_memory(dummy_runner) == 0


def test_input_batch_reinit_allows_startup_with_cpu_offload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warning_calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        gpu_model_runner,
        "logger",
        SimpleNamespace(warning_once=lambda *args, **kwargs: warning_calls.append(args)),
    )
    dummy_runner = SimpleNamespace(
        offload_config=SimpleNamespace(
            uva=SimpleNamespace(cpu_offload_gb=8),
        ),
        requests={},
        _init_block_sizes=[16],
        _init_kernel_block_sizes=[16],
    )

    gpu_model_runner.GPUModelRunner._ensure_input_batch_reinit_allowed(
        dummy_runner,
        [2096, 256],
        [64, 256],
    )

    assert warning_calls


def test_input_batch_reinit_rejects_live_requests_with_cpu_offload() -> None:
    dummy_runner = SimpleNamespace(
        offload_config=SimpleNamespace(
            uva=SimpleNamespace(cpu_offload_gb=8),
        ),
        requests={"req-1": object()},
        _init_block_sizes=[16],
        _init_kernel_block_sizes=[16],
    )

    with pytest.raises(
        AssertionError,
        match="Cannot re-initialize the input batch when CPU weight offloading is enabled",
    ):
        gpu_model_runner.GPUModelRunner._ensure_input_batch_reinit_allowed(
            dummy_runner,
            [2096, 256],
            [64, 256],
        )


def test_batch_invariant_math_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(batch_invariant, "HAS_TRITON", False)

    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32)
    bias = torch.tensor([0.5, -0.5], dtype=torch.float32)
    torch.testing.assert_close(
        batch_invariant.matmul_persistent(a, b, bias=bias),
        torch.matmul(a, b) + bias,
    )

    bmm_a = a.reshape(1, 2, 2)
    bmm_b = b.reshape(1, 2, 2)
    torch.testing.assert_close(
        batch_invariant.bmm_batch_invariant(bmm_a, bmm_b),
        torch.matmul(bmm_a, bmm_b),
    )

    logits = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    torch.testing.assert_close(
        batch_invariant.log_softmax(logits, dim=-1),
        torch.log_softmax(logits, dim=-1),
    )

    values = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    torch.testing.assert_close(
        batch_invariant.mean_dim(values, dim=1),
        torch.mean(values, dim=1),
    )

    weight = torch.tensor([1.0, 0.5], dtype=torch.float32)
    norm_input = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    variance = norm_input.pow(2).mean(dim=-1, keepdim=True)
    expected_norm = norm_input * torch.rsqrt(variance + 1e-6) * weight
    torch.testing.assert_close(
        batch_invariant.rms_norm(norm_input, weight, eps=1e-6),
        expected_norm,
    )


def test_batch_invariant_enable_skips_aten_override_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(batch_invariant, "HAS_TRITON", False)
    monkeypatch.setattr(batch_invariant, "_batch_invariant_MODE", False)
    monkeypatch.setattr(batch_invariant, "_batch_invariant_LIB", None)

    batch_invariant.enable_batch_invariant_mode()

    assert batch_invariant._batch_invariant_MODE is True
    assert batch_invariant._batch_invariant_LIB is None


def test_deep_gemm_ep_scatter_gather_falls_back_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(deep_gemm_utils, "HAS_TRITON", False)

    hidden_size = 128
    recv_x = torch.arange(3 * hidden_size, dtype=torch.float16).reshape(
        3, hidden_size
    )
    recv_x_scale = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
    recv_topk = torch.tensor([[0, 1], [1, -1], [2, 0]], dtype=torch.int32)
    num_recv_tokens_per_expert = torch.tensor([2, 2, 1], dtype=torch.int32)
    expert_start_loc = torch.zeros(3, dtype=torch.int32)
    output_tensor = torch.zeros((384, hidden_size), dtype=torch.float16)
    output_tensor_scale = torch.zeros((384, 1), dtype=torch.float32)
    m_indices = torch.full((384,), -1, dtype=torch.int32)
    output_index = torch.full((3, 2), -1, dtype=torch.int32)

    deep_gemm_utils.ep_scatter(
        recv_x,
        recv_x_scale,
        recv_topk,
        num_recv_tokens_per_expert,
        expert_map=None,
        expert_start_loc=expert_start_loc,
        output_tensor=output_tensor,
        output_tensor_scale=output_tensor_scale,
        m_indices=m_indices,
        output_index=output_index,
    )

    assert torch.equal(expert_start_loc, torch.tensor([2, 130, 257], dtype=torch.int32))
    assert torch.equal(
        output_index,
        torch.tensor([[0, 128], [129, -1], [256, 1]], dtype=torch.int32),
    )
    assert torch.equal(m_indices[:2], torch.tensor([0, 0], dtype=torch.int32))
    assert torch.equal(m_indices[128:130], torch.tensor([1, 1], dtype=torch.int32))
    assert torch.equal(m_indices[256:257], torch.tensor([2], dtype=torch.int32))
    assert torch.equal(output_tensor[0], recv_x[0])
    assert torch.equal(output_tensor[128], recv_x[0])
    assert torch.equal(output_tensor[129], recv_x[1])
    assert torch.equal(output_tensor[256], recv_x[2])
    assert torch.equal(output_tensor[1], recv_x[2])
    assert torch.equal(output_tensor_scale[256], recv_x_scale[2])

    recv_topk_weight = torch.tensor(
        [[0.5, 0.5], [1.0, 0.0], [0.25, 0.75]],
        dtype=torch.float32,
    )
    gathered = torch.empty((3, hidden_size), dtype=torch.float16)
    deep_gemm_utils.ep_gather(
        output_tensor,
        recv_topk,
        recv_topk_weight,
        output_index,
        expert_map=None,
        output_tensor=gathered,
    )

    expected = torch.stack(
        [
            (output_tensor[0].float() * 0.5 + output_tensor[128].float() * 0.5),
            output_tensor[129].float(),
            (output_tensor[256].float() * 0.25 + output_tensor[1].float() * 0.75),
        ]
    ).to(torch.float16)
    assert torch.equal(gathered, expected)


def test_deepgemm_moe_permute_and_unpermute_fall_back_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(deep_gemm_utils, "HAS_TRITON", False)
    monkeypatch.setattr(
        deep_gemm_utils,
        "get_mk_alignment_for_contiguous_layout",
        lambda: [128, 128],
    )

    hidden_size = 128
    aq = torch.arange(3 * hidden_size, dtype=torch.float16).reshape(3, hidden_size)
    aq_scale = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
    topk_ids = torch.tensor([[0, 1], [1, -1], [2, 0]], dtype=torch.int32)
    topk_weights = torch.tensor(
        [[0.5, 0.5], [1.0, 0.0], [0.25, 0.75]],
        dtype=torch.float32,
    )

    aq_out, aq_scale_out, expert_ids, inv_perm = deep_gemm_utils.deepgemm_moe_permute(
        aq=aq,
        aq_scale=aq_scale,
        topk_ids=topk_ids,
        local_num_experts=3,
        expert_map=None,
        expert_tokens_meta=SimpleNamespace(
            expert_num_tokens=torch.tensor([2, 2, 1], dtype=torch.int32),
            expert_num_tokens_cpu=torch.tensor([2, 2, 1], dtype=torch.int32),
        ),
    )

    assert aq_out.shape == (384, hidden_size)
    assert aq_scale_out.shape == (384, 1)
    assert inv_perm[0, 0].item() == 0
    assert inv_perm[0, 1].item() == 128
    assert inv_perm[1, 0].item() == 129
    assert inv_perm[2, 0].item() == 256
    assert inv_perm[2, 1].item() == 1
    assert torch.equal(expert_ids[:2], torch.tensor([0, 0], dtype=torch.int32))
    assert torch.equal(expert_ids[128:130], torch.tensor([1, 1], dtype=torch.int32))
    assert torch.equal(expert_ids[256:257], torch.tensor([2], dtype=torch.int32))
    assert torch.equal(aq_out[0], aq[0])
    assert torch.equal(aq_out[128], aq[0])
    assert torch.equal(aq_out[129], aq[1])
    assert torch.equal(aq_out[256], aq[2])
    assert torch.equal(aq_out[1], aq[2])
    assert torch.equal(aq_scale_out[256], aq_scale[2])

    output = torch.empty((3, hidden_size), dtype=torch.float16)
    deep_gemm_utils.deepgemm_unpermute_and_reduce(
        a=aq_out,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        inv_perm=inv_perm,
        expert_map=None,
        output=output,
    )

    expected = torch.stack(
        [
            (aq_out[0].float() * 0.5 + aq_out[128].float() * 0.5),
            aq_out[129].float(),
            (aq_out[256].float() * 0.25 + aq_out[1].float() * 0.75),
        ]
    ).to(torch.float16)
    assert torch.equal(output, expected)


def test_deepgemm_moe_permute_and_unpermute_with_expert_map_fall_back_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(deep_gemm_utils, "HAS_TRITON", False)
    monkeypatch.setattr(
        deep_gemm_utils,
        "get_mk_alignment_for_contiguous_layout",
        lambda: [128, 128],
    )

    hidden_size = 128
    aq = torch.arange(3 * hidden_size, dtype=torch.float16).reshape(3, hidden_size)
    aq_scale = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
    topk_ids = torch.tensor([[0, 1], [1, -1], [2, 0]], dtype=torch.int32)
    topk_weights = torch.tensor(
        [[0.75, 0.25], [1.0, 0.0], [0.2, 0.8]],
        dtype=torch.float32,
    )
    expert_map = torch.tensor([2, 0, 1], dtype=torch.int32)

    aq_out, aq_scale_out, expert_ids, inv_perm = deep_gemm_utils.deepgemm_moe_permute(
        aq=aq,
        aq_scale=aq_scale,
        topk_ids=topk_ids,
        local_num_experts=3,
        expert_map=expert_map,
        expert_tokens_meta=None,
    )

    assert aq_out.shape == (512, hidden_size)
    assert aq_scale_out.shape == (512, 1)
    assert inv_perm[0, 0].item() == 256
    assert inv_perm[0, 1].item() == 0
    assert inv_perm[1, 0].item() == 1
    assert inv_perm[2, 0].item() == 128
    assert inv_perm[2, 1].item() == 257
    assert torch.equal(expert_ids[:2], torch.tensor([0, 0], dtype=torch.int32))
    assert torch.equal(expert_ids[128:129], torch.tensor([1], dtype=torch.int32))
    assert torch.equal(expert_ids[256:258], torch.tensor([2, 2], dtype=torch.int32))
    assert torch.all(expert_ids[258:] == -1).item()
    assert torch.equal(aq_out[256], aq[0])
    assert torch.equal(aq_out[0], aq[0])
    assert torch.equal(aq_out[1], aq[1])
    assert torch.equal(aq_out[128], aq[2])
    assert torch.equal(aq_out[257], aq[2])
    assert torch.equal(aq_scale_out[257], aq_scale[2])

    output = torch.empty((3, hidden_size), dtype=torch.float16)
    deep_gemm_utils.deepgemm_unpermute_and_reduce(
        a=aq_out,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        inv_perm=inv_perm,
        expert_map=expert_map,
        output=output,
    )

    expected = torch.stack(
        [
            (aq_out[256].float() * 0.75 + aq_out[0].float() * 0.25),
            aq_out[1].float(),
            (aq_out[128].float() * 0.2 + aq_out[257].float() * 0.8),
        ]
    ).to(torch.float16)
    assert torch.equal(output, expected)


def test_block_table_gather_and_slot_mapping_fall_back_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(block_table_mod, "HAS_TRITON", False)

    gather_tables = [
        SimpleNamespace(
            gpu=torch.tensor(
                [[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]],
                dtype=torch.int32,
            )
        ),
        SimpleNamespace(
            gpu=torch.tensor(
                [[100, 101, 102], [200, 201, 202], [300, 301, 302]],
                dtype=torch.int32,
            )
        ),
    ]
    gather_dummy = SimpleNamespace(
        num_kv_cache_groups=2,
        block_tables=gather_tables,
        input_block_tables=[
            torch.zeros((3, 4), dtype=torch.int32),
            torch.zeros((3, 3), dtype=torch.int32),
        ],
        num_blocks=SimpleNamespace(
            gpu=torch.tensor([[3, 1, 2], [2, 1, 1]], dtype=torch.int32)
        ),
    )
    gather_dummy._gather_block_tables_torch = (
        block_table_mod.BlockTables._gather_block_tables_torch.__get__(gather_dummy)
    )

    gathered = block_table_mod.BlockTables.gather_block_tables(
        gather_dummy,
        torch.tensor([2, 0], dtype=torch.int32),
        num_reqs_padded=3,
    )

    assert torch.equal(gathered[0][0, :2], torch.tensor([30, 31], dtype=torch.int32))
    assert torch.equal(gathered[0][1, :3], torch.tensor([10, 11, 12], dtype=torch.int32))
    assert torch.equal(gathered[0][2], torch.zeros(4, dtype=torch.int32))
    assert torch.equal(gathered[1][0, :1], torch.tensor([300], dtype=torch.int32))
    assert torch.equal(gathered[1][1, :2], torch.tensor([100, 101], dtype=torch.int32))
    assert torch.equal(gathered[1][2], torch.zeros(3, dtype=torch.int32))

    slot_dummy = SimpleNamespace(
        num_kv_cache_groups=1,
        block_sizes=[4],
        max_num_batched_tokens=8,
        cp_size=1,
        cp_rank=0,
        cp_interleave=1,
        block_tables=[
            SimpleNamespace(
                gpu=torch.tensor(
                    [[5, 6], [7, 8], [9, 10]],
                    dtype=torch.int32,
                )
            )
        ],
        slot_mappings=torch.zeros((1, 8), dtype=torch.int64),
    )
    slot_dummy._compute_slot_mappings_torch = (
        block_table_mod.BlockTables._compute_slot_mappings_torch.__get__(slot_dummy)
    )

    slot_mappings = block_table_mod.BlockTables.compute_slot_mappings(
        slot_dummy,
        torch.tensor([2, 0], dtype=torch.int32),
        torch.tensor([0, 3, 5], dtype=torch.int32),
        torch.tensor([0, 1, 5, 4, 7], dtype=torch.int64),
        num_tokens_padded=8,
    )

    assert torch.equal(
        slot_mappings,
        torch.tensor([[36, 37, 41, 24, 27, -1, -1, -1]], dtype=torch.int64),
    )


def test_input_batch_prepare_prefill_inputs_falls_back_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(input_batch_mod, "HAS_TRITON", False)

    input_ids = torch.full((4,), -1, dtype=torch.int32)
    next_prefill_tokens = torch.full((2,), -1, dtype=torch.int32)
    idx_mapping = torch.tensor([0, 1], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 2, 4], dtype=torch.int32)
    all_token_ids = torch.tensor(
        [[10, 11, 12, 13, 14], [20, 21, 22, 23, 24]],
        dtype=torch.int32,
    )
    prefill_len = torch.tensor([5, 4], dtype=torch.int32)
    num_computed_tokens = torch.tensor([1, 4], dtype=torch.int32)

    input_batch_mod.prepare_prefill_inputs(
        input_ids,
        next_prefill_tokens,
        idx_mapping,
        query_start_loc,
        all_token_ids,
        prefill_len,
        num_computed_tokens,
    )

    torch.testing.assert_close(
        input_ids,
        torch.tensor([11, 12, -1, -1], dtype=torch.int32),
    )
    torch.testing.assert_close(
        next_prefill_tokens,
        torch.tensor([13, -1], dtype=torch.int32),
    )


def test_input_batch_prepare_prefill_inputs_uses_precompiled_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(input_batch_mod, "HAS_TRITON", False)
    monkeypatch.setattr(
        input_batch_mod._custom_ops,
        "has_precompiled_input_batch_prepare_prefill_inputs",
        lambda: True,
    )

    called = {"value": False}

    def _fake_precompiled(*args):
        called["value"] = True
        assert args[0].is_cuda is True

    monkeypatch.setattr(
        input_batch_mod._custom_ops,
        "input_batch_prepare_prefill_inputs_precompiled",
        _fake_precompiled,
    )

    input_batch_mod.prepare_prefill_inputs(
        _as_fake_cuda(torch.empty(4, dtype=torch.int32)),
        torch.empty(2, dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.empty((1, 4), dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
    )

    assert called["value"] is True


def test_input_batch_prepare_pos_seq_lens_falls_back_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(input_batch_mod, "HAS_TRITON", False)

    idx_mapping = torch.tensor([1, 0], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 2, 5], dtype=torch.int32)
    num_computed_tokens = torch.tensor([4, 1], dtype=torch.int32)
    pos = torch.full((5,), -1, dtype=torch.int64)
    seq_lens = torch.full((4,), -1, dtype=torch.int32)

    input_batch_mod.prepare_pos_seq_lens(
        idx_mapping,
        query_start_loc,
        num_computed_tokens,
        pos,
        seq_lens,
    )

    torch.testing.assert_close(
        pos,
        torch.tensor([1, 2, 4, 5, 6], dtype=torch.int64),
    )
    torch.testing.assert_close(
        seq_lens,
        torch.tensor([3, 7, 0, 0], dtype=torch.int32),
    )


def test_input_batch_prepare_pos_seq_lens_uses_precompiled_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(input_batch_mod, "HAS_TRITON", False)
    monkeypatch.setattr(
        input_batch_mod._custom_ops,
        "has_precompiled_input_batch_prepare_pos_seq_lens",
        lambda: True,
    )

    called = {"value": False}

    def _fake_precompiled(*args):
        called["value"] = True
        assert args[3].is_cuda is True

    monkeypatch.setattr(
        input_batch_mod._custom_ops,
        "input_batch_prepare_pos_seq_lens_precompiled",
        _fake_precompiled,
    )

    input_batch_mod.prepare_pos_seq_lens(
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        _as_fake_cuda(torch.empty(1, dtype=torch.int64)),
        torch.empty(1, dtype=torch.int32),
    )

    assert called["value"] is True


def test_input_batch_combine_sampled_and_draft_tokens_falls_back_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(input_batch_mod, "HAS_TRITON", False)

    input_ids = torch.zeros((6,), dtype=torch.int32)
    idx_mapping = torch.tensor([0, 1], dtype=torch.int32)
    last_sampled_tokens = torch.tensor([99, 77], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 3, 6], dtype=torch.int32)
    seq_lens = torch.tensor([5, 2], dtype=torch.int32)
    prefill_len = torch.tensor([3, 2], dtype=torch.int32)
    draft_tokens = torch.tensor([[88, 66], [55, 44]], dtype=torch.int32)
    cu_num_logits = torch.tensor([0, 2, 3], dtype=torch.int32)

    logits_indices = input_batch_mod.combine_sampled_and_draft_tokens(
        input_ids,
        idx_mapping,
        last_sampled_tokens,
        query_start_loc,
        seq_lens,
        prefill_len,
        draft_tokens,
        cu_num_logits,
        num_logits=3,
    )

    torch.testing.assert_close(
        input_ids,
        torch.tensor([0, 99, 88, 0, 0, 0], dtype=torch.int32),
    )
    torch.testing.assert_close(
        logits_indices,
        torch.tensor([1, 2, 5], dtype=torch.int64),
    )


def test_input_batch_combine_sampled_and_draft_tokens_uses_precompiled_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(input_batch_mod, "HAS_TRITON", False)
    monkeypatch.setattr(
        input_batch_mod._custom_ops,
        "has_precompiled_input_batch_combine_sampled_and_draft_tokens",
        lambda: True,
    )

    expected = torch.tensor([1, 3], dtype=torch.int64)
    monkeypatch.setattr(
        input_batch_mod._custom_ops,
        "input_batch_combine_sampled_and_draft_tokens_precompiled",
        lambda *args: expected,
    )

    actual = input_batch_mod.combine_sampled_and_draft_tokens(
        _as_fake_cuda(torch.empty(4, dtype=torch.int32)),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([9], dtype=torch.int32),
        torch.tensor([0, 2], dtype=torch.int32),
        torch.tensor([4], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
        torch.tensor([[8]], dtype=torch.int32),
        torch.tensor([0, 2], dtype=torch.int32),
        num_logits=2,
    )

    assert actual is expected


def test_input_batch_get_num_sampled_and_rejected_falls_back_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(input_batch_mod, "HAS_TRITON", False)

    num_sampled = torch.tensor([2, 1], dtype=torch.int32)
    seq_lens = torch.tensor([5, 2], dtype=torch.int32)
    cu_num_logits = torch.tensor([0, 3, 5], dtype=torch.int32)
    idx_mapping = torch.tensor([0, 1], dtype=torch.int32)
    prefill_len = torch.tensor([3, 4], dtype=torch.int32)

    actual_sampled, actual_rejected = input_batch_mod.get_num_sampled_and_rejected(
        num_sampled,
        seq_lens,
        cu_num_logits,
        idx_mapping,
        prefill_len,
    )

    torch.testing.assert_close(
        actual_sampled,
        torch.tensor([2, 0], dtype=torch.int32),
    )
    torch.testing.assert_close(
        actual_rejected,
        torch.tensor([1, 0], dtype=torch.int32),
    )


def test_input_batch_get_num_sampled_and_rejected_uses_precompiled_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(input_batch_mod, "HAS_TRITON", False)
    monkeypatch.setattr(
        input_batch_mod._custom_ops,
        "has_precompiled_input_batch_get_num_sampled_and_rejected",
        lambda: True,
    )

    expected_sampled = torch.tensor([1], dtype=torch.int32)
    expected_rejected = torch.tensor([0], dtype=torch.int32)
    monkeypatch.setattr(
        input_batch_mod._custom_ops,
        "input_batch_get_num_sampled_and_rejected_precompiled",
        lambda *args: (expected_sampled, expected_rejected),
    )

    actual_sampled, actual_rejected = input_batch_mod.get_num_sampled_and_rejected(
        _as_fake_cuda(torch.tensor([1], dtype=torch.int32)),
        torch.tensor([2], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
    )

    assert actual_sampled is expected_sampled
    assert actual_rejected is expected_rejected


def test_input_batch_post_update_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(input_batch_mod, "HAS_TRITON", False)

    idx_mapping = torch.tensor([1, 0], dtype=torch.int32)
    num_computed_tokens = torch.tensor([4, 1], dtype=torch.int32)
    last_sampled_tokens = torch.full((2,), -1, dtype=torch.int32)
    output_bin_counts = torch.zeros((2, 10), dtype=torch.int32)
    sampled_tokens = torch.tensor([[7, 8, 9], [3, 4, 5]], dtype=torch.int32)
    num_sampled = torch.tensor([2, 1], dtype=torch.int32)
    num_rejected = torch.tensor([1, 0], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 3, 5], dtype=torch.int32)
    all_token_ids = torch.zeros((2, 8), dtype=torch.int32)
    total_len = torch.tensor([4, 2], dtype=torch.int32)

    input_batch_mod.post_update(
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

    torch.testing.assert_close(
        num_computed_tokens,
        torch.tensor([6, 3], dtype=torch.int32),
    )
    torch.testing.assert_close(
        last_sampled_tokens,
        torch.tensor([3, 8], dtype=torch.int32),
    )
    torch.testing.assert_close(
        total_len,
        torch.tensor([5, 4], dtype=torch.int32),
    )
    torch.testing.assert_close(
        all_token_ids,
        torch.tensor(
            [[0, 0, 0, 0, 3, 0, 0, 0], [0, 0, 7, 8, 0, 0, 0, 0]],
            dtype=torch.int32,
        ),
    )
    assert int(output_bin_counts[0, 3]) == 1
    assert int(output_bin_counts[1, 7]) == 1
    assert int(output_bin_counts[1, 8]) == 1


def test_input_batch_post_update_uses_precompiled_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(input_batch_mod, "HAS_TRITON", False)
    monkeypatch.setattr(
        input_batch_mod._custom_ops,
        "has_precompiled_input_batch_post_update",
        lambda: True,
    )

    called = {"value": False}

    def _fake_precompiled(*args):
        called["value"] = True
        assert args[8].is_cuda is True

    monkeypatch.setattr(
        input_batch_mod._custom_ops,
        "input_batch_post_update_precompiled",
        _fake_precompiled,
    )

    input_batch_mod.post_update(
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        None,
        torch.tensor([[1]], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        _as_fake_cuda(torch.empty((1, 4), dtype=torch.int32)),
        torch.tensor([0], dtype=torch.int32),
    )

    assert called["value"] is True


def test_input_batch_post_update_pool_and_expand_idx_mapping_fall_back_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(input_batch_mod, "HAS_TRITON", False)

    idx_mapping = torch.tensor([1, 0], dtype=torch.int32)
    num_computed_tokens = torch.tensor([4, 1], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 2, 5], dtype=torch.int32)

    input_batch_mod.post_update_pool(
        idx_mapping,
        num_computed_tokens,
        query_start_loc,
    )

    torch.testing.assert_close(
        num_computed_tokens,
        torch.tensor([7, 3], dtype=torch.int32),
    )

    expanded_idx_mapping, expanded_local_pos = input_batch_mod.expand_idx_mapping(
        torch.tensor([4, 2], dtype=torch.int32),
        total_num_logits=5,
        cu_num_logits=torch.tensor([0, 2, 5], dtype=torch.int32),
        max_expand_len=3,
    )

    torch.testing.assert_close(
        expanded_idx_mapping,
        torch.tensor([4, 4, 2, 2, 2], dtype=torch.int32),
    )
    torch.testing.assert_close(
        expanded_local_pos,
        torch.tensor([0, 1, 0, 1, 2], dtype=torch.int32),
    )


def test_input_batch_post_update_pool_uses_precompiled_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(input_batch_mod, "HAS_TRITON", False)
    monkeypatch.setattr(
        input_batch_mod._custom_ops,
        "has_precompiled_input_batch_post_update_pool",
        lambda: True,
    )

    called = {"value": False}

    def _fake_precompiled(*args):
        called["value"] = True
        assert args[1].is_cuda is True

    monkeypatch.setattr(
        input_batch_mod._custom_ops,
        "input_batch_post_update_pool_precompiled",
        _fake_precompiled,
    )

    input_batch_mod.post_update_pool(
        torch.tensor([0], dtype=torch.int32),
        _as_fake_cuda(torch.tensor([0], dtype=torch.int32)),
        torch.tensor([0, 1], dtype=torch.int32),
    )

    assert called["value"] is True


def test_input_batch_expand_idx_mapping_uses_precompiled_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(input_batch_mod, "HAS_TRITON", False)
    monkeypatch.setattr(
        input_batch_mod._custom_ops,
        "has_precompiled_input_batch_expand_idx_mapping",
        lambda: True,
    )

    expected_idx = torch.tensor([3, 3], dtype=torch.int32)
    expected_pos = torch.tensor([0, 1], dtype=torch.int32)
    monkeypatch.setattr(
        input_batch_mod._custom_ops,
        "input_batch_expand_idx_mapping_precompiled",
        lambda *args: (expected_idx, expected_pos),
    )

    actual_idx, actual_pos = input_batch_mod.expand_idx_mapping(
        _as_fake_cuda(torch.tensor([3], dtype=torch.int32)),
        total_num_logits=2,
        cu_num_logits=torch.tensor([0, 2], dtype=torch.int32),
        max_expand_len=2,
    )

    assert actual_idx is expected_idx
    assert actual_pos is expected_pos


def test_eagle_speculator_helpers_fall_back_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(eagle_speculator, "HAS_TRITON", False)

    input_buffers = eagle_speculator.InputBuffers(
        max_num_reqs=4,
        max_num_tokens=8,
        device=torch.device("cpu"),
    )
    input_batch = SimpleNamespace(
        num_reqs=2,
        input_ids=torch.tensor([10, 11, 12, 20, 21], dtype=torch.int32),
        positions=torch.tensor([0, 1, 2, 0, 1], dtype=torch.int64),
        idx_mapping=torch.tensor([2, 0], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 3, 5], dtype=torch.int32),
    )
    last_token_indices = eagle_speculator.prepare_eagle_inputs(
        input_buffers,
        input_batch,
        num_sampled=torch.tensor([1, 0], dtype=torch.int32),
        num_rejected=torch.tensor([1, 0], dtype=torch.int32),
        last_sampled=torch.tensor([100, 101, 102], dtype=torch.int32),
        next_prefill_tokens=torch.tensor([200, 201, 202], dtype=torch.int32),
    )

    assert torch.equal(last_token_indices, torch.tensor([1, 4], dtype=torch.int64))
    assert torch.equal(
        input_buffers.input_ids[:5],
        torch.tensor([11, 102, 0, 21, 200], dtype=torch.int32),
    )
    assert torch.equal(
        input_buffers.positions[:5],
        torch.tensor([0, 1, 0, 0, 1], dtype=torch.int64),
    )

    input_buffers.positions[:2] = torch.tensor([5, 3], dtype=torch.int64)
    decode_hidden_states = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    eagle_hidden_states = torch.zeros((8, 4), dtype=torch.float32)
    eagle_speculator.prepare_eagle_decode(
        draft_tokens=torch.tensor([7, 8], dtype=torch.int64),
        output_hidden_states=decode_hidden_states,
        last_token_indices=last_token_indices,
        target_seq_lens=torch.tensor([6, 4], dtype=torch.int32),
        num_rejected=torch.tensor([1, 0], dtype=torch.int32),
        input_buffers=input_buffers,
        input_hidden_states=eagle_hidden_states,
        max_model_len=16,
        max_num_reqs=4,
    )

    assert torch.equal(
        input_buffers.query_start_loc[:5],
        torch.tensor([0, 1, 2, 2, 2], dtype=torch.int32),
    )
    assert torch.equal(
        input_buffers.seq_lens[:4],
        torch.tensor([6, 5, 0, 0], dtype=torch.int32),
    )
    assert torch.equal(input_buffers.input_ids[:2], torch.tensor([7, 8], dtype=torch.int32))
    assert torch.equal(input_buffers.positions[:2], torch.tensor([6, 4], dtype=torch.int64))
    assert torch.equal(eagle_hidden_states[0], decode_hidden_states[1])
    assert torch.equal(eagle_hidden_states[1], decode_hidden_states[4])

    eagle_speculator.update_eagle_inputs(
        draft_tokens=torch.tensor([9, 10], dtype=torch.int64),
        output_hidden_states=torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=torch.float32,
        ),
        input_buffers=input_buffers,
        hidden_states=eagle_hidden_states[:, :2],
        max_model_len=16,
    )

    assert torch.equal(input_buffers.input_ids[:2], torch.tensor([9, 10], dtype=torch.int32))
    assert torch.equal(input_buffers.positions[:2], torch.tensor([7, 5], dtype=torch.int64))
    assert torch.equal(
        input_buffers.seq_lens[:2],
        torch.tensor([7, 6], dtype=torch.int32),
    )
    assert torch.equal(
        eagle_hidden_states[:2, :2],
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
    )


def test_eagle_speculator_decode_helpers_clamp_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(eagle_speculator, "HAS_TRITON", False)

    input_buffers = eagle_speculator.InputBuffers(
        max_num_reqs=3,
        max_num_tokens=4,
        device=torch.device("cpu"),
    )
    input_buffers.positions[:1] = torch.tensor([15], dtype=torch.int64)
    input_hidden_states = torch.zeros((4, 2), dtype=torch.float32)

    eagle_speculator.prepare_eagle_decode(
        draft_tokens=torch.tensor([7], dtype=torch.int64),
        output_hidden_states=torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=torch.float32,
        ),
        last_token_indices=torch.tensor([1], dtype=torch.int64),
        target_seq_lens=torch.tensor([16], dtype=torch.int32),
        num_rejected=torch.tensor([0], dtype=torch.int32),
        input_buffers=input_buffers,
        input_hidden_states=input_hidden_states,
        max_model_len=16,
        max_num_reqs=3,
    )

    assert torch.equal(input_buffers.input_ids[:1], torch.tensor([7], dtype=torch.int32))
    assert torch.equal(input_buffers.positions[:1], torch.tensor([15], dtype=torch.int64))
    assert torch.equal(input_buffers.seq_lens[:3], torch.tensor([16, 0, 0], dtype=torch.int32))
    assert torch.equal(
        input_buffers.query_start_loc[:4],
        torch.tensor([0, 1, 1, 1], dtype=torch.int32),
    )
    assert torch.equal(
        input_hidden_states[0],
        torch.tensor([3.0, 4.0], dtype=torch.float32),
    )

    eagle_speculator.update_eagle_inputs(
        draft_tokens=torch.tensor([8], dtype=torch.int64),
        output_hidden_states=torch.tensor([[5.0, 6.0]], dtype=torch.float32),
        input_buffers=input_buffers,
        hidden_states=input_hidden_states,
        max_model_len=16,
    )

    assert torch.equal(input_buffers.input_ids[:1], torch.tensor([8], dtype=torch.int32))
    assert torch.equal(input_buffers.positions[:1], torch.tensor([15], dtype=torch.int64))
    assert torch.equal(input_buffers.seq_lens[:1], torch.tensor([16], dtype=torch.int32))
    assert torch.equal(
        input_hidden_states[0],
        torch.tensor([5.0, 6.0], dtype=torch.float32),
    )


def test_gumbel_sampling_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(gumbel_mod, "HAS_TRITON", False)

    logits = torch.tensor(
        [[2.0, 1.0, 0.5], [1.5, 3.0, 0.0]],
        dtype=torch.float32,
    )
    expanded_idx_mapping = torch.tensor([1, 0], dtype=torch.int32)
    temperature = torch.tensor([2.0, 0.0], dtype=torch.float32)
    seeds = torch.tensor([123, 456], dtype=torch.int64)
    positions = torch.tensor([0, 4], dtype=torch.int64)

    temp_logits = logits.clone()
    gumbel_mod.apply_temperature(temp_logits, expanded_idx_mapping, temperature)
    expected_temp_logits = logits.clone()
    expected_temp_logits[1] /= 2.0
    torch.testing.assert_close(temp_logits, expected_temp_logits)

    processed_logits_out = torch.zeros((2, 3), dtype=torch.float32)
    sampled_first = gumbel_mod.gumbel_sample(
        logits.clone(),
        expanded_idx_mapping,
        temperature,
        seeds,
        positions,
        apply_temperature=True,
        processed_logits_out=processed_logits_out,
    )
    sampled_second = gumbel_mod.gumbel_sample(
        logits.clone(),
        expanded_idx_mapping,
        temperature,
        seeds,
        positions,
        apply_temperature=True,
    )

    assert torch.equal(sampled_first, sampled_second)
    assert sampled_first[0].item() == 0
    torch.testing.assert_close(processed_logits_out[1], logits[0])
    torch.testing.assert_close(processed_logits_out[0], logits[1] / 2.0)


def test_spec_decode_rejection_sampler_falls_back_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(spec_rejection_sampler, "HAS_TRITON", False)
    monkeypatch.setattr(
        spec_rejection_sampler,
        "_uniform_from_seed_and_pos",
        lambda seed_value, pos_value: 0.5,
    )
    monkeypatch.setattr(
        spec_rejection_sampler,
        "gumbel_sample",
        lambda logits, *_args, **_kwargs: torch.argmax(logits, dim=-1),
    )

    strict_sampled, strict_num_sampled = spec_rejection_sampler.strict_rejection_sample(
        target_sampled=torch.tensor([50, 51, 52, 60, 61], dtype=torch.int64),
        draft_sampled=torch.tensor([10, 50, 99, 20, 60], dtype=torch.int64),
        cu_num_logits=torch.tensor([0, 3, 5], dtype=torch.int32),
        num_speculative_steps=2,
    )
    assert torch.equal(strict_num_sampled, torch.tensor([2, 2], dtype=torch.int32))
    assert torch.equal(strict_sampled[0, :2], torch.tensor([50, 51], dtype=torch.int64))
    assert torch.equal(strict_sampled[1, :2], torch.tensor([60, 61], dtype=torch.int64))

    target_logits = torch.log(
        torch.tensor(
            [
                [0.05, 0.80, 0.15],
                [0.70, 0.20, 0.10],
                [0.20, 0.30, 0.50],
                [0.80, 0.10, 0.10],
                [0.10, 0.10, 0.80],
            ],
            dtype=torch.float32,
        )
    )
    draft_logits = torch.log(
        torch.tensor(
            [
                [[0.10, 0.70, 0.20], [0.10, 0.10, 0.80]],
                [[0.50, 0.25, 0.25], [0.34, 0.33, 0.33]],
            ],
            dtype=torch.float32,
        )
    )
    probabilistic_sampled, probabilistic_num_sampled = (
        spec_rejection_sampler.probabilistic_rejection_sample(
            target_logits=target_logits,
            draft_logits=draft_logits,
            draft_sampled=torch.tensor([99, 1, 2, 42, 0], dtype=torch.int64),
            cu_num_logits=torch.tensor([0, 3, 5], dtype=torch.int32),
            pos=torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64),
            idx_mapping=torch.tensor([0, 1], dtype=torch.int32),
            temperature=torch.tensor([1.0, 1.0], dtype=torch.float32),
            seed=torch.tensor([7, 9], dtype=torch.int64),
            num_speculative_steps=2,
        )
    )

    assert torch.equal(
        probabilistic_num_sampled,
        torch.tensor([2, 2], dtype=probabilistic_num_sampled.dtype),
    )
    assert torch.equal(
        probabilistic_sampled[0, :2],
        torch.tensor([1, 0], dtype=torch.int64),
    )
    assert torch.equal(
        probabilistic_sampled[1, :2],
        torch.tensor([0, 2], dtype=torch.int64),
    )


def test_spec_decode_utils_slot_mapping_falls_back_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(spec_decode_utils, "HAS_TRITON", False)

    positions = torch.tensor([0, 4], dtype=torch.int64)
    block_table = torch.tensor([[10, 11], [20, 21]], dtype=torch.int32)
    seq_lens = torch.tensor([1, 5], dtype=torch.int32)
    out_clamped_positions = torch.empty_like(positions)
    out_slot_mapping = torch.empty((4,), dtype=torch.int64)

    spec_decode_utils.eagle_step_update_slot_mapping_and_metadata(
        positions,
        block_table,
        seq_lens,
        block_size=4,
        max_model_len=5,
        out_clamped_positions=out_clamped_positions,
        out_slot_mapping=out_slot_mapping,
        input_batch_size=4,
    )

    torch.testing.assert_close(
        out_clamped_positions,
        torch.tensor([1, 0], dtype=torch.int64),
    )
    torch.testing.assert_close(
        out_slot_mapping,
        torch.tensor([41, -1, -1, -1], dtype=torch.int64),
    )
    torch.testing.assert_close(seq_lens, torch.tensor([2, 1], dtype=torch.int32))


def test_spec_decode_utils_prepare_padded_falls_back_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(spec_decode_utils, "HAS_TRITON", False)

    token_indices_to_sample = torch.empty((2,), dtype=torch.int64)
    num_rejected_tokens = torch.empty((2,), dtype=torch.int32)
    spec_decode_utils.eagle_prepare_inputs_padded(
        cu_num_draft_tokens=torch.tensor([2, 5], dtype=torch.int32),
        valid_sampled_tokens_count=torch.tensor([2, 4], dtype=torch.int32),
        query_start_loc_gpu=torch.tensor([0, 4, 9], dtype=torch.int32),
        token_indices_to_sample=token_indices_to_sample,
        num_rejected_tokens_gpu=num_rejected_tokens,
    )

    torch.testing.assert_close(
        token_indices_to_sample,
        torch.tensor([2, 8], dtype=torch.int64),
    )
    torch.testing.assert_close(
        num_rejected_tokens,
        torch.tensor([1, 0], dtype=torch.int32),
    )

    next_token_ids = torch.empty((2,), dtype=torch.int64)
    valid_sampled_tokens_count = torch.empty((2,), dtype=torch.int32)
    spec_decode_utils.eagle_prepare_next_token_padded(
        sampled_token_ids=torch.tensor(
            [[10, 11, -1], [7, 99, 5]],
            dtype=torch.int64,
        ),
        discard_request_mask=torch.tensor([False, True]),
        backup_next_token_ids=torch.tensor([42, 43], dtype=torch.int64),
        next_token_ids=next_token_ids,
        valid_sampled_tokens_count=valid_sampled_tokens_count,
        vocab_size=50,
    )

    torch.testing.assert_close(next_token_ids, torch.tensor([11, 43], dtype=torch.int64))
    torch.testing.assert_close(
        valid_sampled_tokens_count,
        torch.tensor([2, 0], dtype=torch.int32),
    )


def test_spec_decode_utils_copy_expand_falls_back_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(spec_decode_utils, "HAS_TRITON", False)

    out_input_ids = torch.full((8,), -99, dtype=torch.int32)
    out_positions = torch.full((8,), -99, dtype=torch.int64)
    out_is_rejected = torch.ones((8,), dtype=torch.bool)
    out_is_masked = torch.ones((8,), dtype=torch.bool)
    out_new_token_indices = torch.full((4,), -1, dtype=torch.int64)
    out_hidden_state_mapping = torch.full((4,), -1, dtype=torch.int32)

    spec_decode_utils.copy_and_expand_eagle_inputs(
        target_token_ids_ptr=torch.tensor([10, 11, 20, 21], dtype=torch.int32),
        target_positions_ptr=torch.tensor([0, 1, 5, 6], dtype=torch.int64),
        next_token_ids_ptr=torch.tensor([100, 200], dtype=torch.int32),
        out_input_ids_ptr=out_input_ids,
        out_positions_ptr=out_positions,
        out_is_rejected_token_mask_ptr=out_is_rejected,
        out_is_masked_token_mask_ptr=out_is_masked,
        out_new_token_indices_ptr=out_new_token_indices,
        out_hidden_state_mapping_ptr=out_hidden_state_mapping,
        query_start_loc_ptr=torch.tensor([0, 2, 4], dtype=torch.int32),
        query_end_loc_ptr=torch.tensor([1, 3], dtype=torch.int32),
        padding_token_id=-1,
        parallel_drafting_token_id=-2,
        total_input_tokens=4,
        num_padding_slots_per_request=2,
        shift_input_ids=False,
    )

    torch.testing.assert_close(
        out_input_ids,
        torch.tensor([10, 11, 100, -2, 20, 21, 200, -2], dtype=torch.int32),
    )
    torch.testing.assert_close(
        out_positions,
        torch.tensor([0, 1, 2, 3, 5, 6, 7, 8], dtype=torch.int64),
    )
    torch.testing.assert_close(out_is_rejected, torch.zeros((8,), dtype=torch.bool))
    torch.testing.assert_close(
        out_is_masked,
        torch.tensor([False, False, False, True, False, False, False, True]),
    )
    torch.testing.assert_close(
        out_new_token_indices,
        torch.tensor([2, 3, 6, 7], dtype=torch.int64),
    )
    torch.testing.assert_close(
        out_hidden_state_mapping,
        torch.tensor([-1, -1, -1, -1], dtype=torch.int32),
    )


def test_sample_rejection_expand_batch_to_tokens_falls_back_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(sample_rejection_sampler_mod, "HAS_TRITON", False)

    actual = sample_rejection_sampler_mod.expand_batch_to_tokens(
        torch.tensor([0, 2, 5], dtype=torch.int32),
        torch.tensor([2, 3, 6], dtype=torch.int32),
        num_tokens=6,
        replace_from=0,
        replace_to=7,
    )

    torch.testing.assert_close(
        actual,
        torch.tensor([7, 7, 2, 5, 5, 5], dtype=torch.int32),
    )


def test_sample_rejection_expand_batch_to_tokens_uses_precompiled_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(sample_rejection_sampler_mod, "HAS_TRITON", False)
    monkeypatch.setattr(
        sample_rejection_sampler_mod._custom_ops,
        "has_precompiled_expand_batch_to_tokens",
        lambda: True,
    )

    expected = torch.tensor([3, 3, 4], dtype=torch.int32)
    monkeypatch.setattr(
        sample_rejection_sampler_mod._custom_ops,
        "expand_batch_to_tokens_precompiled",
        lambda *args: expected,
    )

    actual = sample_rejection_sampler_mod.expand_batch_to_tokens(
        _as_fake_cuda(torch.tensor([3, 4], dtype=torch.int32)),
        torch.tensor([2, 3], dtype=torch.int32),
        num_tokens=3,
    )

    assert actual is expected


def test_sample_rejection_sample_recovered_tokens_falls_back_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(sample_rejection_sampler_mod, "HAS_TRITON", False)

    actual = sample_rejection_sampler_mod.sample_recovered_tokens(
        max_spec_len=2,
        num_draft_tokens=[2],
        cu_num_draft_tokens=torch.tensor([2], dtype=torch.int32),
        draft_token_ids=torch.tensor([1, 2], dtype=torch.int32),
        draft_probs=torch.tensor(
            [[0.2, 0.5, 0.0], [0.0, 0.1, 0.6]],
            dtype=torch.float32,
        ),
        target_probs=torch.tensor(
            [[0.2, 0.5, 0.3], [0.0, 0.4, 0.6]],
            dtype=torch.float32,
        ),
        sampling_metadata=SimpleNamespace(generators={}),
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(
        actual,
        torch.tensor([2, 1], dtype=torch.int32),
    )


def test_sample_rejection_sample_recovered_tokens_uses_precompiled_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(sample_rejection_sampler_mod, "HAS_TRITON", False)
    monkeypatch.setattr(
        sample_rejection_sampler_mod._custom_ops,
        "has_precompiled_sample_recovered_tokens",
        lambda: True,
    )
    monkeypatch.setattr(
        sample_rejection_sampler_mod.torch,
        "empty",
        _cpu_alloc_ignoring_device(torch.empty),
    )

    expected = torch.tensor([4, 5], dtype=torch.int32)
    monkeypatch.setattr(
        sample_rejection_sampler_mod._custom_ops,
        "sample_recovered_tokens_precompiled",
        lambda *args: expected,
    )

    actual = sample_rejection_sampler_mod.sample_recovered_tokens(
        max_spec_len=2,
        num_draft_tokens=[2],
        cu_num_draft_tokens=torch.tensor([2], dtype=torch.int32),
        draft_token_ids=torch.tensor([1, 2], dtype=torch.int32),
        draft_probs=torch.tensor(
            [[0.2, 0.5, 0.0], [0.0, 0.1, 0.6]],
            dtype=torch.float32,
        ),
        target_probs=torch.tensor(
            [[0.2, 0.5, 0.3], [0.0, 0.4, 0.6]],
            dtype=torch.float32,
        ),
        sampling_metadata=SimpleNamespace(generators={}),
        device=torch.device("cuda"),
    )

    assert actual is expected


def test_sample_rejection_apply_sampling_constraints_falls_back_without_triton(
    monkeypatch,
) -> None:
    captured = {}

    def _fake_apply_top_k_top_p(logits, top_k, top_p):
        captured["logits"] = logits.clone()
        captured["top_k"] = top_k.clone() if top_k is not None else None
        captured["top_p"] = top_p.clone() if top_p is not None else None
        return logits + 1.0

    monkeypatch.setattr(
        sample_rejection_sampler_mod,
        "apply_top_k_top_p",
        _fake_apply_top_k_top_p,
    )

    logits = torch.tensor(
        [[4.0, 2.0], [8.0, 4.0], [6.0, 3.0]],
        dtype=torch.float32,
    )
    actual = sample_rejection_sampler_mod.apply_sampling_constraints(
        logits,
        torch.tensor([2, 3], dtype=torch.int32),
        SimpleNamespace(
            all_greedy=False,
            temperature=torch.tensor([0.0, 2.0], dtype=torch.float32),
            top_k=torch.tensor([1, 3], dtype=torch.int32),
            top_p=torch.tensor([0.5, 0.9], dtype=torch.float32),
        ),
    )

    torch.testing.assert_close(
        captured["logits"],
        torch.tensor(
            [[4.0, 2.0], [8.0, 4.0], [3.0, 1.5]],
            dtype=torch.float32,
        ),
    )
    torch.testing.assert_close(
        captured["top_k"],
        torch.tensor([1, 1, 3], dtype=torch.int32),
    )
    torch.testing.assert_close(
        captured["top_p"],
        torch.tensor([0.5, 0.5, 0.9], dtype=torch.float32),
    )
    torch.testing.assert_close(
        actual,
        captured["logits"] + 1.0,
    )


def test_sample_rejection_sample_greedy_and_random_fall_back_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(sample_rejection_sampler_mod, "HAS_TRITON", False)

    greedy_output = sample_rejection_sampler_mod.rejection_sample(
        draft_token_ids=torch.tensor([1, 2, 3], dtype=torch.int32),
        num_draft_tokens=[2, 1],
        max_spec_len=2,
        cu_num_draft_tokens=torch.tensor([2, 3], dtype=torch.int32),
        draft_probs=None,
        target_logits=torch.log(
            torch.tensor(
                [
                    [0.1, 0.8, 0.1, 0.0],
                    [0.1, 0.1, 0.8, 0.0],
                    [0.7, 0.1, 0.1, 0.1],
                ],
                dtype=torch.float32,
            ).clamp_min(1e-6)
        ),
        bonus_token_ids=torch.tensor([[9], [8]], dtype=torch.int32),
        sampling_metadata=SimpleNamespace(
            all_greedy=True,
            all_random=False,
            temperature=torch.tensor([0.0, 0.0], dtype=torch.float32),
            generators={},
        ),
    )

    torch.testing.assert_close(
        greedy_output,
        torch.tensor(
            [[1, 2, 9], [0, -1, -1]],
            dtype=torch.int32,
        ),
    )

    monkeypatch.setattr(
        sample_rejection_sampler_mod,
        "generate_uniform_probs",
        lambda *args, **kwargs: torch.tensor([0.5, 0.9], dtype=torch.float64),
    )
    monkeypatch.setattr(
        sample_rejection_sampler_mod,
        "sample_recovered_tokens",
        lambda *args, **kwargs: torch.tensor([3, 1], dtype=torch.int32),
    )

    random_output = sample_rejection_sampler_mod.rejection_sample(
        draft_token_ids=torch.tensor([1, 2], dtype=torch.int32),
        num_draft_tokens=[2],
        max_spec_len=2,
        cu_num_draft_tokens=torch.tensor([2], dtype=torch.int32),
        draft_probs=torch.tensor(
            [[0.1, 0.8, 0.1, 0.0], [0.1, 0.1, 0.7, 0.1]],
            dtype=torch.float32,
        ),
        target_logits=torch.log(
            torch.tensor(
                [[0.05, 0.9, 0.03, 0.02], [0.4, 0.4, 0.1, 0.1]],
                dtype=torch.float32,
            ).clamp_min(1e-6)
        ),
        bonus_token_ids=torch.tensor([[9]], dtype=torch.int32),
        sampling_metadata=SimpleNamespace(
            all_greedy=False,
            all_random=True,
            temperature=torch.tensor([1.0], dtype=torch.float32),
            generators={},
        ),
    )

    torch.testing.assert_close(
        random_output,
        torch.tensor([[1, 1, -1]], dtype=torch.int32),
    )


def test_sample_rejection_greedy_precompiled_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(sample_rejection_sampler_mod, "HAS_TRITON", False)
    monkeypatch.setattr(
        sample_rejection_sampler_mod._custom_ops,
        "has_precompiled_rejection_greedy_sample",
        lambda: True,
    )
    monkeypatch.setattr(
        sample_rejection_sampler_mod.torch,
        "full",
        _cpu_alloc_ignoring_device(torch.full),
    )

    expected = torch.tensor([[7, 8, 9]], dtype=torch.int32)

    def _fake_precompiled(output_token_ids, *args):
        output_token_ids.copy_(expected)

    monkeypatch.setattr(
        sample_rejection_sampler_mod._custom_ops,
        "rejection_greedy_sample_precompiled",
        _fake_precompiled,
    )

    actual = sample_rejection_sampler_mod.rejection_sample(
        draft_token_ids=torch.tensor([1, 2], dtype=torch.int32),
        num_draft_tokens=[2],
        max_spec_len=2,
        cu_num_draft_tokens=torch.tensor([2], dtype=torch.int32),
        draft_probs=None,
        target_logits=_as_fake_cuda_device(
            torch.log(
                torch.tensor(
                    [[0.1, 0.8, 0.1], [0.1, 0.2, 0.7]],
                    dtype=torch.float32,
                ).clamp_min(1e-6)
            )
        ),
        bonus_token_ids=torch.tensor([[9]], dtype=torch.int32),
        sampling_metadata=SimpleNamespace(
            all_greedy=True,
            all_random=False,
            temperature=torch.tensor([0.0], dtype=torch.float32),
            generators={},
        ),
    )

    torch.testing.assert_close(actual, expected)


def test_sample_rejection_random_precompiled_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(sample_rejection_sampler_mod, "HAS_TRITON", False)
    monkeypatch.setattr(
        sample_rejection_sampler_mod._custom_ops,
        "has_precompiled_rejection_random_sample",
        lambda: True,
    )
    monkeypatch.setattr(
        sample_rejection_sampler_mod.torch,
        "full",
        _cpu_alloc_ignoring_device(torch.full),
    )
    monkeypatch.setattr(
        sample_rejection_sampler_mod,
        "generate_uniform_probs",
        lambda *args, **kwargs: torch.tensor([0.2, 0.4], dtype=torch.float64),
    )
    monkeypatch.setattr(
        sample_rejection_sampler_mod,
        "sample_recovered_tokens",
        lambda *args, **kwargs: torch.tensor([3, 1], dtype=torch.int32),
    )

    expected = torch.tensor([[5, 6, -1]], dtype=torch.int32)

    def _fake_precompiled(output_token_ids, *args):
        output_token_ids.copy_(expected)

    monkeypatch.setattr(
        sample_rejection_sampler_mod._custom_ops,
        "rejection_random_sample_precompiled",
        _fake_precompiled,
    )

    actual = sample_rejection_sampler_mod.rejection_sample(
        draft_token_ids=torch.tensor([1, 2], dtype=torch.int32),
        num_draft_tokens=[2],
        max_spec_len=2,
        cu_num_draft_tokens=torch.tensor([2], dtype=torch.int32),
        draft_probs=torch.tensor(
            [[0.1, 0.8, 0.1], [0.1, 0.2, 0.7]],
            dtype=torch.float32,
        ),
        target_logits=_as_fake_cuda_device(
            torch.log(
                torch.tensor(
                    [[0.1, 0.7, 0.2], [0.5, 0.2, 0.3]],
                    dtype=torch.float32,
                ).clamp_min(1e-6)
            )
        ),
        bonus_token_ids=torch.tensor([[9]], dtype=torch.int32),
        sampling_metadata=SimpleNamespace(
            all_greedy=False,
            all_random=True,
            temperature=torch.tensor([1.0], dtype=torch.float32),
            generators={},
        ),
    )

    torch.testing.assert_close(actual, expected)


def test_topk_topp_sampler_apply_top_k_top_p_falls_back_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(topk_topp_sampler_mod, "HAS_TRITON", False)
    monkeypatch.setattr(
        topk_topp_sampler_mod.ops,
        "has_precompiled_apply_top_k_top_p",
        lambda: False,
    )

    logits = torch.tensor([[1.0, 3.0, 2.0]], dtype=torch.float32)
    actual = topk_topp_sampler_mod.apply_top_k_top_p(
        logits.clone(),
        torch.tensor([1], dtype=torch.int32),
        None,
    )

    torch.testing.assert_close(
        actual,
        torch.tensor([[-float("inf"), 3.0, -float("inf")]], dtype=torch.float32),
    )


def test_topk_topp_sampler_apply_top_k_top_p_uses_precompiled_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(topk_topp_sampler_mod, "HAS_TRITON", False)
    monkeypatch.setattr(
        topk_topp_sampler_mod.ops,
        "has_precompiled_apply_top_k_top_p",
        lambda: True,
    )

    logits = _as_fake_cuda(torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32))
    called = {"value": False}

    def _fake_precompiled(logits_tensor, k, p):
        called["value"] = True
        logits_tensor.fill_(-7.0)

    monkeypatch.setattr(
        topk_topp_sampler_mod.ops,
        "apply_top_k_top_p_precompiled",
        _fake_precompiled,
    )

    actual = topk_topp_sampler_mod.apply_top_k_top_p(
        logits,
        torch.tensor([1], dtype=torch.int32),
        None,
    )

    assert called["value"] is True
    assert actual is logits
    torch.testing.assert_close(actual, torch.full_like(actual, -7.0))


def test_sampling_helpers_fall_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(min_p_mod, "HAS_TRITON", False)
    monkeypatch.setattr(logits_metrics, "HAS_TRITON", False)
    monkeypatch.setattr(logprob_mod, "HAS_TRITON", False)

    min_p_logits = torch.tensor(
        [[4.0, 3.0, 0.0], [5.0, 1.0, 4.5]],
        dtype=torch.float32,
    )
    min_p_mod.apply_min_p(
        min_p_logits,
        expanded_idx_mapping=torch.tensor([1, 0], dtype=torch.int32),
        min_p=torch.tensor([0.5, 0.2], dtype=torch.float32),
    )
    expected_min_p = torch.tensor(
        [[4.0, 3.0, float("-inf")], [5.0, float("-inf"), 4.5]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(min_p_logits, expected_min_p)

    nan_counts = logits_metrics.get_num_nans(
        torch.tensor(
            [[1.0, float("nan"), float("nan")], [float("nan"), 0.0, 2.0]],
            dtype=torch.float32,
        )
    )
    assert torch.equal(nan_counts, torch.tensor([2, 1], dtype=torch.int32))

    logits = torch.tensor(
        [[2.0, 0.5, -1.0], [0.1, 1.3, 0.8]],
        dtype=torch.float32,
    )
    token_ids = torch.tensor([[0, 2], [1, 0]], dtype=torch.int64)
    expected_token_logprobs = torch.log_softmax(logits, dim=-1).gather(1, token_ids)
    torch.testing.assert_close(
        logprob_mod.compute_token_logprobs(logits, token_ids),
        expected_token_logprobs,
    )

    topk = logprob_mod.compute_topk_logprobs(
        logits=logits,
        num_logprobs=2,
        sampled_token_ids=torch.tensor([0, 1], dtype=torch.int64),
        cu_num_logits=[0, 1, 2],
    )
    expected_ids = torch.cat(
        (
            torch.tensor([[0], [1]], dtype=torch.int64),
            torch.topk(logits, 2, dim=-1).indices,
        ),
        dim=1,
    )
    expected_logprobs = torch.log_softmax(logits, dim=-1).gather(1, expected_ids)
    expected_ranks = torch.tensor([1, 1], dtype=torch.int64)

    assert torch.equal(topk.logprob_token_ids, expected_ids)
    torch.testing.assert_close(topk.logprobs, expected_logprobs)
    assert torch.equal(topk.selected_token_ranks, expected_ranks)
    assert topk.cu_num_generated_tokens == [0, 1, 2]


def test_penalties_logit_bias_and_bad_words_fall_back_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(penalties_mod, "HAS_TRITON", False)
    monkeypatch.setattr(logit_bias_mod, "HAS_TRITON", False)
    monkeypatch.setattr(bad_words_mod, "HAS_TRITON", False)

    prompt_bin_mask = torch.zeros((3, 1), dtype=torch.int32)
    output_bin_counts = torch.zeros((3, 8), dtype=torch.int32)
    penalties_mod.bincount(
        expanded_idx_mapping=torch.tensor([2, 0], dtype=torch.int32),
        all_token_ids=torch.tensor(
            [
                [2, 2, 3, 2, 0],
                [0, 0, 0, 0, 0],
                [1, 3, 4, 1, 4],
            ],
            dtype=torch.int32,
        ),
        prompt_len=torch.tensor([1, 0, 2], dtype=torch.int32),
        prefill_len=torch.tensor([4, 0, 5], dtype=torch.int32),
        prompt_bin_mask=prompt_bin_mask,
        output_bin_counts=output_bin_counts,
        max_prefill_len=5,
    )
    assert prompt_bin_mask[2, 0].item() == ((1 << 1) | (1 << 3))
    assert prompt_bin_mask[0, 0].item() == (1 << 2)
    assert torch.equal(
        output_bin_counts[2, :5],
        torch.tensor([0, 1, 0, 0, 2], dtype=torch.int32),
    )
    assert torch.equal(
        output_bin_counts[0, :4],
        torch.tensor([0, 0, 2, 1], dtype=torch.int32),
    )

    penalty_logits = torch.tensor([[0.5, -0.5, 1.0, 2.0]], dtype=torch.float32)
    penalties_mod.apply_penalties(
        logits=penalty_logits,
        expanded_idx_mapping=torch.tensor([0], dtype=torch.int32),
        token_ids=torch.tensor([7], dtype=torch.int32),
        expanded_local_pos=torch.tensor([0], dtype=torch.int32),
        repetition_penalty=torch.tensor([2.0], dtype=torch.float32),
        frequency_penalty=torch.tensor([0.5], dtype=torch.float32),
        presence_penalty=torch.tensor([1.0], dtype=torch.float32),
        prompt_bin_mask=torch.tensor([[1 << 1]], dtype=torch.int32),
        output_bin_counts=torch.tensor([[0, 0, 2, 1]], dtype=torch.int32),
        num_speculative_tokens=1,
    )
    torch.testing.assert_close(
        penalty_logits,
        torch.tensor([[0.5, -1.0, -1.5, -0.5]], dtype=torch.float32),
    )

    bias_logits = torch.arange(6, dtype=torch.float32).reshape(1, 6)
    logit_bias_mod.apply_logit_bias(
        logits=bias_logits,
        expanded_idx_mapping=torch.tensor([0], dtype=torch.int32),
        pos=torch.tensor([1], dtype=torch.int64),
        num_allowed_token_ids=torch.tensor([2], dtype=torch.int32),
        allowed_token_ids=torch.tensor([[1, 4]], dtype=torch.int32),
        num_logit_bias=torch.tensor([2], dtype=torch.int32),
        logit_bias_token_ids=torch.tensor([[4, 5]], dtype=torch.int32),
        logit_bias=torch.tensor([[0.5, -1.0]], dtype=torch.float32),
        min_lens=torch.tensor([3], dtype=torch.int32),
        num_stop_token_ids=torch.tensor([2], dtype=torch.int32),
        stop_token_ids=torch.tensor([[1, 2]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        bias_logits,
        torch.tensor([[-float("inf"), -float("inf"), -float("inf"), -float("inf"), 4.5, -float("inf")]], dtype=torch.float32),
    )

    bad_word_logits = torch.zeros((1, 16), dtype=torch.float32)
    bad_words_mod.apply_bad_words(
        logits=bad_word_logits,
        expanded_idx_mapping=torch.tensor([0], dtype=torch.int32),
        bad_word_token_ids=torch.tensor([[8, 9]], dtype=torch.int32),
        bad_word_offsets=torch.tensor([[0, 2]], dtype=torch.int32),
        num_bad_words=torch.tensor([1], dtype=torch.int32),
        all_token_ids=torch.tensor([[8, 0, 0]], dtype=torch.int32),
        prompt_len=torch.tensor([0], dtype=torch.int32),
        total_len=torch.tensor([1], dtype=torch.int32),
        input_ids=torch.tensor([42], dtype=torch.int32),
        expanded_local_pos=torch.tensor([0], dtype=torch.int32),
        max_num_bad_words=1,
    )
    assert bad_word_logits[0, 9].item() == float("-inf")


def test_prompt_logprobs_token_ids_fall_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(prompt_logprob_mod, "HAS_TRITON", False)

    token_ids = prompt_logprob_mod.get_prompt_logprobs_token_ids(
        num_tokens=5,
        query_start_loc=torch.tensor([0, 2, 5], dtype=torch.int32),
        idx_mapping=torch.tensor([1, 0], dtype=torch.int32),
        num_computed_tokens=torch.tensor([1, 3], dtype=torch.int32),
        all_token_ids=torch.tensor(
            [
                [10, 11, 12, 13, 14, 15],
                [20, 21, 22, 23, 24, 25],
            ],
            dtype=torch.int64,
        ),
    )

    assert torch.equal(
        token_ids,
        torch.tensor([24, 25, 12, 13, 14], dtype=torch.int64),
    )


def test_structured_outputs_apply_grammar_bitmask_falls_back_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(structured_outputs_mod, "HAS_TRITON", False)

    worker = structured_outputs_mod.StructuredOutputsWorker(
        max_num_logits=4,
        vocab_size=10,
        device=torch.device("cpu"),
    )
    logits = torch.zeros((3, 10), dtype=torch.float32)
    input_batch = SimpleNamespace(
        req_ids=["req0", "req1"],
        cu_num_logits_np=np.array([0, 2, 3], dtype=np.int32),
    )
    grammar_bitmask = np.array(
        [
            [(1 << 0) | (1 << 3)],
            [(1 << 1)],
            [(1 << 2) | (1 << 9)],
        ],
        dtype=np.int32,
    )

    worker.apply_grammar_bitmask(
        logits=logits,
        input_batch=input_batch,
        grammar_req_ids=["req0", "req1"],
        grammar_bitmask=grammar_bitmask,
    )

    assert logits[0, 0].item() == 0.0
    assert logits[0, 3].item() == 0.0
    assert logits[0, 1].item() == float("-inf")
    assert logits[1, 1].item() == 0.0
    assert logits[1, 0].item() == float("-inf")
    assert logits[2, 2].item() == 0.0
    assert logits[2, 9].item() == 0.0
    assert logits[2, 4].item() == float("-inf")


def test_staged_write_tensor_apply_write_falls_back_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(buffer_utils_mod, "HAS_TRITON", False)

    class _FakeUvaBufferPool:
        def __init__(self, _size, dtype, max_concurrency=2):
            self.dtype = dtype

        def copy_to_uva(self, x):
            if isinstance(x, torch.Tensor):
                return x.to(dtype=self.dtype)
            return torch.tensor(x, dtype=self.dtype)

    monkeypatch.setattr(buffer_utils_mod, "UvaBufferPool", _FakeUvaBufferPool)
    monkeypatch.setattr(
        buffer_utils_mod,
        "async_tensor_h2d",
        lambda data, dtype, device, pin_memory=True: torch.tensor(
            data, dtype=dtype, device=device
        ),
    )

    staged = buffer_utils_mod.StagedWriteTensor(
        (2, 6),
        dtype=torch.int32,
        device=torch.device("cpu"),
    )
    staged.stage_write(0, 1, [5, 6])
    staged.stage_write(1, 0, [7, 8, 9])
    staged.apply_write()

    assert torch.equal(
        staged.gpu,
        torch.tensor(
            [[0, 5, 6, 0, 0, 0], [7, 8, 9, 0, 0, 0]],
            dtype=torch.int32,
        ),
    )


def test_prepare_dcp_local_seq_lens_falls_back_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(cp_utils_mod, "HAS_TRITON", False)

    out = torch.full((5,), -1, dtype=torch.int32)
    cp_utils_mod.prepare_dcp_local_seq_lens(
        dcp_local_seq_lens=out,
        seq_lens=torch.tensor([5, 8, 1], dtype=torch.int32),
        num_reqs=3,
        dcp_size=2,
        dcp_rank=1,
        cp_interleave=2,
    )

    assert torch.equal(out, torch.tensor([2, 4, 0, 0, 0], dtype=torch.int32))


def test_prepare_rope_positions_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(rope_state_mod, "HAS_TRITON", False)

    fake_state = SimpleNamespace(
        num_dims=3,
        max_model_len=8,
        positions=torch.zeros((3, 8), dtype=torch.int64),
        prefill_positions=SimpleNamespace(
            gpu=torch.tensor(
                [
                    [100, 101, 102, 103, 104, 105, 106, 107],
                    [110, 111, 112, 113, 114, 115, 116, 117],
                    [120, 121, 122, 123, 124, 125, 126, 127],
                    [200, 201, 202, 203, 204, 205, 206, 207],
                    [210, 211, 212, 213, 214, 215, 216, 217],
                    [220, 221, 222, 223, 224, 225, 226, 227],
                ],
                dtype=torch.int64,
            )
        ),
        prefill_delta=SimpleNamespace(gpu=torch.tensor([7, 0], dtype=torch.int32)),
    )

    rope_state_mod.RopeState.prepare_positions(
        fake_state,
        idx_mapping=torch.tensor([1, 0], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2, 5], dtype=torch.int32),
        prefill_lens=torch.tensor([4, 4], dtype=torch.int32),
        num_computed_tokens=torch.tensor([4, 1], dtype=torch.int32),
    )

    assert torch.equal(
        fake_state.positions[:, :5],
        torch.tensor(
            [
                [201, 202, 11, 12, 13],
                [211, 212, 11, 12, 13],
                [221, 222, 11, 12, 13],
            ],
            dtype=torch.int64,
        ),
    )


def test_merge_attn_states_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(triton_merge_attn_states_mod, "HAS_TRITON", False)

    prefix_output = torch.tensor(
        [
            [[1.0, 1.0], [10.0, 10.0]],
            [[2.0, 2.0], [20.0, 20.0]],
        ],
        dtype=torch.float32,
    )
    suffix_output = torch.tensor(
        [
            [[3.0, 3.0], [30.0, 30.0]],
            [[4.0, 4.0], [40.0, 40.0]],
        ],
        dtype=torch.float32,
    )
    prefix_lse = torch.tensor(
        [
            [0.0, float("inf")],
            [torch.log(torch.tensor(2.0)).item(), 0.0],
        ],
        dtype=torch.float32,
    )
    suffix_lse = torch.tensor(
        [
            [torch.log(torch.tensor(3.0)).item(), 0.0],
            [torch.log(torch.tensor(2.0)).item(), torch.log(torch.tensor(4.0)).item()],
        ],
        dtype=torch.float32,
    )
    output = torch.empty_like(prefix_output)
    output_lse = torch.empty_like(prefix_lse)

    triton_merge_attn_states_mod.merge_attn_states(
        output=output,
        prefix_output=prefix_output,
        prefix_lse=prefix_lse,
        suffix_output=suffix_output,
        suffix_lse=suffix_lse,
        output_lse=output_lse,
    )

    expected_output = torch.tensor(
        [
            [[2.5, 2.5], [20.0, 20.0]],
            [[4.0, 4.0], [36.0, 36.0]],
        ],
        dtype=torch.float32,
    )
    expected_lse = torch.tensor(
        [
            [torch.log(torch.tensor(4.0)).item(), 0.0],
            [torch.log(torch.tensor(4.0)).item(), torch.log(torch.tensor(5.0)).item()],
        ],
        dtype=torch.float32,
    )

    torch.testing.assert_close(output, expected_output)
    torch.testing.assert_close(output_lse, expected_lse)


def test_merge_attn_states_prefers_precompiled_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(triton_merge_attn_states_mod, "HAS_TRITON", False)

    calls: dict[str, object] = {}

    def _fake_merge(
        output: torch.Tensor,
        prefix_output: torch.Tensor,
        prefix_lse: torch.Tensor,
        suffix_output: torch.Tensor,
        suffix_lse: torch.Tensor,
        output_lse: torch.Tensor | None = None,
    ) -> None:
        calls["shape"] = tuple(output.shape)
        output.fill_(42.0)
        if output_lse is not None:
            output_lse.fill_(-1.0)

    monkeypatch.setattr(
        triton_merge_attn_states_mod,
        "_supports_precompiled_merge_attn_states",
        lambda **_: True,
    )
    monkeypatch.setattr(
        triton_merge_attn_states_mod.ops,
        "merge_attn_states",
        _fake_merge,
    )

    output = torch.zeros((2, 1, 8), dtype=torch.float16)
    output_lse = torch.zeros((1, 2), dtype=torch.float32)
    triton_merge_attn_states_mod.merge_attn_states(
        output=output,
        prefix_output=torch.zeros_like(output),
        prefix_lse=torch.zeros_like(output_lse),
        suffix_output=torch.zeros_like(output),
        suffix_lse=torch.zeros_like(output_lse),
        output_lse=output_lse,
    )

    assert calls == {"shape": (2, 1, 8)}
    torch.testing.assert_close(output, torch.full_like(output, 42.0))
    torch.testing.assert_close(output_lse, torch.full_like(output_lse, -1.0))


def test_dcp_lse_combine_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(dcp_alltoall_mod, "HAS_TRITON", False)
    monkeypatch.setattr(
        dcp_alltoall_mod,
        "_supports_precompiled_dcp_lse_combine",
        lambda **_: False,
    )

    recv_output = torch.tensor(
        [
            [[[1.0, 2.0]], [[3.0, 4.0]]],
            [[[5.0, 6.0]], [[7.0, 8.0]]],
        ],
        dtype=torch.float32,
    )
    recv_lse = torch.tensor(
        [
            [[0.0], [float("inf")]],
            [[torch.log(torch.tensor(3.0)).item()], [0.0]],
        ],
        dtype=torch.float32,
    )

    out, out_lse = dcp_alltoall_mod.dcp_lse_combine_triton(
        recv_output,
        recv_lse,
        return_lse=True,
    )

    expected_out = torch.tensor(
        [
            [[4.0, 5.0]],
            [[7.0, 8.0]],
        ],
        dtype=torch.float32,
    )
    expected_lse = torch.tensor(
        [
            [torch.log(torch.tensor(4.0)).item()],
            [0.0],
        ],
        dtype=torch.float32,
    )

    torch.testing.assert_close(out, expected_out)
    torch.testing.assert_close(out_lse, expected_lse)


def test_dcp_lse_combine_prefers_precompiled_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(dcp_alltoall_mod, "HAS_TRITON", False)

    calls: dict[str, object] = {}

    def _fake_precompiled(
        recv_output: torch.Tensor,
        recv_lse: torch.Tensor,
        return_lse: bool = False,
        is_lse_base_on_e: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        calls["shape"] = tuple(recv_output.shape)
        calls["return_lse"] = return_lse
        calls["is_lse_base_on_e"] = is_lse_base_on_e
        return (
            torch.full((recv_output.shape[1], recv_output.shape[2], recv_output.shape[3]), 42.0),
            torch.full((recv_lse.shape[1], recv_lse.shape[2]), -1.0),
        )

    monkeypatch.setattr(
        dcp_alltoall_mod,
        "_supports_precompiled_dcp_lse_combine",
        lambda **_: True,
    )
    monkeypatch.setattr(
        dcp_alltoall_mod.ops,
        "dcp_lse_combine_precompiled",
        _fake_precompiled,
    )
    monkeypatch.setattr(
        dcp_alltoall_mod,
        "_lse_weighted_combine",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("torch fallback should not run")
        ),
    )

    recv_output = torch.zeros((2, 3, 4, 5), dtype=torch.float32)
    recv_lse = torch.zeros((2, 3, 4), dtype=torch.float32)

    out, out_lse = dcp_alltoall_mod.dcp_lse_combine_triton(
        recv_output,
        recv_lse,
        return_lse=True,
    )

    assert calls == {
        "shape": (2, 3, 4, 5),
        "return_lse": True,
        "is_lse_base_on_e": True,
    }
    torch.testing.assert_close(out, torch.full((3, 4, 5), 42.0))
    torch.testing.assert_close(out_lse, torch.full((3, 4), -1.0))


@pytest.mark.parametrize(
    ("num_bits", "is_a_8bit", "use_perm"),
    [
        (4, False, True),
        (8, False, True),
        (4, True, False),
        (8, True, False),
    ],
)
def test_gptq_marlin_repack_torch_fallback_matches_reference(
    monkeypatch,
    num_bits: int,
    is_a_8bit: bool,
    use_perm: bool,
) -> None:
    monkeypatch.setattr(custom_ops, "_get_torch_op", lambda *_: None)

    size_k = 64 if is_a_8bit else 32
    size_n = 64 if is_a_8bit else 128
    logical_qweight = torch.randint(
        0,
        1 << num_bits,
        (size_k, size_n),
        dtype=torch.int32,
    )
    perm = (
        torch.randperm(size_k, dtype=torch.int32)
        if use_perm
        else torch.empty(0, dtype=torch.int32)
    )
    packed_qweight = _pack_rows_for_test(logical_qweight, num_bits)
    expected = _reference_marlin_repack_from_logical(
        logical_qweight.index_select(0, perm.to(torch.long))
        if use_perm
        else logical_qweight,
        num_bits,
        is_a_8bit,
    )

    actual = custom_ops.gptq_marlin_repack(
        packed_qweight,
        perm,
        size_k=size_k,
        size_n=size_n,
        num_bits=num_bits,
        is_a_8bit=is_a_8bit,
    )

    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    ("num_bits", "is_a_8bit"),
    [
        (4, False),
        (8, False),
        (4, True),
        (8, True),
    ],
)
def test_awq_marlin_repack_torch_fallback_matches_reference(
    monkeypatch,
    num_bits: int,
    is_a_8bit: bool,
) -> None:
    monkeypatch.setattr(custom_ops, "_get_torch_op", lambda *_: None)

    size_k = 64 if is_a_8bit else 32
    size_n = 64 if is_a_8bit else 128
    logical_qweight = torch.randint(
        0,
        1 << num_bits,
        (size_k, size_n),
        dtype=torch.int32,
    )
    interleave = [0, 2, 4, 6, 1, 3, 5, 7] if num_bits == 4 else [0, 2, 1, 3]
    awq_layout_qweight = logical_qweight.reshape(-1, len(interleave)).index_select(
        1,
        torch.tensor(interleave, dtype=torch.long),
    ).reshape(size_k, size_n)
    packed_qweight = _pack_cols_for_test(awq_layout_qweight, num_bits)
    expected = _reference_marlin_repack_from_logical(
        logical_qweight,
        num_bits,
        is_a_8bit,
    )

    actual = custom_ops.awq_marlin_repack(
        packed_qweight,
        size_k=size_k,
        size_n=size_n,
        num_bits=num_bits,
        is_a_8bit=is_a_8bit,
    )

    assert torch.equal(actual, expected)


def test_fused_moe_activation_silu_falls_back_without_custom_op(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fused_moe_activation, "_get_moe_custom_op", lambda *_: None)

    input_tensor = torch.tensor(
        [
            [1.0, -1.0, 2.0, 3.0],
            [0.5, 0.25, -4.0, 6.0],
        ],
        dtype=torch.float32,
    )
    output = torch.empty((2, 2), dtype=torch.float32)

    fused_moe_activation.apply_moe_activation(
        fused_moe_activation.MoEActivation.SILU,
        output,
        input_tensor,
    )

    expected = torch.nn.functional.silu(input_tensor[:, :2]) * input_tensor[:, 2:]
    torch.testing.assert_close(output, expected)


def test_fused_moe_package_exports_shared_symbols_without_triton() -> None:
    if triton_importing.HAS_TRITON:
        pytest.skip("This regression only applies when Triton runtime is unavailable.")

    reloaded_pkg = importlib.reload(fused_moe_pkg)

    assert reloaded_pkg.fused_topk is importlib.import_module(
        "cfie.model_executor.layers.fused_moe.router.fused_topk_router"
    ).fused_topk
    assert reloaded_pkg.fused_experts is fused_moe_mod.fused_experts
    assert reloaded_pkg.TritonExperts is fused_moe_mod.TritonExperts
    assert reloaded_pkg.TritonWNA16Experts is fused_moe_mod.TritonWNA16Experts
    assert (
        reloaded_pkg.BatchedTritonExperts is
        fused_batched_moe_mod.BatchedTritonExperts
    )


def test_fused_moe_triton_experts_support_shared_reference_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fused_moe_mod, "HAS_TRITON", False)
    monkeypatch.setattr(fused_moe_mod.current_platform, "is_cuda_alike", lambda: True)
    monkeypatch.setattr(fused_moe_mod.current_platform, "is_xpu", lambda: False)

    moe_config = SimpleNamespace(
        is_act_and_mul=True,
        activation=fused_moe_activation.MoEActivation.SILU,
        moe_parallel_config=SimpleNamespace(use_fi_all2allv_kernels=False),
        routing_method=None,
        router_logits_dtype=None,
        hidden_dim=128,
    )

    supported, reason = fused_moe_mod.TritonExperts.is_supported_config(
        fused_moe_mod.TritonExperts,
        moe_config,
        None,
        None,
        fused_moe_mod.TritonExperts.activation_format(),
    )

    assert supported
    assert reason is None


def test_invoke_fused_moe_triton_kernel_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(fused_moe_mod, "HAS_TRITON", False)

    def _raise_if_triton_path_runs(*_args, **_kwargs):
        raise AssertionError("fused_moe Triton kernel should not run")

    monkeypatch.setattr(fused_moe_mod, "fused_moe_kernel", _raise_if_triton_path_runs)

    A = torch.tensor(
        [
            [1.0, 2.0],
            [-1.5, 0.5],
        ],
        dtype=torch.float32,
    )
    B = torch.tensor(
        [
            [[1.0, -1.0], [0.5, 2.0], [-2.0, 1.0]],
            [[0.25, 1.0], [-1.5, 0.5], [2.0, -0.5]],
        ],
        dtype=torch.float32,
    )
    C = torch.full((2, 2, 3), -123.0, dtype=torch.float32)
    topk_weights = torch.tensor(
        [
            [0.5, 1.5],
            [2.0, 0.25],
        ],
        dtype=torch.float32,
    )
    expert_ids = torch.tensor(
        [
            [0, 1],
            [1, 0],
        ],
        dtype=torch.int32,
    )

    fused_moe_mod.invoke_fused_moe_triton_kernel(
        A=A,
        B=B,
        C=C,
        A_scale=None,
        B_scale=None,
        topk_weights=topk_weights,
        sorted_token_ids=None,
        expert_ids=expert_ids,
        num_tokens_post_padded=torch.tensor([0], dtype=torch.int32),
        mul_routed_weight=True,
        top_k=2,
        config={"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 16},
        compute_type=None,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        block_shape=None,
        B_bias=None,
    )

    expected = torch.zeros_like(C)
    expected[0, 0] = (A[0] @ B[0].transpose(0, 1)) * topk_weights[0, 0]
    expected[0, 1] = (A[0] @ B[1].transpose(0, 1)) * topk_weights[0, 1]
    expected[1, 0] = (A[1] @ B[1].transpose(0, 1)) * topk_weights[1, 0]
    expected[1, 1] = (A[1] @ B[0].transpose(0, 1)) * topk_weights[1, 1]

    torch.testing.assert_close(C, expected)


def test_invoke_fused_moe_triton_kernel_prefers_shared_cuda_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fused_moe_mod, "HAS_TRITON", False)

    calls: dict[str, object] = {}

    def _fake_grouped_cuda(**kwargs):
        calls["shape_A"] = tuple(kwargs["A"].shape)
        calls["shape_B"] = tuple(kwargs["B"].shape)
        calls["shape_C"] = tuple(kwargs["C"].shape)
        kwargs["C"].fill_(13.0)
        return True

    monkeypatch.setattr(
        fused_moe_mod,
        "_try_precompiled_fused_moe_kernel",
        _fake_grouped_cuda,
    )
    monkeypatch.setattr(
        fused_moe_mod,
        "_invoke_fused_moe_reference",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("torch fallback should not run")
        ),
    )

    A = torch.randn(2, 4, dtype=torch.float16)
    B = torch.randn(3, 5, 4, dtype=torch.float16)
    C = torch.zeros((2, 2, 5), dtype=torch.float16)
    expert_ids = torch.tensor([[0, 1], [2, 0]], dtype=torch.int32)

    fused_moe_mod.invoke_fused_moe_triton_kernel(
        A=A,
        B=B,
        C=C,
        A_scale=None,
        B_scale=None,
        topk_weights=torch.ones((2, 2), dtype=torch.float32),
        sorted_token_ids=None,
        expert_ids=expert_ids,
        num_tokens_post_padded=torch.tensor([0], dtype=torch.int32),
        mul_routed_weight=False,
        top_k=2,
        config={"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 16},
        compute_type=None,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        block_shape=None,
        B_bias=None,
    )

    assert calls == {
        "shape_A": (2, 4),
        "shape_B": (3, 5, 4),
        "shape_C": (2, 2, 5),
    }
    assert torch.all(C == 13.0).item()


@pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn"),
    reason="FP8 dtype unavailable",
)
def test_invoke_fused_moe_triton_kernel_fp8_falls_back_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fused_moe_mod, "HAS_TRITON", False)

    def _raise_if_triton_path_runs(*_args, **_kwargs):
        raise AssertionError("fused_moe FP8 Triton kernel should not run")

    monkeypatch.setattr(fused_moe_mod, "fused_moe_kernel", _raise_if_triton_path_runs)

    fp8 = torch.float8_e4m3fn
    A = torch.tensor(
        [
            [2.0, -1.0],
            [1.5, 0.5],
        ],
        dtype=fp8,
    )
    B = torch.tensor(
        [
            [[1.0, 2.0], [-1.0, 0.5]],
            [[0.25, -2.0], [1.5, 1.0]],
        ],
        dtype=fp8,
    )
    C = torch.zeros((2, 1, 2), dtype=torch.float32)
    A_scale = torch.tensor([[0.5], [2.0]], dtype=torch.float32)
    B_scale = torch.tensor(
        [
            [[0.25]],
            [[2.0]],
        ],
        dtype=torch.float32,
    )
    expert_ids = torch.tensor([[0], [1]], dtype=torch.int32)

    fused_moe_mod.invoke_fused_moe_triton_kernel(
        A=A,
        B=B,
        C=C,
        A_scale=A_scale,
        B_scale=B_scale,
        topk_weights=torch.ones((2, 1), dtype=torch.float32),
        sorted_token_ids=None,
        expert_ids=expert_ids,
        num_tokens_post_padded=torch.tensor([0], dtype=torch.int32),
        mul_routed_weight=False,
        top_k=1,
        config={"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 16},
        compute_type=None,
        use_fp8_w8a8=True,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=True,
        block_shape=None,
        B_bias=None,
    )

    expected = torch.zeros_like(C)
    expected[0, 0] = (A[0].to(torch.float32) * A_scale[0]) @ (
        B[0].to(torch.float32) * B_scale[0]
    ).transpose(0, 1)
    expected[1, 0] = (A[1].to(torch.float32) * A_scale[1]) @ (
        B[1].to(torch.float32) * B_scale[1]
    ).transpose(0, 1)

    torch.testing.assert_close(C, expected)


@pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn"),
    reason="FP8 dtype unavailable",
)
def test_invoke_fused_moe_triton_kernel_fp8_block_quant_falls_back_without_triton_runtime_module(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fused_moe_mod, "HAS_TRITON", False)
    monkeypatch.setattr(fused_moe_mod, "triton", SimpleNamespace())

    def _raise_if_triton_path_runs(*_args, **_kwargs):
        raise AssertionError("fused_moe FP8 Triton kernel should not run")

    monkeypatch.setattr(fused_moe_mod, "fused_moe_kernel", _raise_if_triton_path_runs)

    fp8 = torch.float8_e4m3fn
    A = torch.tensor(
        [
            [2.0, -1.0],
            [1.5, 0.5],
        ],
        dtype=fp8,
    )
    B = torch.tensor(
        [
            [[1.0, 2.0], [-1.0, 0.5]],
            [[0.25, -2.0], [1.5, 1.0]],
        ],
        dtype=fp8,
    )
    C = torch.zeros((2, 1, 2), dtype=torch.float32)
    A_scale = torch.tensor([[0.5], [0.5]], dtype=torch.float32)
    B_scale = torch.tensor(
        [
            [[0.25]],
            [[2.0]],
        ],
        dtype=torch.float32,
    )
    expert_ids = torch.tensor([[0], [1]], dtype=torch.int32)

    fused_moe_mod.invoke_fused_moe_triton_kernel(
        A=A,
        B=B,
        C=C,
        A_scale=A_scale,
        B_scale=B_scale,
        topk_weights=torch.ones((2, 1), dtype=torch.float32),
        sorted_token_ids=None,
        expert_ids=expert_ids,
        num_tokens_post_padded=torch.tensor([0], dtype=torch.int32),
        mul_routed_weight=False,
        top_k=1,
        config={"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 16},
        compute_type=None,
        use_fp8_w8a8=True,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        block_shape=[2, 2],
        B_bias=None,
    )

    expected = torch.zeros_like(C)
    expected[0, 0] = (A[0].to(torch.float32) * A_scale[0]) @ (
        B[0].to(torch.float32) * B_scale[0]
    ).transpose(0, 1)
    expected[1, 0] = (A[1].to(torch.float32) * A_scale[1]) @ (
        B[1].to(torch.float32) * B_scale[1]
    ).transpose(0, 1)

    torch.testing.assert_close(C, expected)


@pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn"),
    reason="FP8 dtype unavailable",
)
def test_invoke_fused_moe_triton_kernel_fp8_prefers_shared_cuda_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fused_moe_mod, "HAS_TRITON", False)

    calls: dict[str, object] = {}

    def _fake_grouped_cuda(**kwargs):
        calls["use_fp8_w8a8"] = kwargs["use_fp8_w8a8"]
        calls["per_channel_quant"] = kwargs["per_channel_quant"]
        calls["a_scale_shape"] = (
            None if kwargs["A_scale"] is None else tuple(kwargs["A_scale"].shape)
        )
        calls["b_scale_shape"] = (
            None if kwargs["B_scale"] is None else tuple(kwargs["B_scale"].shape)
        )
        kwargs["C"].fill_(7.0)
        return True

    monkeypatch.setattr(
        fused_moe_mod,
        "_try_precompiled_fused_moe_kernel",
        _fake_grouped_cuda,
    )
    monkeypatch.setattr(
        fused_moe_mod,
        "_invoke_fused_moe_reference",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("torch fallback should not run")
        ),
    )

    fp8 = torch.float8_e4m3fn
    A = torch.randn(2, 4, dtype=torch.float16).to(fp8)
    B = torch.randn(3, 5, 4, dtype=torch.float16).to(fp8)
    C = torch.zeros((2, 1, 5), dtype=torch.float32)
    expert_ids = torch.tensor([[0], [2]], dtype=torch.int32)
    A_scale = torch.ones((2, 1), dtype=torch.float32)
    B_scale = torch.ones((3, 1, 1), dtype=torch.float32)

    fused_moe_mod.invoke_fused_moe_triton_kernel(
        A=A,
        B=B,
        C=C,
        A_scale=A_scale,
        B_scale=B_scale,
        topk_weights=torch.ones((2, 1), dtype=torch.float32),
        sorted_token_ids=None,
        expert_ids=expert_ids,
        num_tokens_post_padded=torch.tensor([0], dtype=torch.int32),
        mul_routed_weight=False,
        top_k=1,
        config={"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 16},
        compute_type=None,
        use_fp8_w8a8=True,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=True,
        block_shape=None,
        B_bias=None,
    )

    assert calls == {
        "use_fp8_w8a8": True,
        "per_channel_quant": True,
        "a_scale_shape": (2, 1),
        "b_scale_shape": (3, 1, 1),
    }
    assert torch.all(C == 7.0).item()


def test_fused_moe_apply_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(fused_moe_mod, "HAS_TRITON", False)

    def _raise_if_triton_path_runs(*_args, **_kwargs):
        raise AssertionError("fused_moe Triton kernel should not run")

    monkeypatch.setattr(fused_moe_mod, "fused_moe_kernel", _raise_if_triton_path_runs)
    monkeypatch.setattr(
        fused_moe_mod,
        "try_get_optimal_moe_config",
        lambda *_args, **_kwargs: {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 16,
            "BLOCK_SIZE_K": 16,
        },
    )
    monkeypatch.setattr(
        fused_moe_mod,
        "moe_align_block_size",
        lambda *_args, **_kwargs: (
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([2], dtype=torch.int32),
        ),
    )

    experts = fused_moe_mod.TritonExperts(
        moe_config=SimpleNamespace(
            is_act_and_mul=False,
            activation=fused_moe_activation.MoEActivation.RELU2_NO_MUL,
            moe_parallel_config=SimpleNamespace(use_fi_all2allv_kernels=False),
            routing_method=None,
            router_logits_dtype=None,
            hidden_dim=2,
        ),
        quant_config=FUSED_MOE_UNQUANTIZED_CONFIG,
    )

    hidden_states = torch.tensor(
        [
            [1.0, -2.0],
            [0.5, 3.0],
        ],
        dtype=torch.float32,
    )
    w1 = torch.tensor(
        [
            [[1.0, 0.5], [-1.0, 2.0]],
            [[0.25, -1.5], [1.5, 0.75]],
        ],
        dtype=torch.float32,
    )
    w2 = torch.tensor(
        [
            [[1.0, -0.5], [0.25, 1.5]],
            [[-1.0, 2.0], [0.5, -0.25]],
        ],
        dtype=torch.float32,
    )
    topk_weights = torch.ones((2, 1), dtype=torch.float32)
    topk_ids = torch.tensor([[0], [1]], dtype=torch.int32)
    output = torch.zeros((2, 2), dtype=torch.float32)
    workspace13 = torch.empty((2, 2), dtype=torch.float32)
    workspace2 = torch.empty((2, 1, 2), dtype=torch.float32)

    experts.apply(
        output=output,
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=fused_moe_activation.MoEActivation.RELU2_NO_MUL,
        global_num_experts=2,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=workspace13,
        workspace2=workspace2,
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )

    expected_intermediate1 = torch.stack(
        [
            hidden_states[0] @ w1[0].transpose(0, 1),
            hidden_states[1] @ w1[1].transpose(0, 1),
        ]
    ).view(2, 1, 2)
    expected_intermediate2 = torch.zeros_like(expected_intermediate1)
    fused_moe_activation.apply_moe_activation(
        fused_moe_activation.MoEActivation.RELU2_NO_MUL,
        expected_intermediate2.view(-1, expected_intermediate2.size(-1)),
        expected_intermediate1.view(-1, expected_intermediate1.size(-1)),
    )
    expected = torch.stack(
        [
            expected_intermediate2[0, 0] @ w2[0].transpose(0, 1),
            expected_intermediate2[1, 0] @ w2[1].transpose(0, 1),
        ]
    )

    torch.testing.assert_close(output, expected)


def test_fused_experts_impl_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(fused_moe_mod, "HAS_TRITON", False)

    def _raise_if_triton_path_runs(*_args, **_kwargs):
        raise AssertionError("fused_moe Triton kernel should not run")

    monkeypatch.setattr(fused_moe_mod, "fused_moe_kernel", _raise_if_triton_path_runs)
    monkeypatch.setattr(
        fused_moe_mod,
        "try_get_optimal_moe_config",
        lambda *_args, **_kwargs: {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 16,
            "BLOCK_SIZE_K": 16,
        },
    )
    monkeypatch.setattr(
        fused_moe_mod,
        "moe_align_block_size",
        lambda *_args, **_kwargs: (
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([2], dtype=torch.int32),
        ),
    )

    hidden_states = torch.tensor(
        [
            [1.0, -2.0],
            [0.5, 3.0],
        ],
        dtype=torch.float32,
    )
    w1 = torch.tensor(
        [
            [[1.0, 0.5], [-1.0, 2.0]],
            [[0.25, -1.5], [1.5, 0.75]],
        ],
        dtype=torch.float32,
    )
    w2 = torch.tensor(
        [
            [[1.0, -0.5], [0.25, 1.5]],
            [[-1.0, 2.0], [0.5, -0.25]],
        ],
        dtype=torch.float32,
    )
    topk_weights = torch.ones((2, 1), dtype=torch.float32)
    topk_ids = torch.tensor([[0], [1]], dtype=torch.int32)

    output = fused_moe_mod.fused_experts_impl(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inplace=False,
        activation="relu2_no_mul",
        apply_router_weight_on_input=False,
    )

    expected_intermediate1 = torch.stack(
        [
            hidden_states[0] @ w1[0].transpose(0, 1),
            hidden_states[1] @ w1[1].transpose(0, 1),
        ]
    ).view(2, 1, 2)
    expected_intermediate2 = torch.zeros_like(expected_intermediate1)
    fused_moe_activation.apply_moe_activation(
        fused_moe_activation.MoEActivation.RELU2_NO_MUL,
        expected_intermediate2.view(-1, expected_intermediate2.size(-1)),
        expected_intermediate1.view(-1, expected_intermediate1.size(-1)),
    )
    expected = torch.stack(
        [
            expected_intermediate2[0, 0] @ w2[0].transpose(0, 1),
            expected_intermediate2[1, 0] @ w2[1].transpose(0, 1),
        ]
    )

    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("quant_mode", ["int8_w8a16", "int4_w4a16"])
def test_fused_experts_forwards_wna16_quant_config(
    monkeypatch,
    quant_mode: str,
) -> None:
    captured: dict[str, object] = {}

    def _fake_dispatch_func(inplace: bool):
        assert inplace is False

        def _runner(**kwargs):
            captured["activation"] = kwargs["activation"]
            captured["use_int8_w8a16"] = kwargs["use_int8_w8a16"]
            captured["use_int4_w4a16"] = kwargs["use_int4_w4a16"]
            captured["block_shape"] = kwargs["block_shape"]
            captured["w1_scale_shape"] = tuple(kwargs["w1_scale"].shape)
            captured["w2_scale_shape"] = tuple(kwargs["w2_scale"].shape)
            captured["w1_zp_is_none"] = kwargs["w1_zp"] is None
            captured["w2_zp_is_none"] = kwargs["w2_zp"] is None
            return torch.full_like(kwargs["hidden_states"], 5.0)

        return _runner

    monkeypatch.setattr(
        fused_moe_mod,
        "dispatch_fused_experts_func",
        _fake_dispatch_func,
    )

    hidden_states = torch.zeros((2, 4), dtype=torch.float16)
    topk_weights = torch.ones((2, 1), dtype=torch.float32)
    topk_ids = torch.tensor([[0], [1]], dtype=torch.int32)

    if quant_mode == "int8_w8a16":
        w1 = torch.zeros((2, 2, 4), dtype=torch.int8)
        w2 = torch.zeros((2, 4, 2), dtype=torch.int8)
        quant_config = int8_w8a16_moe_quant_config(
            w1_scale=torch.ones((2, 2, 2), dtype=torch.float32),
            w2_scale=torch.ones((2, 4, 1), dtype=torch.float32),
            w1_zp=None,
            w2_zp=None,
            block_shape=[0, 2],
        )
        expected = {
            "activation": fused_moe_activation.MoEActivation.RELU2_NO_MUL.value,
            "use_int8_w8a16": True,
            "use_int4_w4a16": False,
            "block_shape": [0, 2],
            "w1_scale_shape": (2, 2, 2),
            "w2_scale_shape": (2, 4, 1),
            "w1_zp_is_none": True,
            "w2_zp_is_none": True,
        }
    else:
        w1 = torch.zeros((2, 2, 2), dtype=torch.int8)
        w2 = torch.zeros((2, 4, 1), dtype=torch.int8)
        quant_config = int4_w4a16_moe_quant_config(
            w1_scale=torch.ones((2, 2, 2), dtype=torch.float32),
            w2_scale=torch.ones((2, 4, 1), dtype=torch.float32),
            w1_zp=torch.zeros((2, 1, 2), dtype=torch.int8),
            w2_zp=torch.zeros((2, 2, 1), dtype=torch.int8),
            block_shape=[0, 2],
        )
        expected = {
            "activation": fused_moe_activation.MoEActivation.RELU2_NO_MUL.value,
            "use_int8_w8a16": False,
            "use_int4_w4a16": True,
            "block_shape": [0, 2],
            "w1_scale_shape": (2, 2, 2),
            "w2_scale_shape": (2, 4, 1),
            "w1_zp_is_none": False,
            "w2_zp_is_none": False,
        }

    output = fused_moe_mod.fused_experts(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inplace=False,
        activation=fused_moe_activation.MoEActivation.RELU2_NO_MUL,
        apply_router_weight_on_input=False,
        quant_config=quant_config,
    )

    assert captured == expected
    torch.testing.assert_close(output, torch.full_like(hidden_states, 5.0))


@pytest.mark.parametrize("quant_mode", ["int8_w8a16", "int4_w4a16"])
def test_fused_experts_impl_wna16_uses_shared_dispatch_without_triton(
    monkeypatch,
    quant_mode: str,
) -> None:
    monkeypatch.setattr(fused_moe_mod, "HAS_TRITON", False)
    monkeypatch.setattr(
        fused_moe_mod,
        "try_get_optimal_moe_config",
        lambda *_args, **_kwargs: {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 16,
            "BLOCK_SIZE_K": 16,
        },
    )

    sorted_token_ids = torch.tensor([0, 1], dtype=torch.int32)
    expert_ids = torch.tensor([0, 1], dtype=torch.int32)
    num_tokens_post_padded = torch.tensor([2], dtype=torch.int32)

    monkeypatch.setattr(
        fused_moe_mod,
        "moe_align_block_size",
        lambda *_args, **_kwargs: (
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
        ),
    )
    monkeypatch.setattr(
        fused_moe_mod.ops,
        "moe_sum",
        lambda intermediate, output: output.copy_(intermediate.sum(dim=1)),
    )

    calls: list[dict[str, object]] = []

    def _fake_dispatch(
        A,
        B,
        C,
        A_scale,
        B_scale,
        B_zp,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight,
        top_k,
        config,
        *,
        compute_type,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        use_int4_w4a16,
        per_channel_quant,
        block_shape,
        B_bias,
    ) -> None:
        calls.append(
            {
                "shape_A": tuple(A.shape),
                "shape_B": tuple(B.shape),
                "shape_C": tuple(C.shape),
                "sorted_token_ids": tuple(sorted_token_ids.tolist())
                if sorted_token_ids is not None
                else None,
                "expert_ids": tuple(expert_ids.tolist()),
                "num_tokens_post_padded": int(num_tokens_post_padded.item()),
                "mul_routed_weight": mul_routed_weight,
                "top_k": top_k,
                "compute_type": compute_type,
                "use_fp8_w8a8": use_fp8_w8a8,
                "use_int8_w8a8": use_int8_w8a8,
                "use_int8_w8a16": use_int8_w8a16,
                "use_int4_w4a16": use_int4_w4a16,
                "per_channel_quant": per_channel_quant,
                "block_shape": block_shape,
                "a_scale_is_none": A_scale is None,
                "b_scale_shape": tuple(B_scale.shape),
                "has_zp": B_zp is not None,
                "has_bias": B_bias is not None,
            }
        )
        if len(calls) == 1:
            C.copy_(
                torch.tensor(
                    [[[-1.0, 2.0]], [[3.0, -4.0]]],
                    dtype=C.dtype,
                    device=C.device,
                )
            )
        elif len(calls) == 2:
            C.copy_(
                torch.tensor(
                    [[[3.0, 4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0, 10.0]]],
                    dtype=C.dtype,
                    device=C.device,
                )
            )
        else:
            raise AssertionError("dispatch_fused_moe_kernel should run exactly twice")

    monkeypatch.setattr(
        fused_moe_mod,
        "dispatch_fused_moe_kernel",
        _fake_dispatch,
    )

    hidden_states = torch.tensor(
        [
            [1.0, -2.0, 0.5, 3.0],
            [0.25, -1.5, 1.5, 0.75],
        ],
        dtype=torch.float16,
    )
    topk_weights = torch.tensor([[0.5], [1.5]], dtype=torch.float32)
    topk_ids = torch.tensor([[0], [1]], dtype=torch.int32)

    if quant_mode == "int8_w8a16":
        w1 = torch.zeros((2, 2, 4), dtype=torch.int8)
        w2 = torch.zeros((2, 4, 2), dtype=torch.int8)
        kwargs = {
            "use_int8_w8a16": True,
            "use_int4_w4a16": False,
            "w1_scale": torch.ones((2, 2, 2), dtype=torch.float32),
            "w2_scale": torch.ones((2, 4, 1), dtype=torch.float32),
            "w1_zp": None,
            "w2_zp": None,
            "block_shape": [0, 2],
        }
        expected_has_zp = False
        expected_shapes = [(2, 4), (2, 2, 4), (2, 1, 2)]
        expected_w2_shape = (2, 4, 2)
    else:
        w1 = torch.zeros((2, 2, 2), dtype=torch.int8)
        w2 = torch.zeros((2, 4, 1), dtype=torch.int8)
        kwargs = {
            "use_int8_w8a16": False,
            "use_int4_w4a16": True,
            "w1_scale": torch.ones((2, 2, 2), dtype=torch.float32),
            "w2_scale": torch.ones((2, 4, 1), dtype=torch.float32),
            "w1_zp": torch.zeros((2, 1, 2), dtype=torch.int8),
            "w2_zp": torch.zeros((2, 2, 1), dtype=torch.int8),
            "block_shape": [0, 2],
        }
        expected_has_zp = True
        expected_shapes = [(2, 4), (2, 2, 2), (2, 1, 2)]
        expected_w2_shape = (2, 4, 1)

    output = fused_moe_mod.fused_experts_impl(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inplace=False,
        activation="relu2_no_mul",
        apply_router_weight_on_input=True,
        per_channel_quant=False,
        **kwargs,
    )

    assert len(calls) == 2
    assert calls[0]["shape_A"] == expected_shapes[0]
    assert calls[0]["shape_B"] == expected_shapes[1]
    assert calls[0]["shape_C"] == expected_shapes[2]
    assert calls[1]["shape_A"] == (2, 2)
    assert calls[1]["shape_B"] == expected_w2_shape
    assert calls[1]["shape_C"] == (2, 1, 4)
    assert all(call["sorted_token_ids"] == (0, 1) for call in calls)
    assert all(call["expert_ids"] == (0, 1) for call in calls)
    assert all(call["num_tokens_post_padded"] == 2 for call in calls)
    assert calls[0]["mul_routed_weight"] is True
    assert calls[1]["mul_routed_weight"] is False
    assert all(call["top_k"] == 1 for call in calls)
    assert all(call["compute_type"] is None for call in calls)
    assert all(call["use_fp8_w8a8"] is False for call in calls)
    assert all(call["use_int8_w8a8"] is False for call in calls)
    assert all(call["per_channel_quant"] is False for call in calls)
    assert all(call["block_shape"] == [0, 2] for call in calls)
    assert all(call["a_scale_is_none"] is True for call in calls)
    assert all(call["has_zp"] is expected_has_zp for call in calls)
    assert all(call["has_bias"] is False for call in calls)
    torch.testing.assert_close(
        output,
        torch.tensor(
            [[3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0]],
            dtype=torch.float16,
        ),
    )


def test_unquantized_moe_oracle_prefers_cuda_aten_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(unquantized_moe_oracle, "HAS_TRITON", False)
    monkeypatch.setattr(
        unquantized_moe_oracle,
        "_has_cuda_aten_moe_backend",
        lambda: True,
    )
    monkeypatch.setattr(
        unquantized_moe_oracle,
        "is_supported_config_trtllm_bf16",
        lambda **_: (False, None),
    )
    monkeypatch.setattr(unquantized_moe_oracle, "has_flashinfer", lambda: False)
    monkeypatch.setattr(
        unquantized_moe_oracle,
        "has_flashinfer_cutlass_fused_moe",
        lambda: False,
    )
    monkeypatch.setattr(
        unquantized_moe_oracle.rocm_aiter_ops,
        "is_fused_moe_enabled",
        lambda: False,
    )
    monkeypatch.setattr(unquantized_moe_oracle.envs, "VLLM_USE_FLASHINFER_MOE_FP16", False)
    monkeypatch.setattr(
        unquantized_moe_oracle.envs,
        "VLLM_FLASHINFER_MOE_BACKEND",
        "latency",
    )
    monkeypatch.setattr(unquantized_moe_oracle.current_platform, "is_rocm", lambda: False)
    monkeypatch.setattr(unquantized_moe_oracle.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(unquantized_moe_oracle.current_platform, "is_xpu", lambda: False)
    monkeypatch.setattr(unquantized_moe_oracle.current_platform, "is_cpu", lambda: False)
    monkeypatch.setattr(unquantized_moe_oracle.current_platform, "is_tpu", lambda: False)
    monkeypatch.setattr(
        unquantized_moe_oracle.current_platform,
        "is_out_of_tree",
        lambda: False,
    )
    monkeypatch.setattr(
        unquantized_moe_oracle.current_platform,
        "has_device_capability",
        lambda *_args, **_kwargs: False,
    )

    moe_config = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(use_batched_activation_format=False),
        moe_backend="auto",
    )

    backend = unquantized_moe_oracle.select_unquantized_moe_backend(
        moe_config=moe_config,
        use_ep=False,
        use_dp=False,
    )

    assert backend == unquantized_moe_oracle.UnquantizedMoeBackend.CUDA_ATEN


def test_fused_batched_moe_triton_experts_support_shared_reference_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fused_batched_moe_mod, "HAS_TRITON", False)
    monkeypatch.setattr(
        fused_batched_moe_mod.current_platform,
        "is_cuda_alike",
        lambda: True,
    )

    moe_config = SimpleNamespace(
        is_act_and_mul=True,
        activation=fused_moe_activation.MoEActivation.SILU,
        moe_parallel_config=SimpleNamespace(use_fi_all2allv_kernels=False),
        routing_method=None,
        router_logits_dtype=None,
        hidden_dim=128,
    )

    supported, reason = fused_batched_moe_mod.BatchedTritonExperts.is_supported_config(
        fused_batched_moe_mod.BatchedTritonExperts,
        moe_config,
        None,
        None,
        fused_batched_moe_mod.BatchedTritonExperts.activation_format(),
    )

    assert supported
    assert reason is None


def test_invoke_batched_moe_kernel_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(fused_batched_moe_mod, "HAS_TRITON", False)

    def _raise_if_triton_path_runs(*_args, **_kwargs):
        raise AssertionError("batched MoE Triton path should not run")

    monkeypatch.setattr(
        fused_batched_moe_mod,
        "batched_triton_kernel",
        _raise_if_triton_path_runs,
    )

    A = torch.tensor(
        [
            [[1.0, 2.0], [0.5, -1.0], [9.0, 9.0]],
            [[-1.5, 0.25], [7.0, 7.0], [7.0, 7.0]],
        ],
        dtype=torch.float32,
    )
    B = torch.tensor(
        [
            [[1.0, -1.0], [0.5, 2.0], [-2.0, 1.0]],
            [[0.25, 1.0], [-1.5, 0.5], [2.0, -0.5]],
        ],
        dtype=torch.float32,
    )
    C = torch.full((2, 3, 3), -123.0, dtype=torch.float32)
    expert_num_tokens = torch.tensor([2, 1], dtype=torch.int32)

    fused_batched_moe_mod.invoke_moe_batched_triton_kernel(
        A=A,
        B=B,
        C=C,
        expert_num_tokens=expert_num_tokens,
        compute_type=None,
        A_scale=None,
        B_scale=None,
        B_zp=torch.empty(0),
        use_fp8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        config={"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 16},
        per_act_token_quant=False,
        block_shape=None,
    )

    expected = torch.zeros_like(C)
    expected[0, :2] = A[0, :2].to(torch.float32) @ B[0].transpose(0, 1)
    expected[1, :1] = A[1, :1].to(torch.float32) @ B[1].transpose(0, 1)

    torch.testing.assert_close(C, expected)


def test_invoke_batched_moe_kernel_prefers_precompiled_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fused_batched_moe_mod, "HAS_TRITON", False)

    calls: dict[str, object] = {}

    def _fake_precompiled(**kwargs):
        calls["shape_A"] = tuple(kwargs["A"].shape)
        calls["shape_B"] = tuple(kwargs["B"].shape)
        calls["shape_C"] = tuple(kwargs["C"].shape)
        kwargs["C"].fill_(17.0)
        return True

    monkeypatch.setattr(
        fused_batched_moe_mod,
        "_try_precompiled_moe_batched_kernel",
        _fake_precompiled,
    )
    monkeypatch.setattr(
        fused_batched_moe_mod,
        "_invoke_moe_batched_reference",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("torch fallback should not run")
        ),
    )

    A = torch.randn(2, 3, 4, dtype=torch.float16)
    B = torch.randn(2, 5, 4, dtype=torch.float16)
    C = torch.zeros((2, 3, 5), dtype=torch.float16)
    expert_num_tokens = torch.tensor([2, 1], dtype=torch.int32)

    fused_batched_moe_mod.invoke_moe_batched_triton_kernel(
        A=A,
        B=B,
        C=C,
        expert_num_tokens=expert_num_tokens,
        compute_type=None,
        A_scale=None,
        B_scale=None,
        B_zp=torch.empty(0),
        use_fp8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        config={"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 16},
        per_act_token_quant=False,
        block_shape=None,
    )

    assert calls == {
        "shape_A": (2, 3, 4),
        "shape_B": (2, 5, 4),
        "shape_C": (2, 3, 5),
    }
    assert torch.all(C == 17.0).item()


def test_fused_batched_moe_apply_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(fused_batched_moe_mod, "HAS_TRITON", False)

    def _raise_if_triton_path_runs(*_args, **_kwargs):
        raise AssertionError("batched MoE Triton kernel should not run")

    monkeypatch.setattr(
        fused_batched_moe_mod,
        "batched_triton_kernel",
        _raise_if_triton_path_runs,
    )

    experts = fused_batched_moe_mod.BatchedTritonExperts(
        moe_config=SimpleNamespace(
            is_act_and_mul=False,
            activation=fused_moe_activation.MoEActivation.RELU2_NO_MUL,
            moe_parallel_config=SimpleNamespace(use_fi_all2allv_kernels=False),
            routing_method=None,
            router_logits_dtype=None,
            hidden_dim=2,
        ),
        quant_config=FUSED_MOE_UNQUANTIZED_CONFIG,
        max_num_tokens=3,
        num_dispatchers=1,
    )

    hidden_states = torch.tensor(
        [
            [[1.0, -2.0], [0.5, 3.0], [7.0, 7.0]],
            [[-1.5, 0.25], [8.0, 8.0], [8.0, 8.0]],
        ],
        dtype=torch.float32,
    )
    w1 = torch.tensor(
        [
            [[1.0, 0.5], [-1.0, 2.0]],
            [[0.25, -1.5], [1.5, 0.75]],
        ],
        dtype=torch.float32,
    )
    w2 = torch.tensor(
        [
            [[1.0, -0.5], [0.25, 1.5]],
            [[-1.0, 2.0], [0.5, -0.25]],
        ],
        dtype=torch.float32,
    )
    output = torch.zeros((2, 3, 2), dtype=torch.float32)
    workspace13 = torch.empty((2, 3, 2), dtype=torch.float32)
    workspace2 = torch.empty((2, 3, 2), dtype=torch.float32)
    expert_num_tokens = torch.tensor([2, 1], dtype=torch.int32)

    experts.apply(
        output=output,
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=torch.ones((1, 1), dtype=torch.float32),
        topk_ids=torch.zeros((1, 1), dtype=torch.int32),
        activation=fused_moe_activation.MoEActivation.RELU2_NO_MUL,
        global_num_experts=2,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=workspace13,
        workspace2=workspace2,
        expert_tokens_meta=SimpleNamespace(
            expert_num_tokens=expert_num_tokens,
            expert_num_tokens_cpu=None,
        ),
        apply_router_weight_on_input=False,
    )

    expected_intermediate1 = torch.zeros((2, 3, 2), dtype=torch.float32)
    expected_intermediate1[0, :2] = hidden_states[0, :2] @ w1[0].transpose(0, 1)
    expected_intermediate1[1, :1] = hidden_states[1, :1] @ w1[1].transpose(0, 1)

    expected_intermediate2 = torch.zeros_like(expected_intermediate1)
    fused_moe_activation.apply_moe_activation(
        fused_moe_activation.MoEActivation.RELU2_NO_MUL,
        expected_intermediate2.view(-1, expected_intermediate2.size(-1)),
        expected_intermediate1.view(-1, expected_intermediate1.size(-1)),
    )

    expected = torch.zeros_like(output)
    expected[0, :2] = expected_intermediate2[0, :2] @ w2[0].transpose(0, 1)
    expected[1, :1] = expected_intermediate2[1, :1] @ w2[1].transpose(0, 1)

    torch.testing.assert_close(output, expected)


@pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn"),
    reason="FP8 dtype unavailable",
)
def test_invoke_batched_moe_kernel_fp8_falls_back_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fused_batched_moe_mod, "HAS_TRITON", False)

    def _raise_if_triton_path_runs(*_args, **_kwargs):
        raise AssertionError("batched FP8 MoE Triton path should not run")

    monkeypatch.setattr(
        fused_batched_moe_mod,
        "batched_triton_kernel",
        _raise_if_triton_path_runs,
    )

    fp8 = torch.float8_e4m3fn
    A = torch.tensor(
        [
            [[2.0, -1.0], [1.5, 0.5]],
            [[-2.0, 1.0], [3.0, -0.5]],
        ],
        dtype=fp8,
    )
    B = torch.tensor(
        [
            [[1.0, 2.0], [-1.0, 0.5]],
            [[0.25, -2.0], [1.5, 1.0]],
        ],
        dtype=fp8,
    )
    C = torch.zeros((2, 2, 2), dtype=torch.float32)
    expert_num_tokens = torch.tensor([2, 1], dtype=torch.int32)
    A_scale = torch.tensor(
        [
            [[0.5], [2.0]],
            [[1.5], [1.0]],
        ],
        dtype=torch.float32,
    )
    B_scale = torch.tensor(
        [
            [[0.25]],
            [[2.0]],
        ],
        dtype=torch.float32,
    )

    fused_batched_moe_mod.invoke_moe_batched_triton_kernel(
        A=A,
        B=B,
        C=C,
        expert_num_tokens=expert_num_tokens,
        compute_type=None,
        A_scale=A_scale,
        B_scale=B_scale,
        B_zp=torch.empty(0),
        use_fp8_w8a8=True,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        config={"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 16},
        per_act_token_quant=True,
        block_shape=None,
    )

    expected = torch.zeros_like(C)
    expected[0, :2] = (A[0, :2].to(torch.float32) * A_scale[0, :2]) @ (
        B[0].to(torch.float32) * B_scale[0]
    ).transpose(0, 1)
    expected[1, :1] = (A[1, :1].to(torch.float32) * A_scale[1, :1]) @ (
        B[1].to(torch.float32) * B_scale[1]
    ).transpose(0, 1)

    torch.testing.assert_close(C, expected)


def test_invoke_batched_moe_kernel_int8_w8a16_falls_back_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fused_batched_moe_mod, "HAS_TRITON", False)

    def _raise_if_triton_path_runs(*_args, **_kwargs):
        raise AssertionError("batched int8_w8a16 Triton path should not run")

    monkeypatch.setattr(
        fused_batched_moe_mod,
        "batched_triton_kernel",
        _raise_if_triton_path_runs,
    )

    A = torch.tensor(
        [
            [[2.0, -1.0], [1.5, 0.5]],
            [[-2.0, 1.0], [3.0, -0.5]],
        ],
        dtype=torch.float32,
    )
    B = torch.tensor(
        [
            [[1, 2], [-1, 0]],
            [[2, -2], [1, 1]],
        ],
        dtype=torch.int8,
    )
    C = torch.zeros((2, 2, 2), dtype=torch.float32)
    expert_num_tokens = torch.tensor([2, 1], dtype=torch.int32)
    B_scale = torch.tensor(
        [
            [[0.5]],
            [[2.0]],
        ],
        dtype=torch.float32,
    )

    fused_batched_moe_mod.invoke_moe_batched_triton_kernel(
        A=A,
        B=B,
        C=C,
        expert_num_tokens=expert_num_tokens,
        compute_type=None,
        A_scale=None,
        B_scale=B_scale,
        B_zp=torch.empty(0),
        use_fp8_w8a8=False,
        use_int8_w8a16=True,
        use_int4_w4a16=False,
        config={"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 16},
        per_act_token_quant=False,
        block_shape=None,
    )

    expected = torch.zeros_like(C)
    expected[0, :2] = A[0, :2].to(torch.float32) @ (
        B[0].to(torch.float32) * B_scale[0]
    ).transpose(0, 1)
    expected[1, :1] = A[1, :1].to(torch.float32) @ (
        B[1].to(torch.float32) * B_scale[1]
    ).transpose(0, 1)

    torch.testing.assert_close(C, expected)


def test_invoke_batched_moe_kernel_int8_w8a16_with_zp_falls_back_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fused_batched_moe_mod, "HAS_TRITON", False)
    monkeypatch.setattr(
        fused_batched_moe_mod,
        "_try_dispatch_batched_wna16_shared",
        lambda **_kwargs: False,
    )

    def _raise_if_triton_path_runs(*_args, **_kwargs):
        raise AssertionError("batched int8_w8a16 Triton path should not run")

    monkeypatch.setattr(
        fused_batched_moe_mod,
        "batched_triton_kernel",
        _raise_if_triton_path_runs,
    )

    A = torch.tensor(
        [
            [[2.0, -1.0, 0.5, 3.0], [1.5, 0.5, -2.0, 1.0]],
            [[-2.0, 1.0, 0.25, -1.5], [8.0, 8.0, 8.0, 8.0]],
        ],
        dtype=torch.float32,
    )
    B = torch.tensor(
        [
            [[9, 5, 12, 6], [3, 8, 14, 7]],
            [[10, 8, 4, 6], [7, 11, 5, 9]],
        ],
        dtype=torch.int8,
    )
    B_zp = torch.tensor(
        [
            [[8, 9], [6, 7]],
            [[9, 8], [5, 6]],
        ],
        dtype=torch.int8,
    )
    B_scale = torch.tensor(
        [
            [[0.5, 0.25], [1.0, 0.75]],
            [[0.25, 0.5], [1.5, 0.125]],
        ],
        dtype=torch.float32,
    )
    C = torch.zeros((2, 2, 2), dtype=torch.float32)
    expert_num_tokens = torch.tensor([2, 1], dtype=torch.int32)

    fused_batched_moe_mod.invoke_moe_batched_triton_kernel(
        A=A,
        B=B,
        C=C,
        expert_num_tokens=expert_num_tokens,
        compute_type=None,
        A_scale=None,
        B_scale=B_scale,
        B_zp=B_zp,
        use_fp8_w8a8=False,
        use_int8_w8a16=True,
        use_int4_w4a16=False,
        config={"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 16},
        per_act_token_quant=False,
        block_shape=[0, 2],
    )

    expected = torch.zeros_like(C)
    for expert_idx, num_tokens in enumerate(expert_num_tokens.tolist()):
        expanded_scale = B_scale[expert_idx].repeat_interleave(2, dim=1)
        expanded_zp = B_zp[expert_idx].to(torch.float32).repeat_interleave(2, dim=1)
        weight_dq = (B[expert_idx].to(torch.float32) - expanded_zp) * expanded_scale
        expected[expert_idx, :num_tokens] = A[expert_idx, :num_tokens] @ weight_dq.transpose(
            0, 1
        )

    torch.testing.assert_close(C, expected)


def test_invoke_batched_moe_kernel_int8_w8a16_prefers_shared_dispatch_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fused_batched_moe_mod, "HAS_TRITON", False)

    calls: dict[str, object] = {}

    def _fake_try_shared(**kwargs):
        calls["use_int4_w4a16"] = kwargs["use_int4_w4a16"]
        calls["use_int8_w8a16"] = kwargs["use_int8_w8a16"]
        calls["block_shape"] = kwargs["block_shape"]
        calls["has_zp"] = kwargs["B_zp"] is not None
        kwargs["C"].fill_(11.0)
        return True

    monkeypatch.setattr(
        fused_batched_moe_mod,
        "_try_dispatch_batched_wna16_shared",
        _fake_try_shared,
    )
    monkeypatch.setattr(
        fused_batched_moe_mod,
        "_try_precompiled_moe_batched_kernel",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("batched precompiled kernel should not run")
        ),
    )
    monkeypatch.setattr(
        fused_batched_moe_mod,
        "_invoke_moe_batched_reference",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("batched torch reference should not run")
        ),
    )

    C = torch.zeros((1, 1, 2), dtype=torch.float32)
    fused_batched_moe_mod.invoke_moe_batched_triton_kernel(
        A=torch.randn((1, 1, 4), dtype=torch.float32),
        B=torch.zeros((1, 2, 4), dtype=torch.int8),
        C=C,
        expert_num_tokens=torch.tensor([1], dtype=torch.int32),
        compute_type=None,
        A_scale=None,
        B_scale=torch.ones((1, 2, 2), dtype=torch.float32),
        B_zp=torch.zeros((1, 2, 2), dtype=torch.int8),
        use_fp8_w8a8=False,
        use_int8_w8a16=True,
        use_int4_w4a16=False,
        config={"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 16},
        per_act_token_quant=False,
        block_shape=[0, 2],
    )

    assert calls == {
        "use_int4_w4a16": False,
        "use_int8_w8a16": True,
        "block_shape": [0, 2],
        "has_zp": True,
    }
    assert torch.all(C == 11.0).item()


def test_invoke_batched_moe_kernel_int8_w8a16_prefers_shared_dispatch_with_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fused_batched_moe_mod, "HAS_TRITON", True)

    calls: dict[str, object] = {}

    def _fake_try_shared(**kwargs):
        calls["use_int4_w4a16"] = kwargs["use_int4_w4a16"]
        calls["use_int8_w8a16"] = kwargs["use_int8_w8a16"]
        calls["block_shape"] = kwargs["block_shape"]
        calls["has_zp"] = kwargs["B_zp"] is not None
        kwargs["C"].fill_(13.0)
        return True

    monkeypatch.setattr(
        fused_batched_moe_mod,
        "_try_dispatch_batched_wna16_shared",
        _fake_try_shared,
    )

    def _raise_if_triton_path_runs(*_args, **_kwargs):
        raise AssertionError("batched int8_w8a16 Triton kernel path should not run")

    monkeypatch.setattr(
        fused_batched_moe_mod,
        "batched_triton_kernel",
        _raise_if_triton_path_runs,
    )

    C = torch.zeros((1, 1, 2), dtype=torch.float32)
    fused_batched_moe_mod.invoke_moe_batched_triton_kernel(
        A=torch.randn((1, 1, 4), dtype=torch.float32),
        B=torch.zeros((1, 2, 4), dtype=torch.int8),
        C=C,
        expert_num_tokens=torch.tensor([1], dtype=torch.int32),
        compute_type=None,
        A_scale=None,
        B_scale=torch.ones((1, 2, 2), dtype=torch.float32),
        B_zp=torch.zeros((1, 2, 2), dtype=torch.int8),
        use_fp8_w8a8=False,
        use_int8_w8a16=True,
        use_int4_w4a16=False,
        config={"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 16},
        per_act_token_quant=False,
        block_shape=[0, 2],
    )

    assert calls == {
        "use_int4_w4a16": False,
        "use_int8_w8a16": True,
        "block_shape": [0, 2],
        "has_zp": True,
    }
    assert torch.all(C == 13.0).item()


def test_invoke_batched_moe_kernel_int8_w8a16_without_zp_prefers_shared_dispatch_with_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fused_batched_moe_mod, "HAS_TRITON", True)

    calls: dict[str, object] = {}

    def _fake_try_shared(**kwargs):
        calls["use_int4_w4a16"] = kwargs["use_int4_w4a16"]
        calls["use_int8_w8a16"] = kwargs["use_int8_w8a16"]
        calls["block_shape"] = kwargs["block_shape"]
        calls["has_zp"] = kwargs["B_zp"] is not None
        kwargs["C"].fill_(17.0)
        return True

    monkeypatch.setattr(
        fused_batched_moe_mod,
        "_try_dispatch_batched_wna16_shared",
        _fake_try_shared,
    )

    def _raise_if_triton_path_runs(*_args, **_kwargs):
        raise AssertionError("batched int8_w8a16 Triton kernel path should not run")

    monkeypatch.setattr(
        fused_batched_moe_mod,
        "batched_triton_kernel",
        _raise_if_triton_path_runs,
    )

    C = torch.zeros((1, 1, 2), dtype=torch.float32)
    fused_batched_moe_mod.invoke_moe_batched_triton_kernel(
        A=torch.randn((1, 1, 4), dtype=torch.float32),
        B=torch.zeros((1, 2, 4), dtype=torch.int8),
        C=C,
        expert_num_tokens=torch.tensor([1], dtype=torch.int32),
        compute_type=None,
        A_scale=None,
        B_scale=torch.ones((1, 2, 2), dtype=torch.float32),
        B_zp=torch.empty(0, dtype=torch.int8),
        use_fp8_w8a8=False,
        use_int8_w8a16=True,
        use_int4_w4a16=False,
        config={"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 16},
        per_act_token_quant=False,
        block_shape=[0, 2],
    )

    assert calls == {
        "use_int4_w4a16": False,
        "use_int8_w8a16": True,
        "block_shape": [0, 2],
        "has_zp": False,
    }
    assert torch.all(C == 17.0).item()


def test_invoke_batched_moe_kernel_int4_w4a16_falls_back_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fused_batched_moe_mod, "HAS_TRITON", False)
    monkeypatch.setattr(
        fused_batched_moe_mod,
        "_try_dispatch_batched_wna16_shared",
        lambda **_kwargs: False,
    )

    def _raise_if_triton_path_runs(*_args, **_kwargs):
        raise AssertionError("batched int4_w4a16 Triton path should not run")

    monkeypatch.setattr(
        fused_batched_moe_mod,
        "batched_triton_kernel",
        _raise_if_triton_path_runs,
    )

    A = torch.tensor(
        [
            [[1.0, -2.0, 0.5, 3.0], [2.0, 1.0, -1.0, 0.0]],
            [[-1.5, 0.25, 2.0, -0.5], [7.0, 7.0, 7.0, 7.0]],
        ],
        dtype=torch.float32,
    )
    logical_B = torch.tensor(
        [
            [[9, 5, 12, 6], [3, 8, 14, 7]],
            [[10, 8, 4, 6], [7, 11, 5, 9]],
        ],
        dtype=torch.int32,
    )
    logical_zp = torch.tensor(
        [
            [[8, 9], [6, 7]],
            [[9, 8], [5, 6]],
        ],
        dtype=torch.int32,
    )
    B = torch.stack(
        [_pack_int4_pairs_on_last_dim_for_test(logical_B[idx]) for idx in range(2)],
        dim=0,
    )
    B_zp = torch.stack(
        [_pack_int4_pairs_on_first_dim_for_test(logical_zp[idx]) for idx in range(2)],
        dim=0,
    )
    B_scale = torch.tensor(
        [
            [[0.5, 0.25], [1.0, 0.75]],
            [[0.25, 0.5], [1.5, 0.125]],
        ],
        dtype=torch.float32,
    )
    C = torch.zeros((2, 2, 2), dtype=torch.float32)
    expert_num_tokens = torch.tensor([2, 1], dtype=torch.int32)

    fused_batched_moe_mod.invoke_moe_batched_triton_kernel(
        A=A,
        B=B,
        C=C,
        expert_num_tokens=expert_num_tokens,
        compute_type=None,
        A_scale=None,
        B_scale=B_scale,
        B_zp=B_zp,
        use_fp8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=True,
        config={"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 16},
        per_act_token_quant=False,
        block_shape=[0, 2],
    )

    expected = torch.zeros_like(C)
    for expert_idx, num_tokens in enumerate(expert_num_tokens.tolist()):
        expanded_scale = B_scale[expert_idx].repeat_interleave(2, dim=1)
        expanded_zp = logical_zp[expert_idx].to(torch.float32).repeat_interleave(
            2, dim=1
        )
        weight_dq = (logical_B[expert_idx].to(torch.float32) - expanded_zp) * expanded_scale
        expected[expert_idx, :num_tokens] = A[expert_idx, :num_tokens] @ weight_dq.transpose(
            0, 1
        )

    torch.testing.assert_close(C, expected)


def test_invoke_batched_moe_kernel_int4_w4a16_prefers_shared_dispatch_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fused_batched_moe_mod, "HAS_TRITON", False)

    calls: dict[str, object] = {}

    def _fake_try_shared(**kwargs):
        calls["use_int4_w4a16"] = kwargs["use_int4_w4a16"]
        calls["use_int8_w8a16"] = kwargs["use_int8_w8a16"]
        calls["block_shape"] = kwargs["block_shape"]
        kwargs["C"].fill_(7.0)
        return True

    monkeypatch.setattr(
        fused_batched_moe_mod,
        "_try_dispatch_batched_wna16_shared",
        _fake_try_shared,
    )
    monkeypatch.setattr(
        fused_batched_moe_mod,
        "_try_precompiled_moe_batched_kernel",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("batched precompiled kernel should not run")
        ),
    )
    monkeypatch.setattr(
        fused_batched_moe_mod,
        "_invoke_moe_batched_reference",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("batched torch reference should not run")
        ),
    )

    C = torch.zeros((1, 1, 2), dtype=torch.float32)
    fused_batched_moe_mod.invoke_moe_batched_triton_kernel(
        A=torch.randn((1, 1, 4), dtype=torch.float32),
        B=torch.zeros((1, 2, 2), dtype=torch.int8),
        C=C,
        expert_num_tokens=torch.tensor([1], dtype=torch.int32),
        compute_type=None,
        A_scale=None,
        B_scale=torch.ones((1, 2, 2), dtype=torch.float32),
        B_zp=torch.zeros((1, 1, 2), dtype=torch.int8),
        use_fp8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=True,
        config={"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 16},
        per_act_token_quant=False,
        block_shape=[0, 2],
    )

    assert calls == {
        "use_int4_w4a16": True,
        "use_int8_w8a16": False,
        "block_shape": [0, 2],
    }
    assert torch.all(C == 7.0).item()


def test_invoke_batched_moe_kernel_int4_w4a16_prefers_shared_dispatch_with_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fused_batched_moe_mod, "HAS_TRITON", True)

    calls: dict[str, object] = {}

    def _fake_try_shared(**kwargs):
        calls["use_int4_w4a16"] = kwargs["use_int4_w4a16"]
        calls["use_int8_w8a16"] = kwargs["use_int8_w8a16"]
        calls["block_shape"] = kwargs["block_shape"]
        kwargs["C"].fill_(5.0)
        return True

    monkeypatch.setattr(
        fused_batched_moe_mod,
        "_try_dispatch_batched_wna16_shared",
        _fake_try_shared,
    )

    def _raise_if_triton_path_runs(*_args, **_kwargs):
        raise AssertionError("batched int4_w4a16 Triton kernel path should not run")

    monkeypatch.setattr(
        fused_batched_moe_mod,
        "batched_triton_kernel",
        _raise_if_triton_path_runs,
    )

    C = torch.zeros((1, 1, 2), dtype=torch.float32)
    fused_batched_moe_mod.invoke_moe_batched_triton_kernel(
        A=torch.randn((1, 1, 4), dtype=torch.float32),
        B=torch.zeros((1, 2, 2), dtype=torch.int8),
        C=C,
        expert_num_tokens=torch.tensor([1], dtype=torch.int32),
        compute_type=None,
        A_scale=None,
        B_scale=torch.ones((1, 2, 2), dtype=torch.float32),
        B_zp=torch.zeros((1, 1, 2), dtype=torch.int8),
        use_fp8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=True,
        config={"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 16},
        per_act_token_quant=False,
        block_shape=[0, 2],
    )

    assert calls == {
        "use_int4_w4a16": True,
        "use_int8_w8a16": False,
        "block_shape": [0, 2],
    }
    assert torch.all(C == 5.0).item()


@pytest.mark.parametrize(
    ("use_int8_w8a16", "use_int4_w4a16"),
    [(True, False), (False, True)],
)
def test_try_dispatch_batched_wna16_shared_builds_padded_payload(
    monkeypatch,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
) -> None:
    class _FakeCudaTensor(torch.Tensor):
        @property
        def is_cuda(self) -> bool:
            return True

    monkeypatch.setattr(fused_batched_moe_mod, "HAS_TRITON", False)
    monkeypatch.setattr(
        fused_batched_moe_mod,
        "_has_moe_wna16_gemm_op",
        lambda: True,
    )

    calls: list[dict[str, object]] = []

    def _fake_dispatch(
        *,
        C: torch.Tensor,
        sorted_token_ids: torch.Tensor | None,
        expert_ids: torch.Tensor,
        num_tokens_post_padded: torch.Tensor,
        **kwargs,
    ) -> None:
        calls.append(
            {
                "sorted_token_ids": None
                if sorted_token_ids is None
                else sorted_token_ids.clone(),
                "expert_ids": expert_ids.clone(),
                "num_tokens_post_padded": num_tokens_post_padded.clone(),
                "use_int8_w8a16": kwargs["use_int8_w8a16"],
                "use_int4_w4a16": kwargs["use_int4_w4a16"],
            }
        )
        C.fill_(1.0)

    monkeypatch.setattr(
        fused_batched_moe_mod,
        "dispatch_fused_moe_kernel",
        _fake_dispatch,
    )

    A = torch.randn((2, 5, 4), dtype=torch.float32).as_subclass(_FakeCudaTensor)
    B = torch.zeros((2, 3, 4), dtype=torch.int8)
    C = torch.zeros((2, 5, 3), dtype=torch.float32)

    ok = fused_batched_moe_mod._try_dispatch_batched_wna16_shared(
        A=A,
        B=B,
        C=C,
        expert_num_tokens=torch.tensor([5, 1], dtype=torch.int32),
        B_scale=torch.ones((2, 3, 2), dtype=torch.float32),
        B_zp=torch.zeros((2, 3, 2), dtype=torch.int8),
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        config={"BLOCK_SIZE_M": 4, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 16},
        compute_type=None,
        block_shape=[0, 2],
    )

    assert ok is True
    assert len(calls) == 2
    assert calls[0]["use_int8_w8a16"] is use_int8_w8a16
    assert calls[0]["use_int4_w4a16"] is use_int4_w4a16
    assert calls[0]["sorted_token_ids"].tolist() == [0, 1, 2, 3, 4, 5, 5, 5]
    assert calls[0]["expert_ids"].tolist() == [0, 0]
    assert calls[0]["num_tokens_post_padded"].tolist() == [8]
    assert calls[1]["sorted_token_ids"].tolist() == [0, 1, 1, 1]
    assert calls[1]["expert_ids"].tolist() == [0]
    assert calls[1]["num_tokens_post_padded"].tolist() == [4]


@pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn"),
    reason="FP8 dtype unavailable",
)
def test_invoke_batched_moe_kernel_fp8_prefers_precompiled_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fused_batched_moe_mod, "HAS_TRITON", False)

    calls: dict[str, object] = {}

    def _fake_precompiled(**kwargs):
        calls["use_fp8_w8a8"] = kwargs["use_fp8_w8a8"]
        calls["per_act_token_quant"] = kwargs["per_act_token_quant"]
        calls["a_scale_shape"] = (
            None if kwargs["A_scale"] is None else tuple(kwargs["A_scale"].shape)
        )
        calls["b_scale_shape"] = (
            None if kwargs["B_scale"] is None else tuple(kwargs["B_scale"].shape)
        )
        kwargs["C"].fill_(9.0)
        return True

    monkeypatch.setattr(
        fused_batched_moe_mod,
        "_try_precompiled_moe_batched_kernel",
        _fake_precompiled,
    )
    monkeypatch.setattr(
        fused_batched_moe_mod,
        "_invoke_moe_batched_reference",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("torch fallback should not run")
        ),
    )

    fp8 = torch.float8_e4m3fn
    A = torch.randn(2, 2, 4, dtype=torch.float16).to(fp8)
    B = torch.randn(2, 3, 4, dtype=torch.float16).to(fp8)
    C = torch.zeros((2, 2, 3), dtype=torch.float32)
    expert_num_tokens = torch.tensor([2, 1], dtype=torch.int32)
    A_scale = torch.ones((2, 2, 1), dtype=torch.float32)
    B_scale = torch.ones((2, 1, 1), dtype=torch.float32)

    fused_batched_moe_mod.invoke_moe_batched_triton_kernel(
        A=A,
        B=B,
        C=C,
        expert_num_tokens=expert_num_tokens,
        compute_type=None,
        A_scale=A_scale,
        B_scale=B_scale,
        B_zp=torch.empty(0),
        use_fp8_w8a8=True,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        config={"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 16},
        per_act_token_quant=True,
        block_shape=None,
    )

    assert calls == {
        "use_fp8_w8a8": True,
        "per_act_token_quant": True,
        "a_scale_shape": (2, 2, 1),
        "b_scale_shape": (2, 1, 1),
    }
    assert torch.all(C == 9.0).item()


def test_gpt_oss_oai_triton_experts_reject_without_triton_kernels(
    monkeypatch,
) -> None:
    monkeypatch.setattr(gpt_oss_triton_moe_mod, "HAS_TRITON_KERNELS", False)
    monkeypatch.setattr(gpt_oss_triton_moe_mod.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(gpt_oss_triton_moe_mod.current_platform, "is_rocm", lambda: False)

    moe_config = SimpleNamespace(
        is_act_and_mul=True,
        activation=fused_moe_activation.MoEActivation.SWIGLUOAI,
        moe_parallel_config=SimpleNamespace(use_fi_all2allv_kernels=False),
        routing_method=None,
        router_logits_dtype=None,
        hidden_dim=128,
    )

    supported, reason = gpt_oss_triton_moe_mod.OAITritonExperts.is_supported_config(
        gpt_oss_triton_moe_mod.OAITritonExperts,
        moe_config,
        None,
        None,
        gpt_oss_triton_moe_mod.OAITritonExperts.activation_format(),
    )

    assert not supported
    assert reason is not None
    assert "current device" in reason


def test_gpt_oss_make_routing_data_rejects_without_triton_kernels(
    monkeypatch,
) -> None:
    monkeypatch.setattr(gpt_oss_triton_moe_mod, "HAS_TRITON_KERNELS", False)

    with pytest.raises(RuntimeError, match="requires Triton \\+ triton_kernels"):
        gpt_oss_triton_moe_mod.make_routing_data(
            torch.tensor([[0]], dtype=torch.int64),
            torch.tensor([[1.0]], dtype=torch.float32),
            1,
        )


def test_gpt_oss_triton_kernel_moe_forward_rejects_without_triton_kernels(
    monkeypatch,
) -> None:
    monkeypatch.setattr(gpt_oss_triton_moe_mod, "HAS_TRITON_KERNELS", False)

    with pytest.raises(RuntimeError, match="requires Triton \\+ triton_kernels"):
        gpt_oss_triton_moe_mod.triton_kernel_moe_forward(
            hidden_states=torch.zeros((2, 2), dtype=torch.bfloat16),
            w1=torch.zeros((2, 2, 4), dtype=torch.bfloat16),
            w2=torch.zeros((2, 2, 2), dtype=torch.bfloat16),
            gating_output=torch.zeros((2, 2), dtype=torch.float32),
            topk=1,
            renormalize=False,
            activation=fused_moe_activation.MoEActivation.SWIGLUOAI,
            quant_config=None,
        )


def test_gpt_oss_triton_kernel_fused_experts_rejects_without_triton_kernels(
    monkeypatch,
) -> None:
    monkeypatch.setattr(gpt_oss_triton_moe_mod, "HAS_TRITON_KERNELS", False)

    with pytest.raises(RuntimeError, match="requires Triton \\+ triton_kernels"):
        gpt_oss_triton_moe_mod.triton_kernel_fused_experts(
            output_tensor=torch.zeros((2, 2), dtype=torch.bfloat16),
            hidden_states=torch.zeros((2, 2), dtype=torch.bfloat16),
            w1=torch.zeros((2, 2, 4), dtype=torch.bfloat16),
            w2=torch.zeros((2, 2, 2), dtype=torch.bfloat16),
            routing_data=None,
            gather_indx=None,
            scatter_indx=None,
            topk=1,
            activation=fused_moe_activation.MoEActivation.SWIGLUOAI,
            quant_config=FUSED_MOE_UNQUANTIZED_CONFIG,
        )


@pytest.mark.parametrize(
    ("experts_cls", "label"),
    [
        (gpt_oss_triton_moe_mod.OAITritonExperts, "fused"),
        (gpt_oss_triton_moe_mod.UnfusedOAITritonExperts, "unfused"),
    ],
)
def test_gpt_oss_experts_apply_rejects_without_triton_kernels(
    monkeypatch,
    experts_cls,
    label: str,
) -> None:
    monkeypatch.setattr(gpt_oss_triton_moe_mod, "HAS_TRITON_KERNELS", False)

    experts = experts_cls(
        moe_config=SimpleNamespace(
            is_act_and_mul=True,
            activation=fused_moe_activation.MoEActivation.SWIGLUOAI,
            moe_parallel_config=SimpleNamespace(use_fi_all2allv_kernels=False),
            routing_method=None,
            router_logits_dtype=None,
            hidden_dim=2,
        ),
        quant_config=FUSED_MOE_UNQUANTIZED_CONFIG,
    )

    with pytest.raises(
        RuntimeError,
        match="requires Triton \\+ triton_kernels runtime",
    ):
        experts.apply(
            output=torch.zeros((2, 2), dtype=torch.bfloat16),
            hidden_states=torch.zeros((2, 2), dtype=torch.bfloat16),
            w1=torch.zeros((2, 2, 4), dtype=torch.bfloat16),
            w2=torch.zeros((2, 2, 2), dtype=torch.bfloat16),
            topk_weights=torch.ones((2, 1), dtype=torch.float32),
            topk_ids=torch.tensor([[0], [1]], dtype=torch.int32),
            activation=fused_moe_activation.MoEActivation.SWIGLUOAI,
            global_num_experts=2,
            expert_map=None,
            a1q_scale=None,
            a2_scale=None,
            workspace13=torch.empty((2, 2), dtype=torch.bfloat16),
            workspace2=torch.empty((2, 4), dtype=torch.bfloat16),
            expert_tokens_meta=None,
            apply_router_weight_on_input=False,
        )


def test_mxfp4_backend_with_lora_prefers_marlin_without_triton_kernels(
    monkeypatch,
) -> None:
    monkeypatch.setattr(mxfp4_mod.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(mxfp4_mod, "has_triton_kernels", lambda: False)
    monkeypatch.setattr(
        mxfp4_mod.current_platform,
        "get_device_capability",
        lambda: (9, 0),
    )
    monkeypatch.setattr(mxfp4_mod.envs, "VLLM_MXFP4_USE_MARLIN", False)

    backend = mxfp4_mod.get_mxfp4_backend_with_lora()

    assert backend == mxfp4_mod.Mxfp4Backend.MARLIN


def test_mxfp4_backend_prefers_marlin_without_triton_kernels(
    monkeypatch,
) -> None:
    monkeypatch.setattr(mxfp4_mod.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(mxfp4_mod.current_platform, "is_xpu", lambda: False)
    monkeypatch.setattr(mxfp4_mod.current_platform, "is_rocm", lambda: False)
    monkeypatch.setattr(
        mxfp4_mod.current_platform,
        "is_device_capability",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        mxfp4_mod.current_platform,
        "is_device_capability_family",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(mxfp4_mod, "has_flashinfer", lambda: False)
    monkeypatch.setattr(mxfp4_mod, "has_triton_kernels", lambda: False)
    monkeypatch.setattr(mxfp4_mod.envs, "VLLM_MXFP4_USE_MARLIN", False)

    backend = mxfp4_mod.get_mxfp4_backend(with_lora_support=False)

    assert backend == mxfp4_mod.Mxfp4Backend.MARLIN


@pytest.mark.parametrize(
    ("use_int8_w8a16", "use_int4_w4a16", "expected_bit"),
    [(True, False, 8), (False, True, 4)],
)
def test_fused_moe_wna16_dispatch_prefers_precompiled_without_triton(
    monkeypatch,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    expected_bit: int,
) -> None:
    monkeypatch.setattr(fused_moe_mod, "HAS_TRITON", False)
    monkeypatch.setattr(fused_moe_mod, "_has_moe_wna16_gemm_op", lambda: True)
    monkeypatch.setattr(
        fused_moe_mod,
        "should_moe_wna16_use_cuda",
        lambda **_kwargs: False,
    )

    calls: dict[str, object] = {"cuda": 0}

    def _fake_invoke_cuda(*_args, **_kwargs):
        calls["cuda"] += 1
        calls["bit"] = _kwargs.get("bit", _args[-1])

    def _raise_if_triton_path_runs(*_args, **_kwargs):
        raise AssertionError("WNA16 Triton path should not run")

    monkeypatch.setattr(
        fused_moe_mod,
        "invoke_fused_moe_wna16_cuda_kernel",
        _fake_invoke_cuda,
    )
    monkeypatch.setattr(
        fused_moe_mod,
        "invoke_fused_moe_wna16_triton_kernel",
        _raise_if_triton_path_runs,
    )

    fused_moe_mod.dispatch_fused_moe_kernel(
        A=torch.randn((1, 4), dtype=torch.float16),
        B=torch.randint(-8, 8, (1, 8, 4), dtype=torch.int8),
        C=torch.empty((1, 1, 8), dtype=torch.float16),
        A_scale=None,
        B_scale=torch.ones((1, 1, 8), dtype=torch.float16),
        B_zp=None,
        topk_weights=torch.ones((1, 1), dtype=torch.float32),
        sorted_token_ids=torch.tensor([0], dtype=torch.int32),
        expert_ids=torch.tensor([0], dtype=torch.int32),
        num_tokens_post_padded=torch.tensor([1], dtype=torch.int32),
        mul_routed_weight=True,
        top_k=1,
        config={"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 16},
        compute_type=None,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        per_channel_quant=False,
        block_shape=[0, 128],
        B_bias=None,
    )

    assert calls == {"cuda": 1, "bit": expected_bit}


@pytest.mark.parametrize("bit", [4, 8])
def test_invoke_fused_moe_wna16_cuda_kernel_forwards_bit_to_precompiled_op(
    monkeypatch,
    bit: int,
) -> None:
    config_calls: dict[str, object] = {}

    monkeypatch.setattr(
        fused_moe_mod,
        "get_moe_wna16_block_config",
        lambda **kwargs: config_calls.update(kwargs)
        or {
            "BLOCK_SIZE_M": kwargs["block_size_m"],
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 64,
        },
    )

    calls: dict[str, object] = {}

    def _fake_moe_wna16_gemm(
        A: torch.Tensor,
        C: torch.Tensor,
        B: torch.Tensor,
        B_scale: torch.Tensor,
        B_zp: torch.Tensor | None,
        topk_weights: torch.Tensor | None,
        sorted_token_ids: torch.Tensor,
        expert_ids: torch.Tensor,
        num_tokens_post_padded: torch.Tensor,
        top_k: int,
        block_size_m: int,
        block_size_n: int,
        block_size_k: int,
        forwarded_bit: int,
    ) -> None:
        calls["top_k"] = top_k
        calls["block_sizes"] = (block_size_m, block_size_n, block_size_k)
        calls["bit"] = forwarded_bit
        calls["topk_weights_is_none"] = topk_weights is None

    monkeypatch.setattr(fused_moe_mod.ops, "moe_wna16_gemm", _fake_moe_wna16_gemm)

    fused_moe_mod.invoke_fused_moe_wna16_cuda_kernel(
        A=torch.randn((1, 4), dtype=torch.float16),
        B=torch.randint(-8, 8, (1, 8, 4), dtype=torch.int8),
        C=torch.empty((1, 1, 8), dtype=torch.float16),
        B_scale=torch.ones((1, 1, 8), dtype=torch.float16),
        B_zp=None,
        topk_weights=torch.ones((1, 1), dtype=torch.float32),
        sorted_token_ids=torch.tensor([0], dtype=torch.int32),
        expert_ids=torch.tensor([0], dtype=torch.int32),
        num_tokens_post_padded=torch.tensor([1], dtype=torch.int32),
        mul_routed_weight=True,
        top_k=1,
        config={"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 16},
        block_shape=[0, 128],
        bit=bit,
    )

    assert calls == {
        "top_k": 1,
        "block_sizes": (16, 32, 64),
        "bit": bit,
        "topk_weights_is_none": False,
    }
    assert config_calls["num_experts"] == 1


def test_invoke_fused_moe_wna16_triton_kernel_uses_expert_dim_for_block_config(
    monkeypatch,
) -> None:
    config_calls: dict[str, object] = {}

    monkeypatch.setattr(
        fused_moe_mod,
        "get_moe_wna16_block_config",
        lambda **kwargs: config_calls.update(kwargs)
        or {
            "BLOCK_SIZE_M": kwargs["block_size_m"],
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 64,
        },
    )
    monkeypatch.setattr(
        fused_moe_mod,
        "triton",
        SimpleNamespace(cdiv=lambda x, y: (x + y - 1) // y),
    )

    calls: dict[str, object] = {}

    class _FakeKernel:
        def __getitem__(self, grid):
            def _runner(*args, **kwargs):
                calls["grid"] = grid(kwargs)
                calls["use_int8_w8a16"] = kwargs["use_int8_w8a16"]
                calls["use_int4_w4a16"] = kwargs["use_int4_w4a16"]
                calls["block_size_n"] = kwargs["BLOCK_SIZE_N"]
                calls["block_size_k"] = kwargs["BLOCK_SIZE_K"]

            return _runner

    monkeypatch.setattr(fused_moe_mod, "fused_moe_kernel_gptq_awq", _FakeKernel())

    fused_moe_mod.invoke_fused_moe_wna16_triton_kernel(
        A=torch.randn((1, 4), dtype=torch.float16),
        B=torch.randint(-8, 8, (1, 8, 4), dtype=torch.int8),
        C=torch.empty((1, 1, 8), dtype=torch.float16),
        B_scale=torch.ones((1, 1, 8), dtype=torch.float16),
        B_zp=None,
        topk_weights=torch.ones((1, 1), dtype=torch.float32),
        sorted_token_ids=torch.tensor([0], dtype=torch.int32),
        expert_ids=torch.tensor([0], dtype=torch.int32),
        num_tokens_post_padded=torch.tensor([1], dtype=torch.int32),
        mul_routed_weight=True,
        top_k=1,
        config={"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 16},
        compute_type=None,
        use_int8_w8a16=True,
        use_int4_w4a16=False,
        block_shape=[0, 128],
    )

    assert config_calls["num_experts"] == 1
    assert calls == {
        "grid": (1,),
        "use_int8_w8a16": True,
        "use_int4_w4a16": False,
        "block_size_n": 32,
        "block_size_k": 64,
    }


def test_get_moe_wna16_block_config_scales_num_blocks_after_promoting_block_k() -> None:
    config = fused_moe_mod.get_moe_wna16_block_config(
        config={"BLOCK_SIZE_M": 16},
        use_moe_wna16_cuda=True,
        num_valid_tokens=256,
        size_k=256,
        size_n=4096,
        num_experts=1,
        group_size=128,
        real_top_k=1,
        block_size_m=16,
    )

    assert config == {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256}


def test_fused_batched_moe_apply_int8_w8a16_falls_back_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fused_batched_moe_mod, "HAS_TRITON", False)

    def _raise_if_triton_path_runs(*_args, **_kwargs):
        raise AssertionError("batched int8_w8a16 Triton path should not run")

    monkeypatch.setattr(
        fused_batched_moe_mod,
        "batched_triton_kernel",
        _raise_if_triton_path_runs,
    )

    experts = fused_batched_moe_mod.BatchedTritonExperts(
        moe_config=SimpleNamespace(
            is_act_and_mul=False,
            activation=fused_moe_activation.MoEActivation.RELU2_NO_MUL,
            moe_parallel_config=SimpleNamespace(use_fi_all2allv_kernels=False),
            routing_method=None,
            router_logits_dtype=None,
            hidden_dim=2,
        ),
        quant_config=int8_w8a16_moe_quant_config(
            w1_scale=torch.tensor([[[0.5]], [[1.5]]], dtype=torch.float32),
            w2_scale=torch.tensor([[[2.0]], [[0.25]]], dtype=torch.float32),
            w1_zp=None,
            w2_zp=None,
            block_shape=None,
        ),
        max_num_tokens=3,
        num_dispatchers=1,
    )

    hidden_states = torch.tensor(
        [
            [[1.0, -2.0], [0.5, 3.0], [7.0, 7.0]],
            [[-1.5, 0.25], [8.0, 8.0], [8.0, 8.0]],
        ],
        dtype=torch.float32,
    )
    w1 = torch.tensor(
        [
            [[1, 0], [-1, 2]],
            [[1, -2], [2, 1]],
        ],
        dtype=torch.int8,
    )
    w2 = torch.tensor(
        [
            [[1, -1], [0, 2]],
            [[-2, 1], [1, 0]],
        ],
        dtype=torch.int8,
    )
    output = torch.zeros((2, 3, 2), dtype=torch.float32)
    workspace13 = torch.empty((2, 3, 2), dtype=torch.float32)
    workspace2 = torch.empty((2, 3, 2), dtype=torch.float32)
    expert_num_tokens = torch.tensor([2, 1], dtype=torch.int32)

    experts.apply(
        output=output,
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=torch.ones((1, 1), dtype=torch.float32),
        topk_ids=torch.zeros((1, 1), dtype=torch.int32),
        activation=fused_moe_activation.MoEActivation.RELU2_NO_MUL,
        global_num_experts=2,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=workspace13,
        workspace2=workspace2,
        expert_tokens_meta=SimpleNamespace(
            expert_num_tokens=expert_num_tokens,
            expert_num_tokens_cpu=None,
        ),
        apply_router_weight_on_input=False,
    )

    w1_dq = w1.to(torch.float32) * torch.tensor(
        [[[0.5]], [[1.5]]], dtype=torch.float32
    )
    w2_dq = w2.to(torch.float32) * torch.tensor(
        [[[2.0]], [[0.25]]], dtype=torch.float32
    )

    expected_intermediate1 = torch.zeros((2, 3, 2), dtype=torch.float32)
    expected_intermediate1[0, :2] = hidden_states[0, :2] @ w1_dq[0].transpose(0, 1)
    expected_intermediate1[1, :1] = hidden_states[1, :1] @ w1_dq[1].transpose(0, 1)

    expected_intermediate2 = torch.zeros_like(expected_intermediate1)
    fused_moe_activation.apply_moe_activation(
        fused_moe_activation.MoEActivation.RELU2_NO_MUL,
        expected_intermediate2.view(-1, expected_intermediate2.size(-1)),
        expected_intermediate1.view(-1, expected_intermediate1.size(-1)),
    )

    expected = torch.zeros_like(output)
    expected[0, :2] = expected_intermediate2[0, :2] @ w2_dq[0].transpose(0, 1)
    expected[1, :1] = expected_intermediate2[1, :1] @ w2_dq[1].transpose(0, 1)

    torch.testing.assert_close(output, expected)


def test_naive_batched_experts_apply_int8_w8a16_with_zp_reference() -> None:
    experts = fused_batched_moe_mod.NaiveBatchedExperts(
        moe_config=SimpleNamespace(
            is_act_and_mul=False,
            activation=fused_moe_activation.MoEActivation.RELU2_NO_MUL,
            moe_parallel_config=SimpleNamespace(use_fi_all2allv_kernels=False),
            routing_method=None,
            router_logits_dtype=None,
            hidden_dim=4,
        ),
        quant_config=int8_w8a16_moe_quant_config(
            w1_scale=torch.tensor(
                [
                    [[0.5, 0.25], [1.0, 0.75]],
                    [[0.25, 0.5], [1.5, 0.125]],
                ],
                dtype=torch.float32,
            ),
            w2_scale=torch.tensor(
                [
                    [[0.5], [0.25], [1.0], [0.75]],
                    [[0.125], [0.5], [0.25], [1.0]],
                ],
                dtype=torch.float32,
            ),
            w1_zp=torch.tensor(
                [
                    [[8, 9], [6, 7]],
                    [[9, 8], [5, 6]],
                ],
                dtype=torch.int8,
            ),
            w2_zp=torch.tensor(
                [
                    [[8], [7], [6], [9]],
                    [[7], [8], [5], [6]],
                ],
                dtype=torch.int8,
            ),
            block_shape=[0, 2],
        ),
        max_num_tokens=3,
        num_dispatchers=1,
    )

    hidden_states = torch.tensor(
        [
            [[1.0, -2.0, 0.5, 3.0], [2.0, 1.0, -1.0, 0.0], [7.0, 7.0, 7.0, 7.0]],
            [[-1.5, 0.25, 2.0, -0.5], [8.0, 8.0, 8.0, 8.0], [8.0, 8.0, 8.0, 8.0]],
        ],
        dtype=torch.float32,
    )
    w1 = torch.tensor(
        [
            [[9, 5, 12, 6], [3, 8, 14, 7]],
            [[10, 8, 4, 6], [7, 11, 5, 9]],
        ],
        dtype=torch.int8,
    )
    w2 = torch.tensor(
        [
            [[9, 7], [8, 6], [10, 5], [11, 9]],
            [[7, 10], [9, 8], [6, 11], [5, 12]],
        ],
        dtype=torch.int8,
    )
    output = torch.zeros((2, 3, 4), dtype=torch.float32)
    workspace13 = torch.empty((2, 3, 4), dtype=torch.float32)
    workspace2 = torch.empty((2, 3, 2), dtype=torch.float32)
    expert_num_tokens = torch.tensor([2, 1], dtype=torch.int32)

    experts.apply(
        output=output,
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=torch.ones((1, 1), dtype=torch.float32),
        topk_ids=torch.zeros((1, 1), dtype=torch.int32),
        activation=fused_moe_activation.MoEActivation.RELU2_NO_MUL,
        global_num_experts=2,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=workspace13,
        workspace2=workspace2,
        expert_tokens_meta=SimpleNamespace(
            expert_num_tokens=expert_num_tokens,
            expert_num_tokens_cpu=None,
        ),
        apply_router_weight_on_input=False,
    )

    logical_w1_zp = torch.tensor(
        [
            [[8, 9], [6, 7]],
            [[9, 8], [5, 6]],
        ],
        dtype=torch.float32,
    )
    logical_w2_zp = torch.tensor(
        [
            [[8], [7], [6], [9]],
            [[7], [8], [5], [6]],
        ],
        dtype=torch.float32,
    )
    w1_dq = (
        w1.to(torch.float32)
        - logical_w1_zp.repeat_interleave(2, dim=2)
    ) * experts.w1_scale.repeat_interleave(2, dim=2)
    w2_dq = (
        w2.to(torch.float32)
        - logical_w2_zp.repeat_interleave(2, dim=2)
    ) * experts.w2_scale.repeat_interleave(2, dim=2)

    expected_intermediate1 = torch.zeros((2, 3, 2), dtype=torch.float32)
    expected_intermediate1[0, :2] = hidden_states[0, :2] @ w1_dq[0].transpose(0, 1)
    expected_intermediate1[1, :1] = hidden_states[1, :1] @ w1_dq[1].transpose(0, 1)

    expected_intermediate2 = torch.zeros_like(expected_intermediate1)
    fused_moe_activation.apply_moe_activation(
        fused_moe_activation.MoEActivation.RELU2_NO_MUL,
        expected_intermediate2.view(-1, expected_intermediate2.size(-1)),
        expected_intermediate1.view(-1, expected_intermediate1.size(-1)),
    )

    expected = torch.zeros_like(output)
    expected[0, :2] = expected_intermediate2[0, :2] @ w2_dq[0].transpose(0, 1)
    expected[1, :1] = expected_intermediate2[1, :1] @ w2_dq[1].transpose(0, 1)

    torch.testing.assert_close(output, expected)


def test_fused_batched_moe_apply_int8_w8a16_with_zp_prefers_shared_dispatch_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fused_batched_moe_mod, "HAS_TRITON", False)
    monkeypatch.setattr(
        fused_batched_moe_mod,
        "try_get_optimal_moe_config",
        lambda *_args, **_kwargs: {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 16,
            "BLOCK_SIZE_K": 16,
        },
    )

    calls: list[dict[str, object]] = []

    def _fake_try_shared(**kwargs):
        calls.append(
            {
                "use_int8_w8a16": kwargs["use_int8_w8a16"],
                "use_int4_w4a16": kwargs["use_int4_w4a16"],
                "has_zp": kwargs["B_zp"] is not None,
                "block_shape": kwargs["block_shape"],
                "shape_C": tuple(kwargs["C"].shape),
            }
        )
        if len(calls) == 1:
            kwargs["C"].copy_(
                torch.tensor(
                    [
                        [[-1.0, 2.0], [0.5, -0.5], [0.0, 0.0]],
                        [[3.0, -4.0], [0.0, 0.0], [0.0, 0.0]],
                    ],
                    dtype=kwargs["C"].dtype,
                    device=kwargs["C"].device,
                )
            )
        elif len(calls) == 2:
            kwargs["C"].copy_(
                torch.tensor(
                    [
                        [[3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0], [0.0, 0.0, 0.0, 0.0]],
                        [[11.0, 12.0, 13.0, 14.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                    ],
                    dtype=kwargs["C"].dtype,
                    device=kwargs["C"].device,
                )
            )
        else:
            raise AssertionError("_try_dispatch_batched_wna16_shared should run exactly twice")
        return True

    def _raise_if_triton_path_runs(*_args, **_kwargs):
        raise AssertionError("batched int8_w8a16 Triton path should not run")

    monkeypatch.setattr(
        fused_batched_moe_mod,
        "_try_dispatch_batched_wna16_shared",
        _fake_try_shared,
    )
    monkeypatch.setattr(
        fused_batched_moe_mod,
        "batched_triton_kernel",
        _raise_if_triton_path_runs,
    )

    experts = fused_batched_moe_mod.BatchedTritonExperts(
        moe_config=SimpleNamespace(
            is_act_and_mul=False,
            activation=fused_moe_activation.MoEActivation.RELU2_NO_MUL,
            moe_parallel_config=SimpleNamespace(use_fi_all2allv_kernels=False),
            routing_method=None,
            router_logits_dtype=None,
            hidden_dim=4,
        ),
        quant_config=int8_w8a16_moe_quant_config(
            w1_scale=torch.ones((2, 2, 2), dtype=torch.float32),
            w2_scale=torch.ones((2, 4, 1), dtype=torch.float32),
            w1_zp=torch.zeros((2, 2, 2), dtype=torch.int8),
            w2_zp=torch.zeros((2, 4, 2), dtype=torch.int8),
            block_shape=[0, 2],
        ),
        max_num_tokens=3,
        num_dispatchers=1,
    )

    hidden_states = torch.tensor(
        [
            [[1.0, -2.0, 0.5, 3.0], [0.5, 3.0, -1.0, 0.0], [7.0, 7.0, 7.0, 7.0]],
            [[-1.5, 0.25, 2.0, -0.5], [8.0, 8.0, 8.0, 8.0], [8.0, 8.0, 8.0, 8.0]],
        ],
        dtype=torch.float32,
    )
    w1 = torch.zeros((2, 2, 4), dtype=torch.int8)
    w2 = torch.zeros((2, 4, 2), dtype=torch.int8)
    output = torch.zeros((2, 3, 4), dtype=torch.float32)
    workspace13 = torch.empty((2, 3, 4), dtype=torch.float32)
    workspace2 = torch.empty((2, 3, 2), dtype=torch.float32)
    expert_num_tokens = torch.tensor([2, 1], dtype=torch.int32)

    experts.apply(
        output=output,
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=torch.ones((1, 1), dtype=torch.float32),
        topk_ids=torch.zeros((1, 1), dtype=torch.int32),
        activation=fused_moe_activation.MoEActivation.RELU2_NO_MUL,
        global_num_experts=2,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=workspace13,
        workspace2=workspace2,
        expert_tokens_meta=SimpleNamespace(
            expert_num_tokens=expert_num_tokens,
            expert_num_tokens_cpu=None,
        ),
        apply_router_weight_on_input=False,
    )

    assert len(calls) == 2
    assert all(call["use_int8_w8a16"] is True for call in calls)
    assert all(call["use_int4_w4a16"] is False for call in calls)
    assert all(call["has_zp"] is True for call in calls)
    assert all(call["block_shape"] == [0, 2] for call in calls)
    torch.testing.assert_close(
        output,
        torch.tensor(
            [
                [[3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0], [0.0, 0.0, 0.0, 0.0]],
                [[11.0, 12.0, 13.0, 14.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            ],
            dtype=torch.float32,
        ),
    )


def test_fused_batched_moe_apply_int8_w8a16_without_zp_prefers_shared_dispatch_with_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fused_batched_moe_mod, "HAS_TRITON", True)
    monkeypatch.setattr(
        fused_batched_moe_mod,
        "tl",
        SimpleNamespace(float16="float16", float32="float32", bfloat16="bfloat16"),
    )
    monkeypatch.setattr(
        fused_batched_moe_mod,
        "try_get_optimal_moe_config",
        lambda *_args, **_kwargs: {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 16,
            "BLOCK_SIZE_K": 16,
        },
    )

    calls: list[dict[str, object]] = []

    def _fake_try_shared(**kwargs):
        calls.append(
            {
                "use_int8_w8a16": kwargs["use_int8_w8a16"],
                "use_int4_w4a16": kwargs["use_int4_w4a16"],
                "has_zp": kwargs["B_zp"] is not None,
                "block_shape": kwargs["block_shape"],
                "shape_C": tuple(kwargs["C"].shape),
            }
        )
        if len(calls) == 1:
            kwargs["C"].copy_(
                torch.tensor(
                    [
                        [[-1.0, 2.0], [0.5, -0.5], [0.0, 0.0]],
                        [[3.0, -4.0], [0.0, 0.0], [0.0, 0.0]],
                    ],
                    dtype=kwargs["C"].dtype,
                    device=kwargs["C"].device,
                )
            )
        elif len(calls) == 2:
            kwargs["C"].copy_(
                torch.tensor(
                    [
                        [[3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0], [0.0, 0.0, 0.0, 0.0]],
                        [[11.0, 12.0, 13.0, 14.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                    ],
                    dtype=kwargs["C"].dtype,
                    device=kwargs["C"].device,
                )
            )
        else:
            raise AssertionError("_try_dispatch_batched_wna16_shared should run exactly twice")
        return True

    def _raise_if_triton_path_runs(*_args, **_kwargs):
        raise AssertionError("batched int8_w8a16 Triton path should not run")

    monkeypatch.setattr(
        fused_batched_moe_mod,
        "_try_dispatch_batched_wna16_shared",
        _fake_try_shared,
    )
    monkeypatch.setattr(
        fused_batched_moe_mod,
        "batched_triton_kernel",
        _raise_if_triton_path_runs,
    )

    experts = fused_batched_moe_mod.BatchedTritonExperts(
        moe_config=SimpleNamespace(
            is_act_and_mul=False,
            activation=fused_moe_activation.MoEActivation.RELU2_NO_MUL,
            moe_parallel_config=SimpleNamespace(use_fi_all2allv_kernels=False),
            routing_method=None,
            router_logits_dtype=None,
            hidden_dim=4,
        ),
        quant_config=int8_w8a16_moe_quant_config(
            w1_scale=torch.ones((2, 2, 2), dtype=torch.float32),
            w2_scale=torch.ones((2, 4, 1), dtype=torch.float32),
            w1_zp=None,
            w2_zp=None,
            block_shape=[0, 2],
        ),
        max_num_tokens=3,
        num_dispatchers=1,
    )

    hidden_states = torch.tensor(
        [
            [[1.0, -2.0, 0.5, 3.0], [0.5, 3.0, -1.0, 0.0], [7.0, 7.0, 7.0, 7.0]],
            [[-1.5, 0.25, 2.0, -0.5], [8.0, 8.0, 8.0, 8.0], [8.0, 8.0, 8.0, 8.0]],
        ],
        dtype=torch.float32,
    )
    w1 = torch.zeros((2, 2, 4), dtype=torch.int8)
    w2 = torch.zeros((2, 4, 2), dtype=torch.int8)
    output = torch.zeros((2, 3, 4), dtype=torch.float32)
    workspace13 = torch.empty((2, 3, 4), dtype=torch.float32)
    workspace2 = torch.empty((2, 3, 2), dtype=torch.float32)
    expert_num_tokens = torch.tensor([2, 1], dtype=torch.int32)

    experts.apply(
        output=output,
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=torch.ones((1, 1), dtype=torch.float32),
        topk_ids=torch.zeros((1, 1), dtype=torch.int32),
        activation=fused_moe_activation.MoEActivation.RELU2_NO_MUL,
        global_num_experts=2,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=workspace13,
        workspace2=workspace2,
        expert_tokens_meta=SimpleNamespace(
            expert_num_tokens=expert_num_tokens,
            expert_num_tokens_cpu=None,
        ),
        apply_router_weight_on_input=False,
    )

    assert len(calls) == 2
    assert all(call["use_int8_w8a16"] is True for call in calls)
    assert all(call["use_int4_w4a16"] is False for call in calls)
    assert all(call["has_zp"] is False for call in calls)
    assert all(call["block_shape"] == [0, 2] for call in calls)
    torch.testing.assert_close(
        output,
        torch.tensor(
            [
                [[3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0], [0.0, 0.0, 0.0, 0.0]],
                [[11.0, 12.0, 13.0, 14.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            ],
            dtype=torch.float32,
        ),
    )


def test_fused_batched_moe_apply_int4_w4a16_prefers_shared_dispatch_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fused_batched_moe_mod, "HAS_TRITON", False)
    monkeypatch.setattr(
        fused_batched_moe_mod,
        "try_get_optimal_moe_config",
        lambda *_args, **_kwargs: {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 16,
            "BLOCK_SIZE_K": 16,
        },
    )

    calls: list[dict[str, object]] = []

    def _fake_try_shared(**kwargs):
        calls.append(
            {
                "use_int8_w8a16": kwargs["use_int8_w8a16"],
                "use_int4_w4a16": kwargs["use_int4_w4a16"],
                "has_zp": kwargs["B_zp"] is not None,
                "block_shape": kwargs["block_shape"],
                "shape_C": tuple(kwargs["C"].shape),
            }
        )
        if len(calls) == 1:
            kwargs["C"].copy_(
                torch.tensor(
                    [
                        [[-1.0, 2.0], [0.5, -0.5], [0.0, 0.0]],
                        [[3.0, -4.0], [0.0, 0.0], [0.0, 0.0]],
                    ],
                    dtype=kwargs["C"].dtype,
                    device=kwargs["C"].device,
                )
            )
        elif len(calls) == 2:
            kwargs["C"].copy_(
                torch.tensor(
                    [
                        [[3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0], [0.0, 0.0, 0.0, 0.0]],
                        [[11.0, 12.0, 13.0, 14.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                    ],
                    dtype=kwargs["C"].dtype,
                    device=kwargs["C"].device,
                )
            )
        else:
            raise AssertionError("_try_dispatch_batched_wna16_shared should run exactly twice")
        return True

    def _raise_if_triton_path_runs(*_args, **_kwargs):
        raise AssertionError("batched int4_w4a16 Triton path should not run")

    monkeypatch.setattr(
        fused_batched_moe_mod,
        "_try_dispatch_batched_wna16_shared",
        _fake_try_shared,
    )
    monkeypatch.setattr(
        fused_batched_moe_mod,
        "batched_triton_kernel",
        _raise_if_triton_path_runs,
    )

    experts = fused_batched_moe_mod.BatchedTritonExperts(
        moe_config=SimpleNamespace(
            is_act_and_mul=False,
            activation=fused_moe_activation.MoEActivation.RELU2_NO_MUL,
            moe_parallel_config=SimpleNamespace(use_fi_all2allv_kernels=False),
            routing_method=None,
            router_logits_dtype=None,
            hidden_dim=4,
        ),
        quant_config=int4_w4a16_moe_quant_config(
            w1_scale=torch.tensor(
                [
                    [[0.5, 0.25], [1.0, 0.75]],
                    [[0.25, 0.5], [0.75, 0.125]],
                ],
                dtype=torch.float32,
            ),
            w2_scale=torch.tensor(
                [
                    [[0.5], [0.25], [1.0], [0.75]],
                    [[0.125], [0.5], [0.25], [1.0]],
                ],
                dtype=torch.float32,
            ),
            w1_zp=torch.zeros((2, 1, 2), dtype=torch.int8),
            w2_zp=torch.zeros((2, 2, 1), dtype=torch.int8),
            block_shape=[0, 2],
        ),
        max_num_tokens=3,
        num_dispatchers=1,
    )

    hidden_states = torch.tensor(
        [
            [[1.0, -2.0, 0.5, 3.0], [0.5, 3.0, -1.0, 0.0], [7.0, 7.0, 7.0, 7.0]],
            [[-1.5, 0.25, 2.0, -0.5], [8.0, 8.0, 8.0, 8.0], [8.0, 8.0, 8.0, 8.0]],
        ],
        dtype=torch.float32,
    )
    w1 = torch.zeros((2, 2, 2), dtype=torch.int8)
    w2 = torch.zeros((2, 4, 1), dtype=torch.int8)
    output = torch.zeros((2, 3, 4), dtype=torch.float32)
    workspace13 = torch.empty((2, 3, 4), dtype=torch.float32)
    workspace2 = torch.empty((2, 3, 2), dtype=torch.float32)
    expert_num_tokens = torch.tensor([2, 1], dtype=torch.int32)

    experts.apply(
        output=output,
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=torch.ones((1, 1), dtype=torch.float32),
        topk_ids=torch.zeros((1, 1), dtype=torch.int32),
        activation=fused_moe_activation.MoEActivation.RELU2_NO_MUL,
        global_num_experts=2,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=workspace13,
        workspace2=workspace2,
        expert_tokens_meta=SimpleNamespace(
            expert_num_tokens=expert_num_tokens,
            expert_num_tokens_cpu=None,
        ),
        apply_router_weight_on_input=False,
    )

    assert len(calls) == 2
    assert all(call["use_int8_w8a16"] is False for call in calls)
    assert all(call["use_int4_w4a16"] is True for call in calls)
    assert all(call["has_zp"] is True for call in calls)
    assert all(call["block_shape"] == [0, 2] for call in calls)
    torch.testing.assert_close(
        output,
        torch.tensor(
            [
                [[3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0], [0.0, 0.0, 0.0, 0.0]],
                [[11.0, 12.0, 13.0, 14.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            ],
            dtype=torch.float32,
        ),
    )


def test_naive_batched_experts_apply_int4_w4a16_reference() -> None:
    logical_w1 = torch.tensor(
        [
            [[9, 7, 12, 4], [6, 8, 10, 5]],
            [[11, 9, 5, 7], [8, 10, 6, 12]],
        ],
        dtype=torch.int32,
    )
    logical_w1_zp = torch.tensor(
        [
            [[8, 9], [7, 6]],
            [[9, 8], [6, 7]],
        ],
        dtype=torch.int32,
    )
    logical_w2 = torch.tensor(
        [
            [[9, 7], [8, 6], [10, 5], [11, 9]],
            [[7, 10], [9, 8], [6, 11], [5, 12]],
        ],
        dtype=torch.int32,
    )
    logical_w2_zp = torch.tensor(
        [
            [[8], [7], [6], [9]],
            [[7], [8], [5], [6]],
        ],
        dtype=torch.int32,
    )

    experts = fused_batched_moe_mod.NaiveBatchedExperts(
        moe_config=SimpleNamespace(
            is_act_and_mul=False,
            activation=fused_moe_activation.MoEActivation.RELU2_NO_MUL,
            moe_parallel_config=SimpleNamespace(use_fi_all2allv_kernels=False),
            routing_method=None,
            router_logits_dtype=None,
            hidden_dim=4,
        ),
        quant_config=int4_w4a16_moe_quant_config(
            w1_scale=torch.tensor(
                [
                    [[0.5, 0.25], [1.0, 0.75]],
                    [[0.25, 0.5], [0.75, 0.125]],
                ],
                dtype=torch.float32,
            ),
            w2_scale=torch.tensor(
                [
                    [[0.5], [0.25], [1.0], [0.75]],
                    [[0.125], [0.5], [0.25], [1.0]],
                ],
                dtype=torch.float32,
            ),
            w1_zp=torch.stack(
                [
                    _pack_int4_pairs_on_first_dim_for_test(logical_w1_zp[idx])
                    for idx in range(2)
                ],
                dim=0,
            ),
            w2_zp=torch.stack(
                [
                    _pack_int4_pairs_on_first_dim_for_test(logical_w2_zp[idx])
                    for idx in range(2)
                ],
                dim=0,
            ),
            block_shape=[0, 2],
        ),
        max_num_tokens=3,
        num_dispatchers=1,
    )

    hidden_states = torch.tensor(
        [
            [[1.0, -2.0, 0.5, 3.0], [2.0, 1.0, -1.0, 0.0], [7.0, 7.0, 7.0, 7.0]],
            [[-1.5, 0.25, 2.0, -0.5], [8.0, 8.0, 8.0, 8.0], [8.0, 8.0, 8.0, 8.0]],
        ],
        dtype=torch.float32,
    )
    w1 = torch.stack(
        [_pack_int4_pairs_on_last_dim_for_test(logical_w1[idx]) for idx in range(2)],
        dim=0,
    )
    w2 = torch.stack(
        [_pack_int4_pairs_on_last_dim_for_test(logical_w2[idx]) for idx in range(2)],
        dim=0,
    )
    output = torch.zeros((2, 3, 4), dtype=torch.float32)
    workspace13 = torch.empty((2, 3, 4), dtype=torch.float32)
    workspace2 = torch.empty((2, 3, 2), dtype=torch.float32)
    expert_num_tokens = torch.tensor([2, 1], dtype=torch.int32)

    experts.apply(
        output=output,
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=torch.ones((1, 1), dtype=torch.float32),
        topk_ids=torch.zeros((1, 1), dtype=torch.int32),
        activation=fused_moe_activation.MoEActivation.RELU2_NO_MUL,
        global_num_experts=2,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=workspace13,
        workspace2=workspace2,
        expert_tokens_meta=SimpleNamespace(
            expert_num_tokens=expert_num_tokens,
            expert_num_tokens_cpu=None,
        ),
        apply_router_weight_on_input=False,
    )

    w1_dq = (
        logical_w1.to(torch.float32)
        - logical_w1_zp.to(torch.float32).repeat_interleave(2, dim=2)
    ) * experts.w1_scale.repeat_interleave(2, dim=2)
    w2_dq = (
        logical_w2.to(torch.float32)
        - logical_w2_zp.to(torch.float32).repeat_interleave(2, dim=2)
    ) * experts.w2_scale.repeat_interleave(2, dim=2)

    expected_intermediate1 = torch.zeros((2, 3, 2), dtype=torch.float32)
    expected_intermediate1[0, :2] = hidden_states[0, :2] @ w1_dq[0].transpose(0, 1)
    expected_intermediate1[1, :1] = hidden_states[1, :1] @ w1_dq[1].transpose(0, 1)

    expected_intermediate2 = torch.zeros_like(expected_intermediate1)
    fused_moe_activation.apply_moe_activation(
        fused_moe_activation.MoEActivation.RELU2_NO_MUL,
        expected_intermediate2.view(-1, expected_intermediate2.size(-1)),
        expected_intermediate1.view(-1, expected_intermediate1.size(-1)),
    )

    expected = torch.zeros_like(output)
    expected[0, :2] = expected_intermediate2[0, :2] @ w2_dq[0].transpose(0, 1)
    expected[1, :1] = expected_intermediate2[1, :1] @ w2_dq[1].transpose(0, 1)

    torch.testing.assert_close(output, expected)


def test_fused_batched_moe_apply_int4_w4a16_falls_back_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fused_batched_moe_mod, "HAS_TRITON", False)
    monkeypatch.setattr(
        fused_batched_moe_mod,
        "try_get_optimal_moe_config",
        lambda *_args, **_kwargs: {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 16,
            "BLOCK_SIZE_K": 16,
        },
    )
    monkeypatch.setattr(
        fused_batched_moe_mod,
        "_try_dispatch_batched_wna16_shared",
        lambda **_kwargs: False,
    )

    def _raise_if_triton_path_runs(*_args, **_kwargs):
        raise AssertionError("batched int4_w4a16 Triton path should not run")

    monkeypatch.setattr(
        fused_batched_moe_mod,
        "batched_triton_kernel",
        _raise_if_triton_path_runs,
    )

    logical_w1 = torch.tensor(
        [
            [[9, 7, 12, 4], [6, 8, 10, 5]],
            [[11, 9, 5, 7], [8, 10, 6, 12]],
        ],
        dtype=torch.int32,
    )
    logical_w1_zp = torch.tensor(
        [
            [[8, 9], [7, 6]],
            [[9, 8], [6, 7]],
        ],
        dtype=torch.int32,
    )
    logical_w2 = torch.tensor(
        [
            [[9, 7], [8, 6], [10, 5], [11, 9]],
            [[7, 10], [9, 8], [6, 11], [5, 12]],
        ],
        dtype=torch.int32,
    )
    logical_w2_zp = torch.tensor(
        [
            [[8], [7], [6], [9]],
            [[7], [8], [5], [6]],
        ],
        dtype=torch.int32,
    )

    experts = fused_batched_moe_mod.BatchedTritonExperts(
        moe_config=SimpleNamespace(
            is_act_and_mul=False,
            activation=fused_moe_activation.MoEActivation.RELU2_NO_MUL,
            moe_parallel_config=SimpleNamespace(use_fi_all2allv_kernels=False),
            routing_method=None,
            router_logits_dtype=None,
            hidden_dim=4,
        ),
        quant_config=int4_w4a16_moe_quant_config(
            w1_scale=torch.tensor(
                [
                    [[0.5, 0.25], [1.0, 0.75]],
                    [[0.25, 0.5], [0.75, 0.125]],
                ],
                dtype=torch.float32,
            ),
            w2_scale=torch.tensor(
                [
                    [[0.5], [0.25], [1.0], [0.75]],
                    [[0.125], [0.5], [0.25], [1.0]],
                ],
                dtype=torch.float32,
            ),
            w1_zp=torch.stack(
                [
                    _pack_int4_pairs_on_first_dim_for_test(logical_w1_zp[idx])
                    for idx in range(2)
                ],
                dim=0,
            ),
            w2_zp=torch.stack(
                [
                    _pack_int4_pairs_on_first_dim_for_test(logical_w2_zp[idx])
                    for idx in range(2)
                ],
                dim=0,
            ),
            block_shape=[0, 2],
        ),
        max_num_tokens=3,
        num_dispatchers=1,
    )

    hidden_states = torch.tensor(
        [
            [[1.0, -2.0, 0.5, 3.0], [2.0, 1.0, -1.0, 0.0], [7.0, 7.0, 7.0, 7.0]],
            [[-1.5, 0.25, 2.0, -0.5], [8.0, 8.0, 8.0, 8.0], [8.0, 8.0, 8.0, 8.0]],
        ],
        dtype=torch.float32,
    )
    w1 = torch.stack(
        [_pack_int4_pairs_on_last_dim_for_test(logical_w1[idx]) for idx in range(2)],
        dim=0,
    )
    w2 = torch.stack(
        [_pack_int4_pairs_on_last_dim_for_test(logical_w2[idx]) for idx in range(2)],
        dim=0,
    )
    output = torch.zeros((2, 3, 4), dtype=torch.float32)
    workspace13 = torch.empty((2, 3, 4), dtype=torch.float32)
    workspace2 = torch.empty((2, 3, 2), dtype=torch.float32)
    expert_num_tokens = torch.tensor([2, 1], dtype=torch.int32)

    experts.apply(
        output=output,
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=torch.ones((1, 1), dtype=torch.float32),
        topk_ids=torch.zeros((1, 1), dtype=torch.int32),
        activation=fused_moe_activation.MoEActivation.RELU2_NO_MUL,
        global_num_experts=2,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=workspace13,
        workspace2=workspace2,
        expert_tokens_meta=SimpleNamespace(
            expert_num_tokens=expert_num_tokens,
            expert_num_tokens_cpu=None,
        ),
        apply_router_weight_on_input=False,
    )

    w1_dq = (
        logical_w1.to(torch.float32)
        - logical_w1_zp.to(torch.float32).repeat_interleave(2, dim=2)
    ) * experts.w1_scale.repeat_interleave(2, dim=2)
    w2_dq = (
        logical_w2.to(torch.float32)
        - logical_w2_zp.to(torch.float32).repeat_interleave(2, dim=2)
    ) * experts.w2_scale.repeat_interleave(2, dim=2)

    expected_intermediate1 = torch.zeros((2, 3, 2), dtype=torch.float32)
    expected_intermediate1[0, :2] = hidden_states[0, :2] @ w1_dq[0].transpose(0, 1)
    expected_intermediate1[1, :1] = hidden_states[1, :1] @ w1_dq[1].transpose(0, 1)

    expected_intermediate2 = torch.zeros_like(expected_intermediate1)
    fused_moe_activation.apply_moe_activation(
        fused_moe_activation.MoEActivation.RELU2_NO_MUL,
        expected_intermediate2.view(-1, expected_intermediate2.size(-1)),
        expected_intermediate1.view(-1, expected_intermediate1.size(-1)),
    )

    expected = torch.zeros_like(output)
    expected[0, :2] = expected_intermediate2[0, :2] @ w2_dq[0].transpose(0, 1)
    expected[1, :1] = expected_intermediate2[1, :1] @ w2_dq[1].transpose(0, 1)

    torch.testing.assert_close(output, expected)


def test_fused_moe_wna16_dispatch_errors_without_backend(monkeypatch) -> None:
    monkeypatch.setattr(fused_moe_mod, "HAS_TRITON", False)
    monkeypatch.setattr(fused_moe_mod, "_has_moe_wna16_gemm_op", lambda: False)
    monkeypatch.setattr(
        fused_moe_mod,
        "should_moe_wna16_use_cuda",
        lambda **_kwargs: False,
    )

    def _raise_if_triton_path_runs(*_args, **_kwargs):
        raise AssertionError("WNA16 Triton path should not run")

    monkeypatch.setattr(
        fused_moe_mod,
        "invoke_fused_moe_wna16_triton_kernel",
        _raise_if_triton_path_runs,
    )

    with pytest.raises(
        RuntimeError,
        match="requires Triton or `_moe_C.moe_wna16_gemm`",
    ):
        fused_moe_mod.dispatch_fused_moe_kernel(
            A=torch.randn((1, 4), dtype=torch.float16),
            B=torch.randint(-8, 8, (1, 8, 4), dtype=torch.int8),
            C=torch.empty((1, 1, 8), dtype=torch.float16),
            A_scale=None,
            B_scale=torch.ones((1, 1, 8), dtype=torch.float16),
            B_zp=None,
            topk_weights=torch.ones((1, 1), dtype=torch.float32),
            sorted_token_ids=torch.tensor([0], dtype=torch.int32),
            expert_ids=torch.tensor([0], dtype=torch.int32),
            num_tokens_post_padded=torch.tensor([1], dtype=torch.int32),
            mul_routed_weight=True,
            top_k=1,
            config={"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 16},
            compute_type=None,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=True,
            use_int4_w4a16=False,
            per_channel_quant=False,
            block_shape=[0, 128],
            B_bias=None,
        )


def test_triton_wna16_experts_apply_uses_shared_dispatch_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fused_moe_mod, "HAS_TRITON", False)
    monkeypatch.setattr(
        fused_moe_mod,
        "try_get_optimal_moe_config",
        lambda *_args, **_kwargs: {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 16,
            "BLOCK_SIZE_K": 16,
        },
    )
    monkeypatch.setattr(
        fused_moe_mod,
        "moe_align_block_size",
        lambda *_args, **_kwargs: (
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([2], dtype=torch.int32),
        ),
    )

    calls: list[dict[str, object]] = []

    def _fake_dispatch(*, C: torch.Tensor, **kwargs) -> None:
        calls.append(kwargs)
        if len(calls) == 1:
            C.copy_(
                torch.tensor(
                    [[[-1.0, 2.0]], [[3.0, -4.0]]],
                    dtype=C.dtype,
                    device=C.device,
                )
            )
        elif len(calls) == 2:
            C.copy_(
                torch.tensor(
                    [[[3.0, 4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0, 10.0]]],
                    dtype=C.dtype,
                    device=C.device,
                )
            )
        else:
            raise AssertionError("dispatch_fused_moe_kernel should run exactly twice")

    def _raise_if_direct_triton_path_runs(*_args, **_kwargs):
        raise AssertionError("WNA16 Triton kernel should not run directly")

    monkeypatch.setattr(
        fused_moe_mod,
        "dispatch_fused_moe_kernel",
        _fake_dispatch,
    )
    monkeypatch.setattr(
        fused_moe_mod,
        "invoke_fused_moe_wna16_triton_kernel",
        _raise_if_direct_triton_path_runs,
    )

    experts = fused_moe_mod.TritonWNA16Experts(
        moe_config=SimpleNamespace(
            is_act_and_mul=False,
            activation=fused_moe_activation.MoEActivation.RELU2_NO_MUL,
            moe_parallel_config=SimpleNamespace(use_fi_all2allv_kernels=False),
            routing_method=None,
            router_logits_dtype=None,
            hidden_dim=4,
        ),
        quant_config=int8_w8a16_moe_quant_config(
            w1_scale=torch.ones((2, 1, 2), dtype=torch.float16),
            w2_scale=torch.ones((2, 1, 4), dtype=torch.float16),
            w1_zp=None,
            w2_zp=None,
            block_shape=[0, 128],
        ),
    )

    hidden_states = torch.tensor(
        [
            [1.0, -2.0, 0.5, 3.0],
            [0.25, -1.5, 1.5, 0.75],
        ],
        dtype=torch.float16,
    )
    w1 = torch.randint(-8, 8, (2, 2, 4), dtype=torch.int8)
    w2 = torch.randint(-8, 8, (2, 4, 2), dtype=torch.int8)
    topk_weights = torch.ones((2, 1), dtype=torch.float32)
    topk_ids = torch.tensor([[0], [1]], dtype=torch.int32)
    output = torch.zeros((2, 4), dtype=torch.float16)
    workspace13 = torch.empty((2, 2), dtype=torch.float16)
    workspace2 = torch.empty((2, 1, 4), dtype=torch.float16)

    experts.apply(
        output=output,
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=fused_moe_activation.MoEActivation.RELU2_NO_MUL,
        global_num_experts=2,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=workspace13,
        workspace2=workspace2,
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )

    assert len(calls) == 2
    assert all(call["compute_type"] is None for call in calls)
    assert all(call["use_int8_w8a16"] is True for call in calls)
    assert all(call["use_int4_w4a16"] is False for call in calls)
    torch.testing.assert_close(
        output,
        torch.tensor(
            [[3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0]],
            dtype=torch.float16,
        ),
    )


def test_fused_moe_count_expert_num_tokens_falls_back_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fused_moe_utils, "HAS_TRITON", False)

    def _raise_if_triton_path_runs(*_args, **_kwargs):
        raise AssertionError("count_expert_num_tokens Triton path should not run")

    monkeypatch.setattr(
        fused_moe_utils,
        "_count_expert_num_tokens",
        _raise_if_triton_path_runs,
    )

    topk_ids = torch.tensor(
        [
            [0, 2, -1],
            [1, 2, 0],
        ],
        dtype=torch.int32,
    )
    expert_map = torch.tensor([1, 0, 2], dtype=torch.int32)

    counts = fused_moe_utils.count_expert_num_tokens(
        topk_ids,
        num_local_experts=3,
        expert_map=expert_map,
    )

    expected = torch.tensor([1, 2, 2], dtype=torch.int32)
    torch.testing.assert_close(counts.cpu(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_moe_count_expert_num_tokens_prefers_precompiled_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fused_moe_utils, "HAS_TRITON", False)

    def _raise_if_reference_runs(*_args, **_kwargs):
        raise AssertionError(
            "count_expert_num_tokens torch reference path should not run"
        )

    monkeypatch.setattr(
        fused_moe_utils,
        "_count_expert_num_tokens_reference",
        _raise_if_reference_runs,
    )
    monkeypatch.setattr(
        fused_moe_utils.ops,
        "has_precompiled_count_expert_num_tokens",
        lambda: True,
    )

    calls: dict[str, object] = {}

    def _fake_precompiled(
        topk_ids: torch.Tensor,
        num_local_experts: int,
        expert_map: torch.Tensor | None,
    ) -> torch.Tensor:
        calls["topk_shape"] = tuple(topk_ids.shape)
        calls["num_local_experts"] = num_local_experts
        calls["has_expert_map"] = expert_map is not None
        return torch.tensor([3, 1, 4], dtype=torch.int32, device=topk_ids.device)

    monkeypatch.setattr(
        fused_moe_utils.ops,
        "count_expert_num_tokens_precompiled",
        _fake_precompiled,
    )

    topk_ids = torch.tensor([[0, 1], [2, -1]], dtype=torch.int32, device="cuda")
    expert_map = torch.tensor([2, 1, 0], dtype=torch.int32, device="cuda")

    counts = fused_moe_utils.count_expert_num_tokens(
        topk_ids,
        num_local_experts=3,
        expert_map=expert_map,
    )

    assert calls == {
        "topk_shape": (2, 2),
        "num_local_experts": 3,
        "has_expert_map": True,
    }
    torch.testing.assert_close(
        counts,
        torch.tensor([3, 1, 4], dtype=torch.int32, device="cuda"),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_moe_count_expert_num_tokens_precompiled_matches_reference() -> None:
    if not custom_ops.has_precompiled_count_expert_num_tokens():
        pytest.skip("count_expert_num_tokens_precompiled is unavailable")

    topk_ids = torch.tensor(
        [
            [0, 2, -1],
            [1, 2, 0],
            [2, -1, 1],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    expert_map = torch.tensor([2, 0, 1], dtype=torch.int32, device="cuda")

    actual = custom_ops.count_expert_num_tokens_precompiled(
        topk_ids,
        num_local_experts=3,
        expert_map=expert_map,
    )
    expected = fused_moe_utils._count_expert_num_tokens_reference(
        topk_ids,
        num_local_experts=3,
        expert_map=expert_map,
    )

    torch.testing.assert_close(actual, expected)


def test_zero_experts_compute_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(fused_moe_mod, "HAS_TRITON", False)

    def _raise_if_triton_path_runs(*_args, **_kwargs):
        raise AssertionError("zero_experts Triton path should not run")

    monkeypatch.setattr(
        fused_moe_mod,
        "compute_identity_kernel",
        _raise_if_triton_path_runs,
    )

    expert_indices = torch.tensor([[0, 4], [5, 2]], dtype=torch.int64)
    expert_scales = torch.tensor([[0.25, 0.75], [0.10, 0.20]], dtype=torch.float32)
    hidden_states = torch.tensor(
        [[1.0, 2.0, -1.0], [0.5, -0.5, 3.0]],
        dtype=torch.float32,
    )

    output = fused_moe_mod.zero_experts_compute_triton(
        expert_indices=expert_indices,
        expert_scales=expert_scales,
        num_experts=3,
        zero_expert_type="identity",
        hidden_states=hidden_states,
    )

    expected = hidden_states * torch.tensor([[0.75], [0.10]], dtype=torch.float32)

    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(
        expert_indices,
        torch.tensor([[0, 0], [0, 2]], dtype=torch.int64),
    )
    torch.testing.assert_close(
        expert_scales,
        torch.tensor([[0.25, 0.0], [0.0, 0.20]], dtype=torch.float32),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_zero_experts_compute_prefers_precompiled_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fused_moe_mod, "HAS_TRITON", False)

    def _raise_if_reference_runs(*_args, **_kwargs):
        raise AssertionError("zero_experts torch reference path should not run")

    monkeypatch.setattr(
        fused_moe_mod,
        "_zero_experts_compute_reference",
        _raise_if_reference_runs,
    )
    monkeypatch.setattr(
        fused_moe_mod.ops,
        "has_precompiled_zero_experts_compute_identity",
        lambda: True,
    )

    calls: dict[str, object] = {}

    def _fake_precompiled(
        expert_indices: torch.Tensor,
        expert_scales: torch.Tensor,
        num_experts: int,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        calls["num_experts"] = num_experts
        calls["device"] = hidden_states.device.type
        expert_indices.zero_()
        expert_scales.zero_()
        return torch.full_like(hidden_states, 7.0)

    monkeypatch.setattr(
        fused_moe_mod.ops,
        "zero_experts_compute_identity_precompiled",
        _fake_precompiled,
    )

    expert_indices = torch.tensor([[0, 4], [5, 2]], dtype=torch.int64, device="cuda")
    expert_scales = torch.tensor(
        [[0.25, 0.75], [0.10, 0.20]],
        dtype=torch.float32,
        device="cuda",
    )
    hidden_states = torch.tensor(
        [[1.0, 2.0, -1.0], [0.5, -0.5, 3.0]],
        dtype=torch.float32,
        device="cuda",
    )

    output = fused_moe_mod.zero_experts_compute_triton(
        expert_indices=expert_indices,
        expert_scales=expert_scales,
        num_experts=3,
        zero_expert_type="identity",
        hidden_states=hidden_states,
    )

    assert calls == {"num_experts": 3, "device": "cuda"}
    torch.testing.assert_close(output, torch.full_like(hidden_states, 7.0))
    torch.testing.assert_close(expert_indices, torch.zeros_like(expert_indices))
    torch.testing.assert_close(expert_scales, torch.zeros_like(expert_scales))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_zero_experts_compute_precompiled_matches_reference() -> None:
    if not custom_ops.has_precompiled_zero_experts_compute_identity():
        pytest.skip("zero_experts_compute_identity_precompiled is unavailable")

    expert_indices = torch.tensor([[0, 4], [5, 2]], dtype=torch.int64, device="cuda")
    expert_scales = torch.tensor(
        [[0.25, 0.75], [0.10, 0.20]],
        dtype=torch.float32,
        device="cuda",
    )
    hidden_states = torch.tensor(
        [[1.0, 2.0, -1.0], [0.5, -0.5, 3.0]],
        dtype=torch.float32,
        device="cuda",
    )

    ref_indices = expert_indices.clone()
    ref_scales = expert_scales.clone()
    expected = fused_moe_mod._zero_experts_compute_reference(
        ref_indices,
        ref_scales,
        num_experts=3,
        zero_expert_type="identity",
        hidden_states=hidden_states,
    )

    actual_indices = expert_indices.clone()
    actual_scales = expert_scales.clone()
    actual = custom_ops.zero_experts_compute_identity_precompiled(
        actual_indices,
        actual_scales,
        num_experts=3,
        hidden_states=hidden_states,
    )

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_indices, ref_indices)
    torch.testing.assert_close(actual_scales, ref_scales)


def test_batched_deep_gemm_persistent_quant_falls_back_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(batched_deep_gemm_moe_mod, "HAS_TRITON", False)
    monkeypatch.setattr(
        batched_deep_gemm_moe_mod,
        "_has_persistent_masked_m_silu_mul_quant_op",
        lambda: False,
    )
    monkeypatch.setattr(
        batched_deep_gemm_moe_mod.current_platform,
        "get_device_capability",
        lambda **_kwargs: SimpleNamespace(to_int=lambda: 70),
    )

    def _raise_if_triton_path_runs(*_args, **_kwargs):
        raise AssertionError(
            "persistent_masked_m_silu_mul_quant Triton path should not run"
        )

    monkeypatch.setattr(
        batched_deep_gemm_moe_mod,
        "_silu_mul_fp8_quant_deep_gemm",
        _raise_if_triton_path_runs,
    )

    gate0 = torch.linspace(-2.0, 2.0, 128, dtype=torch.float32)
    up0 = torch.linspace(0.5, -1.5, 128, dtype=torch.float32)
    gate1 = torch.linspace(1.5, -1.0, 128, dtype=torch.float32)
    up1 = torch.linspace(-0.25, 0.75, 128, dtype=torch.float32)
    invalid = torch.full((256,), 9.0, dtype=torch.float32)

    y = torch.stack(
        [
            torch.cat([gate0, up0]),
            torch.cat([gate1, up1]),
            invalid,
        ],
        dim=0,
    ).unsqueeze(0)
    tokens_per_expert = torch.tensor([2], dtype=torch.int32)

    y_q, y_s = batched_deep_gemm_moe_mod.persistent_masked_m_silu_mul_quant(
        y,
        tokens_per_expert,
        quant_scale_fmt=batched_deep_gemm_moe_mod.DeepGemmQuantScaleFMT.FLOAT32,
    )

    gate = y[0, :2, :128]
    up = y[0, :2, 128:]
    activated = torch.nn.functional.silu(gate) * up
    activated_groups = activated.view(2, 1, 128)
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    expected_scales = (
        activated_groups.abs().amax(dim=-1).clamp_min(1e-10) / fp8_info.max
    )
    expected_q = torch.clamp(
        activated_groups / expected_scales.unsqueeze(-1),
        fp8_info.min,
        fp8_info.max,
    ).to(torch.float8_e4m3fn)

    torch.testing.assert_close(
        y_q[0, :2].to(torch.float32),
        expected_q.view(2, 128).to(torch.float32),
    )
    torch.testing.assert_close(y_s[0, :2].to(torch.float32), expected_scales)
    torch.testing.assert_close(y_q[0, 2].to(torch.float32), torch.zeros(128))
    torch.testing.assert_close(y_s[0, 2].to(torch.float32), torch.zeros(1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_batched_deep_gemm_persistent_quant_prefers_precompiled_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(batched_deep_gemm_moe_mod, "HAS_TRITON", False)
    monkeypatch.setattr(
        batched_deep_gemm_moe_mod,
        "_has_persistent_masked_m_silu_mul_quant_op",
        lambda: True,
    )
    monkeypatch.setattr(
        batched_deep_gemm_moe_mod.current_platform,
        "get_device_capability",
        lambda **_kwargs: SimpleNamespace(to_int=lambda: 80),
    )

    def _raise_if_reference_runs(*_args, **_kwargs):
        raise AssertionError(
            "persistent_masked_m_silu_mul_quant reference path should not run"
        )

    monkeypatch.setattr(
        batched_deep_gemm_moe_mod,
        "_persistent_masked_m_silu_mul_quant_reference",
        _raise_if_reference_runs,
    )

    calls: dict[str, object] = {}

    def _fake_precompiled(
        y: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        y_q: torch.Tensor,
        y_s: torch.Tensor,
        ceil_ue8m0: bool,
    ) -> None:
        calls["y_shape"] = tuple(y.shape)
        calls["tokens"] = tuple(tokens_per_expert.cpu().tolist())
        calls["ceil_ue8m0"] = ceil_ue8m0
        y_q.zero_()
        y_s.fill_(1.25)

    monkeypatch.setattr(
        torch.ops._C,
        "persistent_masked_m_silu_mul_quant",
        _fake_precompiled,
        raising=False,
    )

    y = torch.linspace(
        -2.0,
        2.0,
        2 * 3 * 256,
        dtype=torch.bfloat16,
        device="cuda",
    ).view(2, 3, 256)
    tokens_per_expert = torch.tensor([2, 1], dtype=torch.int32, device="cuda")

    actual_q, actual_s = batched_deep_gemm_moe_mod.persistent_masked_m_silu_mul_quant(
        y,
        tokens_per_expert,
        group_size=128,
        quant_scale_fmt=batched_deep_gemm_moe_mod.DeepGemmQuantScaleFMT.FLOAT32,
    )

    assert calls == {
        "y_shape": (2, 3, 256),
        "tokens": (2, 1),
        "ceil_ue8m0": False,
    }
    torch.testing.assert_close(actual_q.float(), torch.zeros_like(actual_q.float()))
    torch.testing.assert_close(actual_s, torch.full_like(actual_s, 1.25))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_batched_deep_gemm_persistent_quant_precompiled_matches_reference() -> None:
    if not batched_deep_gemm_moe_mod._has_persistent_masked_m_silu_mul_quant_op():
        pytest.skip("persistent_masked_m_silu_mul_quant precompiled op is unavailable")

    capability = batched_deep_gemm_moe_mod.current_platform.get_device_capability(
        device_id=torch.cuda.current_device()
    ).to_int()
    if capability < 80:
        pytest.skip("persistent_masked_m_silu_mul_quant requires SM80+")

    y = torch.linspace(
        -3.0,
        3.0,
        2 * 3 * 256,
        dtype=torch.bfloat16,
        device="cuda",
    ).view(2, 3, 256)
    tokens_per_expert = torch.tensor([2, 1], dtype=torch.int32, device="cuda")
    quant_scale_fmt = batched_deep_gemm_moe_mod.DeepGemmQuantScaleFMT.FLOAT32

    actual_q, actual_s = batched_deep_gemm_moe_mod.persistent_masked_m_silu_mul_quant(
        y,
        tokens_per_expert,
        group_size=128,
        quant_scale_fmt=quant_scale_fmt,
    )

    E, T, H2 = y.shape
    H = H2 // 2
    G = H // 128
    ys_shape, ys_strides, ys_dtype = (
        batched_deep_gemm_moe_mod.scales_shape_stride_dtype(
            E,
            T,
            G,
            quant_scale_fmt,
        )
    )
    expected_q, expected_s = (
        batched_deep_gemm_moe_mod._persistent_masked_m_silu_mul_quant_reference(
            y,
            tokens_per_expert,
            torch.empty((E, T, H), dtype=torch.float8_e4m3fn, device="cuda"),
            torch.empty_strided(
                ys_shape,
                ys_strides,
                dtype=ys_dtype,
                device="cuda",
            ),
            128,
            quant_scale_fmt,
        )
    )

    actual_dq = actual_q.float().view(E, T, G, 128) * actual_s.unsqueeze(-1)
    expected_dq = expected_q.float().view(E, T, G, 128) * expected_s.unsqueeze(-1)

    for expert_idx, num_tokens in enumerate(tokens_per_expert.cpu().tolist()):
        if num_tokens <= 0:
            continue
        torch.testing.assert_close(
            actual_s[expert_idx, :num_tokens],
            expected_s[expert_idx, :num_tokens],
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            actual_dq[expert_idx, :num_tokens],
            expected_dq[expert_idx, :num_tokens],
            atol=3e-2,
            rtol=0.15,
        )


def test_lightning_attention_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(lightning_attn_mod, "HAS_TRITON", False)

    q = torch.tensor([[[[1.0, 0.0], [0.5, 1.0], [1.5, -0.5]]]], dtype=torch.float32)
    k = torch.tensor([[[[0.25, 1.0], [1.0, 0.5], [0.5, -1.0]]]], dtype=torch.float32)
    v = torch.tensor([[[[1.0, 2.0], [0.5, -1.0], [3.0, 1.0]]]], dtype=torch.float32)
    slope = torch.tensor([0.2], dtype=torch.float32)
    initial_state = torch.tensor(
        [[[[0.5, -0.25], [1.0, 0.75]]]],
        dtype=torch.float32,
    )

    output, block_history = lightning_attn_mod.lightning_attention(
        q,
        k,
        v,
        slope,
        block_size=2,
        kv_history=initial_state,
    )

    decay = torch.exp(-slope[0]).item()
    state = initial_state[0, 0].clone()
    expected_output = torch.empty((3, 2), dtype=torch.float32)
    expected_blocks = [initial_state.clone()]
    for token_idx in range(3):
        if token_idx == 2:
            expected_blocks.append(state.view(1, 1, 2, 2).clone())
        state = torch.outer(k[0, 0, token_idx], v[0, 0, token_idx]) + decay * state
        expected_output[token_idx] = torch.matmul(q[0, 0, token_idx], state)
    expected_blocks.append(state.view(1, 1, 2, 2).clone())
    expected_history = torch.stack(expected_blocks, dim=2)

    torch.testing.assert_close(output, expected_output.view(1, 1, 3, 2))
    torch.testing.assert_close(block_history, expected_history)


def test_lightning_attention_autograd_forward_falls_back_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(lightning_attn_mod, "HAS_TRITON", False)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("fallback path should not query CUDA capability")
        ),
    )

    q = torch.tensor([[[[1.0, 0.0], [0.5, 1.0], [1.5, -0.5]]]], dtype=torch.float32)
    k = torch.tensor([[[[0.25, 1.0], [1.0, 0.5], [0.5, -1.0]]]], dtype=torch.float32)
    v = torch.tensor([[[[1.0, 2.0], [0.5, -1.0], [3.0, 1.0]]]], dtype=torch.float32)
    slope = torch.tensor([0.2], dtype=torch.float32)
    initial_state = torch.tensor(
        [[[[0.5, -0.25], [1.0, 0.75]]]],
        dtype=torch.float32,
    )

    expected_output, expected_history = (
        lightning_attn_mod._lightning_attention_reference(
            q,
            k,
            v,
            slope,
            kv_history=initial_state,
        )
    )

    class _ForbiddenCtx:
        def save_for_backward(self, *args, **kwargs) -> None:
            raise AssertionError("fallback path should not save Triton tensors")

    output, block_history = lightning_attn_mod._attention.forward(
        _ForbiddenCtx(),
        q,
        k,
        v,
        slope,
        initial_state,
    )

    torch.testing.assert_close(output, expected_output)
    torch.testing.assert_close(block_history, expected_history)


def test_linear_decode_forward_triton_falls_back_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(lightning_attn_mod, "HAS_TRITON", False)

    q = torch.tensor(
        [
            [[[1.0, -1.0]]],
            [[[0.5, 2.0]]],
        ],
        dtype=torch.float32,
    )
    k = torch.tensor(
        [
            [[[0.25, 1.5]]],
            [[[1.0, -0.5]]],
        ],
        dtype=torch.float32,
    )
    v = torch.tensor(
        [
            [[[2.0, -1.0]]],
            [[[0.5, 3.0]]],
        ],
        dtype=torch.float32,
    )
    kv_cache = torch.tensor(
        [
            [[[0.5, 1.0], [-0.5, 0.25]]],
            [[[2.0, -2.0], [1.0, 0.0]]],
        ],
        dtype=torch.float32,
    )
    slot_idx = torch.tensor([1, -1], dtype=torch.int64)
    slope_rate = torch.tensor([[[0.3]]], dtype=torch.float32)

    output = lightning_attn_mod.linear_decode_forward_triton(
        q,
        k,
        v,
        kv_cache,
        slope_rate,
        slot_idx,
        BLOCK_SIZE=1,
    )

    expected_cache = torch.tensor(
        [
            [[[0.5, 1.0], [-0.5, 0.25]]],
            [[[2.0, -2.0], [1.0, 0.0]]],
        ],
        dtype=torch.float32,
    )
    updated_state = torch.outer(k[0, 0, 0], v[0, 0, 0]) + torch.exp(
        -slope_rate.reshape(-1)[0]
    ) * expected_cache[1, 0]
    expected_cache[1, 0] = updated_state
    expected_output = torch.zeros((2, 2), dtype=torch.float32)
    expected_output[0] = torch.matmul(q[0, 0, 0], updated_state)

    torch.testing.assert_close(output, expected_output)
    torch.testing.assert_close(kv_cache, expected_cache)


def test_linear_decode_forward_triton_uses_ceil_div_grid(monkeypatch) -> None:
    monkeypatch.setattr(lightning_attn_mod, "HAS_TRITON", True)

    captured: dict[str, tuple[int, int, int]] = {}

    class _FakeKernel:
        def __getitem__(self, grid):
            captured["grid"] = grid

            def _launch(*args, **kwargs) -> None:
                output = args[6]
                output.zero_()

            return _launch

    monkeypatch.setattr(
        lightning_attn_mod,
        "_linear_attn_decode_kernel",
        _FakeKernel(),
    )

    q = torch.ones((1, 1, 1, 5), dtype=torch.float32)
    k = torch.ones((1, 1, 1, 5), dtype=torch.float32)
    v = torch.ones((1, 1, 1, 5), dtype=torch.float32)
    kv_cache = torch.zeros((1, 1, 5, 5), dtype=torch.float32)
    slope_rate = torch.tensor([[[0.3]]], dtype=torch.float32)
    slot_idx = torch.tensor([0], dtype=torch.int64)

    output = lightning_attn_mod.linear_decode_forward_triton(
        q,
        k,
        v,
        kv_cache,
        slope_rate,
        slot_idx,
        BLOCK_SIZE=4,
    )

    assert captured["grid"] == (1, 1, 2)
    torch.testing.assert_close(output, torch.zeros((1, 5), dtype=torch.float32))


def test_awq_dequantize_triton_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(awq_triton_mod, "HAS_TRITON", False)
    monkeypatch.setattr(
        awq_triton_mod,
        "_has_awq_dequantize_precompiled",
        lambda: False,
    )

    qweight = torch.tensor([[0x01234567], [0x76543210]], dtype=torch.int32)
    zeros = torch.tensor([[0x11111111]], dtype=torch.int32)
    scales = torch.tensor(
        [[1.0, 0.5, 2.0, 1.5, 1.0, 0.25, 0.75, 1.25]],
        dtype=torch.float32,
    )

    actual = awq_triton_mod.awq_dequantize_triton(qweight, scales, zeros)

    reverse_order = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], dtype=torch.int32)
    unpack = lambda x: ((x.unsqueeze(-1) >> (reverse_order * 4)) & 0xF).reshape(
        x.shape[0], -1
    )
    expected = (unpack(qweight).to(torch.float32) - unpack(zeros).to(torch.float32)) * scales

    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_awq_gemm_triton_prefers_precompiled_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(awq_triton_mod, "HAS_TRITON", False)
    monkeypatch.setattr(
        awq_triton_mod,
        "_has_awq_gemm_precompiled",
        lambda: True,
    )

    calls: dict[str, object] = {}

    def _fake_awq_gemm(
        input: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        qzeros: torch.Tensor,
        split_k_iters: int,
    ) -> torch.Tensor:
        calls["shape"] = tuple(input.shape)
        calls["split_k_iters"] = split_k_iters
        return torch.full(
            (input.shape[0], qweight.shape[1] * 8),
            7.0,
            dtype=scales.dtype,
            device=input.device,
        )

    monkeypatch.setattr(torch.ops._C, "awq_gemm", _fake_awq_gemm, raising=False)

    input = torch.randn((2, 4), dtype=torch.float16, device="cuda")
    qweight = torch.randint(0, 16, (4, 1), dtype=torch.int32, device="cuda")
    scales = torch.ones((1, 8), dtype=torch.float16, device="cuda")
    qzeros = torch.zeros((1, 1), dtype=torch.int32, device="cuda")

    output = awq_triton_mod.awq_gemm_triton(input, qweight, scales, qzeros, 4)

    assert calls == {"shape": (2, 4), "split_k_iters": 4}
    torch.testing.assert_close(output, torch.full((2, 8), 7.0, dtype=torch.float16, device="cuda"))


def test_compressed_tensors_triton_scaled_mm_falls_back_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(compressed_tensors_scaled_mm_mod, "HAS_TRITON", False)
    monkeypatch.setattr(
        compressed_tensors_scaled_mm_mod,
        "_has_cutlass_scaled_mm_op",
        lambda: False,
    )

    input = torch.tensor(
        [
            [1, -2, 3],
            [4, 0, -1],
        ],
        dtype=torch.int8,
    )
    weight = torch.tensor(
        [
            [2, 1],
            [-1, 3],
            [4, -2],
        ],
        dtype=torch.int8,
    )
    scale_a = torch.tensor([[0.5], [2.0]], dtype=torch.float32)
    scale_b = torch.tensor([[1.5], [0.25]], dtype=torch.float32)
    bias = torch.tensor([0.75, -1.25], dtype=torch.float16)

    actual = compressed_tensors_scaled_mm_mod.triton_scaled_mm(
        input,
        weight,
        scale_a,
        scale_b,
        torch.float16,
        bias=bias,
    )

    expected = (
        torch.matmul(input.to(torch.float32), weight.to(torch.float32))
        * scale_a.to(torch.float32)
        * scale_b.to(torch.float32).T
        + bias.to(torch.float32)
    ).to(torch.float16)

    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compressed_tensors_triton_scaled_mm_prefers_cutlass_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(compressed_tensors_scaled_mm_mod, "HAS_TRITON", False)

    calls: dict[str, object] = {}

    def _fake_cutlass_scaled_mm(
        out: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> None:
        calls["a_shape"] = tuple(a.shape)
        calls["b_shape"] = tuple(b.shape)
        calls["scale_a_shape"] = tuple(scale_a.shape)
        calls["scale_b_shape"] = tuple(scale_b.shape)
        calls["bias"] = bias is not None
        out.fill_(5.0)

    monkeypatch.setattr(torch.ops._C, "cutlass_scaled_mm", _fake_cutlass_scaled_mm, raising=False)

    input = torch.ones((2, 16), dtype=torch.int8, device="cuda")
    weight = torch.ones((16, 32), dtype=torch.int8, device="cuda")
    scale_a = torch.ones((2, 1), dtype=torch.float32, device="cuda")
    scale_b = torch.ones((32, 1), dtype=torch.float32, device="cuda")
    bias = torch.zeros((32,), dtype=torch.float16, device="cuda")

    actual = compressed_tensors_scaled_mm_mod.triton_scaled_mm(
        input,
        weight,
        scale_a,
        scale_b,
        torch.float16,
        bias=bias,
    )

    assert calls == {
        "a_shape": (2, 16),
        "b_shape": (16, 32),
        "scale_a_shape": (2, 1),
        "scale_b_shape": (32, 1),
        "bias": True,
    }
    torch.testing.assert_close(
        actual,
        torch.full((2, 32), 5.0, dtype=torch.float16, device="cuda"),
    )


def _reference_qutlass_to_blocked(input_matrix: torch.Tensor) -> torch.Tensor:
    rows, cols = input_matrix.shape
    n_row_blocks = (rows + 127) // 128
    n_col_blocks = (cols + 3) // 4
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4
    padded = input_matrix.new_zeros((padded_rows, padded_cols))
    padded[:rows, :cols].copy_(input_matrix)
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)


def test_qutlass_triton_mx_block_rearrange_falls_back_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(qutlass_utils_mod, "HAS_TRITON", False)

    scale_tensor = torch.arange(129 * 5, dtype=torch.uint8).reshape(129, 5).contiguous()

    actual = qutlass_utils_mod.triton_mx_block_rearrange(scale_tensor)
    expected = _reference_qutlass_to_blocked(scale_tensor).reshape(256, 8)

    assert actual.shape == (256, 8)
    torch.testing.assert_close(actual, expected)


def test_qutlass_to_blocked_triton_backend_uses_reference_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(qutlass_utils_mod, "HAS_TRITON", False)

    def _raise_if_triton_path_runs(*_args, **_kwargs):
        raise AssertionError("qutlass Triton path should not run")

    monkeypatch.setattr(
        qutlass_utils_mod,
        "triton_mx_block_rearrange",
        _raise_if_triton_path_runs,
    )

    input_matrix = torch.arange(128 * 4, dtype=torch.uint8).reshape(128, 4).contiguous()

    actual = qutlass_utils_mod.to_blocked(input_matrix, backend="triton")
    expected = _reference_qutlass_to_blocked(input_matrix).flatten()

    torch.testing.assert_close(actual, expected)


def _reference_fp8_group_quant(
    x: torch.Tensor,
    group_size: int,
    use_ue8m0: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_dtype = fp8_utils_mod.current_platform.fp8_dtype()
    fp8_min, fp8_max = fp8_utils_mod.get_fp8_min_max()
    hidden_dim = x.shape[-1]
    num_groups = hidden_dim // group_size
    grouped = x.reshape(-1, hidden_dim).to(torch.float32).reshape(-1, num_groups, group_size)
    scales = grouped.abs().amax(dim=-1).clamp_min(1e-10) / fp8_max
    if use_ue8m0:
        scales = torch.exp2(torch.ceil(torch.log2(scales)))
    quantized = torch.clamp(
        grouped / scales.unsqueeze(-1),
        fp8_min,
        fp8_max,
    ).to(fp8_dtype).reshape_as(x)
    return quantized, scales.reshape(x.shape[:-1] + (num_groups,))


def _reference_w8a8_block_scaled_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
) -> torch.Tensor:
    block_n, block_k = block_size
    M = A.numel() // A.shape[-1]
    N, K = B.shape
    A_2d = A.reshape(M, K).to(torch.float32)
    As_2d = As.reshape(M, -1).to(torch.float32)
    a_scales = As_2d.index_select(
        1,
        torch.div(torch.arange(K), block_k, rounding_mode="floor"),
    )
    row_groups = torch.div(torch.arange(N), block_n, rounding_mode="floor")
    col_groups = torch.div(torch.arange(K), block_k, rounding_mode="floor")
    b_scales = Bs.to(torch.float32).index_select(0, row_groups).index_select(1, col_groups)
    return (A_2d * a_scales) @ (B.to(torch.float32) * b_scales).T


def test_fp8_per_token_group_quant_prefers_precompiled_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fp8_utils_mod, "HAS_TRITON", False)
    monkeypatch.setattr(fp8_utils_mod, "_has_per_token_group_fp8_quant_op", lambda: True)
    monkeypatch.setattr(fp8_utils_mod.current_platform, "is_cuda", lambda: True)

    calls: dict[str, object] = {}

    def _fake_per_token_group_fp8_quant(
        x: torch.Tensor,
        x_q: torch.Tensor,
        x_s: torch.Tensor,
        group_size: int,
        eps: float,
        fp8_min: float,
        fp8_max: float,
        use_ue8m0: bool,
        column_major_scales: bool,
        tma_aligned_scales: bool,
    ) -> None:
        calls["group_size"] = group_size
        calls["column_major_scales"] = column_major_scales
        calls["tma_aligned_scales"] = tma_aligned_scales
        x_q.fill_(0)
        x_s.fill_(3.0)

    monkeypatch.setattr(
        torch.ops._C,
        "per_token_group_fp8_quant",
        _fake_per_token_group_fp8_quant,
        raising=False,
    )

    x = torch.randn((2, 8), dtype=torch.float32)
    x_q, x_s = fp8_utils_mod.per_token_group_quant_fp8(
        x,
        group_size=4,
        column_major_scales=True,
        use_ue8m0=False,
    )

    assert calls == {
        "group_size": 4,
        "column_major_scales": True,
        "tma_aligned_scales": False,
    }
    torch.testing.assert_close(x_q.to(torch.float32), torch.zeros_like(x_q, dtype=torch.float32))
    torch.testing.assert_close(x_s.to(torch.float32), torch.full((2, 2), 3.0))


def test_fp8_per_token_group_quant_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(fp8_utils_mod, "HAS_TRITON", False)
    monkeypatch.setattr(fp8_utils_mod, "_has_per_token_group_fp8_quant_op", lambda: False)

    def _raise_if_triton_path_runs(*_args, **_kwargs):
        raise AssertionError("fp8 Triton group quant path should not run")

    monkeypatch.setattr(fp8_utils_mod, "_per_token_group_quant_fp8", _raise_if_triton_path_runs)
    monkeypatch.setattr(
        fp8_utils_mod,
        "_per_token_group_quant_fp8_colmajor",
        _raise_if_triton_path_runs,
    )

    x = torch.tensor(
        [
            [1.0, -2.0, 3.0, -4.0, 0.5, 1.5, -0.5, -1.0],
            [2.5, -1.5, 0.25, 4.0, -3.0, 2.0, 1.0, -2.0],
        ],
        dtype=torch.float32,
    )

    x_q, x_s = fp8_utils_mod.per_token_group_quant_fp8(
        x,
        group_size=4,
        column_major_scales=True,
        use_ue8m0=False,
    )
    expected_q, expected_s = _reference_fp8_group_quant(x, 4, use_ue8m0=False)

    torch.testing.assert_close(x_q.to(torch.float32), expected_q.to(torch.float32))
    torch.testing.assert_close(x_s.to(torch.float32), expected_s.to(torch.float32))


def test_fp8_silu_mul_group_quant_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(fp8_utils_mod, "HAS_TRITON", False)

    def _raise_if_triton_path_runs(*_args, **_kwargs):
        raise AssertionError("fp8 Triton silu_mul group quant path should not run")

    monkeypatch.setattr(
        fp8_utils_mod,
        "_silu_mul_per_token_group_quant_fp8_colmajor",
        _raise_if_triton_path_runs,
    )

    gate = torch.linspace(-2.0, 2.0, 128, dtype=torch.float32)
    up = torch.linspace(1.5, -0.5, 128, dtype=torch.float32)
    input = torch.stack(
        [
            torch.cat([gate, up]),
            torch.cat([gate.flip(0), up.flip(0)]),
        ],
        dim=0,
    ).repeat(64, 1)

    output, output_scales = fp8_utils_mod.silu_mul_per_token_group_quant_fp8_colmajor(
        input,
        use_ue8m0=False,
    )

    activated = torch.nn.functional.silu(input[:, :128]) * input[:, 128:]
    expected_q, expected_s = _reference_fp8_group_quant(
        activated,
        group_size=128,
        use_ue8m0=False,
    )

    torch.testing.assert_close(output.to(torch.float32), expected_q.to(torch.float32))
    torch.testing.assert_close(output_scales.to(torch.float32), expected_s.to(torch.float32))


def test_fp8_w8a8_triton_block_scaled_mm_falls_back_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fp8_utils_mod, "HAS_TRITON", False)

    def _raise_if_triton_path_runs(*_args, **_kwargs):
        raise AssertionError("fp8 Triton block scaled mm path should not run")

    monkeypatch.setattr(fp8_utils_mod, "_w8a8_triton_block_scaled_mm", _raise_if_triton_path_runs)

    A = torch.tensor(
        [
            [1.0, 2.0, -1.0, 0.5],
            [0.25, -0.5, 1.5, 2.0],
        ],
        dtype=torch.float32,
    )
    B = torch.tensor(
        [
            [1.0, -1.0, 2.0, 0.5],
            [0.5, 1.5, -0.25, 2.0],
            [-2.0, 0.25, 0.75, 1.0],
        ],
        dtype=torch.float32,
    )
    As = torch.tensor([[0.5, 2.0], [1.5, 0.25]], dtype=torch.float32)
    Bs = torch.tensor([[1.0, 0.5], [0.25, 1.5]], dtype=torch.float32)

    actual = fp8_utils_mod.w8a8_triton_block_scaled_mm(
        A,
        B,
        As,
        Bs,
        [2, 2],
        output_dtype=torch.float32,
    )
    expected = _reference_w8a8_block_scaled_mm(A, B, As, Bs, [2, 2])

    torch.testing.assert_close(actual, expected)


def _reference_per_token_quant_int8(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_2d = x.reshape(-1, x.shape[-1]).to(torch.float32)
    scales = x_2d.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10) / 127.0
    x_q = torch.clamp(x_2d / scales, -128.0, 127.0).to(torch.int8)
    return x_q.reshape_as(x), scales.reshape(*x.shape[:-1], 1)


def _reference_group_quant_int8(
    x: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_dim = x.shape[-1]
    num_groups = hidden_dim // group_size
    grouped = x.reshape(-1, hidden_dim).to(torch.float32).reshape(-1, num_groups, group_size)
    scales = grouped.abs().amax(dim=-1).clamp_min(1e-10) / 127.0
    x_q = torch.clamp(grouped / scales.unsqueeze(-1), -128.0, 127.0).to(torch.int8)
    return x_q.reshape_as(x), scales.reshape(*x.shape[:-1], num_groups)


def _reference_w8a8_block_int8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
) -> torch.Tensor:
    block_n, block_k = block_size
    M = A.numel() // A.shape[-1]
    N, K = B.shape
    A_2d = A.reshape(M, K).to(torch.float32)
    As_2d = As.reshape(M, -1).to(torch.float32)
    a_scales = As_2d.index_select(
        1,
        torch.div(torch.arange(K), block_k, rounding_mode="floor"),
    )
    row_groups = torch.div(torch.arange(N), block_n, rounding_mode="floor")
    col_groups = torch.div(torch.arange(K), block_k, rounding_mode="floor")
    b_scales = Bs.to(torch.float32).index_select(0, row_groups).index_select(1, col_groups)
    return (A_2d * a_scales) @ (B.to(torch.float32) * b_scales).T


def test_int8_per_token_quant_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(int8_utils_mod, "HAS_TRITON", False)

    def _raise_if_triton_path_runs(*_args, **_kwargs):
        raise AssertionError("int8 Triton per-token quant path should not run")

    monkeypatch.setattr(int8_utils_mod, "_per_token_quant_int8", _raise_if_triton_path_runs)

    x = torch.tensor(
        [
            [1.0, -2.0, 3.0, -4.0],
            [0.5, 2.5, -1.5, 0.25],
        ],
        dtype=torch.float32,
    )

    actual_q, actual_s = int8_utils_mod.per_token_quant_int8(x)
    expected_q, expected_s = _reference_per_token_quant_int8(x)

    torch.testing.assert_close(actual_q.to(torch.float32), expected_q.to(torch.float32))
    torch.testing.assert_close(actual_s.to(torch.float32), expected_s.to(torch.float32))


def test_int8_per_token_group_quant_prefers_precompiled_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(int8_utils_mod, "HAS_TRITON", False)
    monkeypatch.setattr(int8_utils_mod, "_has_per_token_group_quant_int8_op", lambda: True)
    monkeypatch.setattr(int8_utils_mod.current_platform, "is_cuda", lambda: True)

    calls: dict[str, object] = {}

    def _fake_per_token_group_quant_int8(
        x: torch.Tensor,
        x_q: torch.Tensor,
        x_s: torch.Tensor,
        group_size: int,
        eps: float,
        int8_min: float,
        int8_max: float,
    ) -> None:
        calls["group_size"] = group_size
        x_q.fill_(1)
        x_s.fill_(2.0)

    monkeypatch.setattr(
        torch.ops._C,
        "per_token_group_quant_int8",
        _fake_per_token_group_quant_int8,
        raising=False,
    )

    x = torch.randn((2, 8), dtype=torch.float32)
    x_q, x_s = int8_utils_mod.per_token_group_quant_int8(x, group_size=4)

    assert calls == {"group_size": 4}
    torch.testing.assert_close(x_q.to(torch.float32), torch.ones_like(x_q, dtype=torch.float32))
    torch.testing.assert_close(x_s.to(torch.float32), torch.full((2, 2), 2.0))


def test_int8_per_token_group_quant_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(int8_utils_mod, "HAS_TRITON", False)
    monkeypatch.setattr(int8_utils_mod, "_has_per_token_group_quant_int8_op", lambda: False)

    def _raise_if_triton_path_runs(*_args, **_kwargs):
        raise AssertionError("int8 Triton group quant path should not run")

    monkeypatch.setattr(int8_utils_mod, "_per_token_group_quant_int8", _raise_if_triton_path_runs)

    x = torch.tensor(
        [
            [1.0, -2.0, 3.0, -4.0, 0.5, 1.5, -0.5, -1.0],
            [2.5, -1.5, 0.25, 4.0, -3.0, 2.0, 1.0, -2.0],
        ],
        dtype=torch.float32,
    )

    actual_q, actual_s = int8_utils_mod.per_token_group_quant_int8(x, group_size=4)
    expected_q, expected_s = _reference_group_quant_int8(x, 4)

    torch.testing.assert_close(actual_q.to(torch.float32), expected_q.to(torch.float32))
    torch.testing.assert_close(actual_s.to(torch.float32), expected_s.to(torch.float32))


def test_int8_w8a8_block_int8_matmul_falls_back_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(int8_utils_mod, "HAS_TRITON", False)

    def _raise_if_triton_path_runs(*_args, **_kwargs):
        raise AssertionError("int8 Triton block matmul path should not run")

    monkeypatch.setattr(int8_utils_mod, "_w8a8_block_int8_matmul", _raise_if_triton_path_runs)

    A = torch.tensor(
        [
            [1, 2, -1, 1],
            [0, -1, 3, 2],
        ],
        dtype=torch.int8,
    )
    B = torch.tensor(
        [
            [1, -1, 2, 0],
            [0, 2, -1, 1],
            [-2, 0, 1, 1],
        ],
        dtype=torch.int8,
    )
    As = torch.tensor([[0.5, 2.0], [1.5, 0.25]], dtype=torch.float32)
    Bs = torch.tensor([[1.0, 0.5], [0.25, 1.5]], dtype=torch.float32)

    actual = int8_utils_mod.w8a8_block_int8_matmul(
        A,
        B,
        As,
        Bs,
        [2, 2],
        output_dtype=torch.float32,
    )
    expected = _reference_w8a8_block_int8_matmul(A, B, As, Bs, [2, 2])

    torch.testing.assert_close(actual, expected)


def test_activation_swiglustep_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(activation_mod, "HAS_TRITON", False)

    def _raise_if_triton_path_runs(*_args, **_kwargs):
        raise AssertionError("SwigluStepAndMul Triton path should not run")

    monkeypatch.setattr(
        activation_mod,
        "swiglustep_and_mul_triton",
        _raise_if_triton_path_runs,
    )

    module = object.__new__(activation_mod.SwigluStepAndMul)
    module.limit = 1.5
    x = torch.tensor(
        [
            [1.0, -2.0, 2.0, -3.0],
            [0.5, 3.0, -4.0, 0.25],
        ],
        dtype=torch.float32,
    )

    actual = activation_mod.SwigluStepAndMul.forward_cuda(module, x)

    gate, up = x.chunk(2, dim=-1)
    expected = torch.nn.functional.silu(gate).clamp(max=1.5) * up.clamp(
        min=-1.5,
        max=1.5,
    )

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_activation_swiglustep_prefers_precompiled_without_triton(
    monkeypatch,
) -> None:
    monkeypatch.setattr(activation_mod, "HAS_TRITON", False)

    def _raise_if_triton_path_runs(*_args, **_kwargs):
        raise AssertionError("SwigluStepAndMul Triton path should not run")

    monkeypatch.setattr(
        activation_mod,
        "swiglustep_and_mul_triton",
        _raise_if_triton_path_runs,
    )

    calls: dict[str, object] = {}

    def _fake_precompiled(out: torch.Tensor, x: torch.Tensor, limit: float) -> None:
        calls["shape"] = tuple(x.shape)
        calls["limit"] = limit
        out.fill_(2.5)

    module = object.__new__(activation_mod.SwigluStepAndMul)
    module.limit = 1.5
    module.op = _fake_precompiled
    x = torch.tensor(
        [
            [1.0, -2.0, 2.0, -3.0],
            [0.5, 3.0, -4.0, 0.25],
        ],
        dtype=torch.float16,
        device="cuda",
    )

    actual = activation_mod.SwigluStepAndMul.forward_cuda(module, x)

    assert calls == {"shape": (2, 4), "limit": 1.5}
    torch.testing.assert_close(actual, torch.full_like(actual, 2.5))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_activation_swiglustep_precompiled_matches_native() -> None:
    op = getattr(torch.ops._C, "swiglustep_and_mul", None)
    if op is None:
        pytest.skip("swiglustep_and_mul precompiled op is unavailable")

    module = object.__new__(activation_mod.SwigluStepAndMul)
    module.limit = 1.5
    module.op = op
    x = torch.tensor(
        [
            [1.0, -2.0, 2.0, -3.0],
            [0.5, 3.0, -4.0, 0.25],
        ],
        dtype=torch.float16,
        device="cuda",
    )

    actual = activation_mod.SwigluStepAndMul.forward_cuda(module, x)
    expected = activation_mod.SwigluStepAndMul.forward_native(module, x)

    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


def test_lora_shrink_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(lora_shrink_mod, "HAS_TRITON", False)

    class _KernelGuard:
        def __getitem__(self, _grid):
            def _runner(*_args, **_kwargs):
                raise AssertionError("LoRA shrink Triton kernel should not run")

            return _runner

    monkeypatch.setattr(lora_shrink_mod, "_lora_shrink_kernel", _KernelGuard())

    inputs = torch.tensor(
        [
            [1.0, 2.0, -1.0],
            [0.5, -0.5, 0.25],
            [-2.0, 1.5, 3.0],
            [1.25, 0.0, -0.75],
        ],
        dtype=torch.float16,
    )
    lora_a_weights = [
        torch.tensor(
            [
                [[1.0, 0.0, 2.0], [0.5, -1.0, 1.0]],
                [[-1.0, 1.0, 0.5], [2.0, 0.0, -0.5]],
            ],
            dtype=torch.float16,
        ),
        torch.tensor(
            [
                [[0.5, 1.0, -1.0], [1.5, 0.5, 0.0]],
                [[1.0, -0.5, 2.0], [0.0, 1.0, 1.0]],
            ],
            dtype=torch.float16,
        ),
    ]
    output_tensor = torch.full((2, 4, 2), -123.0, dtype=torch.float16)
    token_lora_mapping = torch.tensor([0, -1, 1, 0], dtype=torch.long)
    token_indices_sorted_by_lora_ids = torch.tensor([0, 3, 2, 1], dtype=torch.long)
    num_tokens_per_lora = torch.tensor([2, 1], dtype=torch.long)
    lora_token_start_loc = torch.tensor([0, 2, 3], dtype=torch.long)
    lora_ids = torch.tensor([0, 1], dtype=torch.long)
    no_lora_flag_cpu = torch.tensor([0], dtype=torch.bool)
    num_active_loras = torch.tensor([2], dtype=torch.long)
    scaling = 0.75

    lora_shrink_mod._lora_shrink(
        inputs,
        lora_a_weights,
        output_tensor,
        token_lora_mapping,
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
        no_lora_flag_cpu,
        num_active_loras,
        scaling,
    )

    expected = _reference_lora_shrink(
        inputs,
        lora_a_weights,
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
        num_active_loras,
        scaling,
    )
    torch.testing.assert_close(output_tensor, expected)


def test_lora_expand_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(lora_expand_mod, "HAS_TRITON", False)

    class _KernelGuard:
        def __getitem__(self, _grid):
            def _runner(*_args, **_kwargs):
                raise AssertionError("LoRA expand Triton kernel should not run")

            return _runner

    monkeypatch.setattr(lora_expand_mod, "_lora_expand_kernel", _KernelGuard())

    inputs = torch.tensor(
        [
            [
                [1.0, 2.0],
                [0.5, -0.5],
                [-1.0, 3.0],
                [2.0, 1.0],
            ],
            [
                [0.0, 1.0],
                [2.0, -1.0],
                [1.5, 0.5],
                [-0.5, 2.0],
            ],
        ],
        dtype=torch.float16,
    )
    lora_b_weights = [
        torch.tensor(
            [
                [[1.0, 0.0], [0.5, 1.0]],
                [[-1.0, 2.0], [1.5, -0.5]],
            ],
            dtype=torch.float16,
        ),
        torch.tensor(
            [
                [[0.5, -1.0], [1.0, 0.5], [2.0, 1.0]],
                [[1.5, 0.0], [-0.5, 2.0], [0.25, -1.5]],
            ],
            dtype=torch.float16,
        ),
    ]
    output_tensor = torch.tensor(
        [
            [10.0, 1.0, 1.5, -2.0, 0.5, 3.0],
            [11.0, -1.0, 0.0, 1.0, -0.5, 2.5],
            [12.0, 0.5, -1.5, 0.0, 2.0, -3.0],
            [13.0, 2.0, 1.0, -1.0, 1.5, 0.25],
        ],
        dtype=torch.float16,
    )
    token_lora_mapping = torch.tensor([0, -1, 1, 0], dtype=torch.long)
    token_indices_sorted_by_lora_ids = torch.tensor([0, 3, 2, 1], dtype=torch.long)
    num_tokens_per_lora = torch.tensor([2, 1], dtype=torch.long)
    lora_token_start_loc = torch.tensor([0, 2, 3], dtype=torch.long)
    lora_ids = torch.tensor([0, 1], dtype=torch.long)
    no_lora_flag_cpu = torch.tensor([0], dtype=torch.bool)
    num_active_loras = torch.tensor([2], dtype=torch.long)

    expected = _reference_lora_expand(
        inputs,
        lora_b_weights,
        output_tensor,
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
        num_active_loras,
        offset_start=1,
        add_inputs=True,
    )

    lora_expand_mod._lora_expand(
        inputs,
        lora_b_weights,
        output_tensor,
        token_lora_mapping,
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
        no_lora_flag_cpu,
        num_active_loras,
        offset_start=1,
        add_inputs=True,
    )

    torch.testing.assert_close(output_tensor, expected)


def test_lora_shrink_fp8_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(lora_shrink_fp8_mod, "HAS_TRITON", False)

    class _KernelGuard:
        def __getitem__(self, _grid):
            def _runner(*_args, **_kwargs):
                raise AssertionError("LoRA FP8 shrink Triton kernel should not run")

            return _runner

    monkeypatch.setattr(lora_shrink_fp8_mod, "_lora_shrink_kernel_fp8", _KernelGuard())

    inputs = torch.tensor(
        [
            [1.0, 2.0, -1.0],
            [0.5, -0.5, 0.25],
            [-2.0, 1.5, 3.0],
            [1.25, 0.0, -0.75],
        ],
        dtype=torch.float32,
    ).to(torch.float8_e4m3fn)
    lora_a_weights = [
        torch.tensor(
            [
                [[1.0, 0.0, 2.0], [0.5, -1.0, 1.0]],
                [[-1.0, 1.0, 0.5], [2.0, 0.0, -0.5]],
            ],
            dtype=torch.float32,
        ).to(torch.float8_e4m3fn),
        torch.tensor(
            [
                [[0.5, 1.0, -1.0], [1.5, 0.5, 0.0]],
                [[1.0, -0.5, 2.0], [0.0, 1.0, 1.0]],
            ],
            dtype=torch.float32,
        ).to(torch.float8_e4m3fn),
    ]
    output_tensor = torch.full((2, 4, 2), -123.0, dtype=torch.float32)
    token_lora_mapping = torch.tensor([0, -1, 1, 0], dtype=torch.long)
    token_indices_sorted_by_lora_ids = torch.tensor([0, 3, 2, 1], dtype=torch.long)
    num_tokens_per_lora = torch.tensor([2, 1], dtype=torch.long)
    lora_token_start_loc = torch.tensor([0, 2, 3], dtype=torch.long)
    lora_ids = torch.tensor([0, 1], dtype=torch.long)
    no_lora_flag_cpu = torch.tensor([0], dtype=torch.bool)
    b_scale = [
        torch.tensor([0.5, 1.5], dtype=torch.float32),
        torch.tensor([1.25, 0.75], dtype=torch.float32),
    ]
    a_scale = torch.tensor(0.25, dtype=torch.float32)
    scaling = 0.75

    expected = torch.zeros_like(output_tensor)
    for slice_idx, lora_a_weight in enumerate(lora_a_weights):
        for lora_id, token_indices in _iter_lora_segments_for_test(
            token_indices_sorted_by_lora_ids,
            num_tokens_per_lora,
            lora_token_start_loc,
            lora_ids,
            torch.tensor([2], dtype=torch.long),
        ):
            expected[slice_idx, token_indices] = (
                inputs[token_indices].to(torch.float32)
                * a_scale
            ) @ (
                lora_a_weight[lora_id].to(torch.float32) * b_scale[slice_idx][lora_id]
            ).transpose(0, 1) * scaling

    lora_shrink_fp8_mod._lora_shrink_fp8(
        inputs,
        lora_a_weights,
        output_tensor,
        token_lora_mapping,
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
        no_lora_flag_cpu,
        2,
        scaling,
        b_scale,
        a_scale,
        use_fp8_w8a8=True,
    )

    torch.testing.assert_close(output_tensor, expected, atol=1e-5, rtol=1e-5)


def test_lora_expand_fp8_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(lora_expand_fp8_mod, "HAS_TRITON", False)

    class _KernelGuard:
        def __getitem__(self, _grid):
            def _runner(*_args, **_kwargs):
                raise AssertionError("LoRA FP8 expand Triton kernel should not run")

            return _runner

    monkeypatch.setattr(lora_expand_fp8_mod, "_lora_expand_kernel_fp8", _KernelGuard())

    inputs = torch.tensor(
        [
            [
                [1.0, 2.0],
                [0.5, -0.5],
                [-1.0, 3.0],
                [2.0, 1.0],
            ],
            [
                [0.0, 1.0],
                [2.0, -1.0],
                [1.5, 0.5],
                [-0.5, 2.0],
            ],
        ],
        dtype=torch.float32,
    ).to(torch.float8_e4m3fn)
    lora_b_weights = [
        torch.tensor(
            [
                [[1.0, 0.0], [0.5, 1.0]],
                [[-1.0, 2.0], [1.5, -0.5]],
            ],
            dtype=torch.float32,
        ).to(torch.float8_e4m3fn),
        torch.tensor(
            [
                [[0.5, -1.0], [1.0, 0.5], [2.0, 1.0]],
                [[1.5, 0.0], [-0.5, 2.0], [0.25, -1.5]],
            ],
            dtype=torch.float32,
        ).to(torch.float8_e4m3fn),
    ]
    output_tensor = torch.tensor(
        [
            [10.0, 1.0, 1.5, -2.0, 0.5, 3.0],
            [11.0, -1.0, 0.0, 1.0, -0.5, 2.5],
            [12.0, 0.5, -1.5, 0.0, 2.0, -3.0],
            [13.0, 2.0, 1.0, -1.0, 1.5, 0.25],
        ],
        dtype=torch.float32,
    )
    token_lora_mapping = torch.tensor([0, -1, 1, 0], dtype=torch.long)
    token_indices_sorted_by_lora_ids = torch.tensor([0, 3, 2, 1], dtype=torch.long)
    num_tokens_per_lora = torch.tensor([2, 1], dtype=torch.long)
    lora_token_start_loc = torch.tensor([0, 2, 3], dtype=torch.long)
    lora_ids = torch.tensor([0, 1], dtype=torch.long)
    no_lora_flag_cpu = torch.tensor([0], dtype=torch.bool)
    b_scale = [
        torch.tensor([[0.5, 1.0], [1.5, 0.25]], dtype=torch.float32),
        torch.tensor([[0.75, 1.25, 0.5], [1.0, 0.5, 1.5]], dtype=torch.float32),
    ]
    a_scale = torch.tensor([0.5, 1.5, 2.0, 0.75], dtype=torch.float32)

    expected = output_tensor.clone()
    current_offset = 1
    for slice_idx, lora_b_weight in enumerate(lora_b_weights):
        hidden_size = lora_b_weight.size(1)
        slice_output = expected[:, current_offset : current_offset + hidden_size]
        for lora_id, token_indices in _iter_lora_segments_for_test(
            token_indices_sorted_by_lora_ids,
            num_tokens_per_lora,
            lora_token_start_loc,
            lora_ids,
            torch.tensor([2], dtype=torch.long),
        ):
            expanded = (
                inputs[slice_idx, token_indices].to(torch.float32)
                * a_scale[token_indices].view(-1, 1)
            ) @ (
                lora_b_weight[lora_id].to(torch.float32)
                * b_scale[slice_idx][lora_id].view(-1, 1)
            ).transpose(0, 1)
            slice_output[token_indices] = expanded + slice_output[token_indices]
        current_offset += hidden_size

    lora_expand_fp8_mod._lora_expand_fp8(
        inputs,
        lora_b_weights,
        output_tensor,
        token_lora_mapping,
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
        no_lora_flag_cpu,
        2,
        b_scale,
        a_scale,
        offset_start=1,
        add_inputs=True,
        use_fp8_w8a8=True,
        per_channel_quant=True,
    )

    torch.testing.assert_close(output_tensor, expected, atol=1e-5, rtol=1e-5)


def test_fused_moe_lora_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(fused_moe_lora_op_mod, "HAS_TRITON", False)

    class _KernelGuard:
        def __getitem__(self, _grid):
            def _runner(*_args, **_kwargs):
                raise AssertionError("fused_moe_lora Triton kernel should not run")

            return _runner

    monkeypatch.setattr(fused_moe_lora_op_mod, "_fused_moe_lora_kernel", _KernelGuard())

    output = torch.tensor(
        [
            [[1.0, -1.0, 0.5, 2.0], [0.0, 1.5, -0.5, 0.25]],
            [[-1.0, 0.25, 1.0, -0.75], [2.0, -0.5, 0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    qcurr_hidden_states = torch.tensor(
        [
            [1.0, 2.0, -1.0],
            [0.5, -0.5, 0.25],
        ],
        dtype=torch.float32,
    )
    lora_a_stacked = [
        torch.tensor(
            [
                [
                    [[1.0, 0.0, 2.0], [0.5, -1.0, 1.0]],
                    [[0.5, 1.0, -1.0], [1.5, 0.5, 0.0]],
                ],
                [
                    [[-1.0, 1.0, 0.5], [2.0, 0.0, -0.5]],
                    [[1.0, -0.5, 2.0], [0.0, 1.0, 1.0]],
                ],
            ],
            dtype=torch.float32,
        ),
        torch.tensor(
            [
                [
                    [[0.25, 1.0, -0.5], [1.0, 0.0, 0.5]],
                    [[1.5, -0.5, 0.0], [0.5, 1.0, -1.0]],
                ],
                [
                    [[1.0, 0.5, -1.5], [0.0, 2.0, 0.5]],
                    [[-0.5, 1.5, 1.0], [1.0, -1.0, 0.5]],
                ],
            ],
            dtype=torch.float32,
        ),
    ]
    lora_b_stacked = [
        torch.tensor(
            [
                [
                    [[1.0, 0.0], [0.5, 1.0]],
                    [[0.75, -0.25], [1.25, 0.5]],
                ],
                [
                    [[-1.0, 2.0], [1.5, -0.5]],
                    [[0.25, 1.0], [-0.75, 1.5]],
                ],
            ],
            dtype=torch.float32,
        ),
        torch.tensor(
            [
                [
                    [[0.5, -1.0], [1.0, 0.5]],
                    [[1.5, 0.25], [-0.5, 1.25]],
                ],
                [
                    [[1.5, 0.0], [-0.5, 2.0]],
                    [[0.25, -1.5], [1.0, 0.75]],
                ],
            ],
            dtype=torch.float32,
        ),
    ]
    topk_weights = torch.tensor(
        [
            [0.5, 0.2],
            [1.5, 0.75],
        ],
        dtype=torch.float32,
    )
    expert_ids = torch.tensor([0, 1, 1, 0], dtype=torch.int32)
    token_lora_mapping = torch.tensor([0, 1], dtype=torch.int32)
    lora_ids = torch.tensor([0, 1], dtype=torch.int32)
    adapter_enabled = torch.tensor([1, 1], dtype=torch.int32)
    num_active_loras = torch.tensor([2], dtype=torch.int32)

    expected = output.clone()
    top_k_num = 2
    max_lora_rank = 2
    slice_width = 2
    for token_idx in range(qcurr_hidden_states.size(0)):
        lora_id = int(token_lora_mapping[token_idx].item())
        for topk_idx in range(top_k_num):
            expert_id = int(expert_ids[token_idx * top_k_num + topk_idx].item())
            routed_weight = topk_weights[token_idx, topk_idx]
            hidden = qcurr_hidden_states[token_idx].to(torch.float32)
            for slice_idx in range(len(lora_a_stacked)):
                shrink = hidden @ lora_a_stacked[slice_idx][lora_id, expert_id].transpose(0, 1)
                expand = shrink @ lora_b_stacked[slice_idx][lora_id, expert_id].transpose(0, 1)
                expected[
                    token_idx,
                    topk_idx,
                    slice_idx * slice_width : (slice_idx + 1) * slice_width,
                ] += expand * routed_weight

    fused_moe_lora_op_mod._fused_moe_lora(
        output=output,
        qcurr_hidden_states=qcurr_hidden_states,
        lora_a_stacked=lora_a_stacked,
        lora_b_stacked=lora_b_stacked,
        topk_weights=topk_weights,
        sorted_token_ids=None,
        expert_ids=expert_ids,
        num_tokens_post_padded=None,
        token_lora_mapping=token_lora_mapping,
        max_lora_rank=max_lora_rank,
        top_k_num=top_k_num,
        lora_ids=lora_ids,
        num_active_loras=num_active_loras,
        adapter_enabled=adapter_enabled,
        shrink_block_size_m=16,
        shrink_block_size_n=16,
        shrink_block_size_k=16,
        shrink_group_size_m=8,
        shrink_num_warps=4,
        shrink_num_stages=2,
        shrink_split_k=1,
        expand_block_size_m=16,
        expand_block_size_n=16,
        expand_block_size_k=16,
        expand_group_size_m=8,
        expand_num_warps=4,
        expand_num_stages=2,
        expand_split_k=1,
        mul_routed_weight=True,
        fully_sharded=False,
        offset=0,
    )

    torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-5)


def test_fused_moe_lora_fp8_falls_back_without_triton(monkeypatch) -> None:
    monkeypatch.setattr(fused_moe_lora_fp8_op_mod, "HAS_TRITON", False)

    class _KernelGuard:
        def __getitem__(self, _grid):
            def _runner(*_args, **_kwargs):
                raise AssertionError(
                    "fused_moe_lora_fp8 Triton kernel should not run"
                )

            return _runner

    monkeypatch.setattr(
        fused_moe_lora_fp8_op_mod,
        "_fused_moe_lora_kernel_fp8",
        _KernelGuard(),
    )

    output = torch.tensor(
        [
            [[1.0, -1.0, 0.5, 2.0], [0.0, 1.5, -0.5, 0.25]],
            [[-1.0, 0.25, 1.0, -0.75], [2.0, -0.5, 0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    qcurr_hidden_states = torch.tensor(
        [
            [1.0, 2.0, -1.0],
            [0.5, -0.5, 0.25],
        ],
        dtype=torch.float32,
    )
    lora_a_stacked = [
        torch.tensor(
            [
                [
                    [[1.0, 0.0, 2.0], [0.5, -1.0, 1.0]],
                    [[0.5, 1.0, -1.0], [1.5, 0.5, 0.0]],
                ],
                [
                    [[-1.0, 1.0, 0.5], [2.0, 0.0, -0.5]],
                    [[1.0, -0.5, 2.0], [0.0, 1.0, 1.0]],
                ],
            ],
            dtype=torch.float32,
        ),
        torch.tensor(
            [
                [
                    [[0.25, 1.0, -0.5], [1.0, 0.0, 0.5]],
                    [[1.5, -0.5, 0.0], [0.5, 1.0, -1.0]],
                ],
                [
                    [[1.0, 0.5, -1.5], [0.0, 2.0, 0.5]],
                    [[-0.5, 1.5, 1.0], [1.0, -1.0, 0.5]],
                ],
            ],
            dtype=torch.float32,
        ),
    ]
    lora_b_stacked = [
        torch.tensor(
            [
                [
                    [[1.0, 0.0], [0.5, 1.0]],
                    [[0.75, -0.25], [1.25, 0.5]],
                ],
                [
                    [[-1.0, 2.0], [1.5, -0.5]],
                    [[0.25, 1.0], [-0.75, 1.5]],
                ],
            ],
            dtype=torch.float32,
        ),
        torch.tensor(
            [
                [
                    [[0.5, -1.0], [1.0, 0.5]],
                    [[1.5, 0.25], [-0.5, 1.25]],
                ],
                [
                    [[1.5, 0.0], [-0.5, 2.0]],
                    [[0.25, -1.5], [1.0, 0.75]],
                ],
            ],
            dtype=torch.float32,
        ),
    ]
    lora_a_scale_stacked = [
        torch.tensor(
            [
                [[1.0, 0.5], [0.75, 1.25]],
                [[0.6, 1.4], [1.1, 0.9]],
            ],
            dtype=torch.float32,
        ),
        torch.tensor(
            [
                [[1.2, 0.8], [0.9, 1.05]],
                [[1.3, 0.7], [0.95, 1.15]],
            ],
            dtype=torch.float32,
        ),
    ]
    lora_b_scale_stacked = [
        torch.tensor(
            [
                [[1.1, 0.6], [0.85, 1.2]],
                [[0.7, 1.3], [1.05, 0.95]],
            ],
            dtype=torch.float32,
        ),
        torch.tensor(
            [
                [[1.15, 0.75], [0.8, 1.1]],
                [[1.25, 0.65], [0.9, 1.05]],
            ],
            dtype=torch.float32,
        ),
    ]
    topk_weights = torch.tensor(
        [
            [0.5, 0.2],
            [1.5, 0.75],
        ],
        dtype=torch.float32,
    )
    expert_ids = torch.tensor([0, 1, 1, 0], dtype=torch.int32)
    token_lora_mapping = torch.tensor([0, 1], dtype=torch.int32)
    lora_ids = torch.tensor([0, 1], dtype=torch.int32)
    adapter_enabled = torch.tensor([1, 1], dtype=torch.int32)
    shrink_act_scale = torch.tensor([1.25, 0.8], dtype=torch.float32)
    expand_act_scale = torch.tensor([0.9, 1.1, 0.95, 1.05], dtype=torch.float32)

    expected = output.clone()
    top_k_num = 2
    slice_width = 2
    for token_idx in range(qcurr_hidden_states.size(0)):
        lora_id = int(token_lora_mapping[token_idx].item())
        hidden = qcurr_hidden_states[token_idx].to(torch.float32)
        hidden = hidden * shrink_act_scale[token_idx]
        for topk_idx in range(top_k_num):
            expert_id = int(expert_ids[token_idx * top_k_num + topk_idx].item())
            routed_weight = topk_weights[token_idx, topk_idx]
            routed_scale = expand_act_scale[token_idx * top_k_num + topk_idx]
            for slice_idx in range(len(lora_a_stacked)):
                lora_a = (
                    lora_a_stacked[slice_idx][lora_id, expert_id].to(torch.float32)
                    * lora_a_scale_stacked[slice_idx][lora_id, expert_id]
                    .view(-1, 1)
                    .to(torch.float32)
                )
                shrink = hidden @ lora_a.transpose(0, 1)
                intermediate = shrink * routed_scale
                lora_b = (
                    lora_b_stacked[slice_idx][lora_id, expert_id].to(torch.float32)
                    * lora_b_scale_stacked[slice_idx][lora_id, expert_id]
                    .view(-1, 1)
                    .to(torch.float32)
                )
                expand = intermediate @ lora_b.transpose(0, 1)
                expected[
                    token_idx,
                    topk_idx,
                    slice_idx * slice_width : (slice_idx + 1) * slice_width,
                ] += expand * routed_weight

    fused_moe_lora_fp8_op_mod._fused_moe_lora_fp8(
        output=output,
        qcurr_hidden_states=qcurr_hidden_states,
        lora_a_stacked=lora_a_stacked,
        lora_b_stacked=lora_b_stacked,
        topk_weights=topk_weights,
        sorted_token_ids=None,
        expert_ids=expert_ids,
        num_tokens_post_padded=None,
        token_lora_mapping=token_lora_mapping,
        max_lora_rank=2,
        top_k_num=top_k_num,
        lora_ids=lora_ids,
        num_active_loras=2,
        adapter_enabled=adapter_enabled,
        shrink_block_size_m=16,
        shrink_block_size_n=16,
        shrink_block_size_k=16,
        shrink_group_size_m=8,
        shrink_num_warps=4,
        shrink_num_stages=2,
        shrink_split_k=1,
        expand_block_size_m=16,
        expand_block_size_n=16,
        expand_block_size_k=16,
        expand_group_size_m=8,
        expand_num_warps=4,
        expand_num_stages=2,
        expand_split_k=1,
        lora_a_scale_stacked=lora_a_scale_stacked,
        lora_b_scale_stacked=lora_b_scale_stacked,
        shrink_act_scale=shrink_act_scale,
        expand_act_scale=expand_act_scale,
        mul_routed_weight=True,
        fully_sharded=False,
        offset=0,
        use_fp8_w8a8=True,
        per_channel_quant=True,
    )

    torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-5)


def test_lora_kernel_utils_helpers_are_import_safe() -> None:
    assert callable(lora_kernel_utils_mod.mm_k)
    assert callable(lora_kernel_utils_mod.do_expand_kernel)
    assert callable(lora_kernel_utils_mod.do_shrink_kernel)
    assert lora_expand_mod.do_expand_kernel is lora_kernel_utils_mod.do_expand_kernel
    assert lora_shrink_mod.do_shrink_kernel is lora_kernel_utils_mod.do_shrink_kernel


def test_lora_fp8_kernel_utils_helpers_are_import_safe() -> None:
    assert callable(lora_fp8_kernel_utils_mod._accumulate_mm)
    assert callable(lora_fp8_kernel_utils_mod.fp8_mm_k)
    assert callable(lora_fp8_kernel_utils_mod.do_expand_kernel_fp8)
    assert callable(lora_fp8_kernel_utils_mod.do_shrink_kernel_fp8)
    assert (
        lora_expand_fp8_mod.do_expand_kernel_fp8
        is lora_fp8_kernel_utils_mod.do_expand_kernel_fp8
    )
    assert (
        lora_shrink_fp8_mod.do_shrink_kernel_fp8
        is lora_fp8_kernel_utils_mod.do_shrink_kernel_fp8
    )


def test_mla_sparse_convert_req_index_falls_back_without_triton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mla_sparse_utils, "HAS_TRITON", False)

    req_id = torch.tensor([1, 0], dtype=torch.int32)
    block_table = torch.tensor(
        [[3, 4, 5], [10, 11, 12]],
        dtype=torch.int32,
    )
    token_indices = torch.tensor(
        [[0, 2, -1, 99], [1, 5, 8, 9]],
        dtype=torch.int32,
    )

    out, valid_counts = mla_sparse_utils.triton_convert_req_index_to_global_index(
        req_id,
        block_table,
        token_indices,
        BLOCK_SIZE=4,
        NUM_TOPK_TOKENS=4,
        BLOCK_N=2,
        return_valid_counts=True,
    )

    expected = torch.tensor(
        [[40, 42, -1, -1], [13, 17, 20, 21]],
        dtype=torch.int32,
    )
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(valid_counts, torch.tensor([2, 4], dtype=torch.int32))


def test_rocm_aiter_mla_copy_page_indices_reference() -> None:
    page_indices = torch.full((5,), -1, dtype=torch.int32)
    block_table = torch.tensor(
        [[10, 11, 12], [20, 21, 22]],
        dtype=torch.int32,
    )
    cu_num_blocks = torch.tensor([0, 2, 5], dtype=torch.int32)

    rocm_aiter_mla_mod._copy_page_indices_reference(
        page_indices,
        block_table,
        cu_num_blocks,
    )

    torch.testing.assert_close(
        page_indices,
        torch.tensor([10, 11, 20, 21, 22], dtype=torch.int32),
    )


def test_rocm_aiter_mla_sparse_fetch_id_to_ragged_falls_back_without_triton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rocm_aiter_mla_sparse_mod, "HAS_TRITON", False)

    in_tensor = torch.tensor(
        [[5, 6, 7, 8], [1, 2, 3, 4]],
        dtype=torch.int32,
    )
    cumsum = torch.tensor([0, 3, 5], dtype=torch.int32)
    out_tensor = torch.full((5,), -1, dtype=torch.int32)

    rocm_aiter_mla_sparse_mod.fetch_id_to_ragged_triton(
        in_tensor,
        cumsum,
        out_tensor,
        topk=4,
    )

    torch.testing.assert_close(
        out_tensor,
        torch.tensor([5, 6, 7, 1, 2], dtype=torch.int32),
    )


def test_xpu_mla_sparse_falls_back_without_triton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(xpu_mla_sparse_mod, "HAS_TRITON", False)

    q = torch.zeros((1, 2, 576), dtype=torch.float32)
    kv = torch.empty((3, 1, 576), dtype=torch.float32)
    kv[0].fill_(1.0)
    kv[1].fill_(3.0)
    kv[2].fill_(5.0)
    indices = torch.full((1, 1, 16), -1, dtype=torch.int32)
    indices[0, 0, 0] = 0
    indices[0, 0, 1] = 1

    out, max_logits, softmax_lse = (
        xpu_mla_sparse_mod.triton_bf16_mla_sparse_interface(
            q,
            kv,
            indices,
            sm_scale=1.0,
            d_v=512,
        )
    )

    torch.testing.assert_close(out, torch.full((1, 2, 512), 2.0))
    torch.testing.assert_close(max_logits, torch.zeros((1, 2)))
    torch.testing.assert_close(
        softmax_lse,
        torch.full((1, 2), torch.log(torch.tensor(2.0)).item()),
    )


def test_rocm_aiter_fa_gather_cache_reference_nhd() -> None:
    key_cache = torch.arange(2 * 4 * 2 * 3, dtype=torch.float32).reshape(2, 4, 2, 3)
    value_cache = key_cache + 100
    key = torch.zeros((3, 2, 3), dtype=torch.float32)
    value = torch.zeros_like(key)
    block_tables = torch.tensor([[1, 0], [0, 1]], dtype=torch.int32)
    cu_seqlens_kv = torch.tensor([0, 2, 3], dtype=torch.int32)
    token_to_batch = torch.tensor([0, 0, 1], dtype=torch.int32)
    seq_starts = torch.tensor([0, 1], dtype=torch.int32)

    rocm_aiter_fa_mod._cp_mha_gather_cache_reference(
        key_cache,
        value_cache,
        key,
        value,
        block_tables,
        torch.ones(1),
        torch.ones(1),
        cu_seqlens_kv,
        token_to_batch,
        seq_starts,
        dequant=False,
        kv_cache_layout="NHD",
        total_tokens=3,
    )

    expected_key = torch.stack(
        [key_cache[1, 0], key_cache[1, 1], key_cache[0, 1]]
    )
    torch.testing.assert_close(key, expected_key)
    torch.testing.assert_close(value, expected_key + 100)


def test_rocm_aiter_fa_reshape_and_cache_shuffle_reference() -> None:
    key = torch.tensor([[[1.0, 2.0, 3.0, 4.0]], [[5.0, 6.0, 7.0, 8.0]]])
    value = key + 10
    key_cache = torch.zeros((1, 4, 1, 4), dtype=torch.float32)
    value_cache = torch.zeros_like(key_cache)
    slot_mapping = torch.tensor([1, -1], dtype=torch.int32)

    rocm_aiter_fa_mod._reshape_and_cache_shuffle_reference(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype="auto",
    )

    key_cache_view = key_cache.view(1, 1, 1, 4, 4)
    value_cache_view = value_cache.view(1, 1, 1, 4, 4)
    torch.testing.assert_close(key_cache_view[0, 0, 0, 1], key[0, 0])
    torch.testing.assert_close(value_cache_view[0, 0, 0, :, 1], value[0, 0])
    torch.testing.assert_close(key_cache_view[0, 0, 0, 0], torch.zeros(4))


def test_rocm_aiter_mla_sparse_quant_cache_falls_back_without_triton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rocm_aiter_mla_sparse_ops_mod, "HAS_TRITON", False)

    fp8_dtype = current_platform.fp8_dtype()
    k = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0]],
        dtype=torch.float32,
    )
    kv_cache = torch.zeros((1, 2, 8), dtype=torch.uint8)
    slot_mapping = torch.tensor([0, 1], dtype=torch.int32)

    rocm_aiter_mla_sparse_ops_mod.indexer_k_quant_and_cache_triton(
        k,
        kv_cache,
        slot_mapping,
        quant_block_size=1,
        scale_fmt="",
    )

    flat = kv_cache.view(1, -1)
    stored_values = flat[:, :8].view(fp8_dtype)
    stored_scales = flat[:, 8:].view(torch.float32)
    expected_scale = torch.tensor([4.0 / 448.0, 4.0 / 448.0])
    torch.testing.assert_close(stored_scales[0], expected_scale)
    torch.testing.assert_close(
        stored_values[0, :4].float() * stored_scales[0, 0],
        k[0],
        atol=0.2,
        rtol=0.06,
    )

    gathered = torch.empty((2, 4), dtype=fp8_dtype)
    gathered_scales = torch.empty((2,), dtype=torch.float32)
    rocm_aiter_mla_sparse_ops_mod.cp_gather_indexer_k_quant_cache_triton(
        kv_cache,
        gathered,
        gathered_scales,
        block_table=torch.tensor([[0]], dtype=torch.int32),
        cu_seqlen=torch.tensor([0, 2], dtype=torch.int32),
        token_to_seq=torch.tensor([0, 0], dtype=torch.int32),
    )

    torch.testing.assert_close(gathered, stored_values.view(2, 4))
    torch.testing.assert_close(gathered_scales, expected_scale)


def test_attention_common_correct_attn_out_falls_back_without_triton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(attention_common_mod, "HAS_TRITON", False)

    out = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32)
    lses = torch.tensor(
        [
            [[0.0, float("nan")]],
            [[1.0, 2.0]],
        ],
        dtype=torch.float32,
    )

    expected_out, expected_lse = attention_common_mod._correct_attn_out_torch(
        out.clone(),
        lses.clone(),
        cp_rank=1,
    )
    actual_out, actual_lse = attention_common_mod.correct_attn_out(
        out.clone(),
        lses.clone(),
        cp_rank=1,
        ctx=None,
    )

    torch.testing.assert_close(actual_out, expected_out)
    torch.testing.assert_close(actual_lse, expected_lse)


def test_attention_common_correct_attn_out_uses_precompiled_without_triton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(attention_common_mod, "HAS_TRITON", False)
    monkeypatch.setattr(
        attention_common_mod.ops,
        "has_precompiled_correct_attn_out",
        lambda: True,
    )

    expected_lse = torch.tensor([[5.0, 6.0]], dtype=torch.float32)

    def _fake_precompiled(out, lses, cp_rank, is_lse_base_on_e=True):
        out.add_(10.0)
        return expected_lse

    monkeypatch.setattr(
        attention_common_mod.ops,
        "correct_attn_out_precompiled",
        _fake_precompiled,
    )

    out = _as_fake_cuda(torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32))
    lses = _as_fake_cuda(torch.tensor([[[0.0, 1.0]]], dtype=torch.float32))

    actual_out, actual_lse = attention_common_mod.correct_attn_out(
        out,
        lses,
        cp_rank=0,
        ctx=None,
    )

    torch.testing.assert_close(actual_out, torch.tensor([[[11.0, 12.0], [13.0, 14.0]]]))
    assert actual_lse is expected_lse


def test_attention_common_pack_and_unpack_seq_fall_back_without_triton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(attention_common_mod, "HAS_TRITON", False)

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [7.0, 8.0]], dtype=torch.float32)
    lengths = torch.tensor([2, 1], dtype=torch.int32)

    packed = attention_common_mod.pack_seq_triton(
        x,
        lengths,
        pad_value=-9.0,
    )
    torch.testing.assert_close(
        packed,
        torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0]], [[7.0, 8.0], [-9.0, -9.0]]],
            dtype=torch.float32,
        ),
    )

    unpacked = attention_common_mod.unpack_seq_triton(
        packed,
        lengths,
    )
    torch.testing.assert_close(unpacked, x)


def test_attention_common_pack_and_unpack_seq_use_precompiled_without_triton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(attention_common_mod, "HAS_TRITON", False)
    monkeypatch.setattr(attention_common_mod.ops, "has_precompiled_pack_seq", lambda: True)
    monkeypatch.setattr(attention_common_mod.ops, "has_precompiled_unpack_seq", lambda: True)

    packed_expected = torch.tensor([[[1.0, 2.0]]], dtype=torch.float32)
    unpacked_expected = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    monkeypatch.setattr(
        attention_common_mod.ops,
        "pack_seq_precompiled",
        lambda *args, **kwargs: packed_expected,
    )
    monkeypatch.setattr(
        attention_common_mod.ops,
        "unpack_seq_precompiled",
        lambda *args, **kwargs: unpacked_expected,
    )

    packed_actual = attention_common_mod.pack_seq_triton(
        _as_fake_cuda(torch.tensor([[1.0, 2.0]], dtype=torch.float32)),
        torch.tensor([1], dtype=torch.int32),
    )
    unpacked_actual = attention_common_mod.unpack_seq_triton(
        _as_fake_cuda(torch.tensor([[[1.0, 2.0]]], dtype=torch.float32)),
        torch.tensor([1], dtype=torch.int32),
    )

    assert packed_actual is packed_expected
    assert unpacked_actual is unpacked_expected


def test_triton_reshape_and_cache_flash_prefers_precompiled_paths() -> None:
    key = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], dtype=torch.float32)
    value = key + 10
    slot_mapping = torch.tensor([0], dtype=torch.int32)
    k_scale = torch.ones((), dtype=torch.float32)
    v_scale = torch.ones((), dtype=torch.float32)
    called = {"block": False, "head_major": False}

    def _fake_block(*args, **kwargs):
        called["block"] = True

    def _fake_head_major(*args, **kwargs):
        called["head_major"] = True

    original_block = triton_reshape_cache_flash_mod._reshape_and_cache_flash_via_precompiled
    original_head_major = (
        triton_reshape_cache_flash_mod._reshape_and_cache_flash_head_major_via_precompiled
    )

    triton_reshape_cache_flash_mod._reshape_and_cache_flash_via_precompiled = _fake_block
    triton_reshape_cache_flash_mod._reshape_and_cache_flash_head_major_via_precompiled = (
        _fake_head_major
    )
    try:
        triton_reshape_cache_flash_mod.triton_reshape_and_cache_flash(
            key,
            value,
            torch.zeros((1, 1, 1, 4), dtype=torch.float32),
            torch.zeros((1, 1, 1, 4), dtype=torch.float32),
            slot_mapping,
            "auto",
            k_scale,
            v_scale,
        )
        triton_reshape_cache_flash_mod.triton_reshape_and_cache_flash(
            key,
            value,
            torch.zeros((1, 1, 1, 1, 4), dtype=torch.float32),
            torch.zeros((1, 1, 4, 1), dtype=torch.float32),
            slot_mapping,
            "auto",
            k_scale,
            v_scale,
        )
    finally:
        triton_reshape_cache_flash_mod._reshape_and_cache_flash_via_precompiled = original_block
        triton_reshape_cache_flash_mod._reshape_and_cache_flash_head_major_via_precompiled = (
            original_head_major
        )

    assert called == {"block": True, "head_major": True}


def test_triton_reshape_and_cache_flash_raises_without_triton_for_unsupported_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(triton_reshape_cache_flash_mod, "HAS_TRITON", False)

    with pytest.raises(RuntimeError, match="Triton is required"):
        triton_reshape_cache_flash_mod.triton_reshape_and_cache_flash(
            torch.zeros((1, 1, 4), dtype=torch.float32),
            torch.zeros((1, 1, 4), dtype=torch.float32),
            torch.zeros((1, 4, 4), dtype=torch.float32),
            torch.zeros((1, 4, 4), dtype=torch.float32),
            torch.tensor([0], dtype=torch.int32),
            "auto",
            torch.ones((), dtype=torch.float32),
            torch.ones((), dtype=torch.float32),
        )


def test_runtime_fallback_trace_records_pytorch_sampling_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_TRACE_RUNTIME_FALLBACKS", "1")
    runtime_fallback_trace_mod.reset_for_test()
    monkeypatch.setattr(topk_topp_sampler_mod, "HAS_TRITON", False)
    monkeypatch.setattr(
        topk_topp_sampler_mod.ops,
        "has_precompiled_apply_top_k_top_p",
        lambda: False,
    )
    monkeypatch.setattr(sample_rejection_sampler_mod, "HAS_TRITON", False)

    topk_topp_sampler_mod.apply_top_k_top_p(
        torch.tensor([[1.0, 3.0, 2.0]], dtype=torch.float32),
        torch.tensor([1], dtype=torch.int32),
        None,
    )
    sample_rejection_sampler_mod.expand_batch_to_tokens(
        torch.tensor([1, 2], dtype=torch.int32),
        torch.tensor([1, 3], dtype=torch.int32),
        num_tokens=3,
    )

    snapshot = runtime_fallback_trace_mod.snapshot()
    assert snapshot["sample.topk_topp:pytorch"] == 1
    assert snapshot["sample.rejection.expand_batch:torch"] == 1


def test_runtime_fallback_trace_records_precompiled_and_reference_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_TRACE_RUNTIME_FALLBACKS", "1")
    runtime_fallback_trace_mod.reset_for_test()
    monkeypatch.setattr(fused_sigmoid_gating_mod, "HAS_TRITON", False)
    monkeypatch.setattr(
        fused_sigmoid_gating_mod.ops,
        "has_precompiled_fused_sigmoid_gating_delta_rule_update",
        lambda: False,
    )
    monkeypatch.setattr(attention_common_mod, "HAS_TRITON", False)
    monkeypatch.setattr(
        attention_common_mod.ops,
        "has_precompiled_correct_attn_out",
        lambda: True,
    )
    monkeypatch.setattr(
        attention_common_mod.ops,
        "correct_attn_out_precompiled",
        lambda out, lses, cp_rank, is_lse_base_on_e=True: torch.zeros_like(lses[0]),
    )

    fused_sigmoid_gating_mod.fused_sigmoid_gating_delta_rule_update(
        A_log=torch.tensor([0.1], dtype=torch.float32),
        a=torch.tensor([[[0.2], [0.4]]], dtype=torch.float32),
        b=torch.tensor([[[0.3], [0.5]]], dtype=torch.float32),
        dt_bias=torch.tensor([0.05], dtype=torch.float32),
        q=torch.tensor([[[[1.0, 0.0]], [[0.5, 0.5]]]], dtype=torch.float32),
        k=torch.tensor([[[[0.2, 0.8]], [[0.6, 0.4]]]], dtype=torch.float32),
        v=torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]], dtype=torch.float32),
        scale=2**-0.5,
        initial_state=torch.zeros((1, 1, 2, 2), dtype=torch.float32),
        inplace_final_state=False,
    )
    attention_common_mod.correct_attn_out(
        _as_fake_cuda(torch.tensor([[[1.0, 2.0]]], dtype=torch.float32)),
        _as_fake_cuda(torch.tensor([[[0.0]]], dtype=torch.float32)),
        cp_rank=0,
        ctx=None,
    )

    snapshot = runtime_fallback_trace_mod.snapshot()
    assert snapshot["fla.fused_sigmoid_gating:reference"] == 1
    assert snapshot["attention.common.correct_attn_out:precompiled"] == 1


def test_runtime_fallback_trace_records_spec_decode_utils_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_TRACE_RUNTIME_FALLBACKS", "1")
    runtime_fallback_trace_mod.reset_for_test()
    monkeypatch.setattr(spec_decode_utils, "HAS_TRITON", False)

    monkeypatch.setattr(
        spec_decode_utils.ops,
        "has_precompiled_eagle_step_update_slot_mapping_and_metadata",
        lambda: False,
    )
    spec_decode_utils.eagle_step_update_slot_mapping_and_metadata(
        positions_1d=torch.tensor([0], dtype=torch.int64),
        block_table_tensor=torch.tensor([[10, 11]], dtype=torch.int32),
        seq_lens=torch.tensor([1], dtype=torch.int32),
        block_size=4,
        max_model_len=16,
        out_clamped_positions=torch.empty((1,), dtype=torch.int64),
        out_slot_mapping=torch.empty((1,), dtype=torch.int64),
        input_batch_size=1,
    )

    monkeypatch.setattr(
        spec_decode_utils.ops,
        "has_precompiled_eagle_prepare_inputs_padded",
        lambda: True,
    )

    def _fake_prepare_inputs_padded(
        cu_num_draft_tokens: torch.Tensor,
        valid_sampled_tokens_count: torch.Tensor,
        query_start_loc_gpu: torch.Tensor,
        token_indices_to_sample: torch.Tensor,
        num_rejected_tokens_gpu: torch.Tensor,
    ) -> None:
        token_indices_to_sample.copy_(torch.tensor([2], dtype=token_indices_to_sample.dtype))
        num_rejected_tokens_gpu.copy_(
            torch.tensor([1], dtype=num_rejected_tokens_gpu.dtype)
        )

    monkeypatch.setattr(
        spec_decode_utils.ops,
        "eagle_prepare_inputs_padded_precompiled",
        _fake_prepare_inputs_padded,
    )
    spec_decode_utils.eagle_prepare_inputs_padded(
        cu_num_draft_tokens=_as_fake_cuda(torch.tensor([2], dtype=torch.int32)),
        valid_sampled_tokens_count=_as_fake_cuda(
            torch.tensor([2], dtype=torch.int32)
        ),
        query_start_loc_gpu=_as_fake_cuda(torch.tensor([0, 4], dtype=torch.int32)),
        token_indices_to_sample=_as_fake_cuda(torch.empty((1,), dtype=torch.int64)),
        num_rejected_tokens_gpu=_as_fake_cuda(torch.empty((1,), dtype=torch.int32)),
    )

    monkeypatch.setattr(
        spec_decode_utils.ops,
        "has_precompiled_eagle_prepare_next_token_padded",
        lambda: True,
    )

    def _fake_prepare_next_token_padded(
        sampled_token_ids: torch.Tensor,
        discard_request_mask: torch.Tensor,
        backup_next_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        valid_sampled_tokens_count: torch.Tensor,
        vocab_size: int,
    ) -> None:
        next_token_ids.copy_(torch.tensor([11], dtype=next_token_ids.dtype))
        valid_sampled_tokens_count.copy_(
            torch.tensor([1], dtype=valid_sampled_tokens_count.dtype)
        )

    monkeypatch.setattr(
        spec_decode_utils.ops,
        "eagle_prepare_next_token_padded_precompiled",
        _fake_prepare_next_token_padded,
    )
    spec_decode_utils.eagle_prepare_next_token_padded(
        sampled_token_ids=_as_fake_cuda(torch.tensor([[11, -1]], dtype=torch.int64)),
        discard_request_mask=_as_fake_cuda(torch.tensor([False])),
        backup_next_token_ids=_as_fake_cuda(torch.tensor([42], dtype=torch.int64)),
        next_token_ids=_as_fake_cuda(torch.empty((1,), dtype=torch.int64)),
        valid_sampled_tokens_count=_as_fake_cuda(torch.empty((1,), dtype=torch.int32)),
        vocab_size=50,
    )

    monkeypatch.setattr(
        spec_decode_utils.ops,
        "has_precompiled_copy_and_expand_eagle_inputs",
        lambda: False,
    )
    spec_decode_utils.copy_and_expand_eagle_inputs(
        target_token_ids_ptr=torch.tensor([10, 11], dtype=torch.int32),
        target_positions_ptr=torch.tensor([0, 1], dtype=torch.int64),
        next_token_ids_ptr=torch.tensor([12], dtype=torch.int32),
        out_input_ids_ptr=torch.empty((3,), dtype=torch.int32),
        out_positions_ptr=torch.empty((3,), dtype=torch.int64),
        out_is_rejected_token_mask_ptr=torch.empty((3,), dtype=torch.bool),
        out_is_masked_token_mask_ptr=torch.empty((3,), dtype=torch.bool),
        out_new_token_indices_ptr=torch.empty((1,), dtype=torch.int64),
        out_hidden_state_mapping_ptr=torch.full((2,), -1, dtype=torch.int32),
        query_start_loc_ptr=torch.tensor([0, 2], dtype=torch.int32),
        query_end_loc_ptr=torch.tensor([1], dtype=torch.int32),
        padding_token_id=-1,
        parallel_drafting_token_id=-2,
        total_input_tokens=2,
        num_padding_slots_per_request=1,
        shift_input_ids=False,
    )

    snapshot = runtime_fallback_trace_mod.snapshot()
    assert snapshot["spec_decode.eagle.slot_mapping:torch"] == 1
    assert snapshot["spec_decode.eagle.prepare_inputs_padded:precompiled"] == 1
    assert snapshot["spec_decode.eagle.prepare_next_token_padded:precompiled"] == 1
    assert snapshot["spec_decode.eagle.copy_expand_inputs:torch"] == 1


def test_runtime_fallback_trace_records_eagle_speculator_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_TRACE_RUNTIME_FALLBACKS", "1")
    runtime_fallback_trace_mod.reset_for_test()
    monkeypatch.setattr(eagle_speculator, "HAS_TRITON", False)

    real_empty = eagle_speculator.torch.empty

    def _fake_empty(*args, **kwargs):
        kwargs = dict(kwargs)
        kwargs.pop("device", None)
        return _as_fake_cuda(real_empty(*args, **kwargs))

    monkeypatch.setattr(eagle_speculator.torch, "empty", _fake_empty)
    monkeypatch.setattr(
        eagle_speculator.ops,
        "has_precompiled_prepare_eagle_inputs",
        lambda: True,
    )

    def _fake_prepare_eagle_inputs(*args) -> None:
        last_token_indices = args[0]
        last_token_indices.copy_(torch.tensor([1], dtype=last_token_indices.dtype))

    monkeypatch.setattr(
        eagle_speculator.ops,
        "prepare_eagle_inputs_precompiled",
        _fake_prepare_eagle_inputs,
    )

    input_buffers = SimpleNamespace(
        input_ids=_as_fake_cuda(torch.zeros((4,), dtype=torch.int32)),
        positions=_as_fake_cuda(torch.zeros((4,), dtype=torch.int64)),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([2, 0], dtype=torch.int32),
        device="cpu",
    )
    input_batch = SimpleNamespace(
        num_reqs=1,
        input_ids=_as_fake_cuda(torch.tensor([10, 11], dtype=torch.int32)),
        positions=_as_fake_cuda(torch.tensor([0, 1], dtype=torch.int64)),
        idx_mapping=_as_fake_cuda(torch.tensor([0], dtype=torch.int32)),
        query_start_loc=_as_fake_cuda(torch.tensor([0, 2], dtype=torch.int32)),
    )

    eagle_speculator.prepare_eagle_inputs(
        input_buffers=input_buffers,
        input_batch=input_batch,
        num_sampled=_as_fake_cuda(torch.tensor([1], dtype=torch.int32)),
        num_rejected=_as_fake_cuda(torch.tensor([0], dtype=torch.int32)),
        last_sampled=_as_fake_cuda(torch.tensor([12], dtype=torch.int32)),
        next_prefill_tokens=_as_fake_cuda(torch.tensor([13], dtype=torch.int32)),
    )

    monkeypatch.setattr(
        eagle_speculator.ops,
        "has_precompiled_prepare_eagle_decode",
        lambda: False,
    )
    eagle_speculator.prepare_eagle_decode(
        draft_tokens=torch.tensor([7], dtype=torch.int64),
        output_hidden_states=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        last_token_indices=torch.tensor([0], dtype=torch.int64),
        target_seq_lens=torch.tensor([1], dtype=torch.int32),
        num_rejected=torch.tensor([0], dtype=torch.int32),
        input_buffers=SimpleNamespace(
            input_ids=torch.zeros((2,), dtype=torch.int32),
            positions=torch.zeros((2,), dtype=torch.int64),
            query_start_loc=torch.zeros((3,), dtype=torch.int32),
            seq_lens=torch.zeros((2,), dtype=torch.int32),
            device="cpu",
        ),
        input_hidden_states=torch.zeros((2, 2), dtype=torch.float32),
        max_model_len=8,
        max_num_reqs=2,
    )

    monkeypatch.setattr(
        eagle_speculator.ops,
        "has_precompiled_update_eagle_inputs",
        lambda: True,
    )

    def _fake_update_eagle_inputs(*args) -> None:
        input_ids = args[2]
        input_ids.copy_(torch.tensor([9, 0], dtype=input_ids.dtype))

    monkeypatch.setattr(
        eagle_speculator.ops,
        "update_eagle_inputs_precompiled",
        _fake_update_eagle_inputs,
    )
    eagle_speculator.update_eagle_inputs(
        draft_tokens=_as_fake_cuda(torch.tensor([9], dtype=torch.int64)),
        output_hidden_states=_as_fake_cuda(torch.tensor([[3.0, 4.0]], dtype=torch.float32)),
        input_buffers=SimpleNamespace(
            input_ids=_as_fake_cuda(torch.zeros((2,), dtype=torch.int32)),
            positions=_as_fake_cuda(torch.zeros((2,), dtype=torch.int64)),
            seq_lens=_as_fake_cuda(torch.zeros((2,), dtype=torch.int32)),
        ),
        hidden_states=_as_fake_cuda(torch.zeros((2, 2), dtype=torch.float32)),
        max_model_len=8,
    )

    snapshot = runtime_fallback_trace_mod.snapshot()
    assert snapshot["spec_decode.eagle.prepare_inputs:precompiled"] == 1
    assert snapshot["spec_decode.eagle.prepare_decode:torch"] == 1
    assert snapshot["spec_decode.eagle.update_inputs:precompiled"] == 1
