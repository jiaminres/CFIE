# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import dataclasses
import itertools
from collections.abc import Callable
from typing import Any

import torch

from cfie import _custom_ops as ops
from cfie.logger import init_logger
from cfie.config import CacheConfig
from cfie.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    get_conv_copy_spec,
    get_temporal_copy_spec,
)
from cfie.triton_utils import HAS_TRITON, tl, triton
from cfie.utils.math_utils import cdiv
from cfie.v1.core.sched.output import SchedulerOutput
from cfie.v1.kv_cache_interface import KVCacheConfig, MambaSpec
from cfie.v1.utils import CpuGpuBuffer
from cfie.v1.worker.gpu_input_batch import CachedRequestState
from cfie.v1.worker.lora_model_runner_mixin import GPUInputBatch

logger = init_logger(__name__)
_SINGLE_BLOCK_MAPPING_CPU = torch.tensor([[0, 0]], dtype=torch.int64, device="cpu")


@triton.jit
def batch_memcpy_kernel(src_ptrs, dst_ptrs, sizes, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    src_ptr = tl.load(src_ptrs + pid)
    dst_ptr = tl.load(dst_ptrs + pid)
    size = tl.load(sizes + pid)

    offsets = tl.arange(0, BLOCK_SIZE)
    for i in range(0, size, BLOCK_SIZE):
        mask = (i + offsets) < size

        curr_src_ptr = (src_ptr + i + offsets).to(tl.pointer_type(tl.uint8))
        curr_dst_ptr = (dst_ptr + i + offsets).to(tl.pointer_type(tl.uint8))

        data = tl.load(curr_src_ptr, mask=mask)
        tl.store(curr_dst_ptr, data, mask=mask)


def batch_memcpy(src_ptrs, dst_ptrs, sizes):
    if not HAS_TRITON:
        raise RuntimeError(
            "batch_memcpy requires Triton runtime support. "
            "Use do_mamba_copy_block() so the shared PyTorch fallback can "
            "be selected when Triton is unavailable."
        )
    batch = src_ptrs.shape[0]
    assert dst_ptrs.shape[0] == batch
    assert sizes.shape[0] == batch

    grid = (batch,)
    BLOCK_SIZE = 1024
    batch_memcpy_kernel[grid](src_ptrs, dst_ptrs, sizes, BLOCK_SIZE=BLOCK_SIZE)


def get_mamba_groups(kv_cache_config: KVCacheConfig) -> tuple[list[int], MambaSpec]:
    mamba_group_ids: list[int] = []
    mamba_specs: list[MambaSpec] = []
    for i in range(len(kv_cache_config.kv_cache_groups)):
        kv_cache_spec = kv_cache_config.kv_cache_groups[i].kv_cache_spec
        if isinstance(kv_cache_spec, MambaSpec):
            mamba_group_ids.append(i)
            mamba_specs.append(kv_cache_spec)
    assert len(mamba_group_ids) > 0, "no mamba layers in the model"
    assert all(mamba_specs[0] == spec for spec in mamba_specs)
    return mamba_group_ids, mamba_specs[0]


@dataclasses.dataclass
class MambaCopyBuffers:
    src_ptrs: CpuGpuBuffer
    dst_ptrs: CpuGpuBuffer
    sizes: CpuGpuBuffer
    offset: int = 0
    python_copies: list[tuple[torch.Tensor, torch.Tensor, int]] = dataclasses.field(
        default_factory=list,
        repr=False,
    )

    @classmethod
    def create(
        cls,
        max_num_reqs: int,
        kv_cache_config: KVCacheConfig,
        copy_funcs: tuple[MambaStateCopyFunc, ...],
        make_buffer: Callable[..., CpuGpuBuffer],
    ) -> "MambaCopyBuffers":
        mamba_group_ids, _ = get_mamba_groups(kv_cache_config)
        entries_per_req = sum(
            len(kv_cache_config.kv_cache_groups[gid].layer_names)
            for gid in mamba_group_ids
        ) * len(copy_funcs)
        n = max_num_reqs * entries_per_req
        return cls(
            src_ptrs=make_buffer(n, dtype=torch.int64),
            dst_ptrs=make_buffer(n, dtype=torch.int64),
            sizes=make_buffer(n, dtype=torch.int32),
        )


def _reset_copy_buffers(copy_bufs: MambaCopyBuffers) -> None:
    copy_bufs.offset = 0
    copy_bufs.python_copies.clear()


def _resolve_mamba_copy_tensors(
    *,
    state: torch.Tensor,
    block_ids: list[int],
    src_block_idx: int,
    dest_block_idx: int,
    num_accepted_tokens: int,
    state_copy_func: MambaStateCopyFunc,
) -> tuple[torch.Tensor, torch.Tensor]:
    dest_tensor = state[block_ids[dest_block_idx]]
    if state_copy_func is get_conv_copy_spec:
        src_tensor = state[block_ids[src_block_idx], num_accepted_tokens - 1 :]
        return src_tensor, dest_tensor
    if state_copy_func is get_temporal_copy_spec:
        src_tensor = state[block_ids[src_block_idx + num_accepted_tokens - 1]]
        return src_tensor, dest_tensor
    raise NotImplementedError(
        "Unsupported Mamba state copy function for no-Triton fallback: "
        f"{state_copy_func!r}"
    )


def _copy_mamba_state_torch_fallback(
    copy_bufs: MambaCopyBuffers,
    n: int,
) -> None:
    logger.warning_once(
        "Mamba state block copy is falling back to the shared PyTorch path "
        "because Triton runtime is unavailable."
    )
    for src_tensor, dst_tensor, num_elements in copy_bufs.python_copies[:n]:
        if num_elements == 0:
            continue
        src_flat = src_tensor.reshape(-1)[:num_elements].clone()
        dst_flat = dst_tensor.view(-1)
        dst_flat[:num_elements].copy_(src_flat, non_blocking=True)


def _copy_mamba_state_swap_blocks_precompiled(
    copy_bufs: MambaCopyBuffers,
    n: int,
) -> bool:
    if not hasattr(torch.ops, "_C_cache_ops") or not hasattr(
        torch.ops._C_cache_ops, "swap_blocks"
    ):
        return False

    logger.info_once(
        "Using `_C_cache_ops.swap_blocks` for Mamba state block copy because "
        "Triton runtime is unavailable."
    )

    for src_tensor, dst_tensor, num_elements in copy_bufs.python_copies[:n]:
        if num_elements == 0:
            continue
        if not src_tensor.is_cuda or not dst_tensor.is_cuda:
            return False

        copy_num_bytes = int(num_elements * src_tensor.element_size())
        src_byte_view = src_tensor.reshape(-1).view(torch.uint8)
        dst_byte_view = dst_tensor.view(-1).view(torch.uint8)
        ops.swap_blocks(
            src_byte_view,
            dst_byte_view,
            copy_num_bytes,
            _SINGLE_BLOCK_MAPPING_CPU,
        )
    return True


def collect_mamba_copy_meta(
    copy_bufs: MambaCopyBuffers,
    kv_cache_config: KVCacheConfig,
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
    mamba_group_ids: list[int],
    src_block_idx: int,
    dest_block_idx: int,
    accept_token_bias: int,
    req_state: CachedRequestState,
    forward_context: dict[str, Any],
) -> None:
    if src_block_idx == dest_block_idx and accept_token_bias == 0:
        return

    src_ptrs_np = copy_bufs.src_ptrs.np
    dst_ptrs_np = copy_bufs.dst_ptrs.np
    sizes_np = copy_bufs.sizes.np
    offset = copy_bufs.offset

    for mamba_group_id in mamba_group_ids:
        block_ids = req_state.block_ids[mamba_group_id]
        dest_block_id = block_ids[dest_block_idx]
        layer_names = kv_cache_config.kv_cache_groups[mamba_group_id].layer_names
        for layer_name in layer_names:
            attention = forward_context[layer_name]
            kv_caches: list[torch.Tensor] = attention.kv_cache[0]
            for state, state_copy_func in zip(kv_caches, mamba_state_copy_funcs):
                num_accepted_tokens = accept_token_bias + 1
                copy_spec = state_copy_func(
                    state, block_ids, src_block_idx, num_accepted_tokens
                )

                src_ptrs_np[offset] = copy_spec.start_addr
                dst_ptrs_np[offset] = state[dest_block_id].data_ptr()
                sizes_np[offset] = copy_spec.num_elements * state.element_size()
                if not HAS_TRITON:
                    src_tensor, dst_tensor = _resolve_mamba_copy_tensors(
                        state=state,
                        block_ids=block_ids,
                        src_block_idx=src_block_idx,
                        dest_block_idx=dest_block_idx,
                        num_accepted_tokens=num_accepted_tokens,
                        state_copy_func=state_copy_func,
                    )
                    copy_bufs.python_copies.append(
                        (src_tensor, dst_tensor, copy_spec.num_elements)
                    )
                offset += 1

    copy_bufs.offset = offset


def do_mamba_copy_block(copy_bufs: MambaCopyBuffers):
    n = copy_bufs.offset
    if n == 0:
        return
    if not HAS_TRITON:
        if not _copy_mamba_state_swap_blocks_precompiled(copy_bufs, n):
            _copy_mamba_state_torch_fallback(copy_bufs, n)
        copy_bufs.python_copies.clear()
        return
    batch_memcpy(
        copy_bufs.src_ptrs.copy_to_gpu(n),
        copy_bufs.dst_ptrs.copy_to_gpu(n),
        copy_bufs.sizes.copy_to_gpu(n),
    )


def preprocess_mamba(
    scheduler_output: SchedulerOutput,
    kv_cache_config: KVCacheConfig,
    cache_config: CacheConfig,
    mamba_state_idx: dict[str, int],
    input_batch: GPUInputBatch,
    requests: dict[str, CachedRequestState],
    forward_context: dict[str, Any],
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
    copy_bufs: MambaCopyBuffers,
):
    """
    Copy the mamba state of previous step to the last
    (1 + num_speculative_blocks) block.
    """
    mamba_group_ids, mamba_spec = get_mamba_groups(kv_cache_config)
    num_speculative_blocks = mamba_spec.num_speculative_blocks
    # TODO(Chen): we need to optimize this function a lot
    assert cache_config.enable_prefix_caching
    block_size = mamba_spec.block_size
    finished_req_ids = scheduler_output.finished_req_ids
    preempted_req_ids = scheduler_output.preempted_req_ids or set()
    # We need to clear mamba_state_idx for resumed requests. When requests are
    # force-preempted (e.g., during reset_prefix_cache / KV cache flush),
    # they appear in resumed_req_ids without a corresponding entry in
    # preempted_req_ids, leaving stale mamba_state_idx entries that can
    # point to block indices beyond the new (smaller) block allocation.
    resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
    for req_id in itertools.chain(finished_req_ids, preempted_req_ids, resumed_req_ids):
        mamba_state_idx.pop(req_id, None)

    _reset_copy_buffers(copy_bufs)
    for i, req_id in enumerate(input_batch.req_ids):
        req_state = requests[req_id]
        prev_state_idx = mamba_state_idx.get(req_id)
        if prev_state_idx is None:
            # new / resumed request, no previous state
            # if num_computed_tokens is 0, prev_state_idx will be -1
            prev_state_idx = (req_state.num_computed_tokens - 1) // block_size

        num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
        num_blocks: int = (
            cdiv(req_state.num_computed_tokens + num_scheduled_tokens, block_size)
            + num_speculative_blocks
        )

        # We always save the current running state at the last
        # (1 + num_speculative_blocks) block.
        # A corner case worth mention here: assume we have block_size = 4 and
        # num_speculative_tokens = 2. The request is [A, B, C] and contains 2 draft
        # tokens [draft 1, draft 2]. Then we will have:
        # Block 0: [A, B, C, draft 1]
        # Block 1: [draft 2, TOFILL, TOFILL, TOFILL]
        # Block 2: speculative block
        # Block 3: speculative block
        # And use block 1 to save the running state.
        curr_state_idx = num_blocks - 1 - num_speculative_blocks
        mamba_state_idx[req_id] = curr_state_idx
        if prev_state_idx != -1 and prev_state_idx != curr_state_idx:
            collect_mamba_copy_meta(
                copy_bufs,
                kv_cache_config,
                mamba_state_copy_funcs,
                mamba_group_ids,
                prev_state_idx,
                curr_state_idx,
                input_batch.num_accepted_tokens_cpu[i] - 1,
                req_state,
                forward_context,
            )
            input_batch.num_accepted_tokens_cpu[i] = 1
    do_mamba_copy_block(copy_bufs)


def postprocess_mamba(
    scheduler_output: SchedulerOutput,
    kv_cache_config: KVCacheConfig,
    input_batch: GPUInputBatch,
    requests: dict[str, CachedRequestState],
    mamba_state_idx: dict[str, int],
    forward_context: dict[str, Any],
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
    copy_bufs: MambaCopyBuffers,
):
    """
    If a blocks is converted from partial block to full block in this step, copy the
    state from the block for running state to the new full block.
    """
    num_scheduled_tokens_dict = scheduler_output.num_scheduled_tokens
    scheduled_spec_decode_tokens_dict = scheduler_output.scheduled_spec_decode_tokens
    num_accepted_tokens_cpu = input_batch.num_accepted_tokens_cpu
    # NOTE: can be optimized as this function always returns the same result
    mamba_group_ids, mamba_spec = get_mamba_groups(kv_cache_config)
    _reset_copy_buffers(copy_bufs)
    for i, req_id in enumerate(input_batch.req_ids):
        req_state = requests[req_id]
        num_computed_tokens = req_state.num_computed_tokens
        num_draft_tokens = len(scheduled_spec_decode_tokens_dict.get(req_id, []))
        num_scheduled_tokens = num_scheduled_tokens_dict[req_id]
        num_accepted_tokens = num_accepted_tokens_cpu[i]
        num_tokens_running_state = (
            num_computed_tokens + num_scheduled_tokens - num_draft_tokens
        )
        new_num_computed_tokens = num_tokens_running_state + num_accepted_tokens - 1
        aligned_new_computed_tokens = (
            new_num_computed_tokens // mamba_spec.block_size * mamba_spec.block_size
        )
        # TODO: how to ensure all blocks that cache_blocks called are cached here?
        if aligned_new_computed_tokens >= num_tokens_running_state:
            accept_token_bias = aligned_new_computed_tokens - num_tokens_running_state
            src_block_idx = mamba_state_idx[req_id]
            dest_block_idx = aligned_new_computed_tokens // mamba_spec.block_size - 1
            collect_mamba_copy_meta(
                copy_bufs,
                kv_cache_config,
                mamba_state_copy_funcs,
                mamba_group_ids,
                src_block_idx,
                dest_block_idx,
                accept_token_bias,
                req_state,
                forward_context,
            )
            if src_block_idx == dest_block_idx:
                num_accepted_tokens_cpu[i] = 1
    do_mamba_copy_block(copy_bufs)
