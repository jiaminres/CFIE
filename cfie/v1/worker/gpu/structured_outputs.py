# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from cfie.triton_utils import HAS_TRITON, tl, triton
from cfie.utils.math_utils import cdiv
from cfie.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from cfie.v1.worker.gpu.input_batch import InputBatch


class StructuredOutputsWorker:
    def __init__(self, max_num_logits: int, vocab_size: int, device: torch.device):
        self.logits_indices = torch.zeros(
            max_num_logits, dtype=torch.int32, device=device
        )
        self.grammar_bitmask = torch.zeros(
            (max_num_logits, cdiv(vocab_size, 32)), dtype=torch.int32, device=device
        )
        self.device = device
        self.copy_stream = (
            torch.cuda.Stream()
            if device.type == "cuda" and torch.cuda.is_available()
            else None
        )

    def apply_grammar_bitmask(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        grammar_req_ids: list[str],
        grammar_bitmask: np.ndarray,
    ) -> None:
        if not grammar_req_ids:
            return

        if self.copy_stream is not None:
            with torch.cuda.stream(self.copy_stream):
                bitmask = async_copy_to_gpu(
                    grammar_bitmask,
                    out=self.grammar_bitmask[: grammar_bitmask.shape[0]],
                )
        else:
            bitmask = async_copy_to_gpu(
                grammar_bitmask, out=self.grammar_bitmask[: grammar_bitmask.shape[0]]
            )

        # Construct bitmask -> logits mapping
        mapping: list[int] = []
        req_ids = input_batch.req_ids
        cu_num_logits = input_batch.cu_num_logits_np.tolist()
        req_id_to_idx = {req_id: i for i, req_id in enumerate(req_ids)}
        for grammar_req_id in grammar_req_ids:
            req_idx = req_id_to_idx[grammar_req_id]
            logits_start_idx = cu_num_logits[req_idx]
            logits_end_idx = cu_num_logits[req_idx + 1]
            mapping.extend(range(logits_start_idx, logits_end_idx))

        if self.copy_stream is not None:
            with torch.cuda.stream(self.copy_stream):
                logits_indices = torch.tensor(
                    mapping, dtype=torch.int32, device="cpu", pin_memory=True
                )
                logits_indices = self.logits_indices[: len(mapping)].copy_(
                    logits_indices, non_blocking=True
                )

            current_stream = torch.cuda.current_stream()
            current_stream.wait_stream(self.copy_stream)
        else:
            logits_indices = torch.tensor(mapping, dtype=torch.int32, device=self.device)
            current_stream = None

        num_masks = bitmask.shape[0]
        assert num_masks == len(mapping)
        vocab_size = logits.shape[-1]
        if not HAS_TRITON:
            shifts = torch.arange(32, device=bitmask.device, dtype=torch.int64)
            unpacked = ((bitmask.to(torch.int64).unsqueeze(-1) >> shifts) & 1).ne(0)
            allowed = unpacked.reshape(num_masks, -1)[:, :vocab_size]
            selected_logits = logits.index_select(0, logits_indices.to(torch.long))
            selected_logits = selected_logits.masked_fill(~allowed, -float("inf"))
            logits.index_copy_(0, logits_indices.to(torch.long), selected_logits)
            return

        BLOCK_SIZE = 8192
        grid = (num_masks, triton.cdiv(vocab_size, BLOCK_SIZE))
        _apply_grammar_bitmask_kernel[grid](
            logits,
            logits.stride(0),
            logits_indices,
            bitmask,
            bitmask.stride(0),
            vocab_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # Ensure the copy stream waits for the device tensors to finish being used
        # before it re-uses or deallocates them
        if self.copy_stream is not None and current_stream is not None:
            self.copy_stream.wait_stream(current_stream)


# Adapted from
# https://github.com/mlc-ai/xgrammar/blob/main/python/xgrammar/kernels/apply_token_bitmask_inplace_triton.py
@triton.jit
def _apply_grammar_bitmask_kernel(
    logits_ptr,
    logits_stride,
    logits_indices_ptr,
    bitmask_ptr,
    bitmask_stride,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    bitmask_idx = tl.program_id(0)
    logits_idx = tl.load(logits_indices_ptr + bitmask_idx)

    # Load the bitmask.
    block_id = tl.program_id(1)
    bitmask_offset = (block_id * BLOCK_SIZE) // 32 + tl.arange(0, BLOCK_SIZE // 32)
    packed_bitmask = tl.load(
        bitmask_ptr + bitmask_idx * bitmask_stride + bitmask_offset,
        mask=bitmask_offset < bitmask_stride,
    )
    # Unpack the bitmask.
    bitmask = ((packed_bitmask[:, None] >> (tl.arange(0, 32)[None, :])) & 1) == 0
    bitmask = bitmask.reshape(BLOCK_SIZE)

    # Apply the bitmask to the logits.
    block_offset = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(
        logits_ptr + logits_idx * logits_stride + block_offset,
        -float("inf"),
        mask=bitmask & (block_offset < vocab_size),
    )
