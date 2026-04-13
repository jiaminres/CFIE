# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from cfie.triton_utils import HAS_TRITON, tl, triton

_UINT63_MASK = (1 << 63) - 1


def _mix_seed_and_pos(seed_value: int, pos_value: int) -> int:
    seed_value &= _UINT63_MASK
    pos_value &= _UINT63_MASK
    mixed = (
        seed_value
        ^ (
            pos_value
            + 0x9E3779B97F4A7C15
            + ((seed_value << 6) & _UINT63_MASK)
            + (seed_value >> 2)
        )
    ) & _UINT63_MASK
    return mixed or 1


def _make_row_generator(device: torch.device, seed_value: int) -> torch.Generator:
    generator_device: str | torch.device = "cpu"
    if device.type != "cpu":
        generator_device = device
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed_value)
    return generator


def _gumbel_sample_torch(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seed: torch.Tensor,
    pos: torch.Tensor,
    apply_temperature: bool,
    processed_logits_out: torch.Tensor | None = None,
) -> torch.Tensor:
    num_tokens, vocab_size = logits.shape
    sampled = torch.empty(num_tokens, device=logits.device, dtype=torch.int64)
    for token_idx in range(num_tokens):
        req_state_idx = int(expanded_idx_mapping[token_idx].item())
        row = logits[token_idx].to(torch.float32)
        temp = float(temperature[req_state_idx].item())
        if temp != 0.0 and apply_temperature:
            row = row / temp

        if processed_logits_out is not None:
            processed_logits_out[req_state_idx].copy_(
                row.to(dtype=processed_logits_out.dtype)
            )

        if temp != 0.0:
            generator = _make_row_generator(
                logits.device,
                _mix_seed_and_pos(
                    int(seed[req_state_idx].item()),
                    int(pos[token_idx].item()),
                ),
            )
            uniform = torch.rand(
                (vocab_size,),
                generator=generator,
                device=logits.device,
                dtype=torch.float32,
            ).clamp_(min=1e-7)
            row = row + (-torch.log(-torch.log(uniform)))

        sampled[token_idx] = torch.argmax(row, dim=-1)
    return sampled


@triton.jit
def _temperature_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    temperature_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    temperature = tl.load(temperature_ptr + req_state_idx).to(tl.float32)
    if temperature == 0.0 or temperature == 1.0:
        # Early return to avoid loading logits.
        return

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size

    logits = tl.load(logits_ptr + token_idx * logits_stride + block, mask=mask)
    logits = logits.to(tl.float32)
    logits = logits / temperature
    tl.store(logits_ptr + token_idx * logits_stride + block, logits, mask=mask)


def apply_temperature(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
) -> None:
    if not HAS_TRITON:
        req_temperature = temperature[expanded_idx_mapping].to(torch.float32)
        active = (req_temperature != 0.0) & (req_temperature != 1.0)
        if torch.any(active):
            logits[active] /= req_temperature[active].unsqueeze(1)
        return

    num_tokens, vocab_size = logits.shape
    BLOCK_SIZE = 8192
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    _temperature_kernel[(num_tokens, num_blocks)](
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        temperature,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )


@triton.jit
def _gumbel_sample_kernel(
    local_argmax_ptr,
    local_argmax_stride,
    local_max_ptr,
    local_max_stride,
    processed_logits_ptr,
    processed_logits_stride,
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    seeds_ptr,
    pos_ptr,
    temp_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    logits = tl.load(
        logits_ptr + token_idx * logits_stride + block,
        mask=mask,
        other=float("-inf"),
    )
    logits = logits.to(tl.float32)

    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    if temp != 0.0 and APPLY_TEMPERATURE:
        # Apply temperature.
        # NOTE(woosuk): Match the behavior of _temperature_kernel.
        # E.g., if the kernel uses tl.div_rn, we should use tl.div_rn here too.
        logits = logits / temp

    # Store the temperature-applied logits.
    if processed_logits_ptr is not None:
        tl.store(
            processed_logits_ptr + req_state_idx * processed_logits_stride + block,
            logits,
            mask=mask,
        )

    if temp != 0.0:
        # Calculate the seed for gumbel noise.
        seed = tl.load(seeds_ptr + req_state_idx)
        pos = tl.load(pos_ptr + token_idx)
        gumbel_seed = tl.randint(seed, pos)

        # Generate gumbel noise in FP32.
        u = tl.rand(gumbel_seed, block)
        u = tl.maximum(u, 1e-7)
        gumbel_noise = -tl.log(-tl.log(u))

        # Apply gumbel noise.
        logits = tl.where(mask, logits + gumbel_noise, float("-inf"))

    value, idx = tl.max(logits, axis=0, return_indices=True)
    token_id = block_idx * BLOCK_SIZE + idx
    tl.store(local_argmax_ptr + token_idx * local_argmax_stride + block_idx, token_id)
    tl.store(local_max_ptr + token_idx * local_max_stride + block_idx, value)


def gumbel_sample(
    logits: torch.Tensor,  # [num_tokens, vocab_size]
    expanded_idx_mapping: torch.Tensor,  # [num_tokens]
    temperature: torch.Tensor,  # [max_num_reqs]
    seed: torch.Tensor,  # [max_num_reqs]
    pos: torch.Tensor,  # [num_tokens]
    apply_temperature: bool,
    processed_logits_out: torch.Tensor | None = None,  # [num_reqs, vocab_size]
) -> torch.Tensor:
    if not HAS_TRITON:
        return _gumbel_sample_torch(
            logits,
            expanded_idx_mapping,
            temperature,
            seed,
            pos,
            apply_temperature,
            processed_logits_out,
        )

    num_tokens, vocab_size = logits.shape
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    local_argmax = logits.new_empty(num_tokens, num_blocks, dtype=torch.int64)
    local_max = logits.new_empty(num_tokens, num_blocks, dtype=torch.float32)
    _gumbel_sample_kernel[(num_tokens, num_blocks)](
        local_argmax,
        local_argmax.stride(0),
        local_max,
        local_max.stride(0),
        processed_logits_out,
        processed_logits_out.stride(0) if processed_logits_out is not None else 0,
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        seed,
        pos,
        temperature,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
        APPLY_TEMPERATURE=apply_temperature,
    )
    # NOTE(woosuk): Use int64 for later indexing.
    max_block_idx = local_max.argmax(dim=-1, keepdim=True)
    sampled = local_argmax.gather(dim=-1, index=max_block_idx).view(-1)
    return sampled
