"""CFIE 工具模块。

Keep a small, stable top-level API here because a number of migrated vLLM
modules still import common helpers directly from ``cfie.utils``.
"""

from __future__ import annotations

import uuid

import torch

from cfie.utils.logging import configure_logging, get_logger

MASK_64_BITS = (1 << 64) - 1


def random_uuid() -> str:
    """Return a compact request-safe identifier."""

    return f"{uuid.uuid4().int & MASK_64_BITS:016x}"


def length_from_prompt_token_ids_or_embeds(
    prompt_token_ids: list[int] | torch.Tensor | None,
    prompt_embeds: torch.Tensor | None,
) -> int:
    """Return prompt length from either token IDs or prompt embeddings."""

    prompt_token_len = None if prompt_token_ids is None else len(prompt_token_ids)
    prompt_embeds_len = None if prompt_embeds is None else len(prompt_embeds)

    if prompt_token_len is None:
        if prompt_embeds_len is None:
            raise ValueError("Neither prompt_token_ids nor prompt_embeds were defined.")
        return prompt_embeds_len

    if prompt_embeds_len is not None and prompt_embeds_len != prompt_token_len:
        raise ValueError(
            "Prompt token ids and prompt embeds had different lengths"
            f" prompt_token_ids={prompt_token_len}"
            f" prompt_embeds={prompt_embeds_len}"
        )
    return prompt_token_len


__all__ = [
    "MASK_64_BITS",
    "configure_logging",
    "get_logger",
    "length_from_prompt_token_ids_or_embeds",
    "random_uuid",
]
