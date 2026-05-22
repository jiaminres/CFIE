# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

QWEN_REASONING_PREAMBLE_KWARG = "cfie_reasoning_preamble"

_REASONING_PREAMBLES = {
    "minimal": (
        "Current think mode: minimal. Think only if essential, keep internal "
        "reasoning to one very short note, then answer immediately."
    ),
    "low": (
        "Current think mode: low. Think fast, check only the key facts needed "
        "for correctness, and keep internal reasoning brief."
    ),
    "medium": (
        "Current think mode: medium. Reason through the essential steps, avoid "
        "exhaustive exploration, then answer clearly."
    ),
    "high": (
        "Current think mode: high. Spend more internal reasoning on important "
        "edge cases and verification before answering."
    ),
    "xhigh": (
        "Current think mode: xhigh. Use deep internal reasoning, verify edge "
        "cases carefully, and only then answer."
    ),
}

_VALID_EFFORTS = frozenset({"none", *_REASONING_PREAMBLES.keys()})


def normalize_reasoning_effort(effort: Any) -> str | None:
    if effort is None:
        return None

    normalized = str(effort).strip().lower()
    if normalized in ("", "auto"):
        return None
    if normalized not in _VALID_EFFORTS:
        return None
    return normalized


def build_reasoning_chat_template_kwargs(effort: Any) -> dict[str, Any]:
    """Map OpenAI-style reasoning effort to HF chat-template controls.

    Qwen3/Qwen3.5 templates use ``enable_thinking`` rather than
    ``reasoning_effort``. For enabled modes we also provide a small preamble
    that the renderer injects after the generation ``<think>`` marker when
    the active template supports that layout.
    """

    normalized = normalize_reasoning_effort(effort)
    if normalized is None:
        return {}

    kwargs: dict[str, Any] = {"reasoning_effort": normalized}
    if normalized == "none":
        kwargs["enable_thinking"] = False
        return kwargs

    kwargs["enable_thinking"] = True
    kwargs[QWEN_REASONING_PREAMBLE_KWARG] = _REASONING_PREAMBLES[normalized]
    return kwargs
