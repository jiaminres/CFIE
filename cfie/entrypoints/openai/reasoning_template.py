# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

QWEN_REASONING_PREAMBLE_KWARG = "cfie_reasoning_preamble"

_REASONING_PREAMBLES = {
    "minimal": (
        "当前思考模式：minimal。只在思考区写一句极短状态判断，工具调用前"
        "不要输出可见状态文字。\n当前状态："
    ),
    "low": (
        "当前思考模式：low。只在思考区写两行：当前状态和下一步；不要描述"
        "无关画面，工具调用前不要输出可见状态文字。\n当前状态："
    ),
    "medium": (
        "当前思考模式：medium。只在思考区写三到四行：当前状态、不确定性、"
        "下一步和风险；每行很短，不枚举界面细节，工具调用前不要输出"
        "可见状态文字。\n当前状态："
    ),
    "high": (
        "当前思考模式：high。只在思考区核对当前状态、目标、失败风险和"
        "兜底动作；保持边界，不做长篇探索，工具调用前不要输出可见"
        "状态文字。\n当前状态："
    ),
    "xhigh": (
        "当前思考模式：xhigh。只在思考区做较深检查，覆盖边界情况后再"
        "行动；可见输出保持简短。\n当前状态："
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
