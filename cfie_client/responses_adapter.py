from __future__ import annotations

import re
from typing import Any


INLINE_MEDIA_OMITTED = (
    "<inline media data omitted; provided separately if needed>"
)

_INLINE_MEDIA_DATA_URL_RE = re.compile(
    r"data:(?:image|video)/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=\r\n]+"
)


def strip_inline_media_from_tool_output(value: Any) -> Any:
    """Remove inline media blobs before a tool result is rendered to text."""

    if isinstance(value, dict):
        return {
            str(key): strip_inline_media_from_tool_output(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [strip_inline_media_from_tool_output(item) for item in value]
    if isinstance(value, tuple):
        return [strip_inline_media_from_tool_output(item) for item in value]
    if isinstance(value, str):
        return strip_inline_media_from_text(value)
    return value


def strip_inline_media_from_text(text: str) -> str:
    if "data:image/" not in text and "data:video/" not in text:
        return text
    return _INLINE_MEDIA_DATA_URL_RE.sub(INLINE_MEDIA_OMITTED, text)


def sanitize_responses_input_item(item: dict[str, Any]) -> dict[str, Any]:
    """Remove inline media blobs from text-bearing Responses input items.

    Real visual inputs must stay in typed media parts such as ``input_image``.
    The expensive bug this prevents is embedding the same base64 screenshot
    inside a runtime-context JSON text field, which turns one image into tens of
    thousands of text tokens before the model even sees the typed image part.
    """

    item_type = item.get("type")
    if item_type == "message":
        item["content"] = _sanitize_message_content(item.get("content"))
    elif item_type == "function_call_output":
        output = item.get("output")
        if isinstance(output, str):
            item["output"] = strip_inline_media_from_text(output)
        else:
            item["output"] = strip_inline_media_from_tool_output(output)
    return item


def sanitize_responses_input(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [sanitize_responses_input_item(item) for item in items]


def _sanitize_message_content(content: Any) -> Any:
    if isinstance(content, str):
        return strip_inline_media_from_text(content)
    if not isinstance(content, list):
        return content

    sanitized: list[Any] = []
    for part in content:
        if not isinstance(part, dict):
            sanitized.append(
                strip_inline_media_from_text(part)
                if isinstance(part, str)
                else part
            )
            continue
        part_type = part.get("type")
        if part_type in {"input_image", "input_video"}:
            sanitized.append(part)
            continue
        sanitized_part = dict(part)
        text = sanitized_part.get("text")
        if isinstance(text, str):
            sanitized_part["text"] = strip_inline_media_from_text(text)
        sanitized.append(sanitized_part)
    return sanitized
