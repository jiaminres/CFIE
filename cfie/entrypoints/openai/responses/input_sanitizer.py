from __future__ import annotations

import re
from typing import Any


INLINE_MEDIA_OMITTED = (
    "<inline media data omitted; provided separately if needed>"
)

_INLINE_MEDIA_DATA_URL_RE = re.compile(
    r"data:(?:image|video)/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=\r\n]+"
)


def strip_inline_media_from_text(text: str) -> str:
    if "data:image/" not in text and "data:video/" not in text:
        return text
    return _INLINE_MEDIA_DATA_URL_RE.sub(INLINE_MEDIA_OMITTED, text)


def strip_inline_media_from_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): strip_inline_media_from_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [strip_inline_media_from_value(item) for item in value]
    if isinstance(value, tuple):
        return [strip_inline_media_from_value(item) for item in value]
    if isinstance(value, str):
        return strip_inline_media_from_text(value)
    return value


def sanitize_responses_input_data(input_data: Any) -> Any:
    if isinstance(input_data, str):
        return strip_inline_media_from_text(input_data)
    if not isinstance(input_data, list):
        return input_data
    return [
        sanitize_responses_input_item(item)
        if isinstance(item, dict)
        else item
        for item in input_data
    ]


def sanitize_responses_input_item(item: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(item)
    item_type = sanitized.get("type")
    if item_type == "message" or (
        "role" in sanitized and "content" in sanitized
    ):
        sanitized["content"] = _sanitize_message_content(
            sanitized.get("content")
        )
    elif item_type == "function_call_output":
        output = sanitized.get("output")
        sanitized["output"] = strip_inline_media_from_value(output)
    elif item_type not in {"input_image", "input_video"}:
        text = sanitized.get("text")
        if isinstance(text, str):
            sanitized["text"] = strip_inline_media_from_text(text)
    return sanitized


def _sanitize_message_content(content: Any) -> Any:
    if isinstance(content, str):
        return strip_inline_media_from_text(content)
    if not isinstance(content, list):
        return content

    sanitized: list[Any] = []
    for part in content:
        if isinstance(part, str):
            sanitized.append(strip_inline_media_from_text(part))
            continue
        if not isinstance(part, dict):
            sanitized.append(part)
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
