from __future__ import annotations

import json
import urllib.error
import urllib.request
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from cfie_client.responses_adapter import (
    sanitize_responses_input_item,
    strip_inline_media_from_text,
    strip_inline_media_from_tool_output,
)
from cfie_gui_agent.tools import ModelToolRegistry


class OpenAIResponsesAgentError(RuntimeError):
    pass


REQUEST_DEBUG_KEY = "_cfie_request_debug"


@dataclass(slots=True)
class OpenAIResponsesAgent:
    model: str
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = ""
    tool_registry: ModelToolRegistry = field(default_factory=ModelToolRegistry)
    max_output_tokens: int = 512
    temperature: float = 0.0
    timeout: float = 600.0
    store: bool = False
    tool_choice: str = "auto"
    include_tools: bool = True
    parallel_tool_calls: bool = False
    reasoning_effort: str | None = "none"
    chat_template_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"enable_thinking": False}
    )

    def __call__(self, conversation: list[dict[str, Any]]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "input": _normalize_responses_input(conversation),
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "store": self.store,
        }
        if self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}
        if self.chat_template_kwargs:
            payload["chat_template_kwargs"] = self.chat_template_kwargs
        if self.include_tools:
            payload["tools"] = [
                _to_responses_tool(tool) for tool in self.tool_registry.model_tools()
            ]
            payload["tool_choice"] = self.tool_choice
            payload["parallel_tool_calls"] = self.parallel_tool_calls
        return self.create_response(payload)

    def create_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_debug = _build_request_debug_payload(payload)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            _responses_endpoint(self.base_url),
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
                data = json.loads(raw)
                if isinstance(data, dict):
                    data[REQUEST_DEBUG_KEY] = request_debug
                return data
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {"error": raw}
            raise OpenAIResponsesAgentError(
                f"Responses API request failed with HTTP {exc.code}: {payload}"
            ) from exc
        except urllib.error.URLError as exc:
            raise OpenAIResponsesAgentError(
                f"Responses API request failed: {exc.reason}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise OpenAIResponsesAgentError(
                "Responses API returned non-JSON response"
            ) from exc


def _responses_endpoint(base_url: str) -> str:
    root = base_url.rstrip("/")
    if not root:
        return "/v1/responses"
    parsed = urlparse(root)
    path = parsed.path.rstrip("/")
    if path.endswith("/responses"):
        return root
    if path.endswith("/v1"):
        return root + "/responses"
    return root + "/v1/responses"


def _to_responses_tool(tool: Any) -> dict[str, Any]:
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
    }


def _normalize_responses_input(
    conversation: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for source in deepcopy(conversation):
        item = source
        if item.get("type") == "message" and item.get("role") == "developer":
            item["role"] = "system"
        if item.get("type") == "function_call_output":
            output = item.get("output")
            if not isinstance(output, str):
                output = strip_inline_media_from_tool_output(output)
                item["output"] = json.dumps(output, ensure_ascii=False)
            else:
                item["output"] = strip_inline_media_from_text(output)
        item = sanitize_responses_input_item(item)
        normalized.append(item)
    return normalized


def _build_request_debug_payload(payload: dict[str, Any]) -> dict[str, Any]:
    tools = payload.get("tools")
    tool_names: list[str] = []
    if isinstance(tools, list):
        for tool in tools:
            if isinstance(tool, dict):
                name = tool.get("name")
                if isinstance(name, str) and name:
                    tool_names.append(name)
    return {
        "model": payload.get("model"),
        "temperature": payload.get("temperature"),
        "max_output_tokens": payload.get("max_output_tokens"),
        "tool_choice": payload.get("tool_choice"),
        "parallel_tool_calls": payload.get("parallel_tool_calls"),
        "reasoning": _redact_request_media(payload.get("reasoning")),
        "chat_template_kwargs": _redact_request_media(
            payload.get("chat_template_kwargs")
        ),
        "tools": tool_names,
        "input": _redact_request_media(payload.get("input", [])),
    }


def _redact_request_media(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"image_url", "video_url"}:
                redacted[key] = _media_placeholder(
                    item,
                    kind="image" if key == "image_url" else "video",
                )
            else:
                redacted[str(key)] = _redact_request_media(item)
        return redacted
    if isinstance(value, list):
        return [_redact_request_media(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_request_media(item) for item in value]
    if isinstance(value, str):
        return strip_inline_media_from_text(value)
    return value


def _media_placeholder(value: Any, *, kind: str) -> dict[str, Any]:
    label = "[图片]" if kind == "image" else "[视频]"
    if isinstance(value, dict):
        result: dict[str, Any] = {
            "placeholder": label,
            "source_type": str(value.get("type") or "object"),
        }
        for key in ("mime_type", "bytes", "sha256", "path", "detail"):
            if key in value:
                result[key] = value[key]
        return result
    text = str(value or "")
    result = {"placeholder": label}
    if text.startswith("data:"):
        header = text.split(",", 1)[0]
        mime_type = header.removeprefix("data:").split(";", 1)[0]
        result.update(
            {
                "source_type": "data_url",
                "mime_type": mime_type,
                "chars": len(text),
            }
        )
        return result
    result.update({"source_type": "reference", "ref": text})
    return result
