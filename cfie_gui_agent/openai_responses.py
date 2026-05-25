from __future__ import annotations

import json
import urllib.error
import urllib.request
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from cfie_gui_agent.agent_tools import normalize_response_tool_calls
from cfie_gui_agent.tools import ModelToolRegistry


class OpenAIResponsesAgentError(RuntimeError):
    pass


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
    normalize_tool_calls: bool = True
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
        return self.create_response(payload)

    def create_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.base_url.rstrip("/") + "/responses",
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
                data = json.loads(raw)
                if self.normalize_tool_calls:
                    data = normalize_response_tool_calls(data)
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
        item = _normalize_computer_call_output(source)
        if item.get("type") == "message" and item.get("role") == "developer":
            item["role"] = "system"
        if item.get("type") == "function_call_output":
            output = item.get("output")
            if not isinstance(output, str):
                item["output"] = json.dumps(output, ensure_ascii=False)
        normalized.append(item)
    return normalized


def _normalize_computer_call_output(item: dict[str, Any]) -> dict[str, Any]:
    if item.get("type") != "computer_call_output":
        return item
    output = item.get("output")
    if not isinstance(output, dict):
        return {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "computer_use returned an empty screenshot output.",
                }
            ],
        }
    content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": (
                "computer_use completed. The next image is the screenshot after "
                f"call_id={item.get('call_id', '')}."
            ),
        }
    ]
    image_url = output.get("image_url")
    if image_url:
        content.append(
            {
                "type": "input_image",
                "image_url": image_url,
                "detail": output.get("detail", "low"),
            }
        )
    return {"type": "message", "role": "user", "content": content}
