from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from cfie_gui_agent.tools import MODEL_CALLABLE_TOOLS


class AgentToolError(ValueError):
    pass


@dataclass(slots=True, frozen=True)
class AgentToolCall:
    name: str
    call_id: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_response_item(cls, item: Any) -> "AgentToolCall":
        item_type = _read_field(item, "type")
        if item_type not in {"function_call", "tool_call"}:
            raise AgentToolError("agent tool call item must be function_call/tool_call")
        name = str(_read_field(item, "name", "")).strip()
        if not name:
            raise AgentToolError("agent tool call requires name")
        call_id = str(
            _read_field(item, "call_id", _read_field(item, "id", name))
        ).strip()
        if not call_id:
            call_id = name
        arguments = _parse_arguments(_read_field(item, "arguments", {}))
        return cls(
            name=name,
            call_id=call_id,
            arguments=arguments,
            raw=dict(item) if isinstance(item, dict) else {},
        )

    def to_output_dict(self, output: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "function_call_output",
            "call_id": self.call_id,
            "output": output,
        }


def find_agent_tool_calls(response_or_items: Any) -> tuple[AgentToolCall, ...]:
    items = _read_field(response_or_items, "output", response_or_items)
    calls: list[AgentToolCall] = []
    for item in items or ():
        item_type = _read_field(item, "type")
        if item_type not in {"function_call", "tool_call"}:
            continue
        call = AgentToolCall.from_response_item(item)
        if call.name in MODEL_CALLABLE_TOOLS and call.name != "computer_use":
            calls.append(call)
    return tuple(calls)


def _parse_arguments(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise AgentToolError("agent tool call arguments must be JSON") from exc
        if not isinstance(parsed, dict):
            raise AgentToolError("agent tool call arguments must decode to object")
        return parsed
    raise AgentToolError("agent tool call arguments must be dict or JSON string")


def _read_field(value: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(field_name, default)
    return getattr(value, field_name, default)
