from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from cfie_client import ComputerAction, ComputerCall, ProtocolError
from cfie_client.protocol import sort_actions_by_explicit_index
from cfie_gui_agent.tools import LEGACY_AGENT_TOOL_NAMES, MODEL_CALLABLE_TOOLS


PARSABLE_AGENT_TOOL_NAMES = (*MODEL_CALLABLE_TOOLS, *LEGACY_AGENT_TOOL_NAMES)


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
        try:
            arguments = _parse_arguments(_read_field(item, "arguments", {}))
        except AgentToolError as exc:
            raw_arguments = _read_field(item, "arguments", {})
            arguments = {
                "_parse_error": str(exc),
                "_raw_arguments": str(raw_arguments)[:1000],
            }
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
    items = tuple(_read_field(response_or_items, "output", response_or_items) or ())
    calls: list[AgentToolCall] = []
    for item in items or ():
        item_type = _read_field(item, "type")
        if item_type not in {"function_call", "tool_call"}:
            continue
        call = AgentToolCall.from_response_item(item)
        if call.name in PARSABLE_AGENT_TOOL_NAMES and call.name != "computer_use":
            calls.append(call)
        elif call.name == "computer_use":
            if "_parse_error" in call.arguments:
                calls.append(call)
            elif actions_error := _computer_use_actions_parse_error(call):
                calls.append(
                    AgentToolCall(
                        name=call.name,
                        call_id=call.call_id,
                        arguments={
                            "_parse_error": actions_error,
                            "_raw_arguments": str(call.arguments)[:1000],
                        },
                        raw=call.raw,
                    )
                )
    return tuple(calls)


def find_computer_tool_calls(response_or_items: Any) -> tuple[ComputerCall, ...]:
    items = tuple(_read_field(response_or_items, "output", response_or_items) or ())
    calls: list[ComputerCall] = []
    for item in items or ():
        item_type = _read_field(item, "type")
        if item_type not in {"function_call", "tool_call"}:
            continue
        if str(_read_field(item, "name", "")).strip() != "computer_use":
            continue
        call = AgentToolCall.from_response_item(item)
        actions_raw = call.arguments.get("actions", ())
        if isinstance(actions_raw, str):
            stripped = actions_raw.strip()
            if stripped:
                try:
                    actions_raw = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
        if isinstance(actions_raw, dict):
            actions_raw = [actions_raw]
        actions = []
        action_coordinate_spaces: list[str] = []
        for action in actions_raw or ():
            if _is_nested_agent_tool_action(action):
                continue
            if isinstance(action, dict):
                raw_coordinate_space = action.get("coordinate_space")
                if raw_coordinate_space is not None:
                    action_coordinate_spaces.append(str(raw_coordinate_space).strip())
            try:
                actions.append(ComputerAction.from_openai(action))
            except ProtocolError as exc:
                raise AgentToolError(
                    "invalid computer_use action "
                    f"{action!r} in arguments {call.arguments!r}: {exc}"
                ) from exc
        if actions:
            actions = list(sort_actions_by_explicit_index(tuple(actions)))
            call_index = _optional_positive_index(call.arguments.get("index"))
            coordinate_space = call.arguments.get("coordinate_space")
            if coordinate_space is not None:
                coordinate_space = _normalize_coordinate_space_text(coordinate_space)
            if coordinate_space is None and action_coordinate_spaces:
                unique_spaces = {
                    _normalize_coordinate_space_text(item)
                    for item in action_coordinate_spaces
                    if item
                }
                if len(unique_spaces) == 1:
                    coordinate_space = next(iter(unique_spaces))
            calls.append(
                ComputerCall(
                    call_id=call.call_id,
                    actions=tuple(actions),
                    index=call_index,
                    coordinate_space=coordinate_space or None,
                )
            )
    return tuple(calls)


def _normalize_coordinate_space_text(value: Any) -> str:
    text = str(value).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        try:
            decoded = json.loads(text) if text[0] == '"' else text[1:-1]
        except json.JSONDecodeError:
            decoded = text[1:-1]
        text = str(decoded).strip()
    return text


def _optional_positive_index(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise AgentToolError("index must be a positive integer")
    try:
        index = int(value)
    except (TypeError, ValueError) as exc:
        raise AgentToolError("index must be a positive integer") from exc
    if index < 1:
        raise AgentToolError("index must be a positive integer")
    return index


def _computer_use_actions_parse_error(call: AgentToolCall) -> str | None:
    actions_raw = call.arguments.get("actions", ())
    if not isinstance(actions_raw, str):
        return None
    stripped = actions_raw.strip()
    if not stripped:
        return None
    try:
        json.loads(stripped)
    except json.JSONDecodeError as exc:
        return (
            "computer_use.actions must be an array or valid JSON array string; "
            f"{exc.msg} at char {exc.pos}"
        )
    return None


def _is_nested_agent_tool_action(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    action_type = str(value.get("type") or "").strip()
    if action_type in {"call_function", "call_tool"}:
        function_spec = value.get("function")
        if isinstance(function_spec, dict):
            function_name = function_spec.get("name")
        else:
            function_name = None
        name = str(
            function_name
            or value.get("function_name")
            or value.get("tool_name")
            or value.get("name")
            or ""
        ).strip()
    else:
        name = str(value.get("type") or value.get("name") or "").strip()
    return name in PARSABLE_AGENT_TOOL_NAMES and name != "computer_use"


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
            parsed = _loads_jsonish_object(stripped)
        except json.JSONDecodeError as exc:
            snippet = stripped[:500].replace("\n", "\\n")
            raise AgentToolError(
                "agent tool call arguments must be JSON; "
                f"raw={snippet!r}"
            ) from exc
        if not isinstance(parsed, dict):
            raise AgentToolError("agent tool call arguments must decode to object")
        return parsed
    raise AgentToolError("agent tool call arguments must be dict or JSON string")


def _read_field(value: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(field_name, default)
    return getattr(value, field_name, default)


def _loads_jsonish_object(value: str) -> Any:
    attempts = [value]
    extracted = _extract_first_json_object(value)
    if extracted and extracted != value:
        attempts.append(extracted)
    for candidate in tuple(attempts):
        attempts.append(_repair_missing_commas(candidate))
    for candidate in tuple(attempts):
        attempts.append(_repair_truncated_json(candidate))
        attempts.append(_repair_missing_commas(_repair_truncated_json(candidate)))
    last_error: json.JSONDecodeError | None = None
    for candidate in dict.fromkeys(attempts):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def _extract_first_json_object(value: str) -> str | None:
    start = value.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(value[start:], start=start):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return value[start : index + 1]
    return None


def _repair_missing_commas(value: str) -> str:
    # Qwen occasionally emits {"actions": [...] "call_id": "..."}.
    return re.sub(r'(\]|\})(\s*")([A-Za-z_][^"]*"\s*:)', r"\1,\2\3", value)


def _repair_truncated_json(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        return stripped
    stack: list[str] = []
    in_string = False
    escaped = False
    for char in stripped:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char in "{[":
            stack.append(char)
        elif char == "}" and stack and stack[-1] == "{":
            stack.pop()
        elif char == "]" and stack and stack[-1] == "[":
            stack.pop()
    if in_string or not stack:
        return stripped
    while stripped.rstrip().endswith(","):
        stripped = stripped.rstrip()[:-1]
    closers = {"{": "}", "[": "]"}
    return stripped + "".join(closers[item] for item in reversed(stack))
