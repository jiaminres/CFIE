from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any

from cfie_client import ComputerAction, ComputerCall, ProtocolError
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
    items = _response_items_with_text_tool_calls(response_or_items)
    calls: list[AgentToolCall] = []
    for item in items or ():
        item_type = _read_field(item, "type")
        if item_type not in {"function_call", "tool_call"}:
            continue
        call = AgentToolCall.from_response_item(item)
        if call.name in MODEL_CALLABLE_TOOLS and call.name != "computer_use":
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
            else:
                calls.extend(_nested_agent_tool_calls_from_computer_use(call))
    return tuple(calls)


def find_computer_tool_calls(response_or_items: Any) -> tuple[ComputerCall, ...]:
    items = _response_items_with_text_tool_calls(response_or_items)
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
                    coordinate_space=coordinate_space or None,
                )
            )
    return tuple(calls)


def normalize_response_tool_calls(response: Any) -> Any:
    """Return a Responses-style object with Qwen text tool calls split out.

    Qwen-style chat templates often make the model emit tool calls as text, for
    example ``<tool_call>...</tool_call>``.  The runner can parse those directly,
    but application code should receive ordinary assistant text and structured
    ``function_call`` items as separate response output entries.
    """
    if not isinstance(response, dict):
        return response
    output = response.get("output")
    if not isinstance(output, list):
        return response
    items = tuple(output)
    extracted_calls = (
        *_extract_text_tool_call_items(items),
        *_extract_tool_code_call_items(items),
        *_extract_bare_computer_action_items(items),
    )
    if not extracted_calls:
        return response

    normalized_output: list[Any] = []
    for item in output:
        stripped = _strip_text_tool_calls_from_item(item)
        if stripped is not None:
            normalized_output.append(stripped)

    seen = {
        _function_call_signature(item)
        for item in normalized_output
        if _read_field(item, "type") in {"function_call", "tool_call"}
    }
    for call in extracted_calls:
        standardized = _standardize_response_function_call(call)
        signature = _function_call_signature(standardized)
        if signature in seen:
            continue
        normalized_output.append(standardized)
        seen.add(signature)

    normalized = dict(response)
    normalized["output"] = normalized_output
    return normalized


def _response_items_with_text_tool_calls(response_or_items: Any) -> tuple[Any, ...]:
    items = tuple(_read_field(response_or_items, "output", response_or_items) or ())
    text_tool_calls = (
        *_extract_text_tool_call_items(items),
        *_extract_tool_code_call_items(items),
        *_extract_bare_computer_action_items(items),
    )
    if not text_tool_calls:
        return items
    return (*items, *text_tool_calls)


def _standardize_response_function_call(item: dict[str, Any]) -> dict[str, Any]:
    result = dict(item)
    arguments = result.get("arguments", {})
    if not isinstance(arguments, str):
        result["arguments"] = json.dumps(arguments, ensure_ascii=False)
    return result


def _function_call_signature(item: Any) -> tuple[str, str, str]:
    arguments = _read_field(item, "arguments", "")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
    return (
        str(_read_field(item, "type", "")),
        str(_read_field(item, "name", "")),
        arguments,
    )


def _strip_text_tool_calls_from_item(item: Any) -> Any | None:
    if not isinstance(item, dict):
        return item
    result = dict(item)
    if result.get("type") in {"output_text", "text"} and isinstance(result.get("text"), str):
        stripped = _strip_text_tool_call_blocks(result["text"])
        if not stripped and _has_text_tool_call_marker(result["text"]):
            return None
        result["text"] = stripped
        return result

    content = result.get("content")
    if not isinstance(content, list):
        return result
    normalized_content: list[Any] = []
    removed_tool_text = False
    for part in content:
        if (
            isinstance(part, dict)
            and part.get("type") in {"output_text", "text"}
            and isinstance(part.get("text"), str)
        ):
            stripped = _strip_text_tool_call_blocks(part["text"])
            if not stripped and _has_text_tool_call_marker(part["text"]):
                removed_tool_text = True
                continue
            updated = dict(part)
            updated["text"] = stripped
            normalized_content.append(updated)
            removed_tool_text = removed_tool_text or stripped != part["text"]
        else:
            normalized_content.append(part)
    if result.get("type") == "message" and not normalized_content and removed_tool_text:
        return None
    result["content"] = normalized_content
    return result


def _has_text_tool_call_marker(text: str) -> bool:
    lower = text.lower()
    return "<tool_call" in lower or "<tool_code" in lower or "<function=" in lower


def _strip_text_tool_call_blocks(text: str) -> str:
    stripped = re.sub(
        r"<tool_call\b[^>]*>.*?</tool_call>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    stripped = re.sub(
        r"<tool_code\b[^>]*>.*?</tool_code>",
        "",
        stripped,
        flags=re.IGNORECASE | re.DOTALL,
    )
    stripped = re.sub(
        r"<function=([A-Za-z_][A-Za-z0-9_]*)>.*?</function>",
        "",
        stripped,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return stripped.strip()


def _normalize_coordinate_space_text(value: Any) -> str:
    text = str(value).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        try:
            decoded = json.loads(text) if text[0] == '"' else text[1:-1]
        except json.JSONDecodeError:
            decoded = text[1:-1]
        text = str(decoded).strip()
    return text


def _extract_text_tool_call_items(items: tuple[Any, ...]) -> tuple[dict[str, Any], ...]:
    calls: list[dict[str, Any]] = []
    for item in items:
        for text in _iter_response_texts(item):
            for block in re.findall(
                r"<tool_call>(.*?)</tool_call>",
                text,
                flags=re.IGNORECASE | re.DOTALL,
            ):
                for name, body in re.findall(
                    r"<function=([A-Za-z_][A-Za-z0-9_]*)>(.*?)</function>",
                    block,
                    flags=re.IGNORECASE | re.DOTALL,
                ):
                    arguments = _parse_text_tool_arguments(body)
                    calls.append(
                        {
                            "type": "function_call",
                            "name": name.strip(),
                            "call_id": f"text_tool_call_{len(calls) + 1}",
                            "arguments": arguments,
                        }
                    )
            bare_text = re.sub(
                r"<tool_call\b[^>]*>.*?</tool_call>",
                "",
                text,
                flags=re.IGNORECASE | re.DOTALL,
            )
            for name, body in re.findall(
                r"<function=([A-Za-z_][A-Za-z0-9_]*)>(.*?)</function>",
                bare_text,
                flags=re.IGNORECASE | re.DOTALL,
            ):
                arguments = _parse_text_tool_arguments(body)
                calls.append(
                    {
                        "type": "function_call",
                        "name": name.strip(),
                        "call_id": f"text_tool_call_{len(calls) + 1}",
                        "arguments": arguments,
                    }
                )
    return tuple(calls)


def _extract_tool_code_call_items(items: tuple[Any, ...]) -> tuple[dict[str, Any], ...]:
    calls: list[dict[str, Any]] = []
    for item in items:
        for text in _iter_response_texts(item):
            for block in re.findall(
                r"<tool_code>(.*?)</tool_code>",
                text,
                flags=re.IGNORECASE | re.DOTALL,
            ):
                calls.extend(_parse_python_tool_code(block, offset=len(calls)))
    return tuple(calls)


def _parse_python_tool_code(code: str, *, offset: int = 0) -> tuple[dict[str, Any], ...]:
    stripped = code.strip()
    if not stripped:
        return ()
    try:
        tree = ast.parse(stripped, mode="exec")
    except SyntaxError:
        return ()
    calls: list[dict[str, Any]] = []
    for statement in tree.body:
        value = statement.value if isinstance(statement, ast.Expr) else statement
        if not isinstance(value, ast.Call):
            continue
        call = _unwrap_print_tool_call(value)
        name = _call_name(call)
        if name not in MODEL_CALLABLE_TOOLS:
            continue
        calls.append(
            {
                "type": "function_call",
                "name": name,
                "call_id": f"text_tool_code_{offset + len(calls) + 1}",
                "arguments": _python_call_arguments(name, call),
            }
        )
    return tuple(calls)


def _unwrap_print_tool_call(call: ast.Call) -> ast.Call:
    if (
        isinstance(call.func, ast.Name)
        and call.func.id == "print"
        and len(call.args) == 1
        and isinstance(call.args[0], ast.Call)
    ):
        return call.args[0]
    return call


def _call_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return ""


def _python_call_arguments(tool_name: str, call: ast.Call) -> dict[str, Any]:
    arguments: dict[str, Any] = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            continue
        arguments[keyword.arg] = _literal_ast_value(keyword.value)
    if call.args:
        values = [_literal_ast_value(arg) for arg in call.args]
        if len(values) == 1 and isinstance(values[0], dict):
            arguments.update(values[0])
        elif len(values) == 1 and tool_name == "computer_use":
            arguments.setdefault("actions", values[0])
        else:
            arguments.setdefault("_args", values)
    return arguments


def _literal_ast_value(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        try:
            return ast.unparse(node)
        except Exception:
            return ""


def _extract_bare_computer_action_items(
    items: tuple[Any, ...],
) -> tuple[dict[str, Any], ...]:
    calls: list[dict[str, Any]] = []
    for item in items:
        for text in _iter_response_texts(item):
            stripped = text.strip()
            if not stripped or "<tool_call" in stripped.lower():
                continue
            if stripped[0] not in "[{":
                continue
            payload = _parse_text_parameter_value(stripped)
            if isinstance(payload, list):
                arguments = {"actions": payload}
            elif isinstance(payload, dict) and "actions" in payload:
                arguments = payload
            else:
                continue
            calls.append(
                {
                    "type": "function_call",
                    "name": "computer_use",
                    "call_id": f"bare_text_computer_call_{len(calls) + 1}",
                    "arguments": arguments,
                }
            )
    return tuple(calls)


def _iter_response_texts(item: Any) -> tuple[str, ...]:
    if not isinstance(item, dict):
        return ()
    texts: list[str] = []
    if item.get("type") in {"output_text", "text"} and isinstance(item.get("text"), str):
        texts.append(item["text"])
    content = item.get("content")
    if isinstance(content, list):
        for part in content:
            if (
                isinstance(part, dict)
                and part.get("type") in {"output_text", "text"}
                and isinstance(part.get("text"), str)
            ):
                texts.append(part["text"])
    return tuple(texts)


def _parse_text_tool_arguments(body: str) -> dict[str, Any]:
    arguments: dict[str, Any] = {}
    for param_name, raw_value in re.findall(
        r"<parameter=([A-Za-z_][A-Za-z0-9_]*)>(.*?)</parameter>",
        body,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        arguments[param_name.strip()] = _parse_text_parameter_value(raw_value)
    return arguments


def _parse_text_parameter_value(raw_value: str) -> Any:
    stripped = raw_value.strip()
    if not stripped:
        return ""
    if stripped[0] in "[{":
        for candidate in dict.fromkeys(
            (stripped, _repair_missing_y_coordinate(stripped))
        ):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
        if stripped[0] == "{":
            try:
                return _loads_jsonish_object(stripped)
            except json.JSONDecodeError:
                return stripped
        return stripped
    return stripped


def _nested_agent_tool_calls_from_computer_use(
    call: AgentToolCall,
) -> tuple[AgentToolCall, ...]:
    actions_raw = call.arguments.get("actions", ())
    if isinstance(actions_raw, str):
        stripped = actions_raw.strip()
        if not stripped:
            return ()
        try:
            actions_raw = json.loads(stripped)
        except json.JSONDecodeError:
            return ()
    if isinstance(actions_raw, dict):
        actions_raw = [actions_raw]
    calls: list[AgentToolCall] = []
    for index, action in enumerate(actions_raw or ()):
        if not _is_nested_agent_tool_action(action):
            continue
        assert isinstance(action, dict)
        action_type = str(action.get("type") or "").strip()
        function_spec = action.get("function")
        if action_type in {"call_function", "call_tool"}:
            if isinstance(function_spec, dict):
                name = str(
                    function_spec.get("name")
                    or action.get("function_name")
                    or action.get("tool_name")
                    or action.get("name")
                    or ""
                ).strip()
                parameters = (
                    function_spec.get("parameters")
                    if "parameters" in function_spec
                    else function_spec.get("arguments")
                )
            else:
                name = str(
                    action.get("function_name")
                    or action.get("tool_name")
                    or action.get("name")
                    or ""
                ).strip()
                parameters = (
                    action.get("parameters")
                    if "parameters" in action
                    else action.get("arguments")
                )
            arguments = dict(parameters) if isinstance(parameters, dict) else {}
            if parameters is not None and not isinstance(parameters, dict):
                arguments["parameters"] = parameters
        else:
            name = str(action.get("type") or action.get("name")).strip()
            arguments = {
                key: value
                for key, value in action.items()
                if key not in {"type", "name", "call_id"}
            }
        nested_call_id = str(
            action.get("call_id")
            or f"{call.call_id}:{name}:{index}"
        )
        calls.append(
            AgentToolCall(
                name=name,
                call_id=nested_call_id,
                arguments=arguments,
                raw=dict(action),
            )
        )
    return tuple(calls)


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
    return name in MODEL_CALLABLE_TOOLS and name != "computer_use"


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


def _repair_missing_y_coordinate(value: str) -> str:
    # Qwen sometimes emits {"type":"click","x":313,464}.
    # It can also continue with another field:
    # {"type":"click","x":313,464,"button":"left"}.
    return re.sub(
        r'("x"\s*:\s*-?\d+)\s*,\s*(-?\d+)(\s*(?:,\s*"[A-Za-z_][^"]*"\s*:|[}\]]))',
        r'\1, "y": \2\3',
        value,
    )


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
