from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

ACTION_TYPES = (
    "click",
    "double_click",
    "scroll",
    "type",
    "wait",
    "keypress",
    "drag",
    "move",
    "screenshot",
)

MOUSE_ACTION_TYPES = frozenset(
    {
        "click",
        "double_click",
        "drag",
        "move",
        "scroll",
    }
)

COORDINATE_SPACES = (
    "screenshot",
    "qwen_normalized_1000",
    "local_refinement_1000",
    "auto",
)


class ProtocolError(ValueError):
    pass


def _read_field(value: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(field_name, default)
    return getattr(value, field_name, default)


def _require_int(value: Any, field_name: str) -> int:
    if value is None:
        raise ProtocolError(f"{field_name} is required")
    if isinstance(value, bool):
        raise ProtocolError(f"{field_name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"{field_name} must be an integer") from exc


def _require_float(value: Any, field_name: str) -> float:
    if value is None:
        raise ProtocolError(f"{field_name} is required")
    if isinstance(value, bool):
        raise ProtocolError(f"{field_name} must be a number")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"{field_name} must be a number") from exc


def _normalize_keys(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        if "+" in value:
            return tuple(part for part in value.split("+") if part)
        return (value,)
    try:
        return tuple(str(item) for item in value)
    except TypeError as exc:
        raise ProtocolError("keys must be a string or an iterable of strings") from exc


def _normalize_path(value: Any) -> tuple[tuple[int, int], ...]:
    if value is None:
        raise ProtocolError("path is required")
    points: list[tuple[int, int]] = []
    for index, point in enumerate(value):
        if isinstance(point, dict):
            x = _require_int(point.get("x"), f"path[{index}].x")
            y = _require_int(point.get("y"), f"path[{index}].y")
        else:
            try:
                x_raw, y_raw = point
            except (TypeError, ValueError) as exc:
                raise ProtocolError(
                    f"path[{index}] must be a pair or a dict with x/y"
                ) from exc
            x = _require_int(x_raw, f"path[{index}].x")
            y = _require_int(y_raw, f"path[{index}].y")
        points.append((x, y))
    if len(points) < 2:
        raise ProtocolError("drag path must contain at least two points")
    return tuple(points)


@dataclass(slots=True, frozen=True)
class ComputerAction:
    type: str
    x: int | None = None
    y: int | None = None
    button: str | None = None
    keys: tuple[str, ...] = ()
    text: str | None = None
    path: tuple[tuple[int, int], ...] = ()
    scroll_x: int = 0
    scroll_y: int = 0
    duration: float | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_openai(cls, value: Any) -> "ComputerAction":
        value = _normalize_action_payload(value)
        action_type = str(_read_field(value, "type", "")).strip()
        if action_type not in ACTION_TYPES:
            raise ProtocolError(f"unsupported computer action type: {action_type!r}")

        x = _read_field(value, "x")
        y = _read_field(value, "y")
        button = _read_field(value, "button")
        keys = _normalize_keys(_read_field(value, "keys"))
        text = _read_field(value, "text")
        path = ()
        scroll_x = _read_field(value, "scroll_x", _read_field(value, "scrollX", 0))
        scroll_y = _read_field(value, "scroll_y", _read_field(value, "scrollY", 0))
        duration = _read_field(
            value,
            "seconds",
            _read_field(value, "duration", _read_field(value, "timeout", None)),
        )

        if action_type in {"click", "double_click", "move"}:
            x = _require_int(x, "x")
            y = _require_int(y, "y")
        else:
            x = None if x is None else _require_int(x, "x")
            y = None if y is None else _require_int(y, "y")

        if action_type in {"click", "double_click"}:
            button = str(button or "left")
            if button == "middle":
                button = "wheel"
            if button not in {"left", "right", "wheel", "back", "forward"}:
                raise ProtocolError(
                    "button must be left, right, wheel, back, or forward"
                )
        elif button is not None:
            button = str(button)

        if action_type == "drag":
            path = _normalize_path(_read_field(value, "path"))

        if action_type == "keypress" and not keys:
            raise ProtocolError("keypress requires at least one key")

        if action_type == "type":
            if text is None:
                raise ProtocolError("type action requires text")
            text = str(text)

        if action_type == "wait" and duration is not None:
            duration = _require_float(duration, "seconds")
            if duration < 0:
                raise ProtocolError("seconds must be non-negative")
        else:
            duration = None

        return cls(
            type=action_type,
            x=x,
            y=y,
            button=button,
            keys=keys,
            text=text,
            path=path,
            scroll_x=_require_int(scroll_x, "scrollX"),
            scroll_y=_require_int(scroll_y, "scrollY"),
            duration=duration,
            raw=dict(value) if isinstance(value, dict) else {},
        )

    def to_openai_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": self.type}
        if self.x is not None:
            payload["x"] = self.x
        if self.y is not None:
            payload["y"] = self.y
        if self.button is not None:
            payload["button"] = self.button
        if self.keys:
            payload["keys"] = list(self.keys)
        if self.text is not None:
            payload["text"] = self.text
        if self.path:
            payload["path"] = [{"x": x, "y": y} for x, y in self.path]
        if self.type == "scroll":
            payload["scroll_x"] = self.scroll_x
            payload["scroll_y"] = self.scroll_y
        if self.type == "wait" and self.duration is not None:
            payload["seconds"] = self.duration
        return payload

    @property
    def coordinate_points(self) -> tuple[tuple[int, int], ...]:
        if self.path:
            return self.path
        if self.x is not None and self.y is not None:
            return ((self.x, self.y),)
        return ()


def _normalize_action_payload(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    action = dict(value)
    if "type" not in action and "action" in action:
        action["type"] = action["action"]
    action_type = str(action.get("type", "")).strip()
    aliases = {
        "left_click": "click",
        "right_click": "click",
        "mouse_click": "click",
        "input": "type",
        "input_text": "type",
        "text": "type",
        "key": "keypress",
        "keys": "keypress",
        "key_press": "keypress",
        "press": "keypress",
        "hotkey": "keypress",
        "sleep": "wait",
    }
    if action_type in aliases:
        action["type"] = aliases[action_type]
        if action_type == "right_click" and "button" not in action:
            action["button"] = "right"
    coordinate = (
        action.get("coordinate")
        if "coordinate" in action
        else action.get("coordinates", action.get("point"))
    )
    if coordinate is not None and ("x" not in action or "y" not in action):
        try:
            x, y = coordinate
        except (TypeError, ValueError) as exc:
            raise ProtocolError("coordinate must be a pair [x, y]") from exc
        action["x"] = x
        action["y"] = y
    if action.get("type") == "type" and "text" not in action:
        text = action.get("content", action.get("value"))
        if text is not None:
            action["text"] = text
    if action.get("type") == "keypress" and "keys" not in action:
        key = action.get("key")
        if key is not None:
            action["keys"] = key
    if action.get("type") == "wait" and "seconds" not in action:
        seconds = action.get("duration", action.get("wait_time"))
        if seconds is not None:
            action["seconds"] = seconds
    return action


@dataclass(slots=True, frozen=True)
class ComputerCall:
    call_id: str
    actions: tuple[ComputerAction, ...]
    status: str | None = None
    coordinate_space: str | None = None
    pending_safety_checks: tuple[dict[str, Any], ...] = ()
    id: str | None = None

    @classmethod
    def from_openai(cls, value: Any) -> "ComputerCall":
        item_type = _read_field(value, "type")
        if item_type != "computer_call":
            raise ProtocolError("computer_call item must have type='computer_call'")

        call_id = str(_read_field(value, "call_id", "")).strip()
        if not call_id:
            raise ProtocolError("computer_call.call_id is required")

        actions_raw = _read_field(value, "actions")
        if actions_raw is None:
            action_raw = _read_field(value, "action")
            actions_raw = [] if action_raw is None else [action_raw]
        elif isinstance(actions_raw, dict):
            actions_raw = [actions_raw]

        actions = tuple(ComputerAction.from_openai(action) for action in actions_raw)
        coordinate_space = _read_field(value, "coordinate_space")
        if coordinate_space is not None:
            coordinate_space = _normalize_coordinate_space(coordinate_space)
            if coordinate_space not in COORDINATE_SPACES:
                raise ProtocolError(
                    "computer_call.coordinate_space must be one of "
                    f"{COORDINATE_SPACES}"
                )
        pending = tuple(
            dict(check)
            for check in (_read_field(value, "pending_safety_checks", ()) or ())
        )

        return cls(
            call_id=call_id,
            actions=actions,
            status=_read_field(value, "status"),
            coordinate_space=coordinate_space,
            pending_safety_checks=pending,
            id=_read_field(value, "id"),
        )

    def to_openai_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": "computer_call",
            "call_id": self.call_id,
            "actions": [action.to_openai_dict() for action in self.actions],
        }
        if self.coordinate_space is not None:
            payload["coordinate_space"] = self.coordinate_space
        if self.status is not None:
            payload["status"] = self.status
        if self.pending_safety_checks:
            payload["pending_safety_checks"] = list(self.pending_safety_checks)
        if self.id is not None:
            payload["id"] = self.id
        return payload


def _normalize_coordinate_space(value: Any) -> str:
    text = str(value).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        try:
            decoded = json.loads(text) if text[0] == '"' else text[1:-1]
        except json.JSONDecodeError:
            decoded = text[1:-1]
        text = str(decoded).strip()
    return text


@dataclass(slots=True, frozen=True)
class ComputerScreenshot:
    image_url: str | None = None
    file_id: str | None = None
    detail: str | None = "original"

    def to_openai_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": "computer_screenshot"}
        if self.image_url is not None:
            payload["image_url"] = self.image_url
        if self.file_id is not None:
            payload["file_id"] = self.file_id
        if self.detail is not None:
            payload["detail"] = self.detail
        return payload


@dataclass(slots=True, frozen=True)
class ComputerCallOutput:
    call_id: str
    output: ComputerScreenshot
    acknowledged_safety_checks: tuple[dict[str, Any], ...] = ()

    def to_openai_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": "computer_call_output",
            "call_id": self.call_id,
            "output": self.output.to_openai_dict(),
        }
        if self.acknowledged_safety_checks:
            payload["acknowledged_safety_checks"] = list(
                self.acknowledged_safety_checks
            )
        return payload


def build_computer_call_output(
    *,
    call_id: str,
    image_url: str | None = None,
    file_id: str | None = None,
    detail: str | None = "low",
    acknowledged_safety_checks: tuple[dict[str, Any], ...] | None = None,
) -> ComputerCallOutput:
    if not image_url and not file_id:
        raise ProtocolError("computer_call_output requires image_url or file_id")
    return ComputerCallOutput(
        call_id=call_id,
        output=ComputerScreenshot(image_url=image_url, file_id=file_id, detail=detail),
        acknowledged_safety_checks=tuple(acknowledged_safety_checks or ()),
    )


def find_computer_calls(response_or_items: Any) -> tuple[ComputerCall, ...]:
    items = _read_field(response_or_items, "output", response_or_items)
    calls: list[ComputerCall] = []
    for item in items or ():
        if _read_field(item, "type") == "computer_call":
            calls.append(ComputerCall.from_openai(item))
    return tuple(calls)
