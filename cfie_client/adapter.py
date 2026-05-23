from __future__ import annotations

import json
import re
from typing import Any

from cfie_client.protocol import ComputerCall, ProtocolError

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class Qwen35ComputerAdapter:
    def to_computer_call(
        self,
        payload: dict[str, Any],
        *,
        default_call_id: str = "call_qwen35_local",
    ) -> ComputerCall:
        if payload.get("type") == "computer_call":
            return ComputerCall.from_openai(payload)
        return ComputerCall.from_openai(
            {
                "type": "computer_call",
                "call_id": payload.get("call_id", default_call_id),
                "actions": payload.get("actions", []),
                "status": payload.get("status", "completed"),
            }
        )

    def to_computer_call_from_text(
        self,
        text: str,
        *,
        default_call_id: str = "call_qwen35_local",
    ) -> ComputerCall:
        payload = self._parse_json_payload(text)
        if isinstance(payload, list):
            payload = {"actions": payload}
        if not isinstance(payload, dict):
            raise ProtocolError("Qwen computer output must be a JSON object or list")
        payload = self._normalize_payload(payload)
        return self.to_computer_call(payload, default_call_id=default_call_id)

    def _parse_json_payload(self, text: str) -> Any:
        stripped = text.strip()
        if not stripped:
            raise ProtocolError("Qwen computer output is empty")

        fence_match = _JSON_FENCE_RE.search(stripped)
        if fence_match:
            stripped = fence_match.group(1).strip()

        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            start_candidates = [
                index for index in (stripped.find("{"), stripped.find("[")) if index >= 0
            ]
            if not start_candidates:
                raise ProtocolError("Qwen computer output does not contain JSON") from None
            start = min(start_candidates)
            end = max(stripped.rfind("}"), stripped.rfind("]"))
            if end <= start:
                raise ProtocolError("Qwen computer output has incomplete JSON") from None
            try:
                return json.loads(stripped[start : end + 1])
            except json.JSONDecodeError as exc:
                raise ProtocolError("Qwen computer output JSON is invalid") from exc

    def _normalize_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(payload)
        actions = normalized.get("actions")
        if isinstance(actions, dict):
            actions = [actions]
        if isinstance(actions, list):
            normalized["actions"] = [
                self._normalize_action(action) for action in actions
            ]
        elif "action" in normalized:
            normalized["actions"] = [self._normalize_action(normalized)]
        return normalized

    def _normalize_action(self, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        action = dict(value)
        if "type" not in action and "action" in action:
            action["type"] = action["action"]
        coordinate = action.get("coordinate", action.get("coordinates"))
        if coordinate is not None and ("x" not in action or "y" not in action):
            try:
                x, y = coordinate
            except (TypeError, ValueError) as exc:
                raise ProtocolError("coordinate must be a pair [x, y]") from exc
            action["x"] = x
            action["y"] = y
        return action
