from __future__ import annotations

from typing import Any

from cfie_client.protocol import ComputerCall


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
