from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from cfie_gui_agent.context import StepRecord

VERIFICATION_OK = "ok"
VERIFICATION_NO_SCREEN_CHANGE = "no_screen_change"
VERIFICATION_REPEATED_ACTION = "repeated_action"


@dataclass(slots=True, frozen=True)
class StepVerification:
    step_id: int
    status: str
    screen_changed: bool | None = None
    repeated_action_count: int = 1
    action_signature: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_ok(self) -> bool:
        return self.status == VERIFICATION_OK

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "status": self.status,
            "screen_changed": self.screen_changed,
            "repeated_action_count": self.repeated_action_count,
            "action_signature": self.action_signature,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class StepVerifier:
    max_repeated_actions: int = 3
    recent_action_signatures: deque[str] = field(default_factory=deque)

    def verify(self, step: StepRecord) -> StepVerification:
        signature = _action_signature(step.action)
        repeated = self._count_repeated(signature)
        self.recent_action_signatures.append(signature)
        while len(self.recent_action_signatures) > self.max_repeated_actions:
            self.recent_action_signatures.popleft()

        screen_changed: bool | None = None
        if step.before_ref and step.after_ref:
            screen_changed = step.before_ref != step.after_ref

        status = VERIFICATION_OK
        if screen_changed is False:
            status = VERIFICATION_NO_SCREEN_CHANGE
        if repeated >= self.max_repeated_actions:
            status = VERIFICATION_REPEATED_ACTION

        return StepVerification(
            step_id=step.step_id,
            status=status,
            screen_changed=screen_changed,
            repeated_action_count=repeated,
            action_signature=signature,
        )

    def _count_repeated(self, signature: str) -> int:
        count = 1
        for previous in reversed(self.recent_action_signatures):
            if previous != signature:
                break
            count += 1
        return count


def _action_signature(action: dict[str, Any]) -> str:
    return json.dumps(
        _strip_nonsemantic_action_fields(action),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _strip_nonsemantic_action_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_nonsemantic_action_fields(item)
            for key, item in value.items()
            if key not in {"call_id", "id", "status", "pending_safety_checks"}
        }
    if isinstance(value, list):
        return [_strip_nonsemantic_action_fields(item) for item in value]
    if isinstance(value, tuple):
        return [_strip_nonsemantic_action_fields(item) for item in value]
    return value
