from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from time import time_ns
from typing import Any

from cfie_gui_agent.context import StepRecord


@dataclass(slots=True, frozen=True)
class AgentTraceEvent:
    kind: str
    payload: dict[str, Any]
    time_unix_nano: int = field(default_factory=time_ns)

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_unix_nano": self.time_unix_nano,
            "kind": self.kind,
            "payload": self.payload,
        }


@dataclass(slots=True)
class AgentTraceStore:
    path: Path | None = None
    events: list[AgentTraceEvent] = field(default_factory=list)

    def record(self, kind: str, payload: dict[str, Any]) -> AgentTraceEvent:
        event = AgentTraceEvent(kind=kind, payload=payload)
        self.events.append(event)
        self._write_event(event)
        return event

    def record_step(self, step: StepRecord) -> AgentTraceEvent:
        return self.record("step", step.to_summary_dict())

    def record_policy_update(self, payload: dict[str, Any]) -> AgentTraceEvent:
        return self.record("policy_update", payload)

    def record_result(self, payload: dict[str, Any]) -> AgentTraceEvent:
        return self.record("result", payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path) if self.path is not None else None,
            "event_count": len(self.events),
            "recent_events": [event.to_dict() for event in self.events[-20:]],
        }

    def _write_event(self, event: AgentTraceEvent) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
