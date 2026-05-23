from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cfie_gui_agent.jobs import (
    SUBTASK_TYPE_MONITOR,
    JobBoard,
    JobBoardError,
    SubtaskState,
)


@dataclass(slots=True, frozen=True)
class MonitorEvent:
    event_id: str
    target_job_id: str
    subtask_goal: str
    event_type: str = "ui_event"
    priority: int = 0
    evidence_refs: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "target_job_id": self.target_job_id,
            "subtask_goal": self.subtask_goal,
            "event_type": self.event_type,
            "priority": self.priority,
            "evidence_refs": list(self.evidence_refs),
            "metadata": self.metadata,
        }


@dataclass(slots=True, frozen=True)
class MonitorIngestionResult:
    accepted: bool
    reason: str
    event: MonitorEvent
    subtask: SubtaskState | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "event": self.event.to_dict(),
            "subtask": self.subtask.to_dict() if self.subtask is not None else None,
        }


@dataclass(slots=True)
class MonitorController:
    def ingest_event(
        self,
        board: JobBoard,
        event: MonitorEvent,
    ) -> MonitorIngestionResult:
        try:
            board.require_job(event.target_job_id)
        except JobBoardError:
            return MonitorIngestionResult(
                accepted=False,
                reason=f"unknown target job: {event.target_job_id}",
                event=event,
            )
        if not event.subtask_goal.strip():
            return MonitorIngestionResult(
                accepted=False,
                reason="monitor event has empty subtask goal",
                event=event,
            )

        subtask = SubtaskState(
            subtask_id=f"monitor:{event.event_id}",
            job_id=event.target_job_id,
            goal=event.subtask_goal,
            type=SUBTASK_TYPE_MONITOR,
            priority=event.priority,
            evidence_refs=event.evidence_refs,
            metadata={
                "monitor_event": event.to_dict(),
                **event.metadata,
            },
        )
        board.add_subtask(subtask)
        return MonitorIngestionResult(
            accepted=True,
            reason="monitor event accepted",
            event=event,
            subtask=subtask,
        )
