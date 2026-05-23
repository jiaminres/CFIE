from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

QUEUE_ACTIVE = "active"
QUEUE_URGENT = "urgent"
QUEUE_WAITING_HUMAN = "waiting_human"
QUEUE_BLOCKED = "blocked"
QUEUE_FINISHED = "finished"


@dataclass(slots=True, frozen=True)
class QueuedTask:
    task_id: str
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "kind": self.kind,
            "payload": self.payload,
            "priority": self.priority,
            "source": self.source,
            "metadata": self.metadata,
        }


@dataclass(slots=True, frozen=True)
class ScheduledDecision:
    task: QueuedTask | None
    queue: str | None
    reason: str


@dataclass(slots=True)
class AgentScheduler:
    urgent_queue: deque[QueuedTask] = field(default_factory=deque)
    active_queue: deque[QueuedTask] = field(default_factory=deque)
    waiting_human_queue: dict[str, QueuedTask] = field(default_factory=dict)
    blocked_queue: deque[QueuedTask] = field(default_factory=deque)
    finished_queue: list[QueuedTask] = field(default_factory=list)

    def enqueue_active(self, task: QueuedTask) -> None:
        self.active_queue.append(task)

    def enqueue_urgent(self, task: QueuedTask) -> None:
        self.urgent_queue.append(task)
        self._sort_urgent_queue()

    def enqueue_manager_reply(self, payload: dict[str, Any]) -> QueuedTask:
        request_id = str(payload.get("request", {}).get("request_id", ""))
        task = QueuedTask(
            task_id=f"manager_reply:{request_id or len(self.urgent_queue) + 1}",
            kind="manager_reply",
            payload=payload,
            priority=100,
            source="human_loop",
        )
        self.enqueue_urgent(task)
        return task

    def park_waiting_human(self, task: QueuedTask, *, request_id: str) -> None:
        self.waiting_human_queue[request_id] = task

    def resolve_waiting_human(self, request_id: str) -> QueuedTask | None:
        return self.waiting_human_queue.pop(request_id, None)

    def block_task(self, task: QueuedTask) -> None:
        self.blocked_queue.append(task)

    def finish_task(self, task: QueuedTask) -> None:
        self.finished_queue.append(task)

    def next_task(self, *, safe_to_interrupt: bool) -> ScheduledDecision:
        if safe_to_interrupt and self.urgent_queue:
            return ScheduledDecision(
                task=self.urgent_queue.popleft(),
                queue=QUEUE_URGENT,
                reason="urgent task selected at safe interruption point",
            )
        if self.active_queue:
            return ScheduledDecision(
                task=self.active_queue.popleft(),
                queue=QUEUE_ACTIVE,
                reason="active task selected",
            )
        if self.urgent_queue:
            return ScheduledDecision(
                task=None,
                queue=QUEUE_URGENT,
                reason="urgent task waiting for safe interruption point",
            )
        if self.waiting_human_queue:
            return ScheduledDecision(
                task=None,
                queue=QUEUE_WAITING_HUMAN,
                reason="all remaining tasks are waiting for human input",
            )
        return ScheduledDecision(task=None, queue=None, reason="no runnable task")

    def snapshot(self) -> dict[str, Any]:
        return {
            QUEUE_URGENT: [task.to_dict() for task in self.urgent_queue],
            QUEUE_ACTIVE: [task.to_dict() for task in self.active_queue],
            QUEUE_WAITING_HUMAN: {
                request_id: task.to_dict()
                for request_id, task in self.waiting_human_queue.items()
            },
            QUEUE_BLOCKED: [task.to_dict() for task in self.blocked_queue],
            QUEUE_FINISHED: [task.to_dict() for task in self.finished_queue],
        }

    def _sort_urgent_queue(self) -> None:
        ordered = sorted(
            self.urgent_queue,
            key=lambda task: task.priority,
            reverse=True,
        )
        self.urgent_queue = deque(ordered)
