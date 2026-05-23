from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

TASK_STATUS_ACTIVE = "active"
TASK_STATUS_PAUSED = "paused"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"
TASK_STATUS_CANCELLED = "cancelled"
TASK_STATUS_SUPERSEDED = "superseded"

TASK_TYPE_PRIMARY = "primary"
TASK_TYPE_INTERRUPT = "interrupt"
TASK_TYPE_OVERRIDE = "override"
TASK_TYPE_RECOVERY = "recovery"
TASK_TYPE_VERIFICATION = "verification"
TASK_TYPE_MAINTENANCE = "maintenance"

TRANSITION_PUSH_INTERRUPT = "push_interrupt"
TRANSITION_POP_RESUME = "pop_resume"
TRANSITION_OVERRIDE = "override"
TRANSITION_ROLLBACK = "rollback"
TRANSITION_COMPLETE = "complete"
TRANSITION_FAIL = "fail"
TRANSITION_CANCEL = "cancel"


class TaskStateError(RuntimeError):
    pass


@dataclass(slots=True, frozen=True)
class TaskState:
    task_id: str
    goal: str
    type: str = TASK_TYPE_PRIMARY
    status: str = TASK_STATUS_ACTIVE
    success_condition: str | None = None
    failure_condition: str | None = None
    parent_task_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_status(self, status: str) -> "TaskState":
        return replace(self, status=status)

    def with_type(self, task_type: str) -> "TaskState":
        return replace(self, type=task_type)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "goal": self.goal,
            "type": self.type,
            "status": self.status,
            "success_condition": self.success_condition,
            "failure_condition": self.failure_condition,
            "parent_task_id": self.parent_task_id,
            "metadata": self.metadata,
        }


@dataclass(slots=True, frozen=True)
class TaskTransition:
    type: str
    from_task_id: str | None = None
    to_task_id: str | None = None
    reason: str | None = None
    checkpoint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "from_task_id": self.from_task_id,
            "to_task_id": self.to_task_id,
            "reason": self.reason,
            "checkpoint": self.checkpoint,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class TaskStack:
    tasks: list[TaskState] = field(default_factory=list)
    transitions: list[TaskTransition] = field(default_factory=list)

    @classmethod
    def with_root(cls, task: TaskState) -> "TaskStack":
        stack = cls()
        stack.add_root(task)
        return stack

    @property
    def active_task(self) -> TaskState | None:
        for task in reversed(self.tasks):
            if task.status == TASK_STATUS_ACTIVE:
                return task
        return None

    @property
    def active_task_id(self) -> str | None:
        task = self.active_task
        return task.task_id if task is not None else None

    def add_root(self, task: TaskState) -> TaskTransition:
        if self.active_task is not None:
            raise TaskStateError("cannot add root task while another task is active")
        task = task.with_status(TASK_STATUS_ACTIVE)
        self._ensure_unique_task_id(task.task_id)
        self.tasks.append(task)
        transition = TaskTransition(
            type=TRANSITION_PUSH_INTERRUPT,
            to_task_id=task.task_id,
            reason="root_task",
        )
        self.transitions.append(transition)
        return transition

    def push_interrupt(
        self,
        task: TaskState,
        *,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TaskTransition:
        active = self._require_active()
        self._replace_task(active.task_id, active.with_status(TASK_STATUS_PAUSED))
        self._ensure_unique_task_id(task.task_id)
        task = replace(
            task,
            type=task.type or TASK_TYPE_INTERRUPT,
            status=TASK_STATUS_ACTIVE,
            parent_task_id=task.parent_task_id or active.task_id,
        )
        if task.type == TASK_TYPE_PRIMARY:
            task = task.with_type(TASK_TYPE_INTERRUPT)
        self.tasks.append(task)
        transition = TaskTransition(
            type=TRANSITION_PUSH_INTERRUPT,
            from_task_id=active.task_id,
            to_task_id=task.task_id,
            reason=reason,
            metadata=dict(metadata or {}),
        )
        self.transitions.append(transition)
        return transition

    def pop_resume(
        self,
        *,
        reason: str | None = None,
        completed_status: str = TASK_STATUS_COMPLETED,
        metadata: dict[str, Any] | None = None,
    ) -> TaskTransition:
        active = self._require_active()
        self._replace_task(active.task_id, active.with_status(completed_status))
        paused = self._last_paused_task()
        if paused is not None:
            self._replace_task(paused.task_id, paused.with_status(TASK_STATUS_ACTIVE))
        transition = TaskTransition(
            type=TRANSITION_POP_RESUME,
            from_task_id=active.task_id,
            to_task_id=paused.task_id if paused is not None else None,
            reason=reason,
            metadata=dict(metadata or {}),
        )
        self.transitions.append(transition)
        return transition

    def override(
        self,
        task: TaskState,
        *,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TaskTransition:
        active = self._require_active()
        self._replace_task(active.task_id, active.with_status(TASK_STATUS_SUPERSEDED))
        self._ensure_unique_task_id(task.task_id)
        task_type = task.type if task.type != TASK_TYPE_PRIMARY else TASK_TYPE_OVERRIDE
        task = replace(task, type=task_type, status=TASK_STATUS_ACTIVE)
        self.tasks.append(task)
        transition = TaskTransition(
            type=TRANSITION_OVERRIDE,
            from_task_id=active.task_id,
            to_task_id=task.task_id,
            reason=reason,
            metadata=dict(metadata or {}),
        )
        self.transitions.append(transition)
        return transition

    def rollback(
        self,
        checkpoint: str,
        *,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TaskTransition:
        active = self._require_active()
        transition = TaskTransition(
            type=TRANSITION_ROLLBACK,
            from_task_id=active.task_id,
            checkpoint=checkpoint,
            reason=reason,
            metadata=dict(metadata or {}),
        )
        self.transitions.append(transition)
        return transition

    def complete_active(
        self,
        *,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TaskTransition:
        return self._finish_active(
            status=TASK_STATUS_COMPLETED,
            transition_type=TRANSITION_COMPLETE,
            reason=reason,
            metadata=metadata,
        )

    def fail_active(
        self,
        *,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TaskTransition:
        return self._finish_active(
            status=TASK_STATUS_FAILED,
            transition_type=TRANSITION_FAIL,
            reason=reason,
            metadata=metadata,
        )

    def cancel_active(
        self,
        *,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TaskTransition:
        return self._finish_active(
            status=TASK_STATUS_CANCELLED,
            transition_type=TRANSITION_CANCEL,
            reason=reason,
            metadata=metadata,
        )

    def to_context(self) -> list[dict[str, Any]]:
        return [task.to_dict() for task in self.tasks]

    def _finish_active(
        self,
        *,
        status: str,
        transition_type: str,
        reason: str | None,
        metadata: dict[str, Any] | None,
    ) -> TaskTransition:
        active = self._require_active()
        self._replace_task(active.task_id, active.with_status(status))
        transition = TaskTransition(
            type=transition_type,
            from_task_id=active.task_id,
            reason=reason,
            metadata=dict(metadata or {}),
        )
        self.transitions.append(transition)
        return transition

    def _require_active(self) -> TaskState:
        active = self.active_task
        if active is None:
            raise TaskStateError("task stack has no active task")
        return active

    def _last_paused_task(self) -> TaskState | None:
        for task in reversed(self.tasks):
            if task.status == TASK_STATUS_PAUSED:
                return task
        return None

    def _replace_task(self, task_id: str, new_task: TaskState) -> None:
        for index, task in enumerate(self.tasks):
            if task.task_id == task_id:
                self.tasks[index] = new_task
                return
        raise TaskStateError(f"unknown task_id: {task_id}")

    def _ensure_unique_task_id(self, task_id: str) -> None:
        if any(task.task_id == task_id for task in self.tasks):
            raise TaskStateError(f"duplicate task_id: {task_id}")
