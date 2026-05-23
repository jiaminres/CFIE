from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any

from cfie_gui_agent.context import (
    CompactionApplication,
    ContextManager,
    LongHistorySummary,
    PromptContextSelection,
    StepRecord,
)

JOB_STATUS_ACTIVE = "active"
JOB_STATUS_PAUSED = "paused"
JOB_STATUS_DISABLED = "disabled"
JOB_STATUS_COMPLETED = "completed"
JOB_STATUS_FAILED = "failed"

SUBTASK_STATUS_RUNNING = "running"
SUBTASK_STATUS_RUNNABLE = "runnable"
SUBTASK_STATUS_WAITING_HUMAN = "waiting_human"
SUBTASK_STATUS_BLOCKED = "blocked"
SUBTASK_STATUS_COMPLETED = "completed"
SUBTASK_STATUS_FAILED = "failed"
SUBTASK_STATUS_CANCELLED = "cancelled"
SUBTASK_STATUS_SUPERSEDED = "superseded"

SUBTASK_TYPE_NORMAL = "normal"
SUBTASK_TYPE_INTERRUPT = "interrupt"
SUBTASK_TYPE_RECOVERY = "recovery"
SUBTASK_TYPE_VERIFICATION = "verification"
SUBTASK_TYPE_MAINTENANCE = "maintenance"
SUBTASK_TYPE_MONITOR = "monitor"
SUBTASK_TYPE_MANAGER_REPLY = "manager_reply"


class JobBoardError(RuntimeError):
    pass


@dataclass(slots=True, frozen=True)
class SubtaskState:
    subtask_id: str
    job_id: str
    goal: str
    type: str = SUBTASK_TYPE_NORMAL
    status: str = SUBTASK_STATUS_RUNNABLE
    priority: int = 0
    safety_level: str = "normal"
    human_request_id: str | None = None
    retry_count: int = 0
    max_retries: int = 3
    evidence_refs: tuple[str, ...] = ()
    success_condition: str | None = None
    failure_condition: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_status(self, status: str) -> "SubtaskState":
        return replace(self, status=status)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subtask_id": self.subtask_id,
            "job_id": self.job_id,
            "goal": self.goal,
            "type": self.type,
            "status": self.status,
            "priority": self.priority,
            "safety_level": self.safety_level,
            "human_request_id": self.human_request_id,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "evidence_refs": list(self.evidence_refs),
            "success_condition": self.success_condition,
            "failure_condition": self.failure_condition,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class SubtaskQueues:
    running: SubtaskState | None = None
    urgent: deque[SubtaskState] = field(default_factory=deque)
    runnable: deque[SubtaskState] = field(default_factory=deque)
    waiting_human: dict[str, SubtaskState] = field(default_factory=dict)
    blocked: dict[str, SubtaskState] = field(default_factory=dict)
    completed: list[SubtaskState] = field(default_factory=list)
    failed: list[SubtaskState] = field(default_factory=list)
    cancelled: list[SubtaskState] = field(default_factory=list)
    superseded: list[SubtaskState] = field(default_factory=list)

    def add(self, subtask: SubtaskState) -> None:
        if subtask.status == SUBTASK_STATUS_RUNNING:
            self.set_running(subtask)
            return
        if subtask.status == SUBTASK_STATUS_WAITING_HUMAN:
            key = subtask.human_request_id or subtask.subtask_id
            self.waiting_human[key] = subtask
            return
        if subtask.status == SUBTASK_STATUS_BLOCKED:
            self.blocked[subtask.subtask_id] = subtask
            return
        if subtask.status == SUBTASK_STATUS_COMPLETED:
            self.completed.append(subtask)
            return
        if subtask.status == SUBTASK_STATUS_FAILED:
            self.failed.append(subtask)
            return
        if subtask.status == SUBTASK_STATUS_CANCELLED:
            self.cancelled.append(subtask)
            return
        if subtask.status == SUBTASK_STATUS_SUPERSEDED:
            self.superseded.append(subtask)
            return
        if subtask.priority >= 100:
            self.urgent.append(subtask.with_status(SUBTASK_STATUS_RUNNABLE))
            self._sort_urgent()
            return
        self.runnable.append(subtask.with_status(SUBTASK_STATUS_RUNNABLE))
        self._sort_runnable()

    def set_running(self, subtask: SubtaskState) -> None:
        if self.running is not None and self.running.subtask_id != subtask.subtask_id:
            raise JobBoardError("a subtask is already running")
        self.running = subtask.with_status(SUBTASK_STATUS_RUNNING)

    def pop_next(self, *, safe_to_interrupt: bool) -> SubtaskState | None:
        if (
            self.running is not None
            and not safe_to_interrupt
            and self.running.status == SUBTASK_STATUS_RUNNING
        ):
            return self.running
        if self.urgent:
            subtask = self.urgent.popleft().with_status(SUBTASK_STATUS_RUNNING)
            self.running = subtask
            return subtask
        if self.running is not None and self.running.status == SUBTASK_STATUS_RUNNING:
            return self.running
        if self.runnable:
            subtask = self.runnable.popleft().with_status(SUBTASK_STATUS_RUNNING)
            self.running = subtask
            return subtask
        return None

    def move_running_to(self, status: str, *, human_request_id: str | None = None) -> SubtaskState:
        if self.running is None:
            raise JobBoardError("no running subtask")
        subtask = replace(
            self.running,
            status=status,
            human_request_id=human_request_id or self.running.human_request_id,
        )
        self.running = None
        self.add(subtask)
        return subtask

    def promote_waiting_human(
        self,
        request_id: str,
        *,
        priority: int = 100,
        metadata: dict[str, Any] | None = None,
    ) -> SubtaskState | None:
        subtask = self.waiting_human.pop(request_id, None)
        if subtask is None:
            return None
        subtask = replace(
            subtask,
            status=SUBTASK_STATUS_RUNNABLE,
            priority=priority,
            metadata={**subtask.metadata, **dict(metadata or {})},
        )
        self.urgent.append(subtask)
        self._sort_urgent()
        return subtask

    def counts(self) -> dict[str, int]:
        return {
            "running": 1 if self.running else 0,
            "urgent": len(self.urgent),
            "runnable": len(self.runnable),
            "waiting_human": len(self.waiting_human),
            "blocked": len(self.blocked),
            "completed": len(self.completed),
            "failed": len(self.failed),
            "cancelled": len(self.cancelled),
            "superseded": len(self.superseded),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "running": self.running.to_dict() if self.running else None,
            "urgent": [subtask.to_dict() for subtask in self.urgent],
            "runnable": [subtask.to_dict() for subtask in self.runnable],
            "waiting_human": {
                key: subtask.to_dict()
                for key, subtask in self.waiting_human.items()
            },
            "blocked": {
                key: subtask.to_dict() for key, subtask in self.blocked.items()
            },
            "completed": [subtask.to_dict() for subtask in self.completed],
            "failed": [subtask.to_dict() for subtask in self.failed],
            "cancelled": [subtask.to_dict() for subtask in self.cancelled],
            "superseded": [subtask.to_dict() for subtask in self.superseded],
        }

    def _sort_urgent(self) -> None:
        self.urgent = deque(
            sorted(self.urgent, key=lambda subtask: subtask.priority, reverse=True)
        )

    def _sort_runnable(self) -> None:
        self.runnable = deque(
            sorted(self.runnable, key=lambda subtask: subtask.priority, reverse=True)
        )


@dataclass(slots=True)
class JobState:
    job_id: str
    target_app: str
    goal: str
    priority: int = 0
    status: str = JOB_STATUS_ACTIVE
    queues: SubtaskQueues = field(default_factory=SubtaskQueues)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_enabled(self) -> bool:
        return self.status == JOB_STATUS_ACTIVE

    @property
    def has_runnable_work(self) -> bool:
        counts = self.queues.counts()
        return counts["urgent"] > 0 or counts["runnable"] > 0 or counts["running"] > 0

    def to_summary(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "target_app": self.target_app,
            "goal": self.goal,
            "priority": self.priority,
            "status": self.status,
            "queues": self.queues.counts(),
            "metadata": self.metadata,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.to_summary(), "subtasks": self.queues.to_dict()}


@dataclass(slots=True, frozen=True)
class SwitchEvent:
    from_job_id: str | None
    to_job_id: str | None
    from_subtask_id: str | None = None
    to_subtask_id: str | None = None
    reason: str | None = None
    safe_point: bool = False
    result: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_job_id": self.from_job_id,
            "to_job_id": self.to_job_id,
            "from_subtask_id": self.from_subtask_id,
            "to_subtask_id": self.to_subtask_id,
            "reason": self.reason,
            "safe_point": self.safe_point,
            "result": self.result,
            "metadata": self.metadata,
        }


@dataclass(slots=True, frozen=True)
class JobSelection:
    job: JobState | None
    subtask: SubtaskState | None
    reason: str


@dataclass(slots=True)
class JobBoard:
    jobs: dict[str, JobState] = field(default_factory=dict)
    active_job_id: str | None = None
    switch_history: list[SwitchEvent] = field(default_factory=list)

    def add_job(self, job: JobState) -> None:
        if job.job_id in self.jobs:
            raise JobBoardError(f"duplicate job_id: {job.job_id}")
        self.jobs[job.job_id] = job
        if self.active_job_id is None:
            self.active_job_id = job.job_id

    def require_job(self, job_id: str) -> JobState:
        try:
            return self.jobs[job_id]
        except KeyError as exc:
            raise JobBoardError(f"unknown job_id: {job_id}") from exc

    def add_subtask(self, subtask: SubtaskState) -> None:
        self.require_job(subtask.job_id).queues.add(subtask)

    def select_next(self, *, safe_to_interrupt: bool) -> JobSelection:
        active = self.jobs.get(self.active_job_id or "")
        if (
            active is not None
            and active.is_enabled
            and active.queues.running is not None
            and not safe_to_interrupt
        ):
            return JobSelection(
                job=active,
                subtask=active.queues.running,
                reason="continue running subtask; not safe to interrupt",
            )

        urgent = self._select_from_jobs(
            queue_name="urgent",
            safe_to_interrupt=safe_to_interrupt,
        )
        if urgent.subtask is not None:
            return urgent

        if active is not None and active.is_enabled:
            active_subtask = active.queues.pop_next(safe_to_interrupt=safe_to_interrupt)
            if active_subtask is not None:
                return JobSelection(
                    job=active,
                    subtask=active_subtask,
                    reason="continue active job",
                )

        runnable = self._select_from_jobs(
            queue_name="runnable",
            safe_to_interrupt=safe_to_interrupt,
        )
        if runnable.subtask is not None:
            return runnable

        return JobSelection(job=None, subtask=None, reason="no runnable subtask")

    def switch_to(
        self,
        job_id: str,
        *,
        subtask_id: str | None = None,
        reason: str | None = None,
        safe_point: bool = False,
        result: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SwitchEvent:
        if job_id not in self.jobs:
            raise JobBoardError(f"unknown job_id: {job_id}")
        from_job = self.jobs.get(self.active_job_id or "")
        event = SwitchEvent(
            from_job_id=self.active_job_id,
            to_job_id=job_id,
            from_subtask_id=(
                from_job.queues.running.subtask_id
                if from_job is not None and from_job.queues.running is not None
                else None
            ),
            to_subtask_id=subtask_id,
            reason=reason,
            safe_point=safe_point,
            result=result,
            metadata=dict(metadata or {}),
        )
        self.active_job_id = job_id
        self.switch_history.append(event)
        return event

    def promote_human_reply(
        self,
        *,
        job_id: str,
        request_id: str,
        priority: int = 100,
        metadata: dict[str, Any] | None = None,
    ) -> SubtaskState | None:
        return self.require_job(job_id).queues.promote_waiting_human(
            request_id,
            priority=priority,
            metadata=metadata,
        )

    def enqueue_manager_reply(
        self,
        payload: dict[str, Any],
        *,
        default_job_id: str | None = None,
    ) -> SubtaskState:
        request = dict(payload.get("request") or {})
        reply = dict(payload.get("reply") or {})
        request_id = str(request.get("request_id") or reply.get("request_id") or "")
        request_metadata = dict(request.get("metadata") or {})
        job_id = (
            str(request_metadata.get("job_id") or "")
            or default_job_id
            or self.active_job_id
        )
        if not job_id:
            raise JobBoardError("manager reply cannot be assigned without job_id")
        job = self.require_job(job_id)

        promoted = None
        if request_id:
            promoted = self.promote_human_reply(
                job_id=job_id,
                request_id=request_id,
                priority=100,
                metadata={
                    "manager_reply": reply,
                    "human_request": request,
                    "source": "human_loop",
                },
            )
        if promoted is not None:
            return promoted

        subtask = SubtaskState(
            subtask_id=(
                f"manager_reply:{request_id or len(job.queues.urgent) + 1}"
            ),
            job_id=job_id,
            goal=str(reply.get("text") or "Handle manager reply."),
            type=SUBTASK_TYPE_MANAGER_REPLY,
            status=SUBTASK_STATUS_RUNNABLE,
            priority=100,
            human_request_id=request_id or None,
            metadata={
                "manager_reply": reply,
                "human_request": request,
                "source": "human_loop",
            },
        )
        self.add_subtask(subtask)
        return subtask

    def to_global_context(self) -> dict[str, Any]:
        return {
            "active_job_id": self.active_job_id,
            "jobs": [job.to_summary() for job in self.jobs.values()],
            "recent_switches": [
                event.to_dict() for event in self.switch_history[-5:]
            ],
        }

    def _select_from_jobs(
        self,
        *,
        queue_name: str,
        safe_to_interrupt: bool,
    ) -> JobSelection:
        jobs = sorted(
            (job for job in self.jobs.values() if job.is_enabled),
            key=lambda job: job.priority,
            reverse=True,
        )
        for job in jobs:
            counts = job.queues.counts()
            if counts[queue_name] <= 0:
                continue
            subtask = job.queues.pop_next(safe_to_interrupt=safe_to_interrupt)
            if subtask is None:
                continue
            if self.active_job_id != job.job_id:
                self.switch_to(
                    job.job_id,
                    subtask_id=subtask.subtask_id,
                    reason=f"selected {queue_name} subtask",
                    safe_point=safe_to_interrupt,
                )
            return JobSelection(
                job=job,
                subtask=subtask,
                reason=f"selected {queue_name} subtask",
            )
        return JobSelection(job=None, subtask=None, reason=f"no {queue_name} subtask")


@dataclass(slots=True)
class PerJobContextStore:
    steps_by_job: dict[str, list[StepRecord]] = field(default_factory=dict)
    raw_steps_by_job: dict[str, list[StepRecord]] = field(default_factory=dict)
    summaries_by_job: dict[str, LongHistorySummary] = field(default_factory=dict)
    compaction_history_by_job: dict[str, list[dict[str, Any]]] = field(
        default_factory=dict
    )

    def append_step(self, job_id: str, step: StepRecord) -> None:
        self.steps_by_job.setdefault(job_id, []).append(step)
        self.raw_steps_by_job.setdefault(job_id, []).append(step)

    def get_steps(self, job_id: str) -> list[StepRecord]:
        return list(self.steps_by_job.get(job_id, ()))

    def get_raw_steps(self, job_id: str) -> list[StepRecord]:
        return list(self.raw_steps_by_job.get(job_id, ()))

    def set_summary(self, job_id: str, summary: LongHistorySummary) -> None:
        self.summaries_by_job[job_id] = summary

    def get_summary(self, job_id: str) -> LongHistorySummary:
        return self.summaries_by_job.get(job_id, LongHistorySummary())

    def compact_job(
        self,
        job_id: str,
        *,
        manager: ContextManager,
        plan: dict[str, Any],
        current_step_id: int | None = None,
    ) -> CompactionApplication:
        result = manager.apply_compaction_plan(
            self.get_steps(job_id),
            long_history_summary=self.get_summary(job_id),
            plan=plan,
            current_step_id=current_step_id,
        )
        self.steps_by_job[job_id] = list(result.steps)
        self.summaries_by_job[job_id] = result.long_history_summary
        self.compaction_history_by_job.setdefault(job_id, []).append(result.to_dict())
        return result

    def rebuild_prompt_context(
        self,
        job_id: str,
        *,
        manager: ContextManager,
        current_frame_ref: str | None = None,
        usage_ratio: float | None = None,
    ) -> PromptContextSelection:
        if current_frame_ref is None:
            steps = self.get_steps(job_id)
            current_frame_ref = steps[-1].after_ref if steps else None
        return manager.select_prompt_context(
            self.get_steps(job_id),
            current_frame_ref=current_frame_ref,
            long_history_summary=self.get_summary(job_id),
            usage_ratio=usage_ratio,
        )

    def to_job_context_payload(
        self,
        job_id: str,
        *,
        manager: ContextManager,
        current_frame_ref: str | None = None,
        usage_ratio: float | None = None,
    ) -> dict[str, Any]:
        selection = self.rebuild_prompt_context(
            job_id,
            manager=manager,
            current_frame_ref=current_frame_ref,
            usage_ratio=usage_ratio,
        )
        return {
            "job_id": job_id,
            "prompt_context": selection.to_context_payload(),
            "raw_step_count": len(self.raw_steps_by_job.get(job_id, ())),
            "active_step_count": len(self.steps_by_job.get(job_id, ())),
            "compaction_history": self.compaction_history_by_job.get(job_id, []),
        }
