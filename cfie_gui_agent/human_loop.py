from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import uuid4

HUMAN_REQUEST_PENDING = "pending"
HUMAN_REQUEST_CLAIMED = "claimed"
HUMAN_REQUEST_RESOLVED = "resolved"
HUMAN_REQUEST_CANCELLED = "cancelled"


@dataclass(slots=True, frozen=True)
class HumanRequest:
    request_id: str
    task_id: str | None
    question: str
    blocking: bool = True
    evidence_refs: tuple[str, ...] = ()
    risk_reason: str | None = None
    proposed_action: str | None = None
    allowed_reply_format: str | None = None
    urgency: str = "normal"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        question: str,
        task_id: str | None = None,
        blocking: bool = True,
        evidence_refs: tuple[str, ...] = (),
        risk_reason: str | None = None,
        proposed_action: str | None = None,
        allowed_reply_format: str | None = None,
        urgency: str = "normal",
        metadata: dict[str, Any] | None = None,
    ) -> "HumanRequest":
        return cls(
            request_id=f"human_{uuid4().hex}",
            task_id=task_id,
            question=question,
            blocking=bool(blocking),
            evidence_refs=evidence_refs,
            risk_reason=risk_reason,
            proposed_action=proposed_action,
            allowed_reply_format=allowed_reply_format,
            urgency=urgency,
            metadata=dict(metadata or {}),
        )

    def to_task_payload(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "task_id": self.task_id,
            "question": self.question,
            "blocking": self.blocking,
            "evidence_refs": list(self.evidence_refs),
            "risk_reason": self.risk_reason,
            "proposed_action": self.proposed_action,
            "allowed_reply_format": self.allowed_reply_format,
            "urgency": self.urgency,
            "metadata": self.metadata,
        }


@dataclass(slots=True, frozen=True)
class HumanRequestState:
    request: HumanRequest
    status: str = HUMAN_REQUEST_PENDING
    claimed_by: str | None = None
    reply: "HumanReply | None" = None
    result_task: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.to_task_payload(),
            "status": self.status,
            "claimed_by": self.claimed_by,
            "reply": (
                {
                    "request_id": self.reply.request_id,
                    "text": self.reply.text,
                    "metadata": self.reply.metadata,
                }
                if self.reply is not None
                else None
            ),
            "result_task": self.result_task,
        }


@dataclass(slots=True, frozen=True)
class HumanReply:
    request_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class HumanChannel(Protocol):
    def send_request(self, request: HumanRequest) -> None:
        ...

    def poll_replies(self) -> tuple[HumanReply, ...]:
        ...

    def acknowledge(self, reply: HumanReply) -> None:
        ...


@dataclass(slots=True)
class InMemoryHumanChannel:
    sent_requests: list[HumanRequest] = field(default_factory=list)
    replies: deque[HumanReply] = field(default_factory=deque)
    acknowledged: list[HumanReply] = field(default_factory=list)

    def send_request(self, request: HumanRequest) -> None:
        self.sent_requests.append(request)

    def poll_replies(self) -> tuple[HumanReply, ...]:
        items: list[HumanReply] = []
        while self.replies:
            items.append(self.replies.popleft())
        return tuple(items)

    def acknowledge(self, reply: HumanReply) -> None:
        self.acknowledged.append(reply)

    def push_reply(self, reply: HumanReply) -> None:
        self.replies.append(reply)


@dataclass(slots=True)
class HumanLoopManager:
    channel: HumanChannel
    pending: dict[str, HumanRequest] = field(default_factory=dict)
    states: dict[str, HumanRequestState] = field(default_factory=dict)
    completed: dict[str, HumanRequestState] = field(default_factory=dict)
    urgent_queue: deque[dict[str, Any]] = field(default_factory=deque)

    def request_help(
        self,
        *,
        question: str,
        task_id: str | None = None,
        blocking: bool = True,
        evidence_refs: tuple[str, ...] = (),
        risk_reason: str | None = None,
        proposed_action: str | None = None,
        allowed_reply_format: str | None = None,
        urgency: str = "normal",
        metadata: dict[str, Any] | None = None,
    ) -> HumanRequest:
        request = HumanRequest.create(
            question=question,
            task_id=task_id,
            blocking=blocking,
            evidence_refs=evidence_refs,
            risk_reason=risk_reason,
            proposed_action=proposed_action,
            allowed_reply_format=allowed_reply_format,
            urgency=urgency,
            metadata=metadata,
        )
        self.pending[request.request_id] = request
        self.states[request.request_id] = HumanRequestState(request=request)
        self.channel.send_request(request)
        return request

    def list_requests(
        self,
        *,
        include_completed: bool = False,
    ) -> tuple[dict[str, Any], ...]:
        items = [state.to_dict() for state in self.states.values()]
        if include_completed:
            items.extend(state.to_dict() for state in self.completed.values())
        return tuple(items)

    def claim_request(self, request_id: str, *, source: str) -> HumanRequestState:
        state = self._require_active_state(request_id)
        if state.status == HUMAN_REQUEST_CLAIMED and state.claimed_by != source:
            raise ValueError(
                f"human request {request_id} already claimed by {state.claimed_by}"
            )
        claimed = HumanRequestState(
            request=state.request,
            status=HUMAN_REQUEST_CLAIMED,
            claimed_by=source,
            reply=state.reply,
            result_task=state.result_task,
        )
        self.states[request_id] = claimed
        return claimed

    def release_request(self, request_id: str, *, source: str) -> HumanRequestState:
        state = self._require_active_state(request_id)
        if state.status != HUMAN_REQUEST_CLAIMED:
            return state
        if state.claimed_by != source:
            raise ValueError(
                f"human request {request_id} is claimed by {state.claimed_by}"
            )
        released = HumanRequestState(request=state.request)
        self.states[request_id] = released
        return released

    def submit_reply(
        self,
        *,
        request_id: str,
        text: str,
        source: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = self._require_active_state(request_id)
        if state.status == HUMAN_REQUEST_CLAIMED and state.claimed_by != source:
            raise ValueError(
                f"human request {request_id} already claimed by {state.claimed_by}"
            )
        reply = HumanReply(
            request_id=request_id,
            text=text,
            metadata={"source": source, **dict(metadata or {})},
        )
        return self._resolve_reply(state.request, reply, source=source)

    def poll(self) -> tuple[dict[str, Any], ...]:
        tasks: list[dict[str, Any]] = []
        for reply in self.channel.poll_replies():
            state = self.states.get(reply.request_id)
            if state is None:
                self.channel.acknowledge(reply)
                continue
            if (
                state.status == HUMAN_REQUEST_CLAIMED
                and state.claimed_by != "channel"
            ):
                self.channel.acknowledge(reply)
                continue
            task = self._resolve_reply(state.request, reply, source="channel")
            tasks.append(task)
            self.channel.acknowledge(reply)
        return tuple(tasks)

    def pop_urgent_task(self) -> dict[str, Any] | None:
        if not self.urgent_queue:
            return None
        return self.urgent_queue.popleft()

    def _reply_to_urgent_task(
        self,
        request: HumanRequest,
        reply: HumanReply,
    ) -> dict[str, Any]:
        return {
            "type": "manager_reply",
            "priority": "urgent",
            "blocking": request.blocking,
            "request": request.to_task_payload(),
            "reply": {
                "request_id": reply.request_id,
                "text": reply.text,
                "metadata": reply.metadata,
            },
        }

    def _resolve_reply(
        self,
        request: HumanRequest,
        reply: HumanReply,
        *,
        source: str,
    ) -> dict[str, Any]:
        self.pending.pop(request.request_id, None)
        task = self._reply_to_urgent_task(request, reply)
        resolved = HumanRequestState(
            request=request,
            status=HUMAN_REQUEST_RESOLVED,
            claimed_by=source,
            reply=reply,
            result_task=task,
        )
        self.states.pop(request.request_id, None)
        self.completed[request.request_id] = resolved
        self.urgent_queue.append(task)
        return task

    def _require_active_state(self, request_id: str) -> HumanRequestState:
        state = self.states.get(request_id)
        if state is None:
            if request_id in self.completed:
                raise ValueError(f"human request {request_id} is already resolved")
            raise KeyError(f"unknown human request: {request_id}")
        return state
