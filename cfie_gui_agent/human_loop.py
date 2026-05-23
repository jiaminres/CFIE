from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import uuid4


@dataclass(slots=True, frozen=True)
class HumanRequest:
    request_id: str
    task_id: str | None
    question: str
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
            "evidence_refs": list(self.evidence_refs),
            "risk_reason": self.risk_reason,
            "proposed_action": self.proposed_action,
            "allowed_reply_format": self.allowed_reply_format,
            "urgency": self.urgency,
            "metadata": self.metadata,
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
    urgent_queue: deque[dict[str, Any]] = field(default_factory=deque)

    def request_help(
        self,
        *,
        question: str,
        task_id: str | None = None,
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
            evidence_refs=evidence_refs,
            risk_reason=risk_reason,
            proposed_action=proposed_action,
            allowed_reply_format=allowed_reply_format,
            urgency=urgency,
            metadata=metadata,
        )
        self.pending[request.request_id] = request
        self.channel.send_request(request)
        return request

    def poll(self) -> tuple[dict[str, Any], ...]:
        tasks: list[dict[str, Any]] = []
        for reply in self.channel.poll_replies():
            request = self.pending.pop(reply.request_id, None)
            if request is None:
                self.channel.acknowledge(reply)
                continue
            task = self._reply_to_urgent_task(request, reply)
            self.urgent_queue.append(task)
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
            "request": request.to_task_payload(),
            "reply": {
                "request_id": reply.request_id,
                "text": reply.text,
                "metadata": reply.metadata,
            },
        }
