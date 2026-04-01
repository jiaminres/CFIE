"""请求对象与状态机定义。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time
import uuid


class RequestState(str, Enum):
    """请求生命周期状态。"""

    WAITING = "waiting"
    RUNNING = "running"
    SWAPPED = "swapped"
    FINISHED = "finished"
    ABORTED = "aborted"


@dataclass(slots=True)
class InferenceRequest:
    """最小推理请求对象（Phase 1）。"""

    prompt: str
    max_new_tokens: int
    session_id: str = "default"
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prompt_token_ids: list[int] = field(default_factory=list)
    output_token_ids: list[int] = field(default_factory=list)
    output_text: str = ""
    priority: int = 0
    arrival_ts: float = field(default_factory=time.time)
    stop_reason: str | None = None
    state: RequestState = RequestState.WAITING

    def __post_init__(self) -> None:
        if not isinstance(self.max_new_tokens, int) or self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        if not isinstance(self.prompt, str):
            raise ValueError("prompt must be a string")

    def set_running(self) -> None:
        self.state = RequestState.RUNNING

    def append_output(self, token_id: int, token_text: str) -> None:
        self.output_token_ids.append(token_id)
        self.output_text += token_text

    def mark_finished(self, reason: str | None = None) -> None:
        self.state = RequestState.FINISHED
        self.stop_reason = reason or self.stop_reason

    def mark_aborted(self, reason: str = "aborted") -> None:
        self.state = RequestState.ABORTED
        self.stop_reason = reason

    @property
    def is_terminal(self) -> bool:
        return self.state in (RequestState.FINISHED, RequestState.ABORTED)
