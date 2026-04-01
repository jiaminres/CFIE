"""API 协议对象。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class GenerateRequest:
    prompt: str
    max_new_tokens: int = 64
    session_id: str = "default"


@dataclass(slots=True)
class InferenceResult:
    request_id: str
    token_id: int | None
    token_text: str
    text: str
    finished: bool
    stop_reason: str | None = None
