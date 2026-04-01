"""会话管理（Phase 1 最小实现）。"""

from __future__ import annotations

from dataclasses import dataclass, field
import time


@dataclass(slots=True)
class Session:
    session_id: str
    created_ts: float = field(default_factory=time.time)
    request_ids: list[str] = field(default_factory=list)


class SessionManager:
    """最小会话生命周期管理器。"""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def get_or_create(self, session_id: str) -> Session:
        session = self._sessions.get(session_id)
        if session is None:
            session = Session(session_id=session_id)
            self._sessions[session_id] = session
        return session

    def attach_request(self, session_id: str, request_id: str) -> None:
        session = self.get_or_create(session_id)
        session.request_ids.append(request_id)
