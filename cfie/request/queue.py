"""请求队列封装。"""

from __future__ import annotations

from collections import deque
from typing import Iterable

from cfie.request.request import InferenceRequest


class RequestQueue:
    """简单 FIFO 队列。"""

    def __init__(self) -> None:
        self._queue: deque[InferenceRequest] = deque()

    def push(self, request: InferenceRequest) -> None:
        self._queue.append(request)

    def extend(self, requests: Iterable[InferenceRequest]) -> None:
        for req in requests:
            self.push(req)

    def pop(self) -> InferenceRequest:
        return self._queue.popleft()

    def remove(self, request_id: str) -> None:
        self._queue = deque(req for req in self._queue
                            if req.request_id != request_id)

    def __len__(self) -> int:
        return len(self._queue)

    def empty(self) -> bool:
        return not self._queue
