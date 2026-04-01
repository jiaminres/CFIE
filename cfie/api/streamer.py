"""流式输出缓存器。"""

from __future__ import annotations

from collections import defaultdict, deque

from cfie.api.protocol import InferenceResult


class TokenStreamer:
    """按 request_id 缓存增量 token 结果。"""

    def __init__(self) -> None:
        self._buffers: dict[str, deque[InferenceResult]] = defaultdict(deque)

    def emit(self, result: InferenceResult) -> None:
        self._buffers[result.request_id].append(result)

    def emit_many(self, results: list[InferenceResult]) -> None:
        for result in results:
            self.emit(result)

    def drain(self, request_id: str) -> list[InferenceResult]:
        buffer = self._buffers.get(request_id)
        if not buffer:
            return []

        out: list[InferenceResult] = []
        while buffer:
            out.append(buffer.popleft())
        return out
