"""Runtime engine skeleton for Phase 0."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cfie.config.schema import EngineConfig


@dataclass(slots=True)
class InferenceResult:
    request_id: str
    output_token_ids: list[int] = field(default_factory=list)


class Engine:
    """Minimal engine with a no-op step loop."""

    def __init__(self, config: EngineConfig) -> None:
        # 初始化时前置校验，避免运行路径中反复做相同检查。
        self.config = config.validate()
        self._running = False
        self._requests: dict[str, Any] = {}
        self._step_count = 0

    @property
    def running(self) -> bool:
        return self._running

    @property
    def step_count(self) -> int:
        return self._step_count

    def add_request(self, req: Any) -> None:
        # Phase 0 先接受通用请求对象，后续阶段再收敛到强类型请求。
        request_id = str(getattr(req, "request_id", len(self._requests)))
        self._requests[request_id] = req

    def abort(self, request_id: str) -> None:
        self._requests.pop(request_id, None)

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def step(self) -> list[InferenceResult]:
        # 自动启动可简化测试和 CLI 场景下的调用方式。
        if not self._running:
            self.start()
        self._step_count += 1
        return []

    def run(self, steps: int = 1) -> None:
        self.start()
        for _ in range(max(0, steps)):
            self.step()
        self.stop()
