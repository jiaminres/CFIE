"""调度器实现（Phase 1: FCFS + 单序列优先）。"""

from __future__ import annotations

from cfie.request.queue import RequestQueue
from cfie.request.request import InferenceRequest, RequestState
from cfie.runtime.executor import ModelOutput
from cfie.scheduler.schedule_output import ScheduleOutput
from cfie.utils.logging import get_logger

logger = get_logger(__name__)


class FCFSScheduler:
    """最小 FCFS 调度器。"""

    def __init__(self, max_num_seqs: int = 1) -> None:
        if max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be > 0")
        self.max_num_seqs = max_num_seqs
        self._waiting = RequestQueue()
        self._running: dict[str, InferenceRequest] = {}

    def add_request(self, req: InferenceRequest) -> None:
        if req.is_terminal:
            return
        req.state = RequestState.WAITING
        self._waiting.push(req)
        logger.debug("scheduler queued request: request_id=%s", req.request_id)

    def schedule(self) -> ScheduleOutput:
        # 先把 waiting 队列按 FCFS 填充到 running，直到达到并发上限。
        while len(self._running) < self.max_num_seqs and not self._waiting.empty():
            req = self._waiting.pop()
            if req.is_terminal:
                continue
            req.set_running()
            self._running[req.request_id] = req

        return ScheduleOutput(running_requests=list(self._running.values()))

    def update_from_output(self, plan: ScheduleOutput, out: ModelOutput) -> None:
        del plan
        # 调度层只关心“是否完成”，具体 token 处理由 output processor 负责。
        for step in out.step_outputs:
            req = self._running.get(step.request_id)
            if req is None:
                continue
            if step.finished:
                req.mark_finished(step.stop_reason)

    def finish_requests(self, request_ids: list[str]) -> None:
        # 统一回收 running/waiting 两侧，确保中断与完成路径都幂等。
        for request_id in request_ids:
            self._running.pop(request_id, None)
            self._waiting.remove(request_id)
        if request_ids:
            logger.debug("scheduler recycled requests: ids=%s",
                         ",".join(request_ids))

    def get_running_request(self, request_id: str) -> InferenceRequest | None:
        return self._running.get(request_id)
