"""调度输出结构。"""

from __future__ import annotations

from dataclasses import dataclass, field

from cfie.request.request import InferenceRequest


@dataclass(slots=True)
class ScheduleOutput:
    """单轮调度结果。"""

    running_requests: list[InferenceRequest] = field(default_factory=list)
    prefetch_block_ids: list[int] = field(default_factory=list)
    restore_block_ids: list[int] = field(default_factory=list)

    def empty(self) -> bool:
        return not self.running_requests
