"""模型输入构造。"""

from __future__ import annotations

from dataclasses import dataclass, field

from cfie.request.request import InferenceRequest
from cfie.scheduler.schedule_output import ScheduleOutput


@dataclass(slots=True)
class ModelInputBatch:
    requests: list[InferenceRequest] = field(default_factory=list)


class InputBuilder:
    """Phase 1: 直接透传调度结果中的请求列表。"""

    def build(self, plan: ScheduleOutput) -> ModelInputBatch:
        return ModelInputBatch(requests=list(plan.running_requests))
