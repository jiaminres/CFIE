"""输出处理与停止条件收敛。"""

from __future__ import annotations

from cfie.api.protocol import InferenceResult
from cfie.request.request import InferenceRequest
from cfie.runtime.executor import ModelOutput
from cfie.scheduler.schedule_output import ScheduleOutput


class OutputProcessor:
    """把执行器输出转换为外部可消费结果。"""

    def make(self, plan: ScheduleOutput,
             model_output: ModelOutput) -> list[InferenceResult]:
        # 当前轮 running 请求索引，便于把 step 输出回填到请求对象。
        req_map: dict[str, InferenceRequest] = {
            req.request_id: req
            for req in plan.running_requests
        }
        results: list[InferenceResult] = []

        for step in model_output.step_outputs:
            req = req_map.get(step.request_id)
            if req is None:
                continue

            if step.token_id is not None:
                # 维护请求级累积文本，供“非流式”场景直接读取。
                req.append_output(step.token_id, step.token_text)

            if step.finished:
                req.mark_finished(step.stop_reason)

            results.append(
                InferenceResult(
                    request_id=req.request_id,
                    token_id=step.token_id,
                    token_text=step.token_text,
                    text=req.output_text,
                    finished=req.is_terminal,
                    stop_reason=req.stop_reason,
                ))

        return results
