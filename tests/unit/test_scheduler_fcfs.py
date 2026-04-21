"""FCFS 调度器测试。"""

from __future__ import annotations

from cfie.request.request import InferenceRequest, RequestState
from cfie.runtime.executor import ModelOutput
from cfie.runtime.model_runner import ModelStepOutput
from cfie.scheduler.scheduler import FCFSScheduler


def test_fcfs_scheduler_respects_arrival_order() -> None:
    scheduler = FCFSScheduler(max_num_seqs=1)
    req1 = InferenceRequest(prompt="a", max_new_tokens=1, request_id="r1")
    req2 = InferenceRequest(prompt="b", max_new_tokens=1, request_id="r2")

    scheduler.add_request(req1)
    scheduler.add_request(req2)

    plan1 = scheduler.schedule()
    assert [r.request_id for r in plan1.running_requests] == ["r1"]
    assert req1.state == RequestState.RUNNING

    scheduler.update_from_output(
        plan1,
        ModelOutput(step_outputs=[
            ModelStepOutput(request_id="r1",
                            token_id=1,
                            token_text="x",
                            finished=True,
                            stop_reason="length")
        ]),
    )
    scheduler.finish_requests(["r1"])

    plan2 = scheduler.schedule()
    assert [r.request_id for r in plan2.running_requests] == ["r2"]
