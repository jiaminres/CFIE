"""Phase 1 引擎链路测试（mock 执行器）。"""

from __future__ import annotations

from collections import defaultdict

from cfie.config.schema import EngineConfig
from cfie.request.request import InferenceRequest, RequestState
from cfie.runtime.engine import Engine
from cfie.runtime.executor import ModelOutput
from cfie.runtime.model_runner import ModelStepOutput


class FakeExecutor:
    """按预置脚本返回 token，模拟 decode。"""

    def __init__(self) -> None:
        self._aborted: set[str] = set()
        self._steps = defaultdict(int)
        self._script = {
            0: (11, "H", False, None),
            1: (12, "i", True, "length"),
        }

    def execute_model(self, model_inputs) -> ModelOutput:
        outputs: list[ModelStepOutput] = []
        for req in model_inputs.requests:
            if req.request_id in self._aborted:
                continue
            idx = self._steps[req.request_id]
            token_id, token_text, finished, reason = self._script[min(idx, 1)]
            self._steps[req.request_id] += 1
            outputs.append(
                ModelStepOutput(
                    request_id=req.request_id,
                    token_id=token_id,
                    token_text=token_text,
                    finished=finished,
                    stop_reason=reason,
                ))
        return ModelOutput(step_outputs=outputs)

    def abort_request(self, request_id: str) -> None:
        self._aborted.add(request_id)


def test_engine_phase1_e2e_streaming_with_fake_executor() -> None:
    cfg = EngineConfig.from_flat_kwargs(model="./model")
    engine = Engine(cfg, executor=FakeExecutor())

    req = InferenceRequest(prompt="hello", max_new_tokens=2, request_id="r1")
    engine.add_request(req)

    all_results = []
    while not req.is_terminal:
        all_results.extend(engine.step())

    assert req.output_text == "Hi"
    assert req.state == RequestState.FINISHED
    assert len(all_results) == 2
    drained = engine.drain_stream("r1")
    assert [r.token_text for r in drained] == ["H", "i"]


def test_engine_abort_marks_request_and_stops_outputs() -> None:
    cfg = EngineConfig.from_flat_kwargs(model="./model")
    engine = Engine(cfg, executor=FakeExecutor())

    req = InferenceRequest(prompt="abort", max_new_tokens=4, request_id="r_abort")
    engine.add_request(req)
    engine.abort(req.request_id)

    assert req.state == RequestState.ABORTED
    assert engine.step() == []
