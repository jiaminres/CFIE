"""执行器抽象（Phase 1: 单卡同步执行）。"""

from __future__ import annotations

from dataclasses import dataclass, field

from cfie.runtime.input_builder import ModelInputBatch
from cfie.runtime.model_runner import ModelRunner, ModelStepOutput


@dataclass(slots=True)
class ModelOutput:
    step_outputs: list[ModelStepOutput] = field(default_factory=list)


class Executor:
    """最小执行器。"""

    def __init__(self, model_runner: ModelRunner) -> None:
        self.model_runner = model_runner

    def execute_model(self, model_inputs: ModelInputBatch) -> ModelOutput:
        outputs: list[ModelStepOutput] = []
        for req in model_inputs.requests:
            if req.is_terminal:
                continue
            outputs.append(self.model_runner.forward_step(req))
        return ModelOutput(step_outputs=outputs)

    def abort_request(self, request_id: str) -> None:
        self.model_runner.abort(request_id)
