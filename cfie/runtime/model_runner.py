"""模型前向执行（Phase 1: 单请求、贪心解码）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cfie.request.request import InferenceRequest


@dataclass(slots=True)
class _RequestRuntimeState:
    past_key_values: Any
    last_token_id: int
    generated_tokens: int


@dataclass(slots=True)
class ModelStepOutput:
    request_id: str
    token_id: int | None
    token_text: str
    finished: bool
    stop_reason: str | None = None


class ModelRunner:
    """封装 tokenize/prefill/decode 单步执行。"""

    def __init__(self, model: Any, tokenizer: Any, device: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._states: dict[str, _RequestRuntimeState] = {}
        self._eos_token_id = getattr(tokenizer, "eos_token_id", None)

    def tokenize(self, text: str) -> list[int]:
        token_ids = self.tokenizer.encode(text, add_special_tokens=True)
        if not token_ids:
            raise ValueError("prompt must not be empty after tokenization")
        return [int(tid) for tid in token_ids]

    def abort(self, request_id: str) -> None:
        self._states.pop(request_id, None)

    def forward_step(self, request: InferenceRequest) -> ModelStepOutput:
        import torch

        state = self._states.get(request.request_id)
        if state is None:
            # 首轮走 prefill：把整段 prompt 一次送入模型并建立 KV cache。
            if not request.prompt_token_ids:
                request.prompt_token_ids = self.tokenize(request.prompt)
            input_ids = torch.tensor([request.prompt_token_ids], device=self.device)
            model_kwargs = {"input_ids": input_ids, "use_cache": True}
            generated_tokens = 0
        else:
            # 后续轮次走 decode：仅输入上一个 token，复用 past_key_values。
            input_ids = torch.tensor([[state.last_token_id]], device=self.device)
            model_kwargs = {
                "input_ids": input_ids,
                "past_key_values": state.past_key_values,
                "use_cache": True,
            }
            generated_tokens = state.generated_tokens

        with torch.no_grad():
            outputs = self.model(**model_kwargs)

        # Phase 1 使用贪心解码，保证路径简单可验证。
        logits = outputs.logits[:, -1, :]
        next_token_id = int(torch.argmax(logits, dim=-1).item())
        token_text = self.tokenizer.decode([next_token_id],
                                           skip_special_tokens=False)
        generated_tokens += 1

        finished = False
        stop_reason: str | None = None
        # 停止条件：遇到 eos 或达到 max_new_tokens。
        if self._eos_token_id is not None and next_token_id == int(
                self._eos_token_id):
            finished = True
            stop_reason = "eos_token"
        elif generated_tokens >= request.max_new_tokens:
            finished = True
            stop_reason = "length"

        if finished:
            # 终止后及时释放该请求的运行时缓存。
            self._states.pop(request.request_id, None)
        else:
            # 保存下一步 decode 所需状态。
            self._states[request.request_id] = _RequestRuntimeState(
                past_key_values=outputs.past_key_values,
                last_token_id=next_token_id,
                generated_tokens=generated_tokens,
            )

        return ModelStepOutput(
            request_id=request.request_id,
            token_id=next_token_id,
            token_text=token_text,
            finished=finished,
            stop_reason=stop_reason,
        )
