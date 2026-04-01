"""本地 Python SDK 入口。"""

from __future__ import annotations

from collections.abc import Iterator

from cfie.api.protocol import InferenceResult
from cfie.request.request import InferenceRequest
from cfie.runtime.engine import Engine


class LocalServer:
    """基于 Engine 的最小本地推理接口。"""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def stream_generate(self,
                        prompt: str,
                        *,
                        max_new_tokens: int = 64,
                        session_id: str = "default") -> Iterator[InferenceResult]:
        request = InferenceRequest(prompt=prompt,
                                   max_new_tokens=max_new_tokens,
                                   session_id=session_id)
        self.engine.add_request(request)

        while not request.is_terminal:
            results = self.engine.step()
            for result in results:
                if result.request_id == request.request_id:
                    yield result

    def generate(self,
                 prompt: str,
                 *,
                 max_new_tokens: int = 64,
                 session_id: str = "default") -> str:
        text = ""
        for result in self.stream_generate(prompt,
                                           max_new_tokens=max_new_tokens,
                                           session_id=session_id):
            text = result.text
        return text
