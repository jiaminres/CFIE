"""请求状态机测试。"""

from __future__ import annotations

import pytest

from cfie.request.request import InferenceRequest, RequestState


def test_request_append_and_finish() -> None:
    req = InferenceRequest(prompt="hello", max_new_tokens=2)
    req.set_running()
    req.append_output(1, "H")
    req.mark_finished("length")

    assert req.state == RequestState.FINISHED
    assert req.output_token_ids == [1]
    assert req.output_text == "H"
    assert req.stop_reason == "length"


def test_request_requires_positive_max_new_tokens() -> None:
    with pytest.raises(ValueError, match="max_new_tokens"):
        InferenceRequest(prompt="hello", max_new_tokens=0)
