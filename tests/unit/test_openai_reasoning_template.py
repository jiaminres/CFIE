# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from cfie.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from cfie.entrypoints.openai.reasoning_template import (
    QWEN_CLICK_LOCAL_REFINEMENT_PREAMBLE,
    QWEN_REASONING_PREAMBLE_KWARG,
)
from cfie.entrypoints.openai.responses.protocol import ResponsesRequest
from cfie.entrypoints.openai.responses.serving import OpenAIServingResponses
from cfie.reasoning.qwen3_reasoning_parser import Qwen3ReasoningParser
from cfie.renderers.hf import _apply_cfie_reasoning_preamble


def test_responses_reasoning_none_disables_qwen_thinking() -> None:
    request = ResponsesRequest(input="hello", reasoning={"effort": "none"})

    params = request.build_chat_params(None, "auto")

    assert params.chat_template_kwargs["reasoning_effort"] == "none"
    assert params.chat_template_kwargs["enable_thinking"] is False
    assert QWEN_REASONING_PREAMBLE_KWARG not in params.chat_template_kwargs


def test_responses_reasoning_origin_enables_qwen_thinking_without_preamble() -> None:
    request = ResponsesRequest(input="hello", reasoning={"effort": "origin"})

    params = request.build_chat_params(None, "auto")

    assert params.chat_template_kwargs["reasoning_effort"] == "origin"
    assert params.chat_template_kwargs["enable_thinking"] is True
    assert QWEN_REASONING_PREAMBLE_KWARG not in params.chat_template_kwargs


def test_responses_reasoning_efforts_build_distinct_qwen_preambles() -> None:
    prompts = {}
    for effort in ("minimal", "low", "medium", "high", "xhigh"):
        request = ResponsesRequest(input="hello", reasoning={"effort": effort})
        params = request.build_chat_params(None, "auto")

        assert params.chat_template_kwargs["reasoning_effort"] == effort
        assert params.chat_template_kwargs["enable_thinking"] is True
        prompts[effort] = params.chat_template_kwargs[QWEN_REASONING_PREAMBLE_KWARG]

    assert len(set(prompts.values())) == len(prompts)
    assert "minimal" in prompts["minimal"]
    assert "xhigh" in prompts["xhigh"]


def test_responses_reasoning_preserves_request_specific_preamble() -> None:
    request = ResponsesRequest(
        input="hello",
        reasoning={"effort": "low"},
        chat_template_kwargs={
            QWEN_REASONING_PREAMBLE_KWARG: QWEN_CLICK_LOCAL_REFINEMENT_PREAMBLE,
        },
    )

    params = request.build_chat_params(None, "auto")

    assert (
        params.chat_template_kwargs[QWEN_REASONING_PREAMBLE_KWARG]
        == QWEN_CLICK_LOCAL_REFINEMENT_PREAMBLE
    )
    assert "局部图重新选点" in QWEN_CLICK_LOCAL_REFINEMENT_PREAMBLE
    assert "思考最多两行" in QWEN_CLICK_LOCAL_REFINEMENT_PREAMBLE
    assert "recommended_click_1000" not in QWEN_CLICK_LOCAL_REFINEMENT_PREAMBLE
    assert "1000 归一化坐标 x=" in QWEN_CLICK_LOCAL_REFINEMENT_PREAMBLE


def test_chat_completion_reasoning_preserves_request_specific_preamble() -> None:
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hello"}],
        reasoning_effort="medium",
        chat_template_kwargs={
            QWEN_REASONING_PREAMBLE_KWARG: QWEN_CLICK_LOCAL_REFINEMENT_PREAMBLE,
        },
    )

    params = request.build_chat_params(None, "auto")

    assert (
        params.chat_template_kwargs[QWEN_REASONING_PREAMBLE_KWARG]
        == QWEN_CLICK_LOCAL_REFINEMENT_PREAMBLE
    )


def test_responses_reasoning_effort_overrides_default_thinking_disabled() -> None:
    request = ResponsesRequest(
        input="hello",
        chat_template_kwargs={"enable_thinking": False},
        reasoning={"effort": "low"},
    )

    params = request.build_chat_params(None, "auto")

    assert params.chat_template_kwargs["enable_thinking"] is True
    assert params.chat_template_kwargs["reasoning_effort"] == "low"


def test_chat_completion_reasoning_effort_uses_same_template_mapping() -> None:
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hello"}],
        reasoning_effort="low",
    )

    params = request.build_chat_params(None, "auto")

    assert params.chat_template_kwargs["reasoning_effort"] == "low"
    assert params.chat_template_kwargs["enable_thinking"] is True
    assert "当前状态" in params.chat_template_kwargs[
        QWEN_REASONING_PREAMBLE_KWARG
    ]


def test_chat_completion_reasoning_origin_has_no_preamble() -> None:
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hello"}],
        reasoning_effort="origin",
    )

    params = request.build_chat_params(None, "auto")

    assert params.chat_template_kwargs["reasoning_effort"] == "origin"
    assert params.chat_template_kwargs["enable_thinking"] is True
    assert QWEN_REASONING_PREAMBLE_KWARG not in params.chat_template_kwargs


def test_qwen_preamble_is_inserted_inside_generation_think_block() -> None:
    prompt = "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n"

    rendered = _apply_cfie_reasoning_preamble(prompt, "当前思考模式：low。\n当前状态：")

    assert rendered.endswith("<think>\n当前思考模式：low。\n当前状态：\n")


def test_qwen_preamble_does_not_modify_disabled_thinking_prompt() -> None:
    prompt = (
        "<|im_start|>user\nhello<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )

    rendered = _apply_cfie_reasoning_preamble(prompt, "当前思考模式：low。")

    assert rendered == prompt


class _FakeTokenizer:
    def get_vocab(self):
        return {"<think>": 1, "</think>": 2}


def test_qwen_parser_treats_response_effort_none_as_content_without_end_tag() -> None:
    parser = Qwen3ReasoningParser(_FakeTokenizer())
    request = ResponsesRequest(input="hello", reasoning={"effort": "none"})

    reasoning, content = parser.extract_reasoning("final answer", request)

    assert reasoning is None
    assert content == "final answer"


def test_qwen_parser_honors_response_template_kwargs_when_no_effort() -> None:
    parser = Qwen3ReasoningParser(_FakeTokenizer())
    request = ResponsesRequest(
        input="hello",
        chat_template_kwargs={"enable_thinking": False},
    )

    reasoning, content = parser.extract_reasoning("final answer", request)

    assert reasoning is None
    assert content == "final answer"


def test_qwen_parser_effort_overrides_template_kwargs_for_enabled_thinking() -> None:
    parser = Qwen3ReasoningParser(_FakeTokenizer())
    request = ResponsesRequest(
        input="hello",
        chat_template_kwargs={"enable_thinking": False},
        reasoning={"effort": "low"},
    )

    reasoning, content = parser.extract_reasoning("unfinished reasoning", request)

    assert reasoning == "unfinished reasoning"
    assert content is None


def test_responses_build_chat_params_persists_reasoning_preamble_for_parser() -> None:
    request = ResponsesRequest(input="hello", reasoning={"effort": "low"})

    params = request.build_chat_params(None, "auto")

    assert request.chat_template_kwargs == params.chat_template_kwargs
    assert QWEN_REASONING_PREAMBLE_KWARG in request.chat_template_kwargs


def test_qwen_parser_includes_preamble_with_empty_model_reasoning() -> None:
    parser = Qwen3ReasoningParser(_FakeTokenizer())
    request = ResponsesRequest(input="hello", reasoning={"effort": "low"})
    request.build_chat_params(None, "auto")

    reasoning, content = parser.extract_reasoning("</think>final answer", request)

    assert "当前思考模式：low" in reasoning
    assert "final answer" == content


def test_qwen_parser_prefixes_preamble_to_generated_reasoning() -> None:
    parser = Qwen3ReasoningParser(_FakeTokenizer())
    request = ResponsesRequest(input="hello", reasoning={"effort": "medium"})
    request.build_chat_params(None, "auto")

    reasoning, content = parser.extract_reasoning(
        "checked state</think>call tool", request
    )

    assert reasoning.startswith("当前思考模式：medium")
    assert reasoning.endswith("checked state")
    assert content == "call tool"


def test_responses_reasoning_enabled_requires_parser() -> None:
    request = ResponsesRequest(input="hello", reasoning={"effort": "low"})
    serving = object.__new__(OpenAIServingResponses)
    serving.use_harmony = False
    serving.enable_store = False
    serving.parser = None
    serving.default_chat_template_kwargs = {"enable_thinking": False}

    error = OpenAIServingResponses._validate_create_responses_input(serving, request)

    assert error is not None
    assert error.error.param == "reasoning"
    assert "--reasoning-parser qwen3" in error.error.message


def test_responses_reasoning_disabled_does_not_require_parser() -> None:
    request = ResponsesRequest(input="hello", reasoning={"effort": "none"})
    serving = object.__new__(OpenAIServingResponses)
    serving.use_harmony = False
    serving.enable_store = False
    serving.parser = None
    serving.default_chat_template_kwargs = {"enable_thinking": False}

    error = OpenAIServingResponses._validate_create_responses_input(serving, request)

    assert error is None
