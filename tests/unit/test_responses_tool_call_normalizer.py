from __future__ import annotations

import json

from openai.types.responses import ResponseOutputMessage, ResponseOutputText

from cfie.entrypoints.openai.responses.tool_call_normalizer import (
    normalize_text_tool_calls_in_response_outputs,
)


def test_responses_output_normalizer_splits_qwen_xml_tool_call():
    output = [
        ResponseOutputMessage(
            id="msg_1",
            role="assistant",
            status="completed",
            type="message",
            content=[
                ResponseOutputText(
                    type="output_text",
                    text=(
                        "准备点击发送按钮。\n"
                        "<tool_call>"
                        "<function=computer_use>"
                        "<parameter=coordinate_space>qwen_normalized_1000</parameter>"
                        "<parameter=actions>"
                        '[{"type":"click","x":570,"y":470}]'
                        "</parameter>"
                        "</function>"
                        "</tool_call>"
                    ),
                    annotations=[],
                    logprobs=None,
                )
            ],
        )
    ]

    normalized = normalize_text_tool_calls_in_response_outputs(output)

    assert len(normalized) == 2
    message = normalized[0]
    assert message.type == "message"
    assert message.content[0].text == "准备点击发送按钮。"
    call = normalized[1]
    assert call.type == "function_call"
    assert call.name == "computer_use"
    assert json.loads(call.arguments) == {
        "coordinate_space": "qwen_normalized_1000",
        "actions": [{"type": "click", "x": 570, "y": 470}],
    }


def test_responses_output_normalizer_removes_tool_only_message():
    output = [
        ResponseOutputMessage(
            id="msg_1",
            role="assistant",
            status="completed",
            type="message",
            content=[
                ResponseOutputText(
                    type="output_text",
                    text=(
                        "<tool_call>"
                        "<function=computer_use>"
                        "<parameter=actions>"
                        '[{"type":"click","x":570,470}]'
                        "</parameter>"
                        "</function>"
                        "</tool_call>"
                    ),
                    annotations=[],
                    logprobs=None,
                )
            ],
        )
    ]

    normalized = normalize_text_tool_calls_in_response_outputs(output)

    assert len(normalized) == 1
    call = normalized[0]
    assert call.type == "function_call"
    assert json.loads(call.arguments) == {
        "actions": [{"type": "click", "x": 570, "y": 470}],
    }
