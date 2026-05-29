from __future__ import annotations

import json

from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)

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


def test_responses_output_normalizer_splits_bare_qwen_function_block():
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
                        "<function=computer_use>"
                        "<parameter=coordinate_space>qwen_normalized_1000</parameter>"
                        "<parameter=actions>"
                        '[{"type":"click","x":500,"y":900}]'
                        "</parameter>"
                        "</function>"
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
    assert call.name == "computer_use"
    assert json.loads(call.arguments) == {
        "coordinate_space": "qwen_normalized_1000",
        "actions": [{"type": "click", "x": 500, "y": 900}],
    }


def test_responses_output_normalizer_decodes_quoted_parameter_values():
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
                        "<parameter=coordinate_space>"
                        '"qwen_normalized_1000"'
                        "</parameter>"
                        "<parameter=actions>"
                        '[{"type":"click","x":500,"y":900}]'
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
    assert json.loads(call.arguments)["coordinate_space"] == "qwen_normalized_1000"


def test_responses_output_normalizer_splits_nested_function_from_text_computer_use():
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
                        "<parameter=coordinate_space>qwen_normalized_1000</parameter>"
                        "<parameter=actions>"
                        '[{"type":"call_function","function":'
                        '{"name":"finish_subtask","parameters":'
                        '{"completion_reason":"done"}}}]'
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
    assert call.name == "finish_subtask"
    assert json.loads(call.arguments) == {"completion_reason": "done"}


def test_responses_output_normalizer_splits_qwen_tool_code():
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
                        "<tool_code>\n"
                        "print(read_text_file(path='items.jsonl', max_chars=128))\n"
                        "</tool_code>"
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
    assert call.name == "read_text_file"
    assert json.loads(call.arguments) == {
        "path": "items.jsonl",
        "max_chars": 128,
    }


def test_responses_output_normalizer_splits_nested_call_tool_from_computer_use():
    output = [
        ResponseFunctionToolCall(
            id="fc_1",
            call_id="call_1",
            type="function_call",
            status="completed",
            name="computer_use",
            arguments=json.dumps(
                {
                    "coordinate_space": "qwen_normalized_1000",
                    "actions": [
                        {"type": "click", "x": 500, "y": 900},
                        {
                            "type": "call_tool",
                            "name": "append_trace_note",
                            "parameters": {
                                "title": "q1",
                                "summary": "3",
                                "status": "recorded",
                            },
                        },
                    ],
                },
                ensure_ascii=False,
            ),
        )
    ]

    normalized = normalize_text_tool_calls_in_response_outputs(output)

    assert len(normalized) == 2
    computer_call = normalized[0]
    assert computer_call.type == "function_call"
    assert computer_call.name == "computer_use"
    assert json.loads(computer_call.arguments) == {
        "coordinate_space": "qwen_normalized_1000",
        "actions": [{"type": "click", "x": 500, "y": 900}],
    }
    agent_call = normalized[1]
    assert agent_call.type == "function_call"
    assert agent_call.name == "append_trace_note"
    assert json.loads(agent_call.arguments) == {
        "title": "q1",
        "summary": "3",
        "status": "recorded",
    }


def test_responses_output_normalizer_splits_direct_agent_tool_action():
    output = [
        ResponseFunctionToolCall(
            id="fc_1",
            call_id="call_1",
            type="function_call",
            status="completed",
            name="computer_use",
            arguments=json.dumps(
                {
                    "coordinate_space": "qwen_normalized_1000",
                    "actions": [
                        {"type": "click", "x": 500, "y": 900},
                        {
                            "type": "append_trace_note",
                            "title": "q1",
                            "summary": "4",
                            "status": "recorded",
                        },
                    ],
                },
                ensure_ascii=False,
            ),
        )
    ]

    normalized = normalize_text_tool_calls_in_response_outputs(output)

    assert len(normalized) == 2
    computer_call = normalized[0]
    assert computer_call.type == "function_call"
    assert computer_call.name == "computer_use"
    assert json.loads(computer_call.arguments) == {
        "coordinate_space": "qwen_normalized_1000",
        "actions": [{"type": "click", "x": 500, "y": 900}],
    }
    agent_call = normalized[1]
    assert agent_call.type == "function_call"
    assert agent_call.name == "append_trace_note"
    assert json.loads(agent_call.arguments) == {
        "title": "q1",
        "summary": "4",
        "status": "recorded",
    }


def test_responses_output_normalizer_replaces_direct_agent_tool_action_only():
    output = [
        ResponseFunctionToolCall(
            id="fc_1",
            call_id="call_1",
            type="function_call",
            status="completed",
            name="computer_use",
            arguments=json.dumps(
                {
                    "coordinate_space": "qwen_normalized_1000",
                    "actions": [
                        {
                            "type": "finish_subtask",
                            "completion_reason": "done",
                        }
                    ],
                },
                ensure_ascii=False,
            ),
        )
    ]

    normalized = normalize_text_tool_calls_in_response_outputs(output)

    assert len(normalized) == 1
    call = normalized[0]
    assert call.type == "function_call"
    assert call.name == "finish_subtask"
    assert json.loads(call.arguments) == {
        "completion_reason": "done",
    }


def test_responses_output_normalizer_deduplicates_nested_call_ids():
    output = [
        ResponseFunctionToolCall(
            id="fc_1",
            call_id="call_1",
            type="function_call",
            status="completed",
            name="computer_use",
            arguments=json.dumps(
                {
                    "actions": [
                        {
                            "type": "append_trace_note",
                            "title": "first",
                            "summary": "one",
                            "status": "passed",
                        }
                    ],
                },
                ensure_ascii=False,
            ),
        ),
        ResponseFunctionToolCall(
            id="fc_2",
            call_id="call_2",
            type="function_call",
            status="completed",
            name="computer_use",
            arguments=json.dumps(
                {
                    "actions": [
                        {
                            "type": "finish_subtask",
                            "completion_reason": "done",
                        }
                    ],
                },
                ensure_ascii=False,
            ),
        ),
    ]

    normalized = normalize_text_tool_calls_in_response_outputs(output)

    assert [call.name for call in normalized] == [
        "append_trace_note",
        "finish_subtask",
    ]
    call_ids = [call.call_id for call in normalized]
    assert len(call_ids) == len(set(call_ids))


def test_responses_output_normalizer_replaces_computer_use_nested_function_only():
    output = [
        ResponseFunctionToolCall(
            id="fc_1",
            call_id="call_1",
            type="function_call",
            status="completed",
            name="computer_use",
            arguments=json.dumps(
                {
                    "coordinate_space": "qwen_normalized_1000",
                    "actions": [
                        {
                            "type": "call_function",
                            "function": {
                                "name": "finish_subtask",
                                "parameters": {
                                    "completion_reason": "done",
                                },
                            },
                        }
                    ],
                },
                ensure_ascii=False,
            ),
        )
    ]

    normalized = normalize_text_tool_calls_in_response_outputs(output)

    assert len(normalized) == 1
    call = normalized[0]
    assert call.type == "function_call"
    assert call.name == "finish_subtask"
    assert json.loads(call.arguments) == {
        "completion_reason": "done",
    }
