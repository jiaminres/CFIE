# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from typing import Any

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionToolMessageParam,
)
from openai.types.chat.chat_completion_message_tool_call_param import (
    Function as FunctionCallTool,
)
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputItem
from openai.types.responses.response import ToolChoice
from openai.types.responses.response_function_tool_call_output_item import (
    ResponseFunctionToolCallOutputItem,
)
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_reasoning_item import ResponseReasoningItem
from openai.types.responses.tool import Tool

from cfie import envs
from cfie.entrypoints.constants import MCP_PREFIX
from cfie.entrypoints.openai.chat_completion.protocol import ChatCompletionMessageParam
from cfie.entrypoints.openai.responses.input_sanitizer import (
    strip_inline_media_from_text,
)
from cfie.entrypoints.openai.responses.protocol import ResponseInputOutputItem
from cfie.logger import init_logger

logger = init_logger(__name__)


def should_continue_final_message(
    request_input: str | list[ResponseInputOutputItem],
) -> bool:
    """
    Determine if the last input message is a partial assistant message
    that should be continued rather than starting a new generation.

    This enables partial message completion similar to Anthropic's Messages API,
    where users can provide an incomplete assistant message and have the model
    continue from where it left off.

    A message is considered partial if:
    1. It's a ResponseOutputMessage or ResponseReasoningItem
    2. Its status is "in_progress" or "incomplete"

    Args:
        request_input: The input to the Responses API request

    Returns:
        True if the final message should be continued, False otherwise
    """
    if isinstance(request_input, str):
        # Simple string input is always a user message
        return False

    if not request_input:
        return False

    last_item = request_input[-1]

    # Check if the last item is a partial assistant message
    if isinstance(last_item, ResponseOutputMessage):
        return last_item.status in ("in_progress", "incomplete")

    # Check if the last item is a partial reasoning item
    if isinstance(last_item, ResponseReasoningItem):
        return last_item.status in ("in_progress", "incomplete")

    if isinstance(last_item, dict):
        # only support partial completion for messages for now
        if last_item.get("type", "message") not in ("message", "reasoning"):
            return False
        return last_item.get("status") in ("in_progress", "incomplete")

    return False


def construct_input_messages(
    *,
    request_instructions: str | None = None,
    request_input: str | list[ResponseInputOutputItem],
    prev_msg: list[ChatCompletionMessageParam] | None = None,
    prev_response_output: list[ResponseOutputItem] | None = None,
):
    messages: list[ChatCompletionMessageParam] = []
    if request_instructions:
        messages.append(
            {
                "role": "system",
                "content": strip_inline_media_from_text(request_instructions),
            }
        )

    # Prepend the conversation history.
    if prev_msg is not None:
        # Add the previous messages.
        messages.extend(prev_msg)
    if prev_response_output is not None:
        # Add the previous output.
        for output_item in prev_response_output:
            # NOTE: We skip the reasoning output.
            if isinstance(output_item, ResponseOutputMessage):
                for content in output_item.content:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": content.text,
                        }
                    )

    # Append the new input.
    # Responses API supports simple text inputs without chat format.
    if isinstance(request_input, str):
        messages.append(
            {"role": "user", "content": strip_inline_media_from_text(request_input)}
        )
    else:
        input_messages = construct_chat_messages_with_tool_call(request_input)
        messages.extend(input_messages)
    return messages


def _maybe_combine_reasoning_and_tool_call(
    item: ResponseInputOutputItem, messages: list[ChatCompletionMessageParam]
) -> ChatCompletionMessageParam | None:
    """Many models treat MCP calls and reasoning as a single message.
    This function checks if the last message is a reasoning message and
    the current message is a tool call"""
    if not (
        isinstance(item, ResponseFunctionToolCall)
        and item.id
        and item.id.startswith(MCP_PREFIX)
    ):
        return None
    if len(messages) == 0:
        return None
    last_message = messages[-1]
    if not (
        last_message.get("role") == "assistant"
        and last_message.get("reasoning") is not None
    ):
        return None

    last_message["tool_calls"] = [
        ChatCompletionMessageToolCallParam(
            id=item.call_id,
            function=FunctionCallTool(
                name=item.name,
                arguments=item.arguments,
            ),
            type="function",
        )
    ]
    return last_message


def construct_chat_messages_with_tool_call(
    input_messages: list[ResponseInputOutputItem],
) -> list[ChatCompletionMessageParam]:
    """This function wraps _construct_single_message_from_response_item
    Because some chatMessages come from multiple response items
    for example a reasoning item and a MCP tool call are two response items
    but are one chat message
    """
    messages: list[ChatCompletionMessageParam] = []
    for item in input_messages:
        if _response_item_type(item) == "computer_call_output":
            messages.extend(_construct_computer_call_output_messages(item))
            continue
        maybe_combined_message = _maybe_combine_reasoning_and_tool_call(item, messages)
        if maybe_combined_message is not None:
            messages[-1] = maybe_combined_message
        else:
            messages.append(_construct_single_message_from_response_item(item))

    return messages


def _construct_single_message_from_response_item(
    item: ResponseInputOutputItem,
) -> ChatCompletionMessageParam:
    if isinstance(item, ResponseFunctionToolCall):
        # Append the function call as a tool call.
        return ChatCompletionAssistantMessageParam(
            role="assistant",
            tool_calls=[
                ChatCompletionMessageToolCallParam(
                    id=item.call_id,
                    function=FunctionCallTool(
                        name=item.name,
                        arguments=item.arguments,
                    ),
                    type="function",
                )
            ],
        )
    elif isinstance(item, dict) and item.get("type") in {
        "function_call",
        "tool_call",
    }:
        arguments = item.get("arguments", "")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        return ChatCompletionAssistantMessageParam(
            role="assistant",
            tool_calls=[
                ChatCompletionMessageToolCallParam(
                    id=str(item.get("call_id") or item.get("id") or item.get("name")),
                    function=FunctionCallTool(
                        name=str(item.get("name") or ""),
                        arguments=arguments,
                    ),
                    type="function",
                )
            ],
        )
    elif isinstance(item, dict) and item.get("type") == "computer_call":
        arguments = {
            key: value
            for key, value in item.items()
            if key not in {"type", "id", "call_id", "status"}
        }
        return ChatCompletionAssistantMessageParam(
            role="assistant",
            tool_calls=[
                ChatCompletionMessageToolCallParam(
                    id=str(item.get("call_id") or item.get("id") or "computer_use"),
                    function=FunctionCallTool(
                        name="computer_use",
                        arguments=strip_inline_media_from_text(
                            json.dumps(arguments, ensure_ascii=False)
                        ),
                    ),
                    type="function",
                )
            ],
        )
    elif isinstance(item, ResponseReasoningItem):
        reasoning_content = ""
        if item.encrypted_content:
            raise ValueError("Encrypted content is not supported.")
        elif item.content and len(item.content) >= 1:
            reasoning_content = item.content[0].text
        elif len(item.summary) >= 1:
            reasoning_content = item.summary[0].text
            logger.warning(
                "Using summary text as reasoning content for item %s. "
                "Please use content instead of summary for "
                "reasoning items.",
                item.id,
            )
        return {
            "role": "assistant",
            "reasoning": reasoning_content,
        }
    elif isinstance(item, ResponseOutputMessage):
        return {
            "role": "assistant",
            "content": item.content[0].text,
        }
    elif isinstance(item, ResponseFunctionToolCallOutputItem):
        return ChatCompletionToolMessageParam(
            role="tool",
            content=strip_inline_media_from_text(item.output),
            tool_call_id=item.call_id,
        )
    elif isinstance(item, dict) and item.get("type") == "function_call_output":
        # Append the function call output as a tool message.
        output = item.get("output")
        if isinstance(output, str):
            output = strip_inline_media_from_text(output)
        return ChatCompletionToolMessageParam(
            role="tool",
            content=output,
            tool_call_id=item.get("call_id"),
        )
    return item  # type: ignore


def _construct_computer_call_output_messages(
    item: ResponseInputOutputItem,
) -> list[ChatCompletionMessageParam]:
    call_id = str(_response_item_field(item, "call_id", "computer_use"))
    output = _response_item_field(item, "output", {}) or {}
    if not isinstance(output, dict):
        return [
            ChatCompletionToolMessageParam(
                role="tool",
                content="computer_use returned no screenshot output.",
                tool_call_id=call_id,
            )
        ]

    detail = output.get("detail", "low")
    image_url = output.get("image_url")
    summary = str(output.get("summary") or "").strip()
    local_refinements = output.get("local_refinements")
    if not isinstance(local_refinements, list):
        local_refinements = []
    content = (
        f"computer_use completed. {summary} Screenshot observation is provided "
        "as the following image message."
        if summary
        else (
            "computer_use completed. Screenshot observation is provided "
            "as the following image message."
        )
    )
    if local_refinements:
        content += (
            f" {len(local_refinements)} click-local refinement image(s) are "
            "included in the same observation."
        )
    tool_content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": content,
        }
    ]
    if image_url:
        tool_content.append(
            {
                "type": "input_image",
                "image_url": image_url,
                "detail": detail,
            }
        )
        for index, refinement in enumerate(local_refinements, start=1):
            if not isinstance(refinement, dict):
                continue
            refinement_image_url = refinement.get("image_url")
            if not refinement_image_url:
                continue
            refinement_summary = str(refinement.get("summary") or "").strip()
            if not refinement_summary:
                refinement_summary = (
                    f"Click-local refinement image {index} after computer_use "
                    f"call_id={call_id}."
                )
            tool_content.append({"type": "input_text", "text": refinement_summary})
            tool_content.append(
                {
                    "type": "input_image",
                    "image_url": refinement_image_url,
                    "detail": refinement.get("detail", detail),
                }
            )
    return [
        ChatCompletionToolMessageParam(
            role="tool",
            content=tool_content,  # type: ignore[typeddict-item]
            tool_call_id=call_id,
        )
    ]


def _response_item_type(item: ResponseInputOutputItem) -> str | None:
    return _response_item_field(item, "type", None)


def _response_item_field(
    item: ResponseInputOutputItem,
    field_name: str,
    default: Any = None,
) -> Any:
    if isinstance(item, dict):
        return item.get(field_name, default)
    return getattr(item, field_name, default)


def extract_tool_types(tools: list[Tool]) -> set[str]:
    """
    Extracts the tool types from the given tools.
    """
    tool_types: set[str] = set()
    for tool in tools:
        if tool.type == "mcp":
            # Allow the MCP Tool type to enable built in tools if the
            # server_label is allowlisted in
            # envs.VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS
            if tool.server_label in envs.VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS:
                tool_types.add(tool.server_label)
        else:
            tool_types.add(tool.type)
    return tool_types


def convert_tool_responses_to_completions_format(tool: dict) -> dict:
    """
    Convert a flat tool schema:
        {"type": "function", "name": "...", "description": "...", "parameters": {...}}
    into:
        {"type": "function", "function": {...}}
    """
    return {
        "type": "function",
        "function": tool,
    }


def construct_tool_dicts(
    tools: list[Tool], tool_choice: ToolChoice
) -> list[dict[str, Any]] | None:
    if tools is None or (tool_choice == "none"):
        tool_dicts = None
    else:
        tool_dicts = [
            convert_tool_responses_to_completions_format(tool.model_dump())
            for tool in tools
        ]
    return tool_dicts
