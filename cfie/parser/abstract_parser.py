# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from abc import abstractmethod
from collections.abc import Sequence
from functools import cached_property

from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputItem,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
    ToolChoiceFunction,
)
from openai.types.responses.response_output_text import Logprob
from openai.types.responses.response_reasoning_item import (
    Content as ResponseReasoningTextContent,
)
from pydantic import TypeAdapter

from cfie.entrypoints.chat_utils import make_tool_call_id
from cfie.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
)
from cfie.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    ExtractedToolCallInformation,
    FunctionCall,
    FunctionDefinition,
)
from cfie.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
)
from cfie.logger import init_logger
from cfie.reasoning.abs_reasoning_parsers import ReasoningParser
from cfie.tokenizers import TokenizerLike
from cfie.tool_parsers.abstract_tool_parser import ToolParser
from cfie.utils import random_uuid

logger = init_logger(__name__)


class Parser:
    """
    Abstract Parser class that unifies ReasoningParser and ToolParser into
    a single interface for parsing model output.
    """

    reasoning_parser_cls: type[ReasoningParser] | None = None
    tool_parser_cls: type[ToolParser] | None = None

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        self.model_tokenizer = tokenizer
        self._reasoning_parser: ReasoningParser | None = None
        self._tool_parser: ToolParser | None = None

    @cached_property
    def vocab(self) -> dict[str, int]:
        return self.model_tokenizer.get_vocab()

    @property
    def reasoning_parser(self) -> ReasoningParser | None:
        return self._reasoning_parser

    @reasoning_parser.setter
    def reasoning_parser(self, parser: ReasoningParser | None) -> None:
        self._reasoning_parser = parser

    @property
    def tool_parser(self) -> ToolParser | None:
        return self._tool_parser

    @tool_parser.setter
    def tool_parser(self, parser: ToolParser | None) -> None:
        self._tool_parser = parser

    @abstractmethod
    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        """Check if the reasoning content ends in the input_ids."""

    def is_reasoning_end_streaming(
        self, input_ids: list[int], delta_ids: list[int]
    ) -> bool:
        return self.is_reasoning_end(input_ids)

    @abstractmethod
    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """Extract content token IDs from the input_ids."""

    @abstractmethod
    def extract_response_outputs(
        self,
        model_output: str,
        request: ResponsesRequest,
        enable_auto_tools: bool = False,
        tool_call_id_type: str = "random",
        logprobs: list[Logprob] | None = None,
    ) -> list[ResponseOutputItem]:
        """Extract reasoning, content, and tool calls from model output."""

    @abstractmethod
    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        """Extract reasoning content from model output."""

    @abstractmethod
    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """Extract reasoning content from streaming delta."""

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        return request

    @abstractmethod
    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from model output."""

    @abstractmethod
    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        """Extract tool calls from streaming delta."""


class DelegatingParser(Parser):
    """A Parser that delegates to separate ReasoningParser and ToolParser instances."""

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        if self._reasoning_parser is None:
            return None, model_output
        return self._reasoning_parser.extract_reasoning(model_output, request)

    def extract_response_outputs(
        self,
        model_output: str,
        request: ResponsesRequest,
        enable_auto_tools: bool = False,
        tool_call_id_type: str = "random",
        logprobs: list[Logprob] | None = None,
    ) -> list[ResponseOutputItem]:
        reasoning, content = self.extract_reasoning(model_output, request)
        tool_calls, content = self._parse_tool_calls(
            request=request,
            content=content,
            enable_auto_tools=enable_auto_tools,
        )
        outputs: list[ResponseOutputItem] = []

        if reasoning:
            reasoning_item = ResponseReasoningItem(
                id=f"rs_{random_uuid()}",
                summary=[],
                type="reasoning",
                content=[
                    ResponseReasoningTextContent(text=reasoning, type="reasoning_text")
                ],
                status=None,
            )
            outputs.append(reasoning_item)

        if content:
            res_text_part = ResponseOutputText(
                text=content,
                annotations=[],
                type="output_text",
                logprobs=logprobs,
            )
            message_item = ResponseOutputMessage(
                id=f"msg_{random_uuid()}",
                content=[res_text_part],
                role="assistant",
                status="completed",
                type="message",
            )
            outputs.append(message_item)

        if tool_calls:
            for history_tool_call_cnt, tool_call in enumerate(tool_calls):
                tool_call_item = ResponseFunctionToolCall(
                    id=f"fc_{random_uuid()}",
                    call_id=tool_call.id
                    if tool_call.id
                    else make_tool_call_id(
                        id_type=tool_call_id_type,
                        func_name=tool_call.name,
                        idx=history_tool_call_cnt,
                    ),
                    type="function_call",
                    status="completed",
                    name=tool_call.name,
                    arguments=tool_call.arguments,
                )
                outputs.append(tool_call_item)

        return outputs

    def _parse_tool_calls(
        self,
        request: ResponsesRequest,
        content: str | None,
        enable_auto_tools: bool,
    ) -> tuple[list[FunctionCall], str | None]:
        function_calls: list[FunctionCall] = []

        if request.tool_choice and isinstance(request.tool_choice, ToolChoiceFunction):
            assert content is not None
            function_calls.append(
                FunctionCall(name=request.tool_choice.name, arguments=content)
            )
            return function_calls, None

        if request.tool_choice and isinstance(
            request.tool_choice, ChatCompletionNamedToolChoiceParam
        ):
            assert content is not None
            function_calls.append(
                FunctionCall(name=request.tool_choice.function.name, arguments=content)
            )
            return function_calls, None

        if request.tool_choice == "required":
            assert content is not None
            tool_calls = TypeAdapter(list[FunctionDefinition]).validate_json(content)
            function_calls.extend(
                FunctionCall(
                    name=tool_call.name,
                    arguments=json.dumps(tool_call.parameters, ensure_ascii=False),
                )
                for tool_call in tool_calls
            )
            return function_calls, None

        if (
            self._tool_parser is not None
            and enable_auto_tools
            and (request.tool_choice == "auto" or request.tool_choice is None)
        ):
            tool_call_info = self._tool_parser.extract_tool_calls(
                content if content is not None else "",
                request=request,  # type: ignore
            )
            if tool_call_info is not None and tool_call_info.tools_called:
                function_calls.extend(
                    FunctionCall(
                        id=tool_call.id,
                        name=tool_call.function.name,
                        arguments=tool_call.function.arguments,
                    )
                    for tool_call in tool_call_info.tool_calls
                )
                remaining_content = tool_call_info.content
                if remaining_content and remaining_content.strip() == "":
                    remaining_content = None
                return function_calls, remaining_content

        return [], content

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        if self._reasoning_parser is None:
            return DeltaMessage(content=delta_text)
        return self._reasoning_parser.extract_reasoning_streaming(
            previous_text, current_text, delta_text,
            previous_token_ids, current_token_ids, delta_token_ids,
        )

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        if self._tool_parser is None:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )
        return self._tool_parser.extract_tool_calls(model_output, request)

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        if self._tool_parser is None:
            return None
        return self._tool_parser.extract_tool_calls_streaming(
            previous_text, current_text, delta_text,
            previous_token_ids, current_token_ids, delta_token_ids,
            request,
        )


class _WrappedParser(DelegatingParser):
    """A DelegatingParser that instantiates parsers from class attributes."""

    reasoning_parser_cls: type[ReasoningParser] | None = None
    tool_parser_cls: type[ToolParser] | None = None

    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)
        if self.__class__.reasoning_parser_cls is not None:
            self._reasoning_parser = self.__class__.reasoning_parser_cls(tokenizer)
        if self.__class__.tool_parser_cls is not None:
            self._tool_parser = self.__class__.tool_parser_cls(tokenizer)
