# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""MiniMax M2 Parser combining reasoning and tool parsing."""

from cfie.logger import init_logger
from cfie.parser.abstract_parser import DelegatingParser
from cfie.reasoning.minimax_m2_reasoning_parser import MiniMaxM2ReasoningParser
from cfie.tokenizers import TokenizerLike
from cfie.tool_parsers.minimax_m2_tool_parser import MinimaxM2ToolParser

logger = init_logger(__name__)


class MiniMaxM2Parser(DelegatingParser):
    """Unified parser for MiniMax M2 models."""

    reasoning_parser_cls = MiniMaxM2ReasoningParser
    tool_parser_cls = MinimaxM2ToolParser

    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)
        self._reasoning_parser = MiniMaxM2ReasoningParser(tokenizer)
        self._tool_parser = MinimaxM2ToolParser(tokenizer)
        logger.debug(
            "Successfully initialized parser %s!", self.__class__.__name__
        )
