# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import regex as re

from cfie.logger import init_logger
from cfie.tokenizers import TokenizerLike
from cfie.tool_parsers.glm4_moe_tool_parser import Glm4MoeModelToolParser

logger = init_logger(__name__)


class Glm47MoeModelToolParser(Glm4MoeModelToolParser):
    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)
        self.func_detail_regex = re.compile(
            r"<tool_call>(.*?)(<arg_key>.*?)?</tool_call>", re.DOTALL
        )
        self.func_arg_regex = re.compile(
            r"<arg_key>(.*?)</arg_key>(?:\\n|\s)*<arg_value>(.*?)</arg_value>",
            re.DOTALL,
        )
