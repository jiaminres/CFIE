"""
Schemas and utilities for tokenization inputs.
"""

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TypeAlias, TypedDict

from cfie.inputs import EmbedsPrompt, TokensPrompt

DecoderOnlyTokPrompt: TypeAlias = TokensPrompt | EmbedsPrompt
"""
A [`DecoderOnlyDictPrompt`][cfie.renderers.inputs.preprocess.DecoderOnlyDictPrompt]
that has been tokenized.
"""


EncoderTokPrompt: TypeAlias = TokensPrompt
"""
A [`EncoderDictPrompt`][cfie.renderers.inputs.preprocess.EncoderDictPrompt]
that has been tokenized.
"""


DecoderTokPrompt: TypeAlias = TokensPrompt
"""
A [`DecoderDictPrompt`][cfie.renderers.inputs.preprocess.DecoderDictPrompt]
that has been tokenized.
"""


class EncoderDecoderTokPrompt(TypedDict):
    """
    A
    [`EncoderDecoderDictPrompt`][cfie.renderers.inputs.preprocess.EncoderDecoderDictPrompt]
    that has been tokenized.
    """

    encoder_prompt: EncoderTokPrompt

    decoder_prompt: DecoderTokPrompt | None


SingletonTokPrompt: TypeAlias = (
    DecoderOnlyTokPrompt | EncoderTokPrompt | DecoderTokPrompt
)
"""
A [`SingletonDictPrompt`][cfie.renderers.inputs.preprocess.SingletonDictPrompt]
that has been tokenized.
"""


TokPrompt: TypeAlias = DecoderOnlyTokPrompt | EncoderDecoderTokPrompt
"""
A [`DictPrompt`][cfie.renderers.inputs.preprocess.DictPrompt]
that has been tokenized.
"""
