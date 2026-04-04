# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Literal

from pydantic import model_validator
from typing_extensions import Self

from cfie.config.utils import config
from cfie.utils.hashing import safe_hash

# structured outputs backend 的可选枚举。
StructuredOutputsBackend = Literal[
    "auto", "xgrammar", "guidance", "outlines", "lm-format-enforcer"
]


@config
class StructuredOutputsConfig:
    """Dataclass which contains structured outputs config for the engine."""

    # 结构化输出默认 backend。
    backend: StructuredOutputsBackend = "auto"
    """Which engine will be used for structured outputs (e.g. JSON schema,
    regex, etc) by default. With "auto", we will make opinionated choices
    based on request contents and what the backend libraries currently support,
    so the behavior is subject to change in each release."""
    # 是否禁止输出任意空白字符。
    disable_any_whitespace: bool = False
    """If `True`, json output will always be compact without any whitespace.
    If `False`, the model may generate whitespace between JSON fields,
    which is still valid JSON. This is only supported for xgrammar
    and guidance backends."""
    # guidance backend 下是否禁用 additionalProperties。
    disable_additional_properties: bool = False
    """If `True`, the `guidance` backend will not use `additionalProperties`
    in the JSON schema. This is only supported for the `guidance` backend and
    is used to better align its behaviour with `outlines` and `xgrammar`."""
    # 选择 reasoning parser 的名字。
    reasoning_parser: str = ""
    """Select the reasoning parser depending on the model that you're using.
    This is used to parse the reasoning content into OpenAI API format."""
    # 动态加载 reasoning parser plugin 的路径。
    reasoning_parser_plugin: str = ""
    """Path to a dynamically reasoning parser plugin that can be dynamically
    loaded and registered."""
    # 是否在 reasoning 流程里使用 structured input。
    enable_in_reasoning: bool = False
    """Whether to use structured input for reasoning."""

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        # no factors to consider.
        # this config will not affect the computation graph.
        # structured outputs 配置不影响模型图结构。
        factors: list[Any] = []
        # 生成稳定哈希。
        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
        # 返回哈希结果。
        return hash_str

    @model_validator(mode="after")
    def _validate_structured_output_config(self) -> Self:
        # disable_any_whitespace 目前只支持 xgrammar / guidance。
        if self.disable_any_whitespace and self.backend not in ("xgrammar", "guidance"):
            raise ValueError(
                "disable_any_whitespace is only supported for "
                "xgrammar and guidance backends."
            )
        # disable_additional_properties 目前只支持 guidance。
        if self.disable_additional_properties and self.backend != "guidance":
            raise ValueError(
                "disable_additional_properties is only supported "
                "for the guidance backend."
            )
        # 通过校验后返回当前对象。
        return self
