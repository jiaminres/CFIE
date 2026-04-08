# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from typing import Any

from transformers import PretrainedConfig

from cfie.transformers_utils.configs.speculators.algos import (
    SUPPORTED_SPECULATORS_TYPES,
)

__all__ = ["SpeculatorsConfig"]

from cfie.transformers_utils.utils import without_trust_remote_code


class SpeculatorsConfig(PretrainedConfig):
    model_type = "speculators"

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        **kwargs,
    ) -> "SpeculatorsConfig":
        """Load speculators Eagle config and convert to vLLM format."""
        config_dict, _ = cls.get_config_dict(
            pretrained_model_name_or_path, **without_trust_remote_code(kwargs)
        )

        cfie_config = cls.extract_transformers_pre_trained_config(config_dict)
        return cls(**cfie_config)

    @classmethod
    def extract_transformers_pre_trained_config(
        cls, config_dict: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Extract standard Transformers PreTrainedConfig config from speculators config.
        """
        speculators_model_type = config_dict.get("speculators_model_type")
        if speculators_model_type not in SUPPORTED_SPECULATORS_TYPES:
            raise ValueError(
                f"Expected one of: {SUPPORTED_SPECULATORS_TYPES}. "
                "Please ensure you're loading a speculators-format model."
            )

        # Start with transformer layer configuration if present
        pre_trained_config = config_dict.get("transformer_layer_config", {})
        # Apply anything specific to the supported algorithm
        algo_updater = SUPPORTED_SPECULATORS_TYPES[speculators_model_type]
        algo_updater(config_dict=config_dict, pre_trained_config=pre_trained_config)
        return pre_trained_config

    @classmethod
    def extract_cfie_speculative_config(
        # speculators 格式的完整配置字典。
        cls, config_dict: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract vLLM speculative config from speculators config."""

        # 先校验 speculators 配置结构是否完整。

        cls.validate_speculators_config(config_dict=config_dict)
        # Convert from speculators config -> format that can be ingested by vLLM
        # 再把 speculators 配置转换成 cfie/vLLM 可直接消费的格式。
        return cls.build_cfie_speculative_config(config_dict=config_dict)

    @classmethod
    def validate_speculators_config(cls, config_dict: dict[str, Any]) -> None:
        try:
            # 取出 speculators_config 主体。
            spec_config = config_dict["speculators_config"]
            # 取出 proposal_methods 列表。
            methods = spec_config["proposal_methods"]
            # 当前只检查第一种 proposal method。
            first_method = methods[0]
            # 要求该 method 必须声明 speculative_tokens。
            _ = first_method["speculative_tokens"]
            # 要求 verifier 节点必须给出主模型路径。
            _ = spec_config["verifier"]["name_or_path"]
            # 要求顶层必须声明 speculators_model_type。
            _ = config_dict["speculators_model_type"]
        except (KeyError, IndexError, TypeError) as e:
            # 上述关键字段缺失或类型错误时，统一视为非法结构。
            raise ValueError("Invalid speculators config structure") from e

        # 必须显式提供 transformer_layer_config。
        if "transformer_layer_config" not in config_dict:
            raise ValueError("Must provide transformer_layer_config")

        # transformer_layer_config 必须是字典结构。
        if not isinstance(config_dict["transformer_layer_config"], dict):
            raise TypeError(
                "'transformer_layer_config' must be a dictionary if provided"
            )

    @classmethod
    def build_cfie_speculative_config(
        # speculators 格式的完整配置字典。
        cls, config_dict: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Build vLLM-compatible speculative configuration from speculators format.

        This method extracts and transforms speculative configuration from the
        speculators format into the structure expected by vLLM.

        Args:
            config_dict: Configuration dictionary in speculators format

        Returns:
            Dictionary with vLLM-compatible speculative configuration
        """
        # Extract speculators configuration
        # 取出 speculators_config 主体。
        spec_config = config_dict["speculators_config"]


        # 读取 proposal method 列表。
        proposal_methods = spec_config.get("proposal_methods")
        if not proposal_methods:
            raise ValueError("No proposal methods found in speculators config")

        # 当前只使用第一种 proposal method。
        first_method = proposal_methods[0]
        # 提取该 method 声明的 speculative token 数。
        num_speculative_tokens = first_method.get("speculative_tokens")

        if num_speculative_tokens is None:
            raise ValueError(
                f"Missing 'speculative_tokens' in proposal method. Got: {first_method}"
            )

        # Build base vLLM speculative configuration
        # 构造 cfie/vLLM 侧需要的最小 speculative config。
        return {
            # method 直接使用 speculators_model_type。
            "method": config_dict.get("speculators_model_type"),
            # 透传 speculative token 数。
            "num_speculative_tokens": num_speculative_tokens,
        }
