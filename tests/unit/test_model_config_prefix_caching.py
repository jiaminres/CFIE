from unittest.mock import PropertyMock, patch

from cfie.config.model import ModelConfig


def test_hybrid_moe_model_supports_prefix_caching_by_default():
    model_config = object.__new__(ModelConfig)
    model_config.pooler_config = None

    with (
        patch.object(
            ModelConfig,
            "attn_type",
            new_callable=PropertyMock,
            return_value="hybrid",
        ),
        patch.object(
            ModelConfig,
            "is_moe",
            new_callable=PropertyMock,
            return_value=True,
        ),
    ):
        assert model_config.is_prefix_caching_supported is True


def test_hybrid_non_moe_model_keeps_prefix_caching_disabled_by_default():
    model_config = object.__new__(ModelConfig)
    model_config.pooler_config = None

    with (
        patch.object(
            ModelConfig,
            "attn_type",
            new_callable=PropertyMock,
            return_value="hybrid",
        ),
        patch.object(
            ModelConfig,
            "is_moe",
            new_callable=PropertyMock,
            return_value=False,
        ),
    ):
        assert model_config.is_prefix_caching_supported is False
