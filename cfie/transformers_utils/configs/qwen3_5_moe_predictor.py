# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.5-MoE predictor model configuration."""

from cfie.transformers_utils.configs.qwen3_5_moe import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
    Qwen3_5MoeVisionConfig,
)


class Qwen3_5MoePredictorTextConfig(Qwen3_5MoeTextConfig):
    model_type = "qwen3_5_moe_predictor_text"


class Qwen3_5MoePredictorVisionConfig(Qwen3_5MoeVisionConfig):
    model_type = "qwen3_5_moe_predictor"


class Qwen3_5MoePredictorConfig(Qwen3_5MoeConfig):
    model_type = "qwen3_5_moe_predictor"
    sub_configs = {
        "vision_config": Qwen3_5MoePredictorVisionConfig,
        "text_config": Qwen3_5MoePredictorTextConfig,
    }

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=248056,
        video_token_id=248057,
        vision_start_token_id=248053,
        vision_end_token_id=248054,
        tie_word_embeddings=False,
        predictor_bundle_path: str | None = None,
        predictor_map_location: str = "cpu",
        predictor_device: str = "cpu",
        **kwargs,
    ):
        self.predictor_bundle_path = predictor_bundle_path
        self.predictor_map_location = predictor_map_location
        self.predictor_device = predictor_device
        super().__init__(
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
            vision_start_token_id=vision_start_token_id,
            vision_end_token_id=vision_end_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = [
    "Qwen3_5MoePredictorConfig",
    "Qwen3_5MoePredictorTextConfig",
    "Qwen3_5MoePredictorVisionConfig",
]
