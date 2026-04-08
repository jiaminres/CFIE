# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Model configs may be defined in this directory for the following reasons:

- There is no configuration file defined by HF Hub or Transformers library.
- There is a need to override the existing config to support vLLM.
- The HF model_type isn't recognized by the Transformers library but can
  be mapped to an existing Transformers config, such as
  deepseek-ai/DeepSeek-V3.2-Exp.
"""

from __future__ import annotations

import importlib

_CLASS_TO_MODULE: dict[str, str] = {
    "AfmoeConfig": "cfie.transformers_utils.configs.afmoe",
    "AXK1Config": "cfie.transformers_utils.configs.AXK1",
    "BagelConfig": "cfie.transformers_utils.configs.bagel",
    "ChatGLMConfig": "cfie.transformers_utils.configs.chatglm",
    "ColModernVBertConfig": "cfie.transformers_utils.configs.colmodernvbert",
    "ColPaliConfig": "cfie.transformers_utils.configs.colpali",
    "ColQwen3Config": "cfie.transformers_utils.configs.colqwen3",
    "OpsColQwen3Config": "cfie.transformers_utils.configs.colqwen3",
    "Qwen3VLNemotronEmbedConfig": "cfie.transformers_utils.configs.colqwen3",
    "DeepseekVLV2Config": "cfie.transformers_utils.configs.deepseek_vl2",
    "DotsOCRConfig": "cfie.transformers_utils.configs.dotsocr",
    "EAGLEConfig": "cfie.transformers_utils.configs.eagle",
    "FlexOlmoConfig": "cfie.transformers_utils.configs.flex_olmo",
    "FunAudioChatConfig": "cfie.transformers_utils.configs.funaudiochat",
    "FunAudioChatAudioEncoderConfig": "cfie.transformers_utils.configs.funaudiochat",
    "HunYuanVLConfig": "cfie.transformers_utils.configs.hunyuan_vl",
    "HunYuanVLTextConfig": "cfie.transformers_utils.configs.hunyuan_vl",
    "HunYuanVLVisionConfig": "cfie.transformers_utils.configs.hunyuan_vl",
    "IsaacConfig": "cfie.transformers_utils.configs.isaac",
    # RWConfig is for the original tiiuae/falcon-40b(-instruct) and
    # tiiuae/falcon-7b(-instruct) models. Newer Falcon models will use the
    # `FalconConfig` class from the official HuggingFace transformers library.
    "RWConfig": "cfie.transformers_utils.configs.falcon",
    "JAISConfig": "cfie.transformers_utils.configs.jais",
    "Lfm2MoeConfig": "cfie.transformers_utils.configs.lfm2_moe",
    "MedusaConfig": "cfie.transformers_utils.configs.medusa",
    "MiDashengLMConfig": "cfie.transformers_utils.configs.midashenglm",
    "MLPSpeculatorConfig": "cfie.transformers_utils.configs.mlp_speculator",
    "MoonViTConfig": "cfie.transformers_utils.configs.moonvit",
    "KimiLinearConfig": "cfie.transformers_utils.configs.kimi_linear",
    "KimiVLConfig": "cfie.transformers_utils.configs.kimi_vl",
    "KimiK25Config": "cfie.transformers_utils.configs.kimi_k25",
    "NemotronConfig": "cfie.transformers_utils.configs.nemotron",
    "NemotronHConfig": "cfie.transformers_utils.configs.nemotron_h",
    "Olmo3Config": "cfie.transformers_utils.configs.olmo3",
    "OlmoHybridConfig": "cfie.transformers_utils.configs.olmo_hybrid",
    "OvisConfig": "cfie.transformers_utils.configs.ovis",
    "PixelShuffleSiglip2VisionConfig": "cfie.transformers_utils.configs.isaac",
    "RadioConfig": "cfie.transformers_utils.configs.radio",
    "SpeculatorsConfig": "cfie.transformers_utils.configs.speculators.base",
    "UltravoxConfig": "cfie.transformers_utils.configs.ultravox",
    "Step3VLConfig": "cfie.transformers_utils.configs.step3_vl",
    "Step3VisionEncoderConfig": "cfie.transformers_utils.configs.step3_vl",
    "Step3TextConfig": "cfie.transformers_utils.configs.step3_vl",
    "Step3p5Config": "cfie.transformers_utils.configs.step3p5",
    "Qwen3ASRConfig": "cfie.transformers_utils.configs.qwen3_asr",
    "Qwen3NextConfig": "cfie.transformers_utils.configs.qwen3_next",
    "Qwen3_5Config": "cfie.transformers_utils.configs.qwen3_5",
    "Qwen3_5TextConfig": "cfie.transformers_utils.configs.qwen3_5",
    "Qwen3_5MoeConfig": "cfie.transformers_utils.configs.qwen3_5_moe",
    "Qwen3_5MoeTextConfig": "cfie.transformers_utils.configs.qwen3_5_moe",
    "Qwen3_5MoePredictorConfig":
        "cfie.transformers_utils.configs.qwen3_5_moe_predictor",
    "Qwen3_5MoePredictorTextConfig":
        "cfie.transformers_utils.configs.qwen3_5_moe_predictor",
    "Qwen3_5MoePredictorVisionConfig":
        "cfie.transformers_utils.configs.qwen3_5_moe_predictor",
    "Tarsier2Config": "cfie.transformers_utils.configs.tarsier2",
    # Special case: DeepseekV3Config is from HuggingFace Transformers
    "DeepseekV3Config": "transformers",
}

__all__ = [
    "AfmoeConfig",
    "AXK1Config",
    "BagelConfig",
    "ChatGLMConfig",
    "ColModernVBertConfig",
    "ColPaliConfig",
    "ColQwen3Config",
    "OpsColQwen3Config",
    "Qwen3VLNemotronEmbedConfig",
    "DeepseekVLV2Config",
    "DeepseekV3Config",
    "DotsOCRConfig",
    "EAGLEConfig",
    "FlexOlmoConfig",
    "FunAudioChatConfig",
    "FunAudioChatAudioEncoderConfig",
    "HunYuanVLConfig",
    "HunYuanVLTextConfig",
    "HunYuanVLVisionConfig",
    "IsaacConfig",
    "RWConfig",
    "JAISConfig",
    "Lfm2MoeConfig",
    "MedusaConfig",
    "MiDashengLMConfig",
    "MLPSpeculatorConfig",
    "MoonViTConfig",
    "KimiLinearConfig",
    "KimiVLConfig",
    "KimiK25Config",
    "NemotronConfig",
    "NemotronHConfig",
    "Olmo3Config",
    "OlmoHybridConfig",
    "OvisConfig",
    "PixelShuffleSiglip2VisionConfig",
    "RadioConfig",
    "SpeculatorsConfig",
    "UltravoxConfig",
    "Step3VLConfig",
    "Step3VisionEncoderConfig",
    "Step3TextConfig",
    "Step3p5Config",
    "Qwen3ASRConfig",
    "Qwen3NextConfig",
    "Qwen3_5Config",
    "Qwen3_5TextConfig",
    "Qwen3_5MoeConfig",
    "Qwen3_5MoeTextConfig",
    "Qwen3_5MoePredictorConfig",
    "Qwen3_5MoePredictorTextConfig",
    "Qwen3_5MoePredictorVisionConfig",
    "Tarsier2Config",
]


def __getattr__(name: str):
    if name in _CLASS_TO_MODULE:
        module_name = _CLASS_TO_MODULE[name]
        module = importlib.import_module(module_name)
        return getattr(module, name)

    raise AttributeError(f"module 'configs' has no attribute '{name}'")


def __dir__():
    return sorted(list(__all__))
