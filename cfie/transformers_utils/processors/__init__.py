# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Multi-modal processors may be defined in this directory for the following
reasons:

- There is no processing file defined by HF Hub or Transformers library.
- There is a need to override the existing processor to support vLLM.
"""

import importlib

__all__ = [
    "BagelProcessor",
    "DeepseekVLV2Processor",
    "FireRedASR2Processor",
    "FunASRProcessor",
    "GLM4VProcessor",
    "HunYuanVLProcessor",
    "HunYuanVLImageProcessor",
    "KimiAudioProcessor",
    "MistralCommonPixtralProcessor",
    "MistralCommonVoxtralProcessor",
    "OvisProcessor",
    "Ovis2_5Processor",
    "QwenVLProcessor",
    "Qwen3ASRProcessor",
]

_CLASS_TO_MODULE: dict[str, str] = {
    "BagelProcessor": "cfie.transformers_utils.processors.bagel",
    "DeepseekVLV2Processor": "cfie.transformers_utils.processors.deepseek_vl2",
    "FireRedASR2Processor": "cfie.transformers_utils.processors.fireredasr2",
    "FunASRProcessor": "cfie.transformers_utils.processors.funasr",
    "GLM4VProcessor": "cfie.transformers_utils.processors.glm4v",
    "HunYuanVLProcessor": "cfie.transformers_utils.processors.hunyuan_vl",
    "HunYuanVLImageProcessor": "cfie.transformers_utils.processors.hunyuan_vl_image",
    "KimiAudioProcessor": "cfie.transformers_utils.processors.kimi_audio",
    "MistralCommonPixtralProcessor": "cfie.transformers_utils.processors.pixtral",
    "MistralCommonVoxtralProcessor": "cfie.transformers_utils.processors.voxtral",
    "OvisProcessor": "cfie.transformers_utils.processors.ovis",
    "Ovis2_5Processor": "cfie.transformers_utils.processors.ovis2_5",
    "QwenVLProcessor": "cfie.transformers_utils.processors.qwen_vl",
    "Qwen3ASRProcessor": "cfie.transformers_utils.processors.qwen3_asr",
}


def __getattr__(name: str):
    if name in _CLASS_TO_MODULE:
        module_name = _CLASS_TO_MODULE[name]
        module = importlib.import_module(module_name)
        return getattr(module, name)

    raise AttributeError(f"module 'processors' has no attribute '{name}'")


def __dir__():
    return sorted(list(__all__))
