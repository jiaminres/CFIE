# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from typing import Literal, get_args

from cfie.logger import init_logger
from cfie.model_executor.layers.quantization.base_config import QuantizationConfig
from cfie.platforms import current_platform

logger = init_logger(__name__)

QuantizationMethods = Literal[
    "awq",
    "fp8",
    "ptpc_fp8",
    "fbgemm_fp8",
    "fp_quant",
    "modelopt",
    "modelopt_fp4",
    "modelopt_mxfp8",
    "modelopt_mixed",
    "gguf",
    "gptq_marlin",
    "awq_marlin",
    "gptq",
    "compressed-tensors",
    "bitsandbytes",
    "experts_int8",
    "quark",
    "moe_wna16",
    "torchao",
    "inc",
    "mxfp4",
    "petit_nvfp4",
    "cpu_awq",
]
QUANTIZATION_METHODS: list[str] = list(get_args(QuantizationMethods))

DEPRECATED_QUANTIZATION_METHODS = [
    "tpu_int8",
    "ptpc_fp8",
    "fbgemm_fp8",
    "fp_quant",
    "experts_int8",
    "petit_nvfp4",
]

# The customized quantization methods which will be added to this dict.
_CUSTOMIZED_METHOD_TO_QUANT_CONFIG = {}


def register_quantization_config(quantization: str):
    """Register a customized cfie quantization config.

    When a quantization method is not supported by cfie, you can register a customized
    quantization config to support it.

    Args:
        quantization (str): The quantization method name.

    Examples:
        >>> from cfie.model_executor.layers.quantization import (
        ...     register_quantization_config,
        ... )
        >>> from cfie.model_executor.layers.quantization import get_quantization_config
        >>> from cfie.model_executor.layers.quantization.base_config import (
        ...     QuantizationConfig,
        ... )
        >>>
        >>> @register_quantization_config("my_quant")
        ... class MyQuantConfig(QuantizationConfig):
        ...     pass
        >>>
        >>> get_quantization_config("my_quant")
        <class 'MyQuantConfig'>
    """  # noqa: E501

    def _wrapper(quant_config_cls):
        if quantization in QUANTIZATION_METHODS:
            logger.warning(
                "The quantization method '%s' already exists and will be "
                "overwritten by the quantization config %s.",
                quantization,
                quant_config_cls,
            )
        else:
            QUANTIZATION_METHODS.append(quantization)
            # Automatically assume the custom quantization config is supported
            if sq := current_platform.supported_quantization:
                sq.append(quantization)

        if not issubclass(quant_config_cls, QuantizationConfig):
            raise ValueError(
                "The quantization config must be a subclass of `QuantizationConfig`."
            )
        _CUSTOMIZED_METHOD_TO_QUANT_CONFIG[quantization] = quant_config_cls
        return quant_config_cls

    return _wrapper


def get_quantization_config(quantization: str) -> type[QuantizationConfig]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(f"Invalid quantization method: {quantization}")

    method_to_config_ref: dict[str, tuple[str, str]] = {
        "awq": (".awq", "AWQConfig"),
        "fp8": (".fp8", "Fp8Config"),
        "fbgemm_fp8": (".fbgemm_fp8", "FBGEMMFp8Config"),
        "fp_quant": (".fp_quant", "FPQuantConfig"),
        "modelopt": (".modelopt", "ModelOptFp8Config"),
        "modelopt_fp4": (".modelopt", "ModelOptNvFp4Config"),
        "modelopt_mxfp8": (".modelopt", "ModelOptMxFp8Config"),
        "modelopt_mixed": (".modelopt", "ModelOptMixedPrecisionConfig"),
        "gguf": (".gguf", "GGUFConfig"),
        "gptq_marlin": (".gptq_marlin", "GPTQMarlinConfig"),
        "awq_marlin": (".awq_marlin", "AWQMarlinConfig"),
        "gptq": (".gptq", "GPTQConfig"),
        "compressed-tensors": (
            ".compressed_tensors.compressed_tensors",
            "CompressedTensorsConfig",
        ),
        "bitsandbytes": (".bitsandbytes", "BitsAndBytesConfig"),
        "ptpc_fp8": (".ptpc_fp8", "PTPCFp8Config"),
        "experts_int8": (".experts_int8", "ExpertsInt8Config"),
        "quark": (".quark.quark", "QuarkConfig"),
        "moe_wna16": (".moe_wna16", "MoeWNA16Config"),
        "torchao": (".torchao", "TorchAOConfig"),
        "auto-round": (".inc", "INCConfig"),
        "inc": (".inc", "INCConfig"),
        "mxfp4": (".mxfp4", "Mxfp4Config"),
        "petit_nvfp4": (".petit", "PetitNvFp4Config"),
        "cpu_awq": (".cpu_wna16", "CPUAWQConfig"),
    }
    # Update the `method_to_config` with customized quantization methods.
    if quantization in _CUSTOMIZED_METHOD_TO_QUANT_CONFIG:
        return _CUSTOMIZED_METHOD_TO_QUANT_CONFIG[quantization]

    module_name, class_name = method_to_config_ref[quantization]
    module = importlib.import_module(module_name, package=__name__)
    return getattr(module, class_name)


__all__ = [
    "QuantizationConfig",
    "QuantizationMethods",
    "get_quantization_config",
    "register_quantization_config",
    "QUANTIZATION_METHODS",
]
