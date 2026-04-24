# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utilities for selecting and loading models."""

import inspect
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
from torch import nn
from typing_extensions import assert_never

import cfie.envs as envs
from cfie.config import ModelConfig, CfieConfig, set_current_cfie_config
from cfie.logger import init_logger
from cfie.model_executor.layers.attention import Attention, MLAAttention
from cfie.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from cfie.model_executor.model_loader.reload import (
    record_metadata_for_reloading,
    set_torchao_reload_attrs,
)
from cfie.model_executor.models.interfaces import SupportsQuant
from cfie.tracing import instrument
from cfie.utils.platform_utils import is_pin_memory_available
from cfie.utils.torch_utils import get_accelerator_view_from_cpu_tensor

logger = init_logger(__name__)


# 按 `ModelConfig` 解析模型类并实例化出“尚未加载权重”的模型结构。
@instrument(span_name="Initialize model")
def initialize_model(
    cfie_config: CfieConfig,
    *,
    prefix: str = "",
    model_class: type[nn.Module] | None = None,
    model_config: ModelConfig | None = None,
) -> nn.Module:
    """Initialize a model with the given configurations."""
    # 如果调用方没有显式传 model_config，就默认使用 cfie_config 里的主模型配置。
    if model_config is None:
        model_config = cfie_config.model_config

    # 如果调用方没有直接指定 model_class，就根据 HF architectures /
    # registry / convert_type 等信息自动解析。
    if model_class is None:
        model_class, _ = get_model_architecture(model_config)

    # 如果存在量化配置，先把 packed_modules_mapping / 名称映射同步给 quant_config，
    # 方便后续量化模块识别融合层。
    if cfie_config.quant_config is not None:
        configure_quant_config(cfie_config.quant_config, model_class)

    # 读取目标模型类的 __init__ 签名，用来区分“新式模型接口”和“旧式兼容接口”。
    signatures = inspect.signature(model_class.__init__)
    all_params = [param.name for param in signatures.parameters.values()]
    if "cfie_config" in all_params and "prefix" in all_params:
        # 当前 Qwen3.5 主模型 / MTP 模型都属于 new-style model class，
        # 例如：
        # - Qwen3_5MoeForConditionalGeneration
        # - Qwen3_5MoeMTP
        # 因此当前启动命令会走这里，而不是下面的 old-style 兼容分支。
        # new-style model class
        with set_current_cfie_config(cfie_config, check_compile=True, prefix=prefix):
            # 用统一的新式接口直接实例化模型。
            # 当前 122B-A10B 主模型会在这里进入
            # Qwen3_5MoeForConditionalGeneration.__init__。
            model = model_class(cfie_config=cfie_config, prefix=prefix)
            # 记录 reload_weights 可能需要的元数据。
            record_metadata_for_reloading(model)
            return model

    msg = (
        "vLLM model class should accept `cfie_config` and `prefix` as "
        "input arguments. Possibly you have an old-style model class"
        " registered from out of tree and it is used for new vLLM version. "
        "Check https://docs.cfie.ai/en/latest/design/arch_overview.html "
        "for the design and update the model class accordingly."
    )
    warnings.warn(msg, DeprecationWarning, stacklevel=2)

    logger.warning(
        "Trying to guess the arguments for old-style model class %s",
        model_class,
    )
    # try to be compatible with old-style model class
    # 旧式模型类不接受 cfie_config，需要根据参数名手动拼 kwargs。
    kwargs = {}
    if "prefix" in all_params:
        kwargs["prefix"] = prefix
    if "config" in all_params:
        kwargs["config"] = model_config.hf_config
    if "cache_config" in all_params:
        kwargs["cache_config"] = cfie_config.cache_config
    if "quant_config" in all_params:
        kwargs["quant_config"] = cfie_config.quant_config
    if "lora_config" in all_params:
        kwargs["lora_config"] = cfie_config.lora_config
    if "scheduler_config" in all_params:
        kwargs["scheduler_config"] = cfie_config.scheduler_config
    with set_current_cfie_config(cfie_config, check_compile=True, prefix=prefix):
        # 用“猜测出来的参数集合”实例化旧式模型。
        model = model_class(**kwargs)
        # 同样记录 reload 相关元数据。
        record_metadata_for_reloading(model)

    return model


# 入口：checkpoint 权重全部加载完成后的模型级后处理入口。
# 它不会靠同名函数层层递归调用，而是统一扫描 `model.named_modules()`，
# 再把后处理钩子分发给对应模块。
def process_weights_after_loading(
    model: nn.Module, model_config: ModelConfig, target_device: torch.device
) -> None:
    # -----------------
    # 第一轮：量化模块后处理。
    # -----------------
    # 这里主要触发量化方案自己的后处理，例如 repack、scale 初始化、
    # 内核需要的权重重排等。
    for _, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if isinstance(quant_method, QuantizeMethodBase):
            # 量化后处理通常要求参数已经位于最终 target_device 上。
            # 若当前启用了 CPU offload，这里会先临时搬到目标设备，处理完再搬回。
            with device_loading_context(module, target_device):
                # 当前 Qwen3.5-122B-A10B-GPTQ-Int4 的实际重点是：
                # - attention / shared_expert 大多走 UnquantizedLinearMethod，
                #   在 CUDA 路径这里基本没有重后处理
                # - routed experts 走 GPTQMarlinMoEMethod，
                #   这里会触发真正的 qweight repack 与 scales 重排
                quant_method.process_weights_after_loading(module)

    # -----------------
    # 第二轮：注意力模块后处理。
    # -----------------
    # Attention / MLA 的后处理放在量化模块之后，
    # 因为它们可能依赖已经解包或重排好的权重。
    for _, module in model.named_modules():
        if isinstance(module, (Attention, MLAAttention)) and hasattr(
            module, "process_weights_after_loading"
        ):
            # 注意力模块自己的后处理签名与量化方法不同，这里单独分发。
            with device_loading_context(module, target_device):
                module.process_weights_after_loading(model_config.dtype)

    # -----------------
    # 第三轮：补 torchao reload 元数据。
    # -----------------
    # torchao 仍需要额外补一层 reload 所需属性。
    if model_config.quantization == "torchao":
        set_torchao_reload_attrs(model, model_config)


# 在需要后处理时，临时把模块参数搬到目标设备，结束后再恢复原状态。
@contextmanager
def device_loading_context(module: torch.nn.Module, target_device: torch.device):
    # 目标已经是 CPU 时，不需要临时搬运参数。
    if target_device.type == "cpu":
        # 直接把原模块交给调用方。
        yield module
        # 结束当前上下文。
        return

    # 记录原始设备: param_name -> device。
    original_device_states: dict[str, torch.device] = {}
    # 记录原本走 UVA 的参数名。
    uva_offloaded_parameters: list[str] = []

    # 逐个参数准备临时加载状态。
    for name, p in module.named_parameters():
        # CPU 参数需要临时搬到 target_device。
        if p.device.type == "cpu":
            # 记住原始设备，供 finally 恢复。
            original_device_states[name] = p.device
            # CPU -> target_device。
            p.data = p.data.to(target_device)

        # 记录原本是 UVA 视图的参数。
        if getattr(p, "_cfie_is_uva_offloaded", False):
            # 保存参数名，后续重新挂回 UVA。
            uva_offloaded_parameters.append(name)

    try:
        # 把临时加载后的模块交给调用方。
        yield module

    finally:
        # 恢复到 CPU 时按平台决定是否启用 pinned memory。
        use_pin_memory = (
            is_pin_memory_available()
            and not envs.VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY
        )

        # 逐个参数恢复原始设备/UVA 形态。
        for name, p in module.named_parameters():
            # 需要恢复原设备的参数搬回原位。
            if name in original_device_states:
                # 取回该参数记录下来的原始设备。
                original_device: torch.device = original_device_states[name]
                # target_device -> original_device。
                p.data = p.data.to(original_device)

            # 原本是 UVA 参数且当前已丢失 UVA 标记时，重新挂回 UVA。
            if name in uva_offloaded_parameters and not getattr(
                p, "_cfie_is_uva_offloaded", False
            ):
                # 先把当前 device tensor 落回 CPU。
                cpu_data = p.data.to(device="cpu")
                # 按配置把 CPU tensor 升级成 pinned memory。
                if use_pin_memory:
                    cpu_data = cpu_data.pin_memory()
                # CPU tensor -> accelerator view。
                p.data = get_accelerator_view_from_cpu_tensor(cpu_data)
                # 恢复 UVA 标记。
                p._cfie_is_uva_offloaded = True


_MODEL_ARCH_BY_HASH = dict[int, tuple[type[nn.Module], str]]()
"""Caches the outputs of `_get_model_architecture`."""


# 从 HF config / registry 中解析出真正的模型类与架构名。
def _get_model_architecture(model_config: ModelConfig) -> tuple[type[nn.Module], str]:
    from cfie.model_executor.models.adapters import as_embedding_model, as_seq_cls_model

    # 从 HF 配置里取 architectures 字段。
    # 当前 122B-A10B checkpoint 的值是 ["Qwen3_5MoeForConditionalGeneration"]。
    architectures = getattr(model_config.hf_config, "architectures", [])

    # 通过 registry 把 architectures 解析成真正的模型类对象。
    model_cls, arch = model_config.registry.resolve_model_cls(
        architectures,
        model_config=model_config,
    )
    # 当前 122B-A10B checkpoint 的 architectures =
    # ["Qwen3_5MoeForConditionalGeneration"]，
    # 所以主模型这里会解析到 Qwen3_5MoeForConditionalGeneration；
    # 若进入 MTP 草稿模型路径，speculative.py 会把 architectures 改写成
    # ["Qwen3_5MoeMTP"]，这里随之解析到 Qwen3_5MoeMTP。

    if arch == model_config._get_transformers_backend_cls():
        assert model_config.model_impl != "cfie"
        if model_config.model_impl == "auto":
            logger.warning_once(
                "%s has no vLLM implementation, falling back to Transformers "
                "implementation. Some features may not be supported and "
                "performance may not be optimal.",
                arch,
            )

    # convert_type 决定是否把原始生成模型再包一层，转成 embedding/classify 模型。
    convert_type = model_config.convert_type
    if convert_type == "none":
        # 当前 chat 启动命令没有请求 embedding / classify 转换，
        # convert_type="none"，因此保留原始生成模型类。
        pass
    elif convert_type == "embed":
        logger.debug_once("Converting to embedding model.")
        model_cls = as_embedding_model(model_cls)
    elif convert_type == "classify":
        logger.debug_once("Converting to sequence classification model.")
        model_cls = as_seq_cls_model(model_cls)
    else:
        assert_never(convert_type)

    return model_cls, arch


# 带缓存地获取模型架构，避免重复做同一套 registry 解析。
def get_model_architecture(model_config: ModelConfig) -> tuple[type[nn.Module], str]:
    # 用关键字段构造缓存键，避免同一配置反复做架构解析。
    key = hash(
        (
            model_config.model,
            model_config.convert_type,
            model_config.runner_type,
            model_config.trust_remote_code,
            model_config.model_impl,
            tuple(getattr(model_config.hf_config, "architectures", [])),
        )
    )
    # 如果缓存里已有结果，直接返回。
    if key in _MODEL_ARCH_BY_HASH:
        return _MODEL_ARCH_BY_HASH[key]

    # 否则执行真实解析，并把结果写回缓存。
    model_arch = _get_model_architecture(model_config)
    _MODEL_ARCH_BY_HASH[key] = model_arch
    return model_arch


def get_model_cls(model_config: ModelConfig) -> type[nn.Module]:
    return get_model_architecture(model_config)[0]


def get_architecture_class_name(model_config: ModelConfig) -> str:
    return get_model_architecture(model_config)[1]


@dataclass
class ParamMapping:
    """
    A class to handle parameter mapping for model weight loading.
    It creates a bidirectional mapping between packed parameters and their
    constituent parts.
    """

    packed_mapping: dict[str, list[str]]
    inverse_packed_mapping: dict[str, tuple[str, int]] = field(default_factory=dict)

    def __post_init__(self):
        for packed_name, sub_params in self.packed_mapping.items():
            # Skip self-contained cases (e.g., {"W_pack": ["W_pack"]})
            if len(sub_params) == 1 and sub_params[0] == packed_name:
                continue
            for index, param_name in enumerate(sub_params):
                self.inverse_packed_mapping[param_name] = (
                    packed_name,
                    index,
                )

    def get_sub_modules(self, module_name: str) -> tuple[str, list[str]] | None:
        for key, value in self.packed_mapping.items():
            if module_name.endswith(key):
                return key, value
        return None


# 把模型类里的量化相关元数据同步到 quant_config 上。
def configure_quant_config(
    quant_config: QuantizationConfig, model_class: type[nn.Module]
):
    """
    Pass packed_modules_mapping by reference to quant_config so that
    quant_config can properly match fused modules

    Note that model attributes are passed by reference to quant_config,
    enabling them to be updated by model_class.__new__ (ex. chatglm, qwen)

    Once the `SupportsQuant` mixin has been added to all models, this
    function can be removed
    """
    if not issubclass(model_class, SupportsQuant):
        hf_to_cfie_mapper = getattr(model_class, "hf_to_cfie_mapper", None)
        packed_mapping = getattr(model_class, "packed_modules_mapping", None)

        # pass mappings by reference to quant_config
        if hf_to_cfie_mapper is not None:
            quant_config.apply_cfie_mapper(hf_to_cfie_mapper)
        if packed_mapping is not None:
            quant_config.packed_modules_mapping = packed_mapping
