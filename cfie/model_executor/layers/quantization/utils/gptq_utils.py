# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Mapping
from copy import deepcopy
from fractions import Fraction
from types import MappingProxyType
from typing import TYPE_CHECKING

import regex as re
import torch

from cfie.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from cfie.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)

if TYPE_CHECKING:
    from ..gptq import GPTQConfig
    from ..gptq_marlin import GPTQMarlinConfig
else:
    GPTQConfig = object
    GPTQMarlinConfig = object


def override_config(config: GPTQConfig | GPTQMarlinConfig, prefix: str):
    # -------------------- 按模块名前缀应用 dynamic override --------------------
    # 这里的 prefix 就是当前层的模块名前缀。
    # 函数会用它去匹配 quant_config.dynamic 里的正匹配规则，
    # 若命中，就把当前层专属的量化字段覆写到这份 config 上。

    # -------------------- 逐项覆写 GPTQ 通用字段 --------------------
    # 若 dynamic 为当前 prefix 指定了 bits，则以该层覆盖值为准。
    weight_bits = get_dynamic_override(config, prefix, "bits", config.weight_bits)
    if isinstance(weight_bits, int):
        config.weight_bits = weight_bits
    # 若 dynamic 为当前 prefix 指定了 group_size，则用覆盖值替换默认值。
    group_size = get_dynamic_override(config, prefix, "group_size", config.group_size)
    if isinstance(group_size, int):
        config.group_size = group_size
    # 若 dynamic 为当前 prefix 指定了 desc_act，也一并覆盖。
    desc_act = get_dynamic_override(config, prefix, "desc_act", config.desc_act)
    if isinstance(desc_act, bool):
        config.desc_act = desc_act

    # bits 变化后，一个 int32 能容纳多少个量化值也会随之变化，因此需要重算 pack_factor。
    config.pack_factor = Fraction(32, config.weight_bits)  # packed into int32

    # -------------------- 按具体量化后端补齐派生字段 --------------------
    if config.get_name() == "gptq_marlin":
        assert isinstance(config, GPTQMarlinConfig)
        # GPTQ Marlin 还允许按层覆盖 sym，因此这里继续读取并覆写。
        is_sym = get_dynamic_override(config, prefix, "sym", config.is_sym)
        if isinstance(is_sym, bool):
            config.is_sym = is_sym

        # 覆写后的 (bits, sym) 组合必须仍然落在 GPTQ Marlin 的支持表内。
        if (config.weight_bits, config.is_sym) not in config.TYPE_MAP:
            raise ValueError(
                "Unsupported quantization config: "
                f"bits={config.weight_bits}, sym={config.is_sym}"
            )

        # (bits, sym) 一旦改变，对应 kernel 使用的 quant_type 也必须同步重算。
        config.quant_type = config.TYPE_MAP[(config.weight_bits, config.is_sym)]
    elif config.get_name() == "gptq":
        assert isinstance(config, GPTQConfig)
        # 普通 GPTQ 路径只接受有限的 bit 数集合，覆写后同样要重新校验。
        if config.weight_bits not in [2, 3, 4, 8]:
            raise ValueError(
                "Currently, only 2/3/4/8-bit weight quantization is "
                f"supported for GPTQ, but got {config.weight_bits} bits."
            )


def get_dynamic_override(
    config: GPTQConfig | GPTQMarlinConfig,
    layer_name: str,
    key: str | None = None,
    default_value: int | bool | None = None,
) -> dict | int | bool | None:
    for pattern, pattern_dict in config.dynamic.items():
        #
        # 当前本地 Qwen3.5-122B-A10B-GPTQ-Int4 的典型规则：
        # -:.*attn.*           -> self_attn / linear-attn 相关层不量化
        # -:.*shared_expert.*  -> shared expert 不量化
        # -:.*mtp.*            -> MTP 由独立 drafter 处理
        # -:.*visual.*         -> 视觉塔不量化
        #
        # 这里一旦返回 False，外层就不会构造 GPTQ/GPTQ-Marlin 方法，
        # 而会直接退回 UnquantizedLinearMethod / UnquantizedFusedMoEMethod。
        if pattern.startswith("-:"):
            if re.match(pattern.removeprefix("-:"), layer_name):
                return False

        elif re.match(pattern.removeprefix("+:"), layer_name):
            if key is None:
                return pattern_dict
            else:
                return pattern_dict.get(key, default_value)
    return default_value


def is_layer_gptq_quantized(
    prefix: str,
    quantized_layers: list[str],
    fused_mapping: Mapping[str, list[str]] = MappingProxyType({}),
) -> bool:
    # -------------------- 先从完整层名前缀中提取最后一级投影名 --------------------
    # 例如：
    #   prefix = "model.layers.0.self_attn.q_proj"
    # 则：
    #   proj_name = "q_proj"
    #
    # 又比如：
    #   prefix = "model.layers.0.self_attn.qkv_proj"
    # 则：
    #   proj_name = "qkv_proj"

    proj_name = prefix.split(".")[-1]

    # -------------------- 若当前层是融合层，则先映射回磁盘上的非融合分片名 --------------------
    # GPTQ 的 `modules_in_block_to_quantize` 往往记录的是子串规则，例如：
    #   ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"]
    # 而运行时这里拿到的可能是融合层名字，例如：
    #   qkv_proj / gate_up_proj
    #
    # safetensors checkpoint 通常不会把这些层按“融合后名字”存盘，
    # 因此这里要先把融合层名展开成真实 shard 名，再逐个判断它们是否量化。
    #
    # 例子 1：
    #   prefix = "model.layers.0.self_attn.qkv_proj"
    #   fused_mapping["qkv_proj"] = ["q_proj", "k_proj", "v_proj"]
    #
    # 则会展开成：
    #   [
    #     "model.layers.0.self_attn.q_proj",
    #     "model.layers.0.self_attn.k_proj",
    #     "model.layers.0.self_attn.v_proj",
    #   ]
    #
    # 然后分别判断这 3 个 shard 是否都命中 quantized_layers。
    #
    # 例子 2：
    #   prefix = "model.layers.0.mlp.gate_up_proj"
    #   fused_mapping["gate_up_proj"] = ["gate_proj", "up_proj"]
    #
    # 则会展开成：
    #   [
    #     "model.layers.0.mlp.gate_proj",
    #     "model.layers.0.mlp.up_proj",
    #   ]
    if proj_name in fused_mapping:
        # 把当前融合层前缀替换成各个未融合 shard 的前缀列表。
        shard_prefixes = [
            prefix.replace(proj_name, shard_proj_name)
            for shard_proj_name in fused_mapping[proj_name]
        ]

        # 先把最终判断结果置空，后面用第一个 shard 的结果作为基准。
        is_quantized = None
        # 逐个检查融合层对应的每个 shard 前缀是否命中了 quantized_layers 规则。
        for shard_prefix in shard_prefixes:
            # 只要 quantized_layers 中任一子串出现在当前 shard_prefix 里，就认为该 shard 量化。
            #
            # 例如：
            #   quantized_layers = [
            #     "self_attn.q_proj",
            #     "self_attn.k_proj",
            #     "self_attn.v_proj",
            #   ]
            #
            # 那么：
            #   shard_prefix = "model.layers.0.self_attn.q_proj"
            # 会命中 "self_attn.q_proj"，因此 is_shard_quantized=True。
            is_shard_quantized = any(
                layer in shard_prefix for layer in quantized_layers
            )

            # 第一个 shard 的判断结果直接作为整组融合层的初始基准。
            if is_quantized is None:
                is_quantized = is_shard_quantized
            # 若后续 shard 与前面基准不一致，说明同一个融合层的不同分片精度不一致，直接报错。
            #
            # 例如 qkv_proj 的 3 个 shard 中：
            #   q_proj -> True
            #   k_proj -> True
            #   v_proj -> False
            # 这种“同一个融合层只有部分 shard 被量化”的情况是不允许的，
            # 因为运行时会把它们视为一个整体融合层来处理。
            elif is_shard_quantized != is_quantized:
                raise ValueError(
                    f"Detected some but not all shards of {prefix} "
                    "are quantized. All shards of fused layers "
                    "to have the same precision."
                )
    else:
        # -------------------- 非融合层直接按完整 prefix 做子串匹配 --------------------
        # 只要 quantized_layers 中任一规则子串命中当前 prefix，就认为该层走 GPTQ 量化。
        #
        # 例如：
        #   prefix = "model.layers.0.self_attn.q_proj"
        #   quantized_layers = ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"]
        #
        # 因为 "self_attn.q_proj" 是 prefix 的子串，所以结果为 True。
        #
        # 反过来如果：
        #   prefix = "model.layers.0.self_attn.o_proj"
        # 则 quantized_layers 里没有任何一项能命中它，结果就是 False。
        is_quantized = any(layer in prefix for layer in quantized_layers)

    # 经过上面的两条路径后，最终结果必须已经被明确成 True/False。
    assert is_quantized is not None
    # 返回该层是否应按 GPTQ 量化处理。
    return is_quantized


def get_linear_quant_method(
    config: GPTQConfig | GPTQMarlinConfig,
    layer: torch.nn.Module,
    prefix: str,
    linear_method_cls: type,
):
    # -------------------- 先复制一份量化配置，避免原始 config 被按层动态规则原地改写 --------------------
    # 后续 override_config(...) 可能会针对当前层修改 bits/group_size/desc_act 等字段，因此先 deep copy。
    cloned_config = deepcopy(config)
    # 只有当 layer 是 ParallelLMHead 且配置显式允许 lm_head 量化时，才把它视为可量化对象。
    parallel_lm_head_quantized = (
        isinstance(layer, ParallelLMHead) and cloned_config.lm_head_quantized
    )
    # -------------------- 只为 LinearBase 家族或允许量化的 ParallelLMHead 选择线性量化方法 --------------------
    if isinstance(layer, LinearBase) or parallel_lm_head_quantized:
        # 先根据 checkpoint 中的 modules_in_block_to_quantize 规则，判断当前层是否属于 GPTQ 量化层。
        is_layer_quantized = is_layer_gptq_quantized(
            prefix=prefix,
            quantized_layers=cloned_config.modules_in_block_to_quantize,
            fused_mapping=cloned_config.packed_modules_mapping,
        )
        # -------------------- 再检查 dynamic override 是否显式把该层排除出量化 --------------------
        # get_dynamic_override(...) 的返回语义是：
        # - False: 命中负匹配，显式跳过量化
        # - None: 没有额外 override
        # - 其它值/字典: 命中正匹配，后面可能覆盖 bits/group_size 等配置
        if get_dynamic_override(
            cloned_config,
            layer_name=prefix,
        ) == False or (not is_layer_quantized):  # noqa: E712
            # 当前 122B-A10B-GPTQ-Int4 中，这里最常见的“退回非量化”情形是：
            # - prefix 包含 self_attn / attn -> attention QKV/O 走 BF16
            # - prefix 包含 shared_expert    -> shared expert gate_up/down 走 BF16
            #
            # 也就是说，当前模型里真正的大头 GPTQ 权重并不主要来自 LinearBase，
            # 而主要来自 routed experts 的 FusedMoE 分支。
            if parallel_lm_head_quantized:
                # ParallelLMHead 若配置允许量化但当前层被排除，则退回非量化 embedding 方法。
                return UnquantizedEmbeddingMethod()
            # 普通线性层则退回非量化线性方法。
            return UnquantizedLinearMethod()

        # -------------------- 若 prefix 非空，则应用按层动态 override 规则 --------------------
        if prefix:
            # 例如某些层可能把 bits/group_size/desc_act/sym 等字段按模块名改写。
            override_config(cloned_config, prefix=prefix)

        # 当前若真的走到这里，说明：
        # - 该层在 checkpoint 中被标记为量化层
        # - 且没有命中 dynamic 的负匹配排除
        # 然后才会构造 GPTQ / GPTQ-Marlin 线性方法。
        # 用“当前层专属”的 cloned_config 实例化对应的线性量化方法类。
        return linear_method_cls(cloned_config)
    # -------------------- 非线性层且也不是允许量化的 ParallelLMHead，则不提供线性量化方法 --------------------
    return None
