# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rotary Positional Embeddings."""

from typing import Any

import torch

from .base import RotaryEmbedding
from .deepseek_scaling_rope import DeepseekScalingRotaryEmbedding
from .dual_chunk_rope import DualChunkRotaryEmbedding
from .dynamic_ntk_alpha_rope import DynamicNTKAlphaRotaryEmbedding
from .dynamic_ntk_scaling_rope import DynamicNTKScalingRotaryEmbedding
from .fope import FourierRotaryEmbedding
from .linear_scaling_rope import LinearScalingRotaryEmbedding
from .llama3_rope import Llama3RotaryEmbedding
from .llama4_vision_rope import Llama4VisionRotaryEmbedding
from .mrope import MRotaryEmbedding
from .mrope_interleaved import MRotaryEmbeddingInterleaved
from .ntk_scaling_rope import NTKScalingRotaryEmbedding
from .phi3_long_rope_scaled_rope import Phi3LongRoPEScaledRotaryEmbedding
from .xdrope import XDRotaryEmbedding
from .yarn_scaling_rope import YaRNScalingRotaryEmbedding

_ROPE_DICT: dict[tuple[Any, ...], RotaryEmbedding] = {}


def get_rope(
    head_size: int,                                   # 每个 attention head 的总维度
    max_position: int,                                # 支持的最大位置长度
    is_neox_style: bool = True,                       # 是否采用 GPT-NeoX 风格的 RoPE 排布
    rope_parameters: dict[str, Any] | None = None,    # RoPE 的附加配置，如 rope_type / rope_theta / factor 等
    dtype: torch.dtype | None = None,                 # cos/sin cache 使用的数据类型
    dual_chunk_attention_config: dict[str, Any] | None = None,  # Dual Chunk Attention 的附加配置
) -> RotaryEmbedding:
    # 如果没有显式指定 dtype，就用当前默认 dtype
    if dtype is None:
        dtype = torch.get_default_dtype()

    # 为了让配置能够作为缓存 key 使用：
    # 把 rope_parameters 里所有 list 转成 tuple
    if rope_parameters is not None:
        rope_parameters_tuple = {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in rope_parameters.items()
        }
        # 转成 tuple(items) 以便可哈希，用作缓存字典 key
        rope_parameters_args = tuple(rope_parameters_tuple.items())
    else:
        rope_parameters_args = None

    # Dual Chunk Attention 的配置也做同样处理
    # 但跳过 "sparse_attention_config" 这种可能不适合作为缓存 key 的字段
    if dual_chunk_attention_config is not None:
        dual_chunk_attention_tuple = {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in dual_chunk_attention_config.items()
            if k != "sparse_attention_config"
        }
        dual_chunk_attention_args = tuple(dual_chunk_attention_tuple.items())
    else:
        dual_chunk_attention_args = None

    # 若未提供 rope_parameters，则用空字典
    rope_parameters = rope_parameters or {}

    # RoPE 的 base，默认 10000
    base = rope_parameters.get("rope_theta", 10000)

    # RoPE 类型 / 缩放策略，默认 "default"
    scaling_type = rope_parameters.get("rope_type", "default")

    # 只对 head_size 的一部分维度施加 rotary 的比例，默认全部施加
    partial_rotary_factor = rope_parameters.get("partial_rotary_factor", 1.0)

    # 校验部分 rotary 比例范围
    if partial_rotary_factor <= 0.0 or partial_rotary_factor > 1.0:
        raise ValueError(f"{partial_rotary_factor=} must be between 0.0 and 1.0")

    # 实际做 rotary 的维度数
    rotary_dim = int(head_size * partial_rotary_factor)

    # 构造缓存 key：
    # 相同参数下复用已经构造好的 rotary embedding 对象
    key = (
        head_size,                   # 原始 head 维度
        rotary_dim,                  # 实际旋转维度
        max_position,                # 最大位置长度
        is_neox_style,               # NeoX 风格与否
        rope_parameters_args,        # rope 配置
        dual_chunk_attention_args,   # dual chunk 配置
        dtype,                       # 数据类型
    )

    # 若缓存里已有同参数对象，直接复用
    if key in _ROPE_DICT:
        return _ROPE_DICT[key]

    # ------------------------------------------------------------
    # 1. 若启用了 Dual Chunk Attention，则优先构造 DualChunkRotaryEmbedding
    # ------------------------------------------------------------
    if dual_chunk_attention_config is not None:
        # 仅保留 Dual Chunk RotaryEmbedding 构造时需要的参数
        extra_kwargs = {
            k: v
            for k, v in dual_chunk_attention_config.items()
            if k in ("chunk_size", "local_size")
        }

        rotary_emb = DualChunkRotaryEmbedding(
            head_size,         # 每个 head 的总维度
            rotary_dim,        # 实际旋转维度
            max_position,      # 最大位置长度
            base,              # rope theta
            is_neox_style,     # NeoX 风格与否
            dtype,             # 数据类型
            **extra_kwargs,    # chunk_size / local_size
        )

    # ------------------------------------------------------------
    # 2. 默认 RoPE 路径
    # ------------------------------------------------------------
    elif scaling_type == "default":
        # 若启用了 mrope_section，则走多段式 MRotaryEmbedding
        if "mrope_section" in rope_parameters:
            rotary_emb = MRotaryEmbedding(
                head_size,
                rotary_dim,
                max_position,
                base,
                is_neox_style,
                dtype,
                mrope_section=rope_parameters["mrope_section"],   # 各 section 切分方式
                mrope_interleaved=rope_parameters.get("mrope_interleaved", False),  # 是否交错排布
            )

        # 若启用了 Fourier RoPE
        elif "use_fope" in rope_parameters and rope_parameters["use_fope"]:
            # 仅保留 FourierRotaryEmbedding 需要的参数
            extra_kwargs = {
                k: v
                for k, v in rope_parameters.items()
                if k in (
                    "num_key_value_heads",
                    "num_inv_freq",
                    "fope_sep_head",
                    "fope_init_factor",
                )
            }
            # 这里显式禁止初始化 cache
            extra_kwargs["init_cache"] = False

            rotary_emb = FourierRotaryEmbedding(
                head_size,
                rotary_dim,
                max_position,
                base,
                is_neox_style,
                dtype,
                **extra_kwargs,
            )

        # 默认最普通的 RotaryEmbedding
        else:
            rotary_emb = RotaryEmbedding(
                head_size,
                rotary_dim,
                max_position, # 8192
                base, # qwen3.5 10000
                is_neox_style, # qwen3.5 True
                dtype,
            )

    # ------------------------------------------------------------
    # 3. Llama3 的 RoPE 缩放
    # ------------------------------------------------------------
    elif scaling_type == "llama3":
        scaling_factor = rope_parameters["factor"]                           # 缩放倍数
        low_freq_factor = rope_parameters["low_freq_factor"]                 # 低频缩放因子
        high_freq_factor = rope_parameters["high_freq_factor"]               # 高频缩放因子
        original_max_position = rope_parameters["original_max_position_embeddings"]  # 原始训练长度

        rotary_emb = Llama3RotaryEmbedding(
            head_size,
            rotary_dim,
            max_position,
            base,
            is_neox_style,
            dtype,
            scaling_factor,
            low_freq_factor,
            high_freq_factor,
            original_max_position,
        )

    # ------------------------------------------------------------
    # 4. Llama4 Vision 的 RoPE
    # ------------------------------------------------------------
    elif scaling_type == "mllama4":
        rotary_emb = Llama4VisionRotaryEmbedding(
            head_size, rotary_dim, max_position, base, is_neox_style, dtype
        )

    # ------------------------------------------------------------
    # 5. 线性缩放 RoPE
    # ------------------------------------------------------------
    elif scaling_type == "linear":
        scaling_factor = rope_parameters["factor"]

        rotary_emb = LinearScalingRotaryEmbedding(
            head_size,
            rotary_dim,
            max_position,
            base,
            is_neox_style,
            scaling_factor,
            dtype,
        )

    # ------------------------------------------------------------
    # 6. NTK 缩放 RoPE
    # ------------------------------------------------------------
    elif scaling_type == "ntk":
        scaling_factor = rope_parameters["factor"]   # NTK 缩放倍数
        mixed_b = rope_parameters.get("mixed_b")     # 可选混合参数

        rotary_emb = NTKScalingRotaryEmbedding(
            head_size,
            rotary_dim,
            max_position,
            base,
            is_neox_style,
            scaling_factor,
            dtype,
            mixed_b,
        )

    # ------------------------------------------------------------
    # 7. Dynamic NTK 缩放
    # ------------------------------------------------------------
    elif scaling_type == "dynamic":
        # dynamic 模式支持两种参数：
        # 1. alpha
        # 2. factor
        if "alpha" in rope_parameters:
            scaling_alpha = rope_parameters["alpha"]

            rotary_emb = DynamicNTKAlphaRotaryEmbedding(
                head_size,
                rotary_dim,
                max_position,
                base,
                is_neox_style,
                scaling_alpha,
                dtype,
            )

        elif "factor" in rope_parameters:
            scaling_factor = rope_parameters["factor"]

            rotary_emb = DynamicNTKScalingRotaryEmbedding(
                head_size,
                rotary_dim,
                max_position,
                base,
                is_neox_style,
                scaling_factor,
                dtype,
            )
        else:
            raise ValueError(
                "Dynamic rope scaling must contain either 'alpha' or 'factor' field"
            )

    # ------------------------------------------------------------
    # 8. XDRoPE
    # ------------------------------------------------------------
    elif scaling_type == "xdrope":
        scaling_alpha = rope_parameters["alpha"]

        rotary_emb = XDRotaryEmbedding(
            head_size,
            rotary_dim,
            max_position,
            base,
            is_neox_style,
            scaling_alpha,
            dtype,
            xdrope_section=rope_parameters["xdrope_section"],  # xdrope 的 section 划分
        )

    # ------------------------------------------------------------
    # 9. YaRN 缩放
    # ------------------------------------------------------------
    elif scaling_type == "yarn":
        scaling_factor = rope_parameters["factor"]                           # 缩放因子
        original_max_position = rope_parameters["original_max_position_embeddings"]  # 原始训练长度

        # 只保留 YaRN 相关参数
        extra_kwargs = {
            k: v
            for k, v in rope_parameters.items()
            if k in (
                "extrapolation_factor",
                "attn_factor",
                "beta_fast",
                "beta_slow",
                "apply_yarn_scaling",
                "truncate",
            )
        }

        # 若是 mrope + yarn 组合，则仍走 MRotaryEmbedding
        if "mrope_section" in rope_parameters:
            # MRotaryEmbedding 路径不需要 apply_yarn_scaling
            extra_kwargs.pop("apply_yarn_scaling", None)

            rotary_emb = MRotaryEmbedding(
                head_size,
                rotary_dim,
                original_max_position,
                base,
                is_neox_style,
                dtype,
                mrope_section=rope_parameters["mrope_section"],
                mrope_interleaved=rope_parameters.get("mrope_interleaved", False),
                scaling_factor=scaling_factor,
                **extra_kwargs,
            )
        else:
            rotary_emb = YaRNScalingRotaryEmbedding(
                head_size,
                rotary_dim,
                original_max_position,
                base,
                is_neox_style,
                scaling_factor,
                dtype,
                **extra_kwargs,
            )

    # ------------------------------------------------------------
    # 10. DeepSeek 风格缩放
    # ------------------------------------------------------------
    elif scaling_type in ["deepseek_yarn", "deepseek_llama_scaling"]:
        scaling_factor = rope_parameters["factor"]
        original_max_position = rope_parameters["original_max_position_embeddings"]

        # DeepSeek 特有额外参数
        extra_kwargs = {
            k: v
            for k, v in rope_parameters.items()
            if k in (
                "extrapolation_factor",
                "attn_factor",
                "beta_fast",
                "beta_slow",
                "mscale",
                "mscale_all_dim",
            )
        }

        rotary_emb = DeepseekScalingRotaryEmbedding(
            head_size,
            rotary_dim,
            original_max_position,
            base,
            is_neox_style,
            scaling_factor,
            dtype,
            **extra_kwargs,
        )

    # ------------------------------------------------------------
    # 11. LongRoPE（Phi-3 等）
    # ------------------------------------------------------------
    elif scaling_type == "longrope":
        short_factor = rope_parameters["short_factor"]                       # 短上下文缩放因子
        long_factor = rope_parameters["long_factor"]                         # 长上下文缩放因子
        original_max_position = rope_parameters["original_max_position_embeddings"]

        # LongRoPE 额外的 mscale 参数
        extra_kwargs = {
            k: v
            for k, v in rope_parameters.items()
            if k in ("short_mscale", "long_mscale")
        }

        rotary_emb = Phi3LongRoPEScaledRotaryEmbedding(
            head_size,
            rotary_dim,
            max_position,
            original_max_position,
            base,
            is_neox_style,
            dtype,
            short_factor,
            long_factor,
            **extra_kwargs,
        )

    # ------------------------------------------------------------
    # 12. OpenPangu 的 mRoPE 变体
    # ------------------------------------------------------------
    elif scaling_type == "openpangu":
        mrope_interleaved = rope_parameters.get("mrope_interleaved", False)

        # 必须同时具备 mrope_section 且要求 interleaved
        if "mrope_section" in rope_parameters and mrope_interleaved:
            rotary_emb = MRotaryEmbeddingInterleaved(
                head_size,
                rotary_dim,
                max_position,
                base,
                is_neox_style,
                dtype,
                mrope_section=rope_parameters["mrope_section"],
                mrope_interleaved=mrope_interleaved,
            )
        else:
            raise ValueError("Pangu mrope lacks necessary parameters.")

    # ------------------------------------------------------------
    # 13. 未知 RoPE 类型，报错
    # ------------------------------------------------------------
    else:
        raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    # 构造完成后缓存起来
    _ROPE_DICT[key] = rotary_emb

    return rotary_emb
