# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Qwen3.5-MoE model configuration"""

from transformers.configuration_utils import PretrainedConfig, layer_type_validation


class Qwen3_5MoeTextConfig(PretrainedConfig):
    # 当前配置类对应的模型类型名
    # Transformers / vLLM 会用它识别这是哪一类模型配置
    model_type = "qwen3_5_moe_text"

    # 推理阶段可以忽略的键
    # past_key_values 常见于缓存相关返回值，这里声明推理时可忽略
    keys_to_ignore_at_inference = ["past_key_values"]

    # Tensor Parallel（张量并行）切分规划
    # key: 模块路径模式
    # value: 该模块权重应采用的 TP 切分方式
    base_model_tp_plan = {
        # self-attention 中的 Q/K/V 投影按“列切分”
        # 一般对应输出维度方向切分
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",

        # self-attention 输出投影按“行切分”
        # 一般对应输入维度方向切分
        "layers.*.self_attn.o_proj": "rowwise",

        # MoE 专家中的 gate_up_proj 是融合参数（通常对应 gate_proj + up_proj）
        # packed_colwise 表示这是打包后的列切分
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",

        # MoE 专家中的 down_proj 按行切分
        "layers.*.mlp.experts.down_proj": "rowwise",

        # shared expert（共享专家）部分的 gate/up/down 投影切分方式
        "layers.*.mlp.shared_expert.gate_proj": "colwise",
        "layers.*.mlp.shared_expert.up_proj": "colwise",
        "layers.*.mlp.shared_expert.down_proj": "rowwise",
    }

    # Pipeline Parallel（流水线并行）规划
    # 描述各模块在 PP 里的“输入张量名 -> 输出张量名”
    base_model_pp_plan = {
        # embedding 层：输入 input_ids，输出 inputs_embeds
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),

        # decoder layers：输入 hidden_states / attention_mask，输出 hidden_states
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),

        # final norm：输入 hidden_states，输出 hidden_states
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    # 如果这个 text config 被更大配置对象包裹，
    # 它对应的大配置里的字段名叫 text_config
    base_config_key = "text_config"

    def __init__(
        self,
        vocab_size=248320,                     # 词表大小：模型支持的 token 数量
        hidden_size=2048,                     # 隐藏维度：每个 token hidden state 的维数
        num_hidden_layers=40,                 # Transformer/Decoder 总层数
        num_attention_heads=16,               # full attention 的查询头数
        num_key_value_heads=2,                # full attention 的 KV 头数（GQA/MQA 场景下可小于查询头数）
        hidden_act="silu",                    # 前馈网络/MLP 使用的激活函数
        max_position_embeddings=32768,        # 最大位置长度：支持的最大上下文长度
        initializer_range=0.02,               # 参数初始化时的标准范围
        rms_norm_eps=1e-6,                    # RMSNorm 的 eps，防止除零、提升数值稳定性
        use_cache=True,                       # 是否启用缓存（推理时加速常用）
        tie_word_embeddings=False,            # 输入 embedding 和输出 lm_head 是否共享权重
        rope_parameters=None,                 # Rotary Position Embedding（RoPE）的详细配置
        attention_bias=False,                 # 注意力相关线性层是否带 bias
        attention_dropout=0.0,                # 注意力 dropout 比例（训练用，推理一般无效）
        head_dim=256,                         # full attention 每个头的维度

        # ---------- linear attention / GDN 相关 ----------
        linear_conv_kernel_dim=4,             # 线性注意力/GDN 中使用的卷积核长度
        linear_key_head_dim=128,              # linear attention 里 key 头的维度
        linear_value_head_dim=128,            # linear attention 里 value 头的维度
        linear_num_key_heads=16,              # linear attention 里 key 头数
        linear_num_value_heads=32,            # linear attention 里 value 头数

        # ---------- MoE 相关 ----------
        moe_intermediate_size=512,            # routed experts（路由专家）内部 MLP 的中间层维度
        shared_expert_intermediate_size=512,  # shared expert（共享专家）内部 MLP 的中间层维度
        num_experts_per_tok=8,                # 每个 token 会被路由到多少个专家（top-k 专家数）
        num_experts=256,                      # 专家总数
        output_router_logits=False,           # 是否输出路由器 logits（训练分析/辅助损失时可能会用）
        router_aux_loss_coef=0.001,           # 路由辅助损失系数，用于鼓励专家负载均衡

        # ---------- 每层类型 ----------
        layer_types=None,                     # 每层的类型列表；元素通常是 "full_attention" 或 "linear_attention"

        # ---------- 特殊 token ----------
        pad_token_id=None,                    # padding token id
        bos_token_id=None,                    # 句子开始 token id（BOS）
        eos_token_id=None,                    # 句子结束 token id（EOS）
        **kwargs,
    ):
        # RoPE 参数校验时忽略的字段
        # 这些字段若存在，不参与这里的 RoPE 合法性检查
        kwargs["ignore_keys_at_rope_validation"] = [
            "mrope_section",
            "mrope_interleaved",
        ]

        # ===== 基础文本模型结构参数 =====
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim
        self.rope_parameters = rope_parameters

        # 给 partial_rotary_factor 设置默认值
        # 表示 RoPE 只作用到每个 head 的一部分维度
        # 这里默认只作用 25% 的维度
        kwargs.setdefault("partial_rotary_factor", 0.25)

        # ===== 每层类型配置 =====
        self.layer_types = layer_types

        # 如果没有显式提供 layer_types，
        # 则根据 full_attention_interval 自动生成
        if self.layer_types is None:
            # 例如 interval_pattern=4 表示“每隔 4 层放一层 full attention”
            interval_pattern = kwargs.get("full_attention_interval", 4)

            # 生成每层类型列表
            # 规则：
            # - 层号+1 能被 interval_pattern 整除 -> full_attention
            # - 否则 -> linear_attention
            self.layer_types = [
                "linear_attention"
                if bool((i + 1) % interval_pattern)
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

        # 校验 layer_types 是否合法：
        # - 长度必须等于 num_hidden_layers
        # - 元素值必须是允许的类型
        layer_type_validation(self.layer_types, self.num_hidden_layers)

        # ===== linear attention / GDN 部分 =====
        # 卷积核大小
        self.linear_conv_kernel_dim = linear_conv_kernel_dim

        # key/value 头维度
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim

        # key/value 头数
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads

        # ===== MoE 部分 =====
        # 路由专家的中间层维度
        self.moe_intermediate_size = moe_intermediate_size

        # 共享专家的中间层维度
        self.shared_expert_intermediate_size = shared_expert_intermediate_size

        # 每个 token 选几个专家
        self.num_experts_per_tok = num_experts_per_tok

        # 总专家数
        self.num_experts = num_experts

        # 是否输出 router logits
        self.output_router_logits = output_router_logits

        # 路由辅助损失系数
        self.router_aux_loss_coef = router_aux_loss_coef

        # 调用父类 PretrainedConfig 初始化
        # 会处理 kwargs 中的通用配置字段
        super().__init__(**kwargs)

        # 下面这些字段放在 super().__init__() 后面再设置，
        # 因为 transformers 的 PretrainedConfig.__init__ 里对它们有默认值，
        # 若提前设，可能被父类默认值覆盖掉
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings


class Qwen3_5MoeVisionConfig(PretrainedConfig):
    model_type = "qwen3_5_moe"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth=27,
        hidden_size=1152,
        hidden_act="gelu_pytorch_tanh",
        intermediate_size=4304,
        num_heads=16,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=3584,
        num_position_embeddings=2304,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.num_position_embeddings = num_position_embeddings
        self.initializer_range = initializer_range


class Qwen3_5MoeConfig(PretrainedConfig):
    model_type = "qwen3_5_moe"
    sub_configs = {
        "vision_config": Qwen3_5MoeVisionConfig,
        "text_config": Qwen3_5MoeTextConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=248056,
        video_token_id=248057,
        vision_start_token_id=248053,
        vision_end_token_id=248054,
        tie_word_embeddings=False,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"]()

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        super().__init__(**kwargs)
        # Set after super().__init__() to avoid v4 PretrainedConfig overwrite
        self.tie_word_embeddings = tie_word_embeddings


__all__ = ["Qwen3_5MoeConfig", "Qwen3_5MoeTextConfig"]
