# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 The vLLM team.
# Copyright 2025 The Qwen Team.
# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""与 HuggingFace 权重兼容的 Qwen3.5 系列仅推理实现。"""

import typing
from collections.abc import Callable, Iterable

import torch
from einops import rearrange
from torch import nn

from cfie.compilation.decorators import support_torch_compile
from cfie.config.cfie import CfieConfig
from cfie.distributed import get_pp_group
from cfie.logger import init_logger
from cfie.model_executor.layers.layernorm import GemmaRMSNorm as Qwen3_5RMSNorm
from cfie.model_executor.layers.linear import MergedColumnParallelLinear
from cfie.model_executor.layers.logits_processor import LogitsProcessor
from cfie.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from cfie.model_executor.layers.quantization import QuantizationConfig
from cfie.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from cfie.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from cfie.multimodal import MULTIMODAL_REGISTRY
from cfie.sequence import IntermediateTensors
from cfie.transformers_utils.configs.qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig
from cfie.transformers_utils.configs.qwen3_5_moe import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
)

from .interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    MultiModalEmbeddings,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
    _require_is_multimodal,
)
from .qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP
from .qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextDecoderLayer,
    Qwen3NextGatedDeltaNet,
    Qwen3NextModel,
    Qwen3NextSparseMoeBlock,
    QwenNextMixtureOfExperts,
)
from .qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    _merge_multimodal_embeddings,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


class Qwen3_5ProcessingInfo(Qwen3VLProcessingInfo):
    """从运行时上下文解析 Qwen3.5 Dense 多模态 HF 配置。"""

    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen3_5Config)


class Qwen3_5MoeProcessingInfo(Qwen3VLProcessingInfo):
    """从运行时上下文解析 Qwen3.5 MoE 多模态 HF 配置。"""

    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen3_5MoeConfig)


class Qwen3_5GatedDeltaNet(Qwen3NextGatedDeltaNet):
    """Qwen3.5 专用投影布局的 GatedDeltaNet 模块。"""

    def fix_query_key_value_ordering(
            self,
            mixed_qkvz: torch.Tensor,
            mixed_ba: torch.Tensor,
    ):
        # Qwen3.5 checkpoint 中的 Q/K/V 排列已符合后端预期顺序。
        raise NotImplementedError(
            "Qwen3.5 Series dont need to fix query key value ordering"
        )

    def create_qkvz_proj(
            self,
            hidden_size: int,
            key_dim: int,
            value_dim: int,
            quant_config: QuantizationConfig | None,
            prefix: str,
    ) -> MergedColumnParallelLinear:
        # 融合投影输出顺序为 [Q, K, V, Z]。
        return MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[key_dim, key_dim, value_dim, value_dim],
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
        )

    def create_ba_proj(
            self,
            hidden_size: int,
            num_v_heads: int,
            quant_config: QuantizationConfig | None,
            prefix: str,
    ) -> MergedColumnParallelLinear:
        # Qwen3.5 在 checkpoint 中将 in_proj_b 与 in_proj_a 分开存储，
        # 加载时会通过 stacked_params_mapping（shard_id 为 0/1）
        # 合并写入融合参数 in_proj_ba。
        return MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[num_v_heads] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            output: torch.Tensor,
    ):
        """
        前向过程分为三部分：
        1. 输入投影
        2. 核心注意力（自定义算子）
        3. 输出投影
        """
        num_tokens = hidden_states.size(0)

        # ============================================================
        # 第一部分：输入投影
        # ============================================================
        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
        z_size = self.value_dim // self.tp_size
        # 从融合张量中拆出 [Q, K, V] 与门控张量 Z，供注意力后归一化使用。
        mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
        z = z.reshape(z.size(0), -1, self.head_v_dim)
        ba, _ = self.in_proj_ba(hidden_states)
        # B/A 是按头划分的门控向量，供融合 GDN 内核使用。
        b, a = ba.chunk(2, dim=-1)

        b = b.contiguous()
        a = a.contiguous()

        # ============================================================
        # 第二部分：核心注意力（自定义算子）
        # ============================================================
        # 注意：此处不能像其他注意力后端那样使用 torch.empty，
        # 详见 https://github.com/vllm-project/vllm/pull/28182 的讨论。
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        torch.ops.cfie.gdn_attention_core(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            self.prefix,
        )

        # ============================================================
        # 第三部分：输出投影
        # ============================================================
        z_shape_og = z.shape
        # 将输入重排为二维张量后再做归一化与线性映射。
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        output[:num_tokens], _ = self.out_proj(core_attn_out)


class Qwen3_5DecoderLayer(Qwen3NextDecoderLayer):
    """Qwen3.5 的单层 Transformer 模块，支持 Dense/MoE 与混合注意力。"""

    def build_mlp(
            self,
            *,
            cfie_config: CfieConfig,
            config: Qwen3_5TextConfig | Qwen3_5MoeTextConfig,
            quant_config: QuantizationConfig | None,
            prefix: str,
    ) -> nn.Module:
        if config.model_type == "qwen3_5_moe_text":
            return Qwen3NextSparseMoeBlock(
                cfie_config=cfie_config,
                prefix=prefix,
            )
        if config.model_type == "qwen3_5_text":
            return Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=prefix,
            )
        raise ValueError(f"Invalid model_type {config.model_type}")

    def __init__(
            self,
            cfie_config: CfieConfig,
            layer_type: str,
            prefix: str = "",
    ) -> None:
        super(Qwen3NextDecoderLayer, self).__init__()

        config = cfie_config.model_config.hf_text_config
        model_config = cfie_config.model_config
        cache_config = cfie_config.cache_config
        quant_config = cfie_config.quant_config
        speculative_config = cfie_config.speculative_config

        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)

        # 当前 122B-A10B checkpoint 的 text_config.layer_types 共有 48 层，
        # 模式是 3 层 "linear_attention" + 1 层 "full_attention" 循环；
        # 例如 layer 0/1/2 走 linear_attention，layer 3 走 full_attention。
        # 因此这里实际走哪条分支，取决于
        # cfie_config.model_config.hf_text_config.layer_types[self.layer_idx]。
        # Qwen3.5 可按层混用 linear-attention 与 full-attention。
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                speculative_config=speculative_config,
                prefix=f"{prefix}.linear_attn",
            )
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")

        # 根据 model_type 决定前馈模块类型：
        # Qwen3.5 Dense 使用普通 MLP；Qwen3.5-MoE 使用稀疏 MoE 模块。
        # 当前 122B-A10B checkpoint 的 text_config.model_type =
        # "qwen3_5_moe_text"，因此会走下面的 MoE 分支，而不是 dense MLP。
        self.mlp = self.build_mlp(
            cfie_config=cfie_config,
            config=config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

        self.input_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_scale = getattr(config, "layer_scale", False)
        if self.layer_scale:
            self.attn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                ),
            )
            self.ffn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                ),
            )


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # 若启用 mrope（qwen2-vl 默认），positions 形状为 (3, seq_len)；
        # 否则为 (seq_len,)。
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Qwen3_5Model(Qwen3NextModel):
    # Qwen3.5 的语言模型主干（backbone）
    # 只包含：
    # - token embedding
    # - 多层 decoder layer
    # - 最后的 norm
    #
    # 不包含：
    # - lm_head
    #
    # 同时支持流水线并行（PP）

    """支持流水线并行的 Qwen3.5 语言主干（不含 LM Head）。"""

    def build_decoder_layer(
            self,
            *,
            cfie_config: CfieConfig,
            layer_type: str,
            prefix: str,
    ) -> Qwen3_5DecoderLayer:
        return Qwen3_5DecoderLayer(
            cfie_config,
            layer_type=layer_type,
            prefix=prefix,
        )

    def __init__(self, *, cfie_config: CfieConfig, prefix: str = ""):
        # 注意：这里没有调用 Qwen3NextModel.__init__()
        # 而是直接跳过到更上层的 nn.Module.__init__()
        # 说明当前类基本上是“重写整个初始化流程”
        super(Qwen3NextModel, self).__init__()

        # 取出文本配置
        # 这里可能是：
        # - Qwen3_5TextConfig（dense 版）
        # - Qwen3_5MoeTextConfig（MoE 版）
        config: Qwen3_5TextConfig | Qwen3_5MoeTextConfig = (
            cfie_config.model_config.hf_text_config
        )

        # 并行配置
        parallel_config = cfie_config.parallel_config

        # expert parallel / 负载均衡配置
        eplb_config = parallel_config.eplb_config

        # 冗余专家数量（MoE 场景下有意义）
        self.num_redundant_experts = eplb_config.num_redundant_experts

        # 保存配置对象
        self.config = config

        # 词表大小
        self.vocab_size = config.vocab_size

        # 并行词嵌入层
        # 将 token id 映射成 hidden_size 维 embedding
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        def get_layer(prefix: str):
            # 根据 HF 配置中的 layer_types，
            # 为每一层选择对应的注意力类型
            #
            # 例如某层可能是：
            # - full_attention
            # - linear_attention
            #
            # 当前 122B-A10B 的 layer_types 为固定 hybrid 模式：
            # 每隔 4 层出现 1 个 full_attention，其余层都是 linear_attention。
            # extract_layer_index(prefix) 用于从层名前缀中解析层号
            return self.build_decoder_layer(
                cfie_config=cfie_config,
                layer_type=config.layer_types[extract_layer_index(prefix)],
                prefix=prefix,
            )

        # 构造所有 decoder layers
        #
        # make_layers 会根据总层数和 get_layer 工厂函数，
        # 创建出当前 PP rank 负责的层范围：
        # - start_layer: 当前 rank 起始层号
        # - end_layer:   当前 rank 结束层号
        # - layers:      层列表
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )

        # 创建“空中间张量构造器”
        # 主要用于流水线并行时，在不同 stage 之间传递：
        # - hidden_states
        # - residual
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        # 只有最后一个 PP rank 才真正持有最终 norm 层
        if get_pp_group().is_last_rank:
            self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            # 其他 rank 用占位层代替
            self.norm = PPMissingLayer()

        # 需要额外导出的辅助 hidden state 层列表
        # 默认为空，后续可由外部设置
        self.aux_hidden_state_layers: tuple[int, ...] = ()

    def load_fused_expert_weights(
            self,
            name: str,
            params_dict: dict,
            loaded_weight: torch.Tensor,
            shard_id: str,
            num_experts: int,
    ) -> bool:
        # 为当前 rank 的每个本地专家分片加载融合专家权重。
        param = params_dict[name]
        weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
        loaded_local_expert = False
        for expert_id in range(num_experts):
            curr_expert_weight = loaded_weight[expert_id]
            success = weight_loader(
                param,
                curr_expert_weight,
                name,
                shard_id,
                expert_id,
                return_success=True,
            )
            if success:
                loaded_local_expert = True

        return loaded_local_expert

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # -----------------
        # 入口：Qwen3.5 语言主干的手工参数映射入口。
        # -----------------
        # 这个方法负责把 HF checkpoint 的原始命名映射到 CFIE/vLLM 内部的融合参数。
        # 当前主要覆盖四类映射：
        # - q/k/v -> qkv_proj
        # - gate/up -> gate_up_proj
        # - GDN 的 qkvz / ba 融合
        # - MoE routed experts / fused experts 权重
        #
        # 当前本地这份 Qwen3.5-122B-A10B-GPTQ-Int4 checkpoint 的关键上下文：
        # - text_config.model_type = "qwen3_5_moe_text"
        # - text_config.hidden_size = 3072
        # - text_config.num_experts = 256
        # - text_config.num_experts_per_tok = 8
        # - quantization_config.quant_method = "gptq"
        # - bits = 4, group_size = 128, desc_act = false, sym = true
        # - dynamic 里显式排除了 attn.* / shared_expert.* / mtp.*
        #
        # 因而当前大体会分成三条加载路径：
        # 1. attention / linear-attn / GDN 融合层，多数是 BF16 非量化路径
        # 2. shared_expert.gate_up_proj / down_proj，多数也是 BF16 非量化路径
        # 3. routed experts 的 GPTQ Int4 参数，这是下面的主要 MoE 量化加载路径

        # -----------------
        # 初始化：普通融合层映射表。
        # -----------------
        stacked_params_mapping = [
            # (内部参数名, checkpoint 子名, 分片标识)
            # 自注意力 q/k/v -> qkv_proj
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            # 前馈层 gate/up -> gate_up_proj
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            # GDN 融合投影
            ("in_proj_qkvz", "in_proj_qkv", (0, 1, 2)),
            ("in_proj_qkvz", "in_proj_z", 3),
            ("in_proj_ba", "in_proj_b", 0),
            ("in_proj_ba", "in_proj_a", 1),
        ]

        # -----------------
        # 初始化：参数索引与专家映射。
        # -----------------
        # 把当前语言主干里已注册的参数拍平成字典，便于后面按名字直接定位。
        params_dict = dict(self.named_parameters())

        # 记录本次真正从 checkpoint 成功初始化过的参数名，最后返回给上层做对账。
        loaded_params: set[str] = set()

        # routed experts 默认先按“非融合专家 checkpoint”路径处理；
        # 若后面发现 checkpoint 实际使用 fused expert 格式，再切换到融合映射表。
        is_fused_expert = False

        # 非融合专家 checkpoint 的典型映射示例：
        # - ("experts.w13_", "experts.37.gate_proj.", 37, "w1")
        # - ("experts.w13_", "experts.37.up_proj.",   37, "w3")
        # - ("experts.w2_",  "experts.37.down_proj.", 37, "w2")
        expert_params_mapping = self.get_expert_mapping() # 当前路径

        # 融合专家 checkpoint 的专用映射表。
        fused_expert_params_mapping = [
            ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
            ("experts.w2_weight", "experts.down_proj", 0, "w2"),
        ]

        # 当前 text_config.num_experts = 256。
        # 这里读取的是“逻辑专家总数”，不是某层当前设备上真实分配的 local expert slots。
        # local slots / tiered cache 的裁剪发生在 FusedMoE 层内部创建和权重加载阶段。
        num_experts = (
            self.config.num_experts if hasattr(self.config, "num_experts") else 0
        )

        # -----------------
        # 主循环：顺序消费 checkpoint 中的每个权重张量。
        # -----------------
        for name, loaded_weight in weights:
            # rotary_emb.inv_freq 通常不需要从 checkpoint 显式恢复。
            if "rotary_emb.inv_freq" in name:
                continue

            # MTP 参数由 qwen3_5_mtp.py 单独负责，这里显式跳过。
            if name.startswith("mtp."):
                continue

            # kv-scale 名字在少数量化格式下需要重映射；
            # 当前 GPTQ Marlin 路径通常不会命中，这里主要保留兼容。
            if name.endswith("scale"):
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

            # -----------------
            # 第一层分发：普通融合层参数。
            # -----------------
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if "experts.gate_up_proj" in name or "experts.down_proj" in name:
                    # 一旦看到这种命名，说明 checkpoint 的 experts 采用融合存储格式；
                    # 后面的专家加载映射要切换到 fused_expert_params_mapping。
                    is_fused_expert = True # 当前步骤该路径
                    expert_params_mapping = fused_expert_params_mapping

                if weight_name not in name:
                    continue

                if "mlp.experts" in name:
                    # 一旦进入 mlp.experts.*，说明这是 routed experts；
                    # 不能被普通 gate_up_proj / down_proj 映射提前吃掉，
                    # 必须留给后面的 MoE 专家加载逻辑处理。
                    continue

                # 把 checkpoint 子名改写成当前模型内部的融合参数名。
                name = name.replace(weight_name, param_name)

                # GPTQ checkpoint 常会携带额外 bias，但当前模块未必注册了对应参数；
                # 这里按 GPTQ 路径直接跳过。
                if name.endswith(".bias") and name not in params_dict:
                    continue

                # 跳过位于其他设备上的层。
                if is_pp_missing_parameter(name, self):
                    continue

                # 若改写后的名字不在当前模型参数表中，说明它不属于这条普通融合层路径。
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader

                # 这里真正把外部 shard 名交给具体参数自己的 weight_loader：
                # - qkv_proj 使用 q/k/v
                # - gate_up_proj 使用 0/1
                # - in_proj_qkvz / in_proj_ba 使用 GDN 的融合分片标识
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # -----------------
                # 第二层分发：MoE expert 参数。
                # -----------------
                # 这一层才是当前 122B-A10B-GPTQ-Int4 的主量化权重路径：
                # - experts.gate_up_proj.* -> experts.w13_weight.*
                # - experts.down_proj.*    -> experts.w2_weight.*
                # 这里承接 routed experts 的 qweight / scales / qzeros / g_idx 等参数。
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    is_expert_weight = True

                    # 先把 checkpoint 专家名改写成模型内部参数名。
                    name_mapped = name.replace(weight_name, param_name)

                    # 跳过位于其他设备上的层。
                    if is_pp_missing_parameter(name_mapped, self):
                        continue

                    if is_fused_expert:
                        if "experts.gate_up_proj" in name:
                            # 融合专家 checkpoint 会把 gate_up 的 [w1 | w3] 合在一起，
                            # 这里要先拆开，再分别按 w1 / w3 写入。
                            loaded_weight = loaded_weight.chunk(2, dim=-2)
                            success_w1 = self.load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight[0],
                                "w1",
                                num_experts,
                            )
                            success_w3 = self.load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight[1],
                                "w3",
                                num_experts,
                            )
                            success = success_w1 and success_w3
                        else:
                            # down_proj 融合路径不需要再拆分，直接按当前 shard_id 装载。
                            success = self.load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight,
                                shard_id,
                                num_experts,
                            )
                        if success:
                            name = name_mapped
                            break
                    else:
                        # 非融合专家路径会直接调用 FusedMoE.weight_loader(...)。
                        # 它会继续根据 shard_id / expert_id / quant_method，
                        # 把 qweight / scales / qzeros / g_idx 写入对应 expert slot。
                        if (
                                name_mapped.endswith(".bias")
                                or name_mapped.endswith("_bias")
                        ) and name_mapped not in params_dict:
                            continue

                        param = params_dict[name_mapped]
                        weight_loader = param.weight_loader

                        success = weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                    if success:
                        name = name_mapped
                        break
                else:
                    # -----------------
                    # 第三层分发：默认参数加载兜底。
                    # -----------------
                    # 既不是普通融合层，也不是专家参数时，退回默认 weight_loader。
                    # 常见包括 norm、embedding、router gate 等普通参数。
                    if is_expert_weight:
                        # 该参数已经确认是专家权重，但未映射到本地 rank，因此直接跳过。
                        continue
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict:
                        logger.warning_once(
                            f"Parameter {name} not found in params_dict, skip loading"
                        )
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            # 只有真正命中某条加载路径后，才把最终参数名记入 loaded_params。
            loaded_params.add(name)
        return loaded_params


class Qwen3_5ForCausalLMBase(
    nn.Module,
    HasInnerState,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
):
    # 纯文本版 Qwen3.5 CausalLM 的公共基类
    #
    # 这个类主要把几部分组装起来：
    # 1. Qwen3_5Model：语言模型主干（backbone）
    # 2. lm_head：把 hidden_states 投到词表维度
    # 3. logits_processor：统一处理 logits 输出
    #
    # 同时它还声明支持：
    # - HasInnerState：内部状态接口
    # - SupportsEagle3：支持 Eagle3 相关辅助 hidden state
    # - SupportsLoRA：支持 LoRA
    # - SupportsPP：支持流水线并行（PP）

    """纯文本 Qwen3.5 CausalLM 封装：主干 + LM Head + logits 计算。"""

    # 告诉权重加载器：
    # vLLM/cfie 内部哪些融合参数，对应 HF checkpoint 中哪些原始参数名
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["gate_proj", "up_proj"],
        # GDN 模块 fused projections
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    def build_backbone_model(
            self,
            *,
            cfie_config: CfieConfig,
            prefix: str,
    ) -> Qwen3_5Model:
        return Qwen3_5Model(cfie_config=cfie_config, prefix=prefix)

    def __init__(self, *, cfie_config: CfieConfig, prefix: str = ""):
        # 先取出文本配置。
        # 对当前 122B-A10B 模型，这里实际是 Qwen3_5MoeTextConfig。
        config = cfie_config.model_config.hf_text_config

        # 保存完整 cfie_config，后面很多运行时行为都可能回看它。
        self.cfie_config = cfie_config

        # 单独缓存 model_config，便于访问 HF 配置、dtype、架构信息等。
        self.model_config = cfie_config.model_config

        # 取出 cache 配置，后面要据此判断 mamba cache 模式是否合法。
        cache_config = cfie_config.cache_config

        # 取出调度器配置，后面挂到实例上。
        scheduler_config = cfie_config.scheduler_config

        # Qwen3.5 当前不支持 mamba_cache_mode = "all"
        # 只支持 align 模式
        # 当前启动命令未显式传 --mamba-cache-mode，CacheConfig 默认值是 "none"，
        # 所以实际不会进入这个报错分支；只有显式设成 "all" 才会在这里拒绝。
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen3.5 currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )

        # 保存量化配置。
        # 当前命令传了 --quantization gptq_marlin，因此这里不是 None。
        self.quant_config = cfie_config.quant_config

        # 初始化 nn.Module 以及 mixin 父类。
        super().__init__()

        # 把文本配置保存到实例上。
        self.config = config

        # 把调度器配置也保存到实例上。
        self.scheduler_config = scheduler_config

        # 构造语言模型主干：
        # embedding + 多层 decoder + norm
        # 但不包含最终 lm_head
        self.model = self.build_backbone_model(
            cfie_config=cfie_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        # 只有在 PP 的最后一个 rank 上，才真正构造 lm_head
        if get_pp_group().is_last_rank:
            # 当前 checkpoint 配置 tie_word_embeddings = false；
            # 在默认单 PP rank 启动下，is_last_rank=True，因此这里会创建独立 lm_head，
            # 而不是复用 embed_tokens。
            # 如果配置要求词嵌入和输出头共享权重
            if config.tie_word_embeddings:
                # 直接复用输入 embedding 权重
                self.lm_head = self.model.embed_tokens
            else:
                # 否则单独构造一个并行 LM Head
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            # 如果不是最后一个 PP rank，就放一个占位层
            self.lm_head = PPMissingLayer()

        # logits 处理器负责把 hidden_states + lm_head 变成最终词表 logits。
        self.logits_processor = LogitsProcessor(config.vocab_size)

        # 复用主干提供的“空中间张量构造器”
        # 主要给流水线并行等场景使用
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        # 调用主干模型的 token embedding
        return self.model.embed_input_ids(input_ids)

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        # 设置哪些层要额外导出 auxiliary hidden states
        self.model.aux_hidden_state_layers = layers

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        # 返回 Eagle3 默认使用的辅助 hidden state 层索引
        # 这里取：
        # - 第 2 层
        # - 中间层
        # - 倒数第 3 层
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    def forward(
            self,
            input_ids: torch.Tensor,  # 输入 token ids
            positions: torch.Tensor,  # 位置 ids
            intermediate_tensors: IntermediateTensors | None = None,  # PP 中间张量
            inputs_embeds: torch.Tensor | None = None,  # 预先算好的 embedding
            **kwargs: object,
    ):
        # 前向只跑语言模型主干，得到 hidden_states
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

        # 返回 hidden_states
        return hidden_states

    def compute_logits(
            self,
            hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        # 用 lm_head + logits_processor 计算最终词表 logits
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # 自动权重加载器
        loader = AutoWeightsLoader(
            self,
            # 当前启动命令启用了 MTP 草稿模型：
            # --spec-method mtp --num-speculative-tokens 1。
            # 因此主模型只加载主干权重，mtp.* 交给 Qwen3_5MoeMTP。
            skip_prefixes=["mtp."],
        )

        # 按当前模型结构自动加载权重
        return loader.load_weights(weights)


class Qwen3_5ForCausalLM(Qwen3_5ForCausalLMBase):
    pass


class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLMBase, QwenNextMixtureOfExperts):
    # Qwen3.5 的文本 MoE 版 CausalLM
    #
    # 继承两部分能力：
    # 1. Qwen3_5ForCausalLMBase
    #    提供文本模型主体能力：backbone + lm_head + logits 计算
    # 2. QwenNextMixtureOfExperts
    #    提供 MoE 相关的元信息管理与辅助方法

    def __init__(self, *, cfie_config: CfieConfig, prefix: str = ""):
        # 先调用父类初始化
        # 这里主要会完成：
        # - 配置读取
        # - 构造 self.model（语言 backbone）
        # - 构造 lm_head
        # - 构造 logits_processor
        super().__init__(cfie_config=cfie_config, prefix=prefix)

        # 设置 MoE 超参数
        # 这一步会扫描模型各层，找出哪些层是 MoE 层，
        # 并整理出：
        # - num_moe_layers
        # - num_logical_experts
        # - num_physical_experts
        # - num_local_physical_experts
        # - num_routed_experts
        # - num_redundant_experts
        self.set_moe_parameters()

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # 返回 routed experts 的权重映射关系。
        #
        # 这层只是一个很薄的转发：
        # Qwen3_5MoeForCausalLM
        # -> self.model（Qwen3_5Model / Qwen3NextModel）
        # -> Qwen3NextModel.get_expert_mapping()
        # -> SharedFusedMoE/FusedMoE.make_expert_params_mapping(...)
        #
        # 返回的每个 tuple 一般形如：
        # (内部参数名前缀, checkpoint专家权重名前缀, 物理expert_id, shard_id)
        return self.model.get_expert_mapping()


########################################################
# Qwen3_5-Dense（稠密版本）
########################################################


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,  # 注册该模型对应的多模态处理器类
    info=Qwen3_5ProcessingInfo,  # 注册处理时使用的配置解析信息类
    dummy_inputs=Qwen3VLDummyInputsBuilder,  # 注册构造假输入/占位输入的构造器
)
class Qwen3_5ForConditionalGeneration(Qwen3VLForConditionalGeneration, IsHybrid):
    """Qwen3.5-VL 封装：视觉塔 + Qwen3.5 语言模型。"""

    # Qwen3.5 不支持多模态剪枝（EVS）。
    supports_multimodal_pruning = False

    # 告诉权重加载器：哪些模块是“打包/融合参数”
    # 左边是 vLLM 内部参数名，右边是 HF/ckpt 中可能对应的原始权重名列表
    packed_modules_mapping = Qwen3VLForConditionalGeneration.packed_modules_mapping | {
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],  # QKV 与 Z 在 vLLM 中做了融合
        "in_proj_ba": ["in_proj_b", "in_proj_a"],  # B/A 在 vLLM 中做了融合
    }

    def build_language_model(
            self,
            *,
            cfie_config: CfieConfig,
            prefix: str,
    ) -> Qwen3_5ForCausalLM:
        return Qwen3_5ForCausalLM(
            cfie_config=cfie_config,
            prefix=prefix,
        )

    def __init__(self, *, cfie_config: CfieConfig, prefix: str = "model"):
        # 协议类没有 __init__，因此需要显式调用 nn.Module.__init__。
        nn.Module.__init__(self)

        # 取出当前模型的 HF 配置对象，类型是 Qwen3.5-VL 的完整配置
        config: Qwen3_5Config = cfie_config.model_config.hf_config

        # 量化配置，后续视觉塔/语言模型初始化时会用到
        quant_config = cfie_config.quant_config

        # 多模态配置，里面包含 mm encoder 的并行方式等信息
        multimodal_config = cfie_config.model_config.multimodal_config

        # 保存完整模型配置到实例
        self.config = config

        # 保存多模态相关配置到实例
        self.multimodal_config = multimodal_config

        # 如果 mm_encoder_tp_mode == "data"，表示视觉编码器走数据并行模式
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"

        # Qwen3.5 不支持多模态剪枝（EVS），因此这里显式关闭
        self.is_multimodal_pruning_enabled = False

        # 标记下面创建的是“塔模型”（tower model），这里是视觉塔
        # {"image", "video"} 表示这个塔负责图像/视频模态
        with self._mark_tower_model(cfie_config, {"image", "video"}):
            # 构造视觉编码器（Vision Transformer）
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,  # 视觉塔专用配置
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),  # 归一化的 eps
                quant_config=quant_config,  # 量化配置
                prefix=maybe_prefix(prefix, "visual"),  # 当前模块在参数树中的前缀名
            )

        # 标记下面创建的是“语言模型”部分
        with self._mark_language_model(cfie_config):
            # 构造 Qwen3.5 文本生成模型（内部包含 language backbone + lm head）
            self.language_model = self.build_language_model(
                cfie_config=cfie_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )

        # 暴露一个生成空中间张量的工厂，供流水线并行等场景使用
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def embed_input_ids(
            self,
            input_ids: torch.Tensor,  # 文本 token id 序列
            multimodal_embeddings: MultiModalEmbeddings | None = None,  # 视觉/视频等模态编码后的 embedding
            *,
            is_multimodal: torch.Tensor | None = None,  # 哪些位置是多模态占位位点的布尔掩码
    ) -> torch.Tensor:
        # 先把纯文本 input_ids 做成文本 embedding
        # self._embed_text_input_ids 是父类提供的帮助函数
        # self.language_model.embed_input_ids 是底层文本 embedding 函数
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.language_model.embed_input_ids,
            is_multimodal=is_multimodal,
        )

        # 如果没有多模态 embedding，或者长度为 0，直接返回纯文本 embedding
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        # 确保 is_multimodal 一定存在，否则无法知道要把多模态 embedding 填到哪些位置
        is_multimodal = _require_is_multimodal(is_multimodal)

        # 把图像/视频等模态 embedding 按占位位置合并进 inputs_embeds
        inputs_embeds = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,  # 原始文本 embedding
            multimodal_embeddings=multimodal_embeddings,  # 多模态 embedding
            is_multimodal=is_multimodal,  # 多模态占位掩码
        )

        # 返回融合后的统一输入 embedding
        return inputs_embeds

    def recompute_mrope_positions(self, *args, **kwargs):
        # Qwen3.5 不支持多模态剪枝，因此不应调用这个函数
        raise NotImplementedError(
            "Qwen3.5 does not support multimodal pruning (EVS). "
            "recompute_mrope_positions should never be called."
        )

    def forward(
            self,
            input_ids: torch.Tensor,  # 批次拼接后的 input_ids
            positions: torch.Tensor,  # 对应的 position ids；可能是 (seq_len,) 或 (3, seq_len)
            intermediate_tensors: IntermediateTensors | None = None,  # PP 上一阶段传来的中间张量
            inputs_embeds: torch.Tensor | None = None,  # 已预先构造好的 embedding
            **kwargs: object,  # 额外多模态参数，如 pixel_values / video_grid_thw 等
    ) -> torch.Tensor | IntermediateTensors:
        """执行 Qwen3.5 的前向计算。

        Args:
            input_ids: 对应批次的展平（拼接）input_ids。

            positions: 对应批次的展平（拼接）position ids。

                **注意**：若启用 mrope（Qwen3VL 开源模型默认配置），
                其形状为 `(3, seq_len)`，
                否则为 `(seq_len,)`。
            intermediate_tensors: 来自前一流水线阶段的中间张量。

            inputs_embeds: 预先计算好的输入 embedding。
            **kwargs: 额外关键字参数，包括：
                - pixel_values: 输入模型的图像像素值。
                    若未传入图像则为 `None`。
                - image_grid_thw: LLM 侧图像 3D 网格张量 `(n_images, 3)`。
                    若未传入图像则为 `None`。
                - pixel_values_videos: 输入模型的视频像素值。
                    若未传入视频则为 `None`。
                - video_grid_thw: LLM 侧视频 3D 网格张量 `(n_videos, 3)`。
                    若未传入视频则为 `None`。
        """

        # 如果 intermediate_tensors 不为空，说明当前是 PP（流水线并行）的中间阶段
        if intermediate_tensors is not None:
            # 在 PP 中间阶段，embedding 不能本地重算，必须使用前一阶段传来的结果
            # 因此这里强制置空 inputs_embeds，后面由 language_model.model 自己处理
            inputs_embeds = None

        # 注意：这里直接调用的是 language_model.model
        # 也就是语言 backbone，而不是整个 language_model 的 logits 头部
        # 因为这个 forward 主要返回 hidden_states，后续 logits 可能在别处再算
        hidden_states = self.language_model.model(
            input_ids=input_ids,  # 文本 token ids
            positions=positions,  # 位置编码输入
            intermediate_tensors=intermediate_tensors,  # PP 中间张量
            inputs_embeds=inputs_embeds,  # 预计算 embedding（如果有）
        )

        # 返回语言模型 backbone 的输出 hidden states
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # 这是当前多模态主模型的最外层权重加载入口。
        # DefaultModelLoader.load_weights() 会先调用到这里。
        # 这里本身不手工逐个处理张量，而是：
        # 1. 建立 AutoWeightsLoader
        # 2. 跳过 mtp.* 这种不属于主模型的权重
        # 3. 把其余权重继续分发给 visual / language_model 等子模块
        # 使用 AutoWeightsLoader 自动加载权重
        loader = AutoWeightsLoader(
            self,
            # 当前主模型是 Qwen3_5MoeForConditionalGeneration，
            # 同时启动命令启用了 --spec-method mtp --num-speculative-tokens 1；
            # 因此外层多模态模型这里也只加载主模型权重，mtp.* 继续交给
            # Qwen3_5MoeMTP 单独处理。
            skip_prefixes=["mtp."],
        )

        # 优先使用 CFIE 侧映射名，兼容尚未完全迁移的旧字段名。
        mapper = getattr(self, "hf_to_cfie_mapper", None)
        if mapper is None:
            mapper = getattr(self, "hf_to_vllm_mapper", None)
        # 真正执行自动分发加载。
        # 进入 language_model 子树后，会继续调用更深一层的 load_weights。
        return loader.load_weights(weights, mapper=mapper)

    @classmethod
    def get_mamba_state_dtype_from_config(
            cls,
            cfie_config: "CfieConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        # 返回 GatedDeltaNet 循环状态与 SSM 缓存的状态 dtype。
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            cfie_config.model_config.dtype,  # 模型主 dtype
            cfie_config.cache_config.mamba_cache_dtype,  # mamba cache dtype
            cfie_config.cache_config.mamba_ssm_cache_dtype,  # mamba ssm cache dtype
        )

    @classmethod
    def get_mamba_state_shape_from_config(
            cls, cfie_config: "CfieConfig"
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        # 并行配置，主要拿 tensor parallel size
        parallel_config = cfie_config.parallel_config

        # 文本配置，里面有 linear attention / GDN 所需的 head 数和维度
        hf_config = cfie_config.model_config.hf_text_config

        # TP 大小
        tp_size = parallel_config.tensor_parallel_size

        # speculative decoding 的 token 数，如果没开 speculative 就是 0
        num_spec = (
            cfie_config.speculative_config.num_speculative_tokens
            if cfie_config.speculative_config
            else 0
        )

        # 计算 GatedDeltaNet 相关状态张量的形状
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,  # tensor parallel size
            hf_config.linear_num_key_heads,  # 线性注意力 key 头数
            hf_config.linear_num_value_heads,  # 线性注意力 value 头数
            hf_config.linear_key_head_dim,  # key head 维度
            hf_config.linear_value_head_dim,  # value head 维度
            hf_config.linear_conv_kernel_dim,  # 卷积核维度
            num_spec,  # speculative token 数
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        # 返回 GatedDeltaNet 状态复制函数，用于 cache/state 的搬运与更新
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()


########################################################
# Qwen3_5-MoE（专家版本）
########################################################


class Qwen3_5_MoeMixtureOfExperts(MixtureOfExperts):
    # Qwen3.5 的 MoE 元数据辅助类
    # 主要负责两件事：
    # 1. 从语言模型中扫描并提取所有 MoE 层的公共超参数
    # 2. 当运行时物理专家布局发生变化时，把新元数据同步到每个 MoE 层

    """Qwen3.5 稀疏专家层的运行时 MoE 元数据适配器。"""

    def update_physical_experts_metadata(
            self,
            num_physical_experts: int,  # 全局物理专家总数
            num_local_physical_experts: int,  # 当前设备/当前 rank 上本地物理专家数
    ) -> None:
        # 将动态专家布局元数据下发到每个稀疏 MoE 模块。

        # 一致性检查：当前对象记录的本地物理专家数应与传入值一致
        assert self.num_local_physical_experts == num_local_physical_experts

        # 更新模型级别的物理专家总数
        self.num_physical_experts = num_physical_experts

        # 更新模型级别的本地物理专家数
        self.num_local_physical_experts = num_local_physical_experts

        # 冗余专家数 = 物理专家数 - 逻辑专家数
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts

        # 遍历语言模型主干中的所有层
        for layer in self.language_model.model.layers:
            # 只处理 MLP 是稀疏 MoE block 的层
            if isinstance(layer.mlp, Qwen3NextSparseMoeBlock):
                moe = layer.mlp

                # 把最新的本地物理专家数同步到该层
                moe.n_local_physical_experts = num_local_physical_experts

                # 把最新的全局物理专家数同步到该层
                moe.n_physical_experts = num_physical_experts

                # 把最新的冗余专家数同步到该层
                moe.n_redundant_experts = self.num_redundant_experts

                # 重新构建该层 experts 内部的专家映射表
                # 即更新“逻辑专家 -> 物理专家”的映射关系
                moe.experts.update_expert_map()

    def set_moe_parameters(self):
        # 初始化专家权重容器
        # 后续可能用于保存专家参数引用或相关信息
        self.expert_weights = []

        # 用来保存所有 MoE 层的 experts 容器
        self.moe_layers = []

        # 取一个示例 MoE 层，用来读取公共超参数
        example_moe = None

        # 遍历语言模型主干中的所有层
        for layer in self.language_model.model.layers:
            # 只挑出：
            # 1. 层本身是 Qwen3_5DecoderLayer
            # 2. 且其 mlp 是 Qwen3NextSparseMoeBlock
            if isinstance(layer, Qwen3_5DecoderLayer) and isinstance(
                    layer.mlp, Qwen3NextSparseMoeBlock
            ):
                # 记录当前 MoE block 作为示例
                example_moe = layer.mlp

                # 收集该层真正的 experts 容器
                self.moe_layers.append(layer.mlp.experts)

        # 如果一个 MoE 层都没找到，说明当前模型并不是有效的 Qwen3.5-MoE 结构
        if example_moe is None:
            raise RuntimeError(
                "No Qwen3_5 layer found in the language_model.model.layers."
            )

        # 设置模型级别的 MoE 超参数

        # MoE 层总数
        self.num_moe_layers = len(self.moe_layers)

        # 专家组数，这里固定为 1
        self.num_expert_groups = 1

        # 共享专家数，这里固定为 0
        self.num_shared_experts = 0

        # 逻辑专家总数：模型定义上的专家数
        self.num_logical_experts = example_moe.n_logical_experts

        # 物理专家总数：实际部署的专家实例数
        self.num_physical_experts = example_moe.n_physical_experts

        # 当前设备 / 当前 rank 上本地拥有的物理专家数
        self.num_local_physical_experts = example_moe.n_local_physical_experts

        # 路由时可被选中的专家总数
        self.num_routed_experts = example_moe.n_routed_experts

        # 冗余专家数：物理专家数 - 逻辑专家数
        self.num_redundant_experts = example_moe.n_redundant_experts


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5MoeProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3_5MoeForConditionalGeneration(
    Qwen3_5ForConditionalGeneration, Qwen3_5_MoeMixtureOfExperts
):
    # Qwen3.5 多模态 + MoE 版本模型类
    # 继承两部分能力：
    # 1. Qwen3_5ForConditionalGeneration：提供多模态模型主体框架（视觉塔 + 语言模型）
    # 2. Qwen3_5_MoeMixtureOfExperts：提供 MoE 相关元信息与辅助方法

    def build_language_model(
            self,
            *,
            cfie_config: CfieConfig,
            prefix: str,
    ) -> Qwen3_5MoeForCausalLM:
        return Qwen3_5MoeForCausalLM(
            cfie_config=cfie_config,
            prefix=prefix,
        )

    def __init__(self, *, cfie_config: CfieConfig, prefix: str = "model"):
        # 协议类/某些 mixin 没有可靠的 __init__ 链，
        # 因此这里直接显式初始化 nn.Module
        nn.Module.__init__(self)

        # 当前 122B-A10B-GPTQ-Int4 checkpoint 的顶层配置值为：
        # - architectures = ["Qwen3_5MoeForConditionalGeneration"]
        # - model_type = "qwen3_5_moe"
        # 因此真实初始化入口会落到这个“多模态 + MoE”类，而不是 dense 版本。
        # 取出当前模型的 HF 配置，这里是 Qwen3.5 MoE 多模态配置
        config: Qwen3_5MoeConfig = cfie_config.model_config.hf_config

        # 量化配置，后续初始化视觉塔和语言模型时都会显式传入。
        quant_config = cfie_config.quant_config

        # 多模态配置，包含视觉编码器并行方式、预算等设置。
        multimodal_config = cfie_config.model_config.multimodal_config

        # 把完整 HF 配置保存到实例上。
        self.config = config

        # 把多模态配置保存到实例上。
        self.multimodal_config = multimodal_config

        # 判断视觉编码器是否使用 data parallel 模式
        # mm_encoder_tp_mode == "data" 表示视觉塔采用数据并行
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"

        # Qwen3.5 不支持多模态剪枝（EVS），因此显式关闭
        self.is_multimodal_pruning_enabled = False

        # 标记接下来构造的是“塔模型”中的视觉塔部分。
        # 这样 CFIE 的上下文管理器能够知道当前是在创建视觉分支。
        # {"image", "video"} 表示该塔负责图像和视频模态
        with self._mark_tower_model(cfie_config, {"image", "video"}):
            # 创建视觉编码器（Vision Transformer）
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,  # 视觉塔配置
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),  # 归一化 eps
                quant_config=quant_config,  # 量化配置
                prefix=maybe_prefix(prefix, "visual"),  # 参数名前缀，例如 model.visual
            )

        # 标记接下来构造的是语言模型部分。
        # 这样后续参数前缀、上下文状态、某些注册逻辑都能区分“语言分支”。
        with self._mark_language_model(cfie_config):
            # 创建 Qwen3.5 的 MoE 文本生成模型
            # 与 Dense 版不同，这里挂载的是 Qwen3_5MoeForCausalLM
            self.language_model = self.build_language_model(
                cfie_config=cfie_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )

        # 复用语言模型提供的“空中间张量构造器”。
        # 主要给流水线并行（PP）等场景使用
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        # 最后汇总并设置 MoE 相关超参数和元信息。
        # 例如：
        # - 有多少个 MoE 层
        # - logical / physical experts 数量
        # - local experts 数量
        # - routed experts / redundant experts 数量
        self.set_moe_parameters()
