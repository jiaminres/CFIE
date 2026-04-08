# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# 中文注释：https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/qwen2_moe/modeling_qwen2_moe.py
# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
"""中文说明：与 HuggingFace 权重兼容的 Qwen2MoE 仅推理实现。"""

from collections.abc import Iterable
from itertools import islice
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen2MoeConfig

from cfie.compilation.decorators import support_torch_compile
from cfie.config import CacheConfig, CfieConfig
from cfie.distributed import get_pp_group, get_tensor_model_parallel_world_size
from cfie.logger import init_logger
from cfie.model_executor.layers.activation import SiluAndMul
from cfie.model_executor.layers.attention import Attention
from cfie.model_executor.layers.fused_moe import SharedFusedMoE
from cfie.model_executor.layers.layernorm import RMSNorm
from cfie.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from cfie.model_executor.layers.logits_processor import LogitsProcessor
from cfie.model_executor.layers.quantization import QuantizationConfig
from cfie.model_executor.layers.rotary_embedding import get_rope
from cfie.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from cfie.model_executor.model_loader.weight_utils import default_weight_loader
from cfie.sequence import IntermediateTensors

from .interfaces import SupportsLoRA, SupportsPP
from .utils import (
    AutoWeightsLoader,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


class Qwen2MoeMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        expert_gate: torch.nn.Linear | None = None,
        prefix: str = "",
    ) -> None:
        # -------------------- 先初始化 nn.Module 基类状态 --------------------
        # 注册子模块、参数容器等 nn.Module 基础设施。
        super().__init__()

        # -------------------- 构造 gate/up 融合投影 --------------------
        # Qwen2MoE 的 MLP 前半段使用融合版列并行线性层，
        # 一次性同时产出 gate_proj 和 up_proj 两路输出。
        self.gate_up_proj = MergedColumnParallelLinear(
            # 输入维度等于 Transformer hidden size。
            hidden_size,
            # 输出侧拆成两段，每段大小都是 intermediate_size：
            # - 一段给 gate 分支
            # - 一段给 up 分支
            [intermediate_size] * 2,
            # 当前实现中 MLP 投影不带 bias。
            bias=False,
            # 若启用量化，则把量化配置继续传给线性层。
            quant_config=quant_config,
            # 为该权重命名成 `<prefix>.gate_up_proj`，便于加载 checkpoint。
            prefix=f"{prefix}.gate_up_proj",
        )

        # -------------------- 构造 down 投影 --------------------
        # MLP 后半段使用行并行线性层，把 intermediate 维映射回 hidden 维。
        self.down_proj = RowParallelLinear(
            # 输入是激活后的 intermediate 向量。
            intermediate_size,
            # 输出恢复到 hidden size。
            hidden_size,
            # down_proj 同样不带 bias。
            bias=False,
            # 继续沿用外部传入的量化配置。
            quant_config=quant_config,
            # 是否在 TP 维上对 down_proj 输出做 reduce，由调用方决定。
            reduce_results=reduce_results,
            # 为 down_proj 权重设置加载前缀。
            prefix=f"{prefix}.down_proj",
        )

        # -------------------- 校验并构造激活函数 --------------------
        # 当前 Qwen2MoE MLP 只支持 silu 门控激活形式。
        if hidden_act != "silu":
            # 若配置给出其他激活函数，直接拒绝，避免与实现不匹配。
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        # `SiluAndMul` 会把融合输出拆成两半，并执行 SiLU(gate) * up。
        self.act_fn = SiluAndMul()

        # -------------------- 保存可选的 expert gate --------------------
        # shared expert 场景下，这里可能额外挂一个门控线性层，
        # 用来对最终 expert 输出再乘一个 sigmoid 门控系数。
        self.expert_gate = expert_gate

    def forward(self, x):
        # -------------------- 先计算 gate/up 两路融合投影 --------------------
        # `MergedColumnParallelLinear` 返回 `(output, bias)`；
        # 这里当前实现不需要单独使用 bias，因此只取第一个返回值。
        gate_up, _ = self.gate_up_proj(x)

        # -------------------- 执行 SiLU 门控激活 --------------------
        # `SiluAndMul` 会把 gate_up 沿最后一维均分成两半：
        # - 一半经过 SiLU 作为 gate
        # - 另一半作为 up 分支
        # 然后两者逐元素相乘。
        out = self.act_fn(gate_up)

        # -------------------- 经过 down 投影映射回 hidden 维 --------------------
        # `RowParallelLinear` 同样返回 `(output, bias)`；
        # 这里只保留真正的输出张量。
        out, _ = self.down_proj(out)

        # -------------------- 若配置了 expert gate，则再做一次输出门控 --------------------
        # shared expert 路径下，Qwen2MoE 会额外学习一个标量门，
        # 用 sigmoid(expert_gate(x)) 对整条 expert 输出做缩放。
        if self.expert_gate is not None:
            # `expert_gate(x)` 返回 `(logits, bias)`；
            # 这里只取 logits，经 sigmoid 后作为 [0, 1] 区间的门控系数。
            out = F.sigmoid(self.expert_gate(x)[0]) * out

        # -------------------- 返回最终 MLP 输出 --------------------
        return out


class Qwen2MoeSparseMoeBlock(nn.Module):
    def __init__(
        self,
        config: Qwen2MoeConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()

        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )

        self.shared_expert_gate = ReplicatedLinear(
            config.hidden_size,
            1,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.shared_expert_gate",
        )

        if config.shared_expert_intermediate_size > 0:
            self.shared_expert = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                expert_gate=self.shared_expert_gate,
                prefix=f"{prefix}.shared_expert",
            )
        else:
            self.shared_expert = None

        self.experts = SharedFusedMoE(
            shared_experts=self.shared_expert,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 中文注释：NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)

        # 中文注释：router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )
        if self.shared_expert is not None:
            final_hidden_states = final_hidden_states[0] + final_hidden_states[1]
        if self.tp_size > 1:
            final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(  # noqa E501
                final_hidden_states
            )

        return final_hidden_states.view(orig_shape)


class Qwen2MoeAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict[str, Any] | None = None,
        max_position_embeddings: int = 8192,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        dual_chunk_attention_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # 中文注释：Number of KV heads is greater than TP size, so we partition
            # 中文注释：the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # 中文注释：Number of KV heads is less than TP size, so we replicate
            # 中文注释：the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings
        self.dual_chunk_attention_config = dual_chunk_attention_config

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_parameters,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            **{
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config": dual_chunk_attention_config,
            }
            if dual_chunk_attention_config
            else {},
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen2MoeDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen2MoeConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.self_attn = Qwen2MoeAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_parameters=config.rope_parameters,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            dual_chunk_attention_config=dual_chunk_attention_config,
        )

        # 中文注释：Note: Qwen/Qwen2-57B-A14B-Instruct does not have
        # 中文注释：`mlp_only_layers` in the config.
        layer_idx = extract_layer_index(prefix)
        mlp_only_layers = (
            [] if not hasattr(config, "mlp_only_layers") else config.mlp_only_layers
        )
        if (layer_idx not in mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen2MoeSparseMoeBlock(
                config=config, quant_config=quant_config, prefix=f"{prefix}.mlp"
            )
        else:
            self.mlp = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> torch.Tensor:
        # 中文注释：Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # 中文注释：Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile
class Qwen2MoeModel(nn.Module):
    def __init__(self, *, cfie_config: CfieConfig, prefix: str = ""):
        super().__init__()

        config = cfie_config.model_config.hf_config
        cache_config = cfie_config.cache_config
        quant_config = cfie_config.quant_config

        self.vocab_size = config.vocab_size
        self.config = config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Qwen2MoeDecoderLayer(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # 中文注释：Params for weights, fp8 weight scales, fp8 activation scales
        # 中文注释：(param_name, weight_name, expert_id, shard_id)
        return SharedFusedMoE.make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # 中文注释：(param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()
        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # 中文注释：Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # 中文注释：We have mlp.experts[0].gate_proj in the checkpoint.
                # 中文注释：Since we handle the experts below in expert_params_mapping,
                # 中文注释：we need to skip here BEFORE we update the name, otherwise
                # 中文注释：name will be updated to mlp.experts[0].gate_up_proj, which
                # 中文注释：will then be updated below in expert_params_mapping
                # 中文注释：for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # 中文注释：Skip loading extra bias for GPTQ models.
                if (
                    name.endswith(".bias") or name.endswith("_bias")
                ) and name not in params_dict:
                    continue
                # 中文注释：Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)

                    # 中文注释：Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # 中文注释：Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias") or name.endswith("_bias")
                    ) and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # 中文注释：Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias") or name.endswith("_bias")
                    ) and name not in params_dict:
                        continue
                    # 中文注释：Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # 中文注释：Remapping the name of FP8 kv-scale.
                    if name.endswith("kv_scale"):
                        remapped_kv_scale_name = name.replace(
                            ".kv_scale", ".attn.kv_scale"
                        )
                        if remapped_kv_scale_name not in params_dict:
                            logger.warning_once(
                                "Found kv_scale in the checkpoint (e.g. %s), but not found the expected name in the model (e.g. %s). kv_scale is not loaded.",  #  noqa: E501
                                name,
                                remapped_kv_scale_name,
                            )
                            continue
                        else:
                            name = remapped_kv_scale_name
                    # 中文注释：GGUF: make sure that shared_expert_gate is a 2D tensor.
                    if (
                        "mlp.shared_expert_gate" in name
                        and len(loaded_weight.shape) == 1
                    ):
                        loaded_weight = loaded_weight[None, :]
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Qwen2MoeForCausalLM(nn.Module, SupportsPP, SupportsLoRA):
    fall_back_to_pt_during_load = False
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ]
    }

    def __init__(self, *, cfie_config: CfieConfig, prefix: str = ""):
        super().__init__()
        config = cfie_config.model_config.hf_config
        quant_config = cfie_config.quant_config
        self.config = config
        self.quant_config = quant_config
        # 中文注释：Only perform the following mapping when Qwen2MoeMLP exists
        if (
            getattr(config, "mlp_only_layers", [])
            or config.shared_expert_intermediate_size > 0
        ):
            self.packed_modules_mapping["gate_up_proj"] = ["gate_proj", "up_proj"]

        self.model = Qwen2MoeModel(
            cfie_config=cfie_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
