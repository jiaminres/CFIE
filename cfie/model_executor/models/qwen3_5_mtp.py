# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""仅用于推理的 Qwen3_5 MTP 模型实现。"""

import typing
from collections.abc import Callable, Iterable

import torch
from torch import nn

from cfie.compilation.decorators import support_torch_compile
from cfie.config import CfieConfig
from cfie.distributed import get_pp_group
from cfie.logger import init_logger
from cfie.model_executor.layers.fused_moe import FusedMoE
from cfie.model_executor.layers.linear import ColumnParallelLinear
from cfie.model_executor.layers.logits_processor import LogitsProcessor
from cfie.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from cfie.model_executor.model_loader.weight_utils import default_weight_loader
from cfie.sequence import IntermediateTensors
from cfie.transformers_utils.configs.qwen3_5 import Qwen3_5TextConfig
from cfie.transformers_utils.configs.qwen3_5_moe import Qwen3_5MoeTextConfig

from .interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    _require_is_multimodal,
)
from .qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5RMSNorm
from .qwen3_next import QwenNextMixtureOfExperts
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    _merge_multimodal_embeddings,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    maybe_prefix,
)

logger = init_logger(__name__)


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # 若启用 mrope（qwen2-vl 默认），positions 形状为 (3, seq_len)；
        # 否则为 (seq_len,)。
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "hidden_states": 0,
    }
)
class Qwen3_5MultiTokenPredictor(nn.Module):
    """Qwen3.5 的多 Token 预测分支，用于投机解码。"""

    def __init__(self, *, cfie_config: CfieConfig, prefix: str = ""):
        super().__init__()

        model_config = cfie_config.model_config
        quant_config = cfie_config.quant_config

        config: Qwen3_5TextConfig | Qwen3_5MoeTextConfig = model_config.hf_text_config

        self.config = config

        self.vocab_size = config.vocab_size

        # 在逻辑上，MTP 层位于主干语言模型层之后。
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = getattr(config, "mtp_num_hidden_layers", 1)
        # 当前 122B-A10B checkpoint 的 mtp_num_hidden_layers = 1，
        # 所以草稿模型只会构造 1 个 MTP decoder layer。

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        self.fc = ColumnParallelLinear(
            self.config.hidden_size * 2,
            self.config.hidden_size,
            gather_output=True,
            bias=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fc",
        )

        # 当前实现中，MTP 分支固定使用 full-attention 解码层。
        # 即使当前主模型是 hybrid（linear/full 混合），草稿层这里也不读取
        # text_config.layer_types，而是统一走 full_attention。
        self.layers = torch.nn.ModuleList(
            Qwen3_5DecoderLayer(
                cfie_config,
                layer_type="full_attention",
                prefix=f"{prefix}.layers.{idx}",
            )
            for idx in range(self.num_mtp_layers)
        )

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is None:
                inputs_embeds = self.embed_input_ids(input_ids)
            assert hidden_states.shape[-1] == inputs_embeds.shape[-1]
            # 先归一化，再将“当前 hidden + 下一 token embedding”拼接成 MTP 输入。
            inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
            hidden_states = self.pre_fc_norm_hidden(hidden_states)
            hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)
            hidden_states = self.fc(hidden_states)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        # 当前 122B-A10B checkpoint 的 mtp_num_hidden_layers = 1，
        # 启动命令又设置了 --num-speculative-tokens 1，
        # 因此 current_step_idx 恒为 0，实际总是复用唯一一层 MTP layer。
        # 每个投机步都会按循环方式选择一个 MTP 层执行。
        current_step_idx = spec_step_idx % self.num_mtp_layers
        hidden_states, residual = self.layers[current_step_idx](
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
        )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_fused_expert_weights(
        self,
        name: str,
        params_dict: dict,
        loaded_weight: torch.Tensor,
        shard_id: str,
        num_experts: int,
    ) -> bool:
        # 对当前 rank 的每个本地专家分片加载一份融合专家权重。
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
        # 将 HF checkpoint 参数名映射到 vLLM 的融合参数命名。
        stacked_params_mapping = [
            # (参数名, 分片名, 分片标识)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # 专家参数映射：覆盖权重、FP8 权重 scale、FP8 激活 scale。
        # (参数名, 权重名, 专家 ID, 分片标识)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts
            if hasattr(self.config, "num_experts")
            else 0,
        )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        is_fused_expert = False
        fused_expert_params_mapping = [
            ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
            ("experts.w2_weight", "experts.down_proj", 0, "w2"),
        ]
        num_experts = (
            self.config.num_experts if hasattr(self.config, "num_experts") else 0
        )
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if "experts.gate_up_proj" in name or "experts.down_proj" in name:
                    # 当前 draft checkpoint 仍然来自同一个 Qwen3.5 MoE 模型，
                    # architectures 会被 speculative.py 改写成 "Qwen3_5MoeMTP"，
                    # 但专家权重本质仍是 qwen3_5_moe 的融合专家格式，因此这里
                    # 继续沿用 fused_expert_params_mapping。
                    # Qwen3.5 MoE 可能将专家 MLP 以融合形式存储。
                    is_fused_expert = True
                    expert_params_mapping = fused_expert_params_mapping

                if weight_name not in name:
                    continue

                if "mlp.experts" in name:
                    continue

                name = name.replace(weight_name, param_name)
                # 当前链路使用 GPTQ Int4 checkpoint + gptq_marlin 后端，
                # 所以这里需要兼容 ckpt 里额外 bias 的跳过逻辑。
                # 对 GPTQ 模型，跳过额外 bias 的加载。
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # 跳过位于其他设备上的层。
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    # 跳过位于其他设备上的层。
                    if is_pp_missing_parameter(name_mapped, self):
                        continue
                    if is_fused_expert:
                        # Qwen3.5 不需要转置。
                        # 历史逻辑：loaded_weight = loaded_weight.transpose(-1, -2)
                        if "experts.gate_up_proj" in name:
                            # gate_up 内部拼接了 w1/w3，需先拆分再分别加载。
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
                            # down_proj 分支
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
                        # 对 GPTQ 模型，跳过额外 bias 的加载。
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
                    if is_expert_weight:
                        # 该权重已确认属于专家参数，但未映射到本地 rank，直接跳过。
                        continue
                    # 对 GPTQ 模型，跳过额外 bias 的加载。
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
            loaded_params.add(name)
        return loaded_params


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # 若启用 mrope（qwen2-vl 默认），positions 形状为 (3, seq_len)；
        # 否则为 (seq_len,)。
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "hidden_states": 0,
    }
)
class Qwen3_5MTP(nn.Module, SupportsMultiModal):
    """MTP 分支服务封装：包含模型本体与可选 tied LM Head。"""

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, cfie_config: CfieConfig, prefix: str = ""):
        # 取出文本配置。
        # 对当前 122B-A10B 草稿模型路径，这里仍然是 Qwen3_5MoeTextConfig。
        config = cfie_config.model_config.hf_text_config
        # 保存 cfie_config，后面访问 speculative_config / cache_config 时会用到。
        self.cfie_config = cfie_config
        # 取出 cache 配置，并检查当前 MTP 路径是否支持这个 cache 模式。
        cache_config = cfie_config.cache_config
        # 当前启动命令未显式传 --mamba-cache-mode，因此这里的实际值是默认 "none"；
        # 只有显式改成 "all" 才会走到下面的不支持分支。
        if cache_config.mamba_cache_mode == "all":
            # 与主 Qwen3.5 模型行为保持一致。
            raise NotImplementedError(
                "Qwen3_5MTP currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )

        # 保存量化配置；当前命令通常对应 gptq_marlin。
        self.quant_config = cfie_config.quant_config

        # 初始化 nn.Module / mixin。
        super().__init__()
        # 把文本配置保存到实例上。
        self.config = config
        draft_model_config = (
            cfie_config.speculative_config.draft_model_config
            if cfie_config.speculative_config is not None
            else None
        )
        # 当前启动命令只传了：
        # - --spec-method mtp
        # - --num-speculative-tokens 1
        # SpeculativeConfig 会据此构造同路径 draft_model_config，
        # 并把 runner 设为 "draft"。
        is_speculative_draft_runner = (
            getattr(draft_model_config, "runner", None) == "draft"
        )
        self.has_own_lm_head = not (
            is_speculative_draft_runner
            and get_pp_group().is_last_rank
            and not config.tie_word_embeddings
        )
        # 当前值组合为：
        # - draft_model_config.runner == "draft"
        # - get_pp_group().is_last_rank == True（默认单 PP rank）
        # - config.tie_word_embeddings == False（checkpoint config）
        # 因此 self.has_own_lm_head=False，下面会让 draft/MTP 模型放置
        # PPMissingLayer，而不是单独再建一个 lm_head。
        self.model = Qwen3_5MultiTokenPredictor(
            cfie_config=cfie_config, prefix=maybe_prefix(prefix, "mtp")
        )

        # 只有最后一个 PP rank 需要决定是否持有 draft 侧 lm_head。
        if get_pp_group().is_last_rank:
            # 若配置要求共享词表权重，就直接复用 embedding。
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            # 若当前 draft runner 不应持有自己的 lm_head，就放占位层。
            elif not self.has_own_lm_head:
                self.lm_head = PPMissingLayer()
            # 否则单独构造 draft lm_head。
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            # 非最后一个 PP rank 不负责输出 logits。
            self.lm_head = PPMissingLayer()

        # 统一的 logits 处理器。
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.model.embed_input_ids,
            is_multimodal=is_multimodal,
        )

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        is_multimodal = _require_is_multimodal(is_multimodal)

        inputs_embeds = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        hidden_states = self.model(
            input_ids, positions, hidden_states, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def remap_weight_names(weights):
            # 将 MTP checkpoint 的参数名重映射到当前封装命名空间。
            # 当前 draft/MTP 模型只消费 mtp.* 以及必要的 embedding/lm_head 权重；
            # 主模型主体参数仍由 Qwen3_5MoeForConditionalGeneration 加载。
            for name, weight in weights:
                # mtp.* 是草稿模型主体权重，把前缀改写到当前模块的 model.* 子树。
                if name.startswith("mtp."):
                    name = name.replace("mtp.", "model.")
                # embed_tokens / lm_head 这类共享或可选权重单独处理。
                elif any(key in name for key in ["embed_tokens", "lm_head"]):
                    # 如果当前 draft 模型不持有自己的 lm_head，就直接跳过该权重。
                    if "lm_head" in name and not self.has_own_lm_head:
                        continue
                    # embed_tokens 在当前封装下不再带 language_model. 前缀，去掉它。
                    if "embed_tokens" in name:
                        name = name.replace("language_model.", "")
                else:
                    # 其余权重不属于 MTP 模型，直接忽略。
                    continue
                yield name, weight

        # AutoWeightsLoader 会把 remap 后的权重继续分发到
        # Qwen3_5MultiTokenPredictor / lm_head 等实际子模块。
        loader = AutoWeightsLoader(self)
        return loader.load_weights(remap_weight_names(weights))


class Qwen3_5MoeMTP(Qwen3_5MTP, QwenNextMixtureOfExperts):
    def __init__(self, *, cfie_config: CfieConfig, prefix: str = ""):
        super().__init__(cfie_config=cfie_config, prefix=prefix)
        self.set_moe_parameters()
