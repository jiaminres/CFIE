# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""中文说明：仅用于推理的 Qwen3Next 模型实现。"""

from collections.abc import Iterable
from itertools import islice

import torch
import cfie._custom_ops as ops
from einops import rearrange
from torch import nn
from transformers.activations import ACT2FN

from cfie import envs
from cfie.compilation.decorators import support_torch_compile
from cfie.config import (
    CacheConfig,
    ModelConfig,
    SpeculativeConfig,
    CfieConfig,
    get_current_cfie_config,
)
from cfie.distributed import (
    divide,
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from cfie.forward_context import ForwardContext, get_forward_context
from cfie.logger import init_logger
from cfie.model_executor.custom_op import CustomOp
from cfie.model_executor.layers.attention import Attention
from cfie.model_executor.layers.fla.ops import (
    chunk_gated_delta_rule as fla_chunk_gated_delta_rule,
)
from cfie.model_executor.layers.fla.ops import (
    fused_recurrent_gated_delta_rule_packed_decode,
    fused_sigmoid_gating_delta_rule_update,
)
from cfie.model_executor.layers.fla.ops.chunk import l2norm_fwd
from cfie.model_executor.layers.fused_moe import SharedFusedMoE
from cfie.model_executor.layers.layernorm import (
    GemmaRMSNorm as Qwen3NextRMSNorm,
)
from cfie.model_executor.layers.layernorm import RMSNormGated
from cfie.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from cfie.model_executor.layers.logits_processor import LogitsProcessor
from cfie.model_executor.layers.mamba.abstract import MambaBase
from cfie.model_executor.layers.mamba.mamba_mixer2 import mamba_v2_sharded_weight_loader
from cfie.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from cfie.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from cfie.model_executor.layers.quantization import QuantizationConfig
from cfie.model_executor.layers.rotary_embedding import get_rope
from cfie.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from cfie.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
    sharded_weight_loader,
)
from cfie.model_executor.models.qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP
from cfie.model_executor.models.utils import sequence_parallel_chunk
from cfie.model_executor.utils import set_weight_attrs
from cfie.triton_utils.allocation import set_triton_allocator
from cfie.platforms import current_platform
from cfie.sequence import IntermediateTensors
from cfie.transformers_utils.configs import Qwen3NextConfig
from cfie.triton_utils import HAS_TRITON, tl, triton
from cfie.utils.torch_utils import direct_register_custom_op
from cfie.v1.attention.backend import AttentionMetadata
from cfie.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from .interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsLoRA,
    SupportsPP,
)
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)

KVCache = tuple[torch.Tensor, torch.Tensor]


def _try_precompiled_fused_gdn_gating(
        A_log: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        dt_bias: torch.Tensor,
        beta: float = 1.0,
        threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not a.is_cuda:
        return None
    if not ops.has_precompiled_fused_gdn_gating():
        return None
    return ops.fused_gdn_gating_precompiled(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        beta=beta,
        threshold=threshold,
    )


def fi_chunk_gated_delta_rule(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.LongTensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
):
    from flashinfer.gdn_prefill import (
        chunk_gated_delta_rule as chunk_gated_delta_rule_fi,
    )

    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)

    # 中文注释：use flashinfer implementation
    q = q.squeeze(0).contiguous()
    k = k.squeeze(0).contiguous()
    v = v.squeeze(0).contiguous()

    g = g.squeeze(0).contiguous()
    beta = beta.squeeze(0).contiguous()
    fi_state = initial_state.to(torch.float32)
    fi_g = g.to(torch.float32)
    fi_beta = beta.to(torch.float32)
    result = chunk_gated_delta_rule_fi(
        q=q,
        k=k,
        v=v,
        g=torch.exp(fi_g),
        beta=fi_beta,
        initial_state=fi_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    # 中文注释：FlashInfer returns (output, state) when output_final_state=True,
    # 中文注释：or just output when output_final_state=False.
    # 中文注释：Unsqueeze back to 4D (1, L, H, D) to match fla output format
    if output_final_state:
        output, final_state = result
        return output.unsqueeze(0), final_state
    else:
        return result.unsqueeze(0), None


@CustomOp.register("chunk_gated_delta_rule")
class ChunkGatedDeltaRule(CustomOp):
    def __init__(self) -> None:
        super().__init__()
        if current_platform.is_cuda() and current_platform.is_device_capability(90):
            logger.info_once(
                "Using FlashInfer GDN prefill kernel on CUDA compute capability 90"
            )
            self._forward_method = self.forward_cuda
        else:
            self._forward_method = self.forward_native

    def forward_cuda(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            g: torch.Tensor,
            beta: torch.Tensor,
            initial_state: torch.Tensor,
            output_final_state: bool,
            cu_seqlens: torch.LongTensor | None = None,
            use_qk_l2norm_in_kernel: bool = True,
    ):
        return fi_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )

    def forward_native(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            g: torch.Tensor,
            beta: torch.Tensor,
            initial_state: torch.Tensor,
            output_final_state: bool,
            cu_seqlens: torch.LongTensor | None = None,
            use_qk_l2norm_in_kernel: bool = True,
    ):
        return fla_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )


class Qwen3NextSparseMoeBlock(nn.Module):
    def __init__(self, cfie_config: CfieConfig, prefix: str = ""):
        # 先初始化 nn.Module 基类，注册参数/子模块容器等内部状态。
        super().__init__()

        # -------------------- 读取模型配置、并行配置与量化配置 --------------------
        # 取出 HuggingFace 文本模型配置，后续 expert 数量、hidden size 等都从这里读取。
        config = cfie_config.model_config.hf_text_config

        # 取出 CFIE 并行配置，决定 TP/EP/sequence parallel 等运行方式。
        parallel_config = cfie_config.parallel_config

        # 取出量化配置，后续共享专家和 FusedMoE 初始化时会透传进去。
        quant_config = cfie_config.quant_config

        # -------------------- 解析当前 TP / EP 拓扑 --------------------
        # 读取当前 tensor parallel world size，用于校验 TP 与 expert 数的兼容性。
        self.tp_size = get_tensor_model_parallel_world_size()

        # Windows 单卡本地执行会走一个不初始化 torch.distributed 的
        # GroupCoordinator 快捷路径，此时 device_group 为空，但组内 rank/world size
        # 仍然可以从 coordinator 本身读取。
        ep_coordinator = get_ep_group()
        self.ep_group = ep_coordinator.device_group

        # 当前 rank 在 EP 组内的组内 rank。
        self.ep_rank = ep_coordinator.rank_in_group

        # 当前 EP 组的 world size，也就是 expert 并行的 rank 数。
        self.ep_size = ep_coordinator.world_size

        # 记录模型定义的 routed experts 总数，即逻辑专家数的基础值。
        self.n_routed_experts = config.num_experts

        # 是否启用 sequence parallel MoE，会影响 forward 阶段是否先切 token。
        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        # -------------------- 校验 TP 与 expert 数量的基本兼容性 --------------------
        # 若 TP 维度比 expert 总数还大，则无法把 experts 合理映射到各个 TP rank 上。
        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        # -------------------- 读取 EPLB 配置并推导逻辑/物理 expert 数量 --------------------
        # 重新从当前上下文读取一份全局 CfieConfig，确保拿到运行时最终生效的配置。
        cfie_config = get_current_cfie_config()

        # 取出 expert parallel load balancing 的细粒度配置对象。
        eplb_config = cfie_config.parallel_config.eplb_config

        # 记录当前这个 MoE block 是否启用了 EPLB。
        self.enable_eplb = parallel_config.enable_eplb

        # 逻辑专家数默认等于 routed experts 总数。
        self.n_logical_experts = self.n_routed_experts

        # 冗余专家数来自 EPLB 配置，用于负载均衡时增加 physical replica。
        self.n_redundant_experts = eplb_config.num_redundant_experts

        # 物理专家总数 = 逻辑专家数 + 冗余专家副本数。
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts

        # 当前 EP rank 本地持有的物理专家数量，要求物理专家总数可被 EP size 整除。
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        # 当前 EP rank 本地 physical expert 切片的起始下标。
        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts

        # 当前 EP rank 本地 physical expert 切片的结束下标（右开区间）。
        self.physical_expert_end = (
                self.physical_expert_start + self.n_local_physical_experts
        )

        # -------------------- 构造 router gate 与 shared expert gate --------------------
        # 主 router gate 负责为 routed experts 产出 expert logits。
        self.gate = ReplicatedLinear(
            # 输入 hidden state 维度。
            config.hidden_size,
            # 输出维度等于专家数，每个专家对应一个 router logit。
            config.num_experts,
            # Qwen3Next 的 gate 不带 bias。
            bias=False,
            # gate 本身不走量化配置。
            quant_config=None,
            # 模块名前缀，用于权重加载与 state dict 对齐。
            prefix=f"{prefix}.gate",
        )

        # shared expert gate 只输出一个标量门控值，控制 shared expert 分支。
        self.shared_expert_gate = ReplicatedLinear(
            # 输入仍是 hidden_size。
            config.hidden_size,
            # shared expert gate 只需要 1 个输出通道。
            1,
            # 同样不带 bias。
            bias=False,
            # shared expert gate 也不走量化配置。
            quant_config=None,
            # shared expert gate 的权重前缀。
            prefix=f"{prefix}.shared_expert_gate",
        )

        # -------------------- 按配置决定是否创建 shared expert MLP --------------------
        # 只有 shared expert intermediate size 大于 0 时，才真正实例化 shared expert。
        if config.shared_expert_intermediate_size > 0:
            # 构造共享专家 MLP，它和 routed experts 的输出会在 forward 末尾相加。
            self.shared_expert = Qwen3NextMLP(
                # 输入 hidden size。
                hidden_size=config.hidden_size,
                # 共享专家的中间层维度。
                intermediate_size=config.shared_expert_intermediate_size,
                # 激活函数类型，例如 silu。
                hidden_act=config.hidden_act,
                # 共享专家允许复用全局量化配置。
                quant_config=quant_config,
                # 共享专家内部不负责做 TP/EP reduce。
                reduce_results=False,
                # 把 shared_expert_gate 作为这个共享专家的外部门控。
                expert_gate=self.shared_expert_gate,
                # shared expert 模块前缀。
                prefix=f"{prefix}.shared_expert",
            )
        else:
            # 若配置关闭 shared expert，则显式记录为 None。
            self.shared_expert = None

        # -------------------- 构造承载 routed experts 的 SharedFusedMoE --------------------
        # SharedFusedMoE 会统一管理 routed experts，以及可选的 shared expert 输出拼接。
        self.experts = SharedFusedMoE(
            # 共享专家模块；若未启用则为 None。
            shared_experts=self.shared_expert,
            # 外部 router gate 模块。
            gate=self.gate,
            # routed expert 的逻辑数量。
            num_experts=self.n_routed_experts,
            # 每个 token 最多路由到 top-k 个专家。
            top_k=config.num_experts_per_tok,
            # 输入 hidden size。
            hidden_size=config.hidden_size,
            # routed expert 的中间层维度。
            intermediate_size=config.moe_intermediate_size,
            # 本层不在内部做结果归并。
            reduce_results=False,
            # 是否对 top-k 概率重新归一化，默认从模型配置读取。
            renormalize=getattr(config, "norm_topk_prob", True),
            # 透传量化配置给 FusedMoE 内部量化实现。
            quant_config=quant_config,
            # routed experts 模块前缀。
            prefix=f"{prefix}.experts",
            # 是否启用 EPLB，会影响逻辑/物理 expert 映射和运行时重排。
            enable_eplb=self.enable_eplb,
            # 冗余 expert 数量，用于为 EPLB 预留额外 physical experts。
            num_redundant_experts=self.n_redundant_experts,
            # 是否启用 sequence parallel 模式。
            is_sequence_parallel=self.is_sequence_parallel,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # ------------------------------- 规范化输入形状并记录原始布局 -------------------------------
        # 先保存调用方传入的原始形状，函数返回前会按这个形状还原输出。
        orig_shape = hidden_states.shape
        # 当前 MoE 路径按 [num_tokens, hidden_dim] 组织输入，这里取出两个核心维度。
        num_tokens, hidden_dim = hidden_states.shape
        # 显式整理成二维张量，保证后续 gate 和 experts 使用统一输入格式。
        hidden_states = hidden_states.view(-1, hidden_dim)

        # ------------------------------- 按需切分 sequence parallel 的本地 token 分片 -------------------------------
        # 开启 sequence parallel 时，当前 rank 只处理自己负责的那部分 token。
        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        # ------------------------------- 计算路由信息并执行专家前向 -------------------------------
        # 如果 experts 内部自带 router，就直接把输入交给 experts，
        # gate 计算会在 FusedMoE 内部完成。
        if self.experts.is_internal_router:
            final_hidden_states = self.experts(
                hidden_states=hidden_states,
                router_logits=hidden_states
            )
        else:
            # 如果 router 不在 experts 内部，就先用外部 gate 计算每个 token 的 router logits。
            router_logits, _ = self.gate(hidden_states)
            # 再把 token 表示和对应的 router logits 一起交给 experts 执行 MoE 计算。
            final_hidden_states = self.experts(
                hidden_states=hidden_states,
                router_logits=router_logits
            )

        # ------------------------------- 合并 shared expert 与 routed experts 的输出 -------------------------------
        # 开启 shared expert 时，experts 返回的是两个分支的输出，这里把它们逐元素相加。
        if self.shared_expert is not None:
            final_hidden_states = final_hidden_states[0] + final_hidden_states[1]

        # ------------------------------- 按并行策略恢复最终输出布局 -------------------------------
        # 注意：sequence parallel 不是“不考虑 TP”，而是建立在 TP group 之上的另一种切分方式。
        # 在这种模式下，各 TP rank 持有的是不同 token 分片，因此这里要沿 token 维做 all-gather，
        # 把局部序列重新拼回完整序列，而不是像普通 TP 那样对各 rank 的部分结果做 all-reduce。
        if self.is_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )
            # gather 后可能带有补齐 token，这里裁回原始 token 数。
            final_hidden_states = final_hidden_states[:num_tokens]
        elif self.tp_size > 1:
            # 非 sequence parallel 但存在 TP 时，各 rank 通常持有同一批 token 的部分贡献，
            # 因此这里按需做 TP all-reduce，把这些部分结果规约成完整输出。
            final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(
                final_hidden_states
            )

        # 按输入进入本函数前的形状还原输出，保持该层对外接口不变。
        return final_hidden_states.view(orig_shape)


class Qwen3NextGatedDeltaNet(nn.Module, MambaBase):
    # Qwen3Next / Qwen3.5 中 linear_attention 路径使用的 Gated DeltaNet 模块
    # 它不是标准 full attention，而是带有“状态递推 + 门控”的线性注意力实现
    #
    # 继承：
    # - nn.Module：普通 PyTorch 模块
    # - MambaBase：复用“状态型模块”的缓存 / state 管理接口

    @property
    def mamba_type(self) -> str:
        # 返回当前这种“状态型模块”的类型名
        # 在统一的 Mamba/GDN 状态管理框架里，用它区分模块种类
        return "gdn_attention"

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        # 返回 GDN 状态缓存的 dtype 配置
        # 一般会返回两类状态的 dtype，例如：
        # - recurrent/gated delta 状态 dtype
        # - ssm cache dtype
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            self.model_config.dtype,  # 模型主 dtype
            self.cache_config.mamba_cache_dtype,  # mamba/gdn cache dtype
            self.cache_config.mamba_ssm_cache_dtype,  # ssm cache dtype
        )

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        # 返回 GDN 状态缓存需要的形状
        # 形状取决于：
        # - TP 切分后头数
        # - key/value head dim
        # - conv kernel size
        # - speculative decoding token 数
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            self.tp_size,  # tensor parallel 大小
            self.num_k_heads,  # key 头总数
            self.num_v_heads,  # value 头总数
            self.head_k_dim,  # key 单头维度
            self.head_v_dim,  # value 单头维度
            self.conv_kernel_size,  # 卷积核长度
            self.num_spec,  # speculative token 数
        )

    def __init__(
            self,
            config: Qwen3NextConfig,  # 模型结构配置
            model_config: ModelConfig | None = None,  # 外层模型配置
            cache_config: CacheConfig | None = None,  # cache 配置
            quant_config: QuantizationConfig | None = None,  # 量化配置
            speculative_config: SpeculativeConfig | None = None,  # speculative decoding 配置
            prefix: str = "",  # 当前模块名前缀
    ) -> None:
        # 初始化父类
        super().__init__()

        # 当前 tensor parallel 的 world size
        self.tp_size = get_tensor_model_parallel_world_size()

        # 当前 tensor parallel rank
        self.tp_rank = get_tensor_model_parallel_rank()

        # 主隐藏维度
        self.hidden_size = config.hidden_size

        # linear attention / GDN 中 value 头总数
        self.num_v_heads = config.linear_num_value_heads

        # linear attention / GDN 中 key 头总数
        self.num_k_heads = config.linear_num_key_heads

        # key 单头维度
        self.head_k_dim = config.linear_key_head_dim

        # value 单头维度
        self.head_v_dim = config.linear_value_head_dim

        # 所有 key 头拼起来后的总维度
        self.key_dim = self.head_k_dim * self.num_k_heads

        # 所有 value 头拼起来后的总维度
        self.value_dim = self.head_v_dim * self.num_v_heads

        # GDN 中卷积核长度
        self.conv_kernel_size = config.linear_conv_kernel_dim

        # 从模块名前缀里提取当前层号
        self.layer_idx = extract_layer_index(prefix)

        # 激活函数名字，例如 silu
        self.activation = config.hidden_act

        # 实际激活函数对象
        self.act = ACT2FN[config.hidden_act]

        # norm 使用的 eps
        self.layer_norm_epsilon = config.rms_norm_eps

        # 保存模块名前缀
        self.prefix = prefix

        # 保存配置对象
        self.config = config
        self.model_config = model_config
        self.cache_config = cache_config
        self.quant_config = quant_config
        self.speculative_config = speculative_config

        # speculative decoding 时的 speculative token 数
        self.num_spec = (
            self.speculative_config.num_speculative_tokens
            if self.speculative_config
            else 0
        )

        # --------------------------------------------------
        # 1. 卷积分支：用于对 Q/K/V 输入做局部序列变换
        # --------------------------------------------------

        # conv1d 的输出维度：
        # [Q, K, V] 三部分拼接
        # Q 占 key_dim，K 占 key_dim，V 占 value_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim

        # 这里并不是标准 torch Conv1d，而是用 ColumnParallelLinear 封装出的等价形式
        # 输入维度 = conv kernel 长度
        # 输出维度 = Q/K/V 拼接维度
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_kernel_size,
            output_size=self.conv_dim,
            bias=False,
            prefix=f"{prefix}.conv1d",
        )

        # 把权重形状扩成 conv 风格，便于后续按卷积方式使用
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        # --------------------------------------------------
        # 2. 输入投影：hidden_states -> qkvz / ba
        # --------------------------------------------------

        # 生成 in_proj_qkvz：
        # 把输入 hidden_states 投影成 Q/K/V/Z
        # Qwen3.5 子类会重写这个构造逻辑，以适配自己的权重布局
        self.in_proj_qkvz = self.create_qkvz_proj(
            hidden_size=self.hidden_size,
            key_dim=self.key_dim,
            value_dim=self.value_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj_qkvz",
        )

        # 生成 in_proj_ba：
        # 把输入 hidden_states 投影成 b / a 两组门控参数
        # Qwen3.5 子类也会重写这里，以适配不同 checkpoint 格式
        self.in_proj_ba = self.create_ba_proj(
            hidden_size=self.hidden_size,
            num_v_heads=self.num_v_heads,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj_ba",
        )

        # --------------------------------------------------
        # 3. 为 conv1d.weight 重新设置分片加载规则
        # --------------------------------------------------

        # conv1d 对应的权重里，Q/K/V 分别占不同片段
        # 这里配置 sharded weight loader 时，需要告诉它每块大小和切法
        query_key_settings = (self.key_dim, 0, False)
        value_settings = (self.value_dim, 0, False)

        # 先删除默认的 weight_loader
        delattr(self.conv1d.weight, "weight_loader")

        # 重新绑定专用的 mamba_v2_sharded_weight_loader
        set_weight_attrs(
            self.conv1d.weight,
            {
                "weight_loader": mamba_v2_sharded_weight_loader(
                    [
                        query_key_settings,  # Q
                        query_key_settings,  # K
                        value_settings,  # V
                    ],
                    self.tp_size,  # TP world size
                    self.tp_rank,  # 当前 TP rank
                )
            },
        )

        # --------------------------------------------------
        # 4. GDN 递推参数：dt_bias / A_log
        # --------------------------------------------------

        # dt_bias：每个 value head 一个偏置，用于控制时间步长/门控更新
        # TP 下按 value head 均分
        self.dt_bias = nn.Parameter(
            torch.ones(self.num_v_heads // self.tp_size),
        )

        # A_log：状态更新相关的对数参数
        # 一般后续会通过 exp(A_log) 使用
        self.A_log = nn.Parameter(
            torch.empty(
                divide(self.num_v_heads, self.tp_size),
            )
        )

        # 给这两个参数设置分片加载规则
        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(0)})
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        # --------------------------------------------------
        # 5. 输出前的 gated norm
        # --------------------------------------------------

        # RMSNormGated：带门控的 RMSNorm
        # 这里一般会把 core attention 输出和 z 门控一起做归一化/调制
        self.norm = RMSNormGated(
            self.head_v_dim,  # 归一化按 value head dim 做
            eps=self.layer_norm_epsilon,  # 数值稳定系数
            group_size=None,  # 不做 group norm
            norm_before_gate=True,  # 先 norm 再 gate
            device=current_platform.current_device(),  # 放到当前设备
        )

        # --------------------------------------------------
        # 6. 输出投影
        # --------------------------------------------------

        # 把所有 value heads 拼接后的输出重新投影回 hidden_size
        # input_is_parallel=True 表示输入已经是 TP 切分后的并行张量
        self.out_proj = RowParallelLinear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

        # --------------------------------------------------
        # 7. 核心 GDN chunk 计算规则
        # --------------------------------------------------

        # 这是 GDN prefill 路径中的核心 chunked 递推规则模块
        self.chunk_gated_delta_rule = ChunkGatedDeltaRule()

        # 是否启用 packed recurrent decode 快路径
        self.enable_packed_recurrent_decode = (
            envs.VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE
        )

        # --------------------------------------------------
        # 8. 注册到静态 forward context
        # --------------------------------------------------

        # 取出当前全局编译配置
        compilation_config = get_current_cfie_config().compilation_config

        # prefix 不能重复，否则静态 forward context 会混乱
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")

        # 把当前层对象注册进去
        # 这样 torch custom op / 编译逻辑可以通过 prefix 找回当前层实例
        compilation_config.static_forward_context[prefix] = self

    def create_qkvz_proj(
            self,
            hidden_size: int,
            key_dim: int,
            value_dim: int,
            quant_config: QuantizationConfig | None,
            prefix: str,
    ) -> MergedColumnParallelLinear:
        return MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[sum((key_dim, key_dim, value_dim, value_dim))],
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
        # 中文注释：Qwen3-Next stores in_proj_ba as a single fused weight with an
        # 中文注释：interleaved GQA layout: [b_g0, a_g0, b_g1, a_g1, ...] where
        # 中文注释：each group corresponds to a key-head group. We must use a single
        # 中文注释：output shard so that ColumnParallel sharding preserves this
        # 中文注释：interleaved structure across TP ranks.
        # 中文注释：Qwen3.5 overrides this to use [num_v_heads, num_v_heads] since
        # 中文注释：its checkpoint has separate in_proj_b and in_proj_a weights.
        return MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[num_v_heads * 2],
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
        )

    def fix_query_key_value_ordering(
            self,
            mixed_qkvz: torch.Tensor,
            mixed_ba: torch.Tensor,
    ):
        """
        中文说明：Derives `query`, `key` and `value` tensors from `mixed_qkvzba`.
        """
        new_tensor_shape_qkvz = mixed_qkvz.size()[:-1] + (
            self.num_k_heads // self.tp_size,
            (
                    self.head_k_dim
                    + self.head_k_dim
                    + (self.head_v_dim + self.head_v_dim)
                    * self.num_v_heads
                    // self.num_k_heads
            ),
        )
        new_tensor_shape_ba = mixed_qkvz.size()[:-1] + (
            self.num_k_heads // self.tp_size,
            2 * self.num_v_heads // self.num_k_heads,
        )

        mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
        mixed_ba = mixed_ba.view(*new_tensor_shape_ba)

        split_arg_list_qkvz = [
            self.head_k_dim,
            self.head_k_dim,
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
        ]
        split_arg_list_ba = [
            self.num_v_heads // self.num_k_heads,
            self.num_v_heads // self.num_k_heads,
        ]

        # 中文注释：[b, sq, ng, (hn + hn + np/ng * hn + np/ng + np/ng)]
        # 中文注释：--> [b, sq, ng, hn], [b, sq, ng, hn], [b, sq, ng, np/ng * hn],
        # 中文注释：[b, sq, ng, np/ng * hn], [b, sq, ng, np/ng], [b, sq, ng, np/ng]
        (query, key, value, z) = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=2)
        (b, a) = torch.split(mixed_ba, split_arg_list_ba, dim=2)

        # 中文注释：[b, sq, ng, np/ng * hn] -> [b, sq, np, hn]
        value = value.reshape(value.size(0), -1, self.head_v_dim)
        z = z.reshape(z.size(0), -1, self.head_v_dim)
        b = b.reshape(b.size(0), self.num_v_heads // self.tp_size)
        a = a.reshape(a.size(0), self.num_v_heads // self.tp_size)

        return query, key, value, z, b, a

    def rearrange_mixed_qkv(self, mixed_qkv):
        if mixed_qkv is None:
            return None, None, None
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim // self.tp_size,
                self.key_dim // self.tp_size,
                self.value_dim // self.tp_size,
            ],
            dim=-1,
        )
        query, key = map(
            lambda x: rearrange(x, "l (h d) -> 1 l h d", d=self.head_k_dim),
            (query, key),
        )
        value = rearrange(value, "l (h d) -> 1 l h d", d=self.head_v_dim)
        return query.contiguous(), key.contiguous(), value.contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            output: torch.Tensor,
    ):
        """
        中文说明：Forward pass with three parts:
        中文说明：1. Input projection
        中文说明：2. Core attention (custom op)
        中文说明：3. Output projection
        """
        num_tokens = hidden_states.size(0)

        # ============================================================
        # 中文注释：Part 1: Input Projection
        # ============================================================
        projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
        projected_states_ba, _ = self.in_proj_ba(hidden_states)
        query, key, value, z, b, a = self.fix_query_key_value_ordering(
            projected_states_qkvz, projected_states_ba
        )
        query, key, value = map(
            lambda x: rearrange(x, "l p d -> l (p d)"), (query, key, value)
        )
        mixed_qkv = torch.cat((query, key, value), dim=-1)

        # ============================================================
        # 中文注释：Part 2: Core Attention (Custom Op)
        # ============================================================
        # 中文注释：Note: we should not use torch.empty here like other attention backends,
        # 中文注释：see discussions in https://github.com/vllm-project/vllm/pull/28182
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
        # 中文注释：Part 3: Output Projection
        # ============================================================
        z_shape_og = z.shape
        # 中文注释：Reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        output[:num_tokens], _ = self.out_proj(core_attn_out)

    def _warmup_prefill_kernels(self, mixed_qkv: torch.Tensor) -> None:
        """在 V1 画像阶段预热 GDN 预填充内核。

        中文说明：During V1 profile runs, ``_forward_core`` returns early because
        中文说明：``attn_metadata`` is ``None``, so the autotuned kernels used by
        中文说明：``chunk_gated_delta_rule`` (e.g. ``solve_tril``,
        中文说明：``chunk_scaled_dot_kkt``) are never invoked.  After profiling,
        中文说明：vLLM allocates KV cache using most of the remaining GPU memory.
        中文说明：When the first real inference triggers the autotuner it OOMs
        中文说明：because there is not enough memory left for benchmarking.

        中文说明：This method runs minimal forward passes through
        中文说明：``chunk_gated_delta_rule`` with small dummy tensors to force
        中文说明：autotuning while GPU memory is still plentiful.  The autotuner
        中文说明：results are cached globally, so only the first layer incurs
        中文说明：actual benchmarking cost.

        中文说明：Most kernels use a fixed ``BT = chunk_size`` (64), but
        中文说明：``chunk_fwd_kernel_o`` recomputes ``BT`` from the sequence
        中文说明：length: ``min(64, max(16, next_power_of_2(T)))``.  Since ``BT``
        中文说明：is part of its autotune key, we run warmup passes with T = 16,
        中文说明：32, and 64 to cover all possible ``BT`` values.

        中文说明：The decode path uses ``fused_sigmoid_gating_delta_rule_update``
        中文说明：which has fixed kernel parameters (no autotuning), so only the
        中文说明：prefill (chunked) path needs warming up.
        """
        if hasattr(self, "_prefill_kernels_warmed_up"):
            return
        self._prefill_kernels_warmed_up = True

        device = mixed_qkv.device
        dtype = mixed_qkv.dtype
        num_k_heads = self.num_k_heads // self.tp_size
        num_v_heads = self.num_v_heads // self.tp_size
        _, state_dtype = self.get_state_dtype()
        set_triton_allocator(device)

        # 中文注释：Run warmup for each possible BT value of chunk_fwd_kernel_o:
        # 中文注释：T=16 → BT=16, T=32 → BT=32, T=64 → BT=64.
        # 中文注释：Other kernels always use BT=chunk_size(64), so their autotune
        # 中文注释：cache is populated on the first pass and reused thereafter.
        for T in (16, 32, 64):
            q = torch.randn(
                1, T, num_k_heads, self.head_k_dim, device=device, dtype=dtype
            )
            k = torch.randn(
                1, T, num_k_heads, self.head_k_dim, device=device, dtype=dtype
            )
            v = torch.randn(
                1, T, num_v_heads, self.head_v_dim, device=device, dtype=dtype
            )
            g = torch.randn(1, T, num_v_heads, device=device, dtype=dtype)
            beta = torch.randn(1, T, num_v_heads, device=device, dtype=dtype)
            state = torch.zeros(
                1,
                num_v_heads,
                self.head_v_dim,
                self.head_k_dim,
                device=device,
                dtype=state_dtype,
            )
            cu_seqlens = torch.tensor([0, T], device=device, dtype=torch.long)

            try:
                self.chunk_gated_delta_rule(
                    q=q,
                    k=k,
                    v=v,
                    g=g,
                    beta=beta,
                    initial_state=state,
                    output_final_state=False,
                    cu_seqlens=cu_seqlens,
                    use_qk_l2norm_in_kernel=True,
                )
            except Exception:
                logger.warning(
                    "GDN prefill kernel warmup (T=%d) failed for "
                    "layer %s. First inference may OOM due to "
                    "autotuner.",
                    T,
                    self.prefix,
                    exc_info=True,
                )
            else:
                logger.debug(
                    "GDN prefill kernel warmup (T=%d) completed for layer %s",
                    T,
                    self.prefix,
                )
            finally:
                del q, k, v, g, beta, state, cu_seqlens

        torch.accelerator.empty_cache()

    def _forward_core(
            self,
            mixed_qkv: torch.Tensor,  # 形状: [当前传入的总 token 数, 每个 token 的融合 qkv 维度]
            b: torch.Tensor,  # 形状: [当前传入的总 token 数, 当前 TP rank 上的 value 头数]
            a: torch.Tensor,  # 形状: [当前传入的总 token 数, 当前 TP rank 上的 value 头数]
            core_attn_out: torch.Tensor,  # 形状: [当前传入的总 token 数, 当前 TP rank 上的 value 头数, 每个 value 头的维度]
    ):
        forward_context = get_forward_context()
        attn_metadata: AttentionMetadata = forward_context.attn_metadata

        # 如果当前没有真实的注意力元数据，说明现在处于 profiling / warmup 阶段
        # 这里只做 kernel 预热，不做真正的 GDN 计算
        # mixed_qkv 形状: [当前传入的总 token 数, 每个 token 的融合 qkv 维度]
        if attn_metadata is None:
            self._warmup_prefill_kernels(mixed_qkv)
            return

        # 这里要求 attn_metadata 是一个 dict
        # key 是每一层自己的 prefix，value 是该层对应的元数据
        assert isinstance(attn_metadata, dict)

        # 取出当前这一层自己的 attention metadata
        attn_metadata = attn_metadata[self.prefix]

        # 当前层必须使用 GDN 专用的 attention metadata 类型
        assert isinstance(attn_metadata, GDNAttentionMetadata)

        # 如果启用了 packed recurrent decode 快路径，并且：
        # 1. 没有 speculative 部分
        # 2. 没有 prefill token
        # 3. 只有 decode token
        # 那么直接走专门的 decode-only 快路径
        if (
                self.enable_packed_recurrent_decode
                and attn_metadata.spec_sequence_masks is None
                and attn_metadata.num_prefills == 0
                and attn_metadata.num_decodes > 0
        ):
            return self._forward_core_decode_non_spec(
                mixed_qkv=mixed_qkv,  # 形状: [当前传入的总 token 数, 每个 token 的融合 qkv 维度]
                b=b,  # 形状: [当前传入的总 token 数, 当前 TP rank 上的 value 头数]
                a=a,  # 形状: [当前传入的总 token 数, 当前 TP rank 上的 value 头数]
                core_attn_out=core_attn_out,  # 形状: [当前传入的总 token 数, 当前 TP rank 上的 value 头数, 每个 value 头的维度]
                attn_metadata=attn_metadata,
                virtual_engine=forward_context.virtual_engine,
            )

        # 表示每条 non-spec 序列是否已经有历史 recurrent state
        # prefill 场景下常见形状: [non-spec 序列条数]
        has_initial_state = attn_metadata.has_initial_state

        # ------------------ 样本序列划分 -------------------

        # speculative 部分各条序列在拼接 token 序列中的边界
        # 形状: [speculative 序列条数 + 1]
        # 例如 [0, 3, 7] 表示：
        # 第 0 条序列占 [0:3]，第 1 条序列占 [3:7]
        spec_query_start_loc = attn_metadata.spec_query_start_loc

        # non-spec 部分各条序列在拼接 token 序列中的边界
        # 形状: [non-spec 序列条数 + 1]
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        # --------------------------------------------------

        # 若不为 None，表示当前 batch 中既有 speculative token 又有 non-spec token
        spec_sequence_masks = attn_metadata.spec_sequence_masks

        # ---------------- spec/non_spec划分 -------------------

        # speculative token 在原始 token 维度中的位置索引
        # 形状: [speculative token 数]
        spec_token_indx = attn_metadata.spec_token_indx

        # non-spec token 在原始 token 维度中的位置索引
        # 形状: [non-spec token 数]
        non_spec_token_indx = attn_metadata.non_spec_token_indx
        # -----------------------------------------------------

        # -------------- 序列映射到状态缓存的索引张量 ----------------------------

        # 常见形状: [speculative 序列条数, 每条 speculative 序列的最大 query 长度]
        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor

        # non-spec token / non-spec 序列映射到状态缓存的索引张量
        # prefill 常见形状: [non-spec 序列条数]
        # decode 常见形状: [non-spec token 数] 或 [non-spec 序列条数]
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor
        # ----------------------------------------------------------------------

        # 取出当前 virtual engine 对应的状态缓存
        self_kv_cache = self.kv_cache[forward_context.virtual_engine]

        # 第 0 个缓存是卷积状态缓存
        # 原始缓存维度转置后，常见可理解为：
        # [状态槽数量, 当前 TP rank 上的融合 qkv 维度, 卷积核长度]
        conv_state = self_kv_cache[0].transpose(-1, -2)

        # 第 1 个缓存是 recurrent / SSM 状态缓存
        # 常见形状: [状态槽数量, 当前 TP rank 上的 value 头数, 每个 value 头的维度, 每个 key 头的维度]
        ssm_state = self_kv_cache[1]

        # 本次真正参与计算的 token 数
        num_actual_tokens = attn_metadata.num_actual_tokens

        # speculative decode 中被接受的 token 数
        num_accepted_tokens = attn_metadata.num_accepted_tokens

        # 把输入截断到本次真正有效的 token 数
        mixed_qkv = mixed_qkv[:num_actual_tokens]
        # mixed_qkv 形状变成: [本次真正参与计算的 token 数, 每个 token 的融合 qkv 维度]

        b = b[:num_actual_tokens]
        # b 形状变成: [本次真正参与计算的 token 数, 当前 TP rank 上的 value 头数]

        a = a[:num_actual_tokens]
        # a 形状变成: [本次真正参与计算的 token 数, 当前 TP rank 上的 value 头数]

        # 将 conv1d 权重 reshape 成 kernel 期望的二维形式
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        # conv_weights 形状: [当前 TP rank 上的融合 qkv 维度, 卷积核长度]

        # ------- 按 speculative / non-spec 分开 token -------
        if spec_sequence_masks is not None:
            # 如果存在 speculative 与 non-spec 混合
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                # 特殊情况：当前全部 token 都被当作 speculative 处理
                mixed_qkv_spec = mixed_qkv
                # 形状: [本次真正参与计算的 token 数, 每个 token 的融合 qkv 维度]

                mixed_qkv_non_spec = None
            else:
                # 取出 speculative token 对应的 qkv
                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
                # 形状: [speculative token 数, 每个 token 的融合 qkv 维度]

                # 取出 non-spec token 对应的 qkv
                mixed_qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)
                # 形状: [non-spec token 数, 每个 token 的融合 qkv 维度]
        else:
            # 如果没有 speculative，全部都归为 non-spec
            mixed_qkv_spec = None
            mixed_qkv_non_spec = mixed_qkv
            # 形状: [本次真正参与计算的 token 数, 每个 token 的融合 qkv 维度]

        # ------- speculative 部分先做 conv_state 更新 -------
        if spec_sequence_masks is not None:
            mixed_qkv_spec = causal_conv1d_update(
                mixed_qkv_spec,  # 形状: [speculative token 数, 每个 token 的融合 qkv 维度]
                conv_state,  # 形状: [speculative 序列条数, 当前 TP rank 上的融合 qkv 维度, 卷积核长度]
                conv_weights,  # 形状: [当前 TP rank 上的融合 qkv 维度, 卷积核长度]
                self.conv1d.bias,  # 形状通常是: [当前 TP rank 上的融合 qkv 维度]，或 None
                self.activation,
                conv_state_indices=spec_state_indices_tensor[:, 0][
                    : attn_metadata.num_spec_decodes  # speculative decode 分支中一共有多少条“spec 解码序列”需要处理。
                ],
                # 常见形状: [speculative decode 序列条数]
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,  # 形状: [speculative 序列条数 + 1]
                max_query_len=spec_state_indices_tensor.size(-1),
                validate_data=False,
            )
            # 输出 mixed_qkv_spec 形状仍然是:
            # [speculative token 数, 每个 token 的融合 qkv 维度]

        # ------- non-spec 部分做 conv_state 更新 -------
        if attn_metadata.num_prefills > 0:
            mixed_qkv_non_spec_T = mixed_qkv_non_spec.transpose(0, 1)
            # 转置后形状: [每个 token 的融合 qkv 维度, non-spec token 数]

            mixed_qkv_non_spec = causal_conv1d_fn(
                mixed_qkv_non_spec_T,  # 形状: [每个 token 的融合 qkv 维度, non-spec token 数]
                conv_weights,  # 形状: [当前 TP rank 上的融合 qkv 维度, 卷积核长度]
                self.conv1d.bias,  # 形状通常是: [当前 TP rank 上的融合 qkv 维度]，或 None
                activation=self.activation,
                conv_states=conv_state,  # 形状: [状态槽数量, 当前 TP rank 上的融合 qkv 维度, 卷积核长度]
                has_initial_state=has_initial_state,  # 形状: [non-spec 序列条数]
                cache_indices=non_spec_state_indices_tensor,
                # 常见形状: [non-spec 序列条数]

                query_start_loc=non_spec_query_start_loc,  # 形状: [non-spec 序列条数 + 1]
                metadata=attn_metadata,
            ).transpose(0, 1)
            # 转回后 mixed_qkv_non_spec 形状:
            # [non-spec token 数, 每个 token 的融合 qkv 维度]

        elif attn_metadata.num_decodes > 0:
            mixed_qkv_non_spec = causal_conv1d_update(
                mixed_qkv_non_spec,  # 形状: [non-spec token 数, 每个 token 的融合 qkv 维度]
                conv_state,  # 形状: [状态槽数量, 当前 TP rank 上的融合 qkv 维度, 卷积核长度]
                conv_weights,  # 形状: [当前 TP rank 上的融合 qkv 维度, 卷积核长度]
                self.conv1d.bias,
                self.activation,
                conv_state_indices=non_spec_state_indices_tensor[
                    : attn_metadata.num_actual_tokens
                ],
                # decode 时常见形状: [non-spec token 数]

                validate_data=True,
            )
            # 输出 mixed_qkv_non_spec 形状仍然是:
            # [non-spec token 数, 每个 token 的融合 qkv 维度]
        else:
            mixed_qkv_non_spec = None

        # ------- 把 mixed_qkv 拆成 query / key / value -------
        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)
        # 若 mixed_qkv_spec 非 None:
        # query_spec 形状: [1, speculative token 数, 当前 TP rank 上的 key 头数, 每个 key 头的维度]
        # key_spec   形状: [1, speculative token 数, 当前 TP rank 上的 key 头数, 每个 key 头的维度]
        # value_spec 形状: [1, speculative token 数, 当前 TP rank 上的 value 头数, 每个 value 头的维度]
        # 若 mixed_qkv_spec 为 None，则三者都为 None

        query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(
            mixed_qkv_non_spec
        )
        # 若 mixed_qkv_non_spec 非 None:
        # query_non_spec 形状: [1, non-spec token 数, 当前 TP rank 上的 key 头数, 每个 key 头的维度]
        # key_non_spec   形状: [1, non-spec token 数, 当前 TP rank 上的 key 头数, 每个 key 头的维度]
        # value_non_spec 形状: [1, non-spec token 数, 当前 TP rank 上的 value 头数, 每个 value 头的维度]

        # ------- prefill 路径下先计算 gating 参数 g / beta -------
        if attn_metadata.num_prefills > 0:
            g, beta = fused_gdn_gating(self.A_log, a, b, self.dt_bias)
            # g 形状:    [1, 本次真正参与计算的 token 数, 当前 TP rank 上的 value 头数]
            # beta 形状: [1, 本次真正参与计算的 token 数, 当前 TP rank 上的 value 头数]

            if spec_sequence_masks is not None:
                g_non_spec = g.index_select(1, non_spec_token_indx)
                # 形状: [1, non-spec token 数, 当前 TP rank 上的 value 头数]

                beta_non_spec = beta.index_select(1, non_spec_token_indx)
                # 形状: [1, non-spec token 数, 当前 TP rank 上的 value 头数]
            else:
                g_non_spec = g
                # 形状: [1, 本次真正参与计算的 token 数, 当前 TP rank 上的 value 头数]

                beta_non_spec = beta
                # 形状: [1, 本次真正参与计算的 token 数, 当前 TP rank 上的 value 头数]
        else:
            g_non_spec = None
            beta_non_spec = None

        # ------- speculative 部分核心递推 -------
        if spec_sequence_masks is not None:
            core_attn_out_spec, last_recurrent_state = (
                fused_sigmoid_gating_delta_rule_update(
                    A_log=self.A_log,  # 形状: [当前 TP rank 上的 value 头数]
                    a=a,  # 形状: [本次真正参与计算的 token 数, 当前 TP rank 上的 value 头数]
                    b=b,  # 形状: [本次真正参与计算的 token 数, 当前 TP rank 上的 value 头数]
                    dt_bias=self.dt_bias,  # 形状: [当前 TP rank 上的 value 头数]
                    q=query_spec,  # 形状: [1, speculative token 数, 当前 TP rank 上的 key 头数, 每个 key 头的维度]
                    k=key_spec,  # 形状: [1, speculative token 数, 当前 TP rank 上的 key 头数, 每个 key 头的维度]
                    v=value_spec,  # 形状: [1, speculative token 数, 当前 TP rank 上的 value 头数, 每个 value 头的维度]
                    initial_state=ssm_state,
                    # 形状常见: [状态槽数量, 当前 TP rank 上的 value 头数, 每个 value 头的维度, 每个 key 头的维度]

                    inplace_final_state=True,
                    cu_seqlens=spec_query_start_loc[
                        : attn_metadata.num_spec_decodes + 1
                    ],
                    # 形状: [speculative decode 序列条数 + 1]

                    ssm_state_indices=spec_state_indices_tensor,
                    # 常见形状: [speculative 序列条数, 每条 speculative 序列的最大 query 长度]

                    num_accepted_tokens=num_accepted_tokens,
                    use_qk_l2norm_in_kernel=True,
                )
            )
            # core_attn_out_spec 形状:
            # [1, speculative token 数, 当前 TP rank 上的 value 头数, 每个 value 头的维度]
            #
            # last_recurrent_state 常见形状与状态缓存兼容，
            # 通常可理解为:
            # [相关序列数, 当前 TP rank 上的 value 头数, 每个 value 头的维度, 每个 key 头的维度]
        else:
            core_attn_out_spec, last_recurrent_state = None, None

        # ------- non-spec 部分核心递推 -------
        if attn_metadata.num_prefills > 0:
            initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()
            # 取出 non-spec 序列对应的初始状态
            # 常见形状:
            # [non-spec 序列条数, 当前 TP rank 上的 value 头数, 每个 value 头的维度, 每个 key 头的维度]

            initial_state[~has_initial_state, ...] = 0
            # 对没有历史状态的序列置零
            # 形状不变

            (
                core_attn_out_non_spec,
                last_recurrent_state,
            ) = self.chunk_gated_delta_rule(
                q=query_non_spec,
                # 形状: [1, non-spec token 数, 当前 TP rank 上的 key 头数, 每个 key 头的维度]

                k=key_non_spec,
                # 形状: [1, non-spec token 数, 当前 TP rank 上的 key 头数, 每个 key 头的维度]

                v=value_non_spec,
                # 形状: [1, non-spec token 数, 当前 TP rank 上的 value 头数, 每个 value 头的维度]

                g=g_non_spec,
                # 形状: [1, non-spec token 数, 当前 TP rank 上的 value 头数]

                beta=beta_non_spec,
                # 形状: [1, non-spec token 数, 当前 TP rank 上的 value 头数]

                initial_state=initial_state,
                # 形状: [non-spec 序列条数, 当前 TP rank 上的 value 头数, 每个 value 头的维度, 每个 key 头的维度]

                output_final_state=True,
                cu_seqlens=non_spec_query_start_loc,
                # 形状: [non-spec 序列条数 + 1]

                use_qk_l2norm_in_kernel=True,
            )
            # core_attn_out_non_spec 形状:
            # [1, non-spec token 数, 当前 TP rank 上的 value 头数, 每个 value 头的维度]
            #
            # last_recurrent_state 常见形状:
            # [non-spec 序列条数, 当前 TP rank 上的 value 头数, 每个 value 头的维度, 每个 key 头的维度]

            ssm_state[non_spec_state_indices_tensor] = last_recurrent_state.to(
                ssm_state.dtype
            )
            # 写回后 ssm_state 形状不变，仍然是:
            # [状态槽数量, 当前 TP rank 上的 value 头数, 每个 value 头的维度, 每个 key 头的维度]

        elif attn_metadata.num_decodes > 0:
            core_attn_out_non_spec, last_recurrent_state = (
                fused_sigmoid_gating_delta_rule_update(
                    A_log=self.A_log,  # 形状: [当前 TP rank 上的 value 头数]
                    a=a,  # 形状: [本次真正参与计算的 token 数, 当前 TP rank 上的 value 头数]
                    b=b,  # 形状: [本次真正参与计算的 token 数, 当前 TP rank 上的 value 头数]
                    dt_bias=self.dt_bias,  # 形状: [当前 TP rank 上的 value 头数]
                    q=query_non_spec,
                    # 形状: [1, non-spec token 数, 当前 TP rank 上的 key 头数, 每个 key 头的维度]

                    k=key_non_spec,
                    # 形状: [1, non-spec token 数, 当前 TP rank 上的 key 头数, 每个 key 头的维度]

                    v=value_non_spec,
                    # 形状: [1, non-spec token 数, 当前 TP rank 上的 value 头数, 每个 value 头的维度]

                    initial_state=ssm_state,
                    # 形状常见:
                    # [状态槽数量, 当前 TP rank 上的 value 头数, 每个 value 头的维度, 每个 key 头的维度]

                    inplace_final_state=True,
                    cu_seqlens=non_spec_query_start_loc[
                        : attn_metadata.num_decodes + 1
                    ],
                    # 形状: [decode 序列条数 + 1]

                    ssm_state_indices=non_spec_state_indices_tensor,
                    # decode 时常见形状: [non-spec token 数] 或 [decode 序列条数]

                    use_qk_l2norm_in_kernel=True,
                )
            )
            # core_attn_out_non_spec 形状:
            # [1, non-spec token 数, 当前 TP rank 上的 value 头数, 每个 value 头的维度]
        else:
            core_attn_out_non_spec, last_recurrent_state = None, None

        # ------- 合并 speculative / non-spec 输出，恢复原 token 顺序 -------
        if spec_sequence_masks is not None and core_attn_out_non_spec is not None:
            merged_out = torch.empty(
                (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                dtype=core_attn_out_non_spec.dtype,
                device=core_attn_out_non_spec.device,
            )
            # merged_out 形状:
            # [1, 本次真正参与计算的 token 数, 当前 TP rank 上的 value 头数, 每个 value 头的维度]

            merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
            # 按原始 token 索引把 speculative 输出写回
            # core_attn_out_spec 形状:
            # [1, speculative token 数, 当前 TP rank 上的 value 头数, 每个 value 头的维度]

            merged_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
            # 按原始 token 索引把 non-spec 输出写回
            # core_attn_out_non_spec 形状:
            # [1, non-spec token 数, 当前 TP rank 上的 value 头数, 每个 value 头的维度]

            core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)
            # merged_out.squeeze(0) 形状:
            # [本次真正参与计算的 token 数, 当前 TP rank 上的 value 头数, 每个 value 头的维度]
            #
            # 写入后 core_attn_out 的整体形状仍然是:
            # [当前传入的总 token 数, 当前 TP rank 上的 value 头数, 每个 value 头的维度]

        elif spec_sequence_masks is not None:
            core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
            # core_attn_out_spec.squeeze(0) 形状:
            # [本次真正参与计算的 token 数, 当前 TP rank 上的 value 头数, 每个 value 头的维度]

        else:
            core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)
            # core_attn_out_non_spec.squeeze(0) 形状:
            # [本次真正参与计算的 token 数, 当前 TP rank 上的 value 头数, 每个 value 头的维度]

    def _forward_core_decode_non_spec(
            self,
            mixed_qkv: torch.Tensor,
            b: torch.Tensor,
            a: torch.Tensor,
            core_attn_out: torch.Tensor,
            attn_metadata: GDNAttentionMetadata,
            virtual_engine: int,
    ):
        """
        中文说明：Core attention computation with a packed non-spec decode fast path.
        """
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        self_kv_cache = self.kv_cache[virtual_engine]
        conv_state = self_kv_cache[0].transpose(-1, -2)
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        mixed_qkv_non_spec = causal_conv1d_update(
            mixed_qkv,
            conv_state,
            conv_weights,
            self.conv1d.bias,
            self.activation,
            conv_state_indices=non_spec_state_indices_tensor[:num_actual_tokens],
            validate_data=False,
        )
        out_buf = core_attn_out[:num_actual_tokens].unsqueeze(1)
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv_non_spec,
            a=a,
            b=b,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            scale=self.head_k_dim ** -0.5,
            initial_state=ssm_state,
            out=out_buf,
            ssm_state_indices=non_spec_state_indices_tensor[:num_actual_tokens],
            use_qk_l2norm_in_kernel=True,
        )
        return


class Qwen3NextAttention(nn.Module):
    def __init__(
            self,
            config: Qwen3NextConfig,
            model_config: ModelConfig | None = None,
            cache_config: CacheConfig | None = None,
            quant_config: QuantizationConfig | None = None,
            prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            # 中文注释：Number of KV heads is greater than TP size, so we partition
            # 中文注释：the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # 中文注释：Number of KV heads is less than TP size, so we replicate
            # 中文注释：the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim or (self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        self.attn_output_gate = getattr(config, "attn_output_gate", True)

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=getattr(config, "qkv_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
            dual_chunk_attention_config=self.dual_chunk_attention_config,
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
                "dual_chunk_attention_config": self.dual_chunk_attention_config,
            }
            if self.dual_chunk_attention_config
            else {},
        )

        self.q_norm = Qwen3NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
            self,
            positions: torch.Tensor,
            output: torch.Tensor,
            hidden_states: torch.Tensor,
    ):
        qkv, _ = self.qkv_proj(hidden_states)

        if self.attn_output_gate:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
            )
            orig_shape = q_gate.shape[:-1]
            q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
            q, gate = torch.chunk(q_gate, 2, dim=-1)
            q = q.reshape(*orig_shape, -1)
            gate = gate.reshape(*orig_shape, -1)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(
            -1, self.num_heads * self.head_dim
        )
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(
            -1, self.num_kv_heads * self.head_dim
        )

        q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)

        if self.attn_output_gate:
            gate = torch.sigmoid(gate)
            attn_output = attn_output * gate

        output[:], _ = self.o_proj(attn_output)


class Qwen3NextDecoderLayer(nn.Module):
    def __init__(
            self,
            cfie_config: CfieConfig,
            layer_type: str,
            prefix: str = "",
    ) -> None:
        super().__init__()

        config = cfie_config.model_config.hf_config
        model_config = cfie_config.model_config
        cache_config = cfie_config.cache_config
        quant_config = cfie_config.quant_config
        speculative_config = cfie_config.speculative_config

        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)

        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3NextGatedDeltaNet(
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

        mlp_only_layers = (
            [] if not hasattr(config, "mlp_only_layers") else config.mlp_only_layers
        )
        if (self.layer_idx not in mlp_only_layers) and (
                config.num_experts > 0
                and (self.layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3NextSparseMoeBlock(
                cfie_config=cfie_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = Qwen3NextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3NextRMSNorm(
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

    def forward(
            self,
            hidden_states: torch.Tensor,
            residual: torch.Tensor | None,
            positions: torch.Tensor = None,
            **kwargs: object,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        self_attention_output = torch.empty_like(hidden_states)
        if self.layer_type == "linear_attention":
            self.linear_attn(
                hidden_states=hidden_states,
                output=self_attention_output,
            )
        elif self.layer_type == "full_attention":
            self.self_attn(
                hidden_states=hidden_states,
                output=self_attention_output,
                positions=positions,
            )
        else:
            raise ValueError("Invalid layer_type")
        hidden_states = self_attention_output

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                        self.attn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                hidden_states = hidden_states * (
                        self.attn_layer_scale.to(hidden_states.dtype) + 1
                )

        # 中文注释：Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                        self.ffn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                assert len(hidden_states.shape) == len(self.ffn_layer_scale.shape), (
                    f"shape must be the same {len(hidden_states.shape)}, "
                    f"{len(self.ffn_layer_scale.shape)}"
                )
                hidden_states = hidden_states * (
                        self.ffn_layer_scale.to(hidden_states.dtype) + 1
                )

        return hidden_states, residual


@support_torch_compile
class Qwen3NextModel(nn.Module):
    # Qwen3Next 的语言模型主干（backbone）
    # 只包含：
    # - token embedding
    # - 多层 decoder layer
    # - 最后的 norm
    #
    # 不包含：
    # - lm_head
    #
    # @support_torch_compile 表示这个类的 forward 设计成可被 torch.compile 友好支持

    def __init__(self, *, cfie_config: CfieConfig, prefix: str = ""):
        # 初始化 nn.Module 基类
        super().__init__()

        # 取出文本配置
        config: Qwen3NextConfig = cfie_config.model_config.hf_text_config

        # 并行配置
        parallel_config = cfie_config.parallel_config

        # expert parallel / load balancing 配置
        eplb_config = parallel_config.eplb_config

        # 冗余专家数（主要给 MoE 场景使用）
        self.num_redundant_experts = eplb_config.num_redundant_experts

        # 保存配置对象
        self.config = config

        # 词表大小
        self.vocab_size = config.vocab_size

        # 并行词嵌入层
        # 负责把 token id 映射为 hidden_size 维 embedding
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        def get_layer(prefix: str):
            # 为每一层构造一个 Qwen3NextDecoderLayer
            # layer_type 由配置里的 layer_types 决定
            # 也就是说不同层可以是不同类型，例如：
            # - full_attention
            # - linear_attention
            return Qwen3NextDecoderLayer(
                cfie_config,
                layer_type=config.layer_types[extract_layer_index(prefix)],
                prefix=prefix,
            )

        # 构造所有 decoder layers
        #
        # make_layers 会根据总层数和当前 PP（流水线并行）配置，
        # 生成：
        # - start_layer: 当前 rank 负责的起始层号
        # - end_layer:   当前 rank 负责的结束层号
        # - layers:      层列表
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )

        # 构造“空中间张量”的工厂函数
        # 用于流水线并行时在 stage 间传递中间状态
        # 这里会涉及两个张量：
        # - hidden_states
        # - residual
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        # 只有最后一个 PP rank 才真正持有最终 norm
        if get_pp_group().is_last_rank:
            self.norm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            # 非最后一个 rank 用占位层代替
            self.norm = PPMissingLayer()

        # 需要额外导出的辅助 hidden states 的层号
        # 默认空，后续可由外部设置
        self.aux_hidden_state_layers: tuple[int, ...] = ()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
            self,
            input_ids: torch.Tensor | None,
            positions: torch.Tensor,
            intermediate_tensors: IntermediateTensors | None = None,
            inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
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

        aux_hidden_states = []
        for layer_idx, layer in enumerate(
                islice(self.layers, self.start_layer, self.end_layer),
                start=self.start_layer,
        ):
            if layer_idx in self.aux_hidden_state_layers:
                aux_hidden_states.append(
                    hidden_states + residual if residual is not None else hidden_states
                )
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # 为 routed experts 生成“checkpoint 名 -> 内部参数名”的标准映射。

        # 对当前 Qwen3.5 语义：
        # - gate_proj 和 up_proj 会在内部合并到 experts.w13_*
        # - down_proj          会在内部映射到 experts.w2_*
        #
        # 当前本地 checkpoint 还能确定：
        # - self.config.num_experts = 256
        # - self.num_redundant_experts 可能因为 EPLB/tiered cache 而非 0
        #   因此 make_expert_params_mapping(...) 里区分的是 physical expert id
        #   和 logical expert id，而不是简单 0..255 一一对应。
        return SharedFusedMoE.make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=getattr(self.config, "num_experts", 0),
            num_redundant_experts=self.num_redundant_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
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
            if "rotary_emb.inv_freq" in name:
                continue

            if name.startswith("mtp."):
                continue

            # 中文注释：Remapping the name of FP8 kv-scale.
            if name.endswith("scale"):
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                if "mlp.experts" in name:
                    continue

                name = name.replace(weight_name, param_name)
                # 中文注释：Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # 中文注释：Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                # 中文注释：name = apply_attn_prefix(name, params_dict)
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
                    if name not in params_dict:
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


class QwenNextMixtureOfExperts(MixtureOfExperts):
    # QwenNext / Qwen3.x 系列的 MoE 辅助基类
    # 主要负责：
    # 1. 维护整个模型级别的 MoE 元信息
    # 2. 在专家部署信息变化时，把信息同步到各个 MoE 层
    # 3. 从模型层里扫描并提取 MoE 超参数

    def update_physical_experts_metadata(
            self,
            num_physical_experts: int,  # 当前全局物理专家总数
            num_local_physical_experts: int,  # 当前 rank / 当前设备上的本地物理专家数
    ) -> None:
        # 断言：当前对象记录的本地物理专家数应与传入值一致
        # 这里更像是一种一致性检查，避免出现元信息更新时参数错乱
        assert self.num_local_physical_experts == num_local_physical_experts

        # 更新模型级别的物理专家总数
        self.num_physical_experts = num_physical_experts

        # 更新模型级别的本地物理专家数
        self.num_local_physical_experts = num_local_physical_experts

        # 冗余专家数 = 物理专家数 - 逻辑专家数
        # 逻辑专家数是“概念上的专家数”
        # 物理专家数可能因为负载均衡、冗余副本等原因比逻辑专家数更多
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts

        # 遍历模型的所有层
        for layer in self.model.layers:
            # 只处理 MLP 是 Sparse MoE Block 的层
            if isinstance(layer.mlp, Qwen3NextSparseMoeBlock):
                moe = layer.mlp

                # 把最新的本地物理专家数同步到该层
                moe.n_local_physical_experts = num_local_physical_experts

                # 把最新的全局物理专家数同步到该层
                moe.n_physical_experts = num_physical_experts

                # 把最新的冗余专家数同步到该层
                moe.n_redundant_experts = self.num_redundant_experts

                # 更新该层 experts 内部的 expert 映射表
                # 也就是重新建立“逻辑专家 -> 物理专家”的对应关系
                moe.experts.update_expert_map()

    def set_moe_parameters(self):
        # 初始化 expert_weights 容器
        # 后续可能用于保存专家参数引用或相关元信息
        self.expert_weights = []

        # 保存所有 MoE 层的 experts 容器
        self.moe_layers = []

        # 用来记录一个示例 MoE 层，后面直接从它提取公共超参数
        example_moe = None

        # 遍历模型中的所有层
        for layer in self.model.layers:
            # 只挑选：
            # 1. 层本身是 Qwen3NextDecoderLayer
            # 2. 其 mlp 是 Qwen3NextSparseMoeBlock
            if isinstance(layer, Qwen3NextDecoderLayer) and isinstance(
                    layer.mlp, Qwen3NextSparseMoeBlock
            ):
                # 把当前层的 MoE block 记作示例
                example_moe = layer.mlp

                # 收集该层真正的 experts 对象
                self.moe_layers.append(layer.mlp.experts)

        # 如果一个 MoE 层都没找到，说明当前模型其实不是 MoE 模型
        if example_moe is None:
            raise RuntimeError("No Qwen3Next layer found in the model.layers.")

        # 设置整个模型级别的 MoE 超参数
        # 这些值通常从任意一个 MoE 层上读取即可，因为各层配置通常一致

        # MoE 层总数
        self.num_moe_layers = len(self.moe_layers)

        # 专家组数，这里固定为 1
        self.num_expert_groups = 1

        # 共享专家数，这里固定为 0
        # 表示这里没有额外的 shared expert 组
        self.num_shared_experts = 0

        # 逻辑专家总数
        # 即模型概念上有多少个专家
        self.num_logical_experts = example_moe.n_logical_experts

        # 物理专家总数
        # 可能 >= 逻辑专家数，因为可能存在冗余/复制
        self.num_physical_experts = example_moe.n_physical_experts

        # 当前 rank / 当前设备上本地拥有的物理专家数量
        self.num_local_physical_experts = example_moe.n_local_physical_experts

        # 每次路由实际可被选中的专家总数
        self.num_routed_experts = example_moe.n_routed_experts

        # 冗余专家数
        self.num_redundant_experts = example_moe.n_redundant_experts


class Qwen3NextForCausalLM(
    nn.Module,
    HasInnerState,
    SupportsLoRA,
    SupportsPP,
    QwenNextMixtureOfExperts,
    IsHybrid,
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkvz"],
        "in_proj_ba": ["in_proj_ba"],
    }

    def __init__(self, *, cfie_config: CfieConfig, prefix: str = ""):
        config = cfie_config.model_config.hf_text_config
        self.cfie_config = cfie_config
        self.model_config = cfie_config.model_config
        cache_config = cfie_config.cache_config

        scheduler_config = cfie_config.scheduler_config
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen3Next currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )
        self.quant_config = cfie_config.quant_config

        super().__init__()
        self.config = config
        self.scheduler_config = scheduler_config
        self.model = Qwen3NextModel(
            cfie_config=cfie_config, prefix=maybe_prefix(prefix, "model")
        )

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        # 中文注释：Set MoE hyperparameters
        self.set_moe_parameters()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
            self,
            input_ids: torch.Tensor | None,
            positions: torch.Tensor,
            intermediate_tensors: IntermediateTensors | None = None,
            inputs_embeds: torch.Tensor | None = None,
            **kwargs: object,
    ):
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

        return hidden_states

    @classmethod
    def get_mamba_state_dtype_from_config(
            cls,
            cfie_config: "CfieConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            cfie_config.model_config.dtype,
            cfie_config.cache_config.mamba_cache_dtype,
            cfie_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
            cls, cfie_config: "CfieConfig"
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = cfie_config.parallel_config
        hf_config = cfie_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            cfie_config.speculative_config.num_speculative_tokens
            if cfie_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    def compute_logits(
            self,
            hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["mtp."],
        )
        return loader.load_weights(weights)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # CausalLM 外壳不自己生成 expert mapping，只把请求继续转给底层 model。
        return self.model.get_expert_mapping()


def gdn_attention_core(
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
        layer_name: str,
) -> None:
    """
    中文说明：Custom op for the core attention computation.
    中文说明：Only handles the convolution + recurrent attention part.
    中文说明：Input/output projections are handled outside this op.
    """
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self._forward_core(
        mixed_qkv=mixed_qkv,
        b=b,
        a=a,
        core_attn_out=core_attn_out,
    )


def gdn_attention_core_fake(
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
        layer_name: str,
) -> None:
    """中文说明：Fake implementation for torch.compile."""
    return


direct_register_custom_op(
    op_name="gdn_attention_core",
    op_func=gdn_attention_core,
    mutates_args=["core_attn_out"],
    fake_impl=gdn_attention_core_fake,
)


@triton.jit
def fused_gdn_gating_kernel(
        g,
        beta_output,
        A_log,
        a,
        b,
        dt_bias,
        seq_len,
        NUM_HEADS: tl.constexpr,
        beta: tl.constexpr,
        threshold: tl.constexpr,
        BLK_HEADS: tl.constexpr,
):
    i_b, i_s, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
    off = i_b * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off
    mask = head_off < NUM_HEADS
    blk_A_log = tl.load(A_log + head_off, mask=mask)
    blk_a = tl.load(a + off, mask=mask)
    blk_b = tl.load(b + off, mask=mask)
    blk_bias = tl.load(dt_bias + head_off, mask=mask)
    # 中文注释：If the model is loaded in fp16, without the .float() here, A might be -inf
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    softplus_x = tl.where(
        beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
    )
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)
    # 中文注释：compute beta_output = sigmoid(b)
    blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))
    tl.store(
        beta_output + off, blk_beta_output.to(beta_output.dtype.element_ty), mask=mask
    )


def fused_gdn_gating(
        A_log: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        dt_bias: torch.Tensor,
        beta: float = 1.0,
        threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    中文说明：Fused computation of g and beta for Gated Delta Net.
    中文说明：g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    中文说明：beta_output = b.sigmoid()
    中文说明：TODO maybe use torch.compile to replace this triton kernel
    """
    batch, num_heads = a.shape
    seq_len = 1
    if not HAS_TRITON:
        precompiled = _try_precompiled_fused_gdn_gating(
            A_log=A_log,
            a=a,
            b=b,
            dt_bias=dt_bias,
            beta=beta,
            threshold=threshold,
        )
        if precompiled is not None:
            return precompiled
        logger.warning_once(
            "Qwen3Next fused GDN gating is falling back to the PyTorch "
            "reference path because Triton runtime is unavailable."
        )
        x = a.to(torch.float32) + dt_bias.to(torch.float32)
        g = (-torch.exp(A_log.to(torch.float32)) * torch.nn.functional.softplus(
            x,
            beta=beta,
            threshold=threshold,
        )).unsqueeze(0)
        beta_output = torch.sigmoid(b.to(torch.float32)).to(b.dtype).unsqueeze(0)
        return g, beta_output
    grid = (batch, seq_len, triton.cdiv(num_heads, 8))
    g = torch.empty(1, batch, num_heads, dtype=torch.float32, device=a.device)
    beta_output = torch.empty(1, batch, num_heads, dtype=b.dtype, device=b.device)
    fused_gdn_gating_kernel[grid](
        g,
        beta_output,
        A_log,
        a,
        b,
        dt_bias,
        seq_len,
        num_heads,
        beta,
        threshold,
        8,
        num_warps=1,
    )
    return g, beta_output
