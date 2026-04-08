# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Iterable
from enum import Enum
from typing import Literal, cast, get_args, overload

import torch
from torch.nn.parameter import UninitializedParameter

import cfie.envs as envs
from cfie._aiter_ops import rocm_aiter_ops
from cfie.config import CfieConfig, get_current_cfie_config
from cfie.config.parallel import ExpertPlacementStrategy
from cfie.distributed import (
    get_dp_group,
    get_pcp_group,
    get_tensor_model_parallel_world_size,
)
from cfie.distributed.eplb.eplb_state import EplbLayerState, EplbState
from cfie.logger import init_logger
from cfie.model_executor.custom_op import CustomOp
from cfie.model_executor.layers.fused_moe.activation import MoEActivation
from cfie.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from cfie.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from cfie.model_executor.layers.fused_moe.fused_moe_modular_method import (
    FusedMoEModularMethod,
)
from cfie.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
    init_aiter_topK_meta_data,
)
from cfie.model_executor.layers.fused_moe.router.router_factory import (
    create_fused_moe_router,
)
from cfie.model_executor.layers.fused_moe.runner.default_moe_runner import (
    DefaultMoERunner,
)
from cfie.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from cfie.model_executor.layers.fused_moe.utils import (
    disable_inplace,
)
from cfie.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from cfie.offload.policy import get_moe_tiered_cache_plan
from cfie.platforms import current_platform
from cfie.utils.math_utils import round_up

logger = init_logger(__name__)


class FusedMoeWeightScaleSupported(Enum):
    TENSOR = "tensor"
    CHANNEL = "channel"
    GROUP = "group"
    BLOCK = "block"


def determine_expert_map(
        ep_size: int,
        ep_rank: int,
        global_num_experts: int,
        expert_placement_strategy: ExpertPlacementStrategy = "linear",
        num_fused_shared_experts: int = 0,
        return_expert_mask: bool = False,
) -> tuple[int, torch.Tensor | None, torch.Tensor | None]:
    # 按 EP 拓扑计算当前 rank 应持有多少个 experts，并建立 global -> local 的映射表。
    # experts 会尽量均匀分布到各个 rank；若有余数，则优先分配给前面的 rank。
    # 返回值依次是：当前 rank 的 expert 数、本地 expert 映射表，以及 AITER 路径可选的 expert_mask。
    assert ep_size > 0
    if ep_size == 1:
        return (global_num_experts, None, None)

    # 尽量把 experts 平均分配到每个 rank。
    base_experts = global_num_experts // ep_size
    remainder = global_num_experts % ep_size
    local_num_experts = base_experts + 1 if ep_rank < remainder else base_experts

    # 先创建一张默认全为 -1 的 global expert 映射表。
    expert_map = torch.full((global_num_experts,), -1, dtype=torch.int32)
    # 再把当前 rank 实际持有的 experts 映射到连续的本地下标。
    if expert_placement_strategy == "linear":
        start_idx = ep_rank * base_experts + min(ep_rank, remainder)
        expert_map[start_idx: start_idx + local_num_experts] = torch.arange(
            0, local_num_experts, dtype=torch.int32
        )
    elif expert_placement_strategy == "round_robin":
        local_log_experts = torch.arange(
            ep_rank, global_num_experts, ep_size, dtype=torch.int32
        )

        expert_map[local_log_experts] = torch.arange(
            0, local_num_experts, dtype=torch.int32
        )
    else:
        raise ValueError(
            "Unsupported expert placement strategy "
            f"'{expert_placement_strategy}', expected one of "
            f"{get_args(ExpertPlacementStrategy)}"
        )

    expert_mask = None
    if return_expert_mask:
        expert_mask = torch.ones(
            (global_num_experts + num_fused_shared_experts + 1,), dtype=torch.int32
        )
        expert_mask[-1] = 0
        expert_mask[:global_num_experts] = expert_map > -1
        expert_map = torch.cat(
            (
                expert_map,
                torch.tensor(
                    [local_num_experts + i for i in range(num_fused_shared_experts)],
                    dtype=torch.int32,
                ),
            ),
            dim=0,
        )

    return (local_num_experts, expert_map, expert_mask)


def determine_expert_placement_strategy(
        expert_placement_strategy: ExpertPlacementStrategy,
        moe_parallel_config: FusedMoEParallelConfig,
        num_expert_group: int | None,
        num_redundant_experts: int,
        enable_eplb: bool,
) -> ExpertPlacementStrategy:
    if expert_placement_strategy == "round_robin":
        round_robin_supported = (
                (num_expert_group is not None and num_expert_group > 1)
                and num_redundant_experts == 0
                and not enable_eplb
        )

        if not round_robin_supported:
            logger.warning(
                "Round-robin expert placement is only supported for "
                "models with multiple expert groups and no redundant "
                "experts. Falling back to linear expert placement."
            )
            return "linear"
        if (
                moe_parallel_config.use_all2all_kernels
                and not moe_parallel_config.use_deepep_ll_kernels
                and not moe_parallel_config.use_nixl_ep_kernels
        ):
            logger.warning(
                "Round-robin expert placement currently only supports "
                "the DeepEP low-latency or NIXL EP backend, but '%s' was configured. "
                "Falling back to linear expert placement.",
                moe_parallel_config.all2all_backend,
            )
            return "linear"

    return expert_placement_strategy


def get_compressed_expert_map(expert_map: torch.Tensor) -> str:
    """
    Compresses the expert map by removing any -1 entries.

    Args:
        expert_map (torch.Tensor): A tensor of shape (global_num_experts,)
            mapping from global to local index. Contains -1 for experts not
            assigned to the current rank.

    Returns:
        str: A string mapping from local to global index.
            Using str to support hashing for logging once only.
    """
    global_indices = torch.where(expert_map != -1)[0]
    local_indices = expert_map[global_indices]
    return ", ".join(
        f"{local_index.item()}->{global_index.item()}"
        for local_index, global_index in zip(local_indices, global_indices)
    )


def _should_enable_tiered_cache(
        quant_config: QuantizationConfig | None,
        layer: "FusedMoE",
        prefix: str,
) -> bool:
    # -------------------- 当前 helper 不需要直接访问 layer 对象 --------------------
    # 这里的判定暂时只依赖 quant_config 和层名前缀；
    # `layer` 参数保留下来主要是为了后续扩展或保持调用接口一致。
    del layer

    # -------------------- 先过滤掉明显不支持的量化配置 --------------------
    # 未启用量化时，没有必要开启 tiered cache。
    # 若量化配置带 desc_act，也直接关闭；当前这条路径只支持更窄的一组 GPTQ 场景。
    if quant_config is None or bool(getattr(quant_config, "desc_act", False)):
        return False

    # -------------------- 只允许 GPTQ Marlin 路径启用该能力 --------------------
    # 某些量化配置类会把量化方法名挂在类方法 `get_name()` 上，这里做一次安全读取。
    quant_name_getter = getattr(quant_config.__class__, "get_name", None)
    # 若类上确实有可调用的 `get_name`，就取其返回值；否则记为 None。
    quant_name = quant_name_getter() if callable(quant_name_getter) else None
    # 当前 tiered cache 仅在 gptq_marlin 量化路径下允许开启。
    if quant_name != "gptq_marlin":
        return False

    # -------------------- 读取 dynamic override，处理按层排除的例外 --------------------
    # 这里延迟导入，避免模块级循环依赖。
    from cfie.model_executor.layers.quantization.utils.gptq_utils import (
        get_dynamic_override,
    )

    # 查询当前层名前缀是否命中了 dynamic 规则。
    dynamic_override = get_dynamic_override(
        quant_config,
        layer_name=prefix,
    )
    if dynamic_override is False:
        # Qwen3.5 的 GPTQ checkpoint 往往会把 MTP 分支排除在 GPTQ 量化之外。
        # 但这里仍希望给 MTP drafter 的 MoE 层开启 tiered cache，
        # 否则这部分层会退回成整层 FP16 常驻，显存成本过高。
        # 因此：若 dynamic 明确判定“当前层不量化”，但层名前缀属于 MTP 分支，
        # 仍然返回 True，允许 tiered cache 继续生效。
        return prefix.startswith("mtp.") or ".mtp." in prefix
    # 走到这里说明：
    # - 量化配置存在
    # - 不是 desc_act
    # - 量化方法是 gptq_marlin
    # - 且没有命中必须禁用的 dynamic 排除规则
    # 因此允许当前层启用 tiered cache。
    return True


# TODO(rob): 后续把这段逻辑继续下沉到 kernel 内部。
def maybe_roundup_hidden_size(
        hidden_size: int,
        act_dtype: torch.dtype,
        moe_parallel_config: FusedMoEParallelConfig,
        is_lora_enabled: bool,
        model_type: str | None,
        is_mxfp4_quant: bool,
) -> int:
    """
    按当前层配置决定是否需要把 hidden size 向上对齐。

    参数：
    - `hidden_size`：当前层的 hidden size
    - `act_dtype`：激活张量的数据类型
    - `moe_parallel_config`：Fused MoE 的并行策略配置
    - `is_lora_enabled`：当前引擎是否启用了 LoRA
    - `model_type`：模型类型字符串，用于识别 `gpt_oss`
    - `is_mxfp4_quant`：当前层是否走 mxfp4 量化

    返回：
    - 若当前并行后端或量化后端要求补齐，则返回补齐后的 hidden size
    - 否则返回原始 hidden size
    """
    from cfie.model_executor.layers.fused_moe.all2all_utils import (
        maybe_roundup_layer_hidden_size,
    )

    # 先按 all2all / EP 后端的要求补齐 hidden size，确保通信缓冲区布局合法。
    hidden_size = maybe_roundup_layer_hidden_size(
        hidden_size, act_dtype, moe_parallel_config
    )

    # gpt_oss + mxfp4 场景还可能要求进一步做全局补齐，避免 EP 缓冲区分配不匹配。
    if model_type == "gpt_oss" and is_mxfp4_quant:
        from cfie.model_executor.layers.quantization.mxfp4 import (
            Mxfp4Backend,
            get_mxfp4_backend,
        )

        # mxfp4 的具体后端取决于平台和是否启用 LoRA。
        current_mxfp4_backend = get_mxfp4_backend(is_lora_enabled)

        # 部分后端要求 hidden size 按 128 对齐。
        if (
                current_mxfp4_backend == Mxfp4Backend.SM90_FI_MXFP4_BF16
                or current_mxfp4_backend == Mxfp4Backend.SM100_FI_MXFP4_MXFP8_CUTLASS
        ):
            hidden_size = round_up(hidden_size, 128)
        # 另一批后端要求 hidden size 按 64 对齐。
        elif (
                current_platform.is_rocm()
                or current_mxfp4_backend == Mxfp4Backend.SM100_FI_MXFP4_MXFP8_TRTLLM
                or current_mxfp4_backend == Mxfp4Backend.SM100_FI_MXFP4_BF16
                or current_mxfp4_backend == Mxfp4Backend.MARLIN
        ):
            hidden_size = round_up(hidden_size, 256)

    return hidden_size


# --8<-- [start:fused_moe]
@CustomOp.register("fused_moe")
class FusedMoE(CustomOp):
    # MoE 模型使用的 FusedMoE 层。
    # 这一层同时持有 gate_up_proj / w13 这类 MergedColumnParallel 权重，
    # 以及 down_proj / w2 这类 RowParallelLinear 权重。
    # 这里沿用 Mixtral 的 w1 / w2 / w3 命名习惯，模型侧若有不同命名，
    # 统一在各自的 load_weights 映射逻辑里处理。

    # --8<-- [end:fused_moe]

    def __init__(
            self,
            num_experts: int,  # Global number of experts
            top_k: int,
            hidden_size: int,
            intermediate_size: int,
            params_dtype: torch.dtype | None = None,
            reduce_results: bool = False,
            renormalize: bool = True,
            use_grouped_topk: bool = False,
            num_expert_group: int | None = None,
            topk_group: int | None = None,
            quant_config: QuantizationConfig | None = None,
            tp_size: int | None = None,
            ep_size: int | None = None,
            dp_size: int | None = None,
            pcp_size: int | None = None,
            prefix: str = "",
            custom_routing_function: Callable | None = None,
            scoring_func: str = "softmax",
            routed_scaling_factor: float = 1.0,
            e_score_correction_bias: torch.Tensor | None = None,
            apply_router_weight_on_input: bool = False,
            activation: str = "silu",
            is_act_and_mul: bool = True,
            enable_eplb: bool = False,
            num_redundant_experts: int = 0,
            has_bias: bool = False,
            is_sequence_parallel=False,
            expert_mapping: list[tuple[str, str, int, str]] | None = None,
            n_shared_experts: int | None = None,
            router_logits_dtype: torch.dtype | None = None,
            gate: torch.nn.Module | None = None,
            shared_experts: torch.nn.Module | None = None,
            routed_input_transform: torch.nn.Module | None = None,
    ):
        # -------------------- 基础父类初始化 --------------------
        # 先完成 nn.Module / CustomOp 基类初始化，确保后续 register_buffer 等接口可用。
        super().__init__()

        # -------------------- 保存外部注入子模块 --------------------
        # 保存外部注入的 gate / shared experts / routed 输入变换模块，runner 初始化时会直接复用。
        self._gate = gate
        self._shared_experts = shared_experts
        self._routed_input_transform = routed_input_transform

        # -------------------- 解析参数 dtype 与全局配置 --------------------
        # 若调用方未显式指定参数 dtype，则沿用当前默认 dtype。
        if params_dtype is None:
            # 这里读取 PyTorch 当前默认 dtype，作为本层参数创建时的默认精度。
            params_dtype = torch.get_default_dtype()
        # 把最终确定的参数 dtype 保存到层对象上，后续创建权重时直接复用。
        self.params_dtype = params_dtype

        # 读取当前全局 CFIE 配置，后续并行、kernel、tiered cache 都从这里派生。
        cfie_config = get_current_cfie_config()
        # 保存当前层构造时绑定的 CFIE 全局配置。
        self.cfie_config = cfie_config

        # 优先使用 model_config 里声明的模型 dtype 作为 MoE 输入激活 dtype。
        if cfie_config.model_config is not None:
            # 如果模型配置已存在，则输入激活 dtype 以模型配置为准。
            moe_in_dtype = cfie_config.model_config.dtype
        else:
            # 若模型配置尚不可用，则退回到参数 dtype。
            moe_in_dtype = params_dtype

        # -------------------- 推导并行规模 --------------------
        # 若构造参数未覆盖，则从当前并行组读取 TP / DP / PCP 实际规模。
        tp_size_ = (
            tp_size if tp_size is not None else get_tensor_model_parallel_world_size()
        )
        # DP 规模优先用显式传参，否则读取当前 DP group 的 world size。
        dp_size_ = dp_size if dp_size is not None else get_dp_group().world_size
        # PCP 规模优先用显式传参，否则读取当前 PCP group 的 world size。
        pcp_size_ = pcp_size if pcp_size is not None else get_pcp_group().world_size

        # Sequence parallel 打开时，MoE 侧需要把 SP 规模写入并行配置。
        # 记录当前层是否启用了 sequence parallel。
        self.is_sequence_parallel = is_sequence_parallel
        # 若启用 SP，则 SP 规模与 TP 对齐；否则固定视为 1。
        self.sp_size = tp_size_ if is_sequence_parallel else 1

        # 统一构造 MoE 并行配置对象，后续 TP / EP / DP / backend 判断都走这里。
        self.moe_parallel_config: FusedMoEParallelConfig = FusedMoEParallelConfig.make(
            tp_size_=tp_size_,  # 根据是否开启EP,决定tp_size
            pcp_size_=pcp_size_,
            dp_size_=dp_size_,
            sp_size_=self.sp_size,
            cfie_parallel_config=cfie_config.parallel_config,
        )

        # 校验并行配置对象内部记录的 SP 开关与构造参数一致，避免后续路由/执行行为错位。
        assert self.moe_parallel_config.is_sequence_parallel == is_sequence_parallel

        # -------------------- 记录 expert 基本规模 --------------------
        # global_num_experts 包含冗余副本；logical_num_experts 只表示逻辑 expert 数。
        self.global_num_experts = num_experts + num_redundant_experts
        self.logical_num_experts = num_experts

        # 保存可选的权重名映射规则，load_weights 时会按这份映射把 checkpoint 权重导入本层。
        self.expert_mapping = expert_mapping

        # -------------------- 注册到编译期静态上下文 --------------------
        # 读取 compilation 配置，用于把当前层登记到静态 forward 上下文中。
        compilation_config = cfie_config.compilation_config

        # 同一个 prefix 只能注册一次，避免静态上下文里出现重名层。
        if prefix in compilation_config.static_forward_context:
            raise ValueError("Duplicate layer name: {}".format(prefix))

        # 把当前层对象挂到 prefix 对应的静态上下文条目。
        compilation_config.static_forward_context[prefix] = self

        # 把当前层名追加到全量 MoE 层列表，供编译/分析流程统一遍历。
        compilation_config.static_all_moe_layers.append(prefix)

        # 保存层名前缀，后续日志、量化和 controller 挂载都依赖它。
        self.layer_name = prefix

        # -------------------- EPLB / expert 放置 / ROCm AITER 能力探测 --------------------
        # EPLB 会在运行时根据负载更新 expert 放置；其层内状态先挂在这里。
        self.enable_eplb = enable_eplb

        # 创建 EPLB 层级状态对象，供 router / 运行期负载均衡使用。
        self.eplb_state = EplbLayerState()

        # 默认采用全局并行配置里的 expert 放置策略。
        self.expert_placement_strategy: ExpertPlacementStrategy = (
            cfie_config.parallel_config.expert_placement_strategy
        )

        # 检查当前环境是否启用了 ROCm AITER fused moe kernel，同时要求激活路径是 act-and-mul 形态。
        self.rocm_aiter_fmoe_enabled = (
                rocm_aiter_ops.is_fused_moe_enabled() and is_act_and_mul
        )

        # 检查是否还能额外启用 shared expert 融合能力。
        self.aiter_fmoe_shared_expert_enabled = (
                rocm_aiter_ops.is_fusion_moe_shared_experts_enabled() and is_act_and_mul
        )

        # 若外部声明了 shared experts，且当前平台支持融合 shared expert，则记录其数量；否则记为 0。
        self.num_fused_shared_experts = (
            n_shared_experts
            if n_shared_experts is not None and self.aiter_fmoe_shared_expert_enabled
            else 0
        )

        # 若外部传了 shared experts 数量，但当前平台并不支持这条能力，则直接报错阻止继续初始化。
        if (
                not self.aiter_fmoe_shared_expert_enabled
                and self.num_fused_shared_experts != 0
        ):
            raise ValueError(
                "n_shared_experts is only supported on ROCm aiter when "
                "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS is enabled"
            )

        # -------------------- 计算本地 expert 布局 / tiered cache 布局 --------------------

        # 这里建立 expert 的本地布局信息。
        # 在普通 EP 路径下，_expert_map 表示“全局 expert id -> 当前 rank 本地物理 expert 槽位”。
        # 在 CFIE tiered cache 路径下，这张表还承担“哪些 experts 当前已常驻 GPU slots”的语义。
        if self.use_ep:
            # EPLB 当前只支持 expert 在 EP ranks 间平均分布，因此必须能整除。
            if self.enable_eplb:
                assert self.global_num_experts % self.ep_size == 0, (
                    "EPLB currently only supports even distribution of "
                    "experts across ranks."
                )
            else:
                # 冗余 expert 当前只允许和 EPLB 一起使用。
                assert num_redundant_experts == 0, (
                    "Redundant experts are only supported with EPLB."
                )

            # 根据当前并行后端、expert group 和 EPLB 能力，决定最终采用哪种放置策略。
            self.expert_placement_strategy = determine_expert_placement_strategy(
                expert_placement_strategy=self.expert_placement_strategy,
                moe_parallel_config=self.moe_parallel_config,
                num_expert_group=num_expert_group,
                num_redundant_experts=num_redundant_experts,
                enable_eplb=self.enable_eplb,
            )

            # 提前声明 _expert_map 类型，方便静态检查器理解 register_buffer 后的属性类型。
            self._expert_map: torch.Tensor | None

            # 根据 EP 拓扑为当前 rank 计算本地 expert 数、expert 映射表和可选的 expert mask。
            local_num_experts, expert_map, expert_mask = determine_expert_map(
                ep_size=self.ep_size,
                ep_rank=self.ep_rank,
                global_num_experts=self.global_num_experts,
                expert_placement_strategy=self.expert_placement_strategy,
                num_fused_shared_experts=self.num_fused_shared_experts,
                return_expert_mask=self.rocm_aiter_fmoe_enabled,
            )

            # 保存当前 rank 实际持有的本地 expert 数。
            self.local_num_experts = local_num_experts

            # 把 expert_map 注册成 buffer，使其随 device / state_dict 管理。
            self.register_buffer("_expert_map", expert_map)

            # 把 expert_mask 注册成 buffer，供 AITER fused moe 路径使用。
            self.register_buffer("expert_mask", expert_mask)
            # 若需要额外的路由索引表，则在这里一并初始化。
            self._maybe_init_expert_routing_tables()
            # 打一条一次性日志，把当前 rank 的 expert 放置结果打印出来。
            logger.info_once(
                "[EP Rank %s/%s] Expert parallelism is enabled. Expert "
                "placement strategy: %s. Local/global"
                " number of experts: %s/%s. Experts local to global index map:"
                " %s.",
                self.ep_rank,
                self.ep_size,
                self.expert_placement_strategy,
                self.local_num_experts,
                self.global_num_experts,
                get_compressed_expert_map(self._expert_map),
            )
        else:
            # CFIE 的 tiered cache 目前只在非 EP 路径下接管 expert 常驻布局。

            # 读取当前层可用的 tiered cache 规划结果。
            tiered_cache_plan = get_moe_tiered_cache_plan(cfie_config)

            # 只有规划结果有效、规划本身启用、并且量化路径允许时，才真正启用 tiered cache。
            tiered_cache_enabled = (
                    isinstance(tiered_cache_plan, dict)
                    and bool(tiered_cache_plan.get("enabled", False))
                    and _should_enable_tiered_cache(
                quant_config,
                self,
                prefix,
            ))

            if tiered_cache_enabled:
                # initial_gpu_experts 是启动期允许直接落到 GPU 的 experts 集合。
                # 这里把配置中的初始常驻 expert id 统一转成 int tuple。
                initial_gpu_experts = tuple(
                    int(expert_id)
                    for expert_id in tiered_cache_plan.get("initial_gpu_experts", ())
                )

                # local_num_experts 不再等于全量 experts，而是“当前层在 GPU 上真实创建多少 slots”。
                # gpu_slots_per_layer 表示这一层在 GPU 侧允许创建的 expert resident 槽位数量。
                self.local_num_experts = min(
                    int(tiered_cache_plan.get("gpu_slots_per_layer", self.global_num_experts)),
                    self.global_num_experts,
                )

                # 默认把所有全局 experts 标成 -1，表示当前没有映射到 GPU resident slot。
                expert_map = torch.full(
                    (self.global_num_experts,),
                    -1,
                    dtype=torch.int32,
                )

                # 只给启动期常驻的 experts 分配 slot；其余 experts 在 checkpoint 加载时会被跳过。
                for slot, expert_id in enumerate(initial_gpu_experts[: self.local_num_experts]):
                    # 把“全局 expert id -> 当前 GPU resident slot id”的映射写入表中。
                    expert_map[expert_id] = slot

                # 把 tiered cache 版本的 expert_map 注册成 buffer，供后续权重加载和运行时 controller 使用。
                self.register_buffer("_expert_map", expert_map)

                # tiered cache 非 EP 路径下目前不使用 expert_mask。
                self.expert_mask = None

                # 这两个字段标记当前层启用了 CFIE 的分层 expert cache，并保留规划结果。
                self._cfie_tiered_cache_enabled = True
                self._cfie_tiered_cache_plan = tiered_cache_plan
            else:
                # 未启用 tiered cache 时，本地直接创建并持有全量 experts。
                self.local_num_experts, self._expert_map, self.expert_mask = (
                    self.global_num_experts,
                    None,
                    None,
                )

                # 同步把 CFIE tiered cache 标记清空，表明当前层走的是全量常驻路径。
                self._cfie_tiered_cache_enabled = False
                self._cfie_tiered_cache_plan = None

        # -------------------- 补齐 CFIE tiered cache 相关占位字段 --------------------
        # 无论上面走哪条分支，都保证这几个 CFIE 专用字段一定存在。
        if not hasattr(self, "_cfie_tiered_cache_enabled"):
            # 某些旧路径可能没有显式写入这两个字段，这里统一补默认值。
            self._cfie_tiered_cache_enabled = False
            self._cfie_tiered_cache_plan = None

        # controller 会在模型权重全部加载后由 offload 侧挂进来，这里先占位。
        self._cfie_tiered_cache_controller = None

        # -------------------- 记录基本运行超参数 --------------------
        # 保存每个 token 要选出的 top-k experts 数。
        self.top_k = top_k

        # shared experts 需要提前初始化 top-k 元数据缓冲区。
        self._init_aiter_shared_experts_topK_buffer(
            cfie_config=cfie_config, dp_size=dp_size_
        )

        # 若启用了 EP + AITER fused moe，则 expert_mask 必须严格是 0/1 mask。
        if self.use_ep and self.rocm_aiter_fmoe_enabled:
            assert self.expert_mask is None or torch.all(
                (expert_mask == 0) | (expert_mask == 1)
            ), "Aiter Fused MoE kernel only supports expert_map with 0 and 1s."

        # 中间维度会按 TP 切分，因此必须能被 TP 整除。
        assert intermediate_size % self.tp_size == 0

        # 记录当前 TP 分片后的每卡中间维度。
        self.intermediate_size_per_partition = intermediate_size // self.tp_size

        # 保存是否需要对输出执行 all-reduce。
        self.reduce_results = reduce_results  # False

        # 保存路由权重是否需要在 kernel 前后重新归一化。
        self.renormalize = renormalize  # True

        # -------------------- 保存路由相关配置 --------------------
        # 下面这些路由字段主要服务 monolithic / router 相关执行路径。
        # 记录是否启用 grouped top-k 路由。
        self.use_grouped_topk = use_grouped_topk
        if self.use_grouped_topk:  # False
            # grouped top-k 模式下必须显式给出 expert group 数和每组 top-k 数。
            assert num_expert_group is not None and topk_group is not None

        # 保存 expert group 数。
        self.num_expert_group = num_expert_group  # None

        # 保存每个 expert group 内实际选择的 top-k。
        self.topk_group = topk_group  # None

        # 保存可选的自定义路由函数。
        self.custom_routing_function = custom_routing_function  # None

        # 保存路由打分函数名，例如 softmax。
        self.scoring_func = scoring_func  # softmax

        # 保存 routed experts 输出的缩放因子。
        self.routed_scaling_factor = routed_scaling_factor  # 1.0

        # 保存可选的 e-score 校正 bias 张量。
        self.e_score_correction_bias = e_score_correction_bias  # None

        # 保存是否在 routed 路径前先把路由权重乘到输入上。
        self.apply_router_weight_on_input = apply_router_weight_on_input  # False

        # 把字符串形式的激活函数名转换成内部 MoEActivation 枚举。
        self.activation = MoEActivation.from_str(activation)  # silu

        # -------------------- 构造路由器 --------------------
        # 路由器负责根据 router logits 为每个 token 产出 top-k experts。
        self.router = create_fused_moe_router(
            top_k=top_k,
            global_num_experts=self.global_num_experts,
            eplb_state=self.eplb_state,
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            num_fused_shared_experts=self.num_fused_shared_experts,
            enable_eplb=enable_eplb,
            indices_type_getter=lambda: self.quant_method.topk_indices_dtype,
        )
        # 保存当前路由器最终采用的 routing method 类型。
        self.routing_method_type: RoutingMethodType = self.router.routing_method_type

        # -------------------- 对齐 hidden_size 并构造 moe_config --------------------
        # 某些 kernel / 量化路径要求 hidden_size 对齐，这里先做一次规整。
        # 先保留未补齐前的原始 hidden_size，后面创建量化权重时仍会用到。
        unpadded_hidden_size = hidden_size
        # 若模型配置存在，则读取 HF config 里的 model_type，供 kernel 决策使用。
        self.model_type = (
            self.cfie_config.model_config.hf_config.model_type
            if self.cfie_config.model_config is not None
            else None
        )
        # 根据 dtype、并行配置、LoRA、量化方式等条件，对 hidden_size 做必要的向上对齐。
        hidden_size = maybe_roundup_hidden_size(
            hidden_size=hidden_size,
            act_dtype=moe_in_dtype,
            moe_parallel_config=self.moe_parallel_config,
            is_lora_enabled=cfie_config.lora_config is not None,
            model_type=self.model_type,
            is_mxfp4_quant=(
                    quant_config is not None and quant_config.is_mxfp4_quant(prefix, self)
            ),  # False
        )
        # 保存最终用于 kernel / 权重布局的 hidden_size。
        self.hidden_size = hidden_size

        # moe_config 汇总 fused moe 执行所需的结构、并行、dtype 和 backend 信息。
        self.moe_config: FusedMoEConfig = FusedMoEConfig(
            num_experts=self.global_num_experts,
            experts_per_token=top_k,
            hidden_dim=hidden_size,
            intermediate_size_per_partition=self.intermediate_size_per_partition,
            num_local_experts=self.local_num_experts,
            num_logical_experts=self.logical_num_experts,
            moe_parallel_config=self.moe_parallel_config,
            in_dtype=moe_in_dtype,
            moe_backend=cfie_config.kernel_config.moe_backend,
            router_logits_dtype=router_logits_dtype,
            max_num_tokens=envs.VLLM_MOE_DP_CHUNK_SIZE,
            has_bias=has_bias,
            is_act_and_mul=is_act_and_mul,
            is_lora_enabled=cfie_config.lora_config is not None,
            activation=self.activation,
            device=cfie_config.device_config.device,
            routing_method=self.routing_method_type,
            # TODO: in_dtype == out_dtype?
            disable_inplace=disable_inplace() or self._shared_experts is not None,
        )
        # Mori kernel 目前依赖 ROCm AITER fused moe，并且不支持 fusion shared experts。
        if self.moe_config.use_mori_kernels:
            assert self.rocm_aiter_fmoe_enabled, (
                "Mori needs to be used with aiter fused_moe for now."
            )
            assert not self.aiter_fmoe_shared_expert_enabled, (
                "Mori does not support fusion shared expert now. "
                "Turn it off by setting VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0"
            )

        # -------------------- 解析量化方法 --------------------
        # 保存外部传入的量化配置对象。
        self.quant_config = quant_config

        def _get_quant_method() -> FusedMoEMethodBase:
            # 先准备一个空值，表示“尚未从 quant_config 拿到具体量化方法”。
            quant_method = None
            if self.quant_config is not None:
                # 若存在量化配置，则按当前层和 prefix 解析对应的量化方法实现。
                quant_method = self.quant_config.get_quant_method(self, prefix)
            if quant_method is None:
                # 若未命中任何量化方法，则退回到非量化 fused moe 实现。
                quant_method = UnquantizedFusedMoEMethod(self.moe_config)
            # 这里强制保证最终结果一定是 FusedMoEMethodBase 子类。
            assert isinstance(quant_method, FusedMoEMethodBase)
            # 返回最终确定的量化方法对象。
            return quant_method

        # 量化方法会参考 local_num_experts 决定权重布局，因此必须放在映射初始化之后。
        # 根据当前层配置、量化配置和 local expert 布局，解析最终量化方法。
        self.quant_method: FusedMoEMethodBase = _get_quant_method()

        # 非 act-and-mul 模式目前只支持 CUDA/ROCm 类平台，其它平台直接拒绝。
        if not self.moe_config.is_act_and_mul and not current_platform.is_cuda_alike():
            raise NotImplementedError(
                "is_act_and_mul=False is supported only for CUDA and ROCm for now"
            )

        # 若开启 EPLB，则所选量化方法也必须显式支持 EPLB。
        if self.enable_eplb and not self.quant_method.supports_eplb:
            raise NotImplementedError(
                f"EPLB is not supported {self.quant_method.__class__.__name__}."
            )

        # -------------------- 创建权重张量 --------------------
        # create_weights 会真正创建 experts.w13_ / experts.w2_ 以及量化附属参数的张量。
        moe_quant_params = {
            # 当前设备/当前层实际需要创建的 expert 槽位数。
            "num_experts": self.local_num_experts,
            # 对齐后的 hidden_size，用于实际张量布局。
            "hidden_size": hidden_size,
            # 未对齐的 hidden_size，某些量化/重排逻辑仍需参考。
            "unpadded_hidden_size": unpadded_hidden_size,
            # TP 切分后的每分区中间维度。
            "intermediate_size_per_partition": self.intermediate_size_per_partition,
            # 参数 dtype。
            "params_dtype": params_dtype,
            # 权重导入时统一复用本层的 weight_loader。
            "weight_loader": self.weight_loader,
            # 全局 expert 总数，供量化方法做全局到本地的映射判断。
            "global_num_experts": self.global_num_experts,
        }
        # need full intermediate size pre-sharding for WNA16 act order
        if self.quant_method.__class__.__name__ in (
                "GPTQMarlinMoEMethod",
                "CompressedTensorsWNA16MarlinMoEMethod",
                "CompressedTensorsWNA16MoEMethod",
        ):
            # 某些 WNA16 / Marlin 量化路径在切分前仍需要完整 intermediate_size。
            moe_quant_params["intermediate_size_full"] = intermediate_size

        # 当前 Qwen3.5-122B-A10B-GPTQ-Int4 routed experts 通常会在这里走：
        # self.quant_method = GPTQMarlinMoEMethod
        #
        # 当前关键参数上下文：
        # - hidden_size = 3072
        # - intermediate_size = 1024
        # - num_experts(逻辑) = 256
        # - top_k = 8
        #
        # 但这里传给 create_weights 的 num_experts 不是全局 256，
        # 而是 self.local_num_experts，即“当前层当前设备上真实创建多少 expert slots”。
        # 在 tiered cache 打开时，它会显著小于 256，从而避免一层一次性把全部专家塞进显存。
        # 真正按当前量化方法创建专家权重和量化附属张量。
        self.quant_method.create_weights(layer=self, **moe_quant_params)
        # base_quant_method 保留最初的量化方法，后续 modular kernel 包装时还会用到。
        self.base_quant_method = self.quant_method

        # -------------------- 计算 shared expert overlap 策略并初始化 runner --------------------
        # 读取当前 all2all backend，后面判断 shared expert overlap 是否安全启用。
        backend = self.moe_parallel_config.all2all_backend
        # 只有后端允许、当前存在 shared experts 时，才启用 overlap。
        self.use_overlapped = (
                not (
                        (self.enable_eplb and backend != "allgather_reducescatter")
                        or self.moe_parallel_config.use_fi_all2allv_kernels
                )
                and self._shared_experts is not None
        )  # True

        # FusedMoE 本体主要负责元数据和权重；真正执行统一下沉到 runner。
        # 构造默认 runner，后续 forward 会通过 runner 进入具体执行路径。
        self.runner = self._init_runner()

    def _init_runner(self):
        # runner 持有运行时执行入口，并在 CFIE 路径下继续衔接 tiered cache controller。
        return DefaultMoERunner(
            layer=self,
            moe_config=self.moe_config,
            router=self.router,
            routed_input_transform=self._routed_input_transform,
            gate=self.gate,
            shared_experts=self.shared_experts,
            quant_method=self.quant_method,
            reduce_results=self.reduce_results,
            enable_dbo=self.cfie_config.parallel_config.enable_dbo,
        )

    def _replace_quant_method(self, mk: FusedMoEMethodBase):
        # 替换量化方法后，需要连 runner 一起重建，保证执行路径拿到的是新方法。
        self.quant_method = mk
        self.runner = self._init_runner()

    def maybe_init_modular_kernel(self) -> None:
        # 若量化方法自己已经内建 MK，或本身就是 monolithic 路径，则无需再包装。
        if self.quant_method.supports_internal_mk or self.quant_method.is_monolithic:
            return None

        # 某些 prepare/finalize 逻辑依赖量化配置和 expert routing tables。
        self.ensure_moe_quant_config_init()
        routing_tables = self._maybe_init_expert_routing_tables()
        prepare_finalize = self.base_quant_method.maybe_make_prepare_finalize(
            routing_tables=routing_tables
        )
        if prepare_finalize is not None:
            # 这里把“基础量化方法 + prepare/finalize”封装成 modular method。
            logger.debug(
                "%s for %s(%s)", prepare_finalize.__class__.__name__, self, id(self)
            )
            self._replace_quant_method(
                FusedMoEModularMethod.make(
                    self,
                    self.base_quant_method,
                    prepare_finalize,
                    self.shared_experts,
                    inplace=not self.moe_config.disable_inplace,
                )
            )

    @property
    def shared_experts(self) -> torch.nn.Module | None:
        # 只有 overlap 安全开启时，shared experts 才暴露给 runner 参与融合执行。
        return self._shared_experts if self.use_overlapped else None

    @property
    def layer_id(self):
        # 从层名前缀里解析出 transformer block 的层号。
        # 延迟导入，避免循环依赖。
        from cfie.model_executor.models.utils import extract_layer_index

        return extract_layer_index(self.layer_name)

    @property
    def gate(self) -> torch.nn.Module | None:
        # 与 shared_experts 同理，只有允许 overlap 时才把 gate 暴露给 runner。
        return self._gate if self.use_overlapped else None

    @property
    def tp_size(self):
        # 透传 TP 大小，避免调用方反复拆 moe_parallel_config。
        return self.moe_parallel_config.tp_size

    @property
    def ep_size(self):
        # 透传 EP 大小。
        return self.moe_parallel_config.ep_size

    @property
    def tp_rank(self):
        # 透传当前 TP rank。
        return self.moe_parallel_config.tp_rank

    @property
    def ep_rank(self):
        # 透传当前 EP rank。
        return self.moe_parallel_config.ep_rank

    @property
    def use_ep(self):
        # 透传当前层是否启用 expert parallel。
        return self.moe_parallel_config.use_ep

    @property
    def is_internal_router(self) -> bool:
        # gate 不为空时，说明本层自己持有内部路由器前置模块。
        # 默认会在 FusedMoE forward 之前先调用 router / gate。
        return self.gate is not None

    def _maybe_init_expert_routing_tables(
            self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        # 这组表服务 EP round-robin 布局下的索引换算，不是 CFIE 的运行时 tiered cache。
        # routing_tables 目前只在 round-robin expert placement 下使用，
        # 且仅对应 DeepEP-ll 或 NIXL 的 EP all2all 后端。
        if self.expert_placement_strategy != "round_robin" or (
                not self.moe_parallel_config.use_deepep_ll_kernels
                and not self.moe_parallel_config.use_nixl_ep_kernels
        ):
            return None

        # 已经初始化过时直接复用，避免重复构造和重复注册 buffer。
        if hasattr(self, "expert_global_to_physical"):
            return cast(
                tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                (
                    self.expert_global_to_physical,
                    self.expert_physical_to_global,
                    self.expert_local_to_global,
                ),
            )

        if self._expert_map is None:
            return None

        # 生成全局 expert、物理槽位、当前 rank 本地 expert 三者之间的对应关系。
        routing_tables = self.ensure_round_robin_expert_routing_tables(
            global_num_experts=self.global_num_experts,
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
            local_num_experts=self.local_num_experts,
            device=self._expert_map.device,
        )

        global_to_physical, physical_to_global, local_global = routing_tables
        # 注册成 buffer 后，kernel 和通信路径都能直接按名字访问这些查表张量。
        self.register_buffer("expert_global_to_physical", global_to_physical)
        self.register_buffer("expert_physical_to_global", physical_to_global)
        self.register_buffer("expert_local_to_global", local_global)

        return routing_tables

    @staticmethod
    def ensure_round_robin_expert_routing_tables(
            global_num_experts: int,
            ep_size: int,
            ep_rank: int,
            local_num_experts: int,
            device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 让查表张量与目标设备对齐，避免后续再搬运一次。
        device_kwargs = {"device": device} if device is not None else {}
        # 全局 expert id 序列：[0, 1, ..., global_num_experts - 1]。
        global_indices = torch.arange(
            global_num_experts, dtype=torch.long, **device_kwargs
        )
        # round-robin 放置下，expert 的归属 rank 就是 global_id % ep_size。
        owner = torch.remainder(global_indices, ep_size)
        # 同一个 rank 内的本地下标等于 global_id // ep_size。
        local_index = torch.div(global_indices, ep_size, rounding_mode="floor")
        # base / remainder 用于处理不能整除时，各 rank 实际持有 expert 数不同的情况。
        base = global_num_experts // ep_size
        remainder = global_num_experts % ep_size
        # 先算出每个 owner rank 在物理平铺布局中的起始偏移。
        physical_offset = owner * base
        if remainder > 0:
            remainder_tensor = torch.tensor(
                remainder, dtype=torch.long, **device_kwargs
            )
            # 前 remainder 个 rank 会多拿一个 expert，因此偏移还要再补一格。
            physical_offset = physical_offset + torch.minimum(owner, remainder_tensor)

        # 全局 expert id -> 物理槽位 id。
        global_to_physical = physical_offset + local_index
        # 物理槽位 id -> 全局 expert id，便于反查。
        physical_to_global = torch.empty_like(global_to_physical)
        physical_to_global[global_to_physical] = global_indices

        # 当前 rank 实际持有的全局 expert 序列。
        local_global = torch.arange(
            ep_rank,
            global_num_experts,
            ep_size,
            dtype=torch.long,
            **device_kwargs,
        )
        if local_global.numel() != local_num_experts:
            # shared experts / 冗余 expert 改变本地数量时，这里按真实长度截断。
            local_global = local_global[:local_num_experts]

        return (global_to_physical, physical_to_global, local_global)

    def update_expert_map(self):
        # 当 EPLB 或并行拓扑发生变化后，刷新本层的本地 expert 映射表。
        # ep_size and ep_rank should already be updated
        assert self._expert_map is not None
        with self._expert_map.device:
            local_num_experts, expert_map, expert_mask = determine_expert_map(
                ep_size=self.ep_size,
                ep_rank=self.ep_rank,
                global_num_experts=self.global_num_experts,
                expert_placement_strategy=self.expert_placement_strategy,
                num_fused_shared_experts=self.num_fused_shared_experts,
                return_expert_mask=self.rocm_aiter_fmoe_enabled,
            )
            self.local_num_experts = local_num_experts
            # 重新注册 buffer，让后续权重访问和 kernel 查表看到最新映射。
            self.register_buffer("_expert_map", expert_map)
            self.register_buffer("expert_mask", expert_mask)
            self._maybe_init_expert_routing_tables()
            if self.aiter_fmoe_shared_expert_enabled:
                # AITER shared experts 的 top-k 元数据也要跟着新的映射重建。
                self._init_aiter_shared_experts_topK_buffer(
                    cfie_config=get_current_cfie_config(),
                    dp_size=get_dp_group().world_size,
                )

    def _load_per_tensor_weight_scale(
            self,
            shard_id: str,
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            expert_id: int,
    ):
        # per-tensor 量化场景下，每个 expert / shard 只对应一个标量 scale。
        param_data = param.data
        # 这是 per-tensor weight quantization 的装载路径。
        if shard_id in ("w1", "w3"):
            # w1 / w3 的 scale 需要单独保留，因为权重装载后还要再次量化。
            # w13 把 w1 / w3 合并存储，因此这里要先区分写入前半还是后半。
            idx = 0 if shard_id == "w1" else 1
            param_data[expert_id][idx] = loaded_weight
        # down_proj 属于 RowParallel 路径。
        elif shard_id == "w2":
            # w2 没有合并 shard，直接覆盖对应 expert 的 scale。
            param_data[expert_id] = loaded_weight

    def _load_combined_w13_weight_scale(
            self,
            shard_dim: int,
            loaded_weight: torch.Tensor,
            param: torch.Tensor,
            tp_rank: int,
    ):
        # w13 的 scale 可能把 w1 与 w3 合并存在一个张量里；
        # 这里先按 TP rank 切出当前分片，再整体拷贝到参数视图。
        shard_size = param.shape[shard_dim]
        loaded_weight = loaded_weight.narrow(
            shard_dim, shard_size * tp_rank, shard_size
        )
        param.copy_(loaded_weight)

    def _load_model_weight_or_group_weight_scale(
            self,
            shard_dim: int,
            expert_data: torch.Tensor,
            shard_id: str,
            loaded_weight: torch.Tensor,
            tp_rank: int,
            load_full_w2: bool = False,
    ):
        # 统一装载 group quant 的 weight scale 或普通模型权重。
        # shard_dim 指定 TP 切分维度，shard_id 表示当前落的是 w1 / w2 / w3 哪一支。
        # load_full_w2 用于控制 w2 是否保留完整权重而不做 TP 分片。
        if shard_id == "w2":
            # w2 是 RowParallel 方向，必要时可选择保留完整权重而不切 TP 分片。
            # act-order / g_idx 场景下，w2 scale 可按 load_full_w2 要求保持不分片。
            self._load_w2(
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
                load_full=load_full_w2,
            )
        elif shard_id in ("w1", "w3"):
            # w1 / w3 都共享 w13 的装载逻辑，只是落到前半或后半不同。
            self._load_w13(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )

    def _load_per_channel_weight_scale(
            self,
            expert_data: torch.Tensor,
            shard_dim: int,
            shard_id: str,
            loaded_weight: torch.Tensor,
            tp_rank: int,
    ):
        # 入口：装载 per-channel weight scale。
        # per-channel scale 与真实模型权重使用同一套 TP 切分规则。
        if shard_id == "w2":
            expert_data.copy_(loaded_weight)
        elif shard_id in ("w1", "w3"):
            self._load_w13(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )

    def _load_w13(
            self,
            expert_data: torch.Tensor,
            shard_dim: int,
            shard_id: str,
            loaded_weight: torch.Tensor,
            tp_rank: int,
            load_full: bool = False,
    ):
        # 入口：装载 w13。
        # w13 是 gate_proj 与 up_proj 的合并权重，按列并行语义在输出维切分。
        if self.moe_config.is_act_and_mul:  # 默认
            # gated activation 场景下，w13 的同一维度前后两半分别是 w1 / w3。
            shard_size = expert_data.shape[shard_dim] // 2
        else:
            shard_size = expert_data.shape[shard_dim]
        # 非 full-load 且 loaded_weight 不是标量时，只截取当前 TP rank 负责的那一片。
        if not load_full and loaded_weight.ndim > 0:  # 默认
            # 按当前 TP rank 只截取自己负责的输出维分片。
            loaded_weight = loaded_weight.narrow(
                shard_dim, shard_size * tp_rank, shard_size
            )
        # w1 对应 w13 的前半段。
        if shard_id == "w1":
            expert_data = expert_data.narrow(shard_dim, 0, shard_size)
        # w3 对应 w13 的后半段。
        else:
            assert shard_id == "w3"
            expert_data = expert_data.narrow(shard_dim, shard_size, shard_size)
        expert_data.copy_(loaded_weight)

    def _load_w2(
            self,
            expert_data: torch.Tensor,
            shard_dim: int,
            loaded_weight: torch.Tensor,
            tp_rank: int,
            load_full: bool = False,
    ):
        # 入口：装载 w2。
        # w2 对应 down_proj，按行并行语义在输入维切分。
        shard_size = expert_data.shape[shard_dim]
        # 非 full-load 且 loaded_weight 不是标量时，只截取当前 TP rank 负责的那一片。
        if not load_full and loaded_weight.ndim > 0:
            # 按当前 TP rank 只装入自己负责的输入维片段。
            loaded_weight = loaded_weight.narrow(
                shard_dim, shard_size * tp_rank, shard_size
            )
        # w2 没有像 w13 那样的双分支复用，直接整段写入即可。
        expert_data.copy_(loaded_weight)

    def _load_single_value(
            self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, expert_id: int
    ):
        # 入口：装载单值元数据。
        # 这类参数通常是 input_scale、weight_shape 等按 expert 保存的标量。
        param_data = param.data
        param_data[expert_id] = loaded_weight

    def _load_g_idx(
            self,
            shard_id: str,
            expert_data: torch.Tensor,
            shard_dim: int,
            loaded_weight: torch.Tensor,
            tp_rank: int,
    ):
        # 入口：装载 g_idx。
        # g_idx 是 act-order / GPTQ 类量化的索引元数据，和普通权重的切分规则不完全相同。
        if shard_id == "w2":
            self._load_w2(
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )
        else:
            assert shard_id in ("w1", "w3")
            expert_data.copy_(loaded_weight)

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        # 把全局 expert id 映射到当前层本地物理 slot id。
        # tiered cache 打开时，返回 -1 表示该 expert 当前不在 GPU resident slots 中。
        if self._expert_map is None:
            return expert_id
        return self._expert_map[expert_id].item()

    def _init_aiter_shared_experts_topK_buffer(
            self, cfie_config: CfieConfig, dp_size: int
    ):
        # 入口：初始化 AITER fused shared experts 所需的 top-k 元数据缓冲区。
        # 只有当前层真的带有 fused shared experts 时，才需要把路由规模、
        # 并行切分方式和 token 上限预先同步给 AITER。
        if self.num_fused_shared_experts > 0:
            # routed/shared experts 数量使用当前层真实配置。
            # 开启 EP 时改用 expert parallel 视角传递 tp_rank/tp_size，
            # 保证 AITER 看到的分片拓扑与执行期一致。
            # max_num_tokens 按 scheduler 的单批上限乘以 dp_size，
            # 作为当前 DP 波次下的总 token 预算。
            init_aiter_topK_meta_data(
                n_routed_experts=self.global_num_experts,
                n_shared_experts=self.num_fused_shared_experts,
                top_k=self.top_k,
                tp_rank=self.ep_rank if self.use_ep else self.tp_rank,
                tp_size=self.ep_size if self.use_ep else self.tp_size,
                shared_experts_score=1.0,
                max_num_tokens=cfie_config.scheduler_config.max_num_batched_tokens
                               * dp_size,
                is_EP=self.use_ep,
            )
        # fused shared experts 会追加到本地 expert 集合中。
        # 因此无论是否真正触发 AITER 初始化，都要把它们计入 local_num_experts，
        # 让后续权重布局、slot 计算和执行索引都基于统一的本地视图。
        self.local_num_experts += self.num_fused_shared_experts

    # 静态类型提示：调用方如果不关心返回值，就把该接口视作返回 None。
    @overload
    def weight_loader(
            self,
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            weight_name: str,
            shard_id: str,
            expert_id: int,
            return_success: Literal[False],
    ) -> None:
        ...

    # 静态类型提示：调用方若显式要求 success 标志，则返回 bool 表示该 expert 是否被当前层接收。
    @overload
    def weight_loader(
            self,
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            weight_name: str,
            shard_id: str,
            expert_id: int,
            return_success: Literal[True],
    ) -> bool:
        ...

    def weight_loader(
            self,
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            weight_name: str,
            shard_id: str,
            expert_id: int,
            return_success: bool = False,
    ) -> bool | None:
        # -----------------
        # 入口：统一装载一个 expert shard。
        # -----------------
        # 这是 FusedMoE 的统一权重装载入口：
        # 先把全局 expert id 映射到本地 slot，再根据量化格式和 shard 类型分派到细分 loader。
        if self.quant_config and self.quant_config.get_name() == "mxfp4":
            # mxfp4 的某些 checkpoint 会把所有 experts 合在一起存，因此直接按实际 shape 拷贝。

            if "bias" in weight_name:
                dim1 = loaded_weight.shape[1]
                param.data[:, :dim1].copy_(loaded_weight)
            else:
                dim1 = loaded_weight.shape[1]
                dim2 = loaded_weight.shape[2]
                param.data[:, :dim1, :dim2].copy_(loaded_weight)
            return True if return_success else None

        # -----------------
        # 预处理：expert 映射与基础元数据。
        # -----------------
        quant_method_name = self.quant_method.__class__.__name__
        global_expert_id = expert_id
        # 先把全局 expert id 映射成当前层“此刻”可用的本地物理 slot。
        expert_id = self._map_global_expert_id_to_local_expert_id(global_expert_id)

        # 某些量化格式的 input scale 是全局共享的，不受本地 expert 放置约束。
        use_global_sf = (
                getattr(self.quant_method, "use_global_sf", False)
                and "input_scale" in weight_name
        )  # False

        if expert_id == -1 and not use_global_sf:
            # 这正是 CFIE tiered cache 启动期“只装初始 GPU experts”的关键逻辑：
            # 未映射到 resident slot 的 expert 会在这里直接跳过，不会被加载进 GPU 权重张量。
            return False if return_success else None
        # 从这里开始，expert_id 都表示“当前层本地物理 slot id”。

        # -----------------
        # 特殊元数据分支：GGUF / BNB / 参数物化。
        # -----------------
        # 某些量化格式会把“应切分的维度”转置存储，因此这里要先识别出来。
        is_transposed = getattr(param, "is_transposed", False)  # True

        if quant_method_name in (
                "CompressedTensorsWNA16MarlinMoEMethod",
                "CompressedTensorsWNA16MoEMethod",
        ):
            if is_transposed:
                # packed checkpoint 的布局和运行时参数布局相反时，先翻转再继续装载。
                loaded_weight = loaded_weight.t().contiguous()
            else:
                loaded_weight = loaded_weight

        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(f"shard_id must be ['w1','w2','w3'] but got {shard_id}.")

        # shard_id 决定 TP 分片应该落在哪一维。
        SHARD_ID_TO_SHARDED_DIM = {"w1": 0, "w2": 1, "w3": 0}

        is_gguf_weight = getattr(param, "is_gguf_weight", False)  # False
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)  # False
        if is_gguf_weight_type:
            # GGUF 的权重类型元数据不是普通张量权重，直接记录类型和值即可。
            param.weight_type = loaded_weight.item()
            param.data.copy_(loaded_weight)
            return True if return_success else None

        # BitsAndBytes 4bit 的 inflight quantization 已经提前做过分片，
        # 这里不能再按 TP 规则二次切分。
        use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)  # False
        if use_bitsandbytes_4bit:
            shard_dim = 0

            expert_data = param.data[expert_id]
            if shard_id == "w2":
                expert_data.copy_(loaded_weight)
            elif shard_id in ("w1", "w3"):
                full_load = True
                # BNB 已经提前做过分片，因此这里禁止再按 TP 二次切分。
                self._load_w13(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.tp_rank,
                    load_full=full_load,
                )
            return True if return_success else None

        # 普通路径下，先确定真正要切的维度。
        shard_dim = SHARD_ID_TO_SHARDED_DIM[shard_id]
        if is_transposed:  # 默认
            shard_dim = int(not shard_dim)

        # 3D 权重表示 checkpoint 已经把 expert 维合并到张量里，此时要按 full_load 处理。
        full_load = len(loaded_weight.shape) == 3  # False
        if full_load:
            shard_dim += 1

        # GGUF 某些参数在装载前还是未初始化状态；
        # 这里要先按“合并专家 + 当前 TP 分片”的最终尺寸把参数物化出来。
        if is_gguf_weight and isinstance(param, UninitializedParameter):
            assert full_load
            final_shape = list(loaded_weight.shape)
            # w1 / w3 在运行时共享 w13 容器，因此 hidden_out 方向要乘 2。
            if shard_id in {"w1", "w3"}:
                final_shape[1] *= 2
            # 最终参数 shape 要按本 TP rank 的切分后尺寸 materialize。
            final_shape[shard_dim] = final_shape[shard_dim] // self.tp_size
            param.materialize(final_shape, dtype=loaded_weight.dtype)

        # full_load 时直接操作整块参数；否则只拿当前本地 expert 的那一片视图。
        expert_data = param.data if full_load else param.data[expert_id]

        # -----------------
        # 特殊参数分支：input_scale / g_idx。
        # -----------------
        # input_scale 当前只走这条专门分支处理。
        if "input_scale" in weight_name:
            # 输入 scale 需要先和参数所在设备对齐。
            loaded_weight = loaded_weight.to(param.data.device)

            if (
                    "compressed" in quant_method_name.lower()
                    and param.data[expert_id] != 1
                    and (param.data[expert_id] - loaded_weight).abs() > 1e-5
            ):
                # w1 / w3 共用输入 scale，不允许同一 expert 的两个值不一致。
                raise ValueError(
                    "input_scales of w1 and w3 of a layer "
                    f"must be equal. But got {param.data[expert_id]} "
                    f"vs. {loaded_weight}"
                )

            # 某些格式要求按全局 expert 写入 shared scale，其余则按本地 expert 写入。
            self._load_single_value(
                param=param,
                loaded_weight=loaded_weight,
                expert_id=global_expert_id if use_global_sf else expert_id,
            )
            return True if return_success else None

        # g_idx 走索引元数据专用分支。
        if "g_idx" in weight_name:
            # g_idx 不属于普通权重矩阵，单独走索引装载逻辑。
            # 当前 GPTQ Marlin 常见配置下，g_idx 通常表现为“按 group_size 分块递增”的
            # 一维 int32 索引。例如 hidden_size=3072、group_size=128 时，
            # w13_g_idx 往往就是长度 3072、取值分成 24 组的 [0...0, 1...1, ..., 23...23]。
            # 若当前模型是 desc_act=false，这些 g_idx 虽然会先按统一接口加载，
            # 但在 post-load 阶段通常会被替换成空张量，不参与运行时计算。
            self._load_g_idx(
                shard_dim=0,
                shard_id=shard_id,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=self.tp_rank,
            )
            return True if return_success else None

        # -----------------
        # 特殊量化分支：ModelOpt。
        # -----------------
        # TODO @dsikka: ModelOpt 后续应收敛到统一的 MoE 装载模式。
        if "ModelOpt" in quant_method_name:
            # ModelOpt 的命名模式比较特殊，这里按其 scale 约定单独分发。
            # per-tensor scale 的判断依赖具体变体规则，不适合靠脆弱的字符串匹配硬推。
            uses_weight_scale_2 = self.quant_method.uses_weight_scale_2_pattern()
            quant_method = getattr(param, "quant_method", None)

            # 这里要识别“哪些 scale 实际上是 per-tensor 标量”：
            # - input_scale 总是 per-tensor
            # - FP4 往往用 weight_scale_2
            # - FP8 往往用 weight_scale
            # - 但 ModelOpt MXFP8 的 BLOCK 量化里，weight_scale 可能是 block scale，
            #   这时不能误判成 per-tensor 标量
            is_block_weight_scale = (
                    "weight_scale" in weight_name
                    and quant_method == FusedMoeWeightScaleSupported.BLOCK.value
            )
            is_per_tensor = (
                                "weight_scale_2" in weight_name
                                if uses_weight_scale_2
                                else "weight_scale" in weight_name
                            ) or "input_scale" in weight_name
            is_per_tensor = is_per_tensor and not is_block_weight_scale
            if is_per_tensor:
                # 标量 scale 直接走 per-tensor 分支。
                self._load_per_tensor_weight_scale(
                    shard_id=shard_id,
                    param=param,
                    loaded_weight=loaded_weight,
                    expert_id=expert_id,
                )
                return True if return_success else None

            # 若 w13_weight_scale 把多份 w13 scale 合并在同一个 loaded_weight 里，
            # 这里就改走 _load_combined_w13_weight_scale()。
            # 是否命中这类布局，通过 loaded_weight 与参数的 hidden_out 维对比判断。
            if "w13_weight_scale" in weight_name:
                loaded_weight_hidden_out = loaded_weight.shape[-2]
                param_hidden_out = param.data.shape[-2] * self.tp_size
                if loaded_weight_hidden_out == param_hidden_out:
                    # 命中“w13 的 scale 合并存储”模式时，按合并布局切分后写入。
                    self._load_combined_w13_weight_scale(
                        shard_dim=shard_dim,
                        loaded_weight=loaded_weight,
                        param=expert_data,
                        tp_rank=self.tp_rank,
                    )
                    return True if return_success else None

            # 其余 ModelOpt 权重继续复用通用的权重 / group-scale 装载逻辑。
            if "weight" in weight_name:
                # 其余权重和 group scale 统一复用通用装载逻辑。
                self._load_model_weight_or_group_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.tp_rank,
                )
            return True if return_success else None

        # -----------------
        # 通用量化分支：scale / zero / offset。
        # -----------------
        # scale / zero / offset 等量化附属参数统一走这里。
        if "scale" in weight_name or "zero" in weight_name or "offset" in weight_name:
            # 普通量化分支下，根据参数上记录的 quant_method 决定 scale / zp 的装载方式。
            quant_method = getattr(param, "quant_method", None)
            # 当前 GPTQ-Marlin routed experts 的典型情况：
            # - quant_method = GROUP
            # - 因为 bits=4, group_size=128, desc_act=false
            # - qzeros 虽然会被正常加载，但在 sym=true 的 GPTQ Marlin 路径下，
            #   它更多是为了兼容统一的权重接口；kernel 运行时通常不会真正消费它
            # - 4bit 下 qzeros 还是按 int32 打包存储，一个元素里会装 8 个 4-bit zp
            # - 若这 8 个 zp 都等于 8，调试器里就会看到大量相同的 -2004318072，
            #   其无符号位模式其实是 0x88888888
            # 所以大多数 scales / qzeros 会走下面的 GROUP 分支，
            # 调 _load_model_weight_or_group_weight_scale(...)。
            if quant_method == FusedMoeWeightScaleSupported.CHANNEL.value:
                # 每通道 scale 与权重矩阵同形分片。
                self._load_per_channel_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.tp_rank,
                )
            elif quant_method in [
                FusedMoeWeightScaleSupported.GROUP.value,
                FusedMoeWeightScaleSupported.BLOCK.value,
            ]:
                # group / block scale 复用和模型权重一致的 shard 规则。
                self._load_model_weight_or_group_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.tp_rank,
                    load_full_w2=getattr(param, "load_full_w2", False),
                )
            elif quant_method == FusedMoeWeightScaleSupported.TENSOR.value:
                # 每 tensor scale 只是一组标量。
                self._load_per_tensor_weight_scale(
                    shard_id=shard_id,
                    param=param,
                    loaded_weight=loaded_weight,
                    expert_id=expert_id,
                )
            else:
                WEIGHT_SCALE_SUPPORTED = [e.value for e in FusedMoeWeightScaleSupported]
                raise ValueError(
                    f"quant method must be one of {WEIGHT_SCALE_SUPPORTED}"
                )
            return True if return_success else None

        # -----------------
        # 通用元数据分支：weight_shape。
        # -----------------
        # weight_shape 这类元数据参数单独处理。
        if "weight_shape" in weight_name:
            # 某些压缩格式会额外保存权重 shape 元数据，按单值参数处理。
            self._load_single_value(
                param=param, loaded_weight=loaded_weight, expert_id=expert_id
            )
            return True if return_success else None

        # -----------------
        # 通用权重分支：真实模型权重。
        # -----------------
        # 最后兜底处理真正的模型权重张量。
        if "weight" in weight_name:
            # 当前 experts.w13_qweight / experts.w2_qweight 最终就会走这里，
            # 再按 shard_id="w1"/"w2"/"w3" 和当前 tp_rank / local expert slot
            # 切到 expert_data 对应区域。
            self._load_model_weight_or_group_weight_scale(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=self.tp_rank,
            )
            return True if return_success else None

        # -----------------
        # 收尾：返回兼容结果。
        # -----------------
        # return_success=False 时调用方通常不关心结果；这里保留兼容签名。
        return False if return_success else None

    def load_weights(
            self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[str]:
        # -----------------
        # 入口：批量消费一组 expert 权重。
        # -----------------
        # self.expert_mapping 负责把 checkpoint 名字映射到 FusedMoE 内部参数名和 shard 信息。
        if (expert_mapping := self.expert_mapping) is None:
            raise ValueError(
                "`self.expert_mapping` must be provided to "
                "load weights using `self.load_weights`."
            )

        # -----------------
        # 外层循环：遍历 checkpoint 中的每一份 expert 权重。
        # -----------------
        for expert_name, loaded_weight in weights:
            # 把 checkpoint 中的 expert 权重名补成当前层的全限定名。
            qual_name = f"{self.layer_name}.{expert_name}"

            # -----------------
            # 映射匹配：找到它在当前层中应落到哪个内部参数。
            # -----------------
            for param_name, weight_name, expert_id, shard_id in expert_mapping:
                if weight_name not in qual_name:
                    continue
                weight_name = qual_name.replace(weight_name, param_name)
                param_name = weight_name.removeprefix(f"{self.layer_name}.")
                param = getattr(self, param_name)

                # -----------------
                # 规范化输入：统一转成可逐 expert 枚举的视图。
                # -----------------
                # 3D 权重通常表示 checkpoint 已经把多个 experts fuse 在同一张量里。
                if loaded_weight.dim() == 3:
                    # 3D 权重表示多个 experts 被 fuse 在一起，需要先拆出当前 shard 视图。
                    # 对 w1 / w3 来说，这里的 expert_id 会先临时充当 shard_idx，
                    # 用来从合并布局里拆出对应的半边。
                    if shard_id in {"w1", "w3"}:
                        shard_idx = expert_id
                        experts_shard = loaded_weight.chunk(2, dim=1)[shard_idx]
                    else:
                        experts_shard = loaded_weight
                    start = 0
                else:
                    # 单 expert 权重这里补一层假 expert 维，
                    # 以便和 fused 情况统一走同一套循环。
                    experts_shard = loaded_weight.unsqueeze(0)
                    start = expert_id

                # -----------------
                # 内层循环：逐个 expert 调用参数级 weight_loader。
                # -----------------
                loaded_experts = experts_shard.unbind()
                for expert_id, loaded_expert in enumerate(loaded_experts, start=start):
                    # tiered cache 打开时，这里只会真正装入当前映射到 GPU slots 的 experts。
                    success = self.weight_loader(
                        param=param,
                        loaded_weight=loaded_expert,
                        weight_name=weight_name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        logger.debug(
                            "Loaded expert %d of shard %s into %s for layer %s",
                            expert_id,
                            shard_id,
                            param_name,
                            self.layer_name,
                        )
                        yield param_name

    def get_expert_weights(self) -> Iterable[torch.Tensor]:
        def _maybe_make_contiguous(
                name: str, p: torch.nn.Parameter
        ) -> torch.nn.Parameter:
            """
            In some cases, the last 2 dimensions (the non-expert dimensions)
            of the weight scale tensor are transposed. This function
            transforms the tensor (view update) so the tensor is contiguous().
            Example: A non-contiguous scale tensor,
              `x` of shape (E, 32, 16) and stride (512, 1, 32) is transformed to
              `x_` of shape (E, 16, 32) and stride (512, 32, 1).
              Note that we specifically use torch.transpose() so `x_` refers
              to the same underlying memory. The tensors `x` and `x_`, pointing
              to the same underlying memory make this transformation safe in the
              context of EPLB. i.e. It is the same memory and just the view
              is different.
            Note: This function handles the "weight_scale" tensors specifically.
            This could however be generalized to handle similar tensors.
            """
            if p.ndim != 3:
                # 非 expert-major 3D 张量不需要特殊处理。
                return p
            if p.is_contiguous():
                # Already contiguous. do nothing.
                return p
            # p is non-contiguous. We only handle the case where the last 2
            # dimensions of the scales tensor is transposed. We can handle
            # other cases when they become relevant.
            is_transposed_12 = p.stride(1) == 1 and p.stride(2) != 1
            if "weight_scale" not in name or not is_transposed_12:
                # do nothing.
                return p

            # EPLB 导出权重时只需要一个 contiguous 视图，不应反向修改层内原始参数布局。
            # Do not update the layer parameter as the layer's MoE operations would
            # expect the parameter's tensor to the same shape / stride. Instead,
            # make a new torch.nn.Parameter that is used just in the context of
            # EPLB.
            return torch.nn.Parameter(
                torch.transpose(p.data, 1, 2), requires_grad=False
            )

        # 先取出当前层所有参数，再针对 EPLB 特殊情况修正其视图布局。
        weights = list(self.named_parameters())
        weights = [(name, _maybe_make_contiguous(name, p)) for name, p in weights]

        # 除 gate / shared experts 这些非 routed expert 参数外，其余都应是 contiguous。
        assert all(
            weight.is_contiguous()
            for name, weight in weights
            if not (name.startswith("_shared_experts.") or name.startswith("_gate."))
        )

        # Filter out the non-expert weights.
        # `e_score_correction_bias` is a bias for each logical expert,
        # with shape (num_logical_experts,), not an expert weight.
        NON_EXPERT_WEIGHTS = {
            "e_score_correction_bias",
        }

        # 最终把所有 expert 参数拉平成 [local_num_experts, -1]，便于 EPLB 统一打包/迁移。
        return [
            weight.view(self.local_num_experts, -1)
            for name, weight in weights
            if name not in NON_EXPERT_WEIGHTS
               and weight.shape != torch.Size([])
               and not name.startswith("_shared_experts.")
               # exclude parameters from non-expert submodules (e.g. gate/shared)
               and not name.startswith("_gate.")
        ]

    def set_eplb_state(
            self,
            moe_layer_idx: int,
            expert_load_view: torch.Tensor,
            logical_to_physical_map: torch.Tensor,
            logical_replica_count: torch.Tensor,
    ) -> None:
        # 把 EPLB 的层内状态注册到当前层。
        # 后续 forward 会直接读取这些映射关系，并把负载统计写回 expert_load_view。
        # 每层只取自己那一份 EPLB 状态切片，forward 时直接就地读取。
        self.eplb_state.expert_load_view = expert_load_view[moe_layer_idx]
        self.eplb_state.logical_to_physical_map = logical_to_physical_map[moe_layer_idx]
        self.eplb_state.logical_replica_count = logical_replica_count[moe_layer_idx]

    def ensure_moe_quant_config_init(self):
        # 量化配置依赖权重后处理，不能在 __init__ 时过早创建，这里按需懒初始化。
        if self.quant_method.moe_quant_config is None:
            # moe_quant_config 依赖权重装载后的后处理结果，因此只能在这里延迟创建。
            self.quant_method.moe_quant_config = (
                self.quant_method.get_fused_moe_quant_config(self)
            )

    @property
    def moe_quant_config(self) -> FusedMoEQuantConfig | None:
        # 对外暴露量化配置前，先确保懒初始化已经完成。
        self.ensure_moe_quant_config_init()
        return self.quant_method.moe_quant_config

    def must_reduce_shared_expert_outputs(self) -> bool:
        # shared experts 往往由 RowParallelLinear 计算。
        # 纯 TP 场景下，输出可以先不立刻规约，等 MoE 末尾再统一处理；
        # 但 EP + all2all 场景下，各 DP rank 都会得到完整 hidden_states，
        # 因此需要更早地规约 shared experts 输出。
        # 是否需要提前归并 shared experts 输出，交由 runner 根据执行 backend 决定。
        return self.runner.must_reduce_shared_expert_outputs()

    def maybe_all_reduce_tensor_model_parallel(self, final_hidden_states: torch.Tensor):
        # 某些 combine kernel 默认就已经跨 GPU rank 做了规约。
        # 某些 combine kernel 已经隐式完成规约；runner 会在这里做最终兜底判断。
        return self.runner.maybe_all_reduce_tensor_model_parallel(final_hidden_states)

    def forward_native(
            self,
            hidden_states: torch.Tensor,
            router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # FusedMoE 自身不直接展开执行细节，统一转发给 runner。
        return self.runner.forward(
            hidden_states,
            router_logits,
        )

    @property
    def expert_map(self) -> torch.Tensor | None:
        # 对 CFIE tiered cache 而言，这张表也表示“当前哪些全局 experts 映射在 GPU resident slots 上”。
        return (
            self._expert_map if not self.rocm_aiter_fmoe_enabled else self.expert_mask
        )

    def forward_cuda(
            self,
            hidden_states: torch.Tensor,
            router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # CUDA 路径当前直接复用 native 实现，真正的 backend 区分在 runner / quant_method 内部。
        return self.forward_native(hidden_states, router_logits)

    @classmethod
    def make_expert_params_mapping(
            cls,
            model: torch.nn.Module,
            ckpt_gate_proj_name: str,
            ckpt_down_proj_name: str,
            ckpt_up_proj_name: str,
            num_experts: int,
            num_redundant_experts: int = 0,
    ) -> list[tuple[str, str, int, str]]:
        # 这是 routed experts 命名映射的最终生成器。

        num_physical_experts = num_experts + num_redundant_experts

        # 这一步的意义：
        # - checkpoint 里的 experts.37.* 中 “37” 通常是 logical expert id
        # - 运行时当前 rank / EPLB / 冗余专家用的是 physical expert id
        # 因此要先建一个 physical -> logical 的映射表，后面拼 weight_name 时用 logical id，
        # 而返回给加载器的 expert_id 则保留 physical id。

        physical_to_logical_map = (
            EplbState.build_initial_global_physical_to_logical_map(
                num_experts,
                num_redundant_experts
            )
        )

        base_layer = (
            "base_layer."
            if any(".base_layer." in name for name, _ in model.named_parameters())
            else ""
        )

        # 返回值里最关键的命名规则：
        # 1. gate_proj / up_proj 都映射到内部的 experts.w13_
        #    因为内部实现把这两块融合存储成一组 [w1 | w3]
        # 2. down_proj 映射到内部的 experts.w2_
        #
        # shard_id 的语义：
        # - "w1" -> gate_proj 那半边
        # - "w2" -> down_proj
        # - "w3" -> up_proj 那半边
        #
        # 因而 “w13” 这个名字并不是说只有一个矩阵，
        # 而是内部把 w1 和 w3 融合在同一套参数容器里。

        return [
            # (param_name, weight_name, expert_id, shard_id)
            (
                # 当前模型的参数名
                f"experts.{base_layer}w13_"
                if weight_name in [ckpt_gate_proj_name, ckpt_up_proj_name]
                else f"experts.{base_layer}w2_",
                # 权重文件参数名
                f"experts.{physical_to_logical_map[expert_id]}.{weight_name}.{base_layer}",
                expert_id,
                shard_id,
            )
            for expert_id in range(num_physical_experts)
            for shard_id, weight_name in [
                ("w1", ckpt_gate_proj_name),
                ("w2", ckpt_down_proj_name),
                ("w3", ckpt_up_proj_name),
            ]
        ]

    def extra_repr(self) -> str:
        # 打印层对象时，补充关键的 expert / 并行规模信息，便于调试。
        s = (
            f"global_num_experts={self.global_num_experts}, "
            f"local_num_experts={self.local_num_experts}, "
            f"top_k={self.top_k}, "
            f"intermediate_size_per_partition={self.intermediate_size_per_partition}, "  # noqa: E501
            f"tp_size={self.tp_size},\n"
            f"ep_size={self.ep_size}, "
            f"reduce_results={self.reduce_results}, "
        )

        return s


# 标记 FusedMoE 的 weight_loader 支持 MoE 专属参数，
# 避免模型装载路径里做昂贵的运行时反射判断。
# 给模型加载器一个显式标记，表示这个 loader 能直接处理 MoE 专属参数。
FusedMoE.weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]
