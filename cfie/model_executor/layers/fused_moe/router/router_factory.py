# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch

import cfie.envs as envs
from cfie.distributed.eplb.eplb_state import EplbLayerState
from cfie.model_executor.layers.fused_moe.config import RoutingMethodType
from cfie.model_executor.layers.fused_moe.router.custom_routing_router import (
    CustomRoutingRouter,
)
from cfie.model_executor.layers.fused_moe.router.fused_moe_router import (
    FusedMoERouter,
)
from cfie.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    FusedTopKBiasRouter,
)
from cfie.model_executor.layers.fused_moe.router.fused_topk_router import (
    FusedTopKRouter,
)
from cfie.model_executor.layers.fused_moe.router.grouped_topk_router import (
    GroupedTopKRouter,
)
from cfie.model_executor.layers.fused_moe.router.routing_simulator_router import (
    RoutingSimulatorRouter,
)

EMPTY_EPLB_STATE: EplbLayerState = EplbLayerState()


def create_fused_moe_router(
    # 通用参数
    top_k: int,
    global_num_experts: int,
    renormalize: bool = True,
    indices_type_getter: Callable[[], torch.dtype | None] | None = None,
    # grouped top-k 参数
    use_grouped_topk: bool = False,
    num_expert_group: int | None = None,
    topk_group: int | None = None,
    scoring_func: str = "softmax",
    num_fused_shared_experts: int = 0,
    # grouped top-k / fused top-k bias 共享参数
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
    # 自定义 routing 参数
    custom_routing_function: Callable | None = None,
    # EPLB 参数
    enable_eplb: bool = False,
    eplb_state: EplbLayerState = EMPTY_EPLB_STATE,
) -> FusedMoERouter:
    """
    根据当前配置创建合适的 FusedMoERouter 子类。

    路由器选择优先级如下：
    1. 若设置了 `VLLM_MOE_ROUTING_SIMULATION_STRATEGY`，优先返回 `RoutingSimulatorRouter`
    2. 若启用 `use_grouped_topk`，优先尝试构造 `GroupedTopKRouter`
    3. 若提供 `custom_routing_function`，返回 `CustomRoutingRouter`
    4. 若提供 `e_score_correction_bias`，返回 `FusedTopKBiasRouter`
    5. 其余情况回退到默认的 `FusedTopKRouter`

    主要参数语义：
    - `top_k`：每个 token 选择的 expert 数
    - `global_num_experts`：模型全局 expert 总数
    - `renormalize`：是否对路由权重重新归一化
    - `indices_type_getter`：返回路由索引 dtype 的回调
    - `use_grouped_topk`：是否启用 grouped top-k 路由
    - `num_expert_group`：grouped top-k 下的 expert group 数
    - `topk_group`：grouped top-k 下每个 group 内选取的 top-k
    - `scoring_func`：路由打分函数，例如 `softmax` 或 `sigmoid`
    - `num_fused_shared_experts`：fused shared experts 数量
    - `routed_scaling_factor`：routed 权重缩放因子
    - `e_score_correction_bias`：expert score bias 校正张量
    - `custom_routing_function`：自定义路由函数
    - `enable_eplb`：是否启用 expert 并行负载均衡
    - `eplb_state`：当前层对应的 EPLB 状态
    """

    # 若环境变量显式要求 routing simulation，则直接短路到模拟路由器。
    routing_strategy = envs.VLLM_MOE_ROUTING_SIMULATION_STRATEGY
    if routing_strategy != "":
        return RoutingSimulatorRouter(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
            enable_eplb=enable_eplb,
            indices_type_getter=indices_type_getter,
        )

    # grouped top-k 优先于 custom routing / bias routing。
    if use_grouped_topk:
        assert custom_routing_function is None
        # grouped top-k 至少需要 group 数和每组 top-k 两个核心参数。
        if num_expert_group is None or topk_group is None:
            raise ValueError(
                "num_expert_group and topk_group must be provided when "
                "use_grouped_topk is True"
            )
        grouped_topk_router = GroupedTopKRouter(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            renormalize=renormalize,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            num_fused_shared_experts=num_fused_shared_experts,
            enable_eplb=enable_eplb,
            indices_type_getter=indices_type_getter,
        )
        if (
            grouped_topk_router.routing_method_type != RoutingMethodType.Unspecified
            or num_expert_group > 1
            or topk_group > 1
        ):
            return grouped_topk_router

        # 若 grouped router 最终未选择特化 routing 方法，且实际只有单组，
        # 则说明没有继续保留 grouped 路径的必要，回退到标准 top-k。
        use_grouped_topk = False
        num_expert_group = None
        topk_group = None

    # 自定义 routing 的优先级高于 bias router 和默认 top-k router。
    if custom_routing_function is not None:
        return CustomRoutingRouter(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
            custom_routing_function=custom_routing_function,
            renormalize=renormalize,
            enable_eplb=enable_eplb,
            indices_type_getter=indices_type_getter,
        )

    # 若存在 expert score bias 校正，则选择 bias 版本 router。
    if e_score_correction_bias is not None:
        return FusedTopKBiasRouter(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
            e_score_correction_bias=e_score_correction_bias,
            scoring_func=scoring_func,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
            enable_eplb=enable_eplb,
            indices_type_getter=indices_type_getter,
        )

    # 其余场景统一走默认 fused top-k router。
    return FusedTopKRouter( # 默认
        top_k=top_k, # 8
        global_num_experts=global_num_experts, # 256
        eplb_state=eplb_state,
        renormalize=renormalize, # True
        scoring_func=scoring_func, # softmax
        enable_eplb=enable_eplb, # False
        indices_type_getter=indices_type_getter,
    )
