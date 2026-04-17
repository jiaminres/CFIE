# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import abstractmethod
from collections.abc import Callable

import torch

from cfie.distributed.eplb.eplb_state import EplbLayerState
from cfie.model_executor.layers.fused_moe.router.fused_moe_router import (
    FusedMoERouter,
)
from cfie.platforms import current_platform

if current_platform.is_cuda_alike():

    @torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
    def eplb_map_to_physical_and_record(
        topk_ids: torch.Tensor,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> torch.Tensor:
        """
        Map the logical expert ids to physical expert ids
        and record the expert load metrics.

        This will select a pseudo-random replica for each logical expert.
        Only used for EPLB.

        Args:
            topk_ids: The logical expert ids.
            expert_load_view: The expert load view.
            logical_to_physical_map: The logical to physical map.
            logical_replica_count: The logical replica count.

        Returns:
            The physical expert ids.
        """

        # 1. Convert the logical expert ids to physical expert ids
        # Directly select a random replica for each logical expert

        # In case `indices_type` is not `torch.long` or `torch.int`,
        # e.g. `torch.uint32` as required by dispatch/combine kernels
        topk_ids_long = topk_ids.long()
        # Use (token position) modulo (replica count)
        # to deterministically choose a replica
        replica_count = logical_replica_count[topk_ids_long]
        # Flatten-position based index, reshaped back to `topk_ids` shape
        pos_indices = torch.arange(
            topk_ids.numel(), device=topk_ids.device, dtype=torch.long
        ).reshape_as(topk_ids)
        # Compute pseudo-random indices by modulo
        replica_indices = (pos_indices % replica_count).unsqueeze(-1)
        physical_ids = (
            logical_to_physical_map[topk_ids_long]
            .gather(-1, replica_indices)
            .squeeze(-1)
        )

        topk_ids = physical_ids

        # 2. Record expert load metrics.

        # TODO(bowen): When using `FusedMoEModularKernel`, this
        # can be done in a more unified way, since
        # `FusedMoEPrepareAndFinalizeModular` will return the expert
        # token count, in some cases directly from the kernel.
        # However, now there are many code paths not using
        # the modular kernel, e.g. calling `fused_experts`,
        # so we decide to keep the logic here.
        #
        # If later refactor moved all the MoE kernel calls
        # to the modular kernel, we can move this logic there
        # to achieve better efficiency.

        # `expert_load_view`: (num_physical_experts,)

        # `torch.bincount` is not compilable, so use `scatter_add_` instead.
        topk_ids_flatten = topk_ids.flatten()
        expert_load_view.scatter_add_(
            dim=0,
            index=topk_ids_flatten.long(),
            src=torch.ones_like(topk_ids_flatten).to(expert_load_view),
        )
        return topk_ids
else:

    def eplb_map_to_physical_and_record(
        topk_ids: torch.Tensor,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> torch.Tensor:
        # CPU fallback: no EPLB so just return as is
        return topk_ids


class BaseRouter(FusedMoERouter):
    """
    所有 router 实现共享的基类。
    这个类负责统一的前处理和后处理。
    具体路由算法由子类实现 `_compute_routing()`。
    `select_experts()` 负责串起完整模板流程。
    """

    def __init__(
        self,
        top_k: int,
        global_num_experts: int,
        eplb_state: EplbLayerState,
        enable_eplb: bool = False,
        # 当前 indices dtype 可能要等 modular kernel 构造完成后才能确定。
        # 因此这里先保存一个运行时回调，而不是直接保存静态 dtype 值。
        indices_type_getter: Callable[[], torch.dtype | None] | None = None,
    ):
        """
        初始化 router 共享状态。
        indices dtype 在 router 构造阶段可能尚未确定。
        这是因为它来自稍后才创建的 modular kernel。
        因此这里通过回调在运行时再查询目标 dtype。
        """
        # 初始化父类基础状态。
        super().__init__()
        # 保存每个 token 要选取的 expert 数。
        self.top_k = top_k # 8
        # 保存模型全局 expert 总数。
        self.global_num_experts = global_num_experts # 256
        # 保存 EPLB 运行时状态。
        self.eplb_state = eplb_state
        # 记录当前是否启用 EPLB 逻辑。
        self.enable_eplb = enable_eplb # False
        # 保存 expert index dtype 的运行时查询回调。
        self.indices_type_getter = indices_type_getter
        # 可选地捕获 EPLB 映射前的逻辑 expert id。
        self.capture_fn: Callable[[torch.Tensor], None] | None = None

    def set_capture_fn(self, capture_fn: Callable[[torch.Tensor], None] | None) -> None:
        """设置逻辑 routed expert id 的捕获回调。"""
        self.capture_fn = capture_fn

    def _validate_eplb_state(self) -> None:
        """在启用 EPLB 时校验所需状态是否已完整初始化。"""
        if self.enable_eplb:
            if self.eplb_state.expert_load_view is None:
                raise ValueError("enable_eplb=True requires expert_load_view != None")
            if self.eplb_state.logical_to_physical_map is None:
                raise ValueError(
                    "enable_eplb=True requires logical_to_physical_map != None"
                )
            if self.eplb_state.logical_replica_count is None:
                raise ValueError(
                    "enable_eplb=True requires logical_replica_count != None"
                )

    def _get_indices_type(self) -> torch.dtype | None:
        """从回调中读取当前期望的 expert index dtype。"""
        return (
            self.indices_type_getter() if self.indices_type_getter is not None else None
        )

    def _apply_eplb_mapping(self, topk_ids: torch.Tensor) -> torch.Tensor:
        """把逻辑 expert id 映射成物理 expert id，并记录负载信息。"""
        if self.enable_eplb:
            assert self.eplb_state.expert_load_view is not None
            assert self.eplb_state.logical_to_physical_map is not None
            assert self.eplb_state.logical_replica_count is not None
            return eplb_map_to_physical_and_record(
                topk_ids=topk_ids,
                expert_load_view=self.eplb_state.expert_load_view,
                logical_to_physical_map=self.eplb_state.logical_to_physical_map,
                logical_replica_count=self.eplb_state.logical_replica_count,
            )
        return topk_ids

    def _convert_indices_dtype(
        self, topk_ids: torch.Tensor, indices_type: torch.dtype | None
    ) -> torch.Tensor:
        """按需把 topk_ids 转成目标 dtype。"""
        if (indices_type is not None) and topk_ids.dtype != indices_type:
            topk_ids = topk_ids.to(dtype=indices_type)

        assert topk_ids.dtype == indices_type or indices_type is None
        return topk_ids

    @abstractmethod
    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        计算具体路由逻辑。
        子类必须实现这个接口。
        不同子类在这里接入各自的 routing 算法。
        例如 grouped top-k、fused top-k、自定义 routing 等。

        参数：
        - `hidden_states`：输入 hidden states
        - `router_logits`：用于 expert 选择的 router logits
        - `indices_type`：期望的 expert index dtype

        返回：
        - `(topk_weights, topk_ids)`
        """
        raise NotImplementedError

    def select_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        根据 router logits 把输入 token 路由到 top-k experts。

        返回：
        - `(topk_weights, topk_ids)`

        兼容性说明：
        - 当未启用 EPLB 时，返回的 ids 等价于全局逻辑 expert ids
        - 因而可以直接兼容没有冗余 expert 的普通 MoE 路径
        """
        # 第一步：校验 EPLB 依赖状态。
        self._validate_eplb_state()

        # 第二步：查询当前 expert index dtype。
        indices_type = self._get_indices_type()

        # 第三步：把实际路由计算委托给子类实现。
        topk_weights, topk_ids = self._compute_routing(
            hidden_states, router_logits, indices_type
        )

        # 若注册了捕获回调，则在 EPLB 映射前记录逻辑 expert ids。
        if self.capture_fn is not None:
            self.capture_fn(topk_ids)

        # 第四步：按需执行 EPLB 逻辑 id -> 物理 id 映射。
        topk_ids = self._apply_eplb_mapping(topk_ids)

        # 第五步：按目标 dtype 整理 expert index 张量。
        topk_ids = self._convert_indices_dtype(topk_ids, indices_type)

        return topk_weights, topk_ids
