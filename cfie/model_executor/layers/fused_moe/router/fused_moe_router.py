# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod

import torch

from cfie.model_executor.layers.fused_moe.config import RoutingMethodType


class FusedMoERouter(ABC):
    # FusedMoE router 的抽象基类。
    # 统一定义基于 router logits 选择 top-k experts 的接口。

    @property
    @abstractmethod
    def routing_method_type(self) -> RoutingMethodType:
        raise NotImplementedError

    @abstractmethod
    def select_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 根据 router logits 计算每个 token 的 top-k experts。
        # 返回值是 (topk_weights, topk_ids)。
        # 当未启用 EPLB 时，返回的 id 与全局逻辑 expert id 等价，
        # 因而能兼容没有冗余 experts 的普通 MoE 实现。
        raise NotImplementedError
