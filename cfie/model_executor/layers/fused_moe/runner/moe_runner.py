# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod

import torch


class MoERunner(ABC):
    # MoE runner 的抽象基类。
    # 所有具体实现都必须遵循这里定义的执行接口。
    # runner 负责串起 MoE 层前向执行、expert 路由以及并行规约等运行期逻辑。

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def must_reduce_shared_expert_outputs(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def maybe_all_reduce_tensor_model_parallel(
        self,
        final_hidden_states: torch.Tensor,
    ):
        raise NotImplementedError
