# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from cfie.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from cfie.model_executor.layers.fused_moe.layer import FusedMoE


# TODO(bnell): 是否补一个 shared + fused 的组合函数？例如 ...
class SharedFusedMoE(FusedMoE):
    # 这是带 shared experts 的 FusedMoE 实现。
    # 除了 routed experts 的 fused 执行外，它还负责计算 shared experts 的输出。
    # 若底层使用 all2all 通信器，shared experts 计算还可以和 fused all2all
    # dispatch 通信阶段交错执行。

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_overlapped:
            # 非 overlap 模式下，先单独跑 shared experts，再执行 fused routed experts。
            if self._shared_experts is not None:
                shared_out = self._shared_experts(hidden_states)

                # 若 shared experts 的 MLP 是按 reduce_results=False 构建的，
                # 这里需要在必要时补做一次 TP all-reduce。
                if (
                    self.reduce_results
                    and get_tensor_model_parallel_world_size() > 1
                    and self.must_reduce_shared_expert_outputs()
                ):
                    shared_out = tensor_model_parallel_all_reduce(shared_out)
            else:
                shared_out = None

            fused_out = super().forward(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )
        else:
            # overlap 模式下，父类会同时返回 shared experts 和 fused experts 的结果。
            shared_out, fused_out = super().forward(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )
            # 必要时尽早完成 shared experts 输出的 TP 规约。
            if (
                shared_out is not None
                and self.reduce_results
                and get_tensor_model_parallel_world_size() > 1
                and self.must_reduce_shared_expert_outputs()
            ):
                shared_out = tensor_model_parallel_all_reduce(shared_out)
        # 返回 shared experts 输出和 fused routed experts 输出。
        return shared_out, fused_out
