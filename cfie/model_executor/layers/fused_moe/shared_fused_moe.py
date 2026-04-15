# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from cfie.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from cfie.model_executor.layers.fused_moe.layer import FusedMoE


# TODO(bnell): 后续可考虑补一个更明确的 shared + fused 组合辅助接口。
class SharedFusedMoE(FusedMoE):
    # 这是带 shared experts 分支的 FusedMoE 实现。
    # 它在执行 routed experts 的 fused MoE 主路径之外，
    # 还负责补上 shared experts 分支的前向计算与必要的并行规约。
    # 某些后端支持 overlap 模式，此时 shared experts 可以与 routed experts
    # 的通信 / 计算阶段交错执行，以减少整体等待时间。

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # ------------------------------- 非 overlap 模式：先算 shared，再算 routed -------------------------------
        # 这条路径下，shared experts 和 routed experts 按串行顺序执行，
        # 逻辑更直接，也便于在 shared 分支结束后立即决定是否补做 TP 规约。
        if not self.use_overlapped:
            # 若当前层确实配置了 shared experts，就先独立计算 shared 分支输出。
            if self._shared_experts is not None:
                shared_out = self._shared_experts(hidden_states)

                # shared experts 内部通常按 reduce_results=False 构建，
                # 因此这里要根据并行配置判断是否需要在外层补做一次 TP all-reduce。
                if (
                    self.reduce_results
                    and get_tensor_model_parallel_world_size() > 1
                    and self.must_reduce_shared_expert_outputs()
                ):
                    shared_out = tensor_model_parallel_all_reduce(shared_out)
            else:
                # 若未启用 shared experts，就显式记为 None，保持返回结构稳定。
                shared_out = None

            # routed experts 主路径仍复用父类 FusedMoE 的 forward 实现。
            fused_out = super().forward(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )
        else:
            # ------------------------------- overlap 模式：父类统一调度 shared 与 routed -------------------------------
            # overlap 模式下，父类会在内部同时组织 shared experts 和 routed experts，
            # 这里直接接收两个分支的输出结果。
            shared_out, fused_out = super().forward(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )
            # 如果 shared 分支输出还没有在内部完成规约，就在这里尽早补齐 TP all-reduce，
            # 避免后续分支合并时混入未规约的局部结果。
            if (
                shared_out is not None
                and self.reduce_results
                and get_tensor_model_parallel_world_size() > 1
                and self.must_reduce_shared_expert_outputs()
            ):
                shared_out = tensor_model_parallel_all_reduce(shared_out)

        # 返回 shared experts 分支输出与 routed experts 分支输出，
        # 调用方会在更外层决定两者如何合并成最终 hidden states。
        return shared_out, fused_out
