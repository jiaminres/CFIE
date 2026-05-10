"""Real-model adapter for Qwen3.5 MoE expert MLP training.

Provides a PyTorch module that loads hot expert weights from the FP32 store,
runs actual forward/backward through the MoE expert MLP (gate_up projection
→ SiLU activation → down projection), and returns real gradients consumable
by the HotParamTrainingWindow.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from cfie_training.training_base.fp32_shard_store import FP32ShardStore


class QwenMoEExpertMLP(nn.Module):
    """One MoE expert's MLP: gate_up → SiLU → down.

    Follows the Qwen3.5 architecture:
      - w13: fused gate_proj + up_proj, shape [2*intermediate, hidden]
      - w2:  down_proj, shape [hidden, intermediate]
      - activation: SiLU(gate) * up  (SwiGLU)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        dtype: torch.dtype = torch.float16,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dtype = dtype
        self._device = device
        self.w13_weight: torch.nn.Parameter | None = None
        self.w2_weight: torch.nn.Parameter | None = None

    def load_weights(
        self,
        w13: torch.Tensor,
        w2: torch.Tensor,
        *,
        requires_grad: bool = True,
    ) -> None:
        self.w13_weight = nn.Parameter(
            w13.to(dtype=self.dtype, device=self._device).contiguous(),
            requires_grad=requires_grad,
        )
        self.w2_weight = nn.Parameter(
            w2.to(dtype=self.dtype, device=self._device).contiguous(),
            requires_grad=requires_grad,
        )

    def load_from_store(
        self,
        fp32_store: FP32ShardStore,
        w13_param_id: str,
        w2_param_id: str,
        *,
        requires_grad: bool = True,
    ) -> None:
        w13_raw = fp32_store.read_param(w13_param_id)
        w2_raw = fp32_store.read_param(w2_param_id)
        w13 = torch.frombuffer(bytearray(w13_raw), dtype=torch.float32)
        w2 = torch.frombuffer(bytearray(w2_raw), dtype=torch.float32)
        w13 = w13.reshape(2 * self.intermediate_size, self.hidden_size)
        w2 = w2.reshape(self.hidden_size, self.intermediate_size)
        self.load_weights(w13, w2, requires_grad=requires_grad)

    def sync_from_shadow(
        self,
        w13_shadow: torch.Tensor,
        w2_shadow: torch.Tensor,
    ) -> None:
        w13_2d = w13_shadow.reshape(2 * self.intermediate_size, self.hidden_size)
        w2_2d = w2_shadow.reshape(self.hidden_size, self.intermediate_size)
        if self.w13_weight is None:
            self.w13_weight = nn.Parameter(
                w13_2d.to(dtype=self.dtype, device=self._device).contiguous(),
                requires_grad=True,
            )
        else:
            self.w13_weight.data.copy_(
                w13_2d.to(dtype=self.dtype, device=self._device).contiguous()
            )
        if self.w2_weight is None:
            self.w2_weight = nn.Parameter(
                w2_2d.to(dtype=self.dtype, device=self._device).contiguous(),
                requires_grad=True,
            )
        else:
            self.w2_weight.data.copy_(
                w2_2d.to(dtype=self.dtype, device=self._device).contiguous()
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert self.w13_weight is not None and self.w2_weight is not None
        x = hidden_states.to(dtype=self.dtype, device=self._device)
        gate_up = F.linear(x, self.w13_weight)
        gate, up = gate_up.chunk(2, dim=-1)
        act = F.silu(gate) * up
        out = F.linear(act, self.w2_weight)
        return out

    def collect_gradients(self) -> dict[str, torch.Tensor]:
        grads: dict[str, torch.Tensor] = {}
        if self.w13_weight is not None and self.w13_weight.grad is not None:
            grads["w13_weight"] = self.w13_weight.grad.detach().reshape(-1).to(
                dtype=torch.float32, device="cpu"
            )
        if self.w2_weight is not None and self.w2_weight.grad is not None:
            grads["w2_weight"] = self.w2_weight.grad.detach().reshape(-1).to(
                dtype=torch.float32, device="cpu"
            )
        return grads

    def zero_grad(self) -> None:
        if self.w13_weight is not None and self.w13_weight.grad is not None:
            self.w13_weight.grad.zero_()
        if self.w2_weight is not None and self.w2_weight.grad is not None:
            self.w2_weight.grad.zero_()


def _group_expert_param_ids(param_ids: Iterable[str]) -> dict[str, tuple[str, str]]:
    # 将 param_id 列表按 (layer, expert) 分组，返回 {expert_key: (w13_id, w2_id)}
    """将 param_id 列表按 (layer, expert) 分组，返回 {expert_key: (w13_id, w2_id)}。"""
    groups: dict[str, tuple[str, str]] = {}
    for pid in param_ids:
        parsed = _parse_expert_param_id(pid)
        if parsed is None:
            continue
        layer_id, expert_id, weight_name = parsed
        key = f"L{layer_id}_E{expert_id}"
        if key not in groups:
            groups[key] = ("", "")
        w13, w2 = groups[key]
        if weight_name == "w13_weight":
            groups[key] = (pid, w2)
        elif weight_name == "w2_weight":
            groups[key] = (w13, pid)
    return {k: v for k, v in groups.items() if v[0] and v[1]}


def _parse_expert_param_id(param_id: str) -> tuple[int, int, str] | None:
    # 解析 param_id → (layer_id, expert_id, weight_name)
    parts = param_id.split(".")
    if len(parts) != 5 or parts[0] != "layers" or parts[2] != "experts":
        return None
    try:
        return int(parts[1]), int(parts[3]), parts[4]
    except ValueError:
        return None
