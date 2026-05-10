"""Predictor model architectures used by the CFIE training stack."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class FutureExpertPredictor(nn.Module):
    architecture_name = "base"

    def __init__(
            self,
            *,
            input_dim: int,
            hidden_dim: int,
            window_layers: int,
            num_experts: int,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.window_layers = int(window_layers)
        self.num_experts = int(num_experts)

    @staticmethod
    def _normalize_hidden_state(hidden_state: torch.Tensor) -> torch.Tensor:
        if hidden_state.ndim == 1:
            return hidden_state.unsqueeze(0)
        if hidden_state.ndim != 2:
            raise ValueError("hidden_state must be rank-1 or rank-2")
        return hidden_state

    @staticmethod
    def _normalize_layer_index(
            layer_index: int | torch.Tensor,
            *,
            batch_size: int,
            device: torch.device,
    ) -> torch.Tensor:
        if isinstance(layer_index, int):
            return torch.full(
                (batch_size,),
                int(layer_index),
                dtype=torch.float32,
                device=device,
            )

        if layer_index.ndim == 0:
            return torch.full(
                (batch_size,),
                int(layer_index.item()),
                dtype=torch.float32,
                device=device,
            )

        if layer_index.ndim != 1 or int(layer_index.shape[0]) != batch_size:
            raise ValueError("layer_index must be scalar or match batch size")

        return layer_index.to(device=device, dtype=torch.float32)

    @staticmethod
    def _layer_features(layer_index: torch.Tensor) -> torch.Tensor:
        layer_index = layer_index.unsqueeze(-1)
        return torch.cat(
            (
                layer_index,
                layer_index.square(),
                torch.sin(layer_index * 0.1),
                torch.cos(layer_index * 0.1),
                torch.sin(layer_index * 0.01),
                torch.cos(layer_index * 0.01),
            ),
            dim=-1,
        )

    def model_descriptor(self) -> dict[str, Any]:
        descriptor = {
            "architecture": self.architecture_name,
            "hidden_dim": self.hidden_dim,
            "window_layers": self.window_layers,
            "num_experts": self.num_experts,
        }
        descriptor.update(self._descriptor_payload())
        return descriptor

    def _descriptor_payload(self) -> dict[str, Any]:
        return {}


class _SwiGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = x.chunk(2, dim=-1)
        return F.silu(gate) * value


class _ResidualSwiGLUBlock(nn.Module):
    def __init__(
            self,
            *,
            hidden_dim: int,
            ffn_multiplier: int,
            dropout: float,
    ) -> None:
        super().__init__()
        ffn_dim = int(hidden_dim) * int(ffn_multiplier)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim * 2),
            _SwiGLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm(x))


class _CrossAttentionBlock(nn.Module):
    def __init__(
            self,
            *,
            hidden_dim: int,
            num_heads: int,
            ffn_multiplier: int,
            dropout: float,
    ) -> None:
        super().__init__()
        ffn_dim = int(hidden_dim) * int(ffn_multiplier)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim * 2),
            _SwiGLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
            self,
            queries: torch.Tensor,
            context: torch.Tensor,
    ) -> torch.Tensor:
        attn_out, _ = self.attn(
            self.query_norm(queries),
            self.context_norm(context),
            self.context_norm(context),
            need_weights=False,
        )
        queries = queries + self.attn_dropout(attn_out)
        return queries + self.ffn(self.ffn_norm(queries))


class SimpleMLPPredictor(FutureExpertPredictor):
    architecture_name = "mlp"

    def __init__(
            self,
            *,
            input_dim: int,
            hidden_dim: int,
            window_layers: int,
            num_experts: int,
            depth: int,
            dropout: float,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            window_layers=window_layers,
            num_experts=num_experts,
        )
        self.depth = int(depth)
        self.dropout = float(dropout)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layer_proj = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.SiLU(),
        )

        blocks: list[nn.Module] = [nn.SiLU()]
        for _ in range(max(self.depth - 1, 0)):
            blocks.extend(
                (
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                )
            )
            if dropout > 0.0:
                blocks.append(nn.Dropout(dropout))
        blocks.append(nn.Linear(hidden_dim, window_layers * num_experts))
        self.net = nn.Sequential(*blocks)

    def _descriptor_payload(self) -> dict[str, Any]:
        return {
            "depth": self.depth,
            "dropout": self.dropout,
        }

    def forward(
            self,
            hidden_state: torch.Tensor,
            layer_index: int | torch.Tensor,
    ) -> torch.Tensor:
        hidden_state = self._normalize_hidden_state(hidden_state)
        layer_index = self._normalize_layer_index(
            layer_index,
            batch_size=int(hidden_state.shape[0]),
            device=hidden_state.device,
        )
        fused_hidden = self.input_proj(hidden_state) + self.layer_proj(
            self._layer_features(layer_index)
        )
        logits = self.net(fused_hidden)
        return logits.view(-1, self.window_layers, self.num_experts)


class ResidualMLPPredictor(FutureExpertPredictor):
    architecture_name = "residual_mlp"

    def __init__(
            self,
            *,
            input_dim: int,
            hidden_dim: int,
            window_layers: int,
            num_experts: int,
            depth: int,
            dropout: float,
            ffn_multiplier: int,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            window_layers=window_layers,
            num_experts=num_experts,
        )
        self.depth = int(depth)
        self.dropout = float(dropout)
        self.ffn_multiplier = int(ffn_multiplier)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layer_proj = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            _ResidualSwiGLUBlock(
                hidden_dim=hidden_dim,
                ffn_multiplier=ffn_multiplier,
                dropout=dropout,
            )
            for _ in range(max(self.depth, 1))
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, window_layers * num_experts)

    def _descriptor_payload(self) -> dict[str, Any]:
        return {
            "depth": self.depth,
            "dropout": self.dropout,
            "ffn_multiplier": self.ffn_multiplier,
        }

    def forward(
            self,
            hidden_state: torch.Tensor,
            layer_index: int | torch.Tensor,
    ) -> torch.Tensor:
        hidden_state = self._normalize_hidden_state(hidden_state)
        layer_index = self._normalize_layer_index(
            layer_index,
            batch_size=int(hidden_state.shape[0]),
            device=hidden_state.device,
        )
        fused_hidden = self.input_proj(hidden_state) + self.layer_proj(
            self._layer_features(layer_index)
        )
        for block in self.blocks:
            fused_hidden = block(fused_hidden)
        logits = self.output(self.final_norm(fused_hidden))
        return logits.view(-1, self.window_layers, self.num_experts)


class QueryTransformerPredictor(FutureExpertPredictor):
    architecture_name = "query_transformer"

    def __init__(
            self,
            *,
            input_dim: int,
            hidden_dim: int,
            window_layers: int,
            num_experts: int,
            depth: int,
            dropout: float,
            num_heads: int,
            memory_tokens: int,
            ffn_multiplier: int,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            window_layers=window_layers,
            num_experts=num_experts,
        )
        self.depth = int(depth)
        self.dropout = float(dropout)
        self.num_heads = int(num_heads)
        self.memory_tokens = int(memory_tokens)
        self.ffn_multiplier = int(ffn_multiplier)
        self.fused_input_proj = nn.Linear(input_dim, hidden_dim)
        self.layer_proj = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.SiLU(),
        )
        self.memory_proj = nn.Linear(input_dim, hidden_dim * self.memory_tokens)
        self.query_tokens = nn.Parameter(
            torch.randn(window_layers, hidden_dim) / math.sqrt(hidden_dim)
        )
        self.query_condition = nn.Linear(hidden_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            _CrossAttentionBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ffn_multiplier=ffn_multiplier,
                dropout=dropout,
            )
            for _ in range(max(self.depth, 1))
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, num_experts)

    def _descriptor_payload(self) -> dict[str, Any]:
        return {
            "depth": self.depth,
            "dropout": self.dropout,
            "num_heads": self.num_heads,
            "memory_tokens": self.memory_tokens,
            "ffn_multiplier": self.ffn_multiplier,
        }

    def forward(
            self,
            hidden_state: torch.Tensor,
            layer_index: int | torch.Tensor,
    ) -> torch.Tensor:
        hidden_state = self._normalize_hidden_state(hidden_state)
        batch_size = int(hidden_state.shape[0])
        layer_index = self._normalize_layer_index(
            layer_index,
            batch_size=batch_size,
            device=hidden_state.device,
        )
        layer_context = self.layer_proj(self._layer_features(layer_index))
        fused_hidden = self.fused_input_proj(hidden_state) + layer_context
        memory = self.memory_proj(hidden_state).view(
            batch_size,
            self.memory_tokens,
            self.hidden_dim,
        )
        memory = torch.cat((fused_hidden.unsqueeze(1), memory), dim=1)

        queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        queries = queries + self.query_condition(layer_context).unsqueeze(1)
        for block in self.blocks:
            queries = block(queries, memory)
        return self.output(self.output_norm(queries))


class FactorizedExpertPredictor(FutureExpertPredictor):
    architecture_name = "factorized"

    def __init__(
            self,
            *,
            input_dim: int,
            hidden_dim: int,
            window_layers: int,
            num_experts: int,
            depth: int,
            dropout: float,
            ffn_multiplier: int,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            window_layers=window_layers,
            num_experts=num_experts,
        )
        self.depth = int(depth)
        self.dropout = float(dropout)
        self.ffn_multiplier = int(ffn_multiplier)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layer_proj = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            _ResidualSwiGLUBlock(
                hidden_dim=hidden_dim,
                ffn_multiplier=ffn_multiplier,
                dropout=dropout,
            )
            for _ in range(max(self.depth, 1))
        )
        self.per_layer_state = nn.Linear(hidden_dim, window_layers * hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.expert_embeddings = nn.Parameter(
            torch.randn(window_layers, num_experts, hidden_dim)
            / math.sqrt(hidden_dim)
        )
        self.expert_bias = nn.Parameter(torch.zeros(window_layers, num_experts))

    def _descriptor_payload(self) -> dict[str, Any]:
        return {
            "depth": self.depth,
            "dropout": self.dropout,
            "ffn_multiplier": self.ffn_multiplier,
        }

    def forward(
            self,
            hidden_state: torch.Tensor,
            layer_index: int | torch.Tensor,
    ) -> torch.Tensor:
        hidden_state = self._normalize_hidden_state(hidden_state)
        layer_index = self._normalize_layer_index(
            layer_index,
            batch_size=int(hidden_state.shape[0]),
            device=hidden_state.device,
        )
        fused_hidden = self.input_proj(hidden_state) + self.layer_proj(
            self._layer_features(layer_index)
        )
        for block in self.blocks:
            fused_hidden = block(fused_hidden)
        per_layer_state = self.per_layer_state(self.output_norm(fused_hidden)).view(
            -1,
            self.window_layers,
            self.hidden_dim,
        )
        logits = torch.einsum(
            "bwh,weh->bwe",
            per_layer_state,
            self.expert_embeddings,
        )
        return logits + self.expert_bias.unsqueeze(0)


class FrozenRouterDeltaPredictor(FutureExpertPredictor):
    architecture_name = "frozen_router_delta"

    def __init__(
            self,
            *,
            input_dim: int,
            hidden_dim: int,
            window_layers: int,
            num_experts: int,
            num_layers: int,
            router_weights: torch.Tensor,
            depth: int,
            dropout: float,
            ffn_multiplier: int,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            window_layers=window_layers,
            num_experts=num_experts,
        )
        expected_shape = (int(num_layers), int(num_experts), int(input_dim))
        if tuple(router_weights.shape) != expected_shape:
            raise ValueError(
                "router_weights shape must be "
                f"{expected_shape}, got {tuple(router_weights.shape)}"
            )
        self.depth = int(depth)
        self.dropout = float(dropout)
        self.ffn_multiplier = int(ffn_multiplier)
        self.num_layers = int(num_layers)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layer_proj = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            _ResidualSwiGLUBlock(
                hidden_dim=hidden_dim,
                ffn_multiplier=ffn_multiplier,
                dropout=dropout,
            )
            for _ in range(max(self.depth, 1))
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.delta_proj = nn.Linear(hidden_dim, window_layers * input_dim)
        self.future_norm = nn.LayerNorm(input_dim)
        self.relative_hidden_bias = nn.Parameter(
            torch.zeros(window_layers, input_dim)
        )
        self.logit_scale = nn.Parameter(torch.ones(window_layers))
        self.expert_bias = nn.Parameter(torch.zeros(window_layers, num_experts))
        self.register_buffer(
            "router_weights",
            router_weights.to(dtype=torch.float32, device="cpu").contiguous(),
            persistent=False,
        )
        self.register_buffer(
            "future_offsets",
            torch.arange(1, window_layers + 1, dtype=torch.int64),
            persistent=False,
        )

    def _descriptor_payload(self) -> dict[str, Any]:
        return {
            "depth": self.depth,
            "dropout": self.dropout,
            "ffn_multiplier": self.ffn_multiplier,
            "num_layers": self.num_layers,
            "uses_frozen_router": True,
        }

    @staticmethod
    def _normalize_layer_index_long(
            layer_index: int | torch.Tensor,
            *,
            batch_size: int,
            device: torch.device,
    ) -> torch.Tensor:
        if isinstance(layer_index, int):
            return torch.full(
                (batch_size,),
                int(layer_index),
                dtype=torch.int64,
                device=device,
            )
        if layer_index.ndim == 0:
            return torch.full(
                (batch_size,),
                int(layer_index.item()),
                dtype=torch.int64,
                device=device,
            )
        if layer_index.ndim != 1 or int(layer_index.shape[0]) != batch_size:
            raise ValueError("layer_index must be scalar or match batch size")
        return layer_index.to(device=device, dtype=torch.int64)

    def forward(
            self,
            hidden_state: torch.Tensor,
            layer_index: int | torch.Tensor,
    ) -> torch.Tensor:
        hidden_state = self._normalize_hidden_state(hidden_state)
        batch_size = int(hidden_state.shape[0])
        layer_index_long = self._normalize_layer_index_long(
            layer_index,
            batch_size=batch_size,
            device=hidden_state.device,
        )
        fused_hidden = self.input_proj(hidden_state) + self.layer_proj(
            self._layer_features(layer_index_long.to(dtype=torch.float32))
        )
        for block in self.blocks:
            fused_hidden = block(fused_hidden)
        delta_hidden = self.delta_proj(self.output_norm(fused_hidden)).view(
            batch_size,
            self.window_layers,
            self.input_dim,
        )
        future_hidden = self.future_norm(
            hidden_state.unsqueeze(1)
            + self.relative_hidden_bias.unsqueeze(0)
            + delta_hidden
        )
        future_indices = layer_index_long.unsqueeze(1) + self.future_offsets.to(
            device=hidden_state.device
        )
        valid_mask = future_indices < self.num_layers
        router_weights = self.router_weights.to(
            device=hidden_state.device,
            dtype=future_hidden.dtype,
        )
        logits = future_hidden.new_zeros(
            (batch_size, self.window_layers, self.num_experts)
        )
        for future_offset in range(self.window_layers):
            future_layer_ids = future_indices[:, future_offset]
            future_valid = valid_mask[:, future_offset]
            if not bool(future_valid.any().item()):
                continue
            hidden_slice = future_hidden[:, future_offset, :]
            for layer_id in torch.unique(future_layer_ids[future_valid]).tolist():
                layer_mask = future_valid & (future_layer_ids == int(layer_id))
                logits[layer_mask, future_offset, :] = (
                    hidden_slice[layer_mask] @ router_weights[int(layer_id)].T
                )
        logits = (
            logits * self.logit_scale.view(1, self.window_layers, 1)
            + self.expert_bias.unsqueeze(0)
        )
        return logits.masked_fill(~valid_mask.unsqueeze(-1), 0.0)


def build_predictor_model(
        *,
        input_dim: int,
        hidden_dim: int,
        window_layers: int,
        num_experts: int,
        model_architecture: str,
        model_depth: int,
        model_dropout: float,
        model_num_heads: int,
        model_memory_tokens: int,
        model_ffn_multiplier: int,
        num_layers: int | None = None,
        frozen_router_weights: torch.Tensor | None = None,
) -> FutureExpertPredictor:
    common_kwargs = dict(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        window_layers=window_layers,
        num_experts=num_experts,
    )
    if model_architecture == "mlp":
        return SimpleMLPPredictor(
            **common_kwargs,
            depth=model_depth,
            dropout=model_dropout,
        )
    if model_architecture == "residual_mlp":
        return ResidualMLPPredictor(
            **common_kwargs,
            depth=model_depth,
            dropout=model_dropout,
            ffn_multiplier=model_ffn_multiplier,
        )
    if model_architecture == "query_transformer":
        if hidden_dim % model_num_heads != 0:
            raise ValueError(
                "query_transformer hidden_dim must be divisible by model_num_heads"
            )
        return QueryTransformerPredictor(
            **common_kwargs,
            depth=model_depth,
            dropout=model_dropout,
            num_heads=model_num_heads,
            memory_tokens=model_memory_tokens,
            ffn_multiplier=model_ffn_multiplier,
        )
    if model_architecture == "factorized":
        return FactorizedExpertPredictor(
            **common_kwargs,
            depth=model_depth,
            dropout=model_dropout,
            ffn_multiplier=model_ffn_multiplier,
        )
    if model_architecture == "frozen_router_delta":
        if num_layers is None:
            raise ValueError("num_layers is required for frozen_router_delta")
        if frozen_router_weights is None:
            raise ValueError(
                "frozen_router_weights is required for frozen_router_delta"
            )
        return FrozenRouterDeltaPredictor(
            **common_kwargs,
            num_layers=num_layers,
            router_weights=frozen_router_weights,
            depth=model_depth,
            dropout=model_dropout,
            ffn_multiplier=model_ffn_multiplier,
        )
    raise ValueError(f"unsupported predictor model_architecture: {model_architecture}")


def predictor_model_descriptor(
        *,
        model_architecture: str,
        hidden_dim: int,
        model_depth: int,
        model_dropout: float,
        model_num_heads: int,
        model_memory_tokens: int,
        model_ffn_multiplier: int,
) -> dict[str, Any]:
    return {
        "architecture": str(model_architecture),
        "hidden_dim": int(hidden_dim),
        "depth": int(model_depth),
        "dropout": float(model_dropout),
        "num_heads": int(model_num_heads),
        "memory_tokens": int(model_memory_tokens),
        "ffn_multiplier": int(model_ffn_multiplier),
    }
