"""测试 QwenMoEExpertMLP 和工具函数。"""

from __future__ import annotations

import pytest
import torch

from cfie_training.training_base.real_model_adapter import (
    QwenMoEExpertMLP,
    _group_expert_param_ids,
)


class TestQwenMoEExpertMLP:
    """QwenMoEExpertMLP 的真实 forward/backward。"""

    def test_load_weights_and_forward(self):
        hidden, inter = 64, 32
        mlp = QwenMoEExpertMLP(hidden, inter, dtype=torch.float32, device="cpu")
        w13 = torch.randn(2 * inter, hidden)
        w2 = torch.randn(hidden, inter)
        mlp.load_weights(w13, w2, requires_grad=True)

        x = torch.randn(2, 8, hidden)
        out = mlp.forward(x)
        assert out.shape == (2, 8, hidden)

    def test_forward_backward_produces_gradients(self):
        hidden, inter = 64, 32
        mlp = QwenMoEExpertMLP(hidden, inter, dtype=torch.float32, device="cpu")
        w13 = torch.randn(2 * inter, hidden, requires_grad=True)
        w2 = torch.randn(hidden, inter, requires_grad=True)
        mlp.load_weights(w13, w2, requires_grad=True)

        x = torch.randn(2, 8, hidden)
        out = mlp.forward(x)
        loss = out.pow(2).mean()
        loss.backward()

        grads = mlp.collect_gradients()
        assert "w13_weight" in grads
        assert "w2_weight" in grads
        assert not torch.allclose(grads["w13_weight"], torch.zeros_like(grads["w13_weight"]))

    def test_zero_grad_clears(self):
        hidden, inter = 64, 32
        mlp = QwenMoEExpertMLP(hidden, inter, dtype=torch.float32, device="cpu")
        mlp.load_weights(torch.randn(2 * inter, hidden), torch.randn(hidden, inter))
        x = torch.randn(1, 4, hidden)
        loss = mlp.forward(x).pow(2).mean()
        loss.backward()
        mlp.zero_grad()
        grads = mlp.collect_gradients()
        for g in grads.values():
            assert torch.allclose(g, torch.zeros_like(g))

    def test_sync_from_shadow(self):
        hidden, inter = 64, 32
        mlp = QwenMoEExpertMLP(hidden, inter, dtype=torch.float32, device="cpu")
        w13_flat = torch.randn(2 * inter * hidden, dtype=torch.float32)
        w2_flat = torch.randn(hidden * inter, dtype=torch.float32)
        mlp.sync_from_shadow(w13_flat, w2_flat)
        assert mlp.w13_weight is not None
        assert mlp.w2_weight is not None
        assert mlp.w13_weight.shape == (2 * inter, hidden)
        assert mlp.w2_weight.shape == (hidden, inter)


class TestGroupExpertParamIds:
    def test_groups_params_by_expert(self):
        groups = _group_expert_param_ids([
            "layers.0.experts.1.w13_weight",
            "layers.0.experts.1.w2_weight",
            "layers.3.experts.7.w13_weight",
            "layers.3.experts.7.w2_weight",
        ])
        assert len(groups) == 2
        assert "L0_E1" in groups
        assert "L3_E7" in groups

    def test_skips_incomplete_experts(self):
        groups = _group_expert_param_ids(["layers.0.experts.1.w13_weight"])
        assert len(groups) == 0
