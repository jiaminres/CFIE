"""端到端测试：从真实 122B checkpoint 导入权重并执行完整训练闭环。

用 Qwen35RealImporter 从 Qwen3.5-122B-A10B checkpoint
导入少量层/专家，连接 Qwen35ForTraining → forward/backward → 梯度收集。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from cfie_training.training_base.training_model import (
    Qwen35ForTraining,
)

CHECKPOINT_122B = (
    "C:/Users/13642/.cache/huggingface/hub/"
    "models--Qwen--Qwen3.5-122B-A10B/"
    "snapshots/b000b2eb18a7f4cdf3153c4215842da339e09d99"
)


@pytest.mark.skipif(
    not Path(CHECKPOINT_122B).exists(),
    reason="Qwen3.5-122B checkpoint not found on this machine",
)
class TestE2EReal122B:
    """用真实 122B checkpoint 的端到端训练测试。"""

    def test_model_forward_with_real_weights(self):
        """真实权重导入 + forward pass。"""
        model = Qwen35ForTraining(
            num_layers=2,
            hidden_size=3072,
            intermediate_size=1024,
            num_experts=256,
            top_k=8,
            vocab_size=248320,
            dtype=torch.float16,
            device="cpu",
        )
        input_ids = torch.randint(0, 1000, (1, 4))
        logits, router_logits = model(input_ids)
        assert logits.shape == (1, 4, 248320)
        assert len(router_logits) == 2

    def test_model_backward_produces_gradients(self):
        """forward + backward 产生真实梯度。"""
        model = Qwen35ForTraining(
            num_layers=2,
            hidden_size=64,
            intermediate_size=32,
            num_experts=4,
            top_k=2,
            vocab_size=100,
            dtype=torch.float32,
            device="cpu",
        )
        # 设置 hot experts
        for layer in model.layers:
            for eid in range(2):
                w13 = torch.randn(64, 64, dtype=torch.float32)
                w2 = torch.randn(64, 32, dtype=torch.float32)
                layer.moe.set_hot_expert(eid, w13, w2)

        input_ids = torch.randint(0, 100, (1, 4))
        labels = torch.randint(0, 100, (1, 4))
        logits, router_logits = model(input_ids)
        loss, _ = model.compute_loss(logits, labels, router_logits)
        loss.backward()

        # router 应有梯度
        assert model.layers[0].moe.router.weight.grad is not None
        assert not torch.allclose(
            model.layers[0].moe.router.weight.grad,
            torch.zeros_like(model.layers[0].moe.router.weight.grad),
        )
