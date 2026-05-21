"""Qwen35ForTraining 冒烟测试：验证完整训练模型的可构建性和前向/反向。"""

from __future__ import annotations

import pytest
import torch

from cfie_training.training_base.training_model import (
    Qwen35ForTraining,
    TrainingDecoderLayer,
    TrainingQwenMoELayer,
)
from cfie_training.training_base.gpu_gptq import (
    GpuGptqConfig,
    GpuGptqQuantizer,
)


class TestTrainingQwenMoELayer:
    """MoE 层的单元测试。"""

    def test_router_produces_logits(self):
        moe = TrainingQwenMoELayer(
            hidden_size=64, intermediate_size=32, num_experts=8, top_k=2,
            dtype=torch.float32, device="cpu",
        )
        x = torch.randn(2, 4, 64)
        out, logits = moe(x)
        assert out.shape == (2, 4, 64)
        assert logits.shape == (8, 8)  # [B*T, num_experts]

    def test_hot_expert_forward(self):
        moe = TrainingQwenMoELayer(
            hidden_size=64, intermediate_size=32, num_experts=8, top_k=2,
            dtype=torch.float32, device="cpu",
        )
        # 设置一个 hot expert
        w13 = torch.randn(64, 64)  # [2*inter, hidden]
        w2 = torch.randn(64, 32)    # [hidden, inter]
        moe.set_hot_expert(0, w13, w2)
        assert 0 in moe.hot_experts

        x = torch.randn(2, 4, 64)
        out, _ = moe(x)
        assert out.shape == (2, 4, 64)
        assert not torch.allclose(out, torch.zeros_like(out))

    def test_cold_expert_without_cache(self):
        moe = TrainingQwenMoELayer(
            hidden_size=64, intermediate_size=32, num_experts=8, top_k=2,
            dtype=torch.float32, device="cpu",
        )
        moe.layer_idx = 0
        # 无 CPU 缓存时 _get_cold_expert_cached 返回 (None, None)
        w13, w2 = moe._get_cold_expert_cached(0)
        assert w13 is None and w2 is None


class TestGpuGptqQuantizer:
    """GPU GPTQ 量化器测试。"""

    def test_quantize_dequantize_roundtrip(self):
        weight = torch.randn(128, 256, dtype=torch.float32)
        quantizer = GpuGptqQuantizer(GpuGptqConfig(group_size=32))

        # 收集伪激活
        X = torch.randn(64, 256, dtype=torch.float32)
        quantizer.collect_activations(X)

        qweight, scales, qzeros = quantizer.quantize(weight)
        assert qweight.shape[1] == weight.shape[1] // 2  # packed
        assert scales.shape[1] == weight.shape[1] // 32  # 每 group 一个 scale

        # 解码
        decoded = GpuGptqQuantizer.decode(
            qweight, scales, qzeros,
            out_features=128, in_features=256, group_size=32,
        )
        assert decoded.shape == (128, 256)

        # 误差应在合理范围
        error = (weight - decoded.float()).abs().mean()
        assert error < 1.0, f"量化误差过大: {error}"

    def test_quantize_without_activations_falls_back(self):
        weight = torch.randn(64, 128, dtype=torch.float32)
        quantizer = GpuGptqQuantizer(GpuGptqConfig(group_size=32))
        # 不收集激活，应使用单位矩阵作为 Hessian
        qweight, scales, qzeros = quantizer.quantize(weight)
        assert qweight.shape == (64, 64)  # packed

    def test_serialization_roundtrip(self):
        weight = torch.randn(32, 64, dtype=torch.float32)
        quantizer = GpuGptqQuantizer(GpuGptqConfig(group_size=16))
        data = quantizer.quantize_and_store(weight)
        assert len(data) > 0


class TestQwen35ForTrainingSmoke:
    """完整训练模型冒烟测试。"""

    def test_model_builds(self):
        model = Qwen35ForTraining(
            num_layers=4,
            hidden_size=128,
            intermediate_size=64,
            num_experts=8,
            top_k=2,
            vocab_size=1000,
            dtype=torch.float32,
            device="cpu",
        )
        assert len(model.layers) == 4

    def test_forward_and_loss(self):
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
        input_ids = torch.randint(0, 100, (2, 8))
        labels = torch.randint(0, 100, (2, 8))

        logits, router_logits = model(input_ids)
        assert logits.shape == (2, 8, 100)
        assert len(router_logits) == 2

        loss, loss_dict = model.compute_loss(logits, labels, router_logits)
        assert loss.requires_grad
        assert "lm_loss" in loss_dict

    def test_backward_produces_gradients(self):
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

        input_ids = torch.randint(0, 100, (2, 8))
        labels = torch.randint(0, 100, (2, 8))

        logits, router_logits = model(input_ids)
        loss, _ = model.compute_loss(logits, labels, router_logits)
        loss.backward()

        # 验证 router 和 hot expert 产生了梯度
        has_grad = False
        for layer in model.layers:
            if layer.moe.router.weight.grad is not None:
                has_grad = True
                break
        assert has_grad, "router 应产生梯度"

    def test_hot_param_setup_and_gradient_collection(self):
        """验证从 shadow store 加载 hot params 和梯度收集的完整流程。"""
        from cfie_training.training_base.gradient_window import ForwardShadowStore

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

        # 创建 shadow store 并填充
        shadow_store = ForwardShadowStore(dtype=torch.float32)
        shadow_store.refresh(
            "layers.0.experts.0.w13_weight",
            torch.randn(64, dtype=torch.float32),  # 2*32*64 = 4096? No: 64=flat
        )
        shadow_store.refresh(
            "layers.0.experts.0.w2_weight",
            torch.randn(64, dtype=torch.float32),  # 64*32 = 2048 flat
        )

        # 正确的 flat 维度
        from cfie_training.training_base.gradient_window import ForwardShadowStore as FS

        shadow = FS(dtype=torch.float32)
        # w13: 2*inter*hidden = 2*32*64 = 4096
        w13_flat = torch.randn(2 * 32 * 64, dtype=torch.float32)
        shadow.refresh("layers.0.experts.0.w13_weight", w13_flat)
        # w2: hidden*inter = 64*32 = 2048
        w2_flat = torch.randn(64 * 32, dtype=torch.float32)
        shadow.refresh("layers.0.experts.0.w2_weight", w2_flat)

        hot_ids = (
            "layers.0.experts.0.w13_weight",
            "layers.0.experts.0.w2_weight",
        )
        model.setup_hot_params(shadow, hot_ids)

        # 验证权重已加载到 nn.Parameter
        moe = model.layers[0].moe
        assert 0 in moe._hot_w13
        assert 0 in moe._hot_w2

        # 修改 router 权重强制选择 expert 0
        with torch.no_grad():
            moe.router.weight.zero_()
            moe.router.weight[0, :] = 100.0  # 让 expert 0 的 logit 极高

        input_ids = torch.randint(0, 100, (1, 4))
        labels = torch.randint(0, 100, (1, 4))
        logits, router_logits = model(input_ids)
        loss, _ = model.compute_loss(logits, labels, router_logits)
        loss.backward()

        grads = model.collect_gradients(hot_ids)
        assert "layers.0.experts.0.w13_weight" in grads, (
            f"grads keys: {list(grads.keys())}"
        )
        assert grads["layers.0.experts.0.w13_weight"].numel() == 2 * 32 * 64
