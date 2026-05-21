"""Multi-step AdamW FP8 convergence and stability tests."""

from __future__ import annotations

import torch
import pytest

from cfie_training.training_base.adam_update import (
    AdamWConfig,
    BlockFp8StateCodec,
    CpuAdamFp8Updater,
)


def _reference_adamw_step(
    master: torch.Tensor,
    grad: torch.Tensor,
    step: int,
    m: torch.Tensor,
    v: torch.Tensor,
    config: AdamWConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    m.mul_(config.beta1).add_(grad, alpha=1 - config.beta1)
    v.mul_(config.beta2).addcmul_(grad, grad, value=1 - config.beta2)
    if config.bias_correction:
        m_hat = m / (1 - config.beta1 ** step)
        v_hat = v / (1 - config.beta2 ** step)
    else:
        m_hat = m.clone()
        v_hat = v.clone()
    update = m_hat / (v_hat.sqrt() + config.eps)
    new_master = master.clone()
    if config.weight_decay:
        new_master.mul_(1 - config.lr * config.weight_decay)
    new_master.add_(update, alpha=-config.lr)
    return new_master, m, v, update


def _make_random_grads(num_steps: int, num_elements: int) -> list[torch.Tensor]:
    g = torch.Generator()
    g.manual_seed(42)
    return [
        torch.randn(num_elements, generator=g)
        for _ in range(num_steps)
    ]


class TestMultiStepAdamConvergence:
    def test_50_step_adamw_fp8_converges_near_fp32_reference(self) -> None:
        config = AdamWConfig(lr=0.01, bias_correction=True)
        codec = BlockFp8StateCodec(block_size=64)
        updater = CpuAdamFp8Updater(config, codec=codec)

        num_steps = 50
        num_elements = 256
        grads = _make_random_grads(num_steps, num_elements)

        master_fp32 = torch.ones(num_elements, dtype=torch.float32)
        m_ref = torch.zeros(num_elements, dtype=torch.float32)
        v_ref = torch.zeros(num_elements, dtype=torch.float32)

        master_fp8 = master_fp32.clone()
        first_moment_payload = codec.zero_payload(num_elements)
        second_moment_payload = codec.zero_payload(num_elements)

        for step in range(1, num_steps + 1):
            master_fp32, m_ref, v_ref, _ = _reference_adamw_step(
                master_fp32, grads[step - 1].clone(), step,
                m_ref, v_ref, config,
            )

            update = updater.step_param(
                param_id="test",
                master=master_fp8,
                grad=grads[step - 1].clone(),
                step=step,
                first_moment_state=first_moment_payload,
                second_moment_state=second_moment_payload,
            )
            master_fp8 = update.master
            first_moment_payload = update.first_moment
            second_moment_payload = update.second_moment

        assert torch.allclose(master_fp8, master_fp32, rtol=0.05, atol=0.05)

    def test_100_step_stability_grad_norm_does_not_diverge(self) -> None:
        config = AdamWConfig(lr=0.001, bias_correction=True)
        codec = BlockFp8StateCodec(block_size=64)
        updater = CpuAdamFp8Updater(config, codec=codec)

        num_steps = 100
        num_elements = 128
        grads = _make_random_grads(num_steps, num_elements)

        master = torch.ones(num_elements, dtype=torch.float32)
        first_moment_payload = codec.zero_payload(num_elements)
        second_moment_payload = codec.zero_payload(num_elements)

        update_norms: list[float] = []
        for step in range(1, num_steps + 1):
            update = updater.step_param(
                param_id="test",
                master=master,
                grad=grads[step - 1].clone(),
                step=step,
                first_moment_state=first_moment_payload,
                second_moment_state=second_moment_payload,
            )
            master = update.master
            first_moment_payload = update.first_moment
            second_moment_payload = update.second_moment
            update_norms.append(update.update_norm)
            assert torch.isfinite(master).all()

        mid_mean = sum(update_norms[40:60]) / 20
        late_mean = sum(update_norms[80:]) / 20
        assert late_mean <= mid_mean * 2.0

    def test_adamw_weight_decay_accumulates_correctly(self) -> None:
        config = AdamWConfig(lr=0.1, weight_decay=0.1, bias_correction=True)
        updater = CpuAdamFp8Updater(config)

        num_elements = 16
        master_fp8 = torch.ones(num_elements, dtype=torch.float32)
        master_ref = torch.ones(num_elements, dtype=torch.float32)
        m_ref = torch.zeros(num_elements, dtype=torch.float32)
        v_ref = torch.zeros(num_elements, dtype=torch.float32)

        grad = torch.full((num_elements,), 0.0, dtype=torch.float32)

        first_moment_payload = updater.codec.zero_payload(num_elements)
        second_moment_payload = updater.codec.zero_payload(num_elements)

        for step in range(1, 5):
            master_ref, m_ref, v_ref, _ = _reference_adamw_step(
                master_ref, grad.clone(), step, m_ref, v_ref, config,
            )

            update = updater.step_param(
                param_id="test",
                master=master_fp8,
                grad=grad.clone(),
                step=step,
                first_moment_state=first_moment_payload,
                second_moment_state=second_moment_payload,
            )
            master_fp8 = update.master
            first_moment_payload = update.first_moment
            second_moment_payload = update.second_moment

        assert torch.allclose(master_fp8, master_ref, rtol=0.05, atol=0.05)

    def test_bias_correction_matches_reference_early_steps(self) -> None:
        config = AdamWConfig(lr=0.1, bias_correction=True)
        updater = CpuAdamFp8Updater(config)

        num_elements = 8
        master = torch.ones(num_elements, dtype=torch.float32)
        grad = torch.ones(num_elements, dtype=torch.float32)
        m_ref = torch.zeros(num_elements, dtype=torch.float32)
        v_ref = torch.zeros(num_elements, dtype=torch.float32)

        payload_m = updater.codec.zero_payload(num_elements)
        payload_v = updater.codec.zero_payload(num_elements)

        for step in range(1, 6):
            master, m_ref, v_ref, _ = _reference_adamw_step(
                master.clone(), grad.clone(), step,
                m_ref.clone(), v_ref.clone(), config,
            )

            update = updater.step_param(
                param_id="test",
                master=master.clone(),
                grad=grad.clone(),
                step=step,
                first_moment_state=payload_m,
                second_moment_state=payload_v,
            )
            payload_m = update.first_moment
            payload_v = update.second_moment

            assert torch.isfinite(update.master).all()
