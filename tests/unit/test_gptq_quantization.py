"""Regression tests for GPTQ quantization selection."""

from __future__ import annotations

from cfie.model_executor.layers.fused_moe.layer import (
    FusedMoE,
    _should_enable_tiered_cache,
)
from cfie.model_executor.layers.quantization.gptq import GPTQConfig


class _SentinelMethod:
    pass


class _FakeGPTQMarlinConfig:
    def __init__(self, dynamic, *, desc_act: bool = False):
        self.dynamic = dynamic
        self.desc_act = desc_act
        self.group_size = 128

    @classmethod
    def get_name(cls):
        return "gptq_marlin"


def _make_gptq_config(**overrides) -> GPTQConfig:
    config = dict(
        weight_bits=4,
        group_size=128,
        desc_act=False,
        lm_head_quantized=False,
        dynamic={},
        autoround_version="",
        modules_in_block_to_quantize=[],
        checkpoint_format="",
    )
    config.update(overrides)
    return GPTQConfig(**config)


def test_gptq_fused_moe_respects_negative_dynamic_rule() -> None:
    layer = object.__new__(FusedMoE)
    layer.moe_config = object()
    config = _make_gptq_config(dynamic={"-:.*mtp.*": {}})

    method = config.get_quant_method(layer, "mtp.layers.0.mlp.experts")

    assert method is None


def test_gptq_fused_moe_uses_moe_wna16_when_not_excluded(monkeypatch) -> None:
    layer = object.__new__(FusedMoE)
    layer.moe_config = object()
    config = _make_gptq_config()
    sentinel = _SentinelMethod()

    class _FakeMoeWNA16Config:
        def get_quant_method(self, _layer, _prefix):
            return sentinel

    monkeypatch.setattr(
        "cfie.model_executor.layers.quantization.moe_wna16.MoeWNA16Config.from_config",
        lambda _config: _FakeMoeWNA16Config(),
    )

    method = config.get_quant_method(layer, "model.layers.0.mlp.experts")

    assert method is sentinel


def test_tiered_cache_allows_gptq_layers_excluded_by_dynamic_rule() -> None:
    layer = object.__new__(FusedMoE)
    config = _FakeGPTQMarlinConfig(dynamic={"-:.*mtp.*": {}})

    enabled = _should_enable_tiered_cache(
        config,
        layer,
        "mtp.layers.0.mlp.experts",
    )

    assert enabled is True


def test_tiered_cache_allows_non_mtp_layers_excluded_by_dynamic_rule() -> None:
    layer = object.__new__(FusedMoE)
    config = _FakeGPTQMarlinConfig(dynamic={"-:.*experts.*": {}})

    enabled = _should_enable_tiered_cache(
        config,
        layer,
        "model.layers.0.mlp.experts",
    )

    assert enabled is True


def test_tiered_cache_allows_unquantized_fused_moe() -> None:
    layer = object.__new__(FusedMoE)

    enabled = _should_enable_tiered_cache(
        None,
        layer,
        "model.layers.0.mlp.experts",
    )

    assert enabled is True


def test_tiered_cache_allows_gptq_desc_act() -> None:
    layer = object.__new__(FusedMoE)
    config = _FakeGPTQMarlinConfig(dynamic={}, desc_act=True)

    enabled = _should_enable_tiered_cache(
        config,
        layer,
        "model.layers.0.mlp.experts",
    )

    assert enabled is True
