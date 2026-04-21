"""内置优先 + HF 兜底加载策略测试。"""

from __future__ import annotations

import json
from types import SimpleNamespace

from cfie.config.schema import LoadConfig, ModelConfig
from cfie.loader.hf_loader import HFModelLoader
from cfie.model_executor.models.config import MODELS_CONFIG_MAP
from cfie.model_executor.models.registry import ModelRegistry
from cfie.model_executor.models.qwen3_5_predictor import (
    Qwen3_5MoePredictorForCausalLM,
)
from cfie.transformers_utils.config import get_config


class _FakeLoadedModel:
    def __init__(self) -> None:
        self.moved_to: str | None = None
        self.is_eval = False

    def to(self, device: str):
        self.moved_to = device
        return self

    def eval(self):
        self.is_eval = True
        return self


def test_model_registry_resolves_builtin_gpt2() -> None:
    assert "GPT2LMHeadModel" in ModelRegistry.get_supported_archs()


def test_model_registry_resolves_qwen35_predictor_causallm() -> None:
    assert "Qwen3_5MoePredictorForCausalLM" in ModelRegistry.get_supported_archs()


def test_qwen35_predictor_causallm_is_marked_hybrid() -> None:
    assert getattr(Qwen3_5MoePredictorForCausalLM, "is_hybrid", False) is True


def test_qwen35_predictor_causallm_has_model_config_updater() -> None:
    assert "Qwen3_5MoePredictorForCausalLM" in MODELS_CONFIG_MAP


def test_get_config_supports_qwen35_predictor_text_model_type(tmp_path) -> None:
    model_dir = tmp_path / "qwen35_predictor_text"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5MoePredictorForCausalLM"],
                "model_type": "qwen3_5_moe_predictor_text",
                "vocab_size": 32,
                "hidden_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "moe_intermediate_size": 32,
                "shared_expert_intermediate_size": 32,
                "num_experts_per_tok": 2,
                "num_experts": 8,
                "layer_types": ["linear_attention", "full_attention"],
                "predictor_bundle_path": "bundle/predictor_bundle.json",
            }
        ),
        encoding="utf-8",
    )

    config = get_config(str(model_dir), trust_remote_code=False)

    assert config.model_type == "qwen3_5_moe_predictor_text"
    assert config.predictor_bundle_path == "bundle/predictor_bundle.json"


def test_qwen35_predictor_causallm_mapper_strips_mm_prefixes() -> None:
    mapped = Qwen3_5MoePredictorForCausalLM.hf_to_cfie_mapper.apply_list(
        [
            "model.visual.blocks.0.attn.qkv.weight",
            "model.language_model.layers.4.mlp.experts.0.gate_proj.qweight",
            "lm_head.weight",
        ]
    )

    assert mapped == [
        "model.layers.4.mlp.experts.0.gate_proj.qweight",
        "lm_head.weight",
    ]


def test_loader_prefers_native_model(monkeypatch, tmp_path) -> None:
    loader = HFModelLoader(device="cpu")
    model_cfg = ModelConfig(model=str(tmp_path))
    load_cfg = LoadConfig()
    fake_model = _FakeLoadedModel()

    monkeypatch.setattr(loader, "_resolve_model_dir",
                        lambda *_args, **_kwargs: str(tmp_path))
    monkeypatch.setattr(
        loader,
        "_load_hf_config",
        lambda *_args, **_kwargs: SimpleNamespace(
            architectures=["GPT2LMHeadModel"]),
    )
    monkeypatch.setattr(
        loader,
        "_resolve_native_model_cls",
        lambda arch, *_args, **_kwargs:
        object if arch == "GPT2LMHeadModel" else None,
    )
    monkeypatch.setattr(loader, "_build_native_model",
                        lambda *_args, **_kwargs: fake_model)

    def _no_hf_fallback(*_args, **_kwargs):
        raise AssertionError("HF fallback should not be called")

    monkeypatch.setattr(loader, "_build_hf_fallback_model", _no_hf_fallback)

    model = loader.load_model(model_cfg, load_cfg)
    assert model is fake_model
    assert fake_model.moved_to == "cpu"
    assert fake_model.is_eval is True


def test_loader_fallback_to_hf_when_native_missing(monkeypatch, tmp_path) -> None:
    loader = HFModelLoader(device="cpu")
    model_cfg = ModelConfig(model=str(tmp_path))
    load_cfg = LoadConfig()
    fake_model = _FakeLoadedModel()
    call = {"hf": 0}

    monkeypatch.setattr(loader, "_resolve_model_dir",
                        lambda *_args, **_kwargs: str(tmp_path))
    monkeypatch.setattr(
        loader,
        "_load_hf_config",
        lambda *_args, **_kwargs: SimpleNamespace(architectures=["UnknownArch"]),
    )
    monkeypatch.setattr(loader, "_resolve_native_model_cls",
                        lambda *_args, **_kwargs: None)

    def _hf_fallback(*_args, **_kwargs):
        call["hf"] += 1
        return fake_model

    monkeypatch.setattr(loader, "_build_hf_fallback_model", _hf_fallback)

    model = loader.load_model(model_cfg, load_cfg)
    assert model is fake_model
    assert call["hf"] == 1


def test_loader_fallback_to_hf_when_native_load_fails(monkeypatch, tmp_path) -> None:
    loader = HFModelLoader(device="cpu")
    model_cfg = ModelConfig(model=str(tmp_path))
    load_cfg = LoadConfig()
    fake_model = _FakeLoadedModel()
    call = {"hf": 0}

    monkeypatch.setattr(loader, "_resolve_model_dir",
                        lambda *_args, **_kwargs: str(tmp_path))
    monkeypatch.setattr(
        loader,
        "_load_hf_config",
        lambda *_args, **_kwargs: SimpleNamespace(
            architectures=["GPT2LMHeadModel"]),
    )
    monkeypatch.setattr(loader, "_resolve_native_model_cls",
                        lambda *_args, **_kwargs: object)

    def _native_fail(*_args, **_kwargs):
        raise RuntimeError("native build failed")

    def _hf_fallback(*_args, **_kwargs):
        call["hf"] += 1
        return fake_model

    monkeypatch.setattr(loader, "_build_native_model", _native_fail)
    monkeypatch.setattr(loader, "_build_hf_fallback_model", _hf_fallback)

    model = loader.load_model(model_cfg, load_cfg)
    assert model is fake_model
    assert call["hf"] == 1
