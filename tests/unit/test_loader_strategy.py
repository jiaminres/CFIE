"""Unit tests for native + HF loader strategy."""

from __future__ import annotations

from types import SimpleNamespace

from cfie.config.schema import LoadConfig, ModelConfig
from cfie.loader.hf_loader import HFModelLoader
from cfie.model_executor.models.registry import ModelRegistry


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
