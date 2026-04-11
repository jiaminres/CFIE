"""Hugging Face 模型加载器（内置优先，HF 兜底）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from cfie.config.schema import LoadConfig, ModelConfig
from cfie.loader.base_loader import BaseModelLoader
from cfie.loader.weight_utils import (
    PreparedWeights,
    get_total_bytes,
    iter_weight_tensors,
    prepare_weights,
)
from cfie.model_executor.models.registry import ModelRegistry
from cfie.utils.logging import get_logger

logger = get_logger(__name__)


def _resolve_torch_dtype(dtype: str):
    if dtype == "auto":
        return "auto"
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype}")


def _is_local_model(model_ref: str) -> bool:
    return Path(model_ref).expanduser().exists()


class HFModelLoader(BaseModelLoader):
    """Phase 1 加载策略：先走 CFIE 内置模型，缺失时回退 HF。"""

    def __init__(self, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model_dir_cache: dict[tuple[str, str | None, bool], str] = {}
        self._prepared_cache: dict[tuple[str, str], PreparedWeights] = {}
        self._loaded_model: Any | None = None

    def _resolve_model_dir(self, model_config: ModelConfig,
                           load_config: LoadConfig) -> str:
        """把模型引用统一解析成可复用的本地目录。"""

        key = (
            model_config.model,
            load_config.revision,
            load_config.local_files_only,
        )
        cached = self._model_dir_cache.get(key)
        if cached is not None:
            return cached

        if _is_local_model(model_config.model):
            model_dir = str(Path(model_config.model).expanduser())
        else:
            from huggingface_hub import snapshot_download

            model_dir = snapshot_download(
                repo_id=model_config.model,
                revision=load_config.revision,
                local_files_only=load_config.local_files_only,
                cache_dir=load_config.download_dir,
                ignore_patterns=list(load_config.ignore_patterns)
                if load_config.ignore_patterns else None,
            )

        self._model_dir_cache[key] = model_dir
        return model_dir

    def _prepare_weights(self, model_dir: str,
                         load_config: LoadConfig) -> PreparedWeights:
        key = (model_dir, load_config.load_format)
        cached = self._prepared_cache.get(key)
        if cached is not None:
            return cached

        prepared = prepare_weights(model_dir, load_config)
        self._prepared_cache[key] = prepared
        return prepared

    def _log_loader_stats(self, model_ref: str, prepared: PreparedWeights) -> None:
        source = "local" if _is_local_model(model_ref) else "remote"
        total_bytes = get_total_bytes(prepared.weight_files)
        logger.info(
            "model source(%s): files=%s bytes=%s tier_map=gpu:all format=%s",
            source,
            len(prepared.weight_files),
            total_bytes,
            "safetensors" if prepared.use_safetensors else "pt/bin",
        )

    def _load_hf_config(self, model_ref: str, load_config: LoadConfig) -> Any:
        from transformers import AutoConfig

        return AutoConfig.from_pretrained(
            model_ref,
            revision=load_config.revision,
            trust_remote_code=load_config.trust_remote_code,
            local_files_only=load_config.local_files_only,
        )

    def _resolve_primary_arch(self, hf_config: Any) -> str:
        architectures = list(getattr(hf_config, "architectures", []) or [])
        if not architectures:
            raise ValueError(
                "No 'architectures' field in model config; "
                "please provide an explicit architecture override")
        return str(architectures[0])

    def _resolve_native_model_cls(
        self,
        architecture: str,
        model_config: ModelConfig,
    ) -> type[nn.Module] | None:
        if not hasattr(model_config, "model_impl"):
            logger.info(
                "model_config type %s does not expose vLLM model registry "
                "fields; using HF loader path for arch=%s",
                type(model_config).__name__,
                architecture,
            )
            return None
        try:
            model_cls, _ = ModelRegistry.resolve_model_cls(
                [architecture],
                model_config,
            )
        except (RuntimeError, ValueError):
            logger.info(
                "no built-in model implementation resolved for arch=%s; "
                "falling back to HF loader path",
                architecture,
            )
            return None
        return model_cls

    def _copy_weights_into_model(self, model: nn.Module,
                                 prepared: PreparedWeights) -> None:
        named_params = dict(model.named_parameters())
        named_buffers = dict(model.named_buffers())

        loaded_count = 0
        skipped_count = 0
        loaded_bytes = 0

        with torch.no_grad():
            for name, tensor in iter_weight_tensors(prepared):
                target = named_params.get(name)
                if target is None:
                    target = named_buffers.get(name)

                if target is None:
                    skipped_count += 1
                    continue

                if tuple(target.shape) != tuple(tensor.shape):
                    raise ValueError(
                        f"Shape mismatch for {name!r}: "
                        f"expected {tuple(target.shape)}, got {tuple(tensor.shape)}")

                tensor_local = tensor.to(device=target.device, dtype=target.dtype)
                target.data.copy_(tensor_local)
                loaded_count += 1
                loaded_bytes += tensor_local.numel() * tensor_local.element_size()

        if loaded_count == 0:
            raise RuntimeError("No weights were loaded into model")

        logger.info(
            "weights loaded: tensors=%s skipped=%s bytes=%s",
            loaded_count,
            skipped_count,
            loaded_bytes,
        )

    def _load_weights_via_native_impl(self, model: nn.Module,
                                      prepared: PreparedWeights) -> None:
        """
        优先使用内置模型自己的 `load_weights`（vLLM 风格）。

        这样可以复用模型级别的命名映射/转置规则；
        若模型未提供该接口，再回退到通用逐参数拷贝。
        """

        load_fn = getattr(model, "load_weights", None)
        if callable(load_fn):
            loaded = load_fn(iter_weight_tensors(prepared))
            if isinstance(loaded, set):
                logger.info("native load_weights finished: tensors=%s",
                            len(loaded))
            else:
                logger.info("native load_weights finished")
            return

        self._copy_weights_into_model(model, prepared)

    def _build_native_model(self, model_cls: type[nn.Module], model_dir: str,
                            model_config: ModelConfig,
                            load_config: LoadConfig) -> nn.Module:
        hf_config = self._load_hf_config(model_dir, load_config)
        try:
            model = model_cls(hf_config)
        except TypeError:
            model = model_cls(config=hf_config)

        prepared = self._prepare_weights(model_dir, load_config)
        self._load_weights_via_native_impl(model, prepared)
        return model

    def _build_hf_fallback_model(self, model_dir: str, model_config: ModelConfig,
                                 load_config: LoadConfig) -> nn.Module:
        from transformers import AutoModelForCausalLM

        torch_dtype = _resolve_torch_dtype(model_config.dtype)
        # 兜底路径直接使用 HF 标准加载，保证广谱兼容性。
        return AutoModelForCausalLM.from_pretrained(
            model_dir,
            revision=load_config.revision,
            trust_remote_code=load_config.trust_remote_code,
            local_files_only=load_config.local_files_only,
            torch_dtype=torch_dtype,
        )

    def download_model(self,
                       model_config: ModelConfig,
                       load_config: LoadConfig) -> None:
        model_dir = self._resolve_model_dir(model_config, load_config)
        prepared = self._prepare_weights(model_dir, load_config)
        self._log_loader_stats(model_config.model, prepared)

    def load_weights(self,
                     model: Any,
                     model_config: ModelConfig,
                     load_config: LoadConfig) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("load_weights expects a torch.nn.Module model")
        model_dir = self._resolve_model_dir(model_config, load_config)
        prepared = self._prepare_weights(model_dir, load_config)
        self._copy_weights_into_model(model, prepared)

    def load_model(self,
                   model_config: ModelConfig,
                   load_config: LoadConfig) -> Any:
        model_dir = self._resolve_model_dir(model_config, load_config)
        hf_config = self._load_hf_config(model_dir, load_config)
        arch = self._resolve_primary_arch(hf_config)

        native_cls = self._resolve_native_model_cls(arch, model_config)
        backend = "hf_fallback_missing_native"
        model: nn.Module

        if native_cls is not None:
            logger.info("trying built-in model implementation: arch=%s class=%s",
                        arch, native_cls.__name__)
            try:
                model = self._build_native_model(
                    native_cls,
                    model_dir,
                    model_config,
                    load_config,
                )
                backend = "native"
            except Exception:
                logger.exception("built-in load failed, fallback to HF: arch=%s", arch)
                model = self._build_hf_fallback_model(model_dir, model_config,
                                                      load_config)
                backend = "hf_fallback_after_native_error"
        else:
            logger.warning(
                "built-in model is missing for arch=%s, fallback to HF path", arch)
            model = self._build_hf_fallback_model(model_dir, model_config,
                                                  load_config)

        model.to(self.device)
        model.eval()
        self._loaded_model = model
        logger.info("model loaded: model=%s arch=%s backend=%s device=%s dtype=%s",
                    model_config.model, arch, backend, self.device,
                    model_config.dtype)
        return model

    def load_tokenizer(self, model_config: ModelConfig,
                       load_config: LoadConfig) -> Any:
        from transformers import AutoTokenizer

        model_dir = self._resolve_model_dir(model_config, load_config)
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            revision=load_config.revision,
            trust_remote_code=load_config.trust_remote_code,
            local_files_only=load_config.local_files_only,
            use_fast=True,
        )
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("tokenizer loaded: model=%s", model_config.model)
        return tokenizer
