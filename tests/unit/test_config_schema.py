"""Unit tests for CFIE configuration schema."""

from __future__ import annotations

import pytest

from cfie.config.schema import EngineConfig


def test_engine_config_from_flat_kwargs_defaults() -> None:
    cfg = EngineConfig.from_flat_kwargs(model="TinyLlama/TinyLlama-1.1B")

    assert cfg.model.model == "TinyLlama/TinyLlama-1.1B"
    assert cfg.model.dtype == "auto"
    assert cfg.scheduler.max_num_seqs == 1
    assert cfg.runtime.gpu_memory_utilization == pytest.approx(0.9)


def test_engine_config_from_nested_dict() -> None:
    cfg = EngineConfig.from_dict({
        "model": {
            "model": "./local-model",
            "dtype": "bf16",
            "max_model_len": 8192,
        },
        "quant": {
            "quantization": "none",
        },
        "scheduler": {
            "max_num_seqs": 1,
            "policy": "fcfs",
        },
    })

    assert cfg.model.dtype == "bf16"
    assert cfg.model.max_model_len == 8192


def test_invalid_dtype_raises_value_error() -> None:
    with pytest.raises(ValueError, match="dtype"):
        EngineConfig.from_flat_kwargs(model="m", dtype="fp32")


def test_invalid_gpu_memory_ratio_raises_value_error() -> None:
    with pytest.raises(ValueError, match="gpu_memory_utilization"):
        EngineConfig.from_flat_kwargs(model="m", gpu_memory_utilization=1.5)


def test_nvme_backend_disallows_cpu_budget() -> None:
    with pytest.raises(ValueError, match="cpu_offload_gb"):
        EngineConfig.from_flat_kwargs(
            model="m",
            weight_offload_backend="nvme",
            cpu_offload_gb=2.0,
        )


def test_cpu_plus_nvme_backend_is_accepted() -> None:
    cfg = EngineConfig.from_flat_kwargs(
        model="m",
        weight_offload_backend="cpu+nvme",
        kv_offload_backend="cpu+nvme",
    )
    assert cfg.offload.weight_offload_backend == "cpu+nvme"
    assert cfg.offload.kv_offload_backend == "cpu+nvme"
