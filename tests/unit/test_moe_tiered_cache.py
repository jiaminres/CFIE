from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from cfie.offload import weight_offload
from cfie.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
    select_unquantized_moe_backend,
)
from cfie.model_executor.layers.fused_moe.runner.default_moe_runner import (
    DefaultMoERunner,
)
from cfie.offload.cpu_backend import ExpertBundle, PinnedExpertCache
from cfie.offload.nvme_backend import SafetensorExpertStore
from cfie.offload.weight_offload import LayerTieredExpertCacheController
from cfie.offload.policy import (
    PLAN_KEY,
    MTP_RESERVE_MODE_KEY,
    TARGET_OCCUPIED_GPU_BYTES_KEY,
    _estimate_cpu_cache_budget_bytes,
    _estimate_hybrid_cache_bytes,
    _estimate_target_shared_gpu_reserve_bytes,
    build_moe_tiered_cache_plan,
)
from cfie.v1.spec_decode.eagle import SpecDecodeBaseProposer
from cfie.v1.spec_decode.utils import create_vllm_config_for_draft_model

GiB = 1 << 30


def _make_fake_cfie_config(
    model_dir: Path,
    *,
    num_experts: int = 4,
    num_experts_per_tok: int = 2,
    max_num_batched_tokens: int | None = None,
    cpu_offload_gb: float = 0.0,
    moe_cpu_budget_gb: float = 0.0,
    moe_cpu_min_free_gb: float = 0.0,
    model_type: str = "qwen3_5_moe",
    architectures: list[str] | None = None,
    quant_name: str = "gptq_marlin",
    quant_config: object | None = None,
) -> SimpleNamespace:
    if quant_config is None and quant_name == "gptq_marlin":
        quant_config = SimpleNamespace(desc_act=False)
        quant_config.get_name = lambda: "gptq_marlin"
    scheduler_config = SimpleNamespace(max_num_seqs=1)
    if max_num_batched_tokens is not None:
        scheduler_config.max_num_batched_tokens = max_num_batched_tokens
    return SimpleNamespace(
        model_config=SimpleNamespace(
            is_moe=True,
            quantization=quant_name,
            model=str(model_dir),
            dtype=torch.bfloat16,
            max_model_len=4096,
            hf_config=SimpleNamespace(
                model_type=model_type,
                architectures=list(architectures or []),
            ),
            hf_text_config=SimpleNamespace(
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                layer_types=["full_attention", "linear_attention"],
                num_key_value_heads=2,
                head_dim=128,
                linear_conv_kernel_dim=4,
                linear_key_head_dim=16,
                linear_num_key_heads=2,
                linear_value_head_dim=16,
                linear_num_value_heads=2,
                max_position_embeddings=4096,
                mamba_ssm_dtype="auto",
            ),
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=1, enable_eplb=False),
        scheduler_config=scheduler_config,
        cache_config=SimpleNamespace(
            gpu_memory_utilization=0.9,
            cache_dtype="auto",
            mamba_cache_dtype="auto",
            mamba_ssm_cache_dtype="auto",
        ),
        speculative_config=None,
        quant_config=quant_config,
        offload_config=SimpleNamespace(
            moe_cpu_budget_gb=moe_cpu_budget_gb,
            moe_cpu_min_free_gb=moe_cpu_min_free_gb,
            uva=SimpleNamespace(cpu_offload_gb=cpu_offload_gb),
        ),
    )


@dataclass
class _FakeParallelConfig:
    tensor_parallel_size: int = 1
    rank: int = 0


@dataclass
class _FakeSpeculativeConfig:
    draft_parallel_config: _FakeParallelConfig
    draft_model_config: object


@dataclass
class _FakeDraftCfieConfig:
    quant_config: object
    parallel_config: _FakeParallelConfig
    model_config: object
    speculative_config: _FakeSpeculativeConfig
    additional_config: dict


def test_build_moe_tiered_cache_plan_enabled(monkeypatch, tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    metadata = {
        "model.embed_tokens.weight": {"data_offsets": [0, 3 * GiB]},
    }
    for layer_idx in range(2):
        for expert_idx in range(4):
            base = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"
            metadata[f"{base}.gate_proj.qweight"] = {"data_offsets": [0, 2 * GiB]}
            metadata[f"{base}.down_proj.qweight"] = {"data_offsets": [0, 2 * GiB]}

    monkeypatch.setattr(
        "cfie.offload.policy.get_safetensors_params_metadata",
        lambda _model_path: metadata,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(total_memory=32 * GiB),
    )
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda _device: (24 * GiB, 32 * GiB),
    )
    monkeypatch.setattr(
        "cfie.offload.policy._get_available_system_ram_bytes",
        lambda: 64 * GiB,
    )
    monkeypatch.setattr(
        "cfie.offload.policy._get_total_system_ram_bytes",
        lambda: 128 * GiB,
    )

    plan = build_moe_tiered_cache_plan(_make_fake_cfie_config(model_dir))

    assert plan.enabled is True
    assert plan.gpu_slots_per_layer == 2
    assert plan.prefill_burst_slots == 0
    assert plan.gpu_budget_bytes == math.ceil(24 * GiB * 0.9)
    assert plan.cpu_budget_bytes >= plan.staging_bytes
    assert plan.cpu_static_bytes == (
        plan.cpu_slots_per_layer * plan.expert_bytes_per_slot_all_layers
    )
    assert plan.initial_gpu_experts == (0, 1)
    assert plan.cpu_slots_per_layer == plan.num_experts
    assert plan.initial_cpu_experts == (0, 1, 2, 3)
    assert plan.dense_bytes == 3 * GiB


def test_build_moe_tiered_cache_plan_enabled_for_unquantized_model(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    metadata = {
        "model.embed_tokens.weight": {"data_offsets": [0, 3 * GiB]},
    }
    for layer_idx in range(2):
        for expert_idx in range(4):
            base = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"
            metadata[f"{base}.gate_proj.weight"] = {"data_offsets": [0, 2 * GiB]}
            metadata[f"{base}.down_proj.weight"] = {"data_offsets": [0, 2 * GiB]}

    monkeypatch.setattr(
        "cfie.offload.policy.get_safetensors_params_metadata",
        lambda _model_path: metadata,
    )
    monkeypatch.setattr(
        "cfie.offload.policy._get_available_gpu_memory_snapshot",
        lambda: (24 * GiB, True),
    )
    monkeypatch.setattr(
        "cfie.offload.policy._get_available_system_ram_bytes",
        lambda: 96 * GiB,
    )
    monkeypatch.setattr(
        "cfie.offload.policy._get_total_system_ram_bytes",
        lambda: 128 * GiB,
    )

    plan = build_moe_tiered_cache_plan(
        _make_fake_cfie_config(
            model_dir,
            quant_name="",
            quant_config=None,
        )
    )

    assert plan.enabled is True
    assert plan.reason == "enabled"
    assert plan.quantization == "unquantized"


@pytest.mark.parametrize(
    "model_type",
    ["qwen3_5_moe_predictor", "qwen3_5_moe_predictor_text"],
)
def test_build_moe_tiered_cache_plan_supports_qwen35_predictor_model_types(
    monkeypatch,
    tmp_path: Path,
    model_type: str,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    metadata = {
        "model.embed_tokens.weight": {"data_offsets": [0, 3 * GiB]},
    }
    for layer_idx in range(2):
        for expert_idx in range(4):
            base = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"
            metadata[f"{base}.gate_proj.qweight"] = {"data_offsets": [0, 2 * GiB]}
            metadata[f"{base}.down_proj.qweight"] = {"data_offsets": [0, 2 * GiB]}

    monkeypatch.setattr(
        "cfie.offload.policy.get_safetensors_params_metadata",
        lambda _model_path: metadata,
    )
    monkeypatch.setattr(
        "cfie.offload.policy._get_available_gpu_memory_snapshot",
        lambda: (24 * GiB, True),
    )
    monkeypatch.setattr(
        "cfie.offload.policy._get_available_system_ram_bytes",
        lambda: 96 * GiB,
    )
    monkeypatch.setattr(
        "cfie.offload.policy._get_total_system_ram_bytes",
        lambda: 128 * GiB,
    )

    plan = build_moe_tiered_cache_plan(
        _make_fake_cfie_config(
            model_dir,
            model_type=model_type,
        )
    )

    assert plan.enabled is True
    assert plan.reason == "enabled"


def test_build_moe_tiered_cache_plan_allows_gptq_desc_act(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    metadata = {
        "model.embed_tokens.weight": {"data_offsets": [0, 3 * GiB]},
    }
    for layer_idx in range(2):
        for expert_idx in range(4):
            base = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"
            metadata[f"{base}.gate_proj.qweight"] = {"data_offsets": [0, 2 * GiB]}
            metadata[f"{base}.down_proj.qweight"] = {"data_offsets": [0, 2 * GiB]}

    monkeypatch.setattr(
        "cfie.offload.policy.get_safetensors_params_metadata",
        lambda _model_path: metadata,
    )
    monkeypatch.setattr(
        "cfie.offload.policy._get_available_gpu_memory_snapshot",
        lambda: (24 * GiB, True),
    )
    monkeypatch.setattr(
        "cfie.offload.policy._get_available_system_ram_bytes",
        lambda: 96 * GiB,
    )
    monkeypatch.setattr(
        "cfie.offload.policy._get_total_system_ram_bytes",
        lambda: 128 * GiB,
    )

    quant_config = SimpleNamespace(desc_act=True)
    quant_config.get_name = lambda: "gptq_marlin"

    plan = build_moe_tiered_cache_plan(
        _make_fake_cfie_config(
            model_dir,
            quant_name="gptq_marlin",
            quant_config=quant_config,
        )
    )

    assert plan.enabled is True
    assert plan.reason == "enabled"
    assert plan.quantization == "gptq_marlin"


def test_build_moe_tiered_cache_plan_resolves_repo_id_before_reading_metadata(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "resolved-model"
    model_dir.mkdir()

    metadata = {
        "model.embed_tokens.weight": {"data_offsets": [0, 3 * GiB]},
    }
    for layer_idx in range(2):
        for expert_idx in range(4):
            base = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"
            metadata[f"{base}.gate_proj.qweight"] = {"data_offsets": [0, 2 * GiB]}
            metadata[f"{base}.down_proj.qweight"] = {"data_offsets": [0, 2 * GiB]}

    def _metadata_reader(resolved_path: str) -> dict[str, dict[str, list[int]]]:
        assert resolved_path == str(model_dir)
        return metadata

    monkeypatch.setattr(
        "cfie.offload.policy.get_safetensors_params_metadata",
        _metadata_reader,
    )
    monkeypatch.setattr(
        "cfie.offload.policy.prepare_weights",
        lambda model_or_path, load_config: SimpleNamespace(
            model_dir=str(model_dir),
            weight_files=[],
            use_safetensors=True,
        ),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(total_memory=32 * GiB),
    )
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda _device: (24 * GiB, 32 * GiB),
    )
    monkeypatch.setattr(
        "cfie.offload.policy._get_available_system_ram_bytes",
        lambda: 64 * GiB,
    )
    monkeypatch.setattr(
        "cfie.offload.policy._get_total_system_ram_bytes",
        lambda: 128 * GiB,
    )

    cfg = _make_fake_cfie_config(tmp_path)
    cfg.model_config.model = "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"
    cfg.load_config = SimpleNamespace(load_format="auto")

    plan = build_moe_tiered_cache_plan(cfg)

    assert plan.enabled is True
    assert plan.model_path == str(model_dir)


def test_build_moe_tiered_cache_plan_disables_prefill_burst_when_scheduler_budget_is_too_small(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    metadata = {
        "model.embed_tokens.weight": {"data_offsets": [0, 3 * GiB]},
    }
    for layer_idx in range(2):
        for expert_idx in range(4):
            base = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"
            metadata[f"{base}.gate_proj.qweight"] = {"data_offsets": [0, 2 * GiB]}
            metadata[f"{base}.down_proj.qweight"] = {"data_offsets": [0, 2 * GiB]}

    monkeypatch.setattr(
        "cfie.offload.policy.get_safetensors_params_metadata",
        lambda _model_path: metadata,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(total_memory=32 * GiB),
    )
    monkeypatch.setattr(
        "cfie.offload.policy._get_available_system_ram_bytes",
        lambda: 64 * GiB,
    )
    monkeypatch.setattr(
        "cfie.offload.policy._get_total_system_ram_bytes",
        lambda: 128 * GiB,
    )

    plan = build_moe_tiered_cache_plan(
        _make_fake_cfie_config(model_dir, max_num_batched_tokens=4)
    )

    assert plan.enabled is True
    assert plan.gpu_slots_per_layer == 2
    assert plan.prefill_burst_slots == 0
    assert plan.prefill_burst_bytes == 0
    assert plan.gpu_expert_budget_bytes == (
        plan.gpu_slots_per_layer * plan.expert_bytes_per_slot_all_layers
    )


def test_build_moe_tiered_cache_plan_disables_tiered_cache_when_all_experts_fit_on_gpu(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    metadata = {
        "model.embed_tokens.weight": {"data_offsets": [0, 3 * GiB]},
    }
    for expert_idx in range(4):
        base = f"model.layers.0.mlp.experts.{expert_idx}"
        metadata[f"{base}.gate_proj.qweight"] = {"data_offsets": [0, int(1.5 * GiB)]}
        metadata[f"{base}.down_proj.qweight"] = {"data_offsets": [0, int(1.5 * GiB)]}

    monkeypatch.setattr(
        "cfie.offload.policy.get_safetensors_params_metadata",
        lambda _model_path: metadata,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(total_memory=32 * GiB),
    )
    monkeypatch.setattr(
        "cfie.offload.policy._get_available_system_ram_bytes",
        lambda: 64 * GiB,
    )
    monkeypatch.setattr(
        "cfie.offload.policy._get_total_system_ram_bytes",
        lambda: 128 * GiB,
    )

    plan = build_moe_tiered_cache_plan(_make_fake_cfie_config(model_dir))

    assert plan.enabled is False
    assert plan.reason == "full_gpu_residency_already_possible"
    assert plan.gpu_slots_per_layer == 4
    assert plan.gpu_expert_budget_bytes == plan.expert_bytes_total


def test_build_moe_tiered_cache_plan_reserves_dynamic_headroom_for_qwen35_target(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    metadata = {
        "model.embed_tokens.weight": {"data_offsets": [0, 64 * (1 << 20)]},
    }
    expert_proj_bytes = 160 * (1 << 20)
    for expert_idx in range(16):
        base = f"model.layers.0.mlp.experts.{expert_idx}"
        metadata[f"{base}.gate_proj.qweight"] = {"data_offsets": [0, expert_proj_bytes]}
        metadata[f"{base}.down_proj.qweight"] = {"data_offsets": [0, expert_proj_bytes]}

    monkeypatch.setattr(
        "cfie.offload.policy.get_safetensors_params_metadata",
        lambda _model_path: metadata,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(total_memory=8 * GiB),
    )
    monkeypatch.setattr(
        "cfie.offload.policy._get_available_system_ram_bytes",
        lambda: 64 * GiB,
    )
    monkeypatch.setattr(
        "cfie.offload.policy._get_total_system_ram_bytes",
        lambda: 128 * GiB,
    )

    cfg = _make_fake_cfie_config(
        model_dir,
        num_experts=16,
        num_experts_per_tok=2,
    )

    plan = build_moe_tiered_cache_plan(cfg)

    assert plan.enabled is True
    assert plan.reason == "enabled"
    assert plan.dynamic_bytes == 3 * GiB
    assert plan.gpu_slots_per_layer < 16


def test_estimate_target_shared_gpu_reserve_bytes_uses_mtp_draft_plan(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cfg = _make_fake_cfie_config(tmp_path)
    cfg.speculative_config = SimpleNamespace(
        method="mtp",
        draft_model_config=object(),
    )
    captured: dict[str, object] = {}
    draft_cfg = SimpleNamespace(
        additional_config={
            PLAN_KEY: {
                "resident_bytes": 2 * GiB,
                "gpu_expert_budget_bytes": 3 * GiB,
            }
        }
    )

    def _fake_create_draft_cfg(_cfg, additional_config_updates=None):
        captured["updates"] = dict(additional_config_updates or {})
        return draft_cfg

    monkeypatch.setattr(
        "cfie.v1.spec_decode.utils.create_vllm_config_for_draft_model",
        _fake_create_draft_cfg,
    )

    reserve_bytes = _estimate_target_shared_gpu_reserve_bytes(
        cfie_config=cfg,
    )

    assert captured["updates"] == {
        MTP_RESERVE_MODE_KEY: True,
        TARGET_OCCUPIED_GPU_BYTES_KEY: 0,
    }
    assert reserve_bytes == 5 * GiB


def test_build_moe_tiered_cache_plan_reserves_gpu_budget_for_mtp_draft(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    metadata = {
        "model.embed_tokens.weight": {"data_offsets": [0, 3 * GiB]},
    }
    for expert_idx in range(4):
        base = f"model.layers.0.mlp.experts.{expert_idx}"
        metadata[f"{base}.gate_proj.qweight"] = {"data_offsets": [0, int(1.5 * GiB)]}
        metadata[f"{base}.down_proj.qweight"] = {"data_offsets": [0, int(1.5 * GiB)]}

    monkeypatch.setattr(
        "cfie.offload.policy.get_safetensors_params_metadata",
        lambda _model_path: metadata,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(total_memory=32 * GiB),
    )
    monkeypatch.setattr(
        "cfie.offload.policy._get_available_system_ram_bytes",
        lambda: 64 * GiB,
    )
    monkeypatch.setattr(
        "cfie.offload.policy._get_total_system_ram_bytes",
        lambda: 128 * GiB,
    )
    monkeypatch.setattr(
        "cfie.offload.policy._estimate_target_shared_gpu_reserve_bytes",
        lambda **_kwargs: 14 * GiB,
    )

    cfg = _make_fake_cfie_config(model_dir)
    cfg.speculative_config = SimpleNamespace(
        method="mtp",
        draft_model_config=object(),
        num_speculative_tokens=1,
    )

    plan = build_moe_tiered_cache_plan(cfg)

    assert plan.enabled is True
    assert plan.reason == "enabled"
    assert plan.shared_gpu_reserve_bytes == 14 * GiB
    assert plan.gpu_slots_per_layer == cfg.model_config.hf_text_config.num_experts_per_tok
    assert plan.prefill_burst_slots == 1
    assert plan.gpu_expert_budget_bytes == (
        plan.gpu_slots_per_layer * plan.expert_bytes_per_slot_all_layers
        + plan.prefill_burst_bytes
    )


def test_build_moe_tiered_cache_plan_keeps_minimal_base_for_partial_burst(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    metadata = {
        "model.embed_tokens.weight": {"data_offsets": [0, 2 * GiB]},
    }
    for expert_idx in range(8):
        base = f"model.layers.0.mlp.experts.{expert_idx}"
        metadata[f"{base}.gate_proj.qweight"] = {"data_offsets": [0, int(1.5 * GiB)]}
        metadata[f"{base}.down_proj.qweight"] = {"data_offsets": [0, int(1.5 * GiB)]}

    monkeypatch.setattr(
        "cfie.offload.policy.get_safetensors_params_metadata",
        lambda _model_path: metadata,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(total_memory=24 * GiB),
    )
    monkeypatch.setattr(
        "cfie.offload.policy._get_available_system_ram_bytes",
        lambda: 64 * GiB,
    )
    monkeypatch.setattr(
        "cfie.offload.policy._get_total_system_ram_bytes",
        lambda: 128 * GiB,
    )

    cfg = _make_fake_cfie_config(model_dir, num_experts=8)
    cfg.model_config.hf_text_config.layer_types = ["full_attention"]

    plan = build_moe_tiered_cache_plan(cfg)

    assert plan.enabled is True
    assert plan.reason == "enabled"
    assert plan.gpu_slots_per_layer == cfg.model_config.hf_text_config.num_experts_per_tok
    assert 0 < plan.prefill_burst_slots < plan.num_experts


def test_build_moe_tiered_cache_plan_raises_when_gpu_cannot_fit_topk(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    metadata = {
        "model.embed_tokens.weight": {"data_offsets": [0, 3 * GiB]},
    }
    for layer_idx in range(2):
        for expert_idx in range(4):
            base = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"
            metadata[f"{base}.gate_proj.qweight"] = {"data_offsets": [0, 2 * GiB]}
            metadata[f"{base}.down_proj.qweight"] = {"data_offsets": [0, 2 * GiB]}

    monkeypatch.setattr(
        "cfie.offload.policy.get_safetensors_params_metadata",
        lambda _model_path: metadata,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(total_memory=18 * GiB),
    )
    monkeypatch.setattr(
        "cfie.offload.policy._get_available_system_ram_bytes",
        lambda: 64 * GiB,
    )
    monkeypatch.setattr(
        "cfie.offload.policy._get_total_system_ram_bytes",
        lambda: 128 * GiB,
    )

    with pytest.raises(ValueError, match="Insufficient GPU budget for MoE tiered cache"):
        build_moe_tiered_cache_plan(_make_fake_cfie_config(model_dir))


def test_build_moe_tiered_cache_plan_uses_worst_case_layer_expert_size(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    metadata = {
        "model.embed_tokens.weight": {"data_offsets": [0, 3 * GiB]},
    }
    for expert_idx in range(4):
        small = f"model.layers.0.mlp.experts.{expert_idx}"
        metadata[f"{small}.gate_proj.qweight"] = {"data_offsets": [0, 1 * GiB]}
        metadata[f"{small}.down_proj.qweight"] = {"data_offsets": [0, 1 * GiB]}

        large = f"model.layers.1.mlp.experts.{expert_idx}"
        metadata[f"{large}.gate_proj.qweight"] = {"data_offsets": [0, 3 * GiB]}
        metadata[f"{large}.down_proj.qweight"] = {"data_offsets": [0, 3 * GiB]}

    monkeypatch.setattr(
        "cfie.offload.policy.get_safetensors_params_metadata",
        lambda _model_path: metadata,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(total_memory=32 * GiB),
    )
    monkeypatch.setattr(
        "cfie.offload.policy._get_available_system_ram_bytes",
        lambda: 64 * GiB,
    )
    monkeypatch.setattr(
        "cfie.offload.policy._get_total_system_ram_bytes",
        lambda: 128 * GiB,
    )

    plan = build_moe_tiered_cache_plan(_make_fake_cfie_config(model_dir))

    assert plan.enabled is True
    assert plan.gpu_slots_per_layer == 2
    assert plan.expert_bytes_per_expert == 6 * GiB
    assert plan.expert_bytes_per_slot_all_layers == 8 * GiB


def test_build_moe_tiered_cache_plan_disables_tiered_cache_when_deep_moe_still_fits_fully_on_gpu(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    metadata = {
        "model.embed_tokens.weight": {"data_offsets": [0, 3 * GiB]},
    }
    expert_bytes = 384 * (1 << 20)
    for layer_idx in range(8):
        for expert_idx in range(4):
            base = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"
            metadata[f"{base}.gate_proj.qweight"] = {"data_offsets": [0, expert_bytes]}
            metadata[f"{base}.down_proj.qweight"] = {"data_offsets": [0, expert_bytes]}

    monkeypatch.setattr(
        "cfie.offload.policy.get_safetensors_params_metadata",
        lambda _model_path: metadata,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(total_memory=32 * GiB),
    )
    monkeypatch.setattr(
        "cfie.offload.policy._get_available_system_ram_bytes",
        lambda: 64 * GiB,
    )
    monkeypatch.setattr(
        "cfie.offload.policy._get_total_system_ram_bytes",
        lambda: 128 * GiB,
    )

    cfg = _make_fake_cfie_config(model_dir)
    cfg.model_config.hf_text_config.layer_types = ["full_attention"] * 8

    plan = build_moe_tiered_cache_plan(cfg)

    assert plan.enabled is False
    assert plan.reason == "full_gpu_residency_already_possible"
    assert plan.gpu_slots_per_layer == 4
    assert plan.gpu_expert_budget_bytes == plan.expert_bytes_total


def test_build_moe_tiered_cache_plan_keeps_minimal_base_when_full_burst_fits(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    metadata = {
        "model.embed_tokens.weight": {"data_offsets": [0, 3 * GiB]},
    }
    expert_bytes = 384 * (1 << 20)
    for layer_idx in range(8):
        for expert_idx in range(8):
            base = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"
            metadata[f"{base}.gate_proj.qweight"] = {"data_offsets": [0, expert_bytes]}
            metadata[f"{base}.down_proj.qweight"] = {"data_offsets": [0, expert_bytes]}

    monkeypatch.setattr(
        "cfie.offload.policy.get_safetensors_params_metadata",
        lambda _model_path: metadata,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(total_memory=40 * GiB),
    )
    monkeypatch.setattr(
        "cfie.offload.policy._get_available_system_ram_bytes",
        lambda: 64 * GiB,
    )
    monkeypatch.setattr(
        "cfie.offload.policy._get_total_system_ram_bytes",
        lambda: 128 * GiB,
    )

    cfg = _make_fake_cfie_config(model_dir, num_experts=8)
    cfg.model_config.hf_text_config.layer_types = ["full_attention"] * 8

    plan = build_moe_tiered_cache_plan(cfg)

    assert plan.enabled is True
    assert plan.gpu_slots_per_layer == 4
    assert plan.prefill_burst_slots == 8
    assert plan.gpu_expert_budget_bytes == (
        plan.prefill_burst_bytes
        + plan.gpu_slots_per_layer * plan.expert_bytes_per_slot_all_layers
    )


def test_create_vllm_config_for_draft_model_drops_inherited_tiered_cache_plan() -> None:
    draft_model = object()
    cfg = _FakeDraftCfieConfig(
        quant_config=object(),
        parallel_config=_FakeParallelConfig(rank=7),
        model_config=object(),
        speculative_config=_FakeSpeculativeConfig(
            draft_parallel_config=_FakeParallelConfig(rank=0),
            draft_model_config=draft_model,
        ),
        additional_config={
            PLAN_KEY: {"enabled": True, "gpu_slots_per_layer": 8},
            TARGET_OCCUPIED_GPU_BYTES_KEY: 123,
            "keep": "value",
        },
    )

    draft_cfg = create_vllm_config_for_draft_model(
        cfg,
        additional_config_updates={MTP_RESERVE_MODE_KEY: True},
    )

    assert PLAN_KEY in cfg.additional_config
    assert PLAN_KEY not in draft_cfg.additional_config
    assert draft_cfg.additional_config["keep"] == "value"
    assert draft_cfg.additional_config[TARGET_OCCUPIED_GPU_BYTES_KEY] == 123
    assert draft_cfg.additional_config[MTP_RESERVE_MODE_KEY] is True
    assert draft_cfg.model_config is draft_model
    assert draft_cfg.parallel_config.rank == cfg.parallel_config.rank


def test_build_moe_tiered_cache_plan_keeps_qwen35_mtp_base_8_in_reserve_mode(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    metadata = {
        "model.embed_tokens.weight": {"data_offsets": [0, 3 * GiB]},
        "mtp.layers.0.input_layernorm.weight": {
            "data_offsets": [0, 64 * (1 << 20)]
        },
    }
    mtp_proj_bytes = 128 * (1 << 20)
    for expert_idx in range(16):
        base = f"mtp.layers.0.mlp.experts.{expert_idx}"
        metadata[f"{base}.gate_proj.weight"] = {"data_offsets": [0, mtp_proj_bytes]}
        metadata[f"{base}.down_proj.weight"] = {"data_offsets": [0, mtp_proj_bytes]}

    monkeypatch.setattr(
        "cfie.offload.policy.get_safetensors_params_metadata",
        lambda _model_path: metadata,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(total_memory=32 * GiB),
    )
    monkeypatch.setattr(
        "cfie.offload.policy._get_available_system_ram_bytes",
        lambda: 64 * GiB,
    )
    monkeypatch.setattr(
        "cfie.offload.policy._get_total_system_ram_bytes",
        lambda: 128 * GiB,
    )

    cfg = _make_fake_cfie_config(
        model_dir,
        num_experts=16,
        num_experts_per_tok=8,
        model_type="qwen3_5_mtp",
        architectures=["Qwen3_5MoeMTP"],
    )
    cfg.model_config.hf_text_config.layer_types = ["full_attention"] * 48
    cfg.additional_config = {MTP_RESERVE_MODE_KEY: True}

    plan = build_moe_tiered_cache_plan(cfg)

    assert plan.enabled is True
    assert plan.reason == "enabled"
    assert plan.gpu_slots_per_layer == 8
    assert plan.prefill_burst_slots == 0
    assert plan.gpu_expert_budget_bytes == 8 * plan.expert_bytes_per_slot_all_layers



def test_build_moe_tiered_cache_plan_disables_qwen35_mtp_tiered_cache_when_all_experts_fit_on_gpu(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    metadata = {
        "model.embed_tokens.weight": {"data_offsets": [0, 3 * GiB]},
        "mtp.layers.0.input_layernorm.weight": {
            "data_offsets": [0, 64 * (1 << 20)]
        },
    }
    for layer_idx in range(2):
        for expert_idx in range(16):
            base = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"
            metadata[f"{base}.gate_proj.qweight"] = {"data_offsets": [0, 1 * GiB]}
            metadata[f"{base}.down_proj.qweight"] = {"data_offsets": [0, 1 * GiB]}

    mtp_proj_bytes = 128 * (1 << 20)
    for expert_idx in range(16):
        base = f"mtp.layers.0.mlp.experts.{expert_idx}"
        metadata[f"{base}.gate_proj.weight"] = {"data_offsets": [0, mtp_proj_bytes]}
        metadata[f"{base}.down_proj.weight"] = {"data_offsets": [0, mtp_proj_bytes]}

    monkeypatch.setattr(
        "cfie.offload.policy.get_safetensors_params_metadata",
        lambda _model_path: metadata,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(total_memory=32 * GiB),
    )
    monkeypatch.setattr(
        "cfie.offload.policy._get_available_system_ram_bytes",
        lambda: 64 * GiB,
    )
    monkeypatch.setattr(
        "cfie.offload.policy._get_total_system_ram_bytes",
        lambda: 128 * GiB,
    )

    cfg = _make_fake_cfie_config(
        model_dir,
        num_experts=16,
        num_experts_per_tok=8,
        model_type="qwen3_5_mtp",
        architectures=["Qwen3_5MoeMTP"],
    )
    cfg.model_config.hf_text_config.layer_types = ["full_attention"] * 48

    plan = build_moe_tiered_cache_plan(cfg)

    assert plan.enabled is False
    assert plan.reason == "full_gpu_residency_already_possible"
    assert plan.num_moe_layers == 1
    assert plan.dense_bytes == 64 * (1 << 20)
    assert plan.expert_bytes_per_expert == 2 * mtp_proj_bytes
    assert plan.expert_bytes_per_slot_all_layers == 2 * mtp_proj_bytes
    assert plan.gpu_slots_per_layer == 16
    assert plan.prefill_burst_slots == 0
    assert plan.gpu_expert_budget_bytes == 16 * (2 * mtp_proj_bytes)
    assert plan.cpu_slots_per_layer == 0
    assert plan.nvme_expert_bytes == 0
    assert plan.shared_gpu_reserve_bytes == 0


def test_build_moe_tiered_cache_plan_caps_qwen35_mtp_slots_after_target_gpu_is_reserved(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    metadata = {
        "model.embed_tokens.weight": {"data_offsets": [0, 64 * (1 << 20)]},
        "mtp.layers.0.input_layernorm.weight": {
            "data_offsets": [0, 64 * (1 << 20)]
        },
    }
    mtp_proj_bytes = 128 * (1 << 20)
    for expert_idx in range(16):
        base = f"mtp.layers.0.mlp.experts.{expert_idx}"
        metadata[f"{base}.gate_proj.weight"] = {"data_offsets": [0, mtp_proj_bytes]}
        metadata[f"{base}.down_proj.weight"] = {"data_offsets": [0, mtp_proj_bytes]}

    monkeypatch.setattr(
        "cfie.offload.policy.get_safetensors_params_metadata",
        lambda _model_path: metadata,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(total_memory=4 * GiB),
    )
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda _device: (_ for _ in ()).throw(RuntimeError("unavailable")),
    )
    monkeypatch.setattr(
        "cfie.offload.policy._get_available_system_ram_bytes",
        lambda: 64 * GiB,
    )
    monkeypatch.setattr(
        "cfie.offload.policy._get_total_system_ram_bytes",
        lambda: 128 * GiB,
    )

    cfg = _make_fake_cfie_config(
        model_dir,
        num_experts=16,
        num_experts_per_tok=8,
        model_type="qwen3_5_mtp",
        architectures=["Qwen3_5MoeMTP"],
    )
    cfg.model_config.hf_text_config.layer_types = ["full_attention"] * 48
    cfg.additional_config = {
        TARGET_OCCUPIED_GPU_BYTES_KEY: 1 * GiB,
    }

    plan = build_moe_tiered_cache_plan(cfg)

    assert plan.enabled is True
    assert plan.reason == "enabled"
    assert plan.shared_gpu_reserve_bytes == 1 * GiB
    assert plan.gpu_slots_per_layer == 8
    assert plan.prefill_burst_slots == 0
    assert plan.gpu_expert_budget_bytes == 8 * plan.expert_bytes_per_slot_all_layers


def test_build_moe_tiered_cache_plan_does_not_double_count_target_gpu_when_free_memory_is_available(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    metadata = {
        "model.embed_tokens.weight": {"data_offsets": [0, 64 * (1 << 20)]},
        "mtp.layers.0.input_layernorm.weight": {
            "data_offsets": [0, 64 * (1 << 20)]
        },
    }
    mtp_proj_bytes = 128 * (1 << 20)
    for expert_idx in range(16):
        base = f"mtp.layers.0.mlp.experts.{expert_idx}"
        metadata[f"{base}.gate_proj.weight"] = {"data_offsets": [0, mtp_proj_bytes]}
        metadata[f"{base}.down_proj.weight"] = {"data_offsets": [0, mtp_proj_bytes]}

    monkeypatch.setattr(
        "cfie.offload.policy.get_safetensors_params_metadata",
        lambda _model_path: metadata,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(total_memory=32 * GiB),
    )
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda _device: (int(4.4 * GiB), 32 * GiB),
    )
    monkeypatch.setattr(
        "cfie.offload.policy._get_available_system_ram_bytes",
        lambda: 64 * GiB,
    )
    monkeypatch.setattr(
        "cfie.offload.policy._get_total_system_ram_bytes",
        lambda: 128 * GiB,
    )

    cfg = _make_fake_cfie_config(
        model_dir,
        num_experts=16,
        num_experts_per_tok=8,
        model_type="qwen3_5_mtp",
        architectures=["Qwen3_5MoeMTP"],
    )
    cfg.model_config.hf_text_config.layer_types = ["full_attention"] * 48
    cfg.additional_config = {
        TARGET_OCCUPIED_GPU_BYTES_KEY: 23 * GiB,
    }

    plan = build_moe_tiered_cache_plan(cfg)

    assert plan.enabled is True
    assert plan.reason == "enabled"
    assert plan.shared_gpu_reserve_bytes == 0
    assert plan.gpu_slots_per_layer >= plan.top_k


def test_build_moe_tiered_cache_plan_borrows_minimal_staging_for_mtp_nvme_path(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    metadata = {
        "model.embed_tokens.weight": {"data_offsets": [0, 3 * GiB]},
        "mtp.layers.0.input_layernorm.weight": {
            "data_offsets": [0, 64 * (1 << 20)]
        },
    }
    mtp_proj_bytes = 9 * (1 << 20)
    for expert_idx in range(16):
        base = f"mtp.layers.0.mlp.experts.{expert_idx}"
        metadata[f"{base}.gate_proj.weight"] = {"data_offsets": [0, mtp_proj_bytes]}
        metadata[f"{base}.down_proj.weight"] = {"data_offsets": [0, mtp_proj_bytes]}

    monkeypatch.setattr(
        "cfie.offload.policy.get_safetensors_params_metadata",
        lambda _model_path: metadata,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(total_memory=300 * (1 << 20)),
    )
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda _device: (_ for _ in ()).throw(RuntimeError("unavailable")),
    )
    monkeypatch.setattr(
        "cfie.offload.policy._get_available_system_ram_bytes",
        lambda: 12 * GiB + 40 * (1 << 20),
    )
    monkeypatch.setattr(
        "cfie.offload.policy._get_total_system_ram_bytes",
        lambda: 128 * GiB,
    )

    cfg = _make_fake_cfie_config(
        model_dir,
        num_experts=16,
        num_experts_per_tok=8,
        model_type="qwen3_5_mtp",
        architectures=["Qwen3_5MoeMTP"],
    )
    cfg.model_config.hf_text_config.layer_types = ["full_attention"] * 48

    plan = build_moe_tiered_cache_plan(cfg)

    assert plan.enabled is True
    assert plan.gpu_slots_per_layer == 8
    assert plan.prefill_burst_slots == 0
    assert plan.cpu_slots_per_layer == 16
    assert plan.cpu_static_bytes == plan.expert_bytes_total
    assert plan.cpu_budget_bytes == plan.expert_bytes_total
    assert plan.staging_bytes == 0
    assert plan.nvme_expert_bytes == 0


def test_build_moe_tiered_cache_plan_boosts_cpu_budget_to_eliminate_small_spill(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    metadata = {
        "model.embed_tokens.weight": {"data_offsets": [0, 64 * (1 << 20)]},
        "mtp.layers.0.input_layernorm.weight": {
            "data_offsets": [0, 64 * (1 << 20)]
        },
    }
    mtp_proj_bytes = 256 * (1 << 20)
    for expert_idx in range(16):
        base = f"mtp.layers.0.mlp.experts.{expert_idx}"
        metadata[f"{base}.gate_proj.weight"] = {"data_offsets": [0, mtp_proj_bytes]}
        metadata[f"{base}.down_proj.weight"] = {"data_offsets": [0, mtp_proj_bytes]}

    monkeypatch.setattr(
        "cfie.offload.policy.get_safetensors_params_metadata",
        lambda _model_path: metadata,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(total_memory=5 * GiB),
    )
    monkeypatch.setattr(
        "cfie.offload.policy._get_available_system_ram_bytes",
        lambda: 22 * GiB,
    )
    monkeypatch.setattr(
        "cfie.offload.policy._get_total_system_ram_bytes",
        lambda: 128 * GiB,
    )

    cfg = _make_fake_cfie_config(
        model_dir,
        num_experts=16,
        num_experts_per_tok=8,
        model_type="qwen3_5_mtp",
        architectures=["Qwen3_5MoeMTP"],
    )
    cfg.model_config.hf_text_config.layer_types = ["full_attention"] * 48

    plan = build_moe_tiered_cache_plan(cfg)

    assert plan.enabled is True
    assert plan.gpu_slots_per_layer == 8
    assert plan.cpu_slots_per_layer == 16
    assert plan.cpu_static_bytes == plan.expert_bytes_total
    assert plan.cpu_budget_bytes == plan.expert_bytes_total
    assert plan.staging_bytes == 0
    assert plan.nvme_expert_bytes == 0


def test_spec_decode_base_proposer_uses_isolated_cfie_config_for_mtp(
    monkeypatch,
) -> None:
    proposer = SpecDecodeBaseProposer.__new__(SpecDecodeBaseProposer)
    draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="qwen3_5_mtp")
    )
    original_cfie_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="qwen3_5_moe")
        ),
    )
    proposer.cfie_config = original_cfie_config
    proposer.speculative_config = SimpleNamespace(
        method="mtp",
        draft_model_config=draft_model_config,
        draft_load_config="draft-load-config",
    )
    proposer.method = "mtp"

    isolated_cfie_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="qwen3_5_mtp")
        ),
        load_config="isolated-load-config",
    )
    captured: dict[str, object] = {}

    @contextmanager
    def _noop_model_tag(_tag: str):
        yield

    monkeypatch.setattr(
        "cfie.compilation.backends.set_model_tag",
        _noop_model_tag,
    )
    monkeypatch.setattr(
        "cfie.v1.spec_decode.eagle.create_vllm_config_for_draft_model",
        lambda cfg: isolated_cfie_config,
    )
    monkeypatch.setattr(
        "cfie.offload.policy.get_moe_tiered_cache_plan",
        lambda cfg: {
            "model_type": "qwen3_5_mtp",
            "gpu_slots_per_layer": 8,
            "prefill_burst_slots": 0,
            "cpu_slots_per_layer": 246,
        },
    )
    monkeypatch.setattr(
        "cfie.v1.spec_decode.eagle.get_model",
        lambda **kwargs: captured.update(kwargs) or object(),
    )

    SpecDecodeBaseProposer._get_model(proposer)

    assert captured["cfie_config"] is isolated_cfie_config
    assert captured["model_config"] is isolated_cfie_config.model_config
    assert captured["load_config"] == "isolated-load-config"


def test_spec_decode_base_proposer_keeps_default_config_for_non_mtp(
    monkeypatch,
) -> None:
    proposer = SpecDecodeBaseProposer.__new__(SpecDecodeBaseProposer)
    draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="eagle")
    )
    original_cfie_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="qwen3_5_moe")
        ),
    )
    proposer.cfie_config = original_cfie_config
    proposer.speculative_config = SimpleNamespace(
        method="eagle",
        draft_model_config=draft_model_config,
        draft_load_config="draft-load-config",
    )
    proposer.method = "eagle"

    captured: dict[str, object] = {}

    @contextmanager
    def _noop_model_tag(_tag: str):
        yield

    monkeypatch.setattr(
        "cfie.compilation.backends.set_model_tag",
        _noop_model_tag,
    )
    monkeypatch.setattr(
        "cfie.v1.spec_decode.eagle.create_vllm_config_for_draft_model",
        lambda _cfg: (_ for _ in ()).throw(AssertionError("unexpected mtp isolation")),
    )
    monkeypatch.setattr(
        "cfie.offload.policy.get_moe_tiered_cache_plan",
        lambda cfg: None,
    )
    monkeypatch.setattr(
        "cfie.v1.spec_decode.eagle.get_model",
        lambda **kwargs: captured.update(kwargs) or object(),
    )

    SpecDecodeBaseProposer._get_model(proposer)

    assert captured["cfie_config"] is original_cfie_config
    assert captured["model_config"] is draft_model_config
    assert captured["load_config"] == "draft-load-config"


def test_estimate_cpu_cache_budget_bytes_uses_conservative_defaults(
    tmp_path: Path,
) -> None:
    cfg = _make_fake_cfie_config(tmp_path, cpu_offload_gb=10.0)

    budget = _estimate_cpu_cache_budget_bytes(
        cfg,
        available_system_bytes=80 * GiB,
        total_system_bytes=128 * GiB,
    )

    expected = 80 * GiB - 10 * GiB
    assert budget == expected


def test_estimate_cpu_cache_budget_bytes_honors_hard_cap(
    tmp_path: Path,
) -> None:
    cfg = _make_fake_cfie_config(tmp_path, moe_cpu_budget_gb=8.0)

    budget = _estimate_cpu_cache_budget_bytes(
        cfg,
        available_system_bytes=80 * GiB,
        total_system_bytes=128 * GiB,
    )

    assert budget == 8 * GiB


def test_estimate_cpu_cache_budget_bytes_ignores_min_free_override(
    tmp_path: Path,
) -> None:
    cfg = _make_fake_cfie_config(tmp_path, moe_cpu_min_free_gb=24.0)

    budget = _estimate_cpu_cache_budget_bytes(
        cfg,
        available_system_bytes=80 * GiB,
        total_system_bytes=128 * GiB,
    )

    assert budget == 80 * GiB


def test_estimate_hybrid_cache_bytes_scales_with_max_num_seqs(
    tmp_path: Path,
) -> None:
    single = _make_fake_cfie_config(tmp_path)
    single.scheduler_config.max_num_seqs = 1

    multi = _make_fake_cfie_config(tmp_path)
    multi.scheduler_config.max_num_seqs = 4

    kv_single, linear_single = _estimate_hybrid_cache_bytes(single)
    kv_multi, linear_multi = _estimate_hybrid_cache_bytes(multi)

    assert kv_multi == kv_single * 4
    assert linear_multi == linear_single * 4


def test_estimate_hybrid_cache_bytes_treats_qwen35_mtp_as_full_attention_only(
    tmp_path: Path,
) -> None:
    cfg = _make_fake_cfie_config(
        tmp_path,
        model_type="qwen3_5_mtp",
        architectures=["Qwen3_5MoeMTP"],
    )
    cfg.model_config.hf_text_config.layer_types = ["linear_attention"] * 48
    cfg.model_config.hf_text_config.mtp_num_hidden_layers = 1

    kv_bytes, linear_state_bytes = _estimate_hybrid_cache_bytes(cfg)

    expected_kv_bytes = (
        cfg.scheduler_config.max_num_seqs
        * cfg.model_config.max_model_len
        * 1
        * cfg.model_config.hf_text_config.num_key_value_heads
        * cfg.model_config.hf_text_config.head_dim
        * 2
        * 2
    )

    assert kv_bytes == expected_kv_bytes
    assert linear_state_bytes == 0


def test_pinned_expert_cache_eviction_lru() -> None:
    cache = PinnedExpertCache(capacity_bytes=16)
    tensor_a = torch.ones(4, dtype=torch.int32)
    tensor_b = torch.ones(4, dtype=torch.int32) * 2

    cache.put(("layer", 0), {"0.down_proj.qweight": tensor_a})
    assert cache.contains(("layer", 0))

    cache.put(("layer", 1), {"1.down_proj.qweight": tensor_b})
    assert cache.contains(("layer", 1))
    assert not cache.contains(("layer", 0))


def test_safetensor_expert_store_loads_single_expert(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    shard_name = "model-00001-of-00001.safetensors"
    shard_path = model_dir / shard_name
    save_file(
        {
            "model.layers.0.mlp.experts.0.gate_proj.qweight": torch.arange(
                4, dtype=torch.int32
            ).reshape(2, 2),
            "model.layers.0.mlp.experts.0.down_proj.qzeros": torch.arange(
                2, dtype=torch.float16
            ).reshape(1, 2),
            "model.layers.0.mlp.experts.1.gate_proj.qweight": torch.full(
                (2, 2), 7, dtype=torch.int32
            ),
        },
        str(shard_path),
    )
    with (model_dir / "model.safetensors.index.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "weight_map": {
                    "model.layers.0.mlp.experts.0.gate_proj.qweight": shard_name,
                    "model.layers.0.mlp.experts.0.down_proj.qzeros": shard_name,
                    "model.layers.0.mlp.experts.1.gate_proj.qweight": shard_name,
                }
            },
            f,
        )

    store = SafetensorExpertStore(str(model_dir))
    tensors = store.load_expert("model.layers.0.mlp.experts", 0)

    assert store.has_expert("model.layers.0.mlp.experts", 0)
    assert not store.has_expert("model.layers.0.mlp.experts", 2)
    assert set(tensors) == {"0.gate_proj.qweight", "0.down_proj.qzeros"}
    assert torch.equal(
        tensors["0.gate_proj.qweight"], torch.arange(4, dtype=torch.int32).reshape(2, 2)
    )


def test_safetensor_expert_store_resolves_runtime_layer_aliases(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    shard_name = "model-00001-of-00001.safetensors"
    shard_path = model_dir / shard_name
    save_file(
        {
            "model.language_model.layers.0.mlp.experts.48.gate_proj.qweight": (
                torch.arange(4, dtype=torch.int32).reshape(2, 2)
            ),
        },
        str(shard_path),
    )
    with (model_dir / "model.safetensors.index.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "weight_map": {
                    "model.language_model.layers.0.mlp.experts.48.gate_proj.qweight": (
                        shard_name
                    ),
                }
            },
            f,
        )

    store = SafetensorExpertStore(str(model_dir))

    assert store.has_expert("language_model.model.layers.0.mlp.experts", 48)
    tensors = store.load_expert("language_model.model.layers.0.mlp.experts", 48)
    assert set(tensors) == {"48.gate_proj.qweight"}


def test_safetensor_expert_store_resolves_predictor_language_model_aliases(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    shard_name = "model-00001-of-00001.safetensors"
    shard_path = model_dir / shard_name
    save_file(
        {
            "model.language_model.layers.0.mlp.experts.0.gate_proj.qweight": (
                torch.arange(4, dtype=torch.int32).reshape(2, 2)
            ),
        },
        str(shard_path),
    )
    with (model_dir / "model.safetensors.index.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "weight_map": {
                    "model.language_model.layers.0.mlp.experts.0.gate_proj.qweight": (
                        shard_name
                    ),
                }
            },
            f,
        )

    store = SafetensorExpertStore(str(model_dir))

    assert store.has_expert("model.layers.0.mlp.experts", 0)
    tensors = store.load_expert("model.layers.0.mlp.experts", 0)
    assert set(tensors) == {"0.gate_proj.qweight"}


def test_safetensor_expert_store_copies_into_preallocated_slot(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    shard_name = "model-00001-of-00001.safetensors"
    shard_path = model_dir / shard_name
    save_file(
        {
            "model.layers.0.mlp.experts.0.gate_proj.qweight": torch.arange(
                4, dtype=torch.int32
            ).reshape(2, 2),
            "model.layers.0.mlp.experts.0.down_proj.qzeros": torch.arange(
                2, dtype=torch.float16
            ).reshape(1, 2),
        },
        str(shard_path),
    )
    with (model_dir / "model.safetensors.index.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "weight_map": {
                    "model.layers.0.mlp.experts.0.gate_proj.qweight": shard_name,
                    "model.layers.0.mlp.experts.0.down_proj.qzeros": shard_name,
                }
            },
            f,
        )

    store = SafetensorExpertStore(str(model_dir))
    dst_tensors = {
        "slot.gate_proj.qweight": torch.empty((2, 2), dtype=torch.int32),
        "slot.down_proj.qzeros": torch.empty((1, 2), dtype=torch.float16),
    }

    store.copy_expert_into("model.layers.0.mlp.experts", 0, dst_tensors)

    assert torch.equal(
        dst_tensors["slot.gate_proj.qweight"],
        torch.arange(4, dtype=torch.int32).reshape(2, 2),
    )
    assert torch.equal(
        dst_tensors["slot.down_proj.qzeros"],
        torch.arange(2, dtype=torch.float16).reshape(1, 2),
    )


def test_unquantized_bundle_assembly_uses_moe_config_is_act_and_mul() -> None:
    controller = LayerTieredExpertCacheController.__new__(
        LayerTieredExpertCacheController
    )
    controller.layer = SimpleNamespace(
        moe_config=SimpleNamespace(is_act_and_mul=True),
        intermediate_size_per_partition=4,
        hidden_size=3,
        w13_weight=torch.empty(1, dtype=torch.float16),
        w2_weight=torch.empty(1, dtype=torch.float16),
    )
    controller._use_pinned_cpu = False
    controller._cpu_unquantized_raw_buffer = (
        controller._allocate_unquantized_raw_buffer()
    )

    bundle = SimpleNamespace(
        pinned=False,
        tensors={
            "0.gate_proj.weight": torch.arange(12, dtype=torch.float16).reshape(4, 3),
            "0.up_proj.weight": (torch.arange(12, dtype=torch.float16) + 100).reshape(
                4, 3
            ),
            "0.down_proj.weight": torch.arange(12, dtype=torch.float16).reshape(3, 4),
        },
    )

    raw = controller._assemble_unquantized_weights(bundle)

    assert raw.w13_weight.shape == (1, 8, 3)
    assert torch.equal(raw.w13_weight[0, :4], bundle.tensors["0.gate_proj.weight"])
    assert torch.equal(raw.w13_weight[0, 4:], bundle.tensors["0.up_proj.weight"])
    assert torch.equal(raw.w2_weight[0], bundle.tensors["0.down_proj.weight"])


def test_allocate_source_bundle_keeps_checkpoint_source_pageable_cpu() -> None:
    controller = LayerTieredExpertCacheController.__new__(
        LayerTieredExpertCacheController
    )
    controller._mode = "unquantized"
    controller._use_pinned_cpu = True
    controller.layer = SimpleNamespace(
        intermediate_size_per_partition=4,
        hidden_size=3,
        w13_weight=torch.empty(1, dtype=torch.float16),
        w2_weight=torch.empty(1, dtype=torch.float16),
    )

    bundle = controller._allocate_source_bundle()

    assert bundle.pinned is False
    assert bundle.nbytes == 3 * 4 * 3 * torch.empty((), dtype=torch.float16).element_size()
    assert all(tensor.device.type == "cpu" for tensor in bundle.tensors.values())
    assert all(not tensor.is_pinned() for tensor in bundle.tensors.values())


def test_preprocess_unquantized_static_bundle_is_runtime_ready(monkeypatch) -> None:
    controller = LayerTieredExpertCacheController.__new__(
        LayerTieredExpertCacheController
    )
    controller._mode = "unquantized"
    controller._use_pinned_cpu = True
    controller.layer = SimpleNamespace(
        moe_config=SimpleNamespace(is_act_and_mul=True),
        intermediate_size_per_partition=2,
        hidden_size=3,
        w13_weight=torch.empty(1, dtype=torch.float16),
        w2_weight=torch.empty(1, dtype=torch.float16),
    )
    controller.layer_key = "model.layers.0.mlp.experts"
    controller._cpu_unquantized_raw_buffer = (
        controller._allocate_unquantized_raw_buffer()
    )
    monkeypatch.setattr(
        controller,
        "_to_cpu_static_tensor",
        lambda tensor: tensor.detach().to(device="cpu").contiguous(),
    )

    checkpoint_bundle = ExpertBundle(
        tensors={
            "slot.gate_proj.weight": torch.arange(6, dtype=torch.float16).reshape(2, 3),
            "slot.up_proj.weight": (
                torch.arange(6, dtype=torch.float16) + 10
            ).reshape(2, 3),
            "slot.down_proj.weight": torch.arange(6, dtype=torch.float16).reshape(3, 2),
        },
        nbytes=36,
        pinned=False,
        runtime_ready=False,
    )

    runtime_bundle = controller._preprocess_cpu_static_bundle(checkpoint_bundle)

    assert runtime_bundle.runtime_ready is True
    assert set(runtime_bundle.tensors) == {
        "runtime.w13_weight",
        "runtime.w2_weight",
    }
    assert torch.equal(
        runtime_bundle.tensors["runtime.w13_weight"][:2],
        checkpoint_bundle.tensors["slot.gate_proj.weight"],
    )
    assert torch.equal(
        runtime_bundle.tensors["runtime.w13_weight"][2:],
        checkpoint_bundle.tensors["slot.up_proj.weight"],
    )
    assert torch.equal(
        runtime_bundle.tensors["runtime.w2_weight"],
        checkpoint_bundle.tensors["slot.down_proj.weight"],
    )


def test_preprocess_quantized_static_bundles_batch_is_runtime_ready(
    monkeypatch,
) -> None:
    controller = LayerTieredExpertCacheController.__new__(
        LayerTieredExpertCacheController
    )
    controller._mode = "gptq_marlin"
    controller._gptq_desc_act = False
    controller._use_pinned_cpu = False
    controller._use_pinned_cpu_static = False
    controller.layer_key = "model.layers.0.mlp.experts"
    controller.device = torch.device("cpu")
    controller.pack_factor = 2
    controller.num_bits = 4
    controller.group_size = 2
    controller.is_a_8bit = False
    controller.layer = SimpleNamespace(
        hidden_size=4,
        intermediate_size_per_partition=2,
        num_groups_w13=2,
        num_groups_w2=1,
        w13_scales=torch.empty((1, 2, 4), dtype=torch.float16),
        w2_scales=torch.empty((1, 1, 4), dtype=torch.float16),
        w13_qzeros=torch.empty((1, 2, 2), dtype=torch.int32),
        w2_qzeros=torch.empty((1, 1, 2), dtype=torch.int32),
    )
    controller._cpu_quantized_raw_buffer = controller._allocate_quantized_raw_buffer()
    monkeypatch.setattr(
        controller,
        "_to_cpu_static_tensor",
        lambda tensor: tensor.detach().to(device="cpu").contiguous(),
    )

    repack_calls: list[torch.Tensor] = []
    permute_calls: list[torch.Tensor] = []

    def _fake_repack(
        b_q_weight: torch.Tensor,
        perm: torch.Tensor,
        size_k: int,
        size_n: int,
        num_bits: int,
        is_a_8bit: bool = False,
    ) -> torch.Tensor:
        del perm, size_k, size_n, num_bits, is_a_8bit
        repack_calls.append(b_q_weight.clone())
        return b_q_weight + 1000

    def _fake_permute(
        s: torch.Tensor,
        size_k: int,
        size_n: int,
        group_size: int,
        is_a_8bit: bool = False,
    ) -> torch.Tensor:
        del size_k, size_n, group_size, is_a_8bit
        permute_calls.append(s.clone())
        return s + 2000

    monkeypatch.setattr(weight_offload.ops, "gptq_marlin_moe_repack", _fake_repack)
    monkeypatch.setattr(weight_offload, "marlin_moe_permute_scales", _fake_permute)

    bundle0 = ExpertBundle(
        tensors={
            "slot.gate_proj.qweight": torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
            "slot.gate_proj.scales": torch.tensor(
                [[1, 2], [3, 4]],
                dtype=torch.float16,
            ),
            "slot.gate_proj.qzeros": torch.tensor([[1], [2]], dtype=torch.int32),
            "slot.up_proj.qweight": torch.tensor([[11, 12], [13, 14]], dtype=torch.int32),
            "slot.up_proj.scales": torch.tensor(
                [[11, 12], [13, 14]],
                dtype=torch.float16,
            ),
            "slot.up_proj.qzeros": torch.tensor([[11], [12]], dtype=torch.int32),
            "slot.down_proj.qweight": torch.tensor([[21, 22, 23, 24]], dtype=torch.int32),
            "slot.down_proj.scales": torch.tensor([[21, 22, 23, 24]], dtype=torch.float16),
            "slot.down_proj.qzeros": torch.tensor([[21, 22]], dtype=torch.int32),
        },
        nbytes=0,
        pinned=False,
        runtime_ready=False,
    )
    bundle1 = ExpertBundle(
        tensors={
            "slot.gate_proj.qweight": torch.tensor(
                [[101, 102], [103, 104]],
                dtype=torch.int32,
            ),
            "slot.gate_proj.scales": torch.tensor(
                [[101, 102], [103, 104]],
                dtype=torch.float16,
            ),
            "slot.gate_proj.qzeros": torch.tensor([[101], [102]], dtype=torch.int32),
            "slot.up_proj.qweight": torch.tensor(
                [[111, 112], [113, 114]],
                dtype=torch.int32,
            ),
            "slot.up_proj.scales": torch.tensor(
                [[111, 112], [113, 114]],
                dtype=torch.float16,
            ),
            "slot.up_proj.qzeros": torch.tensor([[111], [112]], dtype=torch.int32),
            "slot.down_proj.qweight": torch.tensor(
                [[121, 122, 123, 124]],
                dtype=torch.int32,
            ),
            "slot.down_proj.scales": torch.tensor(
                [[121, 122, 123, 124]],
                dtype=torch.float16,
            ),
            "slot.down_proj.qzeros": torch.tensor([[121, 122]], dtype=torch.int32),
        },
        nbytes=0,
        pinned=False,
        runtime_ready=False,
    )

    runtime_bundles = controller._preprocess_quantized_static_bundles_batch(
        [bundle0, bundle1]
    )

    assert len(runtime_bundles) == 2
    assert all(bundle.runtime_ready for bundle in runtime_bundles)
    assert repack_calls[0].shape == (2, 2, 4)
    assert repack_calls[1].shape == (2, 1, 4)
    assert permute_calls[0].shape == (2, 2, 4)
    assert permute_calls[1].shape == (2, 1, 4)
    assert torch.equal(
        runtime_bundles[0].tensors["runtime.w13_qweight"],
        torch.tensor(
            [[1001, 1002, 1011, 1012], [1003, 1004, 1013, 1014]],
            dtype=torch.int32,
        ),
    )
    assert torch.equal(
        runtime_bundles[1].tensors["runtime.w13_qweight"],
        torch.tensor(
            [[1101, 1102, 1111, 1112], [1103, 1104, 1113, 1114]],
            dtype=torch.int32,
        ),
    )
    assert torch.equal(
        runtime_bundles[0].tensors["runtime.w2_qweight"],
        torch.tensor([[1021, 1022, 1023, 1024]], dtype=torch.int32),
    )
    assert torch.equal(
        runtime_bundles[1].tensors["runtime.w2_qweight"],
        torch.tensor([[1121, 1122, 1123, 1124]], dtype=torch.int32),
    )
    assert torch.equal(
        runtime_bundles[0].tensors["runtime.w13_scales"],
        torch.tensor(
            [[2001, 2002, 2011, 2012], [2003, 2004, 2013, 2014]],
            dtype=torch.float16,
        ),
    )
    assert torch.equal(
        runtime_bundles[1].tensors["runtime.w2_scales"],
        torch.tensor([[2121, 2122, 2123, 2124]], dtype=torch.float16),
    )
    assert torch.equal(
        runtime_bundles[0].tensors["runtime.w13_qzeros"],
        torch.tensor([[1, 11], [2, 12]], dtype=torch.int32),
    )
    assert torch.equal(
        runtime_bundles[1].tensors["runtime.w2_qzeros"],
        torch.tensor([[121, 122]], dtype=torch.int32),
    )


def test_should_pin_cpu_static_bundles_disables_large_static_mirror() -> None:
    controller = LayerTieredExpertCacheController.__new__(
        LayerTieredExpertCacheController
    )
    controller._use_pinned_cpu = True
    controller._cpu_static_pin_limit_bytes = 8 * (1 << 30)

    assert controller._should_pin_cpu_static_bundles(4 * (1 << 30)) is True
    assert controller._should_pin_cpu_static_bundles(16 * (1 << 30)) is False

    controller._use_pinned_cpu = False
    assert controller._should_pin_cpu_static_bundles(4 * (1 << 30)) is False


def test_to_cpu_static_tensor_respects_static_pin_flag() -> None:
    controller = LayerTieredExpertCacheController.__new__(
        LayerTieredExpertCacheController
    )
    controller._use_pinned_cpu = True
    controller._use_pinned_cpu_static = False

    cpu_tensor = controller._to_cpu_static_tensor(torch.arange(4, dtype=torch.float16))

    assert cpu_tensor.device.type == "cpu"
    assert cpu_tensor.is_contiguous()
    assert not cpu_tensor.is_pinned()


def test_load_experts_into_slots_uses_single_batch_write_entrypoint() -> None:
    controller = LayerTieredExpertCacheController.__new__(
        LayerTieredExpertCacheController
    )
    controller.layer_key = "model.layers.0.mlp.experts"
    controller.layer = SimpleNamespace()
    controller._cpu_hits = 0
    controller._nvme_loads = 0
    controller._total_loads = 0
    controller._evictions = 0
    controller._cpu_static_bundles = {}
    controller._slot_to_global = [-1, -1]
    controller._access_count = [0] * 8
    controller._last_used_step = [0] * 8
    batch_calls = []
    installed = []

    def _get_source_bundle(expert_id: int) -> tuple[ExpertBundle, str]:
        return ExpertBundle(
            tensors={"runtime.w13_weight": torch.tensor([expert_id])},
            nbytes=8,
            pinned=True,
            runtime_ready=True,
        ), "cpu_static"

    def _write_expert_bundles(
        bundles_and_sources: list[tuple[int, int, ExpertBundle, str]],
        target: object,
    ) -> None:
        del target
        batch_calls.append(tuple((expert_id, slot) for expert_id, slot, *_ in bundles_and_sources))

    controller._get_source_bundle = _get_source_bundle
    controller._write_expert_bundles = _write_expert_bundles
    controller._install_mapping = lambda expert_id, slot: installed.append(
        (expert_id, slot)
    )

    controller._load_experts_into_slots([(3, 0), (5, 1)])

    assert batch_calls == [((3, 0), (5, 1))]
    assert installed == [(3, 0), (5, 1)]
    assert controller._cpu_hits == 2
    assert controller._total_loads == 2


def test_write_expert_bundles_uses_unquantized_batch_load_op_when_available(
    monkeypatch,
) -> None:
    controller = LayerTieredExpertCacheController.__new__(
        LayerTieredExpertCacheController
    )
    controller._mode = "unquantized"
    fallback_calls = []
    op_calls: dict[str, object] = {}

    monkeypatch.setattr(
        weight_offload.ops,
        "has_precompiled_moe_batch_load_unquantized_runtime",
        lambda: True,
    )

    def _fake_batch_load(
        slot_ids: torch.Tensor,
        w13_src: torch.Tensor,
        w2_src: torch.Tensor,
        w13_dst: torch.Tensor,
        w2_dst: torch.Tensor,
    ) -> None:
        op_calls["slot_ids"] = slot_ids.clone()
        op_calls["w13_src"] = w13_src.clone()
        op_calls["w2_src"] = w2_src.clone()
        op_calls["w13_dst"] = w13_dst
        op_calls["w2_dst"] = w2_dst

    monkeypatch.setattr(
        weight_offload.ops,
        "moe_batch_load_unquantized_runtime_precompiled",
        _fake_batch_load,
    )
    controller._write_expert_bundle = lambda *_args, **_kwargs: fallback_calls.append(
        True
    )

    bundles_and_sources = [
        (
            3,
            1,
            ExpertBundle(
                tensors={
                    "runtime.w13_weight": torch.tensor([[1.0, 2.0]], dtype=torch.float16),
                    "runtime.w2_weight": torch.tensor([[3.0], [4.0]], dtype=torch.float16),
                },
                nbytes=8,
                pinned=False,
                runtime_ready=True,
            ),
            "cpu_static",
        ),
        (
            5,
            3,
            ExpertBundle(
                tensors={
                    "runtime.w13_weight": torch.tensor([[5.0, 6.0]], dtype=torch.float16),
                    "runtime.w2_weight": torch.tensor([[7.0], [8.0]], dtype=torch.float16),
                },
                nbytes=8,
                pinned=False,
                runtime_ready=True,
            ),
            "cpu_static",
        ),
    ]
    target = SimpleNamespace(
        w13_weight=torch.zeros((4, 1, 2), dtype=torch.float16),
        w2_weight=torch.zeros((4, 2, 1), dtype=torch.float16),
    )

    controller._write_expert_bundles(bundles_and_sources, target)

    assert fallback_calls == []
    assert torch.equal(op_calls["slot_ids"], torch.tensor([1, 3], dtype=torch.int64))
    assert torch.equal(
        op_calls["w13_src"],
        torch.tensor([[[1.0, 2.0]], [[5.0, 6.0]]], dtype=torch.float16),
    )
    assert torch.equal(
        op_calls["w2_src"],
        torch.tensor([[[3.0], [4.0]], [[7.0], [8.0]]], dtype=torch.float16),
    )
    assert op_calls["w13_dst"] is target.w13_weight
    assert op_calls["w2_dst"] is target.w2_weight


def test_write_expert_bundles_uses_gptq_batch_load_op_when_available(
    monkeypatch,
) -> None:
    controller = LayerTieredExpertCacheController.__new__(
        LayerTieredExpertCacheController
    )
    controller._mode = "gptq_marlin"
    fallback_calls = []
    op_calls: dict[str, object] = {}

    monkeypatch.setattr(
        weight_offload.ops,
        "has_precompiled_moe_batch_load_gptq_runtime",
        lambda: True,
    )

    def _fake_batch_load(
        slot_ids: torch.Tensor,
        w13_qweight_src: torch.Tensor,
        w2_qweight_src: torch.Tensor,
        w13_scales_src: torch.Tensor,
        w2_scales_src: torch.Tensor,
        w13_qzeros_src: torch.Tensor,
        w2_qzeros_src: torch.Tensor,
        w13_qweight_dst: torch.Tensor,
        w2_qweight_dst: torch.Tensor,
        w13_scales_dst: torch.Tensor,
        w2_scales_dst: torch.Tensor,
        w13_qzeros_dst: torch.Tensor,
        w2_qzeros_dst: torch.Tensor,
        w13_g_idx_src: torch.Tensor | None,
        w2_g_idx_src: torch.Tensor | None,
        w13_g_idx_sort_indices_src: torch.Tensor | None,
        w2_g_idx_sort_indices_src: torch.Tensor | None,
        w13_g_idx_dst: torch.Tensor | None,
        w2_g_idx_dst: torch.Tensor | None,
        w13_g_idx_sort_indices_dst: torch.Tensor | None,
        w2_g_idx_sort_indices_dst: torch.Tensor | None,
    ) -> None:
        del (
            w13_qweight_dst,
            w2_qweight_dst,
            w13_scales_dst,
            w2_scales_dst,
            w13_qzeros_dst,
            w2_qzeros_dst,
            w13_g_idx_dst,
            w2_g_idx_dst,
            w13_g_idx_sort_indices_dst,
            w2_g_idx_sort_indices_dst,
        )
        op_calls["slot_ids"] = slot_ids.clone()
        op_calls["w13_qweight_src"] = w13_qweight_src.clone()
        op_calls["w2_qweight_src"] = w2_qweight_src.clone()
        op_calls["w13_scales_src"] = w13_scales_src.clone()
        op_calls["w2_scales_src"] = w2_scales_src.clone()
        op_calls["w13_qzeros_src"] = w13_qzeros_src.clone()
        op_calls["w2_qzeros_src"] = w2_qzeros_src.clone()
        op_calls["w13_g_idx_src"] = None if w13_g_idx_src is None else w13_g_idx_src.clone()
        op_calls["w2_g_idx_src"] = None if w2_g_idx_src is None else w2_g_idx_src.clone()
        op_calls["w13_g_idx_sort_indices_src"] = (
            None
            if w13_g_idx_sort_indices_src is None
            else w13_g_idx_sort_indices_src.clone()
        )
        op_calls["w2_g_idx_sort_indices_src"] = (
            None
            if w2_g_idx_sort_indices_src is None
            else w2_g_idx_sort_indices_src.clone()
        )

    monkeypatch.setattr(
        weight_offload.ops,
        "moe_batch_load_gptq_runtime_precompiled",
        _fake_batch_load,
    )
    controller._write_expert_bundle = lambda *_args, **_kwargs: fallback_calls.append(
        True
    )

    bundles_and_sources = [
        (
            1,
            0,
            ExpertBundle(
                tensors={
                    "runtime.w13_qweight": torch.tensor([[1, 2]], dtype=torch.int32),
                    "runtime.w2_qweight": torch.tensor([[3, 4]], dtype=torch.int32),
                    "runtime.w13_scales": torch.tensor([[1.0, 2.0]], dtype=torch.float16),
                    "runtime.w2_scales": torch.tensor([[3.0, 4.0]], dtype=torch.float16),
                    "runtime.w13_qzeros": torch.tensor([[5, 6]], dtype=torch.int32),
                    "runtime.w2_qzeros": torch.tensor([[7, 8]], dtype=torch.int32),
                    "runtime.w13_g_idx": torch.tensor([[9, 10]], dtype=torch.int32),
                    "runtime.w2_g_idx": torch.tensor([[11, 12]], dtype=torch.int32),
                    "runtime.w13_g_idx_sort_indices": torch.tensor(
                        [[13, 14]], dtype=torch.int32
                    ),
                    "runtime.w2_g_idx_sort_indices": torch.tensor(
                        [[15, 16]], dtype=torch.int32
                    ),
                },
                nbytes=40,
                pinned=False,
                runtime_ready=True,
            ),
            "cpu_static",
        ),
        (
            4,
            2,
            ExpertBundle(
                tensors={
                    "runtime.w13_qweight": torch.tensor([[21, 22]], dtype=torch.int32),
                    "runtime.w2_qweight": torch.tensor([[23, 24]], dtype=torch.int32),
                    "runtime.w13_scales": torch.tensor([[5.0, 6.0]], dtype=torch.float16),
                    "runtime.w2_scales": torch.tensor([[7.0, 8.0]], dtype=torch.float16),
                    "runtime.w13_qzeros": torch.tensor([[25, 26]], dtype=torch.int32),
                    "runtime.w2_qzeros": torch.tensor([[27, 28]], dtype=torch.int32),
                    "runtime.w13_g_idx": torch.tensor([[29, 30]], dtype=torch.int32),
                    "runtime.w2_g_idx": torch.tensor([[31, 32]], dtype=torch.int32),
                    "runtime.w13_g_idx_sort_indices": torch.tensor(
                        [[33, 34]], dtype=torch.int32
                    ),
                    "runtime.w2_g_idx_sort_indices": torch.tensor(
                        [[35, 36]], dtype=torch.int32
                    ),
                },
                nbytes=40,
                pinned=False,
                runtime_ready=True,
            ),
            "cpu_static",
        ),
    ]
    target = SimpleNamespace(
        w13_qweight=torch.zeros((4, 1, 2), dtype=torch.int32),
        w2_qweight=torch.zeros((4, 1, 2), dtype=torch.int32),
        w13_scales=torch.zeros((4, 1, 2), dtype=torch.float16),
        w2_scales=torch.zeros((4, 1, 2), dtype=torch.float16),
        w13_qzeros=torch.zeros((4, 1, 2), dtype=torch.int32),
        w2_qzeros=torch.zeros((4, 1, 2), dtype=torch.int32),
        w13_g_idx=torch.zeros((4, 1, 2), dtype=torch.int32),
        w2_g_idx=torch.zeros((4, 1, 2), dtype=torch.int32),
        w13_g_idx_sort_indices=torch.zeros((4, 1, 2), dtype=torch.int32),
        w2_g_idx_sort_indices=torch.zeros((4, 1, 2), dtype=torch.int32),
    )

    controller._write_expert_bundles(bundles_and_sources, target)

    assert fallback_calls == []
    assert torch.equal(op_calls["slot_ids"], torch.tensor([0, 2], dtype=torch.int64))
    assert torch.equal(
        op_calls["w13_qweight_src"],
        torch.tensor([[[1, 2]], [[21, 22]]], dtype=torch.int32),
    )
    assert torch.equal(
        op_calls["w2_qweight_src"],
        torch.tensor([[[3, 4]], [[23, 24]]], dtype=torch.int32),
    )
    assert torch.equal(
        op_calls["w13_g_idx_src"],
        torch.tensor([[[9, 10]], [[29, 30]]], dtype=torch.int32),
    )
    assert torch.equal(
        op_calls["w2_g_idx_sort_indices_src"],
        torch.tensor([[[15, 16]], [[35, 36]]], dtype=torch.int32),
    )


def test_get_source_bundle_materializes_cpu_static_expert_lazily() -> None:
    controller = LayerTieredExpertCacheController.__new__(
        LayerTieredExpertCacheController
    )
    controller.layer_key = "model.layers.0.mlp.experts"
    controller._cpu_static_experts = frozenset({3})
    controller._cpu_static_bundles = {}
    controller._cpu_buffer_bytes = 0
    controller._cpu_stage_bundle = None
    controller.expert_store = SimpleNamespace()
    controller.expert_store.copy_expert_into_calls = []

    def _copy_expert_into(
        layer_key: str,
        expert_id: int,
        dst_tensors: dict[str, torch.Tensor],
        *,
        skip_suffixes: tuple[str, ...] = (),
    ) -> None:
        controller.expert_store.copy_expert_into_calls.append(
            (layer_key, expert_id, skip_suffixes)
        )
        dst_tensors["slot.weight"].fill_(expert_id)

    controller.expert_store.copy_expert_into = _copy_expert_into

    def _allocate_source_bundle() -> ExpertBundle:
        return ExpertBundle(
            tensors={"slot.weight": torch.empty((1,), dtype=torch.float16)},
            nbytes=2,
            pinned=False,
        )

    controller._allocate_source_bundle = _allocate_source_bundle

    bundle, source = controller._get_source_bundle(3)

    assert source == "cpu_static"
    assert torch.equal(bundle.tensors["slot.weight"], torch.tensor([3.0], dtype=torch.float16))
    assert controller._cpu_buffer_bytes == 2
    assert controller.expert_store.copy_expert_into_calls == [
        ("model.layers.0.mlp.experts", 3, ("g_idx",))
    ]

    bundle_again, source_again = controller._get_source_bundle(3)

    assert source_again == "cpu_static"
    assert bundle_again is bundle
    assert controller._cpu_buffer_bytes == 2
    assert len(controller.expert_store.copy_expert_into_calls) == 1


def test_get_source_bundle_requires_full_cpu_static_mirror() -> None:
    controller = LayerTieredExpertCacheController.__new__(
        LayerTieredExpertCacheController
    )
    controller.layer_key = "model.layers.0.mlp.experts"
    controller._cpu_static_experts = frozenset({3})
    controller._cpu_static_bundles = {}
    controller._cpu_buffer_bytes = 0
    controller._cpu_stage_bundle = None
    controller.expert_store = SimpleNamespace(copy_expert_into=lambda *_args, **_kwargs: None)
    controller._allocate_source_bundle = lambda: ExpertBundle(
        tensors={"slot.weight": torch.empty((1,), dtype=torch.float16)},
        nbytes=2,
        pinned=False,
    )

    with pytest.raises(RuntimeError, match="CPU static expert pool"):
        controller._get_source_bundle(4)


def test_init_cpu_fixed_pools_routes_gptq_static_eager_materialization_to_batch() -> None:
    controller = LayerTieredExpertCacheController.__new__(
        LayerTieredExpertCacheController
    )
    controller.layer = SimpleNamespace(global_num_experts=8)
    controller.layer_key = "model.layers.0.mlp.experts"
    controller.plan = {
        "initial_cpu_experts": (3, 4, 99),
        "cpu_static_preprocess_batch_size": 8,
    }
    controller._mode = "gptq_marlin"
    controller._cpu_static_bundles = {}
    controller._cpu_stage_bundle = None
    controller._cpu_static_experts = frozenset()
    controller._cpu_buffer_bytes = 0
    controller._use_pinned_cpu = False
    controller.prefill_burst_pool = None
    controller._prefill_burst_min_tokens = 8

    eager_calls: list[tuple[int, ...]] = []
    controller._allocate_quantized_raw_buffer = lambda batch_size=1: object()
    controller._raw_quantized_nbytes = lambda _raw: 6
    controller._materialize_cpu_static_bundles_eager = (
        lambda expert_ids: eager_calls.append(tuple(expert_ids))
    )

    controller._init_cpu_fixed_pools()

    assert controller._cpu_static_experts == frozenset({3, 4})
    assert controller._cpu_buffer_bytes == 6
    assert eager_calls == [(3, 4)]


def test_resolve_cpu_static_preprocess_batch_size_uses_memory_headroom() -> None:
    controller = LayerTieredExpertCacheController.__new__(
        LayerTieredExpertCacheController
    )
    controller.layer_key = "model.layers.0.mlp.experts"
    controller._cpu_static_preprocess_batch_size_cap = 0
    controller._cpu_static_preprocess_cpu_reserve_bytes = 10
    controller._cpu_static_preprocess_gpu_reserve_bytes = 10
    controller._get_available_cpu_memory_bytes = lambda: 210
    controller._get_available_device_memory_bytes = lambda: 175
    controller._estimate_quantized_cpu_static_preprocess_cpu_bytes = (
        lambda batch_size: 20 + 30 * batch_size
    )
    controller._estimate_quantized_cpu_static_preprocess_gpu_bytes = (
        lambda batch_size: 25 * batch_size
    )

    batch_size = controller._resolve_cpu_static_preprocess_batch_size(8)

    assert batch_size == 6


def test_materialize_quantized_cpu_static_batch_streams_source_bundles() -> None:
    controller = LayerTieredExpertCacheController.__new__(
        LayerTieredExpertCacheController
    )
    controller.layer_key = "model.layers.0.mlp.experts"

    load_calls: list[int] = []
    assemble_calls: list[tuple[int, int]] = []
    register_calls: list[tuple[int, str]] = []

    controller._allocate_quantized_raw_buffer = lambda batch_size=1: {
        "batch_size": batch_size
    }

    def _load_source_expert_bundle(expert_id: int) -> ExpertBundle:
        load_calls.append(expert_id)
        return ExpertBundle(
            tensors={"slot.weight": torch.tensor([expert_id], dtype=torch.int32)},
            nbytes=4,
            pinned=False,
            runtime_ready=False,
        )

    def _assemble_raw_weights(
        bundle: ExpertBundle,
        *,
        raw: object | None = None,
        expert_index: int = 0,
    ) -> object:
        del raw
        assemble_calls.append((expert_index, int(bundle.tensors["slot.weight"].item())))
        return object()

    controller._load_source_expert_bundle = _load_source_expert_bundle
    controller._assemble_raw_weights = _assemble_raw_weights
    controller._preprocess_quantized_raw_batch = lambda raw: {
        "runtime.weight": torch.tensor(
            [[100], [200], [300]],
            dtype=torch.int32,
        )
    }
    controller._build_quantized_runtime_bundle = (
        lambda runtime_tensors, expert_index: ExpertBundle(
            tensors={"runtime.weight": runtime_tensors["runtime.weight"][expert_index]},
            nbytes=4,
            pinned=False,
            runtime_ready=True,
        )
    )
    controller._register_cpu_static_bundle = (
        lambda expert_id, bundle, *, eager: register_calls.append(
            (expert_id, bundle.tensors["runtime.weight"].item())
        )
    )

    controller._materialize_quantized_cpu_static_batch([3, 5, 7])

    assert load_calls == [3, 5, 7]
    assert assemble_calls == [(0, 3), (1, 5), (2, 7)]
    assert register_calls == [(3, 100), (5, 200), (7, 300)]


def test_init_cpu_fixed_pools_materializes_static_experts_eagerly() -> None:
    controller = LayerTieredExpertCacheController.__new__(
        LayerTieredExpertCacheController
    )
    controller.layer = SimpleNamespace(global_num_experts=8)
    controller.layer_key = "model.layers.0.mlp.experts"
    controller.plan = {
        "initial_cpu_experts": (3, 4, 99),
        "staging_bytes": 2,
    }
    controller._mode = "unquantized"
    controller._cpu_static_bundles = {}
    controller._cpu_stage_bundle = None
    controller._cpu_static_experts = frozenset()
    controller._cpu_buffer_bytes = 0
    controller.prefill_burst_pool = None
    controller._prefill_burst_min_tokens = 8
    controller.expert_store = SimpleNamespace()
    controller.expert_store.copy_expert_into_calls = []

    def _copy_expert_into(
        layer_key: str,
        expert_id: int,
        dst_tensors: dict[str, torch.Tensor],
        *,
        skip_suffixes: tuple[str, ...] = (),
    ) -> None:
        controller.expert_store.copy_expert_into_calls.append(
            (layer_key, expert_id, skip_suffixes)
        )
        dst_tensors["slot.weight"].fill_(expert_id)

    controller.expert_store.copy_expert_into = _copy_expert_into

    def _allocate_source_bundle() -> ExpertBundle:
        return ExpertBundle(
            tensors={"slot.weight": torch.empty((1,), dtype=torch.float16)},
            nbytes=2,
            pinned=False,
        )

    controller._allocate_source_bundle = _allocate_source_bundle
    controller._allocate_unquantized_raw_buffer = lambda: object()
    controller._raw_unquantized_nbytes = lambda _raw: 6
    controller._preprocess_cpu_static_bundle = lambda bundle: bundle

    controller._init_cpu_fixed_pools()

    assert controller._cpu_static_experts == frozenset({3, 4})
    assert set(controller._cpu_static_bundles) == {3, 4}
    assert controller._cpu_stage_bundle is None
    assert controller._cpu_buffer_bytes == 10
    assert controller.expert_store.copy_expert_into_calls == [
        ("model.layers.0.mlp.experts", 3, ("g_idx",)),
        ("model.layers.0.mlp.experts", 4, ("g_idx",)),
    ]


def test_apply_with_tiered_cache_prefers_prefill_burst() -> None:
    runner = DefaultMoERunner.__new__(DefaultMoERunner)

    class _FakeQuantMethod:
        def apply(self, **_: object) -> torch.Tensor:
            raise AssertionError("base quant apply should not run in burst mode")

    class _FakeController:
        def __init__(self) -> None:
            self.prepare_calls = 0
            self.run_calls = 0
            self.unique_requested = -1

        def prepare(self, topk_ids: torch.Tensor) -> None:
            self.prepare_calls += 1

        def can_run_prefill_burst(
            self,
            num_unique_experts: int,
            num_tokens: int,
        ) -> bool:
            del num_tokens
            self.unique_requested = num_unique_experts
            return True

        def run_prefill_burst(
            self,
            x: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            shared_experts_input: torch.Tensor | None,
        ) -> torch.Tensor:
            del topk_weights, topk_ids, shared_experts_input
            self.run_calls += 1
            return x + 7

    runner.quant_method = _FakeQuantMethod()
    controller = _FakeController()
    layer = SimpleNamespace(
        local_num_experts=2,
        _cfie_tiered_cache_controller=controller,
    )
    x = torch.zeros((3, 4), dtype=torch.float32)
    topk_weights = torch.ones((3, 2), dtype=torch.float32)
    topk_ids = torch.tensor([[0, 1], [2, 3], [1, 2]], dtype=torch.int64)

    output = runner._apply_with_tiered_cache(
        layer=layer,
        x=x,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        shared_experts_input=None,
    )

    assert controller.prepare_calls == 0
    assert controller.run_calls == 1
    assert controller.unique_requested == 4
    assert torch.equal(output, x + 7)


def test_apply_with_tiered_cache_skips_prefill_burst_for_short_prefill() -> None:
    runner = DefaultMoERunner.__new__(DefaultMoERunner)

    class _FakeQuantMethod:
        def __init__(self) -> None:
            self.apply_calls = 0

        def apply(self, *, x: torch.Tensor, **_: object) -> torch.Tensor:
            self.apply_calls += 1
            return x + 5

    class _FakeController:
        def __init__(self) -> None:
            self.prepare_calls = 0
            self.run_calls = 0
            self.prefill_burst_capacity = 4
            self.prefill_burst_min_tokens = 8
            self.burst_checks: list[tuple[int, int]] = []

        def prepare(self, topk_ids: torch.Tensor) -> None:
            del topk_ids
            self.prepare_calls += 1

        def can_run_prefill_burst(
            self,
            num_unique_experts: int,
            num_tokens: int,
        ) -> bool:
            self.burst_checks.append((num_unique_experts, num_tokens))
            return (
                num_unique_experts <= self.prefill_burst_capacity
                and num_tokens >= 8
            )

        def run_prefill_burst(
            self,
            x: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            shared_experts_input: torch.Tensor | None,
        ) -> torch.Tensor:
            del x, topk_weights, topk_ids, shared_experts_input
            self.run_calls += 1
            raise AssertionError("short prefills should stay off the burst path")

    quant_method = _FakeQuantMethod()
    runner.quant_method = quant_method
    controller = _FakeController()
    layer = SimpleNamespace(
        local_num_experts=2,
        _cfie_tiered_cache_controller=controller,
    )
    x = torch.zeros((3, 4), dtype=torch.float32)
    topk_weights = torch.ones((3, 2), dtype=torch.float32)
    topk_ids = torch.tensor([[0, 1], [2, 3], [0, 2]], dtype=torch.int64)

    output = runner._apply_with_tiered_cache(
        layer=layer,
        x=x,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        shared_experts_input=None,
    )

    assert controller.run_calls == 0
    assert controller.prepare_calls == 3
    assert controller.burst_checks == [(4, 3)]
    assert quant_method.apply_calls == 3
    assert torch.equal(output, x + 5)


def test_apply_with_tiered_cache_packs_prefill_into_burst_sized_chunks() -> None:
    runner = DefaultMoERunner.__new__(DefaultMoERunner)

    class _FakeQuantMethod:
        def apply(self, **_: object) -> torch.Tensor:
            raise AssertionError("packed chunks should stay on burst path")

    class _FakeController:
        def __init__(self) -> None:
            self.prepare_calls = 0
            self.run_calls = 0
            self.prefill_burst_capacity = 3
            self.seen_unique: list[int] = []

        def prepare(self, topk_ids: torch.Tensor) -> None:
            self.prepare_calls += 1

        def can_run_prefill_burst(
            self,
            num_unique_experts: int,
            num_tokens: int,
        ) -> bool:
            del num_tokens
            self.seen_unique.append(num_unique_experts)
            return num_unique_experts <= self.prefill_burst_capacity

        def run_prefill_burst(
            self,
            x: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            shared_experts_input: torch.Tensor | None,
        ) -> torch.Tensor:
            del topk_weights, topk_ids, shared_experts_input
            self.run_calls += 1
            return x + 11

    runner.quant_method = _FakeQuantMethod()
    controller = _FakeController()
    layer = SimpleNamespace(
        local_num_experts=2,
        _cfie_tiered_cache_controller=controller,
    )
    x = torch.zeros((4, 3), dtype=torch.float32)
    topk_weights = torch.ones((4, 2), dtype=torch.float32)
    topk_ids = torch.tensor([[0, 1], [1, 2], [3, 4], [4, 5]], dtype=torch.int64)

    output = runner._apply_with_tiered_cache(
        layer=layer,
        x=x,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        shared_experts_input=None,
    )

    assert controller.prepare_calls == 0
    assert controller.run_calls == 2
    assert controller.seen_unique == [6, 3, 3]
    assert torch.equal(output, x + 11)


def test_select_unquantized_moe_backend_uses_cuda_aten_without_triton(
    monkeypatch,
) -> None:
    moe_config = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(use_batched_activation_format=False),
        moe_backend="auto",
    )

    monkeypatch.setattr(
        "cfie.model_executor.layers.fused_moe.oracle.unquantized."
        "is_supported_config_trtllm_bf16",
        lambda **_: (False, None),
    )
    monkeypatch.setattr(
        "cfie.model_executor.layers.fused_moe.oracle.unquantized.has_flashinfer",
        lambda: False,
    )
    monkeypatch.setattr(
        "cfie.model_executor.layers.fused_moe.oracle.unquantized."
        "has_flashinfer_cutlass_fused_moe",
        lambda: False,
    )
    monkeypatch.setattr(
        "cfie.model_executor.layers.fused_moe.oracle.unquantized."
        "HAS_TRITON",
        False,
    )
    monkeypatch.setattr(
        "cfie.model_executor.layers.fused_moe.oracle.unquantized."
        "_has_cuda_aten_moe_backend",
        lambda: True,
    )
    monkeypatch.setattr(
        "cfie.model_executor.layers.fused_moe.oracle.unquantized."
        "current_platform.is_rocm",
        lambda: False,
    )
    monkeypatch.setattr(
        "cfie.model_executor.layers.fused_moe.oracle.unquantized."
        "current_platform.is_cuda",
        lambda: True,
    )
    monkeypatch.setattr(
        "cfie.model_executor.layers.fused_moe.oracle.unquantized."
        "current_platform.is_xpu",
        lambda: False,
    )
    monkeypatch.setattr(
        "cfie.model_executor.layers.fused_moe.oracle.unquantized."
        "current_platform.is_cpu",
        lambda: False,
    )
    monkeypatch.setattr(
        "cfie.model_executor.layers.fused_moe.oracle.unquantized."
        "current_platform.is_tpu",
        lambda: False,
    )
    monkeypatch.setattr(
        "cfie.model_executor.layers.fused_moe.oracle.unquantized."
        "current_platform.is_out_of_tree",
        lambda: False,
    )
    monkeypatch.setattr(
        "cfie.model_executor.layers.fused_moe.oracle.unquantized."
        "rocm_aiter_ops.is_fused_moe_enabled",
        lambda: False,
    )

    backend = select_unquantized_moe_backend(
        moe_config=moe_config,
        use_ep=False,
        use_dp=False,
    )

    assert backend == UnquantizedMoeBackend.CUDA_ATEN


def test_select_unquantized_moe_backend_falls_back_to_torch_without_triton_or_cuda_aten(
    monkeypatch,
) -> None:
    moe_config = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(use_batched_activation_format=False),
        moe_backend="auto",
    )

    monkeypatch.setattr(
        "cfie.model_executor.layers.fused_moe.oracle.unquantized."
        "is_supported_config_trtllm_bf16",
        lambda **_: (False, None),
    )
    monkeypatch.setattr(
        "cfie.model_executor.layers.fused_moe.oracle.unquantized.has_flashinfer",
        lambda: False,
    )
    monkeypatch.setattr(
        "cfie.model_executor.layers.fused_moe.oracle.unquantized."
        "has_flashinfer_cutlass_fused_moe",
        lambda: False,
    )
    monkeypatch.setattr(
        "cfie.model_executor.layers.fused_moe.oracle.unquantized."
        "HAS_TRITON",
        False,
    )
    monkeypatch.setattr(
        "cfie.model_executor.layers.fused_moe.oracle.unquantized."
        "_has_cuda_aten_moe_backend",
        lambda: False,
    )
    monkeypatch.setattr(
        "cfie.model_executor.layers.fused_moe.oracle.unquantized."
        "current_platform.is_rocm",
        lambda: False,
    )
    monkeypatch.setattr(
        "cfie.model_executor.layers.fused_moe.oracle.unquantized."
        "current_platform.is_cuda",
        lambda: True,
    )
    monkeypatch.setattr(
        "cfie.model_executor.layers.fused_moe.oracle.unquantized."
        "current_platform.is_xpu",
        lambda: False,
    )
    monkeypatch.setattr(
        "cfie.model_executor.layers.fused_moe.oracle.unquantized."
        "current_platform.is_cpu",
        lambda: False,
    )
    monkeypatch.setattr(
        "cfie.model_executor.layers.fused_moe.oracle.unquantized."
        "current_platform.is_tpu",
        lambda: False,
    )
    monkeypatch.setattr(
        "cfie.model_executor.layers.fused_moe.oracle.unquantized."
        "current_platform.is_out_of_tree",
        lambda: False,
    )
    monkeypatch.setattr(
        "cfie.model_executor.layers.fused_moe.oracle.unquantized."
        "rocm_aiter_ops.is_fused_moe_enabled",
        lambda: False,
    )

    backend = select_unquantized_moe_backend(
        moe_config=moe_config,
        use_ep=False,
        use_dp=False,
    )

    assert backend == UnquantizedMoeBackend.TORCH


def test_apply_monolithic_with_tiered_cache_chunks_by_resident_capacity() -> None:
    runner = DefaultMoERunner.__new__(DefaultMoERunner)

    class _FakeQuantMethod:
        def __init__(self) -> None:
            self.apply_calls: list[tuple[int, int]] = []

        def apply_monolithic(
            self,
            *,
            layer: object,
            x: torch.Tensor,
            router_logits: torch.Tensor,
        ) -> torch.Tensor:
            del layer
            self.apply_calls.append((x.shape[0], router_logits.shape[0]))
            return x + 13

    class _FakeController:
        def __init__(self) -> None:
            self.prepare_calls: list[list[int]] = []

        def prepare(self, topk_ids: torch.Tensor) -> None:
            self.prepare_calls.append(topk_ids.view(-1).tolist())

    class _FakeRouter:
        def select_experts(
            self,
            *,
            hidden_states: torch.Tensor,
            router_logits: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del hidden_states, router_logits
            return (
                torch.ones((4, 1), dtype=torch.float32),
                torch.tensor([[0], [1], [2], [3]], dtype=torch.int64),
            )

    runner.quant_method = _FakeQuantMethod()
    runner.router = _FakeRouter()
    layer = SimpleNamespace(
        local_num_experts=2,
        _cfie_tiered_cache_controller=_FakeController(),
    )
    x = torch.zeros((4, 3), dtype=torch.float32)
    router_logits = torch.zeros((4, 4), dtype=torch.float32)

    output = runner._apply_monolithic_with_tiered_cache(
        layer=layer,
        x=x,
        router_logits=router_logits,
    )

    assert layer._cfie_tiered_cache_controller.prepare_calls == [[0, 1], [2, 3]]
    assert runner.quant_method.apply_calls == [(2, 2), (2, 2)]
    assert torch.equal(output, x + 13)
