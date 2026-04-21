"""Unit tests for local weight manifest and real-source parameter bootstrapping."""

from __future__ import annotations

import json
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file
import torch

from cfie_training.config import TrainingProjectConfig
from cfie_training.profiles import build_profile_config
from cfie_training.runtime.planner import LayerBucketPlanner
from cfie_training.runtime.source import LocalWeightManifest
from cfie_training.runtime.store import ParameterShardStore
from cfie_training.runtime.types import ParameterShardSnapshot


def test_local_weight_manifest_matches_qwen_bucket_sources() -> None:
    cfg = build_profile_config("qwen35-35b-a3b")
    manifest = LocalWeightManifest(cfg)

    assert manifest.available is True

    non_routed_shard = ParameterShardSnapshot(
        group_id="bucket_non_routed:0",
        component="bucket_non_routed",
        residency_state="nvme_cold",
        committed_version=0,
        pending_version=None,
        logical_params=1,
        bucket_id=0,
        last_touched_step=-1,
    )
    active_experts_shard = ParameterShardSnapshot(
        group_id="bucket_active_experts:0:0-1-2-3-4-5-6-7",
        component="bucket_active_experts",
        residency_state="nvme_cold",
        committed_version=0,
        pending_version=None,
        logical_params=1,
        bucket_id=0,
        expert_ids=(0, 1, 2, 3, 4, 5, 6, 7),
        last_touched_step=-1,
    )
    non_routed = manifest.source_for_shard(non_routed_shard)
    active_experts = manifest.source_for_shard(active_experts_shard)
    non_routed_plan = manifest.plan_parameter_buffer_sources(
        non_routed_shard,
        representative_params=64,
    )
    active_experts_plan = manifest.plan_parameter_buffer_sources(
        active_experts_shard,
        representative_params=64,
    )

    assert non_routed.matched is True
    assert active_experts.matched is True
    assert any(
        ref.tensor_name == "model.language_model.layers.0.input_layernorm.weight"
        for ref in non_routed.tensor_refs
    )
    assert any(
        ref.tensor_name == "model.language_model.layers.0.mlp.experts.down_proj"
        for ref in active_experts.tensor_refs
    )
    assert any(
        entry.layer_index == 0 and entry.semantic_role == "linear_attn_in_proj_qkv"
        for entry in non_routed_plan.source_layout
    )
    assert all(entry.tensor_shape for entry in non_routed_plan.source_layout)
    assert all(entry.slice_shape for entry in non_routed_plan.source_layout)
    assert any(
        entry.layer_index == 3 and entry.semantic_role == "self_attn_q_proj"
        for entry in non_routed_plan.source_layout
    )
    assert any(
        entry.layer_index == 0 and entry.semantic_role == "mlp_router_gate"
        for entry in non_routed_plan.source_layout
    )
    assert any(
        entry.layer_index == 0 and entry.semantic_role == "shared_expert_up_proj"
        for entry in non_routed_plan.source_layout
    )
    assert {
        entry.semantic_role for entry in active_experts_plan.source_layout
    } == {"expert_down_proj", "expert_gate_up_proj"}


def test_parameter_store_bootstraps_from_real_weight_slices() -> None:
    cfg = build_profile_config("qwen35-35b-a3b")
    store = ParameterShardStore(cfg)
    root = Path(cfg.model_source.model_path)

    non_routed_shard = ParameterShardSnapshot(
        group_id="bucket_non_routed:0",
        component="bucket_non_routed",
        residency_state="nvme_cold",
        committed_version=0,
        pending_version=None,
        logical_params=1024,
        bucket_id=0,
        last_touched_step=-1,
    )
    active_experts_shard = ParameterShardSnapshot(
        group_id="bucket_active_experts:0:0-1-2-3-4-5-6-7",
        component="bucket_active_experts",
        residency_state="nvme_cold",
        committed_version=0,
        pending_version=None,
        logical_params=1024,
        bucket_id=0,
        expert_ids=(0, 1, 2, 3, 4, 5, 6, 7),
        last_touched_step=-1,
    )

    non_routed_view = store.parameter_view(non_routed_shard, step_index=0)
    active_experts_view = store.parameter_view(active_experts_shard, step_index=0)
    summary = store.summary()
    source_summary = store.source_summary((non_routed_shard, active_experts_shard))
    load_summary = store.step_load_summary()
    non_routed_layout = store.source_layout(non_routed_shard)
    active_experts_layout = store.source_layout(active_experts_shard)

    with safe_open(
        root / "model.safetensors-00014-of-00014.safetensors",
        framework="pt",
        device="cpu",
    ) as handle:
        input_layernorm_slice = next(
            entry
            for entry in non_routed_layout
            if entry.layer_index == 0 and entry.semantic_role == "input_layernorm"
        )
        expected_non_routed = handle.get_slice(input_layernorm_slice.tensor_name)[
            : input_layernorm_slice.length
        ].to(dtype=torch.float32)
    with safe_open(
        root / "model.safetensors-00013-of-00014.safetensors",
        framework="pt",
        device="cpu",
    ) as handle:
        full_attention_slice = next(
            entry
            for entry in non_routed_layout
            if entry.layer_index == 3 and entry.semantic_role == "self_attn_q_proj"
        )
        full_attention_rows, full_attention_cols = full_attention_slice.slice_shape
        expected_full_attention = handle.get_slice(full_attention_slice.tensor_name)[
            :full_attention_rows,
            :full_attention_cols,
        ].reshape(-1).to(dtype=torch.float32)
        with safe_open(
            root / "model.safetensors-00012-of-00014.safetensors",
            framework="pt",
            device="cpu",
        ) as handle:
            expert_down_slice = next(
                entry
                for entry in active_experts_layout
                if entry.layer_index == 0 and entry.semantic_role == "expert_down_proj"
            )
            sampled_expert_count, sampled_rows, sampled_cols = expert_down_slice.slice_shape
            expected_experts = (
                torch.cat(
                    [
                        handle.get_slice(expert_down_slice.tensor_name)[
                            expert_id:expert_id + 1,
                            :sampled_rows,
                            :sampled_cols,
                        ].reshape(-1)
                        for expert_id in active_experts_shard.expert_ids[:sampled_expert_count]
                    ],
                    dim=0,
                ).to(dtype=torch.float32)
            )

    assert torch.allclose(
        non_routed_view[
            input_layernorm_slice.start_offset:
            input_layernorm_slice.start_offset + input_layernorm_slice.length
        ],
        expected_non_routed,
        atol=1e-6,
    )
    assert torch.allclose(
        non_routed_view[
            full_attention_slice.start_offset:
            full_attention_slice.start_offset + full_attention_slice.length
        ],
        expected_full_attention,
        atol=1e-6,
    )
    assert torch.allclose(
        active_experts_view[
            expert_down_slice.start_offset:
            expert_down_slice.start_offset + expert_down_slice.length
        ],
        expected_experts,
        atol=1e-6,
    )
    assert summary.tracked_shards == 2
    assert summary.manifest_backed_shards == 2
    assert summary.synthetic_seeded_shards == 0
    assert summary.transport_backed_shards == 0
    assert summary.source_file_count > 0
    assert summary.source_tensor_count > 0
    assert input_layernorm_slice.length > 0
    assert full_attention_slice.length > 0
    assert input_layernorm_slice.tensor_shape == (2048,)
    assert len(full_attention_slice.slice_shape) == 2
    assert len(expert_down_slice.slice_shape) == 3
    assert {
        entry.semantic_role for entry in active_experts_layout
    } == {"expert_down_proj", "expert_gate_up_proj"}
    assert source_summary.touched_shards == 2
    assert source_summary.manifest_backed_shards == 2
    assert source_summary.synthetic_seeded_shards == 0
    assert source_summary.transport_backed_shards == 0
    assert source_summary.file_count > 0
    assert source_summary.tensor_count > 0
    non_routed_source = next(
        record
        for record in source_summary.shard_sources
        if record.group_id == non_routed_shard.group_id
    )
    assert non_routed_source.layer_indices == (0, 1, 2, 3)
    assert "linear_attn_in_proj_qkv" in non_routed_source.semantic_roles
    assert "self_attn_q_proj" in non_routed_source.semantic_roles
    assert "mlp_router_gate" in non_routed_source.semantic_roles
    assert "shared_expert_up_proj" in non_routed_source.semantic_roles
    assert load_summary.touched_shards == 2
    assert load_summary.transport_cache_loads == 0
    assert load_summary.direct_manifest_loads == 2
    assert load_summary.buffer_reuses == 0


def test_local_weight_manifest_loads_router_gate_tensor() -> None:
    cfg = build_profile_config("qwen35-35b-a3b")
    manifest = LocalWeightManifest(cfg)
    root = Path(cfg.model_source.model_path)

    router_ref = manifest.router_gate_ref(0)
    router_gate = manifest.load_router_gate_tensor(0)

    assert router_ref is not None
    assert router_ref.tensor_name == "model.language_model.layers.0.mlp.gate.weight"
    assert router_ref.file_name == "model.safetensors-00014-of-00014.safetensors"
    assert router_gate is not None
    assert tuple(router_gate.shape) == (256, 2048)
    assert router_gate.dtype == torch.float32
    mtp_router_ref = manifest.router_gate_ref(cfg.model_spec.num_hidden_layers)
    mtp_router_gate = manifest.load_router_gate_tensor(cfg.model_spec.num_hidden_layers)
    assert mtp_router_ref is not None
    assert mtp_router_ref.tensor_name == "mtp.layers.0.mlp.gate.weight"
    assert mtp_router_gate is not None
    assert tuple(mtp_router_gate.shape) == (256, 2048)
    assert manifest.router_gate_ref(
        cfg.model_spec.num_hidden_layers + cfg.model_spec.mtp_num_hidden_layers
    ) is None
    assert manifest.load_router_gate_tensor(
        cfg.model_spec.num_hidden_layers + cfg.model_spec.mtp_num_hidden_layers
    ) is None

    with safe_open(
        root / router_ref.file_name,
        framework="pt",
        device="cpu",
    ) as handle:
        expected_router_gate = handle.get_tensor(router_ref.tensor_name).to(
            dtype=torch.float32
        )

    assert torch.allclose(router_gate, expected_router_gate, atol=1e-6)


def test_local_weight_manifest_respects_hybrid_and_mtp_bucket_layout() -> None:
    cfg = build_profile_config("qwen35-35b-a3b")
    cfg.bucket_schedule.unit = "hybrid"
    cfg.bucket_schedule.include_mtp_dedicated_bucket = True
    manifest = LocalWeightManifest(cfg)
    buckets = LayerBucketPlanner(cfg).build()

    first_bucket = buckets[0]
    mtp_bucket = buckets[-1]

    hybrid_non_routed = manifest.source_for_shard(
        ParameterShardSnapshot(
            group_id=f"bucket_non_routed:{first_bucket.bucket_id}",
            component="bucket_non_routed",
            residency_state="nvme_cold",
            committed_version=0,
            pending_version=None,
            logical_params=1,
            bucket_id=first_bucket.bucket_id,
            last_touched_step=-1,
        )
    )
    mtp_non_routed = manifest.source_for_shard(
        ParameterShardSnapshot(
            group_id=f"bucket_non_routed:{mtp_bucket.bucket_id}",
            component="bucket_non_routed",
            residency_state="nvme_cold",
            committed_version=0,
            pending_version=None,
            logical_params=1,
            bucket_id=mtp_bucket.bucket_id,
            last_touched_step=-1,
        )
    )
    mtp_active_experts = manifest.source_for_shard(
        ParameterShardSnapshot(
            group_id=f"bucket_active_experts:{mtp_bucket.bucket_id}:0-1-2-3-4-5-6-7",
            component="bucket_active_experts",
            residency_state="nvme_cold",
            committed_version=0,
            pending_version=None,
            logical_params=1,
            bucket_id=mtp_bucket.bucket_id,
            expert_ids=(0, 1, 2, 3, 4, 5, 6, 7),
            last_touched_step=-1,
        )
    )
    mtp_plan = manifest.plan_parameter_buffer_sources(
        ParameterShardSnapshot(
            group_id=f"bucket_non_routed:{mtp_bucket.bucket_id}",
            component="bucket_non_routed",
            residency_state="nvme_cold",
            committed_version=0,
            pending_version=None,
            logical_params=1,
            bucket_id=mtp_bucket.bucket_id,
            last_touched_step=-1,
        ),
        representative_params=64,
    )

    assert first_bucket.layer_indices == (0, 1)
    assert mtp_bucket.attention_types == ("mtp",)
    assert all(
        ref.tensor_name.startswith(("model.language_model.layers.0.", "model.language_model.layers.1."))
        for ref in hybrid_non_routed.tensor_refs
    )
    assert any(
        ref.tensor_name == "mtp.layers.0.self_attn.q_proj.weight"
        for ref in mtp_non_routed.tensor_refs
    )
    assert any(
        ref.tensor_name.startswith("mtp.layers.0.mlp.experts.")
        and ref.tensor_name.endswith(".down_proj.weight")
        for ref in mtp_active_experts.tensor_refs
    )
    assert any(
        entry.layer_index == cfg.model_spec.num_hidden_layers
        and entry.semantic_role == "self_attn_q_proj"
        for entry in mtp_plan.source_layout
    )


def _pack_rows_to_qweight(values: torch.Tensor) -> torch.Tensor:
    rows, cols = values.shape
    pack_factor = 8
    padded_rows = ((rows + pack_factor - 1) // pack_factor) * pack_factor
    padded = torch.zeros((padded_rows, cols), dtype=torch.int32)
    padded[:rows].copy_(values.to(dtype=torch.int32))
    chunks = padded.view(padded_rows // pack_factor, pack_factor, cols).permute(0, 2, 1)
    shifts = torch.arange(0, 32, 4, dtype=torch.int64).view(1, 1, pack_factor)
    packed = torch.sum(chunks.to(dtype=torch.int64) << shifts, dim=-1)
    return packed.to(dtype=torch.int32)


def _pack_cols_to_qzeros(values: torch.Tensor) -> torch.Tensor:
    groups, cols = values.shape
    pack_factor = 8
    padded_cols = ((cols + pack_factor - 1) // pack_factor) * pack_factor
    padded = torch.zeros((groups, padded_cols), dtype=torch.int32)
    padded[:, :cols].copy_(values.to(dtype=torch.int32))
    chunks = padded.view(groups, padded_cols // pack_factor, pack_factor)
    shifts = torch.arange(0, 32, 4, dtype=torch.int64).view(1, 1, pack_factor)
    packed = torch.sum(chunks.to(dtype=torch.int64) << shifts, dim=-1)
    return packed.to(dtype=torch.int32)


def test_local_weight_manifest_dequantizes_gptq_active_expert_shards(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "gptq_model"
    model_dir.mkdir()
    shard_name = "model-00001-of-00001.safetensors"
    weight_map: dict[str, str] = {}
    tensors: dict[str, torch.Tensor] = {}
    logical_weights: dict[str, torch.Tensor] = {}
    zero_points = torch.full((1, 4), 8, dtype=torch.int32)
    scales = torch.full((1, 4), 0.25, dtype=torch.float16)
    g_idx = torch.zeros(8, dtype=torch.int32)
    for expert_id in range(4):
        for proj_name, role_bias in (
            ("down_proj", 0),
            ("gate_proj", 1),
            ("up_proj", 2),
        ):
            tensor_name = (
                f"model.language_model.layers.0.mlp.experts.{expert_id}.{proj_name}"
            )
            logical = torch.tensor(
                [
                    [8 + role_bias + expert_id + column for column in range(4)]
                    for _ in range(8)
                ],
                dtype=torch.int32,
            )
            logical_weights[tensor_name] = (logical - 8).to(dtype=torch.float32) * 0.25
            tensors[f"{tensor_name}.qweight"] = _pack_rows_to_qweight(logical)
            tensors[f"{tensor_name}.qzeros"] = _pack_cols_to_qzeros(zero_points)
            tensors[f"{tensor_name}.scales"] = scales.clone()
            tensors[f"{tensor_name}.g_idx"] = g_idx.clone()
            for suffix in ("qweight", "qzeros", "scales", "g_idx"):
                weight_map[f"{tensor_name}.{suffix}"] = shard_name
    save_file(tensors, model_dir / shard_name)
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}),
        encoding="utf-8",
    )

    cfg = TrainingProjectConfig.from_dict(
        {
            "profile_name": "synthetic-gptq-source",
            "model_spec": {
                "architecture": "Qwen3_5MoeForConditionalGeneration",
                "text_model_type": "qwen3_5_moe_text",
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "num_experts": 4,
                "num_experts_per_tok": 1,
                "moe_intermediate_size": 4,
                "shared_expert_intermediate_size": 4,
                "full_attention_interval": 1,
                "max_position_embeddings": 128,
                "mtp_num_hidden_layers": 1,
                "attention_pattern": ["full_attention"],
                "quantization": "gptq",
                "quant_bits": 4,
                "quant_group_size": 8,
                "quant_sym": True,
                "total_params_billion": 1.0,
            },
            "model_source": {
                "model_path": str(model_dir),
                "index_filename": "model.safetensors.index.json",
                "use_local_weight_manifest": True,
            },
            "expert_rotation": {
                "active_experts_per_step": 2,
                "rotate_every_steps": 1,
                "prefetch_active_overlap": 0,
            },
            "bucket_schedule": {
                "unit": "layer",
                "max_live_buckets": 1,
                "prefetch_buckets": 0,
                "include_mtp_dedicated_bucket": False,
            },
            "execution": {
                "sample_parallelism": 1,
                "max_tokens_per_micro_batch": 8,
            },
            "transport": {"max_staged_file_cache_gb": 1.0},
            "runtime_quantization": {
                "enabled": True,
                "bits": 4,
                "group_size": 8,
                "sym": True,
                "pack_dtype": "int32",
                "compute_view_dtype": "fp32",
            },
        }
    )
    manifest = LocalWeightManifest(cfg)
    shard = ParameterShardSnapshot(
        group_id="bucket_active_experts:0:2-3",
        component="bucket_active_experts",
        residency_state="nvme_cold",
        committed_version=0,
        pending_version=None,
        logical_params=64,
        bucket_id=0,
        expert_ids=(2, 3),
        last_touched_step=-1,
    )

    source = manifest.source_for_shard(shard)
    assert source.matched is True
    assert {
        ref.tensor_name.split(".mlp.experts.", 1)[1].split(".", 1)[0]
        for ref in source.tensor_refs
    } == {"2", "3"}

    load = manifest.build_parameter_buffer(shard, representative_params=16)
    assert load is not None
    assert {
        entry.semantic_role for entry in load.source_layout
    } == {"expert_down_proj", "expert_gate_up_proj"}
    expert_down_entry = next(
        entry
        for entry in load.source_layout
        if entry.tensor_name.endswith(".experts.2.down_proj.qweight")
    )
    expected = logical_weights[
        "model.language_model.layers.0.mlp.experts.2.down_proj"
    ][: expert_down_entry.slice_shape[0], : expert_down_entry.slice_shape[1]].reshape(-1)
    actual = torch.tensor(
        load.values[
            expert_down_entry.start_offset:
            expert_down_entry.start_offset + expert_down_entry.length
        ],
        dtype=torch.float32,
    )
    assert expert_down_entry.tensor_shape == (8, 4)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_parameter_store_materializes_full_logical_active_expert_shard(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "gptq_model_full"
    model_dir.mkdir()
    shard_name = "model-00001-of-00001.safetensors"
    weight_map: dict[str, str] = {}
    tensors: dict[str, torch.Tensor] = {}
    logical_weights: dict[str, torch.Tensor] = {}
    zero_points = torch.full((1, 4), 8, dtype=torch.int32)
    scales = torch.full((1, 4), 0.25, dtype=torch.float16)
    g_idx = torch.zeros(8, dtype=torch.int32)
    for expert_id in range(4):
        for proj_name, role_bias in (
            ("down_proj", 0),
            ("gate_proj", 1),
            ("up_proj", 2),
        ):
            tensor_name = (
                f"model.language_model.layers.0.mlp.experts.{expert_id}.{proj_name}"
            )
            logical = torch.tensor(
                [
                    [8 + role_bias + expert_id + column for column in range(4)]
                    for _ in range(8)
                ],
                dtype=torch.int32,
            )
            logical_weights[tensor_name] = (logical - 8).to(dtype=torch.float32) * 0.25
            tensors[f"{tensor_name}.qweight"] = _pack_rows_to_qweight(logical)
            tensors[f"{tensor_name}.qzeros"] = _pack_cols_to_qzeros(zero_points)
            tensors[f"{tensor_name}.scales"] = scales.clone()
            tensors[f"{tensor_name}.g_idx"] = g_idx.clone()
            for suffix in ("qweight", "qzeros", "scales", "g_idx"):
                weight_map[f"{tensor_name}.{suffix}"] = shard_name
    save_file(tensors, model_dir / shard_name)
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}),
        encoding="utf-8",
    )

    cfg = TrainingProjectConfig.from_dict(
        {
            "profile_name": "synthetic-gptq-full-store",
            "model_spec": {
                "architecture": "Qwen3_5MoeForConditionalGeneration",
                "text_model_type": "qwen3_5_moe_text",
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "num_experts": 4,
                "num_experts_per_tok": 1,
                "moe_intermediate_size": 4,
                "shared_expert_intermediate_size": 4,
                "full_attention_interval": 1,
                "max_position_embeddings": 128,
                "mtp_num_hidden_layers": 1,
                "attention_pattern": ["full_attention"],
                "quantization": "gptq",
                "quant_bits": 4,
                "quant_group_size": 8,
                "quant_sym": True,
                "total_params_billion": 1.0,
            },
            "model_source": {
                "model_path": str(model_dir),
                "index_filename": "model.safetensors.index.json",
                "use_local_weight_manifest": True,
            },
            "expert_rotation": {
                "active_experts_per_step": 2,
                "rotate_every_steps": 1,
                "prefetch_active_overlap": 0,
            },
            "bucket_schedule": {
                "unit": "layer",
                "max_live_buckets": 1,
                "prefetch_buckets": 0,
                "include_mtp_dedicated_bucket": False,
            },
            "execution": {
                "sample_parallelism": 1,
                "max_tokens_per_micro_batch": 8,
                "trainable_shard_materialization": "logical",
            },
            "transport": {"max_staged_file_cache_gb": 1.0},
            "runtime_quantization": {
                "enabled": True,
                "bits": 4,
                "group_size": 8,
                "sym": True,
                "pack_dtype": "int32",
                "compute_view_dtype": "fp32",
            },
        }
    )
    manifest = LocalWeightManifest(cfg)
    store = ParameterShardStore(cfg)
    shard = ParameterShardSnapshot(
        group_id="bucket_active_experts:0:2-3",
        component="bucket_active_experts",
        residency_state="nvme_cold",
        committed_version=0,
        pending_version=None,
        logical_params=192,
        bucket_id=0,
        expert_ids=(2, 3),
        last_touched_step=-1,
    )

    full_load = manifest.build_full_parameter_buffer(shard)
    view = store.parameter_view(shard, step_index=0)
    layout = store.source_layout(shard)
    summary = store.summary()

    assert full_load is not None
    assert isinstance(full_load.values, torch.Tensor)
    assert full_load.values.numel() == 192
    assert view.numel() == 192
    assert sum(entry.length for entry in layout) == 192
    assert summary.tracked_shards == 1
    assert summary.cpu_hot_shards == 1
    expert_down_entry = next(
        entry
        for entry in layout
        if entry.tensor_name.endswith(".experts.2.down_proj.qweight")
    )
    expected = logical_weights[
        "model.language_model.layers.0.mlp.experts.2.down_proj"
    ].reshape(-1)
    full_load_actual = full_load.values[
        expert_down_entry.start_offset:
        expert_down_entry.start_offset + expert_down_entry.length
    ]
    actual = view[
        expert_down_entry.start_offset:
        expert_down_entry.start_offset + expert_down_entry.length
    ]
    assert expert_down_entry.length == 32
    assert torch.allclose(full_load_actual, expected, atol=1e-6)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_parameter_store_prefetch_is_lazy_for_full_logical_manifest_shards(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "gptq_model_lazy_prefetch"
    model_dir.mkdir()
    shard_name = "model-00001-of-00001.safetensors"
    weight_map: dict[str, str] = {}
    tensors: dict[str, torch.Tensor] = {}
    zero_points = torch.full((1, 4), 8, dtype=torch.int32)
    scales = torch.full((1, 4), 0.25, dtype=torch.float16)
    g_idx = torch.zeros(8, dtype=torch.int32)
    for expert_id in range(4):
        for proj_name, role_bias in (
            ("down_proj", 0),
            ("gate_proj", 1),
            ("up_proj", 2),
        ):
            tensor_name = (
                f"model.language_model.layers.0.mlp.experts.{expert_id}.{proj_name}"
            )
            logical = torch.tensor(
                [
                    [8 + role_bias + expert_id + column for column in range(4)]
                    for _ in range(8)
                ],
                dtype=torch.int32,
            )
            tensors[f"{tensor_name}.qweight"] = _pack_rows_to_qweight(logical)
            tensors[f"{tensor_name}.qzeros"] = _pack_cols_to_qzeros(zero_points)
            tensors[f"{tensor_name}.scales"] = scales.clone()
            tensors[f"{tensor_name}.g_idx"] = g_idx.clone()
            for suffix in ("qweight", "qzeros", "scales", "g_idx"):
                weight_map[f"{tensor_name}.{suffix}"] = shard_name
    save_file(tensors, model_dir / shard_name)
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}),
        encoding="utf-8",
    )

    cfg = TrainingProjectConfig.from_dict(
        {
            "profile_name": "synthetic-gptq-lazy-prefetch",
            "model_spec": {
                "architecture": "Qwen3_5MoeForConditionalGeneration",
                "text_model_type": "qwen3_5_moe_text",
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "num_experts": 4,
                "num_experts_per_tok": 1,
                "moe_intermediate_size": 4,
                "shared_expert_intermediate_size": 4,
                "full_attention_interval": 1,
                "max_position_embeddings": 128,
                "mtp_num_hidden_layers": 1,
                "attention_pattern": ["full_attention"],
                "quantization": "gptq",
                "quant_bits": 4,
                "quant_group_size": 8,
                "quant_sym": True,
                "total_params_billion": 1.0,
            },
            "model_source": {
                "model_path": str(model_dir),
                "index_filename": "model.safetensors.index.json",
                "use_local_weight_manifest": True,
            },
            "expert_rotation": {
                "active_experts_per_step": 2,
                "rotate_every_steps": 1,
                "prefetch_active_overlap": 0,
            },
            "bucket_schedule": {
                "unit": "layer",
                "max_live_buckets": 1,
                "prefetch_buckets": 0,
                "include_mtp_dedicated_bucket": False,
            },
            "execution": {
                "sample_parallelism": 1,
                "max_tokens_per_micro_batch": 8,
                "trainable_shard_materialization": "logical",
            },
            "transport": {"max_staged_file_cache_gb": 1.0},
            "runtime_quantization": {
                "enabled": True,
                "bits": 4,
                "group_size": 8,
                "sym": True,
                "pack_dtype": "int32",
                "compute_view_dtype": "fp32",
                "persist_fp32_to_nvme": True,
                "nvme_staging_dir": str(tmp_path / "nvme"),
            },
        }
    )
    store = ParameterShardStore(cfg)
    shard = ParameterShardSnapshot(
        group_id="bucket_active_experts:0:2-3",
        component="bucket_active_experts",
        residency_state="nvme_cold",
        committed_version=0,
        pending_version=None,
        logical_params=192,
        bucket_id=0,
        expert_ids=(2, 3),
        last_touched_step=-1,
    )

    store.prefetch_shards(step_index=0, parameter_shards=(shard,))

    state = store._states[shard.group_id]

    assert state.parameter_tensor is None
    assert state.parameter_buffer_tensor is None
    assert state.parameter_buffer == ()
    assert state.resident_tier == "nvme_cold"
    assert state.source_kind == "local_manifest"
