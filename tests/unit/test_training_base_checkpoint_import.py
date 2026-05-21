"""Unit tests for Qwen3.5 MoE checkpoint import helpers."""

from __future__ import annotations

import struct

import pytest
import torch

from cfie_training.training_base import (
    ManifestShardConfig,
    Qwen35MoeCheckpointImportConfig,
    Qwen35MoeCheckpointImporter,
    decode_gptq_marlin_bundle,
    import_qwen35_moe_checkpoint,
    import_qwen35_moe_checkpoint_to_fp32_store,
    pack_gptq_int4_qweight,
    pack_gptq_int4_qzeros,
    qwen35_moe_checkpoint_key_filter,
)


def _fp32_values(payload: bytes) -> tuple[float, ...]:
    return struct.unpack(f"<{len(payload) // 4}f", payload)


def _gptq_weights() -> dict[str, torch.Tensor]:
    gate_q = torch.tensor([[8, 9], [10, 7]], dtype=torch.int32)
    up_q = torch.tensor([[9, 8], [7, 10]], dtype=torch.int32)
    down_q = torch.tensor([[10, 8], [8, 6]], dtype=torch.int32)
    qzeros = torch.full((1, 2), 7, dtype=torch.int32)
    return {
        "layers.0.mlp.experts.0.gate_proj.qweight": pack_gptq_int4_qweight(
            gate_q
        ),
        "layers.0.mlp.experts.0.gate_proj.scales": torch.tensor([[1.0, 2.0]]),
        "layers.0.mlp.experts.0.gate_proj.qzeros": pack_gptq_int4_qzeros(qzeros),
        "layers.0.mlp.experts.0.gate_proj.g_idx": torch.tensor([0, 0]),
        "layers.0.mlp.experts.0.up_proj.qweight": pack_gptq_int4_qweight(up_q),
        "layers.0.mlp.experts.0.up_proj.scales": torch.tensor([[1.0, 2.0]]),
        "layers.0.mlp.experts.0.up_proj.qzeros": pack_gptq_int4_qzeros(qzeros),
        "layers.0.mlp.experts.0.up_proj.g_idx": torch.tensor([0, 0]),
        "layers.0.mlp.experts.0.down_proj.qweight": pack_gptq_int4_qweight(
            down_q
        ),
        "layers.0.mlp.experts.0.down_proj.scales": torch.tensor([[1.0, 2.0]]),
        "layers.0.mlp.experts.0.down_proj.qzeros": pack_gptq_int4_qzeros(qzeros),
        "layers.0.mlp.experts.0.down_proj.g_idx": torch.tensor([0, 0]),
    }


def test_qwen35_moe_import_fuses_gate_and_up_to_w13() -> None:
    weights = {
        "model.layers.10.mlp.experts.3.gate_proj.weight": torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=torch.bfloat16,
        ),
        "model.layers.10.mlp.experts.3.up_proj.weight": torch.tensor(
            [[5.0, 6.0], [7.0, 8.0]],
            dtype=torch.float16,
        ),
        "model.layers.10.mlp.experts.3.down_proj.weight": torch.tensor(
            [[9.0, 10.0], [11.0, 12.0]],
            dtype=torch.float32,
        ),
        "model.layers.10.self_attn.q_proj.weight": torch.ones(1),
    }

    plan = import_qwen35_moe_checkpoint(weights)

    assert plan.skipped_keys == ("model.layers.10.self_attn.q_proj.weight",)
    assert [param.param_id for param in plan.imported_params] == [
        "layers.10.experts.3.w13_weight",
        "layers.10.experts.3.w2_weight",
    ]
    assert plan.param_to_source_keys["layers.10.experts.3.w13_weight"] == (
        "model.layers.10.mlp.experts.3.gate_proj.weight",
        "model.layers.10.mlp.experts.3.up_proj.weight",
    )
    assert torch.equal(
        plan.fp32_updates["layers.10.experts.3.w13_weight"],
        torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    )
    assert torch.equal(
        plan.fp32_updates["layers.10.experts.3.w2_weight"],
        torch.tensor([9.0, 10.0, 11.0, 12.0]),
    )
    assert plan.specs[0].gptq_bundle_id == "layers.10.experts.3.w13_weight"


def test_qwen35_moe_import_can_filter_layers_and_experts() -> None:
    weights = [
        (
            "layers.9.mlp.experts.1.down_proj.weight",
            torch.tensor([1.0]),
        ),
        (
            "layers.10.mlp.experts.1.down_proj.weight",
            torch.tensor([2.0]),
        ),
        (
            "layers.10.mlp.experts.2.down_proj.weight",
            torch.tensor([3.0]),
        ),
    ]
    config = Qwen35MoeCheckpointImportConfig(
        layer_start=10,
        layer_end_exclusive=11,
        local_expert_ids=(2,),
        include_gptq_cache=False,
    )

    plan = import_qwen35_moe_checkpoint(weights, config=config)

    assert [param.param_id for param in plan.imported_params] == [
        "layers.10.experts.2.w2_weight",
    ]
    assert plan.imported_params[0].gptq_bundle_id is None
    assert plan.skipped_keys == (
        "layers.9.mlp.experts.1.down_proj.weight",
        "layers.10.mlp.experts.1.down_proj.weight",
    )


def test_qwen35_moe_checkpoint_key_filter_matches_selected_experts() -> None:
    key_filter = qwen35_moe_checkpoint_key_filter(
        Qwen35MoeCheckpointImportConfig(
            layer_start=10,
            layer_end_exclusive=11,
            local_expert_ids=(2,),
        )
    )

    assert key_filter("model.layers.10.mlp.experts.2.gate_proj.weight")
    assert key_filter("layers.10.mlp.experts.2.down_proj.qweight")
    assert not key_filter("layers.10.mlp.experts.1.down_proj.weight")
    assert not key_filter("layers.9.mlp.experts.2.down_proj.weight")
    assert not key_filter("layers.10.self_attn.q_proj.weight")


def test_qwen35_moe_streaming_import_returns_completed_w13_pair() -> None:
    importer = Qwen35MoeCheckpointImporter()

    assert importer.consume(
        "layers.0.mlp.experts.0.gate_proj.weight",
        torch.tensor([1.0, 2.0]),
    ) == ()
    completed = importer.consume(
        "layers.0.mlp.experts.0.up_proj.weight",
        torch.tensor([3.0, 4.0]),
    )

    assert len(completed) == 1
    assert completed[0].param_id == "layers.0.experts.0.w13_weight"
    assert torch.equal(completed[0].tensor, torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert importer.finalize().imported_params[0] == completed[0]


def test_qwen35_moe_import_rejects_incomplete_w13_pair() -> None:
    importer = Qwen35MoeCheckpointImporter()
    importer.consume(
        "layers.0.mlp.experts.0.gate_proj.weight",
        torch.tensor([1.0, 2.0]),
    )

    with pytest.raises(ValueError, match="incomplete w13"):
        importer.finalize()


def test_qwen35_moe_import_rejects_gate_up_shape_mismatch() -> None:
    importer = Qwen35MoeCheckpointImporter()
    importer.consume(
        "layers.0.mlp.experts.0.gate_proj.weight",
        torch.ones(2, 2),
    )

    with pytest.raises(ValueError, match="identical shapes"):
        importer.consume(
            "layers.0.mlp.experts.0.up_proj.weight",
            torch.ones(4),
        )


def test_qwen35_moe_import_to_fp32_store_initializes_master_shards(tmp_path) -> None:
    result = import_qwen35_moe_checkpoint_to_fp32_store(
        {
            "layers.0.mlp.experts.0.gate_proj.weight": torch.tensor([1.0, 2.0]),
            "layers.0.mlp.experts.0.up_proj.weight": torch.tensor([3.0, 4.0]),
            "layers.0.mlp.experts.0.down_proj.weight": torch.tensor([5.0, 6.0]),
        },
        root=tmp_path,
        manifest_config=ManifestShardConfig(
            fp32_shard_bytes=16,
            adam_shard_bytes=64,
            gptq_shard_bytes=64,
            gptq_group_size=2,
        ),
        generation=7,
    )

    assert result.fp32_store.generation == 7
    assert result.adam_store.generation == 7
    assert result.gptq_store.generation == 7
    assert result.manifest.total_fp32_bytes == 24
    assert _fp32_values(
        result.fp32_store.read_param("layers.0.experts.0.w13_weight")
    ) == pytest.approx((1.0, 2.0, 3.0, 4.0))
    assert _fp32_values(
        result.fp32_store.read_param("layers.0.experts.0.w2_weight")
    ) == pytest.approx((5.0, 6.0))


def test_qwen35_moe_import_dequantizes_gptq_components() -> None:
    config = Qwen35MoeCheckpointImportConfig(gptq_group_size=8)

    plan = import_qwen35_moe_checkpoint(_gptq_weights(), config=config)

    assert [param.param_id for param in plan.imported_params] == [
        "layers.0.experts.0.w13_weight",
        "layers.0.experts.0.w2_weight",
    ]
    assert torch.equal(
        plan.fp32_updates["layers.0.experts.0.w13_weight"],
        torch.tensor([0.0, 2.0, 2.0, -2.0, 1.0, -1.0, 0.0, 4.0]),
    )
    assert torch.equal(
        plan.fp32_updates["layers.0.experts.0.w2_weight"],
        torch.tensor([2.0, 0.0, 0.0, -4.0]),
    )
    assert set(plan.gptq_cache_updates) == {
        "layers.0.experts.0.w13_weight",
        "layers.0.experts.0.w2_weight",
    }
    w13_bundle = decode_gptq_marlin_bundle(
        plan.gptq_cache_updates["layers.0.experts.0.w13_weight"]
    )
    assert set(w13_bundle.tensors) == {
        "w1.g_idx",
        "w1.qweight",
        "w1.qzeros",
        "w1.scales",
        "w3.g_idx",
        "w3.qweight",
        "w3.qzeros",
        "w3.scales",
    }
    specs = {spec.param_id: spec for spec in plan.specs}
    assert specs["layers.0.experts.0.w13_weight"].gptq_num_bytes == len(
        plan.gptq_cache_updates["layers.0.experts.0.w13_weight"]
    )
    assert specs["layers.0.experts.0.w13_weight"].quant_layout_hash.startswith(
        "sha256:"
    )


def test_qwen35_moe_import_to_store_writes_raw_gptq_cache(tmp_path) -> None:
    result = import_qwen35_moe_checkpoint_to_fp32_store(
        _gptq_weights(),
        root=tmp_path,
        import_config=Qwen35MoeCheckpointImportConfig(gptq_group_size=8),
        manifest_config=ManifestShardConfig(
            fp32_shard_bytes=64,
            adam_shard_bytes=128,
            gptq_shard_bytes=4096,
            gptq_group_size=8,
        ),
        generation=3,
    )

    payload = result.gptq_store.read_bundle("layers.0.experts.0.w2_weight")
    decoded = decode_gptq_marlin_bundle(payload)

    assert result.gptq_store.generation == 3
    assert decoded.metadata.bundle_id == "layers.0.experts.0.w2_weight"
    assert set(decoded.tensors) == {"g_idx", "qweight", "qzeros", "scales"}
    assert result.manifest.gptq_records[
        "layers.0.experts.0.w2_weight"
    ].num_bytes == len(payload)


def test_qwen35_moe_gptq_import_rejects_missing_scales() -> None:
    importer = Qwen35MoeCheckpointImporter()
    importer.consume(
        "layers.0.mlp.experts.0.down_proj.qweight",
        pack_gptq_int4_qweight(torch.full((8, 1), 8, dtype=torch.int32)),
    )

    with pytest.raises(ValueError, match="missing GPTQ scales"):
        importer.finalize()
