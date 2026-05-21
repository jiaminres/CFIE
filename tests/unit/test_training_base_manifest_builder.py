"""Unit tests for training-base manifest construction."""

from __future__ import annotations

import pytest

from cfie_training.training_base import (
    CpuAdamFp8StateStore,
    FP32ShardStore,
    GptqCacheStore,
    ManifestShardConfig,
    Qwen35MoeManifestConfig,
    TrainingBaseManifestBuilder,
    TrainingParamManifestSpec,
    adam_state_num_bytes,
    gptq_bundle_num_bytes,
    gptq_layout_hash,
    make_qwen35_moe_manifest_specs,
    state_key,
)


def test_manifest_builder_lays_out_fp32_adam_and_gptq_records() -> None:
    builder = TrainingBaseManifestBuilder(
        ManifestShardConfig(
            fp32_shard_bytes=32,
            adam_shard_bytes=16,
            gptq_shard_bytes=8,
            adam_block_size=4,
            gptq_group_size=4,
        )
    )
    specs = (
        TrainingParamManifestSpec(
            "param_a",
            num_elements=4,
            trainable=True,
            gptq_bundle_id="bundle_a",
        ),
        TrainingParamManifestSpec(
            "param_b",
            num_elements=6,
            trainable=False,
            gptq_bundle_id=None,
        ),
        TrainingParamManifestSpec(
            "param_c",
            num_elements=4,
            trainable=True,
            gptq_bundle_id="bundle_c",
        ),
    )

    manifest = builder.build(specs)

    assert manifest.fp32_records["param_a"].shard_name == "fp32_0000.bin"
    assert manifest.fp32_records["param_a"].offset_elements == 0
    assert manifest.fp32_records["param_b"].shard_name == "fp32_0001.bin"
    assert manifest.fp32_records["param_b"].offset_elements == 0
    assert manifest.fp32_records["param_c"].shard_name == "fp32_0002.bin"
    assert manifest.fp32_records["param_c"].offset_elements == 0

    state_bytes = adam_state_num_bytes(4, block_size=4)
    assert set(manifest.adam_records) == {
        state_key("param_a", "m"),
        state_key("param_a", "v"),
        state_key("param_c", "m"),
        state_key("param_c", "v"),
    }
    assert manifest.adam_records[state_key("param_a", "m")].num_bytes == state_bytes
    assert manifest.adam_records[state_key("param_a", "m")].shard_name == (
        "adam_0000.bin"
    )
    assert manifest.adam_records[state_key("param_a", "v")].shard_name == (
        "adam_0000.bin"
    )
    assert manifest.adam_records[state_key("param_c", "m")].shard_name == (
        "adam_0001.bin"
    )

    assert manifest.param_to_gptq_bundle == {
        "param_a": "bundle_a",
        "param_c": "bundle_c",
    }
    assert manifest.gptq_records["bundle_a"].num_bytes == gptq_bundle_num_bytes(
        4,
        group_size=4,
    )
    assert manifest.gptq_records["bundle_a"].quant_layout_hash == gptq_layout_hash(
        group_size=4,
    )


def test_manifest_builder_can_create_empty_store_files(tmp_path) -> None:
    manifest = TrainingBaseManifestBuilder().build(
        (
            TrainingParamManifestSpec("param_a", num_elements=2),
        )
    )

    fp32_store, adam_store, gptq_store = manifest.create_stores(
        tmp_path,
        generation=3,
    )

    assert isinstance(fp32_store, FP32ShardStore)
    assert isinstance(adam_store, CpuAdamFp8StateStore)
    assert isinstance(gptq_store, GptqCacheStore)
    assert FP32ShardStore.load(tmp_path / "fp32").generation == 3
    assert CpuAdamFp8StateStore.load(tmp_path / "adam").generation == 3
    assert GptqCacheStore.load(tmp_path / "gptq").generation == 3


def test_manifest_builder_rejects_duplicate_param_ids() -> None:
    builder = TrainingBaseManifestBuilder()

    with pytest.raises(ValueError, match="duplicate param_id"):
        builder.build(
            (
                TrainingParamManifestSpec("param_a", num_elements=1),
                TrainingParamManifestSpec("param_a", num_elements=1),
            )
        )


def test_manifest_builder_rejects_duplicate_gptq_bundle_ids() -> None:
    builder = TrainingBaseManifestBuilder()

    with pytest.raises(ValueError, match="duplicate GPTQ bundle"):
        builder.build(
            (
                TrainingParamManifestSpec(
                    "param_a",
                    num_elements=1,
                    gptq_bundle_id="bundle",
                ),
                TrainingParamManifestSpec(
                    "param_b",
                    num_elements=1,
                    gptq_bundle_id="bundle",
                ),
            )
        )


def test_manifest_builder_uses_explicit_gptq_bundle_bytes() -> None:
    manifest = TrainingBaseManifestBuilder(
        ManifestShardConfig(gptq_shard_bytes=64)
    ).build(
        (
            TrainingParamManifestSpec(
                "param_a",
                num_elements=4,
                gptq_bundle_id="bundle_a",
                gptq_num_bytes=33,
                quant_layout_hash="sha256:bundle",
            ),
        )
    )

    record = manifest.gptq_records["bundle_a"]

    assert record.num_bytes == 33
    assert record.quant_layout_hash == "sha256:bundle"


def test_qwen35_moe_manifest_specs_generate_local_expert_weights() -> None:
    specs = make_qwen35_moe_manifest_specs(
        Qwen35MoeManifestConfig(
            num_layers=2,
            layer_start=10,
            num_experts=4,
            local_expert_ids=(1, 3),
            hidden_size=8,
            intermediate_size=16,
            tp_size=2,
        )
    )

    assert len(specs) == 8
    assert specs[0].param_id == "layers.10.experts.1.w13_weight"
    assert specs[0].num_elements == 2 * (16 // 2) * 8
    assert specs[1].param_id == "layers.10.experts.1.w2_weight"
    assert specs[1].num_elements == 8 * (16 // 2)
    assert specs[-1].param_id == "layers.11.experts.3.w2_weight"
    assert specs[-1].gptq_bundle_id == specs[-1].param_id


def test_qwen35_moe_manifest_specs_can_disable_gptq_cache() -> None:
    specs = make_qwen35_moe_manifest_specs(
        Qwen35MoeManifestConfig(
            num_layers=1,
            num_experts=1,
            hidden_size=8,
            intermediate_size=16,
            include_gptq_cache=False,
        )
    )

    assert {spec.gptq_bundle_id for spec in specs} == {None}


def test_qwen35_moe_manifest_config_requires_tp_divisibility() -> None:
    with pytest.raises(ValueError, match="intermediate_size"):
        Qwen35MoeManifestConfig(
            num_layers=1,
            num_experts=1,
            hidden_size=8,
            intermediate_size=10,
            tp_size=3,
        )
