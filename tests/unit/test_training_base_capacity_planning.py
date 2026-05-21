"""Unit tests for training-base capacity planning."""

from __future__ import annotations

import pytest

from cfie_training.training_base import (
    ManifestShardConfig,
    Qwen35MoeManifestConfig,
    TrainingBaseManifestBuilder,
    bytes_to_gib,
    capacity_report_from_manifest,
    estimate_qwen35_moe_capacity,
    make_qwen35_moe_manifest_specs,
)
from cfie_training.training_base.adam_update import adam_state_num_bytes
from cfie_training.training_base.gptq_requant import gptq_bundle_num_bytes


def test_capacity_report_from_qwen35_moe_config() -> None:
    qwen_config = Qwen35MoeManifestConfig(
        num_layers=2,
        num_experts=3,
        hidden_size=4,
        intermediate_size=8,
        tp_size=2,
    )
    shard_config = ManifestShardConfig(
        fp32_shard_bytes=256,
        adam_shard_bytes=256,
        gptq_shard_bytes=256,
        adam_block_size=16,
        gptq_group_size=8,
    )

    report = estimate_qwen35_moe_capacity(qwen_config, shard_config)

    w13_elements = 2 * (8 // 2) * 4
    w2_elements = 4 * (8 // 2)
    per_expert_elements = w13_elements + w2_elements
    expert_count = 2 * 3
    expected_fp32 = expert_count * per_expert_elements * 4
    expected_adam = expert_count * 2 * (
        adam_state_num_bytes(w13_elements, block_size=16)
        + adam_state_num_bytes(w2_elements, block_size=16)
    )
    expected_gptq = expert_count * (
        gptq_bundle_num_bytes(w13_elements, group_size=8)
        + gptq_bundle_num_bytes(w2_elements, group_size=8)
    )

    assert report.param_count == 12
    assert report.trainable_param_count == 12
    assert report.gptq_bundle_count == 12
    assert report.total_fp32_bytes == expected_fp32
    assert report.total_adam_bytes == expected_adam
    assert report.total_gptq_bytes == expected_gptq
    assert report.total_persistent_bytes == (
        expected_fp32 + expected_adam + expected_gptq
    )
    assert report.fp32_shard_count == 6
    assert report.max_fp32_shard_bytes == per_expert_elements * 4


def test_capacity_report_respects_non_trainable_and_no_gptq() -> None:
    qwen_config = Qwen35MoeManifestConfig(
        num_layers=1,
        num_experts=2,
        hidden_size=4,
        intermediate_size=4,
        trainable=False,
        include_gptq_cache=False,
    )
    specs = make_qwen35_moe_manifest_specs(qwen_config)
    manifest = TrainingBaseManifestBuilder().build(specs)

    report = capacity_report_from_manifest(manifest)

    assert report.param_count == 4
    assert report.trainable_param_count == 0
    assert report.gptq_bundle_count == 0
    assert report.total_adam_bytes == 0
    assert report.total_gptq_bytes == 0
    assert report.adam_shard_count == 0
    assert report.gptq_shard_count == 0


def test_capacity_report_to_dict_includes_gib_total() -> None:
    report = estimate_qwen35_moe_capacity(
        Qwen35MoeManifestConfig(
            num_layers=1,
            num_experts=1,
            hidden_size=4,
            intermediate_size=4,
        )
    )

    payload = report.to_dict()

    assert payload["total_persistent_bytes"] == report.total_persistent_bytes
    assert payload["total_persistent_gib"] == report.total_persistent_gib


def test_bytes_to_gib_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="num_bytes"):
        bytes_to_gib(-1)
