"""End-to-end smoke tests for the training-base prototype pipeline."""

from __future__ import annotations

import struct

import pytest
import torch

from cfie_training.training_base import (
    AdamWConfig,
    decode_gptq_marlin_bundle,
    ManifestShardConfig,
    pack_gptq_int4_qweight,
    pack_gptq_int4_qzeros,
    Qwen35MoeCheckpointImportConfig,
    RoutedExpert,
    RouterPrefetchDepthTuningConfig,
    TrainingSmokePipeline,
    TrainingSmokePipelineConfig,
    TrainingSmokeStepInput,
    TrainingResourceThresholds,
    TrainingWindowBudget,
)


def _weights() -> dict[str, torch.Tensor]:
    return _weights_with_experts(2)


def _weights_with_experts(num_experts: int) -> dict[str, torch.Tensor]:
    weights: dict[str, torch.Tensor] = {}
    for expert_id in range(num_experts):
        base = 1.0 + expert_id * 9.0
        prefix = f"layers.0.mlp.experts.{expert_id}"
        weights[f"{prefix}.gate_proj.weight"] = torch.tensor([base, base + 1])
        weights[f"{prefix}.up_proj.weight"] = torch.tensor([base + 2, base + 3])
        weights[f"{prefix}.down_proj.weight"] = torch.tensor([base + 4, base + 5])
    return weights


def _fp32_values(payload: bytes) -> tuple[float, ...]:
    return struct.unpack(f"<{len(payload) // 4}f", payload)


def _gptq_weights() -> dict[str, torch.Tensor]:
    qzeros = torch.full((1, 2), 7, dtype=torch.int32)
    return {
        "layers.0.mlp.experts.0.gate_proj.qweight": pack_gptq_int4_qweight(
            torch.tensor([[8, 9], [10, 7]], dtype=torch.int32)
        ),
        "layers.0.mlp.experts.0.gate_proj.scales": torch.tensor([[1.0, 2.0]]),
        "layers.0.mlp.experts.0.gate_proj.qzeros": pack_gptq_int4_qzeros(qzeros),
        "layers.0.mlp.experts.0.gate_proj.g_idx": torch.tensor([0, 0]),
        "layers.0.mlp.experts.0.up_proj.qweight": pack_gptq_int4_qweight(
            torch.tensor([[9, 8], [7, 10]], dtype=torch.int32)
        ),
        "layers.0.mlp.experts.0.up_proj.scales": torch.tensor([[1.0, 2.0]]),
        "layers.0.mlp.experts.0.up_proj.qzeros": pack_gptq_int4_qzeros(qzeros),
        "layers.0.mlp.experts.0.up_proj.g_idx": torch.tensor([0, 0]),
        "layers.0.mlp.experts.0.down_proj.qweight": pack_gptq_int4_qweight(
            torch.tensor([[10, 8], [8, 6]], dtype=torch.int32)
        ),
        "layers.0.mlp.experts.0.down_proj.scales": torch.tensor([[1.0, 2.0]]),
        "layers.0.mlp.experts.0.down_proj.qzeros": pack_gptq_int4_qzeros(qzeros),
        "layers.0.mlp.experts.0.down_proj.g_idx": torch.tensor([0, 0]),
    }


def test_training_smoke_pipeline_runs_import_update_requant_prefetch_commit(
    tmp_path,
) -> None:
    hot_param_ids = (
        "layers.0.experts.0.w13_weight",
        "layers.0.experts.0.w2_weight",
    )
    pipeline = TrainingSmokePipeline.from_qwen35_moe_checkpoint(
        _weights(),
        TrainingSmokePipelineConfig(
            root=tmp_path,
            hot_param_ids=hot_param_ids,
            import_config=Qwen35MoeCheckpointImportConfig(gptq_group_size=2),
            manifest_config=ManifestShardConfig(
                fp32_shard_bytes=64,
                adam_shard_bytes=128,
                gptq_shard_bytes=128,
                gptq_group_size=2,
            ),
            adam_config=AdamWConfig(lr=0.1),
            window_budget=TrainingWindowBudget(window_steps=50),
            bucket_capacity_bytes=8,
            resident_cache_capacity_bytes=128,
            resource_thresholds=TrainingResourceThresholds(
                expert_cache_miss_rate=0.0,
                expert_cache_miss_steps=1,
            ),
        ),
    )

    result = pipeline.run_step(
        TrainingSmokeStepInput(
            gradients={
                "layers.0.experts.0.w13_weight": torch.tensor(
                    [0.1, 0.2, 0.3, 0.4],
                    dtype=torch.float32,
                ),
                "layers.0.experts.0.w2_weight": torch.tensor(
                    [0.5, 0.6],
                    dtype=torch.float32,
                ),
            },
            global_step=1,
            epoch=0,
            dataset_cursor="dataset:1",
            current_experts=(RoutedExpert(0, 1, score=1.0),),
            predicted_experts=(RoutedExpert(0, 0, score=0.5),),
            consumed_samples=2,
            consumed_tokens=16,
        )
    )

    assert result.progress_state.global_step == 1
    assert result.progress_state.consumed_tokens == 16
    assert result.touched_param_ids == hot_param_ids
    assert result.update_summary.touched_param_ids == hot_param_ids
    assert result.update_summary.drained_bucket_ids == (0, 1)
    assert _fp32_values(
        pipeline.init_result.fp32_store.read_param(
            "layers.0.experts.0.w13_weight"
        )
    ) == pytest.approx((0.9, 1.9, 2.9, 3.9), abs=1e-6)
    assert _fp32_values(
        pipeline.init_result.fp32_store.read_param(
            "layers.0.experts.0.w2_weight"
        )
    ) == pytest.approx((4.9, 5.9), abs=1e-6)
    assert result.prefetch_result.loaded_bundle_ids == (
        "layers.0.experts.1.w13_weight",
        "layers.0.experts.1.w2_weight",
    )
    assert result.prefetch_result.ready_bundle_ids == (
        "layers.0.experts.1.w13_weight",
        "layers.0.experts.1.w2_weight",
    )
    assert result.next_prefetch_result.loaded_bundle_ids == ()
    assert result.locked_bundle_ids == ()
    assert "layers.0.experts.0.w13_weight" not in result.resident_bundle_ids
    assert result.expert_cache_miss_rate == 1.0
    assert [event.name for event in result.threshold_events] == [
        "expert_cache_miss"
    ]
    assert result.resource_peaks is not None
    assert result.resource_peaks.snapshots_seen == 1
    assert result.flush_seconds >= 0.0


def test_training_smoke_pipeline_rejects_cold_gradient(tmp_path) -> None:
    pipeline = TrainingSmokePipeline.from_qwen35_moe_checkpoint(
        _weights(),
        TrainingSmokePipelineConfig(
            root=tmp_path,
            hot_param_ids=("layers.0.experts.0.w13_weight",),
            import_config=Qwen35MoeCheckpointImportConfig(gptq_group_size=2),
            manifest_config=ManifestShardConfig(
                fp32_shard_bytes=64,
                adam_shard_bytes=128,
                gptq_shard_bytes=128,
                gptq_group_size=2,
            ),
        ),
    )

    with pytest.raises(KeyError, match="hot set"):
        pipeline.run_step(
            TrainingSmokeStepInput(
                gradients={
                    "layers.0.experts.1.w13_weight": torch.ones(4),
                },
                global_step=1,
                epoch=0,
                dataset_cursor="dataset:1",
            )
        )


def test_training_smoke_pipeline_updates_raw_marlin_bundles(tmp_path) -> None:
    hot_param_ids = (
        "layers.0.experts.0.w13_weight",
        "layers.0.experts.0.w2_weight",
    )
    pipeline = TrainingSmokePipeline.from_qwen35_moe_checkpoint(
        _gptq_weights(),
        TrainingSmokePipelineConfig(
            root=tmp_path,
            hot_param_ids=hot_param_ids,
            import_config=Qwen35MoeCheckpointImportConfig(gptq_group_size=8),
            manifest_config=ManifestShardConfig(
                fp32_shard_bytes=64,
                adam_shard_bytes=128,
                gptq_shard_bytes=4096,
                gptq_group_size=8,
            ),
            adam_config=AdamWConfig(lr=0.1),
            window_budget=TrainingWindowBudget(window_steps=50),
            bucket_capacity_bytes=64,
            resident_cache_capacity_bytes=4096,
        ),
    )
    before = pipeline.init_result.gptq_store.read_bundle(
        "layers.0.experts.0.w13_weight"
    )

    pipeline.run_step(
        TrainingSmokeStepInput(
            gradients={
                "layers.0.experts.0.w13_weight": torch.ones(8),
                "layers.0.experts.0.w2_weight": torch.ones(4),
            },
            global_step=1,
            epoch=0,
            dataset_cursor="dataset:1",
        )
    )
    after = pipeline.init_result.gptq_store.read_bundle(
        "layers.0.experts.0.w13_weight"
    )
    decoded = decode_gptq_marlin_bundle(after)

    assert before != after
    assert set(decoded.tensors) == {
        "w1.g_idx",
        "w1.qweight",
        "w1.qzeros",
        "w1.scales",
        "w3.g_idx",
        "w3.qweight",
        "w3.qzeros",
        "w3.scales",
    }


def test_training_smoke_pipeline_prefetches_predicted_experts_one_step_ahead(
    tmp_path,
) -> None:
    pipeline = TrainingSmokePipeline.from_qwen35_moe_checkpoint(
        _weights(),
        TrainingSmokePipelineConfig(
            root=tmp_path,
            hot_param_ids=("layers.0.experts.0.w13_weight",),
            import_config=Qwen35MoeCheckpointImportConfig(gptq_group_size=2),
            manifest_config=ManifestShardConfig(
                fp32_shard_bytes=64,
                adam_shard_bytes=128,
                gptq_shard_bytes=128,
                gptq_group_size=2,
            ),
            adam_config=AdamWConfig(lr=0.1),
            window_budget=TrainingWindowBudget(window_steps=50),
            bucket_capacity_bytes=16,
            resident_cache_capacity_bytes=128,
            resource_thresholds=TrainingResourceThresholds(
                expert_cache_miss_rate=0.0,
                expert_cache_miss_steps=1,
            ),
        ),
    )

    first = pipeline.run_step(
        TrainingSmokeStepInput(
            gradients={
                "layers.0.experts.0.w13_weight": torch.tensor(
                    [0.1, 0.2, 0.3, 0.4],
                    dtype=torch.float32,
                ),
            },
            global_step=1,
            epoch=0,
            dataset_cursor="dataset:1",
            current_experts=(),
            predicted_experts=(RoutedExpert(0, 1, score=1.0),),
        )
    )

    assert first.prefetch_result.loaded_bundle_ids == ()
    assert first.next_prefetch_result.loaded_bundle_ids == (
        "layers.0.experts.1.w13_weight",
        "layers.0.experts.1.w2_weight",
    )
    assert first.expert_cache_miss_rate == 0.0

    second = pipeline.run_step(
        TrainingSmokeStepInput(
            gradients={},
            global_step=2,
            epoch=0,
            dataset_cursor="dataset:2",
            current_experts=(RoutedExpert(0, 1, score=1.0),),
            predicted_experts=(),
        )
    )

    assert second.prefetch_result.loaded_bundle_ids == ()
    assert second.prefetch_result.ready_bundle_ids == (
        "layers.0.experts.1.w13_weight",
        "layers.0.experts.1.w2_weight",
    )
    assert second.expert_cache_miss_rate == 0.0
    assert [event.name for event in second.threshold_events] == []


def test_training_smoke_pipeline_records_next_prefetch_capacity_pressure(
    tmp_path,
) -> None:
    pipeline = TrainingSmokePipeline.from_qwen35_moe_checkpoint(
        _weights_with_experts(3),
        TrainingSmokePipelineConfig(
            root=tmp_path,
            hot_param_ids=("layers.0.experts.0.w13_weight",),
            import_config=Qwen35MoeCheckpointImportConfig(gptq_group_size=2),
            manifest_config=ManifestShardConfig(
                fp32_shard_bytes=64,
                adam_shard_bytes=128,
                gptq_shard_bytes=256,
                gptq_group_size=2,
            ),
            adam_config=AdamWConfig(lr=0.1),
            window_budget=TrainingWindowBudget(window_steps=50),
            bucket_capacity_bytes=16,
            resident_cache_capacity_bytes=256,
            router_prefetch_depth=4,
            prefetch_depth_tuning_config=RouterPrefetchDepthTuningConfig(
                min_prefetch_depth=1,
                max_prefetch_depth=4,
                decrease_step=2,
                capacity_pressure_steps=1,
            ),
        ),
    )
    current_bundle_ids = (
        "layers.0.experts.1.w13_weight",
        "layers.0.experts.1.w2_weight",
    )
    pipeline.resident_cache.capacity_bytes = sum(
        pipeline.init_result.gptq_store.records[bundle_id].num_bytes
        for bundle_id in current_bundle_ids
    )

    result = pipeline.run_step(
        TrainingSmokeStepInput(
            gradients={},
            global_step=1,
            epoch=0,
            dataset_cursor="dataset:1",
            current_experts=(RoutedExpert(0, 1, score=1.0),),
            predicted_experts=(RoutedExpert(0, 2, score=1.0),),
        )
    )

    assert result.progress_state.global_step == 1
    assert result.prefetch_result.loaded_bundle_ids == current_bundle_ids
    assert result.next_prefetch_result.failed_bundle_ids == (
        "layers.0.experts.2.w13_weight",
        "layers.0.experts.2.w2_weight",
    )
    assert result.next_prefetch_result.has_capacity_pressure
    assert result.expert_cache_capacity_pressure_rate == 1.0
    assert [event.name for event in result.threshold_events] == [
        "expert_cache_capacity_pressure"
    ]
    assert result.prefetch_depth_decision is not None
    assert result.prefetch_depth_decision.reason == "capacity_pressure"
    assert result.prefetch_depth_decision.old_depth == 4
    assert result.prefetch_depth_decision.new_depth == 2
    assert result.router_prefetch_depth == 2
    assert result.locked_bundle_ids == ()


def test_training_smoke_pipeline_increases_prefetch_depth_on_current_miss(
    tmp_path,
) -> None:
    pipeline = TrainingSmokePipeline.from_qwen35_moe_checkpoint(
        _weights_with_experts(2),
        TrainingSmokePipelineConfig(
            root=tmp_path,
            hot_param_ids=("layers.0.experts.0.w13_weight",),
            import_config=Qwen35MoeCheckpointImportConfig(gptq_group_size=2),
            manifest_config=ManifestShardConfig(
                fp32_shard_bytes=64,
                adam_shard_bytes=128,
                gptq_shard_bytes=128,
                gptq_group_size=2,
            ),
            adam_config=AdamWConfig(lr=0.1),
            window_budget=TrainingWindowBudget(window_steps=50),
            bucket_capacity_bytes=16,
            resident_cache_capacity_bytes=128,
            router_prefetch_depth=1,
            prefetch_depth_tuning_config=RouterPrefetchDepthTuningConfig(
                min_prefetch_depth=1,
                max_prefetch_depth=4,
                increase_step=1,
                miss_rate_threshold=0.0,
                miss_rate_steps=1,
            ),
        ),
    )

    result = pipeline.run_step(
        TrainingSmokeStepInput(
            gradients={},
            global_step=1,
            epoch=0,
            dataset_cursor="dataset:1",
            current_experts=(RoutedExpert(0, 1, score=1.0),),
            predicted_experts=(),
        )
    )

    assert result.expert_cache_miss_rate == 1.0
    assert result.prefetch_depth_decision is not None
    assert result.prefetch_depth_decision.reason == "cache_miss"
    assert result.prefetch_depth_decision.old_depth == 1
    assert result.prefetch_depth_decision.new_depth == 2
    assert result.router_prefetch_depth == 2


def test_dynamic_hotset_with_scheduler_coverage(tmp_path) -> None:
    hot_param_ids = (
        "layers.0.experts.0.w13_weight",
        "layers.0.experts.0.w2_weight",
    )
    pipeline = TrainingSmokePipeline.from_qwen35_moe_checkpoint(
        _weights_with_experts(4),
        TrainingSmokePipelineConfig(
            root=tmp_path,
            hot_param_ids=hot_param_ids,
            import_config=Qwen35MoeCheckpointImportConfig(gptq_group_size=2),
            manifest_config=ManifestShardConfig(
                fp32_shard_bytes=256,
                adam_shard_bytes=512,
                gptq_shard_bytes=512,
                gptq_group_size=2,
            ),
            adam_config=AdamWConfig(lr=0.1),
            window_budget=TrainingWindowBudget(window_steps=50),
            bucket_capacity_bytes=32,
            resident_cache_capacity_bytes=1024,
            dynamic_hotset_max_experts=3,
            coverage_constraint=None,
        ),
    )

    result = pipeline.run_step(
        TrainingSmokeStepInput(
            gradients={
                "layers.0.experts.0.w13_weight": torch.tensor(
                    [0.1, 0.2, 0.3, 0.4],
                    dtype=torch.float32,
                ),
                "layers.0.experts.0.w2_weight": torch.tensor(
                    [0.5, 0.6],
                    dtype=torch.float32,
                ),
            },
            global_step=1,
            epoch=0,
            dataset_cursor="dataset:1",
            current_experts=(
                RoutedExpert(0, 1, score=1.0),
                RoutedExpert(0, 2, score=0.5),
                RoutedExpert(0, 3, score=0.3),
            ),
            predicted_experts=(),
            consumed_samples=2,
            consumed_tokens=16,
        )
    )

    assert result.hotset_switch_decision is not None
    assert result.hotset_switch_decision.reason == "hot_set_scheduler"
    assert len(result.hot_param_ids) >= 2
