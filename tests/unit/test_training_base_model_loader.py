"""Integration tests for real Qwen3.5 checkpoint import and training adapter."""

from __future__ import annotations

import os
import struct
import tempfile
from pathlib import Path

import pytest
import torch

from cfie_training.training_base import (
    Qwen35RealImporter,
    Qwen35_122B_CONFIG,
    TrainableParamSpec,
)
from cfie_training.training_base.window_plan import TrainingWindowBudget


CHECKPOINT = (
    "C:/Users/13642/.cache/huggingface/hub/"
    "models--Qwen--Qwen3.5-122B-A10B/snapshots/"
    "b000b2eb18a7f4cdf3153c4215842da339e09d99"
)


def _fp32_bytes(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


@pytest.mark.skipif(
    not Path(CHECKPOINT).exists(),
    reason="Qwen3.5-122B checkpoint not found on this machine",
)
class TestRealCheckpointImport:
    def test_importer_parses_packed_expert_keys(self) -> None:
        importer = Qwen35RealImporter(CHECKPOINT, num_layers=1, num_experts=2)
        keys = list(importer.iter_expert_weights(layers=(0,), experts=(0,)))
        assert len(keys) == 2
        param_ids = {k[0] for k in keys}
        assert "layers.0.experts.0.w13_weight" in param_ids
        assert "layers.0.experts.0.w2_weight" in param_ids

    def test_import_smoke_subset_to_stores(self) -> None:
        tmp = tempfile.mkdtemp()
        try:
            importer = Qwen35RealImporter(CHECKPOINT, num_layers=1, num_experts=2)
            fp32, adam, gptq, manifest, progress = importer.import_to_stores(
                tmp, layers=(0,), experts=(0, 1),
            )
            assert len(fp32.records) == 4
            for pid, rec in fp32.records.items():
                data = fp32.read_param(pid)
                assert len(data) == rec.num_bytes
            for pid in adam.records:
                record = adam.records[pid]
                assert record.num_bytes > 0
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    def test_import_then_training_loop(self) -> None:
        from cfie_training.training_base import (
            AdamWConfig,
            NoOpTrainingModel,
            TrainingDataBatchInput,
            TrainingLoop,
            TrainingLoopConfig,
        )

        tmp = tempfile.mkdtemp()
        try:
            importer = Qwen35RealImporter(CHECKPOINT, num_layers=1, num_experts=2)
            fp32, adam, gptq, manifest, progress = importer.import_to_stores(
                tmp, layers=(0,), experts=(0, 1),
            )

            hot_params = tuple(fp32.records.keys())
            loop = TrainingLoop.from_stores(
                fp32_store=fp32, adam_store=adam,
                gptq_store=gptq, manifest=manifest,
                progress_writer=progress,
                hot_param_ids=hot_params,
                config=TrainingLoopConfig(
                    adam_config=AdamWConfig(lr=0.001),
                    bucket_capacity_bytes=4 << 20,
                    window_steps=3, enable_peak_monitor=False,
                    dynamic_hotset_max_experts=1,
                ),
            )

            param_sizes = {}
            for pid, rec in fp32.records.items():
                param_sizes[pid] = rec.num_elements
            loop.attach_model(NoOpTrainingModel(param_sizes=param_sizes))

            batches = [
                TrainingDataBatchInput(
                    input_ids=torch.randint(0, 100, (1, 8)),
                    global_step=i, epoch=0, dataset_cursor=f"d:{i}",
                )
                for i in range(3)
            ]
            loop.attach_dataloader(iter(batches))
            results = loop.run(num_steps=3)

            assert len(results) == 3
            assert any(r.window_committed for r in results)
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)


class TestManifestSpecs:
    def test_builds_correct_specs_for_subset(self) -> None:
        importer = Qwen35RealImporter(".", num_layers=2, num_experts=3)
        specs = importer.build_manifest_specs(layers=(0, 1), experts=(0, 1))
        assert len(specs) == 2 * 2 * 2  # 2 layers × 2 experts × 2 weights
        for spec in specs:
            assert spec.trainable
            assert spec.num_elements > 0
            assert "layers" in spec.param_id
            assert "experts" in spec.param_id
            assert spec.param_id.endswith("_weight")
