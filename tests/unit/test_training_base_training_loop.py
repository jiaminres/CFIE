"""End-to-end tests for the training loop orchestrating all training_base components."""

from __future__ import annotations

import struct

import pytest
import torch

from cfie_training.training_base import (
    AdamStateShardRecord,
    AdamWConfig,
    CpuAdamFp8StateStore,
    FP32ShardStore,
    GptqCacheRecord,
    GptqCacheStore,
    NoOpTrainingModel,
    ParamShardRecord,
    ProgressStateWriter,
    SymmetricInt4GptqCodec,
    SymmetricInt4GptqLayout,
    TrainingDataBatchInput,
    TrainingLoop,
    TrainingLoopConfig,
    adam_state_num_bytes,
    gptq_bundle_num_bytes,
    state_key,
)


def _fp32_bytes(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def _make_fp32_store(tmp_path, param_id, num_elements):
    store = FP32ShardStore.create(
        tmp_path / "fp32",
        {param_id: ParamShardRecord(param_id, "shard_0000.bin", 0, num_elements)},
    )
    store.flush_touched(
        {param_id: _fp32_bytes([float(i + 1) for i in range(num_elements)])},
        generation=0,
    )
    return store


def _make_adam_store(tmp_path, param_ids, num_elements):
    state_bytes = adam_state_num_bytes(num_elements)
    records = {}
    offset = 0
    for pid in param_ids:
        for comp in ("m", "v"):
            records[state_key(pid, comp)] = AdamStateShardRecord(
                pid, comp, "adam_0000.bin", offset, state_bytes,
            )
            offset += state_bytes
    return CpuAdamFp8StateStore.create(tmp_path / "adam", records)


def _make_gptq_store(tmp_path, bundle_ids, group_size=2):
    codec = SymmetricInt4GptqCodec(SymmetricInt4GptqLayout(group_size=group_size))
    records = {}
    offset = 0
    for bid, num_elements in bundle_ids.items():
        num_bytes = gptq_bundle_num_bytes(num_elements, group_size=group_size)
        records[bid] = GptqCacheRecord(
            bid, "gptq_0000.bin", offset, num_bytes,
            quant_layout_hash=codec.layout_hash,
        )
        offset += num_bytes
    return GptqCacheStore.create(tmp_path / "gptq", records)


class TestTrainingLoopE2E:
    def test_runs_forward_backward_commit_cycle(self, tmp_path):
        n = 4
        fp32_store = _make_fp32_store(tmp_path, "param_a", n)
        adam_store = _make_adam_store(tmp_path, ["param_a"], n)
        gptq_store = _make_gptq_store(tmp_path, {"bundle_a": n})
        progress_writer = ProgressStateWriter.in_dir(tmp_path / "state")

        loop = TrainingLoop.from_stores(
            fp32_store=fp32_store, adam_store=adam_store,
            gptq_store=gptq_store, manifest=None,
            progress_writer=progress_writer,
            hot_param_ids=("param_a",),
            config=TrainingLoopConfig(
                adam_config=AdamWConfig(lr=0.01),
                bucket_capacity_bytes=256, shadow_device="cpu",
                window_steps=3, dynamic_hotset_max_experts=1,
            ),
        )
        loop.attach_model(NoOpTrainingModel(param_sizes={"param_a": n}))

        batches = [
            TrainingDataBatchInput(
                input_ids=torch.randint(0, 100, (2, 8)),
                global_step=i, epoch=0, dataset_cursor=f"d:{i}",
                consumed_samples=2, consumed_tokens=16,
            )
            for i in range(4)
        ]
        loop.attach_dataloader(iter(batches))
        results = loop.run(num_steps=4)

        assert len(results) == 4
        for r in results:
            assert r.global_step > 0
            assert isinstance(r.loss, float)
        assert any(r.window_committed for r in results)

    def test_updates_fp32_store(self, tmp_path):
        n = 2
        fp32_store = _make_fp32_store(tmp_path, "param_a", n)
        adam_store = _make_adam_store(tmp_path, ["param_a"], n)
        gptq_store = _make_gptq_store(tmp_path, {"bundle_a": n})
        progress_writer = ProgressStateWriter.in_dir(tmp_path / "state")

        loop = TrainingLoop.from_stores(
            fp32_store=fp32_store, adam_store=adam_store,
            gptq_store=gptq_store, manifest=None,
            progress_writer=progress_writer, hot_param_ids=("param_a",),
            config=TrainingLoopConfig(
                adam_config=AdamWConfig(lr=0.1),
                bucket_capacity_bytes=256, window_steps=1,
                dynamic_hotset_max_experts=1,
            ),
        )
        loop.attach_model(NoOpTrainingModel(param_sizes={"param_a": n}))

        batch = TrainingDataBatchInput(
            input_ids=torch.zeros((1, 4), dtype=torch.long),
            global_step=0, epoch=0, dataset_cursor="d:0",
        )
        loop.attach_dataloader(iter([batch]))
        loop.run(num_steps=1)

        stored = fp32_store.read_param("param_a")
        stored_vals = struct.unpack("<2f", stored)
        assert stored_vals != (1.0, 2.0)
        assert all(v != 0.0 for v in stored_vals)

    def test_honors_window_boundary(self, tmp_path):
        n = 2
        fp32_store = _make_fp32_store(tmp_path, "p", n)
        adam_store = _make_adam_store(tmp_path, ["p"], n)
        gptq_store = _make_gptq_store(tmp_path, {"bundle_p": n})
        progress_writer = ProgressStateWriter.in_dir(tmp_path / "state")

        loop = TrainingLoop.from_stores(
            fp32_store=fp32_store, adam_store=adam_store,
            gptq_store=gptq_store, manifest=None,
            progress_writer=progress_writer, hot_param_ids=("p",),
            config=TrainingLoopConfig(
                adam_config=AdamWConfig(lr=0.01),
                bucket_capacity_bytes=256, window_steps=2,
                dynamic_hotset_max_experts=1,
            ),
        )
        loop.attach_model(NoOpTrainingModel(param_sizes={"p": n}))

        batches = [
            TrainingDataBatchInput(
                input_ids=torch.ones((1, 4), dtype=torch.long),
                global_step=i, epoch=0, dataset_cursor=f"d:{i}",
            )
            for i in range(5)
        ]
        loop.attach_dataloader(iter(batches))
        results = loop.run(num_steps=5)
        commit_count = sum(1 for r in results if r.window_committed)
        assert commit_count >= 2

    def test_handles_empty_gradients(self, tmp_path):
        n = 2
        fp32_store = _make_fp32_store(tmp_path, "p", n)
        adam_store = _make_adam_store(tmp_path, ["p"], n)
        gptq_store = _make_gptq_store(tmp_path, {})
        progress_writer = ProgressStateWriter.in_dir(tmp_path / "state")

        loop = TrainingLoop.from_stores(
            fp32_store=fp32_store, adam_store=adam_store,
            gptq_store=gptq_store, manifest=None,
            progress_writer=progress_writer, hot_param_ids=("p",),
            config=TrainingLoopConfig(
                bucket_capacity_bytes=256, window_steps=2,
                dynamic_hotset_max_experts=1,
            ),
        )
        loop.attach_model(NoOpTrainingModel(param_sizes={"p": n}))
        batch = TrainingDataBatchInput(
            input_ids=torch.zeros((1, 4), dtype=torch.long),
            global_step=0, epoch=0, dataset_cursor="d:0",
        )
        loop.attach_dataloader(iter([batch]))
        results = loop.run(num_steps=1)
        assert len(results) == 1

    def test_with_smoke_pipeline_stores(self, tmp_path):
        n = 4
        fp32_store = _make_fp32_store(tmp_path, "p", n)
        adam_store = _make_adam_store(tmp_path, ["p"], n)
        gptq_store = _make_gptq_store(tmp_path, {"bundle_p": n})
        progress_writer = ProgressStateWriter.in_dir(tmp_path / "state")

        loop = TrainingLoop.from_stores(
            fp32_store=fp32_store, adam_store=adam_store,
            gptq_store=gptq_store, manifest=None,
            progress_writer=progress_writer, hot_param_ids=("p",),
            config=TrainingLoopConfig(
                adam_config=AdamWConfig(lr=0.1),
                bucket_capacity_bytes=256, window_steps=3,
                enable_peak_monitor=False,
                dynamic_hotset_max_experts=1,
            ),
        )
        loop.attach_model(NoOpTrainingModel(param_sizes={"p": n}))

        batches = [
            TrainingDataBatchInput(
                input_ids=torch.randint(0, 100, (1, 8)),
                global_step=i, epoch=0, dataset_cursor=f"d:{i}",
            )
            for i in range(6)
        ]
        loop.attach_dataloader(iter(batches))
        results = loop.run(num_steps=6)
        assert len(results) == 6
        assert any(r.window_committed for r in results)
        assert progress_writer.path.exists()


class TestTrainingLoopErrors:
    def test_no_dataloader_raises(self, tmp_path):
        fp32_store = _make_fp32_store(tmp_path, "p", 1)
        adam_store = _make_adam_store(tmp_path, ["p"], 1)
        gptq_store = _make_gptq_store(tmp_path, {})
        progress_writer = ProgressStateWriter.in_dir(tmp_path / "state")

        loop = TrainingLoop.from_stores(
            fp32_store=fp32_store, adam_store=adam_store,
            gptq_store=gptq_store, manifest=None,
            progress_writer=progress_writer, hot_param_ids=("p",),
            config=TrainingLoopConfig(enable_peak_monitor=False),
        )

        with pytest.raises(RuntimeError, match="dataloader"):
            loop.run(num_steps=1)


class TestTrainingLoopConfig:
    def test_defaults(self):
        cfg = TrainingLoopConfig()
        assert cfg.window_steps == 50
        assert cfg.max_sealed_buckets == 4
        assert cfg.bucket_capacity_bytes == 1 << 20
