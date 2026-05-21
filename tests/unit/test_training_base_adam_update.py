"""Unit tests for CPU AdamW updates with FP8 state payloads."""

from __future__ import annotations

import struct

import pytest
import torch

from cfie_training.training_base import (
    AdamStateShardRecord,
    AdamWConfig,
    BlockFp8StateCodec,
    CpuAdamFp8StateStore,
    CpuAdamFp8Updater,
    FP32ShardStore,
    ParamShardRecord,
    ProgressStateWriter,
    TrainingWindowCommitter,
    adam_state_num_bytes,
    state_key,
)


def _fp32_bytes(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def _fp32_values(payload: bytes) -> tuple[float, ...]:
    return struct.unpack(f"<{len(payload) // 4}f", payload)


def test_block_fp8_codec_round_trips_state_with_block_scales() -> None:
    codec = BlockFp8StateCodec(block_size=2)
    state = torch.tensor([0.0, 1.0, -2.0, 100.0, -300.0],
                         dtype=torch.float32)

    payload = codec.encode(state)
    decoded = codec.decode(payload, state.numel())

    assert len(payload) == adam_state_num_bytes(state.numel(), block_size=2)
    assert decoded.shape == state.shape
    assert torch.allclose(decoded, state, rtol=0.08, atol=0.08)


def test_block_fp8_codec_rejects_wrong_payload_size() -> None:
    codec = BlockFp8StateCodec(block_size=4)

    with pytest.raises(ValueError, match="expected"):
        codec.decode(b"too-short", 8)


def test_cpu_adam_first_step_matches_fp32_adamw_before_state_quantization() -> None:
    updater = CpuAdamFp8Updater(
        AdamWConfig(lr=0.1, beta1=0.9, beta2=0.999, weight_decay=0.01)
    )
    master = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float32)
    grad = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float32)

    update = updater.step_param(
        param_id="param_a",
        master=master,
        grad=grad,
        step=1,
    )

    first_moment = grad * 0.1
    second_moment = grad.square() * 0.001
    expected_update = first_moment / 0.1 / (
        (second_moment / 0.001).sqrt() + 1e-8
    )
    expected_master = master * (1 - 0.1 * 0.01) - 0.1 * expected_update

    assert torch.allclose(update.master, expected_master)
    assert set(update.adam_updates) == {"m", "v"}
    assert len(update.first_moment) == updater.codec.encoded_num_bytes(3)
    assert len(update.second_moment) == updater.codec.encoded_num_bytes(3)
    assert update.grad_norm == pytest.approx(float(grad.norm().item()))
    assert update.update_norm == pytest.approx(float(expected_update.norm().item()))


def test_cpu_adam_update_from_stores_is_flushable_by_committer(tmp_path) -> None:
    fp32_store = FP32ShardStore.create(
        tmp_path / "fp32",
        {
            "param_a": ParamShardRecord("param_a", "fp32_0000.bin", 0, 2),
        },
    )
    fp32_store.flush_touched(
        {"param_a": _fp32_bytes([1.0, -2.0])},
        generation=1,
    )

    state_bytes = adam_state_num_bytes(2)
    adam_store = CpuAdamFp8StateStore.create(
        tmp_path / "adam",
        {
            state_key("param_a", "m"): AdamStateShardRecord(
                "param_a",
                "m",
                "adam_0000.bin",
                0,
                state_bytes,
            ),
            state_key("param_a", "v"): AdamStateShardRecord(
                "param_a",
                "v",
                "adam_0000.bin",
                state_bytes,
                state_bytes,
            ),
        },
    )
    updater = CpuAdamFp8Updater(AdamWConfig(lr=0.1))

    window_update = updater.apply_gradients_from_stores(
        fp32_store=fp32_store,
        adam_store=adam_store,
        grads={"param_a": torch.tensor([0.5, -0.25], dtype=torch.float32)},
        step=1,
    )
    committer = TrainingWindowCommitter(
        fp32_store,
        ProgressStateWriter.in_dir(tmp_path / "state"),
        adam_store=adam_store,
    )
    state = committer.commit_window(
        fp32_updates=window_update.fp32_updates,
        adam_updates=window_update.adam_updates,
        global_step=2,
        epoch=0,
        dataset_cursor="dataset:2",
        round_id=0,
        hot_set=window_update.touched_param_ids,
    )

    assert window_update.touched_param_ids == ("param_a",)
    assert _fp32_values(fp32_store.read_param("param_a")) == pytest.approx(
        (0.9, -1.9),
        abs=1e-6,
    )
    assert len(adam_store.read_state("param_a", "m")) == state_bytes
    assert len(adam_store.read_state("param_a", "v")) == state_bytes
    assert state.global_step == 2
    assert state.fp32_master_generation == 2
    assert state.optimizer_generation == 2


def test_apply_gradients_rejects_missing_fp32_param(tmp_path) -> None:
    fp32_store = FP32ShardStore.create(tmp_path / "fp32", {})
    adam_store = CpuAdamFp8StateStore.create(tmp_path / "adam", {})
    updater = CpuAdamFp8Updater(AdamWConfig(lr=0.1))

    with pytest.raises(KeyError):
        updater.apply_gradients_from_stores(
            fp32_store=fp32_store,
            adam_store=adam_store,
            grads={"missing": torch.ones(1)},
            step=1,
        )
