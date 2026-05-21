"""Unit tests for training-window progress persistence."""

from __future__ import annotations

import json

import pytest

from cfie_training.training_base import (
    ProgressStateWriter,
    TrainingProgressState,
    digest_hot_set,
)


def test_missing_progress_state_loads_initial_state(tmp_path) -> None:
    writer = ProgressStateWriter.in_dir(tmp_path)

    state = writer.load_latest_or_init()

    assert state == TrainingProgressState.initial()
    assert not writer.path.exists()


def test_write_after_flush_persists_atomic_progress_state(tmp_path) -> None:
    writer = ProgressStateWriter.in_dir(tmp_path)

    state = writer.write_after_flush(
        global_step=50,
        epoch=2,
        dataset_cursor="dataset_0003:offset_1024",
        round_id=7,
        hot_set=[
            "layers.1.experts.3.w13_weight",
            "layers.1.experts.3.w2_weight",
        ],
        consumed_samples=128,
        consumed_tokens=8192,
    )

    assert writer.path.exists()
    assert state.global_step == 50
    assert state.epoch == 2
    assert state.dataset_cursor == "dataset_0003:offset_1024"
    assert state.consumed_samples == 128
    assert state.consumed_tokens == 8192
    assert state.flush_generation == 50
    assert state.fp32_master_generation == 50
    assert state.optimizer_generation == 50
    assert state.gptq_cache_generation == 50
    assert writer.load_latest_or_init() == state
    assert not list(tmp_path.glob(".progress_state.json.*.tmp"))


def test_write_after_flush_allows_explicit_store_generations(tmp_path) -> None:
    writer = ProgressStateWriter(tmp_path / "progress.json")

    state = writer.write_after_flush(
        global_step=100,
        epoch=1,
        dataset_cursor="cursor",
        round_id=4,
        flush_generation=9,
        fp32_master_generation=10,
        optimizer_generation=11,
        gptq_cache_generation=12,
    )

    assert state.flush_generation == 9
    assert state.fp32_master_generation == 10
    assert state.optimizer_generation == 11
    assert state.gptq_cache_generation == 12


def test_hot_set_digest_is_stable_for_unordered_inputs() -> None:
    left = digest_hot_set(["b", "a", "c"])
    right = digest_hot_set({"c", "b", "a"})

    assert left == right
    assert left.startswith("sha256:")


def test_progress_state_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="global_step"):
        TrainingProgressState(global_step=-1)


def test_progress_state_rejects_invalid_json(tmp_path) -> None:
    writer = ProgressStateWriter.in_dir(tmp_path)
    writer.path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid progress state JSON"):
        writer.load_latest_or_init()


def test_generation_assertion_reports_mismatch() -> None:
    state = TrainingProgressState(
        fp32_master_generation=10,
        optimizer_generation=11,
        gptq_cache_generation=12,
    )

    with pytest.raises(ValueError, match="optimizer_generation"):
        state.assert_generations(
            fp32_master_generation=10,
            optimizer_generation=99,
            gptq_cache_generation=12,
        )


def test_load_ignores_unknown_future_fields(tmp_path) -> None:
    writer = ProgressStateWriter.in_dir(tmp_path)
    writer.path.write_text(
        json.dumps({
            "schema_version": 1,
            "global_step": 5,
            "epoch": 1,
            "dataset_cursor": "cursor",
            "future_field": "ignored",
        }),
        encoding="utf-8",
    )

    state = writer.load_latest_or_init()

    assert state.global_step == 5
    assert state.epoch == 1
    assert state.dataset_cursor == "cursor"
