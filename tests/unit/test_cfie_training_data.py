"""Unit tests for tokenizer-backed CFIE training batches."""

from __future__ import annotations

import json

from cfie_training.profiles import build_profile_config
from cfie_training.runtime.data import TokenizedDatasetBatchPlanner


def test_tokenized_dataset_batch_planner_reads_text_dataset(tmp_path) -> None:
    dataset_path = tmp_path / "train.txt"
    dataset_path.write_text(
        "\n".join(
            (
                "CFIE keeps GPU residency minimal.",
                "Experts rotate through CPU and NVMe tiers.",
                "Bucket-local updates release gradients immediately.",
            )
        ),
        encoding="utf-8",
    )
    cfg = build_profile_config("qwen35-35b-a3b")

    planner = TokenizedDatasetBatchPlanner(
        config=cfg,
        dataset_path=str(dataset_path),
        base_samples=2,
        tokens_per_sample=16,
    )

    first_batch = planner.batch_for_step(0)
    second_batch = planner.batch_for_step(1)

    assert planner.dataset_name == "train.txt"
    assert planner.sample_count == 3
    assert first_batch.source_kind == "tokenized_dataset"
    assert first_batch.dataset_name == "train.txt"
    assert first_batch.sample_indices == (0, 1)
    assert first_batch.loss_token_count == 32
    assert len(first_batch.token_rows) == 2
    assert len(first_batch.target_rows) == 2
    assert all(len(row) == 16 for row in first_batch.token_rows)
    assert all(len(row) == 16 for row in first_batch.target_rows)
    assert first_batch.token_rows[0] != first_batch.target_rows[0]
    assert second_batch.sample_indices == (2, 0)


def test_tokenized_dataset_batch_planner_reads_jsonl_dataset(tmp_path) -> None:
    dataset_path = tmp_path / "train.jsonl"
    dataset_path.write_text(
        "\n".join(
            (
                json.dumps({"text": "Linear attention layers reduce live KV state."}),
                json.dumps({"text": "Host AdamW state can be FP8-compressed."}),
            )
        ),
        encoding="utf-8",
    )
    cfg = build_profile_config("qwen35-35b-a3b")

    planner = TokenizedDatasetBatchPlanner(
        config=cfg,
        dataset_path=str(dataset_path),
        base_samples=2,
        tokens_per_sample=12,
        dataset_format="jsonl",
    )

    batch = planner.batch_for_step(0)

    assert batch.dataset_name == "train.jsonl"
    assert batch.sample_indices == (0, 1)
    assert batch.loss_token_count == 24
    assert all(len(row) == 12 for row in batch.token_rows)
