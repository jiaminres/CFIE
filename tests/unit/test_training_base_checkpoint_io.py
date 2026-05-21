"""Unit tests for checkpoint tensor loading helpers."""

from __future__ import annotations

import json

import pytest
import torch

from cfie_training.training_base import (
    CheckpointTensorLoadStats,
    iter_checkpoint_tensors,
)


def test_iter_checkpoint_tensors_prefers_hf_index_weight_map(tmp_path) -> None:
    torch.save(
        {
            "layers.0.mlp.experts.0.gate_proj.weight": torch.tensor([1.0]),
            "unreferenced.weight": torch.tensor([99.0]),
        },
        tmp_path / "pytorch_model-00001-of-00002.bin",
    )
    torch.save(
        {
            "layers.0.mlp.experts.0.up_proj.weight": torch.tensor([2.0]),
            "layers.0.mlp.experts.0.down_proj.weight": torch.tensor([3.0]),
        },
        tmp_path / "pytorch_model-00002-of-00002.bin",
    )
    torch.save(
        {"orphan.weight": torch.tensor([100.0])},
        tmp_path / "orphan.bin",
    )
    (tmp_path / "pytorch_model.bin.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 3},
                "weight_map": {
                    "layers.0.mlp.experts.0.gate_proj.weight": (
                        "pytorch_model-00001-of-00002.bin"
                    ),
                    "layers.0.mlp.experts.0.up_proj.weight": (
                        "pytorch_model-00002-of-00002.bin"
                    ),
                    "layers.0.mlp.experts.0.down_proj.weight": (
                        "pytorch_model-00002-of-00002.bin"
                    ),
                },
            }
        ),
        encoding="utf-8",
    )

    stats = CheckpointTensorLoadStats()
    loaded = dict(iter_checkpoint_tensors(tmp_path, stats=stats))

    assert sorted(loaded) == [
        "layers.0.mlp.experts.0.down_proj.weight",
        "layers.0.mlp.experts.0.gate_proj.weight",
        "layers.0.mlp.experts.0.up_proj.weight",
    ]
    assert torch.equal(
        loaded["layers.0.mlp.experts.0.down_proj.weight"],
        torch.tensor([3.0]),
    )
    assert stats.to_dict() == {
        "filtered_key_count": 0,
        "index_file_count": 1,
        "index_key_count": 3,
        "indexed_shard_count": 2,
        "opened_file_count": 2,
        "selected_key_count": 3,
        "selected_shard_count": 2,
        "yielded_tensor_bytes": 12,
        "yielded_tensor_count": 3,
        "yielded_tensor_elements": 3,
    }


def test_iter_checkpoint_tensors_filters_index_before_opening_shards(tmp_path) -> None:
    torch.save(
        {"layers.0.mlp.experts.0.gate_proj.weight": torch.tensor([1.0])},
        tmp_path / "pytorch_model-00001-of-00002.bin",
    )
    (tmp_path / "pytorch_model.bin.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layers.0.mlp.experts.0.gate_proj.weight": (
                        "pytorch_model-00001-of-00002.bin"
                    ),
                    "layers.0.self_attn.q_proj.weight": (
                        "missing-attn-shard.bin"
                    ),
                },
            }
        ),
        encoding="utf-8",
    )

    stats = CheckpointTensorLoadStats()
    loaded = dict(
        iter_checkpoint_tensors(
            tmp_path,
            key_filter=lambda name: ".mlp.experts." in name,
            stats=stats,
        )
    )

    assert sorted(loaded) == ["layers.0.mlp.experts.0.gate_proj.weight"]
    assert torch.equal(
        loaded["layers.0.mlp.experts.0.gate_proj.weight"],
        torch.tensor([1.0]),
    )
    assert stats.index_key_count == 2
    assert stats.selected_key_count == 1
    assert stats.filtered_key_count == 1
    assert stats.opened_file_count == 1


def test_iter_checkpoint_tensors_filters_single_torch_checkpoint(tmp_path) -> None:
    torch.save(
        {
            "layers.0.mlp.experts.0.gate_proj.weight": torch.tensor([1.0]),
            "layers.0.self_attn.q_proj.weight": torch.tensor([2.0]),
        },
        tmp_path / "model.pt",
    )

    stats = CheckpointTensorLoadStats()
    loaded = dict(
        iter_checkpoint_tensors(
            tmp_path / "model.pt",
            key_filter=lambda name: ".mlp.experts." in name,
            stats=stats,
        )
    )

    assert sorted(loaded) == ["layers.0.mlp.experts.0.gate_proj.weight"]
    assert stats.index_file_count == 0
    assert stats.selected_key_count == 1
    assert stats.filtered_key_count == 1
    assert stats.opened_file_count == 1


def test_iter_checkpoint_tensors_rejects_missing_index_shard(tmp_path) -> None:
    (tmp_path / "pytorch_model.bin.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layers.0.mlp.experts.0.gate_proj.weight": (
                        "missing-shard.bin"
                    ),
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="missing-shard"):
        dict(iter_checkpoint_tensors(tmp_path))


def test_iter_checkpoint_tensors_rejects_unsafe_index_shard_path(tmp_path) -> None:
    (tmp_path / "pytorch_model.bin.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layers.0.mlp.experts.0.gate_proj.weight": "../shard.bin",
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="relative"):
        dict(iter_checkpoint_tensors(tmp_path))
