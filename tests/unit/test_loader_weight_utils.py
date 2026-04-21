"""权重加载工具测试。"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from cfie.config.schema import LoadConfig
from cfie.loader.weight_utils import (
    PreparedWeights,
    filter_duplicate_safetensors_files,
    iter_weight_tensors,
    prepare_weights,
)


def test_prepare_weights_auto_prefers_safetensors(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"stub")
    (tmp_path / "pytorch_model.bin").write_bytes(b"stub")

    prepared = prepare_weights(str(tmp_path), LoadConfig(load_format="auto"))

    assert prepared.use_safetensors is True
    assert prepared.weight_files == [str(tmp_path / "model.safetensors")]


def test_filter_duplicate_safetensors_by_index(tmp_path: Path) -> None:
    shard_keep = tmp_path / "model-00001-of-00002.safetensors"
    shard_drop = tmp_path / "model-00002-of-00002.safetensors"
    shard_keep.write_bytes(b"a")
    shard_drop.write_bytes(b"b")

    index_payload = {
        "weight_map": {
            "model.layers.0.weight": shard_keep.name,
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(index_payload), encoding="utf-8")

    filtered = filter_duplicate_safetensors_files(
        [str(shard_keep), str(shard_drop)],
        str(tmp_path),
    )

    assert filtered == [str(shard_keep)]


def test_iter_weight_tensors_for_pt_weights(tmp_path: Path) -> None:
    weight_file = tmp_path / "pytorch_model.bin"
    torch.save({"layer.weight": torch.ones(2, 2)}, weight_file)

    prepared = PreparedWeights(model_dir=str(tmp_path),
                               weight_files=[str(weight_file)],
                               use_safetensors=False)

    items = list(iter_weight_tensors(prepared))

    assert len(items) == 1
    assert items[0][0] == "layer.weight"
    assert items[0][1].shape == (2, 2)
