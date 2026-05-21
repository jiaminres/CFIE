"""Unit tests for training-base command line helpers."""

from __future__ import annotations

import json
import struct

import torch

from cfie_training.training_base import FP32ShardStore, iter_checkpoint_tensors
from cfie_training.training_base.cli import main


def _write_checkpoint(path) -> None:
    weights = {
        "layers.0.mlp.experts.0.gate_proj.weight": torch.tensor([1.0, 2.0]),
        "layers.0.mlp.experts.0.up_proj.weight": torch.tensor([3.0, 4.0]),
        "layers.0.mlp.experts.0.down_proj.weight": torch.tensor([5.0, 6.0]),
    }
    torch.save(weights, path)


def _fp32_values(payload: bytes) -> tuple[float, ...]:
    return struct.unpack(f"<{len(payload) // 4}f", payload)


def test_estimate_qwen35_moe_cli_outputs_capacity_json(capsys) -> None:
    exit_code = main(
        [
            "estimate-qwen35-moe",
            "--num-layers",
            "1",
            "--num-experts",
            "2",
            "--hidden-size",
            "4",
            "--intermediate-size",
            "8",
            "--tp-size",
            "2",
            "--fp32-shard-bytes",
            "128",
            "--adam-shard-bytes",
            "128",
            "--gptq-shard-bytes",
            "128",
            "--gptq-group-size",
            "8",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["command"] == "estimate-qwen35-moe"
    assert payload["dry_run"] is True
    assert payload["capacity"]["param_count"] == 4
    assert payload["capacity"]["gptq_bundle_count"] == 4


def test_iter_checkpoint_tensors_reads_torch_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    _write_checkpoint(checkpoint)

    loaded = dict(iter_checkpoint_tensors(checkpoint))

    assert sorted(loaded) == [
        "layers.0.mlp.experts.0.down_proj.weight",
        "layers.0.mlp.experts.0.gate_proj.weight",
        "layers.0.mlp.experts.0.up_proj.weight",
    ]
    assert torch.equal(
        loaded["layers.0.mlp.experts.0.gate_proj.weight"],
        torch.tensor([1.0, 2.0]),
    )


def test_init_qwen35_moe_cli_dry_run_does_not_write_store(tmp_path, capsys) -> None:
    checkpoint = tmp_path / "model.pt"
    output_root = tmp_path / "training_store"
    _write_checkpoint(checkpoint)

    exit_code = main(
        [
            "init-qwen35-moe",
            "--checkpoint",
            str(checkpoint),
            "--root",
            str(output_root),
            "--dry-run",
            "--fp32-shard-bytes",
            "64",
            "--adam-shard-bytes",
            "128",
            "--gptq-shard-bytes",
            "128",
            "--gptq-group-size",
            "2",
            "--progress-every-tensors",
            "1",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    progress_events = [
        json.loads(line)
        for line in captured.err.splitlines()
        if line.strip()
    ]

    assert exit_code == 0
    assert payload["dry_run"] is True
    assert payload["imported_param_count"] == 2
    assert payload["capacity"]["param_count"] == 2
    assert payload["checkpoint_read"]["yielded_tensor_count"] == 3
    assert payload["checkpoint_read"]["yielded_tensor_bytes"] == 24
    assert payload["phase_seconds"]["checkpoint_import"] >= 0.0
    assert payload["phase_seconds"]["manifest_build"] >= 0.0
    assert payload["phase_seconds"]["store_write"] == 0.0
    assert len(progress_events) == 3
    assert progress_events[-1]["event"] == "checkpoint_read_progress"
    assert progress_events[-1]["yielded_tensor_count"] == 3
    assert not output_root.exists()


def test_init_qwen35_moe_cli_writes_training_store(tmp_path, capsys) -> None:
    checkpoint = tmp_path / "model.pt"
    output_root = tmp_path / "training_store"
    _write_checkpoint(checkpoint)

    exit_code = main(
        [
            "init-qwen35-moe",
            "--checkpoint",
            str(checkpoint),
            "--root",
            str(output_root),
            "--fp32-shard-bytes",
            "64",
            "--adam-shard-bytes",
            "128",
            "--gptq-shard-bytes",
            "128",
            "--gptq-group-size",
            "2",
            "--generation",
            "7",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    store = FP32ShardStore.load(output_root / "fp32")

    assert exit_code == 0
    assert payload["dry_run"] is False
    assert payload["generation"] == 7
    assert payload["checkpoint_read"]["yielded_tensor_count"] == 3
    assert payload["phase_seconds"]["store_write"] >= 0.0
    assert (output_root / "adam" / "adam_state_manifest.json").exists()
    assert (output_root / "gptq" / "gptq_cache_manifest.json").exists()
    assert _fp32_values(
        store.read_param("layers.0.experts.0.w13_weight")
    ) == (1.0, 2.0, 3.0, 4.0)
    assert _fp32_values(
        store.read_param("layers.0.experts.0.w2_weight")
    ) == (5.0, 6.0)


def test_init_qwen35_moe_cli_reads_indexed_checkpoint_dir(tmp_path, capsys) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    output_root = tmp_path / "training_store"
    torch.save(
        {
            "layers.0.mlp.experts.0.gate_proj.weight": torch.tensor([1.0, 2.0]),
            "layers.0.mlp.experts.0.up_proj.weight": torch.tensor([3.0, 4.0]),
        },
        checkpoint_dir / "pytorch_model-00001-of-00002.bin",
    )
    torch.save(
        {
            "layers.0.mlp.experts.0.down_proj.weight": torch.tensor([5.0, 6.0]),
        },
        checkpoint_dir / "pytorch_model-00002-of-00002.bin",
    )
    (checkpoint_dir / "pytorch_model.bin.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layers.0.mlp.experts.0.gate_proj.weight": (
                        "pytorch_model-00001-of-00002.bin"
                    ),
                    "layers.0.mlp.experts.0.up_proj.weight": (
                        "pytorch_model-00001-of-00002.bin"
                    ),
                    "layers.0.mlp.experts.0.down_proj.weight": (
                        "pytorch_model-00002-of-00002.bin"
                    ),
                    "layers.0.self_attn.q_proj.weight": (
                        "missing-attn-shard.bin"
                    ),
                },
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "init-qwen35-moe",
            "--checkpoint",
            str(checkpoint_dir),
            "--root",
            str(output_root),
            "--fp32-shard-bytes",
            "64",
            "--adam-shard-bytes",
            "128",
            "--gptq-shard-bytes",
            "128",
            "--gptq-group-size",
            "2",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    store = FP32ShardStore.load(output_root / "fp32")

    assert exit_code == 0
    assert payload["prefilter_enabled"] is True
    assert payload["skipped_key_count"] == 0
    assert payload["imported_param_count"] == 2
    assert payload["checkpoint_read"]["index_key_count"] == 4
    assert payload["checkpoint_read"]["selected_key_count"] == 3
    assert payload["checkpoint_read"]["filtered_key_count"] == 1
    assert payload["checkpoint_read"]["opened_file_count"] == 2
    assert _fp32_values(
        store.read_param("layers.0.experts.0.w13_weight")
    ) == (1.0, 2.0, 3.0, 4.0)
