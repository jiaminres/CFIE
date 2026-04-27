"""Unit tests for the standalone CFIE training package CLI."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

import cfie_training.runtime.data as runtime_data
from cfie_training.cli.main import main
from cfie_training.predictor import PredictorTrainer
from cfie_training.predictor.trainer import CapturedForwardBatch
from cfie_training.profiles import build_profile_config
from cfie_training.runtime.planner import ExpertRotationScheduler, LayerBucketPlanner
from cfie_training.runtime.types import BatchShape


def _write_config(
    tmp_path: Path,
    *,
    profile_name: str,
    compute_device: str = "cpu",
) -> Path:
    cfg = build_profile_config("qwen35-35b-a3b")
    cfg.profile_name = profile_name
    cfg.execution.compute_device = compute_device
    cfg.validate()
    path = tmp_path / f"{profile_name}.json"
    path.write_text(cfg.to_json(), encoding="utf-8")
    return path


def _expected_window(
    *,
    step_index: int,
    batch: BatchShape,
    next_batch: BatchShape | None = None,
):
    cfg = build_profile_config("qwen35-35b-a3b")
    scheduler = ExpertRotationScheduler(cfg)
    return scheduler.plan_window(
        step_index=step_index,
        batch=batch,
        layer_buckets=LayerBucketPlanner(cfg).build(),
        next_batch=batch if next_batch is None else next_batch,
    )


def _write_text_dataset(
    tmp_path: Path,
    name: str = "train.txt",
) -> Path:
    path = tmp_path / name
    path.write_text(
        "\n".join(
            (
                "CFIE predictor training now requires real dataset-backed batches.",
                "Synthetic trace fallback has been removed from the mainline path.",
                "Forward-captured hidden states supervise future routed experts.",
            )
        ),
        encoding="utf-8",
    )
    return path


class _FakeTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
    ) -> list[int]:
        del add_special_tokens
        pieces = [piece for piece in text.strip().split() if piece]
        if not pieces:
            return [1]
        return [max(1, sum(ord(char) for char in piece) % 997) for piece in pieces]


@pytest.fixture(autouse=True)
def _patch_tokenizer_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runtime_data,
        "_load_tokenizer",
        lambda _: _FakeTokenizer(),
    )


class _FakePredictorBackend:
    def __init__(self, config) -> None:
        self._hidden_dim = config.model_spec.hidden_size
        self._num_layers = config.model_spec.num_hidden_layers
        self._num_experts = config.model_spec.num_experts
        self._executed_experts = (
            config.predictor_routing.executed_experts_per_layer
        )

    def capture_batch(self, batch: BatchShape) -> CapturedForwardBatch:
        token_count = batch.total_tokens
        hidden_template = torch.arange(
            token_count * self._hidden_dim,
            dtype=torch.float32,
        ).reshape(token_count, self._hidden_dim)
        teacher_topk_template = torch.arange(
            token_count * self._executed_experts,
            dtype=torch.long,
        ).reshape(token_count, self._executed_experts) % self._num_experts
        router_logits_template = torch.arange(
            token_count * self._num_experts,
            dtype=torch.float32,
        ).reshape(token_count, self._num_experts)
        return CapturedForwardBatch(
            layer_hidden_states=tuple(
                hidden_template + float(layer_index)
                for layer_index in range(self._num_layers)
            ),
            layer_teacher_topk_ids=tuple(
                (teacher_topk_template + layer_index) % self._num_experts
                for layer_index in range(self._num_layers)
            ),
            layer_teacher_router_logits=tuple(
                router_logits_template + float(layer_index)
                for layer_index in range(self._num_layers)
            ),
        )


class _FakePredictorBatchPlanner:
    def batch_for_step(self, step_index: int) -> BatchShape:
        base = step_index * 6
        return BatchShape(
            samples=2,
            tokens_per_sample=3,
            source_kind="tokenized_dataset",
            dataset_name="fake.txt",
            sample_indices=(step_index, step_index + 1),
            loss_token_count=6,
            token_rows=((base + 1, base + 2, base + 3), (base + 4, base + 5, base + 6)),
            target_rows=((base + 2, base + 3, base + 4), (base + 5, base + 6, base + 7)),
        )


def _install_fake_predictor_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_build_batch_planner(self, **kwargs):
        if kwargs.get("dataset_path") is None:
            raise ValueError("predictor trace generation requires dataset_path")
        return _FakePredictorBatchPlanner()

    def _fake_resolve_teacher_model_backend(self):
        return _FakePredictorBackend(self.config)

    monkeypatch.setattr(
        PredictorTrainer,
        "_build_batch_planner",
        _fake_build_batch_planner,
    )
    monkeypatch.setattr(
        PredictorTrainer,
        "_resolve_teacher_model_backend",
        _fake_resolve_teacher_model_backend,
    )


def test_predictor_trace_command_requires_dataset(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["predictor-trace", "--json"])
    assert excinfo.value.code == 2
    assert "predictor-trace requires --dataset" in capsys.readouterr().err


def test_training_validate_command_reports_success(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["validate"])

    assert code == 0
    assert "configuration is valid" in capsys.readouterr().out


def test_estimate_startup_command_emits_json_estimates(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main([
        "estimate-startup",
        "--samples",
        "1",
        "--tokens-per-sample",
        "128",
        "--gpu-budgets-gb",
        "6,8",
        "--active-expert-candidates",
        "8,16",
        "--max-live-bucket-candidates",
        "1,2",
        "--prefetch-bucket-candidates",
        "0,1",
        "--json",
    ])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["profile_name"] == "qwen35-35b-a3b"
    assert payload["batch"]["samples"] == 1
    assert payload["batch"]["tokens_per_sample"] == 128
    assert payload["gpu_budget_candidates_gb"] == [6.0, 8.0]
    assert payload["active_expert_candidates"] == [8, 16]
    assert len(payload["estimates"]) == 2
    assert payload["estimates"][0]["gpu_hot_budget_gb"] == 6.0
    assert payload["estimates"][1]["gpu_hot_budget_gb"] == 8.0
    assert 0.0 < payload["estimates"][0]["gpu_fill_ratio"] <= 1.0
    assert payload["estimates"][1]["planned_gpu_hot_bytes"] >= (
        payload["estimates"][0]["planned_gpu_hot_bytes"]
    )


def test_predictor_train_command_requires_trace_or_dataset(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["predictor-train", "--json"])
    assert excinfo.value.code == 2
    assert (
        "predictor-train requires --trace-input or --dataset"
        in capsys.readouterr().err
    )


def test_predictor_trace_command_emits_json_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fake_predictor_runtime(monkeypatch)
    dataset_path = _write_text_dataset(tmp_path, "predictor-trace.txt")

    code = main([
        "predictor-trace",
        "--steps",
        "2",
        "--examples-per-step",
        "2",
        "--dataset",
        str(dataset_path),
        "--json",
    ])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["profile_name"] == "qwen35-35b-a3b"
    assert payload["example_count"] == 4
    assert payload["window_layers"] == 8
    assert payload["candidate_experts_per_layer"] == 40
    assert payload["executed_experts_per_layer"] == 8
    assert len(payload["examples"]) == 4
    assert len(payload["examples"][0]["future_layer_indices"]) == 8
    assert len(payload["examples"][0]["future_teacher_topk_ids"][0]) == 8
    assert len(payload["examples"][0]["future_teacher_router_logits"][0]) == 256


def test_predictor_train_command_emits_json_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fake_predictor_runtime(monkeypatch)
    dataset_path = _write_text_dataset(tmp_path, "predictor-train.txt")

    code = main([
        "predictor-train",
        "--steps",
        "2",
        "--examples-per-step",
        "2",
        "--epochs",
        "2",
        "--dataset",
        str(dataset_path),
        "--json",
    ])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["profile_name"] == "qwen35-35b-a3b"
    assert payload["example_count"] == 4
    assert payload["epochs"] == 2
    assert payload["candidate_experts_per_layer"] == 40
    assert payload["executed_experts_per_layer"] == 8
    assert len(payload["epoch_summaries"]) == 2
    assert 0.0 <= payload["final_recall_at_candidate_budget"] <= 1.0


def test_predictor_trace_command_writes_output_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fake_predictor_runtime(monkeypatch)
    dataset_path = _write_text_dataset(tmp_path, "predictor-trace-output.txt")
    output_path = tmp_path / "predictor_trace.json"

    code = main([
        "predictor-trace",
        "--steps",
        "1",
        "--examples-per-step",
        "1",
        "--dataset",
        str(dataset_path),
        "--output",
        str(output_path),
        "--flush-every-steps",
        "1",
        "--json",
    ])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    saved_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert output_path.exists()
    assert payload == saved_payload
    assert payload["example_count"] == 1


def test_predictor_train_command_accepts_trace_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fake_predictor_runtime(monkeypatch)
    dataset_path = _write_text_dataset(tmp_path, "predictor-trace-input.txt")
    trace_path = tmp_path / "predictor_trace.json"

    trace_code = main([
        "predictor-trace",
        "--steps",
        "1",
        "--examples-per-step",
        "1",
        "--dataset",
        str(dataset_path),
        "--output",
        str(trace_path),
        "--json",
    ])

    assert trace_code == 0
    capsys.readouterr()

    code = main([
        "predictor-train",
        "--trace-input",
        str(trace_path),
        "--epochs",
        "1",
        "--json",
    ])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["example_count"] == 1
    assert payload["epochs"] == 1


def test_predictor_train_command_writes_checkpoint_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fake_predictor_runtime(monkeypatch)
    dataset_path = _write_text_dataset(tmp_path, "predictor-checkpoint.txt")
    checkpoint_path = tmp_path / "predictor.ckpt"

    code = main([
        "predictor-train",
        "--steps",
        "1",
        "--examples-per-step",
        "1",
        "--epochs",
        "1",
        "--dataset",
        str(dataset_path),
        "--checkpoint-output",
        str(checkpoint_path),
        "--json",
    ])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert checkpoint_path.exists()
    assert payload["epochs"] == 1

    trainer = PredictorTrainer(build_profile_config("qwen35-35b-a3b"))
    _, metadata = trainer.load_checkpoint(checkpoint_path)
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")

    assert metadata.checkpoint_kind == "cfie_predictor_checkpoint"
    assert metadata.epochs == 1
    assert isinstance(checkpoint_payload["optimizer_state_dict"], dict)
    assert checkpoint_payload["run_trace"]["epochs"] == 1
    assert checkpoint_payload["metadata"]["window_layers"] == 8
    assert checkpoint_payload["metadata"]["candidate_experts_per_layer"] == 40


def test_predictor_train_command_accepts_resume_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fake_predictor_runtime(monkeypatch)
    dataset_path = _write_text_dataset(tmp_path, "predictor-resume.txt")
    checkpoint_path = tmp_path / "predictor.ckpt"

    first_code = main([
        "predictor-train",
        "--steps",
        "1",
        "--examples-per-step",
        "1",
        "--epochs",
        "1",
        "--dataset",
        str(dataset_path),
        "--checkpoint-output",
        str(checkpoint_path),
        "--json",
    ])

    assert first_code == 0
    capsys.readouterr()

    code = main([
        "predictor-train",
        "--resume-checkpoint",
        str(checkpoint_path),
        "--steps",
        "1",
        "--examples-per-step",
        "1",
        "--epochs",
        "1",
        "--dataset",
        str(dataset_path),
        "--json",
    ])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["epochs"] == 2
    assert len(payload["epoch_summaries"]) == 2
    assert payload["epoch_summaries"][0]["epoch_index"] == 0
    assert payload["epoch_summaries"][1]["epoch_index"] == 1


def test_predictor_train_command_accepts_init_checkpoint_warm_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fake_predictor_runtime(monkeypatch)
    dataset_path = _write_text_dataset(tmp_path, "predictor-init.txt")
    checkpoint_path = tmp_path / "predictor.ckpt"

    first_code = main([
        "predictor-train",
        "--steps",
        "1",
        "--examples-per-step",
        "1",
        "--epochs",
        "1",
        "--dataset",
        str(dataset_path),
        "--checkpoint-output",
        str(checkpoint_path),
        "--json",
    ])

    assert first_code == 0
    capsys.readouterr()

    code = main([
        "predictor-train",
        "--init-from-checkpoint",
        str(checkpoint_path),
        "--steps",
        "1",
        "--examples-per-step",
        "1",
        "--epochs",
        "1",
        "--dataset",
        str(dataset_path),
        "--json",
    ])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["epochs"] == 1
    assert len(payload["epoch_summaries"]) == 1
    assert payload["epoch_summaries"][0]["epoch_index"] == 0


def test_predictor_inspect_command_emits_checkpoint_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fake_predictor_runtime(monkeypatch)
    dataset_path = _write_text_dataset(tmp_path, "predictor-inspect.txt")
    checkpoint_path = tmp_path / "predictor.ckpt"

    train_code = main([
        "predictor-train",
        "--steps",
        "1",
        "--examples-per-step",
        "1",
        "--epochs",
        "1",
        "--dataset",
        str(dataset_path),
        "--checkpoint-output",
        str(checkpoint_path),
        "--json",
    ])

    assert train_code == 0
    capsys.readouterr()

    code = main([
        "predictor-inspect",
        "--checkpoint",
        str(checkpoint_path),
        "--json",
    ])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["checkpoint_kind"] == "cfie_predictor_checkpoint"
    assert payload["profile_name"] == "qwen35-35b-a3b"
    assert payload["window_layers"] == 8
    assert payload["candidate_experts_per_layer"] == 40


def test_predictor_eval_command_emits_json_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fake_predictor_runtime(monkeypatch)
    dataset_path = _write_text_dataset(tmp_path, "predictor-eval.txt")
    checkpoint_path = tmp_path / "predictor.ckpt"
    trace_path = tmp_path / "predictor_trace.json"

    train_code = main([
        "predictor-train",
        "--steps",
        "1",
        "--examples-per-step",
        "1",
        "--epochs",
        "1",
        "--dataset",
        str(dataset_path),
        "--checkpoint-output",
        str(checkpoint_path),
        "--json",
    ])

    assert train_code == 0
    capsys.readouterr()

    trace_code = main([
        "predictor-trace",
        "--steps",
        "1",
        "--examples-per-step",
        "1",
        "--dataset",
        str(dataset_path),
        "--output",
        str(trace_path),
        "--json",
    ])

    assert trace_code == 0
    capsys.readouterr()

    code = main([
        "predictor-eval",
        "--checkpoint",
        str(checkpoint_path),
        "--trace-input",
        str(trace_path),
        "--json",
    ])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["profile_name"] == "qwen35-35b-a3b"
    assert payload["example_count"] == 1
    assert payload["mean_loss"] >= 0.0
    assert 0.0 <= payload["recall_at_candidate_budget"] <= 1.0
    assert 0.0 <= payload["recall_at_executed_budget"] <= 1.0
    assert payload["checkpoint_metadata"]["checkpoint_kind"] == "cfie_predictor_checkpoint"


def test_predictor_train_command_accepts_dataset_backed_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fake_predictor_runtime(monkeypatch)
    dataset_path = _write_text_dataset(tmp_path, "predictor_train.txt")
    checkpoint_path = tmp_path / "predictor.ckpt"

    code = main([
        "predictor-train",
        "--steps",
        "2",
        "--examples-per-step",
        "1",
        "--epochs",
        "1",
        "--samples",
        "2",
        "--tokens-per-sample",
        "32",
        "--dataset",
        str(dataset_path),
        "--checkpoint-output",
        str(checkpoint_path),
        "--json",
    ])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert checkpoint_path.exists()
    assert payload["profile_name"] == "qwen35-35b-a3b"
    assert payload["example_count"] == 2
    assert payload["epochs"] == 1
    assert payload["final_mean_loss"] >= 0.0
    assert 0.0 <= payload["final_recall_at_candidate_budget"] <= 1.0


def test_predictor_eval_command_accepts_dataset_backed_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fake_predictor_runtime(monkeypatch)
    dataset_path = _write_text_dataset(tmp_path, "predictor_eval.txt")
    checkpoint_path = tmp_path / "predictor.ckpt"

    train_code = main([
        "predictor-train",
        "--steps",
        "2",
        "--examples-per-step",
        "1",
        "--epochs",
        "1",
        "--samples",
        "2",
        "--tokens-per-sample",
        "32",
        "--dataset",
        str(dataset_path),
        "--checkpoint-output",
        str(checkpoint_path),
        "--json",
    ])

    assert train_code == 0
    capsys.readouterr()

    code = main([
        "predictor-eval",
        "--checkpoint",
        str(checkpoint_path),
        "--steps",
        "2",
        "--examples-per-step",
        "1",
        "--samples",
        "2",
        "--tokens-per-sample",
        "32",
        "--dataset",
        str(dataset_path),
        "--json",
    ])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["profile_name"] == "qwen35-35b-a3b"
    assert payload["example_count"] == 2
    assert payload["mean_loss"] >= 0.0
    assert 0.0 <= payload["recall_at_candidate_budget"] <= 1.0
    assert 0.0 <= payload["recall_at_executed_budget"] <= 1.0
    assert payload["checkpoint_metadata"]["checkpoint_kind"] == "cfie_predictor_checkpoint"


def test_predictor_eval_command_requires_trace_or_dataset(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["predictor-eval", "--checkpoint", "predictor.ckpt", "--json"])
    assert excinfo.value.code == 2
    assert (
        "predictor-eval requires --trace-input or --dataset"
        in capsys.readouterr().err
    )


def test_train_command_requires_dataset_without_resume(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["train", "--steps", "1", "--json"])
    assert excinfo.value.code == 2
    assert (
        "train requires --dataset unless --resume-from points to a "
        "dataset-backed session checkpoint"
        in capsys.readouterr().err
    )


def test_simulate_command_emits_runtime_trace(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main([
        "simulate",
        "--steps",
        "2",
        "--samples",
        "2",
        "--tokens-per-sample",
        "256",
        "--json",
    ])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["profile_name"] == "qwen35-35b-a3b"
    assert payload["step_count"] == 2
    assert payload["resource_plan"]["all_tiers_within_budget"] is True
    assert payload["resource_plan"]["cpu_optimizer_state_storage_dtype"] == "fp8_e4m3fn"
    assert payload["resource_plan"]["host_gradient_buffer_storage_dtype"] == "fp8_e4m3fn"
    assert payload["resource_plan"]["host_gradient_buffer_scope"] == "current_bucket_only"
    assert (
        payload["resource_plan"]["full_model_gradient_buffer_bytes"]
        > payload["resource_plan"]["host_gradient_buffer_bytes"]
    )
    assert payload["resource_plan"]["transport_staged_file_cache_bytes"] > 0
    assert payload["resource_plan"]["weight_stage_buffer_bytes"] > 0
    assert payload["resource_plan"]["transfer_staging_buffer_bytes"] == (
        payload["resource_plan"]["host_gradient_buffer_bytes"]
        + payload["resource_plan"]["weight_stage_buffer_bytes"]
    )
    assert payload["resource_plan"]["transfer_overlap_enabled"] is True
    assert payload["steps"][0]["actions"][0]["name"] == "stage_static_modules"
    assert payload["steps"][0]["residency_transitions"][0]["group_id"] == "static_modules"
    assert len(payload["steps"][0]["scheduled_micro_batches"]) == 2
    assert len(payload["steps"][0]["bucket_stream_traces"]) == 10
    assert len(payload["steps"][0]["stream_operations"]) == 40
    assert payload["steps"][1]["actions"][0]["name"] == "prefetch_routed_experts"
    expected_step0_window = _expected_window(
        step_index=0,
        batch=BatchShape(samples=2, tokens_per_sample=256),
        next_batch=BatchShape(samples=2, tokens_per_sample=256),
    )
    expected_step1_window = _expected_window(
        step_index=1,
        batch=BatchShape(samples=2, tokens_per_sample=256),
        next_batch=BatchShape(samples=2, tokens_per_sample=256),
    )
    assert (
        payload["steps"][0]["expert_window_plan"]["selection_strategy"]
        == expected_step0_window.selection_strategy
    )
    assert payload["steps"][0]["expert_window_plan"]["active_expert_ids"] == list(
        expected_step0_window.active_expert_ids
    )
    assert payload["steps"][1]["active_expert_ids"] == list(
        expected_step1_window.active_expert_ids
    )
    assert payload["steps"][1]["prefetched_expert_ids"] == list(
        expected_step1_window.prefetched_expert_ids
    )
    assert payload["steps"][1]["residency_ending_states"]["expert_window:2"] == "cpu_staged"
    assert payload["steps"][0]["warehouse_summary"]["total_shards"] == 22
    assert payload["steps"][1]["warehouse_summary"]["cpu_staged"] > 0
    step0_parameter_store = payload["steps"][0]["parameter_store_summary"]
    step1_parameter_store = payload["steps"][1]["parameter_store_summary"]
    step0_parameter_source = payload["steps"][0]["parameter_source_summary"]
    step0_parameter_prefetch = payload["steps"][0]["parameter_prefetch_summary"]
    step0_transport = payload["steps"][0]["transport_summary"]
    step0_transport_exec = payload["steps"][0]["transport_execution_summary"]
    assert step0_parameter_store["tracked_shards"] == 20
    assert step1_parameter_store["tracked_shards"] == 30
    assert step0_parameter_source["touched_shards"] == 20
    assert step0_parameter_prefetch["touched_shards"] == 20
    assert payload["steps"][0]["parameter_prefetch_summary"]["cpu_hot_reuses"] == 0
    assert payload["steps"][0]["parameter_load_summary"]["touched_shards"] == 20
    assert payload["steps"][0]["parameter_load_summary"]["transport_cache_loads"] == 0
    assert payload["steps"][0]["parameter_load_summary"]["direct_manifest_loads"] == 0
    assert payload["steps"][0]["parameter_load_summary"]["buffer_reuses"] == 0
    assert payload["steps"][0]["parameter_load_summary"]["cpu_hot_reuses"] == 20
    assert (
        step0_transport["matched_shards"] + step0_transport["unmatched_shards"]
        == payload["steps"][0]["warehouse_summary"]["total_shards"]
    )
    assert step0_transport_exec["manifest_available"] == step0_transport["manifest_available"]
    assert (
        step0_transport_exec["requested_file_count"]
        == step0_transport["file_count"]
    )
    assert step0_transport_exec["weight_stage_buffer_bytes"] > 0
    assert step0_transport_exec["gradient_stage_buffer_bytes"] > 0
    assert step0_transport_exec["h2d_transfer_bytes"] > 0
    assert step0_transport_exec["d2h_transfer_bytes"] > 0
    assert step0_transport_exec["released_buffer_count"] > 0
    assert step0_transport_exec["active_buffer_count"] == 0
    assert step0_transport_exec["pooled_buffer_count"] == 40
    if step0_transport["manifest_available"]:
        assert step0_parameter_store["manifest_backed_shards"] > 0
        assert step0_parameter_store["synthetic_seeded_shards"] == 0
        assert step0_parameter_store["transport_backed_shards"] > 0
        assert step1_parameter_store["manifest_backed_shards"] > 0
        assert step1_parameter_store["synthetic_seeded_shards"] == 0
        assert step1_parameter_store["transport_backed_shards"] > 0
        assert step0_parameter_source["manifest_backed_shards"] > 0
        assert step0_parameter_source["synthetic_seeded_shards"] == 0
        assert step0_parameter_source["transport_backed_shards"] > 0
        assert step0_parameter_prefetch["transport_cache_prefetches"] > 0
        assert step0_parameter_prefetch["direct_manifest_prefetches"] > 0
        assert step0_transport["matched_shards"] > 0
        assert step0_transport["file_count"] > 0
        assert step0_transport_exec["staged_file_count"] > 0
        assert step0_transport_exec["cache_resident_bytes"] > 0
    else:
        assert step0_parameter_store["manifest_backed_shards"] == 0
        assert step0_parameter_store["synthetic_seeded_shards"] == 20
        assert step0_parameter_store["transport_backed_shards"] == 0
        assert step1_parameter_store["manifest_backed_shards"] == 0
        assert step1_parameter_store["synthetic_seeded_shards"] == 30
        assert step1_parameter_store["transport_backed_shards"] == 0
        assert step0_parameter_source["manifest_backed_shards"] == 0
        assert step0_parameter_source["synthetic_seeded_shards"] == 20
        assert step0_parameter_source["transport_backed_shards"] == 0
        assert step0_parameter_prefetch["transport_cache_prefetches"] == 0
        assert step0_parameter_prefetch["direct_manifest_prefetches"] == 0
        assert step0_transport["matched_shards"] == 0
        assert step0_transport["file_count"] == 0
        assert step0_transport_exec["staged_file_count"] == 0
        assert step0_transport_exec["cache_resident_bytes"] == 0
    assert payload["steps"][0]["execution_summary"]["executed_buckets"] == 10
    assert payload["steps"][0]["execution_summary"]["gradient_shards"] == 40
    assert payload["steps"][0]["execution_summary"]["total_loss"] > 0
    assert payload["steps"][0]["execution_summary"]["peak_host_gradient_buffer_bytes"] > 0
    assert (
        payload["steps"][0]["execution_summary"]["gradient_buffer_storage_dtype"]
        == "fp8_e4m3fn"
    )
    assert payload["steps"][0]["stream_overlap_summary"]["micro_batch_count"] == 2
    assert payload["steps"][0]["stream_overlap_summary"]["compute_operation_count"] == 20
    assert payload["steps"][0]["stream_overlap_summary"]["transfer_operation_count"] == 20
    assert payload["steps"][0]["bucket_stream_traces"][0]["cpu_hot_shards_before_prefetch"] == 0
    assert payload["steps"][0]["bucket_stream_traces"][0]["lookahead_prefetched_bucket_ids"] == [1]
    assert payload["steps"][0]["bucket_stream_traces"][0]["prefetch_summary"]["touched_shards"] == 2
    assert payload["steps"][0]["bucket_stream_traces"][0]["load_summary"]["cpu_hot_reuses"] == 2
    assert payload["steps"][0]["bucket_stream_traces"][0]["micro_batch_count"] == 2
    bucket_record = payload["steps"][0]["bucket_stream_traces"][0]["bucket_record"]
    if bucket_record["semantic_layout_used"]:
        assert bucket_record["execution_mode"] == "structured_qwen35_bucket"
        assert "linear_attn_in_proj_qkv" in bucket_record["semantic_roles"]
        assert "self_attn_q_proj" in bucket_record["semantic_roles"]
        assert "mlp_router_gate" in bucket_record["semantic_roles"]
        assert "shared_expert_up_proj" in bucket_record["semantic_roles"]
    else:
        assert bucket_record["execution_mode"] == "synthetic_representative_bucket"
        assert bucket_record["semantic_roles"] == []
    assert payload["steps"][0]["bucket_stream_traces"][0]["optimizer_update_count"] == 4
    assert payload["steps"][0]["bucket_stream_traces"][0]["gradient_release_count"] == 4
    assert (
        payload["steps"][0]["bucket_stream_traces"][0]["gradients_released_immediately"]
        is True
    )
    assert payload["steps"][0]["bucket_stream_traces"][0]["host_gradient_buffer_bytes"] > 0
    assert (
        payload["steps"][0]["bucket_stream_traces"][0]["host_gradient_buffer_storage_dtype"]
        == "fp8_e4m3fn"
    )
    assert payload["steps"][0]["bucket_stream_traces"][0]["offloaded_shards_after_update"] == 1
    assert payload["steps"][0]["bucket_stream_traces"][1]["cpu_hot_shards_before_prefetch"] == 2
    assert payload["steps"][0]["stream_operations"][0]["operation"] == "bucket_compute"
    assert payload["steps"][0]["stream_operations"][1]["operation"] == "bucket_update_release"
    assert len(payload["steps"][0]["optimizer_updates"]) == 40
    assert payload["steps"][0]["optimizer_updates"][0]["representative_params"] == 128
    assert payload["steps"][0]["optimizer_updates"][0]["gradient_l2_norm"] > 0
    assert payload["steps"][0]["optimizer_summary"]["tracked_shards"] == 20
    assert payload["steps"][0]["optimizer_summary"]["state_storage_dtype"] == "fp8_e4m3fn"
    assert (
        payload["steps"][0]["optimizer_summary"]["gradient_buffer_storage_dtype"]
        == "fp8_e4m3fn"
    )
    assert (
        payload["steps"][0]["optimizer_summary"]["gradient_buffer_scope"]
        == "current_bucket_only"
    )
    assert payload["steps"][0]["optimizer_summary"]["last_bucket_staged_gradient_bytes"] > 0
    assert payload["steps"][1]["optimizer_summary"]["cumulative_updates_applied"] == 80


def test_simulate_command_accepts_config_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_config(
        tmp_path,
        profile_name="simulate-config-smoke",
    )

    code = main([
        "simulate",
        "--config",
        str(config_path),
        "--steps",
        "1",
        "--samples",
        "2",
        "--tokens-per-sample",
        "64",
        "--json",
    ])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["profile_name"] == "simulate-config-smoke"
    assert payload["step_count"] == 1


def test_simulate_command_can_resume_from_saved_snapshot(
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    snapshot_path = tmp_path / "runtime_snapshot.json"

    code = main([
        "simulate",
        "--steps",
        "1",
        "--samples",
        "2",
        "--tokens-per-sample",
        "128",
        "--json",
        "--save-snapshot",
        str(snapshot_path),
    ])

    assert code == 0
    first_payload = json.loads(capsys.readouterr().out)
    assert snapshot_path.exists()
    snapshot_payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert snapshot_payload["next_step_index"] == 1
    assert snapshot_payload["parameter_store_shards"][0]["representative_params"] == 128
    assert len(snapshot_payload["parameter_store_shards"][0]["parameter_values"]) == 128
    assert len(snapshot_payload["transport_buffers"]) > 0
    assert first_payload["steps"][0]["step_index"] == 0

    code = main([
        "simulate",
        "--steps",
        "1",
        "--samples",
        "2",
        "--tokens-per-sample",
        "128",
        "--json",
        "--resume-from",
        str(snapshot_path),
    ])

    assert code == 0
    resumed_payload = json.loads(capsys.readouterr().out)
    expected_resumed_window = _expected_window(
        step_index=1,
        batch=BatchShape(samples=2, tokens_per_sample=128),
        next_batch=BatchShape(samples=2, tokens_per_sample=128),
    )
    assert resumed_payload["steps"][0]["step_index"] == 1
    assert resumed_payload["steps"][0]["active_expert_ids"] == list(
        expected_resumed_window.active_expert_ids
    )
    assert resumed_payload["steps"][0]["expert_window_plan"]["prefetched_expert_ids"] == list(
        expected_resumed_window.prefetched_expert_ids
    )
    assert resumed_payload["steps"][0]["optimizer_summary"]["tracked_shards"] == 30
    resumed_transport = resumed_payload["steps"][0]["transport_summary"]
    resumed_transport_exec = resumed_payload["steps"][0]["transport_execution_summary"]
    resumed_parameter_source = resumed_payload["steps"][0]["parameter_source_summary"]
    assert resumed_transport["matched_shards"] + resumed_transport["unmatched_shards"] > 0
    assert (
        resumed_transport_exec["requested_file_count"]
        == resumed_transport["file_count"]
    )
    assert resumed_transport_exec["weight_stage_buffer_bytes"] > 0
    assert resumed_transport_exec["gradient_stage_buffer_bytes"] > 0
    assert resumed_transport_exec["released_buffer_count"] > 0
    assert (
        resumed_payload["steps"][0]["parameter_prefetch_summary"]["buffer_reuses"]
        + resumed_payload["steps"][0]["parameter_prefetch_summary"]["nvme_fp32_mirror_prefetches"]
        > 0
    )
    assert resumed_payload["steps"][0]["parameter_load_summary"]["buffer_reuses"] == 0
    assert resumed_payload["steps"][0]["parameter_load_summary"]["cpu_hot_reuses"] == 20
    if resumed_transport["manifest_available"]:
        assert resumed_transport["matched_shards"] > 0
        assert resumed_transport_exec["cache_file_count"] > 0
        assert resumed_parameter_source["manifest_backed_shards"] > 0
        assert resumed_parameter_source["synthetic_seeded_shards"] == 0
        assert resumed_parameter_source["transport_backed_shards"] > 0
    else:
        assert resumed_transport["matched_shards"] == 0
        assert resumed_transport_exec["cache_file_count"] == 0
        assert resumed_parameter_source["manifest_backed_shards"] == 0
        assert resumed_parameter_source["synthetic_seeded_shards"] == 20
        assert resumed_parameter_source["transport_backed_shards"] == 0


def test_train_command_emits_session_trace_and_checkpoints(
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_path = _write_text_dataset(tmp_path, "train-checkpoint.txt")

    code = main([
        "train",
        "--steps",
        "3",
        "--samples",
        "2",
        "--tokens-per-sample",
        "128",
        "--dataset",
        str(dataset_path),
        "--checkpoint-dir",
        str(tmp_path),
        "--checkpoint-interval",
        "2",
        "--json",
    ])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["total_steps"] == 3
    assert payload["average_loss"] > 0
    assert payload["peak_activation_bytes"] > 0
    assert payload["checkpoint_format"] == "training_session_checkpoint"
    assert len(payload["checkpoint_paths"]) == 1
    assert payload["checkpoint_paths"][0].endswith("step_00002.json")
    checkpoint_payload = json.loads(
        (tmp_path / "step_00002.json").read_text(encoding="utf-8")
    )
    assert checkpoint_payload["checkpoint_kind"] == "training_session_checkpoint"
    assert checkpoint_payload["planner"]["planner_kind"] == "tokenized_dataset"
    assert checkpoint_payload["planner"]["dataset_path"] == str(dataset_path)
    assert checkpoint_payload["runtime_snapshot"]["next_step_index"] == 2
    assert isinstance(checkpoint_payload["runtime_snapshot"]["transport_cached_files"], list)
    assert len(checkpoint_payload["runtime_snapshot"]["transport_buffers"]) > 0


def test_train_command_accepts_config_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_config(
        tmp_path,
        profile_name="train-config-smoke",
    )
    dataset_path = _write_text_dataset(tmp_path, "train-config-smoke.txt")

    code = main([
        "train",
        "--config",
        str(config_path),
        "--steps",
        "1",
        "--samples",
        "1",
        "--tokens-per-sample",
        "32",
        "--dataset",
        str(dataset_path),
        "--json",
    ])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["profile_name"] == "train-config-smoke"
    assert payload["total_steps"] == 1
    assert payload["average_loss"] > 0


def test_train_command_plain_output_reports_compute_device(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_config(
        tmp_path,
        profile_name="train-config-plain",
    )
    dataset_path = _write_text_dataset(tmp_path, "train-config-plain.txt")

    code = main([
        "train",
        "--config",
        str(config_path),
        "--steps",
        "1",
        "--samples",
        "1",
        "--tokens-per-sample",
        "16",
        "--dataset",
        str(dataset_path),
    ])

    assert code == 0
    output = capsys.readouterr().out
    assert "train-config-plain" in output
    assert "compute_device=cpu" in output


def test_train_command_can_use_dataset_backed_batches(
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_path = tmp_path / "train.txt"
    dataset_path.write_text(
        "\n".join(
            (
                "Client-side MoE training prioritizes memory savings.",
                "Current bucket gradients are staged on the host and released.",
                "Experts are prefetched on demand instead of kept hot.",
            )
        ),
        encoding="utf-8",
    )

    code = main([
        "train",
        "--steps",
        "2",
        "--samples",
        "2",
        "--tokens-per-sample",
        "32",
        "--dataset",
        str(dataset_path),
        "--json",
    ])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["total_steps"] == 2
    assert payload["average_loss"] > 0
    assert payload["steps"][0]["batch"]["source_kind"] == "tokenized_dataset"
    assert payload["steps"][0]["batch"]["dataset_name"] == "train.txt"
    assert payload["steps"][0]["batch"]["sample_indices"] == [0, 1]
    assert payload["steps"][0]["batch"]["loss_token_count"] == 64
    assert payload["steps"][1]["batch"]["sample_indices"] == [2, 0]


def test_train_command_can_resume_dataset_checkpoint_without_dataset_flags(
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_path = tmp_path / "train.txt"
    dataset_path.write_text(
        "\n".join(
            (
                "Bucket-local overlap keeps host buffers bounded.",
                "Routed experts are prefetched and released incrementally.",
                "CPU optimizer state stays compressed between updates.",
            )
        ),
        encoding="utf-8",
    )

    code = main([
        "train",
        "--steps",
        "2",
        "--samples",
        "2",
        "--tokens-per-sample",
        "32",
        "--dataset",
        str(dataset_path),
        "--checkpoint-dir",
        str(tmp_path),
        "--checkpoint-interval",
        "2",
        "--json",
    ])

    assert code == 0
    _ = json.loads(capsys.readouterr().out)
    checkpoint_path = tmp_path / "step_00002.json"
    assert checkpoint_path.exists()

    code = main([
        "train",
        "--steps",
        "1",
        "--samples",
        "2",
        "--tokens-per-sample",
        "32",
        "--resume-from",
        str(checkpoint_path),
        "--json",
    ])

    assert code == 0
    resumed_payload = json.loads(capsys.readouterr().out)
    assert resumed_payload["total_steps"] == 1
    assert resumed_payload["checkpoint_format"] == "training_session_checkpoint"
    assert resumed_payload["steps"][0]["step_index"] == 2
    assert resumed_payload["steps"][0]["batch"]["source_kind"] == "tokenized_dataset"
    assert resumed_payload["steps"][0]["batch"]["dataset_name"] == "train.txt"
    assert resumed_payload["steps"][0]["batch"]["sample_indices"] == [1, 2]


def test_module_entrypoint_runs_cli_from_python_m() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    env["PYTHONPATH"] = "CFIE"

    result = subprocess.run(
        [sys.executable, "-m", "cfie_training.cli.main", "validate"],
        cwd=repo_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "configuration is valid" in result.stdout
