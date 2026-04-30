"""Unit tests for the predictor-only CFIE training CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import cfie_training.runtime.data as runtime_data
from cfie_training.cli.main import main
from cfie_training.predictor import (
    PredictorEpochSummary,
    PredictorTraceDataset,
    PredictorTraceExample,
    PredictorTrainer,
)
from cfie_training.predictor.models import PredictorTrainingRunTrace
from cfie_training.predictor.trainer import CapturedForwardBatch
from cfie_training.profiles import build_profile_config
from cfie_training.runtime.types import BatchShape


def _write_text_dataset(
    tmp_path: Path,
    name: str = "train.txt",
) -> Path:
    path = tmp_path / name
    path.write_text(
        "\n".join(
            (
                "CFIE predictor training now requires real dataset-backed batches.",
                "Forward-captured hidden states supervise future routed experts.",
                "Predictor checkpoints are the only training artifact kept.",
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
    monkeypatch.setattr(runtime_data, "_load_tokenizer", lambda _: _FakeTokenizer())


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


def _write_standard_trace_json(tmp_path: Path) -> Path:
    config = build_profile_config("qwen35-35b-a3b")
    hidden_size = config.model_spec.hidden_size
    executed = config.predictor_routing.executed_experts_per_layer
    window = config.predictor_routing.window_layers
    examples = []
    for example_index in range(2):
        future_topk = []
        future_logits = []
        for layer_offset in range(window):
            start = (example_index + layer_offset) % config.model_spec.num_experts
            future_topk.append(
                tuple(
                    (start + offset) % config.model_spec.num_experts
                    for offset in range(executed)
                )
            )
            future_logits.append(
                tuple(
                    float((expert_index + layer_offset) % 11) / 5.0
                    for expert_index in range(config.model_spec.num_experts)
                )
            )
        examples.append(
            PredictorTraceExample(
                example_index=example_index,
                step_index=0,
                token_index=example_index,
                insertion_layer_index=4,
                future_layer_indices=tuple(range(5, 5 + window)),
                hidden_state=tuple(
                    float(example_index) + (hidden_index % 9) * 0.01
                    for hidden_index in range(hidden_size)
                ),
                future_teacher_topk_ids=tuple(future_topk),
                future_teacher_router_logits=tuple(future_logits),
            )
        )
    dataset = PredictorTraceDataset(
        profile_name=config.profile_name,
        example_count=len(examples),
        window_layers=window,
        candidate_experts_per_layer=config.predictor_routing.candidate_experts_per_layer,
        executed_experts_per_layer=executed,
        examples=tuple(examples),
    )
    path = tmp_path / "standard_trace.json"
    path.write_text(dataset.to_json(), encoding="utf-8")
    return path


def _fake_run_trace(
    trainer: PredictorTrainer,
    *,
    example_count: int,
    epochs: int,
) -> PredictorTrainingRunTrace:
    return PredictorTrainingRunTrace(
        profile_name=trainer.config.profile_name,
        example_count=example_count,
        epochs=epochs,
        candidate_experts_per_layer=(
            trainer.config.predictor_routing.candidate_experts_per_layer
        ),
        executed_experts_per_layer=(
            trainer.config.predictor_routing.executed_experts_per_layer
        ),
        epoch_summaries=(
            PredictorEpochSummary(
                epoch_index=max(0, epochs - 1),
                mean_loss=0.25,
                recall_at_candidate_budget=0.5,
                recall_at_executed_budget=0.25,
            ),
        ),
    )


def test_predictor_trace_command_requires_dataset(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["predictor-trace", "--json"])
    assert excinfo.value.code == 2
    assert "predictor-trace requires --dataset" in capsys.readouterr().err


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
    assert len(payload["examples"]) == 4


def test_predictor_trace_command_writes_output_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fake_predictor_runtime(monkeypatch)
    dataset_path = _write_text_dataset(tmp_path, "predictor-trace.txt")
    output_path = tmp_path / "trace.json"

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
        "--json",
    ])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert output_path.is_file()
    assert payload["example_count"] == 1


def test_predictor_train_command_uses_trace_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fake_predictor_runtime(monkeypatch)
    dataset_path = _write_text_dataset(tmp_path, "predictor-trace.txt")
    trace_path = tmp_path / "trace.json"
    checkpoint_path = tmp_path / "predictor.ckpt"

    code = main([
        "predictor-trace",
        "--steps",
        "1",
        "--examples-per-step",
        "1",
        "--dataset",
        str(dataset_path),
        "--output",
        str(trace_path),
    ])
    assert code == 0

    def _fake_fit_trace_file(self, **kwargs):
        output_path = kwargs.get("checkpoint_output_path")
        if output_path is not None:
            Path(output_path).write_text("checkpoint", encoding="utf-8")
        return object(), _fake_run_trace(self, example_count=1, epochs=2), {}

    monkeypatch.setattr(PredictorTrainer, "fit_trace_file", _fake_fit_trace_file)

    code = main([
        "predictor-train",
        "--trace-input",
        str(trace_path),
        "--epochs",
        "2",
        "--checkpoint-output",
        str(checkpoint_path),
        "--json",
    ])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert checkpoint_path.is_file()
    assert payload["epochs"] == 2
    assert payload["example_count"] == 1


def test_predictor_train_command_builds_dataset_inline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fake_predictor_runtime(monkeypatch)
    dataset_path = _write_text_dataset(tmp_path)
    checkpoint_path = tmp_path / "predictor_inline.ckpt"

    def _fake_fit_dataset(self, dataset, **kwargs):
        output_path = kwargs.get("checkpoint_output_path")
        if output_path is not None:
            Path(output_path).write_text("checkpoint", encoding="utf-8")
        return object(), _fake_run_trace(self, example_count=dataset.example_count, epochs=1), {}

    monkeypatch.setattr(PredictorTrainer, "fit_dataset", _fake_fit_dataset)

    code = main([
        "predictor-train",
        "--steps",
        "2",
        "--examples-per-step",
        "2",
        "--dataset",
        str(dataset_path),
        "--epochs",
        "1",
        "--checkpoint-output",
        str(checkpoint_path),
        "--json",
    ])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert checkpoint_path.is_file()
    assert payload["epochs"] == 1
    assert payload["example_count"] == 4


def test_predictor_train_command_accepts_standard_json_trace_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace_path = _write_standard_trace_json(tmp_path)
    checkpoint_path = tmp_path / "predictor_standard.ckpt"

    code = main([
        "predictor-train",
        "--trace-input",
        str(trace_path),
        "--epochs",
        "1",
        "--checkpoint-output",
        str(checkpoint_path),
        "--json",
    ])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert checkpoint_path.is_file()
    assert payload["epochs"] == 1
    assert payload["example_count"] == 2
