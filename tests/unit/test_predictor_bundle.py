"""Predictor deployment bundle loading tests."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from cfie.predictor import (
    PredictorCandidatePlanner,
    PredictorDeploymentManifest,
    PredictorMetricsSummary,
    PredictorRuntimeSchema,
    load_predictor_bundle,
    load_predictor_model,
)
from cfie_training.predictor import PredictorTrainer
from cfie_training.predictor.trainer import CapturedForwardBatch
from cfie_training.profiles import build_profile_config
from cfie_training.runtime.types import BatchShape


class _FakeForwardBackend:
    def __init__(self, hidden_dim: int, num_layers: int, num_experts: int) -> None:
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._num_experts = num_experts

    def capture_batch(self, batch: BatchShape) -> CapturedForwardBatch:
        token_count = batch.total_tokens
        layer_hidden_states = []
        layer_teacher_topk_ids = []
        hidden_template = torch.arange(
            token_count * self._hidden_dim,
            dtype=torch.float32,
        ).reshape(token_count, self._hidden_dim)
        teacher_topk_template = torch.arange(
            token_count * 8,
            dtype=torch.long,
        ).reshape(token_count, 8) % self._num_experts
        for layer_index in range(self._num_layers):
            layer_hidden_states.append(hidden_template + float(layer_index))
            layer_teacher_topk_ids.append(
                (teacher_topk_template + layer_index) % self._num_experts
            )
        return CapturedForwardBatch(
            layer_hidden_states=tuple(layer_hidden_states),
            layer_teacher_topk_ids=tuple(layer_teacher_topk_ids),
        )


class _FakeTokenBatchPlanner:
    def batch_for_step(self, step_index: int) -> BatchShape:
        del step_index
        return BatchShape(
            samples=2,
            tokens_per_sample=3,
            source_kind="tokenized_dataset",
            dataset_name="fake.txt",
            sample_indices=(0, 1),
            loss_token_count=6,
            token_rows=((1, 2, 3), (4, 5, 6)),
            target_rows=((2, 3, 4), (5, 6, 7)),
        )


def _build_predictor_bundle(tmp_path: Path) -> Path:
    # 构造一份最小可训练的 predictor 配置。
    config = build_profile_config("qwen35-35b-a3b")
    trainer = PredictorTrainer(
        config,
        teacher_model_backend=_FakeForwardBackend(
            hidden_dim=config.model_spec.hidden_size,
            num_layers=config.model_spec.num_hidden_layers,
            num_experts=config.model_spec.num_experts,
        ),
    )
    trainer._build_batch_planner = lambda **_: _FakeTokenBatchPlanner()
    # 生成最小 trace 数据集并执行一轮训练。
    dataset = trainer.build_trace_dataset(
        steps=1,
        examples_per_step=1,
        dataset_path="fake.txt",
    )
    model, run_trace, _ = trainer.fit_dataset(dataset, epochs=1)

    # 将训练结果保存为 checkpoint，并手工组装一份最小 bundle。
    checkpoint_path = tmp_path / "predictor.ckpt"
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    weights_path = bundle_dir / "predictor_weights.pt"
    schema_path = bundle_dir / "predictor_schema.json"
    metrics_path = bundle_dir / "predictor_metrics.json"
    manifest_path = bundle_dir / "predictor_bundle.json"
    trainer.save_checkpoint(
        model=model,
        run_trace=run_trace,
        path=checkpoint_path,
    )

    schema = PredictorRuntimeSchema(
        schema_kind="cfie_predictor_runtime_schema",
        profile_name=config.profile_name,
        input_summary_dim=config.model_spec.hidden_size,
        predictor_hidden_dim=config.predictor_trainer.hidden_dim,
        window_layers=config.predictor_routing.window_layers,
        stride_layers=config.predictor_routing.stride_layers,
        num_experts=config.model_spec.num_experts,
        candidate_experts_per_layer=(
            config.predictor_routing.candidate_experts_per_layer
        ),
        executed_experts_per_layer=(
            config.predictor_routing.executed_experts_per_layer
        ),
        selection_mode=config.predictor_routing.selection_mode,
        online_expert_source=config.predictor_routing.online_expert_source,
        allow_candidate_mismatch=config.predictor_routing.allow_candidate_mismatch,
    ).validate()
    metrics = PredictorMetricsSummary(
        metrics_kind="cfie_predictor_metrics_summary",
        profile_name=run_trace.profile_name,
        example_count=run_trace.example_count,
        epochs=run_trace.epochs,
        final_mean_loss=run_trace.final_mean_loss,
        final_recall_at_candidate_budget=(
            run_trace.final_recall_at_candidate_budget
        ),
        final_recall_at_executed_budget=(
            run_trace.final_recall_at_executed_budget
        ),
    ).validate()
    manifest = PredictorDeploymentManifest(
        bundle_kind="cfie_predictor_deployment_bundle",
        profile_name=config.profile_name,
        source_checkpoint=checkpoint_path.name,
        weights_kind="cfie_predictor_weights",
        weights_format="torch_state_dict",
        weights_file=weights_path.name,
        schema_kind=schema.schema_kind,
        schema_file=schema_path.name,
        metrics_kind=metrics.metrics_kind,
        metrics_file=metrics_path.name,
    ).validate()

    torch.save(
        {
            "weights_kind": manifest.weights_kind,
            "profile_name": manifest.profile_name,
            "model_state_dict": model.state_dict(),
        },
        weights_path,
    )
    schema_path.write_text(
        json.dumps(schema.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    metrics_path.write_text(
        json.dumps(metrics.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    # 返回 bundle 目录，供测试继续加载。
    return bundle_dir


def test_load_predictor_bundle_reads_bundle_directory(tmp_path: Path) -> None:
    bundle_dir = _build_predictor_bundle(tmp_path)

    bundle = load_predictor_bundle(bundle_dir)

    assert bundle.manifest.bundle_kind == "cfie_predictor_deployment_bundle"
    assert bundle.schema.schema_kind == "cfie_predictor_runtime_schema"
    assert bundle.metrics.metrics_kind == "cfie_predictor_metrics_summary"
    assert bundle.schema.profile_name == "qwen35-35b-a3b"
    assert bundle.schema.window_layers == 8
    assert bundle.schema.candidate_experts_per_layer == 40
    assert bundle.schema.executed_experts_per_layer == 8
    assert bundle.weights_path.name == "predictor_weights.pt"
    assert "input_proj.weight" in bundle.state_dict


def test_load_predictor_model_rebuilds_predictor_module(tmp_path: Path) -> None:
    bundle_dir = _build_predictor_bundle(tmp_path)

    model, bundle = load_predictor_model(bundle_dir)
    inputs = torch.zeros(2, bundle.schema.input_summary_dim, dtype=torch.float32)
    layer_indices = torch.tensor([0.0, 7.0], dtype=torch.float32)

    outputs = model(inputs, layer_indices)

    assert tuple(outputs.shape) == (
        2,
        bundle.schema.window_layers,
        bundle.schema.num_experts,
    )


def test_load_predictor_model_preserves_bundle_weight_dtype(
    tmp_path: Path,
) -> None:
    bundle_dir = _build_predictor_bundle(tmp_path)
    original_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    try:
        model, bundle = load_predictor_model(bundle_dir)
    finally:
        torch.set_default_dtype(original_default_dtype)

    first_param = next(model.parameters())
    assert first_param.dtype == torch.float32

    outputs = model(
        torch.zeros(1, bundle.schema.input_summary_dim, dtype=torch.float32),
        torch.tensor([0.0], dtype=torch.float32),
    )
    assert outputs.dtype == torch.float32


def test_load_predictor_bundle_rejects_schema_kind_mismatch(tmp_path: Path) -> None:
    bundle_dir = _build_predictor_bundle(tmp_path)
    schema_path = bundle_dir / "predictor_schema.json"
    payload = json.loads(schema_path.read_text(encoding="utf-8"))
    payload["schema_kind"] = "broken_schema_kind"
    schema_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema_kind"):
        load_predictor_bundle(bundle_dir)


def test_predictor_candidate_planner_builds_window_plan(tmp_path: Path) -> None:
    bundle_dir = _build_predictor_bundle(tmp_path)
    model, bundle = load_predictor_model(bundle_dir)
    planner = PredictorCandidatePlanner(
        schema=bundle.schema,
        model=model,
    )
    hidden_state = torch.zeros(bundle.schema.input_summary_dim, dtype=torch.float32)

    plan = planner.plan_window(
        hidden_state,
        insertion_layer_index=7,
        total_layers=64,
    )

    assert plan.profile_name == bundle.schema.profile_name
    assert plan.selection_mode == bundle.schema.selection_mode
    assert len(plan.layer_plans) == bundle.schema.window_layers
    assert plan.shared_gpu_candidate_slots == 256
    assert plan.speculative_experts_per_layer == 32
    assert plan.layer_plans[0].future_layer_index == 8
    assert plan.layer_plans[-1].future_layer_index == 15
    for layer_plan in plan.layer_plans:
        assert len(layer_plan.candidate_expert_ids) == (
            bundle.schema.candidate_experts_per_layer
        )
        assert len(layer_plan.predicted_executed_expert_ids) == (
            bundle.schema.executed_experts_per_layer
        )
        assert len(layer_plan.speculative_expert_ids) == 32


def test_predictor_candidate_planner_rejects_bad_hidden_summary_dim(
    tmp_path: Path,
) -> None:
    bundle_dir = _build_predictor_bundle(tmp_path)
    model, bundle = load_predictor_model(bundle_dir)
    planner = PredictorCandidatePlanner(
        schema=bundle.schema,
        model=model,
    )
    hidden_state = torch.zeros(bundle.schema.input_summary_dim + 1)

    with pytest.raises(ValueError, match="dimension mismatch"):
        planner.plan_window(
            hidden_state,
            insertion_layer_index=0,
            total_layers=64,
        )


def test_predictor_trainer_prefers_forward_capture_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = build_profile_config("qwen35-35b-a3b")
    trainer = PredictorTrainer(
        config,
        teacher_model_backend=_FakeForwardBackend(
            hidden_dim=config.model_spec.hidden_size,
            num_layers=config.model_spec.num_hidden_layers,
            num_experts=config.model_spec.num_experts,
        ),
    )
    monkeypatch.setattr(
        trainer,
        "_build_batch_planner",
        lambda **_: _FakeTokenBatchPlanner(),
    )

    dataset = trainer.build_trace_dataset(
        steps=1,
        examples_per_step=2,
        dataset_path="fake.txt",
    )

    assert dataset.example_count == 2
    assert len(dataset.examples[0].hidden_state) == config.model_spec.hidden_size
    assert len(dataset.examples[0].future_layer_indices) == (
        config.predictor_routing.window_layers
    )


def test_predictor_trainer_rejects_dataset_config_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = build_profile_config("qwen35-35b-a3b")
    trainer = PredictorTrainer(
        config,
        teacher_model_backend=_FakeForwardBackend(
            hidden_dim=config.model_spec.hidden_size,
            num_layers=config.model_spec.num_hidden_layers,
            num_experts=config.model_spec.num_experts,
        ),
    )
    monkeypatch.setattr(
        trainer,
        "_build_batch_planner",
        lambda **_: _FakeTokenBatchPlanner(),
    )

    dataset = trainer.build_trace_dataset(
        steps=1,
        examples_per_step=1,
        dataset_path="fake.txt",
    )
    mismatched_dataset = replace(
        dataset,
        candidate_experts_per_layer=dataset.candidate_experts_per_layer - 1,
    )

    with pytest.raises(ValueError, match="candidate_experts_per_layer"):
        trainer.evaluate_dataset(mismatched_dataset)


def test_predictor_trainer_rejects_resume_run_trace_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = build_profile_config("qwen35-35b-a3b")
    trainer = PredictorTrainer(
        config,
        teacher_model_backend=_FakeForwardBackend(
            hidden_dim=config.model_spec.hidden_size,
            num_layers=config.model_spec.num_hidden_layers,
            num_experts=config.model_spec.num_experts,
        ),
    )
    monkeypatch.setattr(
        trainer,
        "_build_batch_planner",
        lambda **_: _FakeTokenBatchPlanner(),
    )

    dataset = trainer.build_trace_dataset(
        steps=1,
        examples_per_step=1,
        dataset_path="fake.txt",
    )
    _, run_trace, _ = trainer.fit_dataset(dataset, epochs=1)
    mismatched_run_trace = replace(
        run_trace,
        example_count=run_trace.example_count + 1,
    )

    with pytest.raises(ValueError, match="example_count"):
        trainer.fit_dataset(
            dataset,
            epochs=1,
            initial_run_trace=mismatched_run_trace,
        )
