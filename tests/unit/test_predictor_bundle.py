"""Predictor runtime bundle and checkpoint loading tests."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from safetensors.torch import save_file
import torch

from cfie.model_executor.model_loader import utils as model_loader_utils
from cfie.model_executor.models.qwen3_5_predictor import (
    PredictorForwardState,
    PredictorLayerRoutingState,
    Qwen3_5PredictorModel,
    Qwen3_5MoePredictorProcessingInfo,
)
from cfie.predictor import (
    PredictorCandidatePlanner,
    PredictorDeploymentManifest,
    PredictorMetricsSummary,
    PredictorRuntimeSchema,
    load_predictor_bundle,
    load_predictor_model,
)
from cfie.predictor.planner import CandidateLayerPlan
from cfie_training.predictor import PredictorTraceDataset, PredictorTrainer
from cfie_training.predictor.architectures import build_predictor_model
from cfie_training.predictor.models import PredictorCheckpointMetadata
from cfie_training.predictor.trainer import CapturedForwardBatch
from cfie_training.profiles import build_profile_config
from cfie_training.runtime.types import BatchShape
from cfie.transformers_utils.configs.qwen3_5_moe import Qwen3_5MoeConfig
from cfie.transformers_utils.configs.qwen3_5_moe_predictor import (
    Qwen3_5MoePredictorConfig,
)


class _FakeForwardBackend:
    def __init__(self, hidden_dim: int, num_layers: int, num_experts: int) -> None:
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._num_experts = num_experts

    def capture_batch(self, batch: BatchShape) -> CapturedForwardBatch:
        token_count = batch.total_tokens
        layer_hidden_states = []
        layer_teacher_topk_ids = []
        layer_teacher_router_logits = []
        hidden_template = torch.arange(
            token_count * self._hidden_dim,
            dtype=torch.float32,
        ).reshape(token_count, self._hidden_dim)
        teacher_topk_template = torch.arange(
            token_count * 8,
            dtype=torch.long,
        ).reshape(token_count, 8) % self._num_experts
        router_logits_template = torch.arange(
            token_count * self._num_experts,
            dtype=torch.float32,
        ).reshape(token_count, self._num_experts)
        for layer_index in range(self._num_layers):
            layer_hidden_states.append(hidden_template + float(layer_index))
            layer_teacher_topk_ids.append(
                (teacher_topk_template + layer_index) % self._num_experts
            )
            layer_teacher_router_logits.append(
                router_logits_template + float(layer_index)
            )
        return CapturedForwardBatch(
            layer_hidden_states=tuple(layer_hidden_states),
            layer_teacher_topk_ids=tuple(layer_teacher_topk_ids),
            layer_teacher_router_logits=tuple(layer_teacher_router_logits),
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


def _build_training_test_config(profile_name: str = "qwen35-35b-a3b"):
    config = build_profile_config(profile_name)
    config.predictor_trainer.min_insertion_layer_index = 0
    return config


def _build_test_predictor_model(
    *,
    architecture: str = "mlp",
    num_layers: int | None = None,
    frozen_router_weights: torch.Tensor | None = None,
):
    config = build_profile_config("qwen35-35b-a3b")
    return build_predictor_model(
        input_dim=16,
        hidden_dim=24,
        window_layers=config.predictor_routing.window_layers,
        num_experts=64,
        model_architecture=architecture,
        model_depth=2,
        model_dropout=0.0,
        model_num_heads=1,
        model_memory_tokens=1,
        model_ffn_multiplier=2,
        num_layers=num_layers,
        frozen_router_weights=frozen_router_weights,
    ).eval()


def _build_predictor_bundle(tmp_path: Path) -> Path:
    config = build_profile_config("qwen35-35b-a3b")
    model = _build_test_predictor_model()

    checkpoint_path = tmp_path / "predictor.ckpt"
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    weights_path = bundle_dir / "predictor_weights.pt"
    schema_path = bundle_dir / "predictor_schema.json"
    metrics_path = bundle_dir / "predictor_metrics.json"
    manifest_path = bundle_dir / "predictor_bundle.json"

    schema = PredictorRuntimeSchema(
        schema_kind="cfie_predictor_runtime_schema",
        profile_name=config.profile_name,
        input_summary_dim=16,
        predictor_hidden_dim=24,
        window_layers=config.predictor_routing.window_layers,
        stride_layers=config.predictor_routing.stride_layers,
        num_experts=64,
        candidate_experts_per_layer=(
            config.predictor_routing.candidate_experts_per_layer
        ),
        executed_experts_per_layer=(
            config.predictor_routing.executed_experts_per_layer
        ),
        selection_mode=config.predictor_routing.selection_mode,
        online_expert_source=config.predictor_routing.online_expert_source,
        allow_candidate_mismatch=config.predictor_routing.allow_candidate_mismatch,
        model_descriptor=model.model_descriptor(),
        min_insertion_layer_index=config.predictor_trainer.min_insertion_layer_index,
    ).validate()
    metrics = PredictorMetricsSummary(
        metrics_kind="cfie_predictor_metrics_summary",
        profile_name=config.profile_name,
        example_count=32,
        epochs=1,
        final_mean_loss=0.1,
        final_recall_at_candidate_budget=0.75,
        final_recall_at_executed_budget=0.5,
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
    return bundle_dir


def _build_predictor_checkpoint(tmp_path: Path) -> Path:
    config = build_profile_config("qwen35-35b-a3b")
    model = _build_test_predictor_model()
    metadata = PredictorCheckpointMetadata(
        checkpoint_kind="cfie_predictor_checkpoint",
        profile_name=config.profile_name,
        input_summary_dim=16,
        hidden_dim=24,
        model_descriptor=model.model_descriptor(),
        window_layers=config.predictor_routing.window_layers,
        stride_layers=config.predictor_routing.stride_layers,
        num_experts=64,
        candidate_experts_per_layer=(
            config.predictor_routing.candidate_experts_per_layer
        ),
        executed_experts_per_layer=(
            config.predictor_routing.executed_experts_per_layer
        ),
        selection_mode=config.predictor_routing.selection_mode,
        online_expert_source=config.predictor_routing.online_expert_source,
        allow_candidate_mismatch=config.predictor_routing.allow_candidate_mismatch,
        min_insertion_layer_index=config.predictor_trainer.min_insertion_layer_index,
        example_count=32,
        epochs=1,
        final_mean_loss=0.1,
        final_recall_at_candidate_budget=0.75,
        final_recall_at_executed_budget=0.5,
    )
    checkpoint_path = tmp_path / "predictor.ckpt"
    torch.save(
        {
            "metadata": metadata.to_dict(),
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    return checkpoint_path


def _write_fake_router_base_model(
    tmp_path: Path,
    *,
    num_layers: int,
    num_experts: int,
    input_dim: int,
) -> tuple[Path, torch.Tensor]:
    model_dir = tmp_path / "base_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    tensors = {
        f"model.layers.{layer_index}.mlp.gate.weight": (
            torch.arange(
                num_experts * input_dim,
                dtype=torch.float32,
            ).reshape(num_experts, input_dim)
            + float(layer_index)
        )
        for layer_index in range(num_layers)
    }
    save_file(tensors, model_dir / "model.safetensors")
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {name: "model.safetensors" for name in tensors},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    router_weights = torch.stack(
        [
            tensors[f"model.layers.{layer_index}.mlp.gate.weight"]
            for layer_index in range(num_layers)
        ],
        dim=0,
    )
    return model_dir, router_weights


def _build_frozen_router_delta_checkpoint(
    tmp_path: Path,
) -> tuple[Path, Path, torch.nn.Module]:
    config = build_profile_config("qwen35-35b-a3b")
    num_layers = 12
    base_model_dir, router_weights = _write_fake_router_base_model(
        tmp_path,
        num_layers=num_layers,
        num_experts=64,
        input_dim=16,
    )
    model = _build_test_predictor_model(
        architecture="frozen_router_delta",
        num_layers=num_layers,
        frozen_router_weights=router_weights,
    )
    metadata = PredictorCheckpointMetadata(
        checkpoint_kind="cfie_predictor_checkpoint",
        profile_name=config.profile_name,
        input_summary_dim=16,
        hidden_dim=24,
        model_descriptor=model.model_descriptor(),
        window_layers=config.predictor_routing.window_layers,
        stride_layers=config.predictor_routing.stride_layers,
        num_experts=64,
        candidate_experts_per_layer=(
            config.predictor_routing.candidate_experts_per_layer
        ),
        executed_experts_per_layer=(
            config.predictor_routing.executed_experts_per_layer
        ),
        selection_mode=config.predictor_routing.selection_mode,
        online_expert_source=config.predictor_routing.online_expert_source,
        allow_candidate_mismatch=config.predictor_routing.allow_candidate_mismatch,
        min_insertion_layer_index=config.predictor_trainer.min_insertion_layer_index,
        example_count=32,
        epochs=1,
        final_mean_loss=0.1,
        final_recall_at_candidate_budget=0.75,
        final_recall_at_executed_budget=0.5,
    )
    checkpoint_path = tmp_path / "frozen_router_delta.ckpt"
    torch.save(
        {
            "metadata": metadata.to_dict(),
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    return checkpoint_path, base_model_dir, model


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
    assert bundle.schema.model_descriptor["architecture"] == "mlp"
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


def test_load_predictor_model_rebuilds_predictor_module_from_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint_path = _build_predictor_checkpoint(tmp_path)

    model, bundle = load_predictor_model(checkpoint_path)
    outputs = model(
        torch.zeros(2, bundle.schema.input_summary_dim, dtype=torch.float32),
        torch.tensor([0.0, 7.0], dtype=torch.float32),
    )

    assert bundle.manifest is None
    assert bundle.checkpoint_metadata is not None
    assert bundle.schema.model_descriptor["architecture"] == "mlp"
    assert tuple(outputs.shape) == (
        2,
        bundle.schema.window_layers,
        bundle.schema.num_experts,
    )


def test_load_predictor_model_rebuilds_frozen_router_delta_from_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint_path, base_model_dir, reference_model = (
        _build_frozen_router_delta_checkpoint(tmp_path)
    )
    hidden_state = torch.randn(2, 16, dtype=torch.float32)
    layer_index = torch.tensor([1.0, 3.0], dtype=torch.float32)

    loaded_model, bundle = load_predictor_model(
        checkpoint_path,
        base_model_path=base_model_dir,
        num_layers=12,
    )

    expected = reference_model(hidden_state, layer_index)
    actual = loaded_model(hidden_state, layer_index)

    assert bundle.schema.model_descriptor["architecture"] == "frozen_router_delta"
    assert torch.allclose(actual, expected)


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


def test_qwen35_122b_trace_builder_uses_stride4_and_keeps_tail_window() -> None:
    config = build_profile_config("qwen35-122b-a10b")
    trainer = PredictorTrainer(
        config,
        teacher_model_backend=_FakeForwardBackend(
            hidden_dim=config.model_spec.hidden_size,
            num_layers=config.model_spec.num_hidden_layers,
            num_experts=config.model_spec.num_experts,
        ),
    )

    trace_builder = trainer._resolve_trace_builder()

    assert trace_builder._insertion_layer_indices[:4] == (0, 4, 8, 12)
    assert trace_builder._insertion_layer_indices[-1] == 44
    assert trace_builder._future_layer_indices(44) == (45, 46, 47)


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


def test_model_loader_promotes_qwen_moe_to_predictor_runtime_override() -> None:
    class _FakeRegistry:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        def resolve_model_cls(self, architectures, model_config):
            del model_config
            arch_tuple = tuple(architectures)
            self.calls.append(arch_tuple)
            if arch_tuple == ("Qwen3_5MoePredictorForConditionalGeneration",):
                return torch.nn.Module, "Qwen3_5MoePredictorForConditionalGeneration"
            return torch.nn.Module, "Qwen3_5MoeForConditionalGeneration"

    registry = _FakeRegistry()
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=["Qwen3_5MoeForConditionalGeneration"],
            predictor_checkpoint_path="predictor.ckpt",
            predictor_bundle_path=None,
        ),
        hf_text_config=SimpleNamespace(
            predictor_checkpoint_path=None,
            predictor_bundle_path=None,
        ),
        registry=registry,
        _get_transformers_backend_cls=lambda: "TransformersModel",
        model_impl="cfie",
        convert_type="none",
    )

    _model_cls, arch = model_loader_utils._get_model_architecture(model_config)

    assert arch == "Qwen3_5MoePredictorForConditionalGeneration"
    assert registry.calls == [
        ("Qwen3_5MoeForConditionalGeneration",),
        ("Qwen3_5MoePredictorForConditionalGeneration",),
    ]


def test_qwen35_predictor_schedule_routing_state_prefetches_future_experts() -> None:
    prefetch_calls: list[tuple[int, ...]] = []
    controller = SimpleNamespace(
        prefetch=lambda expert_ids: prefetch_calls.append(tuple(expert_ids))
    )
    future_layer = SimpleNamespace(
        mlp=SimpleNamespace(
            experts=SimpleNamespace(_cfie_tiered_cache_controller=controller)
        )
    )
    model = Qwen3_5PredictorModel.__new__(Qwen3_5PredictorModel)
    model.predictor_forward_state = PredictorForwardState()
    model._predictor_decoder_layer = (
        lambda *, layer_index: future_layer if layer_index == 12 else None
    )
    routing_state = PredictorLayerRoutingState(
        insertion_layer_index=8,
        layer_plan=CandidateLayerPlan(
            future_layer_index=12,
            predicted_executed_expert_ids=(5, 7),
            candidate_expert_ids=(5, 7, 11),
            candidate_scores=(0.9, 0.8, 0.7),
        ),
    )

    model._schedule_predictor_routing_state(routing_state)

    assert model.predictor_forward_state.pending_layer_plans[12] is routing_state
    assert prefetch_calls == [(5, 7)]


def test_predictor_processing_info_coerces_base_moe_config() -> None:
    hf_config = Qwen3_5MoeConfig()
    hf_config.predictor_checkpoint_path = "predictor.ckpt"
    hf_config.predictor_map_location = "cpu"
    hf_config.predictor_device = "cpu"
    hf_config.text_config.predictor_checkpoint_path = "predictor.ckpt"
    hf_config.text_config.predictor_map_location = "cpu"
    hf_config.text_config.predictor_device = "cpu"

    info = Qwen3_5MoePredictorProcessingInfo.__new__(
        Qwen3_5MoePredictorProcessingInfo
    )
    info.ctx = SimpleNamespace(get_hf_config=lambda *args, **kwargs: hf_config)

    coerced = info.get_hf_config()

    assert isinstance(coerced, Qwen3_5MoePredictorConfig)
    assert coerced.predictor_checkpoint_path == "predictor.ckpt"
    assert coerced.text_config.predictor_checkpoint_path == "predictor.ckpt"


def test_predictor_trainer_prefers_forward_capture_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_training_test_config()
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


def test_predictor_trainer_streams_trace_dataset_to_json_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_training_test_config()
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

    output_path = tmp_path / "predictor_trace.json"
    progress_updates = []
    result = trainer.build_trace_dataset_to_json_file(
        output_path=output_path,
        steps=2,
        examples_per_step=1,
        dataset_path="fake.txt",
        flush_every_steps=1,
        progress_callback=progress_updates.append,
    )
    dataset = PredictorTraceDataset.from_json_file(output_path)

    assert result.output_path == output_path
    assert result.example_count == 2
    assert dataset.example_count == 2
    assert progress_updates
    assert progress_updates[-1].completed_steps == 2
    assert progress_updates[-1].persisted_steps == 2
    assert any(update.persisted_steps == 1 for update in progress_updates)
    assert not output_path.with_name(
        output_path.name + ".progress.examples.jsonl"
    ).exists()
    assert not output_path.with_name(
        output_path.name + ".progress.json"
    ).exists()


def test_predictor_trainer_rejects_dataset_config_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_training_test_config()
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
    config = _build_training_test_config()
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
