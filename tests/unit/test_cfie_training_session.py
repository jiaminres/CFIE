"""Unit tests for the CFIE training session runner."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

import cfie_training.runtime.data as runtime_data
from cfie_training.config import TrainingProjectConfig
from cfie_training.profiles import build_profile_config
from cfie_training.runtime.project import TrainingProject
from cfie_training.runtime.session import TrainingSessionRunner
from cfie_training.runtime.types import BatchPlannerCheckpoint, BatchShape
from cfie_training.runtime.types import TrainingSessionCheckpoint


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


def _write_dataset(tmp_path, name: str = "train.txt") -> str:
    dataset_path = tmp_path / name
    dataset_path.write_text(
        "\n".join(
            (
                "CFIE keeps training batches dataset-backed.",
                "Predictor cleanup removes synthetic training fallback.",
                "Session resume must preserve deterministic token windows.",
            )
        ),
        encoding="utf-8",
    )
    return str(dataset_path)


class _FixedBatchPlanner:
    def batch_for_step(self, step_index: int) -> BatchShape:
        del step_index
        return BatchShape(samples=1, tokens_per_sample=32)

    def planner_checkpoint(self) -> BatchPlannerCheckpoint:
        return BatchPlannerCheckpoint(
            planner_kind="tokenized_dataset",
            base_samples=1,
            tokens_per_sample=32,
            dataset_path="unused.txt",
            dataset_format="text",
            dataset_text_key="text",
        )


def test_training_project_runs_session_with_periodic_checkpoints(tmp_path) -> None:
    project = TrainingProject(build_profile_config("qwen35-35b-a3b"))
    dataset_path = _write_dataset(tmp_path)

    session = project.train(
        steps=3,
        samples=2,
        tokens_per_sample=128,
        checkpoint_dir=str(tmp_path),
        checkpoint_interval=2,
        dataset_path=dataset_path,
    )

    assert session.total_steps == 3
    assert session.average_loss > 0
    assert session.max_loss >= session.average_loss
    assert session.peak_activation_bytes > 0
    assert len(session.checkpoint_paths) == 1

    checkpoint_path = tmp_path / "step_00002.json"
    assert session.checkpoint_paths[0] == str(checkpoint_path)
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert session.checkpoint_format == "training_session_checkpoint"
    assert payload["checkpoint_kind"] == "training_session_checkpoint"
    assert payload["planner"]["planner_kind"] == "tokenized_dataset"
    assert payload["planner"]["dataset_path"] == dataset_path
    assert payload["runtime_snapshot"]["next_step_index"] == 2
    assert isinstance(payload["runtime_snapshot"]["transport_cached_files"], list)
    assert len(payload["runtime_snapshot"]["transport_buffers"]) > 0


def test_training_project_can_resume_session_from_snapshot(tmp_path) -> None:
    project = TrainingProject(build_profile_config("qwen35-35b-a3b"))
    dataset_path = _write_dataset(tmp_path)

    first_session = project.train(
        steps=2,
        samples=2,
        tokens_per_sample=128,
        checkpoint_dir=str(tmp_path),
        checkpoint_interval=2,
        dataset_path=dataset_path,
    )
    checkpoint_path = tmp_path / "step_00002.json"
    checkpoint = TrainingSessionCheckpoint.from_dict(
        json.loads(checkpoint_path.read_text(encoding="utf-8"))
    )
    resumed_session = project.train(
        steps=1,
        samples=2,
        tokens_per_sample=128,
        snapshot=checkpoint.runtime_snapshot,
        planner_checkpoint=checkpoint.planner,
    )

    assert first_session.steps[-1].step_index == 1
    assert resumed_session.steps[0].step_index == 2
    assert resumed_session.average_loss > 0


def test_training_project_uses_dataset_backed_batches_deterministically(
    tmp_path,
) -> None:
    dataset_path = tmp_path / "train.txt"
    dataset_path.write_text(
        "\n".join(
            (
                "CFIE trains with bucket-local update and release.",
                "Experts are rotated instead of staying active together.",
                "CPU offload keeps host and device residency bounded.",
            )
        ),
        encoding="utf-8",
    )
    project = TrainingProject(build_profile_config("qwen35-35b-a3b"))

    first_session = project.train(
        steps=2,
        samples=2,
        tokens_per_sample=32,
        checkpoint_dir=str(tmp_path),
        checkpoint_interval=2,
        dataset_path=str(dataset_path),
    )
    checkpoint_path = tmp_path / "step_00002.json"
    checkpoint = TrainingSessionCheckpoint.from_dict(
        json.loads(checkpoint_path.read_text(encoding="utf-8"))
    )
    resumed_session = project.train(
        steps=1,
        samples=2,
        tokens_per_sample=32,
        snapshot=checkpoint.runtime_snapshot,
        planner_checkpoint=checkpoint.planner,
    )

    assert first_session.steps[0].batch.source_kind == "tokenized_dataset"
    assert first_session.steps[0].batch.dataset_name == "train.txt"
    assert first_session.steps[0].batch.sample_indices == (0, 1)
    assert first_session.steps[1].batch.sample_indices == (2, 0)
    assert resumed_session.steps[0].step_index == 2
    assert resumed_session.steps[0].batch.source_kind == "tokenized_dataset"
    assert resumed_session.steps[0].batch.sample_indices == (1, 2)
    assert checkpoint.planner.planner_kind == "tokenized_dataset"
    assert checkpoint.planner.dataset_path == str(dataset_path)


def test_training_project_resume_matches_dataset_horizon_when_steps_align(
    tmp_path,
) -> None:
    dataset_path = tmp_path / "train.txt"
    dataset_path.write_text(
        "\n".join(
            (
                "Dataset-backed checkpoint resume should preserve planner horizon.",
                "Direct and resumed sessions must agree on active and prefetched experts.",
                "Loss should stay deterministic when the same future batches remain visible.",
            )
        ),
        encoding="utf-8",
    )
    project = TrainingProject(build_profile_config("qwen35-35b-a3b"))

    direct_session = project.train(
        steps=4,
        samples=2,
        tokens_per_sample=32,
        checkpoint_dir=str(tmp_path),
        checkpoint_interval=2,
        dataset_path=str(dataset_path),
    )
    checkpoint_path = tmp_path / "step_00002.json"
    checkpoint = TrainingSessionCheckpoint.from_dict(
        json.loads(checkpoint_path.read_text(encoding="utf-8"))
    )
    resumed_session = project.train(
        steps=2,
        samples=2,
        tokens_per_sample=32,
        snapshot=checkpoint.runtime_snapshot,
        planner_checkpoint=checkpoint.planner,
    )

    direct_tail = direct_session.steps[2:]
    assert len(direct_tail) == len(resumed_session.steps) == 2
    for direct_step, resumed_step in zip(direct_tail, resumed_session.steps):
        assert resumed_step.step_index == direct_step.step_index
        assert resumed_step.batch.sample_indices == direct_step.batch.sample_indices
        assert resumed_step.active_expert_ids == direct_step.active_expert_ids
        assert resumed_step.prefetched_expert_ids == direct_step.prefetched_expert_ids
        assert resumed_step.execution_summary is not None
        assert direct_step.execution_summary is not None
        assert (
            resumed_step.execution_summary.total_loss
            == direct_step.execution_summary.total_loss
        )
        assert len(resumed_step.optimizer_updates) == len(direct_step.optimizer_updates)


def test_training_project_logical_checkpoint_uses_session_mirror_and_resumes(
    tmp_path,
) -> None:
    checkpoint_dir = tmp_path / "ckpts"
    cfg = TrainingProjectConfig.from_dict({
        "profile_name": "logical-checkpoint-smoke",
        "model_spec": {
            "architecture": "Qwen3_5MoeForConditionalGeneration",
            "text_model_type": "qwen3_5_moe_text",
            "hidden_size": 64,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 64,
            "shared_expert_intermediate_size": 64,
            "full_attention_interval": 4,
            "max_position_embeddings": 2048,
            "mtp_num_hidden_layers": 1,
            "attention_pattern": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
            "quantization": "gptq",
            "quant_bits": 4,
            "quant_group_size": 128,
            "quant_sym": True,
            "total_params_billion": 0.1,
        },
        "model_source": {
            "model_path": "fake-tokenizer",
            "use_local_weight_manifest": False,
        },
        "expert_rotation": {
            "active_experts_per_step": 2,
            "rotate_every_steps": 1,
            "rotate_every_samples": 2,
            "retain_active_window_state_in_memory": True,
        },
        "execution": {
            "compute_device": "cpu",
            "optimizer_device": "cpu",
            "gradient_device": "cpu",
            "trainable_shard_materialization": "logical",
            "logical_cuda_execution_mode": "full_bucket",
        },
        "optimizer": {
            "offload_state_after_update": True,
        },
        "runtime_quantization": {
            "enabled": True,
            "persist_fp32_to_nvme": True,
            "nvme_staging_dir": str(tmp_path / "mirror"),
        },
    })
    project = TrainingProject(cfg)
    dataset_path = _write_dataset(tmp_path, "logical-train.txt")

    direct_session = project.train(
        steps=3,
        samples=1,
        tokens_per_sample=16,
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_interval=2,
        dataset_path=dataset_path,
    )
    checkpoint_path = checkpoint_dir / "step_00002.json"
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    snapshot_payload = payload["runtime_snapshot"]

    assert (
        snapshot_payload["runtime_quantization_session_id"]
        != cfg.runtime_quantization.session_id
    )
    frozen_root = (
        tmp_path
        / "mirror"
        / snapshot_payload["runtime_quantization_session_id"]
    )
    assert frozen_root.exists()
    assert any(
        shard["representative_params"] > 128 and shard["parameter_values"] == []
        for shard in snapshot_payload["parameter_store_shards"]
        if shard["component"] in {"bucket_non_routed", "bucket_active_experts"}
    )
    assert any(
        shard["representative_params"] > 128
        and shard["exp_avg_values"] == []
        and shard["exp_avg_sq_values"] == []
        for shard in snapshot_payload["optimizer_shards"]
    )

    checkpoint = TrainingSessionCheckpoint.from_dict(payload)
    resumed_cfg = TrainingProjectConfig.from_dict({
        **json.loads(cfg.to_json()),
        "runtime_quantization": {
            **json.loads(cfg.to_json())["runtime_quantization"],
            "session_id": "",
        },
    })
    assert resumed_cfg.runtime_quantization.session_id != cfg.runtime_quantization.session_id
    resumed_project = TrainingProject(resumed_cfg)
    resumed_session = resumed_project.train(
        steps=1,
        samples=1,
        tokens_per_sample=16,
        snapshot=checkpoint.runtime_snapshot,
        planner_checkpoint=checkpoint.planner,
    )

    direct_step = direct_session.steps[2]
    resumed_step = resumed_session.steps[0]
    assert resumed_step.step_index == direct_step.step_index
    assert resumed_step.active_expert_ids == direct_step.active_expert_ids
    assert resumed_step.prefetched_expert_ids == direct_step.prefetched_expert_ids
    assert resumed_step.execution_summary is not None
    assert direct_step.execution_summary is not None
    assert (
        resumed_step.execution_summary.total_loss
        == direct_step.execution_summary.total_loss
    )


def test_training_project_logical_dataset_checkpoint_resume_matches_horizon(
    tmp_path,
) -> None:
    dataset_path = tmp_path / "train.txt"
    dataset_path.write_text(
        "\n".join(
            (
                "Logical dataset-backed resume should preserve planner horizon.",
                "Mirror-backed checkpoints should resume without window drift.",
                "Direct and resumed sessions should agree on routed experts.",
            )
        ),
        encoding="utf-8",
    )
    checkpoint_dir = tmp_path / "ckpts"
    tokenizer_root = build_profile_config("qwen35-35b-a3b").model_source.model_path
    cfg = TrainingProjectConfig.from_dict({
        "profile_name": "logical-dataset-checkpoint-smoke",
        "model_spec": {
            "architecture": "Qwen3_5MoeForConditionalGeneration",
            "text_model_type": "qwen3_5_moe_text",
            "hidden_size": 64,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 64,
            "shared_expert_intermediate_size": 64,
            "full_attention_interval": 4,
            "max_position_embeddings": 2048,
            "mtp_num_hidden_layers": 1,
            "attention_pattern": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
            "quantization": "gptq",
            "quant_bits": 4,
            "quant_group_size": 128,
            "quant_sym": True,
            "total_params_billion": 0.1,
        },
        "model_source": {
            "model_path": tokenizer_root,
            "use_local_weight_manifest": False,
        },
        "expert_rotation": {
            "active_experts_per_step": 2,
            "rotate_every_steps": 1,
            "rotate_every_samples": 2,
            "retain_active_window_state_in_memory": True,
        },
        "execution": {
            "compute_device": "cpu",
            "optimizer_device": "cpu",
            "gradient_device": "cpu",
            "trainable_shard_materialization": "logical",
            "logical_cuda_execution_mode": "full_bucket",
        },
        "optimizer": {
            "offload_state_after_update": True,
        },
        "runtime_quantization": {
            "enabled": True,
            "persist_fp32_to_nvme": True,
            "nvme_staging_dir": str(tmp_path / "mirror"),
        },
    })

    direct_project = TrainingProject(cfg)
    direct_session = direct_project.train(
        steps=4,
        samples=1,
        tokens_per_sample=16,
        dataset_path=str(dataset_path),
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_interval=2,
    )
    checkpoint = TrainingSessionCheckpoint.from_dict(
        json.loads((checkpoint_dir / "step_00002.json").read_text(encoding="utf-8"))
    )

    resumed_cfg = TrainingProjectConfig.from_dict({
        **json.loads(cfg.to_json()),
        "runtime_quantization": {
            **json.loads(cfg.to_json())["runtime_quantization"],
            "session_id": "",
        },
    })
    resumed_project = TrainingProject(resumed_cfg)
    resumed_session = resumed_project.train(
        steps=2,
        samples=1,
        tokens_per_sample=16,
        snapshot=checkpoint.runtime_snapshot,
        planner_checkpoint=checkpoint.planner,
    )

    assert checkpoint.planner.planner_kind == "tokenized_dataset"
    assert checkpoint.planner.dataset_path == str(dataset_path)

    direct_tail = direct_session.steps[2:]
    assert len(direct_tail) == len(resumed_session.steps) == 2
    for direct_step, resumed_step in zip(direct_tail, resumed_session.steps):
        assert direct_step.batch.source_kind == "tokenized_dataset"
        assert resumed_step.batch.source_kind == "tokenized_dataset"
        assert resumed_step.step_index == direct_step.step_index
        assert resumed_step.batch.sample_indices == direct_step.batch.sample_indices
        assert resumed_step.active_expert_ids == direct_step.active_expert_ids
        assert resumed_step.prefetched_expert_ids == direct_step.prefetched_expert_ids
        assert resumed_step.execution_summary is not None
        assert direct_step.execution_summary is not None
        assert (
            resumed_step.execution_summary.total_loss
            == direct_step.execution_summary.total_loss
        )
        assert len(resumed_step.optimizer_updates) == len(direct_step.optimizer_updates)


def test_training_session_runner_avoids_snapshot_hot_path_without_checkpoints() -> None:
    project = TrainingProject(build_profile_config("qwen35-35b-a3b"))
    runner = TrainingSessionRunner(project.build_engine())

    with patch.object(
        type(runner.engine),
        "snapshot_state",
        autospec=True,
        side_effect=AssertionError(
            "run() should not snapshot runtime state when checkpoints are disabled"
        ),
    ):
        session = runner.run(
            steps=2,
            batch_planner=_FixedBatchPlanner(),
            retain_step_traces=False,
        )

    assert session.total_steps == 2
    assert session.average_loss > 0
    assert session.steps == ()
