"""Training session runner for the CFIE training runtime."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import shutil
from typing import Protocol

from cfie_training.runtime.data import TokenizedDatasetBatchPlanner
from cfie_training.runtime.engine import FirstVersionTrainingEngine
from cfie_training.runtime.types import (
    BatchPlannerCheckpoint,
    BatchShape,
    TrainingRuntimeSnapshot,
    TrainingSessionCheckpoint,
    TrainingSessionTrace,
    TrainingStepTrace,
)


class BatchPlanner(Protocol):
    def batch_for_step(self, step_index: int) -> BatchShape:
        ...

    def planner_checkpoint(self) -> BatchPlannerCheckpoint:
        ...


@dataclass(slots=True)
class TrainingSessionRunner:
    engine: FirstVersionTrainingEngine

    def _freeze_runtime_snapshot_for_checkpoint(
        self,
        snapshot: TrainingRuntimeSnapshot,
        *,
        checkpoint_name: str,
    ) -> TrainingRuntimeSnapshot:
        staging_dir = self.engine.config.runtime_quantization.nvme_staging_dir
        session_id = snapshot.runtime_quantization_session_id
        if not staging_dir or not session_id:
            return snapshot

        base_root = Path(staging_dir).expanduser()
        live_root = base_root / session_id
        if not live_root.exists():
            return snapshot

        frozen_session_id = f"{session_id}_{checkpoint_name}"
        frozen_root = base_root / frozen_session_id
        if frozen_root.exists():
            shutil.rmtree(frozen_root)
        shutil.copytree(live_root, frozen_root)
        return replace(
            snapshot,
            runtime_quantization_session_id=frozen_session_id,
        )

    def run(
        self,
        *,
        steps: int,
        batch_planner: BatchPlanner,
        checkpoint_dir: Path | None = None,
        checkpoint_interval: int = 0,
        retain_step_traces: bool = True,
    ) -> TrainingSessionTrace:
        if steps < 1:
            raise ValueError("steps must be >= 1")
        if checkpoint_interval < 0:
            raise ValueError("checkpoint_interval must be >= 0")

        checkpoint_paths: list[str] = []
        traces: list[TrainingStepTrace] = []
        total_loss = 0.0
        max_loss = 0.0
        peak_activation_bytes = 0

        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for remaining_step in range(steps):
            current_step_index = self.engine.next_step_index
            batch = batch_planner.batch_for_step(current_step_index)
            next_batch = None
            if remaining_step + 1 < steps:
                next_batch = batch_planner.batch_for_step(current_step_index + 1)

            trace = self.engine.run_step(batch, next_batch=next_batch)
            if retain_step_traces:
                traces.append(trace)

            if trace.execution_summary is not None:
                total_loss += trace.execution_summary.total_loss
                max_loss = max(max_loss, trace.execution_summary.total_loss)
                peak_activation_bytes = max(
                    peak_activation_bytes,
                    trace.execution_summary.peak_activation_bytes,
                )

            if (
                checkpoint_dir is not None
                and checkpoint_interval > 0
                and (trace.step_index + 1) % checkpoint_interval == 0
            ):
                path = checkpoint_dir / f"step_{trace.step_index + 1:05d}.json"
                runtime_snapshot = self._freeze_runtime_snapshot_for_checkpoint(
                    self.engine.snapshot_state(),
                    checkpoint_name=path.stem,
                )
                checkpoint = TrainingSessionCheckpoint(
                    checkpoint_kind="training_session_checkpoint",
                    profile_name=self.engine.config.profile_name,
                    planner=batch_planner.planner_checkpoint(),
                    runtime_snapshot=runtime_snapshot,
                )
                path.write_text(
                    checkpoint.to_json(),
                    encoding="utf-8",
                )
                checkpoint_paths.append(str(path))

        return TrainingSessionTrace(
            profile_name=self.engine.config.profile_name,
            total_steps=steps,
            steps=tuple(traces),
            average_loss=(total_loss / steps) if steps else 0.0,
            max_loss=max_loss,
            peak_activation_bytes=peak_activation_bytes,
            checkpoint_format="training_session_checkpoint",
            checkpoint_paths=tuple(checkpoint_paths),
        )


def build_batch_planner_from_checkpoint(
    engine: FirstVersionTrainingEngine,
    planner_checkpoint: BatchPlannerCheckpoint,
) -> BatchPlanner:
    if planner_checkpoint.planner_kind != "tokenized_dataset":
        raise ValueError(
            "only tokenized_dataset planner checkpoints are supported; "
            f"got {planner_checkpoint.planner_kind}"
        )
    if planner_checkpoint.dataset_path is None:
        raise ValueError("tokenized_dataset checkpoint requires dataset_path")
    return TokenizedDatasetBatchPlanner(
        config=engine.config,
        dataset_path=planner_checkpoint.dataset_path,
        base_samples=planner_checkpoint.base_samples,
        tokens_per_sample=planner_checkpoint.tokens_per_sample,
        tokenizer_path=planner_checkpoint.tokenizer_path,
        dataset_format=planner_checkpoint.dataset_format,
        dataset_text_key=planner_checkpoint.dataset_text_key,
    )
