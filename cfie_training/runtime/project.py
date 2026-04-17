"""Top-level runtime object for the standalone CFIE training package."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cfie_training.config import TrainingProjectConfig
from cfie_training.runtime.data import TokenizedDatasetBatchPlanner
from cfie_training.runtime.engine import FirstVersionTrainingEngine
from cfie_training.runtime.memory import TrainingMemoryPlan
from cfie_training.runtime.session import (
    TrainingSessionRunner,
    build_batch_planner_from_checkpoint,
)
from cfie_training.runtime.types import (
    BatchPlannerCheckpoint,
    BatchShape,
    TrainingRunTrace,
    TrainingRuntimeSnapshot,
    TrainingSessionTrace,
    TrainingStepTrace,
)


@dataclass(slots=True)
class TrainingProject:
    config: TrainingProjectConfig

    def __post_init__(self) -> None:
        self.config.validate()

    def build_engine(
        self,
        snapshot: TrainingRuntimeSnapshot | None = None,
    ) -> FirstVersionTrainingEngine:
        engine = FirstVersionTrainingEngine(self.config)
        if snapshot is not None:
            engine.load_state(snapshot)
        return engine

    def plan_step(
        self,
        *,
        step_index: int,
        samples: int,
        tokens_per_sample: int,
    ) -> TrainingStepTrace:
        engine = self.build_engine()
        return engine.plan_step(
            step_index=step_index,
            batch=BatchShape(samples=samples, tokens_per_sample=tokens_per_sample),
        )

    def simulate(
        self,
        *,
        steps: int,
        samples: int,
        tokens_per_sample: int,
    ) -> TrainingRunTrace:
        engine = self.build_engine()
        return engine.simulate(
            steps=steps,
            batch=BatchShape(samples=samples, tokens_per_sample=tokens_per_sample),
        )

    def build_memory_plan(
        self,
        *,
        samples: int,
        tokens_per_sample: int,
    ) -> TrainingMemoryPlan:
        engine = self.build_engine()
        return engine.build_memory_plan(
            BatchShape(samples=samples, tokens_per_sample=tokens_per_sample)
        )

    def train(
        self,
        *,
        steps: int,
        samples: int,
        tokens_per_sample: int,
        snapshot: TrainingRuntimeSnapshot | None = None,
        checkpoint_dir: str | None = None,
        checkpoint_interval: int = 0,
        dataset_path: str | None = None,
        tokenizer_path: str | None = None,
        dataset_format: str = "auto",
        dataset_text_key: str = "text",
        planner_checkpoint: BatchPlannerCheckpoint | None = None,
        retain_step_traces: bool = True,
    ) -> TrainingSessionTrace:
        engine = self.build_engine(snapshot)
        runner = TrainingSessionRunner(engine)

        if planner_checkpoint is not None and dataset_path is None:
            batch_planner = build_batch_planner_from_checkpoint(
                engine,
                planner_checkpoint,
            )
        elif dataset_path is not None:
            batch_planner = TokenizedDatasetBatchPlanner(
                config=self.config,
                dataset_path=dataset_path,
                base_samples=samples,
                tokens_per_sample=tokens_per_sample,
                tokenizer_path=tokenizer_path,
                dataset_format=dataset_format,
                dataset_text_key=dataset_text_key,
            )
        else:
            raise ValueError(
                "training requires a dataset-backed batch planner; "
                "pass dataset_path or resume from a dataset-backed session checkpoint"
            )

        return runner.run(
            steps=steps,
            batch_planner=batch_planner,
            checkpoint_dir=None if checkpoint_dir is None else Path(checkpoint_dir),
            checkpoint_interval=checkpoint_interval,
            retain_step_traces=retain_step_traces,
        )
