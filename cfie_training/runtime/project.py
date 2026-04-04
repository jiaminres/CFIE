"""Top-level runtime object for the standalone CFIE training package."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cfie_training.blueprint import TrainingBlueprint
from cfie_training.config import TrainingProjectConfig
from cfie_training.predictor_blueprint import PredictorBlueprint, build_predictor_blueprint
from cfie_training.profiles import build_profile_blueprint
from cfie_training.runtime.data import TokenizedDatasetBatchPlanner
from cfie_training.runtime.engine import FirstVersionTrainingEngine
from cfie_training.runtime.memory import TrainingMemoryPlan
from cfie_training.runtime.session import (
    SyntheticBatchPlanner,
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

    # 初始化时立即校验训练项目配置。
    def __post_init__(self) -> None:
        # 直接复用配置对象自己的校验逻辑。
        self.config.validate()

    # 生成训练基座主线蓝图。
    def build_blueprint(self) -> TrainingBlueprint:
        # 按当前 profile 输出训练基座蓝图。
        return build_profile_blueprint(self.config)

    # 生成 predictor 主线蓝图。
    def build_predictor_blueprint(self) -> PredictorBlueprint:
        # 按当前训练配置输出 predictor 蓝图。
        return build_predictor_blueprint(self.config)

    # 构造训练引擎，并按需恢复运行时快照。
    def build_engine(
        self,
        snapshot: TrainingRuntimeSnapshot | None = None,
    ) -> FirstVersionTrainingEngine:
        # 基于当前配置创建一套新的训练引擎。
        engine = FirstVersionTrainingEngine(self.config)
        # 如调用方传入快照，则立刻恢复引擎状态。
        if snapshot is not None:
            engine.load_state(snapshot)
        # 返回已经准备好的引擎实例。
        return engine

    # 规划单个 step 的执行轨迹。
    def plan_step(
        self,
        *,
        step_index: int,
        samples: int,
        tokens_per_sample: int,
    ) -> TrainingStepTrace:
        # 先按当前配置创建临时引擎。
        engine = self.build_engine()
        # 把纯数字 batch 参数包装成 BatchShape。
        return engine.plan_step(
            step_index=step_index,
            batch=BatchShape(samples=samples, tokens_per_sample=tokens_per_sample),
        )

    # 使用固定 batch 形状模拟多步训练。
    def simulate(
        self,
        *,
        steps: int,
        samples: int,
        tokens_per_sample: int,
    ) -> TrainingRunTrace:
        # 先创建一套新的训练引擎。
        engine = self.build_engine()
        # 用固定 batch 形状执行多步模拟。
        return engine.simulate(
            steps=steps,
            batch=BatchShape(samples=samples, tokens_per_sample=tokens_per_sample),
        )

    # 构造给定 batch 形状下的内存规划结果。
    def build_memory_plan(
        self,
        *,
        samples: int,
        tokens_per_sample: int,
    ) -> TrainingMemoryPlan:
        # 先创建训练引擎，以便复用引擎内的内存规划器。
        engine = self.build_engine()
        # 把纯数字 batch 形状交给引擎做内存估算。
        return engine.build_memory_plan(
            BatchShape(samples=samples, tokens_per_sample=tokens_per_sample)
        )

    # 运行完整训练会话，并根据输入选择 batch 规划器。
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
        # -----------------
        # 先构造引擎与会话 runner。
        # 用可选快照恢复训练引擎。
        engine = self.build_engine(snapshot)
        # 再基于引擎创建会话 runner。
        runner = TrainingSessionRunner(engine)

        # -----------------
        # 根据恢复点、数据集或默认参数选择 batch 规划器。
        if planner_checkpoint is not None and dataset_path is None:
            # 优先从 checkpoint 中恢复 batch 规划器。
            batch_planner = build_batch_planner_from_checkpoint(
                engine,
                planner_checkpoint,
            )
        elif dataset_path is not None:
            # 如指定数据集，则使用真实 tokenized dataset 规划器。
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
            # 否则退回到默认的合成 batch 规划器。
            batch_planner = SyntheticBatchPlanner(
                base_samples=samples,
                base_tokens_per_sample=tokens_per_sample,
            )

        # -----------------
        # 交由会话 runner 执行整段训练。
        # 把字符串 checkpoint 目录转换为 Path；未提供时保持 None。
        return runner.run(
            steps=steps,
            batch_planner=batch_planner,
            checkpoint_dir=None if checkpoint_dir is None else Path(checkpoint_dir),
            checkpoint_interval=checkpoint_interval,
            retain_step_traces=retain_step_traces,
        )
