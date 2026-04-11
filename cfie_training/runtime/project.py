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
    # 当前训练项目持有的配置对象，用于驱动蓝图构造、引擎创建与训练会话执行。
    config: TrainingProjectConfig

    # ------------------------------- 在对象初始化完成后校验训练项目配置 -------------------------------
    # 在 dataclass 完成字段注入后，立即对训练配置做统一合法性校验。
    def __post_init__(self) -> None:
        # 调用配置对象自身的校验逻辑，确保当前训练项目配置可用。
        self.config.validate()

    # ------------------------------- 构造训练基座主线蓝图 -------------------------------
    # 基于当前训练项目配置生成训练基座主线蓝图。
    def build_blueprint(self) -> TrainingBlueprint:
        # 按当前配置构造并返回训练基座蓝图对象。
        return build_profile_blueprint(self.config)

    # ------------------------------- 构造 predictor 主线蓝图 -------------------------------
    # 基于当前训练项目配置生成 predictor 蓝图。
    def build_predictor_blueprint(self) -> PredictorBlueprint:
        # 按当前配置构造并返回 predictor 蓝图对象。
        return build_predictor_blueprint(self.config)

    # ------------------------------- 构造训练引擎并按需恢复运行时状态 -------------------------------
    # 基于当前配置创建训练引擎；当提供运行时快照时，同时恢复引擎状态。
    def build_engine(
        self,
        snapshot: TrainingRuntimeSnapshot | None = None,
    ) -> FirstVersionTrainingEngine:
        # 基于当前训练项目配置创建新的训练引擎实例。
        engine = FirstVersionTrainingEngine(self.config)

        # ------------------------------- 按需从运行时快照恢复引擎状态 -------------------------------
        # 当调用方提供运行时快照时，将该快照加载到新建引擎中。
        if snapshot is not None:
            # 将快照中的运行时状态恢复到当前训练引擎。
            engine.load_state(snapshot)

        # ------------------------------- 返回已准备完成的训练引擎实例 -------------------------------
        # 返回已经构造完成且可能已恢复状态的训练引擎。
        return engine

    # ------------------------------- 规划单个训练 step 的执行轨迹 -------------------------------
    # 基于给定的 step 编号与 batch 形状，规划单步训练执行轨迹。
    def plan_step(
        self,
        *,
        step_index: int,
        samples: int,
        tokens_per_sample: int,
    ) -> TrainingStepTrace:
        # 基于当前训练项目配置创建临时训练引擎。
        engine = self.build_engine()

        # ------------------------------- 将标量 batch 参数封装为 BatchShape 并交给引擎规划 -------------------------------
        # 调用训练引擎的单步规划接口，并返回对应的 step 级执行轨迹。
        return engine.plan_step(
            step_index=step_index,
            batch=BatchShape(samples=samples, tokens_per_sample=tokens_per_sample),
        )

    # ------------------------------- 使用固定 batch 形状模拟多步训练 -------------------------------
    # 基于给定的步数与 batch 形状，执行多步训练模拟。
    def simulate(
        self,
        *,
        steps: int,
        samples: int,
        tokens_per_sample: int,
    ) -> TrainingRunTrace:
        # 基于当前训练项目配置创建新的训练引擎。
        engine = self.build_engine()

        # ------------------------------- 将标量 batch 参数封装为 BatchShape 并执行多步模拟 -------------------------------
        # 调用训练引擎的模拟接口，并返回整段多步训练模拟轨迹。
        return engine.simulate(
            steps=steps,
            batch=BatchShape(samples=samples, tokens_per_sample=tokens_per_sample),
        )

    # ------------------------------- 构造指定 batch 形状下的内存规划结果 -------------------------------
    # 基于给定的样本数与每样本 token 数，生成对应的训练内存规划结果。
    def build_memory_plan(
        self,
        *,
        samples: int,
        tokens_per_sample: int,
    ) -> TrainingMemoryPlan:
        # 基于当前训练项目配置创建训练引擎，以复用引擎内部的内存规划能力。
        engine = self.build_engine()

        # ------------------------------- 将标量 batch 参数封装为 BatchShape 并执行内存估算 -------------------------------
        # 调用训练引擎的内存规划接口，并返回对应的内存规划结果。
        return engine.build_memory_plan(
            BatchShape(samples=samples, tokens_per_sample=tokens_per_sample)
        )

    # ------------------------------- 运行完整训练会话并按输入条件选择 batch 规划器 -------------------------------
    # 执行多步训练会话，并根据恢复点、数据集输入或默认参数自动选择 batch 规划器。
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
        # ------------------------------- 构造训练引擎与训练会话运行器 -------------------------------
        # 基于当前训练项目配置与可选运行时快照创建训练引擎。
        engine = self.build_engine(snapshot)

        # 基于训练引擎创建训练会话运行器，用于驱动多步训练执行。
        runner = TrainingSessionRunner(engine)

        # ------------------------------- 根据恢复点、数据集输入或默认参数选择 batch 规划器 -------------------------------
        # 当提供 planner checkpoint 且未指定数据集路径时，优先从 checkpoint 恢复 batch 规划器状态。
        if planner_checkpoint is not None and dataset_path is None:
            # 从 planner checkpoint 中恢复 batch 规划器，以延续先前的批次规划状态。
            batch_planner = build_batch_planner_from_checkpoint(
                engine,
                planner_checkpoint,
            )
        # 当显式提供数据集路径时，使用基于真实 tokenized 数据集的 batch 规划器。
        elif dataset_path is not None:
            # 构造基于 tokenizer 与数据集文件的 batch 规划器。
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
            # 当既没有 planner checkpoint 也没有数据集路径时，退回到默认的合成 batch 规划器。
            batch_planner = SyntheticBatchPlanner(
                base_samples=samples,
                base_tokens_per_sample=tokens_per_sample,
            )

        # ------------------------------- 调用会话运行器执行完整训练流程 -------------------------------
        # 调用训练会话运行器执行多步训练，并将字符串形式的 checkpoint 目录转换为 Path 对象。
        return runner.run(
            steps=steps,
            batch_planner=batch_planner,
            checkpoint_dir=None if checkpoint_dir is None else Path(checkpoint_dir),
            checkpoint_interval=checkpoint_interval,
            retain_step_traces=retain_step_traces,
        )
