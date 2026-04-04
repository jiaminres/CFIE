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
    # 为指定 step 生成 batch 形状。
    def batch_for_step(self, step_index: int) -> BatchShape:
        ...

    # 生成可恢复的 batch 规划器快照。
    def planner_checkpoint(self) -> BatchPlannerCheckpoint:
        ...


@dataclass(slots=True)
class SyntheticBatchPlanner:
    base_samples: int
    base_tokens_per_sample: int

    # 为指定 step 构造带轻微抖动的合成 batch。
    def batch_for_step(self, step_index: int) -> BatchShape:
        # 让 token 长度按 3 步周期轻微摆动。
        token_delta = ((step_index % 3) - 1) * 32
        # 每 4 步里的最后一步额外增加 1 个样本。
        sample_delta = 1 if step_index % 4 == 3 else 0
        # 返回扰动后的合成 batch 形状。
        return BatchShape(
            samples=max(1, self.base_samples + sample_delta),
            tokens_per_sample=max(16, self.base_tokens_per_sample + token_delta),
        )

    # 为合成 batch 规划器生成 checkpoint 快照。
    def planner_checkpoint(self) -> BatchPlannerCheckpoint:
        return BatchPlannerCheckpoint(
            planner_kind="synthetic",
            base_samples=self.base_samples,
            tokens_per_sample=self.base_tokens_per_sample,
        )


@dataclass(slots=True)
class TrainingSessionRunner:
    engine: FirstVersionTrainingEngine

    # 为 checkpoint 冻结一份独立的运行时量化 staging 快照。
    def _freeze_runtime_snapshot_for_checkpoint(
        self,
        snapshot: TrainingRuntimeSnapshot,
        *,
        checkpoint_name: str,
    ) -> TrainingRuntimeSnapshot:
        # 读取运行时量化 staging 根目录配置。
        staging_dir = self.engine.config.runtime_quantization.nvme_staging_dir
        # 读取当前运行时绑定的量化 session id。
        session_id = snapshot.runtime_quantization_session_id
        # 如果没有 staging 目录或 session id，就直接返回原快照。
        if not staging_dir or not session_id:
            return snapshot
        # 展开并规范化 staging 根目录路径。
        base_root = Path(staging_dir).expanduser()
        # 组装当前 live session 目录。
        live_root = base_root / session_id
        # 如果 live session 已经不存在，也无法冻结，直接返回。
        if not live_root.exists():
            return snapshot
        # 用 checkpoint 名生成冻结后的 session id。
        frozen_session_id = f"{session_id}_{checkpoint_name}"
        # 计算冻结目录路径。
        frozen_root = base_root / frozen_session_id
        # 若同名冻结目录已存在，先删除旧目录。
        if frozen_root.exists():
            shutil.rmtree(frozen_root)
        # 复制 live 目录生成冻结快照目录。
        shutil.copytree(live_root, frozen_root)
        # 返回替换过 session id 的新快照对象。
        return replace(
            snapshot,
            runtime_quantization_session_id=frozen_session_id,
        )

    # 执行完整训练会话，并按需写出中间 checkpoint。
    def run(
        self,
        *,
        steps: int,
        batch_planner: BatchPlanner,
        checkpoint_dir: Path | None = None,
        checkpoint_interval: int = 0,
        retain_step_traces: bool = True,
    ) -> TrainingSessionTrace:
        # -----------------
        # 先校验会话级参数，并初始化统计容器。
        # 训练步数至少要大于等于 1。
        if steps < 1:
            raise ValueError("steps must be >= 1")
        # checkpoint 间隔不能为负数。
        if checkpoint_interval < 0:
            raise ValueError("checkpoint_interval must be >= 0")
        # 记录本次会话生成的 checkpoint 路径。
        checkpoint_paths: list[str] = []
        # 保存每个 step 的轨迹，是否保留由 retain_step_traces 决定。
        traces: list[TrainingStepTrace] = []
        # 累计 loss，后面用于计算平均值。
        total_loss = 0.0
        # 记录整次会话的最大 loss。
        max_loss = 0.0
        # 记录整次会话的峰值激活占用。
        peak_activation_bytes = 0

        # 如指定 checkpoint 输出目录，则提前创建目录。
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # -----------------
        # 逐步运行训练 step，并累计 loss / 激活峰值等统计。
        for remaining_step in range(steps):
            # 读取引擎当前的下一个 step 序号。
            current_step_index = self.engine.next_step_index
            # 为当前 step 生成 batch。
            batch = batch_planner.batch_for_step(current_step_index)
            # 默认假设本轮没有 next_batch。
            next_batch = None
            # 只有后面还剩 step 时，才预先生成 next_batch。
            if remaining_step + 1 < steps:
                next_batch = batch_planner.batch_for_step(current_step_index + 1)
            # 执行当前 step，并把 next_batch 传给引擎做 lookahead。
            trace = self.engine.run_step(batch, next_batch=next_batch)
            # 如要求保留 step 轨迹，则把当前 trace 存起来。
            if retain_step_traces:
                traces.append(trace)
            # 只有执行摘要存在时，才累计 loss 与激活统计。
            if trace.execution_summary is not None:
                # 把本步总 loss 累加到会话总 loss。
                total_loss += trace.execution_summary.total_loss
                # 用本步 loss 更新全局最大值。
                max_loss = max(max_loss, trace.execution_summary.total_loss)
                # 用本步峰值激活更新全局峰值。
                peak_activation_bytes = max(
                    peak_activation_bytes,
                    trace.execution_summary.peak_activation_bytes,
                )

            # -----------------
            # 命中 checkpoint 间隔时，冻结运行时并写出会话快照。
            if (
                checkpoint_dir is not None
                and checkpoint_interval > 0
                and (trace.step_index + 1) % checkpoint_interval == 0
            ):
                # 为当前 step 生成 checkpoint 文件名。
                path = checkpoint_dir / f"step_{trace.step_index + 1:05d}.json"
                # 先冻结运行时快照，避免量化 staging 后续被覆盖。
                runtime_snapshot = self._freeze_runtime_snapshot_for_checkpoint(
                    self.engine.snapshot_state(),
                    checkpoint_name=path.stem,
                )
                # 组装训练会话 checkpoint 对象。
                checkpoint = TrainingSessionCheckpoint(
                    checkpoint_kind="training_session_checkpoint",
                    profile_name=self.engine.config.profile_name,
                    planner=batch_planner.planner_checkpoint(),
                    runtime_snapshot=runtime_snapshot,
                )
                # 把 checkpoint 写到磁盘。
                path.write_text(
                    checkpoint.to_json(),
                    encoding="utf-8",
                )
                # 记录 checkpoint 路径，供最终 trace 汇总。
                checkpoint_paths.append(str(path))

        # -----------------
        # 汇总整个训练会话的统计结果。
        # 记录本次会话实际保留下来的 step trace 数量。
        total_steps = len(traces)
        # 返回会话级别的汇总轨迹对象。
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


# 根据 checkpoint 中记录的 planner 类型恢复 batch 规划器。
def build_batch_planner_from_checkpoint(
    engine: FirstVersionTrainingEngine,
    planner_checkpoint: BatchPlannerCheckpoint,
) -> BatchPlanner:
    # synthetic checkpoint 直接恢复为合成规划器。
    if planner_checkpoint.planner_kind == "synthetic":
        return SyntheticBatchPlanner(
            base_samples=planner_checkpoint.base_samples,
            base_tokens_per_sample=planner_checkpoint.tokens_per_sample,
        )
    # tokenized_dataset checkpoint 恢复为真实数据集规划器。
    if planner_checkpoint.planner_kind == "tokenized_dataset":
        # 数据集路径是恢复 tokenized_dataset 规划器的必要字段。
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
    # 其他 planner_kind 当前都视为不支持。
    raise ValueError(
        f"unsupported planner_kind in checkpoint: {planner_checkpoint.planner_kind}"
    )
