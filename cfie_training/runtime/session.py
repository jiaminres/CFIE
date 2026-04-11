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
    # 当前训练会话运行器绑定的训练引擎实例。
    engine: FirstVersionTrainingEngine

    def _freeze_runtime_snapshot_for_checkpoint(
        self,
        snapshot: TrainingRuntimeSnapshot,
        *,
        checkpoint_name: str,
    ) -> TrainingRuntimeSnapshot:
        # ------------------------------- 为 checkpoint 冻结独立的运行时量化快照目录 -------------------------------
        # 读取运行时量化配置中的 NVMe staging 根目录。
        staging_dir = self.engine.config.runtime_quantization.nvme_staging_dir

        # 读取当前快照绑定的运行时量化会话标识。
        session_id = snapshot.runtime_quantization_session_id

        # ------------------------------- 当 staging 目录或会话标识缺失时直接返回原快照 -------------------------------
        # 当未配置 staging 目录，或当前快照没有有效的量化会话标识时，无法执行冻结操作。
        if not staging_dir or not session_id:
            # 直接返回原始运行时快照。
            return snapshot

        # ------------------------------- 计算 live session 与冻结目标目录路径 -------------------------------
        # 展开用户目录并规范化 staging 根目录路径。
        base_root = Path(staging_dir).expanduser()

        # 计算当前 live 量化会话目录路径。
        live_root = base_root / session_id

        # 当 live 会话目录不存在时，说明当前没有可冻结的 staging 内容。
        if not live_root.exists():
            # 直接返回原始运行时快照。
            return snapshot

        # 基于当前 session_id 与 checkpoint 名生成冻结后的会话标识。
        frozen_session_id = f"{session_id}_{checkpoint_name}"

        # 计算冻结后的 staging 目录路径。
        frozen_root = base_root / frozen_session_id

        # ------------------------------- 清理旧冻结目录并复制当前 live 目录 -------------------------------
        # 当同名冻结目录已经存在时，先删除旧目录，避免 copytree 冲突。
        if frozen_root.exists():
            # 递归删除旧的冻结目录。
            shutil.rmtree(frozen_root)

        # 将当前 live staging 目录完整复制为新的冻结快照目录。
        shutil.copytree(live_root, frozen_root)

        # ------------------------------- 返回替换过量化会话标识的新快照对象 -------------------------------
        # 返回仅替换 runtime_quantization_session_id 后的新快照对象。
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
        # ------------------------------- 校验会话参数并初始化会话级统计容器 -------------------------------
        # 训练会话的执行步数至少必须为 1。
        if steps < 1:
            # 抛出非法步数异常。
            raise ValueError("steps must be >= 1")

        # checkpoint 保存间隔不能为负数。
        if checkpoint_interval < 0:
            # 抛出非法 checkpoint 间隔异常。
            raise ValueError("checkpoint_interval must be >= 0")

        # 用于记录本次训练会话生成的 checkpoint 文件路径。
        checkpoint_paths: list[str] = []

        # 用于保存每个 step 的训练轨迹；是否保留由 retain_step_traces 控制。
        traces: list[TrainingStepTrace] = []

        # 累加整次训练会话的总 loss。
        total_loss = 0.0

        # 记录整次训练会话中的最大 loss。
        max_loss = 0.0

        # 记录整次训练会话中的峰值激活显存占用。
        peak_activation_bytes = 0

        # ------------------------------- 按需创建 checkpoint 输出目录 -------------------------------
        # 当调用方提供 checkpoint 输出目录时，提前创建目录结构。
        if checkpoint_dir is not None:
            # 递归创建 checkpoint 目录；若目录已存在则不报错。
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------- 逐步执行训练 step 并累计会话级统计信息 -------------------------------
        # 按顺序执行本次训练会话中的每一个 step。
        for remaining_step in range(steps):
            # 读取当前引擎下一步将要执行的 step 序号。
            current_step_index = self.engine.next_step_index

            # 基于当前 step 序号生成当前 step 所需的 batch。
            batch = batch_planner.batch_for_step(current_step_index)

            # 默认假设当前 step 没有下一步 batch 可供 lookahead 使用。
            next_batch = None

            # 当当前 step 后面仍然还有剩余 step 时，预先生成下一步 batch 供引擎做 lookahead。
            if remaining_step + 1 < steps:
                # 基于下一步 step 序号生成 next_batch。
                next_batch = batch_planner.batch_for_step(current_step_index + 1)

            # ------------------------------- 调用训练引擎执行当前 step -------------------------------
            # 执行当前训练 step，并将 next_batch 传入引擎以支持 lookahead 规划。
            trace = self.engine.run_step(batch, next_batch=next_batch)

            # 当调用方要求保留 step 级轨迹时，将当前 trace 追加到会话轨迹列表中。
            if retain_step_traces:
                # 保存当前 step 的完整训练轨迹。
                traces.append(trace)

            # ------------------------------- 从执行摘要中累计 loss 与峰值激活统计 -------------------------------
            # 仅当当前 step 存在执行摘要时，才更新会话级统计量。
            if trace.execution_summary is not None:
                # 将当前 step 的总 loss 累加到会话总 loss。
                total_loss += trace.execution_summary.total_loss

                # 使用当前 step 的总 loss 更新会话最大 loss。
                max_loss = max(max_loss, trace.execution_summary.total_loss)

                # 使用当前 step 的峰值激活占用更新会话峰值激活占用。
                peak_activation_bytes = max(
                    peak_activation_bytes,
                    trace.execution_summary.peak_activation_bytes,
                )

            # ------------------------------- 命中 checkpoint 间隔时冻结运行时并写出会话 checkpoint -------------------------------
            # 当启用了 checkpoint 输出目录与保存间隔，且当前 step 命中保存时机时，写出 checkpoint。
            if (
                checkpoint_dir is not None
                and checkpoint_interval > 0
                and (trace.step_index + 1) % checkpoint_interval == 0
            ):
                # 基于当前 step 序号生成 checkpoint 文件路径。
                path = checkpoint_dir / f"step_{trace.step_index + 1:05d}.json"

                # 冻结当前运行时快照，避免后续量化 staging 目录被覆盖。
                runtime_snapshot = self._freeze_runtime_snapshot_for_checkpoint(
                    self.engine.snapshot_state(),
                    checkpoint_name=path.stem,
                )

                # 基于当前 planner 状态与运行时快照组装训练会话 checkpoint 对象。
                checkpoint = TrainingSessionCheckpoint(
                    checkpoint_kind="training_session_checkpoint",
                    profile_name=self.engine.config.profile_name,
                    planner=batch_planner.planner_checkpoint(),
                    runtime_snapshot=runtime_snapshot,
                )

                # 将 checkpoint 对象序列化为 JSON 并写入目标文件。
                path.write_text(
                    checkpoint.to_json(),
                    encoding="utf-8",
                )

                # 将当前 checkpoint 文件路径记录到会话结果中。
                checkpoint_paths.append(str(path))

        # ------------------------------- 汇总整次训练会话的统计结果并返回 -------------------------------
        # 统计本次会话实际保留下来的 step 级轨迹数量。
        total_steps = len(traces)

        # 返回训练会话级别的汇总轨迹对象。
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
