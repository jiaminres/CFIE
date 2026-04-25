"""Predictor data models used by the CFIE training stack."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from cfie.outputs import CompletionOutput, RequestOutput


@dataclass(slots=True, frozen=True)
class CapturedForwardBatch:
    """一次真实 teacher 前向捕获后的按层监督数据。

    设：
    - L = 模型层数
    - T = 当前 batch 内所有 request 的有效 prompt token 总数
    - H = hidden size
    - K = routed top-k 专家数
    """

    # 按层保存 hidden state；tuple 长度为 L，每个 tensor 形状为 [T, H]。
    layer_hidden_states: tuple[torch.Tensor, ...]

    # 按层保存 teacher router top-k 专家 id；tuple 长度为 L，每个 tensor 形状为 [T, K]。
    layer_teacher_topk_ids: tuple[torch.Tensor, ...]


@dataclass(slots=True, frozen=True)
class CapturedHiddenStatePayload:
    """trainer 内部使用的标准 hidden-state 捕获载荷。

    worker 侧必须返回结构化字典协议；trainer 侧收到后立即转换成该对象。
    这样合并逻辑只面对具名字段，不再传递难读的多元素 tuple，也不会再
    根据返回位置猜测层号或 rank 语义。
    """

    # 当前 payload 对应的请求 id，用于把多 worker 返回结果归并回同一个 request。
    request_id: str

    # 当前 worker 实际返回的层号集合；tuple 长度必须与 hidden_states 一致。
    layer_ids: tuple[int, ...]

    # 当前 worker 对应层段捕获到的 hidden states；
    # tuple 长度等于 len(layer_ids)，每个 tensor 通常形状为 [request_token_count, hidden_size]。
    hidden_states: tuple[torch.Tensor, ...]

    # pipeline parallel rank，用于区分不同 PP stage 返回的不同层段。
    pp_rank: int

    # tensor parallel rank，用于稳定处理同一层的 TP 重复副本。
    tp_rank: int


@dataclass(slots=True, frozen=True)
class PredictorCaptureCompletion:
    """训练侧保留的单条 completion 快照。

    predictor teacher 训练最终只会消费 `outputs[0].routed_experts`，但直接把
    推理引擎的完整 `CompletionOutput` 挂在 trainer 本地状态里，会让阅读者不得不
    跳到推理侧输出模型才能看懂当前链路到底依赖了哪些字段。

    这里显式定义一份训练侧快照，只保留当前 teacher capture 确实会读到的稳定字段，
    这样 `PredictorCaptureRequest` 的状态边界就会更清楚：
    - 当前 completion 是第几个候选；
    - 对应生成 token ids 是什么；
    - 当前 completion 是否带回 routed experts；
    - 当前 completion 的结束原因是什么。
    """

    # 当前 completion 在 request 输出列表中的序号。
    index: int
    # 当前 completion 已生成的 token ids。
    token_ids: tuple[int, ...]
    # teacher 返回的 routed experts，形状通常为 [seq_len, num_layers, topk]。
    routed_experts: np.ndarray | None
    # 当前 completion 的结束原因。
    finish_reason: str | None
    # 当前 completion 的 stop 原因。
    stop_reason: int | str | None

    @classmethod
    def from_engine_completion(
            cls,
            completion: CompletionOutput,
    ) -> "PredictorCaptureCompletion":
        # routed_experts 可能来自共享底层缓冲；这里显式复制一份只读快照，
        # 避免后续引擎继续推进时 trainer 侧观察到被原地覆盖的内容。
        routed_experts = (
            None
            if completion.routed_experts is None
            else np.array(completion.routed_experts, copy=True)
        )
        return cls(
            index=int(completion.index),
            token_ids=tuple(int(token_id) for token_id in completion.token_ids),
            routed_experts=routed_experts,
            finish_reason=completion.finish_reason,
            stop_reason=completion.stop_reason,
        )


@dataclass(slots=True, frozen=True)
class PredictorCaptureOutput:
    """训练侧保留的 request 输出快照。

    `capture_batch` 不需要持有推理引擎完整 `RequestOutput` 的全部字段；真正依赖的
    只有 request_id、是否结束以及 completion 列表。这里把它们收敛成训练侧模型，
    让 predictor 链路读起来只围绕“teacher 训练需要什么”展开，而不是被推理侧
    大而全的输出对象分散注意力。
    """

    # 当前 request 的唯一标识。
    request_id: str
    # 当前 request 是否已经产生最终输出。
    finished: bool
    # 当前 request 的 completion 快照集合。
    outputs: tuple[PredictorCaptureCompletion, ...]

    @classmethod
    def from_engine_output(
            cls,
            request_output: RequestOutput,
    ) -> "PredictorCaptureOutput":
        return cls(
            request_id=str(request_output.request_id),
            finished=bool(request_output.finished),
            outputs=tuple(
                PredictorCaptureCompletion.from_engine_completion(completion)
                for completion in request_output.outputs
            ),
        )


@dataclass(slots=True)
class PredictorCaptureRequest:
    """一次 teacher capture 请求在 trainer 侧的本地状态。

    capture_batch 同时需要跟踪 request_id、真实 prompt、最终输出对象和按层
    hidden states。如果这些信息分散在多个 `dict[str, ...]` 里，后续维护时很
    难判断哪些字段必须同时存在。这里用具名对象把同一个 request 的状态收拢
    到一起，让提交、step 回收、hidden-state 合并和最终校验都围绕同一份状态。
    """

    # 当前请求的唯一标识。
    request_id: str
    # 当前请求在输入 batch 里的行号。
    row_index: int
    # 当前请求对应的原始 token 序列。
    prompt_row: tuple[int, ...]
    # trainer 侧冻结后的最终输出快照。
    output: PredictorCaptureOutput | None = None
    # 当前请求聚合完成的按层 hidden states。
    hidden_states: tuple[torch.Tensor, ...] | None = None


@dataclass(slots=True, frozen=True)
class PredictorExampleSpec:
    """单个 predictor 监督样本的轻量级选点规格。

    该对象只描述“后续应该从哪里取数据”，不直接保存 hidden state 或 teacher
    标签。trace builder 会先在 token 与插入层组成的二维槽位空间中生成这些
    规格，再根据规格去 `CapturedForwardBatch` 中读取真实前向捕获结果。
    """

    # 当前 step 内的局部样本序号，用来稳定保留本轮采样顺序。
    example_offset: int

    # predictor 插入层号；后续会读取该层、该 token 的 hidden state 作为输入特征。
    insertion_layer_index: int

    # 当前捕获 batch 内的展平 token 序号；后续会用它对齐 hidden state 与 teacher 标签。
    token_index: int


@dataclass(slots=True, frozen=True)
class PredictorTraceExample:
    # 当前样本在整个 trace 数据集里的全局编号。
    example_index: int
    # 当前样本来自哪个 trace step。
    step_index: int
    # 当前样本对应捕获 batch 内的展平 token 序号。
    token_index: int
    # predictor 读取输入 hidden state 的插入层号。
    insertion_layer_index: int
    # 当前样本需要预测的未来层号列表。
    future_layer_indices: tuple[int, ...]
    # 当前插入层位置的 hidden state 摘要。
    hidden_state: tuple[float, ...]
    # 每个未来层对应的 teacher top-k experts。
    future_teacher_topk_ids: tuple[tuple[int, ...], ...]

    # 将单条 predictor trace 样本序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 输出样本编号、插入层、hidden_state 和未来 teacher top-k 标签。
        return {
            "example_index": self.example_index,
            "step_index": self.step_index,
            "token_index": self.token_index,
            "insertion_layer_index": self.insertion_layer_index,
            "future_layer_indices": list(self.future_layer_indices),
            "hidden_state": list(self.hidden_state),
            "future_teacher_topk_ids": [
                list(expert_ids) for expert_ids in self.future_teacher_topk_ids
            ],
        }

    @classmethod
    # 从字典恢复单条 predictor trace 样本。
    def from_dict(cls, payload: dict[str, Any]) -> "PredictorTraceExample":
        # 逐字段恢复样本索引、未来层索引、hidden_state 和 top-k 标签；
        # 同时兼容旧版数据里的 hidden_summary 字段。
        return cls(
            example_index=int(payload["example_index"]),
            step_index=int(payload["step_index"]),
            token_index=int(payload.get("token_index", 0)),
            insertion_layer_index=int(payload["insertion_layer_index"]),
            future_layer_indices=tuple(
                int(layer_index) for layer_index in payload["future_layer_indices"]
            ),
            hidden_state=tuple(
                float(value)
                for value in payload.get(
                    "hidden_state",
                    payload.get("hidden_summary", ()),
                )
            ),
            future_teacher_topk_ids=tuple(
                tuple(int(expert_id) for expert_id in expert_ids)
                for expert_ids in payload["future_teacher_topk_ids"]
            ),
        )


@dataclass(slots=True, frozen=True)
class PredictorTraceDataset:
    # 当前数据集所属的训练档位名。
    profile_name: str
    # 当前数据集记录的样本总数。
    example_count: int
    # 每条样本覆盖的未来窗口层数。
    window_layers: int
    # 每层候选 expert 预算口径。
    candidate_experts_per_layer: int
    # 每层实际执行 expert 预算口径。
    executed_experts_per_layer: int
    # 当前数据集包含的全部 trace 样本。
    examples: tuple[PredictorTraceExample, ...]

    # 将 predictor trace 数据集序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 输出数据集来源、窗口配置和所有样本。
        return {
            "profile_name": self.profile_name,
            "example_count": self.example_count,
            "window_layers": self.window_layers,
            "candidate_experts_per_layer": self.candidate_experts_per_layer,
            "executed_experts_per_layer": self.executed_experts_per_layer,
            "examples": [example.to_dict() for example in self.examples],
        }

    # 将 predictor trace 数据集导出为 JSON。
    def to_json(self, *, indent: int = 2) -> str:
        # 使用稳定排序键和指定缩进导出 JSON 文本。
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    # 将 predictor trace 数据集写入 JSON 文件。
    def write_json(self, path: str | Path, *, indent: int = 2) -> None:
        # 直接把 JSON 文本写到目标路径。
        Path(path).write_text(self.to_json(indent=indent), encoding="utf-8")

    @classmethod
    # 从字典恢复 predictor trace 数据集。
    def from_dict(cls, payload: dict[str, Any]) -> "PredictorTraceDataset":
        # 先逐条恢复 examples 列表。
        examples = tuple(
            PredictorTraceExample.from_dict(example)
            for example in payload.get("examples", [])
        )
        # 再恢复数据集元信息和 examples 元组。
        return cls(
            profile_name=str(payload["profile_name"]),
            example_count=int(payload.get("example_count", len(examples))),
            window_layers=int(payload["window_layers"]),
            candidate_experts_per_layer=int(payload["candidate_experts_per_layer"]),
            executed_experts_per_layer=int(payload["executed_experts_per_layer"]),
            examples=examples,
        )

    @classmethod
    # 从 JSON 文件恢复 predictor trace 数据集。
    def from_json_file(cls, path: str | Path) -> "PredictorTraceDataset":
        # 读取并解析 JSON 文本。
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        # 顶层必须解析成对象。
        if not isinstance(payload, dict):
            raise ValueError("predictor trace dataset JSON must decode to an object")
        # 继续按字典格式恢复数据集。
        return cls.from_dict(payload)


@dataclass(slots=True, frozen=True)
class PredictorEpochSummary:
    # 当前汇总对应的 epoch 编号。
    epoch_index: int
    # 当前 epoch 在全量数据上的平均损失。
    mean_loss: float
    # 当前 epoch 的候选预算 recall。
    recall_at_candidate_budget: float
    # 当前 epoch 的执行预算 recall。
    recall_at_executed_budget: float

    # 将单个 epoch 汇总序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 输出 loss 和两档 recall 指标。
        return {
            "epoch_index": self.epoch_index,
            "mean_loss": self.mean_loss,
            "recall_at_candidate_budget": self.recall_at_candidate_budget,
            "recall_at_executed_budget": self.recall_at_executed_budget,
        }

    @classmethod
    # 从字典恢复单个 epoch 汇总。
    def from_dict(cls, payload: dict[str, Any]) -> "PredictorEpochSummary":
        # 逐字段恢复 epoch 编号、loss 与两档 recall。
        return cls(
            epoch_index=int(payload["epoch_index"]),
            mean_loss=float(payload["mean_loss"]),
            recall_at_candidate_budget=float(
                payload["recall_at_candidate_budget"]
            ),
            recall_at_executed_budget=float(
                payload["recall_at_executed_budget"]
            ),
        )


@dataclass(slots=True, frozen=True)
class PredictorTrainingRunTrace:
    # 当前训练轨迹所属的训练档位名。
    profile_name: str
    # 本次训练使用的数据集样本总数。
    example_count: int
    # 当前轨迹累计完成的 epoch 数。
    epochs: int
    # 当前轨迹采用的候选 expert 预算口径。
    candidate_experts_per_layer: int
    # 当前轨迹采用的执行 expert 预算口径。
    executed_experts_per_layer: int
    # 当前轨迹记录的逐 epoch 指标汇总。
    epoch_summaries: tuple[PredictorEpochSummary, ...]

    @property
    # 返回最后一个 epoch 的平均损失。
    def final_mean_loss(self) -> float:
        # 没有 epoch 时退回 0。
        return self.epoch_summaries[-1].mean_loss if self.epoch_summaries else 0.0

    @property
    # 返回最后一个 epoch 的 candidate budget recall。
    def final_recall_at_candidate_budget(self) -> float:
        # 没有 epoch 时退回 0。
        if not self.epoch_summaries:
            return 0.0
        # 否则取最后一个 epoch 的 recall。
        return self.epoch_summaries[-1].recall_at_candidate_budget

    @property
    # 返回最后一个 epoch 的 executed budget recall。
    def final_recall_at_executed_budget(self) -> float:
        # 没有 epoch 时退回 0。
        if not self.epoch_summaries:
            return 0.0
        # 否则取最后一个 epoch 的 recall。
        return self.epoch_summaries[-1].recall_at_executed_budget

    # 将整个 predictor 训练 run 序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 输出训练来源、epoch 数、最终指标和逐 epoch 汇总。
        return {
            "profile_name": self.profile_name,
            "example_count": self.example_count,
            "epochs": self.epochs,
            "candidate_experts_per_layer": self.candidate_experts_per_layer,
            "executed_experts_per_layer": self.executed_experts_per_layer,
            "final_mean_loss": self.final_mean_loss,
            "final_recall_at_candidate_budget": (
                self.final_recall_at_candidate_budget
            ),
            "final_recall_at_executed_budget": (
                self.final_recall_at_executed_budget
            ),
            "epoch_summaries": [
                summary.to_dict() for summary in self.epoch_summaries
            ],
        }

    # 将 predictor 训练 run 导出为 JSON。
    def to_json(self, *, indent: int = 2) -> str:
        # 使用稳定排序键和指定缩进导出 JSON 文本。
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    # 从字典恢复 predictor 训练 run。
    def from_dict(cls, payload: dict[str, Any]) -> "PredictorTrainingRunTrace":
        # 先恢复逐 epoch 汇总列表。
        epoch_summaries = tuple(
            PredictorEpochSummary.from_dict(summary)
            for summary in payload.get("epoch_summaries", [])
        )
        # 再恢复训练来源、预算和总 epoch 数。
        return cls(
            profile_name=str(payload["profile_name"]),
            example_count=int(payload["example_count"]),
            epochs=int(payload["epochs"]),
            candidate_experts_per_layer=int(
                payload["candidate_experts_per_layer"]
            ),
            executed_experts_per_layer=int(
                payload["executed_experts_per_layer"]
            ),
            epoch_summaries=epoch_summaries,
        )


@dataclass(slots=True, frozen=True)
class PredictorCheckpointMetadata:
    # 当前文件的 checkpoint 类型标识。
    checkpoint_kind: str
    # 当前 checkpoint 对应的训练档位名。
    profile_name: str
    # predictor 输入摘要维度。
    input_summary_dim: int
    # predictor 内部隐层宽度。
    hidden_dim: int
    # predictor 预测的未来窗口层数。
    window_layers: int
    # 未来窗口在层维度上的步长。
    stride_layers: int
    # 当前模型的 expert 总数。
    num_experts: int
    # 每层候选 expert 预算口径。
    candidate_experts_per_layer: int
    # 每层执行 expert 预算口径。
    executed_experts_per_layer: int
    # 运行时候选筛选策略。
    selection_mode: str
    # 在线 expert 来源策略。
    online_expert_source: str
    # 是否允许候选集合不完全匹配。
    allow_candidate_mismatch: bool
    # 训练时使用的数据集样本总数。
    example_count: int
    # checkpoint 保存时累计完成的 epoch 数。
    epochs: int
    # checkpoint 保存时的最终平均损失。
    final_mean_loss: float
    # checkpoint 保存时的最终候选预算 recall。
    final_recall_at_candidate_budget: float
    # checkpoint 保存时的最终执行预算 recall。
    final_recall_at_executed_budget: float

    # 将 predictor checkpoint 元信息序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 输出 checkpoint 对运行时兼容性和最终指标有影响的全部字段。
        return {
            "checkpoint_kind": self.checkpoint_kind,
            "profile_name": self.profile_name,
            "input_summary_dim": self.input_summary_dim,
            "hidden_dim": self.hidden_dim,
            "window_layers": self.window_layers,
            "stride_layers": self.stride_layers,
            "num_experts": self.num_experts,
            "candidate_experts_per_layer": self.candidate_experts_per_layer,
            "executed_experts_per_layer": self.executed_experts_per_layer,
            "selection_mode": self.selection_mode,
            "online_expert_source": self.online_expert_source,
            "allow_candidate_mismatch": self.allow_candidate_mismatch,
            "example_count": self.example_count,
            "epochs": self.epochs,
            "final_mean_loss": self.final_mean_loss,
            "final_recall_at_candidate_budget": (
                self.final_recall_at_candidate_budget
            ),
            "final_recall_at_executed_budget": (
                self.final_recall_at_executed_budget
            ),
        }

    @classmethod
    # 从字典恢复 predictor checkpoint 元信息。
    def from_dict(cls, payload: dict[str, Any]) -> "PredictorCheckpointMetadata":
        # 逐字段恢复 checkpoint 兼容性约束和最终指标。
        return cls(
            checkpoint_kind=str(
                payload.get("checkpoint_kind", "cfie_predictor_checkpoint")
            ),
            profile_name=str(payload["profile_name"]),
            input_summary_dim=int(payload["input_summary_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            window_layers=int(payload["window_layers"]),
            stride_layers=int(payload["stride_layers"]),
            num_experts=int(payload["num_experts"]),
            candidate_experts_per_layer=int(
                payload["candidate_experts_per_layer"]
            ),
            executed_experts_per_layer=int(
                payload["executed_experts_per_layer"]
            ),
            selection_mode=str(payload["selection_mode"]),
            online_expert_source=str(payload["online_expert_source"]),
            allow_candidate_mismatch=bool(
                payload.get("allow_candidate_mismatch", True)
            ),
            example_count=int(payload["example_count"]),
            epochs=int(payload["epochs"]),
            final_mean_loss=float(payload["final_mean_loss"]),
            final_recall_at_candidate_budget=float(
                payload["final_recall_at_candidate_budget"]
            ),
            final_recall_at_executed_budget=float(
                payload.get("final_recall_at_executed_budget", 0.0)
            ),
        )


@dataclass(slots=True, frozen=True)
class PredictorEvaluationTrace:
    # 当前评估结果所属的训练档位名。
    profile_name: str
    # 本次评估使用的数据集样本总数。
    example_count: int
    # 本次评估采用的候选 expert 预算口径。
    candidate_experts_per_layer: int
    # 本次评估采用的执行 expert 预算口径。
    executed_experts_per_layer: int
    # 本次评估的平均损失。
    mean_loss: float
    # 本次评估的候选预算 recall。
    recall_at_candidate_budget: float
    # 本次评估的执行预算 recall。
    recall_at_executed_budget: float
    # 参与本次评估的可选 checkpoint 元信息。
    checkpoint_metadata: PredictorCheckpointMetadata | None = None

    # 将 predictor 评估结果序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 输出评估来源、loss、recall，以及可选的 checkpoint 元信息。
        return {
            "profile_name": self.profile_name,
            "example_count": self.example_count,
            "candidate_experts_per_layer": self.candidate_experts_per_layer,
            "executed_experts_per_layer": self.executed_experts_per_layer,
            "mean_loss": self.mean_loss,
            "recall_at_candidate_budget": self.recall_at_candidate_budget,
            "recall_at_executed_budget": self.recall_at_executed_budget,
            "checkpoint_metadata": (
                None
                if self.checkpoint_metadata is None
                else self.checkpoint_metadata.to_dict()
            ),
        }

    # 将 predictor 评估结果导出为 JSON。
    def to_json(self, *, indent: int = 2) -> str:
        # 使用稳定排序键和指定缩进导出 JSON 文本。
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


__all__ = [
    "CapturedForwardBatch",
    "CapturedHiddenStatePayload",
    "PredictorCaptureCompletion",
    "PredictorCaptureRequest",
    "PredictorCaptureOutput",
    "PredictorCheckpointMetadata",
    "PredictorEpochSummary",
    "PredictorEvaluationTrace",
    "PredictorExampleSpec",
    "PredictorTraceDataset",
    "PredictorTraceExample",
    "PredictorTrainingRunTrace",
]
