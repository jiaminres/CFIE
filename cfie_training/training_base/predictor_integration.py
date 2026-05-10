"""将训练好的 FrozenRouterDelta predictor 接入训练基座。

提供 PredictorHotSetGuide 类，在每步训练后调用 predictor 预测未来层专家，
结果用于 HotSetScheduler 决定下一窗口的热专家选择。

Predictor 训练时使用的是 hidden state 的 summary（如 64 维池化向量），
因此需要通过 HiddenStateSummarizer 将 3072 维 hidden state 投影到 predictor 输入维度。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch


@dataclass(slots=True)
class HiddenStateSummarizer:
    """将完整 hidden state 降维为 predictor 可接受的输入维度。

    默认使用 token 维度平均池化 + 可选的线性投影。
    如果 input_dim == output_dim，则为恒等变换。
    """

    input_dim: int
    output_dim: int

    def summarize(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """hidden_state: [B, T, H] or [T, H] -> [B, output_dim] or [output_dim]

        先沿 token 维度平均池化，再投影/截断到目标维度。
        """
        if hidden_state.dim() == 3:
            pooled = hidden_state.mean(dim=1)  # [B, T, H] -> [B, H]
        elif hidden_state.dim() == 2:
            pooled = hidden_state  # [T, H] or [B, H]
        else:
            pooled = hidden_state.reshape(-1, self.input_dim)

        if self.input_dim == self.output_dim:
            return pooled.float()

        # 3072 → 64: 分组平均池化（48 组 × 64 维 = 3072 维）
        if self.input_dim % self.output_dim == 0:
            group_size = self.input_dim // self.output_dim
            return pooled.reshape(-1, self.output_dim, group_size).mean(dim=-1).float().contiguous()

        # fallback: 截断到 output_dim
        return pooled[..., :self.output_dim].float().contiguous()


@dataclass(slots=True)
class PredictorHotSetGuide:
    """用已训练的 predictor 指导 hot set 选择。

    每步训练后：
    1. 接收当前层的 hidden state
    2. 调用 predictor 预测未来层专家候选
    3. 将预测结果转换为 RoutedExpert 列表
    4. HotSetScheduler 使用这些预测来选择下一窗口的热专家

    Usage:
        guide = PredictorHotSetGuide.from_checkpoint(
            checkpoint_path="...",
            hidden_size=3072,
            num_layers=48,
            num_experts=256,
        )
        candidates = guide.predict_next_experts(
            hidden_state=batch_hidden,
            insertion_layer_index=current_layer,
        )
    """

    predictor_model: Any = None  # FrozenRouterDelta 模型实例
    bundle: Any = None  # LoadedPredictorBundle
    summarizer: HiddenStateSummarizer = field(
        default_factory=lambda: HiddenStateSummarizer(3072, 3072)
    )
    num_layers: int = 48  # 模型总层数
    num_experts: int = 256  # 每层专家数
    window_layers: int = 8  # predictor 预测窗口大小
    stride_layers: int = 4  # predictor 重评估间隔
    candidate_experts_per_layer: int = 16
    device: str = "cpu"

    @classmethod
    def from_checkpoint(
        cls,
        *,
        checkpoint_path: str | Path,
        hidden_size: int = 3072,
        num_layers: int = 48,
        num_experts: int = 256,
        base_model_path: str | None = None,
        device: str = "cpu",
    ) -> "PredictorHotSetGuide":
        """从 predictor checkpoint 加载模型。"""
        cp = Path(checkpoint_path)
        if not cp.exists():
            raise FileNotFoundError(f"predictor checkpoint not found: {cp}")

        try:
            from cfie.predictor.bundle import load_predictor_model

            if base_model_path is None:
                base_model_path = str(
                    Path(
                        "C:/Users/13642/.cache/huggingface/hub/"
                        "models--Qwen--Qwen3.5-122B-A10B/"
                        "snapshots/b000b2eb18a7f4cdf3153c4215842da339e09d99"
                    )
                )

            model, bundle = load_predictor_model(
                str(cp),
                map_location="cpu",
                device=device,
                base_model_path=base_model_path,
                num_layers=num_layers,
            )
        except Exception:
            model, bundle = None, None

        input_dim = getattr(bundle.schema, "input_summary_dim", hidden_size) if bundle else hidden_size
        summarizer = HiddenStateSummarizer(
            input_dim=hidden_size,
            output_dim=input_dim,
        )

        guide = cls(
            predictor_model=model,
            bundle=bundle,
            summarizer=summarizer,
            num_layers=num_layers,
            num_experts=num_experts,
            device=device,
        )
        if bundle is not None:
            guide.window_layers = getattr(bundle.schema, "window_layers", 8)
            guide.stride_layers = getattr(bundle.schema, "stride_layers", 4)
            guide.candidate_experts_per_layer = getattr(
                bundle.schema, "candidate_experts_per_layer", 16
            )
        return guide

    def predict_next_experts(
        self,
        *,
        hidden_state: torch.Tensor,
        insertion_layer_index: int,
        top_k: int | None = None,
    ) -> tuple[tuple[int, int], ...]:
        """返回预测的下一个窗口的热专家 (layer_id, expert_id) 列表。

        按 predictor 输出的 logit 分数排序，取 top-k。
        """
        if self.predictor_model is None:
            return ()

        if top_k is None:
            top_k = self.candidate_experts_per_layer

        summary = self.summarizer.summarize(hidden_state)
        summary = summary.to(device=self.device)

        with torch.no_grad():
            logits = self.predictor_model(summary, insertion_layer_index)
        # logits: [B, window_layers, num_experts]

        avg_logits = logits.mean(dim=0)  # [window_layers, num_experts]

        future_layer_indices = self._future_layer_indices(insertion_layer_index)

        selected: list[tuple[int, int]] = []
        for w_idx, layer_id in enumerate(future_layer_indices):
            if layer_id >= self.num_layers:
                continue
            layer_logits = avg_logits[w_idx]  # [num_experts]
            scores, expert_ids = torch.topk(layer_logits, k=min(top_k, self.num_experts))
            for eid, score in zip(
                expert_ids.tolist(), scores.tolist()
            ):
                selected.append((layer_id, eid))

        return tuple(selected)

    def _future_layer_indices(self, insertion_layer: int) -> list[int]:
        """计算 predictor window 覆盖的未来层索引。"""
        indices: list[int] = []
        for w in range(self.window_layers):
            layer_id = insertion_layer + 1 + w * self.stride_layers
            if layer_id < self.num_layers:
                indices.append(layer_id)
        return indices

    @property
    def loaded(self) -> bool:
        return self.predictor_model is not None
