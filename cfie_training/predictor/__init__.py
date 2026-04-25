"""Predictor training helpers for CFIE."""

# 导出 predictor 训练侧所有公共数据结构与训练器。
from cfie_training.predictor.models import (
    PredictorCheckpointMetadata,
    PredictorEpochSummary,
    PredictorEvaluationTrace,
    PredictorTraceDataset,
    PredictorTraceExample,
    PredictorTrainingRunTrace,
)
from cfie_training.predictor.trainer import PredictorTrainer

# 统一维护 predictor 子包对外暴露的符号。
__all__ = [
    "PredictorCheckpointMetadata",
    "PredictorEpochSummary",
    "PredictorEvaluationTrace",
    "PredictorTraceDataset",
    "PredictorTraceExample",
    "PredictorTrainer",
    "PredictorTrainingRunTrace",
]
