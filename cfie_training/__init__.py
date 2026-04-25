"""Standalone training package for CFIE.

The inference runtime stays under ``cfie/``.
All future training-side Python code should live under ``cfie_training/``.
"""

from cfie_training.config import TrainingProjectConfig
from cfie_training.predictor import (
    PredictorCheckpointMetadata,
    PredictorEvaluationTrace,
    PredictorTrainer,
)
from cfie_training.profiles import build_profile_config
from cfie_training.runtime.engine import FirstVersionTrainingEngine
from cfie_training.runtime.project import TrainingProject

__all__ = [
    "FirstVersionTrainingEngine",
    "PredictorCheckpointMetadata",
    "PredictorEvaluationTrace",
    "PredictorTrainer",
    "TrainingProject",
    "TrainingProjectConfig",
    "build_profile_config",
]
