"""Standalone training package for CFIE.

The inference runtime stays under ``cfie/``.
All future training-side Python code should live under ``cfie_training/``.
"""

# 导出训练侧顶层配置对象。
from cfie_training.config import TrainingProjectConfig
# 导出 predictor 训练与部署相关公共类型。
from cfie_training.predictor import (
    PredictorCheckpointMetadata,
    PredictorDeploymentManifest,
    PredictorEvaluationTrace,
    PredictorMetricsSummary,
    PredictorRuntimeSchema,
    PredictorTrainer,
)
# 导出 predictor 蓝图构造入口。
from cfie_training.predictor_blueprint import PredictorBlueprint, build_predictor_blueprint
# 导出内置 profile 构造入口。
from cfie_training.profiles import build_profile_config
# 导出训练基座主引擎。
from cfie_training.runtime.engine import FirstVersionTrainingEngine
# 导出训练项目装配入口。
from cfie_training.runtime.project import TrainingProject

# 统一维护包级公共导出符号。
__all__ = [
    "FirstVersionTrainingEngine",
    "PredictorCheckpointMetadata",
    "PredictorDeploymentManifest",
    "PredictorBlueprint",
    "PredictorEvaluationTrace",
    "PredictorMetricsSummary",
    "PredictorRuntimeSchema",
    "PredictorTrainer",
    "TrainingProject",
    "TrainingProjectConfig",
    "build_predictor_blueprint",
    "build_profile_config",
]
