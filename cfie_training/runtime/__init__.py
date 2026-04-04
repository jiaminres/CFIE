"""Runtime helpers for the standalone CFIE training package."""

# 导出训练运行时总控引擎。
from cfie_training.runtime.engine import FirstVersionTrainingEngine
# 导出训练项目装配入口。
from cfie_training.runtime.project import TrainingProject

# 统一维护 runtime 子包对外暴露的符号。
__all__ = ["FirstVersionTrainingEngine", "TrainingProject"]
