"""Profile builders for the standalone CFIE training package."""

from cfie_training.blueprint import TrainingBlueprint, build_training_blueprint
from cfie_training.config import TrainingProjectConfig
from cfie_training.profiles.qwen35_35b_a3b import (
    QWEN35_35B_A3B_PROFILE,
    build_qwen35_35b_a3b_config,
)

SUPPORTED_TRAINING_PROFILES = (QWEN35_35B_A3B_PROFILE, "generic")
DEFAULT_TRAINING_PROFILE = QWEN35_35B_A3B_PROFILE


# 按 profile 名称构造训练配置。
def build_profile_config(profile_name: str) -> TrainingProjectConfig:
    if profile_name == QWEN35_35B_A3B_PROFILE:
        return build_qwen35_35b_a3b_config()
    if profile_name == "generic":
        return TrainingProjectConfig()
    raise ValueError(f"unsupported training profile: {profile_name}")


# 基于训练配置生成对应的训练蓝图。
def build_profile_blueprint(config: TrainingProjectConfig) -> TrainingBlueprint:
    return build_training_blueprint(config)


__all__ = [
    "DEFAULT_TRAINING_PROFILE",
    "SUPPORTED_TRAINING_PROFILES",
    "build_profile_blueprint",
    "build_profile_config",
]
