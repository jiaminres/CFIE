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
    # ------------------------------- 根据训练档位名称分发并构造对应配置 -------------------------------
    # 当训练档位名称为 Qwen3.5-35B-A3B 专用档位时，构造并返回对应的专用训练配置。
    if profile_name == QWEN35_35B_A3B_PROFILE:
        # 返回 Qwen3.5-35B-A3B 档位的训练项目配置对象。
        return build_qwen35_35b_a3b_config()

    # ------------------------------- 处理通用 generic 档位配置 -------------------------------
    # 当训练档位名称为 generic 时，直接返回默认初始化的通用训练配置对象。
    if profile_name == "generic":
        # 返回未附加专用模型预设的通用训练配置。
        return TrainingProjectConfig()

    # ------------------------------- 拒绝不受支持的训练档位名称 -------------------------------
    # 当传入的训练档位名称不在当前支持列表中时，抛出异常提示调用方。
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
