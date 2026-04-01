"""权重索引与层级映射（Phase 1 最小占位）。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class WeightTierMap:
    """权重张量到层级的映射。"""

    gpu: list[str] = field(default_factory=list)
    cpu: list[str] = field(default_factory=list)
    nvme: list[str] = field(default_factory=list)


def build_gpu_only_tier_map(weight_names: list[str]) -> WeightTierMap:
    """Phase 1: 全量权重驻留 GPU。"""

    return WeightTierMap(gpu=list(weight_names))
