"""调度策略定义。"""

from __future__ import annotations

from enum import Enum


class SchedulerPolicy(str, Enum):
    """Phase 1 仅实现 FCFS。"""

    FCFS = "fcfs"
    PRIORITY = "priority"
