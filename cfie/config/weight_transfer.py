# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Literal

from cfie.config.utils import config


@config
class WeightTransferConfig:
    """Configuration for weight transfer during RL training."""

    # 指定 RL 训练时权重广播/传输所采用的底层通信后端。
    backend: Literal["nccl", "ipc"] = "nccl"
    """The backend to use for weight transfer."""
