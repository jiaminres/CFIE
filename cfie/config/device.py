# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import field
from typing import Any, Literal

import torch
from pydantic import ConfigDict, SkipValidation

from cfie.config.utils import config
from cfie.utils.hashing import safe_hash

# 设备类型字面量集合。
Device = Literal["auto", "cuda", "cpu", "tpu", "xpu"]


@config(config=ConfigDict(arbitrary_types_allowed=True))
class DeviceConfig:
    """Configuration for the device to use for vLLM execution."""

    # 用户侧传入的设备声明；默认 auto 让平台自行探测。
    device: SkipValidation[Device | torch.device | None] = "auto"
    """Device type for vLLM execution.
    This parameter is deprecated and will be
    removed in a future release.
    It will now be set automatically based
    on the current platform."""
    # 归一化后的设备类型字符串，由 __post_init__ 填充。
    device_type: str = field(init=False)
    """Device type from the current platform. This is set in
    `__post_init__`."""

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        # no factors to consider.
        # the device/platform information will be summarized
        # by torch/cfie automatically.
        # 设备配置不单独改变图结构哈希。
        factors: list[Any] = []
        # 计算并返回稳定哈希。
        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str

    def __post_init__(self):
        # auto 模式下从当前平台推断 device_type。
        if self.device == "auto":
            # Automated device type detection
            from cfie.platforms import current_platform

            self.device_type = current_platform.device_type
            # 若平台无法推断设备类型，则直接报错。
            if not self.device_type:
                raise RuntimeError(
                    "Failed to infer device type, please set "
                    "the environment variable `VLLM_LOGGING_LEVEL=DEBUG` "
                    "to turn on verbose logging to help debug the issue."
                )
        else:
            # Device type is assigned explicitly
            # 字符串形式直接作为 device_type。
            if isinstance(self.device, str):
                self.device_type = self.device
            # torch.device 形式则读取其 type 字段。
            elif isinstance(self.device, torch.device):
                self.device_type = self.device.type

        # Some device types require processing inputs on CPU
        # TPU 路径保持 self.device=None，由下游按 TPU 语义处理。
        if self.device_type in ["tpu"]:
            self.device = None
        else:
            # Set device with device type
            # 其他平台统一构造 torch.device 对象。
            self.device = torch.device(self.device_type)
