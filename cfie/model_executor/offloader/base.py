# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from
# https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/utils/offloader.py
"""Base classes for model parameter offloading."""

from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import TYPE_CHECKING

import torch.nn as nn

from cfie.logger import init_logger

if TYPE_CHECKING:
    from cfie.config import OffloadConfig

logger = init_logger(__name__)

"""
class relation:

BaseOffloader (ABC)
  * implemented by: UVAOffloader
  * implemented by: PrefetchOffloader
    * uses: _ModuleOffloader
        * uses: _BaseParamOffloader (ABC)
            * implemented by: _CpuParamOffloader
"""


class BaseOffloader(ABC):
    """Base class for model parameter offloading strategies.

    Offloaders control how model parameters are stored and loaded during
    inference. Different strategies trade memory for compute/transfer time.
    """

    @abstractmethod
    def wrap_modules(
            self,
            modules_generator: Generator[nn.Module, None, None],
    ) -> list[nn.Module]:
        """Wrap modules with offloading logic.

        Args:
            modules_generator: Generator yielding modules to potentially offload.

        Returns:
            List of modules, potentially with offloading hooks installed.
        """
        pass

    def post_init(self):
        """Called after model construction completes.

        Offloaders can use this to:
        - Finalize parameter storage
        - Start initial prefetching
        - Allocate shared resources
        """
        return

    def sync_prev_onload(self) -> None:  # noqa: B027
        """Sync previous onload operations. Override in subclasses."""
        pass

    def join_after_forward(self) -> None:  # noqa: B027
        """Join streams after forward. Override in subclasses."""
        pass

    def _wait_for_layer(self, layer_idx: int) -> None:  # noqa: B027
        """Wait for layer prefetch. Override in subclasses."""
        pass

    def _start_prefetch(self, layer_idx: int) -> None:  # noqa: B027
        """Start layer prefetch. Override in subclasses."""
        pass


class NoopOffloader(BaseOffloader):
    """No-op offloader that returns modules as-is without any offloading."""

    def wrap_modules(
            self,
            modules_generator: Generator[nn.Module, None, None],
    ) -> list[nn.Module]:
        """Return modules unchanged."""
        return list(modules_generator)


# Global singleton offloader instance (defaults to no-op).
_instance: BaseOffloader = NoopOffloader()


def get_offloader() -> BaseOffloader:
    """Get the global offloader instance."""
    # 返回当前全局生效的 offloader 单例。
    return _instance


def set_offloader(instance: BaseOffloader) -> None:
    """Set the global offloader instance."""
    # 用新实例覆盖当前全局 offloader。
    global _instance
    _instance = instance
    # no-op 后端只打一次本地调试日志。
    if isinstance(instance, NoopOffloader):
        logger.debug_once(
            "Offloader set to NoopOffloader (no offloading).", scope="local"
        )
    else:
        # 实际启用的 offloader 记录到信息日志，便于确认后端选择结果。
        logger.info_once("Offloader set to %s", type(instance).__name__, scope="local")


def create_offloader(offload_config: "OffloadConfig") -> BaseOffloader:
    """
    根据 offload 配置创建一个 offloader。

    使用显式的 ``offload_backend`` 选择器。当设置为 ``"auto"`` 时：
    如果 ``offload_group_size > 0``，则选择 prefetch；
    如果 ``cpu_offload_gb > 0``，则选择 UVA；
    否则选择 noop。
    """

    # 延迟导入具体实现，避免基础模块反向依赖重型后端代码。
    from cfie.model_executor.offloader.prefetch import PrefetchOffloader
    from cfie.model_executor.offloader.uva import UVAOffloader

    # 读取顶层后端选择结果。
    backend = offload_config.offload_backend
    # 读取 UVA 子配置。
    uva = offload_config.uva
    # 读取 prefetch 子配置。
    prefetch = offload_config.prefetch

    # auto 模式下按配置活跃度补全实际后端类型。
    if backend == "auto":
        # 优先选择显式启用的 prefetch 后端。
        if prefetch.offload_group_size > 0:
            backend = "prefetch"
        # 未启用 prefetch 时，再看是否配置了 UVA 容量。
        elif uva.cpu_offload_gb > 0:
            backend = "uva"
        else:
            # 两类 offload 都未启用时，退回 no-op 后端。
            return NoopOffloader()

    # prefetch 后端直接按子配置实例化。
    if backend == "prefetch":
        return PrefetchOffloader(
            group_size=prefetch.offload_group_size,
            num_in_group=prefetch.offload_num_in_group,
            prefetch_step=prefetch.offload_prefetch_step,
            offload_params=prefetch.offload_params,
            mode="cpu",
        )
    # UVA 后端把 GiB 预算换算成字节数后实例化。
    elif backend == "uva":
        return UVAOffloader(
            cpu_offload_max_bytes=int(uva.cpu_offload_gb * 1024 ** 3),
            cpu_offload_params=uva.cpu_offload_params,
        )
    else:
        # 未识别或未启用的后端统一退回 no-op。
        return NoopOffloader()
