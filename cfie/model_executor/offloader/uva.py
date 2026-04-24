# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UVA-based CPU offloading using Unified Virtual Addressing."""

from collections.abc import Generator

import torch
import torch.nn as nn
from torch.func import functional_call

import cfie.envs as envs
from cfie.logger import init_logger
from cfie.model_executor.offloader.base import BaseOffloader
from cfie.utils.mem_utils import format_gib
from cfie.utils.platform_utils import is_pin_memory_available, is_uva_available
from cfie.utils.torch_utils import get_accelerator_view_from_cpu_tensor

logger = init_logger(__name__)


class UVAOffloader(BaseOffloader):
    """使用统一虚拟寻址（UVA）实现零拷贝访问的 Offloader。

    该 Offloader 会将参数移动到 pinned CPU memory（锁页 CPU 内存），
    并基于 UVA 创建 CUDA 视图。这样 GPU 就可以在不显式搬运数据的情况下，
    直接访问 CPU 内存，但代价是会受 PCIe 带宽限制，因此速度比访问 GPU 显存慢。

    当通过环境变量禁用 UVA 时，会回退到基于 functional_call 的实现：
    在需要时按需移动参数。

    参数：
        cpu_offload_max_bytes: 最多允许 offload 到 CPU 的字节数。
        cpu_offload_params: 用于选择性 offload 的参数名片段集合。
            如果为空，则所有参数都可以参与 offload，直到达到字节数上限。
    """

    def __init__(
            self,
            cpu_offload_max_bytes: int,
            cpu_offload_params: set[str] | None = None,
    ):
        # ------------------------------- 记录 UVA offload 的预算与筛选条件 -------------------------------
        # 记录允许下沉到 CPU 的总字节数上限。
        self.cpu_offload_max_bytes = cpu_offload_max_bytes
        # 从 0 开始累计已下沉参数的字节数。
        self.cpu_offload_bytes = 0
        # 记录允许命中的参数名片段集合。
        self.cpu_offload_params = cpu_offload_params or set()

        # 根据平台能力和环境变量决定是否启用 pinned memory。
        self.pin_memory = (
                is_pin_memory_available()
                and not envs.VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY
        )
        # 根据平台能力和环境变量决定是否启用 UVA 视图。
        self.uva_offloading = (
                is_uva_available() and not envs.VLLM_WEIGHT_OFFLOADING_DISABLE_UVA
        )

    def wrap_modules(
            self,
            modules_generator: Generator[nn.Module, None, None],
    ) -> list[nn.Module]:
        """Wrap modules with UVA offloading."""
        # ------------------------------- 逐个模块安装 UVA offload 逻辑 -------------------------------
        # 逐个处理生成器中的模块，并在需要时改写其参数存储。
        modules = [self._maybe_offload_to_cpu(module) for module in modules_generator]
        # 只有实际发生参数下沉时才打印累计规模。
        if self.cpu_offload_bytes > 0:
            logger.info(
                "Total CPU offloaded parameters: %s",
                format_gib(self.cpu_offload_bytes),
            )
        # 返回已经完成包装的模块列表。
        return modules

    def _maybe_offload_to_cpu(self, module: nn.Module) -> nn.Module:
        """Offload module parameters to CPU using UVA if budget allows."""
        # ------------------------------- 先判断当前模块是否需要进入 UVA 处理 -------------------------------
        # 取第一个参数作为模块设备与空模块判断的代表。
        if (params := next(module.parameters(), None)) is None:
            # 没有参数的模块不需要下沉。
            return module

        # 记录当前模块原始所在设备。
        device = params.device

        # 已经在 CPU 上的模块无需重复处理。
        if device == torch.device("cpu"):
            return module

        # 全局预算已经耗尽时直接跳过当前模块。
        if self.cpu_offload_bytes >= self.cpu_offload_max_bytes:
            return module

        # ------------------------------- 按参数粒度把权重下沉到 CPU 或 UVA 视图 -------------------------------
        # 标记当前模块是否至少有一个参数完成了下沉。
        offloaded_parameters = False
        # 逐个参数决定是否需要参与本轮下沉。
        for name, p in module.named_parameters():
            # 达到总预算后立即停止处理剩余参数。
            if self.cpu_offload_bytes >= self.cpu_offload_max_bytes:
                # 当前实现允许同一模块只有部分参数被下沉。
                break

            # 配置了白名单时，只处理命中的参数名片段。
            if self.cpu_offload_params:
                # 在参数名前后补点，避免片段匹配误伤相似名称。
                should_offload = any(
                    f".{param}." in f".{name}." for param in self.cpu_offload_params
                )
                # 未命中白名单的参数保持原位。
                if not should_offload:
                    continue

            # 先把参数数据搬到 CPU。
            cpu_data = p.data.to(device="cpu")
            # 允许时把 CPU tensor 升级成 pinned memory。
            if self.pin_memory:
                cpu_data = cpu_data.pin_memory()

            # 关闭 UVA 时，仅把参数保留在 CPU tensor 上。
            if not self.uva_offloading:
                p.data = cpu_data
            else:
                # 开启 UVA 时，把 CPU tensor 映射成加速器可访问视图。
                p.data = get_accelerator_view_from_cpu_tensor(cpu_data)
                # 给参数打上 UVA 下沉标记，便于后续识别。
                p._cfie_is_uva_offloaded = True

            # 按参数真实字节数累加全局 CPU offload 用量。
            self.cpu_offload_bytes += p.data.numel() * p.data.element_size()
            # 记录当前模块至少有一个参数已经完成下沉。
            offloaded_parameters = True

        # ------------------------------- UVA 关闭时回退到按需搬运的 forward 包装 -------------------------------
        # 只有发生了参数下沉且未启用 UVA 时，才需要改写 forward。
        if offloaded_parameters and not self.uva_offloading:
            # 保存模块原始 forward，供包装逻辑临时调用。
            original_forward = module.forward

            def forward(*args, **kwargs):
                # 先还原原始 forward，避免递归再次进入包装层。
                module.forward = original_forward
                # 把 state_dict 中的参数和缓冲按需搬回原始设备。
                device_state = {
                    k: v.to(device, non_blocking=True)
                    for k, v in module.state_dict().items()
                }

                # 用临时设备态执行一次无状态 forward。
                output = functional_call(
                    module,
                    device_state,
                    args=args,
                    kwargs=kwargs,
                    # 逐参数搬运后共享权重关系已失效，因此禁用 tied weights。
                    tie_weights=False,
                )
                # 执行完成后重新挂回包装后的 forward。
                module.forward = forward
                # 把本次前向结果返回给上层调用方。
                return output

            # 用包装后的 forward 覆盖模块原始实现。
            module.forward = forward

        # 返回已经按预算完成处理的模块对象。
        return module
