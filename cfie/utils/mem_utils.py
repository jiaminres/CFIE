# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import gc
import math
import time
from collections.abc import Generator
from dataclasses import dataclass, field
from functools import cache

import psutil
import torch
import torch.types

from cfie.platforms import current_platform

from .mem_constants import GiB_bytes, MiB_bytes


def format_mib(b: int) -> str:
    return f"{round(b / MiB_bytes, 2)}"


def format_gib(b: int) -> str:
    return f"{round(b / GiB_bytes, 2)}"


def split_gpu_memory_budget(
    total_memory: int,
    gpu_memory_utilization: float,
) -> tuple[int, int]:
    """
    按统一口径把总 GPU 显存切成“静态预算”和“运行时余量”两部分。

    这里的切分规则是当前显存划分主链的基础约定，MoE planner 与 worker
    运行时都必须复用同一套逻辑，避免两边对 `gpu_memory_utilization`
    的理解发生漂移。

    - static_budget:
      可被权重、常驻非 KV 状态以及最终 KV cache 占用的“静态区域”。
    - runtime_headroom:
      必须额外留给 prefill/decode 峰值、临时 workspace、图捕获等
      运行期抖动的余量。
    """
    # ratio 内的显存视为“静态预算”，后续 KV cache 只能从这部分里扣。
    static_budget = math.ceil(total_memory * gpu_memory_utilization)
    # ratio 外的显存全部当作运行期 headroom，不再挪给静态常驻分配。
    runtime_headroom = max(0, total_memory - static_budget)
    return static_budget, runtime_headroom


@cache
def get_max_shared_memory_bytes(gpu: int = 0) -> int:
    """Returns the maximum shared memory per thread block in bytes."""
    from cfie import _custom_ops as ops

    max_shared_mem = ops.get_max_shared_memory_per_block_device_attribute(gpu)
    # value 0 will cause MAX_SEQ_LEN become negative and test_attention.py
    # will fail
    assert max_shared_mem > 0, "max_shared_mem cannot be zero"
    return int(max_shared_mem)


def get_cpu_memory() -> int:
    """Returns the total CPU memory of the node in bytes."""
    return psutil.virtual_memory().total


class DeviceMemoryProfiler:
    def __init__(self, device: torch.types.Device | None = None):
        self.device = device

    def current_memory_usage(self) -> float:
        # Return the memory usage in bytes.
        gc.collect()
        return current_platform.get_current_memory_usage(self.device)

    def __enter__(self):
        self.initial_memory = self.current_memory_usage()
        # This allows us to call methods of the context manager if needed
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.final_memory = self.current_memory_usage()
        self.consumed_memory = self.final_memory - self.initial_memory

        # Force garbage collection
        gc.collect()


@dataclass
class MemorySnapshot:
    """显存快照对象。"""

    # PyTorch 侧观测到的峰值显存占用，单位为字节。
    torch_peak: int = 0

    # 当前时刻设备剩余空闲显存，单位为字节。
    free_memory: int = 0

    # 当前设备总显存容量，单位为字节。
    total_memory: int = 0

    # 当前整个设备已被占用的显存，单位为字节。
    cuda_memory: int = 0

    # 当前由 PyTorch 运行时保留的显存，单位为字节。
    torch_memory: int = 0

    # 当前非 PyTorch 路径占用的显存，单位为字节。
    non_torch_memory: int = 0

    # 本次快照采样的时间戳。
    timestamp: float = 0.0

    # 当前快照绑定的设备对象；允许外部显式传入。
    device: torch.types.Device = None

    # 是否在初始化结束后自动执行一次测量。
    auto_measure: bool = True

    def __post_init__(self) -> None:
        # ------------------------------- 解析当前快照绑定的设备对象 -------------------------------
        # 当调用方没有显式传入设备时，从当前平台读取默认设备。
        if self.device is None:
            # 获取当前平台返回默认设备的函数对象。
            device_fn = current_platform.current_device

            # 校验当前平台提供了默认设备获取函数。
            assert device_fn is not None

            # 使用当前平台返回的默认设备构造标准 torch.device 对象。
            self.device_ = torch.device(device_fn())
        else:
            # 当调用方显式传入了设备时，直接将其规范化为 torch.device 对象。
            self.device_ = torch.device(self.device)

        # ------------------------------- 按配置决定是否在初始化后立即采样 -------------------------------
        # 当启用了自动测量时，在对象初始化完成后立即执行一次显存快照采样。
        if self.auto_measure:
            self.measure()

    def measure(self) -> None:
        # ------------------------------- 读取当前快照绑定的目标设备 -------------------------------
        # 取出当前快照实际绑定的设备对象，后续所有显存查询都基于它执行。
        device = self.device_

        # ------------------------------- 采样 PyTorch 侧的峰值显存占用 -------------------------------
        # 读取当前设备在 PyTorch 统计口径下的 allocated_bytes 峰值。
        self.torch_peak = current_platform.memory_stats(device).get(
            "allocated_bytes.all.peak", 0
        )

        # ------------------------------- 采样当前设备的空闲显存与总显存 -------------------------------
        # 读取当前设备的空闲显存与总显存，作为本次快照的基础容量信息。
        self.free_memory, self.total_memory = current_platform.mem_get_info(device)

        # ------------------------------- 在 UMA 平台上修正 free_memory 的取值口径 -------------------------------
        # 列出共享系统内存的集成式 GPU 设备 capability，用于识别 UMA 平台。
        shared_sysmem_device_mem_sms = ((8, 7), (11, 0), (12, 1))

        # 当当前平台为 CUDA 且设备 capability 命中 UMA 平台集合时，需要改用系统可用内存修正 free_memory。
        if (
            current_platform.is_cuda()
            and current_platform.get_device_capability(device.index)
            in shared_sysmem_device_mem_sms
        ):
            # 在 UMA 平台上，用系统当前可用内存近似替代设备 free_memory。
            self.free_memory = psutil.virtual_memory().available

        # ------------------------------- 计算当前设备总占用显存 -------------------------------
        # 用总显存减去空闲显存，得到当前设备已被占用的总显存。
        self.cuda_memory = self.total_memory - self.free_memory

        # ------------------------------- 采样当前由 PyTorch 保留的显存 -------------------------------
        # 读取当前设备上由 PyTorch 运行时保留的显存字节数。
        self.torch_memory = current_platform.memory_reserved(device)

        # ------------------------------- 计算非 PyTorch 路径占用的显存 -------------------------------
        # 用设备总占用显存减去 PyTorch 保留显存，得到非 PyTorch 路径占用的显存。
        self.non_torch_memory = self.cuda_memory - self.torch_memory

        # ------------------------------- 记录本次快照采样时间戳 -------------------------------
        # 保存当前时间戳，用于后续快照差分与时序分析。
        self.timestamp = time.time()

    def __sub__(self, other: "MemorySnapshot") -> "MemorySnapshot":
        # ------------------------------- 校验参与差分的两个快照来自同一设备 -------------------------------
        # 当两个快照绑定的设备不一致时，拒绝执行差分运算。
        if self.device_ != other.device_:
            raise ValueError(
                "The two snapshots should be from the same device! "
                f"Found: {self.device_} vs. {other.device_}"
            )

        # ------------------------------- 构造两个快照之间的差分结果 -------------------------------
        # 返回一个新快照对象，其各字段值为当前快照减去另一个快照后的差值。
        return MemorySnapshot(
            torch_peak=self.torch_peak - other.torch_peak,
            free_memory=self.free_memory - other.free_memory,
            total_memory=self.total_memory - other.total_memory,
            cuda_memory=self.cuda_memory - other.cuda_memory,
            torch_memory=self.torch_memory - other.torch_memory,
            non_torch_memory=self.non_torch_memory - other.non_torch_memory,
            timestamp=self.timestamp - other.timestamp,
            device=self.device_,
            auto_measure=False,
        )

    def __repr__(self) -> str:
        # ------------------------------- 构造当前显存快照的人类可读字符串表示 -------------------------------
        # 返回包含峰值显存、空闲显存、总显存、PyTorch 显存、非 PyTorch 显存与时间戳的摘要字符串。
        return (
            f"torch_peak={format_gib(self.torch_peak)}GiB, "
            f"free_memory={format_gib(self.free_memory)}GiB, "
            f"total_memory={format_gib(self.total_memory)}GiB, "
            f"{current_platform.device_name}_memory={format_gib(self.cuda_memory)}GiB, "
            f"torch_memory={format_gib(self.torch_memory)}GiB, "
            f"non_torch_memory={format_gib(self.non_torch_memory)}GiB, "
            f"timestamp={self.timestamp}, "
            f"auto_measure={self.auto_measure}"
        )


@dataclass
class MemoryProfilingResult:
    """Memory profiling result. All numbers are in bytes."""

    non_kv_cache_memory: int = 0
    torch_peak_increase: int = 0
    non_torch_increase: int = 0
    weights_memory: int = 0
    before_create: MemorySnapshot = field(default_factory=MemorySnapshot)
    profile_time: float = 0.0

    def __post_init__(self) -> None:
        device = self.before_create.device_

        self.before_profile = MemorySnapshot(device=device, auto_measure=False)
        self.after_profile = MemorySnapshot(device=device, auto_measure=False)

    def __repr__(self) -> str:
        return (
            f"Memory profiling takes {self.profile_time:.2f} seconds. "
            f"Total non KV cache memory: "
            f"{format_gib(self.non_kv_cache_memory)}GiB; "
            f"torch peak memory increase: "
            f"{format_gib(self.torch_peak_increase)}GiB; "
            f"non-torch forward increase memory: "
            f"{format_gib(self.non_torch_increase)}GiB; "
            f"weights memory: {format_gib(self.weights_memory)}GiB."
        )


@contextlib.contextmanager
def memory_profiling(
    baseline_snapshot: MemorySnapshot,
    weights_memory: int = 0,
) -> Generator[MemoryProfilingResult, None, None]:
    """
    Memory profiling context manager.

    baseline_snapshot: the memory snapshot before the current vLLM instance.
    weights_memory: memory used by PyTorch when loading the model weights.
        Note that, before loading the model weights, we also initialize the device
        and distributed environment, which may consume some memory. This part is not
        included in the weights_memory because PyTorch does not control it.

    The memory in one GPU can be classified into 3 categories:
    1. memory used by anything other than the current vLLM instance.
    2. memory used by torch in the current vLLM instance.
    3. memory used in the current vLLM instance, but not by torch.

    A quantitive example:

    Before creating the current vLLM instance:
        category 1: 1 GiB
        category 2: 0 GiB
        category 3: 0 GiB

    After creating the current vLLM instance and loading the model,
    (i.e. before profiling):
        category 1: 1 GiB
        category 2: 2 GiB (model weights take 2 GiB)
        category 3: 0.5 GiB (memory used by NCCL)

    During profiling (peak):
        category 1: 1 GiB
        category 2: 4 GiB (peak activation tensors take 2 GiB)
        category 3: 1 GiB (memory used by NCCL + buffers for some attention backends)

    After profiling:
        category 1: 1 GiB
        category 2: 3 GiB (after garbage-collecting activation tensors)
        category 3: 1 GiB (memory used by NCCL + buffers for some attention backends)

    In this case, non-kv cache takes 5 GiB in total, including:
    a. 2 GiB used by the model weights (category 2)
    b. 2 GiB reserved for the peak activation tensors (category 2)
    c. 1 GiB used by non-torch components (category 3)

    The memory used for loading weights (a.) is directly given from the
    argument `weights_memory`.

    The increase of `torch.cuda.memory_stats()["allocated_bytes.all.peak"]`
    during profiling gives (b.).

    The increase of `non_torch_memory` from creating the current vLLM instance
    until after profiling to get (c.).
    """
    gc.collect()
    torch.accelerator.empty_cache()
    current_platform.reset_peak_memory_stats(baseline_snapshot.device_)

    result = MemoryProfilingResult(
        before_create=baseline_snapshot,
        # the part of memory used for holding the model weights
        weights_memory=weights_memory,
    )

    result.before_profile.measure()

    yield result

    gc.collect()
    torch.accelerator.empty_cache()

    result.after_profile.measure()

    diff_profile = result.after_profile - result.before_profile
    diff_from_create = result.after_profile - result.before_create
    result.torch_peak_increase = diff_profile.torch_peak
    result.non_torch_increase = diff_from_create.non_torch_memory
    result.profile_time = diff_profile.timestamp

    non_torch_memory = result.non_torch_increase
    peak_activation_memory = result.torch_peak_increase
    result.non_kv_cache_memory = (
        non_torch_memory + peak_activation_memory + result.weights_memory
    )
