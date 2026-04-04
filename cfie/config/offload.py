# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for model weight offloading."""

import warnings
from typing import Literal

from pydantic import Field, model_validator

from cfie.config.utils import config

# 当前支持的权重 offload 后端类型。
OffloadBackend = Literal["auto", "uva", "prefetch"]


@config
class UVAOffloadConfig:
    """Configuration for UVA (Unified Virtual Addressing) CPU offloading.

    Uses zero-copy access from CPU-pinned memory. Simple but requires
    fast CPU-GPU interconnect.
    """

    # 每张 GPU 允许借用的 CPU offload 空间上限，单位 GiB。
    cpu_offload_gb: float = Field(default=0, ge=0)
    """The space in GiB to offload to CPU, per GPU. Default is 0, which means
    no offloading. Intuitively, this argument can be seen as a virtual way to
    increase the GPU memory size. For example, if you have one 24 GB GPU and
    set this to 10, virtually you can think of it as a 34 GB GPU. Then you can
    load a 13B model with BF16 weight, which requires at least 26GB GPU memory.
    Note that this requires fast CPU-GPU interconnect, as part of the model is
    loaded from CPU memory to GPU memory on the fly in each model forward pass.
    This uses UVA (Unified Virtual Addressing) for zero-copy access.
    """

    # 只对名称中匹配这些参数段的权重启用 UVA offload；为空则按预算非选择性 offload。
    cpu_offload_params: set[str] = Field(default_factory=set)
    """The set of parameter name segments to target for CPU offloading.
    Unmatched parameters are not offloaded. If this set is empty, parameters
    are offloaded non-selectively until the memory limit defined by
    `cpu_offload_gb` is reached.
    Examples:
        - For parameter name "mlp.experts.w2_weight":
            - "experts" or "experts.w2_weight" will match.
            - "expert" or "w2" will NOT match (must be exact segments).
    This allows distinguishing parameters like "w2_weight" and "w2_weight_scale".
    """


@config
class PrefetchOffloadConfig:
    """Configuration for prefetch-based CPU offloading.

    Groups layers and uses async H2D prefetch to hide transfer latency.
    """

    # 每多少层划成一个 offload group。
    offload_group_size: int = Field(default=0, ge=0)
    """Group every N layers together. Offload last `offload_num_in_group`
    layers of each group. Default is 0 (disabled).
    Example: group_size=8, num_in_group=2 offloads layers 6,7,14,15,22,23,...
    Unlike cpu_offload_gb, this uses explicit async prefetching to hide transfer
    latency.
    """

    # 每个 group 中有多少层真正走 offload。
    offload_num_in_group: int = Field(default=1, ge=1)
    """Number of layers to offload per group.
    Must be <= offload_group_size. Default is 1."""

    # 向前预取多少层。
    offload_prefetch_step: int = Field(default=1, ge=0)
    """Number of layers to prefetch ahead.
    Higher values hide more latency but use more GPU memory. Default is 1."""

    # 仅对这些参数段匹配的权重启用 prefetch offload；为空则整层都 offload。
    offload_params: set[str] = Field(default_factory=set)
    """The set of parameter name segments to target for prefetch offloading.
    Unmatched parameters are not offloaded. If this set is empty, ALL
    parameters of each offloaded layer are offloaded.
    Uses segment matching: "w13_weight" matches "mlp.experts.w13_weight"
    but not "mlp.experts.w13_weight_scale".
    """


@config
class OffloadConfig:
    """Configuration for model weight offloading to reduce GPU memory usage."""

    # 选择使用哪种权重 offload 后端。
    offload_backend: OffloadBackend = "auto"
    """The backend for weight offloading. Options:
    - "auto": Selects based on which sub-config has non-default values
      (prefetch if offload_group_size > 0, uva if cpu_offload_gb > 0).
    - "uva": UVA (Unified Virtual Addressing) zero-copy offloading.
    - "prefetch": Async prefetch with group-based layer offloading.
    """

    # MoE tiered cache 自动规划时可使用的 CPU 总预算上限。
    moe_cpu_budget_gb: float = Field(default=0, ge=0)
    """Hard cap for the auto-enabled MoE tiered cache CPU budget.

    A value of 0 keeps planner-controlled sizing. Positive values bound the
    host RAM that the MoE expert cache may reserve, without affecting the
    generic UVA/prefetch offloader budgets.
    """

    # MoE tiered cache planner 需要为主机预留的最小空闲内存。
    moe_cpu_min_free_gb: float = Field(default=0, ge=0)
    """Minimum host memory to keep free for the MoE tiered cache planner.

    A value of 0 uses the planner default. Positive values override the
    automatic floor and leave more CPU memory available for the OS, page
    cache, pinned buffers, and other runtime allocations.
    """

    # UVA offload 子配置。
    uva: UVAOffloadConfig = Field(default_factory=UVAOffloadConfig)
    """Parameters for UVA offloading backend."""

    # prefetch offload 子配置。
    prefetch: PrefetchOffloadConfig = Field(default_factory=PrefetchOffloadConfig)
    """Parameters for prefetch offloading backend."""

    @model_validator(mode="after")
    def validate_offload_config(self) -> "OffloadConfig":
        """Validate offload configuration constraints."""
        # ----------------- 先校验 prefetch 自身的组大小约束 -----------------
        if self.offload_backend == "prefetch" or self.prefetch.offload_group_size > 0:
            # 每个 group 内要 offload 的层数不能超过 group 总层数。
            if self.prefetch.offload_num_in_group > self.prefetch.offload_group_size:
                raise ValueError(
                    f"offload_num_in_group ({self.prefetch.offload_num_in_group})"
                    f" must be <= offload_group_size"
                    f" ({self.prefetch.offload_group_size})"
                )
            # 一旦启用 prefetch，就要求至少预取 1 层。
            if self.prefetch.offload_prefetch_step < 1:
                raise ValueError(
                    f"offload_prefetch_step"
                    f" ({self.prefetch.offload_prefetch_step})"
                    f" must be >= 1 when prefetch offloading is enabled"
                    f" (offload_group_size > 0)"
                )

        # ----------------- 再检查“后端选择”与“子配置是否激活”是否冲突 -----------------
        # Warn if both backends have non-default values
        uva_active = self.uva.cpu_offload_gb > 0
        prefetch_active = self.prefetch.offload_group_size > 0
        # 显式指定走 UVA 时，prefetch 子配置即使被填了也不会生效。
        if self.offload_backend == "uva" and prefetch_active:
            warnings.warn(
                "Prefetch offload fields are set but offload_backend='uva'. "
                "Prefetch settings will be ignored.",
                stacklevel=2,
            )
        # 显式指定走 prefetch 时，UVA 子配置会被忽略。
        elif self.offload_backend == "prefetch" and uva_active:
            warnings.warn(
                "UVA offload fields are set but offload_backend='prefetch'. "
                "UVA settings will be ignored.",
                stacklevel=2,
            )
        # auto 模式下若两边都被激活，会优先选择 prefetch，并给出提醒。
        elif self.offload_backend == "auto" and uva_active and prefetch_active:
            warnings.warn(
                "Both UVA and prefetch offload fields are set with "
                "offload_backend='auto'. Prefetch backend will be selected. "
                "Set offload_backend explicitly to suppress this warning.",
                stacklevel=2,
            )
        # 返回校验完成后的配置对象。
        return self

    def compute_hash(self) -> str:
        """
        Provide a hash that uniquely identifies all the offload configs.

        All fields are included because PrefetchOffloader patches module
        forwards and inserts custom ops (wait_prefetch, start_prefetch)
        into the computation graph. Changing any offload setting can
        alter which layers are hooked and how prefetch indices are
        computed, so the compilation cache must distinguish them.
        """
        from cfie.config.utils import get_hash_factors, hash_factors

        # offload 配置会影响 forward patching 和 prefetch 索引计算，因此全部字段都参与哈希。
        factors = get_hash_factors(self, ignored_factors=set())
        hash_str = hash_factors(factors)
        return hash_str
