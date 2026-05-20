# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for model weight offloading."""

import warnings
from typing import Literal

from pydantic import Field, model_validator

from cfie.config.utils import config

# 褰撳墠鏀寔鐨勬潈閲?offload 鍚庣绫诲瀷銆?
OffloadBackend = Literal["auto", "uva", "prefetch"]


@config
class UVAOffloadConfig:
    """Configuration for UVA (Unified Virtual Addressing) CPU offloading.

    Uses zero-copy access from CPU-pinned memory. Simple but requires
    fast CPU-GPU interconnect.
    """

    # 姣忓紶 GPU 鍏佽鍊熺敤鐨?CPU offload 绌洪棿涓婇檺锛屽崟浣?GiB銆?
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

    # 鍙鍚嶇О涓尮閰嶈繖浜涘弬鏁版鐨勬潈閲嶅惎鐢?UVA offload锛涗负绌哄垯鎸夐绠楅潪閫夋嫨鎬?offload銆?
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

    # 姣忓灏戝眰鍒掓垚涓€涓?offload group銆?
    offload_group_size: int = Field(default=0, ge=0)
    """Group every N layers together. Offload last `offload_num_in_group`
    layers of each group. Default is 0 (disabled).
    Example: group_size=8, num_in_group=2 offloads layers 6,7,14,15,22,23,...
    Unlike cpu_offload_gb, this uses explicit async prefetching to hide transfer
    latency.
    """

    # 姣忎釜 group 涓湁澶氬皯灞傜湡姝ｈ蛋 offload銆?
    offload_num_in_group: int = Field(default=1, ge=1)
    """Number of layers to offload per group.
    Must be <= offload_group_size. Default is 1."""

    # 鍚戝墠棰勫彇澶氬皯灞傘€?
    offload_prefetch_step: int = Field(default=1, ge=0)
    """Number of layers to prefetch ahead.
    Higher values hide more latency but use more GPU memory. Default is 1."""

    # 浠呭杩欎簺鍙傛暟娈靛尮閰嶇殑鏉冮噸鍚敤 prefetch offload锛涗负绌哄垯鏁村眰閮?offload銆?
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

    # 閫夋嫨浣跨敤鍝鏉冮噸 offload 鍚庣銆?
    offload_backend: OffloadBackend = "auto"
    """The backend for weight offloading. Options:
    - "auto": Selects based on which sub-config has non-default values
      (prefetch if offload_group_size > 0, uva if cpu_offload_gb > 0).
    - "uva": UVA (Unified Virtual Addressing) zero-copy offloading.
    - "prefetch": Async prefetch with group-based layer offloading.
    """

    # MoE tiered cache 鑷姩瑙勫垝鏃跺彲浣跨敤鐨?CPU 鎬婚绠椾笂闄愩€?
    moe_cpu_budget_gb: float = Field(default=0, ge=0)
    """Hard cap for the auto-enabled MoE tiered cache CPU budget.

    A value of 0 keeps planner-controlled sizing. Positive values bound the
    host RAM that the MoE expert cache may reserve, without affecting the
    generic UVA/prefetch offloader budgets.
    """

    # MoE tiered cache planner 闇€瑕佷负涓绘満棰勭暀鐨勬渶灏忕┖闂插唴瀛樸€?
    moe_cpu_min_free_gb: float = Field(default=0, ge=0)
    """Minimum host memory to keep free for the MoE tiered cache planner.

    A value of 0 uses the planner default. Positive values override the
    automatic floor and leave more CPU memory available for the OS, page
    cache, pinned buffers, and other runtime allocations.
    """

    # UVA offload 瀛愰厤缃€?
    uva: UVAOffloadConfig = Field(default_factory=UVAOffloadConfig)
    """Parameters for UVA offloading backend."""

    # prefetch offload 瀛愰厤缃€?
    prefetch: PrefetchOffloadConfig = Field(default_factory=PrefetchOffloadConfig)
    """Parameters for prefetch offloading backend."""

    # 姣忓眰 resident slots 涓婇檺銆?
    gpu_slots_per_layer: int = Field(default=0, ge=0)
    """Explicit cap for GPU resident expert slots per layer. 0 keeps planner
    controlled sizing."""

    prefill_burst_slots: int = Field(default=0, ge=0)
    """Shared GPU temporary expert slots for prefill chunks. 0 lets the
    planner decide when applicable."""

    cpu_static_preprocess_batch_size: int = Field(default=0, ge=0)
    """Number of experts to preprocess per initialization batch. A value of 0
    auto-fits the largest safe batch from current CPU/GPU free memory."""

    cpu_static_pinned_gb: float = Field(default=0.0, ge=0.0)
    """Maximum GiB of runtime-ready CPU static experts to keep in pinned memory."""

    cpu_static_pinned_layers: str = ""
    """Comma-separated MoE layer indices or ranges to pin, e.g. '0-23,30'."""

    prepare_cpu_copy_batch_size: int = Field(default=8, ge=0)
    """Number of missing experts processed by one CPU copy worker during
    prepare-time staging. A value of 0 keeps automatic sizing."""

    @model_validator(mode="after")
    def validate_offload_config(self) -> "OffloadConfig":
        """Validate offload configuration constraints."""
        # ----------------- 鍏堟牎楠?prefetch 鑷韩鐨勭粍澶у皬绾︽潫 -----------------
        if self.offload_backend == "prefetch" or self.prefetch.offload_group_size > 0:
            # 姣忎釜 group 鍐呰 offload 鐨勫眰鏁颁笉鑳借秴杩?group 鎬诲眰鏁般€?
            if self.prefetch.offload_num_in_group > self.prefetch.offload_group_size:
                raise ValueError(
                    f"offload_num_in_group ({self.prefetch.offload_num_in_group})"
                    f" must be <= offload_group_size"
                    f" ({self.prefetch.offload_group_size})"
                )
            # 涓€鏃﹀惎鐢?prefetch锛屽氨瑕佹眰鑷冲皯棰勫彇 1 灞傘€?
            if self.prefetch.offload_prefetch_step < 1:
                raise ValueError(
                    f"offload_prefetch_step"
                    f" ({self.prefetch.offload_prefetch_step})"
                    f" must be >= 1 when prefetch offloading is enabled"
                    f" (offload_group_size > 0)"
                )

        # ----------------- 鍐嶆鏌モ€滃悗绔€夋嫨鈥濅笌鈥滃瓙閰嶇疆鏄惁婵€娲烩€濇槸鍚﹀啿绐?-----------------
        # Warn if both backends have non-default values
        uva_active = self.uva.cpu_offload_gb > 0
        prefetch_active = self.prefetch.offload_group_size > 0
        # 鏄惧紡鎸囧畾璧?UVA 鏃讹紝prefetch 瀛愰厤缃嵆浣胯濉簡涔熶笉浼氱敓鏁堛€?
        if self.offload_backend == "uva" and prefetch_active:
            warnings.warn(
                "Prefetch offload fields are set but offload_backend='uva'. "
                "Prefetch settings will be ignored.",
                stacklevel=2,
            )
        # 鏄惧紡鎸囧畾璧?prefetch 鏃讹紝UVA 瀛愰厤缃細琚拷鐣ャ€?
        elif self.offload_backend == "prefetch" and uva_active:
            warnings.warn(
                "UVA offload fields are set but offload_backend='prefetch'. "
                "UVA settings will be ignored.",
                stacklevel=2,
            )
        # auto 妯″紡涓嬭嫢涓よ竟閮借婵€娲伙紝浼氫紭鍏堥€夋嫨 prefetch锛屽苟缁欏嚭鎻愰啋銆?
        elif self.offload_backend == "auto" and uva_active and prefetch_active:
            warnings.warn(
                "Both UVA and prefetch offload fields are set with "
                "offload_backend='auto'. Prefetch backend will be selected. "
                "Set offload_backend explicitly to suppress this warning.",
                stacklevel=2,
            )
        if self.gpu_slots_per_layer and self.gpu_slots_per_layer < 8:
            raise ValueError("gpu_slots_per_layer must be 0 or at least top-k=8")
        # 杩斿洖鏍￠獙瀹屾垚鍚庣殑閰嶇疆瀵硅薄銆?
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

        # offload 閰嶇疆浼氬奖鍝?forward patching 鍜?prefetch 绱㈠紩璁＄畻锛屽洜姝ゅ叏閮ㄥ瓧娈甸兘鍙備笌鍝堝笇銆?
        factors = get_hash_factors(self, ignored_factors=set())
        hash_str = hash_factors(factors)
        return hash_str
