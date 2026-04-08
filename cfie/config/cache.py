# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import field
from typing import ClassVar, Literal

from pydantic import Field, SkipValidation, field_validator, model_validator

from cfie.config.utils import config
from cfie.logger import init_logger

logger = init_logger(__name__)

# KV cache 可选存储 dtype。
CacheDType = Literal[
    "auto",
    "bfloat16",
    "fp8",
    "fp8_e4m3",
    "fp8_e5m2",
    "fp8_inc",
    "fp8_ds_mla",
]
# Mamba cache 可选 dtype。
MambaDType = Literal["auto", "float32", "float16"]
# Mamba cache 策略。
MambaCacheMode = Literal["all", "align", "none"]
# prefix caching 使用的哈希算法。
PrefixCachingHashAlgo = Literal["sha256", "sha256_cbor", "xxhash", "xxhash_cbor"]
# KV offloading backend 类型。
KVOffloadingBackend = Literal["native", "lmcache"]


@config
class CacheConfig:
    """Configuration for the KV cache."""

    # 未显式指定 block_size 时使用的默认块大小。
    DEFAULT_BLOCK_SIZE: ClassVar[int] = 16

    # KV block 大小；允许先传 None，构造后会在 validator 中补默认值。
    block_size: SkipValidation[int] = None  # type: ignore[assignment]
    """Size of a contiguous cache block in number of tokens.
    Accepts None (meaning "use default"). After construction, always int."""
    # 记录 block_size 是否由用户显式指定。
    user_specified_block_size: bool = field(default=False, init=False)
    """Whether block_size was explicitly provided. Derived automatically."""
    # 当前实例的静态显存预算比例。
    gpu_memory_utilization: float = Field(default=0.9, gt=0, le=1)
    """The fraction of GPU memory reserved for statically planned contents:
    model weights, KV cache, and MoE slot planning. The remaining fraction is
    reserved for runtime peak memory such as activations and temporary
    workspaces. This is a per-instance target and only applies to the current
    vLLM instance."""
    # KV cache 的存储 dtype。
    cache_dtype: CacheDType = "auto"
    """Data type for kv cache storage. If "auto", will use model data type.
    CUDA 11.8+ supports fp8 (=fp8_e4m3) and fp8_e5m2. ROCm (AMD GPU) supports
    fp8 (=fp8_e4m3). Intel Gaudi (HPU) supports fp8 (using fp8_inc).
    Some models (namely DeepSeekV3.2) default to fp8, set to bfloat16 to use
    bfloat16 instead, this is an invalid option for models that do not default
    to fp8.
    """
    # 是否是 attention-free 模型。
    is_attention_free: bool = False
    """Whether the model is attention-free. This is primarily set in
    `ModelConfig` and that value should be manually duplicated here."""
    # 测试场景下手工覆盖 GPU block 数。
    num_gpu_blocks_override: int | None = None
    """Number of GPU blocks to use. This overrides the profiled `num_gpu_blocks`
    if specified. Does nothing if `None`. Used for testing preemption."""
    # sliding window 大小。
    sliding_window: int | None = None
    """Sliding window size for the KV cache. This is primarily set in
    `ModelConfig` and that value should be manually duplicated here."""
    # 是否启用 prefix caching。
    enable_prefix_caching: bool = True
    """Whether to enable prefix caching."""
    # prefix caching 的哈希算法。
    prefix_caching_hash_algo: PrefixCachingHashAlgo = "sha256"
    """Set the hash algorithm for prefix caching:\n
    - "sha256" uses Pickle for object serialization before hashing. This is the
    current default, as SHA256 is the most secure choice to avoid potential
    hash collisions.\n
    - "sha256_cbor" provides a reproducible, cross-language compatible hash. It
    serializes objects using canonical CBOR and hashes them with SHA-256.\n
    - "xxhash" uses Pickle serialization with xxHash (128-bit) for faster,
    non-cryptographic hashing. Requires the optional ``xxhash`` package.
    IMPORTANT: Use of a hashing algorithm that is not considered 
    cryptographically secure theoretically increases the risk of hash collisions,
    which can cause undefined behavior or even leak private information in
    multi-tenant environments. Even if collisions are still very unlikely, it is
    important to consider your security risk tolerance against the performance
    benefits before turning this on.\n
    - "xxhash_cbor" combines canonical CBOR serialization with xxHash for
    reproducible hashing. Requires the optional ``xxhash`` package."""
    # fp8 KV cache 时是否动态计算 k/v scale。
    calculate_kv_scales: bool = False
    """This enables dynamic calculation of `k_scale` and `v_scale` when
    kv_cache_dtype is fp8. If `False`, the scales will be loaded from the model
    checkpoint if available. Otherwise, the scales will default to 1.0."""
    # CPU backend 使用的 CPU KV cache 总空间。
    cpu_kvcache_space_bytes: int | None = None
    """(CPU backend only) CPU key-value cache space."""
    # hybrid mamba/attention 模型下的 mamba page size 覆写值。
    mamba_page_size_padded: int | None = None
    """ Optional override for mamba page size; used by hybrid mamba/attention
    models to ensure exact alignment with attention page size."""
    # Mamba cache block 大小。
    mamba_block_size: int | None = Field(default=None, gt=0)
    """Size of a contiguous cache block in number of tokens for mamba cache.
    Can be set only when prefix caching is enabled.
    Value must be a multiple of 8 to align with causal_conv1d kernel."""
    # conv + ssm 共享的 Mamba cache dtype。
    mamba_cache_dtype: MambaDType = "auto"
    """The data type to use for the Mamba cache (both the conv as well as the
    ssm state). If set to 'auto', the data type will be inferred from the model
    config."""
    # 仅 ssm state 的 Mamba cache dtype。
    mamba_ssm_cache_dtype: MambaDType = "auto"
    """The data type to use for the Mamba cache (ssm state only, conv state will
    still be controlled by mamba_cache_dtype). If set to 'auto', the data type
    for the ssm state will be determined by mamba_cache_dtype."""
    # Mamba cache 策略。
    mamba_cache_mode: MambaCacheMode = "none"
    """The cache strategy for Mamba layers.
    - "none": set when prefix caching is disabled.
    - "all": cache the mamba state of all tokens at position i * block_size. This is 
           the default behavior (for models that support it) when prefix caching is
           enabled.
    - "align": only cache the mamba state of the last token of each scheduler step and
           when the token is at position i * block_size.
    """

    # Will be set after profiling.
    # profile 完成后得到的 GPU block 数。
    num_gpu_blocks: int | None = field(default=None, init=False)
    """The number of blocks to allocate for GPU memory."""
    # profile 完成后得到的 CPU block 数。
    num_cpu_blocks: int | None = field(default=None, init=False)
    """The number of blocks to allocate for CPU memory."""

    # 当前仍在开发中的 fast prefill 共享 KV 开关。
    kv_sharing_fast_prefill: bool = False
    """This feature is work in progress and no prefill optimization takes place
    with this flag enabled currently.

    In some KV sharing setups, e.g. YOCO (https://arxiv.org/abs/2405.05254),
    some layers can skip tokens corresponding to prefill. This flag enables
    attention metadata for eligible layers to be overridden with metadata
    necessary for implementing this optimization in some models (e.g. Gemma3n)
    """

    # 手工指定每张 GPU 上 KV cache 总字节数。
    kv_cache_memory_bytes: int | None = None
    """Size of KV Cache per GPU in bytes. By default, this is set to None
    and cfie can automatically infer the kv cache size based on the static
    budget implied by gpu_memory_utilization. However, users may want to manually specify
    the kv cache memory size. kv_cache_memory_bytes allows more fine-grain
    control of how much memory gets used when compared with using
    gpu_memory_utilization. Note that kv_cache_memory_bytes
    (when not-None) ignores gpu_memory_utilization"""

    # KV offloading buffer 大小，单位 GiB。
    kv_offloading_size: float | None = None
    """Size of the KV cache offloading buffer in GiB. When TP > 1, this is
    the total buffer size summed across all TP ranks. By default, this is set
    to None, which means no KV offloading is enabled. When set, vLLM will
    enable KV cache offloading to CPU using the kv_offloading_backend."""

    # KV cache offloading 的底层后端。
    kv_offloading_backend: KVOffloadingBackend = "native"
    """The backend to use for KV cache offloading. Supported backends include
    'native' (vLLM native CPU offloading), 'lmcache'.
    KV offloading is only activated when kv_offloading_size is set."""

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
        # 这些字段只影响运行时预算或派生值，不影响编译图结构。
        ignored_factors = {
            # Runtime/derived knobs that don't affect compiled graph shape
            "gpu_memory_utilization",
            "is_attention_free",
            "num_gpu_blocks_override",
            "enable_prefix_caching",
            "prefix_caching_hash_algo",
            "cpu_kvcache_space_bytes",
            "mamba_page_size_padded",
            "user_specified_block_size",
            "_block_size_resolved",
            # Post-init/derived counters
            "num_gpu_blocks",
            "num_cpu_blocks",
            # WIP feature toggle not impacting compiled graph shape
            "kv_sharing_fast_prefill",
        }

        from cfie.config.utils import get_hash_factors, hash_factors

        # 对其余字段做哈希，供 compilation cache 区分不同 KV/cache 配置。
        factors = get_hash_factors(self, ignored_factors)
        return hash_factors(factors)

    def metrics_info(self):
        # 把 cache_config 全部字段转成字符串字典，供 Prometheus 指标导出。
        # convert cache_config to dict(key: str, value: str) for prometheus
        # metrics info
        return {key: str(value) for key, value in self.__dict__.items()}

    # 防止嵌套 pydantic 场景下重复执行 block_size 默认值逻辑。
    _block_size_resolved: bool = field(default=False, init=False)
    """Guard against pydantic re-running _apply_block_size_default."""

    @model_validator(mode="after")
    def _apply_block_size_default(self) -> "CacheConfig":
        # ----------------- 统一处理 block_size 的默认值与“是否用户指定”标记 -----------------
        # Pydantic re-runs validators when CacheConfig is nested inside
        # another pydantic model (e.g. CfieConfig). Guard against that.
        if self._block_size_resolved:
            return self
        object.__setattr__(self, "_block_size_resolved", True)
        # 未指定时补 DEFAULT_BLOCK_SIZE。
        if self.block_size is None:
            object.__setattr__(self, "block_size", self.DEFAULT_BLOCK_SIZE)
        else:
            # 否则记下这是用户显式指定的 block_size。
            object.__setattr__(self, "user_specified_block_size", True)
        return self

    @field_validator("cache_dtype", mode="after")
    @classmethod
    def _validate_cache_dtype(cls, cache_dtype: CacheDType) -> CacheDType:
        # fp8 KV cache 会降低显存占用，但可能引入精度损失，因此这里主动打提示日志。
        if cache_dtype.startswith("fp8"):
            logger.info(
                "Using fp8 data type to store kv cache. It reduces the GPU "
                "memory footprint and boosts the performance. "
                "Meanwhile, it may cause accuracy drop without a proper "
                "scaling factor."
            )
        # 返回归一化后的 cache_dtype。
        return cache_dtype
