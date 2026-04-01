"""Dataclass-based configuration schema for CFIE."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from cfie.config import defaults
from cfie.config import validators


@dataclass(slots=True)
class ModelConfig:
    model: str = ""
    dtype: str = defaults.DEFAULT_DTYPE
    max_model_len: int = defaults.DEFAULT_MAX_MODEL_LEN

    def validate(self) -> "ModelConfig":
        validators.validate_non_empty_string("model", self.model)
        validators.validate_choice("dtype", self.dtype,
                                   defaults.SUPPORTED_DTYPES)
        validators.validate_positive_int("max_model_len", self.max_model_len)
        return self


@dataclass(slots=True)
class LoadConfig:
    revision: str | None = None
    trust_remote_code: bool = False
    local_files_only: bool = False
    load_format: str = defaults.DEFAULT_LOAD_FORMAT
    download_dir: str | None = None
    ignore_patterns: tuple[str, ...] = ()

    def validate(self) -> "LoadConfig":
        if self.revision is not None and not self.revision.strip():
            raise ValueError("revision must be None or a non-empty string")
        validators.validate_choice("load_format", self.load_format,
                                   defaults.SUPPORTED_LOAD_FORMATS)
        if self.download_dir is not None and not self.download_dir.strip():
            raise ValueError("download_dir must be None or a non-empty string")
        if not isinstance(self.ignore_patterns, tuple):
            raise ValueError("ignore_patterns must be a tuple of strings")
        for pattern in self.ignore_patterns:
            validators.validate_non_empty_string("ignore_patterns item", pattern)
        return self


@dataclass(slots=True)
class QuantConfig:
    quantization: str = defaults.DEFAULT_QUANTIZATION

    def validate(self) -> "QuantConfig":
        validators.validate_choice("quantization", self.quantization,
                                   defaults.SUPPORTED_QUANTIZATION)
        return self


@dataclass(slots=True)
class CacheConfig:
    kv_cache_dtype: str = defaults.DEFAULT_KV_CACHE_DTYPE

    def validate(self) -> "CacheConfig":
        validators.validate_choice("kv_cache_dtype", self.kv_cache_dtype,
                                   defaults.SUPPORTED_KV_CACHE_DTYPES)
        return self


@dataclass(slots=True)
class OffloadConfig:
    weight_offload_backend: str = defaults.DEFAULT_WEIGHT_OFFLOAD_BACKEND
    kv_offload_backend: str = defaults.DEFAULT_KV_OFFLOAD_BACKEND
    cpu_offload_gb: float = defaults.DEFAULT_CPU_OFFLOAD_GB
    moe_cpu_budget_gb: float = defaults.DEFAULT_MOE_CPU_BUDGET_GB
    moe_cpu_min_free_gb: float = defaults.DEFAULT_MOE_CPU_MIN_FREE_GB
    nvme_offload_path: str = defaults.DEFAULT_NVME_OFFLOAD_PATH
    offload_prefetch_window: int = defaults.DEFAULT_OFFLOAD_PREFETCH_WINDOW

    def validate(self) -> "OffloadConfig":
        # 后端取值必须落在文档定义的能力矩阵内。
        validators.validate_choice("weight_offload_backend",
                                   self.weight_offload_backend,
                                   defaults.SUPPORTED_WEIGHT_OFFLOAD_BACKENDS)
        validators.validate_choice("kv_offload_backend", self.kv_offload_backend,
                                   defaults.SUPPORTED_KV_OFFLOAD_BACKENDS)
        validators.validate_non_negative_float("cpu_offload_gb",
                                               self.cpu_offload_gb)
        validators.validate_non_negative_float("moe_cpu_budget_gb",
                                               self.moe_cpu_budget_gb)
        validators.validate_non_negative_float("moe_cpu_min_free_gb",
                                               self.moe_cpu_min_free_gb)
        validators.validate_positive_int("offload_prefetch_window",
                                         self.offload_prefetch_window,
                                         allow_zero=True)
        # 统一路径格式，避免下游模块重复处理路径展开逻辑。
        self.nvme_offload_path = validators.normalize_path(self.nvme_offload_path)

        # 仅当任一 offload 层级启用 NVMe 时，路径才是必填项。
        weight_use_nvme = self.weight_offload_backend in ("nvme", "cpu+nvme")
        kv_use_nvme = self.kv_offload_backend in ("nvme", "cpu+nvme")
        if weight_use_nvme or kv_use_nvme:
            validators.validate_non_empty_string("nvme_offload_path",
                                                 self.nvme_offload_path)

        # 仅 NVMe 模式下不应再配置 CPU offload 预算。
        if self.weight_offload_backend == "nvme" and self.cpu_offload_gb > 0:
            raise ValueError(
                "cpu_offload_gb must be 0 when weight_offload_backend is 'nvme'")
        return self


@dataclass(slots=True)
class SchedulerConfig:
    max_num_seqs: int = defaults.DEFAULT_MAX_NUM_SEQS
    policy: str = defaults.DEFAULT_SCHEDULER_POLICY

    def validate(self) -> "SchedulerConfig":
        validators.validate_positive_int("max_num_seqs", self.max_num_seqs)
        validators.validate_choice("policy", self.policy,
                                   defaults.SUPPORTED_SCHEDULER_POLICIES)
        return self


@dataclass(slots=True)
class RuntimeConfig:
    gpu_memory_utilization: float = defaults.DEFAULT_GPU_MEMORY_UTILIZATION

    def validate(self) -> "RuntimeConfig":
        validators.validate_ratio("gpu_memory_utilization",
                                  self.gpu_memory_utilization)
        return self


@dataclass(slots=True)
class EngineConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    load: LoadConfig = field(default_factory=LoadConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    offload: OffloadConfig = field(default_factory=OffloadConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def validate(self) -> "EngineConfig":
        # 校验顺序按依赖方向执行：
        # model/load/quant/cache/offload -> scheduler/runtime。
        self.model.validate()
        self.load.validate()
        self.quant.validate()
        self.cache.validate()
        self.offload.validate()
        self.scheduler.validate()
        self.runtime.validate()
        return self

    @classmethod
    def from_flat_kwargs(
        cls,
        *,
        model: str,
        revision: str | None = None,
        trust_remote_code: bool = False,
        local_files_only: bool = False,
        load_format: str = defaults.DEFAULT_LOAD_FORMAT,
        download_dir: str | None = None,
        ignore_patterns: tuple[str, ...] = (),
        dtype: str = defaults.DEFAULT_DTYPE,
        max_model_len: int = defaults.DEFAULT_MAX_MODEL_LEN,
        max_num_seqs: int = defaults.DEFAULT_MAX_NUM_SEQS,
        gpu_memory_utilization: float = defaults.DEFAULT_GPU_MEMORY_UTILIZATION,
        quantization: str = defaults.DEFAULT_QUANTIZATION,
        kv_cache_dtype: str = defaults.DEFAULT_KV_CACHE_DTYPE,
        weight_offload_backend: str = defaults.DEFAULT_WEIGHT_OFFLOAD_BACKEND,
        kv_offload_backend: str = defaults.DEFAULT_KV_OFFLOAD_BACKEND,
        cpu_offload_gb: float = defaults.DEFAULT_CPU_OFFLOAD_GB,
        moe_cpu_budget_gb: float = defaults.DEFAULT_MOE_CPU_BUDGET_GB,
        moe_cpu_min_free_gb: float = defaults.DEFAULT_MOE_CPU_MIN_FREE_GB,
        nvme_offload_path: str = defaults.DEFAULT_NVME_OFFLOAD_PATH,
        offload_prefetch_window: int = defaults.DEFAULT_OFFLOAD_PREFETCH_WINDOW,
    ) -> "EngineConfig":
        # 扁平构造函数供 CLI 使用，字段名需与命令行参数保持一致。
        config = cls(
            model=ModelConfig(model=model,
                              dtype=dtype,
                              max_model_len=max_model_len),
            load=LoadConfig(
                revision=revision,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
                load_format=load_format,
                download_dir=download_dir,
                ignore_patterns=ignore_patterns,
            ),
            quant=QuantConfig(quantization=quantization),
            cache=CacheConfig(kv_cache_dtype=kv_cache_dtype),
            offload=OffloadConfig(
                weight_offload_backend=weight_offload_backend,
                kv_offload_backend=kv_offload_backend,
                cpu_offload_gb=cpu_offload_gb,
                moe_cpu_budget_gb=moe_cpu_budget_gb,
                moe_cpu_min_free_gb=moe_cpu_min_free_gb,
                nvme_offload_path=nvme_offload_path,
                offload_prefetch_window=offload_prefetch_window,
            ),
            scheduler=SchedulerConfig(max_num_seqs=max_num_seqs),
            runtime=RuntimeConfig(
                gpu_memory_utilization=gpu_memory_utilization),
        )
        return config.validate()

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "EngineConfig":
        # 兼容模式：直接接受历史的扁平字典格式。
        if isinstance(raw.get("model"), str):
            return cls.from_flat_kwargs(
                model=raw["model"],
                revision=raw.get("revision"),
                trust_remote_code=raw.get("trust_remote_code", False),
                local_files_only=raw.get("local_files_only", False),
                load_format=raw.get("load_format", defaults.DEFAULT_LOAD_FORMAT),
                download_dir=raw.get("download_dir"),
                ignore_patterns=tuple(raw.get("ignore_patterns", ())),
                dtype=raw.get("dtype", defaults.DEFAULT_DTYPE),
                max_model_len=raw.get("max_model_len",
                                      defaults.DEFAULT_MAX_MODEL_LEN),
                max_num_seqs=raw.get("max_num_seqs",
                                     defaults.DEFAULT_MAX_NUM_SEQS),
                gpu_memory_utilization=raw.get(
                    "gpu_memory_utilization",
                    defaults.DEFAULT_GPU_MEMORY_UTILIZATION,
                ),
                quantization=raw.get("quantization",
                                     defaults.DEFAULT_QUANTIZATION),
                kv_cache_dtype=raw.get("kv_cache_dtype",
                                       defaults.DEFAULT_KV_CACHE_DTYPE),
                weight_offload_backend=raw.get(
                    "weight_offload_backend",
                    defaults.DEFAULT_WEIGHT_OFFLOAD_BACKEND,
                ),
                kv_offload_backend=raw.get("kv_offload_backend",
                                           defaults.DEFAULT_KV_OFFLOAD_BACKEND),
                cpu_offload_gb=raw.get("cpu_offload_gb",
                                       defaults.DEFAULT_CPU_OFFLOAD_GB),
                moe_cpu_budget_gb=raw.get(
                    "moe_cpu_budget_gb",
                    defaults.DEFAULT_MOE_CPU_BUDGET_GB,
                ),
                moe_cpu_min_free_gb=raw.get(
                    "moe_cpu_min_free_gb",
                    defaults.DEFAULT_MOE_CPU_MIN_FREE_GB,
                ),
                nvme_offload_path=raw.get("nvme_offload_path",
                                          defaults.DEFAULT_NVME_OFFLOAD_PATH),
                offload_prefetch_window=raw.get(
                    "offload_prefetch_window",
                    defaults.DEFAULT_OFFLOAD_PREFETCH_WINDOW,
                ),
            )

        # 结构化模式：嵌套字典与各子配置 dataclass 一一对应。
        model_raw = raw.get("model", {})
        if not isinstance(model_raw, Mapping):
            raise ValueError("model must be either string or mapping")

        config = cls(
            model=ModelConfig(**model_raw),
            load=LoadConfig(**raw.get("load", {})),
            quant=QuantConfig(**raw.get("quant", {})),
            cache=CacheConfig(**raw.get("cache", {})),
            offload=OffloadConfig(**raw.get("offload", {})),
            scheduler=SchedulerConfig(**raw.get("scheduler", {})),
            runtime=RuntimeConfig(**raw.get("runtime", {})),
        )
        return config.validate()
