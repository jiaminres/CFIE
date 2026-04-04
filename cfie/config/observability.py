# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import cached_property
from typing import Any, Literal, cast

from packaging.version import parse
from pydantic import Field, field_validator, model_validator

from cfie import version
from cfie.config.utils import config
from cfie.utils.hashing import safe_hash

# 详细 trace 时允许指定的模块集合。
DetailedTraceModules = Literal["model", "worker", "all"]


@config
class ObservabilityConfig:
    """Configuration for observability - metrics and tracing."""

    # 允许临时重新暴露“在某版本后被隐藏”的旧 Prometheus 指标。
    show_hidden_metrics_for_version: str | None = None
    """Enable deprecated Prometheus metrics that have been hidden since the
    specified version. For example, if a previously deprecated metric has been
    hidden since the v0.7.0 release, you use
    `--show-hidden-metrics-for-version=0.7` as a temporary escape hatch while
    you migrate to new metrics. The metric is likely to be removed completely
    in an upcoming release."""

    @cached_property
    def show_hidden_metrics(self) -> bool:
        """Check if the hidden metrics should be shown."""
        # 未给出版本号时，默认不显示隐藏指标。
        if self.show_hidden_metrics_for_version is None:
            return False
        # 若指定版本满足“上一个 minor 版本”规则，则重新展示隐藏指标。
        return version._prev_minor_version_was(self.show_hidden_metrics_for_version)

    # OTLP trace 导出的目标 endpoint。
    otlp_traces_endpoint: str | None = None
    """Target URL to which OpenTelemetry traces will be sent."""

    # 是否采集更细粒度的 model/worker trace。
    collect_detailed_traces: list[DetailedTraceModules] | None = None
    """It makes sense to set this only if `--otlp-traces-endpoint` is set. If
    set, it will collect detailed traces for the specified modules. This
    involves use of possibly costly and or blocking operations and hence might
    have a performance impact.

    Note that collecting detailed timing information for each request can be
    expensive."""

    # 是否启用 KV cache 指标采样。
    kv_cache_metrics: bool = False
    """Enable KV cache residency metrics (lifetime, idle time, reuse gaps).
    Uses sampling to minimize overhead.
    Requires log stats to be enabled (i.e., --disable-log-stats not set)."""

    # KV cache 指标采样率。
    kv_cache_metrics_sample: float = Field(default=0.01, gt=0, le=1)
    """Sampling rate for KV cache metrics (0.0, 1.0]. Default 0.01 = 1% of blocks."""

    # 是否启用 CUDA graph 相关指标。
    cudagraph_metrics: bool = False
    """Enable CUDA graph metrics (number of padded/unpadded tokens, runtime cudagraph
    dispatch modes, and their observed frequencies at every logging interval)."""

    # 是否按层打 NVTX range。
    enable_layerwise_nvtx_tracing: bool = False
    """Enable layerwise NVTX tracing. This traces the execution of each layer or
    module in the model and attach information such as input/output shapes to
    nvtx range markers. Noted that this doesn't work with CUDA graphs enabled."""

    # 是否采集 MFU 指标。
    enable_mfu_metrics: bool = False
    """Enable Model FLOPs Utilization (MFU) metrics."""

    # 是否采集多模态 processor 耗时统计。
    enable_mm_processor_stats: bool = False
    """Enable collection of timing statistics for multimodal processor operations.
    This is for internal use only (e.g., benchmarks) and is not exposed as a CLI
    argument."""

    # 是否记录 iteration 级别的详细日志。
    enable_logging_iteration_details: bool = False
    """Enable detailed logging of iteration details.
    If set, cfie EngineCore will log iteration details
    This includes number of context/generation requests and tokens
    and the elapsed cpu time for the iteration."""

    @cached_property
    def collect_model_forward_time(self) -> bool:
        """Whether to collect model forward time for the request."""
        # 只要 detailed traces 中包含 model 或 all，就采集 forward 时间。
        return self.collect_detailed_traces is not None and (
            "model" in self.collect_detailed_traces
            or "all" in self.collect_detailed_traces
        )

    @cached_property
    def collect_model_execute_time(self) -> bool:
        """Whether to collect model execute time for the request."""
        # 只要 detailed traces 中包含 worker 或 all，就采集 execute 时间。
        return self.collect_detailed_traces is not None and (
            "worker" in self.collect_detailed_traces
            or "all" in self.collect_detailed_traces
        )

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
        # this config will not affect the computation graph.
        # observability 配置不会改变模型图结构。
        factors: list[Any] = []
        # 用空 factors 生成稳定哈希。
        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
        # 返回最终哈希值。
        return hash_str

    @field_validator("show_hidden_metrics_for_version")
    @classmethod
    def _validate_show_hidden_metrics_for_version(cls, value: str | None) -> str | None:
        # 若给出了版本号，则先验证其格式合法。
        if value is not None:
            # Raises an exception if the string is not a valid version.
            parse(value)
        # 合法后原样返回。
        return value

    @field_validator("otlp_traces_endpoint")
    @classmethod
    def _validate_otlp_traces_endpoint(cls, value: str | None) -> str | None:
        # 配置了 OTLP endpoint 时，必须先确认 tracing 依赖可用。
        if value is not None:
            from cfie.tracing import is_tracing_available, otel_import_error_traceback

            if not is_tracing_available():
                raise ValueError(
                    "OpenTelemetry is not available. Unable to configure "
                    "'otlp_traces_endpoint'. Ensure OpenTelemetry packages are "
                    f"installed. Original error:\n{otel_import_error_traceback}"
                )
        # tracing 可用时原样返回 endpoint。
        return value

    @field_validator("collect_detailed_traces")
    @classmethod
    def _validate_collect_detailed_traces(
        cls, value: list[DetailedTraceModules] | None
    ) -> list[DetailedTraceModules] | None:
        """Handle the legacy case where users might provide a comma-separated
        string instead of a list of strings."""
        # 兼容老格式：单元素列表里塞了一个逗号分隔字符串。
        if value is not None and len(value) == 1 and "," in value[0]:
            value = cast(list[DetailedTraceModules], value[0].split(","))
        # 返回标准化后的 detailed trace 列表。
        return value

    @model_validator(mode="after")
    def _validate_tracing_config(self):
        # detailed traces 依赖 OTLP endpoint，因此必须同时配置。
        if self.collect_detailed_traces and not self.otlp_traces_endpoint:
            raise ValueError(
                "collect_detailed_traces requires `--otlp-traces-endpoint` to be set."
            )
        # 通过校验后返回当前对象。
        return self
