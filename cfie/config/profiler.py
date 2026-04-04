# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import Any, Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from cfie.config.utils import config
from cfie.logger import init_logger
from cfie.utils.hashing import safe_hash

logger = init_logger(__name__)

# profiler 类型枚举。
ProfilerKind = Literal["torch", "cuda"]


def _is_uri_path(path: str) -> bool:
    """Check if path is a URI (scheme://...), excluding Windows drive letters.

    Supports custom URI schemes like gs://, s3://, hdfs://, etc.
    These paths should not be converted to absolute paths.
    """
    # 只有包含 :// 的路径才有可能是 URI。
    if "://" in path:
        # 取出 scheme 部分。
        scheme = path.split("://")[0]
        # Windows drive letters are single characters (e.g., C://)
        # Valid URI schemes have more than one character
        # 单字符 scheme 视作 Windows 盘符，不当成 URI。
        return len(scheme) > 1
    # 普通本地路径返回 False。
    return False


@config
class ProfilerConfig:
    """Dataclass which contains profiler config for the engine."""

    # 选择使用 torch profiler 还是 cuda profiler。
    profiler: ProfilerKind | None = None
    """Which profiler to use. Defaults to None. Options are:

    - 'torch': Use PyTorch profiler.\n
    - 'cuda': Use CUDA profiler."""

    # torch profiler trace 的输出目录。
    torch_profiler_dir: str = ""
    """Directory to save torch profiler traces. Both AsyncLLM's CPU traces and
    worker's traces (CPU & GPU) will be saved under this directory. Note that
    it must be an absolute path."""

    # 是否在 torch profiler 中记录 Python stack。
    torch_profiler_with_stack: bool = False
    """If `True`, enables stack tracing in the torch profiler. Disabled by default
    to reduce overhead. Can be enabled via VLLM_TORCH_PROFILER_WITH_STACK=1 env var
    or --profiler-config.torch_profiler_with_stack=true CLI flag."""

    # 是否统计 FLOPS。
    torch_profiler_with_flops: bool = False
    """If `True`, enables FLOPS counting in the torch profiler. Disabled by default."""

    # 是否用 gzip 压缩 trace 文件。
    torch_profiler_use_gzip: bool = True
    """If `True`, saves torch profiler traces in gzip format. Enabled by default"""

    # 是否在 trace 中额外导出总 CUDA 时间。
    torch_profiler_dump_cuda_time_total: bool = True
    """If `True`, dumps total CUDA time in torch profiler traces. Enabled by default."""

    # 是否记录张量 shape。
    torch_profiler_record_shapes: bool = False
    """If `True`, records tensor shapes in the torch profiler. Disabled by default."""

    # 是否开启内存分析。
    torch_profiler_with_memory: bool = False
    """If `True`, enables memory profiling in the torch profiler.
    Disabled by default."""

    # 是否跳过 AsyncLLM 前端的 profiler。
    ignore_frontend: bool = False
    """If `True`, disables the front-end profiling of AsyncLLM when using the
    'torch' profiler. This is needed to reduce overhead when using delay/limit options,
    since the front-end profiling does not track iterations and will capture the
    entire range.
    """

    # profiler 真正开始前，先跳过多少个 engine iteration。
    delay_iterations: int = Field(default=0, ge=0)
    """Number of engine iterations to skip before starting profiling.
    Defaults to 0, meaning profiling starts immediately after receiving /start_profile.
    """

    # profiler 最多记录多少个 iteration；0 表示不限。
    max_iterations: int = Field(default=0, ge=0)
    """Maximum number of engine iterations to profile after starting profiling.
    Defaults to 0, meaning no limit.
    """

    # torch profiler schedule 里的 warmup iteration 数量。
    warmup_iterations: int = Field(default=0, ge=0)
    """Number of warmup iterations for PyTorch profiler schedule.
    During warmup, the profiler runs but data is discarded. This helps reduce
    noise from JIT compilation and other one-time costs in the profiled trace.
    Defaults to 0 (schedule-based profiling disabled, recording all iterations).
    Set to a positive value (e.g., 2) to enable schedule-based profiling.
    """

    # torch profiler schedule 里的 active iteration 数量。
    active_iterations: int = Field(default=5, ge=1)
    """Number of active iterations for PyTorch profiler schedule.
    This is the number of iterations where profiling data is actually collected.
    Defaults to 5 active iterations.
    """

    # torch profiler schedule 里的 wait iteration 数量。
    wait_iterations: int = Field(default=0, ge=0)
    """Number of wait iterations for PyTorch profiler schedule.
    During wait, the profiler is completely off with zero overhead.
    This allows skipping initial iterations before warmup begins.
    Defaults to 0 (no wait period).
    """

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
        # profiler 配置不影响模型图结构，因此 factors 为空。
        factors: list[Any] = []
        # 用空 factors 生成稳定哈希。
        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
        # 返回最终哈希字符串。
        return hash_str

    @model_validator(mode="after")
    def _validate_profiler_config(self) -> Self:
        # 只要配置了 delay/max_iterations，就视作启用了延迟或限长 profiling。
        has_delay_or_limit = self.delay_iterations > 0 or self.max_iterations > 0
        # torch profiler 在这两种选项与前端 profiling 同开时开销会偏高，先打告警。
        if self.profiler == "torch" and has_delay_or_limit and not self.ignore_frontend:
            logger.warning_once(
                "Using 'torch' profiler with delay_iterations or max_iterations "
                "while ignore_frontend is False may result in high overhead."
            )

        # 读取 profiler 输出目录。
        profiler_dir = self.torch_profiler_dir
        # 只有 torch profiler 才允许指定 torch_profiler_dir。
        if profiler_dir and self.profiler != "torch":
            raise ValueError(
                "torch_profiler_dir is only applicable when profiler is set to 'torch'"
            )
        # 反过来，使用 torch profiler 时必须显式给出输出目录。
        if self.profiler == "torch" and not profiler_dir:
            raise ValueError("torch_profiler_dir must be set when profiler is 'torch'")

        # Support any URI scheme (gs://, s3://, hdfs://, etc.)
        # These paths should not be converted to absolute paths
        # 本地路径统一规范化为绝对路径；URI 路径保持原样。
        if profiler_dir and not _is_uri_path(profiler_dir):
            self.torch_profiler_dir = os.path.abspath(os.path.expanduser(profiler_dir))

        # 通过校验后返回当前对象。
        return self
