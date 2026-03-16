"""Default values for CFIE configuration objects."""

from __future__ import annotations

# 默认权重/计算精度：
# - auto: 由后端按设备能力自动选择（通常优先 bf16/fp16）。
DEFAULT_DTYPE = "auto"

# 单请求可用的最大上下文长度（单位：token）。
DEFAULT_MAX_MODEL_LEN = 4096

# 调度器同时运行的最大序列数；Phase 0/1 默认单序列。
DEFAULT_MAX_NUM_SEQS = 1

# 允许引擎使用的 GPU 显存占比阈值，取值范围 (0, 1]。
DEFAULT_GPU_MEMORY_UTILIZATION = 0.9

# 默认关闭权重量化。
DEFAULT_QUANTIZATION = "none"

# KV Cache 精度默认自动选择。
DEFAULT_KV_CACHE_DTYPE = "auto"

# 权重 offload 后端默认走 CPU 分层。
DEFAULT_WEIGHT_OFFLOAD_BACKEND = "cpu"

# KV offload 后端默认走 CPU 分层。
DEFAULT_KV_OFFLOAD_BACKEND = "cpu"

# CPU offload 预算（单位：GB）；0.0 表示不预留 CPU offload 容量。
DEFAULT_CPU_OFFLOAD_GB = 0.0

# NVMe offload 数据目录（仅在 nvme 后端启用时使用）。
DEFAULT_NVME_OFFLOAD_PATH = "/tmp/cfie_offload"

# 预取窗口大小（单位：层/步）；0 表示禁用预取。
DEFAULT_OFFLOAD_PREFETCH_WINDOW = 2

# 调度策略默认 FCFS（先到先服务）。
DEFAULT_SCHEDULER_POLICY = "fcfs"

# 允许的计算 dtype 集合。
SUPPORTED_DTYPES = ("auto", "fp16", "bf16")

# 允许的权重量化后端：
# - none: 不量化
# - gptq/awq/bnb: 对应各量化实现
SUPPORTED_QUANTIZATION = ("none", "gptq", "awq", "bnb")

# 允许的 KV Cache dtype 集合。
SUPPORTED_KV_CACHE_DTYPES = ("auto", "fp16", "fp8")

# 用户可选的权重 offload 后端：
# - cpu: 仅使用 CPU 作为 offload 层
# - nvme: 仅使用 NVMe 作为 offload 层
# - cpu+nvme: CPU + NVMe 两级分层
SUPPORTED_WEIGHT_OFFLOAD_BACKENDS = ("cpu", "nvme", "cpu+nvme")

# 用户可选的 KV offload 后端：
# - cpu: 仅使用 CPU 作为 offload 层
# - nvme: 仅使用 NVMe 作为 offload 层
# - cpu+nvme: CPU + NVMe 两级分层
SUPPORTED_KV_OFFLOAD_BACKENDS = ("cpu", "nvme", "cpu+nvme")

# 当前支持的调度策略集合（后续可扩展 priority 等）。
SUPPORTED_SCHEDULER_POLICIES = ("fcfs",)
