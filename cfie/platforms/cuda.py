# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
这个文件里的代码可以默认认为当前平台是 CUDA，
例如可以安全地 import pynvml。
但是，它不应该主动初始化 CUDA context。
"""

import os
from collections.abc import Callable
from datetime import timedelta
from functools import cache, wraps
from typing import TYPE_CHECKING, TypeVar

import torch
from torch.distributed import PrefixStore, ProcessGroup
from torch.distributed.distributed_c10d import is_nccl_available
from typing_extensions import ParamSpec

from cfie.logger import init_logger
from cfie.utils.import_utils import import_pynvml
from cfie.utils.torch_utils import cuda_device_count_stateless
from cfie.v1.attention.backends.registry import AttentionBackendEnum

from .interface import DeviceCapability, Platform, PlatformEnum

# 仅做类型检查时导入，避免运行时循环依赖
if TYPE_CHECKING:
    from cfie.config import CfieConfig
    from cfie.config.cache import CacheDType
    from cfie.v1.attention.selector import AttentionSelectorConfig
else:
    CfieConfig = None
    CacheDType = None

# 初始化日志器
logger = init_logger(__name__)

# ParamSpec / TypeVar 用于给装饰器保留原函数签名
_P = ParamSpec("_P")
_R = TypeVar("_R")

# 导入自定义 C++/CUDA op，触发算子注册
try:
    import cfie._C  # noqa: F401
except ImportError as e:
    logger.warning("Failed to import from cfie._C with %r", e)

# 导入 pynvml
# 这是 NVIDIA 管理库的 Python 封装，用来查询 GPU 信息
pynvml = import_pynvml()

# PyTorch 2.5 默认可能启用 cudnn sdpa，
# 某些模型上会崩，所以这里主动关闭
torch.backends.cuda.enable_cudnn_sdp(False)


@cache
def _get_backend_priorities(
    use_mla: bool,
    device_capability: DeviceCapability,
    num_heads: int | None = None,
) -> list[AttentionBackendEnum]:
    """获取 attention backend 的优先级列表。使用延迟导入避免循环依赖。"""
    if use_mla:
        # MLA 路径
        if device_capability.major == 10:
            # 计算能力 10.x 的设备上，低 head 数时优先 FlashInfer
            # 因为 FlashMLA 存在 padding 开销
            if num_heads is not None and num_heads <= 16:
                sparse_backends = [
                    AttentionBackendEnum.FLASHINFER_MLA_SPARSE,
                    AttentionBackendEnum.FLASHMLA_SPARSE,
                ]
            else:
                sparse_backends = [
                    AttentionBackendEnum.FLASHMLA_SPARSE,
                    AttentionBackendEnum.FLASHINFER_MLA_SPARSE,
                ]
            return [
                AttentionBackendEnum.FLASHINFER_MLA,
                AttentionBackendEnum.CUTLASS_MLA,
                AttentionBackendEnum.FLASH_ATTN_MLA,
                AttentionBackendEnum.FLASHMLA,
                AttentionBackendEnum.TRITON_MLA,
                *sparse_backends,
            ]
        else:
            # 非 10.x 设备上的 MLA 后端优先级
            return [
                AttentionBackendEnum.FLASH_ATTN_MLA,
                AttentionBackendEnum.FLASHMLA,
                AttentionBackendEnum.FLASHINFER_MLA,
                AttentionBackendEnum.TRITON_MLA,
                AttentionBackendEnum.FLASHMLA_SPARSE,
            ]
    else:
        # 普通 attention 路径
        if device_capability.major == 10:
            return [
                AttentionBackendEnum.FLASHINFER,
                AttentionBackendEnum.FLASH_ATTN,
                AttentionBackendEnum.TRITON_ATTN,
                AttentionBackendEnum.FLEX_ATTENTION,
            ]
        else:
            return [
                AttentionBackendEnum.FLASH_ATTN,
                AttentionBackendEnum.FLASHINFER,
                AttentionBackendEnum.TRITON_ATTN,
                AttentionBackendEnum.FLEX_ATTENTION,
            ]


def with_nvml_context(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    """
    给函数包一层 NVML 上下文：
    调用前 nvmlInit()，调用后 nvmlShutdown()。
    这样每次查询 GPU 信息时都保证 NVML 已初始化。
    """
    @wraps(fn)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        pynvml.nvmlInit()
        try:
            return fn(*args, **kwargs)
        finally:
            pynvml.nvmlShutdown()

    return wrapper


class CudaPlatformBase(Platform):
    # 当前平台枚举类型：CUDA
    _enum = PlatformEnum.CUDA

    # 设备名字 / 类型 / dispatch key 等平台标识
    device_name: str = "cuda"
    device_type: str = "cuda"
    dispatch_key: str = "CUDA"
    ray_device_key: str = "GPU"

    # 分布式后端默认使用 NCCL
    dist_backend: str = "nccl"

    # 控制当前进程可见 GPU 的环境变量
    device_control_env_var: str = "CUDA_VISIBLE_DEVICES"

    # Ray 场景下不设置设备环境变量时用到的环境变量名
    ray_noset_device_env_vars: list[str] = [
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
    ]

    @property
    def supported_dtypes(self) -> list[torch.dtype]:
        """
        返回当前平台支持的数据类型列表。
        """
        if self.has_device_capability(80):
            # Ampere/Hopper 及以后，支持 BF16/FP16/FP32
            return [torch.bfloat16, torch.float16, torch.float32]
        if self.has_device_capability(60):
            # Pascal/Volta/Turing，不支持 BF16
            return [torch.float16, torch.float32]
        # 更老架构只认为支持 FP32
        return [torch.float32]

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        """
        设置当前 CUDA 设备。
        """
        torch.cuda.set_device(device)

        # 通过创建一个张量，强制设备尽早真正初始化
        _ = torch.zeros(1, device=device)

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability | None:
        # 子类实现：查询设备 compute capability
        raise NotImplementedError

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        # 子类实现：查询设备名称
        raise NotImplementedError

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        # 子类实现：查询总显存
        raise NotImplementedError

    @classmethod
    def is_fully_connected(cls, device_ids: list[int]) -> bool:
        # 子类实现：判断多卡之间是否完全互联（例如全部 NVLink 直连）
        raise NotImplementedError

    @classmethod
    def log_warnings(cls):
        # 子类可选实现：打印平台相关告警
        pass

    @classmethod
    def check_and_update_config(cls, cfie_config: "CfieConfig") -> None:
        """
        根据 CUDA 平台特性，检查并修正配置。
        """
        parallel_config = cfie_config.parallel_config
        model_config = cfie_config.model_config

        # 若 worker 类自动选择，则指定为 GPU worker
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "cfie.v1.worker.gpu_worker.Worker"

        scheduler_config = cfie_config.scheduler_config

        # 多模态双向注意力模型上，强制禁用 chunked_mm_input
        if (
            model_config is not None
            and model_config.is_mm_prefix_lm
            and scheduler_config.is_multimodal_model
            and not scheduler_config.disable_chunked_mm_input
        ):
            logger.warning(
                "Forcing --disable_chunked_mm_input for models "
                "with multimodal-bidirectional attention."
            )
            scheduler_config.disable_chunked_mm_input = True

    @classmethod
    def get_current_memory_usage(
        cls, device: torch.types.Device | None = None
    ) -> float:
        """
        返回当前设备上的峰值显存占用。
        调用前会 empty_cache 并 reset peak 统计。
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        return torch.cuda.max_memory_allocated(device)

    @classmethod
    def get_valid_backends(
        cls,
        device_capability: DeviceCapability,
        attn_selector_config: "AttentionSelectorConfig",
        num_heads: int | None = None,
    ) -> tuple[
        list[tuple["AttentionBackendEnum", int]],
        dict["AttentionBackendEnum", tuple[int, list[str]]],
    ]:
        """
        根据设备能力和 attention 配置，找出当前可用的 backend，
        并返回：
        1. 可用 backend 及其优先级
        2. 不可用 backend 及其失败原因
        """
        valid_backends_priorities = []
        invalid_reasons: dict[AttentionBackendEnum, tuple[int, list[str]]] = {}

        backend_priorities = _get_backend_priorities(
            attn_selector_config.use_mla,
            device_capability,
            num_heads,
        )

        for priority, backend in enumerate(backend_priorities):
            try:
                backend_class = backend.get_class()
                invalid_reasons_i = backend_class.validate_configuration(
                    device_capability=device_capability,
                    **attn_selector_config._asdict(),
                )
            except ImportError:
                invalid_reasons_i = ["ImportError"]

            if invalid_reasons_i:
                invalid_reasons[backend] = (priority, invalid_reasons_i)
            else:
                valid_backends_priorities.append((backend, priority))

        return valid_backends_priorities, invalid_reasons

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: "AttentionBackendEnum | None",
        attn_selector_config: "AttentionSelectorConfig",
        num_heads: int | None = None,
    ) -> str:
        """
        选择一个 attention backend，并返回其类路径字符串。
        """
        device_capability = cls.get_device_capability()
        assert device_capability is not None

        # 如果用户显式指定了 backend，先验证它是否可用
        if selected_backend is not None:
            try:
                backend_class = selected_backend.get_class()
                invalid_reasons = backend_class.validate_configuration(
                    device_capability=device_capability,
                    **attn_selector_config._asdict(),
                )
            except ImportError:
                invalid_reasons = ["ImportError"]

            if invalid_reasons:
                raise ValueError(
                    f"Selected backend {selected_backend} is not valid for "
                    f"this configuration. Reason: {invalid_reasons}"
                )
            else:
                logger.info("Using %s backend.", selected_backend)
                return selected_backend.get_path()

        # 否则自动选择一个有效 backend
        valid_backends_priorities, all_invalid_reasons = cls.get_valid_backends(
            device_capability=device_capability,
            attn_selector_config=attn_selector_config,
            num_heads=num_heads,
        )

        reasons_str = (
            "{"
            + ", ".join(
                f"{backend.name}: [{', '.join(reasons)}]"
                for backend, (_, reasons) in all_invalid_reasons.items()
            )
            + "}"
        )
        config_str = attn_selector_config.__repr__()

        logger.debug_once(
            f"Some attention backends are not valid for {cls.device_name} with "
            f"{config_str}. Reasons: {reasons_str}."
        )

        if len(valid_backends_priorities) == 0:
            raise ValueError(
                f"No valid attention backend found for {cls.device_name} "
                f"with {config_str}. Reasons: {reasons_str}."
            )

        # 选择优先级最高（数字最小）的 backend
        sorted_indices = sorted(
            range(len(valid_backends_priorities)),
            key=lambda i: valid_backends_priorities[i][1],
        )
        selected_index = sorted_indices[0]
        selected_backend = valid_backends_priorities[selected_index][0]
        selected_priority = valid_backends_priorities[selected_index][1]

        # 如果用户指定了 block_size，但没显式指定 attention backend，
        # 这里额外提示：block_size 可能排除了更高优先级 backend
        if attn_selector_config.block_size is not None:
            excluded = [
                backend
                for backend, (priority, reasons) in all_invalid_reasons.items()
                if priority < selected_priority
                and reasons == ["block_size not supported"]
            ]
            if excluded:
                names = ", ".join(b.name for b in excluded)
                logger.warning(
                    "--block-size %d precluded higher-priority backend(s) "
                    "%s. Using %s instead, which may result in reduced "
                    "performance. Consider removing --block-size to "
                    "auto-select the optimal block size.",
                    attn_selector_config.block_size,
                    names,
                    selected_backend.name,
                )

        logger.info_once(
            "Using %s attention backend out of potential backends: %s.",
            selected_backend.name,
            "[" + ", ".join(f"'{b[0].name}'" for b in valid_backends_priorities) + "]",
            scope="local",
        )

        return selected_backend.get_path()

    @classmethod
    def get_supported_vit_attn_backends(cls) -> list["AttentionBackendEnum"]:
        """
        返回当前 CUDA 平台支持的 ViT attention backend 列表。
        """
        if cls.has_device_capability(80):
            return [
                AttentionBackendEnum.FLASH_ATTN,
                AttentionBackendEnum.TRITON_ATTN,
                AttentionBackendEnum.TORCH_SDPA,
                AttentionBackendEnum.FLASHINFER,
            ]
        else:
            return [
                AttentionBackendEnum.FLASH_ATTN,
                AttentionBackendEnum.TORCH_SDPA,
                AttentionBackendEnum.TRITON_ATTN,
                AttentionBackendEnum.FLASHINFER,
            ]

    @classmethod
    def get_vit_attn_backend(
        cls,
        head_size: int,
        dtype: torch.dtype,
        backend: "AttentionBackendEnum | None" = None,
    ) -> "AttentionBackendEnum":
        """
        为 Vision Transformer 选择 attention backend。
        """
        # 如果用户显式指定了 backend，则先校验支持性
        if backend is not None:
            assert backend in cls.get_supported_vit_attn_backends(), (
                f"Backend {backend} is not supported for vit attention. "
                f"Supported backends are: {cls.get_supported_vit_attn_backends()}"
            )
            logger.info_once(f"Using backend {backend} for vit attention")
            return backend

        # 否则按支持情况依次尝试
        cc = cls.get_device_capability()
        for vit_attn_backend in cls.get_supported_vit_attn_backends():
            # TORCH_SDPA 视为兜底后端，直接可返回
            if vit_attn_backend == AttentionBackendEnum.TORCH_SDPA:
                return vit_attn_backend
            try:
                backend_class = vit_attn_backend.get_class()
                is_backend_supported = backend_class.supports_head_size(
                    head_size
                ) and backend_class.supports_dtype(dtype)

                if cc is not None:
                    is_backend_supported = (
                        is_backend_supported
                        and backend_class.supports_compute_capability(cc)
                    )

                if is_backend_supported:
                    logger.info_once(
                        f"Using backend {vit_attn_backend} for vit attention"
                    )
                    return vit_attn_backend
            except ImportError:
                # 某个 backend 没装就跳过
                pass

        # 最后兜底用 Torch SDPA
        return AttentionBackendEnum.TORCH_SDPA

    @classmethod
    def get_punica_wrapper(cls) -> str:
        # 返回 GPU LoRA wrapper 路径
        return "cfie.lora.punica_wrapper.punica_gpu.PunicaWrapperGPU"

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        # 返回 CUDA 通信器类路径
        return (
            "cfie.distributed.device_communicators.cuda_communicator.CudaCommunicator"
        )

    @classmethod
    def supports_fp8(cls) -> bool:
        # 计算能力 >= 8.9 认为支持 FP8
        return cls.has_device_capability(89)

    @classmethod
    def use_custom_allreduce(cls) -> bool:
        # CUDA 平台启用自定义 allreduce
        return True

    @classmethod
    def opaque_attention_op(cls) -> bool:
        # attention op 为 opaque custom op
        return True

    @classmethod
    def get_static_graph_wrapper_cls(cls) -> str:
        # 返回 CUDA graph wrapper 类路径
        return "cfie.compilation.cuda_graph.CUDAGraphWrapper"

    @classmethod
    def stateless_init_device_torch_dist_pg(
        cls,
        backend: str,
        prefix_store: PrefixStore,
        group_rank: int,
        group_size: int,
        timeout: timedelta,
    ) -> ProcessGroup:
        """
        在不依赖已有 torch.distributed 全局状态的前提下，
        初始化一个 NCCL ProcessGroup。
        """
        # -------------------- 先校验当前环境确实具备 NCCL 能力 --------------------
        assert is_nccl_available()

        # -------------------- 创建一个通用 ProcessGroup 外壳 --------------------
        # 这里只是先构建 ProcessGroup 容器本身，真正的 NCCL backend 稍后再挂进去。
        pg: ProcessGroup = ProcessGroup(
            # PrefixStore 负责当前组的 rendezvous key 命名空间。
            prefix_store,
            # 当前 rank 在这个 stateless 组里的组内 rank。
            group_rank,
            # 当前 stateless 组的 world size。
            group_size,
        )

        # 延迟导入 NCCL backend 实现类。
        from torch.distributed.distributed_c10d import ProcessGroupNCCL

        # -------------------- 构造 NCCL backend 所需选项 --------------------
        backend_options = ProcessGroupNCCL.Options()
        # 把调用方传入的 timeout 写到 NCCL backend 选项里。
        backend_options._timeout = timeout

        # -------------------- 创建真正的 NCCL backend 实例 --------------------
        backend_class = ProcessGroupNCCL(
            # NCCL backend 复用这份 PrefixStore 做 rendezvous。
            prefix_store, group_rank, group_size, backend_options
        )
        # 标记这个 backend 的类型是 NCCL。
        backend_type = ProcessGroup.BackendType.NCCL
        # 设备 backend 对应的设备类型固定是 cuda。
        device = torch.device("cuda")

        # -------------------- 把 NCCL backend 注册到 ProcessGroup 外壳 --------------------
        # 将 NCCL 设为这个 ProcessGroup 的默认 backend。
        pg._set_default_backend(backend_type)
        # 初始化该 backend 的 sequence number，确保 collective 调用顺序一致。
        backend_class._set_sequence_number_for_group()
        # 把 backend 挂到这个 ProcessGroup 上，并声明其对应设备类型。
        pg._register_backend(device, backend_type, backend_class)
        # 返回已经完成组装的 stateless NCCL ProcessGroup。
        return pg

    @classmethod
    def device_count(cls) -> int:
        # 返回当前可见 CUDA 设备数（无状态查询）
        return cuda_device_count_stateless()

    @classmethod
    def check_if_supports_dtype(cls, dtype: torch.dtype):
        """
        检查当前 GPU 是否支持指定 dtype。
        """
        if dtype == torch.bfloat16:
            if not cls.has_device_capability(80):
                capability = cls.get_device_capability()
                gpu_name = cls.get_device_name()

                if capability is None:
                    compute_str = "does not have a compute capability"
                else:
                    version_str = capability.as_version_str()
                    compute_str = f"has compute capability {version_str}"

                raise ValueError(
                    "Bfloat16 is only supported on GPUs "
                    "with compute capability of at least 8.0. "
                    f"Your {gpu_name} GPU {compute_str}. "
                    "You can use float16 instead by explicitly setting the "
                    "`dtype` flag in CLI, for example: --dtype=half."
                )

    @classmethod
    def insert_blocks_to_device(
        cls,
        src_cache: torch.Tensor,
        dst_cache: torch.Tensor,
        src_block_indices: torch.Tensor,
        dst_block_indices: torch.Tensor,
    ) -> None:
        """
        把 src_cache 指定块复制到 GPU 上的 dst_cache 指定块。
        """
        _src_cache = src_cache[:, src_block_indices]
        dst_cache[:, dst_block_indices] = _src_cache.to(dst_cache.device)

    @classmethod
    def swap_out_blocks_to_host(
        cls,
        src_cache: torch.Tensor,
        dst_cache: torch.Tensor,
        src_block_indices: torch.Tensor,
        dst_block_indices: torch.Tensor,
    ) -> None:
        """
        把 GPU 上的 src_cache 指定块复制到 CPU 上的 dst_cache。
        """
        _src_cache = src_cache[:, src_block_indices]
        dst_cache[:, dst_block_indices] = _src_cache.cpu()

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        # 支持 hybrid KV cache
        return True

    @classmethod
    def support_static_graph_mode(cls) -> bool:
        # 支持静态图模式
        return True

    @classmethod
    def num_compute_units(cls, device_id: int = 0) -> int:
        # 返回 GPU 的 SM 数量（multi_processor_count）
        return torch.cuda.get_device_properties(device_id).multi_processor_count

    @classmethod
    def use_custom_op_collectives(cls) -> bool:
        # 使用自定义 collective 实现
        return True


# ------------------------------- 基于 NVML 的 CUDA 设备查询平台实现 -------------------------------
# 该实现通过 NVML 查询 GPU 信息，而不是依赖 CUDA runtime 初始化后的上下文状态。
class NvmlCudaPlatform(CudaPlatformBase):
    @classmethod
    @cache
    @with_nvml_context
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability | None:
        """
        使用 NVML 查询指定设备的 compute capability。
        """
        # ------------------------------- 将逻辑设备编号映射为物理设备编号并查询 capability -------------------------------
        try:
            # 将当前逻辑设备编号转换为真实物理设备编号。
            physical_device_id = cls.device_id_to_physical_device_id(device_id)

            # 基于物理设备编号获取 NVML 设备句柄。
            handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)

            # 读取设备的 CUDA compute capability 主版本号与次版本号。
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)

            # 返回包装后的设备 capability 对象。
            return DeviceCapability(major=major, minor=minor)
        except RuntimeError:
            # 当 NVML 查询失败时，返回 None 表示 capability 不可用。
            return None

    @classmethod
    @with_nvml_context
    def has_device_capability(
        cls,
        capability: tuple[int, int] | int,
        device_id: int = 0,
    ) -> bool:
        """
        判断指定设备是否满足给定的 compute capability 要求。
        """
        # ------------------------------- 调用父类逻辑判断设备 capability 是否满足要求 -------------------------------
        try:
            # 复用父类的 capability 判断逻辑。
            return super().has_device_capability(capability, device_id)
        except RuntimeError:
            # 当查询过程中发生异常时，按不满足 capability 处理。
            return False

    @classmethod
    @with_nvml_context
    def get_device_name(cls, device_id: int = 0) -> str:
        """
        使用 NVML 查询指定设备的名称。
        """
        # ------------------------------- 将逻辑设备编号映射为物理设备编号并查询设备名称 -------------------------------
        # 将当前逻辑设备编号转换为真实物理设备编号。
        physical_device_id = cls.device_id_to_physical_device_id(device_id)

        # 基于物理设备编号返回设备名称。
        return cls._get_physical_device_name(physical_device_id)

    @classmethod
    @with_nvml_context
    def get_device_uuid(cls, device_id: int = 0) -> str:
        """
        使用 NVML 查询指定设备的 UUID。
        """
        # ------------------------------- 将逻辑设备编号映射为物理设备编号并查询设备 UUID -------------------------------
        # 将当前逻辑设备编号转换为真实物理设备编号。
        physical_device_id = cls.device_id_to_physical_device_id(device_id)

        # 基于物理设备编号获取 NVML 设备句柄。
        handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)

        # 返回该设备对应的 UUID 字符串。
        return pynvml.nvmlDeviceGetUUID(handle)

    @classmethod
    @with_nvml_context
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        """
        使用 NVML 查询指定设备的总显存大小。
        """
        # ------------------------------- 将逻辑设备编号映射为物理设备编号并查询总显存 -------------------------------
        # 将当前逻辑设备编号转换为真实物理设备编号。
        physical_device_id = cls.device_id_to_physical_device_id(device_id)

        # 基于物理设备编号获取 NVML 设备句柄。
        handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)

        # 读取并返回设备总显存字节数。
        return int(pynvml.nvmlDeviceGetMemoryInfo(handle).total)

    @classmethod
    @with_nvml_context
    def is_fully_connected(cls, physical_device_ids: list[int]) -> bool:
        """
        判断一组物理 GPU 是否通过 NVLink 形成完全互联。
        """
        # ------------------------------- 为待检查的物理设备列表构造 NVML 句柄 -------------------------------
        # 为每个物理设备编号获取对应的 NVML 设备句柄。
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in physical_device_ids]

        # ------------------------------- 逐对检查 GPU 之间的 NVLink 连通性 -------------------------------
        # 枚举设备句柄列表中的每一对 GPU 组合。
        for i, handle in enumerate(handles):
            for j, peer_handle in enumerate(handles):
                # 仅检查上三角组合，避免重复检查同一对设备。
                if i < j:
                    try:
                        # 读取当前两张 GPU 在 NVLINK 能力维度上的 P2P 状态。
                        p2p_status = pynvml.nvmlDeviceGetP2PStatus(
                            handle,
                            peer_handle,
                            pynvml.NVML_P2P_CAPS_INDEX_NVLINK,
                        )

                        # 当任意一对 GPU 之间的 NVLink 状态不正常时，说明整体不完全互联。
                        if p2p_status != pynvml.NVML_P2P_STATUS_OK:
                            return False
                    except pynvml.NVMLError:
                        # 当查询 NVLink 状态失败时，记录日志并按“不完全互联”处理。
                        logger.exception(
                            "NVLink detection failed. This is normal if"
                            " your machine has no NVLink equipped."
                        )
                        return False

        # 全部 GPU 两两之间都满足 NVLink 单跳可达时，返回完全互联。
        return True

    @classmethod
    def _get_physical_device_name(cls, device_id: int = 0) -> str:
        """
        使用物理设备编号查询设备名称。
        """
        # ------------------------------- 基于物理设备编号直接查询设备名称 -------------------------------
        # 根据物理设备编号获取 NVML 设备句柄。
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

        # 返回该物理设备的名称。
        return pynvml.nvmlDeviceGetName(handle)

    @classmethod
    @with_nvml_context
    def log_warnings(cls):
        """
        检查系统中的 GPU 是否存在混卡情况，并在需要时打印告警。
        """
        # ------------------------------- 读取系统中可见物理 GPU 数量 -------------------------------
        # 通过 NVML 读取当前系统中的物理 GPU 数量。
        device_ids: int = pynvml.nvmlDeviceGetCount()

        # ------------------------------- 在多 GPU 场景下检查设备名称是否混杂 -------------------------------
        # 仅当系统中 GPU 数量大于 1 时，才有必要检查是否存在混卡情况。
        if device_ids > 1:
            # 读取所有物理 GPU 的设备名称列表。
            device_names = [cls._get_physical_device_name(i) for i in range(device_ids)]

            # 当系统中存在不同型号 GPU，且 CUDA_DEVICE_ORDER 不是 PCI_BUS_ID 时，打印告警。
            if (
                len(set(device_names)) > 1
                and os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
            ):
                logger.warning(
                    "Detected different devices in the system: %s. Please"
                    " make sure to set `CUDA_DEVICE_ORDER=PCI_BUS_ID` to "
                    "avoid unexpected behavior.",
                    ", ".join(device_names),
                )

class NonNvmlCudaPlatform(CudaPlatformBase):
    """
    不依赖 NVML 的 CUDA 平台实现。
    直接使用 torch.cuda 查询设备信息。
    """

    @classmethod
    @cache
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        major, minor = torch.cuda.get_device_capability(device_id)
        return DeviceCapability(major=major, minor=minor)

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return torch.cuda.get_device_name(device_id)

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        device_props = torch.cuda.get_device_properties(device_id)
        return device_props.total_memory

    @classmethod
    def is_fully_connected(cls, physical_device_ids: list[int]) -> bool:
        logger.exception(
            "NVLink detection not possible, as context support was"
            " not found. Assuming no NVLink available."
        )
        return False


# 自动检测当前环境是否可用 NVML
# 如果可用，则使用 NvmlCudaPlatform
# 否则退回到 NonNvmlCudaPlatform
nvml_available = False
try:
    try:
        pynvml.nvmlInit()
        nvml_available = True
    except Exception:
        # Jetson 上通常不支持 NVML
        nvml_available = False
finally:
    if nvml_available:
        pynvml.nvmlShutdown()

CudaPlatform = NvmlCudaPlatform if nvml_available else NonNvmlCudaPlatform

# 初始化时打印必要告警
CudaPlatform.log_warnings()
