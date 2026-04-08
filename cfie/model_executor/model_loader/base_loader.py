# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

import cfie.envs as envs
from cfie.config import ModelConfig, CfieConfig
from cfie.config.load import LoadConfig
from cfie.logger import init_logger
from cfie.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from cfie.offload.weight_offload import maybe_enable_tiered_moe_cache
from cfie.platforms import current_platform
from cfie.tracing import instrument
from cfie.utils.mem_utils import format_gib
from cfie.utils.torch_utils import set_default_torch_dtype

logger = init_logger(__name__)


# 抽象出所有模型加载器共享的下载、建模、灌权重和后处理流程。
class BaseModelLoader(ABC):
    """Base class for model loaders."""

    # 保存当前加载器要遵循的加载配置。
    def __init__(self, load_config: LoadConfig):
        self.load_config = load_config

    @abstractmethod
    # 预下载模型文件，使后续真正加载时不再发生远端拉取。
    def download_model(self, model_config: ModelConfig) -> None:
        """Download a model so that it can be immediately loaded."""
        raise NotImplementedError

    @abstractmethod
    # 把 checkpoint 权重灌入已经创建好的模型结构。
    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        """Load weights into a model. This standalone API allows
        inplace weights loading for an already-initialized model"""
        raise NotImplementedError

    @instrument(span_name="Load model")
    # 完成“建图 -> 灌权重 -> 后处理 -> 启用 MoE offload”这一整段模型加载流程。
    def load_model(
        self, cfie_config: CfieConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module:
        # 先取出设备配置；这里描述模型最终在哪个设备上执行。
        device_config = cfie_config.device_config

        # 再取出加载配置；这里描述权重如何加载、是否覆盖设备等。
        load_config = cfie_config.load_config

        # 若 load_config.device 被显式设置，则优先使用；
        # 否则退回到 device_config.device。
        load_device = (
            device_config.device if load_config.device is None else load_config.device
        )

        # 统一转成 torch.device，后面上下文切换都基于它。
        target_device = torch.device(load_device)

        # 在整个模型创建与加载期间，临时把默认 dtype 设成 model_config.dtype，
        # 保证新建参数/缓冲区时类型正确。
        with set_default_torch_dtype(model_config.dtype):
            # 进入目标设备上下文。当前 Qwen3.5 启动路径通常是 CUDA。
            with target_device:
                # 第一步：只创建模型结构，不加载 checkpoint 权重。
                # 这里会继续进入 initialize_model()。
                model = initialize_model(
                    cfie_config=cfie_config,
                    model_config=model_config,
                    prefix=prefix
                )

            # 如果打开了结构检查日志，就把刚创建出的模块树打印出来。
            log_model_inspection(model)

            # 输出一条调试日志，表示接下来开始真正加载权重。
            logger.debug("Loading weights on %s ...", load_device)

            # 第二步：把 checkpoint 权重灌进模型参数。
            # 注意：量化后的重排/pack 不在这里做，而是在后面的
            # process_weights_after_loading() 做。
            self.load_weights(model, model_config)

            # Log peak GPU memory after loading weights. This is needed
            # to have test coverage on peak memory for online quantization.
            if current_platform.is_cuda():
                # 记录“权重刚加载完”这一时刻的峰值显存，主要用于调试和测试。
                peak_memory = torch.cuda.max_memory_allocated()
                logger.debug_once(
                    "Peak GPU memory after loading weights: %s GiB",
                    format_gib(peak_memory),
                    scope="local",
                )

            # 第三步：统一做权重加载后的后处理。
            # 包括量化模块的后处理、注意力模块的后处理等。
            process_weights_after_loading(model, model_config, target_device)

            # 第四步：如果是 MoE 模型，则根据 cfie_config 决定是否启用
            # CFIE 的分层专家缓存/offload 逻辑。
            maybe_enable_tiered_moe_cache(model, cfie_config)

        # 最后切到 eval 模式并返回，确保推理时关闭训练态行为。
        return model.eval()


# 按环境变量决定是否输出模型结构日志，便于审查模型装配结果。
def log_model_inspection(model: nn.Module) -> None:
    """Log model structure if VLLM_LOG_MODEL_INSPECTION=1."""
    # 默认不开；只有显式设置环境变量时才打印模型结构。
    if not envs.VLLM_LOG_MODEL_INSPECTION:
        return

    from cfie.model_inspection import format_model_inspection

    # 将模型结构格式化输出到日志，便于排查模型装配是否符合预期。
    logger.info("vLLM model structure:\n%s", format_model_inspection(model))
