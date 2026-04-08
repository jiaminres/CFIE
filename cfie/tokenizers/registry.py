# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import huggingface_hub
from typing_extensions import TypeVar, assert_never

import cfie.envs as envs
from cfie.logger import init_logger
from cfie.transformers_utils.gguf_utils import (
    check_gguf_file,
    get_gguf_file_path_from_hf,
    is_gguf,
    is_remote_gguf,
    split_remote_gguf,
)
from cfie.transformers_utils.repo_utils import (
    any_pattern_in_repo_files,
    is_mistral_model_repo,
)
from cfie.utils.import_utils import resolve_obj_by_qualname

from .protocol import TokenizerLike

if TYPE_CHECKING:
    from cfie.config.model import ModelConfig, RunnerType

logger = init_logger(__name__)


_VLLM_TOKENIZERS = {
    "deepseek_v32": ("deepseek_v32", "DeepseekV32Tokenizer"),
    "grok2": ("grok2", "Grok2Tokenizer"),
    "hf": ("hf", "CachedHfTokenizer"),
    "kimi_audio": ("kimi_audio", "KimiAudioTokenizer"),
    "mistral": ("mistral", "MistralTokenizer"),
    "qwen_vl": ("qwen_vl", "QwenVLTokenizer"),
}


@dataclass
class _TokenizerRegistry:
    # tokenizer_mode -> (tokenizer 模块路径, tokenizer 类名)
    tokenizers: dict[str, tuple[str, str]] = field(default_factory=dict)

    def register(self, tokenizer_mode: str, module: str, class_name: str) -> None:
        # 同一个 tokenizer_mode 被重复注册时，打印覆盖告警。
        if tokenizer_mode in self.tokenizers:
            logger.warning(
                "%s.%s is already registered for tokenizer_mode=%r. "
                "It is overwritten by the new one.",
                module,
                class_name,
                tokenizer_mode,
            )

        # 记录 tokenizer_mode 到具体模块和类名的映射。
        self.tokenizers[tokenizer_mode] = (module, class_name)

        return None

    def load_tokenizer_cls(self, tokenizer_mode: str) -> type[TokenizerLike]:
        # 未注册的 tokenizer_mode 直接报错。
        if tokenizer_mode not in self.tokenizers:
            raise ValueError(f"No tokenizer registered for {tokenizer_mode=!r}.")

        # 取出该 mode 对应的 tokenizer 模块与类名。
        module, class_name = self.tokenizers[tokenizer_mode]
        # 打印一次调试日志，说明当前选择了哪个 tokenizer 类。
        logger.debug_once(f"Loading {class_name} for {tokenizer_mode=!r}")

        # 动态解析 tokenizer 类。
        return resolve_obj_by_qualname(f"{module}.{class_name}")

    def load_tokenizer(self, tokenizer_mode: str, *args, **kwargs) -> TokenizerLike:
        # 先解析具体 tokenizer 类。
        tokenizer_cls = self.load_tokenizer_cls(tokenizer_mode)
        # 再调用 from_pretrained 构造 tokenizer。
        return tokenizer_cls.from_pretrained(*args, **kwargs)


TokenizerRegistry = _TokenizerRegistry(
    {
        mode: (f"cfie.tokenizers.{mod_relname}", cls_name)
        for mode, (mod_relname, cls_name) in _VLLM_TOKENIZERS.items()
    }
)


def resolve_tokenizer_args(
    # tokenizer 名称、本地目录或 GGUF 文件路径。
    tokenizer_name: str | Path,
    *args,
    # 当前模型运行类型，决定默认 truncation_side。
    runner_type: "RunnerType" = "generate",
    # tokenizer 模式，可为 auto / hf / mistral / slow 等。
    tokenizer_mode: str = "auto",
    **kwargs,
):
    # 取出常用构造参数，后续要参与 tokenizer 模式判断。
    revision: str | None = kwargs.get("revision")
    download_dir: str | None = kwargs.get("download_dir")

    # 当前执行路径：当前是常规 Hugging Face / 本地目录启动，不走 ModelScope 分支。
    if envs.VLLM_USE_MODELSCOPE:
        # 延迟导入，避免普通 Hugging Face 路径强依赖 modelscope。
        from modelscope.hub.snapshot_download import snapshot_download

        # 延迟导入锁工具，避免循环依赖。
        from cfie.model_executor.model_loader.weight_utils import get_lock

        # 这里只下载 tokenizer 资源，模型权重仍由后续 worker 负责下载。
        if not Path(tokenizer_name).exists():
            # 用文件锁避免多个进程并发下载同一个 tokenizer 目录。
            with get_lock(tokenizer_name, download_dir):
                tokenizer_path = snapshot_download(
                    model_id=str(tokenizer_name),
                    cache_dir=download_dir,
                    revision=revision,
                    local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                    # Ignore weights - we only need the tokenizer.
                    ignore_file_pattern=[".*.pt", ".*.safetensors", ".*.bin"],
                )
                tokenizer_name = tokenizer_path

    # 当前执行路径：当前 tokenizer 指向普通 HF 模型目录，不是 GGUF，因此跳过本分支。
    if is_gguf(tokenizer_name):
        # 本地 GGUF 文件直接拆成父目录 + 文件名。
        if check_gguf_file(tokenizer_name):
            kwargs["gguf_file"] = Path(tokenizer_name).name
            tokenizer_name = Path(tokenizer_name).parent
        # 远端 GGUF 则把 repo_id 与 quant_type 拆开，再反查真正文件路径。
        elif is_remote_gguf(tokenizer_name):
            tokenizer_name, quant_type = split_remote_gguf(tokenizer_name)
            # 从 Hugging Face Hub 侧解析 GGUF 文件路径。
            gguf_file = get_gguf_file_path_from_hf(
                tokenizer_name,
                quant_type,
                revision=revision,
            )
            kwargs["gguf_file"] = gguf_file

    # 当前执行路径：上层未显式传 truncation_side，因此进入默认补值分支。
    if "truncation_side" not in kwargs:
        # 当前执行路径：chat/generate 请求命中这里，默认使用左截断。
        if runner_type == "generate" or runner_type == "draft":
            kwargs["truncation_side"] = "left"
        # pooling 任务默认右截断，优先保留开头部分。
        elif runner_type == "pooling":
            kwargs["truncation_side"] = "right"
        else:
            assert_never(runner_type)

    # 当前执行路径：当前 tokenizer_mode 不是 slow，因此不会进入这个兼容分支。
    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError("Cannot use the fast tokenizer in slow tokenizer mode.")

        # slow 本质上还是走 hf tokenizer，只是强制 use_fast=False。
        tokenizer_mode = "hf"
        kwargs["use_fast"] = False

    # 当前执行路径：当前仓库不是 Mistral tokenizer 形态，因此不会切到 mistral。
    if (
        tokenizer_mode == "auto"
        and is_mistral_model_repo(
            model_name_or_path=str(tokenizer_name), revision=revision
        )
        and any_pattern_in_repo_files(
            model_name_or_path=str(tokenizer_name),
            allow_patterns=["tekken.json", "tokenizer.model.v*"],
            revision=revision,
        )
    ):
        tokenizer_mode = "mistral"

    # 当前执行路径：前面仍保持 tokenizer_mode=auto，因此最终回退到 hf tokenizer。
    if tokenizer_mode == "auto":
        tokenizer_mode = "hf"

    # 返回最终 tokenizer_mode、tokenizer 名称以及透传参数。
    return tokenizer_mode, tokenizer_name, args, kwargs


cached_resolve_tokenizer_args = lru_cache(resolve_tokenizer_args)


def tokenizer_args_from_config(config: "ModelConfig", **kwargs):
    # 从 ModelConfig 中提取 tokenizer 相关字段，并复用缓存后的参数解析逻辑。
    return cached_resolve_tokenizer_args(
        config.tokenizer,
        runner_type=config.runner_type,
        tokenizer_mode=config.tokenizer_mode,
        revision=config.tokenizer_revision,
        trust_remote_code=config.trust_remote_code,
        **kwargs,
    )


_T = TypeVar("_T", bound=TokenizerLike, default=TokenizerLike)


def get_tokenizer(
    # tokenizer 名称、本地路径或 GGUF 输入。
    tokenizer_name: str | Path,
    *args,
    # 可选显式 tokenizer 类；默认按 tokenizer_mode 自动解析。
    tokenizer_cls: type[_T] = TokenizerLike,  # type: ignore[assignment]
    # 是否信任远端仓库中的自定义代码。
    trust_remote_code: bool = False,
    # tokenizer 使用的 revision。
    revision: str | None = None,
    # 下载目录。
    download_dir: str | None = None,
    **kwargs,
) -> _T:
    """Gets a tokenizer for the given model name via HuggingFace or ModelScope."""
    # 先统一解析 tokenizer mode、名称和构造参数。
    tokenizer_mode, tokenizer_name, args, kwargs = cached_resolve_tokenizer_args(
        tokenizer_name,
        *args,
        trust_remote_code=trust_remote_code,
        revision=revision,
        download_dir=download_dir,
        **kwargs,
    )

    # 若未显式指定 tokenizer 类，则按 tokenizer_mode 从注册表里加载。
    if tokenizer_cls == TokenizerLike:
        tokenizer_cls_ = TokenizerRegistry.load_tokenizer_cls(tokenizer_mode)
    else: # 默认
        # 否则直接使用调用方给定的 tokenizer 类。
        tokenizer_cls_ = tokenizer_cls

    # 调用最终 tokenizer 类的 from_pretrained 构造 tokenizer 实例。
    tokenizer = tokenizer_cls_.from_pretrained(tokenizer_name, *args, **kwargs)

    # slow tokenizer 会导致预处理变慢，因此打印告警。
    if not tokenizer.is_fast:
        logger.warning(
            "Using a slow tokenizer. This might cause a significant "
            "slowdown. Consider using a fast tokenizer instead."
        )

    return tokenizer


cached_get_tokenizer = lru_cache(get_tokenizer)


def cached_tokenizer_from_config(model_config: "ModelConfig", **kwargs):
    if model_config.skip_tokenizer_init:
        return None

    return cached_get_tokenizer(
        model_config.tokenizer,
        runner_type=model_config.runner_type,
        tokenizer_mode=model_config.tokenizer_mode,
        revision=model_config.tokenizer_revision,
        trust_remote_code=model_config.trust_remote_code,
        **kwargs,
    )
