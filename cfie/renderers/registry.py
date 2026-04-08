# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from cfie.logger import init_logger
from cfie.tokenizers.registry import tokenizer_args_from_config
from cfie.utils.import_utils import resolve_obj_by_qualname

from .base import BaseRenderer

if TYPE_CHECKING:
    from cfie.config import CfieConfig

logger = init_logger(__name__)

_VLLM_RENDERERS = {
    "deepseek_v32": ("deepseek_v32", "DeepseekV32Renderer"),
    "hf": ("hf", "HfRenderer"),
    "grok2": ("grok2", "Grok2Renderer"),
    "kimi_audio": ("kimi_audio", "KimiAudioRenderer"),
    "mistral": ("mistral", "MistralRenderer"),
    "qwen_vl": ("qwen_vl", "QwenVLRenderer"),
    "terratorch": ("terratorch", "TerratorchRenderer"),
}


@dataclass
class RendererRegistry:
    # renderer_mode -> (renderer 模块路径, renderer 类名)
    renderers: dict[str, tuple[str, str]] = field(default_factory=dict)

    def register(self, renderer_mode: str, module: str, class_name: str) -> None:
        # 若同一个 renderer_mode 已注册过，则打印覆盖告警。
        if renderer_mode in self.renderers:
            logger.warning(
                "%s.%s is already registered for renderer_mode=%r. "
                "It is overwritten by the new one.",
                module,
                class_name,
                renderer_mode,
            )

        # 把 renderer_mode 映射到具体模块和类名。
        self.renderers[renderer_mode] = (module, class_name)

        return None

    def load_renderer_cls(self, renderer_mode: str) -> type[BaseRenderer]:
        # 若该 mode 没有注册 renderer，直接报错。
        if renderer_mode not in self.renderers:
            raise ValueError(f"No renderer registered for {renderer_mode=!r}.")

        # 取出该 renderer_mode 对应的模块与类名。
        module, class_name = self.renderers[renderer_mode]
        # 打印一次调试日志，说明当前选择了哪个 renderer 类。
        logger.debug_once(f"Loading {class_name} for {renderer_mode=!r}")

        # 按全限定类名动态解析 renderer 类。
        return resolve_obj_by_qualname(f"{module}.{class_name}")

    def load_renderer(
            self,
            renderer_mode: str,
            config: "CfieConfig",
            tokenizer_kwargs: dict[str, Any],
    ) -> BaseRenderer:
        # 先解析出 renderer 类。
        renderer_cls = self.load_renderer_cls(renderer_mode)
        # 再调用 renderer 的 from_config 构造实例。
        return renderer_cls.from_config(config, tokenizer_kwargs)  # cfie.renderers.hf.HfRenderer


RENDERER_REGISTRY = RendererRegistry(
    {
        mode: (f"cfie.renderers.{mod_relname}", cls_name)
        for mode, (mod_relname, cls_name) in _VLLM_RENDERERS.items()
    }
)
"""The global `RendererRegistry` instance."""


def renderer_from_config(config: "CfieConfig", **kwargs):
    # 先取出模型配置，后续根据 tokenizer_mode / model_impl 选择 renderer。
    model_config = config.model_config

    # 从模型配置中推导 tokenizer_mode、tokenizer_name 以及 tokenizer 构造参数。
    tokenizer_mode, tokenizer_name, args, kwargs = tokenizer_args_from_config(
        model_config, **kwargs
    )

    # 而是强制切到 terratorch renderer。
    if (
            model_config.tokenizer_mode == "auto"
            and model_config.model_impl == "terratorch"
    ):
        # Terratorch 专用 renderer 分支。
        renderer_mode = "terratorch"
    else:
        # 普通 renderer 分支，直接复用 tokenizer_mode 作为 renderer_mode。
        renderer_mode = tokenizer_mode

    # 按当前分支得到的 renderer_mode 加载对应 renderer，并继续透传 tokenizer 参数。
    return RENDERER_REGISTRY.load_renderer(
        renderer_mode,
        config,
        tokenizer_kwargs={**kwargs, "tokenizer_name": tokenizer_name},
    )
