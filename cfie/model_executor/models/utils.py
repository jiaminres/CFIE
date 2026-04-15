# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, overload

import regex as re
import torch
import torch.nn as nn
from torch.nn.modules.module import register_module_module_registration_hook
from transformers import PretrainedConfig

from cfie.config import CfieConfig
from cfie.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from cfie.logger import init_logger
from cfie.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from cfie.model_executor.model_loader.reload import (
    support_quantized_model_reload_from_hp_weights,
)
from cfie.model_executor.model_loader.weight_utils import default_weight_loader
from cfie.model_executor.models.interfaces import supports_any_eagle
from cfie.multimodal import NestedTensors
from cfie.sequence import IntermediateTensors
from cfie.utils.math_utils import cdiv
from cfie.utils.platform_utils import (
    is_pin_memory_available,
)
from cfie.utils.torch_utils import (
    direct_register_custom_op,
)

logger = init_logger(__name__)


@dataclass
class WeightsMapper:
    """Maps the name of each weight if they match the following patterns.

    If a key maps to a value of `None`, the corresponding weight is ignored."""

    orig_to_new_regex: Mapping[re.Pattern, str | None] = field(default_factory=dict)
    orig_to_new_substr: Mapping[str, str | None] = field(default_factory=dict)
    orig_to_new_prefix: Mapping[str, str | None] = field(default_factory=dict)
    orig_to_new_suffix: Mapping[str, str | None] = field(default_factory=dict)

    def __or__(self, other: "WeightsMapper") -> "WeightsMapper":
        """Combine two `WeightsMapper`s by merging their mappings."""
        return WeightsMapper(
            orig_to_new_substr={**self.orig_to_new_substr, **other.orig_to_new_substr},
            orig_to_new_prefix={**self.orig_to_new_prefix, **other.orig_to_new_prefix},
            orig_to_new_suffix={**self.orig_to_new_suffix, **other.orig_to_new_suffix},
        )

    def _map_name(self, key: str) -> str | None:
        for pattern, new_key in self.orig_to_new_regex.items():
            if pattern.search(key):
                if new_key is None:
                    return None

                key = pattern.sub(new_key, key)

        for substr, new_key in self.orig_to_new_substr.items():
            if substr in key:
                if new_key is None:
                    return None

                key = key.replace(substr, new_key, 1)

        for prefix, new_key in self.orig_to_new_prefix.items():
            if key.startswith(prefix):
                if new_key is None:
                    return None

                key = key.replace(prefix, new_key, 1)

        for suffix, new_key in self.orig_to_new_suffix.items():
            if key.endswith(suffix):
                if new_key is None:
                    return None

                key = new_key.join(key.rsplit(suffix, 1))

        return key

    def apply(
            self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        return (
            (out_name, data)
            for name, data in weights
            if (out_name := self._map_name(name)) is not None
        )

    def apply_list(self, values: list[str]) -> list[str]:
        return [
            out_name
            for name in values
            if (out_name := self._map_name(name)) is not None
        ]

    def apply_dict(self, values: dict[str, Any]) -> dict[str, Any]:
        return {
            out_name: value
            for name, value in values.items()
            if (out_name := self._map_name(name)) is not None
        }


class AutoWeightsLoader:
    """
    Helper class to load weights into a [`torch.nn.Module`][]. It is able
    to automatically detect child modules and parameters while iterating over
    the weights only once.

    The weight loading logic for individual modules can be overridden
    by defining a `load_weights` method.

    Similarly, the weight loading logic for individual parameters can be
    overridden by defining a `weight_loader` method.

    Detailed weight loading information can be viewed by setting the
    environment variable `VLLM_LOGGING_LEVEL=DEBUG`.
    """

    # Models trained using early version ColossalAI or quantized by
    # GPTQModel may include these tensors in checkpoint. Skip them.
    ROTARY_EMBEDS_UNUSED_WEIGHTS = [
        "rotary_pos_emb.inv_freq",
        "rotary_emb.inv_freq",
        "rotary_emb.cos_cached",
        "rotary_emb.sin_cached",
    ]

    def __init__(
            self,
            module: nn.Module,
            *,
            skip_prefixes: list[str] | None = None,
            skip_substrs: list[str] | None = None,
            ignore_unexpected_prefixes: list[str] | None = None,
            ignore_unexpected_suffixes: list[str] | None = None,
    ) -> None:
        super().__init__()

        self.module = module
        self.skip_prefixes = skip_prefixes or []
        self.skip_substrs = skip_substrs or []
        self.ignore_unexpected_prefixes = ignore_unexpected_prefixes or []
        self.ignore_unexpected_suffixes = ignore_unexpected_suffixes or []
        # update default skip_substrs
        self.skip_substrs += self.ROTARY_EMBEDS_UNUSED_WEIGHTS

    def _groupby_prefix(
            self,
            weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[tuple[str, Iterable[tuple[str, torch.Tensor]]]]:
        # 把形如：
        # - "model.layers.0.mlp.experts.w13_qweight"
        # - "lm_head.weight"
        # 的 checkpoint 权重名，按“第一个 '.' 前面的前缀”分组。
        #
        # 例如：
        # - "model.layers.0..." -> prefix = "model"
        # - "lm_head.weight"    -> prefix = "lm_head"
        # - "weight"            -> prefix = "weight"，剩余路径为空串
        weights_by_parts = (
            (weight_name.split(".", 1), weight_data)
            for weight_name, weight_data in weights
        )

        for prefix, group in itertools.groupby(weights_by_parts, key=lambda x: x[0][0]):
            yield (
                prefix,
                # 这里把当前层级已经消费掉的第一级前缀剥掉，
                # 交给下一层递归继续处理。
                #
                # 例子：
                # - 原始 "model.layers.0.weight"
                # - 当前返回 ("model", ("layers.0.weight", tensor))
                (
                    ("" if len(parts) == 1 else parts[1], weights_data)
                    for parts, weights_data in group
                ),
            )

    def _get_qualname(self, prefix: str, rest: str) -> str:
        # 把“当前递归基路径”和“当前分组/参数的相对路径”拼成完整名字。
        #
        # 例子：
        # - prefix=""      rest="model"            -> "model"
        # - prefix="model" rest="layers.0.weight"  -> "model.layers.0.weight"
        if prefix == "":
            return rest
        if rest == "":
            return prefix

        return ".".join((prefix, rest))

    def _can_skip(self, qualname: str) -> bool:
        # 判断某个完整权重名/模块名前缀是否应该被跳过。
        #
        # 当前 Qwen3.5 主模型 load_weights 的典型场景：
        # - self.skip_prefixes = ["mtp."]
        #   因此 "mtp.layers.0..." 会被主模型跳过，交给 Qwen3_5MoeMTP。
        # - self.skip_substrs 还会自动附加 rotary_emb.* 这些无需加载的旧权重名。
        return any(qualname.startswith(p) for p in self.skip_prefixes) or any(
            substr in qualname for substr in self.skip_substrs
        )

    def _can_ignore_unexpected(self, qualname: str) -> bool:
        # 判断“checkpoint 里有，但当前模块树里找不到”的名字是否允许静默忽略。
        # 这通常用于兼容不同 checkpoint 版本的额外字段。
        iup = (qualname.startswith(p) for p in self.ignore_unexpected_prefixes)
        ius = (qualname.endswith(s) for s in self.ignore_unexpected_suffixes)
        return any(iup) or any(ius)

    def _load_param(
            self,
            base_prefix: str,
            param: nn.Parameter,
            weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[str]:
        # 这里处理“已经确认命中某个具体参数对象”的情况。
        #
        # 典型输入：
        # - base_prefix = "model.layers.0.mlp.experts.w13_qweight"
        # - weights 里通常只会有一个元素，且 weight_name == ""
        for weight_name, weight_data in weights:
            # 把参数相对名字拼成完整 checkpoint 路径。
            weight_qualname = self._get_qualname(base_prefix, weight_name)

            # 若这个参数名命中 skip 规则，则直接跳过。
            if self._can_skip(weight_qualname):
                logger.debug("Skipping weight %s", weight_qualname)

                continue

            # 走到这里说明当前已经落到“叶子参数”。
            # 因此 weight_name 必须为空串；若不为空，说明 checkpoint 还想继续往下钻，
            # 例如试图把 "xxx.weight.something" 塞进一个单独参数，这属于非法嵌套。
            if weight_name != "":
                if self._can_ignore_unexpected(weight_qualname):
                    logger.debug("Ignoring weight %s", weight_qualname)

                    continue

                raise ValueError(
                    f"Attempted to load nested weight {weight_qualname!r} "
                    f"into a single parameter {base_prefix!r}"
                )

            # 参数对象若自带 weight_loader，就优先用参数自己的加载逻辑；
            # 否则退回默认的 default_weight_loader。
            #
            # Qwen3.5 / GPTQ / FusedMoE 中很多特殊参数都会挂自定义 weight_loader，
            # 用来做 shard 合并、专家映射、量化格式转换等。
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, weight_data)

            logger.debug("Loaded weight %s with shape %s", weight_qualname, param.shape)

            # 返回真正加载成功的完整参数名，供上层收集到 autoloaded_weights 集合里。
            yield weight_qualname

    def _add_loadable_non_param_tensors(
            self, module: nn.Module, child_params: dict[str, torch.Tensor]
    ):
        """
        Add tensor names that are not in the model params that may be in the
        safetensors, e.g., batch normalization stats.
        """
        # 某些模块需要加载的 tensor 不会注册成 Parameter，
        # 但会出现在 state_dict / checkpoint 里。
        # 这里把它们临时并入 child_params，使递归加载逻辑也能命中这些名字。
        if isinstance(
                module,
                (
                        nn.BatchNorm1d,
                        nn.BatchNorm2d,
                        nn.BatchNorm3d,
                        nn.LazyBatchNorm1d,
                        nn.LazyBatchNorm2d,
                        nn.LazyBatchNorm3d,
                        nn.SyncBatchNorm,
                ),
        ):
            module_state_dict = module.state_dict()
            for stat_name in ("running_mean", "running_var", "num_batches_tracked"):
                # 这些 batchnorm 统计量不是 Parameter，但需要和普通参数一样可被匹配到。
                child_params[stat_name] = module_state_dict[stat_name]

    def _load_module(
            self,
            base_prefix: str,
            module: nn.Module,
            weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[str]:
        # 这是 AutoWeightsLoader 的核心递归入口。
        #
        # 进入当前函数时：
        # - base_prefix 表示当前递归已经走到模块树中的哪个路径
        # - module      表示这个路径对应的 nn.Module
        # - weights     表示所有“前缀仍然匹配这个 module 子树”的 checkpoint 权重
        #
        # 当前 Qwen3.5 主模型的典型顶层入口是：
        # - base_prefix = ""
        # - module = Qwen3_5MoeForConditionalGeneration / Qwen3_5ForConditionalGeneration
        # - self.skip_prefixes = ["mtp."]，因此 mtp.* 会在主模型这里被显式跳过
        if isinstance(module, (StageMissingLayer, PPMissingLayer)):
            # 流水线并行/分阶段加载里，某些层在当前 rank 上只是占位模块，
            # 这里不应该继续向下加载任何真实权重。
            return

        if module != self.module:
            # 对“非根模块”优先给模块自身一个接管机会：
            # 如果子模块自己实现了 load_weights(weights)，
            # 就先让它自行消费这批权重。
            #
            # 这正是 Qwen3.5 / FusedMoE / MTP 等复杂模块常用的分支：
            # 它们需要自定义映射规则，不能只靠通用的 named_children/named_parameters 递归。
            module_load_weights = getattr(module, "load_weights", None)
            if callable(module_load_weights):
                loaded_params = module_load_weights(weights)
                if loaded_params is None:
                    # 某些自定义 load_weights 只负责副作用式加载，未返回“实际加载了哪些参数名”。
                    # 这种情况下只能打 warning，没法精确收集 loaded set。
                    logger.warning(
                        "Unable to collect loaded parameters for module %s", module
                    )
                else:
                    # 若子模块返回了它加载成功的“相对参数名”，
                    # 这里把它们补上当前 base_prefix，变成完整名字后向上汇报。
                    yield from map(
                        lambda x: self._get_qualname(base_prefix, x),
                        loaded_params,
                    )

        # 收集“当前模块的直接子模块”：
        # key 是子模块名，value 是对应的 nn.Module。
        child_modules = dict(module.named_children())
        # 收集“当前模块自己直接持有的参数”，不递归进入孙子层。
        # 这样当前层级只负责当前层级的参数匹配，继续下钻交给下面的递归。
        child_params = dict(module.named_parameters(recurse=False))

        # 把 batchnorm 统计量这类“非 Parameter 但可加载”的 tensor 也并入 child_params。
        self._add_loadable_non_param_tensors(module, child_params)

        # 把当前传入的所有权重，按“下一级前缀”分组。
        # 例如当前在 base_prefix="model.layers.0" 时，
        # 可能会得到：
        # - ("self_attn", ...)
        # - ("mlp", ...)
        # - ("input_layernorm", ...)
        for child_prefix, child_weights in self._groupby_prefix(weights):
            # prefix 是当前组在全模型里的完整名字。
            # 例子：base_prefix="model.layers.0" child_prefix="mlp"
            # -> prefix="model.layers.0.mlp"
            prefix = self._get_qualname(base_prefix, child_prefix)

            if child_prefix in child_modules:
                # 命中“直接子模块”：
                # 继续向该子模块递归下钻。
                if self._can_skip(prefix + "."):
                    # 这里要加 "."，因为是在判断“这个模块子树整体”是否该跳过。
                    #
                    # 当前 Qwen3.5 主模型场景下，若 prefix="mtp"，
                    # 那么会命中 self.skip_prefixes=["mtp."]，从这里整棵跳过。
                    logger.debug("Skipping module %s", prefix)

                    continue

                yield from self._load_module(
                    prefix, child_modules[child_prefix], child_weights
                )
            elif child_prefix in child_params:
                # 命中“当前模块的直接参数”：
                # 转交给 _load_param 做最终参数级加载。
                if self._can_skip(prefix):
                    logger.debug("Skipping param %s", prefix)

                    continue

                yield from self._load_param(
                    prefix, child_params[child_prefix], child_weights
                )
            else:
                # 既不是直接子模块，也不是直接参数。
                # 这里开始处理“缺失名字”的各种合法/非法情况。
                can_skip_module = self._can_skip(prefix + ".")
                can_skip_param = self._can_skip(prefix)
                if can_skip_module or can_skip_param:
                    # 虽然当前模块树里找不到这个名字，但它本来就属于 skip 范围，
                    # 所以直接跳过即可。
                    logger.debug("Skipping missing %s", prefix)

                    continue

                can_ignore_module = self._can_ignore_unexpected(prefix + ".")
                can_ignore_param = self._can_ignore_unexpected(prefix)
                if can_ignore_module or can_ignore_param:
                    # checkpoint 里有额外字段，但当前配置允许忽略这些 unexpected 名字。
                    logger.debug("Ignoring missing %s", prefix)

                    continue

                # 走到这里说明：
                # - 这个 checkpoint 名字不在 skip/ignore 范围内
                # - 但当前模块树里确实找不到它
                # 于是抛出详细错误，帮助定位权重名和模块结构不匹配的问题。
                named_parameters = module.named_parameters(recurse=True)
                desc_param_keys = {
                    maybe_prefix(base_prefix, k) for k, _ in named_parameters
                }
                msg = (
                    f"There is no module or parameter named {prefix!r} "
                    f"in {self.module._get_name()}. "
                    f"The available parameters belonging to {base_prefix} "
                    f"({module._get_name()}) are: {desc_param_keys}"
                )
                raise ValueError(msg)

    @support_quantized_model_reload_from_hp_weights
    # AutoWeightsLoader 的顶层权重加载入口。
    # 它通常只在当前持有的模型/模块上调用一次，
    # 然后从 `_load_module("", self.module, weights)` 开始递归下钻。
    def load_weights(
            self,
            weights: Iterable[tuple[str, torch.Tensor]],
            *,
            mapper: WeightsMapper | None = None,
    ) -> set[str]:
        # 顶层入口：
        # 1. 先按 mapper 重写 checkpoint 名字（若传入）
        # 2. 先做一轮 skip 过滤
        # 3. 从根模块 self.module 开始递归 _load_module("", ...)
        #
        # 当前 Qwen3.5 主模型常见场景：
        # - mapper 可能为 None
        # - skip_prefixes 可能是 ["mtp."]
        # - 根路径固定从 base_prefix="" 开始
        if mapper is not None:
            weights = mapper.apply(weights)
        # filter out weights with first-prefix/substr to skip in name
        weights = (
            (name, weight) for name, weight in weights if not self._can_skip(name)
        )

        # 收集所有“被自动加载成功”的完整权重名，供上层后续校验/对账。
        autoloaded_weights = set(self._load_module("", self.module, weights))
        return autoloaded_weights


def init_cfie_registered_model(
        cfie_config: CfieConfig,
        *,
        prefix: str = "",
        hf_config: PretrainedConfig | None = None,
        architectures: list[str] | None = None,
) -> nn.Module:
    """
    Helper function to initialize an inner model registered to vLLM,
    based on the arguments passed to the outer vLLM model.
    """
    from cfie.model_executor.model_loader.utils import initialize_model

    if hf_config is None and architectures is not None:
        # So that the architectures field is overridden
        hf_config = cfie_config.model_config.hf_config

    if hf_config is not None:
        cfie_config = cfie_config.with_hf_config(hf_config, architectures=architectures)

    return initialize_model(cfie_config=cfie_config, prefix=prefix)


@overload
def flatten_bn(x: torch.Tensor) -> torch.Tensor: ...


@overload
def flatten_bn(x: list[torch.Tensor]) -> list[torch.Tensor]: ...


@overload
def flatten_bn(
        x: list[torch.Tensor] | torch.Tensor,
        *,
        concat: Literal[True],
) -> torch.Tensor: ...


@overload
def flatten_bn(
        x: list[torch.Tensor] | torch.Tensor,
        *,
        concat: bool = False,
) -> list[torch.Tensor] | torch.Tensor: ...


def flatten_bn(
        x: list[torch.Tensor] | torch.Tensor,
        *,
        concat: bool = False,
) -> list[torch.Tensor] | torch.Tensor:
    """
    Flatten the `B` and `N` dimensions of batched multimodal inputs.

    The input tensor should have shape `(B, N, ...)`.
    """
    if isinstance(x, torch.Tensor):
        return x.flatten(0, 1)

    if concat:
        return torch.cat(x)

    return [x_n for x_b in x for x_n in x_b]


def _flatten_embeddings(embeddings: NestedTensors) -> torch.Tensor:
    """
    Recursively flattens and concatenates NestedTensors on all but the last
    dimension.
    """

    if isinstance(embeddings, torch.Tensor):
        # Flatten all but the last dimension.
        return embeddings.flatten(0, -2)

    return torch.cat(tuple(_flatten_embeddings(t) for t in embeddings))


def _embedding_count_expression(embeddings: NestedTensors) -> str:
    """
    Constructs a debugging representation of the number of embeddings in the
    NestedTensors.
    """

    if isinstance(embeddings, torch.Tensor):
        return " x ".join([str(dim) for dim in embeddings.shape[:-1]])

    return " + ".join(_embedding_count_expression(inner) for inner in embeddings)


def split_list_into_ranges(lst: torch.Tensor, interval: int) -> list[list[int]]:
    ranges: list[list[int]] = [[] for _ in range((max(lst) // interval) + 1)]
    for num in lst:
        index = num // interval
        ranges[index].append(num)
    return ranges


def _merge_multimodal_embeddings(
        inputs_embeds: torch.Tensor,
        multimodal_embeddings: NestedTensors,
        is_multimodal: torch.Tensor,
) -> torch.Tensor:
    """
    Merge `multimodal_embeddings` into `inputs_embeds` by overwriting the
    positions in `inputs_embeds` corresponding to placeholder tokens in
    `input_ids`.

    Note:
        This updates `inputs_embeds` in place.
    """
    if len(multimodal_embeddings) == 0:
        return inputs_embeds

    mm_embeds_flat = _flatten_embeddings(multimodal_embeddings)
    input_dtype = inputs_embeds.dtype

    try:
        # For debugging
        # inputs_embeds[is_multimodal] = mm_embeds_flat.to(dtype=input_dtype)

        # NOTE: This can avoid D2H sync (#22105), but fails to
        # raise an error if is_multimodal.sum() < len(mm_embeds_flat)
        inputs_embeds.masked_scatter_(
            is_multimodal.unsqueeze(-1), mm_embeds_flat.to(dtype=input_dtype)
        )
    except RuntimeError as e:
        num_actual_tokens = len(mm_embeds_flat)
        num_expected_tokens = is_multimodal.sum().item()

        if num_actual_tokens != num_expected_tokens:
            expr = _embedding_count_expression(multimodal_embeddings)

            raise ValueError(
                f"Attempted to assign {expr} = {num_actual_tokens} "
                f"multimodal tokens to {num_expected_tokens} placeholders"
            ) from e

        raise ValueError("Error during masked scatter operation") from e

    return inputs_embeds


def isin_list(
        elements: torch.Tensor,
        test_elements_list: list[int],
) -> torch.Tensor:
    test_elements = torch.tensor(
        test_elements_list,
        pin_memory=is_pin_memory_available(),
    ).to(device=elements.device, non_blocking=True)

    return torch.isin(elements, test_elements)


class StageMissingLayer(nn.Module):
    # 用于占位的“缺失阶段层”
    # 当某个 stage（例如 vision_tower）被禁用时，
    # 用这个类替代真实模块，避免其真正参与执行

    def __init__(self, stage_name: str, module: nn.Module | None = None) -> None:
        # 先初始化 PyTorch 基类 nn.Module
        super().__init__()

        # 记录当前占位层对应的阶段名称
        # 例如 "vision_tower"、"image_tower"
        self.stage_name = stage_name

        # 不要把 module 正式注册成子模块
        # 如果写成 self.module = module，
        # PyTorch 会把它加入 _modules，后续加载权重时可能出现 missing keys / unexpected keys 问题
        # 所以这里直接写入 __dict__，绕过 nn.Module 的注册机制
        self.__dict__["module"] = module

    def __getattr__(self, name: str):
        # 当访问当前对象上不存在的属性时，
        # 自动转发到内部保存的原始 module 上
        # 这样即使外面访问某些原模块属性，也还能工作
        return getattr(self.__dict__["module"], name)

    def __call__(self, *args, **kwargs):
        # 这个占位层不应该被真正调用执行 forward
        # 如果被调用，直接报错
        raise RuntimeError(f"{self} should not be called")

    def extra_repr(self) -> str:
        # 自定义打印信息
        # 在 print(module) 时，会额外显示 stage_name
        return f"stage_name={self.stage_name!r}"


@contextmanager
def collect_children(
        module: nn.Module,
        *,
        targets: type[nn.Module] | tuple[type[nn.Module], ...] | None = None,
):
    """
    在这个上下文中，收集赋值到 `module` 上的直接子模块名称，
    并返回一个列表；这个列表会在上下文期间持续被更新。

    如果传入了 `targets`，
    则不再只收集直接子模块，
    而是在上下文结束后，遍历 `module` 的所有后代模块，
    收集其中属于 `targets` 类型的模块名称。
    """
    # 用来保存收集到的子模块名字
    children_names = list[str]()

    # 情况 1：没有指定 targets
    # 这时只关心“直接赋值到 module 上的子模块”
    if targets is None:

        def hook(module_: nn.Module, name: str, submodule: nn.Module):
            # 这个 hook 会在模块注册子模块时被触发
            # module_：当前正在注册子模块的父模块
            # name：子模块名字，例如 "visual"
            # submodule：被赋值进去的那个模块对象

            # 只收集“直接挂在指定 module 上”的子模块
            # 比如 self.visual = ...
            # 如果是别的子模块内部再挂子模块，就不记
            if module_ is module:
                children_names.append(name)

        # 注册一个全局的“模块注册 hook”
        # 在 with 代码块执行期间，只要有子模块注册，就会调用上面的 hook
        with register_module_module_registration_hook(hook):
            # 把 children_names 暴露给外层 with 代码块使用
            yield children_names

    else:
        # 情况 2：指定了 targets
        # 这时不靠 hook 实时收集，而是先执行外层 with 块
        yield children_names

        # 外层 with 块执行完后，遍历 module 自身及其所有后代模块
        for name, module_ in module.named_modules():
            # 如果某个模块实例属于 targets 指定的类型
            if isinstance(module_, targets):
                # 就把它的层级名字记下来
                # 例如 "visual.blocks.0.attn"
                children_names.append(name)


@contextmanager
def no_init_weights(
        module: nn.Module,
        placeholder: Callable[[nn.Module], nn.Module],
        *,
        targets: type[nn.Module] | tuple[type[nn.Module], ...] | None = None,
):
    """
    在这个上下文中：

    1. 阻止模块初始化权重时占用真实设备内存
    2. 把赋值到 `module` 上的子模块，替换成 `placeholder()` 返回的占位模块

    如果未传入 `targets`：
        只处理“直接赋值给 module 的子模块”

    如果传入了 `targets`：
        则处理所有“属于 targets 类型的模块”，
        即使它们不是 `module` 的直接子模块也会被处理
    """

    # 情况 1：没有指定 targets
    # 只针对“直接挂到 module 上的 child module”
    if targets is None:

        def hook(module_: nn.Module, name: str, submodule: nn.Module):
            # 这个 hook 会在 PyTorch 注册子模块时触发
            # module_：当前被挂载子模块的父模块
            # name：子模块名，比如 "visual"
            # submodule：即将被挂进去的模块对象

            # 只处理直接赋值给指定 module 的子模块
            if module_ is module:
                # 不保留真实子模块，而是改成 placeholder(submodule)
                # 比如把一个很大的视觉塔替换成 StageMissingLayer
                return placeholder(submodule)

            # 其他情况不替换，原样返回
            return submodule

        # 两件事同时发生：
        # 1. 注册模块注册 hook，在赋值时替换模块
        # 2. 进入 torch.device("meta") 上下文
        #    使得在这段 with 里新建张量/参数时优先放到 meta device
        #    也就是“只保留形状和 dtype，不真正分配显存/内存”
        with register_module_module_registration_hook(hook), torch.device("meta"):
            yield

    # 情况 2：指定了 targets
    else:

        def hook(module_: nn.Module, name: str, submodule: nn.Module):
            # 如果“当前父模块”本身是目标类型
            if isinstance(module_, targets):
                # 把它移到 meta，释放真实内存
                submodule.to("meta")  # Free memory

            # 如果“即将挂载的子模块”是目标类型
            if isinstance(submodule, targets):
                # 同样把它先移到 meta，避免占用真实内存
                submodule.to("meta")  # Free memory

                # 再用 placeholder 替换掉它
                return placeholder(submodule)

            # 其他非目标模块不处理
            return submodule

        # 这里不能像上面那样直接全局用 torch.device("meta")
        # 因为并不是所有后代模块都属于 targets
        # 只想替换/搬到 meta 的那部分模块需要精细控制
        with register_module_module_registration_hook(hook):
            yield


class LayerFn(Protocol):
    def __call__(self, prefix: str) -> torch.nn.Module: ...


class PPMissingLayer(torch.nn.Identity):
    """
    流水线并行（PP）场景下的缺失层占位模块。
    当某一层不属于当前 PP rank 时，用这个占位层填充。
    它不做任何真实计算，只把输入原样返回。
    """

    def __init__(self, *args, **kwargs):
        # 不关心外部传入的参数，直接初始化父类 Identity
        super().__init__()

    def forward(self, *args, **kwargs):
        """
        直接返回输入中的“主张量”：
        - 若有位置参数，则返回第一个位置参数
        - 否则返回第一个关键字参数的值

        这样做的目的是在当前 rank 不负责该层时，
        让数据能够无损透传，而不真正执行该层计算。
        """
        return args[0] if args else next(iter(kwargs.values()))


def make_layers(
        num_hidden_layers: int,
        layer_fn: LayerFn,
        prefix: str,
) -> tuple[int, int, torch.nn.ModuleList]:
    """
    根据给定的层构造函数创建整组层，并考虑流水线并行（PP）。

    参数：
        num_hidden_layers: 模型总层数
        layer_fn: 给定层名前缀，返回一层模块的工厂函数
        prefix: 层名前缀

    返回：
        (start_layer, end_layer, modules)
        - start_layer: 当前 PP rank 负责的起始层号
        - end_layer: 当前 PP rank 负责的结束层号（右开）
        - modules: 长度等于总层数的 ModuleList
    """
    from cfie.distributed.parallel_state import get_pp_group
    from cfie.distributed.utils import get_pp_indices
    from cfie.model_executor.offloader import get_offloader

    # 根据总层数、当前 rank、PP world size，
    # 计算当前 rank 负责的层区间 [start_layer, end_layer)
    start_layer, end_layer = get_pp_indices(
        num_hidden_layers,
        get_pp_group().rank_in_group,
        get_pp_group().world_size
    )

    # 构造完整层列表：
    # 1. 当前 rank 之前的层：全部放 PPMissingLayer 占位
    # 2. 当前 rank 负责的层：真实创建，并交给 offloader 包装
    # 3. 当前 rank 之后的层：也全部放 PPMissingLayer 占位
    modules = torch.nn.ModuleList(
        [PPMissingLayer() for _ in range(start_layer)]
        # ------------ 实际layer --------------------
        + get_offloader().wrap_modules(
            layer_fn(prefix=f"{prefix}.{idx}") for idx in range(start_layer, end_layer)
        )
        # ------------ 实际layer --------------------
        + [PPMissingLayer() for _ in range(end_layer, num_hidden_layers)]
    )

    # 返回当前 rank 的层区间以及完整层列表
    return start_layer, end_layer, modules


# NOTE: don't use lru_cache here because it can prevent garbage collection
# 按 model 对象缓存“缺失层名前缀”列表。
# key 用 id(model) 而不是模型对象本身，避免额外引用影响回收。
_model_to_pp_missing_layer_names: dict[int, list[str]] = {}


def get_pp_missing_layer_names(model: torch.nn.Module) -> list[str]:
    # 取当前模型对象的唯一标识，用作缓存 key。
    model_id = id(model)

    # 如果这个模型之前已经扫描过，直接复用缓存结果。
    if model_id in _model_to_pp_missing_layer_names:
        return _model_to_pp_missing_layer_names[model_id]

    # 收集所有“当前 PP rank 不实际持有”的层名前缀。
    missing_layer_names = []

    # 遍历模型中所有模块，拿到它们的层级名称和模块对象。
    for name, module in model.named_modules():
        # 只关心两类占位模块：
        # 1. StageMissingLayer：整个 stage 被裁掉时的占位
        # 2. PPMissingLayer：某一层不属于当前 PP rank 时的占位
        if isinstance(module, (StageMissingLayer, PPMissingLayer)):
            # 这里故意在末尾补一个 "."，
            # 后面做参数名前缀匹配时能避免误匹配相似层名。
            missing_layer_names.append(name + ".")

    # 把扫描结果写入缓存，后续同一个模型不必重复遍历。
    _model_to_pp_missing_layer_names[model_id] = missing_layer_names

    # 返回缺失层名前缀列表，例如 ["layers.0.", "layers.1."]。
    return missing_layer_names


def is_pp_missing_parameter(name: str, model: torch.nn.Module) -> bool:
    # 如果传入的 model 自己就是缺失层占位模块，
    # 那么它下面的参数天然都应视为“当前 rank 不持有”。
    if isinstance(model, (StageMissingLayer, PPMissingLayer)):
        return True

    # 否则就看当前参数名是否以某个“缺失层名前缀”开头。
    # 只要命中任意一个缺失层前缀，就说明这个参数不属于当前 PP rank。
    return any(
        name.startswith(missing_layer_name)
        for missing_layer_name in get_pp_missing_layer_names(model)
    )


def make_empty_intermediate_tensors_factory(keys: list[str], hidden_size: int):
    def make_empty_intermediate_tensors(
            batch_size: int,
            dtype: torch.dtype,
            device: torch.device,
    ) -> IntermediateTensors:
        return IntermediateTensors(
            {
                key: torch.zeros((batch_size, hidden_size), dtype=dtype, device=device)
                for key in keys
            }
        )

    return make_empty_intermediate_tensors


def maybe_prefix(prefix: str, name: str) -> str:
    """Add a prefix to a name if the prefix is non-empty.

    Args:
        prefix: The prefix to add. If empty, no prefix will be added.
        name: The name to potentially prefix.

    Returns:
        The string "prefix.name" if prefix was non-empty, otherwise just "name".
    """
    return name if not prefix else f"{prefix}.{name}"


def get_draft_quant_config(
        cfie_config: CfieConfig,
) -> QuantizationConfig | None:
    """Get quantization config for Draft models.

    Draft models should use their own quantization config instead of the verifier/target
    model's config. This helper retrieves the draft model's quantization config.

    Args:
        cfie_config: The vLLM configuration object.

    Returns:
        The draft model's config if available, None otherwise.
    """
    draft_model_config = cfie_config.speculative_config.draft_model_config
    draft_load_config = cfie_config.load_config

    return (
        CfieConfig.get_quantization_config(draft_model_config, draft_load_config)
        if draft_model_config
        else None
    )


def extract_layer_index(layer_name: str, num_attn_module: int = 1) -> int:
    """
    从模块名字中提取层号。
    例子：
    - "encoder.layers.0" -> 0
    - "encoder.layers.1.self_attn" -> 1
    - "2.self_attn" -> 2
    - "model.encoder.layers.0.sub.1" -> 如果 num_attn_module == 1 会报错
    """

    # 按 '.' 拆分模块名
    subnames = layer_name.split(".")

    # 收集模块名中所有能解析成整数的部分
    int_vals: list[int] = []
    for subname in subnames:
        try:
            int_vals.append(int(subname))
        except ValueError:
            # 不是整数就跳过
            continue

    # 普通情况：
    # 1. 每层只有一个 attention 模块
    # 或 2. 模块名里没有 "attn"
    if num_attn_module == 1 or "attn" not in layer_name:
        # 此时模块名中应该且只能有一个整数
        assert len(int_vals) == 1, (
            f"layer name {layer_name} should only contain one integer"
        )

        # 这个唯一整数就是层号
        return int_vals[0]

    else:
        # 多 attention 模块场景：
        # 最多允许两个整数：一个表示层号，一个表示层内 attention 子模块号
        assert len(int_vals) <= 2, (
            f"layer name {layer_name} should contain most two integers"
        )

        # 如果有两个整数：
        # 全局层号 = 层号 * 每层attention模块数 + 层内attention号
        # 如果只有一个整数，则直接返回它
        layer_index = (
            int_vals[0] * num_attn_module + int_vals[1]
            if len(int_vals) == 2
            else int_vals[0]
        )
        return layer_index


def cast_overflow_tensors(
        tensors: torch.Tensor,
        offset: float = 1000,
) -> torch.Tensor:
    if tensors.isinf().any() or tensors.isnan().any():
        clamp_value = torch.finfo(tensors.dtype).max - offset
        tensors = torch.clamp(tensors, min=-clamp_value, max=clamp_value)
    return tensors


def fast_topk(
        values: torch.Tensor, topk: int, dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Optimized topk implementation that uses torch.max for k=1 case.

    This function provides better performance for the common case of k=1
    by using torch.max instead of the more general torch.topk.

    Args:
        values: Input tensor to find top-k values from
        topk: Number of top values to return (k). Must be > 0.
        dim: Dimension along which to compute topk

    Returns:
        Tuple of (values, indices) where values are the top-k values
        and indices are their corresponding indices in the input tensor
    """
    if topk == 1:
        # Use max along the specified dimension to get both value and index
        return torch.max(values, dim=dim, keepdim=True)
    else:
        # Use topk for efficiency with larger k values
        return torch.topk(values, topk, dim=dim)


# 沿 num_tokens 这一维把输入切成当前 rank 负责的 sequence parallel 本地分片。
# 这里包一层 torch custom op，而不是直接用普通 Python / PyTorch 逻辑切分，
# 是为了规避一个已知问题：在输入序列较短时，即使前面已经显式做过 padding，
# 输出张量的序列长度仍然可能意外变成 0。
def sequence_parallel_chunk(x: torch.Tensor) -> torch.Tensor:
    # 调用注册在 `torch.ops.cfie` 命名空间下的自定义算子。
    # 这一层只保留统一入口，真正的切分逻辑在 `sequence_parallel_chunk_impl` 中。
    return torch.ops.cfie.sequence_parallel_chunk_impl(x)



def sequence_parallel_chunk_impl(x: torch.Tensor) -> torch.Tensor:
    # 读取当前 tensor parallel 组大小，以及当前 rank 在组内的编号。
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()

    # 后续 sequence parallel / all-gather 相关逻辑要求 token 维能被 tp_size 整除。
    # 因此这里会先把第 0 维补齐到 tp_size 的整数倍。
    seq_len = x.size(0)
    remainder = seq_len % tp_size
    if remainder != 0:
        pad_len = tp_size - remainder
        y = nn.functional.pad(x, (0, 0, 0, pad_len))
    else:
        y = x

    # 补齐后，把序列平均切成 tp_size 份。
    chunk = y.shape[0] // tp_size
    # 当前 rank 只保留属于自己的那一段连续 token 分片。
    start = tp_rank * chunk
    return torch.narrow(y, 0, start, chunk)


def sequence_parallel_chunk_impl_fake(x: torch.Tensor) -> torch.Tensor:
    # fake 实现不做真实切分，只返回一个形状正确的占位张量。
    # 它主要服务于 fake tensor、torch.compile 和形状传播阶段。
    tp_size = get_tensor_model_parallel_world_size()

    # 第 0 维原本是全局 token 数；fake 路径里按 sequence parallel 的切分结果，
    # 这里只保留单个 TP rank 应看到的本地 token 数，即 ceil(num_tokens / tp_size)。
    seq_len = cdiv(x.size(0), tp_size)

    # 先拷贝输入张量的原始形状，例如 `[num_tokens, hidden_dim]`。
    shape = list(x.shape)

    # 再把第 0 维从全局 token 数改成当前 rank 的本地 token 数；
    # 其余维度保持不变，因此形状会从 `[num_tokens, ...]`
    # 变成 `[ceil(num_tokens / tp_size), ...]`。
    shape[0] = seq_len

    # 构造一个仅用于描述输出形状/类型/设备的占位张量，不承载真实数据。
    out = torch.empty(shape, dtype=x.dtype, device=x.device)
    return out


# 把 Python 实现直接注册为 Torch 自定义算子。
# 注册完成后，上层即可通过 `torch.ops.cfie.sequence_parallel_chunk_impl(...)`
# 调用它；同时也为编译/trace 场景提供 fake 实现。
direct_register_custom_op(
    # 算子在 `torch.ops.cfie` 下暴露的名字。
    op_name="sequence_parallel_chunk_impl",
    # 真实执行 token 分片逻辑的实现函数。
    op_func=sequence_parallel_chunk_impl,
    # fake tensor / compile 场景下使用的形状推导实现。
    fake_impl=sequence_parallel_chunk_impl_fake,
    # 声明该算子依赖固定 stride 顺序，避免布局假设被破坏。
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def process_eagle_weight(
        model: nn.Module,
        name: str,
) -> None:
    """
    Update EAGLE model flags based on loaded weight name.
    This should be called during weight loading to detect if a model
    has its own lm_head or embed_tokens weight.
    Args:
        model: The model instance (must support EAGLE)
        name: The name of the weight to process
    """
    if not supports_any_eagle(model):
        return

    # To prevent overriding with target model's layers
    if "lm_head" in name:
        model.has_own_lm_head = True
    if "embed_tokens" in name:
        model.has_own_embed_tokens = True


def get_layer_index(feature_layer_index: int, num_hidden_layers: int) -> int:
    """Given a signed vision feature layer, get the number of hidden layers
       needed to leverage it.

    Args:
        feature_layer_index: Index of a required layer in the visual encoder.
        num_hidden_layers: The total number of hidden layers in the visual encoder.
    """
    if feature_layer_index < 0:
        return num_hidden_layers + feature_layer_index + 1
    return feature_layer_index
