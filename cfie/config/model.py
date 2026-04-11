# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import warnings
from collections.abc import Callable
from dataclasses import InitVar, field
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal, cast, get_args

import torch
from pydantic import ConfigDict, Field, field_validator, model_validator

import cfie.envs as envs
from cfie.config.model_arch import (
    ModelArchitectureConfig,
)
from cfie.config.multimodal import MMCacheType, MMEncoderTPMode, MultiModalConfig
from cfie.config.pooler import PoolerConfig
from cfie.config.scheduler import RunnerType
from cfie.config.utils import config, getattr_iter
from cfie.logger import init_logger
from cfie.platforms import current_platform
from cfie.tasks import ScoreType
from cfie.transformers_utils.config import (
    ConfigFormat,
    get_config,
    get_hf_image_processor_config,
    get_hf_text_config,
    get_pooling_config,
    get_sentence_transformer_tokenizer_config,
    is_encoder_decoder,
    is_rope_parameters_nested,
    try_get_dense_modules,
    try_get_generation_config,
    try_get_tokenizer_config,
    uses_mrope,
    uses_xdrope_dim,
)
from cfie.transformers_utils.gguf_utils import (
    is_gguf,
    is_remote_gguf,
    maybe_patch_hf_config_from_gguf,
    split_remote_gguf,
)
from cfie.transformers_utils.model_arch_config_convertor import (
    MODEL_ARCH_CONFIG_CONVERTORS,
    ModelArchConfigConvertorBase,
)
from cfie.transformers_utils.runai_utils import ObjectStorageModel, is_runai_obj_uri
from cfie.transformers_utils.utils import maybe_model_redirect
from cfie.utils.import_utils import LazyLoader
from cfie.v1.attention.backends.registry import AttentionBackendEnum

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    import cfie.model_executor.layers.quantization as me_quant
    import cfie.model_executor.models as me_models
    from cfie.config.load import LoadConfig
    from cfie.config.parallel import ParallelConfig
    from cfie.model_executor.layers.quantization import QuantizationMethods
    from cfie.v1.sample.logits_processor import LogitsProcessor
else:
    PretrainedConfig = Any

    me_quant = LazyLoader(
        "model_executor", globals(), "cfie.model_executor.layers.quantization"
    )
    me_models = LazyLoader("model_executor", globals(), "cfie.model_executor.models")
    LoadConfig = Any
    ParallelConfig = Any
    QuantizationMethods = Any
    LogitsProcessor = Any

logger = init_logger(__name__)

RunnerOption = Literal["auto", RunnerType]
ConvertType = Literal["none", "embed", "classify"]
ConvertOption = Literal["auto", ConvertType]
TokenizerMode = Literal["auto", "hf", "slow", "mistral", "deepseek_v32"]
ModelDType = Literal["auto", "half", "float16", "bfloat16", "float", "float32"]
LogprobsMode = Literal[
    "raw_logits", "raw_logprobs", "processed_logits", "processed_logprobs"
]
HfOverrides = dict[str, Any] | Callable[[PretrainedConfig], PretrainedConfig]
ModelImpl = Literal["auto", "cfie", "transformers", "terratorch"]
LayerBlockType = Literal["attention", "linear_attention", "mamba"]

_RUNNER_CONVERTS: dict[RunnerType, list[ConvertType]] = {
    "generate": [],
    "pooling": ["embed", "classify", "reward"],
    "draft": [],
}

AttnTypeStr = Literal[
    "decoder", "encoder", "encoder_only", "encoder_decoder", "attention_free", "hybrid"
]


@config(config=ConfigDict(arbitrary_types_allowed=True))
class ModelConfig:
    """模型配置。"""

    model: str = "Qwen/Qwen3-0.6B"
    """要使用的 Hugging Face 模型名称或路径。
    当未指定 `served_model_name` 时，它也会作为指标输出中 `model_name`
    标签的内容。"""
    model_weights: str = ""
    """原始模型权重路径。
    当模型从对象存储（例如 RunAI）拉取时使用，用于在 `model`
    指向本地目录的同时保留原始 URI。"""
    runner: RunnerOption = "auto"
    """要使用的模型运行器类型。
    每个 vLLM 实例只支持一种 model runner，即使同一个模型可用于多种任务。"""
    convert: ConvertOption = "auto"
    """使用 [cfie.model_executor.models.adapters][] 中定义的适配器转换模型。
    最常见的用途是把文本生成模型适配为 pooling 任务使用。"""
    tokenizer: str = Field(default=None)
    """要使用的 Hugging Face tokenizer 名称或路径。
    如果未指定，则使用模型名称或路径。"""
    tokenizer_mode: TokenizerMode | str = "auto"
    """Tokenizer 模式：\n
    - `"auto"`：Mistral 模型若可用则使用 `mistral_common` 的 tokenizer，
      否则使用 `"hf"` tokenizer。\n
    - `"hf"`：若可用则使用 fast tokenizer。\n
    - `"slow"`：始终使用 slow tokenizer。\n
    - `"mistral"`：始终使用 `mistral_common` 的 tokenizer。\n
    - `"deepseek_v32"`：始终使用 `deepseek_v32` 的 tokenizer。\n
    - `"qwen_vl"`：始终使用 `qwen_vl` 的 tokenizer。\n
    - 其他自定义值可通过插件支持。"""
    trust_remote_code: bool = False
    """下载模型和 tokenizer 时，是否信任远端代码（例如来自 HuggingFace 的代码）。"""
    dtype: ModelDType | torch.dtype = "auto"
    """模型权重与激活使用的数据类型：\n
    - `"auto"`：对 FP32 和 FP16 模型使用 FP16 精度，对 BF16 模型使用 BF16 精度。\n
    - `"half"`：使用 FP16，推荐用于 AWQ 量化。\n
    - `"float16"`：与 `"half"` 相同。\n
    - `"bfloat16"`：在精度和数值范围之间取得平衡。\n
    - `"float"`：FP32 的简写。\n
    - `"float32"`：使用 FP32 精度。"""
    seed: int = 0
    """用于复现的随机种子。

    我们必须设置全局种子，否则不同 tensor parallel worker
    可能会采样出不同 token，导致结果不一致。"""
    hf_config: PretrainedConfig = field(init=False)
    """模型对应的 Hugging Face 配置对象。"""
    hf_text_config: PretrainedConfig = field(init=False)
    """文本模型对应的 Hugging Face 配置对象。
    对纯文本模型来说，它与 `hf_config` 相同。"""
    hf_config_path: str | None = None
    """要使用的 Hugging Face config 名称或路径。
    如果未指定，则使用模型名称或路径。"""
    allowed_local_media_path: str = ""
    """允许 API 请求从服务器文件系统指定目录中读取本地图像或视频。
    这存在安全风险，只应在可信环境中启用。"""
    allowed_media_domains: list[str] | None = None
    """如果设置了该项，则多模态输入只能使用属于这些域名的媒体 URL。"""
    revision: str | None = None
    """要使用的特定模型版本。
    可以是分支名、标签名或 commit id；如果未指定，则使用默认版本。"""
    code_revision: str | None = None
    """要在 Hugging Face Hub 上使用的模型代码 revision。
    可以是分支名、标签名或 commit id；如果未指定，则使用默认版本。"""
    tokenizer_revision: str | None = None
    """要在 Hugging Face Hub 上使用的 tokenizer revision。
    可以是分支名、标签名或 commit id；如果未指定，则使用默认版本。"""
    max_model_len: int = Field(default=None, ge=-1)
    """模型上下文长度（包含 prompt 和输出）。如果未指定，将自动从模型配置中推导。

    通过 `--max-model-len` 传入时，支持 `k/m/g/K/M/G` 这样的易读格式。
    示例：\n
    - `1k` -> 1000\n
    - `1K` -> 1024\n
    - `25.6k` -> 25,600\n
    - `-1` 或 `'auto'` -> 自动选择能够放入 GPU 显存的最大模型长度。
      如果模型的最大上下文长度本身放得下，就使用它；否则会寻找可容纳的最大长度。"""
    spec_target_max_model_len: int | None = None
    """指定 spec decoding 的 draft 模型最大长度。"""
    quantization: QuantizationMethods | str | None = None
    """权重量化方法。
    如果为 `None`，会先检查模型配置文件中的 `quantization_config`
    属性；如果仍为 `None`，则认为模型权重未量化，并使用 `dtype`
    来决定权重数据类型。"""
    allow_deprecated_quantization: bool = False
    """是否允许使用已废弃的量化方式。"""
    enforce_eager: bool = False
    """是否始终使用 eager 模式的 PyTorch。
    如果为 True，将禁用 CUDA graph，并始终以 eager 模式执行模型；
    如果为 False，则会混合使用 CUDA graph 和 eager execution，
    以获得更高性能与更好灵活性。"""
    enable_return_routed_experts: bool = False
    """是否返回路由到的专家信息。"""
    max_logprobs: int = 20
    """当 `SamplingParams` 中指定 `logprobs` 时，最多返回多少个 log probability。
    默认值来自 OpenAI Chat Completions API 的默认设置。
    `-1` 表示不设上限，即允许返回全部
    `(output_length * vocab_size)` 个 logprobs，这可能导致 OOM。"""
    logprobs_mode: LogprobsMode = "raw_logprobs"
    """指定 `logprobs` 和 `prompt_logprobs` 返回的内容。
    支持的模式有：
    1) `raw_logprobs`，2) `processed_logprobs`，3) `raw_logits`，
    4) `processed_logits`。
    `raw` 表示在应用任何 logit processor（如 bad words）之前的值；
    `processed` 表示应用了所有处理器（包括 temperature 和 top_k/top_p）之后的值。"""
    disable_sliding_window: bool = False
    """是否禁用滑窗注意力。
    如果为 True，会禁用模型的 sliding window 功能，并将上下文长度限制在滑窗大小内。
    如果模型本身不支持 sliding window，则该参数会被忽略。"""
    disable_cascade_attn: bool = True
    """在 V1 中禁用 cascade attention。
    虽然 cascade attention 不会改变数学正确性，但禁用它可能有助于避免潜在数值问题。
    该项默认值为 True，因此用户若想启用 cascade attention，必须显式设为 False。
    即使设为 False，也只有在启发式判断认为有收益时才会真正使用。"""
    skip_tokenizer_init: bool = False
    """跳过 tokenizer 和 detokenizer 初始化。
    此时输入需要提供有效的 `prompt_token_ids`，并且 prompt 本身必须为 `None`。
    生成输出将包含 token ids。"""
    enable_prompt_embeds: bool = False
    """如果为 `True`，允许通过 `prompt_embeds` 键直接传入文本 embedding 作为输入。

    警告：如果传入了形状不正确的 embeddings，vLLM 引擎可能会崩溃。
    该选项只应对可信用户启用！"""
    served_model_name: str | list[str] | None = None
    """API 对外使用的模型名。
    如果提供了多个名称，服务端会响应其中任意一个名称。
    响应中的 `model` 字段会使用列表中的第一个名称。
    若未指定，则与 `--model` 参数相同。
    这些名称也会用于 Prometheus 指标中的 `model_name` 标签；
    如果提供了多个名称，则指标标签取第一个。"""
    config_format: str | ConfigFormat = "auto"
    """要加载的模型配置格式：\n
    - `"auto"`：先尝试按 mistral 格式加载，若可用再按 hf 格式加载。\n
    - `"hf"`：按 hf 格式加载配置。\n
    - `"mistral"`：按 mistral 格式加载配置。"""
    hf_token: bool | str | None = None
    """访问远端文件时使用的 HTTP Bearer Token。
    如果为 `True`，则使用执行 `hf auth login` 时生成的 token
    （保存在 `~/.cache/huggingface/token` 中）。"""
    hf_overrides: HfOverrides = field(default_factory=dict)
    """如果是字典，则其中的参数会转发给 Hugging Face 配置对象。
    如果是可调用对象，则会调用它来更新 Hugging Face 配置。"""
    generation_config: str = "auto"
    """generation config 的目录路径。默认为 `"auto"`，此时会从模型路径加载
    generation config。若设为 `"cfie"`，则不加载 generation config，而使用
    vLLM 默认值。若设为某个目录路径，则会从该目录加载 generation config。
    如果 generation config 中指定了 `max_new_tokens`，它会为所有请求设置
    服务级别的输出 token 数上限。"""
    override_generation_config: dict[str, Any] = field(default_factory=dict)
    """覆写或设置 generation config，例如 `{"temperature": 0.5}`。
    若与 `--generation-config auto` 一起使用，覆写参数会与模型默认配置合并。
    若与 `--generation-config cfie` 一起使用，则只使用这些覆写参数。"""
    enable_sleep_mode: bool = False
    """为引擎启用 sleep mode（仅支持 cuda 和 hip 平台）。"""
    model_impl: str | ModelImpl = "auto"
    """选择使用哪种模型实现：\n
    - `"auto"`：如果存在 vLLM 实现则优先使用，否则回退到 Transformers 实现。\n
    - `"cfie"`：使用 vLLM 模型实现。\n
    - `"transformers"`：使用 Transformers 模型实现。\n
    - `"terratorch"`：使用 TerraTorch 模型实现。
    """
    override_attention_dtype: str | None = None
    """覆盖 attention 计算使用的 dtype。"""
    logits_processors: list[str | type[LogitsProcessor]] | None = None
    """一个或多个 logits processor 的完整类名或类定义。"""
    io_processor_plugin: str | None = None
    """模型启动时要加载的 IOProcessor 插件名。"""

    # Pooler 配置
    pooler_config: PoolerConfig | None = None
    """控制 pooling 模型输出池化行为的 Pooler 配置。"""

    # 多模态配置与初始化变量
    multimodal_config: MultiModalConfig | None = None
    """多模态模型的配置对象。如果为 `None`，则会根据 `self.model`
    的架构自动推断。"""
    language_model_only: InitVar[bool] = False
    limit_mm_per_prompt: InitVar[dict[str, int | dict[str, int]] | None] = None
    enable_mm_embeds: InitVar[bool | None] = None
    media_io_kwargs: InitVar[dict[str, dict[str, Any]] | None] = None
    mm_processor_kwargs: InitVar[dict[str, Any] | None] = None
    mm_processor_cache_gb: InitVar[float | None] = None
    mm_processor_cache_type: InitVar[MMCacheType | None] = None
    mm_shm_cache_max_object_size_mb: InitVar[int | None] = None
    mm_encoder_only: InitVar[bool | None] = None
    mm_encoder_tp_mode: InitVar[MMEncoderTPMode | None] = None
    mm_encoder_attn_backend: InitVar[AttentionBackendEnum | str | None] = None
    interleave_mm_strings: InitVar[bool | None] = None
    skip_mm_profiling: InitVar[bool | None] = None
    video_pruning_rate: InitVar[float | None] = None

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
        ignored_factors = {
            "convert",
            "tokenizer",
            "tokenizer_mode",
            "seed",
            "hf_config_path",
            "allowed_local_media_path",
            "allowed_media_domains",
            "tokenizer_revision",
            "spec_target_max_model_len",
            "enforce_eager",
            "logprobs_mode",
            "disable_cascade_attn",
            "skip_tokenizer_init",
            "served_model_name",
            "config_format",
            "hf_token",
            "hf_overrides",
            "override_attention_dtype",
            "logits_processors",
            "io_processor_plugin",
            "pooler_config",
            "multimodal_config",
            "limit_mm_per_prompt",
            "media_io_kwargs",
            "mm_processor_kwargs",
            "mm_processor_cache_gb",
            "mm_processor_cache_type",
            "mm_shm_cache_max_object_size_mb",
            "mm_encoder_tp_mode",
            "interleave_mm_strings",
            "skip_mm_profiling",
        }

        from cfie.config.utils import get_hash_factors, hash_factors

        factors = get_hash_factors(self, ignored_factors)

        # NOTE: For some models (e.g, Qwen3-VL), whether the MM code path is enabled
        # affects the computation graph of the language model, therefore we add it
        # here early.
        if self.multimodal_config:
            factors["language_model_only"] = self.multimodal_config.language_model_only
        return hash_factors(factors)

    def _update_nested(
            self,
            target: PretrainedConfig | dict[str, Any],
            updates: dict[str, Any],
    ) -> None:
        """Recursively updates a config or dict with nested updates."""
        for key, value in updates.items():
            if isinstance(value, dict):
                # Get the nested target
                if isinstance(target, dict):
                    nested_target = target.get(key)
                else:
                    nested_target = getattr(target, key, None)

                # If nested target exists and can be updated recursively
                if nested_target is not None and (
                        isinstance(nested_target, dict)
                        or hasattr(nested_target, "__dict__")
                ):
                    self._update_nested(nested_target, value)
                    continue

            # Set the value (base case)
            if isinstance(target, dict):
                target[key] = value
            else:
                setattr(target, key, value)

    def _apply_dict_overrides(
            self,
            # 已经加载完成、准备被覆写的 Hugging Face 配置对象。
            config: PretrainedConfig,
            # 需要应用到配置对象上的字典型覆写项。
            overrides: dict[str, Any],
    ) -> None:
        """应用字典形式的覆写，同时处理嵌套配置对象和普通字典值。"""
        # 延迟导入 PretrainedConfig，避免模块导入阶段增加不必要依赖。
        from transformers import PretrainedConfig

        # 逐项把外部传入的覆写内容应用到配置对象上。
        for key, value in overrides.items():
            # 先取出当前配置上同名字段的现值，用来判断它的类型。
            attr = getattr(config, key, None)
            # 如果当前字段本身就是嵌套的 PretrainedConfig，则递归更新其内部字段。
            if attr is not None and isinstance(attr, PretrainedConfig):
                # 这是嵌套配置对象，进入递归更新路径。
                self._update_nested(attr, value)
            else:
                # 否则把它当作普通字段或字典值，直接覆盖到 config 上。
                setattr(config, key, value)

    def __post_init__(
            self,
            # Multimodal config init vars
            language_model_only: bool,
            limit_mm_per_prompt: dict[str, int | dict[str, int]] | None,
            enable_mm_embeds: bool | None,
            media_io_kwargs: dict[str, dict[str, Any]] | None,
            mm_processor_kwargs: dict[str, Any] | None,
            mm_processor_cache_gb: float | None,
            mm_processor_cache_type: MMCacheType | None,
            mm_shm_cache_max_object_size_mb: int | None,
            mm_encoder_only: bool | None,
            mm_encoder_tp_mode: MMEncoderTPMode | None,
            mm_encoder_attn_backend: AttentionBackendEnum | str | None,
            interleave_mm_strings: bool | None,
            skip_mm_profiling: bool | None,
            video_pruning_rate: float | None,
    ) -> None:
        # Keep set served_model_name before maybe_model_redirect(self.model)
        # 先确定对外暴露的 served_model_name。
        self.served_model_name = get_served_model_name(
            self.model, self.served_model_name
        )

        # 对 model 路径做必要的重定向或规范化。
        self.model = maybe_model_redirect(self.model)

        # 未显式指定 tokenizer 时，默认复用 model。
        if self.tokenizer is None:
            self.tokenizer = self.model

        # 未显式指定 tokenizer_revision 时，默认复用模型 revision。
        if self.tokenizer_revision is None:
            self.tokenizer_revision = self.revision

        # 对 tokenizer 路径做必要的重定向或规范化。
        self.tokenizer = maybe_model_redirect(self.tokenizer)

        # 若显式给了 hf_config_path，也对其做路径重定向。
        if isinstance(self.hf_config_path, str):
            self.hf_config_path = maybe_model_redirect(self.hf_config_path)

        # callable 形式的 hf_overrides 直接作为回调保留。
        if callable(self.hf_overrides):
            # 平铺型 overrides 参数为空。
            hf_overrides_kw = {}
            # 保存覆写回调，后面传给 get_config。
            hf_overrides_fn = self.hf_overrides
            # 嵌套字典型 overrides 也为空。
            dict_overrides: dict[str, Any] = {}
        else:
            # 平铺 overrides 先初始化为空字典。
            hf_overrides_kw = {}
            # 嵌套 overrides 先初始化为空字典。
            dict_overrides = {}
            # 遍历 hf_overrides，把嵌套项和平铺项拆开。
            for key, value in self.hf_overrides.items():
                if isinstance(value, dict):
                    # 嵌套字典项稍后在加载后的 config 上递归应用。
                    dict_overrides[key] = value
                else:
                    # 标量类项可以直接透传给 get_config。
                    hf_overrides_kw[key] = value
            # 非 callable 场景下不使用覆写回调。
            hf_overrides_fn = None

        # RunAI / 对象存储场景下，必要时先把模型和 tokenizer 拉到本地。
        self.maybe_pull_model_tokenizer_for_runai(self.model, self.tokenizer)

        # 非 ROCm 平台设置 override_attention_dtype 时给出提示。
        if self.override_attention_dtype is not None and not current_platform.is_rocm():
            warnings.warn(
                "override-attention-dtype is set but not using ROCm platform",
                stacklevel=2,
            )

        # 若当前平台不支持 sleep mode，则直接拒绝该配置。
        if self.enable_sleep_mode and not current_platform.is_sleep_mode_available():
            raise ValueError("Sleep mode is not supported on current platform.")

        # 加载 HuggingFace / Mistral / GGUF 对应的基础配置对象。
        hf_config = get_config(
            self.hf_config_path or self.model,
            self.trust_remote_code,
            self.revision,
            self.code_revision,
            self.config_format,
            hf_overrides_kw=hf_overrides_kw,
            hf_overrides_fn=hf_overrides_fn,
        )

        # GGUF 场景下按量化文件补丁修正 hf_config。
        hf_config = maybe_patch_hf_config_from_gguf(
            self.model,
            hf_config,
        )

        # 保存解析后的顶层 HuggingFace 配置。
        self.hf_config = hf_config

        # 若存在嵌套 dict overrides，则在加载后的 config 上递归应用。
        if dict_overrides:
            self._apply_dict_overrides(hf_config, dict_overrides)

        # 提取文本子模型配置。
        self.hf_text_config = get_hf_text_config(self.hf_config)

        # 读取 attention_chunk_size，供后续注意力调度使用。
        self.attention_chunk_size = getattr(
            self.hf_text_config, "attention_chunk_size", None
        )

        # 提取 encoder 相关配置。
        self.encoder_config = self._get_encoder_config()

        # 读取图像处理器配置，供多模态路径使用。
        self.hf_image_processor_config = get_hf_image_processor_config(
            self.model, hf_token=self.hf_token, revision=self.revision
        )

        # 构造统一的模型架构配置对象。
        self.model_arch_config = self.get_model_arch_config()

        # 读取模型声明的 architectures 列表。
        architectures = self.architectures
        # 读取模型注册表。
        registry = self.registry
        # 判断该模型是否支持文本生成。
        is_generative_model = registry.is_text_generation_model(architectures, self)
        # 判断该模型是否支持 pooling。
        is_pooling_model = registry.is_pooling_model(architectures, self)

        # 结合 architectures 和配置解析最终 runner_type。
        self.runner_type = self._get_runner_type(architectures, self.runner)
        # 结合 runner_type 和 convert 配置解析最终 convert_type。
        self.convert_type = self._get_convert_type(
            architectures, self.runner_type, self.convert
        )

        # 若请求 generate，但模型本身不支持生成，则拒绝启动。
        if self.runner_type == "generate" and not is_generative_model:
            # generate 模式目前没有可用的转换器兜底。
            generate_converts = _RUNNER_CONVERTS["generate"]
            if self.convert_type not in generate_converts:
                # Currently we don't have any converters for generative models
                raise ValueError("This model does not support `--runner generate`.")
        # 若请求 pooling，但模型本身不支持 pooling，则检查是否能通过 convert 适配。
        if self.runner_type == "pooling" and not is_pooling_model:
            # pooling 模式允许若干 convert 作为适配路径。
            pooling_converts = _RUNNER_CONVERTS["pooling"]
            if self.convert_type not in pooling_converts:
                # 组织提示文案，告诉用户可选的 convert 方案。
                convert_option = "<" + "|".join(pooling_converts) + ">"
                raise ValueError(
                    "This model does not support `--runner pooling`. "
                    f"You can pass `--convert {convert_option} to adapt "
                    "it into a pooling model."
                )

        # Note: Initialize these attributes early because transformers fallback
        # may fail to load dynamic modules in child processes
        # 尽早解析模型类信息与架构名，避免子进程 fallback 失效。
        model_info, arch = registry.inspect_model_cls(architectures, self)
        # 缓存模型元信息。
        self._model_info = model_info
        # 缓存解析后的架构名。
        self._architecture = arch
        logger.info("Resolved architecture: %s", arch)

        # Set default tokenizer modes based on model architecture
        # tokenizer_mode=auto 时，按架构补默认 tokenizer_mode。
        if self.tokenizer_mode == "auto":
            if arch == "Grok1ForCausalLM":
                self.tokenizer_mode = "grok2"
            elif arch == "MoonshotKimiaForCausalLM":
                self.tokenizer_mode = "kimi_audio"
            elif arch == "QwenVLForConditionalGeneration":
                self.tokenizer_mode = "qwen_vl"
            elif arch == "DeepseekV32ForCausalLM":
                self.tokenizer_mode = "deepseek_v32"

            # 只有真的命中了某个专用 tokenizer_mode 才打印日志。
            if self.tokenizer_mode != "auto":
                logger.info(
                    "Defaulting to tokenizer_mode=%r for %s",
                    self.tokenizer_mode,
                    arch,
                )

        # Init pooler config if needed
        # pooling 模式下补齐 pooler_config 及其默认池化策略。
        if self.runner_type == "pooling":
            # 用户未提供 pooler_config 时创建默认对象。
            if self.pooler_config is None:
                self.pooler_config = PoolerConfig()

            # 读取模型自带的 pooling 配置。
            base_config = get_pooling_config(self.model, self.revision)
            if base_config is not None:
                # Only set values that are not overridden by the user
                # 仅在用户未显式覆盖时，回填模型默认 pooling 配置。
                for k, v in base_config.items():
                    if getattr(self.pooler_config, k) is None:
                        setattr(self.pooler_config, k, v)

            # 读取模型默认的序列池化类型。
            default_seq_pooling_type = self._model_info.default_seq_pooling_type
            # 用户未指定时，回填默认序列池化类型。
            if self.pooler_config.seq_pooling_type is None:
                self.pooler_config.seq_pooling_type = default_seq_pooling_type
            # 读取模型默认的 token 池化类型。
            default_tok_pooling_type = self._model_info.default_tok_pooling_type
            # 用户未指定时，回填默认 token 池化类型。
            if self.pooler_config.tok_pooling_type is None:
                self.pooler_config.tok_pooling_type = default_tok_pooling_type

        # 解析并校验最终使用的 torch dtype。
        self.dtype: torch.dtype = _get_and_verify_dtype(
            self.model,
            self.hf_config,
            self.dtype,
            is_pooling_model=self.runner_type == "pooling",
            revision=self.revision,
            config_format=self.config_format,
        )

        # 记录用户原始给定的 max_model_len。
        self.original_max_model_len = self.max_model_len
        # 解析并校验最终实际生效的 max_model_len。
        self.max_model_len = self.get_and_verify_max_len(self.max_model_len)

        # encoder-decoder 模型下禁用 mm processor cache。
        if self.is_encoder_decoder:
            mm_processor_cache_gb = 0
            logger.info("Encoder-decoder model detected, disabling mm processor cache.")

        # Init multimodal config if needed
        # 仅在模型确实支持多模态时才构造 multimodal_config。
        if self._model_info.supports_multimodal:
            if (
                    mm_encoder_tp_mode == "data"
                    and not self._model_info.supports_multimodal_encoder_tp_data
            ):
                # 若模型不支持 data 模式的多模态 encoder TP，则回退到 weights。
                logger.warning_once(
                    "This model does not support `--mm-encoder-tp-mode data`. "
                    "Falling back to `--mm-encoder-tp-mode weights`."
                )
                mm_encoder_tp_mode = "weights"

            # 把多模态相关 InitVar 汇总成构造参数。
            mm_config_kwargs = dict(
                language_model_only=language_model_only,
                limit_per_prompt=limit_mm_per_prompt,
                enable_mm_embeds=enable_mm_embeds,
                media_io_kwargs=media_io_kwargs,
                mm_processor_kwargs=mm_processor_kwargs,
                mm_processor_cache_gb=mm_processor_cache_gb,
                mm_processor_cache_type=mm_processor_cache_type,
                mm_shm_cache_max_object_size_mb=mm_shm_cache_max_object_size_mb,
                mm_encoder_only=mm_encoder_only,
                mm_encoder_tp_mode=mm_encoder_tp_mode,
                mm_encoder_attn_backend=mm_encoder_attn_backend,
                interleave_mm_strings=interleave_mm_strings,
                skip_mm_profiling=skip_mm_profiling,
                video_pruning_rate=video_pruning_rate,
            )

            # 过滤掉值为 None 的可选参数。
            mm_config_kwargs = {
                k: v for k, v in mm_config_kwargs.items() if v is not None
            }

            # 用整理后的参数构造 multimodal_config。
            self.multimodal_config = MultiModalConfig(**mm_config_kwargs)

        # Multimodal GGUF models must use original repo for mm processing
        # 多模态 GGUF 不能直接复用量化 tokenizer，必须指向原始 HF tokenizer。
        if is_gguf(self.tokenizer) and self.is_multimodal_model:
            raise ValueError(
                "Loading a multimodal GGUF model needs to use original "
                "tokenizer. Please specify the unquantized hf model's "
                "repo name or path using the --tokenizer argument."
            )

        if self.disable_sliding_window:
            # Set after get_and_verify_max_len to ensure that max_model_len
            # can be correctly capped to sliding window size
            # 最终确认禁用滑窗时，把 hf_text_config 上的滑窗标记清空。
            self.hf_text_config.sliding_window = None

        # Avoid running try_verify_and_update_config multiple times
        # 先标记配置尚未完成统一更新。
        self.config_updated = False
        # 统一补齐并校验模型配置。
        self._try_verify_and_update_model_config()
        # 校验量化配置是否合法。
        self._verify_quantization()
        # 校验 CUDA graph 相关配置是否合法。
        self._verify_cuda_graph()
        # 校验 bitsandbytes 相关配置是否合法。
        self._verify_bnb_config()

    def get_model_arch_config(
            self,
    ) -> ModelArchitectureConfig:
        convertor_cls = MODEL_ARCH_CONFIG_CONVERTORS.get(
            self.hf_config.model_type, ModelArchConfigConvertorBase
        )
        convertor = convertor_cls(self.hf_config, self.hf_text_config)
        return convertor.convert()

    @field_validator("tokenizer", "max_model_len", mode="wrap")
    @classmethod
    def _skip_none_validation(cls, value: Any, handler: Callable) -> Any:
        """Skip validation if the value is `None` when initialisation is delayed."""
        if value is None:
            return value
        return handler(value)

    @field_validator("tokenizer_mode", mode="after")
    def _lowercase_tokenizer_mode(cls, tokenizer_mode: str) -> str:
        return tokenizer_mode.lower()

    @field_validator("quantization", mode="before")
    @classmethod
    def validate_quantization_before(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.lower()
        return value

    @model_validator(mode="after")
    def validate_model_config_after(self: "ModelConfig") -> "ModelConfig":
        """Called after __post_init__"""
        if not isinstance(self.tokenizer, str):
            raise ValueError(
                f"tokenizer must be a string, got "
                f"{type(self.tokenizer).__name__}: {self.tokenizer!r}. "
                "Please provide a valid tokenizer path or HuggingFace model ID."
            )
        if not isinstance(self.max_model_len, int):
            raise ValueError(
                f"max_model_len must be a positive integer, "
                f"got {type(self.max_model_len).__name__}: {self.max_model_len!r}. "
                "Example: max_model_len=2048"
            )
        return self

    def _get_transformers_backend_cls(self) -> str:
        """Determine which Transformers modeling backend class will be used if
        `model_impl` is set to `transformers` or `auto`."""
        cls = "Transformers"
        # If 'hf_config != hf_text_config' it's a nested config, i.e. multimodal
        cls += "MultiModal" if self.hf_config != self.hf_text_config else ""
        cls += "MoE" if self.is_moe else ""
        # Check if the architecture we're wrapping has defaults
        runner = None
        task = None
        if defaults := try_match_architecture_defaults(self.architectures[0]):
            _, (runner, task) = defaults
        # User specified value take precedence
        if self.runner != "auto":
            runner = self.runner
        # Only consider Transformers modeling backend pooling classes if we're wrapping
        # an architecture that defaults to pooling. Otherwise, we return the LM class
        # and use adapters.
        if runner == "pooling" and task in {"embed", "classify"}:
            if task == "embed":
                cls += "EmbeddingModel"
            elif task == "classify":
                cls += "ForSequenceClassification"
        else:
            cls += "ForCausalLM"
        return cls

    def using_transformers_backend(self) -> bool:
        """Check if the model is using the Transformers modeling backend class."""
        used_cls = self._model_info.architecture
        transformers_backend_cls = self._get_transformers_backend_cls()
        return used_cls == transformers_backend_cls

    @property
    def registry(self):
        return me_models.ModelRegistry

    @property
    def architectures(self) -> list[str]:
        return self.model_arch_config.architectures

    @property
    def architecture(self) -> str:
        """The architecture cfie actually used."""
        return self._architecture

    def maybe_pull_model_tokenizer_for_runai(self, model: str, tokenizer: str) -> None:
        """Pull model/tokenizer from Object Storage to temporary
        directory when needed.

        Args:
            model: Model name or path
            tokenizer: Tokenizer name or path
        """

        # Skip if model_weights is already set (model already pulled)
        if self.model_weights:
            return

        if not (is_runai_obj_uri(model) or is_runai_obj_uri(tokenizer)):
            return

        if is_runai_obj_uri(model):
            object_storage_model = ObjectStorageModel(url=model)
            object_storage_model.pull_files(
                model, allow_pattern=["*.model", "*.py", "*.json"]
            )
            self.model_weights = model
            self.model = object_storage_model.dir

            # If tokenizer is same as model, download to same directory
            if model == tokenizer:
                object_storage_model.pull_files(
                    model,
                    ignore_pattern=[
                        "*.pt",
                        "*.safetensors",
                        "*.bin",
                        "*.tensors",
                        "*.pth",
                    ],
                )
                self.tokenizer = object_storage_model.dir
                return

        # Only download tokenizer if needed and not already handled
        if is_runai_obj_uri(tokenizer):
            object_storage_tokenizer = ObjectStorageModel(url=tokenizer)
            object_storage_tokenizer.pull_files(
                model,
                ignore_pattern=["*.pt", "*.safetensors", "*.bin", "*.tensors", "*.pth"],
            )
            self.tokenizer = object_storage_tokenizer.dir

    def _get_encoder_config(self) -> dict[str, Any] | None:
        model = self.model
        if is_remote_gguf(model):
            model, _ = split_remote_gguf(model)
        return get_sentence_transformer_tokenizer_config(model, self.revision)

    def _get_default_runner_type(
            self,
            architectures: list[str],
    ) -> RunnerType:
        registry = self.registry

        # Some Sentence Transformers models use *ForCausalLM archs
        if get_pooling_config(self.model, self.revision):
            return "pooling"

        for arch in architectures:
            if arch in registry.get_supported_archs():
                if registry.is_pooling_model(architectures, self):
                    return "pooling"
                if registry.is_text_generation_model(architectures, self):
                    return "generate"

            match = try_match_architecture_defaults(arch)
            if match:
                _, (runner_type, _) = match
                return runner_type

        return "generate"

    def _get_runner_type(
            self,
            architectures: list[str],
            runner: RunnerOption,
    ) -> RunnerType:
        if runner != "auto":
            return runner

        runner_type = self._get_default_runner_type(architectures)

        # Don't log the most common case
        if runner_type != "generate":
            logger.info(
                "Resolved `--runner auto` to `--runner %s`. "
                "Pass the value explicitly to silence this message.",
                runner_type,
            )

        return runner_type

    def _get_default_convert_type(
            self,
            architectures: list[str],
            runner_type: RunnerType,
    ) -> ConvertType:
        registry = self.registry

        for arch in architectures:
            if arch in registry.get_supported_archs():
                if runner_type == "generate" and registry.is_text_generation_model(
                        architectures, self
                ):
                    return "none"
                if runner_type == "pooling" and registry.is_pooling_model(
                        architectures, self
                ):
                    return "none"

            match = try_match_architecture_defaults(arch, runner_type=runner_type)
            if match:
                _, (_, convert_type) = match
                return convert_type

        # This is to handle Sentence Transformers models that use *ForCausalLM
        # and also multi-modal pooling models which are not defined as
        # Sentence Transformers models
        if runner_type == "pooling":
            return "embed"

        return "none"

    def _get_convert_type(
            self,
            architectures: list[str],
            runner_type: RunnerType,
            convert: ConvertOption,
    ) -> ConvertType:
        if convert != "auto":
            return convert

        convert_type = self._get_default_convert_type(architectures, runner_type)

        # Don't log the most common case
        if convert_type != "none":
            logger.info(
                "Resolved `--convert auto` to `--convert %s`. "
                "Pass the value explicitly to silence this message.",
                convert_type,
            )

        return convert_type

    def _verify_quantization(self) -> None:
        supported_quantization = me_quant.QUANTIZATION_METHODS
        if self.quantization is not None:
            self.quantization = cast(me_quant.QuantizationMethods, self.quantization)

        # Parse quantization method from the HF model config, if available.
        quant_cfg = self.model_arch_config.quantization_config

        if quant_cfg is not None:
            quant_method = quant_cfg["quant_method"]
            # Quantization methods which are overrides (i.e. they have a
            # `override_quantization_method` method) must be checked in order
            # of preference (this is particularly important for GPTQ).
            overrides = [
                "gptq_marlin",
                "awq_marlin",
                "inc",
                "moe_wna16",
                "modelopt",
                "modelopt_fp4",
                "modelopt_mxfp8",
                "modelopt_mixed",
                "petit_nvfp4",
                # Ensure heavy backends are probed last to avoid unnecessary
                # imports during override detection (e.g., MXFP4 imports Triton)
                "mxfp4",
                "cpu_awq",
            ]
            quantization_methods = [
                q for q in supported_quantization if q not in overrides
            ]
            # Any custom overrides will be in quantization_methods so we place
            # them at the start of the list so custom overrides have preference
            # over the built-in ones.
            quantization_methods = quantization_methods + overrides

            # Detect which checkpoint is it
            for name in quantization_methods:
                method = me_quant.get_quantization_config(name)
                quantization_override = method.override_quantization_method(
                    quant_cfg, self.quantization
                )
                if quantization_override is not None:
                    # Raise error if the override is not custom (custom would
                    # be in QUANTIZATION_METHODS but not QuantizationMethods)
                    # and hasn't been added to the overrides list.
                    if (
                            name in get_args(me_quant.QuantizationMethods)
                            and name not in overrides
                    ):
                        raise ValueError(
                            f"Quantization method {name} is an override but "
                            "is has not been added to the `overrides` list "
                            "above. This is necessary to ensure that the "
                            "overrides are checked in order of preference."
                        )
                    quant_method = quantization_override
                    self.quantization = quantization_override
                    break

            quant_method = quant_method if quant_method != "" else None
            # Verify quantization configurations.
            if self.quantization is None:
                self.quantization = quant_method
            elif self.quantization != quant_method:
                raise ValueError(
                    "Quantization method specified in the model config "
                    f"({quant_method}) does not match the quantization "
                    f"method specified in the `quantization` argument "
                    f"({self.quantization})."
                )

        if self.quantization is not None:
            if self.quantization not in supported_quantization:
                raise ValueError(
                    f"Unknown quantization method: {self.quantization}. Must "
                    f"be one of {supported_quantization}."
                )
            current_platform.verify_quantization(self.quantization)

        if self.quantization in me_quant.DEPRECATED_QUANTIZATION_METHODS:
            if self.allow_deprecated_quantization:
                logger.warning(
                    "The quantization method %s is deprecated "
                    "and will be removed in future versions of vLLM.",
                    self.quantization,
                )
            else:
                raise ValueError(
                    "The quantization method %s is deprecated "
                    "and will be removed in future versions of vLLM. To bypass, "
                    "set `--allow-deprecated-quantization`.",
                    self.quantization,
                )

    def _verify_cuda_graph(self) -> None:
        # CUDAGraph capture not supported for encoder-decoder models on ROCm
        unsupported_rocm = self.is_encoder_decoder
        if unsupported_rocm and not self.enforce_eager and current_platform.is_rocm():
            logger.warning(
                "CUDA graph is not supported for %s on ROCm yet, fallback "
                "to eager mode.",
                self.model_arch_config.model_type,
            )
            self.enforce_eager = True

    def _verify_bnb_config(self) -> None:
        """
        The current version of bitsandbytes (0.46.1) with 8-bit models does not
        yet support CUDA graph.
        # TODO Remove this when bitsandbytes supports.
        """
        is_bitsandbytes = self.quantization == "bitsandbytes"
        has_quantization_config = self.model_arch_config.quantization_config is not None
        is_8bit = (
            self.model_arch_config.quantization_config.get("load_in_8bit", False)
            if has_quantization_config
            else False
        )
        if all(
                [
                    is_bitsandbytes,
                    has_quantization_config,
                    is_8bit,
                    not self.enforce_eager,
                ]
        ):
            logger.warning(
                "CUDA graph is not supported on BitsAndBytes 8bit yet, "
                "fallback to the eager mode."
            )

            self.enforce_eager = True

    def _verify_with_expert_parallelism(self) -> None:
        if not self.is_moe:
            raise ValueError(
                "Number of experts in the model must be greater than 0 "
                "when expert parallelism is enabled."
            )

    def _try_verify_and_update_model_config(self):
        # Avoid running try_verify_and_update_config multiple times
        if getattr(self, "config_updated", False):
            return

        architecture = self.architecture
        if architecture is None:
            return

        from cfie.model_executor.models.config import (
            MODELS_CONFIG_MAP,
        )

        cls = MODELS_CONFIG_MAP.get(architecture, None)
        if cls is not None:
            cls.verify_and_update_model_config(self)

    def verify_dual_chunk_attention_config(
            self,
            load_config: LoadConfig,
    ) -> None:
        if hasattr(self.hf_config, "dual_chunk_attention_config"):
            # Try loading the sparse attention config
            from cfie.model_executor.model_loader.weight_utils import (
                get_sparse_attention_config,
            )

            sparse_attn_config = get_sparse_attention_config(self, load_config)
            if sparse_attn_config:
                self.hf_config.dual_chunk_attention_config[
                    "sparse_attention_config"
                ] = sparse_attn_config
                if (
                        "sparse_attention_enabled"
                        not in self.hf_config.dual_chunk_attention_config
                ):
                    self.hf_config.dual_chunk_attention_config[
                        "sparse_attention_enabled"
                    ] = True

    def verify_with_parallel_config(
            self,
            parallel_config: ParallelConfig,
    ) -> None:
        total_num_attention_heads = self.model_arch_config.total_num_attention_heads
        tensor_parallel_size = parallel_config.tensor_parallel_size
        if total_num_attention_heads % tensor_parallel_size != 0:
            raise ValueError(
                f"Total number of attention heads ({total_num_attention_heads})"
                " must be divisible by tensor parallel size "
                f"({tensor_parallel_size})."
            )

        if parallel_config.enable_expert_parallel:
            self._verify_with_expert_parallelism()

        pipeline_parallel_size = parallel_config.pipeline_parallel_size
        if pipeline_parallel_size > 1 and not self.registry.is_pp_supported_model(
                self.architectures, self
        ):
            raise NotImplementedError(
                "Pipeline parallelism is not supported for this model. "
                "Supported models implement the `SupportsPP` interface."
            )

        decode_context_parallel_size = parallel_config.decode_context_parallel_size
        if decode_context_parallel_size > 1 and not self.use_mla:
            total_num_kv_heads = self.get_total_num_kv_heads()
            assert tensor_parallel_size > total_num_kv_heads, (
                f"tensor parallel size {tensor_parallel_size} must be greater "
                f"than total num kv heads {total_num_kv_heads} when enable "
                f"decode context parallel for GQA/MQA"
            )

            max_dcp_size = tensor_parallel_size // total_num_kv_heads
            assert decode_context_parallel_size <= max_dcp_size, (
                f"decode context parallel size must less than or equal to "
                f"(tensor parallel size {tensor_parallel_size} // total "
                f"num kv heads {total_num_kv_heads}) = {max_dcp_size}, "
                f"but got {decode_context_parallel_size}"
            )

            num_q_per_kv = total_num_attention_heads // total_num_kv_heads
            assert num_q_per_kv % decode_context_parallel_size == 0, (
                f"Total number of q per kv attn heads ({num_q_per_kv})"
                " must be divisible by dcp world size when enable "
                "decode context parallel for GQA "
                f"({parallel_config.decode_context_parallel_size})."
            )

    def get_sliding_window(self) -> int | None:
        """Get the sliding window size from the HF text config if present."""
        return getattr(self.hf_text_config, "sliding_window", None)

    def get_vocab_size(self) -> int:
        return self.model_arch_config.vocab_size

    def get_hidden_size(self) -> int:
        return self.model_arch_config.hidden_size

    def get_inputs_embeds_size(self) -> int:
        # The size of inputs_embeds is usually identical to the size
        # of the hidden states, however there are exceptions, such as
        # embedding models like CLIP and SigLIP
        names = ("projection_dim", "projection_size")
        return getattr_iter(
            self.hf_text_config, names, default_factory=self.get_hidden_size
        )

    @property
    def is_deepseek_mla(self) -> bool:
        return self.model_arch_config.is_deepseek_mla

    @cached_property
    def is_mm_prefix_lm(self) -> bool:
        """Whether to use bidirectional attention for mm positions."""
        if hasattr(self.hf_config, "is_mm_prefix_lm"):
            return bool(self.hf_config.is_mm_prefix_lm)
        # fallback to list of known models
        MM_PREFIX_LM_MODELS = (
            "bagel",
            "gemma3",
            "molmo2",
            "paligemma",
        )
        if not hasattr(self.hf_config, "model_type"):
            return False
        return self.hf_config.model_type in MM_PREFIX_LM_MODELS

    def get_head_size(self) -> int:
        return self.model_arch_config.head_size

    def get_total_num_kv_heads(self) -> int:
        """Returns the total number of KV heads."""
        return self.model_arch_config.total_num_kv_heads

    def get_num_kv_heads(self, parallel_config: ParallelConfig) -> int:
        """Returns the number of KV heads per GPU."""
        if self.use_mla:
            # When using MLA during decode it becomes MQA
            return 1

        total_num_kv_heads = self.get_total_num_kv_heads()
        # If tensor parallelism is used, we divide the number of KV heads by
        # the tensor parallel size. We will replicate the KV heads in the
        # case where the number of KV heads is smaller than the tensor
        # parallel size so each GPU has at least one KV head.
        return max(1, total_num_kv_heads // parallel_config.tensor_parallel_size)

    def get_num_attention_heads(self, parallel_config: ParallelConfig) -> int:
        num_heads = self.model_arch_config.total_num_attention_heads
        return num_heads // parallel_config.tensor_parallel_size

    def get_num_experts(self) -> int:
        return self.model_arch_config.num_experts

    def get_total_num_hidden_layers(self) -> int:
        return self.model_arch_config.total_num_hidden_layers

    def get_layers_start_end_indices(
            self, parallel_config: ParallelConfig
    ) -> tuple[int, int]:
        from cfie.distributed.utils import get_pp_indices

        total_num_hidden_layers = self.get_total_num_hidden_layers()

        # the layout order is: DP x PP x TP
        pp_rank = (
                          parallel_config.rank // parallel_config.tensor_parallel_size
                  ) % parallel_config.pipeline_parallel_size
        pp_size = parallel_config.pipeline_parallel_size
        start, end = get_pp_indices(total_num_hidden_layers, pp_rank, pp_size)
        return start, end

    def get_num_layers(self, parallel_config: ParallelConfig) -> int:
        start, end = self.get_layers_start_end_indices(parallel_config)
        return end - start

    def get_num_layers_by_block_type(
            self,
            parallel_config: ParallelConfig,
            block_type: LayerBlockType = "attention",
    ) -> int:
        # This function relies on 'layers_block_type' in hf_config,
        # for w/o this attribute, we will need to have workarounds like so
        attn_block_type = block_type == "attention"
        is_transformer = (
                not self.is_hybrid and not self.has_noops and not self.is_attention_free
        )
        start, end = self.get_layers_start_end_indices(parallel_config)

        if is_transformer:
            # Handle the basic case first
            return end - start if attn_block_type else 0
        elif self.is_attention_free:
            # Attention free
            # Note that this code assumes there
            # is only one type of attention-free block type.
            return 0 if attn_block_type else end - start
        elif self.has_noops:
            block_configs = self.hf_config.block_configs
            return sum(not bc.attention.no_op for bc in block_configs[start:end])
        else:
            # Hybrid model Jamba
            layers_block_type_value = getattr(
                self.hf_text_config, "layers_block_type", None
            )
            if layers_block_type_value is not None:
                if self.model_arch_config.text_model_type == "zamba2":
                    if attn_block_type:
                        return sum(
                            t == "hybrid" for t in layers_block_type_value[start:end]
                        )
                    else:
                        return self.get_num_layers(parallel_config)
                return sum(t == block_type for t in layers_block_type_value[start:end])

            # Hybrid model Minimax
            attn_type_list = getattr(self.hf_config, "attn_type_list", None)
            if attn_type_list:
                return sum(t == 1 for t in attn_type_list[start:end])

            # Hybrid model Qwen3Next Qwen3.5 Series
            layer_types_value = getattr(self.hf_text_config, "layer_types", None)
            if layer_types_value is not None:
                if block_type == "attention":
                    return sum(
                        t == "full_attention" for t in layer_types_value[start:end]
                    )
                elif block_type == "linear_attention":
                    return sum(
                        t == "linear_attention" for t in layer_types_value[start:end]
                    )
                else:
                    return sum(t == block_type for t in layer_types_value[start:end])

            if (
                    layers_block_type_value is None
                    and attn_type_list is None
                    and layer_types_value is None
            ):
                raise ValueError(
                    "The model is an hybrid without a layers_block_type or an "
                    "attn_type_list, or a layer_types in the hf_config, "
                    f"cannot determine the num of {block_type} layers"
                )

    def get_mamba_chunk_size(self) -> int | None:
        """
        Returns the mamba chunk size if it exists
        """
        # used by e.g. Bamba, FalconH1, Granite, PLaMo2
        chunk_size = getattr(self.hf_text_config, "mamba_chunk_size", None)
        if chunk_size is None:
            # used by e.g. Mamba2, NemotronH, Zamba
            chunk_size = getattr(self.hf_text_config, "chunk_size", None)

        # Since Mamba1 does not have a chunk notion
        # we use a default chunk size of 1024.
        if chunk_size is None:
            chunk_size = 2048

        return chunk_size

    def get_multimodal_config(self) -> MultiModalConfig:
        """
        Get the multimodal configuration of the model.

        Raises:
            ValueError: If the model is not multimodal.
        """
        if self.multimodal_config is None:
            raise ValueError("The model is not multimodal.")

        return self.multimodal_config

    def try_get_generation_config(self) -> dict[str, Any]:
        """
        This method attempts to retrieve the non-default values of the
        generation config for this model.

        The generation config can contain information about special tokens, as
        well as sampling parameters. Which is why this method exists separately
        to `get_diff_sampling_param`.

        Returns:
            A dictionary containing the non-default generation config.
        """
        if self.generation_config in {"auto", "cfie"}:
            config = try_get_generation_config(
                self.hf_config_path or self.model,
                trust_remote_code=self.trust_remote_code,
                revision=self.revision,
                config_format=self.config_format,
            )
        else:
            config = try_get_generation_config(
                self.generation_config,
                trust_remote_code=self.trust_remote_code,
                config_format=self.config_format,
            )

        if config is None:
            return {}

        return config.to_diff_dict()

    def get_diff_sampling_param(self) -> dict[str, Any]:
        """
        This method returns a dictionary containing the non-default sampling
        parameters with `override_generation_config` applied.

        The default sampling parameters are:

        - vLLM's neutral defaults if `self.generation_config="cfie"`
        - the model's defaults if `self.generation_config="auto"`
        - as defined in `generation_config.json` if
            `self.generation_config="path/to/generation_config/dir"`

        Returns:
            A dictionary containing the non-default sampling parameters.
        """
        src = self.generation_config

        config = {} if src == "cfie" else self.try_get_generation_config()

        # Overriding with given generation config
        config.update(self.override_generation_config)

        available_params = [
            "repetition_penalty",
            "temperature",
            "top_k",
            "top_p",
            "min_p",
            "max_new_tokens",
        ]
        if any(p in config for p in available_params):
            diff_sampling_param = {
                p: config.get(p) for p in available_params if config.get(p) is not None
            }
            # Huggingface definition of max_new_tokens is equivalent
            # to vLLM's max_tokens
            if "max_new_tokens" in diff_sampling_param:
                diff_sampling_param["max_tokens"] = diff_sampling_param.pop(
                    "max_new_tokens"
                )
        else:
            diff_sampling_param = {}

        if diff_sampling_param and src != "cfie":
            logger.warning_once(
                "Default vLLM sampling parameters have been overridden by %s: `%s`. "
                "If this is not intended, please relaunch vLLM instance "
                "with `--generation-config cfie`.",
                "the model's `generation_config.json`" if src == "auto" else src,
                str(diff_sampling_param),
                scope="local",
            )

        return diff_sampling_param

    @cached_property
    def is_encoder_decoder(self) -> bool:
        """Extract the HF encoder/decoder model flag."""
        return is_encoder_decoder(self.hf_config)

    @property
    def uses_alibi(self) -> bool:
        cfg = self.hf_text_config

        return (
                getattr(cfg, "alibi", False)  # Falcon
                or "BloomForCausalLM" in self.architectures  # Bloom
                or getattr(cfg, "position_encoding_type", "") == "alibi"  # codellm_1b_alibi
                or (
                        hasattr(cfg, "attn_config")  # MPT
                        and (
                                (
                                        isinstance(cfg.attn_config, dict)
                                        and cfg.attn_config.get("alibi", False)
                                )
                                or (
                                        not isinstance(cfg.attn_config, dict)
                                        and getattr(cfg.attn_config, "alibi", False)
                                )
                        )
                )
        )

    @property
    def uses_mrope(self) -> bool:
        return uses_mrope(self.hf_config)

    @property
    def uses_xdrope_dim(self) -> int:
        return uses_xdrope_dim(self.hf_config)

    @property
    def is_multimodal_model(self) -> bool:
        return self.multimodal_config is not None

    @property
    def is_multimodal_raw_input_only_model(self) -> bool:
        return self._model_info.supports_multimodal_raw_input_only

    @property
    def requires_raw_input_tokens(self) -> bool:
        return self._model_info.requires_raw_input_tokens

    @property
    def score_type(self) -> ScoreType:
        """
        Score API handles score/rerank for:
        - "score" task (score_type: cross-encoder models)
        - "embed" task (score_type: bi-encoder models)
        - "token_embed" task (score_type: late interaction models)
        """
        # fixme: self._model_info.score_type is the score type before
        #  as_seq_cls_model, which is "bi-encoder", rather than the
        #  score type after as_seq_cls_model, which is "cross-encoder".
        #  Therefore, the following logic is required.
        return (
            "cross-encoder"
            if self.convert_type == "classify"
            else self._model_info.score_type
        )

    @property
    def is_pp_supported(self) -> bool:
        return self._model_info.supports_pp

    @property
    def is_attention_free(self) -> bool:
        return self._model_info.is_attention_free

    @property
    def is_hybrid(self) -> bool:
        if not self._model_info.is_hybrid:
            return False
        # Handle granite-4.0-micro case which uses hybrid config but does not
        # actually contain any non-attention layers.
        layer_types = getattr(self.hf_config, "layer_types", None)
        return layer_types is None or not all(
            layer == "attention" for layer in layer_types
        )

    @property
    def has_noops(self) -> bool:
        return self._model_info.has_noops

    @property
    def has_inner_state(self):
        return self._model_info.has_inner_state

    @property
    def supports_mamba_prefix_caching(self) -> bool:
        return self._model_info.supports_mamba_prefix_caching

    @property
    def use_mla(self) -> bool:
        return self.is_deepseek_mla and not envs.VLLM_MLA_DISABLE

    @property
    def is_matryoshka(self) -> bool:
        return bool(getattr(self.hf_config, "matryoshka_dimensions", None)) or getattr(
            self.hf_config, "is_matryoshka", False
        )

    @property
    def matryoshka_dimensions(self):
        return getattr(self.hf_config, "matryoshka_dimensions", None)

    @property
    def use_sep_token(self) -> bool:
        # cross_encoder models defaults to using separating token.
        # `llm as reranker` defaults to not using separating token.

        use_pad_token = getattr(self.hf_config, "use_pad_token", None)
        if use_pad_token is not None:
            logger.warning_once(
                "use_pad_token has been deprecated; please use use_sep_token instead."
            )
            return use_pad_token

        return getattr(self.hf_config, "use_sep_token", True)

    @property
    def head_dtype(self) -> torch.dtype:
        """
        "head" refers to the last Linear layer(s) of an LLM,
        such as the lm_head in a generation model,
        or the score or classifier in a classification model.

        `head_dtype` currently only supports pooling models.\n
        - The pooling model defaults to using fp32 head,
        you can use --hf-overrides '{"head_dtype": "model"}' to disable it.
        """

        head_dtype = _get_head_dtype(
            config=self.hf_config, dtype=self.dtype, runner_type=self.runner_type
        )

        if self.runner_type != "pooling" and head_dtype != self.dtype:
            logger.warning_once(
                "`head_dtype` currently only supports pooling models, "
                "fallback to model dtype [%s].",
                self.dtype,
            )
            return self.dtype

        if head_dtype not in current_platform.supported_dtypes:
            logger.warning_once(
                "The current platform does not support [%s] head dtype, "
                "fallback to model dtype [%s].",
                head_dtype,
                self.dtype,
            )
            return self.dtype

        logger.debug_once("head dtype: %s", head_dtype)
        return head_dtype

    @property
    def embedding_size(self):
        # Check for embedding_size set by model config (e.g., Voyage models)
        override = getattr(self.hf_config, "embedding_size", None)
        if override is not None:
            return override
        dense_modules = try_get_dense_modules(self.model, revision=self.revision)
        if dense_modules is not None:
            return dense_modules[-1]["out_features"]
        return self.get_hidden_size()

    def get_and_verify_max_len(self, max_model_len: int):
        # Consider max_model_len in tokenizer_config only when
        # pooling models use absolute position_embedding.
        tokenizer_config = None
        if (
                self.runner_type == "pooling"
                and getattr(self.hf_config, "position_embedding_type", "") == "absolute"
        ):
            tokenizer_config = try_get_tokenizer_config(
                self.tokenizer,
                trust_remote_code=self.trust_remote_code,
                revision=self.tokenizer_revision,
            )
        max_model_len = _get_and_verify_max_len(
            hf_config=self.hf_text_config,
            model_arch_config=self.model_arch_config,
            tokenizer_config=tokenizer_config,
            max_model_len=max_model_len,
            disable_sliding_window=self.disable_sliding_window,
            sliding_window=self.get_sliding_window(),
            spec_target_max_model_len=self.spec_target_max_model_len,
            encoder_config=self.encoder_config,
        )
        logger.debug(
            "Resolved initial max model len %s for model %s",
            max_model_len,
            self.model,
        )
        return max_model_len

    @property
    def attn_type(self) -> AttnTypeStr:
        # pooling 模型优先根据 pooler 与配置判断注意力类型。
        if self.pooler_config is not None:
            # 读取模型默认的序列池化方式。
            seq_pooling_type = self._model_info.default_seq_pooling_type
            if seq_pooling_type == "CLS":
                # CLS 池化通常要求双向编码器式注意力。
                return "encoder_only"
            else:
                # 读取 hf_config 中是否为 causal attention。
                is_causal = getattr(self.hf_config, "is_causal", True)
                # 非 causal 时按 encoder_only 处理，否则沿用模型信息中的 attn_type。
                return "encoder_only" if not is_causal else self._model_info.attn_type
        # hybrid 模型直接标记为 hybrid。
        elif self.is_hybrid:
            return "hybrid"
        # attention-free 模型单独标记。
        elif self.is_attention_free:
            return "attention_free"
        # encoder-decoder 模型单独标记。
        elif self.is_encoder_decoder:
            return "encoder_decoder"
        else:
            # 其余生成模型默认视为 decoder-only。
            return "decoder"

    @property
    def is_chunked_prefill_supported(self) -> bool:
        # 根据模型的注意力类型与任务形态，给 EngineArgs 提供默认 chunked prefill 能力。
        attn_type = self.attn_type

        if pooler_config := self.pooler_config:
            # ----------------- pooling 模型能力判定 -----------------
            if attn_type == "encoder_only":
                logger.debug(
                    "Pooling models with bidirectional attn "
                    "do not support chunked prefill."
                )
                return False

            if attn_type == "decoder":
                # causal pooling 里只有部分池化策略允许按 chunk 推进 prompt。
                if (
                        pooler_config.seq_pooling_type in ("MEAN", "CLS")
                        or pooler_config.tok_pooling_type == "STEP"
                ):
                    logger.debug(
                        "Pooling models with causal attn and %s/%s pooling "
                        "do not support chunked prefill.",
                        pooler_config.seq_pooling_type,
                        pooler_config.tok_pooling_type,
                    )
                    return False
                else:
                    # 其余 causal pooling 组合允许 chunked prefill。
                    logger.debug(
                        "Pooling models with causal attn and %s/%s pooling "
                        "support chunked prefill.",
                        pooler_config.seq_pooling_type,
                        pooler_config.tok_pooling_type,
                    )
                    return True

            # 当前除 encoder-decoder 外，其余 pooling 注意力类型默认都放行。
            return attn_type != "encoder_decoder"
        else:
            # ----------------- 生成模型能力判定 -----------------
            if attn_type == "encoder_decoder":
                # 生成式 encoder-decoder 模型不支持 chunked prefill。
                logger.debug("Encoder decoder models do not support chunked prefill.")
                return False

            # 其余生成模型默认支持 chunked prefill。
            logger.debug("Generative models support chunked prefill.")
            return True

    @property
    def is_prefix_caching_supported(self) -> bool:
        # 先根据当前模型推导出的注意力类型做判断。
        attn_type = self.attn_type

        if pooler_config := self.pooler_config:
            # for pooling models
            if attn_type == "encoder_only":
                logger.debug(
                    "Pooling models with bidirectional attn "
                    "do not support prefix caching."
                )
                return False

            if attn_type == "decoder":
                # 对 causal pooling 模型，部分池化策略不支持 prefix caching。
                if (
                        pooler_config.seq_pooling_type in ("MEAN", "CLS")
                        or pooler_config.tok_pooling_type == "STEP"
                ):
                    logger.debug(
                        "Pooling models with causal attn and %s/%s pooling "
                        "do not support prefix caching.",
                        pooler_config.seq_pooling_type,
                        pooler_config.tok_pooling_type,
                    )
                    return False
                else:
                    # 其余 causal pooling 组合允许 prefix caching。
                    logger.debug(
                        "Pooling models with causal attn and %s/%s pooling "
                        "support prefix caching.",
                        pooler_config.seq_pooling_type,
                        pooler_config.tok_pooling_type,
                    )
                    return True

            # cfie currently does not have pooling models using hybrid,
            # attention_free or encoder_decoder attn types.
            # 当前其余 pooling 注意力类型统一视为不支持 prefix caching。
            return False
        else:
            # for generative models
            if attn_type == "hybrid":
                # hybrid 生成模型里，MoE 路径已经具备 prefix caching 所需的
                # hybrid KV / mamba align 支持，因此默认允许开启。
                if self.is_moe:
                    logger.debug(
                        "Hybrid MoE generative models support prefix caching."
                    )
                    return True

                # 非 MoE 的 hybrid 生成模型仍维持保守默认值。
                logger.debug(
                    "Hybrid non-MoE models do not support prefix caching since "
                    "the feature is still experimental."
                )
                return False
            elif attn_type == "attention_free":
                # attention-free 模型的 prefix caching 仍属实验状态，先禁用。
                logger.debug(
                    "Attention free models do not support prefix caching since the "
                    "feature is still experimental."
                )
                return False
            elif attn_type == "encoder_decoder":
                # encoder-decoder 生成模型不支持 prefix caching。
                logger.debug("Encoder decoder models do not support prefix caching.")
                return False
            else:  # attn_type == "decoder"
                # 纯 decoder 生成模型默认支持 prefix caching。
                logger.debug("Generative models support prefix caching.")
                return True

    @property
    def is_moe(self) -> bool:
        return self.get_num_experts() > 0

    @property
    def is_quantized(self) -> bool:
        return getattr(self.hf_config, "quantization_config", None) is not None

    def is_nvfp4_quantized(self) -> bool:
        # ModelOpt NVFP4 checkpoints resolve to modelopt_fp4 quantization method
        if self.quantization in ("modelopt_fp4",):
            return True

        # For Compressed Tensors we look for `"format": "nvfp4-pack-quantized"`
        # in the quantization config
        quant_config = self.model_arch_config.quantization_config
        return (
                self.quantization == "compressed-tensors"
                and quant_config is not None
                and "nvfp4" in quant_config.get("format", "").lower()
        )


def get_served_model_name(model: str, served_model_name: str | list[str] | None):
    """
    If the input is a non-empty list, the first model_name in
    `served_model_name` is taken.
    If the input is a non-empty string, it is used directly.
    For cases where the input is either an empty string or an
    empty list, the fallback is to use `self.model`.
    """
    if not served_model_name:
        return model
    if isinstance(served_model_name, list):
        return served_model_name[0]
    return served_model_name


# Some model suffixes are based on auto classes from Transformers:
# https://huggingface.co/docs/transformers/en/model_doc/auto
# NOTE: Items higher on this list priority over lower ones
_SUFFIX_TO_DEFAULTS: list[tuple[str, tuple[RunnerType, ConvertType]]] = [
    ("ForCausalLM", ("generate", "none")),
    ("ForConditionalGeneration", ("generate", "none")),
    ("ChatModel", ("generate", "none")),
    ("LMHeadModel", ("generate", "none")),
    ("ForTextEncoding", ("pooling", "embed")),
    ("EmbeddingModel", ("pooling", "embed")),
    ("ForSequenceClassification", ("pooling", "classify")),
    ("ForTokenClassification", ("pooling", "classify")),
    ("ForAudioClassification", ("pooling", "classify")),
    ("ForImageClassification", ("pooling", "classify")),
    ("ForVideoClassification", ("pooling", "classify")),
    ("ClassificationModel", ("pooling", "classify")),
    ("ForRewardModeling", ("pooling", "embed")),
    ("RewardModel", ("pooling", "embed")),
    # Let other `*Model`s take priority
    ("Model", ("pooling", "embed")),
]


def iter_architecture_defaults():
    yield from _SUFFIX_TO_DEFAULTS


def try_match_architecture_defaults(
        architecture: str,
        *,
        runner_type: RunnerType | None = None,
        convert_type: ConvertType | None = None,
) -> tuple[str, tuple[RunnerType, ConvertType]] | None:
    for suffix, (
            default_runner_type,
            default_convert_type,
    ) in iter_architecture_defaults():
        if (
                (runner_type is None or runner_type == default_runner_type)
                and (convert_type is None or convert_type == default_convert_type)
                and architecture.endswith(suffix)
        ):
            return suffix, (default_runner_type, default_convert_type)

    return None


_STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.float16,
    "float16": torch.float16,
    "float": torch.float32,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def str_dtype_to_torch_dtype(type: str):
    return _STR_DTYPE_TO_TORCH_DTYPE.get(type)


# model_type -> reason
_FLOAT16_NOT_SUPPORTED_MODELS = {
    "gemma2": "Numerical instability. Please use bfloat16 or float32 instead.",
    "gemma3": "Numerical instability. Please use bfloat16 or float32 instead.",
    "gemma3_text": "Numerical instability. Please use bfloat16 or float32 instead.",
    "plamo2": "Numerical instability. Please use bfloat16 or float32 instead.",
    "glm4": "Numerical instability. Please use bfloat16 or float32 instead.",
}


def _is_valid_dtype(model_type: str, dtype: torch.dtype):
    if model_type in _FLOAT16_NOT_SUPPORTED_MODELS and dtype == torch.float16:  # noqa: E501, SIM103
        return False

    return True


def _check_valid_dtype(model_type: str, dtype: torch.dtype):
    if model_type in _FLOAT16_NOT_SUPPORTED_MODELS and dtype == torch.float16:
        reason = _FLOAT16_NOT_SUPPORTED_MODELS[model_type]
        raise ValueError(
            f"The model type {model_type!r} does not support float16. Reason: {reason}"
        )

    return True


def _resolve_auto_dtype(
        model_type: str,
        config_dtype: torch.dtype,
        *,
        is_pooling_model: bool,
):
    supported_dtypes = [
        dtype
        for dtype in current_platform.supported_dtypes
        if _is_valid_dtype(model_type, dtype)
    ]

    if is_pooling_model and torch.float16 in supported_dtypes:
        preferred_dtype = torch.float16
    else:
        preferred_dtype = supported_dtypes[0]

    # Downcast for float32 models
    if config_dtype == torch.float32:
        config_dtype = preferred_dtype

    if config_dtype in supported_dtypes:
        return config_dtype

    # Ensure device compatibility
    device_name = current_platform.get_device_name()
    device_capability = current_platform.get_device_capability()

    if device_capability is None:
        device_str = f"{device_name!r}"
    else:
        version_str = device_capability.as_version_str()
        device_str = f"{device_name!r} (with compute capability {version_str})"

    logger.warning(
        "Your device %s doesn't support %s. Falling back to %s for compatibility.",
        device_str,
        config_dtype,
        preferred_dtype,
    )

    return preferred_dtype


def _get_and_verify_dtype(
        model_id: str,
        config: PretrainedConfig,
        dtype: str | torch.dtype,
        *,
        is_pooling_model: bool,
        revision: str | None = None,
        config_format: ConfigFormat = "hf",
) -> torch.dtype:
    config_dtype = ModelArchConfigConvertorBase.get_torch_dtype(
        config, model_id, revision=revision, config_format=config_format
    )
    model_type = config.model_type

    if isinstance(dtype, str):
        dtype = dtype.lower()
        if dtype == "auto":
            # Set default dtype from model config
            torch_dtype = _resolve_auto_dtype(
                model_type,
                config_dtype,
                is_pooling_model=is_pooling_model,
            )
        else:
            if dtype not in _STR_DTYPE_TO_TORCH_DTYPE:
                raise ValueError(f"Unknown dtype: {dtype!r}")
            torch_dtype = _STR_DTYPE_TO_TORCH_DTYPE[dtype]
    elif isinstance(dtype, torch.dtype):
        torch_dtype = dtype
    else:
        raise ValueError(f"Unknown dtype: {dtype}")

    _check_valid_dtype(model_type, torch_dtype)

    if torch_dtype != config_dtype:
        if torch_dtype == torch.float32:
            # Upcasting to float32 is allowed.
            logger.info("Upcasting %s to %s.", config_dtype, torch_dtype)
        elif config_dtype == torch.float32:
            # Downcasting from float32 to float16 or bfloat16 is allowed.
            logger.info("Downcasting %s to %s.", config_dtype, torch_dtype)
        else:
            # Casting between float16 and bfloat16 is allowed with a warning.
            logger.warning("Casting %s to %s.", config_dtype, torch_dtype)

    return torch_dtype


def _get_head_dtype(
        config: PretrainedConfig, dtype: torch.dtype, runner_type: str
) -> torch.dtype:
    head_dtype: str | torch.dtype | None = getattr(config, "head_dtype", None)

    if head_dtype == "model":
        return dtype
    elif isinstance(head_dtype, str):
        head_dtype = head_dtype.lower()
        if head_dtype not in _STR_DTYPE_TO_TORCH_DTYPE:
            raise ValueError(f"Unknown dtype: {head_dtype!r}")
        return _STR_DTYPE_TO_TORCH_DTYPE[head_dtype]
    elif isinstance(head_dtype, torch.dtype):
        return head_dtype
    elif head_dtype is None:
        if torch.float32 not in current_platform.supported_dtypes:
            return dtype
        if runner_type == "pooling":
            return torch.float32
        return dtype
    else:
        raise ValueError(f"Unknown dtype: {head_dtype}")


def _get_and_verify_max_len(
        hf_config: PretrainedConfig,
        model_arch_config: ModelArchitectureConfig,
        tokenizer_config: dict | None,
        max_model_len: int | None,
        disable_sliding_window: bool,
        sliding_window: int | None,
        spec_target_max_model_len: int | None = None,
        encoder_config: dict[str, Any] | None = None,
) -> int:
    """Get and verify the model's maximum length."""
    (derived_max_model_len, max_len_key) = (
        model_arch_config.derived_max_model_len_and_key
    )

    # If sliding window is manually disabled, max_length should be less
    # than the sliding window length in the model config.
    if (
            disable_sliding_window
            and sliding_window is not None
            and sliding_window < derived_max_model_len
    ):
        max_len_key = "sliding_window"
        derived_max_model_len = sliding_window

    # Consider model_max_length in tokenizer_config
    if tokenizer_config:
        tokenizer_model_max_length = tokenizer_config.get(
            "model_max_length", derived_max_model_len
        )
        derived_max_model_len = min(derived_max_model_len, tokenizer_model_max_length)

    # If none of the keys were found in the config, use a default and
    # log a warning.
    if derived_max_model_len == float("inf"):
        if max_model_len is not None:
            # If max_model_len is specified, we use it.
            return max_model_len

        if spec_target_max_model_len is not None:
            # If this is a speculative draft model, we use the max model len
            # from the target model.
            return spec_target_max_model_len

        default_max_len = 2048
        logger.warning(
            "The model's config.json does not contain any of the keys "
            "to determine the original maximum length of the model. "
            "Assuming the model's maximum length is %d.",
            default_max_len,
        )
        derived_max_model_len = default_max_len

    # In Transformers v5 rope_parameters could be TypedDict or dict[str, TypedDict].
    # To simplify the verification, we convert it to dict[str, TypedDict].
    rope_parameters = getattr(hf_config, "rope_parameters", None)
    if rope_parameters and not is_rope_parameters_nested(rope_parameters):
        rope_parameters = {"": rope_parameters}

    # NOTE(woosuk): Gemma3's max_model_len (128K) is already scaled by RoPE
    # scaling, so we skip applying the scaling factor again.
    if rope_parameters is not None and "gemma3" not in hf_config.model_type:
        scaling_factor = 1.0
        for rp in rope_parameters.values():
            # No need to consider "type" key because of patch_rope_parameters when
            # loading HF config
            rope_type = rp["rope_type"]

            if rope_type not in ("su", "longrope", "llama3"):
                # NOTE: rope_type == "default" does not define factor https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/modeling_rope_utils.py
                # NOTE: This assumes all layer types have the same scaling factor.
                scaling_factor = rp.get("factor", scaling_factor)

                if rope_type == "yarn":
                    derived_max_model_len = rp["original_max_position_embeddings"]
        # Do this outside loop since all layer types should have the same scaling
        derived_max_model_len *= scaling_factor

    if encoder_config and "max_seq_length" in encoder_config:
        derived_max_model_len = encoder_config["max_seq_length"]

    # If the user didn't specify `max_model_len` or specified -1 (auto-fit),
    # then use that derived from the model config as a default value.
    # When -1 is specified, the engine will later auto-fit to available memory.
    if max_model_len is None or max_model_len == -1:
        # For LongRoPE, default to original_max_position_embeddings to avoid
        # performance degradation for shorter sequences
        if rope_parameters is not None and any(
                rp["rope_type"] == "longrope" for rp in rope_parameters.values()
        ):
            max_model_len = int(
                getattr(
                    hf_config, "original_max_position_embeddings", derived_max_model_len
                )
            )
        else:
            max_model_len = int(derived_max_model_len)
        max_model_len = current_platform.check_max_model_len(max_model_len)

    # If the user specified a max length, make sure it is smaller than the
    # derived length from the HF model config.
    elif max_model_len > derived_max_model_len:
        # Some models might have a separate key for specifying model_max_length
        # that will be bigger than derived_max_model_len. We compare user input
        # with model_max_length and allow this override when it's smaller.
        model_max_length = getattr(hf_config, "model_max_length", None)
        if model_max_length is None or max_model_len > model_max_length:
            msg = (
                f"User-specified max_model_len ({max_model_len}) is greater "
                f"than the derived max_model_len ({max_len_key}="
                f"{derived_max_model_len} or model_max_length="
                f"{model_max_length} in model's config.json)."
            )
            warning = (
                "VLLM_ALLOW_LONG_MAX_MODEL_LEN must be used with extreme "
                "caution. If the model uses relative position encoding (RoPE), "
                "positions exceeding derived_max_model_len lead to nan. If the "
                "model uses absolute position encoding, positions exceeding "
                "derived_max_model_len will cause a CUDA array out-of-bounds "
                "error."
            )
            if envs.VLLM_ALLOW_LONG_MAX_MODEL_LEN:
                logger.warning_once("%s %s", msg, warning)
            else:
                raise ValueError(
                    f"{msg} To allow overriding this maximum, set "
                    f"the env var VLLM_ALLOW_LONG_MAX_MODEL_LEN=1. {warning}"
                )
    return int(max_model_len)
