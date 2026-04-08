# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Callable
from dataclasses import asdict
from functools import cache, partial
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal, TypeAlias

import huggingface_hub
from huggingface_hub import get_safetensors_metadata
from packaging.version import Version
from transformers import GenerationConfig, PretrainedConfig
from transformers.models.auto.image_processing_auto import get_image_processor_config
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
from transformers.models.auto.tokenization_auto import get_tokenizer_config
from transformers.utils import CONFIG_NAME as HF_CONFIG_NAME

from cfie import envs
from cfie.logger import init_logger
from cfie.transformers_utils.repo_utils import is_mistral_model_repo
from cfie.transformers_utils.utils import (
    parse_safetensors_file_metadata,
    without_trust_remote_code,
)

from .config_parser_base import ConfigParserBase
from .gguf_utils import (
    check_gguf_file,
    is_gguf,
    is_remote_gguf,
    split_remote_gguf,
)
from .repo_utils import (
    file_or_path_exists,
    get_hf_file_to_dict,
    list_repo_files,
    try_get_local_file,
    with_retry,
)

try:
    # Transformers v5
    from transformers.configuration_utils import ALLOWED_ATTENTION_LAYER_TYPES
except ImportError:
    # Transformers v4
    from transformers.configuration_utils import (
        ALLOWED_LAYER_TYPES as ALLOWED_ATTENTION_LAYER_TYPES,
    )

if envs.VLLM_USE_MODELSCOPE:
    from modelscope import AutoConfig
else:
    from transformers import AutoConfig

MISTRAL_CONFIG_NAME = "params.json"

logger = init_logger(__name__)


class LazyConfigDict(dict):
    def __getitem__(self, key):
        if isinstance(value := super().__getitem__(key), type):
            return value

        import cfie.transformers_utils.configs as configs

        return getattr(configs, value)


_CONFIG_REGISTRY: dict[str, type[PretrainedConfig]] = LazyConfigDict(
    afmoe="AfmoeConfig",
    bagel="BagelConfig",
    chatglm="ChatGLMConfig",
    colmodernvbert="ColModernVBertConfig",
    colpali="ColPaliConfig",
    colqwen3="ColQwen3Config",
    ops_colqwen3="OpsColQwen3Config",
    qwen3_vl_nemotron_embed="Qwen3VLNemotronEmbedConfig",
    deepseek_vl_v2="DeepseekVLV2Config",
    deepseek_v32="DeepseekV3Config",
    flex_olmo="FlexOlmoConfig",
    funaudiochat="FunAudioChatConfig",
    hunyuan_vl="HunYuanVLConfig",
    isaac="IsaacConfig",
    kimi_k2="DeepseekV3Config",  # Kimi K2 uses same architecture as DeepSeek V3
    kimi_linear="KimiLinearConfig",
    kimi_vl="KimiVLConfig",
    kimi_k25="KimiK25Config",
    RefinedWeb="RWConfig",  # For tiiuae/falcon-40b(-instruct)
    RefinedWebModel="RWConfig",  # For tiiuae/falcon-7b(-instruct)
    jais="JAISConfig",
    mlp_speculator="MLPSpeculatorConfig",
    medusa="MedusaConfig",
    midashenglm="MiDashengLMConfig",
    eagle="EAGLEConfig",
    speculators="SpeculatorsConfig",
    nemotron="NemotronConfig",
    olmo3="Olmo3Config",
    olmo_hybrid="OlmoHybridConfig",
    ovis="OvisConfig",
    ultravox="UltravoxConfig",
    step3_vl="Step3VLConfig",
    step3_text="Step3TextConfig",
    step3p5="Step3p5Config",
    qwen3_asr="Qwen3ASRConfig",
    qwen3_next="Qwen3NextConfig",
    qwen3_5="Qwen3_5Config",
    qwen3_5_moe="Qwen3_5MoeConfig",
    qwen3_5_moe_predictor="Qwen3_5MoePredictorConfig",
    lfm2_moe="Lfm2MoeConfig",
    tarsier2="Tarsier2Config",
)

_CONFIG_ATTRS_MAPPING: dict[str, str] = {
    "llm_config": "text_config",
}

_AUTO_CONFIG_KWARGS_OVERRIDES: dict[str, dict[str, Any]] = {
    "internvl_chat": {"has_no_defaults_at_init": True},
    "Llama_Nemotron_Nano_VL": {"attn_implementation": "eager"},
    "NVLM_D": {"has_no_defaults_at_init": True},
}


def is_rope_parameters_nested(rope_parameters: dict[str, Any]) -> bool:
    """Check if rope_parameters is nested by layer types."""
    # Cannot be nested if rope_parameters is empty
    if not rope_parameters:
        return False
    return set(rope_parameters.keys()).issubset(ALLOWED_ATTENTION_LAYER_TYPES)


class HFConfigParser(ConfigParserBase):
    def parse(
            self,
            model: str | Path,
            trust_remote_code: bool,
            revision: str | None = None,
            code_revision: str | None = None,
            **kwargs,
    ) -> tuple[dict, PretrainedConfig]:
        kwargs["local_files_only"] = huggingface_hub.constants.HF_HUB_OFFLINE
        trust_remote_code |= kwargs.get("trust_remote_code", False)
        kwargs = without_trust_remote_code(kwargs)
        config_dict, _ = PretrainedConfig.get_config_dict(
            model,
            revision=revision,
            code_revision=code_revision,
            **kwargs,
        )
        # Use custom model class if it's in our registry
        model_type = config_dict.get("model_type")
        if model_type is None:
            model_type = (
                "speculators"
                if config_dict.get("speculators_config") is not None
                else model_type
            )
        # Allow hf_overrides to override model_type before checking _CONFIG_REGISTRY
        if (hf_overrides := kwargs.pop("hf_overrides", None)) is not None:
            if isinstance(hf_overrides, dict) and "model_type" in hf_overrides:
                model_type = hf_overrides["model_type"]
            elif callable(hf_overrides):
                # If hf_overrides doesn't modify model_type, it will be passed straight
                # through and remain unchanged by this elif block
                dummy_model_type = f"dummy_{model_type}"
                dummy_kwargs = dict(architectures=[""], model_type=dummy_model_type)
                dummy_config = PretrainedConfig(**dummy_kwargs)
                dummy_model_type = hf_overrides(dummy_config).model_type
                model_type = dummy_model_type.removeprefix("dummy_")

        if model_type in _CONFIG_REGISTRY:
            config_class = _CONFIG_REGISTRY[model_type]
            config = config_class.from_pretrained(
                model,
                revision=revision,
                code_revision=code_revision,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )
        else:
            try:
                kwargs = _maybe_update_auto_config_kwargs(kwargs, model_type=model_type)
                config = AutoConfig.from_pretrained(
                    model,
                    trust_remote_code=trust_remote_code,
                    revision=revision,
                    code_revision=code_revision,
                    **kwargs,
                )
            except ValueError as e:
                if (
                        not trust_remote_code
                        and "requires you to execute the configuration file" in str(e)
                ):
                    err_msg = (
                        "Failed to load the model config. If the model "
                        "is a custom model not yet available in the "
                        "HuggingFace transformers library, consider setting "
                        "`trust_remote_code=True` in LLM or using the "
                        "`--trust-remote-code` flag in the CLI."
                    )
                    raise RuntimeError(err_msg) from e
                else:
                    raise e
        config = _maybe_remap_hf_config_attrs(config)
        return config_dict, config


class MistralConfigParser(ConfigParserBase):
    def parse(
            self,
            model: str | Path,
            trust_remote_code: bool,
            revision: str | None = None,
            code_revision: str | None = None,
            **kwargs,
    ) -> tuple[dict, PretrainedConfig]:
        # This function loads a params.json config which
        # should be used when loading models in mistral format
        config_dict = _download_mistral_config_file(model, revision)
        if (
                max_position_embeddings := config_dict.get("max_position_embeddings")
        ) is None:
            max_position_embeddings = _maybe_retrieve_max_pos_from_hf(
                model, revision, **kwargs
            )
            config_dict["max_position_embeddings"] = max_position_embeddings

        from cfie.transformers_utils.configs.mistral import adapt_config_dict

        # Get missing fields from HF config if available
        try:
            hf_config_dict, _ = PretrainedConfig.get_config_dict(
                model,
                revision=revision,
                code_revision=code_revision,
                **without_trust_remote_code(kwargs),
            )
        except OSError:  # Not found
            hf_config_dict = {}

        config = adapt_config_dict(config_dict, defaults=hf_config_dict)

        return config_dict, config


_CONFIG_FORMAT_TO_CONFIG_PARSER: dict[str, type[ConfigParserBase]] = {
    "hf": HFConfigParser,
    "mistral": MistralConfigParser,
}

ConfigFormat = Literal[
    "auto",
    "hf",
    "mistral",
]


def get_config_parser(config_format: str) -> ConfigParserBase:
    """Get the config parser for a given config format."""
    if config_format not in _CONFIG_FORMAT_TO_CONFIG_PARSER:
        raise ValueError(f"Unknown config format `{config_format}`.")
    return _CONFIG_FORMAT_TO_CONFIG_PARSER[config_format]()


def register_config_parser(config_format: str):
    """Register a customized cfie config parser.
     When a config format is not supported by cfie, you can register a customized
    config parser to support it.
     Args:
         config_format (str): The config parser format name.
     Examples:

         >>> from cfie.transformers_utils.config import (get_config_parser,
                                                         register_config_parser)
         >>> from cfie.transformers_utils.config_parser_base import ConfigParserBase
         >>>
         >>> @register_config_parser("custom_config_parser")
         ... class CustomConfigParser(ConfigParserBase):
         ...     def parse(
         ...         self,
         ...         model: Union[str, Path],
         ...         trust_remote_code: bool,
         ...         revision: str | None = None,
         ...         code_revision: str | None = None,
         ...         **kwargs,
         ...     ) -> tuple[dict, PretrainedConfig]:
         ...         raise NotImplementedError
         >>>
         >>> type(get_config_parser("custom_config_parser"))
         <class 'CustomConfigParser'>
    """  # noqa: E501

    def _wrapper(config_parser_cls):
        if config_format in _CONFIG_FORMAT_TO_CONFIG_PARSER:
            logger.warning(
                "Config format `%s` is already registered, and will be "
                "overwritten by the new parser class `%s`.",
                config_format,
                config_parser_cls,
            )
        if not issubclass(config_parser_cls, ConfigParserBase):
            raise ValueError(
                "The config parser must be a subclass of `ConfigParserBase`."
            )
        _CONFIG_FORMAT_TO_CONFIG_PARSER[config_format] = config_parser_cls
        logger.info(
            "Registered config parser `%s` with config format `%s`",
            config_parser_cls,
            config_format,
        )
        return config_parser_cls

    return _wrapper


def set_default_rope_theta(config: PretrainedConfig, default_theta: float) -> None:
    """Some models may have no rope_theta in their config but still use RoPE.
    This function sets a default rope_theta if it's missing."""
    if getattr(config, "rope_parameters", None) is None:
        config.rope_parameters = {"rope_type": "default"}
    if "rope_theta" not in config.rope_parameters:
        config.rope_parameters["rope_theta"] = default_theta


def patch_rope_parameters(config: PretrainedConfig) -> None:
    """Provide backwards compatibility for RoPE."""
    from cfie.config.utils import getattr_iter

    # Older custom models may use non-standard field names
    # which need patching for both Transformers v4 and v5.
    names = ["rope_theta", "rotary_emb_base"]
    rope_theta = getattr_iter(config, names, None, warn=True)
    names = ["partial_rotary_factor", "rotary_pct", "rotary_emb_fraction"]
    partial_rotary_factor = getattr_iter(config, names, None, warn=True)
    ompe = getattr(config, "original_max_position_embeddings", None)

    if Version(version("transformers")) < Version("5.0.0"):
        # Transformers v4 installed, legacy config fields may be present
        if (rope_scaling := getattr(config, "rope_scaling", None)) is not None:
            config.rope_parameters = rope_scaling
        if (
                rope_theta is not None
                or partial_rotary_factor is not None
                or ompe is not None
        ) and not getattr(config, "rope_parameters", None):
            config.rope_parameters = {"rope_type": "default"}
        # Patch legacy fields into rope_parameters
        if rope_theta is not None:
            config.rope_parameters["rope_theta"] = rope_theta
        if partial_rotary_factor is not None:
            config.rope_parameters["partial_rotary_factor"] = partial_rotary_factor
        if ompe is not None:
            config.rope_parameters["original_max_position_embeddings"] = ompe
    elif rope_theta is not None or getattr(config, "rope_parameters", None):
        # Transformers v5 installed
        # Patch these fields in case they used non-standard names
        if rope_theta is not None:
            config.rope_theta = rope_theta
        if partial_rotary_factor is not None:
            config.partial_rotary_factor = partial_rotary_factor
        # Standardize and validate RoPE parameters
        config.standardize_rope_params()
        config.validate_rope()

    # No RoPE parameters to patch
    if getattr(config, "rope_parameters", None) is None:
        return

    # Handle nested rope_parameters in interleaved sliding attention models
    if is_rope_parameters_nested(config.rope_parameters):
        for rope_parameters_layer_type in config.rope_parameters.values():
            patch_rope_parameters_dict(rope_parameters_layer_type)
    else:
        patch_rope_parameters_dict(config.rope_parameters)


def patch_rope_parameters_dict(rope_parameters: dict[str, Any]) -> None:
    if "rope_type" in rope_parameters and "type" in rope_parameters:
        rope_type = rope_parameters["rope_type"]
        rope_type_legacy = rope_parameters["type"]
        if (rope_type_legacy == "su" and rope_type == "longrope") or (
                rope_type_legacy == "mrope" and rope_type == "default"
        ):
            pass  # No action needed
        elif rope_type != rope_type_legacy:
            raise ValueError(
                f"Found conflicts between 'rope_type={rope_type}' (modern "
                f"field) and 'type={rope_type_legacy}' (legacy field). "
                "You should only specify one of them."
            )

    if "rope_type" not in rope_parameters and "type" in rope_parameters:
        rope_parameters["rope_type"] = rope_parameters["type"]
        logger.info("Replacing legacy 'type' key with 'rope_type'")

    if "rope_type" not in rope_parameters:
        raise ValueError("rope_parameters should have a 'rope_type' key")

    if rope_parameters["rope_type"] == "su":
        rope_parameters["rope_type"] = "longrope"
        logger.warning("Replacing legacy rope_type 'su' with 'longrope'")
    elif rope_parameters["rope_type"] == "mrope":
        if "mrope_section" not in rope_parameters:
            raise ValueError(
                "Legacy rope_type 'mrope' requires 'mrope_section' in rope_parameters"
            )
        rope_parameters["rope_type"] = "default"
        logger.warning("Replacing legacy rope_type 'mrope' with 'default'")


def _uses_mrope(config: PretrainedConfig) -> bool:
    rope_parameters = getattr(config, "rope_parameters", None)
    if rope_parameters is None:
        return False

    return "mrope_section" in rope_parameters


def uses_mrope(config: PretrainedConfig) -> bool:
    """Detect if the model with this config uses M-ROPE."""
    return (
            _uses_mrope(config)
            or _uses_mrope(config.get_text_config())
            or thinker_uses_mrope(config)
    )


def thinker_uses_mrope(config: PretrainedConfig) -> bool:
    """Detect if the model contains a thinker config and it uses M-ROPE."""
    thinker_config = getattr(config, "thinker_config", None)
    if thinker_config is None:
        return False

    thinker_text_config = getattr(thinker_config, "text_config", None)
    if thinker_text_config is None:
        return False

    return uses_mrope(thinker_text_config)


def uses_xdrope_dim(config: PretrainedConfig) -> int:
    """Detect if the model with this config uses XD-ROPE."""
    xdrope_section = getattr(config, "xdrope_section", None)
    if xdrope_section is not None and isinstance(xdrope_section, list):
        return len(xdrope_section)
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling is None:
        return 0

    if isinstance(rope_scaling, dict) and "xdrope_section" in rope_scaling:
        xdrope_section = rope_scaling["xdrope_section"]
        if xdrope_section is not None and isinstance(xdrope_section, list):
            return len(xdrope_section)

    return 0


def is_encoder_decoder(config: PretrainedConfig) -> bool:
    """Detect if the model with this config is used as an encoder/decoder."""

    def _is_encoder_decoder(config: PretrainedConfig) -> bool:
        return getattr(config, "is_encoder_decoder", False)

    return _is_encoder_decoder(config) or _is_encoder_decoder(config.get_text_config())


def is_interleaved(config: PretrainedConfig) -> bool:
    """
    Detect if the model with this config is used with interleaved attention.
    """
    text_config = config.get_text_config()
    if layer_types := getattr(text_config, "layer_types", None):
        return len(set(layer_types)) > 1
    return False


def _maybe_update_auto_config_kwargs(kwargs: dict[str, Any], model_type: str):
    """
    Update kwargs for AutoConfig initialization based on model_type
    """
    if model_type in _AUTO_CONFIG_KWARGS_OVERRIDES:
        kwargs.update(_AUTO_CONFIG_KWARGS_OVERRIDES[model_type])
    return kwargs


def _maybe_remap_hf_config_attrs(config: PretrainedConfig) -> PretrainedConfig:
    """Remap config attributes to match the expected names."""
    for old_attr, new_attr in _CONFIG_ATTRS_MAPPING.items():
        if hasattr(config, old_attr):
            if not hasattr(config, new_attr):
                config.update({new_attr: getattr(config, old_attr)})
            logger.debug("Remapped config attribute '%s' to '%s'", old_attr, new_attr)
    return config


def maybe_override_with_speculators(
        # 当前传入的模型路径、仓库名或本地目录。
        model: str,
        # 当前传入的 tokenizer 路径；为空时通常复用 model。
        tokenizer: str | None,
        # 是否允许加载远端自定义代码。
        trust_remote_code: bool,
        # 模型 revision，如 branch、tag 或 commit。
        revision: str | None = None,
        # 调用侧已经存在的 speculative config，可为空。
        cfie_speculative_config: dict[str, Any] | None = None,
        # 透传给 HuggingFace 配置读取逻辑的额外参数。
        **kwargs,
) -> tuple[str, str | None, dict[str, Any] | None]:
    """
    Resolve model configuration when speculators are detected.

    Checks if the provided model is a speculators model and if so, extracts
    the target model configuration and builds the speculative config.

    Args:
        model: Model name or path
        tokenizer: Tokenizer name or path
        trust_remote_code: Whether to trust remote code
        revision: Model revision
        cfie_speculative_config: Existing vLLM speculative config

    Returns:
        Tuple of (resolved_model, resolved_tokenizer, speculative_config)
    """
    # 本地 GGUF 文件场景下，拆出文件名并把仓库目录作为配置读取根路径。
    if check_gguf_file(model):
        kwargs["gguf_file"] = Path(model).name
        gguf_model_repo = Path(model).parent
    # 远端 GGUF 场景下，从 `<repo>:<quant>` 里拆出 repo_id。
    elif is_remote_gguf(model):
        repo_id, _ = split_remote_gguf(model)
        gguf_model_repo = Path(repo_id)
    else:
        # 非 GGUF 模型直接按原始 model 路径读取配置。
        gguf_model_repo = None

    # 离线模式下强制只读取本地缓存文件。
    kwargs["local_files_only"] = huggingface_hub.constants.HF_HUB_OFFLINE

    # 先读取原模型的 config_dict，用来判断它是否是 speculators 模型。
    config_dict, _ = PretrainedConfig.get_config_dict(
        model if gguf_model_repo is None else gguf_model_repo,
        revision=revision,
        **without_trust_remote_code(kwargs),
    )

    # 从 config 中提取 speculators 配置段。
    speculators_config = config_dict.get("speculators_config")

    # 若不是 speculators 模型，则保持原 model/tokenizer/config 不变返回。
    if speculators_config is None:
        # No speculators config found, return original values
        return model, tokenizer, cfie_speculative_config

    # Speculators format detected - process overrides
    # 进入 speculators 模式后，导入对应配置解析器。
    from cfie.transformers_utils.configs.speculators.base import SpeculatorsConfig

    # 从 speculators config 中提取 cfie 可用的 speculative_config。
    speculative_config = SpeculatorsConfig.extract_cfie_speculative_config(
        config_dict=config_dict
    )

    # 将当前 speculators 模型本身登记为 draft model。
    speculative_config["model"] = model

    # 从 speculators 配置里取出 verifier 模型作为真正要跑的主模型。
    verifier_model = speculators_config["verifier"]["name_or_path"]

    # 用 verifier 模型覆盖 model 与 tokenizer。
    model = tokenizer = verifier_model

    # 返回覆写后的主模型、tokenizer 和 speculative config。
    return model, tokenizer, speculative_config


def get_config(
        # 模型名称、本地目录或模型文件路径。
        model: str | Path,
        # 是否信任远端仓库中的自定义代码。
        trust_remote_code: bool,
        # 模型权重与配置使用的 revision。
        revision: str | None = None,
        # 模型代码实现使用的 revision。
        code_revision: str | None = None,
        # 配置文件格式，可为 auto / hf / mistral。
        config_format: str | ConfigFormat = "auto",
        # 以字典形式覆写 Hugging Face 配置。
        hf_overrides_kw: dict[str, Any] | None = None,
        # 以回调形式二次修改 Hugging Face 配置。
        hf_overrides_fn: Callable[[PretrainedConfig], PretrainedConfig] | None = None,
        # 其他会继续透传给底层配置解析器的附加参数。
        **kwargs,
) -> PretrainedConfig:
    # 先把 GGUF 文件路径和模型目录拆开处理。

    # 判断当前输入是否是 GGUF 模型，以及是否是远端 GGUF 标识。
    _is_gguf = is_gguf(model)
    _is_remote_gguf = is_remote_gguf(model)

    # 如果是 GGUF，则需要把“文件”与“仓库/目录”分离。
    if _is_gguf:
        # 本地 GGUF 文件场景。
        if check_gguf_file(model):
            # 记录真正的 GGUF 文件名，后续 loader 会用到它。
            kwargs["gguf_file"] = Path(model).name
            # 配置读取阶段改为使用它所在目录。
            model = Path(model).parent
        # 远端 GGUF 场景。
        elif _is_remote_gguf:
            # 从 `<repo_id>:<quant_type>` 里拆出 repo_id 供配置读取使用。
            # 真正的 GGUF 文件下载会在后面的 GGUFModelLoader 中完成。
            # 下载链路保留完整模型标识，但配置链路只需要 repo_id。
            model, _ = split_remote_gguf(model)

    # 当配置格式设为 auto 时，自动判断是 HF 还是 Mistral 格式。
    if config_format == "auto":
        try:
            # 先探测 Mistral，避免错误地回退到 Transformers 默认实现。
            if is_mistral_model_repo(
                    model_name_or_path=str(model), revision=revision
            ) and file_or_path_exists(
                model=model, config_name=MISTRAL_CONFIG_NAME, revision=revision
            ):
                # 命中 Mistral 配置文件时，按 Mistral 格式解析。
                config_format = "mistral"

            # 本地 GGUF 或普通 HF 模型只要存在 config.json，就按 HF 格式处理。
            elif (_is_gguf and not _is_remote_gguf) or file_or_path_exists(
                    model, HF_CONFIG_NAME, revision=revision
            ):
                config_format = "hf"

            # 远端 GGUF 仓库必须带 config.json，否则无法正确解析配置。
            # FIXME(Isotr0py): 未来支持没有 config.json 的远端 GGUF 仓库。
            elif _is_remote_gguf and not file_or_path_exists(
                    model, HF_CONFIG_NAME, revision=revision
            ):
                # 给远端 GGUF 缺少 config.json 的场景构造明确报错信息。
                err_msg = (
                    "Could not find config.json for remote GGUF model repo. "
                    "To load remote GGUF model through `<repo_id>:<quant_type>`, "
                    "ensure your model has config.json (HF format) file. "
                    "Otherwise please specify --hf-config-path <original_repo> "
                    "in engine args to fetch config from unquantized hf model."
                )
                logger.error(err_msg)
                raise ValueError(err_msg)
            else:
                # auto 模式下既找不到 HF 配置也找不到 Mistral 配置，直接拒绝。
                raise ValueError(
                    "Could not detect config format for no config file found. "
                    "With config_format 'auto', ensure your model has either "
                    "config.json (HF format) or params.json (Mistral format). "
                    "Otherwise please specify your_custom_config_format "
                    "in engine args for customized config parser."
                )

        except Exception as e:
            # 对仓库 ID 非法、目录无效、配置缺失等情况统一包装成更友好的错误。
            error_message = (
                "Invalid repository ID or local directory specified:"
                " '{model}'.\nPlease verify the following requirements:\n"
                "1. Provide a valid Hugging Face repository ID.\n"
                "2. Specify a local directory that contains a recognized "
                "configuration file.\n"
                "   - For Hugging Face models: ensure the presence of a "
                "'config.json'.\n"
                "   - For Mistral models: ensure the presence of a "
                "'params.json'.\n"
            ).format(model=model)

            raise ValueError(error_message) from e

    # 根据最终确定的配置格式，拿到对应的配置解析器。
    config_parser = get_config_parser(config_format)

    # 解析模型配置，同时把必要的 revision / overrides 等参数透传下去。
    config_dict, config = config_parser.parse(
        model,
        trust_remote_code=trust_remote_code,
        revision=revision,
        code_revision=code_revision,
        hf_overrides=hf_overrides_kw or hf_overrides_fn,
        **kwargs,
    )

    # 针对 GGUF 模型补齐与 HF 默认值不一致的配置项。
    if _is_gguf:
        # 某些模型在 GGUF 与 HF 之间的默认值并不相同。
        def apply_gguf_default(key: str, gguf_default: Any):
            """
            如果用户没有显式配置该项，则应用 GGUF 侧默认值。

            这个函数会读写外层的 `config` 与 `config_dict`。
            当指定的 `key` 不在 `config_dict` 中时，说明用户没有显式配置，
            当前仍在使用 HF 默认值，此时就把 `config` 中对应字段改成
            `gguf_default`。
            """
            # 仅在配置文件没有显式给出该值时才覆盖默认值。
            if key not in config_dict:
                config.update({key: gguf_default})

        # 针对特定架构应用 GGUF 专属默认值。
        if config.model_type in {"qwen3_moe"}:
            # Qwen3 MoE 的 norm_topk_prob 在 GGUF 侧始终应为 true。
            # 注意这个参数在 Qwen2 MoE 的 HF 默认值里通常是 false。
            apply_gguf_default("norm_topk_prob", True)

    # 对 GGUF 模型做一次额外的 architecture 映射校验。
    if _is_gguf:
        # 如果当前 model_type 没有对应的因果语言模型映射，则无法继续。
        if config.model_type not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
            raise RuntimeError(f"Can't get gguf config for {config.model_type}.")
        # 把 GGUF 的 model_type 映射成标准 architectures 字段。
        model_type = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[config.model_type]
        config.update({"architectures": [model_type]})

    # 有些模型配置没有显式给 architectures，需要根据 model_type 反推。
    if not config.architectures:
        # 如果连映射表里都找不到，只能提示调用方自行通过 hf_overrides 补齐。
        if config.model_type not in MODEL_MAPPING_NAMES:
            logger.warning(
                "Model config does not have a top-level 'architectures' field: "
                "expecting `hf_overrides={'architectures': ['...']}` to be passed "
                "in engine args."
            )
        else:
            # 能在映射表中找到时，自动补上 architectures。
            model_type = MODEL_MAPPING_NAMES[config.model_type]
            config.update({"architectures": [model_type]})

    # ModelOpt 0.31.0 及之后会把量化配置直接存进模型配置文件。
    quantization_config = config_dict.get("quantization_config", None)

    # ModelOpt 0.29.0 及之前则会把量化配置单独写在同目录的 hf_quant_config.json 中。
    if quantization_config is None and file_or_path_exists(
            model, "hf_quant_config.json", revision
    ):
        # 如果主配置里没有，就退回去读单独的量化配置文件。
        quantization_config = get_hf_file_to_dict(
            "hf_quant_config.json", model, revision
        )

    # 只要拿到了量化配置，就挂到最终 config 对象上。
    if quantization_config is not None:
        config.quantization_config = quantization_config
        # 如果模型配置要求使用 UE8M0，就自动为 DeepGEMM 打开对应开关。
        scale_fmt = quantization_config.get("scale_fmt", None)
        if scale_fmt in ("ue8m0",):
            # 如果环境变量没显式设置过，就自动打开 UE8M0。
            if not envs.is_set("VLLM_USE_DEEP_GEMM_E8M0"):
                os.environ["VLLM_USE_DEEP_GEMM_E8M0"] = "1"
                logger.info_once(
                    (
                        "Detected quantization_config.scale_fmt=%s; "
                        "enabling UE8M0 for DeepGEMM."
                    ),
                    scale_fmt,
                )
            # 如果用户显式把环境变量设成 0，则只打印告警，不强行覆盖。
            elif not envs.VLLM_USE_DEEP_GEMM_E8M0:
                logger.warning_once(
                    (
                        "Model config requests UE8M0 "
                        "(quantization_config.scale_fmt=%s), but "
                        "VLLM_USE_DEEP_GEMM_E8M0=0 is set; "
                        "UE8M0 for DeepGEMM disabled."
                    ),
                    scale_fmt,
                )

    # 应用字典形式的 HF 配置覆写。
    if hf_overrides_kw:
        logger.debug("Overriding HF config with %s", hf_overrides_kw)
        config.update(hf_overrides_kw)
    # 应用回调形式的 HF 配置覆写。
    if hf_overrides_fn:
        logger.debug("Overriding HF config with %s", hf_overrides_fn)
        config = hf_overrides_fn(config)

    # 尽可能把所有可能存在的 RoPE 参数都统一修补到位。
    patch_rope_parameters(config)
    # 文本子配置也单独修补一次，避免遗漏。
    patch_rope_parameters(config.get_text_config())
    # 约定 `sub_configs` 是一个子配置名字到配置对象的映射。
    SubConfigs: TypeAlias = dict[str, PretrainedConfig]
    # 如果模型带有额外子配置，也一并遍历并修补 RoPE 参数。
    sub_configs: SubConfigs | None = getattr(config, "sub_configs", None)
    if sub_configs:
        for sub_config in sub_configs:
            patch_rope_parameters(getattr(config, sub_config))

    # 信任远端代码时，补注册按值序列化的配置处理逻辑。
    if trust_remote_code:
        maybe_register_config_serialize_by_value()

    # 返回最终整理完成的 Hugging Face 配置对象。
    return config


@cache
def get_pooling_config(
        model: str,
        revision: str | None = "main",
) -> dict[str, Any] | None:
    """
    This function gets the pooling and normalize
    config from the model - only applies to
    sentence-transformers models.

    Args:
        model: The name of the Hugging Face model.
        revision: The specific version of the model to use.
            Defaults to 'main'.

    Returns:
        A dictionary containing the pooling type and whether
            normalization is used, or None if no pooling configuration is found.
    """
    if is_remote_gguf(model):
        model, _ = split_remote_gguf(model)

    modules_file_name = "modules.json"

    modules_dict = None
    if file_or_path_exists(
            model=model, config_name=modules_file_name, revision=revision
    ):
        modules_dict = get_hf_file_to_dict(modules_file_name, model, revision)

    if modules_dict is None:
        return None

    logger.info("Found sentence-transformers modules configuration.")

    pooling = next(
        (
            item
            for item in modules_dict
            if item["type"] == "sentence_transformers.models.Pooling"
        ),
        None,
    )
    normalize = bool(
        next(
            (
                item
                for item in modules_dict
                if item["type"] == "sentence_transformers.models.Normalize"
            ),
            False,
        )
    )

    if pooling:
        from cfie.config.pooler import SEQ_POOLING_TYPES, TOK_POOLING_TYPES

        pooling_file_name = "{}/config.json".format(pooling["path"])
        pooling_dict = get_hf_file_to_dict(pooling_file_name, model, revision) or {}

        logger.info("Found pooling configuration.")

        config: dict[str, Any] = {"use_activation": normalize}
        for key, val in pooling_dict.items():
            if val is True:
                pooling_type = parse_pooling_type(key)
                if pooling_type in SEQ_POOLING_TYPES:
                    config["seq_pooling_type"] = pooling_type
                elif pooling_type in TOK_POOLING_TYPES:
                    config["tok_pooling_type"] = pooling_type
                else:
                    logger.debug("Skipping unrelated field: %r=%r", key, val)

        return config

    return None


def parse_pooling_type(pooling_name: str):
    if "pooling_mode_" in pooling_name:
        pooling_name = pooling_name.replace("pooling_mode_", "")

    if "_" in pooling_name:
        pooling_name = pooling_name.split("_", 1)[0]

    if "lasttoken" in pooling_name:
        pooling_name = "last"

    return pooling_name.upper()


@cache
def get_sentence_transformer_tokenizer_config(
        model: str | Path, revision: str | None = "main"
) -> dict[str, Any] | None:
    """
    Returns the tokenization configuration dictionary for a
    given Sentence Transformer BERT model.

    Parameters:
    - model (str|Path): The name of the Sentence Transformer
    BERT model.
    - revision (str, optional): The revision of the m
    odel to use. Defaults to 'main'.

    Returns:
    - dict: A dictionary containing the configuration parameters
    for the Sentence Transformer BERT model.
    """
    # sentence-transformers 不同底座模型会把 tokenizer 相关配置写在不同文件名里，
    # 这里把常见候选文件名都列出来，后面按顺序查找。
    sentence_transformer_config_files = [
        "sentence_bert_config.json",
        "sentence_roberta_config.json",
        "sentence_distilbert_config.json",
        "sentence_camembert_config.json",
        "sentence_albert_config.json",
        "sentence_xlm-roberta_config.json",
        "sentence_xlnet_config.json",
    ]
    encoder_dict = None

    # 先查本地目录或本地缓存；这是最便宜也最稳定的路径。
    for config_file in sentence_transformer_config_files:
        if (
                try_get_local_file(model=model, file_name=config_file, revision=revision)
                is not None
        ):
            # 找到文件后再真正读 JSON；若文件内容为空或非法，则继续尝试下一个。
            encoder_dict = get_hf_file_to_dict(config_file, model, revision)
            if encoder_dict:
                break

    # 本地没找到且 `model` 像一个 HF repo id 时，再回退到 Hub 查询文件列表。
    if not encoder_dict and not Path(model).is_absolute():
        try:
            # 这里只列目录，不直接下载全部内容。
            repo_files = list_repo_files(model, revision=revision)
        except Exception:
            repo_files = []

        for config_name in sentence_transformer_config_files:
            if config_name in repo_files:
                # 只读取命中的那个配置文件，避免无谓拉取。
                encoder_dict = get_hf_file_to_dict(config_name, model, revision)
                if encoder_dict:
                    break

    if not encoder_dict:
        # 普通生成模型大多不会提供这类 sentence-transformers 配置文件，
        # 因此返回 None 属于常见情况。
        return None

    logger.info("Found sentence-transformers tokenize configuration.")

    # 当前调用方只关心 `do_lower_case` 与 `max_seq_length` 这两个字段；
    # 若配置不完整，就当作不可用处理。
    if all(k in encoder_dict for k in ("max_seq_length", "do_lower_case")):
        return encoder_dict
    return None


def maybe_register_config_serialize_by_value() -> None:
    """Try to register HF model configuration class to serialize by value

    If trust_remote_code is set, and the model's config file specifies an
    `AutoConfig` class, then the config class is typically an instance of
    a custom class imported from the HF modules cache.

    Examples:

    >>> from transformers import AutoConfig
    >>> klass = AutoConfig.from_pretrained(
    ...     "meta-llama/Meta-Llama-3-8B", trust_remote_code=True
    ... )
    >>> klass.__class__  # transformers.models.llama.configuration_llama.LlamaConfig
    >>> import transformers_modules  # error, not initialized
    >>> klass = AutoConfig.from_pretrained(
    ...     "deepseek-ai/DeepSeek-V2.5", trust_remote_code=True
    ... )
    >>> import transformers_modules  # success, initialized
    >>> klass.__class__  # transformers_modules.deepseek-ai.DeepSeek-V2.5.98b11844770b2c3ffc18b175c758a803640f4e77.configuration_deepseek.DeepseekV2Config

    In the DeepSeek example, the config class is an instance of a custom
    class that is not serializable by default. This class will not be
    importable in spawned workers, and won't exist at all on
    other nodes, which breaks serialization of the config.

    In this function we tell the cloudpickle serialization library to pass
    instances of these generated classes by value instead of by reference,
    i.e. the class definition is serialized along with its data so that the
    class module does not need to be importable on the receiving end.

    See: https://github.com/cloudpipe/cloudpickle?tab=readme-ov-file#overriding-pickles-serialization-mechanism-for-importable-constructs
    """  # noqa
    try:
        import transformers_modules

        transformers_modules_available = True
    except ImportError:
        transformers_modules_available = False

    try:
        import multiprocessing
        import pickle

        import cloudpickle

        from cfie.config import CfieConfig

        # Register multiprocessing reducers to handle cross-process
        # serialization of CfieConfig objects that may contain custom configs
        # from transformers_modules
        def _reduce_config(config: CfieConfig):
            return (pickle.loads, (cloudpickle.dumps(config),))

        multiprocessing.reducer.register(CfieConfig, _reduce_config)

        # Register transformers_modules with cloudpickle if available
        if transformers_modules_available:
            cloudpickle.register_pickle_by_value(transformers_modules)

            # ray vendors its own version of cloudpickle
            from cfie.v1.executor.ray_utils import ray

            if ray:
                ray.cloudpickle.register_pickle_by_value(transformers_modules)

    except Exception as e:
        logger.warning(
            "Unable to register remote classes used by"
            " trust_remote_code with by-value serialization. This may"
            " lead to a later error. If remote code is not needed"
            " remove `--trust-remote-code`",
            exc_info=e,
        )


def get_hf_image_processor_config(
        model: str | Path,
        hf_token: bool | str | None = None,
        revision: str | None = None,
        **kwargs,
) -> dict[str, Any]:
    # ModelScope does not provide an interface for image_processor
    if envs.VLLM_USE_MODELSCOPE:
        return dict()
    # Separate model folder from file path for GGUF models
    if check_gguf_file(model):
        model = Path(model).parent
    elif is_remote_gguf(model):
        model, _ = split_remote_gguf(model)
    return get_image_processor_config(
        model, token=hf_token, revision=revision, **kwargs
    )


def get_hf_text_config(config: PretrainedConfig):
    """Get the "sub" config relevant to llm for multi modal models.
    No op for pure text models.
    """
    text_config = config.get_text_config()

    if text_config is not config and not hasattr(text_config, "num_attention_heads"):
        raise ValueError(
            "The text_config extracted from the model config does not have "
            "`num_attention_heads` attribute. This indicates a mismatch "
            "between the model config and vLLM's expectations. Please "
            "ensure that the model config is compatible with vLLM."
        )

    return text_config


def try_get_generation_config(
        model: str,
        trust_remote_code: bool,
        revision: str | None = None,
        config_format: str | ConfigFormat = "auto",
) -> GenerationConfig | None:
    # GGUF files don't have generation_config.json - their config is embedded
    # in the file header. Skip all filesystem lookups to avoid re-reading the
    # memory-mapped file, which can hang in multi-process scenarios when the
    # EngineCore process already has the file mapped.
    if is_gguf(model):
        return None

    try:
        return GenerationConfig.from_pretrained(
            model,
            revision=revision,
        )
    except OSError:  # Not found
        try:
            config = get_config(
                model,
                trust_remote_code=trust_remote_code,
                revision=revision,
                config_format=config_format,
            )
            return GenerationConfig.from_model_config(config)
        except OSError:  # Not found
            return None


def try_get_safetensors_metadata(
        model: str,
        *,
        revision: str | None = None,
):
    get_safetensors_metadata_partial = partial(
        get_safetensors_metadata, model, revision=revision
    )

    try:
        return with_retry(
            get_safetensors_metadata_partial, "Error retrieving safetensors"
        )
    except Exception:
        return None


def try_get_tokenizer_config(
        pretrained_model_name_or_path: str | os.PathLike,
        trust_remote_code: bool,
        revision: str | None = None,
) -> dict[str, Any] | None:
    try:
        return get_tokenizer_config(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            revision=revision,
        )
    except Exception:
        return None


@cache
def try_get_dense_modules(
        model: str | Path,
        revision: str | None = None,
) -> list[dict[str, Any]] | None:
    try:
        modules = get_hf_file_to_dict("modules.json", model, revision)
        if not modules:
            return None

        if isinstance(modules, dict):
            modules = modules.get("modules", [])

        _DENSE_MODULE_TYPES = {
            "sentence_transformers.models.Dense",
            "pylate.models.Dense.Dense",
        }
        dense_modules = [m for m in modules if m.get("type") in _DENSE_MODULE_TYPES]
        if not dense_modules:
            return None

        layer_configs = []
        for module in dense_modules:
            folder = module.get("path", "")

            config_path = f"{folder}/config.json" if folder else "config.json"
            layer_config = get_hf_file_to_dict(config_path, model, revision)
            if not layer_config:
                continue
            layer_config["folder"] = folder
            layer_configs.append(layer_config)
        return layer_configs
    except Exception:
        return None


def get_safetensors_params_metadata(
        model: str,
        *,
        revision: str | None = None,
) -> dict[str, Any]:
    """
    Get the safetensors parameters metadata for remote/local model repository.
    """
    # 用来汇总所有 safetensors 参数元数据；key 是参数名，value 是该参数的元信息。
    full_metadata = {}
    # 如果 model 指向本地路径且该路径存在，就按本地目录模式读取。
    if (model_path := Path(model)).exists():
        # 枚举目录下所有 `.safetensors` 文件，逐个解析它们的参数元数据。
        safetensors_to_check = model_path.glob("*.safetensors")
        # 把所有 safetensors 文件里的参数元数据合并成一个总字典。
        full_metadata = {
            # 参数名作为返回字典的 key。
            param_name: info
            # 逐个遍历命中的 safetensors 文件。
            for file_path in safetensors_to_check
            # 只处理真实文件，跳过目录等非文件条目。
            if file_path.is_file()
            # 解析单个 safetensors 文件头部元数据，并展开其中的参数项。
            for param_name, info in parse_safetensors_file_metadata(file_path).items()
        }
    else:
        # 否则按远端仓库模式读取，尝试从 Hub 侧直接拿 safetensors 元数据。
        repo_mt = try_get_safetensors_metadata(model, revision=revision)
        # 只有远端元数据存在且 files_metadata 非空时，才继续展开参数信息。
        if repo_mt and (files_mt := repo_mt.files_metadata):
            # 把每个远端 safetensors 文件里的 tensor 元信息摊平成一个总字典。
            full_metadata = {
                # 参数名作为返回字典的 key。
                param_name: asdict(info)
                # 遍历仓库中每个 safetensors 文件的元数据对象。
                for file_mt in files_mt.values()
                # 再遍历每个文件中的 tensor 元信息。
                for param_name, info in file_mt.tensors.items()
            }
    # 返回本地目录或远端仓库统一格式的 safetensors 参数元数据映射。
    return full_metadata


def _download_mistral_config_file(model, revision) -> dict:
    config_file_name = "params.json"
    config_dict = get_hf_file_to_dict(config_file_name, model, revision)
    if config_dict is None:
        raise ValueError(
            f"Failed to load mistral '{config_file_name}' config for model "
            f"{model}. Please check if the model is a mistral-format model "
            f"and if the config file exists."
        )
    assert isinstance(config_dict, dict)
    return config_dict


def _maybe_retrieve_max_pos_from_hf(model, revision, **kwargs) -> int:
    max_position_embeddings = 128_000
    try:
        trust_remote_code_val = kwargs.get("trust_remote_code", False)
        hf_config = get_config(
            model=model,
            trust_remote_code=trust_remote_code_val,
            revision=revision,
            config_format="hf",
        )
        if hf_value := hf_config.get_text_config().max_position_embeddings:
            max_position_embeddings = hf_value
    except Exception as e:
        logger.warning(
            "The params.json file is missing 'max_position_embeddings'"
            " and could not get a value from the HF config."
            " Defaulting to 128000",
            exc_info=e,
        )

    return max_position_embeddings
