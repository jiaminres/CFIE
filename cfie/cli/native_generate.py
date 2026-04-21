"""Native v1 engine one-shot generation entrypoint."""

from __future__ import annotations

import argparse
from argparse import Namespace
from typing import Any

from cfie.utils.logging import get_logger

logger = get_logger(__name__)

_GENERATION_SAMPLING_DEFAULTS_CACHE: dict[tuple[str, str | None], dict[str, Any]] = {}
_NEUTRAL_SAMPLING_DEFAULTS: dict[str, Any] = {
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "repetition_penalty": 1.0,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
}


# 注册 `native-generate` 子命令及其一轮生成所需参数。
def add_native_generate_parser(subparsers: Any) -> None:
    # 在顶层子命令集合里注册 native-generate。
    parser = subparsers.add_parser(
        "native-generate",
        help="Run one native CFIE/v1 generation request",
    )
    # 指定模型路径或模型名。
    parser.add_argument("--model", required=True)
    # 指定单轮生成输入的 prompt。
    parser.add_argument("--prompt", required=True)
    # 控制 CLI 级别日志输出。
    parser.add_argument("--log-level",
                        default="INFO",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    # 可选地单独指定 tokenizer。
    parser.add_argument("--tokenizer", default=None)
    # 指定模型 revision。
    parser.add_argument("--revision", default=None)
    # 指定下载目录。
    parser.add_argument("--download-dir", default=None)
    # 指定权重加载格式。
    parser.add_argument("--load-format", default="auto")
    # 指定推理 dtype。
    parser.add_argument("--dtype", default="auto")
    # 指定量化方式。
    parser.add_argument("--quantization", default=None)
    # 指定最大上下文长度。
    parser.add_argument("--max-model-len", type=int, default=32768)
    # 指定最大生成 token 数。
    parser.add_argument("--max-new-tokens", type=int, default=64)
    # 指定显存利用率上限。
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    # 控制 prefix cache 是否启用；不显式传入时沿用引擎默认值。
    parser.add_argument("--enable-prefix-caching",
                        action=argparse.BooleanOptionalAction,
                        default=None)
    # 指定 MoE CPU cache 预算。
    parser.add_argument("--moe-cpu-budget-gb", type=float, default=0.0)
    # 指定系统空闲内存保底。
    parser.add_argument("--moe-cpu-min-free-gb", type=float, default=0.0)
    # 指定通用 CPU offload 预算。
    parser.add_argument("--cpu-offload-gb", type=float, default=0.0)
    # 指定 offload 后端。
    parser.add_argument("--offload-backend", default="auto")
    # 指定张量并行大小。
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    # 指定最大并发序列数。
    parser.add_argument("--max-num-seqs", type=int, default=1)
    # 指定单轮调度的 batched token 预算上限。
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    # 指定 speculative decoding 方法。
    parser.add_argument("--spec-method",
                        choices=("none", "mtp"),
                        default="none")
    # 指定 speculative token 数。
    parser.add_argument("--num-speculative-tokens", type=int, default=None)
    # 指定 attention 后端。
    parser.add_argument("--attention-backend", default=None)
    # 指定 MoE 后端。
    parser.add_argument("--moe-backend", default="auto")
    # 指定 Mamba cache 模式，便于做显存预算诊断。
    parser.add_argument("--mamba-cache-mode", default=None)
    # 文本模式下跳过多模态处理链。
    parser.add_argument("--language-model-only",
                        action=argparse.BooleanOptionalAction,
                        default=False)
    # 跳过多模态 profiling，保持 CLI 与 Python 诊断路径一致。
    parser.add_argument("--skip-mm-profiling",
                        action=argparse.BooleanOptionalAction,
                        default=False)
    # 指定采样温度；未显式传入时优先使用模型 generation_config.json。
    parser.add_argument("--temperature", type=float, default=None)
    # 指定 top-p 采样阈值；未显式传入时优先使用模型 generation_config.json。
    parser.add_argument("--top-p", type=float, default=None)
    # 指定 top-k 采样阈值；未显式传入时优先使用模型 generation_config.json。
    parser.add_argument("--top-k", type=int, default=None)
    # 指定 presence penalty；未显式传入时使用模型配置或中性默认值。
    parser.add_argument("--presence-penalty", type=float, default=None)
    # 指定 frequency penalty；未显式传入时使用模型配置或中性默认值。
    parser.add_argument("--frequency-penalty", type=float, default=None)
    # 指定 repetition penalty；未显式传入时使用模型配置或中性默认值。
    parser.add_argument("--repetition-penalty", type=float, default=None)
    # 重复模式检测默认关闭；chat 子命令会单独给出更适合交互的默认值。
    parser.add_argument("--repetition-detection-max-pattern-size",
                        type=int,
                        default=0)
    parser.add_argument("--repetition-detection-min-pattern-size",
                        type=int,
                        default=4)
    parser.add_argument("--repetition-detection-min-count", type=int, default=3)
    # 指定随机种子。
    parser.add_argument("--seed", type=int, default=None)
    # 支持多次传入 stop 终止串。
    parser.add_argument("--stop", action="append", default=None)
    # 是否强制 eager 执行。
    parser.add_argument("--enforce-eager", action="store_true")
    # 是否启用多进程模式。
    parser.add_argument("--enable-multiprocessing", action="store_true")
    # 是否输出统计信息。
    parser.add_argument("--log-stats", action="store_true")
    # 绑定该子命令的执行函数。
    parser.set_defaults(handler=run_native_generate)


# 延迟导入运行时依赖，避免仅查看 CLI 帮助时就触发重型模块加载。
def _resolve_runtime_symbols():
    from cfie.engine.arg_utils import EngineArgs
    from cfie.sampling_params import RequestOutputKind, SamplingParams
    from cfie.v1.engine.llm_engine import LLMEngine

    return EngineArgs, SamplingParams, RequestOutputKind, LLMEngine


def _load_generation_sampling_defaults(args: Namespace) -> dict[str, Any]:
    cache_key = (args.model, args.revision)
    cached_defaults = _GENERATION_SAMPLING_DEFAULTS_CACHE.get(cache_key)
    if cached_defaults is not None:
        return cached_defaults

    defaults: dict[str, Any] = {}
    try:
        from cfie.transformers_utils.config import try_get_generation_config

        generation_config = try_get_generation_config(
            args.model,
            trust_remote_code=False,
            revision=args.revision,
        )
        if generation_config is not None:
            diff_config = generation_config.to_diff_dict()
            for name in _NEUTRAL_SAMPLING_DEFAULTS:
                value = diff_config.get(name)
                if value is not None:
                    defaults[name] = value
    except Exception:
        logger.debug(
            "failed to load generation sampling defaults for %s",
            args.model,
            exc_info=True,
        )

    _GENERATION_SAMPLING_DEFAULTS_CACHE[cache_key] = defaults
    return defaults


def _resolve_sampling_value(args: Namespace, name: str) -> Any:
    cli_value = getattr(args, name, None)
    if cli_value is not None:
        return cli_value
    generation_defaults = _load_generation_sampling_defaults(args)
    return generation_defaults.get(name, _NEUTRAL_SAMPLING_DEFAULTS[name])


def _build_repetition_detection_params(args: Namespace) -> Any:
    max_pattern_size = int(
        getattr(args, "repetition_detection_max_pattern_size", 0) or 0
    )
    if max_pattern_size <= 0:
        return None

    from cfie.sampling_params import RepetitionDetectionParams

    return RepetitionDetectionParams(
        max_pattern_size=max_pattern_size,
        min_pattern_size=int(
            getattr(args, "repetition_detection_min_pattern_size", 4) or 0
        ),
        min_count=int(getattr(args, "repetition_detection_min_count", 3) or 0),
    )


# 把 CLI 上的 speculative 相关参数整理成引擎可消费的配置字典。
def _build_speculative_config(args: Namespace) -> dict[str, Any] | None:
    # 未启用 speculative decoding 时直接返回空配置。
    if args.spec_method == "none":
        return None

    # 先写入最基本的方法名。
    config: dict[str, Any] = {"method": args.spec_method}
    # 若用户显式给了 speculative token 数，则一并写入。
    if args.num_speculative_tokens is not None:
        config["num_speculative_tokens"] = args.num_speculative_tokens
    # 返回可直接传入 EngineArgs 的 speculative 配置。
    return config


# 解析 dtype 别名，并在 GPTQ/Marlin 场景下为 `auto` 选出实际 dtype。
def _resolve_dtype(args: Namespace) -> str:
    # 将常见缩写统一映射成完整 dtype 名称。
    dtype_aliases = {
        "fp16": "float16",
        "bf16": "bfloat16",
    }
    # 优先把别名解析成标准值。
    dtype = dtype_aliases.get(args.dtype, args.dtype)

    # GPTQ/Marlin 在 dtype=auto 时强制落到 float16。
    if args.quantization in {"gptq", "gptq_marlin"} and dtype == "auto":
        logger.info(
            "forcing dtype=float16 for quantization=%s",
            args.quantization,
        )
        return "float16"

    # 其他情况直接返回解析后的 dtype。
    return dtype


# 把 `native-generate` CLI 参数转换成 `EngineArgs`。
def _build_engine_args(args: Namespace):
    # 动态导入 EngineArgs，避免仅查看 CLI 帮助时触发重模块加载。
    EngineArgs, _, _, _ = _resolve_runtime_symbols()

    # 将 CLI 参数逐项映射到引擎配置对象。
    engine_kwargs = dict(
        model=args.model,
        tokenizer=args.tokenizer,
        trust_remote_code=False,
        revision=args.revision,
        download_dir=args.download_dir,
        load_format=args.load_format,
        dtype=_resolve_dtype(args),
        quantization=args.quantization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=args.enable_prefix_caching,
        moe_cpu_budget_gb=args.moe_cpu_budget_gb,
        moe_cpu_min_free_gb=args.moe_cpu_min_free_gb,
        cpu_offload_gb=args.cpu_offload_gb,
        offload_backend=args.offload_backend,
        enforce_eager=args.enforce_eager,
        attention_backend=args.attention_backend,
        moe_backend=args.moe_backend,
        language_model_only=args.language_model_only,
        skip_mm_profiling=args.skip_mm_profiling,
        disable_log_stats=not args.log_stats,
        speculative_config=_build_speculative_config(args),
    )
    if args.mamba_cache_mode is not None:
        engine_kwargs["mamba_cache_mode"] = args.mamba_cache_mode
    return EngineArgs(**engine_kwargs)


# 把采样相关 CLI 参数整理成 `SamplingParams`。
def _build_sampling_params(args: Namespace):
    # 动态导入采样相关运行时符号。
    _, SamplingParams, RequestOutputKind, _ = _resolve_runtime_symbols()

    # 统一把 stop 参数整理成 None / str / list[str] 三种形态。
    stop: str | list[str] | None
    if not args.stop:
        # 未提供 stop 时不启用停止词。
        stop = None
    elif len(args.stop) == 1:
        # 只传了一个 stop 时直接保留为字符串。
        stop = args.stop[0]
    else:
        # 多个 stop 时转换为列表。
        stop = list(args.stop)

    # 组装 SamplingParams，默认输出 delta 片段供流式打印。
    return SamplingParams(
        presence_penalty=_resolve_sampling_value(args, "presence_penalty"),
        frequency_penalty=_resolve_sampling_value(args, "frequency_penalty"),
        repetition_penalty=_resolve_sampling_value(args, "repetition_penalty"),
        temperature=_resolve_sampling_value(args, "temperature"),
        top_p=_resolve_sampling_value(args, "top_p"),
        top_k=_resolve_sampling_value(args, "top_k"),
        seed=args.seed,
        stop=stop,
        max_tokens=args.max_new_tokens,
        output_kind=RequestOutputKind.DELTA,
        repetition_detection=_build_repetition_detection_params(args),
    )


# 从 engine step 的输出中筛出当前请求的增量文本片段。
def _iter_request_text(outputs: list[Any], request_id: str):
    # 逐个遍历 engine 输出对象。
    for output in outputs:
        # 只保留当前 request_id 对应的输出。
        if getattr(output, "request_id", None) != request_id:
            continue
        # 一个请求可能包含多个 completion，逐个取出文本增量。
        for completion in getattr(output, "outputs", []) or []:
            # 从 completion 上读取本轮新增文本。
            text = getattr(completion, "text", "")
            # 仅产出非空文本，避免打印空串。
            if text:
                yield text


def _render_engine_prompt(engine: Any, args: Namespace) -> Any:
    from cfie.renderers.params import TokenizeParams

    rendered_prompts = engine.renderer.render_cmpl(
        [{"prompt": args.prompt}],
        TokenizeParams(
            max_total_tokens=args.max_model_len,
            max_output_tokens=args.max_new_tokens,
        ),
    )
    return rendered_prompts[0]


# 执行单轮生成：建引擎、提交请求、轮询 step 并流式输出文本。
def run_native_generate(args: Namespace) -> int:
    # 动态解析 LLMEngine，避免 CLI 解析阶段提前加载运行时。
    _, _, _, LLMEngine = _resolve_runtime_symbols()

    # 根据 CLI 参数构建引擎配置。
    engine_args = _build_engine_args(args)
    # 根据 CLI 参数构建采样配置。
    sampling_params = _build_sampling_params(args)
    # 创建底层 v1 引擎实例。
    engine = LLMEngine.from_engine_args(
        engine_args,
        enable_multiprocessing=args.enable_multiprocessing,
    )
    # 为这次 one-shot 请求固定一个 request_id。
    request_id = "native-generate"

    try:
        # 把请求提交给引擎。
        engine.add_request(request_id, _render_engine_prompt(engine, args),
                           sampling_params)

        # 只要引擎里还有未完成请求，就持续拉取输出。
        while engine.has_unfinished_requests():
            # 推进一次引擎 step。
            outputs = engine.step()
            # 打印属于当前请求的文本增量。
            for text in _iter_request_text(outputs, request_id):
                print(text, end="", flush=True)
        # 最后一轮输出结束后补一个换行。
        print()
        # 正常执行完成返回 0。
        return 0
    finally:
        try:
            # 无论成功失败都尽量关闭 engine core。
            engine.engine_core.shutdown()
        except Exception:
            # 若关闭失败，则记录异常便于排查资源泄漏。
            logger.exception("failed to shutdown native CFIE engine cleanly")
