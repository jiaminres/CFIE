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


def add_native_generate_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "native-generate",
        help="Run one native CFIE/v1 generation request",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--log-level", default="INFO",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--download-dir", default=None)
    parser.add_argument("--load-format", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enable-prefix-caching",
                        action=argparse.BooleanOptionalAction,
                        default=None)
    parser.add_argument("--enable-chunked-prefill",
                        action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--moe-cpu-budget-gb", type=float, default=0.0)
    parser.add_argument("--moe-cpu-min-free-gb", type=float, default=0.0)
    parser.add_argument("--cpu-offload-gb", type=float, default=0.0)
    parser.add_argument("--offload-backend", default="auto")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=None)
    parser.add_argument("--gpu-slots-per-layer", type=int, default=0)
    parser.add_argument("--prefill-burst-slots", type=int, default=0)
    parser.add_argument("--prepare-cpu-copy-batch-size", type=int, default=8)
    parser.add_argument("--cpu-static-preprocess-batch-size",
                        type=int,
                        default=0)
    parser.add_argument("--cpu-static-pinned-gb", type=float, default=0.0)
    parser.add_argument("--cpu-static-pinned-layers", default="")
    parser.add_argument("--spec-method", choices=("none", "mtp"),
                        default="none")
    parser.add_argument("--num-speculative-tokens", type=int, default=None)
    parser.add_argument("--attention-backend", default=None)
    parser.add_argument("--moe-backend", default="auto")
    parser.add_argument("--marlin-input-dtype",
                        choices=("auto", "int8", "fp8"),
                        default="auto")
    parser.add_argument("--allow-tiered-moe-compile",
                        action=argparse.BooleanOptionalAction,
                        default=None)
    parser.add_argument("--mamba-cache-mode", default=None)
    parser.add_argument("--language-model-only",
                        action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--skip-mm-profiling",
                        action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--presence-penalty", type=float, default=None)
    parser.add_argument("--frequency-penalty", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--repetition-detection-max-pattern-size",
                        type=int,
                        default=0)
    parser.add_argument("--repetition-detection-min-pattern-size",
                        type=int,
                        default=4)
    parser.add_argument("--repetition-detection-min-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--stop", action="append", default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--enable-multiprocessing", action="store_true")
    parser.add_argument("--log-stats", action="store_true")
    parser.set_defaults(handler=run_native_generate)


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


def _build_speculative_config(args: Namespace) -> dict[str, Any] | None:
    if args.spec_method == "none":
        return None

    config: dict[str, Any] = {"method": args.spec_method}
    if args.num_speculative_tokens is not None:
        config["num_speculative_tokens"] = args.num_speculative_tokens
    return config


def _resolve_dtype(args: Namespace) -> str:
    dtype_aliases = {
        "fp16": "float16",
        "bf16": "bfloat16",
    }
    dtype = dtype_aliases.get(args.dtype, args.dtype)

    if args.quantization in {"gptq", "gptq_marlin"} and dtype == "auto":
        logger.info(
            "forcing dtype=float16 for quantization=%s",
            args.quantization,
        )
        return "float16"

    return dtype


def _build_engine_args(args: Namespace):
    EngineArgs, _, _, _ = _resolve_runtime_symbols()

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
        kv_cache_memory_bytes=getattr(args, "kv_cache_memory_bytes", None),
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=args.enable_prefix_caching,
        enable_chunked_prefill=getattr(args, "enable_chunked_prefill", True),
        moe_cpu_budget_gb=args.moe_cpu_budget_gb,
        moe_cpu_min_free_gb=args.moe_cpu_min_free_gb,
        cpu_offload_gb=args.cpu_offload_gb,
        offload_backend=args.offload_backend,
        gpu_slots_per_layer=getattr(args, "gpu_slots_per_layer", 0),
        prefill_burst_slots=getattr(args, "prefill_burst_slots", 0),
        prepare_cpu_copy_batch_size=getattr(
            args, "prepare_cpu_copy_batch_size", 8
        ),
        cpu_static_preprocess_batch_size=getattr(
            args, "cpu_static_preprocess_batch_size", 0
        ),
        cpu_static_pinned_gb=getattr(args, "cpu_static_pinned_gb", 0.0),
        cpu_static_pinned_layers=getattr(args, "cpu_static_pinned_layers", ""),
        enforce_eager=args.enforce_eager,
        attention_backend=args.attention_backend,
        moe_backend=args.moe_backend,
        marlin_input_dtype=getattr(args, "marlin_input_dtype", "auto"),
        allow_tiered_moe_compile=getattr(
            args, "allow_tiered_moe_compile", None
        ),
        language_model_only=args.language_model_only,
        skip_mm_profiling=args.skip_mm_profiling,
        disable_log_stats=not args.log_stats,
        speculative_config=_build_speculative_config(args),
    )
    if args.mamba_cache_mode is not None:
        engine_kwargs["mamba_cache_mode"] = args.mamba_cache_mode
    return EngineArgs(**engine_kwargs)


def _build_sampling_params(args: Namespace):
    _, SamplingParams, RequestOutputKind, _ = _resolve_runtime_symbols()

    stop: str | list[str] | None
    if not args.stop:
        stop = None
    elif len(args.stop) == 1:
        stop = args.stop[0]
    else:
        stop = list(args.stop)

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


def _iter_request_text(outputs: list[Any], request_id: str):
    for output in outputs:
        if getattr(output, "request_id", None) != request_id:
            continue
        for completion in getattr(output, "outputs", []) or []:
            text = getattr(completion, "text", "")
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


def run_native_generate(args: Namespace) -> int:
    _, _, _, LLMEngine = _resolve_runtime_symbols()

    engine_args = _build_engine_args(args)
    sampling_params = _build_sampling_params(args)
    engine = LLMEngine.from_engine_args(
        engine_args,
        enable_multiprocessing=args.enable_multiprocessing,
    )
    request_id = "native-generate"

    try:
        engine.add_request(
            request_id,
            _render_engine_prompt(engine, args),
            sampling_params,
        )

        while engine.has_unfinished_requests():
            outputs = engine.step()
            for text in _iter_request_text(outputs, request_id):
                print(text, end="", flush=True)
        print()
        return 0
    finally:
        try:
            engine.engine_core.shutdown()
        except Exception:
            logger.exception("failed to shutdown native CFIE engine cleanly")
