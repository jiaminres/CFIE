from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from transformers import AutoTokenizer

from cfie.config import CompilationMode, CUDAGraphMode
from cfie.cli.native_generate import (
    _build_engine_args,
    _build_sampling_params,
    _resolve_runtime_symbols,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure long prefill + decode latency")
    parser.add_argument("--model", required=True)
    parser.add_argument("--target-input-tokens", type=int, default=50000)
    parser.add_argument("--base-text", default="这是一个用于长上下文prefill测试的中文句子。")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--load-format", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--download-dir", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--moe-cpu-budget-gb", type=float, default=0.0)
    parser.add_argument("--moe-cpu-min-free-gb", type=float, default=0.0)
    parser.add_argument("--cpu-offload-gb", type=float, default=0.0)
    parser.add_argument("--offload-backend", default="auto")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=None)
    parser.add_argument("--gpu-slots-per-layer", type=int, default=24)
    parser.add_argument("--prefill-burst-slots", type=int, default=256)
    parser.add_argument("--prepare-cpu-copy-batch-size", type=int, default=8)
    parser.add_argument("--cpu-static-pinned-gb", type=float, default=0.0)
    parser.add_argument("--cpu-static-pinned-layers", default="")
    parser.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable-chunked-prefill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--spec-method", default="none", choices=("none", "mtp"))
    parser.add_argument("--num-speculative-tokens", type=int, default=None)
    parser.add_argument("--attention-backend", default=None)
    parser.add_argument("--moe-backend", default="auto")
    parser.add_argument(
        "--marlin-input-dtype",
        default="auto",
        choices=("auto", "int8", "fp8"),
        help="Optional Marlin activation dtype override.",
    )
    parser.add_argument("--mamba-cache-mode", default=None)
    parser.add_argument("--language-model-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-mm-profiling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--presence-penalty", type=float, default=None)
    parser.add_argument("--frequency-penalty", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--stop", action="append", default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--piecewise-cudagraph", action="store_true")
    parser.add_argument("--cudagraph-capture-sizes", type=int, nargs="+", default=None)
    parser.add_argument(
        "--cudagraph-decode-capture-sizes",
        type=int,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--cudagraph-prefill-capture-sizes",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Prefill/mixed-batch CUDA graph capture sizes. Pass the flag "
            "with no values to disable prefill capture while keeping decode "
            "capture sizes."
        ),
    )
    parser.add_argument(
        "--cudagraph-copy-inputs",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--enable-multiprocessing", action="store_true")
    parser.add_argument("--log-stats", action="store_true")
    parser.add_argument("--ignore-eos", action="store_true", default=True)
    parser.add_argument("--result-json", default=None)
    return parser


def build_prompt_token_ids(args: argparse.Namespace) -> list[int]:
    tokenizer_path = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=False,
        revision=args.revision,
    )
    base_ids = tokenizer.encode(args.base_text, add_special_tokens=False)
    if not base_ids:
        raise ValueError("base text produced no tokens")
    repeat = args.target_input_tokens // len(base_ids) + 1
    token_ids = (base_ids * repeat)[: args.target_input_tokens]
    return token_ids


def main() -> None:
    args = build_parser().parse_args()
    if args.target_input_tokens + args.max_new_tokens > args.max_model_len:
        raise ValueError(
            "target input tokens plus max new tokens must fit max_model_len"
        )

    EngineArgs, SamplingParams, RequestOutputKind, LLMEngine = _resolve_runtime_symbols()
    engine_args = _build_engine_args(args)
    engine_args.kv_cache_memory_bytes = args.kv_cache_memory_bytes
    engine_args.gpu_slots_per_layer = args.gpu_slots_per_layer
    engine_args.prefill_burst_slots = args.prefill_burst_slots
    if args.piecewise_cudagraph:
        engine_args.compilation_config.mode = CompilationMode.VLLM_COMPILE
        engine_args.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
        engine_args.compilation_config.allow_tiered_moe_compile = True
    if args.cudagraph_capture_sizes is not None:
        capture_sizes = sorted(set(int(size) for size in args.cudagraph_capture_sizes))
        engine_args.compilation_config.cudagraph_capture_sizes = capture_sizes
    if args.cudagraph_decode_capture_sizes is not None:
        capture_sizes = sorted(
            set(int(size) for size in args.cudagraph_decode_capture_sizes)
        )
        engine_args.compilation_config.cudagraph_decode_capture_sizes = capture_sizes
    if args.cudagraph_prefill_capture_sizes is not None:
        capture_sizes = sorted(
            set(int(size) for size in args.cudagraph_prefill_capture_sizes)
        )
        engine_args.compilation_config.cudagraph_prefill_capture_sizes = capture_sizes
    if args.cudagraph_copy_inputs is not None:
        engine_args.compilation_config.cudagraph_copy_inputs = (
            args.cudagraph_copy_inputs
        )
    engine_args.prepare_cpu_copy_batch_size = args.prepare_cpu_copy_batch_size
    engine_args.cpu_static_pinned_gb = args.cpu_static_pinned_gb
    engine_args.cpu_static_pinned_layers = args.cpu_static_pinned_layers
    engine_args.language_model_only = args.language_model_only
    engine_args.skip_mm_profiling = args.skip_mm_profiling

    sampling_params = _build_sampling_params(args)
    sampling_params.ignore_eos = True
    if hasattr(sampling_params, "_eos_token_id"):
        sampling_params._eos_token_id = None

    prompt_token_ids = build_prompt_token_ids(args)
    engine = LLMEngine.from_engine_args(
        engine_args,
        enable_multiprocessing=args.enable_multiprocessing,
    )
    request_id = "long-prefill"
    generated_parts: list[str] = []

    try:
        add_t0 = time.perf_counter()
        engine.add_request(
            request_id,
            {"prompt_token_ids": prompt_token_ids},
            sampling_params,
        )
        add_seconds = time.perf_counter() - add_t0

        run_t0 = time.perf_counter()
        step_count = 0
        first_step_seconds = 0.0
        generated_tokens = 0
        while engine.has_unfinished_requests():
            is_first_step = step_count == 0
            step_t0 = time.perf_counter()
            outputs = engine.step()
            step_dt = time.perf_counter() - step_t0
            if is_first_step:
                first_step_seconds = step_dt
            step_count += 1
            for output in outputs:
                if getattr(output, "request_id", None) != request_id:
                    continue
                for completion in getattr(output, "outputs", []) or []:
                    generated_tokens += len(completion.token_ids)
                    text = getattr(completion, "text", "")
                    if text:
                        generated_parts.append(text)
        run_seconds = time.perf_counter() - run_t0
    finally:
        try:
            engine.engine_core.shutdown()
        except Exception:
            pass

    decode_seconds = max(run_seconds - first_step_seconds, 0.0)
    decode_tokens = max(generated_tokens - 1, 0)
    scheduler_config = engine.cfie_config.scheduler_config
    result = {
        "input_tokens": len(prompt_token_ids),
        "decode_tokens": generated_tokens,
        "add_request_seconds": add_seconds,
        "run_seconds": run_seconds,
        "first_step_seconds": first_step_seconds,
        "decode_seconds_after_first_step": decode_seconds,
        "decode_tokens_after_first_step": decode_tokens,
        "decode_tokens_per_sec_after_first_step": (
            decode_tokens / decode_seconds if decode_seconds > 0 else 0.0
        ),
        "steps": step_count,
        "gpu_slots_per_layer": args.gpu_slots_per_layer,
        "prefill_burst_slots": args.prefill_burst_slots,
        "cpu_static_pinned_gb": args.cpu_static_pinned_gb,
        "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "enable_chunked_prefill": args.enable_chunked_prefill,
        "effective_max_num_batched_tokens": scheduler_config.max_num_batched_tokens,
        "effective_max_num_scheduled_tokens": (
            scheduler_config.max_num_scheduled_tokens
        ),
        "marlin_input_dtype": args.marlin_input_dtype,
        "piecewise_cudagraph": args.piecewise_cudagraph,
        "cudagraph_capture_sizes": args.cudagraph_capture_sizes,
        "cudagraph_decode_capture_sizes": args.cudagraph_decode_capture_sizes,
        "cudagraph_prefill_capture_sizes": args.cudagraph_prefill_capture_sizes,
        "cudagraph_copy_inputs": args.cudagraph_copy_inputs,
        "generated_text_prefix": "".join(generated_parts)[:500],
    }
    if args.result_json:
        result_path = Path(args.result_json)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
