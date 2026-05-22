from __future__ import annotations

import argparse
import json
import os
import sys
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
    parser.add_argument(
        "--target-input-tokens-list",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional list of prompt lengths to run sequentially in one engine. "
            "When omitted, --target-input-tokens is used."
        ),
    )
    parser.add_argument(
        "--base-text",
        default="This is a deterministic long-context prefill benchmark sentence. ",
    )
    parser.add_argument(
        "--isolate-prompt-lengths",
        action="store_true",
        help=(
            "Add a deterministic length-specific marker to each prompt length. "
            "This prevents sequential target-input-token runs from sharing a "
            "prefix-cache hit across different prompt lengths."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--turns", type=int, default=1)
    parser.add_argument(
        "--warm-prompt-mode",
        default="rotated",
        choices=("rotated", "same"),
        help=(
            "How to build prompts after the first request. 'rotated' keeps "
            "the same token count but avoids an exact full-prefix-cache hit; "
            "'same' measures the repeated-prompt/prefix-cache path."
        ),
    )
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
    parser.add_argument("--prepare-cpu-copy-threads", type=int, default=32)
    parser.add_argument("--prepare-cpu-copy-batch-size", type=int, default=0)
    parser.add_argument("--cpu-static-pinned-gb", type=float, default=0.0)
    parser.add_argument("--cpu-static-pinned-layers", default="")
    parser.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=True)
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
    parser.add_argument(
        "--force-exit-after-result",
        action="store_true",
        help=(
            "Exit with os._exit(0) after writing the result. This is useful "
            "for WSL benchmark sweeps where multiprocessing resource cleanup "
            "can hang after engine shutdown."
        ),
    )
    return parser


def build_prompt_token_ids(
    args: argparse.Namespace, target_input_tokens: int | None = None
) -> list[int]:
    tokenizer_path = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=False,
        revision=args.revision,
    )
    target_input_tokens = target_input_tokens or args.target_input_tokens
    base_text = args.base_text
    if args.isolate_prompt_lengths:
        base_text = f"[prefill-length={target_input_tokens}] {base_text}"
    base_ids = tokenizer.encode(base_text, add_special_tokens=False)
    if not base_ids:
        raise ValueError("base text produced no tokens")
    repeat = target_input_tokens // len(base_ids) + 1
    token_ids = (base_ids * repeat)[:target_input_tokens]
    return token_ids


def build_turn_prompt_token_ids(
    base_token_ids: list[int], turn_index: int, mode: str
) -> list[int]:
    if turn_index <= 0 or mode == "same" or len(base_token_ids) <= 1:
        return list(base_token_ids)
    offset = turn_index % len(base_token_ids)
    return base_token_ids[offset:] + base_token_ids[:offset]


def run_request(
    engine,
    request_id: str,
    prompt_token_ids: list[int],
    sampling_params,
) -> dict:
    generated_parts: list[str] = []
    add_t0 = time.perf_counter()
    engine.add_request(
        request_id,
        {"prompt_token_ids": prompt_token_ids},
        sampling_params,
    )
    add_seconds = time.perf_counter() - add_t0

    run_t0 = time.perf_counter()
    step_count = 0
    first_engine_step_seconds = 0.0
    time_to_first_output_seconds = 0.0
    first_output_step = 0
    first_output_tokens = 0
    generated_tokens = 0
    while engine.has_unfinished_requests():
        is_first_step = step_count == 0
        step_t0 = time.perf_counter()
        outputs = engine.step()
        step_dt = time.perf_counter() - step_t0
        if is_first_step:
            first_engine_step_seconds = step_dt
        step_count += 1
        step_generated_tokens = 0
        for output in outputs:
            if getattr(output, "request_id", None) != request_id:
                continue
            for completion in getattr(output, "outputs", []) or []:
                token_count = len(completion.token_ids)
                generated_tokens += token_count
                step_generated_tokens += token_count
                text = getattr(completion, "text", "")
                if text:
                    generated_parts.append(text)
        if step_generated_tokens and time_to_first_output_seconds <= 0.0:
            time_to_first_output_seconds = time.perf_counter() - run_t0
            first_output_step = step_count
            first_output_tokens = step_generated_tokens
    run_seconds = time.perf_counter() - run_t0

    decode_seconds = max(run_seconds - time_to_first_output_seconds, 0.0)
    decode_tokens = max(generated_tokens - first_output_tokens, 0)
    generated_text = "".join(generated_parts)
    return {
        "request_id": request_id,
        "input_tokens": len(prompt_token_ids),
        "decode_tokens": generated_tokens,
        "add_request_seconds": add_seconds,
        "run_seconds": run_seconds,
        "first_step_seconds": first_engine_step_seconds,
        "engine_first_step_seconds": first_engine_step_seconds,
        "time_to_first_output_seconds": time_to_first_output_seconds,
        "prefill_to_first_token_seconds": time_to_first_output_seconds,
        "first_output_step": first_output_step,
        "first_output_tokens": first_output_tokens,
        "decode_seconds_after_first_step": decode_seconds,
        "decode_tokens_after_first_step": decode_tokens,
        "decode_tokens_per_sec_after_first_step": (
            decode_tokens / decode_seconds if decode_seconds > 0 else 0.0
        ),
        "steps": step_count,
        "generated_text_prefix": generated_text[:500],
        "generated_text_chars": len(generated_text),
        "generated_text_nonempty": bool(generated_text.strip()),
        "generated_text_has_replacement_char": "\ufffd" in generated_text,
    }


def summarize_request_groups(request_results: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}
    for item in request_results:
        grouped.setdefault(str(item["input_tokens"]), []).append(item)
    summary: dict[str, dict] = {}
    for input_tokens, items in grouped.items():
        cold = items[0]
        warm_items = items[1:]
        summary[input_tokens] = {
            "cold_prefill_to_first_token_seconds": cold[
                "prefill_to_first_token_seconds"
            ],
            "cold_decode_tokens_per_sec_after_first_step": cold[
                "decode_tokens_per_sec_after_first_step"
            ],
            "warm_prefill_to_first_token_seconds": (
                warm_items[0]["prefill_to_first_token_seconds"]
                if warm_items
                else 0.0
            ),
            "warm_decode_tokens_per_sec_after_first_step": (
                warm_items[0]["decode_tokens_per_sec_after_first_step"]
                if warm_items
                else 0.0
            ),
            "warm_prefill_to_first_token_seconds_avg": (
                sum(item["prefill_to_first_token_seconds"] for item in warm_items)
                / len(warm_items)
                if warm_items
                else 0.0
            ),
            "all_outputs_nonempty": all(
                item["generated_text_nonempty"] for item in items
            ),
            "any_output_has_replacement_char": any(
                item["generated_text_has_replacement_char"] for item in items
            ),
            "requests": [item["request_id"] for item in items],
        }
    return summary


def main() -> None:
    args = build_parser().parse_args()
    if args.turns < 1:
        raise ValueError("--turns must be >= 1")
    target_input_tokens_list = (
        args.target_input_tokens_list
        if args.target_input_tokens_list is not None
        else [args.target_input_tokens]
    )
    if any(item < 1 for item in target_input_tokens_list):
        raise ValueError("target input token counts must be >= 1")
    too_long = [
        item
        for item in target_input_tokens_list
        if item + args.max_new_tokens > args.max_model_len
    ]
    if too_long:
        raise ValueError(
            "target input tokens plus max new tokens must fit max_model_len; "
            f"too_long={too_long} max_model_len={args.max_model_len}"
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
    engine_args.prepare_cpu_copy_threads = (
        args.prepare_cpu_copy_batch_size or args.prepare_cpu_copy_threads
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

    prompt_token_ids_by_len = {
        target_tokens: build_prompt_token_ids(args, target_tokens)
        for target_tokens in target_input_tokens_list
    }
    engine = LLMEngine.from_engine_args(
        engine_args,
        enable_multiprocessing=args.enable_multiprocessing,
    )
    request_results: list[dict] = []

    try:
        for target_tokens in target_input_tokens_list:
            prompt_token_ids = prompt_token_ids_by_len[target_tokens]
            for turn_index in range(args.turns):
                turn_prompt_ids = build_turn_prompt_token_ids(
                    prompt_token_ids, turn_index, args.warm_prompt_mode
                )
                request_results.append(
                    run_request(
                        engine,
                        f"long-prefill-{target_tokens}-turn-{turn_index + 1}",
                        turn_prompt_ids,
                        sampling_params,
                    )
                )
    finally:
        try:
            engine.engine_core.shutdown()
        except Exception:
            pass

    first_result = request_results[0]
    warm_results = request_results[1:]
    warm_prefill_avg = (
        sum(item["prefill_to_first_token_seconds"] for item in warm_results)
        / len(warm_results)
        if warm_results
        else 0.0
    )
    scheduler_config = engine.cfie_config.scheduler_config
    result = {
        **first_result,
        "turns": args.turns,
        "warm_prompt_mode": args.warm_prompt_mode,
        "isolate_prompt_lengths": args.isolate_prompt_lengths,
        "target_input_tokens_list": target_input_tokens_list,
        "request_results": request_results,
        "results_by_input_tokens": summarize_request_groups(request_results),
        "cold_prefill_to_first_token_seconds": first_result[
            "prefill_to_first_token_seconds"
        ],
        "warm_prefill_to_first_token_seconds": (
            warm_results[0]["prefill_to_first_token_seconds"]
            if warm_results
            else 0.0
        ),
        "warm_prefill_to_first_token_seconds_avg": warm_prefill_avg,
        "gpu_slots_per_layer": args.gpu_slots_per_layer,
        "prefill_burst_slots": args.prefill_burst_slots,
        "cpu_static_pinned_gb": args.cpu_static_pinned_gb,
        "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "enable_chunked_prefill": args.enable_chunked_prefill,
        "enable_prefix_caching": args.enable_prefix_caching,
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
    }
    if args.result_json:
        result_path = Path(args.result_json)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(result, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=True, indent=2))
    sys.stdout.flush()
    sys.stderr.flush()
    if args.force_exit_after_result:
        os._exit(0)


if __name__ == "__main__":
    main()
