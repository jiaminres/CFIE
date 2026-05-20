from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from cfie.config import CompilationMode, CUDAGraphMode
from cfie.cli.native_generate import (
    _build_engine_args,
    _build_sampling_params,
    _render_engine_prompt,
    _resolve_runtime_symbols,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure 512-token decode latency")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=4096)
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
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=None)
    parser.add_argument("--gpu-slots-per-layer", type=int, default=0)
    parser.add_argument("--prefill-burst-slots", type=int, default=0)
    parser.add_argument("--prepare-cpu-copy-batch-size", type=int, default=8)
    parser.add_argument("--cpu-static-pinned-gb", type=float, default=0.0)
    parser.add_argument("--cpu-static-pinned-layers", default="")
    parser.add_argument(
        "--enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--enable-chunked-prefill",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
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
    parser.add_argument("--language-model-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-mm-profiling", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--presence-penalty", type=float, default=None)
    parser.add_argument("--frequency-penalty", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
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


def _build_args_namespace(raw: argparse.Namespace) -> argparse.Namespace:
    # Reuse the native_generate helpers by providing the same field names.
    ns = argparse.Namespace(**vars(raw))
    if ns.quantization is None:
        ns.quantization = None
    if ns.max_num_batched_tokens is None:
        ns.max_num_batched_tokens = None
    return ns


def main() -> None:
    args = build_parser().parse_args()
    ns = _build_args_namespace(args)

    EngineArgs, SamplingParams, RequestOutputKind, LLMEngine = _resolve_runtime_symbols()
    engine_args = _build_engine_args(ns)
    engine_args.kv_cache_memory_bytes = ns.kv_cache_memory_bytes
    engine_args.gpu_slots_per_layer = ns.gpu_slots_per_layer
    engine_args.prefill_burst_slots = ns.prefill_burst_slots
    if ns.piecewise_cudagraph:
        engine_args.compilation_config.mode = CompilationMode.VLLM_COMPILE
        engine_args.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
        engine_args.compilation_config.allow_tiered_moe_compile = True
    if ns.cudagraph_capture_sizes is not None:
        capture_sizes = sorted(set(int(size) for size in ns.cudagraph_capture_sizes))
        engine_args.compilation_config.cudagraph_capture_sizes = capture_sizes
    if ns.cudagraph_decode_capture_sizes is not None:
        capture_sizes = sorted(
            set(int(size) for size in ns.cudagraph_decode_capture_sizes)
        )
        engine_args.compilation_config.cudagraph_decode_capture_sizes = capture_sizes
    if ns.cudagraph_prefill_capture_sizes is not None:
        capture_sizes = sorted(
            set(int(size) for size in ns.cudagraph_prefill_capture_sizes)
        )
        engine_args.compilation_config.cudagraph_prefill_capture_sizes = capture_sizes
    if ns.cudagraph_copy_inputs is not None:
        engine_args.compilation_config.cudagraph_copy_inputs = ns.cudagraph_copy_inputs
    engine_args.prepare_cpu_copy_batch_size = ns.prepare_cpu_copy_batch_size
    engine_args.cpu_static_pinned_gb = ns.cpu_static_pinned_gb
    engine_args.cpu_static_pinned_layers = ns.cpu_static_pinned_layers
    engine_args.language_model_only = ns.language_model_only
    engine_args.skip_mm_profiling = ns.skip_mm_profiling
    sampling_params = _build_sampling_params(ns)
    sampling_params.ignore_eos = True
    if hasattr(sampling_params, "_eos_token_id"):
        sampling_params._eos_token_id = None

    engine = LLMEngine.from_engine_args(
        engine_args,
        enable_multiprocessing=ns.enable_multiprocessing,
    )
    request_id = "decode512"

    try:
        prompt = _render_engine_prompt(engine, ns)
        engine.add_request(request_id, prompt, sampling_params)

        decode_t0 = time.perf_counter()
        step_count = 0
        first_step_seconds = 0.0
        first_step_tokens = 0
        generated_tokens = 0
        generated_text_parts: list[str] = []
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
                if getattr(output, "outputs", None):
                    for completion in output.outputs:
                        token_count = len(completion.token_ids)
                        generated_tokens += token_count
                        if is_first_step:
                            first_step_tokens += token_count
                        text = getattr(completion, "text", "")
                        if text:
                            generated_text_parts.append(text)
        total_seconds = time.perf_counter() - decode_t0
    finally:
        try:
            engine.engine_core.shutdown()
        except Exception:
            pass

    avg_ms_total = (total_seconds / max(generated_tokens, 1)) * 1000.0
    tps = generated_tokens / total_seconds if total_seconds > 0 else 0.0
    steady_tokens = max(generated_tokens - first_step_tokens, 0)
    steady_seconds = max(total_seconds - first_step_seconds, 0.0)
    steady_tps = steady_tokens / steady_seconds if steady_seconds > 0 else 0.0
    scheduler_config = engine.cfie_config.scheduler_config
    result = {
        "decode_tokens": generated_tokens,
        "total_seconds": total_seconds,
        "avg_ms_per_token_total": avg_ms_total,
        "tokens_per_sec": tps,
        "first_step_seconds": first_step_seconds,
        "first_step_tokens": first_step_tokens,
        "steady_tokens": steady_tokens,
        "steady_seconds": steady_seconds,
        "steady_tokens_per_sec": steady_tps,
        "steps": step_count,
        "gpu_slots_per_layer": ns.gpu_slots_per_layer,
        "prefill_burst_slots": ns.prefill_burst_slots,
        "prepare_cpu_copy_batch_size": ns.prepare_cpu_copy_batch_size,
        "cpu_static_pinned_gb": ns.cpu_static_pinned_gb,
        "cpu_static_pinned_layers": ns.cpu_static_pinned_layers,
        "kv_cache_memory_bytes": ns.kv_cache_memory_bytes,
        "max_model_len": ns.max_model_len,
        "max_num_batched_tokens": ns.max_num_batched_tokens,
        "enable_chunked_prefill": ns.enable_chunked_prefill,
        "effective_max_num_batched_tokens": scheduler_config.max_num_batched_tokens,
        "effective_max_num_scheduled_tokens": (
            scheduler_config.max_num_scheduled_tokens
        ),
        "max_new_tokens": ns.max_new_tokens,
        "marlin_input_dtype": ns.marlin_input_dtype,
        "piecewise_cudagraph": ns.piecewise_cudagraph,
        "cudagraph_capture_sizes": ns.cudagraph_capture_sizes,
        "cudagraph_decode_capture_sizes": ns.cudagraph_decode_capture_sizes,
        "cudagraph_prefill_capture_sizes": ns.cudagraph_prefill_capture_sizes,
        "cudagraph_copy_inputs": ns.cudagraph_copy_inputs,
        "generated_text_prefix": "".join(generated_text_parts)[:500],
    }
    if ns.result_json:
        result_path = Path(ns.result_json)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(
        f"decode_tokens={generated_tokens} total_seconds={total_seconds:.6f} "
        f"avg_ms_per_token_total={avg_ms_total:.3f} "
        f"tokens_per_sec={tps:.3f} "
        f"steady_tokens_per_sec={steady_tps:.3f} "
        f"first_step_seconds={first_step_seconds:.6f} "
        f"steps={step_count}"
    )


if __name__ == "__main__":
    main()
