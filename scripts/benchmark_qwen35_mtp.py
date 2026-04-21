#!/usr/bin/env python3
"""Benchmark Qwen3.5 MTP decoding on CFIE's native v1 engine."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cfie.engine.arg_utils import EngineArgs
from cfie.inputs.data import token_inputs
from cfie.outputs import RequestOutput
from cfie.sampling_params import RequestOutputKind, SamplingParams
from cfie.v1.engine.llm_engine import LLMEngine


@dataclass
class BenchmarkResult:
    spec_tokens: int
    prompt_tokens: int
    max_model_len: int
    generated_tokens: int
    elapsed_seconds: float
    milliseconds_per_token: float
    tokens_per_second: float
    output_preview: str


@dataclass
class ContextProbeResult:
    spec_tokens: int
    requested_prompt_tokens: int
    accepted_prompt_tokens: int | None
    max_model_len: int
    generated_tokens: int
    elapsed_seconds: float
    success: bool
    error: str | None
    output_preview: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Qwen3.5 MoE MTP generation speed on CFIE.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--quantization", default="gptq_marlin")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-model-len", type=int, default=-1)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--gen-lengths", type=int, nargs="+", default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.96)
    parser.add_argument("--cpu-offload-gb", type=float, default=0.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--spec-tokens", type=int, nargs="+", default=[1])
    parser.add_argument("--prompt", default="请用中文简要介绍低显存客户端推理的关键优化点。")
    parser.add_argument("--prompt-token-count", type=int, default=None)
    parser.add_argument("--warmup-new-tokens", type=int, default=8)
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enforce-eager", action="store_true", default=True)
    parser.add_argument(
        "--no-enforce-eager",
        action="store_false",
        dest="enforce_eager",
    )
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def _resolve_dtype(dtype: str, quantization: str | None) -> str:
    dtype_aliases = {
        "fp16": "float16",
        "bf16": "bfloat16",
    }
    resolved = dtype_aliases.get(dtype, dtype)
    if quantization in {"gptq", "gptq_marlin"} and resolved == "auto":
        return "float16"
    return resolved


def _build_engine_args(args: argparse.Namespace, spec_tokens: int) -> EngineArgs:
    return EngineArgs(
        model=args.model,
        tokenizer=args.tokenizer,
        trust_remote_code=False,
        dtype=_resolve_dtype(args.dtype, args.quantization),
        quantization=args.quantization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        cpu_offload_gb=args.cpu_offload_gb,
        disable_log_stats=True,
        enforce_eager=args.enforce_eager,
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": spec_tokens,
        },
    )


def _build_sampling_params(args: argparse.Namespace) -> SamplingParams:
    max_new_tokens = args.max_new_tokens
    if args.gen_lengths:
        max_new_tokens = int(args.gen_lengths[0])
    return SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        seed=args.seed,
        ignore_eos=True,
        min_tokens=max_new_tokens,
        max_tokens=max_new_tokens,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )


def _extract_generated_tokens(output: RequestOutput) -> int:
    if not output.outputs:
        return 0
    return len(output.outputs[0].token_ids)


def _extract_output_text(output: RequestOutput) -> str:
    if not output.outputs:
        return ""
    return output.outputs[0].text


def _resolve_generation_lengths(args: argparse.Namespace) -> list[int]:
    if args.gen_lengths:
        return [int(length) for length in args.gen_lengths]
    return [int(args.max_new_tokens)]


def _load_hf_tokenizer(model: str, tokenizer: str | None):
    from transformers import AutoTokenizer

    tokenizer_path = tokenizer or model
    return AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=False,
    )


def _resolve_synthetic_token_id(hf_tokenizer) -> int:
    for probe in ("a", "hello", "你"):
        token_ids = hf_tokenizer.encode(probe, add_special_tokens=False)
        if token_ids:
            return int(token_ids[0])
    raise RuntimeError("failed to derive a synthetic token id from the tokenizer")


def _build_prompt_input(
    args: argparse.Namespace,
    hf_tokenizer,
    synthetic_token_id: int | None,
):
    if args.prompt_token_count is None:
        prompt_token_ids = hf_tokenizer.encode(args.prompt, add_special_tokens=False)
        return (
            token_inputs(prompt_token_ids=prompt_token_ids, prompt=args.prompt),
            len(prompt_token_ids),
        )
    if synthetic_token_id is None:
        raise ValueError("synthetic_token_id must be provided for token prompts")
    prompt_token_ids = [synthetic_token_id] * int(args.prompt_token_count)
    return (
        token_inputs(
            prompt_token_ids=prompt_token_ids,
            prompt=f"[synthetic_prompt_tokens={args.prompt_token_count}]",
        ),
        len(prompt_token_ids),
    )


def _run_request(
    engine: LLMEngine,
    request_id: str,
    prompt_input: Any,
    sampling_params: SamplingParams,
) -> RequestOutput:
    final_output: RequestOutput | None = None
    engine.add_request(request_id, prompt_input, sampling_params)
    while engine.has_unfinished_requests():
        outputs = engine.step()
        for output in outputs:
            if (
                isinstance(output, RequestOutput)
                and output.request_id == request_id
                and output.finished
            ):
                final_output = output
    if final_output is None:
        raise RuntimeError(f"request {request_id} finished without a final RequestOutput")
    return final_output


def _maybe_warmup(
    engine: LLMEngine,
    args: argparse.Namespace,
    hf_tokenizer,
) -> None:
    if args.skip_warmup or args.warmup_new_tokens <= 0:
        return
    warmup_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        seed=args.seed,
        ignore_eos=True,
        min_tokens=args.warmup_new_tokens,
        max_tokens=args.warmup_new_tokens,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )
    _run_request(
        engine,
        request_id="warmup",
        prompt_input=token_inputs(
            prompt_token_ids=hf_tokenizer.encode("你好", add_special_tokens=False),
            prompt="你好",
        ),
        sampling_params=warmup_params,
    )


def _run_single_benchmark(
    args: argparse.Namespace,
    spec_tokens: int,
) -> list[BenchmarkResult]:
    engine = LLMEngine.from_engine_args(_build_engine_args(args, spec_tokens))
    hf_tokenizer = _load_hf_tokenizer(args.model, args.tokenizer)
    synthetic_token_id = None
    if args.prompt_token_count is not None:
        synthetic_token_id = _resolve_synthetic_token_id(hf_tokenizer)
    prompt_input, requested_prompt_tokens = _build_prompt_input(
        args,
        hf_tokenizer,
        synthetic_token_id,
    )
    prompt_tokens = requested_prompt_tokens or 0
    results: list[BenchmarkResult] = []

    try:
        _maybe_warmup(engine, args, hf_tokenizer)
        for gen_length in _resolve_generation_lengths(args):
            sampling_params = SamplingParams(
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                seed=args.seed,
                ignore_eos=True,
                min_tokens=gen_length,
                max_tokens=gen_length,
                output_kind=RequestOutputKind.FINAL_ONLY,
            )
            request_id = f"benchmark-{spec_tokens}-{gen_length}"
            start = time.perf_counter()
            final_output = _run_request(
                engine,
                request_id=request_id,
                prompt_input=prompt_input,
                sampling_params=sampling_params,
            )
            elapsed = time.perf_counter() - start
            generated_tokens = _extract_generated_tokens(final_output)
            if generated_tokens <= 0:
                raise RuntimeError(f"request {request_id} generated no tokens")
            if prompt_tokens == 0 and final_output.prompt_token_ids is not None:
                prompt_tokens = len(final_output.prompt_token_ids)
            results.append(
                BenchmarkResult(
                    spec_tokens=spec_tokens,
                    prompt_tokens=prompt_tokens,
                    max_model_len=engine.cfie_config.model_config.max_model_len,
                    generated_tokens=generated_tokens,
                    elapsed_seconds=elapsed,
                    milliseconds_per_token=(elapsed / generated_tokens) * 1000.0,
                    tokens_per_second=generated_tokens / elapsed,
                    output_preview=_extract_output_text(final_output)[:200],
                )
            )
        return results
    finally:
        engine.engine_core.shutdown()


def _run_context_probe(
    args: argparse.Namespace,
    spec_tokens: int,
) -> ContextProbeResult:
    if args.prompt_token_count is None:
        raise ValueError("--prompt-token-count is required for context probing")

    engine = LLMEngine.from_engine_args(_build_engine_args(args, spec_tokens))
    hf_tokenizer = _load_hf_tokenizer(args.model, args.tokenizer)
    synthetic_token_id = _resolve_synthetic_token_id(hf_tokenizer)
    prompt_input, requested_prompt_tokens = _build_prompt_input(
        args,
        hf_tokenizer,
        synthetic_token_id,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        seed=args.seed,
        ignore_eos=True,
        min_tokens=1,
        max_tokens=1,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )

    try:
        start = time.perf_counter()
        final_output = _run_request(
            engine,
            request_id=f"context-probe-{spec_tokens}",
            prompt_input=prompt_input,
            sampling_params=sampling_params,
        )
        elapsed = time.perf_counter() - start
        accepted_prompt_tokens = (
            len(final_output.prompt_token_ids)
            if final_output.prompt_token_ids is not None
            else None
        )
        return ContextProbeResult(
            spec_tokens=spec_tokens,
            requested_prompt_tokens=int(requested_prompt_tokens or 0),
            accepted_prompt_tokens=accepted_prompt_tokens,
            max_model_len=engine.cfie_config.model_config.max_model_len,
            generated_tokens=_extract_generated_tokens(final_output),
            elapsed_seconds=elapsed,
            success=True,
            error=None,
            output_preview=_extract_output_text(final_output)[:200],
        )
    except Exception as exc:
        return ContextProbeResult(
            spec_tokens=spec_tokens,
            requested_prompt_tokens=int(requested_prompt_tokens or 0),
            accepted_prompt_tokens=None,
            max_model_len=engine.cfie_config.model_config.max_model_len,
            generated_tokens=0,
            elapsed_seconds=0.0,
            success=False,
            error=str(exc),
            output_preview="",
        )
    finally:
        engine.engine_core.shutdown()


def _print_summary(results: list[BenchmarkResult]) -> None:
    print(
        "spec_tokens,prompt_tokens,max_model_len,generated_tokens,elapsed_seconds,ms_per_token,tokens_per_second"
    )
    for result in results:
        print(
            f"{result.spec_tokens},"
            f"{result.prompt_tokens},"
            f"{result.max_model_len},"
            f"{result.generated_tokens},"
            f"{result.elapsed_seconds:.6f},"
            f"{result.milliseconds_per_token:.3f},"
            f"{result.tokens_per_second:.3f}"
        )


def _print_context_probe(results: list[ContextProbeResult]) -> None:
    print(
        "spec_tokens,requested_prompt_tokens,accepted_prompt_tokens,max_model_len,generated_tokens,elapsed_seconds,success,error"
    )
    for result in results:
        print(
            f"{result.spec_tokens},"
            f"{result.requested_prompt_tokens},"
            f"{result.accepted_prompt_tokens},"
            f"{result.max_model_len},"
            f"{result.generated_tokens},"
            f"{result.elapsed_seconds:.6f},"
            f"{result.success},"
            f"{json.dumps(result.error, ensure_ascii=False)}"
        )


def main() -> int:
    args = _parse_args()
    if args.prompt_token_count is not None and _resolve_generation_lengths(args) == [1]:
        results = [_run_context_probe(args, spec_tokens) for spec_tokens in args.spec_tokens]
        _print_context_probe(results)
        if args.output_json:
            output_path = Path(args.output_json)
            output_path.write_text(
                json.dumps([asdict(result) for result in results], indent=2),
                encoding="utf-8",
            )
        return 0

    results: list[BenchmarkResult] = []
    for spec_tokens in args.spec_tokens:
        results.extend(_run_single_benchmark(args, spec_tokens))
    _print_summary(results)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.write_text(
            json.dumps([asdict(result) for result in results], indent=2),
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
