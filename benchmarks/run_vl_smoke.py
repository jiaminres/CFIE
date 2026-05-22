from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image, ImageDraw

from cfie.engine.arg_utils import EngineArgs
from cfie.sampling_params import RequestOutputKind, SamplingParams
from cfie.v1.engine.llm_engine import LLMEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen3.5 VL smoke benchmark")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="请用一句话描述这张图片。")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2112)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--gpu-slots-per-layer", type=int, default=16)
    parser.add_argument("--prefill-burst-slots", type=int, default=0)
    parser.add_argument("--prepare-cpu-copy-threads", type=int, default=32)
    parser.add_argument("--prepare-cpu-copy-batch-size", type=int, default=0)
    parser.add_argument("--cpu-static-pinned-gb", type=float, default=0.0)
    parser.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable-chunked-prefill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument(
        "--marlin-input-dtype",
        default="auto",
        choices=("auto", "int8", "fp8"),
        help="Optional Marlin activation dtype override.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--result-json", default=None)
    return parser


def make_test_image() -> Image.Image:
    image = Image.new("RGB", (224, 224), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((32, 48, 192, 176), outline="black", width=4)
    draw.ellipse((72, 72, 152, 152), fill="red", outline="black", width=3)
    draw.text((54, 184), "CFIE", fill="black")
    return image


def make_qwen_vl_prompt(user_text: str) -> str:
    return (
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>"
        f"{user_text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def main() -> None:
    args = build_parser().parse_args()

    engine_args = EngineArgs(
        model=args.model,
        tokenizer=None,
        trust_remote_code=False,
        dtype="auto",
        quantization=None,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=args.enable_prefix_caching,
        enable_chunked_prefill=args.enable_chunked_prefill,
        enforce_eager=args.enforce_eager,
        language_model_only=False,
        skip_mm_profiling=False,
        disable_log_stats=True,
        limit_mm_per_prompt={"image": 1, "video": 0},
        marlin_input_dtype=args.marlin_input_dtype,
    )
    engine_args.kv_cache_memory_bytes = args.kv_cache_memory_bytes
    engine_args.gpu_slots_per_layer = args.gpu_slots_per_layer
    engine_args.prefill_burst_slots = args.prefill_burst_slots
    engine_args.prepare_cpu_copy_threads = (
        args.prepare_cpu_copy_batch_size or args.prepare_cpu_copy_threads
    )
    engine_args.prepare_cpu_copy_batch_size = args.prepare_cpu_copy_batch_size
    engine_args.cpu_static_pinned_gb = args.cpu_static_pinned_gb

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        max_tokens=args.max_new_tokens,
        output_kind=RequestOutputKind.DELTA,
    )

    engine = LLMEngine.from_engine_args(engine_args, enable_multiprocessing=False)
    request_id = "vl-smoke"
    generated_parts: list[str] = []
    prompt = {
        "prompt": make_qwen_vl_prompt(args.prompt),
        "multi_modal_data": {"image": make_test_image()},
    }

    try:
        engine.add_request(request_id, prompt, sampling_params)
        t0 = time.perf_counter()
        steps = 0
        while engine.has_unfinished_requests():
            outputs = engine.step()
            steps += 1
            for output in outputs:
                if getattr(output, "request_id", None) != request_id:
                    continue
                for completion in getattr(output, "outputs", []) or []:
                    text = getattr(completion, "text", "")
                    if text:
                        generated_parts.append(text)
        total_seconds = time.perf_counter() - t0
    finally:
        try:
            engine.engine_core.shutdown()
        except Exception:
            pass

    scheduler_config = engine.cfie_config.scheduler_config
    result = {
        "total_seconds": total_seconds,
        "steps": steps,
        "generated_text": "".join(generated_parts),
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
        "gpu_slots_per_layer": args.gpu_slots_per_layer,
        "prefill_burst_slots": args.prefill_burst_slots,
        "enable_chunked_prefill": args.enable_chunked_prefill,
        "effective_max_num_batched_tokens": scheduler_config.max_num_batched_tokens,
        "effective_max_num_scheduled_tokens": (
            scheduler_config.max_num_scheduled_tokens
        ),
        "cpu_static_pinned_gb": args.cpu_static_pinned_gb,
        "marlin_input_dtype": args.marlin_input_dtype,
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
