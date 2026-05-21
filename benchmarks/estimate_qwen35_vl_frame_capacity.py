from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
from transformers import AutoProcessor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate Qwen3.5-VL multi-frame image token capacity."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-context-tokens", type=int, default=128000)
    parser.add_argument("--reserve-output-tokens", type=int, default=1024)
    parser.add_argument(
        "--resolutions",
        nargs="+",
        default=[
            "384x216",
            "512x288",
            "640x360",
            "800x450",
            "960x540",
            "1280x720",
            "1600x900",
            "1920x1080",
            "2560x1440",
        ],
    )
    parser.add_argument("--sample-frame-counts", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--prompt-text", default="Describe these GUI frames.")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    return parser


def parse_resolution(value: str) -> tuple[int, int]:
    try:
        width, height = value.lower().split("x", 1)
        return int(width), int(height)
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            f"resolution must look like WIDTHxHEIGHT, got {value!r}"
        ) from exc


def make_frame(width: int, height: int, index: int) -> Image.Image:
    image = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(image)
    top_h = max(32, height // 10)
    side_w = max(80, width // 5)
    draw.rectangle((0, 0, width, top_h), fill=(30, 41, 59))
    draw.rectangle((0, top_h, side_w, height), fill=(226, 232, 240))
    draw.rectangle(
        (side_w + 24, top_h + 24, width - 24, height - 24),
        fill="white",
        outline=(203, 213, 225),
        width=2,
    )
    button = (width - 170, height - 84, width - 44, height - 42)
    draw.rounded_rectangle(button, radius=6, fill=(37, 99, 235))
    draw.text((button[0] + 22, button[1] + 12), "Submit", fill="white")
    draw.text((side_w + 42, top_h + 48), f"Frame {index}", fill=(15, 23, 42))
    return image


def build_messages(images: list[Image.Image], prompt_text: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for image in images:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt_text})
    return [{"role": "user", "content": content}]


def count_prompt_tokens(
    processor: Any,
    *,
    width: int,
    height: int,
    frames: int,
    prompt_text: str,
) -> dict[str, Any]:
    images = [make_frame(width, height, index) for index in range(frames)]
    messages = build_messages(images, prompt_text)
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    output = processor(text=[text], images=images, return_tensors="pt")
    input_ids = output["input_ids"]
    image_grid = output.get("image_grid_thw")
    image_token_id = getattr(processor, "image_token_id", None)
    if image_token_id is None:
        tokenizer = getattr(processor, "tokenizer", None)
        image_token_id = getattr(tokenizer, "image_token_id", None)
    image_token_count = None
    if image_token_id is not None:
        image_token_count = int((input_ids == int(image_token_id)).sum().item())
    return {
        "frames": frames,
        "prompt_tokens": int(input_ids.shape[-1]),
        "image_token_count": image_token_count,
        "image_grid_thw": (
            image_grid.detach().cpu().tolist() if image_grid is not None else None
        ),
    }


def estimate_capacity(
    *,
    samples: list[dict[str, Any]],
    max_context_tokens: int,
    reserve_output_tokens: int,
) -> dict[str, Any]:
    samples_by_frames = {int(item["frames"]): item for item in samples}
    if 1 not in samples_by_frames or 2 not in samples_by_frames:
        raise ValueError("--sample-frame-counts must include 1 and 2")
    one = samples_by_frames[1]["prompt_tokens"]
    two = samples_by_frames[2]["prompt_tokens"]
    tokens_per_extra_frame = max(two - one, 1)
    fixed_tokens = one - tokens_per_extra_frame
    usable = max_context_tokens - reserve_output_tokens - fixed_tokens
    capacity = max(math.floor(usable / tokens_per_extra_frame), 0)
    return {
        "fixed_tokens_estimate": fixed_tokens,
        "tokens_per_frame_estimate": tokens_per_extra_frame,
        "max_frames_estimate": capacity,
    }


def to_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Qwen3.5 VL Frame Capacity Estimate",
        "",
        f"- Max context tokens: `{result['max_context_tokens']}`",
        f"- Reserve output tokens: `{result['reserve_output_tokens']}`",
        f"- Model: `{result['model']}`",
        "",
        "| Resolution | Tokens / Frame | Fixed Tokens | Estimated Max Frames | 1 Frame Tokens | 2 Frame Tokens | Image Grid THW |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in result["rows"]:
        sample1 = next(item for item in row["samples"] if item["frames"] == 1)
        sample2 = next(item for item in row["samples"] if item["frames"] == 2)
        lines.append(
            "| {resolution} | {tokens_per_frame_estimate} | "
            "{fixed_tokens_estimate} | {max_frames_estimate} | "
            "{one} | {two} | `{grid}` |".format(
                resolution=row["resolution"],
                tokens_per_frame_estimate=row["tokens_per_frame_estimate"],
                fixed_tokens_estimate=row["fixed_tokens_estimate"],
                max_frames_estimate=row["max_frames_estimate"],
                one=sample1["prompt_tokens"],
                two=sample2["prompt_tokens"],
                grid=sample1["image_grid_thw"],
            )
        )
    lines.append("")
    lines.append(
        "Note: this estimates the standard Responses multi-frame image path "
        "(`input_image` parts), not a `video_url` path."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = build_parser().parse_args()
    if 1 not in args.sample_frame_counts or 2 not in args.sample_frame_counts:
        raise ValueError("--sample-frame-counts must include 1 and 2")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=False)
    rows = []
    for item in args.resolutions:
        width, height = parse_resolution(item)
        samples = [
            count_prompt_tokens(
                processor,
                width=width,
                height=height,
                frames=frames,
                prompt_text=args.prompt_text,
            )
            for frames in sorted(set(args.sample_frame_counts))
        ]
        estimate = estimate_capacity(
            samples=samples,
            max_context_tokens=args.max_context_tokens,
            reserve_output_tokens=args.reserve_output_tokens,
        )
        rows.append(
            {
                "resolution": f"{width}x{height}",
                "width": width,
                "height": height,
                "samples": samples,
                **estimate,
            }
        )
    result = {
        "model": args.model,
        "max_context_tokens": args.max_context_tokens,
        "reserve_output_tokens": args.reserve_output_tokens,
        "rows": rows,
    }
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if args.output_md:
        path = Path(args.output_md)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(to_markdown(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
