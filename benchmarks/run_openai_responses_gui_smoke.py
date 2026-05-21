from __future__ import annotations

import argparse
import base64
import json
import time
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke-test the OpenAI Responses API with text and GUI frames."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="")
    parser.add_argument(
        "--case",
        choices=("all", "text", "image", "multiframe"),
        default="all",
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--max-output-tokens", type=int, default=96)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--artifact-dir", default=".bench_logs/responses_gui_smoke")
    parser.add_argument("--result-json", default=None)
    return parser


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def make_gui_frame(
    *,
    width: int,
    height: int,
    frame_index: int = 0,
    frame_count: int = 1,
) -> Image.Image:
    image = Image.new("RGB", (width, height), (246, 248, 250))
    draw = ImageDraw.Draw(image)
    title_font = _font(max(14, width // 42))
    body_font = _font(max(12, width // 56))
    small_font = _font(max(10, width // 72))

    sidebar_w = max(140, width // 5)
    top_h = max(54, height // 9)
    draw.rectangle((0, 0, width, top_h), fill=(31, 41, 55))
    draw.text((18, top_h // 2 - 10), "CFIE Dashboard", fill="white", font=title_font)
    draw.rectangle((0, top_h, sidebar_w, height), fill=(229, 233, 239))
    for idx, label in enumerate(("Overview", "Runs", "Settings")):
        y = top_h + 26 + idx * 42
        fill = (209, 213, 219) if label == "Runs" else (229, 233, 239)
        draw.rounded_rectangle((12, y, sidebar_w - 12, y + 30), radius=6, fill=fill)
        draw.text((24, y + 7), label, fill=(17, 24, 39), font=small_font)

    panel_x0 = sidebar_w + 28
    panel_y0 = top_h + 28
    panel_x1 = width - 28
    panel_y1 = height - 34
    draw.rounded_rectangle(
        (panel_x0, panel_y0, panel_x1, panel_y1),
        radius=8,
        fill="white",
        outline=(209, 213, 219),
        width=2,
    )
    draw.text(
        (panel_x0 + 24, panel_y0 + 24),
        "Deploy test application",
        fill=(17, 24, 39),
        font=title_font,
    )
    draw.text(
        (panel_x0 + 24, panel_y0 + 62),
        "Target: browser UI automation smoke test",
        fill=(75, 85, 99),
        font=body_font,
    )

    cancel = (
        panel_x1 - 260,
        panel_y1 - 86,
        panel_x1 - 150,
        panel_y1 - 44,
    )
    submit = (
        panel_x1 - 132,
        panel_y1 - 86,
        panel_x1 - 24,
        panel_y1 - 44,
    )
    draw.rounded_rectangle(cancel, radius=6, fill=(243, 244, 246), outline=(156, 163, 175))
    draw.text((cancel[0] + 25, cancel[1] + 12), "Cancel", fill=(31, 41, 55), font=body_font)
    draw.rounded_rectangle(submit, radius=6, fill=(37, 99, 235), outline=(29, 78, 216))
    draw.text((submit[0] + 24, submit[1] + 12), "Submit", fill="white", font=body_font)

    progress = 0.0 if frame_count <= 1 else frame_index / max(frame_count - 1, 1)
    cursor_x = int(panel_x0 + 55 + progress * (submit[0] + 54 - panel_x0 - 55))
    cursor_y = int(panel_y0 + 110 + progress * (submit[1] + 21 - panel_y0 - 110))
    draw.polygon(
        (
            (cursor_x, cursor_y),
            (cursor_x + 18, cursor_y + 8),
            (cursor_x + 8, cursor_y + 13),
            (cursor_x + 13, cursor_y + 28),
            (cursor_x + 7, cursor_y + 30),
            (cursor_x + 2, cursor_y + 15),
            (cursor_x - 8, cursor_y + 22),
        ),
        fill=(15, 23, 42),
    )
    draw.text(
        (panel_x0 + 24, panel_y1 - 80),
        f"Frame {frame_index + 1}/{frame_count}",
        fill=(107, 114, 128),
        font=small_font,
    )
    return image


def image_to_data_url(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG", optimize=True)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{data}"


def post_json(
    *,
    base_url: str,
    path: str,
    payload: dict[str, Any],
    api_key: str,
    timeout: float,
) -> tuple[int, dict[str, Any] | str]:
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return response.status, json.loads(raw)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, raw


def extract_text(value: Any) -> str:
    parts: list[str] = []

    def walk(item: Any) -> None:
        if isinstance(item, dict):
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
            for key in ("output_text", "content", "output"):
                if key in item:
                    walk(item[key])
        elif isinstance(item, list):
            for child in item:
                walk(child)

    walk(value)
    return "".join(parts)


def build_text_payload(model: str, max_output_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "input": "Answer briefly: what is 2 + 3?",
        "temperature": 0,
        "max_output_tokens": max_output_tokens,
        "store": False,
    }


def build_image_payload(
    *,
    model: str,
    image: Image.Image,
    max_output_tokens: int,
) -> dict[str, Any]:
    return {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "This is a GUI screenshot. Identify the blue primary "
                            "button label and return its approximate center as "
                            "JSON with keys label, x, y."
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": image_to_data_url(image),
                        "detail": "auto",
                    },
                ],
            }
        ],
        "temperature": 0,
        "max_output_tokens": max_output_tokens,
        "store": False,
    }


def build_multiframe_payload(
    *,
    model: str,
    frames: list[Image.Image],
    max_output_tokens: int,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": (
                "These images are consecutive GUI frames. State which control "
                "the cursor is approaching in the final frame, and give the "
                "approximate center coordinates of that control as JSON."
            ),
        }
    ]
    content.extend(
        {
            "type": "input_image",
            "image_url": image_to_data_url(frame),
            "detail": "auto",
        }
        for frame in frames
    )
    return {
        "model": model,
        "input": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_output_tokens": max_output_tokens,
        "store": False,
    }


def main() -> None:
    args = build_parser().parse_args()
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    frames = [
        make_gui_frame(
            width=args.width,
            height=args.height,
            frame_index=i,
            frame_count=args.frames,
        )
        for i in range(args.frames)
    ]
    frames[0].save(artifact_dir / "gui_single.png")
    for i, frame in enumerate(frames):
        frame.save(artifact_dir / f"gui_frame_{i:03d}.png")

    cases: list[tuple[str, dict[str, Any]]] = []
    if args.case in ("all", "text"):
        cases.append(("text", build_text_payload(args.model, args.max_output_tokens)))
    if args.case in ("all", "image"):
        cases.append(
            (
                "image",
                build_image_payload(
                    model=args.model,
                    image=frames[-1],
                    max_output_tokens=args.max_output_tokens,
                ),
            )
        )
    if args.case in ("all", "multiframe"):
        cases.append(
            (
                "multiframe",
                build_multiframe_payload(
                    model=args.model,
                    frames=frames,
                    max_output_tokens=args.max_output_tokens,
                ),
            )
        )

    results = []
    for name, payload in cases:
        t0 = time.perf_counter()
        status, response = post_json(
            base_url=args.base_url,
            path="/responses",
            payload=payload,
            api_key=args.api_key,
            timeout=args.timeout,
        )
        seconds = time.perf_counter() - t0
        text = extract_text(response)
        results.append(
            {
                "case": name,
                "status": status,
                "seconds": seconds,
                "output_text": text,
                "output_text_nonempty": bool(text.strip()),
                "output_has_replacement_char": "\ufffd" in text,
                "raw_response": response,
            }
        )

    result = {
        "base_url": args.base_url,
        "model": args.model,
        "width": args.width,
        "height": args.height,
        "frames": args.frames,
        "results": results,
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
