from __future__ import annotations

import argparse
import base64
import json
import math
import re
import time
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe Qwen VL coordinate accuracy across screenshot resolutions."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen35-vl")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--base-width", type=int, default=2560)
    parser.add_argument("--base-height", type=int, default=1440)
    parser.add_argument(
        "--widths",
        default="512,768,960,1280,1600,1920,2560",
        help="Comma separated input widths. Heights keep the base aspect ratio.",
    )
    parser.add_argument("--max-output-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--artifact-dir", default=".bench_logs/resolution_probe")
    parser.add_argument("--result-json", default=None)
    return parser


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_probe_image(width: int, height: int) -> tuple[Image.Image, dict[str, Any]]:
    image = Image.new("RGB", (width, height), (250, 249, 247))
    draw = ImageDraw.Draw(image)

    title_font = _font(34, bold=True)
    body_font = _font(24)
    small_font = _font(20)
    button_font = _font(30, bold=True)

    sidebar_w = int(width * 0.22)
    draw.rectangle((0, 0, sidebar_w, height), fill=(243, 239, 234))
    draw.text((42, 42), "CFIE", fill=(18, 24, 38), font=title_font)
    draw.rounded_rectangle(
        (32, 180, sidebar_w - 32, 250),
        radius=22,
        fill=(230, 224, 218),
    )
    draw.text((60, 198), "Doubao Web", fill=(18, 24, 38), font=body_font)
    for idx, label in enumerate(("设置", "历史", "帮助")):
        y = 320 + idx * 74
        draw.text((64, y), label, fill=(87, 96, 111), font=body_font)

    content_x0 = sidebar_w + 64
    content_x1 = width - 72
    draw.text((content_x0, 56), "Doubao Web", fill=(18, 24, 38), font=title_font)
    draw.text((content_x0, 104), "运行中", fill=(82, 92, 108), font=small_font)

    card_x0 = content_x0
    card_x1 = content_x1
    card_y0 = 180
    card_y1 = int(height * 0.56)
    draw.rounded_rectangle(
        (card_x0, card_y0, card_x1, card_y1),
        radius=28,
        fill=(255, 255, 255),
        outline=(229, 224, 218),
        width=2,
    )
    draw.text((card_x0 + 42, card_y0 + 42), "当前任务", fill=(18, 24, 38), font=body_font)
    draw.text(
        (card_x0 + 42, card_y0 + 88),
        "请定位右下角深色圆角按钮。按钮文字是：发送。",
        fill=(43, 50, 63),
        font=body_font,
    )

    # Bottom chat composer with several visual distractors. The target button is
    # intentionally small enough that low-resolution downsampling becomes visible.
    composer_margin_x = 130
    composer_h = 148
    composer_x0 = content_x0
    composer_x1 = content_x1 - composer_margin_x
    composer_y1 = height - 76
    composer_y0 = composer_y1 - composer_h
    draw.rounded_rectangle(
        (composer_x0, composer_y0, composer_x1, composer_y1),
        radius=32,
        fill=(255, 255, 255),
        outline=(224, 218, 211),
        width=2,
    )
    draw.text(
        (composer_x0 + 34, composer_y0 + 36),
        "向当前应用输入问题...",
        fill=(132, 139, 151),
        font=body_font,
    )
    draw.ellipse(
        (composer_x0 + 32, composer_y1 - 60, composer_x0 + 72, composer_y1 - 20),
        outline=(188, 194, 204),
        width=3,
    )

    button_w = 216
    button_h = 72
    button_x1 = content_x1
    button_x0 = button_x1 - button_w
    button_y1 = composer_y1
    button_y0 = button_y1 - button_h
    draw.rounded_rectangle(
        (button_x0, button_y0, button_x1, button_y1),
        radius=28,
        fill=(17, 24, 39),
    )
    draw.text((button_x0 + 58, button_y0 + 19), "发送", fill=(255, 255, 255), font=button_font)

    # A non-target button near the top to make "the dark lower-right send button"
    # necessary instead of just "the only button".
    draw.rounded_rectangle(
        (content_x1 - 168, 52, content_x1 - 72, 110),
        radius=20,
        fill=(255, 255, 255),
        outline=(224, 218, 211),
        width=2,
    )
    draw.text((content_x1 - 137, 67), "▶", fill=(17, 24, 39), font=body_font)

    center_x = (button_x0 + button_x1) / 2
    center_y = (button_y0 + button_y1) / 2
    meta = {
        "target_box": [button_x0, button_y0, button_x1, button_y1],
        "target_center_px": [center_x, center_y],
        "target_center_normalized_1000": [
            center_x / width * 1000.0,
            center_y / height * 1000.0,
        ],
    }
    return image, meta


def image_to_data_url(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG", optimize=True)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def post_response(
    *,
    base_url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        base_url.rstrip("/") + "/responses",
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            error = json.loads(raw)
        except json.JSONDecodeError:
            error = raw
        raise RuntimeError(f"HTTP {exc.code}: {error}") from exc


def extract_text(response: dict[str, Any]) -> str:
    parts: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            text = value.get("text")
            if isinstance(text, str):
                parts.append(text)
            for key in ("content", "output"):
                if key in value:
                    walk(value[key])
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(response.get("output", response))
    return "\n".join(parts).strip()


def parse_xy(text: str) -> tuple[float, float] | None:
    fenced = re.search(r"\{.*?\}", text, flags=re.S)
    candidates = [fenced.group(0)] if fenced else []
    candidates.append(text)
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "x" in value and "y" in value:
            return float(value["x"]), float(value["y"])
        if isinstance(value, dict) and isinstance(value.get("point_2d"), list):
            point = value["point_2d"]
            if len(point) >= 2:
                return float(point[0]), float(point[1])
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and isinstance(item.get("point_2d"), list):
                    point = item["point_2d"]
                    if len(point) >= 2:
                        return float(point[0]), float(point[1])
    match = re.search(
        r"['\"]?x['\"]?\s*[:=]\s*(-?\d+(?:\.\d+)?).*?['\"]?y['\"]?\s*[:=]\s*(-?\d+(?:\.\d+)?)",
        text,
        flags=re.S | re.I,
    )
    if match:
        return float(match.group(1)), float(match.group(2))
    qwen_pair = re.search(
        r"['\"]?x['\"]?\s*[:=]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)",
        text,
        flags=re.S | re.I,
    )
    if qwen_pair:
        return float(qwen_pair.group(1)), float(qwen_pair.group(2))
    tuple_pair = re.search(
        r"\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)?",
        text,
    )
    if tuple_pair:
        return float(tuple_pair.group(1)), float(tuple_pair.group(2))
    return None


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    widths = [int(item.strip()) for item in args.widths.split(",") if item.strip()]
    base_image, meta = make_probe_image(args.base_width, args.base_height)
    base_path = artifact_dir / "synthetic_gui_2560x1440.png"
    base_image.save(base_path)

    expected_x, expected_y = meta["target_center_normalized_1000"]
    results: list[dict[str, Any]] = []
    for width in widths:
        height = round(width * args.base_height / args.base_width)
        image = base_image.resize((width, height), Image.Resampling.LANCZOS)
        image_path = artifact_dir / f"input_{width}x{height}.png"
        image.save(image_path)
        prompt = (
            "请定位图中右下角深色圆角按钮“发送”的中心点。"
            "只输出 JSON，不要解释，格式必须是 {\"x\": number, \"y\": number}。"
            "坐标必须是当前输入图片的 0..1000 归一化坐标：左上角是 (0,0)，"
            "右下角是 (1000,1000)。不要输出物理屏幕像素坐标。"
        )
        payload = {
            "model": args.model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": image_to_data_url(image),
                            "detail": "high",
                        },
                    ],
                }
            ],
            "temperature": 0,
            "max_output_tokens": args.max_output_tokens,
            "store": False,
            "reasoning": {"effort": "none"},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        started = time.perf_counter()
        response = post_response(
            base_url=args.base_url,
            api_key=args.api_key,
            payload=payload,
            timeout=args.timeout,
        )
        seconds = time.perf_counter() - started
        text = extract_text(response)
        xy = parse_xy(text)
        record: dict[str, Any] = {
            "input_size": [width, height],
            "image_path": str(image_path.resolve()),
            "seconds": seconds,
            "output_text": text,
            "usage": response.get("usage"),
            "parsed_xy": list(xy) if xy else None,
            "expected_xy": [expected_x, expected_y],
        }
        if xy:
            dx = xy[0] - expected_x
            dy = xy[1] - expected_y
            normalized_error = math.hypot(dx, dy)
            base_error_px = math.hypot(
                dx / 1000.0 * args.base_width,
                dy / 1000.0 * args.base_height,
            )
            input_error_px = math.hypot(
                dx / 1000.0 * width,
                dy / 1000.0 * height,
            )
            record.update(
                {
                    "normalized_error": normalized_error,
                    "base_error_px": base_error_px,
                    "input_error_px": input_error_px,
                    "hit_within_button": (
                        abs(dx / 1000.0 * args.base_width)
                        <= (meta["target_box"][2] - meta["target_box"][0]) / 2
                        and abs(dy / 1000.0 * args.base_height)
                        <= (meta["target_box"][3] - meta["target_box"][1]) / 2
                    ),
                }
            )
        results.append(record)
        print(
            json.dumps(
                {
                    "input_size": record["input_size"],
                    "parsed_xy": record["parsed_xy"],
                    "expected_xy": record["expected_xy"],
                    "base_error_px": record.get("base_error_px"),
                    "hit_within_button": record.get("hit_within_button"),
                    "seconds": round(seconds, 3),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    result = {
        "base_size": [args.base_width, args.base_height],
        "base_image": str(base_path.resolve()),
        "target": meta,
        "results": results,
    }
    result_path = Path(args.result_json) if args.result_json else artifact_dir / "result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    args = build_parser().parse_args()
    result = run_probe(args)
    print(json.dumps({"result_json": args.result_json or str(Path(args.artifact_dir) / "result.json")}, ensure_ascii=False))
    # Print a compact table for shell logs.
    print("width\theight\tparsed_x\tparsed_y\tbase_error_px\thit")
    for item in result["results"]:
        xy = item.get("parsed_xy") or [None, None]
        err = item.get("base_error_px")
        print(
            f"{item['input_size'][0]}\t{item['input_size'][1]}\t{xy[0]}\t{xy[1]}\t"
            f"{err if err is not None else ''}\t{item.get('hit_within_button')}"
        )


if __name__ == "__main__":
    main()
