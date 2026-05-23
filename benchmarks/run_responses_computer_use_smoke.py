from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageGrab

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cfie_client import ComputerLoop, Qwen35ComputerAdapter
from cfie_client.protocol import ProtocolError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test Responses video input plus cfie_client computer-call "
            "execution on a controlled GUI page."
        )
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen35-vl")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--artifact-dir", default=".bench_logs/responses_computer_use")
    parser.add_argument("--max-output-tokens", type=int, default=160)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--visual-max-width", type=int, default=960)
    parser.add_argument("--video-frames", type=int, default=2)
    parser.add_argument(
        "--crop-right-half",
        action="store_true",
        help="Use the right half of the desktop as the visual input and add the crop offset back before execution.",
    )
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute the model's computer_call on the local desktop.",
    )
    parser.add_argument("--result-json", default=None)
    return parser


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def make_controlled_gui(width: int, height: int) -> Image.Image:
    image = Image.new("RGB", (width, height), (245, 247, 250))
    draw = ImageDraw.Draw(image)
    title_font = _font(28)
    body_font = _font(20)
    small_font = _font(15)

    draw.rectangle((0, 0, width, 64), fill=(20, 34, 57))
    draw.text((24, 18), "CFIE GUI Tool Smoke", fill="white", font=title_font)
    draw.rounded_rectangle((80, 118, width - 80, height - 92), radius=10, fill="white")
    draw.text((112, 146), "Message", fill=(17, 24, 39), font=body_font)
    input_box = (112, 186, width - 112, 286)
    draw.rounded_rectangle(input_box, radius=8, fill=(255, 255, 255),
                           outline=(59, 130, 246), width=3)
    draw.text((input_box[0] + 18, input_box[1] + 32), "type here", fill=(148, 163, 184),
              font=body_font)
    submit = (width - 276, 302, width - 112, 376)
    draw.rounded_rectangle(submit, radius=8, fill=(37, 99, 235))
    draw.text((submit[0] + 46, submit[1] + 24), "Submit", fill="white", font=body_font)
    draw.text(
        (112, height - 126),
        "Target action: click the Message box, type 'hello cfie', then click Submit.",
        fill=(71, 85, 105),
        font=small_font,
    )
    return image


def image_to_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def write_video(path: Path, image: Image.Image, *, frames: int = 4) -> None:
    frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        2.0,
        image.size,
    )
    for _ in range(frames):
        writer.write(frame)
    writer.release()


def write_html(path: Path) -> None:
    path.write_text(
        """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>CFIE GUI Tool Smoke</title>
  <style>
    body { margin: 0; font-family: Segoe UI, Arial, sans-serif; background: #f5f7fa; }
    header { height: 64px; background: #142239; color: white; display: flex; align-items: center; padding-left: 24px; font-size: 28px; font-weight: 700; }
    main { margin: 54px auto; width: 80%; background: white; padding: 32px; border-radius: 10px; }
    label { display: block; font-size: 20px; font-weight: 650; margin-bottom: 14px; }
    textarea { width: 100%; height: 100px; font-size: 24px; border: 3px solid #3b82f6; border-radius: 8px; padding: 12px; box-sizing: border-box; }
    button { float: right; margin-top: 16px; width: 164px; height: 74px; border: 0; border-radius: 8px; background: #2563eb; color: white; font-size: 20px; font-weight: 650; }
    #status { clear: both; padding-top: 34px; color: #475569; font-size: 16px; }
  </style>
</head>
<body>
  <header>CFIE GUI Tool Smoke</header>
  <main>
    <label for="message">Message</label>
    <textarea id="message" autofocus placeholder="type here"></textarea>
    <button id="submit" onclick="document.getElementById('status').textContent='submitted: ' + document.getElementById('message').value">Submit</button>
    <div id="status">Target action: click the Message box, type 'hello cfie', then click Submit.</div>
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )


def post_json(
    *,
    base_url: str,
    payload: dict[str, Any],
    api_key: str,
    timeout: float,
) -> tuple[int, dict[str, Any] | str]:
    body = json.dumps(payload).encode("utf-8")
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
            return response.status, json.loads(raw)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, raw
    except (TimeoutError, urllib.error.URLError) as exc:
        return 0, {"error": repr(exc)}


def extract_text(value: Any) -> str:
    parts: list[str] = []

    def walk(item: Any) -> None:
        if isinstance(item, dict):
            if isinstance(item.get("text"), str):
                parts.append(item["text"])
            for key in ("output_text", "content", "output"):
                if key in item:
                    walk(item[key])
        elif isinstance(item, list):
            for child in item:
                walk(child)

    walk(value)
    return "".join(parts)


def build_payload(
    *,
    model: str,
    screenshot_url: str,
    video_url: str,
    max_output_tokens: int,
    video_frames: int,
) -> dict[str, Any]:
    instruction = (
        "You are controlling a Windows GUI through OpenAI computer_call JSON. "
        "Look at the screenshot and video. Return only valid JSON, no markdown. "
        "The JSON must be one object with an actions array. "
        "Use these action types only: click, type, wait. "
        "Task: click the Message input box, type exactly 'hello cfie', then click Submit. "
        "For click targets, use the center of the textarea and the center of the blue Submit button. "
        "Return all required actions in order in the same response; do not return only the first action. "
        "Use approximate coordinates in the visual input image coordinate system."
    )
    return {
        "model": model,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": instruction},
                    {"type": "input_image", "image_url": screenshot_url, "detail": "auto"},
                    {"type": "input_video", "video_url": video_url},
                ],
            }
        ],
        "temperature": 0,
        "max_output_tokens": max_output_tokens,
        "store": False,
        "media_io_kwargs": {"video": {"num_frames": video_frames}},
        "chat_template_kwargs": {"enable_thinking": False},
    }


def open_browser(path: Path) -> None:
    subprocess.Popen(
        ["cmd", "/c", "start", "", "msedge", "--start-maximized", path.as_uri()],
        shell=False,
    )


def _resize_for_model(image: Image.Image, max_width: int) -> tuple[Image.Image, float]:
    if max_width <= 0 or image.width <= max_width:
        return image, 1.0
    scale = max_width / image.width
    resized = image.resize(
        (max_width, max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    return resized, scale


def _prepare_visual_image(
    image: Image.Image,
    *,
    max_width: int,
    crop_right_half: bool,
) -> tuple[Image.Image, float, tuple[int, int]]:
    offset = (0, 0)
    if crop_right_half:
        left = image.width // 2
        image = image.crop((left, 0, image.width, image.height))
        offset = (left, 0)
    resized, scale = _resize_for_model(image, max_width)
    return resized, scale, offset


def capture_desktop_video(
    path: Path,
    *,
    frames: int = 2,
    max_width: int = 960,
    crop_right_half: bool = False,
) -> tuple[Image.Image, float, tuple[int, int]]:
    images: list[Image.Image] = []
    scale = 1.0
    offset = (0, 0)
    for _ in range(frames):
        image = ImageGrab.grab().convert("RGB")
        resized, scale, offset = _prepare_visual_image(
            image,
            max_width=max_width,
            crop_right_half=crop_right_half,
        )
        images.append(resized)
        time.sleep(0.15)

    first = images[0]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        4.0,
        first.size,
    )
    for image in images:
        frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        writer.write(frame)
    writer.release()
    return first, scale, offset


def scale_call_to_screen(call, scale: float, offset: tuple[int, int] = (0, 0)):
    if scale == 1.0 and offset == (0, 0):
        return call
    factor = 1.0 / scale
    offset_x, offset_y = offset
    actions = []
    for action in call.actions:
        raw = action.to_openai_dict()
        if "x" in raw:
            raw["x"] = round(raw["x"] * factor + offset_x)
        if "y" in raw:
            raw["y"] = round(raw["y"] * factor + offset_y)
        if "path" in raw:
            raw["path"] = [
                {
                    "x": round(point["x"] * factor + offset_x),
                    "y": round(point["y"] * factor + offset_y),
                }
                for point in raw["path"]
            ]
        actions.append(raw)
    payload = call.to_openai_dict()
    payload["actions"] = actions
    return Qwen35ComputerAdapter().to_computer_call(payload)


def main() -> None:
    args = build_parser().parse_args()
    artifact_dir = Path(args.artifact_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    image = make_controlled_gui(args.width, args.height)
    image_path = artifact_dir / "controlled_gui.png"
    video_path = artifact_dir / "controlled_gui.mp4"
    html_path = artifact_dir / "controlled_gui.html"
    result_path = Path(args.result_json).resolve() if args.result_json else artifact_dir / "result.json"
    image.save(image_path)
    write_video(video_path, image)
    write_html(html_path)

    visual_image = image
    visual_video_path = video_path
    visual_scale = 1.0
    visual_offset = (0, 0)

    if args.open_browser or args.execute:
        open_browser(html_path)
        time.sleep(2.0)
        desktop_image_path = artifact_dir / "desktop_gui.png"
        desktop_video_path = artifact_dir / "desktop_gui.mp4"
        visual_image, visual_scale, visual_offset = capture_desktop_video(
            desktop_video_path,
            frames=args.video_frames,
            max_width=args.visual_max_width,
            crop_right_half=args.crop_right_half,
        )
        visual_image.save(desktop_image_path)
        visual_video_path = desktop_video_path

    payload = build_payload(
        model=args.model,
        screenshot_url=image_to_data_url(visual_image),
        video_url=visual_video_path.as_uri(),
        max_output_tokens=args.max_output_tokens,
        video_frames=args.video_frames,
    )
    if args.build_only:
        status = -1
        response: dict[str, Any] | str = {"build_only": True}
        elapsed = 0.0
        output_text = ""
    else:
        start = time.perf_counter()
        status, response = post_json(
            base_url=args.base_url,
            payload=payload,
            api_key=args.api_key,
            timeout=args.timeout,
        )
        elapsed = time.perf_counter() - start
        output_text = extract_text(response)

    execution: dict[str, Any] = {"enabled": args.execute}
    if args.execute:
        adapter = Qwen35ComputerAdapter()
        try:
            call = adapter.to_computer_call_from_text(output_text)
            scaled_call = scale_call_to_screen(call, visual_scale, visual_offset)
            loop = ComputerLoop(
                trace_path=artifact_dir / "computer_trace.jsonl",
                trace_artifact_dir=artifact_dir / "computer_trace_artifacts",
            )
            call_output = loop.handle_call(scaled_call)
            execution.update(
                {
                    "ok": True,
                    "call": call.to_openai_dict(),
                    "scaled_call": scaled_call.to_openai_dict(),
                    "output": call_output.to_openai_dict(),
                }
            )
        except (ProtocolError, RuntimeError, ValueError) as exc:
            execution.update({"ok": False, "error": repr(exc)})

    result = {
        "status": status,
        "seconds": elapsed,
        "output_text": output_text,
        "raw_response": response,
        "artifacts": {
            "image": str(image_path),
            "video": str(video_path),
            "html": str(html_path),
            "visual_video": str(visual_video_path),
            "visual_scale": visual_scale,
            "visual_offset": list(visual_offset),
        },
        "execution": execution,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
