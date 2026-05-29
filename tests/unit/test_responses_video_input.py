from __future__ import annotations

import numpy as np

from cfie.entrypoints.chat_utils import _parse_chat_message_content_mm_part
from cfie.entrypoints.openai.responses.protocol import ResponsesRequest
from cfie.entrypoints.openai.responses.utils import construct_input_messages
from cfie.multimodal.media.connector import MediaConnector
from cfie.multimodal.utils import encode_video_url


def test_responses_request_accepts_input_video_content_part():
    payload = {
        "model": "qwen3.5-vl",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Describe the video."},
                    {
                        "type": "input_video",
                        "video_url": "data:video/jpeg;base64,AAAA",
                    },
                ],
            }
        ],
    }

    request = ResponsesRequest.model_validate(payload)
    messages = construct_input_messages(request_input=request.input)

    assert messages == payload["input"]


def test_responses_request_strips_inline_media_from_text_only():
    image_url = "data:image/jpeg;base64," + ("A" * 4096)
    payload = {
        "model": "qwen3.5-vl",
        "input": [
            {
                "type": "message",
                "role": "developer",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            '{"current_frame":"' + image_url + '",'
                            '"note":"the image is sent separately"}'
                        ),
                    }
                ],
            },
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Inspect the screenshot."},
                    {
                        "type": "input_image",
                        "image_url": image_url,
                        "detail": "high",
                    },
                ],
            },
        ],
    }

    request = ResponsesRequest.model_validate(payload)
    messages = construct_input_messages(request_input=request.input)

    developer_text = messages[0]["content"][0]["text"]
    assert "data:image" not in developer_text
    assert "AAAA" not in developer_text
    assert "inline media data omitted" in developer_text
    assert messages[1]["content"][1]["image_url"] == image_url


def test_construct_input_messages_strips_inline_media_from_tool_output():
    output = "data:image/png;base64," + ("B" * 4096)

    messages = construct_input_messages(
        request_input=[
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": output,
            }
        ]
    )

    assert "data:image" not in messages[0]["content"]
    assert "BBBB" not in messages[0]["content"]
    assert "inline media data omitted" in messages[0]["content"]


def test_construct_input_messages_maps_computer_output_to_tool_observation():
    messages = construct_input_messages(
        request_input=[
            {
                "type": "function_call",
                "name": "computer_use",
                "call_id": "call_screen",
                "arguments": '{"actions":[{"type":"click","x":500,"y":900}]}',
            },
            {
                "type": "computer_call_output",
                "call_id": "call_screen",
                "output": {
                    "type": "computer_screenshot",
                    "image_url": "file:///tmp/screen.jpg",
                    "detail": "low",
                    "summary": "Executed actions: click(500,900)",
                },
            },
        ]
    )

    assert messages[0]["role"] == "assistant"
    assert messages[0]["tool_calls"][0]["function"]["name"] == "computer_use"
    assert messages[1]["role"] == "tool"
    assert messages[1]["tool_call_id"] == "call_screen"
    assert "click(500,900)" in messages[1]["content"][0]["text"]
    assert messages[1]["content"][1]["type"] == "input_image"
    assert messages[1]["content"][1]["image_url"] == "file:///tmp/screen.jpg"
    assert len(messages) == 2


def test_construct_input_messages_keeps_click_refinement_images_in_tool_observation():
    messages = construct_input_messages(
        request_input=[
            {
                "type": "computer_call_output",
                "call_id": "call_screen",
                "output": {
                    "type": "computer_screenshot",
                    "image_url": "file:///tmp/screen.jpg",
                    "detail": "low",
                    "summary": "Executed actions: click(500,900)",
                    "local_refinements": [
                        {
                            "type": "computer_screenshot",
                            "image_url": "file:///tmp/local.jpg",
                            "detail": "high",
                            "summary": "Click-local refinement image.",
                        }
                    ],
                },
            },
        ]
    )

    assert len(messages) == 1
    assert messages[0]["role"] == "tool"
    assert messages[0]["tool_call_id"] == "call_screen"
    assert messages[0]["content"][0]["type"] == "input_text"
    assert "click-local refinement" in messages[0]["content"][0]["text"]
    assert messages[0]["content"][1]["image_url"] == "file:///tmp/screen.jpg"
    assert messages[0]["content"][2]["text"] == "Click-local refinement image."
    assert messages[0]["content"][3]["image_url"] == "file:///tmp/local.jpg"


def test_chat_parser_maps_input_video_to_video_url_content():
    part_type, content = _parse_chat_message_content_mm_part(
        {
            "type": "input_video",
            "video_url": {"url": "file:///tmp/production_video.mp4"},
        }
    )

    assert part_type == "input_video"
    assert content == "file:///tmp/production_video.mp4"


def test_video_jpeg_data_url_round_trip_loads_frames():
    frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    frames[1, :, :, 0] = 255
    video_url = encode_video_url(frames, format="JPEG")

    loaded_frames, metadata = MediaConnector().fetch_video(video_url)

    assert loaded_frames.shape == frames.shape
    assert isinstance(metadata, dict)
