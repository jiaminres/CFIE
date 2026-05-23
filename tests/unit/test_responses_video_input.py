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


def test_chat_parser_maps_input_video_to_video_url_content():
    part_type, content = _parse_chat_message_content_mm_part(
        {
            "type": "input_video",
            "video_url": {"url": "file:///tmp/demo.mp4"},
        }
    )

    assert part_type == "input_video"
    assert content == "file:///tmp/demo.mp4"


def test_video_jpeg_data_url_round_trip_loads_frames():
    frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    frames[1, :, :, 0] = 255
    video_url = encode_video_url(frames, format="JPEG")

    loaded_frames, metadata = MediaConnector().fetch_video(video_url)

    assert loaded_frames.shape == frames.shape
    assert isinstance(metadata, dict)
