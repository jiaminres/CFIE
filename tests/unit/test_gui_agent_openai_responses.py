from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from cfie_gui_agent import OpenAIResponsesAgent


class CaptureHandler(BaseHTTPRequestHandler):
    captured_payload: dict[str, Any] | None = None
    response_payload: dict[str, Any] = {
        "id": "resp_test",
        "output": [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "append_trace_note",
                "arguments": json.dumps({"title": "ok"}),
            }
        ],
    }

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        CaptureHandler.captured_payload = json.loads(raw)
        response = CaptureHandler.response_payload
        body = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: Any) -> None:
        return


def test_openai_responses_agent_posts_flat_tool_schema_and_normalizes_outputs():
    CaptureHandler.captured_payload = None
    CaptureHandler.response_payload = {
        "id": "resp_test",
        "output": [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "append_trace_note",
                "arguments": json.dumps({"title": "ok"}),
            }
        ],
    }
    server = ThreadingHTTPServer(("127.0.0.1", 0), CaptureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        agent = OpenAIResponsesAgent(
            model="qwen35-vl",
            base_url=f"http://127.0.0.1:{server.server_address[1]}/v1",
            max_output_tokens=64,
        )
        response = agent(
            [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "runtime"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_prev",
                    "output": {"status": "accepted"},
                },
                {
                    "type": "computer_call_output",
                    "call_id": "screen_1",
                    "output": {
                        "type": "computer_screenshot",
                        "image_url": "file:///tmp/screen.jpg",
                        "detail": "low",
                    },
                },
                {
                    "type": "computer_call_output",
                    "call_id": "screen_2",
                    "output": {
                        "type": "computer_screenshot",
                        "image_url": "file:///tmp/screen_after.jpg",
                        "detail": "low",
                    },
                },
            ]
        )
    finally:
        server.shutdown()
        server.server_close()

    assert response["output"][0]["name"] == "append_trace_note"
    payload = CaptureHandler.captured_payload
    assert payload is not None
    assert payload["model"] == "qwen35-vl"
    assert payload["input"][0]["role"] == "system"
    assert payload["input"][2]["output"] == '{"status": "accepted"}'
    assert payload["input"][3]["role"] == "user"
    assert payload["input"][3]["content"][1]["type"] == "input_image"
    assert payload["input"][3]["content"][1]["image_url"] == "file:///tmp/screen.jpg"
    assert payload["input"][4]["content"][1]["image_url"] == (
        "file:///tmp/screen_after.jpg"
    )
    tool_names = {tool["name"] for tool in payload["tools"]}
    assert "set_app_viewport" in tool_names
    assert payload["tools"][0]["type"] == "function"
    assert "name" in payload["tools"][0]
    assert "function" not in payload["tools"][0]
    assert payload["tool_choice"] == "auto"
    assert payload["store"] is False
    assert payload["reasoning"] == {"effort": "none"}
    assert payload["chat_template_kwargs"]["enable_thinking"] is False


def test_openai_responses_agent_splits_qwen_text_tool_call_output():
    CaptureHandler.captured_payload = None
    CaptureHandler.response_payload = {
        "id": "resp_text_tool",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": (
                            "Next I will click send.\n"
                            "<tool_call>"
                            "<function=computer_use>"
                            "<parameter=coordinate_space>"
                            "qwen_normalized_1000"
                            "</parameter>"
                            "<parameter=actions>"
                            '[{"type":"click","x":570,"y":470}]'
                            "</parameter>"
                            "</function>"
                            "</tool_call>"
                        ),
                    }
                ],
            }
        ],
    }
    server = ThreadingHTTPServer(("127.0.0.1", 0), CaptureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        agent = OpenAIResponsesAgent(
            model="qwen35-vl",
            base_url=f"http://127.0.0.1:{server.server_address[1]}/v1",
            max_output_tokens=64,
        )
        response = agent(
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "click send"}],
                }
            ]
        )
    finally:
        server.shutdown()
        server.server_close()

    assert response["output"][0]["content"][0]["text"] == "Next I will click send."
    call = response["output"][1]
    assert call["type"] == "function_call"
    assert call["name"] == "computer_use"
    assert json.loads(call["arguments"]) == {
        "coordinate_space": "qwen_normalized_1000",
        "actions": [{"type": "click", "x": 570, "y": 470}],
    }
