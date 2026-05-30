from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from cfie.entrypoints.openai.reasoning_template import (
    QWEN_REASONING_PREAMBLE_KWARG,
    click_local_refinement_preamble,
)
from cfie_gui_agent import OpenAIResponsesAgent


class CaptureHandler(BaseHTTPRequestHandler):
    captured_payload: dict[str, Any] | None = None
    captured_path: str | None = None
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
        CaptureHandler.captured_path = self.path
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


def test_openai_responses_agent_accepts_root_base_url():
    CaptureHandler.captured_payload = None
    CaptureHandler.captured_path = None
    CaptureHandler.response_payload = {"id": "resp_ok", "output": []}
    server = ThreadingHTTPServer(("127.0.0.1", 0), CaptureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        agent = OpenAIResponsesAgent(
            model="qwen35-vl",
            base_url=f"http://127.0.0.1:{server.server_address[1]}",
            include_tools=False,
            max_output_tokens=64,
        )
        agent(
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                }
            ]
        )
    finally:
        server.shutdown()
        server.server_close()

    assert CaptureHandler.captured_path == "/v1/responses"
    assert CaptureHandler.captured_payload is not None


def test_openai_responses_agent_accepts_responses_endpoint_base_url():
    CaptureHandler.captured_payload = None
    CaptureHandler.captured_path = None
    CaptureHandler.response_payload = {"id": "resp_ok", "output": []}
    server = ThreadingHTTPServer(("127.0.0.1", 0), CaptureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        agent = OpenAIResponsesAgent(
            model="qwen35-vl",
            base_url=f"http://127.0.0.1:{server.server_address[1]}/v1/responses",
            include_tools=False,
            max_output_tokens=64,
        )
        agent(
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                }
            ]
        )
    finally:
        server.shutdown()
        server.server_close()

    assert CaptureHandler.captured_path == "/v1/responses"
    assert CaptureHandler.captured_payload is not None


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
    assert "_cfie_request_debug" in response
    payload = CaptureHandler.captured_payload
    assert payload is not None
    assert payload["model"] == "qwen35-vl"
    assert payload["input"][0]["role"] == "system"
    assert payload["input"][2]["output"] == '{"status": "accepted"}'
    assert payload["input"][3]["type"] == "computer_call_output"
    assert payload["input"][3]["output"]["image_url"] == "file:///tmp/screen.jpg"
    assert payload["input"][4]["type"] == "computer_call_output"
    assert payload["input"][4]["output"]["image_url"] == "file:///tmp/screen_after.jpg"
    tool_names = {tool["name"] for tool in payload["tools"]}
    assert "open_url" in tool_names
    assert "append_trace_note" not in tool_names
    assert payload["tools"][0]["type"] == "function"
    assert "name" in payload["tools"][0]
    assert "function" not in payload["tools"][0]
    assert payload["tool_choice"] == "auto"
    assert payload["parallel_tool_calls"] is False
    assert payload["store"] is False
    assert payload["reasoning"] == {"effort": "none"}
    assert payload["chat_template_kwargs"]["enable_thinking"] is False
    debug_input = response["_cfie_request_debug"]["input"]
    assert debug_input[3]["output"]["image_url"] == {
        "placeholder": "[图片]",
        "source_type": "reference",
        "ref": "file:///tmp/screen.jpg",
    }


def test_openai_responses_agent_adds_default_image_detail():
    CaptureHandler.captured_payload = None
    CaptureHandler.response_payload = {
        "id": "resp_image_detail",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "ok"}],
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
            include_tools=False,
            max_output_tokens=64,
        )
        agent(
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "describe"},
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64,AAAA",
                        },
                    ],
                }
            ]
        )
    finally:
        server.shutdown()
        server.server_close()

    payload = CaptureHandler.captured_payload
    assert payload is not None
    assert payload["input"][0]["content"][1]["detail"] == "auto"


def test_openai_responses_agent_uses_local_click_reasoning_preamble():
    CaptureHandler.captured_payload = None
    CaptureHandler.response_payload = {"id": "resp_ok", "output": []}
    server = ThreadingHTTPServer(("127.0.0.1", 0), CaptureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        agent = OpenAIResponsesAgent(
            model="qwen35-vl",
            base_url=f"http://127.0.0.1:{server.server_address[1]}/v1",
            include_tools=False,
            max_output_tokens=64,
            reasoning_effort="none",
        )
        agent(
            [
                {
                    "type": "computer_call_output",
                    "call_id": "call_click_guard",
                    "output": {
                        "type": "computer_screenshot",
                        "image_url": "data:image/png;base64,LOCAL",
                        "precision_mode": "pre_click_refinement",
                        "structured_data": {
                            "mode": "pre_click_refinement",
                            "next_coordinate_space": "local_refinement_1000",
                        },
                    },
                }
            ]
        )
    finally:
        server.shutdown()
        server.server_close()

    payload = CaptureHandler.captured_payload
    assert payload is not None
    assert payload["reasoning"] == {"effort": "minimal"}
    assert payload["chat_template_kwargs"]["enable_thinking"] is True
    assert (
        payload["chat_template_kwargs"][QWEN_REASONING_PREAMBLE_KWARG]
        == click_local_refinement_preamble("none")
    )
    assert "recommended_click_1000" not in payload["chat_template_kwargs"][
        QWEN_REASONING_PREAMBLE_KWARG
    ]


def test_openai_responses_agent_can_use_isolated_local_click_context():
    CaptureHandler.captured_payload = None
    CaptureHandler.response_payload = {"id": "resp_ok", "output": []}
    server = ThreadingHTTPServer(("127.0.0.1", 0), CaptureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        agent = OpenAIResponsesAgent(
            model="qwen35-vl",
            base_url=f"http://127.0.0.1:{server.server_address[1]}/v1",
            include_tools=False,
            max_output_tokens=64,
            reasoning_effort="origin",
            click_local_refinement_reasoning=False,
        )
        agent(
            [
                {
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": "system rules"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "old full history"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_click",
                    "name": "computer_use",
                    "arguments": json.dumps({"actions": [{"type": "click"}]}),
                },
                {
                    "type": "computer_call_output",
                    "call_id": "call_click",
                    "output": {
                        "type": "computer_screenshot",
                        "image_url": "data:image/png;base64,LOCAL",
                        "detail": "high",
                        "intent": "点击目标 A 的红色小圆点",
                        "structured_data": {
                            "mode": "pre_click_refinement",
                            "next_coordinate_space": "local_refinement_1000",
                        },
                    },
                },
            ]
        )
    finally:
        server.shutdown()
        server.server_close()

    payload = CaptureHandler.captured_payload
    assert payload is not None
    text = json.dumps(payload["input"], ensure_ascii=False)
    assert len(payload["input"]) == 1
    assert "system rules" not in text
    assert "old full history" not in text
    assert "call_click" not in text
    assert "点击目标 A 的红色小圆点" in text
    assert "local_refinement_1000" in text
    assert payload["input"][0]["content"][1]["image_url"] == (
        "data:image/png;base64,LOCAL"
    )


def test_openai_responses_agent_does_not_rewrite_qwen_text_tool_call_output():
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

    assert response["output"] == CaptureHandler.response_payload["output"]
    assert "<tool_call>" in response["output"][0]["content"][0]["text"]


def test_openai_responses_agent_does_not_rewrite_bare_qwen_function_block():
    CaptureHandler.captured_payload = None
    CaptureHandler.response_payload = {
        "id": "resp_bare_function_tool",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": (
                            "<function=computer_use>"
                            "<parameter=coordinate_space>"
                            "qwen_normalized_1000"
                            "</parameter>"
                            "<parameter=actions>"
                            '[{"type":"click","x":500,"y":900}]'
                            "</parameter>"
                            "</function>"
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

    assert response["output"] == CaptureHandler.response_payload["output"]
    assert "<function=computer_use>" in response["output"][0]["content"][0]["text"]


def test_openai_responses_agent_strips_inline_media_from_tool_outputs():
    CaptureHandler.captured_payload = None
    CaptureHandler.response_payload = {"id": "resp_ok", "output": []}
    server = ThreadingHTTPServer(("127.0.0.1", 0), CaptureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        agent = OpenAIResponsesAgent(
            model="qwen35-vl",
            base_url=f"http://127.0.0.1:{server.server_address[1]}/v1",
            max_output_tokens=64,
        )
        agent(
            [
                {
                    "type": "function_call_output",
                    "call_id": "tool_with_image",
                    "output": {
                        "status": "accepted",
                        "screenshot_image_url": (
                            "data:image/jpeg;base64,"
                            + ("A" * 4096)
                        ),
                        "nested": [
                            {
                                "after_ref": (
                                    "prefix data:image/png;base64,"
                                    + ("B" * 4096)
                                )
                            }
                        ],
                    },
                }
            ]
        )
    finally:
        server.shutdown()
        server.server_close()

    payload = CaptureHandler.captured_payload
    assert payload is not None
    output_text = payload["input"][0]["output"]
    assert "data:image" not in output_text
    assert "AAAA" not in output_text
    assert "BBBB" not in output_text
    assert "inline media data omitted" in output_text


def test_openai_responses_agent_strips_inline_media_from_runtime_text_only():
    CaptureHandler.captured_payload = None
    CaptureHandler.response_payload = {"id": "resp_ok", "output": []}
    server = ThreadingHTTPServer(("127.0.0.1", 0), CaptureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    image_url = "data:image/jpeg;base64," + ("C" * 4096)
    try:
        agent = OpenAIResponsesAgent(
            model="qwen35-vl",
            base_url=f"http://127.0.0.1:{server.server_address[1]}/v1",
            max_output_tokens=64,
        )
        agent(
            [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                '{"current_frame":"'
                                + image_url
                                + '","note":"image is also sent separately"}'
                            ),
                        }
                    ],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "inspect screenshot"},
                        {
                            "type": "input_image",
                            "image_url": image_url,
                            "detail": "low",
                        },
                    ],
                },
            ]
        )
    finally:
        server.shutdown()
        server.server_close()

    payload = CaptureHandler.captured_payload
    assert payload is not None
    developer_text = payload["input"][0]["content"][0]["text"]
    assert "data:image" not in developer_text
    assert "CCCC" not in developer_text
    assert "inline media data omitted" in developer_text
    assert payload["input"][1]["content"][1]["image_url"] == image_url


def test_openai_responses_agent_request_debug_redacts_typed_media():
    CaptureHandler.captured_payload = None
    CaptureHandler.response_payload = {"id": "resp_ok", "output": []}
    server = ThreadingHTTPServer(("127.0.0.1", 0), CaptureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    image_url = "data:image/jpeg;base64," + ("D" * 4096)
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
                    "content": [
                        {"type": "input_text", "text": "inspect screenshot"},
                        {
                            "type": "input_image",
                            "image_url": image_url,
                            "detail": "low",
                        },
                    ],
                }
            ]
        )
    finally:
        server.shutdown()
        server.server_close()

    payload = CaptureHandler.captured_payload
    assert payload is not None
    assert payload["input"][0]["content"][1]["image_url"] == image_url
    debug = response["_cfie_request_debug"]
    debug_json = json.dumps(debug, ensure_ascii=False)
    assert "data:image" not in debug_json
    assert "DDDD" not in debug_json
    image_debug = debug["input"][0]["content"][1]["image_url"]
    assert image_debug["placeholder"] == "[图片]"
    assert image_debug["source_type"] == "data_url"
    assert image_debug["mime_type"] == "image/jpeg"
