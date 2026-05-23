from __future__ import annotations

import json
import threading
from http.client import HTTPConnection
from urllib.parse import urlparse

from cfie_gui_agent.console import ConsoleState, build_console_server
from cfie_gui_agent.human_loop import HumanLoopManager, InMemoryHumanChannel


def test_console_human_input_api_claims_and_replies():
    state = ConsoleState()
    request = state.human_loop.request_help(
        question="Need command.",
        task_id="subtask_1",
        urgency="high",
        metadata={"job_id": "job:test"},
    )
    server, _ = build_console_server(host="127.0.0.1", port=0, state=state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        claimed = _request_json(
            host,
            port,
            "POST",
            f"/api/human/requests/{request.request_id}/claim",
            {},
        )
        replied = _request_json(
            host,
            port,
            "POST",
            f"/api/human/requests/{request.request_id}/reply",
            {"text": "continue"},
        )
        listing = _request_json(
            host,
            port,
            "GET",
            "/api/human/requests?include_completed=1",
            None,
        )

        assert claimed["state"]["claimed_by"] == "client"
        assert replied["task"]["reply"]["text"] == "continue"
        assert listing["requests"][0]["status"] == "resolved"
        assert listing["urgent_queue"][0]["reply"]["text"] == "continue"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_console_uses_shared_human_loop_manager():
    manager = HumanLoopManager(channel=InMemoryHumanChannel())
    request = manager.request_help(
        question="Shared request.",
        task_id="subtask_shared",
        metadata={"job_id": "job:shared"},
    )
    state = ConsoleState(human_loop=manager)
    server, _ = build_console_server(host="127.0.0.1", port=0, state=state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        listing = _request_json(
            host,
            port,
            "GET",
            "/api/human/requests",
            None,
        )
        assert listing["requests"][0]["request"]["request_id"] == request.request_id

        _request_json(
            host,
            port,
            "POST",
            f"/api/human/requests/{request.request_id}/reply",
            {"text": "Use the safe option."},
        )

        assert request.request_id not in manager.pending
        assert manager.urgent_queue[0]["reply"]["text"] == "Use the safe option."
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _request_json(
    host: str,
    port: int,
    method: str,
    path: str,
    body: dict | None,
) -> dict:
    parsed = urlparse(path)
    payload = None if body is None else json.dumps(body).encode("utf-8")
    connection = HTTPConnection(host, port, timeout=5)
    try:
        connection.request(
            method,
            parsed.geturl(),
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        data = response.read()
        assert response.status < 400, data.decode("utf-8")
        return json.loads(data.decode("utf-8"))
    finally:
        connection.close()
