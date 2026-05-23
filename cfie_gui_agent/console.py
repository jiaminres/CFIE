from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from cfie_gui_agent.human_loop import HumanLoopManager, InMemoryHumanChannel

CLIENT_SOURCE = "client"


@dataclass(slots=True)
class ConsoleState:
    human_loop: HumanLoopManager = field(
        default_factory=lambda: HumanLoopManager(channel=InMemoryHumanChannel())
    )
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass(slots=True)
class ConsoleServerHandle:
    server: ThreadingHTTPServer
    thread: threading.Thread
    state: ConsoleState
    url: str

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def create_handler(state: ConsoleState) -> type[BaseHTTPRequestHandler]:
    class GuiAgentConsoleHandler(BaseHTTPRequestHandler):
        server_version = "CFIEGuiAgentConsole/0.1"

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(_HTML)
                return
            if parsed.path == "/api/human/requests":
                query = parse_qs(parsed.query)
                include_completed = query.get("include_completed", ["0"])[0] in {
                    "1",
                    "true",
                    "yes",
                }
                with state.lock:
                    payload = {
                        "requests": list(
                            state.human_loop.list_requests(
                                include_completed=include_completed
                            )
                        ),
                        "urgent_queue": list(state.human_loop.urgent_queue),
                    }
                self._send_json(payload)
                return
            self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/api/human/demo":
                    body = self._read_json(default={})
                    with state.lock:
                        request = state.human_loop.request_help(
                            question=str(
                                body.get("question")
                                or "Please provide the next human instruction."
                            ),
                            task_id=body.get("task_id"),
                            evidence_refs=tuple(body.get("evidence_refs", ()) or ()),
                            risk_reason=body.get("risk_reason"),
                            proposed_action=body.get("proposed_action"),
                            allowed_reply_format=body.get("allowed_reply_format"),
                            urgency=str(body.get("urgency", "normal")),
                            metadata={
                                "job_id": body.get("job_id"),
                                "source": "console_demo",
                            },
                        )
                    self._send_json({"request": request.to_task_payload()})
                    return

                if parsed.path.endswith("/claim"):
                    request_id = _request_id_from_path(parsed.path)
                    with state.lock:
                        state_data = state.human_loop.claim_request(
                            request_id,
                            source=CLIENT_SOURCE,
                        ).to_dict()
                    self._send_json({"state": state_data})
                    return

                if parsed.path.endswith("/release"):
                    request_id = _request_id_from_path(parsed.path)
                    with state.lock:
                        state_data = state.human_loop.release_request(
                            request_id,
                            source=CLIENT_SOURCE,
                        ).to_dict()
                    self._send_json({"state": state_data})
                    return

                if parsed.path.endswith("/reply"):
                    request_id = _request_id_from_path(parsed.path)
                    body = self._read_json(default={})
                    text = str(body.get("text", "")).strip()
                    if not text:
                        self._send_json(
                            {"error": "reply text is required"},
                            status=HTTPStatus.BAD_REQUEST,
                        )
                        return
                    with state.lock:
                        task = state.human_loop.submit_reply(
                            request_id=request_id,
                            text=text,
                            source=CLIENT_SOURCE,
                            metadata={"client": "local_console"},
                        )
                    self._send_json({"task": task})
                    return
            except (KeyError, ValueError) as exc:
                self._send_json(
                    {"error": str(exc)},
                    status=HTTPStatus.CONFLICT,
                )
                return
            self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _read_json(self, *, default: dict[str, Any]) -> dict[str, Any]:
            content_length = int(self.headers.get("Content-Length", "0") or 0)
            if content_length <= 0:
                return default
            raw = self.rfile.read(content_length)
            if not raw:
                return default
            parsed = json.loads(raw.decode("utf-8"))
            if not isinstance(parsed, dict):
                raise ValueError("request body must be JSON object")
            return parsed

        def _send_html(self, content: str) -> None:
            encoded = content.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _send_json(
            self,
            payload: dict[str, Any],
            *,
            status: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    return GuiAgentConsoleHandler


def run_console(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = False,
    human_loop: HumanLoopManager | None = None,
) -> ThreadingHTTPServer:
    state = ConsoleState(
        human_loop=human_loop
        if human_loop is not None
        else HumanLoopManager(channel=InMemoryHumanChannel())
    )
    server, url = build_console_server(host=host, port=port, state=state)
    print(f"CFIE GUI Agent console: {url}", flush=True)
    if open_browser:
        webbrowser.open(url)
    server.serve_forever()
    return server


def build_console_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    state: ConsoleState | None = None,
    human_loop: HumanLoopManager | None = None,
) -> tuple[ThreadingHTTPServer, str]:
    if state is None:
        state = ConsoleState(
            human_loop=human_loop
            if human_loop is not None
            else HumanLoopManager(channel=InMemoryHumanChannel())
        )
    server = ThreadingHTTPServer((host, port), create_handler(state))
    bound_host, bound_port = server.server_address
    return server, f"http://{bound_host}:{bound_port}/"


def start_console_in_thread(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    human_loop: HumanLoopManager | None = None,
    open_browser: bool = False,
) -> ConsoleServerHandle:
    state = ConsoleState(
        human_loop=human_loop
        if human_loop is not None
        else HumanLoopManager(channel=InMemoryHumanChannel())
    )
    server, url = build_console_server(host=host, port=port, state=state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    if open_browser:
        webbrowser.open(url)
    return ConsoleServerHandle(server=server, thread=thread, state=state, url=url)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CFIE GUI Agent local console.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()
    run_console(host=args.host, port=args.port, open_browser=args.open_browser)


def _request_id_from_path(path: str) -> str:
    parts = [part for part in path.split("/") if part]
    if len(parts) < 4 or parts[:3] != ["api", "human", "requests"]:
        raise KeyError("request id not found in path")
    return parts[3]


_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CFIE GUI Agent</title>
  <style>
    :root {
      color-scheme: light;
      font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
      background: #f5f7fb;
      color: #172033;
    }
    body { margin: 0; }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 16px 20px;
      background: #172033;
      color: white;
    }
    h1 { font-size: 18px; margin: 0; font-weight: 650; }
    main {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      gap: 16px;
      padding: 16px;
    }
    .panel, .request {
      background: white;
      border: 1px solid #d9e1ee;
      border-radius: 8px;
      box-shadow: 0 1px 2px rgba(20, 30, 50, 0.04);
    }
    .panel { padding: 14px; }
    .toolbar { display: flex; gap: 8px; align-items: center; }
    button {
      border: 1px solid #b9c7da;
      border-radius: 6px;
      padding: 8px 10px;
      background: #fff;
      color: #172033;
      cursor: pointer;
      font-weight: 600;
    }
    button.primary { background: #2563eb; border-color: #2563eb; color: white; }
    button.danger { background: #fff5f5; border-color: #f1b4b4; color: #9f1d1d; }
    button:disabled { opacity: 0.45; cursor: not-allowed; }
    .list { display: grid; gap: 12px; }
    .request { padding: 14px; display: grid; gap: 10px; }
    .request-head {
      display: flex;
      align-items: start;
      justify-content: space-between;
      gap: 12px;
    }
    .title { font-weight: 700; line-height: 1.35; }
    .meta { color: #607089; font-size: 12px; line-height: 1.55; }
    .badge {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      background: #eef4ff;
      color: #1c4ed8;
      padding: 3px 8px;
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }
    textarea, input {
      width: 100%;
      box-sizing: border-box;
      border: 1px solid #c7d2e3;
      border-radius: 6px;
      padding: 9px 10px;
      font: inherit;
      resize: vertical;
    }
    textarea { min-height: 84px; }
    .actions { display: flex; flex-wrap: wrap; gap: 8px; }
    .empty {
      border: 1px dashed #b9c7da;
      border-radius: 8px;
      padding: 32px;
      color: #607089;
      text-align: center;
      background: #fbfdff;
    }
    .side-title { font-size: 14px; font-weight: 700; margin: 0 0 10px; }
    pre {
      white-space: pre-wrap;
      background: #0f172a;
      color: #e5e7eb;
      border-radius: 6px;
      padding: 10px;
      max-height: 300px;
      overflow: auto;
      font-size: 12px;
    }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>CFIE GUI Agent 人工介入</h1>
    <div class="toolbar">
      <button onclick="createDemo()">创建示例请求</button>
      <button class="primary" onclick="refresh()">刷新</button>
    </div>
  </header>
  <main>
    <section class="list" id="requests"></section>
    <aside class="panel">
      <p class="side-title">共享状态</p>
      <div class="meta">
        客户端和后续微信等 channel 使用同一个 HumanLoopManager。
        请求被某一端认领后，其他端不能提交回复。
      </div>
      <pre id="raw"></pre>
    </aside>
  </main>
  <script>
    async function api(path, options = {}) {
      const response = await fetch(path, {
        headers: { "Content-Type": "application/json" },
        ...options,
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || response.statusText);
      return payload;
    }

    async function refresh() {
      const payload = await api("/api/human/requests?include_completed=1");
      document.getElementById("raw").textContent = JSON.stringify(payload, null, 2);
      const root = document.getElementById("requests");
      root.innerHTML = "";
      const active = payload.requests.filter(item => item.status !== "resolved");
      if (!active.length) {
        root.innerHTML = '<div class="empty">暂无需要人工处理的请求</div>';
        return;
      }
      for (const item of active) root.appendChild(renderRequest(item));
    }

    function renderRequest(item) {
      const request = item.request;
      const div = document.createElement("article");
      div.className = "request";
      const claimedBy = item.claimed_by || "";
      const isClaimedByClient = claimedBy === "client";
      const locked = item.status === "claimed" && !isClaimedByClient;
      div.innerHTML = `
        <div class="request-head">
          <div>
            <div class="title">${escapeHtml(request.question)}</div>
            <div class="meta">
              request_id=${escapeHtml(request.request_id)}<br>
              task_id=${escapeHtml(request.task_id || "")}
              urgency=${escapeHtml(request.urgency || "normal")}
            </div>
          </div>
          <span class="badge">${escapeHtml(item.status)}${claimedBy ? " · " + escapeHtml(claimedBy) : ""}</span>
        </div>
        ${request.risk_reason ? `<div class="meta">风险：${escapeHtml(request.risk_reason)}</div>` : ""}
        ${request.proposed_action ? `<div class="meta">建议动作：${escapeHtml(request.proposed_action)}</div>` : ""}
        <textarea placeholder="输入给 Agent 的人工指令或回复" ${locked ? "disabled" : ""}></textarea>
        <div class="actions">
          <button onclick="claim('${request.request_id}')" ${item.status === "claimed" ? "disabled" : ""}>认领</button>
          <button onclick="releaseReq('${request.request_id}')" ${!isClaimedByClient ? "disabled" : ""}>释放</button>
          <button class="primary" onclick="reply('${request.request_id}', this)" ${locked ? "disabled" : ""}>提交回复</button>
        </div>
      `;
      return div;
    }

    async function claim(id) {
      try { await api(`/api/human/requests/${id}/claim`, { method: "POST", body: "{}" }); }
      catch (err) { alert(err.message); }
      await refresh();
    }

    async function releaseReq(id) {
      try { await api(`/api/human/requests/${id}/release`, { method: "POST", body: "{}" }); }
      catch (err) { alert(err.message); }
      await refresh();
    }

    async function reply(id, button) {
      const text = button.closest(".request").querySelector("textarea").value.trim();
      if (!text) { alert("请输入回复内容"); return; }
      try {
        await api(`/api/human/requests/${id}/reply`, {
          method: "POST",
          body: JSON.stringify({ text }),
        });
      } catch (err) {
        alert(err.message);
      }
      await refresh();
    }

    async function createDemo() {
      await api("/api/human/demo", {
        method: "POST",
        body: JSON.stringify({
          question: "当前子任务需要人工确认：是否继续执行？",
          task_id: "demo_subtask",
          job_id: "job:demo",
          urgency: "high",
          risk_reason: "模型判断下一步可能影响任务状态。",
          proposed_action: "等待管理者给出明确回复。",
        }),
      });
      await refresh();
    }

    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, ch => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }[ch]));
    }

    refresh();
    setInterval(refresh, 2000);
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
