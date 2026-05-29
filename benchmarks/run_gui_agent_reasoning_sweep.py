from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Experiment:
    name: str
    effort: str
    reasoning_mode: str
    max_output_tokens: int
    max_visual_frames: int
    screenshot_size: int
    image_detail: str
    prompt_variant: str
    timeout_seconds: int


def _default_experiments() -> list[Experiment]:
    return [
        Experiment(
            name="none_tool_only_frames16",
            effort="none",
            reasoning_mode="guided",
            max_output_tokens=768,
            max_visual_frames=16,
            screenshot_size=900,
            image_detail="high",
            prompt_variant="tool_only",
            timeout_seconds=900,
        ),
        Experiment(
            name="low_tool_only_frames24",
            effort="low",
            reasoning_mode="guided",
            max_output_tokens=768,
            max_visual_frames=24,
            screenshot_size=900,
            image_detail="high",
            prompt_variant="tool_only",
            timeout_seconds=900,
        ),
        Experiment(
            name="low_traceable_frames40",
            effort="low",
            reasoning_mode="guided",
            max_output_tokens=1024,
            max_visual_frames=40,
            screenshot_size=1080,
            image_detail="high",
            prompt_variant="traceable",
            timeout_seconds=1200,
        ),
        Experiment(
            name="medium_traceable_frames40",
            effort="medium",
            reasoning_mode="guided",
            max_output_tokens=1280,
            max_visual_frames=40,
            screenshot_size=1080,
            image_detail="high",
            prompt_variant="traceable",
            timeout_seconds=1200,
        ),
        Experiment(
            name="high_traceable_frames40",
            effort="high",
            reasoning_mode="guided",
            max_output_tokens=1536,
            max_visual_frames=40,
            screenshot_size=1080,
            image_detail="high",
            prompt_variant="traceable",
            timeout_seconds=1500,
        ),
        Experiment(
            name="medium_fast_frames24",
            effort="medium",
            reasoning_mode="guided",
            max_output_tokens=1024,
            max_visual_frames=24,
            screenshot_size=900,
            image_detail="high",
            prompt_variant="fast_traceable",
            timeout_seconds=1200,
        ),
    ]


def _confirm_experiments() -> list[Experiment]:
    experiments: list[Experiment] = []

    def add(
        name: str,
        *,
        effort: str = "medium",
        reasoning_mode: str = "guided",
        tokens: int = 1024,
        frames: int = 24,
        size: int = 900,
        variant: str = "fast_traceable",
        timeout: int = 1200,
    ) -> None:
        experiments.append(
            Experiment(
                name=name,
                effort=effort,
                reasoning_mode=reasoning_mode,
                max_output_tokens=tokens,
                max_visual_frames=frames,
                screenshot_size=size,
                image_detail="high",
                prompt_variant=variant,
                timeout_seconds=timeout,
            )
        )

    add("medium_default_thinking_keyboard_s900", reasoning_mode="default", variant="keyboard_submit")
    add("low_default_thinking_keyboard_s900", effort="low", reasoning_mode="default", tokens=768, variant="keyboard_submit")
    add("off_keyboard_s900", effort="none", reasoning_mode="off", tokens=768, variant="keyboard_submit")
    add("medium_guided_keyboard_s900", variant="keyboard_submit")
    add("high_guided_keyboard_s900", effort="high", tokens=1536, variant="keyboard_submit", timeout=1500)
    add("medium_guided_keyboard_s720", variant="keyboard_submit", size=720)
    add("medium_guided_keyboard_s1080", variant="keyboard_submit", size=1080)

    for idx in range(1, 4):
        add(f"medium_fast_frames24_s900_repeat{idx}")

    add("low_fast_frames24_s900", effort="low", tokens=768)
    add("none_fast_frames24_s900", effort="none", tokens=768)
    add("high_fast_frames24_s900", effort="high", tokens=1536, timeout=1500)
    add("medium_fast_frames16_s900", frames=16)
    add("medium_fast_frames32_s900", frames=32)
    add("medium_fast_frames24_s720", size=720)
    add("medium_fast_frames24_s1080", size=1080)
    add("low_tool_only_frames24_s900", effort="low", tokens=768, variant="tool_only")
    add("medium_tool_only_frames24_s900", effort="medium", variant="tool_only")
    return experiments


def _overnight_experiments() -> list[Experiment]:
    experiments: list[Experiment] = []

    def add(
        name: str,
        *,
        effort: str = "none",
        reasoning_mode: str = "off",
        tokens: int = 768,
        frames: int = 24,
        size: int = 900,
        variant: str = "keyboard_submit",
        timeout: int = 1200,
    ) -> None:
        experiments.append(
            Experiment(
                name=name,
                effort=effort,
                reasoning_mode=reasoning_mode,
                max_output_tokens=tokens,
                max_visual_frames=frames,
                screenshot_size=size,
                image_detail="high",
                prompt_variant=variant,
                timeout_seconds=timeout,
            )
        )

    for idx in range(1, 7):
        add(f"off_keyboard_s900_repeat{idx}")
        add(
            f"medium_guided_keyboard_s900_repeat{idx}",
            effort="medium",
            reasoning_mode="guided",
            tokens=1024,
        )
        add(
            f"low_guided_keyboard_s900_repeat{idx}",
            effort="low",
            reasoning_mode="guided",
            tokens=768,
        )
        add(
            f"low_default_keyboard_s900_repeat{idx}",
            effort="low",
            reasoning_mode="default",
            tokens=768,
        )

    experiments.extend(
        [
            experiment
            for experiment in _confirm_experiments()
            if experiment.name
            in {
                "off_keyboard_s900",
                "medium_guided_keyboard_s900",
                "high_guided_keyboard_s900",
                "medium_guided_keyboard_s720",
                "medium_guided_keyboard_s1080",
                "low_default_thinking_keyboard_s900",
                "medium_default_thinking_keyboard_s900",
            }
        ]
    )
    for idx in range(1, 4):
        experiments.extend(
            [
                Experiment(
                    name=f"low_fast_frames16_s900_repeat{idx}",
                    effort="low",
                    reasoning_mode="guided",
                    max_output_tokens=768,
                    max_visual_frames=16,
                    screenshot_size=900,
                    image_detail="high",
                    prompt_variant="fast_traceable",
                    timeout_seconds=1200,
                ),
                Experiment(
                    name=f"medium_fast_frames24_s720_repeat{idx}",
                    effort="medium",
                    reasoning_mode="guided",
                    max_output_tokens=1024,
                    max_visual_frames=24,
                    screenshot_size=720,
                    image_detail="high",
                    prompt_variant="fast_traceable",
                    timeout_seconds=1200,
                ),
            ]
        )
    return experiments


def _experiments_for_preset(preset: str) -> list[Experiment]:
    if preset == "pilot":
        return _default_experiments()
    if preset == "confirm":
        return _confirm_experiments()
    if preset == "overnight":
        return _overnight_experiments()
    raise ValueError(f"unknown preset: {preset}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            records.append({"_raw": line})
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _qa_records_for_experiment(experiment: Experiment) -> list[dict[str, Any]]:
    digest = hashlib.sha1(experiment.name.encode("utf-8")).digest()
    a = 11 + digest[0] % 70
    b = 7 + digest[1] % 40
    c = 3 + digest[2] % 9
    d = 4 + digest[3] % 8
    english_pairs = [
        ("The opposite of hot is what?", "cold"),
        ("A baby cat is called what?", "kitten"),
        ("Water freezes into what?", "ice"),
        ("The day after Monday is what?", "Tuesday"),
        ("The color of grass is usually what?", "green"),
    ]
    question3, answer3 = english_pairs[digest[4] % len(english_pairs)]
    return [
        {
            "item_id": f"{experiment.name}_q1",
            "question": f"请只回答数字：{a} + {b} 等于几？",
            "expected_answer": str(a + b),
        },
        {
            "item_id": f"{experiment.name}_q2",
            "question": f"请只回答数字：{c} * {d} 等于几？",
            "expected_answer": str(c * d),
        },
        {
            "item_id": f"{experiment.name}_q3",
            "question": f"Please answer with only one English word: {question3}",
            "expected_answer": answer3,
        },
    ]


def _task_description(
    *,
    qa_path: Path,
    result_path: Path,
    prompt_variant: str,
) -> str:
    base = (
        "Goal: use Doubao Web to collect answers for 3 JSONL questions and "
        "write a JSONL result file.\n"
        "Target URL: https://www.doubao.com/chat/\n"
        f"Input JSONL file: {qa_path}\n"
        f"Output JSONL file: {result_path}\n\n"
        "Required workflow:\n"
        "1. First call read_text_file on the input JSONL file. Each line has "
        "item_id, question, expected_answer.\n"
        "2. Call open_url for the target URL, or continue if the Doubao page is "
        "already open.\n"
        "3. For each item, submit exactly the question text to Doubao, wait "
        "until the answer is stable, then record the result.\n"
        "4. For every item, call append_text_file and append one JSON object "
        "line with item_id, question, expected_answer, doubao_answer, status, "
        "reason. The append_text_file/write_text_file text argument must be a "
        "plain string containing JSONL text, not a nested JSON object.\n"
        "5. After all 3 items are recorded, call finish_subtask.\n"
        "6. Do not fabricate Doubao answers. If login, captcha, page block, "
        "or uncertainty prevents progress, call request_human_help.\n"
        "7. If one response contains multiple tool calls, each tool call must "
        "include index: 1, 2, 3... for execution order.\n"
        "8. The questions are unique to this experiment. Do not reuse answers "
        "from older browser history. A result is valid only after you submit "
        "the current question and see the answer for that current item.\n"
    )
    variants = {
        "tool_only": (
            "\nResponse policy: prefer tool calls only. Do not emit visible "
            "status text unless it is the final answer or a human request."
        ),
        "traceable": (
            "\nResponse policy: keep reasoning short but traceable. Visible "
            "text may contain one short previous-state and next-action line, "
            "then tool calls."
        ),
        "fast_traceable": (
            "\nResponse policy: one short clause only. If a tool can be called, "
            "call it immediately without prose."
        ),
        "keyboard_submit": (
            "\nResponse policy: use tools immediately. For text submission in "
            "a web or desktop chat box, prefer one computer_use call with "
            "indexed actions: click the input box, keypress CONTROL+A, type "
            "the exact text, then keypress ENTER. Do not split typing and "
            "submitting across separate model turns unless the previous tool "
            "result shows failure. After a visible answer appears for the "
            "current item, do not submit that same item again; record success "
            "or mismatch and move to the next item. Use visible text only for "
            "a very short state/action note."
        ),
    }
    return base + variants.get(prompt_variant, variants["traceable"])


def _load_events(trace_path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not trace_path.exists():
        return events
    for line in trace_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            events.append({"kind": "invalid_jsonl", "raw": line[:1000]})
    return events


def _response_output_items(response: dict[str, Any]) -> list[dict[str, Any]]:
    obj = response.get("response_object")
    if isinstance(obj, dict):
        output = obj.get("output")
        if isinstance(output, list):
            return [item for item in output if isinstance(item, dict)]
    return []


def _tool_calls_from_response(response: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for item in _response_output_items(response):
        if item.get("type") != "function_call":
            continue
        calls.append(
            {
                "name": item.get("name"),
                "call_id": item.get("call_id"),
                "arguments": item.get("arguments"),
            }
        )
    return calls


def _request_context_stats(response: dict[str, Any]) -> dict[str, Any]:
    request_context = response.get("request_context")
    if not isinstance(request_context, list):
        return {}
    image_count = 0
    video_count = 0
    text_chars = 0
    function_outputs = 0
    function_calls = 0
    roles: dict[str, int] = {}
    for item in request_context:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "")
        role = str(item.get("role") or item_type or "unknown")
        roles[role] = roles.get(role, 0) + 1
        if item_type == "function_call":
            function_calls += 1
        if item_type == "function_call_output":
            function_outputs += 1
        content = item.get("content")
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = str(part.get("type") or "")
                if part_type == "input_image":
                    image_count += 1
                elif part_type == "input_video":
                    video_count += 1
                elif part_type == "input_text":
                    text_chars += len(str(part.get("text") or ""))
        elif isinstance(content, str):
            text_chars += len(content)
    return {
        "request_items": len(request_context),
        "roles": roles,
        "image_count": image_count,
        "video_count": video_count,
        "text_chars": text_chars,
        "function_calls_in_context": function_calls,
        "function_outputs_in_context": function_outputs,
    }


def _summarize_trace(
    *,
    experiment: Experiment,
    trace_path: Path,
    result_path: Path,
    qa_path: Path,
    status: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    events = _load_events(trace_path)
    responses = [event["payload"] for event in events if event.get("kind") == "model_response"]
    steps = [event["payload"] for event in events if event.get("kind") == "step"]
    operations = [event["payload"] for event in events if event.get("kind") == "operation"]
    final_ops = [
        op for op in operations
        if op.get("kind") == "agent_run" and op.get("status") != "running"
    ]
    per_response: list[dict[str, Any]] = []
    for response in responses:
        per_response.append(
            {
                "step": response.get("step"),
                "latency_seconds": response.get("latency_seconds"),
                "input_tokens": response.get("input_tokens"),
                "output_tokens": response.get("output_tokens"),
                "total_tokens": response.get("total_tokens"),
                "reasoning_text_chars": response.get("reasoning_text_chars"),
                "reasoning_text": response.get("reasoning_text") or "",
                "output_text_chars": response.get("output_text_chars"),
                "output_text": response.get("output_text") or "",
                "tool_argument_chars": response.get("tool_argument_chars"),
                "function_call_count": response.get("function_call_count"),
                "warnings": response.get("warnings") or [],
                "tool_calls": _tool_calls_from_response(response),
                "request_context_stats": _request_context_stats(response),
            }
        )
    per_step: list[dict[str, Any]] = []
    for step in steps:
        action = step.get("action") if isinstance(step.get("action"), dict) else {}
        metadata = step.get("metadata") if isinstance(step.get("metadata"), dict) else {}
        per_step.append(
            {
                "step_id": step.get("step_id"),
                "action": action.get("name") or action.get("type"),
                "action_payload": action,
                "result": step.get("result"),
                "summary": step.get("summary"),
                "tool_output": metadata.get("output"),
                "verification": metadata.get("verification"),
            }
        )
    result_records = _read_jsonl(result_path)
    qa_records = _read_jsonl(qa_path)
    validation = _validate_submission_evidence(
        steps=per_step,
        qa_records=qa_records,
        result_records=result_records,
    )
    latencies = [
        float(item.get("latency_seconds") or 0)
        for item in per_response
    ]
    return {
        "experiment": experiment.__dict__,
        "status": status,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "trace_path": str(trace_path),
        "result_path": str(result_path),
        "qa_path": str(qa_path),
        "qa_records": qa_records,
        "final_operation": final_ops[-1] if final_ops else None,
        "event_count": len(events),
        "response_count": len(per_response),
        "step_count": len(per_step),
        "result_records": result_records,
        "validation": validation,
        "latency_total_seconds": round(sum(latencies), 3),
        "latency_avg_seconds": round(sum(latencies) / len(latencies), 3) if latencies else 0,
        "latency_max_seconds": round(max(latencies), 3) if latencies else 0,
        "reasoning_chars_total": sum(int(item.get("reasoning_text_chars") or 0) for item in per_response),
        "reasoning_chars_avg": round(
            sum(int(item.get("reasoning_text_chars") or 0) for item in per_response)
            / len(per_response),
            1,
        )
        if per_response
        else 0,
        "output_chars_total": sum(int(item.get("output_text_chars") or 0) for item in per_response),
        "output_chars_avg": round(
            sum(int(item.get("output_text_chars") or 0) for item in per_response)
            / len(per_response),
            1,
        )
        if per_response
        else 0,
        "tool_argument_chars_total": sum(int(item.get("tool_argument_chars") or 0) for item in per_response),
        "tool_argument_chars_avg": round(
            sum(int(item.get("tool_argument_chars") or 0) for item in per_response)
            / len(per_response),
            1,
        )
        if per_response
        else 0,
        "responses": per_response,
        "steps": per_step,
    }


def _validate_submission_evidence(
    *,
    steps: list[dict[str, Any]],
    qa_records: list[dict[str, Any]],
    result_records: list[dict[str, Any]],
) -> dict[str, Any]:
    action_texts: list[str] = []
    computer_action_count = 0
    text_action_count = 0
    submit_key_count = 0
    for step in steps:
        action_payload = step.get("action_payload")
        if not isinstance(action_payload, dict):
            continue
        if action_payload.get("type") == "computer_call":
            actions = action_payload.get("actions")
            if isinstance(actions, list):
                for action in actions:
                    if not isinstance(action, dict):
                        continue
                    computer_action_count += 1
                    action_type = str(action.get("type") or "")
                    text = str(action.get("text") or action.get("keys") or "")
                    action_texts.append(json.dumps(action, ensure_ascii=False, sort_keys=True))
                    if action_type in {"type", "text", "paste"} and text.strip():
                        text_action_count += 1
                    if action_type in {"keypress", "key", "hotkey"} and "ENTER" in text.upper():
                        submit_key_count += 1
        else:
            action_texts.append(json.dumps(action_payload, ensure_ascii=False, sort_keys=True))

    full_action_text = "\n".join(action_texts)
    submitted_item_ids: list[str] = []
    for record in qa_records:
        question = str(record.get("question") or "")
        if question and question in full_action_text:
            submitted_item_ids.append(str(record.get("item_id") or ""))

    expected_ids = {str(record.get("item_id") or "") for record in qa_records}
    result_ids = {str(record.get("item_id") or "") for record in result_records}
    success_records = [
        record for record in result_records
        if str(record.get("status") or "").strip().lower() == "success"
    ]
    mismatch_records = [
        record for record in result_records
        if str(record.get("status") or "").strip().lower() == "mismatch"
    ]
    all_results_present = expected_ids <= result_ids if expected_ids else False
    all_questions_submitted = expected_ids <= set(submitted_item_ids) if expected_ids else False
    suspicious_without_submission = (
        bool(expected_ids)
        and all_results_present
        and not all_questions_submitted
    )
    return {
        "expected_item_count": len(expected_ids),
        "result_item_count": len(result_ids),
        "result_success_count": len(success_records),
        "result_mismatch_count": len(mismatch_records),
        "all_results_present": all_results_present,
        "submitted_item_ids": submitted_item_ids,
        "submitted_question_count": len(set(submitted_item_ids)),
        "all_questions_submitted_by_computer_action": all_questions_submitted,
        "computer_action_count": computer_action_count,
        "computer_text_action_count": text_action_count,
        "computer_submit_key_count": submit_key_count,
        "suspicious_without_submission": suspicious_without_submission,
    }


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    exp = summary["experiment"]
    lines.append(f"# GUI Agent Reasoning Experiment: {exp['name']}")
    lines.append("")
    lines.append(f"- status: `{summary['status']}`")
    lines.append(f"- reasoning_mode: `{exp.get('reasoning_mode', 'guided')}`")
    lines.append(f"- effort: `{exp['effort']}`")
    lines.append(f"- prompt_variant: `{exp['prompt_variant']}`")
    lines.append(f"- max_output_tokens: `{exp['max_output_tokens']}`")
    lines.append(f"- max_visual_frames: `{exp['max_visual_frames']}`")
    lines.append(f"- screenshot_size: `{exp['screenshot_size']}`")
    lines.append(f"- trace: `{summary['trace_path']}`")
    lines.append(f"- result: `{summary['result_path']}`")
    lines.append(f"- qa: `{summary['qa_path']}`")
    lines.append(
        f"- responses: `{summary['response_count']}`, steps: `{summary['step_count']}`, "
        f"avg latency: `{summary['latency_avg_seconds']}s`"
    )
    lines.append(f"- validation: `{json.dumps(summary['validation'], ensure_ascii=False)}`")
    lines.append("")
    lines.append("## Result Records")
    lines.append("")
    if summary["result_records"]:
        for record in summary["result_records"]:
            lines.append(f"- `{json.dumps(record, ensure_ascii=False)}`")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Model Responses")
    lines.append("")
    for response in summary["responses"]:
        lines.append(f"### Step {response['step']}")
        lines.append("")
        lines.append(f"- latency: `{response['latency_seconds']}s`")
        lines.append(
            f"- tokens: input `{response['input_tokens']}`, output `{response['output_tokens']}`, total `{response['total_tokens']}`"
        )
        lines.append(f"- reasoning chars: `{response['reasoning_text_chars']}`")
        lines.append(f"- visible output chars: `{response['output_text_chars']}`")
        lines.append(f"- tool argument chars: `{response['tool_argument_chars']}`")
        lines.append(f"- warnings: `{response['warnings']}`")
        lines.append(f"- request context stats: `{json.dumps(response['request_context_stats'], ensure_ascii=False)}`")
        lines.append("")
        lines.append("Reasoning:")
        lines.append("```text")
        lines.append(response["reasoning_text"])
        lines.append("```")
        lines.append("")
        lines.append("Visible Output:")
        lines.append("```text")
        lines.append(response["output_text"])
        lines.append("```")
        lines.append("")
        lines.append("Tool Calls:")
        lines.append("```json")
        lines.append(json.dumps(response["tool_calls"], ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
    lines.append("## Tool Results")
    lines.append("")
    for step in summary["steps"]:
        lines.append(f"### Step Record {step['step_id']}")
        lines.append("```json")
        lines.append(json.dumps(step, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _wait_for_completion(
    *,
    process: subprocess.Popen[bytes],
    trace_path: Path,
    timeout_seconds: int,
) -> str:
    deadline = time.time() + timeout_seconds
    last_response_count = 0
    last_activity = time.time()
    while time.time() < deadline:
        events = _load_events(trace_path)
        response_count = sum(1 for event in events if event.get("kind") == "model_response")
        if response_count != last_response_count:
            last_response_count = response_count
            last_activity = time.time()
        operations = [event.get("payload", {}) for event in events if event.get("kind") == "operation"]
        for op in reversed(operations):
            if op.get("kind") == "agent_run" and op.get("status") != "running":
                return str(op.get("status") or "completed")
        if process.poll() is not None:
            return f"client_exited_{process.returncode}"
        if last_response_count > 0 and time.time() - last_activity > 420:
            return "stalled"
        time.sleep(5)
    return "timeout"


def _run_one(
    *,
    experiment: Experiment,
    root: Path,
    out_dir: Path,
    qa_path: Path,
    static_qa: bool,
    python_exe: str,
    base_url: str,
    model: str,
) -> dict[str, Any]:
    result_path = out_dir / f"results_{experiment.name}.jsonl"
    experiment_qa_path = qa_path if static_qa else out_dir / f"qa_{experiment.name}.jsonl"
    trace_path = out_dir / f"{experiment.name}_trace.jsonl"
    state_path = out_dir / f"{experiment.name}_state.json"
    stdout_path = out_dir / f"{experiment.name}_client.out.log"
    stderr_path = out_dir / f"{experiment.name}_client.err.log"
    for path in (result_path, trace_path, state_path, stdout_path, stderr_path):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    if not static_qa:
        _write_jsonl(experiment_qa_path, _qa_records_for_experiment(experiment))
    task = _task_description(
        qa_path=experiment_qa_path,
        result_path=result_path,
        prompt_variant=experiment.prompt_variant,
    )
    webbrowser.open("https://www.doubao.com/chat/", new=0, autoraise=True)
    time.sleep(2)
    args = [
        python_exe,
        "-m",
        "cfie_gui_agent.desktop_client",
        "--app-name",
        f"Doubao sweep {experiment.name}",
        "--state-path",
        str(state_path),
        "--trace-path",
        str(trace_path),
        "--task-description",
        task,
        "--process-name",
        "chrome.exe",
        "--window-title-pattern",
        ".*(豆包|Doubao).*",
        "--base-url",
        base_url,
        "--model",
        model,
        "--reasoning-mode",
        experiment.reasoning_mode,
        "--reasoning-effort",
        experiment.effort,
        "--max-output-tokens",
        str(experiment.max_output_tokens),
        "--max-steps",
        "32",
        "--screenshot-max-width",
        str(experiment.screenshot_size),
        "--screenshot-max-height",
        str(experiment.screenshot_size),
        "--max-visual-frames",
        str(experiment.max_visual_frames),
        "--image-detail",
        experiment.image_detail,
        "--tool-profile",
        "core",
        "--no-load-trace",
        "--auto-start",
    ]
    start = time.time()
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(args, cwd=root, stdout=stdout, stderr=stderr)
    status = _wait_for_completion(
        process=process,
        trace_path=trace_path,
        timeout_seconds=experiment.timeout_seconds,
    )
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
    elapsed = time.time() - start
    summary = _summarize_trace(
        experiment=experiment,
        trace_path=trace_path,
        result_path=result_path,
        qa_path=experiment_qa_path,
        status=status,
        elapsed_seconds=elapsed,
    )
    summary["stdout_path"] = str(stdout_path)
    summary["stderr_path"] = str(stderr_path)
    json_path = out_dir / f"{experiment.name}_summary.json"
    md_path = out_dir / f"{experiment.name}_summary.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(summary, md_path)
    return summary


def _write_index(summaries: list[dict[str, Any]], path: Path) -> None:
    lines = ["# GUI Agent Reasoning Sweep", ""]
    lines.append(f"generated_at: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append("")
    lines.append("| Experiment | Status | Mode | Effort | Prompt | Frames | Size | Responses | Steps | Results | OK | Mismatch | Submitted | Suspicious | Avg Latency |")
    lines.append("|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|")
    for summary in summaries:
        exp = summary["experiment"]
        validation = summary.get("validation") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{exp['name']}`",
                    f"`{summary['status']}`",
                    f"`{exp.get('reasoning_mode', 'guided')}`",
                    f"`{exp['effort']}`",
                    f"`{exp['prompt_variant']}`",
                    str(exp["max_visual_frames"]),
                    str(exp["screenshot_size"]),
                    str(summary["response_count"]),
                    str(summary["step_count"]),
                    str(len(summary["result_records"])),
                    str(validation.get("result_success_count", 0)),
                    str(validation.get("result_mismatch_count", 0)),
                    str(validation.get("submitted_question_count", 0)),
                    "`yes`" if validation.get("suspicious_without_submission") else "`no`",
                    f"{summary['latency_avg_seconds']}",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=".bench_logs/gui_agent_reasoning_sweep")
    parser.add_argument("--qa-path", default=".bench_logs/gui_agent_doubao_reasoning/qa_items.jsonl")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen35-vl")
    parser.add_argument("--python", default=None)
    parser.add_argument(
        "--preset",
        choices=("pilot", "confirm", "overnight"),
        default="pilot",
        help="Experiment matrix to run.",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated experiment names to run from the selected preset.",
    )
    parser.add_argument(
        "--static-qa",
        action="store_true",
        help="Use --qa-path directly instead of generating per-experiment QA files.",
    )
    args = parser.parse_args()

    root = Path.cwd()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    qa_path = (root / args.qa_path).resolve()
    python_exe = args.python or str(root.parent / ".venv" / "Scripts" / "python.exe")
    experiments = _experiments_for_preset(args.preset)
    if args.quick:
        experiments = experiments[:2]
    if args.only.strip():
        wanted = {name.strip() for name in args.only.split(",") if name.strip()}
        experiments = [experiment for experiment in experiments if experiment.name in wanted]
        missing = wanted - {experiment.name for experiment in experiments}
        if missing:
            raise SystemExit(f"unknown experiment name(s): {', '.join(sorted(missing))}")

    summaries: list[dict[str, Any]] = []
    progress_path = out_dir / "sweep_progress.jsonl"
    for experiment in experiments:
        progress_path.write_text("", encoding="utf-8") if not progress_path.exists() else None
        progress = {
            "event": "start",
            "experiment": experiment.__dict__,
            "time": datetime.now().isoformat(timespec="seconds"),
        }
        with progress_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(progress, ensure_ascii=False) + "\n")
        summary = _run_one(
            experiment=experiment,
            root=root,
            out_dir=out_dir,
            qa_path=qa_path,
            static_qa=args.static_qa,
            python_exe=python_exe,
            base_url=args.base_url,
            model=args.model,
        )
        summaries.append(summary)
        with progress_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "event": "finish",
                        "experiment": experiment.name,
                        "status": summary["status"],
                        "responses": summary["response_count"],
                        "steps": summary["step_count"],
                        "results": len(summary["result_records"]),
                        "time": datetime.now().isoformat(timespec="seconds"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        _write_index(summaries, out_dir / "sweep_summary.md")
        (out_dir / "sweep_summary.json").write_text(
            json.dumps(summaries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
