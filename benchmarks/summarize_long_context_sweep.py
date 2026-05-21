from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path


def read_exit_code(path: Path) -> str:
    exit_path = path.with_suffix(".exitcode")
    if not exit_path.exists():
        return "missing"
    return exit_path.read_text(encoding="utf-8", errors="replace").strip()


def read_peak_gpu_mib(path: Path) -> tuple[int | None, int | None]:
    csv_path = path.with_suffix(".nvidia_smi.csv")
    if not csv_path.exists():
        return None, None
    peak_mem = None
    peak_util = None
    with csv_path.open(encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            try:
                mem = int(row["memory.used.MiB"].strip())
                util = int(row["utilization.gpu.%"].strip())
            except Exception:
                continue
            peak_mem = mem if peak_mem is None else max(peak_mem, mem)
            peak_util = util if peak_util is None else max(peak_util, util)
    return peak_mem, peak_util


def grep_first(path: Path, pattern: str) -> str:
    if not path.exists():
        return ""
    regex = re.compile(pattern)
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if regex.search(line):
            return line.strip()
    return ""


def grep_last(path: Path, pattern: str) -> str:
    if not path.exists():
        return ""
    regex = re.compile(pattern)
    result = ""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if regex.search(line):
            result = line.strip()
    return result


def summarize_json(json_path: Path) -> list[dict[str, str]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    peak_mem, peak_util = read_peak_gpu_mib(json_path)
    rows: list[dict[str, str]] = []
    by_input = data.get("results_by_input_tokens") or {}
    for input_tokens, result in sorted(by_input.items(), key=lambda x: int(x[0])):
        rows.append(
            {
                "run": json_path.stem,
                "status": "ok",
                "max_model_len": str(data.get("max_model_len")),
                "kv_gib": f"{(data.get('kv_cache_memory_bytes') or 0) / (1 << 30):.2f}",
                "max_batched": str(data.get("effective_max_num_batched_tokens")),
                "prefill": input_tokens,
                "cold_ttft_s": f"{result.get('cold_prefill_to_first_token_seconds', 0):.3f}",
                "warm_ttft_s": f"{result.get('warm_prefill_to_first_token_seconds', 0):.3f}",
                "cold_decode_tps": f"{result.get('cold_decode_tokens_per_sec_after_first_step', 0):.3f}",
                "warm_decode_tps": f"{result.get('warm_decode_tokens_per_sec_after_first_step', 0):.3f}",
                "output_ok": str(
                    result.get("all_outputs_nonempty")
                    and not result.get("any_output_has_replacement_char")
                ),
                "peak_mib": str(peak_mem or ""),
                "peak_util": str(peak_util or ""),
                "note": "",
            }
        )
    return rows


def summarize_failure(stem_path: Path) -> dict[str, str]:
    stderr_path = stem_path.with_suffix(".stderr.log")
    peak_mem, peak_util = read_peak_gpu_mib(stem_path)
    kv_line = grep_first(stderr_path, r"Hybrid GPU KV cache estimated")
    error_line = grep_last(stderr_path, r"(ValueError:|RuntimeError:|CUDA error|out of memory|EngineDeadError)")
    return {
        "run": stem_path.name,
        "status": read_exit_code(stem_path),
        "max_model_len": "",
        "kv_gib": "",
        "max_batched": "",
        "prefill": "",
        "cold_ttft_s": "",
        "warm_ttft_s": "",
        "cold_decode_tps": "",
        "warm_decode_tps": "",
        "output_ok": "",
        "peak_mib": str(peak_mem or ""),
        "peak_util": str(peak_util or ""),
        "note": kv_line or error_line,
    }


def markdown_table(rows: list[dict[str, str]]) -> str:
    headers = [
        "run",
        "status",
        "max_model_len",
        "kv_gib",
        "max_batched",
        "prefill",
        "cold_ttft_s",
        "warm_ttft_s",
        "cold_decode_tps",
        "warm_decode_tps",
        "output_ok",
        "peak_mib",
        "peak_util",
        "note",
    ]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        values = [str(row.get(header, "")).replace("|", "\\|") for header in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: summarize_long_context_sweep.py LOGDIR")
    logdir = Path(sys.argv[1])
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for json_path in sorted(logdir.glob("*.json")):
        seen.add(json_path.stem)
        rows.extend(summarize_json(json_path))
    for exit_path in sorted(logdir.glob("*.exitcode")):
        stem_path = exit_path.with_suffix("")
        if stem_path.name in seen:
            continue
        rows.append(summarize_failure(stem_path))
    print(markdown_table(rows))


if __name__ == "__main__":
    main()
