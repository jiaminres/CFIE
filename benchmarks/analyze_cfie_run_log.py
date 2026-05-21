# SPDX-License-Identifier: Apache-2.0
"""Summarize CFIE 122B benchmark logs.

The parser is intentionally conservative: vLLM logs only carry second-level
timestamps, so per-layer attach timings are approximate but still useful for
spotting the slow startup phases.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Iterable


TS_RE = re.compile(r"\b(?P<month>\d{2})-(?P<day>\d{2}) (?P<hms>\d{2}:\d{2}:\d{2})\b")
LAYER_RE = re.compile(r"layers\.(?P<layer>\d+)\.mlp\.experts")
WEIGHTS_RE = re.compile(r"Loading weights took (?P<seconds>[0-9.]+) seconds")
MODEL_LOADING_RE = re.compile(
    r"Model loading took (?P<gib>[0-9.]+) GiB memory and (?P<seconds>[0-9.]+) seconds"
)
GRAPH_RE = re.compile(
    r"Graph capturing finished in (?P<seconds>[0-9.]+) secs, took (?P<gib>[0-9.]+) GiB"
)
KV_RE = re.compile(
    r"Hybrid GPU KV cache estimated single-request max context: "
    r"(?P<tokens>>= [0-9,]+|[0-9,]+) tokens with (?P<gib>[0-9.]+) GiB"
)
PINNED_RE = re.compile(
    r"pinned_static=(?P<pinned>[0-9.]+) MiB pageable_static=(?P<pageable>[0-9.]+) MiB"
)
PREPARE_RE = re.compile(
    r"CFIE_PREPARE_STATS layer=.*?layers\.(?P<layer>\d+)\.mlp\.experts "
    r"step=(?P<step>\d+) requested=(?P<requested>\d+) final_requested=(?P<final>\d+) "
    r"staged=(?P<staged>\d+) stage_resident_hit=(?P<hit>\d+)% "
    r"missing=(?P<missing>\d+) load_missing=(?P<load_missing>\d+)"
)
TIMING_RE = re.compile(
    r"CFIE_BENCH_TIMING prepare layer=.*?layers\.(?P<layer>\d+)\.mlp\.experts "
    r"step=(?P<step>\d+) plan=(?P<plan>[0-9.]+)ms "
    r"stage=(?P<stage>[0-9.]+)ms "
    r"stage_materialize=(?P<stage_materialize>[0-9.]+)ms "
    r"stage_write=(?P<stage_write>[0-9.]+)ms "
    r"stage_cpu_pack=(?P<stage_cpu_pack>[0-9.]+)ms "
    r"stage_h2d=(?P<stage_h2d>[0-9.]+)ms "
    r"stage_gpu_scatter=(?P<stage_gpu_scatter>[0-9.]+)ms "
    r"stage_install=(?P<stage_install>[0-9.]+)ms "
    r"total=(?P<total>[0-9.]+)ms"
)
TIMING_FIELDS = (
    "plan",
    "stage",
    "stage_materialize",
    "stage_write",
    "stage_cpu_pack",
    "stage_h2d",
    "stage_gpu_scatter",
    "stage_install",
    "total",
)
RUNNER_RE = re.compile(
    r"CFIE_BENCH_TIMING layer=.*?layers\.(?P<layer>\d+)\.mlp\.experts "
    r"prepare=(?P<prepare>[0-9.]+)s "
    r"prepare_sync=(?P<prepare_sync>[0-9.]+)s "
    r"apply=(?P<apply>[0-9.]+)s "
    r"apply_sync=(?P<apply_sync>[0-9.]+)s"
)
RUNNER_FIELDS = ("prepare", "prepare_sync", "apply", "apply_sync", "compute_sync")


@dataclass
class LayerAttach:
    layer: int
    load_ts: datetime | None = None
    pool_ts: datetime | None = None
    pinned_mib: float = 0.0
    pageable_mib: float = 0.0
    load_gap_s: float | None = None
    pool_gap_s: float | None = None


def _timestamp(line: str) -> datetime | None:
    match = TS_RE.search(line)
    if match is None:
        return None
    return datetime.strptime(
        f"2000-{match.group('month')}-{match.group('day')} {match.group('hms')}",
        "%Y-%m-%d %H:%M:%S",
    )


def _read_json_sidecar(path: Path) -> dict | None:
    sidecar = path.with_suffix(".json")
    if not sidecar.exists():
        return None
    try:
        return json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return None


def _fmt(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    return f"{seconds:.1f}s"


def _avg(values: Iterable[float]) -> float:
    values = list(values)
    return mean(values) if values else 0.0


def analyze(path: Path) -> str:
    layers: dict[int, LayerAttach] = {}
    prepare_samples: dict[int, list[dict[str, float]]] = defaultdict(list)
    timing_by_step: dict[int, dict[str, float]] = defaultdict(
        lambda: {field: 0.0 for field in TIMING_FIELDS}
    )
    timing_sample_count_by_step: dict[int, int] = defaultdict(int)
    runner_groups: list[dict[str, float]] = []
    current_runner_group: dict[str, float] | None = None
    previous_runner_layer: int | None = None
    prepare_start: datetime | None = None
    last_pool_ts: datetime | None = None
    weights_seconds: float | None = None
    model_loading_seconds: float | None = None
    model_loading_gib: float | None = None
    graph_seconds: float | None = None
    graph_gib: float | None = None
    kv_context: str | None = None
    kv_gib: float | None = None

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        ts = _timestamp(line)

        if "Loading weights took" in line:
            match = WEIGHTS_RE.search(line)
            if match:
                weights_seconds = float(match.group("seconds"))

        if "Preparing tiered MoE expert cache attachment" in line and ts is not None:
            prepare_start = ts
            last_pool_ts = ts

        if "Loaded Marlin-ready expert cache" in line and ts is not None:
            layer_match = LAYER_RE.search(line)
            if layer_match:
                layer = int(layer_match.group("layer"))
                item = layers.setdefault(layer, LayerAttach(layer=layer))
                item.load_ts = ts
                if last_pool_ts is not None:
                    item.load_gap_s = (ts - last_pool_ts).total_seconds()

        if "Initialized fixed CPU expert pool" in line and ts is not None:
            layer_match = LAYER_RE.search(line)
            if layer_match:
                layer = int(layer_match.group("layer"))
                item = layers.setdefault(layer, LayerAttach(layer=layer))
                item.pool_ts = ts
                if item.load_ts is not None:
                    item.pool_gap_s = (ts - item.load_ts).total_seconds()
                pinned_match = PINNED_RE.search(line)
                if pinned_match:
                    item.pinned_mib = float(pinned_match.group("pinned"))
                    item.pageable_mib = float(pinned_match.group("pageable"))
                last_pool_ts = ts

        if "Model loading took" in line:
            match = MODEL_LOADING_RE.search(line)
            if match:
                model_loading_gib = float(match.group("gib"))
                model_loading_seconds = float(match.group("seconds"))

        if "Hybrid GPU KV cache estimated single-request max context" in line:
            match = KV_RE.search(line)
            if match:
                kv_context = match.group("tokens")
                kv_gib = float(match.group("gib"))

        if "Graph capturing finished" in line:
            match = GRAPH_RE.search(line)
            if match:
                graph_seconds = float(match.group("seconds"))
                graph_gib = float(match.group("gib"))

        if "CFIE_PREPARE_STATS" in line:
            match = PREPARE_RE.search(line)
            if match:
                layer = int(match.group("layer"))
                prepare_samples[layer].append(
                    {
                        "step": float(match.group("step")),
                        "hit": float(match.group("hit")),
                        "missing": float(match.group("missing")),
                        "staged": float(match.group("staged")),
                        "requested": float(match.group("requested")),
                    }
                )

        if "CFIE_BENCH_TIMING prepare" in line:
            match = TIMING_RE.search(line)
            if match:
                step = int(match.group("step"))
                timing_sample_count_by_step[step] += 1
                for field in TIMING_FIELDS:
                    timing_by_step[step][field] += float(match.group(field))

        if "CFIE_BENCH_TIMING layer=" in line:
            match = RUNNER_RE.search(line)
            if match:
                layer = int(match.group("layer"))
                if (
                    current_runner_group is None
                    or (
                        previous_runner_layer is not None
                        and layer <= previous_runner_layer
                    )
                ):
                    current_runner_group = {field: 0.0 for field in RUNNER_FIELDS}
                    current_runner_group["samples"] = 0.0
                    runner_groups.append(current_runner_group)
                prepare = float(match.group("prepare"))
                prepare_sync = float(match.group("prepare_sync"))
                apply = float(match.group("apply"))
                apply_sync = float(match.group("apply_sync"))
                current_runner_group["prepare"] += prepare
                current_runner_group["prepare_sync"] += prepare_sync
                current_runner_group["apply"] += apply
                current_runner_group["apply_sync"] += apply_sync
                current_runner_group["compute_sync"] += max(
                    apply_sync - prepare_sync, 0.0
                )
                current_runner_group["samples"] += 1.0
                previous_runner_layer = layer

    sidecar = _read_json_sidecar(path)
    layer_items = [layers[k] for k in sorted(layers)]
    attach_total = None
    if prepare_start is not None and layer_items and layer_items[-1].pool_ts:
        attach_total = (layer_items[-1].pool_ts - prepare_start).total_seconds()
    pinned_layers = [x.layer for x in layer_items if x.pinned_mib > 0]
    pageable_layers = [x.layer for x in layer_items if x.pageable_mib > 0]
    pinned_gib = sum(x.pinned_mib for x in layer_items) / 1024
    pageable_gib = sum(x.pageable_mib for x in layer_items) / 1024
    load_gap_total = sum(x.load_gap_s or 0 for x in layer_items)
    pool_gap_total = sum(x.pool_gap_s or 0 for x in layer_items)

    lines: list[str] = []
    lines.append(f"# CFIE Run Log Summary: {path.name}")
    lines.append("")
    lines.append("## Startup")
    lines.append(f"- Dense weights load: {_fmt(weights_seconds)}")
    lines.append(f"- Expert cache attach window: {_fmt(attach_total)}")
    lines.append(f"- Approx expert cache file/read gaps: {_fmt(load_gap_total)}")
    lines.append(f"- Approx CPU pool init gaps: {_fmt(pool_gap_total)}")
    if model_loading_seconds is not None:
        lines.append(
            f"- Model loading total: {model_loading_seconds:.1f}s, GPU memory {model_loading_gib:.2f} GiB"
        )
    if graph_seconds is not None:
        lines.append(f"- CUDA graph capture: {graph_seconds:.1f}s, {graph_gib:.2f} GiB")
    if kv_context is not None:
        lines.append(
            f"- Hybrid KV estimated single-request context: {kv_context} tokens "
            f"with {kv_gib:.2f} GiB KV tensors"
        )
    lines.append(
        f"- Static expert mirror: pinned {pinned_gib:.2f} GiB across {len(pinned_layers)} layers; "
        f"pageable {pageable_gib:.2f} GiB across {len(pageable_layers)} layers"
    )
    if pinned_layers:
        lines.append(f"- Pinned layer range: {pinned_layers[0]}..{pinned_layers[-1]}")
    if pageable_layers:
        lines.append(f"- Pageable layer range: {pageable_layers[0]}..{pageable_layers[-1]}")

    if layer_items:
        slow_loads = sorted(layer_items, key=lambda x: x.load_gap_s or 0, reverse=True)[:6]
        slow_pools = sorted(layer_items, key=lambda x: x.pool_gap_s or 0, reverse=True)[:6]
        lines.append("")
        lines.append("## Slowest Attach Gaps")
        lines.append("- Cache/read gaps: " + ", ".join(
            f"L{x.layer}:{_fmt(x.load_gap_s)}" for x in slow_loads
        ))
        lines.append("- CPU pool gaps: " + ", ".join(
            f"L{x.layer}:{_fmt(x.pool_gap_s)}" for x in slow_pools
        ))

    if prepare_samples:
        all_hits = [sample["hit"] for samples in prepare_samples.values() for sample in samples]
        all_missing = [
            sample["missing"] for samples in prepare_samples.values() for sample in samples
        ]
        lines.append("")
        lines.append("## Prepare Runtime")
        lines.append(
            f"- Logged samples: {sum(len(x) for x in prepare_samples.values())}; "
            f"avg resident hit {mean(all_hits):.1f}%; avg missing experts {mean(all_missing):.2f}/layer"
        )
        for start in [0, 16, 32]:
            end = start + 15
            bucket = [
                sample
                for layer, samples in prepare_samples.items()
                if start <= layer <= end
                for sample in samples
            ]
            if bucket:
                lines.append(
                    f"- Layers {start:02d}-{end:02d}: "
                    f"avg hit {_avg(s['hit'] for s in bucket):.1f}%, "
                    f"avg missing {_avg(s['missing'] for s in bucket):.2f}"
                )
        per_layer_missing = {
            layer: mean(sample["missing"] for sample in samples)
            for layer, samples in prepare_samples.items()
        }
        worst = sorted(per_layer_missing.items(), key=lambda x: x[1], reverse=True)[:8]
        best = sorted(per_layer_missing.items(), key=lambda x: x[1])[:8]
        lines.append("- Worst missing layers: " + ", ".join(f"L{k}:{v:.2f}" for k, v in worst))
        lines.append("- Best missing layers: " + ", ".join(f"L{k}:{v:.2f}" for k, v in best))

    if timing_by_step:
        lines.append("")
        lines.append("## Prepare Timing")
        ordered_steps = sorted(timing_by_step)
        decode_steps = int(sidecar.get("steps", 0)) if sidecar is not None else 0
        if decode_steps > 0 and len(ordered_steps) >= decode_steps:
            selected_steps = ordered_steps[-decode_steps:]
            selected_label = f"last {decode_steps} step groups"
        else:
            selected_steps = ordered_steps
            selected_label = "all step groups"

        first_step = selected_steps[0]
        steady_steps = selected_steps[1:] if len(selected_steps) > 1 else selected_steps

        def _sum_for_steps(steps: list[int], field: str) -> float:
            return sum(timing_by_step[step][field] for step in steps)

        def _avg_step(field: str, steps: list[int]) -> float:
            if not steps:
                return 0.0
            return _sum_for_steps(steps, field) / len(steps)

        lines.append(
            f"- Parsed step groups: {len(ordered_steps)}; reporting {selected_label}; "
            f"prepare samples in first reported step: {timing_sample_count_by_step[first_step]}"
        )
        lines.append(
            "- First reported step prepare total: "
            + ", ".join(
                f"{field}={timing_by_step[first_step][field]:.1f}ms"
                for field in ("plan", "stage", "stage_h2d", "stage_gpu_scatter", "total")
            )
        )
        lines.append(
            "- Average steady prepare per generated token: "
            + ", ".join(
                f"{field}={_avg_step(field, steady_steps):.1f}ms"
                for field in (
                    "plan",
                    "stage_write",
                    "stage_cpu_pack",
                    "stage_h2d",
                    "stage_gpu_scatter",
                    "stage_install",
                    "total",
                )
            )
        )

    if runner_groups:
        lines.append("")
        lines.append("## MoE Runner Timing")
        decode_steps = int(sidecar.get("steps", 0)) if sidecar is not None else 0
        if decode_steps > 0 and len(runner_groups) >= decode_steps:
            selected_groups = runner_groups[-decode_steps:]
            selected_label = f"last {decode_steps} layer groups"
        else:
            selected_groups = runner_groups
            selected_label = "all layer groups"
        first_group = selected_groups[0]
        steady_groups = selected_groups[1:] if len(selected_groups) > 1 else selected_groups

        def _avg_group(field: str, groups: list[dict[str, float]]) -> float:
            if not groups:
                return 0.0
            return sum(group[field] for group in groups) / len(groups)

        lines.append(
            f"- Parsed layer groups: {len(runner_groups)}; reporting {selected_label}; "
            f"layers in first reported group: {int(first_group.get('samples', 0))}"
        )
        lines.append(
            "- First reported group: "
            f"prepare_sync={first_group['prepare_sync'] * 1000:.1f}ms, "
            f"compute_sync={first_group['compute_sync'] * 1000:.1f}ms, "
            f"apply_sync_total={first_group['apply_sync'] * 1000:.1f}ms"
        )
        lines.append(
            "- Average steady MoE runner per generated token: "
            f"prepare_sync={_avg_group('prepare_sync', steady_groups) * 1000:.1f}ms, "
            f"compute_sync={_avg_group('compute_sync', steady_groups) * 1000:.1f}ms, "
            f"apply_sync_total={_avg_group('apply_sync', steady_groups) * 1000:.1f}ms"
        )

    if sidecar is not None:
        lines.append("")
        lines.append("## Result JSON")
        for key in [
            "tokens_per_sec",
            "steady_tokens_per_sec",
            "first_step_seconds",
            "engine_init_seconds",
            "cold_first_step_seconds",
            "warm_first_step_seconds_avg",
            "gpu_slots_per_layer",
            "prefill_burst_slots",
            "cpu_static_pinned_gb",
            "piecewise_cudagraph",
            "cudagraph_decode_capture_sizes",
            "cudagraph_prefill_capture_sizes",
        ]:
            if key in sidecar:
                lines.append(f"- {key}: {sidecar[key]}")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = analyze(args.log)
    if args.out is None:
        print(report)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
