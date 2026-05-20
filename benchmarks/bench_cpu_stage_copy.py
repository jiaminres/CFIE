from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import torch

from cfie import _custom_ops as ops


def _time_call(fn, warmup: int, repeat: int) -> list[float]:
    for _ in range(warmup):
        fn()
    times: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return times


def _gbps(total_bytes: int, seconds: float) -> float:
    return total_bytes / seconds / (1024**3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert-mib", type=int, default=40)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--pin-dst", action="store_true")
    parser.add_argument("--h2d", action="store_true")
    parser.add_argument("--probe-pinned-gib", type=float, default=0.0)
    args = parser.parse_args()

    if args.probe_pinned_gib > 0:
        chunk_bytes = 256 * (1 << 20)
        target_bytes = int(args.probe_pinned_gib * (1 << 30))
        chunks: list[torch.Tensor] = []
        allocated = 0
        while allocated < target_bytes:
            try:
                chunks.append(torch.empty(chunk_bytes, dtype=torch.uint8,
                                          device="cpu", pin_memory=True))
                allocated += chunk_bytes
                print(json.dumps({"pinned_allocated_gib": allocated / (1 << 30)}))
            except RuntimeError as exc:
                print(json.dumps({
                    "pinned_allocated_gib": allocated / (1 << 30),
                    "error": str(exc),
                }))
                break
        return

    per_expert_bytes = args.expert_mib * (1 << 20)
    total_bytes = per_expert_bytes * args.experts
    source = torch.empty(total_bytes, dtype=torch.uint8, device="cpu")
    dest = torch.empty(
        total_bytes,
        dtype=torch.uint8,
        device="cpu",
        pin_memory=bool(args.pin_dst),
    )
    offsets = torch.arange(args.experts, dtype=torch.int64) * per_expert_bytes

    def native_copy() -> None:
        ops.copy_expert_slices_to_stage_cpu(
            source,
            offsets,
            dest,
            per_expert_bytes,
            args.workers,
        )

    def python_seq_copy() -> None:
        for expert in range(args.experts):
            start = expert * per_expert_bytes
            dest[start:start + per_expert_bytes].copy_(
                source[start:start + per_expert_bytes]
            )

    def python_thread_copy() -> None:
        def copy_one(expert: int) -> None:
            start = expert * per_expert_bytes
            dest[start:start + per_expert_bytes].copy_(
                source[start:start + per_expert_bytes]
            )

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            list(executor.map(copy_one, range(args.experts)))

    results = {}
    for name, fn in (
        ("native_cpp_pool", native_copy),
        ("python_seq_torch_copy", python_seq_copy),
        ("python_thread_torch_copy", python_thread_copy),
    ):
        times = _time_call(fn, args.warmup, args.repeat)
        best = min(times)
        avg = sum(times) / len(times)
        results[name] = {
            "best_ms": best * 1000.0,
            "avg_ms": avg * 1000.0,
            "best_gib_s": _gbps(total_bytes, best),
            "avg_gib_s": _gbps(total_bytes, avg),
        }

    if args.h2d and torch.cuda.is_available():
        cuda_dest = torch.empty(total_bytes, dtype=torch.uint8, device="cuda")

        def h2d_copy() -> None:
            cuda_dest.copy_(dest, non_blocking=bool(dest.is_pinned()))
            torch.cuda.synchronize()

        times = _time_call(h2d_copy, args.warmup, args.repeat)
        best = min(times)
        avg = sum(times) / len(times)
        results["h2d_from_dest"] = {
            "best_ms": best * 1000.0,
            "avg_ms": avg * 1000.0,
            "best_gib_s": _gbps(total_bytes, best),
            "avg_gib_s": _gbps(total_bytes, avg),
        }

    print(json.dumps({
        "expert_mib": args.expert_mib,
        "experts": args.experts,
        "workers": args.workers,
        "pin_dst": bool(args.pin_dst),
        "total_mib": total_bytes / (1 << 20),
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
