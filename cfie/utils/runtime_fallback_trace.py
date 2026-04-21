# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import atexit
import threading
from collections import Counter

import cfie.envs as envs
from cfie.logger import init_logger

logger = init_logger(__name__)

_COUNTS: Counter[str] = Counter()
_LOCK = threading.Lock()
_REGISTERED = False


def is_enabled() -> bool:
    return envs.VLLM_TRACE_RUNTIME_FALLBACKS


def _ensure_registered() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    atexit.register(_dump_summary)
    _REGISTERED = True


def record(tag: str, branch: str) -> None:
    if not is_enabled():
        return

    _ensure_registered()
    key = f"{tag}:{branch}"
    with _LOCK:
        _COUNTS[key] += 1

    logger.info_once(
        "Runtime fallback tracing enabled; summary will be emitted at process exit.",
        scope="local",
    )


def snapshot() -> dict[str, int]:
    with _LOCK:
        return dict(_COUNTS)


def reset_for_test() -> None:
    with _LOCK:
        _COUNTS.clear()


def _dump_summary() -> None:
    if not is_enabled():
        return

    with _LOCK:
        if not _COUNTS:
            return
        summary = ", ".join(
            f"{key}={value}" for key, value in sorted(_COUNTS.items())
        )

    logger.info("Runtime fallback summary: %s", summary)
