"""Helpers for resolving local Hugging Face snapshot paths."""

from __future__ import annotations

import os
from pathlib import Path


def _candidate_hub_roots() -> tuple[Path, ...]:
    roots: list[Path] = []

    for env_name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        raw_value = os.environ.get(env_name)
        if raw_value:
            roots.append(Path(raw_value).expanduser())

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home).expanduser() / "hub")

    roots.append(Path.home() / ".cache" / "huggingface" / "hub")

    ordered_unique_roots: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered_unique_roots.append(root)
    return tuple(ordered_unique_roots)


def resolve_local_snapshot_path(
    *,
    repo_dir_name: str,
    snapshot_name: str,
) -> str:
    for hub_root in _candidate_hub_roots():
        candidate = hub_root / repo_dir_name / "snapshots" / snapshot_name
        if candidate.exists():
            return str(candidate)

    default_root = _candidate_hub_roots()[0]
    return str(default_root / repo_dir_name / "snapshots" / snapshot_name)
