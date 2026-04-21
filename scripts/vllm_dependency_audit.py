#!/usr/bin/env python3
"""Audit direct ``vllm`` imports under CFIE source tree."""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ImportHit:
    file: Path
    module: str
    lineno: int


def _iter_python_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if p.is_file())


def _collect_imports(py_file: Path) -> list[ImportHit]:
    hits: list[ImportHit] = []
    try:
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
    except Exception:
        return hits

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module
            if mod == "vllm" or (mod is not None and mod.startswith("vllm.")):
                hits.append(ImportHit(py_file, mod, node.lineno))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name
                if mod == "vllm" or mod.startswith("vllm."):
                    hits.append(ImportHit(py_file, mod, node.lineno))
    return hits


def _build_markdown(
    root: Path,
    hits: list[ImportHit],
    top_modules: list[tuple[str, int]],
    top_files: list[tuple[str, int]],
) -> str:
    lines = [
        "# vLLM Dependency Audit",
        "",
        f"- root: `{root}`",
        f"- import lines: **{len(hits)}**",
        f"- files with direct `vllm` imports: **{len(set(h.file for h in hits))}**",
        "",
        "## Top Modules",
        "",
        "| module | count |",
        "|---|---:|",
    ]
    lines.extend(f"| `{m}` | {c} |" for m, c in top_modules)
    lines.extend(
        [
            "",
            "## Top Files",
            "",
            "| file | count |",
            "|---|---:|",
        ]
    )
    lines.extend(f"| `{f}` | {c} |" for f, c in top_files)
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("CFIE/cfie"),
        help="Root directory to audit (default: CFIE/cfie).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="Number of top modules/files to print (default: 30).",
    )
    parser.add_argument(
        "--focus",
        type=Path,
        action="append",
        default=[],
        help="Optional file(s) to print exact vllm import lines.",
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=None,
        help="Optional markdown report output path.",
    )
    args = parser.parse_args()

    root = args.root
    all_hits: list[ImportHit] = []
    for py_file in _iter_python_files(root):
        all_hits.extend(_collect_imports(py_file))

    module_counter = Counter(hit.module for hit in all_hits)
    file_counter = Counter(str(hit.file) for hit in all_hits)

    print(f"[audit] root={root}")
    print(f"[audit] import_lines={len(all_hits)}")
    print(f"[audit] files={len(file_counter)}")
    print("[audit] top_modules:")
    for mod, count in module_counter.most_common(args.top):
        print(f"  {count:5d}  {mod}")

    print("[audit] top_files:")
    for file_path, count in file_counter.most_common(args.top):
        print(f"  {count:5d}  {file_path}")

    if args.focus:
        by_file = defaultdict(list)
        for hit in all_hits:
            by_file[hit.file.resolve()].append(hit)
        for focus in args.focus:
            focus_abs = focus.resolve()
            print(f"[focus] {focus_abs}")
            for hit in sorted(by_file.get(focus_abs, []), key=lambda h: h.lineno):
                print(f"  L{hit.lineno:4d}  {hit.module}")

    if args.markdown_out is not None:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        top_modules = module_counter.most_common(args.top)
        top_files = file_counter.most_common(args.top)
        args.markdown_out.write_text(
            _build_markdown(root, all_hits, top_modules, top_files),
            encoding="utf-8",
        )
        print(f"[audit] markdown={args.markdown_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

