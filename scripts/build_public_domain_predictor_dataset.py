"""Build a public-domain text dataset for predictor trace/training."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import time
import json
from pathlib import Path
import re
from urllib.request import urlopen


DEFAULT_OUTPUT_PATH = Path("assets/datasets/predictor_public_domain_texts.jsonl")
DEFAULT_MANIFEST_PATH = Path("assets/datasets/predictor_public_domain_sources.json")

PUBLIC_DOMAIN_SOURCES = (
    {
        "source_id": "gutenberg_pg1342",
        "title": "Pride and Prejudice",
        "author": "Jane Austen",
        "license": "Project Gutenberg public-domain text",
        "url": "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
    },
    {
        "source_id": "gutenberg_pg11",
        "title": "Alice's Adventures in Wonderland",
        "author": "Lewis Carroll",
        "license": "Project Gutenberg public-domain text",
        "url": "https://www.gutenberg.org/cache/epub/11/pg11.txt",
    },
    {
        "source_id": "gutenberg_pg1661",
        "title": "The Adventures of Sherlock Holmes",
        "author": "Arthur Conan Doyle",
        "license": "Project Gutenberg public-domain text",
        "url": "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",
    },
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download public-domain texts and build a JSONL dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="JSONL output path consumed by predictor-trace.",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="JSON manifest path describing dataset sources.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=700,
        help="Minimum characters per JSONL record.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=1800,
        help="Soft maximum characters per JSONL record.",
    )
    return parser


def _download_text(url: str) -> str:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            try:
                import requests

                response = requests.get(
                    url,
                    timeout=60,
                    stream=True,
                    headers={"User-Agent": "cfie-predictor-dataset-builder/1.0"},
                )
                response.raise_for_status()
                chunks: list[bytes] = []
                try:
                    for chunk in response.iter_content(chunk_size=1 << 16):
                        if chunk:
                            chunks.append(chunk)
                except Exception:
                    partial = b"".join(chunks)
                    if len(partial) >= 100_000:
                        return partial.decode(
                            response.encoding or "utf-8",
                            errors="replace",
                        )
                    raise
                payload = b"".join(chunks)
                return payload.decode(response.encoding or "utf-8", errors="replace")
            except ModuleNotFoundError:
                with urlopen(url, timeout=60) as response:
                    return response.read().decode("utf-8")
        except Exception as exc:
            last_error = exc
            if attempt == 2:
                break
            time.sleep(1.0 + attempt)
    assert last_error is not None
    raise last_error


def _strip_gutenberg_boilerplate(text: str) -> str:
    start_match = re.search(
        r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
        text,
        flags=re.IGNORECASE,
    )
    end_match = re.search(
        r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
        text,
        flags=re.IGNORECASE,
    )
    start = start_match.end() if start_match else 0
    end = end_match.start() if end_match else len(text)
    return text[start:end].strip()


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _paragraphs(text: str) -> list[str]:
    return [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]


def _chunk_paragraphs(
    paragraphs: Iterable[str],
    *,
    min_chars: int,
    max_chars: int,
) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for paragraph in paragraphs:
        paragraph_len = len(paragraph)
        next_len = current_len + paragraph_len + (2 if current else 0)
        if current and next_len > max_chars and current_len >= min_chars:
            chunks.append("\n\n".join(current))
            current = [paragraph]
            current_len = paragraph_len
            continue
        current.append(paragraph)
        current_len = next_len if current_len else paragraph_len

    if current:
        if chunks and current_len < min_chars:
            chunks[-1] = f"{chunks[-1]}\n\n" + "\n\n".join(current)
        else:
            chunks.append("\n\n".join(current))
    return chunks


def build_dataset_records(*, min_chars: int, max_chars: int) -> tuple[list[dict], list[dict]]:
    manifest_entries: list[dict] = []
    records: list[dict] = []

    for source in PUBLIC_DOMAIN_SOURCES:
        raw_text = _download_text(source["url"])
        clean_text = _normalize_text(_strip_gutenberg_boilerplate(raw_text))
        chunks = _chunk_paragraphs(
            _paragraphs(clean_text),
            min_chars=min_chars,
            max_chars=max_chars,
        )
        manifest_entries.append(
            {
                **source,
                "record_count": len(chunks),
                "character_count": len(clean_text),
            }
        )
        for chunk_index, chunk in enumerate(chunks):
            records.append(
                {
                    "source_id": source["source_id"],
                    "source_title": source["title"],
                    "chunk_index": chunk_index,
                    "text": chunk,
                }
            )
    return manifest_entries, records


def main() -> int:
    args = build_parser().parse_args()

    manifest_entries, records = build_dataset_records(
        min_chars=args.min_chars,
        max_chars=args.max_chars,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )

    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(
            {
                "dataset_kind": "cfie_predictor_public_domain_texts",
                "record_count": len(records),
                "sources": manifest_entries,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "output": str(args.output),
                "manifest_output": str(args.manifest_output),
                "record_count": len(records),
                "source_count": len(manifest_entries),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
