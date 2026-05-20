"""Activation-aware GPTQ export for local Qwen3.5-122B-A10B.

This is intentionally isolated from CFIE's inference/offload code.  It uses
GPTQModel when available because GPTQModel is the current maintained offline
GPTQ quantizer with Qwen 3/3.5 MoE support.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_MODEL = Path(
    r"C:\Users\13642\.cache\huggingface\hub"
    r"\models--Qwen--Qwen3.5-122B-A10B"
    r"\snapshots\b000b2eb18a7f4cdf3153c4215842da339e09d99"
)
DEFAULT_OUTPUT_REPO = Path(
    r"C:\Users\13642\.cache\huggingface\hub"
    r"\models--Qwen--Qwen3.5-122B-A10B-CFIE-GPTQ-Int4"
)
DEFAULT_DATA_FILES = "en/c4-train.00001-of-01024.json.gz"


def _default_snapshot_name() -> str:
    return "cfie-gptq-" + datetime.now().strftime("%Y%m%d-%H%M%S")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def _call_accepts_kwargs(func: Any) -> bool:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _filter_kwargs(func: Any, values: dict[str, Any]) -> dict[str, Any]:
    if _call_accepts_kwargs(func):
        return {key: value for key, value in values.items() if value is not None}
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return {}
    return {
        key: value
        for key, value in values.items()
        if value is not None and key in signature.parameters
    }


def _read_text_calibration(path: Path, *, nsamples: int) -> list[str]:
    samples: list[str] = []
    current: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            if current:
                samples.append("\n".join(current))
                current = []
            continue
        current.append(stripped)
        if len(samples) >= nsamples:
            break
    if current and len(samples) < nsamples:
        samples.append("\n".join(current))
    return samples[:nsamples]


def _read_jsonl_calibration(
    path: Path,
    *,
    field: str,
    nsamples: int,
) -> list[str]:
    samples: list[str] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if len(samples) >= nsamples:
                break
            if not line.strip():
                continue
            payload = json.loads(line)
            value = payload.get(field)
            if isinstance(value, str) and value.strip():
                samples.append(value.strip())
    return samples


def _load_dataset_calibration(args: argparse.Namespace) -> list[str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets is not installed. Install it in the active environment "
            "or pass --calibration-file/--calibration-jsonl."
        ) from exc

    dataset_kwargs: dict[str, Any] = {}
    if args.dataset_data_files:
        dataset_kwargs["data_files"] = args.dataset_data_files
    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.dataset_split,
        **dataset_kwargs,
    )
    if args.nsamples:
        dataset = dataset.select(range(min(args.nsamples, len(dataset))))
    samples: list[str] = []
    for row in dataset:
        value = row.get(args.dataset_text_field)
        if isinstance(value, str) and value.strip():
            samples.append(value.strip())
    return samples[: args.nsamples]


def _load_calibration(args: argparse.Namespace) -> list[str]:
    if args.calibration_file:
        return _read_text_calibration(args.calibration_file, nsamples=args.nsamples)
    if args.calibration_jsonl:
        return _read_jsonl_calibration(
            args.calibration_jsonl,
            field=args.calibration_jsonl_field,
            nsamples=args.nsamples,
        )
    if args.use_built_in_calibration:
        base = (
            "Qwen is a large mixture-of-experts language model. "
            "This calibration sample exercises multilingual dialogue, code, "
            "mathematics, long context reasoning, and tool-style instructions."
        )
        return [base for _ in range(args.nsamples)]
    return _load_dataset_calibration(args)


def _default_dynamic_config() -> dict[str, dict[str, Any]]:
    return {
        "lm_head": {},
        "model.language_model.embed_tokens": {},
        "-:.*attn.*": {},
        "-:.*shared_expert.*": {},
        "-:.*mtp.*": {},
        "-:.*visual.*": {},
    }


def _build_quant_config(args: argparse.Namespace) -> Any:
    try:
        from gptqmodel import GPTQConfig
    except ImportError as exc:
        raise RuntimeError(
            "gptqmodel is not installed. Install with: "
            "python -m pip install -v gptqmodel datasets accelerate"
        ) from exc

    requested = {
        "bits": args.bits,
        "group_size": args.group_size,
        "desc_act": args.desc_act,
        "sym": args.sym,
        "true_sequential": args.true_sequential,
        "dynamic": _default_dynamic_config(),
    }
    return GPTQConfig(**_filter_kwargs(GPTQConfig, requested))


def _ensure_output_snapshot(args: argparse.Namespace) -> Path:
    repo_dir = args.output_repo
    snapshot_name = args.snapshot_name or _default_snapshot_name()
    snapshot_dir = repo_dir / "snapshots" / snapshot_name
    if snapshot_dir.exists() and any(snapshot_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"output snapshot already exists and is non-empty: {snapshot_dir}"
        )
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    return snapshot_dir


def _write_hf_ref(repo_dir: Path, snapshot_name: str) -> None:
    refs_dir = repo_dir / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    (refs_dir / "main").write_text(snapshot_name + "\n", encoding="utf-8")


def _write_manifest(
    snapshot_dir: Path,
    args: argparse.Namespace,
    calibration_count: int,
    *,
    status: str,
) -> None:
    manifest = {
        "status": status,
        "source_model": args.source_model,
        "output_snapshot": snapshot_dir,
        "created_at_local": datetime.now().isoformat(timespec="seconds"),
        "quantization": {
            "quant_method": "gptq",
            "bits": args.bits,
            "group_size": args.group_size,
            "desc_act": args.desc_act,
            "sym": args.sym,
            "true_sequential": args.true_sequential,
        },
        "calibration": {
            "samples": calibration_count,
            "dataset": args.dataset if not args.calibration_file else None,
            "dataset_config": args.dataset_config,
            "dataset_split": args.dataset_split,
            "dataset_data_files": args.dataset_data_files,
            "calibration_file": args.calibration_file,
            "calibration_jsonl": args.calibration_jsonl,
            "use_built_in_calibration": args.use_built_in_calibration,
        },
        "notes": [
            "Generated by scripts/quantize_qwen122b_gptq_activation.py.",
            "This job recollects activations through GPTQModel.quantize().",
        ],
    }
    (snapshot_dir / "CFIE_GPTQ_QUANTIZE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _ensure_quantize_config(snapshot_dir: Path) -> None:
    config_path = snapshot_dir / "config.json"
    quantize_config_path = snapshot_dir / "quantize_config.json"
    if quantize_config_path.exists() or not config_path.exists():
        return
    config = json.loads(config_path.read_text(encoding="utf-8"))
    quant_cfg = config.get("quantization_config")
    if isinstance(quant_cfg, dict):
        quantize_config_path.write_text(
            json.dumps(quant_cfg, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _count_quantized_tensors(snapshot_dir: Path) -> dict[str, int]:
    index_path = snapshot_dir / "model.safetensors.index.json"
    if not index_path.exists():
        return {}
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map", {})
    if not isinstance(weight_map, dict):
        return {}
    suffixes = (".qweight", ".scales", ".qzeros", ".g_idx")
    return {
        suffix: sum(str(name).endswith(suffix) for name in weight_map)
        for suffix in suffixes
    }


def quantize(args: argparse.Namespace) -> None:
    os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
    snapshot_dir = _ensure_output_snapshot(args)
    snapshot_name = snapshot_dir.name
    _write_manifest(snapshot_dir, args, 0, status="initializing")

    calibration = _load_calibration(args)
    if not calibration:
        raise RuntimeError("no calibration samples were loaded")
    _write_manifest(snapshot_dir, args, len(calibration), status="calibration_loaded")

    if args.dry_run:
        print(f"dry-run ok: calibration_samples={len(calibration)}")
        print(f"output_snapshot={snapshot_dir}")
        return

    try:
        from gptqmodel import GPTQModel
    except ImportError as exc:
        raise RuntimeError(
            "gptqmodel is not installed. Install with: "
            "python -m pip install -v gptqmodel datasets accelerate"
        ) from exc

    quant_config = _build_quant_config(args)
    load_kwargs = _filter_kwargs(
        GPTQModel.load,
        {
            "trust_remote_code": True,
            "dtype": args.dtype,
            "torch_dtype": args.dtype,
            "profile": args.profile,
            "device_map": args.device_map,
        },
    )
    model = GPTQModel.load(str(args.source_model), quant_config, **load_kwargs)
    quantize_kwargs = _filter_kwargs(
        model.quantize,
        {
            "batch_size": args.batch_size,
            "backend": args.backend,
            "auto_gc": True,
            "buffered_fwd": True,
        },
    )
    model.quantize(calibration, **quantize_kwargs)
    model.save(str(snapshot_dir))
    _ensure_quantize_config(snapshot_dir)
    if args.update_ref:
        _write_hf_ref(args.output_repo, snapshot_name)
    counts = _count_quantized_tensors(snapshot_dir)
    _write_manifest(snapshot_dir, args, len(calibration), status="completed")
    print(json.dumps({"output_snapshot": str(snapshot_dir), "counts": counts}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recollect activations and export local Qwen3.5-122B GPTQ.",
    )
    parser.add_argument("--source-model", type=Path, default=DEFAULT_SOURCE_MODEL)
    parser.add_argument("--output-repo", type=Path, default=DEFAULT_OUTPUT_REPO)
    parser.add_argument("--snapshot-name", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--update-ref", action="store_true", default=True)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--desc-act", action="store_true", default=False)
    parser.add_argument("--asym", dest="sym", action="store_false", default=True)
    parser.add_argument("--true-sequential", action="store_true", default=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--profile", default="low_memory")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--backend", default=None)
    parser.add_argument("--nsamples", type=int, default=512)
    parser.add_argument("--calibration-file", type=Path)
    parser.add_argument("--calibration-jsonl", type=Path)
    parser.add_argument("--calibration-jsonl-field", default="text")
    parser.add_argument("--dataset", default="allenai/c4")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-data-files", default=DEFAULT_DATA_FILES)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--dataset-text-field", default="text")
    parser.add_argument(
        "--use-built-in-calibration",
        action="store_true",
        help="Use only for smoke tests; real exports should use a real dataset.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.source_model.exists():
        parser.error(f"--source-model does not exist: {args.source_model}")
    if args.nsamples < 1:
        parser.error("--nsamples must be >= 1")
    return args


if __name__ == "__main__":
    quantize(parse_args())
