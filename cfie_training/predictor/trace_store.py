"""Compact tensor cache for large predictor trace JSON files."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


_TRACE_CACHE_VERSION = 1


def _parse_header_scalar(line: str) -> Any:
    return json.loads(line.split(":", 1)[1].rstrip(","))


@dataclass(slots=True)
class PredictorTraceTensorStore:
    profile_name: str
    example_count: int
    window_layers: int
    candidate_experts_per_layer: int
    executed_experts_per_layer: int
    hidden_size: int
    target_topk: int
    source_path: Path
    cache_dir: Path
    hidden_states: np.ndarray
    layer_indices: np.ndarray
    step_indices: np.ndarray
    future_layer_mask: np.ndarray
    target_ids: np.ndarray
    teacher_router_logits: np.ndarray | None = None

    @property
    def metadata_path(self) -> Path:
        return self.cache_dir / "metadata.json"

    @property
    def hidden_states_path(self) -> Path:
        return self.cache_dir / "hidden_states.npy"

    @property
    def layer_indices_path(self) -> Path:
        return self.cache_dir / "layer_indices.npy"

    @property
    def step_indices_path(self) -> Path:
        return self.cache_dir / "step_indices.npy"

    @property
    def target_ids_path(self) -> Path:
        return self.cache_dir / "target_ids.npy"

    @property
    def future_layer_mask_path(self) -> Path:
        return self.cache_dir / "future_layer_mask.npy"

    @property
    def teacher_router_logits_path(self) -> Path:
        return self.cache_dir / "teacher_router_logits.npy"

    @classmethod
    def from_trace_json(
            cls,
            path: str | Path,
            *,
            cache_dir: str | Path | None = None,
    ) -> "PredictorTraceTensorStore":
        source_path = Path(path)
        resolved_cache_dir = (
            source_path.with_name(source_path.name + ".tensor_cache")
            if cache_dir is None else Path(cache_dir)
        )
        resolved_cache_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = resolved_cache_dir / "metadata.json"

        if cls._cache_is_fresh(
                metadata_path=metadata_path,
                source_path=source_path,
                cache_dir=resolved_cache_dir,
        ):
            return cls._open_cached(
                metadata_path=metadata_path,
                source_path=source_path,
                cache_dir=resolved_cache_dir,
            )

        return cls._build_cache(
            source_path=source_path,
            cache_dir=resolved_cache_dir,
        )

    @classmethod
    def _cache_is_fresh(
            cls,
            *,
            metadata_path: Path,
            source_path: Path,
            cache_dir: Path,
    ) -> bool:
        if not metadata_path.is_file():
            return False
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False

        expected_paths = (
            cache_dir / "hidden_states.npy",
            cache_dir / "layer_indices.npy",
            cache_dir / "step_indices.npy",
            cache_dir / "future_layer_mask.npy",
            cache_dir / "target_ids.npy",
        )
        if bool(payload.get("has_teacher_router_logits", False)):
            expected_paths = expected_paths + (
                cache_dir / "teacher_router_logits.npy",
            )
        if not all(path.is_file() for path in expected_paths):
            return False

        stat = source_path.stat()
        return (
            int(payload.get("cache_version", -1)) == _TRACE_CACHE_VERSION
            and int(payload.get("source_size", -1)) == int(stat.st_size)
            and int(payload.get("source_mtime_ns", -1)) == int(stat.st_mtime_ns)
        )

    @classmethod
    def _open_cached(
            cls,
            *,
            metadata_path: Path,
            source_path: Path,
            cache_dir: Path,
    ) -> "PredictorTraceTensorStore":
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        return cls(
            profile_name=str(payload["profile_name"]),
            example_count=int(payload["example_count"]),
            window_layers=int(payload["window_layers"]),
            candidate_experts_per_layer=int(
                payload["candidate_experts_per_layer"]
            ),
            executed_experts_per_layer=int(
                payload["executed_experts_per_layer"]
            ),
            hidden_size=int(payload["hidden_size"]),
            target_topk=int(payload["target_topk"]),
            source_path=source_path,
            cache_dir=cache_dir,
            hidden_states=np.load(cache_dir / "hidden_states.npy", mmap_mode="r"),
            layer_indices=np.load(cache_dir / "layer_indices.npy", mmap_mode="r"),
            step_indices=np.load(cache_dir / "step_indices.npy", mmap_mode="r"),
            future_layer_mask=np.load(
                cache_dir / "future_layer_mask.npy",
                mmap_mode="r",
            ),
            target_ids=np.load(cache_dir / "target_ids.npy", mmap_mode="r"),
            teacher_router_logits=(
                None
                if not bool(payload.get("has_teacher_router_logits", False))
                else np.load(
                    cache_dir / "teacher_router_logits.npy",
                    mmap_mode="r",
                )
            ),
        )

    @classmethod
    def _build_cache(
            cls,
            *,
            source_path: Path,
            cache_dir: Path,
    ) -> "PredictorTraceTensorStore":
        try:
            return cls._build_cache_from_streaming_json(
                source_path=source_path,
                cache_dir=cache_dir,
            )
        except json.JSONDecodeError:
            return cls._build_cache_from_standard_json(
                source_path=source_path,
                cache_dir=cache_dir,
            )

    @classmethod
    def _build_cache_from_streaming_json(
            cls,
            *,
            source_path: Path,
            cache_dir: Path,
    ) -> "PredictorTraceTensorStore":
        profile_name = ""
        example_count = 0
        window_layers = 0
        candidate_experts_per_layer = 0
        executed_experts_per_layer = 0

        with source_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped.startswith('"profile_name"'):
                    profile_name = str(_parse_header_scalar(stripped))
                elif stripped.startswith('"example_count"'):
                    example_count = int(_parse_header_scalar(stripped))
                elif stripped.startswith('"window_layers"'):
                    window_layers = int(_parse_header_scalar(stripped))
                elif stripped.startswith('"candidate_experts_per_layer"'):
                    candidate_experts_per_layer = int(
                        _parse_header_scalar(stripped)
                    )
                elif stripped.startswith('"executed_experts_per_layer"'):
                    executed_experts_per_layer = int(
                        _parse_header_scalar(stripped)
                    )
                elif stripped.startswith('"examples"'):
                    break

            if example_count < 1:
                raise ValueError("trace JSON must declare example_count >= 1")

            first_payload = cls._read_next_example(handle)
            if first_payload is None:
                raise ValueError("trace JSON examples array is empty")

            payloads = [first_payload]
            while True:
                payload = cls._read_next_example(handle)
                if payload is None:
                    break
                payloads.append(payload)

        return cls._build_cache_from_payloads(
            source_path=source_path,
            cache_dir=cache_dir,
            profile_name=profile_name,
            example_count=example_count,
            window_layers=window_layers,
            candidate_experts_per_layer=candidate_experts_per_layer,
            executed_experts_per_layer=executed_experts_per_layer,
            payloads=tuple(payloads),
        )

    @classmethod
    def _build_cache_from_standard_json(
            cls,
            *,
            source_path: Path,
            cache_dir: Path,
    ) -> "PredictorTraceTensorStore":
        from cfie_training.predictor.models import PredictorTraceDataset

        dataset = PredictorTraceDataset.from_json_file(source_path)
        payloads = tuple(example.to_dict() for example in dataset.examples)
        return cls._build_cache_from_payloads(
            source_path=source_path,
            cache_dir=cache_dir,
            profile_name=dataset.profile_name,
            example_count=dataset.example_count,
            window_layers=dataset.window_layers,
            candidate_experts_per_layer=dataset.candidate_experts_per_layer,
            executed_experts_per_layer=dataset.executed_experts_per_layer,
            payloads=payloads,
        )

    @classmethod
    def _build_cache_from_payloads(
            cls,
            *,
            source_path: Path,
            cache_dir: Path,
            profile_name: str,
            example_count: int,
            window_layers: int,
            candidate_experts_per_layer: int,
            executed_experts_per_layer: int,
            payloads: tuple[dict[str, Any], ...],
    ) -> "PredictorTraceTensorStore":
        if example_count < 1:
            raise ValueError("trace JSON must declare example_count >= 1")
        if not payloads:
            raise ValueError("trace JSON examples array is empty")
        if len(payloads) != example_count:
            raise ValueError(
                "trace JSON example_count does not match actual example rows: "
                f"declared={example_count} actual={len(payloads)}"
            )

        first_payload = payloads[0]
        hidden_size = len(first_payload["hidden_state"])
        target_topk = len(first_payload["future_teacher_topk_ids"][0])
        has_teacher_router_logits = (
            first_payload.get("future_teacher_router_logits") is not None
        )
        router_logits_size = (
            0
            if not has_teacher_router_logits
            else len(first_payload["future_teacher_router_logits"][0])
        )

        hidden_states = np.lib.format.open_memmap(
            cache_dir / "hidden_states.npy",
            mode="w+",
            dtype=np.float16,
            shape=(example_count, hidden_size),
        )
        layer_indices = np.lib.format.open_memmap(
            cache_dir / "layer_indices.npy",
            mode="w+",
            dtype=np.int16,
            shape=(example_count,),
        )
        step_indices = np.lib.format.open_memmap(
            cache_dir / "step_indices.npy",
            mode="w+",
            dtype=np.int32,
            shape=(example_count,),
        )
        future_layer_mask = np.lib.format.open_memmap(
            cache_dir / "future_layer_mask.npy",
            mode="w+",
            dtype=np.bool_,
            shape=(example_count, window_layers),
        )
        target_ids = np.lib.format.open_memmap(
            cache_dir / "target_ids.npy",
            mode="w+",
            dtype=np.int16,
            shape=(example_count, window_layers, target_topk),
        )
        teacher_router_logits = None
        if has_teacher_router_logits:
            teacher_router_logits = np.lib.format.open_memmap(
                cache_dir / "teacher_router_logits.npy",
                mode="w+",
                dtype=np.float16,
                shape=(example_count, window_layers, router_logits_size),
            )

        seen_examples = 0
        for payload in payloads:
            seen_examples += cls._write_example(
                payload=payload,
                default_index=seen_examples,
                hidden_states=hidden_states,
                layer_indices=layer_indices,
                step_indices=step_indices,
                future_layer_mask=future_layer_mask,
                target_ids=target_ids,
                teacher_router_logits=teacher_router_logits,
            )

        arrays_to_flush = [
            hidden_states,
            layer_indices,
            step_indices,
            future_layer_mask,
            target_ids,
        ]
        if teacher_router_logits is not None:
            arrays_to_flush.append(teacher_router_logits)
        for array in arrays_to_flush:
            array.flush()

        stat = source_path.stat()
        (cache_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "cache_version": _TRACE_CACHE_VERSION,
                    "source_path": str(source_path),
                    "source_size": int(stat.st_size),
                    "source_mtime_ns": int(stat.st_mtime_ns),
                    "profile_name": profile_name,
                    "example_count": example_count,
                    "window_layers": window_layers,
                    "candidate_experts_per_layer": (
                        candidate_experts_per_layer
                    ),
                    "executed_experts_per_layer": (
                        executed_experts_per_layer
                    ),
                    "hidden_size": hidden_size,
                    "target_topk": target_topk,
                    "has_teacher_router_logits": has_teacher_router_logits,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        return cls._open_cached(
            metadata_path=cache_dir / "metadata.json",
            source_path=source_path,
            cache_dir=cache_dir,
        )

    @staticmethod
    def _read_next_example(handle: Any) -> dict[str, Any] | None:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("]"):
                return None
            return json.loads(stripped.rstrip(","))
        return None

    @staticmethod
    def _write_example(
            *,
            payload: dict[str, Any],
            default_index: int,
            hidden_states: np.ndarray,
            layer_indices: np.ndarray,
            step_indices: np.ndarray,
            future_layer_mask: np.ndarray,
            target_ids: np.ndarray,
            teacher_router_logits: np.ndarray | None,
    ) -> int:
        example_index = int(payload.get("example_index", default_index))
        hidden_states[example_index] = np.asarray(
            payload["hidden_state"],
            dtype=np.float16,
        )
        layer_indices[example_index] = int(payload["insertion_layer_index"])
        step_indices[example_index] = int(payload.get("step_index", 0))
        future_teacher_topk_ids = payload["future_teacher_topk_ids"]
        future_count = len(future_teacher_topk_ids)
        target_ids[example_index].fill(0)
        future_layer_mask[example_index].fill(False)
        if future_count > int(target_ids.shape[1]):
            raise ValueError(
                "trace JSON future_teacher_topk_ids exceeds configured window: "
                f"future_count={future_count} window_layers={target_ids.shape[1]}"
            )
        for future_index, teacher_topk_ids in enumerate(future_teacher_topk_ids):
            target_ids[example_index, future_index] = np.asarray(
                teacher_topk_ids,
                dtype=np.int16,
            )
            future_layer_mask[example_index, future_index] = True
        if teacher_router_logits is not None:
            payload_router_logits = payload.get("future_teacher_router_logits")
            if payload_router_logits is None:
                raise ValueError(
                    "trace JSON is missing future_teacher_router_logits for an "
                    "example even though the cache was initialized with logits"
                )
            if len(payload_router_logits) != future_count:
                raise ValueError(
                    "future_teacher_router_logits length does not match "
                    "future_teacher_topk_ids length"
                )
            teacher_router_logits[example_index].fill(0)
            for future_index, layer_logits in enumerate(payload_router_logits):
                teacher_router_logits[example_index, future_index] = np.asarray(
                    layer_logits,
                    dtype=np.float16,
                )
        return 1

    def step_split(
            self,
            *,
            holdout_modulus: int = 10,
            holdout_remainder: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        if holdout_modulus < 2:
            raise ValueError("holdout_modulus must be >= 2")
        mask = (np.asarray(self.step_indices) % int(holdout_modulus)) == int(
            holdout_remainder
        )
        validation_indices = np.nonzero(mask)[0].astype(np.int64, copy=False)
        train_indices = np.nonzero(~mask)[0].astype(np.int64, copy=False)
        return train_indices, validation_indices
