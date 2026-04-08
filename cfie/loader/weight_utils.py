# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""权重加载工具（基于 vLLM loader 设计裁剪）。"""

from __future__ import annotations

from dataclasses import dataclass
import glob
import os
from pathlib import Path
from typing import Iterator

import torch

from cfie.config.schema import LoadConfig
from cfie.loader.safetensor_loader import load_safetensor_index


@dataclass(slots=True)
class PreparedWeights:
    """已解析的权重文件集合。"""

    model_dir: str
    weight_files: list[str]
    use_safetensors: bool


def _resolve_allow_patterns(load_format: str) -> list[str]:
    if load_format in ("auto", "hf"):
        return ["*.safetensors", "*.bin", "*.pt"]
    if load_format == "safetensors":
        return ["*.safetensors"]
    if load_format == "pt":
        return ["*.bin", "*.pt"]
    raise ValueError(f"Unsupported load_format: {load_format}")


def filter_files_not_needed_for_inference(weight_files: list[str]) -> list[str]:
    """过滤明显非推理所需文件。"""

    deny_tokens = (
        "optimizer",
        "training_args",
        "trainer_state",
        "scheduler",
        "rng_state",
        "events.out.tfevents",
    )
    return [
        path for path in weight_files
        if not any(token in os.path.basename(path) for token in deny_tokens)
    ]


def filter_duplicate_safetensors_files(
    weight_files: list[str],
    model_dir: str,
    index_filename: str = "model.safetensors.index.json",
) -> list[str]:
    """按 index 文件过滤重复或无关 safetensors 分片。"""

    del index_filename
    weight_map = load_safetensor_index(model_dir)
    if not weight_map:
        return sorted(weight_files)

    allowed = {str(Path(model_dir) / shard_name) for shard_name in weight_map.values()}
    filtered = [path for path in weight_files if path in allowed]
    return sorted(filtered)


def maybe_download_weights_from_hf(
    model_or_path: str,
    load_config: LoadConfig,
    allow_patterns: list[str],
) -> str:
    """若是远端模型，则下载到本地缓存目录。"""

    if os.path.isdir(model_or_path):
        return model_or_path

    from huggingface_hub import snapshot_download

    snapshot_dir = snapshot_download(
        repo_id=model_or_path,
        revision=load_config.revision,
        local_files_only=load_config.local_files_only,
        cache_dir=load_config.download_dir,
        allow_patterns=allow_patterns,
        ignore_patterns=list(load_config.ignore_patterns)
        if load_config.ignore_patterns else None,
    )
    return snapshot_dir


def prepare_weights(model_or_path: str, load_config: LoadConfig) -> PreparedWeights:
    """解析并返回可用于加载的权重文件列表。"""

    allow_patterns = _resolve_allow_patterns(load_config.load_format)
    model_dir = maybe_download_weights_from_hf(model_or_path, load_config,
                                               allow_patterns)

    weight_files: list[str] = []
    use_safetensors = False

    for pattern in allow_patterns:
        matches = sorted(glob.glob(os.path.join(model_dir, pattern)))
        if matches:
            weight_files = matches
            if pattern == "*.safetensors":
                use_safetensors = True
            break

    if not weight_files:
        raise RuntimeError(
            f"Cannot find model weights with model_or_path={model_or_path!r}")

    if use_safetensors:
        weight_files = filter_duplicate_safetensors_files(weight_files, model_dir)
    else:
        weight_files = filter_files_not_needed_for_inference(weight_files)

    if not weight_files:
        raise RuntimeError("No valid weight files after filtering")

    return PreparedWeights(model_dir=model_dir,
                           weight_files=weight_files,
                           use_safetensors=use_safetensors)


def iter_safetensors_weights(
    weight_files: list[str],
) -> Iterator[tuple[str, torch.Tensor]]:
    """逐文件迭代 safetensors 权重。"""

    from safetensors.torch import safe_open

    for file_path in weight_files:
        with safe_open(file_path, framework="pt", device="cpu") as sf:
            for key in sf.keys():
                yield key, sf.get_tensor(key)


def iter_pt_weights(weight_files: list[str]) -> Iterator[tuple[str, torch.Tensor]]:
    """逐文件迭代 `.bin/.pt` 权重。"""

    for file_path in weight_files:
        state = torch.load(file_path, map_location="cpu")
        if not isinstance(state, dict):
            continue

        if "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]

        for name, tensor in state.items():
            if torch.is_tensor(tensor):
                yield str(name), tensor


def iter_weight_tensors(prepared: PreparedWeights) -> Iterator[tuple[str, torch.Tensor]]:
    """按格式分发权重迭代器。"""

    if prepared.use_safetensors:
        yield from iter_safetensors_weights(prepared.weight_files)
    else:
        yield from iter_pt_weights(prepared.weight_files)


def get_total_bytes(weight_files: list[str]) -> int:
    """统计文件总字节数。"""

    return sum(Path(path).stat().st_size for path in weight_files if Path(path).is_file())
