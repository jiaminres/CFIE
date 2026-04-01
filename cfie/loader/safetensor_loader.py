"""Safetensors 索引读取工具。"""

from __future__ import annotations

import json
from pathlib import Path


def load_safetensor_index(model_dir: str) -> dict[str, str]:
    """读取 `model.safetensors.index.json` 的 weight_map。"""

    index_path = Path(model_dir).expanduser() / "model.safetensors.index.json"
    if not index_path.exists():
        return {}

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map", {})
    if not isinstance(weight_map, dict):
        raise ValueError("Invalid safetensors index: 'weight_map' is not a dict")
    return {str(k): str(v) for k, v in weight_map.items()}
