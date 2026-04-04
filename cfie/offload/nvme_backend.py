"""Local safetensors-backed expert source for tiered MoE caching."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

from cfie.logger import init_logger

logger = init_logger(__name__)

_EXPERT_KEY_RE = re.compile(
    r"^(?P<prefix>.+\.experts)\.(?P<expert>\d+)\.(?P<suffix>.+)$"
)


@dataclass(frozen=True, slots=True)
class TensorRef:
    # 记录该张量所在的 safetensors shard 文件名。
    file_name: str
    # 记录张量在原始 safetensors 文件里的完整 key。
    full_key: str
    # 记录去掉层名前缀后的相对 key，便于与目标 bundle 对齐。
    relative_key: str


class SafetensorExpertStore:
    """Fetch a single expert from local safetensors shards on demand."""

    def __init__(self, model_dir: str):
        # 记录本地模型目录，后续所有 shard 都从这里打开。
        self.model_dir = Path(model_dir)
        # 启动时一次性建立“层名 + expert_id -> TensorRef 列表”索引。
        self._expert_refs = self._build_index()

    def has_expert(self, layer_name: str, expert_id: int) -> bool:
        # 只检查索引里是否存在对应 expert 的张量引用。
        return self._resolve_refs(layer_name, expert_id) is not None

    def load_expert(
        self,
        layer_name: str,
        expert_id: int,
        *,
        skip_suffixes: tuple[str, ...] = (),
    ) -> dict[str, torch.Tensor]:
        # 先从索引中解析该 expert 对应的所有张量引用。
        refs = self._resolve_refs(layer_name, expert_id)
        if refs is None:
            raise KeyError(f"Expert {layer_name}.{expert_id} not found in local store")

        # ----------------- 按 shard 文件分组，减少 safe_open 次数 -----------------
        refs_by_file: dict[str, list[TensorRef]] = defaultdict(list)
        for ref in refs:
            # 对不需要的后缀直接跳过，例如 g_idx 这类可重建字段。
            if skip_suffixes and ref.relative_key.endswith(skip_suffixes):
                continue
            refs_by_file[ref.file_name].append(ref)

        # ----------------- 逐 shard 读取目标 expert 的张量 -----------------
        tensors: dict[str, torch.Tensor] = {}
        for file_name, refs_for_file in refs_by_file.items():
            file_path = self.model_dir / file_name
            with safe_open(file_path, framework="pt", device="cpu") as handle:
                for ref in refs_for_file:
                    # 读取后立即转成 contiguous CPU tensor，便于后续拷贝/缓存。
                    tensors[ref.relative_key] = handle.get_tensor(ref.full_key).contiguous()
        return tensors

    def copy_expert_into(
        self,
        layer_name: str,
        expert_id: int,
        dst_tensors: dict[str, torch.Tensor],
        *,
        skip_suffixes: tuple[str, ...] = (),
    ) -> None:
        # 先定位该 expert 的张量引用。
        refs = self._resolve_refs(layer_name, expert_id)
        if refs is None:
            raise KeyError(f"Expert {layer_name}.{expert_id} not found in local store")

        # 目标 bundle 采用 "slot.xxx" 命名，这里按后缀建立映射关系。
        suffix_to_name = {
            relative_name.split(".", 1)[1]: relative_name for relative_name in dst_tensors
        }
        # 同样先按 shard 分组，减少文件打开次数。
        refs_by_file: dict[str, list[TensorRef]] = defaultdict(list)
        for ref in refs:
            if skip_suffixes and ref.relative_key.endswith(skip_suffixes):
                continue
            refs_by_file[ref.file_name].append(ref)

        # 逐 shard 读取并原位 copy 到调用方给定的目标张量里。
        for file_name, refs_for_file in refs_by_file.items():
            file_path = self.model_dir / file_name
            with safe_open(file_path, framework="pt", device="cpu") as handle:
                for ref in refs_for_file:
                    # relative_key 的前缀是 expert_id，本行再切掉它与目标 bundle 后缀对齐。
                    suffix = ref.relative_key.split(".", 1)[1]
                    dst_name = suffix_to_name.get(suffix)
                    # 目标 bundle 不需要该字段时，直接跳过。
                    if dst_name is None:
                        continue
                    # 直接把磁盘读出的 CPU tensor 拷贝进目标张量。
                    dst_tensors[dst_name].copy_(handle.get_tensor(ref.full_key))

    def _build_index(self) -> dict[tuple[str, int], tuple[TensorRef, ...]]:
        # ----------------- 优先走官方 safetensors index 文件 -----------------
        index_path = self.model_dir / "model.safetensors.index.json"
        if index_path.exists():
            with index_path.open("r", encoding="utf-8") as f:
                weight_map = json.load(f)["weight_map"]
        else:
            # 若缺少 index 文件，则退化成遍历所有 shard 并扫描其中的 key。
            weight_map = {}
            for shard_path in sorted(self.model_dir.glob("*.safetensors")):
                with safe_open(shard_path, framework="pt", device="cpu") as handle:
                    for key in handle.keys():
                        weight_map[key] = shard_path.name

        # ----------------- 从全量权重映射中过滤出专家张量 -----------------
        expert_refs: dict[tuple[str, int], list[TensorRef]] = defaultdict(list)
        for full_key, file_name in weight_map.items():
            match = _EXPERT_KEY_RE.match(full_key)
            # 非 expert 张量不参与 tiered cache 索引。
            if match is None:
                continue
            # prefix 是层名，expert 是全局 expert id，suffix 是专家内部参数名。
            layer_name = match.group("prefix")
            expert_id = int(match.group("expert"))
            suffix = match.group("suffix")
            expert_refs[(layer_name, expert_id)].append(
                TensorRef(
                    file_name=file_name,
                    full_key=full_key,
                    relative_key=f"{expert_id}.{suffix}",
                )
            )

        # 打印索引完成后的专家条目数。
        logger.info(
            "Indexed %d local expert entries from %s",
            len(expert_refs),
            self.model_dir,
        )
        # 内部统一冻结成 tuple，避免后续误改。
        return {key: tuple(value) for key, value in expert_refs.items()}

    def _resolve_refs(
        self, layer_name: str, expert_id: int
    ) -> tuple[TensorRef, ...] | None:
        # 同一层在不同模型实现里可能带有不同前缀，这里依次尝试 alias。
        for candidate in _iter_layer_name_aliases(layer_name):
            refs = self._expert_refs.get((candidate, expert_id))
            if refs is not None:
                return refs
        return None


def _iter_layer_name_aliases(layer_name: str) -> tuple[str, ...]:
    # aliases 保存同一个 MoE 层在不同命名空间下的可选别名。
    aliases: list[str] = []

    def add(candidate: str) -> None:
        # 只追加非空且尚未出现过的别名，避免重复尝试。
        if candidate and candidate not in aliases:
            aliases.append(candidate)

    # 原始层名总是第一个候选。
    add(layer_name)
    # 在是否带 "model." 前缀之间互相补全。
    if layer_name.startswith("model."):
        add(layer_name.removeprefix("model."))
    else:
        add(f"model.{layer_name}")

    # 处理 language_model.model.* 与 model.language_model.* 两套前缀变体。
    if layer_name.startswith("language_model.model."):
        suffix = layer_name.removeprefix("language_model.model.")
        add(f"model.language_model.{suffix}")
        add(f"model.{suffix}")
    if layer_name.startswith("model.language_model."):
        suffix = layer_name.removeprefix("model.language_model.")
        add(f"language_model.model.{suffix}")
        add(f"language_model.{suffix}")
    # 处理 MTP 层既可能带 model. 前缀，也可能不带的情况。
    if layer_name.startswith("mtp."):
        add(f"model.{layer_name}")
    if layer_name.startswith("model.mtp."):
        add(layer_name.removeprefix("model."))

    # 返回去重后的 alias 列表。
    return tuple(aliases)
