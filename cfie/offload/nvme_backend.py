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
    """按需从本地 safetensors 分片中读取单个 expert 的参数。"""

    def __init__(self, model_dir: str):
        # ------------------------------- 初始化本地模型目录与 expert 索引 -------------------------------
        # 将传入的模型目录字符串规范化为 Path 对象，后续所有分片文件都从该目录下读取。
        self.model_dir = Path(model_dir)

        # 启动时预先构建“层名 + expert_id -> TensorRef 列表”的索引，加速后续按 expert 查询。
        self._expert_refs = self._build_index()

    def has_expert(self, layer_name: str, expert_id: int) -> bool:
        # ------------------------------- 判断指定层和 expert 是否存在于本地存储 -------------------------------
        # 通过解析索引中对应的 TensorRef 列表，判断目标 expert 是否存在。
        return self._resolve_refs(layer_name, expert_id) is not None

    def load_expert(
        self,
        layer_name: str,
        expert_id: int,
        *,
        skip_suffixes: tuple[str, ...] = (),
    ) -> dict[str, torch.Tensor]:
        # ------------------------------- 解析目标 expert 对应的张量引用列表 -------------------------------
        # 从本地索引中查找当前层和 expert 对应的全部张量引用。
        refs = self._resolve_refs(layer_name, expert_id)

        # 当索引中不存在目标 expert 时，直接抛出异常。
        if refs is None:
            raise KeyError(f"Expert {layer_name}.{expert_id} not found in local store")

        # ------------------------------- 按 shard 文件对张量引用分组 -------------------------------
        # 用文件名对 TensorRef 做分组，以减少 safe_open 的打开次数。
        refs_by_file: dict[str, list[TensorRef]] = defaultdict(list)

        # 遍历当前 expert 对应的全部张量引用，并按文件名归组。
        for ref in refs:
            # 当当前张量后缀属于跳过列表时，直接忽略该字段。
            if skip_suffixes and ref.relative_key.endswith(skip_suffixes):
                continue

            # 将当前张量引用加入对应 shard 文件的分组列表中。
            refs_by_file[ref.file_name].append(ref)

        # ------------------------------- 逐个 shard 文件读取目标 expert 的张量 -------------------------------
        # 用于保存最终读取出的“相对字段名 -> CPU tensor”映射。
        tensors: dict[str, torch.Tensor] = {}

        # 逐个 shard 文件打开并读取其下属于当前 expert 的张量。
        for file_name, refs_for_file in refs_by_file.items():
            # 拼接当前 shard 文件的完整路径。
            file_path = self.model_dir / file_name

            # 以 CPU 设备方式打开 safetensors 分片文件。
            with safe_open(file_path, framework="pt", device="cpu") as handle:
                # 遍历当前 shard 文件中属于目标 expert 的全部张量引用。
                for ref in refs_for_file:
                    # 读取张量后立即转换为 contiguous CPU tensor，便于后续复制和缓存。
                    tensors[ref.relative_key] = handle.get_tensor(ref.full_key).contiguous()

        # ------------------------------- 返回当前 expert 的张量字典 -------------------------------
        # 返回目标 expert 读取出的全部张量。
        return tensors

    def copy_expert_into(
        self,
        layer_name: str,
        expert_id: int,
        dst_tensors: dict[str, torch.Tensor],
        *,
        skip_suffixes: tuple[str, ...] = (),
    ) -> None:
        # ------------------------------- 解析目标 expert 对应的张量引用列表 -------------------------------
        # 从本地索引中查找当前层和 expert 对应的全部张量引用。
        refs = self._resolve_refs(layer_name, expert_id)

        # 当索引中不存在目标 expert 时，直接抛出异常。
        if refs is None:
            raise KeyError(f"Expert {layer_name}.{expert_id} not found in local store")

        # ------------------------------- 为目标张量 bundle 建立后缀到名字的映射 -------------------------------
        # 目标 bundle 采用 "slot.xxx" 命名方式，这里把后缀映射回完整目标字段名。
        suffix_to_name = {
            relative_name.split(".", 1)[1]: relative_name for relative_name in dst_tensors
        }

        # ------------------------------- 按 shard 文件对张量引用分组 -------------------------------
        # 用文件名对 TensorRef 做分组，以减少 safe_open 的打开次数。
        refs_by_file: dict[str, list[TensorRef]] = defaultdict(list)

        # 遍历当前 expert 的全部张量引用，并按 shard 文件归组。
        for ref in refs:
            # 当当前张量后缀属于跳过列表时，直接忽略该字段。
            if skip_suffixes and ref.relative_key.endswith(skip_suffixes):
                continue

            # 将当前张量引用加入对应 shard 文件的分组列表中。
            refs_by_file[ref.file_name].append(ref)

        # ------------------------------- 逐个 shard 读取并原位拷贝到目标张量 -------------------------------
        # 逐个 shard 文件打开并把张量内容拷贝到调用方提供的目标张量字典中。
        for file_name, refs_for_file in refs_by_file.items():
            # 拼接当前 shard 文件的完整路径。
            file_path = self.model_dir / file_name

            # 以 CPU 设备方式打开 safetensors 分片文件。
            with safe_open(file_path, framework="pt", device="cpu") as handle:
                # 遍历当前 shard 文件中属于目标 expert 的全部张量引用。
                for ref in refs_for_file:
                    # 去掉 relative_key 前缀中的 expert_id，只保留与目标 bundle 对齐的字段后缀。
                    suffix = ref.relative_key.split(".", 1)[1]

                    # 用字段后缀在目标 bundle 中查找对应的目标张量名字。
                    dst_name = suffix_to_name.get(suffix)

                    # 当目标 bundle 中不需要该字段时，直接跳过。
                    if dst_name is None:
                        continue

                    # 将磁盘读出的 CPU tensor 直接拷贝到目标张量中。
                    dst_tensors[dst_name].copy_(handle.get_tensor(ref.full_key))

    def _build_index(self) -> dict[tuple[str, int], tuple[TensorRef, ...]]:
        # ------------------------------- 优先通过官方 safetensors 索引文件构建权重映射 -------------------------------
        # 拼接官方 safetensors 索引文件路径。
        index_path = self.model_dir / "model.safetensors.index.json"

        # 当索引文件存在时，优先读取其中的 weight_map。
        if index_path.exists():
            # 打开索引文件并读取完整 JSON 内容。
            with index_path.open("r", encoding="utf-8") as f:
                # 从索引 JSON 中提取权重到分片文件的映射表。
                weight_map = json.load(f)["weight_map"]
        else:
            # ------------------------------- 当缺少索引文件时退化为遍历所有 shard 扫描 key -------------------------------
            # 初始化空的权重映射表。
            weight_map = {}

            # 遍历模型目录下的所有 safetensors 分片文件。
            for shard_path in sorted(self.model_dir.glob("*.safetensors")):
                # 以 CPU 设备方式打开当前 shard 文件。
                with safe_open(shard_path, framework="pt", device="cpu") as handle:
                    # 遍历当前 shard 文件中的全部张量 key。
                    for key in handle.keys():
                        # 记录当前张量 key 对应的 shard 文件名。
                        weight_map[key] = shard_path.name

        # ------------------------------- 从全量权重映射中过滤并建立 expert 索引 -------------------------------
        # 用于保存“层名 + expert_id -> TensorRef 列表”的临时索引表。
        expert_refs: dict[tuple[str, int], list[TensorRef]] = defaultdict(list)

        # 遍历全量权重映射表中的每一个张量条目。
        for full_key, file_name in weight_map.items():
            # 使用 expert key 正则表达式匹配当前权重名。
            match = _EXPERT_KEY_RE.match(full_key)

            # 当当前张量不是 expert 张量时，不参与 tiered cache 索引。
            if match is None:
                continue

            # 提取匹配结果中的层名前缀。
            layer_name = match.group("prefix")

            # 提取匹配结果中的 expert 编号并转换为整数。
            expert_id = int(match.group("expert"))

            # 提取当前 expert 内部参数字段后缀。
            suffix = match.group("suffix")

            # 将当前张量条目追加到对应“层名 + expert_id”的索引列表中。
            expert_refs[(layer_name, expert_id)].append(
                TensorRef(
                    file_name=file_name,
                    full_key=full_key,
                    relative_key=f"{expert_id}.{suffix}",
                )
            )

        # ------------------------------- 打印 expert 索引构建完成日志 -------------------------------
        # 输出索引构建完成后的 expert 条目数量与模型目录路径。
        logger.info(
            "Indexed %d local expert entries from %s",
            len(expert_refs),
            self.model_dir,
        )

        # ------------------------------- 将内部索引冻结为 tuple 结构并返回 -------------------------------
        # 将每个 expert 的 TensorRef 列表冻结成 tuple，避免后续被误修改。
        return {key: tuple(value) for key, value in expert_refs.items()}

    def _resolve_refs(
        self, layer_name: str, expert_id: int
    ) -> tuple[TensorRef, ...] | None:
        # ------------------------------- 依次尝试层名别名并解析目标 expert 的张量引用 -------------------------------
        # 同一层在不同模型实现中可能对应不同前缀，这里依次尝试所有候选别名。
        for candidate in _iter_layer_name_aliases(layer_name):
            # 从内部索引中读取当前候选层名和 expert_id 对应的张量引用集合。
            refs = self._expert_refs.get((candidate, expert_id))

            # 当找到匹配的张量引用集合时，直接返回。
            if refs is not None:
                return refs

        # 当全部候选层名都未命中时，返回 None。
        return None


def _iter_layer_name_aliases(layer_name: str) -> tuple[str, ...]:
    # ------------------------------- 构造当前层名的候选别名列表 -------------------------------
    # 初始化别名列表，用于收集同一层在不同命名空间下可能出现的层名形式。
    aliases: list[str] = []

    # ------------------------------- 定义去重追加别名的本地辅助函数 -------------------------------
    # 定义本地辅助函数，用于仅在候选别名非空且未出现过时追加到别名列表中。
    def add(candidate: str) -> None:
        # 当候选别名非空且当前列表中尚不存在时，将其追加到别名列表中。
        if candidate and candidate not in aliases:
            aliases.append(candidate)

    # ------------------------------- 先加入原始层名与 model 前缀互补形式 -------------------------------
    # 将原始层名加入候选别名列表，作为最基础的匹配候选。
    add(layer_name)

    # 当原始层名已经带有 model. 前缀时，再补一个去掉该前缀的候选形式。
    if layer_name.startswith("model."):
        add(layer_name.removeprefix("model."))
    else:
        # 当原始层名不带 model. 前缀时，再补一个带 model. 前缀的候选形式。
        add(f"model.{layer_name}")

    # ------------------------------- 处理 language_model.model 与 model.language_model 两套前缀变体 -------------------------------
    # 当层名以 language_model.model. 开头时，补出对应的 model.language_model.* 与 model.* 形式。
    if layer_name.startswith("language_model.model."):
        # 提取去掉 language_model.model. 前缀后的层名后缀部分。
        suffix = layer_name.removeprefix("language_model.model.")

        # 补出 model.language_model.* 形式的候选层名。
        add(f"model.language_model.{suffix}")

        # 补出 model.* 形式的候选层名。
        add(f"model.{suffix}")

    # 当层名以 model.language_model. 开头时，补出对应的 language_model.model.* 与 language_model.* 形式。
    if layer_name.startswith("model.language_model."):
        # 提取去掉 model.language_model. 前缀后的层名后缀部分。
        suffix = layer_name.removeprefix("model.language_model.")

        # 补出 language_model.model.* 形式的候选层名。
        add(f"language_model.model.{suffix}")

        # 补出 language_model.* 形式的候选层名。
        add(f"language_model.{suffix}")

    # ------------------------------- 处理 mtp 层在带或不带 model 前缀下的命名变体 -------------------------------
    # 当层名以 mtp. 开头时，补出带 model. 前缀的候选形式。
    if layer_name.startswith("mtp."):
        add(f"model.{layer_name}")

    # 当层名以 model.mtp. 开头时，补出去掉 model. 前缀的候选形式。
    if layer_name.startswith("model.mtp."):
        add(layer_name.removeprefix("model."))

    # ------------------------------- 返回去重后的层名别名元组 -------------------------------
    # 将收集完成的别名列表转换为元组并返回。
    return tuple(aliases)
