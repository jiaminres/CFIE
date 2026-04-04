"""Weight-manifest based shard source for the CFIE training runtime."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

from safetensors import safe_open
import torch

from cfie_training.config import TrainingProjectConfig
from cfie_training.runtime.types import ParameterShardSnapshot, ParameterSourceSlice


@dataclass(slots=True, frozen=True)
class WeightTensorRef:
    tensor_name: str
    file_name: str


@dataclass(slots=True, frozen=True)
class WeightShardSource:
    group_id: str
    component: str
    model_path: str
    tensor_refs: tuple[WeightTensorRef, ...]

    # 判断当前分片是否成功匹配到任何权重张量。
    @property
    def matched(self) -> bool:
        return bool(self.tensor_refs)

    # 返回当前分片涉及的去重文件名列表。
    @property
    def file_names(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(ref.file_name for ref in self.tensor_refs))


@dataclass(slots=True, frozen=True)
class ParameterBufferLoadResult:
    values: tuple[float, ...] | torch.Tensor
    used_file_names: tuple[str, ...]
    used_tensor_count: int
    source_layout: tuple[ParameterSourceSlice, ...]


@dataclass(slots=True, frozen=True)
class ParameterBufferSourcePlan:
    used_file_names: tuple[str, ...]
    used_tensor_count: int
    source_layout: tuple[ParameterSourceSlice, ...]


class LocalWeightManifest:
    # 初始化本地权重清单，并读取 safetensors index。
    def __init__(self, config: TrainingProjectConfig) -> None:
        # -----------------
        # 先记录模型路径、索引路径与 bucket 规划。
        # 保存训练配置，后续解析 tensor 规则都会依赖它。
        self.config = config
        # 记录模型根目录。
        self._model_path = Path(config.model_source.model_path)
        # 记录 safetensors index 文件路径。
        self._index_path = self._model_path / config.model_source.index_filename
        # 初始化“张量名 -> 文件名”的映射表。
        self._weight_map: dict[str, str] = {}
        # 初始化文件大小缓存。
        self._file_size_cache: dict[str, int] = {}
        from cfie_training.runtime.planner import LayerBucketPlanner

        # 预先构造 layer bucket 规划，后面分片映射会用到。
        self._bucket_plans = LayerBucketPlanner(config).build()
        # 建立 bucket_id 到 bucket 对象的查找表。
        self._bucket_plan_map = {
            bucket.bucket_id: bucket for bucket in self._bucket_plans
        }

        # -----------------
        # 若配置允许且索引存在，则加载 weight_map。
        if (
            config.model_source.use_local_weight_manifest
            and config.model_source.model_path
            and self._index_path.exists()
        ):
            # 读取并解析 index JSON 文件。
            payload = json.loads(self._index_path.read_text(encoding="utf-8"))
            # 尝试从 payload 中取出 weight_map 字段。
            weight_map = payload.get("weight_map", {})
            if isinstance(weight_map, dict):
                # 规范化 key / value 类型，得到稳定的字符串映射表。
                self._weight_map = {
                    str(key): str(value) for key, value in weight_map.items()
                }

    # 判断本地权重清单是否可用。
    @property
    def available(self) -> bool:
        # 只要 weight_map 非空，就认为 manifest 可用。
        return bool(self._weight_map)

    # 返回模型根目录路径。
    @property
    def model_path(self) -> str:
        # 统一返回字符串形式的模型根目录。
        return str(self._model_path)

    # 将文件名解析为模型目录下的绝对路径对象。
    def resolve_file_path(self, file_name: str) -> Path:
        # 默认所有权重文件都相对模型根目录寻址。
        return self._model_path / file_name

    # 返回指定权重文件大小，并做本地缓存。
    def file_size_bytes(self, file_name: str) -> int:
        # 先查文件大小缓存，命中则直接返回。
        cached = self._file_size_cache.get(file_name)
        if cached is not None:
            return cached
        # 解析当前文件的实际路径。
        file_path = self.resolve_file_path(file_name)
        # 文件存在时读取 stat 大小，不存在时按 0 处理。
        size = file_path.stat().st_size if file_path.exists() else 0
        # 把结果写入缓存，避免重复 stat。
        self._file_size_cache[file_name] = size
        # 返回文件大小。
        return size

    # 根据张量名查找其在 manifest 中对应的文件引用。
    def tensor_ref(self, tensor_name: str) -> WeightTensorRef | None:
        # 先从 weight_map 里查文件名。
        file_name = self._weight_map.get(tensor_name)
        # 查不到时返回 None。
        if file_name is None:
            return None
        # 命中时返回张量名和文件名组成的引用对象。
        return WeightTensorRef(tensor_name=tensor_name, file_name=file_name)

    # 按张量名从 safetensors 文件中读取权重。
    def load_tensor(
        self,
        tensor_name: str,
        *,
        dtype: torch.dtype | None = torch.float32,
    ) -> torch.Tensor | None:
        # 先定位张量对应的文件引用。
        tensor_ref = self.tensor_ref(tensor_name)
        # manifest 里没有该张量时直接返回 None。
        if tensor_ref is None:
            return None
        # 解析该张量所在的 safetensors 文件路径。
        file_path = self.resolve_file_path(tensor_ref.file_name)
        # 文件不存在时直接返回 None。
        if not file_path.exists():
            return None
        try:
            # 以 CPU 模式打开 safetensors 文件。
            with safe_open(file_path, framework="pt", device="cpu") as handle:
                # 读取目标张量。
                tensor = handle.get_tensor(tensor_name)
        except (KeyError, RuntimeError, ValueError):
            # 任意读取异常都按“当前张量不可用”处理。
            return None
        # 调用方要求显式 dtype 时，统一转换 dtype。
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        # 返回读取到的张量。
        return tensor

    # 返回指定层 router gate 的张量引用。
    def router_gate_ref(self, layer_index: int) -> WeightTensorRef | None:
        # 负层号无效，直接返回 None。
        if layer_index < 0:
            return None
        # 普通文本层直接映射到 language_model.layers 下的 gate.weight。
        if layer_index < self.config.model_spec.num_hidden_layers:
            return self.tensor_ref(
                f"model.language_model.layers.{layer_index}.mlp.gate.weight"
            )
        # 超出普通层范围时，把统一层索引换算成 MTP 相对层号。
        mtp_index = layer_index - self.config.model_spec.num_hidden_layers
        # 超过 MTP 层范围时返回 None。
        if mtp_index >= self.config.model_spec.mtp_num_hidden_layers:
            return None
        # 否则返回对应 MTP 层的 gate.weight 引用。
        return self.tensor_ref(f"mtp.layers.{mtp_index}.mlp.gate.weight")

    # 读取指定层的 router gate 权重张量。
    def load_router_gate_tensor(
        self,
        layer_index: int,
        *,
        dtype: torch.dtype | None = torch.float32,
    ) -> torch.Tensor | None:
        # 先查当前层的 router gate 引用。
        tensor_ref = self.router_gate_ref(layer_index)
        # 没有匹配到引用时直接返回 None。
        if tensor_ref is None:
            return None
        # 有引用时继续走通用张量读取逻辑。
        return self.load_tensor(tensor_ref.tensor_name, dtype=dtype)

    # 根据 bucket id 返回该 bucket 覆盖的层索引。
    def _bucket_layer_indices(self, bucket_id: int | None) -> tuple[int, ...]:
        # bucket_id 为空时返回空元组。
        if bucket_id is None:
            return ()
        # 从 bucket 映射表中取目标 bucket。
        bucket = self._bucket_plan_map.get(bucket_id)
        # bucket 不存在时返回空元组。
        if bucket is None:
            return ()
        # 返回该 bucket 覆盖的层索引集合。
        return bucket.layer_indices

    # 根据统一层索引返回对应的张量名前缀。
    def _tensor_prefix_for_layer(self, layer_index: int) -> str:
        # 普通文本层用 language_model.layers 前缀。
        if layer_index < self.config.model_spec.num_hidden_layers:
            return f"model.language_model.layers.{layer_index}."
        # MTP 层需要先换算成相对层号。
        mtp_index = layer_index - self.config.model_spec.num_hidden_layers
        # 返回 MTP 层张量名前缀。
        return f"mtp.layers.{mtp_index}."

    # 从专家张量名中解析 routed expert id。
    def _expert_id_for_tensor(self, tensor_name: str) -> int | None:
        # 只有包含 `.mlp.experts.` 的张量才可能带 expert id。
        marker = ".mlp.experts."
        if marker not in tensor_name:
            return None
        # 取出 marker 后面的后缀部分。
        suffix = tensor_name.split(marker, 1)[1]
        # expert id 位于后缀的第一段。
        expert_token, _, _ = suffix.partition(".")
        # 不是纯数字时无法解析 expert id。
        if not expert_token.isdigit():
            return None
        # 转成整数返回。
        return int(expert_token)

    # 判断张量是否为 GPTQ 主 qweight。
    def _is_gptq_qweight_tensor(self, tensor_name: str) -> bool:
        # GPTQ 主权重统一以 `.qweight` 结尾。
        return tensor_name.endswith(".qweight")

    # 判断张量是否为 GPTQ 辅助张量。
    def _is_gptq_aux_tensor(self, tensor_name: str) -> bool:
        # qzeros / scales / g_idx 都属于 GPTQ 辅助张量。
        return tensor_name.endswith((".qzeros", ".scales", ".g_idx"))

    # 判断张量是否应计入逻辑权重集合。
    def _is_logical_weight_tensor(self, tensor_name: str) -> bool:
        # GPTQ qweight 本身一定计入逻辑权重。
        if self._is_gptq_qweight_tensor(tensor_name):
            return True
        # GPTQ 辅助张量只服务于解包，不计入逻辑权重集合。
        if self._is_gptq_aux_tensor(tensor_name):
            return False
        # 专家张量只保留 gate/up/down 这些逻辑权重。
        if ".mlp.experts." in tensor_name:
            return (
                tensor_name.endswith(".weight")
                or tensor_name.endswith(".gate_up_proj")
                or tensor_name.endswith(".down_proj")
            )
        # 其余普通张量默认都计入逻辑权重集合。
        return True

    # 根据 qweight 名称推导其 GPTQ 配套张量名。
    def _gptq_companion_names(
        self,
        qweight_name: str,
    ) -> tuple[str, str, str]:
        # 先去掉 `.qweight` 后缀拿到公共 stem。
        stem = qweight_name[: -len(".qweight")]
        # 再按 GPTQ 命名规则补齐三个伴生张量名。
        return (
            f"{stem}.qzeros",
            f"{stem}.scales",
            f"{stem}.g_idx",
        )

    # 返回当前运行时配置下的 GPTQ bit 数。
    def _gptq_bits(self) -> int:
        # 优先使用运行时量化配置里的 bits。
        return (
            self.config.runtime_quantization.bits
            # 其次回退到模型配置里的 bits。
            or self.config.model_spec.quant_bits
            # 两者都没有时默认按 4 bit 处理。
            or 4
        )

    # 返回一个 int32 word 中可打包的权重量个数。
    def _gptq_pack_factor(self) -> int:
        # int32 一共 32 位，因此 pack_factor=32/bits。
        return 32 // max(self._gptq_bits(), 1)

    # 将 GPTQ 打包 qweight 还原为按逻辑行展开的整数矩阵。
    def _unpack_qweight_rows(
        self,
        packed: torch.Tensor,
        *,
        row_count: int,
    ) -> torch.Tensor:
        # 读取当前 GPTQ bit 数。
        bits = self._gptq_bits()
        # 生成取低 bits 位的掩码。
        mask = (1 << bits) - 1
        # 生成一个 int32 word 内各量化值的位移量。
        shifts = torch.arange(
            0,
            32,
            bits,
            dtype=torch.int64,
            device=packed.device,
        )
        # 把 packed 权重扩成 int64 并新增一维，便于做逐位右移。
        words = packed.to(dtype=torch.int64).unsqueeze(-1)
        # 先右移再与掩码做按位与，得到解包后的整型量化值。
        unpacked = torch.bitwise_and(
            torch.bitwise_right_shift(words, shifts),
            mask,
        )
        # 把解包结果重排成逻辑上的 [rows, cols] 矩阵并裁到 row_count。
        return unpacked.permute(0, 2, 1).reshape(-1, packed.shape[1])[
            :row_count
        ].to(dtype=torch.int32)

    # 将 GPTQ 打包 qzeros 还原为按列展开的整数矩阵。
    def _unpack_qzeros_cols(
        self,
        packed: torch.Tensor,
        *,
        col_count: int,
    ) -> torch.Tensor:
        # 读取当前 GPTQ bit 数。
        bits = self._gptq_bits()
        # 生成取低 bits 位的掩码。
        mask = (1 << bits) - 1
        # 生成一个 int32 word 内各量化值的位移量。
        shifts = torch.arange(
            0,
            32,
            bits,
            dtype=torch.int64,
            device=packed.device,
        )
        # 把 packed qzeros 扩成 int64 并新增一维。
        words = packed.to(dtype=torch.int64).unsqueeze(-1)
        # 先右移再取掩码，得到解包后的零点值。
        unpacked = torch.bitwise_and(
            torch.bitwise_right_shift(words, shifts),
            mask,
        )
        # 重排成 [groups, cols] 形式并裁到 col_count。
        return unpacked.reshape(packed.shape[0], -1)[:, :col_count].to(
            dtype=torch.int32
        )

    # 返回张量在逻辑未打包语义下的形状。
    def _logical_shape_for_ref(
        self,
        handle,
        tensor_name: str,
    ) -> tuple[int, ...]:
        # 非 GPTQ qweight 张量直接返回其原始 shape。
        if not self._is_gptq_qweight_tensor(tensor_name):
            return tuple(handle.get_slice(tensor_name).get_shape())
        # 读取 qweight 自身的物理 shape。
        qweight_shape = tuple(handle.get_slice(tensor_name).get_shape())
        # 推导出与之配套的 g_idx 张量名。
        _, _, g_idx_name = self._gptq_companion_names(tensor_name)
        # 读取 g_idx 的 shape。
        g_idx_shape = tuple(handle.get_slice(g_idx_name).get_shape())
        # shape 不满足预期时退回物理 qweight shape。
        if len(qweight_shape) != 2 or len(g_idx_shape) != 1:
            return qweight_shape
        # 逻辑 shape 的行数由 g_idx 长度决定，列数沿用 qweight 列数。
        return (g_idx_shape[0], qweight_shape[1])

    # 按预算读取并解包 GPTQ 张量的一小块逻辑权重。
    def _gptq_chunk_for_ref(
        self,
        *,
        handle,
        tensor_name: str,
        remaining: int,
    ) -> tuple[torch.Tensor, tuple[int, ...]] | None:
        # -----------------
        # 先根据逻辑形状与预算确定本次读取块大小。
        # 取出当前 qweight 的逻辑 shape。
        logical_shape = self._logical_shape_for_ref(handle, tensor_name)
        # 当前只支持二维逻辑权重矩阵。
        if len(logical_shape) != 2:
            return None
        # 拆出逻辑行列数。
        logical_rows, logical_cols = logical_shape
        # 在预算内选择一个尽量保形的矩形块。
        row_count, col_count = self._matrix_block_shape(
            rows=logical_rows,
            cols=logical_cols,
            budget=remaining,
        )
        # 无法切出有效块时返回 None。
        if row_count <= 0 or col_count <= 0:
            return None
        # 推导 qzeros / scales / g_idx 三个伴生张量名。
        qzeros_name, scales_name, g_idx_name = self._gptq_companion_names(tensor_name)
        # 读取 pack_factor，后面用于把逻辑维度换成物理维度。
        pack_factor = self._gptq_pack_factor()
        # 计算覆盖 row_count 逻辑行需要多少物理 qweight 行。
        qweight_rows = math.ceil(row_count / pack_factor)
        # 只读取当前预算所需的 qweight 子块。
        qweight = handle.get_slice(tensor_name)[:qweight_rows, :col_count]
        # 读取当前逻辑行对应的 g_idx。
        g_idx = handle.get_slice(g_idx_name)[:row_count].to(dtype=torch.int64)
        # g_idx 为空时无法恢复逻辑权重。
        if g_idx.numel() <= 0:
            return None
        # group 数等于 g_idx 最大值加一。
        group_count = int(g_idx.max().item()) + 1
        # 计算覆盖 col_count 逻辑列需要多少物理 qzeros 列。
        qzeros_cols = math.ceil(col_count / pack_factor)
        # 读取预算内需要的 qzeros 子块。
        qzeros = handle.get_slice(qzeros_name)[:group_count, :qzeros_cols]
        # 读取预算内需要的 scales 子块。
        scales = handle.get_slice(scales_name)[:group_count, :col_count].to(
            dtype=torch.float32
        )

        # -----------------
        # 解包 qweight / qzeros，并恢复成浮点逻辑权重块。
        unpacked_weight = self._unpack_qweight_rows(
            qweight,
            row_count=row_count,
        ).to(dtype=torch.float32)
        # 把 qzeros 解包成零点矩阵。
        unpacked_zero_points = self._unpack_qzeros_cols(
            qzeros,
            col_count=col_count,
        ).to(dtype=torch.float32)
        # 按 g_idx 把每行映射到对应的 scale group。
        scale_groups = scales.index_select(0, g_idx)
        # 按 g_idx 把每行映射到对应的 zero-point group。
        zero_point_groups = unpacked_zero_points.index_select(0, g_idx)
        # 用 `(qweight - zero_point) * scale` 恢复浮点逻辑权重。
        chunk = (unpacked_weight - zero_point_groups) * scale_groups
        # 返回连续存储的逻辑块及其逻辑 shape。
        return chunk.contiguous(), logical_shape

    # 读取并解包整个 GPTQ 张量的逻辑权重。
    def _gptq_full_chunk_for_ref(
        self,
        *,
        handle,
        tensor_name: str,
    ) -> tuple[torch.Tensor, tuple[int, ...]] | None:
        # -----------------
        # 读取完整 qweight / qzeros / scales / g_idx 张量。
        logical_shape = self._logical_shape_for_ref(handle, tensor_name)
        if len(logical_shape) != 2:
            return None
        logical_rows, logical_cols = logical_shape
        qzeros_name, scales_name, g_idx_name = self._gptq_companion_names(tensor_name)
        qweight = handle.get_tensor(tensor_name)
        g_idx = handle.get_tensor(g_idx_name).to(dtype=torch.int64)
        if g_idx.numel() <= 0:
            return None
        group_count = int(g_idx.max().item()) + 1
        qzeros = handle.get_tensor(qzeros_name)[:group_count]
        scales = handle.get_tensor(scales_name)[:group_count].to(dtype=torch.float32)

        # -----------------
        # 解包后按 group 索引恢复出完整浮点逻辑权重。
        unpacked_weight = self._unpack_qweight_rows(
            qweight,
            row_count=logical_rows,
        ).to(dtype=torch.float32)
        unpacked_zero_points = self._unpack_qzeros_cols(
            qzeros,
            col_count=logical_cols,
        ).to(dtype=torch.float32)
        scale_groups = scales.index_select(0, g_idx)
        zero_point_groups = unpacked_zero_points.index_select(0, g_idx)
        chunk = (unpacked_weight - zero_point_groups) * scale_groups
        return chunk.contiguous(), logical_shape

    # 判断张量是否属于某组层索引下的可训练层参数。
    def _is_trainable_layer_tensor(
        self,
        tensor_name: str,
        *,
        layer_indices: tuple[int, ...],
        include_experts: bool,
    ) -> bool:
        for layer_index in layer_indices:
            # 先为当前层生成张量名前缀。
            prefix = self._tensor_prefix_for_layer(layer_index)
            # 前缀不匹配时，说明当前张量不属于该层。
            if not tensor_name.startswith(prefix):
                continue
            # 命中前缀后，再判断它是否是专家张量。
            is_expert_tensor = ".mlp.experts." in tensor_name
            # include_experts=True 时只保留专家张量。
            if include_experts:
                return is_expert_tensor
            # include_experts=False 时只保留非专家张量。
            return not is_expert_tensor
        # 所有层都不匹配时返回 False。
        return False

    # 为指定参数分片匹配其对应的 manifest 张量集合。
    def _tensor_refs_for_shard(
        self,
        shard: ParameterShardSnapshot,
    ) -> tuple[WeightTensorRef, ...]:
        # -----------------
        # 按分片组件类型选择不同的张量过滤规则。
        # manifest 不可用时没有任何张量可匹配。
        if not self.available:
            return ()
        # 用列表暂存最终命中的张量名。
        tensor_names: list[str] = []
        if shard.component == "static_modules":
            # static_modules 只保留不属于任意层 / MTP 层的全局张量。
            tensor_names = [
                name
                for name in sorted(self._weight_map)
                if (
                    not name.startswith("model.language_model.layers.")
                    and not name.startswith("mtp.")
                )
            ]
        elif shard.component == "bucket_non_routed":
            # 先取出当前 bucket 覆盖的层索引。
            layer_indices = self._bucket_layer_indices(shard.bucket_id)
            # 只保留这些层里非专家的可训练张量。
            tensor_names = [
                name
                for name in sorted(self._weight_map)
                if self._is_trainable_layer_tensor(
                    name,
                    layer_indices=layer_indices,
                    include_experts=False,
                )
            ]
        elif shard.component == "bucket_active_experts":
            # 当前 active expert 分片同样只关心 bucket 覆盖的层。
            layer_indices = self._bucket_layer_indices(shard.bucket_id)
            # 把 active expert id 集合转成 frozenset，便于高频判断。
            active_expert_ids = frozenset(shard.expert_ids)
            # 只保留当前层里的专家逻辑权重，并按 active_expert_ids 过滤。
            tensor_names = [
                name
                for name in sorted(self._weight_map)
                if self._is_trainable_layer_tensor(
                    name,
                    layer_indices=layer_indices,
                    include_experts=True,
                )
                and self._is_logical_weight_tensor(name)
                and (
                    not active_expert_ids
                    or (expert_id := self._expert_id_for_tensor(name)) is None
                    or expert_id in active_expert_ids
                )
            ]
        elif shard.component == "expert_window_prefetch":
            # prefetch 分片只关心所有层里的专家逻辑权重。
            tensor_names = [
                name
                for name in sorted(self._weight_map)
                if ".mlp.experts." in name
                and self._is_logical_weight_tensor(name)
                and (
                    name.startswith("model.language_model.layers.")
                    or name.startswith("mtp.layers.")
                )
            ]

        # -----------------
        # 将张量名映射回文件引用对象。
        return tuple(
            WeightTensorRef(tensor_name=name, file_name=self._weight_map[name])
            for name in tensor_names
        )

    # 从张量名中解析统一层索引。
    def _layer_index_for_tensor(self, tensor_name: str) -> int | None:
        # 普通文本层张量统一以这个前缀开头。
        prefix = "model.language_model.layers."
        # MTP 层张量统一以这个前缀开头。
        mtp_prefix = "mtp.layers."
        # 先判断是否属于普通文本层张量。
        if tensor_name.startswith(prefix):
            # 去掉前缀后只保留层号开头部分。
            suffix = tensor_name[len(prefix):]
            # 层号位于第一个点号之前。
            layer_id, _, _ = suffix.partition(".")
            # 层号不是纯数字时无法解析。
            if not layer_id.isdigit():
                return None
            # 普通层直接返回其整数层号。
            return int(layer_id)
        # 再判断是否属于 MTP 层张量。
        if tensor_name.startswith(mtp_prefix):
            # 去掉 MTP 前缀后继续解析层号。
            suffix = tensor_name[len(mtp_prefix):]
            # 层号同样位于第一个点号之前。
            layer_id, _, _ = suffix.partition(".")
            # 层号不是纯数字时无法解析。
            if not layer_id.isdigit():
                return None
            # MTP 层统一追加到普通层数量之后。
            return self.config.model_spec.num_hidden_layers + int(layer_id)
        # 既不是普通层也不是 MTP 层时返回 None。
        return None

    # 根据张量名推断其语义角色标签。
    def _semantic_role_for_tensor(self, tensor_name: str) -> str:
        # 下面按“更具体优先”的顺序匹配语义角色。
        if ".input_layernorm.weight" in tensor_name:
            return "input_layernorm"
        if ".post_attention_layernorm.weight" in tensor_name:
            return "post_attention_layernorm"
        if ".linear_attn.A_log" in tensor_name:
            return "linear_attn_A_log"
        if ".linear_attn.conv1d.weight" in tensor_name:
            return "linear_attn_conv1d"
        if ".linear_attn.dt_bias" in tensor_name:
            return "linear_attn_dt_bias"
        if ".linear_attn.in_proj_a.weight" in tensor_name:
            return "linear_attn_in_proj_a"
        if ".linear_attn.in_proj_b.weight" in tensor_name:
            return "linear_attn_in_proj_b"
        if ".linear_attn.in_proj_qkv.weight" in tensor_name:
            return "linear_attn_in_proj_qkv"
        if ".linear_attn.in_proj_z.weight" in tensor_name:
            return "linear_attn_in_proj_z"
        if ".linear_attn.norm.weight" in tensor_name:
            return "linear_attn_norm"
        if ".linear_attn.out_proj.weight" in tensor_name:
            return "linear_attn_out_proj"
        if ".self_attn.q_norm.weight" in tensor_name:
            return "self_attn_q_norm"
        if ".self_attn.k_norm.weight" in tensor_name:
            return "self_attn_k_norm"
        if ".self_attn.q_proj.weight" in tensor_name:
            return "self_attn_q_proj"
        if ".self_attn.k_proj.weight" in tensor_name:
            return "self_attn_k_proj"
        if ".self_attn.v_proj.weight" in tensor_name:
            return "self_attn_v_proj"
        if ".self_attn.o_proj.weight" in tensor_name:
            return "self_attn_o_proj"
        if ".mlp.gate.weight" in tensor_name:
            return "mlp_router_gate"
        if ".mlp.shared_expert_gate.weight" in tensor_name:
            return "shared_expert_gate"
        if ".mlp.shared_expert.gate_proj.weight" in tensor_name:
            return "shared_expert_gate_proj"
        if ".mlp.shared_expert.up_proj.weight" in tensor_name:
            return "shared_expert_up_proj"
        if ".mlp.shared_expert.down_proj.weight" in tensor_name:
            return "shared_expert_down_proj"
        if (
            ".mlp.experts." in tensor_name
            and (
                ".gate_up_proj" in tensor_name
                or ".gate_proj." in tensor_name
                or tensor_name.endswith(".gate_proj")
                or ".up_proj." in tensor_name
                or tensor_name.endswith(".up_proj")
            )
        ):
            return "expert_gate_up_proj"
        if (
            ".mlp.experts." in tensor_name
            and (
                ".down_proj." in tensor_name
                or tensor_name.endswith(".down_proj")
            )
        ):
            return "expert_down_proj"
        if tensor_name.startswith("model.language_model.layers.") or tensor_name.startswith(
            "mtp.layers."
        ):
            # 仍属于层内张量但没命中更具体角色时，统一归到 layer_misc。
            return "layer_misc"
        # 其余都视为静态模块张量。
        return "static_module"

    # 为张量定义采样优先级，便于代表性参数抽样。
    def _tensor_sampling_priority(self, tensor_name: str) -> tuple[int, str]:
        # 先把张量映射成语义角色。
        role = self._semantic_role_for_tensor(tensor_name)
        # 角色优先级越小，代表抽样时越优先。
        priority = {
            "self_attn_q_proj": 0,
            "self_attn_k_proj": 0,
            "self_attn_v_proj": 0,
            "self_attn_o_proj": 0,
            "linear_attn_in_proj_qkv": 0,
            "linear_attn_in_proj_a": 0,
            "linear_attn_in_proj_b": 0,
            "linear_attn_out_proj": 0,
            "expert_gate_up_proj": 0,
            "expert_down_proj": 0,
            "mlp_router_gate": 1,
            "shared_expert_up_proj": 1,
            "shared_expert_down_proj": 1,
            "shared_expert_gate_proj": 1,
            "shared_expert_gate": 2,
            "input_layernorm": 2,
            "post_attention_layernorm": 2,
            "self_attn_q_norm": 2,
            "self_attn_k_norm": 2,
            "linear_attn_norm": 2,
            "linear_attn_in_proj_z": 2,
            "linear_attn_A_log": 3,
            "linear_attn_dt_bias": 3,
            "linear_attn_conv1d": 3,
            "layer_misc": 4,
            "static_module": 5,
        }.get(role, 4)
        # 返回“优先级 + 张量名”作为稳定排序键。
        return priority, tensor_name

    # 根据层索引判断该层当前属于哪种注意力类型。
    def _attention_type_for_layer(self, layer_index: int) -> str:
        # 超出普通层范围的统一视为 MTP。
        if layer_index >= self.config.model_spec.num_hidden_layers:
            return "mtp"
        # 读取模型配置里的注意力模式周期。
        pattern = self.config.model_spec.attention_pattern
        # 没配置模式时，默认按 full_attention 处理。
        if not pattern:
            return "full_attention"
        # 否则按层号对周期长度取模得到该层注意力类型。
        return pattern[layer_index % len(pattern)]

    # 为给定层和分片类型构建角色采样模板。
    def _role_sampling_schema(
        self,
        *,
        shard: ParameterShardSnapshot,
        layer_index: int,
        available_roles: tuple[str, ...],
    ) -> tuple[tuple[str, int], ...]:
        # 先把可用角色转成集合，便于快速判断。
        available = set(available_roles)
        # 专家分片只关心 expert gate/up 和 expert down 这两类角色。
        if shard.component == "bucket_active_experts":
            schema = (
                ("expert_gate_up_proj", 5),
                ("expert_down_proj", 5),
            )
        else:
            # 普通 bucket 共享一套 layernorm/router/shared expert 角色模板。
            common_schema = (
                ("input_layernorm", 1),
                ("post_attention_layernorm", 1),
                ("mlp_router_gate", 3),
                ("shared_expert_gate", 1),
                ("shared_expert_gate_proj", 3),
                ("shared_expert_up_proj", 3),
                ("shared_expert_down_proj", 3),
            )
            # full attention 层使用 self_attn 系列角色模板。
            if self._attention_type_for_layer(layer_index) == "full_attention":
                attention_schema = (
                    ("self_attn_q_proj", 5),
                    ("self_attn_k_proj", 4),
                    ("self_attn_v_proj", 4),
                    ("self_attn_o_proj", 4),
                    ("self_attn_q_norm", 1),
                    ("self_attn_k_norm", 1),
                )
            else:
                # linear attention 层使用 linear_attn 系列角色模板。
                attention_schema = (
                    ("linear_attn_in_proj_qkv", 5),
                    ("linear_attn_in_proj_a", 4),
                    ("linear_attn_in_proj_b", 4),
                    ("linear_attn_in_proj_z", 3),
                    ("linear_attn_out_proj", 4),
                    ("linear_attn_norm", 1),
                    ("linear_attn_conv1d", 2),
                    ("linear_attn_dt_bias", 1),
                    ("linear_attn_A_log", 1),
                )
            # 把注意力角色模板和公共模板拼成完整 schema。
            schema = attention_schema + common_schema
        # 只保留当前 ref_group 实际具备的角色。
        weighted_roles = [
            (role, weight)
            for role, weight in schema
            if role in available
        ]
        # 记录已经被 schema 覆盖的角色。
        covered = {role for role, _ in weighted_roles}
        # 对 schema 中没列到但实际存在的角色，补一个默认权重 1。
        for role in sorted(available - covered):
            weighted_roles.append((role, 1))
        # 返回最终的“角色 + 权重”模板。
        return tuple(weighted_roles)

    # 按总预算为每个语义角色分配抽样配额。
    def _role_quotas(
        self,
        *,
        shard: ParameterShardSnapshot,
        layer_index: int,
        available_roles: tuple[str, ...],
        total_budget: int,
    ) -> tuple[tuple[str, int], ...]:
        # 总预算非正或没有可用角色时直接返回空。
        if total_budget <= 0 or not available_roles:
            return ()
        # 先根据分片类型和层类型构建角色权重模板。
        weighted_roles = list(
            self._role_sampling_schema(
                shard=shard,
                layer_index=layer_index,
                available_roles=available_roles,
            )
        )
        # 没有任何角色被保留下来时返回空。
        if not weighted_roles:
            return ()
        # 先按权重高到低、角色名字典序排序。
        weighted_roles.sort(key=lambda item: (-item[1], item[0]))
        # 总预算比角色数还小时，只保留最优先的一批角色。
        if total_budget < len(weighted_roles):
            weighted_roles = weighted_roles[:total_budget]
        # 每个被保留角色先至少分到 1 个预算单位。
        quotas = {role: 1 for role, _ in weighted_roles}
        # 计算剩余可继续分配的预算。
        remaining = total_budget - len(weighted_roles)
        if remaining > 0:
            # 统计当前模板总权重。
            total_weight = sum(weight for _, weight in weighted_roles)
            # 记录每个角色按比例分配后的余数。
            remainders: list[tuple[float, str]] = []
            for role, weight in weighted_roles:
                # 计算该角色理论上可分到的剩余预算。
                share = remaining * (weight / max(total_weight, 1))
                # 先取其整数部分。
                base = int(share)
                # 把整数部分加到当前角色配额上。
                quotas[role] += base
                # 保存小数余量，后面用于补发剩余名额。
                remainders.append((share - base, role))
            # 统计已经分配出去的预算总量。
            assigned = sum(quotas.values())
            # 计算尚未分配完的名额数。
            leftover = total_budget - assigned
            # 按余量从大到小把剩余名额补给对应角色。
            for _, role in sorted(remainders, key=lambda item: (-item[0], item[1]))[
                :leftover
            ]:
                quotas[role] += 1
        # 返回最终“角色 -> 配额”列表。
        return tuple((role, quotas[role]) for role, _ in weighted_roles)

    # 将张量引用按层分组，供后续分层采样使用。
    def _sampling_layer_groups(
        self,
        *,
        shard: ParameterShardSnapshot,
        tensor_refs: tuple[WeightTensorRef, ...],
    ) -> tuple[tuple[int, tuple[WeightTensorRef, ...]], ...]:
        # 非层内分片直接把所有 ref 放到一个伪层组里。
        if shard.component not in {"bucket_non_routed", "bucket_active_experts"}:
            return ((-1, tensor_refs),) if tensor_refs else ()
        # 用字典临时收集“层索引 -> 对应张量 refs”。
        grouped: dict[int, list[WeightTensorRef]] = {}
        for ref in tensor_refs:
            # 先解析当前张量所属的统一层索引。
            layer_index = self._layer_index_for_tensor(ref.tensor_name)
            # 解析失败时直接跳过当前 ref。
            if layer_index is None:
                continue
            # 把当前 ref 挂到对应层索引下面。
            grouped.setdefault(layer_index, []).append(ref)
        # 对每层内部的 ref 再按采样优先级排序。
        return tuple(
            (
                layer_index,
                tuple(
                sorted(
                    grouped[layer_index],
                    key=lambda ref: self._tensor_sampling_priority(ref.tensor_name),
                )
                ),
            )
            for layer_index in sorted(grouped)
        )

    # 将一组张量按语义角色分组。
    def _refs_by_role(
        self,
        refs: tuple[WeightTensorRef, ...],
    ) -> dict[str, tuple[WeightTensorRef, ...]]:
        # 用字典临时收集“角色 -> 对应 refs”。
        grouped: dict[str, list[WeightTensorRef]] = {}
        for ref in refs:
            # 为当前张量解析语义角色。
            role = self._semantic_role_for_tensor(ref.tensor_name)
            # 把 ref 追加到对应角色下面。
            grouped.setdefault(role, []).append(ref)
        # 对每个角色内部的 ref 再按采样优先级排序。
        return {
            role: tuple(
                sorted(
                    role_refs,
                    key=lambda ref: self._tensor_sampling_priority(ref.tensor_name),
                )
            )
            for role, role_refs in grouped.items()
        }

    # 将总预算平均分给若干层分组。
    def _group_quotas(
        self,
        *,
        group_count: int,
        total_budget: int,
    ) -> tuple[int, ...]:
        # 分组数或总预算非正时直接返回空。
        if group_count <= 0 or total_budget <= 0:
            return ()
        # 先给每组分配相同的基准配额。
        base = total_budget // group_count
        # 计算需要额外补给前几组的剩余额度。
        remainder = total_budget % group_count
        # 前 remainder 组各多拿 1，其余组拿 base。
        return tuple(
            base + (1 if group_index < remainder else 0)
            for group_index in range(group_count)
        )

    # 在代表性抽样模式下，从 safetensors slice 中取出一个小块。
    def _slice_chunk_for_ref(
        self,
        *,
        shard: ParameterShardSnapshot,
        safe_slice,
        remaining: int,
    ) -> torch.Tensor | None:
        # -----------------
        # 先按张量维度选择不同的切片策略。
        # 读取当前 slice 的 shape。
        shape = safe_slice.get_shape()
        # 一维张量直接裁前 remaining 个元素。
        if len(shape) == 1:
            return safe_slice[: min(shape[0], remaining)]
        # 二维张量按预算选一个矩形子块。
        if len(shape) == 2:
            row_count, col_count = self._matrix_block_shape(
                rows=shape[0],
                cols=shape[1],
                budget=remaining,
            )
            return safe_slice[:row_count, :col_count]
        # 非 1/2/3 维张量当前不支持代表性抽样。
        if len(shape) != 3:
            return None
        # active expert 三维张量需要按 expert 维度优先切片。
        if shard.component == "bucket_active_experts" and shard.expert_ids:
            selected_expert_ids, row_count, col_count = self._expert_block_plan(
                shard=shard,
                shape=shape,
                budget=remaining,
            )
            # 逐个 expert 读取对应的小块。
            expert_chunks = []
            for expert_id in selected_expert_ids:
                expert_chunks.append(
                    safe_slice[
                        expert_id:expert_id + 1,
                        :row_count,
                        :col_count,
                    ]
                )
                # 累积元素数达到预算时提前停止。
                if sum(chunk.numel() for chunk in expert_chunks) >= remaining:
                    break
            # 一个 expert 都没读到时返回 None。
            if not expert_chunks:
                return None
            # 只有一个 expert 块时直接返回该块。
            if len(expert_chunks) == 1:
                return expert_chunks[0]
            # 多个 expert 块时沿 expert 维拼接。
            return torch.cat(expert_chunks, dim=0)

        # -----------------
        # 非专家三维张量只取最浅 depth 和预算内的矩形块。
        depth = min(shape[0], 1)
        # 其余三维张量默认只取最浅 depth=1 的切片。
        row_count, col_count = self._matrix_block_shape(
            rows=shape[1],
            cols=shape[2],
            budget=max(1, remaining // depth),
        )
        # 返回裁好的三维子块。
        return safe_slice[:depth, :row_count, :col_count]

    # 在全量物化模式下读取完整张量或完整 expert 子集。
    def _full_slice_chunk_for_ref(
        self,
        *,
        shard: ParameterShardSnapshot,
        safe_slice,
    ) -> torch.Tensor | None:
        # 读取当前 slice 的 shape。
        shape = safe_slice.get_shape()
        # 一维和二维张量在全量模式下直接整块返回。
        if len(shape) in {1, 2}:
            return safe_slice[:]
        # 非 1/2/3 维张量当前不支持全量读取。
        if len(shape) != 3:
            return None
        # active expert 三维张量只取指定的 expert 子集。
        if shard.component == "bucket_active_experts" and shard.expert_ids:
            expert_chunks = [
                safe_slice[expert_id:expert_id + 1, :, :]
                for expert_id in shard.expert_ids
                if 0 <= expert_id < shape[0]
            ]
            # 一个合法 expert 都没有时返回 None。
            if not expert_chunks:
                return None
            # 只有一个 expert 时直接返回。
            if len(expert_chunks) == 1:
                return expert_chunks[0]
            # 多个 expert 时沿 expert 维拼接。
            return torch.cat(expert_chunks, dim=0)
        # 其余三维张量在全量模式下整块返回。
        return safe_slice[:]

    # 返回全量物化模式下某个普通张量实际会读取出的切片形状。
    def _full_slice_shape_for_ref(
        self,
        *,
        shard: ParameterShardSnapshot,
        tensor_shape: tuple[int, ...],
    ) -> tuple[int, ...] | None:
        # 一维和二维张量在全量模式下直接保留原始 shape。
        if len(tensor_shape) in {1, 2}:
            return tensor_shape
        # 非 1/2/3 维张量当前不支持全量读取。
        if len(tensor_shape) != 3:
            return None
        # active expert 三维张量只保留当前 shard 选中的 expert 子集。
        if shard.component == "bucket_active_experts" and shard.expert_ids:
            selected_expert_count = sum(
                1 for expert_id in shard.expert_ids if 0 <= expert_id < tensor_shape[0]
            )
            # 没有任何合法 expert 时返回 None。
            if selected_expert_count <= 0:
                return None
            # 返回“选中的 expert 数 + 原始二维 weight shape”。
            return (
                selected_expert_count,
                tensor_shape[1],
                tensor_shape[2],
            )
        # 其余三维张量在全量模式下保留原始 shape。
        return tensor_shape

    # 估算某个张量切片在当前预算下会贡献多少元素。
    def _planned_contribution_length(
        self,
        *,
        shard: ParameterShardSnapshot,
        shape: list[int],
        remaining: int,
    ) -> int:
        # 一维张量的贡献长度就是预算内可取到的元素数。
        if len(shape) == 1:
            return min(shape[0], remaining)
        # 二维张量先按预算规划一个矩形块。
        if len(shape) == 2:
            # 在预算约束下计算矩形块的行列数。
            row_count, col_count = self._matrix_block_shape(
                rows=shape[0],
                cols=shape[1],
                budget=remaining,
            )
            # 返回该矩形块包含的总元素数。
            return row_count * col_count
        # 非三维张量当前不参与切片长度估算。
        if len(shape) != 3:
            return 0
        # active expert 三维张量要优先按 expert 维度切预算。
        if shard.component == "bucket_active_experts" and shard.expert_ids:
            # 先规划本次会覆盖哪些 expert 以及每个 expert 的块大小。
            selected_expert_ids, row_count, col_count = self._expert_block_plan(
                shard=shard,
                shape=shape,
                budget=remaining,
            )
            # 三维专家块的贡献长度等于 expert 数乘以单 expert 块大小。
            return len(selected_expert_ids) * row_count * col_count
        # 其余三维张量默认只在最浅 depth 上取一个小块。
        depth = min(shape[0], 1)
        # 先按单个 depth 对应的预算计算行列尺寸。
        row_count, col_count = self._matrix_block_shape(
            rows=shape[1],
            cols=shape[2],
            budget=max(1, remaining // depth),
        )
        # 返回 depth 维乘上二维块大小后的总元素数。
        return depth * row_count * col_count

    # 在给定预算下为矩阵选择一个尽量保形的矩形块。
    def _matrix_block_shape(
        self,
        *,
        rows: int,
        cols: int,
        budget: int,
    ) -> tuple[int, int]:
        # 任一维度或预算非法时，退回最小 1x1 块。
        if rows <= 0 or cols <= 0 or budget <= 0:
            return 1, 1
        # 实际可用元素上限不能超过矩阵总元素数。
        max_elements = min(rows * cols, max(1, budget))
        # 默认最佳块先设为 1x1。
        best_rows = 1
        # 默认最佳列数同样设为 1。
        best_cols = 1
        # 用 score 记录当前最优候选块。
        best_score: tuple[int, float] | None = None
        # 原矩阵的长宽比会用于衡量候选块是否“保形”。
        aspect_ratio = rows / max(cols, 1)
        # 枚举所有可能的行数候选。
        for row_count in range(1, min(rows, max_elements) + 1):
            # 在当前行数下，列数取预算允许的最大值。
            col_count = min(cols, max_elements // row_count)
            # 列数不合法时跳过当前候选。
            if col_count < 1:
                continue
            # 计算当前候选块实际能覆盖的元素数。
            used = row_count * col_count
            # 计算当前候选块与原矩阵长宽比的偏差。
            ratio_distance = abs((row_count / max(col_count, 1)) - aspect_ratio)
            # 优先选覆盖元素更多的块；同元素数时选更接近原长宽比的块。
            score = (used, -ratio_distance)
            # 当前候选更优时，更新最佳块记录。
            if best_score is None or score > best_score:
                # 记录新的最佳行数。
                best_rows = row_count
                # 记录新的最佳列数。
                best_cols = col_count
                # 更新最佳分数。
                best_score = score
        # 返回最终选中的矩形块大小。
        return best_rows, best_cols

    # 为专家三维张量规划 expert 数量及每个 expert 的块大小。
    def _expert_block_plan(
        self,
        *,
        shard: ParameterShardSnapshot,
        shape: list[int],
        budget: int,
    ) -> tuple[tuple[int, ...], int, int]:
        # 没有 active expert id 时退回一个占位计划。
        if not shard.expert_ids:
            return (0,), 1, 1
        # 预算不足时也至少选 1 个 expert，但不会超过张量实际 expert 维大小。
        selected_expert_count = min(
            len(shard.expert_ids),
            max(1, min(budget, shape[0])),
        )
        # 按 shard 中既定顺序截取本次需要读取的 expert id。
        selected_expert_ids = tuple(shard.expert_ids[:selected_expert_count])
        # 把总预算平均摊给每个被选中的 expert。
        per_expert_budget = max(1, budget // max(selected_expert_count, 1))
        # 基于单 expert 预算继续规划二维子块大小。
        row_count, col_count = self._matrix_block_shape(
            rows=shape[1],
            cols=shape[2],
            budget=per_expert_budget,
        )
        # 返回“选中的 expert + 单 expert 块大小”的组合计划。
        return selected_expert_ids, row_count, col_count

    # 返回某个参数分片对应的权重来源描述。
    def source_for_shard(self, shard: ParameterShardSnapshot) -> WeightShardSource:
        # 直接把分片元信息和匹配到的张量引用封装成 WeightShardSource。
        return WeightShardSource(
            group_id=shard.group_id,
            component=shard.component,
            model_path=self.model_path,
            tensor_refs=self._tensor_refs_for_shard(shard),
        )

    # 获取某个分片在全量物化模式下的有序张量列表。
    def _full_materialization_refs(
        self,
        shard: ParameterShardSnapshot,
    ) -> tuple[WeightTensorRef, ...]:
        # 先解析当前分片的权重来源。
        source = self.source_for_shard(shard)
        # 没匹配到任何张量时直接返回空。
        if not source.matched:
            return ()
        # 用列表按语义顺序收集最终要全量物化的张量引用。
        ordered_refs: list[WeightTensorRef] = []
        # 先按层分组，再在层内按语义角色展开。
        for layer_index, ref_group in self._sampling_layer_groups(
            shard=shard,
            tensor_refs=source.tensor_refs,
        ):
            # 把当前层张量进一步按角色归类。
            refs_by_role = self._refs_by_role(ref_group)
            # 依照角色采样 schema 的顺序展开每个角色下的全部张量。
            for role, _ in self._role_sampling_schema(
                shard=shard,
                layer_index=layer_index,
                available_roles=tuple(refs_by_role),
            ):
                # 当前角色存在张量时按既定排序追加进去。
                ordered_refs.extend(refs_by_role.get(role, ()))
        # 返回全量物化模式下的有序张量列表。
        return tuple(ordered_refs)

    # 为某个分片构建完整参数缓冲区。
    def build_full_parameter_buffer(
        self,
        shard: ParameterShardSnapshot,
    ) -> ParameterBufferLoadResult | None:
        # -----------------
        # 先按语义顺序确定需要读取的全部张量引用。
        refs = self._full_materialization_refs(shard)
        if not refs:
            return None
        planned_refs: list[tuple[WeightTensorRef, tuple[int, ...], tuple[int, ...], int]] = []
        used_file_names: list[str] = []
        total_length = 0

        # -----------------
        # 第一遍只规划张量布局和总长度，避免先把全部 chunk 常驻在内存里。
        for ref in refs:
            # 先解析当前张量所在的 safetensors 文件路径。
            file_path = self._model_path / ref.file_name
            # 文件缺失时跳过当前张量。
            if not file_path.exists():
                continue
            # 打开 safetensors 文件，只读取 shape 元信息。
            with safe_open(file_path, framework="pt", device="cpu") as handle:
                # 先记录当前张量的逻辑 shape，后面要写入 source_layout。
                tensor_shape = self._logical_shape_for_ref(handle, ref.tensor_name)
                # GPTQ qweight 在全量模式下会恢复成完整逻辑 shape。
                if self._is_gptq_qweight_tensor(ref.tensor_name):
                    slice_shape = tensor_shape
                else:
                    # 普通张量根据 shard 选择规则推导最终切片形状。
                    slice_shape = self._full_slice_shape_for_ref(
                        shard=shard,
                        tensor_shape=tuple(handle.get_slice(ref.tensor_name).get_shape()),
                    )
            # 无法确定合法切片形状时跳过。
            if slice_shape is None:
                continue
            # 当前切片包含的逻辑元素数。
            flat_length = math.prod(slice_shape)
            # 空切片不计入结果。
            if flat_length <= 0:
                continue
            # 记录当前 ref 对应的逻辑布局与线性长度。
            planned_refs.append(
                (
                    ref,
                    tensor_shape,
                    slice_shape,
                    int(flat_length),
                )
            )
            # 维护最终会访问到的去重文件列表。
            # 维护去重后的使用文件列表。
            if ref.file_name not in used_file_names:
                used_file_names.append(ref.file_name)
            # 把当前张量长度累加到最终线性缓冲区长度。
            total_length += int(flat_length)
        # 规划阶段没有拿到任何有效张量时返回 None。
        if not planned_refs:
            return None

        # -----------------
        # 第二遍按规划顺序把每个张量块直接写入预分配缓冲区。
        values = torch.empty(total_length, dtype=torch.float32, device="cpu")
        source_layout: list[ParameterSourceSlice] = []
        start_offset = 0
        for ref, tensor_shape, slice_shape, flat_length in planned_refs:
            # 重新打开来源文件，读取当前张量块的真实内容。
            file_path = self._model_path / ref.file_name
            with safe_open(file_path, framework="pt", device="cpu") as handle:
                # GPTQ qweight 需要先完整解包成逻辑浮点权重。
                if self._is_gptq_qweight_tensor(ref.tensor_name):
                    gptq_chunk = self._gptq_full_chunk_for_ref(
                        handle=handle,
                        tensor_name=ref.tensor_name,
                    )
                    chunk = None if gptq_chunk is None else gptq_chunk[0]
                else:
                    # 普通张量直接按 shard 规则读取完整切片。
                    chunk = self._full_slice_chunk_for_ref(
                        shard=shard,
                        safe_slice=handle.get_slice(ref.tensor_name),
                    )
            # 当前张量块读取失败时跳过，保留后续布局不变。
            if chunk is None:
                continue
            # 将当前张量块展平成 CPU float32 连续向量。
            flat = chunk.reshape(-1).to(dtype=torch.float32, device="cpu").contiguous()
            # 实际长度与规划长度不一致时，说明前后两遍 shape 推导失配。
            if flat.numel() != flat_length:
                raise RuntimeError(
                    "full parameter buffer planning length does not match loaded chunk"
                )
            # 直接把当前块拷入最终线性缓冲区对应区间。
            values.narrow(0, start_offset, flat_length).copy_(flat)
            # 记录当前张量在线性缓冲区中的来源布局。
            source_layout.append(
                ParameterSourceSlice(
                    tensor_name=ref.tensor_name,
                    file_name=ref.file_name,
                    layer_index=self._layer_index_for_tensor(ref.tensor_name),
                    semantic_role=self._semantic_role_for_tensor(ref.tensor_name),
                    start_offset=start_offset,
                    length=flat_length,
                    tensor_shape=tensor_shape,
                    slice_shape=slice_shape,
                )
            )
            # 为下一个张量更新线性偏移。
            start_offset += flat_length
        # 若第二遍有 ref 读取失败，只保留已经成功写入的有效前缀。
        if start_offset <= 0:
            return None
        # 按实际写入长度裁到最终结果，避免尾部未写入空间泄漏出去。
        values = values.narrow(0, 0, start_offset).clone()
        # 返回拼好的完整参数缓冲区及其来源布局。
        return ParameterBufferLoadResult(
            values=values,
            used_file_names=tuple(used_file_names),
            used_tensor_count=len(source_layout),
            source_layout=tuple(source_layout),
        )

    # 为某个分片构建代表性参数缓冲区。
    def build_parameter_buffer(
        self,
        shard: ParameterShardSnapshot,
        *,
        representative_params: int,
    ) -> ParameterBufferLoadResult | None:
        # -----------------
        # 先匹配分片来源，并按层组织张量引用。
        source = self.source_for_shard(shard)
        if not source.matched:
            return None
        values: list[float] = []
        used_file_names: list[str] = []
        used_tensor_count = 0
        source_layout: list[ParameterSourceSlice] = []
        layer_groups = self._sampling_layer_groups(
            shard=shard,
            tensor_refs=source.tensor_refs,
        )

        # -----------------
        # 按层预算和角色预算逐步抽样张量块。
        for (layer_index, ref_group), group_quota in zip(
            layer_groups,
            self._group_quotas(
                group_count=len(layer_groups),
                total_budget=representative_params,
            ),
        ):
            # 当前层没有预算或没有张量时直接跳过。
            if group_quota <= 0 or not ref_group:
                continue
            # 先把当前层内张量按语义角色归类。
            refs_by_role = self._refs_by_role(ref_group)
            # 再把当前层预算按角色模板分下去。
            for role, role_quota in self._role_quotas(
                shard=shard,
                layer_index=layer_index,
                available_roles=tuple(refs_by_role),
                total_budget=group_quota,
            ):
                # 取出当前角色下的有序张量列表。
                role_refs = refs_by_role.get(role, ())
                # 该角色没有张量时继续下一个角色。
                if not role_refs:
                    continue
                # 角色预算按张量数平均切成单 ref 上限。
                per_ref_cap = max(1, math.ceil(role_quota / len(role_refs)))
                # 用 role_remaining 跟踪当前角色还有多少预算待分配。
                role_remaining = role_quota
                # 按优先级顺序遍历当前角色下的张量。
                for ref in role_refs:
                    # 全局预算已满或当前角色预算耗尽时停止。
                    if len(values) >= representative_params or role_remaining <= 0:
                        break
                    # 解析当前张量所在文件路径。
                    file_path = self._model_path / ref.file_name
                    # 文件不存在时跳过当前张量。
                    if not file_path.exists():
                        continue
                    # 当前张量最多只能消费“全局剩余 / 角色剩余 / 单 ref 上限”中的最小值。
                    allowed = min(
                        representative_params - len(values),
                        role_remaining,
                        per_ref_cap,
                    )
                    # 打开 safetensors 文件准备读取代表性小块。
                    with safe_open(file_path, framework="pt", device="cpu") as handle:
                        # 记录当前张量的逻辑 shape，后面要写入布局。
                        tensor_shape = self._logical_shape_for_ref(
                            handle,
                            ref.tensor_name,
                        )
                        # GPTQ qweight 需要走解包采样路径。
                        if self._is_gptq_qweight_tensor(ref.tensor_name):
                            # 只在 allowed 预算内读取并解包一个逻辑块。
                            gptq_chunk = self._gptq_chunk_for_ref(
                                handle=handle,
                                tensor_name=ref.tensor_name,
                                remaining=allowed,
                            )
                            # 解包失败时把 chunk 记为 None。
                            chunk = None if gptq_chunk is None else gptq_chunk[0]
                        else:
                            # 普通张量先获取 safetensors slice。
                            safe_slice = handle.get_slice(ref.tensor_name)
                            # 再按代表性抽样规则切出一个小块。
                            chunk = self._slice_chunk_for_ref(
                                shard=shard,
                                safe_slice=safe_slice,
                                remaining=allowed,
                            )
                    # 当前张量没采到任何数据时跳过。
                    if chunk is None:
                        continue
                    # 当前张量在线性缓冲区中的起始偏移就是 values 当前长度。
                    start_offset = len(values)
                    # 把张量块展平，并再次截断到 allowed 范围内。
                    flat = chunk.reshape(-1)[:allowed]
                    # 展平后为空时跳过。
                    if flat.numel() <= 0:
                        continue
                    # 把采样值转成 Python float 追加进代表性缓冲区。
                    values.extend(float(value) for value in flat.tolist())
                    # 记录这一段采样值的来源布局信息。
                    source_layout.append(
                        ParameterSourceSlice(
                            tensor_name=ref.tensor_name,
                            file_name=ref.file_name,
                            layer_index=self._layer_index_for_tensor(ref.tensor_name),
                            semantic_role=self._semantic_role_for_tensor(ref.tensor_name),
                            start_offset=start_offset,
                            length=int(flat.numel()),
                            tensor_shape=tensor_shape,
                            slice_shape=tuple(chunk.shape),
                        )
                    )
                    # 已使用张量计数加一。
                    used_tensor_count += 1
                    # 当前角色预算扣掉本次真实采到的元素数。
                    role_remaining -= int(flat.numel())
                    # 把当前文件加入去重文件列表。
                    if ref.file_name not in used_file_names:
                        used_file_names.append(ref.file_name)
        # 一个样本都没采到时返回 None。
        if not values:
            return None

        # -----------------
        # 如样本不足则循环补齐到代表性参数目标长度。
        # 用已有样本循环填充，保证结果长度稳定等于 representative_params。
        while len(values) < representative_params:
            values.append(values[len(values) % len(values)])
        # 返回最终代表性参数缓冲区与来源布局。
        return ParameterBufferLoadResult(
            values=tuple(values[:representative_params]),
            used_file_names=tuple(used_file_names),
            used_tensor_count=used_tensor_count,
            source_layout=tuple(source_layout),
        )

    # 仅规划代表性参数缓冲区的来源布局，不实际读取权重。
    def plan_parameter_buffer_sources(
        self,
        shard: ParameterShardSnapshot,
        *,
        representative_params: int,
    ) -> ParameterBufferSourcePlan:
        # -----------------
        # 先匹配分片来源，并准备布局统计容器。
        source = self.source_for_shard(shard)
        if not source.matched:
            return ParameterBufferSourcePlan(
                used_file_names=(),
                used_tensor_count=0,
                source_layout=(),
            )
        used_file_names: list[str] = []
        used_tensor_count = 0
        source_layout: list[ParameterSourceSlice] = []
        start_offset = 0
        layer_groups = self._sampling_layer_groups(
            shard=shard,
            tensor_refs=source.tensor_refs,
        )

        # -----------------
        # 按层 / 角色预算估算每个张量块的贡献长度与 slice 形状。
        for (layer_index, ref_group), group_quota in zip(
            layer_groups,
            self._group_quotas(
                group_count=len(layer_groups),
                total_budget=representative_params,
            ),
        ):
            # 当前层没有预算或没有张量时跳过。
            if group_quota <= 0 or not ref_group:
                continue
            # 把当前层张量先按语义角色分组。
            refs_by_role = self._refs_by_role(ref_group)
            # 再将当前层预算分配到各角色。
            for role, role_quota in self._role_quotas(
                shard=shard,
                layer_index=layer_index,
                available_roles=tuple(refs_by_role),
                total_budget=group_quota,
            ):
                # 取出当前角色下的候选张量。
                role_refs = refs_by_role.get(role, ())
                # 当前角色没有张量时跳过。
                if not role_refs:
                    continue
                # 先给每个 ref 划一个近似均分的预算上限。
                per_ref_cap = max(1, math.ceil(role_quota / len(role_refs)))
                # 记录当前角色仍可分配的预算。
                role_remaining = role_quota
                # 顺序遍历当前角色的张量引用。
                for ref in role_refs:
                    # 当前角色预算已经耗尽时停止。
                    if role_remaining <= 0:
                        break
                    # 解析当前张量文件路径。
                    file_path = self._model_path / ref.file_name
                    # 文件不存在时跳过当前张量。
                    if not file_path.exists():
                        continue
                    # 当前 ref 的预算上限受角色剩余和单 ref 上限共同约束。
                    allowed = min(role_remaining, per_ref_cap)
                    # 打开文件只读取 shape，不实际搬运权重。
                    with safe_open(file_path, framework="pt", device="cpu") as handle:
                        # 取当前张量的逻辑 shape，供后续长度估算与布局记录使用。
                        shape = self._logical_shape_for_ref(handle, ref.tensor_name)
                    # 根据 shape 和预算估算当前张量预计贡献多少元素。
                    contributed = self._planned_contribution_length(
                        shard=shard,
                        shape=list(shape),
                        remaining=allowed,
                    )
                    # 没有有效贡献时跳过当前张量。
                    if contributed <= 0:
                        continue
                    # 扣减当前角色剩余预算。
                    role_remaining -= contributed
                    # 已使用张量计数加一。
                    used_tensor_count += 1
                    # 维护去重后的使用文件列表。
                    if ref.file_name not in used_file_names:
                        used_file_names.append(ref.file_name)
                    # 记录当前张量理论上的来源布局。
                    source_layout.append(
                        ParameterSourceSlice(
                            tensor_name=ref.tensor_name,
                            file_name=ref.file_name,
                            layer_index=self._layer_index_for_tensor(ref.tensor_name),
                            semantic_role=self._semantic_role_for_tensor(ref.tensor_name),
                            start_offset=start_offset,
                            length=contributed,
                            tensor_shape=tuple(shape),
                            slice_shape=self._planned_slice_shape(
                                shard=shard,
                                shape=list(shape),
                                remaining=allowed,
                            ),
                        )
                    )
                    # 更新下一个切片的起始偏移。
                    start_offset += contributed
        # 返回仅规划得到的代表性缓冲区布局信息。
        return ParameterBufferSourcePlan(
            used_file_names=tuple(used_file_names),
            used_tensor_count=used_tensor_count,
            source_layout=tuple(source_layout),
        )

    # 仅规划全量参数缓冲区的来源布局，不实际读取权重。
    def plan_full_parameter_buffer_sources(
        self,
        shard: ParameterShardSnapshot,
    ) -> ParameterBufferSourcePlan:
        # -----------------
        # 先确定全量物化模式下的有序张量列表。
        refs = self._full_materialization_refs(shard)
        if not refs:
            return ParameterBufferSourcePlan(
                used_file_names=(),
                used_tensor_count=0,
                source_layout=(),
            )
        used_file_names: list[str] = []
        used_tensor_count = 0
        source_layout: list[ParameterSourceSlice] = []
        start_offset = 0

        # -----------------
        # 逐个张量推导完整 slice 形状与线性布局偏移。
        for ref in refs:
            # 先解析当前张量所在文件路径。
            file_path = self._model_path / ref.file_name
            # 文件不存在时跳过。
            if not file_path.exists():
                continue
            # 打开 safetensors 文件，仅做 shape 级别推导。
            with safe_open(file_path, framework="pt", device="cpu") as handle:
                # 先取当前张量的逻辑 shape。
                tensor_shape = self._logical_shape_for_ref(handle, ref.tensor_name)
                # GPTQ qweight 的全量布局直接采用逻辑解包后的 shape。
                if self._is_gptq_qweight_tensor(ref.tensor_name):
                    # GPTQ 全量模式下的 slice_shape 就是逻辑 shape。
                    logical_shape = tensor_shape
                    # 长度等于逻辑 shape 的元素积。
                    length = math.prod(logical_shape)
                    # 记录逻辑 slice shape。
                    slice_shape = logical_shape
                else:
                    # 普通张量先读取物理 slice shape。
                    safe_slice = handle.get_slice(ref.tensor_name)
                    # 读取 safetensors 中的原始 shape。
                    shape = tuple(safe_slice.get_shape())
                    # active expert 三维张量在全量模式下只保留合法 expert 子集。
                    if len(shape) == 3 and shard.component == "bucket_active_experts" and shard.expert_ids:
                        # 过滤掉越界的 expert id。
                        selected = tuple(
                            expert_id
                            for expert_id in shard.expert_ids
                            if 0 <= expert_id < shape[0]
                        )
                        # 一个合法 expert 都没有时跳过当前张量。
                        if not selected:
                            continue
                        # slice_shape 记录“选中 expert 数 + 原二维大小”。
                        slice_shape = (len(selected), shape[1], shape[2])
                    else:
                        # 其余情况直接使用张量原始 shape。
                        slice_shape = shape
                    # 长度等于当前 slice_shape 的元素积。
                    length = math.prod(slice_shape)
                # 无效长度不计入布局。
                if length <= 0:
                    continue
                # 记录当前张量在全量缓冲区中的来源布局。
                source_layout.append(
                    ParameterSourceSlice(
                        tensor_name=ref.tensor_name,
                        file_name=ref.file_name,
                        layer_index=self._layer_index_for_tensor(ref.tensor_name),
                        semantic_role=self._semantic_role_for_tensor(ref.tensor_name),
                        start_offset=start_offset,
                        length=length,
                        tensor_shape=tensor_shape,
                        slice_shape=tuple(slice_shape),
                    )
                )
                # 为下一个张量推进线性偏移。
                start_offset += length
                # 已使用张量计数加一。
                used_tensor_count += 1
                # 维护去重后的使用文件列表。
                if ref.file_name not in used_file_names:
                    used_file_names.append(ref.file_name)
        # 返回仅规划得到的全量缓冲区布局信息。
        return ParameterBufferSourcePlan(
            used_file_names=tuple(used_file_names),
            used_tensor_count=used_tensor_count,
            source_layout=tuple(source_layout),
        )

    # 在仅做规划时推导某个切片的理论 shape。
    def _planned_slice_shape(
        self,
        *,
        shard: ParameterShardSnapshot,
        shape: list[int],
        remaining: int,
    ) -> tuple[int, ...]:
        # 一维张量的理论 slice 形状就是预算内的一段前缀。
        if len(shape) == 1:
            return (min(shape[0], remaining),)
        # 二维张量按预算规划一个矩形块。
        if len(shape) == 2:
            # 计算当前预算下最合适的行列数。
            row_count, col_count = self._matrix_block_shape(
                rows=shape[0],
                cols=shape[1],
                budget=remaining,
            )
            # 返回对应的二维 slice 形状。
            return (row_count, col_count)
        # 非三维张量当前没有统一的规划形状。
        if len(shape) != 3:
            return ()
        # active expert 三维张量需要先按 expert 维规划。
        if shard.component == "bucket_active_experts" and shard.expert_ids:
            # 先得到本次理论会选择的 expert 数与二维块大小。
            selected_expert_ids, row_count, col_count = self._expert_block_plan(
                shard=shard,
                shape=shape,
                budget=remaining,
            )
            # 返回“expert 数 + 单 expert 二维块”的理论 shape。
            return (len(selected_expert_ids), row_count, col_count)
        # 其余三维张量默认只取 depth=1 的浅层块。
        depth = min(shape[0], 1)
        # 再按 depth 分摊后的预算确定二维块大小。
        row_count, col_count = self._matrix_block_shape(
            rows=shape[1],
            cols=shape[2],
            budget=max(1, remaining // depth),
        )
        # 返回最终三维 slice 形状。
        return (depth, row_count, col_count)
