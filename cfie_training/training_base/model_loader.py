"""Qwen3.5-122B 真实模型导入：解析 packed checkpoint 中的 expert 权重并构建训练基础 stores。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import torch

from cfie_training.training_base.adam_state_store import CpuAdamFp8StateStore
from cfie_training.training_base.fp32_shard_store import FP32ShardStore
from cfie_training.training_base.gptq_cache_store import GptqCacheStore
from cfie_training.training_base.manifest_builder import (
    ManifestShardConfig,
    TrainingBaseManifest,
    TrainingBaseManifestBuilder,
    TrainingParamManifestSpec,
)
from cfie_training.training_base.progress_state import ProgressStateWriter

# Qwen3.5-122B 的默认结构配置，用于训练适配器恢复 expert 权重形状。
Qwen35_122B_CONFIG = {
    "num_layers": 48,
    "num_experts": 256,
    "hidden_size": 3072,
    "intermediate_size": 1024,
    "num_experts_per_tok": 8,
    "num_attention_heads": 32,
    "num_key_value_heads": 2,
    "vocab_size": 248320,
    "dtype": "bfloat16",
}


def _require_non_empty_string(name: str, value: str) -> None:
    """校验字符串参数非空，避免路径名或字段名在后续解析中失效。"""

    # 字符串去除空白后为空时，说明调用方传入了不可用的配置值。
    if not value.strip():
        # 直接抛出参数错误，避免后续用空字符串构造 key 或路径导致隐蔽失败。
        raise ValueError(f"{name} must be a non-empty string")


@dataclass(slots=True)
class Qwen35RealImporter:
    """从 Qwen3.5 packed checkpoint 中导入 MoE expert 权重到训练基础 stores。"""

    # checkpoint_dir 指向 safetensors checkpoint 目录，导入器会从中扫描 model*.safetensors。
    checkpoint_dir: str | Path

    # num_layers 表示待解析的 MoE 层数 L，默认匹配 Qwen3.5-122B。
    num_layers: int = 48

    # num_experts 表示每层 expert 数量 E，默认匹配 Qwen3.5-122B。
    num_experts: int = 256

    # hidden_size 表示 hidden 维度 H，用于校验和恢复 expert 权重二维形状。
    hidden_size: int = 3072

    # intermediate_size 表示 expert 中间维度 I，用于拆分 gate/up 和恢复 down 权重形状。
    intermediate_size: int = 1024

    # _shard_index 预留 checkpoint 分片索引路径，当前实现通过 glob 直接扫描分片文件。
    _shard_index: Path | None = None

    def __post_init__(self) -> None:
        """把 checkpoint 路径规范化为 Path，统一后续文件扫描逻辑。"""

        # 将字符串路径转换为 Path，避免后续 glob 和路径拼接需要重复适配 str/Path。
        self.checkpoint_dir = Path(self.checkpoint_dir)

    def iter_expert_weights(
        self,
        *,
        layers: tuple[int, ...] | None = None,
        experts: tuple[int, ...] | None = None,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """按参数 ID 逐个产出 expert 权重，输出张量均为展平后的 float32 tensor。"""

        # ------------------------------- 准备 checkpoint 读取范围 -------------------------------
        # 延迟导入 safetensors，避免只构造导入器或 manifest 时强制依赖该库。
        from safetensors import safe_open

        # 将指定层列表转换为集合；未指定时默认覆盖 [0, L) 的全部层。
        target_layers = set(layers) if layers else set(range(self.num_layers))
        # 将指定专家列表转换为集合；未指定时默认覆盖 [0, E) 的全部专家。
        target_experts = set(experts) if experts else set(range(self.num_experts))

        # 扫描 checkpoint 目录下所有 model*.safetensors 分片，并排序保证导入顺序稳定。
        shard_files = sorted(
            p for p in self.checkpoint_dir.glob("model*.safetensors")
            if p.suffix == ".safetensors"
        )

        # ------------------------------- 遍历分片并筛选 expert 权重 -------------------------------
        # 逐个打开 safetensors 分片，避免一次性加载全部 checkpoint 到内存。
        for shard_path in shard_files:
            # 以 PyTorch framework 读取当前分片，使 get_tensor 返回 torch.Tensor。
            with safe_open(str(shard_path), framework="pt") as f:
                # 遍历当前分片内所有 tensor key，逐个识别是否为 packed expert 权重。
                for key in f.keys():
                    # 解析 packed expert key，提取 layer_id 和权重类型；非目标 key 返回 None。
                    parsed = self._parse_packed_expert_key(key)
                    # 跳过非 expert 权重，避免 attention、norm、embedding 等参数进入 expert store。
                    if parsed is None:
                        # 继续处理当前分片的下一个 key。
                        continue
                    # 拆出层编号和权重类型，用于后续层过滤与 gate/up/down 分支处理。
                    layer_id, weight_type = parsed
                    # 跳过不在本次导入范围内的层，减少张量读取和 CPU 转换开销。
                    if layer_id not in target_layers:
                        # 继续处理当前分片的下一个 key。
                        continue

                    # 读取 packed expert tensor；gate_up 通常为 [E, 2I, H]，down 通常为 [E, H, I]。
                    tensor = f.get_tensor(key)
                    # expert_dim 表示当前 packed tensor 的专家维度 E_file，即 tensor.shape[0]。
                    expert_dim = tensor.shape[0]

                    # ------------------------------- 拆分 packed expert 维度 -------------------------------
                    # 遍历 packed tensor 的 expert 维度，把 [E_file, ...] 拆为单 expert 参数。
                    for expert_id in range(expert_dim):
                        # 跳过不在本次导入范围内的专家，避免无关专家进入 FP32 store。
                        if expert_id not in target_experts:
                            # 继续处理当前 packed tensor 中的下一个 expert。
                            continue
                        # 取出单个 expert 权重并复制为 float32；gate_up: [2I, H]，down: [H, I]。
                        expert_tensor = tensor[expert_id].clone().to(torch.float32)

                        # gate_up_proj 分支负责把 packed gate/up 权重拆分后重新融合为 w13_weight。
                        if weight_type == "gate_up_proj":
                            # k 表示 gate+up 的输出通道数 2I，保留该标量用于表达 packed 第一维含义。
                            k = self.intermediate_size * 2
                            # gate: [I, H]，取 expert_tensor 前 I 行作为 gate projection 权重。
                            gate = expert_tensor[:self.intermediate_size, :]
                            # up: [I, H]，取 expert_tensor 后 I 行作为 up projection 权重。
                            up = expert_tensor[self.intermediate_size:, :]
                            # fused: [2 * I * H]，先分别展平 gate/up 再拼接，保持 w13 的连续存储布局。
                            fused = torch.cat([gate.reshape(-1), up.reshape(-1)])
                            # 产出当前 expert 的 w13 参数 ID 和展平权重，供 FP32 store 按 param_id 写入。
                            yield (
                                f"layers.{layer_id}.experts.{expert_id}.w13_weight",
                                fused,
                            )
                        # down_proj 分支负责把单 expert down 权重展平为 w2_weight。
                        elif weight_type == "down_proj":
                            # 产出 w2: [H * I]，由 down 权重 [H, I] 展平得到，用于统一分片写入。
                            yield (
                                f"layers.{layer_id}.experts.{expert_id}.w2_weight",
                                expert_tensor.reshape(-1),
                            )

    def build_manifest_specs(
        self,
        *,
        layers: tuple[int, ...] | None = None,
        experts: tuple[int, ...] | None = None,
    ) -> tuple[TrainingParamManifestSpec, ...]:
        """根据层和专家范围构造 FP32/Adam/GPTQ store 所需的参数规格。"""

        # ------------------------------- 确定 manifest 覆盖范围 -------------------------------
        # 未指定 layers 时覆盖 [0, L) 的全部层；指定时只为目标层生成参数规格。
        target_layers = layers or tuple(range(self.num_layers))

        # 未指定 experts 时覆盖 [0, E) 的全部专家；指定时只为目标专家生成参数规格。
        target_experts = experts or tuple(range(self.num_experts))

        # ------------------------------- 构造每个 expert 的参数规格 -------------------------------
        # 初始化规格列表，后续每个 expert 会追加 w13 和 w2 两条参数规格。
        specs: list[TrainingParamManifestSpec] = []

        # 遍历目标层，保证 manifest 的参数 ID 与实际导入的层范围一致。
        for layer_id in target_layers:

            # 遍历目标专家，保证每个 expert 都有 w13 和 w2 的存储规格。
            for expert_id in target_experts:
                # w13 元素数为 2 * I * H，对应 gate/up 两个 [I, H] 权重展平拼接。
                w13_elements = 2 * self.intermediate_size * self.hidden_size

                # w2 元素数为 H * I，对应 down 权重 [H, I] 展平后的长度。
                w2_elements = self.hidden_size * self.intermediate_size

                # 添加 w13_weight 规格，使 manifest 为该参数分配 FP32/Adam/GPTQ 存储位置。
                specs.append(
                    TrainingParamManifestSpec(
                        param_id=f"layers.{layer_id}.experts.{expert_id}.w13_weight",
                        num_elements=w13_elements,
                        trainable=True,
                    )
                )

                # 添加 w2_weight 规格，使 manifest 为 down projection 参数分配存储位置。
                specs.append(
                    TrainingParamManifestSpec(
                        param_id=f"layers.{layer_id}.experts.{expert_id}.w2_weight",
                        num_elements=w2_elements,
                        trainable=True,
                    )
                )
        # 将规格列表冻结为 tuple，避免 manifest 构建后参数集合被外部修改。
        return tuple(specs)

    def import_to_stores(
        self,
        root: str | Path,
        *,
        manifest_config: ManifestShardConfig | None = None,
        layers: tuple[int, ...] | None = None,
        experts: tuple[int, ...] | None = None,
    ) -> tuple[
        FP32ShardStore,
        CpuAdamFp8StateStore,
        GptqCacheStore,
        TrainingBaseManifest,
        ProgressStateWriter,
    ]:
        """把目标 expert 权重导入 FP32 store，并初始化训练进度状态。"""

        # ------------------------------- 准备输出目录与 manifest -------------------------------
        # 将输出根目录规范化为 Path，统一后续 store 和 state 目录创建逻辑。
        root_path = Path(root)

        # 未显式传入分片配置时使用默认配置，保证 smoke import 可以最小参数运行。
        cfg = manifest_config or ManifestShardConfig()

        # 根据目标层和专家范围生成参数规格，规格中的 num_elements 决定分片布局。
        specs = self.build_manifest_specs(layers=layers, experts=experts)

        # 基于分片配置构建训练基础 manifest，确定每个 param_id 的 shard/offset/size。
        manifest = TrainingBaseManifestBuilder(cfg).build(specs)

        # 根据 manifest 创建 FP32、Adam、GPTQ 三类 store，generation=0 表示初始版本。
        fp32_store, adam_store, gptq_store = manifest.create_stores(
            root_path, generation=0,
        )

        # ------------------------------- 读取 checkpoint 并写入 FP32 store -------------------------------
        # 初始化待写入参数字典，key 为 param_id，value 为展平 float32 tensor。
        updates: dict[str, torch.Tensor] = {}

        # 遍历 checkpoint 中的目标 expert 权重，逐个收集到本次 flush 的更新集合。
        for param_id, tensor in self.iter_expert_weights(
            layers=layers, experts=experts,
        ):
            # 将当前参数加入写入集合；tensor 为 [2 * I * H] 或 [H * I] 的 float32 展平向量。
            updates[param_id] = tensor

        # 把收集到的 FP32 参数批量写入 touched shards，并记录为 generation=0 的初始主参数。
        fp32_store.flush_touched(updates, generation=0)

        # ------------------------------- 初始化训练进度状态 -------------------------------
        # 在 root/state 下创建进度写入器，用于记录当前训练恢复点和各 store generation。
        progress_writer = ProgressStateWriter.in_dir(root_path / "state")

        # 写入初始化后的进度状态，表示所有导入参数在 step 0 已完成 flush。
        progress_writer.write_after_flush(
            global_step=0,
            epoch=0,
            dataset_cursor="",
            round_id=0,
            hot_set=tuple(spec.param_id for spec in specs),
            fp32_master_generation=0,
            optimizer_generation=0,
            gptq_cache_generation=0,
        )

        # 返回三类 store、manifest 和进度写入器，供训练循环继续挂载和执行。
        return fp32_store, adam_store, gptq_store, manifest, progress_writer

    @staticmethod
    def _parse_packed_expert_key(name: str) -> tuple[int, str] | None:
        """解析 packed expert checkpoint key，返回层编号和权重类型。"""

        # ------------------------------- 校验 checkpoint key 前缀 -------------------------------
        # 定义 Qwen3.5 checkpoint 中 language_model 层级 expert 权重的固定前缀。
        prefix = "model.language_model.layers."
        # 非模型层权重直接跳过，避免误解析 embedding、lm_head 或其他模块。
        if not name.startswith(prefix):
            # 返回 None 表示该 key 不属于当前导入器支持的 packed expert 权重。
            return None

        # ------------------------------- 拆分层号与模块路径 -------------------------------
        # 去掉固定前缀，剩余路径应以 layer_id.mlp.experts.xxx 形式开始。
        rest = name[len(prefix):]
        # 最多拆分 4 段，保留 parts[3] 作为后续完整权重类型路径。
        parts = rest.split(".", 3)
        # 路径段不足时无法安全提取 layer/mlp/experts/weight_type。
        if len(parts) < 4:
            # 返回 None 表示该 key 结构不符合 expert 权重命名约定。
            return None

        # ------------------------------- 解析层编号并确认 expert 路径 -------------------------------
        # 尝试把第一段解析为 layer_id，用于后续层过滤和参数 ID 生成。
        try:
            # layer_id 表示 checkpoint 中的模型层编号。
            layer_id = int(parts[0])
        # 第一段不是整数时说明该 key 不是标准层路径。
        except ValueError:
            # 返回 None 避免异常中断整个 checkpoint 扫描流程。
            return None

        # 只接受 mlp.experts 路径，避免把非 MoE expert 参数误导入专家 store。
        if parts[1] != "mlp" or parts[2] != "experts":
            # 返回 None 表示该 key 虽在 layers 下，但不是 expert 权重路径。
            return None

        # ------------------------------- 识别支持的 packed 权重类型 -------------------------------
        # weight_type 保留 experts 后面的剩余路径，用于区分 gate_up_proj 和 down_proj。
        weight_type = parts[3]
        # 当前导入器只处理 packed gate/up 权重和 down projection 权重。
        if weight_type in ("gate_up_proj", "down_proj"):
            # 返回层编号和权重类型，交由上层读取张量并拆分专家维度。
            return layer_id, weight_type

        # 其他 expert 相关 key 暂不导入，返回 None 保持导入范围可控。
        return None


@dataclass(slots=True)
class Qwen35TrainingAdapter:
    """从训练基础 store 读取 Qwen3.5 expert 权重，并恢复为模型可用的二维 tensor。"""

    # FP32 主参数 store，保存从 checkpoint 导入或训练更新后的 expert 权重。
    fp32_store: FP32ShardStore
    # Adam 状态 store，保留在适配器中供后续训练状态读取或扩展使用。
    adam_store: CpuAdamFp8StateStore
    # 模型结构配置，默认使用 Qwen3.5-122B 维度来恢复权重形状。
    import_config: dict = field(default_factory=lambda: dict(Qwen35_122B_CONFIG))

    def build_layer_weights(
        self,
        layer_id: int,
        *,
        expert_ids: tuple[int, ...],
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float16,
    ) -> dict[str, torch.Tensor]:
        """读取指定层的 expert 权重，并恢复为 device/dtype 上的二维 tensor。"""

        # ------------------------------- 准备恢复形状所需元数据 -------------------------------
        # 初始化输出字典，key 为 param_id，value 为恢复后的二维权重 tensor。
        weights: dict[str, torch.Tensor] = {}
        # hidden 表示 H 维度，用于把 w13/w2 的一维存储恢复为二维权重。
        hidden = self.import_config["hidden_size"]
        # intermediate 表示 I 维度，用于恢复 gate/up/down 的中间通道规模。
        intermediate = self.import_config["intermediate_size"]

        # ------------------------------- 按 expert 读取并恢复权重 -------------------------------
        # 遍历目标 expert ID，逐个尝试从 FP32 store 中读取 w13 和 w2。
        for expert_id in expert_ids:
            # 构造 w13 参数 ID，用于定位当前层当前 expert 的 gate/up 融合权重。
            w13_key = f"layers.{layer_id}.experts.{expert_id}.w13_weight"
            # 构造 w2 参数 ID，用于定位当前层当前 expert 的 down projection 权重。
            w2_key = f"layers.{layer_id}.experts.{expert_id}.w2_weight"

            # 仅当 w13 存在于 manifest records 中时读取，允许部分专家或层缺失。
            if w13_key in self.fp32_store.records:
                # 从 FP32 store 读取 w13 原始字节，内容对应 [2 * I * H] 的 float32 展平向量。
                w13_data = self.fp32_store.read_param(w13_key)
                # 将字节缓冲区解释为 float32 tensor，得到一维张量 [2 * I * H]。
                w13 = torch.frombuffer(bytearray(w13_data), dtype=torch.float32)
                # w13: [2 * I * H] -> [2I, H]，再搬到目标 device 并转换为目标 dtype。
                w13 = w13.reshape(2 * intermediate, hidden).to(device=device, dtype=dtype)
                # 把恢复后的 w13 权重写入输出字典，供模型层按 param_id 获取。
                weights[w13_key] = w13

            # 仅当 w2 存在于 manifest records 中时读取，允许只恢复可用专家权重。
            if w2_key in self.fp32_store.records:
                # 从 FP32 store 读取 w2 原始字节，内容对应 [H * I] 的 float32 展平向量。
                w2_data = self.fp32_store.read_param(w2_key)
                # 将字节缓冲区解释为 float32 tensor，得到一维张量 [H * I]。
                w2 = torch.frombuffer(bytearray(w2_data), dtype=torch.float32)
                # w2: [H * I] -> [H, I]，再搬到目标 device 并转换为目标 dtype。
                w2 = w2.reshape(hidden, intermediate).to(device=device, dtype=dtype)
                # 把恢复后的 w2 权重写入输出字典，供模型层按 param_id 获取。
                weights[w2_key] = w2

        # 返回当前层已成功恢复的 expert 权重集合。
        return weights

    def estimate_memory_bytes(self, param_ids: tuple[str, ...]) -> int:
        """估算指定参数 ID 在 FP32 store 中占用的总字节数。"""

        # ------------------------------- 汇总参数记录中的字节数 -------------------------------
        # 初始化累计字节数，用于统计命中的 FP32 参数大小。
        total = 0
        # 遍历调用方指定的参数 ID 列表，逐个查询 store 中的记录。
        for pid in param_ids:
            # 从 FP32 store records 中获取参数记录；不存在时返回 None。
            record = self.fp32_store.records.get(pid)
            # 只有参数存在时才累加字节数，避免缺失参数影响估算流程。
            if record is not None:
                # 累加当前参数的存储字节数，通常等于 num_elements * sizeof(float32)。
                total += record.num_bytes
        # 返回所有命中参数的 FP32 存储总字节数。
        return total


def build_quick_smoke_import(
    checkpoint_dir,
    output_dir,
    *,
    max_layers=2,
    max_experts=4,
    shard_gib=1.0,
):
    """快速导入少量层和专家，用于 smoke test 或本地最小闭环验证。"""

    # ------------------------------- 构造 smoke test 导入配置 -------------------------------
    # 将 GiB 分片大小转换为字节数，供 ManifestShardConfig 控制单个 shard 文件大小。
    gb = int(shard_gib * (1 << 30))
    # 构造只覆盖少量层和专家的导入器，降低 smoke test 的 checkpoint 读取与写入成本。
    importer = Qwen35RealImporter(
        checkpoint_dir=checkpoint_dir,
        num_layers=max_layers,
        num_experts=max_experts,
    )
    # 执行最小范围导入，只导入 [0, max_layers) 层和 [0, max_experts) 专家。
    return importer.import_to_stores(
        output_dir,
        manifest_config=ManifestShardConfig(
            fp32_shard_bytes=gb,
            adam_shard_bytes=gb,
            gptq_shard_bytes=gb,
        ),
        layers=tuple(range(max_layers)),
        experts=tuple(range(max_experts)),
    )
