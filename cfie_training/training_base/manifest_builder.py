"""Manifest 构建器：为 FP32、Adam、GPTQ 三种训练存储生成参数分片索引表。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from cfie_training.training_base.adam_state_store import (
    AdamStateShardRecord,
    CpuAdamFp8StateStore,
    state_key,
)
from cfie_training.training_base.adam_update import adam_state_num_bytes
from cfie_training.training_base.fp32_shard_store import (
    FP32_BYTES,
    FP32ShardStore,
    ParamShardRecord,
)
from cfie_training.training_base.gptq_cache_store import (
    GptqCacheRecord,
    GptqCacheStore,
)
from cfie_training.training_base.gptq_requant import (
    DEFAULT_GPTQ_GROUP_SIZE,
    gptq_bundle_num_bytes,
    gptq_layout_hash,
)


def _require_non_empty_string(name: str, value: str) -> None:
    """校验字符串配置非空，避免后续生成空路径、空 key 或空分片名前缀。"""

    # 去掉空白后为空，说明该字符串无法作为稳定的配置字段使用。
    if not value.strip():
        # 直接抛出参数错误，阻止无效配置进入 manifest 构建流程。
        raise ValueError(f"{name} must be a non-empty string")


def _require_non_negative_int(name: str, value: int) -> None:
    """校验整数配置不为负，适用于 layer_start、expert_id 等从 0 开始的编号。"""

    # 小于 0 的编号或大小会破坏层范围、专家范围或偏移语义。
    if value < 0:
        # 抛出明确错误，使调用方知道该字段必须使用非负整数。
        raise ValueError(f"{name} must be >= 0")


def _require_positive_int(name: str, value: int) -> None:
    """校验整数配置为正数，适用于元素数、分片大小和分组大小。"""

    # 小于 1 的容量或元素数无法生成有效的存储布局。
    if value < 1:
        # 抛出明确错误，避免后续分片分配出现 0 长度或负长度记录。
        raise ValueError(f"{name} must be >= 1")


@dataclass(frozen=True, slots=True)
class TrainingParamManifestSpec:
    """单个训练参数的规格输入，用于描述参数大小、训练属性和 GPTQ cache 关系。"""

    # param_id 是训练系统内部使用的唯一参数 ID，也是 FP32 record 的主键。
    param_id: str
    # num_elements 表示 FP32 主参数元素数量，FP32 字节数等于 num_elements * FP32_BYTES。
    num_elements: int
    # trainable 表示该参数是否需要 Adam 状态；不可训练参数不会生成 Adam record。
    trainable: bool = True
    # gptq_bundle_id 表示该参数对应的 GPTQ cache bundle；为 None 时不生成 GPTQ record。
    gptq_bundle_id: str | None = None
    # gptq_num_elements 可覆盖 GPTQ cache 的元素规模；为 None 时默认复用 num_elements。
    gptq_num_elements: int | None = None
    # gptq_num_bytes 可直接指定 GPTQ cache 字节数；为 None 时按 group size 估算。
    gptq_num_bytes: int | None = None
    # quant_layout_hash 标识量化布局；为空时 builder 会生成默认 GPTQ 布局哈希。
    quant_layout_hash: str = ""

    def __post_init__(self) -> None:
        """校验参数规格字段，确保后续分片规划有稳定的大小和 key。"""

        # param_id 必须非空，否则无法作为 records 字典 key 和参数存储标识。
        _require_non_empty_string("param_id", self.param_id)
        # num_elements 必须为正，确保 FP32 主参数至少占用一个元素。
        _require_positive_int("num_elements", self.num_elements)
        # 只有配置 GPTQ bundle 时才校验 bundle_id，允许普通 FP32/Adam 参数没有 GPTQ cache。
        if self.gptq_bundle_id is not None:
            # GPTQ bundle ID 非空才能作为 gptq_records 的主键。
            _require_non_empty_string("gptq_bundle_id", self.gptq_bundle_id)
        # 只有显式覆盖 GPTQ 元素数时才校验，默认 None 会回退到 num_elements。
        if self.gptq_num_elements is not None:
            # GPTQ 元素数必须为正，避免后续估算出无效 cache 大小。
            _require_positive_int("gptq_num_elements", self.gptq_num_elements)
        # 只有显式覆盖 GPTQ 字节数时才校验，默认 None 会按量化布局估算。
        if self.gptq_num_bytes is not None:
            # GPTQ 字节数必须为正，避免生成 0 字节 cache record。
            _require_positive_int("gptq_num_bytes", self.gptq_num_bytes)

    @property
    def resolved_gptq_num_elements(self) -> int:
        """返回 GPTQ cache 使用的元素数量，未覆盖时回退到 FP32 主参数元素数。"""

        # gptq_num_elements 为空时使用 num_elements，保证 GPTQ cache 默认覆盖同一参数规模。
        return (
            self.num_elements
            if self.gptq_num_elements is None
            else self.gptq_num_elements
        )

    @property
    def resolved_gptq_num_bytes(self) -> int | None:
        """返回显式指定的 GPTQ cache 字节数；为 None 时由 builder 估算。"""

        # 直接暴露 gptq_num_bytes，让 builder 判断是否需要调用 gptq_bundle_num_bytes。
        return self.gptq_num_bytes


@dataclass(frozen=True, slots=True)
class ManifestShardConfig:
    """训练基础存储的分片配置，控制 FP32、Adam、GPTQ 三类文件布局。"""

    # fp32_shard_bytes 表示单个 FP32 分片最大字节数。
    fp32_shard_bytes: int = 1 << 30
    # adam_shard_bytes 表示单个 Adam 状态分片最大字节数。
    adam_shard_bytes: int = 1 << 30
    # gptq_shard_bytes 表示单个 GPTQ cache 分片最大字节数。
    gptq_shard_bytes: int = 1 << 30
    # adam_block_size 表示 Adam 状态按多少元素一组计算 FP8 状态开销。
    adam_block_size: int = 128
    # gptq_group_size 表示 GPTQ 量化分组大小，用于估算 qweight/scales/qzeros 等 bundle 字节数。
    gptq_group_size: int = DEFAULT_GPTQ_GROUP_SIZE
    # fp32_shard_prefix 表示 FP32 分片文件名前缀，例如 fp32_0000.bin。
    fp32_shard_prefix: str = "fp32"
    # adam_shard_prefix 表示 Adam 分片文件名前缀，例如 adam_0000.bin。
    adam_shard_prefix: str = "adam"
    # gptq_shard_prefix 表示 GPTQ 分片文件名前缀，例如 gptq_0000.bin。
    gptq_shard_prefix: str = "gptq"
    # adam_components 表示每个可训练参数需要生成的 Adam 状态分量，默认是一阶矩 m 和二阶矩 v。
    adam_components: tuple[str, ...] = ("m", "v")

    def __post_init__(self) -> None:
        """校验分片配置，确保每类 store 都能生成合法 shard 和 record。"""

        # FP32 分片大小必须为正，才能承载至少一个参数片段。
        _require_positive_int("fp32_shard_bytes", self.fp32_shard_bytes)
        # Adam 分片大小必须为正，才能承载优化器状态记录。
        _require_positive_int("adam_shard_bytes", self.adam_shard_bytes)
        # GPTQ 分片大小必须为正，才能承载量化 cache bundle。
        _require_positive_int("gptq_shard_bytes", self.gptq_shard_bytes)
        # FP32 分片大小必须按 4 字节对齐，保证 offset_bytes 可转换为 offset_elements。
        if self.fp32_shard_bytes % FP32_BYTES:
            # 抛出对齐错误，避免后续 FP32 record 出现非整数元素偏移。
            raise ValueError("fp32_shard_bytes must be divisible by 4")
        # Adam block size 必须为正，保证 Adam FP8 状态字节估算有合法分母。
        _require_positive_int("adam_block_size", self.adam_block_size)
        # GPTQ group size 必须为正，保证量化 bundle 字节估算和布局哈希有效。
        _require_positive_int("gptq_group_size", self.gptq_group_size)
        # FP32 分片前缀必须非空，确保生成的分片文件名可读且稳定。
        _require_non_empty_string("fp32_shard_prefix", self.fp32_shard_prefix)
        # Adam 分片前缀必须非空，确保优化器状态文件名可读且稳定。
        _require_non_empty_string("adam_shard_prefix", self.adam_shard_prefix)
        # GPTQ 分片前缀必须非空，确保量化 cache 文件名可读且稳定。
        _require_non_empty_string("gptq_shard_prefix", self.gptq_shard_prefix)
        # Adam 状态分量不能为空，否则可训练参数不会生成任何优化器状态。
        if not self.adam_components:
            # 抛出配置错误，提示调用方至少保留一个 Adam component。
            raise ValueError("adam_components must not be empty")
        # 遍历每个 Adam 分量名，保证 state_key 能生成有效 key。
        for component in self.adam_components:
            # 单个 Adam 分量名必须非空，例如 m、v 或其他扩展状态名。
            _require_non_empty_string("adam component", component)


@dataclass(frozen=True, slots=True)
class TrainingBaseManifest:
    """训练基础存储总清单，记录 FP32、Adam、GPTQ 参数到分片位置的映射。"""

    # fp32_records 记录 param_id 到 FP32 shard/offset/num_elements 的映射。
    fp32_records: dict[str, ParamShardRecord]
    # adam_records 记录 state_key(param_id, component) 到 Adam 状态 shard/offset/bytes 的映射。
    adam_records: dict[str, AdamStateShardRecord]
    # gptq_records 记录 bundle_id 到 GPTQ cache shard/offset/bytes/layout 的映射。
    gptq_records: dict[str, GptqCacheRecord]
    # param_to_gptq_bundle 记录 param_id 到 GPTQ bundle_id 的反查关系。
    param_to_gptq_bundle: dict[str, str]

    def create_stores(
        self,
        root: str | Path,
        *,
        generation: int = 0,
    ) -> tuple[FP32ShardStore, CpuAdamFp8StateStore, GptqCacheStore]:
        """基于 manifest 记录创建三类 store 对象，供训练流程实际读写数据。"""

        # ------------------------------- 准备 store 根目录 -------------------------------
        # 将根目录规范化为 Path，后续按 fp32/adam/gptq 子目录创建 store。
        root_path = Path(root)
        # 创建并返回三类 store；manifest record 决定每个参数在各自子目录中的分片位置。
        return (
            FP32ShardStore.create(
                root_path / "fp32",
                self.fp32_records,
                generation=generation,
            ),
            CpuAdamFp8StateStore.create(
                root_path / "adam",
                self.adam_records,
                generation=generation,
            ),
            GptqCacheStore.create(
                root_path / "gptq",
                self.gptq_records,
                generation=generation,
            ),
        )

    @property
    def total_fp32_bytes(self) -> int:
        """统计 manifest 中所有 FP32 主参数记录的总字节数。"""

        # 汇总每条 FP32 record 的 num_bytes，用于容量报告和资源预估。
        return sum(record.num_bytes for record in self.fp32_records.values())

    @property
    def total_adam_bytes(self) -> int:
        """统计 manifest 中所有 Adam 状态记录的总字节数。"""

        # 汇总每条 Adam record 的 num_bytes，包含所有可训练参数的所有 Adam component。
        return sum(record.num_bytes for record in self.adam_records.values())

    @property
    def total_gptq_bytes(self) -> int:
        """统计 manifest 中所有 GPTQ cache 记录的总字节数。"""

        # 汇总每条 GPTQ record 的 num_bytes，用于估算量化 cache 的落盘容量。
        return sum(record.num_bytes for record in self.gptq_records.values())


@dataclass(slots=True)
class TrainingBaseManifestBuilder:
    """把参数规格列表转换为 FP32、Adam、GPTQ 三套分片 record。"""

    # config 控制三类 store 的分片大小、命名前缀、Adam block size 和 GPTQ group size。
    config: ManifestShardConfig = field(default_factory=ManifestShardConfig)

    def build(
        self,
        specs: Iterable[TrainingParamManifestSpec],
    ) -> TrainingBaseManifest:
        """根据参数规格构建完整训练基础 manifest。"""

        # ------------------------------- 固定输入顺序并校验唯一性 -------------------------------
        # 将 Iterable 固化为 tuple，确保三类 record 构建使用同一顺序且可重复遍历。
        ordered_specs = tuple(specs)

        # 校验 param_id 不重复，避免后续 records 字典覆盖已有参数布局。
        self._validate_unique_specs(ordered_specs)

        # ------------------------------- 构建三类存储索引表 -------------------------------
        # 为每个参数生成 FP32 主参数 record，记录 shard_name、offset_elements 和 num_elements。
        fp32_records = self._build_fp32_records(ordered_specs)

        # 为可训练参数生成 Adam 状态 record，按 component 拆成 m/v 等独立状态项。
        adam_records = self._build_adam_records(ordered_specs)

        # 为带 GPTQ bundle 的参数生成 GPTQ cache record，并建立 param_id 到 bundle_id 的映射。
        gptq_records, param_to_gptq_bundle = self._build_gptq_records(ordered_specs)

        # 汇总三类索引表为 TrainingBaseManifest，作为后续 create_stores 的唯一布局依据。
        return TrainingBaseManifest(
            fp32_records=fp32_records,
            adam_records=adam_records,
            gptq_records=gptq_records,
            param_to_gptq_bundle=param_to_gptq_bundle,
        )

    def _build_fp32_records(
        self,
        specs: tuple[TrainingParamManifestSpec, ...],
    ) -> dict[str, ParamShardRecord]:
        """为每个参数生成 FP32 主参数分片记录。"""

        # ------------------------------- 初始化 FP32 分片游标 -------------------------------
        # 创建 FP32 分片游标，用于按 fp32_shard_bytes 顺序分配 shard 和 offset。
        cursor = _ShardCursor(
            shard_bytes=self.config.fp32_shard_bytes,
            shard_prefix=self.config.fp32_shard_prefix,
            extension=".bin",
        )

        # 初始化 FP32 records 字典，key 为 param_id。
        records: dict[str, ParamShardRecord] = {}

        # ------------------------------- 顺序分配 FP32 参数布局 -------------------------------
        # 按 specs 顺序分配参数，保证 manifest 生成结果稳定可复现。
        for spec in specs:
            # 为当前参数申请 num_elements * 4 字节空间，得到当前 shard 内字节偏移和分片名。
            offset_bytes, shard_name = cursor.allocate(spec.num_elements * FP32_BYTES)

            # 创建 FP32 参数记录；offset_elements 由字节偏移除以 4 转为 float32 元素偏移。
            records[spec.param_id] = ParamShardRecord(
                param_id=spec.param_id,
                shard_name=shard_name,
                offset_elements=offset_bytes // FP32_BYTES,
                num_elements=spec.num_elements,
            )
        # 返回 param_id 到 FP32 分片记录的映射。
        return records

    def _build_adam_records(
        self,
        specs: tuple[TrainingParamManifestSpec, ...],
    ) -> dict[str, AdamStateShardRecord]:
        """为可训练参数生成 Adam 状态分片记录。"""

        # ------------------------------- 初始化 Adam 分片游标 -------------------------------
        # 创建 Adam 分片游标，用于把优化器状态顺序分配到 adam_XXXX.bin 文件中。
        cursor = _ShardCursor(
            shard_bytes=self.config.adam_shard_bytes,
            shard_prefix=self.config.adam_shard_prefix,
            extension=".bin",
        )
        # 初始化 Adam records 字典，key 为 state_key(param_id, component)。
        records: dict[str, AdamStateShardRecord] = {}
        # ------------------------------- 顺序分配 Adam 状态布局 -------------------------------
        # 遍历全部参数规格，只为 trainable=True 的参数创建优化器状态。
        for spec in specs:
            # 不可训练参数不会参与优化器更新，因此跳过 Adam 状态分配。
            if not spec.trainable:
                # 继续处理下一个参数规格。
                continue
            # 根据参数元素数和 Adam block size 估算单个 Adam component 的状态字节数。
            num_bytes = adam_state_num_bytes(
                spec.num_elements,
                block_size=self.config.adam_block_size,
            )
            # 为每个 Adam component 单独分配 record，例如 m 和 v 各自有独立位置。
            for component in self.config.adam_components:
                # 为当前 Adam component 申请 num_bytes 空间，得到 shard 和字节偏移。
                offset_bytes, shard_name = cursor.allocate(num_bytes)
                # 构造当前参数当前 component 的 Adam 状态记录。
                record = AdamStateShardRecord(
                    param_id=spec.param_id,
                    component=component,
                    shard_name=shard_name,
                    offset_bytes=offset_bytes,
                    num_bytes=num_bytes,
                )
                # 使用 state_key 作为字典 key，避免同一 param_id 的不同 component 相互覆盖。
                records[state_key(spec.param_id, component)] = record
        # 返回 Adam 状态索引表，供 CpuAdamFp8StateStore 按 key 定位状态数据。
        return records

    def _build_gptq_records(
        self,
        specs: tuple[TrainingParamManifestSpec, ...],
    ) -> tuple[dict[str, GptqCacheRecord], dict[str, str]]:
        """为带 GPTQ bundle 的参数生成 GPTQ cache 分片记录和反查表。"""

        # ------------------------------- 初始化 GPTQ 分片游标与默认布局 -------------------------------
        # 创建 GPTQ 分片游标，用于把量化 cache bundle 顺序分配到 gptq_XXXX.bin 文件中。
        cursor = _ShardCursor(
            shard_bytes=self.config.gptq_shard_bytes,
            shard_prefix=self.config.gptq_shard_prefix,
            extension=".bin",
        )
        # 初始化 GPTQ records 字典，key 为 gptq_bundle_id。
        records: dict[str, GptqCacheRecord] = {}
        # 初始化 param 到 bundle 的反查表，使训练参数能定位对应的量化 cache。
        param_to_bundle: dict[str, str] = {}
        # 生成默认 GPTQ 布局哈希，用于未显式指定 quant_layout_hash 的规格。
        default_layout_hash = gptq_layout_hash(
            group_size=self.config.gptq_group_size
        )
        # ------------------------------- 顺序分配 GPTQ cache 布局 -------------------------------
        # 遍历所有参数规格，只为显式配置 gptq_bundle_id 的参数生成量化 cache 记录。
        for spec in specs:
            # 没有 GPTQ bundle 的参数不需要量化 cache，因此跳过。
            if spec.gptq_bundle_id is None:
                # 继续处理下一个参数规格。
                continue
            # GPTQ bundle ID 必须唯一，否则多个参数会写入同一个 cache 记录。
            if spec.gptq_bundle_id in records:
                # 抛出重复 bundle 错误，避免量化 cache 布局被后一个规格覆盖。
                raise ValueError(f"duplicate GPTQ bundle {spec.gptq_bundle_id!r}")
            # 优先使用显式 gptq_num_bytes；未指定时按元素数和 group size 估算 bundle 字节数。
            num_bytes = (
                spec.resolved_gptq_num_bytes
                if spec.resolved_gptq_num_bytes is not None
                else gptq_bundle_num_bytes(
                    spec.resolved_gptq_num_elements,
                    group_size=self.config.gptq_group_size,
                )
            )
            # 为当前 GPTQ bundle 申请 num_bytes 空间，得到 shard 和字节偏移。
            offset_bytes, shard_name = cursor.allocate(num_bytes)
            # 创建 GPTQ cache 记录，保存 bundle 的分片位置、大小和量化布局哈希。
            records[spec.gptq_bundle_id] = GptqCacheRecord(
                bundle_id=spec.gptq_bundle_id,
                shard_name=shard_name,
                offset_bytes=offset_bytes,
                num_bytes=num_bytes,
                quant_layout_hash=spec.quant_layout_hash or default_layout_hash,
            )
            # 建立 param_id 到 bundle_id 的映射，便于通过训练参数反查量化 cache。
            param_to_bundle[spec.param_id] = spec.gptq_bundle_id
        # 返回 GPTQ cache 记录表和 param_id 反查表。
        return records, param_to_bundle

    @staticmethod
    def _validate_unique_specs(
        specs: tuple[TrainingParamManifestSpec, ...],
    ) -> None:
        """校验参数规格中的 param_id 唯一，避免 manifest record 被覆盖。"""

        # ------------------------------- 检查 param_id 唯一性 -------------------------------
        # 初始化已见 param_id 集合，用于检测重复参数规格。
        seen: set[str] = set()
        # 遍历全部规格，按 param_id 做唯一性校验。
        for spec in specs:
            # 如果 param_id 已出现，说明 records 字典会发生覆盖风险。
            if spec.param_id in seen:
                # 抛出重复参数错误，阻止生成不一致 manifest。
                raise ValueError(f"duplicate param_id {spec.param_id!r}")
            # 记录当前 param_id，供后续规格检测重复。
            seen.add(spec.param_id)


@dataclass(frozen=True, slots=True)
class Qwen35MoeManifestConfig:
    """Qwen3.5 MoE 参数规格生成配置，用于批量生成 expert 权重 manifest specs。"""

    # num_layers 表示需要生成规格的层数 L。
    num_layers: int
    # num_experts 表示每层专家总数 E。
    num_experts: int
    # hidden_size 表示 hidden 维度 H。
    hidden_size: int
    # intermediate_size 表示每个 expert 的中间维度 I。
    intermediate_size: int
    # tp_size 表示张量并行规模，intermediate 会按 TP 切分到每个 rank。
    tp_size: int = 1
    # layer_start 表示生成层编号的起点，用于多机或分段导入时偏移层号。
    layer_start: int = 0
    # layer_prefix 表示参数 ID 的层名前缀，默认生成 layers.x.experts.y.xxx。
    layer_prefix: str = "layers"
    # trainable 表示生成的参数是否参与训练，影响 Adam record 是否生成。
    trainable: bool = True
    # include_gptq_cache 表示是否为每个参数同步生成 GPTQ cache bundle 规格。
    include_gptq_cache: bool = True
    # local_expert_ids 表示本 rank 或本次导入只关心的专家 ID；None 表示全部专家。
    local_expert_ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        """校验 Qwen3.5 MoE 规格生成配置，确保维度和专家范围合法。"""

        # 层数必须为正，保证至少生成一个层范围。
        _require_positive_int("num_layers", self.num_layers)
        # 专家数必须为正，保证每层至少有一个 expert。
        _require_positive_int("num_experts", self.num_experts)
        # hidden 维度 H 必须为正，保证权重元素数计算有效。
        _require_positive_int("hidden_size", self.hidden_size)
        # intermediate 维度 I 必须为正，保证 expert MLP 权重规模有效。
        _require_positive_int("intermediate_size", self.intermediate_size)
        # TP size 必须为正，保证 intermediate_per_rank 计算有效。
        _require_positive_int("tp_size", self.tp_size)
        # layer_start 允许从 0 开始，但不能为负数。
        _require_non_negative_int("layer_start", self.layer_start)
        # layer_prefix 必须非空，否则生成的 param_id 不可读且可能冲突。
        _require_non_empty_string("layer_prefix", self.layer_prefix)
        # intermediate 必须能被 TP 整除，保证每个 rank 的专家中间维度一致。
        if self.intermediate_size % self.tp_size:
            # 抛出维度切分错误，避免生成非整数 intermediate_per_rank。
            raise ValueError("intermediate_size must be divisible by tp_size")
        # 只有显式传入 local_expert_ids 时才检查专家编号范围。
        if self.local_expert_ids is not None:
            # 遍历本地专家 ID，确保每个 ID 都落在 [0, num_experts) 内。
            for expert_id in self.local_expert_ids:
                # expert_id 不能为负，保证参数 ID 和 checkpoint 专家维度一致。
                _require_non_negative_int("expert_id", expert_id)
                # expert_id 不能超过每层专家总数，避免生成不存在的专家参数。
                if expert_id >= self.num_experts:
                    # 抛出范围错误，提示调用方修正本地专家列表。
                    raise ValueError("local_expert_ids must be < num_experts")


def make_qwen35_moe_manifest_specs(
    config: Qwen35MoeManifestConfig,
) -> tuple[TrainingParamManifestSpec, ...]:
    """根据 Qwen3.5 MoE 结构配置生成 w13/w2 参数规格列表。"""

    # ------------------------------- 计算专家范围与每 rank 维度 -------------------------------
    # 未指定 local_expert_ids 时生成所有专家；指定时只生成本地专家规格。
    expert_ids = (
        config.local_expert_ids
        if config.local_expert_ids is not None
        else tuple(range(config.num_experts))
    )
    # intermediate_per_rank 表示 TP 后每个 rank 持有的 expert 中间维度 I_rank。
    intermediate_per_rank = config.intermediate_size // config.tp_size
    # w13 元素数为 2 * I_rank * H，对应 gate/up 两个 [I_rank, H] 权重。
    w13_elements = 2 * intermediate_per_rank * config.hidden_size
    # w2 元素数为 H * I_rank，对应 down 权重 [H, I_rank]。
    w2_elements = config.hidden_size * intermediate_per_rank
    # 初始化规格列表，后续按 layer/expert/weight_name 三层循环追加。
    specs: list[TrainingParamManifestSpec] = []

    # ------------------------------- 批量生成每层每专家的参数规格 -------------------------------
    # 遍历目标层编号范围，右边界为 layer_start + num_layers。
    for layer_id in range(config.layer_start, config.layer_start + config.num_layers):
        # 遍历目标专家 ID，生成当前层每个 expert 的 w13 和 w2 规格。
        for expert_id in expert_ids:
            # 遍历当前 expert 的两类权重名与元素数，统一构造 param_id。
            for weight_name, num_elements in (
                ("w13_weight", w13_elements),
                ("w2_weight", w2_elements),
            ):
                # 生成内部参数 ID，作为 FP32 record、Adam record 和 GPTQ bundle 的关联主键。
                param_id = (
                    f"{config.layer_prefix}.{layer_id}."
                    f"experts.{expert_id}.{weight_name}"
                )
                # 添加当前参数规格；include_gptq_cache 为真时使用 param_id 作为 GPTQ bundle_id。
                specs.append(
                    TrainingParamManifestSpec(
                        param_id=param_id,
                        num_elements=num_elements,
                        trainable=config.trainable,
                        gptq_bundle_id=(
                            param_id if config.include_gptq_cache else None
                        ),
                    )
                )
    # 返回不可变 tuple，保证后续 builder 使用稳定规格集合。
    return tuple(specs)


@dataclass(slots=True)
class _ShardCursor:
    """顺序分片游标，用于把连续 record 分配到 shard 文件和字节偏移。"""

    # shard_bytes 表示单个 shard 的目标最大容量。
    shard_bytes: int

    # shard_prefix 表示生成 shard 文件名时使用的前缀。
    shard_prefix: str

    # extension 表示 shard 文件扩展名，例如 .bin。
    extension: str

    # shard_index 表示当前正在分配的 shard 编号。
    shard_index: int = 0

    # offset_bytes 表示当前 shard 内下一个可分配位置的字节偏移。
    offset_bytes: int = 0

    def allocate(self, num_bytes: int) -> tuple[int, str]:
        """为一段连续字节空间分配当前 shard 内偏移和 shard 文件名。"""

        # ------------------------------- 校验申请大小并处理换片 -------------------------------
        # 每次分配的字节数必须为正，避免生成空 record 或倒退游标。
        _require_positive_int("num_bytes", num_bytes)
        # 如果当前 shard 已有内容且新记录放不下，则切换到下一个 shard 起始位置。
        if self.offset_bytes and self.offset_bytes + num_bytes > self.shard_bytes:
            # shard 编号加一，使后续 record 写入新的分片文件。
            self.shard_index += 1
            # 新 shard 从 0 字节偏移开始分配。
            self.offset_bytes = 0
        # 根据当前 shard_index 生成分片文件名。
        shard_name = self._shard_name()
        # 记录当前分配的起始偏移，作为返回给 record 的 offset_bytes。
        offset_bytes = self.offset_bytes
        # 游标向后推进 num_bytes，为下一个 record 预留起点。
        self.offset_bytes += num_bytes
        # 如果单个 record 本身超过 shard_bytes，则允许它独占当前 shard，并把游标切到下一片。
        if self.offset_bytes > self.shard_bytes:
            # shard 编号加一，避免后续 record 继续追加到已超限分片。
            self.shard_index += 1
            # 新 shard 从 0 字节偏移重新开始。
            self.offset_bytes = 0
        # 返回当前 record 在 shard 内的起始字节偏移和所在分片文件名。
        return offset_bytes, shard_name

    def _shard_name(self) -> str:
        """根据当前分片编号生成稳定的 shard 文件名。"""

        # 使用 4 位补零编号生成文件名，保证按字典序排序时也符合分片顺序。
        return f"{self.shard_prefix}_{self.shard_index:04d}{self.extension}"
