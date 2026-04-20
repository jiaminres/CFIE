"""Dataset-backed batch planners for the CFIE training runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Literal

from transformers import AutoTokenizer

from cfie_training.config import TrainingProjectConfig
from cfie_training.runtime.types import BatchPlannerCheckpoint, BatchShape

DatasetFormat = Literal["auto", "text", "jsonl"]


# 校验整数字段必须为正数。
def _require_positive_int(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be >= 1")


# 校验字符串字段不能为空。
def _require_non_empty(name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


@dataclass(slots=True, frozen=True)
class _TokenizedSample:
    sample_index: int
    token_ids: tuple[int, ...]


@dataclass(slots=True)
class TokenizedDatasetBatchPlanner:
    # ------------------------------- 外部输入配置字段 -------------------------------
    # 训练项目全局配置，用于回退 tokenizer 路径与共享运行参数。
    config: TrainingProjectConfig
    # 数据集文件路径，支持 text/jsonl 两类输入。
    dataset_path: str
    # 每步计划采样的基础样本数。
    base_samples: int
    # 每个样本目标 token 长度。
    tokens_per_sample: int
    # 可选 tokenizer 路径；为空时回退到 model_source.model_path。
    tokenizer_path: str | None = None
    # 数据集格式声明；auto 时按后缀推断。
    dataset_format: DatasetFormat = "auto"
    # JSONL 模式下用于提取文本的字段名。
    dataset_text_key: str = "text"
    # 是否在编码后追加 eos/fill token。
    add_eos_token: bool = True

    # ------------------------------- 初始化后缓存字段 -------------------------------
    # 规范化后的数据集 Path 对象，避免后续重复转换。
    _dataset_path: Path = field(init=False, repr=False)
    # 数据集文件名缓存，便于日志与 checkpoint 标识。
    _dataset_name: str = field(init=False, repr=False)
    # 最终解析出的实际数据格式（text 或 jsonl）。
    _resolved_dataset_format: Literal["text", "jsonl"] = field(
        init=False,
        repr=False,
    )
    # 运行期 tokenizer 实例缓存。
    _tokenizer: Any = field(init=False, repr=False)
    # 补齐/尾部追加使用的 token id 缓存。
    _fill_token_id: int = field(init=False, repr=False)
    # 预分词后的样本缓存，供批规划阶段复用。
    _samples: tuple[_TokenizedSample, ...] = field(init=False, repr=False)

    # 初始化数据集批规划器，并在构造阶段完成路径校验、tokenizer 准备与样本缓存。
    def __post_init__(self) -> None:
        # ------------------------------- 校验输入参数并确定数据集基础元信息 -------------------------------
        # 先校验关键入参，避免无效配置进入后续 I/O 与分词流程。
        _require_non_empty("dataset_path", self.dataset_path)
        _require_positive_int("base_samples", self.base_samples)
        _require_positive_int("tokens_per_sample", self.tokens_per_sample)
        _require_non_empty("dataset_text_key", self.dataset_text_key)

        # 将字符串路径转成 Path，后续文件存在性检查与读取都基于该对象。
        dataset_path = Path(self.dataset_path)

        # 数据集文件必须真实存在，否则无法继续构建样本缓存。
        if not dataset_path.exists():
            raise FileNotFoundError(f"dataset file not found: {dataset_path}")

        # 限制可接受的数据格式选项，防止拼写错误导致隐式分支偏移。
        if self.dataset_format not in {"auto", "text", "jsonl"}:
            raise ValueError(
                "dataset_format must be one of auto, text, or jsonl"
            )

        # 缓存标准化后的路径与文件名，供后续 checkpoint 与日志复用。
        self._dataset_path = dataset_path
        self._dataset_name = dataset_path.name

        # 根据用户配置与文件后缀解析最终生效的数据格式。
        self._resolved_dataset_format = self._resolve_dataset_format(dataset_path)

        # ------------------------------- 初始化 tokenizer 并预加载样本缓存 -------------------------------
        # 若未显式指定 tokenizer_path，则默认回退到当前训练配置的模型路径。
        tokenizer_root = self.tokenizer_path or self.config.model_source.model_path

        # tokenizer 根路径不能为空，否则无法加载分词器。
        _require_non_empty("tokenizer_path", tokenizer_root)

        # 加载 tokenizer，后续样本分词与补齐都会依赖该实例。
        self._tokenizer = _load_tokenizer(tokenizer_root)

        # 解析用于补齐与可选 EOS 追加的填充 token id。
        self._fill_token_id = self._resolve_fill_token_id()

        # 在初始化阶段一次性完成样本读取与分词缓存，避免训练时重复 I/O 与重复分词。
        self._samples = self._load_samples()

    # 返回当前数据集的文件名。
    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    # 返回当前已经缓存的样本条数。
    @property
    def sample_count(self) -> int:
        return len(self._samples)

    # 生成可写入 checkpoint 的 batch 规划器快照。
    def planner_checkpoint(self) -> BatchPlannerCheckpoint:
        # 把当前数据集规划参数完整写入 checkpoint。
        return BatchPlannerCheckpoint(
            planner_kind="tokenized_dataset",
            base_samples=self.base_samples,
            tokens_per_sample=self.tokens_per_sample,
            dataset_path=self.dataset_path,
            tokenizer_path=self.tokenizer_path,
            dataset_format=self.dataset_format,
            dataset_text_key=self.dataset_text_key,
        )

    # 根据配置或文件后缀推断实际数据集格式。
    def _resolve_dataset_format(self, dataset_path: Path) -> Literal["text", "jsonl"]:
        # 用户显式指定 text 时直接返回 text。
        if self.dataset_format == "text":
            return "text"
        # 用户显式指定 jsonl 时直接返回 jsonl。
        if self.dataset_format == "jsonl":
            return "jsonl"
        # 自动模式下按文件后缀识别 jsonl。
        if dataset_path.suffix.lower() == ".jsonl":
            return "jsonl"
        # 其余情况默认按普通文本处理。
        return "text"

    # 解析 tokenizer 可用的填充 / 终止 token id。
    def _resolve_fill_token_id(self) -> int:
        # 优先用 eos token 作为补齐与尾部 token。
        if self._tokenizer.eos_token_id is not None:
            return int(self._tokenizer.eos_token_id)

        # 若 eos 不存在，则退回到 pad token。
        if self._tokenizer.pad_token_id is not None:
            return int(self._tokenizer.pad_token_id)
        # 再退化时使用 0。
        return 0

    # 从普通文本数据集中读取非空文本行。
    def _read_text_records(self) -> tuple[str, ...]:
        # 逐行读取文本并去掉空白与空行。
        texts = tuple(
            line.strip()
            for line in self._dataset_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        # 若没有任何有效文本，则报错。
        if not texts:
            raise ValueError(f"dataset file {self._dataset_path} does not contain text")
        # 返回去空后的文本元组。
        return texts

    # 从 JSONL 数据集中读取指定文本字段。
    def _read_jsonl_records(self) -> tuple[str, ...]:
        # ------------------------------- 逐行解析 JSONL 并抽取训练文本 -------------------------------
        # 该函数负责把 JSONL 文件转换为纯文本序列，并严格校验每条记录结构。
        texts: list[str] = []
        for line_number, line in enumerate(
                self._dataset_path.read_text(encoding="utf-8").splitlines(),
                start=1,
        ):
            # 先去掉当前行首尾空白，统一空行与有效行的判定口径。
            raw = line.strip()

            # 空行不参与样本构建，直接跳过。
            if not raw:
                continue

            # 将当前行 JSON 文本解析为 Python 对象。
            payload = json.loads(raw)

            # 每条 JSONL 记录都必须是对象字典，便于按字段取文本。
            if not isinstance(payload, dict):
                raise ValueError(
                    f"jsonl record at line {line_number} must decode to an object"
                )

            # 按配置字段名提取训练文本内容。
            value = payload.get(self.dataset_text_key)

            # 目标文本字段缺失或为空时立即报错，防止脏数据静默混入样本。
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"jsonl record at line {line_number} is missing non-empty "
                    f"'{self.dataset_text_key}' text"
                )
            # 记录规范化后的有效文本，供后续分词编码使用。
            texts.append(value.strip())

        # 全文件扫描后若仍无有效文本，则判定该数据集不可用于训练。
        if not texts:
            raise ValueError(
                f"dataset file {self._dataset_path} does not contain JSONL text records"
            )

        # 返回不可变文本元组，确保后续流程不会意外修改原始记录集合。
        return tuple(texts)

    # 将单条文本编码为 token 序列，并按需补尾部终止 token。
    def _encode_text(self, text: str) -> tuple[int, ...]:
        # 关闭 tokenizer 的自动 special token，保留原始文本编码。
        token_ids = list(
            self._tokenizer.encode(
                text,
                add_special_tokens=False,
            )
        )
        # 如配置要求，则手工补一个 eos / fill token。
        if self.add_eos_token:
            token_ids.append(self._fill_token_id)
        # 如果文本被编码为空，则至少补一个 fill token。
        if not token_ids:
            token_ids = [self._fill_token_id]
        # 强制转成整数元组返回。
        return tuple(int(token_id) for token_id in token_ids)

    # 读取原始文本并预编码为可复用样本缓存。
    def _load_samples(self) -> tuple[_TokenizedSample, ...]:
        # 按已解析的数据集格式选择读取方式。
        if self._resolved_dataset_format == "jsonl":
            texts = self._read_jsonl_records()
        else:
            texts = self._read_text_records()
        # 为每条文本生成带 sample_index 的 tokenized sample。
        return tuple(
            _TokenizedSample(
                sample_index=index,
                token_ids=self._encode_text(text),
            )
            for index, text in enumerate(texts)
        )

    # 将样本 token 补齐到训练窗口所需长度。
    def _pad_tokens(
            self,
            token_ids: tuple[int, ...],
            *,
            required_tokens: int,
    ) -> tuple[int, ...]:
        # 样本已经足够长时，直接复用原 token 序列，避免额外拷贝。
        if len(token_ids) >= required_tokens:
            return token_ids

        # 不足长度的位置统一写入 fill token；有效性由 attention mask 单独表达。
        return token_ids + tuple(
            self._fill_token_id for _ in range(required_tokens - len(token_ids))
        )

    # 构造尾部 padding 使用的 0/1 mask。
    def _tail_padding_mask(
            self,
            *,
            valid_tokens: int,
            width: int,
    ) -> tuple[int, ...]:
        # 有效 token 数被夹到窗口范围内，避免异常样本破坏 mask 形状。
        valid_tokens = max(0, min(int(valid_tokens), width))

        # 前缀位置为 1，尾部 padding 位置为 0。
        return tuple(
            1 if position < valid_tokens else 0
            for position in range(width)
        )

    # 为指定 step / 行号切出一组输入与目标 token。
    def _slice_training_pair(
            self,
            *,
            step_index: int,
            row_index: int,
            sample: _TokenizedSample,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # ------------------------------- 补齐样本并计算当前行的切片起点 -------------------------------
        # next-token 训练需要输入与目标各占 tokens_per_sample，并且目标右移一位，
        # 因此这里要求样本至少有 tokens_per_sample + 1 个 token。
        required_tokens = self.tokens_per_sample + 1

        # 对不足长度的样本只做尾部 padding，不再复制原文 token 制造伪上下文。
        token_ids = self._pad_tokens(
            sample.token_ids,
            required_tokens=required_tokens,
        )

        # 最大起点由“样本有效长度 - 所需窗口长度”决定，小于 0 时夹到 0。
        max_offset = max(0, len(token_ids) - required_tokens)

        # 使用 step、行号与 sample_index 混合生成确定性种子，
        # 保证同输入可复现，同时避免所有行都落在同一切片位置。
        offset_seed = (
                step_index * 104_729
                + row_index * 8_191
                + sample.sample_index * 131
        )

        # 若窗口不可滑动则固定从 0 开始；否则按种子在可用起点范围内取模。
        offset = 0 if max_offset == 0 else offset_seed % (max_offset + 1)

        # ------------------------------- 生成 next-token 输入、目标和 mask 窗口 -------------------------------
        # 输入窗口取 [offset, offset + tokens_per_sample)。
        input_ids = token_ids[offset: offset + self.tokens_per_sample]

        # 目标窗口整体右移一位，与输入形成 next-token 监督对。
        target_ids = token_ids[offset + 1: offset + required_tokens]

        # 输入 mask 标记当前窗口中真实来自样本的 token；尾部补齐 token 不参与 teacher 捕获。
        input_mask = self._tail_padding_mask(
            valid_tokens=len(sample.token_ids) - offset,
            width=self.tokens_per_sample,
        )

        # 目标 mask 标记可参与 next-token loss 的位置；没有真实 next token 的尾部位置置 0。
        target_mask = self._tail_padding_mask(
            valid_tokens=len(sample.token_ids) - offset - 1,
            width=self.tokens_per_sample,
        )
        return input_ids, target_ids, input_mask, target_mask

    # 为指定训练 step 生成一整个 batch 的 token 行。
    def batch_for_step(self, step_index: int) -> BatchShape:
        # ------------------------------- 计算当前 step 的样本索引轮转窗口 -------------------------------
        # 每个 step 按 base_samples 为步长在样本池中环形前移，
        # 使连续 step 能覆盖不同样本而不依赖随机状态。
        start_index = (step_index * self.base_samples) % len(self._samples)

        # 当前 batch 的每一行都从轮转窗口取一个样本索引。
        selected_indices = tuple(
            (start_index + offset) % len(self._samples)
            for offset in range(self.base_samples)
        )

        # ------------------------------- 按行切分输入与目标 token -------------------------------
        # 逐行生成 next-token 训练对，并分别收集输入、目标和对应有效 mask。
        token_rows: list[tuple[int, ...]] = []
        target_rows: list[tuple[int, ...]] = []
        attention_mask_rows: list[tuple[int, ...]] = []
        target_attention_mask_rows: list[tuple[int, ...]] = []
        for row_index, sample_index in enumerate(selected_indices):
            # 对当前行样本切出一组可复现的输入/目标窗口。
            (
                token_row,
                target_row,
                attention_mask,
                target_attention_mask,
            ) = self._slice_training_pair(
                step_index=step_index,
                row_index=row_index,
                sample=self._samples[sample_index],
            )
            # 追加当前行输入 token。
            token_rows.append(token_row)
            # 追加当前行目标 token。
            target_rows.append(target_row)
            # 追加当前行输入有效 mask。
            attention_mask_rows.append(attention_mask)
            # 追加当前行目标 loss mask。
            target_attention_mask_rows.append(target_attention_mask)

        # ------------------------------- 组装并返回 BatchShape -------------------------------
        # 将样本索引、输入行、目标行和 loss token 统计打包成统一批描述对象。
        loss_token_count = sum(
            int(value)
            for row in target_attention_mask_rows
            for value in row
        )
        return BatchShape(
            samples=self.base_samples,
            tokens_per_sample=self.tokens_per_sample,
            source_kind="tokenized_dataset",
            dataset_name=self._dataset_name,
            sample_indices=selected_indices,
            loss_token_count=loss_token_count,
            token_rows=tuple(token_rows),
            target_rows=tuple(target_rows),
            attention_mask_rows=tuple(attention_mask_rows),
            target_attention_mask_rows=tuple(target_attention_mask_rows),
        )


@lru_cache(maxsize=4)
# 按路径缓存加载 tokenizer，避免重复初始化。
def _load_tokenizer(path: str) -> Any:
    # 直接复用 HuggingFace 的 from_pretrained 入口加载 tokenizer。
    return AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=True,
    )
