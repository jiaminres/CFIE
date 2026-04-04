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
    config: TrainingProjectConfig
    dataset_path: str
    base_samples: int
    tokens_per_sample: int
    tokenizer_path: str | None = None
    dataset_format: DatasetFormat = "auto"
    dataset_text_key: str = "text"
    add_eos_token: bool = True
    _dataset_path: Path = field(init=False, repr=False)
    _dataset_name: str = field(init=False, repr=False)
    _resolved_dataset_format: Literal["text", "jsonl"] = field(
        init=False,
        repr=False,
    )
    _tokenizer: Any = field(init=False, repr=False)
    _fill_token_id: int = field(init=False, repr=False)
    _samples: tuple[_TokenizedSample, ...] = field(init=False, repr=False)

    # 初始化数据集 batch 规划器，并完成数据与 tokenizer 的预加载。
    def __post_init__(self) -> None:
        # -----------------
        # 校验输入参数并解析数据集路径与格式。
        _require_non_empty("dataset_path", self.dataset_path)
        _require_positive_int("base_samples", self.base_samples)
        _require_positive_int("tokens_per_sample", self.tokens_per_sample)
        _require_non_empty("dataset_text_key", self.dataset_text_key)
        dataset_path = Path(self.dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"dataset file not found: {dataset_path}")
        if self.dataset_format not in {"auto", "text", "jsonl"}:
            raise ValueError(
                "dataset_format must be one of auto, text, or jsonl"
            )
        self._dataset_path = dataset_path
        self._dataset_name = dataset_path.name
        self._resolved_dataset_format = self._resolve_dataset_format(dataset_path)

        # -----------------
        # 准备 tokenizer、填充 token 与全部样本缓存。
        tokenizer_root = self.tokenizer_path or self.config.model_source.model_path
        _require_non_empty("tokenizer_path", tokenizer_root)
        self._tokenizer = _load_tokenizer(tokenizer_root)
        self._fill_token_id = self._resolve_fill_token_id()
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
        # -----------------
        # 逐行解析 JSONL，并提取目标文本字段。
        texts: list[str] = []
        for line_number, line in enumerate(
            self._dataset_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            # 先去掉当前行首尾空白。
            raw = line.strip()
            # 空行直接跳过。
            if not raw:
                continue
            # 将 JSONL 当前行解码为 Python 对象。
            payload = json.loads(raw)
            # 每一行都必须解码成对象字典。
            if not isinstance(payload, dict):
                raise ValueError(
                    f"jsonl record at line {line_number} must decode to an object"
                )
            # 从对象里取出指定文本字段。
            value = payload.get(self.dataset_text_key)
            # 文本字段不存在或为空时直接报错。
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"jsonl record at line {line_number} is missing non-empty "
                    f"'{self.dataset_text_key}' text"
                )
            # 保存当前记录里的有效文本。
            texts.append(value.strip())
        # 读完后若仍无有效文本，同样报错。
        if not texts:
            raise ValueError(
                f"dataset file {self._dataset_path} does not contain JSONL text records"
            )
        # 返回抽取出的文本元组。
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

    # 将样本 token 扩展到训练窗口所需长度。
    def _extend_tokens(
        self,
        token_ids: tuple[int, ...],
        *,
        required_tokens: int,
    ) -> tuple[int, ...]:
        # 样本已经足够长时，直接复用原 token 序列。
        if len(token_ids) >= required_tokens:
            return token_ids
        # 原样本为空时，全部用 fill token 填满。
        if not token_ids:
            return tuple(self._fill_token_id for _ in range(required_tokens))
        # 计算需要重复多少轮才能覆盖所需长度。
        repeats = (required_tokens + len(token_ids) - 1) // len(token_ids)
        # 通过重复已有 token 序列扩展样本。
        extended = token_ids * repeats
        # 只截取前 required_tokens 个 token。
        return extended[:required_tokens]

    # 为指定 step / 行号切出一组输入与目标 token。
    def _slice_training_pair(
        self,
        *,
        step_index: int,
        row_index: int,
        sample: _TokenizedSample,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        # -----------------
        # 先把样本拉到足够长度，再计算本次切片偏移。
        required_tokens = self.tokens_per_sample + 1
        token_ids = self._extend_tokens(
            sample.token_ids,
            required_tokens=required_tokens,
        )
        # 当前样本允许的最大切片起点由有效长度决定。
        max_offset = max(0, len(token_ids) - required_tokens)
        # 用 step / 行号 / sample_index 共同生成确定性偏移种子。
        offset_seed = (
            step_index * 104_729
            + row_index * 8_191
            + sample.sample_index * 131
        )
        # 若不存在可滑动空间，则偏移固定为 0。
        offset = 0 if max_offset == 0 else offset_seed % (max_offset + 1)

        # -----------------
        # 输入与目标错开一个 token，形成 next-token 训练对。
        # 当前输入窗口取 [offset, offset + tokens_per_sample)。
        input_ids = token_ids[offset : offset + self.tokens_per_sample]
        # 当前目标窗口整体右移一个 token。
        target_ids = token_ids[offset + 1 : offset + required_tokens]
        return input_ids, target_ids

    # 为指定训练 step 生成一整个 batch 的 token 行。
    def batch_for_step(self, step_index: int) -> BatchShape:
        # -----------------
        # 先按 step 位置循环选取样本索引。
        # 每个 step 以 base_samples 为步长在样本池中轮转。
        start_index = (step_index * self.base_samples) % len(self._samples)
        # 当前 batch 里的每一行样本索引都按环形方式取值。
        selected_indices = tuple(
            (start_index + offset) % len(self._samples)
            for offset in range(self.base_samples)
        )

        # -----------------
        # 再逐行切出输入 / 目标 token。
        token_rows: list[tuple[int, ...]] = []
        target_rows: list[tuple[int, ...]] = []
        for row_index, sample_index in enumerate(selected_indices):
            # 先从样本池里取出当前行样本。
            token_row, target_row = self._slice_training_pair(
                step_index=step_index,
                row_index=row_index,
                sample=self._samples[sample_index],
            )
            # 记录当前行输入 token。
            token_rows.append(token_row)
            # 记录当前行目标 token。
            target_rows.append(target_row)

        # -----------------
        # 组装训练运行时需要的 BatchShape。
        # 返回带显式 token 行与目标行的 batch 描述对象。
        return BatchShape(
            samples=self.base_samples,
            tokens_per_sample=self.tokens_per_sample,
            source_kind="tokenized_dataset",
            dataset_name=self._dataset_name,
            sample_indices=selected_indices,
            loss_token_count=self.base_samples * self.tokens_per_sample,
            token_rows=tuple(token_rows),
            target_rows=tuple(target_rows),
        )


@lru_cache(maxsize=4)
# 按路径缓存加载 tokenizer，避免重复初始化。
def _load_tokenizer(path: str) -> Any:
    # 直接复用 HuggingFace 的 from_pretrained 入口加载 tokenizer。
    return AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=True,
    )
