"""训练窗口提交——将 hot set 的 FP32 master/Adam/GPTQ 原子写入 NVMe（设计文档 Section 13.2）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from cfie_training.training_base.adam_state_store import CpuAdamFp8StateStore
from cfie_training.training_base.fp32_shard_store import FP32ShardStore
from cfie_training.training_base.gptq_cache_store import GptqCacheStore
from cfie_training.training_base.progress_state import (
    ProgressStateWriter,
    TrainingProgressState,
)


@dataclass(slots=True)
# ────── TrainingWindowCommitter — 窗口提交：写 NVMe + CPU 缓存 + progress ──────
class TrainingWindowCommitter:
    fp32_store: FP32ShardStore
    progress_writer: ProgressStateWriter
    adam_store: CpuAdamFp8StateStore | None = None
    gptq_store: GptqCacheStore | None = None

    def commit_window(
        self,
        *, fp32_updates: Mapping[str, Any],                          # {param_id: master_tensor}
        adam_updates: Mapping[str, Mapping[str, Any]] | None = None, # {param_id: {m:bytes, v:bytes}}
        gptq_updates: Mapping[str, Any] | None = None,               # {bundle_id: int4_bytes}
        global_step: int, epoch: int, dataset_cursor: str, round_id: int,
        hot_set: Iterable[str] | Mapping[str, Any] | None = None,
        consumed_samples: int = 0, consumed_tokens: int = 0,
        flush_generation: int | None = None,
        optimizer_generation: int | None = None,
        gptq_cache_generation: int | None = None,
    ) -> TrainingProgressState:
        """原子提交训练窗口：FP32 → NVMe, Adam → NVMe, GPTQ → NVMe/CPU, progress state 原子写。"""
        generation = global_step if flush_generation is None else flush_generation

        # 1. FP32 master → NVMe shard 文件（原子 patch）
        self.fp32_store.flush_touched(fp32_updates, generation=generation)

        # 2. Adam FP8 state → NVMe（如果配置了 Adam store）
        if self.adam_store is not None:
            self.adam_store.flush_touched(
                adam_updates or {},
                generation=generation if optimizer_generation is None else optimizer_generation,
            )

        # 3. GPTQ Int4 cache → NVMe 或 CPU 缓存（如果配置了 GPTQ store）
        if self.gptq_store is not None:
            self.gptq_store.flush_touched(
                gptq_updates or {},
                generation=generation if gptq_cache_generation is None else gptq_cache_generation,
            )

        return self.progress_writer.write_after_flush(
            global_step=global_step,
            epoch=epoch,
            dataset_cursor=dataset_cursor,
            round_id=round_id,
            hot_set=hot_set,
            consumed_samples=consumed_samples,
            consumed_tokens=consumed_tokens,
            flush_generation=generation,
            fp32_master_generation=self.fp32_store.generation,
            optimizer_generation=(
                self.adam_store.generation
                if self.adam_store is not None
                else (
                    generation
                    if optimizer_generation is None
                    else optimizer_generation
                )
            ),
            gptq_cache_generation=(
                self.gptq_store.generation
                if self.gptq_store is not None
                else (
                    generation
                    if gptq_cache_generation is None
                    else gptq_cache_generation
                )
            ),
        )

    def commit_fp32_window(
        self,
        **kwargs: Any,
    ) -> TrainingProgressState:
        return self.commit_window(**kwargs)

    def load_committed_progress(self) -> TrainingProgressState:
        state = self.progress_writer.load_latest_or_init()
        state.assert_generations(
            fp32_master_generation=self.fp32_store.generation,
            optimizer_generation=(
                self.adam_store.generation
                if self.adam_store is not None
                else state.optimizer_generation
            ),
            gptq_cache_generation=(
                self.gptq_store.generation
                if self.gptq_store is not None
                else state.gptq_cache_generation
            ),
        )
        return state
