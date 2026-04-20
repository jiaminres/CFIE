"""Microbatch and dual-stream schedule planning for the CFIE training runtime."""

from __future__ import annotations

from dataclasses import dataclass

from cfie_training.config import TrainingProjectConfig
from cfie_training.runtime.types import (
    BatchShape,
    BucketStreamTrace,
    StreamOperationTrace,
    StreamOverlapSummary,
)


@dataclass(slots=True)
class MicroBatchPlanner:
    config: TrainingProjectConfig

    # 估算单个 micro-batch 允许承载的最大样本数。
    def _max_samples_per_micro_batch(self, batch: BatchShape) -> int:
        # sample_parallelism 给出样本并行的基础上限。
        base_limit = max(1, self.config.execution.sample_parallelism)
        # token 预算至少不能小于单样本 token 长度。
        token_budget = max(
            self.config.execution.max_tokens_per_micro_batch,
            batch.tokens_per_sample,
        )
        # 按 token 预算推导 token 维度允许的样本数。
        token_limited = max(1, token_budget // batch.tokens_per_sample)
        if (
            self.config.execution.compute_device == "gpu"
            and self.config.execution.gradient_device == "cpu"
            and self.config.execution.overlap_backward_and_update
        ):
            # GPU 计算 + CPU 更新模式下仍至少保留 1 个样本。
            token_limited = max(1, token_limited)
        # 最终结果取 sample 上限和 token 上限中的较小值。
        return max(1, min(base_limit, token_limited))

    # 将大 batch 按样本维度切分为若干 micro-batch。
    def plan(self, batch: BatchShape) -> tuple[BatchShape, ...]:
        # 先计算单个 micro-batch 最多容纳多少样本。
        max_samples = self._max_samples_per_micro_batch(batch)
        # 剩余待切分的样本数初始等于整个 batch 的样本数。
        remaining_samples = batch.samples
        # batch_start 表示当前 micro-batch 在原 batch 中的起始行号。
        batch_start = 0
        # 收集切分后的 micro-batch 列表。
        micro_batches: list[BatchShape] = []
        while remaining_samples > 0:
            # 当前 micro-batch 实际样本数受剩余样本数约束。
            micro_batch_samples = min(max_samples, remaining_samples)
            # 计算当前 micro-batch 在原 batch 中的结束位置。
            batch_end = batch_start + micro_batch_samples
            # 有目标 mask 时按真实 loss token 计数；否则保持旧的形状计数口径。
            if batch.target_attention_mask_rows:
                loss_token_count = sum(
                    int(value)
                    for row in batch.target_attention_mask_rows[batch_start:batch_end]
                    for value in row
                )
            else:
                loss_token_count = micro_batch_samples * batch.tokens_per_sample
            # 用原 batch 的局部切片构造当前 micro-batch。
            micro_batches.append(
                BatchShape(
                    samples=micro_batch_samples,
                    tokens_per_sample=batch.tokens_per_sample,
                    source_kind=batch.source_kind,
                    dataset_name=batch.dataset_name,
                    sample_indices=batch.sample_indices[batch_start:batch_end],
                    loss_token_count=loss_token_count,
                    token_rows=batch.token_rows[batch_start:batch_end],
                    target_rows=batch.target_rows[batch_start:batch_end],
                    attention_mask_rows=(
                        batch.attention_mask_rows[batch_start:batch_end]
                    ),
                    target_attention_mask_rows=(
                        batch.target_attention_mask_rows[batch_start:batch_end]
                    ),
                )
            )
            # 扣减已经分配出去的样本数。
            remaining_samples -= micro_batch_samples
            # 把起始行号推进到下一个切片起点。
            batch_start = batch_end
        # 返回不可变的 micro-batch 元组。
        return tuple(micro_batches)


@dataclass(slots=True)
class StreamTimelinePlanner:
    config: TrainingProjectConfig

    # 估算单个 bucket 计算流操作的时长。
    def _estimate_compute_duration_us(
        self,
        *,
        micro_batch: BatchShape,
        bucket_trace: BucketStreamTrace,
    ) -> int:
        # full attention bucket 的计算因子略高。
        attention_factor = 1.25 if bucket_trace.bucket_record.contains_full_attention else 0.9
        # 激活占用越高，估算时长越长。
        activation_term = bucket_trace.bucket_record.peak_activation_bytes // 2048
        # token 数乘层数近似刻画计算规模。
        token_term = micro_batch.total_tokens * max(len(bucket_trace.layer_indices), 1)
        # 返回一个带基础常数项的估算时长。
        return max(
            1,
            int(160 + activation_term * attention_factor + token_term * 2),
        )

    # 估算单个 bucket 在传输 / 更新流上的时长。
    def _estimate_transfer_duration_us(
        self,
        *,
        micro_batch: BatchShape,
        bucket_trace: BucketStreamTrace,
    ) -> int:
        # 至少按 1 个更新 group 估算传输 / 更新代价。
        update_groups = max(bucket_trace.optimizer_update_count, 1)
        # 梯度范数近似反映传输与更新工作量。
        gradient_term = (
            bucket_trace.bucket_record.non_routed_gradient_l2_norm
            + bucket_trace.bucket_record.expert_gradient_l2_norm
        )
        # token 数乘更新 group 数刻画规模项。
        token_term = micro_batch.total_tokens * update_groups
        # 返回一个带基础常数项的估算时长。
        return max(
            1,
            int(120 + gradient_term * 320 + token_term * 0.35),
        )

    # 规划 micro-batch 与 bucket 在双流上的重叠时间线。
    def plan(
        self,
        *,
        micro_batches: tuple[BatchShape, ...],
        bucket_stream_traces: tuple[BucketStreamTrace, ...],
    ) -> tuple[StreamOverlapSummary, tuple[StreamOperationTrace, ...]]:
        # -----------------
        # 初始化时间轴游标与统计量。
        # 保存最终生成的双流操作列表。
        operations: list[StreamOperationTrace] = []
        # compute_cursor 表示计算流当前时间游标。
        compute_cursor = 0
        # transfer_cursor 表示更新流当前时间游标。
        transfer_cursor = 0
        # 统计计算流因等待更新流而产生的等待时间。
        compute_wait_us = 0
        # 统计更新流在等待计算结果时的空闲时间。
        transfer_idle_us = 0
        # 记录更新启动相对计算结束的最大滞后。
        max_update_lag_us = 0
        # 累计所有计算操作的串行总时长。
        total_compute_us = 0
        # 累计所有更新操作的串行总时长。
        total_transfer_us = 0

        # -----------------
        # 逐个 micro-batch / bucket 生成计算流与更新流操作。
        for micro_batch_id, micro_batch in enumerate(micro_batches):
            for bucket_trace in bucket_stream_traces:
                # 先估算当前 bucket 的计算时长。
                compute_duration_us = self._estimate_compute_duration_us(
                    micro_batch=micro_batch,
                    bucket_trace=bucket_trace,
                )
                # 默认情况下计算流紧接上一个计算操作开始。
                compute_start_us = compute_cursor
                # 若不允许计算和更新重叠，则计算流必须等待更新流。
                if not self.config.execution.overlap_backward_and_update:
                    compute_start_us = max(compute_cursor, transfer_cursor)
                # 统计因等待造成的计算流空转时间。
                compute_wait_us += max(0, compute_start_us - compute_cursor)
                # 计算当前操作结束时间。
                compute_end_us = compute_start_us + compute_duration_us
                # 追加计算流操作记录。
                operations.append(
                    StreamOperationTrace(
                        stream_name=self.config.execution.compute_stream_name,
                        operation="bucket_compute",
                        micro_batch_id=micro_batch_id,
                        batch=micro_batch,
                        bucket_id=bucket_trace.bucket_id,
                        start_time_us=compute_start_us,
                        end_time_us=compute_end_us,
                        duration_us=compute_duration_us,
                        token_count=micro_batch.total_tokens,
                        activation_bytes=bucket_trace.bucket_record.peak_activation_bytes,
                    )
                )

                # 再估算当前 bucket 的更新 / 释放时长。
                transfer_duration_us = self._estimate_transfer_duration_us(
                    micro_batch=micro_batch,
                    bucket_trace=bucket_trace,
                )
                # 更新流必须等待本 bucket 计算结束以及上一个更新结束。
                transfer_start_us = max(transfer_cursor, compute_end_us)
                # 统计更新流由于等待计算结果造成的空闲时间。
                transfer_idle_us += max(0, compute_end_us - transfer_cursor)
                # 更新最大滞后值。
                max_update_lag_us = max(
                    max_update_lag_us,
                    transfer_start_us - compute_end_us,
                )
                # 计算当前更新操作结束时间。
                transfer_end_us = transfer_start_us + transfer_duration_us
                # 追加更新流操作记录。
                operations.append(
                    StreamOperationTrace(
                        stream_name=self.config.execution.transfer_stream_name,
                        operation="bucket_update_release",
                        micro_batch_id=micro_batch_id,
                        batch=micro_batch,
                        bucket_id=bucket_trace.bucket_id,
                        start_time_us=transfer_start_us,
                        end_time_us=transfer_end_us,
                        duration_us=transfer_duration_us,
                        token_count=micro_batch.total_tokens,
                        activation_bytes=bucket_trace.bucket_record.peak_activation_bytes,
                        update_group_count=bucket_trace.optimizer_update_count,
                    )
                )

                # 推进计算流游标到本次计算结束时间。
                compute_cursor = compute_end_us
                # 推进更新流游标到本次更新结束时间。
                transfer_cursor = transfer_end_us
                # 累加计算流串行总时长。
                total_compute_us += compute_duration_us
                # 累加更新流串行总时长。
                total_transfer_us += transfer_duration_us

        # -----------------
        # 汇总整体 makespan 与流重叠指标。
        # makespan 等于两条流中较晚结束的那一条。
        makespan_us = max(compute_cursor, transfer_cursor)
        # total_serial_us 表示两条流完全串行时的总时长。
        total_serial_us = total_compute_us + total_transfer_us
        # 默认重叠比例为 0。
        overlap_ratio = 0.0
        # 只有存在有效时长时才计算重叠比例。
        if total_serial_us > 0:
            overlap_ratio = max(
                0.0,
                min(1.0, (total_serial_us - makespan_us) / total_serial_us),
            )

        # 组装本步双流重叠摘要。
        summary = StreamOverlapSummary(
            micro_batch_count=len(micro_batches),
            scheduled_samples=sum(micro_batch.samples for micro_batch in micro_batches),
            scheduled_tokens=sum(micro_batch.total_tokens for micro_batch in micro_batches),
            compute_operation_count=len(micro_batches) * len(bucket_stream_traces),
            transfer_operation_count=len(micro_batches) * len(bucket_stream_traces),
            compute_stream_span_us=compute_cursor,
            transfer_stream_span_us=transfer_cursor,
            estimated_step_makespan_us=makespan_us,
            compute_wait_us=compute_wait_us,
            transfer_idle_us=transfer_idle_us,
            max_update_lag_us=max_update_lag_us,
            overlap_ratio=overlap_ratio,
        )
        # 返回汇总和完整操作列表。
        return summary, tuple(operations)
