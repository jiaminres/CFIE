# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
The async worker that transfers experts in the background.
"""

import asyncio
import threading
from typing import TYPE_CHECKING

import torch
from torch.distributed import ProcessGroup

from cfie.distributed.parallel_state import get_eplb_group
from cfie.logger import init_logger

from .rebalance_execute import transfer_layer

if TYPE_CHECKING:
    from .eplb_state import EplbModelState, EplbState

logger = init_logger(__name__)


def start_async_worker(
    state: "EplbState",
    is_profile: bool = False,
) -> threading.Thread:
    # EPLB 异步线程使用独立的 EPLB communicator，而不是普通 EP communicator。
    eplb_group = get_eplb_group().device_group
    # 当前 rank 仅用于日志。
    rank = eplb_group.rank()
    # 异步线程固定绑定到主线程初始化时记录下来的 CUDA device index。
    device_index = state.cuda_device_index
    # 只有开启 async EPLB 时才允许启动这个后台线程。
    assert state.is_async

    def thread_target() -> None:
        # 后台线程必须先切到正确的 CUDA 设备。
        assert device_index is not None
        torch.accelerator.set_device_index(device_index)
        # 为后台传输逻辑准备一条独立 CUDA stream。
        cuda_stream = torch.cuda.Stream(device=device_index)
        # 异步线程内部再起一个独立 asyncio event loop。
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # 常驻运行后台传输循环，等待 rearrange_event 唤醒。
            loop.run_until_complete(
                transfer_run_periodically(
                    state=state,
                    eplb_group=eplb_group,
                    cuda_stream=cuda_stream,
                    is_profile=is_profile,
                )
            )
        except Exception as exc:  # pragma: no cover - diagnostic path
            # 异步线程异常只记日志，不让整个进程直接静默退出。
            logger.exception("async loop error (Rank %d): %s", rank, str(exc))
        finally:
            # 线程结束前关闭自己的 event loop。
            loop.close()

    # 启动 daemon 线程，进程退出时无需额外 join。
    thread = threading.Thread(target=thread_target, daemon=True)
    thread.start()
    return thread


def run_rebalance_experts(
    model_state: "EplbModelState",
    eplb_state: "EplbState",
    physical_to_logical_map_cpu: torch.Tensor,
) -> None:
    # 异步 worker 进入这里时，主线程必须已经准备好了本轮 EPLB 统计。
    assert model_state.eplb_stats is not None
    eplb_stats = model_state.eplb_stats

    # 先等待主线程完成 global_expert_load_window 的 all-reduce 与 clone。
    # Wait for the main thread's all-reduce and clone to complete before
    # accessing the global_expert_load_window tensor.
    assert model_state.window_ready_event is not None
    model_state.window_ready_event.wait()
    model_state.window_ready_event = None

    # 后台重排算法在 CPU 上运行，因此先把全局 logical load 窗口搬到 CPU。
    # Move the global expert load window to CPU for computation.
    global_expert_load_window = eplb_stats.global_expert_load_window.cpu()
    # 计算本轮新的 physical/logical 映射关系。
    # Compute new expert mappings for the model
    (
        new_physical_to_logical_map,
        new_logical_to_physical_map,
        new_logical_replica_count,
    ) = eplb_state.policy.rebalance_experts(
        global_expert_load_window,
        eplb_stats.num_replicas,
        eplb_stats.num_groups,
        eplb_stats.num_nodes,
        eplb_stats.num_gpus,
        physical_to_logical_map_cpu,
    )
    # 异步 worker 当前约定在 CPU 上输出新的 physical 索引。
    assert new_physical_to_logical_map.device == torch.device("cpu")

    # 保存新 physical 映射，后续 transfer_layer / 主线程落盘都要用到。
    model_state.new_physical_to_logical_map = new_physical_to_logical_map

    # logical_to_physical_map 目标张量常常预留了更多副本槽位，因此这里先 pad 再写回。
    max_slots = model_state.logical_to_physical_map.shape[-1]
    padded_logical = torch.nn.functional.pad(
        new_logical_to_physical_map,
        (0, max(0, max_slots - new_logical_to_physical_map.shape[-1])),
        value=-1,
    ).to(model_state.logical_to_physical_map.device)
    new_replica = new_logical_replica_count.to(model_state.logical_replica_count.device)
    model_state.new_logical_to_physical_map = padded_logical
    model_state.new_logical_replica_count = new_replica


async def transfer_run_periodically(
    state: "EplbState",
    eplb_group: ProcessGroup,
    cuda_stream: torch.cuda.Stream,
    is_profile: bool = False,
) -> None:
    # 异步线程常驻循环，只有在 rearrange_event 被置位后才开始一轮传输。
    while True:
        await asyncio.to_thread(state.rearrange_event.wait)
        logger.info("async worker woke up for EPLB transfer")

        # 这里只应被 async EPLB 唤醒。
        assert state.is_async
        for model_state in state.model_states.values():
            # 每个模型在一轮唤醒中只执行一次重排算法，随后逐层传输。
            rebalancing_algorithm_executed = False
            physical_to_logical_map_cpu = None
            current_num_layers = model_state.model.num_moe_layers
            while (
                model_state.rebalanced
                and model_state.layer_to_transfer < current_num_layers
            ):
                if not model_state.ep_buffer_ready and model_state.rebalanced:
                    # 直接轮询 buffer_lock，避免把锁等待再丢给线程池造成额外切换。
                    # Polling the lock directly in the async thread avoids
                    # the thread switch overhead of asyncio.to_thread.
                    # This is typically faster than offloading to a worker thread.
                    while not model_state.buffer_lock.acquire(blocking=False):
                        await asyncio.sleep(0)
                    try:
                        if model_state.layer_to_transfer >= current_num_layers:
                            break
                        if (
                            not rebalancing_algorithm_executed
                            or model_state.new_physical_to_logical_map is None
                        ):
                            # 第一次进入时，先把旧映射搬到 CPU，供重排算法与 transfer_layer 复用。
                            # Move the physical_to_logical_map to CPU
                            # for rebalancing and transfer_layer.
                            physical_to_logical_map_cpu = (
                                model_state.physical_to_logical_map.cpu()
                            )
                            # 计算新的 logical/physical expert 映射。
                            run_rebalance_experts(
                                model_state, state, physical_to_logical_map_cpu
                            )
                            rebalancing_algorithm_executed = True
                            logger.info(
                                "Async worker computed new indices for model %s",
                                model_state.model_name,
                            )

                        assert model_state.new_physical_to_logical_map is not None
                        assert physical_to_logical_map_cpu is not None

                        # 取出当前层的旧索引与新索引，准备做本层权重交换。
                        layer_idx = model_state.layer_to_transfer
                        old_layer_indices = physical_to_logical_map_cpu[layer_idx]
                        new_layer_indices = model_state.new_physical_to_logical_map[
                            layer_idx
                        ]

                        # 若上一层的 buffer 还未被主线程消费完，先等消费完成事件。
                        # Wait for the main thread to finish consuming the buffer
                        # before initiating an EPLB transfer on another layer.
                        if model_state.buffer_consumed_event is not None:
                            cuda_stream.wait_event(model_state.buffer_consumed_event)
                            model_state.buffer_consumed_event = None

                        # 真正执行“当前层 expert 权重 -> 中间 buffer”的传输与 P2P 交换。
                        (
                            model_state.is_unchanged,
                            model_state.is_received_locally,
                            model_state.recv_metadata,
                        ) = await transfer_layer(
                            old_layer_indices=old_layer_indices,
                            new_layer_indices=new_layer_indices,
                            expert_weights=model_state.model.expert_weights[layer_idx],
                            expert_weights_buffer=model_state.expert_buffer,
                            ep_group=eplb_group,
                            is_profile=is_profile,
                            cuda_stream=cuda_stream,
                        )
                        # 在独立 stream 上记录 buffer_ready_event，通知主线程可消费。
                        event = torch.cuda.Event(blocking=False)
                        cuda_stream.record_event(event)
                        model_state.buffer_ready_event = event
                        model_state.ep_buffer_ready = 1
                    finally:
                        # 无论本层传输是否成功，都释放 buffer_lock。
                        model_state.buffer_lock.release()
                else:
                    if not model_state.rebalanced:
                        break
                    # 若主线程还没消费完 buffer，则短暂 sleep，避免纯忙等。
                    await asyncio.sleep(0.001)

        # 一轮所有模型都处理完后清掉事件，等待下一次重排触发。
        state.rearrange_event.clear()
