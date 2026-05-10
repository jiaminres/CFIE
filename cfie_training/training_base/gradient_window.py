"""训练窗口运行时：梯度 bucket 环 + hot 参数管理 + ForwardShadowStore。

实现设计文档 Section 10（反向与梯度 bucket 环）和 Section 4（参数分层）的核心机制。

核心流程:
  add_gradient → bucket seal → apply_buckets(Adam update) → shadow refresh
  → accumulate touched → window commit → NVMe + CPU GPTQ
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol

import torch

from cfie_training.training_base.adam_state_store import CpuAdamFp8StateStore
from cfie_training.training_base.adam_update import (
    BlockFp8StateCodec,
    CpuAdamFp8Updater,
)
from cfie_training.training_base.fp32_shard_store import FP32ShardStore
from cfie_training.training_base.window_runtime import WindowCommitPayload


def _require_non_empty_string(name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_non_negative_int(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


def _require_positive_int(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be >= 1")


# ──────────────────── GradientBucket ────────────────────

@dataclass(frozen=True, slots=True)
class GradientBucket:
    """一个已 seal 的梯度 bucket：不可变快照，包含累积的梯度。"""
    bucket_id: int               # bucket 编号，全局递增
    grads: Mapping[str, torch.Tensor]  # {param_id: flat_cpu_fp32_grad}
    num_bytes: int               # 梯度总字节数

    def __post_init__(self) -> None:
        _require_non_negative_int("bucket_id", self.bucket_id)
        _require_non_negative_int("num_bytes", self.num_bytes)
        for param_id, grad in self.grads.items():
            _require_non_empty_string("param_id", param_id)
            if grad.dtype != torch.float32:
                raise TypeError("GradientBucket grads must be torch.float32")
            if not grad.device.type == "cpu":
                raise ValueError("GradientBucket grads must be CPU tensors")


# ──────────────────── GradientBucketRing ────────────────────

@dataclass(slots=True)
class GradientBucketRing:
    """梯度 bucket 环——设计文档 Section 10.1-10.3。

    状态机: FILLING → SEALED → (外部消费) → drained → FILLING
    分配规则:
      - 参数反向前必须 add_gradient 预定空间
      - 达到 80% 容量时 seal_active()
      - seal 后立即由上层 drain 并 D2H
      - bucket 耗尽时 backpressure
    """
    bucket_capacity_bytes: int           # 每个 bucket 的容量（字节）
    max_sealed_buckets: int = 4          # 最大已 seal bucket 数（触发 backpressure）
    _next_bucket_id: int = 0             # 下一个 bucket 编号
    _active_grads: dict[str, torch.Tensor] = field(default_factory=dict)  # 当前 FILLING 的梯度累积
    _active_bytes: int = 0               # 当前激活 bucket 的已用字节
    _sealed: list[GradientBucket] = field(default_factory=list)  # 已 seal 待 drain 的 bucket 列表

    def __post_init__(self) -> None:
        _require_positive_int("bucket_capacity_bytes", self.bucket_capacity_bytes)
        _require_positive_int("max_sealed_buckets", self.max_sealed_buckets)

    @property
    def active_bytes(self) -> int:
        return self._active_bytes

    @property
    def sealed_count(self) -> int:
        return len(self._sealed)

    @property
    def is_empty(self) -> bool:
        return not self._active_grads and not self._sealed

    def add_gradient(
        self,
        param_id: str,
        grad: Any,
    ) -> tuple[GradientBucket, ...]:
        """向当前激活 bucket 累积梯度。

        如果添加后超过 bucket 容量，先 seal 当前 bucket 再创建新的。
        返回已 seal 且可 drain 的 bucket 列表。
        """
        _require_non_empty_string("param_id", param_id)
        # 将梯度转为 flat CPU float32 向量 [numel]
        grad_tensor = _as_cpu_float32_vector(grad, name="grad")
        if not torch.isfinite(grad_tensor).all():
            raise ValueError("grad must contain only finite values")
        grad_bytes = grad_tensor.numel() * grad_tensor.element_size()

        # 当前 bucket 放不下 → seal，开新 bucket
        if (
            self._active_grads
            and self._active_bytes + grad_bytes > self.bucket_capacity_bytes
        ):
            self.seal_active()

        # 同参数多次添加 → 累加（grad accum）
        if param_id in self._active_grads:
            existing = self._active_grads[param_id]
            if existing.numel() != grad_tensor.numel():
                raise ValueError("duplicate param gradients must have same numel")
            existing.add_(grad_tensor)  # 原地累加，不增加字节数
        else:
            self._active_grads[param_id] = grad_tensor.clone()
        self._active_bytes += grad_bytes
        return self.drain_ready()  # 返回已 seal 可消费的 bucket

    def seal_active(self) -> GradientBucket | None:
        """冻结当前激活 bucket，创建 GradientBucket 快照。

        如果已 seal 超过 max_sealed_buckets，触发 backpressure。
        """
        if not self._active_grads:
            return None
        # 创建不可变快照
        bucket = GradientBucket(
            bucket_id=self._next_bucket_id,
            grads=dict(self._active_grads),
            num_bytes=self._active_bytes,
        )
        self._next_bucket_id += 1
        # 清空激活区，准备下一个 bucket
        self._active_grads.clear()
        self._active_bytes = 0
        self._sealed.append(bucket)
        # backpressure: sealed 数量超限
        if len(self._sealed) > self.max_sealed_buckets:
            raise RuntimeError("gradient bucket ring backpressure")
        return bucket

    def drain_ready(self) -> tuple[GradientBucket, ...]:
        """取出所有已 seal 的 bucket 供上层消费（Adam update）。"""
        sealed = tuple(self._sealed)
        self._sealed.clear()
        return sealed

    def drain_all(self) -> tuple[GradientBucket, ...]:
        """seal 当前激活 bucket + drain 全部（窗口提交时使用）。"""
        self.seal_active()
        return self.drain_ready()


# ──────────────────── ForwardShadowStore ────────────────────

@dataclass(slots=True)
class ForwardShadowStore:
    """GPU 前向影子参数存储（设计文档 Section 4.1）。

    持有 hot 参数的低精度副本（FP16/BF16），CPU Adam 更新后通过 refresh() 同步。
    """
    dtype: torch.dtype = torch.float16          # shadow 的数据类型
    device: torch.device | str = "cpu"           # shadow 的设备（GPU 或 CPU）
    shadows: dict[str, torch.Tensor] = field(default_factory=dict)  # {param_id: shadow_tensor}

    def load_from_masters(self, masters: Mapping[str, torch.Tensor]) -> None:
        """批量从 FP32 master 加载所有 shadow（窗口初始化时调用）。"""
        for param_id, master in masters.items():
            self.refresh(param_id, master)

    def refresh(self, param_id: str, master: Any) -> torch.Tensor:
        """将单个 FP32 master 转换为 shadow dtype/device 并更新缓存。

        返回 shadow tensor（形状与 master 相同，dtype/device 已转换）。
        """
        _require_non_empty_string("param_id", param_id)
        # FP32 master → shadow dtype + device
        shadow = _as_cpu_float32_vector(master, name="master").to(
            device=self.device,
            dtype=self.dtype,
        )
        self.shadows[param_id] = shadow.contiguous()
        return self.shadows[param_id]

    def get(self, param_id: str) -> torch.Tensor:
        """获取 shadow tensor。不存在则报 KeyError。"""
        try:
            return self.shadows[param_id]
        except KeyError as exc:
            raise KeyError(f"unknown forward shadow {param_id!r}") from exc


# ──────────────────── 更新摘要 ────────────────────

@dataclass(frozen=True, slots=True)
class HotParamWindowUpdateSummary:
    """单次 apply_buckets 调用的更新摘要。"""
    touched_param_ids: tuple[str, ...]   # 本轮更新的参数列表
    grad_norms: dict[str, float]         # {param_id: grad_norm}
    update_norms: dict[str, float]       # {param_id: update_norm}
    drained_bucket_ids: tuple[int, ...]  # 消费的 bucket ID 列表


class GptqUpdateBuilder(Protocol):
    """GPTQ 重量化接口：将 FP32 master 重新量化为 Int4 字节。"""
    def requantize_touched(
        self,
        masters: Mapping[str, Any],
        touched_param_ids: Iterable[str],
    ) -> dict[str, bytes]:
        ...


# ──────────────────── HotParamTrainingWindow ────────────────────

@dataclass(slots=True)
class HotParamTrainingWindow:
    """训练窗口核心——设计文档 Section 4.2 + Section 10。

    管理当前 hot set 的完整生命周期:
      1. load_from_stores: 从 NVMe 加载 FP32 master + Adam 状态 → CPU
      2. add_gradient + apply_buckets: GPU 梯度 → CPU Adam → shadow refresh
      3. make_commit_payload: 窗口结束 → 构建提交数据
      4. switch_hot_params: 切换 hot set（触发窗口提交）

    窗口 = hot set 切换边界（设计文档 Section 13.1）。
    """
    fp32_store: FP32ShardStore          # NVMe FP32 存储引用
    adam_store: CpuAdamFp8StateStore     # NVMe Adam 状态存储引用
    updater: CpuAdamFp8Updater           # CPU Adam 更新器
    shadow_store: ForwardShadowStore     # GPU shadow 管理
    bucket_ring: GradientBucketRing      # 梯度 bucket 环
    masters: dict[str, torch.Tensor] = field(default_factory=dict)    # CPU FP32 hot master: {param_id: [numel] fp32}
    adam_states: dict[str, dict[str, bytes]] = field(default_factory=dict)  # CPU Adam FP8 状态: {param_id: {"m":bytes, "v":bytes}}
    touched_param_ids: list[str] = field(default_factory=list)      # 本窗口内被更新过的参数（用于窗口提交）
    grad_norms: dict[str, float] = field(default_factory=dict)
    update_norms: dict[str, float] = field(default_factory=dict)

    @classmethod
    def load_from_stores(
        cls,
        *,
        fp32_store: FP32ShardStore,
        adam_store: CpuAdamFp8StateStore,
        updater: CpuAdamFp8Updater,
        hot_param_ids: tuple[str, ...],
        bucket_capacity_bytes: int,
        shadow_dtype: torch.dtype = torch.float16,
        shadow_device: torch.device | str = "cpu",
        max_sealed_buckets: int = 4,
    ) -> "HotParamTrainingWindow":
        """从 NVMe stores 加载 hot set 的 FP32 master + Adam 状态。

        对每个 hot param:
          - 从 FP32ShardStore 读取 master → CPU float32 tensor
          - 从 AdamFp8StateStore 读取 m/v 状态 → bytes（不存在则 zero_payload）
          - 创建 ForwardShadowStore 并填充初始 shadow
        """
        masters: dict[str, torch.Tensor] = {}
        adam_states: dict[str, dict[str, bytes]] = {}
        for param_id in hot_param_ids:
            _require_non_empty_string("param_id", param_id)
            record = fp32_store.records[param_id]
            # 从 NVMe 读取 FP32 master → CPU tensor [num_elements] float32
            masters[param_id] = _float32_tensor_from_bytes(
                fp32_store.read_param(param_id)
            )
            # 从 NVMe 读取 Adam FP8 m/v 状态 → bytes
            adam_states[param_id] = {
                "m": _read_state_or_zero(
                    adam_store, updater.codec, param_id, "m", record.num_elements,
                ),
                "v": _read_state_or_zero(
                    adam_store, updater.codec, param_id, "v", record.num_elements,
                ),
            }

        shadow_store = ForwardShadowStore(dtype=shadow_dtype, device=shadow_device)
        # 从 masters 批量创建 shadow
        shadow_store.load_from_masters(masters)
        return cls(
            fp32_store=fp32_store,
            adam_store=adam_store,
            updater=updater,
            shadow_store=shadow_store,
            bucket_ring=GradientBucketRing(
                bucket_capacity_bytes=bucket_capacity_bytes,
                max_sealed_buckets=max_sealed_buckets,
            ),
            masters=masters,
            adam_states=adam_states,
        )

    @property
    def hot_param_ids(self) -> tuple[str, ...]:
        return tuple(self.masters)

    def switch_hot_params(self, hot_param_ids: tuple[str, ...]) -> None:
        """切换 hot set（仅当 bucket ring 已清空时可调用）。

        调用前必须已提交当前窗口（commit → mark_committed），
        然后从 NVMe 加载新 hot set 的 master + Adam 状态。
        """
        if not self.bucket_ring.is_empty:
            raise RuntimeError("cannot switch hot params with pending gradients")
        # 从 NVMe 加载新 hot set
        masters, adam_states = self._load_hot_param_state(hot_param_ids)
        # 替换 CPU 状态
        self.masters = masters
        self.adam_states = adam_states
        # 清空旧 shadow，从新 master 重建
        self.shadow_store.shadows.clear()
        self.shadow_store.load_from_masters(masters)
        # 清空 touched 追踪（旧窗口已提交）
        self.mark_committed()

    def mark_committed(self) -> None:
        """窗口提交后清空 touched 追踪。"""
        self.touched_param_ids.clear()
        self.grad_norms.clear()
        self.update_norms.clear()

    def add_gradient(
        self,
        param_id: str,
        grad: Any,
    ) -> tuple[GradientBucket, ...]:
        """将梯度加入 bucket 环。param 必须在 hot set 中。"""
        self._require_hot_param(param_id)
        return self.bucket_ring.add_gradient(param_id, grad)

    def drain_ready(self, *, optimizer_step: int) -> HotParamWindowUpdateSummary:
        """drain 已 seal 的 bucket 并执行 Adam 更新。"""
        return self.apply_buckets(
            self.bucket_ring.drain_ready(),
            optimizer_step=optimizer_step,
        )

    def drain_all(self, *, optimizer_step: int) -> HotParamWindowUpdateSummary:
        """seal + drain 所有 bucket（窗口提交前使用）。"""
        return self.apply_buckets(
            self.bucket_ring.drain_all(),
            optimizer_step=optimizer_step,
        )

    # ------------------------- bucket 消费 + Adam 更新 -------------------------
    def apply_buckets(
        self,
        buckets: tuple[GradientBucket, ...],
        *,
        optimizer_step: int,
    ) -> HotParamWindowUpdateSummary:
        """消费已 seal 的 bucket：对每个梯度执行 CPU Adam 更新 → 刷新 shadow。

        对每个 bucket 中的每个 param:
          1. 从 CPU 读取 master + Adam m/v
          2. Adam step: m,v 更新 → 计算 update → master -= lr * update
          3. 写回 master + Adam m/v
          4. shadow_store.refresh(): 更新 GPU shadow
          5. 记录 touched 供窗口提交使用
        """
        _require_positive_int("optimizer_step", optimizer_step)
        drained_bucket_ids: list[int] = []
        touched_this_call: list[str] = []

        for bucket in buckets:
            drained_bucket_ids.append(bucket.bucket_id)
            for param_id, grad in bucket.grads.items():
                self._require_hot_param(param_id)
                state = self.adam_states[param_id]
                # CPU Adam 更新: master, m, v ← grad
                update = self.updater.step_param(
                    param_id=param_id,
                    master=self.masters[param_id],
                    grad=grad,
                    step=optimizer_step,
                    first_moment_state=state["m"],
                    second_moment_state=state["v"],
                )
                # 写回更新后的 master + Adam 状态
                self.masters[param_id] = update.master
                self.adam_states[param_id] = update.adam_updates
                # 刷新 GPU shadow（下一次 forward 使用最新权重）
                self.shadow_store.refresh(param_id, update.master)
                self.grad_norms[param_id] = update.grad_norm
                self.update_norms[param_id] = update.update_norm
                # 追踪本窗口更新的参数
                if param_id not in self.touched_param_ids:
                    self.touched_param_ids.append(param_id)
                if param_id not in touched_this_call:
                    touched_this_call.append(param_id)

        return HotParamWindowUpdateSummary(
            touched_param_ids=tuple(touched_this_call),
            grad_norms={
                param_id: self.grad_norms[param_id]
                for param_id in touched_this_call
            },
            update_norms={
                param_id: self.update_norms[param_id]
                for param_id in touched_this_call
            },
            drained_bucket_ids=tuple(drained_bucket_ids),
        )

    # ------------------------- 窗口提交 -------------------------
    def make_commit_payload(
        self,
        *,
        global_step: int,
        epoch: int,
        dataset_cursor: str,
        consumed_samples: int = 0,
        consumed_tokens: int = 0,
        gptq_updates: Mapping[str, Any] | None = None,
        gptq_update_builder: GptqUpdateBuilder | None = None,
    ) -> WindowCommitPayload:
        """构建窗口提交数据: FP32 masters + Adam states + GPTQ updates。

        gptq_updates 和 gptq_update_builder 二选一:
          - gptq_updates: 预构建的 GPTQ 字节
          - gptq_update_builder: 现场从 masters 重量化（调用 requantize_touched）
        """
        touched = tuple(self.touched_param_ids)
        if gptq_updates is not None and gptq_update_builder is not None:
            raise ValueError(
                "pass either gptq_updates or gptq_update_builder, not both"
            )
        # 如果没有预构建 gptq_updates，则从 builder 实地重量化
        resolved_gptq_updates = (
            gptq_update_builder.requantize_touched(self.masters, touched)
            if gptq_update_builder is not None
            else gptq_updates
        )
        return WindowCommitPayload(
            fp32_updates={param_id: self.masters[param_id] for param_id in touched},
            adam_updates={param_id: self.adam_states[param_id] for param_id in touched},
            gptq_updates=resolved_gptq_updates,
            global_step=global_step,
            epoch=epoch,
            dataset_cursor=dataset_cursor,
            touched_param_ids=touched,
            consumed_samples=consumed_samples,
            consumed_tokens=consumed_tokens,
        )

    # ------------------------- 内部辅助 -------------------------
    def _require_hot_param(self, param_id: str) -> None:
        _require_non_empty_string("param_id", param_id)
        if param_id not in self.masters:
            raise KeyError(f"param {param_id!r} is not in the hot set")

    def _load_hot_param_state(
        self,
        hot_param_ids: tuple[str, ...],
    ) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, bytes]]]:
        """纯函数：从 NVMe stores 读取给定 hot set 的 master + Adam 状态。"""
        masters: dict[str, torch.Tensor] = {}
        adam_states: dict[str, dict[str, bytes]] = {}
        for param_id in hot_param_ids:
            _require_non_empty_string("param_id", param_id)
            record = self.fp32_store.records[param_id]
            masters[param_id] = _float32_tensor_from_bytes(
                self.fp32_store.read_param(param_id)
            )
            adam_states[param_id] = {
                "m": _read_state_or_zero(
                    self.adam_store, self.updater.codec,
                    param_id, "m", record.num_elements,
                ),
                "v": _read_state_or_zero(
                    self.adam_store, self.updater.codec,
                    param_id, "v", record.num_elements,
                ),
            }
        return masters, adam_states


# ──────────────────── 模块级工具函数 ────────────────────

def _read_state_or_zero(
    store: CpuAdamFp8StateStore,
    codec: BlockFp8StateCodec,
    param_id: str,
    component: str,
    num_elements: int,
) -> bytes:
    """从 Adam store 读取 FP8 状态；不存在则返回 zero_payload。"""
    try:
        return store.read_state(param_id, component)
    except (FileNotFoundError, KeyError):
        return codec.zero_payload(num_elements)


def _float32_tensor_from_bytes(payload: bytes | bytearray | memoryview) -> torch.Tensor:
    """bytes → flat CPU float32 tensor。"""
    tensor = torch.frombuffer(bytearray(payload), dtype=torch.float32)
    return tensor.clone().contiguous()


def _as_cpu_float32_vector(value: Any, *, name: str) -> torch.Tensor:
    """将任意输入（tensor/bytes/bytearray）统一转为 flat CPU float32 向量。"""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _float32_tensor_from_bytes(value)
    if not hasattr(value, "detach"):
        raise TypeError(f"{name} must be a torch.Tensor or FP32 bytes")
    tensor = value.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.to(dtype=torch.float32).reshape(-1).contiguous()
