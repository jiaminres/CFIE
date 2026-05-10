"""CPU AdamW 优化器——FP8 block-quantized 状态（设计文档 Section 10.4）。

AdamW 公式: m=β₁·m+(1-β₁)·g, v=β₂·v+(1-β₂)·g²
             m̂=m/(1-β₁^t), v̂=v/(1-β₂^t)
             u=m̂/(√v̂+eps)+λ·w,  w=w-lr·u

FP8 block quant 编码格式: [num_elements 个 uint8(e4m3)] + [num_blocks×4 bytes float32 scales]
解码: values[e4m3→f32] × scales.repeat_interleave(block_size)[:num_elements]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from cfie_training.training_base.adam_state_store import CpuAdamFp8StateStore
from cfie_training.training_base.fp32_shard_store import FP32ShardStore

# FP8 E4M3 格式的最大可表示值
FLOAT8_E4M3_MAX = 448.0
# Adam 状态 block quantization 的默认块大小
DEFAULT_ADAM_BLOCK_SIZE = 128


# ──────────────────── 参数校验辅助 ────────────────────

def _require_non_negative_int(name: str, value: int) -> None:
    if value < 0: raise ValueError(f"{name} must be >= 0")

def _require_positive_int(name: str, value: int) -> None:
    if value < 1: raise ValueError(f"{name} must be >= 1")

def _require_non_negative_float(name: str, value: float) -> None:
    if value < 0: raise ValueError(f"{name} must be >= 0")

def _require_positive_float(name: str, value: float) -> None:
    if value <= 0: raise ValueError(f"{name} must be > 0")


# ──────────────── BlockFp8StateCodec — FP8 编码/解码 ────────────────

@dataclass(frozen=True, slots=True)
class BlockFp8StateCodec:
    """FP8 block-quantized 编解码器。

    格式: 每 block_size 个 float32 值 → 1 个 float32 scale + block_size 个 e4m3 值。
    payload 布局: [quantized(e4m3→uint8), N] + [scales(float32), num_blocks]
    """
    block_size: int = DEFAULT_ADAM_BLOCK_SIZE  # 每个 scale 覆盖的元素数
    min_scale: float = 1e-12                    # scale 下限，避免除零

    def __post_init__(self) -> None:
        _require_positive_int("block_size", self.block_size)
        _require_positive_float("min_scale", self.min_scale)

    # ── 容量计算 ──

    def encoded_num_bytes(self, num_elements: int) -> int:
        """返回 num_elements 个 float32 的 FP8 编码后字节数。

        = num_elements(uint8) + num_blocks × 4(float32 scale)
        """
        _require_non_negative_int("num_elements", num_elements)
        if num_elements == 0:
            return 0  # 空张量 → 空 payload
        # block 数: 向上取整 num_elements / block_size
        num_blocks = self.num_blocks(num_elements)
        return num_elements + num_blocks * 4

    def num_blocks(self, num_elements: int) -> int:
        """计算给定元素数需要多少个 block（向上取整）。"""
        _require_non_negative_int("num_elements", num_elements)
        if num_elements == 0: return 0
        return (num_elements + self.block_size - 1) // self.block_size

    # ── 零初始化 ──

    def zero_payload(self, num_elements: int) -> bytes:
        """生成全零 Adam 状态的 FP8 payload。

        用于首次初始化 Adam m/v（训练开始时 m=v=0）。
        quantized: uint8 全零 → e4m3 值为 0
        scales: float32 全 1.0 → 解码后值为 0×1.0=0
        """
        _require_non_negative_int("num_elements", num_elements)
        if num_elements == 0: return b""
        # 量化值全零: [num_elements] uint8
        quantized = torch.zeros(num_elements, dtype=torch.uint8)
        # scale 全 1.0: [num_blocks] float32
        scales = torch.ones(self.num_blocks(num_elements), dtype=torch.float32)
        return quantized.numpy().tobytes() + scales.numpy().tobytes()

    # ── 编码: float32 → FP8 ──

    def encode(self, tensor: Any) -> bytes:
        """将 FP32 tensor 编码为 FP8 block-quantized bytes。

        逐 block 处理: 取 absmax → scale = max_abs/F8_MAX → 量化 block/scale → 存 uint8(e4m3)
        """
        # 输入统一转为 flat CPU float32 向量
        state_tensor = _as_cpu_float32_vector(tensor, name="tensor")
        if not torch.isfinite(state_tensor).all():
            raise ValueError("tensor must contain only finite values")
        num_elements = state_tensor.numel()
        if num_elements == 0: return b""

        # 分配输出: [num_elements] uint8 量化值 + [num_blocks] float32 scales
        quantized = torch.empty(num_elements, dtype=torch.uint8)
        scales = torch.empty(self.num_blocks(num_elements), dtype=torch.float32)

        # 逐 block 编码
        for block_index, start in enumerate(range(0, num_elements, self.block_size)):
            end = min(start + self.block_size, num_elements)
            block = state_tensor[start:end]              # [block_size] float32
            max_abs = float(block.abs().max().item())    # block 内绝对值最大值
            # scale: max_abs/F8_MAX，不低于 min_scale
            scale = 1.0 if max_abs == 0 else max(max_abs / FLOAT8_E4M3_MAX, self.min_scale)
            scales[block_index] = scale
            # 量化: round(block/scale) → clamp → e4m3 → uint8 内存视图
            quantized[start:end] = (
                (block / scale)
                .clamp(min=-FLOAT8_E4M3_MAX, max=FLOAT8_E4M3_MAX)
                .to(torch.float8_e4m3fn)     # FP8 e4m3 数据类型
                .view(torch.uint8)            # 按位视为 uint8 存储
            )
        return quantized.numpy().tobytes() + scales.numpy().tobytes()

    # ── 解码: FP8 → float32 ──

    def decode(self, payload: bytes | bytearray | memoryview, num_elements: int) -> torch.Tensor:
        """将 FP8 block-quantized payload 解码为 float32 tensor。

        payload 布局: [quantized: num_elements(uint8)] + [scales: num_blocks×4(float32)]
        """
        _require_non_negative_int("num_elements", num_elements)
        payload_bytes = bytes(payload)
        expected_bytes = self.encoded_num_bytes(num_elements)
        if len(payload_bytes) != expected_bytes:
            raise ValueError(f"FP8 state expected {expected_bytes} bytes, got {len(payload_bytes)}")
        if num_elements == 0: return torch.empty(0, dtype=torch.float32)

        # 分离量化值和 scales
        quantized_payload = bytearray(payload_bytes[:num_elements])     # [num_elements] uint8
        scales_payload = bytearray(payload_bytes[num_elements:])        # [num_blocks×4] float32
        quantized = torch.frombuffer(quantized_payload, dtype=torch.uint8)       # [N] uint8
        scales = torch.frombuffer(scales_payload, dtype=torch.float32).clone()   # [B] float32

        # uint8 → e4m3 → float32
        values = quantized.view(torch.float8_e4m3fn).to(torch.float32)  # [N] float32
        # 扩展 scales 到每个元素: [N] = repeat_interleave(block_size)[:N]
        expanded_scales = scales.repeat_interleave(self.block_size)[:num_elements]
        return (values * expanded_scales).contiguous()


# ──────────────── AdamWConfig — 优化器超参数 ────────────────

@dataclass(frozen=True, slots=True)
class AdamWConfig:
    """AdamW 超参数。"""
    lr: float                        # 学习率
    beta1: float = 0.9               # 一阶动量衰减
    beta2: float = 0.999             # 二阶动量衰减
    eps: float = 1e-8                # 数值稳定项
    weight_decay: float = 0.0        # 权重衰减（L2 正则化）
    bias_correction: bool = True     # 是否启用偏差校正

    def __post_init__(self) -> None:
        _require_positive_float("lr", self.lr)
        _require_non_negative_float("beta1", self.beta1)
        _require_non_negative_float("beta2", self.beta2)
        if self.beta1 >= 1 or self.beta2 >= 1:
            raise ValueError("beta1 and beta2 must be < 1")
        _require_positive_float("eps", self.eps)
        _require_non_negative_float("weight_decay", self.weight_decay)


# ──────────────── CpuAdamParamUpdate — 单参数更新结果 ────────────────

@dataclass(frozen=True, slots=True)
class CpuAdamParamUpdate:
    """单个参数的 Adam 更新结果。"""
    param_id: str                    # 参数 ID
    master: torch.Tensor             # 更新后的 FP32 master 权重 [numel]
    first_moment: bytes              # 更新后的 m，FP8 编码
    second_moment: bytes             # 更新后的 v，FP8 编码
    grad_norm: float                 # 梯度的 L2 范数
    update_norm: float               # 更新量 u 的 L2 范数

    @property
    def adam_updates(self) -> dict[str, bytes]:
        """返回 {m: bytes, v: bytes} 用于写入 Adam store。"""
        return {"m": self.first_moment, "v": self.second_moment}


# ──────────────── CpuAdamWindowUpdate — 批量更新结果 ────────────────

@dataclass(frozen=True, slots=True)
class CpuAdamWindowUpdate:
    """一个窗口内所有参数的批量 Adam 更新结果。"""
    fp32_updates: dict[str, torch.Tensor]       # {param_id: 更新后 master}
    adam_updates: dict[str, dict[str, bytes]]   # {param_id: {m: bytes, v: bytes}}
    touched_param_ids: tuple[str, ...]           # 受影响参数列表
    grad_norms: dict[str, float]
    update_norms: dict[str, float]


# ──────────────────── CpuAdamFp8Updater — CPU Adam 更新器 ────────────────────

@dataclass(slots=True)
class CpuAdamFp8Updater:
    """CPU 端 AdamW 更新器——对单个参数执行一步 Adam 更新。

    输入: master(FP32), grad(FP32), step, m/v state(FP8 bytes)
    输出: 更新后的 master + m/v state
    """
    config: AdamWConfig                              # Adam 超参数
    codec: BlockFp8StateCodec = field(default_factory=BlockFp8StateCodec)  # FP8 编解码器

    # ── 单步更新 ──

    def step_param(
        self, *, param_id: str, master: Any, grad: Any, step: int,
        first_moment_state: bytes | bytearray | memoryview | None = None,
        second_moment_state: bytes | bytearray | memoryview | None = None,
    ) -> CpuAdamParamUpdate:
        """对单个参数执行一步完整的 AdamW 更新。"""
        if not param_id.strip():
            raise ValueError("param_id must be a non-empty string")
        _require_positive_int("step", step)

        # 将输入统一为 flat CPU float32 向量 [numel]
        master_tensor = _as_cpu_float32_vector(master, name="master")
        grad_tensor = _as_cpu_float32_vector(grad, name="grad")
        if master_tensor.numel() != grad_tensor.numel():
            raise ValueError("master and grad must have the same number of elements")
        if not torch.isfinite(master_tensor).all():
            raise ValueError("master must contain only finite values")
        if not torch.isfinite(grad_tensor).all():
            raise ValueError("grad must contain only finite values")

        num_elements = master_tensor.numel()
        # 解码 FP8 m/v state → float32 [numel]（不存在则初始化为零）
        first_moment = self._decode_or_zero(first_moment_state, num_elements)
        second_moment = self._decode_or_zero(second_moment_state, num_elements)

        # AdamW 更新公式
        beta1 = self.config.beta1; beta2 = self.config.beta2
        # m = β₁·m + (1-β₁)·g  —— 一阶动量 EMA 更新
        first_moment.mul_(beta1).add_(grad_tensor, alpha=1 - beta1)
        # v = β₂·v + (1-β₂)·g² —— 二阶动量 EMA 更新
        second_moment.mul_(beta2).addcmul_(grad_tensor, grad_tensor, value=1 - beta2)

        # 偏差校正（训练初期 m,v 偏小）
        if self.config.bias_correction:
            first_moment_hat = first_moment / (1 - beta1**step)       # m̂ = m / (1-β₁^t)
            second_moment_hat = second_moment / (1 - beta2**step)     # v̂ = v / (1-β₂^t)
        else:
            first_moment_hat = first_moment; second_moment_hat = second_moment

        # u = m̂ / (√v̂ + eps) —— 自适应学习率方向
        update = first_moment_hat / (second_moment_hat.sqrt() + self.config.eps)
        # w_new = w - lr·u  （先 weight decay 再减去更新）
        updated_master = master_tensor.clone()
        if self.config.weight_decay:
            # w = w·(1 - lr·λ) —— decoupled weight decay
            updated_master.mul_(1 - self.config.lr * self.config.weight_decay)
        updated_master.add_(update, alpha=-self.config.lr)            # w -= lr·u

        return CpuAdamParamUpdate(
            param_id=param_id,
            master=updated_master,
            first_moment=self.codec.encode(first_moment),              # m → FP8 bytes
            second_moment=self.codec.encode(second_moment),            # v → FP8 bytes
            grad_norm=float(grad_tensor.norm().item()),                # ‖g‖₂
            update_norm=float(update.norm().item()),                   # ‖u‖₂
        )

    # ── 批量从 stores 更新 ──

    def apply_gradients_from_stores(
        self, *, fp32_store: FP32ShardStore, adam_store: CpuAdamFp8StateStore,
        grads: Mapping[str, Any], step: int,
    ) -> CpuAdamWindowUpdate:
        """从 NVMe stores 读取 master + m/v → 执行 Adam → 返回更新后的数据。

        用于从 FP32ShardStore + AdamStore 直接批量更新。
        """
        _require_positive_int("step", step)
        fp32_updates: dict[str, torch.Tensor] = {}
        adam_updates: dict[str, dict[str, bytes]] = {}
        grad_norms: dict[str, float] = {}
        update_norms: dict[str, float] = {}

        for param_id, grad in grads.items():
            # 从 NVMe 读取 FP32 master → flat float32 tensor
            record = fp32_store.records[param_id]
            master = _float32_tensor_from_bytes(fp32_store.read_param(param_id))
            # 从 NVMe 读取 Adam m/v state → bytes（不存在则 zero_payload）
            first_moment_state = _read_state_or_zero(adam_store, self.codec, param_id, "m", record.num_elements)
            second_moment_state = _read_state_or_zero(adam_store, self.codec, param_id, "v", record.num_elements)
            # 执行 Adam 更新
            update = self.step_param(
                param_id=param_id, master=master, grad=grad, step=step,
                first_moment_state=first_moment_state, second_moment_state=second_moment_state,
            )
            fp32_updates[param_id] = update.master
            adam_updates[param_id] = update.adam_updates
            grad_norms[param_id] = update.grad_norm
            update_norms[param_id] = update.update_norm

        return CpuAdamWindowUpdate(
            fp32_updates=fp32_updates, adam_updates=adam_updates,
            touched_param_ids=tuple(grads), grad_norms=grad_norms, update_norms=update_norms,
        )

    # ── 辅助 ──

    def _decode_or_zero(
        self, payload: bytes | bytearray | memoryview | None, num_elements: int,
    ) -> torch.Tensor:
        """解码 FP8 state → float32 [num_elements]。无 payload 则返回零向量。"""
        if payload is None: return torch.zeros(num_elements, dtype=torch.float32)
        return self.codec.decode(payload, num_elements)


# ──────────────────── 模块级工具函数 ────────────────────

def adam_state_num_bytes(num_elements: int, *, block_size: int = DEFAULT_ADAM_BLOCK_SIZE) -> int:
    """计算 num_elements 个参数的 Adam FP8 state 所需字节数。"""
    return BlockFp8StateCodec(block_size=block_size).encoded_num_bytes(num_elements)


def _read_state_or_zero(
    store: CpuAdamFp8StateStore, codec: BlockFp8StateCodec,
    param_id: str, component: str, num_elements: int,
) -> bytes:
    """从 Adam store 读取 FP8 状态；不存在则返回 zero_payload（全零）。"""
    try: return store.read_state(param_id, component)
    except (FileNotFoundError, KeyError): return codec.zero_payload(num_elements)


def _float32_tensor_from_bytes(payload: bytes | bytearray | memoryview) -> torch.Tensor:
    """bytes → flat CPU float32 tensor [num_bytes/4]。"""
    tensor = torch.frombuffer(bytearray(payload), dtype=torch.float32)
    return tensor.clone().contiguous()


def _as_cpu_float32_vector(value: Any, *, name: str) -> torch.Tensor:
    """将任意输入（tensor/bytes/bytearray）统一转为 flat CPU float32 向量 [N]。

    支持 CUDA tensor（自动 detach + cpu）、FP32 bytes、bytearray。
    """
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _float32_tensor_from_bytes(value)            # bytes → float32
    if not hasattr(value, "detach"):
        raise TypeError(f"{name} must be a torch.Tensor or FP32 bytes")
    tensor = value.detach()                                  # 脱离 autograd
    if tensor.is_cuda: tensor = tensor.cpu()                 # GPU → CPU
    return tensor.to(dtype=torch.float32).reshape(-1).contiguous()  # → flat FP32
