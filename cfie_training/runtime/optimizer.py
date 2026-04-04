"""CPU optimizer/update skeleton for the CFIE training runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from typing import TYPE_CHECKING

import torch

from cfie_training.config import TrainingProjectConfig
from cfie_training.runtime.source import LocalWeightManifest
from cfie_training.runtime.store import ParameterShardStore
from cfie_training.runtime.types import (
    OptimizerShardStateSnapshot,
    OptimizerSummary,
    OptimizerUpdateRecord,
    ParameterShardSnapshot,
)

_TRAINABLE_COMPONENTS = frozenset({"bucket_non_routed", "bucket_active_experts"})
_MAX_REPRESENTATIVE_PARAMS = 128

if TYPE_CHECKING:
    from cfie_training.runtime.executor import GradientPayload


def _default_optimizer_state_root(config: TrainingProjectConfig) -> Path:
    # 为当前 profile 创建一个独立的临时优化器状态目录。
    return Path(
        tempfile.mkdtemp(
            prefix=f"cfie_training_optimizer_state_{config.profile_name}_",
        )
    )


def _stable_seed(label: str) -> int:
    # 先把稳定标签做 SHA1 哈希。
    digest = hashlib.sha1(label.encode("utf-8")).digest()
    # 取前 8 字节转成无符号整数，作为可复现种子。
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _zero_buffer(
    size: int,
    *,
    as_tensor: bool = False,
) -> tuple[float, ...] | torch.Tensor:
    # 需要 tensor 形态时直接构造 CPU float32 零张量。
    if as_tensor:
        return torch.zeros(size, dtype=torch.float32, device="cpu")
    # 否则退回 tuple 形式的零缓冲区。
    return tuple(0.0 for _ in range(size))


def _buffer_is_empty(buffer: tuple[float, ...] | torch.Tensor) -> bool:
    # tensor 缓冲区按元素数是否为 0 判断。
    if isinstance(buffer, torch.Tensor):
        return buffer.numel() == 0
    # tuple 缓冲区按长度是否为 0 判断。
    return len(buffer) == 0


def _stabilize_update_tensor(
    tensor: torch.Tensor,
    *,
    max_abs: float = 32.0,
) -> torch.Tensor:
    # 统一把输入转成 CPU float32，并裁掉 NaN/Inf 与过大数值。
    return torch.nan_to_num(
        tensor.detach().to(dtype=torch.float32, device="cpu"),
        nan=0.0,
        posinf=max_abs,
        neginf=-max_abs,
    ).clamp(min=-max_abs, max=max_abs)


def _storage_dtype(dtype_name: str) -> torch.dtype:
    # 建立配置字符串到 torch dtype 的映射。
    mapping = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp8_e4m3fn": torch.float8_e4m3fn,
        "fp8_e5m2": torch.float8_e5m2,
    }
    try:
        # 命中时返回对应的 torch dtype。
        return mapping[dtype_name]
    except KeyError as exc:
        # 未知 dtype 配置直接报错。
        raise ValueError(f"unsupported CPU storage dtype: {dtype_name}") from exc


@dataclass(slots=True)
class _OptimizerShardState:
    group_id: str
    component: str
    logical_params: int
    representative_params: int
    update_count: int = 0
    last_committed_version: int = 0
    last_updated_step: int = -1
    state_tier: str = "nvme_cold"
    exp_avg_buffer: tuple[float, ...] | torch.Tensor = ()
    exp_avg_sq_buffer: tuple[float, ...] | torch.Tensor = ()
    exp_avg_tensor: torch.Tensor | None = None
    exp_avg_sq_tensor: torch.Tensor | None = None
    sparse_exp_avg: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict)
    sparse_exp_avg_sq: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict)

    def to_snapshot(self) -> OptimizerShardStateSnapshot:
        # 默认按“带值快照”导出当前优化器分片状态。
        return self.snapshot_with_values()

    def snapshot_with_values(
        self,
        *,
        include_values: bool = True,
    ) -> OptimizerShardStateSnapshot:
        # 根据 include_values 决定是否把 moment 值写进快照。
        if include_values:
            # 默认先从 buffer 视图读取 exp_avg / exp_avg_sq。
            exp_avg_values = self.exp_avg_buffer
            exp_avg_sq_values = self.exp_avg_sq_buffer
            # 若当前已有物化 tensor，则以 tensor 值为准。
            if self.exp_avg_tensor is not None:
                exp_avg_values = tuple(
                    float(value) for value in self.exp_avg_tensor.tolist()
                )
            elif isinstance(self.exp_avg_buffer, torch.Tensor):
                # tensor buffer 也统一转成 float tuple 便于序列化。
                exp_avg_values = tuple(
                    float(value)
                    for value in self.exp_avg_buffer.to(dtype=torch.float32).tolist()
                )
            # exp_avg_sq 同理优先取物化 tensor。
            if self.exp_avg_sq_tensor is not None:
                exp_avg_sq_values = tuple(
                    float(value) for value in self.exp_avg_sq_tensor.tolist()
                )
            elif isinstance(self.exp_avg_sq_buffer, torch.Tensor):
                # tensor buffer 也统一转成 float tuple 便于序列化。
                exp_avg_sq_values = tuple(
                    float(value)
                    for value in self.exp_avg_sq_buffer.to(dtype=torch.float32).tolist()
                )
        else:
            # 不带值快照时，两个状态向量都写空。
            exp_avg_values = ()
            exp_avg_sq_values = ()
        # 把内部状态导出成可持久化的分片快照。
        return OptimizerShardStateSnapshot(
            group_id=self.group_id,
            component=self.component,
            logical_params=self.logical_params,
            representative_params=self.representative_params,
            update_count=self.update_count,
            last_committed_version=self.last_committed_version,
            last_updated_step=self.last_updated_step,
            state_tier=self.state_tier,
            parameter_values=(),
            exp_avg_values=exp_avg_values,
            exp_avg_sq_values=exp_avg_sq_values,
        )


class _OptimizerStateMirror:
    def __init__(self, config: TrainingProjectConfig) -> None:
        # 仅在“逻辑物化 + 更新后 offload”场景下启用优化器状态镜像。
        self._enabled = (
            config.optimizer.offload_state_after_update
            and config.execution.trainable_shard_materialization == "logical"
        )
        # 优先复用量化 staging 目录作为 NVMe 状态根目录。
        if config.runtime_quantization.nvme_staging_dir:
            base_root = Path(config.runtime_quantization.nvme_staging_dir).expanduser()
            self._root = base_root / config.runtime_quantization.session_id / "optimizer_state"
        else:
            # 未配置 staging 目录时退回临时目录。
            self._root = _default_optimizer_state_root(config).expanduser()

    @property
    def enabled(self) -> bool:
        # 对外暴露当前镜像功能是否启用。
        return self._enabled

    def _path_for_group(self, group_id: str) -> Path:
        # 用 group_id 哈希生成稳定的 dense 状态文件路径。
        digest = hashlib.sha1(group_id.encode("utf-8")).hexdigest()
        return self._root / f"{digest}.pt"

    def _sparse_root_for_group(self, group_id: str) -> Path:
        # 稀疏状态按 group 单独放到一个目录下。
        digest = hashlib.sha1(group_id.encode("utf-8")).hexdigest()
        return self._root / f"{digest}.sparse"

    def _sparse_path_for_slice(
        self,
        group_id: str,
        *,
        start_offset: int,
        size: int,
    ) -> Path:
        # 稀疏切片文件名由起始偏移和切片大小唯一确定。
        return self._sparse_root_for_group(group_id) / f"{start_offset}_{size}.pt"

    def _clear_sparse_group(self, group_id: str) -> None:
        # 切换回 dense 保存前，先删掉该 group 遗留的稀疏目录。
        sparse_root = self._sparse_root_for_group(group_id)
        if sparse_root.exists():
            shutil.rmtree(sparse_root)

    def load(self, group_id: str) -> tuple[torch.Tensor, torch.Tensor] | None:
        # 未启用镜像时直接返回 None。
        if not self._enabled:
            return None
        # 定位当前 group 的 dense 状态文件。
        path = self._path_for_group(group_id)
        # 文件不存在时说明当前 group 还没有镜像状态。
        if not path.exists():
            return None
        # 从磁盘把状态文件加载到 CPU。
        payload = torch.load(path, map_location="cpu")
        # 取出一阶 / 二阶矩。
        exp_avg = payload.get("exp_avg")
        exp_avg_sq = payload.get("exp_avg_sq")
        # 结构不符合预期时按不可用处理。
        if not isinstance(exp_avg, torch.Tensor) or not isinstance(exp_avg_sq, torch.Tensor):
            return None
        # 统一返回展平后的 CPU float32 张量。
        return (
            exp_avg.to(dtype=torch.float32, device="cpu").reshape(-1).contiguous(),
            exp_avg_sq.to(dtype=torch.float32, device="cpu").reshape(-1).contiguous(),
        )

    def load_sparse_slice(
        self,
        group_id: str,
        *,
        start_offset: int,
        size: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        # 未启用镜像时直接返回 None。
        if not self._enabled:
            return None
        # 定位当前切片对应的稀疏状态文件。
        path = self._sparse_path_for_slice(
            group_id,
            start_offset=start_offset,
            size=size,
        )
        # 文件不存在时，说明这段稀疏状态尚未落盘。
        if not path.exists():
            return None
        # 从磁盘加载这段稀疏状态。
        payload = torch.load(path, map_location="cpu")
        exp_avg = payload.get("exp_avg")
        exp_avg_sq = payload.get("exp_avg_sq")
        # 结构不符合预期时按不可用处理。
        if not isinstance(exp_avg, torch.Tensor) or not isinstance(exp_avg_sq, torch.Tensor):
            return None
        # 统一返回展平后的 CPU float32 张量。
        return (
            exp_avg.to(dtype=torch.float32, device="cpu").reshape(-1).contiguous(),
            exp_avg_sq.to(dtype=torch.float32, device="cpu").reshape(-1).contiguous(),
        )

    def save(
        self,
        group_id: str,
        *,
        exp_avg: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        storage_dtype: torch.dtype,
    ) -> str | None:
        # 未启用镜像时不执行落盘。
        if not self._enabled:
            return None
        # 确保镜像根目录存在。
        self._root.mkdir(parents=True, exist_ok=True)
        # dense 保存前先清理稀疏目录，避免新旧格式并存。
        self._clear_sparse_group(group_id)
        # 确定当前 group 的目标文件路径。
        path = self._path_for_group(group_id)
        # 先写到同目录临时文件，后面再原子替换。
        with tempfile.NamedTemporaryFile(
            dir=self._root,
            suffix=".pt.tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
        try:
            # 把一阶 / 二阶矩按指定 dtype 保存到临时文件。
            torch.save(
                {
                    "group_id": group_id,
                    "exp_avg": exp_avg.detach().to(dtype=storage_dtype, device="cpu"),
                    "exp_avg_sq": exp_avg_sq.detach().to(
                        dtype=storage_dtype,
                        device="cpu",
                    ),
                },
                temp_path,
            )
            # 原子替换为正式文件，避免写半截。
            os.replace(temp_path, path)
        finally:
            # 清理可能残留的临时文件。
            if temp_path.exists():
                temp_path.unlink()
        # 返回最终落盘路径。
        return str(path)

    def save_sparse(
        self,
        group_id: str,
        *,
        sparse_exp_avg: dict[tuple[int, int], torch.Tensor],
        sparse_exp_avg_sq: dict[tuple[int, int], torch.Tensor],
        storage_dtype: torch.dtype,
    ) -> str | None:
        # 未启用镜像时不执行稀疏落盘。
        if not self._enabled:
            return None
        # 确保该 group 的稀疏目录存在。
        sparse_root = self._sparse_root_for_group(group_id)
        sparse_root.mkdir(parents=True, exist_ok=True)
        # 稀疏保存时，先删除可能存在的 dense 文件。
        dense_path = self._path_for_group(group_id)
        if dense_path.exists():
            dense_path.unlink()
        # 逐段保存 sparse moment 切片。
        for key, exp_avg in sparse_exp_avg.items():
            # exp_avg_sq 必须和 exp_avg 成对存在。
            exp_avg_sq = sparse_exp_avg_sq.get(key)
            if exp_avg_sq is None:
                continue
            # 当前 key 由起始偏移和切片大小组成。
            start_offset, size = key
            # 生成当前切片的目标路径。
            path = self._sparse_path_for_slice(
                group_id,
                start_offset=start_offset,
                size=size,
            )
            # 先写到临时文件，再原子替换。
            with tempfile.NamedTemporaryFile(
                dir=sparse_root,
                suffix=".pt.tmp",
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
            try:
                # 保存当前稀疏切片的一阶 / 二阶矩。
                torch.save(
                    {
                        "group_id": group_id,
                        "start_offset": start_offset,
                        "size": size,
                        "exp_avg": exp_avg.detach().to(
                            dtype=storage_dtype,
                            device="cpu",
                        ),
                        "exp_avg_sq": exp_avg_sq.detach().to(
                            dtype=storage_dtype,
                            device="cpu",
                        ),
                    },
                    temp_path,
                )
                # 原子替换为正式切片文件。
                os.replace(temp_path, path)
            finally:
                # 清理可能残留的临时文件。
                if temp_path.exists():
                    temp_path.unlink()
        # 返回稀疏目录路径。
        return str(sparse_root)



@dataclass(slots=True, frozen=True)
class OptimizerStepResult:
    updates: tuple[OptimizerUpdateRecord, ...]
    optimizer_summary: OptimizerSummary


@dataclass(slots=True)
class _HostGradientBufferManager:
    storage_dtype_name: str
    scope: str
    _resident_bytes: int = 0
    _last_bucket_staged_bytes: int = 0
    _peak_bucket_staged_bytes: int = 0

    @property
    def storage_dtype(self) -> torch.dtype:
        # 把配置字符串解析成宿主梯度缓冲区实际使用的 dtype。
        return _storage_dtype(self.storage_dtype_name)

    @property
    def last_bucket_staged_bytes(self) -> int:
        # 返回最近一个 bucket 的梯度缓冲区字节数。
        return self._last_bucket_staged_bytes

    @property
    def peak_bucket_staged_bytes(self) -> int:
        # 返回当前 run 期间观测到的峰值宿主梯度缓冲区字节数。
        return self._peak_bucket_staged_bytes

    def _stage_tensor(
        self,
        gradient: torch.Tensor,
    ) -> torch.Tensor:
        # 先把梯度统一转成 CPU float32。
        storage_dtype = self.storage_dtype
        gradient = gradient.detach().to(dtype=torch.float32, device="cpu")
        # 空梯度直接返回空张量。
        if gradient.numel() == 0:
            return torch.empty(0, dtype=torch.float32, device="cpu")
        # 非 FP8 存储路径可以直接 round-trip 到目标 dtype 再转回 float32。
        if storage_dtype not in {torch.float8_e4m3fn, torch.float8_e5m2}:
            return gradient.to(dtype=storage_dtype, device="cpu").to(
                dtype=torch.float32,
                device="cpu",
            )
        # FP8 需要先根据最大绝对值计算缩放因子。
        max_abs = float(gradient.abs().max().item())
        # 全零梯度直接返回零张量。
        if max_abs <= 0.0:
            return torch.zeros_like(gradient, dtype=torch.float32, device="cpu")
        # 缩放因子用于把梯度压到目标 FP8 可表示范围内。
        scale = max_abs / max(float(torch.finfo(storage_dtype).max), 1.0)
        # 先缩放，再转换为 FP8 存储。
        scaled = gradient / max(scale, 1e-12)
        storage = scaled.to(dtype=storage_dtype, device="cpu")
        # 再乘回 scale，得到量化误差注入后的 float32 视图。
        return storage.to(dtype=torch.float32, device="cpu") * scale

    def stage(
        self,
        gradient_payloads: tuple["GradientPayload", ...],
    ) -> dict[str, tuple["GradientPayload", ...]]:
        # current_bucket_only 模式下，每个 bucket 开始前先清空常驻字节计数。
        if self.scope == "current_bucket_only":
            self._resident_bytes = 0
        # 用 group_id 聚合当前 bucket 的梯度 payload。
        staged: dict[str, list["GradientPayload"]] = {}
        bucket_bytes = 0
        # 逐个 payload 进行宿主侧 staging。
        for payload in gradient_payloads:
            # 空梯度不参与 staging。
            if payload.gradient.numel() == 0:
                continue
            # 先按配置 dtype 估算当前 payload 的存储字节数。
            storage = payload.gradient.detach().to(dtype=self.storage_dtype, device="cpu")
            bucket_bytes += storage.numel() * storage.element_size()
            # 将 payload 重新封装成 staged 版本，并按 group_id 聚合。
            staged.setdefault(payload.group_id, []).append(
                payload.__class__(
                    group_id=payload.group_id,
                    logical_params=payload.logical_params,
                    gradient=self._stage_tensor(payload.gradient),
                    start_offset=payload.start_offset,
                )
            )
        # 累加当前 scope 下的宿主常驻梯度字节数。
        self._resident_bytes += bucket_bytes
        # 记录最近一个 bucket 的 staged 字节数。
        self._last_bucket_staged_bytes = bucket_bytes
        # 更新宿主梯度缓冲区峰值。
        self._peak_bucket_staged_bytes = max(
            self._peak_bucket_staged_bytes,
            self._resident_bytes,
        )
        # 返回按 group_id 聚合后的 staged payload 映射。
        return {
            group_id: tuple(payloads)
            for group_id, payloads in staged.items()
        }

    def release_bucket(self) -> None:
        # current_bucket_only 模式下，bucket 结束就把常驻字节数清零。
        if self.scope == "current_bucket_only":
            self._resident_bytes = 0


@dataclass(slots=True)
class CPUOptimizerRuntime:
    config: TrainingProjectConfig
    _states: dict[str, _OptimizerShardState] = field(default_factory=dict)
    _cumulative_updates_applied: int = 0
    _gradient_buffer: _HostGradientBufferManager = field(init=False)
    _manifest: LocalWeightManifest = field(init=False)
    _state_mirror: _OptimizerStateMirror = field(init=False)

    def __post_init__(self) -> None:
        # 初始化宿主梯度缓冲管理器。
        self._gradient_buffer = _HostGradientBufferManager(
            storage_dtype_name=self.config.optimizer.gradient_buffer_storage_dtype,
            scope=self.config.bucket_schedule.host_gradient_buffer_scope,
        )
        # 初始化本地权重 manifest，供 logical 物化时估算参数规模。
        self._manifest = LocalWeightManifest(self.config)
        # 初始化优化器状态镜像管理器。
        self._state_mirror = _OptimizerStateMirror(self.config)

    def _representative_params(self, logical_params: int) -> int:
        # 代表性参数数至少为 1，且不会超过逻辑参数量和全局上限。
        return max(1, min(logical_params, _MAX_REPRESENTATIVE_PARAMS))

    def _materialized_params(
        self,
        shard: ParameterShardSnapshot,
    ) -> int:
        # logical 物化模式下，对可训练组件按完整来源布局长度估算参数量。
        if (
            self.config.execution.trainable_shard_materialization == "logical"
            and shard.component in _TRAINABLE_COMPONENTS
        ):
            plan = self._manifest.plan_full_parameter_buffer_sources(shard)
            return max(
                1,
                sum(entry.length for entry in plan.source_layout) or shard.logical_params,
            )
        # 其余情况退回代表性参数规模。
        return self._representative_params(shard.logical_params)

    def _materialize_cpu_state(
        self,
        state: _OptimizerShardState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 先尝试按需从 NVMe 状态镜像中恢复 dense moment。
        mirrored_state: tuple[torch.Tensor, torch.Tensor] | None = None
        if (
            state.exp_avg_tensor is None
            and state.exp_avg_sq_tensor is None
            and _buffer_is_empty(state.exp_avg_buffer)
            and _buffer_is_empty(state.exp_avg_sq_buffer)
            and self._state_mirror.enabled
        ):
            mirrored_state = self._state_mirror.load(state.group_id)
        # 确保 exp_avg 至少以 CPU tensor 形式可用。
        if state.exp_avg_tensor is None:
            if mirrored_state is not None:
                # 优先使用镜像恢复的一阶矩。
                state.exp_avg_tensor = mirrored_state[0]
            elif isinstance(state.exp_avg_buffer, torch.Tensor):
                # tensor buffer 直接转成 CPU float32。
                state.exp_avg_tensor = state.exp_avg_buffer.to(
                    dtype=torch.float32,
                    device="cpu",
                )
            elif state.exp_avg_buffer:
                # tuple buffer 重新物化成 CPU tensor。
                state.exp_avg_tensor = torch.tensor(
                    state.exp_avg_buffer,
                    dtype=torch.float32,
                    device="cpu",
                )
            else:
                # 当前完全没有状态时，从零初始化。
                state.exp_avg_tensor = torch.zeros(
                    state.representative_params,
                    dtype=torch.float32,
                    device="cpu",
                )
        # 确保 exp_avg_sq 至少以 CPU tensor 形式可用。
        if state.exp_avg_sq_tensor is None:
            if mirrored_state is not None:
                # 优先使用镜像恢复的二阶矩。
                state.exp_avg_sq_tensor = mirrored_state[1]
            elif isinstance(state.exp_avg_sq_buffer, torch.Tensor):
                # tensor buffer 直接转成 CPU float32。
                state.exp_avg_sq_tensor = state.exp_avg_sq_buffer.to(
                    dtype=torch.float32,
                    device="cpu",
                )
            elif state.exp_avg_sq_buffer:
                # tuple buffer 重新物化成 CPU tensor。
                state.exp_avg_sq_tensor = torch.tensor(
                    state.exp_avg_sq_buffer,
                    dtype=torch.float32,
                    device="cpu",
                )
            else:
                # 当前完全没有状态时，从零初始化。
                state.exp_avg_sq_tensor = torch.zeros(
                    state.representative_params,
                    dtype=torch.float32,
                    device="cpu",
                )
        # 返回已物化的一阶 / 二阶矩张量。
        return state.exp_avg_tensor, state.exp_avg_sq_tensor

    def _offload_cpu_state(self, state: _OptimizerShardState) -> None:
        # 按配置取出 CPU/NVMe 落盘时使用的状态存储 dtype。
        storage_dtype = _storage_dtype(self.config.optimizer.cpu_state_storage_dtype)
        # 稀疏 logical 状态优先走 sparse 镜像落盘路径。
        if (
            self._state_mirror.enabled
            and state.representative_params > _MAX_REPRESENTATIVE_PARAMS
            and state.sparse_exp_avg
            and state.sparse_exp_avg_sq
        ):
            self._state_mirror.save_sparse(
                state.group_id,
                sparse_exp_avg=state.sparse_exp_avg,
                sparse_exp_avg_sq=state.sparse_exp_avg_sq,
                storage_dtype=storage_dtype,
            )
            # 稀疏状态已经成功落盘后，清空内存态。
            state.sparse_exp_avg = {}
            state.sparse_exp_avg_sq = {}
            state.exp_avg_buffer = ()
            state.exp_avg_sq_buffer = ()
            state.exp_avg_tensor = None
            state.exp_avg_sq_tensor = None
            return
        # 大状态且已物化为 dense tensor 时，优先走 dense 镜像落盘路径。
        if (
            self._state_mirror.enabled
            and state.representative_params > _MAX_REPRESENTATIVE_PARAMS
            and state.exp_avg_tensor is not None
            and state.exp_avg_sq_tensor is not None
        ):
            self._state_mirror.save(
                state.group_id,
                exp_avg=state.exp_avg_tensor,
                exp_avg_sq=state.exp_avg_sq_tensor,
                storage_dtype=storage_dtype,
            )
            # 落盘完成后释放本地 dense 状态。
            state.exp_avg_buffer = ()
            state.exp_avg_sq_buffer = ()
            state.exp_avg_tensor = None
            state.exp_avg_sq_tensor = None
            return
        # 不能或不需要镜像落盘时，退回 CPU buffer 形式保存。
        if state.exp_avg_tensor is not None:
            state.exp_avg_buffer = state.exp_avg_tensor.to(
                dtype=storage_dtype,
                device="cpu",
            ).clone()
        if state.exp_avg_sq_tensor is not None:
            state.exp_avg_sq_buffer = state.exp_avg_sq_tensor.to(
                dtype=storage_dtype,
                device="cpu",
            ).clone()
        # 最后清空物化 tensor，只保留 buffer。
        state.exp_avg_tensor = None
        state.exp_avg_sq_tensor = None

    def _sync_state_mirror_for_snapshot(
        self,
        state: _OptimizerShardState,
    ) -> bool:
        # 未启用镜像或状态规模较小时，不需要为快照额外刷盘。
        if (
            not self._state_mirror.enabled
            or state.representative_params <= _MAX_REPRESENTATIVE_PARAMS
        ):
            return False
        # 稀疏状态优先按 sparse 格式刷到镜像目录。
        if state.sparse_exp_avg and state.sparse_exp_avg_sq:
            self._state_mirror.save_sparse(
                state.group_id,
                sparse_exp_avg=state.sparse_exp_avg,
                sparse_exp_avg_sq=state.sparse_exp_avg_sq,
                storage_dtype=torch.float32,
            )
            return True
        # 优先尝试直接使用当前物化 tensor。
        exp_avg = state.exp_avg_tensor
        exp_avg_sq = state.exp_avg_sq_tensor
        # 若没有物化 tensor，则从 buffer 中恢复出一阶矩。
        if exp_avg is None and isinstance(state.exp_avg_buffer, torch.Tensor):
            exp_avg = state.exp_avg_buffer.to(dtype=torch.float32, device="cpu")
        elif exp_avg is None and state.exp_avg_buffer:
            exp_avg = torch.tensor(
                state.exp_avg_buffer,
                dtype=torch.float32,
                device="cpu",
            )
        # 若没有物化 tensor，则从 buffer 中恢复出二阶矩。
        if exp_avg_sq is None and isinstance(state.exp_avg_sq_buffer, torch.Tensor):
            exp_avg_sq = state.exp_avg_sq_buffer.to(dtype=torch.float32, device="cpu")
        elif exp_avg_sq is None and state.exp_avg_sq_buffer:
            exp_avg_sq = torch.tensor(
                state.exp_avg_sq_buffer,
                dtype=torch.float32,
                device="cpu",
            )
        # 当前完全没有本地值时，若磁盘已有镜像则视为已同步。
        if exp_avg is None or exp_avg_sq is None:
            return self._state_mirror._path_for_group(state.group_id).exists()
        # 把 dense moment 强制刷到镜像目录。
        self._state_mirror.save(
            state.group_id,
            exp_avg=exp_avg,
            exp_avg_sq=exp_avg_sq,
            storage_dtype=torch.float32,
        )
        # 返回 True，表示快照可省略值本体。
        return True

    def _gradient_tensor(
        self,
        *,
        group_id: str,
        step_index: int,
        target_version: int,
        size: int,
    ) -> torch.Tensor:
        # 用 group_id、step 和版本号生成稳定种子。
        seed = _stable_seed(f"{group_id}:{step_index}:{target_version}")
        # 把种子映射到一个较小的标量尺度。
        scale = ((seed % 97) + 1) * 1e-5
        # 构造一个从 1 到 size 的简单梯度模板。
        index = torch.arange(1, size + 1, dtype=torch.float32, device="cpu")
        # 返回可复现的合成梯度向量。
        return index * (scale * float(step_index + target_version))

    def _apply_adamw_update(
        self,
        *,
        state: _OptimizerShardState,
        params: torch.Tensor,
        gradient: torch.Tensor,
    ) -> tuple[float, float, float]:
        # 先确保当前 shard 的 moment 已经物化到 CPU。
        exp_avg, exp_avg_sq = self._materialize_cpu_state(state)
        # 对输入梯度做数值稳定化处理。
        grad = _stabilize_update_tensor(gradient)
        # 当前 payload 的梯度长度必须和 representative_params 一致。
        if grad.numel() != state.representative_params:
            raise ValueError(
                "gradient payload size must match representative_params for shard "
                f"{state.group_id}"
            )
        # 委托到底层 view 版本执行一次完整的 AdamW 更新。
        return self._apply_adamw_update_views(
            params=params,
            exp_avg=exp_avg,
            exp_avg_sq=exp_avg_sq,
            gradient=grad,
            step_number=state.update_count + 1,
        )

    def _apply_adamw_update_views(
        self,
        *,
        params: torch.Tensor,
        exp_avg: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        gradient: torch.Tensor,
        step_number: int,
    ) -> tuple[float, float, float]:
        # 先对参数和 moment 视图做数值稳定化，避免异常值传播。
        params.copy_(_stabilize_update_tensor(params))
        exp_avg.copy_(_stabilize_update_tensor(exp_avg))
        exp_avg_sq.copy_(_stabilize_update_tensor(exp_avg_sq))
        # 梯度视图同样做稳定化。
        grad = _stabilize_update_tensor(gradient)
        # 记录梯度范数和更新前参数范数。
        grad_norm = float(torch.linalg.vector_norm(grad).item())
        param_norm_before = float(torch.linalg.vector_norm(params).item())

        # 读取 AdamW 所需超参数。
        beta1 = self.config.optimizer.beta1
        beta2 = self.config.optimizer.beta2
        learning_rate = self.config.optimizer.learning_rate
        epsilon = self.config.optimizer.epsilon
        weight_decay = self.config.optimizer.weight_decay

        # 更新一阶矩。
        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        # 更新二阶矩。
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        # 计算 bias correction 系数。
        bias_correction1 = 1.0 - beta1**step_number
        bias_correction2 = 1.0 - beta2**step_number
        # 构造 AdamW 分母项。
        denom = exp_avg_sq.sqrt().div_(bias_correction2**0.5).add_(epsilon)
        # 若配置了权重衰减，则先对参数做 decoupled decay。
        if weight_decay > 0:
            params.mul_(1.0 - learning_rate * weight_decay)
        # 再执行 AdamW 主更新。
        params.addcdiv_(exp_avg, denom, value=-(learning_rate / bias_correction1))
        # 记录更新后参数范数。
        param_norm_after = float(torch.linalg.vector_norm(params).item())
        # 返回梯度范数、更新前参数范数、更新后参数范数。
        return grad_norm, param_norm_before, param_norm_after

    def _get_or_create_state(
        self,
        shard: ParameterShardSnapshot,
    ) -> _OptimizerShardState:
        # 先尝试命中已有优化器分片状态。
        state = self._states.get(shard.group_id)
        if state is None:
            # 先确定当前 shard 实际物化的参数规模。
            representative_params = self._materialized_params(shard)
            # 大状态会直接使用 tensor-backed buffer。
            tensor_backed_state = representative_params > _MAX_REPRESENTATIVE_PARAMS
            # 启用镜像时，大状态允许懒加载，不必立刻初始化 buffer。
            lazy_mirrored_state = tensor_backed_state and self._state_mirror.enabled
            # 构造新的优化器分片状态对象。
            state = _OptimizerShardState(
                group_id=shard.group_id,
                component=shard.component,
                logical_params=shard.logical_params,
                representative_params=representative_params,
                exp_avg_buffer=(
                    ()
                    if lazy_mirrored_state
                    else _zero_buffer(
                        representative_params,
                        as_tensor=tensor_backed_state,
                    )
                ),
                exp_avg_sq_buffer=(
                    ()
                    if lazy_mirrored_state
                    else _zero_buffer(
                        representative_params,
                        as_tensor=tensor_backed_state,
                    )
                ),
            )
            # 把新状态注册到状态表。
            self._states[shard.group_id] = state
        # 返回命中或新建后的状态。
        return state

    def _build_summary(self) -> OptimizerSummary:
        # 先统计 CPU hot / NVMe cold 两个层级下的 shard 数。
        cpu_hot = 0
        nvme_cold = 0
        for state in self._states.values():
            if state.state_tier == "cpu_hot":
                cpu_hot += 1
            else:
                nvme_cold += 1
        # 组装并返回优化器汇总对象。
        return OptimizerSummary(
            tracked_shards=len(self._states),
            cpu_hot_shards=cpu_hot,
            nvme_cold_shards=nvme_cold,
            cumulative_updates_applied=self._cumulative_updates_applied,
            state_storage_dtype=self.config.optimizer.cpu_state_storage_dtype,
            gradient_buffer_storage_dtype=(
                self.config.optimizer.gradient_buffer_storage_dtype
            ),
            gradient_buffer_scope=self.config.bucket_schedule.host_gradient_buffer_scope,
            last_bucket_staged_gradient_bytes=(
                self._gradient_buffer.last_bucket_staged_bytes
            ),
            peak_bucket_staged_gradient_bytes=(
                self._gradient_buffer.peak_bucket_staged_bytes
            ),
        )

    def summary(self) -> OptimizerSummary:
        # 返回当前优化器运行时的汇总视图。
        return self._build_summary()

    def _use_sparse_logical_state(
        self,
        *,
        state: _OptimizerShardState,
        gradient_payloads: tuple["GradientPayload", ...],
    ) -> bool:
        # logical 物化 + 大状态 + 切片梯度 payload 时，启用稀疏状态路径。
        return (
            self.config.execution.trainable_shard_materialization == "logical"
            and state.representative_params > _MAX_REPRESENTATIVE_PARAMS
            and bool(gradient_payloads)
            and any(
                payload.start_offset != 0
                or payload.gradient.numel() != state.representative_params
                for payload in gradient_payloads
            )
        )

    def _sparse_state_views(
        self,
        state: _OptimizerShardState,
        *,
        start_offset: int,
        size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 稀疏状态字典的 key 由切片起始偏移和大小组成。
        key = (start_offset, size)
        # 先尝试命中已缓存的一阶矩切片。
        exp_avg = state.sparse_exp_avg.get(key)
        if exp_avg is None:
            # 若启用镜像，则先尝试从稀疏镜像文件恢复切片。
            mirrored = None
            if self._state_mirror.enabled:
                mirrored = self._state_mirror.load_sparse_slice(
                    state.group_id,
                    start_offset=start_offset,
                    size=size,
                )
            if mirrored is None:
                # 没有镜像时，用零张量初始化这段切片状态。
                exp_avg = torch.zeros(size, dtype=torch.float32, device="cpu")
                exp_avg_sq = torch.zeros(size, dtype=torch.float32, device="cpu")
            else:
                # 镜像命中时直接复用加载出的切片状态。
                exp_avg, exp_avg_sq = mirrored
            # 把恢复好的切片缓存到内存字典中。
            state.sparse_exp_avg[key] = exp_avg
            state.sparse_exp_avg_sq[key] = exp_avg_sq
            return exp_avg, exp_avg_sq
        # 一阶矩命中后，再确保二阶矩也存在。
        exp_avg_sq = state.sparse_exp_avg_sq.get(key)
        if exp_avg_sq is None:
            # 二阶矩缺失时单独补一个零张量。
            exp_avg_sq = torch.zeros(size, dtype=torch.float32, device="cpu")
            state.sparse_exp_avg_sq[key] = exp_avg_sq
        # 返回当前切片的一阶 / 二阶矩视图。
        return exp_avg, exp_avg_sq

    def _apply_gradient_payloads(
        self,
        *,
        state: _OptimizerShardState,
        params: torch.Tensor,
        gradient_payloads: tuple["GradientPayload", ...],
    ) -> tuple[float, float, float]:
        # 没有任何梯度 payload 时直接返回零统计。
        if not gradient_payloads:
            return 0.0, 0.0, 0.0
        # 稀疏 logical 大状态走切片状态路径。
        if self._use_sparse_logical_state(
            state=state,
            gradient_payloads=gradient_payloads,
        ):
            # 同一轮 payload 共享同一个 step_number。
            step_number = state.update_count + 1
            gradient_norm_square = 0.0
            parameter_norm_before_square = 0.0
            parameter_norm_after_square = 0.0
            # 逐个切片 payload 执行 AdamW 更新。
            for payload in gradient_payloads:
                gradient = _stabilize_update_tensor(payload.gradient)
                start_offset = payload.start_offset
                stop_offset = start_offset + gradient.numel()
                # payload 切片越界时直接报错。
                if start_offset < 0 or stop_offset > params.numel():
                    raise ValueError(
                        "gradient payload slice is out of bounds for shard "
                        f"{state.group_id}: {start_offset}:{stop_offset} / {params.numel()}"
                    )
                # 获取该切片对应的稀疏 moment 视图。
                exp_avg, exp_avg_sq = self._sparse_state_views(
                    state,
                    start_offset=start_offset,
                    size=gradient.numel(),
                )
                # 仅更新参数和 moment 的这一段 narrow view。
                grad_norm, param_norm_before, param_norm_after = (
                    self._apply_adamw_update_views(
                        params=params.narrow(0, start_offset, gradient.numel()),
                        exp_avg=exp_avg,
                        exp_avg_sq=exp_avg_sq,
                        gradient=gradient,
                        step_number=step_number,
                    )
                )
                # 用平方和方式聚合多个 payload 的范数统计。
                gradient_norm_square += grad_norm * grad_norm
                parameter_norm_before_square += param_norm_before * param_norm_before
                parameter_norm_after_square += param_norm_after * param_norm_after
            # 返回整组 payload 聚合后的范数统计。
            return (
                gradient_norm_square**0.5,
                parameter_norm_before_square**0.5,
                parameter_norm_after_square**0.5,
            )
        # 非稀疏路径下，先物化整块 CPU moment。
        exp_avg, exp_avg_sq = self._materialize_cpu_state(state)
        step_number = state.update_count + 1
        gradient_norm_square = 0.0
        parameter_norm_before_square = 0.0
        parameter_norm_after_square = 0.0
        # 逐个 payload 对 dense moment 的对应切片执行更新。
        for payload in gradient_payloads:
            gradient = _stabilize_update_tensor(payload.gradient)
            start_offset = payload.start_offset
            stop_offset = start_offset + gradient.numel()
            # payload 切片越界时直接报错。
            if start_offset < 0 or stop_offset > params.numel():
                raise ValueError(
                    "gradient payload slice is out of bounds for shard "
                    f"{state.group_id}: {start_offset}:{stop_offset} / {params.numel()}"
                )
            # 仅更新参数和 moment 的相应 narrow view。
            grad_norm, param_norm_before, param_norm_after = (
                self._apply_adamw_update_views(
                    params=params.narrow(0, start_offset, gradient.numel()),
                    exp_avg=exp_avg.narrow(0, start_offset, gradient.numel()),
                    exp_avg_sq=exp_avg_sq.narrow(0, start_offset, gradient.numel()),
                    gradient=gradient,
                    step_number=step_number,
                )
            )
            # 用平方和方式聚合多个 payload 的范数统计。
            gradient_norm_square += grad_norm * grad_norm
            parameter_norm_before_square += param_norm_before * param_norm_before
            parameter_norm_after_square += param_norm_after * param_norm_after
        # 返回整组 payload 聚合后的范数统计。
        return (
            gradient_norm_square**0.5,
            parameter_norm_before_square**0.5,
            parameter_norm_after_square**0.5,
        )

    def apply_step(
        self,
        *,
        step_index: int,
        parameter_shards: tuple[ParameterShardSnapshot, ...],
        parameter_store: ParameterShardStore,
        gradient_payloads: tuple["GradientPayload", ...] = (),
        keep_resident_group_ids: tuple[str, ...] = (),
    ) -> OptimizerStepResult:
        # 用列表累积当前 step 产生的更新记录。
        updates: list[OptimizerUpdateRecord] = []
        # 先把梯度 payload stage 到宿主侧缓冲区，并按 group_id 聚合。
        gradient_map = self._gradient_buffer.stage(gradient_payloads)
        # 逐个参数分片执行优化器更新。
        for shard in parameter_shards:
            # 仅训练可训练组件。
            if shard.component not in _TRAINABLE_COMPONENTS:
                continue
            # 只处理本 step 实际触达过的 shard。
            if shard.last_touched_step != step_index:
                continue
            # 获取或创建对应的优化器状态。
            state = self._get_or_create_state(shard)
            # 已提交版本未前进时无需重复更新。
            if shard.committed_version <= state.last_committed_version:
                continue
            # 记录这次要补齐多少个版本差。
            version_delta = shard.committed_version - state.last_committed_version
            gradient_l2_norm = 0.0
            parameter_l2_norm_before = 0.0
            parameter_l2_norm_after = 0.0
            # 取出当前 shard 的可写参数张量。
            params = parameter_store.mutable_parameter(
                shard,
                step_index=step_index,
            )
            # 逐个目标版本回放更新，直到赶上 committed_version。
            for target_version in range(
                state.last_committed_version + 1,
                shard.committed_version + 1,
            ):
                # 优先使用真实 staged 梯度；没有时退回合成梯度。
                staged_payloads = gradient_map.get(shard.group_id)
                if staged_payloads is None:
                    gradient = self._gradient_tensor(
                        group_id=state.group_id,
                        step_index=step_index,
                        target_version=target_version,
                        size=state.representative_params,
                    )
                    (
                        gradient_l2_norm,
                        parameter_l2_norm_before,
                        parameter_l2_norm_after,
                    ) = self._apply_adamw_update(
                        state=state,
                        params=params,
                        gradient=gradient,
                    )
                else:
                    # 有真实 payload 时直接按切片梯度更新。
                    (
                        gradient_l2_norm,
                        parameter_l2_norm_before,
                        parameter_l2_norm_after,
                    ) = self._apply_gradient_payloads(
                        state=state,
                        params=params,
                        gradient_payloads=staged_payloads,
                    )
                # 每完成一次版本更新，update_count 加一。
                state.update_count += 1
            # 根据配置和 keep_resident 列表决定更新后是否 offload。
            offload_after_update = (
                self.config.optimizer.offload_state_after_update
                and shard.group_id not in keep_resident_group_ids
            )
            # 更新状态对象里的版本与最近更新步号。
            state.last_committed_version = shard.committed_version
            state.last_updated_step = step_index
            # 维护当前优化器状态所在层级。
            state.state_tier = (
                "nvme_cold"
                if offload_after_update
                else "cpu_hot"
            )
            # 需要时把 CPU 状态落到 buffer 或 NVMe 镜像。
            if offload_after_update:
                self._offload_cpu_state(state)
            # 同步通知参数存储完成了一次参数更新。
            parameter_store.finalize_update(
                shard,
                step_index=step_index,
                offload_after_update=offload_after_update,
                sync_fp32_to_nvme=offload_after_update,
                retain_gpu_quantized_cache=(
                    shard.group_id in keep_resident_group_ids
                ),
            )
            # 累加本 step 实际补齐的版本数。
            self._cumulative_updates_applied += version_delta
            # 追加一条优化器更新记录。
            updates.append(
                OptimizerUpdateRecord(
                    group_id=shard.group_id,
                    component=shard.component,
                    step_index=step_index,
                    target_version=shard.committed_version,
                    logical_params=shard.logical_params,
                    representative_params=state.representative_params,
                    algorithm=self.config.optimizer.algorithm,
                    learning_rate=self.config.optimizer.learning_rate,
                    weight_decay=self.config.optimizer.weight_decay,
                    state_tier=state.state_tier,
                    offloaded_after_update=offload_after_update,
                    shard_update_count=state.update_count,
                    gradient_l2_norm=gradient_l2_norm,
                    parameter_l2_norm_before=parameter_l2_norm_before,
                    parameter_l2_norm_after=parameter_l2_norm_after,
                )
            )
        # 一个 bucket 的更新结束后，根据 scope 释放梯度缓冲计数。
        self._gradient_buffer.release_bucket()
        # 返回本 step 的更新记录和优化器汇总。
        return OptimizerStepResult(
            updates=tuple(updates),
            optimizer_summary=self._build_summary(),
        )

    def apply_gradients(
        self,
        *,
        step_index: int,
        parameter_shards: tuple[ParameterShardSnapshot, ...],
        parameter_store: ParameterShardStore,
        gradient_payloads: tuple["GradientPayload", ...],
        keep_resident_group_ids: tuple[str, ...] = (),
    ) -> OptimizerStepResult:
        # 用列表累积当前 bucket 产生的更新记录。
        updates: list[OptimizerUpdateRecord] = []
        # 先把梯度 payload stage 到宿主侧缓冲区，并按 group_id 聚合。
        gradient_map = self._gradient_buffer.stage(gradient_payloads)
        # 逐个参数分片执行真实梯度更新。
        for shard in parameter_shards:
            # 仅训练可训练组件。
            if shard.component not in _TRAINABLE_COMPONENTS:
                continue
            # 当前 shard 没有 staged 梯度时直接跳过。
            staged_payloads = gradient_map.get(shard.group_id)
            if staged_payloads is None:
                continue
            # 获取或创建当前 shard 的优化器状态。
            state = self._get_or_create_state(shard)
            # 取出当前 shard 的可写参数张量。
            params = parameter_store.mutable_parameter(
                shard,
                step_index=step_index,
            )
            # 按 staged payload 对参数和 optimizer state 执行更新。
            (
                gradient_l2_norm,
                parameter_l2_norm_before,
                parameter_l2_norm_after,
            ) = self._apply_gradient_payloads(
                state=state,
                params=params,
                gradient_payloads=staged_payloads,
            )
            # 当前 shard 已完成一次更新。
            state.update_count += 1
            # 根据配置和 keep_resident 列表决定是否 offload。
            offload_after_update = (
                self.config.optimizer.offload_state_after_update
                and shard.group_id not in keep_resident_group_ids
            )
            # committed_version 至少要推进到当前 shard 的版本。
            state.last_committed_version = max(
                state.last_committed_version,
                shard.committed_version,
            )
            state.last_updated_step = step_index
            # 更新状态所在层级。
            state.state_tier = (
                "nvme_cold"
                if offload_after_update
                else "cpu_hot"
            )
            # 需要时落盘并释放 CPU state。
            if offload_after_update:
                self._offload_cpu_state(state)
            # 通知参数存储完成本次参数更新。
            parameter_store.finalize_update(
                shard,
                step_index=step_index,
                offload_after_update=offload_after_update,
                sync_fp32_to_nvme=offload_after_update,
                retain_gpu_quantized_cache=(
                    shard.group_id in keep_resident_group_ids
                ),
            )
            # 累计一次真实梯度更新。
            self._cumulative_updates_applied += 1
            # 追加一条优化器更新记录。
            updates.append(
                OptimizerUpdateRecord(
                    group_id=shard.group_id,
                    component=shard.component,
                    step_index=step_index,
                    target_version=state.update_count,
                    logical_params=shard.logical_params,
                    representative_params=state.representative_params,
                    algorithm=self.config.optimizer.algorithm,
                    learning_rate=self.config.optimizer.learning_rate,
                    weight_decay=self.config.optimizer.weight_decay,
                    state_tier=state.state_tier,
                    offloaded_after_update=offload_after_update,
                    shard_update_count=state.update_count,
                    gradient_l2_norm=gradient_l2_norm,
                    parameter_l2_norm_before=parameter_l2_norm_before,
                    parameter_l2_norm_after=parameter_l2_norm_after,
                )
            )
        # 一个 bucket 的更新结束后，根据 scope 释放梯度缓冲计数。
        self._gradient_buffer.release_bucket()
        # 返回当前 bucket 的更新记录和优化器汇总。
        return OptimizerStepResult(
            updates=tuple(updates),
            optimizer_summary=self._build_summary(),
        )

    def offload_group_ids(
        self,
        *,
        group_ids: tuple[str, ...],
    ) -> None:
        # 逐个 group 把优化器状态落冷。
        for group_id in group_ids:
            state = self._states.get(group_id)
            if state is None:
                continue
            self._offload_cpu_state(state)
            state.state_tier = "nvme_cold"

    def snapshot(self) -> tuple[OptimizerShardStateSnapshot, ...]:
        # 按 group_id 排序导出优化器分片快照。
        return tuple(
            state.snapshot_with_values(
                include_values=not self._sync_state_mirror_for_snapshot(state)
            )
            for _, state in sorted(self._states.items(), key=lambda item: item[0])
        )

    def load_snapshot(
        self,
        snapshots: tuple[OptimizerShardStateSnapshot, ...],
    ) -> None:
        # 用快照列表重建整个优化器状态表。
        self._states = {
            snapshot.group_id: _OptimizerShardState(
                group_id=snapshot.group_id,
                component=snapshot.component,
                logical_params=snapshot.logical_params,
                representative_params=max(
                    snapshot.representative_params,
                    self._representative_params(snapshot.logical_params),
                ),
                update_count=snapshot.update_count,
                last_committed_version=snapshot.last_committed_version,
                last_updated_step=snapshot.last_updated_step,
                state_tier=snapshot.state_tier,
                exp_avg_buffer=(
                    snapshot.exp_avg_values
                    if snapshot.exp_avg_values
                    else (
                        ()
                        if (
                            self._state_mirror.enabled
                            and max(
                                snapshot.representative_params,
                                self._representative_params(snapshot.logical_params),
                            )
                            > _MAX_REPRESENTATIVE_PARAMS
                        )
                        else _zero_buffer(
                            max(
                                snapshot.representative_params,
                                self._representative_params(snapshot.logical_params),
                            ),
                            as_tensor=max(
                                snapshot.representative_params,
                                self._representative_params(snapshot.logical_params),
                            ) > _MAX_REPRESENTATIVE_PARAMS,
                        )
                    )
                ),
                exp_avg_sq_buffer=(
                    snapshot.exp_avg_sq_values
                    if snapshot.exp_avg_sq_values
                    else (
                        ()
                        if (
                            self._state_mirror.enabled
                            and max(
                                snapshot.representative_params,
                                self._representative_params(snapshot.logical_params),
                            )
                            > _MAX_REPRESENTATIVE_PARAMS
                        )
                        else _zero_buffer(
                            max(
                                snapshot.representative_params,
                                self._representative_params(snapshot.logical_params),
                            ),
                            as_tensor=max(
                                snapshot.representative_params,
                                self._representative_params(snapshot.logical_params),
                            ) > _MAX_REPRESENTATIVE_PARAMS,
                        )
                    )
                ),
            )
            for snapshot in snapshots
        }
        # 重新累计总更新次数。
        self._cumulative_updates_applied = sum(
            snapshot.update_count for snapshot in snapshots
        )

    def summary(self) -> OptimizerSummary:
        # 返回当前优化器运行时的汇总视图。
        return self._build_summary()
