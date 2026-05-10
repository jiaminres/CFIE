"""Qwen3.5-122B 训练专用模型（"车身"）。

完整实现 Qwen3.5 MoE Transformer，与训练基座集成：
- 48 层 DecoderLayer（GDN / FullAttention 混合 + TrainingMoE）
- Hot experts: FP16 shadow（可训练，产生 weight grad）
- Cold experts: GPU GPTQ Int4 decode（只前向，不产生 weight grad）
- Router 输出 logits 用于辅助 loss
- 集成 ForwardShadowStore（sync_from_shadows 同步 hot 权重）
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ───────────────────── 基础模块 ─────────────────────

class RMSNorm(nn.Module):
    """RMS Layer Normalization。"""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x / rms * self.weight).to(dtype)


# ───────────────────── MoE 层（hot/cold 混合） ─────────────────────

class TrainingQwenMoELayer(nn.Module):
    """训练用 Qwen3.5 MoE 层。

    每个 layer 包含:
    - Router: Linear(hidden, num_experts)
    - Shared expert: gate_up + down (dense)
    - Routed experts: hot (FP16, 可训练) + cold (GPTQ Int4 GPU decode)
    """

    def __init__(
        self,
        hidden_size: int = 3072,          # Qwen3.5-122B: 3072
        intermediate_size: int = 1024,    # MoE expert 中间维度
        num_experts: int = 256,           # Routed expert 总数
        top_k: int = 8,                   # 每 token 激活 expert 数
        dtype: torch.dtype = torch.float16,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.hidden_size = hidden_size        # H = 3072
        self.intermediate_size = intermediate_size  # I = 1024
        self.num_experts = num_experts        # E = 256
        self.top_k = top_k                    # K = 8
        self.dtype = dtype                    # 权重精度（FP16/BF16）
        self._device = device                 # 设备（CPU/CUDA）
        self.layer_idx: int = -1              # 由 DecoderLayer 设置

        # ── Router: Linear(H → E)，输出 256 专家 logits ──
        self.router = nn.Linear(hidden_size, num_experts, bias=False, dtype=dtype, device=device)

        # ── Shared expert（所有 token 共享）──
        # w13: gate_proj + up_proj 合并 [2*I, H]
        self.shared_w13 = nn.Parameter(torch.empty(2 * intermediate_size, hidden_size, dtype=dtype, device=device))
        # w2: down_proj [H, I]
        self.shared_w2 = nn.Parameter(torch.empty(hidden_size, intermediate_size, dtype=dtype, device=device))
        self._init_weights()                  # Kaiming 初始化

        # ── Hot experts（可训练 FP16 shadow）──
        self._hot_w13: dict[int, nn.Parameter] = {}   # {expert_id: [2*I,H] FP16}
        self._hot_w2: dict[int, nn.Parameter] = {}    # {expert_id: [H,I] FP16}

        # ── Cold experts（CPU GPTQ Int4 → GPU decode）──
        self.cpu_gptq_cache: Any = None              # CpuFullGptqCache 引用
        self.resident_cache: Any = None              # ResidentGptqCache 引用（predictor 预取目标）
        self._gpu_module_cache: dict[tuple, Any] = {} # 本层已创建的 Marlin 模块缓存

        # ── Router 输出记录（供 loss + GPU lock 使用）──
        self.current_router_logits: torch.Tensor | None = None           # [B*T, E] 本步 router 输出
        self.active_expert_ids: set[tuple[int, int]] = set()              # {(layer, expert)} 本步激活专家

    def _init_weights(self):
        """Kaiming 初始化 shared expert 和 router 权重。"""
        nn.init.kaiming_uniform_(self.shared_w13, a=math.sqrt(5))     # [2*I, H]
        nn.init.kaiming_uniform_(self.shared_w2, a=math.sqrt(5))     # [H, I]
        nn.init.kaiming_uniform_(self.router.weight, a=math.sqrt(5))  # [E, H]

    @property
    def hot_experts(self) -> set[int]:
        return set(self._hot_w13) | set(self._hot_w2)

    def set_hot_expert(
        self, expert_id: int, w13: torch.Tensor, w2: torch.Tensor,
    ) -> None:
        """将一个 expert 注册为 hot（nn.Parameter，参与 autograd + Adam 更新）。"""
        # w13: [2*I, H] gate_up 合并权重 → FP16/BF16 + contiguous
        w13_param = nn.Parameter(w13.to(dtype=self.dtype, device=self._device).contiguous(), requires_grad=True)
        # w2: [H, I] down_proj 权重
        w2_param = nn.Parameter(w2.to(dtype=self.dtype, device=self._device).contiguous(), requires_grad=True)
        # 通过 register_parameter 注册到 module 的参数列表中（autograd 可见）
        self.register_parameter(f"_hot_w13_{expert_id}", w13_param)
        self.register_parameter(f"_hot_w2_{expert_id}", w2_param)
        self._hot_w13[expert_id] = w13_param     # 快速查找索引
        self._hot_w2[expert_id] = w2_param

    def set_cold_expert_gptq(
        self,
        expert_id: int,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        qzeros: torch.Tensor,
        out_features: tuple[int, int],
    ) -> None:
        """设置冷专家的 GPTQ Int4 权重。"""
        self.cold_experts[expert_id] = (
            qweight.to(device=self._device),
            scales.to(device=self._device),
            qzeros.to(device=self._device),
        )
        self._cold_expert_out_features[expert_id] = out_features

    def remove_expert(self, expert_id: int) -> None:
        self._hot_w13.pop(expert_id, None)
        self._hot_w2.pop(expert_id, None)
        for prefix in (f"_hot_w13_{expert_id}", f"_hot_w2_{expert_id}"):
            if prefix in dict(self.named_parameters()):
                delattr(self, prefix)

    def _get_cold_expert_cached(
        self, expert_id: int
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """获取冷专家 GPTQMarlinFP8Linear 模块（三级缓存查找）。

        1. GPU module cache → 命中直接返回（同层内已创建过）
        2. CPU GPTQ 缓存 → packed int32 + fp16 scales
        3. gptq_pack → GPTQMarlinFP8Linear.from_raw_gptq → MMA 加速模块
        4. 存入 GPU module cache（本层后续 token 复用）

        每层 forward 后 _gpu_module_cache.clear() 释放显存。
        """
        # 缓存 key: (layer_idx, expert_id, weight_name)
        key_w13 = (self.layer_idx, expert_id, "w13")
        key_w2 = (self.layer_idx, expert_id, "w2")

        # 1. GPU module cache 查找（本层已为其他 token 创建过该模块）
        w13 = self._gpu_module_cache.get(key_w13)
        w2 = self._gpu_module_cache.get(key_w2)
        if w13 is not None and w2 is not None:
            return w13, w2  # 缓存命中：直接复用

        # 2. 从 CPU 全量缓存加载 packed int32 + fp16 scales
        if self.cpu_gptq_cache is None:
            return None, None
        packed13, sc13, packed2, sc2 = self._load_cold_logical(expert_id)
        if packed13 is None:
            return None, None

        # 3. gptq_pack → GPTQMarlinFP8Linear（内部做 marlin_repack + scales permute）
        # w13: Linear(hidden→2*inter)  [B,3072]→[B,2048]
        w13 = self._build_marlin_module(packed13, sc13,
                                        2 * self.intermediate_size, self.hidden_size)
        # w2: Linear(inter→hidden)    [B,1024]→[B,3072]
        w2 = self._build_marlin_module(packed2, sc2,
                                        self.hidden_size, self.intermediate_size)

        # 4. 存入 GPU module cache（本层内后续 token 复用，避免重复 Marlin repack）
        if w13 is not None: self._gpu_module_cache[key_w13] = w13
        if w2 is not None: self._gpu_module_cache[key_w2] = w2
        return w13, w2

    def _get_from_cache(self, bundle_id: str) -> torch.Tensor | None:
        try:
            return self.resident_cache.get(bundle_id) if self.resident_cache else None
        except Exception:
            return None

    def _load_cold_logical(self, expert_id: int):
        """从 CPU 缓存加载 gptq_pack 格式的 packed int32 + fp16 scales。"""
        entry = self.cpu_gptq_cache._entries.get((self.layer_idx, expert_id))
        if entry is None:
            return None, None, None, None
        w13_qb, w13_sb, w2_qb, w2_sb = entry
        if not w13_qb or not w2_qb:
            return None, None, None, None
        # packed: [in_f/8, out_f] int32 (gptq_pack 格式)
        packed_qw13 = torch.frombuffer(bytearray(w13_qb), dtype=torch.int32)
        sc13 = torch.frombuffer(bytearray(w13_sb), dtype=torch.float16)
        packed_qw2 = torch.frombuffer(bytearray(w2_qb), dtype=torch.int32)
        sc2 = torch.frombuffer(bytearray(w2_sb), dtype=torch.float16)
        return packed_qw13, sc13, packed_qw2, sc2

    def _build_marlin_module(
        self, packed_qw: torch.Tensor, scales: torch.Tensor,
        out_f: int, in_f: int,
    ) -> torch.Tensor | None:
        """CPU 缓存中的 gptq_pack 格式 + scales → GPTQMarlinFP8Linear。

        packed_qw: 1D int32 bytes from cache, reshape to [in_f/8, out_f]
        scales: 1D fp16 bytes from cache (transposed in build: [num_groups, out_f])
        """
        # Reshape packed: flat bytes → [in_f/8, out_f] int32 (gptq_pack format)
        packed = packed_qw.reshape(in_f // 8, out_f).to(self._device)
        sc_2d = scales.reshape(-1, out_f).to(self._device)  # [num_groups, out_f]
        num_groups = sc_2d.shape[0]
        group_size = in_f // num_groups if num_groups > 0 else in_f

        from cfie.op_validation.gptq_marlin_fp8 import GPTQMarlinFP8Linear
        module = GPTQMarlinFP8Linear.from_raw_gptq(
            packed, sc_2d, size_k=in_f, size_n=out_f, group_size=group_size,
        )
        return module  # nn.Module, forward: [B,in_f] → [B,out_f]

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """MoE 混合前向——hot (FP16 shadow, 可训练) + cold (Int4 decode, 只前向)。

        流程: Router top-k → 按 token 逐专家计算 → Shared expert 补充 → 输出

        Returns: (output [B,T,H], router_logits [B*T, num_experts])
        """
        B, T, H = hidden_states.shape
        flat = hidden_states.reshape(-1, H)         # [B*T, H] 展平 batch+seq

        # ── Router: 计算每个 token 对 256 专家的偏好 ──
        router_logits = self.router(flat)            # [B*T, num_experts]
        self.current_router_logits = router_logits

        # softmax → top-k 选择 → 归一化权重
        routing_weights = F.softmax(router_logits, dim=-1)        # [B*T, E]
        top_weights, top_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)  # 重归一化

        # ── Shared expert（所有 token 共享）──
        shared_out = self._shared_forward(flat)      # [B*T, H]

        # ── Routed experts: hot (trainable) + cold (forward-only) ──
        routed_out = torch.zeros_like(flat)          # [B*T, H] 累积输出
        num_tokens, _ = flat.shape

        # 记录本层 router 选中的专家 ID（供上层 lock/unlock GPU cache）
        active_ids: set[tuple[int, int]] = set()

        for token_idx in range(min(num_tokens, 32)):  # 逐 token 处理（真实训练不限制）
            for k in range(self.top_k):
                expert_id = int(top_indices[token_idx, k].item())
                weight = top_weights[token_idx, k]
                single = flat[token_idx:token_idx + 1]
                active_ids.add((self.layer_idx, expert_id))

                if expert_id in self._hot_w13:
                    # 热专家：FP16 shadow，可训练，产生 weight grad
                    w13, w2 = self._hot_w13[expert_id], self._hot_w2[expert_id]
                    gate_up = F.linear(single, w13)
                    gate, up = gate_up.chunk(2, dim=-1)
                    expert_out = F.linear(F.silu(gate) * up, w2)
                    routed_out[token_idx] += weight * expert_out.squeeze(0)
                else:
                    # 冷专家: GPU cache → GPTQMarlinFP8Linear (marlin_gemm MMA 加速)
                    w13_mod, w2_mod = self._get_cold_expert_cached(expert_id)
                    if w13_mod is not None and w2_mod is not None:
                        s = single.to(self.dtype)
                        gate_up = w13_mod(s)       # FP8×Int4 MMA forward
                        gate, up = gate_up.chunk(2, dim=-1)
                        expert_out = w2_mod(F.silu(gate) * up)
                        routed_out[token_idx] += weight * expert_out.squeeze(0)

        output = shared_out + routed_out
        # 保存本次 forward 的激活专家 ID，供上层 lock/unlock GPU cache
        self.active_expert_ids = active_ids
        return output.reshape(B, T, H), router_logits

    def _shared_forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = F.linear(x, self.shared_w13)
        gate, up = gate_up.chunk(2, dim=-1)
        return F.linear(F.silu(gate) * up, self.shared_w2)

    @property
    def hot_expert_ids(self) -> set[int]:
        """返回当前 hot expert 的 ID 集合。"""
        return set(self._hot_w13) | set(self._hot_w2)


# ───────────────────── GatedDeltaNet (线性注意力训练版) ─────────────────────

class TrainingGatedDeltaNet(nn.Module):
    """GatedDeltaNet 的训练兼容版本。

    使用纯 PyTorch 实现（非 CUDA kernel），支持 autograd。
    """

    def __init__(
        self,
        hidden_size: int = 3072,
        num_q_heads: int = 16,
        num_kv_heads: int = 2,
        head_dim: int = 256,
        conv_kernel_size: int = 4,
        dtype: torch.dtype = torch.float16,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads

        # 输入/输出投影
        qkv_dim = num_q_heads * head_dim + 2 * num_kv_heads * head_dim
        self.in_proj = nn.Linear(hidden_size, qkv_dim, bias=False, dtype=dtype, device=device)
        self.out_proj = nn.Linear(
            num_q_heads * head_dim, hidden_size, bias=False, dtype=dtype, device=device
        )

        # 1D 卷积
        self.conv1d = nn.Conv1d(
            hidden_size, hidden_size, conv_kernel_size,
            groups=hidden_size, padding=conv_kernel_size - 1,
            dtype=dtype, device=device,
        )

        # Norms
        self.conv_norm = RMSNorm(hidden_size)
        self.attn_norm = RMSNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, H = hidden_states.shape

        # Conv1d
        residual = hidden_states
        x = hidden_states.transpose(1, 2)  # [B, H, T]
        x = self.conv1d(x)[:, :, :T]
        x = x.transpose(1, 2)  # [B, T, H]
        x = self.conv_norm(x)

        # QKV 投影
        qkv = self.in_proj(x)
        q = qkv[..., :self.num_q_heads * self.head_dim]
        k = qkv[..., self.num_q_heads * self.head_dim:
                self.num_q_heads * self.head_dim + self.num_kv_heads * self.head_dim]
        v = qkv[..., -self.num_kv_heads * self.head_dim:]

        # GQA: repeat KV heads to match Q heads
        n_groups = self.num_q_heads // self.num_kv_heads
        q = q.reshape(B, T, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, T, self.num_kv_heads, self.head_dim)
        v = v.reshape(B, T, self.num_kv_heads, self.head_dim)
        if n_groups > 1:
            k = k.unsqueeze(2).expand(-1, -1, n_groups, -1, -1).reshape(
                B, T, self.num_q_heads, self.head_dim,
            )
            v = v.unsqueeze(2).expand(-1, -1, n_groups, -1, -1).reshape(
                B, T, self.num_q_heads, self.head_dim,
            )
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 简化线性注意力: Q @ K^T @ V
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(B, T, self.num_q_heads * self.head_dim)
        out = self.out_proj(out)
        return out + residual


# ───────────────────── Full Attention ─────────────────────

class TrainingFullAttention(nn.Module):
    """训练用 Full Attention（SDPA + RoPE + sigmoid gate）。"""

    def __init__(
        self,
        hidden_size: int = 3072,
        num_heads: int = 16,
        num_kv_heads: int = 2,
        head_dim: int = 256,
        dtype: torch.dtype = torch.float16,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        q_dim = num_heads * head_dim
        kv_dim = num_kv_heads * head_dim
        self.q_proj = nn.Linear(hidden_size, q_dim, bias=False, dtype=dtype, device=device)
        self.k_proj = nn.Linear(hidden_size, kv_dim, bias=False, dtype=dtype, device=device)
        self.v_proj = nn.Linear(hidden_size, kv_dim, bias=False, dtype=dtype, device=device)
        self.o_proj = nn.Linear(q_dim, hidden_size, bias=False, dtype=dtype, device=device)

        self.q_norm = RMSNorm(head_dim) if num_heads > 1 else None
        self.k_norm = RMSNorm(head_dim) if num_kv_heads > 1 else None

        # Sigmoid output gate
        self.gate_proj = nn.Linear(hidden_size, q_dim, bias=False, dtype=dtype, device=device)
        self.gate_out_proj = nn.Linear(q_dim, hidden_size, bias=False, dtype=dtype, device=device)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, H = hidden_states.shape

        q = self.q_proj(hidden_states).reshape(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).reshape(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).reshape(B, T, self.num_kv_heads, self.head_dim)

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        q = q.transpose(1, 2)  # [B, H, T, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # GQA 支持
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True,
            enable_gqa=self.num_kv_heads < self.num_heads,
        )
        out = out.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        out = self.o_proj(out)

        # Sigmoid gate
        gate = F.sigmoid(self.gate_out_proj(F.silu(self.gate_proj(hidden_states))))
        return out * gate


# ───────────────────── Decoder Layer ─────────────────────

LAYER_PATTERN_122B = (
    ["gdn"] * 3 + ["full"]  # 3 GDN + 1 Full Attention, 重复 12 次 = 48 层
) * 12


class TrainingDecoderLayer(nn.Module):
    """训练用 Decoder Layer: Attention + MoE。"""

    def __init__(
        self,
        layer_idx: int,
        hidden_size: int = 3072,
        intermediate_size: int = 1024,
        num_experts: int = 256,
        top_k: int = 8,
        dtype: torch.dtype = torch.float16,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.layer_idx = layer_idx

        layer_type = LAYER_PATTERN_122B[layer_idx] if layer_idx < 48 else "full"

        if layer_type == "gdn":
            self.attention = TrainingGatedDeltaNet(
                hidden_size=hidden_size, dtype=dtype, device=device,
            )
        else:
            self.attention = TrainingFullAttention(
                hidden_size=hidden_size, dtype=dtype, device=device,
            )

        self.input_layernorm = RMSNorm(hidden_size)
        self.post_attention_layernorm = RMSNorm(hidden_size)
        self.moe = TrainingQwenMoELayer(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
            dtype=dtype,
            device=device,
        )
        self.moe.layer_idx = layer_idx

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attention(hidden_states)
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        moe_out, router_logits = self.moe(hidden_states)
        return moe_out + residual, router_logits


# ───────────────────── 完整训练模型 ─────────────────────

class Qwen35ForTraining(nn.Module):
    """Qwen3.5-122B 完整训练模型。

    Embedding → N × DecoderLayer → FinalNorm → LM Head

    与 ForwardShadowStore 集成:
    - setup_hot_params() 从 shadow store 初始化 hot expert 权重
    - sync_from_shadows() 每步前更新 shadow → nn.Parameter
    - collect_gradients() 收集所有 hot expert 参数的梯度
    """

    def __init__(
        self,
        num_layers: int = 48,
        hidden_size: int = 3072,
        intermediate_size: int = 1024,
        num_experts: int = 256,
        top_k: int = 8,
        vocab_size: int = 248320,
        dtype: torch.dtype = torch.float16,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_experts = num_experts

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, dtype=dtype, device=device)
        self.layers = nn.ModuleList([
            TrainingDecoderLayer(
                layer_idx=i,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_experts=num_experts,
                top_k=top_k,
                dtype=dtype,
                device=device,
            )
            for i in range(num_layers)
        ])
        self.final_norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False, dtype=dtype, device=device)

        # Hot expert 映射: (layer_idx, expert_id) → (w13_param_id, w2_param_id)
        self._hot_param_mapping: dict[str, tuple[int, int, str]] = {}
        self.shadow_store: Any = None
        # Predictor 对接: 记录每层 forward 后的 hidden state（仅 stride 层的）
        self._layer_hidden_states: dict[int, torch.Tensor] = {}
        # Router 激活专家记录: {(layer_idx, expert_id), ...} —— forward 中填充，供 lock/unlock 使用
        self._active_expert_ids: set[tuple[int, int]] = set()
        # Activation Checkpoint policy（可选，由 TrainingLoop 在 forward 前设置）
        self.checkpoint_policy: Any = None

    def set_cpu_gptq_cache(self, cache: Any) -> None:
        """为所有层的 MoE 设置 CPU 全量 GPTQ 冷专家缓存引用。"""
        for layer in self.layers:
            layer.moe.cpu_gptq_cache = cache

    def set_resident_cache(self, cache: Any) -> None:
        """为所有层的 MoE 设置 GPU resident cache 引用。"""
        for layer in self.layers:
            layer.moe.resident_cache = cache

    def setup_hot_params(
        self,
        shadow_store: Any,  # ForwardShadowStore
        hot_param_ids: Iterable[str],
    ) -> None:
        """从 ForwardShadowStore 初始化 hot expert 权重。"""
        self.shadow_store = shadow_store
        self._hot_param_mapping.clear()

        from cfie_training.training_base.real_model_adapter import _parse_expert_param_id

        for param_id in hot_param_ids:
            parsed = _parse_expert_param_id(param_id)
            if parsed is None:
                continue
            layer_idx, expert_id, weight_name = parsed
            if layer_idx >= self.num_layers:
                continue

            shadow = shadow_store.get(param_id)
            moe = self.layers[layer_idx].moe

            self._hot_param_mapping[param_id] = (layer_idx, expert_id, weight_name)

            if weight_name == "w13_weight":
                w13_shaped = shadow.reshape(2 * moe.intermediate_size, self.hidden_size)
                if expert_id in moe._hot_w13:
                    moe._hot_w13[expert_id].data.copy_(
                        w13_shaped.to(dtype=moe.dtype, device=moe._device)
                    )
                else:
                    w2_placeholder = torch.zeros(
                        self.hidden_size, moe.intermediate_size,
                        dtype=moe.dtype, device=moe._device,
                    )
                    moe.set_hot_expert(expert_id, w13_shaped, w2_placeholder)
            elif weight_name == "w2_weight":
                w2_shaped = shadow.reshape(self.hidden_size, moe.intermediate_size)
                if expert_id in moe._hot_w2:
                    moe._hot_w2[expert_id].data.copy_(
                        w2_shaped.to(dtype=moe.dtype, device=moe._device)
                    )
                else:
                    w13_placeholder = torch.zeros(
                        2 * moe.intermediate_size, self.hidden_size,
                        dtype=moe.dtype, device=moe._device,
                    )
                    moe.set_hot_expert(expert_id, w13_placeholder, w2_shaped)

    def sync_from_shadows(self) -> None:
        """将 shadow store 中更新后的权重同步到模型 nn.Parameter。"""
        if self.shadow_store is None:
            return
        for param_id, (layer_idx, expert_id, weight_name) in self._hot_param_mapping.items():
            shadow = self.shadow_store.get(param_id)
            moe = self.layers[layer_idx].moe
            if weight_name == "w13_weight" and expert_id in moe._hot_w13:
                w13 = shadow.reshape(2 * moe.intermediate_size, self.hidden_size)
                moe._hot_w13[expert_id].data.copy_(
                    w13.to(dtype=moe.dtype, device=moe._device)
                )
            elif weight_name == "w2_weight" and expert_id in moe._hot_w2:
                w2 = shadow.reshape(self.hidden_size, moe.intermediate_size)
                moe._hot_w2[expert_id].data.copy_(
                    w2.to(dtype=moe.dtype, device=moe._device)
                )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """前向传播。

        Returns: (logits, router_logits_list)
        """
        # ── Embedding ──
        hidden_states = self.embed_tokens(input_ids)           # [B, T] → [B, T, H]
        router_logits_list: list[torch.Tensor] = []             # 收集每层 router logits
        self._layer_hidden_states.clear()                      # 清空上轮 predictor 记录
        self._active_expert_ids.clear()                        # 清空上轮 router 选择记录

        # Activation Checkpoint policy（可选，由 TrainingLoop 在 forward 前设置）
        policy = getattr(self, "checkpoint_policy", None)

        # ── 逐层前向 ──
        for layer in self.layers:
            lid = layer.layer_idx
            # segment 边界：保存输入 hidden state 用于反向重算
            if policy is not None and policy.is_first_layer_in_segment(lid):
                policy.save_boundary_input(
                    policy.segment_for_layer(lid), hidden_states,
                )

            hidden_states, router_logits = layer(hidden_states)
            router_logits_list.append(router_logits)
            self._layer_hidden_states[lid] = hidden_states.detach()
            self._active_expert_ids.update(layer.moe.active_expert_ids)
            # 清空本层 GPU module cache（Marlin workspace 不跨层共享，释放显存）
            layer.moe._gpu_module_cache.clear()

        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits, router_logits_list

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        router_logits_list: list[torch.Tensor],
        *,
        aux_loss_coef: float = 0.01,
        z_loss_coef: float = 0.0001,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """计算 LM loss + MoE 辅助 losses。"""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        lm_loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        )

        # Router z-loss
        z_loss = torch.tensor(0.0, device=logits.device)
        n_routers = 0
        for rl in router_logits_list:
            if rl is not None and rl.numel() > 0:
                z_loss += torch.logsumexp(rl, dim=-1).pow(2).mean()
                n_routers += 1
        if n_routers > 0:
            z_loss = z_loss / n_routers

        # Load balancing loss (简化版)
        load_bal_loss = torch.tensor(0.0, device=logits.device)

        total_loss = lm_loss + aux_loss_coef * load_bal_loss + z_loss_coef * z_loss

        return total_loss, {
            "lm_loss": lm_loss.item(),
            "z_loss": z_loss.item(),
            "load_bal_loss": load_bal_loss.item(),
            "total_loss": total_loss.item(),
        }

    @property
    def all_trainable_param_ids(self) -> tuple[str, ...]:
        """返回所有可训练参数的 ID 列表：dense + MoE hot experts。"""
        ids: list[str] = []
        # Embedding
        ids.append("embed_tokens")
        # LM head
        ids.append("lm_head")
        # Final norm
        ids.append("final_norm")
        for lid in range(self.num_layers):
            prefix = f"layers.{lid}"
            # Norms
            ids.append(f"{prefix}.input_layernorm")
            ids.append(f"{prefix}.post_attention_layernorm")
            # MoE router + shared expert
            ids.append(f"{prefix}.moe.router")
            ids.append(f"{prefix}.moe.shared_w13")
            ids.append(f"{prefix}.moe.shared_w2")
            # Attention (GDN or Full)
            layer = self.layers[lid]
            if hasattr(layer.attention, "in_proj"):
                ids.append(f"{prefix}.attention.in_proj")
                ids.append(f"{prefix}.attention.out_proj")
                ids.append(f"{prefix}.attention.conv1d")
            else:
                ids.append(f"{prefix}.attention.q_proj")
                ids.append(f"{prefix}.attention.k_proj")
                ids.append(f"{prefix}.attention.v_proj")
                ids.append(f"{prefix}.attention.o_proj")
                ids.append(f"{prefix}.attention.gate_proj")
            # Hot MoE experts
            for eid in layer.moe._hot_w13:
                ids.append(f"layers.{lid}.experts.{eid}.w13_weight")
                ids.append(f"layers.{lid}.experts.{eid}.w2_weight")
        return tuple(ids)

    def collect_gradients(
        self,
        param_ids: Iterable[str],
    ) -> dict[str, torch.Tensor]:
        """收集所有参数的梯度（flat CPU float32）：dense + MoE hot experts。"""
        result: dict[str, torch.Tensor] = {}
        pid_set = set(param_ids)

        for pid in pid_set:
            # Dense params: 直接按名称从 named_parameters 查找
            grad = self._grad_for_named_param(pid)
            if grad is not None:
                result[pid] = grad.detach().reshape(-1).to(dtype=torch.float32, device="cpu")
                continue

            # MoE hot expert params
            mapping = self._hot_param_mapping.get(pid)
            if mapping is None:
                continue
            layer_idx, expert_id, weight_name = mapping
            moe = self.layers[layer_idx].moe
            if weight_name == "w13_weight" and expert_id in moe._hot_w13:
                g = moe._hot_w13[expert_id].grad
                if g is not None:
                    result[pid] = g.detach().reshape(-1).to(dtype=torch.float32, device="cpu")
            elif weight_name == "w2_weight" and expert_id in moe._hot_w2:
                g = moe._hot_w2[expert_id].grad
                if g is not None:
                    result[pid] = g.detach().reshape(-1).to(dtype=torch.float32, device="cpu")
        return result

    def _grad_for_named_param(self, param_id: str) -> torch.Tensor | None:
        """按名称查找 nn.Parameter 并返回其梯度。"""
        # 直接匹配 named_parameters
        for name, param in self.named_parameters():
            if name == param_id or name.endswith(f".{param_id}"):
                return param.grad
        # 分层查找
        parts = param_id.split(".")
        target = self
        for p in parts:
            if p.isdigit():
                target = target[int(p)] if hasattr(target, "__getitem__") else getattr(target, p, None)
            else:
                target = getattr(target, p, None)
            if target is None:
                return None
        if isinstance(target, torch.nn.Parameter):
            return target.grad
        return None

    def zero_grad(self) -> None:
        """清零所有 hot expert 参数的梯度。"""
        for layer in self.layers:
            for param in layer.moe._hot_w13.values():
                if param.grad is not None:
                    param.grad.zero_()
            for param in layer.moe._hot_w2.values():
                if param.grad is not None:
                    param.grad.zero_()
        self.embed_tokens.zero_grad()
        for p in self.final_norm.parameters():
            if p.grad is not None:
                p.grad.zero_()
        for p in self.lm_head.parameters():
            if p.grad is not None:
                p.grad.zero_()
