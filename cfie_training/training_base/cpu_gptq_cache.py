"""CPU 全量 GPTQ Int4 冷专家缓存（设计文档 Section 6.1）。

CPU 内存持有全部 256 routed experts 的 GPTQ Int4 权重作为冷专家权威运行时副本。
GPU resident cache 只是其子集。

职责:
- build_from_safetensors: 启动时从 safetensors 逐批量化构建全量缓存 → CPU
- get_expert: 返回 GPU-ready FP16 权重（用于 H2D 到 GPU resident cache）
- update_expert: 训练后将 FP32 master 重新量化为 Int4 并更新 CPU 缓存
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
import torch
from cfie_training.training_base.gpu_gptq import GpuGptqConfig, GpuGptqQuantizer


@dataclass(slots=True)
class CpuFullGptqCache:
    """CPU 全量 GPTQ Int4 冷专家缓存。

    存储格式: _entries[(layer,expert)] = (w13_packed_bytes, w13_sc_bytes, w2_packed_bytes, w2_sc_bytes)
    packed: gptq_pack 格式 [in_f/8, out_f] int32
    scales: fp16 [out_f, num_groups]，已转置为 (num_groups, out_f)
    """
    hidden_size: int = 3072                        # Qwen3.5-122B: 3072
    intermediate_size: int = 1024                  # MoE expert 中间维度
    num_layers: int = 48                           # Transformer 层数
    num_experts: int = 256                         # 每层 routed expert 数
    gptq_group_size: int = 128                     # GPTQ 量化 group 大小
    device: str = "cuda"                           # 量化时使用的 GPU 设备

    _entries: dict[tuple[int, int], tuple[bytes, bytes, bytes, bytes]] = field(default_factory=dict)
    _built: bool = False                           # 是否已构建完成

    @property
    def expert_count(self) -> int:
        """已缓存的专家总数。"""
        return len(self._entries)

    @property
    def total_bytes(self) -> int:
        """CPU 缓存的总字节数。"""
        return sum(len(e[0]) + len(e[1]) + len(e[2]) + len(e[3]) for e in self._entries.values())

    # ──────────── calibrate_and_build: 激活感知 GPTQ 构建 ────────────
    def calibrate_and_build(
        self, model: Any, *, calibration_batches: int = 4, seq_len: int = 8, progress: bool = True,
    ) -> int:
        """激活感知 GPTQ 构建（完整 Cholesky + 逐列补偿，最终导出用，较慢）。

        1. 前向 calibration 收集每层 MoE 输入激活
        2. 逐层从 safetensors 加载权重 → GPU 量化 → CPU 缓存
        """
        from safetensors import safe_open
        model.eval()
        device = next(model.parameters()).device

        # ── Step 1: Calibration forward（只收集激活，不跑 MoE 前向）──
        if progress: print(f"  [GPTQ] calibration: {calibration_batches} batches", flush=True)
        layer_activations: dict[int, list[torch.Tensor]] = {}       # {layer_idx: [act_tensors]}
        with torch.no_grad():
            for b in range(calibration_batches):
                x = torch.randint(0, 1000, (1, seq_len), device=device)  # 随机 token 校准
                h = model.embed_tokens(x)                                 # [1, T] → [1, T, H]
                for layer in model.layers:
                    lid = layer.layer_idx
                    h_norm_in = layer.input_layernorm(h)                 # pre-attention norm
                    attn_out = layer.attention(h_norm_in)                # attention 前向
                    h = h + attn_out                                     # 残差连接
                    h_norm = layer.post_attention_layernorm(h)           # post-attention norm → MoE 输入
                    layer_activations.setdefault(lid, []).append(
                        h_norm.reshape(-1, model.hidden_size).detach())  # 保存激活 [B*T, H]
                    h = h + torch.zeros_like(h)                          # 跳过 MoE（冷专家未加载）

        # ── Step 2: 逐层量化 ──
        if progress: print(f"  [GPTQ] per-layer quantize ({self.num_layers} layers)...", flush=True)
        quantizer = GpuGptqQuantizer(GpuGptqConfig(group_size=self.gptq_group_size))
        shards = sorted(str(p) for p in Path(
            "C:/Users/13642/.cache/huggingface/hub/models--Qwen--Qwen3.5-122B-A10B/"
            "snapshots/b000b2eb18a7f4cdf3153c4215842da339e09d99"
        ).glob("model*.safetensors") if p.suffix == ".safetensors")

        for lid in range(self.num_layers):
            layer_weights: dict[int, dict[str, torch.Tensor]] = {}      # {expert_id: {wtype: tensor}}
            for sp in shards:
                with safe_open(sp, framework="pt") as f:
                    for key in f.keys():
                        parsed = self._parse_checkpoint_key(key)
                        if parsed is None or parsed[0] != lid: continue  # 跳过非本层
                        _, wtype = parsed
                        t = f.get_tensor(key)                            # [256, ...] packed expert
                        for eid in range(t.shape[0]):
                            layer_weights.setdefault(eid, {})[wtype] = t[eid].float()  # [out_f, in_f]

            acts = torch.cat(layer_activations.get(lid, []), dim=0).to(device)  # [N, H] 本层校准激活

            for eid in range(self.num_experts):
                w = layer_weights.get(eid, {})
                if "gate_up_proj" not in w or "down_proj" not in w: continue

                # w13: gate_up → 拼成 [2*I, H]
                t = w["gate_up_proj"]                                      # [2*I, H]
                gate = t[:self.intermediate_size, :]                       # [I, H]
                up = t[self.intermediate_size:, :]                         # [I, H]
                w13 = torch.cat([gate.reshape(-1), up.reshape(-1)]).reshape(2 * self.intermediate_size, self.hidden_size).to(device)
                w2 = w["down_proj"].reshape(self.hidden_size, self.intermediate_size).to(device)  # [H, I]

                # 激活感知量化 w13
                quantizer.reset(); quantizer.collect_activations(acts)
                qw13, sc13, _ = quantizer.quantize(w13)

                # 激活感知量化 w2（SwiGLU 输出作为激活）
                quantizer.reset()
                gate_up = torch.nn.functional.linear(acts.to(w13.dtype), w13)  # [N, 2*I]
                g, u = gate_up.chunk(2, dim=-1)                                 # [N, I], [N, I]
                w2_acts = (torch.nn.functional.silu(g) * u)                     # SwiGLU 激活
                quantizer.collect_activations(w2_acts)
                qw2, sc2, _ = quantizer.quantize(w2)

                self._entries[(lid, eid)] = (qw13.cpu().numpy().tobytes(), sc13.cpu().numpy().tobytes(),
                                            qw2.cpu().numpy().tobytes(), sc2.cpu().numpy().tobytes())

            if progress and lid % 8 == 0: print(f"    layer {lid}/{self.num_layers} | {len(self._entries)} experts", flush=True)

        self._built = True
        return len(self._entries)

    # ──────────── build_from_safetensors: 快速 absmax 量化构建 ────────────
    def build_from_safetensors(
        self, checkpoint_dir: str | Path, *, batch_layers: int = 4, progress: bool = True,
    ) -> int:
        """从 safetensors 逐批 GPU 量化 → CPU 缓存（per-group absmax，毫秒级）。

        使用 safetensors index 过滤 shard，减少 IO。
        返回缓存的专家数。
        """
        from safetensors import safe_open
        cp = Path(checkpoint_dir)
        all_shards = sorted(p for p in cp.glob("model*.safetensors") if p.suffix == ".safetensors")
        if not all_shards: raise FileNotFoundError(f"no safetensors found in {cp}")

        # index.json 过滤：只打开包含目标层的 shard
        index_path = cp / "model.safetensors.index.json"
        if index_path.exists():
            import json as _json
            with open(index_path) as _f: idx = _json.load(_f)          # 读取 weight_map
            needed = set()                                              # 收集需要的 shard 文件名
            target_layers = set(range(self.num_layers))
            for k, v in idx["weight_map"].items():
                if "mlp.experts" in k:
                    parts = k.split(".")
                    for i, p in enumerate(parts):
                        if p == "layers":
                            lid = int(parts[i + 1])
                            if lid in target_layers: needed.add(Path(v).name)
                            break
            shards = [p for p in all_shards if p.name in needed]       # 过滤后的 shard 列表
        else:
            shards = all_shards
        if not shards: raise FileNotFoundError(f"no safetensors found in {cp}")

        quantizer = GpuGptqQuantizer(GpuGptqConfig(group_size=self.gptq_group_size))

        for batch_start in range(0, self.num_layers, batch_layers):        # 逐批处理
            batch_end = min(batch_start + batch_layers, self.num_layers)
            batch_layers_set = set(range(batch_start, batch_end))

            # 收集本批所有专家权重
            experts_in_batch: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
            for sp in shards:
                with safe_open(str(sp), framework="pt") as f:
                    for key in f.keys():
                        parsed = self._parse_checkpoint_key(key)            # 解析 layer/wtype
                        if parsed is None: continue
                        lid, wtype = parsed
                        if lid not in batch_layers_set: continue           # 跳过非本批
                        tensor = f.get_tensor(key)                         # [256, out_f, in_f]
                        for eid in range(tensor.shape[0]):
                            experts_in_batch.setdefault((lid, eid), {})[wtype] = tensor[eid].float()

            # GPU 量化 → gptq_pack → CPU bytes
            for (lid, eid), weights in experts_in_batch.items():
                w13_fp32 = self._build_w13(weights)                        # gate_up → [2*I, H]
                w2_fp32 = self._build_w2(weights)                          # down → [H, I]
                if w13_fp32 is not None:
                    logical, sc = quantizer.quantize_logical(w13_fp32.to(self.device))  # GPU 量化
                    from cfie.model_executor.layers.quantization.utils.quant_utils import gptq_pack
                    out_f, in_f = logical.shape                            # [2*I, H]
                    packed = gptq_pack((logical.T.contiguous() + 8).to(torch.int32), 4, in_f, out_f)  # 转置+pack
                    self._entries[(lid, eid)] = (packed.cpu().numpy().tobytes(), sc.T.contiguous().cpu().numpy().tobytes(), b"", b"")
                if w2_fp32 is not None:
                    logical, sc = quantizer.quantize_logical(w2_fp32.to(self.device))
                    from cfie.model_executor.layers.quantization.utils.quant_utils import gptq_pack
                    out_f, in_f = logical.shape                            # [H, I]
                    packed = gptq_pack((logical.T.contiguous() + 8).to(torch.int32), 4, in_f, out_f)
                    if (lid, eid) in self._entries:
                        entry = list(self._entries[(lid, eid)])            # 更新已有 entry 的 w2 部分
                        entry[2] = packed.cpu().numpy().tobytes()
                        entry[3] = sc.T.contiguous().cpu().numpy().tobytes()
                        self._entries[(lid, eid)] = tuple(entry)

            if progress:
                pct = batch_end / self.num_layers * 100
                print(f"  [CpuFullGptqCache] 层 {batch_start}-{batch_end-1} 量化完成 ({pct:.0f}%) | {len(self._entries)} experts", flush=True)

        self._built = True
        return len(self._entries)

    # ──────────── get_expert: CPU 缓存 → GPU-ready FP16 ────────────
    def get_expert(self, layer_id: int, expert_id: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """返回 (w13_fp16, w2_fp16) 用于 H2D 到 GPU resident cache。

        w13: [2*I, H] fp16, w2: [H, I] fp16
        """
        entry = self._entries.get((layer_id, expert_id))                  # 查找缓存
        if entry is None: return None
        w13_qb, w13_sb, w2_qb, w2_sb = entry                             # 拆包 4 个 bytes 字段
        if not w13_qb or not w2_qb: return None                           # 不完整数据

        w13 = self._decode_one(w13_qb, w13_sb, out_features=2 * self.intermediate_size, in_features=self.hidden_size)  # [2*I, H] fp16
        w2 = self._decode_one(w2_qb, w2_sb, out_features=self.hidden_size, in_features=self.intermediate_size)          # [H, I] fp16
        return w13, w2

    # ──────────── update_expert: 训练后重量化 ────────────
    def update_expert(self, layer_id: int, expert_id: int, w13_fp32: torch.Tensor, w2_fp32: torch.Tensor) -> None:
        """训练后将 FP32 master 重新量化为 Int4，更新 CPU 缓存（设计文档 Section 4.2 冷专家生命周期）。"""
        quantizer = GpuGptqQuantizer(GpuGptqConfig(group_size=self.gptq_group_size))
        qw13, sc13, _ = quantizer.quantize(w13_fp32.reshape(2 * self.intermediate_size, self.hidden_size).to(self.device))  # [2*I,H] → 量化
        qw2, sc2, _ = quantizer.quantize(w2_fp32.reshape(self.hidden_size, self.intermediate_size).to(self.device))         # [H,I] → 量化
        self._entries[(layer_id, expert_id)] = (qw13.cpu().numpy().tobytes(), sc13.cpu().numpy().tobytes(),
                                                 qw2.cpu().numpy().tobytes(), sc2.cpu().numpy().tobytes())

    # ──────────── 内部辅助 ────────────
    def _decode_one(self, qw_bytes: bytes, sc_bytes: bytes, *, out_features: int, in_features: int) -> torch.Tensor:
        """单个 Int4 packed weight → FP16 tensor。"""
        qw = torch.frombuffer(bytearray(qw_bytes), dtype=torch.uint8)     # bytes → uint8 tensor
        sc = torch.frombuffer(bytearray(sc_bytes), dtype=torch.float16)    # bytes → fp16 tensor
        qw_2d = qw.reshape(out_features, -1)                               # [out_f, in_f//2] packed uint8
        sc_2d = sc.reshape(out_features, -1)                               # [out_f, num_groups] fp16
        return GpuGptqQuantizer.decode(qw_2d, sc_2d, torch.empty(0, dtype=torch.uint8),
                                        out_features=out_features, in_features=in_features, group_size=self.gptq_group_size)

    def _build_w13(self, weights: dict[str, torch.Tensor]) -> torch.Tensor | None:
        """gate_up_proj: [2*I, H] → 拼成 flat → [2*I, H] fp32。"""
        if "gate_up_proj" not in weights: return None
        t = weights["gate_up_proj"]                                        # [2*I, H]
        gate = t[:self.intermediate_size, :]                               # [I, H]
        up = t[self.intermediate_size:, :]                                 # [I, H]
        return torch.cat([gate.reshape(-1), up.reshape(-1)]).reshape(2 * self.intermediate_size, self.hidden_size)

    def _build_w2(self, weights: dict[str, torch.Tensor]) -> torch.Tensor | None:
        """down_proj: [H, I] → [H, I] fp32。"""
        if "down_proj" not in weights: return None
        return weights["down_proj"].reshape(self.hidden_size, self.intermediate_size)

    @staticmethod
    def _parse_checkpoint_key(name: str) -> tuple[int, str] | None:
        """解析 safetensors key: model.language_model.layers.{N}.mlp.experts.{gate_up_proj|down_proj} → (layer_id, wtype)。"""
        prefix = "model.language_model.layers."
        if not name.startswith(prefix): return None
        rest = name[len(prefix):].split(".", 3)                            # [N, mlp, experts, wtype]
        if len(rest) < 4: return None
        try: lid = int(rest[0])
        except ValueError: return None
        if rest[1] != "mlp" or rest[2] != "experts": return None
        if rest[3] in ("gate_up_proj", "down_proj"): return lid, rest[3]
        return None
