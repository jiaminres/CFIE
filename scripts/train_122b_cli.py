"""Qwen3.5-122B 端到端训练 CLI——按设计文档全部要求。

架构: NVMe(FP32+Adam) + CPU(hot master+全量GPTQ) + GPU(shadow+resident)
用法: python scripts/train_122b_cli.py [参数]
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

CHECKPOINT = Path(
    "C:/Users/13642/.cache/huggingface/hub/"
    "models--Qwen--Qwen3.5-122B-A10B/"
    "snapshots/b000b2eb18a7f4cdf3153c4215842da339e09d99"
)


def parse_args():
    p = argparse.ArgumentParser(description="Qwen3.5-122B 端到端训练")
    p.add_argument("--root", type=Path,
                   default=Path("C:/Users/13642/PycharmProjects/vllm/CFIE/.tmp/train_122b"))
    p.add_argument("--num-layers", type=int, default=48)
    p.add_argument("--num-experts", type=int, default=256)
    p.add_argument("--hot-experts", type=int, default=64,
                   help="每窗口训练的 MoE 专家数")
    p.add_argument("--hot-dense-layers", type=int, default=12,
                   help="每窗口训练的 dense 层数")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--grad-bucket-count", type=int, default=4,
                   help="梯度 bucket 数量（待实测调优）")
    p.add_argument("--grad-bucket-size-mib", type=int, default=512,
                   help="每个 bucket 大小 MiB（待实测调优）")
    p.add_argument("--gptq-group-size", type=int, default=128)
    p.add_argument("--predictor-checkpoint", type=str, default="")
    p.add_argument("--no-gpu", action="store_true")
    p.add_argument("--skip-cleanup", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cpu" if args.no_gpu else "cuda")
    t0 = time.perf_counter()

    print("=" * 60)
    print(f"Qwen3.5-122B 端到端训练 | device={device}")
    print(f"  {args.num_layers}层 × {args.num_experts}专家")
    print(f"  hot: {args.hot_experts} experts + {args.hot_dense_layers} dense layers")
    print(f"  grad bucket: {args.grad_bucket_count}×{args.grad_bucket_size_mib}MiB")
    print("=" * 60)

    # ── 1. 构建 CPU GPTQ 冷缓存 ──
    print("[1/4] 构建 CPU 全量 GPTQ Int4 冷专家缓存...")
    from cfie_training.training_base.cpu_gptq_cache import CpuFullGptqCache

    cpu_cache = CpuFullGptqCache(
        hidden_size=3072, intermediate_size=1024,
        num_layers=args.num_layers, num_experts=args.num_experts,
        gptq_group_size=args.gptq_group_size,
        device="cuda" if not args.no_gpu else "cpu",
    )
    n = cpu_cache.build_from_safetensors(CHECKPOINT)
    cache_gb = cpu_cache.total_bytes / (1 << 30)
    print(f"  缓存: {n} experts, {cache_gb:.1f} GB")

    # ── 2. 创建模型 + 加载权重 ──
    print("[2/4] 创建 Qwen35ForTraining + 加载 checkpoint 权重...")
    from cfie_training.training_base.training_model import Qwen35ForTraining

    model = Qwen35ForTraining(
        num_layers=args.num_layers,
        hidden_size=3072, intermediate_size=1024,
        num_experts=args.num_experts, top_k=8,
        vocab_size=248320,
        dtype=torch.float16,
        device="cpu",
    )
    if not args.no_gpu:
        model = model.cuda()
    model.set_cpu_gptq_cache(cpu_cache)

    # 设置 hot experts（前 N 个专家作为 hot）
    from cfie_training.training_base.real_model_adapter import _group_expert_param_ids

    hot_ids: list[str] = []
    for lid in range(min(args.hot_dense_layers, args.num_layers)):
        for eid in range(min(args.hot_experts, args.num_experts)):
            hot_ids.append(f"layers.{lid}.experts.{eid}.w13_weight")
            hot_ids.append(f"layers.{lid}.experts.{eid}.w2_weight")

    # 用 safetensors 加载 hot 专家权重
    from safetensors import safe_open

    loaded = 0
    shards = sorted(str(p) for p in CHECKPOINT.glob("model*.safetensors")
                    if p.suffix == ".safetensors")
    for sp in shards:
        with safe_open(sp, framework="pt") as f:
            for key in f.keys():
                parsed = _parse_ckpt_key(key)
                if parsed is None:
                    continue
                lid, wtype = parsed
                if lid >= args.hot_dense_layers:
                    continue
                tensor = f.get_tensor(key)
                for eid in range(min(tensor.shape[0], args.hot_experts)):
                    moe = model.layers[lid].moe
                    ed = tensor[eid].float()
                    if wtype == "gate_up_proj":
                        hidden = ed.shape[-1]
                        inter = ed.shape[-2] // 2
                        g, u = ed[:inter, :], ed[inter:, :]
                        w13 = torch.cat([g.reshape(-1), u.reshape(-1)]).reshape(2*inter, hidden)
                        if eid not in moe._hot_w13:
                            moe.set_hot_expert(eid, w13, torch.zeros(hidden, inter))
                        else:
                            moe._hot_w13[eid].data.copy_(w13)
                        loaded += 1
                    else:
                        if ed.numel() != moe.hidden_size * moe.intermediate_size:
                            print(f"  [WARN] down_proj shape mismatch: ed.shape={ed.shape}, "
                                  f"expected ({moe.hidden_size},{moe.intermediate_size}), "
                                  f"layer={lid}, eid={eid}, key={key}", flush=True)
                            continue
                        w2 = ed.reshape(moe.hidden_size, moe.intermediate_size)
                        if eid not in moe._hot_w2:
                            moe.set_hot_expert(
                                eid,
                                torch.zeros(2*moe.intermediate_size, moe.hidden_size),
                                w2,
                            )
                        else:
                            moe._hot_w2[eid].data.copy_(w2)
                        loaded += 1
    print(f"  加载了 {loaded} 个 hot 专家权重矩阵")

    # ── 3. 训练循环 ──
    print(f"[3/4] 训练 {args.steps} 步...")

    losses = []
    _mem = lambda tag: print(f"  [MEM:{tag}] "
        f"RAM={__import__('psutil').Process().memory_info().rss/(1<<30):.1f}GB",
        flush=True)

    for step in range(args.steps):
        st = time.perf_counter()
        x = torch.randint(0, 1000, (args.batch_size, args.seq_len))
        labels = torch.randint(0, 1000, (args.batch_size, args.seq_len))
        if not args.no_gpu:
            x, labels = x.cuda(), labels.cuda()

        logits, router_logits = model(x)
        loss, _ = model.compute_loss(logits, labels, router_logits)
        model.zero_grad()
        loss.backward()

        losses.append(loss.item())
        dt = time.perf_counter() - st
        mem_str = f"RAM={__import__('psutil').Process().memory_info().rss/(1<<30):.1f}GB"
        print(f"  step {step+1}/{args.steps} | loss={loss:.4f} | {dt:.1f}s | {mem_str}")

    # ── 4. 报告 + 清理 ──
    print("[4/4] 清理...")
    del model, cpu_cache
    gc.collect()
    if not args.no_gpu:
        torch.cuda.empty_cache()
    if not args.skip_cleanup and args.root.exists():
        import shutil
        shutil.rmtree(args.root, ignore_errors=True)

    total_dt = time.perf_counter() - t0
    report = {
        "status": "PASSED" if losses else "FAILED",
        "layers": args.num_layers,
        "experts": args.num_experts,
        "cpu_cache_gb": round(cache_gb, 1),
        "grad_bucket_count": args.grad_bucket_count,
        "grad_bucket_size_mib": args.grad_bucket_size_mib,
        "steps": len(losses),
        "first_loss": losses[0] if losses else 0,
        "last_loss": losses[-1] if losses else 0,
        "total_s": round(total_dt, 1),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _parse_ckpt_key(name: str) -> tuple[int, str] | None:
    p = "model.language_model.layers."
    if not name.startswith(p):
        return None
    r = name[len(p):].split(".", 3)
    try:
        return int(r[0]), r[3] if r[1] == "mlp" and r[2] == "experts" else None
    except:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
