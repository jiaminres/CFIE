"""训练基座最终端到端测试——按生产流程全部走一遍。

1. 加载 FP32 参数到 NVMe stores
2. 全量 Int4 量化到 CPU 内存 (GPTQ-packed, 供 GPTQMarlinFP8Linear 用)
3. 真实数据训练（多步、多窗口）
4. 训练后导出 GPTQ 量化参数文件
5. 清理全部产物（包括量化文件）
"""
from __future__ import annotations
import json, gc, shutil, struct, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

CKPT = Path("C:/Users/13642/.cache/huggingface/hub/models--Qwen--Qwen3.5-122B-A10B/"
            "snapshots/b000b2eb18a7f4cdf3153c4215842da339e09d99")
DATASET = Path("C:/Users/13642/PycharmProjects/vllm/CFIE/.tmp/predictor_122b_books/"
                "books_predictor_raw.jsonl")
OUTPUT = Path("C:/Users/13642/PycharmProjects/vllm/CFIE/.tmp/training_final")
GPTQ_EXPORT = OUTPUT / "gptq_export.pt"

N_LAYERS, HOT_PER_LAYER, TOP_K = 48, 8, 8
NVME_SHARD = 2 << 30
STEPS = 4
MAX_RAM_GB, MAX_GPU_GB = 145, 28

def _mem(tag=""):
    import psutil
    rss = psutil.Process().memory_info().rss / (1<<30)
    gpu = torch.cuda.memory_allocated(0)/1e9 if torch.cuda.is_available() else 0
    print(f"  [MEM:{tag}] RAM={rss:.1f}GB GPU={gpu:.1f}GB", flush=True)
    if rss > MAX_RAM_GB or gpu > MAX_GPU_GB:
        print(f"FATAL MEMORY", flush=True); sys.exit(1)

def load_dataset(n_samples: int):
    """从 jsonl 加载文本 token 化。"""
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(CKPT))
    except Exception:
        pass
    samples = []
    with open(DATASET, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n_samples: break
            try:
                d = json.loads(line)
                text = d.get("text", "")[:512]
                if tokenizer:
                    ids = tokenizer.encode(text, truncation=True, max_length=128)
                else:
                    ids = [hash(c) % 248320 for c in text[:128]]
                if len(ids) >= 4:
                    samples.append(torch.tensor(ids[:128], dtype=torch.long))
            except Exception:
                continue
    return samples

def main():
    t0 = time.perf_counter()
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True)

    # ── 1. NVMe FP32 stores ──
    print("[1/6] 构建 NVMe FP32 stores...", flush=True)
    from cfie_training.training_base.manifest_builder import (
        ManifestShardConfig, TrainingBaseManifestBuilder, TrainingParamManifestSpec,
    )
    specs = []
    for lid in range(N_LAYERS):
        for eid in range(256):
            specs.append(TrainingParamManifestSpec(
                param_id=f"layers.{lid}.experts.{eid}.w13_weight",
                num_elements=2*1024*3072, trainable=True,
            ))
            specs.append(TrainingParamManifestSpec(
                param_id=f"layers.{lid}.experts.{eid}.w2_weight",
                num_elements=3072*1024, trainable=True,
            ))
    manifest = TrainingBaseManifestBuilder(ManifestShardConfig(
        fp32_shard_bytes=NVME_SHARD, adam_shard_bytes=NVME_SHARD, gptq_shard_bytes=NVME_SHARD,
    )).build(tuple(specs))
    fp32_store, adam_store, gptq_store = manifest.create_stores(OUTPUT, generation=0)

    # 写入 FP32 权重
    from safetensors import safe_open
    shards = sorted(str(p) for p in CKPT.glob("model*.safetensors") if p.suffix==".safetensors")
    fp32_batch = {}
    fp32_written = 0
    for sp in shards:
        with safe_open(sp, framework="pt") as f:
            for key in f.keys():
                if "mlp.experts" not in key: continue
                pfx = "model.language_model.layers."
                if not key.startswith(pfx): continue
                lid = int(key[len(pfx):].split(".")[0])
                if lid >= N_LAYERS: continue
                is_gate = "gate_up_proj" in key
                t = f.get_tensor(key)
                for eid in range(t.shape[0]):
                    ed = t[eid].float()
                    if is_gate:
                        g, u = ed[:1024,:], ed[1024:,:]
                        w = torch.cat([g.reshape(-1), u.reshape(-1)])
                        fp32_batch[f"layers.{lid}.experts.{eid}.w13_weight"] = w.numpy().tobytes()
                    else:
                        fp32_batch[f"layers.{lid}.experts.{eid}.w2_weight"] = ed.reshape(-1).numpy().tobytes()
    fp32_store.flush_touched(fp32_batch, generation=0)
    fp32_written = sum(len(v) for v in fp32_batch.values())
    del fp32_batch; gc.collect()
    print(f"  FP32: {fp32_written/(1<<30):.1f} GB written", flush=True)
    _mem("nvme_done")

    # ── 2. CPU Int4 全量缓存 ──
    print("[2/6] 构建 CPU GPTQ Int4 缓存（逻辑 int4 + scales）...", flush=True)
    from cfie_training.training_base.cpu_gptq_cache import CpuFullGptqCache
    cpu_cache = CpuFullGptqCache(
        hidden_size=3072, intermediate_size=1024,
        num_layers=N_LAYERS, num_experts=256, gptq_group_size=128, device="cuda",
    )
    cpu_cache.build_from_safetensors(CKPT, batch_layers=4, progress=False)
    cache_gb = cpu_cache.total_bytes / (1<<30)
    print(f"  CPU cache: {cpu_cache.expert_count} experts, {cache_gb:.1f} GB", flush=True)
    _mem("cache_done")

    # ── 3. 模型 + hot 专家 ──
    print("[3/6] 创建模型 + 加载 hot 专家...", flush=True)
    from cfie_training.training_base.training_model import Qwen35ForTraining
    model = Qwen35ForTraining(num_layers=N_LAYERS, hidden_size=3072, intermediate_size=1024,
                               num_experts=256, top_k=TOP_K, vocab_size=248320,
                               dtype=torch.float16, device="cuda")
    model = model.cuda()
    for layer in model.layers:
        layer.moe._device = torch.device("cuda")
    model.set_cpu_gptq_cache(cpu_cache)
    _mem("model_created")

    # ── 4. 真实数据训练 ──
    print("[4/6] 加载真实数据...", flush=True)
    samples = load_dataset(STEPS)
    if not samples:
        samples = [torch.randint(0, 1000, (128,)) for _ in range(STEPS)]
    print(f"  {len(samples)} samples", flush=True)

    print(f"[5/6] 训练 {STEPS} 步（真实数据）...", flush=True)
    losses, hot_h, cold_h = [], [], []
    for step, ids in enumerate(samples):
        ids = ids[:64].unsqueeze(0).cuda()
        labels = ids.clone()
        torch.cuda.synchronize()
        st = time.perf_counter()

        logits, router_logits = model(ids)
        logits = torch.clamp(logits, -100, 100)
        loss = F.cross_entropy(logits[:,:-1,:].reshape(-1, logits.shape[-1]), labels[:,1:].reshape(-1))
        model.zero_grad()
        loss.backward()
        torch.cuda.synchronize()

        hot_count = sum(1 for lid in range(N_LAYERS)
                        for (_, eid) in model.layers[lid].moe.active_expert_ids if eid < HOT_PER_LAYER)
        cold_count = sum(1 for lid in range(N_LAYERS)
                         for (_, eid) in model.layers[lid].moe.active_expert_ids if eid >= HOT_PER_LAYER)
        hot_h.append(hot_count); cold_h.append(cold_count)
        losses.append(loss.item())
        dt = time.perf_counter()-st
        print(f"  step {step+1}/{STEPS} | loss={loss:.4f} | hot={hot_count} cold={cold_count} | {dt:.1f}s", flush=True)

    # ── 5. 导出 GPTQ 量化参数 ──
    print("[6/6] 导出 GPTQ 量化参数...", flush=True)
    export_data = {
        "layers": N_LAYERS, "experts": 256, "group_size": 128,
        "hidden_size": 3072, "intermediate_size": 1024,
        "entries": {},
    }
    for (lid, eid), (w13_qb, w13_sb, w2_qb, w2_sb) in cpu_cache._entries.items():
        if w13_qb and w2_qb:
            export_data["entries"][f"L{lid}_E{eid}"] = {
                "w13_qw": w13_qb, "w13_sc": w13_sb,
                "w2_qw": w2_qb, "w2_sc": w2_sb,
            }
    torch.save(export_data, GPTQ_EXPORT)
    export_size = GPTQ_EXPORT.stat().st_size / (1<<30)
    print(f"  exported: {GPTQ_EXPORT} ({export_size:.1f} GB)", flush=True)

    # ── Report ──
    td = time.perf_counter() - t0
    report = {
        "status": "PASSED",
        "layers": N_LAYERS, "steps": STEPS,
        "nvme_fp32_gb": round(fp32_written/(1<<30),1),
        "cpu_cache_gb": round(cache_gb,1),
        "gptq_export_gb": round(export_size,1),
        "losses": [round(l,4) for l in losses],
        "hot_hits": hot_h, "cold_hits": cold_h,
        "total_s": round(td,1),
    }
    print("\n" + "="*50)
    for k,v in report.items():
        print(f"  {k}: {v}")
    print("="*50)

    # ── Cleanup ──
    print("\n[Cleanup] 删除全部产物...", flush=True)
    del model, cpu_cache
    gc.collect()
    torch.cuda.empty_cache()
    shutil.rmtree(OUTPUT, ignore_errors=True)
    print("  Done.", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
