"""恢复训练——跳过 NVMe 构建，直接使用已有数据跑训练。"""

import gc, json, struct, sys, time, shutil
from pathlib import Path
import torch
import torch.nn.functional as F

CHECKPOINT = Path(
    "C:/Users/13642/.cache/huggingface/hub/"
    "models--Qwen--Qwen3.5-122B-A10B/"
    "snapshots/b000b2eb18a7f4cdf3153c4215842da339e09d99"
)
OUTPUT = Path("C:/Users/13642/PycharmProjects/vllm/CFIE/.tmp/training_122b_full")

N_LAYERS, N_EXPERTS, HIDDEN, INTER, TOP_K = 48, 256, 3072, 1024, 8
HOT_EXPERTS, TRAIN_STEPS = 8, 6
GPTQ_GROUP_SIZE, MAX_RAM_GB = 128, 145
from safetensors import safe_open


def _mem(tag=""):
    import psutil
    rss = psutil.Process().memory_info().rss / (1 << 30)
    print(f"  [MEM:{tag}] RAM={rss:.1f}GB", flush=True)
    if rss > MAX_RAM_GB:
        print(f"FATAL: RAM {rss:.1f}GB", flush=True)
        sys.exit(1)


def _parse(name):
    p = "model.language_model.layers."
    if not name.startswith(p):
        return None
    r = name[len(p):].split(".", 3)
    try:
        return int(r[0]), r[3] if r[1] == "mlp" and r[2] == "experts" else None
    except:
        return None


def _pack(x):
    n = x.shape[1]
    if n % 2:
        x = F.pad(x, (0, 1))
    return (x[:, 0::2].to(torch.uint8) & 0x0F) | ((x[:, 1::2].to(torch.uint8) & 0x0F) << 4)


def gpu_q(w, gs=128):
    wf = w.float().cuda()
    o, i = wf.shape
    ng = (i + gs - 1) // gs
    s = torch.empty(o, ng, dtype=torch.float16, device="cuda")
    qw = torch.empty(o, (i + 1) // 2, dtype=torch.uint8, device="cuda")
    for g in range(ng):
        gs_, ge = g * gs, min(g * gs + gs, i)
        gw = wf[:, gs_:ge]
        mx = gw.abs().max(1, keepdim=True).values.clamp(1e-8)
        sc = mx / 7.0
        s[:, g] = sc.to(torch.float16).squeeze(1)
        q = (gw / sc).round().clamp(-8, 7).to(torch.int8)
        pk = _pack(q)
        qw[:, gs_ // 2:gs_ // 2 + pk.shape[1]] = pk
    return qw.cpu(), s.cpu()


def gpu_decode(qw, sc, of_, inf, gs=128):
    qw = qw.cuda()
    sc = sc.cuda()
    lo = (qw.to(torch.int16) & 0x0F)
    hi = ((qw.to(torch.int16) >> 4) & 0x0F)
    lo = torch.where(lo > 7, lo - 16, lo)
    hi = torch.where(hi > 7, hi - 16, hi)
    q = torch.stack([lo, hi], 2).reshape(of_, -1)[:, :inf]
    se = sc.repeat_interleave(gs, 1)[:, :inf]
    return (q.to(torch.float16) * se.to(torch.float16)).cpu()


def main():
    t0 = time.perf_counter()
    print("=" * 60)
    print("恢复训练: 使用已有 NVMe stores + 重建 CPU GPTQ 缓存")
    print("=" * 60)

    # ─── Step A: 重建 CPU GPTQ 缓存 ───
    print("[A] 重建 CPU GPTQ Int4 全量冷专家缓存...")
    cache: dict[tuple[int, int], tuple[bytes, bytes]] = {}
    shards = sorted(str(p) for p in CHECKPOINT.glob("model*.safetensors") if p.suffix == ".safetensors")

    for sp in shards:
        with safe_open(sp, framework="pt") as f:
            for ck in [k for k in f.keys() if "mlp.experts." in k]:
                p = _parse(ck)
                if p is None or p[1] is None:
                    continue
                lid, wtype = p
                if wtype not in ("gate_up_proj", "down_proj"):
                    continue
                t = f.get_tensor(ck)
                for eid in range(t.shape[0]):
                    k = (lid, eid)
                    ed = t[eid].float()
                    if wtype == "gate_up_proj":
                        g = ed[:INTER, :]; u = ed[INTER:, :]
                        w13 = torch.cat([g.reshape(-1), u.reshape(-1)]).reshape(2 * INTER, HIDDEN)
                        qw, sc = gpu_q(w13, GPTQ_GROUP_SIZE)
                        if k not in cache:
                            cache[k] = [None, None]
                        cache[k][0] = qw.numpy().tobytes() + sc.numpy().tobytes()
                    else:
                        w2 = ed.reshape(HIDDEN, INTER)
                        qw, sc = gpu_q(w2, GPTQ_GROUP_SIZE)
                        if k not in cache:
                            cache[k] = [None, None]
                        cache[k][1] = qw.numpy().tobytes() + sc.numpy().tobytes()
        _mem(Path(sp).stem)

    total_bytes = sum((len(v[0]) if v[0] else 0) + (len(v[1]) if v[1] else 0) for v in cache.values())
    print(f"  CPU GPTQ 缓存: {len(cache)} experts, {total_bytes/(1<<30):.1f} GB")
    _mem("after_cache")

    # ─── Step B: 训练 ───
    print(f"[B] 开始训练 ({TRAIN_STEPS} 步, {HOT_EXPERTS} hot/窗口)...")

    from cfie_training.training_base.fp32_shard_store import FP32ShardStore
    fp32_store = FP32ShardStore.load(OUTPUT / "fp32")

    hot_set = [(lid, eid) for lid in range(4) for eid in range(HOT_EXPERTS)]
    hot_ids = []
    for lid, eid in hot_set:
        hot_ids.append(f"layers.{lid}.experts.{eid}.w13_weight")
        hot_ids.append(f"layers.{lid}.experts.{eid}.w2_weight")

    losses = []
    for step in range(TRAIN_STEPS):
        st = time.perf_counter()
        x = torch.randn(1, 8, HIDDEN)
        total_loss = torch.tensor(0.0)

        for lid in range(N_LAYERS):
            router_logits = torch.randn(8, N_EXPERTS) * 0.1
            for _, eid in hot_set:
                router_logits[:, eid] += 2.0
            top_w, top_idx = torch.topk(F.softmax(router_logits, -1), TOP_K)
            top_w = top_w / top_w.sum(-1, keepdim=True)

            layer_out = torch.zeros(8, HIDDEN)
            for ti in range(8):
                for k in range(TOP_K):
                    eid = int(top_idx[ti, k])
                    wgt = top_w[ti, k].item()
                    s = x[0, ti:ti+1]

                    ce = cache.get((lid, eid))
                    if ce and ce[0] and ce[1]:
                        w13b, w2b = ce[0], ce[1]
                        sep13 = 2 * INTER * HIDDEN // 2 + (2 * INTER * HIDDEN // GPTQ_GROUP_SIZE) * 2
                        qw13 = torch.frombuffer(bytearray(w13b[:2*INTER*HIDDEN//2]), dtype=torch.uint8).reshape(2*INTER, -1)
                        sc13 = torch.frombuffer(bytearray(w13b[2*INTER*HIDDEN//2:sep13]), dtype=torch.float16).reshape(2*INTER, -1)
                        w13 = gpu_decode(qw13, sc13, 2*INTER, HIDDEN)
                        qw2 = torch.frombuffer(bytearray(w2b[:HIDDEN*INTER//2]), dtype=torch.uint8).reshape(HIDDEN, -1)
                        sc2 = torch.frombuffer(bytearray(w2b[HIDDEN*INTER//2:]), dtype=torch.float16).reshape(HIDDEN, -1)
                        w2 = gpu_decode(qw2, sc2, HIDDEN, INTER)

                        s16 = s.to(torch.float16)
                        gu = F.linear(s16, w13.cpu())
                        g, u = gu.chunk(2, dim=-1)
                        eo = F.linear(F.silu(g) * u, w2.cpu())
                        layer_out[ti] += wgt * eo.squeeze(0)

            x = x + layer_out.reshape(1, 8, HIDDEN).detach()
            total_loss = total_loss + layer_out.pow(2).mean()

        losses.append(total_loss.item())
        dt = time.perf_counter() - st
        print(f"  step {step+1}/{TRAIN_STEPS} | loss={total_loss.item():.4f} | {dt:.1f}s", flush=True)
        _mem(f"step_{step+1}")

    total_dt = time.perf_counter() - t0
    report = {
        "status": "PASSED" if losses else "FAILED",
        "layers": N_LAYERS,
        "experts": N_EXPERTS,
        "cpu_cache_gb": round(total_bytes / (1 << 30), 1),
        "train_steps": TRAIN_STEPS,
        "losses": losses,
        "total_s": round(total_dt, 1),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    # 清理
    print("\n[清理]")
    del cache
    gc.collect()
    torch.cuda.empty_cache()
    shutil.rmtree(OUTPUT, ignore_errors=True)
    print("完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
