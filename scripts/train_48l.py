"""48层全量训练——逐层 NVMe→CPU→GPU量化→CPU 缓存→训练→导出→清理"""
from __future__ import annotations
import gc, json, shutil, sys, time
from pathlib import Path
import torch, torch.nn.functional as F

CKPT = Path("C:/Users/13642/.cache/huggingface/hub/models--Qwen--Qwen3.5-122B-A10B/"
            "snapshots/b000b2eb18a7f4cdf3153c4215842da339e09d99")
OUTPUT = Path("C:/Users/13642/PycharmProjects/vllm/CFIE/.tmp/train_48l")

N_LAYERS, HOT, TOP_K, STEPS, BATCH_L = 48, 4, 8, 2, 4

def _mem(tag=""):
    import psutil
    rss=psutil.Process().memory_info().rss/(1<<30)
    gpu=torch.cuda.memory_allocated(0)/1e9 if torch.cuda.is_available() else 0
    print(f"  [{tag}] RAM={rss:.1f}G GPU={gpu:.1f}G", flush=True)
    if rss>145 or gpu>28: print("FATAL"); sys.exit(1)

def main():
    t0=time.perf_counter()
    if OUTPUT.exists(): shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir()

    # ── 1. NVMe FP32——逐层从 safetensors 流式写入 shard ──
    print("[1/5] NVMe FP32 stores（逐层写入）...", flush=True)
    from cfie_training.training_base.fp32_shard_store import FP32ShardStore, ParamShardRecord
    from safetensors import safe_open
    import struct

    fp32_root=OUTPUT/"fp32"; fp32_root.mkdir()
    records={}; shard_offsets={}
    for lid in range(N_LAYERS):
        sn=f"fp32_{lid//2:04d}.bin"
        for eid in range(256):
            for wname,n in [("w13_weight",2*1024*3072),("w2_weight",3072*1024)]:
                pid=f"layers.{lid}.experts.{eid}.{wname}"
                off=shard_offsets.get(sn,0)
                records[pid]=ParamShardRecord(pid,sn,off,n)
                shard_offsets[sn]=off+n
    fp32_store=FP32ShardStore.create(fp32_root,records,generation=0)
    shards=sorted(str(p) for p in CKPT.glob("model*.safetensors") if p.suffix==".safetensors")
    fp32_written=0
    BATCH=4  # 每次处理 4 层，控制 RAM
    for batch_start in range(0,N_LAYERS,BATCH):
        batch_end=min(batch_start+BATCH,N_LAYERS)
        target=set(range(batch_start,batch_end))
        batch={}
        for sp in shards:
            with safe_open(sp,framework="pt") as f:
                for key in f.keys():
                    if "mlp.experts" not in key: continue
                    pfx="model.language_model.layers."
                    if not key.startswith(pfx): continue
                    lid=int(key[len(pfx):].split(".")[0])
                    if lid not in target: continue
                    is_gate="gate_up_proj" in key
                    t=f.get_tensor(key)
                    for eid in range(t.shape[0]):
                        ed=t[eid].float()
                        if is_gate:
                            g,u=ed[:1024,:],ed[1024:,:]
                            batch[f"layers.{lid}.experts.{eid}.w13_weight"]=torch.cat([g.reshape(-1),u.reshape(-1)]).numpy().tobytes()
                        else:
                            batch[f"layers.{lid}.experts.{eid}.w2_weight"]=ed.reshape(-1).numpy().tobytes()
        fp32_store.flush_touched(batch,generation=0)
        fp32_written+=sum(len(v) for v in batch.values())
        del batch; gc.collect()
        print(f"  layers {batch_start}-{batch_end-1}/{N_LAYERS}",flush=True)
    print(f"  FP32: {fp32_written/(1<<30):.1f}G written",flush=True)
    _mem("nvme")

    # ── 2. CPU GPTQ 缓存——逐层从 NVMe 读 FP32→GPU 量化+repack→CPU ──
    print("[2/5] CPU Int4 缓存（逐层 NVMe→GPU量化+repack→CPU）...", flush=True)
    from cfie_training.training_base.gpu_gptq import GpuGptqQuantizer, GpuGptqConfig
    from cfie.model_executor.layers.quantization.utils.quant_utils import gptq_pack

    quant=GpuGptqQuantizer(GpuGptqConfig(group_size=128))
    cpu_cache={}  # {(lid,eid): (w13_packed_bytes, w13_sc_bytes, w2_packed_bytes, w2_sc_bytes)}
    for lid in range(N_LAYERS):
        # 从 NVMe 读本层全部 256 专家的 FP32 → GPU
        w13_fp32=[]; w2_fp32=[]
        for eid in range(256):
            w13_raw=fp32_store.read_param(f"layers.{lid}.experts.{eid}.w13_weight")
            w2_raw=fp32_store.read_param(f"layers.{lid}.experts.{eid}.w2_weight")
            w13_fp32.append(torch.frombuffer(bytearray(w13_raw),dtype=torch.float32).reshape(2048,3072).cuda())
            w2_fp32.append(torch.frombuffer(bytearray(w2_raw),dtype=torch.float32).reshape(3072,1024).cuda())
        # GPU 量化 + gptq_pack → CPU bytes
        for eid in range(256):
            l13,sc13=quant.quantize_logical(w13_fp32[eid])
            l2,sc2=quant.quantize_logical(w2_fp32[eid])
            # gptq_pack: 逻辑 [out,in] → 转置 [in,out] → pack
            out_f,in_f=l13.shape  # [2048,3072]
            p13=gptq_pack((l13.T.contiguous()+8).to(torch.int32),4,in_f,out_f)
            out_f2,in_f2=l2.shape  # [3072,1024]
            p2=gptq_pack((l2.T.contiguous()+8).to(torch.int32),4,in_f2,out_f2)
            cpu_cache[(lid,eid)]=(p13.cpu().numpy().tobytes(),sc13.T.contiguous().cpu().numpy().tobytes(),
                                   p2.cpu().numpy().tobytes(),sc2.T.contiguous().cpu().numpy().tobytes())
        del w13_fp32,w2_fp32; gc.collect(); torch.cuda.empty_cache()
        if lid%8==0: print(f"  layer {lid}/{N_LAYERS} | {len(cpu_cache)} experts",flush=True)
    cache_gb=sum(len(e[0])+len(e[1])+len(e[2])+len(e[3]) for e in cpu_cache.values())/(1<<30)
    print(f"  CPU cache: {len(cpu_cache)} experts, {cache_gb:.1f}G",flush=True)
    _mem("cache")

    # ── 3. 模型 ──
    print("[3/5] Qwen35ForTraining 48层 + hot专家...", flush=True)
    from cfie_training.training_base.training_model import Qwen35ForTraining
    model=Qwen35ForTraining(num_layers=N_LAYERS,hidden_size=3072,intermediate_size=1024,
                             num_experts=256,top_k=TOP_K,vocab_size=248320,
                             dtype=torch.float16,device="cuda")
    model=model.cuda()
    for layer in model.layers: layer.moe._device=torch.device("cuda")

    # 加载 hot 专家
    loaded=0
    for lid in range(N_LAYERS):
        for sp in shards:
            with safe_open(sp,framework="pt") as f:
                for key in f.keys():
                    if "mlp.experts" not in key: continue
                    pfx="model.language_model.layers."
                    if not key.startswith(pfx): continue
                    if int(key[len(pfx):].split(".")[0])!=lid: continue
                    is_gate="gate_up_proj" in key
                    t=f.get_tensor(key)
                    moe=model.layers[lid].moe
                    for eid in range(min(t.shape[0],HOT)):
                        ed=t[eid].float()
                        if is_gate:
                            inter=ed.shape[-2]//2
                            w13=torch.cat([ed[:inter,:].reshape(-1),ed[inter:,:].reshape(-1)]).reshape(2*inter,ed.shape[-1])
                            moe.set_hot_expert(eid,w13,torch.zeros(ed.shape[-1],inter)) if eid not in moe._hot_w13 else moe._hot_w13[eid].data.copy_(w13)
                        else:
                            moe.set_hot_expert(eid,torch.zeros(2*moe.intermediate_size,moe.hidden_size),
                                               ed.reshape(moe.hidden_size,moe.intermediate_size)) if eid not in moe._hot_w2 else moe._hot_w2[eid].data.copy_(ed.reshape(moe.hidden_size,moe.intermediate_size))
                        loaded+=1
    print(f"  {loaded} hot weights",flush=True)
    _mem("model")

    # 设置 CPU 缓存到模型（通过 _get_cold_expert_cached 会走 cpu_gptq_cache._entries）
    class SimpleCache: pass
    sc=SimpleCache(); sc._entries=cpu_cache
    for layer in model.layers: layer.moe.cpu_gptq_cache=sc
    del cpu_cache; gc.collect()

    # ── 4. 训练 ──
    print(f"[4/5] 训练 {STEPS} 步...", flush=True)
    losses=[]
    for step in range(STEPS):
        torch.cuda.synchronize(); st=time.perf_counter()
        x=torch.randint(0,1000,(1,32),device="cuda")
        labels=torch.randint(0,1000,(1,32),device="cuda")
        logits,_=model(x)
        logits=torch.clamp(logits,-100,100)
        loss=F.cross_entropy(logits[:,:-1,:].reshape(-1,logits.shape[-1]),labels[:,1:].reshape(-1))
        model.zero_grad(); loss.backward(); torch.cuda.synchronize()
        losses.append(loss.item())
        dt=time.perf_counter()-st
        print(f"  step{step+1} loss={loss:.4f} {dt:.1f}s GPU={torch.cuda.max_memory_allocated()/1e9:.1f}G",flush=True)
        _mem(f"s{step+1}")

    # ── 5. 导出（流式，每 4 层一批） + 清理 ──
    print("[5/5] 导出 GPTQ + 清理...", flush=True)
    export_file=OUTPUT/"gptq_export.pt"
    # 分批导出，避免 55 GB dict OOM
    all_entries=list(sc._entries.items())
    torch.save({"layers":N_LAYERS,"experts":256,"total_batches":(len(all_entries)+1023)//1024},export_file/"meta.pt")
    for bi in range(0,len(all_entries),1024):
        batch_data={"entries":{}}
        for (lid,eid),v in all_entries[bi:bi+1024]:
            batch_data["entries"][f"L{lid}_E{eid}"]={"w13_qw":v[0],"w13_sc":v[1],"w2_qw":v[2],"w2_sc":v[3]}
        torch.save(batch_data,export_file/f"batch_{bi//1024:04d}.pt")
        del batch_data; gc.collect()
    total_sz=sum(f.stat().st_size for f in export_file.glob("*.pt"))
    print(f"  exported: {total_sz/(1<<30):.1f}G",flush=True)

    td=time.perf_counter()-t0
    print(f"\n{'='*40}\n  PASSED | {N_LAYERS}L | {len(losses)}steps | {td:.0f}s\n{'='*40}")

    # Cleanup
    del model; gc.collect(); torch.cuda.empty_cache()
    shutil.rmtree(OUTPUT,ignore_errors=True)
    print("Cleaned.",flush=True)
    return 0

if __name__=="__main__": raise SystemExit(main())
