# CFIE

Capacity-First Inference Engine (CFIE).

This folder contains the standalone CFIE prototype implementation.

## Built-in vLLM Source

CFIE now carries a local `vllm/` source snapshot so it can run without
installing an external `vllm` package.

## Training Subproject

Training code now lives in a separate top-level Python package:

- `cfie/`: inference runtime and serving code
- `cfie_training/`: training-only scaffold for the client-side resource-first
  training engine

Bootstrap the training subproject with:

```bash
PYTHONPATH=CFIE .venv/bin/python -m cfie_training.cli.main plan --json
```

Dedicated Qwen3.5-35B-A3B training blueprint:

```bash
PYTHONPATH=CFIE .venv/bin/python -m cfie_training.cli.main qwen35-plan --json
```

First-version Qwen3.5 training runtime simulation:

```bash
PYTHONPATH=CFIE .venv/bin/python -m cfie_training.cli.main simulate --steps 2 --json
```

The simulation output now includes a memory-first residency plan covering:

- GPU hot state
- CPU hot state
- NVMe cold state
- bucket/expert parameter residency transitions
- shard-to-checkpoint transport planning backed by the local safetensors manifest
- stateful staged-file transport execution with bounded host-side file reuse across steps
- bounded parameter store summaries for staged/offloaded representative shards
- parameter source provenance showing which representative shards came from real local weights
- transport-backed source coverage showing how many representative shard loads were satisfied by the currently staged checkpoint files
- representative-load summaries showing how many shard loads hit the staged transport cache, fell back to direct manifest access, or reused buffered values
- host-prefetch summaries showing how many representative shard buffers were prepared before compute, separate from compute-time buffer reuse
- compute-time load summaries now make it explicit when execution is running purely on `cpu_hot` reused buffers after host prefetch
- the representative runtime now streams bucket-by-bucket: prefetch, execute, CPU update, and offload happen per bucket instead of only at whole-step granularity
- bucket lookahead prefetch now stages the next bucket window ahead of compute so later buckets can begin already hot on CPU
- microbatch planning that splits oversized batches into bounded sample-parallel waves
- dual-stream overlap summaries that estimate compute/update makespan and CPU-update lag for each step
- CPU optimizer update summaries
- bounded representative CPU tensor updates with real AdamW math
- bounded representative bucket forward/backward execution with real autograd gradients
- runtime snapshot save/resume for step-level continuation

Resume a simulated training run from a saved snapshot:

```bash
PYTHONPATH=CFIE .venv/bin/python -m cfie_training.cli.main simulate \
  --steps 1 \
  --save-snapshot /tmp/cfie_training_state.json

PYTHONPATH=CFIE .venv/bin/python -m cfie_training.cli.main simulate \
  --steps 1 \
  --resume-from /tmp/cfie_training_state.json
```

Run a synthetic training session with periodic checkpoints:

```bash
PYTHONPATH=CFIE .venv/bin/python -m cfie_training.cli.main train \
  --steps 3 \
  --checkpoint-dir /tmp/cfie_training_ckpts \
  --checkpoint-interval 2
```

Refresh snapshot from upstream tree:

```bash
./scripts/sync_vllm_snapshot.sh
```

## Dependency Audit

```bash
python3 scripts/vllm_dependency_audit.py --root cfie --top 30
```
