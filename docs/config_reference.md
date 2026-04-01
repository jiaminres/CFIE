# CFIE Config Reference (Phase 1)

The canonical schema lives in `cfie/config/schema.py`.

## Top-Level

`EngineConfig` composes:

- `ModelConfig`
- `LoadConfig`
- `QuantConfig`
- `CacheConfig`
- `OffloadConfig`
- `SchedulerConfig`
- `RuntimeConfig`

## Key Fields

- `model` (`str`, required)
- `revision` (`str | None`)
- `trust_remote_code` (`bool`)
- `local_files_only` (`bool`)
- `load_format` (`auto|hf|safetensors|pt`)
- `download_dir` (`str | None`)
- `dtype` (`auto|fp16|bf16`)
- `max_model_len` (`int > 0`)
- `max_num_seqs` (`int > 0`, default `1`)
- `gpu_memory_utilization` (`float`, range `(0, 1]`)
- `quantization` (`none|gptq|awq|bnb`)
- `kv_cache_dtype` (`auto|fp16|fp8`)
- `weight_offload_backend` (`cpu|nvme|cpu+nvme`)
- `kv_offload_backend` (`cpu|nvme|cpu+nvme`)
- `cpu_offload_gb` (`float >= 0`)
- `moe_cpu_budget_gb` (`float >= 0`, `0` means auto planner)
- `moe_cpu_min_free_gb` (`float >= 0`, `0` means planner default)
- `nvme_offload_path` (`str`, normalized path)
- `offload_prefetch_window` (`int >= 0`)

## Validation Rules

- Invalid enum values raise `ValueError`.
- Empty model path/name raises `ValueError`.
- `load_format` 必须在支持集合内。
- `weight_offload_backend = nvme` 时，`cpu_offload_gb` 必须为 `0`。
