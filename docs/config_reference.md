# CFIE Config Reference (Phase 0)

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
- `dtype` (`auto|fp16|bf16`)
- `max_model_len` (`int > 0`)
- `max_num_seqs` (`int > 0`, default `1`)
- `gpu_memory_utilization` (`float`, range `(0, 1]`)
- `quantization` (`none|gptq|awq|bnb`)
- `kv_cache_dtype` (`auto|fp16|fp8`)
- `weight_offload_backend` (`cpu|nvme|cpu+nvme`)
- `kv_offload_backend` (`cpu|nvme|cpu+nvme`)
- `cpu_offload_gb` (`float >= 0`)
- `nvme_offload_path` (`str`, normalized path)
- `offload_prefetch_window` (`int >= 0`)

## Validation Rules

- Invalid enum values raise `ValueError`.
- Empty model path/name raises `ValueError`.
- `weight_offload_backend = nvme` 时，`cpu_offload_gb` 必须为 `0`。
