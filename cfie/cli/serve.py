"""Serve command entrypoint for CFIE."""

from __future__ import annotations

from argparse import Namespace

from cfie.config.schema import EngineConfig
from cfie.runtime.engine import Engine


def config_from_namespace(args: Namespace) -> EngineConfig:
    # 显式映射 CLI 参数到配置对象，便于测试与定位问题。
    return EngineConfig.from_flat_kwargs(
        model=args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        quantization=args.quantization,
        kv_cache_dtype=args.kv_cache_dtype,
        weight_offload_backend=args.weight_offload_backend,
        kv_offload_backend=args.kv_offload_backend,
        cpu_offload_gb=args.cpu_offload_gb,
        nvme_offload_path=args.nvme_offload_path,
        offload_prefetch_window=args.offload_prefetch_window,
    )


def run_serve(args: Namespace) -> int:
    # Phase 0 目标：先打通启动生命周期与参数校验链路。
    config = config_from_namespace(args)
    engine = Engine(config)
    engine.start()
    print("CFIE engine started")
    for _ in range(args.steps):
        engine.step()
    engine.stop()
    print("CFIE engine stopped")
    return 0
