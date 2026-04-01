"""Serve command entrypoint for CFIE."""

from __future__ import annotations

from argparse import Namespace

from cfie.config.schema import EngineConfig
from cfie.request.request import InferenceRequest
from cfie.runtime.engine import Engine
from cfie.utils.logging import get_logger

logger = get_logger(__name__)


def config_from_namespace(args: Namespace) -> EngineConfig:
    # 显式映射 CLI 参数到配置对象，便于测试与定位问题。
    return EngineConfig.from_flat_kwargs(
        model=args.model,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        load_format=args.load_format,
        download_dir=args.download_dir,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        quantization=args.quantization,
        kv_cache_dtype=args.kv_cache_dtype,
        weight_offload_backend=args.weight_offload_backend,
        kv_offload_backend=args.kv_offload_backend,
        cpu_offload_gb=args.cpu_offload_gb,
        moe_cpu_budget_gb=args.moe_cpu_budget_gb,
        moe_cpu_min_free_gb=args.moe_cpu_min_free_gb,
        nvme_offload_path=args.nvme_offload_path,
        offload_prefetch_window=args.offload_prefetch_window,
    )


def _run_prompt(engine: Engine, args: Namespace) -> int:
    request = InferenceRequest(prompt=args.prompt,
                               max_new_tokens=args.max_new_tokens,
                               session_id=args.session_id)
    engine.add_request(request)

    try:
        # 逐 step 拉取增量输出并立即打印，实现最小流式体验。
        while not request.is_terminal:
            results = engine.step()
            for result in results:
                if result.request_id != request.request_id:
                    continue
                if result.token_text:
                    print(result.token_text, end="", flush=True)
    except KeyboardInterrupt:
        # 允许用户通过 Ctrl+C 中断当前请求，而不是直接杀进程。
        engine.abort(request.request_id)
        logger.warning("request aborted by user: request_id=%s", request.request_id)
        print()

    print()
    return 0


def run_serve(args: Namespace) -> int:
    # Phase 1: 启动后若给定 prompt，则执行最小 prefill+decode+stream 链路。
    config = config_from_namespace(args)
    engine = Engine(config)
    engine.start()
    logger.info("CFIE engine started")

    try:
        if args.prompt:
            return _run_prompt(engine, args)

        # 未传 prompt 时保持 Phase 0 的空步进行为，便于快速启动测试。
        for _ in range(args.steps):
            engine.step()
        return 0
    finally:
        engine.stop()
        logger.info("CFIE engine stopped")
