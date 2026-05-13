"""OpenAI 协议 + Predictor 集成测试。

验证在开启 predictor 的前提下，OpenAI 兼容 API 的 chat completion 管线
是否能正常工作（chat template 渲染 → engine → predictor 路由 → 流式输出）。

用法:
    python scripts/test_predictor_openai.py
"""
from __future__ import annotations

import json
import multiprocessing
import sys
import time
from pathlib import Path

MODEL = (
    "C:/Users/13642/.cache/huggingface/hub/"
    "models--Qwen--Qwen3.5-122B-A10B-GPTQ-Int4/snapshots/"
    "5b9f0050d3ec98b0c81a7716776533c5eacebb64"
)
PREDICTOR = (
    "C:/Users/13642/PycharmProjects/vllm/CFIE/.tmp/predictor_122b_books/"
    "mask_layer0_frozen_router_delta_384_purekl_e200/"
    "predictor_runtime_epoch_200.ckpt"
)
RESULT_FILE = Path(
    "C:/Users/13642/PycharmProjects/vllm/CFIE/.tmp/predictor_openai_result.json"
)


def main() -> int:
    from cfie.engine.arg_utils import EngineArgs
    from cfie.renderers.params import ChatParams
    from cfie.sampling_params import RequestOutputKind, SamplingParams
    from cfie.usage.usage_lib import UsageContext
    from cfie.v1.engine.llm_engine import LLMEngine
    from transformers import AutoTokenizer

    print("=" * 60)
    print("[test] predictor + OpenAI chat protocol integration")
    print("=" * 60)

    t0 = time.perf_counter()

    # 步骤 1: 构建引擎参数（与已验证的 smoke test 一致 + predictor）
    print("[1] Building engine args (predictor enabled)...", flush=True)
    engine_args = EngineArgs(
        model=MODEL, tokenizer=MODEL, trust_remote_code=False,
        dtype='float16', quantization='gptq_marlin',
        max_model_len=128, max_num_seqs=1, max_num_batched_tokens=128,
        gpu_memory_utilization=0.60, enforce_eager=True,
        enable_prefix_caching=False, enable_chunked_prefill=True,
        moe_cpu_budget_gb=0.0, moe_cpu_min_free_gb=0.0,
        cpu_offload_gb=0.0, offload_backend='auto',
        hf_overrides={'predictor_checkpoint_path': PREDICTOR},
    )

    # 步骤 2: 创建引擎（predictor bundle 将在此步骤加载）
    print("[2] Starting engine (loading predictor)...", flush=True)
    engine = LLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.ENGINE_CONTEXT)
    print(f"    ready ({time.perf_counter() - t0:.1f}s)", flush=True)

    # 步骤 3: 模拟 OpenAI /v1/chat/completions 的 chat template 渲染
    print("[3] Rendering chat template (simulating OpenAI chat completion)...", flush=True)
    messages = [{"role": "user", "content": "请用一句话介绍你自己"}]

    chat_params = ChatParams(
        chat_template_kwargs={
            "add_generation_prompt": True,
            "tokenize": False,
        },
    )
    _, engine_prompts = engine.renderer.render_chat(
        [messages], chat_params)
    prompt = engine_prompts[0]

    # 步骤 4: 提交请求并获取结果
    print("[4] Submitting request + running...", flush=True)
    sampling_params = SamplingParams(
        max_tokens=32, temperature=0.0, top_p=1.0, top_k=0,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )
    engine.add_request("openai-predictor-1", prompt, sampling_params)

    final = None
    for step in range(64):
        outputs = engine.step()
        if outputs:
            final = outputs[-1]
            if getattr(final, 'finished', False):
                break

    total_time = time.perf_counter() - t0

    if final and getattr(final, 'finished', False):
        text = final.outputs[0].text
        print(f"\n    output: {text}")
        print(f"    time: {total_time:.1f}s")

        result = {
            "ok": bool(text),
            "mode": "chat_template",
            "predictor_enabled": True,
            "predictor_path": PREDICTOR,
            "model": MODEL,
            "messages": [{"role": "user", "content": "请用一句话介绍你自己"}],
            "response": text,
            "total_seconds": round(total_time, 1),
        }
        RESULT_FILE.write_text(
            json.dumps(result, ensure_ascii=False, indent=2))
        print(f"\n    result saved: {RESULT_FILE}")

        print("\n[PASS] predictor + OpenAI chat protocol inference OK")
        try:
            engine.engine_core.shutdown()
        except Exception:
            pass
        return 0
    else:
        print("\n[FAIL] inference did not complete")
        try:
            engine.engine_core.shutdown()
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
