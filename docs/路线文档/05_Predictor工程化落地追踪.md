# Predictor 工程化落地追踪

> 最后更新：2026-04-20

## 1. 完成标准

当前 Predictor 什么时候算“真正完成”，看的是下面四件工程事实：

- 文档口径是否统一到真实主链；
- 训练侧是否只保留真实 dataset-backed + engine-forward-capture 路线；
- 训练实现是否与普通推理严格隔离；
- 35B / 122B 真实训练 smoke 与普通推理回归是否完成。

## 2. 筋斗云追踪

| 编号 | 事项 | 状态 | 说明 |
| --- | --- | --- | --- |
| P0 | 文档统一到真实主链 | 已完成 | 路线文档与架构图文档已按 `EngineRouterTeacherModelBackend + teacher top-k` 主链收口 |
| P1 | teacher 改为复用推理引擎 | 已完成 | `EngineRouterTeacherModelBackend` 已接入 `LLMEngine` |
| P2 | GPTQ teacher 路径禁止回退 base snapshot | 已完成 | `_resolve_model_path()` 直接使用配置路径 |
| P3 | hidden-state capture 与普通推理隔离 | 已完成 | worker/model runner 默认关闭，仅训练显式打开 |
| P4 | trace builder 直接消费 teacher top-k | 已完成 | `CapturedForwardBatch` 已改为 `layer_teacher_topk_ids` |
| P5 | Predictor 相关单测回归 | 已完成 | `test_predictor_bundle.py`、`test_cfie_training_config.py`、`test_cfie_training_profiles.py`、`test_cfie_training_cli.py -k predictor` 已通过 |
| P6 | `test_cfie_training_cli.py` 全量回归 | 进行中 | 全文件在当前环境超时，需继续拆分定位慢用例 |
| P7 | 35B 真实 predictor 训练 smoke | 已完成 | `predictor-train --profile qwen35-35b-a3b` 最小链路已复跑通过 |
| P8 | 122B 真实 predictor 训练 smoke | 已完成 | `predictor-trace` 与 `predictor-train --profile qwen35-122b-a10b` 最小链路已通过 |
| P9 | 普通推理回归 | 已完成 | 已执行普通 `chat` smoke，训练侧 capture 默认关闭，未污染普通推理配置 |

## 3. 本轮代码落地结论

本轮针对 Predictor 训练侧已经完成的主线收口如下：

- teacher 不再直建 HF 模型，而是复用 CFIE 推理引擎；
- teacher 捕获路径改为真实 `hidden_state + routed_experts top-k`；
- GPTQ teacher 不再隐式退回到非量化 base snapshot；
- worker 侧新增 predictor capture RPC，并保持默认关闭；
- 训练单测桩已从“router logits”切到“teacher top-k ids”；
- 122B routed experts capture 已修正为按 attention KV 池真实 slot 地址空间分配共享缓冲，而不再错误使用聚合 token 下界。

## 4. 当前主链事实

### 4.1 Predictor 训练主链

```text
dataset
  -> predictor-trace
  -> EngineRouterTeacherModelBackend
  -> LLMEngine real forward
  -> hidden_state + routed_experts top-k
  -> PredictorTraceDataset
  -> predictor-train
  -> checkpoint / schema
  -> predictor-eval
  -> predictor-export
```

### 4.2 当前隔离事实

```text
普通推理
  -> predictor_capture_enabled = False
  -> 不抓 hidden states

训练 teacher 引擎
  -> collective_rpc(enable_predictor_capture)
  -> 训练完成后 collective_rpc(disable_predictor_capture)
```

## 5. 当前测试结果

### 5.1 已通过

- `tests/unit/test_predictor_bundle.py`
- `tests/unit/test_cfie_training_config.py`
- `tests/unit/test_cfie_training_profiles.py`
- `tests/unit/test_cfie_training_cli.py -k predictor`
- `tests/unit/test_predictor_capture_resolution.py`
- `tests/unit/test_routed_experts_buffer_size.py`
- `tests/unit/test_cfie_training_engine.py -k predictor_teacher_backend_maps_training_offload_policy`
- `predictor-train --profile qwen35-35b-a3b --steps 1 --examples-per-step 1 --samples 1 --tokens-per-sample 16 --epochs 1`
- `predictor-trace --profile qwen35-122b-a10b --steps 1 --examples-per-step 1 --samples 1 --tokens-per-sample 16`
- `predictor-train --profile qwen35-122b-a10b --steps 1 --examples-per-step 1 --samples 1 --tokens-per-sample 16 --epochs 1`
- `cfie chat --model <Qwen3.5-35B-A3B-GPTQ-Int4> --spec-method mtp ...`

### 5.2 未完成

- `tests/unit/test_cfie_training_cli.py` 全量执行在当前环境超时，仍需拆分定位
- predictor 推理侧异步上载链路尚未开始接入；当前仍以训练侧闭环与普通推理隔离为准

## 6. 下一步顺序

建议接下来的执行顺序固定为：

1. 拆分并完成 `test_cfie_training_cli.py` 全量回归；
2. 继续清理训练侧残留的 blueprint / synthetic / mock 非主链内容；
3. 若要接 predictor 推理侧，再新增显式开关并保持与普通推理隔离；
4. 在 predictor 推理侧完成前，维持当前“训练闭环已通、普通推理不受影响”的工程基线。
