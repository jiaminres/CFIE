# 推理主线二：Predictor 隔离接入

> 最后更新：2026-04-17  
> 当前默认训练-推理对接口径：真实前向 `hidden_state`

## 1. 文档定位

本文件描述 Predictor 在 CFIE 推理侧的最新工程接入方式。

这里的重点不是“还能怎么设计”，而是“当前已经怎样接入、还缺哪几块、回退边界是什么”。

## 2. 当前主目标

推理主线二的目标是：

- 加载训练侧导出的 predictor bundle
- 在推理 forward 中按层发射 predictor window plan
- 把 future candidate experts 绑定到对应 MoE 层
- 在不破坏默认主链的前提下，按候选池执行 masked candidate routing

## 3. 当前真实接入闭环

当前已经落地的接入闭环如下：

```mermaid
flowchart LR
    A["predictor bundle"] --> B["load_predictor_bundle / load_predictor_model"]
    B --> C["PredictorRuntimeState"]
    C --> D["PredictorCandidatePlanner"]
    D --> E["Qwen3_5PredictorModel._maybe_emit_predictor_window_plan"]
    E --> F["pending layer plans"]
    F --> G["Qwen3_5PredictorSparseMoeBlock"]
    G --> H["masked candidate routing"]
    H --> I["observed online expert state commit"]
    I --> J["next step fallback / hot-state reuse"]
```

## 4. 当前关键代码入口

### 4.1 Bundle 与 planner

- `CFIE/cfie/predictor/bundle.py`
- `CFIE/cfie/predictor/planner.py`

### 4.2 推理模型隔离接入

- `CFIE/cfie/model_executor/models/qwen3_5_predictor.py`

当前真正把 Predictor 接回执行链的核心类包括：

- `PredictorRuntimeState`
- `PredictorCandidatePlanner`
- `Qwen3_5PredictorModel`
- `Qwen3_5PredictorSparseMoeBlock`

## 5. 当前已经真正落地的能力

### 5.1 Bundle 读取与校验

已经落地：

- runtime schema 校验
- metrics / manifest / weights 一致性校验
- runtime model 重建

### 5.2 真实 hidden state 驱动 planner

已经落地：

- planner 输入支持真实 `hidden_state`
- 支持 rank-1 与 rank-2+ 输入
- token 维 logits 先聚合，再输出未来层 candidate plan

### 5.3 推理层内发射 future plan

已经落地：

- 在 `Qwen3_5PredictorModel.forward()` 中按 stride 发射 predictor window plan
- 生成 `pending_layer_plans`
- 支持跨 pipeline stage 传递 pending layer plan

### 5.4 MoE 层消费 candidate plan

已经落地：

- `Qwen3_5PredictorSparseMoeBlock` 已接管 custom routing
- 对 candidate pool 之外的 experts 做 `-inf` mask
- 在 candidate pool 不足时，按 `allow_candidate_mismatch` 选择回退或报错

这意味着：**masked candidate routing 已经进入实际执行链，而不只是文档预留位。**

### 5.5 在线 expert 状态跟踪

已经落地：

- 观测 routed experts
- 汇总 observed online expert states
- 下一轮 forward 可复用在线 expert 热状态

## 6. 当前仍未闭环的事项

以下事项仍未完全落地：

### 6.1 shared GPU candidate slots

目前状态：

- `PredictorCandidatePlan.shared_gpu_candidate_slots` 只是规划结果中的统计量
- 尚未看到统一的 GPU candidate slot 资源管理器真正接管显存槽位

因此这部分仍属于 **未闭环**。

### 6.2 更完整的调度/缓存联动

当前尚未完全落地：

- predictor 候选池与 tiered cache 资源调度的统一闭环
- predictor 结果与更广义 prefetch/hotness 语义的统一调度

### 6.3 大模型验收

当前状态：

- 35B 真实 smoke 测试已通过
- 122B 已完成进一步打通：
  - `native-generate` / `chat` 已补齐 `skip_mm_profiling`、`language_model_only`、`enable_prefix_caching`、`mamba_cache_mode`
  - 文本型 predictor 已补齐 planner `model_type` 识别、Predictor wrapper 下的 expert layer alias 映射
  - 大体积 CPU static mirror 已改为默认不走 pinned host memory，避免 GPTQ static preprocessing 再次卡在 host-pinned / CUDA OOM
  - 启动期已允许在 `cpu_offload_gb > 0` 时按真实 KV layout 重建 `InputBatch`；文本 Qwen3.5 基类也已补齐 text-only `M-RoPE` positions
  - 用户态 `native-generate` 已越过 KV 初始化与首请求执行阶段，并在真实 122B predictor text-only 临时目录上产出首 token

## 7. 当前回退边界

当前回退原则很清楚：

- predictor runtime 没有加载：走默认路由
- candidate pool 不足且允许 mismatch：回退默认 top-k
- predictor 相关状态缺失：不拖垮主链

所以当前主线二仍然符合“隔离接入、失败可回退”的设计目标。

## 8. 当前已过时口径

以下说法已经不准确：

- “推理侧还只有 predictor-aware helper，没有真正进入执行链”
- “masked candidate routing 还只是预留位”

更准确的说法是：

- helper、planner、pending plan 传递、MoE 路由消费已经接上了
- 真正还没闭环的是 shared GPU candidate slots 与更大范围的资源联动

## 9. 当前推荐阅读顺序

1. `CFIE/cfie/predictor/bundle.py`
2. `CFIE/cfie/predictor/planner.py`
3. `CFIE/cfie/model_executor/models/qwen3_5_predictor.py`
4. `./05_Predictor工程化落地追踪.md`

## 10. 相关文档

- `./02_训练主线二_Predictor路由MoE.md`
- `./05_Predictor工程化落地追踪.md`
- `../架构图文档/04_推理主线二_Predictor隔离接入/00_目录导航.md`

## 11. 最新验收结论

截至 2026-04-17，推理主线二的最新验收结论如下：

- 35B predictor 临时模型目录已经通过真实 `native-generate` smoke。
- 35B 当前稳定验收口径是 `spec-method none`；`mtp` 对 predictor 临时架构仍未打通。
- 122B 的 CLI 路径现在已经可直接带上 `skip_mm_profiling` 等参数，不再需要额外 Python 诊断脚本。
- 122B 文本 predictor 本轮继续清除了多处真实工程缺口：
  - predictor `model_type` 未被 tiered cache planner 识别。
  - Predictor wrapper 下 `model.language_model.layers.*` 与 runtime `model.layers.*` 的 expert 名称未对齐。
  - 大体积 CPU static mirror 长期 pin memory，导致 GPTQ static preprocessing 路径再次暴露 host-pinned / CUDA OOM。
  - `cpu_offload_gb > 0` 与 `InputBatch` 启动期重建之间存在硬断言，阻塞 hybrid KV layout 落地。
  - 文本 Qwen3.5 基类缺少 text-only `M-RoPE` 输入位置接口，导致首请求执行前失败。
- 修复后，122B 已不再命中 `No available memory for the cache blocks`，并已在真实 `native-generate` 验收中输出首 token（当前日志末尾为 `.`）。
- 需要注意：通过 PowerShell 的 `2>&1 | Tee-Object` 包装 `cfie.exe` 时，日志 stderr 仍可能表现为 `NativeCommandError`；验收应以“无 Python traceback + 日志末尾出现实际输出 token”为准。
- 因此，当前 122B predictor 文本主线应归类为 **已完成单 token smoke 验收**；剩余问题主要是 `mtp` 未打通与命令行包装层的退出码噪声。
