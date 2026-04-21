# Windows运行时回退追踪

> 本文记录 Windows 无 Triton 环境下，如何观测某条算子路径最终走到了 `precompiled`、`torch/reference` 还是 `triton`。

## 1. 文档定位

这份文档解决的问题不是“某个算子有没有实现”，而是：

- 在真实运行时，当前请求到底走了哪条分支。
- 哪些 `_C` / CUDA 路径已经真正被命中。
- 哪些地方仍然落回共享 torch/reference。

## 2. 目标

运行时回退追踪主要服务三类判断：

- **主链判断**：35B / 122B 在 Windows 上启动时，真实命中的到底是不是我们预期的共享路径。
- **收口判断**：某次 C++ / CUDA 替换之后，是否真的进入 `_C`，而不是还停留在 Python fallback。
- **文档判断**：后续专项文档只记录稳定结论，避免反复翻原始日志。

## 3. 机制总览

```mermaid
flowchart TD
    A[运行时入口] --> B[共享 Python selector / guard]
    B --> C{分支选择}
    C -->|triton| D[Triton kernel]
    C -->|precompiled| E[_C / CUDA / ATen]
    C -->|torch/reference| F[Torch / Reference 路径]
    D --> G[record(tag, branch)]
    E --> G
    F --> G
    G --> H[runtime_fallback_trace 计数器]
    H --> I[退出时汇总 summary]
```

## 4. 关键代码入口

### 4.1 追踪器实现

- `cfie/utils/runtime_fallback_trace.py`
  - `is_enabled()`：读取 `VLLM_TRACE_RUNTIME_FALLBACKS`
  - `record(tag, branch)`：记录一次命中
  - `snapshot()`：导出当前快照
  - `reset_for_test()`：测试时清空状态

### 4.2 环境变量入口

- `cfie/envs.py`
  - 注册 `VLLM_TRACE_RUNTIME_FALLBACKS`
  - 默认关闭，只有在显式开启时才记录运行时分支

## 5. 当前已接入的代表性标签

| 标签 | 代码位置 | 分支口径 | 用途 |
| --- | --- | --- | --- |
| `sample.topk_topp` | `cfie/v1/sample/ops/topk_topp_sampler.py` | `triton / precompiled / pytorch` | 观测 top-k / top-p 采样路径 |
| `sample.rejection.greedy` | `cfie/v1/sample/rejection_sampler.py` | `precompiled / torch` | 观测 greedy rejection |
| `sample.rejection.random` | `cfie/v1/sample/rejection_sampler.py` | `precompiled / torch` | 观测 random rejection |
| `sample.rejection.expand_batch` | `cfie/v1/sample/rejection_sampler.py` | `precompiled / torch` | 观测 batch 扩展 |
| `sample.rejection.recovered_tokens` | `cfie/v1/sample/rejection_sampler.py` | `precompiled / torch` | 观测 recovered token 路径 |
| `fla.fused_sigmoid_gating` | `cfie/model_executor/layers/fla/ops/fused_sigmoid_gating.py` | `precompiled / reference` | 观测 FLA gating |
| `attention.common.correct_attn_out` | `cfie/v1/attention/ops/common.py` | `precompiled / torch` | 观测 attention 输出修正 |
| `attention.common.pack_seq` | `cfie/v1/attention/ops/common.py` | `precompiled / torch` | 观测 pack 路径 |
| `attention.common.unpack_seq` | `cfie/v1/attention/ops/common.py` | `precompiled / torch` | 观测 unpack 路径 |
| `attention.reshape_cache_flash` | `cfie/v1/attention/ops/triton_reshape_and_cache_flash.py` | `precompiled_block / precompiled_head_major / triton_required` | 观测 KV cache reshape |

## 6. 使用方式

### 6.1 开启追踪

```powershell
$env:VLLM_TRACE_RUNTIME_FALLBACKS = '1'
```

### 6.2 使用建议

- 优先在 `35B` 或 `122B` 的真实 Windows 启动中打开。
- 与 focused pytest、runtime compat 测试配合使用，能更快判断 selector 是否真正生效。
- 只把汇总结论写入文档，不再把整段原始日志堆到 `docs/` 顶层。

## 7. 阶段观察

按已有阶段结论，可以稳定保留以下观察：

- `35B` traced run 已能看到 `sample.topk_topp` 命中 `precompiled`。
- `35B + MTP` traced run 中，`fla.fused_sigmoid_gating`、`rejection sampler` 与 `topk_topp` 已出现 `precompiled` 命中记录。
- `122B` traced run 已能用于判断主链是否仍依赖 CPU static mirror，而不是意外退回 NVMe 或纯 Python 热链。

这意味着，`runtime tracing` 已经能承担“验证替换是否真正进入运行时”的职责。

## 8. 当前限制

这套机制当前仍有两个边界：

- 它回答的是“走了哪条分支”，不是“性能一定最好”。
- 它适合辅助定位 `_C` 是否命中，但不能替代正式 benchmark 和端到端性能分析。

## 9. 相关文档

- `./03_Windows算子替换总计划.md`
- `./04_Windows算子替换执行台账.md`
- `./07_FusedRecurrent共享路径进展.md`
- `./08_MoE批量矩阵乘预编译进展.md`