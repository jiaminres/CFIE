# Windows算子替换总计划

> 本文是推理主线一在 Windows 无 Triton 环境下的长期计划文档。它只保留稳定结论、模块划分和验收原则；逐轮增量结果统一沉淀到 `04_Windows算子替换执行台账.md`。

## 1. 文档定位

这份总计划回答三个问题：

- Windows 无 Triton 时，我们到底替换什么。
- 哪些地方必须与 Linux 共享同一条 Python 主链。
- 每一轮替换完成后，应该怎样验证才算真正接入推理链。

## 2. 总目标

Windows 算子替换工作的目标不是“为 Windows 另开一条高层业务通道”，而是：

- 保持 Linux / Windows 尽量共用同一层 Python 调度入口。
- 把原本强依赖 Triton 的关键路径，收口为 `_C` / CUDA / ATen 预编译实现，或共享 torch/reference 回退实现。
- 让 `35B` 与 `122B` 在 Windows 上都能沿同一条主链启动与运行。
- 把 MoE 主链继续稳定在 `tiered cache + expert unload/on-demand reload` 口径下，而不是引入额外高层分叉。

## 3. 基础约束

当前总计划默认遵守以下约束：

- **共享优先**：能与 Linux 共用 Python 主链，就不新增 Windows 专属高层分支。
- **热链优先**：优先处理真正影响 `35B/122B` 推理可用性和速度的热链模块。
- **回退清晰**：每个入口都要明确 `precompiled`、`torch/reference` 或 `triton` 的选择关系。
- **MoE 原生优先**：Qwen3.5 MoE 的默认显存压力卸载方式仍是 `tiered cache + expert unload/on-demand reload`。
- **日志收束**：原始 smoke / pytest / trace 日志不再直接堆在 `docs/` 顶层，文档里只保留结论与关键观察。

## 4. 82 项历史盘点口径

此前 Windows 适配阶段以 **82 个核心模块/文件** 作为历史盘点口径。为了方便后续长期维护，当前不再把 82 项逐条会话化记录在一个文件里，而是按模块族维护：

| 模块族 | 代表范围 | 主要目标 |
| --- | --- | --- |
| MoE 热链 | `fused_moe.py`、`fused_batched_moe.py`、专家上载链 | 保住 Qwen3.5 MoE 主链与动态专家加载 |
| Sampling / Spec Decode | `topk_topp`、`rejection_sampler`、`EAGLE` 相关 | 保住采样、草稿模型与 speculative decode |
| Attention / KV | `common.py`、`triton_reshape_and_cache_flash.py`、attention backend | 保住 block/slot/KV 相关热链 |
| FLA | `fused_sigmoid_gating.py`、`fused_recurrent.py` 等 | 收口无 Triton 时的前向执行路径 |
| Mamba / SSD | `mamba_ssm.py`、`ssd_*`、`causal_conv1d.py` | 收口线性注意力与状态传递路径 |
| Quant / Model Helper | `awq_triton.py`、`triton_scaled_mm.py`、`fp8_utils.py` 等 | 收口量化 helper 与模型依赖点 |
| Worker / Input Batch | `input_batch.py`、worker utils | 保住 worker 侧输入准备与执行 glue code |
| Runtime Tracing / Smoke | `runtime_fallback_trace.py`、35B/122B traced run | 证明替换后的路径确实进入运行时 |

## 5. 已写入长期文档的关键闭环

当前已经单独形成长期文档沉淀的关键闭环包括：

- `05_8路BaseSlot从CPU到GPU批量上载架构图.md`
  - 说明 runtime-ready CPU mirror 到 GPU resident slot 的批量上载链。
- `06_Windows运行时回退追踪.md`
  - 说明 `precompiled / torch/reference / triton` 的运行时命中记录机制。
- `07_FusedRecurrent共享路径进展.md`
  - 说明 FLA `fused_recurrent.py` 的共享路径收口。
- `08_MoE批量矩阵乘预编译进展.md`
  - 说明 `moe_batched_mm_precompiled` 的 `_C` 闭环。

## 6. 统一验收方式

后续每一批替换都按同一套口径验收：

1. **单点验证**：先跑对应 focused pytest，确认 selector、fallback 与结果一致性。
2. **运行时验证**：再跑 `tests/unit/test_windows_runtime_compat.py` 这类整包兼容用例，确认共享路径没有被破坏。
3. **主链冒烟**：至少做一次 `35B` 主链冒烟；必要时补 `122B` traced run 或专项启动验证。
4. **文档沉淀**：只把稳定结论写回长期文档，不把原始日志堆回 `docs/` 顶层。

## 7. 当前工作重点

文档整理后的下一阶段重点不是继续堆会话记录，而是做三件事：

- 继续用统一 selector 收口剩余热链模块。
- 用 `runtime fallback tracing` 追踪哪些路径已经真正进入 `_C` / CUDA。
- 把 `35B` 与 `122B` 的 Windows 主链验证继续维持在同一套文档口径下。

## 8. 相关文档

- `./04_Windows算子替换执行台账.md`
- `./05_8路BaseSlot从CPU到GPU批量上载架构图.md`
- `./06_Windows运行时回退追踪.md`
- `./07_FusedRecurrent共享路径进展.md`
- `./08_MoE批量矩阵乘预编译进展.md`