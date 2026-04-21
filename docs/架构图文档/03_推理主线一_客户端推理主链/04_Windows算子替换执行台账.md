# Windows算子替换执行台账

> 本文只记录已经形成稳定结论的阶段性事项。零散测试日志、命令输出和会话交接不再直接堆进 `docs/`。

## 1. 文档定位

如果 `03_Windows算子替换总计划.md` 讲的是“要做什么”，这份台账记录的就是“已经明确做成了什么、验证到什么程度”。

## 2. 阶段台账

| 日期 | 主题 | 覆盖范围 | 关键结果 | 验证结论 | 当前状态 |
| --- | --- | --- | --- | --- | --- |
| 2026-04-12 ~ 2026-04-14 | 运行时回退追踪引导层 | `cfie/utils/runtime_fallback_trace.py`、`cfie/envs.py`、采样与 attention 关键入口 | 建立 `precompiled / torch/reference / triton` 运行时命中记录机制 | 可用于 35B / 122B traced run 观察真实命中分支 | 已沉淀 |
| 2026-04-13 | 8 路 BaseSlot 批量上载 | weight offload、`_custom_ops`、`windows_triton_compat_ops.cpp` | CPU static mirror -> GPU resident slot 收口为批量上载语义 | 架构链路已明确，作为 122B 大权重模型的基础能力保留 | 已沉淀 |
| 2026-04-14 | Fused Recurrent 共享路径 | `cfie/model_executor/layers/fla/ops/fused_recurrent.py` | 无 Triton 时优先尝试共享 `_C`，否则回退共享 torch/reference | focused pytest、runtime compat 与 35B smoke 均已有阶段结论 | 已沉淀 |
| 2026-04-14 | MoE 批量矩阵乘预编译 | `fused_batched_moe.py`、`_custom_ops.py`、`torch_bindings.cpp`、`windows_triton_compat_ops.cpp` | `moe_batched_mm_precompiled` 形成真实 `_C` 闭环 | 直接 CUDA 调用、focused pytest、runtime compat 与 35B 冒烟均已有阶段结论 | 已沉淀 |
| 2026-04-14 | 35B / 122B traced run | Windows 单卡 `35B`、`35B+MTP`、`122B` traced run | 可以用统一 tracing 口径观察真实运行时分支 | 已能辅助判断哪些路径真正进入 `precompiled` | 持续更新 |
| 2026-04-14 | 文档体系重整 | `docs/` 导航、路线文档、架构图文档、历史归档 | 清除了顶层无用交接文档与原始日志，统一改为长期文档结构 | 当前 `docs/` 目录已更适合后续 AI / 人工继续维护 | 已完成 |

## 3. 当前如何使用这份台账

- 想看长期目标：先看 `03_Windows算子替换总计划.md`。
- 想查具体闭环：直接跳到对应专项文档。
- 想核对某一轮零散日志：优先从 Git 历史、代码变更和专项文档结论回溯，不再依赖顶层原始日志文件。

## 4. 当前结论

到当前整理完成为止，可以稳定保留的结论是：

- Windows 适配的长期重点，已经从“散乱会话记录”转成“共享主链 + 关键闭环文档”。
- `runtime tracing`、`8 路 BaseSlot 批量上载`、`fused_recurrent`、`moe_batched_mm_precompiled` 这些关键点都已经有独立文档可追溯。
- 后续继续推进新算子时，应复用这套台账方式，而不是重新堆一批无结构的历史记录。

## 5. 相关文档

- `./03_Windows算子替换总计划.md`
- `./05_8路BaseSlot从CPU到GPU批量上载架构图.md`
- `./06_Windows运行时回退追踪.md`
- `./07_FusedRecurrent共享路径进展.md`
- `./08_MoE批量矩阵乘预编译进展.md`