# Windows 适配阶段交接汇总

## 1. 文档定位

本文件把此前零散分布在 `docs/` 顶层的阶段性交接文档合并为一份可持续查阅的摘要。

## 2. 这批交接文档主要覆盖的主题

- **推理主链与算子替换**
  - attention、sampling、spec decode、FLA、Mamba、fused recurrent、Windows no-Triton 兼容。
- **MoE 与量化路径**
  - fused MoE、batched MoE、WNA16、GPTQ / AWQ / FP8 / int8、专家装载与共享 dispatch。
- **LoRA 与辅助路径**
  - LoRA、LoRA FP8、kernel utils、shared path 收口。
- **验证与构建**
  - Windows 可编辑安装、重编译、focused pytest、整包 runtime compat、35B / 122B 真机 smoke 与 tracing。

## 3. 合并后的长期有效落点

这些阶段性工作里，真正需要长期维护的信息，已经沉淀到下面几类文档：

- `路线文档/03_推理主线一_客户端推理主链.md`
- `架构图文档/03_推理主线一_客户端推理主链/03_Windows算子替换总计划.md`
- `架构图文档/03_推理主线一_客户端推理主链/04_Windows算子替换执行台账.md`
- `架构图文档/03_推理主线一_客户端推理主链/06_Windows运行时回退追踪.md`
- `历史归档/02_Windows单卡推理验证基线.md`

## 4. 当前阶段结论

- Windows `no-Triton` 主链已经形成统一的 Python 层共享通道，尽量不单独开高层分支。
- 35B / 122B 的客户端推理主链已经建立可复现基线。
- Windows 算子替换工作的长期追踪入口，不再依赖零散 handoff，而是“总计划 + 执行台账 + 运行时回退追踪”三份文档。
- MoE offload、KV 规划、spec decode、sampling、attention、MoE 量化路径等关键主题都已在现有主文档中有稳定落点。

## 5. 历史来源说明

本次归档吸收了此前按日期拆分的会话交接文档，包括：

- `会话交接_*.md`
- `handoff_*.md`
- `session_handoff_*.md`

这些文档原本用于跨账号、跨会话续做任务；在整理后，它们的有效结论已转移到主文档或本归档摘要中，因此不再单独保留。
