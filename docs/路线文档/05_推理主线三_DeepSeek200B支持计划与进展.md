# 05_推理主线三_DeepSeek200B支持计划与进展

> 更新时间：2026-04-16  
> 负责人：CFIE 推理链路（DeepSeek V2 / V2.5）
> 口径说明：本文件属于并行专题计划，不纳入 `05_Predictor工程化落地追踪.md` 的 Predictor 验收总账。

## 1. 目标与范围

- 目标：在 Windows + CFIE 上支持 DeepSeek V2 / V2.5（200B 级 MoE）稳定推理。
- 优先策略：先落地 4bit（优先 GPTQ-Marlin，可选 AWQ），再做高性能收敛。
- 复用原则：优先复用 Qwen3.5 已验证的高性能算子与 MoE 卸载策略，避免重复造轮子。

## 2. 分阶段工作计划

### Phase 0：基线冻结（约 1~2 天）

- 冻结目标模型清单（V2、V2.5 各 1 个主模型 + 1 个备用量化版本）。
- 冻结硬件与运行预算（GPU 显存、CPU 内存、磁盘、并发目标、max_model_len）。
- 输出《模型-硬件-指标矩阵》作为后续统一验收标准。

### Phase 1：模型资产准备（约 2~4 天）

- 下载并校验 4bit 模型资产（config/tokenizer/safetensors index）。
- 若官方不可用，补齐离线量化转换链路（FP16/BF16 -> GPTQ/AWQ）。
- 固化目录与版本追踪（revision、哈希、来源）。

### Phase 2：最小可运行 MVP（约 3~5 天）

- 跑通 `native-generate` 的单请求短输出链路。
- 先保正确性（eager），再评估 graph/capture 的可用性。
- 固化最小可运行命令模板和失败回退参数。

### Phase 3：高性能复用与收敛（约 5~8 天）

- 复用 Qwen3.5 相关 GPTQ-Marlin 重排、MoE batch load、batched mm 路径。
- 对齐 DeepSeek 的 MoE 路由与共享专家策略，减少无效 fallback。
- 对比 MVP 输出吞吐、首 token 延迟、显存峰值。

### Phase 4：Windows Triton 缺口补齐（约 5~10 天）

- 构建 Triton 缺失算子清单并按阻塞级别排序。
- 优先补 `CUDA/C++ + torch.ops._C` 路径，保留 torch fallback 兜底。
- 每个新算子配套正确性测试与性能回归。

### Phase 5：端到端验收（约 2~4 天）

- 维度：模型（V2/V2.5）× 场景（短/长上下文）× 并发（低/中）× util 档位。
- 指标：启动成功率、OOM 率、token/s、P95 延迟、峰值显存。
- 输出：上线建议参数与降级方案。

## 3. 当前工作进展

### 已完成

- 已确认仓库内存在 DeepSeek 主实现与注册入口：
  - `cfie/model_executor/models/deepseek_v2.py`
  - `cfie/model_executor/models/registry.py`
- 已确认配置层对 DeepSeek V2.5（trust_remote_code 场景）存在兼容路径线索：
  - `cfie/transformers_utils/config.py`
- 已形成首版总体推进方案（模型资产、MVP、高性能复用、Windows 算子补齐、验收闭环）。

### 进行中

- 正在细化模型资产清单（优先 GPTQ 4bit 可直接落地版本）。
- 正在准备“可复现下载 + 校验 + 启动参数模板”。

### 待开始

- DeepSeek 122B/200B 级实际下载与完整性校验。
- DeepSeek 专项推理冒烟（短 prompt、长上下文、稳定性）。
- Windows 缺失算子的逐项补齐与强约束验证。

## 4. 风险与应对

- 风险：量化格式差异导致加载失败或性能退化。  
  应对：统一优先级 GPTQ-Marlin > AWQ > 保守 fallback，并保留格式探测。
- 风险：Triton-only 路径在 Windows 不可用。  
  应对：落地 `torch.ops._C` 对应 CUDA 实现，建立严格模式阻断隐式 fallback。
- 风险：MoE 卸载与 graph capture 兼容性差。  
  应对：默认 eager 稳定优先，capture 作为可选优化路径。

## 5. 本周里程碑（下一次更新前）

- 完成 DeepSeek V2/V2.5 4bit 候选清单与下载脚本。
- 跑通至少 1 条 DeepSeek 4bit 的 `native-generate` MVP 日志。
- 输出首版 Windows 缺失算子清单（按优先级排序）。
