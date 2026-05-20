# GUI Agent 122B 长上下文优化计划

## 目标

面向 GUI agent 场景优化 Qwen3.5-122B-A10B-GPTQ-Int4 的长视频帧输入与文本 decode：

- 支持单请求 256K 上下文能力验证。
- 支持约 5w token 视频帧上下文 prefill，并统计 prefill 耗时。
- 在 decode 阶段保持 tiered MoE cache + pinned/static mirror + 8 路 CPU copy 的加速路径。
- 在 prefill 阶段支持 burst 执行路径，避免 unique experts 超过 resident slots 后退化为逐 expert resident swap。
- 控制 graph capture 显存，避免默认录制大量无用 shape。
- 评估 dense 参数进一步量化与 W4A8/FP8 activation 路径的收益。

## 当前关键判断

1. `prefill-burst-slots=0` 只适合短 prompt decode 基线，不适合生产长 prefill。
2. 生产长 prefill 应按 `--prefill-burst-slots 256` 设计，使单层最多 256 个 unique experts 可以进入 burst 执行。
3. 设置 `prefill-burst-slots=256` 后，GPU 上至少存在：
   - resident expert cache：`gpu_slots_per_layer * num_layers`
   - 共享 burst pool：一层 `256` expert slots
   - runtime H2D GPU stage：用于 CPU pinned stage 到 GPU 后 scatter
4. 如果复用 decode 的 batch H2D 设计，并让 `cpu stage`、`gpu stage`、`burst pool` 都支持 256 slots，则 burst 相关额外显存可能接近两份 256-slot expert 参数，加上 graph pool 显存，显存压力会很高。
5. 当前 graph capture 录制 shape 过多，单请求 GUI 场景不应该默认捕获大量中间 shape。

## 阶段 1：代码副本与开发环境

目标：减少 WSL 读写 C 盘代码与模型文件带来的测试开销。

任务：

- 将 Windows 工程代码同步到 WSL ext4 路径，例如 `/home/jiamin/projects/CFIE`。
- 排除 `.git`、`.bench_logs`、`.tmp` 大日志、build 产物、venv、模型 cache 等不必要目录。
- 后续代码修改必须同步 Windows 与 WSL 两份代码。
- Windows 仍作为主工作区，WSL 副本用于编译与 122B 测试。

验收：

- WSL 副本可以 import `cfie`。
- WSL 副本可以运行已有 benchmark help / py_compile。
- WSL 运行不再依赖 `/mnt/c/.../CFIE` 的 Python 源码。

## 阶段 2：prefill burst 路径修正

目标：burst 路径复用 decode 的高效数据搬运路径。

当前问题：

- normal resident prepare 已经有近似目标路径：
  `CPU static expert-major -> CPU pinned stage prefix -> GPU runtime stage -> scatter to resident slots`
- burst target 当前会绕过 batch write，导致 CPU miss expert 可能逐 expert 写入 burst pool。

目标路径：

```text
当前层 requested unique experts = N, N <= prefill_burst_slots

resident hits:
  resident GPU slots -> burst pool slots

CPU misses:
  CPU static expert-major bundles
  -> CPU pinned stage prefix [missing_count, expert_bytes]
  -> one H2D to GPU runtime stage prefix
  -> GPU scatter to burst pool field tensors

execute:
  burst_pool._expert_map[global_expert_id] = burst_slot
  fused_marlin_moe uses burst execution layer
```

任务：

- 让 `_write_expert_bundles(..., target=SharedPrefillBurstPool)` 也可以走 runtime-ready batch write。
- 确认 CPU stage 和 GPU stage 可按 256 slots 容量复用，不在每层反复申请。
- 确认 missing expert 数小于 256 时只搬运 stage 前缀。
- 保留 resident GPU -> burst GPU copy 路径，后续再评估是否可避免 copy。
- 增加日志：burst unique experts、resident hits、CPU pinned hits、CPU pageable hits、H2D bytes、stage CPU pack ms、H2D ms、GPU scatter ms。

验收：

- `prefill_burst_slots=256` 时，unique experts 超过 `gpu_slots_per_layer` 不报错。
- CPU miss experts 合并为 batch H2D，不再逐 expert H2D。
- 缺失 2 个 expert 时只搬运 2-slot 前缀。

## 阶段 3：graph capture 收窄

目标：GUI agent 常见场景只录少数 shape，避免 graph pool 吃掉过多显存。

建议 shape：

- `500`：普通对话 / 小图输入
- `5000`：较长对话 / 少量视觉输入
- `50000`：视频帧理解主场景

注意：

- 50k graph capture 可能因为 activation/workspace 过大导致 OOM，必须先实验。
- 若 50k PIECE graph 显存不可接受，则长 prefill 应走 eager/Piece 部分捕获，decode 保留小 shape graph。

任务：

- 添加或确认 CLI 可显式传入 `--cudagraph-capture-sizes 500 5000 50000`。
- 确认不在列表内的 batch 通过 padding 到最近可用 graph 或 fallback eager。
- 分别测试 capture sizes：
  - `1`
  - `1 2 4 8`
  - `500 5000 50000`
  - `50000` only
- 记录 graph capture 时间与显存。

验收：

- graph 显存占用可解释。
- 生产配置不再默认录制大量无关 shape。

## 阶段 4：显存预算与参数组合

需要统计：

- 模型静态 dense + non-MoE 参数显存。
- MoE resident cache 显存：`gpu_slots_per_layer=16/24/32`。
- prefill burst pool 显存：`prefill_burst_slots=256`。
- runtime GPU stage 显存：最大 256 slots。
- CUDA graph pool 显存：三档 capture shape。
- KV cache 显存：256K 上下文。
- 其他峰值显存：attention/Mamba workspace、temporary tensor、compile workspace。

优先测试组合：

```bash
--gpu-slots-per-layer 16
--prefill-burst-slots 256
--prepare-cpu-copy-batch-size 8
--cpu-static-pinned-gb 56
--marlin-input-dtype fp8
--kv-cache-memory-bytes <逐步扫>
--cudagraph-capture-sizes 500 5000 50000
```

如果显存不足，按顺序降级：

1. 降低 graph capture sizes。
2. 降低 `gpu_slots_per_layer` 到 16。
3. 降低 KV cache 预算验证 decode，再单独评估 256K。
4. 暂时关闭 50k graph capture，只保留 decode graph。

## 阶段 5：dense 参数全量 GPTQ 量化

目标：当前 GPTQ 权重主要覆盖 MoE expert，dense 参数仍可能未充分量化。尝试对 dense 参数也做 GPTQ / Marlin 兼容量化，牺牲少量精度换显存与速度。

任务：

- 选择成熟量化工具链，优先考虑 GPTQModel / AutoGPTQ / llm-compressor / AutoRound 中与 Marlin 兼容性最好的路径。
- 准备校准数据，覆盖：
  - 中文 GUI agent 文本
  - OCR/屏幕描述
  - 多轮任务指令
  - 视觉 token 后的文本 decode 分布
- 生成全量 GPTQ 权重到 C 盘 HuggingFace cache 目录。
- 生成项目专用 Marlin-ready cache。
- 同步 cache 到 WSL 内部模型目录与 D 盘测试目录。

注意：

- W4A8/FP8 activation 不应该要求两份权重 cache；权重仍是 W4，activation scale 在推理时维护。
- 需要单独验证输出质量，不只看速度。

验收：

- 新权重可启动。
- 静态显存低于当前模型。
- decode 与 prefill 速度不低于当前 W4A16/W4A8 路径。
- 关键中文 GUI agent 样例输出可接受。

## 阶段 6：最终评测

必须输出以下数据：

- 三档 graph capture 显存：
  - 500
  - 5000
  - 50000
- 256K KV cache 显存。
- 模型静态参数显存。
- resident cache / burst pool / runtime stage 显存。
- pinned CPU static 覆盖层数与总 pinned GB。
- pageable -> pinned stage 带宽。
- pinned static -> H2D 带宽。
- burst prefill 中 resident hits / CPU hits / H2D bytes。
- 5w token prefill 耗时。
- decode 512 / 1024 tokens 速度：
  - 含首 token
  - 不含首 token steady TPS
- 输出文本正确性样例。

## 风险

- 50k CUDA Graph capture 可能本身占用过高显存，不一定适合作为默认生产配置。
- 256K KV cache 与 `prefill-burst-slots=256`、runtime GPU stage、graph pool 同时存在时，显存可能不足。
- dense 全量 GPTQ 量化是独立大任务，需要校准数据、量化时间和质量验收，不能和 runtime 路径改动混为一个验证口径。
- WSL ext4 可以提升源码和模型读取速度，但 122B 启动仍会受 pinned allocation、CPU static attach 和 graph capture 影响。

## 当前开工顺序

1. 同步 Windows 工程代码到 WSL ext4。
2. 检查并修改 burst target，让它复用 batch H2D + GPU scatter 路径。
3. 添加/整理 graph capture 三档配置脚本。
4. 先用小模型或 35B 调通，再跑 122B。
5. 122B 跑通后再进入 dense 全量 GPTQ 量化。
