# CFIE

CFIE（Capacity-First Inference Engine）是一个面向大模型本地部署场景的训推基础设施项目。
项目当前以 `vllm` 本地源码快照为基础，围绕 Windows / Linux 双平台下的大模型推理、
训练、MoE 卸载、量化执行、predictor 数据采集与训练流程持续推进。

## 项目立项目的

CFIE 的立项目标，不是单纯做一个“能跑起来”的推理 demo，而是先把后续客户端应用层所需的
基础设施搭起来。当前重点包括：

- Windows / Linux 双平台本地训推基建
  - 为后续客户端本地部署、调试、验证与交付提供统一的运行时底座。
- 面向大内存缓存利用的大模型推理
  - 充分利用本机 GPU / CPU / NVMe 的分层容量，把大内存当作热缓存和冷存储的一部分，
    支撑更大参数规模模型在本地运行。
- 面向 MoE 架构的加速与资源调度策略
  - 围绕 routed experts、权重卸载、批量上载、并行执行与通信开销控制，持续完善
    MoE 模型的执行效率。
- predictor 路线建设
  - 通过真实前向采集 hidden state 与路由标签，训练 predictor，用于后续推理侧的候选专家预测、
    预取与执行优化。
- 为客户端应用层提前建设基础设施
  - 目标是先把“模型加载、推理调度、训练采集、部署导出、资源规划”这些底层能力打牢，
    让后续上层产品形态可以直接复用这套基础设施。

当前仓库主要解决两类问题：

- 推理侧：复用真实推理引擎执行模型加载、调度、KV 规划、MoE 路由、量化推理与权重卸载。
- 训练侧：提供 `predictor-trace / predictor-train / predictor-eval / predictor-export`
  四段式流程，用真实前向采集 hidden state 与 router 标签，再训练 predictor 并导出部署产物。

项目当前更偏向“工程验证 + 主链落地”，而不是对外发布的通用发行版。README 重点说明：

1. 这个项目是什么。
2. 当前目录结构和主入口在哪里。
3. 应该如何构建与验证。
4. 目前还有哪些工作尚未完成。

## 项目结构

仓库中最重要的目录如下：

- `cfie/`
  - 推理运行时主链。
  - 包含 CLI、引擎、worker、模型执行、量化、并行与服务相关代码。
- `cfie_training/`
  - 训练侧子项目。
  - 包含训练配置、数据规划、predictor 采集 / 训练 / 评估 / 导出 CLI。
- `csrc/`
  - 原生扩展与 CUDA / C++ 实现。
- `tests/`
  - 单测与集成测试。
- `docs/`
  - 项目工作文档、注释规范、Smoke 测试记录与历史归档。
- `third_party/`
  - 第三方依赖源码快照，例如 `cutlass`、`vllm-flash-attn`。

## 当前能力概览

### 推理侧

- 支持 `chat` / `native-generate` / `serve` / `run-local` 等命令行入口。
- 支持量化模型加载，当前工程内重点验证过 GPTQ Marlin 路线。
- 支持 MoE 相关执行路径、CPU/NVMe 卸载与资源预算控制。
- 推理侧尽量复用真实引擎链路，而不是额外维护一套“简化版执行器”。

### 训练侧

- 支持基于真实模型前向的 predictor 训练数据采集。
- 支持从真实 hidden state 与 routed experts 构造教师轨迹。
- 支持 predictor checkpoint 训练、评估、导出 bundle。
- 当前已验证 `Qwen3.5-122B-A10B` 档位最小 Smoke 流程可跑通。

## 环境要求

### 基础要求

- Python：`>=3.10,<3.14`
- PyTorch：CUDA 版本 `torch==2.10.0`
- CMake：`>=3.26.1`
- Ninja
- CUDA Toolkit（需要可用的 `nvcc`）

### Windows 额外要求

- Visual Studio 2022 Build Tools
- 建议已正确安装 CUDA Toolkit，并设置好 `CUDA_HOME` 或 `CUDA_PATH`

### Linux 额外要求

- GCC / G++ 或 Clang 等可用本地编译工具链

## 构建命令

下面给出推荐构建方式。项目会通过 `setup.py + CMake` 自动编译原生扩展，因此最常用的入口是
直接执行 `pip install -e .`。

### 1. 创建虚拟环境

Windows：

```powershell
cd C:\Users\13642\PycharmProjects\vllm\CFIE
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux：

```bash
cd /path/to/CFIE
python3.12 -m venv .venv
source .venv/bin/activate
```

### 2. 安装构建依赖

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install "torch==2.10.0"
python -m pip install cmake ninja packaging jinja2
```

如果你的 `torch` 已经是匹配 CUDA 的可用版本，可以跳过重新安装。

### 3. 编译并以开发模式安装

```bash
python -m pip install -e .
```

这一步会自动：

- 读取 `pyproject.toml`
- 调用 `setup.py`
- 进入 CMake 构建原生扩展
- 安装 `cfie` 与 `cfie_training` 两个 Python 包

### 4. 最小验证命令

验证推理 CLI 是否已注册：

```bash
python -m cfie.cli.main --help
```

验证训练 CLI 是否已注册：

```bash
python -m cfie_training.cli.main --help
```

## 常用命令

### 推理侧命令

交互式聊天：

```bash
python -m cfie.cli.main chat --model <模型路径>
```

单次生成：

```bash
python -m cfie.cli.main native-generate --model <模型路径> --prompt "你好"
```

### 训练侧命令

构造 predictor 教师轨迹：

```bash
python -m cfie_training.cli.main predictor-trace \
  --profile qwen35-122b-a10b \
  --steps 1 \
  --examples-per-step 1 \
  --samples 1 \
  --tokens-per-sample 8 \
  --dataset <数据集路径> \
  --output <trace.json> \
  --json
```

训练 predictor：

```bash
python -m cfie_training.cli.main predictor-train \
  --profile qwen35-122b-a10b \
  --trace-input <trace.json> \
  --epochs 1 \
  --checkpoint-output <predictor.ckpt> \
  --schema-output <predictor.schema.json> \
  --json
```

评估 predictor：

```bash
python -m cfie_training.cli.main predictor-eval \
  --profile qwen35-122b-a10b \
  --checkpoint <predictor.ckpt> \
  --trace-input <trace.json> \
  --json
```

导出部署 bundle：

```bash
python -m cfie_training.cli.main predictor-export \
  --checkpoint <predictor.ckpt> \
  --output-dir <bundle_dir> \
  --json
```

## 当前验证情况

截至当前仓库状态，以下方向已经形成可复现工程链路：

- Windows 单卡下的最小推理主链可运行。
- `Qwen3.5-35B-A3B` 与 `Qwen3.5-122B-A10B` 路线已有验证基线。
- predictor 训练侧四段式 CLI 已打通：
  - `predictor-trace`
  - `predictor-train`
  - `predictor-eval`
  - `predictor-export`
- `Qwen3.5-122B-A10B` 的最小 predictor Smoke 记录见：
  - `docs/122B_predictor_CLI_smoke.md`

## 当前未完成工作

项目仍处于持续落地阶段，以下事项目前尚未完全收敛：

### 1. predictor 推理侧正式挂载仍需继续完善

- 当前训练侧数据采集、训练、评估、导出主链已经建立。
- predictor 在真实推理链路中的长期挂载、策略切换、更多运行时门禁仍需要继续打磨。

### 2. 更多并行 / 模型组合回归仍需补齐

- 当前已验证单卡主链与部分关键模型档位。
- 更复杂的 TP / PP / 多卡组合仍需要更系统的回归矩阵。

### 3. Windows 下部分高频回退路径仍待继续收敛

- 在当前 CUDA / Triton 可用性受限的环境下，部分路径仍会回退到较保守实现。
- 后续仍需继续减少关键热点上的 fallback，并扩大真实推理回归覆盖。

### 4. 文档与配置收敛仍在进行中

- 历史阶段文档较多，部分已经归档。
- README 之外的文档体系仍需继续清理、合并与统一口径。

### 5. 发布级工程封装尚未完成

- 当前更适合开发、验证和实验使用。
- 面向稳定发布的安装包、默认配置、自动化测试矩阵与发布流程还未完全定型。

## 文档索引

- 文档导航：`docs/00_文档导航.md`
- 122B predictor Smoke：`docs/122B_predictor_CLI_smoke.md`
- 注释规范：`docs/代码注释工作文档.md`

## 说明

- 本项目当前以工程主链跑通为优先目标。
- 若你准备将其上传到 GitHub，建议先补充 `.gitignore`，避免把模型、缓存、编译产物、
  `.venv`、`.tmp`、日志和本地 IDE 文件直接提交。
