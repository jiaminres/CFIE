# CFIE

CFIE（Capacity-First Inference Engine）是一个基于本地 `vllm` 源码快照演进的大模型本地推理与训练基础设施项目。
它的目标是在显存不足时，把 GPU、CPU 内存和 NVMe 组织成分层容量体系，让更大参数规模的模型可以在本地设备上完成推理、训练、回放和应用承载。

## 当前主线

当前推理架构已经从旧的 predictor / predicator 预取路线切换到确定性的 MoE 分层专家缓存路线：

- CPU static mirror：初始化阶段把需要兜底的 MoE experts 物化成 CPU 侧 runtime-ready mirror。
- GPU resident slots：每个 MoE 层只保留有限数量的常驻 GPU expert slots。
- Expert-major staging：按 expert 粒度把 runtime-ready bundle 打包到可复用 stage storage，再搬运到 GPU。
- Pinned / pageable 分层：CPU static mirror 可以按预算或层号选择 pinned；runtime staging 优先使用 pinned buffer，失败时回退 pageable。
- 多路并行 CPU copy：pageable -> pinned / stage 的整理使用原生 C++ thread pool 或 Python 线程池兜底。
- Prefill burst：大 prefill chunk 装不进 resident slots 时，使用共享临时 GPU burst pool 执行，不强行污染常驻 slots。
- FP8 Marlin activation：GPTQ Marlin 路径支持标准 CLI `--marlin-input-dtype fp8`，Marlin-ready cache 会把 activation dtype 纳入缓存 key。

旧专家预测实验和旧 runtime load strategy 对比结论不再代表当前推理主链。

## 推理架构

### MoE Tiered Cache

当前 MoE 分层缓存由 `cfie/offload/policy.py` 规划，并在模型加载后通过 `cfie/offload/weight_offload.py` 挂接到 `FusedMoE` 层。

核心对象：

- `MoeTieredCachePlan`
  - 规划 GPU resident slots、prefill burst slots、CPU static mirror、pinned 层和 prepare 阶段 copy batch。
- `LayerTieredExpertCacheController`
  - 每层一个 controller，负责 slot 映射、router-score replacement、CPU static source bundle、H2D staging 和 GPU scatter。
- `SharedRuntimeExpertStagePool`
  - 模型级复用的 CPU/GPU expert-major stage storage，避免每层、每批次重复分配。
- `SharedPrefillBurstPool`
  - 模型级共享临时 GPU expert slots，用于大 prefill chunk 的 burst 执行。

### 执行路径

`cfie/model_executor/layers/fused_moe/runner/default_moe_runner.py` 是 MoE runner 与 tiered cache 的桥接层。

当前路径分两类：

1. Decode 或能装入 resident slots 的小 prefill chunk：
   - runner 统计本 chunk 触达的 unique experts。
   - 调用 controller 的 `prepare(topk_ids, topk_weights, router_probs)`。
   - `prepare()` 使用当前 router scores 选择缺失 experts 和 victim slots。
   - 缺失 experts 只从 CPU static mirror 读取 runtime-ready bundle。
   - stage 到 GPU 后，执行原 quant method。

2. unique experts 超过 resident slots 但能装入 burst pool 的大 prefill chunk：
   - runner 调用 `controller.run_prefill_burst(...)`。
   - burst pool 使用临时 expert map 代理当前 `FusedMoE` 层执行。
   - 已 resident 的 experts 可直接从 GPU 拷入 burst slots。
   - 未 resident 的 experts 从 CPU static mirror 写入 burst pool。
   - 这条路径不要求把所有 experts 换入常驻 resident slots。

因此，`prepare()` 现在是 resident 路径的当前请求 staging 函数，不再承担未来层预测式预取。大 prefill 的 burst 路径会绕过 resident replacement，直接使用临时 burst pool。

### CPU Static Mirror

当前主线要求 CPU static mirror 覆盖本层可路由 experts。`prepare()` 中的 source bundle 只接受 CPU static pool：

- 若 expert 不在 CPU static pool，会直接报错。
- 若 bundle 不是 runtime-ready，会直接报错。
- prepare 阶段不再按需从 NVMe materialize 或重新 repack expert。

GPTQ Marlin 路线会在初始化阶段把原始 expert 权重转换成 runtime-ready tensors，例如：

- `runtime.w13_qweight`
- `runtime.w2_qweight`
- `runtime.w13_scales`
- `runtime.w2_scales`
- `runtime.w13_qzeros`
- `runtime.w2_qzeros`
- desc_act 场景下的 `g_idx` 与 sort indices

这一步会调用 Marlin repack 和 scale permutation。CPU static mirror 初始化完成后，prepare 阶段只做当前缺失 expert 的整理、H2D 和 slot scatter。

### Marlin-Ready Cache

Marlin-ready cache 用于避免每次启动都重复执行 GPTQ Marlin runtime-ready 预处理。

默认缓存位置：

```text
<model_path>/.cfie_marlin_ready_cache/v1/
```

相关开关：

- `CFIE_MARLIN_READY_CACHE=0`：禁用 Marlin-ready cache。
- `CFIE_MARLIN_READY_CACHE_DIR=<path>`：覆盖缓存根目录。

缓存 metadata 会记录模型 stamp、layer、expert ids、量化参数、desc_act、`is_a_8bit` 和 `marlin_input_dtype`。因此 `a16`、`int8`、`fp8` activation 模式不会误用同一份缓存。

### Stage Copy 与 GPU Scatter

prepare 阶段的缺失 experts 会按 `_prepare_cpu_copy_batch_size` 分批处理。

主要数据流：

```text
CPU static runtime-ready bundle
  -> reusable expert-major CPU stage storage
  -> reusable GPU stage storage
  -> index_copy_ scatter into target GPU slots
```

如果 CPU static bundle 已经是 contiguous expert storage，路径会优先使用原生 CPU copy op：

```text
copy_expert_slices_to_stage_cpu
```

该 op 在 `csrc/torch_bindings.cpp` 中维护持久线程池，按字节区间并行复制 expert slices。没有命中原生 fast path 时，会使用 `SharedRuntimeExpertStagePool` 的持久 Python `ThreadPoolExecutor`，再退化到本地线程池或串行复制。

### FP8 Activation

GPTQ Marlin activation dtype 由标准 CLI 控制：

```bash
--marlin-input-dtype fp8
```

可选值目前为：

- `int8`
- `fp8`
- 未设置时使用默认 activation 路线

benchmark 脚本也提供 `--marlin-input-dtype`，会直接写入 `CompilationConfig.marlin_input_dtype`。

## 关键参数

这些参数由 `OffloadConfig`、engine args、native CLI 和 benchmark 脚本共享：

| 参数 | 作用 |
| --- | --- |
| `--moe-cpu-budget-gb` | MoE tiered cache 可使用的 CPU 内存预算上限，`0` 表示 planner 自动控制。 |
| `--moe-cpu-min-free-gb` | 为系统、page cache、pinned buffer 等保留的最小 CPU 空闲内存。 |
| `--gpu-slots-per-layer` | 每层 GPU resident expert slots 上限，`0` 表示 planner 自动控制；显式正值需要满足 top-k 执行约束。 |
| `--prefill-burst-slots` | 共享 prefill burst pool 的临时 GPU expert slots 数量。 |
| `--cpu-static-preprocess-batch-size` | 初始化 CPU static mirror 时每批预处理的 expert 数量，`0` 表示自动估算。 |
| `--cpu-static-pinned-gb` | 允许放入 pinned memory 的 CPU static expert mirror 预算。 |
| `--cpu-static-pinned-layers` | 指定 pinned 的 MoE layer index 或范围，例如 `0-23,30`。 |
| `--prepare-cpu-copy-batch-size` | prepare 阶段 CPU copy / stage pack 的并行 batch 粒度。 |
| `CFIE_BENCH_TIMING=1` | 打开 prepare、H2D、GPU scatter 等细分计时日志，仅用于诊断。 |
| `--marlin-input-dtype fp8` | 覆盖 GPTQ Marlin activation dtype。 |
| `--allow-tiered-moe-compile` | 允许 tiered MoE 在 prepare/MoE 边界外执行 PIECEWISE compile/cudagraph。 |

## Qwen3.5-122B 训练基座架构

下面是 122B 训练基座的目标架构。核心思路是把全量 FP32 主参数放在 NVMe，CPU 内存承载当前训练热参数、GPTQ 冷专家缓存和 Adam 状态，GPU 只保留前向所需的 dense / hot MoE 影子参数、冷专家 resident cache 与固定大小梯度 bucket 环。

![Qwen3.5-122B 训练基座架构](docs/assets/qwen35_122b_training_base.svg)

详细设计见 `docs/架构图文档/01_训练主线一_训练基座/01_Qwen3.5-122B训练基座详细设计.md`。

## 项目结构

仓库中最重要的目录如下：

- `cfie/`
  - 推理运行时主链，包含 CLI、engine、worker、模型执行、量化、OpenAI-compatible API 和服务入口。
- `cfie/offload/`
  - MoE tiered cache、CPU static mirror、Marlin-ready cache、pinned staging、prefill burst 和 planner。
- `cfie/model_executor/`
  - 模型层执行、FusedMoE runner、量化后端和 GPU worker 集成。
- `cfie/entrypoints/openai/`
  - 从 vLLM 继承并改造的 OpenAI-compatible HTTP API，包括 chat completions、completions、responses、models 等接口。
- `cfie_training/`
  - 训练侧子项目和训练基座相关代码。当前 README 只描述当前训练基座和通用训练侧能力。
- `csrc/`
  - 原生扩展与 CUDA / C++ 实现，包括 stage copy thread pool、MoE batch load、Marlin / FP8 相关算子。
- `benchmarks/`
  - decode、long prefill + decode、VL smoke 等当前推理路径 benchmark。
- `tests/`
  - 单测与集成测试，包含当前 tiered cache 参数和 MoE cache 行为测试。
- `docs/`
  - 项目工作文档、架构图、Smoke 记录与历史归档。
- `third_party/`
  - 第三方依赖源码快照，例如 `cutlass`、`vllm-flash-attn`。

## 当前能力概览

### 推理侧

- 支持 native v1 engine 的 `chat` / `native-generate` / `serve` / `run-local` 入口。
- 支持 OpenAI-compatible API server 入口。
- 支持 GPTQ Marlin 与非量化 MoE tiered cache 路线。
- 支持 CPU static mirror、GPU resident slots、prefill burst pool 和 reusable staging pool。
- 支持 prepare 阶段 router-score slot replacement。
- 支持 GPTQ Marlin `int8` / `fp8` activation dtype override。
- 支持 benchmark timing 日志拆分 prepare、CPU pack、H2D、GPU scatter 和 apply。

### 训练侧

- 训练基座继续围绕本地大模型训练、参数分层、数据规划和后续 GUI 自动化闭环建设。
- 历史专家预测采集和训练代码可能仍存在于历史或工作区中，但它不再是当前推理架构的默认说明对象。

## 环境要求

基础要求：

- Python：`>=3.10,<3.14`
- PyTorch：当前工程按 CUDA 版 `torch==2.10.0` 维护
- CMake：`>=3.26.1`
- Ninja
- CUDA Toolkit，需要可用的 `nvcc`

Windows 额外要求：

- Visual Studio 2022 Build Tools
- 建议本机 CUDA Toolkit 与 PyTorch CUDA 运行时尽量对齐

Linux / WSL 额外要求：

- GCC / G++ 或 Clang 等可用本地编译工具链

当前工作区的解释器约定：

- Windows：父目录 `.venv`，例如 `C:\Users\13642\PycharmProjects\vllm\.venv\Scripts\python.exe`
- WSL：当前目录 `.wsl-venv`，例如 `CFIE/.wsl-venv/bin/python`

不要混用 Windows venv 和 WSL venv 的 site-packages、编译产物或路径。

## 构建命令

项目通过 `setup.py + CMake` 编译原生扩展。推荐使用 `--no-build-isolation`，避免 `pip` 在隔离构建环境中拉起 CPU 版 PyTorch，导致 CUDA 扩展误判。

安装构建依赖：

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install cmake ninja packaging jinja2
```

安装项目约定的 GPU 版 PyTorch：

```bash
python -m pip install "torch==2.10.0" "torchvision==0.25.0" --index-url https://download.pytorch.org/whl/cu126
```

开发模式安装：

```bash
python -m pip install --no-build-isolation -e .
```

限制编译并发：

Windows PowerShell：

```powershell
$env:MAX_JOBS = "8"
python -m pip -v install --no-build-isolation -e .
```

Linux / WSL：

```bash
export MAX_JOBS=8
python -m pip -v install --no-build-isolation -e .
```

说明：

- `MAX_JOBS` 会传给 `setup.py` 控制 `cmake --build -j` 的并发数。
- 直接使用 `python -m pip install -e .` 可能触发隔离构建，并安装错误的 CPU 版 `torch`。
- Windows 下若已安装 `ninja`，CFIE 会优先使用 Ninja 生成器。
- MSVC 输出里的 `注意: 包含文件:` / `Note: including file:` 通常只是依赖扫描信息，不是编译错误。

## 常用命令

验证 CLI 是否注册：

```bash
python -m cfie.cli.main --help
python -m cfie.entrypoints.cli.main --help
python -m cfie_training.cli.main --help
```

Native 单次生成：

```bash
python -m cfie.cli.main native-generate \
  --model <model_path> \
  --prompt "你好" \
  --gpu-slots-per-layer 24 \
  --prefill-burst-slots 256 \
  --prepare-cpu-copy-batch-size 8 \
  --cpu-static-pinned-gb 0
```

Native 交互聊天：

```bash
python -m cfie.cli.main chat --model <model_path>
```

OpenAI-compatible API server：

```bash
python -m cfie.entrypoints.cli.main serve <model_path> \
  --gpu-slots-per-layer 24 \
  --prefill-burst-slots 256 \
  --prepare-cpu-copy-batch-size 8
```

OpenAI-compatible 客户端 quick chat：

```bash
python -m cfie.entrypoints.cli.main chat \
  --url http://localhost:8000/v1 \
  --quick "你好"
```

Decode benchmark：

```bash
python benchmarks/run_decode_512.py \
  --model <model_path> \
  --gpu-slots-per-layer 24 \
  --prepare-cpu-copy-batch-size 8 \
  --marlin-input-dtype fp8
```

Long prefill + decode benchmark：

```bash
python benchmarks/run_long_prefill_decode.py \
  --model <model_path> \
  --gpu-slots-per-layer 24 \
  --prefill-burst-slots 256 \
  --prepare-cpu-copy-batch-size 8 \
  --marlin-input-dtype fp8
```

打开细分计时：

```bash
export CFIE_BENCH_TIMING=1
```

Windows PowerShell：

```powershell
$env:CFIE_BENCH_TIMING = "1"
```

## 当前验证重点

截至当前 README 更新，推理主线的代码关注点是：

- `OffloadConfig` 与 engine args 已暴露当前 tiered cache 参数。
- MoE runner 会在 resident 路径调用 `prepare()`，在大 prefill 场景调用 burst pool。
- CPU static mirror 会在初始化阶段 eager materialize runtime-ready expert bundles。
- prepare 阶段不再执行未来专家预测预取，也不再从 NVMe 按需 repack。
- C++ 原生 op 已提供 CPU expert slice 到 stage storage 的并行复制能力。
- 单测覆盖了当前 engine 参数注册和 MoE tiered cache 的关键行为。

项目仍在快速迭代中，README 只描述当前推理主线和主要入口；历史实验结论请以 `docs/` 中归档文档为准。
