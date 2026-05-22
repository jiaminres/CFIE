# CFIE

![CFIE logo](docs/assets/cfie_logo.svg)

CFIE, Capacity-First Inference Engine, 是基于 vLLM 代码演进的客户端本地大模型推理项目。当前主线目标是：在单卡显存不足以完整常驻大模型 MoE experts 时，通过 CPU 内存、pinned memory、GPU resident slots 和 NVMe 模型缓存配合，让 Qwen3.5 GPTQ MoE 模型可以在本地 32GB 显卡上完成长上下文推理和 OpenAI 兼容服务。

CFIE 的定位是让智能设备拥有自己的本地推理大脑。未来的 GUI Agent、桌面助手、移动端自动化、个人数据分析和本地应用控制，不应该把所有感知、记忆和决策都交给远端共享推理集群。更合理的方向是：每个客户端设备都具备低延迟、可离线、可私有化、可持续适配用户习惯的独立大脑，上层应用通过标准 API 调用它完成自动化任务。

## 支持模型

当前重点支持 Qwen3.5 MoE GPTQ Int4 量化模型：

| 模型 | Hugging Face |
| --- | --- |
| Qwen3.5-122B-A10B-GPTQ-Int4 | [Qwen/Qwen3.5-122B-A10B-GPTQ-Int4](https://huggingface.co/Qwen/Qwen3.5-122B-A10B-GPTQ-Int4) |
| Qwen3.5-35B-A3B-GPTQ-Int4 | [Qwen/Qwen3.5-35B-A3B-GPTQ-Int4](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4) |

推荐把模型文件和 CFIE 生成的 Marlin-ready cache 放在高速 NVMe 盘上。当前测试机器使用 D 盘 PCIe 5 高速盘承载模型和 cache。

## 推理架构

CFIE 使用 MoE tiered cache，把模型权重、CPU 内存、pinned memory 和 GPU resident slots 组织成可运行的本地推理能力。

![CFIE inference flow](docs/assets/cfie_inference_flow.svg)

核心机制：

- CPU static mirror：启动期把 MoE expert 转成运行时可直接使用的 Marlin-ready expert-major bundle。
- CPU pinned static mirror：按预算把部分 expert 放入 pinned memory，减少 H2D 前的中转成本。
- GPU resident slots：每层只常驻有限数量的高频 expert，例如 `--gpu-slots-per-layer 16`。
- Runtime stage：推理期只处理当前缺失 expert，写入可复用 CPU/GPU stage 后再 scatter 到目标 GPU slot。
- Prefill burst pool：长 prefill 中某层触发 expert 数超过 resident slots 时，用临时 burst pool 承接执行。
- W4A8 / FP8 activation：GPTQ Marlin 路径支持 `--marlin-input-dtype fp8`，当前推荐默认使用。

## 实测环境

以下配置是本文档中推荐参数和测速结果的来源，其他机器需要重新验证：

| 硬件 | 配置 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5090 32GB，driver 591.86 |
| 内存 | 160 GiB installed，约 157.5 GiB usable，3600 MT/s |
| 存储 | D 盘 PCIe 5 高速 NVMe，用于模型文件和 CFIE cache |
| 测试模型 | Qwen3.5-122B-A10B-GPTQ-Int4 |

本地模型路径示例：

```text
D:\models\Qwen3.5-122B-A10B-GPTQ-Int4\snapshots\5b9f0050d3ec98b0c81a7716776533c5eacebb64
```

## 实测速度

以下数据来自当前 Windows + RTX 5090 32GB + Qwen3.5-122B-A10B-GPTQ-Int4 的本地 decode 测试，配置为 `gpu_slots_per_layer=16`、`prefill_burst_slots=256`、`prepare_cpu_copy_threads=32`、`cpu_static_pinned_gb=44`、`marlin_input_dtype=fp8`、`MTP=1`、关闭 CUDA graph。

| 场景 | 结果 |
| --- | ---: |
| 512 token decode steady TPS | 约 `7.3-7.7 tok/s` |
| 512 token decode tail 64 TPS | 最高约 `8.2 tok/s` |
| 首 token 延迟 | 约 `1.8-2.0 s` |

当前 Windows 环境下 pinned static mirror 主要覆盖前 37 层，后 11 层仍需要 pageable CPU memory 到 pinned runtime stage 的中转，prepare 端主要瓶颈在这部分 CPU pack。若部署在能够让 MoE static mirror 全量进入 pinned memory 的环境中，例如更充足内存和更高 pinned 上限的原生 Linux，按当前逐层 prepare 耗时估算，decode 速度有机会提升到约 `11 tok/s`。该数值是基于瓶颈拆分的工程估算，不是当前 Windows 配置的实测值。

## 安装

以下命令以 Windows PowerShell 为例。

创建虚拟环境：

```powershell
py -3.12 -m venv ..\.venv
..\.venv\Scripts\python.exe -m pip install -U pip setuptools wheel
```

安装构建依赖：

```powershell
..\.venv\Scripts\python.exe -m pip install cmake ninja packaging jinja2
```

安装 CUDA 版 PyTorch。必须是 CUDA wheel，不能是 CPU wheel；本项目 `pyproject.toml` 约束 `torch==2.10.0`：

```powershell
..\.venv\Scripts\python.exe -m pip install torch==2.10.0 torchvision --index-url <your-torch-cuda-wheel-index>
```

安装 CFIE：

```powershell
$env:MAX_JOBS = "8"
..\.venv\Scripts\python.exe -m pip install --no-build-isolation -e .
```

检查 CLI：

```powershell
..\.venv\Scripts\python.exe -m cfie.cli.main --help
..\.venv\Scripts\python.exe -m cfie.entrypoints.openai.api_server --help
```

安装注意事项：

- 使用 `--no-build-isolation`，避免 pip 在隔离构建环境里误装 CPU 版 torch。
- Windows 需要 Visual Studio 2022 Build Tools、CUDA Toolkit、CMake、Ninja。
- Windows 和 WSL 应分别使用自己的 venv，不要混用 site-packages 或编译产物。

## 推荐 CLI 参数

128K GUI Agent / 长上下文推荐基线：

```text
--max-model-len 128000
--max-num-seqs 1
--max-num-batched-tokens 8192
--kv-cache-memory-bytes 4000000000
--gpu-slots-per-layer 16
--prefill-burst-slots 256
--prepare-cpu-copy-threads 32
--cpu-static-pinned-gb 40
--enable-prefix-caching
--marlin-input-dtype fp8
--speculative-config '{"method":"mtp","num_speculative_tokens":1}'
--default-chat-template-kwargs '{"enable_thinking":false}'
--enforce-eager
```

参数含义：

| 参数 | 推荐值 | 说明 |
| --- | ---: | --- |
| `--max-model-len` | `128000` | 当前实测更稳的长上下文档位。 |
| `--max-num-seqs` | `1` | GUI Agent 单会话优先，避免为多并发序列放大 KV 预算。 |
| `--max-num-batched-tokens` | `8192` | prefill chunk 大小，长上下文输入推荐使用该档位。 |
| `--kv-cache-memory-bytes` | `4000000000` | 手动预留约 3.73 GiB KV，适配 128K 上下文。 |
| `--gpu-slots-per-layer` | `16` | 每层 GPU resident expert slots。 |
| `--prefill-burst-slots` | `256` | 长 prefill 的临时 expert 执行池。 |
| `--prepare-cpu-copy-threads` | `32` | CPU static mirror 到 runtime stage 的并行 copy 线程数。 |
| `--cpu-static-pinned-gb` | `40` | OpenAI / VL 服务推荐值，给 runtime stage 留 pinned headroom。 |
| `--enable-prefix-caching` | 开启 | GUI Agent 场景必须显式开启。 |
| `--marlin-input-dtype` | `fp8` | 当前推荐 W4A8 路径。 |
| `--enforce-eager` | 开启 | 当前默认不启用 CUDA graph。 |

## 启动 OpenAI 服务

### 文本服务

```powershell
..\.venv\Scripts\python.exe -m cfie.entrypoints.openai.api_server `
  --model D:\models\Qwen3.5-122B-A10B-GPTQ-Int4\snapshots\5b9f0050d3ec98b0c81a7716776533c5eacebb64 `
  --served-model-name qwen35 `
  --host 127.0.0.1 `
  --port 8000 `
  --max-model-len 128000 `
  --max-num-seqs 1 `
  --max-num-batched-tokens 8192 `
  --kv-cache-memory-bytes 4000000000 `
  --gpu-slots-per-layer 16 `
  --prefill-burst-slots 256 `
  --prepare-cpu-copy-threads 32 `
  --cpu-static-pinned-gb 40 `
  --enable-prefix-caching `
  --language-model-only `
  --skip-mm-profiling `
  --marlin-input-dtype fp8 `
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' `
  --default-chat-template-kwargs '{"enable_thinking":false}' `
  --enforce-eager
```

### VL / GUI Agent 服务

```powershell
..\.venv\Scripts\python.exe -m cfie.entrypoints.openai.api_server `
  --model D:\models\Qwen3.5-122B-A10B-GPTQ-Int4\snapshots\5b9f0050d3ec98b0c81a7716776533c5eacebb64 `
  --served-model-name qwen35-vl `
  --host 127.0.0.1 `
  --port 8000 `
  --max-model-len 128000 `
  --max-num-seqs 1 `
  --max-num-batched-tokens 8192 `
  --kv-cache-memory-bytes 4000000000 `
  --gpu-slots-per-layer 16 `
  --prefill-burst-slots 256 `
  --prepare-cpu-copy-threads 32 `
  --cpu-static-pinned-gb 40 `
  --enable-prefix-caching `
  --marlin-input-dtype fp8 `
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' `
  --default-chat-template-kwargs '{"enable_thinking":false}' `
  --limit-mm-per-prompt '{"image":8,"video":0}' `
  --enforce-eager
```

## API 示例

Responses 文本请求：

```powershell
curl http://127.0.0.1:8000/v1/responses `
  -H "Content-Type: application/json" `
  -d '{
    "model": "qwen35",
    "input": "用一句话介绍 CFIE。"
  }'
```




## 微调模块设计稿

`cfie_training/` 是后续训练、微调和 GUI Agent 闭环数据工作的开发区域。当前微调模块仍在开发中，README 暂不承诺稳定训练 API；设计目标是把本地推理、数据回流、参数分层、训练调度和评估闭环逐步接入同一套客户端智能体基础设施。

![Qwen3.5-122B training base design](docs/assets/qwen35_122b_training_base.svg)

## 文档与实验记录

详细实验记录放在：

```text
docs/experiments/
```

当前长上下文和 GUI Agent 相关记录主要参考：

```text
docs/experiments/2026-05-21_long_context_gui_agent.md
docs/experiments/2026-05-21_kv_prefill_nograph_warm.md
```

后续实验应按日期新增文档，不覆盖旧记录，避免重复验证同一组参数。
