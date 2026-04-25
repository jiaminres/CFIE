# Qwen3.5-122B-A10B Predictor CLI 冒烟记录

## 1. 测试时间

- 日期：2026-04-20
- 工作目录：`C:\Users\13642\PycharmProjects\vllm`

## 2. 模型与环境

- 训练档位：`qwen35-122b-a10b`
- 122B 模型快照：`C:\Users\13642\.cache\huggingface\hub\models--Qwen--Qwen3.5-122B-A10B-GPTQ-Int4\snapshots\5b9f0050d3ec98b0c81a7716776533c5eacebb64`
- 启动解释器：`py -V:Astral/CPython3.12.13`
- `PYTHONPATH`：
  - `C:\Users\13642\PycharmProjects\vllm\.venv\Lib\site-packages`
  - `C:\Users\13642\PycharmProjects\vllm\CFIE`
- 说明：本机直接执行 `.\.venv\Scripts\python.exe` 会被系统策略拦截，因此本次统一使用 `py` 的 3.12 解释器挂载 `.venv` 依赖运行。

## 3. 测试数据

- 数据集文件：`C:\Users\13642\PycharmProjects\vllm\CFIE\.tmp\predictor_122b_smoke\dataset.txt`
- 数据集内容为 3 行极小文本，仅用于 CLI 冒烟验证。

## 4. 最小可跑通参数

这组参数已经实测可跑通 `predictor-trace / predictor-train / predictor-eval`：

- `--profile qwen35-122b-a10b`
- `--steps 1`
- `--examples-per-step 1`
- `--samples 1`
- `--tokens-per-sample 8`
- `--dataset C:\Users\13642\PycharmProjects\vllm\CFIE\.tmp\predictor_122b_smoke\dataset.txt`
- `--epochs 1`（仅 `predictor-train`）

## 5. 四条主命令测试结果

### 5.1 predictor-trace

- 状态：通过
- 命令：

```powershell
$env:PYTHONPATH='C:\Users\13642\PycharmProjects\vllm\.venv\Lib\site-packages;C:\Users\13642\PycharmProjects\vllm\CFIE'
py -V:Astral/CPython3.12.13 -m cfie_training.cli.main predictor-trace `
  --profile qwen35-122b-a10b `
  --steps 1 `
  --examples-per-step 1 `
  --samples 1 `
  --tokens-per-sample 8 `
  --dataset C:\Users\13642\PycharmProjects\vllm\CFIE\.tmp\predictor_122b_smoke\dataset.txt `
  --output C:\Users\13642\PycharmProjects\vllm\CFIE\.tmp\predictor_122b_smoke\trace_122b.json `
  --json
```

- 产物：
  - `C:\Users\13642\PycharmProjects\vllm\CFIE\.tmp\predictor_122b_smoke\trace_122b.json`
- 结果摘要：
  - `example_count = 1`
  - `profile_name = qwen35-122b-a10b`
- 补充：
  - 该命令真实加载了 122B GPTQ 模型并完成了一次 teacher forward capture。

### 5.2 predictor-train

- 状态：通过
- 直接 122B teacher 采样版命令：

```powershell
$env:PYTHONPATH='C:\Users\13642\PycharmProjects\vllm\.venv\Lib\site-packages;C:\Users\13642\PycharmProjects\vllm\CFIE'
py -V:Astral/CPython3.12.13 -m cfie_training.cli.main predictor-train `
  --profile qwen35-122b-a10b `
  --steps 1 `
  --examples-per-step 1 `
  --samples 1 `
  --tokens-per-sample 8 `
  --dataset C:\Users\13642\PycharmProjects\vllm\CFIE\.tmp\predictor_122b_smoke\dataset.txt `
  --epochs 1 `
  --checkpoint-output C:\Users\13642\PycharmProjects\vllm\CFIE\.tmp\predictor_122b_smoke\predictor_122b_direct.ckpt `
  --json
```

- 更快的 trace 复用版命令：

```powershell
$env:PYTHONPATH='C:\Users\13642\PycharmProjects\vllm\.venv\Lib\site-packages;C:\Users\13642\PycharmProjects\vllm\CFIE'
py -V:Astral/CPython3.12.13 -m cfie_training.cli.main predictor-train `
  --profile qwen35-122b-a10b `
  --trace-input C:\Users\13642\PycharmProjects\vllm\CFIE\.tmp\predictor_122b_smoke\trace_122b.json `
  --epochs 1 `
  --checkpoint-output C:\Users\13642\PycharmProjects\vllm\CFIE\.tmp\predictor_122b_smoke\predictor_122b.ckpt `
  --json
```

- 产物：
  - `C:\Users\13642\PycharmProjects\vllm\CFIE\.tmp\predictor_122b_smoke\predictor_122b_direct.ckpt`
  - `C:\Users\13642\PycharmProjects\vllm\CFIE\.tmp\predictor_122b_smoke\predictor_122b.ckpt`
- 结果摘要：
  - `epochs = 1`
  - `example_count = 1`
  - `final_mean_loss = 0.694229006767273`
  - `final_recall_at_candidate_budget = 0.171875`
  - `final_recall_at_executed_budget = 0.0625`

### 5.3 predictor-eval

- 状态：通过
- 直接 122B teacher 采样版命令：

```powershell
$env:PYTHONPATH='C:\Users\13642\PycharmProjects\vllm\.venv\Lib\site-packages;C:\Users\13642\PycharmProjects\vllm\CFIE'
py -V:Astral/CPython3.12.13 -m cfie_training.cli.main predictor-eval `
  --profile qwen35-122b-a10b `
  --checkpoint C:\Users\13642\PycharmProjects\vllm\CFIE\.tmp\predictor_122b_smoke\predictor_122b.ckpt `
  --steps 1 `
  --examples-per-step 1 `
  --samples 1 `
  --tokens-per-sample 8 `
  --dataset C:\Users\13642\PycharmProjects\vllm\CFIE\.tmp\predictor_122b_smoke\dataset.txt `
  --json
```

- 更快的 trace 复用版命令：

```powershell
$env:PYTHONPATH='C:\Users\13642\PycharmProjects\vllm\.venv\Lib\site-packages;C:\Users\13642\PycharmProjects\vllm\CFIE'
py -V:Astral/CPython3.12.13 -m cfie_training.cli.main predictor-eval `
  --profile qwen35-122b-a10b `
  --checkpoint C:\Users\13642\PycharmProjects\vllm\CFIE\.tmp\predictor_122b_smoke\predictor_122b.ckpt `
  --trace-input C:\Users\13642\PycharmProjects\vllm\CFIE\.tmp\predictor_122b_smoke\trace_122b.json `
  --json
```

- 结果摘要：
  - `example_count = 1`
  - `mean_loss = 0.6911218166351318`
  - `recall_at_candidate_budget = 0.171875`
  - `recall_at_executed_budget = 0.0625`

## 6. 运行观察

- 本机单命令串行执行时，上述三条命令均可跑通。
- `predictor-trace / predictor-train(直接采样) / predictor-eval(直接采样)` 都会真实加载 122B teacher，因此耗时主要集中在模型加载。
- `predictor-train --trace-input ...` 与 `predictor-eval --trace-input ...` 明显更快，适合回归测试。
- 日志中会看到：
  - `Triton not installed or not compatible`
  - `Falling back from the inductor compilation backend to eager`
- 这些告警不会阻断本次 smoke，通过结果表明当前这组参数在本机上是可用的。

## 7. 注意事项

- 不要并行同时跑两个需要真实加载 122B teacher 的 predictor 命令；我在一次并行测试里复现过 `torch.AcceleratorError: CUDA error: out of memory`。
- 若只是验证 CLI 连通性，优先复用已生成的 `trace_122b.json`。
- 若要复现本次结果，建议保持单命令串行执行。
