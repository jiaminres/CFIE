# 122B Predictor CLI Smoke

## 1. 目标

验证 122B 档位下两条保留命令可以跑通：

- `predictor-trace`
- `predictor-train`

## 2. 最小参数

- `--profile qwen35-122b-a10b`
- `--steps 1`
- `--examples-per-step 1`
- `--samples 1`
- `--tokens-per-sample 8`
- `--dataset C:\Users\13642\PycharmProjects\vllm\CFIE\.tmp\predictor_122b_smoke\dataset.txt`
- `--epochs 1`（仅 `predictor-train`）

## 3. predictor-trace

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

- 产物：`C:\Users\13642\PycharmProjects\vllm\CFIE\.tmp\predictor_122b_smoke\trace_122b.json`
- 说明：该命令会真实加载 122B teacher 并执行一次 forward capture。

## 4. predictor-train

### 4.1 直接采样训练

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

### 4.2 复用 trace 训练

```powershell
$env:PYTHONPATH='C:\Users\13642\PycharmProjects\vllm\.venv\Lib\site-packages;C:\Users\13642\PycharmProjects\vllm\CFIE'
py -V:Astral/CPython3.12.13 -m cfie_training.cli.main predictor-train `
  --profile qwen35-122b-a10b `
  --trace-input C:\Users\13642\PycharmProjects\vllm\CFIE\.tmp\predictor_122b_smoke\trace_122b.json `
  --epochs 1 `
  --checkpoint-output C:\Users\13642\PycharmProjects\vllm\CFIE\.tmp\predictor_122b_smoke\predictor_122b.ckpt `
  --json
```

- 产物只保留 checkpoint。
- `--trace-input` 路径用于快速回归。

## 5. 运行观察

- `predictor-trace` 与直接采样版 `predictor-train` 都会真实加载 122B teacher。
- `predictor-train --trace-input ...` 明显更快，适合回归测试。
- 不建议并行运行多个需要真实加载 122B teacher 的命令。
