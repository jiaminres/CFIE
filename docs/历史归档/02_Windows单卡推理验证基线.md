# Windows 单卡推理验证记录（2026-04-12）

## 1. 目的

本记录用于固定当前 Windows 单卡环境下，`Qwen3.5-35B-A3B-GPTQ-Int4` 与 `Qwen3.5-122B-A10B-GPTQ-Int4` 的最小可复现启动命令。

这份记录的意义是：

- 先确认“当前单卡主线已经真实能跑”。
- 后续再做 `TP / EPLB / 更广泛量化路径 / desc_act=True` 兼容时，有一份明确的单卡回归基线。

## 2. 环境摘要

- 系统：Windows 本地桌面环境
- GPU：`NVIDIA GeForce RTX 5090`
- 显存总量：约 `32 GiB`
- 当次验证空闲显存：约 `30.2 GiB`
- Python：`C:\Users\13642\PycharmProjects\vllm\.venv\Scripts\python.exe`
- 额外环境变量：`VLLM_NO_USAGE_STATS=1`

说明：

- 需要显式设置 `VLLM_NO_USAGE_STATS=1`，否则当前环境会因为 `C:\Users\13642\.config\cfie\usage_stats.json` 没有写权限而抛出后台线程权限错误。
- 当前 Windows CUDA 运行时下 `Triton` 不可用，因此会看到一批 `falling back to ... PyTorch reference path` 的 warning；但这不影响本次最小推理主线成立。

## 3. 35B 验证

### 3.1 模型路径

`C:\Users\13642\.cache\huggingface\hub\models--Qwen--Qwen3.5-35B-A3B-GPTQ-Int4\snapshots\33f4e5e615e1f29a7b218906555ea6fe2d09c741`

### 3.2 最终通过命令

```powershell
$env:VLLM_NO_USAGE_STATS='1'
.venv\Scripts\python.exe -m cfie.cli.main native-generate `
  --model "C:\Users\13642\.cache\huggingface\hub\models--Qwen--Qwen3.5-35B-A3B-GPTQ-Int4\snapshots\33f4e5e615e1f29a7b218906555ea6fe2d09c741" `
  --prompt "请只回答三个字：一加一。" `
  --quantization gptq_marlin `
  --dtype fp16 `
  --max-model-len 2048 `
  --max-new-tokens 16 `
  --gpu-memory-utilization 0.88 `
  --moe-cpu-budget-gb 120 `
  --max-num-seqs 1 `
  --max-num-batched-tokens 2048 `
  --enforce-eager `
  --log-level INFO
```

### 3.3 输出结果

模型成功返回：

```text
等于二
```

### 3.4 关键观察

- 第一次尝试时，`max_num_batched_tokens=64` 太小，触发：
  `block_size (1056) must be <= max_num_batched_tokens (64)`。
- 第二次尝试时，`gpu_memory_utilization=0.9` 在 WDDM 桌面环境下只差极小边际，触发：
  `static budget + profiled runtime peak > free memory`。
- 把 `gpu_memory_utilization` 收到 `0.88` 后，35B 可稳定启动。
- 成功启动时的关键口径：
  - model loading took `21.06 GiB memory`
  - available KV cache memory 约 `6.5 GiB`

## 4. 122B 验证

### 4.1 模型路径

`C:\Users\13642\.cache\huggingface\hub\models--Qwen--Qwen3.5-122B-A10B-GPTQ-Int4\snapshots\5b9f0050d3ec98b0c81a7716776533c5eacebb64`

### 4.2 最终通过命令

```powershell
$env:VLLM_NO_USAGE_STATS='1'
.venv\Scripts\python.exe -m cfie.cli.main native-generate `
  --model "C:\Users\13642\.cache\huggingface\hub\models--Qwen--Qwen3.5-122B-A10B-GPTQ-Int4\snapshots\5b9f0050d3ec98b0c81a7716776533c5eacebb64" `
  --prompt "请只回答三个字：一加一。" `
  --quantization gptq_marlin `
  --dtype fp16 `
  --max-model-len 2048 `
  --max-new-tokens 16 `
  --gpu-memory-utilization 0.88 `
  --moe-cpu-budget-gb 138 `
  --max-num-seqs 1 `
  --max-num-batched-tokens 4096 `
  --enforce-eager `
  --log-level WARNING
```

### 4.3 输出结果

模型成功返回：

```text
等于二
```

### 4.4 关键观察

- 第一次尝试时，`max_num_batched_tokens=2048` 仍然过小，触发：
  `block_size (2096) must be <= max_num_batched_tokens (2048)`。
- 把 `max_num_batched_tokens` 提到 `4096` 后，122B 成功启动。
- 本次 122B 启动日志确认：
  - `Auto-enabled MoE tiered cache`
  - `gpu_slots/layer=58`
  - `prefill_burst_slots=256`
  - `cpu_slots/layer=256`
  - `cpu_static=56.44 GiB`
- 成功启动时的关键口径：
  - model loading took `26.31 GiB memory`
  - available KV cache memory 约 `1.11 GiB`

## 5. 当前结论

截至 `2026-04-12`，当前代码主线已经在 Windows 单卡环境下真实完成以下两项验证：

1. `Qwen3.5-35B-A3B-GPTQ-Int4` 可完成一次最小推理。
2. `Qwen3.5-122B-A10B-GPTQ-Int4` 可完成一次最小推理。

并且两者都走的是同一条主线特征：

- `native-generate`
- `gptq_marlin`
- `Windows 单卡`
- `enforce-eager`
- `MoE tiered cache`
- `CPU static full expert mirror`

## 6. 下一阶段工作边界

在这份记录之上，后续再继续推进：

- `TP` 兼容
- `EPLB` 兼容
- 更广泛量化路径兼容
- `desc_act=True` 兼容

后续这些工作如果破坏了本记录中的两条命令，就说明回归了当前已经成立的 Windows 单卡基线。

## 7. 2026-04-12 补充：PyCharm 下 MTP drafter 二阶段显存规划修正

### 7.1 现象

在 PyCharm 中使用 `chat --spec-method mtp --num-speculative-tokens 1` 启动
`Qwen3.5-35B-A3B-GPTQ-Int4` 时，主模型权重已经完成加载，但在继续加载
`MTP drafter` 时抛出：

```text
ValidationError: Insufficient GPU budget for MoE tiered cache
model_type=qwen3_5_mtp
raw_gpu_slots_per_layer=0
shared_gpu_reserve=23.12 GiB
gpu_expert_budget=0.00 GiB
required_expert_budget=0.05 GiB
```

### 7.2 根因

- `draft`/`MTP` 侧的 MoE planner 仍然会重新做一次 tiered cache 预算。
- 当前主线的 GPU 预算口径已经改成“基于当前 `torch.cuda.mem_get_info()` 的剩余可用显存”。
- 但 `draft` 规划又额外扣减了一次 `target_occupied_gpu_bytes`，导致主模型已经占用的显存被重复计算。
- 在单卡 Windows 场景下，这会把 `shared_gpu_reserve_bytes` 人为放大到接近主模型整体验证占用，从而把草稿模型的 `gpu_expert_budget_bytes` 压到 0。

### 7.3 修正

已修改 `CFIE/cfie/offload/policy.py`：

- 当 `draft/MTP` 规划能够直接读取当前 GPU 的真实 `free memory` 时：
  不再额外扣减 `target_occupied_gpu_bytes`。
- 只有当环境拿不到 `free memory`，不得不退回到 `total GPU memory` 估算时：
  才继续使用 `target_occupied_gpu_bytes` 作为补偿项。

### 7.4 回归保护

已补充单测 `CFIE/tests/unit/test_moe_tiered_cache.py`：

- 保留“仅知道 total GPU memory 时，需要扣减 target 占用”的旧语义。
- 新增“能够读到 free GPU memory 时，不能再重复扣减 target 占用”的回归测试。

本次最小验证结果：

- `py_compile` 通过
- 目标回归测试通过：
  - `test_build_moe_tiered_cache_plan_replans_qwen35_mtp_slots_after_target_gpu_is_reserved`
  - `test_build_moe_tiered_cache_plan_does_not_double_count_target_gpu_when_free_memory_is_available`

## 8. 2026-04-12 补充：35B MTP Windows 无 Triton 实跑打通

### 8.1 本轮修复范围

本轮是沿着 `Qwen3.5-35B-A3B-GPTQ-Int4 + --spec-method mtp` 的真实启动链路，
按报错顺序逐段补齐 Windows / 无 Triton 运行时的 fallback，核心修复点如下：

- `CFIE/cfie/model_executor/layers/fused_moe/oracle/unquantized.py`
  - 当 CUDA 运行时无 Triton 时，不再错误选择 `TRITON` backend。
  - 新增并启用 `PyTorch` backend fallback。
- `CFIE/cfie/model_executor/layers/fused_moe/unquantized_fused_moe_method.py`
  - 让 `PyTorch` backend 走 monolithic 路径。
- `CFIE/cfie/model_executor/layers/fused_moe/cpu_fused_moe.py`
  - 避免 CUDA 张量误入 CPU packed GEMM 路径。
- `CFIE/cfie/model_executor/layers/fused_moe/runner/default_moe_runner.py`
  - 补齐 monolithic backend 与 tiered-cache controller 的桥接。
  - 保证 monolithic 路径下同样会做 `prepare/chunking`。
- `CFIE/cfie/offload/weight_offload.py`
  - 放宽 tiered-cache 对非量化路径 backend 的限制，允许 `TORCH`。
- `CFIE/cfie/v1/sample/rejection_sampler.py`
  - 补齐 rejection sampler 在无 Triton 运行时的纯 `torch` fallback。
- `CFIE/cfie/v1/spec_decode/utils.py`
  - 补齐 `eagle_step_update_slot_mapping_and_metadata`
  - 补齐 `eagle_prepare_inputs_padded`
  - 补齐 `eagle_prepare_next_token_padded`
  - 补齐 `copy_and_expand_eagle_inputs`
  - 以上均提供无 Triton 的纯 `torch` fallback。
- `CFIE/cfie/v1/spec_decode/eagle.py`
  - 改为统一调用带 fallback 的 helper，而不是直接 `kernel[...]()`。
- `CFIE/cfie/model_executor/layers/fla/ops/fused_sigmoid_gating.py`
  - 补齐 Qwen3Next decode 路径 `fused_sigmoid_gating_delta_rule_update`
    的纯 `torch` fallback。

### 8.2 新增回归测试

新增文件：`CFIE/tests/unit/test_spec_decode_eagle.py`

已覆盖并通过的无 Triton 回归点：

- `eagle_prepare_next_token_padded`
- `eagle_prepare_inputs_padded`
- `eagle_step_update_slot_mapping_and_metadata`
- `copy_and_expand_eagle_inputs`
- `fused_sigmoid_gating_delta_rule_update`

本轮 focused test 结果：

```text
5 passed in 3.05s
```

### 8.3 35B MTP 最终通过命令

```powershell
$env:PYTHONPATH='C:\Users\13642\PycharmProjects\vllm\CFIE'
$env:VLLM_NO_USAGE_STATS='1'
$env:HF_HUB_DISABLE_TELEMETRY='1'
.venv\Scripts\python.exe -m cfie.cli.main native-generate `
  --model "C:\Users\13642\.cache\huggingface\hub\models--Qwen--Qwen3.5-35B-A3B-GPTQ-Int4\snapshots\33f4e5e615e1f29a7b218906555ea6fe2d09c741" `
  --prompt "你好，请回复OK。" `
  --quantization gptq_marlin `
  --max-model-len 100000 `
  --max-new-tokens 8 `
  --spec-method mtp `
  --num-speculative-tokens 1 `
  --gpu-memory-utilization 0.88 `
  --enforce-eager `
  --log-level INFO
```

### 8.4 35B MTP 实跑结果

- 进程退出码：`0`
- 终端实际返回：`OK`
- 说明当前 Windows 单卡下，`35B GPTQ + MTP + MoE tiered cache + 无 Triton`
  的推理主线已经真实打通。

### 8.5 35B MTP 关键运行观察

- `gpu_memory_utilization=0.9` 在当前桌面/WDDM 环境下仍然过于贴边；
  本轮继续以 `0.88` 作为已验证通过值。
- 日志确认：
  - `Skipping torch.distributed init_process_group for single-rank Windows local execution`
  - `Triton is unavailable on the current CUDA runtime; falling back to the PyTorch Unquantized MoE backend.`
  - `Qwen3Next fused GDN gating is falling back to the PyTorch reference path because Triton runtime is unavailable.`

## 9. 2026-04-12 补充：122B GPTQ 在当前代码下重新验证

### 9.1 首次复跑的阻塞点

使用当前统一主线直接复跑 `122B GPTQ` 时，首先命中了配置断言：

```text
In Mamba cache align mode, block_size (2096) must be <= max_num_batched_tokens (2048).
```

这说明对 `122B` 而言，当前默认的 `max_num_batched_tokens=2048`
已经不足以承载 `align` 模式下推导出的 block size。

### 9.2 最终通过命令

```powershell
$env:PYTHONPATH='C:\Users\13642\PycharmProjects\vllm\CFIE'
$env:VLLM_NO_USAGE_STATS='1'
$env:HF_HUB_DISABLE_TELEMETRY='1'
.venv\Scripts\python.exe -m cfie.cli.main native-generate `
  --model "C:\Users\13642\.cache\huggingface\hub\models--Qwen--Qwen3.5-122B-A10B-GPTQ-Int4\snapshots\5b9f0050d3ec98b0c81a7716776533c5eacebb64" `
  --prompt "你好，请回复OK。" `
  --quantization gptq_marlin `
  --max-model-len 100000 `
  --max-new-tokens 8 `
  --max-num-batched-tokens 4096 `
  --gpu-memory-utilization 0.88 `
  --enforce-eager `
  --log-level INFO
```

### 9.3 122B 实跑结果

- 进程退出码：`0`
- 本轮日志确认模型完成：
  - 权重加载
  - tiered-cache 附着
  - KV cache 初始化
  - 实际 decode 阶段执行
- 说明当前 Windows 单卡下，`122B GPTQ + MoE tiered cache + 无 Triton`
  的推理主线也已经真实跑通。

### 9.4 122B 关键运行观察

- `max_num_batched_tokens` 需要至少覆盖 `block_size=2096`；
  本轮使用 `4096` 已通过。
- 当前日志显示：
  - `Auto-enabled MoE tiered cache`
  - `gpu_slots/layer=48`
  - `prefill_burst_slots=256`
  - `cpu_slots/layer=256`
  - `cpu_static=56.44 GiB`
- 122B 在当前单卡方案下依然强依赖：
  - MoE tiered cache
  - CPU static full expert mirror
  - 更保守的 batched token 配置

## 10. 2026-04-12 补充：Windows 预编译算子首批落地与 `-j 12` 构建验证

### 10.1 本轮目标

本轮不是再新开一条 Windows 专用 Python 主线，而是在现有 `_C` 扩展中补齐一批
“Triton 不可用时仍可走 CUDA 预编译实现”的高收益算子，让 Windows 与 Linux
尽量共享同一条 Python 调度链路。

### 10.2 本轮新增的 Windows 预编译算子

已在 `_C` 中新增并接入统一 Python 调度的算子如下：

- `mrope_rotary_embedding`
- `gated_layer_norm`
- `fused_sigmoid_gating_delta_rule_update_precompiled`

对应代码落点：

- `CFIE/csrc/windows_triton_compat_ops.cpp`
- `CFIE/csrc/torch_bindings.cpp`
- `CFIE/csrc/ops.h`
- `CFIE/cfie/_custom_ops.py`
- `CFIE/cfie/model_executor/layers/rotary_embedding/mrope.py`
- `CFIE/cfie/model_executor/layers/fla/ops/layernorm_guard.py`
- `CFIE/cfie/model_executor/layers/fla/ops/fused_sigmoid_gating.py`

### 10.3 Windows 高并行重编命令

本轮在稳定构建目录 `CFIE/build-codex-win` 中，以更高并行度重编 `_C`：

```powershell
cmd /c 'call "C:\Users\13642\vs_buildtools\Common7\Tools\VsDevCmd.bat" -arch=amd64 && ^
set PATH=C:\Users\13642\vs_buildtools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64;C:\Users\13642\PycharmProjects\vllm\.venv\Scripts;%PATH% && ^
set CUDAHOSTCXX=C:\Users\13642\vs_buildtools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe && ^
set CUDA_PATH=C:\PROGRA~1\NVIDIA~2\CUDA\v13.1 && ^
cd /d C:\Users\13642\PycharmProjects\vllm\CFIE\build-codex-win && ^
cmake --build . --target _C --clean-first -j 12 && ^
cmake --install . --component _C --prefix C:/Users/13642/PycharmProjects/vllm/CFIE'
```

说明：

- 与之前相比，本轮将构建并行度提升到 `-j 12`
- 安装结果确认写回：`CFIE/cfie/_C.pyd`

### 10.4 本轮验证结果

#### 10.4.1 Python 单测

命令：

```powershell
C:\Users\13642\PycharmProjects\vllm\.venv\Scripts\python.exe -m pytest `
  C:\Users\13642\PycharmProjects\vllm\CFIE\tests\unit\test_windows_runtime_compat.py `
  C:\Users\13642\PycharmProjects\vllm\CFIE\tests\unit\test_spec_decode_eagle.py -q
```

结果：

```text
35 passed
```

#### 10.4.2 `_C` 注册检查

本轮实际检查通过：

- `torch.ops._C.gated_layer_norm`
- `torch.ops._C.mrope_rotary_embedding`
- `torch.ops._C.fused_sigmoid_gating_delta_rule_update_precompiled`

三者均已成功注册。

#### 10.4.3 GPU 对拍

本轮针对 `fused_sigmoid_gating_delta_rule_update_precompiled` 做了
CUDA 实测对拍，并与 Python 参考实现
`_fused_sigmoid_gating_delta_rule_update_ref(...)` 对比。

实测结果：

```text
out_max_abs_diff = 0.0
state_max_abs_diff = 0.0
```

说明当前 Windows 预编译路径在该实测样例上与参考实现数值一致。

### 10.5 本轮修复的关键点

- 将此前会在 Windows CUDA 路径上触发问题的 `index_select(...)`
  替换为更稳妥的 `repeat_interleave(...)`
- 保持 Python 层 fallback 入口不变，只在 `_C` 内补 Windows 可用实现
- `CFIE/csrc/ops.h` 中 `weak_ref_tensor` 已改为 `inline`，
  避免 Windows 链接期多重定义

### 10.6 当前边界

本轮已完成的是“首批高收益算子”的 Windows 预编译接入与验证，
当前仍未纳入这一批的热点包括：

- `causal_conv1d`

因此，Windows 推理链路虽然已经不再完全依赖纯 PyTorch 参考实现，
但距离把所有 Triton 热点都替换成 C++/CUDA 预编译算子，仍有后续工作。

## 11. 2026-04-12 补充：`causal_conv1d` 预编译替换与批次后推理验证

### 11.1 本轮替换范围

本轮继续遵循“Windows 尽量与 Linux 共用同一条 Python 主线”的原则，
没有额外分叉新的高层执行路径，而是在现有 `_C` 扩展中补齐：

- `causal_conv1d_fn_precompiled`
- `causal_conv1d_update_precompiled`

对应改动文件：

- `CFIE/csrc/windows_triton_compat_ops.cpp`
- `CFIE/csrc/ops.h`
- `CFIE/csrc/torch_bindings.cpp`
- `CFIE/cfie/_custom_ops.py`
- `CFIE/cfie/model_executor/layers/mamba/ops/causal_conv1d.py`
- `CFIE/tests/unit/test_windows_runtime_compat.py`

### 11.2 设计原则

本轮做法不是在 Python 层重写一套 Windows 专用逻辑，而是：

1. 保留原有 `causal_conv1d_fn` / `causal_conv1d_update` 入口
2. 在 `HAS_TRITON=False` 时，优先尝试 `_C` 预编译实现
3. 若 `_C` 不可用，再退回原有 PyTorch reference path

也就是说：

- Linux / 有 Triton：仍走原 Triton 主线
- Windows / 无 Triton：优先走 `_C` 预编译 CUDA 实现
- 再兜底：PyTorch reference

### 11.3 本轮构建

延续稳定构建目录：

- `C:\Users\13642\PycharmProjects\vllm\CFIE\build-codex-win`

本轮构建仍使用较高并行度：

```powershell
cmake --build . --target _C --clean-first -j 12
cmake --install . --component _C --prefix C:/Users/13642/PycharmProjects/vllm/CFIE
```

本轮新增构建日志：

- `C:\Users\13642\PycharmProjects\vllm\CFIE\build-codex-win\rebuild6.log`
- `C:\Users\13642\PycharmProjects\vllm\CFIE\build-codex-win\install_c.log`

### 11.4 本轮针对性回归

#### 11.4.1 Python 单测

执行：

```powershell
C:\Users\13642\PycharmProjects\vllm\.venv\Scripts\python.exe -m pytest `
  C:\Users\13642\PycharmProjects\vllm\CFIE\tests\unit\test_windows_runtime_compat.py `
  C:\Users\13642\PycharmProjects\vllm\CFIE\tests\unit\test_spec_decode_eagle.py -q
```

结果：

```text
37 passed
```

#### 11.4.2 CUDA 对拍

本轮针对 `causal_conv1d` 做了三组 CUDA 对拍：

1. `causal_conv1d_fn_precompiled` 对比 `_causal_conv1d_fn_ref`
2. `causal_conv1d_update_precompiled` fast-path 对比 `_causal_conv1d_update_ref`
3. `causal_conv1d_update_precompiled` varlen/spec-path 对比 `_causal_conv1d_update_ref`

实测结果：

```text
prefill_max_diff = 0.0
update_fast_max_diff = 0.0
update_var_max_diff = 0.0
```

说明本轮 `causal_conv1d` 的 Windows 预编译路径在上述样例上与参考实现数值一致。

### 11.5 按批次门禁执行的真实推理验证

按照“每替换一批，就跑一次推理测试”的约束，本轮在 `causal_conv1d`
替换完成后，立即执行了一次真实 `35B + MTP` 推理验证。

执行命令：

```powershell
$env:PYTHONPATH='C:\Users\13642\PycharmProjects\vllm\CFIE'
$env:VLLM_NO_USAGE_STATS='1'
$env:HF_HUB_DISABLE_TELEMETRY='1'
C:\Users\13642\PycharmProjects\vllm\.venv\Scripts\python.exe -m cfie.cli.main native-generate `
  --model "C:\Users\13642\.cache\huggingface\hub\models--Qwen--Qwen3.5-35B-A3B-GPTQ-Int4\snapshots\33f4e5e615e1f29a7b218906555ea6fe2d09c741" `
  --prompt "请只回复OK。" `
  --quantization gptq_marlin `
  --max-model-len 100000 `
  --max-new-tokens 8 `
  --spec-method mtp `
  --num-speculative-tokens 1 `
  --gpu-memory-utilization 0.88 `
  --enforce-eager `
  --log-level INFO
```

结果：

```text
OK
```

并且本轮真实日志中没有再出现：

- `Mamba causal_conv1d prefill is falling back ...`
- `Mamba causal_conv1d decode is falling back ...`

这说明在当前 Windows 无 Triton 运行时下，`causal_conv1d` 已经优先接入 `_C`
预编译实现，而不是继续停留在 Python reference path。

### 11.6 当前仍可见的其他 fallback

需要注意的是，本轮日志中仍能看到其他无 Triton fallback，例如：

- `cfie_flash_attn rotary ... falling back to the native rotary embedding path`
- `FLA chunk gated delta rule ... falling back to the PyTorch recurrent reference path`
- `KV block zeroing ... falling back to the PyTorch reference path`
- `Qwen3Next fused GDN gating ... falling back to the PyTorch reference path`

因此本轮结论是：

- `causal_conv1d` 这一批已经切走
- 但 Windows 推理链路中仍有其他 Triton 热点尚未全部替换完成
## 20. 2026-04-12 补充：`rotary / FLA chunk / KV zero / Qwen3Next GDN` 这一批 `_C` 替换

### 20.1 本轮新增的 Windows 预编译算子

本轮继续沿用“同一条 Python 主线，Windows 仅在 Triton 不可用时优先切到 `_C` 预编译实现”的策略，
没有再单独分叉新的 Windows 高层执行链路。

本轮新增并接入的 `_C` 算子如下：

- `apply_rotary_emb_precompiled`
- `chunk_gated_delta_rule_precompiled`
- `fused_gdn_gating_precompiled`
- `zero_kv_blocks_precompiled`

对应改动文件：

- `C:\Users\13642\PycharmProjects\vllm\CFIE\csrc\ops.h`
- `C:\Users\13642\PycharmProjects\vllm\CFIE\csrc\torch_bindings.cpp`
- `C:\Users\13642\PycharmProjects\vllm\CFIE\csrc\windows_triton_compat_ops.cpp`
- `C:\Users\13642\PycharmProjects\vllm\CFIE\cfie\_custom_ops.py`
- `C:\Users\13642\PycharmProjects\vllm\CFIE\cfie\model_executor\layers\rotary_embedding\common.py`
- `C:\Users\13642\PycharmProjects\vllm\CFIE\cfie\model_executor\layers\fla\ops\chunk.py`
- `C:\Users\13642\PycharmProjects\vllm\CFIE\cfie\model_executor\models\qwen3_next.py`
- `C:\Users\13642\PycharmProjects\vllm\CFIE\cfie\v1\worker\utils.py`
- `C:\Users\13642\PycharmProjects\vllm\CFIE\tests\unit\test_windows_runtime_compat.py`

### 20.2 本轮验证顺序

#### 20.2.1 Python selector 回归

执行：

```powershell
C:\Users\13642\PycharmProjects\vllm\.venv\Scripts\python.exe -m pytest `
  C:\Users\13642\PycharmProjects\vllm\CFIE\tests\unit\test_windows_runtime_compat.py -q
```

结果：

```text
35 passed
```

#### 20.2.2 `spec_decode_eagle` 定向回归

执行：

```powershell
C:\Users\13642\PycharmProjects\vllm\.venv\Scripts\python.exe -m pytest `
  C:\Users\13642\PycharmProjects\vllm\CFIE\tests\unit\test_spec_decode_eagle.py -q
```

结果：

```text
6 passed
```

#### 20.2.3 `_C` 重编与安装

构建目录仍然使用：

- `C:\Users\13642\PycharmProjects\vllm\CFIE\build-codex-win`

执行：

```powershell
cmake --build . --target _C --clean-first -j 12
cmake --install . --component _C --prefix C:/Users/13642/PycharmProjects/vllm/CFIE
```

本轮日志：

- `C:\Users\13642\PycharmProjects\vllm\CFIE\build-codex-win\rebuild_windows_ops.log`
- `C:\Users\13642\PycharmProjects\vllm\CFIE\build-codex-win\install_windows_ops.log`

安装结果确认 `_C.pyd` 已重新落到：

- `C:\Users\13642\PycharmProjects\vllm\CFIE\cfie\_C.pyd`

### 20.3 小尺寸 CUDA 对拍

本轮对新增 4 条链都做了小尺寸 GPU 对拍：

1. `ApplyRotaryEmb.forward_cuda` 对比 `forward_native`
2. `chunk_gated_delta_rule` 对比 `_chunk_gated_delta_rule_ref`
3. `fused_gdn_gating` 对比当前 PyTorch reference 公式
4. `KVBlockZeroer.zero_block_ids` 对比期望的 block 置零结果

实际结果：

```text
[PASS] rotary - max_diff=0.000000
[PASS] chunk_output - max_diff=0.000000
[PASS] chunk_final - max_diff=0.000000
[PASS] gdn_g - max_diff=0.000000
[PASS] gdn_beta - max_diff=0.000000
[PASS] zero_kv_kv1
[PASS] zero_kv_kv2
GPU smoke OK
```

这说明本轮新补的 `_C` 路径在上述覆盖范围内与 reference 数值一致。

### 20.4 按批次要求执行的真实推理验证

按“每替换一批，就跑一次真实推理”的约束，本轮完成代码接入、单测、重编、GPU 对拍后，
立即执行了 1 次真实 `35B + MTP` 推理。

执行命令：

```powershell
$env:PYTHONPATH='C:\Users\13642\PycharmProjects\vllm\CFIE'
$env:VLLM_NO_USAGE_STATS='1'
$env:HF_HUB_DISABLE_TELEMETRY='1'
C:\Users\13642\PycharmProjects\vllm\.venv\Scripts\python.exe -m cfie.cli.main native-generate `
  --model "C:\Users\13642\.cache\huggingface\hub\models--Qwen--Qwen3.5-35B-A3B-GPTQ-Int4\snapshots\33f4e5e615e1f29a7b218906555ea6fe2d09c741" `
  --prompt "请只回复OK。" `
  --quantization gptq_marlin `
  --max-model-len 100000 `
  --max-new-tokens 8 `
  --spec-method mtp `
  --num-speculative-tokens 1 `
  --gpu-memory-utilization 0.88 `
  --enforce-eager `
  --log-level INFO
```

结果：

```text
OK
```

进程退出码：

```text
0
```

### 20.5 本轮最重要的结论

这次真实日志里，以下 4 条 fallback warning 已经不再出现：

- `cfie_flash_attn rotary is unavailable ...`
- `FLA chunk gated delta rule is falling back ...`
- `KV block zeroing is falling back ...`
- `Qwen3Next fused GDN gating is falling back ...`

因此可以确认：

1. 这 4 条 Windows 无 Triton 热点路径已经切到 `_C` 预编译实现
2. 切换后没有破坏真实 `35B + MTP` 推理
3. 当前这一批替换已经完成“实现 -> 编译 -> 对拍 -> 真实推理”的闭环

### 20.6 当前仍然可见的其它非 Triton 路径现象

本轮真实推理日志里仍可看到的、但不属于本批新替换失败的现象包括：

- Triton runtime 不可用，因此 `inductor` 退回 `eager`
- `torch.compile` / `CUDAGraphs` 在当前配置下未启用
- `Triton allocator API is unavailable ...`
- `Unquantized MoE` 在当前 CUDA runtime 上继续走 PyTorch backend

也就是说，本轮已经解决的是“本批 4 条高频 fallback 路径”，
但 Windows 侧整体吞吐仍然会继续受到“无 Triton runtime / 无 Triton allocator / MoE backend 仍偏保守”等因素影响。

## 21. 2026-04-13 设计更新：CPU static mirror 应保存 runtime-ready 专家权重

### 21.1 新的设计约束

对 `122B` 这类频繁发生 expert load/unload 的大模型，`CPU static mirror` 不应继续保存 checkpoint 原始格式，
而应保存**已经完成推理预处理**的 runtime-ready 权重。

这里的“预处理”包括：

- 非量化路径：
  - `gate_proj + up_proj -> w13`
  - `down_proj -> w2`
- `gptq_marlin` 路径：
  - `gate/up/down -> raw w13/w2`
  - `w13/w2 qweight repack`
  - `w13/w2 scales permute`
  - 必要时连同 `g_idx / g_idx_sort_indices` 一并整理成 runtime 格式

结论：

- `w13/w2` 合并不应在每次 cache miss 时重复做。
- `repack / permute` 也不应在每次 cache miss 时重复做。
- 这些步骤原则上都应在 CPU static mirror 物化时完成一次。

### 21.2 当前代码已开始按这个方向收敛

本轮已经在 `CFIE/cfie/offload/weight_offload.py` 中做了第一步收口：

- `CPU static bundle` 在物化后，会立即进入 `_preprocess_cpu_static_bundle(...)`
- `gptq_marlin` 路径会生成 runtime-ready CPU bundle
- 非量化路径也会把 checkpoint 形态转成运行时可直接消费的 `runtime.w13_weight / runtime.w2_weight`
- `cache miss` 时，如果命中的 bundle 已经是 `runtime_ready=True`，运行时将不再重复做：
  - `w13/w2` 合并
  - `gptq_marlin_moe_repack(...)`
  - `marlin_moe_permute_scales(...)`

也就是说，当前主线已经从：

`CPU checkpoint bundle -> 每次 miss 重做预处理 -> GPU slot`

开始收敛为：

`CPU runtime-ready bundle -> miss 时直接写入 GPU slot`

### 21.3 pinned memory 的原则

新的目标不是让 checkpoint 原始 bundle 全量 pinned，而是：

- **长期驻留的 CPU static mirror** 尽量保存为 runtime-ready，并优先尝试 pinned memory
- 如果平台/内存压力不允许 pin，则允许回退到 pageable CPU memory，不能因为 pin 失败破坏 correctness
- checkpoint 形态的 `source bundle` 只作为短暂中转容器，不再视为长期缓存形态

### 21.4 下一步必须完成的事

当前虽然已经建立了 `runtime-ready CPU static bundle` 语义，但批量加载还只是 Python 入口语义，
真正的“8 个专家一次上载到显存”还没有彻底下沉成 `_C` / C++/CUDA 算子。

下一阶段明确目标：

1. 新增 batch load 入口：一次接收 `N<=8` 个 experts 与目标 slots
2. CPU 侧输入为 runtime-ready + pinned 的 batched tensors
3. 用 `_C` 算子一次调用完成 `N` 个 experts 对 resident GPU slots 的覆盖
4. `122B` 主线优先验证该路径

### 21.5 当前不要再回退到旧思路

后续代码如果继续在 miss 路径里反复做：

- `gate/up/down -> w13/w2`
- `gptq_marlin_moe_repack(...)`
- `scale permute`

则说明又退回了低效设计，应视为回归。

## 22. 2026-04-13 更新：runtime-ready batch load `_C` op 已接入并完成 35B / 122B 回归

### 22.1 本轮新增代码落点

- `CFIE/csrc/ops.h`
  - 新增
    - `moe_batch_load_unquantized_runtime_precompiled(...)`
    - `moe_batch_load_gptq_runtime_precompiled(...)`
- `CFIE/csrc/torch_bindings.cpp`
  - 新增上述两个 `_C` op 的 schema 与 `torch::kCUDA` 注册
- `CFIE/csrc/windows_triton_compat_ops.cpp`
  - 新增 runtime-ready expert 的批量上载实现
  - 第一版实现策略是：
    - 将 batched CPU tensor 一次搬到目标 GPU
    - 再用 `index_copy_` 一次覆盖多个 resident slots
- `CFIE/cfie/_custom_ops.py`
  - 新增两个 wrapper 与 `has_precompiled_*` 检测接口
- `CFIE/cfie/offload/weight_offload.py`
  - `_write_expert_bundles(...)` 现在优先走 runtime-ready 批量加载路径
  - 仅当 op 不可用或 bundle 不是 runtime-ready 时，才回退到旧逐 expert 路径
- `CFIE/tests/unit/test_moe_tiered_cache.py`
  - 新增 unquantized / GPTQ 两条批量加载入口测试

### 22.2 本轮验证结果

- `_C` op 导出检查通过：
  - `has_precompiled_moe_batch_load_unquantized_runtime() == True`
  - `has_precompiled_moe_batch_load_gptq_runtime() == True`
- 真 GPU 冒烟测试通过：
  - 小尺寸 CPU batched tensor -> GPU resident slots 的 unquantized / GPTQ 两条路径均返回 `ok`
- 单测通过：
  - `CFIE/tests/unit/test_moe_tiered_cache.py -k "runtime_ready or load_experts_into_slots or source_bundle or batch_load"`
- 真实推理回归通过：
  - `35B GPTQ + MTP`：返回 `OK`
    - 日志：`CFIE/docs/推理验证日志_2026-04-13_35B_MTP_batch_runtime_load_op_OK.txt`
  - `122B GPTQ + MTP`：返回 `OK`
    - 日志：`CFIE/docs/推理验证日志_2026-04-13_122B_MTP_batch_runtime_load_op_OK.txt`

### 22.3 122B 本轮额外约束

- `122B` 首次回归时触发：
  - `AssertionError: In Mamba cache align mode, block_size (2096) must be <= max_num_batched_tokens (2048)`
- 处理方式：
  - 将启动参数补为 `--max-num-batched-tokens 4096`
- 结论：
  - 这不是 batch load op 引入的新错误，而是 `122B + Mamba align mode` 下已有的启动参数约束

### 22.4 当前阶段结论

- 现在 CPU static mirror 已经不仅是 runtime-ready 语义，**批量 expert 上载也已经真正下沉到 `_C`**
- Python 层没有新增 Windows 专属主线；Linux / Windows 仍共享同一条高层批量加载入口
- 当前第一版 `_C` op 仍以 ATen 搬运 + `index_copy_` 为主，后续若继续压缩延迟，可再下沉为更专用的 CUDA kernel

## 2026-04-13 批次 2 进展：Sampling + Spec Decode rejection 预编译替换

### 本批目标

本批继续推进 `03_Windows_Triton算子全量盘点与C++替换总计划.md` 中的 `Sampling + Spec Decode` 方向。目标不是给 Windows 新开一条高层业务通道，而是在现有 Python 主线上继续保持统一 selector：

1. Triton 可用：优先走原 Triton kernel。
2. Triton 不可用且 `_C` 可用：优先走 Windows/Linux 都能调用的预编译 `_C` op。
3. `_C` 不可用：最后回退到现有 PyTorch reference / torch fallback。

### 已新增/接入的 `_C` op

本批覆盖 `cfie/v1/sample/rejection_sampler.py` 中的以下 Triton 依赖或 Triton fallback 热点：

- `expand_batch_to_tokens_precompiled`：替代 `expand_kernel` 的非 Triton预编译入口，当前为 `_C + ATen repeat_interleave/where`。
- `sample_recovered_tokens_precompiled`：替代 `sample_recovered_tokens_kernel` 的非 Triton入口，当前为 `_C + ATen` 向量化实现，消除 Python 双层 token/request 循环。
- `rejection_greedy_sample_precompiled`：替代 `rejection_greedy_sample_kernel` 的非 Triton入口，当前为轻量 CUDA kernel。
- `rejection_random_sample_precompiled`：替代 `rejection_random_sample_kernel` 的非 Triton入口，当前为轻量 CUDA kernel。

### 代码落点

- C++ schema：`C:\Users\13642\PycharmProjects\vllm\CFIE\csrc\ops.h`
- `_C` 绑定：`C:\Users\13642\PycharmProjects\vllm\CFIE\csrc\torch_bindings.cpp`
- `_C + ATen` 实现：`C:\Users\13642\PycharmProjects\vllm\CFIE\csrc\windows_triton_compat_ops.cpp`
- CUDA kernel 实现：`C:\Users\13642\PycharmProjects\vllm\CFIE\csrc\sampler.cu`
- Python 包装层：`C:\Users\13642\PycharmProjects\vllm\CFIE\cfie\_custom_ops.py`
- Python selector：`C:\Users\13642\PycharmProjects\vllm\CFIE\cfie\v1\sample\rejection_sampler.py`
- 单测：`C:\Users\13642\PycharmProjects\vllm\CFIE\tests\unit\test_windows_spec_decode_precompiled_ops.py`

### 编译与验证

- 增量编译：`C:\Users\13642\PycharmProjects\vllm\.venv\Scripts\ninja.exe -C C:\Users\13642\PycharmProjects\vllm\CFIE\build-codex-win -j12 _C`，结果通过。
- 已将 `C:\Users\13642\PycharmProjects\vllm\CFIE\build-codex-win\_C.pyd` 覆盖到 `C:\Users\13642\PycharmProjects\vllm\CFIE\cfie\_C.pyd`。
- `_C` 导出检查：四个 `has_precompiled_*` 均为 `True`。
- 单测：`python -m pytest C:\Users\13642\PycharmProjects\vllm\CFIE\tests\unit\test_windows_spec_decode_precompiled_ops.py -q -s`，结果 `5 passed`。
- 单测日志：`C:\Users\13642\PycharmProjects\vllm\CFIE\docs\算子验证日志_2026-04-13_spec_decode_precompiled_ops_pytest.txt`
- 真实推理：`35B GPTQ-Int4 + MTP + random sampling` 已通过 `native-generate` 一次回归。
- 推理日志：`C:\Users\13642\PycharmProjects\vllm\CFIE\docs\推理验证日志_2026-04-13_35B_MTP_spec_decode_precompiled_ops.txt`

### 当前限制

- 本批没有替换完整 `topk_topp_triton.py` 的 Qrita 风格 top-k/top-p Triton kernel；当 Windows `HAS_TRITON=False` 且启用 top-p/top-k 时，`apply_top_k_top_p` 仍可能走 PyTorch sort 路径。
- `sample_recovered_tokens_precompiled` 当前是 `_C + ATen` 向量化第一版，会额外构造 `expanded_inv_q` 与中间 `scores`；它已经消除了 Python token/request 循环，但后续仍可继续下沉为更省显存的 fused CUDA kernel。
- 本批真实推理回归使用 35B GPTQ + MTP；122B 已在前一批 batch runtime load op 中验证通过，本批未重复启动 122B，避免无必要地长时间占用显存。

### 下一步建议

下一批优先继续处理 `Sampling + Spec Decode` 中仍未收口的 `topk_topp_triton.py`：

1. 先做 `top-k only` 的 `_C` 快速路径，优先复用现有 `top_k_per_row_*` / top-k CUDA 资产。
2. 再评估 `top-p only` 与 `top-k + top-p` 是否先以 `_C + ATen` 中间态替换 PyTorch sort，还是直接实现 CUDA pivot/select kernel。
3. 每完成一批仍按“编译 `_C` -> focused 单测 -> 真实推理 -> docs 记录”的顺序推进。

## 2026-04-13 验证补充：Top-K / Top-P 预编译路径

### 验证对象

- 代码入口：`apply_top_k_top_p_precompiled`
- selector：`C:\Users\13642\PycharmProjects\vllm\CFIE\cfie\v1\sample\ops\topk_topp_sampler.py`
- 模型：`Qwen3.5-35B-A3B-GPTQ-Int4`
- 运行模式：`native-generate + MTP + top-k + top-p`

### 定向单测

- 命令：`python -m pytest C:\Users\13642\PycharmProjects\vllm\CFIE\tests\unit\test_windows_topk_topp_precompiled_ops.py -q -s`
- 结果：`3 passed`
- 覆盖点：
  - `top-k only`
  - `top-p only`
  - `top-k + top-p`
- 日志：`C:\Users\13642\PycharmProjects\vllm\CFIE\docs\算子验证日志_2026-04-13_topk_topp_precompiled_ops_pytest.txt`

### 真实推理

- 命令类型：`native-generate`
- 参数特征：
  - `--quantization gptq_marlin`
  - `--spec-method mtp`
  - `--top-k 20`
  - `--top-p 0.8`
  - `--enforce-eager`
- 结果：
  - 推理链路完成启动与生成
  - 运行时统计到 `WINDOWS_TOPK_TOPP_PRECOMPILED_OP_COUNT=35`
  - 说明 `top-k/top-p` 已不再停留在 Windows 下的 Python fallback
- 日志：`C:\Users\13642\PycharmProjects\vllm\CFIE\docs\推理验证日志_2026-04-13_35B_topk_topp_precompiled_ops_clean.txt`

### 本次验证结论

- Windows `HAS_TRITON=False` 情况下，`top-k/top-p` 已具备 `_C` 预编译 fallback。
- 当前版本功能正确，但仍属于 `_C + ATen` 第一版。
- 后续仍需继续追求更快的专用 CUDA kernel 版本，尤其是 `top-k only` 快速路径。

## 2026-04-13 验证补充：Top-K only / Top-K + Top-P 快路径

### 定向单测

- 命令：`python -m pytest C:\Users\13642\PycharmProjects\vllm\CFIE\tests\unit\test_windows_topk_topp_precompiled_ops.py -q -s`
- 结果：`5 passed`
- 新覆盖点：
  - `top-k only`
  - `top-k only` 含部分 `k == vocab_size`
  - `top-k + top-p`
  - `top-k + top-p` 含部分 `k == vocab_size`
- 日志：`C:\Users\13642\PycharmProjects\vllm\CFIE\docs\算子验证日志_2026-04-13_topk_topp_precompiled_ops_pytest_v3.txt`

### `35B GPTQ + MTP + top-k only`

- 结果：通过
- 运行时统计：`WINDOWS_TOPK_ONLY_PRECOMPILED_OP_COUNT=33`
- 日志：`C:\Users\13642\PycharmProjects\vllm\CFIE\docs\推理验证日志_2026-04-13_35B_topk_only_fastpath.txt`

### `35B GPTQ + MTP + top-k + top-p`

- 结果：通过
- 运行时统计：`WINDOWS_TOPK_TOPP_PRECOMPILED_OP_COUNT=33`
- 日志：`C:\Users\13642\PycharmProjects\vllm\CFIE\docs\推理验证日志_2026-04-13_35B_topk_topp_fastpath_v2.txt`

### 本次验证结论

- Windows 采样链路中，`top-k only` 已不再依赖全量 `sort`。
- 当 `max(k) < vocab_size` 时，`top-k + top-p` 也已不再依赖全量 `sort`。
- 当前剩余的主要采样热点已经收敛到 `top-p only` 与更深层的专用 CUDA kernel 替换。

## 2026-04-13 验证补充：`top-p only` exact 快路径

### 定向单测

- 命令：`python -m pytest C:\Users\13642\PycharmProjects\vllm\CFIE\tests\unit\test_windows_topk_topp_precompiled_ops.py -q -s`
- 结果：`7 passed`
- 新增覆盖点：
  - `top-p only`
  - `top-p only` 含部分 `p == 1.0`
  - `top-k + top-p` 含部分 `k == vocab_size` 且 `p == 1.0`
- 日志：`C:\Users\13642\PycharmProjects\vllm\CFIE\docs\算子验证日志_2026-04-13_topk_topp_precompiled_ops_pytest_v4.txt`

### `35B GPTQ + MTP + top-p only`

- 命令类型：`native-generate`
- 关键参数：
  - `--quantization gptq_marlin`
  - `--spec-method mtp`
  - `--temperature 0.8`
  - `--top-p 0.8`
  - `--top-k 0`
  - `--enforce-eager`
- 结果：通过
- 日志：`C:\Users\13642\PycharmProjects\vllm\CFIE\docs\推理验证日志_2026-04-13_35B_topp_only_fastpath.txt`

### `122B GPTQ + MTP + top-p only`

- 命令类型：`native-generate`
- 关键参数：
  - `--quantization gptq_marlin`
  - `--spec-method mtp`
  - `--temperature 0.8`
  - `--top-p 0.8`
  - `--top-k 0`
  - `--max-num-batched-tokens 4096`
  - `--enforce-eager`
- 结果：通过
- 日志：`C:\Users\13642\PycharmProjects\vllm\CFIE\docs\推理验证日志_2026-04-13_122B_topp_only_fastpath.txt`

### 122B 启动参数约束补记

- `122B` 在当前 `Mamba cache align mode` 下，`block_size=2096`，因此不能继续使用默认 `max_num_batched_tokens=2048`。
- 之前 `122B top-k + top-p` 的首次回归失败，原因正是这个约束：
  - `AssertionError: In Mamba cache align mode, block_size (2096) must be <= max_num_batched_tokens (2048).`
- 调整为 `--max-num-batched-tokens 4096` 后：
  - `122B top-k + top-p` 回归通过
  - 本轮 `122B top-p only` 回归也通过

### 本次验证结论

- Windows 采样链路中的三条主要路径：
  - `top-k only`
  - `top-k + top-p`
  - `top-p only`

  现在都已经具备 `_C` 预编译快路径。
- 当前仍未完成的是“把这些 `_C + ATen` 快路径继续下沉成专用 CUDA kernel”。

## 2026-04-13 验证补充：`top-k only` CUDA 选择核复用

### 本轮改动点

- `top-k only` 内部实现不再只依赖 `_C + ATen topk`。
- 现在会优先复用 `CFIE/csrc/sampler.cu` 中现有的 `top_k_per_row_decode` CUDA 选择核，先在 CUDA 上取出每行 `max_k` 候选，再对这个小集合做阈值排序。

### 定向单测

- 命令：`python -m pytest C:\Users\13642\PycharmProjects\vllm\CFIE\tests\unit\test_windows_topk_topp_precompiled_ops.py -q -s`
- 结果：`7 passed`
- 日志：`C:\Users\13642\PycharmProjects\vllm\CFIE\docs\算子验证日志_2026-04-13_topk_topp_precompiled_ops_pytest_v5.txt`

### `spec decode` 回归

- 命令：`python -m pytest C:\Users\13642\PycharmProjects\vllm\CFIE\tests\unit\test_windows_spec_decode_precompiled_ops.py -q -s`
- 结果：`5 passed`
- 日志：`C:\Users\13642\PycharmProjects\vllm\CFIE\docs\算子验证日志_2026-04-13_spec_decode_precompiled_ops_pytest_v4.txt`

### `35B GPTQ + MTP + top-k only`

- 命令类型：`native-generate`
- 关键参数：
  - `--quantization gptq_marlin`
  - `--spec-method mtp`
  - `--temperature 0.8`
  - `--top-p 1.0`
  - `--top-k 20`
  - `--enforce-eager`
- 结果：通过
- 日志：`C:\Users\13642\PycharmProjects\vllm\CFIE\docs\推理验证日志_2026-04-13_35B_topk_only_cuda_reuse_fastpath.txt`

### 本次验证结论

- `top-k only` 已经开始真正复用现有 `sampler.cu` 里的 CUDA 选择核。
- 当前还不是完全专用 CUDA 版本，但已经比“纯 `_C + ATen topk`”更接近最终收口形态。

## 2026-04-13 验证补充：Attention cache `_C_cache_ops` 收口回归

### 本轮改动点

- `cfie/v1/attention/ops/triton_reshape_and_cache_flash.py`
  - 4D 普通 cache path：优先转 `_custom_ops.reshape_and_cache_flash`
  - 4D diffkv path：优先转 `_custom_ops.reshape_and_cache_flash_diffkv`
- `_C_cache_ops` 新增：
  - `reshape_and_cache_flash_diffkv`
- `cache_kernels.cu` 额外修复：
  - 普通 `reshape_and_cache_flash` 在 HND 小 head 布局下的卡死问题

### Attention cache 定向单测

- 命令：`python -m pytest C:\Users\13642\PycharmProjects\vllm\CFIE\tests\unit\test_windows_attention_cache_precompiled_ops.py -q -s`
- 结果：`4 passed`
- 日志：`C:\Users\13642\PycharmProjects\vllm\CFIE\docs\算子验证日志_2026-04-13_attention_cache_precompiled_ops_pytest.txt`

### `spec decode` 回归

- 命令：`python -m pytest C:\Users\13642\PycharmProjects\vllm\CFIE\tests\unit\test_windows_spec_decode_precompiled_ops.py -q -s`
- 结果：`5 passed`
- 日志：`C:\Users\13642\PycharmProjects\vllm\CFIE\docs\算子验证日志_2026-04-13_spec_decode_precompiled_ops_pytest_v6.txt`

### `35B GPTQ + MTP` 主线回归

- 命令类型：`native-generate`
- 关键参数：
  - `--quantization gptq_marlin`
  - `--spec-method mtp`
  - `--top-p 0.8`
  - `--top-k 20`
  - `--enforce-eager`
- 结果：通过，完成文本生成
- 日志：`C:\Users\13642\PycharmProjects\vllm\CFIE\docs\推理验证日志_2026-04-13_35B_attention_cache_precompiled_regression.txt`

### 本次验证结论

- Attention cache update 已经开始从 Triton 收口到 `_C_cache_ops`。
- 这批没有改变 35B Windows 单卡主线的高层调用方式，但把 cache update 的低层后端进一步统一到了预编译算子。
- 下一批建议直接转向 `unified_attention` / `TreeAttention` 主核。
