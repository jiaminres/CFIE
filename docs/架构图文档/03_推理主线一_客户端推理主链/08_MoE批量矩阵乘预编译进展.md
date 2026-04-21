# MoE批量矩阵乘预编译进展

## 1. 本轮目标

- 给 `cfie/model_executor/layers/fused_moe/fused_batched_moe.py` 的无 Triton 路径补上真正的 `_C` / C++ / CUDA 闭环。
- 保持 Windows 与 Linux 共用同一层 Python 调度入口。
- 不为 Windows 单独增加高层业务分支。

## 2. 本轮代码改动

### 2.1 Python 调度层

- `cfie/model_executor/layers/fused_moe/fused_batched_moe.py`
  - 新增 `_try_precompiled_moe_batched_kernel(...)`
  - `invoke_moe_batched_triton_kernel(...)` 在 `HAS_TRITON=False` 时改为：
    1. 先尝试 `_C.moe_batched_mm_precompiled`
    2. 若 `_C` 不可用，再落回共享 torch/reference

- `cfie/_custom_ops.py`
  - 新增 `has_precompiled_moe_batched_mm()`
  - 新增 `moe_batched_mm_precompiled(...)`

### 2.2 C++ / CUDA 注册层

- `csrc/ops.h`
  - 声明 `moe_batched_mm_precompiled(...)`
- `csrc/torch_bindings.cpp`
  - 注册 `torch.ops._C.moe_batched_mm_precompiled`

### 2.3 Windows 预编译实现

- `csrc/windows_triton_compat_ops.cpp`
  - 新增 `moe_batched_mm_precompiled(...)`
  - 支持非量化 `fp16/bf16/fp32` 与 `fp8 w8a8`
  - 量化路径先按 reference 语义反量化，再执行 `matmul`

## 3. 验证结论

- focused pytest：`5 passed`
- 直接 CUDA 调用：`has_op True`、`max_diff_fp16 0.0`、`max_diff_fp8 0.0`
- full runtime compat：`169 passed`
- 35B 冒烟：在 `gpu_memory_utilization=0.88` 的 `native-generate` 条件下已成功启动并退出

## 4. 当前状态判断

- `moe_batched_mm_precompiled` 这一项可视为 **已完成**。
- `fused_batched_moe.py` 整体仍处于 **进行中**，因为更大范围的 Triton -> C++ / CUDA 替换还在继续。

## 5. 对长期计划的意义

本轮最重要的不是“补了一个 fallback”，而是把 batched MoE MM 真正收口成了 `_C` 闭环：

- Windows 无 Triton 运行时，不再只能依赖 Python reference。
- Linux / Windows 继续共用同一条 Python selector。
- 后续继续压缩 `fused_batched_moe` / `fused_moe` 热链中的 Python fallback 占比时，有了稳定落点。

## 6. 相关文档

- `./03_Windows算子替换总计划.md`
- `./04_Windows算子替换执行台账.md`
- `./06_Windows运行时回退追踪.md`