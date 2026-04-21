# FusedRecurrent共享路径进展

## 1. 本轮目标

- 补齐 `cfie/model_executor/layers/fla/ops/fused_recurrent.py` 中 `fused_recurrent_gated_delta_rule_fwd(...)` 的 Windows 无 Triton 主链。
- 保持 Windows 与 Linux 共享同一套 Python 入口。
- 优先复用已有 `_C` 能力，不新增高层 Windows 特殊业务分支。

## 2. 本轮代码改动

### 2.1 核心文件

- `cfie/model_executor/layers/fla/ops/fused_recurrent.py`
  - 新增 `_resolve_state_index(...)`
  - 新增 `_try_precompiled_fused_recurrent_gated_delta_rule_fwd(...)`
  - 新增 `_fused_recurrent_gated_delta_rule_fwd_ref(...)`
  - 调整 `fused_recurrent_gated_delta_rule_fwd(...)` 的执行顺序：
    - 无 Triton 时先尝试 `_C.chunk_gated_delta_rule_precompiled`
    - 当前场景不适合 `_C` 复用时回落到共享 torch/reference
    - 只在 `HAS_TRITON=True` 时再访问 Triton 专有 helper 与 kernel

### 2.2 当前收口策略

- 基础序列场景：优先复用 `_C.chunk_gated_delta_rule_precompiled`
- 连续批 / 无效 slot / spec decode 场景：回退到共享 torch/reference
- `packed_decode` 路径：继续优先复用 `_C.fused_recurrent_gated_delta_rule_packed_decode_precompiled`

## 3. 验证结论

- focused pytest：`6 passed`
- full runtime compat：`167 passed`
- 35B 冒烟：`EXIT=0`
- 阶段观察：35B smoke 中未再重复出现 FLA packed recurrent / fused recurrent 的回退告警

## 4. 当前状态判断

- `fused_recurrent.py` 这一模块仍标记为 **进行中**。
- 原因不是主路径不可用，而是连续批 / spec decode 仍未形成专用 `_C` / CUDA 闭环。

## 5. 对长期计划的意义

这一步的重要性在于：

- 把“Windows 无 Triton 只能纯 Python 回退”的局面，推进到“普通序列场景可优先命中共享 `_C`”的阶段。
- 保住 Linux / Windows 共用同一层 Python selector，而不是分裂成两套高层主链。
- 为后续继续深挖 spec decode 与连续批场景留下稳定接口。

## 6. 相关文档

- `./03_Windows算子替换总计划.md`
- `./04_Windows算子替换执行台账.md`
- `./06_Windows运行时回退追踪.md`