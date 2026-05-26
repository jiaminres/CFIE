# 2026-05-26 GUI Agent 多题豆包验证

## 目标

验证 GUI Agent 在单个 APP 会话中连续处理多条题目时，是否能稳定完成：

- 读取题目清单。
- 将每道题提交到目标网页应用。
- 根据网页输出记录 `workflow_result`。
- 遇到不可访问或无法完成的题目时记录失败状态，而不是把结果错绑到下一题。
- 所有题目处理完毕后调用 `finish_subtask` 结束任务。

## 环境

- Windows 本地客户端。
- OpenAI Responses 服务：`http://127.0.0.1:8000/v1`
- 模型：`qwen35-vl`
- 输入文件：`.bench_logs\gui_agent_gaia\gaia_level1_text_only.limit3.jsonl`
- 目标应用：Doubao Web
- 题目数量：3

## 本轮修复

### Responses 工具拆分

服务端 Responses 工具标准化层补充处理：

- 当 Qwen 把 `record_workflow_result`、`finish_subtask`、`read_text_file` 等 Agent 工具错误塞进 `computer_use.actions` 时，拆成标准 `function_call`。
- 保留真正的鼠标、键盘、等待等 `computer_use` 操作。

### Runner 工具恢复

Runner 增加保守恢复逻辑：

- 当模型文本明确表达“记录结果 / 保存答案 / 记录失败状态”，但结构化工具仍是错误的 `computer_use` 时，恢复为 `record_workflow_result`。
- 记录目标使用“下一个未归档题目”，避免失败题未记录时把下一题答案错绑到上一题。
- 修正中文“失败 / 无法 / 不能”等词没有被当作 workflow 结果证据的问题。

### 防止测试预算被修复轮耗尽

`GuiAgentRunner` 增加 repair turn 预算；工具解析修复不再直接耗尽业务步骤预算。

### repeated-action 检测修正

修正 click-only 重复检测过严的问题：中间存在输入、等待、混合操作时，不再把远距离点击误判为连续重复点击。

## 验证结果

### v11

Trace:

`.bench_logs\20260526_gui_agent_gaia\doubao_gaia_trace_protocol_v11_limit3.jsonl`

结果：未通过。

关键问题：

- 第 1 题成功归档：`4`。
- 第 2 题模型已经判断 YouTube 视频因区域限制无法访问，但中文“失败”未被识别为 workflow 结果证据。
- 第 2 题未记录，后续动作持续误点，最终进入人工等待。

修复：

- 将中文 `失败 / 无法 / 不能` 加入 workflow 结果证据判断。

### v12

Trace:

`.bench_logs\20260526_gui_agent_gaia\doubao_gaia_trace_protocol_v12_limit3.jsonl`

结果：通过，最终状态 `completed`。

归档结果：

| 题目 | 状态 | 输出 |
|---|---|---|
| Mercedes Sosa 专辑数量 | passed | `4` |
| YouTube 视频鸟类数量 | failed | 因区域限制无法访问 YouTube 视频，无法获取答案 |
| `left` 的反义词 | passed | `right` |

最终调用：

- `record_workflow_result` 三次。
- `finish_subtask` 一次。

## 暴露的问题

### 输入动作仍然不稳

模型多次把“点击输入框、输入文本、发送”拆成多轮执行，甚至重复输入同一题目。v12 中第 2 题出现过：

- 先输入问题。
- 再次输入同一问题。
- 检测到输入框混入无关文本后，执行 `Ctrl+A`、删除、重新输入、发送。

这说明当前让模型直接控制低层输入动作可用，但效率不稳定。

后续建议：

- 引入通用 `submit_text_to_active_app` 类工具，由 harness 完成清空输入框、输入文本、提交、等待变化。
- 工具仍保持通用，不绑定豆包；APP 配置提供输入区域和提交方式。

### Agent 工具与 computer_use 混合返回

v12 第 1 步模型同时返回：

- 一个无意义的 `computer_use.click`
- 一个正确的 `read_text_file`

当前两个都执行了。后续可以加策略：

- 当同一轮同时存在 Agent 工具和疑似无意义的 `computer_use` 时，优先执行 Agent 工具，并忽略无关屏幕点击。

### 响应延迟仍偏高

v12 单轮模型响应大多在 20-35 秒。主要原因仍需拆分：

- Windows prefill 路径较慢。
- 每轮新增截图和 runtime context 仍偏大。
- 模型多轮低层输入动作导致额外往返。

## 当前结论

多题 workflow 的“读取清单 -> 连续提交 -> 逐题归档 -> 失败题不串题 -> 结束任务”链路已经跑通。

下一步应把低层文本提交动作收敛为 harness 侧稳定工具，减少模型多轮重复点击和重复输入，让 GUI Agent 更接近可用的自动化执行器。

## 思考模式实验

### 背景

为了验证“用更多时间换操作准确性”是否成立，额外测试了 Qwen3.5 thinking 路径。

初始尝试只传：

- `reasoning={"effort":"low"}`
- `chat_template_kwargs={"enable_thinking": true}`

但服务端没有启用 reasoning parser 时，Qwen 的内部思考会直接混进 `output_text`，并带有 `</think>`。这会污染 GUI Agent 的工具解析和用户可见输出，因此不能直接用于客户端。

随后重启 OpenAI 服务，加入：

```powershell
--reasoning-parser qwen3
```

直接小请求验证通过：Responses 输出被拆成：

- `type="reasoning"`：内部思考
- `type="message"`：普通可见输出

### v14: low thinking + qwen3 parser

Trace:

`.bench_logs\20260526_gui_agent_gaia\doubao_gaia_trace_protocol_v14_low_thinking_parser_limit3.jsonl`

配置：

- `reasoning_effort=low`
- `chat_template_kwargs.enable_thinking=true`
- `max_output_tokens=1024`
- `reasoning_parser=qwen3`

结果：不通过。

现象：

- 第 1 步和第 3 步出现 `reasoning` item，parser 能正确分离思考。
- 第 3 步仍然重复输入第 1 题，没有改善低层输入不稳问题。
- 第 4 步模型把 1024 个输出 token 全部花在 reasoning 上，没有输出任何工具调用。
- 旧 runner 逻辑把“无可见输出且无工具调用”误判为完成，导致任务 4 步结束，0 条 workflow result。

修复：

- Runner 增加 `empty_response_without_tool_call` 修复路径：当任务仍 active 且模型只输出 reasoning / 空响应 / 无工具调用时，不再完成任务，而是要求模型返回一个完整工具调用。
- 单测覆盖：`test_runner_retries_reasoning_only_response_for_active_task`。

### 结论

当前 `low thinking` 不适合作为 GUI Agent 默认路径：

- 响应时间更长。
- 对重复输入、提交按钮定位没有明显改善。
- 容易把输出预算耗尽在 reasoning 上，降低工具调用稳定性。

保留策略：

- 默认仍使用 `reasoning_effort=none`。
- 对高风险步骤可以后续设计“局部二次确认”模式，而不是全流程开启 thinking。
- 更实际的优化方向仍是增加通用 harness 工具，例如 `submit_text_to_active_app`，由程序稳定完成清空输入框、输入、提交、等待变化。
