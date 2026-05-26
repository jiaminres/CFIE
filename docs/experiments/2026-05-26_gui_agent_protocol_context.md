# 2026-05-26 GUI Agent 协议与上下文修正

## 设计边界

- `cfie_gui_agent` 是通用 GUI Agent 应用层，不应硬编码“测试条目输入器”之类的场景专用能力。
- AI 应用测试、豆包自动化、游戏自动化、电商后台等都只是 APP 会话里的任务配置，不应反向污染产品代码的通用抽象。
- Responses / computer-use 协议适配属于通用协议层，后续应继续从 `cfie_gui_agent` 下沉到 `cfie_client` 或服务端 Responses 兼容层。

## 本轮结论

- 撤销 `type_current_workflow_item` / 自动输入路线。
- 豆包工作流脚本只把当前待处理条目作为任务上下文提供，模型必须通过通用 `computer_use` 的 `type` 动作完成输入。
- 当前需要优先修正的是上下文异常，而不是用程序代替模型做判断：
  - `function_call_output.output` 中的 `data:image/...base64` 会被转成 tool message 文本，导致一张小截图膨胀成数千到上万 token。
  - 截图应只进入 trace/UI 或显式 `input_image`，不应作为工具输出 JSON 文本进入模型上下文。

## 后续验证方向

- 修复 base64 URL 膨胀后，重新测试豆包自动化流程。
- 如果模型仍然出现简单操作判断异常，再测试短思考模式，而不是添加场景硬编码。
- 短思考目标：只允许模型简短描述上一状态和下一动作，避免长篇思考拖慢 GUI 响应。

## 截图分辨率

- GUI Agent 不能长期依赖场景专用提交工具规避定位问题；通用自动化需要模型能够稳定输出精准操作坐标。
- `512x288` 对网页底部发送按钮这类小控件过于紧，按钮只占很少像素，容易放大坐标误差。
- Qwen3.5 VL processor 粗测视觉 token：
  - `512x288`: 约 144 visual tokens
  - `768x432`: 约 336 visual tokens
  - `960x540`: 约 510 visual tokens
  - `1280x720`: 约 880 visual tokens
- 当前将 GUI Agent 工作流默认截图提升到 `1920x1080`、`image_detail=high`、JPEG quality `90`。这能减少小控件定位误差，但也要求上下文增量必须严格控制，不能把 base64 或冗长说明塞进文本上下文。

## Responses 输入组织

- developer message 不再把“运行规则”和运行状态 JSON 混在一个 `input_text` 里。
- 现在 developer message 的 `content` 固定拆成两段：
  - 第 1 段 `input_text`：中文运行规则、响应效率要求、工具调用约束。
  - 第 2 段 `input_text`：本轮运行状态 JSON，例如 active APP、active subtask、queue counts、recent steps、action macros、context budget。
- user message 继续承载当前任务或本轮观察：
  - 文本部分描述任务、坐标协议、局部定位兜底等；
  - 视觉部分使用 `input_image` / `input_video`，不把图片 base64 展开成文本。
- tool result 使用独立的 `function_call_output` item，不再和普通用户指令混在一起。

## 客户端详情栏

- 详情栏删除单独的“Responses 输出”摘要，避免把协议返回又人为压成一段难读文本。
- 详情栏保留：
  - “最新新增输入”：直接引用完整请求上下文中本轮新增的 user/tool input item；
  - “完整请求上下文”：按 system/developer/user/tool 分组展示；
  - “Responses 返回对象”：按固定协议字段做嵌套折叠，不再显示一个整块 JSON 文本。
- request context 的 content part 采用中文标签，例如“文本输入”“图片输入”“视频输入”。JSON 字段可逐层展开，方便查看 `output`、`tool_calls`、`usage` 等结构。

## 详情栏验证补充

- 实测发现原始编号容易误导：外层 input item 已经显示“第 N 条 / 用户 / message”，message 内部 content 再显示 `1. 文本输入` 会让用户误以为有两个第 1 条。
- 现在 message 内部改为：
  - 单段文本：`文本内容`
  - 单张图片：`图片内容`
  - 多段文本：`文本内容 1/2`、`文本内容 2/2`
  - 多张图片：`图片内容 1/2`、`图片内容 2/2`
- 详情栏新增左侧拖拽手柄，可调整右侧详情窗口宽度。
- Responses 返回对象中的 JSON 字符串会继续尝试递归解析；例如 `output[0].arguments` 这类字符串字段可以继续展开查看内部参数。
- 豆包验证中模型曾把 `record_workflow_result` 错误包进 `computer_use.actions` 的 `call_function` 动作。协议解析层已补充兼容：这类嵌套调用会被提升为真正的 Agent 工具调用，不再作为电脑操作重复执行。
## Responses 工具调用标准化

- 结论：Qwen3.5 的 tokenizer chat template 已支持 `tools`、`<tool_call>`、`<tool_response>` 和 `enable_thinking`。服务端应使用该模板，并在 Responses 协议层把模型吐出的 Qwen 原生工具文本转换成标准 `function_call` 输出对象。
- 已将以下兼容从 GUI Agent 客户端兜底下沉到 `cfie/entrypoints/openai/responses/tool_call_normalizer.py`：
  - 标准 `<tool_call><function=...><parameter=...>`；
  - 裸 `<function=...>`；
  - `<tool_code>print(tool_name(...))</tool_code>`；
  - `computer_use.actions` 中嵌套的 `call_tool` / `call_function`；
  - `function.parameters` 形式的嵌套调用，例如 `{"type":"call_function","function":{"name":"finish_subtask","parameters":{...}}}`。
- 标准化规则：
  - 如果 `computer_use` 同时包含真实电脑动作和嵌套 Agent 工具，保留真实电脑动作，并额外追加标准 `function_call`；
  - 如果 `computer_use` 只包含嵌套 Agent 工具，则删除这个伪 `computer_use`，只返回真正的标准 `function_call`；
  - 参数里的 JSON 字符串和被引号包住的参数会被解析，避免 `coordinate_space` 变成 `"\"qwen_normalized_1000\""`。
- 验证：
  - `tests/unit/test_responses_tool_call_normalizer.py`
  - `tests/unit/test_gui_agent_openai_responses.py`
  - 相关 GUI Agent 嵌套工具解析单测

## 工具调用历史回写修正

- 发现问题：runner 之前只把工具执行结果写回上下文，没有把上一轮 assistant 输出的工具调用本体写回。结果是下一轮模型只能看到“执行后的截图/工具结果”，但不能结构化看到自己上一轮实际提交过哪些参数。
- 修正后上下文顺序改为：
  - assistant `function_call` / `computer_call`
  - tool observation：`function_call_output` 或 `computer_call_output`
  - 后续 user/runtime context
- `computer_call_output` 不再在客户端伪装成 `role=user` 的截图消息。请求 payload 中保留 `type=computer_call_output`，服务端 Responses 输入转换时再映射为：
  - `role=tool` 的 computer_use observation 文本；
  - 紧随其后的截图 image message，保证 Qwen VL 仍能看到操作后的画面。
- 这样模型下一轮既能看到视觉状态，也能知道该视觉状态来自哪个 `call_id` 的工具执行结果。
- 验证：
  - `tests/unit/test_gui_agent_openai_responses.py`
  - `tests/unit/test_cfie_client_gui_agent.py`
  - `tests/unit/test_gui_agent_workflow.py`
  - `tests/unit/test_responses_video_input.py`
