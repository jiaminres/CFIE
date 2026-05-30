# 2026-05-29 GUI Agent 点击局部兜底实验

## 目标

验证 GUI Agent 在小目标点击场景下，`computer_use` 的坐标输出、局部截图兜底和 `local_refinement_1000` 坐标口径是否能稳定工作。测试页面由本地 benchmark 生成，目标是多个小圆点，页面会记录每次点击是否命中。

本轮实验只用于验证通用点击能力，不把页面反馈或 DOM 信息作为产品能力硬编码进 GUI Agent。

## 代码变更

- 新增 `benchmarks/run_gui_agent_click_refinement_targets.py`：
  - 启动本地 tiny target 页面；
  - 打开独立 Edge app 窗口；
  - 使用真实 `GuiAgentRunner` + `ComputerLoop` + Responses Agent；
  - 记录 `result.json`、`agent_trace.jsonl`、`computer_trace.jsonl`。
- `ScaledPillowScreenCapture.screenshot_region_around()` 支持局部截图时关闭鼠标光标绘制。
- 点击后的局部修正图默认不绘制鼠标光标，避免光标遮挡 10px/18px 小目标。
- GUI Agent 默认 `local_click_refinement_radius` 调整为 `120`。

## 实验结果

| 场景 | 局部半径 | 思考 | 结果 | 点击 | 未命中 | 局部修正调用 | 耗时 |
|---|---:|---|---|---:|---:|---:|---:|
| 2 个 18px 目标 | 260 | none | 成功 | 2 | 0 | 0 | 62.7s |
| 2 个 10px 目标 | 220 | none | 失败 | 12 | 12 | 8 | 249.1s |
| 2 个 10px 目标 | 120 | none | 失败 | 12 | 12 | 6 | 241.4s |
| 2 个 10px 目标 | 120 | low | 失败 | 12 | 12 | 8 | 702.4s |
| 3 个 18px 目标 | 260 | none | 失败 | 10 | 10 | 9 | 200.9s |
| 3 个 18px 目标 | 120 | none | 目标已全部点中，但未 finish | 10 | 7 | 0 | 210.7s |
| 2 个 18px 目标 | 120 | none | 成功 | 4 | 2 | 3 | 170.5s |

关键日志：

- `quick2_radius260_nocursor`：2 个 18px 目标全部命中，说明不绘制鼠标光标后，基础点击链路可以正常工作。
- `quick2_size10_radius120`：多次点击距离目标中心约 8-9px，但 10px 目标半径只有 5px，因此纯模型坐标不足以稳定命中。
- `quick3_size18_radius120`：A/B/C 三个 18px 目标都被点中，但模型没有调用 `finish_subtask`，后续继续重复点击 C。这里是任务完成判断/提示词问题，不是点击链路问题。

## 结论

- 当前 Qwen3.5-VL + 1080p 截图口径下，18px 左右的小控件是比较现实的稳定点击下限。
- 10px 级别目标不能只依赖模型坐标，需要 harness 提供更强的高精度点击策略。
- 局部修正 crop 不应该绘制鼠标光标；光标会遮挡小目标并误导模型。
- 局部 crop 半径 120 比 260/520 更适合小控件，已经改为默认值。
- 提高思考强度不是解决小目标点击的主要路径。`low` 思考在 10px 目标上耗时显著增加，但仍未稳定命中。

## 后续优化方向

- 对 close miss/repeated click 建立高精度点击路径：
  - 第一次点击后固定返回无光标局部图；
  - 如果模型仍旧重复偏移，进入二阶段放大 crop；
  - 让模型先在局部图中给出目标中心，再由 harness 映射到全局坐标执行。
- 对“页面目标已经完成但模型未调用 `finish_subtask`”增加更明确的完成提示和停止策略。
- 测试页可以继续保留为 benchmark，但产品代码不能依赖 DOM 命中反馈。

## 2026-05-30 协议观察格式修正

用户指出：`computer_call_output` 的观察值不应该把所有信息压缩成一个大 JSON。更合理的格式是：

1. 外部自然语言观察说明，直接告诉模型当前发生了什么。
2. 局部截图或完整截图，作为独立 `input_image`。
3. 结构化观察数据 JSON，只保留坐标口径、是否执行、局部 box、click index 等机器可读字段。

已完成调整：

- Responses 协议层把 `computer_call_output.output.observation_text`、`image_url`、`structured_data` 拆成独立的 tool message content part。
- 局部修正图 `local_refinements` 也按“文本 + 图片 + 结构化 JSON”展开，即使没有主截图也不会丢失局部观察。
- GUI Agent 详情页不再优先展示一整块 `原始 output`，而是按“观察说明 / 观察截图 / 结构化观察数据 / 点击局部观察”渲染，降低用户和模型调试时的混乱度。

补充修正：

- 发现模型会把 `coordinate_space="local_refinement_1000"` 和全屏坐标混用，例如输出 `x=1356,y=321`。
- 旧逻辑会把这个非法局部坐标继续映射，导致点击飞到目标右侧。
- 新逻辑严格要求 `local_refinement_1000` 的坐标必须在 `0..1000` 内；超出范围直接拒绝执行并回传错误，避免危险误点击。

验证结果：

- `guard_reject_bad_local_size18_20260530_0038`
  - 2 个 18px 网页目标全部命中。
  - 耗时 `170.494s`，`9` 步。
  - 真实点击 `4` 次，其中 miss `2` 次，最终 A/B 都命中。
  - 第 7 步捕获到非法局部坐标并拒绝：`local_refinement_1000 coordinates must stay within 0..1000`。
  - 拒绝后模型回到 `qwen_normalized_1000`，第 8 步命中 B。

## 验证

```text
143 passed
```

## 2026-05-30 小目标尺寸递减验证

上一轮发现：即使局部图里目标对人眼很清楚，Qwen3.5-VL 仍可能把完整截图坐标复制成 `local_refinement_1000`，或者在 10px 目标上产生 8px 左右的偏移。因此继续做了三项修正：

- 局部精修 observation 的模型可见内容不再包含完整屏幕坐标；`proposed_click` / `attempted_click` 只保留在内部 trace metadata。
- 局部图中心标记从容易混淆的实心十字改成品红色角标，避免遮挡 8px/10px 小目标。
- harness 对局部图提取通用小控件候选 `local_visual_candidates`，以局部 0..1000 坐标给出候选中心；该逻辑只基于图像像素，不依赖 DOM 或测试页内部状态。候选目前覆盖高饱和组件和暗色紧凑组件，用于兼容极小目标在截图/JPEG 后变暗的情况。

验证配置：

- 引擎：Windows OpenAI Responses 服务，`qwen35-vl`。
- 截图：主图 1920x1080 low detail，局部精修图 high detail。
- 坐标：首次点击使用 `qwen_normalized_1000`，未确认的首次 click 必须先经过局部确认；修正点击使用 `local_refinement_1000`。
- 局部半径：`80`。
- 思考：`none`。
- 目标：本地生成网页上的 2 个彩色小圆点，页面只用于记录最终命中，不把 DOM 信息提供给 Agent。

结果：

| 目标尺寸 | 结果 | 步数 | 点击数 | 未命中 | 局部确认/修正 | 最近命中距离 | 耗时 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 18px | 成功 | 5 | 2 | 0 | 2 | 约 1.1px / 1.2px | 约 84s |
| 14px | 成功 | 5 | 2 | 0 | 2 | 0.5px / 0.5px | 89.3s |
| 10px | 成功 | 5 | 2 | 0 | 2 | 0.5px / 0.5px | 82.7s |
| 8px | 成功 | 5 | 2 | 0 | 2 | 1.1px / 0.5px | 81.9s |
| 6px | 成功 | 5 | 2 | 0 | 2 | 0.5px / 0.5px | 96.1s |
| 4px | 成功 | 5 | 2 | 0 | 2 | 0.5px / 1.2px | 87.3s |
| 3px | 成功 | 5 | 2 | 0 | 2 | 1.1px / 0.5px | 102.0s |
| 2px | 失败 | 10 | 5 | 5 | 5 | 最近约 1.1px | 205.5s |
| 2px / 半径 40 | 失败 | 10 | 5 | 5 | 5 | 最近约 1.1px | 203.2s |
| 2px / 半径 24 | 失败 | 10 | 5 | 4 | 5 | B 命中，A 偏约 42.5px | 193.7s |

关键 artifact：

- `.bench_logs/click_refinement_targets/candidate_r80_20260530_032444_size18`
- `.bench_logs/click_refinement_targets/candidate_r80_20260530_033328_size14`
- `.bench_logs/click_refinement_targets/candidate_r80_20260530_033328_size10`
- `.bench_logs/click_refinement_targets/candidate_r80_20260530_033328_size8`
- `.bench_logs/click_refinement_targets/candidate_recommended_r80_20260530_044247_size6`
- `.bench_logs/click_refinement_targets/candidate_recommended_r80_20260530_044450_size4`
- `.bench_logs/click_refinement_targets/candidate_recommended_r80_20260530_044646_size3`
- `.bench_logs/click_refinement_targets/candidate_recommended_r80_20260530_044646_size2`
- `.bench_logs/click_refinement_targets/candidate_recommended_r40_20260530_045707_size2`
- `.bench_logs/click_refinement_targets/candidate_recommended_r24_20260530_045707_size2`

结论：

- 在网页小目标场景中，`首次 click -> 局部确认 -> recommended_click_1000 -> local_refinement_1000` 已能稳定点击 3px 目标。
- 对 3px-10px 目标，不能把成功寄托在模型肉眼估坐标上；harness 需要提供局部图中的推荐视觉锚点。
- `local_visual_candidates` / `recommended_click_1000` 是通用图像处理辅助能力，不是测试场景硬编码：它只识别局部 crop 中的小型视觉组件，并用局部归一化坐标交给模型决策。
- 2px 目标目前不稳定：模型和 harness 已能把点击压到距目标中心约 1.1px，但测试页命中半径只有 1px，整数坐标、浏览器事件坐标和截图压缩后的候选中心会把点击推到边界外。继续把局部半径缩到 40/24 也不能稳定解决，24 半径还更容易被标签边缘和抗锯齿噪声干扰。该尺寸已低于多数真实 GUI 控件可操作范围，不建议作为通用自动化默认目标。
- 代价是每个未信任 click 会多一次模型确认，当前每个目标约增加 14-16s。后续对已成功、重复出现或人工批准的坐标，可以进入宏/可信坐标路径，减少二次确认。

补充验证：

```text
18 passed
```

## 2026-05-30 模型主导局部确认矩阵

用户判断是正确的：局部点击确认应该优先交给模型判断，高清手段、提示词、思考模式、上下文范围才是主要实验变量；机器视觉候选只能作为 fallback 对照，不能默认替代模型决策。

本轮代码调整：

- `GuiAgentRunner.local_click_refinement_assist` 默认从 `recommended` 改成 `none`。
- `benchmarks/run_gui_agent_click_refinement_targets.py --local-click-refinement-assist` 默认也改成 `none`。
- 保留 `candidates` / `recommended` 两档开关，方便做 fallback 对照。
- 局部 crop 放大仍使用非 AI 插值，默认 `nearest`。本轮没有使用 AI 超分或图像修复。
- 增加单测，确认默认局部确认不会把 `recommended_click_1000` / `local_visual_candidates` 暴露给模型。
- 增加单测，确认 fallback 候选排序不会把小噪点排在真正控件前面。

单测结果：

```text
85 passed
```

小矩阵口径：

- 引擎：Windows OpenAI Responses 服务，`qwen35-vl`。
- 截图：全局 `1920x1080`，局部 crop 半径 `80`，局部图放大到 `1080`，插值 `nearest`。
- 目标：本地网页 1 个 `10px` 红色小圆点。
- 步数：`max_steps=6`。
- 产物目录：`.bench_logs/click_refinement_targets/model_matrix_20260530/`。

| 配置 | 结果 | 点击 | miss | 局部确认 | 耗时 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| `assist=none, low, full` | 目标已命中，但未正常 finish | 3 | 2 | 3 | 138.5s | 模型能在局部图中自行确认 10px 目标，第一轮真实点击命中；后续错误在工具选择/完成停止。 |
| `assist=none, origin, full, no click preamble` | 失败 | 3 | 3 | 3 | 326.8s | 原生思考更慢，且定位明显偏离，不适合作为默认。 |
| `assist=none, low, local` | 失败 | 4 | 4 | 2 | 164.7s | local-only 能把局部确认输入降到约 3900 token，但精度没有改善。 |
| `assist=recommended, low, full` | 目标已命中，但未正常 finish | 3 | 2 | 3 | 165.8s | fallback 推荐点精度最高，第一轮命中距离约 1.1px；但不应作为默认模型能力结论。 |

关键观察：

- 纯模型路径并非不可行。`assist=none + low + click 专用引导 + full history` 在 10px 目标上第一次真实点击命中，说明局部高清图 + 引导词能让模型完成修正。
- `origin` 原生思考会显著拖慢，且容易在局部确认中把品红角标当作目标或继续使用错误全局估计。
- `local-only` 对延迟有帮助，但会让模型缺少任务状态背景；它适合作为优化选项，不适合直接替代 full history。
- 当前最大问题不是局部图质量，而是完成状态后的工具选择：模型思考里写“调用 finish_subtask”，但 Responses 返回的 function_call 仍是 `computer_use`，且 action intent 甚至写成“任务完成”。后续需要在工具 schema/提示中明确：完成任务必须调用 `finish_subtask`，不能用 `computer_use` 或点击动作表达。

暂定默认：

- 默认 `local_click_refinement_assist=none`，让模型根据高清局部图自行判断。
- 默认保留 click 专用 reasoning preamble，因为它能把局部坐标口径固定在 `local_refinement_1000`。
- 默认局部图放大使用 `nearest`；`bicubic` 可作为对照，`lanczos` 在小 UI 目标上不优先。
- `recommended` 保留为高精度 fallback 或可配置加速路径，而不是主路径。

## 2026-05-30 isolated 局部视觉子任务

用户提出的判断成立：局部确认不应该让模型知道完整历史、全局截图、上一轮坐标或“局部图中心是上次点击点”这类语义。更干净的方式是把局部确认降级为一个独立视觉子任务：

- 第一次 `computer_use` 只表达点击目标与全局坐标。
- 如果坐标未被认证，harness 以该坐标为中心截取局部高清图并放大。
- 下一轮只给模型一条局部任务消息：目标意图 + 局部图 + `local_refinement_1000` 坐标要求。
- 不提供完整历史，不提供全局截图，不提供上一条 `function_call`，不提供机器推荐点，不绘制中心标记。
- 模型只需要在这张局部图中重新选择目标元素中心。

代码变更：

- `OpenAIResponsesAgent.click_local_refinement_context_mode` 新增并默认使用 `isolated`。
- `benchmarks/run_gui_agent_click_refinement_targets.py --local-click-refinement-context` 新增 `isolated`，并设为默认值。
- 局部确认的默认中心标记关闭，避免诱导模型机械输出 `x=500,y=500`。
- click-local reasoning preamble 改为“重新定位目标中心”，不再声明局部图中心是上次点击点。

对照实验：

| 配置 | 目标 | 结果 | 局部轮输入 | 命中 | miss | 耗时 |
|---|---:|---|---:|---:|---:|---:|
| `origin + local + no preamble` | 1 个 10px | 失败 | 约 3895 tokens | 0 | 3 | 433s |
| `origin + isolated + no preamble` | 1 个 10px | 成功 | 约 2575 tokens | 1 | 0 | 142s |
| `origin + isolated + no preamble` | 2 个 10px | 成功 | 约 2575 tokens/局部轮 | 2 | 0 | 255s |

关键 artifact：

- `.bench_logs/click_refinement_targets/origin_local_only_no_preamble_cleanprompt_1target_20260530_150231`
- `.bench_logs/click_refinement_targets/origin_isolated_no_preamble_1target_20260530_151509`
- `.bench_logs/click_refinement_targets/origin_isolated_no_preamble_2targets_20260530_151806`

结论：

- `isolated` 明显优于原先的 `local`：它删除了历史失败和中心点语义干扰，输入 tokens 更少，点击准确率更高。
- `origin` 原生思考可以完成局部图重新定位，但延迟仍高，每个局部轮约 54-58s；后续默认可继续测试 `low/minimal + isolated + 短提示`，争取保留准确率同时降低延迟。
- 当前默认策略应先采用 `isolated` 局部上下文；机器候选点只作为 fallback，不作为模型能力主路径。

## 2026-05-30 ??????

??? click ????????????

- ?????`isolated` ??????? + ???????
- ???????? `local`/`recommended`/`candidates` ???????? `recommended_click_1000` ? `local_visual_candidates` ??????
- ???????????? click ?????????????????? `local_refinement_1000` ?????????
- ?? crop ???? PNG ????? JPEG ??????????????????? JPEG ????????
- benchmark ??????????????????????????????????????????????????

?????

| ?? | ?? | ?? | ???? | miss | ???? | ?? |
|---|---:|---|---:|---:|---:|---:|
| `low + isolated + PNG + no marker` | 5 ? 10px | ???? | 5 | 0 | 5 | 305.6s |
| `low + isolated + PNG + no marker` | 3 ? 6px | ???????????? finish | 15 | 12 | 15 | 842.2s |

?? artifact?

- `.bench_logs/click_refinement_targets/low_isolated_short_png_nomarker_5target_20260530_162819`
- `.bench_logs/click_refinement_targets/low_isolated_short_png_nomarker_3target6px_20260530_163401`

???

- 10px ?????????????????????????
- 6px ????????? GUI ?????????????????????/finish ?????????????????????????????? click ?????
- ???? GUI Agent ?????????? click ?????????????????? `guard_untrusted_clicks`?

