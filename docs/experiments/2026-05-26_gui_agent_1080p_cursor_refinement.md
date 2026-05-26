# 2026-05-26 GUI Agent 1080p 截图与局部定位兜底

## 背景

豆包网页自动化测试中，模型已经能大致识别输入框和发送按钮，但点击点仍可能偏差几十像素。继续单纯提高全屏截图分辨率会增加视觉 token 和 prefill 延迟，因此需要在 harness 层提供定位兜底。

## 本轮修改

- GUI Agent workflow 默认截图改为 `1920x1080`。
- workflow 默认 `image_detail=high`，JPEG 质量改为 `90`。
- 截图采集层默认绘制当前鼠标指针。PIL `ImageGrab.grab()` 默认不会包含鼠标，这会让用户和模型都难以判断“当前点在哪里”。
- `ScaledPillowScreenCapture` 新增 `screenshot_region_around()`，可围绕某个截图坐标生成局部高清 crop。
- harness 对 click-only / 可疑点击步骤增加局部定位兜底：
  - 以模型上一次点击点为中心截取小图；
  - 下一轮把局部图作为额外视觉输入；
  - 注册临时坐标口径 `local_refinement_1000`；
  - 如果模型基于局部图重新定位，harness 会把局部 `0..1000` 坐标换算回完整截图坐标再执行。

## 设计原则

- 不在业务层硬编码“豆包发送按钮”等具体规则。
- 不让模型直接输出物理屏幕坐标。
- 默认仍使用 `qwen_normalized_1000` 对完整截图定位。
- 只有在局部兜底图生效时，才允许使用 `local_refinement_1000`。
- 局部 crop 只作为定位补偿，不替代主历史上下文。

## 验证

- 真实截图采样：当前 `2560x1440` 桌面会输出 `1920x1080` 图像。
- 采样图中已经可见鼠标指针。
- 单元测试：
  - `tests/unit/test_cfie_client_gui_agent.py`
  - `tests/unit/test_gui_agent_desktop_client.py`
  - `tests/unit/test_gui_agent_workflow.py`
  - `tests/unit/test_gui_agent_openai_responses.py`
  - `tests/unit/test_gui_agent_architecture.py`
  - `tests/unit/test_responses_tool_call_normalizer.py`
  - `tests/unit/test_responses_video_input.py`
- 结果：`114 passed`。
