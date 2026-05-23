# cfie_client

`cfie_client` 是 CFIE 的通用 computer-use 客户端 harness。它只负责理解 OpenAI Responses 风格的 `computer_call` / `computer_call_output`，执行本地或远端设备动作，并回传新的屏幕观察结果。

它不是具体业务应用。自动化测试、GUI Agent、benchmark、judge、手机 ADB 控制、远端桌面控制等能力应作为上层应用或 backend 接入，不写进核心协议层。

## 边界

```text
模型 / 推理服务
  -> computer_call { call_id, actions[] }
cfie_client
  -> SafetyGate 校验
  -> ComputerExecutor 调用 backend
  -> ScreenCapture 截图
  -> computer_call_output { call_id, computer_screenshot }
上层应用
  -> 继续把 computer_call_output 发回模型
```

## 主要对象

- `ComputerAction`: 标准化 click、double_click、scroll、type、wait、keypress、drag、move、screenshot。
- `ComputerCall`: 一轮模型请求的动作集合。
- `ComputerLoop`: 执行 `computer_call` 并生成 `computer_call_output`。
- `ComputerExecutor`: 把协议动作分发到实际 backend。
- `SafetyGate`: 执行动作前做本地边界检查。
- `ScreenCapture`: 生成 `computer_screenshot` 所需的 data URL。
- `TraceStore`: 记录 OpenTelemetry GenAI 风格 JSONL 轨迹，并把大图像 data URL 落盘为 artifact。

## Backend

当前内置 Windows backend，用于电脑界面控制。后续可以按同一个 `ComputerBackend` 协议接入：

- ADB / 远端 Android 设备
- iOS 远控桥接
- VNC / RDP / 浏览器远端环境
- 沙箱化桌面

这些 backend 不需要改变 `computer_call` 协议，只需要实现鼠标、键盘、滚动、等待等基础动作。

## Responses 视频输入

CFIE 的 Responses 入口额外接受 `input_video` 内容块，并把它映射到 Qwen3.5-VL 的原生 video 多模态路径：

```json
{
  "type": "message",
  "role": "user",
  "content": [
    { "type": "input_text", "text": "概括这段 GUI 操作视频。" },
    {
      "type": "input_video",
      "video_url": "file:///D:/videos/gui_trace.mp4"
    }
  ]
}
```

本地文件仍受服务端 `--allowed-local-media-path` 约束；也可以传 `data:video/...;base64,...` 或 HTTP URL。

## 最小示例

```python
from cfie_client import ComputerLoop

loop = ComputerLoop(trace_path="logs/gui_trace.jsonl")

response = {
    "output": [
        {
            "type": "computer_call",
            "call_id": "call_001",
            "actions": [
                {"type": "click", "x": 640, "y": 420, "button": "left"},
                {"type": "wait", "seconds": 0.5},
            ],
        }
    ]
}

outputs = loop.handle_response(response)
payload = [item.to_openai_dict() for item in outputs]
```

`payload` 可以作为下一轮 Responses API 的输入，继续让模型观察执行后的屏幕。
