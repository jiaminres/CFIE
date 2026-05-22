# cfie_gui_testing

`cfie_gui_testing` 是面向 AI 应用自动化测试的业务包。

它可以复用 `cfie_client` 提供的 OpenAI computer harness，但不把测试用例、benchmark、judge、评分口径写进 `cfie_client`。

推荐边界：

```text
cfie_client
  -> 通用 computer 协议、执行器、截图、OpenTelemetry GenAI 轨迹

cfie_gui_testing
  -> AI 应用测试用例、benchmark 适配、结果汇总、评估框架对接
```

典型链路：

```text
GuiTestCase
  -> GUI Agent 调用 computer 协议
  -> cfie_client 执行动作并记录轨迹
  -> 外部 bench / judge 读取轨迹与应用结果
  -> GuiTestResult
```
