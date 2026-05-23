# cfie_gui_agent

`cfie_gui_agent` is the general GUI automation Agent application layer built on
top of `cfie_client` and the CFIE local inference service.

It is not limited to test automation. It is intended for desktop application
automation, browser workflows, e-commerce operations, future mobile/remote
device control, and GUI Agent orchestration.

## Boundary

```text
CFIE
  -> local multimodal model inference service

cfie_client
  -> generic computer-use execution SDK

cfie_gui_agent
  -> application-level Agent harness
  -> Job/Subtask scheduling
  -> context management
  -> human intervention
  -> trace and reward data
```

`cfie_client` executes generic computer actions. `cfie_gui_agent` owns
long-horizon automation semantics: Job state, Subtask queues, runtime context,
human-loop state, and scheduler metadata.

## Main Objects

- `GuiAgentTaskSpec`: input spec for one GUI Agent run.
- `WorkspaceProfile`: long-horizon workspace/business profile with target apps,
  rules, references, and SOPs.
- `GuiAgentRunner`: minimal runner that connects a model response function with
  `ComputerLoop`.
- `GuiAgentResult`: run result with metadata for traces and debugging.
- `JobBoard`: global view of Jobs, active Job, and recent switch history.
- `JobState`: one APP/window/logical workspace context container.
- `SubtaskState`: concrete work item under one Job.
- `SubtaskQueues`: running, urgent, runnable, waiting-human, blocked, completed,
  failed, cancelled, and superseded queues under one Job.
- `MonitorController`: validates monitor events and creates Subtasks under the
  matching Job.
- `ActionMacroRegistry`: user-registered low-latency shortcut/combo macros.
- `NavigationPlanner`: harness-owned target navigation planning from source,
  target, and obstacle coordinates.
- `ContextManager`: selects current frame, recent video, mid-history
  after-frames, and long summary under the 43-frame default policy.
- `ModelToolRegistry`: model-callable tool allowlist and harness-internal tool
  boundary.
- `HumanLoopManager`: human intervention request/reply abstraction.
- `PolicyStore`: structured user/business constraints produced by corrections
  and `update_constraints`.
- `StepVerifier`: no-screen-change and repeated-action detection for loop
  governance.
- `AgentTraceStore`: application-level trace for Job/Subtask/tool/policy events.
- `RuntimeContextBuilder`: builds the bounded model-facing context payload from
  JobBoard, per-Job context, policy, and tool metadata.

## Low-Latency Mode

For high-frequency interaction such as games, the model should not output every
single key press. The application can register action macros such as:

```text
combo_asd = A, S, D
select_all_then_b = CTRL+A, B
```

The model calls `run_action_macro` with the macro name. The harness expands and
validates the registered human-like key sequence.

Agility context mode uses after-action frames only:

```text
current frame: 1
recent video: 0
after-frame history: 42
total visual frames: 43
```

This avoids frequent video-prefill overhead in low-latency scenarios.

For continuous movement, the model can call `navigate_to_target` with source and
target coordinates plus obstacle polygons. The harness owns the route planning,
real-time tracking, retry, and stop condition; model output is only the semantic
navigation request.

`TaskStack` still exists as a compatibility/helper structure for local
interrupt/resume semantics, but the main scheduling model is Job/Subtask.

## Minimal Example

```python
from cfie_gui_agent import GuiAgentRunner, GuiAgentTaskSpec

task = GuiAgentTaskSpec(
    task_id="open-settings",
    instruction="Open the settings page and report the page title.",
    target_app="demo_app",
)

runner = GuiAgentRunner(max_steps=6)
result = runner.run_task(task, agent=my_responses_agent)
print(result.to_trace_payload())
```

`my_responses_agent` is an injected model-call function. It receives a Responses
conversation list and returns a response object containing `computer_call` items
or a final `message`.
