# 2026-05-23 GUI_Agent Task Orchestration and Context Policy

## Scope

This note records the agreed engineering direction before implementation. No code
rename is required yet.

The application-level automation module should be treated as `GUI_Agent`: the first
application-layer paradigm built on top of the local CFIE inference engine.

Layer boundary:

- `CFIE`: local model inference engine, the model brain.
- `cfie_client`: generic computer-use tool client and execution SDK.
- `GUI_Agent`: application-level harness, task orchestration, context
  management, trace, judge, and automation workflow.

The goal is to make the design directly comparable with interview expectations:
context management, task orchestration, scheduling, development workflow, and
stability governance.

## Task Definition

Long-horizon tasks should come from user intent or external configuration, not
from hard-coded scripts. For example, in a game automation scenario:

- level up to a target level;
- prioritize main quests;
- avoid paid items;
- describe game rules, map rules, dungeon rules, NPC names, skill behavior, and
  resource constraints;
- allow revive after death;
- clean inventory only under explicit policy.

Recommended hierarchy:

```text
Campaign / UserGoal
  -> Quest / Subtask
    -> Step / Atomic Tool Action
```

Example:

```text
Campaign:
  Goal: level character from 1 to 20.

Subtasks:
  accept quest
  navigate to dungeon
  enter dungeon
  clear mobs
  claim reward
  repeat or replan based on new state

Atomic actions:
  click
  type
  press
  wait
  screenshot
  drag
  finish
```

Important decision: medium-level subtasks must not be generated once and then
blindly executed to the end. The environment can change. The agent should use
rolling planning, also known as receding-horizon planning:

```text
observe current state
validate current subtask
continue, interrupt, override, rollback, or replan
execute only the next bounded step
verify result
repeat
```

## Subtask Types

Subtasks should carry explicit semantics so that the harness does not treat all
failures as the same retry loop.

Primary task:

- The normal task being pursued.
- Example: clear dungeon A, finish daily quest, send a message in an AI app.

Interrupt task:

- A temporary task inserted above the current task.
- The original task is paused and should be resumed after the interrupt is
  completed.
- Examples: revive after death, close blocking pop-up, reconnect network,
  clear inventory under a permitted policy, confirm a safe dialog.

Override task:

- A replacement task that makes the previous task no longer active.
- The old task is marked `superseded` and should not be resumed.
- Examples: user changes the goal, current quest is no longer valuable, dungeon
  cannot be entered, strategy switches to a higher-priority task.

Recovery / rollback task:

- A task used to return to a stable checkpoint after a bad or uncertain state.
- Examples: return to town, reopen quest panel, refresh page, go back to a known
  route, restart a browser flow.

Verification task:

- A task whose purpose is to check whether the expected result is already true.
- Examples: confirm that the reply appeared, confirm that the quest progress is
  complete, confirm that a page changed after a click.

Maintenance task:

- A low-priority support task that can be inserted when needed.
- Examples: clean inventory, wait for cooldown, save trace, take an additional
  screenshot, recover focus.

## Task State Machine

Each task should have one explicit state:

- `active`: currently executing.
- `paused`: temporarily suspended and expected to resume.
- `completed`: success condition satisfied.
- `failed`: failed without being explicitly cancelled.
- `cancelled`: stopped by external/user control.
- `superseded`: replaced by a new task and should not resume.

Task transition operations:

```text
push_interrupt(task_b)
pop_resume()
override(task_b)
rollback(checkpoint)
```

`push_interrupt(task_b)`:

- Pause current task A.
- Push temporary task B onto the task stack.
- B becomes active.

Example:

```text
[A active]
death detected
[A paused, B revive active]
```

`pop_resume()`:

- Complete or cancel the interrupt task.
- Pop it from the task stack.
- Resume the previous paused task.

Example:

```text
[A paused, B completed]
pop B
[A active]
```

`override(task_b)`:

- Replace current task A with task B.
- A becomes `superseded`.
- B becomes active.

Example:

```text
[A active]
new higher-priority target detected
[A superseded, B active]
```

`rollback(checkpoint)`:

- Return the environment to a known stable state.
- Then replan from that checkpoint.

Example:

```text
enter dungeon failed
rollback to town or quest panel
replan route
```

## Harness-Controlled Constraints

The task framework must be enforced by program logic, not only by prompt text.
The model may propose, but the harness must validate and decide.

Model responsibilities:

- understand current UI state;
- identify controls and coordinates;
- suggest next action;
- suggest whether a state looks like death, dialog, reward, loading, or task
  completion;
- output structured tool calls.

Harness responsibilities:

- maintain `JobBoard`, per-Job `SubtaskQueues`, and switch history;
- decide whether `interrupt`, `override`, `rollback`, or `continue` is legal;
- enforce max steps, max duration, retry limits, and safety policies;
- validate tool schema;
- validate coordinates against screen bounds;
- detect no-op clicks and repeated loops;
- detect unchanged screens after actions;
- decide whether a subtask is done, invalid, failed, or still active;
- record trace and artifacts;
- prevent dangerous or policy-forbidden actions.

Recommended runtime loop:

```text
while automation is running:
    state = observe()
    update screen state and action trace

    if multi_app_monitor_enabled and monitor_due:
        monitor_result = model/classifier.detect_events(state)
        harness validates detected events
        harness creates Subtasks under matching Jobs

    if current_subtask.completed(state):
        move current Subtask to completed queue
    elif current_subtask.needs_human(state):
        move current Subtask to waiting_human queue
    elif current_subtask.blocked(state):
        move current Subtask to blocked queue

    job, subtask = job_board.select_next(safe_to_interrupt)
    runtime_context = build_context(job_board, job, subtask, trace, memory)
    model_action = model.propose(runtime_context)
    checked_action = harness.validate(model_action)
    result = executor.run(checked_action)
    verifier.update(result)
    compactor.update(trace, result)
```

The key rule is:

```text
Model proposes -> Scheduler/TaskManager validates -> Harness executes -> Verifier checks.
```

## Prompt and Runtime Context Split

Stable orchestration rules belong in system/developer instructions and code.
Task-specific goals belong in user/task configuration. Dynamic state belongs in
runtime context.

System/developer level:

- agent role;
- output schema;
- safety rules;
- task stack semantics;
- interrupt/override/rollback definitions;
- tool-call restrictions.

User / TaskSpec level:

- current long-horizon objective;
- game/application rules;
- success condition;
- user constraints;
- resource policy;
- allowed and forbidden operations.

Runtime context:

- current task stack;
- current subtask;
- current screenshot/frame;
- recent actions;
- failure memory;
- known UI targets;
- artifact references;
- current verifier status.

Critical implementation decision: task migration must not rely on model text
alone. The harness should convert model proposals into task operations only
after programmatic checks.

## Historical Context Policy

The model benefits from history, but long visual history increases latency and
context cost. The agreed policy is layered by recency and fidelity:

- current state: highest fidelity;
- recent history: video or dense frame sequence;
- mid history: one after-frame per action;
- long history: structured text summary with artifact references.

Important storage rule:

```text
Do not store both before and after images for every step in the model context.
The after image of step N is the before image of step N+1.
```

The artifact store may keep all raw screenshots and videos for replay, but the
model input should carry only the selected budgeted subset.

## Default 128K Visual Budget

Based on previous observations:

- 2K 50 frames used roughly 140K context tokens.
- 1080p has about `1 / 1.78` of the 2K display-pixel area.
- 128K context can roughly fit about 80 frames of 1080p if the context were
  mostly visual.

The default GUI_Agent policy should not spend the full context on images. The
selected default is 43 frames at 1080p-equivalent resolution.

Frame allocation:

```text
current frame:
  1 frame

recent high-fidelity video history:
  last 2 steps * 6 frames per step = 12 frames

mid-history after frames:
  last 25 steps * 1 after frame = 25 frames

key evidence frames:
  5 frames

total:
  1 + 12 + 25 + 5 = 43 frames
```

Estimated visual context:

```text
1080p frame estimate: about 1.57K tokens
43 frames: about 67K tokens
remaining from 128K: about 60K tokens
```

The remaining context is reserved for:

- system/developer instructions;
- task stack;
- user long-horizon goal;
- tool schema;
- harness execution state;
- failure memory;
- mid/long text summaries;
- model output budget.

Default policy object:

```json
{
  "vision_context_policy": {
    "resolution": "1080p",
    "max_visual_frames": 43,
    "current_frame": 1,
    "recent_video_steps": 2,
    "frames_per_recent_step": 6,
    "mid_history_after_frames": 25,
    "key_evidence_frames": 5
  }
}
```

Downgrade order if context budget is exceeded:

1. Reduce mid-history after frames from 25 to 15.
2. Reduce key evidence frames from 5 to 3.
3. Reduce recent video frames per step from 6 to 4.
4. Reduce recent video steps from 2 to 1.
5. Keep only current frame plus text summaries and artifact references.

Upgrade order if budget is available:

1. Add frames to recent video history.
2. Add evidence frames around failures and state transitions.
3. Add more mid-history after frames.
4. Avoid adding long-range video unless explicitly requested by the task.

## Context Compaction Design

The GUI_Agent context manager should never rely on the model to directly write a
large compressed history. The client environment cannot afford large model
outputs during every compaction cycle, and free-form summaries are hard to verify
or replay.

Agreed principle:

```text
Model decides semantic importance and proposes a compaction plan.
Program executes the compaction deterministically.
```

The system should maintain two separate layers:

Artifact store:

- complete raw screenshots;
- video frames;
- tool calls;
- execution results;
- verifier outputs;
- failure labels;
- reward/preference labels;
- trace metadata.

Prompt context:

- current frame;
- recent video history;
- mid-history after frames;
- key evidence frames;
- structured long-history text;
- artifact references.

The artifact store is the source of truth. The prompt context is only a bounded
decision view.

### Time-Based Degradation

The 43-frame policy should behave like a sliding degradation pipeline:

```text
new step
  -> recent high-fidelity video history
  -> mid-history after frame
  -> long-history text summary + artifact refs
```

Every completed action adds one new step. The context manager then performs:

1. New step enters recent video history.
2. Steps older than the last 2 recent steps lose video fidelity and keep only
   one after-frame.
3. Mid-history after frames older than the last 25 steps are compacted into
   structured text plus artifact references.
4. Failure, rollback, override, first-arrival, task-completion, and other key
   evidence frames may be preserved in the 5-frame evidence budget.

Important optimization:

```text
Only one after-frame is needed per action in model context.
Step N after-frame is also step N+1 before-frame.
```

### Compaction Trigger Lines

Compaction should happen before the context window is exhausted.

Recommended thresholds:

```text
normal:
  estimated_context_usage < 70%
  keep default 43-frame policy

warning:
  70% <= estimated_context_usage < 85%
  reduce low-value mid-history frames and merge unimportant text records

compact:
  85% <= estimated_context_usage < 95%
  ask the model for a structured compaction plan

emergency:
  estimated_context_usage >= 95%
  programmatic forced downgrade; keep current frame, last 1 step, active task
  state, failure memory, and long-summary refs only
```

Additional event-based triggers:

- every 10 steps: lightweight cleanup;
- every 30 steps: episode-level compaction;
- on `override`: compact superseded task history;
- on `rollback`: preserve rollback evidence and compact the failed path;
- on task completion: summarize the completed subtask;
- on repeated no-op actions: compact repeated wait/click history.

### Model Compaction Plan

The model should output a small structured compaction plan, not a long
replacement context.

Example schema:

```json
{
  "merge_steps": [
    {
      "steps": [12, 13, 14],
      "summary": "Tried the login button and reached an irrelevant QR login page.",
      "keep_evidence": ["step_14_after"],
      "drop_video": true,
      "downgrade_to_text": true,
      "tags": ["failed_path", "do_not_repeat"]
    }
  ],
  "keep_visual_steps": [
    {
      "step": 21,
      "reason": "First arrival at the target page; keep after-frame as evidence."
    }
  ],
  "drop_visual_steps": [
    {
      "steps": [8, 9, 10],
      "reason": "Repeated waits with no visual change."
    }
  ],
  "do_not_repeat": [
    "Do not click the login button for this task."
  ],
  "known_targets": {
    "chat_input": {
      "description": "Input box at the bottom of the page.",
      "evidence": "step_27_after"
    }
  },
  "open_questions": [
    "Need to verify whether the send button is disabled before typing."
  ]
}
```

The program should validate the plan before applying it:

- referenced step IDs must exist;
- evidence refs must exist in the artifact store;
- step ranges must not include the current active step unless explicitly
  allowed;
- active interrupt-task evidence should not be dropped;
- safety and failure records should not be removed without a replacement text
  record;
- summaries should be short and bounded.

### Programmatic Compaction Actions

After validating the model plan, the context manager may:

- remove video frames from prompt context while keeping artifacts on disk;
- convert a recent step video into a single after-frame;
- merge multiple step records into one text record;
- update `long_history_summary.completed`;
- update `long_history_summary.failed_attempts`;
- update `long_history_summary.do_not_repeat`;
- update `known_targets`;
- preserve key evidence frames;
- update context hashes for replay and audit.

The model may suggest, but code performs:

```text
drop frames
downgrade video
merge step records
write artifact refs
update long summary
update failure memory
```

### Step Record Requirements

Every step must be indexable so compaction can be deterministic.

Recommended step record:

```json
{
  "step_id": 27,
  "task_id": "send_message",
  "subtask_id": "find_chat_input",
  "action": {
    "type": "click",
    "target": "chat input",
    "coords": [642, 718]
  },
  "result": "success",
  "before_ref": "artifact://frames/26_after.png",
  "after_ref": "artifact://frames/27_after.png",
  "video_refs": [
    "artifact://frames/27_00.png",
    "artifact://frames/27_01.png"
  ],
  "screen_summary_before": "Home page with bottom input.",
  "screen_summary_after": "Input focused; cursor visible.",
  "importance": "normal",
  "tags": ["target_found"],
  "verifier": {
    "page_changed": true,
    "task_progress": "advanced"
  }
}
```

### Retention Priority

Keep first:

1. Current frame.
2. Current active/interrupt task state.
3. Recent failure and recovery evidence.
4. Page transitions and first-arrival states.
5. Task completion evidence.
6. Known control coordinate evidence.
7. User constraints and safety records.

Compact first:

1. Repeated waits.
2. No-screen-change action records.
3. Ordinary successful clicks with no lasting semantic value.
4. Superseded task paths after `override`.
5. Long-range video that is unrelated to the active task.
6. Duplicate visual frames that can be represented by one after-frame.

### Context Manager Responsibilities

The future implementation should include a `ContextManager` or equivalent with
these responsibilities:

- estimate token/vision budget;
- choose visual records under the 43-frame default policy;
- downgrade history by recency and importance;
- request model compaction plans only at compact thresholds;
- validate model compaction plans;
- apply deterministic compaction;
- keep artifact references stable;
- keep replay trace independent of prompt context;
- expose enough metadata for benchmark and later training data construction.

### Interview Framing for Compaction

Concise explanation:

> I would not let the model directly rewrite a huge GUI history. The system keeps
> all screenshots, videos, actions, and verifier results in an artifact store.
> The prompt only contains a bounded decision context. Recent history is kept as
> short video frames, mid-history becomes after-frame snapshots, and long history
> becomes structured text with artifact references. When the context approaches a
> threshold, the model only outputs a small compaction plan: which steps can be
> merged, which visual evidence must be preserved, and which failed paths should
> not be repeated. Program logic validates and applies the plan deterministically.
> This keeps compaction cheap, auditable, and replayable.

## Runtime Context Shape

The eventual implementation should build a bounded context object similar to:

```json
{
  "job_board": {
    "active_job_id": "job_qianniu",
    "jobs": [
      {
        "job_id": "job_qianniu",
        "target_app": "QianNiu",
        "goal": "handle customer conversations in this APP",
        "queues": {
          "running": 1,
          "urgent": 0,
          "runnable": 3,
          "waiting_human": 1,
          "blocked": 0
        }
      },
      {
        "job_id": "job_wechat",
        "target_app": "WeChat",
        "goal": "manager communication channel",
        "queues": {
          "running": 0,
          "urgent": 1,
          "runnable": 0,
          "waiting_human": 0,
          "blocked": 0
        }
      }
    ],
    "recent_switches": [
      {
        "from_job_id": "job_qianniu",
        "to_job_id": "job_wechat",
        "reason": "manager reply arrived",
        "safe_point": true
      }
    ]
  },
  "active_subtask": {
    "subtask_id": "reply_buyer_023",
    "job_id": "job_qianniu",
    "status": "running",
    "goal": "reply to buyer message"
  },
  "current_state": {
    "screen_summary": "death dialog is visible",
    "current_frame": "artifact://frames/current.png",
    "visible_targets": [
      {
        "name": "revive button",
        "bbox": [900, 700, 1100, 780]
      }
    ]
  },
  "recent_video_history": [
    {
      "step": 41,
      "action": "click dungeon entrance",
      "result": "entered loading screen",
      "frames": [
        "artifact://frames/41_00.png",
        "artifact://frames/41_01.png"
      ]
    }
  ],
  "mid_history_after_frames": [
    {
      "step": 31,
      "after": "artifact://frames/31_after.png",
      "summary": "quest panel opened"
    }
  ],
  "long_history_summary": {
    "completed": [
      "opened quest panel",
      "accepted main quest"
    ],
    "failed_attempts": [
      "clicking the login button led to an irrelevant QR page"
    ],
    "do_not_repeat": [
      "do not click the login button for this task"
    ],
    "evidence_refs": [
      "artifact://frames/12_after.png"
    ]
  }
}
```

## Trace and Training Data

The same harness records should support later SFT, DPO, PPO, or reward-shaping
experiments. Each step should preserve:

- task stack before action;
- active job and active subtask before action;
- runtime context hash or serialized context;
- current screenshot/frame references;
- model input and output;
- parsed tool call;
- execution result;
- before/after state summary;
- verifier result;
- reward or preference label if available;
- failure classification;
- latency and error metadata.

This is important because reinforcement learning for GUI Agent depends more on
high-quality harness data collection than on the final loss formula.

## Reward Design for Later RL

The harness should record enough information to assign reward at multiple
levels:

- atomic action reward;
- subtask completion reward;
- task transition reward;
- recovery/rollback reward;
- long-horizon task reward.

The model should be allowed to propose a bounded sequence of operations when a
subtask naturally requires several continuous steps. However, every atomic tool
call must still be executed, observed, verified, and recorded separately.

Example:

```text
Subtask: send message
  action 1: click input box
  action 2: type text
  action 3: click send
  terminal state: reply generation started
```

When the terminal state marks the subtask as completed, reward should be
assigned back to the actions inside that subtask with temporal discounting.

Simple discounted return:

```text
G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...
```

For a completed subtask, the completion reward should flow backward through the
subtask's action chain:

```text
subtask completion reward = +R

last action before completion:
  reward contribution = R

previous action:
  reward contribution = gamma * R

two actions earlier:
  reward contribution = gamma^2 * R
```

This is important because the earlier actions may have no immediate positive
reward, but they enabled the final successful state.

### Reward Events

Recommended positive events:

- target UI control found;
- action changed the screen in the expected direction;
- subtask terminal state reached;
- interrupt task completed and original task resumed;
- rollback returned to a stable checkpoint;
- full user task completed;
- model selected a safe and valid tool call;
- repeated failure path was avoided.

Recommended negative events:

- invalid tool schema;
- out-of-bounds coordinates;
- click with no screen change when progress was expected;
- repeated action loop;
- timeout;
- unsafe operation attempt;
- wrong task override;
- rollback failed;
- user task failed;
- action directly caused a bad state, such as death, app crash, or leaving the
  required workflow.

### Transition Rewards

Task transitions need explicit reward accounting.

The final step in task A before switching to B is not always negative. It must
be classified by cause:

- If A's last action caused the bad state, assign negative reward to that A
  action even if B later recovers.
- If A's last action correctly detected or responded to an external state
  change, assign positive reward to that A action because it selected the right
  transition.
- If the state change was neutral or unrelated to A's action, do not blindly
  assign credit or blame; attach reward to the transition decision and B's
  outcome.

Example positive A -> B transition:

```text
A: continue filling a web form
external state: unexpected modal appears
A final decision/action: detect modal and switch to close-modal task B
B: close modal
B completed -> resume A
```

The final A transition step should receive positive reward because switching to
B was the correct behavior. The later success of B should also propagate reward
through B's own actions and the resume transition.

Interrupt transition:

```text
A active -> B interrupt active -> B completed -> A resumed
```

Reward should distinguish two cases:

1. If the transition was a correct response to the environment, reward the
   detection and recovery path.

   Example:

   ```text
   death dialog appears
   push_interrupt(revive)
   revive completed
   resume original task
   ```

   Positive reward should apply to:

   - the step that correctly detected the interrupt condition;
   - the action that selected or entered the interrupt task;
   - the successful actions inside B;
   - the `pop_resume()` transition back to A.

2. If the previous A action caused the bad state, that action may receive a
   penalty even if recovery succeeds.

   Example:

   ```text
   bad combat decision -> character death -> revive succeeds
   ```

   The recovery path can be rewarded, but the action that caused death should
   still carry negative reward.

3. If the interrupt was caused by an external or expected environment change,
   the A -> B transition action can be positive.

   Example:

   ```text
   unexpected dialog appears
   model detects it and starts close_dialog interrupt
   close_dialog succeeds
   ```

   Reward should apply to:

   - the A-side detection/transition decision;
   - B's successful close-dialog actions;
   - the `pop_resume()` transition back to A.

Override transition:

```text
A active -> B active, A superseded
```

Reward should apply to the final tool call or decision that made the valid
override possible when B is truly better or A is no longer valid.

Examples:

- user changed the goal;
- A became impossible;
- a higher-priority task became available;
- the current subtask's success condition is no longer relevant.

Wrong override should be penalized because it destroys continuity.

Rollback transition:

```text
A active -> recovery/rollback -> checkpoint -> replan
```

Reward should apply to the rollback path only if it returns to a verified stable
checkpoint. The failed path before rollback may still carry negative reward.

### Subtask-Level Reward Assignment

Each subtask should record:

- `subtask_id`;
- start step;
- terminal step;
- terminal state;
- success/failure flag;
- completion reward;
- failure penalty;
- discount factor;
- included step ids;
- transition type that started or ended it.

Recommended step-level metadata:

```json
{
  "step_id": 42,
  "task_id": "level_up",
  "subtask_id": "revive",
  "transition": {
    "type": "push_interrupt",
    "from_task": "clear_dungeon",
    "to_task": "revive"
  },
  "tool_call": {
    "type": "click",
    "target": "revive button"
  },
  "verifier": {
    "screen_changed": true,
    "subtask_done": false,
    "task_done": false
  },
  "reward_events": [
    {
      "type": "valid_interrupt_detection",
      "value": 0.3
    }
  ]
}
```

When a subtask completes, the harness can create training labels such as:

```json
{
  "subtask_id": "revive",
  "terminal_step": 45,
  "terminal_reward": 2.0,
  "discounted_credit": [
    {"step": 45, "credit": 2.0},
    {"step": 44, "credit": 1.8},
    {"step": 43, "credit": 1.62},
    {"step": 42, "credit": 1.458}
  ]
}
```

The exact reward constants can be tuned later. The important requirement now is
to record the causal structure.

### Bounded Multi-Action Proposals

For continuity, the model may propose several related actions for a subtask, but
the harness should not blindly execute a long sequence without observation.

Recommended policy:

- allow short action batches only inside an active subtask;
- cap batch length, for example 2-5 atomic actions;
- execute each action one by one;
- observe after each action or after a safe micro-batch;
- stop the batch immediately if verifier sees completion, failure, interrupt,
  or unexpected state transition;
- record reward per atomic action, not only per batch.

This gives the model enough continuity while preserving harness control.

### Training Interpretation

Later algorithms can consume the same records in different ways:

- SFT: learn high-quality successful action traces.
- DPO: compare successful vs failed actions under similar states.
- PPO/RL: use discounted rewards, advantages, and transition rewards from
  harness rollout.

The immediate implementation target is not the loss function. The target is to
make the harness record:

```text
state -> action -> result -> verifier -> task transition -> reward event
```

with enough precision that SFT, DPO, or PPO can be built later.

## Desktop Console and Human Intervention Channel

The GUI_Agent application should eventually provide a desktop console, not only
a raw command-line runner. The console is the user-facing surface for long task
configuration, runtime monitoring, human intervention, and correction feedback.

### Desktop Console Goals

The user should be able to configure long-horizon business context before the
agent runs.

Examples for an e-commerce operations scenario:

- business goal: process seller/buyer messages, inspect orders, handle routine
  after-sales cases, generate daily reports;
- software list: browser, e-commerce backend, chat tool, spreadsheet, WeChat,
  and other operational software;
- software purpose: what each application is used for;
- UI references: screenshots for important pages, buttons, tables, dialogs, and
  workflows;
- business rules: which cases can be auto-replied, which require approval, which
  operations are forbidden;
- safety rules: refund, price change, deletion, payment, account/security
  changes, and other high-risk actions must request human confirmation;
- SOP references: documents, images, or videos describing standard procedures.

The user input surface should support:

- long task text prompt;
- extra constraints;
- reference images;
- reference videos;
- SOP documents;
- manual corrections during runtime.

These inputs should become structured business profile / workspace profile data
instead of remaining as one-time prompt text.

### Runtime Monitor

The desktop UI should clearly show the execution chain:

- current long-horizon task;
- current subtask;
- current task stack;
- model proposed action;
- harness validation result;
- actual executed action;
- verifier result;
- success/failure state;
- current screenshot;
- recent trace entries.

The UI should show model reasoning only as a bounded decision summary, not as an
unbounded hidden chain-of-thought dump. A collapsible area can show:

- decision summary;
- evidence used;
- risk judgment;
- raw model output;
- parsed tool call;
- harness validation details;
- trace metadata.

### Human Intervention Channel

Some cases should not be auto-resolved by the model.

Examples:

- seller/buyer explicitly asks for a human;
- refund, price change, deletion, account, payment, or legal/safety-sensitive
  operation;
- model confidence is low;
- repeated failures or loops;
- unexpected UI state;
- business rule conflict.

The system should support a human-in-the-loop channel. WeChat is the preferred
first design target because it is a practical remote management surface in the
local desktop environment.

Important layering decision:

```text
WeChat remote channel belongs to GUI_Agent application layer.
It does not belong to CFIE inference engine.
It does not belong to cfie_client core computer-use SDK.
```

Recommended module layout after the application layer is renamed:

```text
GUI_Agent/
  human_loop/
    manager.py
    channels/
      base.py
      wechat.py
      email.py
      websocket.py
      local_console.py
```

Current project location:

```text
cfie_gui_agent/
  human_loop/
    manager.py
    channels/
      base.py
      wechat.py
```

Channel abstraction:

```python
class HumanChannel:
    def send_request(self, request): ...
    def poll_replies(self): ...
    def acknowledge(self, reply): ...
```

`HumanLoopManager` responsibilities:

- detect that a task needs human intervention;
- create a human request with task state, screenshot, proposed answer/action,
  risk reason, and allowed reply format;
- send the request through WeChat or another channel;
- mark the original task as `waiting_human`;
- release the main scheduler to process other runnable tasks;
- receive manager replies;
- convert replies into high-priority tasks;
- insert those tasks into `urgent_queue`;
- resume or override the original task after the human-provided instruction is
  executed.

### Waiting Human Does Not Stop Automation

When a task waits for human input, the whole automation flow must not terminate.
The scheduler should park the blocked task and continue with other safe tasks.

Recommended queues:

```text
urgent_queue:
  manager replies and high-priority interventions

active_queue:
  runnable automation tasks

waiting_human_queue:
  tasks waiting for manager input

blocked_queue:
  tasks blocked by environment or repeated failure

finished_queue:
  completed/cancelled/failed tasks
```

Example:

```text
Task A: answer seller question
seller asks: "do not use AI, ask a human"

HumanLoopManager:
  send screenshot + question to manager via WeChat
  mark A as waiting_human
  scheduler continues with task C: inspect orders

Manager reply:
  "Reply to seller: xxxx"

Scheduler:
  creates high-priority ManagerReplyTask B
  inserts B into urgent_queue
  waits for a safe interruption point
  executes B
  marks A resolved or resumes related workflow
```

Safe interruption points:

- after one atomic action completes;
- after a subtask completes;
- after page loading finishes;
- before a high-risk action;
- before starting a long typed response;
- never in the middle of drag, partial input, payment confirmation, or other
  unsafe half-complete actions.

### User Correction and Constraint Update

The desktop console should allow the user to pause automation and correct the
agent.

Example user correction:

```text
Do not click refund directly. Refund requests must ask me first.
```

The system should convert this into structured policy, not just append it to the
chat history:

```json
{
  "rule": "refund actions require human approval",
  "scope": "ecommerce_after_sales",
  "priority": "high",
  "source": "user_correction",
  "created_from_step": 128
}
```

The policy should update:

- business profile;
- safety policy;
- failure memory;
- prompt constraints;
- harness validation rules if possible.

This makes user correction durable and enforceable by program logic.

### Console Architecture

High-level architecture:

```text
Desktop Console
  -> Task / Business Profile Editor
  -> Agent Runtime Monitor
  -> Human Intervention Inbox
  -> Trace / Replay Viewer
  -> Policy & Constraint Manager

GUI_Agent Harness
  -> Task Scheduler
  -> Context Manager
  -> HumanLoopManager
  -> Tool Executor
  -> Verifier / Judge
  -> Trace Store

cfie_client
  -> screenshot / click / type / keyboard / window / future ADB

CFIE
  -> local Qwen3.5 model brain
```

Interview framing:

> The desktop program should be a configurable, monitorable, interruptible Agent
> console. Users define long-horizon business knowledge and safety policies.
> The agent handles low-risk work automatically. When it encounters uncertain or
> high-risk cases, it parks the task, asks the manager through a human channel
> such as WeChat, continues other safe work, and later inserts the manager's
> reply as a high-priority task. The harness still validates and executes the
> operation at a safe interruption point.

## Model Tool Registry Boundary

The GUI_Agent should not expose every low-level capability directly to the
model. Tools should be split into:

```text
model-callable tools:
  small set of high-level semantic tools exposed to the model

harness-internal tools:
  deterministic implementation details controlled by program logic
```

This boundary prevents the model from bypassing context policy, task scheduling,
human-loop rules, safety checks, or trace/reward accounting.

### Model-Callable Tools

Recommended initial tool set:

1. `computer_use`

   Core desktop operation surface:

   ```text
   screenshot
   click
   double_click
   type
   press
   hotkey
   scroll
   drag
   wait
   ```

   The harness still validates coordinates, focus state, risk, and task state
   before execution.

2. `read_image`

   Read a referenced image artifact into model context.

   Inputs:

   ```text
   artifact_id or file_path
   purpose
   ```

   Use cases:

   - inspect a historical screenshot;
   - compare current state with a reference UI image;
   - inspect a user-provided SOP image.

3. `read_video_clip`

   Read a bounded short video clip or selected frame set into model context.

   Inputs:

   ```text
   artifact_id
   time_range or step_range
   frame_policy
   purpose
   ```

   The model should not directly control low-level ffmpeg parameters. It should
   request a semantic clip, and the harness chooses frames according to context
   budget.

4. `request_human_help`

   Request manager intervention through the configured human channel.

   Inputs:

   ```text
   question
   current_task
   screenshot/evidence refs
   risk reason
   proposed answer/action if any
   allowed reply format
   urgency
   ```

   The underlying channel can be WeChat, local console, WebSocket, email, or
   another backend. The model should not know or control the low-level channel
   details.

5. `update_constraints`

   Convert user correction or manager instruction into structured policy
   proposal.

   Inputs:

   ```text
   user correction text
   affected task/domain
   evidence step
   ```

   Output should be a structured rule proposal. The harness must review and
   install it into policy storage before it becomes enforceable.

6. `finish_subtask`

   Mark that the current subtask appears complete.

   Inputs:

   ```text
   subtask_id
   completion reason
   evidence refs
   next-step suggestion
   ```

   The verifier and TaskManager decide whether the completion claim is accepted.

7. `report_blocked`

   Report that the model cannot safely continue the current task.

   Inputs:

   ```text
   blocked reason
   attempted actions
   requested support: human / rollback / replan
   evidence refs
   ```

8. `ask_replan`

   Ask the TaskManager to consider `interrupt`, `override`, `rollback`, or a new
   subtask.

   Inputs:

   ```text
   reason
   suggested transition
   target subtask if known
   evidence refs
   ```

   The model only proposes. TaskManager validates and applies.

9. `query_memory`

   Query workspace/business memory.

   Inputs:

   ```text
   query
   current task
   desired memory type: SOP / known target / failure path / policy / UI ref
   ```

   Outputs may include relevant SOP snippets, known UI targets, historical
   failures, and artifact references.

`check_human_reply` is intentionally not part of the recommended model-callable
set. Scheduler or `HumanLoopManager` should poll replies internally and create a
high-priority task when a reply arrives.

### Harness-Internal Tools

These capabilities should remain program-controlled:

- video decoding and frame extraction;
- image resizing/compression;
- OCR;
- frame budget selection;
- context compaction execution;
- model compaction plan validation;
- trace writing;
- reward calculation;
- task queue scheduling;
- WeChat low-level send/poll implementation;
- coordinate bounds validation;
- safety policy validation;
- screen-change detection;
- repeated-action detection;
- window handle and focus management;
- artifact persistence;
- prompt/context serialization;
- tool execution retry policy.

The model may request high-level operations such as "read recent video evidence"
or "ask human for help", but the harness must choose the concrete implementation
and enforce all constraints.

### Tool Boundary Interview Framing

> I would expose only a small, semantic tool set to the model: computer-use,
> image/video evidence reading, human-help request, constraint update,
> subtask-finish/block reporting, replan request, and memory query. Low-level
> video extraction, context compaction, trace persistence, reward calculation,
> WeChat polling, queue scheduling, coordinate validation, and safety checks
> stay inside the harness. This keeps the model useful for perception and
> planning without allowing it to bypass deterministic program constraints.

## Implementation Round 1

Date: 2026-05-23

Scope:

- implement pure-Python architecture skeleton only;
- no real desktop execution changes;
- no WeChat transport implementation yet;
- no model/API call changes yet.

Added modules:

- `cfie_gui_agent.task`
  - `TaskState`
  - `TaskTransition`
  - `TaskStack`
  - task statuses: active, paused, completed, failed, cancelled, superseded
  - transition operations: `push_interrupt`, `pop_resume`, `override`,
    `rollback`, `complete_active`, `fail_active`, `cancel_active`
- `cfie_gui_agent.context`
  - `VisionContextPolicy` with default 43-frame 1080p policy
  - `StepRecord`
  - `LongHistorySummary`
  - `ContextManager`
  - context usage classes: normal, warning, compact, emergency
  - compaction-plan validation skeleton
- `cfie_gui_agent.tools`
  - model-callable tool allowlist
  - harness-internal tool deny boundary
  - `ModelToolRegistry`
- `cfie_gui_agent.human_loop`
  - `HumanRequest`
  - `HumanReply`
  - `HumanChannel` protocol
  - `InMemoryHumanChannel`
  - `HumanLoopManager`
  - manager replies become urgent tasks
- `cfie_gui_agent.rewards`
  - `RewardEvent`
  - `SubtaskRewardAssignment`
  - discounted subtask completion credit assignment
  - transition reward event helper

Added tests:

- `tests/unit/test_gui_agent_architecture.py`
  - interrupt -> resume behavior;
  - override supersedes the old task;
  - 43-frame context policy selection;
  - compaction-plan step validation;
  - model tool registry blocks harness-internal tools;
  - human reply becomes urgent task;
  - subtask completion reward discounting.

Verification:

```text
python -m pytest tests/unit/test_gui_agent_architecture.py tests/unit/test_cfie_client_gui_agent.py -q
10 passed

python -m py_compile cfie_gui_agent/task.py cfie_gui_agent/context.py \
  cfie_gui_agent/tools.py cfie_gui_agent/human_loop.py \
  cfie_gui_agent/rewards.py cfie_gui_agent/__init__.py
passed
```

Current limitation:

- `ContextManager` only selects and validates context records; it does not yet
  mutate/persist compacted context.
- `HumanLoopManager` only provides channel abstraction and urgent-task creation;
  WeChat backend is not implemented.
- `ModelToolRegistry` uses placeholder parameter schemas; full OpenAI tool
  schema should be filled when the API loop is integrated.
- Reward logic records discounted labels but is not connected to a real harness
  rollout trace yet.

## Implementation Round 2

Date: 2026-05-23

Naming update:

- renamed the application-layer package from `cfie_gui_testing` to
  `cfie_gui_agent`;
- renamed public task runner API:
  - `GuiTestCase` -> `GuiAgentTaskSpec`
  - `GuiTestRunner` -> `GuiAgentRunner`
  - `GuiTestResult` -> `GuiAgentResult`
- renamed the generic task spec file from `cases.py` to `specs.py`;
- updated tests and docs to avoid the narrow "testing-only" module framing.

Runner integration:

- `GuiAgentRunner` now creates a root `TaskStack` for each run;
- each executed `computer_call` is validated through `ModelToolRegistry`;
- each executed `computer_call` creates a `StepRecord`;
- result metadata now includes:
  - task stack state;
  - task transitions;
  - step summaries;
  - selected prompt-context payload;
  - model-callable tool names.

Verification:

```text
python -m pytest tests/unit/test_gui_agent_architecture.py tests/unit/test_cfie_client_gui_agent.py -q
10 passed

python -m py_compile cfie_gui_agent/specs.py cfie_gui_agent/runner.py \
  cfie_gui_agent/task.py cfie_gui_agent/context.py cfie_gui_agent/tools.py \
  cfie_gui_agent/human_loop.py cfie_gui_agent/rewards.py \
  cfie_gui_agent/__init__.py
passed
```

## Implementation Round 3

Date: 2026-05-23

Scheduler skeleton:

- added `cfie_gui_agent.scheduler`;
- added `QueuedTask`;
- added `ScheduledDecision`;
- added `AgentScheduler`;
- queues:
  - `urgent_queue`;
  - `active_queue`;
  - `waiting_human_queue`;
  - `blocked_queue`;
  - `finished_queue`;
- manager replies can be converted into urgent tasks;
- urgent tasks execute only when `safe_to_interrupt=True`;
- tasks waiting for human input do not block other active work.

Added tests:

- waiting-human tasks do not block active automation tasks;
- manager replies wait until a safe interruption point before execution.

Verification:

```text
python -m pytest tests/unit/test_gui_agent_architecture.py tests/unit/test_cfie_client_gui_agent.py -q
15 passed

python -m py_compile cfie_gui_agent/specs.py cfie_gui_agent/runner.py \
  cfie_gui_agent/jobs.py cfie_gui_agent/scheduler.py \
  cfie_gui_agent/task.py cfie_gui_agent/context.py cfie_gui_agent/tools.py \
  cfie_gui_agent/human_loop.py cfie_gui_agent/rewards.py \
  cfie_gui_agent/__init__.py
passed
```

## Implementation Round 4

Date: 2026-05-23

One-app-one-job refactor:

- added `cfie_gui_agent.jobs`;
- added `JobState`;
- added `SubtaskState`;
- added `SubtaskQueues`;
- added `JobBoard`;
- added `SwitchEvent`;
- added `PerJobContextStore`;
- `TaskStack` remains as a compatibility/helper structure, but the main runner
  metadata now follows Job/Subtask semantics.

Runner changes:

- `GuiAgentRunner` now creates one initial Job for the task target app/workspace;
- root work is represented as a running Subtask under that Job;
- each executed `computer_call` produces a `StepRecord` under the active Job;
- result metadata now exposes:
  - `job_board`;
  - per-Job details and Subtask queues;
  - `active_job_id`;
  - Job-specific step records;
  - selected prompt context.

Data model decision:

```text
Job = APP / window / logical workspace context container
Subtask = work item under a Job
ComputerAction = atomic tool operation
```

Added tests:

- JobBoard selects urgent work across Jobs when safe to interrupt;
- JobBoard keeps the current running Subtask when not safe to interrupt;
- PerJobContextStore keeps different Job histories separate;
- runner result metadata uses JobBoard/Subtask queues.

Verification:

```text
python -m pytest tests/unit/test_gui_agent_architecture.py tests/unit/test_cfie_client_gui_agent.py -q
15 passed

python -m py_compile cfie_gui_agent/jobs.py cfie_gui_agent/specs.py \
  cfie_gui_agent/runner.py cfie_gui_agent/scheduler.py \
  cfie_gui_agent/task.py cfie_gui_agent/context.py cfie_gui_agent/tools.py \
  cfie_gui_agent/human_loop.py cfie_gui_agent/rewards.py \
  cfie_gui_agent/__init__.py
passed
```

## Implementation Round 5

Date: 2026-05-23

Job context compaction:

- `ContextManager.apply_compaction_plan()` now applies a validated model
  compaction plan;
- old step videos can be dropped from prompt context;
- step ranges can be merged into long-history text summary;
- evidence refs are preserved;
- `do_not_repeat` and `known_targets` are merged into `LongHistorySummary`;
- current active step cannot be compacted unless explicitly allowed.

Per-Job context store:

- `PerJobContextStore` now keeps both active prompt steps and raw step history;
- compaction only mutates the active prompt history for one Job;
- raw history remains available as the future artifact-store source of truth;
- `rebuild_prompt_context()` rebuilds the current Job prompt context from stored
  steps plus long summary;
- `to_job_context_payload()` exposes active step count, raw step count, selected
  prompt context, and compaction history.

Added tests:

- applying a compaction plan removes merged steps from active prompt history,
  drops video refs, and preserves text/evidence;
- compacting one Job does not touch another Job's context;
- raw step history remains available after compaction.

Verification:

```text
python -m pytest tests/unit/test_gui_agent_architecture.py tests/unit/test_cfie_client_gui_agent.py -q
17 passed

python -m py_compile cfie_gui_agent/jobs.py cfie_gui_agent/context.py \
  cfie_gui_agent/runner.py cfie_gui_agent/__init__.py
passed
```

## Updated Job/Subtask Constraint

Date: 2026-05-23

The first `TaskStack` implementation captured interrupt/resume semantics, but
the formal GUI_Agent scheduling model should be simpler and more stable:

```text
one APP / work window / logical workspace = one Job
Job owns Subtask queues
Subtask owns actual computer actions
```

`TaskStack` should no longer be treated as the main scheduling structure. It can
remain as a compatibility/helper representation for local interrupt semantics,
but the main path should move toward:

- `JobState`
- `SubtaskState`
- `SubtaskQueues`
- `JobBoard`
- `SwitchHistory`
- `PerJobContextStore`

### One-App-One-Job

A Job is an application/window/workspace context container, not a fine-grained
business operation.

Examples:

```text
job_wechat
job_qianniu
job_doudian_browser
job_pdd_browser
job_excel_orders
job_android_phone_window
```

The term APP is logical. It can mean:

- an OS process;
- a browser tab;
- a desktop window;
- a remote phone window;
- an RDP/VNC desktop;
- any long-lived workspace with its own UI and history.

### Job Context vs Global Context

Each Job owns its own detailed context:

- current APP/window/tab identity;
- subtask queues;
- recent video frames;
- mid-history after frames;
- long-history summary;
- failure memory;
- known controls;
- SOP references;
- current page/screen state;
- artifact refs;
- context compaction state.

Global context only contains scheduling-level summaries:

- active job;
- all job state summaries;
- each job's queue counts;
- urgent/waiting_human summaries;
- recent job switch history;
- safe interruption state;
- monitor enabled/disabled;
- target apps.

The model should receive:

```text
current Job detailed context
+
global JobBoard summary
+
recent Job switch history
```

The model should not receive full visual histories for every Job at once.

### Switching Jobs

When switching from Job A to Job B:

1. Save Job A context:
   - current subtask;
   - subtask queues;
   - recent steps;
   - current screenshot;
   - compaction state;
   - artifact refs.
2. Append a `SwitchEvent`:
   - from_job;
   - to_job;
   - reason;
   - safe interruption point;
   - interrupted subtask state;
   - result if known.
3. Activate Job B:
   - focus/open the target APP/window when possible;
   - read Job B's persisted step/artifact history;
   - rebuild Job B prompt context with the 43-frame policy;
   - attach global JobBoard summary;
   - run the next model prefill using the new Job-specific context.

This is why the GUI_Agent service baseline must keep `--enable-prefix-caching`:
system/developer prompt, tool schema, general scheduling rules, and safety policy
can be shared across jobs, while Job-specific visual/history context changes.

### Monitor Job

If there is only one target APP/workspace, no separate monitor Job is required.
The active Job can observe its own UI.

If there are multiple target APPs/workspaces, enable a logical monitor Job:

```text
job_monitor
  -> inspect desktop / taskbar / notifications / target windows
  -> identify which APP has a new event
  -> classify event with model assistance
  -> create Subtask under the correct Job
```

Model responsibilities:

- identify which APP/window has a new event;
- infer whether a notification/message belongs to an existing Job;
- propose subtask type and priority;
- suggest whether a Job switch is useful.

Harness responsibilities:

- validate that the target Job exists;
- validate allowed subtask types;
- create `SubtaskState`;
- enqueue into that Job's subtask queues;
- enforce safe interruption;
- persist switch history.

### JobBoard Scheduling

Job itself should stay lightweight. The scheduler should choose a Job by looking
at its subtask queues.

Recommended priority:

```text
1. current running subtask if not safe to interrupt
2. urgent subtasks across all Jobs if safe to interrupt
3. runnable subtasks across Jobs by Job priority and subtask priority
4. monitor scan if monitor is due
5. idle / poll human replies / recovery
```

Important rule:

```text
Job runnable = Job is enabled and has urgent/runnable Subtasks.
```

Waiting-human subtasks must not block the whole Job when other runnable subtasks
exist under the same Job.

## Engine Serves the Application

CFIE engine validation should be driven by the GUI_Agent workload. The engine is
not an isolated benchmark target; it is the local model brain for application
automation.

Application-driven engine validation:

```text
GUI_Agent real task
  -> observe latency / output / tool-call / context issue
  -> classify as application, harness, prompt, API, template, or engine issue
  -> fix the responsible layer
  -> rerun end-to-end task
```

Examples:

- If thinking output is too long and hurts GUI responsiveness, tune the Qwen
  thinking template, thinking mode mapping, output limit, or GUI_Agent decision
  summary policy.
- If tool calls are unstable, test Responses/Chat tool schema mapping and
  harness validation.
- If multi-frame input is too slow, inspect frame budget, vision token count,
  prefix cache, and prefill settings.
- If long-context tasks stall, test KV allocation, prefill chunking, and
  context compaction policy.

Important default CLI requirement:

```text
GUI_Agent service baseline must explicitly enable prefix cache.
```

The recommended server command/configuration must include:

```text
--enable-prefix-caching
```

This should not be left implicit. GUI_Agent workloads repeatedly reuse long
system/developer prompts, tool schemas, business profiles, task policies, and
visual-context structure. Prefix cache is therefore part of the default
application-serving contract.

Engine metrics should include both microbenchmarks and application metrics:

Engine microbench:

- decode TPS;
- TTFT;
- prefill latency;
- KV capacity;
- vision input token cost;
- W4A8 correctness and speed.

Application E2E:

- task success rate;
- first action latency;
- tool-call latency;
- model output length;
- thinking output length;
- visual context build time;
- context compaction cost;
- human intervention round-trip time;
- end-to-end task duration;
- failure category.

## Implementation Round 6

Date: 2026-05-23

Model tool schema boundary:

- added concrete JSON-style parameter schemas for all model-callable tools:
  - `computer_use`;
  - `read_image`;
  - `read_video_clip`;
  - `request_human_help`;
  - `update_constraints`;
  - `finish_subtask`;
  - `report_blocked`;
  - `ask_replan`;
  - `query_memory`.
- added `ModelToolRegistry.validate_model_tool_call()`;
- the registry now validates:
  - tool name allowlist;
  - harness-internal tool denial;
  - required fields;
  - basic JSON types;
  - enum values;
  - array item types;
  - minimum string length / minimum numeric values where needed.

Runner integration:

- high-level model tools are parsed through `find_agent_tool_calls()`;
- `GuiAgentRunner` now validates high-level tool arguments before mutating
  Subtask state;
- `finish_subtask`, `report_blocked`, `request_human_help`, `ask_replan`,
  `update_constraints`, `read_image`, `read_video_clip`, and `query_memory`
  return structured tool outputs;
- `StepRecord` preserves the affected Subtask id even if the tool call moves
  the running Subtask into another queue.

Added tests:

- JSON argument parsing for model tool calls;
- `computer_use` function-call items are not treated as generic Agent tools;
- tool schema export contains non-empty OpenAI-compatible function schemas;
- missing required fields and invalid enum values are rejected;
- runner rejects invalid `request_human_help` arguments.

Verification:

```text
..\.venv\Scripts\python.exe -m pytest tests\unit\test_gui_agent_architecture.py tests\unit\test_cfie_client_gui_agent.py -q
23 passed

..\.venv\Scripts\python.exe -m py_compile cfie_gui_agent\agent_tools.py cfie_gui_agent\runner.py cfie_gui_agent\jobs.py cfie_gui_agent\context.py cfie_gui_agent\tools.py cfie_gui_agent\__init__.py
passed
```

## Implementation Round 7

Date: 2026-05-23

Execution verifier skeleton:

- added `cfie_gui_agent.verifier`;
- added `StepVerifier`;
- added `StepVerification`;
- verification statuses:
  - `ok`;
  - `no_screen_change`;
  - `repeated_action`.

Runner integration:

- `GuiAgentRunner` now stores the initial screenshot reference as the first
  `before_ref`;
- every `computer_call` StepRecord now records:
  - `before_ref`;
  - `after_ref`;
  - verifier metadata;
  - verifier status tag when the step is suspicious.

Purpose:

- detect no-op clicks / unchanged screens;
- detect repeated action loops;
- provide structured labels for future reward/preference data;
- expose verifier data to the desktop console and trace layer later.

Added tests:

- no-screen-change detection;
- repeated-action detection;
- runner metadata contains verifier output.

Verification:

```text
..\.venv\Scripts\python.exe -m pytest tests\unit\test_gui_agent_architecture.py tests\unit\test_cfie_client_gui_agent.py -q
24 passed

..\.venv\Scripts\python.exe -m py_compile cfie_gui_agent\agent_tools.py cfie_gui_agent\runner.py cfie_gui_agent\jobs.py cfie_gui_agent\context.py cfie_gui_agent\tools.py cfie_gui_agent\verifier.py cfie_gui_agent\__init__.py
passed
```

## Implementation Round 8

Date: 2026-05-23

Human reply -> JobBoard queue integration:

- `SubtaskQueues.promote_waiting_human()` now accepts metadata when moving a
  waiting-human Subtask back into the urgent queue;
- `JobBoard.enqueue_manager_reply()` converts a `HumanLoopManager` reply payload
  into urgent work under the correct Job;
- if the original waiting-human Subtask still exists, it is promoted and
  enriched with:
  - `manager_reply`;
  - `human_request`;
  - `source=human_loop`;
- if the original Subtask is unavailable, the JobBoard creates a fallback
  `manager_reply` Subtask.

Context budget downgrade:

- `VisionContextPolicy.downgraded()` now implements the warning / compact /
  emergency visual-budget reduction path;
- `ContextManager.policy_for_usage()` maps usage ratio to the downgraded policy;
- `select_prompt_context()` can now receive `usage_ratio`;
- `PerJobContextStore` forwards usage ratio when rebuilding Job prompt context.

Current downgrade behavior:

```text
normal:
  43-frame default

warning:
  mid-history after frames <= 15

compact:
  recent video frames per step <= 4
  mid-history after frames <= 15
  key evidence frames <= 3

emergency:
  recent video steps <= 1
  recent video frames per step <= 1
  mid-history after frames = 0
  key evidence frames <= 2
```

Added tests:

- manager reply promotes a waiting-human Subtask into urgent work;
- context manager downgrades visual frame policy by usage ratio.

Verification:

```text
..\.venv\Scripts\python.exe -m pytest tests\unit\test_gui_agent_architecture.py tests\unit\test_cfie_client_gui_agent.py -q
26 passed

..\.venv\Scripts\python.exe -m py_compile cfie_gui_agent\agent_tools.py cfie_gui_agent\runner.py cfie_gui_agent\jobs.py cfie_gui_agent\context.py cfie_gui_agent\tools.py cfie_gui_agent\verifier.py cfie_gui_agent\human_loop.py cfie_gui_agent\__init__.py
passed
```

## Implementation Round 9

Date: 2026-05-23

Structured policy store:

- added `cfie_gui_agent.policy`;
- added `PolicyRule`;
- added `PolicyUpdate`;
- added `PolicyStore`;
- `PolicyStore.apply_update()` converts user/model corrections into structured
  rules with:
  - rule id;
  - text;
  - scope;
  - source;
  - severity;
  - evidence refs;
  - metadata.

Runner integration:

- `GuiAgentRunner` now owns a `PolicyStore`;
- `update_constraints` is no longer only a proposal in the minimal runner;
- after schema validation, `update_constraints` writes a policy update and
  returns:
  - `status=accepted`;
  - `update_id`;
  - `rules_added`;
  - serialized `policy_update`;
- result metadata now includes `policy`, so later prompt-context builders can
  inject active user/business constraints.

Purpose:

- user correction can persist as structured policy;
- future desktop console can show active constraints;
- prompt generation can include only the compact rules instead of long free-form
  correction history;
- reward / trace data can link bad actions to later corrective policies.

Added tests:

- constraints are converted into context rules;
- runner persists `update_constraints` output into result metadata.

Verification:

```text
..\.venv\Scripts\python.exe -m pytest tests\unit\test_gui_agent_architecture.py tests\unit\test_cfie_client_gui_agent.py -q
28 passed

..\.venv\Scripts\python.exe -m py_compile cfie_gui_agent\agent_tools.py cfie_gui_agent\runner.py cfie_gui_agent\jobs.py cfie_gui_agent\context.py cfie_gui_agent\tools.py cfie_gui_agent\verifier.py cfie_gui_agent\human_loop.py cfie_gui_agent\policy.py cfie_gui_agent\__init__.py
passed
```

## Implementation Round 10

Date: 2026-05-23

Application-layer trace:

- added `cfie_gui_agent.trace`;
- added `AgentTraceEvent`;
- added `AgentTraceStore`;
- trace store supports:
  - in-memory events;
  - optional JSONL persistence;
  - `record_step()`;
  - `record_policy_update()`;
  - `record_result()`.

Runner integration:

- each `computer_call` StepRecord is recorded into `AgentTraceStore`;
- each high-level Agent tool StepRecord is recorded into `AgentTraceStore`;
- `update_constraints` additionally records a `policy_update` event;
- result metadata now includes trace summary:
  - trace file path;
  - event count;
  - recent events.

Boundary:

- `cfie_client.TraceStore` remains the low-level OpenTelemetry-style computer
  tool trace;
- `cfie_gui_agent.AgentTraceStore` records application semantics:
  Job/Subtask, verifier, policy, and high-level tool activity.

Added tests:

- trace store writes JSONL events;
- runner result metadata exposes trace event count;
- policy update tool creates both step and policy trace events.

Verification:

```text
..\.venv\Scripts\python.exe -m pytest tests\unit\test_gui_agent_architecture.py tests\unit\test_cfie_client_gui_agent.py -q
29 passed

..\.venv\Scripts\python.exe -m py_compile cfie_gui_agent\agent_tools.py cfie_gui_agent\runner.py cfie_gui_agent\jobs.py cfie_gui_agent\context.py cfie_gui_agent\tools.py cfie_gui_agent\verifier.py cfie_gui_agent\human_loop.py cfie_gui_agent\policy.py cfie_gui_agent\trace.py cfie_gui_agent\__init__.py
passed
```

## Implementation Round 11

Date: 2026-05-23

Runtime context builder:

- added `cfie_gui_agent.runtime_context`;
- added `RuntimeContext`;
- added `RuntimeContextBuilder`;
- the builder combines:
  - global `JobBoard` summary;
  - active Job summary;
  - active Subtask;
  - per-Job prompt context;
  - structured policy rules;
  - model-callable tool names.

Runner integration:

- `GuiAgentRunner._build_result_metadata()` now uses the runtime context builder;
- result metadata includes `runtime_context` as the future model-facing payload;
- legacy metadata keys (`prompt_context`, `job_context`, `model_tools`, `policy`)
  are still exposed for existing tests and debugging.

Purpose:

- keep prompt/runtime context construction out of ad hoc runner code;
- make the model explicitly aware of current Job/Subtask state;
- preserve one-app-one-job separation when switching Jobs later;
- provide one stable object for Responses/Chat integration.

Added tests:

- runtime context exposes active Job;
- runtime context exposes active Subtask;
- runtime context includes current-frame prompt context;
- runtime context includes structured policy rules.

Verification:

```text
..\.venv\Scripts\python.exe -m pytest tests\unit\test_gui_agent_architecture.py tests\unit\test_cfie_client_gui_agent.py -q
30 passed

..\.venv\Scripts\python.exe -m py_compile cfie_gui_agent\agent_tools.py cfie_gui_agent\runner.py cfie_gui_agent\jobs.py cfie_gui_agent\context.py cfie_gui_agent\tools.py cfie_gui_agent\verifier.py cfie_gui_agent\human_loop.py cfie_gui_agent\policy.py cfie_gui_agent\trace.py cfie_gui_agent\runtime_context.py cfie_gui_agent\__init__.py
passed
```

## Implementation Round 12

Date: 2026-05-23

Runtime context in model input:

- `GuiAgentRunner` now injects a developer message containing runtime context
  JSON;
- the runtime context message is refreshed before every model call;
- the message is stored at the beginning of the conversation so the latest
  `computer_call_output` / `function_call_output` remains the final item;
- this preserves tool-call continuity while still giving the model current
  Job/Subtask/Policy/PromptContext state.

Input shape:

```text
developer:
  Runtime context JSON

user:
  task instruction
  current screenshot

tool outputs:
  appended after execution
```

Reason:

- the model must not infer task queues from stale text;
- the harness-owned state is serialized every turn;
- future Job switching can update only the runtime context message while
  keeping shared system/developer instructions prefix-cache friendly.

Verification:

```text
..\.venv\Scripts\python.exe -m pytest tests\unit\test_responses_video_input.py tests\unit\test_gui_agent_architecture.py tests\unit\test_cfie_client_gui_agent.py -q
33 passed, 2 dependency warnings

..\.venv\Scripts\python.exe -m py_compile cfie_gui_agent\runner.py cfie_gui_agent\runtime_context.py cfie_gui_agent\__init__.py
passed
```

## Implementation Round 13

Date: 2026-05-23

Workspace profile:

- added `WorkspaceProfile`;
- `GuiAgentTaskSpec` can now carry a structured workspace/business profile;
- profile fields:
  - `profile_id`;
  - `name`;
  - `description`;
  - `target_apps`;
  - `business_rules`;
  - `reference_image_refs`;
  - `reference_video_refs`;
  - `sop_refs`;
  - `metadata`.

Runner integration:

- initial Job target app can be inferred from `workspace_profile.target_apps[0]`
  when `target_app` is not explicitly set;
- Job metadata and root Subtask metadata include the serialized workspace
  profile.

Purpose:

- desktop UI can let the user configure long-horizon business context once;
- model prompt/runtime context can reference compact SOP/artifact ids instead
  of large free-form text;
- multi-APP workflows can declare target workspaces before monitor scheduling is
  enabled.

Verification:

```text
..\.venv\Scripts\python.exe -m pytest tests\unit\test_responses_video_input.py tests\unit\test_gui_agent_architecture.py tests\unit\test_cfie_client_gui_agent.py -q
34 passed, 2 dependency warnings

..\.venv\Scripts\python.exe -m py_compile cfie_gui_agent\specs.py cfie_gui_agent\runner.py cfie_gui_agent\__init__.py
passed
```

## Implementation Round 14

Date: 2026-05-23

Multi-APP JobBoard initialization:

- when `WorkspaceProfile.target_apps` contains multiple APP/workspace names,
  `GuiAgentRunner` now creates one Job per APP;
- if `target_app` is explicitly provided and missing from the profile, it is
  inserted as the first Job;
- if multiple APPs exist, the runner also creates `job:monitor`;
- single-APP tasks do not create a monitor Job.

Behavior:

```text
WorkspaceProfile.target_apps = ("QianNiu", "WeChat")

JobBoard:
  job:QianNiu
  job:WeChat
  job:monitor
```

The root Subtask is attached to the active target Job. Other Jobs start with
empty queues and can receive Subtasks later from monitor classification, manager
replies, or user scheduling.

Added test:

- runner creates one Job per target APP plus monitor Job for multi-APP
  workspace profiles.

Verification:

```text
..\.venv\Scripts\python.exe -m pytest tests\unit\test_responses_video_input.py tests\unit\test_gui_agent_architecture.py tests\unit\test_cfie_client_gui_agent.py -q
35 passed, 2 dependency warnings

..\.venv\Scripts\python.exe -m py_compile cfie_gui_agent\runner.py cfie_gui_agent\specs.py cfie_gui_agent\__init__.py
passed
```

## Implementation Round 15

Date: 2026-05-23

Monitor event ingestion:

- added `cfie_gui_agent.monitor`;
- added `MonitorEvent`;
- added `MonitorIngestionResult`;
- added `MonitorController`.

Behavior:

- monitor/classifier output is represented as a `MonitorEvent`;
- `MonitorController` validates that the target Job exists;
- empty Subtask goals are rejected;
- accepted monitor events create a runnable Subtask under the target Job;
- monitor event evidence refs are attached to the Subtask.

Purpose:

- multi-APP notification detection stays harness-mediated;
- the model/classifier may propose an event, but program logic validates the
  target Job and Subtask creation;
- monitor work does not become a free-form prompt-only convention.

Added tests:

- monitor event creates a Subtask under the target Job;
- unknown target Job is rejected.

Verification:

```text
..\.venv\Scripts\python.exe -m pytest tests\unit\test_responses_video_input.py tests\unit\test_gui_agent_architecture.py tests\unit\test_cfie_client_gui_agent.py -q
37 passed, 2 dependency warnings

..\.venv\Scripts\python.exe -m py_compile cfie_gui_agent\monitor.py cfie_gui_agent\__init__.py
passed
```

## Implementation Round 16

Date: 2026-05-23

Low-latency mode and harness-owned continuous movement:

The GUI Agent needs two execution styles:

- normal business automation:
  - lower operation frequency;
  - richer recent visual history;
  - more careful reasoning allowed.
- game / low-latency automation:
  - frequent actions;
  - minimal thinking;
  - no long video history in every turn;
  - model should output compact semantic intents.

Added action macro layer:

- added `cfie_gui_agent.macros`;
- added `ActionMacroStep`;
- added `ActionMacro`;
- added `ActionMacroRegistry`;
- model-callable tool `run_action_macro`;
- action macros expand into validated `ComputerAction` sequences.

Example:

```text
combo_asd:
  keypress A
  keypress S
  keypress D

select_all_then_b:
  keypress CTRL+A
  keypress B
```

This allows one model tool call to trigger a known sequence of human-like key
operations without requiring the model to emit ten individual actions.

Added agility context mode:

- `VisionContextPolicy.agility()`;
- no recent video window;
- all history frames are after-action images until compaction is needed.

Default agility visual budget:

```text
current frame: 1
recent video steps: 0
frames per recent step: 0
mid-history after frames: 42
key evidence frames: 0
total: 43 frames
```

Added navigation planning layer:

- added `cfie_gui_agent.navigation`;
- added `Point`;
- added `ObstaclePolygon`;
- added `NavigationRequest`;
- added `NavigationPlan`;
- added `NavigationPlanner`;
- model-callable tool `navigate_to_target`.

The model provides:

```json
{
  "source": [10, 20],
  "target": [400, 300],
  "target_label": "monster",
  "obstacles": [
    [[100, 100], [180, 100], [180, 200], [100, 200]]
  ]
}
```

The harness owns:

- path planning;
- real-time target/source tracking;
- obstacle avoidance;
- retry;
- stop condition;
- future OpenCV integration.

Current implementation is a deterministic geometry skeleton with bounding-box
detours. It intentionally does not hard-code OpenCV into the runner yet. The
OpenCV/live-screen execution loop should be added behind this interface.

Runtime context:

- registered action macros are included in `RuntimeContextBuilder` output;
- the model can see available macro names and descriptions, but cannot invent a
  macro that the registry has not accepted.

Added tests:

- agility context policy uses after frames only;
- action macro expands to `ComputerAction` sequence;
- navigation planner adds a detour around an obstacle;
- runner handles `run_action_macro`;
- runner handles `navigate_to_target`.

Verification:

```text
..\.venv\Scripts\python.exe -m pytest tests\unit\test_responses_video_input.py tests\unit\test_gui_agent_architecture.py tests\unit\test_cfie_client_gui_agent.py -q
42 passed, 2 dependency warnings

..\.venv\Scripts\python.exe -m py_compile cfie_gui_agent\macros.py cfie_gui_agent\navigation.py cfie_gui_agent\runner.py cfie_gui_agent\runtime_context.py cfie_gui_agent\tools.py cfie_gui_agent\context.py cfie_gui_agent\__init__.py
passed
```

## Implementation Round 17

Date: 2026-05-24

Local human input console:

- WeChat and other remote channels are deferred.
- The local client now has a usable browser console for human intervention.
- All channels share the same `HumanLoopManager` state model.
- A request can be:
  - pending;
  - claimed;
  - resolved;
  - cancelled.

Concurrency rule:

```text
one human request can be answered exactly once
```

If the local client claims a request, an external channel reply for the same
request is ignored. If another source has claimed it, the local client cannot
submit until the request is released.

Added shared-state APIs:

- `HumanLoopManager.list_requests()`;
- `HumanLoopManager.claim_request()`;
- `HumanLoopManager.release_request()`;
- `HumanLoopManager.submit_reply()`;
- `HumanRequestState`.

Added local console:

- module: `cfie_gui_agent.console`;
- embedded entrypoint: `start_console_in_thread(human_loop=runner.human_loop)`;
- command:

```powershell
..\.venv\Scripts\python.exe -m cfie_gui_agent.console --host 127.0.0.1 --port 8765
```

HTTP API:

```text
GET  /api/human/requests?include_completed=1
POST /api/human/requests/{request_id}/claim
POST /api/human/requests/{request_id}/release
POST /api/human/requests/{request_id}/reply
```

Browser console:

- request list;
- request status and claimant;
- manager text input;
- claim / release / submit buttons;
- raw shared state panel.

Embedding rule:

```python
from cfie_gui_agent import GuiAgentRunner
from cfie_gui_agent.console import start_console_in_thread

runner = GuiAgentRunner(max_steps=6)
console = start_console_in_thread(human_loop=runner.human_loop, port=8765)
```

This keeps model-created `request_human_help` items and local-client replies on
the same `HumanLoopManager` instance. External channels such as WeChat should
later attach to the same manager, not create a second request store.

Added project script:

```text
cfie-gui-agent-console = cfie_gui_agent.console:main
```

Verification:

```text
..\.venv\Scripts\python.exe -m pytest tests\unit\test_gui_agent_architecture.py tests\unit\test_gui_agent_console.py tests\unit\test_cfie_client_gui_agent.py -q
41 passed

..\.venv\Scripts\python.exe -m pytest tests\unit\test_responses_video_input.py tests\unit\test_gui_agent_architecture.py tests\unit\test_gui_agent_console.py tests\unit\test_cfie_client_gui_agent.py -q
45 passed, 2 warnings

..\.venv\Scripts\python.exe -m py_compile cfie_gui_agent\human_loop.py cfie_gui_agent\console.py cfie_gui_agent\__init__.py
passed
```

## Implementation Round 18

Date: 2026-05-24

Desktop client direction:

- The main GUI Agent client should be a Windows desktop client, not the browser
  debug console.
- The browser console remains useful only as a low-level human-loop API smoke
  test.
- The desktop client is designed for Chinese users and centers on:
  - one Target APP mapped to one JOB;
  - APP task description text;
  - reference images/videos cited from text, for example `[image:map_main]`;
  - continuous key-control macros with model-facing descriptions;
  - model execution trace, including prompt visual context, reasoning text,
    output text, and tool results;
  - JOB/Subtask queue visibility;
  - blocked Subtask human intervention.

Added desktop client module:

```powershell
..\.venv\Scripts\python.exe -m cfie_gui_agent.desktop_client
```

Human intervention form:

- A structured human reply contains:
  - handling type;
  - direct command;
  - manager input text;
  - extra constraints.
- Direct command is a separate field, so the model/harness does not need to
  infer path changes only from free text.

Initial direct commands:

```text
none
continue
do_not_reply
change_path
pause_job
cancel_subtask
mark_complete
```

Example:

```text
direct_command=do_not_reply
manager_input=不要回复这个卖家，切换到下一个可处理会话。
```

This supports the user case where the manager wants to change the operation
path rather than merely provide missing text.

## Implementation Round 19

Date: 2026-05-24

Production-mode client rule:

- Client code must not add throwaway controls, temporary prompts, or seeded
  startup state.
- The desktop client starts from an empty production state.
- Human intervention requests are created by model/harness runtime paths, not
  by a visible request-creation shortcut.
- Tests may inject state directly, but runtime UI should not expose test data
  creation controls.

Cleaned up:

- removed desktop "create human block example" controls;
- removed desktop startup seed data;
- removed browser console request-creation endpoint;
- removed browser console request-creation button;
- updated tests to create requests by direct state injection.

## Future Implementation Checklist

Task orchestration:

- Add `TaskSpec`, `TaskState`, `TaskStack`, and transition operations.
- Add explicit task states: `active`, `paused`, `completed`, `failed`,
  `cancelled`, `superseded`.
- Add subtask types: primary, interrupt, override, recovery, verification,
  maintenance.
- Implement rolling planning instead of fixed long-plan execution.

Harness constraints:

- Programmatic tool schema validation.
- Coordinate bounds validation.
- Max-step and timeout limits.
- Repeated action and no-screen-change detection.
- Safe interrupt/override/rollback validation.
- Trace and artifact persistence.

Context management:

- Implement 43-frame default visual context policy.
- Keep raw artifacts out of prompt by default.
- Keep only selected frame references plus structured summaries in model input.
- Implement downgrade rules when the estimated token budget is exceeded.
- Avoid duplicate before/after image history in the model context.
- Add model-assisted compaction plan generation.
- Validate and apply compaction plans programmatically.
- Preserve artifact-store source of truth independent from prompt context.
- Add warning/compact/emergency context thresholds.

Reward data:

- Add subtask start/end step tracking.
- Add terminal state markers for subtask completion.
- Add reward-event records per atomic action.
- Add transition reward metadata for interrupt, override, rollback, and resume.
- Add discounted credit assignment for completed subtasks.
- Add bounded multi-action proposal policy with per-action verification.

Desktop console / human loop:

- Add business/workspace profile design.
- Add UI surface for long task prompt, constraints, reference images, reference
  videos, and SOP documents.
- Add runtime monitor fields for model proposal, harness validation, executed
  action, verifier result, and task state.
- Add `HumanLoopManager`.
- Add replaceable `HumanChannel` interface.
- Add WeChat channel as GUI_Agent application-layer backend, not cfie_client
  core.
- Add `waiting_human_queue` and `urgent_queue`.
- Add safe interruption point handling.
- Add user correction -> structured policy conversion.

Tool registry:

- Define model-callable tool schemas for `computer_use`, `read_image`,
  `read_video_clip`, `request_human_help`, `update_constraints`,
  `finish_subtask`, `report_blocked`, `ask_replan`, and `query_memory`.
- Keep `check_human_reply` scheduler-owned rather than model-callable.
- Keep video decoding, OCR, frame selection, compaction execution, trace,
  reward, queue scheduling, WeChat low-level transport, and safety validation as
  harness-internal tools.
- Add schema validation and allowlist enforcement for all model-callable tools.

Engine/application validation:

- Keep `--enable-prefix-caching` in the default GUI_Agent service CLI.
- Add E2E tests that catch excessive thinking output length.
- Add tests for tool-call schema stability under Responses/Chat APIs.
- Track both engine microbench metrics and GUI_Agent task metrics.

Testing:

- Unit test task stack transitions.
- Unit test interrupt -> resume behavior.
- Unit test override does not resume old task.
- Unit test rollback checkpoint metadata.
- Unit test 43-frame budget selection.
- Unit test context downgrade order.
- Unit test model compaction plan validation.
- Unit test step merge and evidence retention.
- Unit test emergency forced downgrade.
- Unit test subtask completion reward back-propagation.
- Unit test interrupt reward does not erase penalty for the action that caused
  the bad state.
- Unit test override reward and wrong-override penalty.
- Unit test multi-action proposal stops when verifier detects completion or
  interrupt.
- Unit test human intervention parks a task without stopping the scheduler.
- Unit test manager reply becomes urgent task.
- Unit test urgent task executes only at safe interruption point.
- Unit test user correction persists as structured policy.
- Unit test model cannot call harness-internal tool names.
- Unit test `ask_replan` is only a proposal and cannot mutate task stack without
  TaskManager validation.
- Unit test `request_human_help` creates a human request but does not block the
  scheduler.
- Smoke test with a simple GUI task: open browser, navigate, type, verify page
  change.
- Later game-like simulation test: death interrupt, revive, resume original
  task.

## Interview Framing

Concise explanation:

> The GUI Agent should not execute a static long plan. User configuration defines
> the long-horizon goal and constraints. The harness maintains a task stack and
> uses rolling planning to select the next subtask. Interrupt tasks temporarily
> pause and resume the original task; override tasks replace it; rollback returns
> to a stable checkpoint. The model proposes semantic actions, but the harness
> validates state transitions, tool calls, retries, timeouts, loop detection, and
> safety. For context, recent history is high-fidelity video, mid-history is
> after-frame snapshots, and long history is structured text with artifact
> references. This keeps the system stable, replayable, and trainable.
