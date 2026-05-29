from cfie_gui_agent.agent_tools import (
    AgentToolCall,
    AgentToolError,
    find_agent_tool_calls,
    find_computer_tool_calls,
)
from cfie_gui_agent.context import (
    CompactionApplication,
    ContextManager,
    LongHistorySummary,
    PromptContextSelection,
    StepRecord,
    VisionContextPolicy,
)
from cfie_gui_agent.human_loop import (
    HUMAN_REQUEST_CANCELLED,
    HUMAN_REQUEST_CLAIMED,
    HUMAN_REQUEST_PENDING,
    HUMAN_REQUEST_RESOLVED,
    HumanLoopManager,
    HumanReply,
    HumanRequest,
    HumanRequestState,
    InMemoryHumanChannel,
)
from cfie_gui_agent.jobs import (
    JobBoard,
    JobBoardError,
    JobSelection,
    JobState,
    PerJobContextStore,
    SubtaskQueues,
    SubtaskState,
    SwitchEvent,
)
from cfie_gui_agent.macros import (
    ActionMacro,
    ActionMacroError,
    ActionMacroRegistry,
    ActionMacroStep,
    action_macro_from_proposal,
)
from cfie_gui_agent.monitor import (
    MonitorController,
    MonitorEvent,
    MonitorIngestionResult,
)
from cfie_gui_agent.navigation import (
    NavigationPlan,
    NavigationPlanError,
    NavigationPlanner,
    NavigationRequest,
    ObstaclePolygon,
    Point,
)
from cfie_gui_agent.openai_responses import (
    OpenAIResponsesAgent,
    OpenAIResponsesAgentError,
)
from cfie_gui_agent.policy import PolicyRule, PolicyStore, PolicyUpdate
from cfie_gui_agent.rewards import (
    RewardEvent,
    SubtaskRewardAssignment,
    assign_subtask_completion_credit,
    transition_reward_event,
)
from cfie_gui_agent.runtime_context import RuntimeContext, RuntimeContextBuilder
from cfie_gui_agent.runner import GuiAgentRunner, ResponseAgent
from cfie_gui_agent.scheduler import AgentScheduler, QueuedTask, ScheduledDecision
from cfie_gui_agent.specs import GuiAgentResult, GuiAgentTaskSpec, WorkspaceProfile
from cfie_gui_agent.task import (
    TASK_STATUS_ACTIVE,
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PAUSED,
    TASK_STATUS_SUPERSEDED,
    TASK_TYPE_INTERRUPT,
    TASK_TYPE_OVERRIDE,
    TASK_TYPE_PRIMARY,
    TASK_TYPE_RECOVERY,
    TaskStack,
    TaskState,
    TaskStateError,
    TaskTransition,
)
from cfie_gui_agent.tools import (
    CORE_MODEL_CALLABLE_TOOLS,
    MINIMAL_MODEL_CALLABLE_TOOLS,
    MODEL_CALLABLE_TOOLS,
    ModelToolRegistry,
    ModelToolSpec,
    ToolRegistryError,
    model_tool_names_for_profile,
)
from cfie_gui_agent.trace import AgentTraceEvent, AgentTraceStore
from cfie_gui_agent.verifier import (
    StepVerification,
    StepVerifier,
    VERIFICATION_NO_SCREEN_CHANGE,
    VERIFICATION_OK,
    VERIFICATION_REPEATED_ACTION,
)

__all__ = [
    "AgentToolCall",
    "AgentToolError",
    "ActionMacro",
    "ActionMacroError",
    "ActionMacroRegistry",
    "ActionMacroStep",
    "action_macro_from_proposal",
    "ContextManager",
    "CompactionApplication",
    "GuiAgentRunner",
    "GuiAgentTaskSpec",
    "GuiAgentResult",
    "HumanLoopManager",
    "HumanReply",
    "HumanRequest",
    "HumanRequestState",
    "HUMAN_REQUEST_CANCELLED",
    "HUMAN_REQUEST_CLAIMED",
    "HUMAN_REQUEST_PENDING",
    "HUMAN_REQUEST_RESOLVED",
    "InMemoryHumanChannel",
    "JobBoard",
    "JobBoardError",
    "JobSelection",
    "JobState",
    "LongHistorySummary",
    "CORE_MODEL_CALLABLE_TOOLS",
    "MINIMAL_MODEL_CALLABLE_TOOLS",
    "MODEL_CALLABLE_TOOLS",
    "ModelToolRegistry",
    "ModelToolSpec",
    "MonitorController",
    "MonitorEvent",
    "MonitorIngestionResult",
    "NavigationPlan",
    "NavigationPlanError",
    "NavigationPlanner",
    "NavigationRequest",
    "ObstaclePolygon",
    "OpenAIResponsesAgent",
    "OpenAIResponsesAgentError",
    "Point",
    "PromptContextSelection",
    "PerJobContextStore",
    "PolicyRule",
    "PolicyStore",
    "PolicyUpdate",
    "AgentScheduler",
    "QueuedTask",
    "ResponseAgent",
    "RewardEvent",
    "RuntimeContext",
    "RuntimeContextBuilder",
    "ScheduledDecision",
    "SubtaskRewardAssignment",
    "TASK_STATUS_ACTIVE",
    "TASK_STATUS_COMPLETED",
    "TASK_STATUS_FAILED",
    "TASK_STATUS_PAUSED",
    "TASK_STATUS_SUPERSEDED",
    "TASK_TYPE_INTERRUPT",
    "TASK_TYPE_OVERRIDE",
    "TASK_TYPE_PRIMARY",
    "TASK_TYPE_RECOVERY",
    "TaskStack",
    "TaskState",
    "TaskStateError",
    "TaskTransition",
    "ToolRegistryError",
    "AgentTraceEvent",
    "AgentTraceStore",
    "StepRecord",
    "StepVerification",
    "StepVerifier",
    "SubtaskQueues",
    "SubtaskState",
    "SwitchEvent",
    "VERIFICATION_NO_SCREEN_CHANGE",
    "VERIFICATION_OK",
    "VERIFICATION_REPEATED_ACTION",
    "VisionContextPolicy",
    "WorkspaceProfile",
    "assign_subtask_completion_credit",
    "find_agent_tool_calls",
    "find_computer_tool_calls",
    "load_desktop_state",
    "model_tool_names_for_profile",
    "save_desktop_state",
    "transition_reward_event",
]


def __getattr__(name: str):
    if name in {"load_desktop_state", "save_desktop_state"}:
        from cfie_gui_agent.state_store import load_desktop_state, save_desktop_state

        return {
            "load_desktop_state": load_desktop_state,
            "save_desktop_state": save_desktop_state,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
