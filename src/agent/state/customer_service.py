"""Agent Supervisor 主图使用的公共状态。"""

from typing import Any, Literal, NotRequired, TypedDict

from langgraph.graph import MessagesState

from agent.state.preferences import PreferenceState


SpecialistName = Literal[
    "general_qa_agent",
    "rental_recommendation_agent",
    "rental_booking_agent",
    "order_history_agent",
    "order_cancellation_agent",
]
MissingField = Literal[
    "city",
    "districts",
    "budget_min",
    "budget_max",
    "room_types",
    "rental_mode",
    "house_id",
    "phone",
    "check_in_date",
    "check_out_date",
    "order_no",
]


class CustomerServiceInput(MessagesState):
    """主图公共输入：当前轮新增的用户消息。"""


class CustomerServiceOutput(MessagesState):
    """主图公共输出：合并后的完整对话消息。"""


class SpecialistResult(TypedDict):
    """专业 Agent 对 Supervisor 暴露的唯一完成结果。"""

    agent: SpecialistName
    status: Literal["success", "needs_input", "failed"]
    summary: str
    user_facing_answer: str
    completed_tasks: list[str]
    remaining_tasks: list[str]


class DelegationRecord(TypedDict):
    """一次 handoff 的任务、工具调用和可选完成结果。"""

    agent: SpecialistName
    task: str
    tool_call_id: str
    result: NotRequired[SpecialistResult]


class UserInputGuardEvent(TypedDict):
    """最近一次普通模型回复的用户输入等待判定。"""

    message_id: str | None
    result: Literal["request", "terminal", "pass"]
    source: Literal["hard_rule", "llm", "fallback"]
    requires_user_input: bool
    missing_fields: list[MissingField]
    reason: str
    error: str


class CustomerServiceState(PreferenceState):
    """主图状态；完整消息由 checkpoint 保存，Supervisor 自主委派任务。

    专业 Agent 的模型上下文由官方 middleware 在每次模型调用前统一管理。
    """

    # 本轮委派记录同时承担次数限制、重复检测和结果追踪。
    delegations: NotRequired[list[DelegationRecord]]
    # 只保存最新 Guard 诊断；完整变化由 checkpoint 历史承担。
    last_guard_event: NotRequired[UserInputGuardEvent]
    # 当前专业任务的循环预算；新委派/新用户轮次重置，中断恢复保留。
    specialist_budget: NotRequired[dict[str, Any]]
