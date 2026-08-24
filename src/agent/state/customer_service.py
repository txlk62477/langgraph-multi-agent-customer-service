"""Agent Supervisor 主图使用的公共状态。"""

from typing import Any, Literal, NotRequired

from langchain.agents.middleware.types import AgentState
from langgraph.graph import MessagesState

from agent.common.preferences import PreferenceField
from agent.state.preferences import PreferenceUpdates


SpecialistName = Literal[
    "general_qa_agent",
    "rental_recommendation_agent",
    "rental_booking_agent",
    "order_history_agent",
    "order_cancellation_agent",
]


class CustomerServiceInput(MessagesState):
    """主图公共输入：当前轮新增的用户消息。"""


class CustomerServiceOutput(MessagesState):
    """主图公共输出：合并后的完整对话消息。"""


class CustomerServiceState(AgentState):
    """主图状态；完整消息由 checkpoint 保存，Supervisor 自主委派任务。

    专业 Agent 的模型上下文由官方 middleware 在每次模型调用前统一管理。
    """

    # 用户身份：load_preferences 节点按 state → configurable → CHAT_USER_ID
    # 顺序解析后写入，业务子图（预订/历史订单）直接继承此字段。
    user_id: NotRequired[str | None]

    # load_preferences 记录的当前轮第一条用户消息 ID。
    current_turn_start_message_id: NotRequired[str | None]

    # 当前轮已完成的 handoff 数量和目标，工具以此强制限制委派预算。
    delegation_count: NotRequired[int]
    delegated_agents: NotRequired[list[SpecialistName]]

    # load_preferences 从 Store 读取的跨Thread长期租房偏好。
    user_preferences: NotRequired[dict[str, Any]]
    # update_preferences 在单节点内提取并保存的临时字段增量。
    preference_updates: NotRequired[PreferenceUpdates]
    # 当前轮明确要求清除的偏好字段；写入后将其清空。
    preference_clear_fields: NotRequired[list[PreferenceField]]
    # 偏好提取失败原因；提取失败不会阻断当前业务回答。
    preference_extraction_error: NotRequired[str]
    # update_preferences 是否在当前轮实际写入了偏好变更。
    preferences_saved: NotRequired[bool]
    # 偏好写入失败原因；失败时清空本轮增量但不阻断核心业务。
    preference_save_error: NotRequired[str]
    # load_preferences 读取 Store 失败的原因；失败时使用空偏好继续执行。
    preference_load_error: NotRequired[str]
