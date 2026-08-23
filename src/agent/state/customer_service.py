"""智能客服主图使用的公共状态。"""

from typing import Any, Literal, NotRequired

from langchain_core.messages import AnyMessage
from langgraph.graph import MessagesState

from agent.state.general_qa import GeneralQAState
from agent.common.preferences import PreferenceField
from agent.state.preferences import PreferenceUpdates


CustomerIntent = Literal[
    "general_qa",
    "recommend_rental",
    "reserve_rental",
    "cancel_order",
    "order_history",
]


class CustomerServiceInput(MessagesState):
    """主图公共输入：当前轮新增的用户消息。"""


class CustomerServiceOutput(MessagesState):
    """主图公共输出：合并后的完整对话消息。"""


class CustomerServiceState(GeneralQAState):
    """主图状态；完整消息由checkpoint保存，节点只补充路由和偏好数据。

    继承链为 CustomerServiceState → GeneralQAState → ContextState →
    MessagesState，因此主图自动拥有 messages、上下文处理、问答路由和联网搜索
    字段。各节点接收当前状态中与其输入Schema同名的字段，并返回局部更新。
    """

    # 用户身份：load_preferences 节点按 state → configurable → CHAT_USER_ID
    # 顺序解析后写入，业务子图（预订/历史订单）直接继承此字段。
    user_id: NotRequired[str | None]

    # prepare_routing_context 从完整 messages 中截取的轻量路由窗口；主图意图识别
    # 只读取最近几条消息，不执行完整的话题总结和向量召回。
    routing_messages: NotRequired[list[AnyMessage]]
    # 当前业务流程第一条用户消息的ID；偏好提取完成后直接清空。
    current_turn_start_message_id: NotRequired[str | None]
    # identify_intent 选出的业务分支，由主图条件边读取并完成节点路由。
    customer_intent: NotRequired[CustomerIntent]
    # 意图模型给出的决策理由，主要用于 LangSmith 观测和问题排查。
    intent_reason: NotRequired[str]
    # 意图识别失败原因；失败时 customer_intent 会安全降级为 general_qa。
    intent_error: NotRequired[str]

    # load_preferences 从 Store 读取的跨Thread长期租房偏好。
    user_preferences: NotRequired[dict[str, Any]]
    # extract_preference_updates 从当前轮对话提取、等待写入 Store 的字段增量。
    preference_updates: NotRequired[PreferenceUpdates]
    # 当前轮明确要求清除的偏好字段；save_preferences 持久化后将其清空。
    preference_clear_fields: NotRequired[list[PreferenceField]]
    # 偏好提取失败原因；提取失败不会阻断当前业务回答。
    preference_extraction_error: NotRequired[str]
    # save_preferences 是否在当前轮实际写入了偏好变更。
    preferences_saved: NotRequired[bool]
    # 偏好写入失败原因；失败时清空本轮增量但不阻断核心业务。
    preference_save_error: NotRequired[str]
    # load_preferences 读取 Store 失败的原因；失败时使用空偏好继续执行。
    preference_load_error: NotRequired[str]
