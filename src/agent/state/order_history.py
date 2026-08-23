"""历史订单子图状态。"""

from typing import Any, Literal, NotRequired

from langgraph.graph import MessagesState


OrderHistoryStatus = Literal["querying", "success", "empty", "failed"]


class OrderHistoryInput(MessagesState):
    """历史订单子图输入：对话消息和主图解析出的可选用户身份。"""

    user_id: NotRequired[str | None]


class OrderHistoryOutput(MessagesState):
    """历史订单子图输出：仅返回格式化后的用户可见消息。"""


class OrderHistoryState(OrderHistoryInput):
    """历史订单业务状态；用户身份由主图解析后继承。"""

    order_limit: NotRequired[int]
    history_status: NotRequired[OrderHistoryStatus]
    history_error: NotRequired[str]
    orders: NotRequired[list[dict[str, Any]]]
