"""取消订单子图的公共接口与内部状态。"""

from typing import Any, Literal, NotRequired

from langgraph.graph import MessagesState

from agent.state.database_query import QueryStatus


CancellationStatus = Literal[
    "collecting",
    "input_invalid",
    "querying",
    "order_not_found",
    "query_failed",
    "awaiting_selection",
    "awaiting_confirmation",
    "selection_failed",
    "cancelled_by_user",
    "already_cancelled",
    "already_started",
    "not_cancellable",
    "cancel_failed",
    "success",
]


class OrderCancellationInput(MessagesState):
    """取消子图输入：对话消息以及主图解析出的可选用户身份。"""

    user_id: NotRequired[str | None]


class OrderCancellationOutput(MessagesState):
    """取消子图输出：只向父图返回用户可见消息。"""


class OrderCancellationState(MessagesState):
    """取消订单全过程的内部状态。

    查询条件由 extract_order_filters 提取，check_order 复用通用只读查询
    子图；选择、确认和软取消所需字段只在本子图内部流转。
    """

    # 当前用户身份，由主图同名字段传入；查询提示和最终写入都限定此用户。
    user_id: NotRequired[str | None]

    # 用户可选提供的订单线索；全部为空时查询最近可取消订单。
    order_no: NotRequired[str | None]
    house_title: NotRequired[str | None]
    check_in_date_start: NotRequired[str | None]
    check_in_date_end: NotRequired[str | None]

    # database_query 子图的公开输入和输出字段。
    query_request: NotRequired[str]
    table_name: NotRequired[str]
    max_rows: NotRequired[int]
    query_status: NotRequired[QueryStatus]
    query_result: NotRequired[str]
    query_error: NotRequired[str]

    # 从原始查询结果中解析出的候选订单以及用户最终选择。
    order_candidates: NotRequired[list[dict[str, Any]]]
    selected_order: NotRequired[dict[str, Any]]

    # 全局业务状态和稳定错误原因，由 generate_answer 统一生成回复。
    cancellation_status: NotRequired[CancellationStatus]
    cancellation_error: NotRequired[str]
