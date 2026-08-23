"""历史订单子图编排：数量识别、参数化查询与固定格式结果。"""

from collections.abc import Callable
from typing import Any

from langgraph.graph import END, START, StateGraph

from agent.common.booking_db import BookingDB, PostgresBookingDB
from agent.common.llm import build_chat_model
from agent.node.order_history import OrderHistoryNodes
from agent.state.order_history import (
    OrderHistoryInput,
    OrderHistoryOutput,
    OrderHistoryState,
)


def build_order_history_graph(
    *,
    model_factory: Callable[[], Any] = build_chat_model,
    booking_db_factory: Callable[[], BookingDB] = PostgresBookingDB,
    name: str = "order_history",
):
    """构建可嵌入主图、也可在Studio独立运行的历史订单子图。"""

    nodes = OrderHistoryNodes(
        booking_db_factory=booking_db_factory,
        model_factory=model_factory,
    )
    builder = StateGraph(
        OrderHistoryState,
        input_schema=OrderHistoryInput,
        output_schema=OrderHistoryOutput,
    )
    builder.add_node("reset", nodes.reset)
    builder.add_node("extract_order_limit", nodes.extract_order_limit)
    builder.add_node("query_orders", nodes.query_orders)
    builder.add_node("format_result", nodes.format_result)
    builder.add_edge(START, "reset")
    builder.add_edge("reset", "extract_order_limit")
    builder.add_edge("extract_order_limit", "query_orders")
    builder.add_edge("query_orders", "format_result")
    builder.add_edge("format_result", END)
    return builder.compile(name=name)


# 独立Studio图：默认使用环境变量中的 PostgreSQL 与 DeepSeek。
order_history_graph = build_order_history_graph()
