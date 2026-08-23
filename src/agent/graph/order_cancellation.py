"""取消订单子图：条件抽取、只读查询、选择确认和事务软取消。"""

from collections.abc import Callable
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from agent.common.booking_db import BookingDB, PostgresBookingDB
from agent.common.database import SQLTools
from agent.common.llm import build_chat_model
from agent.graph.database_query import build_database_query_graph
from agent.node.order_cancellation import OrderCancellationNodes
from agent.state.order_cancellation import (
    OrderCancellationInput,
    OrderCancellationOutput,
    OrderCancellationState,
)


def build_order_cancellation_graph(
    *,
    model_factory: Callable[[], Any] = build_chat_model,
    booking_db_factory: Callable[[], BookingDB] = PostgresBookingDB,
    sql_tools_factory: Callable[[Any], SQLTools] | None = None,
    name: str = "order_cancellation",
):
    """构建可嵌入主图、也可在 Studio 独立运行的取消订单子图。"""

    nodes = OrderCancellationNodes(
        booking_db_factory=booking_db_factory,
        model_factory=model_factory,
    )
    order_query_subgraph = build_database_query_graph(
        allowed_tables={"booking_order"},
        model_factory=model_factory,
        sql_tools_factory=sql_tools_factory,
        name=f"{name}_order_query",
    )

    builder = StateGraph(
        OrderCancellationState,
        input_schema=OrderCancellationInput,
        output_schema=OrderCancellationOutput,
    )
    builder.add_node("initialize", nodes.initialize)
    builder.add_node("extract_order_filters", nodes.extract_order_filters)
    builder.add_node("prepare_order_query", nodes.prepare_order_query)
    builder.add_node("check_order", order_query_subgraph)
    builder.add_node("cancel_order", nodes.cancel_order)
    builder.add_node("generate_answer", nodes.generate_answer)

    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "extract_order_filters")
    builder.add_conditional_edges(
        "extract_order_filters",
        _route_after_filter_extraction,
        {"query": "prepare_order_query", "answer": "generate_answer"},
    )
    builder.add_edge("prepare_order_query", "check_order")
    builder.add_conditional_edges(
        "check_order",
        _route_after_query,
        {"cancel": "cancel_order", "answer": "generate_answer"},
    )
    builder.add_edge("cancel_order", "generate_answer")
    builder.add_edge("generate_answer", END)
    return builder.compile(name=name)


def _route_after_filter_extraction(
    state: OrderCancellationState,
) -> Literal["query", "answer"]:
    return (
        "answer"
        if state.get("cancellation_status") == "input_invalid"
        else "query"
    )


def _route_after_query(
    state: OrderCancellationState,
) -> Literal["cancel", "answer"]:
    return "cancel" if state.get("query_status") == "success" else "answer"


order_cancellation_graph = build_order_cancellation_graph()
