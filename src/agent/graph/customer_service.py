"""智能客服主图：轻量路由、业务子图分发和统一偏好持久化。"""

from collections.abc import Callable
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore

from agent.common.booking_db import BookingDB, PostgresBookingDB
from agent.common.database import SQLTools
from agent.common.llm import build_chat_model
from agent.graph.general_qa import build_general_qa_graph
from agent.graph.order_history import build_order_history_graph
from agent.graph.order_cancellation import build_order_cancellation_graph
from agent.graph.rental_booking import build_rental_booking_graph
from agent.graph.rental_recommendation import build_rental_recommendation_graph
from agent.node.customer_service import CustomerServiceNodes
from agent.node.preferences import (
    PreferenceExtractionNodes,
    load_preferences,
    save_preferences,
)
from agent.state.customer_service import (
    CustomerIntent,
    CustomerServiceInput,
    CustomerServiceOutput,
    CustomerServiceState,
)


def build_customer_service_graph(
    *,
    model_factory: Callable[[], Any] = build_chat_model,
    preference_model_factory: Callable[[], Any] | None = None,
    rental_sql_tools_factory: Callable[[Any], SQLTools] | None = None,
    booking_db_factory: Callable[[], BookingDB] | None = None,
    store: BaseStore | None = None,
    name: str = "customer_service",
):
    """构建可注册到Agent Server的智能客服主图。"""

    nodes = CustomerServiceNodes(model_factory=model_factory)
    preference_nodes = PreferenceExtractionNodes(
        model_factory=preference_model_factory or model_factory
    )
    general_qa_subgraph = build_general_qa_graph(
        model_factory=model_factory,
        name=f"{name}_general_qa",
    )
    rental_recommendation_subgraph = build_rental_recommendation_graph(
        model_factory=model_factory,
        sql_tools_factory=rental_sql_tools_factory,
        include_preference_loading=False,
        name=f"{name}_rental_recommendation",
    )
    rental_booking_subgraph = build_rental_booking_graph(
        model_factory=model_factory,
        booking_db_factory=booking_db_factory or PostgresBookingDB,
        sql_tools_factory=rental_sql_tools_factory,
        name=f"{name}_rental_booking",
    )
    order_history_subgraph = build_order_history_graph(
        model_factory=model_factory,
        booking_db_factory=booking_db_factory or PostgresBookingDB,
        name=f"{name}_order_history",
    )
    order_cancellation_subgraph = build_order_cancellation_graph(
        model_factory=model_factory,
        booking_db_factory=booking_db_factory or PostgresBookingDB,
        sql_tools_factory=rental_sql_tools_factory,
        name=f"{name}_order_cancellation",
    )

    # 主图公共接口只接收和返回消息；身份通过 configurable 传入，路由、上下文、
    # 偏好和业务过程字段全部保留在内部状态与 checkpoint 中。
    builder = StateGraph(
        CustomerServiceState,
        input_schema=CustomerServiceInput,
        output_schema=CustomerServiceOutput,
    )
    builder.add_node("load_preferences", load_preferences)
    builder.add_node("prepare_routing_context", nodes.prepare_routing_context)
    builder.add_node("identify_intent", nodes.identify_intent)
    builder.add_node("general_qa", general_qa_subgraph)
    builder.add_node("recommend_rental", rental_recommendation_subgraph)
    builder.add_node("reserve_rental", rental_booking_subgraph)
    builder.add_node("cancel_order", order_cancellation_subgraph)
    builder.add_node("order_history", order_history_subgraph)
    builder.add_node(
        "extract_preference_updates",
        preference_nodes.extract_preference_updates,
    )
    builder.add_node("save_preferences", save_preferences)

    builder.add_edge(START, "load_preferences")
    builder.add_edge("load_preferences", "prepare_routing_context")
    builder.add_edge("prepare_routing_context", "identify_intent")
    builder.add_conditional_edges(
        "identify_intent",
        _route_customer_intent,
        {
            "general_qa": "general_qa",
            "recommend_rental": "recommend_rental",
            "reserve_rental": "reserve_rental",
            "cancel_order": "cancel_order",
            "order_history": "order_history",
        },
    )

    # 所有业务分支先从当前轮提取偏好，再统一持久化；没有增量时不会访问数据库。
    builder.add_edge("general_qa", "extract_preference_updates")
    builder.add_edge(
        "recommend_rental",
        "extract_preference_updates",
    )
    builder.add_edge(
        "reserve_rental",
        "extract_preference_updates",
    )
    builder.add_edge(
        "cancel_order",
        "extract_preference_updates",
    )
    builder.add_edge(
        "order_history",
        "extract_preference_updates",
    )
    builder.add_edge("extract_preference_updates", "save_preferences")
    builder.add_edge("save_preferences", END)
    # Agent Server 会自动注入持久化 Store；测试或直接 Python 调用时可显式传入。
    return builder.compile(name=name, store=store)


def _route_customer_intent(state: CustomerServiceState) -> CustomerIntent:
    """把结构化意图转换为条件边标签，缺失时安全进入常规问答。"""

    return state.get("customer_intent", "general_qa")


customer_service_graph = build_customer_service_graph()
