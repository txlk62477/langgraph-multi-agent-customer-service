"""主图：公共身份与上下文、专业 Agent 路由、统一偏好持久化。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore

from agent.agents.general_qa import build_general_qa_agent
from agent.agents.order_cancellation import build_order_cancellation_agent
from agent.agents.order_history import build_order_history_agent
from agent.agents.rental_booking import build_rental_booking_agent
from agent.agents.rental_recommendation import build_rental_recommendation_agent
from agent.common.booking_db import BookingDB, PostgresBookingDB
from agent.common.llm import build_chat_model
from agent.node.customer_service import CustomerServiceNodes
from agent.node.preferences import PreferenceExtractionNodes, load_preferences, save_preferences
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
    booking_db_factory: Callable[[], BookingDB] = PostgresBookingDB,
    store: BaseStore | None = None,
    checkpointer: Any = None,
    specialists: dict[CustomerIntent, Any] | None = None,
    name: str = "customer_service",
):
    """构建一次只路由到一个专业 Agent 的客服 Supervisor。"""

    supervisor_nodes = CustomerServiceNodes(model_factory=model_factory)
    preference_nodes = PreferenceExtractionNodes(
        model_factory=preference_model_factory or model_factory
    )
    resolved_specialists = specialists or {
        "general_qa": build_general_qa_agent(
            model_factory=model_factory,
            name=f"{name}_general_qa_agent",
        ),
        "recommend_rental": build_rental_recommendation_agent(
            model_factory=model_factory,
            name=f"{name}_rental_recommendation_agent",
        ),
        "reserve_rental": build_rental_booking_agent(
            model_factory=model_factory,
            booking_db_factory=booking_db_factory,
            name=f"{name}_rental_booking_agent",
        ),
        "order_history": build_order_history_agent(
            model_factory=model_factory,
            booking_db_factory=booking_db_factory,
            name=f"{name}_order_history_agent",
        ),
        "cancel_order": build_order_cancellation_agent(
            model_factory=model_factory,
            booking_db_factory=booking_db_factory,
            name=f"{name}_order_cancellation_agent",
        ),
    }
    required = {
        "general_qa",
        "recommend_rental",
        "reserve_rental",
        "order_history",
        "cancel_order",
    }
    missing = required - resolved_specialists.keys()
    if missing:
        raise ValueError("缺少专业 Agent：" + "、".join(sorted(missing)))

    builder = StateGraph(
        CustomerServiceState,
        input_schema=CustomerServiceInput,
        output_schema=CustomerServiceOutput,
    )
    builder.add_node("load_preferences", load_preferences)
    builder.add_node("prepare_routing_context", supervisor_nodes.prepare_routing_context)
    builder.add_node("identify_intent", supervisor_nodes.identify_intent)
    for route, specialist in resolved_specialists.items():
        builder.add_node(route, specialist)
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
        {route: route for route in sorted(required)},
    )
    for route in required:
        builder.add_edge(route, "extract_preference_updates")
    builder.add_edge("extract_preference_updates", "save_preferences")
    builder.add_edge("save_preferences", END)
    return builder.compile(name=name, store=store, checkpointer=checkpointer)


def _route_customer_intent(state: CustomerServiceState) -> CustomerIntent:
    """缺失或失败时安全进入常规问答 Agent。"""

    return state.get("customer_intent", "general_qa")


customer_service_graph = build_customer_service_graph()
