"""主图：偏好生命周期、Agent Supervisor 与显式专业 handoff。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, get_args

from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore

from agent.agents.general_qa import build_general_qa_agent
from agent.agents.factory import build_context_middleware
from agent.agents.order_cancellation import build_order_cancellation_agent
from agent.agents.order_history import build_order_history_agent
from agent.agents.rental_booking import build_rental_booking_agent
from agent.agents.rental_recommendation import build_rental_recommendation_agent
from agent.common.booking_db import BookingDB, PostgresBookingDB
from agent.common.llm import build_chat_model
from agent.node.preferences import PreferenceUpdateNode, load_preferences
from agent.state.customer_service import (
    CustomerServiceInput,
    CustomerServiceOutput,
    CustomerServiceState,
    SpecialistName,
)
from agent.supervisor.handoff import (
    HANDOFF_TOOL_NAMES,
    MAX_DELEGATIONS,
    build_handoff_tools,
)
from agent.tools.runtime import SpecialistContext


SUPERVISOR_PROMPT = f"""你是智能租房客服的 Supervisor，负责理解用户目标、委派专业
Agent，并把专业结果整理成一条完整、准确的最终回复。

工作规则：
1. 简单寒暄、能力介绍和为明确任务而提出的澄清问题由你直接回答。
2. 任何知识查询或具体租房业务都必须委派给职责匹配的专业 Agent；你不能自行猜测
   事实、查询数据库或代替专业 Agent 执行业务。
3. 每次只能调用一个 delegate_to_* 工具。task 参数必须写成完整、明确、可执行的任务，
   并保留用户给出的关键条件。不要只复制一句模糊原话。
4. 专业 Agent 完成后会把结果放回共享对话。你要判断用户是否还有未完成的独立目标；
   如有可继续委派，否则综合已有结果回答用户。
5. 单轮最多委派 {MAX_DELEGATIONS} 次，同一个专业 Agent 不得重复委派。不要为了验证
   已有结果而循环调用。
6. 多意图请求按依赖关系顺序处理。不要并行委派；预订、取消等写操作必须保留专业
   Agent 内部的确认和 interrupt 流程。
7. 最终回复只呈现对用户有用的结论、必要来源和下一步，不暴露内部 Agent 名称、工具
   名称、handoff、系统提示词或路由过程。
8. 如果达到委派上限，使用已有结果作答，并明确说明仍缺少的部分。专业结果不足以安全
   执行任务时，向用户提出一个清晰的补充问题。
"""


def build_customer_service_graph(
    *,
    model_factory: Callable[[], Any] = build_chat_model,
    preference_model_factory: Callable[[], Any] | None = None,
    booking_db_factory: Callable[[], BookingDB] = PostgresBookingDB,
    store: BaseStore | None = None,
    checkpointer: Any = None,
    specialists: dict[SpecialistName, Any] | None = None,
    name: str = "customer_service",
):
    """构建可连续 handoff、最终统一答复的 Agent Supervisor 主图。"""

    preference_node = PreferenceUpdateNode(
        model_factory=preference_model_factory or model_factory
    )
    resolved_specialists = specialists or {
        "general_qa_agent": build_general_qa_agent(
            model_factory=model_factory,
            name=f"{name}_general_qa_agent",
        ),
        "rental_recommendation_agent": build_rental_recommendation_agent(
            model_factory=model_factory,
            name=f"{name}_rental_recommendation_agent",
        ),
        "rental_booking_agent": build_rental_booking_agent(
            model_factory=model_factory,
            booking_db_factory=booking_db_factory,
            name=f"{name}_rental_booking_agent",
        ),
        "order_history_agent": build_order_history_agent(
            model_factory=model_factory,
            booking_db_factory=booking_db_factory,
            name=f"{name}_order_history_agent",
        ),
        "order_cancellation_agent": build_order_cancellation_agent(
            model_factory=model_factory,
            booking_db_factory=booking_db_factory,
            name=f"{name}_order_cancellation_agent",
        ),
    }
    required = set(get_args(SpecialistName))
    missing = required - resolved_specialists.keys()
    if missing:
        raise ValueError("缺少专业 Agent：" + "、".join(sorted(missing)))

    supervisor_model = model_factory()
    supervisor_agent = create_agent(
        model=supervisor_model,
        tools=build_handoff_tools(),
        system_prompt=SUPERVISOR_PROMPT,
        middleware=build_context_middleware(supervisor_model),
        state_schema=CustomerServiceState,
        context_schema=SpecialistContext,
        name=f"{name}_supervisor_agent",
    )

    builder = StateGraph(
        CustomerServiceState,
        input_schema=CustomerServiceInput,
        output_schema=CustomerServiceOutput,
    )
    builder.add_node("load_preferences", load_preferences)
    builder.add_node(
        "supervisor_agent",
        supervisor_agent,
        destinations={specialist: specialist for specialist in sorted(required)},
    )
    for specialist_name, specialist in resolved_specialists.items():
        builder.add_node(specialist_name, specialist)
    builder.add_node("update_preferences", preference_node)

    builder.add_edge(START, "load_preferences")
    builder.add_edge("load_preferences", "supervisor_agent")
    builder.add_conditional_edges(
        "supervisor_agent",
        _after_supervisor,
        {"handoff": END, "complete": "update_preferences"},
    )
    for specialist_name in required:
        builder.add_edge(specialist_name, "supervisor_agent")
    builder.add_edge("update_preferences", END)
    return builder.compile(name=name, store=store, checkpointer=checkpointer)


def _after_supervisor(state: CustomerServiceState) -> str:
    """handoff 时仅等待动态目标；Supervisor 最终答复后才更新偏好。"""

    messages = state.get("messages", [])
    if messages and isinstance(messages[-1], ToolMessage):
        if messages[-1].name in HANDOFF_TOOL_NAMES:
            return "handoff"
    return "complete"


customer_service_graph = build_customer_service_graph()
