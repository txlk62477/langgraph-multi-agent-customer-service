"""Supervisor、专业 Agent 工具循环和中断恢复的行为测试。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
import unittest

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
from pydantic import PrivateAttr

from agent.agents import (
    general_qa_agent,
    order_cancellation_agent,
    order_history_agent,
    rental_booking_agent,
    rental_recommendation_agent,
)
from agent.agents.order_cancellation import build_order_cancellation_agent
from agent.agents.order_history import build_order_history_agent
from agent.agents.rental_booking import build_rental_booking_agent
from agent.common.booking_db import (
    BookingCancellationResult,
    BookingCreateResult,
    OrderRecord,
)
from agent.node.customer_service import CustomerIntentDecision
from agent.node.preferences import PreferenceExtractionDecision
from agent.supervisor.graph import build_customer_service_graph
from agent.tools.conversation import build_request_user_input_tool
from agent.tools.orders import build_cancel_order_tool, build_find_cancellable_orders_tool


class ScriptedToolModel(BaseChatModel):
    """按顺序返回工具调用与最终回答的最小聊天模型。"""

    _responses: list[AIMessage] = PrivateAttr()
    _index: int = PrivateAttr(default=0)
    _bound_tool_names: list[str] = PrivateAttr(default_factory=list)

    def __init__(self, responses: list[AIMessage]) -> None:
        super().__init__()
        self._responses = responses

    @property
    def _llm_type(self) -> str:
        return "scripted-tool-model"

    @property
    def bound_tool_names(self) -> list[str]:
        return self._bound_tool_names

    def bind_tools(self, tools, **kwargs):
        del kwargs
        self._bound_tool_names = [tool.name for tool in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        del messages, stop, run_manager, kwargs
        response = self._responses[self._index]
        self._index += 1
        return ChatResult(generations=[ChatGeneration(message=response)])


class FakeBookingDB:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.cancelled: list[dict[str, str]] = []

    def create_booking(self, **kwargs) -> BookingCreateResult:
        self.created.append(kwargs)
        return BookingCreateResult(
            success=True,
            order_no="agent-order-1",
            house_id=1,
            house_title=kwargs["house_title"],
            price=1800,
        )

    def list_recent_orders(self, *, user_id: str, limit: int) -> list[OrderRecord]:
        return [
            OrderRecord(
                order_no="history-order-1",
                house_id=1,
                house_title="测试公寓",
                phone="13800138000",
                check_in_date="2027-09-01",
                check_out_date="2027-09-02",
                status="confirmed",
                price=1800,
            )
        ][:limit]

    def cancel_booking(self, *, user_id: str, order_no: str):
        self.cancelled.append({"user_id": user_id, "order_no": order_no})
        return BookingCancellationResult(success=True)


class SpecialistAgentTests(unittest.TestCase):
    def test_all_specialists_are_react_tool_loops(self) -> None:
        for graph in (
            general_qa_agent,
            rental_recommendation_agent,
            rental_booking_agent,
            order_history_agent,
            order_cancellation_agent,
        ):
            with self.subTest(agent=graph.name):
                nodes = set(graph.get_graph().nodes)
                self.assertEqual(nodes, {"__start__", "model", "tools", "__end__"})

    def test_order_history_agent_uses_runtime_identity_and_tool(self) -> None:
        db = FakeBookingDB()
        model = ScriptedToolModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "list_recent_orders",
                            "args": {"limit": 1},
                            "id": "history-call",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="您最近有一笔测试公寓订单。"),
            ]
        )
        graph = build_order_history_agent(
            model_factory=lambda: model,
            booking_db_factory=lambda: db,
            name="test_order_history_agent",
        )

        result = graph.invoke(
            {"messages": [HumanMessage(content="查询最近订单")]},
            config={"configurable": {"user_id": "agent-user"}},
        )

        self.assertIn("list_recent_orders", model.bound_tool_names)
        self.assertIn("测试公寓", result["messages"][-1].content)

    def test_specialist_creates_model_once_when_graph_is_built(self) -> None:
        db = FakeBookingDB()
        model = ScriptedToolModel([AIMessage(content="模型已复用。")])
        factory_calls = 0

        def model_factory() -> ScriptedToolModel:
            nonlocal factory_calls
            factory_calls += 1
            return model

        graph = build_order_history_agent(
            model_factory=model_factory,
            booking_db_factory=lambda: db,
            name="test_eager_model_agent",
        )

        self.assertEqual(factory_calls, 1)
        graph.invoke(
            {"messages": [HumanMessage(content="你好")]},
            config={"configurable": {"user_id": "agent-user"}},
        )
        self.assertEqual(factory_calls, 1)

    def test_booking_agent_interrupts_then_resumes_and_writes_once(self) -> None:
        db = FakeBookingDB()
        model = ScriptedToolModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "request_user_input",
                            "args": {
                                "question": "请补充手机号",
                                "reason": "缺少预订信息",
                                "missing_fields": ["phone"],
                            },
                            "id": "ask-phone",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "create_booking",
                            "args": {
                                "phone": "13800138000",
                                "house_title": "测试公寓",
                                "check_in_date": "2027-09-01",
                                "check_out_date": "2027-09-02",
                            },
                            "id": "create-booking",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="预订成功，订单号 agent-order-1。"),
            ]
        )
        graph = build_rental_booking_agent(
            model_factory=lambda: model,
            booking_db_factory=lambda: db,
            checkpointer=InMemorySaver(),
            name="test_rental_booking_agent",
        )
        config = {"configurable": {"thread_id": "booking-thread", "user_id": "u-1"}}

        first = graph.invoke(
            {"messages": [HumanMessage(content="我要预订测试公寓")]},
            config=config,
        )
        self.assertEqual(first["__interrupt__"][0].value["missing_required_fields"], ["phone"])

        result = graph.invoke(Command(resume="13800138000"), config=config)

        self.assertEqual(len(db.created), 1)
        self.assertEqual(db.created[0]["user_id"], "u-1")
        self.assertIn("agent-order-1", result["messages"][-1].content)

    def test_cancellation_tool_forces_confirmation_before_write(self) -> None:
        db = FakeBookingDB()
        lookup_calls: list[dict[str, Any]] = []

        def lookup(**kwargs) -> list[dict[str, Any]]:
            lookup_calls.append(kwargs)
            return [
                {
                    "order_no": "cancel-order-1",
                    "house_title": "测试公寓",
                    "check_in_date": "2027-09-01",
                    "check_out_date": "2027-09-02",
                    "status": "confirmed",
                    "price": 1800,
                }
            ]

        tools = [
            build_find_cancellable_orders_tool(lookup=lookup),
            build_request_user_input_tool(),
            build_cancel_order_tool(booking_db_factory=lambda: db, lookup=lookup),
        ]
        model = ScriptedToolModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "find_cancellable_orders",
                            "args": {"limit": 5},
                            "id": "find-order",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "cancel_order",
                            "args": {"order_no": "cancel-order-1"},
                            "id": "cancel-order",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="订单 cancel-order-1 已取消。"),
            ]
        )
        graph = build_order_cancellation_agent(
            model_factory=lambda: model,
            booking_db_factory=lambda: db,
            tools=tools,
            checkpointer=InMemorySaver(),
            name="test_order_cancellation_agent",
        )
        config = {"configurable": {"thread_id": "cancel-thread", "user_id": "u-2"}}

        first = graph.invoke(
            {"messages": [HumanMessage(content="取消我的订单")]},
            config=config,
        )
        self.assertEqual(first["__interrupt__"][0].value["type"], "confirm_order_cancellation")
        self.assertEqual(db.cancelled, [])

        result = graph.invoke(Command(resume="确认取消"), config=config)

        self.assertEqual(db.cancelled, [{"user_id": "u-2", "order_no": "cancel-order-1"}])
        self.assertIn("已取消", result["messages"][-1].content)
        self.assertTrue(all(call["user_id"] == "u-2" for call in lookup_calls))


class SupervisorTests(unittest.TestCase):
    def test_supervisor_routes_to_one_specialist_and_returns_to_preferences(self) -> None:
        class StructuredModel:
            def with_structured_output(self, schema, *, method=None):
                del method

                class Invoker:
                    def invoke(self, messages):
                        del messages
                        if schema is CustomerIntentDecision:
                            return CustomerIntentDecision(
                                intent="order_history",
                                reason="用户要查询订单",
                            )
                        if schema is PreferenceExtractionDecision:
                            return PreferenceExtractionDecision(
                                rental_related=False,
                                reason="本轮没有偏好变化",
                            )
                        raise AssertionError(f"未处理的结构化模型：{schema}")

                return Invoker()

        def specialist(label: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
            return lambda state: {"messages": [AIMessage(content=label)]}

        specialists = {
            "general_qa": specialist("general_qa"),
            "recommend_rental": specialist("recommend_rental"),
            "reserve_rental": specialist("reserve_rental"),
            "order_history": specialist("order_history"),
            "cancel_order": specialist("cancel_order"),
        }
        graph = build_customer_service_graph(
            model_factory=StructuredModel,
            specialists=specialists,
            store=InMemoryStore(),
            name="test_supervisor",
        )

        result = graph.invoke(
            {"messages": [HumanMessage(content="查询我的历史订单")]},
            config={"configurable": {"user_id": "supervisor-user"}},
        )

        self.assertEqual(result["messages"][-1].content, "order_history")

    def test_supervisor_resume_returns_to_interrupted_specialist(self) -> None:
        class RouterModel:
            def with_structured_output(self, schema, *, method=None):
                del method

                class Invoker:
                    def invoke(self, messages):
                        del messages
                        if schema is CustomerIntentDecision:
                            return CustomerIntentDecision(
                                intent="reserve_rental",
                                reason="用户要预订房源",
                            )
                        if schema is PreferenceExtractionDecision:
                            return PreferenceExtractionDecision(
                                rental_related=False,
                                reason="本轮没有偏好变化",
                            )
                        raise AssertionError(schema)

                return Invoker()

        db = FakeBookingDB()
        specialist_model = ScriptedToolModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "request_user_input",
                            "args": {
                                "question": "请补充手机号",
                                "reason": "缺少预订信息",
                                "missing_fields": ["phone"],
                            },
                            "id": "nested-ask",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "create_booking",
                            "args": {
                                "phone": "13800138000",
                                "house_title": "测试公寓",
                                "check_in_date": "2027-09-01",
                                "check_out_date": "2027-09-02",
                            },
                            "id": "nested-create",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="预订成功，订单号 agent-order-1。"),
            ]
        )
        booking_agent = build_rental_booking_agent(
            model_factory=lambda: specialist_model,
            booking_db_factory=lambda: db,
            name="nested_booking_agent",
        )

        def unused_specialist(state):
            del state
            return {"messages": [AIMessage(content="不应调用")]}

        graph = build_customer_service_graph(
            model_factory=RouterModel,
            preference_model_factory=RouterModel,
            specialists={
                "general_qa": unused_specialist,
                "recommend_rental": unused_specialist,
                "reserve_rental": booking_agent,
                "order_history": unused_specialist,
                "cancel_order": unused_specialist,
            },
            store=InMemoryStore(),
            checkpointer=InMemorySaver(),
            name="test_resumable_supervisor",
        )
        config = {
            "configurable": {
                "thread_id": "supervisor-booking-thread",
                "user_id": "supervisor-user",
            }
        }

        first = graph.invoke(
            {"messages": [HumanMessage(content="我要预订测试公寓")]},
            config=config,
        )
        self.assertEqual(first["__interrupt__"][0].value["missing_required_fields"], ["phone"])

        result = graph.invoke(Command(resume="13800138000"), config=config)

        self.assertEqual(len(db.created), 1)
        self.assertEqual(db.created[0]["user_id"], "supervisor-user")
        self.assertIn("agent-order-1", result["messages"][-1].content)


if __name__ == "__main__":
    unittest.main()
