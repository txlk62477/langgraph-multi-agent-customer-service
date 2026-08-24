"""Supervisor、专业 Agent 工具循环和中断恢复的行为测试。"""

from __future__ import annotations

import json
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
from agent.agents.general_qa import build_general_qa_agent
from agent.agents.factory import build_specialist_agent
from agent.agents.specialist_budget import SpecialistBudgetPolicy
from agent.agents.order_cancellation import build_order_cancellation_agent
from agent.agents.order_history import build_order_history_agent
from agent.agents.rental_booking import build_rental_booking_agent
from agent.agents.rental_recommendation import build_rental_recommendation_agent
from agent.common.booking_db import (
    BookingCancellationResult,
    BookingCreateResult,
    OrderRecord,
)
from agent.node.preferences import PreferenceExtractionDecision
from agent.state.customer_service import SpecialistResult
from agent.supervisor.graph import build_customer_service_graph
from agent.tools.conversation import build_request_user_input_tool
from agent.tools.orders import build_cancel_order_tool, build_find_cancellable_orders_tool


class ScriptedToolModel(BaseChatModel):
    """按顺序返回工具调用与最终回答的最小聊天模型。"""

    _responses: list[AIMessage] = PrivateAttr()
    _index: int = PrivateAttr(default=0)
    _bound_tool_names: list[str] = PrivateAttr(default_factory=list)
    _seen_messages: list[Any] = PrivateAttr(default_factory=list)

    def __init__(self, responses: list[AIMessage]) -> None:
        super().__init__()
        self._responses = responses

    @property
    def _llm_type(self) -> str:
        return "scripted-tool-model"

    @property
    def bound_tool_names(self) -> list[str]:
        return self._bound_tool_names

    @property
    def seen_messages(self) -> list[Any]:
        return self._seen_messages

    def bind_tools(self, tools, **kwargs):
        del kwargs
        self._bound_tool_names = [
            tool.name
            if hasattr(tool, "name")
            else tool["function"]["name"]
            for tool in tools
        ]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        del stop, run_manager, kwargs
        self._seen_messages = list(messages)
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
            house_title="测试公寓",
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

    def search_orders(self, *, user_id: str, limit: int, **kwargs) -> list[OrderRecord]:
        del user_id, kwargs
        return self.list_recent_orders(user_id="ignored", limit=limit)

    def get_order(self, *, user_id: str, order_no: str) -> OrderRecord | None:
        del user_id
        return OrderRecord(
            order_no=order_no,
            house_id=1,
            house_title="测试公寓",
            phone="13800138000",
            check_in_date="2099-09-01",
            check_out_date="2099-09-02",
            status="confirmed",
            price=1800,
        )

    def cancel_booking(self, *, user_id: str, order_no: str):
        self.cancelled.append({"user_id": user_id, "order_no": order_no})
        return BookingCancellationResult(success=True)


class FakeRentalCatalog:
    def find_houses(self, *, query: str, limit: int) -> list[dict[str, Any]]:
        del query, limit
        return [
            {
                "id": 1,
                "title": "省心租·新世界路2号楼 1室1厅1卫",
                "price": 1000,
                "area": 55,
            },
            {
                "id": 2,
                "title": "省心租·新世界路2号楼 1室1厅1卫",
                "price": 1100,
                "area": 52,
            },
        ]


class SpecialistAgentTests(unittest.TestCase):
    def test_specialist_returns_structured_result_after_finishing(self) -> None:
        model = ScriptedToolModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "SpecialistResult",
                            "args": {
                                "agent": "general_qa_agent",
                                "status": "success",
                                "summary": "已完成知识回答",
                                "user_facing_answer": "合肥是安徽省省会。",
                                "completed_tasks": ["回答省会问题"],
                                "remaining_tasks": [],
                            },
                            "id": "structured-result",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        )
        graph = build_general_qa_agent(
            model_factory=lambda: model,
            name="structured_general_qa_agent",
        )

        result = graph.invoke({"messages": [HumanMessage(content="安徽省会是哪里？")]})

        self.assertEqual(
            result["structured_response"],
            SpecialistResult(
                agent="general_qa_agent",
                status="success",
                summary="已完成知识回答",
                user_facing_answer="合肥是安徽省省会。",
                completed_tasks=["回答省会问题"],
                remaining_tasks=[],
            ),
        )

    def test_all_specialists_are_react_tool_loops(self) -> None:
        guarded = {
            rental_recommendation_agent.name,
            rental_booking_agent.name,
            order_cancellation_agent.name,
        }
        for graph in (
            general_qa_agent,
            rental_recommendation_agent,
            rental_booking_agent,
            order_history_agent,
            order_cancellation_agent,
        ):
            with self.subTest(agent=graph.name):
                nodes = set(graph.get_graph().nodes)
                expected = {
                    "__start__",
                    "SpecialistBudgetMiddleware.before_agent",
                    "SummarizationMiddleware.before_model",
                    "SpecialistBudgetMiddleware.before_model",
                    "model",
                    "SpecialistBudgetMiddleware.after_model",
                    "tools",
                    "__end__",
                }
                if graph.name in guarded:
                    expected.add("UserInputGuardMiddleware.after_model")
                self.assertEqual(nodes, expected)

    def test_general_qa_agent_can_plan_with_three_research_tools(self) -> None:
        model = ScriptedToolModel([AIMessage(content="无需联网的直接回答。")])
        graph = build_general_qa_agent(
            model_factory=lambda: model,
            name="test_autonomous_general_qa_agent",
        )

        graph.invoke({"messages": [HumanMessage(content="你好")]})
        self.assertEqual(
            model.bound_tool_names,
            [
                "anysearch_search",
                "playwright_read_page",
                "analyze_page_visuals",
                "SpecialistResult",
            ],
        )
        prompt = "\n".join(
            str(message.content)
            for message in model.seen_messages
            if isinstance(message.content, str)
        )
        for policy in (
            "自主决定",
            "交叉验证",
            "Markdown",
            "最多调用3次",
            "最多读取4个",
            "最多分析2个",
            "不得用模型记忆猜测",
        ):
            with self.subTest(policy=policy):
                self.assertIn(policy, prompt)

    def test_each_business_agent_exposes_granular_tools(self) -> None:
        cases = [
            (
                build_rental_recommendation_agent,
                {
                    "get_rental_preferences",
                    "inspect_rental_market",
                    "request_user_input",
                    "search_houses",
                    "get_house_details",
                },
            ),
            (
                build_rental_booking_agent,
                {
                    "find_bookable_houses",
                    "check_booking_availability",
                    "request_user_input",
                    "create_booking",
                },
            ),
            (
                build_order_history_agent,
                {"list_recent_orders", "search_orders", "get_order_details"},
            ),
            (
                build_order_cancellation_agent,
                {
                    "find_cancellable_orders",
                    "check_cancellation_eligibility",
                    "request_user_input",
                    "cancel_order",
                },
            ),
        ]
        for builder, expected in cases:
            with self.subTest(builder=builder.__name__):
                model = ScriptedToolModel([AIMessage(content="完成")])
                graph = builder(model_factory=lambda: model, name="tool-set-test")
                graph.invoke(
                    {"messages": [HumanMessage(content="测试")]},
                    config={"configurable": {"user_id": "user-1"}},
                )
                self.assertEqual(
                    set(model.bound_tool_names),
                    expected | {"SpecialistResult"},
                )

    def test_order_history_agent_uses_runtime_identity_and_tool(self) -> None:
        db = FakeBookingDB()
        model = ScriptedToolModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "list_recent_orders",
                            "args": {
                                "limit": 1,
                                "selection_reason": "用户要求查询最近订单",
                            },
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
                                "selection_reason": "缺少预订所需手机号",
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
                                "house_id": 1,
                                "check_in_date": "2027-09-01",
                                "check_out_date": "2027-09-02",
                                "selection_reason": "预订信息齐全，可以创建订单",
                            },
                            "id": "create-booking",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="预订成功，订单号 agent-order-1。"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "SpecialistResult",
                            "args": {
                                "agent": "rental_booking_agent",
                                "status": "success",
                                "summary": "已开始新的预订任务",
                                "user_facing_answer": "请告诉我需要预订哪套房源。",
                                "completed_tasks": [],
                                "remaining_tasks": ["确认房源"],
                            },
                            "id": "new-turn-result",
                            "type": "tool_call",
                        }
                    ],
                ),
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
        self.assertEqual(
            first["__interrupt__"][0].value["selection_reason"],
            "缺少预订所需手机号",
        )
        self.assertNotIn("reason", first["__interrupt__"][0].value)

        result = graph.invoke(Command(resume="13800138000"), config=config)

        self.assertEqual(len(db.created), 1)
        self.assertEqual(db.created[0]["user_id"], "u-1")
        self.assertIn("agent-order-1", result["messages"][-1].content)
        resumed_budget = graph.get_state(config).values["specialist_budget"]
        self.assertEqual(resumed_budget["model_calls"], 3)
        self.assertEqual(resumed_budget["business_tool_calls"], 1)

        graph.invoke(
            {"messages": [HumanMessage(content="再预订另一套房")]},
            config=config,
        )
        new_turn_budget = graph.get_state(config).values["specialist_budget"]
        self.assertEqual(new_turn_budget["model_calls"], 1)
        self.assertEqual(new_turn_budget["business_tool_calls"], 0)

    def test_booking_plain_text_question_is_converted_to_interrupt(self) -> None:
        """模型忘记调用工具时，用户补充问题仍必须进入 interrupt。"""

        model = ScriptedToolModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "find_bookable_houses",
                            "args": {
                                "query": "省心租·新世界路2号楼 1室1厅1卫",
                                "max_results": 5,
                                "selection_reason": "需要定位用户指定的可预订房源",
                            },
                            "id": "find-two-booking-candidates",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(
                    content=(
                        "请问您想预订哪一套？另外，还需要您提供手机号、"
                        "入住日期和退房日期。"
                    )
                ),
                AIMessage(content="已收到您补充的预订信息。"),
            ]
        )
        graph = build_rental_booking_agent(
            model_factory=lambda: model,
            catalog_factory=FakeRentalCatalog,
            checkpointer=InMemorySaver(),
            name="booking_plain_question_guard",
        )
        config = {
            "configurable": {
                "thread_id": "booking-plain-question",
                "user_id": "booking-user",
            }
        }

        result = graph.invoke(
            {
                "messages": [
                    HumanMessage(content="预订省心租·新世界路2号楼 1室1厅1卫")
                ]
            },
            config=config,
        )

        self.assertIn("__interrupt__", result)
        payload = result["__interrupt__"][0].value
        self.assertEqual(payload["type"], "agent_request_user_input")
        self.assertEqual(
            payload["missing_required_fields"],
            ["house_id", "phone", "check_in_date", "check_out_date"],
        )
        guard_event = graph.get_state(config).values["last_guard_event"]
        self.assertEqual(guard_event["result"], "request")
        self.assertEqual(guard_event["source"], "hard_rule")
        self.assertEqual(
            guard_event["missing_fields"],
            ["house_id", "phone", "check_in_date", "check_out_date"],
        )

        resumed = graph.invoke(
            Command(resume="选择第1套，13800138000，2027-10-01至2027-10-03"),
            config=config,
        )
        self.assertEqual(resumed["messages"][-1].content, "已收到您补充的预订信息。")

    def test_all_user_input_agents_guard_plain_text_requests(self) -> None:
        cases = [
            (
                build_rental_recommendation_agent,
                "请提供城市、最低预算和最高预算。",
                ["city", "budget_min", "budget_max"],
            ),
            (
                build_order_cancellation_agent,
                "请选择您要取消的具体订单号。",
                ["order_no"],
            ),
        ]
        for index, (builder, question, expected_fields) in enumerate(cases):
            with self.subTest(builder=builder.__name__):
                model = ScriptedToolModel([AIMessage(content=question)])
                graph = builder(
                    model_factory=lambda: model,
                    checkpointer=InMemorySaver(),
                    name=f"guarded-input-agent-{index}",
                )
                result = graph.invoke(
                    {"messages": [HumanMessage(content="请继续处理")]},
                    config={
                        "configurable": {
                            "thread_id": f"guarded-input-{index}",
                            "user_id": "guarded-user",
                        }
                    },
                )

                payload = result["__interrupt__"][0].value
                self.assertEqual(payload["message"], question)
                self.assertEqual(payload["missing_required_fields"], expected_fields)

    def test_cancellation_tool_forces_confirmation_before_write(self) -> None:
        db = FakeBookingDB()
        tools = [
            build_find_cancellable_orders_tool(booking_db_factory=lambda: db),
            build_request_user_input_tool(),
            build_cancel_order_tool(booking_db_factory=lambda: db),
        ]
        model = ScriptedToolModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "find_cancellable_orders",
                            "args": {
                                "limit": 5,
                                "selection_reason": "需要查找当前用户可取消的订单",
                            },
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
                            "args": {
                                "order_no": "cancel-order-1",
                                "selection_reason": "用户已指定需要取消的订单",
                            },
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


class SupervisorTests(unittest.TestCase):
    @staticmethod
    def _preference_model():
        class PreferenceModel:
            def with_structured_output(self, schema, *, method=None):
                del method
                if schema is not PreferenceExtractionDecision:
                    raise AssertionError(schema)
                return self

            def invoke(self, messages):
                del messages
                return PreferenceExtractionDecision(
                    rental_related=False,
                    reason="本轮没有偏好变化",
                )

        return PreferenceModel()

    @staticmethod
    def _specialists(**overrides):
        def specialist(agent: str, label: str):
            return lambda state: {
                "messages": [AIMessage(content=f"内部消息：{label}", name=agent)],
                "structured_response": SpecialistResult(
                    agent=agent,
                    status="success",
                    summary=label,
                    user_facing_answer=label,
                    completed_tasks=[label],
                    remaining_tasks=[],
                ),
            }

        defaults = {
            "general_qa_agent": specialist("general_qa_agent", "常规问答结果"),
            "rental_recommendation_agent": specialist(
                "rental_recommendation_agent", "房源推荐结果"
            ),
            "rental_booking_agent": specialist("rental_booking_agent", "预订结果"),
            "order_history_agent": specialist("order_history_agent", "历史订单结果"),
            "order_cancellation_agent": specialist(
                "order_cancellation_agent", "取消结果"
            ),
        }
        defaults.update(overrides)
        return defaults

    def test_supervisor_handoffs_then_synthesizes_final_answer(self) -> None:
        supervisor_model = ScriptedToolModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "delegate_to_order_history",
                            "args": {
                                "task": "查询当前用户的历史订单",
                                "selection_reason": "用户要求查看自己的历史订单",
                            },
                            "id": "handoff-history",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="您有一笔历史订单。"),
            ]
        )
        graph = build_customer_service_graph(
            model_factory=lambda: supervisor_model,
            preference_model_factory=self._preference_model,
            specialists=self._specialists(),
            store=InMemoryStore(),
            checkpointer=InMemorySaver(),
            name="test_supervisor",
        )
        config = {
            "configurable": {
                "thread_id": "handoff-supervisor-thread",
                "user_id": "supervisor-user",
            }
        }
        result = graph.invoke(
            {"messages": [HumanMessage(content="查询我的历史订单")]},
            config=config,
        )
        state = graph.get_state(config).values

        self.assertEqual(result["messages"][-1].content, "您有一笔历史订单。")
        self.assertEqual(
            state["delegations"],
            [
                {
                    "agent": "order_history_agent",
                    "task": "查询当前用户的历史订单",
                    "selection_reason": "用户要求查看自己的历史订单",
                    "tool_call_id": "handoff-history",
                    "result": {
                        "agent": "order_history_agent",
                        "status": "success",
                        "summary": "历史订单结果",
                        "user_facing_answer": "历史订单结果",
                        "completed_tasks": ["历史订单结果"],
                        "remaining_tasks": [],
                    },
                }
            ],
        )
        self.assertNotIn("delegation_count", state)
        self.assertNotIn("delegated_agents", state)
        handoff_message = next(
            message
            for message in state["messages"]
            if getattr(message, "tool_call_id", None) == "handoff-history"
        )
        self.assertEqual(
            json.loads(handoff_message.content)["selection_reason"],
            "用户要求查看自己的历史订单",
        )
        self.assertTrue(
            any("历史订单结果" in str(message.content) for message in supervisor_model.seen_messages)
        )
        self.assertTrue(
            any(
                "用户要求查看自己的历史订单" in str(message.content)
                for message in supervisor_model.seen_messages
            )
        )
        self.assertFalse(
            any("内部消息" in str(message.content) for message in supervisor_model.seen_messages)
        )
        self.assertTrue(
            any("内部消息：历史订单结果" in str(message.content) for message in state["messages"])
        )
        self.assertEqual(
            set(supervisor_model.bound_tool_names),
            {
                "delegate_to_general_qa",
                "delegate_to_rental_recommendation",
                "delegate_to_rental_booking",
                "delegate_to_order_history",
                "delegate_to_order_cancellation",
            },
        )

    def test_supervisor_collects_specialist_budget_fallback(self) -> None:
        supervisor_model = ScriptedToolModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "delegate_to_general_qa",
                            "args": {
                                "task": "查询天气",
                                "selection_reason": "实时天气需要常规问答 Agent 联网查询",
                            },
                            "id": "handoff-budget-fallback",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="本次天气查询达到执行上限，请稍后重试。"),
            ]
        )
        specialist_model = ScriptedToolModel([AIMessage(content="未按结构结束")])
        limited_specialist = build_specialist_agent(
            name="limited_general_qa_agent",
            specialist_name="general_qa_agent",
            system_prompt="处理测试任务",
            tools=[],
            model_factory=lambda: specialist_model,
            middleware=[],
            budget_policy=SpecialistBudgetPolicy(
                business_tool_calls=1,
                model_calls=1,
            ),
        )
        graph = build_customer_service_graph(
            model_factory=lambda: supervisor_model,
            preference_model_factory=self._preference_model,
            specialists=self._specialists(general_qa_agent=limited_specialist),
            store=InMemoryStore(),
            checkpointer=InMemorySaver(),
            name="test_supervisor_budget_fallback",
        )
        config = {
            "configurable": {
                "thread_id": "supervisor-budget-fallback",
                "user_id": "supervisor-user",
            }
        }

        graph.invoke(
            {"messages": [HumanMessage(content="合肥天气怎么样？")]},
            config=config,
        )

        delegation = graph.get_state(config).values["delegations"][0]
        self.assertEqual(delegation["result"]["status"], "failed")
        self.assertEqual(
            delegation["result"]["summary"],
            "专业 Agent 达到执行上限",
        )

    def test_supervisor_answers_greeting_without_handoff(self) -> None:
        supervisor_model = ScriptedToolModel(
            [AIMessage(content="你好，我可以协助租房和订单服务。")]
        )
        graph = build_customer_service_graph(
            model_factory=lambda: supervisor_model,
            preference_model_factory=self._preference_model,
            specialists=self._specialists(),
            store=InMemoryStore(),
            checkpointer=InMemorySaver(),
            name="test_direct_supervisor",
        )
        config = {
            "configurable": {
                "thread_id": "direct-supervisor-thread",
                "user_id": "supervisor-user",
            }
        }
        result = graph.invoke(
            {"messages": [HumanMessage(content="你好")]},
            config=config,
        )

        self.assertEqual(graph.get_state(config).values["delegations"], [])
        self.assertIn("你好", result["messages"][-1].content)

    def test_supervisor_enforces_three_delegation_limit(self) -> None:
        handoffs = [
            ("delegate_to_general_qa", "知识查询", "limit-qa"),
            ("delegate_to_rental_recommendation", "推荐房源", "limit-rec"),
            ("delegate_to_order_history", "查询订单", "limit-history"),
            ("delegate_to_order_cancellation", "取消订单", "limit-cancel"),
        ]
        responses = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": tool_name,
                        "args": {
                            "task": task,
                            "selection_reason": f"该任务适合委派给 {tool_name}",
                        },
                        "id": call_id,
                        "type": "tool_call",
                    }
                ],
            )
            for tool_name, task, call_id in handoffs
        ]
        responses.append(AIMessage(content="已根据前三项专业结果完成回复。"))
        supervisor_model = ScriptedToolModel(responses)
        graph = build_customer_service_graph(
            model_factory=lambda: supervisor_model,
            preference_model_factory=self._preference_model,
            specialists=self._specialists(),
            store=InMemoryStore(),
            checkpointer=InMemorySaver(),
            name="test_supervisor_limit",
        )
        config = {
            "configurable": {
                "thread_id": "supervisor-limit-thread",
                "user_id": "supervisor-user",
            }
        }

        result = graph.invoke(
            {"messages": [HumanMessage(content="处理四项不同任务")]},
            config=config,
        )
        state = graph.get_state(config).values

        self.assertEqual(len(state["delegations"]), 3)
        self.assertNotIn(
            "order_cancellation_agent",
            [record["agent"] for record in state["delegations"]],
        )
        self.assertIn("前三项", result["messages"][-1].content)
        denial = next(
            message
            for message in state["messages"]
            if getattr(message, "tool_call_id", None) == "limit-cancel"
        )
        self.assertIn("最多委派3次", denial.content)

    def test_supervisor_rejects_repeated_specialist_handoff(self) -> None:
        supervisor_model = ScriptedToolModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "delegate_to_order_history",
                            "args": {
                                "task": "第一次查询订单",
                                "selection_reason": "用户要求查询历史订单",
                            },
                            "id": "repeat-first",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "delegate_to_order_history",
                            "args": {
                                "task": "再次查询相同订单",
                                "selection_reason": "尝试重复查询相同订单",
                            },
                            "id": "repeat-second",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="已使用第一次订单查询结果。"),
            ]
        )
        graph = build_customer_service_graph(
            model_factory=lambda: supervisor_model,
            preference_model_factory=self._preference_model,
            specialists=self._specialists(),
            store=InMemoryStore(),
            checkpointer=InMemorySaver(),
            name="test_supervisor_repeat",
        )
        config = {
            "configurable": {
                "thread_id": "supervisor-repeat-thread",
                "user_id": "supervisor-user",
            }
        }

        graph.invoke(
            {"messages": [HumanMessage(content="查询订单后再重复查一次")]},
            config=config,
        )
        state = graph.get_state(config).values

        self.assertEqual(len(state["delegations"]), 1)
        denial = next(
            message
            for message in state["messages"]
            if getattr(message, "tool_call_id", None) == "repeat-second"
        )
        self.assertIn("已经委派过", denial.content)
        self.assertEqual(json.loads(denial.content)["status"], "rejected")
        self.assertEqual(
            json.loads(denial.content)["selection_reason"],
            "尝试重复查询相同订单",
        )

    def test_supervisor_resume_returns_to_interrupted_specialist(self) -> None:
        db = FakeBookingDB()
        supervisor_model = ScriptedToolModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "delegate_to_rental_booking",
                            "args": {
                                "task": "预订测试公寓并补齐缺失信息",
                                "selection_reason": "用户要求创建租房预订",
                            },
                            "id": "handoff-booking",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="预订已完成，订单号 agent-order-1。"),
            ]
        )
        specialist_model = ScriptedToolModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "request_user_input",
                            "args": {
                                "question": "请补充手机号",
                                "selection_reason": "缺少预订所需手机号",
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
                                "house_id": 1,
                                "check_in_date": "2027-09-01",
                                "check_out_date": "2027-09-02",
                                "selection_reason": "信息齐全，可以创建预订",
                            },
                            "id": "nested-create",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "SpecialistResult",
                            "args": {
                                "agent": "rental_booking_agent",
                                "status": "success",
                                "summary": "预订已完成",
                                "user_facing_answer": "预订成功，订单号 agent-order-1。",
                                "completed_tasks": ["创建预订"],
                                "remaining_tasks": [],
                            },
                            "id": "booking-result",
                            "type": "tool_call",
                        }
                    ],
                ),
            ]
        )
        booking_agent = build_rental_booking_agent(
            model_factory=lambda: specialist_model,
            booking_db_factory=lambda: db,
            name="nested_booking_agent",
        )

        graph = build_customer_service_graph(
            model_factory=lambda: supervisor_model,
            preference_model_factory=self._preference_model,
            specialists=self._specialists(rental_booking_agent=booking_agent),
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
        self.assertFalse(
            any(
                "13800138000" in str(message.content)
                for message in supervisor_model.seen_messages
            )
        )
        checkpoint_messages = graph.get_state(config).values["messages"]
        self.assertTrue(
            any(
                "13800138000" in str(message.content)
                for message in checkpoint_messages
            )
        )
        booking_result = graph.get_state(config).values["delegations"][0]["result"]
        self.assertEqual(booking_result["status"], "success")


if __name__ == "__main__":
    unittest.main()
