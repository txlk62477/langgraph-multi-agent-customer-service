"""智能客服主图的离线测试。"""

from __future__ import annotations

import json
from pathlib import Path
import unittest
from unittest.mock import Mock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore

from agent.common.booking_db import BookingCreateResult, OrderRecord
from agent.common.preferences import PREFERENCE_STORE_KEY, preference_namespace
from agent.graph.customer_service import (
    _route_customer_intent,
    build_customer_service_graph,
    customer_service_graph,
)
from agent.node.customer_service import (
    CustomerIntentDecision,
    CustomerServiceNodes,
    empty_business_placeholder,
)
from agent.node.database_query import SQLQueryPlan
from agent.node.order_history import OrderLimit
from agent.node.preferences import PreferenceExtractionDecision, load_preferences
from agent.graph.rental_booking import BookingInformation
from agent.state.information_collection import RecommendInformation


class FakeStructuredIntentModel:
    def __init__(self, decision: CustomerIntentDecision) -> None:
        self.decision = decision
        self.messages = None
        self.method = None

    def with_structured_output(self, schema, *, method=None):
        if schema is not CustomerIntentDecision:
            raise AssertionError(f"未处理的结构化类型：{schema}")
        self.method = method
        return self

    def invoke(self, messages):
        self.messages = messages
        return self.decision


class FakeStructuredPreferenceModel:
    def __init__(self, decision: PreferenceExtractionDecision) -> None:
        self.decision = decision

    def with_structured_output(self, schema, *, method=None):
        if schema is not PreferenceExtractionDecision:
            raise AssertionError(f"未处理的结构化类型：{schema}")
        return self

    def invoke(self, messages):
        return self.decision


class FakeCustomerFlowModel:
    def with_structured_output(self, schema, *, method=None):
        class StructuredInvoker:
            def invoke(self, _messages):
                if schema is CustomerIntentDecision:
                    return CustomerIntentDecision(
                        intent="recommend_rental",
                        reason="用户要求推荐房源",
                    )
                if schema is RecommendInformation:
                    return RecommendInformation(
                        city="合肥",
                        budget_min=1500,
                        budget_max=3000,
                    )
                if schema is SQLQueryPlan:
                    return SQLQueryPlan.model_validate(
                        {
                            "select_columns": ["title", "price", "city_name"],
                            "filter_groups": [
                                {
                                    "conditions": [
                                        {
                                            "column": "city_name",
                                            "operator": "eq",
                                            "value": "合肥",
                                        }
                                    ]
                                }
                            ],
                        }
                    )
                raise AssertionError(f"未处理的结构化类型：{schema}")

        return StructuredInvoker()

    def invoke(self, _messages):
        return AIMessage(content="数据库房源推荐答案")


class FakeCustomerSQLTools:
    def inspect_schema(self, table_name: str) -> str:
        return f"可用表：{table_name}"

    def check_query(self, query: str) -> str:
        return query

    def execute_query(self, query: str) -> str:
        return "[('测试房源', 2000, '合肥')]"


class FakeBookingFlowModel:
    """主图预订分支：意图识别 + 预订信息抽取 + SQL 规划。"""

    def with_structured_output(self, schema, *, method=None):
        class StructuredInvoker:
            def invoke(self, _messages):
                if schema is CustomerIntentDecision:
                    return CustomerIntentDecision(
                        intent="reserve_rental",
                        reason="用户要求预订房源",
                    )
                if schema is BookingInformation:
                    return BookingInformation(
                        phone="13800138000",
                        house_title="合肥北城一号院",
                        check_in_date="2026-09-01",
                        check_out_date="2026-09-05",
                    )
                if schema is SQLQueryPlan:
                    return SQLQueryPlan.model_validate(
                        {
                            "select_columns": ["id", "title", "price"],
                            "filter_groups": [
                                {
                                    "conditions": [
                                        {
                                            "column": "title",
                                            "operator": "eq",
                                            "value": "合肥北城一号院",
                                        }
                                    ]
                                }
                            ],
                        }
                    )
                raise AssertionError(f"未处理的结构化类型：{schema}")

        return StructuredInvoker()

    def invoke(self, _messages):
        return AIMessage(content="预订占位")


class FakeBookingDB:
    """预订与历史订单子图使用的假数据适配器。"""

    def __init__(self, orders: list[OrderRecord] | None = None) -> None:
        self.calls: list[dict] = []
        self.order_calls: list[dict] = []
        self._orders = orders if orders is not None else []

    def create_booking(self, **kwargs) -> BookingCreateResult:
        self.calls.append(kwargs)
        return BookingCreateResult(
            success=True,
            order_no="11111111-2222-3333-4444-555555555555",
            house_id=11000010001,
            house_title="合肥北城一号院",
            price=2200.0,
        )

    def list_recent_orders(self, *, user_id: str, limit: int) -> list[OrderRecord]:
        self.order_calls.append({"user_id": user_id, "limit": limit})
        return list(self._orders)


class FakeOrderHistoryFlowModel:
    """主图历史订单分支：意图识别 + 订单数量识别。"""

    def with_structured_output(self, schema, *, method=None):
        class StructuredInvoker:
            def invoke(self, _messages):
                if schema is CustomerIntentDecision:
                    return CustomerIntentDecision(
                        intent="order_history",
                        reason="用户查询历史订单",
                    )
                if schema is OrderLimit:
                    return OrderLimit(limit=None)
                raise AssertionError(f"未处理的结构化类型：{schema}")

        return StructuredInvoker()

    def invoke(self, _messages):
        return AIMessage(content="订单占位")


class CustomerServiceNodeTests(unittest.TestCase):
    def test_routing_context_keeps_latest_five_human_ai_messages(self) -> None:
        nodes = CustomerServiceNodes()
        conversation = [
            HumanMessage(content=f"用户消息{index}")
            if index % 2
            else AIMessage(content=f"助手消息{index}")
            for index in range(1, 8)
        ]
        messages = [SystemMessage(content="系统消息"), *conversation]

        result = nodes.prepare_routing_context({"messages": messages})

        self.assertEqual(result["routing_messages"], conversation[-5:])
        self.assertEqual(messages[0].content, "系统消息")

    def test_routing_context_records_latest_human_as_turn_start(self) -> None:
        messages = [
            HumanMessage(content="旧问题", id="old-human"),
            AIMessage(content="旧回答", id="old-ai"),
            HumanMessage(content="当前问题", id="current-human"),
        ]

        result = CustomerServiceNodes().prepare_routing_context(
            {"messages": messages}
        )

        self.assertEqual(result["current_turn_start_message_id"], "current-human")

    def test_identifies_recommendation_from_recent_context(self) -> None:
        model = FakeStructuredIntentModel(
            CustomerIntentDecision(
                intent="recommend_rental",
                reason="用户要求推荐房源",
            )
        )
        nodes = CustomerServiceNodes(model_factory=lambda: model)

        result = nodes.identify_intent(
            {
                "messages": [HumanMessage(content="帮我推荐上海的两居室")],
                "routing_messages": [
                    HumanMessage(content="帮我推荐上海的两居室")
                ],
            }
        )

        self.assertEqual(result["customer_intent"], "recommend_rental")
        self.assertEqual(result["intent_error"], "")
        self.assertEqual(model.method, "function_calling")
        self.assertIn("帮我推荐上海的两居室", model.messages[-1].content)

    def test_intent_model_failure_falls_back_to_general_qa(self) -> None:
        nodes = CustomerServiceNodes(
            model_factory=lambda: (_ for _ in ()).throw(RuntimeError("不可用"))
        )

        result = nodes.identify_intent(
            {"messages": [HumanMessage(content="你好")]}
        )

        self.assertEqual(result["customer_intent"], "general_qa")
        self.assertIn("RuntimeError", result["intent_error"])

    def test_empty_business_placeholder_does_not_modify_state(self) -> None:
        self.assertEqual(empty_business_placeholder({"messages": []}), {})


class PreferenceLoadingTests(unittest.TestCase):
    def test_loads_preferences_for_configured_user(self) -> None:
        store = InMemoryStore()
        store.put(
            preference_namespace("lk"),
            PREFERENCE_STORE_KEY,
            {"user_id": "lk", "city": "上海"},
            index=False,
        )
        result = load_preferences(
            {},
            {"configurable": {"user_id": "lk"}},
            Runtime(store=store),
        )

        self.assertEqual(result["user_preferences"]["city"], "上海")
        self.assertEqual(result["preference_load_error"], "")

    def test_store_failure_does_not_block_main_graph(self) -> None:
        store = Mock()
        store.get.side_effect = RuntimeError("Store不可用")
        result = load_preferences(
            {"user_id": "lk"},
            {"configurable": {}},
            Runtime(store=store),
        )

        self.assertEqual(result["user_preferences"], {})
        self.assertEqual(result["user_id"], "lk")
        self.assertIn("Store不可用", result["preference_load_error"])


class CustomerServiceGraphTests(unittest.TestCase):
    @staticmethod
    def _run_with_internal_state(graph, input_state, config=None) -> dict:
        """运行主图并返回 values 流中的最终完整内部状态。"""

        states = list(
            graph.stream(
                input_state,
                config,
                stream_mode="values",
            )
        )
        if not states:
            raise AssertionError("主图没有产生状态快照")
        return states[-1]

    def test_graph_exposes_messages_only_public_schema(self) -> None:
        """主图公共接口只暴露消息，过程状态留在图内部。"""

        input_properties = customer_service_graph.get_input_jsonschema()[
            "properties"
        ]
        output_properties = customer_service_graph.get_output_jsonschema()[
            "properties"
        ]

        self.assertEqual(set(input_properties), {"messages"})
        self.assertEqual(set(output_properties), {"messages"})

    def test_graph_contains_main_pipeline_and_placeholders(self) -> None:
        node_names = set(customer_service_graph.get_graph().nodes)
        self.assertTrue(
            {
                "load_preferences",
                "prepare_routing_context",
                "identify_intent",
                "general_qa",
                "recommend_rental",
                "reserve_rental",
                "cancel_order",
                "order_history",
                "extract_preference_updates",
                "save_preferences",
            }
            <= node_names
        )

    def test_routes_each_supported_intent(self) -> None:
        for intent in (
            "general_qa",
            "recommend_rental",
            "reserve_rental",
            "cancel_order",
            "order_history",
        ):
            with self.subTest(intent=intent):
                self.assertEqual(
                    _route_customer_intent({"customer_intent": intent}),
                    intent,
                )
        self.assertEqual(_route_customer_intent({}), "general_qa")

    def test_booking_branch_creates_order_then_reaches_preference_save(
        self,
    ) -> None:
        model = FakeBookingFlowModel()
        store = InMemoryStore()
        store.put(
            preference_namespace("lk"),
            PREFERENCE_STORE_KEY,
            {"user_id": "lk", "city": "上海"},
            index=False,
        )
        graph = build_customer_service_graph(
            model_factory=lambda: model,
            preference_model_factory=lambda: FakeStructuredPreferenceModel(
                PreferenceExtractionDecision(
                    rental_related=False,
                    reason="本轮没有明确租房条件",
                )
            ),
            rental_sql_tools_factory=lambda _model: FakeCustomerSQLTools(),
            booking_db_factory=FakeBookingDB,
            store=store,
            name="test_customer_service_booking",
        )
        result = self._run_with_internal_state(
            graph,
            {"messages": [HumanMessage(content="帮我预订一套房")]},
            {"configurable": {"user_id": "lk"}},
        )

        self.assertEqual(result["customer_intent"], "reserve_rental")
        self.assertEqual(result["user_preferences"]["city"], "上海")
        self.assertIn("预订成功", result["messages"][-1].content)
        self.assertIn(
            "11111111-2222-3333-4444-555555555555",
            result["messages"][-1].content,
        )
        self.assertFalse(result["preferences_saved"])
        self.assertIsNone(result["current_turn_start_message_id"])

    def test_booking_branch_extracts_and_saves_current_turn_preferences(
        self,
    ) -> None:
        intent_model = FakeBookingFlowModel()
        preference_model = FakeStructuredPreferenceModel(
            PreferenceExtractionDecision(
                rental_related=True,
                city="合肥",
                districts_to_add=["北城"],
                reason="用户明确提出本人租房地点",
            )
        )
        store = InMemoryStore()
        graph = build_customer_service_graph(
            model_factory=lambda: intent_model,
            preference_model_factory=lambda: preference_model,
            rental_sql_tools_factory=lambda _model: FakeCustomerSQLTools(),
            booking_db_factory=FakeBookingDB,
            store=store,
            name="test_customer_service_preference_extraction",
        )
        result = self._run_with_internal_state(
            graph,
            {
                "messages": [
                    HumanMessage(content="我想预订一个合肥北城的房子")
                ],
            },
            {"configurable": {"user_id": "lk"}},
        )
        item = store.get(preference_namespace("lk"), PREFERENCE_STORE_KEY)
        self.assertEqual(item.value["city"], "合肥")
        self.assertEqual(item.value["districts"], ["北城"])
        self.assertTrue(result["preferences_saved"])
        self.assertEqual(result["user_preferences"]["city"], "合肥")
        self.assertEqual(result["preference_extraction_error"], "")
        self.assertIn("预订成功", result["messages"][-1].content)
        self.assertIn(
            "11111111-2222-3333-4444-555555555555",
            result["messages"][-1].content,
        )
        self.assertIsNone(result["current_turn_start_message_id"])

    def test_order_history_branch_queries_orders_then_saves_preferences(
        self,
    ) -> None:
        model = FakeOrderHistoryFlowModel()
        orders = [
            OrderRecord(
                order_no="11111111-2222-3333-4444-555555555555",
                house_id=11000010001,
                house_title="合肥北城一号院",
                phone="13800138000",
                check_in_date="2026-09-01",
                check_out_date="2026-09-05",
                status="confirmed",
                price=2200.0,
            )
        ]
        store = InMemoryStore()
        graph = build_customer_service_graph(
            model_factory=lambda: model,
            preference_model_factory=lambda: FakeStructuredPreferenceModel(
                PreferenceExtractionDecision(
                    rental_related=False,
                    reason="本轮没有明确租房条件",
                )
            ),
            booking_db_factory=lambda: FakeBookingDB(orders),
            store=store,
            name="test_customer_service_order_history",
        )

        result = self._run_with_internal_state(
            graph,
            {"messages": [HumanMessage(content="查看我的订单")]},
            {"configurable": {"user_id": "lk"}},
        )

        self.assertEqual(result["customer_intent"], "order_history")
        self.assertEqual(result["user_id"], "lk")
        self.assertIn("您最近的 1 笔订单", result["messages"][-1].content)
        self.assertIn("合肥北城一号院", result["messages"][-1].content)
        self.assertFalse(result["preferences_saved"])

    def test_recommendation_subgraph_returns_answer_then_saves_preferences(
        self,
    ) -> None:
        preference_model = FakeStructuredPreferenceModel(
            PreferenceExtractionDecision(
                rental_related=True,
                city="合肥",
                budget_min=1500,
                budget_max=3000,
                reason="用户明确表达租房条件",
            )
        )
        store = InMemoryStore()
        graph = build_customer_service_graph(
            model_factory=FakeCustomerFlowModel,
            preference_model_factory=lambda: preference_model,
            rental_sql_tools_factory=lambda _model: FakeCustomerSQLTools(),
            store=store,
            name="test_full_recommendation_flow",
        )

        result = self._run_with_internal_state(
            graph,
            {
                "messages": [
                    HumanMessage(content="推荐合肥1500到3000元的房子")
                ],
            },
            {"configurable": {"user_id": "lk"}},
        )

        self.assertEqual(result["customer_intent"], "recommend_rental")
        self.assertEqual(result["messages"][-1].content, "数据库房源推荐答案")
        self.assertTrue(result["preferences_saved"])
        item = store.get(preference_namespace("lk"), PREFERENCE_STORE_KEY)
        self.assertEqual(item.value["city"], "合肥")
        self.assertEqual(item.value["budget_min"], 1500)
        self.assertEqual(item.value["budget_max"], 3000)

    def test_langgraph_config_registers_main_graph_and_postgres(self) -> None:
        config_path = Path(__file__).parents[2] / "langgraph.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(
            config["graphs"]["customer_service"],
            "./src/agent/supervisor/graph.py:customer_service_graph",
        )
        self.assertEqual(
            config["checkpointer"]["path"],
            "./src/agent/checkpointer.py:generate_checkpointer",
        )
        self.assertEqual(
            config["graphs"]["rental_recommendation_agent"],
            "./src/agent/agents/rental_recommendation.py:rental_recommendation_agent",
        )
        self.assertEqual(
            config["graphs"]["rental_booking_agent"],
            "./src/agent/agents/rental_booking.py:rental_booking_agent",
        )
        self.assertEqual(
            config["graphs"]["order_history_agent"],
            "./src/agent/agents/order_history.py:order_history_agent",
        )
        self.assertEqual(
            config["graphs"]["order_cancellation_agent"],
            "./src/agent/agents/order_cancellation.py:order_cancellation_agent",
        )


if __name__ == "__main__":
    unittest.main()
