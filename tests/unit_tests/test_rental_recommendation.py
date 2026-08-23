"""推荐租房子图的离线测试。"""

from __future__ import annotations

from collections import deque
import unittest
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from agent.node.rental_recommendation import (
    RentalRecommendationNodes,
)
from agent.node.database_query import SQLQueryPlan
from agent.graph.rental_recommendation import build_rental_recommendation_graph
from agent.state.information_collection import RecommendInformation
from agent.state.rental_recommendation import RentalRecommendationState


class FakeRentalModel:
    def __init__(
        self,
        structured_outputs: list[dict],
        query_plans: list[dict] | None = None,
    ) -> None:
        self._structured_outputs = deque(structured_outputs)
        self._query_plans = deque(query_plans or [_house_query_plan()])
        self.final_messages = None
        self.structured_methods: list[str | None] = []

    def with_structured_output(self, schema, *, method=None):
        if schema not in {RecommendInformation, SQLQueryPlan}:
            raise AssertionError(f"未处理的结构化类型：{schema}")
        self.structured_methods.append(method)
        owner = self

        class StructuredInvoker:
            def invoke(self, _messages):
                outputs = (
                    owner._structured_outputs
                    if schema is RecommendInformation
                    else owner._query_plans
                )
                return schema.model_validate(outputs.popleft())

        return StructuredInvoker()

    def invoke(self, messages):
        self.final_messages = messages
        return AIMessage(content="这是根据数据库结果生成的推荐答案")


class FakeHouseSQLTools:
    def __init__(self, results: list[str]) -> None:
        self._results = deque(results)
        self.checked_queries: list[str] = []
        self.executed_queries: list[str] = []
        self.schema_calls = 0

    def inspect_schema(self, table_name: str) -> str:
        self.schema_calls += 1
        return f"可用表：{table_name}\nCREATE TABLE {table_name} (...)"

    def check_query(self, query: str) -> str:
        self.checked_queries.append(query)
        return query

    def execute_query(self, query: str) -> str:
        self.executed_queries.append(query)
        return self._results.popleft()


class FailingCheckSQLTools(FakeHouseSQLTools):
    def __init__(self) -> None:
        super().__init__([])

    def check_query(self, query: str) -> str:
        self.checked_queries.append(query)
        raise RuntimeError("checker unavailable")


def _result_row(title: str = "测试房源") -> str:
    return str(
        [
            (
                title,
                2200,
                "合肥",
                "庐阳区",
                "测试小区",
                "测试路1号",
                "2室1厅1卫",
                "two",
                70,
                5,
                18,
                "whole_rent",
                "近地铁",
                "空调,冰箱",
                "image.jpg",
            )
        ]
    )


def _house_query_plan(*, city: str = "合肥") -> dict:
    return {
        "select_columns": [
            "title",
            "price",
            "city_name",
            "region_name",
            "community_name",
            "detail_address",
            "house_type",
            "rooms",
            "area",
            "floor",
            "all_floor",
            "rent_type",
            "intro",
            "devices",
            "head_image",
        ],
        "filter_groups": [
            {
                "conditions": [
                    {"column": "city_name", "operator": "eq", "value": city}
                ]
            },
            {
                "conditions": [
                    {
                        "column": "price",
                        "operator": "between",
                        "value": [1500, 3000],
                    }
                ]
            },
        ],
        "order_by": [{"column": "price", "direction": "asc"}],
    }


class RentalRecommendationNodeTests(unittest.TestCase):
    def test_prefills_missing_required_fields_and_requests_confirmation(self) -> None:
        nodes = RentalRecommendationNodes()

        result = nodes.prefill_from_preferences(
            {
                "messages": [HumanMessage(content="给我推荐房子")],
                "user_preferences": {
                    "city": "合肥",
                    "budget_min": 1500,
                    "budget_max": 3000,
                    "districts": ["北城"],
                },
                "explicit_requirement_fields": [],
            }
        )

        self.assertEqual(result["city"], "合肥")
        self.assertEqual(result["budget_min"], 1500)
        self.assertEqual(result["districts"], ["北城"])
        self.assertTrue(result["needs_preference_confirmation"])

    def test_partial_preferences_go_directly_to_collection(self) -> None:
        nodes = RentalRecommendationNodes()

        result = nodes.prefill_from_preferences(
            {
                "user_preferences": {"city": "合肥"},
                "explicit_requirement_fields": [],
            }
        )

        self.assertEqual(result["city"], "合肥")
        self.assertFalse(result["needs_preference_confirmation"])

    def test_new_explicit_city_does_not_reuse_old_districts(self) -> None:
        nodes = RentalRecommendationNodes()

        result = nodes.prefill_from_preferences(
            {
                "city": "成都",
                "user_preferences": {
                    "city": "合肥",
                    "budget_min": 1500,
                    "budget_max": 3000,
                    "districts": ["北城"],
                },
                "explicit_requirement_fields": ["city"],
            }
        )

        self.assertNotIn("districts", result)
        self.assertEqual(result["budget_min"], 1500)

class RentalRecommendationGraphTests(unittest.TestCase):
    def _graph(self, model: FakeRentalModel, tools: FakeHouseSQLTools):
        return build_rental_recommendation_graph(
            model_factory=lambda: model,
            sql_tools_factory=lambda _model: tools,
            include_preference_loading=False,
            name="test_rental_recommendation",
        )

    @staticmethod
    def _run_with_internal_state(graph, input_state, config=None) -> dict:
        """运行推荐图并返回 values 流中的最终完整内部状态。"""

        states = list(
            graph.stream(
                input_state,
                config,
                stream_mode="values",
            )
        )
        if not states:
            raise AssertionError("推荐图没有产生状态快照")
        return states[-1]

    def test_graph_exposes_small_public_schema(self) -> None:
        model = FakeRentalModel([{}])
        graph = self._graph(model, FakeHouseSQLTools([]))

        input_properties = graph.get_input_jsonschema()["properties"]
        output_properties = graph.get_output_jsonschema()["properties"]

        self.assertEqual(
            set(input_properties),
            {"messages", "user_id", "user_preferences"},
        )
        self.assertEqual(
            set(output_properties),
            {"messages", "recommendation_status"},
        )

    def test_complete_request_queries_database_and_generates_answer(self) -> None:
        model = FakeRentalModel(
            [
                {
                    "city": "合肥",
                    "budget_min": 1500,
                    "budget_max": 3000,
                    "districts": ["北城"],
                }
            ]
        )
        tools = FakeHouseSQLTools([_result_row()])
        graph = self._graph(model, tools)

        result = self._run_with_internal_state(
            graph,
            {
                "messages": [
                    HumanMessage(content="推荐合肥北城1500到3000元的房子")
                ],
                "user_preferences": {},
            }
        )

        self.assertEqual(result["collection_status"], "complete")
        self.assertEqual(result["recommendation_status"], "complete")
        self.assertEqual(tools.schema_calls, 1)
        self.assertEqual(len(tools.checked_queries), 1)
        self.assertEqual(len(tools.executed_queries), 1)
        self.assertEqual(result["messages"][-1].content, "这是根据数据库结果生成的推荐答案")
        self.assertEqual(
            model.structured_methods,
            ["function_calling", "function_calling"],
        )

    def test_empty_query_returns_clear_no_matching_houses_answer(self) -> None:
        model = FakeRentalModel(
            [
                {
                    "city": "合肥",
                    "budget_min": 1500,
                    "budget_max": 3000,
                    "districts": ["北城"],
                    "room_types": ["两室一厅"],
                    "rental_mode": "whole_rent",
                }
            ]
        )
        tools = FakeHouseSQLTools([""])
        graph = self._graph(model, tools)

        result = self._run_with_internal_state(
            graph,
            {
                "messages": [HumanMessage(content="按这些条件推荐租房")],
                "user_preferences": {},
            }
        )

        self.assertEqual(result["query_status"], "empty")
        self.assertEqual(len(tools.executed_queries), 1)
        self.assertEqual(result["recommendation_status"], "no_match")
        self.assertIsInstance(result["messages"][-1], AIMessage)
        self.assertEqual(
            result["messages"][-1].content,
            (
                "没有找到符合以下条件的房源：合肥 · 北城 · "
                "预算 1500–3000 元/月 · 两室一厅 · 整租。"
                "您可以调整区域或预算后再试。"
            ),
        )

    def test_complete_preferences_interrupt_for_confirmation(self) -> None:
        model = FakeRentalModel([{}])
        tools = FakeHouseSQLTools([_result_row()])
        subgraph = self._graph(model, tools)
        parent = StateGraph(RentalRecommendationState)
        parent.add_node("recommend", subgraph)
        parent.add_edge(START, "recommend")
        parent.add_edge("recommend", END)
        graph = parent.compile(checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": str(uuid4())}}

        interrupted = graph.invoke(
            {
                "messages": [HumanMessage(content="给我推荐房子")],
                "user_preferences": {
                    "city": "合肥",
                    "budget_min": 1500,
                    "budget_max": 3000,
                },
            },
            config,
        )

        payload = interrupted["__interrupt__"][0].value
        self.assertEqual(payload["type"], "confirm_rental_preferences")
        self.assertIn("合肥", payload["message"])

        result = graph.invoke(Command(resume="确认"), config)
        self.assertEqual(result["recommendation_status"], "complete")

    def test_rejected_preferences_are_cleared_and_recollected(self) -> None:
        model = FakeRentalModel(
            [
                {},
                {"city": "成都", "budget_min": 2000, "budget_max": 3500},
            ],
            query_plans=[_house_query_plan(city="成都")],
        )
        tools = FakeHouseSQLTools([_result_row()])
        subgraph = self._graph(model, tools)
        parent = StateGraph(RentalRecommendationState)
        parent.add_node("recommend", subgraph)
        parent.add_edge(START, "recommend")
        parent.add_edge("recommend", END)
        graph = parent.compile(checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": str(uuid4())}}

        interrupted = graph.invoke(
            {
                "messages": [HumanMessage(content="给我推荐房子")],
                "user_preferences": {
                    "city": "合肥",
                    "budget_min": 1500,
                    "budget_max": 3000,
                },
            },
            config,
        )
        self.assertIn("__interrupt__", interrupted)

        result = graph.invoke(
            Command(resume="不对，改成成都，预算2000到3500"),
            config,
        )

        self.assertEqual(result["recommendation_status"], "complete")
        self.assertIn("'成都'", tools.executed_queries[0])

    def test_partial_preferences_share_one_collection_interrupt(self) -> None:
        model = FakeRentalModel([{}, {}])
        tools = FakeHouseSQLTools([])
        subgraph = self._graph(model, tools)
        parent = StateGraph(RentalRecommendationState)
        parent.add_node("recommend", subgraph)
        parent.add_edge(START, "recommend")
        parent.add_edge("recommend", END)
        graph = parent.compile(checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": str(uuid4())}}

        interrupted = graph.invoke(
            {
                "messages": [HumanMessage(content="给我推荐房子")],
                "user_preferences": {"city": "合肥"},
            },
            config,
        )

        payload = interrupted["__interrupt__"][0].value
        self.assertEqual(payload["type"], "information_required")
        self.assertIn("租房城市=合肥", payload["message"])
        self.assertEqual(
            payload["missing_required_fields"],
            ["budget_min", "budget_max"],
        )

    def test_registered_graph_contains_preference_loading_and_sql_pipeline(self) -> None:
        graph = build_rental_recommendation_graph(
            model_factory=lambda: FakeRentalModel([{}]),
            sql_tools_factory=lambda _model: FakeHouseSQLTools([]),
            include_preference_loading=True,
            name="test_registered_recommendation",
        )
        nodes = set(graph.get_graph().nodes)

        self.assertTrue(
            {
                "load_preferences",
                "information_collection",
                "prepare_house_query_request",
                "database_query",
                "respond_to_query_result",
            }
            <= nodes
        )

    def test_sql_checker_failure_retries_three_times_then_degrades(self) -> None:
        model = FakeRentalModel(
            [{"city": "合肥", "budget_min": 1500, "budget_max": 3000}],
            query_plans=[_house_query_plan(), _house_query_plan(), _house_query_plan()],
        )
        tools = FailingCheckSQLTools()
        graph = self._graph(model, tools)

        result = self._run_with_internal_state(
            graph,
            {
                "messages": [HumanMessage(content="推荐合肥1500到3000元的房子")],
                "user_preferences": {},
            }
        )

        self.assertEqual(len(tools.checked_queries), 3)
        self.assertEqual(result["query_status"], "failed")
        self.assertEqual(result["recommendation_status"], "failed")
        self.assertIsInstance(result["messages"][-1], AIMessage)
        self.assertEqual(
            result["messages"][-1].content,
            "房源查询暂时失败，请稍后重试。",
        )


if __name__ == "__main__":
    unittest.main()
