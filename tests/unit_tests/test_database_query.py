"""通用只读数据库查询子图测试。"""

from collections import deque
import json
from pathlib import Path
import unittest

from agent.graph.database_query import build_database_query_graph
from agent.node.database_query import (
    SQLQueryPlan,
    compile_select_query,
    validate_readonly_query,
)


class FakeQueryModel:
    def __init__(self, plans: list[dict]) -> None:
        self._plans = deque(plans)
        self.method = None
        self.messages = None
        self.message_history: list = []

    def with_structured_output(self, schema, *, method=None):
        if schema is not SQLQueryPlan:
            raise AssertionError(f"未处理的结构化类型：{schema}")
        self.method = method
        owner = self

        class StructuredInvoker:
            def invoke(self, messages):
                owner.messages = messages
                owner.message_history.append(messages)
                return SQLQueryPlan.model_validate(owner._plans.popleft())

        return StructuredInvoker()


class FakeSQLTools:
    def __init__(self, results: list[str]) -> None:
        self._results = deque(results)
        self.schema_tables: list[str] = []
        self.checked_queries: list[str] = []
        self.executed_queries: list[str] = []

    def inspect_schema(self, table_name: str) -> str:
        self.schema_tables.append(table_name)
        return f"CREATE TABLE {table_name} (title text, price numeric)"

    def check_query(self, query: str) -> str:
        self.checked_queries.append(query)
        return query

    def execute_query(self, query: str) -> str:
        self.executed_queries.append(query)
        return self._results.popleft()


class FailingChecker(FakeSQLTools):
    def check_query(self, query: str) -> str:
        self.checked_queries.append(query)
        raise RuntimeError("checker unavailable")


class RewritingOnceChecker(FakeSQLTools):
    def __init__(self, results: list[str]) -> None:
        super().__init__(results)
        self._should_rewrite = True

    def check_query(self, query: str) -> str:
        self.checked_queries.append(query)
        if self._should_rewrite:
            self._should_rewrite = False
            return query.replace('"price"', '"monthly_price"', 1)
        return query


class FailingSchemaInspector(FakeSQLTools):
    def inspect_schema(self, table_name: str) -> str:
        self.schema_tables.append(table_name)
        raise RuntimeError("schema unavailable")


def _plan() -> dict:
    return {
        "select_columns": ["title", "price"],
        "filter_groups": [
            {
                "logic": "and",
                "conditions": [
                    {
                        "column": "price",
                        "operator": "between",
                        "value": [1500, 3000],
                    }
                ],
            }
        ],
        "order_by": [{"column": "price", "direction": "asc"}],
    }


class DatabaseQueryCompilerTests(unittest.TestCase):
    def test_compiles_structured_plan_for_runtime_table(self) -> None:
        query = compile_select_query(
            table_name="orders",
            plan=SQLQueryPlan.model_validate(_plan()),
            max_rows=20,
        )

        self.assertIn('FROM public."orders"', query)
        self.assertIn('"price" BETWEEN 1500 AND 3000', query)
        self.assertIn("LIMIT 20", query)
        self.assertEqual(
            validate_readonly_query(
                query,
                table_name="orders",
                max_rows=20,
            ),
            query,
        )

    def test_validator_rejects_cross_table_write_join_and_excess_limit(self) -> None:
        invalid_queries = (
            'DELETE FROM public."house";',
            'SELECT * FROM public."orders" LIMIT 5;',
            'SELECT * FROM public."house" JOIN public."orders" ON true LIMIT 5;',
            'SELECT * FROM public."house" LIMIT 6;',
            'SELECT * FROM public."house"',
        )
        for query in invalid_queries:
            with self.subTest(query=query), self.assertRaises(ValueError):
                validate_readonly_query(
                    query,
                    table_name="house",
                    max_rows=5,
                )


class DatabaseQueryGraphTests(unittest.TestCase):
    @staticmethod
    def _run_with_internal_state(graph, input_state) -> dict:
        """运行查询图并返回 values 流中的最终完整内部状态。"""

        states = list(graph.stream(input_state, stream_mode="values"))
        if not states:
            raise AssertionError("数据库查询图没有产生状态快照")
        return states[-1]

    def test_graph_exposes_small_public_schema(self) -> None:
        graph = build_database_query_graph(
            allowed_tables={"house"},
            model_factory=lambda: FakeQueryModel([]),
            sql_tools_factory=lambda _model: FakeSQLTools([]),
        )

        input_properties = graph.get_input_jsonschema()["properties"]
        output_properties = graph.get_output_jsonschema()["properties"]

        self.assertEqual(
            set(input_properties),
            {"query_request", "table_name", "max_rows"},
        )
        self.assertEqual(
            set(output_properties),
            {"query_status", "query_result", "query_error"},
        )

    def test_graph_uses_merged_sql_nodes(self) -> None:
        graph = build_database_query_graph(
            allowed_tables={"house"},
            model_factory=lambda: FakeQueryModel([]),
            sql_tools_factory=lambda _model: FakeSQLTools([]),
        )

        node_names = set(graph.get_graph().nodes)

        self.assertTrue({"generate_sql", "check_sql"} <= node_names)
        self.assertTrue(
            {
                "plan_query",
                "compile_query",
                "validate_query",
                "check_query",
                "inspect_table_schema",
            }.isdisjoint(node_names)
        )

    def test_returns_toolkit_raw_result_string(self) -> None:
        model = FakeQueryModel([_plan()])
        tools = FakeSQLTools(["[('房源A', 2000)]"])
        graph = build_database_query_graph(
            allowed_tables={"house"},
            model_factory=lambda: model,
            sql_tools_factory=lambda _model: tools,
            name="test_database_query",
        )

        result = graph.invoke(
            {
                "query_request": "查询1500到3000元的房源，按价格升序",
                "table_name": "house",
                "max_rows": 5,
            }
        )

        self.assertEqual(
            set(result),
            {"query_status", "query_result", "query_error"},
        )
        self.assertEqual(result["query_status"], "success")
        self.assertEqual(result["query_result"], "[('房源A', 2000)]")
        self.assertEqual(result["query_error"], "")
        self.assertEqual(tools.schema_tables, ["house"])
        self.assertEqual(len(tools.executed_queries), 1)
        self.assertEqual(model.method, "function_calling")
        self.assertIn("查询1500到3000元", model.messages[-1].content)

    def test_house_location_filters_are_forced_to_contains(self) -> None:
        model = FakeQueryModel(
            [
                {
                    "select_columns": ["title", "price"],
                    "filter_groups": [
                        {
                            "conditions": [
                                {
                                    "column": "city_name",
                                    "operator": "eq",
                                    "value": "合肥",
                                },
                                {
                                    "column": "region_name",
                                    "operator": "eq",
                                    "value": "包河",
                                },
                                {
                                    "column": "community_name",
                                    "operator": "eq",
                                    "value": "滨湖",
                                },
                                {
                                    "column": "price",
                                    "operator": "between",
                                    "value": [500, 5000],
                                },
                            ]
                        }
                    ],
                }
            ]
        )
        tools = FakeSQLTools(["[('房源A', 2000)]"])
        graph = build_database_query_graph(
            allowed_tables={"house"},
            model_factory=lambda: model,
            sql_tools_factory=lambda _model: tools,
        )

        result = graph.invoke(
            {
                "query_request": "查询合肥包河滨湖500到5000元的房源",
                "table_name": "house",
                "max_rows": 5,
            }
        )

        self.assertEqual(result["query_status"], "success")
        query = tools.executed_queries[0]
        self.assertIn('"region_name" ILIKE', query)
        self.assertIn("'%包河%'", query)
        self.assertIn('"community_name" ILIKE', query)
        self.assertIn("'%滨湖%'", query)
        self.assertIn('"city_name" = \'合肥\'', query)

    def test_empty_result_ends_without_retry(self) -> None:
        model = FakeQueryModel([_plan()])
        tools = FakeSQLTools([""])
        graph = build_database_query_graph(
            allowed_tables={"house"},
            model_factory=lambda: model,
            sql_tools_factory=lambda _model: tools,
        )

        result = self._run_with_internal_state(
            graph,
            {
                "query_request": "查询不存在的房源",
                "table_name": "house",
            }
        )

        self.assertEqual(result["query_status"], "empty")
        self.assertEqual(result["query_attempt_status"], "success")
        self.assertEqual(result["query_attempt_count"], 1)
        self.assertEqual(len(tools.executed_queries), 1)

    def test_checker_failure_retries_three_complete_attempts(self) -> None:
        model = FakeQueryModel([_plan(), _plan(), _plan()])
        tools = FailingChecker([])
        graph = build_database_query_graph(
            allowed_tables={"house"},
            model_factory=lambda: model,
            sql_tools_factory=lambda _model: tools,
        )

        result = self._run_with_internal_state(
            graph,
            {
                "query_request": "查询房源",
                "table_name": "house",
            }
        )

        self.assertEqual(result["query_status"], "failed")
        self.assertEqual(result["query_attempt_status"], "failed")
        self.assertEqual(result["query_attempt_count"], 3)
        self.assertEqual(len(tools.checked_queries), 3)
        self.assertIn("checker unavailable", result["query_error"])

    def test_checker_rewrite_is_fed_back_to_next_generation(self) -> None:
        model = FakeQueryModel([_plan(), _plan()])
        tools = RewritingOnceChecker(["[('房源A', 2000)]"])
        graph = build_database_query_graph(
            allowed_tables={"house"},
            model_factory=lambda: model,
            sql_tools_factory=lambda _model: tools,
        )

        result = self._run_with_internal_state(
            graph,
            {
                "query_request": "查询房源",
                "table_name": "house",
            },
        )

        feedback = model.message_history[1][-1].content
        self.assertEqual(result["query_status"], "success")
        self.assertEqual(result["query_error"], "")
        self.assertEqual(result["query_attempt_count"], 2)
        self.assertEqual(tools.schema_tables, ["house", "house"])
        self.assertEqual(len(tools.executed_queries), 1)
        self.assertIn("上一次 SQL", feedback)
        self.assertIn("检查器建议修改查询", feedback)
        self.assertIn("monthly_price", feedback)

    def test_failed_attempt_skips_remaining_steps_before_retry(self) -> None:
        model = FakeQueryModel([])
        tools = FailingSchemaInspector([])
        graph = build_database_query_graph(
            allowed_tables={"house"},
            model_factory=lambda: model,
            sql_tools_factory=lambda _model: tools,
        )

        result = self._run_with_internal_state(
            graph,
            {
                "query_request": "查询房源",
                "table_name": "house",
            }
        )

        self.assertEqual(result["query_attempt_count"], 3)
        self.assertEqual(result["query_attempt_status"], "failed")
        self.assertEqual(tools.schema_tables, ["house", "house", "house"])
        self.assertIsNone(model.messages)
        self.assertEqual(tools.checked_queries, [])
        self.assertEqual(tools.executed_queries, [])
        self.assertIn("schema unavailable", result["query_error"])

    def test_runtime_cannot_expand_compile_time_table_allowlist(self) -> None:
        model = FakeQueryModel([_plan()])
        tools = FakeSQLTools([])
        graph = build_database_query_graph(
            allowed_tables={"house"},
            model_factory=lambda: model,
            sql_tools_factory=lambda _model: tools,
        )

        result = self._run_with_internal_state(
            graph,
            {
                "query_request": "查询用户",
                "table_name": "pg_user",
                "max_rows": 5,
            }
        )

        self.assertEqual(result["query_status"], "failed")
        self.assertEqual(result["query_attempt_status"], "failed")
        self.assertEqual(result["query_attempt_count"], 0)
        self.assertEqual(tools.schema_tables, [])

    def test_database_query_is_internal_tool_workflow(self) -> None:
        config_path = Path(__file__).parents[2] / "langgraph.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertNotIn("database_query", config["graphs"])
        self.assertIn("rental_recommendation_agent", config["graphs"])


if __name__ == "__main__":
    unittest.main()
