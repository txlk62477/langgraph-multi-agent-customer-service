"""取消订单子图的离线测试。"""

from __future__ import annotations

import unittest
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from agent.common.booking_db import (
    BookingCancellationResult,
    OrderRecord,
)
from agent.graph.order_cancellation import build_order_cancellation_graph
from agent.node.database_query import SQLQueryPlan
from agent.node.order_cancellation import (
    CancellationInformation,
    OrderCancellationNodes,
    OrderSelection,
)
from agent.state.order_cancellation import OrderCancellationState


ORDER_1 = "11111111-2222-3333-4444-555555555555"
ORDER_2 = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


class FakeCancellationModel:
    def __init__(
        self,
        information: CancellationInformation | None = None,
        selection: int | None = 2,
    ) -> None:
        self.information = information or CancellationInformation()
        self.selection = selection
        self.planner_messages = None
        self.selection_messages = None

    def with_structured_output(self, schema, *, method=None):
        owner = self

        class StructuredInvoker:
            def invoke(self, messages):
                if schema is CancellationInformation:
                    return owner.information
                if schema is OrderSelection:
                    owner.selection_messages = messages
                    return OrderSelection(selection=owner.selection)
                if schema is SQLQueryPlan:
                    owner.planner_messages = messages
                    return SQLQueryPlan.model_validate(
                        {
                            "select_columns": [
                                "order_no",
                                "user_id",
                                "house_id",
                                "house_title",
                                "check_in_date",
                                "check_out_date",
                                "status",
                                "price",
                            ],
                            "filter_groups": [
                                {
                                    "conditions": [
                                        {
                                            "column": "user_id",
                                            "operator": "eq",
                                            "value": "lk",
                                        },
                                        {
                                            "column": "status",
                                            "operator": "eq",
                                            "value": "confirmed",
                                        },
                                    ]
                                }
                            ],
                            "order_by": [
                                {"column": "created_at", "direction": "desc"}
                            ],
                        }
                    )
                raise AssertionError(f"未处理的结构化类型：{schema}")

        return StructuredInvoker()


class FakeCancellationSQLTools:
    def __init__(self, result: str) -> None:
        self.result = result
        self.executed_queries: list[str] = []

    def inspect_schema(self, table_name: str) -> str:
        return (
            f"表：{table_name}；列：order_no,user_id,house_id,house_title,"
            "check_in_date,check_out_date,status,price,created_at"
        )

    def check_query(self, query: str) -> str:
        return query

    def execute_query(self, query: str) -> str:
        self.executed_queries.append(query)
        return self.result


class FakeCancellationDB:
    def __init__(self, result: BookingCancellationResult) -> None:
        self.result = result
        self.calls: list[dict[str, str]] = []

    def cancel_booking(self, *, user_id: str, order_no: str):
        self.calls.append({"user_id": user_id, "order_no": order_no})
        return self.result


def _row(order_no: str, house_id: int, title: str, user_id: str = "lk") -> tuple:
    return (
        order_no,
        user_id,
        house_id,
        title,
        "2099-09-01",
        "2099-09-05",
        "confirmed",
        2200,
    )


def _success(order_no: str = ORDER_1, title: str = "合肥北城一号院"):
    return BookingCancellationResult(
        success=True,
        order=OrderRecord(
            order_no=order_no,
            house_id=1,
            house_title=title,
            phone="13800138000",
            check_in_date="2099-09-01",
            check_out_date="2099-09-05",
            status="cancelled",
            price=2200,
        ),
    )


class OrderCancellationGraphTests(unittest.TestCase):
    def _subgraph(self, model, tools, booking_db):
        return build_order_cancellation_graph(
            model_factory=lambda: model,
            sql_tools_factory=lambda _model: tools,
            booking_db_factory=lambda: booking_db,
            name="test_order_cancellation",
        )

    @staticmethod
    def _interruptible_graph(subgraph):
        builder = StateGraph(OrderCancellationState)
        builder.add_node("cancellation", subgraph)
        builder.add_edge(START, "cancellation")
        builder.add_edge("cancellation", END)
        return builder.compile(checkpointer=InMemorySaver())

    def test_graph_has_small_schema_and_confirmed_pipeline(self) -> None:
        graph = self._subgraph(
            FakeCancellationModel(),
            FakeCancellationSQLTools("[]"),
            FakeCancellationDB(_success()),
        )

        self.assertEqual(
            set(graph.get_input_jsonschema()["properties"]),
            {"messages", "user_id"},
        )
        self.assertEqual(
            set(graph.get_output_jsonschema()["properties"]),
            {"messages"},
        )
        self.assertTrue(
            {
                "initialize",
                "extract_order_filters",
                "prepare_order_query",
                "check_order",
                "cancel_order",
                "generate_answer",
            }
            <= set(graph.get_graph().nodes)
        )

    def test_month_filter_and_user_id_are_sent_to_query_subgraph(self) -> None:
        model = FakeCancellationModel(
            CancellationInformation(
                check_in_date_start="2099-09-01",
                check_in_date_end="2099-09-30",
            )
        )
        tools = FakeCancellationSQLTools("[]")
        graph = self._subgraph(model, tools, FakeCancellationDB(_success()))

        result = graph.invoke(
            {"messages": [HumanMessage(content="取消9月的订单")], "user_id": "lk"}
        )

        planner_request = model.planner_messages[-1].content
        self.assertIn("user_id 必须完全等于 \"lk\"", planner_request)
        self.assertIn("check_in_date 必须小于等于 2099-09-30", planner_request)
        self.assertIn("check_out_date 必须大于 2099-09-01", planner_request)
        self.assertIn("没有找到", result["messages"][-1].content)

    def test_single_order_requires_confirmation_then_soft_cancels(self) -> None:
        db = FakeCancellationDB(_success())
        subgraph = self._subgraph(
            FakeCancellationModel(),
            FakeCancellationSQLTools(repr([_row(ORDER_1, 1, "合肥北城一号院")])),
            db,
        )
        graph = self._interruptible_graph(subgraph)
        config = {"configurable": {"thread_id": str(uuid4())}}

        interrupted = graph.invoke(
            {"messages": [HumanMessage(content="取消我的订单")], "user_id": "lk"},
            config,
        )
        payload = interrupted["__interrupt__"][0].value
        self.assertEqual(payload["type"], "confirm_order_cancellation")
        self.assertEqual(db.calls, [])

        result = graph.invoke(Command(resume="确认取消"), config)

        self.assertEqual(db.calls, [{"user_id": "lk", "order_no": ORDER_1}])
        self.assertIn("订单已取消", result["messages"][-1].content)

    def test_multiple_orders_are_selected_then_confirmed(self) -> None:
        db = FakeCancellationDB(_success(ORDER_2, "上海浦东二号院"))
        rows = [
            _row(ORDER_1, 1, "合肥北城一号院"),
            _row(ORDER_2, 2, "上海浦东二号院"),
        ]
        graph = self._interruptible_graph(
            self._subgraph(
                FakeCancellationModel(),
                FakeCancellationSQLTools(repr(rows)),
                db,
            )
        )
        config = {"configurable": {"thread_id": str(uuid4())}}

        first = graph.invoke(
            {"messages": [HumanMessage(content="取消订单")], "user_id": "lk"},
            config,
        )
        self.assertEqual(
            first["__interrupt__"][0].value["type"],
            "select_order_for_cancellation",
        )

        second = graph.invoke(Command(resume="2"), config)
        self.assertEqual(
            second["__interrupt__"][0].value["type"],
            "confirm_order_cancellation",
        )
        self.assertEqual(db.calls, [])

        result = graph.invoke(Command(resume="确认取消"), config)
        self.assertEqual(db.calls, [{"user_id": "lk", "order_no": ORDER_2}])
        self.assertIn("上海浦东二号院", result["messages"][-1].content)

    def test_chinese_ordinal_is_parsed_by_llm_with_candidate_boundary(self) -> None:
        model = FakeCancellationModel(selection=2)
        nodes = OrderCancellationNodes(
            model_factory=lambda: model,
            booking_db_factory=lambda: FakeCancellationDB(_success()),
        )
        candidates = [
            {
                "order_no": ORDER_1,
                "house_title": "合肥北城一号院",
                "check_in_date": "2099-09-01",
                "check_out_date": "2099-09-05",
                "price": 2200,
            },
            {
                "order_no": ORDER_2,
                "house_title": "上海浦东二号院",
                "check_in_date": "2099-10-01",
                "check_out_date": "2099-10-05",
                "price": 2600,
            },
        ]

        selected = nodes._resolve_selection("第二个", candidates)

        self.assertEqual(selected["order_no"], ORDER_2)
        prompt = model.selection_messages[0].content
        self.assertIn("[1, 2]", prompt)
        self.assertEqual(model.selection_messages[-1].content, "第二个")

    def test_llm_selection_outside_candidate_boundary_is_rejected(self) -> None:
        class OutOfBoundaryModel(FakeCancellationModel):
            def with_structured_output(self, schema, *, method=None):
                if schema is OrderSelection:
                    class Invoker:
                        def invoke(self, _messages):
                            return {"selection": 3}

                    return Invoker()
                return super().with_structured_output(schema, method=method)

        nodes = OrderCancellationNodes(
            model_factory=OutOfBoundaryModel,
            booking_db_factory=lambda: FakeCancellationDB(_success()),
        )
        candidates = [
            {"order_no": ORDER_1, "house_title": "一号房"},
            {"order_no": ORDER_2, "house_title": "二号房"},
        ]

        self.assertIsNone(nodes._resolve_selection("第三个", candidates))

    def test_missing_user_id_fails_without_query_or_write(self) -> None:
        tools = FakeCancellationSQLTools(repr([_row(ORDER_1, 1, "测试房源")]))
        db = FakeCancellationDB(_success())
        graph = self._subgraph(FakeCancellationModel(), tools, db)

        result = graph.invoke({"messages": [HumanMessage(content="取消订单")]})

        self.assertIn("缺少用户身份", result["messages"][-1].content)
        self.assertEqual(tools.executed_queries, [])
        self.assertEqual(db.calls, [])

    def test_query_result_never_exposes_another_users_order(self) -> None:
        tools = FakeCancellationSQLTools(
            repr([_row(ORDER_1, 1, "其他用户房源", user_id="another-user")])
        )
        db = FakeCancellationDB(_success())
        graph = self._subgraph(FakeCancellationModel(), tools, db)

        result = graph.invoke(
            {"messages": [HumanMessage(content="取消订单")], "user_id": "lk"}
        )

        self.assertIn("没有找到", result["messages"][-1].content)
        self.assertNotIn("其他用户房源", result["messages"][-1].content)
        self.assertEqual(db.calls, [])

    def test_parser_accepts_database_uuid_date_and_decimal_repr(self) -> None:
        raw = (
            "[(UUID('11111111-2222-3333-4444-555555555555'), 'lk', 1, "
            "'测试房源', datetime.date(2099, 9, 1), "
            "datetime.date(2099, 9, 5), 'confirmed', Decimal('2200.00'))]"
        )

        candidates = OrderCancellationNodes._parse_candidates(raw)

        self.assertEqual(candidates[0]["order_no"], ORDER_1)
        self.assertEqual(candidates[0]["check_in_date"], "2099-09-01")
        self.assertEqual(candidates[0]["price"], 2200.0)


if __name__ == "__main__":
    unittest.main()
