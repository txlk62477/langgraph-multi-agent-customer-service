"""预订租房子图的离线测试。"""

from __future__ import annotations

from collections import deque
import unittest
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from agent.common.booking_db import BookingCreateResult, HouseCandidate
from agent.node.database_query import SQLQueryPlan
from agent.node.rental_booking import HouseSelection, RentalBookingNodes
from agent.graph.rental_booking import (
    BookingInformation,
    build_rental_booking_graph,
)
from agent.state.rental_booking import BookingState


class FakeBookingModel:
    """处理 BookingInformation、SQLQueryPlan 与 HouseSelection 的假模型。"""

    def __init__(
        self,
        structured_outputs: list[dict],
        query_plans: list[dict] | None = None,
        selections: list[int | None] | None = None,
    ) -> None:
        self._structured_outputs = deque(structured_outputs)
        self._query_plans = deque(query_plans or [_house_query_plan()])
        self._selections = deque(selections if selections is not None else [2])
        self.structured_methods: list[str | None] = []
        self.final_messages = None

    def with_structured_output(self, schema, *, method=None):
        if schema not in {BookingInformation, SQLQueryPlan, HouseSelection}:
            raise AssertionError(f"未处理的结构化类型：{schema}")
        self.structured_methods.append(method)
        owner = self

        class StructuredInvoker:
            def invoke(self, _messages):
                if schema is BookingInformation:
                    return schema.model_validate(owner._structured_outputs.popleft())
                if schema is SQLQueryPlan:
                    return schema.model_validate(owner._query_plans.popleft())
                return HouseSelection(selection=owner._selections.popleft())

        return StructuredInvoker()

    def invoke(self, messages):
        self.final_messages = messages
        return AIMessage(content="占位回答")


class FakeBookingSQLTools:
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


class FailingExecuteSQLTools(FakeBookingSQLTools):
    def __init__(self) -> None:
        super().__init__([])

    def execute_query(self, query: str) -> str:
        self.executed_queries.append(query)
        raise RuntimeError("数据库不可用")


class FakeBookingDB:
    """按预设结果响应的可注入写适配器。"""

    def __init__(self, results: list[BookingCreateResult]) -> None:
        self._results = deque(results)
        self.calls: list[dict] = []

    def create_booking(self, **kwargs) -> BookingCreateResult:
        self.calls.append(kwargs)
        return self._results.popleft()


def _house_query_plan() -> dict:
    return {
        "select_columns": ["id", "title", "price"],
        "filter_groups": [
            {
                "conditions": [
                    {"column": "title", "operator": "eq", "value": "合肥北城"}
                ]
            }
        ],
        "order_by": [{"column": "id", "direction": "asc"}],
    }


def _house_row() -> str:
    return "[{'id': 11000010001, 'title': '合肥北城一号院', 'price': 2200}]"


def _booking_input() -> dict:
    return {
        "messages": [
            HumanMessage(
                content=(
                    "我要预订合肥北城，手机号13800138000，"
                    "2026-09-01入住，2026-09-05退房"
                )
            )
        ],
        "user_id": "lk",
    }


def _booking_information(**overrides: object) -> dict:
    information: dict[str, object] = {
        "phone": "13800138000",
        "house_title": "合肥北城",
        "check_in_date": "2026-09-01",
        "check_out_date": "2026-09-05",
    }
    information.update(overrides)
    return information


def _success_result() -> BookingCreateResult:
    return BookingCreateResult(
        success=True,
        order_no="11111111-2222-3333-4444-555555555555",
        house_id=11000010001,
        house_title="合肥北城一号院",
        price=2200.0,
    )


class RentalBookingNodeTests(unittest.TestCase):
    def test_generate_answer_lists_missing_fields(self) -> None:
        nodes = RentalBookingNodes(booking_db_factory=lambda: FakeBookingDB([]))

        result = nodes.generate_answer(
            {
                "collection_status": "incomplete",
                "missing_required_fields": ["phone", "house_title"],
            }
        )

        self.assertEqual(result["booking_status"], "information_incomplete")
        self.assertIn("手机号", result["messages"][0].content)
        self.assertIn("房源名称", result["messages"][0].content)

    def test_house_selection_model_failure_returns_none(self) -> None:
        nodes = RentalBookingNodes(
            booking_db_factory=lambda: FakeBookingDB([]),
            model_factory=lambda: (_ for _ in ()).throw(
                RuntimeError("模型不可用")
            ),
        )

        selected = nodes._resolve_candidate_selection(
            "预定第二套",
            (
                HouseCandidate(
                    house_id=11000010001,
                    title="合租·泊澜地小区 3室2厅2卫 有地铁 房子",
                    price=2200.0,
                ),
                HouseCandidate(
                    house_id=11000010002,
                    title="合租·泊澜地小区 3室2厅1卫 有地铁 房子",
                    price=1800.0,
                ),
            ),
        )

        self.assertIsNone(selected)


class RentalBookingGraphTests(unittest.TestCase):
    def _graph(self, model, tools, booking_db):
        return build_rental_booking_graph(
            model_factory=lambda: model,
            booking_db_factory=lambda: booking_db,
            sql_tools_factory=lambda _model: tools,
            name="test_rental_booking",
        )

    @staticmethod
    def _run_with_internal_state(graph, input_state) -> dict:
        """通过 values 流读取被公开输出 Schema 隐藏的内部过程状态。"""

        states = list(graph.stream(input_state, stream_mode="values"))
        if not states:
            raise AssertionError("预订子图没有产生状态快照")
        return states[-1]

    def test_graph_contains_booking_pipeline(self) -> None:
        graph = self._graph(
            FakeBookingModel([{}]),
            FakeBookingSQLTools([""]),
            FakeBookingDB([]),
        )
        nodes = set(graph.get_graph().nodes)
        input_properties = graph.get_input_jsonschema()["properties"]
        output_properties = graph.get_output_jsonschema()["properties"]

        self.assertEqual(set(input_properties), {"messages", "user_id"})
        self.assertEqual(set(output_properties), {"messages"})
        self.assertTrue(
            {
                "initialize",
                "information_collection",
                "prepare_house_validation",
                "check_house",
                "create_order",
                "generate_answer",
            }
            <= nodes
        )
        self.assertTrue(
            {
                "reset",
                "collection_incomplete_answer",
                "validate_inputs",
                "prepare_house_query_request",
                "validate_house",
                "mark_house_unavailable",
                "format_result",
            }.isdisjoint(nodes)
        )

    def test_complete_booking_creates_order_and_returns_fixed_format(self) -> None:
        model = FakeBookingModel([_booking_information()])
        tools = FakeBookingSQLTools([_house_row()])
        booking_db = FakeBookingDB([_success_result()])
        graph = self._graph(model, tools, booking_db)

        result = self._run_with_internal_state(graph, _booking_input())

        self.assertEqual(result["booking_status"], "success")
        self.assertEqual(
            result["order_no"], "11111111-2222-3333-4444-555555555555"
        )
        self.assertEqual(len(tools.executed_queries), 1)
        self.assertEqual(len(booking_db.calls), 1)
        self.assertEqual(booking_db.calls[0]["house_title"], "合肥北城")
        self.assertEqual(booking_db.calls[0]["phone"], "13800138000")
        self.assertEqual(booking_db.calls[0]["check_in_date"], "2026-09-01")
        self.assertEqual(booking_db.calls[0]["check_out_date"], "2026-09-05")
        self.assertEqual(booking_db.calls[0]["user_id"], "lk")

        content = result["messages"][-1].content
        self.assertIn("预订成功", content)
        self.assertIn("11111111-2222-3333-4444-555555555555", content)
        self.assertIn("合肥北城一号院", content)
        self.assertIn("13800138000", content)
        self.assertIn("2200", content)

    def test_invalid_phone_returns_error_without_database_access(self) -> None:
        tools = FakeBookingSQLTools([])
        booking_db = FakeBookingDB([])
        graph = self._graph(
            FakeBookingModel([_booking_information(phone="12345")]),
            tools,
            booking_db,
        )

        result = self._run_with_internal_state(
            graph,
            _booking_input(),
        )

        self.assertEqual(result["booking_status"], "input_invalid")
        self.assertIn("手机号格式不正确", result["messages"][-1].content)
        self.assertEqual(len(tools.executed_queries), 0)
        self.assertEqual(len(booking_db.calls), 0)

    def test_check_out_before_check_in_returns_error(self) -> None:
        graph = self._graph(
            FakeBookingModel(
                [
                    _booking_information(
                        check_in_date="2026-09-05",
                        check_out_date="2026-09-01",
                    )
                ]
            ),
            FakeBookingSQLTools([]),
            FakeBookingDB([]),
        )

        result = self._run_with_internal_state(
            graph,
            _booking_input(),
        )

        self.assertEqual(result["booking_status"], "input_invalid")
        self.assertIn("退房日期必须晚于入住日期", result["messages"][-1].content)

    def test_invalid_date_format_returns_error(self) -> None:
        graph = self._graph(
            FakeBookingModel(
                [_booking_information(check_in_date="2026/09/01")]
            ),
            FakeBookingSQLTools([]),
            FakeBookingDB([]),
        )

        result = self._run_with_internal_state(
            graph,
            _booking_input(),
        )

        self.assertEqual(result["booking_status"], "input_invalid")
        self.assertIn("日期格式不正确", result["messages"][-1].content)

    def test_past_check_in_date_returns_error_without_database_access(self) -> None:
        graph = self._graph(
            FakeBookingModel(
                [_booking_information(check_in_date="2020-01-01", check_out_date="2020-01-05")]
            ),
            FakeBookingSQLTools([]),
            FakeBookingDB([]),
        )

        result = self._run_with_internal_state(graph, _booking_input())

        self.assertEqual(result["booking_status"], "input_invalid")
        self.assertIn("入住日期必须晚于今天", result["messages"][-1].content)

    def test_house_not_found_returns_fixed_error(self) -> None:
        tools = FakeBookingSQLTools([""])
        booking_db = FakeBookingDB([])
        graph = self._graph(
            FakeBookingModel([_booking_information()]), tools, booking_db
        )

        result = self._run_with_internal_state(graph, _booking_input())

        self.assertEqual(result["booking_status"], "house_not_found")
        self.assertIn("该房源不存在", result["messages"][-1].content)
        self.assertEqual(len(booking_db.calls), 0)

    def test_house_query_failure_retries_then_returns_not_found(self) -> None:
        tools = FailingExecuteSQLTools()
        model = FakeBookingModel(
            [_booking_information()], query_plans=[_house_query_plan()] * 3
        )
        booking_db = FakeBookingDB([])
        graph = self._graph(model, tools, booking_db)

        result = self._run_with_internal_state(graph, _booking_input())

        self.assertEqual(result["query_status"], "failed")
        self.assertEqual(result["booking_status"], "house_query_failed")
        self.assertIn("房源查询暂时不可用", result["messages"][-1].content)

    def test_date_overlap_rejected_by_booking_db(self) -> None:
        tools = FakeBookingSQLTools([_house_row()])
        booking_db = FakeBookingDB(
            [
                BookingCreateResult(
                    success=False, error="该房源在所选日期已被预订"
                )
            ]
        )
        graph = self._graph(
            FakeBookingModel([_booking_information()]), tools, booking_db
        )

        result = self._run_with_internal_state(graph, _booking_input())

        self.assertEqual(result["booking_status"], "order_failed")
        self.assertIn("已被预订", result["messages"][-1].content)
        self.assertEqual(len(tools.executed_queries), 1)

    def test_multiple_house_candidates_interrupt_for_confirmation(self) -> None:
        candidates = (
            HouseCandidate(
                house_id=11000010001,
                title="合租·泊澜地小区 3室2厅2卫 有地铁 房子",
                price=2200.0,
            ),
            HouseCandidate(
                house_id=11000010002,
                title="合租·泊澜地小区 3室2厅1卫 有地铁 房子",
                price=1800.0,
            ),
        )
        booking_db = FakeBookingDB(
            [
                BookingCreateResult(success=False, candidates=candidates),
                # interrupt 恢复时 create_order 会重放一次，需再消耗一条候选结果。
                BookingCreateResult(success=False, candidates=candidates),
                _success_result(),
            ]
        )
        tools = FakeBookingSQLTools([_house_row()])
        subgraph = self._graph(
            FakeBookingModel([_booking_information()], selections=[2]),
            tools,
            booking_db,
        )
        parent = StateGraph(BookingState)
        parent.add_node("booking", subgraph)
        parent.add_edge(START, "booking")
        parent.add_edge("booking", END)
        graph = parent.compile(checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": str(uuid4())}}

        interrupted = graph.invoke(_booking_input(), config)

        payload = interrupted["__interrupt__"][0].value
        self.assertEqual(payload["type"], "confirm_house_selection")
        self.assertEqual(len(payload["candidates"]), 2)
        self.assertIn("泊澜地", payload["message"])

        result = graph.invoke(Command(resume="预定第二套"), config)

        self.assertEqual(
            booking_db.calls[-1]["house_title"],
            "合租·泊澜地小区 3室2厅1卫 有地铁 房子",
        )
        self.assertIn("预订成功", result["messages"][-1].content)

    def test_unrecognized_house_selection_returns_error(self) -> None:
        candidates = (
            HouseCandidate(
                house_id=11000010001,
                title="合租·泊澜地小区 3室2厅2卫 有地铁 房子",
                price=2200.0,
            ),
        )
        booking_db = FakeBookingDB(
            [
                BookingCreateResult(success=False, candidates=candidates),
                BookingCreateResult(success=False, candidates=candidates),
            ]
        )
        tools = FakeBookingSQLTools([_house_row()])
        subgraph = self._graph(
            FakeBookingModel([_booking_information()], selections=[None]),
            tools,
            booking_db,
        )
        parent = StateGraph(BookingState)
        parent.add_node("booking", subgraph)
        parent.add_edge(START, "booking")
        parent.add_edge("booking", END)
        graph = parent.compile(checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": str(uuid4())}}

        interrupted = graph.invoke(_booking_input(), config)
        self.assertEqual(
            interrupted["__interrupt__"][0].value["type"],
            "confirm_house_selection",
        )

        result = graph.invoke(Command(resume="随便哪套都行"), config)

        self.assertIn("未能识别", result["messages"][-1].content)

    def test_collection_interrupt_asks_for_missing_fields_then_resumes(
        self,
    ) -> None:
        model = FakeBookingModel(
            [
                {},  # 首次抽取仍缺字段，触发中断询问
                {
                    "phone": "13900139000",
                    "house_title": "合肥北城",
                    "check_in_date": "2026-10-01",
                    "check_out_date": "2026-10-03",
                },
            ]
        )
        tools = FakeBookingSQLTools([_house_row()])
        booking_db = FakeBookingDB([_success_result()])
        subgraph = self._graph(model, tools, booking_db)
        parent = StateGraph(BookingState)
        parent.add_node("booking", subgraph)
        parent.add_edge(START, "booking")
        parent.add_edge("booking", END)
        graph = parent.compile(checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": str(uuid4())}}

        interrupted = graph.invoke(
            {
                "messages": [HumanMessage(content="我要预订合肥北城的房子")],
                "user_id": "lk",
            },
            config,
        )

        payload = interrupted["__interrupt__"][0].value
        self.assertEqual(payload["type"], "information_required")
        self.assertIn("手机号", payload["message"])

        result = graph.invoke(
            Command(
                resume=(
                    "手机号13900139000，10月1日入住，10月3日退房，"
                    "房源合肥北城"
                )
            ),
            config,
        )

        self.assertIn("预订成功", result["messages"][-1].content)
        self.assertIn(
            "11111111-2222-3333-4444-555555555555",
            result["messages"][-1].content,
        )


if __name__ == "__main__":
    unittest.main()
