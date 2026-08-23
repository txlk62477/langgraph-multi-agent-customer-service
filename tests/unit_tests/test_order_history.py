"""历史订单子图的离线测试。"""

from __future__ import annotations

from collections import deque
import unittest

from langchain_core.messages import AIMessage, HumanMessage

from agent.common.booking_db import OrderRecord
from agent.node.order_history import OrderHistoryNodes, OrderLimit
from agent.graph.order_history import build_order_history_graph


class FakeOrderHistoryModel:
    """处理 OrderLimit 抽取的假模型。"""

    def __init__(self, limits: list[int | None]) -> None:
        self._limits = deque(limits)
        self.structured_methods: list[str | None] = []
        self.limit_messages = None

    def with_structured_output(self, schema, *, method=None):
        if schema is not OrderLimit:
            raise AssertionError(f"未处理的结构化类型：{schema}")
        self.structured_methods.append(method)
        owner = self

        class StructuredInvoker:
            def invoke(self, messages):
                owner.limit_messages = messages
                return OrderLimit(limit=owner._limits.popleft())

        return StructuredInvoker()

    def invoke(self, messages):
        return AIMessage(content="占位回答")


class FakeOrderHistoryDB:
    """按调用参数返回预设订单的假数据适配器。"""

    def __init__(self, orders: list[OrderRecord]) -> None:
        self._orders = orders
        self.calls: list[dict] = []

    def create_booking(self, **kwargs):
        raise AssertionError("历史订单子图不应调用 create_booking")

    def list_recent_orders(self, *, user_id: str, limit: int) -> list[OrderRecord]:
        self.calls.append({"user_id": user_id, "limit": limit})
        return list(self._orders)[:limit]


def _order(index: int, *, order_no: str = "") -> OrderRecord:
    return OrderRecord(
        order_no=order_no or f"11111111-2222-3333-4444-{index:012d}",
        house_id=11000010000 + index,
        house_title=f"合肥北城一号院{index}栋",
        phone="13800138000",
        check_in_date=f"2026-09-0{index}",
        check_out_date=f"2026-09-1{index}",
        status="confirmed",
        price=2200.0 + index,
        created_at=f"2026-08-{index:02d}T10:00:00+08:00",
    )


class OrderHistoryNodeTests(unittest.TestCase):
    def test_format_result_empty(self) -> None:
        nodes = OrderHistoryNodes(
            booking_db_factory=lambda: FakeOrderHistoryDB([])
        )

        result = nodes.format_result({"history_status": "empty"})

        self.assertIn("您还没有历史订单", result["messages"][0].content)

    def test_format_result_failed(self) -> None:
        nodes = OrderHistoryNodes(
            booking_db_factory=lambda: FakeOrderHistoryDB([])
        )

        result = nodes.format_result(
            {"history_status": "failed", "history_error": "数据库不可用"}
        )

        self.assertIn("查询失败：数据库不可用", result["messages"][0].content)


class OrderHistoryGraphTests(unittest.TestCase):
    def _graph(self, model_factory, db):
        return build_order_history_graph(
            model_factory=model_factory,
            booking_db_factory=lambda: db,
            name="test_order_history",
        )

    def _history_input(self, content: str, *, user_id: str = "lk") -> dict:
        """历史订单子图从继承的 state.user_id 读取用户。"""

        return {"messages": [HumanMessage(content=content)], "user_id": user_id}

    @staticmethod
    def _run_with_internal_state(graph, input_state) -> dict:
        """通过 values 流读取被公开输出 Schema 隐藏的内部过程状态。"""

        states = list(graph.stream(input_state, stream_mode="values"))
        if not states:
            raise AssertionError("历史订单子图没有产生状态快照")
        return states[-1]

    def test_graph_exposes_small_public_schema(self) -> None:
        graph = self._graph(lambda: FakeOrderHistoryModel([]), FakeOrderHistoryDB([]))

        input_properties = graph.get_input_jsonschema()["properties"]
        output_properties = graph.get_output_jsonschema()["properties"]

        self.assertEqual(set(input_properties), {"messages", "user_id"})
        self.assertEqual(set(output_properties), {"messages"})

        result = graph.invoke(self._history_input("查订单"))
        self.assertEqual(set(result), {"messages"})

    def test_default_limit_returns_one_recent_order(self) -> None:
        model = FakeOrderHistoryModel([None])  # 用户未说数量
        db = FakeOrderHistoryDB([_order(1), _order(2)])
        graph = self._graph(lambda: model, db)

        result = self._run_with_internal_state(
            graph,
            self._history_input("查一下我的订单")
        )

        self.assertEqual(result["history_status"], "success")
        self.assertEqual(result["order_limit"], 1)
        self.assertEqual(db.calls[0]["user_id"], "lk")
        self.assertEqual(db.calls[0]["limit"], 1)
        content = result["messages"][-1].content
        self.assertIn("您最近的 1 笔订单", content)
        self.assertIn("合肥北城一号院1栋", content)
        self.assertIn("11111111-2222-3333-4444-000000000001", content)

    def test_user_requested_three_orders(self) -> None:
        model = FakeOrderHistoryModel([3])
        db = FakeOrderHistoryDB([_order(1), _order(2), _order(3)])
        graph = self._graph(lambda: model, db)

        result = self._run_with_internal_state(
            graph,
            self._history_input("最近的三个订单")
        )

        self.assertEqual(result["order_limit"], 3)
        self.assertEqual(db.calls[0]["limit"], 3)
        self.assertIn("您最近的 3 笔订单", result["messages"][-1].content)
        self.assertIn("合肥北城一号院3栋", result["messages"][-1].content)

    def test_no_orders_returns_empty_message(self) -> None:
        model = FakeOrderHistoryModel([None])
        db = FakeOrderHistoryDB([])
        graph = self._graph(lambda: model, db)

        result = self._run_with_internal_state(
            graph,
            self._history_input("查订单")
        )

        self.assertEqual(result["history_status"], "empty")
        self.assertIn("您还没有历史订单", result["messages"][-1].content)
        self.assertEqual(db.calls[0]["user_id"], "lk")

    def test_missing_user_id_returns_error_without_query(self) -> None:
        model = FakeOrderHistoryModel([None])
        db = FakeOrderHistoryDB([_order(1)])
        graph = self._graph(lambda: model, db)

        result = self._run_with_internal_state(
            graph,
            {"messages": [HumanMessage(content="查订单")]},
        )

        self.assertEqual(result["history_status"], "failed")
        self.assertIn("缺少用户身份", result["messages"][-1].content)
        self.assertEqual(len(db.calls), 0)

    def test_limit_extraction_failure_uses_default_limit(self) -> None:
        db = FakeOrderHistoryDB([_order(1)])
        graph = self._graph(
            lambda: (_ for _ in ()).throw(RuntimeError("模型不可用")),
            db,
        )

        result = self._run_with_internal_state(
            graph,
            self._history_input("最近的三个订单")
        )

        self.assertEqual(result["order_limit"], 1)
        self.assertEqual(db.calls[0]["limit"], 1)
        self.assertIn("您最近的 1 笔订单", result["messages"][-1].content)

    def test_booking_db_failure_returns_error(self) -> None:
        class FailingDB(FakeOrderHistoryDB):
            def list_recent_orders(self, *, user_id: str, limit: int):
                self.calls.append({"user_id": user_id, "limit": limit})
                raise RuntimeError("数据库不可用")

        model = FakeOrderHistoryModel([None])
        db = FailingDB([])
        graph = self._graph(lambda: model, db)

        result = self._run_with_internal_state(
            graph,
            self._history_input("查订单")
        )

        self.assertEqual(result["history_status"], "failed")
        self.assertIn("查询订单失败", result["messages"][-1].content)


if __name__ == "__main__":
    unittest.main()
