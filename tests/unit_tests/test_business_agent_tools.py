"""房源、预订和订单原子工具的公共接口测试。"""

from __future__ import annotations

import json
import unittest

from langchain.tools import ToolRuntime

from agent.common.booking_db import BookingCreateResult, OrderRecord
from agent.tools.orders import (
    build_check_cancellation_eligibility_tool,
    build_get_order_details_tool,
    build_search_orders_tool,
)
from agent.tools.rental import (
    build_check_booking_availability_tool,
    build_find_bookable_houses_tool,
    build_get_house_details_tool,
    build_inspect_rental_market_tool,
    build_search_houses_tool,
)


def _runtime(user_id: str = "user-1") -> ToolRuntime:
    return ToolRuntime(
        state={"messages": []},
        context=None,
        config={"configurable": {"user_id": user_id}},
        stream_writer=lambda _: None,
        tool_call_id="tool-call",
        store=None,
    )


class FakeCatalog:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def inspect_market(self, **kwargs):
        self.calls.append(("inspect_market", kwargs))
        return [{"city_name": "合肥", "region_name": "蜀山", "price_min": 1800}]

    def search_houses(self, **kwargs):
        self.calls.append(("search_houses", kwargs))
        return [{"id": 7, "title": "测试公寓", "price": 1800}]

    def get_house_details(self, **kwargs):
        self.calls.append(("get_house_details", kwargs))
        return {"id": kwargs["house_id"], "title": "测试公寓", "devices": "空调"}

    def find_houses(self, **kwargs):
        self.calls.append(("find_houses", kwargs))
        return [{"id": 7, "title": "测试公寓"}]

    def check_availability(self, **kwargs):
        self.calls.append(("check_availability", kwargs))
        return {"available": True, "reason": "available", "house": {"id": 7}}


class FakeOrderDB:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.order = OrderRecord(
            order_no="order-1",
            house_id=7,
            house_title="测试公寓",
            phone="13800138000",
            check_in_date="2099-09-01",
            check_out_date="2099-09-02",
            status="confirmed",
            price=1800,
        )

    def list_recent_orders(self, **kwargs):
        self.calls.append(("list_recent_orders", kwargs))
        return [self.order]

    def search_orders(self, **kwargs):
        self.calls.append(("search_orders", kwargs))
        return [self.order]

    def get_order(self, **kwargs):
        self.calls.append(("get_order", kwargs))
        return self.order

    def create_booking(self, **kwargs):
        self.calls.append(("create_booking", kwargs))
        return BookingCreateResult(success=True, order_no="new", house_id=7)

    def cancel_booking(self, **kwargs):
        raise AssertionError("本测试不应写数据库")


class RentalToolTests(unittest.TestCase):
    def test_agent_sees_business_parameters_but_not_runtime(self) -> None:
        catalog = FakeCatalog()
        tool = build_search_houses_tool(catalog_factory=lambda: catalog)

        properties = tool.tool_call_schema.model_json_schema()["properties"]

        self.assertIn("city", properties)
        self.assertIn("budget_min", properties)
        self.assertIn("max_results", properties)
        self.assertNotIn("runtime", properties)
        self.assertNotIn("user_id", properties)

    def test_recommendation_tools_are_independent_catalog_operations(self) -> None:
        catalog = FakeCatalog()
        factory = lambda: catalog
        tools = [
            build_inspect_rental_market_tool(catalog_factory=factory),
            build_search_houses_tool(catalog_factory=factory),
            build_get_house_details_tool(catalog_factory=factory),
        ]

        market = json.loads(tools[0].func(_runtime(), "合肥", 8))
        search = json.loads(
            tools[1].func("合肥", 1500, 2500, _runtime(), ["蜀山"], None, None, 5)
        )
        details = json.loads(tools[2].func(7, _runtime()))

        self.assertEqual(market["status"], "success")
        self.assertEqual(search["houses"][0]["id"], 7)
        self.assertEqual(details["house"]["devices"], "空调")
        self.assertEqual([name for name, _ in catalog.calls], [
            "inspect_market", "search_houses", "get_house_details"
        ])

    def test_booking_discovery_and_availability_are_separate_tools(self) -> None:
        catalog = FakeCatalog()
        factory = lambda: catalog
        find_tool = build_find_bookable_houses_tool(catalog_factory=factory)
        check_tool = build_check_booking_availability_tool(catalog_factory=factory)

        found = json.loads(find_tool.func("测试公寓", _runtime(), 5))
        checked = json.loads(check_tool.func(7, "2099-09-01", "2099-09-02", _runtime()))

        self.assertEqual(found["houses"][0]["id"], 7)
        self.assertTrue(checked["available"])
        self.assertEqual([name for name, _ in catalog.calls], [
            "find_houses", "check_availability"
        ])


class OrderToolTests(unittest.TestCase):
    def test_history_tools_always_inject_runtime_user_id(self) -> None:
        database = FakeOrderDB()
        factory = lambda: database
        search_tool = build_search_orders_tool(booking_db_factory=factory)
        detail_tool = build_get_order_details_tool(booking_db_factory=factory)

        searched = json.loads(search_tool.func(_runtime("owner"), "测试", "confirmed", None, None, 5))
        detailed = json.loads(detail_tool.func("order-1", _runtime("owner")))

        self.assertEqual(searched["status"], "success")
        self.assertEqual(detailed["order"]["order_no"], "order-1")
        self.assertTrue(all(kwargs["user_id"] == "owner" for _, kwargs in database.calls))

    def test_cancellation_eligibility_does_not_write(self) -> None:
        database = FakeOrderDB()
        tool = build_check_cancellation_eligibility_tool(
            booking_db_factory=lambda: database
        )

        result = json.loads(tool.func("order-1", _runtime("owner")))

        self.assertTrue(result["eligible"])
        self.assertEqual(database.calls, [
            ("get_order", {"user_id": "owner", "order_no": "order-1"})
        ])


if __name__ == "__main__":
    unittest.main()
