"""房源目录 adapter 的固定查询模板和参数绑定测试。"""

from __future__ import annotations

from decimal import Decimal
import unittest
from unittest.mock import patch

from agent.common.rental_catalog import PostgresRentalCatalog


class FakeCursor:
    def __init__(self, rows):
        self.rows = list(rows)

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.rows[0] if self.rows else None


class FakeConnection:
    def __init__(self, groups):
        self.groups = list(groups)
        self.executed = []

    def execute(self, sql, params):
        self.executed.append((sql, params))
        return FakeCursor(self.groups.pop(0) if self.groups else [])

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class RentalCatalogTests(unittest.TestCase):
    def _catalog(self, connection: FakeConnection) -> PostgresRentalCatalog:
        patcher = patch(
            "agent.common.rental_catalog.psycopg.connect",
            return_value=connection,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        return PostgresRentalCatalog("postgresql://test")

    def test_search_uses_fixed_sql_and_bound_user_filters(self) -> None:
        connection = FakeConnection(
            [[{"id": 7, "title": "房源", "price": Decimal("1888.50")}]]
        )
        houses = self._catalog(connection).search_houses(
            city="上海' OR TRUE --",
            budget_min=1000,
            budget_max=3000,
            districts=["浦东%"],
            room_types=[],
            rental_mode=None,
            limit=5,
        )

        sql, params = connection.executed[0]
        self.assertNotIn("OR TRUE", sql)
        self.assertIn("city_name ILIKE %(city_pattern)s", sql)
        self.assertIn("region_name ILIKE %(district_pattern_0)s", sql)
        self.assertEqual(params["city_pattern"], "%上海' OR TRUE --%")
        self.assertEqual(params["district_pattern_0"], "%浦东\\%%")
        self.assertEqual(houses[0]["price"], 1888.5)

    def test_search_matches_city_and_district_by_escaped_keywords(self) -> None:
        connection = FakeConnection([[]])

        self._catalog(connection).search_houses(
            city="合肥",
            budget_min=1000,
            budget_max=5000,
            districts=["包河", "肥东_"],
            room_types=[],
            rental_mode=None,
            limit=5,
        )

        sql, params = connection.executed[0]
        self.assertIn("city_name ILIKE %(city_pattern)s", sql)
        self.assertIn("region_name ILIKE %(district_pattern_0)s", sql)
        self.assertIn("region_name ILIKE %(district_pattern_1)s", sql)
        self.assertEqual(params["city_pattern"], "%合肥%")
        self.assertEqual(params["district_pattern_0"], "%包河%")
        self.assertEqual(params["district_pattern_1"], "%肥东\\_%")

    def test_find_houses_escapes_like_wildcards(self) -> None:
        connection = FakeConnection([[]])
        self._catalog(connection).find_houses(query="泊澜地%_", limit=99)

        sql, params = connection.executed[0]
        self.assertIn("ILIKE %(pattern)s", sql)
        self.assertEqual(params["pattern"], "%泊澜地\\%\\_%")
        self.assertEqual(params["limit"], 5)

    def test_market_query_is_read_only_and_limited(self) -> None:
        connection = FakeConnection([[]])
        self._catalog(connection).inspect_market(city="上海", limit=100)

        sql, params = connection.executed[0]
        self.assertIn("SELECT city_name", sql)
        self.assertIn("city_name ILIKE %(city_pattern)s", sql)
        self.assertNotIn("INSERT", sql)
        self.assertNotIn("UPDATE", sql)
        self.assertEqual(params, {"city_pattern": "%上海%", "limit": 20})


if __name__ == "__main__":
    unittest.main()
