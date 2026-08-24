"""订单数据库 adapter 的参数化查询和事务行为测试。"""

from __future__ import annotations

import contextlib
from datetime import date, datetime
import os
import unittest
from unittest.mock import patch

import psycopg

from agent.common.booking_db import PostgresBookingDB


class FakeCursor:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = list(rows)

    def fetchone(self):
        return self.rows.pop(0) if self.rows else None

    def fetchall(self):
        rows, self.rows = self.rows, []
        return rows


class FakeConnection:
    def __init__(self, groups: list[list[dict]]) -> None:
        self.groups = [list(group) for group in groups]
        self.executed: list[tuple[str, dict | None]] = []
        self.isolation_level = None

    def execute(self, sql: str, params: dict | None = None) -> FakeCursor:
        self.executed.append((sql, params))
        return FakeCursor(self.groups.pop(0) if self.groups else [])

    def transaction(self):
        return contextlib.nullcontext()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def _order_row(status: str = "confirmed") -> dict:
    return {
        "order_no": "order-1",
        "house_id": 7,
        "house_title": "测试公寓",
        "phone": "13800138000",
        "check_in_date": date(2099, 9, 1),
        "check_out_date": date(2099, 9, 2),
        "status": status,
        "price": 1800,
        "created_at": datetime(2026, 8, 20, 10),
        "cancelled_at": None,
    }


class PostgresBookingDBTests(unittest.TestCase):
    def _db(self, connection: FakeConnection) -> PostgresBookingDB:
        patcher = patch("agent.common.booking_db.psycopg.connect", return_value=connection)
        patcher.start()
        self.addCleanup(patcher.stop)
        return PostgresBookingDB("postgresql://test")

    def test_create_booking_uses_house_id_and_serializable_transaction(self) -> None:
        connection = FakeConnection(
            [[], [], [], [{"id": 7, "title": "测试公寓", "price": 1800}], [], []]
        )
        result = self._db(connection).create_booking(
            house_id=7,
            phone="13800138000",
            check_in_date="2099-09-01",
            check_out_date="2099-09-02",
            user_id="user-1",
        )

        self.assertTrue(result.success)
        self.assertEqual(result.house_id, 7)
        self.assertEqual(connection.isolation_level, psycopg.IsolationLevel.SERIALIZABLE)
        house_sql, house_params = connection.executed[3]
        self.assertIn("WHERE id = %(house_id)s", house_sql)
        self.assertEqual(house_params, {"house_id": 7})
        insert_sql, insert_params = connection.executed[-1]
        self.assertIn("INSERT INTO booking_order", insert_sql)
        self.assertEqual(insert_params["user_id"], "user-1")

    def test_create_booking_rechecks_date_conflict_before_insert(self) -> None:
        connection = FakeConnection(
            [[], [], [], [{"id": 7, "title": "测试公寓", "price": 1800}], [{"one": 1}]]
        )
        result = self._db(connection).create_booking(
            house_id=7,
            phone="13800138000",
            check_in_date="2099-09-01",
            check_out_date="2099-09-02",
            user_id="user-1",
        )

        self.assertFalse(result.success)
        self.assertIn("已被预订", result.error)
        self.assertFalse(any("INSERT INTO" in sql for sql, _ in connection.executed))

    def test_search_orders_always_scopes_to_user_and_binds_filters(self) -> None:
        connection = FakeConnection([[_order_row()]])
        orders = self._db(connection).search_orders(
            user_id="user-1",
            house_title="测试%公寓",
            status="confirmed",
            check_in_date_start="2099-01-01",
            check_in_date_end="2099-12-31",
            limit=10,
        )

        self.assertEqual(len(orders), 1)
        sql, params = connection.executed[0]
        self.assertIn("user_id = %(user_id)s", sql)
        self.assertIn("house_title ILIKE", sql)
        self.assertEqual(params["user_id"], "user-1")
        self.assertEqual(params["house_title"], "%测试\\%公寓%")

    def test_get_order_cannot_cross_user_scope(self) -> None:
        connection = FakeConnection([[]])
        order = self._db(connection).get_order(user_id="user-1", order_no="order-2")

        self.assertIsNone(order)
        sql, params = connection.executed[0]
        self.assertIn("user_id = %(user_id)s", sql)
        self.assertEqual(params, {"user_id": "user-1", "order_no": "order-2"})

    def test_cancel_booking_rechecks_user_status_and_future_date_atomically(self) -> None:
        connection = FakeConnection([[], [], [], [_order_row("cancelled")]])
        result = self._db(connection).cancel_booking(user_id="user-1", order_no="order-1")

        self.assertTrue(result.success)
        sql, params = connection.executed[3]
        self.assertIn("status = 'confirmed'", sql)
        self.assertIn("check_in_date > CURRENT_DATE", sql)
        self.assertIn("user_id = %(user_id)s", sql)
        self.assertEqual(params["user_id"], "user-1")

    def test_missing_postgres_uri_raises(self) -> None:
        with patch.dict(os.environ, {"POSTGRES_URI": ""}):
            with self.assertRaises(ValueError):
                PostgresBookingDB()


if __name__ == "__main__":
    unittest.main()
