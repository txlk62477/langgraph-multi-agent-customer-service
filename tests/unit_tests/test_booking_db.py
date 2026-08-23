"""PostgresBookingDB 事务逻辑的离线测试（mock psycopg 连接）。"""

from __future__ import annotations

import contextlib
import os
import unittest
from datetime import date, datetime
from unittest.mock import patch

import psycopg

from agent.common.booking_db import (
    MAX_HOUSE_CANDIDATES,
    BookingCancellationResult,
    HouseCandidate,
    PostgresBookingDB,
    normalize_house_title,
)


class FakeCursor:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = list(rows)

    def fetchone(self) -> dict | None:
        return self._rows.pop(0) if self._rows else None

    def fetchall(self) -> list[dict]:
        rows = self._rows
        self._rows = []
        return rows


class FakeConnection:
    """按执行顺序消费预设行集的假连接。"""

    def __init__(self, row_groups: list[list[dict]]) -> None:
        self._row_groups = [list(group) for group in row_groups]
        self.executed: list[tuple[str, dict | None]] = []
        self.isolation_level = None

    def execute(self, sql: str, params: dict | None = None) -> FakeCursor:
        self.executed.append((sql, params))
        group = self._row_groups.pop(0) if self._row_groups else []
        return FakeCursor(group)

    def transaction(self):
        return contextlib.nullcontext()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class RaisingInsertConnection(FakeConnection):
    """INSERT 时模拟可串行化冲突的假连接。"""

    def execute(self, sql: str, params: dict | None = None) -> FakeCursor:
        self.executed.append((sql, params))
        if "INSERT INTO booking_order" in sql:
            raise psycopg.errors.SerializationFailure("serialization failure")
        group = self._row_groups.pop(0) if self._row_groups else []
        return FakeCursor(group)


MULTIPLE_CANDIDATE_ROWS = [
    {
        "id": 11000010001,
        "title": "合租·泊澜地小区 3室2厅2卫 有地铁 房子",
        "price": 2200,
    },
    {
        "id": 11000010002,
        "title": "合租·泊澜地小区 3室2厅1卫 有地铁 房子",
        "price": 1800,
    },
    {
        "id": 11000010003,
        "title": "合租·泊澜地小区 3室2厅3卫 有地铁 房子",
        "price": 2600,
    },
]


def _house_groups(
    *,
    found: bool,
    overlap: bool,
    multiple: bool = False,
) -> list[list[dict]]:
    groups: list[list[dict]] = [[], [], []]  # ensure_schema 三条 DDL
    if found:
        if multiple:
            groups.append([])  # 规范化精确匹配无结果
            groups.append(list(MULTIPLE_CANDIDATE_ROWS))  # ILIKE 多行
        else:
            groups.append(
                [{"id": 11000010001, "title": "合肥北城一号院", "price": 2200}]
            )
    else:
        groups.append([])  # 精确匹配无结果
        groups.append([])  # ILIKE 也无结果
    groups.append([{"matched": 1}] if overlap else [])
    groups.append([])  # INSERT
    return groups


def _booking_args(**overrides: object) -> dict:
    args: dict = {
        "house_title": "合租·泊澜地小区 3室2厅",
        "phone": "13800138000",
        "check_in_date": "2026-09-01",
        "check_out_date": "2026-09-05",
        "user_id": "lk",
    }
    args.update(overrides)
    return args


class PostgresBookingDBTests(unittest.TestCase):
    def _db(self, conn: FakeConnection) -> PostgresBookingDB:
        patcher = patch(
            "agent.common.booking_db.psycopg.connect", return_value=conn
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        return PostgresBookingDB(postgres_uri="postgresql://test")

    def test_create_booking_inserts_order_in_serializable_transaction(
        self,
    ) -> None:
        conn = FakeConnection(_house_groups(found=True, overlap=False))
        db = self._db(conn)

        result = db.create_booking(**_booking_args())

        self.assertTrue(result.success)
        self.assertEqual(result.house_id, 11000010001)
        self.assertEqual(result.house_title, "合肥北城一号院")
        self.assertEqual(result.price, 2200.0)
        self.assertTrue(result.order_no)
        self.assertEqual(
            conn.isolation_level, psycopg.IsolationLevel.SERIALIZABLE
        )

        insert_sql, insert_params = conn.executed[-1]
        self.assertIn("INSERT INTO booking_order", insert_sql)
        self.assertIn("(%(order_no)s", insert_sql)
        self.assertEqual(insert_params["phone"], "13800138000")
        self.assertEqual(insert_params["user_id"], "lk")
        self.assertEqual(insert_params["house_id"], 11000010001)
        self.assertEqual(insert_params["house_title"], "合肥北城一号院")
        self.assertEqual(insert_params["check_in_date"], "2026-09-01")
        self.assertEqual(insert_params["check_out_date"], "2026-09-05")
        self.assertEqual(insert_params["price"], 2200)

    def test_multiple_candidates_returned_without_insert(self) -> None:
        conn = FakeConnection(
            _house_groups(found=True, overlap=False, multiple=True)
        )
        db = self._db(conn)

        result = db.create_booking(**_booking_args())

        self.assertFalse(result.success)
        self.assertIn("多套房源", result.error)
        self.assertEqual(len(result.candidates), 3)
        self.assertIsInstance(result.candidates[0], HouseCandidate)
        self.assertEqual(
            result.candidates[0].title,
            "合租·泊澜地小区 3室2厅2卫 有地铁 房子",
        )
        self.assertNotIn(
            "INSERT INTO booking_order",
            [sql for sql, _ in conn.executed],
        )

    def test_normalized_spacing_and_wide_chars_use_regexp_remove(self) -> None:
        conn = FakeConnection(_house_groups(found=True, overlap=False))
        db = self._db(conn)

        result = db.create_booking(
            **_booking_args(house_title=" 合租·泊澜地小区 3室２厅 ")
        )

        self.assertTrue(result.success)
        exact_sql, exact_params = conn.executed[3]  # ensure×3 之后的精确匹配
        self.assertIn("regexp_replace(title, '\\s', '', 'g')", exact_sql)
        self.assertEqual(exact_params["title"], "合租·泊澜地小区3室2厅")

    def test_like_wildcards_in_title_are_escaped(self) -> None:
        conn = FakeConnection(_house_groups(found=False, overlap=False))
        db = self._db(conn)

        result = db.create_booking(
            **_booking_args(house_title="泊澜地%小区")
        )

        self.assertFalse(result.success)
        self.assertIn("该房源不存在", result.error)
        ilike_sql, ilike_params = conn.executed[4]  # ensure×3 + 精确之后
        self.assertIn("ILIKE %(pattern)s", ilike_sql)
        self.assertEqual(ilike_params["pattern"], "%泊澜地\\%小区%")
        self.assertEqual(ilike_params["limit"], MAX_HOUSE_CANDIDATES)

    def test_date_overlap_rejects_booking_without_insert(self) -> None:
        conn = FakeConnection(_house_groups(found=True, overlap=True))
        db = self._db(conn)

        result = db.create_booking(**_booking_args())

        self.assertFalse(result.success)
        self.assertIn("已被预订", result.error)
        self.assertNotIn(
            "INSERT INTO booking_order",
            [sql for sql, _ in conn.executed],
        )
        overlap_sql = next(
            sql for sql, _ in conn.executed if "SELECT 1 FROM booking_order" in sql
        )
        self.assertIn("status = 'confirmed'", overlap_sql)

    def test_house_not_found_rejects_booking(self) -> None:
        conn = FakeConnection(_house_groups(found=False, overlap=False))
        db = self._db(conn)

        result = db.create_booking(**_booking_args())

        self.assertFalse(result.success)
        self.assertIn("该房源不存在", result.error)
        self.assertEqual(
            len([sql for sql, _ in conn.executed]),
            5,  # ensure×3 + 精确匹配 + ILIKE
        )

    def test_serialization_failure_is_reported_as_overlap(self) -> None:
        conn = RaisingInsertConnection(_house_groups(found=True, overlap=False))
        db = self._db(conn)

        result = db.create_booking(**_booking_args())

        self.assertFalse(result.success)
        self.assertIn("已被预订", result.error)

    def test_list_recent_orders_queries_by_user_and_limit(self) -> None:
        conn = FakeConnection(
            [
                [
                    {
                        "order_no": "11111111-2222-3333-4444-555555555555",
                        "house_id": 11000010001,
                        "house_title": "合肥北城一号院",
                        "phone": "13800138000",
                        "check_in_date": date(2026, 9, 1),
                        "check_out_date": date(2026, 9, 5),
                        "status": "confirmed",
                        "price": 2200,
                        "created_at": datetime(2026, 8, 19, 10, 0, 0),
                    }
                ]
            ]
        )
        db = self._db(conn)

        orders = db.list_recent_orders(user_id="lk", limit=3)

        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].order_no, "11111111-2222-3333-4444-555555555555")
        self.assertEqual(orders[0].house_title, "合肥北城一号院")
        self.assertEqual(orders[0].check_in_date, "2026-09-01")
        self.assertEqual(orders[0].check_out_date, "2026-09-05")
        self.assertEqual(orders[0].status, "confirmed")
        self.assertEqual(orders[0].price, 2200.0)
        self.assertEqual(orders[0].created_at, "2026-08-19T10:00:00")

        sql, params = conn.executed[0]
        self.assertIn("FROM booking_order", sql)
        self.assertIn("WHERE user_id = %(user_id)s", sql)
        self.assertIn("ORDER BY created_at DESC", sql)
        self.assertIn("LIMIT %(limit)s", sql)
        self.assertEqual(params["user_id"], "lk")
        self.assertEqual(params["limit"], 3)

    def test_list_recent_orders_returns_empty_when_table_missing(self) -> None:
        class MissingTableConnection(FakeConnection):
            def execute(self, sql: str, params: dict | None = None) -> FakeCursor:
                self.executed.append((sql, params))
                raise psycopg.errors.UndefinedTable("relation does not exist")

        conn = MissingTableConnection([])
        db = self._db(conn)

        orders = db.list_recent_orders(user_id="lk", limit=1)

        self.assertEqual(orders, [])

    def test_cancel_booking_soft_cancels_only_current_users_future_order(
        self,
    ) -> None:
        cancelled_at = datetime(2026, 8, 21, 12, 0, 0)
        conn = FakeConnection(
            [
                [],
                [],
                [],
                [
                    {
                        "order_no": "11111111-2222-3333-4444-555555555555",
                        "house_id": 11000010001,
                        "house_title": "合肥北城一号院",
                        "phone": "13800138000",
                        "check_in_date": date(2099, 9, 1),
                        "check_out_date": date(2099, 9, 5),
                        "status": "cancelled",
                        "price": 2200,
                        "created_at": datetime(2026, 8, 19, 10, 0, 0),
                        "cancelled_at": cancelled_at,
                    }
                ],
            ]
        )
        db = self._db(conn)

        result = db.cancel_booking(
            user_id="lk",
            order_no="11111111-2222-3333-4444-555555555555",
        )

        self.assertTrue(result.success)
        self.assertEqual(result.order.status, "cancelled")
        self.assertEqual(result.order.cancelled_at, cancelled_at.isoformat())
        update_sql, params = conn.executed[3]
        self.assertIn("SET status = 'cancelled', cancelled_at = now()", update_sql)
        self.assertIn("user_id = %(user_id)s", update_sql)
        self.assertIn("status = 'confirmed'", update_sql)
        self.assertIn("check_in_date > CURRENT_DATE", update_sql)
        self.assertEqual(params["user_id"], "lk")

    def test_cancel_booking_reports_already_cancelled_within_same_user(self) -> None:
        conn = FakeConnection(
            [
                [],
                [],
                [],
                [],
                [{"status": "cancelled", "check_in_date": date(2099, 9, 1)}],
            ]
        )
        db = self._db(conn)

        result = db.cancel_booking(user_id="lk", order_no="existing-order")

        self.assertFalse(result.success)
        self.assertEqual(result.reason, "already_cancelled")
        diagnostic_sql, params = conn.executed[4]
        self.assertIn("user_id = %(user_id)s", diagnostic_sql)
        self.assertEqual(params, {"order_no": "existing-order", "user_id": "lk"})

    def test_missing_postgres_uri_raises(self) -> None:
        with patch.dict(os.environ, {"POSTGRES_URI": ""}):
            with self.assertRaises(ValueError):
                PostgresBookingDB()

    def test_normalize_house_title_unifies_whitespace_and_width(self) -> None:
        self.assertEqual(
            normalize_house_title(" 合租·泊澜地 小区 3室２厅 "),
            "合租·泊澜地小区3室2厅",
        )
        self.assertEqual(normalize_house_title(""), "")
        self.assertEqual(
            normalize_house_title("ＡＢＣ　ＤＥＦ"), "ABCDEF"
        )


if __name__ == "__main__":
    unittest.main()
