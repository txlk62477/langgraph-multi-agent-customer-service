"""预订订单的写数据库适配器：参数化 SQL 与可串行化事务。

与只读的 LangChainSQLTools 不同，本模块直接使用 psycopg 连接，
所有写操作都在一个可串行化事务内完成，防止并发下同一房源在
重叠日期段被重复预订。

房源匹配使用“规范化精确优先、规范化包含兜底”的策略：
- 精确匹配：去掉标题中所有空白后与用户输入逐字相同；
- 包含匹配：去掉标题中所有空白后包含用户输入片段；
- 包含匹配命中多套时返回候选列表，由业务层中断让用户确认，
  避免静默订到不想要的房源。
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol
from unicodedata import normalize as unicode_normalize
from uuid import uuid4

import psycopg
from psycopg.rows import dict_row


MAX_HOUSE_CANDIDATES = 5


@dataclass(frozen=True, slots=True)
class HouseCandidate:
    """包含匹配命中的房源候选。"""

    house_id: int
    title: str
    price: float | None = None


@dataclass(frozen=True, slots=True)
class BookingCreateResult:
    """一次预订事务的结果；命中多套候选时 success=False 且携带 candidates。"""

    success: bool
    order_no: str = ""
    house_id: int | None = None
    house_title: str = ""
    price: float | None = None
    error: str = ""
    candidates: tuple[HouseCandidate, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class BookingCancellationResult:
    """一次软取消事务的结果；失败时 reason 表示稳定的业务原因。"""

    success: bool
    reason: str = ""
    order: OrderRecord | None = None


@dataclass(frozen=True, slots=True)
class OrderRecord:
    """历史订单的一条查询结果。"""

    order_no: str
    house_id: int
    house_title: str
    phone: str
    check_in_date: str
    check_out_date: str
    status: str
    price: float | None = None
    created_at: str = ""
    cancelled_at: str = ""


class BookingDB(Protocol):
    """预订与历史订单子图依赖的最小数据接口，便于测试注入假实现。"""

    def create_booking(
        self,
        *,
        house_title: str,
        phone: str,
        check_in_date: str,
        check_out_date: str,
        user_id: str,
    ) -> BookingCreateResult:
        """创建一条订单；房源不存在、日期重叠或多候选时返回失败结果。"""

    def list_recent_orders(
        self,
        *,
        user_id: str,
        limit: int,
    ) -> list[OrderRecord]:
        """按创建时间倒序返回指定用户最近的订单，最多 limit 条。"""

    def cancel_booking(
        self,
        *,
        user_id: str,
        order_no: str,
    ) -> BookingCancellationResult:
        """仅软取消属于该用户且尚未入住的 confirmed 订单。"""


def normalize_house_title(value: str) -> str:
    """规范化房源名称：统一全角/半角并去掉所有空白。"""

    normalized = unicode_normalize("NFKC", value)
    return "".join(ch for ch in normalized if not ch.isspace())


def format_price(price: Any) -> str:
    """把订单/房源价格格式化为展示文本，缺省时用占位符。"""

    if isinstance(price, (int, float)):
        return f"{price:g}"
    return str(price or "—")


def _escape_like(value: str) -> str:
    """转义 LIKE 通配符，让用户输入按字面匹配。"""

    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class _BookingRejected(Exception):
    """事务内主动拒绝预订，退出时回滚事务。"""

    def __init__(
        self,
        reason: str,
        candidates: tuple[HouseCandidate, ...] = (),
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.candidates = candidates


class PostgresBookingDB:
    """基于 psycopg 的参数化 SQL 写适配器。"""

    def __init__(self, postgres_uri: str | None = None) -> None:
        uri = (postgres_uri or os.getenv("POSTGRES_URI", "")).strip()
        if not uri:
            raise ValueError("缺少 POSTGRES_URI，无法初始化预订数据库")
        self._postgres_uri = uri

    def ensure_schema(self) -> None:
        """幂等创建订单表和重叠查询索引，重复运行安全。"""

        with psycopg.connect(self._postgres_uri) as conn:
            with conn.transaction():
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS booking_order (
                        id BIGSERIAL PRIMARY KEY,
                        order_no UUID NOT NULL UNIQUE,
                        user_id TEXT NOT NULL,
                        house_id BIGINT NOT NULL,
                        house_title TEXT NOT NULL,
                        phone TEXT NOT NULL,
                        check_in_date DATE NOT NULL,
                        check_out_date DATE NOT NULL,
                        status TEXT NOT NULL DEFAULT 'confirmed',
                        price NUMERIC(12, 2),
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        cancelled_at TIMESTAMPTZ
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_booking_order_house_dates
                    ON booking_order (house_id, check_in_date, check_out_date)
                    """
                )
                # CREATE TABLE IF NOT EXISTS 不会给旧表补列，因此单独做幂等迁移。
                conn.execute(
                    """
                    ALTER TABLE booking_order
                    ADD COLUMN IF NOT EXISTS cancelled_at TIMESTAMPTZ
                    """
                )

    def create_booking(
        self,
        *,
        house_title: str,
        phone: str,
        check_in_date: str,
        check_out_date: str,
        user_id: str,
    ) -> BookingCreateResult:
        """在单个可串行化事务内完成房源匹配、日期重叠检查和订单插入。"""

        self.ensure_schema()
        order_no = str(uuid4())
        try:
            with psycopg.connect(
                self._postgres_uri, row_factory=dict_row
            ) as conn:
                conn.isolation_level = psycopg.IsolationLevel.SERIALIZABLE
                with conn.transaction():
                    houses = self._find_houses(conn, house_title)
                    if not houses:
                        raise _BookingRejected("该房源不存在")
                    if len(houses) > 1:
                        raise _BookingRejected(
                            "匹配到多套房源，请确认要预订哪一套",
                            candidates=tuple(houses),
                        )
                    house = houses[0]
                    if self._has_date_overlap(
                        conn,
                        house_id=house.house_id,
                        check_in_date=check_in_date,
                        check_out_date=check_out_date,
                    ):
                        raise _BookingRejected("该房源在所选日期已被预订")
                    conn.execute(
                        """
                        INSERT INTO booking_order
                            (order_no, user_id, house_id, house_title, phone,
                             check_in_date, check_out_date, status, price)
                        VALUES
                            (%(order_no)s, %(user_id)s, %(house_id)s,
                             %(house_title)s, %(phone)s, %(check_in_date)s,
                             %(check_out_date)s, 'confirmed', %(price)s)
                        """,
                        {
                            "order_no": order_no,
                            "user_id": user_id,
                            "house_id": house.house_id,
                            "house_title": house.title,
                            "phone": phone,
                            "check_in_date": check_in_date,
                            "check_out_date": check_out_date,
                            "price": house.price,
                        },
                    )
            return BookingCreateResult(
                success=True,
                order_no=order_no,
                house_id=house.house_id,
                house_title=house.title,
                price=house.price,
            )
        except _BookingRejected as rejected:
            return BookingCreateResult(
                success=False,
                error=rejected.reason,
                candidates=rejected.candidates,
            )
        except psycopg.errors.SerializationFailure:
            return BookingCreateResult(
                success=False, error="该房源在所选日期已被预订"
            )
        except psycopg.errors.UndefinedTable:
            return BookingCreateResult(
                success=False, error="房源表或订单表不存在，请先导入房源数据"
            )
        except psycopg.Error as error:
            return BookingCreateResult(
                success=False,
                error=f"创建订单失败：{type(error).__name__}",
            )

    def list_recent_orders(
        self,
        *,
        user_id: str,
        limit: int,
    ) -> list[OrderRecord]:
        """按创建时间倒序返回指定用户最近的订单，最多 limit 条。

        user_id 走参数化绑定，只读查询；订单表尚未创建时视为无订单。
        """

        try:
            with psycopg.connect(
                self._postgres_uri, row_factory=dict_row
            ) as conn:
                rows = conn.execute(
                    """
                    SELECT order_no, house_id, house_title, phone,
                           check_in_date, check_out_date, status, price,
                           created_at
                    FROM booking_order
                    WHERE user_id = %(user_id)s
                    ORDER BY created_at DESC
                    LIMIT %(limit)s
                    """,
                    {"user_id": user_id, "limit": limit},
                ).fetchall()
        except psycopg.errors.UndefinedTable:
            return []
        return [
            OrderRecord(
                order_no=row["order_no"],
                house_id=row["house_id"],
                house_title=row["house_title"],
                phone=row["phone"],
                check_in_date=str(row["check_in_date"]),
                check_out_date=str(row["check_out_date"]),
                status=row["status"],
                price=float(row["price"]) if row["price"] is not None else None,
                created_at=row["created_at"].isoformat()
                if row["created_at"] is not None
                else "",
                cancelled_at=row.get("cancelled_at").isoformat()
                if row.get("cancelled_at") is not None
                else "",
            )
            for row in rows
        ]

    def cancel_booking(
        self,
        *,
        user_id: str,
        order_no: str,
    ) -> BookingCancellationResult:
        """原子软取消订单，并在写入条件中再次校验归属、状态和入住日期。"""

        self.ensure_schema()
        try:
            with psycopg.connect(
                self._postgres_uri, row_factory=dict_row
            ) as conn:
                with conn.transaction():
                    row = conn.execute(
                        """
                        UPDATE booking_order
                        SET status = 'cancelled', cancelled_at = now()
                        WHERE order_no = %(order_no)s
                          AND user_id = %(user_id)s
                          AND status = 'confirmed'
                          AND check_in_date > CURRENT_DATE
                        RETURNING order_no, house_id, house_title, phone,
                                  check_in_date, check_out_date, status, price,
                                  created_at, cancelled_at
                        """,
                        {"order_no": order_no, "user_id": user_id},
                    ).fetchone()
                    if row is not None:
                        return BookingCancellationResult(
                            success=True,
                            order=_order_record_from_row(row),
                        )

                    # UPDATE 未命中时仅在同一用户边界内诊断稳定业务原因。
                    existing = conn.execute(
                        """
                        SELECT status, check_in_date
                        FROM booking_order
                        WHERE order_no = %(order_no)s
                          AND user_id = %(user_id)s
                        LIMIT 1
                        """,
                        {"order_no": order_no, "user_id": user_id},
                    ).fetchone()
                    if existing is None:
                        reason = "order_not_found"
                    elif existing["status"] == "cancelled":
                        reason = "already_cancelled"
                    elif existing["check_in_date"] <= date.today():
                        reason = "already_started"
                    else:
                        reason = "not_cancellable"
                    return BookingCancellationResult(
                        success=False,
                        reason=reason,
                    )
        except psycopg.errors.UndefinedTable:
            return BookingCancellationResult(
                success=False,
                reason="order_not_found",
            )
        except psycopg.Error as error:
            return BookingCancellationResult(
                success=False,
                reason=f"database_error:{type(error).__name__}",
            )

    @staticmethod
    def _find_houses(
        conn: psycopg.Connection,
        house_title: str,
        limit: int = MAX_HOUSE_CANDIDATES,
    ) -> list[HouseCandidate]:
        """规范化精确匹配优先，失败后规范化包含匹配（最多 limit 条）。

        两侧都去掉所有空白再比较，因此“泊澜地 小区”与“泊澜地小区”、
        “3室 2厅”与“3室2厅”视为一致；全角字符统一为半角。
        """

        norm = normalize_house_title(house_title)
        row = conn.execute(
            """
            SELECT id, title, price FROM house
            WHERE regexp_replace(title, '\\s', '', 'g') = %(title)s
            ORDER BY id LIMIT 1
            """,
            {"title": norm},
        ).fetchone()
        if row is not None:
            return [
                HouseCandidate(
                    house_id=row["id"],
                    title=row["title"],
                    price=float(row["price"]),
                )
            ]
        if not norm:
            return []

        pattern = f"%{_escape_like(norm)}%"
        rows = conn.execute(
            """
            SELECT id, title, price FROM house
            WHERE regexp_replace(title, '\\s', '', 'g') ILIKE %(pattern)s
            ESCAPE '\\'
            ORDER BY id LIMIT %(limit)s
            """,
            {"pattern": pattern, "limit": limit},
        ).fetchall()
        return [
            HouseCandidate(
                house_id=row["id"],
                title=row["title"],
                price=float(row["price"]),
            )
            for row in rows
        ]

    @staticmethod
    def _has_date_overlap(
        conn: psycopg.Connection,
        *,
        house_id: int,
        check_in_date: str,
        check_out_date: str,
    ) -> bool:
        """同一房源是否已有订单与目标日期段重叠。"""

        row = conn.execute(
            """
            SELECT 1 FROM booking_order
            WHERE house_id = %(house_id)s
              AND status = 'confirmed'
              AND check_in_date < %(check_out_date)s
              AND check_out_date > %(check_in_date)s
            LIMIT 1
            """,
            {
                "house_id": house_id,
                "check_in_date": check_in_date,
                "check_out_date": check_out_date,
            },
        ).fetchone()
        return row is not None


def _order_record_from_row(row: Mapping[str, Any]) -> OrderRecord:
    """把 psycopg 字典行稳定转换成业务订单对象。"""

    created_at = row.get("created_at")
    cancelled_at = row.get("cancelled_at")
    return OrderRecord(
        order_no=str(row["order_no"]),
        house_id=int(row["house_id"]),
        house_title=str(row["house_title"]),
        phone=str(row["phone"]),
        check_in_date=str(row["check_in_date"]),
        check_out_date=str(row["check_out_date"]),
        status=str(row["status"]),
        price=float(row["price"]) if row.get("price") is not None else None,
        created_at=created_at.isoformat() if created_at is not None else "",
        cancelled_at=(
            cancelled_at.isoformat() if cancelled_at is not None else ""
        ),
    )
