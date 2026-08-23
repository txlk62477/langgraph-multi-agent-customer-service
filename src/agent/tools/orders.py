"""订单查询与取消 Agent 使用的业务工具。"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping
from datetime import date
from typing import Any

import psycopg
from psycopg.rows import dict_row
from langchain.tools import ToolRuntime, tool
from langgraph.types import interrupt

from agent.common.booking_db import BookingDB, PostgresBookingDB
from agent.tools.runtime import json_result, resolve_user_id


def build_list_recent_orders_tool(
    *,
    booking_db_factory: Callable[[], BookingDB] = PostgresBookingDB,
):
    """创建按可信用户身份查询历史订单的工具。"""

    @tool("list_recent_orders")
    def list_recent_orders(limit: int, runtime: ToolRuntime) -> str:
        """查询当前用户最近的订单；数量范围为1到20。"""

        try:
            user_id = resolve_user_id(runtime)
            safe_limit = max(1, min(int(limit), 20))
            orders = booking_db_factory().list_recent_orders(
                user_id=user_id,
                limit=safe_limit,
            )
            return json_result(
                status="success" if orders else "empty",
                orders=[
                    {
                        "order_no": order.order_no,
                        "house_title": order.house_title,
                        "check_in_date": order.check_in_date,
                        "check_out_date": order.check_out_date,
                        "status": order.status,
                        "price": order.price,
                    }
                    for order in orders
                ],
            )
        except Exception as error:
            return json_result(
                status="failed",
                orders=[],
                error=f"{type(error).__name__}: {error}",
            )

    return list_recent_orders


def _postgres_uri() -> str:
    uri = os.getenv("POSTGRES_URI", "").strip()
    if not uri:
        raise ValueError("缺少 POSTGRES_URI，无法查询订单")
    return uri


def _find_orders(
    *,
    user_id: str,
    order_no: str | None,
    house_title: str | None,
    check_in_date_start: str | None,
    check_in_date_end: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    clauses = [
        "user_id = %(user_id)s",
        "status = 'confirmed'",
        "check_in_date > CURRENT_DATE",
    ]
    params: dict[str, Any] = {"user_id": user_id, "limit": limit}
    if order_no:
        clauses.append("order_no = %(order_no)s")
        params["order_no"] = order_no
    if house_title:
        clauses.append("house_title ILIKE %(house_title)s ESCAPE '\\'")
        escaped = house_title.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        params["house_title"] = f"%{escaped}%"
    if check_in_date_start and check_in_date_end:
        start = date.fromisoformat(check_in_date_start)
        end = date.fromisoformat(check_in_date_end)
        if start > end:
            raise ValueError("入住日期范围起点不能晚于终点")
        clauses.extend(
            [
                "check_in_date <= %(date_end)s",
                "check_out_date > %(date_start)s",
            ]
        )
        params.update(date_start=start, date_end=end)
    sql = f"""
        SELECT order_no, house_title, check_in_date, check_out_date, status, price
        FROM booking_order
        WHERE {' AND '.join(clauses)}
        ORDER BY created_at DESC
        LIMIT %(limit)s
    """
    with psycopg.connect(_postgres_uri(), row_factory=dict_row) as connection:
        rows = connection.execute(sql, params).fetchall()
    return [
        {
            "order_no": str(row["order_no"]),
            "house_title": str(row["house_title"]),
            "check_in_date": str(row["check_in_date"]),
            "check_out_date": str(row["check_out_date"]),
            "status": str(row["status"]),
            "price": float(row["price"]) if row["price"] is not None else None,
        }
        for row in rows
    ]


def build_find_cancellable_orders_tool(
    *,
    lookup: Callable[..., list[dict[str, Any]]] = _find_orders,
):
    """创建只返回当前用户未来可取消订单的查询工具。"""

    @tool("find_cancellable_orders")
    def find_cancellable_orders(
        runtime: ToolRuntime,
        order_no: str | None = None,
        house_title: str | None = None,
        check_in_date_start: str | None = None,
        check_in_date_end: str | None = None,
        limit: int = 5,
    ) -> str:
        """按订单号、房源或入住日期范围查找当前用户可取消的订单。"""

        try:
            orders = lookup(
                user_id=resolve_user_id(runtime),
                order_no=(order_no or "").strip() or None,
                house_title=(house_title or "").strip() or None,
                check_in_date_start=check_in_date_start,
                check_in_date_end=check_in_date_end,
                limit=max(1, min(int(limit), 5)),
            )
            return json_result(
                status="success" if orders else "empty",
                orders=orders,
            )
        except psycopg.errors.UndefinedTable:
            return json_result(status="empty", orders=[])
        except Exception as error:
            return json_result(
                status="failed",
                orders=[],
                error=f"{type(error).__name__}: {error}",
            )

    return find_cancellable_orders


def _confirmed(answer: Any) -> bool:
    if isinstance(answer, Mapping):
        value = answer.get("confirmed")
        if isinstance(value, bool):
            return value
        answer = answer.get("answer", "")
    if isinstance(answer, bool):
        return answer
    normalized = re.sub(r"[\s，。！!]", "", str(answer or "")).lower()
    return normalized in {"是", "确认", "确认取消", "取消吧", "yes", "y"}


def build_cancel_order_tool(
    *,
    booking_db_factory: Callable[[], BookingDB] = PostgresBookingDB,
    lookup: Callable[..., list[dict[str, Any]]] = _find_orders,
):
    """创建内部强制二次确认的订单取消工具。"""

    @tool("cancel_order")
    def cancel_order(order_no: str, runtime: ToolRuntime) -> str:
        """在代码级确认后，软取消当前用户指定的未来订单。"""

        try:
            user_id = resolve_user_id(runtime)
            matches = lookup(
                user_id=user_id,
                order_no=order_no.strip(),
                house_title=None,
                check_in_date_start=None,
                check_in_date_end=None,
                limit=1,
            )
            if not matches:
                return json_result(status="not_found", error="没有找到可取消的订单")
            order = matches[0]
        except Exception as error:
            return json_result(status="failed", error=f"{type(error).__name__}: {error}")

        answer = interrupt(
            {
                "type": "confirm_order_cancellation",
                "message": (
                    "请确认是否取消以下订单：\n"
                    f"订单号：{order['order_no']}；房源：{order['house_title']}；"
                    f"入住：{order['check_in_date']}；退房：{order['check_out_date']}。\n"
                    "回复“确认取消”后才会执行。"
                ),
                "order": order,
            }
        )
        if not _confirmed(answer):
            return json_result(status="cancelled_by_user", order=order)

        try:
            result = booking_db_factory().cancel_booking(
                user_id=user_id,
                order_no=str(order["order_no"]),
            )
        except Exception as error:
            return json_result(status="failed", error=f"{type(error).__name__}: {error}")
        if result.success:
            return json_result(status="success", order=order)
        return json_result(status=result.reason or "failed", order=order)

    return cancel_order
