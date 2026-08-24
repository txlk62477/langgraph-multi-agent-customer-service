"""历史订单与取消 Agent 使用的原子工具。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import date
import re
from typing import Any

from langchain.tools import ToolRuntime, tool
from langgraph.types import interrupt

from agent.common.booking_db import BookingDB, OrderRecord, PostgresBookingDB
from agent.tools.runtime import (
    SelectionReason,
    SpecialistContext,
    json_result,
    resolve_user_id,
)


BookingDBFactory = Callable[[], BookingDB]
ORDER_STATUSES = {"confirmed", "cancelled"}


def build_list_recent_orders_tool(
    *, booking_db_factory: BookingDBFactory = PostgresBookingDB
):
    @tool("list_recent_orders")
    def list_recent_orders(
        limit: int,
        selection_reason: SelectionReason,
        runtime: ToolRuntime[SpecialistContext],
    ) -> str:
        """查询当前用户最近的订单；数量范围为1到20。"""

        del selection_reason
        try:
            orders = booking_db_factory().list_recent_orders(
                user_id=resolve_user_id(runtime),
                limit=max(1, min(int(limit), 20)),
            )
            return json_result(
                status="success" if orders else "empty",
                orders=[_public_order(order) for order in orders],
            )
        except Exception as error:
            return _failed_orders(error)

    return list_recent_orders


def build_search_orders_tool(
    *, booking_db_factory: BookingDBFactory = PostgresBookingDB
):
    @tool("search_orders")
    def search_orders(
        selection_reason: SelectionReason,
        runtime: ToolRuntime[SpecialistContext],
        house_title: str | None = None,
        status: str | None = None,
        check_in_date_start: str | None = None,
        check_in_date_end: str | None = None,
        limit: int = 10,
    ) -> str:
        """按房源名称、状态或入住日期范围筛选当前用户的订单。"""

        del selection_reason
        cleaned_status = (status or "").strip() or None
        if cleaned_status and cleaned_status not in ORDER_STATUSES:
            return json_result(status="invalid", orders=[], error="不支持的订单状态")
        try:
            if check_in_date_start:
                date.fromisoformat(check_in_date_start)
            if check_in_date_end:
                date.fromisoformat(check_in_date_end)
            if check_in_date_start and check_in_date_end:
                if date.fromisoformat(check_in_date_start) > date.fromisoformat(check_in_date_end):
                    raise ValueError("日期范围起点不能晚于终点")
            orders = booking_db_factory().search_orders(
                user_id=resolve_user_id(runtime),
                house_title=(house_title or "").strip() or None,
                status=cleaned_status,
                check_in_date_start=check_in_date_start,
                check_in_date_end=check_in_date_end,
                limit=max(1, min(int(limit), 20)),
            )
            return json_result(
                status="success" if orders else "empty",
                orders=[_public_order(order) for order in orders],
            )
        except ValueError as error:
            return json_result(status="invalid", orders=[], error=str(error))
        except Exception as error:
            return _failed_orders(error)

    return search_orders


def build_get_order_details_tool(
    *, booking_db_factory: BookingDBFactory = PostgresBookingDB
):
    @tool("get_order_details")
    def get_order_details(
        order_no: str,
        selection_reason: SelectionReason,
        runtime: ToolRuntime[SpecialistContext],
    ) -> str:
        """按明确订单号读取当前用户的一笔订单详情。"""

        del selection_reason
        try:
            order = booking_db_factory().get_order(
                user_id=resolve_user_id(runtime),
                order_no=order_no.strip(),
            )
            return json_result(
                status="success" if order else "not_found",
                order=_public_order(order) if order else None,
            )
        except Exception as error:
            return json_result(status="failed", order=None, error=_error(error))

    return get_order_details


def build_find_cancellable_orders_tool(
    *, booking_db_factory: BookingDBFactory = PostgresBookingDB
):
    @tool("find_cancellable_orders")
    def find_cancellable_orders(
        selection_reason: SelectionReason,
        runtime: ToolRuntime[SpecialistContext],
        house_title: str | None = None,
        check_in_date_start: str | None = None,
        check_in_date_end: str | None = None,
        limit: int = 5,
    ) -> str:
        """查找当前用户尚未入住且状态为 confirmed 的可取消订单候选。"""

        del selection_reason
        try:
            orders = booking_db_factory().search_orders(
                user_id=resolve_user_id(runtime),
                house_title=(house_title or "").strip() or None,
                status="confirmed",
                check_in_date_start=check_in_date_start or date.today().isoformat(),
                check_in_date_end=check_in_date_end,
                limit=max(1, min(int(limit), 5)),
            )
            eligible = [
                order
                for order in orders
                if date.fromisoformat(order.check_in_date) > date.today()
            ]
            return json_result(
                status="success" if eligible else "empty",
                orders=[_public_order(order) for order in eligible],
            )
        except Exception as error:
            return _failed_orders(error)

    return find_cancellable_orders


def build_check_cancellation_eligibility_tool(
    *, booking_db_factory: BookingDBFactory = PostgresBookingDB
):
    @tool("check_cancellation_eligibility")
    def check_cancellation_eligibility(
        order_no: str,
        selection_reason: SelectionReason,
        runtime: ToolRuntime[SpecialistContext],
    ) -> str:
        """检查当前用户指定订单是否仍可取消，并返回取消预览。"""

        del selection_reason
        try:
            order = booking_db_factory().get_order(
                user_id=resolve_user_id(runtime),
                order_no=order_no.strip(),
            )
        except Exception as error:
            return json_result(status="failed", eligible=False, error=_error(error))
        if order is None:
            return json_result(status="not_found", eligible=False)
        reason = _cancellation_reason(order)
        return json_result(
            status="success",
            eligible=reason == "eligible",
            reason=reason,
            order=_public_order(order),
        )

    return check_cancellation_eligibility


def build_cancel_order_tool(
    *, booking_db_factory: BookingDBFactory = PostgresBookingDB
):
    @tool("cancel_order")
    def cancel_order(
        order_no: str,
        selection_reason: SelectionReason,
        runtime: ToolRuntime[SpecialistContext],
    ) -> str:
        """强制二次确认后，原子软取消当前用户指定的未来订单。"""

        del selection_reason
        try:
            user_id = resolve_user_id(runtime)
            database = booking_db_factory()
            order = database.get_order(user_id=user_id, order_no=order_no.strip())
        except Exception as error:
            return json_result(status="failed", error=_error(error))
        if order is None:
            return json_result(status="not_found", error="没有找到该订单")
        reason = _cancellation_reason(order)
        if reason != "eligible":
            return json_result(status=reason, order=_public_order(order))

        answer = interrupt(
            {
                "type": "confirm_order_cancellation",
                "message": (
                    "请确认是否取消以下订单：\n"
                    f"订单号：{order.order_no}；房源：{order.house_title}；"
                    f"入住：{order.check_in_date}；退房：{order.check_out_date}。\n"
                    "回复“确认取消”后才会执行。"
                ),
                "order": _public_order(order),
            }
        )
        if not _confirmed(answer):
            return json_result(status="cancelled_by_user", order=_public_order(order))
        try:
            result = database.cancel_booking(user_id=user_id, order_no=order.order_no)
        except Exception as error:
            return json_result(status="failed", error=_error(error))
        return json_result(
            status="success" if result.success else result.reason or "failed",
            order=_public_order(result.order or order),
        )

    return cancel_order


def _cancellation_reason(order: OrderRecord) -> str:
    if order.status == "cancelled":
        return "already_cancelled"
    if order.status != "confirmed":
        return "not_cancellable"
    if date.fromisoformat(order.check_in_date) <= date.today():
        return "already_started"
    return "eligible"


def _public_order(order: OrderRecord) -> dict[str, Any]:
    return {
        "order_no": order.order_no,
        "house_id": order.house_id,
        "house_title": order.house_title,
        "check_in_date": order.check_in_date,
        "check_out_date": order.check_out_date,
        "status": order.status,
        "price": order.price,
        "created_at": order.created_at,
        "cancelled_at": order.cancelled_at,
    }


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


def _failed_orders(error: Exception) -> str:
    return json_result(status="failed", orders=[], error=_error(error))


def _error(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"
