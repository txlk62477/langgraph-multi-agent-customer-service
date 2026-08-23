"""房源推荐与预订 Agent 使用的深业务工具。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import date
from typing import Any

from langchain.tools import ToolRuntime, tool

from agent.common.booking_db import BookingDB, PostgresBookingDB
from agent.common.preferences import PREFERENCE_STORE_KEY, PreferenceProfile, preference_namespace
from agent.graph.database_query import build_database_query_graph
from agent.node.rental_booking import PHONE_PATTERN
from agent.tools.runtime import json_result, resolve_user_id


def build_get_rental_preferences_tool():
    """创建读取当前用户长期租房偏好的工具。"""

    @tool("get_rental_preferences")
    def get_rental_preferences(runtime: ToolRuntime) -> str:
        """读取当前用户跨会话保存的城市、区域、预算和房型偏好。"""

        try:
            user_id = resolve_user_id(runtime)
            if runtime.store is None:
                return json_result(status="unavailable", preferences={})
            item = runtime.store.get(preference_namespace(user_id), PREFERENCE_STORE_KEY)
            if item is None:
                return json_result(status="empty", preferences={})
            profile = PreferenceProfile.model_validate(
                {**item.value, "user_id": user_id}
            ).model_dump(mode="json", exclude_none=True, exclude={"user_id"})
            return json_result(status="success", preferences=profile)
        except Exception as error:
            return json_result(
                status="unavailable",
                preferences={},
                error=f"{type(error).__name__}: {error}",
            )

    return get_rental_preferences


def build_search_houses_tool(
    *,
    graph_factory: Callable[[], Any] | None = None,
):
    """创建受限房源查询工具；SQL 细节保持在工具实现内部。"""

    resolved_graph_factory = graph_factory or (
        lambda: build_database_query_graph(
            allowed_tables={"house"},
            name="rental_recommendation_agent_house_query",
        )
    )
    graph: Any | None = None

    @tool("search_houses")
    async def search_houses(
        city: str,
        budget_min: float,
        budget_max: float,
        runtime: ToolRuntime,
        districts: list[str] | None = None,
        room_types: list[str] | None = None,
        rental_mode: str | None = None,
        max_rows: int = 5,
    ) -> str:
        """按城市和月租预算查询房源，可附加区域、房型和租赁方式条件。"""

        nonlocal graph
        cleaned_city = city.strip()
        if not cleaned_city:
            return json_result(status="failed", error="城市不能为空")
        if budget_min < 0 or budget_max < 0 or budget_min > budget_max:
            return json_result(status="failed", error="租金预算范围无效")
        limit = max(1, min(int(max_rows), 10))
        filters = [
            f"city_name 完全等于 {cleaned_city}",
            f"price 在 {budget_min:g} 到 {budget_max:g} 之间",
        ]
        if districts:
            filters.append("region_name 包含以下任一区域：" + "、".join(districts))
        if room_types:
            filters.append("house_type 或 rooms 符合以下任一房型：" + "、".join(room_types))
        if rental_mode:
            filters.append(f"rent_type 符合租赁方式 {rental_mode}")
        request = (
            "查询符合条件的租房房源："
            + "；".join(filters)
            + "。只返回 title、price、city_name、region_name、community_name、"
            "house_type、rent_type、area、floor、all_floor、intro，按价格升序。"
        )
        try:
            if graph is None:
                graph = resolved_graph_factory()
            result = await graph.ainvoke(
                {
                    "query_request": request,
                    "table_name": "house",
                    "max_rows": limit,
                },
                config=runtime.config,
            )
            return json_result(
                status=result.get("query_status", "failed"),
                result=result.get("query_result", ""),
                error=result.get("query_error", ""),
            )
        except Exception as error:
            return json_result(
                status="failed",
                result="",
                error=f"{type(error).__name__}: {error}",
            )

    return search_houses


def build_create_booking_tool(
    *,
    booking_db_factory: Callable[[], BookingDB] = PostgresBookingDB,
):
    """创建包含输入保护和事务写入的预订工具。"""

    @tool("create_booking")
    def create_booking(
        phone: str,
        house_title: str,
        check_in_date: str,
        check_out_date: str,
        runtime: ToolRuntime,
    ) -> str:
        """校验完整预订信息，并为当前用户原子创建租房订单。"""

        cleaned_phone = phone.strip()
        cleaned_title = house_title.strip()
        try:
            user_id = resolve_user_id(runtime)
            if not PHONE_PATTERN.fullmatch(cleaned_phone):
                raise ValueError("手机号必须是11位大陆手机号")
            if not cleaned_title:
                raise ValueError("房源名称不能为空")
            check_in = date.fromisoformat(check_in_date)
            check_out = date.fromisoformat(check_out_date)
            if check_in <= date.today():
                raise ValueError("入住日期必须晚于今天")
            if check_out <= check_in:
                raise ValueError("退房日期必须晚于入住日期")
        except Exception as error:
            return json_result(status="invalid", error=str(error))

        try:
            result = booking_db_factory().create_booking(
                house_title=cleaned_title,
                phone=cleaned_phone,
                check_in_date=check_in.isoformat(),
                check_out_date=check_out.isoformat(),
                user_id=user_id,
            )
        except Exception as error:
            return json_result(
                status="failed",
                error=f"{type(error).__name__}: {error}",
            )
        if result.success:
            return json_result(
                status="success",
                order_no=result.order_no,
                house_title=result.house_title,
                check_in_date=check_in.isoformat(),
                check_out_date=check_out.isoformat(),
                price=result.price,
            )
        candidates = [
            {"house_title": item.title, "price": item.price}
            for item in result.candidates
        ]
        return json_result(
            status="multiple_candidates" if candidates else "rejected",
            error=result.error,
            candidates=candidates,
        )

    return create_booking
