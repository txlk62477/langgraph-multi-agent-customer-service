"""房源推荐和预订 Agent 的原子工具。"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
import re

from langchain.tools import ToolRuntime, tool

from agent.common.booking_db import BookingDB, PostgresBookingDB
from agent.common.preferences import PREFERENCE_STORE_KEY, PreferenceProfile, preference_namespace
from agent.common.rental_catalog import PostgresRentalCatalog, RentalCatalog
from agent.tools.runtime import (
    SelectionReason,
    SpecialistContext,
    json_result,
    resolve_user_id,
)


PHONE_PATTERN = re.compile(r"^1[3-9]\d{9}$")
CatalogFactory = Callable[[], RentalCatalog]


def build_get_rental_preferences_tool():
    @tool("get_rental_preferences")
    def get_rental_preferences(
        selection_reason: SelectionReason,
        runtime: ToolRuntime[SpecialistContext],
    ) -> str:
        """读取当前用户跨会话保存的城市、区域、预算和房型偏好。"""

        del selection_reason
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
            return json_result(status="unavailable", preferences={}, error=_error(error))

    return get_rental_preferences


def build_inspect_rental_market_tool(
    *, catalog_factory: CatalogFactory = PostgresRentalCatalog
):
    @tool("inspect_rental_market")
    def inspect_rental_market(
        selection_reason: SelectionReason,
        runtime: ToolRuntime[SpecialistContext],
        city: str | None = None,
        max_regions: int = 12,
    ) -> str:
        """查看可用城市、区域、房源数量和租金范围；条件模糊或无结果时使用。"""

        del runtime, selection_reason
        try:
            rows = catalog_factory().inspect_market(
                city=(city or "").strip() or None,
                limit=max_regions,
            )
            return json_result(status="success" if rows else "empty", markets=rows)
        except Exception as error:
            return json_result(status="failed", markets=[], error=_error(error))

    return inspect_rental_market


def build_search_houses_tool(
    *, catalog_factory: CatalogFactory = PostgresRentalCatalog
):
    @tool("search_houses")
    def search_houses(
        city: str,
        budget_min: float,
        budget_max: float,
        selection_reason: SelectionReason,
        runtime: ToolRuntime[SpecialistContext],
        districts: list[str] | None = None,
        room_types: list[str] | None = None,
        rental_mode: str | None = None,
        max_results: int = 5,
    ) -> str:
        """按城市和预算查询房源，可附加区域、房型和租赁方式条件。"""

        del runtime, selection_reason
        cleaned_city = city.strip()
        if not cleaned_city:
            return json_result(status="invalid", houses=[], error="城市不能为空")
        if budget_min < 0 or budget_max < 0 or budget_min > budget_max:
            return json_result(status="invalid", houses=[], error="租金预算范围无效")
        try:
            houses = catalog_factory().search_houses(
                city=cleaned_city,
                budget_min=budget_min,
                budget_max=budget_max,
                districts=_clean_list(districts),
                room_types=_clean_list(room_types),
                rental_mode=(rental_mode or "").strip() or None,
                limit=max_results,
            )
            return json_result(status="success" if houses else "empty", houses=houses)
        except Exception as error:
            return json_result(status="failed", houses=[], error=_error(error))

    return search_houses


def build_get_house_details_tool(
    *, catalog_factory: CatalogFactory = PostgresRentalCatalog
):
    @tool("get_house_details")
    def get_house_details(
        house_id: int,
        selection_reason: SelectionReason,
        runtime: ToolRuntime[SpecialistContext],
    ) -> str:
        """读取某个候选房源的完整详情。house_id 必须来自房源工具结果。"""

        del runtime, selection_reason
        try:
            house = catalog_factory().get_house_details(house_id=house_id)
            return json_result(status="success" if house else "not_found", house=house)
        except Exception as error:
            return json_result(status="failed", house=None, error=_error(error))

    return get_house_details


def build_find_bookable_houses_tool(
    *, catalog_factory: CatalogFactory = PostgresRentalCatalog
):
    @tool("find_bookable_houses")
    def find_bookable_houses(
        query: str,
        selection_reason: SelectionReason,
        runtime: ToolRuntime[SpecialistContext],
        max_results: int = 5,
    ) -> str:
        """按房源名称或小区查找可用于后续预订的明确候选。"""

        del runtime, selection_reason
        if not query.strip():
            return json_result(status="invalid", houses=[], error="查询内容不能为空")
        try:
            houses = catalog_factory().find_houses(query=query.strip(), limit=max_results)
            return json_result(status="success" if houses else "empty", houses=houses)
        except Exception as error:
            return json_result(status="failed", houses=[], error=_error(error))

    return find_bookable_houses


def build_check_booking_availability_tool(
    *, catalog_factory: CatalogFactory = PostgresRentalCatalog
):
    @tool("check_booking_availability")
    def check_booking_availability(
        house_id: int,
        check_in_date: str,
        check_out_date: str,
        selection_reason: SelectionReason,
        runtime: ToolRuntime[SpecialistContext],
    ) -> str:
        """检查指定候选房源在目标日期范围内是否可以预订。"""

        del runtime, selection_reason
        try:
            result = catalog_factory().check_availability(
                house_id=house_id,
                check_in_date=check_in_date,
                check_out_date=check_out_date,
            )
            return json_result(status="success", **result)
        except Exception as error:
            return json_result(status="invalid", available=False, error=str(error))

    return check_booking_availability


def build_create_booking_tool(
    *, booking_db_factory: Callable[[], BookingDB] = PostgresBookingDB
):
    @tool("create_booking")
    def create_booking(
        phone: str,
        house_id: int,
        check_in_date: str,
        check_out_date: str,
        selection_reason: SelectionReason,
        runtime: ToolRuntime[SpecialistContext],
    ) -> str:
        """重新校验全部信息，并为当前用户原子创建指定房源的订单。"""

        del selection_reason
        try:
            user_id = resolve_user_id(runtime)
            if not PHONE_PATTERN.fullmatch(phone.strip()):
                raise ValueError("手机号必须是11位大陆手机号")
            check_in = date.fromisoformat(check_in_date)
            check_out = date.fromisoformat(check_out_date)
            if check_in <= date.today() or check_out <= check_in:
                raise ValueError("预订日期范围无效")
        except Exception as error:
            return json_result(status="invalid", error=str(error))
        try:
            result = booking_db_factory().create_booking(
                house_id=house_id,
                phone=phone.strip(),
                check_in_date=check_in.isoformat(),
                check_out_date=check_out.isoformat(),
                user_id=user_id,
            )
        except Exception as error:
            return json_result(status="failed", error=_error(error))
        if result.success:
            return json_result(
                status="success",
                order_no=result.order_no,
                house_id=result.house_id,
                house_title=result.house_title,
                check_in_date=check_in.isoformat(),
                check_out_date=check_out.isoformat(),
                price=result.price,
            )
        return json_result(status="rejected", error=result.error)

    return create_booking


def _clean_list(values: list[str] | None) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values or [] if value.strip()))


def _error(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"
