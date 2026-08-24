"""房源目录的参数化只读查询 adapter。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from decimal import Decimal
import os
from typing import Any, Protocol

import psycopg
from psycopg.rows import dict_row


HOUSE_SUMMARY_COLUMNS = (
    "id, title, price, city_name, region_name, community_name, "
    "house_type, rooms, rent_type, area"
)
HOUSE_DETAIL_COLUMNS = (
    HOUSE_SUMMARY_COLUMNS
    + ", detail_address, floor, all_floor, intro, devices, head_image"
)


class RentalCatalog(Protocol):
    """专业 Agent 可使用的房源目录接口。"""

    def inspect_market(self, *, city: str | None, limit: int) -> list[dict[str, Any]]: ...

    def search_houses(
        self,
        *,
        city: str,
        budget_min: float,
        budget_max: float,
        districts: Sequence[str],
        room_types: Sequence[str],
        rental_mode: str | None,
        limit: int,
    ) -> list[dict[str, Any]]: ...

    def get_house_details(self, *, house_id: int) -> dict[str, Any] | None: ...

    def find_houses(self, *, query: str, limit: int) -> list[dict[str, Any]]: ...

    def check_availability(
        self,
        *,
        house_id: int,
        check_in_date: str,
        check_out_date: str,
    ) -> dict[str, Any]: ...


class PostgresRentalCatalog:
    """只执行固定模板和参数绑定的 PostgreSQL 房源目录 adapter。"""

    def __init__(self, postgres_uri: str | None = None) -> None:
        uri = (postgres_uri or os.getenv("POSTGRES_URI", "")).strip()
        if not uri:
            raise ValueError("缺少 POSTGRES_URI，无法初始化房源目录")
        self._postgres_uri = uri

    def inspect_market(
        self,
        *,
        city: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        clause = "WHERE city_name = %(city)s" if city else ""
        params: dict[str, Any] = {"limit": _limit(limit, maximum=20)}
        if city:
            params["city"] = city.strip()
        sql = f"""
            SELECT city_name, region_name, COUNT(*) AS house_count,
                   MIN(price) AS price_min, MAX(price) AS price_max
            FROM house
            {clause}
            GROUP BY city_name, region_name
            ORDER BY house_count DESC, city_name, region_name
            LIMIT %(limit)s
        """
        return self._fetch_all(sql, params)

    def search_houses(
        self,
        *,
        city: str,
        budget_min: float,
        budget_max: float,
        districts: Sequence[str],
        room_types: Sequence[str],
        rental_mode: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        clauses = [
            "city_name = %(city)s",
            "price BETWEEN %(budget_min)s AND %(budget_max)s",
        ]
        params: dict[str, Any] = {
            "city": city.strip(),
            "budget_min": budget_min,
            "budget_max": budget_max,
            "limit": _limit(limit, maximum=10),
        }
        if districts:
            clauses.append("region_name = ANY(%(districts)s)")
            params["districts"] = list(districts)
        if room_types:
            clauses.append(
                "(house_type = ANY(%(room_types)s) "
                "OR rooms::text = ANY(%(room_types)s))"
            )
            params["room_types"] = list(room_types)
        if rental_mode:
            clauses.append("rent_type = %(rental_mode)s")
            params["rental_mode"] = rental_mode
        sql = f"""
            SELECT {HOUSE_SUMMARY_COLUMNS}
            FROM house
            WHERE {' AND '.join(clauses)}
            ORDER BY price ASC, id ASC
            LIMIT %(limit)s
        """
        return self._fetch_all(sql, params)

    def get_house_details(self, *, house_id: int) -> dict[str, Any] | None:
        sql = f"SELECT {HOUSE_DETAIL_COLUMNS} FROM house WHERE id = %(house_id)s LIMIT 1"
        rows = self._fetch_all(sql, {"house_id": house_id})
        return rows[0] if rows else None

    def find_houses(self, *, query: str, limit: int) -> list[dict[str, Any]]:
        pattern = f"%{_escape_like(query.strip())}%"
        sql = f"""
            SELECT {HOUSE_SUMMARY_COLUMNS}
            FROM house
            WHERE title ILIKE %(pattern)s ESCAPE '\\'
               OR community_name ILIKE %(pattern)s ESCAPE '\\'
            ORDER BY CASE WHEN title = %(exact)s THEN 0 ELSE 1 END, id
            LIMIT %(limit)s
        """
        return self._fetch_all(
            sql,
            {
                "pattern": pattern,
                "exact": query.strip(),
                "limit": _limit(limit, maximum=5),
            },
        )

    def check_availability(
        self,
        *,
        house_id: int,
        check_in_date: str,
        check_out_date: str,
    ) -> dict[str, Any]:
        check_in = date.fromisoformat(check_in_date)
        check_out = date.fromisoformat(check_out_date)
        if check_in <= date.today() or check_out <= check_in:
            raise ValueError("预订日期范围无效")
        house = self.get_house_details(house_id=house_id)
        if house is None:
            return {"available": False, "reason": "house_not_found"}
        try:
            with psycopg.connect(self._postgres_uri, row_factory=dict_row) as connection:
                overlap = connection.execute(
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
                        "check_in_date": check_in,
                        "check_out_date": check_out,
                    },
                ).fetchone()
        except psycopg.errors.UndefinedTable:
            overlap = None
        return {
            "available": overlap is None,
            "reason": "available" if overlap is None else "date_conflict",
            "house": house,
        }

    def _fetch_all(self, sql: str, params: Mapping[str, Any]) -> list[dict[str, Any]]:
        try:
            with psycopg.connect(self._postgres_uri, row_factory=dict_row) as connection:
                rows = connection.execute(sql, dict(params)).fetchall()
        except psycopg.errors.UndefinedTable:
            return []
        return [_json_row(row) for row in rows]


def _json_row(row: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, Decimal):
            result[key] = float(value)
        elif isinstance(value, (date,)):
            result[key] = value.isoformat()
        else:
            result[key] = value
    return result


def _limit(value: int, *, maximum: int) -> int:
    return max(1, min(int(value), maximum))


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
