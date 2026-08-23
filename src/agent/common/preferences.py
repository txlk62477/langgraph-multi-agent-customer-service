"""用户偏好的字段、校验规则和 LangGraph Store 定位约定。"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


PreferenceRentalMode = Literal["whole_rent", "share_rent"]
PreferenceField = Literal[
    "city",
    "districts",
    "budget_min",
    "budget_max",
    "room_types",
    "rental_mode",
    "commute_location",
    "max_commute_minutes",
]
PREFERENCE_FIELDS = (
    "city",
    "districts",
    "budget_min",
    "budget_max",
    "room_types",
    "rental_mode",
    "commute_location",
    "max_commute_minutes",
)
PREFERENCE_STORE_KEY = "profile"


class PreferenceUpdatesModel(BaseModel):
    """第一版允许业务子图写入的偏好字段。"""

    model_config = ConfigDict(extra="forbid")

    city: str | None = None
    districts: list[str] | None = None
    budget_min: float | None = Field(default=None, ge=0)
    budget_max: float | None = Field(default=None, ge=0)
    room_types: list[str] | None = None
    rental_mode: PreferenceRentalMode | None = None
    commute_location: str | None = None
    max_commute_minutes: int | None = Field(default=None, gt=0, le=360)

    @field_validator("city", "commute_location")
    @classmethod
    def _clean_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("文本偏好不能为空")
        return cleaned

    @field_validator("districts", "room_types")
    @classmethod
    def _clean_list(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        # 保持用户给出的顺序，同时删除空字符串和重复值。
        cleaned = list(dict.fromkeys(item.strip() for item in value if item.strip()))
        if not cleaned:
            raise ValueError("列表偏好至少需要一个有效值")
        return cleaned

    @field_validator("rental_mode", mode="before")
    @classmethod
    def _normalize_rental_mode(cls, value: Any) -> Any:
        return {"整租": "whole_rent", "合租": "share_rent"}.get(value, value)

    @field_validator("budget_min", "budget_max")
    @classmethod
    def _budget_must_be_finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("预算必须是有限数值")
        return value

    @model_validator(mode="after")
    def _validate_budget_order(self) -> "PreferenceUpdatesModel":
        if (
            self.budget_min is not None
            and self.budget_max is not None
            and self.budget_min > self.budget_max
        ):
            raise ValueError("最低预算不能高于最高预算")
        return self

    def explicit_updates(self) -> dict[str, Any]:
        """只返回调用方明确传入且非 null 的字段，不把默认值写入数据库。"""

        return self.model_dump(exclude_unset=True, exclude_none=True)


class PreferenceProfile(PreferenceUpdatesModel):
    """LangGraph Store 中一个用户的完整偏好快照。"""

    user_id: str


def validate_user_id(user_id: str) -> str:
    """清理并校验用于 Store namespace 的用户标识。"""

    cleaned = user_id.strip()
    if not cleaned:
        raise ValueError("user_id 不能为空")
    if len(cleaned) > 128:
        raise ValueError("user_id 不能超过 128 个字符")
    return cleaned


def preference_namespace(user_id: str) -> tuple[str, str, str]:
    """返回用户偏好在官方 LangGraph Store 中的固定 namespace。"""

    return ("users", validate_user_id(user_id), "preferences")
