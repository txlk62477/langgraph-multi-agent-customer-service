"""信息收集功能的公共状态和示例业务状态。"""

from typing import Literal, NotRequired

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field, field_validator, model_validator


CollectionStatus = Literal["collecting", "complete", "incomplete"]


class CollectionState(MessagesState):
    """所有信息收集业务状态都需要继承的公共字段。"""

    collection_status: NotRequired[CollectionStatus]  # 当前收集阶段
    missing_required_fields: NotRequired[list[str]]  # 尚未收集的必要字段名
    llm_call_count: NotRequired[int]  # 已执行的结构化抽取次数
    max_llm_calls: NotRequired[int]  # 本业务允许的最大抽取次数
    collection_error: NotRequired[str]  # 结构化抽取失败原因


class RecommendCollectionState(CollectionState):
    """对外注册的租房推荐信息收集图所使用的最小业务状态。"""

    city: NotRequired[str | None]
    budget_min: NotRequired[int | None]
    budget_max: NotRequired[int | None]
    districts: NotRequired[list[str] | None]
    room_types: NotRequired[list[str] | None]
    rental_mode: NotRequired[str | None]


class RecommendInformation(BaseModel):
    """让 DeepSeek 按固定结构抽取租房推荐信息的输出模型。"""

    city: str | None = Field(default=None, description="租房所在城市")
    budget_min: int | None = Field(
        default=None,
        ge=0,
        description="可接受的最低月租，单位为元",
    )
    budget_max: int | None = Field(
        default=None,
        ge=0,
        description="可接受的最高月租，单位为元",
    )
    districts: list[str] | None = Field(
        default=None,
        description="一个或多个偏好的行政区或片区",
    )
    room_types: list[str] | None = Field(
        default=None,
        description="一个或多个偏好房型，例如一室一厅、两居室",
    )
    rental_mode: str | None = Field(
        default=None,
        description="租赁方式，只能是whole_rent或share_rent",
    )

    @field_validator("districts", "room_types", mode="before")
    @classmethod
    def _normalize_list(cls, value):
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("districts", "room_types")
    @classmethod
    def _clean_list(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        cleaned = list(dict.fromkeys(item.strip() for item in value if item.strip()))
        return cleaned or None

    @field_validator("rental_mode", mode="before")
    @classmethod
    def _normalize_rental_mode(cls, value):
        return {"整租": "whole_rent", "合租": "share_rent"}.get(value, value)

    @field_validator("rental_mode")
    @classmethod
    def _validate_rental_mode(cls, value: str | None) -> str | None:
        if value is not None and value not in {"whole_rent", "share_rent"}:
            raise ValueError("rental_mode只能是whole_rent或share_rent")
        return value

    @model_validator(mode="after")
    def _validate_budget_order(self) -> "RecommendInformation":
        if (
            self.budget_min is not None
            and self.budget_max is not None
            and self.budget_min > self.budget_max
        ):
            raise ValueError("最低预算不能高于最高预算")
        return self
