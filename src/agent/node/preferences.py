"""用户偏好读取与保存节点。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import json
import os
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agent.common.llm import build_chat_model
from agent.common.preferences import (
    PREFERENCE_FIELDS,
    PREFERENCE_STORE_KEY,
    PreferenceField,
    PreferenceProfile,
    PreferenceRentalMode,
    PreferenceUpdatesModel,
    preference_namespace,
)
from agent.state.preferences import PreferenceState


ModelFactory = Callable[[], Any]


class PreferenceExtractionDecision(BaseModel):
    """DeepSeek从当前一轮对话中返回的偏好变更操作。"""

    model_config = ConfigDict(extra="forbid")

    rental_related: bool = Field(
        description="用户当前轮是否明确表达了本人的租房条件或对旧租房偏好的修改"
    )
    city: str | None = Field(default=None, description="新明确表达的租房城市")
    clear_city: bool = Field(default=False, description="是否明确删除原租房城市")
    districts_to_add: list[str] = Field(
        default_factory=list,
        description="本轮明确新增的租房区域",
    )
    districts_to_remove: list[str] = Field(
        default_factory=list,
        description="本轮明确表示不再考虑的租房区域",
    )
    clear_districts: bool = Field(
        default=False,
        description="是否明确取消全部区域偏好",
    )
    budget_min: float | None = Field(
        default=None,
        ge=0,
        description="本轮明确表达的最低月租预算，仅限租房费用",
    )
    budget_max: float | None = Field(
        default=None,
        ge=0,
        description="本轮明确表达的最高月租预算，仅限租房费用",
    )
    clear_budget_min: bool = Field(default=False, description="是否删除最低月租预算")
    clear_budget_max: bool = Field(default=False, description="是否删除最高月租预算")
    room_types_to_add: list[str] = Field(
        default_factory=list,
        description="本轮明确新增的房型偏好",
    )
    room_types_to_remove: list[str] = Field(
        default_factory=list,
        description="本轮明确表示不再考虑的房型",
    )
    clear_room_types: bool = Field(
        default=False,
        description="是否明确取消全部房型偏好",
    )
    rental_mode: PreferenceRentalMode | None = Field(
        default=None,
        description="租赁方式：whole_rent为整租，share_rent为合租",
    )
    clear_rental_mode: bool = Field(default=False, description="是否删除租赁方式偏好")
    commute_location: str | None = Field(
        default=None,
        description="本轮明确表达的通勤目的地",
    )
    clear_commute_location: bool = Field(
        default=False,
        description="是否删除通勤目的地偏好",
    )
    max_commute_minutes: int | None = Field(
        default=None,
        gt=0,
        le=360,
        description="本轮明确表达的最长通勤分钟数",
    )
    clear_max_commute_minutes: bool = Field(
        default=False,
        description="是否删除最长通勤时间偏好",
    )
    reason: str = Field(default="", description="简短说明提取依据")

    @field_validator("city", "commute_location")
    @classmethod
    def _clean_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @field_validator(
        "districts_to_add",
        "districts_to_remove",
        "room_types_to_add",
        "room_types_to_remove",
    )
    @classmethod
    def _clean_operation_list(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(item.strip() for item in value if item.strip()))

    @field_validator("rental_mode", mode="before")
    @classmethod
    def _normalize_rental_mode(cls, value: Any) -> Any:
        return {"整租": "whole_rent", "合租": "share_rent"}.get(value, value)

    @field_validator("reason")
    @classmethod
    def _clean_reason(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def _validate_budget_order(self) -> "PreferenceExtractionDecision":
        if (
            self.budget_min is not None
            and self.budget_max is not None
            and self.budget_min > self.budget_max
        ):
            raise ValueError("最低租金预算不能高于最高租金预算")
        return self


class PreferenceExtractionNodes:
    """从当前轮对话提取偏好操作，并与已加载偏好合并。"""

    def __init__(self, *, model_factory: ModelFactory = build_chat_model) -> None:
        self._model_factory = model_factory

    def extract_preference_updates(
        self,
        state: PreferenceState,
    ) -> dict[str, Any]:
        """分析当前业务轮并在提取完成后清空本轮起点。"""

        current_turn = _current_turn_messages(
            state.get("messages", []),
            state.get("current_turn_start_message_id"),
        )
        if not current_turn:
            return {
                "preference_extraction_error": "",
                "current_turn_start_message_id": None,
            }

        try:
            extraction_model = self._model_factory().with_structured_output(
                PreferenceExtractionDecision,
                method="function_calling",
            )
            decision = extraction_model.invoke(
                [
                    SystemMessage(content=_PREFERENCE_EXTRACTION_PROMPT),
                    HumanMessage(
                        content=(
                            "LangGraph Store 中已加载的偏好：\n"
                            + json.dumps(
                                state.get("user_preferences", {}),
                                ensure_ascii=False,
                                default=str,
                            )
                            + "\n\n当前轮对话：\n"
                            + _format_current_turn(current_turn)
                        )
                    ),
                ]
            )
            if not isinstance(decision, PreferenceExtractionDecision):
                decision = PreferenceExtractionDecision.model_validate(decision)
            if not decision.rental_related:
                return {
                    "preference_extraction_error": "",
                    "current_turn_start_message_id": None,
                }

            updates, clear_fields = _build_preference_delta(
                stored_preferences=state.get("user_preferences", {}),
                pending_updates=state.get("preference_updates", {}),
                pending_clear_fields=state.get("preference_clear_fields", []),
                decision=decision,
            )
            return {
                "preference_updates": updates,
                "preference_clear_fields": clear_fields,
                "preference_extraction_error": "",
                "current_turn_start_message_id": None,
            }
        except Exception as error:
            # 偏好属于附加能力；提取失败不得覆盖业务答案或阻断整轮图运行。
            return {
                "preference_extraction_error": f"{type(error).__name__}: {error}",
                "current_turn_start_message_id": None,
            }


_PREFERENCE_EXTRACTION_PROMPT = """你是租房客服的用户偏好提取器，不回答用户问题。
只分析当前业务流程起点之后的全部对话。所有用户消息都是事实来源；助手回复只帮助
理解上下文，绝不能把助手自行推荐的新地点、房型或预算当成
用户偏好。所有业务分支都可能进入本节点，但只有用户明确表达本人租房条件、修改或
删除旧租房偏好时，rental_related才为true。天气、旅游、买房、商品价格、工资、订单
金额等内容都不是租房月租预算。“帮朋友找”“假设我去某地”等第三方或假设条件也不
保存。租金预算必须明确与租房/月租有关；不要把房价、押金或其他金额写入budget。
对于“不要、取消、不考虑、改成”等表达返回相应remove或clear操作。只提取用户本轮
明确表达的字段，不从旧偏好补写字段，也不要自行推断未说出的上下限。"""


def _resolve_user_id(state: PreferenceState, config: RunnableConfig) -> str:
    """按状态、运行配置、开发环境变量的顺序确定用户。"""

    candidates = (
        state.get("user_id"),
        config.get("configurable", {}).get("user_id"),
        os.getenv("CHAT_USER_ID"),
    )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    raise ValueError("缺少 user_id，请在状态、configurable 或 CHAT_USER_ID 中提供")


def load_preferences(
    state: PreferenceState,
    config: RunnableConfig,
    runtime: Runtime[Any],
) -> dict[str, Any]:
    """从官方 LangGraph Store 读取跨 Thread 用户偏好。"""

    # 先解析身份，再访问 Store。这样即使 Store 暂时不可用，后续核心业务
    # 仍能继承 user_id 创建订单、查询订单或执行推荐。
    user_id = ""
    try:
        user_id = _resolve_user_id(state, config)
        if runtime.store is None:
            raise RuntimeError("当前运行环境没有提供 LangGraph Store")
        item = runtime.store.get(
            preference_namespace(user_id),
            PREFERENCE_STORE_KEY,
        )
        if item is None:
            profile: dict[str, Any] = {}
        else:
            # namespace 中的 user_id 是可信边界；即使有人在 Studio 手工写错
            # Value.user_id，也不能把另一名用户的标识加载进当前状态。
            profile = PreferenceProfile.model_validate(
                {**item.value, "user_id": user_id}
            ).model_dump(mode="json", exclude_none=True)
    except Exception as error:
        result: dict[str, Any] = {
            "user_preferences": {},
            "preference_load_error": f"{type(error).__name__}: {error}",
        }
        if user_id:
            result["user_id"] = user_id
        return result
    return {
        "user_id": user_id,
        "user_preferences": profile,
        "preference_load_error": "",
    }


def save_preferences(
    state: PreferenceState,
    config: RunnableConfig,
    runtime: Runtime[Any],
) -> dict[str, Any]:
    """把明确的偏好增量合并后保存到官方 LangGraph Store。"""

    raw_updates = state.get("preference_updates", {})
    clear_fields = list(dict.fromkeys(state.get("preference_clear_fields", [])))
    user_id = ""
    try:
        updates = PreferenceUpdatesModel.model_validate(
            raw_updates
        ).explicit_updates()
        if not updates and not clear_fields:
            # 没有新增偏好时不访问 Store，也不改变已有 user_preferences。
            return {
                "preferences_saved": False,
                "preference_save_error": "",
            }

        user_id = _resolve_user_id(state, config)
        if runtime.store is None:
            raise RuntimeError("当前运行环境没有提供 LangGraph Store")

        invalid_clear_fields = set(clear_fields) - set(PREFERENCE_FIELDS)
        if invalid_clear_fields:
            raise ValueError(
                "存在不支持清空的偏好字段："
                + "、".join(sorted(invalid_clear_fields))
            )

        namespace = preference_namespace(user_id)
        stored_item = runtime.store.get(namespace, PREFERENCE_STORE_KEY)
        if stored_item is None:
            merged: dict[str, Any] = {}
        else:
            stored_profile = PreferenceProfile.model_validate(
                {**stored_item.value, "user_id": user_id}
            )
            merged = stored_profile.model_dump(exclude={"user_id"})

        for field in clear_fields:
            merged[field] = None
        merged.update(updates)
        # 合并完整快照后再次校验，避免单独更新预算边界造成 min > max。
        profile = PreferenceProfile.model_validate(
            {"user_id": user_id, **merged}
        )
        profile_value = profile.model_dump(mode="json", exclude_none=True)
        runtime.store.put(
            namespace,
            PREFERENCE_STORE_KEY,
            profile_value,
            index=False,
        )
    except Exception as error:
        # 偏好是增强能力，不应让推荐、预订和订单查询失败。清空本轮增量，
        # 避免同一批坏数据在下一轮被自动重复提交。
        result = {
            "preference_updates": {},
            "preference_clear_fields": [],
            "preferences_saved": False,
            "preference_save_error": f"{type(error).__name__}: {error}",
        }
        if user_id:
            result["user_id"] = user_id
        return result

    return {
        "user_id": user_id,
        "user_preferences": profile_value,
        # 保存成功后清空增量，防止父图后续误把同一批数据再次当作新增内容。
        "preference_updates": {},
        "preference_clear_fields": [],
        "preferences_saved": True,
        "preference_save_error": "",
    }


def _current_turn_messages(
    messages: Sequence[Any],
    start_message_id: str | None,
) -> list[BaseMessage]:
    """从业务流程起点读取到末尾，忽略起点前的历史消息。"""

    if not start_message_id:
        return []

    start_index = next(
        (
            index
            for index, message in enumerate(messages)
            if getattr(message, "id", None) == start_message_id
        ),
        None,
    )
    if start_index is None:
        return []

    return [
        message
        for message in messages[start_index:]
        if isinstance(message, (HumanMessage, AIMessage))
        and isinstance(message.content, str)
        and message.content.strip()
    ]


def _format_current_turn(messages: Sequence[BaseMessage]) -> str:
    """用明确角色标记格式化当前轮，防止模型混淆用户和助手内容。"""

    return "\n".join(
        f"{'用户' if isinstance(message, HumanMessage) else '助手'}：{message.content}"
        for message in messages
    )


def _build_preference_delta(
    *,
    stored_preferences: dict[str, Any],
    pending_updates: dict[str, Any],
    pending_clear_fields: Sequence[PreferenceField],
    decision: PreferenceExtractionDecision,
) -> tuple[dict[str, Any], list[PreferenceField]]:
    """应用提取操作，返回相对 Store 快照的最小更新与清空字段。"""

    stored = {
        field: stored_preferences.get(field)
        for field in PREFERENCE_FIELDS
    }
    desired = dict(stored)

    # 兼容业务子图已经明确产生的偏好增量，再叠加当前提取结果。
    for field in pending_clear_fields:
        desired[field] = None
    desired.update(
        PreferenceUpdatesModel.model_validate(pending_updates).explicit_updates()
    )

    old_city = _optional_string(desired.get("city"))
    if decision.city is not None:
        desired["city"] = decision.city
        if old_city != decision.city:
            # 区域从属于城市；换城市时不能继续携带旧城市的区域。
            desired["districts"] = None
    elif decision.clear_city:
        desired["city"] = None
        desired["districts"] = None

    _apply_list_operations(
        desired,
        field="districts",
        additions=decision.districts_to_add,
        removals=decision.districts_to_remove,
        clear=decision.clear_districts,
    )
    _apply_list_operations(
        desired,
        field="room_types",
        additions=decision.room_types_to_add,
        removals=decision.room_types_to_remove,
        clear=decision.clear_room_types,
    )

    _apply_scalar_operation(
        desired,
        field="budget_min",
        value=decision.budget_min,
        clear=decision.clear_budget_min,
    )
    _apply_scalar_operation(
        desired,
        field="budget_max",
        value=decision.budget_max,
        clear=decision.clear_budget_max,
    )
    _apply_scalar_operation(
        desired,
        field="rental_mode",
        value=decision.rental_mode,
        clear=decision.clear_rental_mode,
    )
    _apply_scalar_operation(
        desired,
        field="commute_location",
        value=decision.commute_location,
        clear=decision.clear_commute_location,
    )
    _apply_scalar_operation(
        desired,
        field="max_commute_minutes",
        value=decision.max_commute_minutes,
        clear=decision.clear_max_commute_minutes,
    )

    # 新预算边界与旧边界冲突时，以本轮明确表达的新边界为准，清除旧边界。
    budget_min = desired.get("budget_min")
    budget_max = desired.get("budget_max")
    if budget_min is not None and budget_max is not None and budget_min > budget_max:
        if decision.budget_min is not None and decision.budget_max is None:
            desired["budget_max"] = None
        elif decision.budget_max is not None and decision.budget_min is None:
            desired["budget_min"] = None
        else:
            raise ValueError("本轮最低租金预算不能高于最高租金预算")

    updates: dict[str, Any] = {}
    clear_fields: list[PreferenceField] = []
    for field in PREFERENCE_FIELDS:
        old_value = stored.get(field)
        new_value = desired.get(field)
        if new_value == old_value:
            continue
        if new_value is None:
            clear_fields.append(field)
        else:
            updates[field] = new_value

    validated_updates = PreferenceUpdatesModel.model_validate(
        updates
    ).explicit_updates()
    return validated_updates, clear_fields


def _apply_list_operations(
    preferences: dict[str, Any],
    *,
    field: PreferenceField,
    additions: Sequence[str],
    removals: Sequence[str],
    clear: bool,
) -> None:
    """对列表偏好执行清空、删除、追加，并保持原顺序去重。"""

    if not clear and not additions and not removals:
        return
    current = [] if clear else list(preferences.get(field) or [])
    removal_set = set(removals)
    remaining = [item for item in current if item not in removal_set]
    merged = list(dict.fromkeys([*remaining, *additions]))
    preferences[field] = merged or None


def _apply_scalar_operation(
    preferences: dict[str, Any],
    *,
    field: PreferenceField,
    value: Any,
    clear: bool,
) -> None:
    """正向值优先于清空标记；未提及时保持旧值。"""

    if value is not None:
        preferences[field] = value
    elif clear:
        preferences[field] = None


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None
