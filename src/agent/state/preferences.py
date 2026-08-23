"""用户偏好保存节点使用的状态定义。"""

from typing import Any, NotRequired, TypedDict

from langchain_core.messages import AnyMessage

from agent.common.preferences import PreferenceField


class PreferenceUpdates(TypedDict, total=False):
    """业务子图本轮明确收集到的偏好增量。"""

    city: str
    districts: list[str]
    budget_min: float
    budget_max: float
    room_types: list[str]
    rental_mode: str
    commute_location: str
    max_commute_minutes: int


class PreferenceState(TypedDict, total=False):
    """主图中与跨会话用户偏好有关的公共状态。"""

    # 正式运行时应由调用方传入 user_id；Studio 调试可回退到 CHAT_USER_ID。
    user_id: str
    # 偏好提取节点从当前业务流程起点读取全部Human/AI消息。
    messages: NotRequired[list[AnyMessage]]
    # 由主图入口设置，供偏好提取节点确定本轮业务消息的起点。
    current_turn_start_message_id: NotRequired[str | None]
    preference_updates: PreferenceUpdates
    # 需要显式写成数据库NULL的字段；与“本轮没有提到”严格区分。
    preference_clear_fields: list[PreferenceField]
    user_preferences: dict[str, Any]
    preference_load_error: str
    preference_extraction_error: str
    preferences_saved: NotRequired[bool]
    preference_save_error: NotRequired[str]
