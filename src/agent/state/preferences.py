"""可供主图继承的用户偏好状态接口。"""

from typing import Any, NotRequired, TypedDict

from langchain.agents.middleware.types import AgentState

class PreferenceStatus(TypedDict):
    """最近一次偏好读取或更新的可观测状态。"""

    loaded: bool
    saved: bool
    load_error: str
    extraction_error: str
    save_error: str


class PreferenceState(AgentState[Any]):
    """所有需要跨会话租房偏好的 Agent 图状态基类。"""

    # 正式运行时应由调用方传入 user_id；Studio 调试可回退到 CHAT_USER_ID。
    user_id: NotRequired[str]
    # 由主图入口设置，供偏好提取节点确定本轮业务消息的起点。
    current_turn_start_message_id: NotRequired[str | None]
    # Store 中加载的完整偏好快照，专业 Agent 只读使用。
    user_preferences: NotRequired[dict[str, Any]]
    # 偏好节点的紧凑诊断；提取增量和清空字段只在节点内部流转。
    preference_status: NotRequired[PreferenceStatus]
