"""工具层共享的运行时身份与序列化辅助函数。"""

from __future__ import annotations

import json
import os
from typing import Any, Mapping, NotRequired, TypedDict

from langchain.tools import ToolRuntime


class SpecialistContext(TypedDict):
    """Agent Server 注入给专业 Agent 和工具的可信运行上下文。"""

    user_id: NotRequired[str]


def resolve_user_id(runtime: ToolRuntime[SpecialistContext]) -> str:
    """从可信运行时解析用户身份，不把 user_id 暴露给模型参数。"""

    state = runtime.state if isinstance(runtime.state, Mapping) else {}
    context = runtime.context if isinstance(runtime.context, Mapping) else {}
    configurable = runtime.config.get("configurable", {})
    candidates = (
        state.get("user_id"),
        context.get("user_id"),
        configurable.get("user_id"),
        os.getenv("CHAT_USER_ID"),
    )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    raise ValueError("缺少 user_id，请通过 configurable.user_id 提供用户身份")


def json_result(**values: Any) -> str:
    """返回稳定、可供模型读取但不执行的 JSON 工具结果。"""

    return json.dumps(values, ensure_ascii=False, default=str)
