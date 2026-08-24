"""把模型声明的工具选择原因写入可观察的 ToolMessage。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
import json
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command


class ToolSelectionReasonMiddleware(AgentMiddleware):
    """在不侵入业务工具返回分支的情况下统一回显选择原因。"""

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        return _attach_selection_reason(
            handler(request),
            request.tool_call.get("args", {}).get("selection_reason"),
        )

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[
            [ToolCallRequest], Awaitable[ToolMessage | Command[Any]]
        ],
    ) -> ToolMessage | Command[Any]:
        return _attach_selection_reason(
            await handler(request),
            request.tool_call.get("args", {}).get("selection_reason"),
        )


def _attach_selection_reason(
    result: ToolMessage | Command[Any],
    raw_reason: Any,
) -> ToolMessage | Command[Any]:
    """只改写普通 ToolMessage；handoff Command 自己维护父图结构化状态。"""

    if not isinstance(result, ToolMessage):
        return result
    reason = str(raw_reason or "").strip()
    if not reason:
        return result
    payload = _message_payload(result)
    payload["selection_reason"] = reason
    return result.model_copy(
        update={"content": json.dumps(payload, ensure_ascii=False, default=str)}
    )


def _message_payload(message: ToolMessage) -> dict[str, Any]:
    content = message.content
    if isinstance(content, str):
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            return payload
    return {
        "status": "failed" if message.status == "error" else "success",
        "result": content,
    }
