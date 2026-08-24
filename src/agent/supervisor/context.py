"""Supervisor 模型上下文投影：保留审计状态，隐藏专业 Agent 内部消息。"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

from agent.state.customer_service import CustomerServiceState, DelegationRecord
from agent.supervisor.handoff import HANDOFF_TOOL_NAMES
from agent.tools.runtime import SpecialistContext


class SupervisorContextProjectionMiddleware(
    AgentMiddleware[CustomerServiceState, SpecialistContext, Any]
):
    """只向 Supervisor 模型展示用户消息和结构化 handoff 结果。

    该中间件只替换单次模型请求的 ``messages``，不更新图状态，因此专业 Agent
    的原始工具消息仍完整保存在 checkpoint 中，可供 Studio 和 LangSmith 排错。
    """

    state_schema = CustomerServiceState

    def __init__(self, *, supervisor_name: str) -> None:
        self._supervisor_name = supervisor_name

    def wrap_model_call(
        self,
        request: ModelRequest[SpecialistContext],
        handler: Callable[[ModelRequest[SpecialistContext]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        projected = _project_messages(
            request.messages,
            request.state.get("delegations", []),
            supervisor_name=self._supervisor_name,
        )
        return handler(request.override(messages=projected))

    async def awrap_model_call(
        self,
        request: ModelRequest[SpecialistContext],
        handler: Callable[
            [ModelRequest[SpecialistContext]], Awaitable[ModelResponse[Any]]
        ],
    ) -> ModelResponse[Any]:
        projected = _project_messages(
            request.messages,
            request.state.get("delegations", []),
            supervisor_name=self._supervisor_name,
        )
        return await handler(request.override(messages=projected))


def _project_messages(
    messages: list[AnyMessage],
    delegations: list[DelegationRecord],
    *,
    supervisor_name: str,
) -> list[AnyMessage]:
    """构造仅供 Supervisor 推理的消息视图，不改变 checkpoint 消息。"""

    records_by_call_id = {
        record["tool_call_id"]: record for record in delegations
    }
    projected: list[AnyMessage] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            projected.append(message)
            continue
        if isinstance(message, AIMessage):
            is_supervisor = message.name == supervisor_name
            is_handoff = any(
                call.get("name") in HANDOFF_TOOL_NAMES
                for call in message.tool_calls
            )
            if is_supervisor or is_handoff:
                projected.append(message)
            continue
        if not isinstance(message, ToolMessage) or message.name not in HANDOFF_TOOL_NAMES:
            continue

        record = records_by_call_id.get(message.tool_call_id)
        if record is None:
            projected.append(message)
            continue
        payload: dict[str, Any] = {
            "delegated_agent": record["agent"],
            "task": record["task"],
        }
        if result := record.get("result"):
            payload["result"] = result
        projected.append(
            message.model_copy(
                update={
                    "content": json.dumps(payload, ensure_ascii=False),
                }
            )
        )
    return projected

