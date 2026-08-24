"""专业 Agent 的统一执行预算与结构化结束保障。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, NotRequired, TypedDict

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    hook_config,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime

from agent.state.customer_service import SpecialistName, SpecialistResult


@dataclass(frozen=True, slots=True)
class SpecialistBudgetPolicy:
    """一个专业 Agent 在单次业务任务中的硬执行上限。"""

    business_tool_calls: int
    model_calls: int = 12

    def __post_init__(self) -> None:
        if self.business_tool_calls < 1:
            raise ValueError("business_tool_calls 必须大于0")
        if self.model_calls < 1:
            raise ValueError("model_calls 必须大于0")


class SpecialistBudgetSnapshot(TypedDict):
    """可写入 checkpoint 的当前专业任务预算快照。"""

    owner_key: str
    agent: SpecialistName
    model_calls: int
    business_tool_calls: int
    final_attempt: bool
    block_model_call: bool


class SpecialistBudgetState(AgentState[Any]):
    """预算中间件向 Agent 状态增加的字段。"""

    specialist_budget: NotRequired[SpecialistBudgetSnapshot]


_NON_BUSINESS_TOOLS = frozenset({"request_user_input", "SpecialistResult"})


class SpecialistBudgetMiddleware(AgentMiddleware[SpecialistBudgetState, Any, SpecialistResult]):
    """限制专业 Agent 循环，并保证预算耗尽时产生结构化结果。"""

    state_schema = SpecialistBudgetState

    def __init__(self, *, agent: SpecialistName, policy: SpecialistBudgetPolicy) -> None:
        super().__init__()
        self.agent = agent
        self.policy = policy

    def before_agent(
        self,
        state: SpecialistBudgetState,
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        del runtime
        current = state.get("specialist_budget")
        owner_key = _resolve_owner_key(state, self.agent)
        # 摘要中间件可能清除最初的 HumanMessage。没有发现新的委派或用户消息时，
        # 继续沿用 checkpoint 中的 owner_key，不能把压缩上下文误判为新任务。
        if owner_key is None and current and current.get("agent") == self.agent:
            return None
        owner_key = owner_key or f"agent:{self.agent}"
        if (
            current
            and current.get("owner_key") == owner_key
            and current.get("agent") == self.agent
        ):
            return None
        return {"specialist_budget": _new_snapshot(owner_key, self.agent)}

    @hook_config(can_jump_to=["end"])
    def before_model(
        self,
        state: SpecialistBudgetState,
        runtime: Runtime[Any],
    ) -> dict[str, Any]:
        del runtime
        budget = _budget_from_state(state, self.agent)
        if budget["model_calls"] >= self.policy.model_calls:
            budget["block_model_call"] = True
        else:
            budget["model_calls"] += 1
            budget["block_model_call"] = False
        return {"specialist_budget": budget}

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[SpecialistResult]],
    ) -> ModelResponse[SpecialistResult]:
        budget = _budget_from_state(request.state, self.agent)
        if budget["block_model_call"]:
            return _fallback_response(self.agent)

        effective_request = request
        if budget["final_attempt"]:
            # structured output 工具由 create_agent 根据 response_format 动态加入；
            # 清空这里只会移除普通业务工具，最终模型只能选择 SpecialistResult。
            effective_request = request.override(tools=[])
        response = handler(effective_request)
        if response.structured_response is not None:
            return response
        if budget["final_attempt"] or budget["model_calls"] >= self.policy.model_calls:
            return _fallback_response(self.agent, response)
        return response

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[
            [ModelRequest[Any]], Awaitable[ModelResponse[SpecialistResult]]
        ],
    ) -> ModelResponse[SpecialistResult]:
        budget = _budget_from_state(request.state, self.agent)
        if budget["block_model_call"]:
            return _fallback_response(self.agent)

        effective_request = request.override(tools=[]) if budget["final_attempt"] else request
        response = await handler(effective_request)
        if response.structured_response is not None:
            return response
        if budget["final_attempt"] or budget["model_calls"] >= self.policy.model_calls:
            return _fallback_response(self.agent, response)
        return response

    def after_model(
        self,
        state: SpecialistBudgetState,
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        del runtime
        if state.get("structured_response") is not None:
            return None
        message = _last_ai_message(state.get("messages", ()))
        if message is None or not message.tool_calls:
            return None

        budget = _budget_from_state(state, self.agent)
        remaining = max(
            self.policy.business_tool_calls - budget["business_tool_calls"],
            0,
        )
        blocked_calls: list[dict[str, Any]] = []
        business_seen = 0
        for call in message.tool_calls:
            if call.get("name") in _NON_BUSINESS_TOOLS:
                continue
            if business_seen < remaining:
                business_seen += 1
            else:
                blocked_calls.append(call)

        if business_seen == 0 and not blocked_calls:
            return None

        budget["business_tool_calls"] += business_seen
        if budget["business_tool_calls"] >= self.policy.business_tool_calls:
            budget["final_attempt"] = True

        update: dict[str, Any] = {"specialist_budget": budget}
        if blocked_calls:
            blocked_ids = {str(call.get("id", "")) for call in blocked_calls}
            kept_calls = [
                call
                for call in message.tool_calls
                if str(call.get("id", "")) not in blocked_ids
            ]
            update["messages"] = [
                message.model_copy(update={"tool_calls": kept_calls}),
                *[
                    ToolMessage(
                        content="专业 Agent 已达到业务工具调用上限，本次调用未执行。",
                        tool_call_id=str(call.get("id", "")),
                        name=str(call.get("name", "")),
                        status="error",
                    )
                    for call in blocked_calls
                ],
            ]
        return update

    async def aafter_model(
        self,
        state: SpecialistBudgetState,
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        return self.after_model(state, runtime)


def _new_snapshot(owner_key: str, agent: SpecialistName) -> SpecialistBudgetSnapshot:
    return {
        "owner_key": owner_key,
        "agent": agent,
        "model_calls": 0,
        "business_tool_calls": 0,
        "final_attempt": False,
        "block_model_call": False,
    }


def _budget_from_state(
    state: Mapping[str, Any],
    agent: SpecialistName,
) -> SpecialistBudgetSnapshot:
    value = state.get("specialist_budget")
    if isinstance(value, Mapping) and value.get("agent") == agent:
        return {
            "owner_key": str(value.get("owner_key", agent)),
            "agent": agent,
            "model_calls": int(value.get("model_calls", 0)),
            "business_tool_calls": int(value.get("business_tool_calls", 0)),
            "final_attempt": bool(value.get("final_attempt", False)),
            "block_model_call": bool(value.get("block_model_call", False)),
        }
    return _new_snapshot(_resolve_owner_key(state, agent) or f"agent:{agent}", agent)


def _resolve_owner_key(
    state: Mapping[str, Any],
    agent: SpecialistName,
) -> str | None:
    delegations = state.get("delegations", ())
    if isinstance(delegations, Sequence):
        for record in reversed(delegations):
            if (
                isinstance(record, Mapping)
                and record.get("agent") == agent
                and "result" not in record
                and record.get("tool_call_id")
            ):
                return f"delegation:{record['tool_call_id']}"

    messages = state.get("messages", ())
    if isinstance(messages, Sequence):
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if not isinstance(message, HumanMessage):
                continue
            if message.id:
                return f"human:{message.id}"
            digest = sha256(str(message.content).encode("utf-8")).hexdigest()[:16]
            return f"human:{index}:{digest}"
    return None


def _last_ai_message(messages: Sequence[Any]) -> AIMessage | None:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    return None


def _fallback_response(
    agent: SpecialistName,
    original: ModelResponse[SpecialistResult] | None = None,
) -> ModelResponse[SpecialistResult]:
    fallback = SpecialistResult(
        agent=agent,
        status="failed",
        summary="专业 Agent 达到执行上限",
        user_facing_answer="本次处理已达到执行上限，请稍后重试。",
        completed_tasks=[],
        remaining_tasks=["重新执行当前专业任务"],
    )
    message_id = None
    message_name = agent
    if original:
        latest = _last_ai_message(original.result)
        if latest is not None:
            message_id = latest.id
            message_name = latest.name or agent
    return ModelResponse(
        result=[
            AIMessage(
                content=fallback["user_facing_answer"],
                id=message_id,
                name=message_name,
            )
        ],
        structured_response=fallback,
    )
