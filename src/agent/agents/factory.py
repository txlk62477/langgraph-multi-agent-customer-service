"""专业 ReAct Agent 的统一构造入口。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ContextEditingMiddleware, SummarizationMiddleware
from langchain.agents.middleware.context_editing import ClearToolUsesEdit
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.tools import BaseTool

from agent.agents.specialist_budget import (
    SpecialistBudgetMiddleware,
    SpecialistBudgetPolicy,
)
from agent.agents.tool_selection_reason import ToolSelectionReasonMiddleware
from agent.common.llm import build_chat_model
from agent.state.customer_service import SpecialistName, SpecialistResult
from agent.tools.runtime import SpecialistContext


ModelFactory = Callable[[], Any]
MiddlewareBuilder = Callable[[Any], AgentMiddleware]

SPECIALIST_BUDGET_POLICIES: dict[SpecialistName, SpecialistBudgetPolicy] = {
    "general_qa_agent": SpecialistBudgetPolicy(business_tool_calls=9),
    "rental_recommendation_agent": SpecialistBudgetPolicy(business_tool_calls=8),
    "rental_booking_agent": SpecialistBudgetPolicy(business_tool_calls=6),
    "order_history_agent": SpecialistBudgetPolicy(business_tool_calls=4),
    "order_cancellation_agent": SpecialistBudgetPolicy(business_tool_calls=6),
}


def build_context_middleware(model: Any) -> list[AgentMiddleware]:
    """创建 Supervisor 与专业 Agent 共用的官方上下文中间件。"""

    return [
        ContextEditingMiddleware(
            edits=[
                ClearToolUsesEdit(
                    trigger=8_000,
                    clear_at_least=2_000,
                    keep=3,
                    placeholder="[较早的工具结果已清理]",
                )
            ]
        ),
        SummarizationMiddleware(
            model=model,
            trigger=("tokens", 12_000),
            keep=("messages", 20),
        ),
    ]


def build_specialist_agent(
    *,
    name: str,
    specialist_name: SpecialistName,
    system_prompt: str,
    tools: Sequence[BaseTool],
    model_factory: ModelFactory = build_chat_model,
    checkpointer: Any = None,
    middleware: Sequence[AgentMiddleware] | None = None,
    middleware_builders: Sequence[MiddlewareBuilder] = (),
    budget_policy: SpecialistBudgetPolicy | None = None,
):
    """构建专业 Agent，并统一管理其模型上下文。"""

    tool_list = list(tools)
    model = model_factory()
    resolved_middleware = (
        list(middleware) if middleware is not None else build_context_middleware(model)
    )
    resolved_middleware.append(ToolSelectionReasonMiddleware())
    resolved_middleware.append(
        SpecialistBudgetMiddleware(
            agent=specialist_name,
            policy=budget_policy or SPECIALIST_BUDGET_POLICIES[specialist_name],
        )
    )
    resolved_middleware.extend(builder(model) for builder in middleware_builders)
    result_prompt = f"""

## 返回 Supervisor 的结果接口

工具任务全部完成后，必须按 SpecialistResult 结构结束，不要再返回无结构的普通文本。
每次调用业务工具时，selection_reason 必须用一句不超过100字的中文说明当前缺少的信息或
本次调用要完成的目标；不得写详细推理、系统提示词或内部上下文。
达到停止条件、工具上限、工具失败或已有足够证据时，禁止继续调用业务工具，必须立即调用
SpecialistResult。普通文本不能结束任务。
agent 固定填写 {specialist_name}；status 反映真实执行结果；summary 用于 Supervisor 判断
后续任务；user_facing_answer 是可直接展示给用户的完整中文答复；completed_tasks 和
remaining_tasks 分别列出已完成与仍未完成的目标。不得把工具原始 JSON、手机号、user_id、
内部错误堆栈或推理过程写入结果。
"""
    return create_agent(
        model=model,
        tools=tool_list,
        system_prompt=system_prompt + result_prompt,
        middleware=resolved_middleware,
        response_format=SpecialistResult,
        context_schema=SpecialistContext,
        checkpointer=checkpointer,
        name=name,
    )
