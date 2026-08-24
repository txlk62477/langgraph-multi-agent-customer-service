"""专业 ReAct Agent 的统一构造入口。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ContextEditingMiddleware, SummarizationMiddleware
from langchain.agents.middleware.context_editing import ClearToolUsesEdit
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.tools import BaseTool

from agent.common.llm import build_chat_model
from agent.tools.runtime import SpecialistContext


ModelFactory = Callable[[], Any]
MiddlewareBuilder = Callable[[Any], AgentMiddleware]


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
    system_prompt: str,
    tools: Sequence[BaseTool],
    model_factory: ModelFactory = build_chat_model,
    checkpointer: Any = None,
    middleware: Sequence[AgentMiddleware] | None = None,
    middleware_builders: Sequence[MiddlewareBuilder] = (),
):
    """构建专业 Agent，并统一管理其模型上下文。"""

    tool_list = list(tools)
    model = model_factory()
    resolved_middleware = (
        list(middleware) if middleware is not None else build_context_middleware(model)
    )
    resolved_middleware.extend(builder(model) for builder in middleware_builders)
    return create_agent(
        model=model,
        tools=tool_list,
        system_prompt=system_prompt,
        middleware=resolved_middleware,
        context_schema=SpecialistContext,
        checkpointer=checkpointer,
        name=name,
    )
