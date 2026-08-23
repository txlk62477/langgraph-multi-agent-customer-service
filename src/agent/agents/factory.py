"""专业 ReAct Agent 的统一构造入口。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import BaseTool

from agent.common.llm import build_chat_model


ModelFactory = Callable[[], Any]


def build_specialist_agent(
    *,
    name: str,
    system_prompt: str,
    tools: Sequence[BaseTool],
    model_factory: ModelFactory = build_chat_model,
    checkpointer: Any = None,
):
    """构建专业 Agent，并在构图时创建一次可复用的模型实例。"""

    tool_list = list(tools)
    return create_agent(
        model=model_factory(),
        tools=tool_list,
        system_prompt=system_prompt,
        checkpointer=checkpointer,
        name=name,
    )
