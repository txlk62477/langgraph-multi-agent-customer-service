"""可直接回答或自主调用联网工具的常规问答 Agent。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.tools import BaseTool

from agent.agents.factory import build_specialist_agent
from agent.common.llm import build_chat_model
from agent.tools.general_qa import build_web_search_tool


GENERAL_QA_PROMPT = """你是智能客服的常规问答专业 Agent。

你的职责是完成不属于房源推荐、预订、订单历史或订单取消的请求。
- 问候、翻译、改写、总结、数学推理和稳定的一般知识可以直接回答。
- 用户明确要求联网，或问题涉及新闻、天气、时间、价格、政策、赛程、营业状态等
  可能变化的信息时，必须调用 search_web。
- search_web 成功时只依据工具证据回答并保留来源；失败且问题依赖实时数据时，明确
  说明当前无法可靠确认，不得用模型旧知识猜测。
- 不要调用不存在的工具，不要向用户展示工具 JSON、内部错误堆栈或思考过程。
- 最终只给出一条自然、简洁的中文回答。
"""


def build_general_qa_agent(
    *,
    model_factory: Callable[[], Any] = build_chat_model,
    tools: Sequence[BaseTool] | None = None,
    checkpointer: Any = None,
    name: str = "general_qa_agent",
):
    return build_specialist_agent(
        name=name,
        system_prompt=GENERAL_QA_PROMPT,
        tools=list(tools) if tools is not None else [build_web_search_tool()],
        model_factory=model_factory,
        checkpointer=checkpointer,
    )


general_qa_agent = build_general_qa_agent()
