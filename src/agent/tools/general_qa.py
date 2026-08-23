"""常规问答 Agent 的联网研究工具。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AIMessage, HumanMessage

from agent.graph.web_search import build_web_search_graph
from agent.tools.runtime import json_result


def build_web_search_tool(
    *,
    graph_factory: Callable[[], Any] | None = None,
):
    """把证据收集工作流封装为 Agent 看到的一个深工具。"""

    resolved_graph_factory = graph_factory or (
        lambda: build_web_search_graph(name="general_qa_agent_web_search")
    )
    graph: Any | None = None

    @tool("search_web")
    async def search_web(query: str, runtime: ToolRuntime) -> str:
        """搜索实时或外部信息，读取网页证据并返回带来源的综合结果。"""

        nonlocal graph
        cleaned = query.strip()
        if not cleaned:
            return json_result(status="failed", error="搜索词不能为空")
        try:
            if graph is None:
                graph = resolved_graph_factory()
            result = await graph.ainvoke(
                {
                    "messages": [HumanMessage(content=cleaned)],
                    "search_query": cleaned,
                },
                config=runtime.config,
            )
            answer = next(
                (
                    message.content
                    for message in reversed(result.get("messages", []))
                    if isinstance(message, AIMessage)
                    and isinstance(message.content, str)
                    and message.content.strip()
                ),
                "",
            )
            error = str(result.get("search_error") or "")
            return json_result(
                status="failed" if error else "success",
                answer=answer,
                error=error,
            )
        except Exception as error:
            return json_result(
                status="failed",
                error=f"{type(error).__name__}: {error}",
            )

    return search_web
