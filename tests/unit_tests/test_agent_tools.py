"""Agent 深工具与异步 Agent Server 运行时的兼容性测试。"""

from __future__ import annotations

import json
import unittest
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage

from agent.tools.general_qa import build_web_search_tool
from agent.tools.rental import build_search_houses_tool


class AsyncOnlyGraph:
    """模拟只支持异步 checkpointer 的 Agent Server 子图。"""

    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.calls: list[tuple[dict[str, Any], dict[str, Any]]] = []

    def invoke(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        raise NotImplementedError("异步 checkpointer 不支持同步 invoke")

    async def ainvoke(
        self,
        graph_input: dict[str, Any],
        *,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append((graph_input, config))
        return self.result


def _runtime() -> ToolRuntime:
    return ToolRuntime(
        state={},
        context=None,
        config={"configurable": {"thread_id": "async-checkpointer-test"}},
        stream_writer=lambda _: None,
        tool_call_id="tool-call",
        store=None,
    )


class AsyncSubgraphToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_web_awaits_async_subgraph(self) -> None:
        graph = AsyncOnlyGraph(
            {
                "messages": [AIMessage(content="联网搜索结果")],
                "search_error": "",
            }
        )
        tool = build_web_search_tool(graph_factory=lambda: graph)

        self.assertIsNotNone(tool.coroutine)
        raw_result = await tool.coroutine("上海天气", _runtime())

        result = json.loads(raw_result)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["answer"], "联网搜索结果")
        self.assertEqual(len(graph.calls), 1)
        self.assertEqual(graph.calls[0][0]["search_query"], "上海天气")

    async def test_search_houses_awaits_async_subgraph(self) -> None:
        graph = AsyncOnlyGraph(
            {
                "query_status": "success",
                "query_result": "测试房源",
                "query_error": "",
            }
        )
        tool = build_search_houses_tool(graph_factory=lambda: graph)

        self.assertIsNotNone(tool.coroutine)
        raw_result = await tool.coroutine(
            "上海",
            2000,
            4000,
            _runtime(),
        )

        result = json.loads(raw_result)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["result"], "测试房源")
        self.assertEqual(len(graph.calls), 1)
        self.assertEqual(graph.calls[0][0]["table_name"], "house")


if __name__ == "__main__":
    unittest.main()
