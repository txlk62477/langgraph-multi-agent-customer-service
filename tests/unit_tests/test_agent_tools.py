"""General QA Agent 自主搜索工具的公共行为测试。"""

from __future__ import annotations

import json
import unittest
from unittest.mock import Mock

from langchain.tools import ToolRuntime
from langchain_core.messages import HumanMessage, ToolMessage

from agent.common.vision import VisionEvidence
from agent.tools.general_qa import (
    build_analyze_page_visuals_tool,
    build_anysearch_search_tool,
    build_playwright_read_page_tool,
)


def _runtime(messages=None) -> ToolRuntime:
    return ToolRuntime(
        state={"messages": list(messages or [])},
        context=None,
        config={"configurable": {"thread_id": "general-qa-tool-test"}},
        stream_writer=lambda _: None,
        tool_call_id="current-tool-call",
        store=None,
    )


def _search_message(*urls: str) -> ToolMessage:
    return ToolMessage(
        name="anysearch_search",
        tool_call_id="search-call",
        content=json.dumps(
            {
                "status": "success",
                "results": [
                    {"rank": rank, "title": f"来源{rank}", "url": url, "snippet": "摘要"}
                    for rank, url in enumerate(urls, start=1)
                ],
            },
            ensure_ascii=False,
        ),
    )


class GeneralQASearchToolTests(unittest.TestCase):
    def test_anysearch_returns_compact_candidates_without_content(self) -> None:
        search = Mock(
            return_value=[
                {
                    "rank": 1,
                    "title": "上海天气",
                    "url": "https://weather.example.com/shanghai",
                    "snippet": "今天有雨",
                    "content": "不应直接交给Agent的完整正文",
                }
            ]
        )
        tool = build_anysearch_search_tool(search=search)

        raw_result = tool.func("上海今日天气", 3, _runtime())

        result = json.loads(raw_result)
        self.assertEqual(result["status"], "success")
        self.assertEqual(
            result["results"],
            [
                {
                    "rank": 1,
                    "title": "上海天气",
                    "url": "https://weather.example.com/shanghai",
                    "snippet": "今天有雨",
                }
            ],
        )
        search.assert_called_once_with("上海今日天气", max_results=3)

    def test_playwright_reads_only_a_url_from_search_results(self) -> None:
        page_reader = Mock(
            return_value={
                "title": "上海天气",
                "url": "https://weather.example.com/shanghai",
                "browser_status": "success",
                "rendered_text": "上海今天有雨。",
                "json_responses": [{"url": "https://weather.example.com/api", "body": "{}"}],
                "browser_error": "",
                "screenshot_base64": "must-not-leak",
            }
        )
        tool = build_playwright_read_page_tool(page_reader=page_reader)
        runtime = _runtime([_search_message("https://weather.example.com/shanghai")])

        raw_result = tool.func(
            "https://weather.example.com/shanghai",
            runtime,
        )

        result = json.loads(raw_result)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["rendered_text"], "上海今天有雨。")
        self.assertNotIn("screenshot_base64", result)
        page_reader.assert_called_once_with(
            {
                "title": "",
                "url": "https://weather.example.com/shanghai",
            },
            capture_screenshot=False,
        )

    def test_page_tools_reject_an_invented_url(self) -> None:
        page_reader = Mock()
        runtime = _runtime([_search_message("https://allowed.example.com")])
        read_tool = build_playwright_read_page_tool(page_reader=page_reader)
        vision_tool = build_analyze_page_visuals_tool(page_reader=page_reader)

        read_result = json.loads(
            read_tool.func("https://invented.example.com", runtime)
        )
        vision_result = json.loads(
            vision_tool.func(
                "https://invented.example.com",
                "页面价格是多少？",
                runtime,
            )
        )

        self.assertEqual(read_result["status"], "rejected")
        self.assertEqual(vision_result["status"], "rejected")
        page_reader.assert_not_called()

    def test_visual_analysis_accepts_a_user_supplied_url(self) -> None:
        page_reader = Mock(
            return_value={
                "title": "价格看板",
                "url": "https://dashboard.example.com/prices",
                "browser_status": "success",
                "rendered_text": "",
                "json_responses": [],
                "browser_error": "",
                "screenshot_status": "success",
                "screenshot_base64": "anBlZw==",
                "screenshot_error": "",
            }
        )
        vision_client = Mock()
        vision_client.analyze_webpage.return_value = VisionEvidence(
            relevant=True,
            description="图表显示价格上涨。",
            visible_facts=["当前价格为100"],
            uncertainties=[],
        )
        tool = build_analyze_page_visuals_tool(
            page_reader=page_reader,
            vision_factory=lambda: vision_client,
        )
        runtime = _runtime(
            [HumanMessage(content="请分析 https://dashboard.example.com/prices 的图表")]
        )

        raw_result = tool.func(
            "https://dashboard.example.com/prices",
            "当前价格和趋势是什么？",
            runtime,
        )

        result = json.loads(raw_result)
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["relevant"])
        self.assertEqual(result["visible_facts"], ["当前价格为100"])
        self.assertNotIn("screenshot_base64", result)
        vision_client.analyze_webpage.assert_called_once_with(
            query="当前价格和趋势是什么？",
            title="价格看板",
            url="https://dashboard.example.com/prices",
            screenshot_base64="anBlZw==",
        )

    def test_each_tool_enforces_its_per_turn_budget(self) -> None:
        search = Mock(return_value=[])
        page_reader = Mock()
        search_tool = build_anysearch_search_tool(search=search)
        read_tool = build_playwright_read_page_tool(page_reader=page_reader)
        vision_tool = build_analyze_page_visuals_tool(page_reader=page_reader)
        searched_url = "https://allowed.example.com"

        search_messages = [
            ToolMessage(name="anysearch_search", tool_call_id=f"s-{index}", content="{}")
            for index in range(3)
        ]
        read_messages = [
            _search_message(searched_url),
            *[
                ToolMessage(name="playwright_read_page", tool_call_id=f"r-{index}", content="{}")
                for index in range(4)
            ],
        ]
        vision_messages = [
            _search_message(searched_url),
            *[
                ToolMessage(name="analyze_page_visuals", tool_call_id=f"v-{index}", content="{}")
                for index in range(2)
            ],
        ]

        search_result = json.loads(
            search_tool.func("继续搜索", 5, _runtime(search_messages))
        )
        read_result = json.loads(
            read_tool.func(searched_url, _runtime(read_messages))
        )
        vision_result = json.loads(
            vision_tool.func(
                searched_url,
                "继续分析",
                _runtime(vision_messages),
            )
        )

        self.assertEqual(search_result["status"], "limit_reached")
        self.assertEqual(read_result["status"], "limit_reached")
        self.assertEqual(vision_result["status"], "limit_reached")
        search.assert_not_called()
        page_reader.assert_not_called()

    def test_tool_budget_resets_after_the_next_user_message(self) -> None:
        search = Mock(return_value=[])
        tool = build_anysearch_search_tool(search=search)
        messages = [
            HumanMessage(content="第一轮问题"),
            *[
                ToolMessage(
                    name="anysearch_search",
                    tool_call_id=f"old-{index}",
                    content="{}",
                )
                for index in range(3)
            ],
            HumanMessage(content="第二轮问题"),
        ]

        result = json.loads(tool.func("新的搜索", 5, _runtime(messages)))

        self.assertEqual(result["status"], "empty")
        search.assert_called_once_with("新的搜索", max_results=5)


if __name__ == "__main__":
    unittest.main()
