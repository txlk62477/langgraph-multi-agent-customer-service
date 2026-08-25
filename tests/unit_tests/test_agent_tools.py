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
    def test_playwright_tool_schema_requires_a_bounded_urls_array(self) -> None:
        schema = build_playwright_read_page_tool(
            pages_reader=Mock()
        ).tool_call_schema.model_json_schema()

        self.assertIn("urls", schema["properties"])
        self.assertNotIn("url", schema["properties"])
        self.assertEqual(schema["properties"]["urls"]["minItems"], 1)
        self.assertEqual(schema["properties"]["urls"]["maxItems"], 4)

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

        raw_result = tool.func("上海今日天气", "查询最新天气来源", 3, _runtime())

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

    def test_playwright_reads_multiple_urls_in_one_batch(self) -> None:
        urls = [
            "https://weather.example.com/shanghai",
            "https://weather.example.com/hefei",
        ]
        pages_reader = Mock(
            return_value=[
                {
                    "title": "上海天气",
                    "url": urls[0],
                    "browser_status": "success",
                    "http_status": 200,
                    "rendered_text": "上海今天有雨。",
                    "json_responses": [],
                    "browser_error": "",
                },
                {
                    "title": "合肥天气",
                    "url": urls[1],
                    "browser_status": "success",
                    "http_status": 200,
                    "rendered_text": "合肥今天晴。",
                    "json_responses": [],
                    "browser_error": "",
                },
            ]
        )
        tool = build_playwright_read_page_tool(pages_reader=pages_reader)
        runtime = _runtime([_search_message(*urls)])

        raw_result = tool.func(
            urls,
            "读取两个独立天气来源进行比较",
            runtime,
        )

        result = json.loads(raw_result)
        self.assertEqual(result["status"], "success")
        self.assertEqual(
            [page["rendered_text"] for page in result["pages"]],
            ["上海今天有雨。", "合肥今天晴。"],
        )
        pages_reader.assert_called_once_with(
            [
                {"title": "", "url": urls[0]},
                {"title": "", "url": urls[1]},
            ],
            max_concurrency=3,
        )

    def test_playwright_keeps_successful_pages_when_one_page_is_blocked(self) -> None:
        allowed_url = "https://weather.example.com/shanghai"
        blocked_url = "https://weather.example.com/protected"
        pages_reader = Mock(
            return_value=[
                {
                    "title": "上海天气",
                    "url": allowed_url,
                    "browser_status": "success",
                    "http_status": 200,
                    "rendered_text": "上海今天有雨。",
                    "json_responses": [],
                    "browser_error": "",
                },
                {
                    "title": "安全验证",
                    "url": blocked_url,
                    "browser_status": "success",
                    "http_status": 403,
                    "rendered_text": "访问过于频繁，请完成人机验证后继续访问。",
                    "json_responses": [],
                    "browser_error": "",
                },
            ]
        )
        tool = build_playwright_read_page_tool(pages_reader=pages_reader)

        result = json.loads(
            tool.func(
                [allowed_url, blocked_url],
                "读取两个天气来源并保留可用证据",
                _runtime([_search_message(allowed_url, blocked_url)]),
            )
        )

        self.assertEqual(result["status"], "partial_success")
        self.assertEqual(result["pages"][0]["rendered_text"], "上海今天有雨。")
        self.assertEqual(result["pages"][1]["status"], "failed")
        self.assertEqual(result["pages"][1]["http_status"], 403)
        self.assertIn("人机验证", result["pages"][1]["error"])

    def test_playwright_rejects_only_unapproved_urls_in_a_batch(self) -> None:
        allowed_url = "https://allowed.example.com"
        invented_url = "https://invented.example.com"
        pages_reader = Mock(
            return_value=[
                {
                    "title": "允许的来源",
                    "url": allowed_url,
                    "browser_status": "success",
                    "http_status": 200,
                    "rendered_text": "可用正文",
                    "json_responses": [],
                    "browser_error": "",
                }
            ]
        )
        runtime = _runtime([_search_message(allowed_url)])
        read_tool = build_playwright_read_page_tool(pages_reader=pages_reader)

        result = json.loads(
            read_tool.func(
                [allowed_url, invented_url],
                "读取已批准来源并拒绝模型编造的URL",
                runtime,
            )
        )

        self.assertEqual(result["status"], "partial_success")
        self.assertEqual(result["pages"][0]["status"], "success")
        self.assertEqual(result["pages"][1]["status"], "rejected")
        pages_reader.assert_called_once_with(
            [{"title": "", "url": allowed_url}],
            max_concurrency=3,
        )

    def test_playwright_deduplicates_urls_in_first_seen_order(self) -> None:
        first_url = "https://source.example.com/first"
        second_url = "https://source.example.com/second"
        pages_reader = Mock(
            return_value=[
                {
                    "title": "来源一",
                    "url": first_url,
                    "browser_status": "success",
                    "http_status": 200,
                    "rendered_text": "第一篇",
                    "json_responses": [],
                    "browser_error": "",
                },
                {
                    "title": "来源二",
                    "url": second_url,
                    "browser_status": "success",
                    "http_status": 200,
                    "rendered_text": "第二篇",
                    "json_responses": [],
                    "browser_error": "",
                },
            ]
        )
        tool = build_playwright_read_page_tool(pages_reader=pages_reader)

        result = json.loads(
            tool.func(
                [first_url, first_url, second_url, first_url],
                "去重后读取两个独立来源",
                _runtime([_search_message(first_url, second_url)]),
            )
        )

        self.assertEqual(
            [page["url"] for page in result["pages"]],
            [first_url, second_url],
        )
        pages_reader.assert_called_once_with(
            [
                {"title": "", "url": first_url},
                {"title": "", "url": second_url},
            ],
            max_concurrency=3,
        )

    def test_playwright_rejects_more_than_four_distinct_urls(self) -> None:
        urls = [f"https://source.example.com/{index}" for index in range(5)]
        pages_reader = Mock()
        tool = build_playwright_read_page_tool(pages_reader=pages_reader)

        result = json.loads(
            tool.func(
                urls,
                "尝试一次读取过多来源",
                _runtime([_search_message(*urls)]),
            )
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("最多4个", result["error"])
        pages_reader.assert_not_called()

    def test_visual_tool_rejects_an_invented_url(self) -> None:
        page_reader = Mock()
        runtime = _runtime([_search_message("https://allowed.example.com")])
        vision_tool = build_analyze_page_visuals_tool(page_reader=page_reader)

        result = json.loads(
            vision_tool.func(
                "https://invented.example.com",
                "页面价格是多少？",
                "尝试分析未经搜索批准的页面",
                runtime,
            )
        )

        self.assertEqual(result["status"], "rejected")
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
            "文本不足，需要读取看板中的价格趋势",
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

    def test_visual_analysis_does_not_call_model_for_a_verification_page(self) -> None:
        url = "https://dashboard.example.com/protected"
        page_reader = Mock(
            return_value={
                "title": "安全验证",
                "url": url,
                "browser_status": "success",
                "http_status": 403,
                "rendered_text": "请完成人机验证",
                "json_responses": [],
                "browser_error": "",
                "screenshot_status": "success",
                "screenshot_base64": "anBlZw==",
                "screenshot_error": "",
            }
        )
        vision_client = Mock()
        tool = build_analyze_page_visuals_tool(
            page_reader=page_reader,
            vision_factory=lambda: vision_client,
        )

        result = json.loads(
            tool.func(
                url,
                "看板显示了什么？",
                "需要分析受保护看板中的可视内容",
                _runtime([HumanMessage(content=f"请分析 {url}")]),
            )
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("人机验证", result["error"])
        vision_client.analyze_webpage.assert_not_called()

    def test_each_tool_enforces_its_per_turn_budget(self) -> None:
        search = Mock(return_value=[])
        page_reader = Mock()
        pages_reader = Mock()
        search_tool = build_anysearch_search_tool(search=search)
        read_tool = build_playwright_read_page_tool(pages_reader=pages_reader)
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
            search_tool.func(
                "继续搜索",
                "已有来源不足，需要继续搜索",
                5,
                _runtime(search_messages),
            )
        )
        read_result = json.loads(
            read_tool.func(
                [searched_url],
                "需要继续读取网页正文",
                _runtime(read_messages),
            )
        )
        vision_result = json.loads(
            vision_tool.func(
                searched_url,
                "继续分析",
                "需要继续分析页面视觉内容",
                _runtime(vision_messages),
            )
        )

        self.assertEqual(search_result["status"], "limit_reached")
        self.assertEqual(read_result["status"], "limit_reached")
        self.assertEqual(vision_result["status"], "limit_reached")
        search.assert_not_called()
        pages_reader.assert_not_called()
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

        result = json.loads(
            tool.func(
                "新的搜索",
                "新一轮问题需要新的候选来源",
                5,
                _runtime(messages),
            )
        )

        self.assertEqual(result["status"], "empty")
        search.assert_called_once_with("新的搜索", max_results=5)


if __name__ == "__main__":
    unittest.main()
