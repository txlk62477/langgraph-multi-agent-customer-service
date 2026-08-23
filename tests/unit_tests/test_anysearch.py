"""AnySearch 搜索测试图的离线测试。"""

import base64
import json
import os
import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage

from agent.common.anysearch import AnySearchError, search_anysearch
from agent.common.browser_reader import _compact_text, _is_allowed_url
from agent.common.vision import VisionEvidence
from agent.node import web_search as web_search_nodes
from agent.node.web_search import WebPageProcessingNodes
from agent.graph.web_search import build_web_search_graph, web_search_graph

class AnySearchToolTests(unittest.TestCase):
    @patch("agent.common.anysearch.httpx.post")
    def test_normalizes_search_response(self, post: Mock) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "code": 0,
            "message": "success",
            "request_id": "request-1",
            "data": {
                "results": [
                    {
                        "title": "示例标题",
                        "url": "https://example.com/news",
                        "snippet": "示例摘要",
                        "content": "清洗后的网页正文",
                    }
                ],
                "metadata": {"total_results": 1, "search_time_ms": 100},
            },
        }
        post.return_value = response

        with patch.dict(os.environ, {"ANYSEARCH_API_KEY": "test-key"}):
            results = search_anysearch("测试问题")

        self.assertEqual(results[0]["rank"], 1)
        self.assertEqual(results[0]["title"], "示例标题")
        self.assertEqual(results[0]["snippet"], "示例摘要")
        self.assertEqual(results[0]["content"], "清洗后的网页正文")
        post.assert_called_once()
        request = post.call_args.kwargs
        self.assertEqual(request["json"]["query"], "测试问题")
        self.assertNotIn("tag", request["json"])
        self.assertNotIn("params", request["json"])

    @patch("agent.common.anysearch.httpx.post")
    def test_raises_for_api_error(self, post: Mock) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "code": -1,
            "message": "invalid request",
            "request_id": "request-2",
        }
        post.return_value = response

        with (
            patch.dict(os.environ, {"ANYSEARCH_API_KEY": "test-key"}),
            self.assertRaises(AnySearchError),
        ):
            search_anysearch("测试问题")

    def test_params_require_tag(self) -> None:
        with self.assertRaises(ValueError):
            search_anysearch("测试问题", params={"library": "golang"})


class BrowserReaderTests(unittest.TestCase):
    def test_rejects_local_and_private_urls(self) -> None:
        self.assertFalse(_is_allowed_url("http://localhost:8000/private"))
        self.assertFalse(_is_allowed_url("http://127.0.0.1/private"))
        self.assertFalse(_is_allowed_url("http://192.168.1.5/private"))
        self.assertTrue(_is_allowed_url("https://example.com/news"))

    def test_compacts_and_limits_text(self) -> None:
        self.assertEqual(_compact_text(" 第一行 \n\n 第二行 ", 100), "第一行\n第二行")
        self.assertIn("内容已截断", _compact_text("123456", 3))


class WebSearchSubgraphTests(unittest.TestCase):
    def test_formal_subgraph_exposes_minimal_public_schema(self) -> None:
        """正式子图只公开消息输入以及消息、错误输出。"""

        input_properties = web_search_graph.get_input_jsonschema()["properties"]
        output_properties = web_search_graph.get_output_jsonschema()["properties"]

        self.assertEqual(set(input_properties), {"messages", "search_query"})
        self.assertEqual(set(output_properties), {"messages", "search_error"})

    def test_formal_subgraph_contains_complete_search_pipeline(self) -> None:
        """正式子图应包含搜索、单网页子图、聚合和答案生成。"""

        node_names = set(web_search_graph.get_graph().nodes)
        self.assertTrue(
            {
                "anysearch_search",
                "process_webpage",
                "finalize_webpages",
                "generate_answer",
            }
            <= node_names
        )

    def test_formal_pipeline_processes_every_result_with_vision(self) -> None:
        """每条AnySearch结果都应进入网页读取和视觉分析流水线。"""

        search_results = [
            {
                "rank": 1,
                "title": "第一个网页",
                "url": "https://example.com/one",
            },
            {
                "rank": 2,
                "title": "第二个网页",
                "url": "https://example.com/two",
            },
        ]

        def fake_page_reader(page, *, capture_screenshot):
            self.assertTrue(capture_screenshot)
            return {
                **page,
                "browser_status": "success",
                "rendered_text": f"网页正文{page['rank']}",
                "json_responses": [],
                "browser_error": "",
                "screenshot_status": "success",
                "screenshot_mime_type": "image/jpeg",
                "screenshot_base64": base64.b64encode(b"jpeg").decode(),
                "screenshot_error": "",
            }

        vision_client = Mock()
        vision_client.analyze_webpage.side_effect = lambda **kwargs: VisionEvidence(
            relevant=True,
            description=f"识别到{kwargs['title']}中的动态信息",
            visible_facts=["页面显示值为42"],
            uncertainties=[],
        )
        page_nodes = WebPageProcessingNodes(
            page_reader=fake_page_reader,
            vision_factory=lambda: vision_client,
            browser_max_concurrency=3,
            vision_max_concurrency=2,
            screenshot_retention="none",
        )
        graph = build_web_search_graph(
            page_nodes=page_nodes,
            name="test_visual_web_search",
        )
        search_tool = Mock()
        search_tool.invoke.return_value = json.dumps(
            search_results, ensure_ascii=False
        )
        model = Mock()
        model.invoke.return_value = AIMessage(content="综合答案。[1]")

        with (
            patch.object(web_search_nodes, "anysearch_web_search", search_tool),
            patch.object(web_search_nodes, "build_chat_model", return_value=model),
        ):
            result = graph.invoke(
                {"messages": [HumanMessage(content="页面显示了什么？")]}
            )

        self.assertEqual(result["messages"][-1].content, "综合答案。[1]")
        self.assertNotIn("browser_results", result)
        self.assertNotIn("search_results", result)
        self.assertNotIn("page_results", result)
        self.assertEqual(vision_client.analyze_webpage.call_count, 2)
        model_input = model.invoke.call_args.args[0][1].content
        self.assertIn("网页正文1", model_input)
        self.assertIn("vision_description", model_input)

        system_prompt = model.invoke.call_args.args[0][0].content
        self.assertIn("json_responses", system_prompt)
        self.assertIn("vision_description", system_prompt)
        self.assertIn("rendered_text", system_prompt)
        self.assertIn("content 和 snippet", system_prompt)

    def test_generate_answer_never_sends_retained_base64_to_deepseek(self) -> None:
        """state调试模式可保留截图，但最终答案模型只接收视觉文字证据。"""

        model = Mock()
        model.invoke.return_value = AIMessage(content="答案")
        with patch.object(
            web_search_nodes,
            "build_chat_model",
            return_value=model,
        ):
            web_search_nodes.generate_answer(
                {
                    "query": "测试问题",
                    "browser_results": [
                        {
                            "rank": 1,
                            "title": "测试网页",
                            "url": "https://example.com",
                            "vision_description": "页面显示值为42",
                            "screenshot_base64": "THIS_MUST_NOT_REACH_DEEPSEEK",
                        }
                    ],
                }
            )

        model_input = model.invoke.call_args.args[0][1].content
        self.assertIn("页面显示值为42", model_input)
        self.assertNotIn("THIS_MUST_NOT_REACH_DEEPSEEK", model_input)

    def test_formal_search_uses_latest_human_message_each_turn(self) -> None:
        """同一Thread的新问题必须覆盖上一轮遗留的query。"""

        search_tool = Mock()
        search_tool.invoke.return_value = json.dumps(
            [
                {
                    "rank": 1,
                    "title": "上海天气",
                    "url": "https://example.com/weather",
                    "snippet": "天气摘要",
                    "content": "天气正文",
                }
            ],
            ensure_ascii=False,
        )

        with patch.object(web_search_nodes, "anysearch_web_search", search_tool):
            result = web_search_nodes.search_web(
                {
                    "messages": [
                        HumanMessage(content="现在几点了？"),
                        HumanMessage(content="北京现在几点了？"),
                    ],
                    "query": "现在几点了？",
                },
                vision_warmup=lambda: None,
            )

        self.assertEqual(result["query"], "北京现在几点了？")
        self.assertEqual(result["search_error"], "")
        search_tool.invoke.assert_called_once_with({"query": "北京现在几点了？"})

    def test_formal_search_prefers_planned_search_query(self) -> None:
        """主图传入补全后的搜索词时，不应退回省略的最新用户消息。"""

        search_tool = Mock()
        search_tool.invoke.return_value = "[]"

        with patch.object(web_search_nodes, "anysearch_web_search", search_tool):
            result = web_search_nodes.search_web(
                {
                    "messages": [HumanMessage(content="那后天呢？")],
                    "search_query": "上海后天天气",
                },
                vision_warmup=lambda: None,
            )

        self.assertEqual(result["query"], "上海后天天气")
        search_tool.invoke.assert_called_once_with({"query": "上海后天天气"})

    def test_vision_warmup_failure_keeps_anysearch_text_fallback(self) -> None:
        """视觉模型无法预热时仍应继续调用 AnySearch。"""

        search_tool = Mock()
        search_tool.invoke.return_value = json.dumps(
            [{"rank": 1, "title": "摘要", "url": "https://example.com", "content": "正文"}],
            ensure_ascii=False,
        )

        def reject_warmup() -> None:
            raise RuntimeError("视觉模型预热失败")

        with patch.object(web_search_nodes, "anysearch_web_search", search_tool):
            result = web_search_nodes.search_web(
                {"messages": [HumanMessage(content="查询最新天气")]},
                vision_warmup=reject_warmup,
            )

        self.assertEqual(result["search_error"], "")
        self.assertEqual(result["vision_error"], "RuntimeError: 视觉模型预热失败")
        self.assertTrue(result["search_results"][0]["vision_unavailable"])
        search_tool.invoke.assert_called_once()

    def test_vision_warmup_failure_ends_web_search_graph_after_anysearch(self) -> None:
        """预检失败时先走 AnySearch，再由文本证据继续处理。"""

        vision_client = Mock()
        vision_client.warm_up.side_effect = RuntimeError(
            "视觉模型预热失败"
        )
        page_reader = Mock()
        page_reader.side_effect = lambda page, *, capture_screenshot: {
            **page,
            "browser_status": "failed",
            "rendered_text": "",
            "json_responses": [],
            "browser_error": "Playwright unavailable",
            "screenshot_status": "failed",
            "screenshot_base64": "",
            "screenshot_error": "",
        }
        page_nodes = WebPageProcessingNodes(
            page_reader=page_reader,
            vision_factory=lambda: vision_client,
        )
        graph = build_web_search_graph(
            page_nodes=page_nodes,
            name="test_vision_warmup_failure",
        )
        search_tool = Mock()
        search_tool.invoke.return_value = json.dumps(
            [{"rank": 1, "title": "摘要", "url": "https://example.com", "content": "正文"}],
            ensure_ascii=False,
        )

        with patch.object(web_search_nodes, "anysearch_web_search", search_tool):
            result = graph.invoke(
                {"messages": [HumanMessage(content="查询最新天气")]}
            )

        self.assertEqual(result["search_error"], "")
        search_tool.invoke.assert_called_once()
        page_reader.assert_called_once()

    def test_all_browser_reads_failed_sets_search_error(self) -> None:
        """所有文本、JSON和视觉证据都失败时，应交给父图安全降级。"""

        failed_results = [
            {
                "rank": 1,
                "title": "失败页面",
                "url": "https://example.com/failure",
                "browser_status": "failed",
                "rendered_text": "",
                "json_responses": [],
                "browser_error": "timeout",
                "vision_status": "failed",
                "vision_relevant": False,
                "vision_description": "",
                "vision_error": "Ollama timeout",
            }
        ]
        result = web_search_nodes.finalize_webpages(
            {"page_results": failed_results}
        )

        self.assertIn("均不可用", result["search_error"])

    def test_anysearch_content_or_snippet_is_usable_when_playwright_fails(self) -> None:
        result = web_search_nodes.finalize_webpages(
            {
                "page_results": [
                    {
                        "rank": 1,
                        "title": "失败页面",
                        "url": "https://example.com/failure",
                        "browser_status": "failed",
                        "rendered_text": "",
                        "json_responses": [],
                        "content": "AnySearch 原始正文",
                        "snippet": "AnySearch 摘要",
                    }
                ]
            }
        )

        self.assertEqual(result["search_error"], "")

    def test_final_answer_failure_returns_ai_fallback(self) -> None:
        model = Mock()
        model.invoke.side_effect = TimeoutError("summary timeout")
        with patch.object(web_search_nodes, "build_chat_model", return_value=model):
            result = web_search_nodes.generate_answer(
                {
                    "query": "测试问题",
                    "browser_results": [{"content": "AnySearch 正文"}],
                }
            )

        self.assertEqual(result["messages"][0].type, "ai")
        self.assertIn("总结服务暂时不可用", result["messages"][0].content)


if __name__ == "__main__":
    unittest.main()
