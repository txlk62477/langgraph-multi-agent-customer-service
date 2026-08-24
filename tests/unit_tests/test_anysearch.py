"""AnySearch 和浏览器安全适配器的离线测试。"""

import os
import unittest
from unittest.mock import Mock, patch

from agent.common.anysearch import AnySearchError, search_anysearch
from agent.common.browser_reader import _compact_text, _is_allowed_url


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
                ]
            },
        }
        post.return_value = response

        with patch.dict(os.environ, {"ANYSEARCH_API_KEY": "test-key"}):
            results = search_anysearch("测试问题")

        self.assertEqual(
            results[0],
            {
                "rank": 1,
                "title": "示例标题",
                "url": "https://example.com/news",
                "snippet": "示例摘要",
                "content": "清洗后的网页正文",
            },
        )
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


if __name__ == "__main__":
    unittest.main()
