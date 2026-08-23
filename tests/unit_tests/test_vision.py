"""Ollama网页视觉识别客户端和截图保留策略测试。"""

import base64
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import time
import unittest
from unittest.mock import Mock, patch

from agent.common.vision import OllamaVisionClient, VisionEvidence
from agent.node.web_search import WebPageProcessingNodes


class OllamaVisionClientTests(unittest.TestCase):
    @patch("agent.common.vision.httpx.post")
    def test_warm_up_loads_model_and_refreshes_keep_alive(self, post: Mock) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        post.return_value = response
        client = OllamaVisionClient(
            base_url="http://ollama.test:11434",
            model="qwen3-vl:4b-instruct",
            timeout=30.0,
            warmup_timeout=10.0,
        )

        client.warm_up()

        post.assert_called_once_with(
            "http://ollama.test:11434/api/generate",
            json={
                "model": "qwen3-vl:4b-instruct",
                "stream": False,
                "keep_alive": "10m",
            },
            timeout=10.0,
        )

    @patch("agent.common.vision.httpx.post")
    def test_sends_base64_image_and_parses_structured_evidence(self, post: Mock) -> None:
        evidence = VisionEvidence(
            relevant=True,
            description="页面中央显示动态时间。",
            visible_facts=["时间为15:42:18", "时区为UTC+8"],
            uncertainties=[],
        )
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "message": {"role": "assistant", "content": evidence.model_dump_json()}
        }
        post.return_value = response
        client = OllamaVisionClient(
            base_url="http://ollama.test:11434",
            model="qwen3-vl:4b-instruct",
            timeout=30.0,
            warmup_timeout=10.0,
        )

        result = client.analyze_webpage(
            query="现在北京时间几点？",
            title="北京时间",
            url="https://example.com/time",
            screenshot_base64="anBlZw==",
        )

        self.assertTrue(result.relevant)
        self.assertIn("15:42:18", result.as_text())
        request = post.call_args.kwargs["json"]
        self.assertEqual(request["model"], "qwen3-vl:4b-instruct")
        self.assertEqual(request["messages"][0]["images"], ["anBlZw=="])
        self.assertFalse(request["stream"])
        self.assertIn("properties", request["format"])
        self.assertIn("现在北京时间几点", request["messages"][0]["content"])
        self.assertEqual(post.call_args.kwargs["timeout"], 30.0)


class ScreenshotRetentionTests(unittest.TestCase):
    @staticmethod
    def _page_reader(page, *, capture_screenshot):
        return {
            **page,
            "browser_status": "success",
            "rendered_text": "正文",
            "json_responses": [],
            "browser_error": "",
            "screenshot_status": "success",
            "screenshot_mime_type": "image/jpeg",
            "screenshot_base64": base64.b64encode(b"jpeg-bytes").decode(),
            "screenshot_error": "",
        }

    @staticmethod
    def _vision_factory():
        client = Mock()
        client.analyze_webpage.return_value = VisionEvidence(
            relevant=True,
            description="视觉证据",
            visible_facts=[],
            uncertainties=[],
        )
        return client

    def _run_nodes(self, nodes: WebPageProcessingNodes) -> dict:
        state = {
            "query": "测试问题",
            "page": {
                "rank": 1,
                "title": "测试网页",
                "url": "https://example.com",
            },
        }
        read_update = nodes.playwright_read_page(state)
        vision_update = nodes.analyze_page_visuals(
            {**state, **read_update}
        )
        return vision_update["page_results"][0]

    def test_none_mode_drops_screenshot_from_final_state(self) -> None:
        nodes = WebPageProcessingNodes(
            page_reader=self._page_reader,
            vision_factory=self._vision_factory,
            screenshot_retention="none",
        )

        result = self._run_nodes(nodes)

        self.assertNotIn("screenshot_base64", result)
        self.assertNotIn("screenshot_path", result)

    def test_state_mode_keeps_base64(self) -> None:
        nodes = WebPageProcessingNodes(
            page_reader=self._page_reader,
            vision_factory=self._vision_factory,
            screenshot_retention="state",
        )

        result = self._run_nodes(nodes)

        self.assertEqual(
            base64.b64decode(result["screenshot_base64"]),
            b"jpeg-bytes",
        )

    def test_disk_mode_writes_jpeg_and_keeps_only_path(self) -> None:
        with TemporaryDirectory() as directory:
            nodes = WebPageProcessingNodes(
                page_reader=self._page_reader,
                vision_factory=self._vision_factory,
                screenshot_retention="disk",
                screenshot_dir=directory,
            )

            result = self._run_nodes(nodes)
            path = Path(result["screenshot_path"])

            self.assertTrue(path.exists())
            self.assertEqual(path.read_bytes(), b"jpeg-bytes")
            self.assertNotIn("screenshot_base64", result)

    def test_vision_requests_are_limited_to_two(self) -> None:
        """即使三个网页同时到达视觉节点，也最多并行两个Ollama请求。"""

        lock = threading.Lock()
        active = 0
        maximum = 0

        class CountingVisionClient:
            def analyze_webpage(client_self, **kwargs):
                nonlocal active, maximum
                with lock:
                    active += 1
                    maximum = max(maximum, active)
                time.sleep(0.03)
                with lock:
                    active -= 1
                return VisionEvidence(
                    relevant=True,
                    description="视觉证据",
                    visible_facts=[],
                    uncertainties=[],
                )

        client = CountingVisionClient()
        nodes = WebPageProcessingNodes(
            page_reader=self._page_reader,
            vision_factory=lambda: client,
            vision_max_concurrency=2,
            screenshot_retention="none",
        )
        states = [
            {
                "query": "测试并发",
                "page": {"rank": rank},
                "page_result": {
                    "rank": rank,
                    "title": f"网页{rank}",
                    "url": f"https://example.com/{rank}",
                    "screenshot_status": "success",
                    "screenshot_base64": "anBlZw==",
                },
            }
            for rank in range(1, 4)
        ]

        with ThreadPoolExecutor(max_workers=3) as executor:
            list(executor.map(nodes.analyze_page_visuals, states))

        self.assertEqual(maximum, 2)


if __name__ == "__main__":
    unittest.main()
