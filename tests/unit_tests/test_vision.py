"""Ollama 网页视觉识别客户端测试。"""

import unittest
from unittest.mock import Mock, patch

from agent.common.vision import OllamaVisionClient, VisionEvidence


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


if __name__ == "__main__":
    unittest.main()
