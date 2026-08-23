"""Tests for the configured chat model adapter."""

import os
import unittest
from unittest.mock import patch

from langchain_deepseek import ChatDeepSeek

from agent.common.llm import build_chat_model


class ChatModelTests(unittest.TestCase):
    def test_builds_official_deepseek_integration(self) -> None:
        environment = {
            "DEEPSEEK_API_KEY": "test-key",
            "DEEPSEEK_BASE_URL": "https://api.deepseek.com",
            "DEEPSEEK_MODEL": "deepseek-chat",
            "MAX_OUTPUT_TOKENS": "2048",
            "DEEPSEEK_REQUEST_TIMEOUT": "37",
            "DEEPSEEK_MAX_RETRIES": "2",
        }

        with patch.dict(os.environ, environment, clear=False):
            model = build_chat_model()

        self.assertIsInstance(model, ChatDeepSeek)
        self.assertEqual(model.model_name, "deepseek-chat")
        self.assertEqual(
            str(model.api_base).rstrip("/"), "https://api.deepseek.com"
        )
        self.assertEqual(model.request_timeout, 37.0)
        self.assertEqual(model.max_retries, 2)


if __name__ == "__main__":
    unittest.main()
