"""专业 Agent 统一构造与上下文中间件测试。"""

import unittest
from unittest.mock import Mock, patch

from langchain.agents.middleware import ContextEditingMiddleware, SummarizationMiddleware

from agent.agents.factory import SpecialistContext, build_specialist_agent


class SpecialistAgentFactoryTests(unittest.TestCase):
    def test_default_context_middlewares_are_applied_to_every_agent(self) -> None:
        model = Mock()
        compiled = Mock()
        with patch("agent.agents.factory.create_agent", return_value=compiled) as create:
            result = build_specialist_agent(
                name="test-agent",
                system_prompt="test",
                tools=[],
                model_factory=lambda: model,
            )

        self.assertIs(result, compiled)
        middleware = create.call_args.kwargs["middleware"]
        self.assertEqual(len(middleware), 2)
        self.assertIsInstance(middleware[0], ContextEditingMiddleware)
        self.assertIsInstance(middleware[1], SummarizationMiddleware)
        edit = middleware[0].edits[0]
        self.assertEqual(edit.trigger, 8_000)
        self.assertEqual(edit.clear_at_least, 2_000)
        self.assertEqual(edit.keep, 3)
        self.assertEqual(middleware[1].trigger, ("tokens", 12_000))
        self.assertEqual(middleware[1].keep, ("messages", 20))
        self.assertIs(create.call_args.kwargs["model"], model)
        self.assertIs(create.call_args.kwargs["context_schema"], SpecialistContext)

    def test_explicit_middleware_can_replace_defaults_for_tests(self) -> None:
        custom = Mock()
        with patch("agent.agents.factory.create_agent", return_value=Mock()) as create:
            build_specialist_agent(
                name="test-agent",
                system_prompt="test",
                tools=[],
                model_factory=Mock,
                middleware=[custom],
            )

        self.assertEqual(create.call_args.kwargs["middleware"], [custom])


if __name__ == "__main__":
    unittest.main()
