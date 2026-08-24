"""专业 Agent 统一构造与上下文中间件测试。"""

import unittest
from unittest.mock import Mock, patch

from langchain.agents.middleware import ContextEditingMiddleware, SummarizationMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from pydantic import PrivateAttr

from agent.agents.factory import (
    SPECIALIST_BUDGET_POLICIES,
    SpecialistContext,
    build_specialist_agent,
)
from agent.agents.specialist_budget import (
    SpecialistBudgetMiddleware,
    SpecialistBudgetPolicy,
)
from agent.state.customer_service import SpecialistResult


class BudgetScriptModel(BaseChatModel):
    """按脚本返回工具调用，并记录每次实际模型请求。"""

    _responses: list[AIMessage] = PrivateAttr()
    _index: int = PrivateAttr(default=0)
    _bound_tools: list[list[str]] = PrivateAttr(default_factory=list)

    def __init__(self, responses: list[AIMessage]) -> None:
        super().__init__()
        self._responses = responses

    @property
    def _llm_type(self) -> str:
        return "budget-script-model"

    @property
    def call_count(self) -> int:
        return self._index

    @property
    def bound_tools(self) -> list[list[str]]:
        return self._bound_tools

    def bind_tools(self, tools, **kwargs):
        del kwargs
        self._bound_tools.append(
            [
                item.name if hasattr(item, "name") else item["function"]["name"]
                for item in tools
            ]
        )
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        del messages, stop, run_manager, kwargs
        response = self._responses[self._index]
        self._index += 1
        return ChatResult(generations=[ChatGeneration(message=response)])


@tool
def inspect_source(query: str) -> str:
    """读取一个测试来源。"""

    return f"evidence:{query}"


class SpecialistAgentFactoryTests(unittest.TestCase):
    def test_each_specialist_has_the_confirmed_business_tool_budget(self) -> None:
        self.assertEqual(
            {
                agent: policy.business_tool_calls
                for agent, policy in SPECIALIST_BUDGET_POLICIES.items()
            },
            {
                "general_qa_agent": 9,
                "rental_recommendation_agent": 8,
                "rental_booking_agent": 6,
                "order_history_agent": 4,
                "order_cancellation_agent": 6,
            },
        )
        self.assertTrue(
            all(policy.model_calls == 12 for policy in SPECIALIST_BUDGET_POLICIES.values())
        )

    def test_default_context_middlewares_are_applied_to_every_agent(self) -> None:
        model = Mock()
        compiled = Mock()
        with patch("agent.agents.factory.create_agent", return_value=compiled) as create:
            result = build_specialist_agent(
                name="test-agent",
                specialist_name="general_qa_agent",
                system_prompt="test",
                tools=[],
                model_factory=lambda: model,
            )

        self.assertIs(result, compiled)
        middleware = create.call_args.kwargs["middleware"]
        self.assertEqual(len(middleware), 3)
        self.assertIsInstance(middleware[0], ContextEditingMiddleware)
        self.assertIsInstance(middleware[1], SummarizationMiddleware)
        self.assertIsInstance(middleware[2], SpecialistBudgetMiddleware)
        edit = middleware[0].edits[0]
        self.assertEqual(edit.trigger, 8_000)
        self.assertEqual(edit.clear_at_least, 2_000)
        self.assertEqual(edit.keep, 3)
        self.assertEqual(middleware[1].trigger, ("tokens", 12_000))
        self.assertEqual(middleware[1].keep, ("messages", 20))
        self.assertIs(create.call_args.kwargs["model"], model)
        self.assertIs(create.call_args.kwargs["context_schema"], SpecialistContext)
        self.assertIs(create.call_args.kwargs["response_format"], SpecialistResult)
        prompt = create.call_args.kwargs["system_prompt"]
        self.assertIn("必须立即调用\nSpecialistResult", prompt)
        self.assertIn("普通文本不能结束任务", prompt)

    def test_explicit_middleware_can_replace_defaults_for_tests(self) -> None:
        custom = Mock()
        with patch("agent.agents.factory.create_agent", return_value=Mock()) as create:
            build_specialist_agent(
                name="test-agent",
                specialist_name="general_qa_agent",
                system_prompt="test",
                tools=[],
                model_factory=Mock,
                middleware=[custom],
            )

        middleware = create.call_args.kwargs["middleware"]
        self.assertEqual(middleware[0], custom)
        self.assertIsInstance(middleware[1], SpecialistBudgetMiddleware)

    def test_business_tool_limit_leaves_only_specialist_result_for_final_call(self) -> None:
        model = BudgetScriptModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "inspect_source",
                            "args": {"query": "first"},
                            "id": "source-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "SpecialistResult",
                            "args": {
                                "agent": "general_qa_agent",
                                "status": "success",
                                "summary": "已使用现有证据完成回答",
                                "user_facing_answer": "测试答案",
                                "completed_tasks": ["读取来源"],
                                "remaining_tasks": [],
                            },
                            "id": "result-1",
                            "type": "tool_call",
                        }
                    ],
                ),
            ]
        )
        graph = build_specialist_agent(
            name="budgeted-agent",
            specialist_name="general_qa_agent",
            system_prompt="测试提示词",
            tools=[inspect_source],
            model_factory=lambda: model,
            middleware=[],
            budget_policy=SpecialistBudgetPolicy(
                business_tool_calls=1,
                model_calls=12,
            ),
        )

        result = graph.invoke({"messages": [HumanMessage(content="查询测试来源")]})

        self.assertEqual(model.call_count, 2)
        self.assertEqual(model.bound_tools[0], ["inspect_source", "SpecialistResult"])
        self.assertEqual(model.bound_tools[1], ["SpecialistResult"])
        self.assertEqual(result["structured_response"]["status"], "success")
        self.assertEqual(result["specialist_budget"]["business_tool_calls"], 1)
        self.assertTrue(result["specialist_budget"]["final_attempt"])

    def test_final_attempt_falls_back_when_model_ignores_specialist_result(self) -> None:
        model = BudgetScriptModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "inspect_source",
                            "args": {"query": "first"},
                            "id": "source-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="我还想继续搜索。"),
            ]
        )
        graph = build_specialist_agent(
            name="fallback-agent",
            specialist_name="general_qa_agent",
            system_prompt="测试提示词",
            tools=[inspect_source],
            model_factory=lambda: model,
            middleware=[],
            budget_policy=SpecialistBudgetPolicy(
                business_tool_calls=1,
                model_calls=12,
            ),
        )

        result = graph.invoke({"messages": [HumanMessage(content="查询测试来源")]})

        self.assertEqual(model.call_count, 2)
        self.assertEqual(model.bound_tools[1], ["SpecialistResult"])
        self.assertEqual(result["structured_response"]["status"], "failed")
        self.assertEqual(
            result["structured_response"]["summary"],
            "专业 Agent 达到执行上限",
        )

    def test_model_call_limit_blocks_more_business_tools_and_returns_failure(self) -> None:
        model = BudgetScriptModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "inspect_source",
                            "args": {"query": "first"},
                            "id": "source-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "inspect_source",
                            "args": {"query": "second"},
                            "id": "source-2",
                            "type": "tool_call",
                        }
                    ],
                ),
            ]
        )
        graph = build_specialist_agent(
            name="model-limit-agent",
            specialist_name="general_qa_agent",
            system_prompt="测试提示词",
            tools=[inspect_source],
            model_factory=lambda: model,
            middleware=[],
            budget_policy=SpecialistBudgetPolicy(
                business_tool_calls=10,
                model_calls=2,
            ),
        )

        result = graph.invoke({"messages": [HumanMessage(content="反复查询来源")]})

        self.assertEqual(model.call_count, 2)
        self.assertEqual(result["structured_response"]["status"], "failed")
        self.assertEqual(result["specialist_budget"]["model_calls"], 2)
        self.assertEqual(result["specialist_budget"]["business_tool_calls"], 1)

    def test_compacted_context_does_not_reset_an_active_budget(self) -> None:
        middleware = SpecialistBudgetMiddleware(
            agent="general_qa_agent",
            policy=SpecialistBudgetPolicy(business_tool_calls=9),
        )
        state = {
            "messages": [AIMessage(content="此前消息已摘要")],
            "specialist_budget": {
                "owner_key": "human:original-message",
                "agent": "general_qa_agent",
                "model_calls": 5,
                "business_tool_calls": 4,
                "final_attempt": False,
                "block_model_call": False,
            },
        }

        update = middleware.before_agent(state, Mock())

        self.assertIsNone(update)


if __name__ == "__main__":
    unittest.main()
