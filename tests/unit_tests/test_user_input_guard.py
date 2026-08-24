"""用户输入保护中间件的硬规则与 LLM 组合判定测试。"""

from __future__ import annotations

import unittest

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

from agent.agents.user_input_guard import (
    UserInputDecision,
    UserInputGuardMiddleware,
)


class ClassifierModel:
    def __init__(self, decision=None, error: Exception | None = None) -> None:
        self.decision = decision
        self.error = error
        self.calls = 0
        self.messages = None

    def with_structured_output(self, schema, *, method=None):
        if schema is not UserInputDecision or method != "function_calling":
            raise AssertionError((schema, method))
        return self

    def invoke(self, messages):
        self.calls += 1
        self.messages = messages
        if self.error is not None:
            raise self.error
        return self.decision


class UserInputGuardTests(unittest.TestCase):
    def _guard(self, model: ClassifierModel) -> UserInputGuardMiddleware:
        return UserInputGuardMiddleware(
            classifier_model=model,
            agent_role="测试专业任务",
        )

    def test_hard_request_rule_skips_classifier_and_creates_tool_call(self) -> None:
        model = ClassifierModel(error=AssertionError("不应调用分类器"))
        result = self._guard(model).after_model(
            {
                "messages": [
                    AIMessage(content="请选择具体房源，并提供手机号和入住日期。")
                ]
            },
            Runtime(),
        )

        self.assertEqual(model.calls, 0)
        call = result["messages"][0].tool_calls[0]
        self.assertEqual(call["name"], "request_user_input")
        self.assertEqual(
            call["args"]["missing_fields"],
            ["house_id", "phone", "check_in_date"],
        )

    def test_ambiguous_question_uses_structured_classifier(self) -> None:
        model = ClassifierModel(
            UserInputDecision(
                requires_user_input=True,
                reason="必须确定候选订单",
                missing_fields=["order_no"],
            )
        )
        result = self._guard(model).after_model(
            {
                "messages": [
                    HumanMessage(content="取消我的订单"),
                    AIMessage(content="第一笔和第二笔订单，您倾向处理哪一笔？"),
                ]
            },
            Runtime(),
        )

        self.assertEqual(model.calls, 1)
        call = result["messages"][0].tool_calls[0]
        self.assertEqual(call["args"]["reason"], "必须确定候选订单")
        self.assertEqual(call["args"]["missing_fields"], ["order_no"])

    def test_classifier_false_leaves_ordinary_final_answer_unchanged(self) -> None:
        model = ClassifierModel(
            UserInputDecision(
                requires_user_input=False,
                reason="只是礼貌结束",
            )
        )
        result = self._guard(model).after_model(
            {"messages": [AIMessage(content="这些是推荐结果，希望能帮到您。")]},
            Runtime(),
        )

        self.assertEqual(result["last_guard_event"]["result"], "pass")
        self.assertEqual(result["last_guard_event"]["source"], "llm")
        self.assertEqual(result["last_guard_event"]["reason"], "只是礼貌结束")
        self.assertEqual(model.calls, 1)

    def test_explicit_closing_rule_skips_classifier(self) -> None:
        model = ClassifierModel(error=AssertionError("不应调用分类器"))
        result = self._guard(model).after_model(
            {"messages": [AIMessage(content="如有需要，随时可以继续咨询。")]},
            Runtime(),
        )

        self.assertEqual(result["last_guard_event"]["result"], "terminal")
        self.assertEqual(result["last_guard_event"]["source"], "hard_rule")
        self.assertEqual(model.calls, 0)

    def test_classifier_failure_is_fail_open(self) -> None:
        model = ClassifierModel(error=RuntimeError("分类服务不可用"))
        result = self._guard(model).after_model(
            {
                "messages": [
                    AIMessage(
                        content="您倾向第一套还是第二套？",
                        id="ambiguous-question",
                    )
                ]
            },
            Runtime(),
        )

        self.assertNotIn("messages", result)
        self.assertEqual(
            result["last_guard_event"],
            {
                "message_id": "ambiguous-question",
                "result": "pass",
                "source": "fallback",
                "requires_user_input": False,
                "missing_fields": [],
                "reason": "分类器失败且没有识别出明确缺失字段",
                "error": "RuntimeError: 分类服务不可用",
            },
        )
        self.assertEqual(model.calls, 1)

    def test_classifier_failure_with_explicit_fields_falls_back_to_interrupt(self) -> None:
        model = ClassifierModel(error=RuntimeError("分类服务不可用"))
        message = AIMessage(
            content="请问您的手机号是多少？入住日期和退房日期分别是哪天？",
            id="explicit-fields-question",
        )

        result = self._guard(model).after_model(
            {"messages": [message]},
            Runtime(),
        )

        call = result["messages"][0].tool_calls[0]
        self.assertEqual(call["name"], "request_user_input")
        self.assertEqual(
            call["args"]["missing_fields"],
            ["phone", "check_in_date", "check_out_date"],
        )
        self.assertEqual(
            result["last_guard_event"],
            {
                "message_id": "explicit-fields-question",
                "result": "request",
                "source": "fallback",
                "requires_user_input": True,
                "missing_fields": ["phone", "check_in_date", "check_out_date"],
                "reason": "分类器失败，但已识别出明确缺失字段",
                "error": "RuntimeError: 分类服务不可用",
            },
        )

    def test_existing_tool_call_is_never_modified(self) -> None:
        model = ClassifierModel(error=AssertionError("不应调用分类器"))
        result = self._guard(model).after_model(
            {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "request_user_input",
                                "args": {"question": "请选择", "reason": "缺少选择"},
                                "id": "existing-call",
                                "type": "tool_call",
                            }
                        ],
                    )
                ]
            },
            Runtime(),
        )

        self.assertIsNone(result)
        self.assertEqual(model.calls, 0)

    def test_classifier_receives_only_six_previous_messages(self) -> None:
        model = ClassifierModel(
            UserInputDecision(
                requires_user_input=False,
                reason="无需等待",
            )
        )
        messages = [HumanMessage(content=f"context-{index}") for index in range(8)]
        messages.append(AIMessage(content="当前是否已经足够？"))

        self._guard(model).after_model({"messages": messages}, Runtime())

        prompt = model.messages[-1].content
        self.assertNotIn("context-0", prompt)
        self.assertNotIn("context-1", prompt)
        for index in range(2, 8):
            self.assertIn(f"context-{index}", prompt)


if __name__ == "__main__":
    unittest.main()
