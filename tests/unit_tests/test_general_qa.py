"""常规问答子图的离线测试。"""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from langchain_core.messages import AIMessage, HumanMessage

from agent.graph.general_qa import (
    _route_after_decision,
    _route_after_web_search,
    general_qa_graph,
)
from agent.node.general_qa import GeneralQANodes, SearchDecision


class FakeStructuredRouter:
    def __init__(self, owner: "FakeModel") -> None:
        self._owner = owner

    def invoke(self, messages):
        self._owner.router_messages = messages
        return self._owner.decision


class FakeModel:
    def __init__(self, decision: SearchDecision | None = None) -> None:
        self.decision = decision
        self.router_messages = None
        self.answer_messages = None
        self.structured_method = None
        self.answer = AIMessage(content="测试回答")

    def with_structured_output(self, schema, *, method=None):
        if schema is not SearchDecision:
            raise AssertionError(f"未处理的结构化类型：{schema}")
        self.structured_method = method
        return FakeStructuredRouter(self)

    def invoke(self, messages):
        self.answer_messages = messages
        return self.answer


class GeneralQANodeTests(unittest.TestCase):
    def test_context_completes_follow_up_search_query(self) -> None:
        """“那后天呢”应结合最近对话改写为可独立搜索的问题。"""

        model = FakeModel(
            SearchDecision(
                need_search=True,
                search_query="上海后天天气",
                reason="用户询问实时天气",
                requires_fresh_data=True,
            )
        )
        nodes = GeneralQANodes(model_factory=lambda: model)
        context = [
            HumanMessage(content="上海明天天气怎么样？"),
            AIMessage(content="明天可能有雨。"),
            HumanMessage(content="那后天呢？"),
        ]

        result = nodes.decide_search(
            {
                "messages": context,
                "context_messages": context,
            }
        )

        self.assertEqual(result["qa_route"], "search")
        self.assertEqual(result["search_query"], "上海后天天气")
        self.assertTrue(result["requires_fresh_data"])
        self.assertEqual(model.structured_method, "function_calling")
        router_prompt = model.router_messages[-1].content
        self.assertIn("上海明天天气怎么样", router_prompt)
        self.assertIn("那后天呢", router_prompt)

    def test_explicit_search_cannot_be_overridden_by_model(self) -> None:
        """用户明确要求联网时，即使模型误判也必须进入搜索路径。"""

        model = FakeModel(
            SearchDecision(
                need_search=False,
                search_query="",
                reason="模型误判为无需搜索",
                requires_fresh_data=False,
            )
        )
        nodes = GeneralQANodes(model_factory=lambda: model)

        result = nodes.decide_search(
            {"messages": [HumanMessage(content="帮我搜索 LangGraph 是什么")]}
        )

        self.assertEqual(result["qa_route"], "search")
        self.assertEqual(
            result["search_query"],
            "帮我搜索 LangGraph 是什么",
        )

    def test_greeting_uses_direct_rule_without_router_model(self) -> None:
        """简单问候不浪费一次路由 LLM 调用。"""

        nodes = GeneralQANodes(
            model_factory=lambda: (_ for _ in ()).throw(
                AssertionError("问候不应创建路由模型")
            )
        )

        result = nodes.decide_search(
            {"messages": [HumanMessage(content="你好！")]}
        )

        self.assertEqual(result["qa_route"], "direct")
        self.assertEqual(result["search_query"], "")

    def test_translation_with_time_word_stays_direct(self) -> None:
        """翻译任务中的“今天”只是待处理文本，不代表需要实时搜索。"""

        nodes = GeneralQANodes(
            model_factory=lambda: (_ for _ in ()).throw(
                AssertionError("明确翻译任务不应调用路由模型")
            )
        )

        result = nodes.decide_search(
            {"messages": [HumanMessage(content="翻译：今天天气很好")]}
        )

        self.assertEqual(result["qa_route"], "direct")

    def test_direct_answer_uses_filtered_context(self) -> None:
        """直接回答只读取 context_messages，不把被裁掉的旧消息重新加入。"""

        model = FakeModel()
        nodes = GeneralQANodes(model_factory=lambda: model)
        nodes.generate_direct_answer(
            {
                "messages": [HumanMessage(content="应被裁掉的旧消息")],
                "context_messages": [HumanMessage(content="当前有效问题")],
            }
        )

        prompt_text = "\n".join(str(item.content) for item in model.answer_messages)
        self.assertIn("当前有效问题", prompt_text)
        self.assertNotIn("应被裁掉的旧消息", prompt_text)

    def test_realtime_search_failure_does_not_call_model(self) -> None:
        """强实时问题搜索失败后不能用模型旧知识猜答案。"""

        nodes = GeneralQANodes(
            model_factory=lambda: (_ for _ in ()).throw(
                AssertionError("实时搜索失败后不应调用回答模型")
            )
        )

        result = nodes.generate_search_failure_answer(
            {
                "messages": [HumanMessage(content="今天上海天气如何？")],
                "requires_fresh_data": True,
                "search_error": "测试错误",
            }
        )

        self.assertIn("依赖实时信息", result["messages"][0].content)
        self.assertIn("无法可靠确认", result["messages"][0].content)


class GeneralQAGraphTests(unittest.TestCase):
    def test_graph_exposes_minimal_public_schema(self) -> None:
        """子图公共接口不应暴露上下文、路由和搜索过程状态。"""

        input_properties = general_qa_graph.get_input_jsonschema()["properties"]
        output_properties = general_qa_graph.get_output_jsonschema()["properties"]

        self.assertEqual(
            set(input_properties),
            {"messages", "user_id", "thread_id"},
        )
        self.assertEqual(set(output_properties), {"messages", "search_error"})

    def test_graph_contains_confirmed_pipeline(self) -> None:
        node_names = set(general_qa_graph.get_graph().nodes)
        self.assertTrue(
            {
                "prepare_context",
                "decide_search",
                "generate_direct_answer",
                "web_search",
                "generate_search_failure_answer",
            }
            <= node_names
        )

    def test_conditional_routes_use_state_values(self) -> None:
        self.assertEqual(_route_after_decision({"qa_route": "search"}), "search")
        self.assertEqual(_route_after_decision({"qa_route": "direct"}), "direct")
        self.assertEqual(
            _route_after_web_search({"search_error": "失败"}),
            "failure",
        )
        self.assertEqual(
            _route_after_web_search({"search_error": ""}),
            "success",
        )

    def test_langgraph_config_registers_general_qa_agent(self) -> None:
        config_path = Path(__file__).parents[2] / "langgraph.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(
            config["graphs"]["general_qa_agent"],
            "./src/agent/agents/general_qa.py:general_qa_agent",
        )


if __name__ == "__main__":
    unittest.main()
