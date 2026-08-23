"""Offline tests for the reusable information-collection subgraph."""

import unittest
from collections import deque
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from agent.common.collection import CollectionSpec, find_missing_required_fields
from agent.graph.information_collection import build_information_collection_graph
from agent.node.information_collection import InformationCollectionNodes
from agent.state.information_collection import (
    RecommendCollectionState,
    RecommendInformation,
)


class FakeStructuredModel:
    def __init__(self, outputs: list[dict]) -> None:
        self._outputs = deque(outputs)
        self._schema = RecommendInformation
        self.structured_output_method = None

    def with_structured_output(self, schema, *, method=None):
        self._schema = schema
        self.structured_output_method = method
        return self

    def invoke(self, _messages):
        return self._schema.model_validate(self._outputs.popleft())


def build_test_graph(outputs: list[dict], max_llm_calls: int = 3):
    spec = CollectionSpec(
        required_fields={
            "city": "租房城市",
            "budget_min": "最低预算",
            "budget_max": "最高预算",
        },
        optional_fields={
            "districts": "区域",
            "room_types": "房型",
            "rental_mode": "租赁方式",
        },
        max_llm_calls=max_llm_calls,
    )
    fake_model = FakeStructuredModel(outputs)
    subgraph = build_information_collection_graph(
        state_schema=RecommendCollectionState,
        extraction_schema=RecommendInformation,
        spec=spec,
        model_factory=lambda: fake_model,
    )

    parent = StateGraph(RecommendCollectionState)
    parent.add_node("collect_information", subgraph)
    parent.add_edge(START, "collect_information")
    parent.add_edge("collect_information", END)
    return parent.compile(checkpointer=InMemorySaver())


class CollectionSpecTests(unittest.TestCase):
    def test_required_and_optional_fields_cannot_overlap(self) -> None:
        with self.assertRaisesRegex(ValueError, "同时"):
            CollectionSpec(
                required_fields={"city": "城市"},
                optional_fields={"city": "偏好城市"},
            )

    def test_only_required_fields_block_completion(self) -> None:
        spec = CollectionSpec(
            required_fields={"city": "城市", "budget_min": "最低预算"},
            optional_fields={"district": "区域"},
        )
        self.assertEqual(
            find_missing_required_fields(
                {"city": "杭州", "budget_min": 0, "district": None}, spec
            ),
            [],
        )


class InformationCollectionGraphTests(unittest.TestCase):
    def config(self) -> dict:
        return {"configurable": {"thread_id": str(uuid4())}}

    def test_extraction_uses_deepseek_function_calling(self) -> None:
        spec = CollectionSpec(
            required_fields={"city": "租房城市"},
            max_llm_calls=1,
        )
        fake_model = FakeStructuredModel([{"city": "杭州"}])
        nodes = InformationCollectionNodes(
            spec=spec,
            extraction_schema=RecommendInformation,
            model_factory=lambda: fake_model,
        )

        nodes.extract({"messages": [HumanMessage(content="我想在杭州租房")]})

        self.assertEqual(fake_model.structured_output_method, "function_calling")

    def test_completes_when_first_extraction_has_all_required_fields(self) -> None:
        graph = build_test_graph(
            [{"city": "杭州", "budget_min": 2000, "budget_max": 3500}]
        )

        result = graph.invoke(
            {"messages": [HumanMessage(content="杭州租房，预算两千到三千五")]},
            self.config(),
        )

        self.assertEqual(result["collection_status"], "complete")
        self.assertEqual(result["missing_required_fields"], [])
        self.assertEqual(result["llm_call_count"], 1)
        self.assertIsNone(result.get("districts"))

    def test_interrupts_then_resumes_until_required_fields_are_complete(self) -> None:
        graph = build_test_graph(
            [
                {"city": "杭州"},
                {"budget_min": 2000, "budget_max": 3500},
            ]
        )
        config = self.config()

        interrupted = graph.invoke(
            {"messages": [HumanMessage(content="我想在杭州租房")]}, config
        )
        payload = interrupted["__interrupt__"][0].value
        self.assertEqual(
            payload["missing_required_fields"], ["budget_min", "budget_max"]
        )

        result = graph.invoke(Command(resume="预算 2000 到 3500 元"), config)

        self.assertEqual(result["collection_status"], "complete")
        self.assertEqual(result["llm_call_count"], 2)
        self.assertEqual(result["city"], "杭州")
        self.assertEqual(result["budget_max"], 3500)

    def test_stops_as_incomplete_after_maximum_llm_calls(self) -> None:
        graph = build_test_graph(
            [{"city": "杭州"}, {"budget_min": 2000}], max_llm_calls=2
        )
        config = self.config()

        interrupted = graph.invoke(
            {"messages": [HumanMessage(content="我想在杭州租房")]}, config
        )
        self.assertIn("__interrupt__", interrupted)

        result = graph.invoke(Command(resume="最低预算 2000 元"), config)

        self.assertEqual(result["collection_status"], "incomplete")
        self.assertEqual(result["missing_required_fields"], ["budget_max"])
        self.assertEqual(result["llm_call_count"], 2)
        self.assertNotIn("__interrupt__", result)

    def test_llm_failure_finishes_collection_without_repeated_interrupt(self) -> None:
        class FailingModel:
            def with_structured_output(self, *_args, **_kwargs):
                return self

            def invoke(self, _messages):
                raise TimeoutError("LLM timeout")

        spec = CollectionSpec(required_fields={"city": "租房城市"}, max_llm_calls=3)
        graph = build_information_collection_graph(
            state_schema=RecommendCollectionState,
            extraction_schema=RecommendInformation,
            spec=spec,
            model_factory=FailingModel,
        )
        result = graph.invoke(
            {"messages": [HumanMessage(content="我想租房")]},
            self.config(),
        )

        self.assertNotIn("__interrupt__", result)
        self.assertEqual(result["collection_status"], "incomplete")
        self.assertIn("LLM timeout", result["collection_error"])


if __name__ == "__main__":
    unittest.main()
