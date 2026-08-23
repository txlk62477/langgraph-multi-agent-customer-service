"""独立上下文管理节点的离线测试。"""

from __future__ import annotations

import os
import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.common.embedding import (
    EMBEDDING_DIMENSION,
    OllamaEmbeddingError,
    get_embedding_client,
)
from agent.common.topic_memory import TopicMemory
from agent.node.context import (
    ContextNodes,
    TopicExtraction,
    TopicMergeDecision,
    TopicSegment,
    TopicSelection,
)


class FakeEmbeddingClient:
    def embed(self, text: str) -> list[float]:
        return [0.01] * EMBEDDING_DIMENSION


class FailingEmbeddingClient:
    def embed(self, text: str) -> list[float]:
        raise OllamaEmbeddingError("测试中的Ollama不可用")


class OllamaEmbeddingConfigurationTests(unittest.TestCase):
    @patch("agent.common.embedding.httpx.post")
    def test_factory_uses_configured_short_timeout(self, post: Mock) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "embeddings": [[0.01] * EMBEDDING_DIMENSION]
        }
        post.return_value = response

        get_embedding_client.cache_clear()
        try:
            with patch.dict(
                os.environ,
                {"OLLAMA_EMBEDDING_TIMEOUT": "8"},
            ):
                get_embedding_client().embed("测试")
        finally:
            get_embedding_client.cache_clear()

        self.assertEqual(post.call_args.kwargs["timeout"], 8.0)


class FakeRepository:
    def __init__(self, memories: list[TopicMemory] | None = None) -> None:
        self.memories = list(memories or [])
        self.updated = False

    def processed_message_ids(self, *, user_id: str, thread_id: str) -> set[str]:
        return {
            message_id
            for memory in self.memories
            for message_id in memory.source_message_ids
        }

    def search_similar(
        self,
        *,
        user_id: str,
        thread_id: str,
        embedding: list[float],
        limit: int,
    ) -> list[TopicMemory]:
        return [
            TopicMemory(
                memory.memory_id,
                memory.topic,
                memory.summary,
                memory.important_facts,
                memory.source_message_ids,
                0.90,
            )
            for memory in self.memories[:limit]
        ]

    def recent(
        self, *, user_id: str, thread_id: str, limit: int
    ) -> list[TopicMemory]:
        return self.memories[:limit]

    def important_facts(self, *, user_id: str, thread_id: str) -> list[str]:
        return [
            fact
            for memory in self.memories
            for fact in memory.important_facts
        ]

    def insert(self, **kwargs) -> TopicMemory:
        memory = TopicMemory(
            "memory-new",
            kwargs["topic"],
            kwargs["summary"],
            kwargs["important_facts"],
            kwargs["source_message_ids"],
        )
        self.memories.insert(0, memory)
        return memory

    def update(self, **kwargs) -> TopicMemory:
        self.updated = True
        memory = TopicMemory(
            kwargs["memory_id"],
            kwargs["topic"],
            kwargs["summary"],
            kwargs["important_facts"],
            kwargs["source_message_ids"],
        )
        self.memories = [
            memory if item.memory_id == memory.memory_id else item
            for item in self.memories
        ]
        return memory


class FakeStructuredModel:
    def __init__(self, schema, owner: "FakeModel") -> None:
        self._schema = schema
        self._owner = owner

    def invoke(self, messages):
        if self._schema is TopicExtraction:
            return TopicExtraction(
                segments=[
                    TopicSegment(
                        topic="北京租房",
                        summary="用户正在寻找北京租房。",
                        important_facts=["用户关注北京房源"],
                        message_numbers=[1, 2],
                    )
                ]
            )
        if self._schema is TopicMergeDecision:
            return TopicMergeDecision(
                action=self._owner.merge_action,
                topic="北京租房",
                summary="用户寻找北京租房，预算已更新。",
                important_facts=["用户关注北京房源", "预算以最新说明为准"],
            )
        if self._schema is TopicSelection:
            if self._owner.fail_selection:
                raise RuntimeError("测试中的DeepSeek筛选失败")
            return TopicSelection(
                current_topic="北京租房",
                selected_memory_ids=self._owner.selection_ids,
            )
        raise AssertionError(f"未处理的结构化模型：{self._schema}")


class FakeModel:
    def __init__(
        self,
        *,
        selection_ids: list[str],
        merge_action: str = "create",
        fail_selection: bool = False,
    ) -> None:
        self.selection_ids = selection_ids
        self.merge_action = merge_action
        self.fail_selection = fail_selection

    def with_structured_output(self, schema, *, method=None):
        self.method = method
        return FakeStructuredModel(schema, self)


def conversation_messages(count: int):
    messages = []
    for index in range(count):
        message_type = HumanMessage if index % 2 == 0 else AIMessage
        messages.append(
            message_type(content=f"第{index + 1}条对话", id=f"message-{index + 1}")
        )
    return messages


class ContextNodeTests(unittest.TestCase):
    def test_summarizes_old_messages_and_keeps_recent_ten(self) -> None:
        repository = FakeRepository()
        model = FakeModel(selection_ids=["memory-new"])
        nodes = ContextNodes(
            model_factory=lambda: model,
            embedding_factory=FakeEmbeddingClient,
            repository_factory=lambda: repository,
        )
        original_messages = [SystemMessage(content="系统规则")] + conversation_messages(12)

        result = nodes.prepare_context(
            {"messages": original_messages},
            {"configurable": {"user_id": "u1", "thread_id": "t1"}},
        )

        self.assertNotIn("messages", result)
        self.assertEqual(result["context_stats"]["summarized_message_count"], 2)
        self.assertEqual(result["context_stats"]["selected_summary_count"], 1)
        self.assertEqual(result["context_stats"]["retrieval_mode"], "vector")
        self.assertEqual(
            result["context_messages"][-10:], original_messages[-10:]
        )
        self.assertEqual(repository.memories[0].source_message_ids, ["id:message-1", "id:message-2"])

    def test_similar_topic_is_merged_only_after_model_decision(self) -> None:
        existing = TopicMemory(
            "memory-old",
            "北京租房",
            "用户寻找北京住房。",
            ["用户关注北京房源"],
            [],
        )
        repository = FakeRepository([existing])
        model = FakeModel(selection_ids=["memory-old"], merge_action="merge")
        nodes = ContextNodes(
            model_factory=lambda: model,
            embedding_factory=FakeEmbeddingClient,
            repository_factory=lambda: repository,
        )

        result = nodes.prepare_context(
            {"messages": conversation_messages(12)},
            {"configurable": {"user_id": "u1", "thread_id": "t1"}},
        )

        self.assertTrue(repository.updated)
        self.assertEqual(len(repository.memories), 1)
        self.assertEqual(result["active_topic"], "北京租房")

    def test_ollama_failure_uses_recent_summaries(self) -> None:
        existing = TopicMemory(
            "memory-old",
            "北京租房",
            "历史摘要",
            [],
            [],
        )
        repository = FakeRepository([existing])
        model = FakeModel(selection_ids=["memory-old"])
        nodes = ContextNodes(
            model_factory=lambda: model,
            embedding_factory=FailingEmbeddingClient,
            repository_factory=lambda: repository,
        )

        result = nodes.prepare_context(
            {"messages": conversation_messages(4)},
            {"configurable": {"user_id": "u1", "thread_id": "t1"}},
        )

        self.assertEqual(
            result["context_stats"]["retrieval_mode"],
            "recent_summary_fallback",
        )
        self.assertEqual(result["context_stats"]["selected_summary_count"], 1)

    def test_deepseek_selection_failure_keeps_only_recent_messages(self) -> None:
        repository = FakeRepository()
        model = FakeModel(selection_ids=[], fail_selection=True)
        nodes = ContextNodes(
            model_factory=lambda: model,
            embedding_factory=FakeEmbeddingClient,
            repository_factory=lambda: repository,
        )
        messages = conversation_messages(12)

        result = nodes.prepare_context(
            {"messages": messages},
            {"configurable": {"user_id": "u1", "thread_id": "t1"}},
        )

        self.assertEqual(
            result["context_stats"]["retrieval_mode"], "recent_messages_only"
        )
        self.assertEqual(result["context_messages"], messages[-10:])


if __name__ == "__main__":
    unittest.main()
