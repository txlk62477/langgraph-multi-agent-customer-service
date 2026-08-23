"""PostgreSQL pgvector 话题记忆仓储。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import os
from typing import Any
from uuid import uuid4

import psycopg
from psycopg.types.json import Jsonb

from agent.common.embedding import EMBEDDING_DIMENSION


@dataclass(frozen=True, slots=True)
class TopicMemory:
    """从数据库读取的一条历史话题摘要。"""

    memory_id: str
    topic: str
    summary: str
    important_facts: list[str]
    source_message_ids: list[str]
    similarity: float | None = None

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "memory_id": self.memory_id,
            "topic": self.topic,
            "summary": self.summary,
            "important_facts": self.important_facts,
        }
        if self.similarity is not None:
            result["similarity"] = self.similarity
        return result


class PostgresTopicMemoryRepository:
    """隐藏 pgvector 查询、JSONB 和消息去重等持久化细节。"""

    def __init__(self, postgres_uri: str) -> None:
        postgres_uri = postgres_uri.strip()
        if not postgres_uri:
            raise ValueError("POSTGRES_URI 不能为空")
        self._postgres_uri = postgres_uri

    def search_similar(
        self,
        *,
        user_id: str,
        thread_id: str,
        embedding: list[float],
        limit: int,
    ) -> list[TopicMemory]:
        """在当前用户和线程内按余弦相似度检索话题。"""

        user_id, thread_id = _validate_scope(user_id, thread_id)
        vector = _vector_literal(embedding)
        if limit <= 0:
            return []

        with psycopg.connect(self._postgres_uri) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT memory_id, topic, summary, important_facts,
                           source_message_ids,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM public.conversation_topic_memory
                    WHERE user_id = %s AND thread_id = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (vector, user_id, thread_id, vector, limit),
                )
                rows = cursor.fetchall()
        return [_row_to_memory(row, has_similarity=True) for row in rows]

    def recent(
        self,
        *,
        user_id: str,
        thread_id: str,
        limit: int,
    ) -> list[TopicMemory]:
        """Ollama 不可用时，按更新时间读取最近的摘要候选。"""

        user_id, thread_id = _validate_scope(user_id, thread_id)
        if limit <= 0:
            return []
        with psycopg.connect(self._postgres_uri) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT memory_id, topic, summary, important_facts,
                           source_message_ids
                    FROM public.conversation_topic_memory
                    WHERE user_id = %s AND thread_id = %s
                    ORDER BY updated_at DESC
                    LIMIT %s
                    """,
                    (user_id, thread_id, limit),
                )
                rows = cursor.fetchall()
        return [_row_to_memory(row, has_similarity=False) for row in rows]

    def processed_message_ids(self, *, user_id: str, thread_id: str) -> set[str]:
        """返回已经写入任意话题摘要的原始消息标识。"""

        user_id, thread_id = _validate_scope(user_id, thread_id)
        with psycopg.connect(self._postgres_uri) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT DISTINCT unnest(source_message_ids)
                    FROM public.conversation_topic_memory
                    WHERE user_id = %s AND thread_id = %s
                    """,
                    (user_id, thread_id),
                )
                return {str(row[0]) for row in cursor.fetchall()}

    def important_facts(self, *, user_id: str, thread_id: str) -> list[str]:
        """读取线程内所有去重后的重要事实，保持最新摘要优先。"""

        user_id, thread_id = _validate_scope(user_id, thread_id)
        with psycopg.connect(self._postgres_uri) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT important_facts
                    FROM public.conversation_topic_memory
                    WHERE user_id = %s AND thread_id = %s
                    ORDER BY updated_at DESC
                    """,
                    (user_id, thread_id),
                )
                rows = cursor.fetchall()

        facts: list[str] = []
        seen: set[str] = set()
        for (row_facts,) in rows:
            for fact in row_facts or []:
                cleaned = str(fact).strip()
                if cleaned and cleaned not in seen:
                    facts.append(cleaned)
                    seen.add(cleaned)
        return facts

    def insert(
        self,
        *,
        user_id: str,
        thread_id: str,
        topic: str,
        summary: str,
        important_facts: list[str],
        embedding: list[float],
        source_message_ids: list[str],
    ) -> TopicMemory:
        """新增一条动态话题摘要。"""

        user_id, thread_id = _validate_scope(user_id, thread_id)
        memory_id = str(uuid4())
        vector = _vector_literal(embedding)
        facts = _clean_strings(important_facts)
        message_ids = _clean_strings(source_message_ids)
        topic, summary = _validate_topic_text(topic, summary)

        with psycopg.connect(self._postgres_uri) as connection:
            connection.execute(
                """
                INSERT INTO public.conversation_topic_memory (
                    memory_id, user_id, thread_id, topic, summary,
                    important_facts, embedding, source_message_ids
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s)
                """,
                (
                    memory_id,
                    user_id,
                    thread_id,
                    topic,
                    summary,
                    Jsonb(facts),
                    vector,
                    message_ids,
                ),
            )
        return TopicMemory(memory_id, topic, summary, facts, message_ids)

    def update(
        self,
        *,
        memory_id: str,
        user_id: str,
        thread_id: str,
        topic: str,
        summary: str,
        important_facts: list[str],
        embedding: list[float],
        source_message_ids: list[str],
    ) -> TopicMemory:
        """更新一条已由 DeepSeek 确认属于同一话题的摘要。"""

        user_id, thread_id = _validate_scope(user_id, thread_id)
        topic, summary = _validate_topic_text(topic, summary)
        facts = _clean_strings(important_facts)
        message_ids = _clean_strings(source_message_ids)
        vector = _vector_literal(embedding)

        with psycopg.connect(self._postgres_uri) as connection:
            cursor = connection.execute(
                """
                UPDATE public.conversation_topic_memory
                SET topic = %s,
                    summary = %s,
                    important_facts = %s,
                    embedding = %s::vector,
                    source_message_ids = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE memory_id = %s AND user_id = %s AND thread_id = %s
                """,
                (
                    topic,
                    summary,
                    Jsonb(facts),
                    vector,
                    message_ids,
                    memory_id,
                    user_id,
                    thread_id,
                ),
            )
            if cursor.rowcount != 1:
                raise LookupError("要更新的话题记忆不存在或不属于当前会话")
        return TopicMemory(memory_id, topic, summary, facts, message_ids)


def _row_to_memory(row: tuple[Any, ...], *, has_similarity: bool) -> TopicMemory:
    similarity = float(row[5]) if has_similarity else None
    return TopicMemory(
        memory_id=str(row[0]),
        topic=str(row[1]),
        summary=str(row[2]),
        important_facts=[str(item) for item in (row[3] or [])],
        source_message_ids=[str(item) for item in (row[4] or [])],
        similarity=similarity,
    )


def _vector_literal(embedding: list[float]) -> str:
    if len(embedding) != EMBEDDING_DIMENSION:
        raise ValueError(
            f"话题向量维度应为 {EMBEDDING_DIMENSION}，实际为 {len(embedding)}"
        )
    values = [float(value) for value in embedding]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("话题向量包含非有限数值")
    return "[" + ",".join(format(value, ".9g") for value in values) + "]"


def _validate_scope(user_id: str, thread_id: str) -> tuple[str, str]:
    user_id = user_id.strip()
    thread_id = thread_id.strip()
    if not user_id or not thread_id:
        raise ValueError("user_id 和 thread_id 不能为空")
    if len(user_id) > 128 or len(thread_id) > 128:
        raise ValueError("user_id 和 thread_id 不能超过 128 个字符")
    return user_id, thread_id


def _validate_topic_text(topic: str, summary: str) -> tuple[str, str]:
    topic = topic.strip()
    summary = summary.strip()
    if not topic or not summary:
        raise ValueError("话题名称和摘要不能为空")
    return topic, summary


def _clean_strings(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


@lru_cache(maxsize=1)
def get_topic_memory_repository() -> PostgresTopicMemoryRepository:
    postgres_uri = os.getenv("POSTGRES_URI", "").strip()
    if not postgres_uri:
        raise ValueError("缺少 POSTGRES_URI，请在 .env 中配置 PostgreSQL 连接")
    return PostgresTopicMemoryRepository(postgres_uri)
