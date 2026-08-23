"""通过本地 Ollama 生成上下文话题向量。"""

from __future__ import annotations

from functools import lru_cache
import math
import os

import httpx


EMBEDDING_DIMENSION = 1024


class OllamaEmbeddingError(RuntimeError):
    """Ollama embedding 请求或响应不符合预期。"""


class OllamaEmbeddingClient:
    """封装 Ollama `/api/embed`，对调用方隐藏 HTTP 和校验细节。"""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout: float = 30.0,
        expected_dimension: int = EMBEDDING_DIMENSION,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model.strip()
        self._timeout = timeout
        self._expected_dimension = expected_dimension
        if not self._base_url:
            raise ValueError("OLLAMA_BASE_URL 不能为空")
        if not self._model:
            raise ValueError("OLLAMA_EMBEDDING_MODEL 不能为空")
        if timeout <= 0:
            raise ValueError("OLLAMA_EMBEDDING_TIMEOUT 必须大于 0")

    def embed(self, text: str) -> list[float]:
        """为一段非空文本生成固定维度向量。"""

        text = text.strip()
        if not text:
            raise ValueError("embedding 文本不能为空")

        try:
            response = httpx.post(
                f"{self._base_url}/api/embed",
                json={"model": self._model, "input": text},
                timeout=self._timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError as error:
            raise OllamaEmbeddingError(f"Ollama embedding 请求失败：{error}") from error
        except ValueError as error:
            raise OllamaEmbeddingError("Ollama embedding 返回的不是合法 JSON") from error

        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, list) or not embeddings:
            raise OllamaEmbeddingError("Ollama 响应缺少 embeddings")
        vector = embeddings[0]
        if not isinstance(vector, list):
            raise OllamaEmbeddingError("Ollama 返回的首个 embedding 不是数组")

        try:
            normalized = [float(value) for value in vector]
        except (TypeError, ValueError) as error:
            raise OllamaEmbeddingError("embedding 中包含非数值元素") from error

        if len(normalized) != self._expected_dimension:
            raise OllamaEmbeddingError(
                f"embedding 维度应为 {self._expected_dimension}，"
                f"实际为 {len(normalized)}"
            )
        if not all(math.isfinite(value) for value in normalized):
            raise OllamaEmbeddingError("embedding 中包含非有限数值")
        return normalized


@lru_cache(maxsize=1)
def get_embedding_client() -> OllamaEmbeddingClient:
    """根据环境变量创建并复用 Ollama embedding 客户端。"""

    return OllamaEmbeddingClient(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        model=os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:0.6b"),
        timeout=float(os.getenv("OLLAMA_EMBEDDING_TIMEOUT", "8")),
    )
