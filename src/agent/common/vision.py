"""调用本地 Ollama 视觉模型，从网页截图中提取问题相关证据。"""

from __future__ import annotations

from functools import lru_cache
import os

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator


class OllamaVisionError(RuntimeError):
    """Ollama 视觉请求失败或响应结构不符合预期。"""


class VisionEvidence(BaseModel):
    """视觉模型从单张网页截图中提取的结构化证据。"""

    model_config = ConfigDict(extra="forbid")

    relevant: bool = Field(description="截图是否包含与用户问题相关的可见信息")
    description: str = Field(description="对截图中相关信息的简短客观说明")
    visible_facts: list[str] = Field(
        default_factory=list,
        description="截图中能够直接读到的时间、价格、状态、图表或表格事实",
    )
    uncertainties: list[str] = Field(
        default_factory=list,
        description="看不清、可能误读或无法从截图确认的内容",
    )

    @field_validator("description")
    @classmethod
    def _clean_description(cls, value: str) -> str:
        return value.strip()

    @field_validator("visible_facts", "uncertainties")
    @classmethod
    def _clean_list(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(item.strip() for item in value if item.strip()))

    def as_text(self) -> str:
        """转换成适合交给最终答案模型的紧凑文本。"""

        sections: list[str] = []
        if self.description:
            sections.append(self.description)
        if self.visible_facts:
            sections.append("可见事实：" + "；".join(self.visible_facts))
        if self.uncertainties:
            sections.append("不确定项：" + "；".join(self.uncertainties))
        return "\n".join(sections) or "截图中没有发现可用的相关信息。"


class OllamaVisionClient:
    """封装 Ollama `/api/chat` 图片请求和结构化结果校验。"""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout: float = 30.0,
        warmup_timeout: float = 10.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model.strip()
        self._timeout = timeout
        self._warmup_timeout = warmup_timeout
        if not self._base_url:
            raise ValueError("OLLAMA_BASE_URL 不能为空")
        if not self._model:
            raise ValueError("OLLAMA_VISION_MODEL 不能为空")
        if timeout <= 0:
            raise ValueError("OLLAMA_VISION_TIMEOUT 必须大于 0")
        if warmup_timeout <= 0:
            raise ValueError("OLLAMA_VISION_WARMUP_TIMEOUT 必须大于 0")

    def warm_up(self) -> None:
        """主动加载视觉模型，并刷新它在 Ollama 中的保活时间。"""

        try:
            response = httpx.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": self._model,
                    "stream": False,
                    "keep_alive": "10m",
                },
                timeout=self._warmup_timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise OllamaVisionError(
                f"Ollama 视觉模型预热失败：{error}"
            ) from error

    def analyze_webpage(
        self,
        *,
        query: str,
        title: str,
        url: str,
        screenshot_base64: str,
    ) -> VisionEvidence:
        """根据用户问题识别一张网页顶部截图中的可见证据。"""

        if not screenshot_base64.strip():
            raise ValueError("网页截图不能为空")

        prompt = (
            "你是网页截图证据提取器，不负责生成最终答案。只识别截图中与用户问题"
            "直接相关、肉眼可见的内容。优先读取动态时间、日期、价格、状态、图表、"
            "表格和 Canvas 内容；忽略导航栏、广告、推荐、装饰图片和无关功能。"
            "不得根据常识补全截图中看不到的内容，文字或数字看不清时必须写入"
            " uncertainties。输出应简短、客观。\n\n"
            f"用户问题：{query}\n"
            f"网页标题：{title}\n"
            f"网页地址：{url}"
        )
        try:
            response = httpx.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                            "images": [screenshot_base64],
                        }
                    ],
                    "format": VisionEvidence.model_json_schema(),
                    "stream": False,
                    "think": False,
                    "keep_alive": "10m",
                    "options": {"temperature": 0},
                },
                timeout=self._timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError as error:
            raise OllamaVisionError(f"Ollama 视觉请求失败：{error}") from error
        except ValueError as error:
            raise OllamaVisionError("Ollama 视觉响应不是合法 JSON") from error

        message = payload.get("message")
        if not isinstance(message, dict):
            raise OllamaVisionError("Ollama 视觉响应缺少 message")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise OllamaVisionError("Ollama 视觉响应缺少 message.content")
        try:
            return VisionEvidence.model_validate_json(content)
        except ValueError as error:
            raise OllamaVisionError("视觉模型返回结果不符合约定结构") from error


@lru_cache(maxsize=1)
def get_vision_client() -> OllamaVisionClient:
    """根据环境变量创建并复用 Ollama 视觉客户端。"""

    return OllamaVisionClient(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        model=os.getenv("OLLAMA_VISION_MODEL", "qwen3-vl:4b-instruct"),
        timeout=float(os.getenv("OLLAMA_VISION_TIMEOUT", "30")),
        warmup_timeout=float(
            os.getenv("OLLAMA_VISION_WARMUP_TIMEOUT", "10")
        ),
    )
