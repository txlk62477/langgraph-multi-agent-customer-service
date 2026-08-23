"""供各个 Agent 图模块共用的 LLM 构造函数。"""

import os

from langchain_deepseek import ChatDeepSeek


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    """读取并校验整数配置，避免错误环境变量导致模型节点启动失败。"""

    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= minimum else default


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    """读取并校验浮点配置。"""

    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value >= minimum else default


def build_chat_model() -> ChatDeepSeek:
    """按环境变量创建 DeepSeek 对话模型。"""

    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is missing. Copy .env.example to .env and set it."
        )

    # 构建 Agent 时创建客户端；构造过程不发起网络请求，后续调用复用该实例。
    return ChatDeepSeek(
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        temperature=0,
        max_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", "2048")),
        # 所有节点统一使用有限请求超时和少量重试，避免某个 LLM 调用
        # 长时间占住图执行线程。结构化输出模型会继承这两个配置。
        timeout=_env_float("DEEPSEEK_REQUEST_TIMEOUT", 60.0, minimum=0.1),
        max_retries=_env_int("DEEPSEEK_MAX_RETRIES", 1, minimum=0),
    )
