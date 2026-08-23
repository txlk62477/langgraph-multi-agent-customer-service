"""常规问答子图的搜索决策、直接回答和失败降级节点。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
import re
from typing import Any, Literal

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent.common.llm import build_chat_model
from agent.state.general_qa import GeneralQAState


ModelFactory = Callable[[], Any]


class SearchDecision(BaseModel):
    """DeepSeek 对模糊问题给出的结构化路由结果。"""

    model_config = ConfigDict(extra="forbid")

    need_search: bool = Field(description="回答是否必须调用联网搜索")
    search_query: str = Field(
        default="",
        description="补全上下文后的独立搜索词；不联网时返回空字符串",
    )
    reason: str = Field(description="简短说明作出该判断的原因")
    requires_fresh_data: bool = Field(
        description="问题是否依赖天气、价格、新闻等会变化的实时信息"
    )

    @field_validator("search_query", "reason")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        return value.strip()


# 用户明确要求查询互联网时，必须尊重该要求。
_EXPLICIT_SEARCH_PATTERN = re.compile(
    r"(联网|上网|网上|搜索|搜一下|搜下|查一下|查下|检索|帮我查|在线查询)"
)

# 这类任务只处理用户提供的文本，不应因为文本里出现“今天”等词而误触发搜索。
_DIRECT_TASK_PATTERN = re.compile(
    r"^\s*(请)?(帮我)?(翻译|改写|润色|扩写|缩写|总结|提取|纠错|续写)"
)

_GREETING_PATTERN = re.compile(
    r"^\s*(你好|您好|嗨|hi|hello|早上好|下午好|晚上好|谢谢|感谢)[！!。.\s]*$",
    re.IGNORECASE,
)

# 这些表述通常只有联网后才可能可靠回答。
_FRESH_DATA_PATTERN = re.compile(
    r"(最新|实时|今天|明天|后天|刚刚|近期新闻|新闻|天气|气温|降雨|"
    r"股价|汇率|比分|赛程|时刻表|票价|油价|金价|房价|当前价格|"
    r"现在几点|几点了|现行政策|最新政策|最新规定|营业时间)"
)


class GeneralQANodes:
    """封装常规问答节点，并允许测试时注入假的模型。"""

    def __init__(self, *, model_factory: ModelFactory = build_chat_model) -> None:
        self._model_factory = model_factory

    def decide_search(self, state: GeneralQAState) -> dict[str, Any]:
        """按固定规则优先、DeepSeek 处理模糊情况的方式决定回答路径。"""

        question = _latest_user_question(state)
        fixed_route = _deterministic_route(question)
        fresh_data_hint = _requires_fresh_data(question)

        # 问候、翻译和改写等明确任务不需要额外调用一次路由模型。
        if fixed_route == "direct":
            return {
                "qa_route": "direct",
                "search_query": "",
                "routing_reason": "固定规则判断该任务不需要联网",
                "routing_error": "",
                "requires_fresh_data": False,
                "search_error": "",
            }

        context_messages = _model_context(state)
        forced_instruction = (
            "固定规则已判定必须联网。need_search 必须返回 true；你只需结合上下文"
            "补全准确的 search_query。"
            if fixed_route == "search"
            else "没有固定路由结果，请根据问题是否需要外部或最新信息作出判断。"
        )

        try:
            decision_model = self._model_factory().with_structured_output(
                SearchDecision,
                method="function_calling",
            )
            decision = decision_model.invoke(
                [
                    SystemMessage(
                        content=(
                            "你是常规问答的联网路由器。需要搜索的情况包括：用户明确"
                            "要求联网，或问题涉及会变化的新闻、天气、时间、价格、政策、"
                            "赛程、营业状态等信息。闲聊、翻译、改写、数学推理以及不依赖"
                            "当前信息的一般知识可以直接回答。遇到“那里、那个、明天、"
                            "后天、多少钱”等省略表达时，结合最近对话补全搜索词，但绝不"
                            "添加用户没有表达的地点、对象或条件。search_query 必须能够"
                            "脱离聊天记录独立用于搜索。"
                        )
                    ),
                    HumanMessage(
                        content=(
                            f"当前时间：{datetime.now().astimezone().isoformat(timespec='minutes')}\n"
                            f"路由约束：{forced_instruction}\n\n"
                            f"经过裁剪的对话上下文：\n{_format_messages(context_messages)}\n\n"
                            f"当前问题：{question}"
                        )
                    ),
                ]
            )
            if not isinstance(decision, SearchDecision):
                decision = SearchDecision.model_validate(decision)
        except Exception as error:
            # 路由模型失败时仍遵守强制联网规则；普通问题则安全退回直接回答。
            need_search = fixed_route == "search"
            query = question if need_search else ""
            return {
                "qa_route": "search" if need_search else "direct",
                "search_query": query,
                "routing_reason": (
                    "路由模型不可用，按固定实时/联网规则继续搜索"
                    if need_search
                    else "路由模型不可用，降级为直接回答"
                ),
                "routing_error": f"{type(error).__name__}: {error}",
                "requires_fresh_data": fresh_data_hint,
                "search_error": "",
            }

        need_search = decision.need_search or fixed_route == "search"
        query = (decision.search_query or question).strip() if need_search else ""
        return {
            "qa_route": "search" if need_search else "direct",
            "search_query": query,
            "routing_reason": decision.reason,
            "routing_error": "",
            "requires_fresh_data": (
                fresh_data_hint or decision.requires_fresh_data
            ),
            "search_error": "",
        }

    def generate_direct_answer(self, state: GeneralQAState) -> dict[str, list]:
        """使用处理后的上下文直接回答，不读取完整历史消息。"""

        try:
            response = self._model_factory().invoke(
                [
                    SystemMessage(
                        content=(
                            "你是智能客服中的常规问答助手。请结合给出的有效上下文，用中文"
                            "直接回答用户当前问题。不要声称已经联网，不要编造来源；如果问题"
                            "实际依赖无法确认的最新信息，应明确说明信息可能需要联网核实。"
                        )
                    ),
                    *_model_context(state),
                ]
            )
        except Exception:
            # 直接回答是非关键增强能力，LLM 超时或服务异常时仍返回可理解的
            # 用户消息，避免图以异常结束。
            response = AIMessage(
                content="抱歉，当前问答服务暂时不可用，请稍后重试。"
            )
        return {"messages": [response]}

    def generate_search_failure_answer(
        self,
        state: GeneralQAState,
    ) -> dict[str, list]:
        """联网失败时，按问题是否强依赖实时数据选择安全降级方式。"""

        if state.get("requires_fresh_data", False):
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "抱歉，当前联网搜索暂时不可用。这个问题依赖实时信息，为避免"
                            "提供过时或不准确的答案，我现在无法可靠确认，请稍后重试。"
                        )
                    )
                ]
            }

        try:
            response = self._model_factory().invoke(
                [
                    SystemMessage(
                        content=(
                            "联网搜索本次失败。请根据一般知识回答用户问题，并在回答开头"
                            "明确说明未能完成联网核实。不要提供假来源，不要把可能变化的"
                            "事实说成已经确认。"
                        )
                    ),
                    *_model_context(state),
                ]
            )
        except Exception:
            response = AIMessage(
                content="抱歉，当前联网搜索和回答服务暂时不可用，请稍后重试。"
            )
        return {"messages": [response]}


def _latest_user_question(state: GeneralQAState) -> str:
    """取得最近一条非空用户消息。"""

    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            content = message.content
            if isinstance(content, str) and content.strip():
                return content.strip()
    raise ValueError("常规问答状态中没有找到用户问题")


def _model_context(state: GeneralQAState) -> list[AnyMessage]:
    """优先使用上下文子图的裁剪结果，缺失时回退到原始消息。"""

    context = state.get("context_messages")
    if context is not None:
        return list(context)
    return list(state.get("messages", []))


def _deterministic_route(question: str) -> Literal["direct", "search"] | None:
    """只处理置信度很高的固定规则，其余问题交给 DeepSeek。"""

    if _EXPLICIT_SEARCH_PATTERN.search(question):
        return "search"
    if _DIRECT_TASK_PATTERN.search(question) or _GREETING_PATTERN.fullmatch(question):
        return "direct"
    if _requires_fresh_data(question):
        return "search"
    return None


def _requires_fresh_data(question: str) -> bool:
    """判断问题是否明显依赖会变化的实时数据。"""

    return bool(_FRESH_DATA_PATTERN.search(question))


def _format_messages(messages: Sequence[BaseMessage]) -> str:
    """把裁剪后的消息变成路由模型容易理解的角色文本。"""

    if not messages:
        return "无"
    lines: list[str] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            role = "用户"
        elif isinstance(message, AIMessage):
            role = "助手"
        elif isinstance(message, SystemMessage):
            role = "上下文说明"
        else:
            role = message.type
        content = message.content
        text = content if isinstance(content, str) else str(content)
        lines.append(f"{role}：{text}")
    return "\n".join(lines)
