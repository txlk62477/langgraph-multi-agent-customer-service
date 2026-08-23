"""主图的轻量上下文、意图识别和业务占位节点。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent.common.llm import build_chat_model
from agent.state.customer_service import CustomerIntent, CustomerServiceState


ModelFactory = Callable[[], Any]
ROUTING_MESSAGE_COUNT = 5


class CustomerIntentDecision(BaseModel):
    """主图意图识别模型必须返回的结构化结果。"""

    model_config = ConfigDict(extra="forbid")

    intent: CustomerIntent = Field(description="用户当前请求所属的业务类型")
    reason: str = Field(description="简短说明分类依据")

    @field_validator("reason")
    @classmethod
    def _strip_reason(cls, value: str) -> str:
        return value.strip()


class CustomerServiceNodes:
    """封装主图节点，并允许测试时替换意图识别模型。"""

    def __init__(self, *, model_factory: ModelFactory = build_chat_model) -> None:
        self._model_factory = model_factory

    def prepare_routing_context(
        self,
        state: CustomerServiceState,
    ) -> dict[str, Any]:
        """记录业务流程起点，并为主图路由保留最近五条对话消息。"""

        conversation = [
            message
            for message in state.get("messages", [])
            if isinstance(message, (HumanMessage, AIMessage))
            and isinstance(message.content, str)
            and message.content.strip()
        ]
        latest_human = next(
            (
                message
                for message in reversed(conversation)
                if isinstance(message, HumanMessage)
            ),
            None,
        )
        return {
            "routing_messages": conversation[-ROUTING_MESSAGE_COUNT:],
            "current_turn_start_message_id": (
                latest_human.id if latest_human is not None else None
            ),
        }

    def identify_intent(
        self,
        state: CustomerServiceState,
    ) -> dict[str, Any]:
        """使用最近对话识别业务意图；模型失败时降级到常规问答。"""

        messages = state.get("routing_messages") or _recent_messages(state)
        try:
            model = self._model_factory().with_structured_output(
                CustomerIntentDecision,
                method="function_calling",
            )
            decision = model.invoke(
                [
                    SystemMessage(
                        content=(
                            "你是租房智能客服的主图意图路由器。只负责选择业务，不回答"
                            "问题。可选意图：general_qa=常规知识、天气、新闻、人物、"
                            "政策或联网查询；recommend_rental=查找、筛选或推荐房源；"
                            "reserve_rental=创建新的租房预订；cancel_order=取消已有"
                            "租房订单；order_history=查询历史订单或订单状态。结合最近"
                            "对话理解省略表达，但不要编造用户"
                            "没有表达的业务目标。无法确定时选择general_qa。"
                        )
                    ),
                    HumanMessage(
                        content="最近对话：\n" + _format_messages(messages)
                    ),
                ]
            )
            if not isinstance(decision, CustomerIntentDecision):
                decision = CustomerIntentDecision.model_validate(decision)
        except Exception as error:
            return {
                "customer_intent": "general_qa",
                "intent_reason": "意图模型不可用，降级到常规问答",
                "intent_error": f"{type(error).__name__}: {error}",
            }

        return {
            "customer_intent": decision.intent,
            "intent_reason": decision.reason,
            "intent_error": "",
        }


def empty_business_placeholder(state: CustomerServiceState) -> dict[str, Any]:
    """尚未实现的业务占位节点；不读取也不修改主图状态。"""

    return {}


def _recent_messages(state: CustomerServiceState) -> list[BaseMessage]:
    """缺少预处理结果时安全回退到最近五条有效对话消息。"""

    conversation = [
        message
        for message in state.get("messages", [])
        if isinstance(message, (HumanMessage, AIMessage))
        and isinstance(message.content, str)
        and message.content.strip()
    ]
    return conversation[-ROUTING_MESSAGE_COUNT:]


def _format_messages(messages: Sequence[BaseMessage]) -> str:
    """把路由消息格式化为模型容易区分角色的文本。"""

    if not messages:
        return "（没有有效对话）"
    lines: list[str] = []
    for message in messages:
        role = "用户" if isinstance(message, HumanMessage) else "助手"
        lines.append(f"{role}：{message.content}")
    return "\n".join(lines)
