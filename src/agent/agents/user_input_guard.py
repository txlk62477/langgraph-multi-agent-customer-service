"""把专业 Agent 遗漏的用户提问转换成可恢复的 interrupt 工具调用。"""

from __future__ import annotations

import json
import re
from typing import Any, Literal, NotRequired
from uuid import uuid4

from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.runtime import Runtime
from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent.tools.runtime import SpecialistContext
from agent.state.customer_service import MissingField, UserInputGuardEvent


GuardResult = Literal["request", "terminal", "ambiguous"]


class UserInputGuardState(AgentState[Any]):
    """Guard 中间件向专业 Agent 状态增加的诊断字段。"""

    last_guard_event: NotRequired[UserInputGuardEvent]


class UserInputDecision(BaseModel):
    """模糊文本是否确实需要等待用户回复。"""

    model_config = ConfigDict(extra="forbid")

    requires_user_input: bool = Field(
        description="没有用户的新回复，当前专业任务是否无法继续"
    )
    reason: str = Field(description="简短说明判断依据")
    missing_fields: list[MissingField] = Field(
        default_factory=list,
        description="需要用户补充、选择或确认的结构化字段",
    )

    @field_validator("reason")
    @classmethod
    def _clean_reason(cls, value: str) -> str:
        return value.strip()

    @field_validator("missing_fields")
    @classmethod
    def _deduplicate_fields(cls, value: list[MissingField]) -> list[MissingField]:
        return list(dict.fromkeys(value))


class UserInputGuardMiddleware(
    AgentMiddleware[UserInputGuardState, SpecialistContext, Any]
):
    """使用硬规则和结构化 LLM，阻止 Agent 用普通文本等待用户。

    专业 Agent 有时会直接输出“请提供手机号”之类的文本并结束本轮执行。
    这种回复不会经过工具节点，因此也不会触发 LangGraph 的 interrupt。
    本中间件在模型调用后检查最后一条回复，并在确实需要用户输入时将其
    改写成 ``request_user_input`` 工具调用，让 Agent 的标准工具循环负责
    调度工具、产生 interrupt，并在恢复后继续原有执行流程。
    """

    state_schema = UserInputGuardState

    def __init__(self, *, classifier_model: Any, agent_role: str) -> None:
        self._agent_role = agent_role.strip()
        self._classifier_error = ""
        try:
            self._classifier = classifier_model.with_structured_output(
                UserInputDecision,
                method="function_calling",
            )
        except Exception as error:
            # 部分测试模型或临时模型不支持结构化输出；硬规则仍可正常工作。
            self._classifier = None
            self._classifier_error = f"{type(error).__name__}: {error}"

    def after_model(
        self,
        state: UserInputGuardState,
        runtime: Runtime[SpecialistContext],
    ) -> dict[str, Any] | None:
        """必要时把最新普通 AI 提问替换为 request_user_input 调用。"""

        del runtime
        messages = state.get("messages", [])
        # after_model 理论上紧跟模型节点，但仍要防御空状态和非 AI 尾消息。
        if not messages or not isinstance(messages[-1], AIMessage):
            return None
        message = messages[-1]
        # 模型已经主动选择工具时交回 Agent 的正常工具循环，避免覆盖其决策。
        if message.tool_calls or not isinstance(message.content, str):
            return None
        question = message.content.strip()
        if not question:
            return None

        # 明确的请求或结束语优先由确定性规则判断；只有模糊文本才调用 LLM，
        # 这样既降低额外模型调用次数，也避免仅凭问号误判普通结束语。
        hard_result = _hard_rule_result(question)
        if hard_result == "terminal":
            return {
                "last_guard_event": _guard_event(
                    message,
                    result="terminal",
                    source="hard_rule",
                    requires_user_input=False,
                    reason="硬规则识别为任务结束回复",
                )
            }

        fields = _extract_missing_fields(question)
        reason = "模型回复明确要求用户补充、选择或确认信息"
        source: Literal["hard_rule", "llm", "fallback"] = "hard_rule"
        error = ""
        if hard_result == "ambiguous":
            decision, error = self._classify(messages[:-1], question)
            if decision is None:
                source = "fallback"
                if not fields:
                    return {
                        "last_guard_event": _guard_event(
                            message,
                            result="pass",
                            source=source,
                            requires_user_input=False,
                            reason="分类器失败且没有识别出明确缺失字段",
                            error=error,
                        )
                    }
                reason = "分类器失败，但已识别出明确缺失字段"
            elif not decision.requires_user_input:
                return {
                    "last_guard_event": _guard_event(
                        message,
                        result="pass",
                        source="llm",
                        requires_user_input=False,
                        missing_fields=decision.missing_fields,
                        reason=decision.reason or "分类器判断无需等待用户回复",
                    )
                }
            else:
                source = "llm"
                fields = decision.missing_fields or fields
                reason = decision.reason or "完成当前任务需要用户回复"

        # selection_reason 是 Guard 对这次工具选择给出的简短依据；
        # missing_fields 则给恢复端提供结构化的待补充字段信息。
        tool_args: dict[str, Any] = {
            "question": question,
            "selection_reason": reason.strip()[:100],
        }
        if fields:
            tool_args["missing_fields"] = fields
        # 保留原 AIMessage 的 id，仅清空普通文本并注入工具调用。消息 reducer
        # 会按相同 id 替换旧消息，避免历史中同时出现“文本提问”和工具提问。
        guarded_message = message.model_copy(
            update={
                "content": "",
                "tool_calls": [
                    {
                        "name": "request_user_input",
                        "args": tool_args,
                        "id": f"guard-{uuid4().hex}",
                        "type": "tool_call",
                    }
                ],
            }
        )
        # 返回消息更新后，Agent 内置的 ToolNode 会调度 request_user_input；
        # interrupt 发生在工具内部，而不是发生在这个 after_model 钩子中。
        return {
            "messages": [guarded_message],
            "last_guard_event": _guard_event(
                message,
                result="request",
                source=source,
                requires_user_input=True,
                missing_fields=fields,
                reason=reason,
                error=error,
            ),
        }

    def _classify(
        self,
        previous_messages: list[BaseMessage],
        response: str,
    ) -> tuple[UserInputDecision | None, str]:
        """只在硬规则不确定时分类，并把失败原因交给 checkpoint 诊断。"""

        if self._classifier is None:
            return None, self._classifier_error or "分类器不支持结构化输出"
        try:
            result = self._classifier.invoke(
                [
                    SystemMessage(content=_CLASSIFIER_PROMPT),
                    HumanMessage(
                        content=(
                            f"当前专业 Agent 职责：{self._agent_role}\n\n"
                            "最近上下文：\n"
                            + _format_messages(previous_messages[-6:])
                            + "\n\n待判断的模型回复：\n"
                            + response
                        )
                    ),
                ]
            )
            if isinstance(result, UserInputDecision):
                return result, ""
            return UserInputDecision.model_validate(result), ""
        except Exception as error:
            return None, f"{type(error).__name__}: {error}"


def _guard_event(
    message: AIMessage,
    *,
    result: Literal["request", "terminal", "pass"],
    source: Literal["hard_rule", "llm", "fallback"],
    requires_user_input: bool,
    reason: str,
    missing_fields: list[MissingField] | None = None,
    error: str = "",
) -> UserInputGuardEvent:
    """构造可直接写入 JSON checkpoint 的最新 Guard 诊断。"""

    return {
        "message_id": message.id,
        "result": result,
        "source": source,
        "requires_user_input": requires_user_input,
        "missing_fields": list(missing_fields or []),
        "reason": reason,
        "error": error,
    }


_CLASSIFIER_PROMPT = """你是专业 Agent 的用户输入等待判定器，不回答业务问题。
判断待判断回复是否要求用户提供新的事实、从候选中选择、确认某项操作，且没有用户回复
任务就无法继续。只有这种情况 requires_user_input 才为 true。普通结果说明、失败说明、
礼貌结束语、“如有需要可以继续帮助”和“还有其他问题吗”都为 false。missing_fields 只能
从给定 Schema 选择；不能确定字段时返回空列表。不要因文本带问号就自动判定为 true。"""

_REQUEST_PATTERNS = (
    re.compile(
        r"请(?:您)?(?:先|再|一并|直接)?(?:选择|提供|补充|确认|告诉|告知|回复|输入)"
    ),
    re.compile(r"请问您?(?:想|希望|要|选择|确认)"),
    re.compile(r"(?:还|仍|另外)?需要您(?:选择|提供|补充|确认|告诉|告知|回复)"),
    re.compile(r"麻烦您(?:选择|提供|补充|确认|告诉|告知|回复)"),
)
_TERMINAL_PATTERNS = (
    re.compile(r"如(?:果)?有需要.*(?:可以|欢迎|随时)"),
    re.compile(r"如需.*(?:可以|欢迎|随时)"),
    re.compile(r"还有其他(?:问题|需要|需求).*[吗？?]"),
    re.compile(r"(?:预订|订单).*(?:成功|已完成|已创建|已取消)"),
)


def _hard_rule_result(text: str) -> GuardResult:
    """明确请求优先于结束信号，其余交给结构化分类器。"""

    if any(pattern.search(text) for pattern in _REQUEST_PATTERNS):
        return "request"
    if any(pattern.search(text) for pattern in _TERMINAL_PATTERNS):
        return "terminal"
    return "ambiguous"


def _extract_missing_fields(text: str) -> list[MissingField]:
    """从原始提问提取稳定字段，保持业务表单使用的固定顺序。"""

    checks: tuple[tuple[MissingField, tuple[str, ...]], ...] = (
        ("house_id", ("哪一套", "选择房源", "具体房源", "哪套房")),
        ("phone", ("手机号", "手机号码", "联系电话", "电话")),
        ("check_in_date", ("入住日期", "入住时间", "何时入住")),
        ("check_out_date", ("退房日期", "退房时间", "离店日期")),
        ("city", ("城市", "哪个城市", "哪里租房")),
        ("districts", ("区域", "地区", "商圈")),
        ("budget_min", ("最低预算", "预算下限")),
        ("budget_max", ("最高预算", "预算上限", "预算", "租金范围")),
        ("room_types", ("户型", "房型")),
        ("rental_mode", ("整租", "合租", "租赁方式")),
        ("order_no", ("订单号", "哪笔订单", "哪个订单")),
    )
    return [field for field, terms in checks if any(term in text for term in terms)]


def _format_messages(messages: list[BaseMessage]) -> str:
    """为分类器压缩最近上下文，避免传递无限工具结果。"""

    if not messages:
        return "（无）"
    lines: list[str] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            role = "用户"
        elif isinstance(message, AIMessage):
            role = "专业Agent"
        elif isinstance(message, ToolMessage):
            role = f"工具({message.name or 'unknown'})"
        else:
            role = type(message).__name__
        content = message.content
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, default=str)
        lines.append(f"{role}：{content[:1500]}")
    return "\n".join(lines)


def build_user_input_guard(agent_role: str):
    """返回使用 Agent 主模型构造 guard 的 middleware builder。"""

    def builder(model: Any) -> AgentMiddleware:
        return UserInputGuardMiddleware(
            classifier_model=model,
            agent_role=agent_role,
        )

    return builder
