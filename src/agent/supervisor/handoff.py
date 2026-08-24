"""Supervisor 显式移交专业 Agent 的 handoff 工具。"""

from __future__ import annotations

from typing import Annotated, Any, Mapping

from langchain.tools import InjectedToolCallId, ToolRuntime, tool
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command

from agent.state.customer_service import CustomerServiceState, SpecialistName


MAX_DELEGATIONS = 3

SPECIALIST_DESCRIPTIONS: dict[SpecialistName, tuple[str, str]] = {
    "general_qa_agent": (
        "delegate_to_general_qa",
        "委派常规知识、天气、新闻、人物、政策或需要联网研究的问题。",
    ),
    "rental_recommendation_agent": (
        "delegate_to_rental_recommendation",
        "委派房源查找、筛选、市场探索和租房推荐任务。",
    ),
    "rental_booking_agent": (
        "delegate_to_rental_booking",
        "委派新建租房预订、检查房源档期和收集预订信息任务。",
    ),
    "order_history_agent": (
        "delegate_to_order_history",
        "委派历史订单、订单状态、订单筛选和订单详情查询。",
    ),
    "order_cancellation_agent": (
        "delegate_to_order_cancellation",
        "委派查找可取消订单、检查取消资格和执行订单取消。",
    ),
}

HANDOFF_TOOL_NAMES = {
    tool_name for tool_name, _ in SPECIALIST_DESCRIPTIONS.values()
}


def build_handoff_tools() -> list[BaseTool]:
    """为每个专业 Agent 创建一个返回父图 Command 的委派工具。"""

    return [
        _build_handoff_tool(specialist, tool_name, description)
        for specialist, (tool_name, description) in SPECIALIST_DESCRIPTIONS.items()
    ]


def _build_handoff_tool(
    specialist: SpecialistName,
    tool_name: str,
    description: str,
) -> BaseTool:
    @tool(tool_name, description=description)
    def handoff(
        task: str,
        runtime: ToolRuntime[CustomerServiceState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command | str:
        """把明确任务和共享消息交给一个专业 Agent。"""

        state = runtime.state if isinstance(runtime.state, Mapping) else {}
        messages = list(state.get("messages", []))
        current_calls = _current_handoff_calls(messages)
        if len(current_calls) != 1:
            return "一次只能委派一个专业 Agent，请重新选择最合适的单个 Agent。"

        delegations = list(state.get("delegations", []))
        count = len(delegations)
        delegated = [record["agent"] for record in delegations]
        if count >= MAX_DELEGATIONS:
            return "本轮最多委派3次。请根据已有专业结果直接回答用户。"
        if specialist in delegated:
            return "本轮已经委派过该专业 Agent。请使用已有结果或直接向用户澄清。"

        clean_task = task.strip()
        if not clean_task:
            return "委派任务不能为空，请说明专业 Agent 需要完成的具体目标。"

        return Command(
            graph=Command.PARENT,
            goto=specialist,
            update={
                "messages": [
                    messages[-1],
                    ToolMessage(
                        content=(
                            f"Supervisor 已将任务移交给 {specialist}。\n"
                            f"明确任务：{clean_task}\n"
                            "请结合此前共享对话完成任务；完成后把结论交回 Supervisor。"
                        ),
                        name=tool_name,
                        tool_call_id=tool_call_id,
                    )
                ],
                "delegations": [
                    *delegations,
                    {
                        "agent": specialist,
                        "task": clean_task,
                        "tool_call_id": tool_call_id,
                    },
                ],
            },
        )

    return handoff


def _current_handoff_calls(messages: list[Any]) -> list[dict[str, Any]]:
    """读取当前模型消息中的 handoff 调用，拒绝并行委派。"""

    if not messages or not isinstance(messages[-1], AIMessage):
        return []
    return [
        call
        for call in messages[-1].tool_calls
        if call.get("name") in HANDOFF_TOOL_NAMES
    ]
