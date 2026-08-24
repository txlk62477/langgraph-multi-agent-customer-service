"""需要用户补充或确认时使用的通用中断工具。"""

from __future__ import annotations

from typing import Any

from langchain.tools import ToolRuntime, tool
from langgraph.types import interrupt

from agent.tools.runtime import SpecialistContext, json_result


def build_request_user_input_tool():
    """创建一个通过 LangGraph interrupt 暂停并等待用户回复的工具。"""

    @tool("request_user_input")
    def request_user_input(
        question: str,
        reason: str,
        missing_fields: list[str] | None = None,
        runtime: ToolRuntime[SpecialistContext] = None,
    ) -> str:
        """缺少必要信息或需要非写操作确认时暂停任务并向用户提问。"""

        del runtime
        payload: dict[str, Any] = {
            "type": "agent_request_user_input",
            "message": question.strip(),
            "reason": reason.strip(),
        }
        if missing_fields:
            payload["missing_required_fields"] = list(dict.fromkeys(missing_fields))
        answer = interrupt(payload)
        return json_result(status="resumed", user_answer=answer)

    return request_user_input
