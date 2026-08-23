"""自主收集条件、读取偏好并查询房源的推荐 Agent。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.tools import BaseTool

from agent.agents.factory import build_specialist_agent
from agent.common.llm import build_chat_model
from agent.tools.conversation import build_request_user_input_tool
from agent.tools.rental import (
    build_get_rental_preferences_tool,
    build_search_houses_tool,
)


RENTAL_RECOMMENDATION_PROMPT = """你是租房客服的房源推荐专业 Agent。

目标是取得可靠条件后查询真实房源，不得编造推荐。
1. 先理解当前轮用户明确给出的条件，再调用 get_rental_preferences 读取长期偏好。
2. 当前轮条件永远优先；长期偏好只能补缺失字段。用户明确换城市时不得沿用旧城市区域。
3. 查询所需必要字段是 city、budget_min、budget_max。缺失时必须调用
   request_user_input 暂停并询问，不要只生成普通问题文本。
4. 如果三个必要字段全部来自长期偏好，查询前必须调用 request_user_input 请用户确认；
   用户修改时采用新值。
5. 条件齐全后调用 search_houses。城市和预算不得自动放宽；无结果时可以建议用户主动
   调整条件，但不得再次用未获确认的宽松条件查询。
6. 只依据工具结果展示标题、月租、城市、区域、小区、房型、租赁方式和面积，不展示
   原始 SQL、房源 ID、user_id 或经纬度。
7. 每轮完成一次有效查询后就生成最终中文回复，不重复调用相同工具。
"""


def build_rental_recommendation_agent(
    *,
    model_factory: Callable[[], Any] = build_chat_model,
    tools: Sequence[BaseTool] | None = None,
    checkpointer: Any = None,
    name: str = "rental_recommendation_agent",
):
    resolved_tools = list(tools) if tools is not None else [
        build_get_rental_preferences_tool(),
        build_request_user_input_tool(),
        build_search_houses_tool(),
    ]
    return build_specialist_agent(
        name=name,
        system_prompt=RENTAL_RECOMMENDATION_PROMPT,
        tools=resolved_tools,
        model_factory=model_factory,
        checkpointer=checkpointer,
    )


rental_recommendation_agent = build_rental_recommendation_agent()
