"""自主收集条件、读取偏好并查询房源的推荐 Agent。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.tools import BaseTool

from agent.agents.factory import build_specialist_agent
from agent.agents.user_input_guard import build_user_input_guard
from agent.common.llm import build_chat_model
from agent.common.rental_catalog import PostgresRentalCatalog, RentalCatalog
from agent.tools.conversation import build_request_user_input_tool
from agent.tools.rental import (
    build_get_house_details_tool,
    build_get_rental_preferences_tool,
    build_inspect_rental_market_tool,
    build_search_houses_tool,
)


RENTAL_RECOMMENDATION_PROMPT = """你是租房客服的房源推荐专业 Agent。

目标是取得可靠条件后查询真实房源，不得编造推荐。
1. 先理解当前轮用户明确给出的条件，再自主决定是否调用 get_rental_preferences 补全长期偏好。
2. 当前轮条件永远优先；长期偏好只能补缺失字段。用户明确换城市时不得沿用旧城市区域。
3. 查询所需必要字段是 city、budget_min、budget_max。缺失时必须调用
   request_user_input 暂停并询问，不要只生成普通问题文本。
4. 如果三个必要字段全部来自长期偏好，查询前必须调用 request_user_input 请用户确认；
   用户修改时采用新值。
5. 条件模糊、用户想了解市场或搜索无结果时，可以调用 inspect_rental_market 查看真实
   城市、区域和价格范围；不得把市场统计当成具体房源。
6. 条件齐全后调用 search_houses。城市和预算不得自动放宽；无结果时只能在用户确认后
   修改条件。需要介绍某套候选的详细地址、楼层、设施或描述时再调用 get_house_details。
7. 只依据工具结果展示房源，不展示原始 SQL、内部 user_id 或经纬度。house_id 只用于
   后续工具定位，不主动展示给用户。
8. 已有充分结果时停止，不重复调用相同工具。
"""


def build_rental_recommendation_agent(
    *,
    model_factory: Callable[[], Any] = build_chat_model,
    tools: Sequence[BaseTool] | None = None,
    catalog_factory: Callable[[], RentalCatalog] = PostgresRentalCatalog,
    checkpointer: Any = None,
    name: str = "rental_recommendation_agent",
):
    resolved_tools = list(tools) if tools is not None else [
        build_get_rental_preferences_tool(),
        build_inspect_rental_market_tool(catalog_factory=catalog_factory),
        build_request_user_input_tool(),
        build_search_houses_tool(catalog_factory=catalog_factory),
        build_get_house_details_tool(catalog_factory=catalog_factory),
    ]
    return build_specialist_agent(
        name=name,
        system_prompt=RENTAL_RECOMMENDATION_PROMPT,
        tools=resolved_tools,
        model_factory=model_factory,
        checkpointer=checkpointer,
        middleware_builders=[
            build_user_input_guard(
                "收集城市、区域、预算、房型和租赁方式，并查询真实房源进行推荐"
            )
        ],
    )


rental_recommendation_agent = build_rental_recommendation_agent()
