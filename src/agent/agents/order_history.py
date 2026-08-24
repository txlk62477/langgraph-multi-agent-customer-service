"""自主识别查询数量并调用订单工具的历史订单 Agent。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.tools import BaseTool

from agent.agents.factory import build_specialist_agent
from agent.common.booking_db import BookingDB, PostgresBookingDB
from agent.common.llm import build_chat_model
from agent.tools.orders import (
    build_get_order_details_tool,
    build_list_recent_orders_tool,
    build_search_orders_tool,
)


ORDER_HISTORY_PROMPT = """你是租房客服的历史订单专业 Agent。

- 只处理当前用户自己的历史订单。
- 用户问最近订单时调用 list_recent_orders；按房源、状态或日期查找时调用 search_orders；
  给出明确订单号或从候选中选定一笔后调用 get_order_details。
- 自主选择最匹配的工具，不要为了展示能力依次调用全部工具，也绝不能根据聊天历史编造订单。
- 工具返回 empty 时明确说明暂无历史订单；failed 时给出稳定的稍后重试提示。
- 成功时展示订单号、房源、入住、退房、月租和状态，不展示手机号、user_id 或内部数据。
- 最终只给出一条中文回复。
"""


def build_order_history_agent(
    *,
    model_factory: Callable[[], Any] = build_chat_model,
    booking_db_factory: Callable[[], BookingDB] = PostgresBookingDB,
    tools: Sequence[BaseTool] | None = None,
    checkpointer: Any = None,
    name: str = "order_history_agent",
):
    resolved_tools = list(tools) if tools is not None else [
        build_list_recent_orders_tool(booking_db_factory=booking_db_factory),
        build_search_orders_tool(booking_db_factory=booking_db_factory),
        build_get_order_details_tool(booking_db_factory=booking_db_factory),
    ]
    return build_specialist_agent(
        name=name,
        system_prompt=ORDER_HISTORY_PROMPT,
        tools=resolved_tools,
        model_factory=model_factory,
        checkpointer=checkpointer,
    )


order_history_agent = build_order_history_agent()
