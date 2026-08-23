"""自主识别查询数量并调用订单工具的历史订单 Agent。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.tools import BaseTool

from agent.agents.factory import build_specialist_agent
from agent.common.booking_db import BookingDB, PostgresBookingDB
from agent.common.llm import build_chat_model
from agent.tools.orders import build_list_recent_orders_tool


ORDER_HISTORY_PROMPT = """你是租房客服的历史订单专业 Agent。

- 只处理当前用户自己的历史订单。
- 从最近对话识别用户希望查看的数量；未说明时使用1，最大20。
- 必须调用 list_recent_orders，绝不能根据聊天历史编造订单。
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
        build_list_recent_orders_tool(booking_db_factory=booking_db_factory)
    ]
    return build_specialist_agent(
        name=name,
        system_prompt=ORDER_HISTORY_PROMPT,
        tools=resolved_tools,
        model_factory=model_factory,
        checkpointer=checkpointer,
    )


order_history_agent = build_order_history_agent()
