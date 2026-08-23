"""自主收集信息并调用安全写工具的租房预订 Agent。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.tools import BaseTool

from agent.agents.factory import build_specialist_agent
from agent.common.booking_db import BookingDB, PostgresBookingDB
from agent.common.llm import build_chat_model
from agent.tools.conversation import build_request_user_input_tool
from agent.tools.rental import build_create_booking_tool


RENTAL_BOOKING_PROMPT = """你是租房客服的预订专业 Agent。

- 创建订单需要手机号、准确房源名称、入住日期和退房日期。
- 缺少任何字段时必须调用 request_user_input 暂停并一次性列出缺失字段。
- 日期有相对表达时结合当前日期理解，但调用工具必须使用 YYYY-MM-DD。
- 信息齐全后调用 create_booking；手机号、日期、用户身份、房源匹配、日期冲突和事务安全
  都由工具最终校验，不要绕过工具。
- multiple_candidates 时必须调用 request_user_input 展示候选并让用户选择，然后使用选中
  的准确房源名称重新调用 create_booking。
- success 时展示订单号、房源、入住、退房和月租；rejected/invalid/failed 时依据工具结果
  给出稳定说明。绝不能声称未成功的订单已经创建。
- 每轮最多执行一次成功写入，最终只给出一条中文回复。
"""


def build_rental_booking_agent(
    *,
    model_factory: Callable[[], Any] = build_chat_model,
    booking_db_factory: Callable[[], BookingDB] = PostgresBookingDB,
    tools: Sequence[BaseTool] | None = None,
    checkpointer: Any = None,
    name: str = "rental_booking_agent",
):
    resolved_tools = list(tools) if tools is not None else [
        build_request_user_input_tool(),
        build_create_booking_tool(booking_db_factory=booking_db_factory),
    ]
    return build_specialist_agent(
        name=name,
        system_prompt=RENTAL_BOOKING_PROMPT,
        tools=resolved_tools,
        model_factory=model_factory,
        checkpointer=checkpointer,
    )


rental_booking_agent = build_rental_booking_agent()
