"""自主收集信息并调用安全写工具的租房预订 Agent。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.tools import BaseTool

from agent.agents.factory import build_specialist_agent
from agent.agents.user_input_guard import build_user_input_guard
from agent.common.booking_db import BookingDB, PostgresBookingDB
from agent.common.llm import build_chat_model
from agent.common.rental_catalog import PostgresRentalCatalog, RentalCatalog
from agent.tools.conversation import build_request_user_input_tool
from agent.tools.rental import (
    build_check_booking_availability_tool,
    build_create_booking_tool,
    build_find_bookable_houses_tool,
)


RENTAL_BOOKING_PROMPT = """你是租房客服的预订专业 Agent。

- 创建订单需要手机号、明确 house_id、入住日期和退房日期。
- 缺少任何字段时必须调用 request_user_input 暂停并一次性列出缺失字段。
- 日期有相对表达时结合当前日期理解，但调用工具必须使用 YYYY-MM-DD。
- 用户只给出房源名称或小区时调用 find_bookable_houses；候选不唯一时调用
  request_user_input 展示候选并让用户选择。
- 房源与日期明确后调用 check_booking_availability。可用时才能调用 create_booking；最终
  工具仍会重新校验手机号、日期、身份、房源和日期冲突，不要绕过工具。
- success 时展示订单号、房源、入住、退房和月租；rejected/invalid/failed 时依据工具结果
  给出稳定说明。绝不能声称未成功的订单已经创建。
- 每轮最多执行一次成功写入，最终只给出一条中文回复。
"""


def build_rental_booking_agent(
    *,
    model_factory: Callable[[], Any] = build_chat_model,
    booking_db_factory: Callable[[], BookingDB] = PostgresBookingDB,
    catalog_factory: Callable[[], RentalCatalog] = PostgresRentalCatalog,
    tools: Sequence[BaseTool] | None = None,
    checkpointer: Any = None,
    name: str = "rental_booking_agent",
):
    resolved_tools = list(tools) if tools is not None else [
        build_find_bookable_houses_tool(catalog_factory=catalog_factory),
        build_check_booking_availability_tool(catalog_factory=catalog_factory),
        build_request_user_input_tool(),
        build_create_booking_tool(booking_db_factory=booking_db_factory),
    ]
    return build_specialist_agent(
        name=name,
        system_prompt=RENTAL_BOOKING_PROMPT,
        tools=resolved_tools,
        model_factory=model_factory,
        checkpointer=checkpointer,
        middleware_builders=[
            build_user_input_guard(
                "定位明确房源，补齐手机号、入住日期和退房日期，并安全创建预订"
            )
        ],
    )


rental_booking_agent = build_rental_booking_agent()
