"""自主筛选订单并通过确认工具执行软取消的 Agent。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.tools import BaseTool

from agent.agents.factory import build_specialist_agent
from agent.agents.user_input_guard import build_user_input_guard
from agent.common.booking_db import BookingDB, PostgresBookingDB
from agent.common.llm import build_chat_model
from agent.tools.conversation import build_request_user_input_tool
from agent.tools.orders import (
    build_cancel_order_tool,
    build_check_cancellation_eligibility_tool,
    build_find_cancellable_orders_tool,
)


ORDER_CANCELLATION_PROMPT = """你是租房客服的订单取消专业 Agent。

- 先从最近对话提取订单号、房源名称或入住日期范围。没有唯一订单时调用
  find_cancellable_orders；已有明确订单号时调用 check_cancellation_eligibility 获取取消预览。
- 查询工具已经按当前 user_id、confirmed 状态和未来入住日期隔离，不得要求或修改 user_id。
- 没有候选时明确说明；多个候选时调用 request_user_input 展示候选并让用户选择。
- 多个候选时调用 request_user_input 让用户选择。确定唯一且可取消的订单后调用
  cancel_order。该工具内部还会强制执行一次 interrupt 确认；不得用
  普通文本确认代替，也不得声称跳过确认。
- 根据工具最终状态回复：success、cancelled_by_user、already_cancelled、already_started、
  not_cancellable、not_found 或 failed。绝不能声称未成功的取消已经完成。
- 最终只给出一条中文回复，不展示数据库错误堆栈或 user_id。
"""


def build_order_cancellation_agent(
    *,
    model_factory: Callable[[], Any] = build_chat_model,
    booking_db_factory: Callable[[], BookingDB] = PostgresBookingDB,
    tools: Sequence[BaseTool] | None = None,
    checkpointer: Any = None,
    name: str = "order_cancellation_agent",
):
    resolved_tools = list(tools) if tools is not None else [
        build_find_cancellable_orders_tool(booking_db_factory=booking_db_factory),
        build_check_cancellation_eligibility_tool(
            booking_db_factory=booking_db_factory
        ),
        build_request_user_input_tool(),
        build_cancel_order_tool(booking_db_factory=booking_db_factory),
    ]
    return build_specialist_agent(
        name=name,
        specialist_name="order_cancellation_agent",
        system_prompt=ORDER_CANCELLATION_PROMPT,
        tools=resolved_tools,
        model_factory=model_factory,
        checkpointer=checkpointer,
        middleware_builders=[
            build_user_input_guard(
                "定位用户要取消的订单、检查取消资格，并在确认后安全取消"
            )
        ],
    )


order_cancellation_agent = build_order_cancellation_agent()
