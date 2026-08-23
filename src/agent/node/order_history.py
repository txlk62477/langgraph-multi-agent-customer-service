"""历史订单子图内部节点：数量识别、参数化查询与固定格式结果。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agent.common.booking_db import BookingDB, format_price
from agent.common.llm import build_chat_model


DEFAULT_ORDER_LIMIT = 1
MAX_ORDER_LIMIT = 10

STATUS_LABELS = {
    "confirmed": "已确认",
    "pending": "待确认",
    "cancelled": "已取消",
}


class OrderLimit(BaseModel):
    """让 DeepSeek 从最近对话中识别用户想要查看的订单数量。"""

    limit: int | None = Field(
        default=None,
        ge=1,
        le=MAX_ORDER_LIMIT,
        description="用户要求返回的最近订单数量（1到10），未明确说明时为null",
    )


class OrderHistoryNodes:
    """封装历史订单的纯逻辑节点；数据通过 BookingDB 注入。"""

    def __init__(
        self,
        *,
        booking_db_factory: Callable[[], BookingDB],
        model_factory: Callable[[], Any] = build_chat_model,
    ) -> None:
        self._booking_db_factory = booking_db_factory
        self._model_factory = model_factory

    def reset(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """进入查询流程时初始化内部状态。"""

        return {
            "order_limit": DEFAULT_ORDER_LIMIT,
            "history_status": "querying",
            "history_error": "",
            "orders": [],
        }

    def extract_order_limit(
        self,
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        """让 LLM 从最近对话中识别订单数量，识别不到时用默认 1。

        只把最近几条对话消息喂给模型，不传完整历史。
        """

        limit = DEFAULT_ORDER_LIMIT
        try:
            structured = self._model_factory().with_structured_output(
                OrderLimit,
                method="function_calling",
            )
            result = structured.invoke(self._build_limit_messages(state))
            if isinstance(result, BaseModel) and result.limit is not None:
                limit = max(1, min(int(result.limit), MAX_ORDER_LIMIT))
        except Exception:
            # 模型调用失败时使用默认数量，不阻断查询。
            pass
        return {"order_limit": limit}

    def query_orders(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """从继承的 state.user_id 读取用户，参数化查询最近订单。"""

        user_id = (state.get("user_id") or "").strip()
        if not user_id:
            return {
                "history_status": "failed",
                "history_error": "缺少用户身份（user_id），无法查询订单",
            }
        try:
            orders = self._booking_db_factory().list_recent_orders(
                user_id=user_id,
                limit=int(state.get("order_limit", DEFAULT_ORDER_LIMIT)),
            )
        except Exception as error:
            return {
                "history_status": "failed",
                "history_error": f"查询订单失败：{type(error).__name__}",
            }
        if not orders:
            return {"history_status": "empty", "orders": []}
        return {
            "history_status": "success",
            "orders": [
                {
                    "order_no": order.order_no,
                    "house_title": order.house_title,
                    "check_in_date": order.check_in_date,
                    "check_out_date": order.check_out_date,
                    "status": order.status,
                    "price": order.price,
                }
                for order in orders
            ],
            "history_error": "",
        }

    def format_result(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """按固定格式返回订单列表、空结果或失败原因。"""

        status = state.get("history_status")
        if status == "success":
            orders = state.get("orders") or []
            lines = [
                (
                    f"{index}. 订单号：{order['order_no']}，"
                    f"房源：{order['house_title']}，"
                    f"入住：{order['check_in_date']}，退房：{order['check_out_date']}，"
                    f"月租：{format_price(order.get('price'))}元，"
                    f"状态：{STATUS_LABELS.get(order.get('status', ''), order.get('status', ''))}"
                )
                for index, order in enumerate(orders, start=1)
            ]
            content = f"您最近的 {len(orders)} 笔订单：\n" + "\n".join(lines)
        elif status == "empty":
            content = "您还没有历史订单。"
        else:
            content = "查询失败：" + (state.get("history_error") or "未知原因")
        return {"messages": [AIMessage(content=content)]}

    def _build_limit_messages(
        self,
        state: Mapping[str, Any],
    ) -> list[SystemMessage | Any]:
        """构造数量识别消息：系统提示 + 最近 6 条对话。"""

        messages = list(state.get("messages") or [])
        recent = messages[-6:]
        prompt = (
            "你是订单数量识别器。用户想查询自己的历史订单，"
            "请从最近对话中识别用户想要查看的最近订单数量。\n"
            "- 用户说“最近的三个订单”“最近3单”“最近三笔”等 → 返回对应数量\n"
            "- 用户没说数量（如“查一下我的订单”）→ limit 返回 null\n"
            f"- 数量范围 1 到 {MAX_ORDER_LIMIT}"
        )
        return [SystemMessage(content=prompt), *recent]
