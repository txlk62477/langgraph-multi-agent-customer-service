"""取消订单子图节点：条件抽取、候选确认、软取消和统一回复。"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt
from pydantic import BaseModel, Field, field_validator, model_validator

from agent.common.booking_db import BookingDB, format_price
from agent.common.llm import build_chat_model


MAX_CANCELLATION_CANDIDATES = 5


class CancellationInformation(BaseModel):
    """从取消请求中抽取的可选订单筛选条件。"""

    order_no: str | None = Field(default=None, description="订单唯一编号")
    house_title: str | None = Field(default=None, description="订单中的房源名称")
    check_in_date_start: str | None = Field(
        default=None,
        description="入住时间范围起点，格式YYYY-MM-DD",
    )
    check_in_date_end: str | None = Field(
        default=None,
        description="入住时间范围终点，格式YYYY-MM-DD",
    )

    @field_validator(
        "order_no",
        "house_title",
        "check_in_date_start",
        "check_in_date_end",
    )
    @classmethod
    def _strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @field_validator("check_in_date_start", "check_in_date_end")
    @classmethod
    def _validate_iso_date(cls, value: str | None) -> str | None:
        if value is not None:
            date.fromisoformat(value)
        return value

    @model_validator(mode="after")
    def _validate_date_range(self) -> "CancellationInformation":
        if bool(self.check_in_date_start) != bool(self.check_in_date_end):
            raise ValueError("时间范围必须同时包含开始和结束日期")
        if (
            self.check_in_date_start
            and self.check_in_date_end
            and self.check_in_date_start > self.check_in_date_end
        ):
            raise ValueError("时间范围起点不能晚于终点")
        return self


class OrderSelection(BaseModel):
    """LLM 对用户候选订单回复的结构化解析结果。"""

    selection: int | None = Field(
        default=None,
        ge=1,
        description="用户选择的候选订单序号（从1开始），无法确定时为null",
    )


class OrderCancellationNodes:
    """取消子图的业务节点；只读查询由独立 database_query 子图完成。"""

    def __init__(
        self,
        *,
        booking_db_factory: Callable[[], BookingDB],
        model_factory: Callable[[], Any] = build_chat_model,
    ) -> None:
        self._booking_db_factory = booking_db_factory
        self._model_factory = model_factory

    def initialize(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """初始化本轮取消状态，不把上一轮订单线索带入新请求。"""

        return {
            "order_no": None,
            "house_title": None,
            "check_in_date_start": None,
            "check_in_date_end": None,
            "query_request": "",
            "table_name": "booking_order",
            "max_rows": MAX_CANCELLATION_CANDIDATES,
            "query_status": "pending",
            "query_result": "",
            "query_error": "",
            "order_candidates": [],
            "selected_order": {},
            "cancellation_status": "collecting",
            "cancellation_error": "",
        }

    def extract_order_filters(
        self,
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        """一次性抽取可选订单线索；全部缺省时查询最近订单。"""

        user_id = str(state.get("user_id") or "").strip()
        if not user_id:
            return {
                "cancellation_status": "input_invalid",
                "cancellation_error": "缺少用户身份（user_id），无法取消订单",
            }

        try:
            extractor = self._model_factory().with_structured_output(
                CancellationInformation,
                method="function_calling",
            )
            result = extractor.invoke(
                [
                    SystemMessage(
                        content=(
                            "你是取消订单条件抽取器。只抽取用户明确表达的订单号、"
                            "房源名称和入住时间范围，不回答问题。今天是"
                            f"{date.today().isoformat()}。月份没有年份时，按最近一个"
                            "尚未过去的该月份解析；月份转换为完整自然月的起止日期。"
                            "单个日期的起止日期相同。用户没有提供任何线索时所有字段"
                            "返回null，不要虚构。"
                        )
                    ),
                    *self._recent_conversation(state.get("messages", [])),
                ]
            )
            if not isinstance(result, CancellationInformation):
                result = CancellationInformation.model_validate(result)
            return {
                **result.model_dump(),
                "cancellation_status": "collecting",
                "cancellation_error": "",
            }
        except Exception as error:
            # 条件抽取不可用时仍可安全查询当前用户最近的可取消订单。
            return {
                "cancellation_status": "collecting",
                "cancellation_error": (
                    f"订单条件抽取失败：{type(error).__name__}: {error}"
                ),
            }

    def prepare_order_query(
        self,
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        """把当前用户和可选线索转换成通用查询子图的自然语言请求。"""

        if state.get("cancellation_status") == "input_invalid":
            return {}

        today = date.today().isoformat()
        filters = [
            f"user_id 必须完全等于 {json.dumps(str(state['user_id']), ensure_ascii=False)}",
            "status 必须完全等于 confirmed",
            f"check_in_date 必须晚于 {today}",
        ]
        if state.get("order_no"):
            filters.append(
                "order_no 必须完全等于 "
                + json.dumps(str(state["order_no"]), ensure_ascii=False)
            )
        if state.get("house_title"):
            filters.append(
                "house_title 必须包含 "
                + json.dumps(str(state["house_title"]), ensure_ascii=False)
            )
        start = state.get("check_in_date_start")
        end = state.get("check_in_date_end")
        if start and end:
            # 用户说“9月订单”时，入住区间与该月份相交即可命中。
            filters.extend(
                [
                    f"check_in_date 必须小于等于 {end}",
                    f"check_out_date 必须大于 {start}",
                ]
            )

        return {
            "query_request": (
                "查询 booking_order 表中可取消的订单。筛选条件："
                + "；".join(filters)
                + "。只返回以下列并严格保持顺序：order_no、user_id、"
                "house_id、house_title、check_in_date、check_out_date、"
                "status、price。"
                "按 created_at 降序，最多返回5条。"
            ),
            "table_name": "booking_order",
            "max_rows": MAX_CANCELLATION_CANDIDATES,
            "query_status": "pending",
            "query_error": "",
            "cancellation_status": "querying",
        }

    def cancel_order(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """解析候选、让用户选择并确认，然后调用参数化软取消接口。"""

        try:
            candidates = self._parse_candidates(str(state.get("query_result") or ""))
        except (TypeError, ValueError, SyntaxError) as error:
            return {
                "cancellation_status": "query_failed",
                "cancellation_error": f"订单查询结果无法解析：{error}",
                "order_candidates": [],
            }
        user_id = str(state.get("user_id") or "")
        # 查询提示已经要求 user_id；这里再做确定性过滤，避免把模型漏筛的
        # 其他用户记录展示到选择/确认中断里。
        candidates = [
            candidate
            for candidate in candidates
            if candidate.get("user_id") == user_id
        ]
        if not candidates:
            return {
                "cancellation_status": "order_not_found",
                "cancellation_error": "没有找到符合条件且可取消的订单",
                "order_candidates": [],
            }

        selected = candidates[0]
        if len(candidates) > 1:
            selection = interrupt(
                {
                    "type": "select_order_for_cancellation",
                    "message": (
                        "找到多笔可取消订单，请先选择一笔：\n"
                        + self._format_candidates(candidates)
                        + "\n请回复序号或订单号。"
                    ),
                    "candidates": candidates,
                }
            )
            selected = self._resolve_selection(selection, candidates)
            if selected is None:
                return {
                    "cancellation_status": "selection_failed",
                    "cancellation_error": "未能识别要取消的订单",
                    "order_candidates": candidates,
                }

        confirmation = interrupt(
            {
                "type": "confirm_order_cancellation",
                "message": (
                    "请确认是否取消以下订单：\n"
                    + self._format_order(selected)
                    + "\n回复“确认取消”后才会执行。"
                ),
                "order": selected,
            }
        )
        if not self._is_confirmation(confirmation):
            return {
                "cancellation_status": "cancelled_by_user",
                "cancellation_error": "您没有确认取消，订单保持不变",
                "order_candidates": candidates,
                "selected_order": selected,
            }

        result = self._booking_db_factory().cancel_booking(
            user_id=str(state["user_id"]),
            order_no=str(selected["order_no"]),
        )
        if result.success:
            order = result.order
            saved = (
                {
                    "order_no": order.order_no,
                    "house_id": order.house_id,
                    "house_title": order.house_title,
                    "check_in_date": order.check_in_date,
                    "check_out_date": order.check_out_date,
                    "status": order.status,
                    "price": order.price,
                }
                if order is not None
                else selected
            )
            return {
                "cancellation_status": "success",
                "cancellation_error": "",
                "order_candidates": candidates,
                "selected_order": saved,
            }

        status_by_reason = {
            "order_not_found": "order_not_found",
            "already_cancelled": "already_cancelled",
            "already_started": "already_started",
            "not_cancellable": "not_cancellable",
        }
        return {
            "cancellation_status": status_by_reason.get(
                result.reason, "cancel_failed"
            ),
            "cancellation_error": result.reason,
            "order_candidates": candidates,
            "selected_order": selected,
        }

    def generate_answer(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """根据全局取消状态生成唯一一条用户可见回复。"""

        status = str(state.get("cancellation_status") or "cancel_failed")
        if status == "querying":
            query_status = state.get("query_status")
            if query_status == "empty":
                status = "order_not_found"
            elif query_status == "failed":
                status = "query_failed"

        selected = state.get("selected_order") or {}
        messages = {
            "input_invalid": str(state.get("cancellation_error") or "取消信息无效"),
            "order_not_found": "没有找到属于您且符合条件的可取消订单。",
            "query_failed": "订单查询暂时不可用，请稍后重试。",
            "selection_failed": "没有识别出您选择的订单，本次未执行取消。",
            "cancelled_by_user": "已停止取消，订单保持不变。",
            "already_cancelled": "该订单已经取消，无需重复操作。",
            "already_started": "该订单已经开始入住，无法在线取消。",
            "not_cancellable": "该订单当前状态不允许取消。",
            "cancel_failed": "订单取消失败，请稍后重试。",
        }
        if status == "success":
            content = (
                f"订单已取消。订单号：{selected.get('order_no', '')}，"
                f"房源：{selected.get('house_title', '')}，入住："
                f"{selected.get('check_in_date', '')}，退房："
                f"{selected.get('check_out_date', '')}。"
            )
        else:
            content = messages.get(status, "订单取消失败，请稍后重试。")
        return {
            "cancellation_status": status,
            "messages": [AIMessage(content=content)],
        }

    @staticmethod
    def _recent_conversation(messages: Sequence[Any]) -> list[Any]:
        return [
            message
            for message in messages
            if isinstance(message, (HumanMessage, AIMessage))
        ][-5:]

    @staticmethod
    def _parse_candidates(raw: str) -> list[dict[str, Any]]:
        """解析 SQLDatabase 的元组字符串，不执行其中任何表达式。"""

        normalized = re.sub(r"UUID\('([^']+)'\)", r"'\1'", raw)
        normalized = re.sub(r"Decimal\('([^']+)'\)", r"'\1'", normalized)

        def replace_date(match: re.Match[str]) -> str:
            year, month, day = (int(part) for part in match.groups())
            return repr(date(year, month, day).isoformat())

        normalized = re.sub(
            r"datetime\.date\((\d+),\s*(\d+),\s*(\d+)\)",
            replace_date,
            normalized,
        )
        rows = ast.literal_eval(normalized)
        if not isinstance(rows, (list, tuple)):
            raise ValueError("查询结果不是记录列表")

        candidates: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) != 8:
                raise ValueError("订单记录列数不正确")
            candidates.append(
                {
                    "order_no": str(row[0]),
                    "user_id": str(row[1]),
                    "house_id": int(row[2]),
                    "house_title": str(row[3]),
                    "check_in_date": str(row[4]),
                    "check_out_date": str(row[5]),
                    "status": str(row[6]),
                    "price": float(row[7]) if row[7] is not None else None,
                }
            )
        return candidates

    @classmethod
    def _format_candidates(cls, candidates: Sequence[Mapping[str, Any]]) -> str:
        return "\n".join(
            f"{index}. {cls._format_order(order)}"
            for index, order in enumerate(candidates, start=1)
        )

    @staticmethod
    def _format_order(order: Mapping[str, Any]) -> str:
        return (
            f"订单号：{order.get('order_no', '')}；房源："
            f"{order.get('house_title', '')}；入住："
            f"{order.get('check_in_date', '')}；退房："
            f"{order.get('check_out_date', '')}；月租："
            f"{format_price(order.get('price'))}元"
        )

    def _resolve_selection(
        self,
        answer: Any,
        candidates: Sequence[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """让 LLM 把自然语言选择统一解析成 [1, n] 内的候选序号。"""

        if not candidates:
            return None
        answer_text = (
            json.dumps(answer, ensure_ascii=False, default=str)
            if isinstance(answer, Mapping)
            else str(answer or "").strip()
        )
        if not answer_text:
            return None

        try:
            parser = self._model_factory().with_structured_output(
                OrderSelection,
                method="function_calling",
            )
            result = parser.invoke(
                [
                    SystemMessage(
                        content=(
                            "你是候选订单选择解析器。请根据候选列表和用户回复，"
                            "只返回用户选择的候选序号。有效边界是"
                            f"[1, {len(candidates)}]。用户可以回复数字、中文序数、"
                            "订单号、房源名称或其他能唯一指向某条候选的描述。"
                            "如果无法唯一确定或用户给出的序号越界，selection返回null。"
                            "\n\n候选订单：\n"
                            + self._format_candidates(candidates)
                        )
                    ),
                    HumanMessage(content=answer_text),
                ]
            )
            if not isinstance(result, OrderSelection):
                result = OrderSelection.model_validate(result)
        except Exception:
            return None

        index = result.selection
        if index is None or not 1 <= index <= len(candidates):
            return None
        return candidates[index - 1]

    @staticmethod
    def _is_confirmation(answer: Any) -> bool:
        if isinstance(answer, Mapping):
            value = answer.get("confirmed")
            if isinstance(value, bool):
                return value
            answer = answer.get("answer", "")
        if isinstance(answer, bool):
            return answer
        normalized = re.sub(r"[\s，。！!]", "", str(answer or "")).lower()
        return normalized in {"是", "确认", "确认取消", "取消吧", "yes", "y"}
