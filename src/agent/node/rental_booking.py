"""预订租房子图内部节点：输入校验、下单事务和固定格式结果。"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from datetime import date
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from agent.common.booking_db import BookingDB, HouseCandidate, format_price
from agent.common.llm import build_chat_model


PHONE_PATTERN = re.compile(r"^1[3-9]\d{9}$")

BOOKING_REQUIRED_FIELDS: dict[str, str] = {
    "phone": "手机号",
    "house_title": "房源名称",
    "check_in_date": "入住日期（YYYY-MM-DD）",
    "check_out_date": "退房日期（YYYY-MM-DD）",
}


class HouseSelection(BaseModel):
    """让 DeepSeek 从用户回复中识别所选房源序号。"""

    selection: int | None = Field(
        default=None,
        ge=1,
        description="用户选择的房源序号（从1开始，对应候选列表中的第几套），无法确定时为null",
    )


class RentalBookingNodes:
    """封装预订子图的纯逻辑节点；写库通过 BookingDB 注入。"""

    def __init__(
        self,
        *,
        booking_db_factory: Callable[[], BookingDB],
        model_factory: Callable[[], Any] = build_chat_model,
    ) -> None:
        self._booking_db_factory = booking_db_factory
        self._model_factory = model_factory

    def initialize(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """进入预订流程时初始化内部状态，不覆盖用户已提供的收集字段。"""

        return {
            "collection_status": "collecting",
            "missing_required_fields": list(BOOKING_REQUIRED_FIELDS),
            "llm_call_count": 0,
            "max_llm_calls": 5,
            "booking_status": "collecting",
            "booking_error": "",
            "query_request": "",
            "table_name": "house",
            "max_rows": 1,
            "query_status": "pending",
            "query_result": "",
            "query_error": "",
            "order_no": "",
            "order_info": {},
        }

    def prepare_house_validation(
        self,
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        """校验预订信息，并在通过后准备房源存在性查询。"""

        phone = (state.get("phone") or "").strip()
        title = (state.get("house_title") or "").strip()
        check_in = (state.get("check_in_date") or "").strip()
        check_out = (state.get("check_out_date") or "").strip()

        errors: list[str] = []
        if not PHONE_PATTERN.fullmatch(phone):
            errors.append("手机号格式不正确")
        try:
            in_date = date.fromisoformat(check_in)
            out_date = date.fromisoformat(check_out)
        except ValueError:
            errors.append("入住/退房日期格式不正确")
        else:
            if in_date <= date.today():
                errors.append("入住日期必须晚于今天")
            if out_date <= in_date:
                errors.append("退房日期必须晚于入住日期")

        if errors:
            return {
                "booking_status": "input_invalid",
                "booking_error": "；".join(errors),
            }
        return {
            "booking_status": "validating_house",
            "booking_error": "",
            "query_request": (
                f"查询 house 表中标题包含「{title}」的房源是否存在，"
                "只返回 id、title、price 三列，最多 1 条。"
            ),
            "table_name": "house",
            "max_rows": 1,
            "query_status": "pending",
            "query_error": "",
        }

    def create_order(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """在写适配器的事务内创建订单，并把结果写回状态。

        用户身份由主图解析后继承自 state.user_id。
        """

        phone = (state.get("phone") or "").strip()
        check_in = (state.get("check_in_date") or "").strip()
        check_out = (state.get("check_out_date") or "").strip()
        house_title = (state.get("house_title") or "").strip()
        user_id = (state.get("user_id") or "").strip()
        if not user_id:
            return {
                "booking_status": "order_failed",
                "booking_error": "缺少用户身份（user_id），无法创建订单",
            }

        result = self._booking_db_factory().create_booking(
            house_title=house_title,
            phone=phone,
            check_in_date=check_in,
            check_out_date=check_out,
            user_id=user_id,
        )
        if result.candidates:
            # 规范化包含匹配命中多套房源，中断让用户确认后再下单。
            selected = self._confirm_house_selection(result.candidates)
            if selected is None:
                return {
                    "booking_status": "order_failed",
                    "booking_error": "未能识别您选择的房源，请从候选中选择一套",
                }
            result = self._booking_db_factory().create_booking(
                house_title=selected,
                phone=phone,
                check_in_date=check_in,
                check_out_date=check_out,
                user_id=user_id,
            )
        if not result.success:
            if result.candidates:
                return {
                    "booking_status": "order_failed",
                    "booking_error": "房源匹配异常，请重新发起预订",
                }
            return {
                "booking_status": "order_failed",
                "booking_error": result.error,
            }

        return {
            "booking_status": "success",
            "order_no": result.order_no,
            "order_info": {
                "order_no": result.order_no,
                "house_id": result.house_id,
                "house_title": result.house_title or house_title,
                "phone": phone,
                "check_in_date": check_in,
                "check_out_date": check_out,
                "price": result.price,
            },
            "booking_error": "",
        }

    @staticmethod
    def _format_candidate_lines(
        candidates: tuple[HouseCandidate, ...],
    ) -> str:
        """把候选房源格式化为“序号. 标题（价格元/月）”列表。"""

        return "\n".join(
            f"{index}. {candidate.title}（{format_price(candidate.price)}元/月）"
            for index, candidate in enumerate(candidates, start=1)
        )

    def _confirm_house_selection(
        self,
        candidates: tuple[HouseCandidate, ...],
    ) -> str | None:
        """命中多套候选时中断询问用户，返回选中的完整房源名称。"""

        lines = self._format_candidate_lines(candidates)
        answer = interrupt(
            {
                "type": "confirm_house_selection",
                "message": (
                    f"匹配到以下 {len(candidates)} 套房源，请确认要预订哪一套：\n"
                    f"{lines}\n请回复序号或完整房源名称。"
                ),
                "candidates": [
                    {
                        "house_id": candidate.house_id,
                        "title": candidate.title,
                        "price": candidate.price,
                    }
                    for candidate in candidates
                ],
            }
        )
        return self._resolve_candidate_selection(answer, candidates)

    def _resolve_candidate_selection(
        self,
        answer: Any,
        candidates: tuple[HouseCandidate, ...],
    ) -> str | None:
        """让 LLM 从用户回复中识别用户选择的是第几套候选。

        只把候选列表和用户最新回复喂给模型，不传完整对话历史。
        """

        if not isinstance(answer, str) or not answer.strip():
            return None
        try:
            structured = self._model_factory().with_structured_output(
                HouseSelection,
                method="function_calling",
            )
            result = structured.invoke(
                self._build_selection_messages(answer, candidates)
            )
        except Exception:
            # 模型调用失败时按“无法识别”处理，不阻断预订主流程。
            return None
        if not isinstance(result, BaseModel) or result.selection is None:
            return None
        index = int(result.selection)
        if 1 <= index <= len(candidates):
            return candidates[index - 1].title
        return None

    def _build_selection_messages(
        self,
        answer: str,
        candidates: tuple[HouseCandidate, ...],
    ) -> list[SystemMessage | HumanMessage]:
        """构造识别消息：系统提示（含候选列表）+ 用户最新回复。"""

        lines = self._format_candidate_lines(candidates)
        prompt = (
            "你是房源选择识别器。用户在预订流程中被要求从候选房源中选择一套，"
            "下面给出候选列表和用户回复，请识别用户选择的是第几套。\n\n"
            f"候选列表：\n{lines}\n\n"
            f"用户回复：{answer}\n\n"
            "规则：\n"
            "- 用户说“第2套”“第二个”“2”“选2号”等 → 返回 2\n"
            "- 用户回复某个候选的完整名称或明显指向某一套 → 返回对应序号\n"
            "- 用户回复无法确定对应哪一套（如只说“随便”“都可以”，"
            "或片段同时匹配多套）→ selection 返回 null"
        )
        return [
            SystemMessage(content=prompt),
            HumanMessage(content=answer),
        ]

    def generate_answer(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """归一化最终预订状态，并生成唯一的用户可见回复。"""

        status = state.get("booking_status")
        error = str(state.get("booking_error") or "").strip()
        if state.get("collection_status") != "complete":
            status = "information_incomplete"
            missing = list(state.get("missing_required_fields", []))
            labels = [BOOKING_REQUIRED_FIELDS.get(name, name) for name in missing]
            if state.get("collection_error"):
                error = "信息收集服务暂时不可用，请稍后重新发起预订"
            else:
                error = "仍缺少必要信息：" + "、".join(labels)
        elif status == "validating_house":
            query_status = state.get("query_status")
            if query_status == "empty":
                status = "house_not_found"
                error = "该房源不存在"
            elif query_status == "failed":
                status = "house_query_failed"
                error = "房源查询暂时不可用，请稍后重试"

        if status == "success":
            info = state.get("order_info") or {}
            price_text = format_price(info.get("price"))
            content = (
                "预订成功！订单号："
                f"{info.get('order_no', '')}，房源：{info.get('house_title', '')}，"
                f"手机号：{info.get('phone', '')}，入住：{info.get('check_in_date', '')}，"
                f"退房：{info.get('check_out_date', '')}，月租：{price_text}元"
            )
        else:
            content = "预订失败：" + (error or "未知原因")
        return {
            "booking_status": status,
            "booking_error": error,
            "messages": [AIMessage(content=content)],
        }
