"""推荐租房子图的偏好补全、查询请求准备和答案节点。"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
import re
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt

from agent.common.collection import is_missing
from agent.common.llm import build_chat_model
from agent.state.information_collection import RecommendInformation
from agent.state.rental_recommendation import RentalRecommendationState


ModelFactory = Callable[[], Any]
REQUIRED_FIELDS = ("city", "budget_min", "budget_max")
OPTIONAL_FIELDS = ("districts", "room_types", "rental_mode")
RECOMMENDATION_FIELDS = (*REQUIRED_FIELDS, *OPTIONAL_FIELDS)
MAX_RECOMMENDATIONS = 5

_CONFIRMATIONS = {
    "是",
    "确认",
    "可以",
    "继续",
    "没问题",
    "对",
    "好的",
    "好",
    "yes",
    "ok",
}
class RentalRecommendationNodes:
    """把复杂推荐流程封装在一组可注入依赖的图节点后。"""

    def __init__(
        self,
        *,
        model_factory: ModelFactory = build_chat_model,
    ) -> None:
        self._model_factory = model_factory

    def extract_current_requirements(
        self,
        state: RentalRecommendationState,
    ) -> dict[str, Any]:
        """先从最新用户消息提取本轮条件，防止旧偏好覆盖新要求。"""

        reset: dict[str, Any] = {
            field: None for field in RECOMMENDATION_FIELDS
        }
        reset.update(
            {
                "explicit_requirement_fields": [],
                "prefilled_fields": [],
                "needs_preference_confirmation": False,
                "preference_confirmation_status": "not_required",
                "collection_status": "collecting",
                "missing_required_fields": list(REQUIRED_FIELDS),
                "llm_call_count": 0,
                "max_llm_calls": 5,
                "requirement_extraction_error": "",
                "query_request": "",
                "table_name": "house",
                "max_rows": MAX_RECOMMENDATIONS,
                "query_status": "pending",
                "query_result": "",
                "query_error": "",
                "recommendation_status": "collecting",
            }
        )

        question = _latest_human_text(state.get("messages", []))
        if not question:
            return reset

        try:
            extractor = self._model_factory().with_structured_output(
                RecommendInformation,
                method="function_calling",
            )
            result = extractor.invoke(
                [
                    SystemMessage(
                        content=(
                            "你是租房推荐条件提取器。只从当前用户消息提取用户明确表达的"
                            "租房条件，不使用历史消息，不猜测。city是城市；budget_min和"
                            "budget_max是月租最低和最高预算；districts和room_types均返回"
                            "列表；rental_mode只能是whole_rent或share_rent。未提到的字段"
                            "返回null。"
                        )
                    ),
                    HumanMessage(content=question),
                ]
            )
            if not isinstance(result, RecommendInformation):
                result = RecommendInformation.model_validate(result)
            extracted = result.model_dump(exclude_none=True)
            reset.update(extracted)
            reset["explicit_requirement_fields"] = list(extracted)
            reset["llm_call_count"] = 1
        except Exception as error:
            reset["llm_call_count"] = 1
            reset["requirement_extraction_error"] = (
                f"{type(error).__name__}: {error}"
            )
        return reset

    def prefill_from_preferences(
        self,
        state: RentalRecommendationState,
    ) -> dict[str, Any]:
        """只用长期偏好填空，并记录哪些字段需要用户确认。"""

        preferences = state.get("user_preferences", {})
        updates: dict[str, Any] = {}
        prefilled: list[str] = []
        explicit = set(state.get("explicit_requirement_fields", []))
        current_city = state.get("city")
        preference_city = preferences.get("city")

        for field in REQUIRED_FIELDS:
            value = state.get(field)
            preferred = preferences.get(field)
            if is_missing(value) and not is_missing(preferred):
                updates[field] = _normalize_preference_value(field, preferred)
                prefilled.append(field)

        effective_city = updates.get("city", current_city)
        for field in OPTIONAL_FIELDS:
            value = state.get(field)
            preferred = preferences.get(field)
            if not is_missing(value) or is_missing(preferred):
                continue
            # 用户本轮明确换城市时，不带入旧城市的区域偏好。
            if (
                field == "districts"
                and "city" in explicit
                and not is_missing(preference_city)
                and effective_city != preference_city
            ):
                continue
            updates[field] = _normalize_preference_value(field, preferred)
            prefilled.append(field)

        effective = {field: updates.get(field, state.get(field)) for field in REQUIRED_FIELDS}
        missing = [field for field, value in effective.items() if is_missing(value)]
        needs_confirmation = not missing and any(
            field in REQUIRED_FIELDS for field in prefilled
        )
        updates.update(
            {
                "prefilled_fields": prefilled,
                "needs_preference_confirmation": needs_confirmation,
                "preference_confirmation_status": (
                    "pending" if needs_confirmation else "not_required"
                ),
            }
        )
        return updates

    def confirm_prefilled_requirements(
        self,
        state: RentalRecommendationState,
    ) -> dict[str, Any]:
        """确认由长期偏好补齐的必要字段；修改时交回信息收集子图。"""

        answer = interrupt(
            {
                "type": "confirm_rental_preferences",
                "message": (
                    "我将按照以下条件推荐房源："
                    + _format_requirements(state)
                    + "。是否继续？如果需要修改，请直接告诉我新条件。"
                ),
                "requirements": {
                    field: state.get(field)
                    for field in RECOMMENDATION_FIELDS
                    if not is_missing(state.get(field))
                },
                "prefilled_fields": list(state.get("prefilled_fields", [])),
            }
        )
        answer_text = _answer_to_text(answer)
        if _is_confirmation(answer_text):
            return {
                "needs_preference_confirmation": False,
                "preference_confirmation_status": "confirmed",
            }

        # 用户未确认时只清除来自偏好的值；本轮明确提供的条件仍然保留。
        updates = {
            field: None for field in state.get("prefilled_fields", [])
        }
        updates.update(
            {
                "messages": [HumanMessage(content=answer_text)],
                "prefilled_fields": [],
                "needs_preference_confirmation": False,
                "preference_confirmation_status": "corrected",
                "collection_status": "collecting",
            }
        )
        return updates

    def collection_incomplete_answer(
        self,
        state: RentalRecommendationState,
    ) -> dict[str, Any]:
        missing = state.get("missing_required_fields", [])
        labels = {
            "city": "租房城市",
            "budget_min": "最低月租预算",
            "budget_max": "最高月租预算",
        }
        names = "、".join(labels.get(field, field) for field in missing)
        if state.get("collection_error"):
            message = "信息收集服务暂时不可用，请稍后重新发起推荐。"
        else:
            message = (
                f"暂时无法查询房源，因为仍缺少必要信息：{names}。"
                "请补充后重新发起推荐。"
            )
        return {
            "messages": [
                AIMessage(
                    content=message
                )
            ],
            "recommendation_status": "failed",
        }

    def prepare_house_query_request(
        self,
        state: RentalRecommendationState,
    ) -> dict[str, Any]:
        """把已确认条件写成通用数据库子图能够规划的自然语言请求。"""

        requested_columns = (
            "title、price、city_name、region_name、community_name、"
            "detail_address、house_type、rooms、area、floor、all_floor、"
            "rent_type、intro、devices、head_image"
        )
        return {
            "query_request": (
                "查询严格满足以下租房条件的房源，不要自行放宽任何条件："
                + _format_requirements(state)
                + "。只返回这些字段："
                + requested_columns
                + "。优先按价格从低到高排序。"
            ),
            "table_name": "house",
            "max_rows": MAX_RECOMMENDATIONS,
            "query_status": "pending",
            "query_result": "",
            "query_error": "",
            "recommendation_status": "querying",
        }

    def respond_to_query_result(
        self,
        state: RentalRecommendationState,
    ) -> dict[str, Any]:
        """根据查询状态生成成功、无匹配或失败三类最终回复。"""

        query_status = state.get("query_status")
        if query_status == "empty":
            conditions = _format_no_match_requirements(state)
            return {
                "messages": [
                    AIMessage(
                        content=(
                            f"没有找到符合以下条件的房源：{conditions}。"
                            "您可以调整区域或预算后再试。"
                        )
                    )
                ],
                "recommendation_status": "no_match",
            }

        if query_status != "success":
            # 未知状态也按失败处理，避免误用空结果生成或编造推荐。
            return {
                "messages": [
                    AIMessage(content="房源查询暂时失败，请稍后重试。")
                ],
                "recommendation_status": "failed",
            }

        try:
            response = self._model_factory().invoke(
                [
                    SystemMessage(
                        content=(
                            "你是租房推荐客服。只能依据给出的PostgreSQL查询结果推荐，"
                            "不得编造房源。最多推荐5套，不展示房源ID、发布者ID、经纬度"
                            "或原始SQL。每套说明标题、月租、城市区域、小区和地址、房型、"
                            "面积、楼层、租赁方式、主要配套与推荐理由。数据库字段顺序为："
                            "title, price, city_name, region_name, community_name, "
                            "detail_address, house_type, rooms, area, floor, all_floor, "
                            "rent_type, intro, devices, head_image。将whole/whole_rent译为"
                            "整租，shared/share_rent译为合租。只输出一个综合推荐结果。"
                        )
                    ),
                    HumanMessage(
                        content=(
                            "用户确认的条件："
                            + _format_requirements(state)
                            + "\n数据库查询结果：\n"
                            + state.get("query_result", "")
                        )
                    ),
                ]
            )
            return {
                "messages": [response],
                "recommendation_status": "complete",
            }
        except Exception as error:
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "已经查到符合条件的房源，但推荐说明生成失败，请稍后重试。"
                        )
                    )
                ],
                "query_error": f"推荐生成失败：{type(error).__name__}: {error}",
                "recommendation_status": "failed",
            }


def _normalize_preference_value(field: str, value: Any) -> Any:
    if field in {"budget_min", "budget_max"}:
        return int(value)
    if field in {"districts", "room_types"}:
        return list(value) if isinstance(value, list) else [str(value)]
    return value


def _format_requirements(state: Mapping[str, Any]) -> str:
    labels = {
        "city": "城市",
        "budget_min": "最低月租",
        "budget_max": "最高月租",
        "districts": "区域",
        "room_types": "房型",
        "rental_mode": "租赁方式",
    }
    parts = [
        f"{labels[field]}={state.get(field)}"
        for field in RECOMMENDATION_FIELDS
        if not is_missing(state.get(field))
    ]
    return "；".join(parts)


def _format_no_match_requirements(state: Mapping[str, Any]) -> str:
    """把无匹配时的已确认条件格式化为用户可读摘要。"""

    parts: list[str] = []
    city = state.get("city")
    districts = state.get("districts") or []
    budget_min = state.get("budget_min")
    budget_max = state.get("budget_max")
    room_types = state.get("room_types") or []
    rental_mode = state.get("rental_mode")

    if not is_missing(city):
        parts.append(str(city))
    if districts:
        parts.append("、".join(str(item) for item in districts))
    if not is_missing(budget_min) and not is_missing(budget_max):
        parts.append(
            f"预算 {_format_money(budget_min)}–{_format_money(budget_max)} 元/月"
        )
    if room_types:
        parts.append("、".join(str(item) for item in room_types))
    if rental_mode:
        parts.append(
            {"whole_rent": "整租", "share_rent": "合租"}.get(
                str(rental_mode), str(rental_mode)
            )
        )
    return " · ".join(parts) or "当前筛选条件"


def _format_money(value: Any) -> str:
    """价格去掉无意义的小数 .0。"""

    if isinstance(value, (int, float)):
        return f"{value:g}"
    return str(value)


def _latest_human_text(messages: Sequence[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage) and isinstance(message.content, str):
            if message.content.strip():
                return message.content.strip()
    return ""


def _answer_to_text(answer: Any) -> str:
    if isinstance(answer, str):
        return answer.strip()
    return json.dumps(answer, ensure_ascii=False, default=str)


def _is_confirmation(answer: str) -> bool:
    normalized = re.sub(r"[\s，。！？,.!?]", "", answer).lower()
    return normalized in _CONFIRMATIONS
