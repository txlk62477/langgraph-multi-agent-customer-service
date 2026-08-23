"""预订租房子图：信息收集、房源校验、事务创建订单与固定格式结果。"""

from collections.abc import Callable
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agent.common.booking_db import BookingDB, PostgresBookingDB
from agent.common.collection import CollectionSpec
from agent.common.database import SQLTools
from agent.common.llm import build_chat_model
from agent.graph.database_query import build_database_query_graph
from agent.graph.information_collection import build_information_collection_graph
from agent.node.rental_booking import RentalBookingNodes
from agent.state.rental_booking import (
    BookingState,
    RentalBookingInput,
    RentalBookingOutput,
)


class BookingInformation(BaseModel):
    """让 DeepSeek 按固定结构抽取预订信息的输出模型。"""

    phone: str | None = Field(default=None, description="用户11位大陆手机号")
    house_title: str | None = Field(
        default=None, description="要预订的房源名称（house表的title）"
    )
    check_in_date: str | None = Field(
        default=None, description="入住日期，格式YYYY-MM-DD"
    )
    check_out_date: str | None = Field(
        default=None, description="退房日期，格式YYYY-MM-DD"
    )


BOOKING_COLLECTION = CollectionSpec(
    required_fields={
        "phone": "手机号",
        "house_title": "房源名称",
        "check_in_date": "入住日期（YYYY-MM-DD）",
        "check_out_date": "退房日期（YYYY-MM-DD）",
    },
    optional_fields={},
    max_llm_calls=5,
)


def build_rental_booking_graph(
    *,
    model_factory: Callable[[], Any] = build_chat_model,
    booking_db_factory: Callable[[], BookingDB] = PostgresBookingDB,
    sql_tools_factory: Callable[[Any], SQLTools] | None = None,
    name: str = "rental_booking",
):
    """构建可嵌入主图、也可在Studio独立运行的预订子图。"""

    nodes = RentalBookingNodes(
        booking_db_factory=booking_db_factory,
        model_factory=model_factory,
    )
    collection_subgraph = build_information_collection_graph(
        state_schema=BookingState,
        extraction_schema=BookingInformation,
        spec=BOOKING_COLLECTION,
        model_factory=model_factory,
        name=f"{name}_information_collection",
    )
    house_query_subgraph = build_database_query_graph(
        allowed_tables={"house"},
        model_factory=model_factory,
        sql_tools_factory=sql_tools_factory,
        name=f"{name}_house_query",
    )

    builder = StateGraph(
        BookingState,
        input_schema=RentalBookingInput,
        output_schema=RentalBookingOutput,
    )
    builder.add_node("initialize", nodes.initialize)
    builder.add_node("information_collection", collection_subgraph)
    builder.add_node(
        "prepare_house_validation", nodes.prepare_house_validation
    )
    builder.add_node("check_house", house_query_subgraph)
    builder.add_node("create_order", nodes.create_order)
    builder.add_node("generate_answer", nodes.generate_answer)

    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "information_collection")
    builder.add_conditional_edges(
        "information_collection",
        _route_after_collection,
        {
            "validate": "prepare_house_validation",
            "incomplete": "generate_answer",
        },
    )
    builder.add_conditional_edges(
        "prepare_house_validation",
        _route_after_input_validation,
        {"query": "check_house", "fail": "generate_answer"},
    )
    builder.add_conditional_edges(
        "check_house",
        _route_after_house_check,
        {"create": "create_order", "fail": "generate_answer"},
    )
    builder.add_edge("create_order", "generate_answer")
    builder.add_edge("generate_answer", END)
    return builder.compile(name=name)


def _route_after_collection(state: dict[str, Any]) -> Literal["validate", "incomplete"]:
    return "validate" if state.get("collection_status") == "complete" else "incomplete"


def _route_after_input_validation(
    state: dict[str, Any],
) -> Literal["query", "fail"]:
    return (
        "query"
        if state.get("booking_status") == "validating_house"
        else "fail"
    )


def _route_after_house_check(
    state: dict[str, Any],
) -> Literal["create", "fail"]:
    return "create" if state.get("query_status") == "success" else "fail"


# 独立Studio图：默认使用环境变量中的 PostgreSQL 与 DeepSeek。
rental_booking_graph = build_rental_booking_graph()
