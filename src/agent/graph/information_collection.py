"""可复用的必要信息收集子图编排入口。"""

from collections.abc import Callable
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from agent.common.collection import CollectionSpec, validate_collection_contract
from agent.common.llm import build_chat_model
from agent.node.information_collection import InformationCollectionNodes
from agent.state.information_collection import (
    RecommendCollectionState,
    RecommendInformation,
)


def build_information_collection_graph(
    *,
    state_schema: type[Any],
    extraction_schema: type[BaseModel],
    spec: CollectionSpec,
    model_factory: Callable[[], Any] = build_chat_model,
    name: str = "information_collection",
):
    """构建一个收齐必要字段后结束的信息收集子图。

    返回的子图自身不创建检查点存储器，作为子图使用时会继承父图的持久化能力。
    父图调用时必须提供 ``thread_id``，中断后才能通过
    ``Command(resume=...)`` 恢复同一个会话。
    """

    validate_collection_contract(state_schema, extraction_schema, spec)
    nodes = InformationCollectionNodes(
        spec=spec,
        extraction_schema=extraction_schema,
        model_factory=model_factory,
    )

    # 节点职责保持单一：判断状态、抽取信息、再次判断、询问缺失信息。
    builder = StateGraph(state_schema)
    builder.add_node("evaluate_initial_state", nodes.evaluate)
    builder.add_node("extract_information", nodes.extract)
    builder.add_node("evaluate_collection", nodes.evaluate)
    builder.add_node(
        "ask_for_missing_information", nodes.ask_for_missing_information
    )

    # 首次进入先检查已有状态。字段已经完整或次数已经耗尽时无需调用 LLM。
    builder.add_edge(START, "evaluate_initial_state")
    builder.add_conditional_edges(
        "evaluate_initial_state",
        _route_before_extraction,
        {"extract": "extract_information", "end": END},
    )
    builder.add_edge("extract_information", "evaluate_collection")
    builder.add_conditional_edges(
        "evaluate_collection",
        _route_after_extraction,
        {"ask": "ask_for_missing_information", "end": END},
    )
    # 用户补充信息并恢复执行后，再次抽取并判断，直到完成或达到次数上限。
    builder.add_edge("ask_for_missing_information", "extract_information")

    return builder.compile(name=name)


def _route_before_extraction(
    state: dict[str, Any],
) -> Literal["extract", "end"]:
    return "extract" if state["collection_status"] == "collecting" else "end"


def _route_after_extraction(state: dict[str, Any]) -> Literal["ask", "end"]:
    return "ask" if state["collection_status"] == "collecting" else "end"


RECOMMEND_COLLECTION = CollectionSpec(
    required_fields={
        "city": "租房城市",
        "budget_min": "最低预算",
        "budget_max": "最高预算",
    },
    optional_fields={
        "districts": "一个或多个区域",
        "room_types": "一个或多个房型",
        "rental_mode": "租赁方式",
    },
    max_llm_calls=5,
)


# 这个具体实例注册在 langgraph.json 中，便于直接通过 Studio 测试。
# 其他业务只需提供自己的 State、抽取模型和 CollectionSpec，即可复用同一套编排。
information_collection_graph = build_information_collection_graph(
    state_schema=RecommendCollectionState,
    extraction_schema=RecommendInformation,
    spec=RECOMMEND_COLLECTION,
    name="recommend_information_collection",
)
