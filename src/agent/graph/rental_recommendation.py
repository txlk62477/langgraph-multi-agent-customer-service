"""推荐租房子图：偏好补全、信息收集、只读SQL查询与推荐。"""

from collections.abc import Callable
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore

from agent.common.database import SQLTools
from agent.common.llm import build_chat_model
from agent.graph.database_query import build_database_query_graph
from agent.graph.information_collection import (
    RECOMMEND_COLLECTION,
    build_information_collection_graph,
)
from agent.node.preferences import load_preferences
from agent.node.rental_recommendation import RentalRecommendationNodes
from agent.state.information_collection import RecommendInformation
from agent.state.rental_recommendation import (
    RentalRecommendationInput,
    RentalRecommendationOutput,
    RentalRecommendationState,
)


def build_rental_recommendation_graph(
    *,
    model_factory: Callable[[], Any] = build_chat_model,
    sql_tools_factory: Callable[[Any], SQLTools] | None = None,
    include_preference_loading: bool = False,
    store: BaseStore | None = None,
    name: str = "rental_recommendation",
):
    """构建可嵌入主图、也可在Studio独立运行的推荐子图。"""

    nodes = RentalRecommendationNodes(model_factory=model_factory)
    collection_subgraph = build_information_collection_graph(
        state_schema=RentalRecommendationState,
        extraction_schema=RecommendInformation,
        spec=RECOMMEND_COLLECTION,
        model_factory=model_factory,
        name=f"{name}_information_collection",
    )
    database_query_subgraph = build_database_query_graph(
        allowed_tables={"house"},
        model_factory=model_factory,
        sql_tools_factory=sql_tools_factory,
        name=f"{name}_database_query",
    )

    # 外部接口只接收对话、身份和已加载偏好，只返回消息与推荐业务状态；
    # 信息收集、SQL 和重试字段保留在子图内部状态中。
    builder = StateGraph(
        RentalRecommendationState,
        input_schema=RentalRecommendationInput,
        output_schema=RentalRecommendationOutput,
    )
    if include_preference_loading:
        builder.add_node("load_preferences", load_preferences)
    builder.add_node(
        "extract_current_requirements",
        nodes.extract_current_requirements,
    )
    builder.add_node("prefill_from_preferences", nodes.prefill_from_preferences)
    builder.add_node(
        "confirm_prefilled_requirements",
        nodes.confirm_prefilled_requirements,
    )
    builder.add_node("information_collection", collection_subgraph)
    builder.add_node(
        "collection_incomplete_answer",
        nodes.collection_incomplete_answer,
    )
    builder.add_node(
        "prepare_house_query_request",
        nodes.prepare_house_query_request,
    )
    builder.add_node("database_query", database_query_subgraph)
    builder.add_node(
        "respond_to_query_result",
        nodes.respond_to_query_result,
    )

    first_node = "load_preferences" if include_preference_loading else "extract_current_requirements"
    builder.add_edge(START, first_node)
    if include_preference_loading:
        builder.add_edge("load_preferences", "extract_current_requirements")
    builder.add_edge("extract_current_requirements", "prefill_from_preferences")
    builder.add_conditional_edges(
        "prefill_from_preferences",
        _route_after_prefill,
        {
            "confirm": "confirm_prefilled_requirements",
            "collect": "information_collection",
        },
    )
    builder.add_edge("confirm_prefilled_requirements", "information_collection")
    builder.add_conditional_edges(
        "information_collection",
        _route_after_collection,
        {
            "query": "prepare_house_query_request",
            "incomplete": "collection_incomplete_answer",
        },
    )
    builder.add_edge("collection_incomplete_answer", END)
    builder.add_edge("prepare_house_query_request", "database_query")
    builder.add_edge("database_query", "respond_to_query_result")
    builder.add_edge("respond_to_query_result", END)
    return builder.compile(name=name, store=store)


def _route_after_prefill(
    state: RentalRecommendationState,
) -> Literal["confirm", "collect"]:
    return "confirm" if state.get("needs_preference_confirmation") else "collect"


def _route_after_collection(
    state: RentalRecommendationState,
) -> Literal["query", "incomplete"]:
    return "query" if state.get("collection_status") == "complete" else "incomplete"


# 独立Studio图自行读取偏好；嵌入客服主图时使用不重复读取的构建函数。
rental_recommendation_graph = build_rental_recommendation_graph(
    include_preference_loading=True,
)
