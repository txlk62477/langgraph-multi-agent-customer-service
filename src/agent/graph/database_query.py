"""可复用的自然语言单表只读查询子图编排。"""

from collections.abc import Callable, Collection
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from agent.common.database import SQLTools
from agent.common.llm import build_chat_model
from agent.node.database_query import DatabaseQueryNodes
from agent.state.database_query import (
    DatabaseQueryInput,
    DatabaseQueryOutput,
    DatabaseQueryState,
)


def build_database_query_graph(
    *,
    allowed_tables: Collection[str],
    model_factory: Callable[[], Any] = build_chat_model,
    sql_tools_factory: Callable[[Any], SQLTools] | None = None,
    name: str = "database_query",
):
    """构建自然语言规划、双重校验、执行和重试的查询子图。"""

    nodes = DatabaseQueryNodes(
        allowed_tables=allowed_tables,
        model_factory=model_factory,
        sql_tools_factory=sql_tools_factory,
    )
    # 调用方只需要提供查询请求并接收查询结果；Schema、SQL 和重试状态
    # 全部封装在数据库查询子图内部。
    builder = StateGraph(
        DatabaseQueryState,
        input_schema=DatabaseQueryInput,
        output_schema=DatabaseQueryOutput,
    )
    builder.add_node("initialize_query", nodes.initialize_query)
    builder.add_node("begin_attempt", nodes.begin_attempt)
    builder.add_node("generate_sql", nodes.generate_sql)
    builder.add_node("check_sql", nodes.check_sql)
    builder.add_node("execute_query", nodes.execute_query)

    builder.add_edge(START, "initialize_query")
    builder.add_conditional_edges(
        "initialize_query",
        _route_after_initialization,
        {"query": "begin_attempt", "end": END},
    )
    builder.add_conditional_edges(
        "begin_attempt",
        _route_after_begin_attempt,
        {"query": "generate_sql", "end": END},
    )
    # generate_sql 失败时 check_sql 保持空操作，并由检查节点的条件边重试；
    # 检查通过后才允许进入真正的数据库执行节点。
    builder.add_edge("generate_sql", "check_sql")
    builder.add_conditional_edges(
        "check_sql",
        _route_after_check,
        {"retry": "begin_attempt", "execute": "execute_query"},
    )
    builder.add_conditional_edges(
        "execute_query",
        _route_after_execution,
        {"retry": "begin_attempt", "end": END},
    )
    return builder.compile(name=name)


def _route_after_initialization(
    state: DatabaseQueryState,
) -> Literal["query", "end"]:
    return "end" if state.get("query_status") == "failed" else "query"


def _route_after_begin_attempt(
    state: DatabaseQueryState,
) -> Literal["query", "end"]:
    """只有重试入口负责判断是否已经达到最大尝试次数。"""

    return "end" if state.get("query_status") == "failed" else "query"


def _route_after_execution(
    state: DatabaseQueryState,
) -> Literal["retry", "end"]:
    return (
        "retry"
        if state.get("query_attempt_status") == "failed"
        else "end"
    )


def _route_after_check(
    state: DatabaseQueryState,
) -> Literal["retry", "execute"]:
    return (
        "retry"
        if state.get("query_attempt_status") == "failed"
        else "execute"
    )


# Studio 独立图第一版只开放 house；其他业务通过构建函数固定自己的表白名单。
database_query_graph = build_database_query_graph(allowed_tables={"house"})
