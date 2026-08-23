"""可复用的联网搜索图和动态单网页处理子图编排。"""

from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agent.node.web_search import (
    WebPageProcessingNodes,
    build_search_web_node,
    build_webpage_processing_nodes,
    finalize_webpages,
    generate_answer,
)
from agent.state.web_search import (
    WebPageInput,
    WebPageOutput,
    WebPageState,
    WebSearchInput,
    WebSearchOutput,
    WebSearchState,
)


def build_webpage_processing_graph(
    *,
    nodes: WebPageProcessingNodes | None = None,
    name: str = "process_webpage",
):
    """构建“读取单页→立即视觉识别”的流水线子图。"""

    page_nodes = nodes or build_webpage_processing_nodes()
    builder = StateGraph(
        WebPageState,
        input_schema=WebPageInput,
        output_schema=WebPageOutput,
    )
    builder.add_node("playwright_read_page", page_nodes.playwright_read_page)
    builder.add_node("analyze_page_visuals", page_nodes.analyze_page_visuals)
    builder.add_edge(START, "playwright_read_page")
    builder.add_edge("playwright_read_page", "analyze_page_visuals")
    builder.add_edge("analyze_page_visuals", END)
    return builder.compile(name=name)


def _route_after_search(state: WebSearchState) -> list[Send] | Literal[END]:
    """AnySearch 失败或无结果时结束；否则为每条结果并行启动一个网页处理子图。"""

    if state.get("search_error"):
        return END
    return [
        Send(
            "process_webpage",
            {
                "query": state["query"],
                "page": result,
            },
        )
        for result in state.get("search_results", [])
    ]


def _route_after_finalize(
    state: WebSearchState,
) -> Literal["answer", "end"]:
    """所有网页均无文本、JSON和视觉证据时不调用最终总结模型。"""

    return "end" if state.get("search_error") else "answer"


def build_web_search_graph(
    *,
    page_nodes: WebPageProcessingNodes | None = None,
    name: str = "web_search",
):
    """构建 AnySearch、并行网页视觉处理和 DeepSeek 组成的联网搜索图。"""

    resolved_page_nodes = page_nodes or build_webpage_processing_nodes()
    webpage_subgraph = build_webpage_processing_graph(
        nodes=resolved_page_nodes,
        name=f"{name}_process_webpage"
    )
    # 对外只接收消息，只返回消息和失败原因；搜索证据等字段留在子图内部。
    builder = StateGraph(
        WebSearchState,
        input_schema=WebSearchInput,
        output_schema=WebSearchOutput,
    )
    builder.add_node(
        "anysearch_search",
        build_search_web_node(resolved_page_nodes),
    )
    builder.add_node("process_webpage", webpage_subgraph)
    builder.add_node("finalize_webpages", finalize_webpages)
    builder.add_node("generate_answer", generate_answer)

    builder.add_edge(START, "anysearch_search")
    # Send会并行运行同一个子图；每个子图内部截图一完成就进入视觉分析节点。
    builder.add_conditional_edges(
        "anysearch_search",
        _route_after_search,
        ["process_webpage", END],
    )
    builder.add_edge("process_webpage", "finalize_webpages")
    builder.add_conditional_edges(
        "finalize_webpages",
        _route_after_finalize,
        {"answer": "generate_answer", "end": END},
    )
    builder.add_edge("generate_answer", END)

    # 不在子图内部配置检查点；嵌入父图时继承父图的持久化设置。
    return builder.compile(name=name)


# 注册到 langgraph.json，既可单独在 Studio 调试，也可作为节点嵌入父图。
web_search_graph = build_web_search_graph()
