"""常规问答子图：上下文处理、联网判断、直接回答或联网回答。"""

from collections.abc import Callable
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from agent.common.llm import build_chat_model
from agent.node.context import build_context_nodes
from agent.node.general_qa import GeneralQANodes
from agent.state.general_qa import GeneralQAInput, GeneralQAOutput, GeneralQAState
from agent.graph.web_search import build_web_search_graph


def build_general_qa_graph(
    *,
    model_factory: Callable[[], Any] = build_chat_model,
    name: str = "general_qa",
):
    """构建可单独调试、也可嵌入客服主图的常规问答子图。"""

    nodes = GeneralQANodes(model_factory=model_factory)
    context_nodes = build_context_nodes()
    web_search_subgraph = build_web_search_graph(name=f"{name}_web_search")

    # 输入、输出Schema只约束子图的公共接口，内部仍使用完整GeneralQAState。
    builder = StateGraph(
        GeneralQAState,
        input_schema=GeneralQAInput,
        output_schema=GeneralQAOutput,
    )
    builder.add_node("prepare_context", context_nodes.prepare_context)
    builder.add_node("decide_search", nodes.decide_search)
    builder.add_node("generate_direct_answer", nodes.generate_direct_answer)
    builder.add_node("web_search", web_search_subgraph)
    builder.add_node(
        "generate_search_failure_answer",
        nodes.generate_search_failure_answer,
    )

    # 每轮只在入口处理一次上下文，后续 LLM 节点按需读取 context_messages。
    builder.add_edge(START, "prepare_context")
    builder.add_edge("prepare_context", "decide_search")
    builder.add_conditional_edges(
        "decide_search",
        _route_after_decision,
        {
            "direct": "generate_direct_answer",
            "search": "web_search",
        },
    )
    builder.add_edge("generate_direct_answer", END)

    # 联网子图把失败原因写入 search_error；实时问题不会用旧知识猜测。
    builder.add_conditional_edges(
        "web_search",
        _route_after_web_search,
        {
            "failure": "generate_search_failure_answer",
            "success": END,
        },
    )
    builder.add_edge("generate_search_failure_answer", END)

    # 子图不自行设置检查点，嵌入主图后继承父图的会话持久化配置。
    return builder.compile(name=name)


def _route_after_decision(state: GeneralQAState) -> Literal["direct", "search"]:
    """把 decide_search 的状态值转换为 LangGraph 条件边标签。"""

    return "search" if state.get("qa_route") == "search" else "direct"


def _route_after_web_search(
    state: GeneralQAState,
) -> Literal["failure", "success"]:
    """联网失败进入降级节点，成功生成答案后直接结束。"""

    return "failure" if state.get("search_error") else "success"


general_qa_graph = build_general_qa_graph()
