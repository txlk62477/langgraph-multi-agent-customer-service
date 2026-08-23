"""常规问答子图使用的状态。"""

from typing import Literal, NotRequired

from langgraph.graph import MessagesState

from agent.state.context import ContextState


QARoute = Literal["direct", "search"]


class GeneralQAInput(MessagesState):
    """常规问答子图的公共输入，不暴露内部上下文处理字段。"""

    # 单独调试时可以直接传入；嵌入主图时由主图的同名状态自动传递。
    user_id: NotRequired[str | None]
    thread_id: NotRequired[str]


class GeneralQAOutput(MessagesState):
    """常规问答子图的公共输出：回答消息和联网失败原因。"""

    search_error: NotRequired[str]


class GeneralQAState(ContextState):
    """在上下文状态之上保存问答路由和联网搜索过程数据。"""

    # decide_search 节点输出的路由信息，便于在 LangSmith 中检查决策原因。
    qa_route: NotRequired[QARoute]
    routing_reason: NotRequired[str]
    routing_error: NotRequired[str]
    requires_fresh_data: NotRequired[bool]

    # 结合上下文补全后的搜索词，由联网搜索子图优先使用。
    search_query: NotRequired[str]
    search_error: NotRequired[str]
