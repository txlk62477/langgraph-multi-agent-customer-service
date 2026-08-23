"""联网搜索和单网页处理子图使用的状态定义。"""

from typing import Annotated, Any, NotRequired, TypedDict
import operator

from langgraph.graph import MessagesState


class WebSearchState(MessagesState):
    """保存问题、搜索结果和浏览器读取结果。"""

    # search_query 是调用方结合上下文规划出的搜索词，可以不提供。
    search_query: NotRequired[str]
    # query 记录本次真正提交给AnySearch的搜索词，供内部节点和调试使用。
    query: NotRequired[str]
    # search_results 保存 AnySearch 返回的原始结构化结果。
    search_results: NotRequired[list[dict[str, Any]]]
    # browser_results 保存 Playwright 渲染网页后的增强结果。
    browser_results: NotRequired[list[dict[str, Any]]]
    # page_results 接收动态网页子图的并行输出；每次搜索开始时会显式清空。
    page_results: Annotated[list[dict[str, Any]], operator.add]
    # 外部搜索或网页读取失败时保存简化错误，供父图决定如何降级。
    search_error: NotRequired[str]
    # 视觉模型不可用时的增强能力错误；不影响 AnySearch 文本兜底。
    vision_error: NotRequired[str]


class WebSearchInput(MessagesState):
    """联网搜索子图的公共输入：消息以及可选的已规划搜索词。"""

    search_query: NotRequired[str]


class WebSearchOutput(MessagesState):
    """联网搜索子图的公共输出：回答消息和可选失败原因。"""

    # 成功时为空字符串；失败时供调用方决定如何生成降级回复。
    search_error: NotRequired[str]


class WebPageInput(TypedDict):
    """动态分发给一个网页处理子图的输入。"""

    query: str
    page: dict[str, Any]


class WebPageState(WebPageInput, total=False):
    """单网页内部状态；截图只在网页读取和视觉分析节点之间短暂存在。"""

    page_result: dict[str, Any]


class WebPageOutput(TypedDict):
    """单网页子图返回父图的最小输出。"""

    page_results: list[dict[str, Any]]
