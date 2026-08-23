"""独立上下文管理节点使用的状态。"""

from typing import Any, NotRequired

from langchain_core.messages import AnyMessage
from langgraph.graph import MessagesState


class ContextState(MessagesState):
    """保留完整消息，同时提供本轮经过裁剪的模型输入。"""

    # 长期记忆所属的用户；优先从State读取，也可由运行配置或环境变量补充。
    user_id: NotRequired[str]
    # 当前会话标识，用于隔离不同Thread的话题摘要和上下文记忆。
    thread_id: NotRequired[str]
    # 上下文节点根据本轮问题识别出的当前话题名称。
    active_topic: NotRequired[str]
    # 最近消息与相关历史摘要合并后的模型输入，不会覆盖原始messages。
    context_messages: NotRequired[list[AnyMessage]]
    # 从PostgreSQL召回并经模型筛选后，与当前问题相关的历史话题摘要。
    relevant_topic_summaries: NotRequired[list[dict[str, Any]]]
    # 上下文处理统计，包括消息数量、召回模式以及降级过程中产生的错误。
    context_stats: NotRequired[dict[str, Any]]
