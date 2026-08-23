"""推荐租房子图使用的状态。"""

from typing import Any, Literal, NotRequired

from langgraph.graph import MessagesState

from agent.state.information_collection import RecommendCollectionState
from agent.state.database_query import QueryStatus


PreferenceConfirmationStatus = Literal[
    "not_required",
    "pending",
    "confirmed",
    "corrected",
]
RecommendationStatus = Literal[
    "collecting",
    "querying",
    "complete",
    "no_match",
    "failed",
]


class RentalRecommendationInput(MessagesState):
    """推荐子图输入：对话，以及主图已经加载的可选用户偏好。"""

    # 独立调用可直接传入；正式主图通常通过 configurable.user_id 提供身份。
    user_id: NotRequired[str]
    # 嵌入主图时接收已加载偏好，避免推荐子图重复访问 Store。
    user_preferences: NotRequired[dict[str, Any]]


class RentalRecommendationOutput(MessagesState):
    """推荐子图输出：用户可见消息和结构化业务结果。"""

    recommendation_status: NotRequired[RecommendationStatus]


class RentalRecommendationState(RecommendCollectionState):
    """封装偏好补全、信息收集、SQL查询和推荐结果。

    从 RecommendCollectionState 继承 messages、租房条件、信息收集状态和
    模型调用次数等字段，本类只补充推荐流程特有的状态。
    """

    # 当前用户标识，用于独立运行推荐图时从 Store 加载对应用户的长期偏好。
    user_id: NotRequired[str]
    # 已从 Store 加载的长期租房偏好；只用于填补本轮没有明确提供的条件。
    user_preferences: NotRequired[dict[str, Any]]
    # 偏好加载失败原因；加载失败时推荐流程仍可依靠本轮输入继续运行。
    preference_load_error: NotRequired[str]

    # 用户本轮明确给出的条件字段，用于防止长期偏好覆盖本轮新要求。
    explicit_requirement_fields: NotRequired[list[str]]
    # 本轮从长期偏好补入的字段，用户拒绝确认时只清除这些字段。
    prefilled_fields: NotRequired[list[str]]
    # 必要条件由长期偏好补齐后是否需要暂停流程并请求用户确认。
    needs_preference_confirmation: NotRequired[bool]
    # 偏好确认阶段：无需确认、等待确认、已确认或用户已经修正。
    preference_confirmation_status: NotRequired[PreferenceConfirmationStatus]
    # 从最新用户消息提取本轮租房条件时产生的错误。
    requirement_extraction_error: NotRequired[str]

    # 交给通用数据库查询子图的自然语言房源查询要求。
    query_request: NotRequired[str]
    # 本次查询允许访问的目标表，推荐流程固定使用 house。
    table_name: NotRequired[str]
    # 数据库最多返回的记录数，也是最终推荐数量的上限。
    max_rows: NotRequired[int]
    # 数据库查询阶段状态，例如等待、查询成功、空结果或失败。
    query_status: NotRequired[QueryStatus]
    # 数据库工具返回的原始查询结果，推荐生成节点只能依据该结果回答。
    query_result: NotRequired[str]
    # SQL规划、校验、检查或执行过程中产生的错误详情。
    query_error: NotRequired[str]
    # 整个推荐业务的最终状态：收集中、查询中、完成、无匹配或失败。
    recommendation_status: NotRequired[RecommendationStatus]
