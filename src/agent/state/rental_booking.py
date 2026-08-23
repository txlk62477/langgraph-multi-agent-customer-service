"""预订租房子图状态。"""

from typing import Any, Literal, NotRequired

from langgraph.graph import MessagesState

from agent.state.database_query import QueryStatus
from agent.state.information_collection import CollectionState


BookingStatus = Literal[
    "collecting",
    "information_incomplete",
    "input_invalid",
    "validating_house",
    "house_not_found",
    "house_query_failed",
    "order_failed",
    "success",
]


class RentalBookingInput(MessagesState):
    """预订子图输入：当前对话消息和主图解析出的可选用户身份。"""

    # 主图从运行配置中解析后传入；独立调用缺失时会返回用户身份错误。
    user_id: NotRequired[str | None]


class RentalBookingOutput(MessagesState):
    """预订子图输出：仅返回信息收集提示或最终预订结果消息。"""


class BookingState(CollectionState):
    """预订子图内部状态：信息收集、房源查询和订单创建全过程。

    从 CollectionState 继承 messages、collection_status、
    missing_required_fields、llm_call_count 和 max_llm_calls。
    """

    # 当前用户标识，由主图同名字段传入，创建订单时作为订单归属用户。
    user_id: NotRequired[str | None]

    # information_collection 从对话中提取的大陆手机号。
    phone: NotRequired[str | None]
    # information_collection 从对话中提取的目标房源名称。
    house_title: NotRequired[str | None]
    # information_collection 提取的入住日期，格式为 YYYY-MM-DD。
    check_in_date: NotRequired[str | None]
    # information_collection 提取的退房日期，格式为 YYYY-MM-DD。
    check_out_date: NotRequired[str | None]

    # prepare_house_validation 生成的房源存在性查询要求。
    query_request: NotRequired[str]
    # 房源查询允许访问的目标表，预订流程固定为 house。
    table_name: NotRequired[str]
    # 房源存在性查询的最大返回行数，预订流程固定为 1。
    max_rows: NotRequired[int]
    # database_query 子图返回的查询阶段状态。
    query_status: NotRequired[QueryStatus]
    # database_query 子图返回的数据库原始结果。
    query_result: NotRequired[str]
    # database_query 规划、检查或执行失败时返回的原因。
    query_error: NotRequired[str]

    # 全局预订结果状态，generate_answer 根据它统一选择用户回复。
    booking_status: NotRequired[BookingStatus]
    # 输入校验、用户身份或创建订单失败时的可读原因。
    booking_error: NotRequired[str]
    # create_order 成功后返回的订单唯一编号。
    order_no: NotRequired[str]
    # create_order 保存的订单展示字段，供 generate_answer 生成最终消息。
    order_info: NotRequired[dict[str, Any]]
