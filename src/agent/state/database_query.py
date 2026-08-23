"""通用只读数据库查询子图状态。"""

from typing import Literal, NotRequired, TypedDict


QueryStatus = Literal["pending", "querying", "success", "empty", "failed"]
QueryAttemptStatus = Literal["running", "success", "failed"]


class DatabaseQueryInput(TypedDict):
    """数据库查询子图输入：自然语言请求、目标表和可选行数上限。"""

    # 调用方描述的查询目标；规划模型会结合表结构将其转换为结构化查询计划。
    query_request: str
    # 本次查询的目标表名；运行时还会校验它是否位于构图时声明的白名单中。
    table_name: str
    # 最大返回行数；未提供时使用子图默认值，并受系统硬上限约束。
    max_rows: NotRequired[int]


class DatabaseQueryOutput(TypedDict):
    """数据库查询子图输出：结构化状态、原始结果和失败原因。"""

    # 整个查询的最终状态：成功、空结果或失败。
    query_status: QueryStatus
    # 数据库工具返回的原始结果字符串；空结果或失败时为空字符串。
    query_result: str
    # 最后一次失败的阶段和异常摘要；查询成功时为空字符串。
    query_error: str


class DatabaseQueryState(TypedDict):
    """数据库查询子图内部状态；包含公开接口和查询过程字段。"""

    # 以下三个字段来自 DatabaseQueryInput，并在初始化节点完成规范化和校验。
    query_request: str
    table_name: str
    max_rows: NotRequired[int]

    # generate_sql 内部读取 Schema、吸收上次错误反馈并编译出的只读 SQL。
    sql_query: NotRequired[str]
    # 已经开始的完整查询尝试次数，由 begin_attempt 统一递增。
    query_attempt_count: NotRequired[int]
    # 单次图运行允许的最大尝试次数，达到上限后不再重试。
    max_query_attempts: NotRequired[int]
    # 当前一次完整查询尝试的状态；None 表示尚未开始尝试。
    query_attempt_status: NotRequired[QueryAttemptStatus | None]
    # 整个查询流程的状态；成功执行但没有记录时使用 empty。
    query_status: NotRequired[QueryStatus]
    # execute_query 保存的数据库原始结果，供父图生成业务回答。
    query_result: NotRequired[str]
    # 当前或最后一次失败原因；重试时保留供 generate_sql 修正，最终成功后清空。
    query_error: NotRequired[str]
