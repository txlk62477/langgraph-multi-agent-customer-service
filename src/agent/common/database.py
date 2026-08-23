"""基于官方 SQLDatabaseToolkit 的通用只读数据库适配器。"""

from __future__ import annotations

from collections.abc import Collection
import os
from typing import Any, Protocol

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase


class SQLTools(Protocol):
    """数据库查询子图依赖的最小工具接口。"""

    def inspect_schema(self, table_name: str) -> str:
        """返回指定授权表的结构和样例。"""

    def check_query(self, query: str) -> str:
        """使用官方 SQL checker 检查查询。"""

    def execute_query(self, query: str) -> str:
        """使用官方 SQL query 工具执行只读查询。"""


class LangChainSQLTools:
    """把 SQLDatabaseToolkit 隐藏在通用、可替换的小接口后。"""

    def __init__(
        self,
        *,
        model: Any,
        postgres_uri: str,
        allowed_tables: Collection[str],
    ) -> None:
        normalized_tables = sorted(
            {validate_table_name(table_name) for table_name in allowed_tables}
        )
        if not normalized_tables:
            raise ValueError("allowed_tables 至少需要一个表")

        database = SQLDatabase.from_uri(
            _sqlalchemy_postgres_uri(postgres_uri),
            engine_args={
                "pool_pre_ping": True,
                # 上层即使校验失误，PostgreSQL 连接本身也会拒绝写操作。
                "connect_args": {
                    "options": "-c default_transaction_read_only=on",
                },
            },
            include_tables=normalized_tables,
            sample_rows_in_table_info=2,
            max_string_length=500,
        )
        toolkit = SQLDatabaseToolkit(db=database, llm=model)
        tools = {tool.name: tool for tool in toolkit.get_tools()}
        required = {
            "sql_db_list_tables",
            "sql_db_schema",
            "sql_db_query_checker",
            "sql_db_query",
        }
        missing = required - tools.keys()
        if missing:
            raise RuntimeError(
                "SQLDatabaseToolkit 缺少必要工具：" + "、".join(sorted(missing))
            )
        self._allowed_tables = frozenset(normalized_tables)
        self._tools = tools

    @classmethod
    def from_environment(
        cls,
        *,
        model: Any,
        allowed_tables: Collection[str],
    ) -> "LangChainSQLTools":
        """从 POSTGRES_URI 创建只暴露授权表的官方工具集。"""

        postgres_uri = os.getenv("POSTGRES_URI", "").strip()
        if not postgres_uri:
            raise ValueError("缺少 POSTGRES_URI，无法查询数据库")
        return cls(
            model=model,
            postgres_uri=postgres_uri,
            allowed_tables=allowed_tables,
        )

    def inspect_schema(self, table_name: str) -> str:
        table_name = validate_table_name(table_name)
        if table_name not in self._allowed_tables:
            raise ValueError(f"不允许访问表：{table_name}")
        tables = self._tools["sql_db_list_tables"].invoke("")
        schema = self._tools["sql_db_schema"].invoke(table_name)
        return f"可用表：{tables}\n\n{table_name} 表结构与样例：\n{schema}"

    def check_query(self, query: str) -> str:
        result = self._tools["sql_db_query_checker"].invoke(query)
        return str(result).strip()

    def execute_query(self, query: str) -> str:
        result = self._tools["sql_db_query"].invoke(query)
        return str(result).strip()


def validate_table_name(table_name: str) -> str:
    """只允许简单 PostgreSQL 标识符，禁止把 SQL 片段伪装成表名。"""

    cleaned = table_name.strip()
    if not cleaned or not cleaned.replace("_", "a").isalnum():
        raise ValueError(f"非法表名：{table_name!r}")
    if cleaned[0].isdigit():
        raise ValueError(f"非法表名：{table_name!r}")
    return cleaned


def _sqlalchemy_postgres_uri(postgres_uri: str) -> str:
    """让 SQLAlchemy 明确使用项目已经安装的 psycopg 3 驱动。"""

    cleaned = postgres_uri.strip()
    if cleaned.startswith("postgresql://"):
        return "postgresql+psycopg://" + cleaned.removeprefix("postgresql://")
    return cleaned
