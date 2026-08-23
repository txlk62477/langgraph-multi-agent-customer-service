"""供LangGraph Agent Server使用的PostgreSQL检查点工厂。"""

from __future__ import annotations

from contextlib import asynccontextmanager
import os
from typing import AsyncIterator

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver


@asynccontextmanager
async def generate_checkpointer() -> AsyncIterator[AsyncPostgresSaver]:
    """在Server生命周期内复用连接，并自动创建checkpoint所需表。"""

    postgres_uri = os.getenv("POSTGRES_URI", "").strip()
    if not postgres_uri:
        raise ValueError("缺少 POSTGRES_URI，无法初始化 PostgreSQL checkpointer")

    async with AsyncPostgresSaver.from_conn_string(postgres_uri) as saver:
        # setup使用CREATE TABLE IF NOT EXISTS等迁移，重复启动时可以安全执行。
        await saver.setup()
        yield saver
