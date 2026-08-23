"""AnySearch 联网搜索的 LangChain 工具。"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from langchain_core.tools import tool


ANYSEARCH_SEARCH_URL = "https://api.anysearch.com/v1/search"


class AnySearchError(RuntimeError):
    """AnySearch API 返回失败响应。"""


def search_anysearch(
    query: str,
    *,
    max_results: int = 5,
    tag: str | None = None,
    params: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """调用 AnySearch，并返回适合交给大模型的结构化结果。

    默认不指定 ``tag``，由 AnySearch 根据问题自动选择搜索来源。只有明确
    使用垂直搜索时才传入 ``tag`` 和对应的 ``params``。
    """

    query = query.strip()
    if not query:
        raise ValueError("搜索词不能为空")
    if not 1 <= max_results <= 20:
        raise ValueError("max_results 必须在 1 到 20 之间")
    if params is not None and tag is None:
        raise ValueError("传入 params 时必须同时指定 tag")

    api_key = os.getenv("ANYSEARCH_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "缺少 ANYSEARCH_API_KEY，请在 .env 或系统环境变量中配置"
        )

    request_body: dict[str, Any] = {
        "query": query,
        "max_results": max_results,
        "format": "json",
    }
    if tag is not None:
        request_body["tag"] = tag
    if params is not None:
        request_body["params"] = params

    try:
        response = httpx.post(
            ANYSEARCH_SEARCH_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json=request_body,
            timeout=30.0,
        )
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPError as error:
        raise AnySearchError(f"AnySearch 网络请求失败：{error}") from error
    except ValueError as error:
        raise AnySearchError("AnySearch 返回的内容不是合法 JSON") from error

    if payload.get("code") != 0:
        message = payload.get("message", "未知错误")
        request_id = payload.get("request_id", "unknown")
        raise AnySearchError(
            f"AnySearch 搜索失败：{message}（request_id={request_id}）"
        )

    data = payload.get("data")
    if not isinstance(data, dict):
        raise AnySearchError("AnySearch 响应缺少 data")
    results = data.get("results")
    if not isinstance(results, list):
        raise AnySearchError("AnySearch 响应缺少 data.results")

    normalized: list[dict[str, Any]] = []
    for rank, result in enumerate(results, start=1):
        if not isinstance(result, dict):
            continue
        normalized.append(
            {
                "rank": rank,
                "title": str(result.get("title", "")),
                "url": str(result.get("url", "")),
                "snippet": str(result.get("snippet", "")),
                "content": str(result.get("content", "")),
            }
        )
    return normalized


@tool("anysearch_web_search")
def anysearch_web_search(
    query: str,
    tag: str | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    """搜索互联网及垂直数据源；普通搜索不要传 tag，垂直搜索才传 tag 和 params。"""

    return json.dumps(
        search_anysearch(query, tag=tag, params=params),
        ensure_ascii=False,
    )
