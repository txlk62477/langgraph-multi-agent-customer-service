"""常规问答 Agent 可自主组合的联网研究工具。"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
import re
from typing import Annotated, Any

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import Field

from agent.common.anysearch import search_anysearch
from agent.common.browser_reader import (
    MAX_CONCURRENCY,
    read_search_result_sync,
    read_search_results_sync,
)
from agent.common.vision import OllamaVisionClient, get_vision_client
from agent.tools.runtime import SelectionReason, SpecialistContext, json_result


ANYSEARCH_CALL_LIMIT = 3
PLAYWRIGHT_CALL_LIMIT = 4
PLAYWRIGHT_BATCH_LIMIT = 4
VISION_CALL_LIMIT = 2
_URL_PATTERN = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
_URL_TRAILING_PUNCTUATION = ".,;:!?，。；：！？、）)]}"
_ACCESS_CHALLENGE_PATTERN = re.compile(
    r"(?:人机验证|安全验证|验证码|访问过于频繁|访问受限|access denied|"
    r"verify (?:you are|that you are) human|captcha|cloudflare ray id)",
    re.IGNORECASE,
)

Search = Callable[..., list[dict[str, Any]]]
PageReader = Callable[..., dict[str, Any]]
PagesReader = Callable[..., list[dict[str, Any]]]
VisionFactory = Callable[[], OllamaVisionClient]
PageUrls = Annotated[
    list[str],
    Field(
        min_length=1,
        max_length=PLAYWRIGHT_BATCH_LIMIT,
        description="本次需要读取的1到4个已批准网页URL，多个来源应合并在一次调用中",
    ),
]


def _messages(runtime: ToolRuntime[SpecialistContext]) -> Sequence[Any]:
    state = runtime.state
    if not isinstance(state, Mapping):
        return ()
    messages = state.get("messages", ())
    return messages if isinstance(messages, Sequence) else ()


def _current_turn_messages(
    runtime: ToolRuntime[SpecialistContext],
) -> Sequence[Any]:
    """只返回最新用户消息开始后的状态，确保预算与URL授权按轮隔离。"""

    messages = _messages(runtime)
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return messages[index:]
    return messages


def _budget_reached(
    runtime: ToolRuntime[SpecialistContext],
    tool_name: str,
    limit: int,
) -> bool:
    """计算已完成调用和当前并行批次序号，可靠限制每轮工具次数。"""

    messages = _current_turn_messages(runtime)
    completed = sum(
        isinstance(message, ToolMessage) and message.name == tool_name
        for message in messages
    )
    current_position = 1
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        calls = [
            call
            for call in message.tool_calls
            if call.get("name") == tool_name
        ]
        for index, call in enumerate(calls, start=1):
            if call.get("id") == runtime.tool_call_id:
                current_position = index
                break
        break
    return completed + current_position > limit


def _extract_urls(value: Any) -> set[str]:
    if isinstance(value, str):
        return {
            match.rstrip(_URL_TRAILING_PUNCTUATION)
            for match in _URL_PATTERN.findall(value)
        }
    if isinstance(value, Sequence):
        urls: set[str] = set()
        for item in value:
            urls.update(_extract_urls(item))
        return urls
    if isinstance(value, Mapping):
        urls: set[str] = set()
        for item in value.values():
            urls.update(_extract_urls(item))
        return urls
    return set()


def _approved_urls(runtime: ToolRuntime[SpecialistContext]) -> set[str]:
    """收集用户明确提供以及本轮搜索工具实际返回的 URL。"""

    approved: set[str] = set()
    for message in _current_turn_messages(runtime):
        if isinstance(message, HumanMessage):
            approved.update(_extract_urls(message.content))
            continue
        if not isinstance(message, ToolMessage) or message.name != "anysearch_search":
            continue
        content = message.content
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        for result in payload.get("results", []):
            if isinstance(result, Mapping):
                url = str(result.get("url", "")).strip()
                if url:
                    approved.add(url)
    return approved


def _reject_unapproved_url(
    url: str,
    runtime: ToolRuntime[SpecialistContext],
) -> str | None:
    cleaned = url.strip()
    if not cleaned:
        return json_result(status="rejected", error="URL不能为空")
    if cleaned not in _approved_urls(runtime):
        return json_result(
            status="rejected",
            error="只能访问本轮AnySearch返回或用户明确提供的URL；请先搜索来源",
        )
    return None


def _page_access_error(result: Mapping[str, Any]) -> tuple[int | None, str]:
    """识别浏览器成功渲染但没有取得真实页面内容的访问拦截。"""

    raw_http_status = result.get("http_status")
    http_status = int(raw_http_status) if isinstance(raw_http_status, int) else None
    title = str(result.get("title", ""))
    rendered_text = str(result.get("rendered_text", ""))
    if _ACCESS_CHALLENGE_PATTERN.search(f"{title}\n{rendered_text[:2_000]}"):
        return http_status, "网页要求人机验证，未取得可用正文"
    if http_status is not None and http_status >= 400:
        return http_status, f"网页返回 HTTP {http_status}"
    if result.get("browser_status") != "success":
        return http_status, str(result.get("browser_error") or "网页读取失败")
    return http_status, ""


def build_anysearch_search_tool(*, search: Search = search_anysearch):
    """创建只返回候选来源摘要的搜索工具。"""

    @tool("anysearch_search")
    def anysearch_search(
        query: str,
        selection_reason: SelectionReason,
        max_results: int = 5,
        runtime: ToolRuntime[SpecialistContext] = None,
    ) -> str:
        """搜索候选网页；返回标题、URL和摘要，不读取网页正文。"""

        del selection_reason
        if runtime is None:
            return json_result(status="failed", error="缺少工具运行时")
        if _budget_reached(runtime, "anysearch_search", ANYSEARCH_CALL_LIMIT):
            return json_result(
                status="limit_reached",
                error=f"本轮搜索次数已达到上限{ANYSEARCH_CALL_LIMIT}",
            )
        cleaned = query.strip()
        if not cleaned:
            return json_result(status="failed", error="搜索词不能为空")
        if not 1 <= int(max_results) <= 10:
            return json_result(status="failed", error="max_results必须在1到10之间")
        try:
            results = search(cleaned, max_results=int(max_results))
        except Exception as error:
            return json_result(
                status="failed",
                results=[],
                error=f"{type(error).__name__}: {error}",
            )
        compact_results = [
            {
                "rank": int(result.get("rank", index)),
                "title": str(result.get("title", "")),
                "url": str(result.get("url", "")),
                "snippet": str(result.get("snippet", "")),
            }
            for index, result in enumerate(results, start=1)
            if isinstance(result, Mapping) and str(result.get("url", "")).strip()
        ]
        return json_result(
            status="success" if compact_results else "empty",
            results=compact_results,
        )

    return anysearch_search


def build_playwright_read_page_tool(
    *,
    pages_reader: PagesReader = read_search_results_sync,
):
    """创建由 Agent 一次选择多个 URL 的动态网页文本读取工具。"""

    @tool("playwright_read_page")
    def playwright_read_page(
        urls: PageUrls,
        selection_reason: SelectionReason,
        runtime: ToolRuntime[SpecialistContext],
    ) -> str:
        """批量读取已批准URL的渲染正文和JSON响应；不进行视觉分析。"""

        del selection_reason
        if _budget_reached(runtime, "playwright_read_page", PLAYWRIGHT_CALL_LIMIT):
            return json_result(
                status="limit_reached",
                error=f"本轮网页读取次数已达到上限{PLAYWRIGHT_CALL_LIMIT}",
            )
        # 先标准化并按首次出现顺序去重，避免同一页面重复占用浏览器和模型上下文。
        cleaned_urls = list(dict.fromkeys(url.strip() for url in urls))
        if not cleaned_urls:
            return json_result(status="failed", pages=[], error="URL列表不能为空")
        if len(cleaned_urls) > PLAYWRIGHT_BATCH_LIMIT:
            return json_result(
                status="failed",
                pages=[],
                error=f"单次最多{PLAYWRIGHT_BATCH_LIMIT}个不同URL",
            )

        # 逐项预留输出位置，只把用户明确提供或本轮搜索返回的URL送入浏览器。
        # 未授权项留在原位置返回 rejected，不影响同批其他合法页面。
        approved_urls = _approved_urls(runtime)
        pages: list[dict[str, Any] | None] = []
        readable_urls: list[str] = []
        readable_indexes: list[int] = []
        for url in cleaned_urls:
            if not url:
                pages.append(
                    {"status": "rejected", "url": url, "error": "URL不能为空"}
                )
                continue
            if url not in approved_urls:
                pages.append(
                    {
                        "status": "rejected",
                        "url": url,
                        "error": (
                            "只能访问本轮AnySearch返回或用户明确提供的URL；"
                            "请先搜索来源"
                        ),
                    }
                )
                continue
            readable_indexes.append(len(pages))
            readable_urls.append(url)
            pages.append(None)
        try:
            # 底层一次启动一个Chromium，并在独立Context中并发读取合法页面。
            results = (
                pages_reader(
                    [{"title": "", "url": url} for url in readable_urls],
                    max_concurrency=MAX_CONCURRENCY,
                )
                if readable_urls
                else []
            )
        except Exception as error:
            return json_result(
                status="failed",
                pages=[],
                error=f"{type(error).__name__}: {error}",
            )
        # 将浏览器结果填回预留位置，保证最终pages顺序与去重后的输入一致。
        for result_index, (index, url) in enumerate(
            zip(readable_indexes, readable_urls, strict=True)
        ):
            if result_index >= len(results):
                pages[index] = {
                    "status": "failed",
                    "title": "",
                    "url": url,
                    "http_status": None,
                    "rendered_text": "",
                    "json_responses": [],
                    "error": "批量网页读取器未返回该URL的结果",
                }
                continue
            result = results[result_index]
            http_status, access_error = _page_access_error(result)
            pages[index] = {
                "status": "success" if not access_error else "failed",
                "title": str(result.get("title", "")),
                "url": url,
                "http_status": http_status,
                "rendered_text": (
                    str(result.get("rendered_text", ""))
                    if not access_error
                    else ""
                ),
                "json_responses": result.get("json_responses", []),
                "error": str(result.get("browser_error", "")) or access_error,
            }
        resolved_pages = [page for page in pages if page is not None]
        success_count = sum(
            page["status"] == "success" for page in resolved_pages
        )
        return json_result(
            status=(
                "success"
                if success_count == len(resolved_pages)
                else "partial_success" if success_count else "failed"
            ),
            pages=resolved_pages,
        )

    return playwright_read_page


def build_analyze_page_visuals_tool(
    *,
    page_reader: PageReader = read_search_result_sync,
    vision_factory: VisionFactory = get_vision_client,
):
    """创建按需打开网页、截图并提取视觉证据的工具。"""

    @tool("analyze_page_visuals")
    def analyze_page_visuals(
        url: str,
        question: str,
        selection_reason: SelectionReason,
        runtime: ToolRuntime[SpecialistContext],
    ) -> str:
        """分析已批准URL顶部截图中的图表、价格、时间、状态等可见事实。"""

        del selection_reason
        if _budget_reached(runtime, "analyze_page_visuals", VISION_CALL_LIMIT):
            return json_result(
                status="limit_reached",
                error=f"本轮视觉分析次数已达到上限{VISION_CALL_LIMIT}",
            )
        rejected = _reject_unapproved_url(url, runtime)
        if rejected is not None:
            return rejected
        cleaned_url = url.strip()
        cleaned_question = question.strip()
        if not cleaned_question:
            return json_result(status="failed", error="视觉分析问题不能为空")
        try:
            page = page_reader(
                {"title": "", "url": cleaned_url},
                capture_screenshot=True,
            )
            http_status, access_error = _page_access_error(page)
            if access_error:
                return json_result(
                    status="failed",
                    url=cleaned_url,
                    http_status=http_status,
                    error=access_error,
                )
            screenshot = str(page.get("screenshot_base64", ""))
            if page.get("screenshot_status") != "success" or not screenshot:
                return json_result(
                    status="failed",
                    url=cleaned_url,
                    error=str(
                        page.get("screenshot_error")
                        or page.get("browser_error")
                        or "网页截图不可用"
                    ),
                )
            evidence = vision_factory().analyze_webpage(
                query=cleaned_question,
                title=str(page.get("title", "")),
                url=cleaned_url,
                screenshot_base64=screenshot,
            )
        except Exception as error:
            return json_result(
                status="failed",
                url=cleaned_url,
                error=f"{type(error).__name__}: {error}",
            )
        return json_result(
            status="success",
            title=str(page.get("title", "")),
            url=cleaned_url,
            relevant=evidence.relevant,
            description=evidence.description,
            visible_facts=evidence.visible_facts,
            uncertainties=evidence.uncertainties,
        )

    return analyze_page_visuals


def build_general_qa_search_tools():
    """返回 General QA Agent 可自主编排的全部联网工具。"""

    return [
        build_anysearch_search_tool(),
        build_playwright_read_page_tool(),
        build_analyze_page_visuals_tool(),
    ]
