"""联网搜索、单网页读取、视觉识别和最终答案节点。"""

from __future__ import annotations

import base64
from collections.abc import Callable
from functools import partial
import json
import os
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4
from threading import BoundedSemaphore

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Overwrite

from agent.common.anysearch import anysearch_web_search
from agent.common.browser_reader import read_search_result_sync
from agent.common.llm import build_chat_model
from agent.common.vision import get_vision_client
from agent.state.web_search import WebPageState, WebSearchState


ScreenshotRetention = Literal["none", "disk", "state"]
VisionFactory = Callable[[], Any]
PageReader = Callable[..., dict[str, Any]]
VisionWarmup = Callable[[], None]


def _latest_user_question(state: WebSearchState) -> str:
    """从消息历史中取得最近一条非空的用户问题。"""

    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            content = message.content
            if isinstance(content, str) and content.strip():
                return content.strip()
    raise ValueError("没有找到可搜索的用户问题")


def search_web(
    state: WebSearchState,
    *,
    vision_warmup: VisionWarmup | None = None,
) -> dict[str, Any]:
    """预热视觉模型后调用 AnySearch，并清空上一轮结果。"""

    # 外部主图可以传入结合上下文补全后的search_query；单独使用本子图时，
    # 则从最新HumanMessage提取。不能读取旧query，避免同一Thread复用上轮搜索词。
    planned_query = state.get("search_query", "")
    query = (
        planned_query.strip()
        if isinstance(planned_query, str) and planned_query.strip()
        else _latest_user_question(state)
    )
    reset = {
        "query": query,
        "browser_results": [],
        # page_results 使用列表加法 reducer，Overwrite 用于开始新一轮时彻底清空。
        "page_results": Overwrite(value=[]),
    }
    vision_error = ""
    try:
        warm_up = vision_warmup or (lambda: get_vision_client().warm_up())
        warm_up()
    except Exception as error:
        # 视觉识别是增强能力，不是搜索硬依赖。继续调用 AnySearch，并在
        # 每条结果上标记跳过视觉，避免页面子图重复尝试同一个不可用模型。
        vision_error = f"{type(error).__name__}: {error}"
    try:
        # 使用工具的 invoke 接口，使搜索作为独立 Tool Run 出现在 LangSmith 中。
        raw_results = anysearch_web_search.invoke({"query": query})
        search_results = json.loads(raw_results)
    except Exception as error:
        return {
            **reset,
            "search_results": [],
            "search_error": f"AnySearch 搜索失败：{type(error).__name__}: {error}",
            "vision_error": vision_error,
        }
    if not search_results:
        return {
            **reset,
            "search_results": [],
            "search_error": "AnySearch 没有返回搜索结果",
            "vision_error": vision_error,
        }
    if vision_error:
        search_results = [
            {**result, "vision_unavailable": True}
            for result in search_results
        ]
    return {
        **reset,
        "search_results": search_results,
        "search_error": "",
        "vision_error": vision_error,
    }


class WebPageProcessingNodes:
    """单网页子图节点，并分别限制浏览器与视觉模型并发。"""

    def __init__(
        self,
        *,
        page_reader: PageReader = read_search_result_sync,
        vision_factory: VisionFactory = get_vision_client,
        browser_max_concurrency: int = 3,
        vision_max_concurrency: int = 2,
        screenshot_retention: ScreenshotRetention = "none",
        screenshot_dir: str = "artifacts/web_screenshots",
    ) -> None:
        if browser_max_concurrency <= 0:
            raise ValueError("WEB_BROWSER_MAX_CONCURRENCY 必须大于 0")
        if vision_max_concurrency <= 0:
            raise ValueError("WEB_VISION_MAX_CONCURRENCY 必须大于 0")
        if screenshot_retention not in {"none", "disk", "state"}:
            raise ValueError(
                "WEB_SCREENSHOT_RETENTION 只能是 none、disk 或 state"
            )
        if screenshot_retention == "disk" and not screenshot_dir.strip():
            raise ValueError("磁盘保留截图时 WEB_SCREENSHOT_DIR 不能为空")

        self._page_reader = page_reader
        self._vision_factory = vision_factory
        self._screenshot_retention = screenshot_retention
        self._screenshot_dir = Path(screenshot_dir)
        # 动态Send会在线程池中并行运行单网页子图，两把独立信号量分别限制
        # Chromium实例数量和同时发送给Windows Ollama的图片请求数量。
        self._browser_gate = BoundedSemaphore(browser_max_concurrency)
        self._vision_gate = BoundedSemaphore(vision_max_concurrency)

    def warm_up_vision_model(self) -> None:
        """预热页面视觉分析将使用的模型。"""

        self._vision_factory().warm_up()

    def playwright_read_page(
        self,
        state: WebPageState,
    ) -> dict[str, dict[str, Any]]:
        """读取一个网页并截取顶部1440×1000的JPEG可视区域。"""

        with self._browser_gate:
            # 单网页子图本身是同步节点；不同网页由LangGraph在线程间并发。
            result = self._page_reader(
                state["page"],
                capture_screenshot=True,
            )

        if (
            self._screenshot_retention == "disk"
            and result.get("screenshot_base64")
        ):
            try:
                result["screenshot_path"] = self._save_screenshot(result)
            except Exception as error:
                # 保存调试图片失败不影响视觉模型使用内存中的截图继续识别。
                result["screenshot_retention_error"] = (
                    f"{type(error).__name__}: {error}"
                )
        return {"page_result": result}

    def analyze_page_visuals(
        self,
        state: WebPageState,
    ) -> dict[str, list[dict[str, Any]]]:
        """识别截图中的问题相关证据，并清理不需要保留的图片数据。"""

        result = dict(state.get("page_result", state["page"]))
        screenshot = str(result.get("screenshot_base64", ""))
        result.update(
            {
                "vision_status": "skipped",
                "vision_relevant": False,
                "vision_description": "",
                "vision_error": "",
            }
        )

        if result.get("vision_unavailable"):
            result["vision_error"] = "视觉模型不可用，已跳过视觉识别"
        elif screenshot and result.get("screenshot_status") == "success":
            try:
                with self._vision_gate:
                    evidence = self._vision_factory().analyze_webpage(
                        query=state["query"],
                        title=str(result.get("title", "")),
                        url=str(result.get("url", "")),
                        screenshot_base64=screenshot,
                    )
                result.update(
                    {
                        "vision_status": "success",
                        "vision_relevant": evidence.relevant,
                        "vision_description": evidence.as_text(),
                    }
                )
            except Exception as error:
                # 视觉识别是补充证据，失败时仍保留网页文本和JSON继续汇总。
                result.update(
                    {
                        "vision_status": "failed",
                        "vision_error": f"{type(error).__name__}: {error}",
                    }
                )
        elif result.get("screenshot_error"):
            result["vision_error"] = "没有可识别截图：" + str(
                result["screenshot_error"]
            )

        # 默认和disk模式都不把大体积Base64带回父图；state模式用于显式调试。
        if self._screenshot_retention != "state":
            result.pop("screenshot_base64", None)
        return {"page_results": [result]}

    def _save_screenshot(self, result: dict[str, Any]) -> str:
        """在显式disk模式下保存JPEG，并返回绝对路径。"""

        image = base64.b64decode(result["screenshot_base64"], validate=True)
        self._screenshot_dir.mkdir(parents=True, exist_ok=True)
        rank = int(result.get("rank", 0))
        path = self._screenshot_dir / f"web_{rank}_{uuid4().hex}.jpg"
        path.write_bytes(image)
        return str(path.resolve())



def build_webpage_processing_nodes() -> WebPageProcessingNodes:
    """根据环境变量构建正式的单网页处理节点集合。"""

    retention = os.getenv("WEB_SCREENSHOT_RETENTION", "none").strip().lower()
    return WebPageProcessingNodes(
        browser_max_concurrency=int(
            os.getenv("WEB_BROWSER_MAX_CONCURRENCY", "3")
        ),
        vision_max_concurrency=int(os.getenv("WEB_VISION_MAX_CONCURRENCY", "2")),
        screenshot_retention=retention,  # type: ignore[arg-type]
        screenshot_dir=os.getenv(
            "WEB_SCREENSHOT_DIR",
            "artifacts/web_screenshots",
        ),
    )


def build_search_web_node(
    page_nodes: WebPageProcessingNodes,
) -> Callable[[WebSearchState], dict[str, Any]]:
    """构建与页面处理节点共享视觉客户端的 AnySearch 节点。"""

    return partial(
        search_web,
        vision_warmup=page_nodes.warm_up_vision_model,
    )


def finalize_webpages(state: WebSearchState) -> dict[str, Any]:
    """排序并行网页结果，判断是否存在可用于最终回答的浏览器证据。"""

    results = sorted(
        state.get("page_results", []),
        key=lambda item: int(item.get("rank", 0)),
    )
    usable_results = [
        result
        for result in results
        if str(result.get("rendered_text", "")).strip()
        or result.get("json_responses")
        or str(result.get("content", "")).strip()
        or str(result.get("snippet", "")).strip()
        or (
            result.get("vision_status") == "success"
            and result.get("vision_relevant")
            and str(result.get("vision_description", "")).strip()
        )
    ]
    return {
        "browser_results": results,
        # 聚合已经结束，清空临时reducer，避免检查点重复保存同一批证据。
        "page_results": Overwrite(value=[]),
        "search_error": (
            "所有网页的Playwright文本、JSON和视觉证据均不可用"
            if not usable_results
            else ""
        ),
    }


def generate_answer(state: WebSearchState) -> dict[str, list]:
    """让 DeepSeek 根据文本、JSON和视觉证据生成一个带来源的综合答案。"""

    # 即使显式选择state保留截图，也不能把大体积Base64再次发送给DeepSeek。
    answer_sources = [
        {
            key: value
            for key, value in result.items()
            if key != "screenshot_base64"
        }
        for result in state.get(
            "browser_results",
            state.get("search_results", []),
        )
    ]
    sources = json.dumps(
        answer_sources,
        ensure_ascii=False,
        indent=2,
    )
    try:
        response = build_chat_model().invoke(
            [
                SystemMessage(
                    content=(
                        "你是联网搜索问答助手，只能根据提供的搜索结果回答。"
                        "证据优先级为：第一，网页加载得到的 json_responses；第二，视觉模型"
                        "从截图提取的 vision_description，尤其是动态时间、价格、状态、"
                        "图表和Canvas内容；第三，Playwright提取的 rendered_text；第四，"
                        "AnySearch的 content 和 snippet。"
                        "JSON与视觉结果一致时可视为强证据。视觉动态值与静态正文冲突时，"
                        "可以提高视觉结果权重，但要考虑OCR误读；视觉与JSON冲突时不能"
                        "直接猜测，应结合其他来源交叉验证。只有单张截图提供关键数字且"
                        "无法验证时，要明确说明这是页面显示值。"
                        "同一网页中，正文前半部分通常更接近主要答案，应优先提取其中与"
                        "用户问题直接相关的内容；正文后半部分常包含导航、广告、推荐和"
                        "其他功能，应降低参考权重，但不能仅因位置靠后就完全忽略。"
                        "每次只给出一个综合结论，不要按不同来源返回多个答案。"
                        "如果多个来源的关键事实冲突严重且无法判断，不要猜测，也不要给出"
                        "具体答案，只说明暂时无法确认并提供参考来源。"
                        "无法确认的信息要明确说明。回答中的事实使用 [1]、[2] 等编号标注"
                        "来源，最后列出对应的标题和 URL。"
                    )
                ),
                HumanMessage(
                    content=(
                        f"用户问题：{state['query']}\n\n"
                        f"搜索结果：\n{sources}"
                    )
                ),
            ]
        )
    except Exception:
        # 搜索证据已经保留，最终总结模型失败时仍返回明确的服务降级提示。
        response = AIMessage(
            content="抱歉，搜索结果已获取，但当前总结服务暂时不可用，请稍后重试。"
        )
    return {"messages": [response]}
