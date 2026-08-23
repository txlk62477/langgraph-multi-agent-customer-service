"""使用 Playwright 读取 JavaScript 渲染后的网页内容。"""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Mapping
from contextlib import suppress
import ipaddress
import json
from typing import Any
from urllib.parse import urlparse

from playwright.async_api import Browser, BrowserContext, Response, async_playwright
from playwright.sync_api import (
    Browser as SyncBrowser,
    BrowserContext as SyncBrowserContext,
    Response as SyncResponse,
    sync_playwright,
)


MAX_CONCURRENCY = 3
NAVIGATION_TIMEOUT_MS = 10_000
RENDER_WAIT_MS = 1_000
MAX_RENDERED_TEXT_LENGTH = 8_000
MAX_JSON_RESPONSES = 8
MAX_JSON_BODY_LENGTH = 4_000


def _is_allowed_url(url: str) -> bool:
    """拒绝本机、内网地址和非 HTTP(S) URL，降低服务端请求伪造风险。"""

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False

    hostname = parsed.hostname.lower().rstrip(".")
    if hostname == "localhost" or hostname.endswith(".localhost"):
        return False

    # 如果主机名本身就是 IP，则直接检查是否属于不可访问的内网地址。
    with suppress(ValueError):
        address = ipaddress.ip_address(hostname)
        return not (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_multicast
            or address.is_reserved
            or address.is_unspecified
        )
    return True


def _compact_text(text: str, limit: int) -> str:
    """移除连续空行并限制长度，避免整页内容耗尽大模型上下文。"""

    compact = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}\n……（内容已截断）"


async def _collect_json_response(response: Response) -> dict[str, Any] | None:
    """读取网页加载过程中返回的 JSON，忽略失败响应和非 JSON 内容。"""

    content_type = response.headers.get("content-type", "").lower()
    if not response.ok or "json" not in content_type:
        return None
    if not _is_allowed_url(response.url):
        return None

    try:
        body = await response.text()
    except Exception:
        return None
    if not body.strip():
        return None

    # 验证响应确实是 JSON，避免只依赖错误的 Content-Type。
    try:
        json.loads(body)
    except json.JSONDecodeError:
        return None

    return {
        "url": response.url,
        "status": response.status,
        "body": _compact_text(body, MAX_JSON_BODY_LENGTH),
    }


async def _read_one_page(
    browser: Browser,
    result: Mapping[str, Any],
    *,
    capture_screenshot: bool = False,
) -> dict[str, Any]:
    """打开一条搜索结果，并合并渲染文本、JSON响应及降级信息。"""

    enriched = dict(result)
    enriched.update(
        {
            "browser_status": "failed",
            "rendered_text": "",
            "json_responses": [],
            "browser_error": "",
        }
    )
    if capture_screenshot:
        enriched.update(
            {
                "screenshot_status": "failed",
                "screenshot_mime_type": "image/jpeg",
                "screenshot_base64": "",
                "screenshot_error": "",
            }
        )
    url = str(result.get("url", "")).strip()
    if not _is_allowed_url(url):
        enriched["browser_error"] = "URL为空、格式错误或指向本机/内网地址"
        return enriched

    context: BrowserContext | None = None
    response_tasks: list[asyncio.Task[dict[str, Any] | None]] = []
    try:
        # 每个URL使用独立Context，避免不同网站共享Cookie或本地存储。
        context = await browser.new_context(
            locale="zh-CN",
            viewport={"width": 1440, "height": 1000},
            device_scale_factor=1,
        )
        page = await context.new_page()
        page.set_default_timeout(NAVIGATION_TIMEOUT_MS)

        # 视频和字体不会帮助文本抽取，阻止加载可缩短等待时间。
        async def block_heavy_resources(route: Any) -> None:
            request = route.request
            if request.resource_type in {"media", "font"}:
                await route.abort()
                return
            if not _is_allowed_url(request.url):
                await route.abort()
                return
            await route.continue_()

        await page.route("**/*", block_heavy_resources)

        # Playwright的事件回调不能直接等待，因此把JSON读取任务集中起来。
        def schedule_response(response: Response) -> None:
            task = asyncio.create_task(_collect_json_response(response))
            response_tasks.append(task)

        page.on("response", schedule_response)
        await page.goto(
            url,
            wait_until="domcontentloaded",
            timeout=NAVIGATION_TIMEOUT_MS,
        )
        await page.wait_for_timeout(RENDER_WAIT_MS)

        # 滚动一次以触发常见的懒加载，然后返回顶部再读取完整正文。
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await page.wait_for_timeout(300)
        await page.evaluate("window.scrollTo(0, 0)")

        body = page.locator("body")
        rendered_text = await body.inner_text(timeout=NAVIGATION_TIMEOUT_MS)
        if response_tasks:
            responses = await asyncio.gather(*response_tasks, return_exceptions=True)
        else:
            responses = []

        json_responses = [
            item
            for item in responses
            if isinstance(item, dict)
        ][:MAX_JSON_RESPONSES]
        enriched.update(
            {
                "browser_status": "success",
                "rendered_text": _compact_text(
                    rendered_text,
                    MAX_RENDERED_TEXT_LENGTH,
                ),
                "json_responses": json_responses,
            }
        )
        if capture_screenshot:
            try:
                # 只截取网页顶部可视区域；完整长图会缩小文字并带入大量广告。
                screenshot = await page.screenshot(
                    type="jpeg",
                    quality=80,
                    full_page=False,
                )
                enriched.update(
                    {
                        "screenshot_status": "success",
                        "screenshot_base64": base64.b64encode(screenshot).decode(
                            "ascii"
                        ),
                    }
                )
            except Exception as screenshot_error:
                # 截图失败不影响已经成功取得的文本和 JSON 证据。
                enriched["screenshot_error"] = (
                    f"{type(screenshot_error).__name__}: {screenshot_error}"
                )
    except Exception as error:
        # 浏览器失败不会删除AnySearch原始内容，后续模型仍可使用降级证据。
        enriched["browser_error"] = f"{type(error).__name__}: {error}"
    finally:
        if response_tasks:
            for task in response_tasks:
                task.cancel()
            await asyncio.gather(*response_tasks, return_exceptions=True)
        if context is not None:
            with suppress(Exception):
                await context.close()
    return enriched


def _collect_json_response_sync(
    response: SyncResponse,
) -> dict[str, Any] | None:
    """同步读取网页请求返回的 JSON，规则与异步版本保持一致。"""

    content_type = response.headers.get("content-type", "").lower()
    if not response.ok or "json" not in content_type:
        return None
    if not _is_allowed_url(response.url):
        return None

    try:
        body = response.text()
    except Exception:
        return None
    if not body.strip():
        return None

    try:
        json.loads(body)
    except json.JSONDecodeError:
        return None

    return {
        "url": response.url,
        "status": response.status,
        "body": _compact_text(body, MAX_JSON_BODY_LENGTH),
    }


def _read_one_page_sync(
    browser: SyncBrowser,
    result: Mapping[str, Any],
    *,
    capture_screenshot: bool = False,
) -> dict[str, Any]:
    """同步打开一条搜索结果并提取正文、JSON响应和顶部截图。"""

    enriched = dict(result)
    enriched.update(
        {
            "browser_status": "failed",
            "rendered_text": "",
            "json_responses": [],
            "browser_error": "",
        }
    )
    if capture_screenshot:
        enriched.update(
            {
                "screenshot_status": "failed",
                "screenshot_mime_type": "image/jpeg",
                "screenshot_base64": "",
                "screenshot_error": "",
            }
        )

    url = str(result.get("url", "")).strip()
    if not _is_allowed_url(url):
        enriched["browser_error"] = "URL为空、格式错误或指向本机/内网地址"
        return enriched

    context: SyncBrowserContext | None = None
    json_responses: list[dict[str, Any]] = []
    try:
        # 一个单网页子图只创建一个Context，网站之间不会共享Cookie和本地存储。
        context = browser.new_context(
            locale="zh-CN",
            viewport={"width": 1440, "height": 1000},
            device_scale_factor=1,
        )
        page = context.new_page()
        page.set_default_timeout(NAVIGATION_TIMEOUT_MS)

        def block_heavy_resources(route: Any) -> None:
            request = route.request
            if request.resource_type in {"media", "font"}:
                route.abort()
                return
            if not _is_allowed_url(request.url):
                route.abort()
                return
            route.continue_()

        page.route("**/*", block_heavy_resources)

        def collect_response(response: SyncResponse) -> None:
            if len(json_responses) >= MAX_JSON_RESPONSES:
                return
            item = _collect_json_response_sync(response)
            if item is not None:
                json_responses.append(item)

        page.on("response", collect_response)
        page.goto(
            url,
            wait_until="domcontentloaded",
            timeout=NAVIGATION_TIMEOUT_MS,
        )
        page.wait_for_timeout(RENDER_WAIT_MS)

        # 滚动一次触发常见懒加载，再回到顶部读取正文和截取首屏。
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(300)
        page.evaluate("window.scrollTo(0, 0)")

        rendered_text = page.locator("body").inner_text(
            timeout=NAVIGATION_TIMEOUT_MS
        )
        enriched.update(
            {
                "browser_status": "success",
                "rendered_text": _compact_text(
                    rendered_text,
                    MAX_RENDERED_TEXT_LENGTH,
                ),
                "json_responses": json_responses[:MAX_JSON_RESPONSES],
            }
        )
        if capture_screenshot:
            try:
                screenshot = page.screenshot(
                    type="jpeg",
                    quality=80,
                    full_page=False,
                )
                enriched.update(
                    {
                        "screenshot_status": "success",
                        "screenshot_base64": base64.b64encode(screenshot).decode(
                            "ascii"
                        ),
                    }
                )
            except Exception as screenshot_error:
                enriched["screenshot_error"] = (
                    f"{type(screenshot_error).__name__}: {screenshot_error}"
                )
    except Exception as error:
        enriched["browser_error"] = f"{type(error).__name__}: {error}"
    finally:
        if context is not None:
            with suppress(Exception):
                context.close()
    return enriched


def read_search_result_sync(
    result: Mapping[str, Any],
    *,
    capture_screenshot: bool = True,
) -> dict[str, Any]:
    """同步启动独立Chromium读取一个网页，供正式单网页子图调用。"""

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                return _read_one_page_sync(
                    browser,
                    result,
                    capture_screenshot=capture_screenshot,
                )
            finally:
                browser.close()
    except Exception as error:
        failed = {
            **dict(result),
            "browser_status": "failed",
            "rendered_text": "",
            "json_responses": [],
            "browser_error": f"{type(error).__name__}: {error}",
        }
        if capture_screenshot:
            failed.update(
                {
                    "screenshot_status": "failed",
                    "screenshot_mime_type": "image/jpeg",
                    "screenshot_base64": "",
                    "screenshot_error": f"{type(error).__name__}: {error}",
                }
            )
        return failed


async def read_search_results(
    results: list[dict[str, Any]],
    *,
    max_concurrency: int = MAX_CONCURRENCY,
) -> list[dict[str, Any]]:
    """读取全部搜索结果，并保持原始顺序和结果数量不变。"""

    if not results:
        return []
    if max_concurrency < 1:
        raise ValueError("max_concurrency 必须大于等于 1")

    semaphore = asyncio.Semaphore(max_concurrency)
    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True)

            async def read_with_limit(result: dict[str, Any]) -> dict[str, Any]:
                async with semaphore:
                    return await _read_one_page(browser, result)

            # gather会并行执行，但返回顺序仍与AnySearch结果顺序一致。
            enriched = await asyncio.gather(
                *(read_with_limit(result) for result in results)
            )
            await browser.close()
            return enriched
    except Exception as error:
        # Chromium未安装或启动失败时，全部结果统一降级为AnySearch原文。
        message = f"{type(error).__name__}: {error}"
        return [
            {
                **result,
                "browser_status": "failed",
                "rendered_text": "",
                "json_responses": [],
                "browser_error": message,
            }
            for result in results
        ]
