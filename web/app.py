"""简单 Python 后端：托管前端页面，并把聊天请求转发给 LangGraph Agent Server。

架构（相比纯静态 HTML 的简化点）：
    浏览器(8080)  --POST /api/chat-->  本服务  --REST-->  langgraph dev(2024)
    - 页面和接口同源，浏览器没有跨域问题；
    - thread_id、assistant、interrupt/resume 等协议细节都在这处理，
      前端 JS 只剩"发消息、显示回复、新会话"三件事。

用法：
    终端 1（启动 AI 大脑）：
        cd 项目目录
        source .venv/bin/activate
        langgraph dev
    终端 2（启动页面 + 聊天接口）：
        python3 web/app.py
    浏览器打开：http://127.0.0.1:8080

环境变量（也可写在项目根 .env 里）：
    LANGGRAPH_URL   LangGraph Agent Server 地址，默认 http://127.0.0.1:2024
    CHAT_USER_ID    用户 ID，默认 lk
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent
PROJECT_DIR = WEB_DIR.parent
RUN_TIMEOUT = 180  # 一次对话最多等待秒数（联网搜索可能较慢）


def _env(key: str, default: str) -> str:
    """优先读进程环境变量，其次读项目根 .env，最后用默认值。"""
    value = os.getenv(key)
    if value:
        return value.strip()
    try:
        for line in (PROJECT_DIR / ".env").read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(key + "="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return default


LANGGRAPH_URL = _env("LANGGRAPH_URL", "http://127.0.0.1:2024").rstrip("/")
ASSISTANT_ID = "customer_service"  # langgraph.json 里注册的主图名
USER_ID = _env("CHAT_USER_ID", "lk")
PORT = int(_env("PORT", "8080"))
DEBUG_GRAPH = os.getenv("DEBUG_GRAPH", "1") != "0"  # 图执行调试日志开关

# 只有 Supervisor 的模型输出是最终给用户的回答。专业 Agent 的中间结论
# 会返回共享上下文，但不直接泄露到页面。
FINAL_ANSWER_NODES = {
    "agent",
    "model",
}

# 页面只显示已确认的业务节点。子图包装节点、SQL 内部节点和
# 后台偏好写入节点不放入白名单，避免流程冗余和技术细节泄露。
PROGRESS_LABELS = {
    "agent": "专业 Agent 正在处理",
    "model": "专业 Agent 正在处理",
    "load_preferences": "读取您的租房偏好",
    "anysearch_search": "搜索相关网页",
    "playwright_read_page": "读取网页动态内容",
    "analyze_page_visuals": "分析网页可视信息",
    "get_rental_preferences": "读取租房偏好",
    "inspect_rental_market": "了解租房市场",
    "search_houses": "搜索匹配房源",
    "get_house_details": "读取房源详情",
    "find_bookable_houses": "查找预订房源",
    "check_booking_availability": "检查房源档期",
    "create_booking": "创建预订订单",
    "list_recent_orders": "查询最近订单",
    "search_orders": "筛选历史订单",
    "get_order_details": "读取订单详情",
    "find_cancellable_orders": "查找可取消订单",
    "check_cancellation_eligibility": "检查取消资格",
    "cancel_order": "确认并取消订单",
    "request_user_input": "等待您补充或确认",
}

PROGRESS_DESCRIPTIONS = {
    "agent": "正在理解当前任务并选择合适的业务工具。",
    "model": "正在理解当前任务并选择合适的业务工具。",
    "load_preferences": "正在读取跨会话保存的城市、区域、预算和房型。",
    "anysearch_search": "正在按当前问题搜索相关网页。",
    "playwright_read_page": "正在加载网页并提取动态渲染后的内容。",
    "analyze_page_visuals": "正在识别网页截图中的时间、价格和状态等信息。",
    "get_rental_preferences": "正在读取跨会话保存的租房偏好。",
    "inspect_rental_market": "正在查看真实区域、房源数量和租金范围。",
    "search_houses": "正在按已确认条件查询真实房源。",
    "get_house_details": "正在读取选定房源的地址、楼层和设施。",
    "find_bookable_houses": "正在按名称或小区查找明确候选。",
    "check_booking_availability": "正在检查目标日期是否存在预订冲突。",
    "create_booking": "正在事务中重新校验并创建订单。",
    "list_recent_orders": "正在查询当前用户最近的订单。",
    "search_orders": "正在按房源、状态或日期筛选当前用户订单。",
    "get_order_details": "正在读取当前用户的一笔明确订单。",
    "find_cancellable_orders": "正在查找当前用户尚未入住的订单。",
    "check_cancellation_eligibility": "正在检查订单状态、入住时间和取消资格。",
    "cancel_order": "等待您选择并确认，然后安全取消订单。",
    "request_user_input": "需要您补充信息或确认后才能继续。",
}

# create_agent 内部节点都叫 agent/model，必须结合 SSE 子图命名空间才能
# 判断当前真正运行的是哪个专业 Agent。canonical name 同时作为前端步骤 ID，
# 避免一轮中的多个专业 Agent 被通用节点名错误合并。
SPECIALIST_PROGRESS_LABELS = {
    "general_qa_agent": "一般问答 Agent 正在处理",
    "rental_recommendation_agent": "房源推荐 Agent 正在处理",
    "rental_booking_agent": "房源预订 Agent 正在处理",
    "order_history_agent": "历史订单 Agent 正在处理",
    "order_cancellation_agent": "订单取消 Agent 正在处理",
}
SPECIALIST_INTERNAL_NODES = frozenset({"agent", "model"})


def _log(*parts: object) -> None:
    """图执行调试日志；DEBUG_GRAPH=0 时静默。"""
    if DEBUG_GRAPH:
        print(*parts, flush=True)


def post_json(url: str, payload: dict, timeout: int = 30) -> dict | list:
    """向 LangGraph Agent Server 发送 POST 并解析 JSON。"""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_json(url: str, timeout: int = 30) -> dict:
    """向 LangGraph Agent Server 发送 GET 并解析 JSON。"""
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _summarize(obj: object, limit: int = 300) -> str:
    """把节点输出转成可读文本并截断，避免刷屏。"""
    try:
        text = json.dumps(obj, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(obj)
    if len(text) > limit:
        return text[:limit] + f"…（共 {len(text)} 字符）"
    return text


def _handle_stream_event(event_name: str, raw: str) -> None:
    """解析一个 SSE 事件，并打印主图或子图的当前节点。"""

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return

    # 开启 stream_subgraphs 后，event 名称形如
    # updates|recommend_rental:任务ID|tools:任务ID。
    parts = event_name.split("|") if event_name else ["message"]
    event_type = parts[0]
    namespace = [part.split(":", 1)[0] for part in parts[1:] if part]

    if event_type == "metadata" and isinstance(data, dict):
        _log(f"[图] 运行: {str(data.get('run_id', ''))[:8]}")
    elif event_type == "updates" and isinstance(data, dict):
        # updates 会经过当前图的公共输出 Schema 过滤。内部字段全部被隐藏时，
        # Agent Server 会发送 {node: null}，因此这里只处理不会被 tasks 替代的中断。
        for item in data.get("__interrupt__", []) or []:
            value = item.get("value") if isinstance(item, dict) else item
            # 中断内容需要完整展示，不能使用 300 字摘要。
            detail = json.dumps(value, ensure_ascii=False, default=str)
            _log(f"[图]   ◀── 中断: {detail}")
    elif event_type == "tasks":
        # tasks 是节点真实的生命周期事件：开始事件带 input，完成事件带
        # result/error，结果不会因为父图或子图的公共输出 Schema 变成 null。
        tasks = data if isinstance(data, list) else [data]
        for task in tasks:
            if not isinstance(task, dict):
                continue
            node = task.get("name")
            if not isinstance(node, str) or not node:
                continue
            node_path = " / ".join([*namespace, node])
            if "input" in task:
                _log(f"[图] ──▶ {node_path}")
            if task.get("error"):
                _log(f"[图]     错误: {_summarize(task['error'])}")
            elif "result" in task:
                _log(f"[图]     结果: {_summarize(task['result'])}")
    elif event_type == "error":
        _log(f"[图] 执行错误: {_summarize(data)}")


def _iter_agent_events(
    url: str,
    payload: dict,
    timeout: int = RUN_TIMEOUT,
) -> Iterator[tuple[str, object]]:
    """运行 Agent Server SSE 接口，逐个产出已解析的事件。"""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    _log("[客服] 开始运行主图 …")
    event_name = "message"
    data_lines: list[str] = []

    def take_event() -> tuple[str, object] | None:
        nonlocal event_name, data_lines
        event = None
        if data_lines:
            raw = "\n".join(data_lines)
            _handle_stream_event(event_name, raw)
            try:
                event = (event_name, json.loads(raw))
            except json.JSONDecodeError:
                pass
        event_name = "message"
        data_lines = []
        return event

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            if line.startswith("event:"):
                event_name = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())
            elif not line:
                event = take_event()
                if event is not None:
                    yield event
        # 兼容最后一个事件没有以空行结束的响应。
        event = take_event()
        if event is not None:
            yield event


def _thread_interrupts(state: dict) -> list:
    """从线程状态响应里提取中断列表。

    新版 Agent Server：中断数据在响应顶层 `interrupts`（元素是
    {id, value, resumable, ns, when}，value 才是业务 payload）；
    旧版/部分接口：在 `values.__interrupt__`。两种都兼容。
    """
    values = state.get("values") or {}
    return (
        state.get("interrupts")
        or values.get("__interrupt__")
        or state.get("__interrupt__")
        or []
    )


def _event_type(event_name: str) -> str:
    """取得带子图命名空间的 SSE 事件类型。"""

    return event_name.split("|", 1)[0]


def _event_namespace(event_name: str) -> list[str]:
    """取得 SSE 事件经过的主图和子图节点名称。"""

    return [
        part.split(":", 1)[0]
        for part in event_name.split("|")[1:]
        if part
    ]


def _event_specialist(event_name: str) -> str | None:
    """从嵌套事件路径中解析当前专业 Agent，排除 Supervisor 内部模型。"""

    for component in reversed(_event_namespace(event_name)):
        for specialist in SPECIALIST_PROGRESS_LABELS:
            if component == specialist or component.endswith(f"_{specialist}"):
                return specialist
    return None


def _message_text(message: object) -> str:
    """从 Agent Server 的消息块中提取纯文本 token。"""

    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") in {"text", "output_text"}:
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _message_event_parts(data: object) -> tuple[object, dict]:
    """兼容 messages/messages-tuple 的 ``[消息块, 元数据]`` 结构。"""

    if isinstance(data, (list, tuple)) and len(data) >= 2 and isinstance(data[1], dict):
        return data[0], data[1]
    return {}, {}


def _interrupt_payload(data: object) -> object | None:
    """从 updates 事件中提取第一个中断的业务数据。"""

    if not isinstance(data, dict):
        return None
    interrupts = data.get("__interrupt__") or []
    if not interrupts:
        return None
    entry = interrupts[0]
    if isinstance(entry, dict) and "value" in entry:
        return entry["value"]
    return entry


def _interrupt_text(payload: object) -> str:
    """把中断数据转成用户可读提示。"""

    if isinstance(payload, dict):
        message = payload.get("message")
        if isinstance(message, str):
            return message.strip()
        return json.dumps(payload, ensure_ascii=False, default=str)
    return str(payload).strip() if payload is not None else ""


def _new_ai_reply(messages: list, previous_count: int) -> str:
    """只读取本次运行新增的 AI 消息，避免误返回上一轮答案。"""

    for msg in reversed(messages[previous_count:]):
        if not isinstance(msg, dict) or msg.get("type") != "ai":
            continue
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    return ""


def _fallback_reply(values: dict, interrupt_data: object | None = None) -> str:
    """生成信息缺失或无回复时的用户可读文本。"""

    missing = None
    if isinstance(interrupt_data, dict):
        missing = interrupt_data.get("missing_required_fields")
    if not missing:
        missing = values.get("missing_required_fields") or []
    if missing:
        labels = {
            "city": "租房城市", "budget_min": "最低预算",
            "budget_max": "最高预算", "districts": "区域",
            "room_types": "房型", "rental_mode": "租赁方式",
            "phone": "手机号", "house_title": "房源名称",
            "house_id": "具体房源", "order_no": "订单号",
            "check_in_date": "入住日期", "check_out_date": "退房日期",
        }
        names = "、".join(labels.get(field, str(field)) for field in missing)
        return f"信息收集还未完成，还需要：{names}。请直接告诉我。"
    return "已收到您的消息，但没有生成回复。请换个说法再试一次，或直接告诉我您的需求。"


FIELD_LABELS = {
    "city": "城市",
    "budget_min": "最低预算",
    "budget_max": "最高预算",
    "districts": "区域",
    "room_types": "房型",
    "rental_mode": "租赁方式",
    "house_title": "房源",
    "check_in_date": "入住日期",
    "check_out_date": "退房日期",
}
RENTAL_MODE_LABELS = {
    "whole_rent": "整租",
    "shared": "合租",
    "share_rent": "合租",
}


def _list_text(value: object) -> str:
    """把白名单列表值压缩成一行文本。"""

    if isinstance(value, list):
        return "、".join(str(item) for item in value[:5] if str(item).strip())
    return str(value).strip() if value is not None else ""


def _rental_parameters(
    data: dict,
    *,
    use_stored_preferences: bool = False,
) -> str:
    """格式化租房业务白名单字段，并显式选择长期偏好或本轮状态。"""

    stored = data.get("user_preferences")
    source = (
        stored
        if use_stored_preferences and isinstance(stored, dict)
        else data
    )
    parts: list[str] = []
    city = _list_text(source.get("city"))
    districts = _list_text(source.get("districts"))
    room_types = _list_text(source.get("room_types"))
    minimum = source.get("budget_min")
    maximum = source.get("budget_max")
    mode = RENTAL_MODE_LABELS.get(str(source.get("rental_mode")), "")
    if city:
        parts.append(city)
    if districts:
        parts.append(districts)
    if minimum is not None and maximum is not None:
        parts.append(f"预算 {_money(minimum)}–{_money(maximum)} 元/月")
    elif minimum is not None:
        parts.append(f"最低 {_money(minimum)} 元/月")
    elif maximum is not None:
        parts.append(f"最高 {_money(maximum)} 元/月")
    if room_types:
        parts.append(room_types)
    if mode:
        parts.append(mode)
    return " · ".join(parts)


def _money(value: object) -> str:
    """价格显示去掉无意义的 .0，其他类型安全转文本。"""

    if isinstance(value, (int, float)):
        return f"{value:g}"
    return str(value)


def _page_data(data: dict) -> dict:
    """从单页节点的不同输出外层中取出页面字段。"""

    page = data.get("page_result") or data.get("page")
    if isinstance(page, dict):
        return page
    pages = data.get("page_results")
    if isinstance(pages, list) and pages and isinstance(pages[0], dict):
        return pages[0]
    return {}


def _progress_detail(node: str, data: object) -> str:
    """根据节点白名单读取少量安全参数。"""

    if not isinstance(data, dict):
        return PROGRESS_DESCRIPTIONS[node]

    if node == "load_preferences":
        parameters = _rental_parameters(
            data,
            use_stored_preferences=True,
        )
        if parameters:
            return f"当前条件：{parameters}"

    if node == "anysearch_search":
        query = str(data.get("query") or "").strip()
        results = data.get("search_results")
        if isinstance(results, list):
            return f"搜索“{query}”，找到 {len(results)} 个相关结果。"
        if query:
            return f"正在搜索：{query}"

    if node in {"playwright_read_page", "analyze_page_visuals"}:
        page = _page_data(data)
        title = str(page.get("title") or "").strip()
        if title:
            action = "正在读取" if node == "playwright_read_page" else "正在分析"
            return f"{action}：{title[:60]}"

    return PROGRESS_DESCRIPTIONS[node]


def _progress_event(
    node: str,
    data: object,
    *,
    completed: bool,
    specialist: str | None = None,
) -> dict:
    """构造稳定的前端步骤协议，不传递原始节点输出。"""

    is_specialist_step = node in SPECIALIST_INTERNAL_NODES and specialist is not None
    label = (
        SPECIALIST_PROGRESS_LABELS[specialist]
        if is_specialist_step
        else PROGRESS_LABELS[node]
    )
    return {
        "type": "progress",
        # 专业 Agent 使用 canonical name 区分同一轮的多次委派；普通业务节点
        # 继续用中文标题作为更新键，浏览器不会看到 Python 工具节点名。
        "step_id": specialist if is_specialist_step else label,
        "title": label,
        # 节点开始时的 input 可能还带着上一步的旧值，因此
        # 运行中只显示固定说明；完成后才显示节点的新参数。
        "detail": (
            _progress_detail(node, data)
            if completed
            else PROGRESS_DESCRIPTIONS[node]
        ),
        "status": "completed" if completed else "running",
    }


def iter_chat_events(message: str, thread_id: str | None) -> Iterator[dict]:
    """把一轮对话转换成浏览器可直接消费的安全事件流。"""

    if not thread_id:
        thread_id = post_json(f"{LANGGRAPH_URL}/threads", {})["thread_id"]
    yield {"type": "thread", "thread_id": thread_id}

    initial_state = get_json(f"{LANGGRAPH_URL}/threads/{thread_id}/state")
    initial_values = initial_state.get("values") or {}
    previous_message_count = len(initial_values.get("messages") or [])
    if _thread_interrupts(initial_state):
        run_data = {"command": {"resume": message}}
    else:
        run_data = {
            "input": {"messages": [{"role": "human", "content": message}]}
        }

    payload = {
        "assistant_id": ASSISTANT_ID,
        **run_data,
        "config": {"configurable": {"user_id": USER_ID}},
        "stream_mode": ["messages-tuple", "tasks", "updates"],
        "stream_subgraphs": True,
    }
    started_progress: set[str] = set()
    completed_progress: set[str] = set()
    progress_inputs: dict[str, dict] = {}
    emitted_interrupt = False
    streamed_answer = False

    for event_name, data in _iter_agent_events(
        f"{LANGGRAPH_URL}/threads/{thread_id}/runs/stream",
        payload,
    ):
        kind = _event_type(event_name)
        if kind in {"messages", "messages-tuple"}:
            chunk, metadata = _message_event_parts(data)
            if (
                metadata.get("langgraph_node") in FINAL_ANSWER_NODES
                and "supervisor_agent" in _event_namespace(event_name)
            ):
                token = _message_text(chunk)
                if token:
                    streamed_answer = True
                    yield {"type": "token", "text": token}
            continue

        if kind == "updates" and isinstance(data, dict):
            interrupt_data = _interrupt_payload(data)
            if interrupt_data is not None and not emitted_interrupt:
                emitted_interrupt = True
                yield {
                    "type": "interrupt",
                    "text": _interrupt_text(interrupt_data),
                    "interrupt": interrupt_data,
                }
            continue

        if kind == "tasks":
            specialist = _event_specialist(event_name)
            tasks = data if isinstance(data, list) else [data]
            for task in tasks:
                if not isinstance(task, dict):
                    continue
                node = task.get("name")
                # tasks 同时包含开始和完成事件。完成事件的 result 是节点真实
                # 返回值，不受图公共输出 Schema 对 updates 的过滤。
                is_internal_agent_node = node in SPECIALIST_INTERNAL_NODES
                progress_enabled = (
                    node in PROGRESS_LABELS
                    and (not is_internal_agent_node or specialist is not None)
                )
                progress_key = specialist if is_internal_agent_node else node
                if (
                    progress_enabled
                    and "input" in task
                    and progress_key not in started_progress
                ):
                    started_progress.add(progress_key)
                    task_input = task.get("input")
                    if isinstance(task_input, dict):
                        progress_inputs[node] = task_input
                    yield _progress_event(
                        node,
                        task_input,
                        completed=False,
                        specialist=specialist,
                    )
                if (
                    progress_enabled
                    and "result" in task
                    and progress_key not in completed_progress
                ):
                    completed_progress.add(progress_key)
                    task_result = task.get("result")
                    input_data = progress_inputs.get(node, {})
                    if isinstance(task_result, dict):
                        progress_data = {**input_data, **task_result}
                    else:
                        progress_data = task_result
                    yield _progress_event(
                        node,
                        progress_data,
                        completed=True,
                        specialist=specialist,
                    )
            continue

        if kind == "error":
            raise RuntimeError(_summarize(data, limit=1000))

    state = get_json(f"{LANGGRAPH_URL}/threads/{thread_id}/state")
    values = state.get("values") or {}
    messages = values.get("messages") or []
    interrupts = _thread_interrupts(state)
    interrupt_data = None
    if interrupts:
        entry = interrupts[0]
        interrupt_data = (
            entry.get("value", entry) if isinstance(entry, dict) else entry
        )
    if interrupts and not emitted_interrupt:
        emitted_interrupt = True
        yield {
            "type": "interrupt",
            "text": _interrupt_text(interrupt_data),
            "interrupt": interrupt_data,
        }

    if not emitted_interrupt and not streamed_answer:
        reply = _new_ai_reply(messages, previous_message_count)
        yield {"type": "token", "text": reply or _fallback_reply(values)}

    yield {"type": "done", "interrupted": emitted_interrupt}


def list_sessions(limit: int = 50) -> list[dict]:
    """返回最近的会话列表（按更新时间倒序），带最后一条消息预览。"""
    try:
        # extract 直接让 Agent Server 从每个线程里取出最后一条消息，避免拉全量状态
        threads = post_json(
            f"{LANGGRAPH_URL}/threads/search",
            {"limit": limit, "offset": 0, "extract": {"last_msg": "values.messages[-1]"}},
        )
    except Exception:
        # 个别版本不支持 extract 时降级：只返回线程基本信息
        threads = post_json(f"{LANGGRAPH_URL}/threads/search", {"limit": limit, "offset": 0})

    sessions = []
    for thread in threads or []:
        if not isinstance(thread, dict):
            continue
        thread_id = thread.get("thread_id")
        if not thread_id:
            continue
        last_msg = (thread.get("extracted") or {}).get("last_msg")
        last_message = None
        if isinstance(last_msg, dict) and isinstance(last_msg.get("content"), str):
            last_message = last_msg["content"][:80]
        sessions.append({
            "thread_id": thread_id,
            "updated_at": thread.get("updated_at"),
            "last_message": last_message,
        })
    sessions.sort(key=lambda s: s.get("updated_at") or "", reverse=True)
    return sessions


def session_messages(thread_id: str) -> list[dict]:
    """返回某个线程的人类/AI 消息历史，用于恢复会话时回显。"""
    state = get_json(f"{LANGGRAPH_URL}/threads/{thread_id}/state")
    values = state.get("values") or {}
    messages = []
    for msg in values.get("messages") or []:
        role = msg.get("type")
        content = msg.get("content")
        if role in ("human", "ai") and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content})
    # 线程停在中断点且历史里没有对应文字时，把中断的提示补成一条 AI 消息，
    # 恢复会话时能看到"系统刚才在问什么"
    interrupts = _thread_interrupts(state)
    if interrupts and isinstance(interrupts[0], dict):
        entry = interrupts[0]
        payload = entry.get("value") or entry
        if isinstance(payload, dict) and payload.get("message"):
            messages.append({"role": "ai", "content": payload["message"]})
    return messages


MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
}


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj, status: int = 200) -> None:
        self._send(
            status,
            json.dumps(obj, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def _start_event_stream(self) -> None:
        """开始 NDJSON 响应；每行是一个完整的前端事件。"""

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Connection", "close")
        self.end_headers()

    def _write_event(self, event: dict) -> None:
        """立即写入并刷新一个浏览器事件。"""

        line = json.dumps(event, ensure_ascii=False, default=str) + "\n"
        self.wfile.write(line.encode("utf-8"))
        self.wfile.flush()

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send(200, (WEB_DIR / "index.html").read_bytes(),
                       "text/html; charset=utf-8")
            return
        if path == "/api/sessions":
            try:
                self._send_json({"sessions": list_sessions()})
            except Exception as exc:  # noqa: BLE001 —— 前端需要可读的错误信息
                self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)
            return
        if path.startswith("/api/sessions/") and path.endswith("/messages"):
            thread_id = path[len("/api/sessions/"):-len("/messages")]
            try:
                self._send_json({"messages": session_messages(thread_id)})
            except Exception as exc:  # noqa: BLE001 —— 前端需要可读的错误信息
                self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)
            return
        # 静态资源（如 /images/house-1.jpg），限制只能访问 web 目录内文件
        file = (WEB_DIR / path.lstrip("/")).resolve()
        if file.is_relative_to(WEB_DIR) and file.is_file():
            self._send(200, file.read_bytes(),
                       MIME_TYPES.get(file.suffix.lower(), "application/octet-stream"))
            return
        self.send_error(404, "Not Found")

    def do_POST(self) -> None:
        if self.path != "/api/chat":
            self.send_error(404, "Not Found")
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(req, dict):
                raise ValueError("请求主体必须是 JSON 对象")
            raw_message = req.get("message")
            if raw_message is not None and not isinstance(raw_message, str):
                raise ValueError("message 必须是字符串")
            message = (raw_message or "").strip()
        except (UnicodeDecodeError, ValueError) as exc:
            self._send_json({"error": f"请求格式错误：{exc}"}, status=400)
            return
        if not message:
            self._send_json({"error": "message 不能为空"}, status=400)
            return

        # 从这里开始响应已发出，后续错误也必须作为流事件返回。
        self._start_event_stream()
        try:
            for event in iter_chat_events(message, req.get("thread_id") or None):
                self._write_event(event)
        except (BrokenPipeError, ConnectionResetError):
            _log("[客服] 浏览器已断开流式连接")
        except Exception as exc:  # noqa: BLE001 —— 流已开始，只能写 error 事件
            _log(f"[客服] 流式对话失败: {type(exc).__name__}: {exc}")
            try:
                self._write_event({
                    "type": "error",
                    "message": "客服服务暂时不可用，请稍后重试。",
                })
            except (BrokenPipeError, ConnectionResetError):
                pass

    def log_message(self, fmt, *args) -> None:  # 精简默认日志
        super().log_message(fmt, *args)


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"客服页面:     http://127.0.0.1:{PORT}")
    print(f"AI 大脑(LangGraph Agent Server): {LANGGRAPH_URL}")
    print(f"用户 ID:      {USER_ID}")
    print("按 Ctrl+C 停止")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")


if __name__ == "__main__":
    main()
