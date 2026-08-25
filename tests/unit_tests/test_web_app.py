"""Web 聊天后端的流式事件与中断恢复测试。"""

from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path
import unittest
from unittest.mock import patch

from web import app


INDEX_PATH = Path(__file__).parents[2] / "web" / "index.html"


def _capture_events(payloads: list, events=()):
    """构造可记录 Agent Server 请求的假事件流。"""

    def iterator(_url, payload):
        payloads.append(payload)
        yield from events

    return iterator


class WebChatStreamTests(unittest.TestCase):
    def test_current_interrupt_prompt_has_priority_over_old_ai_message(self) -> None:
        """本轮中断提示不能被线程中的上一轮 AI 回复覆盖。"""

        states = [
            {"values": {"messages": [{"type": "ai", "content": "旧答案"}]}, "interrupts": []},
            {
                "values": {"messages": [{"type": "ai", "content": "旧答案"}]},
                "interrupts": [{
                    "id": "interrupt-1",
                    "value": {"message": "是否按这些条件继续推荐？"},
                }],
            },
        ]
        with (
            patch.object(app, "get_json", side_effect=states),
            patch.object(app, "_iter_agent_events", return_value=iter(())),
        ):
            events = list(app.iter_chat_events("给我推荐房子", "thread-1"))

        interrupt = next(event for event in events if event["type"] == "interrupt")
        self.assertEqual(interrupt["text"], "是否按这些条件继续推荐？")
        self.assertFalse(any(event.get("text") == "旧答案" for event in events))

    def test_interrupted_thread_sends_top_level_resume_command(self) -> None:
        """恢复中断必须使用 Agent Server 的顶层 command 字段。"""

        state = {
            "values": {"messages": []},
            "interrupts": [{"value": {"message": "是否继续？"}}],
        }
        payloads = []
        with (
            patch.object(app, "get_json", return_value=state),
            patch.object(app, "_iter_agent_events", side_effect=_capture_events(payloads)),
        ):
            list(app.iter_chat_events("继续", "thread-1"))

        self.assertEqual(payloads[0]["command"], {"resume": "继续"})
        self.assertNotIn("input", payloads[0])

    def test_active_thread_sends_human_input_and_stream_configuration(self) -> None:
        """普通轮次应启用子图、步骤与消息 token 流。"""

        initial = {"values": {"messages": []}, "interrupts": []}
        final = {
            "values": {"messages": [{"type": "ai", "content": "正常回答"}]},
            "interrupts": [],
        }
        payloads = []
        with (
            patch.object(app, "get_json", side_effect=[initial, final]),
            patch.object(app, "_iter_agent_events", side_effect=_capture_events(payloads)),
        ):
            events = list(app.iter_chat_events("新的问题", "thread-1"))

        self.assertEqual(
            payloads[0]["input"],
            {"messages": [{"role": "human", "content": "新的问题"}]},
        )
        self.assertEqual(payloads[0]["stream_mode"], ["messages-tuple", "tasks", "updates"])
        self.assertTrue(payloads[0]["stream_subgraphs"])
        self.assertIn({"type": "token", "text": "正常回答"}, events)

    def test_only_supervisor_model_emits_tokens(self) -> None:
        """专业 Agent 中间结论不能绕过 Supervisor 发送给浏览器。"""

        initial = {"values": {"messages": []}, "interrupts": []}
        final = {"values": {"messages": []}, "interrupts": []}
        agent_events = (
            ("messages|rental_recommendation_agent:1|model:2", [
                {"content": "专业 Agent 中间结论"}, {"langgraph_node": "model"},
            ]),
            ("messages|supervisor_agent:1|model:2", [
                {"content": "为您推荐"}, {"langgraph_node": "model"},
            ]),
            ("messages|supervisor_agent:1|model:2", [
                {"content": "三套房源"}, {"langgraph_node": "model"},
            ]),
        )
        with (
            patch.object(app, "get_json", side_effect=[initial, final]),
            patch.object(app, "_iter_agent_events", return_value=iter(agent_events)),
        ):
            events = list(app.iter_chat_events("推荐房子", "thread-1"))

        tokens = "".join(event["text"] for event in events if event["type"] == "token")
        self.assertEqual(tokens, "为您推荐三套房源")
        self.assertNotIn("中间结论", tokens)

    def test_progress_uses_chinese_allowlist_without_node_output(self) -> None:
        """进度事件只包含中文描述，不包含 SQL 或原始节点名。"""

        initial = {"values": {"messages": []}, "interrupts": []}
        final = {"values": {"messages": [{"type": "ai", "content": "完成"}]}, "interrupts": []}
        agent_events = (("tasks", {
            "name": "search_houses",
            "error": None,
            "result": {"sql": "SELECT secret"},
        }),)
        with (
            patch.object(app, "get_json", side_effect=[initial, final]),
            patch.object(app, "_iter_agent_events", return_value=iter(agent_events)),
        ):
            events = list(app.iter_chat_events("推荐", "thread-1"))

        progress = next(event for event in events if event["type"] == "progress")
        self.assertEqual(progress["title"], "搜索匹配房源")
        self.assertEqual(
            progress["detail"],
            "正在按已确认条件查询真实房源。",
        )
        self.assertEqual(progress["status"], "completed")
        self.assertNotIn("search_houses", str(progress))
        self.assertNotIn("SELECT", str(progress))

    def test_each_specialist_uses_its_own_progress_title(self) -> None:
        """同一轮连续委派时，每个专业 Agent 都应显示独立中文名称。"""

        initial = {"values": {"messages": []}, "interrupts": []}
        final = {"values": {"messages": [{"type": "ai", "content": "完成"}]}, "interrupts": []}
        agent_events = (
            ("tasks|rental_recommendation_agent:1", {
                "name": "model",
                "input": {},
            }),
            ("tasks|rental_recommendation_agent:1", {
                "name": "model",
                "result": {},
            }),
            ("tasks|rental_booking_agent:2", {
                "name": "model",
                "input": {},
            }),
            ("tasks|rental_booking_agent:2", {
                "name": "model",
                "result": {},
            }),
            ("tasks|supervisor_agent:3", {
                "name": "model",
                "input": {},
            }),
        )
        with (
            patch.object(app, "get_json", side_effect=[initial, final]),
            patch.object(app, "_iter_agent_events", return_value=iter(agent_events)),
        ):
            events = list(app.iter_chat_events("推荐并预订房源", "thread-1"))

        progress = [event for event in events if event["type"] == "progress"]
        self.assertEqual(
            [event["title"] for event in progress],
            [
                "房源推荐 Agent 正在处理",
                "房源推荐 Agent 正在处理",
                "房源预订 Agent 正在处理",
                "房源预订 Agent 正在处理",
            ],
        )
        self.assertEqual(
            {event["step_id"] for event in progress},
            {"rental_recommendation_agent", "rental_booking_agent"},
        )

    def test_preference_progress_contains_only_safe_business_parameters(self) -> None:
        """偏好步骤可显示城市预算，但不传递用户 ID 和原始状态。"""

        event = app._progress_event(
            "load_preferences",
            {
                "user_id": "lk",
                "user_preferences": {
                    "city": "合肥",
                    "districts": ["包河区"],
                    "budget_min": 2000,
                    "budget_max": 4000,
                },
            },
            completed=True,
        )

        self.assertEqual(event["title"], "读取您的租房偏好")
        self.assertEqual(
            event["detail"],
            "当前条件：合肥 · 包河区 · 预算 2000–4000 元/月",
        )
        self.assertNotIn("lk", str(event))

    def test_confirmed_technical_nodes_are_hidden(self) -> None:
        """包装节点、SQL 内部节点和偏好写入节点不对用户展示。"""

        hidden = {
            "supervisor_agent", "general_qa_agent", "rental_recommendation_agent",
            "rental_booking_agent", "order_history_agent", "order_cancellation_agent",
            "tools", "SummarizationMiddleware.before_model", "update_preferences",
        }
        self.assertTrue(hidden.isdisjoint(app.PROGRESS_LABELS))

    def test_tool_progress_does_not_expose_raw_results(self) -> None:
        """Agent 工具进度只显示固定业务说明，不返回工具原始数据。"""

        event = app._progress_event(
            "search_houses",
            {
                "houses": [{"title": "房源A", "price": 2000}],
            },
            completed=True,
        )

        self.assertEqual(event["detail"], "正在按已确认条件查询真实房源。")
        self.assertNotIn("房源A", str(event))
        self.assertNotIn("Decimal", str(event))

    def test_running_step_does_not_show_stale_input_parameters(self) -> None:
        """节点开始时不显示旧状态，完成后再更新为新条件。"""

        initial = {"values": {"messages": []}, "interrupts": []}
        final = {"values": {"messages": [{"type": "ai", "content": "完成"}]}, "interrupts": []}
        agent_events = (
            ("tasks", {
                "id": "task-1",
                "name": "search_houses",
                "input": {
                    "city": "合肥", "districts": ["包河区"],
                    "budget_min": 1000, "budget_max": 4000,
                },
            }),
            ("tasks", {
                "name": "search_houses",
                "error": None,
                "result": {
                    "city": "合肥", "districts": ["包河区"],
                    "budget_min": 2000, "budget_max": 4000,
                },
            }),
        )
        with (
            patch.object(app, "get_json", side_effect=[initial, final]),
            patch.object(app, "_iter_agent_events", return_value=iter(agent_events)),
        ):
            events = list(app.iter_chat_events("推荐房子", "thread-1"))

        progress = [event for event in events if event["type"] == "progress"]
        self.assertEqual(progress[0]["status"], "running")
        self.assertNotIn("1000", progress[0]["detail"])
        self.assertEqual(progress[1]["status"], "completed")
        self.assertEqual(progress[1]["detail"], "正在按已确认条件查询真实房源。")

    def test_sse_parser_prints_nested_node_path(self) -> None:
        """标准 SSE event/data 应解析并打印子图节点路径。"""

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def __iter__(self):
                return iter([
                    b"event: tasks|recommend_rental:task-1\n",
                    b'data: {"name":"search_houses","input":{}}\n',
                    b"\n",
                ])

        output = io.StringIO()
        with (
            patch.object(app.urllib.request, "urlopen", return_value=FakeResponse()),
            patch.object(app, "DEBUG_GRAPH", True),
            redirect_stdout(output),
        ):
            events = list(app._iter_agent_events("http://agent/runs/stream", {}))

        self.assertEqual(events[0][0], "tasks|recommend_rental:task-1")
        self.assertIn("recommend_rental / search_houses", output.getvalue())

    def test_terminal_debug_uses_task_result_instead_of_null_update(self) -> None:
        """公共输出 Schema 过滤 updates 时，终端应显示 tasks 的真实结果。"""

        output = io.StringIO()
        task_result = app.json.dumps(
            {
                "name": "load_preferences",
                "error": None,
                "result": {
                    "user_id": "debug-user",
                    "user_preferences": {"city": "杭州"},
                },
            },
            ensure_ascii=False,
        )
        with patch.object(app, "DEBUG_GRAPH", True), redirect_stdout(output):
            app._handle_stream_event("updates", '{"load_preferences": null}')
            app._handle_stream_event("tasks", task_result)

        log = output.getvalue()
        self.assertNotIn("输出: null", log)
        self.assertIn("结果:", log)
        self.assertIn("杭州", log)

    def test_progress_completes_from_task_result_after_null_update(self) -> None:
        """前端进度应使用 tasks.result，而不是被 null updates 提前完成。"""

        initial = {"values": {"messages": []}, "interrupts": []}
        final = {
            "values": {"messages": [{"type": "ai", "content": "完成"}]},
            "interrupts": [],
        }
        agent_events = (
            ("tasks", {
                "name": "load_preferences",
                "input": {"messages": []},
            }),
            ("updates", {"load_preferences": None}),
            ("tasks", {
                "name": "load_preferences",
                "error": None,
                "result": {
                    "user_id": "debug-user",
                    "user_preferences": {
                        "city": "杭州",
                        "budget_min": 2000,
                        "budget_max": 3500,
                    },
                },
            }),
        )
        with (
            patch.object(app, "get_json", side_effect=[initial, final]),
            patch.object(app, "_iter_agent_events", return_value=iter(agent_events)),
        ):
            events = list(app.iter_chat_events("你好", "thread-1"))

        progress = [event for event in events if event["type"] == "progress"]
        self.assertEqual([event["status"] for event in progress], ["running", "completed"])
        self.assertIn("杭州", progress[-1]["detail"])
        self.assertIn("2000–3500", progress[-1]["detail"])

    def test_interrupt_message_is_not_truncated_in_terminal(self) -> None:
        """中断需要用户响应，因此终端提示也必须完整。"""

        message = "请补充必要信息。" * 40
        raw = app.json.dumps(
            {"__interrupt__": [{"value": {"message": message}}]}, ensure_ascii=False,
        )
        output = io.StringIO()
        with patch.object(app, "DEBUG_GRAPH", True), redirect_stdout(output):
            app._handle_stream_event("updates|recommend_rental:task-1", raw)
        self.assertIn(message, output.getvalue())


class WebFrontendTests(unittest.TestCase):
    def test_browser_refresh_does_not_restore_active_thread(self) -> None:
        html = INDEX_PATH.read_text(encoding="utf-8")
        self.assertIn("localStorage.removeItem('cw_thread');", html)
        self.assertIn("let threadId = null;", html)
        self.assertNotIn("localStorage.getItem('cw_thread')", html)

    def test_frontend_consumes_stream_and_collapses_progress(self) -> None:
        html = INDEX_PATH.read_text(encoding="utf-8")
        self.assertIn("response.body.getReader()", html)
        self.assertIn("progress.panel.open = false", html)
        self.assertIn("event.type === 'token'", html)
        self.assertIn("await readNdjson(res", html)


if __name__ == "__main__":
    unittest.main()
