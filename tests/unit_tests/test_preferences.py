"""用户偏好保存节点的离线测试。"""

import os
import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore
from pydantic import ValidationError

from agent.common.preferences import (
    PREFERENCE_STORE_KEY,
    PreferenceUpdatesModel,
    preference_namespace,
)
from agent.node.preferences import (
    PreferenceExtractionDecision,
    PreferenceUpdateNode,
    _build_preference_delta,
)


class FakePreferenceExtractionModel:
    def __init__(self, decision: PreferenceExtractionDecision) -> None:
        self.decision = decision
        self.messages = None
        self.method = None

    def with_structured_output(self, schema, *, method=None):
        if schema is not PreferenceExtractionDecision:
            raise AssertionError(f"未处理的结构化类型：{schema}")
        self.method = method
        return self

    def invoke(self, messages):
        self.messages = messages
        return self.decision


class PreferenceValidationTests(unittest.TestCase):
    def test_normalizes_explicit_preference_updates(self) -> None:
        updates = PreferenceUpdatesModel.model_validate(
            {
                "city": " 北京 ",
                "districts": ["朝阳区", " 朝阳区 ", "海淀区"],
                "budget_min": 3000,
                "budget_max": 6000,
                "room_types": ["一室一厅"],
                "rental_mode": "整租",
                "commute_location": " 国贸 ",
                "max_commute_minutes": 45,
            }
        ).explicit_updates()

        self.assertEqual(updates["city"], "北京")
        self.assertEqual(updates["districts"], ["朝阳区", "海淀区"])
        self.assertEqual(updates["rental_mode"], "whole_rent")
        self.assertEqual(updates["commute_location"], "国贸")

    def test_rejects_unknown_or_invalid_preferences(self) -> None:
        with self.assertRaises(ValidationError):
            PreferenceUpdatesModel.model_validate({"favorite_color": "蓝色"})
        with self.assertRaises(ValidationError):
            PreferenceUpdatesModel.model_validate(
                {"budget_min": 6000, "budget_max": 3000}
            )


class UpdatePreferencesNodeTests(unittest.TestCase):
    def test_saves_only_explicit_updates_and_clears_delta(self) -> None:
        store = InMemoryStore()
        result = PreferenceUpdateNode(model_factory=lambda: Mock())(
            {
                "preference_updates": {
                    "city": "上海",
                    "districts": ["浦东新区"],
                    "budget_min": None,
                }
            },
            {"configurable": {"user_id": "user-1"}},
            Runtime(store=store),
        )

        item = store.get(preference_namespace("user-1"), PREFERENCE_STORE_KEY)
        self.assertIsNotNone(item)
        self.assertEqual(item.value["city"], "上海")
        self.assertEqual(item.value["districts"], ["浦东新区"])
        self.assertTrue(result["preferences_saved"])
        self.assertEqual(result["preference_updates"], {})
        self.assertEqual(result["user_preferences"]["city"], "上海")

    def test_uses_chat_user_id_for_studio_and_skips_empty_updates(self) -> None:
        store = InMemoryStore()
        runtime = Runtime(store=store)

        with patch.dict(os.environ, {"CHAT_USER_ID": "studio-user"}):
            node = PreferenceUpdateNode(model_factory=lambda: Mock())
            saved = node(
                {"preference_updates": {"city": "成都"}}, {}, runtime
            )
            skipped = node(
                {"preference_updates": {}}, {}, runtime
            )

        item = store.get(
            preference_namespace("studio-user"), PREFERENCE_STORE_KEY
        )
        self.assertEqual(item.value["city"], "成都")
        self.assertTrue(saved["preferences_saved"])
        self.assertFalse(skipped["preferences_saved"])

    def test_clears_explicit_field_in_store_profile(self) -> None:
        store = InMemoryStore()
        store.put(
            preference_namespace("lk"),
            PREFERENCE_STORE_KEY,
            {"user_id": "lk", "city": "合肥", "districts": ["北城"]},
            index=False,
        )
        result = PreferenceUpdateNode(model_factory=lambda: Mock())(
            {
                "user_id": "lk",
                "preference_updates": {},
                "preference_clear_fields": ["districts"],
            },
            {},
            Runtime(store=store),
        )

        item = store.get(preference_namespace("lk"), PREFERENCE_STORE_KEY)
        self.assertNotIn("districts", item.value)
        self.assertEqual(item.value["city"], "合肥")
        self.assertTrue(result["preferences_saved"])
        self.assertEqual(result["preference_clear_fields"], [])

    def test_store_failure_is_fail_open_and_clears_pending_delta(self) -> None:
        store = Mock()
        store.get.side_effect = RuntimeError("Store暂时不可用")

        result = PreferenceUpdateNode(model_factory=lambda: Mock())(
            {
                "preference_updates": {"city": "上海"},
                "preference_clear_fields": ["districts"],
            },
            {"configurable": {"user_id": "user-1"}},
            Runtime(store=store),
        )

        self.assertFalse(result["preferences_saved"])
        self.assertEqual(result["user_id"], "user-1")
        self.assertEqual(result["preference_updates"], {})
        self.assertEqual(result["preference_clear_fields"], [])
        self.assertIn("Store暂时不可用", result["preference_save_error"])

class PreferenceExtractionAndSaveTests(unittest.TestCase):
    def test_sends_all_messages_from_current_turn_to_model(self) -> None:
        model = FakePreferenceExtractionModel(
            PreferenceExtractionDecision(
                rental_related=True,
                city="合肥",
                districts_to_add=["北城"],
                reason="用户本轮提出租房地点",
            )
        )
        node = PreferenceUpdateNode(model_factory=lambda: model)
        store = InMemoryStore()
        messages = [
            HumanMessage(content="以前想住上海", id="old-human"),
            AIMessage(content="已了解", id="old-ai"),
            HumanMessage(content="给我推荐合肥北城的房子", id="turn-start"),
            AIMessage(content="我来帮你找", id="turn-ai"),
            HumanMessage(content="预算3000以内", id="turn-follow-up"),
        ]

        result = node(
            {
                "messages": messages,
                "user_preferences": {},
                "current_turn_start_message_id": "turn-start",
            },
            {"configurable": {"user_id": "turn-user"}},
            Runtime(store=store),
        )

        prompt = model.messages[-1].content
        self.assertNotIn("以前想住上海", prompt)
        self.assertIn("用户：给我推荐合肥北城的房子", prompt)
        self.assertIn("助手：我来帮你找", prompt)
        self.assertIn("用户：预算3000以内", prompt)
        self.assertEqual(model.method, "function_calling")
        self.assertEqual(result["user_preferences"]["city"], "合肥")
        self.assertEqual(result["user_preferences"]["districts"], ["北城"])
        self.assertTrue(result["preferences_saved"])
        self.assertIsNone(result["current_turn_start_message_id"])

    def test_non_rental_turn_does_not_create_updates(self) -> None:
        model = FakePreferenceExtractionModel(
            PreferenceExtractionDecision(
                rental_related=False,
                budget_max=3000,
                reason="这是商品价格而不是租房预算",
            )
        )
        node = PreferenceUpdateNode(model_factory=lambda: model)

        result = node(
            {
                "messages": [
                    HumanMessage(content="推荐一台3000元以内的手机", id="turn-start")
                ],
                "user_preferences": {"city": "合肥"},
                "current_turn_start_message_id": "turn-start",
            },
            {"configurable": {"user_id": "phone-user"}},
            Runtime(store=InMemoryStore()),
        )

        self.assertNotIn("preference_updates", result)
        self.assertEqual(result["preference_extraction_error"], "")
        self.assertIsNone(result["current_turn_start_message_id"])

    def test_extraction_failure_does_not_discard_existing_updates(self) -> None:
        node = PreferenceUpdateNode(
            model_factory=lambda: (_ for _ in ()).throw(RuntimeError("不可用"))
        )

        store = Mock()
        result = node(
            {
                "messages": [HumanMessage(content="想在合肥租房", id="turn-start")],
                "preference_updates": {"city": "合肥"},
                "current_turn_start_message_id": "turn-start",
            },
            {"configurable": {"user_id": "failure-user"}},
            Runtime(store=store),
        )

        self.assertNotIn("preference_updates", result)
        self.assertIn("RuntimeError", result["preference_extraction_error"])
        self.assertIsNone(result["current_turn_start_message_id"])
        store.get.assert_not_called()


class PreferenceMergeTests(unittest.TestCase):
    def test_appends_districts_in_same_city(self) -> None:
        updates, clear_fields = _build_preference_delta(
            stored_preferences={"city": "合肥", "districts": ["北城"]},
            pending_updates={},
            pending_clear_fields=[],
            decision=PreferenceExtractionDecision(
                rental_related=True,
                city="合肥",
                districts_to_add=["蜀山"],
            ),
        )

        self.assertEqual(updates, {"districts": ["北城", "蜀山"]})
        self.assertEqual(clear_fields, [])

    def test_changing_city_discards_old_districts(self) -> None:
        updates, clear_fields = _build_preference_delta(
            stored_preferences={"city": "上海", "districts": ["浦东"]},
            pending_updates={},
            pending_clear_fields=[],
            decision=PreferenceExtractionDecision(
                rental_related=True,
                city="合肥",
                districts_to_add=["北城"],
            ),
        )

        self.assertEqual(
            updates,
            {"city": "合肥", "districts": ["北城"]},
        )
        self.assertEqual(clear_fields, [])

    def test_removing_last_list_value_clears_store_field(self) -> None:
        updates, clear_fields = _build_preference_delta(
            stored_preferences={"city": "合肥", "districts": ["北城"]},
            pending_updates={},
            pending_clear_fields=[],
            decision=PreferenceExtractionDecision(
                rental_related=True,
                districts_to_remove=["北城"],
            ),
        )

        self.assertEqual(updates, {})
        self.assertEqual(clear_fields, ["districts"])

    def test_new_budget_boundary_clears_conflicting_old_boundary(self) -> None:
        updates, clear_fields = _build_preference_delta(
            stored_preferences={"budget_min": 4000, "budget_max": 6000},
            pending_updates={},
            pending_clear_fields=[],
            decision=PreferenceExtractionDecision(
                rental_related=True,
                budget_max=3000,
            ),
        )

        self.assertEqual(updates, {"budget_max": 3000})
        self.assertEqual(clear_fields, ["budget_min"])


if __name__ == "__main__":
    unittest.main()
