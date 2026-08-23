"""可复用信息收集子图的内部节点。"""

import json
from collections.abc import Callable, Mapping
from typing import Any

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from pydantic import BaseModel
from langgraph.types import interrupt

from agent.common.collection import (
    CollectionSpec,
    find_missing_required_fields,
    is_missing,
)


ModelFactory = Callable[[], Any]


class InformationCollectionNodes:
    """封装信息抽取、完成判断和中断询问的具体实现。"""

    def __init__(
        self,
        *,
        spec: CollectionSpec,
        extraction_schema: type[BaseModel],
        model_factory: ModelFactory,
    ) -> None:
        self._spec = spec
        self._extraction_schema = extraction_schema
        self._model_factory = model_factory

    def evaluate(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """不调用模型，只根据当前状态判断信息是否收集完成。"""

        missing = find_missing_required_fields(state, self._spec)
        llm_call_count = int(state.get("llm_call_count", 0))

        # 优先判断必要字段是否已经收齐。因此即使调用次数刚好达到上限，
        # 只要 missing 为空，仍然是正常完成，而不是超限失败。
        if not missing:
            status = "complete"
        elif llm_call_count >= self._spec.max_llm_calls:
            status = "incomplete"
        else:
            status = "collecting"

        return {
            "collection_status": status,
            "missing_required_fields": missing,
            "llm_call_count": llm_call_count,
            "max_llm_calls": self._spec.max_llm_calls,
        }

    def extract(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """调用一次 LLM，从对话中抽取配置指定的业务字段。"""

        prompt = self._build_extraction_prompt(state)
        messages: list[AnyMessage] = [SystemMessage(content=prompt)]
        messages.extend(state.get("messages", []))

        try:
            # DeepSeek 使用工具调用返回 Pydantic 结构，避免发送其接口不支持的
            # OpenAI 原生 response_format。
            structured_model = self._model_factory().with_structured_output(
                self._extraction_schema,
                method="function_calling",
            )
            result = structured_model.invoke(messages)
            if isinstance(result, BaseModel):
                extracted = result.model_dump(exclude_none=True)
            elif isinstance(result, Mapping):
                extracted = dict(result)
            else:
                raise TypeError("结构化抽取结果必须是 Pydantic 模型或字典")
        except Exception as error:
            # LLM 不可用时结束本轮收集，不再反复 interrupt；业务父图会根据
            # collection_error 生成稳定的降级提示。
            return {
                "collection_status": "incomplete",
                "missing_required_fields": find_missing_required_fields(
                    state, self._spec
                ),
                "llm_call_count": self._spec.max_llm_calls,
                "max_llm_calls": self._spec.max_llm_calls,
                "collection_error": f"{type(error).__name__}: {error}",
            }

        # 只合并本业务声明过且非空的字段。模型没有提到的字段不会覆盖旧值。
        update = {
            name: value
            for name, value in extracted.items()
            if name in self._spec.all_fields and not is_missing(value)
        }
        update["llm_call_count"] = int(state.get("llm_call_count", 0)) + 1
        update["collection_error"] = ""
        return update

    def ask_for_missing_information(
        self, state: Mapping[str, Any]
    ) -> dict[str, list[HumanMessage]]:
        """暂停图执行，并在恢复后把用户补充内容追加为一条消息。"""

        missing = list(state.get("missing_required_fields", []))
        labels = [self._spec.required_fields[name] for name in missing]
        known = [
            f"{label}={state.get(name)}"
            for name, label in {
                **self._spec.required_fields,
                **self._spec.optional_fields,
            }.items()
            if not is_missing(state.get(name))
        ]
        message = ""
        if known:
            message += "当前将使用以下信息：" + "；".join(known) + "。"
        message += "还需要以下必要信息：" + "、".join(labels) + "。请一次性告诉我。"
        if known:
            message += " 如果已有信息需要修改，也请在本次回复中一起说明。"

        if self._spec.optional_fields:
            optional_labels = "、".join(self._spec.optional_fields.values())
            message += f" 如果方便，也可以补充可选信息：{optional_labels}。"

        # interrupt 会保存当前图状态。调用方随后使用 Command(resume=...)
        # 恢复执行，返回值 answer 就是用户本次补充的信息。
        answer = interrupt(
            {
                "type": "information_required",
                "message": message,
                "missing_required_fields": missing,
                "llm_call_count": int(state.get("llm_call_count", 0)),
                "max_llm_calls": self._spec.max_llm_calls,
            }
        )
        return {"messages": [HumanMessage(content=self._answer_to_text(answer))]}

    def _build_extraction_prompt(self, state: Mapping[str, Any]) -> str:
        required = "\n".join(
            f"- {name}: {label}（必要）"
            for name, label in self._spec.required_fields.items()
        )
        optional = "\n".join(
            f"- {name}: {label}（可选）"
            for name, label in self._spec.optional_fields.items()
        ) or "- 无"
        known = {
            name: state.get(name)
            for name in self._spec.all_fields
            if not is_missing(state.get(name))
        }

        return (
            "你是一个严格的信息抽取器。请从对话中抽取字段，并返回指定的结构化结果。\n"
            "只填写用户明确提供或明确修改的信息，不要猜测；未提及的字段返回 null。\n"
            "已有信息应保留，只有用户明确给出新值时才更新。\n\n"
            f"必要字段：\n{required}\n\n"
            f"可选字段：\n{optional}\n\n"
            "当前已有信息："
            + json.dumps(known, ensure_ascii=False, default=str)
        )

    @staticmethod
    def _answer_to_text(answer: Any) -> str:
        if isinstance(answer, str):
            return answer
        return json.dumps(answer, ensure_ascii=False, default=str)
