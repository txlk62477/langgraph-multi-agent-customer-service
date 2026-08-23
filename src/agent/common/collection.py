"""信息收集子图的公共配置接口。"""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, get_type_hints

from pydantic import BaseModel


@dataclass(frozen=True, slots=True)
class CollectionSpec:
    """描述信息收集子图需要收集哪些业务字段。

    必要字段和可选字段属于业务规则，刻意不根据 Python 类型注解中是否包含
    ``None`` 来推断，避免把“当前允许为空”和“业务上可以不填”混为一谈。
    """

    required_fields: Mapping[str, str]
    optional_fields: Mapping[str, str] = field(default_factory=dict)
    max_llm_calls: int = 3

    def __post_init__(self) -> None:
        required = dict(self.required_fields)
        optional = dict(self.optional_fields)

        if not required:
            raise ValueError("required_fields 至少需要配置一个字段")
        if self.max_llm_calls < 1:
            raise ValueError("max_llm_calls 必须大于等于 1")

        duplicated = required.keys() & optional.keys()
        if duplicated:
            names = ", ".join(sorted(duplicated))
            raise ValueError(f"字段不能同时是 required 和 optional: {names}")

        invalid = [
            name
            for name, label in {**required, **optional}.items()
            if not name.strip() or not label.strip()
        ]
        if invalid:
            raise ValueError("字段名和显示名称不能为空")

        # 复制并冻结调用方传入的字典，防止图编译完成后，外部修改字典而悄悄
        # 改变已经生效的业务规则。
        object.__setattr__(self, "required_fields", MappingProxyType(required))
        object.__setattr__(self, "optional_fields", MappingProxyType(optional))

    @property
    def all_fields(self) -> tuple[str, ...]:
        return (*self.required_fields, *self.optional_fields)


def is_missing(value: Any) -> bool:
    """判断一个值是否应当视为“尚未收集”。

    ``None``、空字符串和空容器都算缺失；数字 ``0`` 和布尔值 ``False``
    是有效值，不能误判为缺失。
    """

    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, dict)):
        return not value
    return False


def find_missing_required_fields(
    state: Mapping[str, Any], spec: CollectionSpec
) -> list[str]:
    """按照配置顺序返回仍然缺失的必要字段名。"""

    return [name for name in spec.required_fields if is_missing(state.get(name))]


def validate_collection_contract(
    state_schema: type[Any],
    extraction_schema: type[BaseModel],
    spec: CollectionSpec,
) -> None:
    """校验业务状态、抽取模型和字段配置是否一致，不一致时立即报错。"""

    state_fields = get_type_hints(state_schema, include_extras=True)
    extraction_fields = extraction_schema.model_fields

    # 编译图之前完成契约检查，避免运行到 LLM 节点后才发现字段拼写错误。
    missing_from_state = set(spec.all_fields) - state_fields.keys()
    missing_from_extraction = set(spec.all_fields) - extraction_fields.keys()

    errors: list[str] = []
    if missing_from_state:
        errors.append(
            "state 缺少字段: " + ", ".join(sorted(missing_from_state))
        )
    if missing_from_extraction:
        errors.append(
            "抽取模型缺少字段: " + ", ".join(sorted(missing_from_extraction))
        )
    if errors:
        raise ValueError("；".join(errors))
