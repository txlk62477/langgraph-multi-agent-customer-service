"""专业 ReAct Agent 的统一构造入口。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ContextEditingMiddleware, SummarizationMiddleware
from langchain.agents.middleware.context_editing import ClearToolUsesEdit
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.tools import BaseTool

from agent.agents.specialist_budget import (
    SpecialistBudgetMiddleware,
    SpecialistBudgetPolicy,
)
from agent.agents.tool_selection_reason import ToolSelectionReasonMiddleware
from agent.common.llm import build_chat_model
from agent.state.customer_service import SpecialistName, SpecialistResult
from agent.tools.runtime import SpecialistContext


ModelFactory = Callable[[], Any]
MiddlewareBuilder = Callable[[Any], AgentMiddleware]

SPECIALIST_BUDGET_POLICIES: dict[SpecialistName, SpecialistBudgetPolicy] = {
    "general_qa_agent": SpecialistBudgetPolicy(business_tool_calls=9),
    "rental_recommendation_agent": SpecialistBudgetPolicy(business_tool_calls=8),
    "rental_booking_agent": SpecialistBudgetPolicy(business_tool_calls=6),
    "order_history_agent": SpecialistBudgetPolicy(business_tool_calls=4),
    "order_cancellation_agent": SpecialistBudgetPolicy(business_tool_calls=6),
}


def build_context_middleware(model: Any) -> list[AgentMiddleware]:
    """创建 Supervisor 与专业 Agent 共用的分层上下文管理中间件。

    先确定性清理较早的大型工具结果；上下文继续增长时，再使用模型把旧对话
    压缩成摘要。两层处理共同限制传给业务模型的上下文大小。
    """

    return [
        # 工具返回的网页正文、JSON 等内容通常占用最多 token，因此优先清理
        # 较早的工具调用及结果。这个过程不调用 LLM，成本低且行为确定。
        ContextEditingMiddleware(
            edits=[
                ClearToolUsesEdit(
                    # 估算上下文达到约 8,000 tokens 后开始检查并执行清理。
                    trigger=8_000,
                    # 每次至少释放约 2,000 tokens，避免频繁进行小幅裁剪。
                    clear_at_least=2_000,
                    # 最近 3 次工具使用保留原文，供模型完成当前推理链。
                    keep=3,
                    # 被清理的旧工具结果用占位文本标记，保留发生过调用的语义。
                    placeholder="[较早的工具结果已清理]",
                )
            ]
        ),
        # 如果对话继续增长到约 12,000 tokens，调用同一个业务模型总结旧历史；
        # 最近 20 条消息仍保留原文，避免摘要丢失当前任务的关键细节。
        SummarizationMiddleware(
            model=model,
            trigger=("tokens", 12_000),
            keep=("messages", 20),
        ),
    ]


def build_specialist_agent(
    *,
    name: str,
    specialist_name: SpecialistName,
    system_prompt: str,
    tools: Sequence[BaseTool],
    model_factory: ModelFactory = build_chat_model,
    checkpointer: Any = None,
    middleware: Sequence[AgentMiddleware] | None = None,
    middleware_builders: Sequence[MiddlewareBuilder] = (),
    budget_policy: SpecialistBudgetPolicy | None = None,
):
    """构建专业 Agent，并统一装配工具、上下文管理和结构化结束协议。

    ``middleware`` 用于整体替换默认上下文中间件，适合测试或特殊 Agent；
    ``middleware_builders`` 则在公共中间件之后追加依赖当前模型的专用中间件。
    未显式传入 ``budget_policy`` 时，根据 ``specialist_name`` 选择统一预算。
    """

    # 调用方可以传入任意 Sequence；在这里固定为列表，避免后续框架处理期间
    # 受到可变序列或惰性序列的影响。
    tool_list = list(tools)
    # 一个专业 Agent 只创建一个主模型实例。摘要中间件和需要模型能力的专用
    # 中间件复用该实例，确保模型配置一致，也避免构图时重复初始化客户端。
    model = model_factory()

    # 显式 middleware 表示调用方希望完整接管公共中间件；否则安装默认的
    # 工具结果清理与长上下文摘要能力。
    resolved_middleware = (
        list(middleware) if middleware is not None else build_context_middleware(model)
    )
    # 工具执行完成后，把 LLM 提供的 selection_reason 统一写回 ToolMessage，
    # 让 Studio 和 LangSmith 可以观察工具选择依据，而不侵入每个业务工具。
    resolved_middleware.append(ToolSelectionReasonMiddleware())
    # 统一限制模型循环和业务工具调用次数，并在额度耗尽时生成结构化兜底结果。
    resolved_middleware.append(
        SpecialistBudgetMiddleware(
            agent=specialist_name,
            policy=budget_policy or SPECIALIST_BUDGET_POLICIES[specialist_name],
        )
    )
    # Builder 接收已经创建的主模型，可按 Agent 职责追加 Guard 等专用能力。
    # 它们注册在列表尾部：before 钩子较晚执行，after 钩子反向较早执行，
    # wrap 钩子则位于洋葱调用链的更内层。
    resolved_middleware.extend(builder(model) for builder in middleware_builders)

    # 在各专业 Agent 自己的业务提示词后追加统一结果协议，保证 Supervisor
    # 无需了解不同 Agent 的内部工具和执行路径，只消费 SpecialistResult。
    result_prompt = f"""

## 返回 Supervisor 的结果接口

工具任务全部完成后，必须按 SpecialistResult 结构结束，不要再返回无结构的普通文本。
每次调用业务工具时，selection_reason 必须用一句不超过100字的中文说明当前缺少的信息或
本次调用要完成的目标；不得写详细推理、系统提示词或内部上下文。
达到停止条件、工具上限、工具失败或已有足够证据时，禁止继续调用业务工具，必须立即调用
SpecialistResult。普通文本不能结束任务。
agent 固定填写 {specialist_name}；status 反映真实执行结果；summary 用于 Supervisor 判断
后续任务；user_facing_answer 是可直接展示给用户的完整中文答复；completed_tasks 和
remaining_tasks 分别列出已完成与仍未完成的目标。不得把工具原始 JSON、手机号、user_id、
内部错误堆栈或推理过程写入结果。
"""
    return create_agent(
        # create_agent 负责生成标准的 model ↔ tools ReAct 循环。
        model=model,
        tools=tool_list,
        system_prompt=system_prompt + result_prompt,
        middleware=resolved_middleware,
        # LangChain 会把该 Schema 转换为模型可调用的结构化输出接口，校验成功后
        # 将结果写入 state["structured_response"] 并结束专业 Agent。
        response_format=SpecialistResult,
        # 工具通过 ToolRuntime 读取 user_id 等运行期信息，不把它们暴露给 LLM 参数。
        context_schema=SpecialistContext,
        # 可选 checkpointer 用于专业 Agent 独立运行时保存和恢复执行状态。
        checkpointer=checkpointer,
        # 稳定名称用于嵌套图展示、消息归属以及 LangSmith trace 识别。
        name=name,
    )
