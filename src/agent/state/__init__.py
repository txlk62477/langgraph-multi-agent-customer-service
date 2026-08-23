"""Agent 图使用的状态结构。"""

from agent.state.information_collection import (
    CollectionState,
    RecommendCollectionState,
)
from agent.state.general_qa import GeneralQAState
from agent.state.customer_service import CustomerServiceState

__all__ = [
    "CollectionState",
    "CustomerServiceState",
    "GeneralQAState",
    "RecommendCollectionState",
]
