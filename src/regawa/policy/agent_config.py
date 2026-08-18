from dataclasses import dataclass
from enum import Enum

from torch import nn

from ..embedding.boolean import BooleanEmbedderType


class ActionMode(Enum):
    ACTION_THEN_NODE = 0
    NODE_THEN_ACTION = 1
    ACTION_AND_NODE = 2


# dataclasses for easy serialization
@dataclass(frozen=True)
class GNNParams:
    embedding_dim: int
    layers: int
    aggregation: str
    activation: nn.Module
    action_mode: ActionMode
    recurrent = False
    # node embedding
    boolean_embedder_type: BooleanEmbedderType = BooleanEmbedderType.NEGATIVE_BIAS

    # custom equality check for activation function since pytorch is dumb
    def __eq__(self, value: object) -> bool:
        if not isinstance(value, GNNParams):
            return False
        return (
            self.embedding_dim == value.embedding_dim
            and self.layers == value.layers
            and self.aggregation == value.aggregation
            and isinstance(self.activation, type(value.activation))
            and self.action_mode == value.action_mode
            and self.recurrent == value.recurrent
            and self.boolean_embedder_type == value.boolean_embedder_type
        )


@dataclass(frozen=True)
class AgentConfig:
    # environment parameters
    num_object_classes: int
    num_predicate_classes: int
    num_actions: int

    # GNN parameters
    hyper_params: GNNParams
    arity: int


__all__ = ["ActionMode", "AgentConfig", "BooleanEmbedderType", "GNNParams"]
