from dataclasses import dataclass
from enum import Enum
import torch.nn as nn


class ActionMode(Enum):
    ACTION_THEN_NODE = 0
    NODE_THEN_ACTION = 1
    ACTION_AND_NODE = 2


@dataclass
class Config:
    num_object_classes: int
    num_predicate_classes: int
    num_actions: int
    embedding_dim: int
    layers: int
    aggregation: str
    activation: nn.Module
    action_mode: ActionMode
    recurrent = False
