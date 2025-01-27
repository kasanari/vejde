from enum import Enum

from torch import Tensor


class ActionMode(Enum):
    ACTION_THEN_NODE = 0
    NODE_THEN_ACTION = 1
    ACTION_AND_NODE = 2


def num_graphs(batch_idx: Tensor) -> int:
    return int(batch_idx.max().item() + 1)
