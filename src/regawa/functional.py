from collections.abc import Callable
from torch import Tensor


def node_mask(action_mask: Tensor) -> Tensor:
    return action_mask[:, 1:].any(1)


def predicate_mask(action_mask: Tensor, segsum: Callable[[Tensor], Tensor]) -> Tensor:
    return segsum(action_mask) > 0


def num_graphs(batch_idx: Tensor) -> int:
    return int(batch_idx.max().item() + 1)
