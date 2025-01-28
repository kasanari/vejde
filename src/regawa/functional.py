from torch import Tensor

from gnn_policy.functional import segment_sum


def node_mask(action_mask: Tensor) -> Tensor:
    return action_mask[:, 1:].any(1)


def predicate_mask(action_mask: Tensor, index: Tensor) -> Tensor:
    return segment_sum(action_mask, index, num_graphs(index)) > 0


def num_graphs(batch_idx: Tensor) -> int:
    return int(batch_idx.max().item() + 1)
