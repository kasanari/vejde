from __future__ import annotations

from typing import NamedTuple

from torch import FloatTensor, Tensor, min

from regawa.data import SparseTensor


class QValue(NamedTuple):
    q1: Tensor | SparseTensor[FloatTensor]
    q2: Tensor | SparseTensor[FloatTensor]

    def min(self, other: QValue) -> QValue:
        is_action_then_node = isinstance(self.q1, Tensor)

        if is_action_then_node:
            return QValue(
                min(self.q1, other.q1),
                self.q2.min(other.q2),
            )
        else:
            return QValue(
                self.q1.min(other.q1),
                self.q2.min(other.q2),
            )
