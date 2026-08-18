from __future__ import annotations

from typing import NamedTuple

import torch
from torch import FloatTensor, Tensor

from regawa.data import SparseTensor


class QValue(NamedTuple):
    q1: Tensor | SparseTensor[FloatTensor]
    q2: Tensor | SparseTensor[FloatTensor]

    def min(self, other: QValue) -> QValue:
        is_action_then_node = isinstance(self.q1, Tensor)

        return (
            QValue(
                torch.min(self.q1, other.q1),  # type: ignore
                self.q2.min(other.q2),  # type: ignore
            )
            if is_action_then_node
            else QValue(
                self.q1.min(other.q1),  # type: ignore
                self.q2.min(other.q2),  # type: ignore
            )
        )
