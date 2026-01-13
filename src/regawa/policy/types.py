from typing import NamedTuple

from torch import FloatTensor, Tensor

from regawa.data.torch import SparseTensor


class PolicyOutput(NamedTuple):
    action: Tensor
    entropy: Tensor
    log_pi: Tensor
    value: Tensor
    p_a1: Tensor | SparseTensor[FloatTensor]
    p_a2: SparseTensor[FloatTensor]
