from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


class RenderGraph(NamedTuple):
    variable_labels: Sequence[str]
    factor_labels: Sequence[str]
    v_to_f: NDArray[np.int64]
    f_to_v: NDArray[np.int64]
    edge_attributes: Sequence[int]
    global_variables: Sequence[str]
