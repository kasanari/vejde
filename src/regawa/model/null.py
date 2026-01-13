from typing import Literal, NamedTuple


class NullConstClass(NamedTuple):
    """Constants representing null or no-operation values in the model."""

    idx: Literal[0] = 0
    action: Literal["NOP"] = "NOP"
    type: Literal["NoneType"] = "NoneType"
    id: Literal["None"] = "None"


NullConst = NullConstClass()
