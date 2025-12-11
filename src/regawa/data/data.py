from __future__ import annotations
from collections.abc import Iterable
import json
from itertools import chain

import numpy as np
from numpy.typing import NDArray

from regawa.data.graph import VariableDomain

from .batch import BatchData, HeteroBatchData, batch
from .obs import HeteroObsData, ObsData


class Serializer(json.JSONEncoder):
    def default(self, o: object):
        if isinstance(o, NDArray):  # type: ignore
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)  # type: ignore
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def obs_to_statedata(obs: ObsData[VariableDomain]) -> BatchData[VariableDomain]:
    return obslist_to_statedata([obs])


def obslist_to_statedata(
    obs: list[ObsData[VariableDomain]],
) -> BatchData[VariableDomain]:
    return batch(obs)


attrs_from_obs = [
    attr for attr in ObsData._fields if attr not in ["n_factor", "n_variable"]
]


def heterostatedata(
    obs: Iterable[HeteroObsData],
) -> HeteroBatchData:
    return HeteroBatchData(
        boolean=batch([o.bool for o in obs]),
        numeric=batch([o.float for o in obs]),
    )


def single_obs_to_heterostatedata(obs: HeteroObsData) -> HeteroBatchData:
    return heterostatedata([obs])


# def heterostatedata_from_batched_obs(obs: list[HeteroObsData]) -> HeteroBatchData:
#     return heterostatedata({k: batched_dict_to_obsdata(obs[k]) for k in obs})


# def batched_hetero_dict_to_hetero_obs_list(
#     obs: list[HeteroObsData],
# ) -> dict[str, tuple[ObsData, ...]]:
#     return {k: batched_dict_to_obsdata(o) for o in obs}


def statedata_from_buffer(buf: list[tuple[ObsData[VariableDomain], ...]]):
    return batch(list(chain(*buf)))


def heterostatedata_from_buffer(
    obs: dict[str, list[tuple[ObsData[VariableDomain], ...]]],
) -> HeteroBatchData:
    return HeteroBatchData(
        boolean=statedata_from_buffer(obs["bool"]),  # type: ignore
        numeric=statedata_from_buffer(obs["float"]),  # type: ignore
    )


def heterostatedata_from_obslist(obs: Iterable[HeteroObsData]) -> HeteroBatchData:
    boolean_data = [d.bool for d in obs]
    numeric_data = [d.float for d in obs]

    return HeteroBatchData(
        boolean=batch(boolean_data),
        numeric=batch(numeric_data),
    )
