import json
from collections import deque
from collections.abc import Iterable
from itertools import chain
from typing import Generic, NamedTuple, TypeVar

# import torch
import numpy as np
# import torch as th
# from torch import NDArray
from numpy import asarray, concatenate, cumsum, ones_like
from numpy.typing import NDArray

from regawa.gnn.gnn_classes import SparseArray

ObsDict = dict[str, NDArray]
BatchedObsDict = dict[str, tuple[NDArray]]

HeteroObsDict = dict[str, ObsDict]
HeteroBatchedObsDict = dict[str, BatchedObsDict]


class Serializer(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, NDArray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class ObsData(NamedTuple):
    var_value: NDArray
    var_type: NDArray
    factor: NDArray
    senders: NDArray
    receivers: NDArray
    edge_attr: NDArray
    length: NDArray
    # globals: NDArray
    n_factor: int
    n_variable: int
    global_vars: NDArray
    global_vals: NDArray
    global_length: NDArray
    action_mask: NDArray


V = TypeVar("V", np.float32, np.bool_)


class StateData(NamedTuple, Generic[V]):
    var_value: SparseArray[V]
    var_type: SparseArray[np.int64]
    factor: SparseArray[np.int64]
    senders: NDArray[np.int64]  # variable
    receivers: NDArray[np.int64]  # factor
    edge_attr: NDArray[np.int64]
    n_factor: NDArray[np.int64]
    n_variable: NDArray[np.int64]
    n_graphs: np.int64
    length: NDArray[np.int64]
    global_vars: SparseArray[np.int64]
    global_vals: SparseArray[V]
    global_length: NDArray[np.int64]
    action_mask: NDArray[np.int8]


class HeteroStateData(NamedTuple):
    boolean: StateData[np.bool_]
    numeric: StateData[np.float32]


class GraphBuffer:
    def __init__(self) -> None:
        self.data: deque[ObsData] = deque()

    def extend(self, obs: Iterable[ObsData]) -> None:
        self.data.extend(obs)

    def add_single(self, obs: ObsData) -> None:
        self.data.append(obs)

    def add_single_dict(self, obs: ObsDict) -> None:
        self.data.append(dict_to_obsdata(obs))

    def batch(self) -> StateData:
        return batch(list(self.data))

    def __getitem__(self, index: int) -> ObsData:
        return self.data[index]

    def minibatch(self, indices: Iterable[int]) -> StateData:
        return batch([self.data[i] for i in indices])


class HeteroGraphBuffer:
    def __init__(self) -> None:
        types = ("bool", "float")
        self.buffers = {t: GraphBuffer() for t in types}

    def extend(self, obs: dict[str, list[ObsData]]) -> None:
        for t in self.buffers:
            self.buffers[t].extend(obs[t])

    def __iter__(self):
        return self.buffers.__iter__()

    def add_single_dict(self, obs: HeteroObsDict) -> None:
        for t in self.buffers:
            self.buffers[t].add_single_dict(obs[t])

    @property
    def batch(self) -> HeteroStateData:
        return heterostatedata({k: list(self.buffers[k].data) for k in self.buffers})

    def minibatch(self, indices: Iterable[int]) -> HeteroStateData:
        return heterostatedata(
            {k: [self.buffers[k].data[i] for i in indices] for k in self.buffers}
        )


class Rollout(NamedTuple):
    rewards: list[float]
    obs: HeteroGraphBuffer
    actions: list[tuple[int, int]]
    values: list[float]


def save_rollout(rollout: Rollout, path: str):
    with open(path, "w") as f:
        json.dump(rollout._asdict(), f, cls=Serializer)


def load_rollout(path: str) -> Rollout:
    with open(path, "r") as f:
        data = json.load(f)
    return Rollout(**data)


class RolloutCollector:
    rewards: deque[float]
    obs: HeteroGraphBuffer
    actions: deque[tuple[int, int]]

    def __init__(self) -> None:
        self.rewards = deque()
        self.obs = HeteroGraphBuffer()
        self.actions = deque()

    def add_single(
        self, obs: HeteroObsDict, action: tuple[int, int], reward: float
    ) -> None:
        self.rewards.append(reward)
        self.obs.add_single_dict(obs)
        self.actions.append(action)

    def export(self) -> Rollout:
        return Rollout(
            rewards=list(self.rewards),
            obs=self.obs,
            actions=list(self.actions),
            values=self.values,
        )

    @property
    def return_(self) -> float:
        return sum(self.rewards)

    @property
    def values(self) -> list[float]:
        returns = [sum(list(self.rewards)[i:]) for i in range(len(self.rewards))]
        return returns


def batch(graphs: list[ObsData] | tuple[ObsData, ...]) -> StateData:
    """Returns batched graph given a list of graphs and a numpy-like module."""
    # Calculates offsets for sender and receiver NDArrays, caused by concatenating
    # the nodes NDArrays.
    factor_offsets = cumsum(
        asarray([0] + [g.n_factor for g in graphs[:-1]]),
        axis=0,
    )
    factor_batch = concatenate([ones_like(g.factor) * i for i, g in enumerate(graphs)])
    receivers = concatenate([g.receivers + o for g, o in zip(graphs, factor_offsets)])

    variable_offsets = cumsum(
        asarray([0] + [g.n_variable for g in graphs[:-1]]),
        axis=0,
    )
    var_batch = concatenate(
        [np.ones(g.n_variable, dtype=np.int64) * i for i, g in enumerate(graphs)]
    )
    senders = concatenate([g.senders + o for g, o in zip(graphs, variable_offsets)])

    global_vars = np.concatenate([g.global_vars for g in graphs])
    global_vals = np.concatenate([g.global_vals for g in graphs])
    global_batch = np.concatenate(
        [ones_like(g.global_vars) * i for i, g in enumerate(graphs)]
    )

    var_vals = concatenate([g.var_value for g in graphs])

    action_mask = concatenate([g.action_mask for g in graphs])

    return StateData(
        # n_node=stack([g.n_node for g in graphs]),
        var_value=SparseArray(var_vals, var_batch),
        var_type=SparseArray(concatenate([g.var_type for g in graphs]), var_batch),
        factor=SparseArray(concatenate([g.factor for g in graphs]), factor_batch),
        edge_attr=concatenate([g.edge_attr for g in graphs]),
        senders=senders,
        receivers=receivers,
        n_factor=asarray([g.n_factor for g in graphs]),
        n_variable=asarray([g.n_variable for g in graphs]),
        n_graphs=len(graphs),
        length=concatenate([g.length for g in graphs]),
        global_vars=SparseArray(global_vars, global_batch),
        global_vals=SparseArray(global_vals, global_batch),
        global_length=concatenate([g.global_length for g in graphs]),
        action_mask=action_mask,
    )


def obs_to_statedata(obs: ObsDict) -> StateData:
    return obslist_to_statedata([dict_to_obsdata(obs)])


def obslist_to_statedata(obs: list[ObsData]) -> StateData:
    return batch(obs)


attrs_from_obs = [
    attr for attr in ObsData._fields if attr not in ["n_factor", "n_variable"]
]


def batched_dict_to_obsdata(obs: BatchedObsDict) -> tuple[ObsData, ...]:
    def create_data(i: int) -> ObsData:
        data = {attr: obs[attr][i] for attr in attrs_from_obs}
        return ObsData(
            **data,
            n_factor=obs["factor"][i].shape[0],  # + obs["var_value"].shape[0]
            n_variable=obs["length"][i].shape[0],
        )

    return tuple(create_data(i) for i in range(len(obs["factor"])))


def dict_to_obsdata(obs: ObsDict) -> ObsData:
    data = {attr: obs[attr] for attr in attrs_from_obs}
    return ObsData(
        **data,
        n_factor=obs["factor"].shape[0],  # + obs["var_value"].shape[0]
        n_variable=obs["length"].shape[
            0
        ],  # length is used since var_values may be flattened node histories
    )


def heterostatedata(
    obs: dict[str, list[ObsData] | tuple[ObsData, ...]],
) -> HeteroStateData:
    return HeteroStateData(
        boolean=batch(obs["bool"]),
        numeric=batch(obs["float"]),
    )


def single_obs_to_heterostatedata(obs: HeteroObsDict) -> HeteroStateData:
    return heterostatedata({k: [dict_to_obsdata(obs[k])] for k in obs})


def heterostatedata_from_batched_obs(obs: HeteroBatchedObsDict) -> HeteroStateData:
    return heterostatedata({k: batched_dict_to_obsdata(obs[k]) for k in obs})


def batched_hetero_dict_to_hetero_obs_list(
    obs: HeteroBatchedObsDict,
) -> dict[str, tuple[ObsData, ...]]:
    return {k: batched_dict_to_obsdata(obs[k]) for k in obs}


def statedata_from_buffer(buf: list[tuple[ObsData, ...]]):
    return batch(list(chain(*buf)))


def heterostatedata_from_buffer(
    obs: dict[str, list[tuple[ObsData, ...]]],
) -> HeteroStateData:
    return HeteroStateData(
        boolean=statedata_from_buffer(obs["bool"]),
        numeric=statedata_from_buffer(obs["float"]),
    )


def heterostatedata_from_obslist(obs: list[dict[str, ObsData]]) -> HeteroStateData:
    boolean_data = list(chain(*[d["bool"] for d in obs]))
    numeric_data = list(chain(*[d["float"] for d in obs]))

    return HeteroStateData(
        boolean=batch(boolean_data),
        numeric=batch(numeric_data),
    )


def heterostatedata_from_obslist_alt(obs: list[dict[str, ObsData]]) -> HeteroStateData:
    boolean_data = [d["bool"] for d in obs]
    numeric_data = [d["float"] for d in obs]

    return HeteroStateData(
        boolean=batch(boolean_data),
        numeric=batch(numeric_data),
    )


def heterodict_to_obsdata(obs: HeteroObsDict):
    return {k: (dict_to_obsdata(obs[k]),) for k in obs}
