from collections import deque
from collections.abc import Iterable
from itertools import chain
import json
from typing import NamedTuple

import torch as th
from torch import Tensor
from torch import cumsum, ones_like, concatenate
import torch
import numpy as np

from regawa.gnn.gnn_classes import SparseTensor


ObsDict = dict[str, Tensor]
BatchedObsDict = dict[str, tuple[Tensor]]

HeteroObsDict = dict[str, ObsDict]
HeteroBatchedObsDict = dict[str, BatchedObsDict]


class Serializer(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Tensor):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class ObsData(NamedTuple):
    var_value: Tensor
    var_type: Tensor
    factor: Tensor
    senders: Tensor
    receivers: Tensor
    edge_attr: Tensor
    length: Tensor
    # globals: Tensor
    n_factor: int
    n_variable: int
    global_vars: Tensor
    global_vals: Tensor
    global_length: Tensor
    action_mask: Tensor


class StateData(NamedTuple):
    var_value: SparseTensor
    var_type: SparseTensor
    factor: SparseTensor
    senders: Tensor  # variable
    receivers: Tensor  # factor
    edge_attr: Tensor
    n_factor: Tensor
    n_variable: Tensor
    n_graphs: int
    length: Tensor
    global_vars: SparseTensor
    global_vals: SparseTensor
    global_length: Tensor
    action_mask: Tensor


class HeteroStateData(NamedTuple):
    boolean: StateData
    numeric: StateData


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
        )

    @property
    def return_(self) -> float:
        return sum(self.rewards)


def batch(graphs: list[ObsData] | tuple[ObsData, ...]) -> StateData:
    """Returns batched graph given a list of graphs and a numpy-like module."""
    # Calculates offsets for sender and receiver Tensors, caused by concatenating
    # the nodes Tensors.
    factor_offsets = cumsum(
        torch.as_tensor([0] + [g.n_factor for g in graphs[:-1]]), dim=0
    )
    factor_batch = concatenate([ones_like(g.factor) * i for i, g in enumerate(graphs)])
    receivers = concatenate([g.receivers + o for g, o in zip(graphs, factor_offsets)])

    variable_offsets = cumsum(
        torch.as_tensor([0] + [g.n_variable for g in graphs[:-1]]), dim=0
    )
    var_batch = concatenate(
        [torch.ones(g.n_variable, dtype=th.int64) * i for i, g in enumerate(graphs)]
    )
    senders = concatenate([g.senders + o for g, o in zip(graphs, variable_offsets)])

    global_vars = torch.cat([g.global_vars for g in graphs])
    global_vals = torch.cat([g.global_vals for g in graphs])
    global_batch = torch.cat(
        [ones_like(g.global_vars) * i for i, g in enumerate(graphs)]
    )

    var_vals = concatenate([g.var_value for g in graphs])

    action_mask = concatenate([g.action_mask for g in graphs])

    return StateData(
        # n_node=stack([g.n_node for g in graphs]),
        var_value=SparseTensor(var_vals, var_batch),
        var_type=SparseTensor(concatenate([g.var_type for g in graphs]), var_batch),
        factor=SparseTensor(concatenate([g.factor for g in graphs]), factor_batch),
        edge_attr=concatenate([g.edge_attr for g in graphs]),
        senders=senders,
        receivers=receivers,
        n_factor=torch.as_tensor([g.n_factor for g in graphs]),
        n_variable=torch.as_tensor([g.n_variable for g in graphs]),
        n_graphs=len(graphs),
        length=concatenate([g.length for g in graphs]),
        global_vars=SparseTensor(global_vars, global_batch),
        global_vals=SparseTensor(global_vals, global_batch),
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
        data = {attr: torch.as_tensor(obs[attr][i]) for attr in attrs_from_obs}
        return ObsData(
            **data,
            n_factor=obs["factor"][i].shape[0],  # + obs["var_value"].shape[0]
            n_variable=len(obs["length"][i]),
        )

    return tuple(create_data(i) for i in range(len(obs["factor"])))


def dict_to_obsdata(obs: ObsDict) -> ObsData:
    data = {attr: torch.as_tensor(obs[attr]) for attr in attrs_from_obs}
    return ObsData(
        **data,
        n_factor=obs["factor"].shape[0],  # + obs["var_value"].shape[0]
        n_variable=len(
            obs["length"]
        ),  # length is used since var_values may be flattened node histories
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


def heterodict_to_obsdata(obs: HeteroObsDict):
    return {k: (dict_to_obsdata(obs[k]),) for k in obs}
