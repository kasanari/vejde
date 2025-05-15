import json
from collections import deque
from collections.abc import Iterable
from itertools import chain
from typing import Generic, NamedTuple, TypeVar, Any

# import torch
import numpy as np

# import torch as th
# from torch import NDArray
from numpy.typing import NDArray
from torch import Tensor

from regawa.gnn.gnn_classes import SparseArray, SparseTensor


class Serializer(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, NDArray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


V = TypeVar("V", np.float32, np.bool_)


class ObsData(NamedTuple, Generic[V]):
    var_value: NDArray[V]
    var_type: NDArray[np.int64]
    factor: NDArray[np.int64]
    senders: NDArray[np.int64]
    receivers: NDArray[np.int64]
    edge_attr: NDArray[np.int64]
    length: NDArray[np.int64]
    n_factor: int
    n_variable: int
    global_vars: NDArray[np.int64]
    global_vals: NDArray[V]
    global_length: NDArray[np.int64]
    action_type_mask: NDArray[np.bool_]
    action_arity_mask: NDArray[np.bool_]


class HeteroObsData(NamedTuple):
    bool: ObsData[np.bool_]
    float: ObsData[np.float32]


class BatchData(NamedTuple, Generic[V]):
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
    action_arity_mask: NDArray[np.bool_]
    action_type_mask: NDArray[np.bool_]


class HeteroBatchData(NamedTuple):
    boolean: BatchData[np.bool_]
    numeric: BatchData[np.float32]


class GraphBuffer:
    def __init__(self) -> None:
        self.data: deque[ObsData] = deque()

    def extend(self, obs: Iterable[ObsData]) -> None:
        self.data.extend(obs)

    def add_single(self, obs: ObsData) -> None:
        self.data.append(obs)

    def add_single_dict(self, obs: ObsData) -> None:
        self.data.append(obs)

    def batch(self) -> BatchData:
        return batch(list(self.data))

    def __getitem__(self, index: int) -> ObsData:
        return self.data[index]

    def minibatch(self, indices: Iterable[int]) -> BatchData:
        return batch([self.data[i] for i in indices])


class HeteroGraphBuffer:
    def __init__(self) -> None:
        types = ("bool", "float")
        self.buffers = {t: GraphBuffer() for t in types}

    def extend(self, obs: list[HeteroObsData]) -> None:
        for o in obs:
            for t in self.buffers:
                self.buffers[t].add_single(o.__getattribute__(t))

    def __iter__(self):
        return self.buffers.__iter__()

    def add_single_dict(self, obs: HeteroObsData) -> None:
        for t in self.buffers:
            self.buffers[t].add_single_dict(obs.__getattribute__(t))

    @property
    def batch(self) -> HeteroBatchData:
        return HeteroBatchData(
            boolean=batch(list(self.buffers["bool"].data)),
            numeric=batch(list(self.buffers["float"].data)),
        )

    def minibatch(self, indices: Iterable[int]) -> HeteroBatchData:
        return HeteroBatchData(
            boolean=batch([self.buffers["bool"].data[i] for i in indices]),
            numeric=batch([self.buffers["float"].data[i] for i in indices]),
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
        self, obs: HeteroObsData, action: tuple[int, int], reward: float
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


def batch(graphs: list[ObsData[V]]) -> BatchData[V]:
    num_graphs = len(graphs)
    total_factors = sum(g.n_factor for g in graphs)
    total_variables = sum(g.n_variable for g in graphs)
    total_edges = sum(g.senders.size for g in graphs)
    total_globals = sum(g.global_vars.size for g in graphs)

    g0 = graphs[0]

    var_value = np.empty((total_variables,), dtype=g0.var_value.dtype)
    var_type = np.empty((total_variables,), dtype=np.int64)
    var_batch = np.empty((total_variables,), dtype=np.int64)
    factor = np.empty((total_factors,), dtype=np.int64)
    factor_batch = np.empty((total_factors,), dtype=np.int64)
    senders = np.empty((total_edges,), dtype=np.int64)
    receivers = np.empty((total_edges,), dtype=np.int64)
    edge_attr = np.empty((total_edges,), dtype=np.int64)
    n_factor = np.empty((num_graphs,), dtype=np.int64)
    n_variable = np.empty((num_graphs,), dtype=np.int64)
    length = np.empty((total_variables,), dtype=np.int64)
    global_vars = np.empty((total_globals,), dtype=np.int64)
    global_vals = np.empty((total_globals,), dtype=g0.global_vals.dtype)
    global_length = np.empty((total_globals,), dtype=np.int64)
    global_batch = np.empty((total_globals,), dtype=np.int64)
    action_arity_mask = np.empty(
        (total_factors, g0.action_arity_mask.shape[1]), dtype=np.bool_
    )
    action_type_mask = np.empty(
        (total_factors, g0.action_type_mask.shape[1]), dtype=np.bool_
    )

    factor_offsets, variable_offsets, globals_offset = 0, 0, 0
    edge_offsets = 0
    for i, g in enumerate(graphs):
        var_len = g.n_variable
        fac_len = g.n_factor
        edge_len = g.senders.size
        globals_len = g.global_vars.size
        var_value[variable_offsets : variable_offsets + var_len] = g.var_value
        var_type[variable_offsets : variable_offsets + var_len] = g.var_type
        var_batch[variable_offsets : variable_offsets + var_len] = i
        length[variable_offsets : variable_offsets + var_len] = g.length
        factor[factor_offsets : factor_offsets + fac_len] = g.factor
        factor_batch[factor_offsets : factor_offsets + fac_len] = i
        senders[edge_offsets : edge_offsets + edge_len] = g.senders + variable_offsets
        receivers[edge_offsets : edge_offsets + edge_len] = g.receivers + factor_offsets
        edge_attr[edge_offsets : edge_offsets + edge_len] = g.edge_attr
        global_vars[globals_offset : globals_offset + globals_len] = g.global_vars
        global_vals[globals_offset : globals_offset + globals_len] = g.global_vals
        global_length[globals_offset : globals_offset + globals_len] = g.global_length
        global_batch[globals_offset : globals_offset + globals_len] = i
        action_arity_mask[factor_offsets : factor_offsets + fac_len] = (
            g.action_arity_mask
        )
        action_type_mask[factor_offsets : factor_offsets + fac_len] = g.action_type_mask
        n_factor[i] = fac_len
        n_variable[i] = var_len

        factor_offsets += fac_len
        variable_offsets += var_len
        edge_offsets += edge_len
        globals_offset += globals_len

    return BatchData(
        var_value=SparseArray(var_value, var_batch),
        var_type=SparseArray(var_type, var_batch),
        factor=SparseArray(factor, factor_batch),
        edge_attr=edge_attr,
        senders=senders,
        receivers=receivers,
        n_factor=n_factor,
        n_variable=n_variable,
        n_graphs=num_graphs,
        length=length,
        global_vars=SparseArray(global_vars, global_batch),
        global_vals=SparseArray(global_vals, global_batch),
        global_length=global_length,
        action_arity_mask=action_arity_mask,
        action_type_mask=action_type_mask,
    )


def obs_to_statedata(obs: ObsData) -> BatchData:
    return obslist_to_statedata([obs])


def obslist_to_statedata(obs: list[ObsData]) -> BatchData:
    return batch(obs)


attrs_from_obs = [
    attr for attr in ObsData._fields if attr not in ["n_factor", "n_variable"]
]


def heterostatedata(
    obs: list[HeteroObsData],
) -> HeteroBatchData:
    return HeteroBatchData(
        boolean=batch([o.bool for o in obs]),
        numeric=batch([o.float for o in obs]),
    )


def single_obs_to_heterostatedata(obs: HeteroObsData) -> HeteroBatchData:
    return heterostatedata([obs])


def heterostatedata_from_batched_obs(obs: list[HeteroObsData]) -> HeteroBatchData:
    return heterostatedata({k: batched_dict_to_obsdata(obs[k]) for k in obs})


def batched_hetero_dict_to_hetero_obs_list(
    obs: list[HeteroObsData],
) -> dict[str, tuple[ObsData, ...]]:
    return {k: batched_dict_to_obsdata(o) for o in obs}


def statedata_from_buffer(buf: list[tuple[ObsData, ...]]):
    return batch(list(chain(*buf)))


def heterostatedata_from_buffer(
    obs: dict[str, list[tuple[ObsData, ...]]],
) -> HeteroBatchData:
    return HeteroBatchData(
        boolean=statedata_from_buffer(obs["bool"]),
        numeric=statedata_from_buffer(obs["float"]),
    )


def heterostatedata_from_obslist(obs: list[HeteroObsData]) -> HeteroBatchData:
    boolean_data = list([d.bool for d in obs])
    numeric_data = list([d.float for d in obs])

    return HeteroBatchData(
        boolean=batch(boolean_data),
        numeric=batch(numeric_data),
    )


def heterostatedata_from_obslist_alt(obs: list[HeteroObsData]) -> HeteroBatchData:
    boolean_data = [d.bool for d in obs]
    numeric_data = [d.float for d in obs]

    return HeteroBatchData(
        boolean=batch(boolean_data),
        numeric=batch(numeric_data),
    )


class FactorGraph(NamedTuple):
    variables: SparseTensor
    factors: SparseTensor
    globals: SparseTensor
    v_to_f: Tensor
    f_to_v: Tensor
    edge_attr: Tensor
    n_variable: Tensor
    n_factor: Tensor
