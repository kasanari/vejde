from typing import NamedTuple

import torch as th
from torch import Tensor
from torch import cumsum, ones_like, stack, concatenate, sum
import torch


class ObsData(NamedTuple):
    var_value: Tensor
    var_type: Tensor
    factor: Tensor
    edge_index: Tensor
    edge_attr: Tensor
    length: Tensor
    # globals: Tensor
    n_factor: int
    n_variable: int


class StateData(NamedTuple):
    var_value: Tensor
    var_type: Tensor
    factor: Tensor
    senders: Tensor  # variable
    receivers: Tensor  # factor
    edge_attr: Tensor
    batch: Tensor
    n_factor: Tensor
    n_variable: Tensor
    n_graphs: int
    length: Tensor


class StackedStateData(NamedTuple):
    var_value: Tensor
    var_type: Tensor
    factor: Tensor
    senders: Tensor
    receivers: Tensor
    edge_attr: Tensor
    batch: Tensor
    n_factor: Tensor
    n_variable: Tensor
    n_graphs: int
    length: Tensor


def batch(graphs: list[ObsData]) -> StateData:
    """Returns batched graph given a list of graphs and a numpy-like module."""
    # Calculates offsets for sender and receiver Tensors, caused by concatenating
    # the nodes Tensors.
    factor_offsets = cumsum(
        torch.as_tensor([0] + [g.n_factor for g in graphs[:-1]]), dim=0
    )
    variable_offsets = cumsum(
        torch.as_tensor([0] + [g.n_variable for g in graphs[:-1]]), dim=0
    )

    batch = concatenate([ones_like(g.factor) * i for i, g in enumerate(graphs)])

    return StateData(
        # n_node=stack([g.n_node for g in graphs]),
        var_value=concatenate([g.var_value for g in graphs]),
        var_type=concatenate([g.var_type for g in graphs]),
        factor=concatenate([g.factor for g in graphs]),
        edge_attr=concatenate([g.edge_attr for g in graphs]),
        # globals=_map_concat([g.globals for g in graphs]),
        senders=concatenate(
            [g.edge_index[0] + o for g, o in zip(graphs, variable_offsets)]
        ),
        receivers=concatenate(
            [g.edge_index[1] + o for g, o in zip(graphs, factor_offsets)]
        ),
        batch=batch,
        n_factor=torch.as_tensor([g.n_factor for g in graphs]),
        n_variable=torch.as_tensor([g.n_variable for g in graphs]),
        n_graphs=len(graphs),
        length=concatenate([g.length for g in graphs]),
    )


def statedata_from_single_obs(obs: dict[str, Tensor]) -> StateData:
    return statedata_from_obs([dict_to_data(obs)])


def statedata_from_obs(obs: list[ObsData]) -> StateData:
    return batch(obs)


def stackedstatedata_from_single_obs(obs: dict[str, Tensor]) -> StackedStateData:
    return stackedstatedata_from_obs([stacked_dict_to_data(obs)])


def stackedstatedata_from_obs(obs: list[ObsData]) -> StackedStateData:
    return batch(obs)


attrs_from_obs = [
    attr for attr in ObsData._fields if attr not in ["n_factor", "n_variable"]
]


def batched_dict_to_data(obs: dict[str, tuple[Tensor]]) -> list[ObsData]:
    def create_data(i: int) -> ObsData:
        data = {attr: Tensor(obs[attr][i]) for attr in attrs_from_obs}
        return ObsData(
            **data,
            n_factor=obs["factor"][i].shape[0],  # + obs["var_value"].shape[0]
            n_variable=obs["var_value"][i].shape[0],
        )

    return [create_data(i) for i in range(len(obs["factor"]))]


def dict_to_data(obs: dict[str, Tensor]) -> ObsData:
    data = {attr: torch.as_tensor(obs[attr]) for attr in attrs_from_obs}
    return ObsData(
        **data,
        n_factor=obs["factor"].shape[0],  # + obs["var_value"].shape[0]
        n_variable=obs["var_value"].shape[0],
    )


def stacked_dict_to_data(obs: dict[str, Tensor]) -> ObsData:
    data = {attr: torch.as_tensor(obs[attr]) for attr in attrs_from_obs}
    return ObsData(
        **data,
        n_factor=obs["factor"].shape[0],  # + obs["var_value"].shape[0]
        n_variable=len(
            obs["length"]
        ),  # lenght is used since var_values is flattened node histories
    )
