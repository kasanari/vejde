from typing import NamedTuple

import torch as th
from torch import Tensor
from torch_geometric.data import Batch, Data  # type: ignore


class StateData(NamedTuple):
    var_value: Tensor
    var_type: Tensor
    factor: Tensor
    edge_index: Tensor
    edge_attr: Tensor
    batch: Tensor
    length: Tensor
    # source_targets: (
    #     Tensor  # 2xN tensor of source and target nodes for next state prediction
    # )


class StackedStateData(NamedTuple):
    var_value: Tensor
    var_type: Tensor
    factor: Tensor
    edge_index: Tensor
    edge_attr: Tensor
    batch: Tensor
    length: Tensor


class BipartiteData(Data):
    var_type: Tensor
    factor: Tensor

    def __inc__(self, key: str, value, *args, **kwargs):  # type: ignore
        if key == "edge_index":
            return th.tensor([[self.length.size(0)], [self.factor.size(0)]])
        return super().__inc__(key, value, *args, **kwargs)  # type: ignore


def statedata_from_single_obs(obs: dict[str, Tensor]) -> StateData:
    return statedata_from_obs([dict_to_data(obs)])


def statedata_from_obs(obs: list[Data]) -> StateData:
    b = Batch.from_data_list(obs)  # type: ignore
    attrs = StateData._fields
    data = {attr: b[attr] for attr in attrs}
    return StateData(
        **data,
    )


def stackedstatedata_from_single_obs(obs: dict[str, Tensor]) -> StackedStateData:
    return stackedstatedata_from_obs([stacked_dict_to_data(obs)])


def stackedstatedata_from_obs(obs: list[Data]) -> StackedStateData:
    b = Batch.from_data_list(obs)  # type: ignore
    attrs = StackedStateData._fields
    data = {attr: b[attr] for attr in attrs}
    return StackedStateData(
        **data,
    )


def batched_dict_to_data(
    obs: dict[str, tuple[Tensor]], num_envs: int
) -> list[BipartiteData]:
    attrs = StackedStateData._fields

    def create_data(i: int) -> BipartiteData:
        data = {attr: th.as_tensor(obs[attr][i]) for attr in attrs}
        return BipartiteData(
            **data,
            num_nodes=obs["factor"][i].shape[0],  # + obs["var_value"][i].shape[0]
        )

    return [create_data(i) for i in range(num_envs)]


def dict_to_data(obs: dict[str, Tensor]) -> BipartiteData:
    attrs = StateData._fields
    data = {attr: th.as_tensor(obs[attr]) for attr in attrs if attr != "batch"}
    return BipartiteData(
        **data,
        num_nodes=obs["factor"].shape[0],  # + obs["var_value"].shape[0]
    )


def stacked_dict_to_data(obs: dict[str, Tensor]) -> BipartiteData:
    attrs = StackedStateData._fields
    data = {attr: th.as_tensor(obs[attr]) for attr in attrs if attr != "batch"}
    return BipartiteData(
        **data,
        num_nodes=obs["factor"].shape[0],  # + obs["var_value"].shape[0]
    )
