from typing import NamedTuple

import torch as th
from torch import Tensor
from torch_geometric.data import Batch, Data  # type: ignore

StateData = NamedTuple(
    "StateData",
    [
        ("var_val", Tensor),
        ("var_type", Tensor),
        ("object_class", Tensor),
        ("edge_index", Tensor),
        ("edge_attr", Tensor),
        ("batch_idx", Tensor),
    ],
)

StackedStateData = NamedTuple(
    "StateData",
    [
        ("var_val", Tensor),
        ("var_type", Tensor),
        ("object_class", Tensor),
        ("edge_index", Tensor),
        ("edge_attr", Tensor),
        ("batch_idx", Tensor),
        ("lengths", Tensor),
    ],
)


class BipartiteData(Data):
    var_type: Tensor
    factor: Tensor

    def __inc__(self, key: str, value, *args, **kwargs):  # type: ignore
        if key == "edge_index":
            return th.tensor([[self.var_type.size(0)], [self.factor.size(0)]])
        return super().__inc__(key, value, *args, **kwargs)  # type: ignore


def statedata_from_single_obs(obs: dict[str, Tensor]) -> StateData:
    return statedata_from_obs([dict_to_data(obs)])


def statedata_from_obs(obs: list[Data]) -> StateData:
    b = Batch.from_data_list(obs)  # type: ignore
    return StateData(
        var_val=b.var_value,  # type: ignore
        var_type=b.var_type,  # type: ignore
        object_class=b.factor,  # type: ignore
        edge_index=b.edge_index,  # type: ignore
        edge_attr=b.edge_attr,  # type: ignore
        batch_idx=b.batch,  # type: ignore
    )


def stackedstatedata_from_single_obs(obs: dict[str, Tensor]) -> StackedStateData:
    return stackedstatedata_from_obs([stacked_dict_to_data(obs)])


def stackedstatedata_from_obs(obs: list[Data]) -> StackedStateData:
    b = Batch.from_data_list(obs)  # type: ignore
    return StackedStateData(
        var_val=b.var_value,  # type: ignore
        var_type=b.var_type,  # type: ignore
        object_class=b.factor,  # type: ignore
        edge_index=b.edge_index,  # type: ignore
        edge_attr=b.edge_attr,  # type: ignore
        batch_idx=b.batch,  # type: ignore
        lengths=b.length,  # type: ignore
    )


def batched_dict_to_data(
    obs: dict[str, tuple[Tensor]], num_envs: int
) -> list[BipartiteData]:
    return [
        BipartiteData(
            var_value=th.as_tensor(obs["var_value"][i], dtype=th.float32),
            var_type=th.as_tensor(obs["var_type"][i], dtype=th.int64),
            factor=th.as_tensor(obs["factor"][i], dtype=th.int64),
            edge_index=th.as_tensor(obs["edge_index"][i], dtype=th.int64).T,
            edge_attr=th.as_tensor(obs["edge_attr"][i], dtype=th.int64),
            num_nodes=obs["factor"][i].shape[0],  # + obs["var_value"].shape[0]
        )
        for i in range(num_envs)
    ]


def dict_to_data(obs: dict[str, Tensor]) -> BipartiteData:
    return BipartiteData(
        var_value=th.as_tensor(obs["var_value"], dtype=th.float32),
        var_type=th.as_tensor(obs["var_type"], dtype=th.int64),
        factor=th.as_tensor(obs["factor"], dtype=th.int64),
        edge_index=th.as_tensor(obs["edge_index"], dtype=th.int64).T,
        edge_attr=th.as_tensor(obs["edge_attr"], dtype=th.int64),
        num_nodes=obs["factor"].shape[0],  # + obs["var_value"].shape[0]
    )


def stacked_dict_to_data(obs: dict[str, Tensor]) -> BipartiteData:
    return BipartiteData(
        var_value=th.as_tensor(obs["var_value"], dtype=th.float32),
        var_type=th.as_tensor(obs["var_type"], dtype=th.int64),
        factor=th.as_tensor(obs["factor"], dtype=th.int64),
        edge_index=th.as_tensor(obs["edge_index"], dtype=th.int64).T,
        edge_attr=th.as_tensor(obs["edge_attr"], dtype=th.int64),
        length=th.as_tensor(obs["length"], dtype=th.int64),
        num_nodes=obs["factor"].shape[0],  # + obs["var_value"].shape[0]
    )
