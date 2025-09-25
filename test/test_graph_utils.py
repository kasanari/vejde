from regawa.data.torch import FactorGraph, SparseTensor
from regawa.embedding import merge_graphs


def test_merge_graph():
    from torch import tensor
    import torch

    boolean = FactorGraph(
        variables=SparseTensor(
            values=tensor(
                [
                    [-0.6177, 0.8340],
                    [-0.6218, 0.8374],
                    [-0.6218, 0.8374],
                    [-0.6177, 0.8340],
                ]
            ),
            indices=tensor([0, 0, 0, 0]),
        ),
        factors=SparseTensor(
            values=tensor(
                [
                    [0.0000, 0.0000],
                    [-1.0000, 1.0000],
                    [-1.0000, 1.0000],
                    [-1.0000, 1.0000],
                    [-1.0000, 1.0000],
                ]
            ),
            indices=tensor([0, 0, 0, 0, 0]),
        ),
        globals=SparseTensor(values=tensor([]), indices=tensor([], dtype=torch.int64)),
        v_to_f=tensor([0, 0, 1, 1, 2, 2, 3, 3]),
        f_to_v=tensor([2, 3, 2, 1, 4, 3, 4, 1]),
        edge_attr=tensor(
            [
                [1.0000, -1.0000],
                [1.0000, -1.0000],
                [1.0000, -1.0000],
                [1.0000, -1.0000],
                [1.0000, -1.0000],
                [1.0000, -1.0000],
                [1.0000, -1.0000],
                [1.0000, -1.0000],
            ]
        ),
        n_variable=tensor([4]),
        n_factor=tensor([5]),
    )
    numeric = FactorGraph(
        variables=SparseTensor(
            values=tensor(
                [
                    [-0.4635, 0.8672],
                    [-0.4635, 0.8672],
                    [-0.2378, 0.5640],
                    [-0.2378, 0.5640],
                ]
            ),
            indices=tensor([0, 0, 0, 0]),
        ),
        factors=SparseTensor(
            values=tensor(
                [
                    [0.0000, 0.0000],
                    [-1.0000, 1.0000],
                    [-1.0000, 1.0000],
                    [-1.0000, 1.0000],
                    [-1.0000, 1.0000],
                ]
            ),
            indices=tensor([0, 0, 0, 0, 0]),
        ),
        globals=SparseTensor(values=tensor([[-0.9062, 0.9993]]), indices=tensor([0])),
        v_to_f=tensor([0, 1, 2, 3]),
        f_to_v=tensor([3, 1, 3, 1]),
        edge_attr=tensor(
            [[1.0000, -1.0000], [1.0000, -1.0000], [1.0000, -1.0000], [1.0000, -1.0000]]
        ),
        n_variable=tensor([4]),
        n_factor=tensor([5]),
    )

    merged = merge_graphs(boolean, numeric)

    assert merged.n_variable.item() == 8
    assert merged.n_factor.item() == 5
    assert merged.variables.values.shape == (8, 2)
    assert merged.factors.values.shape == (5, 2)
    assert merged.globals.values.shape == (1, 2)

    assert merged.f_to_v.tolist() == [2, 3, 2, 1, 4, 3, 4, 1, 3, 1, 3, 1]
    assert merged.v_to_f.tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7]  # offset by 4
    assert merged.edge_attr.shape == (12, 2)
