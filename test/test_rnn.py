from gnn.gnn_embedder import RecurrentEmbedder
import torch.nn as nn
import itertools
import torch as th
from torch import as_tensor, isclose, all


def test_rnn():
    e = RecurrentEmbedder(4, 4, 4, nn.Mish())

    factors = [0, 1, 2, 3]
    var_values = [[1], [1, 1], [1, 1, 1]]

    var_types = [1, 1, 1, 2, 2, 2]

    lengths = [len(x) for x in var_values]

    var_values = list(itertools.chain(*var_values))

    embedded_vars, embedded_facs = e(
        as_tensor(var_values),
        as_tensor(var_types),
        as_tensor(factors),
        as_tensor(lengths),
    )
    print(embedded_vars)
    print(embedded_facs)

    print(e.predicate_embedding.embedding.weight)

    assert all(isclose(embedded_facs[0], th.zeros(4)))


if __name__ == "__main__":
    test_rnn()
