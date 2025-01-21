from regawa.gnn.gnn_classes import EmbeddingLayer
from regawa.gnn.gnn_embedder import BooleanEmbedder, RecurrentEmbedder
import itertools
import torch as th
from torch import as_tensor, isclose, all


def test_factor_embedding():
    embedder = EmbeddingLayer(4, 4)
    factors = [0, 1, 2, 3]
    embedded_facs = embedder(as_tensor(factors))
    assert all(isclose(embedded_facs[0], th.zeros(4)))


def test_rnn():
    predicate_embedding = EmbeddingLayer(4, 4)
    base_embedder = BooleanEmbedder(4, predicate_embedding)
    e = RecurrentEmbedder(4, base_embedder)

    var_values = [[1], [1, 1], [1, 1, 1]]

    var_types = [1, 1, 1, 2, 2, 2]

    lengths = [len(x) for x in var_values]

    var_values = list(itertools.chain(*var_values))

    f = e(as_tensor(lengths))
    embedded_vars = f(as_tensor(var_values), as_tensor(var_types))
    print(embedded_vars)

    assert embedded_vars.shape == (3, 4)

    print(e.embedder.boolean_embedding.embedding.weight)


if __name__ == "__main__":
    test_rnn()
