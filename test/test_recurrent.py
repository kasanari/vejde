from itertools import chain
import torch

from regawa.embedding.recurrent import packed_from_concatenated_sequences


def test_segmented_sort():
    lengths = torch.tensor([3, 2, 5, 4, 3], dtype=torch.long)
    indices = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    n_variables = torch.tensor([2, 3], dtype=torch.long)

    sort, new_indices = sort_segments(lengths, n_variables)
    assert torch.equal(sort, torch.tensor([5, 4, 3, 3, 2], dtype=torch.long))
    assert torch.equal(new_indices, torch.tensor([2, 3, 0, 4, 1], dtype=torch.long))

    lengths = torch.tensor([5, 3, 2, 4, 3], dtype=torch.long)
    indices = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    n_variables = torch.tensor([3, 2], dtype=torch.long)

    sort, new_indices = sort_segments(lengths, n_variables)
    assert torch.equal(sort, torch.tensor([5, 4, 3, 3, 2], dtype=torch.long))
    assert torch.equal(new_indices, torch.tensor([0, 3, 1, 4, 2], dtype=torch.long))

from torch import Tensor

def get_packed(h: Tensor, length: Tensor) -> Tensor:
    from torch.nn.utils.rnn import pack_padded_sequence
    from torch import zeros, long, roll, cumsum
    padded = zeros(
        length.size(0),
        length.max().item(),
        h.size(-1),
        device=h.device,
    )

    offsets = roll(cumsum(length, axis=0), 1, 0)
    offsets[0] = 0
    for i, node_l in enumerate(length):
        padded[i, : node_l.item()] = h[offsets[i] : offsets[i] + node_l.item()]

    return pack_padded_sequence(padded, length, batch_first=True, enforce_sorted=False)



def test_packed_from_concatenated_sequences():
    from torch import tensor
    from torch.nn.utils.rnn import PackedSequence
    lengths = torch.tensor([3, 2, 5, 4, 3], dtype=torch.long)
    data = torch.tensor(list(chain(*[[i] * i for i in lengths])), dtype=torch.float32).unsqueeze(-1)
    assert data.shape == (lengths.sum().item(), 1)
    indices = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    n_variables = torch.tensor([2, 3], dtype=torch.long)

    d = packed_from_concatenated_sequences(
        data,
        lengths, include_sort_info=True,
    )

    expected = get_packed(data, lengths)

    assert torch.equal(d.data, expected.data)
    assert torch.equal(d.batch_sizes, expected.batch_sizes)
    assert torch.equal(d.sorted_indices, expected.sorted_indices)
    assert torch.equal(d.unsorted_indices, expected.unsorted_indices)

    



if __name__ == "__main__":

    test_packed_from_concatenated_sequences()
    print("All tests passed.")
