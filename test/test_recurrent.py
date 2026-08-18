from itertools import chain

import torch

# def test_segmented_sort():
#     lengths = torch.tensor([3, 2, 5, 4, 3], dtype=torch.long)
#     indices = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
#     n_variables = torch.tensor([2, 3], dtype=torch.long)
#     sort, new_indices = sort_segments(lengths, n_variables)
#     assert torch.equal(sort, torch.tensor([5, 4, 3, 3, 2], dtype=torch.long))
#     assert torch.equal(new_indices, torch.tensor([2, 3, 0, 4, 1], dtype=torch.long))
#     lengths = torch.tensor([5, 3, 2, 4, 3], dtype=torch.long)
#     indices = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
#     n_variables = torch.tensor([3, 2], dtype=torch.long)
#     sort, new_indices = sort_segments(lengths, n_variables)
#     assert torch.equal(sort, torch.tensor([5, 4, 3, 3, 2], dtype=torch.long))
#     assert torch.equal(new_indices, torch.tensor([0, 3, 1, 4, 2], dtype=torch.long))
from torch import Tensor, cumsum, roll, zeros
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence

from regawa.embedding.recurrent import packed_from_concatenated_sequences


def get_packed(h: Tensor, length: Tensor):

    padded = zeros(
        length.size(0),
        length.max().item(),
        h.size(-1),
        device=h.device,
    )

    offsets = roll(cumsum(length, axis=0), 1, 0)  # type: ignore
    offsets[0] = 0
    for i, node_l in enumerate(length):
        padded[i, : node_l.item()] = h[offsets[i] : offsets[i] + node_l.item()]

    return pack_padded_sequence(padded, length, batch_first=True, enforce_sorted=False)


def test_packed_from_concatenated_sequences():
    lengths = torch.tensor([3, 2, 5, 4, 3], dtype=torch.long)
    data = torch.tensor(
        list(chain(*[[i] * i for i in lengths])), dtype=torch.float32
    ).unsqueeze(-1)
    assert data.shape == (lengths.sum().item(), 1)
    # indices = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    # n_variables = torch.tensor([2, 3], dtype=torch.long)

    d = packed_from_concatenated_sequences(
        data,
        lengths,
        include_sort_info=True,
    )
    d = PackedSequence(*d)

    expected = get_packed(data, lengths)

    assert torch.equal(d.data, expected.data)
    assert torch.equal(d.batch_sizes, expected.batch_sizes)
    assert torch.equal(d.sorted_indices, expected.sorted_indices)
    assert torch.equal(d.unsorted_indices, expected.unsorted_indices)


def test_compress_index():
    indices = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2])
    lengths = torch.tensor([2, 3, 4])

    expected = torch.tensor([0, 1, 2])

    def compress_index_alt(data: Tensor, lengths: Tensor) -> Tensor:
        """
        data:    1D tensor of values laid out in consecutive segments
        lengths: 1D tensor of positive segment lengths (sum(lengths) == len(data))
        returns: 1D tensor with the first element from each segment
        """
        # Ensure indices are on the same device and of integer type
        lengths = lengths.to(device=data.device, dtype=torch.long)

        if lengths.numel() == 0:
            return data.new_empty((0,), dtype=data.dtype)

        # Start index of each segment: [cumsum(lengths) - lengths]
        offsets = torch.cumsum(lengths, dim=0) - lengths

        # Pick the first element of each segment
        return data.index_select(0, offsets)

    def compress_index(indices: Tensor, lengths: Tensor) -> Tensor:
        new_data = torch.empty(len(lengths), dtype=indices.dtype, device=indices.device)
        offsets = torch.cumsum(
            torch.cat((torch.tensor([0], device=indices.device), lengths[:-1])), 0
        )
        for i, o in enumerate(offsets):
            new_data[i] = indices[o : o + lengths[i]][0]
        return new_data

    assert torch.equal(compress_index(indices, lengths), expected)
    assert torch.equal(compress_index_alt(indices, lengths), expected)


if __name__ == "__main__":
    test_packed_from_concatenated_sequences()
    print("All tests passed.")
