from collections.abc import Callable
from torch.nn.utils.rnn import PackedSequence
from regawa.embedding.node_embedders import logger
from torch import long

import torch.nn as nn
import torch.nn.init as init
from torch import Tensor, arange, argsort, cat, repeat_interleave


def _batch_sizes_from_lengths(lengths: Tensor) -> Tensor:
    # lengths: [B] long
    T = int(lengths.max().item())
    t = arange(T, device=lengths.device)  # [T]
    # batch_sizes[t] = #seqs with length > t
    batch_sizes = (t.unsqueeze(0) < lengths.unsqueeze(1)).sum(0).to(long)  # [T]
    return batch_sizes.to("cpu")


def packed_from_concatenated_sequences(
    data: Tensor,
    lengths: Tensor,
    *,
    include_sort_info: bool = True,
) -> PackedSequence:
    """
    Build a PackedSequence when rows are concatenated per sequence (sequence-major order).
    Example row order: [s0:t0, s0:t1, ..., s0:tL0-1, s1:t0, ..., sN-1:tL(N-1)-1]

    Args:
        data: Tensor of shape [sum(lengths), *feat]
        lengths: 1-D ints, one per sequence
        include_sort_info: attach (sorted_indices, unsorted_indices)

    Returns:
        nn.utils.rnn.PackedSequence
    """
    device = data.device
    if (lengths <= 0).any():
        raise ValueError("All sequence lengths must be > 0.")

    B = lengths.numel()
    N = int(lengths.sum().item())

    if data.size(0) != N:
        raise ValueError(f"data has {data.size(0)} rows, but sum(lengths)={N}.")

    # Map each row -> which sequence it came from
    row_to_seq = repeat_interleave(arange(B), lengths)  # [N]

    # Time index inside its sequence (0..length-1), following input order
    # Since data is sequence-major, this is just [0..L0-1, 0..L1-1, ...]
    time_index = cat([arange(int(L)) for L in lengths])  # [N]

    # Sort sequences by length (desc) to make a canonical pack order
    sorted_indices = argsort(lengths, descending=True)  # [B] sorted->orig
    unsorted_indices = argsort(sorted_indices)  # [B] orig->sorted (rank)
    rank = unsorted_indices[row_to_seq]  # [N] seq rank used within each time step

    # Reorder rows to time-major: lexicographic by (time_index, rank)
    B_val = B if B > 0 else 1
    key = time_index * B_val + rank
    perm = argsort(key)  # [N]
    packed_data = data.index_select(0, perm.to(device))  # [N, *feat]

    # Build batch_sizes
    batch_sizes = _batch_sizes_from_lengths(lengths.index_select(0, sorted_indices))

    return (
        PackedSequence(
            packed_data,
            batch_sizes,
            sorted_indices.to(device),
            unsorted_indices.to(device),
        )
        if include_sort_info
        else PackedSequence(packed_data, batch_sizes)
    )


def compress_time(recurrent: Callable[[PackedSequence], tuple[Tensor, Tensor]], h: Tensor, length: Tensor) -> Tensor:
    custom_h_c = packed_from_concatenated_sequences(h, length, include_sort_info=True)
    _, variables = recurrent(custom_h_c)
    return variables


class RecurrentEmbedder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
    ):
        super().__init__()  # type: ignore

        self.recurrent = nn.RNN(
            embedding_dim,
            embedding_dim,
            batch_first=True,
        )

        for name, param in self.recurrent.named_parameters():
            if "weight" in name:
                init.orthogonal_(param)  # type: ignore
            elif "bias" in name:
                init.zeros_(param)

    def forward(
        self,
        h: Tensor,
        length: Tensor,
    ):
        logger.debug("h:\n%s", h)

        variables = compress_time(self.recurrent, h, length) if h.shape[0] > 0 else h

        logger.debug("variables:\n%s", variables)

        return variables.squeeze(0)
