from collections.abc import Callable
from torch.nn.utils.rnn import PackedSequence

import torch.nn as nn
import torch.nn.init as init
from torch import Tensor, arange

import torch

from regawa.data.torch import SparseTensor


@torch.jit.script  # type: ignore
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


@torch.jit.script  # type: ignore
def _batch_sizes_from_lengths(lengths: Tensor) -> Tensor:
    # lengths: [B] long
    T = int(lengths.max().item())
    t = arange(T, device=lengths.device)  # [T]
    # batch_sizes[t] = #seqs with length > t
    batch_sizes = (t.unsqueeze(0) < lengths.unsqueeze(1)).sum(0).to(torch.long)  # [T]
    return batch_sizes.to("cpu")


@torch.jit.script  # type: ignore
def packed_from_concatenated_sequences(
    data: Tensor,
    lengths: Tensor,
    include_sort_info: bool = True,
):
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
    # Validate input
    if lengths.dim() != 1:
        raise ValueError("lengths must be a 1-D tensor.")
    if (lengths <= 0).any():
        raise ValueError("All sequence lengths must be > 0.")

    B = lengths.numel()
    N = int(lengths.sum().item())

    if data.size(0) != N:
        raise ValueError(f"data has {data.size(0)} rows, but sum(lengths)={N}.")

    # Handle the degenerate empty case explicitly (matches semantics cleanly)
    if B == 0:
        empty = torch.zeros(0, dtype=torch.long)
        return (
            PackedSequence(data, empty, empty.to(data.device), empty.to(data.device))
            if include_sort_info
            else PackedSequence(data, empty, None, None)
        )

    # --- 1) Sort sequences by length (desc) and build the inverse permutation (rank) ---
    sorted_indices = torch.argsort(lengths, descending=True)  # [B] sorted -> orig
    unsorted_indices = torch.empty_like(sorted_indices)  # [B] orig   -> sorted (rank)
    unsorted_indices.scatter_(0, sorted_indices, torch.arange(B, device=lengths.device))

    # --- 2) Batch sizes and per-time-step start offsets in the packed output ---
    lengths_sorted = lengths.index_select(0, sorted_indices)  # [B]
    batch_sizes = _batch_sizes_from_lengths(lengths_sorted)  # [T] # type: ignore
    time_offsets: Tensor = (
        batch_sizes.cumsum(0) - batch_sizes  # type: ignore
    )  # [T], start index of each time block

    # --- 3) For each input row, compute its (time_index, rank) in O(N) ---
    # start offset of each sequence in the concatenated input
    starts = lengths.cumsum(0) - lengths  # [B]
    # time index within its sequence (0..Li-1) without a Python loop
    time_index = torch.arange(N, device=lengths.device) - starts.repeat_interleave(
        lengths
    )  # [N]
    # sorted rank of the owning sequence for each row (orig -> sorted), repeated per row
    rank_per_row = unsorted_indices.repeat_interleave(lengths)  # [N]

    # --- 4) Directly place rows into packed order in O(N) (no global sort) ---
    # destination position for each input row in the packed output:
    #   pos = time_offsets[time_index] + rank_per_row
    dest = time_offsets.index_select(0, time_index) + rank_per_row  # [N] in [0, N)
    perm = torch.empty(N, dtype=torch.long, device=data.device)
    perm.scatter_(0, dest.to(data.device), torch.arange(N, device=data.device))

    packed_data = data.index_select(0, perm)

    if include_sort_info:
        return (
            packed_data,
            batch_sizes,
            sorted_indices.to(data.device),
            unsorted_indices.to(data.device),
        )
    else:
        return (packed_data, batch_sizes, None, None)


def compress_time(
    recurrent: Callable[[PackedSequence], tuple[Tensor, Tensor]],
    h: Tensor,
    length: Tensor,
) -> Tensor:
    custom_h_c = packed_from_concatenated_sequences(h, length, include_sort_info=True)
    _, variables = recurrent(custom_h_c)
    return variables


class RNNLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()  # type: ignore
        recurrent = nn.RNN(
            embedding_dim,
            embedding_dim,
            batch_first=True,
        )

        for name, param in recurrent.named_parameters():
            if "weight" in name:
                init.orthogonal_(param)  # type: ignore
            elif "bias" in name:
                init.zeros_(param)

        self.recurrent = recurrent  # type: ignore
        self.recurrent.to(device)
        self.recurrent.flatten_parameters()

    def forward(self, packed_sequence: PackedSequence) -> Tensor:
        _, variables = self.recurrent.forward(packed_sequence, None)
        return variables


class LSTMLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()  # type: ignore
        recurrent = nn.LSTM(
            embedding_dim,
            embedding_dim,
            batch_first=True,
        )

        for name, param in recurrent.named_parameters():
            if "weight" in name:
                init.orthogonal_(param)  # type: ignore
            elif "bias" in name:
                init.zeros_(param)

        self.recurrent = recurrent  # type: ignore
        self.recurrent.to(device)
        self.recurrent.flatten_parameters()

    def forward(self, packed_sequence: PackedSequence) -> Tensor:
        _, (variables, _) = self.recurrent.forward(packed_sequence, None)
        return variables


class GRULayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()  # type: ignore
        recurrent = nn.GRU(
            embedding_dim,
            embedding_dim,
            batch_first=True,
        )

        for name, param in recurrent.named_parameters():
            if "weight" in name:
                init.orthogonal_(param)  # type: ignore
            elif "bias" in name:
                init.zeros_(param)

        self.recurrent = recurrent  # type: ignore
        self.recurrent.to(device)
        self.recurrent.flatten_parameters()

    def forward(self, packed_sequence: PackedSequence) -> Tensor:
        _, variables = self.recurrent.forward(packed_sequence, None)
        return variables


class RecurrentEmbedder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        device: str | torch.device = "cpu",
    ):
        super().__init__()  # type: ignore
        self.recurrent = RNNLayer(embedding_dim, device)

    def compress_time(self, h: Tensor, length: Tensor) -> Tensor:
        custom_h_c = packed_from_concatenated_sequences(
            h, length, include_sort_info=True
        )
        packed_sequence = PackedSequence(*custom_h_c)
        return self.recurrent.forward(packed_sequence)

    def forward(
        self,
        h: SparseTensor,
        length: Tensor,
    ):
        variables = self.compress_time(h.values, length)

        return SparseTensor(variables.squeeze(0), compress_index_alt(h.indices, length))
