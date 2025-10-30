import numpy as np
from torch import Tensor, stack
from functools import cache


@cache
def pos_emb(L: int, dim: int, k: int):
    if dim % 2 != 0:
        raise ValueError("Dimension must be even.")
    p, i = np.meshgrid(np.arange(float(L)), np.arange(dim / 2) * 2)
    theta = (p / k ** (i / dim)).T
    pos_emb = np.stack([np.sin(theta), np.cos(theta)], axis=-1)
    pos_emb = pos_emb.reshape((L, dim))  # (maxlen, dim)
    sin_freqs = np.repeat(pos_emb[..., ::2], repeats=2, axis=-1)
    cos_freqs = np.repeat(pos_emb[..., 1::2], repeats=2, axis=-1)
    return sin_freqs, cos_freqs


def minus_swap_alternate(x: Tensor):
    """
    [0., 1., 2., 3., 4., 5., 6., 7.]
    ->
    [-1., 0., -3., 2., -5., 4., -7., 6.]
    """
    return stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape(x.shape)


def rotate(
    x: Tensor,
    positions: Tensor,
    sin_freqs: Tensor,
    cos_freqs: Tensor,
):

    # (T, d)*(T, dq) + (T, dq)*(T, dq)
    x = x * cos_freqs[positions, :] + minus_swap_alternate(x) * sin_freqs[positions, :]
    return x  # (T, d)
