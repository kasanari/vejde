import torch
from torch import Tensor

def symlog(x: Tensor):
    # return x
    return torch.sign(x) * torch.log(1 + torch.abs(x))


def symexp(x: Tensor):
    # return x
    x = torch.clip(
        x, -20, 20
    )  # Clipped to prevent extremely rare occurence where critic throws a huge value
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)
