import math

from torch import Generator, nn
from torch.nn import init


def linear_reset_parameters(linear: nn.Linear, rng: Generator) -> nn.Linear:
    # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
    # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
    # https://github.com/pytorch/pytorch/issues/57109
    init.kaiming_uniform_(linear.weight, a=math.sqrt(5), generator=rng)
    if linear.bias is not None:
        fan_in, _ = init._calculate_fan_in_and_fan_out(linear.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(linear.bias, -bound, bound, generator=rng)

    return linear
