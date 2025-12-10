from torch import Tensor
from typing import NamedTuple

class PolicyOutput(NamedTuple):
	action: Tensor
	entropy: Tensor
	log_pi: Tensor
	value: Tensor
	p_a1: Tensor
	p_a2: Tensor