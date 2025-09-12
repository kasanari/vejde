from regawa.data import HeteroBatchData
from regawa.policy import AgentConfig, GraphAgent
from .symexp import symexp


import torch.nn as nn
from torch import Tensor


from typing import Any


class Agent(nn.Module):
    def __init__(
        self,
        agent: GraphAgent,
        **kwargs: dict[str, Any],
    ):
        super().__init__()  # type: ignore
        self.agent = agent

    def get_value(
        self,
        s: HeteroBatchData,
    ):
        value = self.agent.value(s)
        return symexp(value)

    def sample_action_and_value(self, s: HeteroBatchData):
        action, logprob, entropy, value, *_ = self.agent.sample(
            s,
        )
        return action, logprob, entropy, symexp(value)

    def evaluate_action_and_value(
        self,
        action: Tensor,
        s: HeteroBatchData,
    ) -> tuple[Tensor, Tensor, Tensor]:
        # num_graphs = batch_idx.max() + 1
        # action_mask = action_mask.reshape(num_graphs, -1)
        logprob, entropy, value, *_ = self.agent.forward(
            action,
            s,
            # num_graphs,
            # action_mask,
            # node_mask,
        )
        entropy = entropy.unsqueeze(0)
        return (
            logprob,
            entropy,
            value,
        )