import logging
import random
from collections import deque

import numpy as np
import torch
import torch.optim
from torch_geometric.data import Batch, Data
from tqdm import tqdm

from gnn.bipartite_gnn import GraphActorCritic, obs_to_data
from wrappers.wrapper import GroundedRDDLGraphWrapper


def test_biparite():
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    instance = 1
    domain = "Elevators_MDP_ippc2011"
    # domain = "SysAdmin_MDP_ippc2011"
    env = GroundedRDDLGraphWrapper(domain, instance)
    gnn = GraphActorCritic(env.num_types, env.num_relations)
    optimizer = torch.optim.Adam(gnn.parameters(), lr=0.01)
    action_space = env.action_space
    action_space.seed(0)
    iterations = 1000

    pbar = tqdm(total=iterations)

    logging.basicConfig(level=logging.INFO)

    for j in range(iterations):
        obs, _ = env.reset(seed=0)
        data = obs_to_data(obs)
        step = 0
        obs_buf: deque[Data] = deque()
        actions: deque[int] = deque()
        rewards: deque[float] = deque()
        logprobs: deque[float] = deque()
        while True:
            step += 1
            b = Batch.from_data_list([data])
            # action, logprob, _, _ = gnn.forward(
            #     b.x_p, b.x_c, b.x_o, b.edge_index, b.edge_attr, b.batch
            # )
            action = action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            data = obs_to_data(next_obs)
            done = terminated or truncated

            obs_buf.append(data)
            actions.append(action)
            rewards.append(float(reward))
            logprobs.append(logprob)

            if done:
                break

        b = Batch.from_data_list(obs_buf)
        _, _, _, values = gnn.forward(
            b.x_p, b.x_c, b.x_o, b.edge_index, b.edge_attr, b.batch
        )
        returns = torch.zeros_like(values)
        reward_a = np.array(rewards)
        for i in reversed(range(len(reward_a))):
            returns[i] = reward_a[i:].sum()

        # advantages = returns - values

        # value_loss = advantages.pow(2).mean()
        value_loss = torch.zeros(1)

        policy_loss = torch.mul(
            torch.stack(list(logprobs)).squeeze(), returns.squeeze()
        )
        policy_loss = -policy_loss.mean()

        loss = value_loss

        pbar.update(1)
        pbar.set_description(
            f"{j}: {value_loss.item():.2f}, {np.array(rewards).sum():.2f}"
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # value_loss = advantages.round()

    np.set_printoptions(precision=2)
    print(value_loss.squeeze().detach().numpy())

    pass


if __name__ == "__main__":
    test_biparite()
