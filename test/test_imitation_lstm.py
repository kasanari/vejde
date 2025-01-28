import json
import random


import gymnasium as gym
import numpy as np
import pytest
import torch as th
from gymnasium.spaces import Dict, MultiDiscrete


from regawa.gnn import ActionMode, Config, RecurrentGraphAgent


import regawa.model.utils as model_utils


from regawa.rl.util import evaluate, rollout, save_eval_data, update
from regawa.rddl import register_pomdp_env as register_env

import logging

logger = logging.getLogger("train")
logger.setLevel(logging.INFO)


class Serializer(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, th.Tensor):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.int64):
            return int(obj)
        return super().default(obj)


def knowledge_graph_policy(obs):
    nonzero = np.flatnonzero(obs["edge_attr"])
    if len(nonzero) == 0:
        return [0, 0]

    obj = next(iter(nonzero))

    obj = obs["edge_index"][obj][0]

    button = next(
        p for p, c in obs["edge_index"] if c == obj and not ((p, c) == (obj, obj))
    )

    return [1, button]


def policy(state: dict[str, bool]) -> tuple[int, int]:
    if (
        state["enough_light___r_m"]
        and state["light___r_m"]
        and not state["empty___r_m"]
    ):
        return (1, 4)

    if (
        state["enough_light___g_m"]
        and state["light___g_m"]
        and not state["empty___g_m"]
    ):
        return (1, 2)

    return (0, 0)


def counting_policy(state):
    if np.array(state["light_observed___r_m"], dtype=bool).sum() == 3:
        return [1, 3]

    if np.array(state["light_observed___g_m"], dtype=bool).sum() == 3:
        return [1, 1]

    return [0, 0]


def count_above_policy(state):
    if np.array(state["light_observed___r_m"], dtype=bool).sum() > 3:
        return [1, 3]

    if np.array(state["light_observed___g_m"], dtype=bool).sum() > 3:
        return [1, 1]

    return [0, 0]


@pytest.mark.parametrize(
    "action_mode, iterations",
    [
        (ActionMode.NODE_THEN_ACTION, 120),
        (ActionMode.ACTION_THEN_NODE, 120),
    ],
)
def test_imitation_rnn(action_mode: ActionMode, iterations: int):
    domain = "rddl/blink_enough_bandit.rddl"
    instance = "rddl/blink_enough_bandit_i0.rddl"

    # logfile = "test_imitation_rnn.log"
    # file_handler = logging.FileHandler(logfile)
    # logging.basicConfig(
    #     level=logging.INFO,
    # )
    # logger.addHandler(file_handler)

    logger.addHandler(logging.StreamHandler())

    th.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    env_id = register_env()
    env: gym.Env[Dict, MultiDiscrete] = gym.make(
        env_id,
        domain=domain,
        instance=instance,
        # add_inverse_relations=False,
        # types_instead_of_objects=False,
    )
    n_types = model_utils.n_types(env.observation_space)
    n_relations = model_utils.n_relations(env.observation_space)
    n_actions = model_utils.n_actions(env.action_space)

    config = Config(
        n_types,
        n_relations,
        n_actions,
        layers=3,
        embedding_dim=16,
        activation=th.nn.Mish(),
        aggregation="sum",
        action_mode=action_mode,
    )

    agent = RecurrentGraphAgent(config)

    optimizer = th.optim.AdamW(agent.parameters(), lr=0.01, amsgrad=True)

    data = [evaluate(env, agent, 0) for i in range(10)]
    rewards, _, _ = zip(*data)
    logger.info("Sum Reward Before Training: %s", np.mean([np.sum(r) for r in rewards]))

    num_seeds = 10

    losses = [iteration(i, env, agent, optimizer, 0) for i in range(iterations)]

    max_loss = 4e-6
    assert losses[-1] < max_loss, "Loss was too high: expected less than %s, got %s" % (
        max_loss,
        losses[-1],
    )

    pass

    data = [evaluate(env, agent, 0) for i in range(3)]
    rewards, _, _ = zip(*data)
    logger.info("Sum Reward After Training: %s", np.mean([np.sum(r) for r in rewards]))

    save_eval_data(data)

    agent.save_agent("blink_enough_bandit.pth")


def iteration(i, env, agent, optimizer, seed: int):
    r, length = rollout(env, seed, policy, 2.0)
    loss, grad_norm = update(agent, optimizer, r.actions, r.obs.batch)
    logger.info(f"{i} Loss: {loss:.3f}, Grad Norm: {grad_norm:.3f}, Length: {length}")
    return loss


if __name__ == "__main__":
    test_imitation_rnn(ActionMode.ACTION_THEN_NODE, 120)
