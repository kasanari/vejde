import gymnasium as gym
import torch
from test_imitation_mdp import evaluate, save_eval_data

from gnn import Config, GraphAgent, RecurrentGraphAgent

import logging


def test_rnn_agent():
    from rddl import register_pomdp_env as register_env

    path = "blink_enough_bandit.pth"
    agent, config = RecurrentGraphAgent.load_agent(path)
    domain = "rddl/blink_enough_bandit.rddl"
    instance = "rddl/blink_enough_bandit_i0.rddl"
    env: gym.Env = gym.make(
        register_env(),
        domain=domain,
        instance=instance,
        # add_inverse_relations=False,
        # types_instead_of_objects=False,
    )
    torch.set_printoptions(precision=2, sci_mode=False)
    logfile = open("rnn_test.log", "w")
    logging.basicConfig(level=logging.DEBUG, format="%(message)s", stream=logfile)
    data = [evaluate(env, agent, 0) for i in range(1)]
    save_eval_data(data)
    logfile.close()


def test_agent():
    from rddl import register_env

    path = "conditional_bandit.pth"
    agent, config = GraphAgent.load_agent(path)
    domain = "rddl/conditional_bandit.rddl"
    instance = "rddl/conditional_bandit_i0.rddl"
    env: gym.Env = gym.make(
        register_env(),
        domain=domain,
        instance=instance,
        # add_inverse_relations=False,
        # types_instead_of_objects=False,
    )
    torch.set_printoptions(precision=2, sci_mode=False)
    logfile = open("test.log", "w")
    logging.basicConfig(level=logging.DEBUG, format="%(message)s", stream=logfile)
    data = [evaluate(env, agent, 0) for i in range(1)]
    save_eval_data(data)
    logfile.close()


if __name__ == "__main__":
    test_rnn_agent()
