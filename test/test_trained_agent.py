import logging

import gymnasium as gym
import torch

from regawa.gnn import GraphAgent, RecurrentGraphAgent


def test_rnn_agent():
    from test_imitation_lstm import evaluate, save_eval_data

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
    from test_imitation_mdp import evaluate, save_eval_data

    from regawa.rddl import register_env

    logfile = open("test_trained_agent.log", "w")

    logging.basicConfig(level=logging.DEBUG, format="%(message)s", stream=logfile)

    path = "SysAdmin_MDP_ippc2011__1__ppo_gnn__0.pth"
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
    data = [evaluate(env, agent, 0) for i in range(1)]
    save_eval_data(data)
    logfile.close()


if __name__ == "__main__":
    test_agent()
