import cProfile
from regawa.rl.ppo_gnn import setup, Args
from regawa import GNNParams, ActionMode
from torch import nn
from regawa.rddl import register_env, register_shuffle_env

env_id = register_shuffle_env()


def main():
    args = Args(
        env_id=env_id,
        total_timesteps=4000,
        num_steps=128,
        num_minibatches=2,
        learning_rate=0.001,
        # domain="rddl/conditional_bandit/domain.rddl",
        # instance="rddl/conditional_bandit/instance_1.rddl",
        domain="Tamarisk_MDP_ippc2014",
        instance=[1, 2, 3, 4, 5],
        clip_coef=0.3,
        ent_coef=0.1,
        remove_false=True,
        debug=True,
        num_envs=1,
        vf_coef=1.0,
        weight_decay=0.0,
        gae_lambda=0.95,
        update_epochs=10,
        max_grad_norm=1.0,
        agent_config=GNNParams(
            layers=8,
            embedding_dim=16,
            activation=nn.Tanh(),
            aggregation="max",
            action_mode=ActionMode.ACTION_THEN_NODE,
        ),
    )
    setup(args)


if __name__ == "__main__":
    with cProfile.Profile() as pr:
        main()
    pr.print_stats(sort="time")
