from regawa.rl.ppo_gnn import setup, Args
from regawa import GNNParams, ActionMode
from torch import nn
from regawa.rddl import register_env

env_id = register_env()
args = Args(
    env_id=env_id,
    total_timesteps=2000,
    num_steps=20,
    domain="rddl/conditional_bandit/domain.rddl",
    instance="rddl/conditional_bandit/instance_1.rddl",
    remove_false=True,
    debug=True,
    agent_config=GNNParams(
        layers=4,
        embedding_dim=16,
        activation=nn.Mish(),
        aggregation="max",
        action_mode=ActionMode.ACTION_THEN_NODE,
    ),
)
setup(args)
