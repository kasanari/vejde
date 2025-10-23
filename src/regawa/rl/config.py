from typing import Literal
from regawa import GNNParams


from dataclasses import dataclass


@dataclass
class Args:
    env_id: str
    agent_class: Literal["GraphAgent", "RecurrentGraphAgent"]
    agent_config: GNNParams
    resume_from: str | None = None
    multiprocess: bool = False
    """path to a model checkpoint to resume from"""
    debug: bool = False
    """the name of this experiment"""
    seed: int = 0
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `npl.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str | None = None
    """the entity (team) of wandb's project"""
    mlflow_tracking_uri: str = ""
    """the tracking uri for mlflow. If empty, mlflow will log locally"""
    # Algorithm specific arguments
    total_timesteps: int = 2000
    """total timesteps of the experiments"""
    learning_rate: float = 1.0e-2
    """the learning rate of the optimizer"""
    weight_decay: float = 0.1
    """the weight decay of the optimizer"""
    num_envs: int = 5
    """the number of parallel game environments"""
    num_steps: int = 20
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 1.0
    """the discount factor gamma"""
    gae_lambda: float = 0.0
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 10
    """the number of mini-batches"""
    update_epochs: int = 8
    """the K epochs to update the policy"""
    norm_adv: bool = False
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 1.0
    """coefficient of the value function"""
    max_grad_norm: float = 1.0
    """the maximum norm for the gradient clipping"""
    target_kl: float | None = None
    """the target KL divergence threshold"""
    checkpoint_period: int = 0
    """period in number of iterations to save a checkpoint, 0 means no checkpoint"""
