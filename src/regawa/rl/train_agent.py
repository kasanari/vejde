# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import logging
import os
from functools import partial

from regawa.rl.mlflow_log import logging_and_saving

from .iteration_func import iteration_step
from .ppo import update
from .rollout_func import rollout
from .update_func import update_step

os.environ["DO_NOT_TRACK"] = "true"
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, NamedTuple, TypeVar

import gymnasium as gym
import mlflow  # type: ignore
import numpy as np
import torch
import tyro
from gymnasium.spaces import Dict, MultiDiscrete
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from torch import Generator, optim
from tqdm import tqdm

from regawa import (
    GraphAgent,
    GraphAgentInterface,
    RecurrentGraphAgent,
    agent_from_env,
    load_agent,
)

from . import lambda_return
from .agent import Agent
from .config import Args, ConcurrencySetting
from .gae import gae
from .types import (
    BatchData,
    IterationCarry,
    PPOParams,
)

logger = logging.getLogger(__name__)

V = TypeVar("V", np.float32, np.bool_, np.int64)
npl = torch


def make_env(
    env_id: str,
):
    def thunk() -> gym.Env[Dict, MultiDiscrete]:
        env: gym.Env[Dict, MultiDiscrete] = gym.make(  # type: ignore
            env_id,
        )
        return gym.wrappers.RecordEpisodeStatistics(env)

    return thunk


def main(
    envs: gym.vector.SyncVectorEnv | gym.vector.AsyncVectorEnv,
    run_name: str,
    args: Args,
    device: str | npl.device,
    graph_agent: GraphAgentInterface,
    np_rng: np.random.Generator,
    torch_rng: Generator,
) -> GraphAgentInterface:
    batch_size = int(args.num_envs * args.rollout_length)
    minibatch_size = args.minibatch_size
    assert batch_size % minibatch_size == 0, (
        "batch_size must be divisible by minibatch_size"
    )
    num_rollouts = 0

    if args.total_timesteps:
        num_rollouts = args.total_timesteps // batch_size

    if args.total_updates:
        inner_updates = (batch_size // minibatch_size) * args.update_epochs
        assert args.total_updates % inner_updates == 0, (
            "total_updates must be multiple of (batch_size / minibatch_size) * update_epochs"
        )
        num_rollouts = args.total_updates // inner_updates

    pbar = tqdm(total=num_rollouts)
    checkpoint_period = args.checkpoint_period // batch_size
    start_time = time.time()

    mlflow.log_params(  # type: ignore
        {
            "batch_size": batch_size,
            "minibatch_size": minibatch_size,
            "num_iterations": num_rollouts,
        }
    )

    agent = Agent(graph_agent)
    agent = agent.to(device)
    optimizer = optim.AdamW(
        agent.parameters(),
        lr=args.learning_rate,
        eps=1e-5,
        amsgrad=True,
        weight_decay=args.weight_decay,
    )
    iter_step_func = iteration_step(
        args.anneal_lr,
        agent,
        batch_size,
        args.update_epochs,
        envs,
        optimizer,
        args.learning_rate,
        num_rollouts,
        rollout(
            agent,
            envs,
            args.rollout_length,
            args.num_envs,
            rng=torch_rng,
            device=device,
        ),
        gae(args.rollout_length, args.gamma, args.gae_lambda, device),
        update_step(
            batch_size,
            minibatch_size,
            device=device,
            update_func=update(
                agent,
                optimizer,
                PPOParams(
                    args.clip_coef,
                    args.norm_adv,
                    args.clip_vloss,
                    args.ent_coef,
                    args.vf_coef,
                    args.max_grad_norm,
                    args.target_kl,
                ),
            ),
            rng=np_rng,
        ),
        lambda_return.lambda_returns(args.gamma, args.gae_lambda),
        device=device,
        ema_decay=args.ema_decay,
    )

    actions = npl.zeros(
        (args.rollout_length, args.num_envs, *envs.single_action_space.shape)  # type: ignore
    ).to(device)
    b = BatchData(
        actions,
        npl.zeros((args.rollout_length, args.num_envs)).to(device),
        npl.zeros((args.rollout_length, args.num_envs)).to(device),
        npl.zeros((args.rollout_length, args.num_envs)).to(device),
        npl.zeros((args.rollout_length, args.num_envs)).to(device),
        npl.zeros((args.rollout_length, args.num_envs)).to(device),
        npl.zeros((args.rollout_length, args.num_envs)).to(device),
    )

    next_obs, _ = envs.reset(seed=args.seed)  # type: ignore
    carry = IterationCarry(
        b,
        next_obs,
        npl.zeros(args.num_envs).to(device),
        0,
        0,
    )
    highest_return: float | None = -float("inf")
    try:
        for iteration in range(1, num_rollouts + 1):
            (
                r_data,
                u_data,
                explained_var,
                return_scale,
                carry,
            ) = iter_step_func(iteration, carry)
            highest_return = logging_and_saving(
                agent.agent,
                optimizer,
                start_time,
                batch_size,
                run_name,
                iteration,
                r_data,
                u_data,
                explained_var,
                return_scale,
                carry,
                carry.b,
                checkpoint_period,
                highest_return,
                pbar,
            )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user. Returning agent as is...")
        return agent.agent
    finally:
        try:
            envs.close()
        except Exception as e:
            logger.warning(f"Exception while closing environments: {e}")
    return agent.agent


def create_run_folder(run_name: str) -> Path:
    runs_folder = Path("runs")
    runs_folder.mkdir(exist_ok=True)
    run_folder = runs_folder / run_name
    run_folder.mkdir(exist_ok=True)
    return run_folder


AGENT_CLASSES: dict[str, GraphAgent | RecurrentGraphAgent] = {
    c.__name__: c
    for c in [
        GraphAgent,
        RecurrentGraphAgent,
    ]
}


env_settings = {
    ConcurrencySetting.MULTI: partial(
        AsyncVectorEnv,
        shared_memory=False,
    ),
    ConcurrencySetting.SINGLE: SyncVectorEnv,
}


class TrainStats(NamedTuple):
    env_id: str
    config: dict[str, Any]
    run_name: str
    seed: int
    run_folder: Path


def train(
    args: Args | None = None,
) -> tuple[dict[str, int | str | Path], GraphAgentInterface]:
    args = tyro.cli(Args) if args is None else args
    random.seed(args.seed)
    npl.manual_seed(args.seed)  # type: ignore
    npl.backends.cudnn.deterministic = args.torch_deterministic
    torch.use_deterministic_algorithms(args.torch_deterministic)

    run_name = f"{args.env_id}__ppo"
    run_folder = create_run_folder(run_name)
    logger.addHandler(logging.FileHandler(run_folder / f"{run_name}.log", mode="w"))
    device = npl.device(
        "cuda:0" if npl.cuda.is_available() and args.cuda else npl.device("cpu")
    )
    np_rng = np.random.default_rng(args.seed)
    torch_rng = torch.Generator(device).manual_seed(args.seed)
    init_rng = torch.Generator("cpu").manual_seed(args.seed)
    logger.info(f"Using device: {device}")
    agent_class = AGENT_CLASSES[args.agent_class]

    envs = env_settings[args.multiprocess](
        [make_env(args.env_id) for _ in range(args.num_envs)]
    )

    if args.resume_from:
        agent, *_ = load_agent(agent_class, args.resume_from, device)
    else:
        agent = agent_from_env(agent_class, envs, args.agent_config, device, init_rng)

    logged_config = vars(args) | asdict(agent.config)

    try:
        trained_agent = main(
            envs, run_name, args, device, agent, np_rng=np_rng, torch_rng=torch_rng
        )
    except Exception as e:
        logger.exception("Exception during training:")
        raise e

    stats = TrainStats(
        env_id=args.env_id,
        config=logged_config,
        run_name=run_name,
        seed=args.seed,
        run_folder=run_folder,
    )

    return stats, trained_agent
