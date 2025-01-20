from collections import deque
from collections.abc import Callable, Iterable
import json
import numpy as np
import torch as th
import gymnasium as gym

from regawa.gnn.data import Rollout, RolloutCollector, single_obs_to_heterostatedata
from torch import Tensor

from regawa.gnn.gnn_agent import GraphAgent
from torch.utils._foreach_utils import (
    _group_tensors_by_device_and_dtype,
)


@th.no_grad()
def evaluate(env: gym.Env, agent: GraphAgent, seed: int):
    obs, info = env.reset(seed=seed)
    done = False
    time = 0

    rewards = deque()
    obs_buf = deque()
    actions = deque()

    while not done:
        time += 1

        obs_buf.append(info["rddl_state"])

        s = single_obs_to_heterostatedata(obs)

        action, _, _, _ = agent.sample(
            s,
            deterministic=True,
        )

        next_obs, reward, terminated, truncated, info = env.step(action.squeeze(0))  # type: ignore

        done = terminated or truncated
        actions.append(info["rddl_action"])  # type: ignore
        rewards.append(float(reward))
        obs = next_obs

    return (
        list(rewards),
        list(obs_buf),
        list(actions),
    )


@th.no_grad()  # type: ignore
def grad_norm_(
    parameters: th.Tensor | Iterable[th.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
) -> th.Tensor:
    if isinstance(parameters, th.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(grads) == 0:
        return th.tensor(0.0)
    first_device = grads[0].device
    grouped_grads: dict[
        tuple[th.device, th.dtype], tuple[list[list[th.Tensor]], list[int]]
    ] = _group_tensors_by_device_and_dtype([grads])  # type: ignore[assignment]

    norms: list[th.Tensor] = []
    for (device, _), ([device_grads], _) in grouped_grads.items():  # type: ignore[assignment]
        norms.extend([th.linalg.vector_norm(g, norm_type) for g in device_grads])

    total_norm = th.linalg.vector_norm(
        th.stack([norm.to(first_device) for norm in norms]), norm_type
    )

    if error_if_nonfinite and th.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from "
            "`parameters` is non-finite, so it cannot be clipped. To disable "
            "this error and scale the gradients by the non-finite norm anyway, "
            "set `error_if_nonfinite=False`"
        )

    return total_norm


def compare_rollouts(r1: Rollout, r2: Rollout):
    assert r1.rewards == r2.rewards
    assert r1.actions == r2.actions
    for o1, o2 in zip(r1.obs, r2.obs):
        assert o1.keys() == o2.keys()
        for k in o1:
            if isinstance(o1[k], np.ndarray):
                o = o1[k].tolist()
            else:
                o = o1[k]

            assert o == o2[k], f"{o} != {o2[k]} for {k}"


def update(
    iteration: int,
    agent: GraphAgent,
    optimizer: th.optim.Optimizer,
    rollout: Rollout,
):
    _, obs, actions = rollout
    s = obs.batch
    # b = th.stack([d.var_value for d in obs])
    actions = th.as_tensor(actions, dtype=th.int32)
    logprob, _, _ = agent.forward(
        actions,
        s,
    )

    l2_norms = [th.sum(th.square(w)) for w in agent.parameters()]
    loss = calc_loss(l2_norms, logprob)

    optimizer.zero_grad()
    loss.backward()

    d = dict(agent.named_parameters())
    grads = {k: th.nonzero((-d[k].grad).clamp(min=0)) for k in d}
    grads = {k: v for k, v in grads.items() if len(v) > 0}

    # if iteration % 100 == 0:
    #     per_sample_grads = per_sample_grad(agent, b, actions)

    grad_norms = {
        k: round(grad_norm_(param, 1.0).item(), 3)
        for k, param in agent.named_parameters()
    }
    max_grad_norm = max(grad_norms.values())

    grad_norm = th.nn.utils.clip_grad_norm_(
        agent.parameters(), 1.0, error_if_nonfinite=True
    )

    optimizer.step()

    return loss.item(), grad_norm.item()


def calc_loss(l2_norms: list[Tensor], logprob: Tensor) -> th.Tensor:
    l2_weight = 0.0
    l2_norm = sum(l2_norms) / 2  # divide by 2 to cancel with gradient of square
    l2_loss = l2_weight * l2_norm
    loss = -logprob.mean() + l2_loss
    return loss


def rollout(
    env: gym.Env,
    seed: int,
    expert_policy: Callable[[dict[str, bool]], tuple[int, int]],
    expected_return: float,
) -> Rollout:
    obs, info = env.reset(seed=seed)
    done = False
    time = 0
    collector = RolloutCollector()
    while not done:
        action = expert_policy(info["rddl_state"])
        next_obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        collector.add_single(obs, action, float(reward))

        obs = next_obs
        time += 1

    if seed == 0:
        assert (
            collector.return_ == expected_return
        ), f"Expert policy failed: {collector.return_}"

    return collector.export(), time


class Serializer(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, th.Tensor):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.int64):
            return int(obj)
        return super().default(obj)


def save_eval_data(data):
    to_write = {
        f"ep_{i}": [
            {
                "reward": r,
                "obs": s,
                "action": a,
            }
            for r, s, a in zip(*episode)
        ]
        for i, episode in enumerate(data)
    }

    with open("evaluation.json", "w") as f:
        import json

        json.dump(to_write, f, cls=Serializer, indent=2)
