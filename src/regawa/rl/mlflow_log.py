import os
import time

import mlflow
import numpy as np
from numpy.typing import NDArray
from torch import Tensor, optim
from tqdm import tqdm

from regawa import BaseModel, GraphAgentInterface

from .types import BatchData, IterationCarry, RolloutData, UpdateData


def mlflow_log(
    artifact_name: str | None,
    learning_rate: float,
    u_data: list[UpdateData],
    total_loss: float,
    grad_norm: float,
    value_loss: float,
    pg_loss: float,
    entropy_loss: float,
    explained_var: float,
    return_scale: Tensor,
    carry: IterationCarry,
    b: BatchData,
    r: NDArray[np.float32] | None,
    length: NDArray[np.float32] | None,
    global_step: int,
    start_time: float,
    gradient_steps: int,
):
    mlflow.log_artifact(
        artifact_name, artifact_path="checkpoints"
    ) if artifact_name else None
    mlflow.log_metric("charts/learning_rate", learning_rate, global_step)
    mlflow.log_metric("rollout/return_scale", return_scale.item(), global_step)
    mlflow.log_metric(
        "rollout/return_scale_low", carry.low_ema.item(), global_step
    ) if carry.low_ema is not None else None
    mlflow.log_metric("charts/num_gradient_steps", gradient_steps, global_step)
    mlflow.log_metric(
        "rollout/return_scale_high", carry.high_ema.item(), global_step
    ) if carry.high_ema is not None else None
    mlflow.log_metric("rollout/mean_reward", b.rewards.mean().item(), global_step)

    mlflow.log_metric("rollout/num_resets", b.dones.sum().item(), global_step)

    if r is not None:
        mlflow.log_metric("rollout/mean_episodic_return", r.mean(), global_step)  # type: ignore
        mlflow.log_metric("rollout/max_episodic_return", r.max(), global_step)  # type: ignore
        mlflow.log_metric("rollout/min_episodic_return", r.min(), global_step)  # type: ignore
    if length is not None:
        mlflow.log_metric("rollout/mean_episodic_length", length.mean(), global_step)  # type: ignore
        mlflow.log_metric("rollout/max_episodic_length", length.max(), global_step)  # type: ignore
        mlflow.log_metric("rollout/min_episodic_length", length.min(), global_step)  # type: ignore

    mlflow.log_metric(
        "losses/total_loss",
        total_loss,  # type: ignore
        global_step,
    )
    mlflow.log_metric(
        "losses/grad_norm",
        grad_norm,  # type: ignore
        global_step,
    )  # type: ignore
    mlflow.log_metric("losses/value_loss", value_loss, global_step)  # type: ignore
    mlflow.log_metric(
        "losses/policy_loss",
        pg_loss,
        global_step,
    )  # type: ignore
    mlflow.log_metric(
        "losses/entropy",
        entropy_loss,
        global_step,
    )  # type: ignore
    mlflow.log_metric(
        "losses/old_approx_kl",
        np.mean([u.loss.old_approx_kl.item() for u in u_data]),
        global_step,
    )  # type: ignore
    mlflow.log_metric(
        "losses/approx_kl",
        np.mean([u.loss.approx_kl.item() for u in u_data]),
        global_step,
    )  # type: ignore
    mlflow.log_metric(
        "losses/clipfrac", np.mean([u.loss.clipfrac for u in u_data]), global_step
    )  # type: ignore
    mlflow.log_metric("losses/explained_variance", explained_var, global_step)  # type: ignore
    mlflow.log_metric(
        "charts/SPS", int(global_step / (time.time() - start_time)), global_step
    )
    mlflow.log_metric("rollout/advantage_mean", b.advantages.mean().item(), global_step)
    mlflow.log_metric(
        "rollout/return_targets_mean", b.returns.mean().item(), global_step
    )
    mlflow.log_metric(
        "rollout/predicted_values_mean", b.returns.mean().item(), global_step
    )


def logging_and_saving(
    agent: GraphAgentInterface,
    optimizer: optim.Optimizer,
    start_time: float,
    batch_size: int,
    run_name: str,
    iteration: int,
    r_data: RolloutData,
    u_data: list[UpdateData],
    explained_var: float,
    return_scale: Tensor,
    carry: IterationCarry,
    b: BatchData,
    checkpoint_period: int,
    highest_return: float,
    pbar: tqdm,
    model: BaseModel | None = None,
):
    artifact_name = None
    if checkpoint_period > 0 and iteration % checkpoint_period == 0:
        artifact_name = f"runs/{run_name}/checkpoint_{iteration * batch_size}.zip"
        agent.save_agent(artifact_name, model)
        # hard link to "checkpoint_latest.pth"
        latest_path = f"runs/{run_name}/checkpoint_latest.zip"
        if os.path.exists(latest_path):
            os.remove(latest_path)
        os.link(artifact_name, latest_path)
        mlflow.log_artifact(latest_path, artifact_path="checkpoints")  # type: ignore

        # use ema return scale to determine best model
        if carry.high_ema and carry.high_ema.item() > highest_return:
            new_highest_return = carry.high_ema.item()
            best_path = f"runs/{run_name}/checkpoint_best.zip"
            if os.path.exists(best_path):
                os.remove(best_path)
            os.link(artifact_name, best_path)
            mlflow.log_artifact(best_path, artifact_path="checkpoints")  # type: ignore
        else:
            new_highest_return = highest_return
    else:
        new_highest_return = highest_return

    r = float(np.mean(r_data.returns)) if r_data.returns else None
    length = float(np.mean(r_data.lengths)) if r_data.lengths else None

    loss_data = [u.loss for u in u_data]
    grad_norm = float(np.mean([u.grad_norm.item() for u in u_data]))
    total_loss = float(np.mean([u.loss.item() for u in loss_data]))
    entropy_loss = float(np.mean([u.entropy_loss.item() for u in loss_data]))
    value_loss = float(np.mean([u.v_loss.item() for u in loss_data]))
    pg_loss = float(np.mean([u.pg_loss.item() for u in loss_data]))

    disp_r = f"{r:.2f}" if r is not None else "N/A"
    disp_l = f"{length:.2f}" if length is not None else "N/A"
    desc = f"R:{disp_r} | L:{disp_l} | ENT:{entropy_loss:.2f} | V: {value_loss:.2f} | PG: {pg_loss:.2f} | EXPL_VARIANCE:{explained_var:.2f}"
    pbar.set_description(desc)
    pbar.update(1)

    if mlflow.active_run() is not None:  # type: ignore
        mlflow_log(
            artifact_name,
            optimizer.param_groups[0]["lr"],
            u_data,
            total_loss,
            grad_norm,
            value_loss,
            pg_loss,
            entropy_loss,
            explained_var,
            return_scale,
            carry,
            b,
            np.asarray(r_data.returns) if r_data.returns else None,
            np.asarray(r_data.lengths) if r_data.lengths else None,
            carry.global_step,
            start_time,
            carry.num_updates,
        )
    return new_highest_return
