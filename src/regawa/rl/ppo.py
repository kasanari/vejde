from regawa.data import TorchHeteroBatchData
from regawa.rl.agent import Agent
import torch as npl
from regawa.rl.types import BatchData, LossData, PPOParams, UpdateData
import logging
from torch import Tensor, nn, optim


logger = logging.getLogger(__name__)



@npl.no_grad()  # type: ignore
def approximate_kl(logprob_new: Tensor, logprob_old: Tensor) -> tuple[Tensor, Tensor]:
    # calculate approx_kl http://joschu.net/blog/kl-approx.html
    log_ratio = logprob_new - logprob_old
    ratio = npl.exp(log_ratio)
    old_approx_kl = npl.mean(-log_ratio)
    approx_kl = npl.mean((ratio - 1) - log_ratio)
    return old_approx_kl, approx_kl


def calculate_loss(agent: Agent, params: PPOParams):
    def f(s: TorchHeteroBatchData, b: BatchData):
        actions, logprob_old, advantages, returns, values_old, _, _ = b
        (
            clip_coef,
            norm_adv,
            clip_range_vf,
            ent_coef,
            vf_coef,
            _,
            _,
        ) = params

        logprob_new, entropy, values_new = agent.evaluate_action_and_value(
            actions,
            s,
            # npl.ones_like(obs.action_mask),
            # npl.ones_like(obs.node_mask),
        )
        assert not logprob_new.isinf().any()
        assert logprob_new.dim() == 1
        assert entropy.dim() == 1

        old_approx_kl, approx_kl = approximate_kl(logprob_new, logprob_old)

        if norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy loss
        ratio = npl.exp(logprob_new - logprob_old)
        pg_loss1 = advantages * ratio
        pg_loss2 = advantages * npl.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
        pg_loss = -npl.min(pg_loss1, pg_loss2).mean()
        clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean().item()

        # Value loss
        if clip_range_vf is None:
            # No clipping
            values_pred = values_new
        else:
            values_pred = values_old + npl.clamp(
                values_new - values_old, -clip_range_vf, clip_range_vf
            )

        # Value loss using the TD(gae_lambda) target
        value_loss = nn.functional.mse_loss(returns, values_pred)

        entropy_loss = entropy.mean()
        loss = pg_loss - ent_coef * entropy_loss + value_loss * vf_coef

        assert not npl.isnan(loss).any(), loss
        return LossData(
            loss,
            pg_loss,
            value_loss,
            entropy_loss,
            old_approx_kl,
            approx_kl,
            clipfrac,
        )

    return f


MAX_ALLOWED_GRAD_NORM = 100.0


def update(agent: Agent, optimizer: optim.Optimizer, params: PPOParams):
    loss_func = calculate_loss(agent, params)

    def _update(
        s: TorchHeteroBatchData,
        b: BatchData,
    ) -> UpdateData:
        loss = loss_func(s, b)

        assert not npl.isnan(loss.loss).any(), loss

        optimizer.zero_grad()
        loss.loss.backward()  # type: ignore

        # per_param_grad = {
        #         k: v.grad for k, v in dict(agent.named_parameters()).items()
        #     }
        # per_param_grad_norm = {k: v.norm().item() if v is not None else None for k, v in per_param_grad.items()}
        # sorted_per_param_grad = sorted(
        #         per_param_grad_norm.items(), key=lambda item: item[1] if item[1] is not None else -1, reverse=True
        # )

        grad_norm = nn.utils.clip_grad_norm_(
            agent.parameters(), params.max_grad_norm, error_if_nonfinite=True
        )

        stop_training = params.target_kl is not None and bool(
            (loss.approx_kl > 1.5 * params.target_kl).item()
        )

        if grad_norm.item() > MAX_ALLOWED_GRAD_NORM:
            logger.warning(f"grad_norm: {grad_norm.item()}")
            # logger.warning(f"per_param_grad: {per_param_grad}")

        optimizer.step()

        return UpdateData(
            loss,
            grad_norm,
            stop_training,
        )

    return _update


