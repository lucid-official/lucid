"""Lambda-returns over an imagined trajectory.

Shared because both Dreamer families compute the same quantity — the
paper's TD(:math:`\\lambda`) target — and the second one arrived wanting a
per-step discount the first did not have.  One implementation, one place
to check the arithmetic against the paper.
"""

from collections.abc import Callable

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models.generative._common._rssm import RSSMState

__all__ = [
    "lambda_return",
    "prepare_imagination_starts",
    "imagination_weights",
    "pcont_bce_loss",
    "world_model_parameters",
    "isolated_world_actor_value_backward",
]


def lambda_return(
    reward: Tensor, value: Tensor, discount: float | Tensor, lambda_: float
) -> Tensor:
    r"""TD(:math:`\lambda`) returns over an imagined trajectory.

    Parameters
    ----------
    reward : Tensor
        Predicted reward at each imagined state, ``(N, H + 1)``.
    value : Tensor
        The critic's estimate at the same states, ``(N, H + 1)``.
    discount : float or Tensor
        :math:`\gamma`.  A scalar holds it constant; a ``(N, H + 1)``
        tensor is the *predicted* continuation probability at each state,
        which is how the paper handles episodes that can terminate — a
        state the agent will not survive discounts everything after it to
        nothing.
    lambda_ : float
        :math:`\lambda`; ``0`` gives the one-step TD target, ``1`` the full
        Monte-Carlo return bootstrapped at the horizon.

    Returns
    -------
    Tensor
        Targets ``(N, H)``, one per state except the last, which is only
        ever used as the bootstrap.

    Notes
    -----
    Computed by the backward recursion the released implementation uses,

    .. math::

        V_\lambda(s_t) = r_t + \gamma\big[(1 - \lambda)\, v(s_{t+1})
                          + \lambda\, V_\lambda(s_{t+1})\big],

    terminated at :math:`V_\lambda(s_H) = v(s_H)`.  Written this way it
    costs one pass rather than the :math:`O(H^2)` the closed form suggests,
    and each :math:`\lambda` power appears exactly once.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative._common._returns import lambda_return
    >>> rewards = lucid.zeros(2, 5)
    >>> values = lucid.zeros(2, 5)
    >>> lambda_return(rewards, values, 0.99, 0.95).shape
    (2, 4)

    One step shorter than the input: the last value has nothing after it
    to bootstrap from, so it is the bootstrap rather than a return.
    """
    horizon = int(reward.shape[1]) - 1
    if horizon < 1:
        raise ValueError(f"lambda-returns need at least two states, got {int(reward.shape[1])}")

    agg = value[:, horizon]
    out: list[Tensor] = []
    for t in range(horizon - 1, -1, -1):
        gamma = discount if isinstance(discount, float) else discount[:, t]
        inputs = reward[:, t] + gamma * (1.0 - lambda_) * value[:, t + 1]
        agg = inputs + gamma * lambda_ * agg
        out.append(agg)
    out.reverse()
    return lucid.stack(out, dim=1)


def prepare_imagination_starts(
    posteriors: RSSMState, *, has_pcont: bool
) -> tuple[RSSMState, RSSMState]:
    """Drop the final filtered step when ``has_pcont`` is set, then flatten and detach starts."""
    kept = posteriors
    if has_pcont:
        keep = int(posteriors.deter.shape[1]) - 1
        if keep < 1:
            raise ValueError(
                "pcont drops the last filtered step, so it needs a "
                f"sequence of at least 2, got {int(posteriors.deter.shape[1])}"
            )
        kept = posteriors.map(lambda x: x[:, :keep])
    batch = int(kept.deter.shape[0])
    steps = int(kept.deter.shape[1])
    start = kept.map(lambda x: x.reshape(batch * steps, *(int(v) for v in x.shape[2:])).detach())
    return kept, start


def imagination_weights(
    pcont: Tensor | None, discount: float, horizon: int, *, like: Tensor
) -> Tensor:
    """Compute detached cumulative continuation weights of shape ``(1, H)`` or ``(N, H)``."""
    if pcont is None:
        return lucid.tensor(
            [[discount**i for i in range(horizon)]], device=like.device, dtype=like.dtype
        )
    ones = lucid.ones((int(pcont.shape[0]), 1), device=pcont.device, dtype=pcont.dtype)
    running = lucid.cat([ones, pcont[:, : horizon - 1]], dim=1)
    return lucid.cumprod(running, dim=1).detach()


def pcont_bce_loss(
    pcont_head: nn.Module | None,
    predict_pcont: Callable[[RSSMState], Tensor],
    posteriors: RSSMState,
    discounts: Tensor | None,
    *,
    discount: float,
    scale: float,
) -> Tensor | None:
    """Compute the binary cross-entropy continuation loss when ``pcont_head`` is present."""
    if pcont_head is None:
        return None
    if discounts is None:
        raise ValueError(
            "pcont=True needs `discounts` — 1 where the episode "
            "continued past a step and 0 where it ended."
        )
    target = discount * discounts
    return scale * F.binary_cross_entropy_with_logits(predict_pcont(posteriors), target)


def world_model_parameters(
    encoder: nn.Module,
    rssm: nn.Module,
    decoder: nn.Module,
    reward_head: nn.Module,
    pcont_head: nn.Module | None = None,
) -> list[nn.Parameter]:
    """Return all parameters trained by the world-model objective across backbone and heads."""
    modules: list[nn.Module] = [encoder, rssm, decoder, reward_head]
    if pcont_head is not None:
        modules.append(pcont_head)
    return [p for module in modules for p in module.parameters()]


def isolated_world_actor_value_backward(
    module: nn.Module,
    *,
    world_loss: Tensor | None,
    actor_loss: Tensor | None,
    value_loss: Tensor | None,
    actor_params: list[nn.Parameter],
    value_params: list[nn.Parameter],
    model_name: str,
    wrapper_name: str,
) -> None:
    """Backpropagate actor, critic, and world losses in isolation onto their parameter groups."""
    if world_loss is None or actor_loss is None or value_loss is None:
        raise ValueError(
            "backward() needs the losses; this output came from "
            f"{model_name} rather than {wrapper_name}."
        )

    def take(params: list[nn.Parameter]) -> list[Tensor | None]:
        return [None if p.grad is None else p.grad.clone() for p in params]

    module.zero_grad()
    actor_loss.backward(retain_graph=True)
    actor_grads = take(actor_params)

    module.zero_grad()
    value_loss.backward(retain_graph=True)
    value_grads = take(value_params)

    module.zero_grad()
    world_loss.backward()

    for param, grad in zip(actor_params, actor_grads, strict=True):
        param.grad = grad
    for param, grad in zip(value_params, value_grads, strict=True):
        param.grad = grad
