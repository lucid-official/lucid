"""Lambda-returns over an imagined trajectory.

Shared because both Dreamer families compute the same quantity — the
paper's TD(:math:`\\lambda`) target — and the second one arrived wanting a
per-step discount the first did not have.  One implementation, one place
to check the arithmetic against the paper.
"""

import lucid
from lucid._tensor.tensor import Tensor

__all__ = ["lambda_return"]


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
