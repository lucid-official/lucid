r"""How the methods that prune during training rate what they might prune: smoothed sensitivity and its uncertainty, and a decaying budget.

AdaLoRA ranks singular triplets and AdaMSS subspaces by the same score.  After a backward pass each
trained entry :math:`w` has a sensitivity :math:`|w\,\partial L/\partial w|`; a moving average of it
says how much the entry matters, a second average of its deviation says how much that estimate
wanders, and their product is the entry's importance.  Both methods then let the number they keep
fall from where it starts to a target along a power of the training progress.  The averages and the
curve are here; what is ranked, and what is done to the losers, is each method's own.
"""

import lucid
from lucid._tensor.tensor import Tensor
from lucid.nn.parameter import Parameter


class _Sensitivity:
    """The moving averages of one parameter's sensitivity :math:`|w\\,g|` and of its uncertainty.

    ``score`` — their product — is what ranks the entries.  The state is allocated on first use
    and again if the parameter's shape changes (a layer was pruned or resized under it).
    """

    __slots__ = ("_mean", "_spread")

    def __init__(self) -> None:
        self._mean: Tensor | None = None
        self._spread: Tensor | None = None

    def update(
        self, parameter: Parameter, beta1: float, beta2: float, *, from_previous_mean: bool = False
    ) -> None:
        """Fold in the sensitivity of ``parameter`` at its current value and gradient.

        The uncertainty averages how far each new sensitivity is from the sensitivity average:
        the one just updated with it (AdaLoRA), or the one it is about to update (AdaMSS,
        ``from_previous_mean=True``).
        """
        grad = parameter.grad
        if grad is None:
            raise RuntimeError(
                "update_and_allocate needs the gradients of the adapter: call it after "
                "loss.backward() and before the gradients are cleared"
            )
        with lucid.no_grad():
            sensitivity = (parameter * grad).abs()
            mean, spread = self._mean, self._spread
            if mean is None or spread is None or mean.shape != sensitivity.shape:
                mean, spread = lucid.zeros_like(sensitivity), lucid.zeros_like(sensitivity)
            smoothed = beta1 * mean + (1 - beta1) * sensitivity
            reference = mean if from_previous_mean else smoothed
            self._spread = beta2 * spread + (1 - beta2) * (sensitivity - reference).abs()
            self._mean = smoothed

    @property
    def score(self) -> Tensor:
        """Smoothed sensitivity times smoothed uncertainty, shaped like the parameter."""
        if self._mean is None or self._spread is None:
            raise RuntimeError("no sensitivity has been recorded yet")
        return self._mean * self._spread


def _decayed_budget(start: int, end: int, remaining: float, exponent: float) -> int:
    """What a budget falling from ``start`` to ``end`` is when ``remaining`` of its ramp is left: ``end + (start - end) * remaining ** exponent``, truncated.

    ``remaining`` is 1 where the ramp begins and 0 where it ends.
    """
    return int((start - end) * remaining**exponent + end)
