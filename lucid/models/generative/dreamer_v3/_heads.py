r"""Distributional regression heads — predict a distribution, read a scalar.

DreamerV3 predicts reward and value as categorical distributions over
exponentially spaced bins rather than as numbers.  Two reasons, and the
first is the one the paper is built on.

A squared error's gradient is proportional to the error, so a head fitted
to rewards around ``1`` and a head fitted to rewards around ``10000``
receive gradients four orders of magnitude apart, and no single learning
rate suits both.  A cross-entropy's gradient is bounded regardless, which
is what lets one configuration span domains.

The second is representational: the mean of a squared-error head is a
single number, so a genuinely bimodal outcome — a value that is either 0
or 10 — is fitted as 5, which never happens.  A distribution can hold
both and still report 5 when asked for a point estimate.

Kept private to the family.  It is a general idea and a reasonable
candidate for promotion, but it has one consumer.
"""

from typing import final, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models.generative._common._pixel_nets import DenseHead

__all__ = ["TwoHotHead"]


@final
class TwoHotHead(nn.Module):
    r"""A dense trunk whose output is a distribution over symlog bins.

    Parameters
    ----------
    in_features : int
        Width of the latent it reads.
    hidden, layers : int
        Shape of the dense trunk.
    num_bins : int, default=255
        Bins in the output distribution.
    bin_range : float, default=20.0
        The grid spans ``[-bin_range, bin_range]`` **in symlog space**, so
        it reaches ``symexp(bin_range)`` in the target's own units —
        about 5e8 at the default, which is why one grid covers every
        domain the paper reports.
    act_fn : str, default="silu"
        Activation in the trunk.
    zero_init : bool, default=False
        Start the output layer at zero, so the head predicts exactly
        ``symexp(0) = 0`` before it has learned anything.  The paper does
        this for the critic: a value function that begins by asserting
        large returns sends the actor chasing them.

    Notes
    -----
    The grid is uniform in symlog space and therefore exponential in the
    target's, which is the whole trick — near zero the bins are dense and
    far out they are sparse, matching where precision is worth spending.

    Reading a scalar back is the grid's expectation put through
    ``symexp``.  Note this is *not* the same as the mode: a bimodal
    prediction reports its mean, which is the answer to "what should I
    expect", the question a value function is asked.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.dreamer_v3._heads import TwoHotHead
    >>> head = TwoHotHead(8, 16, 2, num_bins=41)
    >>> head.predict(lucid.zeros((2, 3, 8))).shape
    (2, 3)
    """

    bins: Tensor

    def __init__(
        self,
        in_features: int,
        hidden: int,
        layers: int,
        num_bins: int = 255,
        bin_range: float = 20.0,
        act_fn: str = "silu",
        zero_init: bool = False,
    ) -> None:
        """Initialise the head. See the class docstring for parameters."""
        super().__init__()
        if num_bins < 2:
            raise ValueError(
                f"num_bins must be at least 2 — a two-hot interpolates "
                f"between a pair; got {num_bins}"
            )
        if bin_range <= 0.0:
            raise ValueError(f"bin_range must be positive, got {bin_range}")

        self.num_bins = num_bins
        self.trunk = DenseHead(in_features, hidden, layers, out_features=num_bins, act_fn=act_fn)
        if zero_init:
            with lucid.no_grad():
                self.trunk.out.weight[:] = lucid.zeros_like(self.trunk.out.weight)
                if self.trunk.out.bias is not None:
                    self.trunk.out.bias[:] = lucid.zeros_like(self.trunk.out.bias)

        self.register_buffer(
            "bins", lucid.linspace(-bin_range, bin_range, num_bins), persistent=False
        )

    @property
    def grid(self) -> Tensor:
        """The bin locations, narrowed from the buffer registry."""
        return self.bins

    @override
    def forward(self, feature: Tensor) -> Tensor:
        """Return the raw bin logits — ``(B, T, num_bins)``.

        Parameters
        ----------
        feature : Tensor
            Latent state, ``(B, T, in_features)``.

        Returns
        -------
        Tensor
            Unnormalised scores over the grid.
        """
        return self.trunk(feature)

    def predict(self, feature: Tensor) -> Tensor:
        """Read the distribution back as a scalar — ``(B, T)``.

        Parameters
        ----------
        feature : Tensor
            Latent state, ``(B, T, in_features)``.

        Returns
        -------
        Tensor
            ``symexp`` of the grid's expectation, in the target's units.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models.generative.dreamer_v3._heads import TwoHotHead
        >>> head = TwoHotHead(8, 16, 2, num_bins=41)
        >>> head.predict(lucid.randn((2, 3, 8))).shape
        (2, 3)

        A zero-initialised head is uniform over a grid symmetric about zero,
        so it predicts ``symexp(0) = 0``, up to the rounding of the grid's
        float32 sum — which is how the critic starts:

        >>> critic = TwoHotHead(8, 16, 2, num_bins=41, zero_init=True)
        >>> float(critic.predict(lucid.randn((2, 3, 8))).abs().max().item()) < 1e-6
        True
        """
        probabilities = F.softmax(self(feature), dim=-1)
        return F.symexp((probabilities * self.grid).sum(dim=-1))

    def cross_entropy(self, feature: Tensor, target: Tensor) -> Tensor:
        """The loss before it is averaged — ``(B, T)``.

        Parameters
        ----------
        feature : Tensor
            Latent state, ``(B, T, in_features)``.
        target : Tensor
            Scalars to fit, ``(B, T)``, in their own units.

        Returns
        -------
        Tensor
            One cross-entropy per step.

        Notes
        -----
        Separate from :meth:`loss` because the critic weights its steps by
        the probability the episode has survived that far, and a mean
        taken first cannot be weighted afterwards.

        The target is detached: it is data, and a head that could move its
        own target would fit nothing.  The encoding is a soft label rather
        than a class index, so this is a cross-entropy over a distribution
        and not a classification.

        Examples
        --------
        >>> import math
        >>> import lucid
        >>> from lucid.models.generative.dreamer_v3._heads import TwoHotHead
        >>> head = TwoHotHead(8, 16, 2, num_bins=41, zero_init=True)
        >>> target = lucid.tensor([[0.0, 1.0, -5.0], [1e4, 0.5, 2.0]])
        >>> per_step = head.cross_entropy(lucid.randn((2, 3, 8)), target)
        >>> per_step.shape
        (2, 3)

        An untrained head is uniform over the bins, so every target costs
        ``log(num_bins)`` — a reward of 10000 no more than one of 0.5:

        >>> sorted({round(v, 4) for v in per_step.reshape(-1).tolist()})
        [3.7136]
        >>> round(math.log(41), 4)
        3.7136
        """
        encoded = F.two_hot(F.symlog(target.detach()), self.grid)
        logits = F.log_softmax(self(feature), dim=-1)
        return -(encoded * logits).sum(dim=-1)

    def loss(self, feature: Tensor, target: Tensor) -> Tensor:
        """Cross-entropy against the two-hot encoding of ``symlog(target)``.

        Parameters
        ----------
        feature : Tensor
            Latent state, ``(B, T, in_features)``.
        target : Tensor
            Scalars to fit, ``(B, T)``, in their own units.

        Returns
        -------
        Tensor
            A scalar, averaged over the batch and time.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models.generative.dreamer_v3._heads import TwoHotHead
        >>> head = TwoHotHead(8, 16, 2, num_bins=41)
        >>> feature = lucid.randn((2, 3, 8))
        >>> target = lucid.randn((2, 3), requires_grad=True)
        >>> loss = head.loss(feature, target)
        >>> loss.shape
        ()
        >>> loss.item() == head.cross_entropy(feature, target).mean().item()
        True

        The target is data, so it is detached: the gradient reaches the head
        and stops there.

        >>> loss.backward()
        >>> head.trunk.out.weight.grad.shape, target.grad is None
        ((41, 16), True)
        """
        return self.cross_entropy(feature, target).mean()
