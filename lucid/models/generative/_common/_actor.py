"""The policy head the world models share.

Every Dreamer after the first proposes actions the same way: a dense
trunk over the latent, read as a truncated Gaussian when the action is a
box and as a one-hot categorical when it is a button.  DreamerV2 needed
it first, DreamerV3 needs the same two branches, so it lives here rather
than in either family.

The only thing the two families disagree about is ``unimix``, and it is a
constructor argument.
"""

from typing import override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models.generative._common._dists import OneHotCategorical, TruncatedNormal
from lucid.models.generative._common._pixel_nets import DenseHead

__all__ = ["Actor"]


class Actor(nn.Module):
    r"""A truncated Gaussian, or a one-hot categorical, over actions.

    Dreamer squashed a normal with ``tanh``; this truncates one to
    ``[-1, 1]`` instead.  The reason is the entropy bonus these families
    add: a squashed normal has no closed-form entropy and has to be
    estimated by sampling, while a truncated one is exact.

    Parameters
    ----------
    latent_size, hidden, layers : int
        Shape of the dense trunk.
    action_dim : int
        Width of the action, or the number of alternatives when discrete.
    act_fn : str
        Activation in the trunk.
    min_std : float
        Floor on the scale.
    discrete : bool, default=False
        Whether an action is a choice rather than a vector.
    unimix : float, default=0.0
        Uniform mass mixed into a discrete policy.  Ignored when the
        action space is continuous, which is why it is not an error to
        set both.

    Notes
    -----
    Both continuous parameterisations follow the released implementation:
    ``mean = tanh(raw)``, which lands inside the interval without
    saturating the truncation, and
    ``std = 2 * sigmoid(raw / 2) + min_std``, which is bounded above as
    well as below — an unbounded ``softplus`` would let the policy widen
    until it is uniform and the entropy bonus stops pushing.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative._common._actor import Actor
    >>> actor = Actor(8, 16, 2, 3, "silu", 0.1)
    >>> actor(lucid.zeros((2, 4, 8))).shape
    (2, 4, 3)
    """

    def __init__(
        self,
        latent_size: int,
        hidden: int,
        layers: int,
        action_dim: int,
        act_fn: str,
        min_std: float,
        discrete: bool = False,
        unimix: float = 0.0,
    ) -> None:
        """Initialise the actor. See the class docstring for parameters."""
        super().__init__()
        self.action_dim = action_dim
        self.min_std = min_std
        self.discrete = discrete
        self.unimix = unimix
        # A one-hot needs one score per alternative; a truncated Gaussian
        # needs a location and a scale per dimension.
        width = action_dim if discrete else 2 * action_dim
        self.head = DenseHead(latent_size, hidden, layers, out_features=width, act_fn=act_fn)

    def distribution(self, feature: Tensor) -> TruncatedNormal | OneHotCategorical:
        """Return the policy at a state.

        Parameters
        ----------
        feature : Tensor
            Latent state, ``(B, T, latent_size)``.

        Returns
        -------
        TruncatedNormal or OneHotCategorical
            A truncated Gaussian over ``[-1, 1]^action_dim`` for a
            continuous action space, a one-hot over ``action_dim``
            alternatives for a discrete one.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models.generative._common._actor import Actor
        >>> feature = lucid.randn((2, 4, 8))
        >>> box = Actor(8, 16, 2, 3, "silu", 0.1)
        >>> policy = box.distribution(feature)
        >>> type(policy).__name__, policy.mode.shape
        ('TruncatedNormal', (2, 4, 3))
        >>> bool((policy.mean.abs() <= 1.0).all().item())  # inside the box
        True
        >>> buttons = Actor(8, 16, 2, 3, "silu", 0.1, discrete=True)
        >>> choice = buttons.distribution(feature)
        >>> type(choice).__name__, choice.probs.shape
        ('OneHotCategorical', (2, 4, 3))
        >>> bool(((choice.probs.sum(dim=-1) - 1.0).abs() < 1e-5).all().item())
        True
        """
        out = self.head(feature)
        if self.discrete:
            return OneHotCategorical(out, unimix=self.unimix)
        raw_mean = out[..., : self.action_dim]
        raw_std = out[..., self.action_dim :]
        mean = lucid.tanh(raw_mean)
        std = 2.0 * F.sigmoid(raw_std / 2.0) + self.min_std
        return TruncatedNormal(mean, std)

    def log_prob(self, feature: Tensor, action: Tensor) -> Tensor:
        """Log probability of an action, summed over the event.

        Parameters
        ----------
        feature : Tensor
            Latent state, ``(B, T, latent_size)``.
        action : Tensor
            ``(B, T, action_dim)``.

        Returns
        -------
        Tensor
            ``(B, T)``.  The event axis is summed here rather than by the
            caller, because what counts as one event differs: a
            continuous action is ``action_dim`` independent draws, a
            one-hot is a single choice.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models.generative._common._actor import Actor
        >>> feature = lucid.randn((2, 4, 8))
        >>> box = Actor(8, 16, 2, 3, "silu", 0.1)
        >>> action = box(feature)
        >>> summed = box.log_prob(feature, action)
        >>> summed.shape
        (2, 4)
        >>> per_dim = box.distribution(feature).log_prob(action)  # (2, 4, 3)
        >>> bool(lucid.allclose(per_dim.sum(dim=-1), summed))
        True
        >>> buttons = Actor(8, 16, 2, 3, "silu", 0.1, discrete=True)
        >>> choices = [lucid.eye(3)[k] * lucid.ones((2, 4, 1)) for k in range(3)]
        >>> total = sum(buttons.log_prob(feature, c).exp() for c in choices)
        >>> bool(((total - 1.0).abs() < 1e-5).all().item())  # one event, not three
        True
        """
        policy = self.distribution(feature)
        value = policy.log_prob(action)
        return value if self.discrete else value.sum(dim=-1)

    def entropy(self, feature: Tensor) -> Tensor:
        """Policy entropy at a state, summed over the event — ``(B, T)``.

        Parameters
        ----------
        feature : Tensor
            Latent state, ``(B, T, latent_size)``.

        Returns
        -------
        Tensor
            ``(B, T)``.

        Examples
        --------
        >>> import math
        >>> import lucid
        >>> from lucid.models.generative._common._actor import Actor
        >>> feature = lucid.randn((2, 4, 8))
        >>> box = Actor(8, 16, 2, 3, "silu", 0.1)
        >>> entropy = box.entropy(feature)
        >>> entropy.shape
        (2, 4)
        >>> bool((entropy <= 3 * math.log(2.0)).all().item())  # at most the uniform's
        True
        >>> buttons = Actor(8, 16, 2, 5, "silu", 0.1, discrete=True)
        >>> bool((buttons.entropy(feature) <= math.log(5) + 1e-6).all().item())
        True
        """
        value = self.distribution(feature).entropy()
        return value if self.discrete else value.sum(dim=-1)

    @override
    def forward(self, feature: Tensor, *, sample: bool = True) -> Tensor:
        """Propose an action — ``(B, T, action_dim)`` inside ``(-1, 1)``.

        Parameters
        ----------
        feature : Tensor
            Latent state, ``(B, T, latent_size)`` or ``(B, latent_size)``.
        sample : bool, default=True, keyword-only
            Draw, or take the distribution's mode.

        Returns
        -------
        Tensor
            Bounded actions, at the rank that went in.
        """
        stepwise = feature.ndim == 2
        if stepwise:
            feature = feature.reshape(int(feature.shape[0]), 1, -1)
        policy = self.distribution(feature)
        action = policy.rsample() if sample else policy.mode
        return action[:, 0] if stepwise else action
