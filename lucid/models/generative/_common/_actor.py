"""The policy head the world models share.

Every Dreamer after the first proposes actions the same way: a dense
trunk over the latent, read as a Gaussian when the action is a box and as
a one-hot categorical when it is a button.  DreamerV2 needed it first,
DreamerV3 needs the same two branches, so it lives here rather than in
either family.

The two families disagree about three things, all constructor arguments:
``unimix``, which Gaussian a box reads as (``continuous_dist``, with its
``max_std``), and how the output layer starts (``outscale``).  A family
that passes none of them gets DreamerV2's head.
"""

from typing import Literal, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models.generative._common._dists import Normal, OneHotCategorical, TruncatedNormal
from lucid.models.generative._common._pixel_nets import DenseHead

__all__ = ["Actor", "ContinuousPolicy"]

#: The continuous policy parameterisations, named after the released
#: configurations that select them: ``truncated_normal`` is DreamerV2's head,
#: ``bounded_normal`` is DreamerV3's (``policy_dist_cont`` in its config).
ContinuousPolicy = Literal["truncated_normal", "bounded_normal"]


class Actor(nn.Module):
    r"""A Gaussian, or a one-hot categorical, over actions.

    Dreamer squashed a normal with ``tanh``; DreamerV2 truncates one to
    ``[-1, 1]`` instead.  The reason is the entropy bonus these families
    add: a squashed normal has no closed-form entropy and has to be
    estimated by sampling, while a truncated one is exact.  DreamerV3's
    release does neither — it squashes only the *mean* and leaves the
    Gaussian around it whole, which is also exact — so both are offered.

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
    continuous_dist : {"truncated_normal", "bounded_normal"}, keyword-only
        Which Gaussian a continuous action is read as; ``"truncated_normal"``
        by default, and ignored when ``discrete``.
    max_std : float, default=1.0, keyword-only
        Ceiling on the scale under ``bounded_normal``, where it is part of
        the formula; ignored under ``truncated_normal``, whose ceiling is
        fixed by its own formula.  Must be at least ``min_std`` when used.
    outscale : float or None, default=None, keyword-only
        How the output layer starts.  ``None`` leaves it at the layer's
        default initialisation.  A number starts it as the released
        implementations do: weights scaled by ``outscale`` and bias zero,
        so ``0.0`` is an output that is exactly zero and ``0.01`` one that
        is nearly so.  Applies to the location and the scale alike.

    Raises
    ------
    ValueError
        If ``continuous_dist`` is not one of the two, ``outscale`` is
        negative, or ``max_std`` is below ``min_std`` under
        ``bounded_normal``.

    Notes
    -----
    Both continuous parameterisations take ``mean = tanh(raw)``, which
    lands inside the interval without saturating anything, and bound the
    scale on both sides — an unbounded ``softplus`` would let the policy
    widen until it is uniform and the entropy bonus stops pushing.  They
    differ in the scale and in what surrounds the mean:

    - ``"truncated_normal"``: ``std = 2 * sigmoid(raw / 2) + min_std``,
      truncated to ``[-1, 1]``.
    - ``"bounded_normal"``:
      ``std = (max_std - min_std) * sigmoid(raw + 2) + min_std``, with no
      truncation.

    The first is DreamerV2's; the second is DreamerV3's released
    ``bounded_normal`` head (``embodied/jax/heads.py``).  The ``+ 2``
    inside the sigmoid starts the scale near its ceiling — 0.89 at the
    defaults, from a zero output — rather than mid-range, and the
    untruncated draw means an action can leave ``[-1, 1]``: clipping it to
    the environment's box is the caller's, as the released environment
    wrapper does.

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
        *,
        continuous_dist: ContinuousPolicy = "truncated_normal",
        max_std: float = 1.0,
        outscale: float | None = None,
    ) -> None:
        """Initialise the actor. See the class docstring for parameters."""
        super().__init__()
        if continuous_dist not in ("truncated_normal", "bounded_normal"):
            raise ValueError(
                f"continuous_dist must be 'truncated_normal' or 'bounded_normal', "
                f"got {continuous_dist!r}"
            )
        if continuous_dist == "bounded_normal" and max_std < min_std:
            raise ValueError(
                f"max_std must be at least min_std under bounded_normal, got "
                f"max_std={max_std} and min_std={min_std}"
            )
        if outscale is not None and outscale < 0.0:
            raise ValueError(f"outscale must be non-negative, got {outscale}")
        self.action_dim = action_dim
        self.min_std = min_std
        self.max_std = max_std
        self.discrete = discrete
        self.unimix = unimix
        self.continuous_dist: ContinuousPolicy = continuous_dist
        # A one-hot needs one score per alternative; a Gaussian needs a
        # location and a scale per dimension.
        width = action_dim if discrete else 2 * action_dim
        self.head = DenseHead(latent_size, hidden, layers, out_features=width, act_fn=act_fn)
        if outscale is not None:
            self._scale_output(outscale)

    def _scale_output(self, outscale: float) -> None:
        """Start the output layer as the released implementations do.

        The released layers scale the *kernel's* initial draw by
        ``outscale`` and start the bias at zero, for the location and the
        scale rows alike.  Scaling consumes no random numbers, so a model
        built with and without it is otherwise initialised identically.
        """
        out = self.head.out
        with lucid.no_grad():
            out.weight[:] = out.weight * outscale
            if out.bias is not None:
                out.bias[:] = lucid.zeros_like(out.bias)

    def distribution(self, feature: Tensor) -> TruncatedNormal | Normal | OneHotCategorical:
        """Return the policy at a state.

        Parameters
        ----------
        feature : Tensor
            Latent state, ``(B, T, latent_size)``.

        Returns
        -------
        TruncatedNormal, Normal or OneHotCategorical
            For a continuous action space, a Gaussian around
            ``tanh(raw)`` — truncated to ``[-1, 1]^action_dim`` under
            ``"truncated_normal"``, untruncated under ``"bounded_normal"``.
            For a discrete one, a one-hot over ``action_dim``
            alternatives.

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

        The released DreamerV3 head keeps the Gaussian whole and bounds its
        scale to ``[min_std, max_std]``; with a zero output it is
        ``(max_std - min_std) * sigmoid(2) + min_std``:

        >>> bounded = Actor(8, 16, 2, 3, "silu", 0.1, continuous_dist="bounded_normal",
        ...                 outscale=0.0)
        >>> policy = bounded.distribution(feature)
        >>> type(policy).__name__, float(policy.scale.max().item()) < 1.0
        ('Normal', True)
        >>> round(float(policy.scale[0, 0, 0].item()), 4)
        0.8927
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
        if self.continuous_dist == "bounded_normal":
            std = (self.max_std - self.min_std) * F.sigmoid(raw_std + 2.0) + self.min_std
            return Normal(mean, std)
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
        """Propose an action — ``(B, T, action_dim)``.

        Parameters
        ----------
        feature : Tensor
            Latent state, ``(B, T, latent_size)`` or ``(B, latent_size)``.
        sample : bool, default=True, keyword-only
            Draw, or take the distribution's mode.

        Returns
        -------
        Tensor
            Actions at the rank that went in, one-hot when discrete.  A
            continuous action lies in ``[-1, 1]`` for ``"truncated_normal"``
            and for either policy's mode; a *draw* from ``"bounded_normal"``
            is not clipped and may fall outside, which the environment
            boundary corrects (the rollout driver clips).
        """
        stepwise = feature.ndim == 2
        if stepwise:
            feature = feature.reshape(int(feature.shape[0]), 1, -1)
        policy = self.distribution(feature)
        action = policy.rsample() if sample else policy.mode
        return action[:, 0] if stepwise else action
