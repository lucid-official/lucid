"""Recurrent State-Space Model — the shared trunk of the world-model families.

Introduced by Hafner, Lillicrap, Fischer, Villegas, Ha, Lee & Davidson,
*"Learning Latent Dynamics for Planning from Pixels"* (ICML, 2019) as the
dynamics model inside PlaNet, and reused with modifications by the Dreamer
line.  It lives here rather than inside a family for the same reason
``_schedulers.py`` does: it is the substrate several families share, not
something any one of them owns.

The state is deliberately **split in two**.  A purely stochastic latent
cannot reliably remember information across many steps — each step's noise
erodes it — while a purely deterministic one cannot represent the several
futures a partially-observed environment allows.  So each step carries

    deter  — a GRU hidden state, the path information survives along
    stoch  — a Gaussian sample, the path uncertainty enters through

and the transition conditions on both.  That split is the paper's central
architectural claim, and it is why the ablations against a pure RNN and a
pure state-space model both underperform it.

Two distributions are produced at every step:

    prior      p(s_t | h_t)          — what the dynamics predict
    posterior  q(s_t | h_t, e_t)     — what the dynamics predict *after*
                                       seeing the encoded observation

Training pulls the prior toward the posterior; planning then runs the prior
alone, which is what :meth:`RSSM.imagine` does.  Reconstruction reads off
the posterior, which is what :meth:`RSSM.observe` returns.

Layout convention: every public method takes and returns ``(B, T, ...)``,
matching the rest of the zoo.  The unroll itself is time-major internally
because a recurrence has to be — the same shape ``nn.GRU.forward`` takes.
"""

import math
from collections.abc import Callable
from typing import NamedTuple, cast, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._utils._generative import (
    generative_activation,
    model_dtype,
    normal_kl,
    resolve_generation_device,
)

__all__ = ["RSSMState", "RSSM", "BlockLinear", "rssm_kl"]


class RSSMState(NamedTuple):
    r"""One step (or one unrolled sequence) of recurrent state-space state.

    Attributes
    ----------
    deter : Tensor
        Deterministic path :math:`h`, shaped ``(B, deter_size)`` for a
        single step or ``(B, T, deter_size)`` after an unroll.
    stoch : Tensor
        The stochastic latent :math:`s`, drawn so that gradients flow
        through it — reparameterised for a Gaussian latent,
        straight-through for a categorical one.  Always flat: a
        categorical grid arrives here already reshaped to
        ``(B, stoch_size * discrete)``, so anything reading
        :attr:`feature` sees one convention.
    mean, std : Tensor or None
        The Gaussian ``stoch`` was drawn from.  ``None`` when the latent
        is categorical.
    logits : Tensor or None
        Unnormalised class scores, ``(B, stoch_size, discrete)``, when the
        latent is categorical.  ``None`` when it is Gaussian.

    Notes
    -----
    The distribution's parameters ride along with the sample because the
    objective needs the *distributions*, not just draws: the divergence
    between prior and posterior is computed in closed form from them.

    Exactly one of the two parameterisations is present, and which one is
    a property of the model that produced the state rather than of the
    state itself.  Use :meth:`map` to transform a state without having to
    know which — writing ``RSSMState(*(f(x) for x in state))`` will hit a
    ``None`` and raise.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative._common._rssm import RSSMState
    >>> gaussian = RSSMState(lucid.zeros((2, 4)), lucid.zeros((2, 3)),
    ...                      lucid.zeros((2, 3)), lucid.ones((2, 3)))
    >>> gaussian.is_discrete
    False
    >>> gaussian.map(lambda t: t[:1]).deter.shape
    (1, 4)
    """

    deter: Tensor
    stoch: Tensor
    mean: Tensor | None = None
    std: Tensor | None = None
    logits: Tensor | None = None

    @property
    def feature(self) -> Tensor:
        """``[h; s]`` — what a decoder or a reward head reads off the state."""
        return lucid.cat([self.deter, self.stoch], dim=-1)

    @property
    def is_discrete(self) -> bool:
        """Whether this state carries a categorical latent."""
        return self.logits is not None

    @property
    def logvar(self) -> Tensor:
        r""":math:`\log \sigma^2`, for the helpers that parameterise that way.

        Returns
        -------
        Tensor
            Twice the log standard deviation.

        Raises
        ------
        ValueError
            If the latent is categorical, which has no variance to report.

        Notes
        -----
        Safe without an epsilon: ``std`` is floored at ``min_std`` by
        construction, so the logarithm never approaches its pole.
        """
        if self.std is None:
            raise ValueError("a categorical latent has no log-variance")
        return 2.0 * self.std.log()

    def gaussian(self) -> tuple[Tensor, Tensor]:
        """The ``(mean, std)`` this state was drawn from.

        Returns
        -------
        mean, std : Tensor
            The Gaussian's parameters, narrowed from the optional fields.

        Raises
        ------
        ValueError
            If the latent is categorical.  The families that call this are
            Gaussian by construction, so the raise is a contract check
            rather than a branch anyone is expected to take.

        Notes
        -----
        Exists so that Gaussian-only code can read the parameters once and
        keep working with plain tensors, instead of threading an optional
        through every expression that uses them.
        """
        if self.mean is None or self.std is None:
            raise ValueError(
                "this state carries a categorical latent; it has no mean or "
                "standard deviation"
            )
        return self.mean, self.std

    def map(self, fn: Callable[[Tensor], Tensor]) -> RSSMState:
        """Apply ``fn`` to every tensor present, leaving absent ones absent.

        Parameters
        ----------
        fn : callable
            ``Tensor -> Tensor``.  Applied to each field that is not
            ``None``.

        Returns
        -------
        RSSMState
            A new state; nothing is mutated.

        Notes
        -----
        This exists because a state is a *partial* record — a Gaussian one
        has no ``logits``, a categorical one has no ``mean`` — so the
        obvious ``RSSMState(*(fn(x) for x in state))`` fails on whichever
        half is missing.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models.generative._common._rssm import RSSMState
        >>> unroll = RSSMState(lucid.zeros((2, 5, 4)), lucid.zeros((2, 5, 6)),
        ...                    logits=lucid.zeros((2, 5, 3, 2)))
        >>> last = unroll.map(lambda t: t[:, -1])  # the final step of the unroll
        >>> last.deter.shape, last.logits.shape
        ((2, 4), (2, 3, 2))
        >>> last.mean is None
        True
        """

        def apply(t: Tensor | None) -> Tensor | None:
            return None if t is None else fn(t)

        return RSSMState(
            deter=fn(self.deter),
            stoch=fn(self.stoch),
            mean=apply(self.mean),
            std=apply(self.std),
            logits=apply(self.logits),
        )


def _gumbel_argmax(logits: Tensor) -> Tensor:
    r"""Draw a categorical index without leaving the accelerator.

    Parameters
    ----------
    logits : Tensor
        Unnormalised scores, ``(..., classes)``.

    Returns
    -------
    Tensor
        Indices, ``(...)``, on the device the logits were on.

    Notes
    -----
    The Gumbel-max trick: adding :math:`-\log(-\log u)` with
    :math:`u \sim \mathrm{Uniform}(0, 1)` to a set of logits and taking
    the ``argmax`` samples from exactly the categorical those logits
    define.  This is not a relaxation — the draw is exact, and the
    empirical frequencies match ``multinomial``'s.

    The reason to prefer it is mechanical.  ``lucid.multinomial`` is
    data-dependent, so it returns on the CPU, and a recurrence pays that
    synchronisation once per step — with the imagination on top, dozens of
    times per training step.  Measured on a DreamerV2 step: 11.5 s on
    Metal against 0.7 s on the CPU, which made the accelerator sixteen
    times *slower* than not using it.  ``rand``, ``log`` and ``argmax``
    are each device-resident, so the same draw costs nothing.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative._common._rssm import _gumbel_argmax
    >>> _gumbel_argmax(lucid.zeros((4, 5))).shape
    (4,)
    """
    # Both bounds are exclusive on purpose.  `-log(-log(u))` is +inf at
    # u = 1 and -inf at u = 0, and a clip whose upper bound is 1.0
    # *inclusive* leaves the first of those reachable — rarely, which is
    # the worst frequency for a numerical landmine.
    uniform = lucid.rand(
        tuple(int(s) for s in logits.shape), device=logits.device, dtype=logits.dtype
    ).clip(1e-9, 1.0 - 1e-7)
    gumbel = -lucid.log(-lucid.log(uniform))
    return (logits + gumbel).argmax(dim=-1)


class BlockLinear(nn.Module):
    r"""A linear map that never mixes its blocks.

    Parameters
    ----------
    in_features, out_features : int
        Total widths, both divisible by ``blocks``.  Flat in, flat out —
        the grouping is an implementation detail of the weight, not of
        the call site.
    blocks : int
        How many independent maps to run side by side.

    Raises
    ------
    ValueError
        If ``blocks`` is not positive, or does not divide both widths.

    Notes
    -----
    Group :math:`i` of the output reads group :math:`i` of the input and
    nothing else, so the weight is
    :math:`(g,\, d_{\mathrm{in}}/g,\, d_{\mathrm{out}}/g)` rather than
    :math:`(d_{\mathrm{in}},\, d_{\mathrm{out}})` — a factor of :math:`g`
    fewer parameters for the same widths.

    This is what lets DreamerV3's sequence model be eight times wider than
    its hidden layers without the recurrence dominating the parameter
    count: at the ``200m`` rung a dense recurrence over 8192 units would
    cost 200M parameters by itself, which is the entire model's budget.
    Written as one map with a block-structured weight rather than as
    ``blocks`` separate maps because that keeps it a single matmul, and
    because a list of submodules would put ``blocks`` entries in
    ``state_dict`` where there is conceptually one layer.

    Initialised exactly as :class:`~lucid.nn.Linear` is, with the fan-in
    taken *within* a block, which is the fan-in that actually reaches an
    output.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative._common._rssm import BlockLinear
    >>> layer = BlockLinear(12, 24, blocks=4)
    >>> layer(lucid.zeros((2, 12))).shape
    (2, 24)
    >>> layer.weight.shape
    (4, 3, 6)
    """

    def __init__(self, in_features: int, out_features: int, blocks: int) -> None:
        """Initialise the layer. See the class docstring for parameters."""
        super().__init__()
        if blocks < 1:
            raise ValueError(f"blocks must be positive, got {blocks}")
        if in_features % blocks or out_features % blocks:
            raise ValueError(
                f"blocks must divide both widths, got {in_features} and "
                f"{out_features} for {blocks} blocks"
            )
        self.blocks = blocks
        self.in_block = in_features // blocks
        self.out_block = out_features // blocks
        self.weight = nn.Parameter(lucid.empty(blocks, self.in_block, self.out_block))
        self.bias = nn.Parameter(lucid.empty(blocks, self.out_block))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Uniform on the within-block fan-in, as :class:`~lucid.nn.Linear` is."""
        bound = 1.0 / math.sqrt(self.in_block)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Apply the block map — ``(..., in_features) -> (..., out_features)``.

        Parameters
        ----------
        x : Tensor
            Flat, with the groups laid out contiguously along the last
            axis.

        Returns
        -------
        Tensor
            Flat, in the same layout.
        """
        leading = tuple(int(v) for v in x.shape[:-1])
        grouped = x.reshape(*leading, self.blocks, 1, self.in_block)
        out = lucid.matmul(grouped, self.weight)
        return (out.reshape(*leading, self.blocks, self.out_block) + self.bias).reshape(
            *leading, self.blocks * self.out_block
        )


class RSSM(nn.Module):
    r"""Recurrent state-space model (Hafner et al., 2019).

    One step of the recurrence is

    .. math::

        h_t = f\big(h_{t-1},\, [s_{t-1};\, a_{t-1}]\big),

    a GRU update whose input is the previous stochastic latent concatenated
    with the action, followed by two heads reading off that deterministic
    path:

    .. math::

        p(s_t \mid h_t) = \mathcal{N}\big(\mu_p(h_t),\, \sigma_p(h_t)\big),
        \qquad
        q(s_t \mid h_t, e_t)
            = \mathcal{N}\big(\mu_q(h_t, e_t),\, \sigma_q(h_t, e_t)\big),

    where :math:`e_t` is the encoded observation.  The prior is the model's
    guess before it looks; the posterior is its belief after.

    Parameters
    ----------
    stoch_size : int
        Width of the stochastic latent :math:`s`.
    deter_size : int
        Width of the GRU hidden state :math:`h`.
    hidden_size : int
        Width of the hidden layer inside each head and inside the
        pre-recurrence projection.
    action_dim : int
        Width of the action vector conditioning the transition.
    embed_size : int
        Width of the encoded observation the posterior head consumes.
    act_fn : {"silu", "swish", "relu", "gelu"}, default="relu"
        Activation used by the projections.  The paper uses ReLU.
    min_std : float, default=0.1
        Floor added after ``softplus``.  Without it a head can drive the
        standard deviation to zero, at which point the KL diverges and the
        sample stops carrying gradient.
    discrete : int, default=0
        Classes per stochastic variable.  ``0`` keeps the Gaussian latent
        the paper describes; ``>= 2`` makes the latent a grid of
        categoricals, which is what DreamerV2 changed.
    unimix : float, default=0.0
        Uniform mass mixed into every categorical, so no class can reach
        probability zero and strand its gradient.  DreamerV3 sets 1%.
    blocks : int, default=0
        ``0`` uses the GRU recurrence above.  ``>= 1`` uses DreamerV3's
        block-diagonal sequence model instead, in which the ``deter_size``
        units are split into this many independent groups — see
        :meth:`_block_recurrent`.  Must divide ``deter_size``.

    Attributes
    ----------
    cell : nn.GRUCell
        The deterministic recurrence.  Absent when ``blocks`` is set,
        which replaces it with :class:`BlockLinear` gates.
    prior_head : nn.Module
        Maps :math:`h_t` to :math:`(\mu_p, \sigma_p)`.
    posterior_head : nn.Module
        Maps :math:`[h_t; e_t]` to :math:`(\mu_q, \sigma_q)`.

    Notes
    -----
    Reference: Hafner, Lillicrap, Fischer, Villegas, Ha, Lee, and Davidson,
    *"Learning Latent Dynamics for Planning from Pixels"*, ICML, 2019
    (arXiv:1811.04551).

    The Gaussian is parameterised by ``(mean, std)`` because that is what
    ``softplus(raw) + min_std`` produces; :attr:`RSSMState.logvar` converts
    for the helpers that want the other form.

    The divergence goes through :func:`normal_kl`, which is elementwise
    between two *free* Gaussians and applies no reduction — leaving free
    nats to :func:`rssm_kl`.  :func:`gaussian_kl_divergence` in the same
    module is the wrong tool here: it measures against a fixed
    :math:`\mathcal{N}(0, I)`, and the entire point of the prior head is
    that the reference is learned.  ``lucid.distributions.kl_divergence``
    would serve equally (the two agree to ~4e-06), but no other module
    under ``lucid/models/`` imports ``lucid.distributions``, and one
    family is not a reason to open that direction.

    **Action alignment.**  ``actions[:, t]`` is the action taken *into*
    step ``t`` — :math:`a_{t-1}` in the paper's indexing, the one that
    produced the observation at ``t``.  A caller holding a trajectory
    :math:`(o_0, a_0, o_1, a_1, \dots)` must therefore shift by one and
    pass a zero action at index 0.  Nothing can enforce this: a
    mis-aligned caller gets a model that trains to a worse optimum, never
    an error.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative._common._rssm import RSSM
    >>> rssm = RSSM(stoch_size=4, deter_size=8, hidden_size=8,
    ...             action_dim=2, embed_size=6).eval()
    >>> embed = lucid.randn((2, 5, 6))
    >>> actions = lucid.randn((2, 5, 2))
    >>> priors, posteriors = rssm.observe(embed, actions)
    >>> posteriors.stoch.shape, posteriors.deter.shape
    ((2, 5, 4), (2, 5, 8))
    >>> rssm.imagine(rssm.initial(2), actions).stoch.shape
    (2, 5, 4)
    """

    def __init__(
        self,
        stoch_size: int,
        deter_size: int,
        hidden_size: int,
        action_dim: int,
        embed_size: int,
        act_fn: str = "relu",
        min_std: float = 0.1,
        discrete: int = 0,
        unimix: float = 0.0,
        blocks: int = 0,
    ) -> None:
        """Initialise the RSSM. See the class docstring for parameter semantics."""
        super().__init__()
        if not 0.0 <= unimix < 1.0:
            raise ValueError(f"unimix must be in [0, 1), got {unimix}")
        if blocks < 0:
            raise ValueError(f"blocks must be non-negative, got {blocks}")
        if blocks and deter_size % blocks:
            raise ValueError(
                f"blocks must divide deter_size, got {deter_size} and {blocks}"
            )
        if discrete < 0 or discrete == 1:
            raise ValueError(
                "discrete must be 0 for a Gaussian latent or at least 2 for a "
                f"categorical one; got {discrete}"
            )
        for name, value in (
            ("stoch_size", stoch_size),
            ("deter_size", deter_size),
            ("hidden_size", hidden_size),
            ("embed_size", embed_size),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        if min_std <= 0.0:
            raise ValueError(f"min_std must be positive, got {min_std}")

        self.stoch_size = stoch_size
        self.deter_size = deter_size
        self.action_dim = action_dim
        self.embed_size = embed_size
        self.min_std = min_std
        self.discrete = discrete
        self.unimix = unimix
        self.blocks = blocks
        self._act_name = act_fn

        # A categorical latent is a grid: `stoch_size` variables of
        # `discrete` classes each, flattened once it leaves the head so
        # everything downstream sees one vector.
        self.stoch_width = stoch_size * discrete if discrete else stoch_size
        head_width = self.stoch_width if discrete else 2 * stoch_size

        if blocks:
            # DreamerV3's sequence model.  Three normalised projections
            # instead of one concatenation, then a gate computed
            # block-diagonally so the recurrence can be eight times wider
            # than the hidden layers at a fraction of the parameters.
            self.deter_in = nn.Linear(deter_size, hidden_size)
            self.stoch_in = nn.Linear(self.stoch_width, hidden_size)
            self.action_in = nn.Linear(action_dim, hidden_size)
            # eps pinned to the value this port has always used, so the
            # module default following the dtype changes nothing here.
            self.deter_in_norm = nn.RMSNorm(hidden_size, eps=1e-8)
            self.stoch_in_norm = nn.RMSNorm(hidden_size, eps=1e-8)
            self.action_in_norm = nn.RMSNorm(hidden_size, eps=1e-8)
            self.block_hidden = BlockLinear(
                deter_size + 3 * hidden_size * blocks, deter_size, blocks
            )
            self.block_hidden_norm = nn.RMSNorm(deter_size, eps=1e-8)
            self.block_gate = BlockLinear(deter_size, 3 * deter_size, blocks)
        else:
            self.pre_cell = nn.Linear(self.stoch_width + action_dim, hidden_size)
            self.cell = nn.GRUCell(hidden_size, deter_size)

        self.prior_head: nn.ModuleList[nn.Linear] = nn.ModuleList(
            [
                nn.Linear(deter_size, hidden_size),
                nn.Linear(hidden_size, head_width),
            ]
        )
        self.posterior_head: nn.ModuleList[nn.Linear] = nn.ModuleList(
            [
                nn.Linear(deter_size + embed_size, hidden_size),
                nn.Linear(hidden_size, head_width),
            ]
        )

    # ── internals ────────────────────────────────────────────────────────

    def _recurrent(self, stoch: Tensor, action: Tensor, deter: Tensor) -> Tensor:
        r"""Advance the deterministic path one step — :math:`h_{t-1} \to h_t`.

        Parameters
        ----------
        stoch, action, deter : Tensor
            The previous latent, the action taken into this step, and the
            previous deterministic state; all ``(B, ·)``.

        Returns
        -------
        Tensor
            The new deterministic state, ``(B, deter_size)``.

        Notes
        -----
        One of two recurrences, chosen by ``blocks``.  Both are here
        rather than in a subclass because the *rest* of the model — the
        heads, the draw, the unrolls, the divergence — is identical
        either way, and a subclass would have to inherit all of it to
        change these ten lines.
        """
        if not self.blocks:
            x = lucid.cat([stoch, action], dim=-1)
            x = generative_activation(self._act_name, self.pre_cell(x))
            return self.cell(x, deter)
        return self._block_recurrent(stoch, action, deter)

    def _block_recurrent(self, stoch: Tensor, action: Tensor, deter: Tensor) -> Tensor:
        r"""DreamerV3's block-diagonal gated recurrence.

        Parameters
        ----------
        stoch, action, deter : Tensor
            As in :meth:`_recurrent`.

        Returns
        -------
        Tensor
            The new deterministic state.

        Notes
        -----
        .. math::

            r, c, u = \mathrm{BlockLinear}(x), \qquad
            h_t = \sigma(u - 1)\,\tanh(\sigma(r)\, c)
                  + \big(1 - \sigma(u - 1)\big)\, h_{t-1},

        where :math:`x` is each block's slice of :math:`h_{t-1}`
        concatenated with the three normalised projections of
        :math:`h_{t-1}`, :math:`s_{t-1}` and :math:`a_{t-1}`, broadcast to
        every block.

        Two details are not the textbook GRU and both are deliberate.  The
        update gate is offset by one before the sigmoid, so a freshly
        initialised model *keeps* its state rather than replacing it —
        which is what makes a very wide recurrence trainable from step
        one.  And the three inputs are projected and normalised
        separately rather than concatenated and projected once, so a
        large observation embedding cannot drown out a one-dimensional
        action.

        The action is divided by its own magnitude when that exceeds one.
        Bounded policies never trigger it; it is there so that an
        environment handing over unnormalised actions degrades rather than
        destabilises.
        """
        blocks = self.blocks
        scaled = action / action.abs().clip(1.0, None).detach()

        def project(layer: nn.Linear, norm: nn.RMSNorm, x: Tensor) -> Tensor:
            return generative_activation(self._act_name, norm(layer(x)))

        parts = lucid.cat(
            [
                project(self.deter_in, self.deter_in_norm, deter),
                project(self.stoch_in, self.stoch_in_norm, stoch),
                project(self.action_in, self.action_in_norm, scaled),
            ],
            dim=-1,
        )
        leading = tuple(int(v) for v in deter.shape[:-1])
        width = self.deter_size // blocks
        # Every block sees the same summary of the step and its own slice
        # of the state.
        shared = parts.reshape(*leading, 1, -1).repeat(*(1,) * len(leading), blocks, 1)
        x = lucid.cat([deter.reshape(*leading, blocks, width), shared], dim=-1)
        x = self.block_hidden(x.reshape(*leading, -1))
        x = generative_activation(self._act_name, self.block_hidden_norm(x))

        gates = self.block_gate(x).reshape(*leading, blocks, 3 * width)
        reset = F.sigmoid(gates[..., :width].reshape(*leading, -1))
        candidate = gates[..., width : 2 * width].reshape(*leading, -1)
        update = F.sigmoid(gates[..., 2 * width :].reshape(*leading, -1) - 1.0)
        candidate = lucid.tanh(reset * candidate)
        return update * candidate + (1.0 - update) * deter

    def _head(self, head: nn.ModuleList[nn.Linear], x: Tensor) -> Tensor:
        """Run a two-layer head and return its raw output."""
        first, second = head[0], head[1]
        h = generative_activation(self._act_name, first(x))
        return second(h)

    def _latent(
        self, head: nn.ModuleList[nn.Linear], x: Tensor, sample: bool
    ) -> RSSMState:
        """Read a head and turn it into the latent half of a state.

        Parameters
        ----------
        head : Module
            ``prior_head`` or ``posterior_head``.
        x : Tensor
            Whatever that head reads.
        sample : bool
            Draw, or take the distribution's mode.

        Returns
        -------
        RSSMState
            With ``deter`` left as a placeholder for the caller to fill —
            only the stochastic fields are set here.
        """
        out = self._head(head, x)
        if self.discrete:
            leading = tuple(int(v) for v in out.shape[:-1])
            logits = out.reshape(*leading, self.stoch_size, self.discrete)
            stoch, logits = self._draw_discrete(logits, sample)
            return RSSMState(deter=out, stoch=stoch, logits=logits)
        mean = out[..., : self.stoch_size]
        std = nn.functional.softplus(out[..., self.stoch_size :]) + self.min_std
        return RSSMState(
            deter=out, stoch=self._draw(mean, std, sample), mean=mean, std=std
        )

    def _draw_discrete(self, logits: Tensor, sample: bool) -> tuple[Tensor, Tensor]:
        r"""One-hot draw with a straight-through gradient, flattened.

        Parameters
        ----------
        logits : Tensor
            ``(..., stoch_size, discrete)``.
        sample : bool
            Draw from the categorical, or take its mode.

        Returns
        -------
        sample : Tensor
            ``(..., stoch_size * discrete)`` — one-hot per variable.
        logits : Tensor
            The logits actually used, which are the *mixed* ones when
            ``unimix`` is set.  Returned rather than recomputed because
            the divergence has to measure the same distribution the
            sample came from.

        Notes
        -----
        A one-hot has no gradient, so the sample is passed forward and the
        class probabilities' gradient backward:

        .. math::

            \tilde{s} = \mathrm{sample} + p - \mathrm{sg}(p).

        That is :func:`lucid.nn.functional.straight_through`, the same
        estimator vector quantisation uses — the codebook lookup and the
        categorical draw are the same problem.

        The draw itself goes through :func:`_gumbel_argmax` rather than
        ``multinomial``, which is what keeps the recurrence on the
        accelerator — see that function.
        """
        probs = nn.functional.softmax(logits, dim=-1)
        if self.unimix:
            # Mix in a uniform floor, then recover the logits it implies —
            # every downstream user (the draw, the divergence) has to see
            # the *mixed* distribution, not the raw one, or the floor is
            # only cosmetic.
            classes = int(probs.shape[-1])
            probs = (1.0 - self.unimix) * probs + self.unimix / classes
            logits = probs.log()
        index = _gumbel_argmax(logits) if sample else probs.argmax(dim=-1)
        hard = nn.functional.one_hot(index, num_classes=self.discrete)
        hard = hard.to(probs.device).to(probs.dtype)
        onehot = nn.functional.straight_through(hard, probs)
        leading = tuple(int(v) for v in onehot.shape[:-2])
        return onehot.reshape(*leading, self.stoch_width), logits

    @staticmethod
    def _draw(mean: Tensor, std: Tensor, sample: bool) -> Tensor:
        """Reparameterised draw — gradient flows through ``mean`` and ``std``.

        With ``sample=False`` the mean is returned instead, which makes the
        whole recurrence deterministic.  Training wants the draw; planning
        wants the mean, because a search that re-samples every candidate is
        ranking its own noise rather than the actions.
        """
        if not sample:
            return mean
        noise = lucid.randn(
            tuple(int(s) for s in mean.shape), device=mean.device, dtype=mean.dtype
        )
        return mean + std * noise

    # ── single steps ─────────────────────────────────────────────────────

    def initial(self, batch_size: int, device: str | None = None) -> RSSMState:
        """Return the all-zero state the unroll starts from.

        Parameters
        ----------
        batch_size : int
            Leading dimension.
        device : str or None, default=None
            Where to allocate.  ``None`` follows this model's parameters,
            so a model moved to Metal starts its unroll there; an explicit
            value still wins.

        Returns
        -------
        RSSMState
            Zeros throughout, parameterised the way this model is, in its
            parameters' dtype — a float32 start ran a float16 model's whole
            unroll in float32.

        Notes
        -----
        The zero *logits* of a categorical start are a uniform
        distribution, which is the honest prior before anything has been
        seen.  The Gaussian start instead carries zero mean and zero
        standard deviation — degenerate, but nothing reads it: the first
        step overwrites both.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models.generative._common._rssm import RSSM
        >>> rssm = RSSM(stoch_size=4, deter_size=8, hidden_size=8,
        ...             action_dim=2, embed_size=6)
        >>> start = rssm.initial(3)
        >>> start.deter.shape, start.stoch.shape
        ((3, 8), (3, 4))
        >>> bool((start.feature == 0).all().item())
        True
        >>> grid = RSSM(stoch_size=4, deter_size=8, hidden_size=8,
        ...             action_dim=2, embed_size=6, discrete=5)
        >>> start = grid.initial(3)
        >>> start.stoch.shape, start.logits.shape  # 4 variables of 5 classes
        ((3, 20), (3, 4, 5))
        """
        # A hard "cpu" default handed a Metal model a CPU start state, and
        # the first recurrent step raised on the concatenation.
        device = resolve_generation_device(self, device)
        dtype = model_dtype(self)
        deter = lucid.zeros(batch_size, self.deter_size, dtype=dtype, device=device)
        stoch = lucid.zeros(batch_size, self.stoch_width, dtype=dtype, device=device)
        if self.discrete:
            logits = lucid.zeros(
                batch_size, self.stoch_size, self.discrete, dtype=dtype, device=device
            )
            return RSSMState(deter=deter, stoch=stoch, logits=logits)
        return RSSMState(deter=deter, stoch=stoch, mean=stoch, std=stoch)

    def prior_step(
        self, state: RSSMState, action: Tensor, *, sample: bool = True
    ) -> RSSMState:
        r"""Advance one step without an observation — :math:`p(s_t \mid h_t)`.

        Parameters
        ----------
        state : RSSMState
            The previous step's state, ``(B, ·)``.
        action : Tensor
            The action taken into this step, ``(B, action_dim)``.
        sample : bool, default=True
            Draw the latent, or take its mean when ``False``, which makes
            the step deterministic.

        Returns
        -------
        RSSMState
            The predicted state, ``(B, ·)``.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models.generative._common._rssm import RSSM
        >>> rssm = RSSM(stoch_size=4, deter_size=8, hidden_size=8,
        ...             action_dim=2, embed_size=6).eval()
        >>> start, action = rssm.initial(2), lucid.zeros((2, 2))
        >>> state = rssm.prior_step(start, action)
        >>> state.deter.shape, state.stoch.shape
        ((2, 8), (2, 4))
        >>> guess = rssm.prior_step(start, action, sample=False)
        >>> bool(lucid.allclose(guess.stoch, guess.mean))  # the mean, not a draw
        True
        """
        deter = self._recurrent(state.stoch, action, state.deter)
        latent = self._latent(self.prior_head, deter, sample)
        return latent._replace(deter=deter)

    def posterior_step(
        self,
        state: RSSMState,
        action: Tensor,
        embed: Tensor,
        *,
        sample: bool = True,
    ) -> tuple[RSSMState, RSSMState]:
        r"""Advance one step *with* an observation.

        Parameters
        ----------
        state : RSSMState
            The previous step's state, ``(B, ·)``.
        action : Tensor
            The action taken into this step, ``(B, action_dim)``.
        embed : Tensor
            The encoded observation at this step, ``(B, embed_size)``.
        sample : bool, default=True
            Draw both latents, or take their means when ``False``.

        Returns
        -------
        prior : RSSMState
            :math:`p(s_t \mid h_t)` — the prediction made before looking.
        posterior : RSSMState
            :math:`q(s_t \mid h_t, e_t)` — the belief after looking.

        Notes
        -----
        Both share the same ``deter``: the observation refines the belief
        about :math:`s_t`, it does not change the path that led there.
        The pair is returned together because the KL term needs both, and
        computing them apart would mean running the recurrence twice.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models.generative._common._rssm import RSSM
        >>> rssm = RSSM(stoch_size=4, deter_size=8, hidden_size=8,
        ...             action_dim=2, embed_size=6).eval()
        >>> prior, posterior = rssm.posterior_step(
        ...     rssm.initial(2), lucid.zeros((2, 2)), lucid.randn((2, 6)))
        >>> posterior.stoch.shape
        (2, 4)
        >>> bool((prior.deter == posterior.deter).all().item())  # one shared path
        True
        >>> bool((prior.mean == posterior.mean).all().item())  # the frame moved it
        False
        """
        prior = self.prior_step(state, action, sample=sample)
        x = lucid.cat([prior.deter, embed], dim=-1)
        posterior = self._latent(self.posterior_head, x, sample)._replace(
            deter=prior.deter
        )
        return prior, posterior

    # ── unrolls ──────────────────────────────────────────────────────────

    @staticmethod
    def _stack(states: list[RSSMState]) -> RSSMState:
        """Stack per-step states into ``(B, T, ...)``.

        Only the parameterisation the states actually carry is stacked —
        a categorical unroll has no ``mean`` to gather, and reaching for
        one would raise on the first step rather than at the end.
        """

        def gather(pick: Callable[[RSSMState], Tensor | None]) -> Tensor | None:
            first = pick(states[0])
            if first is None:
                return None
            return lucid.stack([cast(Tensor, pick(s)) for s in states], dim=1)

        return RSSMState(
            deter=lucid.stack([s.deter for s in states], dim=1),
            stoch=lucid.stack([s.stoch for s in states], dim=1),
            mean=gather(lambda s: s.mean),
            std=gather(lambda s: s.std),
            logits=gather(lambda s: s.logits),
        )

    def observe(
        self,
        embed: Tensor,
        actions: Tensor,
        state: RSSMState | None = None,
        *,
        sample: bool = True,
    ) -> tuple[RSSMState, RSSMState]:
        r"""Filter a sequence of observations into posterior states.

        Parameters
        ----------
        embed : Tensor
            Encoded observations, ``(B, T, embed_size)``.
        actions : Tensor
            Actions taken *into* each step, ``(B, T, action_dim)`` — see
            the class docstring on action alignment.  ``actions[:, t]`` is
            the action that produced ``embed[:, t]``, so index 0 is
            normally a zero action.
        state : RSSMState or None, optional
            Starting state; ``None`` starts from :meth:`initial`.
        sample : bool, default=True
            Draw each latent, or take its mean when ``False``, which makes
            the whole filter deterministic.

        Returns
        -------
        priors : RSSMState
            The dynamics' predictions, ``(B, T, ...)``.
        posteriors : RSSMState
            The beliefs after seeing each observation, ``(B, T, ...)``.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models.generative._common._rssm import RSSM
        >>> rssm = RSSM(stoch_size=4, deter_size=8, hidden_size=8,
        ...             action_dim=2, embed_size=6).eval()
        >>> embed, actions = lucid.randn((2, 5, 6)), lucid.randn((2, 5, 2))
        >>> priors, posteriors = rssm.observe(embed, actions)
        >>> priors.mean.shape, posteriors.mean.shape
        ((2, 5, 4), (2, 5, 4))
        >>> _, full = rssm.observe(embed, actions, sample=False)
        >>> _, head = rssm.observe(embed[:, :3], actions[:, :3], sample=False)
        >>> carry = head.map(lambda t: t[:, -1])
        >>> _, tail = rssm.observe(embed[:, 3:], actions[:, 3:], carry, sample=False)
        >>> bool(lucid.allclose(tail.stoch, full.stoch[:, 3:]))  # chunked == one pass
        True
        """
        if embed.ndim != 3 or actions.ndim != 3:
            raise ValueError(
                f"observe expects (B, T, ·) inputs, got embed {embed.shape} "
                f"and actions {actions.shape}"
            )
        if int(embed.shape[1]) != int(actions.shape[1]):
            raise ValueError(
                f"embed and actions must agree on T, got {int(embed.shape[1])} "
                f"and {int(actions.shape[1])}"
            )

        if state is None:
            state = self.initial(int(embed.shape[0]), device=embed.device.type)
        priors: list[RSSMState] = []
        posteriors: list[RSSMState] = []
        for t in range(int(embed.shape[1])):
            prior, state = self.posterior_step(
                state, actions[:, t], embed[:, t], sample=sample
            )
            priors.append(prior)
            posteriors.append(state)
        return self._stack(priors), self._stack(posteriors)

    def imagine(
        self, state: RSSMState, actions: Tensor, *, sample: bool = True
    ) -> RSSMState:
        r"""Roll the prior forward with no observations at all.

        This is the model dreaming: every step's latent comes from the
        dynamics' own prediction rather than from an encoded frame, so the
        trajectory is a function of the parameters and the actions only.

        Parameters
        ----------
        state : RSSMState
            Starting state — typically one step of a posterior produced by
            :meth:`observe`.
        actions : Tensor
            Actions to imagine taking, ``(B, T, action_dim)``.
        sample : bool, default=True
            Draw each latent, or take its mean when ``False``.  Planning
            wants the mean: a search that re-samples every candidate ranks
            its own noise instead of the actions.

        Returns
        -------
        RSSMState
            The imagined prior states, ``(B, T, ...)``.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models.generative._common._rssm import RSSM
        >>> rssm = RSSM(stoch_size=4, deter_size=8, hidden_size=8,
        ...             action_dim=2, embed_size=6).eval()
        >>> _, posteriors = rssm.observe(lucid.randn((2, 5, 6)),
        ...                              lucid.randn((2, 5, 2)))
        >>> start = posteriors.map(lambda t: t[:, -1])
        >>> plan = lucid.zeros((2, 7, 2))  # seven steps, past the observed five
        >>> dream = rssm.imagine(start, plan, sample=False)
        >>> dream.deter.shape, dream.stoch.shape
        ((2, 7, 8), (2, 7, 4))
        >>> again = rssm.imagine(start, plan, sample=False)
        >>> bool(lucid.allclose(dream.stoch, again.stoch))  # start + actions only
        True
        """
        if actions.ndim != 3:
            raise ValueError(f"imagine expects (B, T, ·) actions, got {actions.shape}")
        states: list[RSSMState] = []
        for t in range(int(actions.shape[1])):
            state = self.prior_step(state, actions[:, t], sample=sample)
            states.append(state)
        return self._stack(states)

    @override
    def forward(
        self,
        embed: Tensor,
        actions: Tensor,
        state: RSSMState | None = None,
    ) -> tuple[RSSMState, RSSMState]:
        """Alias for :meth:`observe`, so the module is callable."""
        return self.observe(embed, actions, state)

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return (
            f"stoch_size={self.stoch_size}, deter_size={self.deter_size}, "
            f"action_dim={self.action_dim}, embed_size={self.embed_size}"
        )


def rssm_kl(
    posterior: RSSMState, prior: RSSMState, *, free_nats: float = 3.0
) -> Tensor:
    r"""Scalar :math:`\mathrm{KL}(q \,\|\, p)` with a free-nats floor.

    The divergence is summed over the latent dimension, averaged over batch
    and time, and then clipped from below:

    .. math::

        \mathcal{L}_{\mathrm{KL}} =
            \max\big(0,\;
                \mathbb{E}\,\mathrm{KL}\big(q(s \mid h, o)
                    \,\|\, p(s \mid h)\big) - \texttt{free\_nats}\big).

    Parameters
    ----------
    posterior : RSSMState
        The belief that has seen the observation — the first argument of
        the divergence.
    prior : RSSMState
        The dynamics' own prediction.
    free_nats : float, default=3.0
        Divergence below this many nats costs nothing.

    Returns
    -------
    Tensor
        A 0-d tensor.

    Notes
    -----
    Without the floor the posterior collapses onto the prior: the cheapest
    way to drive the divergence to zero is to stop encoding the
    observation, and then the latent carries nothing. The threshold buys
    the model that much divergence for free, so gradient is only spent
    once the two distributions genuinely disagree.

    Two forms of this appear in the literature and they are **not**
    interchangeable to read, only to differentiate: PlaNet clips the
    excess, :math:`\max(0, \mathrm{KL} - \texttt{free\_nats})`, while the
    Dreamer line clips the value, :math:`\max(\mathrm{KL},
    \texttt{free\_nats})`. Their gradients are identical — zero below the
    threshold, one above — but the reported numbers differ by a constant.
    This is the PlaNet form, so a value printed here is comparable with the
    PlaNet paper and sits ``free_nats`` below a Dreamer-style report.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative._common._rssm import RSSM, rssm_kl
    >>> rssm = RSSM(stoch_size=4, deter_size=8, hidden_size=8,
    ...             action_dim=2, embed_size=6).eval()
    >>> priors, posteriors = rssm.observe(lucid.randn((2, 3, 6)),
    ...                                   lucid.randn((2, 3, 2)))
    >>> float(rssm_kl(posteriors, priors, free_nats=1e9).item())
    0.0
    """
    if free_nats < 0.0:
        raise ValueError(f"free_nats must be non-negative, got {free_nats}")
    if posterior.is_discrete != prior.is_discrete:
        raise ValueError(
            "prior and posterior disagree on their parameterisation: one is "
            "categorical and the other Gaussian"
        )

    if posterior.is_discrete:
        per_step = categorical_kl(
            cast(Tensor, posterior.logits), cast(Tensor, prior.logits)
        )
    else:
        posterior_mean, _ = posterior.gaussian()
        prior_mean, _ = prior.gaussian()
        per_step = normal_kl(
            posterior_mean, posterior.logvar, prior_mean, prior.logvar
        ).sum(dim=-1)

    kl = per_step.mean()
    return (kl - free_nats).clip(0.0, None) if free_nats > 0.0 else kl


def categorical_kl(posterior_logits: Tensor, prior_logits: Tensor) -> Tensor:
    r"""KL between two grids of independent categoricals, summed over the grid.

    Parameters
    ----------
    posterior_logits, prior_logits : Tensor
        Unnormalised class scores, ``(B, T, stoch_size, discrete)``.

    Returns
    -------
    Tensor
        ``(B, T)`` — one divergence per step, the variables' summed.

    Notes
    -----
    Computed from log-probabilities rather than through
    :func:`lucid.distributions.kl_divergence`, which has no registered
    pair for one-hot categoricals.  The closed form is a sum, so there is
    nothing to approximate:

    .. math::

        \mathrm{KL}(q \,\|\, p)
            = \sum_k q_k \big(\log q_k - \log p_k\big).

    Taking both sides through ``log_softmax`` keeps it stable: the
    subtraction happens in log space, so a class the prior has written off
    contributes a large finite number rather than ``inf``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative._common._rssm import categorical_kl
    >>> logits = lucid.zeros(2, 4, 8)
    >>> categorical_kl(logits, logits).shape
    (2,)

    Summed over the stochastic axes and kept per batch element, which is
    what the free-bits clamp is applied to.
    """
    posterior_log = F.log_softmax(posterior_logits, dim=-1)
    prior_log = F.log_softmax(prior_logits, dim=-1)
    probs = posterior_log.exp()
    per_variable = (probs * (posterior_log - prior_log)).sum(dim=-1)
    return per_variable.sum(dim=-1)
