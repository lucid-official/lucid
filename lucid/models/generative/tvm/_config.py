r"""Terminal Velocity Matching configuration — Zhou, Parger, Haque & Song, 2026.

The sixth step of the flow lineage, and MeanFlow's dual.  Flow Matching
learns the instantaneous velocity and integrates it at sampling time;
MeanFlow learns the displacement over an interval by differentiating it
at the interval's **start**.  TVM learns the same displacement but
differentiates it at the interval's **end** — the terminal time — where
the ground-truth velocity of the point the jump lands on is what it must
equal.  That shift buys a bound: the objective upper-bounds the
2-Wasserstein distance between data and model, provided the velocity
network is Lipschitz.  A Diffusion Transformer is not, so the family
also carries the paper's minimal architectural fix for it.
"""

from dataclasses import dataclass
from typing import ClassVar, Literal, override

from lucid.models._meta import model_family_meta
from lucid.models.generative._common._config import (
    GenerativeModelConfig,
    check_patch_grid,
    check_sample_space_channels,
    patch_grid_side,
)

# How ``(t, s)`` are drawn (Section 7.3, Table 3).  All three bias ``t``
# toward noise and ``s`` toward data so the network learns long strides.
#
#   * ``"gap"``   — the gap ``t - s`` is logit-normal, ``s`` is logit-normal
#     truncated so that ``s + gap <= 1``, and ``t = s + gap``.  The paper's
#     choice: it keeps improving after the other two plateau (Figure 8).
#   * ``"trunc"`` — ``t`` is logit-normal and ``s`` logit-normal truncated
#     to ``s <= t``.
#   * ``"clamp"`` — both are logit-normal and independent, and ``s`` is set
#     to ``t`` wherever it exceeds it.
TimeSampler = Literal["gap", "trunc", "clamp"]

# How the time-embedding MLPs are initialised (Table 6, "Time Embed init").
# Every other linear layer gets the spectral (Lipschitz) initialisation;
# these two choices are the only place the released runs differ.
TimeEmbeddingInit = Literal["spectral", "normal"]


@model_family_meta(
    canonical_name="TVM",
    citation=(
        "Zhou, Linqi, Mathias Parger, Ayaan Haque, and Jiaming Song. "
        '"Terminal Velocity Matching." International Conference on '
        "Learning Representations (ICLR), 2026.  arXiv:2511.19797."
    ),
    theory=r"""
    Write the ground-truth flow map from time :math:`t` to time :math:`s`
    as :math:`\psi(x_t, t, s) = x_t + \int_t^s u(x_r, r)\,dr` and its net
    displacement as :math:`f(x_t, t, s) = \psi(x_t, t, s) - x_t`.  The
    displacement satisfies two conditions: it *is* the integral, and its
    derivative at :math:`s = t` is the instantaneous velocity.  One
    two-time network parameterises both,

    .. math::

        f_\theta(x_t, t, s) = (s - t)\,F_\theta(x_t, t, s), \qquad
        u_\theta(x_t, t) = F_\theta(x_t, t, t),

    and the scaling :math:`(s - t)` makes :math:`f_\theta(x_t, t, t) = 0`
    hold by construction.

    The integral condition cannot be trained directly — it would need an
    ODE solve per sample.  Differentiating it with respect to the
    **terminal** time :math:`s` instead gives a condition with no
    integral in it,

    .. math::

        \frac{d}{ds} f(x_t, t, s) = u\bigl(\psi(x_t, t, s), s\bigr),

    and the squared error in that condition, integrated over
    :math:`s \in [0, t]`, upper-bounds the displacement error.  The
    unknowns on the right are replaced by the model itself — the jump
    lands at :math:`x_t + f_\theta(x_t, t, s)` and the velocity there is
    :math:`u_\theta` — which is only sound if :math:`u_\theta` is right,
    so Flow Matching is trained jointly.  It is the same condition at
    zero displacement:

    .. math::

        \mathcal{L}^{t,s}_{\mathrm{TVM}}
        = \Bigl\lVert \tfrac{d}{ds} f_\theta(x_t, t, s)
          - u_{\theta^*}\!\bigl(x_t + f_{\theta}(x_t, t, s), s\bigr)
          \Bigr\rVert^2
        + \bigl\lVert u_\theta(x_s, s) - v_s \bigr\rVert^2 .

    :math:`\theta^*` is an exponential moving average of the weights,
    held under a stop-gradient, and the derivative expands to
    :math:`F_\theta + (s - t)\,\partial_s F_\theta` — one Jacobian-vector
    product along :math:`s`, whose result is itself back-propagated
    through.  A weighted integral of this objective over :math:`s` bounds
    :math:`W_2^2` between the data and the model's push-forward when
    :math:`u_\theta(\cdot, s)` is Lipschitz (Theorem 1).

    **MeanFlow's dual.**  MeanFlow enforces
    :math:`\frac{d}{dt} f_\theta = -u(x_t, t)`, differentiating at the
    *start*; the tangent of that derivative contains the sampled velocity
    :math:`v_t`, whose magnitude under guidance varies widely.  TVM's
    tangent is a unit step in :math:`s`, independent of :math:`x_t` and
    :math:`t`, which is why it trains stably even with random guidance.

    **Guidance is learned, not applied.**  The class-conditional field is
    the guided one, :math:`w\,u(x, r, c) + (1 - w)\,u(x, r)`, with the
    unconditional half supplied by the EMA network.  Its magnitude grows
    linearly in :math:`w`, so the loss is weighted by :math:`1/w^2` and
    the network optionally predicts :math:`F` with the factor :math:`w`
    pulled out — the *scaled parameterisation*.

    **Semi-Lipschitz control.**  Every LayerNorm becomes a parameter-free
    RMSNorm, queries and keys are RMS-normalised, every adaLN modulation
    vector is itself RMS-normalised so its scale cannot grow without
    bound, and linear layers start at unit spectral norm.

    **Where Lucid departs from the authors' release, on purpose.**  The
    model matches the authors' code to float32 rounding (forward pass,
    terminal derivative, loss, every gradient and the sampler), but three
    choices differ, and they matter to anyone reproducing a released run
    from scratch:

    * *The patch embedding starts at unit spectral norm.*  Lucid divides
      the patch-embedding weight by its largest singular value, as the
      paper prescribes for linear layers.  The release's initialiser
      divides a rebound alias of the weight (``w.data = w.data / s[0]``)
      rather than the parameter, so its patch embedding keeps the
      Xavier-normal draw, :math:`\sigma_{\max} \approx 1.53` at full size.
    * *Precision.*  The release trains under bf16 autocast, and its
      Jacobian-vector product runs attention in bf16 even when autocast
      is off.  Lucid computes in the model's own dtype throughout —
      float32 unless the model is cast — including the terminal
      derivative.  A converted checkpoint sampled in Lucid is therefore
      not rounded where the authors' bf16 sampler is.
    * *No NaN masking.*  The release replaces a NaN loss with zero before
      the backward pass; Lucid returns the loss as computed, so a
      diverging step is seen instead of trained through.
    """,
)
@dataclass(frozen=True)
class TVMConfig(GenerativeModelConfig):
    r"""Configuration for the Terminal Velocity Matching family.

    Parameters
    ----------
    sample_size : int, default=32
        Side of the latent the network operates on — ``32`` for
        256-pixel ImageNet, ``64`` for 512-pixel, after the Stable
        Diffusion VAE's factor of eight.
    in_channels, out_channels : int, default=4, 4
        Latent channels.  The network predicts a velocity in the latent's
        own space, which the loss and the sampler add to the input, so
        the two must be equal.
    patch_size : int, default=2
        Side of the square patch each token covers.
    hidden_size : int, default=1152
        Transformer width.
    depth : int, default=28
        Number of transformer blocks.
    num_heads : int, default=18
        Attention heads per block.  DiT-XL uses 16; the paper uses 18 so
        the head dimension is 64, which its attention kernel runs 1.25x
        faster at, and reports the two train to the same FID.
    mlp_ratio : float, default=4.0
        Feed-forward expansion inside each block.
    frequency_embedding_size : int, default=256
        Width of the sinusoid each scalar condition is expanded into
        before its MLP.
    num_classes : int, default=1000
        Label vocabulary.  Index ``num_classes`` is the null label the
        guidance dropout substitutes.
    class_dropout : float, default=0.1
        Probability of training a sample unconditionally — its label is
        replaced by the null token and its guidance scale by one.
    qk_norm : bool, default=True
        RMS-normalise queries and keys (with a learnable scale) before
        attention.  Part of the semi-Lipschitz control.
    modulation_norm : bool, default=True
        RMS-normalise each adaLN shift, scale and gate vector (Eq. 16 and
        Appendix B.2).  Without it the modulation's magnitude, and with it
        the layer's Lipschitz constant, can grow unbounded — the
        activation blow-up of Figure 4.
    time_embedding_init : {"spectral", "normal"}, default="spectral"
        How the time and guidance MLPs start.  Every other linear layer
        is initialised to unit spectral norm regardless.
    guidance_scale : float, default=2.0
        The classifier-free guidance weight :math:`w` trained at, and the
        default a sampler uses for labelled draws.  The paper's best
        one-step 256-pixel model trains at ``2.0``.
    guidance_range : tuple of float or None, default=None
        ``(w_min, w_max)`` to train with a random guidance weight instead
        of a constant one: :math:`\beta = 1/w` is drawn uniformly from
        :math:`[1/w_{\max}, 1/w_{\min}]`.  Constant guidance wins in every
        comparison the paper makes (Table 4a), which is why ``None``.
    scaled_parameterization : bool, default=True
        Predict :math:`F` with the guidance weight factored out, so the
        model's velocity is :math:`w\,F_\theta` (Table 4c).
    time_sampler : {"gap", "trunc", "clamp"}, default="gap"
        Joint distribution of ``(t, s)``.
    gap_mean, gap_std : float, default=-0.8, 1.0
        Logit-normal parameters of ``t - s`` under ``"gap"``.
    t_mean, t_std : float, default=1.0, 1.0
        Logit-normal parameters of ``t`` under ``"trunc"`` and ``"clamp"``
        — Table 3's best setting for both.
    s_mean, s_std : float, default=-0.4, 1.0
        Logit-normal parameters of ``s``, and of the Flow Matching time
        when that is drawn separately.
    separate_flow_matching_time : bool, default=True
        Draw the Flow Matching term's time independently of the pair
        instead of reusing ``s`` — the sampler the paper names ``gap*``,
        which lets the Flow Matching term keep the time distribution that
        works for Flow Matching on its own (Figure 9).
    ratio_t_equals_s : float, default=0.0
        Fraction of the batch trained with ``s = t``, where the terminal
        term vanishes and only Flow Matching remains.  The paper finds any
        non-zero share hurts one-step quality (Table 4d).
    target_ema_decay : float, default=0.99
        :math:`\gamma`, the EMA rate of the network that supplies the
        terminal and unconditional targets.  ``0`` uses the live weights
        under a stop-gradient; ``0.99`` is the paper's sweet spot
        (Table 4b).
    eval_ema_decay : float or None, default=0.9999
        :math:`\gamma^{**}`, the rate of Algorithm 1's second average, the
        one that is evaluated and released (Table 6, "Eval EMA rate").
        ``None`` keeps no such copy, saving a network's worth of memory.
    detach_jvp : bool, default=False
        Stop the gradient at :math:`\partial_s F_\theta`.  Faster and
        lighter, but biases the gradient — the paper's runtime study
        offers it as a trade, not a default (Figure 10).
    latent_mean, latent_std : tuple of float
        Per-channel statistics the VAE latent is whitened with before the
        network sees it, :math:`z = (\ell - \mu)/\sigma` — IMM's setting,
        which the paper follows (Appendix F.1), in place of the usual
        single factor 0.18215.  The defaults are the 256-pixel ones; the
        512-pixel latent has its own.

    Notes
    -----
    Reference: Zhou, Parger, Haque, and Song, *"Terminal Velocity
    Matching"*, ICLR 2026 (arXiv:2511.19797).  Training settings are
    Table 6; the defaults here are its first column, the 256-pixel model
    at :math:`w = 2` behind the 3.29 one-step FID.

    The released code agrees with the paper on every setting above, with
    one exception it does not exercise: its random-guidance path draws
    :math:`w` rather than :math:`1/w` uniformly.  The paper's
    description is followed here.

    Examples
    --------
    >>> from lucid.models.generative.tvm import TVMConfig
    >>> config = TVMConfig()
    >>> config.hidden_size, config.depth, config.num_heads
    (1152, 28, 18)
    >>> config.num_patches
    256

    The 512-pixel model is the same network over a wider latent:

    >>> TVMConfig(sample_size=64).num_patches
    1024
    """

    model_type: ClassVar[str] = "tvm"

    sample_size: int | tuple[int, int] = 32
    in_channels: int = 4
    out_channels: int = 4

    patch_size: int = 2
    hidden_size: int = 1152
    depth: int = 28
    num_heads: int = 18
    mlp_ratio: float = 4.0
    frequency_embedding_size: int = 256

    num_classes: int = 1000
    class_dropout: float = 0.1

    qk_norm: bool = True
    modulation_norm: bool = True
    time_embedding_init: TimeEmbeddingInit = "spectral"

    guidance_scale: float = 2.0
    guidance_range: tuple[float, float] | None = None
    scaled_parameterization: bool = True

    time_sampler: TimeSampler = "gap"
    gap_mean: float = -0.8
    gap_std: float = 1.0
    t_mean: float = 1.0
    t_std: float = 1.0
    s_mean: float = -0.4
    s_std: float = 1.0
    separate_flow_matching_time: bool = True
    ratio_t_equals_s: float = 0.0

    target_ema_decay: float = 0.99
    eval_ema_decay: float | None = 0.9999
    detach_jvp: bool = False

    latent_mean: tuple[float, ...] = (0.86488, -0.27787343, 0.21616915, 0.3738409)
    latent_std: tuple[float, ...] = (4.85503674, 5.31922414, 3.93725398, 3.9870003)

    @property
    def num_patches(self) -> int:
        """Tokens the latent is cut into — ``(sample_size / patch_size)^2``."""
        return (patch_grid_side(self.sample_size) // self.patch_size) ** 2

    @override
    def __post_init__(self) -> None:
        super().__post_init__()
        if isinstance(self.sample_size, tuple) and (
            self.sample_size[0] != self.sample_size[1]
        ):
            raise ValueError(
                f"sample_size must be square — the positional table is built "
                f"over a square grid, got {self.sample_size}"
            )
        check_patch_grid(
            self.sample_size, self.patch_size, self.hidden_size, self.num_heads
        )
        if self.depth <= 0:
            raise ValueError(f"depth must be positive, got {self.depth}")
        self._check_guidance()
        self._check_times()
        self._check_latent()
        self._check_averages()

    def _check_guidance(self) -> None:
        """Class dropout is a probability; ``w`` and its range are positive."""
        if not 0.0 <= self.class_dropout <= 1.0:
            raise ValueError(
                f"class_dropout is a probability, got {self.class_dropout}"
            )
        if self.guidance_scale <= 0.0:
            raise ValueError(
                f"guidance_scale must be positive — the network is conditioned "
                f"on its reciprocal, got {self.guidance_scale}"
            )
        if self.guidance_range is not None:
            low, high = self.guidance_range
            if not 0.0 < low <= high:
                raise ValueError(
                    f"guidance_range must be an ordered pair of positive "
                    f"weights, got {self.guidance_range}"
                )

    def _check_times(self) -> None:
        """Logit-normal spreads are positive; the ``s = t`` share is a probability."""
        for name in ("gap_std", "t_std", "s_std"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if not 0.0 <= self.ratio_t_equals_s <= 1.0:
            raise ValueError(
                f"ratio_t_equals_s is a probability, got {self.ratio_t_equals_s}"
            )

    def _check_latent(self) -> None:
        """Output channels match the latent's; each channel's whitening is valid."""
        check_sample_space_channels(self.in_channels, self.out_channels, "the velocity")
        if not len(self.latent_mean) == len(self.latent_std) == self.in_channels:
            raise ValueError(
                f"latent_mean and latent_std need one entry per latent channel "
                f"({self.in_channels}), got {len(self.latent_mean)} and "
                f"{len(self.latent_std)}"
            )
        if any(value <= 0.0 for value in self.latent_std):
            raise ValueError(f"latent_std must be positive, got {self.latent_std}")

    def _check_averages(self) -> None:
        """Both EMA rates lie in ``[0, 1)``; the evaluation one may be absent."""
        if not 0.0 <= self.target_ema_decay < 1.0:
            raise ValueError(
                f"target_ema_decay must be in [0, 1) — at 1 the target never "
                f"leaves its initialisation, got {self.target_ema_decay}"
            )
        if self.eval_ema_decay is not None and not 0.0 <= self.eval_ema_decay < 1.0:
            raise ValueError(
                f"eval_ema_decay must be in [0, 1) or None, got {self.eval_ema_decay}"
            )
