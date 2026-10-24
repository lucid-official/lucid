"""Rectified Flow configuration — Liu, Gong & Liu, 2023.

The fourth and last step of the flow lineage.  NICE and RealNVP compose
hand-designed bijections; Neural ODE replaces them with one vector field
and pays for a solve on every gradient step; Flow Matching removes the
solve by regressing onto a closed-form target.  All three still integrate
a curved trajectory at sampling time.

Rectified Flow asks what it would take for that trajectory to be
*straight*, and answers with a procedure — **reflow** — that can be
applied repeatedly to an already-trained model.  A straight path is one an
Euler step follows exactly, so the payoff is measured in function
evaluations: not fewer than a hundred, but one.
"""

from dataclasses import dataclass, field
from typing import ClassVar, Literal, override

from lucid.models._meta import model_family_meta
from lucid.models.generative._common._config import (
    GenerativeModelConfig,
    check_sample_space_channels,
)

# Which times the training objective draws.  ``"uniform"`` is the
# rectified-flow objective itself and the one reflow retrains under;
# the rest fix ``t`` and turn the same loss into a distillation.
#
#   * ``"t0"``  — one-step distillation.  Regressing the field at
#     ``t = 0`` onto ``x1 - x0`` makes a single Euler step land on the
#     data, which is the paper's k = 1 distillation.
#   * ``"t1"``  — the same at ``t = 1``, distilling the reverse map.
#   * ``int k`` — k-step distillation: ``t`` is drawn from the k-point
#     grid an Euler sampler with k steps would visit.
TimeSchedule = Literal["uniform", "t0", "t1"]

# Where the time embedding's frequencies come from.  ``"positional"`` is
# the sinusoidal ladder of discrete-time diffusion models; ``"fourier"``
# is the random Gaussian projection score-based models use at high
# resolution.
TimeEmbedding = Literal["positional", "fourier"]

# Whether the network carries a resolution pyramid alongside the U-Net
# trunk.  ``"none"`` is the plain architecture; the skip variants feed the
# input down / read the output up through progressively resized copies.
Progressive = Literal["none", "output_skip"]
ProgressiveInput = Literal["none", "input_skip"]

# How the divergence is obtained when scoring likelihood.  Training needs
# none of this — the objective is a regression — but reporting bits/dim
# does.
TraceMethod = Literal["exact", "hutchinson"]

# Distribution the Hutchinson probe is drawn from.
TraceNoise = Literal["rademacher", "gaussian"]


@model_family_meta(
    canonical_name="Rectified Flow",
    citation=(
        "Liu, Xingchao, Chengyue Gong, and Qiang Liu. "
        '"Flow Straight and Fast: Learning to Generate and Transfer Data '
        'with Rectified Flow." International Conference on Learning '
        "Representations, 2023."
    ),
    theory=r"""
    Given any coupling of two distributions — in the generative case
    :math:`X_0 \sim \pi_0` noise and :math:`X_1 \sim \pi_1` data, drawn
    independently — the **rectified flow** induced by that pair is the ODE

    .. math::

        \mathrm{d}Z_t = v(Z_t, t)\,\mathrm{d}t, \qquad Z_0 \sim \pi_0,

    whose field is fitted along the straight line between the endpoints,

    .. math::

        \min_v \int_0^1 \mathbb{E}\Bigl[
            \bigl\| (X_1 - X_0) - v(X_t, t) \bigr\|^2
        \Bigr]\,\mathrm{d}t,
        \qquad X_t = t X_1 + (1 - t) X_0 .

    This is the paper's eq. (1), and its minimiser is the conditional
    expectation :math:`v^X(x, t) = \mathbb{E}[X_1 - X_0 \mid X_t = x]`.

    **Relation to what came before.**  The objective is exactly Flow
    Matching's optimal-transport path at :math:`\sigma_{\min} = 0`: the
    interpolation is the same, and the regression target
    :math:`X_1 - X_0` is the same and equally independent of :math:`t`.
    What Rectified Flow adds is not a different loss but what is done
    *with* the trained model.

    **Marginal preservation (Theorem 3.3).**  The solve is not merely a
    sampler that happens to work — :math:`\mathrm{Law}(Z_t) =
    \mathrm{Law}(X_t)` at every :math:`t`, so :math:`Z_1` is distributed
    as the data while :math:`(Z_0, Z_1)` is a *different*, non-independent
    coupling of the same two marginals.  That is what makes the next step
    legitimate.

    **Reflow.**  Because the endpoints of the solve are themselves a valid
    coupling, the whole procedure can be applied to its own output:

    .. math::

        \mathbf{Z}^{k+1} = \mathtt{RectFlow}\bigl((Z_0^k, Z_1^k)\bigr),
        \qquad (Z_0^0, Z_1^0) = (X_0, X_1).

    Each round trains on the *previous* model's (noise, sample) pairs
    instead of independent draws.  The marginals never move — every
    :math:`\mathbf{Z}^k` generates the same data distribution — but the
    transport cost never increases and the paths get straighter.

    **Straightness** is what improves, and it is measurable:

    .. math::

        S(\mathbf{Z}) = \int_0^1 \mathbb{E}\Bigl[
            \bigl\| (Z_1 - Z_0) - \dot{Z}_t \bigr\|^2
        \Bigr]\,\mathrm{d}t ,

    zero exactly when every trajectory is a straight line traversed at
    constant speed.  A straight flow is one the forward Euler method
    integrates **without discretisation error**, so :math:`S \to 0` is
    precisely the statement that one step suffices — which is why the
    paper reports a state-of-the-art one-step FID of 4.85 on CIFAR-10.

    **Distillation** is the same loss with :math:`t` pinned rather than
    drawn: regressing :math:`v(\cdot, 0)` onto :math:`Z_1 - Z_0` over the
    reflow pairs makes :math:`z_0 + v(z_0, 0)` a direct one-step map.
    Reflow is what makes that regression easy — it is fitting an already
    nearly-straight path, not shortcutting a curved one.

    Time runs from noise to data: :math:`t = 0` is :math:`\pi_0` and
    :math:`t = 1` is :math:`\pi_1`.
    """,
)
@dataclass(frozen=True)
class RectifiedFlowConfig(GenerativeModelConfig):
    r"""Frozen configuration for every Rectified Flow variant.

    Parameters
    ----------
    sample_size : int or tuple of int, default=32
        Spatial resolution of a sample.
    in_channels : int, default=3
        Channels per sample.
    out_channels : int, default=3
        Kept equal to ``in_channels``: the field is a velocity in the
        sample's own space, so it cannot change the dimension.
    act_fn : {"silu", "swish", "relu", "gelu"}, default="silu"
        Activation inside the velocity field.  The reference
        implementation calls this ``swish``, which is the same function.
    base_channels : int, default=128
        Width of the first stage — the reference configs' ``nf``.
    channel_mult : tuple of int, default=(1, 2, 2, 2)
        Per-stage width multipliers; the length sets the ladder depth.
    num_res_blocks : int, default=4
        Residual blocks per resolution.  Four, not two, on CIFAR-10.
    attention_resolutions : tuple of int, default=(16,)
        Spatial sizes at which self-attention is applied — one resolution
        only, unlike the multi-resolution attention of the U-Net Flow
        Matching borrows.
    dropout : float, default=0.15
        Dropout inside residual blocks.
    resnet_groups : int, default=32
        Upper bound on GroupNorm groups; the actual count is
        ``min(channels // 4, resnet_groups)``, as in the reference.
    fir : bool, default=False
        Whether resampling goes through a finite-impulse-response filter
        rather than nearest-neighbour / average pooling.  Off at
        ``32 x 32``, on at ``256 x 256``.
    fir_kernel : tuple of int, default=(1, 3, 3, 1)
        Separable filter taps, normalised to unit sum at build time.
    progressive : {"none", "output_skip"}, default="none"
        Whether the output is accumulated through a resolution pyramid.
    progressive_input : {"none", "input_skip"}, default="none"
        Whether a downsampled copy of the input is injected at each stage.
    embedding_type : {"positional", "fourier"}, default="positional"
        Where the time embedding's frequencies come from.
    fourier_scale : float, default=16.0
        Standard deviation of the random frequencies when
        ``embedding_type="fourier"``.  Ignored otherwise.
    skip_rescale : bool, default=True
        Divide residual and attention sums by :math:`\sqrt{2}`, keeping
        the activation variance from growing with depth.
    init_scale : float, default=0.0
        Gain applied to the last convolution of each block.  Zero makes
        every residual branch start as the identity, which is what the
        reference configs set.
    scale_by_sigma : bool, default=False
        Divide the network output by the conditioning value ``t * 999``.
        All four released 256x256 configs set this; the CIFAR-10 one does
        not, hence the default.  It is part of the trained function, so a
        checkpoint and its flag have to agree.
    t_schedule : {"uniform", "t0", "t1"} or int, default="uniform"
        Which times the objective draws.  ``"uniform"`` is the
        rectified-flow / reflow objective; the others pin ``t`` and turn
        the same loss into k-step distillation.
    time_eps : float, default=1e-3
        Lower end of the time interval.  Times are drawn from
        ``[time_eps, 1]`` and solves run over the same range, matching the
        reference implementation.
    solver : str, default="dopri5"
        Method handed to :func:`lucid.diffeq.odeint`.  ``dopri5`` is the
        embedded Runge–Kutta 4(5) pair the reference calls ``rk45``.
    rtol, atol : float, default=1e-5, 1e-5
        Solver tolerances — the reference's ``ode_tol``.
    trace_method : {"exact", "hutchinson"} or None, default=None
        How the divergence is obtained when scoring likelihood.  ``None``
        picks by dimension.  Irrelevant to training.
    trace_noise : {"rademacher", "gaussian"}, default="rademacher"
        Probe distribution for the Hutchinson estimate.
    exact_trace_max_dim : int, default=32
        Dimension below which ``trace_method=None`` resolves to
        ``"exact"``.

    Notes
    -----
    ``in_channels`` and ``out_channels`` must agree: the field lives in
    the sample's own space.

    Examples
    --------
    >>> from lucid.models.generative.rectified_flow import RectifiedFlowConfig
    >>> RectifiedFlowConfig().t_schedule
    'uniform'
    >>> RectifiedFlowConfig(sample_size=32, in_channels=3).data_dim
    3072
    >>> RectifiedFlowConfig(t_schedule=4).is_distillation
    True
    """

    model_type: ClassVar[str] = "rectified_flow"

    base_channels: int = 128
    channel_mult: tuple[int, ...] = field(default_factory=lambda: (1, 2, 2, 2))
    num_res_blocks: int = 4
    attention_resolutions: tuple[int, ...] = field(default_factory=lambda: (16,))
    dropout: float = 0.15
    resnet_groups: int = 32

    fir: bool = False
    fir_kernel: tuple[int, ...] = field(default_factory=lambda: (1, 3, 3, 1))
    progressive: Progressive = "none"
    progressive_input: ProgressiveInput = "none"
    embedding_type: TimeEmbedding = "positional"
    fourier_scale: float = 16.0
    # The reference shifts [0, 1] inputs to [-1, 1] inside the network
    # when ``data.centered`` is False.  The CelebA-HQ and AFHQ configs
    # set it True (so no shift); the LSUN bedroom/church configs inherit
    # False from default_lsun_configs, so those two released models *do*
    # apply x -> 2x - 1 internally.  Building all four identically meant
    # the LSUN nets saw inputs an octave off.
    data_centered: bool = True
    skip_rescale: bool = True
    init_scale: float = 0.0
    scale_by_sigma: bool = False

    t_schedule: TimeSchedule | int = "uniform"
    time_eps: float = 1e-3

    solver: str = "dopri5"
    rtol: float = 1e-5
    atol: float = 1e-5

    trace_method: TraceMethod | None = None
    trace_noise: TraceNoise = "rademacher"
    exact_trace_max_dim: int = 32

    @property
    def data_dim(self) -> int:
        """int: Flattened width of one sample."""
        size = self.sample_size
        height, width = size if isinstance(size, tuple) else (size, size)
        return self.in_channels * height * width

    @property
    def is_distillation(self) -> bool:
        """bool: Whether ``t_schedule`` pins the time rather than drawing it.

        True for ``"t0"``, ``"t1"`` and any integer ``k``.  A distilling
        model is trained to be stepped a fixed number of times, so its
        loss is only meaningful on reflow pairs.
        """
        return self.t_schedule != "uniform"

    @property
    def resolved_trace_method(self) -> TraceMethod:
        """TraceMethod: ``trace_method``, or the one implied by the dimension."""
        if self.trace_method is not None:
            return self.trace_method
        return "exact" if self.data_dim <= self.exact_trace_max_dim else "hutchinson"

    @override
    def __post_init__(self) -> None:
        super().__post_init__()
        check_sample_space_channels(
            self.in_channels, self.out_channels, "the velocity field"
        )
        if self.base_channels <= 0:
            raise ValueError(
                f"base_channels must be positive, got {self.base_channels}"
            )
        if not self.channel_mult:
            raise ValueError("channel_mult must name at least one stage")
        if any(m <= 0 for m in self.channel_mult):
            raise ValueError(
                f"channel_mult entries must be positive, got {self.channel_mult}"
            )
        if self.num_res_blocks <= 0:
            raise ValueError(
                f"num_res_blocks must be positive, got {self.num_res_blocks}"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must lie in [0, 1), got {self.dropout}")
        if self.resnet_groups <= 0:
            raise ValueError(
                f"resnet_groups must be positive, got {self.resnet_groups}"
            )
        if not self.fir_kernel or any(k <= 0 for k in self.fir_kernel):
            raise ValueError(f"fir_kernel taps must be positive, got {self.fir_kernel}")
        if self.fourier_scale <= 0.0:
            raise ValueError(
                f"fourier_scale must be positive, got {self.fourier_scale}"
            )
        if self.init_scale < 0.0:
            raise ValueError(f"init_scale must be non-negative, got {self.init_scale}")
        # ``bool`` is a subtype of ``int``, so a stray ``True`` would sail
        # through an ``isinstance(..., int)`` test and silently mean k = 1.
        schedule = self.t_schedule
        if isinstance(schedule, int) and not isinstance(schedule, bool):
            if schedule <= 1:
                # k = 1 would put every draw at t = time_eps, which is what
                # ``"t0"`` already names; refusing it keeps one meaning per
                # value instead of two spellings for the same thing.
                raise ValueError(
                    f"an integer t_schedule is the k of k-step distillation and "
                    f"must exceed 1; for one-step distillation use 't0', got "
                    f"{schedule}"
                )
        elif schedule not in ("uniform", "t0", "t1"):
            raise ValueError(
                f"t_schedule must be 'uniform', 't0', 't1' or an integer "
                f"k > 1, got {schedule!r}"
            )
        if not 0.0 <= self.time_eps < 1.0:
            raise ValueError(f"time_eps must lie in [0, 1), got {self.time_eps}")
        if self.rtol <= 0.0 or self.atol <= 0.0:
            raise ValueError(
                f"solver tolerances must be positive, got rtol={self.rtol}, "
                f"atol={self.atol}"
            )
        if self.exact_trace_max_dim < 1:
            raise ValueError(
                f"exact_trace_max_dim must be at least 1, got "
                f"{self.exact_trace_max_dim}"
            )


__all__ = [
    "RectifiedFlowConfig",
    "TimeSchedule",
    "TimeEmbedding",
    "Progressive",
    "ProgressiveInput",
    "TraceMethod",
    "TraceNoise",
]
