"""MeanFlow configuration — Geng, Deng, Bai, Kolter & He, 2025.

The fifth step of the flow lineage, and the one that stops integrating.
NICE and RealNVP compose hand-designed bijections; Neural ODE replaces
them with one vector field and pays for a solve on every gradient step;
Flow Matching removes the training solve by regressing onto a closed-form
target; Rectified Flow straightens the trajectory so an Euler step can
follow it.  All four still model the *instantaneous* velocity, and all
four therefore approximate an integral at sampling time.

MeanFlow models the **average** velocity over an interval instead —
the quantity that integral would have produced.  A network that has it
needs no integral: one evaluation of :math:`u_\\theta(\\epsilon, 0, 1)`
carries the whole path.  What makes this trainable is that the average
velocity's definition can be rearranged into an identity involving only
the instantaneous velocity, which is available in closed form.
"""

from dataclasses import dataclass
from typing import ClassVar, Literal, override

from lucid.models._meta import model_family_meta
from lucid.models.generative._common._config import (
    GenerativeModelConfig,
    check_patch_grid,
    patch_grid_side,
)

# How ``(r, t)`` reach the network.  The field is a function of both, but
# nothing forces the network to be conditioned on that exact pair — the
# paper compares four encodings (Table 1c) and the interval turns out to
# matter more than the endpoint.
#
#   * ``"t_interval"`` — :math:`(t,\\, t - r)`.  The paper's default and
#     its best result (FID 61.06 vs 61.75 for the raw pair).
#   * ``"t_r"``        — the raw :math:`(t,\\, r)`.
#   * ``"t_r_interval"`` — all three, :math:`(t,\\, r,\\, t - r)`.
#   * ``"interval"``   — :math:`t - r` alone.  Still produces meaningful
#     one-step samples (63.13), which is the ablation's point: the
#     interval is what the average velocity is really about.
TimeConditioning = Literal["t_interval", "t_r", "t_r_interval", "interval"]

# Where the pair ``(r, t)`` is drawn from before the larger is assigned to
# ``t``.  ``"lognorm"`` draws from a normal and squashes through the
# logistic, concentrating samples away from the endpoints; ``"uniform"``
# is the obvious alternative and measurably worse (65.90 vs 61.06,
# Table 1d).
TimeSampler = Literal["lognorm", "uniform"]


@model_family_meta(
    canonical_name="MeanFlow",
    citation=(
        "Geng, Zhengyang, Mingyang Deng, Xingjian Bai, J. Zico Kolter, "
        'and Kaiming He. "Mean Flows for One-step Generative Modeling." '
        "arXiv preprint arXiv:2505.13447, 2025."
    ),
    theory=r"""
    Flow Matching models the **instantaneous** velocity :math:`v(z_t, t)`
    — the tangent of the path at one instant — and recovers a sample by
    integrating it.  MeanFlow models the **average** velocity over an
    interval, defined as that integral divided by its width:

    .. math::

        u(z_t, r, t) \triangleq \frac{1}{t - r}\int_r^t v(z_\tau, \tau)\,d\tau .

    The two coincide only in the limit :math:`r \to t`.  Away from it the
    average velocity points along the *displacement* between the two
    times rather than along the curve, which is exactly what a one-step
    sampler needs: with :math:`u` in hand the whole path is
    :math:`z_r = z_t - (t - r)\,u(z_t, r, t)`, and generation from noise
    is a single evaluation at :math:`(r, t) = (0, 1)`.

    Training on that definition directly would require the integral.  The
    way through is to multiply it out and differentiate with respect to
    :math:`t`, holding :math:`r` fixed, which turns the integral into the
    **MeanFlow Identity**:

    .. math::

        u(z_t, r, t) = v(z_t, t) - (t - r)\,\frac{d}{dt}u(z_t, r, t).

    Every term on the right is available.  The instantaneous velocity is
    the closed-form conditional velocity Flow Matching already regresses
    onto, and the total derivative expands by the chain rule into

    .. math::

        \frac{d}{dt}u(z_t, r, t) = v(z_t, t)\,\partial_z u + \partial_t u,

    which is one Jacobian-vector product of :math:`u_\theta` along the
    tangent :math:`(v,\, 0,\, 1)` — a single extra forward-mode pass, not
    a second-order solve.  The identity becomes a regression target,

    .. math::

        u_{\mathrm{tgt}} = v_t - (t - r)\bigl(v_t\,\partial_z u_\theta
                                            + \partial_t u_\theta\bigr),

    and the loss is :math:`\lVert u_\theta - \mathrm{sg}(u_{\mathrm{tgt}})
    \rVert^2`.  The stop-gradient is what keeps the cost at one extra
    pass: without it the Jacobian-vector product would itself need to be
    differentiated.

    Setting :math:`r = t` collapses the second term and recovers Flow
    Matching exactly, so the method is Flow Matching with a modified
    target rather than a different framework.  The paper draws
    :math:`r \neq t` for only a quarter of samples; at 0% it *is* Flow
    Matching and one-step generation fails outright (FID 328.91 against
    61.06).
    """,
)
@dataclass(frozen=True)
class MeanFlowConfig(GenerativeModelConfig):
    r"""Configuration for the MeanFlow family.

    Parameters
    ----------
    sample_size : int, default=32
        Spatial extent of the field the model operates on.  For the
        paper's ImageNet models this is the VAE latent, ``32``, not the
        256-pixel image.
    in_channels, out_channels : int, default=4, 4
        Latent channels.  Four is the standard VAE tokenizer's width;
        the CIFAR-10 experiment works in pixel space at three.  The
        average velocity is a displacement in the sample's own space, so
        the two must be equal.
    patch_size : int, default=2
        Side of the square patch each token covers.  The suffix in the
        paper's variant names — ``B/2`` is the Base backbone at patch 2.
    hidden_size : int, default=768
        Transformer width.
    depth : int, default=12
        Number of transformer blocks.
    num_heads : int, default=12
        Attention heads per block.
    mlp_ratio : float, default=4.0
        Feed-forward expansion inside each block.
    num_classes : int, default=1000
        Label vocabulary for class conditioning.  The extra index
        ``num_classes`` is the unconditional token the guidance dropout
        substitutes.
    class_dropout : float, default=0.1
        Probability of replacing the label with the unconditional token
        during training, which is what leaves the model able to produce
        the unconditional field guidance needs.
    time_conditioning : {"t_interval", "t_r", "t_r_interval", "interval"}, default="t_interval"
        Which time variables the network is conditioned on.  The
        Jacobian-vector product is always taken with respect to
        :math:`u_\theta(\cdot, r, t)` regardless of this choice — the
        encoding changes the network's inputs, not the identity.
    time_sampler : {"lognorm", "uniform"}, default="lognorm"
        Distribution the pair ``(r, t)`` is drawn from.
    lognorm_mean, lognorm_std : float, default=-0.4, 1.0
        Parameters of the normal that ``"lognorm"`` squashes through the
        logistic.  The paper's ImageNet setting; its CIFAR-10 setting is
        ``(-2.0, 2.0)``.
    ratio_r_not_t : float, default=0.25
        Fraction of the batch that gets :math:`r \neq t`.  The rest
        trains the instantaneous velocity, which is what anchors the
        field.  Zero reduces the method to Flow Matching and does not
        produce usable one-step samples.
    adaptive_weight_power : float, default=1.0
        :math:`p` in the loss weight :math:`w = 1/(\lVert\Delta\rVert^2 +
        c)^p`.  ``0`` is the plain squared error; ``0.5`` is close to the
        pseudo-Huber loss of prior one-step work.
    adaptive_weight_eps : float, default=1e-3
        :math:`c` above — keeps the weight finite where the error is
        near zero.
    guidance_scale : float, default=1.0
        :math:`\omega` in the guided target.  One disables guidance and
        recovers the plain objective.
    guidance_mix : float, default=0.0
        :math:`\kappa`, which mixes the model's own class-conditional
        average velocity into the target alongside the class-unconditional
        one.  The effective scale a sampler sees is
        :math:`\omega' = \omega / (1 - \kappa)`.
    guidance_interval : tuple of float, default=(0.0, 1.0)
        The range of :math:`t` over which guidance is applied at all.
        The larger models restrict it; ``XL/2+`` uses ``(0.3, 0.8)``.

    Notes
    -----
    Reference: Geng, Deng, Bai, Kolter, and He, *"Mean Flows for One-step
    Generative Modeling"*, arXiv:2505.13447, 2025.  Architecture and
    training settings are Table 4; the ablations that pin the defaults
    are Table 1.

    The backbone is DiT's, unchanged — the paper is explicit that it
    keeps "the DiT architecture blocks untouched" and that architectural
    improvements are orthogonal.  What differs from a diffusion DiT is
    only the conditioning: two time variables rather than one, each
    embedded and passed through a two-layer MLP before being summed.

    Examples
    --------
    >>> from lucid.models.generative.mean_flow import MeanFlowConfig
    >>> config = MeanFlowConfig()
    >>> config.hidden_size, config.depth, config.patch_size
    (768, 12, 2)

    The variant names encode the backbone size and the patch side, so
    ``L/2`` is the Large backbone tokenised at two:

    >>> large = MeanFlowConfig(hidden_size=1024, depth=24, num_heads=16)
    >>> large.num_patches
    256
    """

    model_type: ClassVar[str] = "mean_flow"

    sample_size: int | tuple[int, int] = 32
    in_channels: int = 4
    out_channels: int = 4

    patch_size: int = 2
    hidden_size: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    frequency_embedding_size: int = 256

    num_classes: int = 1000
    class_dropout: float = 0.1

    time_conditioning: TimeConditioning = "t_interval"
    time_sampler: TimeSampler = "lognorm"
    lognorm_mean: float = -0.4
    lognorm_std: float = 1.0
    ratio_r_not_t: float = 0.25

    adaptive_weight_power: float = 1.0
    adaptive_weight_eps: float = 1e-3

    guidance_scale: float = 1.0
    guidance_mix: float = 0.0
    guidance_interval: tuple[float, float] = (0.0, 1.0)

    @property
    def num_patches(self) -> int:
        """Tokens the field is cut into — ``(sample_size / patch_size)ˆ2``."""
        return (patch_grid_side(self.sample_size) // self.patch_size) ** 2

    @override
    def __post_init__(self) -> None:
        super().__post_init__()
        check_patch_grid(
            self.sample_size, self.patch_size, self.hidden_size, self.num_heads
        )
        if self.depth <= 0:
            raise ValueError(f"depth must be positive, got {self.depth}")
        if self.in_channels != self.out_channels:
            raise ValueError(
                f"the average velocity lives in the sample's own space, so "
                f"in_channels must equal out_channels; got {self.in_channels} "
                f"and {self.out_channels}"
            )
        if not 0.0 <= self.ratio_r_not_t <= 1.0:
            raise ValueError(
                f"ratio_r_not_t is a probability, got {self.ratio_r_not_t}"
            )
        if not 0.0 <= self.class_dropout <= 1.0:
            raise ValueError(
                f"class_dropout is a probability, got {self.class_dropout}"
            )
        if self.adaptive_weight_eps <= 0.0:
            raise ValueError(
                f"adaptive_weight_eps must be positive — it exists to keep the "
                f"weight finite at zero error, got {self.adaptive_weight_eps}"
            )
        if self.guidance_mix >= 1.0:
            raise ValueError(
                f"guidance_mix must be below 1 — the effective scale is "
                f"omega / (1 - kappa), got {self.guidance_mix}"
            )
        low, high = self.guidance_interval
        if not 0.0 <= low <= high <= 1.0:
            raise ValueError(
                f"guidance_interval must be an ordered sub-range of [0, 1], "
                f"got {self.guidance_interval}"
            )
