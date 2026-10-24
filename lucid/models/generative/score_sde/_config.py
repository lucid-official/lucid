"""Score-SDE configuration — Song et al., 2021.

The family's knobs are the choice of SDE and its noise range; the network
underneath is DDPM's U-Net, reused rather than rewritten, because the
paper's NCSN++/DDPM++ are that architecture with modifications this
implementation does not carry (see :class:`ScoreSDEConfig`'s Notes).
"""

from dataclasses import dataclass, field
from typing import ClassVar, override

from lucid.models._meta import model_family_meta
from lucid.models.generative._common._config import (
    GenerativeModelConfig,
    check_sample_space_channels,
)


@model_family_meta(
    canonical_name="ScoreSDE",
    citation=(
        'Song, Yang, et al. "Score-Based Generative Modeling through '
        'Stochastic Differential Equations." International Conference on '
        "Learning Representations, 2021."
    ),
    theory=r"""
    Score-SDE's contribution is that two families which looked separate
    are the same object viewed at different resolutions.

    Noise-conditional score networks perturb data with a geometric ladder
    of Gaussian noise; denoising diffusion perturbs it with a Markov
    chain of small Gaussian steps.  Take the number of levels to infinity
    and both become stochastic differential equations:

    .. math::

        dx = f(x, t)\,dt + g(t)\,dw.

    NCSN's limit has zero drift and an exploding variance (**VE**);
    DDPM's has a contracting drift and a variance held at one (**VP**).
    The paper adds a third, **sub-VP**, whose variance is bounded by the
    VP process at every instant and which gives its best likelihoods.

    What the continuous view buys is not a better model but better
    machinery, and three pieces of it.

    **Any solver.**  Sampling is integrating the reverse-time SDE

    .. math::

        dx = \big[f(x, t) - g(t)^2 \nabla_x \log p_t(x)\big] dt
             + g(t)\, d\bar{w},

    so any SDE solver works, and the number of steps stops being a
    property of the trained model.

    **Predictor-Corrector.**  A solver step can be followed by Langevin
    steps at fixed :math:`t`, which correct the discretisation error the
    solver leaves behind rather than letting it accumulate.

    **The probability flow ODE.**  Every diffusion has a deterministic
    process sharing its marginals exactly:

    .. math::

        dx = \Big[f(x, t) - \tfrac{1}{2} g(t)^2
             \nabla_x \log p_t(x)\Big] dt.

    That turns a generative model into a continuous normalising flow —
    adaptive step sizes, an invertible encoding, and exact likelihoods —
    without retraining anything.
    """,
)
@dataclass(frozen=True)
class ScoreSDEConfig(GenerativeModelConfig):
    r"""Frozen configuration for the Score-SDE family.

    Parameters
    ----------
    sde_type : {"vp", "ve", "subvp"}, default="vp"
        Which of the three processes.  ``"ve"`` is NCSN's continuous
        limit, ``"vp"`` is DDPM's, ``"subvp"`` is the one this paper
        introduces.
    sigma_min, sigma_max : float, default=0.01, 50.0
        VE's geometric noise range.  ``sigma_min`` is the paper's fixed
        0.01; ``sigma_max`` is chosen per dataset and 50 is its CIFAR-10
        value.
    beta_min, beta_max : float, default=0.1, 20.0
        VP and sub-VP's linear rate range, set by the paper "to match the
        settings in Ho et al. (2020)".
    num_scales : int, default=1000
        Steps a discrete sampler takes.  The continuous model does not
        depend on it — it is a property of the sampler, and the whole
        point of the framework is that it can be changed after training.
    snr : float, default=0.16
        Signal-to-noise ratio of the Langevin corrector step.
    corrector_steps : int, default=1
        Langevin steps per predictor step.  ``0`` makes the sampler
        predictor-only.
    base_channels, channel_mult, num_res_blocks, attention_resolutions : ...
        The U-Net, forwarded to the DDPM U-Net.
    num_heads, dropout, resnet_groups : ...
        As above.

    Notes
    -----
    Reference: Song, Sohl-Dickstein, Kingma, Kumar, Ermon, and Poole,
    *"Score-Based Generative Modeling through Stochastic Differential
    Equations"*, ICLR, 2021 (arXiv:2011.13456).

    **The network is DDPM's U-Net.**  The paper's NCSN++ and DDPM++ are
    that architecture plus a set of separately-ablated modifications —
    FIR up/downsampling, rescaled skip connections, progressive growing,
    and doubled depth for the "deep" rows of its Table 3.  None of those
    are implemented here, so a configuration in this family is the
    paper's SDE machinery over the backbone this zoo already ships, and
    the reported FID numbers do not transfer.  The SDEs, the samplers and
    the objective are the paper's; the backbone is not.

    **Continuous time reaches the U-Net through the discrete embedding.**
    The DDPM U-Net conditions on an integer timestep, so a continuous
    :math:`t \in [0, 1]` is scaled by ``num_scales - 1`` before it is
    handed over — which is what the reference implementation does for its
    continuous VP models.

    Examples
    --------
    >>> from lucid.models.generative.score_sde import ScoreSDEConfig
    >>> cfg = ScoreSDEConfig()
    >>> cfg.sde_type, cfg.beta_min, cfg.beta_max
    ('vp', 0.1, 20.0)
    """

    model_type: ClassVar[str] = "score_sde"

    sde_type: str = "vp"

    sigma_min: float = 0.01
    sigma_max: float = 50.0
    beta_min: float = 0.1
    beta_max: float = 20.0

    num_scales: int = 1000
    snr: float = 0.16
    corrector_steps: int = 1

    base_channels: int = 128
    channel_mult: tuple[int, ...] = (1, 2, 2, 2)
    num_res_blocks: int = 2
    attention_resolutions: tuple[int, ...] = field(default_factory=lambda: (16,))
    num_heads: int = 4
    dropout: float = 0.1
    resnet_groups: int = 32

    @override
    def __post_init__(self) -> None:
        super().__post_init__()
        check_sample_space_channels(self.in_channels, self.out_channels, "the score")
        if self.sde_type not in ("vp", "ve", "subvp"):
            raise ValueError(
                f"sde_type must be 'vp', 've' or 'subvp', got {self.sde_type!r}"
            )
        if not 0.0 < self.sigma_min < self.sigma_max:
            raise ValueError(
                f"need 0 < sigma_min < sigma_max, got {self.sigma_min}, "
                f"{self.sigma_max}"
            )
        if not 0.0 < self.beta_min < self.beta_max:
            raise ValueError(
                f"need 0 < beta_min < beta_max, got {self.beta_min}, {self.beta_max}"
            )
        if self.num_scales < 1:
            raise ValueError(f"num_scales must be at least 1, got {self.num_scales}")
        if self.snr <= 0.0:
            raise ValueError(f"snr must be positive, got {self.snr}")
        if self.corrector_steps < 0:
            raise ValueError(
                f"corrector_steps must be non-negative, got {self.corrector_steps}"
            )
