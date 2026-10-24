"""Shared base configs for ``lucid.models.generative``.

Mirrors the role of :class:`LanguageModelConfig` for text models: the common
fields every generative family needs are captured here once, so family
configs only add their unique knobs.

Two tiers:

    * :class:`GenerativeModelConfig` — every generative family (VAE / DDPM /
      NCSN / future flow models).  Holds the image shape and the trunk
      activation.
    * :class:`DiffusionModelConfig` — adds noise-schedule knobs used by
      every diffusion family.  VAE skips this tier.
"""

from dataclasses import dataclass
from typing import ClassVar, Literal, override

from lucid.models._base import ModelConfig

# Activation alias accepted by every generative family.  Diffusion U-Nets
# typically use SiLU (Swish); VAEs use either SiLU or ReLU; the world
# models split — PlaNet specifies ReLU, Dreamer specifies ELU.
GenerativeActivation = Literal["silu", "swish", "relu", "gelu", "elu"]

# Noise schedule shape — see ``make_beta_schedule`` in ``_utils/_generative``.
BetaSchedule = Literal["linear", "cosine", "scaled_linear"]

# Factorised base distribution of a normalizing flow's latent space — see
# ``flow_prior_log_prob`` in ``_utils/_generative``.
FlowPrior = Literal["logistic", "gaussian"]


@dataclass(frozen=True)
class GenerativeModelConfig(ModelConfig):
    """Shared base for image-generative families.

    Args:
        sample_size: Output spatial resolution in pixels.  Square images use
            an int (``32`` → ``32 × 32``); rectangular targets pass a tuple.
        in_channels: Input image channels (3 for RGB, 1 for greyscale, 4
            for VAE latent samples, …).
        out_channels: Output channels — usually equal to ``in_channels``;
            diffusion models that predict variance use ``2 * in_channels``.
        act_fn: Activation used inside the trunk.  Most modern image
            generators default to ``"silu"``.

    Examples
    --------
    >>> from lucid.models.generative._common._config import GenerativeModelConfig
    >>> cfg = GenerativeModelConfig()
    >>> cfg.model_type
    'generative'
    >>> cfg.in_channels
    3
    """

    model_type: ClassVar[str] = "generative"

    sample_size: int | tuple[int, int] = 32
    in_channels: int = 3
    out_channels: int = 3
    act_fn: GenerativeActivation = "silu"

    def __post_init__(self) -> None:
        if isinstance(self.sample_size, tuple):
            # Annotated as a pair, but a dataclass does not enforce the arity.
            sides: tuple[int, ...] = self.sample_size
            if len(sides) != 2 or any(s <= 0 for s in sides):
                raise ValueError(
                    f"sample_size tuple must be (H, W) with both positive, got {self.sample_size}"
                )
        elif self.sample_size <= 0:
            raise ValueError(f"sample_size must be positive, got {self.sample_size}")
        if self.in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {self.in_channels}")
        if self.out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {self.out_channels}")


@dataclass(frozen=True)
class DiffusionModelConfig(GenerativeModelConfig):
    """Shared base for *diffusion*-family generative models.

    Args:
        num_train_timesteps: Number of forward-process steps ``T``.  DDPM
            uses 1000, NCSN traditionally a smaller grid.
        beta_start: First step's noise rate ``β_1``, the low end of the
            linear schedule.  Ignored when ``beta_schedule == "cosine"``.
        beta_end: Last step's noise rate ``β_T``, the high end of the
            same schedule.  Ignored when ``beta_schedule == "cosine"``.
        beta_schedule: ``"linear"`` (Ho et al., 2020), ``"scaled_linear"``
            (Rombach et al., 2022 — linear in ``sqrt(beta)``, then squared;
            *not* the same curve as ``"linear"`` between the same
            endpoints) or ``"cosine"`` (Nichol
            & Dhariwal, 2021 — improves low-resolution sample quality).
        prediction_type: What the network predicts at each step.  ``"epsilon"``
            (the noise, default) is the canonical Ho parameterisation;
            ``"sample"`` predicts ``x_0`` directly; ``"v_prediction"`` is the
            Imagen / Progressive Distillation reparameterisation.

    Examples
    --------
    >>> from lucid.models.generative._common._config import DiffusionModelConfig
    >>> cfg = DiffusionModelConfig()
    >>> cfg.model_type
    'diffusion'
    >>> cfg.in_channels
    3
    """

    model_type: ClassVar[str] = "diffusion"

    num_train_timesteps: int = 1_000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    beta_schedule: BetaSchedule = "linear"
    prediction_type: Literal["epsilon", "sample", "v_prediction"] = "epsilon"

    @override
    def __post_init__(self) -> None:
        super().__post_init__()
        if self.num_train_timesteps <= 0:
            raise ValueError(
                f"num_train_timesteps must be positive, got {self.num_train_timesteps}"
            )
        if not 0.0 < self.beta_start < self.beta_end < 1.0:
            raise ValueError(
                f"Require 0 < beta_start < beta_end < 1, got "
                f"start={self.beta_start}, end={self.beta_end}"
            )


@dataclass(frozen=True)
class NormalizingFlowConfig(GenerativeModelConfig):
    """Shared base for *normalizing-flow* families (exact likelihood).

    A flow is a bijection ``f`` between data and a factorised latent space,
    trained by maximising ``log p(x) = log p_H(f(x)) + log|det ∂f/∂x|``.
    Every family therefore needs to declare which base distribution the
    latent is measured against; the bijection's own knobs (coupling depth,
    scale parameterisation, …) stay in the family config.

    Args:
        prior: Factorised base distribution over ``h = f(x)``.
            ``"logistic"`` — heavier-tailed, the default in Dinh et al.,
            2014 for dequantised pixel data; ``"gaussian"`` — standard
            normal.

    Examples
    --------
    >>> from lucid.models.generative._common._config import NormalizingFlowConfig
    >>> cfg = NormalizingFlowConfig()
    >>> cfg.model_type
    'normalizing_flow'
    >>> cfg.in_channels
    3
    """

    model_type: ClassVar[str] = "normalizing_flow"

    prior: FlowPrior = "logistic"


__all__ = [
    "GenerativeModelConfig",
    "DiffusionModelConfig",
    "NormalizingFlowConfig",
    "WorldModelConfig",
    "GenerativeActivation",
    "BetaSchedule",
    "FlowPrior",
    "patch_grid_side",
    "check_patch_grid",
    "check_sample_space_channels",
]


#: The only frame size the shared world-model convolutional schedule produces —
#: see :mod:`lucid.models.generative._common._pixel_nets`.
WORLD_MODEL_IMAGE_SIZE = 64


@dataclass(frozen=True)
class WorldModelConfig(GenerativeModelConfig):
    r"""Shared base for *world-model* families (latent dynamics from pixels).

    A world model learns an environment's transition function in a compact
    latent space and generates futures there, never rendering a frame in
    order to decide anything.  Every family in this group carries PlaNet's
    recurrent state-space model — a deterministic path ``h`` beside a
    stochastic latent ``s`` — and reads pixels through the same
    convolutional stack, so the state geometry, the encoder width and the
    variational bound's two knobs are declared once here.

    What each family does *with* the learned dynamics is what separates
    them, and that stays in the family config: PlaNet plans over them,
    Dreamer trains an actor and a critic on them.

    Args:
        sample_size: Frame resolution.  Must be 64 — the shared
            convolutional schedule lands on that and nothing else.
        action_dim: Width of the action vector conditioning the
            transition.  Genuinely task-dependent — the Control Suite
            tasks range from 1 to 12 — so there is no paper value to
            inherit and it must be set per use.
        stoch_size: Width of the stochastic latent :math:`s`.
        deter_size: Width of the deterministic latent :math:`h`.
        hidden_size: Width of the hidden layer inside the RSSM heads.
        cnn_depth: Channel width of the first encoder convolution; the
            stack widens ``depth, 2*depth, 4*depth, 8*depth``.  A scale
            knob, not a shape knob — the spatial schedule is fixed.
        min_std: Floor on the latent standard deviation.
        mean_only: Take the latent's mean instead of sampling it, making
            the whole recurrence deterministic.
        free_nats: The KL is clamped below at this value before it enters
            the loss, so no gradient is spent driving an already-small
            divergence lower.
        kl_weight: Multiplier :math:`\beta` on the clamped KL.

    Notes:
        Every default here is a value **both** PlaNet (Hafner et al., 2019)
        and Dreamer (Hafner et al., 2020) state.  Fields the two papers
        specify differently — ``act_fn``, the reward head's width and depth
        — are deliberately *not* hoisted, because this class would then
        have to pick a number neither paper backs.  Promoting a field here
        means the families were observed to agree, not that they could.

    Examples
    --------
    >>> from lucid.models.generative._common._config import WorldModelConfig
    >>> cfg = WorldModelConfig()
    >>> cfg.model_type
    'world_model'
    >>> cfg.hidden_size, cfg.in_channels
    (200, 3)
    """

    model_type: ClassVar[str] = "world_model"

    sample_size: int | tuple[int, int] = WORLD_MODEL_IMAGE_SIZE

    action_dim: int = 1

    stoch_size: int = 30
    deter_size: int = 200
    hidden_size: int = 200
    cnn_depth: int = 32
    min_std: float = 0.1
    mean_only: bool = False

    free_nats: float = 3.0
    kl_weight: float = 1.0

    @override
    def __post_init__(self) -> None:
        super().__post_init__()

        size = self.sample_size
        square = size if isinstance(size, int) else None
        if isinstance(size, tuple):
            square = size[0] if size[0] == size[1] else None
        if square != WORLD_MODEL_IMAGE_SIZE:
            raise ValueError(
                f"the world models' convolutional schedule only produces "
                f"{WORLD_MODEL_IMAGE_SIZE}x{WORLD_MODEL_IMAGE_SIZE} frames; got "
                f"sample_size={self.sample_size}. Scale the model with "
                f"cnn_depth / deter_size / stoch_size instead."
            )

        for name, value in (
            ("action_dim", self.action_dim),
            ("stoch_size", self.stoch_size),
            ("deter_size", self.deter_size),
            ("hidden_size", self.hidden_size),
            ("cnn_depth", self.cnn_depth),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.min_std <= 0.0:
            raise ValueError(f"min_std must be positive, got {self.min_std}")
        if self.free_nats < 0.0:
            raise ValueError(f"free_nats must be non-negative, got {self.free_nats}")
        if self.kl_weight < 0.0:
            raise ValueError(f"kl_weight must be non-negative, got {self.kl_weight}")

    @property
    def embed_size(self) -> int:
        """Width of the encoder output — ``8 * cnn_depth`` over a 2x2 grid."""
        return 8 * self.cnn_depth * 2 * 2

    @property
    def latent_size(self) -> int:
        """Width of the full latent ``[h; s]`` the decoder and heads read."""
        return self.deter_size + self.stoch_size


# ── the patch grid of the DiT-backbone families ──────────────────────────────


def patch_grid_side(sample_size: int | tuple[int, int]) -> int:
    """The side of a square latent, given as an int or a ``(side, side)`` pair.

    Parameters
    ----------
    sample_size : int or tuple of int
        A config's ``sample_size``.

    Returns
    -------
    int
        ``sample_size`` itself, or its first entry.
    """
    return sample_size if isinstance(sample_size, int) else sample_size[0]


def check_patch_grid(
    sample_size: int | tuple[int, int],
    patch_size: int,
    hidden_size: int,
    num_heads: int,
) -> None:
    """Refuse a patch grid a DiT backbone cannot be built over.

    DiT, MeanFlow and TVM cut the latent into ``patch_size`` squares, embed
    each with a two-axis sine-cosine table and attend over them with
    ``num_heads`` heads; each of those needs the divisibility checked here.

    Parameters
    ----------
    sample_size : int or tuple of int
        Side of the latent.
    patch_size : int
        Side of a patch.
    hidden_size : int
        Token width.
    num_heads : int
        Attention heads.

    Raises
    ------
    ValueError
        If ``patch_size`` is not positive, does not divide the side, or the
        width is not divisible by 4 (the positional table) or by the heads.
    """
    side = patch_grid_side(sample_size)
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    if side % patch_size != 0:
        raise ValueError(
            f"sample_size {side} must be divisible by patch_size "
            f"{patch_size} — a partial patch has no token to live in"
        )
    if hidden_size % 4 != 0:
        raise ValueError(
            f"hidden_size must be divisible by 4 — the positional table "
            f"splits it into sine and cosine over each of two axes, got "
            f"{hidden_size}"
        )
    if hidden_size % num_heads != 0:
        raise ValueError(
            f"hidden_size must be divisible by num_heads, got "
            f"{hidden_size} and {num_heads}"
        )


# ── fields that live in the sample's own space ───────────────────────────────


def check_sample_space_channels(
    in_channels: int, out_channels: int, quantity: str
) -> None:
    """Refuse an output width other than the input's, for an in-place field.

    A family whose network predicts something the objective or the sampler
    adds to, or compares with, the input — a velocity, a displacement, a
    bijection's image — needs the two widths equal.  A mismatch would
    otherwise pass construction and fail as a broadcast error deep in the
    forward pass or the sampler.

    Parameters
    ----------
    in_channels : int
        The config's ``in_channels``.
    out_channels : int
        The config's ``out_channels``.
    quantity : str
        What the network predicts, for the message — ``"the velocity"``.

    Raises
    ------
    ValueError
        If the widths differ.
    """
    if in_channels != out_channels:
        raise ValueError(
            f"{quantity} lives in the sample's own space, so in_channels must "
            f"equal out_channels; got {in_channels} and {out_channels}"
        )
