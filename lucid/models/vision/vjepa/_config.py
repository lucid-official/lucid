"""V-JEPA configuration — Bardes et al., 2024.

I-JEPA's question asked of video.  A clip is cut into tubelets — two
frames deep, sixteen pixels square — most of them are hidden, and a
predictor must say what an averaged copy of the encoder would report
about the hidden ones.  Prediction happens in representation space, so
nothing has to account for the texture of a wall that moved.

Two things differ from the image model beyond the extra axis.  The masks
come in two collections at once, one made of eight small blocks and one
of two large ones, each running through the whole clip in time; and the
discrepancy is an :math:`\\ell_1` regression, which the paper says it
found more stable.  Here paper and released code agree, unlike I-JEPA's.
"""

from dataclasses import dataclass
from typing import ClassVar, Literal

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta

#: Which discrepancy the predictor is trained under.  The paper and the
#: released code both use L1; the others are here to be measured against.
VJEPAObjective = Literal["l1", "l2", "smooth_l1"]


@model_family_meta(
    canonical_name="V-JEPA",
    citation=(
        "Bardes, Adrien, et al. "
        '"Revisiting Feature Prediction for Learning Visual '
        'Representations from Video." '
        "arXiv:2404.08471, 2024."
    ),
    theory=r"""
    V-JEPA learns from video by predicting *features* of what it cannot
    see.  A clip of :math:`T` frames becomes :math:`T/2 \times H/16
    \times W/16` tubelet tokens; a mask hides most of them; the encoder
    :math:`f_\theta` reads what is left, and a predictor :math:`g_\phi`
    must produce the output of an exponentially averaged encoder
    :math:`f_{\bar\theta}` at the hidden positions:

    .. math::

        \mathcal{L} = \frac{1}{M} \sum_{k=1}^{M}
        \bigl\| g_\phi\bigl(f_\theta(x_{\text{context}}),\, p_k\bigr)
        - \mathrm{sg}\bigl[f_{\bar\theta}(x)\bigr]_k \bigr\|_1 .

    **Two mask collections, not one.**  Eight blocks covering 15% of the
    frame each, and two covering 70%, both running the full depth of the
    clip in time.  Their union hides about 90% of the tokens.  A mask
    that leaves a tube visible through time makes the task trivial ---
    the answer is in the next frame --- so both collections extend
    through it, and the model is asked the two questions at once, with a
    learned mask token of its own for each.

    **The target encoder sees everything; the context encoder does not.**
    Masking here removes tokens rather than attending around them, so the
    context encoder's sequence is genuinely shorter, while the averaged
    encoder reads the whole clip and its output is normalised before any
    block is taken from it.  No gradient flows into it at all.

    **Prediction in feature space is what makes the video tractable.**
    A generative video objective must settle every detail that differs
    between plausible continuations; this one may drop whatever the
    averaged encoder has already learned to drop.  Frozen, with an
    attentive probe on top, the result reaches 82.0% on Kinetics-400 and
    71.4% on Something-Something-v2 --- the latter being the benchmark
    that punishes models which only recognise appearance.
    """,
)
@dataclass(frozen=True)
class VJEPAConfig(ModelConfig):
    r"""Frozen configuration for the V-JEPA family.

    Defaults are the paper's ViT-L/16 at 224 pixels (Table 6, Table 8).

    Parameters
    ----------
    image_size : int, default=224
        Side of each frame.  The largest variant uses 384.
    patch_size : int, default=16
        Spatial side of a tubelet.
    tubelet_size : int, default=2
        Frames a tubelet spans.  Sixteen frames become eight temporal
        positions.
    num_frames : int, default=16
        Frames in a clip.
    sampling_rate : int, default=4
        Stride, in original frames, between the frames of a clip.  Carried
        because it defines what a clip is; nothing in the model reads it.
    in_channels : int, default=3
        Channels per frame.
    num_classes : int, default=400
        Classes the probe predicts.  Kinetics-400's count, since that is
        the benchmark the paper leads with.
    dim, depth, num_heads : int, default=1024, 24, 16
        The encoder, a ViT of the named size.
    mlp_ratio : float, default=4.0
        Feed-forward width as a multiple of ``dim``.
    layer_norm_eps : float, default=1e-6
        Epsilon of every LayerNorm.
    uniform_power : bool, default=True
        How the position table splits its width between time, row and
        column.  ``True`` gives each axis an equal share, which is what
        every released configuration sets; the code's own default is
        ``False``, which would give time half the width and the spatial
        axes a quarter each.  Getting this wrong loads released weights
        onto positions they were not trained with.
    predictor_dim : int, default=384
        Predictor width, for every variant.
    predictor_depth : int, default=12
        Predictor blocks, for every variant.
    predictor_heads : int or None, default=None
        Heads inside the predictor.  ``None`` takes the encoder's count,
        as the released code does — 16 heads across 384 channels, so 24
        per head.
    short_range_blocks : int, default=8
        Blocks in the first mask collection.
    short_range_scale : tuple of float, default=(0.15, 0.15)
        Fraction of a frame each of those blocks covers.  The paper fixes
        it rather than sampling a range.
    long_range_blocks : int, default=2
        Blocks in the second collection.
    long_range_scale : tuple of float, default=(0.7, 0.7)
        Fraction of a frame each of those covers.
    aspect_ratio : tuple of float, default=(0.75, 1.5)
        Aspect-ratio range, shared by both collections.
    ema : tuple of float, default=(0.998, 1.0)
        Momentum of the target encoder at the first and last step of the
        *schedule*.
    ema_schedule_scale : float, default=1.25
        How much longer the momentum schedule is than training.  The
        released runs stretch it by a quarter and stop early, so the
        momentum never actually reaches 1.
    objective : {"l1", "l2", "smooth_l1"}, default="l1"
        How predicted and target features are compared.  L1 is what both
        the paper and the released code use — Section 3.1 says it was
        found more stable than I-JEPA's regression.
    smooth_l1_beta : float, default=1.0
        Where smooth L1 turns from quadratic to linear, when chosen.

    Notes
    -----
    Reference: Bardes, Adrien, et al., *"Revisiting Feature Prediction for
    Learning Visual Representations from Video"*, arXiv:2404.08471, 2024
    — Table 6 for the variants and their results, Table 8 for the
    architecture and the schedules, Section 3.2 for the masking.

    Examples
    --------
    >>> from lucid.models.vision.vjepa import VJEPAConfig
    >>> config = VJEPAConfig()
    >>> config.token_grid, config.num_tokens
    ((8, 14, 14), 1568)
    >>> config.short_range_blocks, config.long_range_blocks
    (8, 2)

    Four times the pixels is four times the tokens:

    >>> VJEPAConfig(image_size=384).num_tokens
    4608
    """

    model_type: ClassVar[str] = "vjepa"

    image_size: int = 224
    patch_size: int = 16
    tubelet_size: int = 2
    num_frames: int = 16
    sampling_rate: int = 4
    in_channels: int = 3
    num_classes: int = 400

    dim: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    layer_norm_eps: float = 1e-6
    uniform_power: bool = True

    predictor_dim: int = 384
    predictor_depth: int = 12
    predictor_heads: int | None = None

    short_range_blocks: int = 8
    short_range_scale: tuple[float, float] = (0.15, 0.15)
    long_range_blocks: int = 2
    long_range_scale: tuple[float, float] = (0.7, 0.7)
    aspect_ratio: tuple[float, float] = (0.75, 1.5)

    ema: tuple[float, float] = (0.998, 1.0)
    ema_schedule_scale: float = 1.25
    objective: VJEPAObjective = "l1"
    smooth_l1_beta: float = 1.0

    @property
    def token_grid(self) -> tuple[int, int, int]:
        """Tokens along time, row and column."""
        side = self.image_size // self.patch_size
        return (self.num_frames // self.tubelet_size, side, side)

    @property
    def num_tokens(self) -> int:
        """Tokens in a clip.  There is no class token to add to it."""
        duration, rows, cols = self.token_grid
        return duration * rows * cols

    @property
    def resolved_predictor_heads(self) -> int:
        """Heads the predictor runs with, taking the encoder's when unset."""
        return self.num_heads if self.predictor_heads is None else self.predictor_heads

    def __post_init__(self) -> None:
        for name in ("short_range_scale", "long_range_scale", "aspect_ratio", "ema"):
            value = getattr(self, name)
            object.__setattr__(self, name, (float(value[0]), float(value[1])))

        positive = [
            "image_size",
            "patch_size",
            "tubelet_size",
            "num_frames",
            "sampling_rate",
            "in_channels",
            "num_classes",
            "dim",
            "depth",
            "num_heads",
            "predictor_dim",
            "predictor_depth",
            "short_range_blocks",
            "long_range_blocks",
        ]
        for name in positive:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")

        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size {self.image_size} is not a multiple of patch_size "
                f"{self.patch_size}; a patch grid would not tile the frame"
            )
        if self.num_frames % self.tubelet_size != 0:
            raise ValueError(
                f"num_frames {self.num_frames} is not a multiple of tubelet_size "
                f"{self.tubelet_size}; a tubelet would straddle the end of the clip"
            )
        if self.dim % self.num_heads != 0:
            raise ValueError(f"dim {self.dim} is not divisible by num_heads {self.num_heads}")
        heads = self.resolved_predictor_heads
        if heads < 1 or self.predictor_dim % heads != 0:
            raise ValueError(
                f"predictor_dim {self.predictor_dim} is not divisible by its {heads} heads"
            )
        if self.mlp_ratio <= 0.0:
            raise ValueError(f"mlp_ratio must be positive, got {self.mlp_ratio}")

        for name in ("short_range_scale", "long_range_scale"):
            low, high = getattr(self, name)
            if not 0.0 < low <= high <= 1.0:
                raise ValueError(
                    f"{name} is a fraction of a frame and must satisfy "
                    f"0 < low <= high <= 1, got ({low}, {high})"
                )
        low, high = self.aspect_ratio
        if not 0.0 < low <= high:
            raise ValueError(f"aspect_ratio must satisfy 0 < low <= high, got ({low}, {high})")
        start, end = self.ema
        if not 0.0 <= start <= end <= 1.0:
            raise ValueError(
                f"the target encoder's momentum must rise through [0, 1], got ({start}, {end})"
            )
        if self.ema_schedule_scale <= 0.0:
            raise ValueError(f"ema_schedule_scale must be positive, got {self.ema_schedule_scale}")
        if self.smooth_l1_beta <= 0.0:
            raise ValueError(f"smooth_l1_beta must be positive, got {self.smooth_l1_beta}")
