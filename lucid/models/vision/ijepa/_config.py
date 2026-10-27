"""I-JEPA configuration — Assran et al., CVPR 2023.

Self-supervised pretraining that predicts *representations* rather than
pixels.  A context block of an image is encoded, and a narrow predictor is
asked what the encoder would have said about four other blocks it never
saw.  The answer is compared against an exponential moving average of the
encoder itself, so the target is a moving description of the image rather
than the image.

Nothing here reconstructs anything, and that is the point: a pixel target
spends capacity on detail no downstream task asks for, while a
representation target is free to drop it.

Two of the values below come from the released code rather than the paper,
and one of those is a disagreement rather than a silence — the paper's loss
is :math:`\\ell_2`, the code's is smooth L1.  Each says so where it stands.
"""

from dataclasses import dataclass
from typing import ClassVar, Literal

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta

#: Which discrepancy measure the predictor is trained under.
IJEPAObjective = Literal["smooth_l1", "l2"]


@model_family_meta(
    canonical_name="I-JEPA",
    citation=(
        "Assran, Mahmoud, et al. "
        '"Self-Supervised Learning from Images with a Joint-Embedding '
        'Predictive Architecture." '
        "Proceedings of the IEEE/CVF Conference on Computer Vision and "
        "Pattern Recognition (CVPR), 2023."
    ),
    theory=r"""
    I-JEPA asks a question about an image and grades the answer in
    representation space.  One *context* block is encoded by
    :math:`f_\theta`; four *target* blocks are encoded by a second network
    :math:`f_{\bar\theta}`, and a predictor :math:`g_\phi` must produce the
    target encoder's output for each target block from the context alone,
    told only *where* each block is:

    .. math::

        \mathcal{L} = \frac{1}{M} \sum_{i=1}^{M} \sum_{j \in B_i}
        d\bigl(\hat s_{y_j},\, s_{y_j}\bigr), \qquad
        \hat s_{y} = g_\phi\bigl(f_\theta(x_{\text{context}}),\, p_y\bigr).

    **The target is a representation, not a pixel.**  A generative
    objective must account for every detail that distinguishes one plausible
    completion from another --- the exact texture of grass, the grain of a
    wall --- and spends capacity doing so.  Predicting :math:`f_{\bar\theta}`'s
    output lets the model drop whatever that encoder has already learned to
    drop.  The paper's ablation is unusually clean: the same architecture
    trained against pixels reaches 40.7% on 1% of ImageNet, and against
    representations 66.9%.

    **The target encoder is the model's own past.**  :math:`f_{\bar\theta}`
    is not trained; its weights are an exponential moving average of
    :math:`f_\theta`, with the momentum moving linearly from 0.996 to 1 over
    training.  There is no gradient path into it at all, which is what stops
    the pair from agreeing on a constant --- the target moves only as fast
    as the average lets it.

    **The masking carries the difficulty.**  Four target blocks covering
    15--20% of the image each, at aspect ratios from 3:4 to 3:2, with a
    context block of 85--100% from which every target region is then
    removed.  Predicting a large connected region from a large connected
    region is what forces semantics; the paper's ablation against
    rasterised or small random masks (54.2% versus 15.5% and 17.6%) is a
    statement about the task, not the architecture.

    **No** :math:`[\text{cls}]` **token.**  Evaluation average-pools the
    target encoder's patch outputs.
    """,
)
@dataclass(frozen=True)
class IJEPAConfig(ModelConfig):
    r"""Frozen configuration for the I-JEPA family.

    Defaults are the paper's ViT-B/16 (Table 1, Appendix A.1).

    Parameters
    ----------
    image_size : int, default=224
        Input resolution.  The 448-pixel variant is the one exception.
    patch_size : int, default=16
        Patch side; ``image_size`` must be a multiple of it.
    in_channels : int, default=3
        Image channels.
    num_classes : int, default=1000
        Classes the linear probe predicts.  Pretraining ignores it.
    dim, depth, num_heads : int, default=768, 12, 12
        The encoder, which is a ViT of the named size.
    mlp_ratio : float, default=4.0
        Feed-forward width as a multiple of ``dim``.  ViT-g/16 is the one
        variant where the paper uses a different ratio.
    layer_norm_eps : float, default=1e-6
        Epsilon of every LayerNorm.
    predictor_dim : int, default=384
        Predictor width, for every variant (Appendix A.1; Table 14 measures
        384 against 1024 and prefers it).  The predictor is deliberately
        narrower than the encoder.
    predictor_depth : int, default=6
        Predictor blocks: 6 for ViT-B/16, 12 for the larger encoders.
    predictor_heads : int or None, default=None
        Heads inside the predictor.  ``None`` takes the encoder's count,
        which is what the released code does — so a 16-head predictor 384
        wide has 24-dimensional heads.
    num_target_blocks : int, default=4
        Target blocks predicted per image (Section 3).  They may overlap
        each other.
    target_scale : tuple of float, default=(0.15, 0.2)
        Fraction of the image each target block covers.
    target_aspect : tuple of float, default=(0.75, 1.5)
        Aspect-ratio range of a target block.
    context_scale : tuple of float, default=(0.85, 1.0)
        Fraction of the image the context block covers.  Its aspect ratio
        is 1 — *not stated* in the paper, fixed in the released code.
    min_keep : int, default=10
        Patches a sampled block must exceed, or it is resampled.  *Not
        stated*; the released configs set it.
    allow_overlap : bool, default=False
        Whether the context may keep patches that fall inside a target
        block.  The paper removes them; the released code makes it a flag
        and sets it False.
    ema : tuple of float, default=(0.996, 1.0)
        Momentum of the target encoder at the first and last training step,
        moving linearly between them.
    objective : {"smooth_l1", "l2"}, default="smooth_l1"
        How predicted and target representations are compared.  ⚠️ The
        paper states an :math:`\ell_2` loss; the released code — which
        produced the released checkpoints — uses smooth L1.  The default
        follows the code, and ``"l2"`` is the paper's.
    smooth_l1_beta : float, default=1.0
        Where smooth L1 turns from quadratic to linear.

    Notes
    -----
    Reference: Assran, Mahmoud, et al., *"Self-Supervised Learning from
    Images with a Joint-Embedding Predictive Architecture"*, CVPR 2023
    (arXiv:2301.08243), Table 1 for the variants and Appendix A.1 for the
    masking and the momentum schedule.

    Examples
    --------
    >>> from lucid.models.vision.ijepa import IJEPAConfig
    >>> config = IJEPAConfig()
    >>> config.num_patches, config.grid_size
    (196, 14)
    >>> config.predictor_dim, config.num_target_blocks
    (384, 4)

    The predictor takes the encoder's head count when it is not given:

    >>> IJEPAConfig(num_heads=16, predictor_heads=None).resolved_predictor_heads
    16
    """

    model_type: ClassVar[str] = "ijepa"

    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    num_classes: int = 1000

    dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    layer_norm_eps: float = 1e-6

    predictor_dim: int = 384
    predictor_depth: int = 6
    predictor_heads: int | None = None

    num_target_blocks: int = 4
    target_scale: tuple[float, float] = (0.15, 0.2)
    target_aspect: tuple[float, float] = (0.75, 1.5)
    context_scale: tuple[float, float] = (0.85, 1.0)
    min_keep: int = 10
    allow_overlap: bool = False

    ema: tuple[float, float] = (0.996, 1.0)
    objective: IJEPAObjective = "smooth_l1"
    smooth_l1_beta: float = 1.0

    @property
    def grid_size(self) -> int:
        """Patches along one side of the image."""
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        """Patches in an image.  There is no class token to add to it."""
        return self.grid_size**2

    @property
    def resolved_predictor_heads(self) -> int:
        """Heads the predictor runs with, taking the encoder's when unset."""
        return self.num_heads if self.predictor_heads is None else self.predictor_heads

    def __post_init__(self) -> None:
        # JSON round-trips turn tuples into lists; a frozen dataclass has to
        # put them back or equality and hashing break.
        for name in ("target_scale", "target_aspect", "context_scale", "ema"):
            value = getattr(self, name)
            object.__setattr__(self, name, (float(value[0]), float(value[1])))

        for name in (
            "image_size",
            "patch_size",
            "dim",
            "depth",
            "num_heads",
            "predictor_dim",
            "predictor_depth",
            "num_target_blocks",
            "min_keep",
            "num_classes",
            "in_channels",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")

        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size {self.image_size} is not a multiple of patch_size "
                f"{self.patch_size}; a patch grid would not tile the image"
            )
        if self.dim % self.num_heads != 0:
            raise ValueError(f"dim {self.dim} is not divisible by num_heads {self.num_heads}")
        heads = self.resolved_predictor_heads
        if heads < 1 or self.predictor_dim % heads != 0:
            raise ValueError(
                f"predictor_dim {self.predictor_dim} is not divisible by its " f"{heads} heads"
            )
        if self.mlp_ratio <= 0.0:
            raise ValueError(f"mlp_ratio must be positive, got {self.mlp_ratio}")

        for name in ("target_scale", "context_scale"):
            low, high = getattr(self, name)
            if not 0.0 < low <= high <= 1.0:
                raise ValueError(
                    f"{name} is a fraction of the image and must satisfy "
                    f"0 < low <= high <= 1, got ({low}, {high})"
                )
        low, high = self.target_aspect
        if not 0.0 < low <= high:
            raise ValueError(f"target_aspect must satisfy 0 < low <= high, got ({low}, {high})")
        if self.min_keep > self.num_patches:
            raise ValueError(
                f"min_keep {self.min_keep} exceeds the {self.num_patches} patches "
                f"an image has, so no block could ever be kept"
            )
        start, end = self.ema
        if not 0.0 <= start <= end <= 1.0:
            raise ValueError(
                f"the target encoder's momentum must rise through [0, 1], got " f"({start}, {end})"
            )
        if self.smooth_l1_beta <= 0.0:
            raise ValueError(f"smooth_l1_beta must be positive, got {self.smooth_l1_beta}")
