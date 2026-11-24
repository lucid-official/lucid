"""Mask2Former configuration (Cheng et al., CVPR 2022)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta
from lucid.models._utils._common import check_positive


@model_family_meta(
    canonical_name="Mask2Former",
    citation=(
        'Cheng, Bowen, et al. "Masked-attention Mask Transformer for '
        'Universal Image Segmentation." Proceedings of the IEEE/CVF '
        "Conference on Computer Vision and Pattern Recognition, 2022, "
        "pp. 1290–1299."
    ),
    theory=r"""
    Mask2Former generalises MaskFormer into a single architecture
    that wins on **semantic**, **instance**, and **panoptic**
    segmentation simultaneously.  It keeps the mask-classification
    framing — :math:`N` object queries each predict a class and a
    binary mask — and improves it with three orthogonal changes.

    **Masked cross-attention.**  In every decoder layer the queries
    attend only to feature locations that the *previous* layer
    already considered foreground for that query:

    .. math::

        \mathrm{attn}_{ij} = \mathrm{softmax}_j\!\bigl(
            Q_i K_j^{\top} / \sqrt{d} + \mathcal{M}_{ij}^{\,\ell-1}
        \bigr),

    where :math:`\mathcal{M}^{\ell-1}_{ij} = 0` if the predicted
    binary mask of query :math:`i` at layer :math:`\ell - 1`
    activates pixel :math:`j` and :math:`-\infty` otherwise.  This
    confines query updates to plausible mask regions and accelerates
    convergence.

    **Multi-scale features.**  The decoder layers cycle through three
    pyramid levels :math:`(P_3, P_4, P_5)` of an enhanced FPN-style
    *pixel decoder*, so high-resolution semantic detail and broad
    contextual features are both available to every query.

    **Improved pixel decoder.**  A multi-scale deformable-attention
    encoder fuses the backbone feature maps before producing the
    per-pixel embeddings against which the query embeddings are dotted
    to form binary masks.

    Training keeps MaskFormer's Hungarian-matched class CE + mask
    BCE/Dice objective and its auxiliary losses at intermediate decoder
    layers, but computes the mask term on *sampled points* rather than
    densely: Section 3.2.2 draws :math:`K = 12{,}544` (:math:`112 \times
    112`) points by uniform plus importance sampling, in both the
    matching and the final loss, cutting training memory roughly 3x.
    That point-sampled loss is one of the paper's three enumerated
    contributions, alongside masked attention and the multi-scale
    strategy described above.  Mask2Former sets a new SOTA on each of
    ADE20K, Cityscapes, and COCO with a *single* model, eliminating the
    need for task-specific architectures.
    """,
)
@dataclass(frozen=True)
class Mask2FormerConfig(ModelConfig):
    """Configuration for Mask2Former (Cheng et al., CVPR 2022).

    The field set mirrors the reference framework's ``Mask2FormerConfig``
    so the pretrained-weight converter is a near-identity key map.  The
    pipeline is:

      Image → Swin backbone → [stage1..4] feature maps
        → MSDeformAttn pixel decoder → 3 multi-scale memory levels
                                     + 1/4-scale mask features
        → 9-layer masked-attention transformer decoder (cycling levels)
        → class head (Linear → K+1) + mask head (MLP → dot mask features)

    Args:
        num_classes:           Number of semantic classes (foreground; the
                               class head emits ``num_classes + 1``).
        in_channels:           Input image channels.
        swin_embed_dim:        Swin patch-embedding dimension.
        swin_depths:           Swin per-stage block counts.
        swin_num_heads:        Swin per-stage head counts.
        swin_window_size:      Swin attention window size.
        swin_mlp_ratio:        Swin MLP expansion ratio.
        d_model:               Transformer / pixel-decoder feature dim.
        mask_feature_size:     Per-pixel mask-feature channel width.
        n_head:                Number of attention heads.
        num_encoder_layers:    Deformable pixel-decoder encoder depth.
        encoder_feedforward_dim: Pixel-decoder FFN inner dim.
        num_decoder_layers:    Transformer decoder depth (the decoder uses
                               ``num_decoder_layers - 1`` masked layers; the
                               extra slot is the pre-layer mask prediction).
        dim_feedforward:       Transformer-decoder FFN inner dim.
        dropout:               Dropout probability (0 at inference).
        num_queries:           Number of learnable object queries N.
        num_feature_levels:    Number of multi-scale memory levels (3).
        feature_strides:       Backbone output strides.
        common_stride:         Finest pixel-decoder stride.

        -- Training objective (§3.2.2) --
        class_weight:      Weight of the classification term, and of the
                           class cost inside the Hungarian matcher.
        mask_weight:       Weight of the point-sampled mask BCE.
        dice_weight:       Weight of the point-sampled dice term.
        no_object_weight:  Class weight of the "no object" slot (0.1).
        train_num_points:  K points per mask (12,544 = 112 x 112).
        oversample_ratio:  Candidate multiplier for the importance sampler.
        importance_sample_ratio: Share of K taken from the most uncertain
                           candidates; the rest are uniform.
        deep_supervision:  Apply the criterion to every decoder layer, not
                           just the last.

    .. note::

       **Training.**  Pass ``targets`` to :meth:`forward` for §3.2.2's
       objective: Hungarian matching on a class + mask-BCE + dice cost,
       the same three terms as the loss, all mask terms evaluated on
       ``train_num_points`` importance-sampled points, and — under
       ``deep_supervision`` — the whole criterion repeated on every
       decoder layer.

    Examples
    --------
    >>> from lucid.models.vision.mask2former import Mask2FormerConfig
    >>> cfg = Mask2FormerConfig()
    >>> cfg.num_classes, cfg.num_queries
    (150, 100)
    >>> cfg.model_type
    'mask2former'
    """

    model_type: ClassVar[str] = "mask2former"

    num_classes: int = 150
    in_channels: int = 3

    # Swin backbone
    swin_embed_dim: int = 96
    swin_depths: tuple[int, int, int, int] = (2, 2, 6, 2)
    swin_num_heads: tuple[int, int, int, int] = (3, 6, 12, 24)
    swin_window_size: int = 7
    swin_mlp_ratio: float = 4.0

    # Transformer / pixel decoder
    d_model: int = 256
    mask_feature_size: int = 256
    n_head: int = 8
    num_encoder_layers: int = 6
    encoder_feedforward_dim: int = 1024
    num_decoder_layers: int = 10
    dim_feedforward: int = 2048
    dropout: float = 0.0
    num_queries: int = 100

    # Training objective (§3.2.2 + the released SetCriterion / HungarianMatcher)
    class_weight: float = 2.0
    mask_weight: float = 5.0
    dice_weight: float = 5.0
    # Down-weights the "no object" class so the N - M unmatched queries do
    # not drown the M matched ones in the classification term.
    no_object_weight: float = 0.1
    # §3.2.2 evaluates the mask terms on K sampled points rather than
    # densely, cutting training memory roughly 3x.  12544 = 112 x 112.
    train_num_points: int = 12_544
    oversample_ratio: float = 3.0
    importance_sample_ratio: float = 0.75
    # Deep supervision: the same criterion on every decoder layer's output.
    deep_supervision: bool = True

    # Multi-scale memory levels
    num_feature_levels: int = 3
    feature_strides: tuple[int, int, int, int] = (4, 8, 16, 32)
    common_stride: int = 4

    def __post_init__(self) -> None:
        object.__setattr__(self, "swin_depths", tuple(self.swin_depths))
        object.__setattr__(self, "swin_num_heads", tuple(self.swin_num_heads))
        object.__setattr__(self, "feature_strides", tuple(self.feature_strides))
        owner = type(self).__name__
        check_positive(owner, num_classes=self.num_classes, in_channels=self.in_channels)
