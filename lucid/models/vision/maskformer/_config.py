"""MaskFormer configuration (Cheng et al., NeurIPS 2021)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta
from lucid.models._utils._common import check_positive, check_residual_layers


@model_family_meta(
    canonical_name="MaskFormer",
    citation=(
        'Cheng, Bowen, et al. "Per-Pixel Classification is Not All You '
        'Need for Semantic Segmentation." Advances in Neural Information '
        "Processing Systems, 2021."
    ),
    theory=r"""
    MaskFormer recasts **semantic segmentation** as a
    *mask-classification* problem rather than the standard
    per-pixel-classification one.  Instead of predicting a
    :math:`K`-way softmax at every pixel, the model emits :math:`N`
    queries; each query produces a class label and a binary mask,
    and the final segmentation is the soft sum

    .. math::

        \mathrm{seg}_{k}[h, w] =
            \sum_{n=1}^{N} \mathrm{softmax}(\hat{c}_n)_k \,
            \sigma(\hat{m}_n[h, w]),

    where :math:`\hat{c}_n \in \mathbb{R}^{K+1}` is the per-query
    class logit and :math:`\hat{m}_n` is its binary mask logit.  The
    argmax over :math:`k` yields the predicted label map.

    The architecture has three modules:

    - **Pixel decoder** — an FPN-like up-sampler that produces a
      per-pixel embedding :math:`\mathcal{E}_{\mathrm{pixel}} \in
      \mathbb{R}^{C_\epsilon \times H/4 \times W/4}`.
    - **Transformer decoder** — :math:`N` learnable queries attend
      to a flattened backbone feature, producing per-query
      embeddings :math:`\mathcal{E}_{\mathrm{query}} \in
      \mathbb{R}^{N \times C_\epsilon}`.
    - **Heads** — a linear classifier on each query and a
      dot-product :math:`\mathcal{E}_{\mathrm{query}} \cdot
      \mathcal{E}_{\mathrm{pixel}}` to produce per-query masks.

    Training uses Hungarian matching (as in DETR) between predicted
    and ground-truth segments and combines cross-entropy on the
    class logits with focal/BCE + Dice losses on the mask logits.

    The key insight — that mask classification subsumes per-pixel
    classification — also unifies *semantic* and *instance*
    segmentation under one model, foreshadowing the more general
    Mask2Former.
    """,
)
@dataclass(frozen=True)
class MaskFormerConfig(ModelConfig):
    """Configuration for MaskFormer.

    MaskFormer (Cheng et al., NeurIPS 2021) reformulates semantic segmentation
    as mask classification: N learnable queries each predict a class label and
    a binary mask.  Predictions are matched to ground-truth segments via
    Hungarian matching during training.

    Architecture overview:
      Image → ResNet backbone → [C2, C3, C4, C5]
        → FPN Pixel Decoder → per-pixel embeddings (B, d_model, H/4, W/4)
        → Transformer Decoder (N queries attend to pixel embeddings)
        → Class head: Linear(d_model, num_classes+1) per query
        → Mask head: query-dot-pixel → binary mask per query

    Inference:
      seg_logits[b, k, h, w] = sum_n softmax(class_logits)[b,n,k]
                                      * sigmoid(mask_logits)[b,n,h,w]
      Upsampled to input resolution and argmaxed for final prediction.

    Args:
        num_classes:       Number of semantic classes (background = class 0).
        in_channels:       Input image channels.
        backbone_layers:   ResNet layer counts (default ResNet-50: 3,4,6,3).
        d_model:           Transformer embedding dimension.
        n_head:            Number of attention heads.
        num_encoder_layers: Depth of the transformer encoder applied to the
                           flattened C5 memory before the decoder reads it.
                           ``0`` (the reference semantic setting) skips it.
        num_decoder_layers: Query transformer decoder depth.
        dim_feedforward:   FFN inner dimension in each transformer layer.
        dropout:           Dropout probability.
        num_queries:       Number of learnable object queries N.
        fpn_out_channels:  FPN lateral / output channel width.

    Examples
    --------
    >>> from lucid.models.vision.maskformer import MaskFormerConfig
    >>> cfg = MaskFormerConfig()
    >>> cfg.num_classes, cfg.num_queries
    (150, 100)
    >>> cfg.model_type
    'maskformer'
    """

    model_type: ClassVar[str] = "maskformer"

    num_classes: int = 150
    in_channels: int = 3

    # Backbone
    # ResNet-50.  The stage counts are the only backbone knob: the stages
    # are built from bottleneck blocks unconditionally, which is what every
    # published MaskFormer ResNet backbone uses.
    backbone_layers: tuple[int, int, int, int] = (3, 4, 6, 3)

    # Transformer
    d_model: int = 256
    n_head: int = 8
    # The semantic-segmentation config these checkpoints come from sets
    # ENC_LAYERS: 0 -- the C5 feature goes straight into the decoder.  The
    # "+6 Enc" row is the COCO-panoptic comparison against DETR, not the
    # semantic result.  Values > 0 now build a real encoder stack.
    num_encoder_layers: int = 0
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    num_queries: int = 100

    # FPN pixel decoder
    fpn_out_channels: int = 256

    def __post_init__(self) -> None:
        object.__setattr__(self, "backbone_layers", tuple(self.backbone_layers))
        owner = type(self).__name__
        check_positive(owner, num_classes=self.num_classes, in_channels=self.in_channels)
        check_positive(owner, num_queries=self.num_queries, fpn_out_channels=self.fpn_out_channels)
        check_residual_layers(owner, backbone_layers=self.backbone_layers)
