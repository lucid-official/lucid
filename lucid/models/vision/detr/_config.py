"""DETR configuration (Carion et al., 2020)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta
from lucid.models._utils._common import (
    check_divisible,
    check_non_negative,
    check_positive,
    check_residual_layers,
)


@model_family_meta(
    canonical_name="DETR",
    citation=(
        'Carion, Nicolas, et al. "End-to-End Object Detection with '
        'Transformers." Proceedings of the European Conference on '
        "Computer Vision, 2020, pp. 213–229."
    ),
    theory=r"""
    DETR (DEtection TRansformer) reframes object detection as a
    **direct set prediction** problem.  A CNN backbone produces a
    feature map :math:`f \in \mathbb{R}^{C \times H \times W}`, which
    is flattened, projected to :math:`d_{\mathrm{model}}`, augmented
    with 2-D sinusoidal positional encodings, and fed into a standard
    Transformer encoder–decoder.  The decoder additionally takes
    :math:`N` learnable **object queries**; each query produces one
    detection through FFN heads (class logits + normalised box
    :math:`(c_x, c_y, w, h)` with a sigmoid output).

    Training enforces a one-to-one assignment between predictions
    and ground-truth objects via the **Hungarian algorithm**.  The
    optimal permutation :math:`\hat{\sigma}` over :math:`N` slots
    minimises

    .. math::

        \hat{\sigma} = \arg\min_{\sigma \in S_N}
            \sum_{i=1}^{N} \mathcal{L}_{\mathrm{match}}\!
                \bigl(y_i, \hat{y}_{\sigma(i)}\bigr),

    where the per-pair cost combines class probability and a
    box-similarity term (an :math:`\ell_1` distance plus generalised
    IoU).  Given the matched assignment, the final training loss is

    .. math::

        \mathcal{L}_{\mathrm{Hung}}(y, \hat{y}) =
            \sum_{i=1}^{N} \bigl[
                -\log \hat{p}_{\hat{\sigma}(i)}(c_i)
                + \mathbb{1}_{\{c_i \neq \varnothing\}}
                  \mathcal{L}_{\mathrm{box}}(b_i, \hat{b}_{\hat{\sigma}(i)})
            \bigr].

    Because each query is matched to **at most one** ground-truth
    object, DETR needs neither anchor priors, nor a separate proposal
    stage, nor non-maximum suppression at inference — duplicates are
    suppressed by the bipartite matching during training.  The result
    is the first fully end-to-end detector with a Transformer trunk;
    later variants (Deformable DETR, DAB-DETR, DINO, …) build on this
    set-prediction framing while accelerating convergence.
    """,
)
@dataclass(frozen=True)
class DETRConfig(ModelConfig):
    """Configuration for DETR (DEtection TRansformer).

    DETR (Carion et al., ECCV 2020) reformulates object detection as a direct
    set prediction problem.  A CNN backbone extracts image features, a
    Transformer encoder-decoder processes them with N learned object queries,
    and FFN heads predict class labels + normalised (cx, cy, w, h) boxes.
    Training uses the Hungarian algorithm to match predictions to ground-truth
    objects (no hand-crafted anchors, no NMS at inference).

    Architecture overview:
      Image → ResNet-50 (C5 feature map) → 1×1 projection (d_model)
        → flatten + 2-D sinusoidal positional encoding
      → Transformer encoder (N_enc layers, d_model, n_head)
      → Transformer decoder (N_dec layers, N_queries object queries)
      → FFN: cls head (num_classes + 1) + box head (4, sigmoid)

    Args:
        num_classes:     Number of foreground classes (background = class 0).
        in_channels:     Input image channels.

        -- Backbone --
        backbone_layers: ResNet-50 layer counts (default 3,4,6,3).

        -- Transformer --
        d_model:       Transformer embedding dimension.
        n_head:        Number of self-/cross-attention heads.
        num_encoder_layers: Encoder depth.
        aux_loss:          Apply the set loss after every decoder layer, not
                           only the last (paper: "auxiliary decoding losses").
        num_decoder_layers: Decoder depth.
        dim_feedforward:   FFN inner dimension in each transformer layer.
        dropout:           Dropout in transformer.
        num_queries:   Number of object queries (N).

        -- FFN heads --
        num_bbox_layers:   MLP depth for bounding-box head.
        bbox_hidden_dim:   Hidden width inside the bbox MLP.

        -- Inference --
        score_thresh:  Minimum predicted class probability (before NMS).

    Examples
    --------
    >>> from lucid.models.vision.detr import DETRConfig
    >>> cfg = DETRConfig()
    >>> cfg.num_classes, cfg.num_queries
    (80, 100)
    >>> cfg.model_type
    'detr'
    """

    model_type: ClassVar[str] = "detr"

    num_classes: int = 80
    in_channels: int = 3

    # Backbone
    backbone_layers: tuple[int, int, int, int] = (3, 4, 6, 3)  # ResNet-50

    # Transformer
    d_model: int = 256
    n_head: int = 8
    num_encoder_layers: int = 6
    # Auxiliary decoding losses.  On by default, matching the reference,
    # whose training entry point exposes ``--no_aux_loss`` as the opt-out.
    # Inference is unaffected: the heads still read the last decoder layer.
    aux_loss: bool = True
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    num_queries: int = 100

    # FFN heads
    num_bbox_layers: int = 3
    bbox_hidden_dim: int = 256

    # Inference
    score_thresh: float = 0.7

    def __post_init__(self) -> None:
        object.__setattr__(self, "backbone_layers", tuple(self.backbone_layers))
        owner = type(self).__name__
        check_positive(owner, num_classes=self.num_classes, in_channels=self.in_channels)
        # A box MLP of ``n`` layers is ``n`` linears; below one it builds one.
        check_positive(
            owner,
            num_queries=self.num_queries,
            num_bbox_layers=self.num_bbox_layers,
            d_model=self.d_model,
            n_head=self.n_head,
            dim_feedforward=self.dim_feedforward,
            bbox_hidden_dim=self.bbox_hidden_dim,
        )
        check_divisible(owner, "d_model", self.d_model, "n_head", self.n_head)
        check_non_negative(
            owner,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
        )
        check_residual_layers(owner, backbone_layers=self.backbone_layers)
