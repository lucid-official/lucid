"""DETR — DEtection TRansformer (Carion et al., ECCV 2020).

Paper: "End-to-End Object Detection with Transformers"

Key innovation
--------------
DETR casts object detection as a **direct set prediction** problem:

1. A CNN backbone extracts a spatial feature map.
2. A Transformer encoder attends globally over the (flattened) feature map.
3. A fixed set of N learned *object queries* decode the encoder memory via
   cross-attention in the Transformer decoder.
4. Independent FFN heads map each query to a class distribution and a
   bounding-box tuple (cx, cy, w, h) normalised to [0, 1].
5. At training time, a **Hungarian matching** finds the optimal bijection
   between predictions and ground-truth objects and computes the set loss.
6. At inference, the N predictions are thresholded by class score — no NMS,
   no anchors, no proposal stages.

Architecture
------------
  Image (B, C, H, W)
    ↓  ResNet (conv1 → pool → layer1–layer4) — C5 (2048ch, stride 32)
    ↓  1×1 Conv: 2048 → d_model  (default 256)
    ↓  Flatten: (B, d_model, H', W') → (H'·W', B, d_model)
    ↓  + 2D sinusoidal positional encoding re-injected at every layer
  Encoder: N_enc × (self-attention + FFN)
  Decoder: N_dec × (self-attention on queries + cross-attention to memory + FFN)
    ↓  (N_queries, B, d_model)
  Class head: Linear → (B, N, num_classes + 1)   softmax at inference
  Box head:   3-layer MLP → (B, N, 4)   sigmoid to enforce [0,1]

Faithfulness notes
------------------
* ResNet C5 feature map (stride 32) with **frozen** batch-norm (eval-only
  affine + running-stat math, no ``num_batches_tracked``).
* d_model=256, n_head=8, 6 enc / 6 dec layers, dim_ffn=2048.
* N=100 object queries.
* Sinusoidal 2-D positional encoding identical to the reference
  ``PositionEmbeddingSine`` (num_pos_feats=d_model/2, temperature=10000,
  normalize=True, scale=2π) — verified to ~1e-6.
* The transformer (``_DETRTransformer``) mirrors the reference layer
  layout verbatim: ``encoder.layers.{N}`` (no final encoder norm),
  ``decoder.layers.{N}`` + a final ``decoder.norm`` LayerNorm.  Positional
  encodings are re-injected at every layer (added to Q/K of encoder
  self-attention; query positions added to Q/K of decoder self-attention
  and to Q of decoder cross-attention, spatial positions added to the
  memory K), exactly as the reference forward.  Post-norm
  (normalize_before=False), ReLU activation.
* Hungarian cost matrix: L_cls + 5·L_L1 + 2·L_GIoU (matched queries).
* Background class weight 0.1 for unmatched queries.
"""

from typing import ClassVar, cast, final, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._tasks import ObjectDetectionModel
from lucid.models._output import ObjectDetectionOutput
from lucid.models._utils._detection import (
    _FrozenBatchNorm2d,
    _make_resnet_layer,
    box_cxcywh_to_xyxy,
    box_xyxy_to_cxcywh,
    generalized_box_iou,
    solve_assignment,
)
from lucid.models._utils._detection_results import (
    check_per_image,
    detection_result,
    empty_detection_result,
)
from lucid.models.vision._common._position import SinePositionEmbedding
from lucid.models.vision.detr._config import DETRConfig

# The reference DETR backbone freezes its BatchNorm with eps = 1e-5, where
# the R-CNN trunks sharing these blocks use 0.
_BN_EPS = 1e-5

# ---------------------------------------------------------------------------
# ResNet backbone (C5 only — frozen BN, reference key layout)
# ---------------------------------------------------------------------------


@final
class _ResNetC5(nn.Module):
    """ResNet backbone, returns C5 feature map (stride 32, 2048ch).

    Submodule names (``conv1`` / ``bn1`` / ``layer1``…``layer4``) mirror
    the reference ResNet body so the converter map for the backbone is a
    pure prefix strip (``backbone.0.body.<rest>`` → ``backbone.<rest>``).
    """

    def __init__(self, in_channels: int, layers: tuple[int, int, int, int]) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = _FrozenBatchNorm2d(64, eps=_BN_EPS)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1, c2 = _make_resnet_layer(64, 64, layers[0], 1, _BN_EPS)
        self.layer2, c3 = _make_resnet_layer(c2, 128, layers[1], 2, _BN_EPS)
        self.layer3, c4 = _make_resnet_layer(c3, 256, layers[2], 2, _BN_EPS)
        self.layer4, c5 = _make_resnet_layer(c4, 512, layers[3], 2, _BN_EPS)
        self.out_channels: int = c5

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)


# ---------------------------------------------------------------------------
# DETR transformer (reference layer layout + per-layer pos re-injection)
# ---------------------------------------------------------------------------


def _with_pos(tensor: Tensor, pos: Tensor | None) -> Tensor:
    """Add positional embedding to a tensor (identity when ``pos`` is None)."""
    return tensor if pos is None else tensor + pos


@final
class _DETREncoderLayer(nn.Module):
    """One post-norm encoder layer with pos re-injection on Q/K.

    Submodule names match the reference: ``self_attn`` (fused-``in_proj``
    :class:`nn.MultiheadAttention`), ``linear1`` / ``linear2``, ``norm1`` /
    ``norm2``.  Dropout is a no-op in eval mode.
    """

    def __init__(self, d_model: int, n_head: int, dim_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.linear2 = nn.Linear(dim_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    @override
    def forward(
        self,
        src: Tensor,
        pos: Tensor | None,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        q = _with_pos(src, pos)
        k = q
        src2 = self.self_attn(
            q, k, value=src, key_padding_mask=key_padding_mask, need_weights=False
        )[0]
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))
        return src


@final
class _DETRDecoderLayer(nn.Module):
    """One post-norm decoder layer.

    Submodule names match the reference: ``self_attn``, ``multihead_attn``
    (both fused-``in_proj`` :class:`nn.MultiheadAttention`), ``linear1`` /
    ``linear2``, ``norm1`` / ``norm2`` / ``norm3``.
    """

    def __init__(self, d_model: int, n_head: int, dim_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.linear2 = nn.Linear(dim_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    @override
    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        pos: Tensor | None,
        query_pos: Tensor | None,
        memory_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        q = _with_pos(tgt, query_pos)
        k = q
        tgt2 = self.self_attn(q, k, value=tgt, need_weights=False)[0]
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        tgt2 = self.multihead_attn(
            _with_pos(tgt, query_pos),
            _with_pos(memory, pos),
            value=memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )[0]
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(tgt2))
        return tgt


@final
class _DETREncoder(nn.Module):
    """Stack of encoder layers — NO final norm (reference removes it)."""

    def __init__(
        self, d_model: int, n_head: int, dim_ff: int, dropout: float, depth: int
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_DETREncoderLayer(d_model, n_head, dim_ff, dropout) for _ in range(depth)]
        )

    @override
    def forward(
        self,
        src: Tensor,
        pos: Tensor | None,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, pos=pos, key_padding_mask=key_padding_mask)
        return output


@final
class _DETRDecoder(nn.Module):
    """Stack of decoder layers + a final ``norm`` (LayerNorm).

    Inference needs only the last (norm-applied) activation, but training
    needs every layer: the paper's auxiliary decoding losses apply the
    prediction heads and the full set loss after each decoder layer, which
    is what makes the intermediate layers learn to predict rather than
    merely to pass features along.  ``return_intermediate`` selects between
    the two; the final element of the stack is identical to what the
    single-output path returns, so inference is unaffected either way.
    """

    def __init__(
        self, d_model: int, n_head: int, dim_ff: int, dropout: float, depth: int
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_DETRDecoderLayer(d_model, n_head, dim_ff, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(d_model)

    @override
    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        pos: Tensor | None,
        query_pos: Tensor | None,
        memory_key_padding_mask: Tensor | None = None,
        return_intermediate: bool = False,
    ) -> Tensor:
        output = tgt
        intermediate: list[Tensor] = []
        for layer in self.layers:
            output = layer(
                output,
                memory,
                pos=pos,
                query_pos=query_pos,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            if return_intermediate:
                intermediate.append(self.norm(output))

        normed = self.norm(output)
        if not return_intermediate:
            return normed
        # The reference replaces the last collected entry with the same
        # normalised output, so element -1 is exactly the single-output path.
        intermediate[-1] = normed
        return lucid.stack(intermediate, dim=0)  # (L, N, B, c)


@final
class _DETRTransformer(nn.Module):
    """Reference DETR transformer: ``encoder`` + ``decoder``.

    Forward takes the projected feature map ``(B, d, H, W)``, the spatial
    positional encoding ``pos`` ``(B, d, H, W)`` and the learned query
    embedding ``query_embed`` ``(N, d)``; it flattens, runs the encoder /
    decoder with per-layer positional re-injection, and returns the final
    decoder activation as ``(B, N, d)``.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.encoder = _DETREncoder(
            d_model, n_head, dim_feedforward, dropout, num_encoder_layers
        )
        self.decoder = _DETRDecoder(
            d_model, n_head, dim_feedforward, dropout, num_decoder_layers
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Xavier-uniform every multi-dimensional parameter.

        The reference transformer ends its constructor with exactly this
        sweep, which overrides the per-module defaults across all twelve
        encoder and decoder layers.  Without it the attention and
        feed-forward weights keep Lucid's ``kaiming_uniform(a=sqrt(5))``,
        a different distribution and gain from the one DETR's very long
        schedule was tuned against.
        """
        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    @override
    def forward(
        self,
        src: Tensor,
        pos_embed: Tensor,
        query_embed: Tensor,
        key_padding_mask: Tensor | None = None,
        return_intermediate: bool = False,
    ) -> Tensor:
        B = int(src.shape[0])
        c = int(src.shape[1])
        h = int(src.shape[2])
        w = int(src.shape[3])

        # (B, c, H, W) → (H*W, B, c)
        src_flat = src.reshape(B, c, h * w).permute(2, 0, 1)
        pos_flat = pos_embed.reshape(B, c, h * w).permute(2, 0, 1)
        # (N, c) → (N, B, c)
        query_pos = query_embed.unsqueeze(1).repeat(1, B, 1)
        # The reference's ``zeros_like(query_embed)``: the decoder's first
        # input takes the queries' dtype as well as their shape.
        tgt = lucid.zeros_like(query_pos)

        memory = self.encoder(src_flat, pos_flat, key_padding_mask=key_padding_mask)
        hs = self.decoder(
            tgt,
            memory,
            pos=pos_flat,
            query_pos=query_pos,
            memory_key_padding_mask=key_padding_mask,
            return_intermediate=return_intermediate,
        )  # (N, B, c) or (L, N, B, c)
        if return_intermediate:
            return hs.permute(0, 2, 1, 3)  # (L, B, N, c)
        return hs.permute(1, 0, 2)  # (B, N, c)


# ---------------------------------------------------------------------------
# MLP head (for bounding-box prediction)
# ---------------------------------------------------------------------------


class _MLP(nn.Module):
    """Simple N-layer MLP with ReLU activations (last layer has no activation)."""

    def __init__(
        self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int
    ) -> None:
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Hungarian matching (assignment)
# ---------------------------------------------------------------------------


def _hungarian_match(
    pred_logits: Tensor,  # (N, K+1)
    pred_boxes: Tensor,  # (N, 4) cxcywh normalised
    gt_labels: Tensor,  # (M,)
    gt_boxes: Tensor,  # (M, 4) cxcywh normalised
    cost_cls: float = 1.0,
    cost_l1: float = 5.0,
    cost_giou: float = 2.0,
) -> tuple[list[int], list[int]]:
    """Minimum-cost bipartite matching between N queries and M GT objects.

    Uses a simple O(N·M) cost matrix solved by the Hungarian algorithm
    (O(N²M) greedy augmenting-path variant — N is small, typically 100).

    Returns:
        (pred_indices, gt_indices) — matched pairs, ``pred_indices[i]``
        paired with ``gt_indices[i]``.  The cost matrix is built with
        ground truths as rows, so the pairs come back in ascending *GT*
        order; ``pred_indices`` is not sorted.  Callers zip the two, so
        the order carries no meaning beyond the pairing.
    """
    N = int(pred_logits.shape[0])
    M = int(gt_labels.shape[0])
    if M == 0:
        return [], []

    # Classification cost: negative softmax probability for the GT class
    scores = F.softmax(pred_logits, dim=-1)  # (N, K+1)

    # L1 cost: sum of absolute differences in cxcywh space
    # GIoU cost: negative GIoU between decoded boxes
    gt_xy = box_cxcywh_to_xyxy(gt_boxes)  # (M, 4)
    pred_xy = box_cxcywh_to_xyxy(pred_boxes)  # (N, 4)
    giou_mat = generalized_box_iou(pred_xy, gt_xy)  # (N, M)

    # The solver is a data-dependent host loop (the H3 CPU round-trip), so each
    # operand crosses to the host in one transfer.  Reading the matrix out one
    # ``.item()`` at a time cost N·M·10 device syncs per image on Metal; the
    # values are the same float32 entries either way.
    scores_l = cast(list[list[float]], scores.tolist())
    pred_l = cast(list[list[float]], pred_boxes.tolist())
    gt_l = cast(list[list[float]], gt_boxes.tolist())
    giou_l = cast(list[list[float]], giou_mat.tolist())
    labels_l = [int(v) for v in cast(list[int], gt_labels.tolist())]

    # Build M × N cost matrix (rows = GTs, cols = queries — M ≤ N for DETR)
    cost: list[list[float]] = []
    for m in range(M):
        gt_cls = labels_l[m]
        row: list[float] = []
        for n in range(N):
            c_cls = -scores_l[n][gt_cls]
            c_l1 = sum(abs(pred_l[n][d] - gt_l[m][d]) for d in range(4))
            c_giou = -giou_l[n][m]
            row.append(cost_cls * c_cls + cost_l1 * c_l1 + cost_giou * c_giou)
        cost.append(row)

    gt_idx, pred_idx = solve_assignment(cost)
    return pred_idx, gt_idx


# ---------------------------------------------------------------------------
# DETR model
# ---------------------------------------------------------------------------


class DETRForObjectDetection(ObjectDetectionModel):
    r"""DETR end-to-end object detector (Carion et al., ECCV 2020).

    The first detection model to formulate object detection as a direct
    **set prediction** problem, eliminating anchors and NMS.  A ResNet
    backbone produces a stride-32 feature map :math:`(B, C, H', W')` that
    is projected to :math:`d_\mathrm{model}` channels, flattened to a
    sequence, and processed by a stack of transformer encoder layers.
    A learned set of :math:`N` *object queries* (default 100) is then
    decoded against this memory by the transformer decoder, with each
    query producing one class logit vector (K+1, including
    "no-object") and one box prediction :math:`(c_x, c_y, w, h)`
    normalised to :math:`[0, 1]`.

    During training, predictions and ground-truth boxes are matched by
    the Hungarian algorithm under a cost combining class probability and
    box L1 + GIoU, and the same combination defines the per-pair loss.

    Parameters
    ----------
    config : DETRConfig
        Frozen architecture spec.  Use :func:`detr_resnet50` /
        :func:`detr_resnet101` for the paper-cited variants.

    Attributes
    ----------
    config : DETRConfig
        Stored copy of the config that built this model.
    backbone : _ResNetC5
        ResNet trunk through stage 5 (frozen BN), producing a stride-32
        feature map.
    input_proj : nn.Conv2d
        1x1 conv projecting backbone output to ``d_model``.
    query_embed : nn.Embedding
        Learned object queries of shape ``(num_queries, d_model)``.
    transformer : _DETRTransformer
        Reference DETR transformer (``encoder.layers.{N}`` with no final
        encoder norm; ``decoder.layers.{N}`` + a final ``decoder.norm``),
        with per-layer positional re-injection.
    class_embed : nn.Linear
        Per-query class head with output dim ``num_classes + 1`` (the
        extra "no-object" slot is essential for matching).
    bbox_embed : _MLP
        Three-layer MLP producing per-query 4-D box predictions; the
        final activation is a sigmoid so boxes lie in :math:`[0, 1]`.

    Notes
    -----
    See Carion et al., "End-to-End Object Detection with Transformers",
    ECCV 2020 (arXiv:2005.12872).  The bipartite Hungarian assignment
    finds the permutation :math:`\hat{\sigma} \in \mathfrak{S}_N` that
    minimises

    .. math::

        \hat{\sigma} = \arg\min_{\sigma \in \mathfrak{S}_N}
                       \sum_{i=1}^{N}
                       \mathcal{L}_\mathrm{match}\bigl(y_i, \hat{y}_{\sigma(i)}\bigr),

    with the pair cost

    .. math::

        \mathcal{L}_\mathrm{match}(y, \hat{y}) =
            -\mathbb{1}_{c \ne \varnothing}\,\hat{p}(c)
            + \mathbb{1}_{c \ne \varnothing}\,
              \bigl(\lambda_{\mathrm{L_1}}\|b - \hat{b}\|_1
                    + \lambda_\mathrm{GIoU}\,
                      (1 - \mathrm{GIoU}(b, \hat{b}))\bigr).

    The final loss reuses the same per-pair terms, with "no-object"
    matches penalised at reduced weight.  Removing NMS / anchors makes
    DETR conceptually simple but training-data hungry — the paper
    reports needing 500 epochs to fully converge on COCO.

    Examples
    --------
    Inference returns class logits and normalised cxcywh boxes for every
    query:

    >>> import lucid
    >>> from lucid.models.vision.detr import detr_resnet50
    >>> model = detr_resnet50()
    >>> x = lucid.randn(1, 3, 800, 800)
    >>> out = model(x)
    >>> out.logits.shape   # (B, num_queries, K + 1)
    (1, 100, 92)
    >>> out.pred_boxes.shape
    (1, 100, 4)

    Training pass with normalised xyxy ground-truth boxes (Hungarian
    matching computes the set loss):

    >>> targets = [{
    ...     "boxes":  lucid.tensor([[0.1, 0.1, 0.5, 0.5]]),
    ...     "labels": lucid.tensor([3], dtype=lucid.int64),
    ... }]
    >>> out = model(x, targets=targets)
    >>> out.loss.shape
    ()
    """

    config_class: ClassVar[type[DETRConfig]] = DETRConfig
    base_model_prefix: ClassVar[str] = "detr"

    def __init__(self, config: DETRConfig) -> None:
        super().__init__(config)
        self._cfg = config
        d = config.d_model

        # Backbone (frozen BN)
        self.backbone = _ResNetC5(config.in_channels, config.backbone_layers)
        self.input_proj = nn.Conv2d(self.backbone.out_channels, d, 1)

        # Object queries
        self.query_embed = nn.Embedding(config.num_queries, d)

        # Transformer
        self.transformer = _DETRTransformer(
            d_model=d,
            n_head=config.n_head,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
        )

        # Positional encoding (reference PositionEmbeddingSine)
        self._pos_embed = SinePositionEmbedding(num_position_features=d // 2)

        # Prediction heads
        self.class_embed = nn.Linear(d, config.num_classes + 1)
        self.bbox_embed = _MLP(
            in_dim=d,
            hidden_dim=config.bbox_hidden_dim,
            out_dim=4,
            n_layers=config.num_bbox_layers,
        )

    @override
    def forward(
        self,
        x: Tensor,
        targets: list[dict[str, Tensor]] | None = None,
        pixel_mask: Tensor | None = None,
    ) -> ObjectDetectionOutput:
        """Run DETR.

        Args:
            x:       (B, C, H, W) image batch.
            pixel_mask: Optional ``(B, H, W)`` bool/float mask, True on real
                pixels and False on padding.  Supply it whenever images were
                letter-boxed onto a common canvas, so queries do not attend to
                the padded region.
            targets: Optional training targets (list of dicts with
                     "boxes" and "labels").  Divergence, deliberate:
                     ``boxes`` are **xyxy** normalised to [0, 1], while
                     the reference criterion takes cxcywh.  Lucid's other
                     detection families all take xyxy targets, so DETR
                     matches its siblings rather than its paper's code
                     and converts internally for the L1 term — feeding
                     cxcywh here trains against the wrong corners
                     silently.  Ported reference training loops must
                     convert their targets first.

        Returns:
            ``ObjectDetectionOutput``:
              ``logits``    : (B, N, K+1) raw class logits.
              ``pred_boxes``: (B, N, 4) cxcywh normalised boxes.
              ``loss``      : set-prediction loss when targets provided.
        """
        B = int(x.shape[0])
        check_per_image(type(self).__name__, B, targets=targets)

        # 1. Backbone + projection → (B, d, H', W')
        feat: Tensor = self.backbone(x)
        feat = self.input_proj(feat)  # (B, d, H', W')

        fH = int(feat.shape[2])
        fW = int(feat.shape[3])

        # 2. 2-D positional encoding (B, d, H', W')
        pos_embed = self._pos_embed(feat)

        # 3. Transformer — returns (B, N, d)
        queries: Tensor = self.query_embed.weight  # (N, d)
        # The shipped eval preset pads every image onto a 1333x1333 canvas, so
        # without a key-padding mask each query attends to zero padding as if it
        # were image content.  ``pixel_mask`` is True on real pixels; it is
        # nearest-down-sampled to the feature grid and flattened to (B, H*W).
        kpm: Tensor | None = None
        if pixel_mask is not None:
            m = pixel_mask.to(feat.dtype)
            if m.ndim == 3:
                m = m.unsqueeze(1)
            m = F.interpolate(m, size=(fH, fW), mode="nearest")
            kpm = m.reshape(B, fH * fW) < 0.5  # True marks positions to ignore

        # Auxiliary decoding losses need every decoder layer, so the stack is
        # only requested when it will actually be used.
        want_aux = targets is not None and self._cfg.aux_loss
        hs_out = self.transformer(
            feat,
            pos_embed,
            queries,
            key_padding_mask=kpm,
            return_intermediate=want_aux,
        )  # (B, N, d), or (L, B, N, d) when want_aux

        # 4. Prediction heads.  The heads are shared across decoder layers, as
        # in the reference — the auxiliary losses supervise the same weights
        # at every depth rather than adding per-layer heads.
        hs_bn = hs_out[-1] if want_aux else hs_out
        logits: Tensor = self.class_embed(hs_bn)  # (B, N, K+1)
        pred_boxes: Tensor = F.sigmoid(self.bbox_embed(hs_bn))  # (B, N, 4)

        # 5. Loss
        loss: Tensor | None = None
        if targets is not None:
            loss = self._set_loss(logits, pred_boxes, targets)
            if want_aux:
                # Every layer but the last, weighted identically to the final
                # one — the reference simply sums the per-layer set losses.
                n_layers = int(hs_out.shape[0])
                for li in range(n_layers - 1):
                    h_l = hs_out[li]
                    aux_logits = self.class_embed(h_l)
                    aux_boxes = F.sigmoid(self.bbox_embed(h_l))
                    loss = loss + self._set_loss(aux_logits, aux_boxes, targets)

        return ObjectDetectionOutput(
            logits=logits,
            pred_boxes=pred_boxes,
            loss=loss,
        )

    # ------------------------------------------------------------------
    # Set-prediction loss
    # ------------------------------------------------------------------

    def _set_loss(
        self,
        logits: Tensor,
        pred_boxes: Tensor,
        targets: list[dict[str, Tensor]],
    ) -> Tensor:
        """Hungarian-matched set loss across the batch.

        Every tensor the loss builds itself — per-query targets and weights,
        match indices, the zero fallbacks — is created on the predictions'
        device and, if floating, in their dtype: a CPU one raised
        ``DeviceMismatch`` against Metal logits, a float32 one widened a half
        loss.  ``targets`` must already be on that device, as the predictions are.
        """
        B = int(logits.shape[0])
        N = int(logits.shape[1])
        dev, dt = logits.device.type, logits.dtype

        cls_losses: list[Tensor] = []
        l1_losses: list[Tensor] = []
        giou_losses: list[Tensor] = []
        cls_num = 0.0  # sum of the selected CE weights, the reference denominator
        num_boxes = 0  # batch-wide GT box count, the box-loss denominator

        bg_weight = 0.1  # down-weight "no object" in CE loss
        # No-object occupies the last of the ``num_classes + 1`` logit columns.
        bg_index = int(logits.shape[-1]) - 1

        for b in range(B):
            lg_b = logits[b]  # (N, K+1)
            pb_b = pred_boxes[b]  # (N, 4) cxcywh

            gt_boxes_xyxy = targets[b]["boxes"]  # (M, 4) xyxy normalised [0,1]
            gt_labels = targets[b]["labels"]  # (M,)
            M = int(gt_boxes_xyxy.shape[0])

            # Convert GT from xyxy → cxcywh for L1 cost
            gt_boxes_cxcywh = box_xyxy_to_cxcywh(gt_boxes_xyxy)  # (M, 4)

            if M == 0:
                # All queries should predict no-object.  That class is the LAST
                # logit column (index ``num_classes``), matching the reference
                # layout the shipped COCO checkpoint was trained with — index 0
                # is a real foreground class, not background.
                # The reference has no special case here: every query's target
                # is the no-object class and ``empty_weight[-1] = eos_coef``
                # applies to all of them.  An unweighted mean gave these queries
                # full weight 1.0, i.e. 10x the weight an unmatched query gets
                # in any image that *does* contain objects.
                log_sm_e = F.log_softmax(lg_b, dim=-1)
                ce_e = lucid.cat(
                    [(-log_sm_e[n, bg_index] * bg_weight).reshape(1) for n in range(N)],
                    dim=0,
                )
                cls_num += float(N) * bg_weight
                cls_losses.append(ce_e.sum())
                continue

            # Hungarian matching
            pred_idx, gt_idx = _hungarian_match(
                lg_b,
                pb_b,
                gt_labels,
                gt_boxes_cxcywh,
                cost_cls=1.0,
                cost_l1=5.0,
                cost_giou=2.0,
            )

            # Build per-query target labels (unmatched → no-object, the LAST
            # column).  Foreground labels are used as-is: they are 0-indexed and
            # address columns 0..num_classes-1 directly, which is what the
            # matcher above already assumes.
            labels_l = [int(v) for v in cast(list[int], gt_labels.tolist())]
            cls_targets_data: list[int] = [bg_index] * N
            for pi, gi in zip(pred_idx, gt_idx, strict=True):
                cls_targets_data[pi] = labels_l[gi]

            # Weight mask: matched queries get full weight; background = bg_weight
            weight_data: list[float] = [bg_weight] * N
            for pi in pred_idx:
                weight_data[pi] = 1.0

            weight = lucid.tensor(weight_data, dtype=dt, device=dev)  # (N,)
            cls_tgt = lucid.tensor(cls_targets_data, device=dev).long()  # (N,)

            # Weighted CE (per-sample weight): pick each query's target column.
            log_sm = F.log_softmax(lg_b, dim=-1)  # (N, K+1)
            picked = lucid.gather(log_sm, 1, cls_tgt.reshape(N, 1)).reshape(N)
            # ``cross_entropy(..., weight=w)`` with reduction='mean' divides by
            # ``sum_i w[target_i]``, not by N.  Accumulate the numerator and the
            # weight sum separately so the batch-level division matches.
            cls_num += float(sum(weight_data))
            cls_losses.append((-picked * weight).sum())

            if not pred_idx:
                continue

            # L1 loss on matched pairs.  The *predictions* have to be gathered
            # with a tensor op: reading them out with ``.item()`` and
            # rebuilding a tensor detaches the box head, and the 5·L1 + 2·GIoU
            # terms would keep appearing in the reported loss while
            # contributing no gradient at all.  The ground truth is a constant
            # and is gathered the same way, so it stays on its device.
            pi_t = lucid.tensor(pred_idx, device=dev).long()
            gi_t = lucid.tensor(gt_idx, device=dev).long()
            pred_matched = pb_b.index_select(0, pi_t)  # (P, 4)
            gt_matched = gt_boxes_cxcywh.index_select(0, gi_t)  # (P, 4)
            # sum over the 4 coordinates; the /num_boxes division happens
            # once at batch level, not per image and not over coordinates.
            l1_losses.append(lucid.abs(pred_matched - gt_matched).sum())
            num_boxes += len(pred_idx)

            # GIoU loss on matched pairs — the diagonal has to stay in the
            # graph for the same reason.
            pred_xyxy = box_cxcywh_to_xyxy(pred_matched)  # (P, 4)
            gt_xyxy = box_cxcywh_to_xyxy(gt_matched)  # (P, 4)
            giou_mat = generalized_box_iou(pred_xyxy, gt_xyxy)  # (P, P)
            n_match = len(pred_idx)
            diag = lucid.tensor(
                [i * n_match + i for i in range(n_match)], device=dev
            ).long()
            giou_diag = giou_mat.reshape(-1).index_select(0, diag)  # (P,)
            giou_losses.append((1.0 - giou_diag).sum())

        denom_cls = max(cls_num, 1.0)
        denom_box = float(max(num_boxes, 1))
        cls_l = (
            lucid.cat([l.reshape(1) for l in cls_losses]).sum() / denom_cls
            if cls_losses
            else logits.new_zeros()
        )
        l1_l = (
            lucid.cat([l.reshape(1) for l in l1_losses]).sum() / denom_box
            if l1_losses
            else logits.new_zeros()
        )
        giou_l = (
            lucid.cat([l.reshape(1) for l in giou_losses]).sum() / denom_box
            if giou_losses
            else logits.new_zeros()
        )

        # Scalar in every branch — the zero fallbacks used to return shape (1,),
        # so a batch with no boxes produced a differently-shaped loss.
        return (1.0 * cls_l + 5.0 * l1_l + 2.0 * giou_l).reshape(())

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def postprocess(
        self,
        output: ObjectDetectionOutput,
        image_sizes: list[tuple[int, int]],
    ) -> list[dict[str, Tensor]]:
        """Filter queries by score threshold and convert to pixel coordinates.

        Args:
            output:      Forward pass output.
            image_sizes: List of (H, W) per image.

        Returns:
            List of per-image result dicts with "boxes" (xyxy pixels),
            "scores", and "labels".
        """
        B = int(output.logits.shape[0])
        N = int(output.logits.shape[1])
        # Results live where the predictions do, like every other output.
        dev = output.logits.device.type
        results: list[dict[str, Tensor]] = []

        for b in range(B):
            lg_b = output.logits[b]  # (N, K+1)
            pb_b = output.pred_boxes[b]  # (N, 4) cxcywh [0,1]
            iH, iW = image_sizes[b]

            probs = F.softmax(lg_b, dim=-1)  # (N, K+1)
            num_fg = int(probs.shape[1]) - 1
            # One host transfer per tensor; an ``.item()`` per entry was
            # N·(K+4) device syncs per image on Metal.
            probs_l = cast(list[list[float]], probs.tolist())
            boxes_l = cast(list[list[float]], pb_b.tolist())

            keep_boxes: list[Tensor] = []
            keep_scores: list[Tensor] = []
            keep_labels: list[Tensor] = []

            for n in range(N):
                # Best foreground class.  No-object is the LAST column, so the
                # foreground columns are 0..K-1 and their index *is* the label —
                # treating column 0 as background would drop one real class and
                # shift every reported id by one against the shipped checkpoint.
                best_cls = 0
                best_sc = -1.0
                for c in range(num_fg):
                    sc = probs_l[n][c]
                    if sc > best_sc:
                        best_sc = sc
                        best_cls = c

                if best_sc < self._cfg.score_thresh:
                    continue

                # Convert cxcywh [0,1] → xyxy pixels
                cx = boxes_l[n][0] * iW
                cy = boxes_l[n][1] * iH
                w2 = boxes_l[n][2] * iW / 2.0
                h2 = boxes_l[n][3] * iH / 2.0
                box_data = [
                    [
                        max(0.0, cx - w2),
                        max(0.0, cy - h2),
                        min(float(iW), cx + w2),
                        min(float(iH), cy + h2),
                    ]
                ]
                keep_boxes.append(lucid.tensor(box_data, device=dev))
                keep_scores.append(lucid.tensor([[best_sc]], device=dev))
                keep_labels.append(
                    lucid.tensor([[best_cls]], dtype=lucid.int64, device=dev)
                )

            if keep_boxes:
                results.append(
                    detection_result(
                        lucid.cat(keep_boxes, dim=0).squeeze(1),
                        lucid.cat(keep_scores, dim=0).squeeze(1),
                        lucid.cat(keep_labels, dim=0).squeeze(1),
                    )
                )
            else:
                results.append(empty_detection_result(dev))
        return results
