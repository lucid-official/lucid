"""Shared config validation for FPN two-stage R-CNN detectors (Faster R-CNN, Mask R-CNN)."""

from typing import Protocol

from lucid.models._utils._common import check_non_negative, check_positive, check_residual_layers

# P2-P5 from the FPN plus its max-pooled P6.
FPN_LEVELS = 5


class _FPNRCNNConfig(Protocol):
    """Fields shared by every FPN + RPN + RoI detector configuration."""

    @property
    def num_classes(self) -> int: ...
    @property
    def in_channels(self) -> int: ...
    @property
    def backbone_layers(self) -> tuple[int, int, int, int]: ...
    @property
    def backbone_bn_eps(self) -> float: ...
    @property
    def fpn_out_channels(self) -> int: ...
    @property
    def rpn_anchor_sizes(self) -> tuple[int, ...]: ...
    @property
    def rpn_anchor_ratios(self) -> tuple[float, ...]: ...
    @property
    def rpn_pre_nms_top_n(self) -> int: ...
    @property
    def rpn_post_nms_top_n(self) -> int: ...
    @property
    def rpn_min_size(self) -> float: ...
    @property
    def rpn_fg_iou_thresh(self) -> float: ...
    @property
    def rpn_bg_iou_thresh(self) -> float: ...
    @property
    def rpn_batch_size_per_image(self) -> int: ...
    @property
    def roi_sampling_ratio(self) -> int: ...
    @property
    def roi_fg_iou_thresh(self) -> float: ...
    @property
    def roi_bg_iou_thresh(self) -> float: ...
    @property
    def roi_batch_size_per_image(self) -> int: ...
    @property
    def canonical_scale(self) -> int: ...
    @property
    def max_detections(self) -> int: ...


def validate_fpn_rcnn_config(cfg: _FPNRCNNConfig, *, fpn_levels: int = FPN_LEVELS) -> None:
    """Validate the FPN backbone, RPN, and shared RoI fields of an R-CNN config.

    Parameters
    ----------
    cfg : _FPNRCNNConfig
        Frozen configuration instance (``FasterRCNNConfig`` or ``MaskRCNNConfig``).
    fpn_levels : int, default=5
        Number of FPN output levels (P2–P6) paired with ``rpn_anchor_sizes``.

    Raises
    ------
    ValueError
        If anchor levels, IoU thresholds, residual stages, or numeric counts
        are outside their valid domain.
    """
    owner = type(cfg).__name__
    if len(cfg.rpn_anchor_sizes) != fpn_levels:
        raise ValueError(
            f"{owner}: rpn_anchor_sizes holds one anchor size per FPN level "
            f"and the backbone emits {fpn_levels} (P2-P6); got "
            f"{len(cfg.rpn_anchor_sizes)}: {cfg.rpn_anchor_sizes}"
        )
    for name in ("rpn", "roi"):
        fg = float(getattr(cfg, f"{name}_fg_iou_thresh"))
        bg = float(getattr(cfg, f"{name}_bg_iou_thresh"))
        if bg > fg:
            raise ValueError(
                f"{name}_bg_iou_thresh ({bg}) must not exceed "
                f"{name}_fg_iou_thresh ({fg}); the band between them is "
                "the ignore region, and inverting them would make every "
                "prediction either a positive or a negative with no "
                "ambiguous zone."
            )
    check_positive(
        owner,
        num_classes=cfg.num_classes,
        in_channels=cfg.in_channels,
        fpn_out_channels=cfg.fpn_out_channels,
        rpn_anchor_sizes=cfg.rpn_anchor_sizes,
        rpn_anchor_ratios=cfg.rpn_anchor_ratios,
        rpn_pre_nms_top_n=cfg.rpn_pre_nms_top_n,
        rpn_post_nms_top_n=cfg.rpn_post_nms_top_n,
        rpn_batch_size_per_image=cfg.rpn_batch_size_per_image,
        roi_batch_size_per_image=cfg.roi_batch_size_per_image,
        canonical_scale=cfg.canonical_scale,
        max_detections=cfg.max_detections,
    )
    check_residual_layers(owner, backbone_layers=cfg.backbone_layers)
    check_non_negative(
        owner,
        roi_sampling_ratio=cfg.roi_sampling_ratio,
        rpn_min_size=cfg.rpn_min_size,
        backbone_bn_eps=cfg.backbone_bn_eps,
    )
