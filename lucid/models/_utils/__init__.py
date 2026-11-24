"""lucid.models._utils — task-specific model utilities.

Sub-modules
-----------
_common         Cross-task helpers (make_divisible …)
_classification Image-classification helpers (LayerScale …)
_detection      Detection helpers (box_ops, NMS, anchors, RoI ops, FPN, RPN,
                Hungarian/Kuhn-Munkres assignment …)
_text           Text-LM helpers (activation dispatch, attention mask …)
_generative     Generative helpers (β-schedule, Gaussian KL, reparam, …)
_patch_grid     The input-size check of a fixed-grid patch model (ViT, I-JEPA, CLIP, DiT)

Most callers should import directly from the sub-module:

    from lucid.models._utils._common import make_divisible
    from lucid.models._utils._classification import LayerScale
    from lucid.models._utils._detection import (
        box_iou, nms, AnchorGenerator, FPN, RPN, RoIHead,
        solve_assignment,
    )
    from lucid.models._utils._text import text_activation, extended_attention_mask
    from lucid.models._utils._generative import (
        make_beta_schedule, extract_into_tensor,
        gaussian_kl_divergence, reparameterize, generative_activation,
        flow_prior_log_prob, flow_prior_sample,
    )

Alternatively the most-used names are re-exported here for convenience:

    from lucid.models._utils import make_divisible, LayerScale
"""

from lucid.models._utils._common import make_divisible
from lucid.models._utils._classification import DropPath, LayerScale
from lucid.models._utils._text import text_activation, extended_attention_mask
from lucid.models._utils._generative import (
    extract_into_tensor,
    flow_prior_log_prob,
    flow_prior_sample,
    gaussian_kl_divergence,
    generative_activation,
    make_beta_schedule,
    make_sigma_schedule,
    reparameterize,
)
from lucid.models._utils._detection import (
    # Box operations
    box_area,
    box_iou,
    generalized_box_iou,
    box_xyxy_to_cxcywh,
    box_cxcywh_to_xyxy,
    clip_boxes_to_image,
    remove_small_boxes,
    encode_boxes,
    decode_boxes,
    # NMS
    nms,
    batched_nms,
    # Anchors
    AnchorGenerator,
    # RoI ops
    roi_align,
    roi_pool,
    # Shared modules
    FPN,
    RPN,
    RoIHead,
    # Assignment
    solve_assignment,
)

__all__ = [
    # common
    "make_divisible",
    # classification
    "LayerScale",
    "DropPath",
    # text
    "text_activation",
    "extended_attention_mask",
    # generative
    "make_beta_schedule",
    "make_sigma_schedule",
    "extract_into_tensor",
    "gaussian_kl_divergence",
    "reparameterize",
    "flow_prior_log_prob",
    "flow_prior_sample",
    "generative_activation",
    # detection — box ops
    "box_area",
    "box_iou",
    "generalized_box_iou",
    "box_xyxy_to_cxcywh",
    "box_cxcywh_to_xyxy",
    "clip_boxes_to_image",
    "remove_small_boxes",
    "encode_boxes",
    "decode_boxes",
    # detection — NMS
    "nms",
    "batched_nms",
    # detection — anchors
    "AnchorGenerator",
    # detection — RoI ops
    "roi_align",
    "roi_pool",
    # detection — shared modules
    "FPN",
    "RPN",
    "RoIHead",
    # detection — assignment
    "solve_assignment",
]
