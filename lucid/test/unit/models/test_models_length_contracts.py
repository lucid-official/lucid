"""Inputs the zoo walks in step are refused, by name, when their lengths differ.

A lint pass made 115 ``zip`` calls strict.  Where two sequences are equal
by construction that is a free assertion; where a caller chooses them — a
config's per-stage tuples, a detector's per-image ``targets`` — a length
mismatch had been silently truncated, and became a bare ``zip() argument 2
is shorter than argument 1`` from deep inside a builder or a loss.  Each
such input is now checked at the boundary it enters by, with a message that
names it.  One test per family of such inputs:

* per-stage config tuples (``depths`` / ``dims`` / ``num_heads`` …), checked
  in the config;
* a per-level config tuple (R-CNN ``rpn_anchor_sizes``) that must match the
  backbone's fixed level count;
* per-image forward inputs (``targets``, ``proposals``), checked first in
  every detector's forward, so a refused call has changed no state.

YOLO's per-scale prior table is held in ``test_models_yolo_scale_table.py``.
"""

from typing import Any

import pytest

import lucid
import lucid.models as models
from lucid.models._base import ModelConfig
from lucid.models.generative.diamond import DIAMONDConfig
from lucid.models.vision.convnext import ConvNeXtConfig
from lucid.models.vision.cvt import CvTConfig
from lucid.models.vision.efficientformer import EfficientFormerConfig
from lucid.models.vision.faster_rcnn import FasterRCNNConfig
from lucid.models.vision.inception_next import InceptionNeXtConfig
from lucid.models.vision.mask_rcnn import MaskRCNNConfig
from lucid.models.vision.maxvit import MaxViTConfig
from lucid.models.vision.pvt import PVTConfig
from lucid.models.vision.swin import SwinConfig

_STAGES = r"hold one entry per stage and must be the same length"

#: (config class, the short override, what the message must say)
_PER_STAGE: list[tuple[type[ModelConfig], dict[str, Any], str]] = [
    (ConvNeXtConfig, {"depths": (1, 1, 1)}, rf"depths, dims {_STAGES}"),
    (CvTConfig, {"depths": (1, 1)}, r"dims, depths, num_heads, embed_strides"),
    (EfficientFormerConfig, {"depths": (1, 1, 1)}, r"depths, embed_dims, mlp_ratios"),
    (InceptionNeXtConfig, {"depths": (1, 1, 1)}, rf"depths, dims {_STAGES}"),
    (MaxViTConfig, {"dims": (8, 16)}, rf"depths, dims {_STAGES}"),
    (PVTConfig, {"num_heads": (1, 2)}, r"embed_dims, depths, num_heads"),
    (SwinConfig, {"num_heads": (1, 2, 4)}, rf"depths, num_heads {_STAGES}"),
    (
        DIAMONDConfig,
        {
            "upsampler_channels": (8, 8),
            "upsampler_layers": (1,),
            "upsampling_factor": 2,
        },
        r"upsampler_channels and upsampler_layers describe the same resolutions",
    ),
    (
        DIAMONDConfig,
        {
            "upsampler_channels": (8, 8),
            "upsampler_layers": (1, 1),
            "upsampler_attn_depths": (0,),
            "upsampling_factor": 2,
        },
        r"upsampler_attn_depths names one flag per upsampler resolution",
    ),
]


@pytest.mark.parametrize(
    ("config_class", "overrides", "message"),
    _PER_STAGE,
    ids=[f"{c.__name__}-{'-'.join(o)}" for c, o, _ in _PER_STAGE],
)
def test_per_stage_tuples_must_agree(
    config_class: type[ModelConfig], overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        config_class(**overrides)


@pytest.mark.parametrize("config_class", [FasterRCNNConfig, MaskRCNNConfig])
def test_rcnn_anchor_sizes_are_one_per_fpn_level(
    config_class: type[ModelConfig],
) -> None:
    with pytest.raises(ValueError, match=r"one anchor size per FPN level .* emits 5 .*got 4"):
        config_class(rpn_anchor_sizes=(32, 64, 128, 256))


# -- per-image forward inputs ------------------------------------------------

_SIDE = 64


def _targets(n: int, *, normalised: bool = False) -> list[dict[str, lucid.Tensor]]:
    scale = 1.0 if normalised else float(_SIDE)
    return [
        {
            "boxes": lucid.tensor([[0.25, 0.25, 0.75, 0.75]]) * scale,
            "labels": lucid.tensor([1]).long(),
            "masks": lucid.ones(1, _SIDE, _SIDE),
        }
        for _ in range(n)
    ]


_DETR = {
    "num_classes": 3,
    "backbone_layers": (1, 1, 1, 1),
    "n_head": 2,
    "num_encoder_layers": 1,
    "num_decoder_layers": 1,
    "dim_feedforward": 32,
    "num_queries": 4,
}
_EFFDET = {"num_classes": 3, "fpn_channels": 16, "fpn_repeats": 1, "head_repeats": 1}
_RCNN = {"num_classes": 3, "backbone_layers": (1, 1, 1, 1)}

#: (factory, config overrides, whether its boxes are normalised)
_DETECTORS: list[Any] = [
    ("yolo_tiny", {"split_size": 2, "num_classes": 3}, False),
    ("yolo_v2", {"num_classes": 3}, False),
    ("yolo_v3", {"num_classes": 3}, False),
    ("yolo_v3_tiny", {"num_classes": 3}, False),
    ("yolo_v4", {"num_classes": 3}, False),
    ("detr_resnet50", _DETR, True),
    ("efficientdet_d0", _EFFDET, False),
    ("faster_rcnn", _RCNN, False),
    ("mask_rcnn", _RCNN, False),
    ("rcnn", {"num_classes": 3}, False),
    pytest.param(
        "fast_rcnn",
        {"num_classes": 3},
        False,
        # A VGG-16 trunk with no config field that shrinks it.
        marks=pytest.mark.heavy,
    ),
    ("mask2former_swin_tiny", {"num_queries": 4, "num_decoder_layers": 1}, False),
]


@pytest.mark.parametrize(("name", "overrides", "normalised"), _DETECTORS)
def test_detectors_want_one_target_per_image(
    name: str, overrides: dict[str, Any], normalised: bool
) -> None:
    model = models.create_model(name, **overrides).train()
    x = lucid.randn(2, 3, _SIDE, _SIDE)
    for n in (1, 3):
        kwargs: dict[str, Any] = {"targets": _targets(n, normalised=normalised)}
        if name in ("rcnn", "fast_rcnn"):
            kwargs["proposals"] = [lucid.tensor([[8.0, 8.0, 40.0, 40.0]])] * 2
        with pytest.raises(ValueError, match=rf"got 2 images but {n} entries in targets"):
            model(x, **kwargs)


@pytest.mark.parametrize(
    ("name", "overrides"), [("rcnn", {"num_classes": 3}), ("faster_rcnn", _RCNN)]
)
def test_detectors_want_one_proposal_set_per_image(name: str, overrides: dict[str, Any]) -> None:
    model = models.create_model(name, **overrides).train()
    x = lucid.randn(2, 3, _SIDE, _SIDE)
    proposals = [lucid.tensor([[8.0, 8.0, 40.0, 40.0]])]
    with pytest.raises(ValueError, match=r"got 2 images but 1 entries in proposals"):
        model(x, proposals=proposals)


def test_matching_per_image_inputs_pass_the_check() -> None:
    """The check refuses only a mismatch: one entry per image trains."""
    model = models.create_model("yolo_v3_tiny", num_classes=3).train()
    out = model(lucid.randn(2, 3, _SIDE, _SIDE), targets=_targets(2))
    assert out.loss is not None
