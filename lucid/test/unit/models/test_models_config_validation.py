"""A vision config refuses a value its model cannot build, at construction.

The newer configs (BERT, CLIP, DiT, I-JEPA …) validate in ``__post_init__``;
the older vision ones did not, so a value the architecture cannot use
surfaced far from where it was written: a ``ShapeMismatch`` from the first
forward (an embedding width that does not split across its heads), an
``IndexError`` from the builder (a per-stage tuple of the wrong length), or
no error at all (a zero-width ``Linear`` head from ``num_classes=0``, a
misspelt ``variant`` quietly selecting the other layout).  One test per kind
of input:

* the sweep, which holds *every* vision config class to the ``num_classes`` /
  ``in_channels`` rule, so a family added later cannot skip it;
* the structural rules one family needs (heads, patches, stage counts);
* the same refusal reached through the public factory's ``**overrides``;
* every registered default still constructs and round-trips, and so does
  the shrunken config the zoo's compiled / device tests build from it;
* the values the models do handle, which stay accepted.
"""

import dataclasses
from typing import Any

import pytest

import lucid.models as models
from lucid.models._base import ModelConfig
from lucid.models._registry import _REGISTRY
from lucid.models.vision.attention_unet import AttentionUNetConfig
from lucid.models.vision.coatnet import CoAtNetConfig
from lucid.models.vision.crossvit import CrossViTConfig
from lucid.models.vision.convnext import ConvNeXtConfig
from lucid.models.vision.cspnet import CSPNetConfig
from lucid.models.vision.cvt import CvTConfig
from lucid.models.vision.densenet import DenseNetConfig
from lucid.models.vision.detr import DETRConfig
from lucid.models.vision.efficientnet import EfficientNetConfig
from lucid.models.vision.faster_rcnn import FasterRCNNConfig
from lucid.models.vision.fcn import FCNConfig
from lucid.models.vision.mask_rcnn import MaskRCNNConfig
from lucid.models.vision.maskformer import MaskFormerConfig
from lucid.models.vision.maxvit import MaxViTConfig
from lucid.models.vision.mobilenet import MobileNetConfig
from lucid.models.vision.mobilenet_v2 import MobileNetV2Config
from lucid.models.vision.mobilenet_v3 import MobileNetV3Config
from lucid.models.vision.pvt import PVTConfig
from lucid.models.vision.resnest import ResNeStConfig
from lucid.models.vision.resnet import ResNetConfig
from lucid.models.vision.resnext import ResNeXtConfig
from lucid.models.vision.senet import SENetConfig
from lucid.models.vision.sknet import SKNetConfig
from lucid.models.vision.swin import SwinConfig
from lucid.models.vision.unet import UNetConfig
from lucid.models.vision.vgg import VGGConfig
from lucid.models.vision.vit import ViTConfig
from lucid.models.vision.yolo import YOLOConfig, YOLOV2Config
from lucid.test.unit.compile import _zoo_matrix as zoo


def _vision_config_classes() -> list[type[ModelConfig]]:
    """Every concrete config class defined under ``lucid.models.vision``."""
    found: dict[str, type[ModelConfig]] = {}
    stack: list[type[ModelConfig]] = [ModelConfig]
    while stack:
        for sub in stack.pop().__subclasses__():
            stack.append(sub)
            if sub.__module__.startswith("lucid.models.vision.") and sub.model_type != "base":
                found[sub.__name__] = sub
    return [found[name] for name in sorted(found)]


_VISION_CONFIGS = _vision_config_classes()
_WITH_FIELD = {
    name: [c for c in _VISION_CONFIGS if name in {f.name for f in dataclasses.fields(c)}]
    for name in ("num_classes", "in_channels")
}


def _constructs(
    config_class: type[ModelConfig], overrides: dict[str, Any]
) -> type[Exception] | None:
    """The exception class construction raised, or ``None`` if it was accepted."""
    try:
        config_class(**overrides)
    except Exception as exc:
        return type(exc)
    return None


def test_the_sweep_sees_the_vision_zoo() -> None:
    """A guard on the guard: the enumeration must not quietly come up short."""
    names = {c.__name__ for c in _VISION_CONFIGS}
    assert {"ViTConfig", "SwinConfig", "ResNetConfig", "YOLOV3Config", "UNetConfig"} <= names
    assert len(_WITH_FIELD["num_classes"]) >= 45
    assert len(_WITH_FIELD["in_channels"]) >= 45


@pytest.mark.parametrize("bad", [0, -3])
@pytest.mark.parametrize("field", ["num_classes", "in_channels"])
def test_every_vision_config_refuses_an_empty_or_negative_width(field: str, bad: int) -> None:
    """``num_classes=0`` built a zero-width head; ``-3`` was an engine ``InvalidArgument``."""
    accepted = [
        c.__name__ for c in _WITH_FIELD[field] if _constructs(c, {field: bad}) is not ValueError
    ]
    assert accepted == [], f"{accepted} accept {field}={bad}"


def test_the_zero_width_refusal_names_the_field_and_the_value() -> None:
    with pytest.raises(ValueError, match=r"ViTConfig: num_classes must be positive, got 0"):
        ViTConfig(num_classes=0)
    with pytest.raises(ValueError, match=r"ResNetConfig: in_channels must be positive, got -3"):
        ResNetConfig(in_channels=-3)


_HEADS = r"must be divisible by"
_STAGES = r"hold one entry per stage and the network has"

#: (config class, the bad override, what the message must say)
_STRUCTURAL: list[tuple[type[ModelConfig], dict[str, Any], str]] = [
    # -- attention widths that must split across their heads ---------------
    (ViTConfig, {"dim": 30, "num_heads": 4}, rf"dim \(30\) {_HEADS} num_heads \(4\)"),
    (ViTConfig, {"image_size": 30, "patch_size": 16}, rf"image_size \(30\) {_HEADS} patch_size"),
    (ViTConfig, {"num_heads": 0}, r"num_heads must be positive, got 0"),
    (ViTConfig, {"patch_size": 0}, r"patch_size must be positive, got 0"),
    (ViTConfig, {"image_size": -224}, r"image_size must be positive, got -224"),
    (
        SwinConfig,
        {"embed_dim": 16, "num_heads": (3, 3, 3, 3)},
        rf"stage width\[0\] \(16\) {_HEADS}",
    ),
    (
        SwinConfig,
        {"embed_dim": 16, "num_heads": (2, 3, 2, 2)},
        rf"stage width\[1\] \(32\) {_HEADS}",
    ),
    (SwinConfig, {"window_size": 0}, r"window_size must be positive, got 0"),
    (SwinConfig, {"embed_dim": 0}, r"embed_dim must be positive, got 0"),
    (SwinConfig, {"num_heads": (3, 0, 12, 24)}, r"num_heads\[1\] must be positive, got 0"),
    (
        PVTConfig,
        {"embed_dims": (64, 129, 320, 512)},
        rf"embed_dims\[1\] \(129\) {_HEADS} num_heads",
    ),
    (CvTConfig, {"dims": (64, 100, 384)}, rf"dims\[1\] \(100\) {_HEADS} num_heads\[1\] \(3\)"),
    (MaxViTConfig, {"head_dim": 12}, rf"dims\[0\] \(64\) {_HEADS} head_dim \(12\)"),
    (MaxViTConfig, {"head_dim": 0}, r"head_dim must be positive, got 0"),
    (MaxViTConfig, {"window_size": 0}, r"window_size must be positive, got 0"),
    (CoAtNetConfig, {"attn_heads": (7, 24)}, rf"attention width of stage 3 \(384\) {_HEADS}"),
    (CoAtNetConfig, {"attn_heads": (12, 25)}, rf"dims\[3\] \(768\) {_HEADS} attn_heads\[1\]"),
    (CrossViTConfig, {"num_heads": (5, 3)}, rf"embed_dims\[0\] \(96\) {_HEADS} num_heads\[0\]"),
    # -- per-stage tuples of the wrong length ------------------------------
    (ResNetConfig, {"hidden_sizes": (8, 16)}, rf"layers, hidden_sizes {_STAGES} 4 stages"),
    (ResNetConfig, {"layers": (1, 1, 1)}, rf"layers, hidden_sizes {_STAGES} 4 stages"),
    (ResNetConfig, {"layers": (1, 1, 1, 1, 1)}, rf"layers, hidden_sizes {_STAGES} 4 stages"),
    (ResNeXtConfig, {"layers": (3, 4, 6)}, rf"layers {_STAGES} 4 stages"),
    (ResNeStConfig, {"layers": (3, 4, 6)}, rf"layers {_STAGES} 4 stages"),
    (SENetConfig, {"layers": (3, 4, 6)}, rf"layers {_STAGES} 4 stages"),
    (SKNetConfig, {"layers": (3, 4, 6)}, rf"layers {_STAGES} 4 stages"),
    (DETRConfig, {"backbone_layers": (3, 4, 6)}, rf"backbone_layers {_STAGES} 4 stages"),
    (FasterRCNNConfig, {"backbone_layers": (3, 4)}, rf"backbone_layers {_STAGES} 4 stages"),
    (MaskRCNNConfig, {"backbone_layers": (3, 4)}, rf"backbone_layers {_STAGES} 4 stages"),
    (MaskFormerConfig, {"backbone_layers": (3, 4)}, rf"backbone_layers {_STAGES} 4 stages"),
    (CoAtNetConfig, {"dims": (96, 192, 384)}, rf"blocks_per_stage, dims {_STAGES} 4 stages"),
    (CrossViTConfig, {"patch_sizes": (12,)}, rf"patch_sizes, embed_dims, num_heads {_STAGES} 2"),
    # -- a depth the builder would silently round up, and zero-width layers -
    (ResNetConfig, {"layers": (3, 0, 6, 3)}, r"layers\[1\] must be positive, got 0"),
    (DETRConfig, {"backbone_layers": (0, 1, 1, 1)}, r"backbone_layers\[0\] must be positive"),
    (FasterRCNNConfig, {"backbone_layers": (1, 1, 0, 1)}, r"backbone_layers\[2\] must be positive"),
    (MaskRCNNConfig, {"backbone_layers": (1, 1, 1, -1)}, r"backbone_layers\[3\] must be positive"),
    (MaskFormerConfig, {"backbone_layers": (0, 1, 1, 1)}, r"backbone_layers\[0\] must be positive"),
    (CoAtNetConfig, {"blocks_per_stage": (2, 0, 5, 2)}, r"blocks_per_stage\[1\] must be positive"),
    (
        CoAtNetConfig,
        {"mixed_s3": (8, 42)},
        r"mixed_s3 is \(MBConv blocks, .*holds 3 entries; got 2",
    ),
    (CoAtNetConfig, {"mixed_s3": (0, 42, 1536)}, r"mixed_s3\[0\] must be positive, got 0"),
    (CoAtNetConfig, {"mixed_s3": (8, 42, 1535)}, rf"attention width of stage 3 \(1535\) {_HEADS}"),
    (SKNetConfig, {"min_attn_channels": -1}, r"min_attn_channels must be non-negative, got -1"),
    (ResNetConfig, {"stem_channels": 0}, r"stem_channels must be positive, got 0"),
    (ResNetConfig, {"hidden_sizes": (64, 0, 256, 512)}, r"hidden_sizes\[1\] must be positive"),
    (ResNeXtConfig, {"cardinality": 0}, r"cardinality must be positive, got 0"),
    (ResNeStConfig, {"radix": 0}, r"radix must be positive, got 0"),
    (SENetConfig, {"reduction": 0}, r"reduction must be positive, got 0"),
    (SKNetConfig, {"rd_divisor": 0}, r"rd_divisor must be positive, got 0"),
    (VGGConfig, {"arch": (2, 2, 0, 3, 3)}, r"arch\[2\] must be positive, got 0"),
    (DenseNetConfig, {"growth_rate": 0}, r"growth_rate must be positive, got 0"),
    (CSPNetConfig, {"stem_out_chs": 0}, r"stem_out_chs must be positive, got 0"),
    (YOLOConfig, {"split_size": 0}, r"split_size must be positive, got 0"),
    (YOLOV2Config, {"num_anchors": 0, "anchors": ()}, r"num_anchors must be positive, got 0"),
    # -- scale factors and enumerations the builder branches on ------------
    (EfficientNetConfig, {"width_mult": 0.0}, r"width_mult must be positive, got 0.0"),
    (EfficientNetConfig, {"depth_mult": -1.0}, r"depth_mult must be positive, got -1.0"),
    (MobileNetConfig, {"width_mult": 0.0}, r"width_mult must be positive, got 0.0"),
    (MobileNetV2Config, {"width_mult": -0.5}, r"width_mult must be positive, got -0.5"),
    (MobileNetV3Config, {"width_mult": 0.0}, r"width_mult must be positive, got 0.0"),
    (MobileNetV3Config, {"variant": "smal"}, r"variant must be one of \('large', 'small'\)"),
    (
        SENetConfig,
        {"block_type": "Bottleneck"},
        r"block_type must be one of \('basic', 'bottleneck'",
    ),
    (SKNetConfig, {"block_type": "bottle"}, r"block_type must be one of \('basic', 'bottleneck'"),
    (FCNConfig, {"backbone": "resnet18"}, r"backbone must be one of \('resnet50', 'resnet101'\)"),
    (UNetConfig, {"depth": 0}, r"depth must be positive, got 0"),
    (UNetConfig, {"base_channels": 0}, r"base_channels must be positive, got 0"),
    (UNetConfig, {"dim": 4}, r"dim must be one of \(2, 3\), got 4"),
    (UNetConfig, {"block": "resnet"}, r"block must be one of \('basic', 'res'\)"),
    (AttentionUNetConfig, {"depth": 0}, r"depth must be positive, got 0"),
    (AttentionUNetConfig, {"spatial_dims": 4}, r"spatial_dims must be one of \(2, 3\), got 4"),
]


@pytest.mark.parametrize(
    ("config_class", "overrides", "message"),
    _STRUCTURAL,
    ids=[f"{c.__name__}-{'-'.join(o)}" for c, o, _ in _STRUCTURAL],
)
def test_a_value_the_architecture_cannot_use_is_refused_at_construction(
    config_class: type[ModelConfig], overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        config_class(**overrides)


#: Values the models handle, so the configs must keep accepting them: the
#: refusals above are policy only where the request would be silently ignored.
_STILL_ACCEPTED: list[tuple[type[ModelConfig], dict[str, Any]]] = [
    # No floor (Eq. 4): ``make_divisible`` already floors at the divisor.
    (SKNetConfig, {"min_attn_channels": 0}),
    # The model skips DropBlock when the block is smaller than one pixel.
    (ResNeStConfig, {"drop_block_size": 0}),
    # Transformer stages the builders build empty without complaint.
    (SwinConfig, {"depths": (1, 0, 1, 1)}),
    (ConvNeXtConfig, {"depths": (1, 0, 1, 1)}),
    (ViTConfig, {"depth": 0}),
    # A stage count other than four is the point of these configs.
    (ConvNeXtConfig, {"depths": (1, 1, 1), "dims": (8, 16, 32)}),
    (CoAtNetConfig, {"mixed_s3": (8, 42, 1536)}),
]


@pytest.mark.parametrize(
    ("config_class", "overrides"),
    _STILL_ACCEPTED,
    ids=[f"{c.__name__}-{'-'.join(o)}" for c, o in _STILL_ACCEPTED],
)
def test_a_value_the_model_handles_is_still_accepted(
    config_class: type[ModelConfig], overrides: dict[str, Any]
) -> None:
    config_class(**overrides)


#: (factory, bad override): the refusal must reach the public entry point,
#: before a model is built (the override is merged into the config first).
_FACTORIES: list[tuple[str, dict[str, Any]]] = [
    ("vit_base_16_cls", {"image_size": 32, "patch_size": 16, "dim": 30, "num_heads": 4}),
    ("vit_base_16_cls", {"num_classes": 0}),
    ("swin_tiny_cls", {"embed_dim": 16, "num_heads": (3, 3, 3, 3)}),
    ("resnet_18_cls", {"hidden_sizes": (8, 16)}),
    ("resnet_18_cls", {"num_classes": -3}),
    ("resnet_50", {"in_channels": 0}),
    ("faster_rcnn", {"backbone_layers": (3, 4, 6)}),
    ("unet", {"depth": 0}),
    ("mobilenet_v3_small_cls", {"variant": "smal"}),
    ("efficientnet_b0_cls", {"width_mult": 0.0}),
]


@pytest.mark.parametrize(
    ("name", "overrides"), _FACTORIES, ids=[f"{n}-{'-'.join(o)}" for n, o in _FACTORIES]
)
def test_a_bad_override_is_refused_through_the_factory(
    name: str, overrides: dict[str, Any]
) -> None:
    with pytest.raises(ValueError):
        models.create_model(name, **overrides)


def test_a_config_loaded_from_json_is_validated_too() -> None:
    """``from_dict`` is the third door in (after the class and the factory)."""
    saved = ResNetConfig().to_dict()
    saved["layers"] = [3, 4, 6]
    with pytest.raises(ValueError, match=rf"layers, hidden_sizes {_STAGES} 4 stages"):
        ResNetConfig.from_dict(saved)


_VISION_DEFAULTS = sorted(
    name
    for name, entry in _REGISTRY.items()
    if ".vision." in entry.model_class.__module__ and entry.default_config is not None
)


@pytest.mark.parametrize("name", _VISION_DEFAULTS)
def test_every_registered_default_config_still_constructs(name: str) -> None:
    """The new refusals reject nothing the zoo ships: a default survives a rebuild.

    ``dataclasses.replace`` is what each factory does with its ``**overrides``,
    and ``from_dict(to_dict())`` is what ``from_pretrained`` does with a saved
    ``config.json`` (lists back to tuples) — both run ``__post_init__`` again.
    """
    config = _REGISTRY[name].default_config
    assert dataclasses.replace(config) == config
    assert type(config).from_dict(config.to_dict()) == config


@pytest.mark.parametrize("name", _VISION_DEFAULTS)
def test_the_zoo_matrix_shrunken_config_is_not_refused(name: str) -> None:
    """The accept side: the compiled / device / dtype zoo tests all build each
    factory from ``shrink(default) + SPECS[family].overrides``.

    Their size-by-field-name rules know nothing of these refusals: shrinking
    every int field called ``dim`` to 32 hands UNet (where ``dim`` is the
    spatial rank, 2 or 3) a rank its config refuses, and five unrelated zoo
    tests failed.  Replaying the same shrink here fails in this file, next to
    the refusal that caused it.
    """
    entry = _REGISTRY[name]
    overrides = {
        **zoo.shrink(entry.default_config),
        **zoo.SPECS.get(str(entry.family), zoo.Spec()).overrides,
    }
    dataclasses.replace(entry.default_config, **overrides)
