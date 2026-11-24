"""``reset_classifier`` across the zoo (LCD-1106).

Transfer learning loads a model, moves it (``.half()``, ``.to(bfloat16)``,
``.to("metal")``) and swaps the head.  The swap used to build a float32 CPU
``Linear`` and to leave ``config.num_classes`` at the old width, so the next
forward raised a dtype or device mismatch, and a ``save_pretrained`` copy did
not load back ("size mismatch for classifier.weight").  Six families repeated
the construction, and two more (GoogLeNet and Inception v3, whose auxiliary
classifiers score the same labels) left a class-dependent head behind.

The invariant is owned by ``ClassificationHeadMixin.reset_classifier``; this
file guards it for *every* family the registry holds a head-swapping class
for, so a new family has to join the table below before it can land.

* Structure, every family at its real default size, built under
  ``shadow_alloc`` (no storage): a model built from the updated config has the
  same ``state_dict`` shapes, which is what makes the saved copy loadable.
* End to end, every family at a tiny config override of a registered factory
  (never an invented factory), across device x dtype: the model runs after the
  reset, every parameter keeps the model's dtype and device, ``config`` follows,
  the saved copy reproduces the logits, and a training step reaches the new head.
* The mixin itself, on a minimal host: head shapes the families do not cover,
  and a refusal that leaves the model as it was.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pytest

import lucid
import lucid.nn as nn
from lucid.models import ModelConfig, PretrainedModel, create_model
from lucid.models._mixins import ClassificationHeadMixin
from lucid.models._output import ImageClassificationOutput
from lucid.models._registry import _REGISTRY
from lucid.nn._shadow import shadow_alloc
from lucid.test._fixtures.devices import device_dtype_params

if TYPE_CHECKING:
    from _pytest.mark import ParameterSet

type Tiny = tuple[str, dict[str, object], tuple[int, ...]]

# family -> (a registered factory, config overrides that shrink it, input shape).
# A config override is how a test gets a small model (H11); the factory is the
# family's own.  Input shapes carry batch 2 where the model has BatchNorm.
_TINY: dict[str, Tiny] = {
    "alexnet": ("alexnet_cls", {}, (1, 3, 64, 64)),
    "coatnet": (
        "coatnet_0_cls",
        dict(
            image_size=32,
            blocks_per_stage=(1, 1, 1, 1),
            dims=(8, 16, 32, 64),
            stem_width=8,
            attn_heads=(2, 2),
            head_hidden_size=16,
        ),
        (2, 3, 32, 32),
    ),
    "convnext": (
        "convnext_tiny_cls",
        dict(depths=(1, 1, 1, 1), dims=(8, 16, 32, 64)),
        (2, 3, 32, 32),
    ),
    "crossvit": (
        "crossvit_tiny_cls",
        dict(image_size=32, embed_dims=(8, 16), depths=((1, 1, 0),), num_heads=(1, 1)),
        (2, 3, 32, 32),
    ),
    "cspnet": (
        "cspresnext_50_cls",
        dict(depths=(1, 1, 1, 1), out_chs=(16, 16, 16, 16), stem_out_chs=8, groups=(1, 1, 1, 1)),
        (2, 3, 32, 32),
    ),
    "cvt": (
        "cvt_13_cls",
        dict(dims=(8, 16, 32), depths=(1, 1, 1), num_heads=(1, 1, 2)),
        (2, 3, 32, 32),
    ),
    "densenet": (
        "densenet_121_cls",
        dict(block_config=(1, 1, 1, 1), growth_rate=4, num_init_features=8),
        (2, 3, 32, 32),
    ),
    "efficientformer": (
        "efficientformer_l1_cls",
        dict(depths=(1, 1, 1, 1), embed_dims=(8, 16, 32, 64), key_dim=4, num_heads=2, resolution=1),
        (2, 3, 32, 32),
    ),
    "efficientnet": (
        "efficientnet_b0_cls",
        dict(width_mult=0.25, depth_mult=0.5),
        (2, 3, 32, 32),
    ),
    "googlenet": ("googlenet_cls", {}, (2, 3, 64, 64)),
    "ijepa": (
        "ijepa_base_16_cls",
        dict(
            image_size=64,
            patch_size=16,
            dim=16,
            depth=1,
            num_heads=2,
            predictor_dim=8,
            predictor_depth=1,
            min_keep=1,
        ),
        (2, 3, 64, 64),
    ),
    "inception": ("inception_v3_cls", {}, (2, 3, 96, 96)),
    "inception_next": (
        "inception_next_tiny_cls",
        dict(depths=(1, 1, 1, 1), dims=(8, 16, 32, 64)),
        (2, 3, 32, 32),
    ),
    "inception_resnet": ("inception_resnet_v2_cls", {}, (2, 3, 96, 96)),
    "inception_v4": ("inception_v4_cls", {}, (2, 3, 96, 96)),
    "lenet": ("lenet_5_cls", {}, (1, 1, 32, 32)),
    "maxvit": (
        "maxvit_tiny_cls",
        dict(depths=(1, 1, 1, 1), dims=(8, 16, 32, 64), window_size=2, head_dim=8),
        (2, 3, 64, 64),
    ),
    "mobilenet": ("mobilenet_025_cls", {}, (2, 3, 32, 32)),
    "mobilenet_v2": ("mobilenet_v2_075_cls", {}, (2, 3, 32, 32)),
    "mobilenet_v3": ("mobilenet_v3_small_cls", {}, (2, 3, 32, 32)),
    "mobilenet_v4": ("mobilenet_v4_conv_small_cls", {}, (2, 3, 32, 32)),
    "pvt": (
        "pvt_v2_b0_cls",
        dict(
            embed_dims=(8, 16, 32, 64),
            depths=(1, 1, 1, 1),
            num_heads=(1, 1, 2, 2),
            sr_ratios=(2, 2, 1, 1),
            mlp_ratios=(2.0, 2.0, 2.0, 2.0),
        ),
        (2, 3, 32, 32),
    ),
    "resnest": ("resnest_14_cls", dict(layers=(1, 1, 1, 1)), (2, 3, 32, 32)),
    "resnet": (
        "resnet_18_cls",
        dict(layers=(1, 1, 1, 1), hidden_sizes=(8, 8, 8, 8), stem_channels=8),
        (2, 3, 32, 32),
    ),
    "resnext": (
        "resnext_50_32x4d_cls",
        dict(layers=(1, 1, 1, 1), cardinality=2, width_per_group=2),
        (2, 3, 32, 32),
    ),
    "senet": ("se_resnet_18_cls", dict(layers=(1, 1, 1, 1)), (2, 3, 32, 32)),
    "sknet": ("sk_resnet_18_cls", dict(layers=(1, 1, 1, 1)), (2, 3, 32, 32)),
    "swin": (
        "swin_tiny_cls",
        dict(
            image_size=32,
            patch_size=2,
            embed_dim=8,
            depths=(1, 1, 1, 1),
            num_heads=(1, 1, 2, 2),
            window_size=2,
        ),
        (2, 3, 32, 32),
    ),
    "vgg": ("vgg_11_cls", {}, (1, 3, 32, 32)),
    "vit": (
        "vit_base_16_cls",
        dict(image_size=32, patch_size=16, dim=16, depth=1, num_heads=2),
        (2, 3, 32, 32),
    ),
    "vjepa": (
        "vjepa_huge_16_384_cls",
        dict(
            image_size=32,
            patch_size=16,
            tubelet_size=2,
            num_frames=4,
            dim=16,
            depth=1,
            num_heads=2,
            predictor_dim=8,
            predictor_depth=1,
        ),
        (1, 4, 3, 32, 32),
    ),
    "vjepa2": (
        "vjepa2_vit_giant_384_cls",
        dict(
            image_size=32,
            patch_size=16,
            tubelet_size=2,
            num_frames=4,
            dim=16,
            depth=1,
            num_heads=2,
            mlp_ratio=2.0,
            predictor_dim=8,
            predictor_depth=1,
            predictor_heads=2,
            num_pooler_layers=1,
        ),
        (1, 4, 3, 32, 32),
    ),
    "xception": ("xception_cls", {}, (2, 3, 64, 64)),
    "zfnet": ("zfnet_cls", {}, (1, 3, 64, 64)),
}

# Families whose config has no size knob: even their smallest registered factory is
# 10M+ parameters and seconds to build.  They run a reduced device x dtype sweep and
# skip the on-disk round trip (the shadow test covers what that load would reject).
_HEAVY = frozenset(
    {
        "alexnet",
        "googlenet",
        "inception",
        "inception_resnet",
        "inception_v4",
        "vgg",
        "xception",
        "zfnet",
    }
)

_OLD, _NEW = 11, 13
_FLOATS = (lucid.float32, lucid.float16, lucid.bfloat16)


def _label(dtype: lucid.dtype) -> str:
    return str(dtype).removeprefix("lucid.")


def _cells() -> list[ParameterSet]:
    swept = device_dtype_params(_FLOATS)
    cells: list[ParameterSet] = []
    for family in sorted(_TINY):
        if family not in _HEAVY:
            cells += [
                pytest.param(family, *p.values, id=f"{family}-{p.values[0]}-{_label(p.values[1])}")
                for p in swept
            ]
            continue
        # float32 (the training step), then a dtype change alone, then dtype and
        # device together -- each fails on a head built float32 on the CPU.
        last = swept[-1].values[0]
        for device, dtype in (
            ("cpu", lucid.float32),
            ("cpu", lucid.float16),
            (last, lucid.bfloat16),
        ):
            cells.append(
                pytest.param(family, device, dtype, id=f"{family}-{device}-{_label(dtype)}")
            )
    return cells


def _registered_head_families() -> set[str]:
    return {
        entry.family
        for entry in _REGISTRY.values()
        if entry.model_class is not None and hasattr(entry.model_class, "reset_classifier")
    }


def _shapes(model: nn.Module) -> dict[str, tuple[int, ...]]:
    return {name: tuple(t.shape) for name, t in model.state_dict().items()}


def _linears(model: nn.Module, out_features: int) -> list[str]:
    return [
        name
        for name, m in model.named_modules()
        if isinstance(m, nn.Linear) and m.out_features == out_features
    ]


def test_the_table_names_every_family_that_can_swap_a_head() -> None:
    """A family that gains a ``reset_classifier`` has to be added here to land."""
    assert set(_TINY) == _registered_head_families()
    for family, (factory, _, _) in _TINY.items():
        assert _REGISTRY[factory].family == family


# Class-sized Linear layers a family has at its default config, where that is more than one:
# CrossViT's two branch heads, EfficientFormer's head and head_dist, GoogLeNet's two
# auxiliary classifiers beside the main head.
_HEADS = {"crossvit": 2, "efficientformer": 2, "googlenet": 3}

# (family, config overrides, class-sized Linear count): every family at its defaults, plus the
# config that switches on a head the defaults leave off -- Inception v3's auxiliary classifier.
_SHADOW_RUNS = [pytest.param(f, {}, _HEADS.get(f, 1), id=f) for f in sorted(_TINY)] + [
    pytest.param("inception", {"aux_logits": True}, 2, id="inception-aux_logits")
]


@pytest.mark.parametrize(("family", "overrides", "heads"), _SHADOW_RUNS)
def test_reset_gives_what_a_model_built_at_the_new_width_has(
    family: str, overrides: dict[str, object], heads: int
) -> None:
    """Saved weights load back only if the config and every class-sized head moved together."""
    factory = _TINY[family][0]
    with shadow_alloc():
        model = create_model(factory, **overrides)
        old = model.config.num_classes
        assert len(_linears(model, old)) == heads
        model.reset_classifier(old + 3)
        rebuilt = type(model)(model.config)
        assert _shapes(model) == _shapes(rebuilt)
        assert not _linears(model, old)
        assert len(_linears(model, old + 3)) == heads
    assert model.config.num_classes == old + 3


@pytest.mark.parametrize(("family", "device", "dtype"), _cells())
def test_reset_keeps_the_model_working(
    family: str, device: str, dtype: lucid.dtype, tmp_path: Path
) -> None:
    factory, overrides, shape = _TINY[family]
    lucid.manual_seed(0)
    model = create_model(factory, num_classes=_OLD, **overrides).eval().to(dtype).to(device)
    heads = len(_linears(model, _OLD))
    assert heads

    model.reset_classifier(_NEW)

    # Every class-sized head moved, none was left at the old width, and each is
    # where the rest of the model is -- the forward below would raise otherwise.
    assert len(_linears(model, _NEW)) == heads
    assert not _linears(model, _OLD)
    for name, p in model.named_parameters():
        assert p.dtype == dtype, name
        assert p.device.type == device, name
    assert model.config.num_classes == _NEW

    x = lucid.randn(*shape).to(dtype).to(device)
    logits = model(x).logits
    assert tuple(logits.shape) == (shape[0], _NEW)
    assert logits.dtype == dtype
    assert logits.device.type == device

    if family not in _HEAVY:
        model.save_pretrained(str(tmp_path))
        again = type(model).from_pretrained(str(tmp_path)).to(dtype).to(device).eval()
        assert again.config == model.config
        np.testing.assert_array_equal(
            logits.float().cpu().numpy(), again(x).logits.float().cpu().numpy()
        )

    if dtype == lucid.float32:
        _one_training_step(model, x, [model.get_submodule(n) for n in _linears(model, _NEW)])


def _one_training_step(model: PretrainedModel, x: lucid.Tensor, heads: list[nn.Module]) -> None:
    """Forward, backward and an optimizer step: the new heads must receive gradients."""
    model.train()
    labels = lucid.zeros(int(x.shape[0]), dtype=lucid.int64).to(x.device)
    optimizer = lucid.optim.SGD(model.parameters(), lr=1e-2)
    before = [h.weight.numpy().copy() for h in heads]
    loss = model(x, labels=labels).loss
    assert loss is not None and bool(lucid.isfinite(loss).item())
    optimizer.zero_grad()
    loss.backward()
    for h in heads:
        assert h.weight.grad is not None
    optimizer.step()
    assert any(not np.array_equal(b, h.weight.numpy()) for b, h in zip(before, heads, strict=True))


# ── the mixin on a minimal host ──────────────────────────────────────────────


@dataclass(frozen=True)
class _ProbeConfig(ModelConfig):
    model_type: ClassVar[str] = "reset_probe"
    num_classes: int = 3


class _Probe(PretrainedModel, ClassificationHeadMixin):
    config_class: ClassVar[type[ModelConfig]] = _ProbeConfig

    def __init__(self, config: _ProbeConfig, *, dropout: float = 0.0) -> None:
        super().__init__(config)
        self._build_classifier(4, config.num_classes, dropout=dropout)

    def forward(self, x: lucid.Tensor) -> ImageClassificationOutput:
        return ImageClassificationOutput(logits=self.classifier(x))


class _ProbeWithMissingHead(_Probe):
    _class_heads: ClassVar[tuple[str, ...]] = ("classifier", "aux.fc")


@pytest.mark.parametrize("dtype", [lucid.float16, lucid.bfloat16])
def test_a_dropout_head_keeps_its_dropout_and_takes_the_models_dtype(dtype: lucid.dtype) -> None:
    model = _Probe(_ProbeConfig(), dropout=0.5).to(dtype)
    dropout = model.classifier[0]
    model.reset_classifier(5)
    assert model.classifier[0] is dropout
    head = model.classifier[1]
    assert isinstance(head, nn.Linear)
    assert head.weight.dtype == dtype
    assert tuple(model(lucid.randn(2, 4).to(dtype)).logits.shape) == (2, 5)


def test_the_new_head_keeps_the_old_heads_bias_and_mode() -> None:
    model = _Probe(_ProbeConfig())
    model.classifier = nn.Linear(4, 3, bias=False)
    model.eval()
    model.reset_classifier(5)
    assert isinstance(model.classifier, nn.Linear)
    assert model.classifier.bias is None
    assert not model.classifier.training


@pytest.mark.parametrize("width", [0, -2])
def test_a_width_that_is_not_positive_is_refused_and_nothing_changes(width: int) -> None:
    model = _Probe(_ProbeConfig())
    head = model.classifier
    with pytest.raises(ValueError, match="positive"):
        model.reset_classifier(width)
    assert model.classifier is head
    assert model.config.num_classes == 3


def test_a_head_path_that_names_nothing_leaves_the_model_as_it_was() -> None:
    model = _ProbeWithMissingHead(_ProbeConfig())
    head = model.classifier
    with pytest.raises(AttributeError, match="aux.fc"):
        model.reset_classifier(5)
    assert model.classifier is head
    assert model.config.num_classes == 3
