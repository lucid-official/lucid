"""One training step for every model family, through its public factory.

The zoo's rule is that a family which cannot be trained from
``create_model`` is not finished, whatever its own tests say.  Those tests
reach inside — build a config directly, call a sub-module, assert on a
shape — and none of it takes the path a caller takes.  When this file was
written, 44 of the 70 files in this directory never called ``backward``,
and ``test_models_recent_e2e.py`` covers the families added since
2026-07-27 while leaving the older vision families out on purpose.  The
classifiers and the text models — most of the zoo by count — were trained
nowhere.

So each family here takes one step: factory in, forward with labels,
``.loss``, backward, SGD.  Three things are asserted, each a way the
wiring breaks without a shape ever looking wrong:

* the loss is a finite scalar;
* every parameter received a finite gradient — a head the loss does not
  reach, or a branch the forward skips, shows up here and nowhere else;
* the step moved the weights.

Every model is built at unit-test size through config overrides on a real
factory (H11), never an invented variant.  Families trained elsewhere, and
families not yet trained anywhere, are named at the bottom with where or
why — ``test_every_family_takes_a_step_or_says_where`` fails for a family
that is neither.
"""

from collections.abc import Callable
from typing import Any

import pytest

import lucid
import lucid.models as models
import lucid.optim as optim
from lucid.models._registry import _REGISTRY

_BATCH = 2
_CLASSES = 4

Inputs = tuple[tuple[Any, ...], dict[str, Any]]


def _images(side: int, channels: int = 3) -> Callable[[], Inputs]:
    def make() -> Inputs:
        x = lucid.randn(_BATCH, channels, side, side)
        labels = lucid.randint(0, _CLASSES, (_BATCH,))
        return (x,), {"labels": labels}

    return make


def _tokens(vocab: int, length: int = 8) -> Callable[[], Inputs]:
    def make() -> Inputs:
        ids = lucid.randint(0, vocab, (_BATCH, length))
        return (ids,), {"labels": ids}

    return make


def _seq2seq(vocab: int, length: int = 8) -> Callable[[], Inputs]:
    def make() -> Inputs:
        source = lucid.randint(0, vocab, (_BATCH, length))
        target = lucid.randint(0, vocab, (_BATCH, length))
        return (source, target), {"labels": target}

    return make


def _pixels(side: int) -> Callable[[], Inputs]:
    def make() -> Inputs:
        return (lucid.randn(_BATCH, 3, side, side),), {}

    return make


_TEXT = {
    "vocab_size": 64,
    "hidden_size": 16,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "intermediate_size": 32,
    "max_position_embeddings": 32,
}

#: family -> (factory, config overrides, inputs).  One representative per
#: family, the smallest real variant, shrunk through its own config.
CASES: list[Any] = [
    # -- image classification: ``model(x, labels=...)`` answers ``.loss`` --
    ("alexnet", "alexnet_cls", {}, _images(64)),
    (
        "coatnet",
        "coatnet_0_cls",
        {
            "image_size": 64,
            "blocks_per_stage": (1, 1, 1, 1),
            "dims": (16, 32, 64, 128),
            "stem_width": 16,
            "attn_heads": (2, 4),
            "head_hidden_size": 32,
        },
        _images(64),
    ),
    (
        "convnext",
        "convnext_tiny_cls",
        {"depths": (1, 1, 1, 1), "dims": (16, 32, 64, 128)},
        _images(64),
    ),
    (
        "crossvit",
        "crossvit_tiny_cls",
        {"embed_dims": (24, 48), "depths": ((1, 1, 0),), "num_heads": (1, 1)},
        _images(240),
    ),
    ("cspnet", "cspresnet_50_cls", {"depths": (1, 1, 1, 1)}, _images(64)),
    (
        "cvt",
        "cvt_13_cls",
        {"dims": (16, 32, 64), "depths": (1, 1, 1), "num_heads": (1, 2, 4)},
        _images(64),
    ),
    (
        "densenet",
        "densenet_121_cls",
        {"growth_rate": 8, "block_config": (1, 1, 1, 1)},
        _images(64),
    ),
    (
        "efficientformer",
        "efficientformer_l1_cls",
        {
            "depths": (1, 1, 1, 1),
            "embed_dims": (16, 32, 48, 64),
            "key_dim": 8,
            "num_heads": 2,
            "resolution": 2,
        },
        _images(64),
    ),
    (
        "efficientnet",
        "efficientnet_b0_cls",
        {"width_mult": 0.25, "depth_mult": 0.25},
        _images(64),
    ),
    # Auxiliary classifiers on, as the paper trains it: the loss has to
    # reach them or their parameters get no gradient.
    ("googlenet", "googlenet_cls", {}, _images(224)),
    ("inception", "inception_v3_cls", {}, _images(96)),
    (
        "inception_next",
        "inception_next_tiny_cls",
        {"depths": (1, 1, 1, 1), "dims": (16, 32, 64, 128)},
        _images(64),
    ),
    ("inception_resnet", "inception_resnet_v2_cls", {}, _images(96)),
    ("inception_v4", "inception_v4_cls", {}, _images(96)),
    ("lenet", "lenet_5_cls", {}, _images(32, channels=1)),
    (
        "maxvit",
        "maxvit_tiny_cls",
        {
            "depths": (1, 1, 1, 1),
            "dims": (16, 32, 64, 128),
            "window_size": 2,
            "head_dim": 8,
            "stem_width": 16,
        },
        _images(64),
    ),
    ("mobilenet", "mobilenet_025_cls", {}, _images(64)),
    ("mobilenet_v2", "mobilenet_v2_075_cls", {}, _images(64)),
    ("mobilenet_v3", "mobilenet_v3_small_cls", {}, _images(64)),
    # A hybrid, so the Mobile MQA blocks and layer scales take a step too.
    ("mobilenet_v4", "mobilenet_v4_hybrid_medium_cls", {}, _images(64)),
    (
        "pvt",
        "pvt_v2_b0_cls",
        {
            "embed_dims": (16, 32, 64, 128),
            "depths": (1, 1, 1, 1),
            "num_heads": (1, 2, 4, 8),
        },
        _images(64),
    ),
    (
        "resnest",
        "resnest_14_cls",
        {"bottleneck_width": 16, "stem_width": 16},
        _images(64),
    ),
    (
        "resnet",
        "resnet_18_cls",
        {
            "layers": (1, 1, 1, 1),
            "stem_channels": 16,
            "hidden_sizes": (16, 32, 64, 128),
        },
        _images(64),
    ),
    (
        "resnext",
        "resnext_50_32x4d_cls",
        {"layers": (1, 1, 1, 1), "width_per_group": 2},
        _images(64),
    ),
    ("senet", "se_resnet_18_cls", {"layers": (1, 1, 1, 1)}, _images(64)),
    (
        "sknet",
        "sk_resnet_18_cls",
        {"layers": (1, 1, 1, 1), "base_width": 16},
        _images(64),
    ),
    (
        "swin",
        "swin_tiny_cls",
        {
            "image_size": 64,
            "embed_dim": 16,
            "depths": (2, 2, 2, 2),
            "num_heads": (1, 2, 4, 8),
            "window_size": 2,
        },
        _images(64),
    ),
    pytest.param(
        "vgg",
        "vgg_11_cls",
        {},
        _images(32),
        # 132.9M parameters and no config field that shrinks the classifier.
        marks=pytest.mark.heavy,
        id="vgg",
    ),
    (
        "vit",
        "vit_base_16_cls",
        {"image_size": 32, "patch_size": 8, "dim": 32, "depth": 2, "num_heads": 2},
        _images(32),
    ),
    ("xception", "xception_cls", {}, _images(96)),
    ("zfnet", "zfnet_cls", {}, _images(64)),
    # -- text: ``model(ids, labels=ids)`` answers ``.loss`` --
    ("bert", "bert_base_mlm", _TEXT, _tokens(64)),
    ("gpt", "gpt_lm", _TEXT, _tokens(64)),
    ("gpt2", "gpt2_small_lm", _TEXT, _tokens(64)),
    ("roformer", "roformer_mlm", {**_TEXT, "embedding_size": 16}, _tokens(64)),
    (
        "transformer",
        "transformer_base_seq2seq",
        {**_TEXT, "num_decoder_layers": 1},
        _seq2seq(64),
    ),
    # -- generative: ``model(x)`` answers ``.loss`` --
    (
        "vae",
        "vae_gen",
        {"latent_dim": 8, "down_block_channels": (8, 16, 32)},
        _pixels(32),
    ),
    # Class-conditional, ``model(x, labels=...)``.  The loss back-propagates
    # through a Jacobian-vector product, so every parameter has to be
    # reached through it as well as through the plain forward.
    (
        "tvm",
        "tvm_imagenet256_gen",
        {
            "sample_size": 8,
            "hidden_size": 32,
            "depth": 1,
            "num_heads": 4,
            "num_classes": _CLASSES,
        },
        _images(8, channels=4),
    ),
]


# -- detection and segmentation: the loss needs targets, and each family
# -- wants them in its own shape --------------------------------------------


def _boxes(side: int, *, normalised: bool) -> Callable[[], Inputs]:
    """One box and label per image, ``xyxy`` — in pixels or in [0, 1]."""

    def make() -> Inputs:
        x = lucid.randn(_BATCH, 3, side, side)
        scale = 1.0 if normalised else float(side)
        box = lucid.tensor([[0.25, 0.25, 0.75, 0.75]]) * scale
        targets = [
            {"boxes": box, "labels": lucid.randint(0, _CLASSES, (1,))}
            for _ in range(_BATCH)
        ]
        return (x,), {"targets": targets}

    return make


def _class_map(side: int, channels: int = 3) -> Callable[[], Inputs]:
    """A per-pixel class map, ``(B, H, W)``."""

    def make() -> Inputs:
        x = lucid.randn(_BATCH, channels, side, side)
        classes = lucid.randint(0, _CLASSES, (_BATCH, side, side))
        return (x,), {"targets": classes}

    return make


def _mask_map(side: int) -> Callable[[], Inputs]:
    """MaskFormer's targets: a mapping holding the ``(B, H, W)`` class map."""

    def make() -> Inputs:
        x = lucid.randn(_BATCH, 3, side, side)
        masks = lucid.randint(0, _CLASSES, (_BATCH, side, side))
        return (x,), {"targets": {"masks": masks}}

    return make


CASES += [
    (
        "yolo",
        "yolo_v3_tiny",
        {"num_classes": _CLASSES},
        _boxes(64, normalised=False),
    ),
    (
        "detr",
        "detr_resnet50",
        {
            "num_classes": _CLASSES,
            "backbone_layers": (1, 1, 1, 1),
            "n_head": 2,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "dim_feedforward": 32,
            "num_queries": 4,
        },
        _boxes(64, normalised=True),
    ),
    (
        "efficientdet",
        "efficientdet_d0",
        {
            "num_classes": _CLASSES,
            "image_size": 256,
            "fpn_channels": 16,
            "fpn_repeats": 1,
            "head_repeats": 1,
        },
        _boxes(256, normalised=False),
    ),
    (
        "unet",
        "unet",
        {"num_classes": _CLASSES, "base_channels": 8, "depth": 2},
        _class_map(32, channels=1),
    ),
    (
        "fcn",
        "fcn_resnet50",
        {
            "num_classes": _CLASSES,
            "classifier_hidden_channels": 16,
            "aux_hidden_channels": 16,
        },
        _class_map(64),
    ),
    (
        "maskformer",
        "maskformer_resnet50",
        {
            "num_classes": _CLASSES,
            "backbone_layers": (1, 1, 1, 1),
            "n_head": 2,
            "num_decoder_layers": 1,
            "dim_feedforward": 32,
            "num_queries": 4,
            "fpn_out_channels": 32,
        },
        _mask_map(64),
    ),
]


def _family_of(case: Any) -> str:
    return str(case.values[0] if hasattr(case, "values") else case[0])


#: Parameters a family builds and its training loss never reaches, by
#: design.  BERT's and RoFormer's masked-LM heads sit on a base model that
#: always builds the pooler, and masked-LM never calls it.  The reference's
#: MLM model does not build one; dropping it here would change the names
#: checkpoints load against, so it stays and is named instead.
_UNUSED: dict[str, tuple[str, ...]] = {
    "bert": ("bert.pooler.",),
    "roformer": ("roformer.pooler.",),
}


@pytest.mark.parametrize(
    ("family", "factory", "overrides", "make_inputs"),
    CASES,
    ids=[_family_of(c) for c in CASES],
)
def test_a_family_takes_a_training_step(
    family: str,
    factory: str,
    overrides: dict[str, Any],
    make_inputs: Callable[[], Inputs],
) -> None:
    assert (
        _REGISTRY[factory].family == family
    ), f"{factory} belongs to {_REGISTRY[factory].family!r}, not {family!r}"
    model = models.create_model(factory, **overrides)
    model.train()
    named = list(model.named_parameters())
    # A parameter that does not require grad is one the step can never
    # move — BERT's word table was built that way, and filtering on
    # ``requires_grad`` here is exactly how a first draft missed it.
    frozen = [n for n, p in named if not p.requires_grad]
    assert not frozen, f"built with frozen parameters: {frozen[:6]}"
    before = {n: p.detach().clone() for n, p in named}

    args, kwargs = make_inputs()
    loss = model(*args, **kwargs).loss
    assert loss.ndim == 0, f"loss has shape {loss.shape}, not a scalar"
    assert bool(loss.isfinite().item()), f"loss is {float(loss.item())}"
    loss.backward()

    unused = _UNUSED.get(family, ())
    missing = [n for n, p in named if p.grad is None and not n.startswith(unused)]
    assert (
        not missing
    ), f"{len(missing)} of {len(named)} parameters got no gradient: {missing[:6]}"
    broken = [
        n
        for n, p in named
        if p.grad is not None and not bool(p.grad.isfinite().all().item())
    ]
    assert not broken, f"non-finite gradients in {broken[:6]}"

    optim.SGD([p for _n, p in named], lr=1e-2).step()
    moved = [
        n for n, p in named if float((p.detach() - before[n]).abs().max().item()) > 0
    ]
    assert moved, "the optimizer step changed no parameter"


#: Families trained by another file in this directory, and where.
ELSEWHERE: dict[str, str] = {
    "nice": "test_models_recent_e2e.py — negative log-likelihood",
    "realnvp": "test_models_recent_e2e.py — negative log-likelihood",
    "neural_ode": "test_models_recent_e2e.py",
    "flow_matching": "test_models_recent_e2e.py — .loss",
    "rectified_flow": "test_models_recent_e2e.py — .loss",
    "score_sde": "test_models_recent_e2e.py — .loss",
    "dit": "test_models_recent_e2e.py — .loss",
    "mean_flow": "test_models_recent_e2e.py — .loss",
    "vqvae": "test_models_recent_e2e.py — .loss",
    "stable_diffusion": "test_models_recent_e2e.py — return_loss=True",
    "clip": "test_models_recent_e2e.py — return_loss=True",
    "dreamer": "test_models_recent_e2e.py — model.backward, three optimizers",
    "dreamer_v2": "test_models_recent_e2e.py — model.backward, three optimizers",
    "dreamer_v3": "test_models_recent_e2e.py — model.backward, three optimizers",
    "diamond": "test_models_recent_e2e.py — three losses",
    "genie": "test_models_genie.py — one backward trains all three networks",
    "ijepa": (
        "test_models_ijepa.py — the target encoder is frozen by design, which "
        "the check above refuses"
    ),
    "vjepa": (
        "test_models_vjepa.py — the target encoder is frozen by design, as "
        "I-JEPA's is"
    ),
    "vjepa2": (
        "test_models_vjepa2.py — the target encoder is frozen and nine of "
        "the ten mask-token slots sit out any one step, which the check "
        "above refuses; the probe takes its own step there too"
    ),
    "vjepa2_ac": (
        "test_models_vjepa2.py — the forward takes a video, actions and "
        "states together, which no single-input case here can supply"
    ),
    "ncsn": "test_models_recent_e2e.py",
    "ddpm": "test_models_recent_e2e.py — the caller noises the input",
    "planet": "test_models_planet.py — rewards=",
    "faster_rcnn": "test_models_detection.py — targets=, overfit",
    "mask_rcnn": "test_models_detection.py — targets with masks",
    "fast_rcnn": "test_models_detection.py — proposals=",
    "rcnn": "test_models_detection.py — proposals=",
    "attention_unet": "test_models_segmentation.py",
    "mask2former": "test_models_segmentation.py — overfit",
}

#: Families trained nowhere yet, with what stands in the way.  Empty since
#: the detection and segmentation cases above went in — yolo, detr,
#: efficientdet, unet, fcn and maskformer were the last six.
NOT_YET: dict[str, str] = {}


def test_every_family_takes_a_step_or_says_where() -> None:
    """A family added to the zoo must be trained here, or named with where."""
    families = {entry.family for entry in _REGISTRY.values()}
    here = {_family_of(c) for c in CASES}
    named = set(ELSEWHERE) | set(NOT_YET)
    assert not (here & named), f"trained here and also named: {sorted(here & named)}"
    missing = families - here - named
    assert not missing, (
        f"families trained nowhere and not named: {sorted(missing)} — add a case "
        "here, or say in ELSEWHERE where they are trained"
    )
    unknown = (here | named) - families
    assert not unknown, f"not families in the registry: {sorted(unknown)}"
