#!/usr/bin/env python3
"""tools/check_pretrained_parity.py — the published checkpoint, run against
the implementation it was converted from.

The zoo's checks stack up to here and stop.  ``check_weight_fit`` asks
whether a checkpoint *loads*; the model parity suite asks whether the
architecture matches a reference, but transfers **random** weights to do
it.  Neither looks at the numbers actually published: a conversion that
transposed a kernel, mapped the wrong source, or dropped half a tensor
produces a file that loads into the right shapes and answers wrongly.

This loads the real checkpoint on both sides and compares outputs.  No
labels and no dataset are involved, so it runs anywhere the references
install — it is not an accuracy benchmark and does not pretend to be
one.  It catches conversion, not training.

**The reference is read, not guessed.**  Each entry's ``meta["source"]``
already records where its weights came from, and the two families in
this zoo do not share one: ``resnet_18`` is a reference-vision
checkpoint while ``sk_resnet_18`` is a timm one.  Comparing a model
against the wrong source shows every weight differing and looks exactly
like a defect — which is how this tool's first run was misread.

Text and CLIP checkpoints come from transformers.  Their outputs are
compared field by field — hidden states, logits, span logits, image and
text embeddings — relative to the reference's own scale, because a
language model's logits run to the hundreds and an absolute bound that
suits a classifier would fail them on float32 rounding.  The first of
these compared by hand, ``bert_base``, was 4.14 apart: the embedding
lookup zeroed the trained ``[PAD]`` row.  Models over ``_MAX_PARAMS`` are
reported rather than loaded: both copies at once do not fit a 16 GB
MacBook or a hosted runner.  ``--max-params`` raises the bound on a
machine that can hold them.

A full run otherwise keeps both sides of every checkpoint on disk until
it ends — 81 GB on the hosted runner, leaving 13 of its 94 free.
``--clean-downloads`` runs in a private temporary cache, deletes each
comparison's downloads when done, and never sweeps shared user caches.

Only sources whose reference package is installed can be checked; the
rest are reported as unreachable rather than skipped silently.

Run::

    python -m tools.check_pretrained_parity --limit 5
    python -m tools.check_pretrained_parity --model resnet_18_cls
    python -m tools.check_pretrained_parity --source timm
    python -m tools.check_pretrained_parity --source transformers
    python -m tools.check_pretrained_parity --source transformers/gpt2 --max-params 2e9
    python -m tools.check_pretrained_parity --list

Exit codes
----------
0 — every checkpoint that could be compared agrees.
1 — at least one disagrees beyond tolerance.
2 — nothing could be compared (no reference installed).
"""

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
import importlib
import importlib.metadata
import importlib.util
import hashlib
import json
import math
import os
import platform
import socket
import struct
import subprocess
import sys
import tempfile
import urllib.request
import warnings
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

_TOOL_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

warnings.filterwarnings("ignore")

import numpy as np

import lucid
import lucid.models  # noqa: F401 — populates the registry
import lucid.weights as weights_mod
from lucid.models import DiffusionModelOutput, create_model
from lucid.models._registry import _REGISTRY
from lucid.nn._shadow import shadow_alloc
from lucid.weights import WeightsEnum
from lucid.weights._hub import _cache_dir as _lucid_weight_cache
from lucid.weights._registry import _WEIGHTS_BY_MODEL
from lucid.test._fixtures.ref_framework import ref_module, ref_vision_module, zoo_module

#: Outputs agreeing to this are the same computation in a different order.
_ATOL = 1e-4

#: The same, relative to the reference's largest magnitude — for outputs
#: whose scale is not a classifier's.
_RTOL = 1e-4

#: Below this the comparison is against noise rather than a model.
_MIN_SCALE = 1e-3

#: Larger than this is reported, not loaded.  Two float32 copies of a
#: 774M-parameter model are over 6 GB before any activation.
_MAX_PARAMS = 500_000_000

#: The transformers class that reproduces each Lucid class, and the output
#: fields the two share.  A Lucid class not listed here is reported as
#: unsupported rather than compared against the wrong head.
_HF_HEADS: dict[str, tuple[str, tuple[str, ...]]] = {
    "BERTModel": ("AutoModel", ("last_hidden_state",)),
    "BERTForMaskedLM": ("AutoModelForMaskedLM", ("logits",)),
    "BERTForQuestionAnswering": ("AutoModelForQuestionAnswering", ("start_logits", "end_logits")),
    "BERTForTokenClassification": ("AutoModelForTokenClassification", ("logits",)),
    "GPTModel": ("AutoModel", ("last_hidden_state",)),
    "GPTLMHeadModel": ("AutoModelForCausalLM", ("logits",)),
    "GPT2Model": ("AutoModel", ("last_hidden_state",)),
    "GPT2LMHeadModel": ("AutoModelForCausalLM", ("logits",)),
    "RoFormerModel": ("AutoModel", ("last_hidden_state",)),
    "RoFormerForMaskedLM": ("AutoModelForMaskedLM", ("logits",)),
    "CvTForImageClassification": ("AutoModelForImageClassification", ("logits",)),
    "CLIPModel": ("CLIPModel", ("image_embeds", "text_embeds", "logits_per_image")),
    "CLIPForZeroShotImageClassification": ("CLIPModel", ("image_embeds", "text_embeds", "logits")),
    "MaskFormerForSemanticSegmentation": (
        "MaskFormerForInstanceSegmentation",
        ("class_queries_logits", "masks_queries_logits"),
    ),
    "Mask2FormerForSemanticSegmentation": (
        "Mask2FormerForUniversalSegmentation",
        ("semantic_logits",),
    ),
}

#: CLIP's entries record only that OpenAI's weights were re-hosted by
#: transformers, not where; these are the repositories they came from.
_CLIP_REPOS: dict[str, tuple[str, int]] = {
    "clip_vit_base_16": ("openai/clip-vit-base-patch16", 224),
    "clip_vit_base_32": ("openai/clip-vit-base-patch32", 224),
    "clip_vit_large_14": ("openai/clip-vit-large-patch14", 224),
    "clip_vit_large_14_336": ("openai/clip-vit-large-patch14-336", 336),
}


def _reference_for(source: str) -> tuple[str, str] | None:
    """Split a source string into (package, identifier), or None.

    ``reference_vision/ResNet18_Weights.IMAGENET1K_V1`` names a weights
    enum in the reference vision package; ``timm/resnet18.a1_in1k`` names
    a model in the zoo oracle; ``transformers/google-bert/bert-base-uncased``
    names a repository; CLIP's entries say only that transformers
    re-hosted them. Explicit adapters also cover DDPM, latent diffusion,
    semantic-mask models and the original DETR repository. Other research
    repositories and darknet remain unverified without an oracle adapter.
    """
    if source.startswith("reference_vision/"):
        return ("vision", source.split("/", 1)[1])
    if source.startswith("timm/"):
        return ("timm", source.split("/", 1)[1])
    if source.startswith("transformers/"):
        return ("transformers", source.split("/", 1)[1])
    if source.startswith("openai/CLIPModel"):
        return ("clip", "")
    if source.startswith("diffusers/"):
        return ("diffusers", source.split("/", 1)[1])
    if source.startswith(("facebook/maskformer-", "facebook/mask2former-")):
        return ("transformers", source)
    if source.startswith("facebook/vjepa2-"):
        # ``facebook/vjepa2-vitl-fpc64-256/model.safetensors`` names the
        # file inside the repository the checkpoint was converted from.
        return ("vjepa2", source.rsplit("/", 1)[0])
    if source.startswith("facebook/DiT-"):
        return ("dit", source.split(" ", 1)[0])
    if source == "stable-diffusion-v1-5 (diffusers layout)":
        return ("stable_diffusion", "stable-diffusion-v1-5/stable-diffusion-v1-5")
    if source.startswith("facebookresearch/detr/"):
        return ("detr", source.rsplit("/", 1)[1])
    if source.startswith("darknet/"):
        return ("darknet", source.split("/", 1)[1])
    if source.startswith("eloialonso/diamond "):
        return ("diamond", "eloialonso/diamond")
    return None


def _transformers_module() -> object | None:
    try:
        return importlib.import_module("transformers")
    except ImportError:
        return None


def _build_reference(kind: str, identifier: str) -> object:
    if kind == "timm":
        timm = zoo_module()
        assert timm is not None
        # The whole identifier, tag included. ``resnext101_32x4d`` alone
        # loads timm's default tag, ``fb_ssl_yfcc100m_ft_in1k``, while this
        # zoo's checkpoint came from ``gluon_in1k`` — dropping the tag
        # compared two different checkpoints and reported the gap (5.75,
        # top-1 differing) as a conversion defect. seresnet50 the same:
        # ``a1_in1k`` by default, ``ra2_in1k`` here.
        return timm.create_model(identifier, pretrained=True).eval()

    vision = ref_vision_module()
    assert vision is not None
    enum_name, tag = identifier.split(".", 1)
    # ``ResNet18_Weights`` → ``resnet18``: the enum is named for the model
    # it belongs to, which is the only link between the two.  Keep the
    # underscores — ``ConvNeXt_Base_Weights`` is ``convnext_base`` and
    # ``VGG11_BN_Weights`` is ``vgg11_bn``; stripping them worked for
    # resnet and densenet by luck and broke four convnext entries into
    # what read like conversion defects.
    # Classifiers live on ``models``; segmenters and detectors live one
    # level down, and the source string does not say which. Look rather
    # than assume — the flat lookup reported six of them as conversion
    # defects when the enum was simply somewhere else.
    namespaces = [vision.models]
    for extra in ("segmentation", "detection"):
        try:
            namespaces.append(importlib.import_module(f"{vision.__name__}.models.{extra}"))
        except ImportError:  # pragma: no cover - depends on the install
            continue

    factory_name = enum_name.removesuffix("_Weights").lower()
    for namespace in namespaces:
        weights_enum = getattr(namespace, enum_name, None)
        factory = getattr(namespace, factory_name, None)
        if weights_enum is not None and factory is not None:
            return factory(weights=getattr(weights_enum, tag)).eval()

    raise AttributeError(
        f"the reference vision package has no {enum_name} paired with "
        f"{factory_name!r} in models, models.segmentation or "
        f"models.detection — the rule this derives one name from the "
        f"other with does not hold here"
    )


def _compare_arrays(wanted: np.ndarray, got: np.ndarray) -> dict[str, object]:
    """Reject invalid comparisons before a NaN can make ``diff > tol`` false."""
    if got.shape != wanted.shape:
        return {"error": f"shape {got.shape} against {wanted.shape}"}
    if wanted.size == 0:
        return {"degenerate": "the output is empty"}
    if not np.isfinite(wanted).all() or not np.isfinite(got).all():
        return {"error": "the reference or Lucid output contains NaN or infinity"}
    scale = float(np.abs(wanted).max())
    if scale < _MIN_SCALE:
        return {"degenerate": f"the reference output reaches only {scale:.2g}"}
    return {
        "max_diff": float(np.abs(got - wanted).max()),
        "scale": scale,
        "top1_agrees": bool((got.argmax(-1) == wanted.argmax(-1)).all()),
    }


def _compare(
    model_name: str,
    source: str,
    shape: tuple[int, ...],
    params: int | None,
    max_params: float = _MAX_PARAMS,
) -> dict[str, object]:
    resolved = _reference_for(source)
    if resolved is None:
        return {"unreachable": f"no loader for {source.split('/')[0]!r}"}
    if params is not None and params > max_params:
        return {"unreachable": f"{params / 1e6:.0f}M parameters exceed --max-params={max_params:g}"}
    kind, identifier = resolved
    if kind == "diffusers":
        return _compare_diffusers(model_name, identifier)
    if kind in ("dit", "stable_diffusion"):
        return _compare_latent_diffusion(model_name, kind, identifier)
    if kind == "detr":
        return _compare_detr(model_name, identifier, shape)
    if kind == "darknet":
        return _compare_darknet(model_name, shape)
    if kind == "diamond":
        return _compare_diamond(model_name, identifier)
    if kind == "vjepa2":
        return _compare_vjepa2(model_name, identifier)
    if kind in ("transformers", "clip"):
        return _compare_transformers(model_name, kind, identifier, params, max_params)
    if (kind == "timm" and zoo_module() is None) or (
        kind == "vision" and ref_vision_module() is None
    ):
        return {"unreachable": f"the {kind} reference is not installed"}

    ref = ref_module()
    if ref is None:
        return {"unreachable": "the reference framework is not installed"}

    try:
        ours = create_model(model_name, pretrained=True).eval()
        theirs = _build_reference(kind, identifier)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"[:200]}

    x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    if kind == "vision" and identifier.startswith(("FasterRCNN_", "MaskRCNN_")):
        return _compare_detection_components(ours, theirs, x)
    try:
        with ref.no_grad(), lucid.no_grad():
            reference_out = theirs(ref.from_numpy(x.copy()))
            answer = ours(lucid.from_numpy(x.copy()))
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"[:200]}

    # Segmenters and detectors answer with a mapping, and the two sides
    # do not agree on its keys — a detector's output is a list of boxes
    # whose length depends on what it found, which is not a thing to
    # compare elementwise against another implementation's list. Say so
    # rather than reaching for a ``.numpy()`` that is not there.
    if (
        isinstance(reference_out, dict)
        and "out" in reference_out
        and isinstance(getattr(answer, "logits", None), lucid.Tensor)
    ):
        pairs = [
            ("semantic_logits", reference_out["out"].numpy(), getattr(answer, "logits").numpy())
        ]
        if "aux" in reference_out:
            auxiliary = getattr(answer, "aux_logits", None)
            if not isinstance(auxiliary, lucid.Tensor):
                return {"error": "reference auxiliary logits are missing from Lucid output"}
            pairs.append(("auxiliary_scores", reference_out["aux"].numpy(), auxiliary.numpy()))
        return _compare_fields(pairs)
    if not hasattr(reference_out, "numpy"):
        return {
            "unsupported": (
                f"the reference answers with " f"{type(reference_out).__name__}, not a tensor"
            )
        }

    wanted = reference_out.numpy()
    output = answer if isinstance(answer, lucid.Tensor) else getattr(answer, "logits", None)
    if not isinstance(output, lucid.Tensor):
        return {"unsupported": f"Lucid answers with {type(answer).__name__}, not tensor logits"}
    got = output.numpy()

    return _compare_arrays(wanted, got)


def _compare_detection_components(
    ours: object, theirs: object, values: np.ndarray
) -> dict[str, object]:
    """Exercise trained dense stages with identical proposals, before NMS."""
    from lucid.models._utils._detection import multiscale_roi_align

    ref = ref_module()
    if ref is None:
        return {"unreachable": "the reference framework is not installed"}
    try:
        cfg = getattr(ours, "config")
        box = np.array([[4, 8, 60, 80], [20, 16, 210, 208]], dtype=np.float32)
        pairs = []
        with ref.no_grad(), lucid.no_grad():
            reference_features = getattr(theirs, "backbone")(ref.from_numpy(values.copy()))
            features = getattr(ours, "backbone")(lucid.from_numpy(values.copy()))
            for index, (wanted, got) in enumerate(
                zip(reference_features.values(), features, strict=True)
            ):
                pairs.append((f"fpn_{index}", wanted.numpy(), got.numpy()))
            reference_rpn = getattr(theirs, "rpn").head(list(reference_features.values()))
            actual_rpn = getattr(ours, "rpn").head(features)
            for label, want_group, got_group in zip(
                ("rpn_objectness", "rpn_box_deltas"), reference_rpn, actual_rpn, strict=True
            ):
                for index, (wanted, got) in enumerate(zip(want_group, got_group, strict=True)):
                    pairs.append((f"{label}_{index}", wanted.numpy(), got.numpy()))
            reference_heads = getattr(theirs, "roi_heads")
            heads = getattr(ours, "roi_heads")
            size = getattr(cfg, "roi_det_size", getattr(cfg, "roi_size", 7))
            actual_pool = multiscale_roi_align(
                features[:4],
                [lucid.from_numpy(box.copy())],
                size,
                [0.25, 0.125, 0.0625, 0.03125],
                cfg.roi_sampling_ratio,
                cfg.canonical_scale,
                cfg.canonical_level,
            )
            reference_pool = reference_heads.box_roi_pool(
                reference_features, [ref.from_numpy(box.copy())], [tuple(values.shape[-2:])]
            )
            actual_boxes = heads(actual_pool)
            reference_boxes = reference_heads.box_predictor(
                reference_heads.box_head(reference_pool)
            )
            for name, wanted, got in zip(
                ("box_class_logits", "box_deltas"), reference_boxes, actual_boxes, strict=True
            ):
                pairs.append((name, wanted.numpy(), got.numpy()))
            if hasattr(heads, "mask_head"):
                actual_masks = multiscale_roi_align(
                    features[:4],
                    [lucid.from_numpy(box.copy())],
                    cfg.roi_mask_size,
                    [0.25, 0.125, 0.0625, 0.03125],
                    cfg.roi_sampling_ratio,
                    cfg.canonical_scale,
                    cfg.canonical_level,
                )
                reference_masks = reference_heads.mask_roi_pool(
                    reference_features, [ref.from_numpy(box.copy())], [tuple(values.shape[-2:])]
                )
                wanted = reference_heads.mask_predictor(reference_heads.mask_head(reference_masks))
                got = heads.predict_masks(actual_masks)
                pairs.append(("instance_mask_scores", wanted.numpy(), got.numpy()))
        result = _compare_fields(pairs)
        result["scope"] = (
            "backbone/FPN, RPN, fixed-proposal RoI box/mask heads; no image transform, proposal selection, NMS or detection metric"
        )
        return result
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"[:200]}


def _compare_transformers(
    model_name: str, kind: str, identifier: str, params: int | None, max_params: float = _MAX_PARAMS
) -> dict[str, object]:
    """One checkpoint against the transformers model it was converted from.

    Token ids are drawn at random below the vocabulary size, pixels from a
    unit normal; no tokenizer is involved, since the question is whether
    the same ids reach the same numbers, not what the text means.
    """
    transformers = _transformers_module()
    ref = ref_module()
    if transformers is None or ref is None:
        return {"unreachable": "transformers or the reference framework is not installed"}

    model_class = _REGISTRY[model_name].model_class
    if model_class is None:
        return {"unsupported": f"no model class recorded for {model_name}"}
    cls_name = model_class.__name__
    head = _HF_HEADS.get(cls_name)
    if head is None:
        return {"unsupported": f"no transformers head mapped for {cls_name}"}
    auto_name, fields = head
    repo, side = identifier, 224
    if kind == "clip":
        clip_name = model_name.removesuffix("_zero_shot")
        if clip_name not in _CLIP_REPOS:
            return {"unsupported": f"no repository recorded for {model_name}"}
        repo, side = _CLIP_REPOS[clip_name]
    if params is not None and params > max_params:
        return {
            "unreachable": (
                f"{params / 1e6:.0f}M parameters — too large to hold both "
                f"copies here (limit {max_params / 1e6:.0f}M; see --max-params)"
            )
        }

    try:
        ours = create_model(model_name, pretrained=True).eval()
        theirs = getattr(transformers, auto_name).from_pretrained(repo).eval()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"[:200]}

    rng = np.random.default_rng(0)
    ours_args: list[lucid.Tensor] = []
    theirs_kwargs: dict[str, object] = {}
    vision_only = cls_name in (
        "CvTForImageClassification",
        "MaskFormerForSemanticSegmentation",
        "Mask2FormerForSemanticSegmentation",
    )
    if vision_only or kind == "clip":
        pixels = rng.standard_normal((1, 3, side, side)).astype(np.float32)
        ours_args.append(lucid.from_numpy(pixels.copy()))
        theirs_kwargs["pixel_values"] = ref.from_numpy(pixels.copy())
    if not vision_only:
        vocab = int(getattr(ours.config, "vocab_size"))
        # CLIP's text tower takes exactly its context length and pools at
        # the [EOS] token, found as the largest id. Ending every sequence
        # on the top id pins that position on both sides, whichever way
        # the reference locates it.
        length = int(getattr(ours.config, "context_length", 16))
        # Multiple candidates make zero-shot ranking a nontrivial comparison.
        prompts = 3 if kind == "clip" else 1
        ids = rng.integers(1, vocab - 1, size=(prompts, length)).astype(np.int64)
        if kind == "clip":
            ids[:, -1] = vocab - 1
        ours_args.append(lucid.from_numpy(ids.copy()).long())
        theirs_kwargs["input_ids"] = ref.from_numpy(ids.copy())

    try:
        with ref.no_grad(), lucid.no_grad():
            reference_out = theirs(**theirs_kwargs)
            answer = ours(*ours_args)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"[:200]}

    pairs = []
    for field in fields:
        if field == "semantic_logits":
            # The published semantic postprocessor interpolates masks BEFORE
            # sigmoid and weighted query reduction, not the resulting scores.
            masks = ref.nn.functional.interpolate(
                reference_out.masks_queries_logits,
                size=(side, side),
                mode="bilinear",
                align_corners=False,
            ).sigmoid()
            classes = reference_out.class_queries_logits.softmax(-1)[..., :-1]
            wanted = (
                (classes.transpose(1, 2) @ masks.flatten(2))
                .reshape(1, classes.shape[-1], side, side)
                .numpy()
            )
            got = getattr(answer, "logits").numpy()
        else:
            reference_field = "logits_per_image" if kind == "clip" and field == "logits" else field
            wanted = getattr(reference_out, reference_field).numpy()
            got = getattr(answer, field).numpy()
        pairs.append((field, wanted, got))
    return _compare_fields(pairs)


def _compare_vjepa2(model_name: str, repo: str) -> dict[str, object]:
    """A published V-JEPA 2 backbone against the repository it came from.

    Two differences from the shared image path, both forced by what the
    model takes.  The input is a clip, ``(B, T, C, H, W)`` on both sides.
    And it is a two-frame one rather than the configured sixty-four: the
    rotary geometry is computed from token positions rather than read
    from a table, so a short clip exercises the same weights, while a
    full one would put 8192 tokens through a twenty-four-block encoder
    twice over.

    What is compared is Lucid's *target* encoder against the reference's
    only one — the conversion mirrors the context encoder into it, so a
    mirroring that silently dropped a tensor shows up here.

    The predictor is fed by hand rather than through
    ``VJEPA2Model.forward``, because the two sides mean different things
    by "the context".  Lucid's masked path runs the encoder *on the
    context subset*, which is the pretraining contract — the context
    encoder must not see the region it is asked to predict.  The
    reference port runs one encoder over the whole clip and selects the
    context positions afterwards, which is the inference contract.  Both
    are right; comparing one against the other measures the difference
    between them (0.36 relative on ViT-L) rather than the conversion.
    """
    transformers = _transformers_module()
    ref = ref_module()
    if transformers is None or ref is None:
        return {"unreachable": "transformers or the reference framework is not installed"}
    try:
        ours = create_model(model_name, pretrained=True, num_frames=2).eval()
        theirs = getattr(transformers, "AutoModel").from_pretrained(repo).eval()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"[:200]}

    side = int(getattr(ours.config, "image_size"))
    video = np.random.default_rng(0).standard_normal((1, 2, 3, side, side)).astype(np.float32)
    tokens = (side // int(getattr(ours.config, "patch_size"))) ** 2
    half = tokens // 2
    context = np.arange(half, dtype=np.int64).reshape(1, -1)
    target = np.arange(half, tokens, dtype=np.int64).reshape(1, -1)
    try:
        with ref.no_grad(), lucid.no_grad():
            reference = theirs(
                pixel_values_videos=ref.from_numpy(video.copy()),
                context_mask=[ref.from_numpy(context.copy())],
                target_mask=[ref.from_numpy(target.copy())],
            )
            encoded = getattr(ours(lucid.from_numpy(video.copy())), "tokens")
            # The reference's context: the whole clip encoded, then the
            # context positions taken out of it.
            indices = lucid.from_numpy(context.copy())
            gather = indices.unsqueeze(-1) + lucid.zeros(
                1, half, int(encoded.shape[2]), dtype=indices.dtype, device=encoded.device
            )
            predicted = getattr(
                ours.predictor(
                    lucid.gather(encoded, 1, gather), indices, lucid.from_numpy(target.copy())
                ),
                "numpy",
            )()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"[:200]}
    return _compare_fields(
        [
            ("encoder", reference.last_hidden_state.numpy(), encoded.numpy()),
            ("predictor", reference.predictor_output.last_hidden_state.numpy(), predicted),
        ]
    )


def _compare_fields(pairs: list[tuple[str, np.ndarray, np.ndarray]]) -> dict[str, object]:
    if not pairs:
        return {"degenerate": "no output fields were compared"}
    worst, scale, agrees = 0.0, 0.0, True
    for field, wanted, got in pairs:
        metrics = _compare_arrays(wanted, got)
        for problem in ("error", "degenerate"):
            if problem in metrics:
                return {problem: f"{field}: {metrics[problem]}"}
        field_scale = float(metrics["scale"])  # type: ignore[arg-type]
        worst = max(worst, float(metrics["max_diff"]) / field_scale)  # type: ignore[arg-type]
        scale = max(scale, field_scale)
        # Spatial mask values are not class scores. Semantic NCHW scores
        # classify along C, not the final (width) dimension; comparing the
        # brightest pixel in each row can reject numerically matching masks.
        if field == "semantic_logits":
            agrees = agrees and bool((got.argmax(1) == wanted.argmax(1)).all())
        elif "logits" in field and field != "masks_queries_logits":
            agrees = agrees and bool((got.argmax(-1) == wanted.argmax(-1)).all())

    return {
        "max_diff": worst,
        "scale": scale,
        "top1_agrees": agrees,
        "tolerance": _RTOL,
        "relative": True,
        "fields": [field for field, _, _ in pairs],
    }


def _compare_diffusers(model_name: str, identifier: str) -> dict[str, object]:
    """Compare a fixed DDPM denoising step, not stochastic image generation."""
    cls = _REGISTRY[model_name].model_class
    if cls is None or cls.__name__ not in ("DDPMModel", "DDPMForImageGeneration"):
        return {"unsupported": f"no diffusion adapter mapped for {model_name}"}
    ref = ref_module()
    try:
        diffusers = importlib.import_module("diffusers")
    except ImportError:
        return {"unreachable": "diffusers is not installed"}
    if ref is None:
        return {"unreachable": "the reference framework is not installed"}
    try:
        ours = create_model(model_name, pretrained=True).eval()
        theirs = diffusers.UNet2DModel.from_pretrained(identifier).eval()
        size = getattr(ours.config, "sample_size")
        height, width = (size, size) if isinstance(size, int) else size
        values = (
            np.random.default_rng(0)
            .standard_normal((1, getattr(ours.config, "in_channels"), height, width))
            .astype(np.float32)
        )
        with ref.no_grad(), lucid.no_grad():
            wanted = theirs(ref.from_numpy(values.copy()), 500).sample.numpy()
            answer = ours(lucid.from_numpy(values.copy()), lucid.tensor([500], dtype=lucid.int64))
            got = getattr(answer, "sample").numpy()
        result = _compare_fields([("denoising_sample_t500", wanted, got)])
        result["scope"] = "single deterministic denoising step; no sampling or image-quality claim"
        return result
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"[:200]}


def _compare_latent_diffusion(model_name: str, kind: str, identifier: str) -> dict[str, object]:
    """Compare published components on deterministic inputs, without a sampler."""
    ref = ref_module()
    try:
        diffusers = importlib.import_module("diffusers")
    except ImportError:
        return {"unreachable": "diffusers is not installed"}
    if ref is None:
        return {"unreachable": "the reference framework is not installed"}
    try:
        ours = create_model(model_name, pretrained=True).eval()
        rng = np.random.default_rng(0)
        pairs = []
        export_comparison: dict[str, object] = {}
        if kind == "dit":
            theirs = diffusers.Transformer2DModel.from_pretrained(
                identifier, subfolder="transformer"
            ).eval()
            denoiser = getattr(ours, "dit", ours)
            size = int(getattr(denoiser.config, "sample_size"))
            values = rng.standard_normal((1, 4, size, size)).astype(np.float32)
            steps = np.array([500], dtype=np.int64)
            labels = np.array([42], dtype=np.int64)
            with ref.no_grad(), lucid.no_grad():
                wanted = theirs(
                    ref.from_numpy(values.copy()),
                    timestep=ref.from_numpy(steps),
                    class_labels=ref.from_numpy(labels),
                ).sample.numpy()
                answer = denoiser(
                    lucid.from_numpy(values.copy()),
                    lucid.from_numpy(steps),
                    lucid.from_numpy(labels),
                )
                if not isinstance(answer, DiffusionModelOutput):
                    return {"error": "DiT denoiser did not return a DiffusionModelOutput"}
                got = answer.sample.numpy()
                export_comparison = _compare_fields([("noise_and_variance", wanted, got)])
                _dit_original_frequencies(theirs)
                wanted = theirs(
                    ref.from_numpy(values.copy()),
                    timestep=ref.from_numpy(steps),
                    class_labels=ref.from_numpy(labels),
                ).sample.numpy()
            pairs.append(("denoising_noise_and_variance_t500_class42", wanted, got))
        else:
            ours = getattr(ours, "stable_diffusion", ours)
            theirs = diffusers.UNet2DConditionModel.from_pretrained(
                identifier, subfolder="unet"
            ).eval()
            values = rng.standard_normal((1, 4, 16, 16)).astype(np.float32)
            context = rng.standard_normal((1, 77, 768)).astype(np.float32)
            with ref.no_grad(), lucid.no_grad():
                wanted = theirs(
                    ref.from_numpy(values.copy()),
                    500,
                    encoder_hidden_states=ref.from_numpy(context.copy()),
                ).sample.numpy()
                got = getattr(ours, "unet")(
                    lucid.from_numpy(values.copy()),
                    lucid.tensor([500.0]),
                    lucid.from_numpy(context.copy()),
                ).numpy()
            pairs.append(("conditional_denoising_t500_latent16", wanted, got))
            del theirs
            theirs_vae = diffusers.AutoencoderKL.from_pretrained(identifier, subfolder="vae").eval()
            images = rng.standard_normal((1, 3, 128, 128)).astype(np.float32)
            ours_vae = getattr(ours, "vae")
            with ref.no_grad(), lucid.no_grad():
                pairs.append(
                    (
                        "vae_posterior_mean_image128",
                        theirs_vae.encode(ref.from_numpy(images.copy())).latent_dist.mean.numpy(),
                        ours_vae.encode(lucid.from_numpy(images.copy())).mean.numpy(),
                    )
                )
                pairs.append(
                    (
                        "vae_decode_latent16",
                        theirs_vae.decode(ref.from_numpy(values.copy())).sample.numpy(),
                        ours_vae.decode(lucid.from_numpy(values.copy())).numpy(),
                    )
                )
        result = _compare_fields(pairs)
        if kind == "dit":
            result["unadjusted_export_comparison"] = export_comparison
            result["reference_adjustment"] = (
                "original DiT timestep denominator half, not the exported layout's half-1"
            )
        result["scope"] = (
            "deterministic component outputs; no text encoder, sampling, dataset or image-quality claim"
        )
        return result
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"[:200]}


def _dit_original_frequencies(model: object) -> None:
    """Restore the original implementation's timestep equation in the oracle.

    The retro-dit note records that original training used half, whereas
    the exported layout embeds the same checkpoint with half-1. Preserve
    the unadjusted discrepancy in the report; never change Lucid to fit it.
    """
    blocks = getattr(model, "transformer_blocks")
    if not blocks:
        raise ValueError("the DiT reference has no transformer blocks")
    for block in blocks:
        projection = block.norm1.emb.time_proj
        if projection.downscale_freq_shift not in (0, 1):
            raise ValueError("unexpected DiT reference frequency convention")
        projection.downscale_freq_shift = 0


def _compare_detr(model_name: str, identifier: str, shape: tuple[int, ...]) -> dict[str, object]:
    """Use the converter's original implementation, pinned to a reviewed revision."""
    ref = ref_module()
    if ref is None or ref_vision_module() is None:
        return {"unreachable": "the reference framework and vision package are required"}
    if identifier not in ("detr_resnet50", "detr_resnet101"):
        return {"unsupported": f"unknown DETR release {identifier}"}
    revision = "29901c51d7fe8712168b8d0d64351170bc0f83e0"
    try:
        ours = create_model(model_name, pretrained=True).eval()
        theirs = ref.hub.load(
            f"facebookresearch/detr:{revision}", identifier, pretrained=True, trust_repo=True
        ).eval()
        values = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
        with ref.no_grad(), lucid.no_grad():
            wanted = theirs(ref.from_numpy(values.copy()))
            got = ours(lucid.from_numpy(values.copy()))
        result = _compare_fields(
            [
                ("query_logits", wanted["pred_logits"].numpy(), getattr(got, "logits").numpy()),
                ("query_boxes", wanted["pred_boxes"].numpy(), getattr(got, "pred_boxes").numpy()),
            ]
        )
        result["reference_revision"] = revision
        result["scope"] = "raw query logits and normalized boxes; no detection metric"
        return result
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"[:200]}


@contextmanager
def _diamond_reference(revision: str) -> Iterator[ModuleType]:
    """Load unchanged, pinned inference files without the training application.

    The original package initializer imports environment/training dependencies.
    A private package namespace lets the two pure inference modules retain
    their relative imports without importing or replacing those dependencies.
    """
    namespace = "_lucid_parity_diamond_reference"
    if any(name == namespace or name.startswith(namespace + ".") for name in sys.modules):
        raise RuntimeError("the DIAMOND oracle namespace is already active")
    with tempfile.TemporaryDirectory(prefix="lucid-diamond-source-") as directory:
        root = Path(directory)
        try:
            for suffix in ("", ".diffusion"):
                package = ModuleType(namespace + suffix)
                package.__path__ = [str(root)]
                sys.modules[package.__name__] = package
            for relative, suffix in (
                ("blocks.py", ".blocks"),
                ("diffusion/inner_model.py", ".diffusion.inner_model"),
            ):
                url = f"https://raw.githubusercontent.com/eloialonso/diamond/{revision}/src/models/{relative}"
                path = root / Path(relative).name
                with urllib.request.urlopen(url, timeout=30) as response:
                    path.write_bytes(response.read())
                spec = importlib.util.spec_from_file_location(namespace + suffix, path)
                if spec is None or spec.loader is None:
                    raise RuntimeError(f"cannot load the original {relative}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[module.__name__] = module
                spec.loader.exec_module(module)
            yield sys.modules[namespace + ".diffusion.inner_model"]
        finally:
            for name in list(sys.modules):
                if name == namespace or name.startswith(namespace + "."):
                    del sys.modules[name]


def _compare_diamond(model_name: str, identifier: str) -> dict[str, object]:
    """Verify trained denoiser components, not rollout quality or agent returns."""
    ref = ref_module()
    if ref is None:
        return {"unreachable": "the reference framework is not installed"}
    csgo = model_name == "diamond_csgo"
    revision = (
        "851cefb497733d27f1b85c804104638765860fca"
        if csgo
        else "5bcd1599755b4f2fae8e5e079e02f0728e174965"
    )
    try:
        from lucid.models.generative.diamond import DIAMONDConfig

        hub = importlib.import_module("huggingface_hub")
        member = _factory_default(model_name)
        filename = (
            "csgo/model/csgo.pt" if csgo else f"atari_100k/models/{member.value.meta['game']}.pt"
        )
        state = ref.load(
            hub.hf_hub_download(identifier, filename), map_location="cpu", weights_only=True
        )
        model = create_model(model_name, pretrained=True).eval()
        ours = getattr(model, "diamond", model)
        cfg = ours.config
        if not isinstance(cfg, DIAMONDConfig):
            return {"error": "unexpected DIAMOND configuration"}
        rng = np.random.default_rng(0)
        pairs = []
        with _diamond_reference(revision) as source:
            components = ["denoiser", "upsampler"] if csgo else ["denoiser"]
            for component in components:
                upsampler = component == "upsampler"
                depths = cfg.upsampler_layers if upsampler else cfg.unet_layers
                channels = cfg.upsampler_channels if upsampler else cfg.unet_channels
                attention = cfg.upsampler_attn_depths if upsampler else cfg.attn_depths
                if depths is None or channels is None or attention is None:
                    return {"error": f"missing {component} configuration"}
                kwargs = dict(
                    img_channels=cfg.in_channels,
                    num_steps_conditioning=1 if upsampler else cfg.conditioning_frames,
                    cond_channels=cfg.cond_dim,
                    depths=list(depths),
                    channels=list(channels),
                    attn_depths=list(attention),
                    num_actions=cfg.num_actions,
                )
                if csgo:
                    kwargs["is_upsampler"] = upsampler
                theirs = source.InnerModel(source.InnerModelConfig(**kwargs)).eval()
                prefix = component + ".inner_model."
                theirs.load_state_dict(
                    {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)},
                    strict=True,
                )
                height, width = cfg.frame_shape
                if upsampler:
                    height, width = height * cfg.upsampling_factor, width * cfg.upsampling_factor
                noisy = rng.standard_normal((1, cfg.in_channels, height, width)).astype(np.float32)
                history = rng.standard_normal(
                    (
                        1,
                        cfg.in_channels * (2 if upsampler else cfg.conditioning_frames),
                        height,
                        width,
                    )
                ).astype(np.float32)
                noise = np.array([0.25], dtype=np.float32)
                cond_noise = np.array([-0.5], dtype=np.float32)
                actions = rng.integers(
                    0, cfg.num_actions, size=(1, cfg.conditioning_frames), dtype=np.int64
                )
                network = getattr(ours, component)
                with ref.no_grad(), lucid.no_grad():
                    if csgo:
                        wanted = theirs(
                            ref.from_numpy(noisy),
                            ref.from_numpy(noise),
                            ref.from_numpy(cond_noise),
                            ref.from_numpy(history),
                            None if upsampler else ref.from_numpy(actions),
                        )
                    else:
                        wanted = theirs(
                            ref.from_numpy(noisy),
                            ref.from_numpy(noise),
                            ref.from_numpy(history),
                            ref.from_numpy(actions),
                        )
                    if upsampler:
                        cond = network.conditioning(
                            lucid.from_numpy(noise), lucid.from_numpy(cond_noise)
                        )
                    else:
                        cond = network.conditioning(
                            lucid.from_numpy(noise),
                            lucid.from_numpy(actions),
                            lucid.from_numpy(cond_noise) if csgo else None,
                        )
                    got = network(lucid.from_numpy(np.concatenate((history, noisy), axis=1)), cond)
                    pairs.append((component, wanted.numpy().copy(), got.numpy().copy()))
                del theirs, got, wanted
        result = _compare_fields(pairs)
        result["reference_revision"] = revision
        result["scope"] = (
            "trained raw denoiser components; no EDM preconditioning, agent heads, rollout, sampling or game-score claim"
        )
        return result
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"[:200]}


def _darknet_oracle_blob(path: Path, convolutions: list[tuple[int, int, bool]]) -> np.ndarray:
    """Adapt only the oracle importer's BN epsilon, in an owned memory copy.

    OpenCV converts variance to an affine scale during import, so setParam
    afterwards is too late. sqrt((variance + 9e-6) + 1e-6) reproduces the
    original accelerator's sqrt(variance + 1e-5); no published file is edited.
    """
    blob = bytearray(path.read_bytes())
    major, minor, _ = struct.unpack_from("<iii", blob)
    header = 20 if major * 10 + minor >= 2 else 16
    payload = np.frombuffer(blob, dtype="<f4", offset=header)
    pos = 0
    for channels, weights, normalized in convolutions:
        if normalized:
            payload[pos + 3 * channels : pos + 4 * channels] += np.float32(1e-5 - 1e-6)
        pos += channels * (4 if normalized else 1) + weights
    if pos != payload.size:
        raise ValueError("darknet oracle weight census does not consume the full payload")
    return np.frombuffer(blob, dtype=np.uint8)


def _compare_darknet(model_name: str, shape: tuple[int, ...]) -> dict[str, object]:
    """Run original cfg/weights through an independent CPU DNN implementation.

    Compare raw prediction convolutions before the two libraries' different
    decoding/NMS conventions. This is not an end-to-end detection metric.
    """
    try:
        cv2 = importlib.import_module("cv2")
    except ImportError:
        return {"unreachable": "the optional OpenCV DNN oracle is not installed"}
    if not callable(getattr(cv2.dnn, "readNetFromDarknet", None)):
        return {
            "unreachable": "this OpenCV build has no darknet reader; use the optional 4.x oracle"
        }
    converter = importlib.import_module("tools.convert_weights.yolo")

    if model_name not in converter._SOURCES:
        return {"unsupported": f"unknown darknet release {model_name}"}
    spec = converter._SOURCES[model_name]
    if model_name == "yolo_v4":
        repo, revision = "AlexeyAB/darknet", "59596d7880f6504768df41d6daa586f5cb2b932f"
    else:
        repo, revision = "pjreddie/darknet", "f6afaabcdf85f77e7aff2ec55c020c0e297c77f9"
    cfg_name = Path(spec.url).stem + ".cfg"
    try:
        weights_path = converter._fetch(spec.url, spec.sha256)
        ours = create_model(model_name, pretrained=True).eval()
        order = converter._darknet_order(ours, model_name)
        modules = dict(ours.named_modules())
        convolution_shapes = [
            (
                int(getattr(modules[name], "weight").shape[0]),
                int(getattr(modules[name], "weight").numel()),
                norm is not None,
            )
            for name, norm in order
        ]
        oracle_blob = _darknet_oracle_blob(weights_path, convolution_shapes)
        with tempfile.TemporaryDirectory(prefix="lucid-darknet-cfg-") as directory:
            cfg_path = Path(directory) / cfg_name
            url = f"https://raw.githubusercontent.com/{repo}/{revision}/cfg/{cfg_name}"
            with urllib.request.urlopen(url, timeout=30) as response:
                cfg_path.write_bytes(response.read())
            theirs = cv2.dnn.readNetFromDarknet(
                np.frombuffer(cfg_path.read_bytes(), dtype=np.uint8), oracle_blob
            )
        del oracle_blob
        theirs.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        theirs.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        theirs.enableFusion(False)
        conv_names = [
            name
            for name in theirs.getLayerNames()
            if theirs.getLayer(theirs.getLayerId(name)).type == "Convolution"
        ]
        if len(conv_names) != len(order):
            return {"error": f"convolution census mismatch: {len(conv_names)} vs {len(order)}"}
        prediction_names = [
            (conv_names[i], name) for i, (name, norm) in enumerate(order) if norm is None
        ]
        captured = {}
        handles = []
        for _, name in prediction_names:

            def capture(
                _module: object, _args: tuple[object, ...], output: object, *, key: str = name
            ) -> None:
                if not isinstance(output, lucid.Tensor):
                    raise TypeError(f"prediction head {key} did not return a Tensor")
                captured[key] = output.numpy().copy()

            handles.append(modules[name].register_forward_hook(capture))
        values = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
        try:
            with lucid.no_grad():
                ours(lucid.from_numpy(values.copy()))
            theirs.setInput(values.copy())
            wanted = theirs.forward([name for name, _ in prediction_names])
            result = _compare_fields(
                [
                    (name, reference, captured[name])
                    for (_, name), reference in zip(prediction_names, wanted, strict=True)
                ]
            )
        finally:
            for handle in handles:
                handle.remove()
        result["reference_revision"] = revision
        result["reference_runtime"] = f"OpenCV {cv2.__version__} CPU DNN, fusion disabled"
        result["reference_normalization"] = "original accelerator equation: sqrt(variance + 1e-5)"
        result["scope"] = (
            "raw prediction heads from original cfg/weights; no decoding, NMS or detection metric"
        )
        return result
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"[:200]}


def _download_roots() -> list[Path]:
    """Every directory a comparison downloads into.

    Lucid's own weight cache; the reference framework's hub directory,
    where the reference vision package and older zoo-oracle checkpoints
    land; and the Hugging Face cache, which holds transformers, CLIP and
    most zoo-oracle checkpoints.
    """
    roots = [_lucid_weight_cache(), _lucid_weight_cache().parent / "darknet-src"]
    ref = ref_module()
    if ref is not None:
        roots.append(Path(ref.hub.get_dir()))
    try:
        from huggingface_hub import constants
    except ImportError:  # pragma: no cover - depends on the install
        pass
    else:
        roots.append(Path(constants.HF_HUB_CACHE))
    return roots


def _files_under(roots: list[Path]) -> set[Path]:
    found: set[Path] = set()
    for root in roots:
        if root.is_dir():
            found.update(p for p in root.rglob("*") if p.is_file() or p.is_symlink())
            # Directories too, so a directory that appeared can go as well.
            found.update(p for p in root.rglob("*") if p.is_dir() and not p.is_symlink())
    return found


def _remove_new(before: set[Path], roots: list[Path]) -> int:
    """Delete what appeared under ``roots`` since ``before``; bytes freed.

    Called only inside the owned temporary cache, never a shared cache
    where another process could have created the newly observed files.
    """
    freed = 0
    new = _files_under(roots) - before
    for path in new:
        if path.is_dir() and not path.is_symlink():
            continue
        try:
            if not path.is_symlink():
                freed += path.stat().st_size
            path.unlink()
        except FileNotFoundError:
            continue
    # Then the directories that appeared, deepest first.  Left behind empty,
    # a hub repository's directory told the next load "cached" and it read
    # a hubconf.py that had just been deleted — the second DETR checkpoint
    # failed with FileNotFoundError every week.
    for path in sorted((p for p in new if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        try:
            path.rmdir()
        except OSError:
            continue
    return freed


@contextmanager
def _reference_io() -> Iterator[None]:
    """Read existing oracles only; bound sockets with no explicit timeout.

    Some reference loaders otherwise start background format-conversion jobs
    that can contact a publishing service and keep the interpreter alive.
    Explicit SDK timeouts are unaffected; stricter caller defaults are retained.
    """
    key = "DISABLE_SAFETENSORS_CONVERSION"
    previous = os.environ.get(key)
    timeout = socket.getdefaulttimeout()
    os.environ[key] = "1"
    socket.setdefaulttimeout(60.0 if timeout is None else min(timeout, 60.0))
    try:
        yield
    finally:
        socket.setdefaulttimeout(timeout)
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def _isolated_run(arguments: list[str]) -> int:
    """Own the entire cleanup boundary; never sweep a shared user cache."""
    with tempfile.TemporaryDirectory(prefix="lucid-pretrained-parity-") as directory:
        root = Path(directory)
        environment = os.environ.copy()
        try:
            from huggingface_hub import constants
        except ImportError:
            pass
        else:
            # Preserve normal client authentication when moving only download
            # caches. The client reads its token; this tool never copies it.
            environment.setdefault("HF_TOKEN_PATH", str(constants.HF_TOKEN_PATH))
        environment.update(
            {
                "LUCID_HOME": str(root / "lucid"),
                "HF_HOME": str(root / "hf"),
                "HF_HUB_CACHE": str(root / "hf" / "hub"),
                "HF_XET_CACHE": str(root / "hf" / "xet"),
                "HF_ASSETS_CACHE": str(root / "hf" / "assets"),
                "HUGGINGFACE_HUB_CACHE": str(root / "hf" / "hub"),
                "TRANSFORMERS_CACHE": str(root / "hf" / "hub"),
            }
        )
        # run() waits for (or kills on interruption) this direct child before
        # TemporaryDirectory removes its cache; it does not own grandchildren.
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.check_pretrained_parity",
                *arguments,
                "--_owned-cache",
                str(root),
            ],
            env=environment,
            check=False,
        ).returncode


def _params_of(name: str, member: object) -> int | None:
    meta = getattr(getattr(member, "value", None), "meta", {}) or {}
    count = meta.get("num_params") or getattr(_REGISTRY.get(name), "params", None)
    return int(count) if count else None


def _factory_default(name: str) -> WeightsEnum:
    """Ask the factory, not its shared enum, what pretrained=True selects.

    Stop before download/loading. Shadow construction also protects factories
    which instantiate the model before resolving its checkpoint.
    """
    resolve = weights_mod.resolve_weights
    selected: WeightsEnum | None = None
    finished = StopIteration()

    def capture(
        enum_cls: type[WeightsEnum], pretrained: bool | str, weights: WeightsEnum | None
    ) -> None:
        nonlocal selected
        selected = resolve(enum_cls, pretrained, weights)
        raise finished

    with shadow_alloc(), patch.object(weights_mod, "resolve_weights", capture):
        try:
            create_model(name, pretrained=True)
        except StopIteration as exc:
            if exc is not finished:
                raise
    if selected is None:
        raise RuntimeError(f"{name}: factory did not resolve a default checkpoint")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--model", help="only this factory")
    parser.add_argument("--source", help="only sources starting with this")
    parser.add_argument("--limit", type=int, help="stop after this many")
    parser.add_argument("--list", action="store_true", help="print what is checkable")
    parser.add_argument(
        "--json", type=Path, help="write per-checkpoint evidence, including unverified entries"
    )
    parser.add_argument("--_owned-cache", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--max-params",
        type=float,
        default=_MAX_PARAMS,
        help=(
            "report rather than load any model larger than this "
            f"(default {_MAX_PARAMS:.0e}; raise it where two copies fit)"
        ),
    )
    parser.add_argument(
        "--clean-downloads",
        action="store_true",
        help=(
            "use an isolated temporary cache and remove downloads after each comparison; "
            "shared user caches are neither read nor modified"
        ),
    )
    args = parser.parse_args()
    if not math.isfinite(args.max_params) or args.max_params <= 0:
        parser.error("--max-params must be positive and finite")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    targets: list[tuple[str, str, int | None]] = []
    for name in sorted(_WEIGHTS_BY_MODEL):
        if args.model and name != args.model:
            continue
        member = _factory_default(name)
        source = str(member.value.meta.get("source", ""))
        if args.source and not source.startswith(args.source):
            continue
        targets.append((name, source, _params_of(name, member)))

    if args.list:
        for name, source, _params in targets:
            loader = "loader available" if _reference_for(source) is not None else "no loader"
            print(f"{name}: {source} [{loader}]")
        print(f"\n{len(targets)} selected entries (loader availability is not verification)")
        return 0
    if not targets:
        print("nothing selected — no entry matched", file=sys.stderr)
        return 1

    if args.limit:
        targets = targets[: args.limit]

    if args.clean_downloads:
        if args._owned_cache is None:
            return _isolated_run(sys.argv[1:])
        ref = ref_module()
        if ref is not None:
            ref.hub.set_dir(str(args._owned_cache / "reference"))

    print(f"comparing {len(targets)} published checkpoints against their sources")
    bad: list[str] = []
    unreachable = 0
    checked = 0
    evidence: list[dict[str, object]] = []
    packages = ["numpy", "mlx", "transformers", "diffusers", "timm", "opencv-python-headless"]
    reference_name = getattr(ref_module(), "__name__", None)
    if reference_name:
        packages.append(reference_name)
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "tool_source_sha256_at_import": _TOOL_SOURCE_SHA256,
        "source_match": "tool source is identified; full working-tree identity is not inferred",
    }

    def write_report(*, finished: bool = False) -> None:
        if args.json is None:
            return
        args.json.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "version": lucid.__version__,
            "environment": environment,
            "selected": len(targets),
            "processed": len(evidence),
            "finished": finished,
            "compared": checked,
            "unreachable": unreachable,
            "problems": bad,
            "complete": finished and checked == len(targets) and not bad,
            "evidence": evidence,
            "scope": "one default checkpoint per factory; output parity, no dataset metric or training claim",
        }
        temporary = args.json.with_suffix(args.json.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.json)

    roots = _download_roots() if args.clean_downloads else []
    if roots and any(
        not root.resolve().is_relative_to(args._owned_cache.resolve()) for root in roots
    ):
        parser.error("cleanup refused: a download cache escaped the isolated directory")
    freed = 0
    for index, (name, source, params) in enumerate(targets, 1):
        # Preserve completed comparisons if a later native process crashes,
        # runs out of memory, or is interrupted. Partial means incomplete.
        write_report()
        before = _files_under(roots)
        with _reference_io():
            report = _compare(name, source, (1, 3, 224, 224), params, args.max_params)
        evidence.append({"model": name, "source": source, **report})
        freed += _remove_new(before, roots)
        if "unreachable" in report:
            unreachable += 1
            print(f"  [{index}/{len(targets)}] {name:26s} — {report['unreachable']}")
            continue
        if "unsupported" in report:
            unreachable += 1
            print(f"  [{index}/{len(targets)}] {name:26s} — {report['unsupported']}")
            continue
        if "degenerate" in report:
            print(f"  [{index}/{len(targets)}] {name:26s} — {report['degenerate']}")
            continue
        if "error" in report:
            bad.append(f"  {name}: {report['error']}")
            print(f"  [{index}/{len(targets)}] {name:26s} ERROR", flush=True)
            continue
        checked += 1
        diff = float(report["max_diff"])  # type: ignore[arg-type]
        agrees = bool(report["top1_agrees"])
        tolerance = float(report.get("tolerance", _ATOL))  # type: ignore[arg-type]
        label = "rel|Δ|" if report.get("relative") else "max|Δ|"
        ok = diff <= tolerance and agrees
        print(
            f"  [{index}/{len(targets)}] {name:26s} {label}={diff:.2e} "
            f"top1={'=' if agrees else '≠'} {'ok' if ok else 'MISMATCH'}",
            flush=True,
        )
        if not ok:
            bad.append(
                f"  {name}: {label}={diff:.3e} against {source}"
                + ("" if agrees else " — and the top-1 class differs")
            )

    print(f"\n{checked} compared, {unreachable} unreachable")
    write_report(finished=True)
    if args.clean_downloads:
        print(f"removed {freed / 1e9:.2f} GB of downloads along the way")
    if bad:
        print(f"\n{len(bad)} problem(s):", file=sys.stderr)
        for line in bad:
            print(line, file=sys.stderr)
        print(
            "\nComparison failed. Numeric mismatches require conversion/model investigation; "
            "download, adapter and runtime errors do not establish a numeric mismatch.",
            file=sys.stderr,
        )
        return 1
    if checked == 0:
        print(
            "nothing was verified — no reference installed, or every " "selection was unreachable",
            file=sys.stderr,
        )
        return 2
    print(
        f"[check_pretrained_parity] OK — {checked}/{len(targets)} selected checkpoints "
        "were compared and reproduce their sources; unverified entries remain unverified."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
