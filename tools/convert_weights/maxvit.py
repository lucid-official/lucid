"""MaxViT weight converter — timm → Lucid.

Lucid's MaxViT (:mod:`lucid.models.vision.maxvit`) was deliberately
built to mirror timm's ``maxvit_*_tf_224`` module tree *verbatim*, so
the converter is a pure identity map — no ``map_key`` rewrites and no
``transform_value`` reshapes.  Both state dicts agree on all 560 keys
(MaxViT-Tiny) with identical shapes; the same holds for Small / Base /
Large, which only scale ``depths`` / ``dims``.

==============================================  =============================
timm                                            Lucid
==============================================  =============================
``stem.{conv1,norm1,conv2}.*``                  identical
``stages.S.blocks.N.conv.*`` (MBConv)           identical
``stages.S.blocks.N.attn_{block,grid}.*``       identical
``head.{norm,pre_logits.fc,fc}.*``              identical
==============================================  =============================

Source provenance
-----------------
The ``tf_224.in1k`` checkpoints are the official Google MaxViT weights
(Tu et al., ECCV 2022) re-hosted by timm.  ``maxvit_xlarge_tf_224`` only
ships an ``in21k`` (21 841-class) checkpoint, so it has no ImageNet-1k
head and is *not* convertible into a 1000-class classifier — it is
intentionally absent here.

.. warning::

   The ``convert()`` gate (key-set + shape + ``load_state_dict``) PASSES
   for every variant, but the converted weights do **not** reach
   numerical parity with timm until three forward-semantic bugs in
   :mod:`lucid.models.vision.maxvit._model` are fixed (BatchNorm
   ``eps``, GELU tanh-approximation, relative-position-bias sign).  See
   the conversion report's ``needs_model_change`` field.  This module is
   correct as a *key map* and is ready to use once the model is patched.
"""

import dataclasses

from lucid.nn import Module
from tools.convert_weights._base import Architecture, ConversionSpec, register_arch

_MAXVIT_CITATION = (
    "@inproceedings{tu2022maxvit,\n"
    "  title={MaxViT: Multi-Axis Vision Transformer},\n"
    "  author={Tu, Zhengzhong and Talebi, Hossein and Zhang, Han and "
    "Yang, Feng and Milanfar, Peyman and Bovik, Alan and Li, Yinxiao},\n"
    "  booktitle={ECCV}, year={2022}\n"
    "}"
)

_MAXVIT_PAPER_URL = (
    "Tu et al., 2022 — *MaxViT: Multi-Axis Vision Transformer* " "(arXiv:2204.01697)"
)

# arch -> (timm_arch, lucid_cls_factory, repo_id, title, paper_acc1, paper_acc5)
_MAXVIT_VARIANTS: dict[str, tuple[str, str, str, str, float, float]] = {
    "maxvit_tiny": (
        "maxvit_tiny_tf_224",
        "maxvit_tiny_cls",
        "lucid-dl/maxvit-tiny",
        "MaxViT-Tiny",
        83.62,
        96.49,
    ),
    "maxvit_small": (
        "maxvit_small_tf_224",
        "maxvit_small_cls",
        "lucid-dl/maxvit-small",
        "MaxViT-Small",
        84.45,
        96.98,
    ),
    "maxvit_base": (
        "maxvit_base_tf_224",
        "maxvit_base_cls",
        "lucid-dl/maxvit-base",
        "MaxViT-Base",
        84.95,
        97.04,
    ),
    "maxvit_large": (
        "maxvit_large_tf_224",
        "maxvit_large_cls",
        "lucid-dl/maxvit-large",
        "MaxViT-Large",
        85.17,
        97.17,
    ),
}


class MaxViTArch(Architecture):
    """Converter for one timm ``maxvit_*_tf_224`` variant + tag.

    Lucid mirrors timm's module tree exactly, so :meth:`map_key` is the
    identity and :meth:`transform_value` is left at its base default.
    """

    def __init__(self, arch: str, tag: str) -> None:
        import timm

        if arch not in _MAXVIT_VARIANTS:
            raise KeyError(f"MaxViTArch: unknown arch {arch!r}")
        self.arch = arch
        # Lucid keeps weight-enum tags uppercase (``IN1K``); timm's own
        # model registry is lowercase, so the lookup uses a lowered copy.
        self.tag = tag
        self._timm_arch = _MAXVIT_VARIANTS[arch][0]
        self._timm_name = f"{self._timm_arch}.{tag.lower()}"
        self._model = timm.create_model(self._timm_name, pretrained=True)
        self._model.eval()

        import lucid.models as models

        self._lucid_factory = _MAXVIT_VARIANTS[arch][1]
        self._lucid_model = getattr(models, self._lucid_factory)()

    def source_state_dict(self) -> dict[str, object]:
        return {k: v.detach().cpu().numpy() for k, v in self._model.state_dict().items()}

    def target_model(self) -> Module:
        return self._lucid_model

    def map_key(self, src_key: str) -> str | None:
        # Identity — Lucid mirrors the timm MaxViT layout verbatim.
        return src_key

    def spec(self) -> ConversionSpec:
        _timm_arch, factory_name, repo_id, title, acc1, acc5 = _MAXVIT_VARIANTS[self.arch]
        config = {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in dataclasses.asdict(self._lucid_model.config).items()
        }

        cfg = self._model.default_cfg
        from lucid.utils.transforms import ImageClassification

        crop = int(cfg["input_size"][1])
        crop_pct = float(cfg.get("crop_pct", 0.95))
        # timm uses floor division when deriving the resize edge.
        resize = int(crop / crop_pct)
        preset = ImageClassification(
            crop_size=crop,
            resize_size=resize,
            mean=tuple(float(m) for m in cfg.get("mean", (0.485, 0.456, 0.406))),
            std=tuple(float(s) for s in cfg.get("std", (0.229, 0.224, 0.225))),
            interpolation=str(cfg.get("interpolation", "bicubic")),
        )
        preprocessing = preset.to_dict()

        meta = {
            "num_params": int(sum(p.numel() for p in self._model.parameters())),
            "recipe": f"timm/{self._timm_name}",
            "metrics": {"ImageNet-1k": {"acc@1": acc1, "acc@5": acc5}},
        }

        return ConversionSpec(
            model_name=factory_name,
            architecture=self.arch,
            repo_id=repo_id,
            tag=self.tag,
            task="image-classification",
            model_type="maxvit",
            source=f"timm/{self._timm_name}",
            license="apache-2.0",
            num_classes=int(self._lucid_model.config.num_classes),
            config=config,
            preprocessing=preprocessing,
            citation=_MAXVIT_CITATION,
            title=title,
            paper_url=_MAXVIT_PAPER_URL,
            categories=[],
            datasets=["imagenet-1k"],
            meta=meta,
        )


@register_arch("maxvit_tiny")
def _build_tiny(tag: str) -> Architecture:
    return MaxViTArch("maxvit_tiny", tag)


@register_arch("maxvit_small")
def _build_small(tag: str) -> Architecture:
    return MaxViTArch("maxvit_small", tag)


@register_arch("maxvit_base")
def _build_base(tag: str) -> Architecture:
    return MaxViTArch("maxvit_base", tag)


@register_arch("maxvit_large")
def _build_large(tag: str) -> Architecture:
    return MaxViTArch("maxvit_large", tag)
