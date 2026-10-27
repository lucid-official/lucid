"""ResNeXt weight converter — torchvision / timm → Lucid.

ResNeXt's grouped-convolution bottleneck is macroscopically identical to
ResNet-50+, and both torchvision and timm name their state dicts exactly
like their ResNet counterparts.  Lucid mirrors that layout too, so the
mapping is the same small stem/head rename used by the ResNet converter:

==============================  =========================
source                          Lucid
==============================  =========================
``conv1.weight``                ``stem.0.weight``
``bn1.{w,b,rm,rv,nbt}``         ``stem.1.{w,b,rm,rv,nbt}``
``layerN.*``                    ``layerN.*``  (identical)
``fc.{weight,bias}``            ``classifier.{weight,bias}``
==============================  =========================

The ``cardinality=32`` grouping is baked into the ``conv2`` weight shapes
(e.g. ``(128, 4, 3, 3)`` for a 32×4d block), so no value reshaping is
needed — the grouped weights carry through verbatim.

Two source preset wrinkles, replicated exactly in :meth:`spec`:

* The torchvision V2 checkpoints (``resnext_50_32x4d`` /
  ``resnext_101_32x8d``) use a **232** resize (not the usual 256) ahead
  of the 224 center crop, per the "new training recipe".
* The timm Gluon checkpoint (``resnext_101_32x4d``) uses a **bicubic**
  0.875-crop_pct preset (resize 256 → crop 224).
"""

import dataclasses

from lucid.nn import Module
from tools.convert_weights._base import Architecture, ConversionSpec, register_arch

_RESNEXT_CITATION = (
    "@inproceedings{xie2017aggregated,\n"
    "  title={Aggregated Residual Transformations for Deep Neural "
    "Networks},\n"
    "  author={Xie, Saining and Girshick, Ross and Doll{\\'a}r, Piotr "
    "and Tu, Zhuowen and He, Kaiming},\n"
    "  booktitle={CVPR}, year={2017}\n"
    "}"
)

_PAPER_URL = (
    "Xie et al., 2017 — *Aggregated Residual Transformations for Deep "
    "Neural Networks* (arXiv:1611.05431)"
)

# arch -> (lucid_cls_factory, repo_id, title)
_RESNEXT_VARIANTS: dict[str, tuple[str, str, str]] = {
    "resnext_50_32x4d": ("resnext_50_32x4d_cls", "lucid-dl/resnext-50-32x4d", "ResNeXt-50 (32x4d)"),
    "resnext_101_32x8d": (
        "resnext_101_32x8d_cls",
        "lucid-dl/resnext-101-32x8d",
        "ResNeXt-101 (32x8d)",
    ),
    "resnext_101_32x4d": (
        "resnext_101_32x4d_cls",
        "lucid-dl/resnext-101-32x4d",
        "ResNeXt-101 (32x4d)",
    ),
}

# Which provenance each arch pulls from.
_TORCHVISION_ARCHS = {"resnext_50_32x4d", "resnext_101_32x8d"}
_TIMM_ARCHS = {"resnext_101_32x4d"}

# torchvision builders keyed by arch (lazily resolved in __init__).
_TV_BUILDER_NAMES: dict[str, tuple[str, str]] = {
    "resnext_50_32x4d": ("resnext50_32x4d", "ResNeXt50_32X4D_Weights"),
    "resnext_101_32x8d": ("resnext101_32x8d", "ResNeXt101_32X8D_Weights"),
}

# timm canonical model name keyed by arch.
_TIMM_NAMES: dict[str, str] = {"resnext_101_32x4d": "resnext101_32x4d"}

# Published acc@1 / acc@5 for the timm Gluon checkpoint (timm
# results-imagenet.csv — ``gluon_resnext101_32x4d``).
_TIMM_METRICS: dict[str, tuple[float, float]] = {"resnext_101_32x4d": (80.342, 94.926)}


class ResNeXtArch(Architecture):
    """Converter for one ResNeXt variant + tag (torchvision or timm)."""

    def __init__(self, arch: str, tag: str) -> None:
        if arch not in _RESNEXT_VARIANTS:
            raise KeyError(f"ResNeXtArch: unknown arch {arch!r}")
        self.arch = arch
        self.tag = tag

        import lucid.models as models

        self._lucid_factory = _RESNEXT_VARIANTS[arch][0]
        self._lucid_model = getattr(models, self._lucid_factory)()

        if arch in _TORCHVISION_ARCHS:
            import torchvision.models as tvm

            builder_name, enum_name = _TV_BUILDER_NAMES[arch]
            self._builder = getattr(tvm, builder_name)
            self._weights_enum = getattr(tvm, enum_name)
            self._tv_weights = self._weights_enum[tag]
            self._timm_name = ""
            self._model = None
        else:
            import timm

            # Tags ship uppercase per Lucid convention (e.g. ``GLUON_IN1K``);
            # timm wants them lowercase (``gluon_in1k``).
            self._timm_name = f"{_TIMM_NAMES[arch]}.{tag.lower()}"
            self._model = timm.create_model(self._timm_name, pretrained=True)
            self._model.eval()

    def source_state_dict(self) -> dict[str, object]:
        if self.arch in _TORCHVISION_ARCHS:
            model = self._builder(weights=self._tv_weights)
            model.eval()
            return {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
        return {k: v.detach().cpu().numpy() for k, v in self._model.state_dict().items()}

    def target_model(self) -> Module:
        return self._lucid_model

    def map_key(self, src_key: str) -> str | None:
        # Same stem/head rename as ResNet — the four stages share
        # identical naming with both torchvision and timm.
        if src_key == "conv1.weight":
            return "stem.0.weight"
        if src_key.startswith("bn1."):
            return "stem.1." + src_key[len("bn1.") :]
        if src_key.startswith("fc."):
            return "classifier." + src_key[len("fc.") :]
        return src_key

    def _spec_torchvision(self) -> ConversionSpec:
        factory_name, repo_id, title = _RESNEXT_VARIANTS[self.arch]
        config = {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in dataclasses.asdict(self._lucid_model.config).items()
        }

        tv_meta = dict(self._tv_weights.meta)
        categories = list(tv_meta.get("categories", []))
        from lucid.utils.transforms import ImageClassification

        tf = self._tv_weights.transforms()
        preset = ImageClassification(
            crop_size=int(tf.crop_size[0]),
            resize_size=int(tf.resize_size[0]),
            mean=tuple(float(m) for m in tf.mean),
            std=tuple(float(s) for s in tf.std),
            interpolation=str(tf.interpolation.value),
        )
        preprocessing = preset.to_dict()
        meta = {
            "num_params": int(tv_meta.get("num_params", 0)),
            "gflops": float(tv_meta.get("_ops", 0.0)),
            "recipe": str(tv_meta.get("recipe", "")),
            "metrics": dict(tv_meta.get("_metrics", {})),
        }

        return ConversionSpec(
            model_name=factory_name,
            architecture=self.arch,
            repo_id=repo_id,
            tag=self.tag,
            task="image-classification",
            model_type="resnext",
            source=f"torchvision/{self._weights_enum.__name__}.{self.tag}",
            license="bsd-3-clause",
            num_classes=int(self._lucid_model.config.num_classes),
            config=config,
            preprocessing=preprocessing,
            citation=_RESNEXT_CITATION,
            title=title,
            paper_url=_PAPER_URL,
            categories=categories,
            datasets=["imagenet-1k"],
            meta=meta,
        )

    def _spec_timm(self) -> ConversionSpec:
        factory_name, repo_id, title = _RESNEXT_VARIANTS[self.arch]
        config = {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in dataclasses.asdict(self._lucid_model.config).items()
        }

        from lucid.utils.transforms import ImageClassification

        cfg = self._model.default_cfg
        crop = int(cfg["input_size"][1])
        resize = int(round(crop / float(cfg.get("crop_pct", 0.875))))
        preset = ImageClassification(
            crop_size=crop,
            resize_size=resize,
            mean=tuple(float(m) for m in cfg.get("mean", (0.485, 0.456, 0.406))),
            std=tuple(float(s) for s in cfg.get("std", (0.229, 0.224, 0.225))),
            interpolation=str(cfg.get("interpolation", "bicubic")),
        )
        preprocessing = preset.to_dict()

        n_params = int(sum(p.numel() for p in self._model.parameters()))
        acc1, acc5 = _TIMM_METRICS[self.arch]
        meta = {
            "num_params": n_params,
            "recipe": str(cfg.get("origin_url", cfg.get("url", ""))),
            "metrics": {"ImageNet-1k": {"acc@1": acc1, "acc@5": acc5}},
        }

        return ConversionSpec(
            model_name=factory_name,
            architecture=self.arch,
            repo_id=repo_id,
            tag=self.tag,
            task="image-classification",
            model_type="resnext",
            source=f"timm/{self._timm_name}",
            license=str(cfg.get("license", "apache-2.0")),
            num_classes=int(self._lucid_model.config.num_classes),
            config=config,
            preprocessing=preprocessing,
            citation=_RESNEXT_CITATION,
            title=title,
            paper_url=_PAPER_URL,
            categories=[],
            datasets=["imagenet-1k"],
            meta=meta,
        )

    def spec(self) -> ConversionSpec:
        if self.arch in _TORCHVISION_ARCHS:
            return self._spec_torchvision()
        return self._spec_timm()


@register_arch("resnext_50_32x4d")
def _50_32x4d(tag: str) -> Architecture:
    return ResNeXtArch("resnext_50_32x4d", tag)


@register_arch("resnext_101_32x8d")
def _101_32x8d(tag: str) -> Architecture:
    return ResNeXtArch("resnext_101_32x8d", tag)


@register_arch("resnext_101_32x4d")
def _101_32x4d(tag: str) -> Architecture:
    return ResNeXtArch("resnext_101_32x4d", tag)
