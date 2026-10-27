"""ResNet weight converter — torchvision → Lucid.

Maps torchvision's ResNet ``state_dict`` keys onto Lucid's module
layout.  Only the stem and head differ; the four stages
(``layerN.*``) share identical naming, so the mapping is small:

==============================  =========================
torchvision                     Lucid
==============================  =========================
``conv1.weight``                ``stem.0.weight``
``bn1.{w,b,rm,rv,nbt}``         ``stem.1.{w,b,rm,rv,nbt}``
``layerN.*``                    ``layerN.*``  (identical)
``fc.{weight,bias}``            ``classifier.{weight,bias}``
==============================  =========================
"""

import dataclasses

import torchvision.models as tvm

from lucid.nn import Module
from tools.convert_weights._base import Architecture, ConversionSpec, register_arch

_RESNET_CITATION = (
    "@inproceedings{he2016deep,\n"
    "  title={Deep Residual Learning for Image Recognition},\n"
    "  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},\n"
    "  booktitle={CVPR}, year={2016}\n"
    "}"
)

_WIDE_RESNET_CITATION = (
    "@inproceedings{zagoruyko2016wide,\n"
    "  title={Wide Residual Networks},\n"
    "  author={Zagoruyko, Sergey and Komodakis, Nikos},\n"
    "  booktitle={BMVC}, year={2016}\n"
    "}"
)

_RESNET_PAPER_URL = (
    "He et al., 2015 — *Deep Residual Learning for Image " "Recognition* (arXiv:1512.03385)"
)
_WIDE_RESNET_PAPER_URL = (
    "Zagoruyko & Komodakis, 2016 — *Wide Residual Networks* " "(arXiv:1605.07146)"
)

# Lucid factory + torchvision (builder, weights-enum) per architecture.
_RESNET_VARIANTS: dict[str, tuple[str, str, str]] = {
    # arch -> (lucid_cls_factory, repo_id, title)
    "resnet_18": ("resnet_18_cls", "lucid-dl/resnet-18", "ResNet-18"),
    "resnet_34": ("resnet_34_cls", "lucid-dl/resnet-34", "ResNet-34"),
    "resnet_50": ("resnet_50_cls", "lucid-dl/resnet-50", "ResNet-50"),
    "resnet_101": ("resnet_101_cls", "lucid-dl/resnet-101", "ResNet-101"),
    "resnet_152": ("resnet_152_cls", "lucid-dl/resnet-152", "ResNet-152"),
    "wide_resnet_50": ("wide_resnet_50_cls", "lucid-dl/wide-resnet-50-2", "Wide ResNet-50-2"),
    "wide_resnet_101": ("wide_resnet_101_cls", "lucid-dl/wide-resnet-101-2", "Wide ResNet-101-2"),
}

# torchvision builders keyed by arch.
_TV_BUILDERS = {
    "resnet_18": (tvm.resnet18, tvm.ResNet18_Weights),
    "resnet_34": (tvm.resnet34, tvm.ResNet34_Weights),
    "resnet_50": (tvm.resnet50, tvm.ResNet50_Weights),
    "resnet_101": (tvm.resnet101, tvm.ResNet101_Weights),
    "resnet_152": (tvm.resnet152, tvm.ResNet152_Weights),
    "wide_resnet_50": (tvm.wide_resnet50_2, tvm.Wide_ResNet50_2_Weights),
    "wide_resnet_101": (tvm.wide_resnet101_2, tvm.Wide_ResNet101_2_Weights),
}


class ResNetArch(Architecture):
    """Converter for one ResNet variant + tag."""

    def __init__(self, arch: str, tag: str) -> None:
        if arch not in _RESNET_VARIANTS:
            raise KeyError(f"ResNetArch: unknown arch {arch!r}")
        self.arch = arch
        self.tag = tag
        self._builder, self._weights_enum = _TV_BUILDERS[arch]
        self._tv_weights = self._weights_enum[tag]

    def source_state_dict(self) -> dict[str, object]:
        model = self._builder(weights=self._tv_weights)
        model.eval()
        return {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}

    def target_model(self) -> Module:
        import lucid.models as models

        factory = _RESNET_VARIANTS[self.arch][0]
        return getattr(models, factory)()

    def map_key(self, src_key: str) -> str | None:
        if src_key == "conv1.weight":
            return "stem.0.weight"
        if src_key.startswith("bn1."):
            return "stem.1." + src_key[len("bn1.") :]
        if src_key.startswith("fc."):
            return "classifier." + src_key[len("fc.") :]
        return src_key

    def spec(self) -> ConversionSpec:
        import lucid.models as models

        factory_name, repo_id, title = _RESNET_VARIANTS[self.arch]
        model = getattr(models, factory_name)()
        config = {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in dataclasses.asdict(model.config).items()
        }

        tv_meta = dict(self._tv_weights.meta)
        categories = list(tv_meta.get("categories", []))
        from lucid.utils.transforms import ImageClassification

        tf = self._tv_weights.transforms()
        # Construct a Lucid preset and serialise via its own to_dict
        # so the on-Hub schema stays in lock-step with the runtime
        # contract (``AutoTransformsPreset.from_dict`` consumes this
        # exact shape).
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

        is_wide = self.arch.startswith("wide_")
        return ConversionSpec(
            model_name=factory_name,
            architecture=self.arch,
            repo_id=repo_id,
            tag=self.tag,
            task="image-classification",
            model_type="resnet",
            source=f"torchvision/{self._weights_enum.__name__}.{self.tag}",
            license="bsd-3-clause",
            num_classes=int(model.config.num_classes),
            config=config,
            preprocessing=preprocessing,
            citation=_WIDE_RESNET_CITATION if is_wide else _RESNET_CITATION,
            title=title,
            paper_url=_WIDE_RESNET_PAPER_URL if is_wide else _RESNET_PAPER_URL,
            categories=categories,
            datasets=["imagenet-1k"],
            meta=meta,
        )


@register_arch("resnet_18")
def _build_resnet_18(tag: str) -> Architecture:
    return ResNetArch("resnet_18", tag)


@register_arch("resnet_34")
def _build_resnet_34(tag: str) -> Architecture:
    return ResNetArch("resnet_34", tag)


@register_arch("resnet_50")
def _build_resnet_50(tag: str) -> Architecture:
    return ResNetArch("resnet_50", tag)


@register_arch("resnet_101")
def _build_resnet_101(tag: str) -> Architecture:
    return ResNetArch("resnet_101", tag)


@register_arch("resnet_152")
def _build_resnet_152(tag: str) -> Architecture:
    return ResNetArch("resnet_152", tag)


@register_arch("wide_resnet_50")
def _build_wide_resnet_50(tag: str) -> Architecture:
    return ResNetArch("wide_resnet_50", tag)


@register_arch("wide_resnet_101")
def _build_wide_resnet_101(tag: str) -> Architecture:
    return ResNetArch("wide_resnet_101", tag)
