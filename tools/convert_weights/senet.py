"""SE-ResNet weight converter — timm → Lucid.

Five paper-cited SE-ResNet variants (Hu et al., CVPR 2018).  Four of
them have no modern timm equivalent and ship from the ``legacy_seresnet``
line, whose module layout differs from Lucid in three spots that need a
small rename map:

==============================  =========================
timm ``legacy_seresnet*``       Lucid
==============================  =========================
``layer0.conv1.*``              ``conv1.*``
``layer0.bn1.*``                ``bn1.*``
``*.se_module.*``               ``*.se.*``
``last_linear.*``               ``fc.*``
``layer{1-4}.*`` (else)         identical
==============================  =========================

SE-ResNet-50 instead pulls the stronger ``seresnet50.ra2_in1k`` recipe
(78.5 top-1).  That checkpoint is a plain SE-augmented ResNet whose
state-dict naming (``conv1`` / ``bn1`` / ``*.se.*`` / ``fc``) already
matches Lucid one-for-one, so its mapping is a pure identity.

The canonical SE block uses ``rd_channels = channels // reduction``
(plain floor division); Lucid's :class:`_SEBlock` was aligned to this so
the converted SE 1×1 ``fc1`` / ``fc2`` shapes round-trip exactly.
"""

import dataclasses

from lucid.nn import Module
from tools.convert_weights._base import Architecture, ConversionSpec, register_arch

_SENET_CITATION = (
    "@inproceedings{hu2018squeeze,\n"
    "  title={Squeeze-and-Excitation Networks},\n"
    "  author={Hu, Jie and Shen, Li and Sun, Gang},\n"
    "  booktitle={CVPR}, year={2018}\n"
    "}"
)

_SENET_PAPER_URL = "Hu et al., 2018 — *Squeeze-and-Excitation Networks* " "(arXiv:1709.01507)"

# arch -> (lucid_cls_factory, repo_id, title)
_SENET_VARIANTS: dict[str, tuple[str, str, str]] = {
    "se_resnet_18": ("se_resnet_18_cls", "lucid-dl/se-resnet-18", "SE-ResNet-18"),
    "se_resnet_34": ("se_resnet_34_cls", "lucid-dl/se-resnet-34", "SE-ResNet-34"),
    "se_resnet_50": ("se_resnet_50_cls", "lucid-dl/se-resnet-50", "SE-ResNet-50"),
    "se_resnet_101": ("se_resnet_101_cls", "lucid-dl/se-resnet-101", "SE-ResNet-101"),
    "se_resnet_152": ("se_resnet_152_cls", "lucid-dl/se-resnet-152", "SE-ResNet-152"),
}

# arch -> timm canonical model name (without the tag suffix).  SE-ResNet-50
# uses the modern ``seresnet50`` recipe (cleaner topology + higher acc);
# the rest only exist in the ``legacy_seresnet`` line.
_TIMM_NAMES: dict[str, str] = {
    "se_resnet_18": "legacy_seresnet18",
    "se_resnet_34": "legacy_seresnet34",
    "se_resnet_50": "seresnet50",
    "se_resnet_101": "legacy_seresnet101",
    "se_resnet_152": "legacy_seresnet152",
}

# Architectures whose timm state-dict already matches Lucid one-for-one.
_IDENTITY_ARCHS = frozenset({"se_resnet_50"})

# Published timm acc@1 (ImageNet-1k val) per source checkpoint.
_ACC1: dict[str, float] = {
    "se_resnet_18": 70.6,
    "se_resnet_34": 73.31,
    "se_resnet_50": 78.498,
    "se_resnet_101": 78.32,
    "se_resnet_152": 78.66,
}


class SENetArch(Architecture):
    """Converter for one paper-cited SE-ResNet variant + tag."""

    def __init__(self, arch: str, tag: str) -> None:
        import timm

        if arch not in _SENET_VARIANTS:
            raise KeyError(f"SENetArch: unknown arch {arch!r}")
        self.arch = arch
        # Tags ship uppercase per Lucid convention (e.g. ``IN1K`` /
        # ``RA2_IN1K``); timm wants them lowercase.
        self.tag = tag
        self._timm_name = f"{_TIMM_NAMES[arch]}.{tag.lower()}"
        self._model = timm.create_model(self._timm_name, pretrained=True)
        self._model.eval()

        import lucid.models as models

        self._lucid_factory = _SENET_VARIANTS[arch][0]
        self._lucid_model = getattr(models, self._lucid_factory)()

    def source_state_dict(self) -> dict[str, object]:
        return {k: v.detach().cpu().numpy() for k, v in self._model.state_dict().items()}

    def target_model(self) -> Module:
        return self._lucid_model

    def map_key(self, src_key: str) -> str | None:
        if self.arch in _IDENTITY_ARCHS:
            return src_key
        # legacy_seresnet rename map.
        if src_key.startswith("layer0.conv1."):
            return "conv1." + src_key[len("layer0.conv1.") :]
        if src_key.startswith("layer0.bn1."):
            return "bn1." + src_key[len("layer0.bn1.") :]
        if src_key.startswith("last_linear."):
            return "fc." + src_key[len("last_linear.") :]
        return src_key.replace(".se_module.", ".se.")

    def spec(self) -> ConversionSpec:
        factory_name, repo_id, title = _SENET_VARIANTS[self.arch]
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
            interpolation=str(cfg.get("interpolation", "bilinear")),
        )
        preprocessing = preset.to_dict()

        n_params = int(sum(p.numel() for p in self._model.parameters()))
        meta = {
            "num_params": n_params,
            "recipe": str(cfg.get("url", "")),
            "metrics": {"ImageNet-1k": {"acc@1": _ACC1[self.arch]}},
        }

        return ConversionSpec(
            model_name=factory_name,
            architecture=self.arch,
            repo_id=repo_id,
            tag=self.tag,
            task="image-classification",
            model_type="senet",
            source=f"timm/{self._timm_name}",
            license=str(cfg.get("license", "apache-2.0")),
            num_classes=int(self._lucid_model.config.num_classes),
            config=config,
            preprocessing=preprocessing,
            citation=_SENET_CITATION,
            title=title,
            paper_url=_SENET_PAPER_URL,
            categories=[],
            datasets=["imagenet-1k"],
            meta=meta,
        )


@register_arch("se_resnet_18")
def _se18(tag: str) -> Architecture:
    return SENetArch("se_resnet_18", tag)


@register_arch("se_resnet_34")
def _se34(tag: str) -> Architecture:
    return SENetArch("se_resnet_34", tag)


@register_arch("se_resnet_50")
def _se50(tag: str) -> Architecture:
    return SENetArch("se_resnet_50", tag)


@register_arch("se_resnet_101")
def _se101(tag: str) -> Architecture:
    return SENetArch("se_resnet_101", tag)


@register_arch("se_resnet_152")
def _se152(tag: str) -> Architecture:
    return SENetArch("se_resnet_152", tag)
