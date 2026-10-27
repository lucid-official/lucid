"""SK-ResNet weight converter — timm → Lucid.

Lucid's SK-ResNet module / state-dict layout was aligned with the
reference ``skresnet18`` / ``skresnet34`` implementation, so the
converter is a tiny stem + head rename (everything in between carries
through unchanged):

==============================  =========================
timm                            Lucid
==============================  =========================
``conv1.weight``                ``stem.0.weight``
``bn1.{w,b,rm,rv,nbt}``         ``stem.1.{w,b,rm,rv,nbt}``
``layerN.*``                    ``layerN.*``  (identical)
``fc.{weight,bias}``            ``classifier.{weight,bias}``
==============================  =========================

Each basic block keeps the same ``conv1`` (SelectiveKernel) /
``conv2`` (plain conv → BN) / ``downsample`` naming on both sides, so
``map_key`` only has to special-case the stem and the classifier head.
"""

import dataclasses

from lucid.nn import Module
from tools.convert_weights._base import Architecture, ConversionSpec, register_arch

_SKNET_CITATION = (
    "@inproceedings{li2019selective,\n"
    "  title={Selective Kernel Networks},\n"
    "  author={Li, Xiang and Wang, Wenhai and Hu, Xiaolin and Yang, Jian},\n"
    "  booktitle={CVPR}, year={2019}\n"
    "}"
)

# arch -> (lucid_cls_factory, repo_id, title)
_SKNET_VARIANTS: dict[str, tuple[str, str, str]] = {
    "sk_resnet_18": ("sk_resnet_18_cls", "lucid-dl/sk-resnet-18", "SK-ResNet-18"),
    "sk_resnet_34": ("sk_resnet_34_cls", "lucid-dl/sk-resnet-34", "SK-ResNet-34"),
}

# arch -> timm canonical model name (used to download the source weights)
_TIMM_NAMES: dict[str, str] = {"sk_resnet_18": "skresnet18", "sk_resnet_34": "skresnet34"}

# top-1 ImageNet-1k accuracy from timm's results-imagenet.csv (ra_in1k).
_ACC1: dict[str, float] = {"sk_resnet_18": 73.020, "sk_resnet_34": 76.956}


class SKNetArch(Architecture):
    """Converter for one paper-cited SK-ResNet variant + tag."""

    def __init__(self, arch: str, tag: str) -> None:
        import timm

        if arch not in _SKNET_VARIANTS:
            raise KeyError(f"SKNetArch: unknown arch {arch!r}")
        self.arch = arch
        # Tags ship uppercase per Lucid convention (e.g. ``RA_IN1K``);
        # timm wants them lowercase (``ra_in1k``).
        self.tag = tag
        self._timm_name = f"{_TIMM_NAMES[arch]}.{tag.lower()}"
        self._model = timm.create_model(self._timm_name, pretrained=True)
        self._model.eval()
        import lucid.models as models

        self._lucid_factory = _SKNET_VARIANTS[arch][0]
        self._lucid_model = getattr(models, self._lucid_factory)()

    def source_state_dict(self) -> dict[str, object]:
        return {k: v.detach().cpu().numpy() for k, v in self._model.state_dict().items()}

    def target_model(self) -> Module:
        return self._lucid_model

    def map_key(self, src_key: str) -> str | None:
        if src_key == "conv1.weight":
            return "stem.0.weight"
        if src_key.startswith("bn1."):
            return "stem.1." + src_key[len("bn1.") :]
        if src_key.startswith("fc."):
            return "classifier." + src_key[len("fc.") :]
        return src_key

    def spec(self) -> ConversionSpec:
        factory_name, repo_id, title = _SKNET_VARIANTS[self.arch]
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
        meta = {
            "num_params": n_params,
            "recipe": str(cfg.get("hf_hub_id", "")),
            "metrics": {"ImageNet-1k": {"acc@1": _ACC1.get(self.arch, 0.0)}},
        }

        return ConversionSpec(
            model_name=factory_name,
            architecture=self.arch,
            repo_id=repo_id,
            tag=self.tag,
            task="image-classification",
            model_type="sknet",
            source=f"timm/{self._timm_name}",
            license=str(cfg.get("license", "apache-2.0")),
            num_classes=int(self._lucid_model.config.num_classes),
            config=config,
            preprocessing=preprocessing,
            citation=_SKNET_CITATION,
            title=title,
            paper_url="Li et al., 2019 — *Selective Kernel Networks* " "(arXiv:1903.06586)",
            categories=[],
            datasets=["imagenet-1k"],
            meta=meta,
        )


@register_arch("sk_resnet_18")
def _sk18(tag: str) -> Architecture:
    return SKNetArch("sk_resnet_18", tag)


@register_arch("sk_resnet_34")
def _sk34(tag: str) -> Architecture:
    return SKNetArch("sk_resnet_34", tag)
