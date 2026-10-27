"""CSPNet weight converter — timm → Lucid.

Module + state-dict naming was mirrored when the Lucid model was
rewritten so the converter is a *single* trivial rename for all three
paper-cited variants:

    head.fc.{weight, bias}  →  classifier.{weight, bias}

Everything else (``stem.conv1`` / ``stages.S.conv_down`` /
``stages.S.conv_exp`` / ``stages.S.blocks.B.conv{1,2,3}`` /
``stages.S.conv_transition_b`` / ``stages.S.conv_transition``)
carries through unchanged.
"""

import dataclasses

from lucid.nn import Module
from tools.convert_weights._base import Architecture, ConversionSpec, register_arch

_CSPNET_CITATION = (
    "@inproceedings{wang2020cspnet,\n"
    "  title={CSPNet: A New Backbone that can Enhance Learning "
    "Capability of CNN},\n"
    "  author={Wang, Chien-Yao and Liao, Hong-Yuan Mark and Wu, Yueh-Hua "
    "and Chen, Ping-Yang and Hsieh, Jun-Wei and Yeh, I-Hau},\n"
    "  booktitle={CVPR Workshops}, year={2020}\n"
    "}"
)

# arch -> (lucid_cls_factory, repo_id, title)
_CSPNET_VARIANTS: dict[str, tuple[str, str, str]] = {
    "cspresnet_50": ("cspresnet_50_cls", "lucid-dl/cspresnet-50", "CSPResNet-50"),
    "cspresnext_50": ("cspresnext_50_cls", "lucid-dl/cspresnext-50", "CSPResNeXt-50"),
    "cspdarknet_53": ("cspdarknet_53_cls", "lucid-dl/cspdarknet-53", "CSPDarknet-53"),
}

# arch -> timm canonical model name (used to download the source weights)
_TIMM_NAMES: dict[str, str] = {
    "cspresnet_50": "cspresnet50",
    "cspresnext_50": "cspresnext50",
    "cspdarknet_53": "cspdarknet53",
}


def _datasets_from_timm_tag(tag: str) -> list[str]:
    """Map an uppercase Lucid tag to the dataset list (CSPNet line)."""
    t = tag.lower()
    ds: list[str] = []
    if "in22k" in t:
        ds.append("imagenet-22k")
    if "in1k" in t:
        ds.append("imagenet-1k")
    return ds


class CSPNetArch(Architecture):
    """Converter for one paper-cited CSPNet variant + tag."""

    def __init__(self, arch: str, tag: str) -> None:
        import timm

        if arch not in _CSPNET_VARIANTS:
            raise KeyError(f"CSPNetArch: unknown arch {arch!r}")
        self.arch = arch
        # Tags ship uppercase per Lucid convention (e.g. ``RA_IN1K``);
        # timm wants them lowercase (``ra_in1k``).
        self.tag = tag
        self._timm_name = f"{_TIMM_NAMES[arch]}.{tag.lower()}"
        self._model = timm.create_model(self._timm_name, pretrained=True)
        self._model.eval()
        import lucid.models as models

        self._lucid_factory = _CSPNET_VARIANTS[arch][0]
        self._lucid_model = getattr(models, self._lucid_factory)()

    def source_state_dict(self) -> dict[str, object]:
        return {k: v.detach().cpu().numpy() for k, v in self._model.state_dict().items()}

    def target_model(self) -> Module:
        return self._lucid_model

    def map_key(self, src_key: str) -> str | None:
        # Head rename: timm ``head.fc.*`` → Lucid ``classifier.*``.
        if src_key.startswith("head.fc."):
            return "classifier." + src_key[len("head.fc.") :]
        # ``head.flatten`` / ``head.global_pool.*`` carry no parameters
        # in timm's ClassifierHead so they don't appear in the state-dict
        # in the first place — but be defensive.
        if src_key.startswith("head."):
            return None
        return src_key

    def spec(self) -> ConversionSpec:
        factory_name, repo_id, title = _CSPNET_VARIANTS[self.arch]
        config = {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in dataclasses.asdict(self._lucid_model.config).items()
        }

        from lucid.utils.transforms import ImageClassification

        cfg = self._model.default_cfg
        crop = int(cfg["input_size"][1])
        # Floor, as the reference does — rounding puts the two 256px
        # presets at 289 instead of the 288 they were evaluated with.
        resize = int(crop / float(cfg.get("crop_pct", 0.875)))
        preset = ImageClassification(
            crop_size=crop,
            resize_size=resize,
            mean=tuple(float(m) for m in cfg.get("mean", (0.485, 0.456, 0.406))),
            std=tuple(float(s) for s in cfg.get("std", (0.229, 0.224, 0.225))),
            interpolation=str(cfg.get("interpolation", "bilinear")),
        )
        preprocessing = preset.to_dict()

        n_params = int(sum(p.numel() for p in self._model.parameters()))
        # Acc@1 numbers from timm's recipe sweep on the ``ra_in1k`` tag.
        _acc_table = {"cspresnet_50": 76.74, "cspresnext_50": 80.04, "cspdarknet_53": 80.06}
        meta = {
            "num_params": n_params,
            "recipe": str(cfg.get("url", "")),
            "metrics": {"ImageNet-1k": {"acc@1": _acc_table.get(self.arch, 0.0)}},
        }

        return ConversionSpec(
            model_name=factory_name,
            architecture=self.arch,
            repo_id=repo_id,
            tag=self.tag,
            task="image-classification",
            model_type="cspnet",
            source=f"timm/{self._timm_name}",
            license=str(cfg.get("license", "apache-2.0")),
            num_classes=int(self._lucid_model.config.num_classes),
            config=config,
            preprocessing=preprocessing,
            citation=_CSPNET_CITATION,
            title=title,
            paper_url="Wang et al., 2020 — *CSPNet: A New Backbone that "
            "can Enhance Learning Capability of CNN* (arXiv:1911.11929)",
            categories=[],
            datasets=_datasets_from_timm_tag(self.tag),
            meta=meta,
        )


@register_arch("cspresnet_50")
def _r(tag: str) -> Architecture:
    return CSPNetArch("cspresnet_50", tag)


@register_arch("cspresnext_50")
def _x(tag: str) -> Architecture:
    return CSPNetArch("cspresnext_50", tag)


@register_arch("cspdarknet_53")
def _d(tag: str) -> Architecture:
    return CSPNetArch("cspdarknet_53", tag)
