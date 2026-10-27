"""ResNeSt weight converter — timm → Lucid.

The Lucid ResNeSt module + state-dict layout was deliberately mirrored
onto timm's ``resnest*`` checkpoints (deep stem ``conv1.{0,1,3,4,6}`` +
top-level ``bn1``, Split-Attention bottleneck ``layerN.M.conv2.{conv,
bn0,fc1,bn1,fc2}``, AvgPool downsample ``layerN.0.downsample.{1,2}``),
so the converter is a *single* trivial rename for every paper-cited
variant:

    fc.{weight, bias}  →  classifier.{weight, bias}

Everything else carries through unchanged (verified: 0 missing / 0 extra
/ 0 shape mismatches across resnest_50/101/200/269).

Only the four paper-cited depths ship weights.  timm's ``resnest14d`` /
``resnest26d`` are *not* in Zhang et al., 2020 (they are timm-invented
shallow variants), so they are skipped per H11.
"""

import dataclasses

from lucid.nn import Module
from tools.convert_weights._base import Architecture, ConversionSpec, register_arch

_RESNEST_CITATION = (
    "@inproceedings{zhang2022resnest,\n"
    "  title={ResNeSt: Split-Attention Networks},\n"
    "  author={Zhang, Hang and Wu, Chongruo and Zhang, Zhongyue and Zhu, Yi "
    "and Lin, Haibin and Zhang, Zhi and Sun, Yue and He, Tong and Mueller, "
    "Jonas and Manmatha, R. and Li, Mu and Smola, Alexander},\n"
    "  booktitle={CVPR Workshops}, year={2022}\n"
    "}"
)

# arch -> (lucid_cls_factory, repo_id, title)
_RESNEST_VARIANTS: dict[str, tuple[str, str, str]] = {
    "resnest_50": ("resnest_50_cls", "lucid-dl/resnest-50", "ResNeSt-50"),
    "resnest_101": ("resnest_101_cls", "lucid-dl/resnest-101", "ResNeSt-101"),
    "resnest_200": ("resnest_200_cls", "lucid-dl/resnest-200", "ResNeSt-200"),
    "resnest_269": ("resnest_269_cls", "lucid-dl/resnest-269", "ResNeSt-269"),
}

# arch -> timm canonical model name (used to download the source weights).
# The ``d`` / ``e`` suffixes are timm's stem-style markers: ``d`` = deep
# stem (stem_width 32), ``e`` = deep stem with the wider 64-channel
# convolutions.  Both correspond to the paper's "deep stem + avg-down"
# topology that Lucid's config reproduces.
_TIMM_NAMES: dict[str, str] = {
    "resnest_50": "resnest50d",
    "resnest_101": "resnest101e",
    "resnest_200": "resnest200e",
    "resnest_269": "resnest269e",
}

# Paper Table 4 (Zhang et al., 2022) ImageNet-1k top-1 accuracies — the
# exact figures already cited in the Lucid factory docstrings.
_ACC1: dict[str, float] = {
    "resnest_50": 81.1,
    "resnest_101": 82.8,
    "resnest_200": 83.9,
    "resnest_269": 84.5,
}


def _datasets_from_timm_tag(tag: str) -> list[str]:
    """Map an uppercase Lucid tag to its dataset list (ResNeSt line)."""
    t = tag.lower()
    ds: list[str] = []
    if "in22k" in t:
        ds.append("imagenet-22k")
    if "in1k" in t:
        ds.append("imagenet-1k")
    return ds


class ResNeStArch(Architecture):
    """Converter for one paper-cited ResNeSt variant + tag."""

    def __init__(self, arch: str, tag: str) -> None:
        import timm

        if arch not in _RESNEST_VARIANTS:
            raise KeyError(f"ResNeStArch: unknown arch {arch!r}")
        self.arch = arch
        # Tags ship uppercase per Lucid convention (e.g. ``IN1K``);
        # timm wants them lowercase (``in1k``).
        self.tag = tag
        self._timm_name = f"{_TIMM_NAMES[arch]}.{tag.lower()}"
        self._model = timm.create_model(self._timm_name, pretrained=True)
        self._model.eval()
        import lucid.models as models

        self._lucid_factory = _RESNEST_VARIANTS[arch][0]
        self._lucid_model = getattr(models, self._lucid_factory)()

    def source_state_dict(self) -> dict[str, object]:
        return {k: v.detach().cpu().numpy() for k, v in self._model.state_dict().items()}

    def target_model(self) -> Module:
        return self._lucid_model

    def map_key(self, src_key: str) -> str | None:
        # Head rename: timm ``fc.*`` → Lucid ``classifier.*``.
        if src_key.startswith("fc."):
            return "classifier." + src_key[len("fc.") :]
        # Everything else (stem / stages / Split-Attention / downsample)
        # is a verbatim identity map.
        return src_key

    def spec(self) -> ConversionSpec:
        factory_name, repo_id, title = _RESNEST_VARIANTS[self.arch]
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
            "metrics": {"ImageNet-1k": {"acc@1": _ACC1.get(self.arch, 0.0)}},
        }

        return ConversionSpec(
            model_name=factory_name,
            architecture=self.arch,
            repo_id=repo_id,
            tag=self.tag,
            task="image-classification",
            model_type="resnest",
            source=f"timm/{self._timm_name}",
            license=str(cfg.get("license", "apache-2.0")),
            num_classes=int(self._lucid_model.config.num_classes),
            config=config,
            preprocessing=preprocessing,
            citation=_RESNEST_CITATION,
            title=title,
            paper_url="Zhang et al., 2022 — *ResNeSt: Split-Attention "
            "Networks* (arXiv:2004.08955)",
            categories=[],
            datasets=_datasets_from_timm_tag(self.tag),
            meta=meta,
        )


@register_arch("resnest_50")
def _r50(tag: str) -> Architecture:
    return ResNeStArch("resnest_50", tag)


@register_arch("resnest_101")
def _r101(tag: str) -> Architecture:
    return ResNeStArch("resnest_101", tag)


@register_arch("resnest_200")
def _r200(tag: str) -> Architecture:
    return ResNeStArch("resnest_200", tag)


@register_arch("resnest_269")
def _r269(tag: str) -> Architecture:
    return ResNeStArch("resnest_269", tag)
