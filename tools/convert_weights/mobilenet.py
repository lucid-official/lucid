"""MobileNet v1 weight converter — timm → Lucid.

timm's ``mobilenetv1_100`` keeps the depthwise-separable body in nested
``blocks.G.B`` groups while fusing each BatchNorm with its activation in
a ``BatchNormAct2d``; Lucid flattens the whole network into one
``features`` Sequential where the stem occupies slots 0-2 and the 13
depthwise+pointwise blocks occupy slots 3-15 (each block being a
six-slot Sequential ``dw-conv / dw-bn / act / pw-conv / pw-bn / act``).
The 13 timm ``(group, block)`` pairs enumerate in stride order
identically to the paper's 13 DW+PW layers, so the flat enumeration
index ``i`` maps directly to Lucid ``features.{3 + i}``:

==================================  =========================
timm                                Lucid
==================================  =========================
``conv_stem.weight``                ``features.0.weight``
``bn1.{w,b,rm,rv,nbt}``             ``features.1.{w,b,rm,rv,nbt}``
``blocks.G.B.conv_dw.weight``       ``features.{3+i}.0.weight``
``blocks.G.B.bn1.{...}``            ``features.{3+i}.1.{...}``
``blocks.G.B.conv_pw.weight``       ``features.{3+i}.3.weight``
``blocks.G.B.bn2.{...}``            ``features.{3+i}.4.{...}``
``classifier.{weight,bias}``        ``classifier.{weight,bias}``
==================================  =========================

The activation-parity blocker this file used to warn about is resolved
in both directions: ``MobileNet`` uses :class:`~lucid.nn.ReLU6`
throughout (stem and both activations of every separable block), matching
the timm checkpoint and the TensorFlow reference, and the converted
weights *are* wired — ``_weights.py`` carries a pinned sha256 and
``_pretrained.py`` resolves and loads it.
"""

import dataclasses
import re

from lucid.nn import Module
from tools.convert_weights._base import (
    Architecture,
    ConversionSpec,
    UnknownSourceKey,
    register_arch,
)

_MOBILENET_V1_CITATION = (
    "@article{howard2017mobilenets,\n"
    "  title={MobileNets: Efficient Convolutional Neural Networks for "
    "Mobile Vision Applications},\n"
    "  author={Howard, Andrew G. and Zhu, Menglong and Chen, Bo and "
    "Kalenichenko, Dmitry and Wang, Weijun and Weyand, Tobias and "
    "Andreetto, Marco and Adam, Hartwig},\n"
    "  journal={arXiv preprint arXiv:1704.04861}, year={2017}\n"
    "}"
)

_MOBILENET_V1_PAPER_URL = (
    "Howard et al., 2017 — *MobileNets: Efficient Convolutional Neural "
    "Networks for Mobile Vision Applications* (arXiv:1704.04861)"
)

# arch -> (lucid_cls_factory, repo_id, title, timm_model_name)
_MOBILENET_V1_VARIANTS: dict[str, tuple[str, str, str, str]] = {
    "mobilenet": (
        "mobilenet_cls",
        "lucid-dl/mobilenet-v1",
        "MobileNet V1",
        "mobilenetv1_100.ra4_e3600_r224_in1k",
    ),
}

_STEM_DW_PW_BLOCKS = 13  # paper Table 1 — 13 depthwise+pointwise blocks


class MobileNetArch(Architecture):
    """Converter for one timm MobileNet-v1 variant + tag."""

    def __init__(self, arch: str, tag: str) -> None:
        import timm

        if arch not in _MOBILENET_V1_VARIANTS:
            raise KeyError(f"MobileNetArch: unknown arch {arch!r}")
        self.arch = arch
        self.tag = tag
        self._timm_name = _MOBILENET_V1_VARIANTS[arch][3]
        self._model = timm.create_model(self._timm_name, pretrained=True)
        self._model.eval()
        self._gb_to_feat = self._build_block_index()

    def _build_block_index(self) -> dict[tuple[int, int], int]:
        """Map each timm ``(group, block)`` pair to a Lucid feature slot.

        The pairs enumerate in stride order (matching the paper's 13
        DW+PW layers), so the i-th distinct pair lands at
        ``features.{3 + i}``.
        """
        seen: list[tuple[int, int]] = []
        for key in self._model.state_dict():
            match = re.match(r"blocks\.(\d+)\.(\d+)\.", key)
            if match:
                pair = (int(match.group(1)), int(match.group(2)))
                if pair not in seen:
                    seen.append(pair)
        seen.sort()
        return {pair: 3 + idx for idx, pair in enumerate(seen)}

    def source_state_dict(self) -> dict[str, object]:
        return {
            k: v.detach().cpu().numpy() for k, v in self._model.state_dict().items()
        }

    def target_model(self) -> Module:
        import lucid.models as models

        factory = _MOBILENET_V1_VARIANTS[self.arch][0]
        return getattr(models, factory)()

    def map_key(self, src_key: str) -> str | None:
        # Stem: conv_stem -> features.0 ; bn1.* -> features.1.*
        if src_key == "conv_stem.weight":
            return "features.0.weight"
        if src_key.startswith("bn1."):
            return "features.1." + src_key[len("bn1.") :]
        # Head linear is identical in both layouts.
        if src_key.startswith("classifier."):
            return "classifier." + src_key[len("classifier.") :]
        # Depthwise+pointwise blocks: blocks.G.B.* -> features.{3+i}.*
        match = re.match(r"blocks\.(\d+)\.(\d+)\.(.*)", src_key)
        if match:
            group = int(match.group(1))
            block = int(match.group(2))
            rest = match.group(3)
            feat = self._gb_to_feat.get((group, block))
            if feat is None:
                raise UnknownSourceKey(src_key)
            if rest.startswith("conv_dw."):
                return f"features.{feat}.0." + rest[len("conv_dw.") :]
            if rest.startswith("bn1."):
                return f"features.{feat}.1." + rest[len("bn1.") :]
            if rest.startswith("conv_pw."):
                return f"features.{feat}.3." + rest[len("conv_pw.") :]
            if rest.startswith("bn2."):
                return f"features.{feat}.4." + rest[len("bn2.") :]
        raise UnknownSourceKey(src_key)

    def spec(self) -> ConversionSpec:
        import lucid.models as models

        factory_name, repo_id, title, _ = _MOBILENET_V1_VARIANTS[self.arch]
        model = getattr(models, factory_name)()
        config = {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in dataclasses.asdict(model.config).items()
        }

        cfg = self._model.default_cfg
        from lucid.utils.transforms import ImageClassification

        crop = int(cfg["input_size"][1])
        resize = int(round(crop / float(cfg.get("crop_pct", 0.875))))
        preset = ImageClassification(
            crop_size=crop,
            resize_size=resize,
            mean=tuple(float(m) for m in cfg.get("mean", (0.5, 0.5, 0.5))),
            std=tuple(float(s) for s in cfg.get("std", (0.5, 0.5, 0.5))),
            interpolation=str(cfg.get("interpolation", "bicubic")),
        )
        preprocessing = preset.to_dict()

        meta = {
            "num_params": int(sum(p.numel() for p in self._model.parameters())),
            "recipe": str(cfg.get("hf_hub_id", "")),
            "metrics": {"ImageNet-1k": {"acc@1": 0.0, "acc@5": 0.0}},
        }

        return ConversionSpec(
            model_name=factory_name,
            architecture=self.arch,
            repo_id=repo_id,
            tag=self.tag,
            task="image-classification",
            model_type="mobilenet",
            source=f"timm/{self._timm_name}",
            license=str(cfg.get("license", "apache-2.0")),
            num_classes=int(model.config.num_classes),
            config=config,
            preprocessing=preprocessing,
            citation=_MOBILENET_V1_CITATION,
            title=title,
            paper_url=_MOBILENET_V1_PAPER_URL,
            categories=[],
            datasets=["imagenet-1k"],
            meta=meta,
        )


@register_arch("mobilenet")
def _build_mobilenet(tag: str) -> Architecture:
    return MobileNetArch("mobilenet", tag)
