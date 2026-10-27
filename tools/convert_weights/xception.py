"""Xception weight converter — timm → Lucid.

Lucid's Xception mirrors timm's ``legacy_xception`` module layout
*exactly* (``conv1`` / ``bn1`` / ``conv2`` / ``bn2`` stem, the
``blockN.rep.*`` / ``blockN.skip`` / ``blockN.skipbn`` entry/exit
blocks, the ``blockN.rep.*`` middle blocks, and the exit
``conv3``/``bn3`` / ``conv4``/``bn4`` separable convolutions with
``conv1``/``pointwise`` depthwise/pointwise sub-keys).  The single
difference is the head: timm uses a bare ``fc`` linear, Lucid wraps the
classifier in a ``Sequential(Dropout, Linear)`` so the linear lands at
index 1.

    fc.{weight, bias}  →  classifier.1.{weight, bias}

Everything else carries through unchanged (identity map).

The ``legacy_xception.tf_in1k`` preset is non-standard: a 299×299 crop
with ``crop_pct = 0.8975`` (→ 333 resize), **bicubic** interpolation,
and a symmetric ``mean = std = 0.5`` normalisation (NOT ImageNet
stats).  These are read straight off ``model.default_cfg`` and
replicated exactly in the :class:`ConversionSpec` preprocessing.
"""

import dataclasses

from lucid.nn import Module
from tools.convert_weights._base import Architecture, ConversionSpec, register_arch

_XCEPTION_CITATION = (
    "@inproceedings{chollet2017xception,\n"
    "  title={Xception: Deep Learning with Depthwise Separable "
    "Convolutions},\n"
    "  author={Chollet, Fran\\c{c}ois},\n"
    "  booktitle={CVPR}, year={2017}\n"
    "}"
)

# arch -> (lucid_cls_factory, repo_id, title)
_XCEPTION_VARIANTS: dict[str, tuple[str, str, str]] = {
    "xception": ("xception_cls", "lucid-dl/xception", "Xception")
}

# arch -> timm canonical model name (used to download the source weights)
_TIMM_NAMES: dict[str, str] = {"xception": "legacy_xception"}


class XceptionArch(Architecture):
    """Converter for the paper-cited Xception variant + tag."""

    def __init__(self, arch: str, tag: str) -> None:
        import timm

        if arch not in _XCEPTION_VARIANTS:
            raise KeyError(f"XceptionArch: unknown arch {arch!r}")
        self.arch = arch
        # Tags ship uppercase per Lucid convention (e.g. ``TF_IN1K``);
        # timm wants them lowercase (``tf_in1k``).
        self.tag = tag
        self._timm_name = f"{_TIMM_NAMES[arch]}.{tag.lower()}"
        self._model = timm.create_model(self._timm_name, pretrained=True)
        self._model.eval()
        import lucid.models as models

        self._lucid_factory = _XCEPTION_VARIANTS[arch][0]
        self._lucid_model = getattr(models, self._lucid_factory)()

    def source_state_dict(self) -> dict[str, object]:
        return {k: v.detach().cpu().numpy() for k, v in self._model.state_dict().items()}

    def target_model(self) -> Module:
        return self._lucid_model

    def map_key(self, src_key: str) -> str | None:
        # Head rename: timm bare ``fc.*`` → Lucid ``classifier.1.*``
        # (Lucid wraps the head in Sequential(Dropout, Linear)).
        if src_key.startswith("fc."):
            return "classifier.1." + src_key[len("fc.") :]
        return src_key

    def spec(self) -> ConversionSpec:
        factory_name, repo_id, title = _XCEPTION_VARIANTS[self.arch]
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
            mean=tuple(float(m) for m in cfg.get("mean", (0.5, 0.5, 0.5))),
            std=tuple(float(s) for s in cfg.get("std", (0.5, 0.5, 0.5))),
            interpolation=str(cfg.get("interpolation", "bicubic")),
        )
        preprocessing = preset.to_dict()

        n_params = int(sum(p.numel() for p in self._model.parameters()))
        # Acc@1 from the original Cadene checkpoint that timm re-hosts.
        meta = {
            "num_params": n_params,
            "recipe": str(cfg.get("url", "")),
            "metrics": {"ImageNet-1k": {"acc@1": 79.0}},
        }

        return ConversionSpec(
            model_name=factory_name,
            architecture=self.arch,
            repo_id=repo_id,
            tag=self.tag,
            task="image-classification",
            model_type="xception",
            source=f"timm/{self._timm_name}",
            license=str(cfg.get("license", "apache-2.0")),
            num_classes=int(self._lucid_model.config.num_classes),
            config=config,
            preprocessing=preprocessing,
            citation=_XCEPTION_CITATION,
            title=title,
            paper_url="Chollet, 2017 — *Xception: Deep Learning with "
            "Depthwise Separable Convolutions* (arXiv:1610.02357)",
            categories=[],
            datasets=["imagenet-1k"],
            meta=meta,
        )


@register_arch("xception")
def _xception(tag: str) -> Architecture:
    return XceptionArch("xception", tag)
