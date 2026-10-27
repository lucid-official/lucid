"""Inception-v4 weight converter — timm → Lucid.

Lucid's :class:`InceptionV4ForImageClassification` was built to mirror
timm's ``inception_v4`` module layout *exactly*: the 22-module
``features`` trunk (``features.0``–``features.5`` stem, then the
Inception-A / Reduction-A / Inception-B / Reduction-B / Inception-C
stacks, each branch's ``ConvNormAct`` exposing ``.conv`` / ``.bn``) and
the ``last_linear`` head.  The two state dicts agree key-for-key and
shape-for-shape (896 tensors each, including the ``num_batches_tracked``
buffers), so the converter is a pure identity map.

Source preset: the ``tf_in1k`` checkpoint (the TensorFlow-Slim weights
ported via Cadene's pretrained-models) evaluates at ``299×299`` with
``crop_pct=0.875`` (→ resize 341), bicubic interpolation, and
``(0.5, 0.5, 0.5)`` mean/std — read straight from timm's ``default_cfg``
below.  Hosted as ``lucid-dl/inception-v4`` tag ``TF_IN1K``; the
family's ``_weights.py`` pins its sha256.
"""

import dataclasses
import math

from lucid.nn import Module
from tools.convert_weights._base import Architecture, ConversionSpec, register_arch

_INCEPTION_V4_CITATION = (
    "@inproceedings{szegedy2017inception,\n"
    "  title={Inception-v4, Inception-ResNet and the Impact of Residual "
    "Connections on Learning},\n"
    "  author={Szegedy, Christian and Ioffe, Sergey and Vanhoucke, "
    "Vincent and Alemi, Alexander A.},\n"
    "  booktitle={AAAI}, year={2017}\n"
    "}"
)

# arch -> (lucid_cls_factory, repo_id, title)
_INCEPTION_V4_VARIANTS: dict[str, tuple[str, str, str]] = {
    "inception_v4": ("inception_v4_cls", "lucid-dl/inception-v4", "Inception-v4")
}


class InceptionV4Arch(Architecture):
    """Converter for the single paper-cited Inception-v4 + tag."""

    def __init__(self, arch: str, tag: str) -> None:
        import timm

        if arch not in _INCEPTION_V4_VARIANTS:
            raise KeyError(f"InceptionV4Arch: unknown arch {arch!r}")
        self.arch = arch
        # Tags ship uppercase per Lucid convention (``TF_IN1K``); timm
        # wants them lowercase (``tf_in1k``).
        self.tag = tag
        self._timm_name = f"{arch}.{tag.lower()}"
        self._model = timm.create_model(self._timm_name, pretrained=True)
        self._model.eval()
        import lucid.models as models

        self._lucid_factory = _INCEPTION_V4_VARIANTS[arch][0]
        self._lucid_model = getattr(models, self._lucid_factory)()

    def source_state_dict(self) -> dict[str, object]:
        return {k: v.detach().cpu().numpy() for k, v in self._model.state_dict().items()}

    def target_model(self) -> Module:
        return self._lucid_model

    def map_key(self, src_key: str) -> str | None:
        # Identity — Lucid mirrors the timm Inception-v4 layout exactly.
        return src_key

    def spec(self) -> ConversionSpec:
        factory_name, repo_id, title = _INCEPTION_V4_VARIANTS[self.arch]
        config = {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in dataclasses.asdict(self._lucid_model.config).items()
        }

        from lucid.utils.transforms import ImageClassification

        cfg = self._model.default_cfg
        crop = int(cfg["input_size"][1])
        # Floored, as the reference's eval transform floors it: 299 / 0.875
        # is 341.7, and 341 — not the rounded 342 — is the resize the
        # published accuracy was measured under.
        resize = int(math.floor(crop / float(cfg.get("crop_pct", 0.875))))
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
            # Top-1 / Top-5 from timm's results CSV for inception_v4.tf_in1k
            # (299 crop, crop_pct 0.875).  The paper's Table 2 figures
            # (80.0 / 95.0) are its own evaluation, not this checkpoint's.
            "metrics": {"ImageNet-1k": {"acc@1": 80.144, "acc@5": 94.982}},
        }

        return ConversionSpec(
            model_name=factory_name,
            architecture=self.arch,
            repo_id=repo_id,
            tag=self.tag,
            task="image-classification",
            model_type="inception_v4",
            source=f"timm/{self._timm_name}",
            license=str(cfg.get("license", "apache-2.0")),
            num_classes=int(self._lucid_model.config.num_classes),
            config=config,
            preprocessing=preprocessing,
            citation=_INCEPTION_V4_CITATION,
            title=title,
            paper_url="Szegedy et al., 2017 — *Inception-v4, Inception-ResNet "
            "and the Impact of Residual Connections on Learning* "
            "(arXiv:1602.07261)",
            categories=[],
            datasets=["imagenet-1k"],
            meta=meta,
        )


@register_arch("inception_v4")
def _build_inception_v4(tag: str) -> Architecture:
    return InceptionV4Arch("inception_v4", tag)
