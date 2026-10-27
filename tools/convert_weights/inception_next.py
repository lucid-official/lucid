"""InceptionNeXt weight converter — timm → Lucid.

Lucid's InceptionNeXt mirrors timm's ``inception_next_*`` module layout
*exactly* (``stem.0`` / ``stem.1`` patchify stem, ``stages.N.downsample.``
``{0,1}``, ``stages.N.blocks.M.{gamma,token_mixer.dwconv_{hw,w,h},norm,``
``mlp.fc1,mlp.fc2}``, and the ``head.{fc1,norm,fc2}`` MLP classifier), so
the converter is a pure identity map — no ``map_key`` rewrites at all.

Source checkpoints come from timm's ``sail_in1k`` weights (Yu et al.,
"InceptionNeXt: When Inception Meets ConvNeXt", CVPR 2024):

==============================================  ============================
timm                                            Lucid
==============================================  ============================
``stem.0.{w,b}``                (patchify Conv)  ``stem.0.{w,b}``
``stem.1.{w,b,rm,rv,nbt}``      (BatchNorm)      ``stem.1.{...}``
``stages.N.downsample.0.{...}`` (BatchNorm)      ``stages.N.downsample.0.{...}``
``stages.N.downsample.1.{w,b}`` (Conv)           ``stages.N.downsample.1.{w,b}``
``stages.N.blocks.M.gamma``     (layer scale)    ``stages.N.blocks.M.gamma``
``stages.N.blocks.M.token_mixer.dwconv_*``       (identical)
``stages.N.blocks.M.norm.{...}`` (BatchNorm)      (identical)
``stages.N.blocks.M.mlp.fc{1,2}.{w,b}``          (identical)
``head.fc1.{w,b}`` / ``head.norm.{w,b}`` /        (identical)
``head.fc2.{w,b}``
==============================================  ============================

``gamma`` already ships as ``(C,)`` and Lucid stores it the same way, so
no ``transform_value`` reshape is required.  Stage 0 has no downsample
(an :class:`~lucid.nn.Identity`), exactly matching timm.
"""

import dataclasses

from lucid.nn import Module
from tools.convert_weights._base import Architecture, ConversionSpec, register_arch

_INCEPTION_NEXT_CITATION = (
    "@inproceedings{yu2024inceptionnext,\n"
    "  title={InceptionNeXt: When Inception Meets ConvNeXt},\n"
    "  author={Yu, Weihao and Zhou, Pan and Yan, Shuicheng and Wang, "
    "Xinchao},\n"
    "  booktitle={CVPR}, year={2024}\n"
    "}"
)

_INCEPTION_NEXT_PAPER_URL = (
    "Yu et al., 2024 — *InceptionNeXt: When Inception Meets ConvNeXt* " "(arXiv:2303.16900)"
)

# arch -> (lucid_cls_factory, repo_id, title)
_INCEPTION_NEXT_VARIANTS: dict[str, tuple[str, str, str]] = {
    "inception_next_tiny": (
        "inception_next_tiny_cls",
        "lucid-dl/inception-next-tiny",
        "InceptionNeXt-Tiny",
    ),
    "inception_next_small": (
        "inception_next_small_cls",
        "lucid-dl/inception-next-small",
        "InceptionNeXt-Small",
    ),
    "inception_next_base": (
        "inception_next_base_cls",
        "lucid-dl/inception-next-base",
        "InceptionNeXt-Base",
    ),
}


class InceptionNeXtArch(Architecture):
    """Converter for one timm InceptionNeXt variant + tag.

    timm's key layout matches Lucid's exactly, so :meth:`map_key` is a
    pure identity — the safety gates in :func:`convert` still verify the
    1:1 key set, per-key shapes, and a real ``load_state_dict``.
    """

    def __init__(self, arch: str, tag: str) -> None:
        import timm

        if arch not in _INCEPTION_NEXT_VARIANTS:
            raise KeyError(f"InceptionNeXtArch: unknown arch {arch!r}")
        self.arch = arch
        # Lucid keeps weight-enum tags uppercase (``SAIL_IN1K``); timm's
        # registry is lowercase, so the lookup uses a lowered copy.
        self.tag = tag
        self._timm_name = f"{arch}.{tag.lower()}"
        self._model = timm.create_model(self._timm_name, pretrained=True)
        self._model.eval()

        import lucid.models as models

        # Use the registry (create_model) rather than a top-level attribute
        # so newly-added family-local factories are reachable without a
        # shared lucid/models/__init__.py hoist.
        self._lucid_factory = _INCEPTION_NEXT_VARIANTS[arch][0]
        self._lucid_model = models.create_model(self._lucid_factory)

    def source_state_dict(self) -> dict[str, object]:
        return {k: v.detach().cpu().numpy() for k, v in self._model.state_dict().items()}

    def target_model(self) -> Module:
        return self._lucid_model

    def map_key(self, src_key: str) -> str | None:
        # Identity — Lucid mirrors the timm InceptionNeXt layout exactly.
        return src_key

    def spec(self) -> ConversionSpec:
        factory_name, repo_id, title = _INCEPTION_NEXT_VARIANTS[self.arch]
        config = {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in dataclasses.asdict(self._lucid_model.config).items()
        }

        cfg = self._model.default_cfg
        from lucid.utils.transforms import ImageClassification

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

        meta = {
            "num_params": int(sum(p.numel() for p in self._model.parameters())),
            "recipe": str(cfg.get("url", "")),
            "metrics": {"ImageNet-1k": dict(_ACC1.get(self.arch, {}))},
        }

        return ConversionSpec(
            model_name=factory_name,
            architecture=self.arch,
            repo_id=repo_id,
            tag=self.tag,
            task="image-classification",
            model_type="inception_next",
            source=f"timm/{self._timm_name}",
            license=str(cfg.get("license", "apache-2.0")),
            num_classes=int(self._lucid_model.config.num_classes),
            config=config,
            preprocessing=preprocessing,
            citation=_INCEPTION_NEXT_CITATION,
            title=title,
            paper_url=_INCEPTION_NEXT_PAPER_URL,
            categories=[],
            datasets=["imagenet-1k"],
            meta=meta,
        )


# Paper-cited ImageNet-1k top-1/top-5 accuracies (Yu et al., 2024, Table 2).
_ACC1: dict[str, dict[str, float]] = {
    "inception_next_tiny": {"acc@1": 82.3, "acc@5": 96.1},
    "inception_next_small": {"acc@1": 83.5, "acc@5": 96.6},
    "inception_next_base": {"acc@1": 84.0, "acc@5": 96.9},
}


@register_arch("inception_next_tiny")
def _build_inception_next_tiny(tag: str) -> Architecture:
    return InceptionNeXtArch("inception_next_tiny", tag)


@register_arch("inception_next_small")
def _build_inception_next_small(tag: str) -> Architecture:
    return InceptionNeXtArch("inception_next_small", tag)


@register_arch("inception_next_base")
def _build_inception_next_base(tag: str) -> Architecture:
    return InceptionNeXtArch("inception_next_base", tag)
