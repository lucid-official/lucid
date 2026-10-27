"""Inception-ResNet v2 weight converter — timm → Lucid.

Lucid's :class:`InceptionResNetV2ForImageClassification` was built to
mirror timm's ``inception_resnet_v2`` module layout *exactly*: the flat
``conv2d_{1a,2a,2b,3b,4a,7b}`` stem / projection ConvNormAct keys, the
``mixed_5b`` / ``mixed_6a`` / ``mixed_7a`` Inception blocks, the
``repeat`` / ``repeat_1`` / ``repeat_2`` residual stacks, the trailing
``block8`` no-ReLU block, and the ``classif`` linear head (note the
``classif`` spelling, not ``classifier`` — kept for TF-Slim parity).

The two state dicts agree key-for-key and shape-for-shape (1306 tensors
each, including the ``num_batches_tracked`` buffers), so the converter
is a pure identity map — no ``map_key`` rewrites at all.

Source preset is *not* the ImageNet default: the ``tf_in1k`` checkpoint
trained at ``299×299`` with ``crop_pct=0.8975`` (→ resize 333), bicubic
interpolation, and ``(0.5, 0.5, 0.5)`` mean/std (the TensorFlow-Slim
``[-1, 1]`` normalisation).  These are read straight from timm's
``default_cfg`` and replicated in the ConversionSpec.
"""

import dataclasses

from lucid.nn import Module
from tools.convert_weights._base import Architecture, ConversionSpec, register_arch

_INCEPTION_RESNET_CITATION = (
    "@inproceedings{szegedy2017inception,\n"
    "  title={Inception-v4, Inception-ResNet and the Impact of Residual "
    "Connections on Learning},\n"
    "  author={Szegedy, Christian and Ioffe, Sergey and Vanhoucke, "
    "Vincent and Alemi, Alexander A.},\n"
    "  booktitle={AAAI}, year={2017}\n"
    "}"
)

# arch -> (lucid_cls_factory, repo_id, title)
_INCEPTION_RESNET_VARIANTS: dict[str, tuple[str, str, str]] = {
    "inception_resnet_v2": (
        "inception_resnet_v2_cls",
        "lucid-dl/inception-resnet-v2",
        "Inception-ResNet v2",
    )
}

# arch -> timm canonical model name (used to download the source weights)
_TIMM_NAMES: dict[str, str] = {"inception_resnet_v2": "inception_resnet_v2"}


class InceptionResNetArch(Architecture):
    """Converter for one paper-cited Inception-ResNet variant + tag."""

    def __init__(self, arch: str, tag: str) -> None:
        import timm

        if arch not in _INCEPTION_RESNET_VARIANTS:
            raise KeyError(f"InceptionResNetArch: unknown arch {arch!r}")
        self.arch = arch
        # Tags ship uppercase per Lucid convention (e.g. ``TF_IN1K``);
        # timm wants them lowercase (``tf_in1k``).
        self.tag = tag
        self._timm_name = f"{_TIMM_NAMES[arch]}.{tag.lower()}"
        self._model = timm.create_model(self._timm_name, pretrained=True)
        self._model.eval()
        import lucid.models as models

        self._lucid_factory = _INCEPTION_RESNET_VARIANTS[arch][0]
        self._lucid_model = getattr(models, self._lucid_factory)()

    def source_state_dict(self) -> dict[str, object]:
        return {k: v.detach().cpu().numpy() for k, v in self._model.state_dict().items()}

    def target_model(self) -> Module:
        return self._lucid_model

    def map_key(self, src_key: str) -> str | None:
        # Identity — Lucid mirrors the timm Inception-ResNet layout exactly.
        return src_key

    def spec(self) -> ConversionSpec:
        factory_name, repo_id, title = _INCEPTION_RESNET_VARIANTS[self.arch]
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
            # Top-1 / Top-5 from timm's results CSV for inception_resnet_v2.tf_in1k.
            "metrics": {"ImageNet-1k": {"acc@1": 80.46, "acc@5": 95.31}},
        }

        return ConversionSpec(
            model_name=factory_name,
            architecture=self.arch,
            repo_id=repo_id,
            tag=self.tag,
            task="image-classification",
            model_type="inception_resnet",
            source=f"timm/{self._timm_name}",
            license=str(cfg.get("license", "apache-2.0")),
            num_classes=int(self._lucid_model.config.num_classes),
            config=config,
            preprocessing=preprocessing,
            citation=_INCEPTION_RESNET_CITATION,
            title=title,
            paper_url="Szegedy et al., 2017 — *Inception-v4, Inception-ResNet "
            "and the Impact of Residual Connections on Learning* "
            "(arXiv:1602.07261)",
            categories=[],
            datasets=["imagenet-1k"],
            meta=meta,
        )


@register_arch("inception_resnet_v2")
def _build_inception_resnet_v2(tag: str) -> Architecture:
    return InceptionResNetArch("inception_resnet_v2", tag)
