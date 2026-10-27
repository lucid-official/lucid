"""CrossViT weight converter — timm → Lucid.

Six paper-cited variants (Chen et al., ICCV 2021, Table 2), all sourced
from timm's ``crossvit_<variant>_240.in1k`` tag (Facebook/IBM
ImageNet-1k training; no 22k pretrain).

Module naming was deliberately mirrored when the Lucid model was
rewritten, so the only structural rename is::

    blocks.S.*    →   stages.S.*

Everything else (``patch_embed.{0,1}.proj``, ``cls_token_{0,1}``,
``pos_embed_{0,1}``, per-stage ``blocks/projs/fusion/revert_projs``,
``norm.{0,1}``, ``head.{0,1}``) carries through unchanged.
"""

import dataclasses

from lucid.nn import Module
from tools.convert_weights._base import Architecture, ConversionSpec, register_arch

_CROSSVIT_CITATION = (
    "@inproceedings{chen2021crossvit,\n"
    "  title={CrossViT: Cross-Attention Multi-Scale Vision Transformer for "
    "Image Classification},\n"
    "  author={Chen, Chun-Fu (Richard) and Fan, Quanfu and Panda, Rameswar},\n"
    "  booktitle={ICCV}, year={2021}\n"
    "}"
)

# arch -> (lucid_cls_factory, repo_id, title, paper_param_count_int)
_CROSSVIT_VARIANTS: dict[str, tuple[str, str, str]] = {
    "crossvit_tiny": ("crossvit_tiny_cls", "lucid-dl/crossvit-tiny", "CrossViT-Ti"),
    "crossvit_small": ("crossvit_small_cls", "lucid-dl/crossvit-small", "CrossViT-S"),
    "crossvit_base": ("crossvit_base_cls", "lucid-dl/crossvit-base", "CrossViT-B"),
    "crossvit_9": ("crossvit_9_cls", "lucid-dl/crossvit-9", "CrossViT-9"),
    "crossvit_15": ("crossvit_15_cls", "lucid-dl/crossvit-15", "CrossViT-15"),
    "crossvit_18": ("crossvit_18_cls", "lucid-dl/crossvit-18", "CrossViT-18"),
}


def _datasets_from_timm_tag(tag: str) -> list[str]:
    """Map an uppercase Lucid tag to the dataset list (CrossViT line)."""
    t = tag.lower()
    ds: list[str] = []
    if "in22k" in t:
        ds.append("imagenet-22k")
    if "in1k" in t:
        ds.append("imagenet-1k")
    return ds


class CrossViTArch(Architecture):
    """Converter for one CrossViT variant + tag (always timm-sourced)."""

    def __init__(self, arch: str, tag: str) -> None:
        import timm

        if arch not in _CROSSVIT_VARIANTS:
            raise KeyError(f"CrossViTArch: unknown arch {arch!r}")
        self.arch = arch
        # Lucid keeps the tag uppercase; timm wants lowercase.  The
        # canonical CrossViT tag is ``in1k`` everywhere.
        self.tag = tag
        # timm's model name: ``crossvit_tiny_240.in1k`` (note the ``_240``
        # resolution suffix, which Lucid drops since the variant
        # already encodes 240 via ``image_size``).
        self._timm_name = f"{arch}_240.{tag.lower()}"
        self._model = timm.create_model(self._timm_name, pretrained=True)
        self._model.eval()
        import lucid.models as models

        self._lucid_factory = _CROSSVIT_VARIANTS[arch][0]
        self._lucid_model = getattr(models, self._lucid_factory)()

    def source_state_dict(self) -> dict[str, object]:
        return {k: v.detach().cpu().numpy() for k, v in self._model.state_dict().items()}

    def target_model(self) -> Module:
        return self._lucid_model

    def map_key(self, src_key: str) -> str | None:
        # The only structural rename: timm's outer ``blocks`` → ``stages``.
        if src_key.startswith("blocks."):
            return "stages." + src_key[len("blocks.") :]
        # Everything else maps to itself.
        return src_key

    def spec(self) -> ConversionSpec:
        factory_name, repo_id, title = _CROSSVIT_VARIANTS[self.arch]
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
        # Paper Table 2 ImageNet-1k acc@1 (the IN1K-trained-from-scratch
        # column — no 22k pretrain for CrossViT).
        _acc_table = {
            "crossvit_tiny": 73.4,
            "crossvit_9": 73.9,
            "crossvit_small": 81.0,
            "crossvit_15": 81.5,
            "crossvit_18": 82.5,
            "crossvit_base": 82.2,
        }
        meta = {
            "num_params": n_params,
            "recipe": "Chen et al., ICCV 2021 — paper Table 2 (IN1K column)",
            "metrics": {"ImageNet-1k": {"acc@1": _acc_table.get(self.arch, 0.0)}},
        }

        return ConversionSpec(
            model_name=factory_name,
            architecture=self.arch,
            repo_id=repo_id,
            tag=self.tag,
            task="image-classification",
            model_type="crossvit",
            source=f"timm/{self._timm_name}",
            license=str(cfg.get("license", "apache-2.0")),
            num_classes=int(self._lucid_model.config.num_classes),
            config=config,
            preprocessing=preprocessing,
            citation=_CROSSVIT_CITATION,
            title=title,
            paper_url="Chen et al., 2021 — *CrossViT: Cross-Attention "
            "Multi-Scale Vision Transformer for Image Classification* "
            "(arXiv:2103.14899)",
            categories=[],
            datasets=_datasets_from_timm_tag(self.tag),
            meta=meta,
        )


@register_arch("crossvit_tiny")
def _t(tag: str) -> Architecture:
    return CrossViTArch("crossvit_tiny", tag)


@register_arch("crossvit_small")
def _s(tag: str) -> Architecture:
    return CrossViTArch("crossvit_small", tag)


@register_arch("crossvit_base")
def _b(tag: str) -> Architecture:
    return CrossViTArch("crossvit_base", tag)


@register_arch("crossvit_9")
def _9(tag: str) -> Architecture:
    return CrossViTArch("crossvit_9", tag)


@register_arch("crossvit_15")
def _15(tag: str) -> Architecture:
    return CrossViTArch("crossvit_15", tag)


@register_arch("crossvit_18")
def _18(tag: str) -> Architecture:
    return CrossViTArch("crossvit_18", tag)
