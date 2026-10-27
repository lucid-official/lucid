"""Mask2Former semantic-segmentation weight converter — HF transformers → Lucid.

Four paper-cited Swin-backbone ADE20k variants (Cheng et al., CVPR 2022):
``mask2former_swin_{tiny,small,base,large}``.  Source = the
``facebook/mask2former-swin-{tiny,small,base,large}-ade-semantic``
:class:`Mask2FormerForUniversalSegmentation` checkpoints (150 ADE20k
classes; ``class_predictor`` is ``(151, 256)`` = 150 foreground + 1
no-object).

Lucid's Mask2Former was rebuilt to mirror the reference module layout
verbatim, so the mapping is almost a pure identity.  The differing keys are
all pure prefix manipulations / drops:

================================================  =============================
HF Mask2FormerForUniversalSegmentation            Lucid
================================================  =============================
``model.pixel_level_module.<rest>``               ``pixel_level_module.<rest>``
``model.transformer_module.<rest>``               ``transformer_module.<rest>``
``class_predictor.*``                             identity
``criterion.empty_weight``                        DROP (loss buffer)
``*.relative_position_index``                     DROP (non-persistent buffer)
``*.swin.layernorm.*``                            DROP (unused by Mask2Former)
================================================  =============================

A full key/shape diff (swin-tiny) shows **556** HF keys vs **553** Lucid
keys; after dropping ``criterion.empty_weight`` (1), the 12 non-persistent
``relative_position_index`` buffers, and the 2 unused ``swin.layernorm``
tensors, the 553 remaining keys map 1:1 with **zero** shape mismatches.
The cross-attention ``out_proj.{weight,bias}`` keys are consumed by Lucid's
``nn.MultiheadAttention`` load hook (which renames them to the flat
``out_proj_weight`` / ``out_proj_bias`` attributes).
"""

import dataclasses

from lucid.nn import Module
from tools.convert_weights._base import Architecture, ConversionSpec, register_arch

_MASK2FORMER_CITATION = (
    "@inproceedings{cheng2022mask2former,\n"
    "  title={Masked-attention Mask Transformer for Universal Image "
    "Segmentation},\n"
    "  author={Cheng, Bowen and Misra, Ishan and Schwing, Alexander G. "
    "and Kirillov, Alexander and Girdhar, Rohit},\n"
    "  booktitle={Proceedings of the IEEE/CVF Conference on Computer "
    "Vision and Pattern Recognition (CVPR)},\n"
    "  year={2022}\n"
    "}"
)

# arch -> (lucid_factory, repo_id, title, hf_source, ADE20k mIoU)
_MASK2FORMER_VARIANTS: dict[str, tuple[str, str, str, str, float]] = {
    "mask2former_swin_tiny": (
        "mask2former_swin_tiny",
        "lucid-dl/mask2former-swin-tiny-ade",
        "Mask2Former (Swin-Tiny)",
        "facebook/mask2former-swin-tiny-ade-semantic",
        47.7,
    ),
    "mask2former_swin_small": (
        "mask2former_swin_small",
        "lucid-dl/mask2former-swin-small-ade",
        "Mask2Former (Swin-Small)",
        "facebook/mask2former-swin-small-ade-semantic",
        51.3,
    ),
    "mask2former_swin_base": (
        "mask2former_swin_base",
        "lucid-dl/mask2former-swin-base-ade",
        "Mask2Former (Swin-Base)",
        "facebook/mask2former-swin-base-ade-semantic",
        53.9,
    ),
    "mask2former_swin_large": (
        "mask2former_swin_large",
        "lucid-dl/mask2former-swin-large-ade",
        "Mask2Former (Swin-Large)",
        "facebook/mask2former-swin-large-ade-semantic",
        56.1,
    ),
}


class Mask2FormerArch(Architecture):
    """Converter for one HF Mask2Former ADE20k Swin variant + tag."""

    def __init__(self, arch: str, tag: str) -> None:
        from transformers import Mask2FormerForUniversalSegmentation

        if arch not in _MASK2FORMER_VARIANTS:
            raise KeyError(f"Mask2FormerArch: unknown arch {arch!r}")
        self.arch = arch
        self.tag = tag
        _, _, _, hf_source, _ = _MASK2FORMER_VARIANTS[arch]
        self._hf_source = hf_source
        self._model = Mask2FormerForUniversalSegmentation.from_pretrained(hf_source)
        self._model.eval()

        import lucid.models as models

        self._lucid_factory = _MASK2FORMER_VARIANTS[arch][0]
        # ADE20k checkpoint = 150 foreground classes.
        self._lucid_model = getattr(models, self._lucid_factory)(num_classes=150)

    def source_state_dict(self) -> dict[str, object]:
        return {k: v.detach().cpu().numpy() for k, v in self._model.state_dict().items()}

    def target_model(self) -> Module:
        return self._lucid_model

    def map_key(self, src_key: str) -> str | None:
        # criterion.empty_weight is a loss buffer, not a model parameter.
        if src_key.startswith("criterion."):
            return None
        # relative_position_index is a non-persistent buffer (recreated on load).
        if "relative_position_index" in src_key:
            return None
        # swin.layernorm is newly-init in the seg checkpoint and unused by
        # Mask2Former — Lucid omits it.
        if src_key.endswith("swin.layernorm.weight") or src_key.endswith("swin.layernorm.bias"):
            return None
        # Pixel-level module + transformer module: strip the ``model.`` joiner.
        if src_key.startswith("model.pixel_level_module."):
            return "pixel_level_module." + src_key[len("model.pixel_level_module.") :]
        if src_key.startswith("model.transformer_module."):
            mapped = "transformer_module." + src_key[len("model.transformer_module.") :]
            # Lucid's nn.MultiheadAttention stores the output projection as flat
            # ``out_proj_weight`` / ``out_proj_bias`` attributes (its load hook
            # also accepts the sub-module ``out_proj.*`` names, but the
            # converter's key-set check runs against the flat state_dict keys).
            mapped = mapped.replace(".cross_attn.out_proj.weight", ".cross_attn.out_proj_weight")
            mapped = mapped.replace(".cross_attn.out_proj.bias", ".cross_attn.out_proj_bias")
            return mapped
        # class_predictor → identity.
        return src_key

    def spec(self) -> ConversionSpec:
        factory_name, repo_id, title, hf_source, miou = _MASK2FORMER_VARIANTS[self.arch]
        config = {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in dataclasses.asdict(self._lucid_model.config).items()
        }

        from transformers import Mask2FormerImageProcessor

        from lucid.utils.transforms import Segmentation

        # Source the ImageNet normalisation stats from the upstream image
        # processor; geometry follows the 384² ADE20k semantic eval crop.
        ip = Mask2FormerImageProcessor.from_pretrained(hf_source)
        preset = Segmentation(
            crop_size=384,
            resize_size=384,
            mean=tuple(float(m) for m in ip.image_mean),
            std=tuple(float(s) for s in ip.image_std),
        )
        preprocessing = preset.to_dict()

        meta = {
            "num_params": int(sum(p.numel() for p in self._model.parameters())),
            "recipe": f"{hf_source} (ADE20k semantic, 150 classes)",
            "metrics": {"ADE20K": {"mIoU": miou}},
        }

        return ConversionSpec(
            model_name=factory_name,
            architecture=self.arch,
            repo_id=repo_id,
            tag=self.tag,
            task="semantic-segmentation",
            model_type="mask2former",
            source=hf_source,
            license="other",
            num_classes=150,
            config=config,
            preprocessing=preprocessing,
            citation=_MASK2FORMER_CITATION,
            title=title,
            paper_url="Cheng et al., 2022 — *Masked-attention Mask "
            "Transformer for Universal Image Segmentation* (arXiv:2112.01527)",
            categories=[],
            datasets=["ade20k"],
            meta=meta,
        )


@register_arch("mask2former_swin_tiny")
def _m2f_tiny(tag: str) -> Architecture:
    return Mask2FormerArch("mask2former_swin_tiny", tag)


@register_arch("mask2former_swin_small")
def _m2f_small(tag: str) -> Architecture:
    return Mask2FormerArch("mask2former_swin_small", tag)


@register_arch("mask2former_swin_base")
def _m2f_base(tag: str) -> Architecture:
    return Mask2FormerArch("mask2former_swin_base", tag)


@register_arch("mask2former_swin_large")
def _m2f_large(tag: str) -> Architecture:
    return Mask2FormerArch("mask2former_swin_large", tag)
