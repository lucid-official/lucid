"""MaskFormer semantic-segmentation weight converter — HF transformers → Lucid.

Two paper-cited ADE20k variants (Cheng et al., NeurIPS 2021):
``maskformer_resnet50`` / ``maskformer_resnet101``.  Source = the
``facebook/maskformer-resnet{50,101}-ade``
:class:`MaskFormerForInstanceSegmentation` checkpoints (150 ADE20k
classes; ``class_predictor`` is ``(151, 256)`` = 150 foreground + 1
no-object).

Lucid's MaskFormer was rebuilt to mirror the reference module layout
verbatim, so the mapping is almost a pure identity.  Three families of
keys differ — all pure prefix manipulations:

================================================  =============================
HF MaskFormerForInstanceSegmentation              Lucid
================================================  =============================
``model.pixel_level_module.<rest>``               ``pixel_level_module.<rest>``
``model.transformer_module.<rest>``               ``transformer_module.<rest>``
``class_predictor.*`` / ``mask_embedder.*``       identity
``criterion.empty_weight``                        DROP (not a model param)
================================================  =============================

A full key/shape diff shows **511** HF keys vs **510** Lucid keys; after
dropping ``criterion.empty_weight`` the 510 remaining keys map 1:1 with
**zero** shape mismatches.  The backbone is a regular-BatchNorm HF-style
ResNet (``embedder.embedder`` stem + ``encoder.stages.{s}.layers.{l}``
bottlenecks, stride on the 3x3 conv), the FPN pixel decoder uses
conv + GroupNorm pairs, and the DETR-style decoder uses separate
``q/k/v/o_proj`` attentions — all replicated 1:1 in Lucid's submodule
names so ``map_key`` only strips the ``model.`` joiner prefix.
"""

import dataclasses

from lucid.nn import Module
from tools.convert_weights._base import Architecture, ConversionSpec, register_arch

_MASKFORMER_CITATION = (
    "@inproceedings{cheng2021maskformer,\n"
    "  title={Per-Pixel Classification is Not All You Need for "
    "Semantic Segmentation},\n"
    "  author={Cheng, Bowen and Schwing, Alexander G. and Kirillov, "
    "Alexander},\n"
    "  booktitle={Advances in Neural Information Processing Systems "
    "(NeurIPS)},\n"
    "  year={2021}\n"
    "}"
)

_MASKFORMER_VARIANTS: dict[str, tuple[str, str, str, str, float]] = {
    # arch -> (lucid_factory, repo_id, title, hf_source, ADE20k mIoU)
    "maskformer_resnet50": (
        "maskformer_resnet50",
        "lucid-dl/maskformer-resnet-50",
        "MaskFormer (ResNet-50)",
        "facebook/maskformer-resnet50-ade",
        44.5,
    ),
    "maskformer_resnet101": (
        "maskformer_resnet101",
        "lucid-dl/maskformer-resnet-101",
        "MaskFormer (ResNet-101)",
        "facebook/maskformer-resnet101-ade",
        45.5,
    ),
}


class MaskFormerArch(Architecture):
    """Converter for one HF MaskFormer ADE20k variant + tag."""

    def __init__(self, arch: str, tag: str) -> None:
        from transformers import MaskFormerForInstanceSegmentation

        if arch not in _MASKFORMER_VARIANTS:
            raise KeyError(f"MaskFormerArch: unknown arch {arch!r}")
        self.arch = arch
        self.tag = tag
        _, _, _, hf_source, _ = _MASKFORMER_VARIANTS[arch]
        self._hf_source = hf_source
        self._model = MaskFormerForInstanceSegmentation.from_pretrained(hf_source)
        self._model.eval()

        import lucid.models as models

        self._lucid_factory = _MASKFORMER_VARIANTS[arch][0]
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
        # Pixel-level module + transformer module: strip the ``model.`` joiner.
        if src_key.startswith("model.pixel_level_module."):
            return "pixel_level_module." + src_key[len("model.pixel_level_module.") :]
        if src_key.startswith("model.transformer_module."):
            return "transformer_module." + src_key[len("model.transformer_module.") :]
        # class_predictor / mask_embedder → identity.
        return src_key

    def spec(self) -> ConversionSpec:
        factory_name, repo_id, title, hf_source, miou = _MASKFORMER_VARIANTS[self.arch]
        config = {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in dataclasses.asdict(self._lucid_model.config).items()
        }

        from transformers import MaskFormerImageProcessor

        from lucid.utils.transforms import Segmentation

        # Source the ImageNet normalisation stats from the upstream
        # image processor; geometry follows the canonical 512² ADE20k
        # semantic eval crop.
        ip = MaskFormerImageProcessor.from_pretrained(hf_source)
        preset = Segmentation(
            crop_size=512,
            resize_size=512,
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
            model_type="maskformer",
            source=hf_source,
            license="other",
            num_classes=150,
            config=config,
            preprocessing=preprocessing,
            citation=_MASKFORMER_CITATION,
            title=title,
            paper_url="Cheng et al., 2021 — *Per-Pixel Classification is "
            "Not All You Need for Semantic Segmentation* (arXiv:2107.06278)",
            categories=[],
            datasets=["ade20k"],
            meta=meta,
        )


@register_arch("maskformer_resnet50")
def _mf50(tag: str) -> Architecture:
    return MaskFormerArch("maskformer_resnet50", tag)


@register_arch("maskformer_resnet101")
def _mf101(tag: str) -> Architecture:
    return MaskFormerArch("maskformer_resnet101", tag)
