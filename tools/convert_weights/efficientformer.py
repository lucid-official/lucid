"""EfficientFormer weight converter — timm -> Lucid.

Three paper-cited variants (Li et al., NeurIPS 2022): ``efficientformer_l1``
/ ``efficientformer_l3`` / ``efficientformer_l7``.  Source = timm's
``efficientformer_l{1,3,7}.snap_dist_in1k`` snapshot-distilled checkpoints.

Lucid's EfficientFormer module tree was rebuilt to mirror the timm key
naming verbatim — top-level ``stem.{conv1,norm1,conv2,norm2}``,
``stages.S.downsample.{conv,norm}``, ``stages.S.blocks.B.*`` (with the
parameter-free ``Flat`` placeholder occupying the same block index so the
4-D -> 3-D transition lines up), ``norm`` (channel-last LayerNorm), and the
distilled dual head ``head`` / ``head_dist``.  The mapping is therefore a
pure identity with a single drop:

==================================================  =========================
timm                                                Lucid
==================================================  =========================
``stem.*``                                          ``stem.*``    (identity)
``stages.S.downsample.*``                           identity
``stages.S.blocks.B.*``                             identity
``stages.S.blocks.B.token_mixer.attention_biases``  identity
``...token_mixer.attention_bias_idxs``              dropped (non-persistent
                                                    buffer in Lucid; derived)
``norm.*``                                          ``norm.*``    (identity)
``head.*`` / ``head_dist.*``                         identity
==================================================  =========================

Both heads are kept (the checkpoints are distilled); Lucid averages
``head`` and ``head_dist`` at inference, matching the reference eval path.
"""

import dataclasses

from lucid.nn import Module
from tools.convert_weights._base import Architecture, ConversionSpec, register_arch

_EF_CITATION = (
    "@article{li2022efficientformer,\n"
    "  title={EfficientFormer: Vision Transformers at MobileNet Speed},\n"
    "  author={Li, Yanyu and Yuan, Geng and Wen, Yang and Hu, Eric and "
    "Evangelidis, Georgios and Tulyakov, Sergey and Wang, Yanzhi and Ren, "
    "Jian},\n"
    "  journal={Advances in Neural Information Processing Systems},\n"
    "  volume={35}, year={2022}\n"
    "}"
)

_EF_VARIANTS: dict[str, tuple[str, str, str]] = {
    # arch -> (lucid_cls_factory, repo_id, title)
    "efficientformer_l1": (
        "efficientformer_l1_cls",
        "lucid-dl/efficientformer-l1",
        "EfficientFormer-L1",
    ),
    "efficientformer_l3": (
        "efficientformer_l3_cls",
        "lucid-dl/efficientformer-l3",
        "EfficientFormer-L3",
    ),
    "efficientformer_l7": (
        "efficientformer_l7_cls",
        "lucid-dl/efficientformer-l7",
        "EfficientFormer-L7",
    ),
}

# Paper Table 4 top-1 accuracies @ 224x224 on ImageNet-1k (distilled).
_ACC1: dict[str, float] = {
    "efficientformer_l1": 79.2,
    "efficientformer_l3": 82.4,
    "efficientformer_l7": 83.3,
}


class EfficientFormerArch(Architecture):
    """Converter for one timm EfficientFormer variant + tag."""

    def __init__(self, arch: str, tag: str) -> None:
        import timm

        if arch not in _EF_VARIANTS:
            raise KeyError(f"EfficientFormerArch: unknown arch {arch!r}")
        self.arch = arch
        # Lucid keeps weight-enum tags uppercase across all families; timm's
        # registry is lowercase, so the lookup uses a lowered copy.
        self.tag = tag
        self._timm_name = f"{arch}.{tag.lower()}"
        self._model = timm.create_model(self._timm_name, pretrained=True)
        self._model.eval()

        import lucid.models as models

        self._lucid_factory = _EF_VARIANTS[arch][0]
        self._lucid_model = getattr(models, self._lucid_factory)()

    def source_state_dict(self) -> dict[str, object]:
        return {k: v.detach().cpu().numpy() for k, v in self._model.state_dict().items()}

    def target_model(self) -> Module:
        return self._lucid_model

    def map_key(self, src_key: str) -> str | None:
        # The relative-position index grid is a fixed, derived buffer that
        # Lucid recreates at construction time (non-persistent), so drop it.
        if src_key.endswith("attention_bias_idxs"):
            return None
        # Everything else is an identity map — Lucid mirrors the timm
        # EfficientFormer key layout, distilled dual head included.
        return src_key

    def spec(self) -> ConversionSpec:
        factory_name, repo_id, title = _EF_VARIANTS[self.arch]
        config = {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in dataclasses.asdict(self._lucid_model.config).items()
        }

        cfg = self._model.default_cfg
        from lucid.utils.transforms import ImageClassification

        crop = int(cfg["input_size"][1])
        resize = int(round(crop / float(cfg.get("crop_pct", 0.95))))
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
            "metrics": {"ImageNet-1k": {"acc@1": _ACC1.get(self.arch, 0.0)}},
        }

        return ConversionSpec(
            model_name=factory_name,
            architecture=self.arch,
            repo_id=repo_id,
            tag=self.tag,
            task="image-classification",
            model_type="efficientformer",
            source=f"timm/{self._timm_name}",
            license=str(cfg.get("license", "apache-2.0")),
            num_classes=int(self._lucid_model.config.num_classes),
            config=config,
            preprocessing=preprocessing,
            citation=_EF_CITATION,
            title=title,
            paper_url="Li et al., 2022 — *EfficientFormer: Vision "
            "Transformers at MobileNet Speed* (arXiv:2206.01191)",
            categories=[],
            datasets=["imagenet-1k"],
            meta=meta,
        )


@register_arch("efficientformer_l1")
def _l1(tag: str) -> Architecture:
    return EfficientFormerArch("efficientformer_l1", tag)


@register_arch("efficientformer_l3")
def _l3(tag: str) -> Architecture:
    return EfficientFormerArch("efficientformer_l3", tag)


@register_arch("efficientformer_l7")
def _l7(tag: str) -> Architecture:
    return EfficientFormerArch("efficientformer_l7", tag)
