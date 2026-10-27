"""PVT v2 weight converter — timm → Lucid.

Five paper-cited variants (Wang et al., CVMJ 2022): ``pvt_v2_b0`` /
``pvt_v2_b2`` / ``pvt_v2_b3`` / ``pvt_v2_b4`` / ``pvt_v2_b5``.  Source =
timm's ``pvt_v2_b{0,2,3,4,5}.in1k`` checkpoints.

Lucid's PVT v2 module layout was built to mirror timm's key naming
verbatim (top-level ``patch_embed``, ``stages.S.downsample`` /
``stages.S.blocks.B`` / ``stages.S.norm``, depthwise-conv MLP under
``mlp.dwconv``, Spatial-Reduction Attention under ``attn.{q,kv,proj,
sr,norm}``), so the mapping is a pure identity except for the head:

==============================  =========================
timm                            Lucid
==============================  =========================
``patch_embed.*``               ``patch_embed.*``       (identity)
``stages.S.downsample.*``       ``stages.S.downsample.*`` (identity)
``stages.S.blocks.B.*``         ``stages.S.blocks.B.*`` (identity)
``stages.S.norm.*``             ``stages.S.norm.*``     (identity)
``head.{weight,bias}``          ``classifier.{weight,bias}``
==============================  =========================

The B1 note this file used to carry is obsolete: ``_CFG_B1`` is
``depths=(2, 2, 2, 2)``, matching the paper and timm's ``pvt_v2_b1``, so
the config edit it said would be required has already happened.
"""

import dataclasses

from lucid.nn import Module
from tools.convert_weights._base import Architecture, ConversionSpec, register_arch

_PVT_CITATION = (
    "@article{wang2022pvtv2,\n"
    "  title={PVT v2: Improved Baselines with Pyramid Vision "
    "Transformer},\n"
    "  author={Wang, Wenhai and Xie, Enze and Li, Xiang and Fan, "
    "Deng-Ping and Song, Kaitao and Liang, Ding and Lu, Tong and Luo, "
    "Ping and Shao, Ling},\n"
    "  journal={Computational Visual Media}, year={2022}\n"
    "}"
)

_PVT_VARIANTS: dict[str, tuple[str, str, str]] = {
    # arch -> (lucid_cls_factory, repo_id, title)
    "pvt_v2_b0": ("pvt_v2_b0_cls", "lucid-dl/pvt-v2-b0", "PVT v2-B0"),
    "pvt_v2_b1": ("pvt_v2_b1_cls", "lucid-dl/pvt-v2-b1", "PVT v2-B1"),
    "pvt_v2_b2": ("pvt_v2_b2_cls", "lucid-dl/pvt-v2-b2", "PVT v2-B2"),
    "pvt_v2_b3": ("pvt_v2_b3_cls", "lucid-dl/pvt-v2-b3", "PVT v2-B3"),
    "pvt_v2_b4": ("pvt_v2_b4_cls", "lucid-dl/pvt-v2-b4", "PVT v2-B4"),
    "pvt_v2_b5": ("pvt_v2_b5_cls", "lucid-dl/pvt-v2-b5", "PVT v2-B5"),
}

# Paper Table 1 top-1 accuracies @ 224x224 on ImageNet-1k.
_ACC1: dict[str, float] = {
    "pvt_v2_b0": 70.5,
    "pvt_v2_b1": 78.7,
    "pvt_v2_b2": 82.0,
    "pvt_v2_b3": 83.1,
    "pvt_v2_b4": 83.6,
    "pvt_v2_b5": 83.8,
}


class PVTArch(Architecture):
    """Converter for one timm PVT v2 variant + tag."""

    def __init__(self, arch: str, tag: str) -> None:
        import timm

        if arch not in _PVT_VARIANTS:
            raise KeyError(f"PVTArch: unknown arch {arch!r}")
        self.arch = arch
        # Lucid keeps weight-enum tags uppercase across all families;
        # timm's registry is lowercase, so the lookup uses a lowered copy.
        self.tag = tag
        self._timm_name = f"{arch}.{tag.lower()}"
        self._model = timm.create_model(self._timm_name, pretrained=True)
        self._model.eval()

        import lucid.models as models

        self._lucid_factory = _PVT_VARIANTS[arch][0]
        self._lucid_model = getattr(models, self._lucid_factory)()

    def source_state_dict(self) -> dict[str, object]:
        return {k: v.detach().cpu().numpy() for k, v in self._model.state_dict().items()}

    def target_model(self) -> Module:
        return self._lucid_model

    def map_key(self, src_key: str) -> str | None:
        # Head: timm ``head`` → Lucid ``classifier``; everything else is
        # an identity map (Lucid mirrors the timm PVT v2 key layout).
        if src_key.startswith("head."):
            return "classifier." + src_key[len("head.") :]
        return src_key

    def spec(self) -> ConversionSpec:
        factory_name, repo_id, title = _PVT_VARIANTS[self.arch]
        config = {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in dataclasses.asdict(self._lucid_model.config).items()
        }

        cfg = self._model.default_cfg
        from lucid.utils.transforms import ImageClassification

        crop = int(cfg["input_size"][1])
        resize = int(round(crop / float(cfg.get("crop_pct", 0.9))))
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
            model_type="pvt",
            source=f"timm/{self._timm_name}",
            license=str(cfg.get("license", "apache-2.0")),
            num_classes=int(self._lucid_model.config.num_classes),
            config=config,
            preprocessing=preprocessing,
            citation=_PVT_CITATION,
            title=title,
            paper_url="Wang et al., 2022 — *PVT v2: Improved Baselines "
            "with Pyramid Vision Transformer* (arXiv:2106.13797)",
            categories=[],
            datasets=["imagenet-1k"],
            meta=meta,
        )


@register_arch("pvt_v2_b0")
def _b0(tag: str) -> Architecture:
    return PVTArch("pvt_v2_b0", tag)


@register_arch("pvt_v2_b1")
def _b1(tag: str) -> Architecture:
    return PVTArch("pvt_v2_b1", tag)


@register_arch("pvt_v2_b2")
def _b2(tag: str) -> Architecture:
    return PVTArch("pvt_v2_b2", tag)


@register_arch("pvt_v2_b3")
def _b3(tag: str) -> Architecture:
    return PVTArch("pvt_v2_b3", tag)


@register_arch("pvt_v2_b4")
def _b4(tag: str) -> Architecture:
    return PVTArch("pvt_v2_b4", tag)


@register_arch("pvt_v2_b5")
def _b5(tag: str) -> Architecture:
    return PVTArch("pvt_v2_b5", tag)
