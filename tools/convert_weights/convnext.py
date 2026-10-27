"""ConvNeXt weight converter — reference framework → Lucid.

Maps torchvision's ConvNeXt ``state_dict`` onto Lucid's module layout.
torchvision flattens the entire body into a single ``features`` Sequential
where even-indexed sub-modules are downsamplers and odd-indexed sub-modules
are stages; Lucid keeps them as separate ``stem`` / ``downsamplers.{0,1,2}``
/ ``stages.{0,1,2,3}`` attributes.  Block internals also differ:

============================================  ===================================
torchvision                                   Lucid
============================================  ===================================
``features.0.0.{weight,bias}``                ``stem.conv.0.{weight,bias}``
``features.0.1.{weight,bias}``                ``stem.norm.{weight,bias}``
``features.1.N.layer_scale`` (C, 1, 1)        ``stages.0.N.gamma`` (C,)  ← reshape
``features.1.N.block.0.{w,b}`` (dwconv)       ``stages.0.N.dwconv.{w,b}``
``features.1.N.block.2.{w,b}`` (LayerNorm)    ``stages.0.N.norm.{w,b}``
``features.1.N.block.3.{w,b}`` (Linear)       ``stages.0.N.fc1.{w,b}``
``features.1.N.block.5.{w,b}`` (Linear)       ``stages.0.N.fc2.{w,b}``
``features.{3,5,7}.N.*``                      ``stages.{1,2,3}.N.*``
``features.{2,4,6}.0.{w,b}`` (downsample LN)  ``downsamplers.{0,1,2}.norm.{w,b}``
``features.{2,4,6}.1.{w,b}`` (downsample Conv) ``downsamplers.{0,1,2}.conv.{w,b}``
``classifier.0.{w,b}`` (head LayerNorm)       ``head_norm.{w,b}``
``classifier.2.{w,b}`` (head Linear)          ``classifier.{w,b}``
============================================  ===================================

The ``layer_scale → gamma`` reshape collapses the broadcasting-friendly
NCHW shape ``(C, 1, 1)`` down to ``(C,)``; Lucid applies the scale via
an explicit elementwise multiply rather than implicit broadcasting.
"""

import dataclasses

import torchvision.models as tvm

from lucid.nn import Module
from tools.convert_weights._base import (
    Architecture,
    ConversionSpec,
    UnknownSourceKey,
    register_arch,
)

_CONVNEXT_CITATION = (
    "@inproceedings{liu2022convnet,\n"
    "  title={A ConvNet for the 2020s},\n"
    "  author={Liu, Zhuang and Mao, Hanzi and Wu, Chao-Yuan and Feichtenhofer, "
    "Christoph and Darrell, Trevor and Xie, Saining},\n"
    "  booktitle={CVPR}, year={2022}\n"
    "}"
)

_CONVNEXT_VARIANTS: dict[str, tuple[str, str, str]] = {
    # arch -> (lucid_cls_factory, repo_id, title)
    "convnext_tiny": ("convnext_tiny_cls", "lucid-dl/convnext-tiny", "ConvNeXt-Tiny"),
    "convnext_small": ("convnext_small_cls", "lucid-dl/convnext-small", "ConvNeXt-Small"),
    "convnext_base": ("convnext_base_cls", "lucid-dl/convnext-base", "ConvNeXt-Base"),
    "convnext_large": ("convnext_large_cls", "lucid-dl/convnext-large", "ConvNeXt-Large"),
}

_TV_BUILDERS = {
    "convnext_tiny": (tvm.convnext_tiny, tvm.ConvNeXt_Tiny_Weights),
    "convnext_small": (tvm.convnext_small, tvm.ConvNeXt_Small_Weights),
    "convnext_base": (tvm.convnext_base, tvm.ConvNeXt_Base_Weights),
    "convnext_large": (tvm.convnext_large, tvm.ConvNeXt_Large_Weights),
}


class ConvNeXtArch(Architecture):
    """Converter for one torchvision ConvNeXt variant + tag."""

    def __init__(self, arch: str, tag: str) -> None:
        if arch not in _CONVNEXT_VARIANTS:
            raise KeyError(f"ConvNeXtArch: unknown arch {arch!r}")
        self.arch = arch
        self.tag = tag
        self._builder, self._weights_enum = _TV_BUILDERS[arch]
        self._tv_weights = self._weights_enum[tag]

    def source_state_dict(self) -> dict[str, object]:
        model = self._builder(weights=self._tv_weights)
        model.eval()
        return {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}

    def target_model(self) -> Module:
        import lucid.models as models

        factory = _CONVNEXT_VARIANTS[self.arch][0]
        return getattr(models, factory)()

    def map_key(self, src_key: str) -> str | None:
        # Stem
        if src_key.startswith("features.0.0."):
            return "stem.conv.0." + src_key[len("features.0.0.") :]
        if src_key.startswith("features.0.1."):
            return "stem.norm." + src_key[len("features.0.1.") :]
        # Downsamplers
        for tv_idx, luc_idx in (("2", "0"), ("4", "1"), ("6", "2")):
            prefix0 = f"features.{tv_idx}.0."
            prefix1 = f"features.{tv_idx}.1."
            if src_key.startswith(prefix0):
                return f"downsamplers.{luc_idx}.norm." + src_key[len(prefix0) :]
            if src_key.startswith(prefix1):
                return f"downsamplers.{luc_idx}.conv." + src_key[len(prefix1) :]
        # Stages: features.{1,3,5,7}.N.* → stages.{0,1,2,3}.N.*
        for tv_stage, luc_stage in (("1", "0"), ("3", "1"), ("5", "2"), ("7", "3")):
            prefix = f"features.{tv_stage}."
            if src_key.startswith(prefix):
                rest = src_key[len(prefix) :]
                parts = rest.split(".", 1)
                if len(parts) != 2:
                    raise UnknownSourceKey(src_key)
                block_idx, inner = parts
                stage_prefix = f"stages.{luc_stage}.{block_idx}"
                if inner == "layer_scale":
                    return f"{stage_prefix}.gamma"
                if inner.startswith("block.0."):
                    return f"{stage_prefix}.dwconv." + inner[len("block.0.") :]
                if inner.startswith("block.2."):
                    return f"{stage_prefix}.norm." + inner[len("block.2.") :]
                if inner.startswith("block.3."):
                    return f"{stage_prefix}.fc1." + inner[len("block.3.") :]
                if inner.startswith("block.5."):
                    return f"{stage_prefix}.fc2." + inner[len("block.5.") :]
                raise UnknownSourceKey(src_key)
        # Classifier head
        if src_key == "classifier.0.weight":
            return "head_norm.weight"
        if src_key == "classifier.0.bias":
            return "head_norm.bias"
        if src_key == "classifier.2.weight":
            return "classifier.weight"
        if src_key == "classifier.2.bias":
            return "classifier.bias"
        raise UnknownSourceKey(src_key)

    def transform_value(self, src_key: str, arr: object) -> object:
        # layer_scale ships as (C, 1, 1) for NCHW broadcasting; Lucid keeps
        # gamma as a flat (C,) and multiplies elementwise after the channel
        # axis is materialised, so we squeeze the trailing singleton dims.
        if "layer_scale" in src_key:
            return arr.squeeze()  # type: ignore[attr-defined]
        return arr

    def spec(self) -> ConversionSpec:
        import lucid.models as models

        factory_name, repo_id, title = _CONVNEXT_VARIANTS[self.arch]
        model = getattr(models, factory_name)()
        config = {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in dataclasses.asdict(model.config).items()
        }

        tv_meta = dict(self._tv_weights.meta)
        categories = list(tv_meta.get("categories", []))
        from lucid.utils.transforms import ImageClassification

        tf = self._tv_weights.transforms()
        preset = ImageClassification(
            crop_size=int(tf.crop_size[0]),
            resize_size=int(tf.resize_size[0]),
            mean=tuple(float(m) for m in tf.mean),
            std=tuple(float(s) for s in tf.std),
            interpolation=str(tf.interpolation.value),
        )
        preprocessing = preset.to_dict()
        meta = {
            "num_params": int(tv_meta.get("num_params", 0)),
            "gflops": float(tv_meta.get("_ops", 0.0)),
            "recipe": str(tv_meta.get("recipe", "")),
            "metrics": dict(tv_meta.get("_metrics", {})),
        }

        return ConversionSpec(
            model_name=factory_name,
            architecture=self.arch,
            repo_id=repo_id,
            tag=self.tag,
            task="image-classification",
            model_type="convnext",
            source=f"torchvision/{self._weights_enum.__name__}.{self.tag}",
            license="bsd-3-clause",
            num_classes=int(model.config.num_classes),
            config=config,
            preprocessing=preprocessing,
            citation=_CONVNEXT_CITATION,
            title=title,
            paper_url="Liu et al., 2022 — *A ConvNet for the 2020s* " "(arXiv:2201.03545)",
            categories=categories,
            meta=meta,
        )


@register_arch("convnext_tiny")
def _build_tiny(tag: str) -> Architecture:
    return ConvNeXtArch("convnext_tiny", tag)


@register_arch("convnext_small")
def _build_small(tag: str) -> Architecture:
    return ConvNeXtArch("convnext_small", tag)


@register_arch("convnext_base")
def _build_base(tag: str) -> Architecture:
    return ConvNeXtArch("convnext_base", tag)


@register_arch("convnext_large")
def _build_large(tag: str) -> Architecture:
    return ConvNeXtArch("convnext_large", tag)


# ---------------------------------------------------------------------------
# CONVNEXT-XLARGE — sourced from timm's ``fb_in22k_ft_in1k`` checkpoint.
# torchvision does not ship a 1k-class XLarge head; timm wraps the official
# Facebook AI Research weights instead.
# ---------------------------------------------------------------------------


class ConvNeXtTimmArch(Architecture):
    """Converter for a timm-hosted ConvNeXt variant + tag.

    timm uses a different key layout from torchvision:

    ===============================================  ===========================
    timm                                             Lucid
    ===============================================  ===========================
    ``stem.0.{w,b}``                                 ``stem.conv.0.{w,b}``
    ``stem.1.{w,b}``                                 ``stem.norm.{w,b}``
    ``stages.S.blocks.N.gamma``     (C,)             ``stages.S.N.gamma``  (C,)
    ``stages.S.blocks.N.conv_dw.{w,b}``              ``stages.S.N.dwconv.{w,b}``
    ``stages.S.blocks.N.norm.{w,b}``                 ``stages.S.N.norm.{w,b}``
    ``stages.S.blocks.N.mlp.fc1.{w,b}``              ``stages.S.N.fc1.{w,b}``
    ``stages.S.blocks.N.mlp.fc2.{w,b}``              ``stages.S.N.fc2.{w,b}``
    ``stages.{1,2,3}.downsample.0.{w,b}`` (LN)       ``downsamplers.{0,1,2}.norm.{w,b}``
    ``stages.{1,2,3}.downsample.1.{w,b}`` (Conv)     ``downsamplers.{0,1,2}.conv.{w,b}``
    ``head.norm.{w,b}``                              ``head_norm.{w,b}``
    ``head.fc.{w,b}``                                ``classifier.{w,b}``
    ===============================================  ===========================

    ``gamma`` already ships as ``(C,)`` here (timm uses a 1-D layer_scale
    parameter), so no ``transform_value`` reshape is required.
    """

    def __init__(self, arch: str, tag: str) -> None:
        import timm

        self.arch = arch
        # Lucid keeps weight-enum tags uppercase across all families
        # (``IMAGENET1K_V1``, ``FB_IN22K_FT_IN1K`` …) — the Hub URL path
        # then mirrors this verbatim.  timm's own model registry is
        # case-sensitive and lowercase, so the lookup uses a lowered copy.
        self.tag = tag
        self._timm_name = f"{arch}.{tag.lower()}"
        self._model = timm.create_model(self._timm_name, pretrained=True)
        self._model.eval()
        # Build a Lucid model just to read its config + state_dict layout.
        import lucid.models as models

        self._lucid_factory = _TIMM_VARIANTS[arch][0]
        self._lucid_model = getattr(models, self._lucid_factory)()

    def source_state_dict(self) -> dict[str, object]:
        return {k: v.detach().cpu().numpy() for k, v in self._model.state_dict().items()}

    def target_model(self) -> Module:
        return self._lucid_model

    def map_key(self, src_key: str) -> str | None:
        # Stem
        if src_key.startswith("stem.0."):
            return "stem.conv.0." + src_key[len("stem.0.") :]
        if src_key.startswith("stem.1."):
            return "stem.norm." + src_key[len("stem.1.") :]
        # Head
        if src_key.startswith("head.norm."):
            return "head_norm." + src_key[len("head.norm.") :]
        if src_key.startswith("head.fc."):
            return "classifier." + src_key[len("head.fc.") :]
        # Stages — both block body and (per-stage) downsample.
        if src_key.startswith("stages."):
            parts = src_key.split(".")
            # parts = ["stages", "S", "blocks", "N", ...rest...]
            #       OR ["stages", "S", "downsample", {"0","1"}, "weight"/"bias"]
            stage = parts[1]
            kind = parts[2]
            if kind == "downsample":
                # timm: stages.S.downsample.0/1 → Lucid: downsamplers.{S-1}.norm/conv
                # Stage 0 has no downsample (it's after the stem); ds blocks live on
                # stages 1, 2, 3 in timm, so a stage-0 key is not one.
                if stage == "0":
                    raise UnknownSourceKey(src_key)
                luc_idx = str(int(stage) - 1)
                ds_part = parts[3]  # "0" (LN) or "1" (Conv)
                rest = ".".join(parts[4:])
                if ds_part == "0":
                    return f"downsamplers.{luc_idx}.norm.{rest}"
                if ds_part == "1":
                    return f"downsamplers.{luc_idx}.conv.{rest}"
                raise UnknownSourceKey(src_key)
            if kind == "blocks":
                block = parts[3]
                inner_parts = parts[4:]
                inner = ".".join(inner_parts)
                lucid_prefix = f"stages.{stage}.{block}"
                if inner == "gamma":
                    return f"{lucid_prefix}.gamma"
                if inner.startswith("conv_dw."):
                    return f"{lucid_prefix}.dwconv." + inner[len("conv_dw.") :]
                if inner.startswith("norm."):
                    return f"{lucid_prefix}.norm." + inner[len("norm.") :]
                if inner.startswith("mlp.fc1."):
                    return f"{lucid_prefix}.fc1." + inner[len("mlp.fc1.") :]
                if inner.startswith("mlp.fc2."):
                    return f"{lucid_prefix}.fc2." + inner[len("mlp.fc2.") :]
        raise UnknownSourceKey(src_key)

    def spec(self) -> ConversionSpec:
        factory_name, repo_id, title = _TIMM_VARIANTS[self.arch]
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
            # timm's default_cfg doesn't carry accuracy numbers directly; the
            # tag is the convention (paper Table 11 reports 87.0% top-1 for
            # the in22k_ft_in1k variant at 224x224).
            "metrics": {"ImageNet-1k": {"acc@1": 87.0, "acc@5": 98.2}},
        }

        return ConversionSpec(
            model_name=factory_name,
            architecture=self.arch,
            repo_id=repo_id,
            tag=self.tag,
            task="image-classification",
            model_type="convnext",
            source=f"timm/{self._timm_name}",
            license=str(cfg.get("license", "apache-2.0")),
            num_classes=int(self._lucid_model.config.num_classes),
            config=config,
            preprocessing=preprocessing,
            citation=_CONVNEXT_CITATION,
            title=title,
            paper_url="Liu et al., 2022 — *A ConvNet for the 2020s* " "(arXiv:2201.03545)",
            categories=[],
            datasets=_datasets_from_timm_tag(self.tag),
            meta=meta,
        )


_TIMM_VARIANTS: dict[str, tuple[str, str, str]] = {
    "convnext_xlarge": ("convnext_xlarge_cls", "lucid-dl/convnext-xlarge", "ConvNeXt-XLarge")
}


def _datasets_from_timm_tag(tag: str) -> list[str]:
    """Map a timm tag string to the list of datasets it touched.

    Order matters — the YAML frontmatter lists pretrain first then
    finetune.  ``fb_in22k_ft_in1k`` resolves to
    ``["imagenet-22k", "imagenet-1k"]``; a plain ``fb_in22k`` to
    ``["imagenet-22k"]``; a plain ``in1k`` to ``["imagenet-1k"]``.
    Other unknown tokens (``laion2b``, ``clip_*``, …) are added best-
    effort.
    """
    t = tag.lower()
    ds: list[str] = []
    if "in22k" in t:
        ds.append("imagenet-22k")
    elif "in12k" in t:
        ds.append("imagenet-12k")
    if "laion2b" in t:
        ds.append("laion-2b")
    elif "laion" in t:
        ds.append("laion")
    # An "ft_in1k" or "in1k" token in a tag means the final evaluation
    # set is ImageNet-1k — whether or not there was a preceding pretrain.
    if "in1k" in t:
        ds.append("imagenet-1k")
    return ds


@register_arch("convnext_xlarge")
def _build_xlarge(tag: str) -> Architecture:
    return ConvNeXtTimmArch("convnext_xlarge", tag)
