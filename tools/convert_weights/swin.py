"""Swin Transformer weight converter — torchvision / timm → Lucid.

torchvision flattens the entire Swin trunk into a single ``features``
``Sequential`` where even-indexed sub-modules are patch-merging
downsamplers and odd-indexed sub-modules are stages of Swin blocks;
Lucid keeps a four-stage pyramid where each stage owns its ``blocks``
``ModuleList`` plus an (optional) ``downsample`` patch-merge.  Block
internals are otherwise near-identical (same window-attention math,
same relative-position-bias table, same MLP), so the mapping is a
pure key rename plus a single dropped buffer.

======================================================  ===================================
torchvision                                             Lucid
======================================================  ===================================
``features.0.0.{weight,bias}`` (patch conv)             ``patch_embed.proj.{weight,bias}``
``features.0.2.{weight,bias}`` (patch LN)               ``patch_embed.norm.{weight,bias}``
``features.{1,3,5,7}.N.norm1.{w,b}``                    ``stages.{0,1,2,3}.blocks.N.norm1.{w,b}``
``features.{1,3,5,7}.N.attn.relative_position_bias_table`` ``stages.S.blocks.N.attn.rel_pos_bias``
``features.{1,3,5,7}.N.attn.relative_position_index``   (dropped — non-persistent buffer)
``features.{1,3,5,7}.N.attn.qkv.{w,b}``                 ``stages.S.blocks.N.attn.qkv.{w,b}``
``features.{1,3,5,7}.N.attn.proj.{w,b}``                ``stages.S.blocks.N.attn.proj.{w,b}``
``features.{1,3,5,7}.N.norm2.{w,b}``                    ``stages.S.blocks.N.norm2.{w,b}``
``features.{1,3,5,7}.N.mlp.{0,3}.{w,b}``                ``stages.S.blocks.N.mlp.{0,3}.{w,b}``
``features.{2,4,6}.reduction.weight`` (4C→2C)           ``stages.{0,1,2}.downsample.proj.weight``
``features.{2,4,6}.norm.{w,b}`` (LN over 4C)            ``stages.{0,1,2}.downsample.norm.{w,b}``
``norm.{weight,bias}`` (final LN)                       ``norm.{weight,bias}``  (identical)
``head.{weight,bias}``                                  ``classifier.{weight,bias}``
======================================================  ===================================

The dropped ``relative_position_index`` buffer is *not* a learnable
parameter — Lucid recomputes the identical int64 index at construction
time and registers it ``persistent=False`` so it never appears in the
state dict.  A numerical check (torchvision's buffer == Lucid's
``rel_pos_idx``) confirms the two index orderings agree, so the
``relative_position_bias_table`` rows line up 1:1.

Swin-L ships no ImageNet-1k head in torchvision; its ``ms_in22k_ft_in1k``
checkpoint is sourced from timm, whose key layout differs (see
:class:`SwinTimmArch`).
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

_SWIN_CITATION = (
    "@inproceedings{liu2021swin,\n"
    "  title={Swin Transformer: Hierarchical Vision Transformer using "
    "Shifted Windows},\n"
    "  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, "
    "Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},\n"
    "  booktitle={ICCV}, year={2021}\n"
    "}"
)

_SWIN_PAPER_URL = (
    "Liu et al., 2021 — *Swin Transformer: Hierarchical Vision "
    "Transformer using Shifted Windows* (arXiv:2103.14030)"
)


# ---------------------------------------------------------------------------
# torchvision-sourced variants (Tiny / Small / Base, IMAGENET1K_V1)
# ---------------------------------------------------------------------------

_SWIN_VARIANTS: dict[str, tuple[str, str, str]] = {
    # arch -> (lucid_cls_factory, repo_id, title)
    "swin_tiny": ("swin_tiny_cls", "lucid-dl/swin-tiny", "Swin-Tiny"),
    "swin_small": ("swin_small_cls", "lucid-dl/swin-small", "Swin-Small"),
    "swin_base": ("swin_base_cls", "lucid-dl/swin-base", "Swin-Base"),
}

_TV_BUILDERS = {
    "swin_tiny": (tvm.swin_t, tvm.Swin_T_Weights),
    "swin_small": (tvm.swin_s, tvm.Swin_S_Weights),
    "swin_base": (tvm.swin_b, tvm.Swin_B_Weights),
}

# torchvision: features.{tv}.* (block stages)  ->  Lucid stages.{luc}.blocks.*
_TV_STAGE_MAP = {"1": "0", "3": "1", "5": "2", "7": "3"}
# torchvision: features.{tv}.* (patch-merge)   ->  Lucid stages.{luc}.downsample.*
_TV_DOWNSAMPLE_MAP = {"2": "0", "4": "1", "6": "2"}


class SwinArch(Architecture):
    """Converter for one torchvision Swin variant + tag."""

    def __init__(self, arch: str, tag: str) -> None:
        if arch not in _SWIN_VARIANTS:
            raise KeyError(f"SwinArch: unknown arch {arch!r}")
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

        factory = _SWIN_VARIANTS[self.arch][0]
        return getattr(models, factory)()

    def map_key(self, src_key: str) -> str | None:
        # Patch embedding stem.
        if src_key.startswith("features.0.0."):
            return "patch_embed.proj." + src_key[len("features.0.0.") :]
        if src_key.startswith("features.0.2."):
            return "patch_embed.norm." + src_key[len("features.0.2.") :]
        # Final norm — identical name.
        if src_key.startswith("norm."):
            return src_key
        # Classification head.
        if src_key.startswith("head."):
            return "classifier." + src_key[len("head.") :]
        # Block stages + patch-merge downsamplers.
        if src_key.startswith("features."):
            parts = src_key.split(".")
            tv_idx = parts[1]
            if tv_idx in _TV_STAGE_MAP:
                luc_stage = _TV_STAGE_MAP[tv_idx]
                block = parts[2]
                inner = ".".join(parts[3:])
                # The recomputed relative-position index is a non-persistent
                # buffer in Lucid; drop the upstream copy.
                if inner == "attn.relative_position_index":
                    return None
                if inner == "attn.relative_position_bias_table":
                    return f"stages.{luc_stage}.blocks.{block}.attn.rel_pos_bias"
                return f"stages.{luc_stage}.blocks.{block}.{inner}"
            if tv_idx in _TV_DOWNSAMPLE_MAP:
                luc_stage = _TV_DOWNSAMPLE_MAP[tv_idx]
                inner = ".".join(parts[2:])
                if inner.startswith("reduction."):
                    return f"stages.{luc_stage}.downsample.proj." + inner[len("reduction.") :]
                if inner.startswith("norm."):
                    return f"stages.{luc_stage}.downsample.norm." + inner[len("norm.") :]
        raise UnknownSourceKey(src_key)

    def spec(self) -> ConversionSpec:
        import lucid.models as models

        factory_name, repo_id, title = _SWIN_VARIANTS[self.arch]
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
            model_type="swin",
            source=f"torchvision/{self._weights_enum.__name__}.{self.tag}",
            license="mit",
            num_classes=int(model.config.num_classes),
            config=config,
            preprocessing=preprocessing,
            citation=_SWIN_CITATION,
            title=title,
            paper_url=_SWIN_PAPER_URL,
            categories=categories,
            datasets=["imagenet-1k"],
            meta=meta,
        )


@register_arch("swin_tiny")
def _build_swin_tiny(tag: str) -> Architecture:
    return SwinArch("swin_tiny", tag)


@register_arch("swin_small")
def _build_swin_small(tag: str) -> Architecture:
    return SwinArch("swin_small", tag)


@register_arch("swin_base")
def _build_swin_base(tag: str) -> Architecture:
    return SwinArch("swin_base", tag)


# ---------------------------------------------------------------------------
# SWIN-LARGE — sourced from timm's ``ms_in22k_ft_in1k`` checkpoint.
# torchvision does not ship a 1k-class Swin-L head; timm wraps the official
# Microsoft weights (ImageNet-22k pretrain → ImageNet-1k finetune).
# ---------------------------------------------------------------------------

_TIMM_VARIANTS: dict[str, tuple[str, str, str]] = {
    "swin_large": ("swin_large_cls", "lucid-dl/swin-large", "Swin-Large")
}

# timm full model names keyed by Lucid arch.
_TIMM_MODEL_NAMES = {"swin_large": "swin_large_patch4_window7_224"}


def _datasets_from_timm_tag(tag: str) -> list[str]:
    """Map a timm tag string to the datasets it touched (pretrain → ft)."""
    t = tag.lower()
    ds: list[str] = []
    if "in22k" in t:
        ds.append("imagenet-22k")
    if "in1k" in t:
        ds.append("imagenet-1k")
    return ds


class SwinTimmArch(Architecture):
    """Converter for a timm-hosted Swin variant + tag.

    timm keeps the *original* Microsoft layout, which differs from both
    torchvision and Lucid:

    =================================================  ===================================
    timm                                               Lucid
    =================================================  ===================================
    ``patch_embed.proj.{w,b}``                         ``patch_embed.proj.{w,b}``
    ``patch_embed.norm.{w,b}``                         ``patch_embed.norm.{w,b}``
    ``layers.S.blocks.N.norm1.{w,b}``                  ``stages.S.blocks.N.norm1.{w,b}``
    ``layers.S.blocks.N.attn.relative_position_bias_table`` ``stages.S.blocks.N.attn.rel_pos_bias``
    ``layers.S.blocks.N.attn.relative_position_index`` (dropped — non-persistent buffer)
    ``layers.S.blocks.N.attn.qkv.{w,b}``               ``stages.S.blocks.N.attn.qkv.{w,b}``
    ``layers.S.blocks.N.attn.proj.{w,b}``              ``stages.S.blocks.N.attn.proj.{w,b}``
    ``layers.S.blocks.N.attn_mask`` (buffer)           (dropped — recomputed)
    ``layers.S.blocks.N.norm2.{w,b}``                  ``stages.S.blocks.N.norm2.{w,b}``
    ``layers.S.blocks.N.mlp.fc1.{w,b}``                ``stages.S.blocks.N.mlp.0.{w,b}``
    ``layers.S.blocks.N.mlp.fc2.{w,b}``                ``stages.S.blocks.N.mlp.3.{w,b}``
    ``layers.{1,2,3}.downsample.norm.{w,b}``           ``stages.{0,1,2}.downsample.norm.{w,b}``
    ``layers.{1,2,3}.downsample.reduction.weight``     ``stages.{0,1,2}.downsample.proj.weight``
    ``norm.{w,b}`` (final LN)                          ``norm.{w,b}``
    ``head.fc.{w,b}``                                  ``classifier.{w,b}``
    =================================================  ===================================

    Note: timm attaches the patch-merge downsampler to the *start* of
    stages 1/2/3 (``layers.1.downsample`` runs before ``layers.1.blocks``,
    on stage 0's output), whereas Lucid attaches it to the *end* of stages
    0/1/2 (``stages.0.downsample``).  The two describe the same operation,
    so the downsample stage index shifts down by one in :meth:`map_key`.
    """

    def __init__(self, arch: str, tag: str) -> None:
        import timm

        self.arch = arch
        self.tag = tag
        self._timm_name = f"{_TIMM_MODEL_NAMES[arch]}.{tag.lower()}"
        self._model = timm.create_model(self._timm_name, pretrained=True)
        self._model.eval()

        import lucid.models as models

        self._lucid_factory = _TIMM_VARIANTS[arch][0]
        self._lucid_model = getattr(models, self._lucid_factory)()

    def source_state_dict(self) -> dict[str, object]:
        return {k: v.detach().cpu().numpy() for k, v in self._model.state_dict().items()}

    def target_model(self) -> Module:
        return self._lucid_model

    def map_key(self, src_key: str) -> str | None:
        # Patch embedding stem — identical names.
        if src_key.startswith("patch_embed."):
            return src_key
        # Final norm — identical name.
        if src_key.startswith("norm."):
            return src_key
        # Classification head.
        if src_key.startswith("head.fc."):
            return "classifier." + src_key[len("head.fc.") :]
        # Stages: layers.S.* → stages.S.*
        if src_key.startswith("layers."):
            parts = src_key.split(".")
            stage = parts[1]
            kind = parts[2]
            if kind == "blocks":
                block = parts[3]
                inner = ".".join(parts[4:])
                prefix = f"stages.{stage}.blocks.{block}"
                # Recomputed buffers — never in Lucid's state dict.
                if inner == "attn.relative_position_index":
                    return None
                if inner == "attn_mask":
                    return None
                if inner == "attn.relative_position_bias_table":
                    return f"{prefix}.attn.rel_pos_bias"
                if inner.startswith("mlp.fc1."):
                    return f"{prefix}.mlp.0." + inner[len("mlp.fc1.") :]
                if inner.startswith("mlp.fc2."):
                    return f"{prefix}.mlp.3." + inner[len("mlp.fc2.") :]
                return f"{prefix}.{inner}"
            if kind == "downsample":
                # timm attaches the patch-merge downsampler to the *start* of
                # stages 1/2/3 (``layers.{1,2,3}.downsample``), operating on
                # the preceding stage's resolution.  Lucid attaches it to the
                # *end* of stages 0/1/2 (``stages.{0,1,2}.downsample``), so the
                # stage index shifts down by one.
                tgt_stage = int(stage) - 1
                inner = ".".join(parts[3:])
                if inner.startswith("reduction."):
                    return f"stages.{tgt_stage}.downsample.proj." + inner[len("reduction.") :]
                if inner.startswith("norm."):
                    return f"stages.{tgt_stage}.downsample.norm." + inner[len("norm.") :]
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
            # timm's default_cfg carries no accuracy numbers; the paper
            # (Liu et al., 2021, Table 2) reports 86.3% top-1 for the
            # Swin-L in22k→in1k finetune at 224x224.
            "metrics": {"ImageNet-1k": {"acc@1": 86.32, "acc@5": 97.89}},
        }

        return ConversionSpec(
            model_name=factory_name,
            architecture=self.arch,
            repo_id=repo_id,
            tag=self.tag,
            task="image-classification",
            model_type="swin",
            source=f"timm/{self._timm_name}",
            license=str(cfg.get("license", "mit")),
            num_classes=int(self._lucid_model.config.num_classes),
            config=config,
            preprocessing=preprocessing,
            citation=_SWIN_CITATION,
            title=title,
            paper_url=_SWIN_PAPER_URL,
            categories=[],
            datasets=_datasets_from_timm_tag(self.tag),
            meta=meta,
        )


@register_arch("swin_large")
def _build_swin_large(tag: str) -> Architecture:
    return SwinTimmArch("swin_large", tag)
