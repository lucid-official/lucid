"""Vision Transformer (ViT) weight converter — torchvision → Lucid.

torchvision packs the entire transformer encoder into an
``encoder.layers.encoder_layer_N`` ``Sequential`` and fuses Q/K/V into a
single ``nn.MultiheadAttention`` (``in_proj_weight`` / ``in_proj_bias``
of width ``3*dim``).  Lucid keeps the same fused QKV layout
(``attn.qkv``) and the same per-block pre-norm structure, so every
tensor maps 1:1 by a pure key rename — no reshaping or re-stacking:

==================================================================  ============================
torchvision                                                         Lucid
==================================================================  ============================
``class_token``                                                     ``cls_token``
``conv_proj.{weight,bias}``                                         ``patch_embed.proj.{weight,bias}``
``encoder.pos_embedding``                                           ``pos_embed``
``encoder.layers.encoder_layer_N.ln_1.{w,b}``                       ``blocks.N.norm1.{w,b}``
``encoder.layers.encoder_layer_N.self_attention.in_proj_{w,b}``     ``blocks.N.attn.qkv.{w,b}``
``encoder.layers.encoder_layer_N.self_attention.out_proj.{w,b}``    ``blocks.N.attn.proj.{w,b}``
``encoder.layers.encoder_layer_N.ln_2.{w,b}``                       ``blocks.N.norm2.{w,b}``
``encoder.layers.encoder_layer_N.mlp.0.{w,b}`` (Linear)             ``blocks.N.mlp.fc1.{w,b}``
``encoder.layers.encoder_layer_N.mlp.3.{w,b}`` (Linear)             ``blocks.N.mlp.fc2.{w,b}``
``encoder.ln.{w,b}``                                                ``norm.{w,b}``
``heads.head.{w,b}``                                                ``classifier`` head ``head.{w,b}``
==================================================================  ============================

The fused QKV weight stacks ``[W_q; W_k; W_v]`` along axis 0 in both
frameworks, and Lucid's ``_Attention.forward`` splits it via
``reshape(B, N, 3, H, D)`` — the leading ``3`` axis recovers the same
q/k/v order — so the raw ``(3*dim, dim)`` tensor transfers verbatim.

Only the four 224x224 ``IMAGENET1K_V1`` checkpoints (B/16, B/32, L/16,
L/32) ship here; their positional-embedding token count matches Lucid's
default ``image_size=224`` config.  ViT-Huge/14 is intentionally not
shipped — torchvision only distributes it as a 518x518 SWAG checkpoint,
whose ``pos_embedding`` has 1370 tokens (vs the default config's 257) and
would require an ``image_size`` config change to load.
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

_VIT_CITATION = (
    "@inproceedings{dosovitskiy2021image,\n"
    "  title={An Image is Worth 16x16 Words: Transformers for Image "
    "Recognition at Scale},\n"
    "  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, "
    "Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, "
    "Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, "
    "Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},\n"
    "  booktitle={ICLR}, year={2021}\n"
    "}"
)

_VIT_PAPER_URL = (
    "Dosovitskiy et al., 2021 — *An Image is Worth 16x16 Words: "
    "Transformers for Image Recognition at Scale* (arXiv:2010.11929)"
)

# arch -> (lucid_cls_factory, repo_id, title)
_VIT_VARIANTS: dict[str, tuple[str, str, str]] = {
    "vit_base_16": ("vit_base_16_cls", "lucid-dl/vit-base-16", "ViT-Base/16"),
    "vit_base_32": ("vit_base_32_cls", "lucid-dl/vit-base-32", "ViT-Base/32"),
    "vit_large_16": ("vit_large_16_cls", "lucid-dl/vit-large-16", "ViT-Large/16"),
    "vit_large_32": ("vit_large_32_cls", "lucid-dl/vit-large-32", "ViT-Large/32"),
}

# torchvision builders + weights enums keyed by arch.
_TV_BUILDERS = {
    "vit_base_16": (tvm.vit_b_16, tvm.ViT_B_16_Weights),
    "vit_base_32": (tvm.vit_b_32, tvm.ViT_B_32_Weights),
    "vit_large_16": (tvm.vit_l_16, tvm.ViT_L_16_Weights),
    "vit_large_32": (tvm.vit_l_32, tvm.ViT_L_32_Weights),
}

_ENCODER_LAYER_PREFIX = "encoder.layers.encoder_layer_"


class ViTArch(Architecture):
    """Converter for one torchvision ViT variant + tag."""

    def __init__(self, arch: str, tag: str) -> None:
        if arch not in _VIT_VARIANTS:
            raise KeyError(f"ViTArch: unknown arch {arch!r}")
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

        factory = _VIT_VARIANTS[self.arch][0]
        return getattr(models, factory)()

    def map_key(self, src_key: str) -> str | None:
        # Token / patch-embed / final-norm singletons.
        if src_key == "class_token":
            return "cls_token"
        if src_key == "encoder.pos_embedding":
            return "pos_embed"
        if src_key.startswith("conv_proj."):
            return "patch_embed.proj." + src_key[len("conv_proj.") :]
        if src_key.startswith("encoder.ln."):
            return "norm." + src_key[len("encoder.ln.") :]
        if src_key.startswith("heads.head."):
            return "head." + src_key[len("heads.head.") :]

        # Per-block encoder layers: encoder.layers.encoder_layer_N.<inner>
        if src_key.startswith(_ENCODER_LAYER_PREFIX):
            rest = src_key[len(_ENCODER_LAYER_PREFIX) :]
            idx, _, inner = rest.partition(".")
            if not inner:
                raise UnknownSourceKey(src_key)
            block = f"blocks.{idx}"
            if inner.startswith("ln_1."):
                return f"{block}.norm1." + inner[len("ln_1.") :]
            if inner.startswith("ln_2."):
                return f"{block}.norm2." + inner[len("ln_2.") :]
            if inner.startswith("self_attention.in_proj_"):
                # in_proj_weight / in_proj_bias → qkv.weight / qkv.bias
                suffix = inner[len("self_attention.in_proj_") :]
                return f"{block}.attn.qkv.{suffix}"
            if inner.startswith("self_attention.out_proj."):
                return f"{block}.attn.proj." + inner[len("self_attention.out_proj.") :]
            if inner.startswith("mlp.0."):
                return f"{block}.mlp.fc1." + inner[len("mlp.0.") :]
            if inner.startswith("mlp.3."):
                return f"{block}.mlp.fc2." + inner[len("mlp.3.") :]
        raise UnknownSourceKey(src_key)

    def spec(self) -> ConversionSpec:
        factory_name, repo_id, title = _VIT_VARIANTS[self.arch]
        model = self.target_model()
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
            model_type="vit",
            source=f"torchvision/{self._weights_enum.__name__}.{self.tag}",
            license="bsd-3-clause",
            num_classes=int(model.config.num_classes),
            config=config,
            preprocessing=preprocessing,
            citation=_VIT_CITATION,
            title=title,
            paper_url=_VIT_PAPER_URL,
            categories=categories,
            datasets=["imagenet-1k"],
            meta=meta,
        )


@register_arch("vit_base_16")
def _build_vit_base_16(tag: str) -> Architecture:
    return ViTArch("vit_base_16", tag)


@register_arch("vit_base_32")
def _build_vit_base_32(tag: str) -> Architecture:
    return ViTArch("vit_base_32", tag)


@register_arch("vit_large_16")
def _build_vit_large_16(tag: str) -> Architecture:
    return ViTArch("vit_large_16", tag)


@register_arch("vit_large_32")
def _build_vit_large_32(tag: str) -> Architecture:
    return ViTArch("vit_large_32", tag)
