"""V-JEPA 2 checkpoint conversion from the official Hub safetensors files.

The published files use separate query/key/value projections and the naming
of the reference model.  Lucid stores one fused QKV projection and keeps an
EMA target encoder alongside the trainable encoder, so the converter combines
the three projections and mirrors the encoder tensors into ``target_encoder``.
"""

import dataclasses

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file

from lucid.nn import Module
from tools.convert_weights._base import Architecture, ConversionSpec, register_arch

_CITATION = (
    'Assran et al., "V-JEPA 2: Self-Supervised Video Models Enable '
    'Understanding, Prediction and Planning," arXiv:2506.09985, 2025.'
)
_PAPER_URL = "https://arxiv.org/abs/2506.09985"

_VARIANTS: dict[str, tuple[str, str, str, str, int, int, int, int, str]] = {
    # name -> (factory, source repo, Hub slug, title, dim, depth, heads, size, license)
    "vjepa2_vit_large": (
        "vjepa2_vit_large",
        "facebook/vjepa2-vitl-fpc64-256",
        "vjepa2-vitl",
        "V-JEPA 2 ViT-L/16",
        1024,
        24,
        16,
        256,
        "mit",
    ),
    "vjepa2_vit_huge": (
        "vjepa2_vit_huge",
        "facebook/vjepa2-vith-fpc64-256",
        "vjepa2-vith",
        "V-JEPA 2 ViT-H/16",
        1280,
        32,
        16,
        256,
        "apache-2.0",
    ),
    "vjepa2_vit_giant": (
        "vjepa2_vit_giant",
        "facebook/vjepa2-vitg-fpc64-256",
        "vjepa2-vitg",
        "V-JEPA 2 ViT-g/16",
        1408,
        40,
        22,
        256,
        "apache-2.0",
    ),
    "vjepa2_vit_giant_384": (
        "vjepa2_vit_giant_384",
        "facebook/vjepa2-vitg-fpc64-384",
        "vjepa2-vitg-384",
        "V-JEPA 2 ViT-g/16 (384px)",
        1408,
        40,
        22,
        384,
        "apache-2.0",
    ),
}

_SOURCE_PARAMS = {
    "vjepa2_vit_large": 325_971_328,
    "vjepa2_vit_huge": 653_930_880,
    "vjepa2_vit_giant": 1_034_555_264,
    "vjepa2_vit_giant_384": 1_034_555_264,
}


def _fused_projection(
    state: dict[str, np.ndarray], prefix: str, width: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return QKV weight and bias in the order used by Lucid attention."""
    weights = [state[f"{prefix}.attention.{part}.weight"] for part in ("query", "key", "value")]
    biases = [state[f"{prefix}.attention.{part}.bias"] for part in ("query", "key", "value")]
    fused_weight = np.concatenate(weights, axis=0)
    fused_bias = np.concatenate(biases, axis=0)
    if fused_weight.shape != (3 * width, width):
        raise RuntimeError(
            f"{prefix}: fused QKV shape {fused_weight.shape} does not match "
            f"the configured width {width}"
        )
    return fused_weight, fused_bias


class VJEPA2Arch(Architecture):
    """Converter for one published V-JEPA 2 representation checkpoint."""

    def __init__(self, arch: str, tag: str) -> None:
        if arch not in _VARIANTS:
            raise KeyError(f"VJEPA2Arch: unknown architecture {arch!r}")
        if tag != "FPC64_256" and not (arch == "vjepa2_vit_giant_384" and tag == "FPC64_384"):
            raise KeyError(f"VJEPA2Arch: unsupported tag {tag!r} for {arch!r}")
        self.arch = arch
        self.tag = tag
        (
            self._factory,
            self._source_repo,
            self._slug,
            self._title,
            self._dim,
            self._depth,
            self._heads,
            self._image_size,
            self._license,
        ) = _VARIANTS[arch]
        import lucid.models as models

        self._model: Module = getattr(models, self._factory)()

    def source_state_dict(self) -> dict[str, object]:
        path = hf_hub_download(self._source_repo, "model.safetensors")
        source = load_file(path)
        converted: dict[str, np.ndarray] = {}

        for source_prefix, target_prefix, width, depth in (
            ("encoder", "encoder", self._dim, self._depth),
            ("encoder", "target_encoder", self._dim, self._depth),
        ):
            converted[f"{target_prefix}.patch_embed.weight"] = source[
                "encoder.embeddings.patch_embeddings.proj.weight"
            ]
            converted[f"{target_prefix}.patch_embed.bias"] = source[
                "encoder.embeddings.patch_embeddings.proj.bias"
            ]
            converted[f"{target_prefix}.norm.weight"] = source["encoder.layernorm.weight"]
            converted[f"{target_prefix}.norm.bias"] = source["encoder.layernorm.bias"]
            for index in range(depth):
                source_prefix = f"encoder.layer.{index}"
                fused_weight, fused_bias = _fused_projection(source, source_prefix, width)
                target_block = f"{target_prefix}.blocks.{index}"
                converted[f"{target_block}.attn.qkv.weight"] = fused_weight
                converted[f"{target_block}.attn.qkv.bias"] = fused_bias
                for name in ("norm1", "norm2"):
                    converted[f"{target_block}.{name}.weight"] = source[
                        f"{source_prefix}.{name}.weight"
                    ]
                    converted[f"{target_block}.{name}.bias"] = source[
                        f"{source_prefix}.{name}.bias"
                    ]
                for name in ("attn.proj", "mlp.fc1", "mlp.fc2"):
                    source_name = name.replace("attn.proj", "attention.proj")
                    converted[f"{target_block}.{name}.weight"] = source[
                        f"{source_prefix}.{source_name}.weight"
                    ]
                    converted[f"{target_block}.{name}.bias"] = source[
                        f"{source_prefix}.{source_name}.bias"
                    ]

        converted["predictor.predictor_embed.weight"] = source[
            "predictor.embeddings.predictor_embeddings.weight"
        ]
        converted["predictor.predictor_embed.bias"] = source[
            "predictor.embeddings.predictor_embeddings.bias"
        ]
        mask_tokens = source["predictor.embeddings.mask_tokens"]
        for index in range(int(mask_tokens.shape[0])):
            converted[f"predictor.mask_tokens.{index}"] = mask_tokens[index]
        for index in range(12):
            source_prefix = f"predictor.layer.{index}"
            fused_weight, fused_bias = _fused_projection(source, source_prefix, 384)
            target_block = f"predictor.predictor_blocks.{index}"
            converted[f"{target_block}.attn.qkv.weight"] = fused_weight
            converted[f"{target_block}.attn.qkv.bias"] = fused_bias
            for name in ("norm1", "norm2"):
                converted[f"{target_block}.{name}.weight"] = source[
                    f"{source_prefix}.{name}.weight"
                ]
                converted[f"{target_block}.{name}.bias"] = source[f"{source_prefix}.{name}.bias"]
            for name in ("attn.proj", "mlp.fc1", "mlp.fc2"):
                source_name = name.replace("attn.proj", "attention.proj")
                converted[f"{target_block}.{name}.weight"] = source[
                    f"{source_prefix}.{source_name}.weight"
                ]
                converted[f"{target_block}.{name}.bias"] = source[
                    f"{source_prefix}.{source_name}.bias"
                ]
        converted["predictor.predictor_norm.weight"] = source["predictor.layernorm.weight"]
        converted["predictor.predictor_norm.bias"] = source["predictor.layernorm.bias"]
        converted["predictor.predictor_proj.weight"] = source["predictor.proj.weight"]
        converted["predictor.predictor_proj.bias"] = source["predictor.proj.bias"]
        return converted

    def target_model(self) -> Module:
        return self._model

    def map_key(self, src_key: str) -> str | None:
        return src_key

    def spec(self) -> ConversionSpec:
        config = dataclasses.asdict(self._model.config)
        return ConversionSpec(
            model_name=self._factory,
            architecture=self.arch,
            repo_id=f"lucid-dl/{self._slug}",
            tag=self.tag,
            task="video-classification",
            model_type="vjepa2",
            source=f"{self._source_repo}/model.safetensors",
            license=self._license,
            num_classes=self._dim,
            config=config,
            preprocessing={
                "type": "video",
                "frames_per_clip": 64,
                "size": self._image_size,
                "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            },
            citation=_CITATION,
            title=self._title,
            paper_url=_PAPER_URL,
            datasets=["kinetics-700", "something-something-v2", "diving48"],
            meta={
                "num_params": _SOURCE_PARAMS[self.arch],
                "recipe": "facebookresearch/vjepa2 official Hub checkpoint",
                "metrics": {},
                "source_repo": self._source_repo,
                "encoder_num_heads": self._heads,
            },
        )


@register_arch("vjepa2_vit_large")
def _build_large(tag: str) -> Architecture:
    return VJEPA2Arch("vjepa2_vit_large", tag)


@register_arch("vjepa2_vit_huge")
def _build_huge(tag: str) -> Architecture:
    return VJEPA2Arch("vjepa2_vit_huge", tag)


@register_arch("vjepa2_vit_giant")
def _build_giant(tag: str) -> Architecture:
    return VJEPA2Arch("vjepa2_vit_giant", tag)


@register_arch("vjepa2_vit_giant_384")
def _build_giant_384(tag: str) -> Architecture:
    return VJEPA2Arch("vjepa2_vit_giant_384", tag)
