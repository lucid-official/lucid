"""Pretrained SafeTensors declarations for the V-JEPA 2 family.

The four representation checkpoints are the public FPC64 releases: ViT-L,
ViT-H and ViT-g at 256 pixels, plus the ViT-g 384-pixel evaluation release.
They are converted into Lucid's fused-QKV / EMA-target layout and hosted under
the ``lucid-dl`` organisation.

Each tag carries the released inference preprocessing:
:class:`~lucid.utils.transforms.VideoClassification`, which runs the ImageNet
evaluation pipeline on every frame — shortest side to ``int(crop * 256 / 224)``,
centre crop, rescale, ImageNet normalise.  *Decoding* is not part of it and
cannot be: reading a container needs a library Lucid's compute path may not
take (H4).  Hand it an already-decoded clip; which frames to hand it is the
decoder's decision, and ``meta`` records the released frame count.
"""

from lucid.utils.transforms import VideoClassification
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights

__all__ = [
    "VJEPA2ViTLargeWeights",
    "VJEPA2ViTHugeWeights",
    "VJEPA2ViTGiantWeights",
    "VJEPA2ViTGiant384Weights",
]

_PRESET_256 = VideoClassification(crop_size=256)
_PRESET_384 = VideoClassification(crop_size=384)


def _requires(*, image_size: int, dim: int, depth: int, num_heads: int) -> dict[str, object]:
    """Return the architecture fields that must match a checkpoint."""
    return {
        "image_size": image_size,
        "patch_size": 16,
        "tubelet_size": 2,
        "in_channels": 3,
        "dim": dim,
        "depth": depth,
        "num_heads": num_heads,
        "mlp_ratio": 4.0 if dim != 1408 else 48.0 / 11.0,
        "predictor_dim": 384,
        "predictor_depth": 12,
        "predictor_heads": 12,
        "predictor_mlp_ratio": 4.0,
        "predictor_num_mask_tokens": 10,
    }


def _url(slug: str, tag: str) -> str:
    return f"{HUB_BASE}/{slug}/resolve/main/{tag}/model.safetensors"


def _meta(
    *,
    tag: str,
    source_repo: str,
    license_name: str,
    num_params: int,
    file_size_mb: float,
    image_size: int,
    dim: int,
    depth: int,
    num_heads: int,
) -> dict[str, object]:
    return {
        "tag": tag,
        "source": f"{source_repo}/model.safetensors",
        "license": license_name,
        "num_params": num_params,
        "file_size_mb": file_size_mb,
        "encoder_num_heads": num_heads,
        "preprocessing": {
            "type": "video",
            "frames_per_clip": 64,
            "size": image_size,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "encoder": {"dim": dim, "depth": depth, "num_heads": num_heads},
        "predictor": {"dim": 384, "depth": 12, "num_heads": 12},
    }


@register_weights("vjepa2_vit_large")
class VJEPA2ViTLargeWeights(WeightsEnum):
    r"""FPC64 ViT-L/16 representation weights at 256 pixels.

    Converted from ``facebook/vjepa2-vitl-fpc64-256``: separate query, key
    and value projections fused, and the context encoder mirrored into the
    EMA target.  Reproduces its source to ``1.7e-5`` relative over the
    encoder and the masked predictor — the closest of the four.
    """

    FPC64_256 = WeightEntry(
        url=_url("vjepa2-vitl", "FPC64_256"),
        sha256="ad1c6f0894438b0a549ad3fff0a49a5f563c663ca5f2c9474be43a9e58469b3e",
        num_classes=1024,
        transforms=_PRESET_256,
        requires_config=_requires(image_size=256, dim=1024, depth=24, num_heads=16),
        meta=_meta(
            tag="FPC64_256",
            source_repo="facebook/vjepa2-vitl-fpc64-256",
            license_name="mit",
            num_params=325_971_328,
            file_size_mb=2402.79,
            image_size=256,
            dim=1024,
            depth=24,
            num_heads=16,
        ),
    )
    DEFAULT = FPC64_256


@register_weights("vjepa2_vit_huge")
class VJEPA2ViTHugeWeights(WeightsEnum):
    r"""FPC64 ViT-H/16 representation weights at 256 pixels.

    Converted from ``facebook/vjepa2-vith-fpc64-256`` the same way as the
    ViT-L tag.  Reproduces its source to ``5.3e-5`` relative; the residual
    grows with depth, and this network is eight blocks deeper.
    """

    FPC64_256 = WeightEntry(
        url=_url("vjepa2-vith", "FPC64_256"),
        sha256="4ccd2fe7944df5970c0cd10bdbcd75a7b5a25b2f095601859600650daa4f0780",
        num_classes=1280,
        transforms=_PRESET_256,
        requires_config=_requires(image_size=256, dim=1280, depth=32, num_heads=16),
        meta=_meta(
            tag="FPC64_256",
            source_repo="facebook/vjepa2-vith-fpc64-256",
            license_name="apache-2.0",
            num_params=653_930_880,
            file_size_mb=4904.19,
            image_size=256,
            dim=1280,
            depth=32,
            num_heads=16,
        ),
    )
    DEFAULT = FPC64_256


@register_weights("vjepa2_vit_giant")
class VJEPA2ViTGiantWeights(WeightsEnum):
    r"""FPC64 ViT-g/16 representation weights at 256 pixels.

    Converted from ``facebook/vjepa2-vitg-fpc64-256``.  This is the encoder
    V-JEPA 2-AC is post-trained from, and the only one that widens its
    feed-forward to ``48 / 11``.  Reproduces its source to ``1.0e-4``
    relative — just over the parity tool's global bound, which was left
    where it is because the gap tracks depth rather than a defect.
    """

    FPC64_256 = WeightEntry(
        url=_url("vjepa2-vitg", "FPC64_256"),
        sha256="8bbf146683b9aea9592fc2b85a7c1506c03edce08b225fe68ddd96443c0093e3",
        num_classes=1408,
        transforms=_PRESET_256,
        requires_config=_requires(image_size=256, dim=1408, depth=40, num_heads=22),
        meta=_meta(
            tag="FPC64_256",
            source_repo="facebook/vjepa2-vitg-fpc64-256",
            license_name="apache-2.0",
            num_params=1_034_555_264,
            file_size_mb=7807.77,
            image_size=256,
            dim=1408,
            depth=40,
            num_heads=22,
        ),
    )
    DEFAULT = FPC64_256


@register_weights("vjepa2_vit_giant_384")
class VJEPA2ViTGiant384Weights(WeightsEnum):
    r"""FPC64 ViT-g/16 representation weights at 384 pixels.

    The same network as the 256-pixel ViT-g tag, read over a wider grid:
    the rotary geometry is computed from token positions, so no parameter
    changes and no table is interpolated.  Reproduces its source to
    ``1.9e-4`` relative over 18432 tokens against the other's 8192.
    """

    FPC64_384 = WeightEntry(
        url=_url("vjepa2-vitg-384", "FPC64_384"),
        sha256="60e3349300380d4c0c3b3cfc7c21b8acc474eddbb7f8b0eb5cc54e0c710d889a",
        num_classes=1408,
        transforms=_PRESET_384,
        requires_config=_requires(image_size=384, dim=1408, depth=40, num_heads=22),
        meta=_meta(
            tag="FPC64_384",
            source_repo="facebook/vjepa2-vitg-fpc64-384",
            license_name="apache-2.0",
            num_params=1_034_555_264,
            file_size_mb=7807.77,
            image_size=384,
            dim=1408,
            depth=40,
            num_heads=22,
        ),
    )
    DEFAULT = FPC64_384
