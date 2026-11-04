"""Pretrained-weight declarations for the Drifting Model family.

The authors released every checkpoint the paper reports, in Flax msgpack
form, at ``huggingface.co/Goodeat/drifting`` (MIT).  Five generators — the
four system-level models of Tables 5 and 6 plus the ablation default of
Table 8 — and the three MAE feature encoders they were trained against.
:mod:`tools.convert_weights.drifting` turns each into a Lucid safetensors
file and re-hosts it under the ``lucid-dl`` org.

What a caller needs to know:

**A latent generator emits an SD-VAE latent, not an image.**  The latent
models were trained against ``pcuenq/sd-vae-ft-mse-flax`` — the
``sd-vae-ft-mse`` finetune with the usual ``0.18215`` scale — so turning
``generate()``'s output into pixels takes that decoder.  The pixel models
emit ``[-1, 1]`` RGB directly.

**The feature encoders are only for training.**  Sampling never touches
them.  The MAE-640 encoders are large (the 640-wide ResNet's last stage is
5120 channels), which is why they are separate entries loaded on request by
:meth:`DriftingForImageGeneration.load_feature_encoder` rather than part of
any generator checkpoint.

The ``sha256`` of every entry is the digest of the *converted* file, which
exists only once the converter has been run; ``_PENDING_SHA256`` marks the
entries still waiting for it, and a download against one fails its
integrity check instead of loading an unverified file.
"""

from lucid.utils.transforms import ImageClassification
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights

__all__ = [
    "DriftingBase2Weights",
    "DriftingLarge2Weights",
    "DriftingBase16Weights",
    "DriftingLarge16Weights",
    "DriftingMAEWeights",
    "DriftingConvNeXtV2Weights",
]

# Replace with the digest ``python -m tools.convert_weights <name> --tag <TAG>``
# prints for the converted file, once it is uploaded.
_PENDING_SHA256 = "0" * 64

_SOURCE = "https://huggingface.co/Goodeat/drifting"

# Generators see no images; what consumes pixels is the VAE (latent) or
# nothing (pixel models emit [-1, 1] RGB).  The preset records that range.
_PRESET_256 = ImageClassification(
    crop_size=256, resize_size=256, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
)

# The architecture flags no shape check can see: RoPE, the norms and the
# in-context tokens change the function a checkpoint computes, and only
# some of them change a parameter shape.
_LIGHTNING = {
    "use_qknorm": True,
    "use_swiglu": True,
    "use_rope": True,
    "use_rmsnorm": True,
    "n_cls_tokens": 16,
    "noise_classes": 64,
    "noise_coords": 32,
}

_RECIPE_COMMON = {
    "optimizer": "adamw",
    "adam_b1": 0.9,
    "adam_b2": 0.95,
    "lr_schedule": "const",
    "lr_warmup_init": 1e-6,
    "max_grad_norm": 2.0,
    "gen_per_label": 64,
    "eval_cfg_list": [1.0, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5],
}


@register_weights("drifting_base_2")
@register_weights("drifting_base_2_gen")
class DriftingBase2Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.drifting_base_2`.

    Two checkpoints share this architecture.  ``IMAGENET1K_256`` is the
    system-level ``B/2`` of Table 5; ``IMAGENET1K_256_ABLATION`` is the
    100-epoch ablation default every study in Section 5 runs on, and it
    was trained against the narrower MAE-256 with different guidance
    sampling — which is why its tag carries ``config`` overrides that the
    factory applies.

    Notes
    -----
    Reference: Deng, Li, Li, Du, and He, arXiv:2602.04770, 2026, Tables 4,
    5 and 8.  FID 1.75 (paper) / 1.74 (release) at guidance 1.1, one
    function evaluation; the ablation checkpoint is 8.46 / 8.49 at 2.0.

    Examples
    --------
    >>> from lucid.models.generative.drifting._weights import DriftingBase2Weights
    >>> list(DriftingBase2Weights.__members__)
    ['IMAGENET1K_256', 'IMAGENET1K_256_ABLATION', 'DEFAULT']
    >>> DriftingBase2Weights.DEFAULT is DriftingBase2Weights.IMAGENET1K_256
    True
    """

    IMAGENET1K_256 = WeightEntry(
        url=f"{HUB_BASE}/drifting-base-2/resolve/main/IMAGENET1K_256/model.safetensors",
        sha256=_PENDING_SHA256,
        num_classes=1000,
        transforms=_PRESET_256,
        requires_config={**_LIGHTNING, "patch_size": 2, "sample_size": 32},
        meta={
            "tag": "IMAGENET1K_256",
            "source": f"{_SOURCE}/tree/main/models/gen/jax/latent_B_sota",
            "license": "mit",
            "num_params": 132_708_880,
            "vae": "pcuenq/sd-vae-ft-mse-flax (scale 0.18215)",
            "feature_encoder": "LATENT_640",
            "metrics": {"imagenet-1k": {"fid": 1.75, "inception_score": 263.2, "cfg": 1.1}},
            "recipe": {
                **_RECIPE_COMMON,
                "config_path": "configs/gen/latent_sota_B.yaml",
                "learning_rate": 4e-4,
                "warmup_steps": 10_000,
                "weight_decay": 0.0,
                "train_batch_size": 128,
                "total_steps": 200_000,
                "ema_decay": 0.9999,
                "push_per_step": 128,
            },
        },
    )
    IMAGENET1K_256_ABLATION = WeightEntry(
        url=f"{HUB_BASE}/drifting-base-2/resolve/main/IMAGENET1K_256_ABLATION/model.safetensors",
        sha256=_PENDING_SHA256,
        num_classes=1000,
        transforms=_PRESET_256,
        requires_config={**_LIGHTNING, "patch_size": 2, "sample_size": 32},
        meta={
            "tag": "IMAGENET1K_256_ABLATION",
            "source": f"{_SOURCE}/tree/main/models/gen/jax/ablation",
            "license": "mit",
            "num_params": 132_708_880,
            "vae": "pcuenq/sd-vae-ft-mse-flax (scale 0.18215)",
            "feature_encoder": "LATENT_256",
            "metrics": {"imagenet-1k": {"fid": 8.46, "inception_score": 144.0, "cfg": 2.0}},
            # The config this tag was trained with where it differs from
            # ``B/2``'s: applied by the factory, overridable by the caller.
            "config": {
                "attn_fp32": True,
                "mae_base_channels": 256,
                "neg_cfg_pw": 3.0,
                "pos_per_sample": 64,
                "neg_per_sample": 16,
            },
            "recipe": {
                **_RECIPE_COMMON,
                "config_path": "configs/gen/latent_ablation.yaml",
                "learning_rate": 2e-4,
                "warmup_steps": 5_000,
                "weight_decay": 0.01,
                "train_batch_size": 64,
                "total_steps": 30_000,
                "ema_decay": 0.999,
                "push_per_step": 64,
                "eval_cfg_list": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
            },
        },
    )
    DEFAULT = IMAGENET1K_256


@register_weights("drifting_large_2")
@register_weights("drifting_large_2_gen")
class DriftingLarge2Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.drifting_large_2`.

    Notes
    -----
    Reference: Deng, Li, Li, Du, and He, arXiv:2602.04770, 2026, Tables 5
    and 8 — the paper's headline latent model.  FID 1.54 (paper) / 1.53
    (release) at guidance 1.0, one function evaluation.

    Examples
    --------
    >>> from lucid.models.generative.drifting._weights import DriftingLarge2Weights
    >>> DriftingLarge2Weights.DEFAULT is DriftingLarge2Weights.IMAGENET1K_256
    True
    """

    IMAGENET1K_256 = WeightEntry(
        url=f"{HUB_BASE}/drifting-large-2/resolve/main/IMAGENET1K_256/model.safetensors",
        sha256=_PENDING_SHA256,
        num_classes=1000,
        transforms=_PRESET_256,
        requires_config={**_LIGHTNING, "patch_size": 2, "sample_size": 32},
        meta={
            "tag": "IMAGENET1K_256",
            "source": f"{_SOURCE}/tree/main/models/gen/jax/latent_L_sota",
            "license": "mit",
            "vae": "pcuenq/sd-vae-ft-mse-flax (scale 0.18215)",
            "feature_encoder": "LATENT_640",
            "metrics": {"imagenet-1k": {"fid": 1.54, "inception_score": 258.9, "cfg": 1.0}},
            "recipe": {
                **_RECIPE_COMMON,
                "config_path": "configs/gen/latent_sota_L.yaml",
                "learning_rate": 4e-4,
                "warmup_steps": 10_000,
                "weight_decay": 0.01,
                "train_batch_size": 128,
                "total_steps": 200_000,
                "ema_decay": 0.9999,
                "push_per_step": 64,
            },
        },
    )
    DEFAULT = IMAGENET1K_256


@register_weights("drifting_base_16")
@register_weights("drifting_base_16_gen")
class DriftingBase16Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.drifting_base_16`.

    Notes
    -----
    Reference: Deng, Li, Li, Du, and He, arXiv:2602.04770, 2026, Tables 6,
    8 and 10.  Pixel space, FID 1.76 (paper) / 1.73 (release) at guidance
    1.0.  Trained against MAE-640 (pixel) and ConvNeXt-V2-Base features.

    Examples
    --------
    >>> from lucid.models.generative.drifting._weights import DriftingBase16Weights
    >>> DriftingBase16Weights.DEFAULT is DriftingBase16Weights.IMAGENET1K_256
    True
    """

    IMAGENET1K_256 = WeightEntry(
        url=f"{HUB_BASE}/drifting-base-16/resolve/main/IMAGENET1K_256/model.safetensors",
        sha256=_PENDING_SHA256,
        num_classes=1000,
        transforms=_PRESET_256,
        requires_config={**_LIGHTNING, "patch_size": 16, "sample_size": 256},
        meta={
            "tag": "IMAGENET1K_256",
            "source": f"{_SOURCE}/tree/main/models/gen/jax/pixel_B_sota",
            "license": "mit",
            "feature_encoder": "PIXEL_640 + ConvNeXt-V2-Base-22k",
            "metrics": {"imagenet-1k": {"fid": 1.76, "inception_score": 299.7, "cfg": 1.0}},
            "recipe": {
                **_RECIPE_COMMON,
                "config_path": "configs/gen/pixel_sota_B.yaml",
                "learning_rate": 2e-4,
                "warmup_steps": 10_000,
                "weight_decay": 0.01,
                "train_batch_size": 128,
                "total_steps": 100_000,
                "ema_decay": 0.999,
                "push_per_step": 128,
            },
        },
    )
    DEFAULT = IMAGENET1K_256


@register_weights("drifting_large_16")
@register_weights("drifting_large_16_gen")
class DriftingLarge16Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.drifting_large_16`.

    Notes
    -----
    Reference: Deng, Li, Li, Du, and He, arXiv:2602.04770, 2026, Tables 6,
    8 and 10.  Pixel space, FID 1.61 (paper) / 1.62 (release) at guidance
    1.0.

    Examples
    --------
    >>> from lucid.models.generative.drifting._weights import DriftingLarge16Weights
    >>> DriftingLarge16Weights.DEFAULT is DriftingLarge16Weights.IMAGENET1K_256
    True
    """

    IMAGENET1K_256 = WeightEntry(
        url=f"{HUB_BASE}/drifting-large-16/resolve/main/IMAGENET1K_256/model.safetensors",
        sha256=_PENDING_SHA256,
        num_classes=1000,
        transforms=_PRESET_256,
        requires_config={**_LIGHTNING, "patch_size": 16, "sample_size": 256},
        meta={
            "tag": "IMAGENET1K_256",
            "source": f"{_SOURCE}/tree/main/models/gen/jax/pixel_L_sota",
            "license": "mit",
            "feature_encoder": "PIXEL_640 + ConvNeXt-V2-Base-22k",
            "metrics": {"imagenet-1k": {"fid": 1.61, "inception_score": 307.5, "cfg": 1.0}},
            "recipe": {
                **_RECIPE_COMMON,
                "config_path": "configs/gen/pixel_sota_L.yaml",
                "learning_rate": 4e-4,
                "warmup_steps": 10_000,
                "weight_decay": 0.01,
                "train_batch_size": 128,
                "total_steps": 100_000,
                "ema_decay": 0.999,
                "push_per_step": 128,
            },
        },
    )
    DEFAULT = IMAGENET1K_256


class DriftingMAEWeights(WeightsEnum):
    r"""The three released MAE feature encoders (encoder half only).

    Loaded into :attr:`DriftingFeatureEncoder.mae` by
    :meth:`DriftingForImageGeneration.load_feature_encoder`, which picks
    the tag matching the config.  The MAE decoder and the classification
    head are dropped by the converter: the drifting loss reads only the
    encoder's multi-scale features (Appendix A.3).

    Notes
    -----
    Reference: Deng, Li, Li, Du, and He, arXiv:2602.04770, 2026, Table 3
    and Table 8.  ``LATENT_640`` and ``PIXEL_640`` are 1280-epoch,
    classification-finetuned encoders; ``LATENT_256`` is the 192-epoch
    ablation encoder.

    Examples
    --------
    >>> from lucid.models.generative.drifting._weights import DriftingMAEWeights
    >>> list(DriftingMAEWeights.__members__)
    ['LATENT_640', 'LATENT_256', 'PIXEL_640']
    """

    LATENT_640 = WeightEntry(
        url=f"{HUB_BASE}/drifting-mae/resolve/main/LATENT_640/model.safetensors",
        sha256=_PENDING_SHA256,
        num_classes=1000,
        transforms=_PRESET_256,
        meta={
            "tag": "LATENT_640",
            "source": f"{_SOURCE}/tree/main/models/mae/jax/mae_latent_640",
            "license": "mit",
            "config": {"in_channels": 4, "mae_base_channels": 640, "mae_input_patch_size": 1},
        },
    )
    LATENT_256 = WeightEntry(
        url=f"{HUB_BASE}/drifting-mae/resolve/main/LATENT_256/model.safetensors",
        sha256=_PENDING_SHA256,
        num_classes=1000,
        transforms=_PRESET_256,
        meta={
            "tag": "LATENT_256",
            "source": f"{_SOURCE}/tree/main/models/mae/jax/mae_latent_256",
            "license": "mit",
            "config": {"in_channels": 4, "mae_base_channels": 256, "mae_input_patch_size": 1},
        },
    )
    PIXEL_640 = WeightEntry(
        url=f"{HUB_BASE}/drifting-mae/resolve/main/PIXEL_640/model.safetensors",
        sha256=_PENDING_SHA256,
        num_classes=1000,
        transforms=_PRESET_256,
        meta={
            "tag": "PIXEL_640",
            "source": f"{_SOURCE}/tree/main/models/mae/jax/mae_pixel_640",
            "license": "mit",
            "config": {"in_channels": 3, "mae_base_channels": 640, "mae_input_patch_size": 8},
        },
    )


class DriftingConvNeXtV2Weights(WeightsEnum):
    r"""ConvNeXt-V2-Base (ImageNet-22k, 224) — the pixel models' second encoder.

    The release loads ``facebook/convnextv2-base-22k-224`` and reads its
    four stage outputs; the classifier is dropped on conversion.  The
    weights are CC-BY-NC-4.0 from their source, unlike the MIT generators.

    Notes
    -----
    Reference: Deng, Li, Li, Du, and He, arXiv:2602.04770, 2026,
    Appendix A.4 and Table 9; Woo et al., *"ConvNeXt V2"*, CVPR 2023.

    Examples
    --------
    >>> from lucid.models.generative.drifting._weights import DriftingConvNeXtV2Weights
    >>> DriftingConvNeXtV2Weights.DEFAULT is DriftingConvNeXtV2Weights.IMAGENET22K_224
    True
    """

    IMAGENET22K_224 = WeightEntry(
        url=f"{HUB_BASE}/drifting-convnext-v2/resolve/main/IMAGENET22K_224/model.safetensors",
        sha256=_PENDING_SHA256,
        num_classes=1000,
        transforms=_PRESET_256,
        meta={
            "tag": "IMAGENET22K_224",
            "source": "https://huggingface.co/facebook/convnextv2-base-22k-224",
            "license": "cc-by-nc-4.0",
        },
    )
    DEFAULT = IMAGENET22K_224
