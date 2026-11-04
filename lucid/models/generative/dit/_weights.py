"""Pretrained-weight declarations for the DiT family.

Two checkpoints were released, and they are the same model: the XL
backbone at patch 2, trained on class-conditional ImageNet, at the two
resolutions the paper reports state-of-the-art FID for.  The other
eleven configurations in the scaling study were trained but never
published, which is why ``dit_xlarge_2`` (and its ``_gen`` wrapper) is
the only factory here whose ``pretrained=True`` loads anything — the
others refuse it rather than hand back an untrained model.

The weights are converted from the ``facebook/DiT-XL-2-*`` repos and
re-hosted under the ``lucid-dl`` org.

Two things a caller needs to know before using them.

**The licence is non-commercial.**  The architecture is not encumbered,
but these tensors carry CC-BY-NC-4.0 from their source and a converted
copy does not launder that.

**A DiT emits a latent, not an image.**  The paper works in the latent
space of "the off-the-shelf pre-trained VAE from Stable Diffusion" — the
``sd-vae-ft-ema`` finetune — so turning ``generate()``'s output into
pixels takes that decoder.  Lucid already implements it, as
:class:`~lucid.models.generative.stable_diffusion.AutoencoderKL`, and
the configuration it defaults to is exactly the one these checkpoints
were trained against.
"""

from lucid.utils.transforms import ImageClassification
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights

__all__ = ["DiTXLarge2Weights"]

# A diffusion model in latent space sees images mapped to [-1, 1]; there
# is no dataset mean to subtract, because what consumes the pixels is an
# autoencoder rather than a classifier.
_PRESET_256 = ImageClassification(
    crop_size=256, resize_size=256, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
)
_PRESET_512 = ImageClassification(
    crop_size=512, resize_size=512, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
)


@register_weights("dit_xlarge_2")
@register_weights("dit_xlarge_2_gen")
class DiTXLarge2Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.dit_xlarge_2`.

    ``num_classes`` is 1000: unlike the unconditional diffusion families
    here, DiT is class-conditional, and the label embedding carries one
    extra row for the null token classifier-free guidance drops to.

    Notes
    -----
    Reference: Peebles and Xie, ICCV 2023 (arXiv:2212.09748), Table 2
    and Table 3.  The reported FIDs — 2.27 at 256 and 3.04 at 512 — are
    measured with classifier-free guidance at scale 1.5 over **250 DDPM
    sampling steps**, which :meth:`DiTForImageGeneration.generate` reaches
    as ``steps=250, eta=1.0``; the default ``eta=0.0`` is DDIM's
    deterministic step, matching the released pipeline's scheduler but
    not the protocol behind those numbers.

    The two tags differ only in the latent they were trained on — a
    32-side latent for 256-pixel images and 64 for 512 — so the 512 tag
    needs a model built at that size, which its factory override does.
    Their parameter counts are identical, because the positional table
    that grows with the latent is a non-persistent buffer rather than a
    parameter and is rebuilt from the config on construction.

    Examples
    --------
    >>> from lucid.models.generative.dit._weights import DiTXLarge2Weights
    >>> list(DiTXLarge2Weights.__members__)
    ['IMAGENET1K_256', 'IMAGENET1K_512', 'DEFAULT']

    ``DEFAULT`` is an alias rather than a fourth entry, so a bare
    ``pretrained=True`` and the tag it resolves to cannot drift apart.

    >>> DiTXLarge2Weights.DEFAULT is DiTXLarge2Weights.IMAGENET1K_256
    True
    >>> DiTXLarge2Weights.IMAGENET1K_256.num_classes
    1000
    """

    IMAGENET1K_256 = WeightEntry(
        url=f"{HUB_BASE}/dit-xlarge-2/resolve/main/IMAGENET1K_256/model.safetensors",
        sha256="996d0cbdfd290d647dc4ace3b70aa4105c75f12dfcad0195eb4707d44c2003de",
        num_classes=1000,
        transforms=_PRESET_256,
        meta={
            "tag": "IMAGENET1K_256",
            "source": "facebook/DiT-XL-2-256 (diffusers layout)",
            "license": "cc-by-nc-4.0",
            "num_params": 674_834_720,
            "latent_size": 32,
            "vae": "stabilityai/sd-vae-ft-ema",
            "metrics": {"imagenet-1k": {"fid": 2.27}},
        },
    )
    IMAGENET1K_512 = WeightEntry(
        url=f"{HUB_BASE}/dit-xlarge-2/resolve/main/IMAGENET1K_512/model.safetensors",
        sha256="93e3aa970d53daf4dcce471364630d63646ba4853f343ed96badf5101444073a",
        num_classes=1000,
        transforms=_PRESET_512,
        meta={
            "tag": "IMAGENET1K_512",
            "source": "facebook/DiT-XL-2-512 (diffusers layout)",
            "license": "cc-by-nc-4.0",
            "num_params": 674_834_720,
            "latent_size": 64,
            "vae": "stabilityai/sd-vae-ft-ema",
            "metrics": {"imagenet-1k": {"fid": 3.04}},
        },
    )
    DEFAULT = IMAGENET1K_256
