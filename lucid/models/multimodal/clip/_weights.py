"""Pretrained-weight declarations for CLIP.

The four ViT releases OpenAI published, converted by
:mod:`tools.convert_weights.clip` from the ``transformers``-layout
re-hosting of the original checkpoints.  These are the authors' own
trained parameters — no reimplementation was trained to produce them.

Each was checked numerically against the reference before publication,
not merely loaded: with the converted weights in place, Lucid's
``logits_per_image`` agrees with the reference to a relative error
between 3.8e-07 and 1.5e-06, which is float32 round-off.  That check is
the one that matters here, because the conversion transposes two
projection matrices and fuses three attention matrices into one — and
the transpose of ``text_projection`` is *square* in the base models, so
getting it backwards would have produced a model of exactly the right
shape that silently ranks nothing.

Preprocessing is CLIPModel's own, and is not ImageNet's: bicubic resize to
the tower's resolution, centre crop, and the dataset statistics below,
which come from the released preprocessor configuration rather than
from the paper.

One measured detail, because it looks like a discrepancy and is not.
Every released checkpoint stores ``logit_scale = 4.605170``, whose
exponential is 100.000006 — the trained temperature sat *at* the
paper's ceiling, which is what "clipped to prevent scaling the logits
by more than 100" produces when the constraint binds. Lucid applies
that cap in :attr:`CLIPModel.scale`, so it returns exactly 100.0 where the
reference — which clamps in its training loop rather than its model —
returns 100.000006. The resulting relative difference is 6e-08, three
orders below the float32 agreement reported above, and the cap is doing
what the paper describes rather than distorting the checkpoint.

No accuracy metrics are recorded.  The paper reports zero-shot numbers
against prompt ensembles it does not fully enumerate, so a single
figure here would be attributing a score to a protocol this repository
has not reproduced.
"""

from lucid.utils.transforms import ImageClassification
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights

__all__ = [
    "CLIPViTBase32Weights",
    "CLIPViTBase16Weights",
    "CLIPViTLarge14Weights",
    "CLIPViTLarge14_336Weights",
]

_MEAN = (0.48145466, 0.4578275, 0.40821073)
_STD = (0.26862954, 0.26130258, 0.27577711)

_PRESET_224 = ImageClassification(
    crop_size=224, resize_size=224, mean=_MEAN, std=_STD, interpolation="bicubic"
)
_PRESET_336 = ImageClassification(
    crop_size=336, resize_size=336, mean=_MEAN, std=_STD, interpolation="bicubic"
)

_SOURCE = "openai/CLIPModel (via transformers re-hosting)"


@register_weights("clip_vit_base_32")
@register_weights("clip_vit_base_32_zero_shot")
class CLIPViTBase32Weights(WeightsEnum):
    """Pretrained weights for :func:`lucid.models.clip_vit_base_32`.

    Trained on WIT-400M, the paper's 400-million image-text pair
    dataset.  ``num_classes`` is 0 because CLIPModel has no fixed label set —
    that is the point of it.

    Examples
    --------
    >>> from lucid.models.multimodal.clip._weights import CLIPViTBase32Weights
    >>> list(CLIPViTBase32Weights.__members__)
    ['OPENAI_WIT400M', 'DEFAULT']

    ``DEFAULT`` is an alias rather than a fourth entry, so a bare
    ``pretrained=True`` and the tag it resolves to cannot drift apart.

    >>> CLIPViTBase32Weights.DEFAULT is CLIPViTBase32Weights.OPENAI_WIT400M
    True
    >>> CLIPViTBase32Weights.OPENAI_WIT400M.num_classes
    0
    """

    OPENAI_WIT400M = WeightEntry(
        url=f"{HUB_BASE}/clip-vit-base-32/resolve/main/OPENAI_WIT400M/model.safetensors",
        sha256="f1ed9118391f459539ecac57ba7c267804f93107d6353187c52c9ba14a5313a1",
        num_classes=0,
        transforms=_PRESET_224,
        meta={
            "tag": "OPENAI_WIT400M",
            "source": _SOURCE,
            "license": "mit",
            "num_params": 151_277_313,
            "file_size_mb": 577.1,
        },
    )
    DEFAULT = OPENAI_WIT400M


@register_weights("clip_vit_base_16")
@register_weights("clip_vit_base_16_zero_shot")
class CLIPViTBase16Weights(WeightsEnum):
    """Pretrained weights for :func:`lucid.models.clip_vit_base_16`.

    The same tower shape as ViT-B/32 with 16-pixel patches, so it reads
    196 patch tokens where the other reads 49 — four times the attention
    for the same 151M-scale budget, and the paper's better-scoring base
    model because of it.

    Examples
    --------
    >>> from lucid.models.multimodal.clip._weights import CLIPViTBase16Weights
    >>> list(CLIPViTBase16Weights.__members__)
    ['OPENAI_WIT400M', 'DEFAULT']

    ``DEFAULT`` is an alias rather than a fourth entry, so a bare
    ``pretrained=True`` and the tag it resolves to cannot drift apart.

    >>> CLIPViTBase16Weights.DEFAULT is CLIPViTBase16Weights.OPENAI_WIT400M
    True
    >>> CLIPViTBase16Weights.OPENAI_WIT400M.num_classes
    0
    """

    OPENAI_WIT400M = WeightEntry(
        url=f"{HUB_BASE}/clip-vit-base-16/resolve/main/OPENAI_WIT400M/model.safetensors",
        sha256="5f6a67b9f82e51e6ed543f76035338c0850a5f17e1cb4e603346b200f9a14ea5",
        num_classes=0,
        transforms=_PRESET_224,
        meta={
            "tag": "OPENAI_WIT400M",
            "source": _SOURCE,
            "license": "mit",
            "num_params": 149_620_737,
            "file_size_mb": 570.8,
        },
    )
    DEFAULT = OPENAI_WIT400M


@register_weights("clip_vit_large_14")
@register_weights("clip_vit_large_14_zero_shot")
class CLIPViTLarge14Weights(WeightsEnum):
    """Pretrained weights for :func:`lucid.models.clip_vit_large_14`.

    The paper's best variant, and the only one whose text tower is
    widened along with the image tower — 768 wide with 12 heads against
    the base models' 512 and 8, at the same 12 layers.

    Examples
    --------
    >>> from lucid.models.multimodal.clip._weights import CLIPViTLarge14Weights
    >>> list(CLIPViTLarge14Weights.__members__)
    ['OPENAI_WIT400M', 'DEFAULT']

    ``DEFAULT`` is an alias rather than a fourth entry, so a bare
    ``pretrained=True`` and the tag it resolves to cannot drift apart.

    >>> CLIPViTLarge14Weights.DEFAULT is CLIPViTLarge14Weights.OPENAI_WIT400M
    True
    >>> CLIPViTLarge14Weights.OPENAI_WIT400M.num_classes
    0
    """

    OPENAI_WIT400M = WeightEntry(
        url=f"{HUB_BASE}/clip-vit-large-14/resolve/main/OPENAI_WIT400M/model.safetensors",
        sha256="ba14d29218993b5ecba02ed93eff21a2b69d1eb857d47714626dd7ce4ac4a27f",
        num_classes=0,
        transforms=_PRESET_224,
        meta={
            "tag": "OPENAI_WIT400M",
            "source": _SOURCE,
            "license": "mit",
            "num_params": 427_616_513,
            "file_size_mb": 1631.3,
        },
    )
    DEFAULT = OPENAI_WIT400M


@register_weights("clip_vit_large_14_336")
@register_weights("clip_vit_large_14_336_zero_shot")
class CLIPViTLarge14_336Weights(WeightsEnum):
    """Pretrained weights for :func:`lucid.models.clip_vit_large_14_336`.

    Notes
    -----
    Not interchangeable with :class:`CLIPViTLarge14Weights` despite the
    identical tower shape: the positional table has 577 rows here
    against 257 there, so loading either into the other's model fails on
    that one tensor.

    Examples
    --------
    >>> from lucid.models.multimodal.clip._weights import CLIPViTLarge14_336Weights
    >>> list(CLIPViTLarge14_336Weights.__members__)
    ['OPENAI_WIT400M', 'DEFAULT']

    ``DEFAULT`` is an alias rather than a fourth entry, so a bare
    ``pretrained=True`` and the tag it resolves to cannot drift apart.

    >>> CLIPViTLarge14_336Weights.DEFAULT is CLIPViTLarge14_336Weights.OPENAI_WIT400M
    True
    >>> CLIPViTLarge14_336Weights.OPENAI_WIT400M.num_classes
    0
    """

    OPENAI_WIT400M = WeightEntry(
        url=(f"{HUB_BASE}/clip-vit-large-14-336/resolve/main/" "OPENAI_WIT400M/model.safetensors"),
        sha256="961cfbc43d8e08c52debb0a513a22ec29d8ea01ad117bdbdee4ac096eb74515e",
        num_classes=0,
        transforms=_PRESET_336,
        meta={
            "tag": "OPENAI_WIT400M",
            "source": _SOURCE,
            "license": "mit",
            "num_params": 427_944_193,
            "file_size_mb": 1632.5,
        },
    )
    DEFAULT = OPENAI_WIT400M
