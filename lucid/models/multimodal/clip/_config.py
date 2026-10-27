"""CLIP configuration (Radford et al., 2021)."""

from dataclasses import dataclass
from typing import ClassVar, override

from lucid.models._meta import model_family_meta
from lucid.models.multimodal._common._config import MultimodalModelConfig

__all__ = ["CLIPConfig"]


@model_family_meta(
    canonical_name="CLIP",
    citation=(
        'Radford, Alec, et al. "Learning Transferable Visual Models '
        'From Natural Language Supervision." ICML, 2021, pp. 8748–8763.'
    ),
    theory=r"""
    CLIP learns a joint embedding space for images and text by
    contrasting the pairings that actually occurred in a batch against
    the ones that did not.  Given :math:`N` image-caption pairs, an image
    tower :math:`f` and a text tower :math:`g` produce :math:`N`
    embeddings each; the objective maximises the cosine similarity of the
    :math:`N` matched pairs and minimises it for the :math:`N^2 - N`
    mismatched ones.  Writing :math:`u_i = f(I_i)/\|f(I_i)\|` and
    :math:`v_j = g(T_j)/\|g(T_j)\|`, the logits are

    .. math::

        \ell_{ij} = \exp(t)\, u_i^\top v_j,

    and the loss is a cross-entropy over rows and columns alike,

    .. math::

        \mathcal{L} = \tfrac{1}{2}\big(
            \mathrm{CE}(\ell, \mathrm{diag}) +
            \mathrm{CE}(\ell^\top, \mathrm{diag})\big).

    The symmetry matters: a one-sided loss lets one tower collapse onto
    the other's geometry, because only one direction is ever scored.

    The scalar :math:`t` is learned rather than fixed.  It is stored in
    log space and exponentiated, so the optimiser moves it
    multiplicatively and it cannot become negative — a temperature is a
    scale, and gradient descent on the scale itself would happily cross
    zero.  The paper initialises it at the equivalent of
    :math:`\tau = 0.07` and caps the resulting multiplier at 100, which
    keeps the softmax from saturating early in training when the
    embeddings are still arbitrary.

    What makes the representation transferable is that the text tower is
    an open-ended classifier.  A fixed-label model can only answer with
    the classes it was fitted to; CLIP answers with whatever captions it
    is given, so a new task is posed by writing its labels as sentences
    rather than by training a new head.  This is why the prompt matters
    at inference time and why the paper reports prompt ensembling as an
    accuracy lever rather than a curiosity.
    """,
)
@dataclass(frozen=True)
class CLIPConfig(MultimodalModelConfig):
    r"""Frozen configuration dataclass for every CLIP variant.

    Parameters
    ----------
    embed_dim : int, default=512
        Inherited from :class:`~lucid.models.multimodal.MultimodalModelConfig`
        — the width of the joint image-text space both towers project
        into, and the dimension a downstream consumer sees.
    image_size : int, default=224
        Input resolution of the image tower.  ``336`` for the
        high-resolution ViT-L/14 variant.
    patch_size : int, default=32
        Side of the square patch the image is tokenised into.  The token
        count is ``(image_size / patch_size) ** 2 + 1``, the ``+ 1``
        being the class token, so halving this quadruples the sequence.
    vision_layers : int, default=12
        Transformer blocks in the image tower.
    vision_width : int, default=768
        Residual width of the image tower.
    vision_heads : int, default=12
        Attention heads in the image tower.  The paper's variants all
        use ``vision_width / 64``.
    context_length : int, default=77
        Maximum text sequence length, counting the ``[SOS]`` and
        ``[EOS]`` sentinels.
    vocab_size : int, default=49408
        Byte-pair vocabulary size of the text tower.
    text_width : int, default=512
        Residual width of the text tower.
    text_heads : int, default=8
        Attention heads in the text tower.
    text_layers : int, default=12
        Transformer blocks in the text tower.  Fixed across every
        variant the paper trains — see Notes.
    logit_scale_init : float, default=0.07
        The temperature the learned scale starts at; stored internally
        as :math:`\log(1/\tau)`.
    logit_scale_max : float, default=100.0
        Ceiling on the exponentiated scale.

    Notes
    -----
    **Two numbers here disagree with the paper's prose**, and the
    implementation's are used because the released weights have those
    shapes:

    =================  ==================  ==================
    field              paper §A            official release
    =================  ==================  ==================
    ``context_length``  "capped at 76"      77
    ``vocab_size``      "49,152 vocab size" 49408
    =================  ==================  ==================

    The gap in each case is the sentinels: 76 caption tokens plus
    ``[SOS]`` and ``[EOS]`` does not fit in 77, so the released models
    treat 77 as the whole budget; and 49408 − 49152 = 256, one slot per
    byte, which is what a byte-level BPE needs to stay total.

    Depth scales only in the image tower.  Every variant the paper
    trains keeps ``text_layers = 12``, and for the ResNet towers it says
    so outright — "we only scale the width of the model to be
    proportional to the calculated increase in width of the ResNet and do
    not scale the depth at all".  The ViT variants follow the same rule:
    ViT-L/14 widens the text tower to 768 and leaves it 12 deep.

    Examples
    --------
    >>> from lucid.models.multimodal.clip import CLIPConfig
    >>> config = CLIPConfig()
    >>> config.embed_dim, config.patch_size
    (512, 32)
    >>> large = CLIPConfig(embed_dim=768, patch_size=14, vision_layers=24,
    ...                    vision_width=1024, vision_heads=16,
    ...                    text_width=768, text_heads=12)
    >>> large.vision_width // large.vision_heads
    64
    """

    model_type: ClassVar[str] = "clip"

    image_size: int = 224
    patch_size: int = 32
    vision_layers: int = 12
    vision_width: int = 768
    vision_heads: int = 12

    context_length: int = 77
    vocab_size: int = 49408
    text_width: int = 512
    text_heads: int = 8
    text_layers: int = 12

    logit_scale_init: float = 0.07
    logit_scale_max: float = 100.0

    @override
    def __post_init__(self) -> None:
        """Reject configurations that cannot be built or trained."""
        super().__post_init__()
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size must be divisible by patch_size, got "
                f"{self.image_size} and {self.patch_size} — a partial patch "
                f"at the border would be silently dropped"
            )
        for name, width, heads in (
            ("vision", self.vision_width, self.vision_heads),
            ("text", self.text_width, self.text_heads),
        ):
            if heads < 1:
                raise ValueError(f"{name}_heads must be positive, got {heads}")
            if width % heads != 0:
                raise ValueError(
                    f"{name}_width must be divisible by {name}_heads, got " f"{width} and {heads}"
                )
        if self.context_length < 2:
            raise ValueError(
                f"context_length must leave room for [SOS] and [EOS], got " f"{self.context_length}"
            )
        if not 0.0 < self.logit_scale_init:
            raise ValueError(
                f"logit_scale_init is a temperature and must be positive, "
                f"got {self.logit_scale_init}"
            )
        if self.logit_scale_max <= 1.0:
            raise ValueError(f"logit_scale_max must exceed 1, got {self.logit_scale_max}")
