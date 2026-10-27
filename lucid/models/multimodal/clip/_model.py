"""CLIP model (Radford et al., 2021) — two towers, one embedding space.

The architecture is two ordinary Transformers and a dot product.  What
makes it work is not the towers but three details that a shape check
cannot see, and each is a place this file is deliberate about:

**The text feature is read at ``[EOS]``, not at the end of the tensor.**
Captions are padded, so the last position holds padding for every
sequence but the longest.  The reference implementation locates the
sentinel with ``argmax`` over the token ids — which is correct only
because ``[EOS]`` is the highest id in the vocabulary — and this file
does the same for the same reason, with that reason written down.

**Both embeddings are L2-normalised before the dot product.**  Without
it the objective can be minimised by growing the norms rather than
aligning the directions, and the model trains happily to a
representation that ranks nothing.

**The temperature is learned in log space.**  A scale that is optimised
directly can cross zero and flip the sign of every logit; storing
:math:`\\log(1/\\tau)` makes the update multiplicative and the scale
positive by construction.
"""

import math
from dataclasses import dataclass
from typing import ClassVar, cast, final, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._tasks import ImageClassificationModel
from lucid.models._output import ModelOutput
from lucid.models.multimodal.clip._towers import (
    TransformerTower,
)
from lucid.models.multimodal.clip._config import CLIPConfig

__all__ = [
    "CLIPModel",
    "CLIPForZeroShotImageClassification",
    "CLIPOutput",
    "CLIPZeroShotOutput",
]


@final
class _VisionTransformer(nn.Module):
    """The image tower — patches, a class token, and a projection.

    Parameters
    ----------
    config : CLIPConfig
        Read for the image-side fields and ``embed_dim``.

    Notes
    -----
    The patch embedding is a stride-``patch_size`` convolution with no
    bias, which is the same linear map as flattening each patch and
    multiplying, expressed so the framework can use a convolution
    kernel for it.

    ``ln_pre`` before the stack and ``ln_post`` after it are both
    present, and both are needed: the first normalises the concatenation
    of three differently-scaled sources (class token, patch projections,
    positional table), the second conditions the class token before it
    is projected out.
    """

    def __init__(self, config: CLIPConfig) -> None:
        """Initialise the tower. See the class docstring for parameters."""
        super().__init__()
        width = config.vision_width
        self.patch_size = config.patch_size
        grid = config.image_size // config.patch_size
        self.num_patches = grid * grid

        self.conv1 = nn.Conv2d(
            3,
            width,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )
        # Created at zeros and filled by :meth:`CLIPModel._init_weights`.
        self.class_embedding = nn.Parameter(lucid.zeros((width,)))
        self.positional_embedding = nn.Parameter(lucid.zeros((self.num_patches + 1, width)))
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = TransformerTower(width, config.vision_layers, config.vision_heads)
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(lucid.zeros((width, config.embed_dim)))

    @override
    def forward(self, pixel_values: Tensor) -> Tensor:
        """Embed a batch of images.

        Parameters
        ----------
        pixel_values : Tensor
            ``(B, 3, image_size, image_size)``.

        Returns
        -------
        Tensor
            ``(B, embed_dim)`` — unnormalised.
        """
        batch = int(pixel_values.shape[0])
        x = self.conv1(pixel_values)
        x = x.reshape(batch, int(x.shape[1]), -1).swapaxes(1, 2)

        cls = self.class_embedding.reshape(1, 1, -1)
        cls = cls.repeat(batch, 1, 1)
        x = lucid.cat([cls, x], dim=1)
        x = x + self.positional_embedding.reshape(1, self.num_patches + 1, -1)

        x = self.ln_pre(x)
        x = self.transformer(x)
        # Only the class token leaves the tower; the patch tokens have
        # done their work by conditioning it.
        pooled = self.ln_post(x[:, 0])
        return pooled @ self.proj


@final
class _TextTransformer(nn.Module):
    """The text tower — token embeddings, causal attention, ``[EOS]``.

    Parameters
    ----------
    config : CLIPConfig
        Read for the text-side fields and ``embed_dim``.
    """

    def __init__(self, config: CLIPConfig) -> None:
        """Initialise the tower. See the class docstring for parameters."""
        super().__init__()
        width = config.text_width
        self.context_length = config.context_length

        self.token_embedding = nn.Embedding(config.vocab_size, width)
        self.positional_embedding = nn.Parameter(lucid.zeros((config.context_length, width)))
        self.transformer = TransformerTower(
            width, config.text_layers, config.text_heads, causal=True
        )
        self.ln_final = nn.LayerNorm(width)
        self.text_projection = nn.Parameter(lucid.zeros((width, config.embed_dim)))

    @override
    def forward(self, input_ids: Tensor) -> Tensor:
        """Embed a batch of tokenised captions.

        Parameters
        ----------
        input_ids : Tensor
            ``(B, context_length)`` of token ids, each sequence framed by
            ``[SOS]`` and ``[EOS]`` and right-padded with zeros.

        Returns
        -------
        Tensor
            ``(B, embed_dim)`` — unnormalised.

        Notes
        -----
        The feature is taken at the ``[EOS]`` position, located as the
        argmax of the ids.  That is not a heuristic: ``[EOS]`` is the
        highest id in a CLIPModel vocabulary and padding is ``0``, so the
        argmax is exactly the sentinel.  Reading the last column instead
        would return padding for every caption shorter than the context,
        which trains and ranks nothing.
        """
        length = int(input_ids.shape[1])
        if length != self.context_length:
            raise ValueError(
                f"input_ids must be padded to context_length="
                f"{self.context_length}, got {length} — the positional "
                f"table has no entries beyond it"
            )
        x = self.token_embedding(input_ids)
        x = x + self.positional_embedding.reshape(1, length, -1)
        x = self.transformer(x)
        x = self.ln_final(x)

        eos = input_ids.argmax(dim=-1)
        # ``.type`` — a bare ``device`` object renders as "device('metal')",
        # which the creation factories do not parse, and an index tensor
        # left on the CPU only fails once it meets the activations.
        rows = lucid.arange(int(x.shape[0]), dtype=lucid.int64, device=x.device.type)
        pooled = x[rows, eos]
        return pooled @ self.text_projection


@dataclass(slots=True)
class CLIPOutput(ModelOutput):
    """What :class:`CLIPModel` returns.

    Attributes
    ----------
    image_embeds, text_embeds : Tensor
        ``(B, embed_dim)``, L2-normalised — so a dot product between them
        is a cosine similarity and nothing further is needed to compare
        them.
    logits_per_image : Tensor
        ``(B_image, B_text)``, scaled by the learned temperature.
    logits_per_text : Tensor
        The transpose of ``logits_per_image``, carried explicitly because
        the symmetric loss reads both and transposing at the call site is
        where the two directions get accidentally tied.
    loss : Tensor or None
        The symmetric contrastive loss, present only when asked for.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.multimodal.clip._model import CLIPOutput
    >>> out = CLIPOutput(
    ...     image_embeds=lucid.zeros(2, 512),
    ...     text_embeds=lucid.zeros(3, 512),
    ...     logits_per_image=lucid.zeros(2, 3),
    ...     logits_per_text=lucid.zeros(3, 2),
    ... )
    >>> out.logits_per_image.shape, out.logits_per_text.shape
    ((2, 3), (3, 2))

    Both directions are carried because the contrastive loss reads the
    matrix by rows and by columns, and one is not the other's transpose
    once the temperature has been applied.
    """

    image_embeds: Tensor
    text_embeds: Tensor
    logits_per_image: Tensor
    logits_per_text: Tensor
    loss: Tensor | None = None


@dataclass(slots=True)
class CLIPZeroShotOutput(ModelOutput):
    """What :class:`CLIPForZeroShotImageClassification` returns.

    Attributes
    ----------
    logits : Tensor
        ``(B, num_prompts)`` — one score per candidate caption, in the
        order the prompts were given.
    image_embeds, text_embeds : Tensor
        The normalised embeddings the scores came from.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.multimodal.clip._model import CLIPZeroShotOutput
    >>> out = CLIPZeroShotOutput(
    ...     logits=lucid.zeros(2, 10),
    ...     image_embeds=lucid.zeros(2, 512),
    ...     text_embeds=lucid.zeros(10, 512),
    ... )
    >>> out.logits.shape
    (2, 10)

    Ten classes because ten prompts were embedded, not because anything
    was trained on ten: the class list is whatever text was handed in.
    """

    logits: Tensor
    image_embeds: Tensor
    text_embeds: Tensor


class CLIPModel(PretrainedModel):
    r"""Contrastive Language-Image Pre-training.

    Parameters
    ----------
    config : CLIPConfig
        The variant to build.

    Attributes
    ----------
    visual : _VisionTransformer
        Image tower.
    textual : _TextTransformer
        Text tower.
    logit_scale : nn.Parameter
        :math:`\log(1/\tau)`, a scalar.

    Notes
    -----
    Reference: Radford et al., *"Learning Transferable Visual Models From
    Natural Language Supervision"*, ICML, 2021
    (`arXiv:2103.00020 <https://arxiv.org/abs/2103.00020>`_).

    The two towers never meet except in the dot product, so they can be
    run independently — :meth:`encode_image` and :meth:`encode_text` are
    the useful entry points for retrieval, where one side is indexed once
    and the other is queried repeatedly.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.multimodal.clip import CLIPModel, CLIPConfig
    >>> model = CLIPModel(CLIPConfig(image_size=32, patch_size=16, vision_layers=1,
    ...                         vision_width=32, vision_heads=2,
    ...                         context_length=8, vocab_size=64, text_width=32,
    ...                         text_heads=2, text_layers=1, embed_dim=16)).eval()
    >>> images = lucid.randn((2, 3, 32, 32))
    >>> captions = lucid.zeros((2, 8), dtype=lucid.int64)
    >>> out = model(images, captions)
    >>> out.logits_per_image.shape
    (2, 2)
    """

    config_class: ClassVar[type[CLIPConfig]] = CLIPConfig
    base_model_prefix = "clip"

    def __init__(self, config: CLIPConfig) -> None:
        """Initialise the model. See the class docstring for parameters."""
        super().__init__(config)
        self.config: CLIPConfig = config
        self.visual = _VisionTransformer(config)
        self.textual = _TextTransformer(config)
        self.logit_scale = nn.Parameter(lucid.zeros((1,)))
        self._init_weights()

    def _init_weights(self) -> None:
        r"""Apply the released initialisation, which is not the default.

        Notes
        -----
        Two of these are load-bearing rather than cosmetic.

        The output projection of every residual branch is scaled by
        :math:`(2L)^{-1/2}`, so the variance a residual stream accumulates
        stays bounded as depth grows — without it a 24-layer tower starts
        with activations an order of magnitude larger than a 12-layer one
        and needs a different learning rate to survive the first steps.

        ``logit_scale`` starts at :math:`\log(1/\tau)` with
        :math:`\tau = 0.07`.  Left at the framework default of zero the
        scale would be :math:`e^0 = 1`, and cosine similarities in
        :math:`[-1, 1]` put through a softmax at unit scale are nearly
        uniform — the gradient would be almost flat for as long as it
        takes the scale itself to climb.
        """
        config = self.config
        with lucid.no_grad():
            nn.init.normal_(self.textual.token_embedding.weight, std=0.02)
            nn.init.normal_(self.textual.positional_embedding, std=0.01)
            nn.init.normal_(
                self.textual.text_projection,
                std=config.text_width**-0.5,
            )

            vision_std = config.vision_width**-0.5
            nn.init.normal_(self.visual.class_embedding, std=vision_std)
            nn.init.normal_(self.visual.positional_embedding, std=vision_std)
            nn.init.normal_(self.visual.proj, std=vision_std)

            for tower, width, layers in (
                (self.visual.transformer, config.vision_width, config.vision_layers),
                (self.textual.transformer, config.text_width, config.text_layers),
            ):
                attn_std = width**-0.5
                proj_std = (width**-0.5) * ((2 * layers) ** -0.5)
                fc_std = (2 * width) ** -0.5
                for module in tower.resblocks:
                    block = module
                    # Fused only when the three projections share a
                    # width, which self-attention always does; asserting
                    # rather than casting so a future kdim/vdim split
                    # fails here instead of silently skipping the init.
                    fused = block.attn.in_proj_weight
                    assert fused is not None
                    nn.init.normal_(fused, std=attn_std)
                    nn.init.normal_(block.attn.out_proj_weight, std=proj_std)
                    nn.init.normal_(cast(nn.Linear, block.mlp[0]).weight, std=fc_std)
                    nn.init.normal_(cast(nn.Linear, block.mlp[2]).weight, std=proj_std)

            self.logit_scale[:] = lucid.full((1,), math.log(1.0 / config.logit_scale_init))

    @property
    def scale(self) -> Tensor:
        r"""The temperature as a multiplier — ``exp(logit_scale)``, capped.

        Returns
        -------
        Tensor
            A scalar in ``(0, logit_scale_max]``.

        Notes
        -----
        The cap is the paper's: the scale is "clipped to prevent scaling
        the logits by more than 100".  Applying it here rather than in a
        training loop means every consumer gets it, including one that
        writes its own loop — which is where the published implementation
        puts it and therefore where it is easy to omit.
        """
        return lucid.clip(lucid.exp(self.logit_scale), None, self.config.logit_scale_max)

    def encode_image(self, pixel_values: Tensor) -> Tensor:
        """Embed images into the joint space, L2-normalised.

        Parameters
        ----------
        pixel_values : Tensor
            ``(B, 3, image_size, image_size)``.

        Returns
        -------
        Tensor
            ``(B, embed_dim)`` with unit rows.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models.multimodal.clip import CLIPConfig, CLIPModel
        >>> config = CLIPConfig(image_size=32, patch_size=16, vision_layers=1,
        ...                     vision_width=32, vision_heads=2, context_length=8,
        ...                     vocab_size=64, text_width=32, text_heads=2,
        ...                     text_layers=1, embed_dim=16)
        >>> model = CLIPModel(config).eval()
        >>> embeds = model.encode_image(lucid.randn((3, 3, 32, 32)))
        >>> embeds.shape
        (3, 16)

        The rows are unit length, so the Gram matrix is already cosine
        similarity — each image scores exactly 1 against itself.

        >>> gram = embeds @ embeds.T
        >>> [round(gram[i, i].item(), 4) for i in range(3)]
        [1.0, 1.0, 1.0]
        """
        return _l2_normalise(self.visual(pixel_values))

    def encode_text(self, input_ids: Tensor) -> Tensor:
        """Embed captions into the joint space, L2-normalised.

        Parameters
        ----------
        input_ids : Tensor
            ``(B, context_length)``.

        Returns
        -------
        Tensor
            ``(B, embed_dim)`` with unit rows.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models.multimodal.clip import CLIPConfig, CLIPModel
        >>> config = CLIPConfig(image_size=32, patch_size=16, vision_layers=1,
        ...                     vision_width=32, vision_heads=2, context_length=8,
        ...                     vocab_size=64, text_width=32, text_heads=2,
        ...                     text_layers=1, embed_dim=16)
        >>> model = CLIPModel(config).eval()

        The feature is read at ``[EOS]`` — the highest id, 63 in this
        64-token vocabulary — through a causal tower, so nothing after the
        sentinel can move it.  These two captions differ only there.

        >>> ids = lucid.tensor([[62, 5, 7, 63, 0, 0, 0, 0],
        ...                     [62, 5, 7, 63, 9, 9, 9, 9]], dtype=lucid.int64)
        >>> embeds = model.encode_text(ids)
        >>> embeds.shape
        (2, 16)
        >>> bool(lucid.allclose(embeds[0], embeds[1], atol=1e-6))
        True

        Scoring against images is a dot product and the temperature, which
        is all :meth:`forward` does with the two embeddings.

        >>> images = lucid.randn((2, 3, 32, 32))
        >>> sims = model.encode_image(images) @ embeds.T
        >>> out = model(images, ids)
        >>> bool(lucid.allclose(out.logits_per_image, model.scale * sims, atol=1e-5))
        True
        """
        return _l2_normalise(self.textual(input_ids))

    @override
    def forward(
        self,
        pixel_values: Tensor,
        input_ids: Tensor,
        return_loss: bool = False,
    ) -> CLIPOutput:
        """Score every image against every caption in the batch.

        Parameters
        ----------
        pixel_values : Tensor
            ``(B, 3, image_size, image_size)``.
        input_ids : Tensor
            ``(B, context_length)``.
        return_loss : bool, default=False
            Also compute the symmetric contrastive loss, which assumes
            the two batches are aligned pairwise.

        Returns
        -------
        CLIPOutput
            Embeddings, both logit matrices, and optionally the loss.
        """
        image_embeds = self.encode_image(pixel_values)
        text_embeds = self.encode_text(input_ids)

        logits_per_image = self.scale * (image_embeds @ text_embeds.T)
        logits_per_text = logits_per_image.T

        loss: Tensor | None = None
        if return_loss:
            loss = _contrastive_loss(logits_per_image, logits_per_text)
        return CLIPOutput(
            image_embeds=image_embeds,
            text_embeds=text_embeds,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            loss=loss,
        )


class CLIPForZeroShotImageClassification(ImageClassificationModel):
    """CLIP posed as a classifier, by writing the labels as sentences.

    Parameters
    ----------
    config : CLIPConfig
        The variant to build.

    Notes
    -----
    There is no head and nothing new is trained.  The classes arrive as
    tokenised prompts and the scores are the same cosine similarities the
    contrastive objective was fitted on, which is the whole claim of the
    paper — a fixed-label model cannot be asked a question it was not
    trained for, and this can.

    A caller that classifies many images against one label set should
    embed the prompts once with :meth:`~CLIPModel.encode_text` and reuse them;
    :meth:`forward` re-embeds them on every call because it does not know
    that the set is fixed.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.multimodal.clip import (
    ...     CLIPConfig, CLIPForZeroShotImageClassification)
    >>> config = CLIPConfig(image_size=32, patch_size=16, vision_layers=1,
    ...                     vision_width=32, vision_heads=2, context_length=8,
    ...                     vocab_size=64, text_width=32, text_heads=2,
    ...                     text_layers=1, embed_dim=16)
    >>> model = CLIPForZeroShotImageClassification(config).eval()
    >>> out = model(lucid.randn((2, 3, 32, 32)), lucid.zeros((5, 8), dtype=lucid.int64))
    >>> out.logits.shape
    (2, 5)
    """

    config_class: ClassVar[type[CLIPConfig]] = CLIPConfig
    base_model_prefix = "clip"

    def __init__(self, config: CLIPConfig) -> None:
        """Initialise the model. See the class docstring for parameters."""
        super().__init__(config)
        self.clip = CLIPModel(config)

    @override
    def forward(self, pixel_values: Tensor, prompt_ids: Tensor) -> CLIPZeroShotOutput:
        """Score each image against each candidate prompt.

        Parameters
        ----------
        pixel_values : Tensor
            ``(B, 3, image_size, image_size)``.
        prompt_ids : Tensor
            ``(num_prompts, context_length)`` — one tokenised caption per
            candidate class.  Unlike :meth:`CLIPModel.forward` this batch is
            *not* paired with the images and may be any size.

        Returns
        -------
        CLIPZeroShotOutput
            ``logits`` is ``(B, num_prompts)``.
        """
        image_embeds = self.clip.encode_image(pixel_values)
        text_embeds = self.clip.encode_text(prompt_ids)
        logits = self.clip.scale * (image_embeds @ text_embeds.T)
        return CLIPZeroShotOutput(logits=logits, image_embeds=image_embeds, text_embeds=text_embeds)


def _l2_normalise(x: Tensor) -> Tensor:
    """Scale rows to unit length.

    Parameters
    ----------
    x : Tensor
        ``(B, D)``.

    Returns
    -------
    Tensor
        ``(B, D)``.
    """
    return x / ((x**2).sum(dim=-1, keepdim=True) ** 0.5)


def _contrastive_loss(logits_per_image: Tensor, logits_per_text: Tensor) -> Tensor:
    r"""The symmetric cross-entropy CLIP is trained with.

    Parameters
    ----------
    logits_per_image : Tensor
        ``(N, N)``, scaled cosine similarities.
    logits_per_text : Tensor
        ``(N, N)``, its transpose.

    Returns
    -------
    Tensor
        A scalar.

    Notes
    -----
    The targets are the diagonal — pair :math:`i` is the only correct
    answer for image :math:`i` and for caption :math:`i` alike.  Both
    directions are scored and averaged because a one-sided loss leaves
    one tower free to collapse: if only images are classified over
    captions, nothing scores whether the captions are separable from each
    other.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.multimodal.clip._model import _contrastive_loss
    >>> perfect = lucid.tensor([[10.0, -10.0], [-10.0, 10.0]])
    >>> float(_contrastive_loss(perfect, perfect.T).item()) < 1e-4
    True
    """
    n = int(logits_per_image.shape[0])
    if int(logits_per_image.shape[1]) != n:
        raise ValueError(
            f"the contrastive loss pairs a batch with itself, so the logits "
            f"must be square; got {tuple(logits_per_image.shape)}"
        )
    target = lucid.arange(n, dtype=lucid.int64, device=logits_per_image.device.type)
    image_side = F.cross_entropy(logits_per_image, target)
    text_side = F.cross_entropy(logits_per_text, target)
    return (image_side + text_side) / 2.0
