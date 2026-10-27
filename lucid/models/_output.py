"""Model output dataclasses — iterable, dict-like, tuple-compatible.

All ``forward()`` methods on :class:`PretrainedModel` subclasses return one
of these.  Each output behaves both as a dataclass with named fields and
as a tuple/sequence (skipping ``None`` fields), so callers can pick either
style without a wrapper.
"""

from collections.abc import Iterator
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor
    from lucid.utils.cache import Cache, EncoderDecoderCache


class ModelOutput:
    r"""Base for all model-forward output dataclasses.

    Provides a uniform protocol: outputs are simultaneously dataclass-like
    (named fields, dict-style lookup) and sequence-like (iteration / tuple
    indexing, with ``None`` fields silently elided).  Each concrete
    subclass adds a task-specific set of ``Tensor`` (or
    ``tuple[Tensor, ...]``) fields.

    Notes
    -----
    Subclasses must be decorated with ``@dataclass`` so
    :func:`dataclasses.fields` works.  Iteration order follows field
    declaration order, and ``None`` fields are skipped — this matches the
    behaviour expected by callers that destructure outputs with
    ``logits, loss = model(x)``.

    Direct instantiation of :class:`ModelOutput` itself is not useful (no
    fields) — always subclass and add ``@dataclass``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models import ImageClassificationOutput
    >>> out = ImageClassificationOutput(logits=lucid.randn(1, 10))
    >>> out["logits"].shape           # dict-style access
    (1, 10)
    >>> first, = out                  # tuple-style unpacking
    >>> first is out.logits
    True
    """

    def __iter__(self) -> Iterator[Tensor]:
        r"""Yield each non-``None`` field value in declaration order."""
        for f in fields(self):  # type: ignore[arg-type]
            value: Tensor | None = getattr(self, f.name)
            if value is not None:
                yield value

    def __getitem__(self, idx: int | str) -> Tensor:
        r"""Look up a field by name (``str``) or position (``int``).

        Parameters
        ----------
        idx : int or str
            Field name or positional index into the non-``None``
            iteration.

        Returns
        -------
        Tensor
            The requested field value.

        Raises
        ------
        KeyError
            If ``idx`` is a string naming a missing / ``None`` field.
        IndexError
            If ``idx`` is an integer outside the non-``None`` range.
        """
        if isinstance(idx, str):
            value2: Tensor | None = getattr(self, idx, None)
            if value2 is None:
                raise KeyError(idx)
            return value2
        return tuple(self)[idx]

    def __len__(self) -> int:
        r"""Return the number of non-``None`` fields."""
        count = 0
        for f in fields(self):  # type: ignore[arg-type]
            if getattr(self, f.name) is not None:
                count += 1
        return count

    def __contains__(self, key: object) -> bool:
        r"""Return whether ``key`` names a present (non-``None``) field."""
        if not isinstance(key, str):
            return False
        return getattr(self, key, None) is not None

    def to_tuple(self) -> tuple[Tensor, ...]:
        r"""Return a tuple of non-``None`` field values in declaration order.

        Examples
        --------
        >>> out = BaseModelOutput(last_hidden_state=h)
        >>> out.to_tuple()
        (h,)
        """
        return tuple(self)

    def keys(self) -> list[str]:
        r"""Names of fields whose value is not ``None``, in declaration order."""
        return [
            f.name
            for f in fields(self)  # type: ignore[arg-type]
            if getattr(self, f.name) is not None
        ]

    def values(self) -> list[Tensor]:
        r"""Values of non-``None`` fields, in declaration order."""
        return list(self)

    def items(self) -> list[tuple[str, Tensor]]:
        r"""``(name, value)`` pairs of non-``None`` fields."""
        return [(k, getattr(self, k)) for k in self.keys()]


@dataclass(slots=True)
class BaseModelOutput(ModelOutput):
    r"""Generic backbone forward output — sequence model variant.

    Returned by backbone classes that emit a single hidden-state stream
    (encoder-only and decoder-only language models, ViT-family backbones).

    Attributes
    ----------
    last_hidden_state : Tensor
        Final layer's output, shaped ``(B, T, H)`` for sequence models or
        ``(B, C, H, W)`` for spatial backbones.
    hidden_states : tuple[Tensor, ...] or None, optional
        Per-layer hidden states when the caller passed
        ``output_hidden_states=True`` (else ``None``).
    attentions : tuple[Tensor, ...] or None, optional
        Per-layer attention weights when the caller passed
        ``output_attentions=True`` (else ``None``).

    Notes
    -----
    Produced by transformer-family backbones (BERT, RoFormer, ViT, …) and
    most other encoder-only architectures that don't need a pooled
    representation.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models import BaseModelOutput
    >>> out = BaseModelOutput(last_hidden_state=lucid.zeros(1, 128, 768))
    >>> out.last_hidden_state.shape
    (1, 128, 768)

    The two optional fields are populated only when a model is asked for
    them, so an ordinary forward leaves them empty.

    >>> out.hidden_states is None, out.attentions is None
    (True, True)
    """

    last_hidden_state: Tensor
    hidden_states: tuple[Tensor, ...] | None = None
    attentions: tuple[Tensor, ...] | None = None


@dataclass(slots=True)
class BaseModelOutputWithPooling(ModelOutput):
    r"""Backbone output + pooled representation (e.g. CLS).

    Attributes
    ----------
    last_hidden_state : Tensor
        Per-token / per-spatial-location output, shape ``(B, T, H)``.
    pooler_output : Tensor
        Aggregated representation — typically the ``[CLS]`` position
        passed through ``tanh(Linear(H, H))`` (BERT-style) or a learned
        attention pool.  Shape ``(B, H)``.
    hidden_states : tuple[Tensor, ...] or None, optional
        Per-layer hidden states; populated when
        ``output_hidden_states=True``.
    attentions : tuple[Tensor, ...] or None, optional
        Per-layer attention weights; populated when
        ``output_attentions=True``.

    Notes
    -----
    Used by encoder LMs that ship a sentence-level pooling head out of
    the box — BERT and friends.  Downstream classifiers attach to
    ``pooler_output`` rather than re-pooling the per-token sequence.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models import BaseModelOutputWithPooling
    >>> out = BaseModelOutputWithPooling(
    ...     last_hidden_state=lucid.zeros(1, 128, 768),
    ...     pooler_output=lucid.zeros(1, 768),
    ... )
    >>> out.last_hidden_state.shape, out.pooler_output.shape
    ((1, 128, 768), (1, 768))
    """

    last_hidden_state: Tensor
    pooler_output: Tensor
    hidden_states: tuple[Tensor, ...] | None = None
    attentions: tuple[Tensor, ...] | None = None


@dataclass(slots=True)
class ImageClassificationOutput(ModelOutput):
    r"""Output of any ``{Family}ForImageClassification`` model.

    Attributes
    ----------
    logits : Tensor
        Pre-softmax class scores, shape ``(B, num_classes)``.
    loss : Tensor or None, optional
        Scalar cross-entropy loss when ``labels`` were supplied to
        ``forward``; otherwise ``None``.
    hidden_states : tuple[Tensor, ...] or None, optional
        Per-layer feature maps when requested via
        ``output_hidden_states=True``.
    attentions : tuple[Tensor, ...] or None, optional
        Per-layer attention weights for transformer-family classifiers.

    Notes
    -----
    Returned by every classifier registered under
    :class:`AutoModelForImageClassification` — CNN-family
    (ResNet, EfficientNet, …) and ViT-family backbones alike.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models import create_model
    >>> model = create_model("resnet_18_cls", num_classes=10).eval()
    >>> out = model(lucid.randn(1, 3, 224, 224))
    >>> type(out).__name__
    'ImageClassificationOutput'
    >>> out.logits.shape
    (1, 10)
    >>> out.loss is None              # no labels were passed
    True
    """

    logits: Tensor
    loss: Tensor | None = None
    hidden_states: tuple[Tensor, ...] | None = None
    attentions: tuple[Tensor, ...] | None = None


@dataclass(slots=True)
class ObjectDetectionOutput(ModelOutput):
    r"""Output of any ``{Family}ForObjectDetection`` model.

    Attributes
    ----------
    logits : Tensor
        Per-proposal / per-query class logits.  Shape depends on family:
        ``(B, num_queries, num_classes)`` for DETR, ``(N, num_classes)``
        for R-CNN-family per-RoI scoring.
    pred_boxes : Tensor
        Predicted bounding box coordinates.  Box format is family-specific
        (cxcywh-normalised for DETR, xyxy-pixel for R-CNN).
    loss : Tensor or None, optional
        Total detection loss (classification + box regression + matching
        cost where applicable) when targets were supplied.
    hidden_states : tuple[Tensor, ...] or None, optional
        Optional intermediate feature maps.
    proposals : tuple[Tensor, ...] or None, optional
        Per-image RoI proposals for two-stage detectors (R-CNN / Fast
        R-CNN / Faster R-CNN).  Lets downstream ``postprocess()`` run
        from the output alone, without re-running the proposal stage.
    objectness : Tensor or None, optional
        ``(B, num_anchors)`` per-anchor objectness probability for the
        single-stage YOLO family.  A YOLO detection score is
        :math:`\Pr(\text{object}) \cdot \Pr(\text{class} \mid \text{object})`,
        so ``postprocess`` needs this alongside ``logits``; without it the
        class score alone would rank and threshold every background anchor.

    Notes
    -----
    Returned by :class:`AutoModelForObjectDetection` registrations —
    R-CNN family, DETR, EfficientDet, YOLO v1–v4.  Anchor-based and
    set-prediction families share this contract.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models import ObjectDetectionOutput
    >>> out = ObjectDetectionOutput(
    ...     logits=lucid.zeros(1, 100, 92),
    ...     pred_boxes=lucid.zeros(1, 100, 4),
    ... )
    >>> out.logits.shape, out.pred_boxes.shape
    ((1, 100, 92), (1, 100, 4))

    ``proposals`` and ``objectness`` carry the two-stage detectors'
    intermediate results and stay empty for a single-stage one.

    >>> out.proposals is None, out.objectness is None
    (True, True)
    """

    logits: Tensor
    pred_boxes: Tensor
    loss: Tensor | None = None
    hidden_states: tuple[Tensor, ...] | None = None
    # Two-stage detectors (R-CNN / Fast R-CNN / Faster R-CNN) emit detections
    # *per RoI*; ``proposals`` carries the per-image RoI list so downstream
    # ``postprocess()`` can be called against the output alone.
    proposals: tuple[Tensor, ...] | None = None
    objectness: Tensor | None = None


@dataclass(slots=True)
class InstanceSegmentationOutput(ModelOutput):
    r"""Output of any instance-segmentation model.

    Attributes
    ----------
    logits : Tensor
        Per-proposal class logits.
    pred_boxes : Tensor
        Predicted bounding boxes.
    pred_masks : Tensor
        Per-instance binary mask logits, typically shape
        ``(N, num_classes, mh, mw)`` for R-CNN-style or
        ``(B, num_queries, H, W)`` for transformer-based instance heads.
    loss : Tensor or None, optional
        Total loss (cls + box + mask) when targets were supplied.
    hidden_states : tuple[Tensor, ...] or None, optional
        Optional intermediate feature maps.
    proposals : tuple[Tensor, ...] or None, optional
        Per-image RoI proposals used by ``forward``.  Carried out for the
        same reason Faster R-CNN carries them: ``postprocess`` needs the
        proposal boxes to map rows of ``pred_boxes`` / ``pred_masks`` back
        to images, and when ``forward`` generated the proposals itself the
        caller has no other way to supply them.
    input_size : tuple[int, int] or None, optional
        ``(H, W)`` of the image batch ``forward`` ran on.  Mask R-CNN
        derives its pyramid strides from it, and ``postprocess`` must use
        the same ones to re-align masks on the final boxes; the feature
        maps alone do not pin it down (every height from 97 to 100 gives
        a 25-row P2, and the strides differ across them).

    Notes
    -----
    Returned today by Mask R-CNN; Mask2Former when configured for
    instance segmentation will also produce this shape.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models import InstanceSegmentationOutput
    >>> out = InstanceSegmentationOutput(
    ...     logits=lucid.zeros(1, 100, 81),
    ...     pred_boxes=lucid.zeros(1, 100, 4),
    ...     pred_masks=lucid.zeros(1, 100, 28, 28),
    ... )
    >>> out.pred_masks.shape          # one mask per detection
    (1, 100, 28, 28)
    """

    logits: Tensor
    pred_boxes: Tensor
    pred_masks: Tensor
    loss: Tensor | None = None
    hidden_states: tuple[Tensor, ...] | None = None
    proposals: tuple[Tensor, ...] | None = None
    input_size: tuple[int, int] | None = None


@dataclass(slots=True)
class SemanticSegmentationOutput(ModelOutput):
    r"""Output of any ``{Family}ForSemanticSegmentation`` model.

    Attributes
    ----------
    logits : Tensor
        Per-pixel class logits, shape ``(B, num_classes, H, W)``.  Spatial
        resolution may match the input or be downsampled — call
        ``F.interpolate`` to upsample when needed.
    loss : Tensor or None, optional
        Scalar segmentation loss (cross-entropy / Dice / per-class) when
        targets were supplied.
    hidden_states : tuple[Tensor, ...] or None, optional
        Optional encoder / decoder feature maps.

    Notes
    -----
    Returned by FCN, U-Net, Attention U-Net, MaskFormer, and Mask2Former
    when used for semantic segmentation.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models import create_model
    >>> model = create_model("fcn_resnet50", num_classes=21).eval()
    >>> out = model(lucid.randn(1, 3, 224, 224))
    >>> out.logits.shape              # (B, K, H, W) at input resolution
    (1, 21, 224, 224)
    """

    logits: Tensor
    loss: Tensor | None = None
    hidden_states: tuple[Tensor, ...] | None = None


@dataclass(slots=True)
class CausalLMOutput(ModelOutput):
    r"""Output of any causal (left-to-right) language model head.

    Attributes
    ----------
    logits : Tensor
        Per-position vocabulary logits, shape ``(B, T, vocab_size)``.
    loss : Tensor or None, optional
        Scalar cross-entropy loss when ``labels`` were supplied.  Labels
        are typically shifted internally so ``loss`` reflects next-token
        prediction.
    past_key_values : tuple[tuple[Tensor, Tensor], ...] or None, optional
        Per-layer ``(key, value)`` cache for incremental decoding.  Each
        tensor shape ``(B, num_heads, T, head_dim)``.
    hidden_states : tuple[Tensor, ...] or None, optional
        Per-layer hidden states.
    attentions : tuple[Tensor, ...] or None, optional
        Per-layer attention weights.

    Notes
    -----
    Produced by GPT-1, GPT-2 (``GPT2LMHeadModel``), and any future
    decoder-only LM head.  :class:`CausalLMMixin` consumes the
    ``logits`` field; future cache-aware sampling will plumb through
    ``past_key_values``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models import CausalLMOutput
    >>> out = CausalLMOutput(logits=lucid.zeros(1, 128, 50257))
    >>> out.logits.shape              # (B, T, vocab)
    (1, 128, 50257)

    ``past_key_values`` holds the cache that makes the next step cheap,
    and is filled only when generation asks for it.

    >>> out.past_key_values is None
    True
    """

    logits: Tensor
    loss: Tensor | None = None
    past_key_values: Cache | tuple[tuple[Tensor, Tensor], ...] | None = None
    hidden_states: tuple[Tensor, ...] | None = None
    attentions: tuple[Tensor, ...] | None = None


@dataclass(slots=True)
class MaskedLMOutput(ModelOutput):
    r"""Output of any masked-LM head (BERT, RoFormer, …).

    Attributes
    ----------
    logits : Tensor
        Per-position vocabulary logits, shape ``(B, T, vocab_size)``.
    loss : Tensor or None, optional
        Scalar cross-entropy loss reduced over masked positions only
        (``ignore_index=-100`` for non-masked tokens).
    hidden_states : tuple[Tensor, ...] or None, optional
        Per-layer hidden states.
    attentions : tuple[Tensor, ...] or None, optional
        Per-layer attention weights.

    Notes
    -----
    Returned by every ``{Family}ForMaskedLM`` registration —
    ``BERTForMaskedLM``, ``RoFormerForMaskedLM``, and future encoder LM
    heads.  Loss is computed via :meth:`MaskedLMMixin.compute_lm_loss`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models import MaskedLMOutput
    >>> out = MaskedLMOutput(logits=lucid.zeros(1, 128, 30522))
    >>> out.logits.shape              # (B, T, vocab)
    (1, 128, 30522)
    """

    logits: Tensor
    loss: Tensor | None = None
    hidden_states: tuple[Tensor, ...] | None = None
    attentions: tuple[Tensor, ...] | None = None


@dataclass(slots=True)
class Seq2SeqLMOutput(ModelOutput):
    r"""Output of any encoder-decoder seq2seq language model.

    Attributes
    ----------
    logits : Tensor
        Decoder vocabulary logits, shape ``(B, T_dec, vocab_size)``.
    loss : Tensor or None, optional
        Scalar cross-entropy loss when ``labels`` were supplied (label
        smoothing / shift handled internally).
    past_key_values : tuple[tuple[Tensor, Tensor], ...] or None, optional
        Per-layer decoder ``(key, value)`` cache for autoregressive
        decoding.
    decoder_hidden_states : tuple[Tensor, ...] or None, optional
        Per-layer decoder hidden states.
    decoder_attentions : tuple[Tensor, ...] or None, optional
        Per-layer decoder self-attention weights.
    encoder_last_hidden_state : Tensor or None, optional
        Final encoder output, cached so callers can run multiple decoder
        passes (e.g. beam search) without re-encoding.
    encoder_hidden_states : tuple[Tensor, ...] or None, optional
        Per-layer encoder hidden states.
    encoder_attentions : tuple[Tensor, ...] or None, optional
        Per-layer encoder self-attention weights.

    Notes
    -----
    Produced by ``TransformerForSeq2SeqLM`` today; T5, BART, and mBART
    are the natural future consumers.  Caching the encoder output is the
    standard pattern for batched generation.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models import Seq2SeqLMOutput
    >>> out = Seq2SeqLMOutput(
    ...     logits=lucid.zeros(1, 32, 32128),
    ...     encoder_last_hidden_state=lucid.zeros(1, 128, 768),
    ... )
    >>> out.logits.shape, out.encoder_last_hidden_state.shape
    ((1, 32, 32128), (1, 128, 768))

    The decoder and encoder keep separate hidden-state and attention
    fields, which is what distinguishes this from CausalLMOutput.

    >>> out.decoder_hidden_states is None, out.encoder_hidden_states is None
    (True, True)
    """

    logits: Tensor
    loss: Tensor | None = None
    past_key_values: EncoderDecoderCache | tuple[tuple[Tensor, Tensor], ...] | None = None
    decoder_hidden_states: tuple[Tensor, ...] | None = None
    decoder_attentions: tuple[Tensor, ...] | None = None
    encoder_last_hidden_state: Tensor | None = None
    encoder_hidden_states: tuple[Tensor, ...] | None = None
    encoder_attentions: tuple[Tensor, ...] | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Generative-family outputs
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class DiffusionModelOutput(ModelOutput):
    r"""Single-step output of a diffusion U-Net's forward pass.

    Attributes
    ----------
    sample : Tensor
        Network prediction at the supplied timestep.  The semantic
        interpretation depends on the parameterisation declared in
        :attr:`DiffusionModelConfig.prediction_type`:

        * ``"epsilon"`` — predicted noise :math:`\varepsilon`.
        * ``"sample"`` — predicted clean signal :math:`x_0`.
        * ``"v"`` — predicted velocity :math:`v = \alpha_t \varepsilon -
          \sigma_t x_0`.
    loss : Tensor or None, optional
        Scalar training loss against ground-truth noise (MSE for the
        ``"epsilon"`` parameterisation).

    Notes
    -----
    Returned by ``_DDPMUNet.forward`` and other diffusion noise predictors.
    The sampling loop in :class:`DiffusionMixin` reads ``sample`` and
    passes it to the scheduler's ``step`` method.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models import DiffusionModelOutput
    >>> out = DiffusionModelOutput(sample=lucid.zeros(1, 3, 32, 32))
    >>> out.sample.shape
    (1, 3, 32, 32)
    """

    sample: Tensor
    loss: Tensor | None = None


@dataclass(slots=True)
class VAEOutput(ModelOutput):
    r"""End-to-end VAE forward output.

    Attributes
    ----------
    sample : Tensor
        Reconstructed image :math:`\hat{x}` shaped ``(B, C, H, W)``.
    latent : Tensor
        Sampled latent :math:`z = \mu + \sigma \cdot \varepsilon`, shape
        ``(B, latent_dim)``.
    mu : Tensor
        Encoder mean :math:`\mu(x)`.
    logvar : Tensor
        Encoder log-variance :math:`\log \sigma^2(x)`.
    loss : Tensor or None, optional
        Total ELBO loss (``recon_loss + β · kl_loss``) when targets were
        supplied to ``forward``.
    recon_loss : Tensor or None, optional
        Reconstruction term alone (typically BCE or MSE).
    kl_loss : Tensor or None, optional
        KL divergence term :math:`D_{\mathrm{KL}}(q(z|x) \| p(z))` alone.

    Notes
    -----
    Returned by :class:`VAEModel.forward`.  :class:`VAEForImageGeneration`
    wraps the decoder for unconditional sampling and returns
    :class:`GenerationOutput` from its ``generate`` method.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models import VAEOutput
    >>> out = VAEOutput(
    ...     sample=lucid.zeros(1, 3, 32, 32),
    ...     latent=lucid.zeros(1, 128),
    ...     mu=lucid.zeros(1, 128),
    ...     logvar=lucid.zeros(1, 128),
    ... )
    >>> out.sample.shape, out.latent.shape
    ((1, 3, 32, 32), (1, 128))

    The reconstruction and KL terms are kept apart from the total so a
    caller can watch them move against each other during training.

    >>> out.recon_loss is None, out.kl_loss is None
    (True, True)
    """

    sample: Tensor
    latent: Tensor
    mu: Tensor
    logvar: Tensor
    loss: Tensor | None = None
    recon_loss: Tensor | None = None
    kl_loss: Tensor | None = None


@dataclass(slots=True)
class NormalizingFlowOutput(ModelOutput):
    r"""Forward output of an exact-likelihood normalizing flow.

    A flow's forward direction is the *inference* direction — it maps data
    to the latent space and reports how much volume the map contracted, so
    the exact log-density comes out of a single pass.

    Attributes
    ----------
    latent : Tensor
        Latent code :math:`h = f(x)`, shape ``(B, D)`` with
        ``D = C · H · W`` for image flows (the bijection is defined on the
        flattened sample).
    log_det_jacobian : Tensor
        Per-sample :math:`\log\big|\det \partial f / \partial x\big|`,
        shape ``(B,)``.  Volume-preserving stages (additive coupling,
        permutations) contribute exactly zero.
    log_prob : Tensor
        Per-sample exact log-likelihood, shape ``(B,)``:

        .. math::

            \log p_X(x) = \log p_H(f(x))
                + \log\big|\det \partial f / \partial x\big|.

    loss : Tensor or None, optional
        Scalar negative log-likelihood ``-log_prob.mean()``, populated by
        the task wrapper (the bare trunk leaves it ``None``).

    Notes
    -----
    Returned by ``NICEModel.forward`` and every other flow trunk /
    ``…ForImageGeneration`` head.  Sampling runs the bijection backwards
    and returns :class:`GenerationOutput` instead.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models import NormalizingFlowOutput
    >>> out = NormalizingFlowOutput(
    ...     latent=lucid.zeros(4, 128),
    ...     log_det_jacobian=lucid.zeros(4),
    ...     log_prob=lucid.zeros(4),
    ... )
    >>> out.latent.shape, out.log_prob.shape
    ((4, 128), (4,))
    """

    latent: Tensor
    log_det_jacobian: Tensor
    log_prob: Tensor
    loss: Tensor | None = None


@dataclass(slots=True)
class GenerationOutput(ModelOutput):
    r"""Final result of a generative model's sampling loop.

    Attributes
    ----------
    samples : Tensor
        Final ``(n_samples, C, H, W)`` (image), ``(n_samples, D)``
        (vector flow) or ``(n_samples, T)`` (text) batch produced by the
        sampler.
    intermediates : tuple[Tensor, ...] or None, optional
        Per-step latents / samples — populated only when the caller
        passes ``return_intermediates=True`` to ``generate``.  Useful for
        animating trajectories or debugging schedulers.

    Notes
    -----
    Returned by :meth:`DiffusionMixin.generate` and
    :class:`VAEForImageGeneration.generate`.  Text generation through
    :meth:`CausalLMMixin.generate` currently returns a bare
    :class:`Tensor` rather than this wrapper — that may change.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models import GenerationOutput
    >>> out = GenerationOutput(samples=lucid.zeros(4, 3, 32, 32))
    >>> out.samples.shape
    (4, 3, 32, 32)

    ``intermediates`` is filled only when the caller asks to keep the
    trajectory, which costs one tensor per step.

    >>> out.intermediates is None
    True
    """

    samples: Tensor
    intermediates: tuple[Tensor, ...] | None = None


@dataclass(slots=True)
class SequenceClassificationOutput(ModelOutput):
    r"""Output of any whole-sequence classification head.

    Attributes
    ----------
    logits : Tensor
        Per-sequence class logits, shape ``(B, num_labels)``.  A
        ``num_labels`` of 1 marks a regression head, whose loss is MSE
        rather than cross-entropy.
    loss : Tensor or None, optional
        Scalar loss when ``labels`` were supplied.
    hidden_states : tuple[Tensor, ...] or None, optional
        Per-layer hidden states.
    attentions : tuple[Tensor, ...] or None, optional
        Per-layer attention weights.

    Notes
    -----
    Returned by every ``{Family}ForSequenceClassification`` head.  It
    exists so a caller can tell a ``(B, num_labels)`` sentence logit
    apart from a ``(B, T, vocab_size)`` masked-LM logit by type rather
    than by inspecting shapes.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models import SequenceClassificationOutput
    >>> out = SequenceClassificationOutput(logits=lucid.zeros(1, 2))
    >>> out.logits.shape              # (B, num_labels)
    (1, 2)
    """

    logits: Tensor
    loss: Tensor | None = None
    hidden_states: tuple[Tensor, ...] | None = None
    attentions: tuple[Tensor, ...] | None = None


@dataclass(slots=True)
class TokenClassificationOutput(ModelOutput):
    r"""Output of any per-token classification head (NER, POS, chunking).

    Attributes
    ----------
    logits : Tensor
        Per-token class logits, shape ``(B, T, num_labels)``.
    loss : Tensor or None, optional
        Scalar cross-entropy loss over the non-ignored positions.
    hidden_states : tuple[Tensor, ...] or None, optional
        Per-layer hidden states.
    attentions : tuple[Tensor, ...] or None, optional
        Per-layer attention weights.

    Notes
    -----
    Returned by every ``{Family}ForTokenClassification`` head.  The
    logits have the same rank as a masked-LM head's, but the last axis
    is a label set rather than a vocabulary — the distinct type keeps
    the two from being confused.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models import TokenClassificationOutput
    >>> out = TokenClassificationOutput(logits=lucid.zeros(1, 128, 9))
    >>> out.logits.shape              # (B, T, num_labels)
    (1, 128, 9)
    """

    logits: Tensor
    loss: Tensor | None = None
    hidden_states: tuple[Tensor, ...] | None = None
    attentions: tuple[Tensor, ...] | None = None


@dataclass(slots=True)
class QuestionAnsweringOutput(ModelOutput):
    r"""Output of any extractive span-prediction head.

    Attributes
    ----------
    start_logits : Tensor
        Per-position start-of-span logits, shape ``(B, T)``.
    end_logits : Tensor
        Per-position end-of-span logits, shape ``(B, T)``.
    loss : Tensor or None, optional
        Mean of the start and end cross-entropy terms when both
        ``start_positions`` and ``end_positions`` were supplied.
    hidden_states : tuple[Tensor, ...] or None, optional
        Per-layer hidden states.
    attentions : tuple[Tensor, ...] or None, optional
        Per-layer attention weights.

    Notes
    -----
    Returned by every ``{Family}ForQuestionAnswering`` head.  The two
    logit tensors are kept separate rather than fused into a single
    ``(B, T, 2)`` tensor: a caller that wants the best span argmaxes
    each independently, and a fused tensor forces every one of them to
    remember which trailing index means "start".

    Examples
    --------
    >>> import lucid
    >>> from lucid.models import QuestionAnsweringOutput
    >>> out = QuestionAnsweringOutput(
    ...     start_logits=lucid.zeros(1, 384),
    ...     end_logits=lucid.zeros(1, 384),
    ... )
    >>> out.start_logits.shape, out.end_logits.shape
    ((1, 384), (1, 384))
    """

    start_logits: Tensor
    end_logits: Tensor
    loss: Tensor | None = None
    hidden_states: tuple[Tensor, ...] | None = None
    attentions: tuple[Tensor, ...] | None = None
