"""Composable mixins for model families.

Tier-2 reusable behaviour — added only when ≥3 families would otherwise
duplicate the same logic.  Mixins carry no state (no ``__init__`` of
their own); they expose methods that operate on attributes the host class
is contracted to provide.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, is_dataclass, replace
from typing import TYPE_CHECKING, ClassVar, cast

import lucid
from lucid._C import engine as _C_engine
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor.tensor import Tensor
from lucid.utils.cache import Cache, DynamicCache, StaticCache
from lucid.models._base import ModelConfig, PretrainedModel
from lucid.models._sampling import _SamplingParams, _select_next_ondevice

if TYPE_CHECKING:
    from lucid.models._output import GenerationOutput, ImageClassificationOutput
    from lucid.models.generative._common._schedulers import DiffusionScheduler


@dataclass(frozen=True, slots=True)
class FeatureInfo:
    r"""Specification of one stage emitted by a feature-extracting backbone.

    Used by detection / segmentation heads that consume a *feature
    pyramid*: each entry describes the channel count and spatial
    down-sampling factor of one backbone stage relative to the input.

    Attributes
    ----------
    stage : int
        Zero-based stage index, in the order the backbone emits features
        (typically deepening from stage 0 toward higher receptive field).
    num_channels : int
        Channel count of the feature map at this stage.
    reduction : int
        Spatial down-sampling factor vs the network input — e.g. ``4``,
        ``8``, ``16``, ``32`` for the canonical CNN stage taps.

    Notes
    -----
    The layout is timm-compatible: detection heads that consume timm
    backbones drop in unchanged.  See :class:`BackboneMixin` for the
    protocol that produces a ``list[FeatureInfo]``.

    Examples
    --------
    >>> from lucid.models._mixins import FeatureInfo
    >>> info = FeatureInfo(stage=3, num_channels=512, reduction=32)
    >>> info.reduction
    32
    """

    stage: int
    num_channels: int
    reduction: int  # spatial down-sampling factor vs the input


class BackboneMixin(ABC):
    r"""Marker mixin for models usable as feature extractors.

    Detection and segmentation heads consume backbones through this small
    protocol: a ``forward_features(x)`` that emits the deepest stage's
    feature map, and a ``feature_info`` property enumerating every
    emitted stage's spec.  Implementing classes get a uniform interface
    that head builders (FPN, BiFPN, …) can target.

    Notes
    -----
    The mixin carries no state — it only declares two abstract slots.
    Subclasses MUST override:

    * :meth:`forward_features` — returns the final stage's feature map
      (and may stash intermediate stages on ``self`` for downstream
      use).
    * :attr:`feature_info` — returns a ``list[FeatureInfo]`` with one
      entry per stage the backbone exposes.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from dataclasses import dataclass
    >>> from typing import ClassVar
    >>> from lucid.models import ModelConfig, PretrainedModel
    >>> from lucid.models._mixins import BackboneMixin, FeatureInfo
    >>> @dataclass(frozen=True)
    ... class TinyConfig(ModelConfig):
    ...     model_type: ClassVar[str] = "tiny"
    ...     hidden_size: int = 64
    >>> class TinyBackbone(PretrainedModel, BackboneMixin):
    ...     config_class: ClassVar[type[TinyConfig]] = TinyConfig
    ...     def __init__(self, config):
    ...         super().__init__(config)
    ...         self.stages = nn.Conv2d(3, config.hidden_size, 3, stride=4, padding=1)
    ...     def forward_features(self, x):
    ...         return self.stages(x)
    ...     @property
    ...     def feature_info(self):
    ...         return [FeatureInfo(stage=0, num_channels=64, reduction=4)]
    >>> backbone = TinyBackbone(TinyConfig()).eval()
    >>> backbone.forward_features(lucid.randn(1, 3, 32, 32)).shape
    (1, 64, 8, 8)

    ``feature_info`` is what a detector reads to size its neck, so the
    reduction it advertises has to be the stride the features actually
    came out at.

    >>> backbone.feature_info[0].reduction
    4
    """

    @abstractmethod
    def forward_features(self, x: Tensor) -> Tensor:
        r"""Run the backbone's feature-extraction pass.

        Parameters
        ----------
        x : Tensor
            Input image batch, shape ``(B, C, H, W)``.

        Returns
        -------
        Tensor
            Feature map at the deepest stage.  Concrete subclasses may
            additionally stash earlier stages on the instance so
            multi-stage consumers can fetch them.

        Examples
        --------
        The map returned is the one :attr:`feature_info` lists last, so a
        neck can size itself from that entry without running the model.

        >>> import lucid
        >>> from lucid.models import create_model
        >>> backbone = create_model("resnet_18").eval()
        >>> last = backbone.feature_info[-1]
        >>> last.num_channels, last.reduction
        (512, 32)
        >>> backbone.forward_features(lucid.randn((1, 3, 64, 96))).shape
        (1, 512, 2, 3)

        Each side is divided by the advertised ``reduction`` on its own, so
        a non-square input keeps its aspect ratio.
        """
        ...

    _feature_info: list[FeatureInfo]

    @property
    def feature_info(self) -> list[FeatureInfo]:
        r"""Return the per-stage feature pyramid specification."""
        return self._feature_info


def _child_module(root: nn.Module, parent: nn.Module, name: str, path: str) -> nn.Module:
    """The submodule ``name`` of ``parent``, or an ``AttributeError`` naming ``path`` on ``root``."""
    child = getattr(parent, name, None)
    if isinstance(child, nn.Module):
        return child
    raise AttributeError(
        f"{type(root).__name__} has no submodule at {path!r}; "
        f"fix the head path its class declares"
    )


def _locate_class_head(root: nn.Module, path: str) -> tuple[nn.Module, str, nn.Linear]:
    """Resolve a dotted head ``path`` to ``(owner, attribute name, Linear)``.

    Containers keep their children under their position (``"0"``, ``"1"``),
    so a path walks a ``ModuleList`` the same way it walks an attribute.  A
    path that ends on a ``Sequential`` stands for the last ``Linear`` inside
    it: the head of a ``[Dropout, Linear]`` or a multi-layer classifier is
    the projection to the classes, not the layers before it.
    """
    *walk, name = path.split(".")
    owner = root
    for step in walk:
        owner = _child_module(root, owner, step, path)
    target = _child_module(root, owner, name, path)
    if isinstance(target, nn.Linear):
        return owner, name, target
    if isinstance(target, nn.Sequential):
        for child_name, child in reversed(list(target.named_children())):
            if isinstance(child, nn.Linear):
                return target, child_name, child
        raise RuntimeError(f"reset_classifier: no Linear layer found in {path!r}")
    raise NotImplementedError(f"reset_classifier not implemented for {type(target).__name__}")


def _renewed_linear(old: nn.Linear, num_classes: int) -> nn.Linear:
    """A freshly initialised ``Linear`` standing in for ``old``, with ``num_classes`` outputs.

    It is built where ``old`` lives: same device, same dtype, same bias and
    train/eval mode.  A model already moved with ``.half()`` /
    ``.to("metal")`` would otherwise get a float32 CPU head and fail its next
    forward on a dtype or device mismatch.
    """
    new = nn.Linear(
        old.in_features,
        num_classes,
        bias=old.bias is not None,
        device=old.weight.device,
        dtype=old.weight.dtype,
    )
    return new.train(old.training)


def _with_num_classes(config: ModelConfig, num_classes: int) -> ModelConfig:
    """``config`` with ``num_classes`` replaced, which re-runs its own validation."""
    if not is_dataclass(config) or isinstance(config, type):
        raise TypeError(f"{type(config).__name__} must be a dataclass instance")
    return replace(config, num_classes=num_classes)


def _reset_class_heads(model: PretrainedModel, paths: tuple[str, ...], num_classes: int) -> None:
    """Replace the ``Linear`` at each of ``paths`` and move ``model.config`` to ``num_classes``.

    Every replacement and the new config are built first and installed after, so
    a path that names nothing, or a config that refuses the width, leaves the
    model as it was.
    """
    config = _with_num_classes(model.config, num_classes)
    swaps = []
    for path in paths:
        owner, name, old = _locate_class_head(model, path)
        swaps.append((owner, name, _renewed_linear(old, num_classes)))
    for owner, name, head in swaps:
        setattr(owner, name, head)
    model.config = config


class ClassificationHeadMixin:
    r"""Standard Linear classification head + transfer-learning helper.

    Adds two methods to any classifier subclass: :meth:`_build_classifier`
    constructs a ``Linear`` (optionally wrapped in a ``Dropout``)
    according to the config, and :meth:`reset_classifier` swaps in a
    freshly-initialised Linear with a new ``num_classes`` for transfer
    learning.

    Notes
    -----
    The contract: subclasses must call ``self._build_classifier(...)``
    inside their ``__init__`` *after* ``super().__init__(config)`` so the
    backbone is in place before the head is attached.

    The host class exposes ``self.classifier`` as either a plain
    ``nn.Linear`` (no dropout configured) or an ``nn.Sequential`` of
    ``[Dropout, Linear]``; :meth:`reset_classifier` handles both shapes.

    The host is a :class:`PretrainedModel` whose config carries ``num_classes``;
    :meth:`reset_classifier` keeps that field in step with the head.

    A family whose head lives elsewhere, or is more than one ``Linear``,
    declares where in ``_class_heads`` — dotted paths from the model to each
    ``Linear`` whose width is ``config.num_classes`` (``("head",)`` for ViT,
    ``("head", "head_dist")`` for a model with a distillation head,
    ``("head.0", "head.1")`` for a ``ModuleList`` of heads).  A head that
    exists only under some config, such as an auxiliary classifier, adds its
    path by overriding :meth:`_class_head_paths`.  Nothing else about the
    swap is the family's to repeat.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from dataclasses import dataclass
    >>> from typing import ClassVar
    >>> from lucid.models import ModelConfig, PretrainedModel
    >>> from lucid.models._mixins import ClassificationHeadMixin
    >>> @dataclass(frozen=True)
    ... class TinyConfig(ModelConfig):
    ...     model_type: ClassVar[str] = "tiny"
    ...     hidden_size: int = 64
    ...     num_classes: int = 1000
    ...     dropout: float = 0.0
    >>> class TinyClassifier(PretrainedModel, ClassificationHeadMixin):
    ...     config_class: ClassVar[type[TinyConfig]] = TinyConfig
    ...     def __init__(self, config):
    ...         super().__init__(config)
    ...         self.pool = nn.AdaptiveAvgPool2d(1)
    ...         self._build_classifier(
    ...             config.hidden_size, config.num_classes, dropout=config.dropout
    ...         )
    ...     def forward(self, x):
    ...         return self.classifier(self.pool(x).reshape(x.shape[0], -1))
    >>> model = TinyClassifier(TinyConfig()).eval()
    >>> model(lucid.randn(2, 64, 7, 7)).shape
    (2, 1000)

    Transfer learning swaps the head and leaves the body where it is.  The
    new head takes the old one's dtype and device, and the config follows it,
    so a model already cast with ``.half()`` keeps working and a saved copy
    reloads at the new width.

    >>> model.reset_classifier(num_classes=10)
    >>> model(lucid.randn(2, 64, 7, 7)).shape
    (2, 10)
    >>> model.config.num_classes
    10
    """

    classifier: nn.Linear | nn.Sequential
    #: Dotted paths to every ``Linear`` sized by ``config.num_classes``.
    _class_heads: ClassVar[tuple[str, ...]] = ("classifier",)

    def _build_classifier(
        self, in_features: int, num_classes: int, *, dropout: float = 0.0
    ) -> None:
        r"""Install the classification head on ``self.classifier``.

        Parameters
        ----------
        in_features : int
            Channel count entering the head (backbone pooled-feature
            dimension).
        num_classes : int
            Output class count.
        dropout : float, optional, keyword-only, default=0.0
            Drop probability.  When positive, the head becomes
            ``Sequential(Dropout(p), Linear(in, out))``; when zero, just a
            bare ``Linear``.

        Notes
        -----
        Called from the subclass ``__init__`` after the backbone is
        constructed.  Idempotent: calling twice replaces the previous
        head.
        """
        if dropout > 0.0:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(in_features, num_classes)
            )
        else:
            self.classifier = nn.Linear(in_features, num_classes)

    def _class_head_paths(self) -> tuple[str, ...]:
        r"""Dotted paths to the ``Linear`` layers :meth:`reset_classifier` replaces.

        ``_class_heads`` for the family.  Override to add a head that exists
        only under some configs, such as an auxiliary classifier.
        """
        return type(self)._class_heads

    def reset_classifier(self, num_classes: int) -> None:
        r"""Replace the classification head with a freshly initialised one.

        Parameters
        ----------
        num_classes : int
            New output dimensionality.  ``in_features`` is preserved from
            the current head.

        Raises
        ------
        ValueError
            If ``num_classes`` is not positive.
        RuntimeError
            If the head is a ``Sequential`` containing no ``Linear``.
        NotImplementedError
            If the head is some other custom module type.
        AttributeError
            If a path the family declares in ``_class_heads`` names no
            submodule.

        Notes
        -----
        Used during transfer learning — typically called immediately
        after :meth:`from_pretrained` to adapt a 1000-class ImageNet head
        to a downstream task with a different label set.

        The new head is built with the dtype, device and train/eval mode of
        the head it replaces, so the model keeps working after ``.half()``,
        ``.to(lucid.bfloat16)`` or ``.to("metal")``.  ``config.num_classes``
        is updated too: ``save_pretrained`` then ``from_pretrained`` rebuilds
        the model at the new width instead of failing on a size mismatch.
        Every new head is built before anything is changed, so a refusal
        leaves the model as it was.

        Examples
        --------
        >>> model = AutoModelForImageClassification.from_pretrained("resnet_50")
        >>> model.reset_classifier(num_classes=10)
        >>> model(lucid.randn(1, 3, 224, 224)).logits.shape
        (1, 10)
        >>> model.config.num_classes
        10
        """
        if num_classes < 1:
            raise ValueError(f"reset_classifier: num_classes must be positive, got {num_classes}")
        if not isinstance(self, PretrainedModel):
            raise TypeError(
                f"{type(self).__name__} mixes in ClassificationHeadMixin but is not a PretrainedModel"
            )
        _reset_class_heads(self, self._class_head_paths(), num_classes)

    def _classify(
        self, features: Tensor, labels: Tensor | None = None, *, label_smoothing: float = 0.0
    ) -> ImageClassificationOutput:
        r"""Project ``features`` through ``self.classifier`` and compute optional cross-entropy loss."""
        from lucid.models._output import ImageClassificationOutput

        logits: Tensor = self.classifier(features)
        loss: Tensor | None = None
        if labels is not None:
            loss = (
                F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
                if label_smoothing != 0.0
                else F.cross_entropy(logits, labels)
            )
        return ImageClassificationOutput(logits=logits, loss=loss)


# ─────────────────────────────────────────────────────────────────────────────
# MaskedLMMixin — per-token cross-entropy loss for masked / token-level heads
# ─────────────────────────────────────────────────────────────────────────────


class MaskedLMMixin:
    r"""Per-token cross-entropy loss helper for masked-LM / token-class heads.

    Encoder text families (BERT, RoFormer, …) all reduce a ``(B, T, V)``
    logit tensor against ``(B, T)`` labels with the same recipe: flatten
    the sequence axis into the batch axis, then run
    :func:`lucid.nn.functional.cross_entropy` with ``ignore_index=-100``.
    Centralising the call here removes a 3-line stanza repeated across
    every ``ForMaskedLM`` / ``ForTokenClassification`` class.

    Notes
    -----
    The mixin is stateless and exposes a single ``@staticmethod`` —
    inheriting from it is purely a documentation / discoverability
    convenience; the same effect is obtained by calling
    ``MaskedLMMixin.compute_lm_loss(...)`` directly.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models import create_model
    >>> model = create_model("bert_base_mlm").eval()
    >>> input_ids = lucid.zeros(1, 16, dtype=lucid.int64)
    >>> out = model(input_ids)
    >>> out.logits.shape              # (B, T, vocab)
    (1, 16, 30522)

    ``compute_lm_loss`` is only reached when labels are supplied, so a
    forward without them leaves the loss empty rather than zero.

    >>> out.loss is None
    True
    """

    @staticmethod
    def compute_lm_loss(logits: Tensor, labels: Tensor, *, ignore_index: int = -100) -> Tensor:
        r"""Compute per-token cross-entropy loss for an LM head.

        Parameters
        ----------
        logits : Tensor
            ``(B, T, V)`` per-token logits.
        labels : Tensor
            ``(B, T)`` int target token ids.  Entries equal to
            ``ignore_index`` contribute zero loss and zero gradient.
        ignore_index : int, optional, keyword-only, default=-100
            Token id to skip during reduction.  Convention from the wider
            ML ecosystem: ``-100`` marks non-masked positions in MLM and
            padding tokens in token classification.

        Returns
        -------
        Tensor
            Scalar loss tensor (mean over the contributing positions).

        Notes
        -----
        Implementation: flatten ``logits`` to ``(B*T, V)`` and ``labels``
        to ``(B*T,)``, cast labels to ``long``, then call
        :func:`lucid.nn.functional.cross_entropy`.  No probabilistic
        reweighting; the loss is a plain mean over non-ignored positions.

        Examples
        --------
        >>> logits = lucid.randn(2, 10, 32000)
        >>> labels = lucid.randint(0, 32000, (2, 10))
        >>> loss = MaskedLMMixin.compute_lm_loss(logits, labels)
        >>> loss.shape
        ()
        """
        B, T, V = logits.shape
        return F.cross_entropy(
            logits.reshape(B * T, V), labels.reshape(B * T).long(), ignore_index=ignore_index
        )


def _extract_last_logits(outputs: object) -> Tensor:
    """Return ``outputs[:, -1, :]`` (or ``outputs.logits[:, -1, :]``)."""
    logits = cast(Tensor, outputs.logits if hasattr(outputs, "logits") else outputs)
    return logits[:, -1, :]


def _validate_static_cache_bounds(cfg: object, max_cache_len: int | None, stop_len: int) -> None:
    """Verify ``stop_len`` fits inside ``max_position_embeddings`` and ``max_cache_len``."""
    max_pos = getattr(cfg, "max_position_embeddings", None) if cfg else None
    if max_pos is not None and stop_len > int(max_pos):
        raise ValueError(
            f"generate(): target length {stop_len} exceeds the model's "
            f"max_position_embeddings {int(max_pos)}"
        )
    if max_cache_len is not None and int(max_cache_len) < stop_len:
        raise ValueError(
            f"generate(): max_cache_len={int(max_cache_len)} is smaller than "
            f"the target length {stop_len} (prompt + generated) — the static "
            f"buffer cannot hold every position"
        )


def _setup_causal_cache(
    model: nn.Module,
    use_cache: bool,
    cache_implementation: str,
    max_cache_len: int | None,
    stop_len: int,
) -> tuple[Cache | None, Callable[[Tensor, Tensor], Tensor] | None]:
    """Build ``(past_key_values, compiled_step)`` for causal generation."""
    if not use_cache:
        return None, None
    if cache_implementation not in ("static", "compiled"):
        return DynamicCache(), None
    past = StaticCache(max_cache_len=(max_cache_len if max_cache_len is not None else stop_len))
    if cache_implementation != "compiled":
        return past, None
    from lucid.compile._entry.decode_step import compiled_decode_step

    def _decode_fwd(tok: Tensor, cp: Tensor) -> Tensor:
        return _extract_last_logits(
            model(tok, use_cache=True, past_key_values=past, cache_position=cp)
        )

    return past, compiled_decode_step(_decode_fwd, past)


# ─────────────────────────────────────────────────────────────────────────────
# CausalLMMixin — autoregressive text sampling
# ─────────────────────────────────────────────────────────────────────────────


class CausalLMMixin:
    r"""Decoder-only autoregressive sampling for causal LM heads.

    The generation counterpart of :class:`MaskedLMMixin` (the two canonical LM
    objectives: causal vs masked).  This drives **autoregressive token-by-token**
    text generation — distinct from :class:`DiffusionMixin`, which runs an
    iterative denoising loop for image-generation models.

    Provides :meth:`generate` — greedy or stochastic next-token sampling
    with temperature, top-k, top-p (nucleus), and repetition-penalty
    knobs.  Per-sequence stopping at ``eos_token_id`` is honoured.

    Concrete subclasses must:

    * Inherit from :class:`lucid.models.PretrainedModel`.
    * Define ``forward(input_ids, ...)`` returning a
      :class:`lucid.models.CausalLMOutput` with ``logits`` of shape
      ``(B, T, vocab_size)``.
    * Expose a ``config`` attribute carrying ``vocab_size`` /
      ``pad_token_id`` / ``bos_token_id`` / ``eos_token_id`` (fields
      :class:`lucid.models.text.LanguageModelConfig` defines).

    Notes
    -----
    By default (:code:`use_cache=True`), :meth:`generate` threads a
    key/value cache through ``forward`` so each step only encodes the newly
    sampled token:

    * ``cache_implementation="dynamic"`` (default) uses
      :class:`~lucid.utils.cache.DynamicCache` (grows by concatenation).
    * ``cache_implementation="static"`` uses a pre-allocated
      :class:`~lucid.utils.cache.StaticCache` updated in place at
      ``cache_position``.
    * ``cache_implementation="compiled"`` pairs
      :class:`~lucid.utils.cache.StaticCache` with
      :func:`~lucid.compile._entry.decode_step.compiled_decode_step`,
      compiling the single-token decode forward once into a reused
      :class:`MPSGraphExecutable` after an eager prompt prefill.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models import create_model
    >>> model = create_model("gpt2_small_lm").eval()
    >>> prompt = lucid.zeros(1, 3, dtype=lucid.int64)
    >>> out = model.generate(prompt, max_new_tokens=10)
    >>> out.shape                     # the prompt, then what was written
    (1, 13)
    """

    @lucid.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        *,
        max_length: int = 20,
        max_new_tokens: int | None = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        pad_token_id: int | None = None,
        eos_token_id: int | None = None,
        use_cache: bool = True,
        cache_implementation: str = "dynamic",
        max_cache_len: int | None = None,
    ) -> Tensor:
        r"""Autoregressively extend ``input_ids`` until a stop condition.

        Parameters
        ----------
        input_ids : Tensor
            ``(B, T_prompt)`` int prompt tokens.
        max_length : int, optional, keyword-only, default=20
            Total sequence length cap (prompt + generated).  Ignored when
            ``max_new_tokens`` is supplied.
        max_new_tokens : int or None, optional, keyword-only
            Cap on the number of *new* tokens to generate, additive over
            the prompt length.  Takes precedence over ``max_length``.
        do_sample : bool, optional, keyword-only, default=False
            ``False`` → greedy argmax decoding; ``True`` → stochastic
            sampling honouring ``temperature`` / ``top_k`` / ``top_p``.
        temperature : float, optional, keyword-only, default=1.0
            Logit divisor before softmax.  ``< 1`` sharpens, ``> 1``
            flattens.  Ignored under greedy decoding.
        top_k : int or None, optional, keyword-only
            Keep only the K highest-probability tokens before sampling.
            ``None`` disables.
        top_p : float or None, optional, keyword-only
            Nucleus sampling — keep the smallest token set whose
            cumulative probability ≥ ``top_p``.  ``None`` disables.
        repetition_penalty : float, optional, keyword-only, default=1.0
            Multiply the logit of every previously generated token by
            ``1 / penalty`` (and by ``penalty`` if the logit is
            negative).  ``1.0`` → no effect; ``> 1`` discourages
            repetition.
        pad_token_id : int or None, optional, keyword-only
            Token id used to pad finished sequences after they emit
            ``eos_token_id``.  Defaults to ``config.pad_token_id``, then
            ``0`` if unset.
        eos_token_id : int or None, optional, keyword-only
            Stop generating per-sequence once this id is emitted.
            Defaults to ``config.eos_token_id``.
        use_cache : bool, optional, keyword-only, default=True
            Use a KV cache so each step only encodes the new token (O(T²)
            generation) instead of re-running the whole prefix (O(T³)).  Falls
            back to re-encoding automatically when the host model has no cache.
        cache_implementation : str, optional, keyword-only, default="dynamic"
            ``"dynamic"`` → :class:`~lucid.utils.cache.DynamicCache` (grows by
            concatenation); ``"static"`` →
            :class:`~lucid.utils.cache.StaticCache` (fixed pre-allocated buffer
            written in place at ``cache_position``, attending over only the filled
            prefix); ``"compiled"`` → :class:`~lucid.utils.cache.StaticCache`
            driven by
            :func:`~lucid.compile._entry.decode_step.compiled_decode_step`
            (prefills the prompt eagerly, then compiles the single-token decode
            step once into a reused :class:`MPSGraphExecutable`).
        max_cache_len : int or None, optional, keyword-only
            Buffer capacity for ``cache_implementation="static"`` or
            ``"compiled"``; defaults to the total target length (prompt +
            generated).  An over-sized buffer is free in eager ``"static"``
            mode — attention narrows to the filled prefix regardless.

        Returns
        -------
        Tensor
            ``(B, T_final)`` int tensor where ``T_final`` ≤ ``max_length``
            (or ``T_prompt + max_new_tokens``).  Sequences that hit
            ``eos_token_id`` early are right-padded with ``pad_token_id``.

        Raises
        ------
        ValueError
            If ``input_ids`` is not 2-D.

        Notes
        -----
        Decorated with :func:`lucid.no_grad` — generation never builds an
        autograd graph.  Each step:

        1. Materialise the current prefix as ``(B, T_cur)``.
        2. Run a forward pass; take logits at the last position.
        3. Apply repetition penalty.
        4. Greedy argmax, or apply temperature / top-k / top-p and
           inverse-CDF sample one token per row.
        5. Replace finished rows' tokens with ``pad_token_id``.

        Loop breaks early when every row has emitted EOS.

        Examples
        --------
        >>> model = AutoModelForCausalLM.from_pretrained("gpt")
        >>> tokens = lucid.tensor([[1, 2, 3]]).long()
        >>> out = model.generate(tokens, max_new_tokens=5)
        >>> out.shape
        (1, 8)
        """
        # ── Resolve token id defaults from the host model's config ────────
        cfg = getattr(self, "config", None)
        if eos_token_id is None and cfg is not None:
            eos_token_id = getattr(cfg, "eos_token_id", None)
        if pad_token_id is None and cfg is not None:
            pad_token_id = getattr(cfg, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = 0

        if input_ids.ndim != 2:
            raise ValueError(
                f"generate(): input_ids must be 2-D (B, T), got shape {tuple(input_ids.shape)}"
            )

        B = int(input_ids.shape[0])
        T_prompt = int(input_ids.shape[1])

        stop_len = T_prompt + int(max_new_tokens) if max_new_tokens is not None else int(max_length)
        if stop_len <= T_prompt:
            return input_ids

        if use_cache and cache_implementation in ("static", "compiled"):
            _validate_static_cache_bounds(cfg, max_cache_len, stop_len)

        dev = input_ids.device.type
        finished: Tensor = lucid.zeros((B,), device=dev).bool()
        out_tokens: list[Tensor] = [input_ids[:, t] for t in range(T_prompt)]
        model = cast(nn.Module, self)
        sampling = _SamplingParams(
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            dev=dev,
        )

        past, compiled_step = _setup_causal_cache(
            model, use_cache, cache_implementation, max_cache_len, stop_len
        )
        model_input = input_ids

        for _step in range(stop_len - T_prompt):
            if compiled_step is not None and _step > 0:
                cp = lucid.tensor([T_prompt + _step - 1], device=dev).long()
                last_logits = compiled_step(model_input, cp)
            elif past is not None:
                try:
                    outputs = model(model_input, use_cache=True, past_key_values=past)
                except TypeError as exc:
                    if isinstance(exc, _C_engine.LucidError):
                        raise
                    past = None
                    compiled_step = None
                    outputs = model(lucid.stack(out_tokens, dim=1))
                last_logits = _extract_last_logits(outputs)
            else:
                last_logits = _extract_last_logits(model(lucid.stack(out_tokens, dim=1)))
            finished = _select_next_ondevice(last_logits, out_tokens, finished, sampling)
            if sampling.eos_token_id is not None and bool(lucid.all(finished).item()):
                break
            model_input = out_tokens[-1].reshape(B, 1)

        return lucid.stack(out_tokens, dim=1).long()


# ─────────────────────────────────────────────────────────────────────────────
# DiffusionMixin — image-generation sampling loop
# ─────────────────────────────────────────────────────────────────────────────


class DiffusionMixin:
    r"""Reverse-process sampling loop for diffusion-family generative models.

    Concrete subclasses must:

    * Inherit from :class:`lucid.models.PretrainedModel`.
    * Define ``forward(sample, timestep, ...) -> DiffusionModelOutput``
      that returns the network's prediction at the supplied timestep.
    * Expose a ``config`` attribute carrying ``sample_size`` /
      ``in_channels`` (fields :class:`DiffusionModelConfig` defines) — or
      explicitly pass ``generator_shape`` to :meth:`generate`.

    The mixin then provides :meth:`generate` — a vanilla reverse-process
    loop parametrised by an externally-supplied
    :class:`DiffusionScheduler`.  Sampler choice (DDPM, DDIM, …) is thus
    a runtime knob rather than baked into the model class.

    Notes
    -----
    The mixin is stateless.  All hyper-parameters (number of inference
    steps, image shape, batch size) flow through :meth:`generate`'s
    keyword arguments; the model class itself stays sampler-agnostic.

    Examples
    --------
    >>> from lucid.models import AutoModelForImageGeneration
    >>> from lucid.models.generative import DDPMScheduler
    >>> model = AutoModelForImageGeneration.from_pretrained("ddpm_cifar_gen")
    >>> scheduler = DDPMScheduler(num_train_timesteps=1000)
    >>> out = model.generate(scheduler, n_samples=4, num_inference_steps=50)
    >>> out.samples.shape
    (4, 3, 32, 32)
    """

    @lucid.no_grad()
    def generate(
        self,
        scheduler: DiffusionScheduler,
        *,
        n_samples: int = 1,
        num_inference_steps: int | None = None,
        generator_shape: tuple[int, ...] | None = None,
        return_intermediates: bool = False,
        device: str | None = None,
    ) -> GenerationOutput:
        r"""Sample ``n_samples`` images via the reverse diffusion process.

        Parameters
        ----------
        scheduler : DiffusionScheduler
            Already-constructed scheduler (DDPM, DDIM, etc.).  Encapsulates
            the noise schedule and the per-step update rule.
        n_samples : int, optional, keyword-only, default=1
            Batch size of the generated output.
        num_inference_steps : int or None, optional, keyword-only
            If ``None``, the scheduler's ``timesteps`` are used as-is;
            otherwise ``scheduler.set_timesteps(num_inference_steps, ...)``
            is called first to (re)build the timestep schedule.
        generator_shape : tuple[int, ...] or None, optional, keyword-only
            Per-sample tensor shape, ``(C, H, W)``.  When omitted,
            defaults to ``(in_channels, H, W)`` derived from
            ``self.config``.
        return_intermediates : bool, optional, keyword-only, default=False
            If ``True``, every per-step sample is recorded and returned
            in :attr:`GenerationOutput.intermediates`.
        device : str or None, optional, keyword-only, default=None
            Device to allocate the initial Gaussian noise, the timestep
            batch and the scheduler's timesteps on.  ``None`` follows the
            model's own parameters, so ``model.to("metal").generate(...)``
            samples on Metal; an explicit value still wins.

        Returns
        -------
        GenerationOutput
            ``.samples`` is the final ``(n_samples, C, H, W)`` batch;
            ``.intermediates`` is populated when ``return_intermediates``
            is ``True``.

        Raises
        ------
        RuntimeError
            If ``generator_shape`` is ``None`` and the model lacks a
            ``config`` with ``sample_size`` / ``in_channels`` fields.

        Notes
        -----
        Decorated with :func:`lucid.no_grad` — sampling never builds an
        autograd graph.  Standard reverse-process loop:

        1. Draw :math:`x_T \sim \mathcal{N}(0, I)`.
        2. For each ``t`` in ``scheduler.timesteps`` (descending):
           run the model forward, then apply
           ``scheduler.step(model_pred, t, x_t)`` to obtain :math:`x_{t-1}`.

        Examples
        --------
        >>> model = AutoModelForImageGeneration.from_pretrained("ddpm_cifar_gen")
        >>> scheduler = DDPMScheduler(num_train_timesteps=1000)
        >>> out = model.generate(scheduler, n_samples=4, num_inference_steps=50,
        ...                      device="cpu")
        >>> out.samples.shape
        (4, 3, 32, 32)
        """
        from lucid.models._output import GenerationOutput
        from lucid.models._utils import _generative
        from lucid.models._utils._common import host_ints

        module = cast(nn.Module, self)
        # A hard "cpu" default drew the noise off-device for any model that
        # had been moved to Metal, so the first forward raised.
        device = _generative.resolve_generation_device(module, device)

        if num_inference_steps is not None:
            scheduler.set_timesteps(num_inference_steps, device=device)

        # Resolve image shape from the config when caller doesn't override.
        if generator_shape is None:
            cfg = getattr(self, "config", None)
            if cfg is None:
                raise RuntimeError(
                    "DiffusionMixin.generate needs either generator_shape or a "
                    "model.config with sample_size / in_channels fields."
                )
            sample_size = getattr(cfg, "sample_size", None)
            in_channels = getattr(cfg, "in_channels", None)
            if sample_size is None or in_channels is None:
                raise RuntimeError("model.config must provide sample_size and in_channels")
            if isinstance(sample_size, tuple):
                H, W = int(sample_size[0]), int(sample_size[1])
            else:
                H = W = int(sample_size)
            generator_shape = (int(in_channels), H, W)

        sample = _generative.start_noise(module, (n_samples, *generator_shape), device)

        intermediates: list[Tensor] = []
        # Schedulers keep their timesteps on the sampling device; read them
        # back once rather than syncing on every step.
        for t_int in host_ints(scheduler.timesteps):
            t_batch = lucid.tensor([t_int] * n_samples, device=device).long()
            output = module(sample, t_batch)
            model_pred = output.sample if hasattr(output, "sample") else output
            sample = scheduler.step(cast(Tensor, model_pred), t_int, sample)
            if return_intermediates:
                intermediates.append(sample)

        return GenerationOutput(
            samples=sample, intermediates=tuple(intermediates) if return_intermediates else None
        )


# Re-import for the inner ``Scheduler`` forward reference above — the
# scheduler module imports lucid.nn.functional which transitively pulls in
# this mixin module, so we must keep the import lazy (inside generate).
