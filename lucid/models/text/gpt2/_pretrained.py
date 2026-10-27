"""Registry factories for GPT-2 variants (Radford et al., 2019).

Canonical sizes from the OpenAI release.  All four trunks and their causal-LM
heads ship WebText pretrained weights through :mod:`lucid.weights` (per-factory
``*Weights`` enums + ``weights=`` / ``pretrained=``).  The
``*_cls`` fine-tune head carries no canonical pretrained checkpoint.
"""

import lucid.weights as weights_mod
from lucid.models._registry import register_model
from lucid.models.text.gpt2._config import GPT2Config
from lucid.models.text.gpt2._model import GPT2ForSequenceClassification, GPT2LMHeadModel, GPT2Model
from lucid.models.text.gpt2._weights import (
    GPT2LargeLMWeights,
    GPT2LargeWeights,
    GPT2MediumLMWeights,
    GPT2MediumWeights,
    GPT2SmallLMWeights,
    GPT2SmallWeights,
    GPT2XLargeLMWeights,
    GPT2XLargeWeights,
)
from lucid.models._utils._common import apply_overrides

# OpenAI release table — Radford et al., 2019 §2.3 (the only published
# GPT-2 sizes are small / medium / large / XL).
_CFG_SMALL = GPT2Config()  # 124M — the default
_CFG_MEDIUM = GPT2Config(
    hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, intermediate_size=4096
)
_CFG_LARGE = GPT2Config(
    hidden_size=1280, num_hidden_layers=36, num_attention_heads=20, intermediate_size=5120
)
_CFG_XLARGE = GPT2Config(
    hidden_size=1600, num_hidden_layers=48, num_attention_heads=25, intermediate_size=6400
)


# ── Backbones ─────────────────────────────────────────────────────────────────


@register_model(
    task="base", family="gpt2", model_type="gpt2", model_class=GPT2Model, default_config=_CFG_SMALL
)
def gpt2_small(
    pretrained: bool | str = False, *, weights: GPT2SmallWeights | None = None, **overrides: object
) -> GPT2Model:
    r"""Construct a GPT-2 small (124M) decoder trunk.

    Smallest of the four canonical OpenAI variants from Radford, Wu, Child,
    Luan, Amodei, and Sutskever, 2019: :math:`L=12` transformer blocks,
    :math:`H=768` hidden, :math:`A=12` attention heads, intermediate width
    3072.  Roughly 124M parameters (historically released as the "117M"
    file before a re-count).  This is the default entry point for GPT-2
    research and the parameter-count match for GPT-1.

    Parameters
    ----------
    pretrained : bool or str, default=False
        ``True`` loads the ``DEFAULT`` checkpoint of :class:`GPT2SmallWeights`; a
        string names a specific tag.  ``False`` returns a randomly initialised
        model.
    weights : GPT2SmallWeights, optional, keyword-only
        Explicit weights enum member; takes precedence over ``pretrained``.
    **overrides : object
        Optional :class:`GPT2Config` field overrides forwarded into the
        underlying config.

    Returns
    -------
    GPT2Model
        Decoder trunk configured with the GPT-2 small size and any
        overrides.

    Notes
    -----
    Reference: Radford, Wu, Child, Luan, Amodei, and Sutskever,
    *"Language Models are Unsupervised Multitask Learners"*, OpenAI
    Technical Report, 2019.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.gpt2 import gpt2_small
    >>> model = gpt2_small().eval()
    >>> input_ids = lucid.tensor([[15496, 11, 995, 13]])   # "Hello, world."
    >>> out = model(input_ids)
    >>> out.last_hidden_state.shape   # (1, 4, 768)
    (1, 4, 768)
    """
    entry = weights_mod.resolve_weights(GPT2SmallWeights, pretrained, weights)
    model = GPT2Model(apply_overrides(_CFG_SMALL, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="gpt2_small")
    return model


@register_model(
    task="base", family="gpt2", model_type="gpt2", model_class=GPT2Model, default_config=_CFG_MEDIUM
)
def gpt2_medium(
    pretrained: bool | str = False, *, weights: GPT2MediumWeights | None = None, **overrides: object
) -> GPT2Model:
    r"""Construct a GPT-2 medium (355M) decoder trunk.

    Second of the four canonical OpenAI variants: :math:`L=24` transformer
    blocks, :math:`H=1024` hidden, :math:`A=16` attention heads,
    intermediate width 4096.  Roughly 355M parameters — the inflection
    point at which zero-shot performance on reading comprehension /
    summarisation begins to substantially outperform random baselines
    (Radford 2019 Table 3).

    Parameters
    ----------
    pretrained : bool or str, default=False
        ``True`` loads the ``DEFAULT`` checkpoint of :class:`GPT2MediumWeights`; a
        string names a specific tag.  ``False`` returns a randomly initialised
        model.
    weights : GPT2MediumWeights, optional, keyword-only
        Explicit weights enum member; takes precedence over ``pretrained``.
    **overrides : object
        Optional :class:`GPT2Config` field overrides forwarded into the
        underlying config.

    Returns
    -------
    GPT2Model
        Decoder trunk configured with the GPT-2 medium size and any
        overrides.

    Notes
    -----
    Reference: Radford et al., *"Language Models are Unsupervised
    Multitask Learners"*, OpenAI Technical Report, 2019.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.gpt2 import gpt2_medium
    >>> model = gpt2_medium().eval()
    >>> input_ids = lucid.tensor([[15496, 11, 995, 13]])
    >>> out = model(input_ids)
    >>> out.last_hidden_state.shape   # (1, 4, 1024)
    (1, 4, 1024)
    """
    entry = weights_mod.resolve_weights(GPT2MediumWeights, pretrained, weights)
    model = GPT2Model(apply_overrides(_CFG_MEDIUM, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="gpt2_medium")
    return model


@register_model(
    task="base", family="gpt2", model_type="gpt2", model_class=GPT2Model, default_config=_CFG_LARGE
)
def gpt2_large(
    pretrained: bool | str = False, *, weights: GPT2LargeWeights | None = None, **overrides: object
) -> GPT2Model:
    r"""Construct a GPT-2 large (774M) decoder trunk.

    Third of the four canonical OpenAI variants: :math:`L=36` transformer
    blocks, :math:`H=1280` hidden, :math:`A=20` attention heads,
    intermediate width 5120.  Roughly 774M parameters — released in
    August 2019 after the staged-release decision; SOTA zero-shot on
    LAMBADA at release time.

    Parameters
    ----------
    pretrained : bool or str, default=False
        ``True`` loads the ``DEFAULT`` checkpoint of :class:`GPT2LargeWeights`; a
        string names a specific tag.  ``False`` returns a randomly initialised
        model.
    weights : GPT2LargeWeights, optional, keyword-only
        Explicit weights enum member; takes precedence over ``pretrained``.
    **overrides : object
        Optional :class:`GPT2Config` field overrides forwarded into the
        underlying config.

    Returns
    -------
    GPT2Model
        Decoder trunk configured with the GPT-2 large size and any
        overrides.

    Notes
    -----
    Reference: Radford et al., *"Language Models are Unsupervised
    Multitask Learners"*, OpenAI Technical Report, 2019.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.gpt2 import gpt2_large
    >>> model = gpt2_large().eval()
    >>> input_ids = lucid.tensor([[15496, 11, 995, 13]])
    >>> out = model(input_ids)
    >>> out.last_hidden_state.shape   # (1, 4, 1280)
    (1, 4, 1280)
    """
    entry = weights_mod.resolve_weights(GPT2LargeWeights, pretrained, weights)
    model = GPT2Model(apply_overrides(_CFG_LARGE, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="gpt2_large")
    return model


@register_model(
    task="base", family="gpt2", model_type="gpt2", model_class=GPT2Model, default_config=_CFG_XLARGE
)
def gpt2_xlarge(
    pretrained: bool | str = False, *, weights: GPT2XLargeWeights | None = None, **overrides: object
) -> GPT2Model:
    r"""Construct a GPT-2 XL (1.5B) decoder trunk.

    Largest of the four canonical OpenAI variants: :math:`L=48` transformer
    blocks, :math:`H=1600` hidden, :math:`A=25` attention heads,
    intermediate width 6400.  Roughly 1.5B parameters — fully released in
    November 2019 after staged disclosure; remained the largest publicly
    available autoregressive LM until GPT-3 (Brown 2020).

    Parameters
    ----------
    pretrained : bool or str, default=False
        ``True`` loads the ``DEFAULT`` checkpoint of :class:`GPT2XLargeWeights`; a
        string names a specific tag.  ``False`` returns a randomly initialised
        model.
    weights : GPT2XLargeWeights, optional, keyword-only
        Explicit weights enum member; takes precedence over ``pretrained``.
    **overrides : object
        Optional :class:`GPT2Config` field overrides forwarded into the
        underlying config.

    Returns
    -------
    GPT2Model
        Decoder trunk configured with the GPT-2 XL size and any overrides.

    Notes
    -----
    Reference: Radford et al., *"Language Models are Unsupervised
    Multitask Learners"*, OpenAI Technical Report, 2019.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.gpt2 import gpt2_xlarge
    >>> model = gpt2_xlarge().eval()
    >>> input_ids = lucid.tensor([[15496, 11, 995, 13]])
    >>> out = model(input_ids)
    >>> out.last_hidden_state.shape   # (1, 4, 1600)
    (1, 4, 1600)
    """
    entry = weights_mod.resolve_weights(GPT2XLargeWeights, pretrained, weights)
    model = GPT2Model(apply_overrides(_CFG_XLARGE, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="gpt2_xlarge")
    return model


# ── Causal-LM heads ───────────────────────────────────────────────────────────


@register_model(
    task="language-modeling",
    family="gpt2",
    model_type="gpt2",
    model_class=GPT2LMHeadModel,
    default_config=_CFG_SMALL,
)
def gpt2_small_lm(
    pretrained: bool | str = False,
    *,
    weights: GPT2SmallLMWeights | None = None,
    **overrides: object,
) -> GPT2LMHeadModel:
    r"""Construct a GPT-2 small (124M) model with the tied causal-LM head.

    Same trunk as :func:`gpt2_small` (L=12, H=768, A=12, ~124M parameters),
    wrapped with the tied LM head used for pre-training and free-form
    generation via :meth:`lucid.models._mixins.CausalLMMixin.generate`.

    Parameters
    ----------
    pretrained : bool or str, default=False
        ``True`` loads the ``DEFAULT`` checkpoint of :class:`GPT2SmallLMWeights`; a
        string names a specific tag.  ``False`` returns a randomly initialised
        model.
    weights : GPT2SmallLMWeights, optional, keyword-only
        Explicit weights enum member; takes precedence over ``pretrained``.
    **overrides : object
        Optional :class:`GPT2Config` field overrides forwarded into the
        underlying config.

    Returns
    -------
    GPT2LMHeadModel
        GPT-2 small trunk wrapped with the tied LM head.

    Notes
    -----
    Reference: Radford et al., *"Language Models are Unsupervised
    Multitask Learners"*, OpenAI Technical Report, 2019.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.gpt2 import gpt2_small_lm
    >>> model = gpt2_small_lm().eval()
    >>> input_ids = lucid.tensor([[15496, 11, 995, 13]])
    >>> out = model(input_ids)
    >>> out.logits.shape   # (1, 4, 50257)
    (1, 4, 50257)
    """
    entry = weights_mod.resolve_weights(GPT2SmallLMWeights, pretrained, weights)
    model = GPT2LMHeadModel(apply_overrides(_CFG_SMALL, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="gpt2_small_lm")
    return model


@register_model(
    task="language-modeling",
    family="gpt2",
    model_type="gpt2",
    model_class=GPT2LMHeadModel,
    default_config=_CFG_MEDIUM,
)
def gpt2_medium_lm(
    pretrained: bool | str = False,
    *,
    weights: GPT2MediumLMWeights | None = None,
    **overrides: object,
) -> GPT2LMHeadModel:
    r"""Construct a GPT-2 medium (355M) model with the tied causal-LM head.

    Same trunk as :func:`gpt2_medium` (L=24, H=1024, A=16, ~355M parameters),
    wrapped with the tied LM head for pre-training and free-form generation.

    Parameters
    ----------
    pretrained : bool or str, default=False
        ``True`` loads the ``DEFAULT`` checkpoint of :class:`GPT2MediumLMWeights`; a
        string names a specific tag.  ``False`` returns a randomly initialised
        model.
    weights : GPT2MediumLMWeights, optional, keyword-only
        Explicit weights enum member; takes precedence over ``pretrained``.
    **overrides : object
        Optional :class:`GPT2Config` field overrides forwarded into the
        underlying config.

    Returns
    -------
    GPT2LMHeadModel
        GPT-2 medium trunk wrapped with the tied LM head.

    Notes
    -----
    Reference: Radford et al., *"Language Models are Unsupervised
    Multitask Learners"*, OpenAI Technical Report, 2019.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.gpt2 import gpt2_medium_lm
    >>> model = gpt2_medium_lm().eval()
    >>> input_ids = lucid.tensor([[15496, 11, 995, 13]])
    >>> out = model(input_ids)
    >>> out.logits.shape   # (1, 4, 50257)
    (1, 4, 50257)
    """
    entry = weights_mod.resolve_weights(GPT2MediumLMWeights, pretrained, weights)
    model = GPT2LMHeadModel(apply_overrides(_CFG_MEDIUM, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="gpt2_medium_lm")
    return model


@register_model(
    task="language-modeling",
    family="gpt2",
    model_type="gpt2",
    model_class=GPT2LMHeadModel,
    default_config=_CFG_LARGE,
)
def gpt2_large_lm(
    pretrained: bool | str = False,
    *,
    weights: GPT2LargeLMWeights | None = None,
    **overrides: object,
) -> GPT2LMHeadModel:
    r"""Construct a GPT-2 large (774M) model with the tied causal-LM head.

    Same trunk as :func:`gpt2_large` (L=36, H=1280, A=20, ~774M parameters),
    wrapped with the tied LM head for pre-training and free-form generation.

    Parameters
    ----------
    pretrained : bool or str, default=False
        ``True`` loads the ``DEFAULT`` checkpoint of :class:`GPT2LargeLMWeights`; a
        string names a specific tag.  ``False`` returns a randomly initialised
        model.
    weights : GPT2LargeLMWeights, optional, keyword-only
        Explicit weights enum member; takes precedence over ``pretrained``.
    **overrides : object
        Optional :class:`GPT2Config` field overrides forwarded into the
        underlying config.

    Returns
    -------
    GPT2LMHeadModel
        GPT-2 large trunk wrapped with the tied LM head.

    Notes
    -----
    Reference: Radford et al., *"Language Models are Unsupervised
    Multitask Learners"*, OpenAI Technical Report, 2019.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.gpt2 import gpt2_large_lm
    >>> model = gpt2_large_lm().eval()
    >>> input_ids = lucid.tensor([[15496, 11, 995, 13]])
    >>> out = model(input_ids)
    >>> out.logits.shape   # (1, 4, 50257)
    (1, 4, 50257)
    """
    entry = weights_mod.resolve_weights(GPT2LargeLMWeights, pretrained, weights)
    model = GPT2LMHeadModel(apply_overrides(_CFG_LARGE, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="gpt2_large_lm")
    return model


@register_model(
    task="language-modeling",
    family="gpt2",
    model_type="gpt2",
    model_class=GPT2LMHeadModel,
    default_config=_CFG_XLARGE,
)
def gpt2_xlarge_lm(
    pretrained: bool | str = False,
    *,
    weights: GPT2XLargeLMWeights | None = None,
    **overrides: object,
) -> GPT2LMHeadModel:
    r"""Construct a GPT-2 XL (1.5B) model with the tied causal-LM head.

    Same trunk as :func:`gpt2_xlarge` (L=48, H=1600, A=25, ~1.5B parameters),
    wrapped with the tied LM head for pre-training and free-form generation.

    Parameters
    ----------
    pretrained : bool or str, default=False
        ``True`` loads the ``DEFAULT`` checkpoint of :class:`GPT2XLargeLMWeights`; a
        string names a specific tag.  ``False`` returns a randomly initialised
        model.
    weights : GPT2XLargeLMWeights, optional, keyword-only
        Explicit weights enum member; takes precedence over ``pretrained``.
    **overrides : object
        Optional :class:`GPT2Config` field overrides forwarded into the
        underlying config.

    Returns
    -------
    GPT2LMHeadModel
        GPT-2 XL trunk wrapped with the tied LM head.

    Notes
    -----
    Reference: Radford et al., *"Language Models are Unsupervised
    Multitask Learners"*, OpenAI Technical Report, 2019.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.gpt2 import gpt2_xlarge_lm
    >>> model = gpt2_xlarge_lm().eval()
    >>> input_ids = lucid.tensor([[15496, 11, 995, 13]])
    >>> out = model(input_ids)
    >>> out.logits.shape   # (1, 4, 50257)
    (1, 4, 50257)
    """
    entry = weights_mod.resolve_weights(GPT2XLargeLMWeights, pretrained, weights)
    model = GPT2LMHeadModel(apply_overrides(_CFG_XLARGE, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="gpt2_xlarge_lm")
    return model


# ── Sequence-classification head ──────────────────────────────────────────────


@register_model(
    task="sequence-classification",
    family="gpt2",
    model_type="gpt2",
    model_class=GPT2ForSequenceClassification,
    default_config=_CFG_SMALL,
)
def gpt2_small_cls(
    pretrained: bool | str = False, *, weights: GPT2SmallWeights | None = None, **overrides: object
) -> GPT2ForSequenceClassification:
    r"""Construct a GPT-2 small (124M) model with a sequence-classification head.

    Same trunk as :func:`gpt2_small` (L=12, H=768, A=12), augmented with a
    last-token linear classifier of output width ``config.num_labels``.
    The canonical recipe for fine-tuning GPT-2 on GLUE-style sentence
    classification tasks when decoder-style features are preferred to
    bidirectional ones.

    Parameters
    ----------
    pretrained : bool or str, default=False
        Encoder-weight selector.  ``False`` → fully random init; ``True``
        → loads the pretrained :func:`gpt2_small` decoder trunk
        (:attr:`GPT2SmallWeights.DEFAULT`) into the ``.transformer``
        submodule; a tag string selects a specific encoder checkpoint.
        **The classifier head is always randomly initialised** (fine-tuning
        starting point — no GLUE-fine-tuned head ships).
    weights : GPT2SmallWeights, optional, keyword-only
        Explicit encoder-weights enum member; takes precedence over
        ``pretrained``.
    **overrides : object
        Optional :class:`GPT2Config` field overrides forwarded into the
        underlying config.  Pass ``num_labels=N`` to set the number of
        classes (default 2).  Overrides that change the trunk shape are
        incompatible with loading pretrained encoder weights.

    Returns
    -------
    GPT2ForSequenceClassification
        GPT-2 small trunk wrapped with the last-token classifier head
        (encoder pretrained when requested; head random).

    Notes
    -----
    Reference: Radford et al., *"Language Models are Unsupervised
    Multitask Learners"*, OpenAI Technical Report, 2019.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.gpt2 import gpt2_small_cls
    >>> model = gpt2_small_cls(num_labels=3).eval()
    >>> input_ids = lucid.tensor([[15496, 11, 995, 13]])
    >>> out = model(input_ids)
    >>> out.logits.shape   # (1, num_labels=3)
    (1, 3)
    """
    entry = weights_mod.resolve_weights(GPT2SmallWeights, pretrained, weights)
    model = GPT2ForSequenceClassification(apply_overrides(_CFG_SMALL, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model.transformer, entry, name="gpt2_small")
    return model
