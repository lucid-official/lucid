"""Registry factories for GPT-1 (Radford et al., 2018).

The original paper specifies a single architecture: ``L=12, H=768, A=12``.
That's the only published variant; we therefore expose just one nominal
factory ``gpt`` (plus the ``_lm`` and ``_cls`` task heads).  Test code
that needs a small variant should override config fields via
``create_model("gpt", hidden_size=..., num_hidden_layers=..., ...)``.
"""

import lucid.weights as weights_mod
from lucid.models._registry import register_model
from lucid.models.text.gpt._config import GPTConfig
from lucid.models.text.gpt._model import GPTForSequenceClassification, GPTLMHeadModel, GPTModel
from lucid.models.text.gpt._weights import GPTLMWeights, GPTWeights
from lucid.models._utils._common import apply_overrides

_CFG_BASE = GPTConfig()  # Radford et al. defaults — the paper's only size


# ── Backbone ──────────────────────────────────────────────────────────────────


@register_model(
    task="base", family="gpt", model_type="gpt", model_class=GPTModel, default_config=_CFG_BASE
)
def gpt(
    pretrained: bool | str = False, *, weights: GPTWeights | None = None, **overrides: object
) -> GPTModel:
    r"""Construct a GPT-1 decoder-only Transformer trunk.

    The original generative pre-training transformer from Radford, Narasimhan,
    Salimans, and Sutskever, 2018: :math:`L=12` transformer blocks,
    :math:`H=768` hidden, :math:`A=12` attention heads, intermediate width
    3072, learned absolute position embeddings, and a 40,478-piece BPE
    vocabulary trained on BookCorpus.  Roughly 117M parameters — the seed
    of the modern decoder-only LLM family.

    Parameters
    ----------
    pretrained : bool or str, default=False
        ``True`` loads the ``DEFAULT`` checkpoint of :class:`GPTWeights`; a string
        names a specific tag.  ``False`` returns a randomly initialised model.
    weights : GPTWeights, optional, keyword-only
        Explicit weights enum member; takes precedence over ``pretrained``.
    **overrides : object
        Optional :class:`GPTConfig` field overrides (e.g. ``vocab_size=...``,
        ``hidden_size=...``, ``num_hidden_layers=...``) forwarded into the
        underlying config.

    Returns
    -------
    GPTModel
        Decoder trunk configured with GPT-1 defaults plus any overrides.

    Notes
    -----
    Reference: Radford, Narasimhan, Salimans, and Sutskever, *"Improving
    Language Understanding by Generative Pre-Training"*, OpenAI Technical
    Report, 2018.

    Causal language-model factorisation:

    .. math::

        \mathcal{L}_{\mathrm{LM}}(\theta)
            = \sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t}),

    enforced architecturally by a lower-triangular mask inside scaled
    dot-product attention so position :math:`t` only attends to
    :math:`\{1, \dots, t\}`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.gpt import gpt
    >>> model = gpt().eval()
    >>> input_ids = lucid.tensor([[101, 7592, 2088, 102]])
    >>> out = model(input_ids)
    >>> out.last_hidden_state.shape   # (B=1, T=4, H=768)
    (1, 4, 768)
    """
    entry = weights_mod.resolve_weights(GPTWeights, pretrained, weights)
    model = GPTModel(apply_overrides(_CFG_BASE, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="gpt")
    return model


# ── Causal-LM head (CausalLMMixin host) ─────────────────────────────────────


@register_model(
    task="language-modeling",
    family="gpt",
    model_type="gpt",
    model_class=GPTLMHeadModel,
    default_config=_CFG_BASE,
)
def gpt_lm(
    pretrained: bool | str = False, *, weights: GPTLMWeights | None = None, **overrides: object
) -> GPTLMHeadModel:
    r"""Construct a GPT-1 model with the tied causal-LM head.

    Same trunk as :func:`gpt` (L=12, H=768, A=12, ~117M parameters), wrapped
    with a tied LM head that reuses the input token-embedding matrix as the
    output projection.  This is the pre-training entry point and the host
    of :meth:`lucid.models._mixins.CausalLMMixin.generate` for free-form
    autoregressive sampling.

    Parameters
    ----------
    pretrained : bool or str, default=False
        ``True`` loads the ``DEFAULT`` checkpoint of :class:`GPTLMWeights`; a string
        names a specific tag.  ``False`` returns a randomly initialised model.
    weights : GPTLMWeights, optional, keyword-only
        Explicit weights enum member; takes precedence over ``pretrained``.
    **overrides : object
        Optional :class:`GPTConfig` field overrides forwarded into the
        underlying config.

    Returns
    -------
    GPTLMHeadModel
        GPT-1 trunk wrapped with the tied LM head.

    Notes
    -----
    Reference: Radford et al., *"Improving Language Understanding by
    Generative Pre-Training"*, OpenAI Technical Report, 2018.

    When ``labels`` is supplied the head computes the standard
    next-token-shifted causal-LM loss

    .. math::

        \mathcal{L}_{\mathrm{CLM}}
            = -\frac{1}{B(T-1)} \sum_{b,t}
              \log p_\theta(y_{b, t+1} \mid x_{b, \le t}),

    with positions labelled ``-100`` excluded.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.gpt import gpt_lm
    >>> model = gpt_lm().eval()
    >>> input_ids = lucid.tensor([[101, 7592, 2088, 102]])
    >>> out = model(input_ids)
    >>> out.logits.shape   # (B=1, T=4, V=40478)
    (1, 4, 40478)
    """
    entry = weights_mod.resolve_weights(GPTLMWeights, pretrained, weights)
    model = GPTLMHeadModel(apply_overrides(_CFG_BASE, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="gpt_lm")
    return model


# ── Sequence-classification head ──────────────────────────────────────────────


@register_model(
    task="sequence-classification",
    family="gpt",
    model_type="gpt",
    model_class=GPTForSequenceClassification,
    default_config=_CFG_BASE,
)
def gpt_cls(
    pretrained: bool | str = False, *, weights: GPTWeights | None = None, **overrides: object
) -> GPTForSequenceClassification:
    r"""Construct a GPT-1 model with a last-token sequence-classification head.

    Same trunk as :func:`gpt` (L=12, H=768, A=12), augmented with a linear
    classifier on the **last non-pad token's** hidden state — the canonical
    decoder-style pooling strategy.  Use for GLUE-style classification when
    you specifically want autoregressive features rather than bidirectional
    BERT-style ones.

    Parameters
    ----------
    pretrained : bool or str, default=False
        Encoder-weight selector.  ``False`` → fully random init; ``True``
        → loads the pretrained :func:`gpt` decoder trunk
        (:attr:`GPTWeights.DEFAULT`) into the ``.transformer`` submodule;
        a tag string selects a specific encoder checkpoint.  **The
        classifier head is always randomly initialised** (fine-tuning
        starting point — no GLUE-fine-tuned head ships).
    weights : GPTWeights, optional, keyword-only
        Explicit encoder-weights enum member; takes precedence over
        ``pretrained``.
    **overrides : object
        Optional :class:`GPTConfig` field overrides forwarded into the
        underlying config.  Pass ``num_labels=N`` to set the number of
        output classes (default 2).  Overrides that change the trunk shape
        are incompatible with loading pretrained encoder weights.

    Returns
    -------
    GPTForSequenceClassification
        GPT-1 trunk wrapped with the last-token classifier head (encoder
        pretrained when requested; head random).

    Notes
    -----
    Reference: Radford et al., *"Improving Language Understanding by
    Generative Pre-Training"*, OpenAI Technical Report, 2018, §3.3.

    Pooling picks index :math:`t^\star = \sum_t \mathbf{1}[\mathrm{mask}_t]
    - 1` (i.e. the rightmost real token) per row, then applies dropout +
    linear:

    .. math::

        z = W_{\mathrm{cls}}\, \mathrm{Dropout}(h_{t^\star}) + b_{\mathrm{cls}}.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.gpt import gpt_cls
    >>> model = gpt_cls(num_labels=3).eval()
    >>> input_ids = lucid.tensor([[101, 7592, 102]])
    >>> out = model(input_ids)
    >>> out.logits.shape   # (B=1, num_labels=3)
    (1, 3)
    """
    entry = weights_mod.resolve_weights(GPTWeights, pretrained, weights)
    model = GPTForSequenceClassification(apply_overrides(_CFG_BASE, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model.transformer, entry, name="gpt")
    return model
