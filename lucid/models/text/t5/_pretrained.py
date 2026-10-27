"""Registry factories for T5 (Raffel et al., 2020).

The paper trains five sizes (§3.7) — Small, Base, Large, 3B and 11B —
and each is registered twice: as the bare trunk and as the text-to-text
model.  3B and 11B keep Large's depth and width and grow the
feed-forward and the attention instead, with heads of width 128.

=========  =======  ======  =====  =====  ======  ===============
factory    blocks   d_model  d_ff   heads  d_kv    parameters
=========  =======  ======  =====  =====  ======  ===============
small      6 + 6    512     2048   8      64      60 506 624
base       12 + 12  768     3072   12     64      222 903 552
large      24 + 24  1024    4096   16     64      737 668 096
3b         24 + 24  1024    16384  32     128     2 851 598 336
11b        24 + 24  1024    65536  128    128     11 307 321 344
=========  =======  ======  =====  =====  ======  ===============

Google released all five checkpoints; they have not been converted and
hosted for Lucid yet (``tools/convert_weights/t5.py`` converts them), so
``pretrained=True`` raises rather than return random weights.
"""

from lucid.models._registry import register_model
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained
from lucid.models.text.t5._config import T5Config
from lucid.models.text.t5._model import T5ForSeq2SeqLM, T5Model

__all__ = [
    "t5_small",
    "t5_base",
    "t5_large",
    "t5_3b",
    "t5_11b",
    "t5_small_seq2seq",
    "t5_base_seq2seq",
    "t5_large_seq2seq",
    "t5_3b_seq2seq",
    "t5_11b_seq2seq",
]

# §3.7: "Base" matches BERT-Base on each side; "Small", "Large", "3B" and
# "11B" scale it.  The defaults of T5Config are Small.
_CFG_SMALL = T5Config()
_CFG_BASE = T5Config(
    hidden_size=768,
    num_hidden_layers=12,
    num_decoder_layers=12,
    num_attention_heads=12,
    intermediate_size=3_072,
)
_CFG_LARGE = T5Config(
    hidden_size=1_024,
    num_hidden_layers=24,
    num_decoder_layers=24,
    num_attention_heads=16,
    intermediate_size=4_096,
)
_CFG_3B = T5Config(
    hidden_size=1_024,
    num_hidden_layers=24,
    num_decoder_layers=24,
    num_attention_heads=32,
    head_dim=128,
    intermediate_size=16_384,
)
_CFG_11B = T5Config(
    hidden_size=1_024,
    num_hidden_layers=24,
    num_decoder_layers=24,
    num_attention_heads=128,
    head_dim=128,
    intermediate_size=65_536,
)


def _trunk(name: str, cfg: T5Config, pretrained: bool, overrides: dict[str, object]) -> T5Model:
    if pretrained:
        reject_unavailable_pretrained(name)
    return T5Model(apply_overrides(cfg, overrides))


def _text_to_text(
    name: str, cfg: T5Config, pretrained: bool, overrides: dict[str, object]
) -> T5ForSeq2SeqLM:
    if pretrained:
        reject_unavailable_pretrained(name)
    return T5ForSeq2SeqLM(apply_overrides(cfg, overrides))


# The parameter counts are the architecture's own (a tied head adds none,
# so trunk and text-to-text model agree); the paper rounds them to 60M,
# 220M, 770M, 3B and 11B.  Each decorator spells its count out because the
# docs site reads it as a literal.

# ── Trunks ────────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="t5",
    model_type="t5",
    model_class=T5Model,
    default_config=_CFG_SMALL,
    params=60_506_624,
    summary="auto",
)
def t5_small(pretrained: bool = False, **overrides: object) -> T5Model:
    r"""T5-Small trunk — the 60M model of the paper's scaling study.

    Six blocks on each side, :math:`d_{\text{model}} = 512`,
    :math:`d_{ff} = 2048` and eight heads of width 64.

    Parameters
    ----------
    pretrained : bool, default=False
        Google's T5-Small checkpoint is not hosted for Lucid yet, so
        ``True`` raises ``NotImplementedError`` instead of returning an
        untrained model.
    **overrides : object
        :class:`T5Config` fields to change from the Small defaults.

    Returns
    -------
    T5Model
        Encoder-decoder trunk with no language-modelling head.

    Notes
    -----
    Reference: Raffel et al., *"Exploring the Limits of Transfer Learning
    with a Unified Text-to-Text Transformer"*, JMLR 21(140), 2020
    (arXiv:1910.10683), §3.7.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.t5 import t5_small
    >>> model = t5_small().eval()
    >>> out = model(lucid.tensor([[13, 5, 42, 1]]), lucid.tensor([[0, 7]]))
    >>> out.last_hidden_state.shape
    (1, 2, 512)
    """
    return _trunk("t5_small", _CFG_SMALL, pretrained, overrides)


@register_model(
    task="base",
    family="t5",
    model_type="t5",
    model_class=T5Model,
    default_config=_CFG_BASE,
    params=222_903_552,
    summary="auto",
)
def t5_base(pretrained: bool = False, **overrides: object) -> T5Model:
    r"""T5-Base trunk — the size every ablation of §3.1–§3.6 runs at.

    BERT-Base on each side: twelve blocks, :math:`d_{\text{model}} = 768`,
    :math:`d_{ff} = 3072`, twelve heads of width 64.

    Parameters
    ----------
    pretrained : bool, default=False
        ``True`` raises: the released T5-Base weights have not been
        converted for Lucid.
    **overrides : object
        :class:`T5Config` fields to change from the Base defaults.

    Returns
    -------
    T5Model
        The 220M-parameter trunk, about twice BERT-Base for its two stacks.

    Notes
    -----
    Reference: Raffel et al., JMLR 2020 (arXiv:1910.10683), §3.1.1 and
    §3.7.  The baseline of the paper's systematic study pre-trains this
    model for :math:`2^{19}` steps on C4.

    Examples
    --------
    >>> from lucid.models.text.t5 import t5_base
    >>> t5_base().config.num_hidden_layers  # doctest: +SKIP
    12
    """
    return _trunk("t5_base", _CFG_BASE, pretrained, overrides)


@register_model(
    task="base",
    family="t5",
    model_type="t5",
    model_class=T5Model,
    default_config=_CFG_LARGE,
    params=737_668_096,
    summary="auto",
)
def t5_large(pretrained: bool = False, **overrides: object) -> T5Model:
    r"""T5-Large trunk — BERT-Large on each side.

    Twenty-four blocks per stack, :math:`d_{\text{model}} = 1024`,
    :math:`d_{ff} = 4096`, sixteen heads of width 64.

    Parameters
    ----------
    pretrained : bool, default=False
        ``True`` raises until the released T5-Large checkpoint is hosted
        for Lucid.
    **overrides : object
        Field overrides applied on top of the Large configuration.

    Returns
    -------
    T5Model
        Trunk of 737.7M parameters (the paper quotes about 770M).

    Notes
    -----
    Reference: Raffel et al., JMLR 2020 (arXiv:1910.10683), §3.7.

    Examples
    --------
    >>> from lucid.models.text.t5 import t5_large
    >>> t5_large().config.hidden_size  # doctest: +SKIP
    1024
    """
    return _trunk("t5_large", _CFG_LARGE, pretrained, overrides)


@register_model(
    task="base",
    family="t5",
    model_type="t5",
    model_class=T5Model,
    default_config=_CFG_3B,
    params=2_851_598_336,
    summary="auto",
)
def t5_3b(pretrained: bool = False, **overrides: object) -> T5Model:
    r"""T5-3B trunk — Large's depth and width, a wider inside.

    The feed-forward grows to :math:`d_{ff} = 16384` and attention to 32
    heads of width 128, so each block attends at 4096 channels while the
    residual stream stays at 1024.

    Parameters
    ----------
    pretrained : bool, default=False
        No converted 3B checkpoint exists for Lucid yet; ``True`` raises.
    **overrides : object
        Changes to the 3B configuration.

    Returns
    -------
    T5Model
        2.85B-parameter trunk.

    Notes
    -----
    Reference: Raffel et al., JMLR 2020 (arXiv:1910.10683), §3.7.
    Building it allocates about 11 GB of float32 weights.

    Examples
    --------
    >>> from lucid.models.text.t5 import t5_3b
    >>> t5_3b().config.inner_dim  # doctest: +SKIP
    4096
    """
    return _trunk("t5_3b", _CFG_3B, pretrained, overrides)


@register_model(
    task="base",
    family="t5",
    model_type="t5",
    model_class=T5Model,
    default_config=_CFG_11B,
    params=11_307_321_344,
    summary="auto",
)
def t5_11b(pretrained: bool = False, **overrides: object) -> T5Model:
    r"""T5-11B trunk — the paper's largest model.

    Large's depth and :math:`d_{\text{model}} = 1024`, with
    :math:`d_{ff} = 65536` and 128 heads of width 128: attention runs at
    sixteen times the residual width.

    Parameters
    ----------
    pretrained : bool, default=False
        The 11B weights are not hosted for Lucid; ``True`` raises.
    **overrides : object
        Changes to the 11B configuration.

    Returns
    -------
    T5Model
        11.3B-parameter trunk.

    Notes
    -----
    Reference: Raffel et al., JMLR 2020 (arXiv:1910.10683), §3.7 and
    Table 14, where the fine-tuned 11B model set the state of the art on
    GLUE, SuperGLUE, SQuAD and CNN/Daily Mail.  Building it allocates
    about 45 GB of float32 weights.

    Examples
    --------
    >>> from lucid.models.text.t5 import t5_11b
    >>> t5_11b().config.num_attention_heads  # doctest: +SKIP
    128
    """
    return _trunk("t5_11b", _CFG_11B, pretrained, overrides)


# ── Text-to-text models ──────────────────────────────────────────────────────


@register_model(
    task="language-modeling",
    family="t5",
    model_type="t5",
    model_class=T5ForSeq2SeqLM,
    default_config=_CFG_SMALL,
    params=60_506_624,
    summary="auto",
)
def t5_small_seq2seq(pretrained: bool = False, **overrides: object) -> T5ForSeq2SeqLM:
    r""":func:`t5_small` with the tied text-to-text head.

    Parameters
    ----------
    pretrained : bool, default=False
        ``True`` raises until the T5-Small checkpoint is hosted for Lucid.
    **overrides : object
        :class:`T5Config` fields to change from the Small defaults.

    Returns
    -------
    T5ForSeq2SeqLM
        The model T5 trains and fine-tunes every task with.

    Notes
    -----
    Reference: Raffel et al., JMLR 2020 (arXiv:1910.10683), §2.4 and §3.7.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.t5 import t5_small_seq2seq
    >>> model = t5_small_seq2seq().eval()
    >>> source = lucid.tensor([[13, 5, 42, 1]])
    >>> model(source, labels=lucid.tensor([[7, 9, 1]])).logits.shape
    (1, 3, 32128)
    >>> model.generate(source, max_length=4).shape[0]
    1
    """
    return _text_to_text("t5_small_seq2seq", _CFG_SMALL, pretrained, overrides)


@register_model(
    task="language-modeling",
    family="t5",
    model_type="t5",
    model_class=T5ForSeq2SeqLM,
    default_config=_CFG_BASE,
    params=222_903_552,
    summary="auto",
)
def t5_base_seq2seq(pretrained: bool = False, **overrides: object) -> T5ForSeq2SeqLM:
    r""":func:`t5_base` with the tied text-to-text head.

    Parameters
    ----------
    pretrained : bool, default=False
        The T5-Base text-to-text weights are not hosted for Lucid, so
        ``True`` raises.
    **overrides : object
        Field overrides applied on top of the Base configuration.

    Returns
    -------
    T5ForSeq2SeqLM
        The baseline model of the paper's systematic study.

    Notes
    -----
    Reference: Raffel et al., JMLR 2020 (arXiv:1910.10683), §3.1.

    Examples
    --------
    >>> from lucid.models.text.t5 import t5_base_seq2seq
    >>> t5_base_seq2seq().lm_head.out_features  # doctest: +SKIP
    32128
    """
    return _text_to_text("t5_base_seq2seq", _CFG_BASE, pretrained, overrides)


@register_model(
    task="language-modeling",
    family="t5",
    model_type="t5",
    model_class=T5ForSeq2SeqLM,
    default_config=_CFG_LARGE,
    params=737_668_096,
    summary="auto",
)
def t5_large_seq2seq(pretrained: bool = False, **overrides: object) -> T5ForSeq2SeqLM:
    r""":func:`t5_large` with the tied text-to-text head.

    Parameters
    ----------
    pretrained : bool, default=False
        No converted T5-Large checkpoint exists for Lucid; ``True`` raises.
    **overrides : object
        Changes to the Large configuration.

    Returns
    -------
    T5ForSeq2SeqLM
        Large trunk and head, 737.7M parameters.

    Notes
    -----
    Reference: Raffel et al., JMLR 2020 (arXiv:1910.10683), §3.7.

    Examples
    --------
    >>> from lucid.models.text.t5 import t5_large_seq2seq
    >>> t5_large_seq2seq().config.num_decoder_layers  # doctest: +SKIP
    24
    """
    return _text_to_text("t5_large_seq2seq", _CFG_LARGE, pretrained, overrides)


@register_model(
    task="language-modeling",
    family="t5",
    model_type="t5",
    model_class=T5ForSeq2SeqLM,
    default_config=_CFG_3B,
    params=2_851_598_336,
    summary="auto",
)
def t5_3b_seq2seq(pretrained: bool = False, **overrides: object) -> T5ForSeq2SeqLM:
    r""":func:`t5_3b` with the tied text-to-text head.

    Parameters
    ----------
    pretrained : bool, default=False
        ``True`` raises until the 3B weights are converted for Lucid.
    **overrides : object
        :class:`T5Config` fields to change from the 3B defaults.

    Returns
    -------
    T5ForSeq2SeqLM
        3B trunk and head, 2.85B parameters.

    Notes
    -----
    Reference: Raffel et al., JMLR 2020 (arXiv:1910.10683), §3.7.

    Examples
    --------
    >>> from lucid.models.text.t5 import t5_3b_seq2seq
    >>> t5_3b_seq2seq().config.intermediate_size  # doctest: +SKIP
    16384
    """
    return _text_to_text("t5_3b_seq2seq", _CFG_3B, pretrained, overrides)


@register_model(
    task="language-modeling",
    family="t5",
    model_type="t5",
    model_class=T5ForSeq2SeqLM,
    default_config=_CFG_11B,
    params=11_307_321_344,
    summary="auto",
)
def t5_11b_seq2seq(pretrained: bool = False, **overrides: object) -> T5ForSeq2SeqLM:
    r""":func:`t5_11b` with the tied text-to-text head.

    Parameters
    ----------
    pretrained : bool, default=False
        The 11B weights are not hosted for Lucid, so ``True`` raises.
    **overrides : object
        Changes to the 11B configuration.

    Returns
    -------
    T5ForSeq2SeqLM
        11B trunk and head, 11.3B parameters — the model behind the
        paper's Table 14 results.

    Notes
    -----
    Reference: Raffel et al., JMLR 2020 (arXiv:1910.10683), §3.7.

    Examples
    --------
    >>> from lucid.models.text.t5 import t5_11b_seq2seq
    >>> t5_11b_seq2seq().config.head_dim  # doctest: +SKIP
    128
    """
    return _text_to_text("t5_11b_seq2seq", _CFG_11B, pretrained, overrides)
