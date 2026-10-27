"""T5 (Raffel et al., 2020) — encoder-decoder with relative position bias.

The paper's architecture: a stack of pre-norm encoder blocks and a stack
of pre-norm decoder blocks reading one token table, which the language
modelling head shares as well.  Attention adds a learned per-head scalar
looked up by bucketed relative distance, and is not scaled by
1/sqrt(d_kv).

Task heads
----------
    T5Model          — bare encoder-decoder trunk
    T5ForSeq2SeqLM   — the text-to-text head every T5 task trains

T5 casts classification and regression as decoding the label's text, so
the paper has no classification heads and this family adds none.

Generation
----------
``T5ForSeq2SeqLM.generate`` encodes once and unrolls the decoder from
``decoder_start_token_id`` through the loop the encoder-decoder families
share (:mod:`lucid.models.text._common._seq2seq`).
"""

import functools
import math
from dataclasses import dataclass
from typing import ClassVar, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._output import ModelOutput, Seq2SeqLMOutput
from lucid.models._sampling import _SamplingParams
from lucid.models._tasks import LanguageModelingModel
from lucid.models._utils._text import extended_attention_mask, text_activation
from lucid.models.text._common._seq2seq import seq2seq_generate
from lucid.models.text.t5._config import T5Config
from lucid.utils.cache import EncoderDecoderCache

# ─────────────────────────────────────────────────────────────────────────────
# Relative position buckets
# ─────────────────────────────────────────────────────────────────────────────


def _bucket(
    relative_position: int, *, bidirectional: bool, num_buckets: int, max_distance: int
) -> int:
    """T5's bucket for one key-minus-query distance.

    Exact below half the buckets of a side, log-spaced up to
    ``max_distance``, saturated beyond.  The encoder gives each sign half
    the buckets; the causal decoder spends all of them on keys at or
    before the query.  Plain integer and float arithmetic: the table is
    built on the host, once per shape, so no ``log(0)`` reaches a device
    and no float32 rounding moves an exact power-of-two boundary.
    """
    offset = 0
    if bidirectional:
        num_buckets //= 2
        if relative_position > 0:
            offset = num_buckets
        distance = abs(relative_position)
    else:
        distance = max(-relative_position, 0)
    max_exact = num_buckets // 2
    if distance < max_exact:
        return offset + distance
    scaled = math.log(distance / max_exact) / math.log(max_distance / max_exact)
    return offset + min(max_exact + int(scaled * (num_buckets - max_exact)), num_buckets - 1)


@functools.lru_cache(maxsize=64)
def _bucket_table(
    lowest: int, highest: int, bidirectional: bool, num_buckets: int, max_distance: int
) -> tuple[int, ...]:
    """Buckets of the relative positions ``lowest, …, highest`` in order."""
    return tuple(
        _bucket(r, bidirectional=bidirectional, num_buckets=num_buckets, max_distance=max_distance)
        for r in range(lowest, highest + 1)
    )


def _causal_bias(query_length: int, key_length: int, past_length: int, like: Tensor) -> Tensor:
    """``(1, 1, T_q, T_k)`` additive mask hiding keys after each query.

    Query ``i`` sits at absolute position ``past_length + i``, so with a
    cache the mask covers the whole history the new tokens attend over.
    Built on ``like``'s device and in its dtype: a float32 mask added to a
    float16 bias would widen every attention score after it to float32.
    """
    device = like.device.type
    queries = lucid.arange(query_length, device=device).long() + past_length
    keys = lucid.arange(key_length, device=device).long()
    future = keys.reshape(1, key_length) > queries.reshape(query_length, 1)
    return (future.to(like.dtype) * -1e4).reshape(1, 1, query_length, key_length)


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────


class _T5Attention(nn.Module):
    """Multi-head attention without logit scaling or projection biases.

    Self-attention when ``memory`` is ``None``, cross-attention over the
    encoder output otherwise.  With an :class:`EncoderDecoderCache` the
    self-attention keys and values grow by the new tokens each step, and
    the cross-attention ones are projected from ``memory`` once and reused.
    """

    def __init__(self, config: T5Config) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.dropout_p = config.attention_dropout
        width = config.inner_dim
        self.q = nn.Linear(config.hidden_size, width, bias=False)
        self.k = nn.Linear(config.hidden_size, width, bias=False)
        self.v = nn.Linear(config.hidden_size, width, bias=False)
        self.o = nn.Linear(width, config.hidden_size, bias=False)

    def _heads(self, x: Tensor) -> Tensor:
        """``(B, T, H * D)`` to ``(B, H, T, D)``."""
        batch, length = int(x.shape[0]), int(x.shape[1])
        return x.reshape(batch, length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def _memory_heads(
        self, memory: Tensor, cache: EncoderDecoderCache | None, layer_idx: int
    ) -> tuple[Tensor, Tensor]:
        """Cross-attention keys and values, projected once per cache."""
        if cache is not None and cache.is_updated.get(layer_idx, False):
            return cache.cross_attention_cache[layer_idx]
        k, v = self._heads(self.k(memory)), self._heads(self.v(memory))
        if cache is not None:
            k, v = cache.cross_attention_cache.update(k, v, layer_idx)
            cache.is_updated[layer_idx] = True
        return k, v

    @override
    def forward(
        self,
        hidden: Tensor,
        bias: Tensor | None,
        *,
        memory: Tensor | None = None,
        cache: EncoderDecoderCache | None = None,
        layer_idx: int = 0,
    ) -> Tensor:
        q = self._heads(self.q(hidden))
        if memory is None:
            k, v = self._heads(self.k(hidden)), self._heads(self.v(hidden))
            if cache is not None:
                k, v = cache.update(k, v, layer_idx)
        else:
            k, v = self._memory_heads(memory, cache, layer_idx)
        # scale=1: the 1/sqrt(d_kv) lives in the query projection's init.
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, dropout_p=self.dropout_p if self.training else 0.0, scale=1.0
        )
        batch, heads, length, width = (int(d) for d in out.shape)
        return self.o(out.permute(0, 2, 1, 3).reshape(batch, length, heads * width))


class _T5FeedForward(nn.Module):
    """``wo(dropout(act(wi(x))))`` — two bias-free dense layers (§3.1.1)."""

    def __init__(self, config: T5Config) -> None:
        super().__init__()
        self.activation = config.hidden_act
        self.wi = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(p=config.hidden_dropout)

    @override
    def forward(self, hidden: Tensor) -> Tensor:
        return self.wo(self.dropout(text_activation(self.activation, self.wi(hidden))))


class _T5Block(nn.Module):
    """Pre-norm residual sub-layers: self-attention, cross-attention
    (decoder only) and the feed-forward network."""

    def __init__(self, config: T5Config, *, is_decoder: bool) -> None:
        super().__init__()
        width, eps = config.hidden_size, config.layer_norm_eps
        self.self_attention_norm = nn.RMSNorm(width, eps=eps)
        self.self_attention = _T5Attention(config)
        self.cross_attention_norm = nn.RMSNorm(width, eps=eps) if is_decoder else None
        self.cross_attention = _T5Attention(config) if is_decoder else None
        self.feed_forward_norm = nn.RMSNorm(width, eps=eps)
        self.feed_forward = _T5FeedForward(config)
        self.dropout = nn.Dropout(p=config.hidden_dropout)

    @override
    def forward(
        self,
        hidden: Tensor,
        bias: Tensor,
        *,
        memory: Tensor | None = None,
        cross_bias: Tensor | None = None,
        cache: EncoderDecoderCache | None = None,
        layer_idx: int = 0,
    ) -> Tensor:
        attended = self.self_attention(
            self.self_attention_norm(hidden), bias, cache=cache, layer_idx=layer_idx
        )
        hidden = hidden + self.dropout(attended)
        if self.cross_attention is not None and self.cross_attention_norm is not None:
            if memory is None:
                raise ValueError("a T5 decoder block needs the encoder output")
            read = self.cross_attention(
                self.cross_attention_norm(hidden),
                cross_bias,
                memory=memory,
                cache=cache,
                layer_idx=layer_idx,
            )
            hidden = hidden + self.dropout(read)
        return hidden + self.dropout(self.feed_forward(self.feed_forward_norm(hidden)))


class _T5Stack(nn.Module):
    """One side of the trunk: the blocks, their shared position bias and
    the closing norm.  Embeddings come in from the trunk's shared table."""

    def __init__(self, config: T5Config, *, is_decoder: bool) -> None:
        super().__init__()
        self.is_decoder = is_decoder
        self.num_buckets = config.relative_attention_num_buckets
        self.max_distance = config.relative_attention_max_distance
        # One table per stack, read by every block: the released models keep
        # it in block 0 and hand its output down, which is the same thing.
        self.relative_attention_bias = nn.Embedding(self.num_buckets, config.num_attention_heads)
        depth = config.num_decoder_layers if is_decoder else config.num_hidden_layers
        self.blocks = nn.ModuleList([_T5Block(config, is_decoder=is_decoder) for _ in range(depth)])
        self.final_layer_norm = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout)

    def position_bias(self, query_length: int, key_length: int, past_length: int) -> Tensor:
        """``(1, H, T_q, T_k)`` learned bias for queries at ``past_length + i``.

        Parameters
        ----------
        query_length : int
            New positions being attended from.
        key_length : int
            Positions attended over — ``past_length + query_length`` for
            self-attention.
        past_length : int
            Positions already in the cache before the new ones.

        Returns
        -------
        Tensor
            The bias every block of this stack adds to its self-attention.
        """
        device = self.relative_attention_bias.weight.device.type
        lowest = -(past_length + query_length - 1)
        table = _bucket_table(
            lowest, key_length - 1, not self.is_decoder, self.num_buckets, self.max_distance
        )
        buckets = lucid.tensor(list(table), dtype=lucid.int64, device=device)
        queries = lucid.arange(query_length, device=device).long() + past_length
        keys = lucid.arange(key_length, device=device).long()
        # Key minus query, offset so the most negative distance indexes 0.
        index = keys.reshape(1, key_length) - queries.reshape(query_length, 1) - lowest
        ids = buckets.index_select(0, index.reshape(-1)).reshape(query_length, key_length)
        return self.relative_attention_bias(ids).permute(2, 0, 1).unsqueeze(0)

    @override
    def forward(
        self,
        embeddings: Tensor,
        attention_mask: Tensor | None = None,
        *,
        memory: Tensor | None = None,
        memory_mask: Tensor | None = None,
        cache: EncoderDecoderCache | None = None,
    ) -> Tensor:
        batch, length = int(embeddings.shape[0]), int(embeddings.shape[1])
        past = cache.get_seq_length() if cache is not None else 0
        keys = past + length
        # Every bias in the activations' dtype — the padding masks come back
        # float32 whatever the model computes in.
        bias = self.position_bias(length, keys, past).to(embeddings.dtype)
        if self.is_decoder:
            bias = bias + _causal_bias(length, keys, past, bias)
        padding = extended_attention_mask(attention_mask, (batch, length), keys)
        if padding is not None:
            bias = bias + padding.to(bias.dtype)
        cross_bias = None
        if memory is not None:
            cross_bias = extended_attention_mask(memory_mask, (batch, int(memory.shape[1])))
            if cross_bias is not None:
                cross_bias = cross_bias.to(memory.dtype)

        hidden = self.dropout(embeddings)
        for layer_idx, block in enumerate(self.blocks):
            hidden = block(
                hidden, bias, memory=memory, cross_bias=cross_bias, cache=cache, layer_idx=layer_idx
            )
        return self.dropout(self.final_layer_norm(hidden))


def _init_t5_weights(model: nn.Module, config: T5Config) -> None:
    """Mesh TensorFlow's initialisation, which the unscaled logits assume.

    Every projection is drawn at the inverse square root of its fan-in;
    the query's counts ``d_kv`` once more, which is where the 1/sqrt(d_kv)
    a scaled attention would apply went.
    """
    factor, width = config.initializer_factor, config.hidden_size
    for module in model.modules():
        if isinstance(module, _T5Attention):
            nn.init.normal_(module.q.weight, std=factor * (width * config.head_dim) ** -0.5)
            nn.init.normal_(module.k.weight, std=factor * width**-0.5)
            nn.init.normal_(module.v.weight, std=factor * width**-0.5)
            nn.init.normal_(module.o.weight, std=factor * config.inner_dim**-0.5)
        elif isinstance(module, _T5FeedForward):
            nn.init.normal_(module.wi.weight, std=factor * width**-0.5)
            nn.init.normal_(module.wo.weight, std=factor * config.intermediate_size**-0.5)
        elif isinstance(module, _T5Stack):
            nn.init.normal_(module.relative_attention_bias.weight, std=factor * width**-0.5)
        elif isinstance(module, nn.RMSNorm) and module.weight is not None:
            nn.init.constant_(module.weight, factor)


# ─────────────────────────────────────────────────────────────────────────────
# Bare encoder-decoder trunk
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class T5ModelOutput(ModelOutput):
    r"""Output of the bare T5 trunk (no language-modelling head).

    Attributes
    ----------
    last_hidden_state : Tensor
        Decoder output after the final norm, ``(B, T_dec, d_model)``.
    encoder_last_hidden_state : Tensor
        Encoder output after the final norm, ``(B, T_src, d_model)`` —
        what every decoder block cross-attends to.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.t5._model import T5ModelOutput
    >>> out = T5ModelOutput(
    ...     last_hidden_state=lucid.zeros(2, 7, 512),
    ...     encoder_last_hidden_state=lucid.zeros(2, 20, 512),
    ... )
    >>> out.last_hidden_state.shape, out.encoder_last_hidden_state.shape
    ((2, 7, 512), (2, 20, 512))

    These are activations of width ``d_model``; :class:`T5ForSeq2SeqLM`
    is the model that turns the decoder side into vocabulary logits.
    """

    last_hidden_state: Tensor
    encoder_last_hidden_state: Tensor


class T5Model(PretrainedModel):
    r"""Bare T5 encoder-decoder trunk.

    Implements the architecture of Raffel, Shazeer, Roberts, Lee, Narang,
    Matena, Zhou, Li and Liu, 2020.  One token table embeds both the
    source and the target.  The encoder stack reads the source with
    bidirectional attention; the decoder stack reads the target causally
    and cross-attends to the encoder output.  Neither side adds a
    position to its tokens: each stack's blocks add one learned
    per-head bias, indexed by bucketed relative distance, to their
    self-attention logits.

    Parameters
    ----------
    config : T5Config
        Vocabulary, width, depth on each side, head count and width,
        relative-position bucketing, dropout and the init scale.

    Attributes
    ----------
    shared : nn.Embedding
        The ``(vocab_size, d_model)`` table both stacks embed with.
    encoder : nn.Module
        Bidirectional stack of ``num_hidden_layers`` blocks.
    decoder : nn.Module
        Causal stack of ``num_decoder_layers`` blocks with cross-attention.

    Notes
    -----
    Reference: Raffel et al., *"Exploring the Limits of Transfer Learning
    with a Unified Text-to-Text Transformer"*, JMLR 21(140), 2020
    (arXiv:1910.10683), §2.1 and §3.1.

    Head :math:`h` of every block scores query :math:`i` against key
    :math:`j` as

    .. math::

        A_{ij} = q_i^{\top} k_j + b_{h,\,\beta(j - i)},

    unscaled, and every sub-layer :math:`f` updates the stream as
    :math:`x \leftarrow x + f\bigl(x / \mathrm{RMS}(x) \odot \gamma\bigr)`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.t5 import T5Config, T5Model
    >>> cfg = T5Config(vocab_size=100, hidden_size=32, num_hidden_layers=2,
    ...                num_decoder_layers=2, num_attention_heads=4,
    ...                head_dim=8, intermediate_size=64)
    >>> model = T5Model(cfg).eval()
    >>> src = lucid.tensor([[13, 5, 42, 1]])
    >>> tgt = lucid.tensor([[0, 7, 9]])
    >>> out = model(src, decoder_input_ids=tgt)
    >>> out.last_hidden_state.shape, out.encoder_last_hidden_state.shape
    ((1, 3, 32), (1, 4, 32))
    """

    config_class: ClassVar[type[T5Config]] = T5Config
    base_model_prefix: ClassVar[str] = "t5"

    def __init__(self, config: T5Config) -> None:
        """Initialise the trunk. See the class docstring for parameters."""
        super().__init__(config)
        self.config: T5Config = config
        self.shared = nn.Embedding(config.vocab_size, config.hidden_size)
        self.encoder = _T5Stack(config, is_decoder=False)
        self.decoder = _T5Stack(config, is_decoder=True)
        nn.init.normal_(self.shared.weight, std=config.initializer_factor)
        _init_t5_weights(self, config)

    @override
    def get_input_embeddings(self) -> nn.Module:
        return self.shared

    @override
    def set_input_embeddings(self, value: nn.Module) -> None:
        if not isinstance(value, nn.Embedding):
            raise TypeError(
                f"T5Model input embeddings must be nn.Embedding, got {type(value).__name__}"
            )
        self.shared = value

    def encode(self, src_ids: Tensor, src_attention_mask: Tensor | None = None) -> Tensor:
        """Run the encoder alone.

        Parameters
        ----------
        src_ids : Tensor
            ``(B, S)`` source token ids.
        src_attention_mask : Tensor or None, optional
            ``(B, S)`` mask, 1 for a token and 0 for padding.

        Returns
        -------
        Tensor
            ``(B, S, d_model)`` encoder output.
        """
        return self.encoder(self.shared(src_ids), src_attention_mask)

    def decode(
        self,
        tgt_ids: Tensor,
        memory: Tensor,
        *,
        tgt_attention_mask: Tensor | None = None,
        memory_attention_mask: Tensor | None = None,
        past_key_value: EncoderDecoderCache | None = None,
        use_cache: bool = False,
    ) -> Tensor:
        """Run the decoder against an already-computed encoder output.

        Parameters
        ----------
        tgt_ids : Tensor
            ``(B, T)`` target ids to decode now — the whole prefix, or
            only the new ones when a cache holds the rest.
        memory : Tensor
            ``(B, S, d_model)`` from :meth:`encode`.
        tgt_attention_mask : Tensor or None, optional, keyword-only
            Target padding mask, as wide as the cached history plus ``T``.
        memory_attention_mask : Tensor or None, optional, keyword-only
            ``(B, S)`` source padding mask, applied in cross-attention.
        past_key_value : EncoderDecoderCache or None, optional, keyword-only
            Decoder self-attention history and the cross-attention keys
            and values; read and extended in place.
        use_cache : bool, default=False, keyword-only
            Read and write ``past_key_value``.

        Returns
        -------
        Tensor
            ``(B, T, d_model)`` decoder output for the ``T`` new positions.
        """
        cache = past_key_value if use_cache else None
        return self.decoder(
            self.shared(tgt_ids),
            tgt_attention_mask,
            memory=memory,
            memory_mask=memory_attention_mask,
            cache=cache,
        )

    @override
    def forward(
        self,
        input_ids: Tensor,
        decoder_input_ids: Tensor,
        attention_mask: Tensor | None = None,
        decoder_attention_mask: Tensor | None = None,
    ) -> T5ModelOutput:
        """Encode the source and decode the target against it.

        Parameters
        ----------
        input_ids : Tensor
            ``(B, S)`` source token ids.
        decoder_input_ids : Tensor
            ``(B, T)`` target ids, already shifted right.
        attention_mask : Tensor or None, optional
            ``(B, S)`` source padding mask.
        decoder_attention_mask : Tensor or None, optional
            ``(B, T)`` target padding mask.

        Returns
        -------
        T5ModelOutput
            Decoder and encoder outputs; :class:`T5ForSeq2SeqLM` projects
            the decoder side to the vocabulary.
        """
        memory = self.encode(input_ids, attention_mask)
        hidden = self.decode(
            decoder_input_ids,
            memory,
            tgt_attention_mask=decoder_attention_mask,
            memory_attention_mask=attention_mask,
        )
        return T5ModelOutput(last_hidden_state=hidden, encoder_last_hidden_state=memory)


# ─────────────────────────────────────────────────────────────────────────────
# Text-to-text head
# ─────────────────────────────────────────────────────────────────────────────


class T5ForSeq2SeqLM(LanguageModelingModel):
    r"""T5 with its language-modelling head — the model every T5 task trains.

    Wraps :class:`T5Model` with a bias-free projection to the vocabulary.
    With ``config.tie_word_embeddings`` (the paper's setting) the
    projection *is* the shared token table, and the decoder output is
    scaled by :math:`d_{\text{model}}^{-1/2}` first: the table is drawn at
    unit variance for the input side, so the tied output would otherwise
    be :math:`\sqrt{d_{\text{model}}}` too large.

    Pass ``labels`` alone and the decoder input is built from them —
    ``decoder_start_token_id`` followed by the labels shifted right, with
    ``-100`` replaced by the pad id — as the paper's teacher forcing does.

    Parameters
    ----------
    config : T5Config
        Hyperparameters; ``tie_word_embeddings`` and
        ``decoder_start_token_id`` shape the head.

    Attributes
    ----------
    t5 : T5Model
        The encoder-decoder trunk.
    lm_head : nn.Linear
        ``(d_model → vocab_size)`` projection, the shared table when tied.

    Notes
    -----
    Reference: Raffel et al., JMLR 2020 (arXiv:1910.10683), §2.1 and §3.1.4.

    With ``labels`` the loss is the mean per-token cross-entropy over
    positions not labelled ``-100``:

    .. math::

        \mathcal{L} = -\frac{1}{|V|} \sum_{(b, t) \in V}
            \log p_\theta(y_{b,t} \mid x_b,\, y_{b,<t}).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.t5 import T5Config, T5ForSeq2SeqLM
    >>> cfg = T5Config(vocab_size=100, hidden_size=32, num_hidden_layers=2,
    ...                num_decoder_layers=2, num_attention_heads=4,
    ...                head_dim=8, intermediate_size=64)
    >>> model = T5ForSeq2SeqLM(cfg).eval()
    >>> src = lucid.tensor([[13, 5, 42, 1]])
    >>> labels = lucid.tensor([[7, 9, 1]])
    >>> out = model(src, labels=labels)
    >>> out.logits.shape, out.loss.shape
    ((1, 3, 100), ())
    """

    config_class: ClassVar[type[T5Config]] = T5Config
    base_model_prefix: ClassVar[str] = "t5"

    def __init__(self, config: T5Config) -> None:
        """Initialise the model. See the class docstring for parameters."""
        super().__init__(config)
        self.config: T5Config = config
        self.t5 = T5Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self._tie_word_embeddings()
        else:
            nn.init.normal_(self.lm_head.weight, std=config.initializer_factor)

    @override
    def _tie_word_embeddings(self) -> None:
        weight = self.t5.shared.weight
        self.lm_head.weight = weight
        self.lm_head.out_features = int(weight.shape[0])

    def _logits(self, hidden: Tensor) -> Tensor:
        """Vocabulary logits, rescaled first when the head is the token table."""
        if self.config.tie_word_embeddings:
            hidden = hidden * self.config.hidden_size**-0.5
        return self.lm_head(hidden)

    def _shift_right(self, labels: Tensor) -> Tensor:
        """``decoder_start_token_id`` then ``labels[:, :-1]``, ``-100`` → pad."""
        pad = self.config.pad_token_id
        if pad is None:
            raise ValueError("T5 shifts labels into decoder inputs, which needs pad_token_id")
        labels = labels.long()
        start = lucid.full(
            (int(labels.shape[0]), 1),
            self.config.decoder_start_token_id,
            dtype=lucid.int64,
            device=labels.device.type,
        )
        shifted = lucid.cat([start, labels[:, :-1]], dim=1)
        return lucid.where(shifted == -100, lucid.full_like(shifted, pad), shifted)

    @override
    def forward(
        self,
        input_ids: Tensor,
        decoder_input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        decoder_attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> Seq2SeqLMOutput:
        """Teacher-forced pass, with the loss when ``labels`` are given.

        Parameters
        ----------
        input_ids : Tensor
            ``(B, S)`` source token ids.
        decoder_input_ids : Tensor or None, optional
            ``(B, T)`` target ids, shifted right.  Built from ``labels``
            when omitted.
        attention_mask : Tensor or None, optional
            ``(B, S)`` source padding mask.
        decoder_attention_mask : Tensor or None, optional
            ``(B, T)`` target padding mask.
        labels : Tensor or None, optional
            ``(B, T)`` target ids; ``-100`` marks a position the loss skips.

        Returns
        -------
        Seq2SeqLMOutput
            ``logits`` of shape ``(B, T, vocab_size)``, the loss when
            ``labels`` were given, and the encoder output.

        Raises
        ------
        ValueError
            If neither ``decoder_input_ids`` nor ``labels`` is given.
        """
        if decoder_input_ids is None:
            if labels is None:
                raise ValueError(
                    "T5ForSeq2SeqLM needs decoder_input_ids, or labels to shift " "right into them"
                )
            decoder_input_ids = self._shift_right(labels)
        out = self.t5(
            input_ids,
            decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )
        logits = self._logits(out.last_hidden_state)

        loss: Tensor | None = None
        if labels is not None:
            batch, length, vocab = (int(d) for d in logits.shape)
            loss = F.cross_entropy(
                logits.reshape(batch * length, vocab),
                labels.reshape(batch * length).long(),
                ignore_index=-100,
            )
        return Seq2SeqLMOutput(
            logits=logits, loss=loss, encoder_last_hidden_state=out.encoder_last_hidden_state
        )

    @lucid.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        *,
        max_length: int = 20,
        attention_mask: Tensor | None = None,
        use_cache: bool = True,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        decoder_start_token_id: int | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
    ) -> Tensor:
        """Decode an answer for each source — greedy unless ``do_sample``.

        Parameters
        ----------
        input_ids : Tensor
            ``(B, S)`` source ids, task prefix included (``"translate
            English to German: …"``).
        max_length : int, default=20, keyword-only
            Cap on the returned length, start token included.
        attention_mask : Tensor or None, optional, keyword-only
            ``(B, S)`` source padding mask.
        use_cache : bool, default=True, keyword-only
            Decode one token per step against an encoder-decoder cache.
        do_sample : bool, default=False, keyword-only
            Sample instead of taking the ``argmax``.
        temperature : float, default=1.0, keyword-only
            Logit temperature for sampling.
        top_k : int or None, optional, keyword-only
            Sample among the ``top_k`` most likely tokens only.
        top_p : float or None, optional, keyword-only
            Sample from the smallest set reaching probability ``top_p``.
        repetition_penalty : float, default=1.0, keyword-only
            Down-weights tokens the row has already produced.
        decoder_start_token_id : int or None, optional, keyword-only
            First target token; ``config.decoder_start_token_id`` (the
            pad id) by default.
        eos_token_id : int or None, optional, keyword-only
            Token that finishes a row; ``config.eos_token_id`` by default.
        pad_token_id : int or None, optional, keyword-only
            Fills a finished row; ``config.pad_token_id`` by default.

        Returns
        -------
        Tensor
            ``(B, T_out)`` int64 target ids beginning with the start token.
        """
        config = self.config
        start = (
            decoder_start_token_id
            if decoder_start_token_id is not None
            else config.decoder_start_token_id
        )
        eos = eos_token_id if eos_token_id is not None else config.eos_token_id
        pad = pad_token_id if pad_token_id is not None else config.pad_token_id
        dev = input_ids.device.type
        memory = self.t5.encode(input_ids, attention_mask)

        def next_logits(tokens: Tensor, cache: EncoderDecoderCache | None) -> Tensor:
            hidden = self.t5.decode(
                tokens,
                memory,
                memory_attention_mask=attention_mask,
                past_key_value=cache,
                use_cache=cache is not None,
            )
            return self._logits(hidden[:, -1, :])

        sampling = _SamplingParams(
            do_sample=do_sample, temperature=temperature, top_k=top_k, top_p=top_p,
            repetition_penalty=repetition_penalty, eos_token_id=eos,
            pad_token_id=pad if pad is not None else start, dev=dev,
        )  # fmt: skip
        return seq2seq_generate(
            next_logits, batch=int(input_ids.shape[0]), device=dev, start_token_id=start,
            max_length=max_length, use_cache=use_cache, sampling=sampling,
        )  # fmt: skip
