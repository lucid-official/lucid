"""BERT model (Devlin et al., 2018) — encoder-only Transformer.

Module / parameter naming matches HuggingFace Transformers' ``BERTModel`` so
state dicts can be ported with a flat key rename.  Top-level layout:

    bert.embeddings.{word, position, token_type}_embeddings
    bert.embeddings.LayerNorm
    bert.encoder.layer.{i}.attention.self.{query, key, value}
    bert.encoder.layer.{i}.attention.output.{dense, LayerNorm}
    bert.encoder.layer.{i}.intermediate.dense
    bert.encoder.layer.{i}.output.{dense, LayerNorm}
    bert.pooler.dense
    cls.predictions.{transform.{dense, LayerNorm}, decoder}   (MLM head)
    classifier                                                (cls / token / qa)
"""

import math
from dataclasses import dataclass
from typing import ClassVar, final, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.nn.functional.attention import _masked_attention
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._tasks import (
    LanguageModelingModel,
    SequenceClassificationModel,
    TokenClassificationModel,
)
from lucid.models._mixins import MaskedLMMixin
from lucid.models._output import (
    BaseModelOutputWithPooling,
    CausalLMOutput,
    MaskedLMOutput,
    QuestionAnsweringOutput,
    SequenceClassificationOutput,
    TokenClassificationOutput,
    ModelOutput,
)
from lucid.models._utils._text import (
    extended_attention_mask,
    init_text_weights as _init_weights,
    text_activation,
)
from lucid.models.text.bert._config import BERTConfig

# ─────────────────────────────────────────────────────────────────────────────
# Embeddings
# ─────────────────────────────────────────────────────────────────────────────


@final
class _BERTEmbeddings(nn.Module):
    """Word + position + token-type embedding sum, then LN + Dropout."""

    position_ids: Tensor

    def __init__(self, config: BERTConfig) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        # HF names: ``LayerNorm`` (capitalised) so checkpoints port directly.
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout)

        # Position ids are deterministic [0, max_pos) — register as a buffer so
        # they ride along with .to(device=...).
        pos = lucid.arange(config.max_position_embeddings).long().unsqueeze(0)
        self.register_buffer("position_ids", pos, persistent=False)

    @override
    def forward(
        self,
        input_ids: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
    ) -> Tensor:
        """Sum the word, position and token-type embeddings.

        Exactly one of ``input_ids`` and ``inputs_embeds`` is required.
        ``inputs_embeds`` skips the word-embedding lookup, which is what a
        caller doing soft prompts or adapter tuning needs; ``position_ids``
        overrides the default ``0..T-1``, which packed or shifted sequences
        need.
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError(
                "pass exactly one of input_ids and inputs_embeds; got "
                f"input_ids={'a tensor' if input_ids is not None else None} "
                f"and inputs_embeds="
                f"{'a tensor' if inputs_embeds is not None else None}."
            )
        if inputs_embeds is not None:
            B, T = int(inputs_embeds.shape[0]), int(inputs_embeds.shape[1])
            dev = inputs_embeds.device.type
            words = inputs_embeds
        else:
            assert input_ids is not None
            B, T = int(input_ids.shape[0]), int(input_ids.shape[1])
            dev = input_ids.device.type
            words = self.word_embeddings(input_ids)

        if position_ids is None:
            pos_ids = self.position_ids[:, :T]
        else:
            pos_ids = position_ids
        positions = self.position_embeddings(pos_ids)

        if token_type_ids is None:
            token_type_ids = lucid.zeros((B, T), device=dev).long()
        types = self.token_type_embeddings(token_type_ids)

        emb = words + positions + types
        emb = self.LayerNorm(emb)
        return self.dropout(emb)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-head self-attention
# ─────────────────────────────────────────────────────────────────────────────


def _per_head_mask(head_mask: Tensor, batch: int, heads: int) -> Tensor:
    """``head_mask`` as ``(1 or B, H, 1, 1)``; a finer mask (``(B, H, T, T)``)
    would broadcast against the ``(B, H, T, D)`` context, silently at T == D."""
    shape = tuple(int(d) for d in head_mask.shape)
    if shape == (heads,):
        return head_mask.reshape(1, heads, 1, 1)
    if len(shape) == 4 and shape[0] in (1, batch) and shape[1:] == (heads, 1, 1):
        return head_mask
    raise ValueError(
        f"head_mask must be ({heads},) or (1 or {batch}, {heads}, 1, 1) — one "
        f"multiplier per head — got shape {shape}"
    )


@final
class _BERTSelfAttention(nn.Module):
    """Multi-head self-attention with separate Q / K / V projections.

    HF stores Q/K/V as three independent ``Linear``s (not fused), which is the
    convention we mirror here so weight porting is a direct rename.
    """

    def __init__(self, config: BERTConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = math.sqrt(self.head_dim)

        self.query = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.key = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.value = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.dropout = nn.Dropout(p=config.attention_dropout)

    def _shape(self, x: Tensor, B: int, T: int) -> Tensor:
        # (B, T, hidden) → (B, H, T, head_dim)
        return x.reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def _unfused_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention_mask: Tensor | None,
        head_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """``softmax(q kᵀ / scale + mask) v``, keeping the weights.

        The fused kernel is faster precisely because it never forms the
        ``(B, H, T, T)`` score matrix.  Anything that needs to *see* or
        *edit* those weights — returning them, or masking whole heads — has
        to pay for materialising them, so this path exists alongside rather
        than replacing the fused one.

        Args:
            q, k, v:        ``(B, H, T, D)`` projected heads.
            attention_mask: Additive mask (0 keep / -inf drop).
            head_mask:      ``(H,)`` or ``(B, H, 1, 1)`` multiplier applied
                to the post-softmax weights.  Zeroing an entry removes that
                head's contribution entirely, which is what the probing
                literature uses it for.

        Returns:
            ``(context, attention_weights)`` with weights ``(B, H, T, T)`` —
            the ones applied to ``v``, so after dropout and the head mask, as
            the reference returns them.
        """
        # The fused path's masked core, asked for its weights: a hand-written
        # softmax here gave NaN for an all ``-inf`` row (LCD-387).
        drop = self.dropout.p if self.training else 0.0
        ctx, weights = _masked_attention(
            q, k, v, attention_mask, 1.0 / self.scale, dropout_p=drop, need_weights=True
        )
        assert weights is not None  # need_weights=True always returns them
        if head_mask is not None:
            # Per-head scalar: scaling after ``@ v`` equals scaling before.
            shaped = _per_head_mask(head_mask, int(q.shape[0]), self.num_heads)
            ctx = ctx * shaped
            weights = weights * shaped
        return ctx, weights

    @override
    def forward(
        self,
        hidden: Tensor,
        attention_mask: Tensor | None = None,
        head_mask: Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        B, T, _ = hidden.shape
        q = self._shape(self.query(hidden), B, T)
        k = self._shape(self.key(hidden), B, T)
        v = self._shape(self.value(hidden), B, T)

        weights: Tensor | None = None
        if output_attentions or head_mask is not None:
            ctx, weights = self._unfused_attention(q, k, v, attention_mask, head_mask)
            if not output_attentions:
                weights = None
        else:
            # Fused scaled-dot-product attention: one kernel that skips
            # materializing the (B, H, T, T) scores tensor.  ``attention_mask``
            # is the standard additive mask (0 keep / -inf drop).  Q/K/V stay
            # separate so weight porting is a direct rename, and this is
            # bit-exact with the manual path above.
            ctx = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                scale=1.0 / self.scale,
            )

        # (B, H, T, D) → (B, T, H*D)
        ctx = ctx.permute(0, 2, 1, 3).reshape(B, T, self.num_heads * self.head_dim)
        return ctx, weights


@final
class _BERTSelfOutput(nn.Module):
    """Post-attention dense + LN + residual."""

    def __init__(self, config: BERTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout)

    @override
    def forward(self, hidden: Tensor, input_tensor: Tensor) -> Tensor:
        h = self.dropout(self.dense(hidden))
        return self.LayerNorm(h + input_tensor)


@final
class _BERTAttention(nn.Module):
    def __init__(self, config: BERTConfig) -> None:
        super().__init__()
        # HF names ``self`` for the projection block — keep the same key.
        self.self = _BERTSelfAttention(config)
        self.output = _BERTSelfOutput(config)

    @override
    def forward(
        self,
        hidden: Tensor,
        attention_mask: Tensor | None = None,
        head_mask: Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        attn_out, weights = self.self(
            hidden,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        return self.output(attn_out, hidden), weights


# ─────────────────────────────────────────────────────────────────────────────
# Feed-forward block
# ─────────────────────────────────────────────────────────────────────────────


@final
class _BERTIntermediate(nn.Module):
    def __init__(self, config: BERTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self._act_name = config.hidden_act

    @override
    def forward(self, x: Tensor) -> Tensor:
        return text_activation(self._act_name, self.dense(x))


@final
class _BERTOutput(nn.Module):
    def __init__(self, config: BERTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout)

    @override
    def forward(self, hidden: Tensor, input_tensor: Tensor) -> Tensor:
        h = self.dropout(self.dense(hidden))
        return self.LayerNorm(h + input_tensor)


# ─────────────────────────────────────────────────────────────────────────────
# Transformer layer + encoder stack
# ─────────────────────────────────────────────────────────────────────────────


@final
class _BERTLayer(nn.Module):
    def __init__(self, config: BERTConfig) -> None:
        super().__init__()
        self.attention = _BERTAttention(config)
        self.intermediate = _BERTIntermediate(config)
        self.output = _BERTOutput(config)

    @override
    def forward(
        self,
        hidden: Tensor,
        attention_mask: Tensor | None = None,
        head_mask: Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        attn_out, weights = self.attention(
            hidden,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        inter = self.intermediate(attn_out)
        return self.output(inter, attn_out), weights


@final
class _BERTEncoder(nn.Module):
    def __init__(self, config: BERTConfig) -> None:
        super().__init__()
        self.layer = nn.ModuleList([_BERTLayer(config) for _ in range(config.num_hidden_layers)])

    @override
    def forward(
        self,
        hidden: Tensor,
        attention_mask: Tensor | None = None,
        output_hidden_states: bool = False,
        head_mask: Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, ...] | None, tuple[Tensor, ...] | None]:
        """Run the layer stack, optionally keeping every intermediate state.

        Args:
            hidden:         ``(B, T, H)`` embedding output.
            attention_mask: Additive mask, already extended.
            output_hidden_states: Collect the input embedding plus each
                layer's output.  Off by default so the common path holds
                only one activation at a time.

        Returns:
            ``(last_hidden_state, hidden_states)``.  The second is ``None``
            unless requested, and otherwise has ``num_layers + 1`` entries —
            the embedding output first, as the reference orders them.
        """
        states: list[Tensor] = [hidden] if output_hidden_states else []
        attentions: list[Tensor] = []
        for i, layer in enumerate(self.layer):
            # A per-layer head mask is indexed by depth; a single (H,) mask
            # applies to every layer, which is the usual probing setup.
            layer_head_mask = (
                head_mask[i] if head_mask is not None and head_mask.ndim > 1 else head_mask
            )
            hidden, weights = layer(
                hidden,
                attention_mask=attention_mask,
                head_mask=layer_head_mask,
                output_attentions=output_attentions,
            )
            if output_hidden_states:
                states.append(hidden)
            if output_attentions and weights is not None:
                attentions.append(weights)
        return (
            hidden,
            tuple(states) if output_hidden_states else None,
            tuple(attentions) if output_attentions else None,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Pooler — first-token tanh projection feeding sentence-level heads
# ─────────────────────────────────────────────────────────────────────────────


@final
class _BERTPooler(nn.Module):
    def __init__(self, config: BERTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

    @override
    def forward(self, hidden: Tensor) -> Tensor:
        # CLS token is position 0 by tokenization convention.
        cls_hidden = hidden[:, 0]
        return F.tanh(self.dense(cls_hidden))


# ─────────────────────────────────────────────────────────────────────────────
# Top-level encoder model
# ─────────────────────────────────────────────────────────────────────────────


class BERTModel(PretrainedModel):
    r"""Bare BERT encoder returning hidden states and pooled CLS embedding.

    Implements the bidirectional transformer encoder of Devlin et al., 2018.
    Token, position, and segment embeddings are summed, LayerNormed, and
    dropout-regularised, then passed through :math:`L` transformer blocks of
    multi-head self-attention plus position-wise feed-forward.  A single
    tanh-activated linear ("pooler") on the first ``[CLS]`` token produces a
    sentence-level embedding used by classification heads.

    Use this class as the trunk when you want raw hidden states; the
    task-specific subclasses (``BERTFor*``) wrap it with appropriate heads.

    Parameters
    ----------
    config : BERTConfig
        Hyperparameters controlling vocabulary, depth, width, head count, and
        regularisation.  See :class:`BERTConfig` for the full field list.

    Attributes
    ----------
    embeddings : nn.Module
        Token + position + token-type embedding block followed by LayerNorm
        and dropout.
    encoder : nn.Module
        Stack of ``config.num_hidden_layers`` transformer encoder layers.
    pooler : nn.Module
        Dense + tanh projection of the ``[CLS]`` hidden state.
    config_class : type[BERTConfig]
        Class-level pointer used by the registry to instantiate a matching
        config from disk.
    base_model_prefix : str
        Prefix (``"bert"``) under which sub-module checkpoints are nested in
        task-head variants — used during weight loading.

    Notes
    -----
    Reference: Devlin, Chang, Lee, and Toutanova, *"BERT: Pre-training of
    Deep Bidirectional Transformers for Language Understanding"*, NAACL 2019
    (arXiv:1810.04805).

    Self-attention follows the scaled dot-product form

    .. math::

        \mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(
            \frac{Q K^{\top}}{\sqrt{d_k}}
        \right) V

    with :math:`d_k = H / A`.  Each layer applies multi-head attention,
    followed by a feed-forward block

    .. math::

        \mathrm{FFN}(x) = \mathrm{GELU}(x W_1 + b_1) W_2 + b_2,

    each wrapped by a residual connection and post-LayerNorm.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.bert import BERTConfig, BERTModel
    >>> cfg = BERTConfig(num_hidden_layers=2, hidden_size=128, num_attention_heads=2,
    ...                  intermediate_size=512)
    >>> model = BERTModel(cfg).eval()
    >>> input_ids = lucid.tensor([[101, 7592, 2088, 102]])   # [CLS] hello world [SEP]
    >>> out = model(input_ids)
    >>> out.last_hidden_state.shape   # (B=1, T=4, H=128)
    (1, 4, 128)
    >>> out.pooler_output.shape       # (B=1, H=128)
    (1, 128)
    """

    config_class: ClassVar[type[BERTConfig]] = BERTConfig
    base_model_prefix: ClassVar[str] = "bert"

    def __init__(self, config: BERTConfig) -> None:
        super().__init__(config)
        self.embeddings = _BERTEmbeddings(config)
        self.encoder = _BERTEncoder(config)
        self.pooler = _BERTPooler(config)
        _init_weights(self, config.initializer_range)

    @override
    def get_input_embeddings(self) -> nn.Module:
        return self.embeddings.word_embeddings

    @override
    def set_input_embeddings(self, value: nn.Module) -> None:
        if not isinstance(value, nn.Embedding):
            raise TypeError(
                f"BERTModel input embeddings must be nn.Embedding, got {type(value).__name__}"
            )
        self.embeddings.word_embeddings = value

    @override
    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        output_hidden_states: bool = False,
        head_mask: Tensor | None = None,
        output_attentions: bool = False,
    ) -> BaseModelOutputWithPooling:
        """Encode a batch and return the sequence output plus the pooled CLS.

        ``position_ids`` and ``inputs_embeds`` are forwarded to the embedding
        layer; see :meth:`_BERTEmbeddings.forward`.  ``output_hidden_states``
        additionally returns every layer's output, embedding first.

        ``head_mask`` and ``output_attentions`` both need the per-head
        attention weights, so requesting either switches that layer to the
        unfused ``softmax(qk^T) v`` path — the fused kernel never
        materialises the ``(B, H, T, T)`` matrix, which is exactly why it
        is fast.  Neither flag changes the result when off, and the fused
        path stays the default.
        """
        if inputs_embeds is not None:
            B, T = int(inputs_embeds.shape[0]), int(inputs_embeds.shape[1])
        else:
            if input_ids is None:
                raise ValueError("pass either input_ids or inputs_embeds.")
            B, T = int(input_ids.shape[0]), int(input_ids.shape[1])

        ext_mask = extended_attention_mask(attention_mask, (B, T))

        hidden = self.embeddings(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        sequence_output, all_hidden, all_attn = self.encoder(
            hidden,
            attention_mask=ext_mask,
            output_hidden_states=output_hidden_states,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        pooled_output = self.pooler(sequence_output)

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=all_hidden,
            attentions=all_attn,
        )


# ─────────────────────────────────────────────────────────────────────────────
# MLM head — used by BERTForMaskedLM
# ─────────────────────────────────────────────────────────────────────────────


@final
class _BERTPredictionHeadTransform(nn.Module):
    def __init__(self, config: BERTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self._act_name = config.hidden_act

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = self.dense(x)
        x = text_activation(self._act_name, x)
        return self.LayerNorm(x)


@final
class _BERTLMPredictionHead(nn.Module):
    """Decoder linear (weight tied to input embeddings) + standalone bias."""

    bias: Tensor

    def __init__(self, config: BERTConfig) -> None:
        super().__init__()
        self.transform = _BERTPredictionHeadTransform(config)
        # Decoder is created untied; ``BERTForMaskedLM`` re-binds the weight to
        # the input embedding table when ``tie_word_embeddings`` is set.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # HF keeps the output bias as a standalone parameter on the head, not on
        # the Linear — so checkpoint keys are ``cls.predictions.bias`` /
        # ``cls.predictions.decoder.weight``.
        self.bias = nn.Parameter(lucid.zeros(config.vocab_size))

    def tie_to(self, weight: nn.Parameter) -> None:
        """Decode with ``weight`` (shared, not copied) and fit the bias to it.

        The bias holds one entry per output token, so a table with another
        number of rows needs one of matching length: surviving entries carry
        over and new tokens start at zero.
        """
        self.decoder.weight = weight
        vocab = int(weight.shape[0])
        self.decoder.out_features = vocab
        have = int(self.bias.shape[0])
        if have != vocab:
            kept = self.bias.detach()[: min(have, vocab)]
            if vocab > have:
                pad = lucid.zeros(vocab - have, dtype=kept.dtype, device=kept.device.type)
                kept = lucid.cat([kept, pad])
            self.bias = nn.Parameter(kept)

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = self.transform(x)
        logits = self.decoder(x)
        return logits + self.bias


@final
class _BERTOnlyMLMHead(nn.Module):
    def __init__(self, config: BERTConfig) -> None:
        super().__init__()
        self.predictions = _BERTLMPredictionHead(config)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.predictions(x)


# ─────────────────────────────────────────────────────────────────────────────
# Task heads
# ─────────────────────────────────────────────────────────────────────────────


class BERTForMaskedLM(LanguageModelingModel, MaskedLMMixin):
    r"""BERT with a tied masked-language-modeling head.

    Implements the masked-LM half of the Devlin et al. (2018) pre-training
    objective.  A two-layer projection (dense + GELU + LayerNorm) maps each
    hidden state to vocabulary logits via a decoder whose weight matrix is
    tied to the input ``word_embeddings`` table when
    ``config.tie_word_embeddings`` is True.  Use for pre-training from
    scratch, continued pre-training on domain corpora, or fill-in-the-blank
    inference.

    Parameters
    ----------
    config : BERTConfig
        BERT hyperparameters.  ``config.tie_word_embeddings`` (default True)
        controls whether the decoder weight is bound to the input embedding
        matrix to halve the parameter count.

    Attributes
    ----------
    bert : BERTModel
        Underlying bidirectional encoder trunk.
    cls : nn.Module
        Masked-LM prediction head with its own dense + LayerNorm transform
        and an output decoder of shape ``(hidden_size, vocab_size)``.

    Notes
    -----
    Reference: Devlin, Chang, Lee, and Toutanova, *"BERT: Pre-training of
    Deep Bidirectional Transformers for Language Understanding"*, NAACL 2019
    (arXiv:1810.04805), section 3.1 Task #1.

    When ``labels`` is supplied the head computes

    .. math::

        \mathcal{L}_{\mathrm{MLM}} = -\frac{1}{|M|}
            \sum_{i \in M} \log p_{\theta}(x_i \mid x_{\setminus M})

    over the set :math:`M` of masked positions, with positions where the
    label equals ``-100`` excluded from the sum.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.bert import BERTConfig, BERTForMaskedLM
    >>> cfg = BERTConfig(num_hidden_layers=2, hidden_size=128,
    ...                  num_attention_heads=2, intermediate_size=512)
    >>> model = BERTForMaskedLM(cfg).eval()
    >>> input_ids = lucid.tensor([[101, 7592, 103, 102]])   # [CLS] hello [MASK] [SEP]
    >>> out = model(input_ids)
    >>> out.logits.shape   # (B=1, T=4, V=30522)
    (1, 4, 30522)
    """

    config_class: ClassVar[type[BERTConfig]] = BERTConfig
    base_model_prefix: ClassVar[str] = "bert"

    def __init__(self, config: BERTConfig) -> None:
        super().__init__(config)
        self.bert = BERTModel(config)
        self.cls = _BERTOnlyMLMHead(config)
        if config.tie_word_embeddings:
            self._tie_word_embeddings()

    @override
    def _tie_word_embeddings(self) -> None:
        # Bind the decoder weight to the input embedding matrix so the two
        # share storage and gradients.  HF does the same thing.
        self.cls.predictions.tie_to(self.bert.embeddings.word_embeddings.weight)

    @override
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> MaskedLMOutput:
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        prediction_scores = self.cls(outputs.last_hidden_state)

        loss: Tensor | None = None
        if labels is not None:
            loss = self.compute_lm_loss(prediction_scores, labels)

        return MaskedLMOutput(logits=prediction_scores, loss=loss)


class BERTForSequenceClassification(SequenceClassificationModel):
    r"""BERT with a pooled-CLS linear classifier for sequence-level tasks.

    Wraps the bidirectional encoder with a dropout-regularised linear head
    operating on the ``[CLS]`` pooled embedding.  This is the standard
    fine-tuning recipe for GLUE-style sentence/sentence-pair tasks (SST-2,
    MNLI, QQP, RTE, ...) introduced in Devlin et al., 2018 §4.1.

    Parameters
    ----------
    config : BERTConfig
        BERT hyperparameters.  ``config.num_labels`` sets the output
        dimension; ``config.classifier_dropout`` (falling back to
        ``hidden_dropout``) sets the dropout applied before the linear.

    Attributes
    ----------
    bert : BERTModel
        Underlying bidirectional encoder trunk.
    dropout : nn.Dropout
        Dropout layer applied to the pooled ``[CLS]`` embedding.
    classifier : nn.Linear
        Final linear of shape ``(hidden_size, num_labels)`` producing logits.

    Notes
    -----
    Reference: Devlin, Chang, Lee, and Toutanova, *"BERT: Pre-training of
    Deep Bidirectional Transformers for Language Understanding"*, NAACL 2019
    (arXiv:1810.04805).

    The pooled representation is
    :math:`p = \tanh(W_{\mathrm{pool}}\, h_{[\mathrm{CLS}]} + b_{\mathrm{pool}})`,
    and the final logits are :math:`z = W_{\mathrm{cls}}\,\mathrm{Dropout}(p) + b_{\mathrm{cls}}`.
    When ``labels`` is provided the loss is exposed as ``output.loss``:
    cross-entropy over ``num_labels`` classes, or — when ``num_labels == 1``,
    the single-output regression setting GLUE's STS-B uses — mean squared
    error against the float targets.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.bert import BERTConfig, BERTForSequenceClassification
    >>> cfg = BERTConfig(num_labels=3, num_hidden_layers=2, hidden_size=128,
    ...                  num_attention_heads=2, intermediate_size=512)
    >>> model = BERTForSequenceClassification(cfg).eval()
    >>> input_ids = lucid.tensor([[101, 7592, 102]])
    >>> out = model(input_ids)
    >>> out.logits.shape   # (B=1, num_labels=3)
    (1, 3)
    """

    config_class: ClassVar[type[BERTConfig]] = BERTConfig
    base_model_prefix: ClassVar[str] = "bert"

    def __init__(self, config: BERTConfig) -> None:
        super().__init__(config)
        self.bert = BERTModel(config)
        drop = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout
        )
        self.dropout = nn.Dropout(p=drop)
        self.num_labels = config.num_labels
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    @override
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> SequenceClassificationOutput:
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = self.dropout(outputs.pooler_output)
        logits = self.classifier(pooled)

        loss: Tensor | None = None
        if labels is not None:
            if self.num_labels == 1:
                # Single output unit means regression (GLUE's STS-B).  Softmax
                # cross-entropy over one class is identically zero, so the
                # gradient would vanish and the head would never train.
                loss = F.mse_loss(logits.reshape(-1), labels.float().reshape(-1))
            else:
                loss = F.cross_entropy(logits, labels)

        return SequenceClassificationOutput(logits=logits, loss=loss)


class BERTForTokenClassification(TokenClassificationModel, MaskedLMMixin):
    r"""BERT with a per-token linear classifier for tagging tasks.

    Wraps the bidirectional encoder with a dropout-regularised linear head
    applied independently at every sequence position.  Used for token-level
    fine-tunes such as named-entity recognition (CoNLL-2003), part-of-speech
    tagging, and chunking — see Devlin et al., 2018 §4.3.

    Parameters
    ----------
    config : BERTConfig
        BERT hyperparameters.  ``config.num_labels`` sets the per-position
        output dimension; ``config.classifier_dropout`` (falling back to
        ``hidden_dropout``) sets the dropout applied before the linear.

    Attributes
    ----------
    bert : BERTModel
        Underlying bidirectional encoder trunk.
    dropout : nn.Dropout
        Dropout applied to the full sequence hidden states.
    classifier : nn.Linear
        Final linear of shape ``(hidden_size, num_labels)`` mapping each
        token's hidden state to per-class logits.

    Notes
    -----
    Reference: Devlin, Chang, Lee, and Toutanova, *"BERT: Pre-training of
    Deep Bidirectional Transformers for Language Understanding"*, NAACL 2019
    (arXiv:1810.04805).

    The loss (when ``labels`` is provided) is the masked cross-entropy
    inherited from :class:`MaskedLMMixin`:

    .. math::

        \mathcal{L} = -\frac{1}{|V|}
            \sum_{(b, t) \in V} \log p_{\theta}\!\left(y_{b,t} \mid x_b\right),

    where :math:`V` is the set of positions with ``label != -100``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.bert import BERTConfig, BERTForTokenClassification
    >>> cfg = BERTConfig(num_labels=9, num_hidden_layers=2, hidden_size=128,
    ...                  num_attention_heads=2, intermediate_size=512)
    >>> model = BERTForTokenClassification(cfg).eval()
    >>> input_ids = lucid.tensor([[101, 7592, 2088, 102]])
    >>> out = model(input_ids)
    >>> out.logits.shape   # (B=1, T=4, num_labels=9)
    (1, 4, 9)
    """

    config_class: ClassVar[type[BERTConfig]] = BERTConfig
    base_model_prefix: ClassVar[str] = "bert"

    def __init__(self, config: BERTConfig) -> None:
        super().__init__(config)
        self.bert = BERTModel(config)
        drop = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout
        )
        self.dropout = nn.Dropout(p=drop)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    @override
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> TokenClassificationOutput:
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        seq = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(seq)

        loss: Tensor | None = None
        if labels is not None:
            loss = self.compute_lm_loss(logits, labels)

        return TokenClassificationOutput(logits=logits, loss=loss)


class BERTForQuestionAnswering(SequenceClassificationModel):
    r"""BERT with a 2-way span head for extractive question answering.

    Wraps the bidirectional encoder with a single linear of output width 2,
    producing start- and end-position logits over each token in the input.
    This is the SQuAD v1.1 / v2.0 fine-tuning recipe of Devlin et al., 2018
    §4.2 — given a ``(question, context)`` pair concatenated with ``[SEP]``,
    the model predicts the answer span inside the context.

    Parameters
    ----------
    config : BERTConfig
        BERT hyperparameters.  The QA head is always 2-way; ``num_labels``
        is ignored here.

    Attributes
    ----------
    bert : BERTModel
        Underlying bidirectional encoder trunk.
    qa_outputs : nn.Linear
        Final linear of shape ``(hidden_size, 2)`` mapping each token's
        hidden state to ``(start_logit, end_logit)``.

    Notes
    -----
    Reference: Devlin, Chang, Lee, and Toutanova, *"BERT: Pre-training of
    Deep Bidirectional Transformers for Language Understanding"*, NAACL 2019
    (arXiv:1810.04805).

    When both ``start_positions`` and ``end_positions`` are provided, the
    loss is the symmetric average of two cross-entropies:

    .. math::

        \mathcal{L} = \tfrac{1}{2}\!\left(
            \mathrm{CE}(z^{\mathrm{start}}, y^{\mathrm{start}})
          + \mathrm{CE}(z^{\mathrm{end}},   y^{\mathrm{end}})
        \right).

    The forward returns a :class:`QuestionAnsweringOutput` carrying
    ``start_logits`` and ``end_logits``, each of shape ``(B, T)``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.bert import BERTConfig, BERTForQuestionAnswering
    >>> cfg = BERTConfig(num_hidden_layers=2, hidden_size=128,
    ...                  num_attention_heads=2, intermediate_size=512)
    >>> model = BERTForQuestionAnswering(cfg).eval()
    >>> input_ids = lucid.tensor([[101, 2040, 2003, 102, 1045, 2572, 102]])
    >>> out = model(input_ids)
    >>> out.start_logits.shape, out.end_logits.shape
    ((1, 7), (1, 7))
    """

    config_class: ClassVar[type[BERTConfig]] = BERTConfig
    base_model_prefix: ClassVar[str] = "bert"

    def __init__(self, config: BERTConfig) -> None:
        super().__init__(config)
        self.bert = BERTModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

    @override
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        start_positions: Tensor | None = None,
        end_positions: Tensor | None = None,
    ) -> QuestionAnsweringOutput:
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        logits = self.qa_outputs(outputs.last_hidden_state)
        # ``logits`` is (B, T, 2); split along the last dim into start / end
        # and return them as named fields rather than making every caller
        # remember which trailing index is which.
        start_logits = logits[..., 0]  # (B, T)
        end_logits = logits[..., 1]  # (B, T)

        loss: Tensor | None = None
        if start_positions is not None and end_positions is not None:
            # Callers are responsible for keeping span positions inside [0, T).
            # SQuAD's sliding window (§4.2) puts some answers outside the
            # current doc-stride span.  The reference clamps those positions
            # to a sentinel at the sequence end and ignores that index in the
            # loss, so an unanswerable window contributes zero rather than
            # training the model towards an arbitrary in-window token.
            ignored = int(start_logits.shape[1])
            start_t = start_positions.long().clip(min=0, max=ignored)
            end_t = end_positions.long().clip(min=0, max=ignored)
            # A batch in which *every* window is unanswerable leaves the mean
            # with no terms to average, which surfaces as NaN and would poison
            # the whole step.  Zero is the honest value: nothing was asked.
            # It stays in the graph, with a zero gradient, so the step that
            # calls ``backward()`` on it runs like any other.
            n_valid = float((start_t != ignored).float().sum().item())
            if n_valid == 0.0:
                loss = (start_logits.sum() + end_logits.sum()) * 0.0
            else:
                loss = (
                    F.cross_entropy(start_logits, start_t, ignore_index=ignored)
                    + F.cross_entropy(end_logits, end_t, ignore_index=ignored)
                ) / 2.0

        return QuestionAnsweringOutput(start_logits=start_logits, end_logits=end_logits, loss=loss)


# ─────────────────────────────────────────────────────────────────────────────
# Additional task heads — pre-training, NSP, causal-LM
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class BERTForPreTrainingOutput(ModelOutput):
    r"""Combined output for :class:`BERTForPreTraining`.

    Aggregates the masked-LM logits, next-sentence-prediction logits, and
    optional per-objective and combined losses produced by the full BERT
    pre-training pipeline of Devlin et al., 2018.

    Parameters
    ----------
    prediction_logits : Tensor
        MLM head logits of shape ``(B, T, vocab_size)`` — one distribution
        over the WordPiece vocabulary per input position.
    seq_relationship_logits : Tensor
        NSP head logits of shape ``(B, 2)`` — binary IsNext / NotNext scores
        derived from the pooled ``[CLS]`` embedding.
    loss : Tensor or None, default=None
        Sum of ``mlm_loss`` and ``nsp_loss`` when both are available;
        otherwise the single available loss, or ``None`` if neither label
        set was supplied.
    mlm_loss : Tensor or None, default=None
        Cross-entropy on masked positions when ``labels`` was supplied.
    nsp_loss : Tensor or None, default=None
        Binary cross-entropy on the NSP head when ``next_sentence_label``
        was supplied.

    Notes
    -----
    Reference: Devlin, Chang, Lee, and Toutanova, *"BERT: Pre-training of
    Deep Bidirectional Transformers for Language Understanding"*, NAACL 2019
    (arXiv:1810.04805) §3.1.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.bert import BERTConfig, BERTForPreTraining
    >>> cfg = BERTConfig(num_hidden_layers=2, hidden_size=128,
    ...                  num_attention_heads=2, intermediate_size=512)
    >>> model = BERTForPreTraining(cfg).eval()
    >>> input_ids = lucid.tensor([[101, 7592, 102, 2088, 102]])
    >>> out = model(input_ids)
    >>> out.prediction_logits.shape, out.seq_relationship_logits.shape
    ((1, 5, 30522), (1, 2))
    """

    prediction_logits: Tensor
    seq_relationship_logits: Tensor
    loss: Tensor | None = None
    mlm_loss: Tensor | None = None
    nsp_loss: Tensor | None = None


@final
class _BERTOnlyNSPHead(nn.Module):
    def __init__(self, config: BERTConfig) -> None:
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    @override
    def forward(self, pooled_output: Tensor) -> Tensor:
        return self.seq_relationship(pooled_output)


@final
class _BERTPreTrainingHeads(nn.Module):
    """MLM prediction head + NSP head — used by :class:`BERTForPreTraining`."""

    def __init__(self, config: BERTConfig) -> None:
        super().__init__()
        self.predictions = _BERTLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    @override
    def forward(self, sequence_output: Tensor, pooled_output: Tensor) -> tuple[Tensor, Tensor]:
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BERTForPreTraining(SequenceClassificationModel, MaskedLMMixin):
    r"""BERT with the original joint MLM + NSP pre-training objective.

    Combines the masked-language-modeling head (decoder weight tied to input
    embeddings when ``config.tie_word_embeddings`` is True) with the
    next-sentence-prediction head on top of the pooled ``[CLS]`` embedding.
    This is the exact head configuration used in Devlin et al., 2018 to
    train BERT-Base and BERT-Large from scratch.

    Supply ``labels`` (MLM targets) and/or ``next_sentence_label`` (binary
    NSP target) to compute the corresponding losses; their sum is exposed as
    ``output.loss``.  Use this class only when reproducing the original
    pre-training recipe — newer encoder-only LMs typically drop NSP and use
    :class:`BERTForMaskedLM` directly.

    Parameters
    ----------
    config : BERTConfig
        BERT hyperparameters.  ``config.tie_word_embeddings`` (default True)
        controls whether the MLM decoder weight is tied to the input
        embedding matrix.

    Attributes
    ----------
    bert : BERTModel
        Underlying bidirectional encoder trunk.
    cls : nn.Module
        Combined head holding both the MLM prediction projection
        (``cls.predictions``) and the NSP binary linear
        (``cls.seq_relationship``).

    Notes
    -----
    Reference: Devlin, Chang, Lee, and Toutanova, *"BERT: Pre-training of
    Deep Bidirectional Transformers for Language Understanding"*, NAACL 2019
    (arXiv:1810.04805) §3.1.

    The combined loss when both objectives are supplied is

    .. math::

        \mathcal{L}_{\mathrm{pretrain}}
            = \mathcal{L}_{\mathrm{MLM}} + \mathcal{L}_{\mathrm{NSP}}.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.bert import BERTConfig, BERTForPreTraining
    >>> cfg = BERTConfig(num_hidden_layers=2, hidden_size=128,
    ...                  num_attention_heads=2, intermediate_size=512)
    >>> model = BERTForPreTraining(cfg).eval()
    >>> input_ids = lucid.tensor([[101, 7592, 103, 102, 2088, 102]])
    >>> out = model(input_ids)
    >>> out.prediction_logits.shape    # MLM logits  (B=1, T=6, V=30522)
    (1, 6, 30522)
    >>> out.seq_relationship_logits.shape   # NSP logits  (B=1, 2)
    (1, 2)
    """

    config_class: ClassVar[type[BERTConfig]] = BERTConfig
    base_model_prefix: ClassVar[str] = "bert"

    def __init__(self, config: BERTConfig) -> None:
        super().__init__(config)
        self.bert = BERTModel(config)
        self.cls = _BERTPreTrainingHeads(config)
        if config.tie_word_embeddings:
            self._tie_word_embeddings()

    @override
    def _tie_word_embeddings(self) -> None:
        self.cls.predictions.tie_to(self.bert.embeddings.word_embeddings.weight)

    @override
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        labels: Tensor | None = None,
        next_sentence_label: Tensor | None = None,
    ) -> BERTForPreTrainingOutput:
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        prediction_scores, seq_relationship_score = self.cls(
            outputs.last_hidden_state, outputs.pooler_output
        )

        mlm_loss: Tensor | None = None
        nsp_loss: Tensor | None = None
        total_loss: Tensor | None = None
        if labels is not None:
            mlm_loss = self.compute_lm_loss(prediction_scores, labels)
        if next_sentence_label is not None:
            nsp_loss = F.cross_entropy(seq_relationship_score, next_sentence_label.long())
        if mlm_loss is not None and nsp_loss is not None:
            total_loss = mlm_loss + nsp_loss
        elif mlm_loss is not None:
            total_loss = mlm_loss
        elif nsp_loss is not None:
            total_loss = nsp_loss

        return BERTForPreTrainingOutput(
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            loss=total_loss,
            mlm_loss=mlm_loss,
            nsp_loss=nsp_loss,
        )


class BERTForNextSentencePrediction(SequenceClassificationModel):
    r"""BERT with the standalone next-sentence-prediction head.

    Wraps the bidirectional encoder with a single binary linear classifier
    operating on the pooled ``[CLS]`` embedding.  This is pre-training task 2
    of Devlin et al., 2018 §3.1 in isolation — useful for reproducing
    historical experiments or as a sanity check for sentence-pair coherence.

    NSP was abandoned by RoBERTa, ALBERT, and DeBERTa as offering no
    downstream value, so prefer :class:`BERTForMaskedLM` (MLM-only) or
    :class:`BERTForSequenceClassification` for new work.

    Parameters
    ----------
    config : BERTConfig
        BERT hyperparameters.

    Attributes
    ----------
    bert : BERTModel
        Underlying bidirectional encoder trunk.
    cls : nn.Module
        NSP head — a single ``Linear(hidden_size, 2)`` over the pooled
        embedding.

    Notes
    -----
    Reference: Devlin, Chang, Lee, and Toutanova, *"BERT: Pre-training of
    Deep Bidirectional Transformers for Language Understanding"*, NAACL 2019
    (arXiv:1810.04805) §3.1 (task #2).

    When ``labels`` is provided, the loss is the binary cross-entropy

    .. math::

        \mathcal{L}_{\mathrm{NSP}}
            = -\frac{1}{B}\sum_{b=1}^{B}
              \log p_{\theta}(y_b \mid x_b^{(A)}, x_b^{(B)}),

    where :math:`y_b \in \{0, 1\}` denotes IsNext vs. NotNext.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.bert import BERTConfig, BERTForNextSentencePrediction
    >>> cfg = BERTConfig(num_hidden_layers=2, hidden_size=128,
    ...                  num_attention_heads=2, intermediate_size=512)
    >>> model = BERTForNextSentencePrediction(cfg).eval()
    >>> input_ids = lucid.tensor([[101, 7592, 102, 2088, 102]])
    >>> token_type_ids = lucid.tensor([[0, 0, 0, 1, 1]])
    >>> out = model(input_ids, token_type_ids=token_type_ids)
    >>> out.logits.shape   # (B=1, 2)
    (1, 2)
    """

    config_class: ClassVar[type[BERTConfig]] = BERTConfig
    base_model_prefix: ClassVar[str] = "bert"

    def __init__(self, config: BERTConfig) -> None:
        super().__init__(config)
        self.bert = BERTModel(config)
        self.cls = _BERTOnlyNSPHead(config)

    @override
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> SequenceClassificationOutput:
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        seq_relationship_score = self.cls(outputs.pooler_output)

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(seq_relationship_score, labels.long())

        return SequenceClassificationOutput(logits=seq_relationship_score, loss=loss)


class BERTForCausalLM(LanguageModelingModel):
    r"""BERT trunk repurposed as a left-to-right (causal) language model.

    Standard BERT attends bidirectionally; this wrapper injects a
    lower-triangular causal mask on top of the existing additive
    attention/padding mask so the same encoder weights behave as a decoder.
    The LM head is the same tied projection used by
    :class:`BERTForMaskedLM`.  Use this class when you want to apply
    pre-trained BERT weights to a generative or sequence-continuation task.

    Parameters
    ----------
    config : BERTConfig
        BERT hyperparameters.  ``config.tie_word_embeddings`` (default True)
        ties the LM decoder weight to the input embedding matrix.

    Attributes
    ----------
    bert : BERTModel
        Underlying transformer trunk; only ``embeddings`` and ``encoder`` are
        invoked in ``forward`` (the pooler is bypassed).
    cls : nn.Module
        Tied LM prediction head — same architecture as
        :class:`BERTForMaskedLM`.

    Notes
    -----
    Reference: Devlin, Chang, Lee, and Toutanova, *"BERT: Pre-training of
    Deep Bidirectional Transformers for Language Understanding"*, NAACL 2019
    (arXiv:1810.04805); causal adaptation follows the standard left-to-right
    LM masking scheme.

    The additive causal mask :math:`M \in \mathbb{R}^{T \times T}` satisfies

    .. math::

        M_{ij} =
        \begin{cases}
            0, & j \le i \\
            -10^{4}, & j > i
        \end{cases}

    and is broadcast against a padding mask when present.  Loss (when
    ``labels`` is supplied) uses the standard next-token shift:

    .. math::

        \mathcal{L}_{\mathrm{CLM}}
            = -\frac{1}{B(T-1)} \sum_{b,t}
              \log p_{\theta}(y_{b, t+1} \mid x_{b,\le t}),

    with positions labelled ``-100`` excluded.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.bert import BERTConfig, BERTForCausalLM
    >>> cfg = BERTConfig(num_hidden_layers=2, hidden_size=128,
    ...                  num_attention_heads=2, intermediate_size=512)
    >>> model = BERTForCausalLM(cfg).eval()
    >>> input_ids = lucid.tensor([[101, 7592, 2088, 102]])
    >>> out = model(input_ids)
    >>> out.logits.shape   # (B=1, T=4, V=30522)
    (1, 4, 30522)
    """

    config_class: ClassVar[type[BERTConfig]] = BERTConfig
    base_model_prefix: ClassVar[str] = "bert"

    def __init__(self, config: BERTConfig) -> None:
        super().__init__(config)
        self.bert = BERTModel(config)
        self.cls = _BERTOnlyMLMHead(config)
        if config.tie_word_embeddings:
            self._tie_word_embeddings()

    @override
    def _tie_word_embeddings(self) -> None:
        self.cls.predictions.tie_to(self.bert.embeddings.word_embeddings.weight)

    def _causal_attention_mask(
        self,
        attention_mask: Tensor | None,
        B: int,
        T: int,
        device: str,
    ) -> Tensor:
        """Combine a ``(B, T)`` padding mask with a lower-triangular causal
        mask, returning the additive ``(B, 1, T, T)`` form.
        """
        causal = lucid.tril(lucid.ones((T, T), device=device))  # (T, T)
        causal_add = (1.0 - causal) * -1e4  # 0 / -1e4
        causal_add = causal_add.reshape(1, 1, T, T)
        if attention_mask is None:
            return causal_add
        pad_add = (1.0 - attention_mask.float()) * -1e4  # (B, T)
        pad_add = pad_add.reshape(B, 1, 1, T)
        return causal_add + pad_add

    @override
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> CausalLMOutput:
        B, T = int(input_ids.shape[0]), int(input_ids.shape[1])
        dev = input_ids.device.type
        ext_mask = self._causal_attention_mask(attention_mask, B, T, dev)

        hidden = self.bert.embeddings(input_ids, token_type_ids=token_type_ids)
        sequence_output, _, _ = self.bert.encoder(hidden, attention_mask=ext_mask)
        prediction_scores = self.cls(sequence_output)

        loss: Tensor | None = None
        if labels is not None:
            # Standard causal-LM shift: predict token t+1 from positions [0..t].
            B_, T_, V = prediction_scores.shape
            shift_logits = prediction_scores[:, :-1, :].reshape(B_ * (T_ - 1), V)
            shift_labels = labels[:, 1:].reshape(B_ * (T_ - 1)).long()
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

        return CausalLMOutput(logits=prediction_scores, loss=loss)
