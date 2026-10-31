"""RoFormer model (Su et al., 2021) — BERT with rotary position embedding.

Architecturally identical to :class:`lucid.models.text.bert.BERTModel` except
that:

    * No additive position embedding (``position_embeddings`` is removed).
    * Inside every self-attention layer, ``q`` and ``k`` are rotated by
      :func:`lucid.nn.functional.apply_rotary_emb` before the dot product.

State-dict key naming mirrors HF ``RoFormerModel`` so future weight ports
amount to a flat rename.
"""

import math
from typing import ClassVar, final, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import ModelConfig, PretrainedModel
from lucid.models._tasks import (
    LanguageModelingModel,
    SequenceClassificationModel,
    TokenClassificationModel,
)
from lucid.models._mixins import MaskedLMMixin
from lucid.models._output import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    MaskedLMOutput,
    QuestionAnsweringOutput,
    SequenceClassificationOutput,
    TokenClassificationOutput,
)
from lucid.models._utils._text import (
    compute_qa_loss,
    embed_tokens,
    extended_attention_mask,
    flatten_multiple_choice_inputs,
    init_text_module_weights,
    kv_prefix,
    last_kept_token,
    prepend_kv_prefix,
    text_activation,
)
from lucid.nn.functional import apply_rotary_emb
from lucid.nn.module import Module
from lucid.models.text.roformer._config import RoFormerConfig
from lucid.utils.cache import Cache


class _RoFormerInitMixin:
    config: ModelConfig

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range if isinstance(self.config, RoFormerConfig) else 0.02
        init_text_module_weights(module, std, reset_layer_norm=True)


# ─────────────────────────────────────────────────────────────────────────────
# Rotary tables — *interleaved* (original RoPE) layout, matching the RoFormer
# reference checkpoints.  Distinct from the shared LLaMA-style
# :class:`lucid.nn.RotaryEmbedding` (half-split layout): here each frequency
# is repeated twice so cos / sin line up with the consecutive feature pairs
# ``(x_{2i}, x_{2i+1})`` that ``apply_rotary_emb(..., interleaved=True)`` rotates.
# ─────────────────────────────────────────────────────────────────────────────


@final
class _RoFormerRotaryEmbedding(Module):
    """Precompute interleaved-layout ``(cos, sin)`` tables for RoPE.

    Each table has shape ``(max_position_embeddings, head_dim)`` and repeats
    every frequency twice along the last dimension:

        cos[p] = [cos(p·θ_0), cos(p·θ_0), cos(p·θ_1), cos(p·θ_1), ...]
        sin[p] = [sin(p·θ_0), sin(p·θ_0), sin(p·θ_1), sin(p·θ_1), ...]

    with θ_i = base^(-2 i / head_dim).  This is the layout consumed by
    :func:`lucid.nn.functional.apply_rotary_emb` with ``interleaved=True``.
    """

    cos_cached: Tensor
    sin_cached: Tensor

    def __init__(self, head_dim: int, max_position_embeddings: int, base: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"_RoFormerRotaryEmbedding requires an even head_dim, got {head_dim}")
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        half = head_dim // 2
        cos_rows: list[list[float]] = []
        sin_rows: list[list[float]] = []
        for p in range(max_position_embeddings):
            cos_row: list[float] = [0.0] * head_dim
            sin_row: list[float] = [0.0] * head_dim
            for i in range(half):
                theta = p / (base ** (2.0 * i / head_dim))
                c = math.cos(theta)
                s = math.sin(theta)
                cos_row[2 * i] = c
                cos_row[2 * i + 1] = c
                sin_row[2 * i] = s
                sin_row[2 * i + 1] = s
            cos_rows.append(cos_row)
            sin_rows.append(sin_row)

        self.register_buffer("cos_cached", lucid.tensor(cos_rows), persistent=False)
        self.register_buffer("sin_cached", lucid.tensor(sin_rows), persistent=False)

    @override
    def forward(self) -> tuple[Tensor, Tensor]:
        return self.cos_cached, self.sin_cached


# ─────────────────────────────────────────────────────────────────────────────
# Embeddings — *no* position embedding (RoPE handles position implicitly)
# ─────────────────────────────────────────────────────────────────────────────


@final
class _RoFormerEmbeddings(nn.Module):
    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__()
        emb_dim = config.embedding_size or config.hidden_size
        self.word_embeddings = nn.Embedding(
            config.vocab_size, emb_dim, padding_idx=config.pad_token_id
        )
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, emb_dim)
        self.LayerNorm = nn.LayerNorm(emb_dim, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout)

    @override
    def forward(
        self,
        input_ids: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
    ) -> Tensor:
        words = embed_tokens(self.word_embeddings, input_ids, inputs_embeds)
        B, T = int(words.shape[0]), int(words.shape[1])
        dev = words.device.type
        if token_type_ids is None:
            token_type_ids = lucid.zeros((B, T), device=dev).long()
        types = self.token_type_embeddings(token_type_ids)

        emb = self.LayerNorm(words + types)
        return self.dropout(emb)


# ─────────────────────────────────────────────────────────────────────────────
# Rotary self-attention
# ─────────────────────────────────────────────────────────────────────────────


@final
class _RoFormerSelfAttention(nn.Module):
    """Multi-head self-attention with RoPE applied to Q and K."""

    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = math.sqrt(self.head_dim)

        self.query = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.key = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.value = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.dropout = nn.Dropout(p=config.attention_dropout)
        self.rotary_value = config.rotary_value

    def _shape(self, x: Tensor, B: int, T: int) -> Tensor:
        return x.reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    @override
    def forward(
        self,
        hidden: Tensor,
        cos: Tensor,
        sin: Tensor,
        attention_mask: Tensor | None = None,
        prefix: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        B, T, _ = hidden.shape
        q = self._shape(self.query(hidden), B, T)
        k = self._shape(self.key(hidden), B, T)
        v = self._shape(self.value(hidden), B, T)

        # Rotate Q / K by the absolute position phase.  RoFormer uses the
        # original interleaved RoPE pairing (x_2i, x_2i+1).
        q, k = apply_rotary_emb(q, k, cos, sin, interleaved=True)
        if self.rotary_value:
            # ``rotary_value``: the reference optionally rotates V by the same
            # phase.  ``apply_rotary_emb`` rotates a pair, so V is passed
            # twice and only the first result is kept.
            v, _ = apply_rotary_emb(v, v, cos, sin, interleaved=True)

        # A prefix (prefix tuning) is key/value states with no position: it goes in
        # after the rotation, so it is never rotated, and the input's phases start at 0.
        k, v = prepend_kv_prefix(k, v, prefix)

        # One masked-attention core for every path (RoPE is already applied,
        # so it sees rotated q/k).  A hand-written softmax took the dropout
        # path and turned a fully masked query row into NaN where the fused
        # kernel gives ``0`` (LCD-387).
        ctx: Tensor = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            scale=1.0 / self.scale,
        )
        return ctx.permute(0, 2, 1, 3).reshape(B, T, self.num_heads * self.head_dim)


@final
class _RoFormerSelfOutput(nn.Module):
    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout)

    @override
    def forward(self, hidden: Tensor, input_tensor: Tensor) -> Tensor:
        h = self.dropout(self.dense(hidden))
        return self.LayerNorm(h + input_tensor)


@final
class _RoFormerAttention(nn.Module):
    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__()
        self.self = _RoFormerSelfAttention(config)
        self.output = _RoFormerSelfOutput(config)

    @override
    def forward(
        self,
        hidden: Tensor,
        cos: Tensor,
        sin: Tensor,
        attention_mask: Tensor | None = None,
        prefix: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        attn_out = self.self(hidden, cos, sin, attention_mask=attention_mask, prefix=prefix)
        return self.output(attn_out, hidden)


@final
class _RoFormerIntermediate(nn.Module):
    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self._act_name = config.hidden_act

    @override
    def forward(self, x: Tensor) -> Tensor:
        return text_activation(self._act_name, self.dense(x))


@final
class _RoFormerOutput(nn.Module):
    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout)

    @override
    def forward(self, hidden: Tensor, input_tensor: Tensor) -> Tensor:
        h = self.dropout(self.dense(hidden))
        return self.LayerNorm(h + input_tensor)


@final
class _RoFormerLayer(nn.Module):
    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__()
        self.attention = _RoFormerAttention(config)
        self.intermediate = _RoFormerIntermediate(config)
        self.output = _RoFormerOutput(config)

    @override
    def forward(
        self,
        hidden: Tensor,
        cos: Tensor,
        sin: Tensor,
        prefix: tuple[Tensor, Tensor] | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        attn_out = self.attention(hidden, cos, sin, attention_mask=attention_mask, prefix=prefix)
        inter = self.intermediate(attn_out)
        return self.output(inter, attn_out)


@final
class _RoFormerEncoder(nn.Module):
    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__()
        self.layer = nn.ModuleList(
            [_RoFormerLayer(config) for _ in range(config.num_hidden_layers)]
        )

    @override
    def forward(
        self,
        hidden: Tensor,
        cos: Tensor,
        sin: Tensor,
        attention_mask: Tensor | None = None,
        past_key_values: Cache | None = None,
    ) -> Tensor:
        for i, layer in enumerate(self.layer):
            hidden = layer(
                hidden,
                cos,
                sin,
                attention_mask=attention_mask,
                prefix=kv_prefix(past_key_values, i),
            )
        return hidden


@final
class _RoFormerPooler(nn.Module):
    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

    @override
    def forward(self, hidden: Tensor) -> Tensor:
        cls_hidden = hidden[:, 0]
        return F.tanh(self.dense(cls_hidden))


@final
class _RoFormerClassificationHead(nn.Module):
    """The reference's ``RoFormerClassificationHead``: dropout, ``dense``,
    activation, dropout, ``out_proj`` on the first token.

    It is its own module (mounted as ``classifier``) so the head's two linears
    sit at ``classifier.dense`` / ``classifier.out_proj`` — the names a
    reference checkpoint or a hub adapter uses, and a prefix a fine-tuning
    wrapper can address without also matching the encoder's ``*.dense``.
    """

    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__()
        drop = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout
        )
        self._act_name = config.hidden_act
        self.dropout = nn.Dropout(p=drop)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    @override
    def forward(self, hidden: Tensor) -> Tensor:
        x = self.dropout(hidden[:, 0, :])
        x = text_activation(self._act_name, self.dense(x))
        return self.out_proj(self.dropout(x))


def _migrate_flat_classification_head(
    module: nn.Module,
    state_dict: dict[str, Tensor],
    prefix: str,
    *_hook_args: object,
) -> None:
    """Load-time key migration for checkpoints saved before the head was nested.

    The old layout kept the head's two linears at the top level (``dense.*``
    and ``classifier.*``).  They become ``classifier.dense.*`` /
    ``classifier.out_proj.*``, so a checkpoint written by an earlier release
    still loads strictly.  (Its unused ``roformer.pooler.*`` is dropped by the
    trunk's own hook.)  A checkpoint already in the new layout is left alone.
    """
    del module
    if f"{prefix}dense.weight" not in state_dict:
        return
    # ``classifier.*`` must move before ``dense.*`` takes the ``classifier.`` prefix.
    for leaf in ("weight", "bias"):
        if f"{prefix}classifier.{leaf}" in state_dict:
            state_dict[f"{prefix}classifier.out_proj.{leaf}"] = state_dict.pop(
                f"{prefix}classifier.{leaf}"
            )
        if f"{prefix}dense.{leaf}" in state_dict:
            state_dict[f"{prefix}classifier.dense.{leaf}"] = state_dict.pop(f"{prefix}dense.{leaf}")


def _drop_absent_pooler(
    module: nn.Module,
    state_dict: dict[str, Tensor],
    prefix: str,
    *_hook_args: object,
) -> None:
    """Let a pooler-less trunk load a checkpoint that carries a pooler.

    The shipped encoder weights (and any checkpoint a head saved before it
    stopped building the pooler) include ``pooler.*``; the encoder-only trunk
    has nowhere to put them and the pooler never affects ``last_hidden_state``,
    so they are discarded.
    """
    del module
    for key in [k for k in state_dict if k.startswith(f"{prefix}pooler.")]:
        del state_dict[key]


# ─────────────────────────────────────────────────────────────────────────────
# Top-level encoder + heads
# ─────────────────────────────────────────────────────────────────────────────


class _RoFormerTrunk(_RoFormerInitMixin, PretrainedModel):
    """Embeddings, rotary tables and encoder stack — what every RoFormer model shares.

    The reference trunk has no pooler.  :class:`RoFormerModel` adds one (the
    shipped base / masked-LM checkpoints carry its keys); a head that reads
    the raw hidden states builds :class:`_RoFormerEncoderOnly` instead, so no
    parameter is left that its loss can never reach.
    """

    config_class: ClassVar[type[RoFormerConfig]] = RoFormerConfig
    base_model_prefix: ClassVar[str] = "roformer"
    kv_prefix_only: ClassVar[bool] = True

    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__(config)
        head_dim = config.hidden_size // config.num_attention_heads
        self._max_pos = config.max_position_embeddings
        self.embeddings = _RoFormerEmbeddings(config)
        # Only present when the embedding is factorised, matching the
        # reference's ``if config.embedding_size != config.hidden_size``.
        emb_dim = config.embedding_size or config.hidden_size
        self.embeddings_project: nn.Linear | None = (
            nn.Linear(emb_dim, config.hidden_size) if emb_dim != config.hidden_size else None
        )
        self.rotary = _RoFormerRotaryEmbedding(
            head_dim=head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rotary_base,
        )
        self.encoder = _RoFormerEncoder(config)

    @override
    def get_input_embeddings(self) -> nn.Module:
        return self.embeddings.word_embeddings

    @override
    def set_input_embeddings(self, value: nn.Module) -> None:
        if not isinstance(value, nn.Embedding):
            raise TypeError(
                f"RoFormerModel input embeddings must be nn.Embedding, got {type(value).__name__}"
            )
        self.embeddings.word_embeddings = value

    def _encode(
        self,
        input_ids: Tensor | None,
        attention_mask: Tensor | None,
        token_type_ids: Tensor | None,
        inputs_embeds: Tensor | None,
        past_key_values: Cache | None = None,
    ) -> Tensor:
        hidden = self.embeddings(
            input_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        batch_size, seq_len = int(hidden.shape[0]), int(hidden.shape[1])
        if seq_len > self._max_pos:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_position_embeddings {self._max_pos}"
            )

        cos, sin = self.rotary()
        # Slice the precomputed tables to the current sequence length so we
        # don't carry around (max_pos, head_dim) when the prompt is short.
        cos = cos[:seq_len]
        sin = sin[:seq_len]

        if self.embeddings_project is not None:
            hidden = self.embeddings_project(hidden)
        prefix_len = 0 if past_key_values is None else past_key_values.get_seq_length()
        ext_mask = extended_attention_mask(
            attention_mask, (batch_size, seq_len), key_length=prefix_len + seq_len
        )
        return self.encoder(
            hidden, cos, sin, attention_mask=ext_mask, past_key_values=past_key_values
        )


@final
class _RoFormerEncoderOnly(_RoFormerTrunk):
    """The trunk without a pooler, for heads that read the raw hidden states."""

    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__(config)
        self.apply(self._init_weights)
        self.register_load_state_dict_pre_hook(_drop_absent_pooler)

    @override
    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        *,
        past_key_values: Cache | None = None,
        inputs_embeds: Tensor | None = None,
    ) -> BaseModelOutput:
        return BaseModelOutput(
            last_hidden_state=self._encode(
                input_ids, attention_mask, token_type_ids, inputs_embeds, past_key_values
            )
        )


class RoFormerModel(_RoFormerTrunk, PretrainedModel):
    r"""Bare RoFormer encoder trunk with rotary position embeddings.

    Architecturally a BERT-base encoder (12 layers, 768 hidden, 12 heads,
    intermediate width 3072) where the absolute learned position embedding
    is removed and rotary position embedding (RoPE) is applied to the query
    and key vectors inside every self-attention layer.  RoPE injects
    relative-position information **multiplicatively** inside attention so
    the dot product :math:`\langle R_m q, R_n k \rangle` depends only on
    the offset :math:`m - n` — granting better length extrapolation and
    composing naturally with linear attention.

    Parameters
    ----------
    config : RoFormerConfig
        Hyperparameters controlling vocabulary, depth, width, head count,
        and the RoPE ``rotary_base``.  The per-head dimension
        ``hidden_size / num_attention_heads`` must be even (enforced by
        :class:`RoFormerConfig`).

    Attributes
    ----------
    embeddings : nn.Module
        Token + token-type embedding block followed by LayerNorm and
        dropout.  Notably *no* learned position embedding.
    rotary : Module
        Pre-computed interleaved-layout ``(cos, sin)`` tables of shape
        ``(max_position_embeddings, head_dim)``, sliced to the actual
        sequence length at each ``forward``.  Uses the original RoPE
        pairing ``(x_2i, x_2i+1)`` (each frequency repeated twice) rather
        than the shared half-split layout, matching the RoFormer reference
        checkpoints.
    encoder : nn.Module
        Stack of ``config.num_hidden_layers`` rotary-attention transformer
        layers.
    pooler : nn.Module
        Dense + tanh projection of the first token's hidden state, kept so
        the shipped checkpoints load.  The task heads that read the raw
        hidden states (sequence / token classification, multiple choice,
        question answering) build the trunk without it.

    Notes
    -----
    Reference: Su, Lu, Pan, Murtadha, Wen, and Liu, *"RoFormer: Enhanced
    Transformer with Rotary Position Embedding"*, Neurocomputing,
    vol. 568, 2024, article 127063 (arXiv:2104.09864).

    Rotary rotation per :math:`(\cos\theta_i, \sin\theta_i)` pair, with
    :math:`\theta_i = \mathrm{base}^{-2i/d_{\text{head}}}`:

    .. math::

        R_\theta = \begin{pmatrix}
            \cos\theta & -\sin\theta \\
            \sin\theta & \cos\theta
        \end{pmatrix}.

    Applied independently to every :math:`(x_{2i}, x_{2i+1})` pair of
    :math:`Q` and :math:`K` so :math:`q_m^\top k_n` becomes a function of
    :math:`m - n` only.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.roformer import RoFormerConfig, RoFormerModel
    >>> cfg = RoFormerConfig(num_hidden_layers=2, hidden_size=128,
    ...                      num_attention_heads=2, intermediate_size=512)
    >>> model = RoFormerModel(cfg).eval()
    >>> input_ids = lucid.tensor([[101, 7592, 2088, 102]])
    >>> out = model(input_ids)
    >>> out.last_hidden_state.shape, out.pooler_output.shape
    ((1, 4, 128), (1, 128))
    """

    config_class: ClassVar[type[RoFormerConfig]] = RoFormerConfig
    base_model_prefix: ClassVar[str] = "roformer"

    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__(config)
        # The reference trunk has no pooler; this one keeps it so the shipped
        # base / masked-LM checkpoints (which carry its keys) still load.
        self.pooler = _RoFormerPooler(config)
        self.apply(self._init_weights)

    @override
    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        *,
        inputs_embeds: Tensor | None = None,
        past_key_values: Cache | None = None,
    ) -> BaseModelOutputWithPooling:
        sequence_output = self._encode(
            input_ids, attention_mask, token_type_ids, inputs_embeds, past_key_values
        )
        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output, pooler_output=self.pooler(sequence_output)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Task heads — mirror the BERT set since RoFormer is encoder-only
# ─────────────────────────────────────────────────────────────────────────────


@final
class _RoFormerPredictionHeadTransform(nn.Module):
    def __init__(self, config: RoFormerConfig) -> None:
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
class _RoFormerLMPredictionHead(nn.Module):
    bias: Tensor

    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__()
        self.transform = _RoFormerPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
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
class _RoFormerOnlyMLMHead(nn.Module):
    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__()
        self.predictions = _RoFormerLMPredictionHead(config)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.predictions(x)


class RoFormerForMaskedLM(_RoFormerInitMixin, LanguageModelingModel, MaskedLMMixin):
    r"""RoFormer with a tied masked-language-modeling head.

    Wraps :class:`RoFormerModel` with the BERT-style two-layer projection
    (dense + activation + LayerNorm) mapping each hidden state to
    vocabulary logits.  The decoder weight is tied to the input
    ``word_embeddings`` table when ``config.tie_word_embeddings`` is
    ``True``.  Use for MLM pre-training when RoPE-based relative position
    information is preferred to absolute position embeddings.

    Parameters
    ----------
    config : RoFormerConfig
        Hyperparameters.  ``config.tie_word_embeddings`` (default True)
        controls whether the MLM decoder weight is bound to the input
        embedding matrix.

    Attributes
    ----------
    roformer : RoFormerModel
        Underlying RoPE encoder trunk.
    cls : nn.Module
        Masked-LM prediction head — dense + LayerNorm transform plus a
        decoder of shape ``(hidden_size, vocab_size)``.

    Notes
    -----
    Reference: Su, Lu, Pan, Murtadha, Wen, and Liu, *"RoFormer: Enhanced
    Transformer with Rotary Position Embedding"*, Neurocomputing,
    vol. 568, 2024 (arXiv:2104.09864).

    When ``labels`` is provided, the loss is the standard masked
    cross-entropy

    .. math::

        \mathcal{L}_{\mathrm{MLM}} = -\frac{1}{|M|}
            \sum_{i \in M} \log p_\theta(x_i \mid x_{\setminus M}),

    over the set :math:`M` of masked positions (positions whose label
    equals ``-100`` are excluded).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.roformer import RoFormerConfig, RoFormerForMaskedLM
    >>> cfg = RoFormerConfig(num_hidden_layers=2, hidden_size=128,
    ...                      num_attention_heads=2, intermediate_size=512)
    >>> model = RoFormerForMaskedLM(cfg).eval()
    >>> input_ids = lucid.tensor([[101, 7592, 103, 102]])   # [CLS] hello [MASK] [SEP]
    >>> out = model(input_ids)
    >>> out.logits.shape   # (B=1, T=4, V=50000)
    (1, 4, 50000)
    """

    config_class: ClassVar[type[RoFormerConfig]] = RoFormerConfig
    base_model_prefix: ClassVar[str] = "roformer"

    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__(config)
        self.roformer = RoFormerModel(config)
        self.cls = _RoFormerOnlyMLMHead(config)
        self.apply(self._init_weights)
        if config.tie_word_embeddings:
            self._tie_word_embeddings()

    @override
    def _tie_word_embeddings(self) -> None:
        weight = self.roformer.embeddings.word_embeddings.weight
        self.cls.predictions.tie_to(weight)

    @override
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> MaskedLMOutput:
        outputs = self.roformer(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        prediction_scores = self.cls(outputs.last_hidden_state)

        loss: Tensor | None = None
        if labels is not None:
            loss = self.compute_lm_loss(prediction_scores, labels)

        return MaskedLMOutput(logits=prediction_scores, loss=loss)


class RoFormerForSequenceClassification(_RoFormerInitMixin, SequenceClassificationModel):
    r"""RoFormer with a pooled-CLS sequence-classification head.

    Wraps :class:`RoFormerModel` (built without its pooler, which this head
    never reads) with the reference's own classification head: the raw CLS
    hidden state passes through dropout, a ``hidden_size -> hidden_size``
    dense, the configured activation, dropout again, and finally the
    ``num_labels`` projection.  Use for GLUE-style fine-tunes when the
    RoPE-based relative position bias is preferred.

    Parameters
    ----------
    config : RoFormerConfig
        Hyperparameters.  ``config.num_labels`` sets the output dimension;
        ``config.classifier_dropout`` (falling back to ``hidden_dropout``)
        sets the dropout applied before the linear.

    Attributes
    ----------
    roformer : RoFormerModel
        Underlying RoPE encoder trunk.
    classifier : nn.Module
        The reference's classification head: dropout, ``classifier.dense``
        (``hidden_size -> hidden_size``), activation, dropout, then
        ``classifier.out_proj`` (``hidden_size -> num_labels``) producing
        per-class logits.

    Notes
    -----
    Reference: Su, Lu, Pan, Murtadha, Wen, and Liu, *"RoFormer: Enhanced
    Transformer with Rotary Position Embedding"*, Neurocomputing,
    vol. 568, 2024 (arXiv:2104.09864).

    Checkpoints saved before the head was nested (top-level ``dense.*`` /
    ``classifier.*``, plus an unused ``roformer.pooler.*``) are migrated on
    ``load_state_dict``.

    Loss (when ``labels`` is provided) is the standard softmax
    cross-entropy:

    .. math::

        \mathcal{L} = -\frac{1}{B} \sum_{b=1}^{B}
            \log p_\theta(y_b \mid x_b).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.roformer import (
    ...     RoFormerConfig, RoFormerForSequenceClassification,
    ... )
    >>> cfg = RoFormerConfig(num_labels=3, num_hidden_layers=2,
    ...                      hidden_size=128, num_attention_heads=2,
    ...                      intermediate_size=512)
    >>> model = RoFormerForSequenceClassification(cfg).eval()
    >>> input_ids = lucid.tensor([[101, 7592, 102]])
    >>> out = model(input_ids)
    >>> out.logits.shape   # (B=1, num_labels=3)
    (1, 3)
    """

    config_class: ClassVar[type[RoFormerConfig]] = RoFormerConfig
    base_model_prefix: ClassVar[str] = "roformer"

    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__(config)
        # RoFormer's own classification head, not BERT's.  The reference reads
        # the *raw* CLS hidden state and puts a full dense + activation stage
        # in front of the projection; routing through the tanh pooler instead
        # drops that stage and squashes the feature into [-1, 1] first — so
        # the trunk is built without a pooler that would never be called.
        self.roformer = _RoFormerEncoderOnly(config)
        self.classifier = _RoFormerClassificationHead(config)
        self.apply(self._init_weights)
        self.register_load_state_dict_pre_hook(_migrate_flat_classification_head)

    @override
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> SequenceClassificationOutput:
        outputs = self.roformer(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        logits = self.classifier(outputs.last_hidden_state)

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels.long())

        return SequenceClassificationOutput(logits=logits, loss=loss)


class RoFormerForTokenClassification(_RoFormerInitMixin, TokenClassificationModel, MaskedLMMixin):
    r"""RoFormer with a per-token classification head.

    Wraps :class:`RoFormerModel` with a dropout-regularised linear
    classifier applied independently at every sequence position.  Pattern
    mirrors :class:`BERTForTokenClassification` — used for sequence-
    labelling tasks (NER / POS / chunking) on RoPE-trained encoders.

    Parameters
    ----------
    config : RoFormerConfig
        Hyperparameters.  ``config.num_labels`` sets the per-position output
        dimension; ``config.classifier_dropout`` (falling back to
        ``hidden_dropout``) sets the dropout applied before the linear.

    Attributes
    ----------
    roformer : RoFormerModel
        Underlying RoPE encoder trunk.
    dropout : nn.Dropout
        Dropout applied to the full sequence hidden states.
    classifier : nn.Linear
        Final linear of shape ``(hidden_size, num_labels)`` mapping each
        token's hidden state to per-class logits.

    Notes
    -----
    Reference: Su, Lu, Pan, Murtadha, Wen, and Liu, *"RoFormer: Enhanced
    Transformer with Rotary Position Embedding"*, Neurocomputing,
    vol. 568, 2024 (arXiv:2104.09864).

    Loss when ``labels`` is provided is the masked cross-entropy inherited
    from :class:`MaskedLMMixin`:

    .. math::

        \mathcal{L} = -\frac{1}{|V|} \sum_{(b, t) \in V}
            \log p_\theta(y_{b, t} \mid x_b),

    where :math:`V` is the set of positions with ``label != -100``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.roformer import (
    ...     RoFormerConfig, RoFormerForTokenClassification,
    ... )
    >>> cfg = RoFormerConfig(num_labels=9, num_hidden_layers=2,
    ...                      hidden_size=128, num_attention_heads=2,
    ...                      intermediate_size=512)
    >>> model = RoFormerForTokenClassification(cfg).eval()
    >>> input_ids = lucid.tensor([[101, 7592, 2088, 102]])
    >>> out = model(input_ids)
    >>> out.logits.shape   # (B=1, T=4, num_labels=9)
    (1, 4, 9)
    """

    config_class: ClassVar[type[RoFormerConfig]] = RoFormerConfig
    base_model_prefix: ClassVar[str] = "roformer"

    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__(config)
        self.roformer = _RoFormerEncoderOnly(config)
        drop = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout
        )
        self.dropout = nn.Dropout(p=drop)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.apply(self._init_weights)

    @override
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> TokenClassificationOutput:
        outputs = self.roformer(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        seq = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(seq)

        loss: Tensor | None = None
        if labels is not None:
            loss = self.compute_lm_loss(logits, labels)

        return TokenClassificationOutput(logits=logits, loss=loss)


# ─────────────────────────────────────────────────────────────────────────────
# Multiple-choice + Question-Answering heads
# ─────────────────────────────────────────────────────────────────────────────


class RoFormerForMultipleChoice(_RoFormerInitMixin, SequenceClassificationModel):
    r"""RoFormer with a per-choice multiple-choice classification head.

    Wraps :class:`RoFormerModel` with a per-choice CLS classifier.  Input
    ``(N, C, L)`` flattens to ``(N*C, L)`` for the encoder; the pooled CLS
    embedding of each choice is projected to a scalar and the resulting
    ``(N, C)`` matrix is the per-choice logit.  Used for fine-tunes on
    SWAG / RACE-style commonsense and reading-comprehension tasks.

    Parameters
    ----------
    config : RoFormerConfig
        Hyperparameters.  ``config.classifier_dropout`` (falling back to
        ``hidden_dropout``) sets the dropout applied before the linear.

    Attributes
    ----------
    roformer : RoFormerModel
        Underlying RoPE encoder trunk.
    dropout : nn.Dropout
        Dropout layer applied to each choice's pooled embedding.
    classifier : nn.Linear
        Per-choice scalar projection of shape ``(hidden_size, 1)``.

    Notes
    -----
    Reference: Su, Lu, Pan, Murtadha, Wen, and Liu, *"RoFormer: Enhanced
    Transformer with Rotary Position Embedding"*, Neurocomputing,
    vol. 568, 2024 (arXiv:2104.09864).

    When ``labels`` is supplied the loss is the softmax cross-entropy over
    the ``C`` candidates:

    .. math::

        \mathcal{L}_{\mathrm{MC}}
            = -\frac{1}{N} \sum_{n=1}^{N}
              \log \mathrm{softmax}(z_n)_{y_n}.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.roformer import (
    ...     RoFormerConfig, RoFormerForMultipleChoice,
    ... )
    >>> cfg = RoFormerConfig(num_hidden_layers=2, hidden_size=128,
    ...                      num_attention_heads=2, intermediate_size=512)
    >>> model = RoFormerForMultipleChoice(cfg).eval()
    >>> input_ids = lucid.tensor([[[101, 7592, 102], [101, 2088, 102]]])    # (N=1, C=2, L=3)
    >>> out = model(input_ids)
    >>> out.logits.shape   # (N=1, C=2)
    (1, 2)
    """

    config_class: ClassVar[type[RoFormerConfig]] = RoFormerConfig
    base_model_prefix: ClassVar[str] = "roformer"

    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__(config)
        self.roformer = _RoFormerEncoderOnly(config)
        drop = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout
        )
        self.dropout = nn.Dropout(p=drop)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.apply(self._init_weights)

    @override
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> SequenceClassificationOutput:
        N, C, _, flat_ids, flat_mask, flat_tt = flatten_multiple_choice_inputs(
            "RoFormerForMultipleChoice", input_ids, attention_mask, token_type_ids
        )
        outputs = self.roformer(flat_ids, attention_mask=flat_mask, token_type_ids=flat_tt)
        # The reference multiple-choice head reads the *last* token's hidden
        # state directly, with no pooler projection — pooling the CLS token
        # through the tanh pooler puts a trained-elsewhere projection between
        # the encoder and the scorer, and reads the wrong position besides.
        last = last_kept_token(outputs.last_hidden_state, flat_mask)  # (N*C, H)
        pooled = self.dropout(last)
        logits_flat = self.classifier(pooled)  # (N*C, 1)
        logits = logits_flat.reshape(N, C)  # (N, C)

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels.long())

        return SequenceClassificationOutput(logits=logits, loss=loss)


class RoFormerForQuestionAnswering(_RoFormerInitMixin, SequenceClassificationModel):
    r"""RoFormer with a 2-way span head for extractive question answering.

    Wraps :class:`RoFormerModel` with a single linear of output width 2,
    producing start- and end-position logits over each token in the input.
    Identical contract to :class:`BERTForQuestionAnswering`; the SQuAD
    v1.1 / v2.0 fine-tuning recipe applied to a RoPE-trained encoder.

    Parameters
    ----------
    config : RoFormerConfig
        Hyperparameters.  The QA head is always 2-way; ``num_labels`` is
        ignored.

    Attributes
    ----------
    roformer : RoFormerModel
        Underlying RoPE encoder trunk.
    qa_outputs : nn.Linear
        Final linear of shape ``(hidden_size, 2)`` mapping each token's
        hidden state to ``(start_logit, end_logit)``.

    Notes
    -----
    Reference: Su, Lu, Pan, Murtadha, Wen, and Liu, *"RoFormer: Enhanced
    Transformer with Rotary Position Embedding"*, Neurocomputing,
    vol. 568, 2024 (arXiv:2104.09864).

    When both ``start_positions`` and ``end_positions`` are supplied the
    loss is the symmetric average of two cross-entropies:

    .. math::

        \mathcal{L} = \tfrac{1}{2}\!\left(
            \mathrm{CE}(z^{\mathrm{start}}, y^{\mathrm{start}})
          + \mathrm{CE}(z^{\mathrm{end}},   y^{\mathrm{end}})
        \right).

    The two score vectors are returned separately as ``start_logits`` and
    ``end_logits``, each ``(B, T)``, in a
    :class:`~lucid.models.QuestionAnsweringOutput` — the same shape
    :class:`~lucid.models.BERTForQuestionAnswering` returns.  It used to
    hand back one ``(B, T, 2)`` tensor and leave the caller to remember
    which index was which, which made the "identical contract" claimed
    above false for the one thing a caller actually touches.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.roformer import (
    ...     RoFormerConfig, RoFormerForQuestionAnswering,
    ... )
    >>> cfg = RoFormerConfig(num_hidden_layers=2, hidden_size=128,
    ...                      num_attention_heads=2, intermediate_size=512)
    >>> model = RoFormerForQuestionAnswering(cfg).eval()
    >>> input_ids = lucid.tensor([[101, 2040, 2003, 102, 1045, 2572, 102]])
    >>> out = model(input_ids)
    >>> out.start_logits.shape, out.end_logits.shape
    ((1, 7), (1, 7))
    """

    config_class: ClassVar[type[RoFormerConfig]] = RoFormerConfig
    base_model_prefix: ClassVar[str] = "roformer"

    def __init__(self, config: RoFormerConfig) -> None:
        super().__init__(config)
        self.roformer = _RoFormerEncoderOnly(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.apply(self._init_weights)

    @override
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        start_positions: Tensor | None = None,
        end_positions: Tensor | None = None,
    ) -> QuestionAnsweringOutput:
        outputs = self.roformer(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        logits = self.qa_outputs(outputs.last_hidden_state)  # (B, T, 2)
        start_logits = logits[..., 0]  # (B, T)
        end_logits = logits[..., 1]  # (B, T)

        loss = compute_qa_loss(start_logits, end_logits, start_positions, end_positions)

        return QuestionAnsweringOutput(start_logits=start_logits, end_logits=end_logits, loss=loss)
