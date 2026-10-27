"""GPT-1 model (Radford et al., 2018) — decoder-only Transformer.

Module / parameter naming mirrors HuggingFace ``OpenAIGPTModel`` for state-
dict portability.  Layout:

    tokens_embed       (vocab → hidden)
    positions_embed    (max_pos → hidden)
    drop               (post-embedding Dropout)
    h.{i}.attn.{c_attn, c_proj}        (fused QKV + output projection)
    h.{i}.ln_1                         (post-attention LayerNorm)
    h.{i}.mlp.{c_fc, c_proj}           (2-layer FFN, intermediate × 4)
    h.{i}.ln_2                         (post-MLP LayerNorm)

The classifier / LM heads then sit on top of ``GPTModel``:

    GPTLMHeadModel             — tied LM head (CausalLMMixin host)
    GPTForSequenceClassification — last-token classifier
"""

import math
from dataclasses import dataclass
from typing import ClassVar, final, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._tasks import LanguageModelingModel
from lucid.models._tasks import SequenceClassificationModel
from lucid.models._mixins import CausalLMMixin
from lucid.models._output import (
    BaseModelOutput,
    CausalLMOutput,
    SequenceClassificationOutput,
    ModelOutput,
)
from lucid.models._utils._text import (
    extended_attention_mask,
    init_text_weights as _init_weights,
    last_kept_token,
    text_activation,
    token_at,
)
from lucid.models.text.gpt._config import GPTConfig
from lucid.utils.cache import Cache, DynamicCache, StaticCache

# ─────────────────────────────────────────────────────────────────────────────
# Multi-head causal self-attention (fused QKV)
# ─────────────────────────────────────────────────────────────────────────────


@final
class _GPTSelfAttention(nn.Module):
    """Fused QKV projection + causal masking.

    Parameter names match HF: ``c_attn`` is the ``(hidden → 3·hidden)`` fused
    projection, ``c_proj`` is the post-attention output projection.
    """

    causal_mask: Tensor

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = math.sqrt(self.head_dim)

        self.c_attn = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=True)
        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.attn_dropout = nn.Dropout(p=config.attention_dropout)
        self.resid_dropout = nn.Dropout(p=config.hidden_dropout)

        # Precomputed lower-triangular causal mask: 1 on / below diagonal.
        # Shape (1, 1, T_max, T_max) so it broadcasts over (B, H).
        T = config.max_position_embeddings
        mask = lucid.tril(lucid.ones((T, T))).reshape(1, 1, T, T)
        self.register_buffer("causal_mask", mask, persistent=False)

    @override
    def forward(
        self,
        hidden: Tensor,
        attention_mask: Tensor | None = None,
        *,
        past_key_value: Cache | None = None,
        layer_idx: int = 0,
        cache_position: Tensor | None = None,
        use_cache: bool = False,
    ) -> Tensor:
        B, T, _ = hidden.shape
        H, D = self.num_heads, self.head_dim

        qkv = self.c_attn(hidden)  # (B, T, 3*hidden)
        qkv = qkv.reshape(B, T, 3, H, D).permute(2, 0, 3, 1, 4)
        q: Tensor = qkv[0]
        k: Tensor = qkv[1]
        v: Tensor = qkv[2]

        # KV cache: keys/values span the full history, query stays length T.
        past_len = 0
        if past_key_value is not None:
            past_len = past_key_value.get_seq_length(layer_idx)
            # cache_position is consumed by StaticCache (the in-place write
            # location); DynamicCache ignores it and simply appends.
            cache_kwargs: dict[str, object] = {"cache_position": cache_position}
            if isinstance(past_key_value, StaticCache):
                # Attend over only the filled prefix (``past_len + T``), not the
                # full max_cache_len buffer — the latter is max_cache_len/(filled+1)×
                # wasted q·kᵀ.  StaticCache.update narrows its returned view to it.
                cache_kwargs["read_len"] = past_len + T
            k, v = past_key_value.update(k, v, layer_idx, cache_kwargs=cache_kwargs)
        t_total = int(k.shape[2])  # DynamicCache: past_len+T; StaticCache: max_cache_len

        if isinstance(past_key_value, StaticCache) and cache_position is not None:
            # StaticCache returns the fixed max_len buffer with a dynamic write
            # position.  Row p of the lower-triangular causal_mask is exactly the
            # allowed-key pattern for a query at absolute position p (keys j <= p),
            # so gather the rows named by cache_position — keeps the compiled graph
            # position-agnostic and avoids ``arange`` (no MPSGraph emitter).
            causal = self.causal_mask.index_select(2, cache_position)[:, :, :, :t_total]
        else:
            # Causal slice for query positions [past_len, past_len+T) over keys
            # [0, t_total); with no cache this is the original [:, :, :T, :T] block.
            causal = self.causal_mask[:, :, past_len : past_len + T, :t_total]
        # Additive attention bias: ``(1 - causal) * -1e4`` blocks the upper
        # triangle; add any padding mask on top.
        bias: Tensor = (1.0 - causal) * -1e4
        if attention_mask is not None:
            pad_mask = attention_mask
            mask_len = int(pad_mask.shape[-1])
            if mask_len < t_total:
                # StaticCache hands back its full max_cache_len buffer; the
                # unwritten tail carries no real key, and ``causal`` already
                # blocks it, so extend the padding mask with neutral zeros.
                pad_mask = F.pad(pad_mask, (0, t_total - mask_len))
            bias = bias + pad_mask

        # One masked-attention core for every path: the fused kernel at
        # inference, the materialised weights when attention dropout is on.
        # A hand-written ``softmax(q kᵀ + bias) v`` here once took the dropout
        # path, and a query row whose every key was ``-inf`` came out NaN there
        # but ``0`` from the fused kernel (LCD-387).
        out: Tensor = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=bias,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            scale=1.0 / self.scale,
        )
        out = out.permute(0, 2, 1, 3).reshape(B, T, H * D)  # (B, T, hidden)
        out = self.c_proj(out)
        return self.resid_dropout(out)


# ─────────────────────────────────────────────────────────────────────────────
# Feed-forward block — Conv1d-style ``c_fc`` / ``c_proj`` per HF naming
# ─────────────────────────────────────────────────────────────────────────────


@final
class _GPTMLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.hidden_size, config.intermediate_size)
        self.c_proj = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(p=config.hidden_dropout)
        self._act_name = config.hidden_act

    @override
    def forward(self, x: Tensor) -> Tensor:
        h = text_activation(self._act_name, self.c_fc(x))
        h = self.c_proj(h)
        return self.dropout(h)


# ─────────────────────────────────────────────────────────────────────────────
# Transformer block — *post*-LN per GPT-1 (BERT-style), unlike GPT-2's pre-LN
# ─────────────────────────────────────────────────────────────────────────────


@final
class _GPTBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.attn = _GPTSelfAttention(config)
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = _GPTMLP(config)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    @override
    def forward(
        self,
        hidden: Tensor,
        attention_mask: Tensor | None = None,
        *,
        past_key_value: Cache | None = None,
        layer_idx: int = 0,
        cache_position: Tensor | None = None,
        use_cache: bool = False,
    ) -> Tensor:
        # Residual + post-LN, as in the original paper.
        a = self.attn(
            hidden,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            layer_idx=layer_idx,
            cache_position=cache_position,
            use_cache=use_cache,
        )
        hidden = self.ln_1(hidden + a)
        m = self.mlp(hidden)
        hidden = self.ln_2(hidden + m)
        return hidden


# ─────────────────────────────────────────────────────────────────────────────
# Bare GPT trunk
# ─────────────────────────────────────────────────────────────────────────────


class GPTModel(PretrainedModel):
    r"""Bare GPT-1 decoder-only Transformer trunk.

    Implements the original generative pre-training architecture from
    Radford, Narasimhan, Salimans, and Sutskever, 2018: a 12-layer stack
    of decoder blocks with fused-QKV multi-head causal self-attention, a
    2-layer feed-forward block, post-LayerNorm residuals, learned absolute
    position embeddings, and a tanh-approximated GELU activation
    (``gelu_new``).  Returns the per-token hidden states; the LM and
    classifier heads sit on top of this trunk.

    Parameters
    ----------
    config : GPTConfig
        Hyperparameters controlling vocabulary, depth, width, head count,
        and regularisation.  See :class:`GPTConfig` for the full field
        list.

    Attributes
    ----------
    tokens_embed : nn.Embedding
        Token embedding table of shape ``(vocab_size, hidden_size)``.
        HuggingFace name retained for state-dict portability.
    positions_embed : nn.Embedding
        Learned absolute position embedding of shape
        ``(max_position_embeddings, hidden_size)``.
    drop : nn.Dropout
        Post-embedding dropout layer.
    h : nn.ModuleList
        Stack of ``config.num_hidden_layers`` GPT-1 transformer blocks.
    config_class : type[GPTConfig]
        Registry pointer for matching-config instantiation.
    base_model_prefix : str
        State-dict prefix (``"transformer"``) under which the trunk is
        nested in task-head variants.

    Notes
    -----
    Reference: Radford, Narasimhan, Salimans, and Sutskever, *"Improving
    Language Understanding by Generative Pre-Training"*, OpenAI Technical
    Report, 2018.

    Causal LM factorisation:

    .. math::

        \mathcal{L}_{\mathrm{LM}}(\theta)
            = \sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t}).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.gpt import GPTConfig, GPTModel
    >>> cfg = GPTConfig(num_hidden_layers=2, hidden_size=128,
    ...                 num_attention_heads=2, intermediate_size=512)
    >>> model = GPTModel(cfg).eval()
    >>> input_ids = lucid.tensor([[101, 7592, 2088, 102]])
    >>> out = model(input_ids)
    >>> out.last_hidden_state.shape   # (B=1, T=4, H=128)
    (1, 4, 128)
    """

    config_class: ClassVar[type[GPTConfig]] = GPTConfig
    base_model_prefix: ClassVar[str] = "transformer"

    position_ids: Tensor

    def __init__(self, config: GPTConfig) -> None:
        super().__init__(config)
        self._max_pos = config.max_position_embeddings
        self.tokens_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.positions_embed = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.drop = nn.Dropout(p=config.hidden_dropout)
        self.h = nn.ModuleList([_GPTBlock(config) for _ in range(config.num_hidden_layers)])

        pos = lucid.arange(config.max_position_embeddings).long().unsqueeze(0)
        self.register_buffer("position_ids", pos, persistent=False)

        _init_weights(self, config.initializer_range)

    @override
    def get_input_embeddings(self) -> nn.Module:
        return self.tokens_embed

    @override
    def set_input_embeddings(self, value: nn.Module) -> None:
        if not isinstance(value, nn.Embedding):
            raise TypeError(
                "GPTModel input embeddings must be nn.Embedding, got " f"{type(value).__name__}"
            )
        self.tokens_embed = value

    @override
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        *,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
        cache_position: Tensor | None = None,
    ) -> BaseModelOutput:
        B, T = int(input_ids.shape[0]), int(input_ids.shape[1])
        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        if past_len + T > self._max_pos:
            raise ValueError(
                f"Sequence length {past_len + T} exceeds max_position_embeddings "
                f"{self._max_pos}"
            )

        # Positions are offset by the cached length so new tokens get their
        # true absolute positions.  Under a compiled StaticCache decode the
        # write position arrives as the runtime cache_position tensor (keeping
        # the graph position-agnostic); otherwise slice the static buffer.
        if isinstance(past_key_values, StaticCache) and cache_position is not None:
            pos_ids = cache_position.reshape(1, T)
        else:
            pos_ids = self.position_ids[:, past_len : past_len + T]
        tok_emb = self.tokens_embed(input_ids)
        pos_emb = self.positions_embed(pos_ids)
        hidden = self.drop(tok_emb + pos_emb)

        # Keys span the cached history plus the new tokens, so the padding
        # mask has to be that wide -- a T-wide mask would broadcast onto the
        # wrong columns (or fail outright) once a cache is in play.
        ext_mask = extended_attention_mask(attention_mask, (B, T), key_length=past_len + T)
        for layer_idx, block in enumerate(self.h):
            hidden = block(
                hidden,
                attention_mask=ext_mask,
                past_key_value=past_key_values,
                layer_idx=layer_idx,
                cache_position=cache_position,
                use_cache=use_cache,
            )

        return BaseModelOutput(last_hidden_state=hidden)


# ─────────────────────────────────────────────────────────────────────────────
# Causal-LM head — host for CausalLMMixin
# ─────────────────────────────────────────────────────────────────────────────


class GPTLMHeadModel(LanguageModelingModel, CausalLMMixin):
    r"""GPT-1 with a tied causal-language-modeling head.

    Wraps :class:`GPTModel` with an output linear whose weight matrix is
    bound to the input ``tokens_embed`` table when
    ``config.tie_word_embeddings`` is ``True`` — halving the parameter
    cost of the softmax layer.  This is both the pre-training entry point
    and the host of :meth:`lucid.models._mixins.CausalLMMixin.generate` for
    free-form autoregressive sampling.

    Parameters
    ----------
    config : GPTConfig
        Hyperparameters.  ``config.tie_word_embeddings`` (default True)
        controls weight tying.

    Attributes
    ----------
    transformer : GPTModel
        Underlying decoder trunk.
    lm_head : nn.Linear
        Output projection of shape ``(hidden_size, vocab_size)`` mapping
        each token's hidden state to vocabulary logits.

    Notes
    -----
    Reference: Radford et al., *"Improving Language Understanding by
    Generative Pre-Training"*, OpenAI Technical Report, 2018.

    Loss when ``labels`` is supplied is the standard next-token-shifted
    causal-LM cross-entropy

    .. math::

        \mathcal{L}_{\mathrm{CLM}}
            = -\frac{1}{B(T-1)} \sum_{b, t}
              \log p_\theta(y_{b, t+1} \mid x_{b, \le t}),

    with positions labelled ``-100`` excluded.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.gpt import GPTConfig, GPTLMHeadModel
    >>> cfg = GPTConfig(num_hidden_layers=2, hidden_size=128,
    ...                 num_attention_heads=2, intermediate_size=512)
    >>> model = GPTLMHeadModel(cfg).eval()
    >>> input_ids = lucid.tensor([[101, 7592, 2088, 102]])
    >>> out = model(input_ids)
    >>> out.logits.shape   # (B=1, T=4, V=40478)
    (1, 4, 40478)
    """

    config_class: ClassVar[type[GPTConfig]] = GPTConfig
    base_model_prefix: ClassVar[str] = "transformer"

    def __init__(self, config: GPTConfig) -> None:
        super().__init__(config)
        self.transformer = GPTModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self._tie_word_embeddings()

    @override
    def _tie_word_embeddings(self) -> None:
        # Share the token table's storage and gradient.  ``out_features``
        # follows it, so a swapped-in table of another size is reported
        # truthfully (the quantized converters build from it).
        weight = self.transformer.tokens_embed.weight
        self.lm_head.weight = weight
        self.lm_head.out_features = int(weight.shape[0])

    @override
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        *,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
        cache_position: Tensor | None = None,
    ) -> CausalLMOutput:
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden = outputs.last_hidden_state
        logits = self.lm_head(hidden)

        loss: Tensor | None = None
        if labels is not None:
            # Standard causal LM shift: predict token t+1 from positions [0..t].
            B, T, V = logits.shape
            shift_logits = logits[:, :-1, :].reshape((B * (T - 1), V))
            shift_labels = labels[:, 1:].reshape((B * (T - 1),)).long()
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

        return CausalLMOutput(
            logits=logits, loss=loss, past_key_values=past_key_values if use_cache else None
        )


# ─────────────────────────────────────────────────────────────────────────────
# Sequence classification — uses the LAST non-pad token's hidden state
# ─────────────────────────────────────────────────────────────────────────────


class GPTForSequenceClassification(SequenceClassificationModel):
    r"""GPT-1 with a last-token sequence-classification head.

    Wraps :class:`GPTModel` with a dropout-regularised linear classifier
    operating on the **rightmost non-pad token's** hidden state — the
    canonical decoder-style pooling.  Unlike BERT (CLS at position 0),
    decoder LMs only have a meaningful summary at the final attended
    position because every preceding position has attended to a strict
    prefix only.  When ``attention_mask`` is provided we honour it;
    otherwise the last position of every row is used.

    Parameters
    ----------
    config : GPTConfig
        Hyperparameters.  ``config.num_labels`` sets the output dimension;
        ``config.classifier_dropout`` (falling back to ``hidden_dropout``)
        sets the dropout applied before the linear.

    Attributes
    ----------
    transformer : GPTModel
        Underlying decoder trunk.
    dropout : nn.Dropout
        Dropout layer applied to the pooled last-token embedding.
    classifier : nn.Linear
        Final linear of shape ``(hidden_size, num_labels)`` producing
        per-class logits.

    Notes
    -----
    Reference: Radford et al., *"Improving Language Understanding by
    Generative Pre-Training"*, OpenAI Technical Report, 2018, §3.3.

    Pooling picks index :math:`t^\star = \max\{t : \mathrm{mask}_t = 1\}`
    (the rightmost real token, so left padding pools correctly; 0 for an
    all-pad row) per row.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.gpt import GPTConfig, GPTForSequenceClassification
    >>> cfg = GPTConfig(num_labels=3, num_hidden_layers=2, hidden_size=128,
    ...                 num_attention_heads=2, intermediate_size=512)
    >>> model = GPTForSequenceClassification(cfg).eval()
    >>> input_ids = lucid.tensor([[101, 7592, 102]])
    >>> out = model(input_ids)
    >>> out.logits.shape   # (B=1, num_labels=3)
    (1, 3)
    """

    config_class: ClassVar[type[GPTConfig]] = GPTConfig
    base_model_prefix: ClassVar[str] = "transformer"

    def __init__(self, config: GPTConfig) -> None:
        super().__init__(config)
        self.transformer = GPTModel(config)
        drop = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout
        )
        self.dropout = nn.Dropout(p=drop)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels, bias=False)

    @override
    def forward(
        self, input_ids: Tensor, attention_mask: Tensor | None = None, labels: Tensor | None = None
    ) -> SequenceClassificationOutput:
        outputs = self.transformer(input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (B, T, H)
        pooled = last_kept_token(hidden, attention_mask)  # (B, H)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels.long())

        return SequenceClassificationOutput(logits=logits, loss=loss)


# ─────────────────────────────────────────────────────────────────────────────
# DoubleHeadsModel — LM head + multiple-choice classification head
# ─────────────────────────────────────────────────────────────────────────────


# Section 3.3's auxiliary-LM weight: L = L_task + lambda * L_lm, lambda = 0.5.
_AUX_LM_WEIGHT: float = 0.5


@dataclass(slots=True)
class GPTDoubleHeadsOutput(ModelOutput):
    r"""Joint LM + multiple-choice output for :class:`GPTDoubleHeadsModel`.

    Aggregates the per-choice causal-LM logits, per-choice classification
    logits, and the optional per-objective and combined losses produced
    when training Radford-style multiple-choice fine-tunes.

    Parameters
    ----------
    lm_logits : Tensor
        LM head logits of shape ``(N, C, L, vocab_size)`` — one
        distribution per token of every batch x choice pair.
    mc_logits : Tensor
        Multiple-choice head logits of shape ``(N, C)`` — one scalar per
        candidate completion.
    loss : Tensor or None, default=None
        Sum of ``lm_loss`` and ``mc_loss`` when both are available;
        otherwise the single available loss, or ``None``.
    lm_loss : Tensor or None, default=None
        Shift-cross-entropy on the LM head when ``labels`` is supplied.
    mc_loss : Tensor or None, default=None
        Cross-entropy on the multiple-choice head when ``mc_labels`` is
        supplied.

    Notes
    -----
    Reference: Radford, Narasimhan, Salimans, and Sutskever, *"Improving
    Language Understanding by Generative Pre-Training"*, OpenAI Technical
    Report, 2018, §3.3.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.gpt import GPTConfig, GPTDoubleHeadsModel
    >>> cfg = GPTConfig(num_hidden_layers=2, hidden_size=64,
    ...                 num_attention_heads=2, intermediate_size=256)
    >>> model = GPTDoubleHeadsModel(cfg).eval()
    >>> input_ids = lucid.tensor([[[101, 7592, 102]]])      # (N=1, C=1, L=3)
    >>> mc_ids = lucid.tensor([[2]]).long()
    >>> out = model(input_ids, mc_token_ids=mc_ids)
    >>> out.lm_logits.shape, out.mc_logits.shape
    ((1, 1, 3, 40478), (1, 1))
    """

    lm_logits: Tensor
    mc_logits: Tensor
    loss: Tensor | None = None
    lm_loss: Tensor | None = None
    mc_loss: Tensor | None = None


@final
class _GPTMultipleChoiceHead(nn.Module):
    """Pool the hidden state at ``mc_token_ids`` per choice → 1-way Linear.

    Maps ``(N, C, L, H) → (N, C)``: for each (batch, choice) pair, gather the
    hidden state at the choice-specific token (typically a CLS / final token)
    and project to a single logit.  The C-way softmax / argmax happens
    outside.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.summary = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(p=config.hidden_dropout)

    @override
    def forward(self, hidden_states: Tensor, mc_token_ids: Tensor) -> Tensor:
        # Shapes — hidden_states (N, C, L, H), mc_token_ids (N, C)
        N, C = int(hidden_states.shape[0]), int(hidden_states.shape[1])
        pooled = token_at(hidden_states, mc_token_ids)  # (N, C, H)
        pooled = self.dropout(pooled)
        return self.summary(pooled).reshape(N, C)


class GPTDoubleHeadsModel(LanguageModelingModel):
    r"""GPT-1 + LM head + multiple-choice classification head.

    Implements the Radford 2018 §3.3 multi-task fine-tuning recipe.  Used
    for tasks like RACE / Story Cloze where each example presents a prompt
    followed by :math:`C` candidate completions; the model is jointly
    trained with the standard LM objective on each choice and a
    :math:`C`-way classification loss picking the correct completion.

    Inputs are 3-D: ``input_ids`` of shape ``(N, C, L)`` flattens to
    ``(N * C, L)`` for the trunk and reshapes back to ``(N, C, ...)`` at
    the heads.  ``mc_token_ids`` of shape ``(N, C)`` indexes the per-choice
    pooling position (typically the trailing CLF token of each completion).

    Parameters
    ----------
    config : GPTConfig
        Hyperparameters.  ``config.tie_word_embeddings`` (default True)
        ties the LM head weight to the input embedding matrix.

    Attributes
    ----------
    transformer : GPTModel
        Underlying decoder trunk.
    lm_head : nn.Linear
        Tied output projection of shape ``(hidden_size, vocab_size)``.
    mc_head : nn.Module
        Per-choice pooled-token scalar projection of shape
        ``(hidden_size, 1)``.

    Notes
    -----
    Reference: Radford, Narasimhan, Salimans, and Sutskever, *"Improving
    Language Understanding by Generative Pre-Training"*, OpenAI Technical
    Report, 2018, §3.3.

    Combined loss when both objectives are supplied:

    .. math::

        \mathcal{L}_{\mathrm{joint}}
            = \mathcal{L}_{\mathrm{CLM}} + \mathcal{L}_{\mathrm{MC}}.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.gpt import GPTConfig, GPTDoubleHeadsModel
    >>> cfg = GPTConfig(num_hidden_layers=2, hidden_size=64,
    ...                 num_attention_heads=2, intermediate_size=256)
    >>> model = GPTDoubleHeadsModel(cfg).eval()
    >>> input_ids = lucid.tensor([[[101, 7592, 102],
    ...                            [101, 2088, 102]]])    # (N=1, C=2, L=3)
    >>> mc_ids = lucid.tensor([[2, 2]]).long()
    >>> out = model(input_ids, mc_token_ids=mc_ids)
    >>> out.mc_logits.shape   # (N=1, C=2)
    (1, 2)
    """

    config_class: ClassVar[type[GPTConfig]] = GPTConfig
    base_model_prefix: ClassVar[str] = "transformer"

    def __init__(self, config: GPTConfig) -> None:
        super().__init__(config)
        self.transformer = GPTModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mc_head = _GPTMultipleChoiceHead(config)
        if config.tie_word_embeddings:
            self._tie_word_embeddings()

    @override
    def _tie_word_embeddings(self) -> None:
        weight = self.transformer.tokens_embed.weight
        self.lm_head.weight = weight
        self.lm_head.out_features = int(weight.shape[0])

    @override
    def forward(
        self,
        input_ids: Tensor,
        mc_token_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        mc_labels: Tensor | None = None,
    ) -> GPTDoubleHeadsOutput:
        if input_ids.ndim != 3:
            raise ValueError(
                f"GPTDoubleHeadsModel expects input_ids of shape (N, C, L), "
                f"got {tuple(input_ids.shape)}"
            )
        N = int(input_ids.shape[0])
        C = int(input_ids.shape[1])
        L = int(input_ids.shape[2])

        flat_ids = input_ids.reshape(N * C, L)
        flat_mask = attention_mask.reshape(N * C, L) if attention_mask is not None else None
        outputs = self.transformer(flat_ids, attention_mask=flat_mask)
        hidden = outputs.last_hidden_state  # (N*C, L, H)
        H = int(hidden.shape[-1])

        lm_logits_flat = self.lm_head(hidden)  # (N*C, L, V)
        V = int(lm_logits_flat.shape[-1])
        lm_logits = lm_logits_flat.reshape(N, C, L, V)
        mc_logits = self.mc_head(hidden.reshape(N, C, L, H), mc_token_ids)  # (N, C)

        lm_loss: Tensor | None = None
        mc_loss: Tensor | None = None
        if labels is not None:
            shift_logits = lm_logits[:, :, :-1, :].reshape(N * C * (L - 1), V)
            shift_labels = labels[:, :, 1:].reshape(N * C * (L - 1)).long()
            lm_loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
        if mc_labels is not None:
            mc_loss = F.cross_entropy(mc_logits, mc_labels.long())

        total_loss: Tensor | None = None
        if lm_loss is not None and mc_loss is not None:
            # Section 3.3: L_3(C) = L_2(C) + lambda * L_1(C), with lambda = 0.5
            # in the paper's experiments.  Weighting the auxiliary LM term at
            # 1.0 gives it the same pull as the task objective it is only
            # meant to regularise.
            total_loss = mc_loss + _AUX_LM_WEIGHT * lm_loss
        elif lm_loss is not None:
            total_loss = lm_loss
        elif mc_loss is not None:
            total_loss = mc_loss

        return GPTDoubleHeadsOutput(
            lm_logits=lm_logits,
            mc_logits=mc_logits,
            loss=total_loss,
            lm_loss=lm_loss,
            mc_loss=mc_loss,
        )
