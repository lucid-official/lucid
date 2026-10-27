"""Original Transformer model (Vaswani et al., 2017) — encoder-decoder seq2seq.

The bare paper architecture — six-layer encoder + six-layer decoder, sinusoidal
positional encoding, multi-head attention, position-wise FFN.  Public API
mirrors the rest of :mod:`lucid.models.text` (BERT / GPT / GPT-2 / RoFormer)
so callers move between families with no extra ceremony.

Task heads
----------
    TransformerModel                     — bare encoder-decoder trunk
    TransformerForSeq2SeqLM              — translation / summarisation head
    TransformerForSequenceClassification — encoder-only sentence classifier
    TransformerForTokenClassification    — encoder-only per-token classifier

Generation
----------
``TransformerForSeq2SeqLM.generate`` performs greedy / sampled decoding by
running the encoder once and unrolling the decoder one token at a time.  It
does **not** reuse :class:`lucid.models._mixins.CausalLMMixin` because that mixin
is decoder-only — encoder-decoder semantics differ enough to warrant a
local implementation.
"""

import math
from dataclasses import dataclass, replace
from typing import ClassVar, cast, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._tasks import (
    LanguageModelingModel,
    SequenceClassificationModel,
    TokenClassificationModel,
)
from lucid.models._mixins import MaskedLMMixin
from lucid.models._output import (
    ModelOutput,
    TokenClassificationOutput,
    SequenceClassificationOutput,
    Seq2SeqLMOutput,
)
from lucid.models.text._common._seq2seq import seq2seq_generate
from lucid.models.text.transformer._config import TransformerConfig
from lucid.models._sampling import _SamplingParams
from lucid.utils.cache import EncoderDecoderCache

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _causal_mask(T: int, device: str) -> Tensor:
    """``(T, T)`` additive mask: ``0`` on / below the diagonal, ``-1e4`` above.

    Plugs into :class:`lucid.nn.Transformer` as ``tgt_mask`` so the decoder
    cannot attend to future positions during training.
    """
    base = lucid.tril(lucid.ones((T, T), device=device))
    return (1.0 - base) * -1e4


def _key_padding_to_kpm(mask: Tensor | None) -> Tensor | None:
    """Convert HF-style 0/1 padding mask to ``nn.Transformer`` key_padding_mask.

    HF convention: ``1`` = attend, ``0`` = ignore.
    nn.Transformer convention: ``True`` = ignore, ``False`` = attend.
    """
    if mask is None:
        return None
    return mask.float() == 0.0


def _activation(config: TransformerConfig) -> str:
    """``hidden_act`` mapped onto the set the stock encoder layers accept."""
    return "gelu" if config.hidden_act in ("gelu", "gelu_new") else "relu"


class _EncoderStack(nn.Module):
    """The encoder half of ``nn.Transformer``, for encoder-only heads.

    Holds the stack as ``encoder`` so its state-dict keys match the
    encoder-decoder trunk's ``transformer.encoder.*`` exactly.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.hidden_dropout,
            attention_dropout=config.attention_dropout,
            activation=_activation(config),
            batch_first=True,
            layer_norm_eps=config.layer_norm_eps,
        )
        norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.encoder = nn.TransformerEncoder(layer, config.num_hidden_layers, norm=norm)


# ─────────────────────────────────────────────────────────────────────────────
# Bare encoder-decoder trunk
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class TransformerModelOutput(ModelOutput):
    r"""Output of the bare encoder-decoder trunk (no LM head).

    Attributes
    ----------
    last_hidden_state : Tensor
        Decoder hidden states, shape ``(B, T_dec, d_model)``.
    encoder_last_hidden_state : Tensor
        Encoder memory, shape ``(B, T_src, d_model)``.

    Notes
    -----
    The trunk used to return :class:`Seq2SeqLMOutput` with the decoder
    hidden states parked in its ``logits`` field.  Those are activations
    of width ``d_model``, not scores over a vocabulary, and a caller that
    softmaxed them — as the name invites — got a distribution over
    hidden units.  :class:`TransformerForSeq2SeqLM` is the model that
    produces real logits.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.transformer._model import (
    ...     TransformerModelOutput,
    ... )
    >>> out = TransformerModelOutput(
    ...     last_hidden_state=lucid.zeros(1, 32, 512),
    ...     encoder_last_hidden_state=lucid.zeros(1, 128, 512),
    ... )
    >>> out.last_hidden_state.shape, out.encoder_last_hidden_state.shape
    ((1, 32, 512), (1, 128, 512))

    ``TransformerModel(cfg)(input_ids, decoder_input_ids)`` returns this
    type. The two lengths differ because the decoder is partway through a
    sequence the encoder has already read in full.
    """

    last_hidden_state: Tensor
    encoder_last_hidden_state: Tensor


class TransformerModel(PretrainedModel):
    r"""Bare Vaswani encoder-decoder Transformer trunk.

    Implements the canonical seq2seq architecture of Vaswani, Shazeer,
    Parmar, Uszkoreit, Jones, Gomez, Kaiser, and Polosukhin, 2017.  A
    stack of :math:`N` encoder layers (multi-head self-attention + FFN)
    processes the source ``input_ids``; a stack of :math:`N` decoder
    layers (masked self-attention + cross-attention + FFN) consumes the
    right-shifted ``decoder_input_ids`` and the encoder memory.  Token
    embeddings are scaled by :math:`\sqrt{d_{\text{model}}}` and combined
    with sinusoidal positional encodings before the trunk.

    Forward inputs / outputs follow the :class:`Seq2SeqLMOutput` contract.
    For task-specific outputs use the ``TransformerForXxx`` wrappers; this
    bare trunk is exposed so callers needing raw decoder hidden states
    (e.g. for custom heads) can skip the default LM projection.

    Parameters
    ----------
    config : TransformerConfig
        Hyperparameters controlling vocabularies, depth, width, head count,
        max sequence length, and the ``share_embeddings`` /
        ``tie_word_embeddings`` toggles.

    Attributes
    ----------
    src_tok_emb : nn.Embedding
        Source-side token embedding of shape ``(vocab_size, d_model)``.
    tgt_tok_emb : nn.Embedding
        Target-side token embedding.  Identical object as ``src_tok_emb``
        when ``config.share_embeddings`` is ``True``.
    positional_encoding : nn.SinusoidalEmbedding
        Fixed sinusoidal positional encoding of size
        ``(max_position_embeddings, d_model)`` (Vaswani 2017 §3.5).
    dropout : nn.Dropout
        Post-embedding dropout layer.
    transformer : nn.Transformer
        Underlying encoder-decoder stack built from
        ``nn.TransformerEncoder`` + ``nn.TransformerDecoder``.  With
        ``config.encoder_only`` it holds the encoder stack alone (same
        ``transformer.encoder.*`` keys, no decoder parameters).

    Notes
    -----
    Reference: Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser,
    and Polosukhin, *"Attention Is All You Need"*, NeurIPS, 2017
    (arXiv:1706.03762).

    Scaled dot-product attention:

    .. math::

        \mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(
            \frac{Q K^{\top}}{\sqrt{d_k}}
        \right) V,

    with :math:`d_k = d_{\text{model}} / h`.  The decoder applies an
    additional lower-triangular causal mask on its self-attention so
    position :math:`t` cannot attend to positions :math:`> t`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.transformer import TransformerConfig, TransformerModel
    >>> cfg = TransformerConfig(num_hidden_layers=2, num_decoder_layers=2,
    ...                         hidden_size=64, num_attention_heads=2,
    ...                         intermediate_size=128, vocab_size=1000)
    >>> model = TransformerModel(cfg).eval()
    >>> src = lucid.tensor([[1, 234, 567, 2]])
    >>> tgt = lucid.tensor([[1, 100, 200]])
    >>> out = model(src, decoder_input_ids=tgt)
    >>> out.last_hidden_state.shape   # (B=1, T_tgt=3, d_model=64)
    (1, 3, 64)
    >>> out.encoder_last_hidden_state.shape
    (1, 4, 64)
    """

    config_class: ClassVar[type[TransformerConfig]] = TransformerConfig
    base_model_prefix: ClassVar[str] = "transformer"

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        self._max_pos = config.max_position_embeddings
        self._d_model = config.hidden_size
        tgt_vocab = config.effective_decoder_vocab_size

        # Token embeddings.  When ``share_embeddings`` is True the two tables
        # are the same parameter (saves memory, requires identical vocabs).
        # ``encoder_only`` consumers (the classification heads) never touch
        # the target side either, so they alias it rather than allocating a
        # second table nothing will read.
        self.src_tok_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        if config.share_embeddings or config.encoder_only:
            self.tgt_tok_emb = self.src_tok_emb
        else:
            self.tgt_tok_emb = nn.Embedding(tgt_vocab, config.hidden_size)

        # Sinusoidal positional encoding from :mod:`lucid.nn`.
        self.positional_encoding = nn.SinusoidalEmbedding(
            num_positions=config.max_position_embeddings,
            embedding_dim=config.hidden_size,
        )
        self.dropout = nn.Dropout(p=config.hidden_dropout)

        # A classification head consumes ``encode()`` only.  Even with zero
        # decoder layers, ``nn.Transformer`` (like the reference) still builds
        # the decoder's final norm, which then sat in every checkpoint as
        # weight no forward reads -- so build the encoder stack alone, under
        # the same ``transformer.encoder.*`` keys.
        self.transformer: nn.Transformer | _EncoderStack
        if config.encoder_only:
            self.transformer = _EncoderStack(config)
        else:
            self.transformer = nn.Transformer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                num_encoder_layers=config.num_hidden_layers,
                num_decoder_layers=config.num_decoder_layers,
                dim_feedforward=config.intermediate_size,
                dropout=config.hidden_dropout,
                # Declared and set to 0.3 by _CFG_LARGE, but nn.Transformer
                # used to drive both dropouts off one knob, so it never applied.
                attention_dropout=config.attention_dropout,
                activation=_activation(config),
                batch_first=True,
                layer_norm_eps=config.layer_norm_eps,
            )

    @override
    def get_input_embeddings(self) -> nn.Module:
        return self.src_tok_emb

    @override
    def set_input_embeddings(self, value: nn.Module) -> None:
        if not isinstance(value, nn.Embedding):
            raise TypeError(
                "TransformerModel input embeddings must be nn.Embedding, got "
                f"{type(value).__name__}"
            )
        # With ``share_embeddings`` (or no decoder) the target side is the
        # source table under a second name.  Replacing only the source would
        # split them: the decoder would keep reading the old table, and the
        # checkpoint would carry it.
        if self.tgt_tok_emb is self.src_tok_emb:
            self.tgt_tok_emb = value
        self.src_tok_emb = value

    def get_output_embeddings(self) -> nn.Module:
        """Target-side embedding — used to tie an LM head against."""
        return self.tgt_tok_emb

    def _embed(self, ids: Tensor, table: nn.Embedding, pos_offset: int = 0) -> Tensor:
        """Token-embed + scale-by-√d + sin/cos PE + dropout.

        ``pos_offset`` shifts the positional-encoding slice so cached decode
        steps embed the new tokens at their true absolute positions.
        """
        T = int(ids.shape[1])
        emb = table(ids) * math.sqrt(self._d_model)
        full_pe = self.positional_encoding()
        if pos_offset + T > self._max_pos:
            raise ValueError(
                f"Sequence length {pos_offset + T} exceeds "
                f"max_position_embeddings {self._max_pos}"
            )
        pe = full_pe[pos_offset : pos_offset + T]
        emb = emb + pe.unsqueeze(0)  # broadcast over batch
        return self.dropout(emb)

    def encode(
        self,
        src_ids: Tensor,
        src_attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Run only the encoder; returns memory ``(B, S, d_model)``."""
        src_emb = self._embed(src_ids, self.src_tok_emb)
        kpm = _key_padding_to_kpm(src_attention_mask)
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=kpm)
        return memory

    def _decoder(self) -> nn.TransformerDecoder:
        if isinstance(self.transformer, nn.Transformer):
            return self.transformer.decoder
        raise RuntimeError(
            "This TransformerModel was built encoder-only; it has no decoder "
            "stack to run.  Build it from a config with "
            "encoder_only=False for seq2seq use."
        )

    def decode(
        self,
        tgt_ids: Tensor,
        memory: Tensor,
        tgt_attention_mask: Tensor | None = None,
        memory_attention_mask: Tensor | None = None,
        *,
        past_key_value: EncoderDecoderCache | None = None,
        use_cache: bool = False,
    ) -> Tensor:
        """Run only the decoder against precomputed encoder memory.

        When ``use_cache`` is set, ``past_key_value`` accumulates the decoder
        self-attention keys/values (and caches the cross-attention projection
        of ``memory`` once), so each step only needs the freshly appended
        ``tgt_ids`` token(s).

        Args:
            tgt_ids:        ``(B, T)`` target token ids to decode this step.
            memory:         ``(B, S, d_model)`` encoder output to cross-attend.
            tgt_attention_mask:    Optional ``(B, T)`` target padding mask.
            memory_attention_mask: Optional ``(B, S)`` source padding mask.
            past_key_value: Optional :class:`EncoderDecoderCache` accumulating
                            decoder self-attention and the once-computed
                            cross-attention key/value.
            use_cache:      Read/write ``past_key_value`` so each step only
                            decodes the freshly appended token(s).

        Returns:
            ``(B, T, d_model)`` decoder hidden states (the ``ForSeq2SeqLM``
            wrapper projects these through the LM head).
        """
        decoder = self._decoder()
        T = int(tgt_ids.shape[1])
        dev = tgt_ids.device.type
        past_len = (
            past_key_value.get_seq_length() if (use_cache and past_key_value is not None) else 0
        )
        tgt_emb = self._embed(tgt_ids, self.tgt_tok_emb, pos_offset=past_len)
        # A single new token attends over the whole cached history (no causal
        # mask needed); the first/full pass uses the (T, T) causal mask.
        tgt_mask = None if past_len > 0 else _causal_mask(T, device=dev)
        tgt_kpm = _key_padding_to_kpm(tgt_attention_mask)
        mem_kpm = _key_padding_to_kpm(memory_attention_mask)
        return decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_kpm,
            memory_key_padding_mask=mem_kpm,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )

    @override
    def forward(
        self,
        input_ids: Tensor,
        decoder_input_ids: Tensor,
        attention_mask: Tensor | None = None,
        decoder_attention_mask: Tensor | None = None,
    ) -> TransformerModelOutput:
        """Full encode + decode pass.

        Args:
            input_ids:            ``(B, S)`` source token ids.
            decoder_input_ids:    ``(B, T)`` target token ids (right-shifted).
            attention_mask:       Optional ``(B, S)`` HF-style padding mask.
            decoder_attention_mask: Optional ``(B, T)`` HF-style padding mask.

        Returns:
            :class:`TransformerModelOutput` carrying the decoder hidden
            states and the encoder memory.  This trunk has no LM head —
            :class:`TransformerForSeq2SeqLM` projects these into logits.
        """
        memory = self.encode(input_ids, attention_mask)
        decoded = self.decode(
            decoder_input_ids,
            memory,
            tgt_attention_mask=decoder_attention_mask,
            memory_attention_mask=attention_mask,
        )
        return TransformerModelOutput(
            last_hidden_state=decoded,
            encoder_last_hidden_state=memory,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Seq2SeqLM head — translation / summarisation
# ─────────────────────────────────────────────────────────────────────────────


class TransformerForSeq2SeqLM(LanguageModelingModel):
    r"""Encoder-decoder Transformer with a tied seq2seq language-modeling head.

    Wraps :class:`TransformerModel` with a linear LM head projecting decoder
    hidden states to the target vocabulary.  The head weight is tied to
    ``tgt_tok_emb.weight`` when ``config.tie_word_embeddings`` is ``True``,
    matching the Vaswani 2017 §3.4 convention that halves the parameter cost
    of the output softmax.  Standard interface for translation and
    summarisation — pass ``input_ids`` (the source) and
    ``decoder_input_ids`` (the right-shifted target); supply ``labels`` to
    also compute the shift-loss for training.  Greedy inference is exposed
    through :meth:`generate`.

    Parameters
    ----------
    config : TransformerConfig
        Hyperparameters.  ``config.tie_word_embeddings`` (default True)
        controls the LM-head weight tying.

    Attributes
    ----------
    transformer : TransformerModel
        Underlying encoder-decoder trunk.
    lm_head : nn.Linear
        Output projection of shape ``(d_model, effective_decoder_vocab_size)``
        mapping decoder hidden states to target-vocabulary logits.

    Notes
    -----
    Reference: Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser,
    and Polosukhin, *"Attention Is All You Need"*, NeurIPS, 2017
    (arXiv:1706.03762).

    Loss when ``labels`` is supplied is the per-token cross-entropy over
    the right-shifted target sequence:

    .. math::

        \mathcal{L}_{\mathrm{S2S}}
            = -\frac{1}{BT} \sum_{b, t}
              \log p_\theta(y_{b, t} \mid x_b, y_{b, < t}),

    with positions labelled ``-100`` excluded.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.transformer import TransformerConfig, TransformerForSeq2SeqLM
    >>> cfg = TransformerConfig(num_hidden_layers=2, num_decoder_layers=2,
    ...                         hidden_size=64, num_attention_heads=2,
    ...                         intermediate_size=128, vocab_size=1000)
    >>> model = TransformerForSeq2SeqLM(cfg).eval()
    >>> src = lucid.tensor([[1, 234, 567, 2]])
    >>> tgt = lucid.tensor([[1, 100, 200]])
    >>> out = model(src, decoder_input_ids=tgt)
    >>> out.logits.shape   # (B=1, T_tgt=3, vocab=1000)
    (1, 3, 1000)
    """

    config_class: ClassVar[type[TransformerConfig]] = TransformerConfig
    base_model_prefix: ClassVar[str] = "transformer"

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        self.transformer = TransformerModel(config)
        tgt_vocab = config.effective_decoder_vocab_size
        self.lm_head = nn.Linear(config.hidden_size, tgt_vocab, bias=False)
        if config.tie_word_embeddings:
            self._tie_word_embeddings()

    @override
    def _tie_word_embeddings(self) -> None:
        # The head decodes into the *target* vocabulary, so it shares the
        # target table -- the source table too only when ``share_embeddings``
        # is set.
        weight = self.transformer.tgt_tok_emb.weight
        self.lm_head.weight = weight
        self.lm_head.out_features = int(weight.shape[0])

    @override
    def forward(
        self,
        input_ids: Tensor,
        decoder_input_ids: Tensor,
        attention_mask: Tensor | None = None,
        decoder_attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> Seq2SeqLMOutput:
        outputs = self.transformer(
            input_ids,
            decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )
        # decoder hidden → vocab
        logits = self.lm_head(outputs.last_hidden_state)

        loss: Tensor | None = None
        if labels is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.reshape(B * T, V),
                labels.reshape(B * T).long(),
                ignore_index=-100,
                # Section 5.4 / Table 3: both the base and big rows train with
                # eps_ls = 0.1.  It "hurts perplexity but improves accuracy and
                # BLEU", so omitting it changes what the model optimises.
                label_smoothing=cast(TransformerConfig, self.config).label_smoothing,
            )

        return Seq2SeqLMOutput(
            logits=logits,
            loss=loss,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
        )

    @lucid.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        *,
        max_length: int = 32,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
        attention_mask: Tensor | None = None,
        use_cache: bool = True,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
    ) -> Tensor:
        """Seq2seq decoding — greedy (default) or stochastic sampling.

        Encodes ``input_ids`` once, then unrolls the decoder one token at a
        time (BOS-seeded) until ``max_length`` (or every batch row hits
        ``eos_token_id``).  Token selection routes through the shared sampling
        primitives, so ``do_sample`` / ``temperature`` / ``top_k`` / ``top_p`` /
        ``repetition_penalty`` behave exactly as for decoder-only models.

        Args:
            input_ids:     ``(B, S)`` source token ids.
            max_length:    Cap on generated target length (incl. BOS).
            bos_token_id:  Start-of-sequence id; defaults to
                           ``config.bos_token_id`` then 0.
            eos_token_id:  End-of-sequence id; defaults to
                           ``config.eos_token_id``.
            pad_token_id:  Right-padding for finished rows; defaults to
                           ``config.pad_token_id`` then 0.
            attention_mask: Optional source padding mask.
            use_cache:     Use an ``EncoderDecoderCache`` so each step only
                           decodes the new token (default True).
            do_sample:     Sample from the (filtered) distribution instead of
                           greedy ``argmax`` (default False).
            temperature:   Logit temperature applied before sampling.
            top_k:         Keep only the ``top_k`` highest-probability tokens.
            top_p:         Nucleus threshold — keep the smallest set whose
                           cumulative probability reaches ``top_p``.
            repetition_penalty: Down-weight logits of already-generated tokens.

        Returns:
            ``(B, T_out)`` int Tensor of generated target tokens.
        """
        cfg = cast(TransformerConfig, self.config)
        bos = bos_token_id if bos_token_id is not None else cfg.bos_token_id
        eos = eos_token_id if eos_token_id is not None else cfg.eos_token_id
        pad = pad_token_id if pad_token_id is not None else cfg.pad_token_id
        if bos is None:
            bos = 0
        if pad is None:
            pad = 0

        dev = input_ids.device.type
        memory = self.transformer.encode(input_ids, attention_mask)

        # Decoder self-attention grows each step; the cross-attention
        # projection of ``memory`` is cached once.  Without a cache the
        # whole prefix is re-decoded every step.
        def next_logits(tokens: Tensor, cache: EncoderDecoderCache | None) -> Tensor:
            decoded = self.transformer.decode(
                tokens,
                memory,
                memory_attention_mask=attention_mask,
                past_key_value=cache,
                use_cache=cache is not None,
            )
            return self.lm_head(decoded[:, -1, :])  # (B, V)

        sampling = _SamplingParams(
            do_sample=do_sample, temperature=temperature, top_k=top_k, top_p=top_p,
            repetition_penalty=repetition_penalty, eos_token_id=eos, pad_token_id=pad,
            dev=dev,
        )  # fmt: skip
        return seq2seq_generate(
            next_logits, batch=int(input_ids.shape[0]), device=dev, start_token_id=bos,
            max_length=max_length, use_cache=use_cache, sampling=sampling,
        )  # fmt: skip


# ─────────────────────────────────────────────────────────────────────────────
# Encoder-only downstream task wrappers
# ─────────────────────────────────────────────────────────────────────────────


class TransformerForSequenceClassification(SequenceClassificationModel):
    r"""Encoder-only sentence classifier — pools the first source-side token.

    Wraps :class:`TransformerModel` and runs only the encoder half during
    ``forward``; the first source-side token's hidden state is
    dropout-regularised and projected through a linear of output width
    ``config.num_labels``.  Pattern mirrors
    :class:`BERTForSequenceClassification` — the decoder half of the trunk
    is unused so this head trains the same encoder weights efficiently for
    GLUE-style fine-tunes on the Vaswani backbone.

    Parameters
    ----------
    config : TransformerConfig
        Hyperparameters.  ``config.num_labels`` sets the output dimension;
        ``config.classifier_dropout`` (falling back to ``hidden_dropout``)
        sets the dropout applied before the linear.

    Attributes
    ----------
    transformer : TransformerModel
        Underlying encoder-decoder trunk; only ``encode`` is invoked here.
    dropout : nn.Dropout
        Dropout layer applied to the pooled first-token embedding.
    classifier : nn.Linear
        Final linear of shape ``(d_model, num_labels)`` producing per-class
        logits.

    Notes
    -----
    Reference: Vaswani et al., *"Attention Is All You Need"*, NeurIPS, 2017
    (arXiv:1706.03762).

    When ``labels`` is supplied, the loss is the standard cross-entropy:

    .. math::

        \mathcal{L} = -\frac{1}{B} \sum_{b=1}^{B}
            \log p_\theta(y_b \mid x_b).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.transformer import (
    ...     TransformerConfig, TransformerForSequenceClassification,
    ... )
    >>> cfg = TransformerConfig(num_labels=3, num_hidden_layers=2,
    ...                         hidden_size=64, num_attention_heads=2,
    ...                         intermediate_size=128, vocab_size=1000)
    >>> model = TransformerForSequenceClassification(cfg).eval()
    >>> input_ids = lucid.tensor([[1, 234, 567, 2]])
    >>> out = model(input_ids)
    >>> out.logits.shape   # (B=1, num_labels=3)
    (1, 3)
    """

    config_class: ClassVar[type[TransformerConfig]] = TransformerConfig
    base_model_prefix: ClassVar[str] = "transformer"

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        self.transformer = TransformerModel(replace(config, encoder_only=True))
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
        labels: Tensor | None = None,
    ) -> SequenceClassificationOutput:
        memory = self.transformer.encode(input_ids, attention_mask)  # (B, S, d)
        pooled = memory[:, 0]  # (B, d)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels.long())

        return SequenceClassificationOutput(logits=logits, loss=loss)


class TransformerForTokenClassification(TokenClassificationModel, MaskedLMMixin):
    r"""Encoder-only per-token classifier — NER / POS / chunking heads.

    Wraps :class:`TransformerModel` and runs only the encoder half during
    ``forward``; every source-side token's hidden state is
    dropout-regularised and projected through a linear of output width
    ``config.num_labels``.  Uses :class:`MaskedLMMixin` for the standard
    masked ``(B, T, V) → loss`` reduction shared with BERT and RoFormer.

    Parameters
    ----------
    config : TransformerConfig
        Hyperparameters.  ``config.num_labels`` sets the per-position output
        dimension; ``config.classifier_dropout`` (falling back to
        ``hidden_dropout``) sets the dropout applied before the linear.

    Attributes
    ----------
    transformer : TransformerModel
        Underlying encoder-decoder trunk; only ``encode`` is invoked here.
    dropout : nn.Dropout
        Dropout layer applied to the full sequence hidden states.
    classifier : nn.Linear
        Final linear of shape ``(d_model, num_labels)`` mapping each token's
        hidden state to per-class logits.

    Notes
    -----
    Reference: Vaswani et al., *"Attention Is All You Need"*, NeurIPS, 2017
    (arXiv:1706.03762).

    Loss (when ``labels`` is provided) is the masked cross-entropy over
    positions whose label is not ``-100``:

    .. math::

        \mathcal{L} = -\frac{1}{|V|} \sum_{(b, t) \in V}
            \log p_\theta(y_{b, t} \mid x_b).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.transformer import (
    ...     TransformerConfig, TransformerForTokenClassification,
    ... )
    >>> cfg = TransformerConfig(num_labels=9, num_hidden_layers=2,
    ...                         hidden_size=64, num_attention_heads=2,
    ...                         intermediate_size=128, vocab_size=1000)
    >>> model = TransformerForTokenClassification(cfg).eval()
    >>> input_ids = lucid.tensor([[1, 234, 567, 2]])
    >>> out = model(input_ids)
    >>> out.logits.shape   # (B=1, T=4, num_labels=9)
    (1, 4, 9)
    """

    config_class: ClassVar[type[TransformerConfig]] = TransformerConfig
    base_model_prefix: ClassVar[str] = "transformer"

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        self.transformer = TransformerModel(replace(config, encoder_only=True))
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
        labels: Tensor | None = None,
    ) -> TokenClassificationOutput:
        memory = self.transformer.encode(input_ids, attention_mask)  # (B, S, d)
        seq = self.dropout(memory)
        logits = self.classifier(seq)

        loss: Tensor | None = None
        if labels is not None:
            loss = self.compute_lm_loss(logits, labels)

        return TokenClassificationOutput(logits=logits, loss=loss)
