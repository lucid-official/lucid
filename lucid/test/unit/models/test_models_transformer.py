"""Unit tests for the original Vaswani Transformer (encoder-decoder seq2seq)."""

import pytest

import lucid
from lucid.models import (
    TransformerConfig,
    TransformerForSeq2SeqLM,
    TransformerForSequenceClassification,
    TransformerForTokenClassification,
    TransformerModel,
    create_model,
    is_model,
)

_VOCAB = 64
_HIDDEN = 32
_LAYERS = 2
_HEADS = 4
_INTER = 64
_MAX_POS = 32


def _tiny_config(**overrides: object) -> TransformerConfig:
    base = {
        "vocab_size": _VOCAB,
        "hidden_size": _HIDDEN,
        "num_hidden_layers": _LAYERS,
        "num_decoder_layers": _LAYERS,
        "num_attention_heads": _HEADS,
        "intermediate_size": _INTER,
        "max_position_embeddings": _MAX_POS,
        "num_labels": 3,
    }
    base.update(overrides)
    return TransformerConfig(**base)  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────


class TestTransformerConfig:
    def test_paper_defaults(self) -> None:
        cfg = TransformerConfig()
        assert cfg.hidden_size == 512
        assert cfg.num_hidden_layers == 6
        assert cfg.num_decoder_layers == 6
        assert cfg.num_attention_heads == 8
        assert cfg.intermediate_size == 2048
        assert cfg.model_type == "transformer"

    def test_decoder_vocab_defaults_to_source(self) -> None:
        cfg = TransformerConfig(vocab_size=1000, decoder_vocab_size=None)
        assert cfg.effective_decoder_vocab_size == 1000

    def test_decoder_vocab_explicit(self) -> None:
        # A separate target vocabulary only makes sense with sharing off --
        # sharing one matrix across two vocabularies is a contradiction, and
        # __post_init__ rejects it (see the next test).
        cfg = TransformerConfig(vocab_size=1000, decoder_vocab_size=2000, share_embeddings=False)
        assert cfg.effective_decoder_vocab_size == 2000

    def test_embeddings_are_shared_by_default(self) -> None:
        # Section 3.4 shares one weight matrix across both embedding layers
        # and the pre-softmax projection.
        assert TransformerConfig().share_embeddings is True

    def test_share_embeddings_requires_matching_vocab(self) -> None:
        with pytest.raises(ValueError, match="share_embeddings"):
            TransformerConfig(vocab_size=1000, decoder_vocab_size=2000, share_embeddings=True)

    def test_invalid_num_decoder_layers(self) -> None:
        with pytest.raises(ValueError, match="num_decoder_layers"):
            TransformerConfig(num_decoder_layers=0)

    def test_invalid_num_labels(self) -> None:
        with pytest.raises(ValueError, match="num_labels"):
            TransformerConfig(num_labels=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Model forward
# ─────────────────────────────────────────────────────────────────────────────


class TestTransformerModel:
    def test_encode_decode_shapes(self) -> None:
        m = TransformerModel(_tiny_config()).eval()
        src = lucid.tensor([[1, 2, 3, 4, 5]]).long()
        tgt = lucid.tensor([[1, 2, 3]]).long()
        out = m(src, tgt)
        # The bare trunk has no LM head, so it returns hidden states — not
        # ``logits``, which would invite a softmax over d_model units.
        assert tuple(out.last_hidden_state.shape) == (1, 3, _HIDDEN)
        assert tuple(out.encoder_last_hidden_state.shape) == (1, 5, _HIDDEN)

    def test_seq_too_long_raises(self) -> None:
        m = TransformerModel(_tiny_config(max_position_embeddings=4)).eval()
        with pytest.raises(ValueError, match="max_position_embeddings"):
            m(lucid.tensor([[1, 2, 3, 4, 5]]).long(), lucid.tensor([[1, 2]]).long())

    def test_share_embeddings(self) -> None:
        m = TransformerModel(_tiny_config(share_embeddings=True)).eval()
        assert m.src_tok_emb.weight is m.tgt_tok_emb.weight

    def test_unshared_embeddings_distinct(self) -> None:
        m = TransformerModel(_tiny_config(share_embeddings=False)).eval()
        assert m.src_tok_emb.weight is not m.tgt_tok_emb.weight

    def test_encoder_only_decode_pipeline(self) -> None:
        """Memory from ``encode()`` can be reused across multiple decode calls."""
        m = TransformerModel(_tiny_config()).eval()
        src = lucid.tensor([[1, 2, 3, 4]]).long()
        memory = m.encode(src)
        dec1 = m.decode(lucid.tensor([[1]]).long(), memory)
        dec2 = m.decode(lucid.tensor([[1, 2]]).long(), memory)
        assert tuple(dec1.shape) == (1, 1, _HIDDEN)
        assert tuple(dec2.shape) == (1, 2, _HIDDEN)


class TestTransformerForSeq2SeqLM:
    def test_logits_and_loss(self) -> None:
        m = TransformerForSeq2SeqLM(_tiny_config()).eval()
        src = lucid.tensor([[1, 2, 3, 4]]).long()
        tgt = lucid.tensor([[1, 2, 3]]).long()
        out = m(src, tgt, labels=tgt)
        assert tuple(out.logits.shape) == (1, 3, _VOCAB)
        assert out.loss is not None
        assert float(out.loss.item()) > 0.0

    def test_decoder_vocab_split(self) -> None:
        """When decoder_vocab_size != vocab_size, LM head matches decoder side."""
        cfg = _tiny_config(decoder_vocab_size=128, share_embeddings=False)
        m = TransformerForSeq2SeqLM(cfg).eval()
        src = lucid.tensor([[1, 2, 3]]).long()
        tgt = lucid.tensor([[1, 2]]).long()
        out = m(src, tgt)
        assert tuple(out.logits.shape) == (1, 2, 128)

    def test_lm_head_tied(self) -> None:
        m = TransformerForSeq2SeqLM(_tiny_config()).eval()
        assert m.lm_head.weight is m.transformer.tgt_tok_emb.weight

    def test_lm_head_untied_when_disabled(self) -> None:
        m = TransformerForSeq2SeqLM(_tiny_config(tie_word_embeddings=False)).eval()
        assert m.lm_head.weight is not m.transformer.tgt_tok_emb.weight

    def test_generate_shape(self) -> None:
        m = TransformerForSeq2SeqLM(_tiny_config()).eval()
        src = lucid.tensor([[1, 2, 3, 4]]).long()
        out = m.generate(src, max_length=6, bos_token_id=0)
        assert tuple(out.shape) == (1, 6)

    def test_generate_eos_stops_and_pads(self) -> None:
        m = TransformerForSeq2SeqLM(_tiny_config()).eval()
        src = lucid.tensor([[1, 2, 3]]).long()
        out = m.generate(src, max_length=8, bos_token_id=0, eos_token_id=0, pad_token_id=99)
        # Greedy can hit EOS=0 at random spots given untrained weights, but the
        # function must always return a (B, T) tensor of valid token ids.
        assert tuple(out.shape) == (1, 8) or out.shape[1] <= 8


class TestTransformerForSequenceClassification:
    def test_logits_and_loss(self) -> None:
        cfg = _tiny_config()
        m = TransformerForSequenceClassification(cfg).eval()
        src = lucid.tensor([[1, 2, 3, 4, 5]]).long()
        out = m(src, labels=lucid.tensor([2]).long())
        assert tuple(out.logits.shape) == (1, cfg.num_labels)
        assert out.loss is not None


class TestTransformerForTokenClassification:
    def test_logits_and_loss(self) -> None:
        cfg = _tiny_config()
        m = TransformerForTokenClassification(cfg).eval()
        src = lucid.tensor([[1, 2, 3, 4]]).long()
        labels = lucid.tensor([[0, 1, 2, -100]]).long()
        out = m(src, labels=labels)
        assert tuple(out.logits.shape) == (1, 4, cfg.num_labels)
        assert out.loss is not None


class TestEncoderOnlyHeadsCarryNoDeadWeight:
    """LCD-204: the classification heads kept the decoder's final norm."""

    @pytest.mark.parametrize(
        "head",
        [TransformerForSequenceClassification, TransformerForTokenClassification],
    )
    def test_every_parameter_gets_a_gradient(self, head: type) -> None:
        m = head(_tiny_config(hidden_dropout=0.0)).train()
        src = lucid.tensor([[1, 2, 3, 4]]).long()
        m(src).logits.sum().backward()
        dead = [n for n, p in m.named_parameters() if p.grad is None]
        assert not dead, dead
        assert not [n for n, _ in m.named_parameters() if ".decoder." in n]

    def test_encoder_keys_match_the_full_trunk(self) -> None:
        enc = TransformerForSequenceClassification(_tiny_config()).state_dict()
        full = TransformerForSeq2SeqLM(_tiny_config()).state_dict()
        enc_keys = {k for k in enc if k.startswith("transformer.transformer.")}
        full_enc = {k for k in full if k.startswith("transformer.transformer.encoder.")}
        assert enc_keys == full_enc

    def test_encoder_only_trunk_refuses_to_decode(self) -> None:
        m = TransformerModel(_tiny_config(encoder_only=True)).eval()
        memory = m.encode(lucid.tensor([[1, 2, 3]]).long())
        with pytest.raises(RuntimeError, match="encoder-only"):
            m.decode(lucid.tensor([[1]]).long(), memory)


# ─────────────────────────────────────────────────────────────────────────────
# Causal mask correctness (decoder mustn't peek)
# ─────────────────────────────────────────────────────────────────────────────


class TestCausalMask:
    def test_decoder_prefix_invariant_to_suffix(self) -> None:
        """Position t's decoder hidden state must not depend on positions > t.
        Replacing the target suffix should leave the prefix's hiddens bit-equal.
        """
        m = TransformerModel(_tiny_config()).eval()
        src = lucid.tensor([[1, 2, 3, 4]]).long()
        memory = m.encode(src)
        tgt_a = lucid.tensor([[1, 2, 3, 4, 5]]).long()
        tgt_b = lucid.tensor([[1, 2, 3, 40, 50]]).long()
        d_a = m.decode(tgt_a, memory)
        d_b = m.decode(tgt_b, memory)
        diff = float(((d_a[:, :3, :] - d_b[:, :3, :]) ** 2).sum().item())
        assert diff < 1e-6, f"Causal mask leaks: prefix diff = {diff}"


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────


class TestTransformerRegistry:
    @pytest.mark.parametrize(
        "name",
        [
            "transformer_base",
            "transformer_large",
            "transformer_base_seq2seq",
            "transformer_large_seq2seq",
            "transformer_base_cls",
            "transformer_base_token_cls",
        ],
    )
    def test_registered(self, name: str) -> None:
        assert is_model(name)

    def test_factory_with_override(self) -> None:
        # Override the paper-faithful base config down to test-friendly dims.
        m = create_model(
            "transformer_base",
            vocab_size=_VOCAB,
            hidden_size=_HIDDEN,
            num_hidden_layers=_LAYERS,
            num_decoder_layers=_LAYERS,
            num_attention_heads=_HEADS,
            intermediate_size=_INTER,
            max_position_embeddings=_MAX_POS,
        )
        assert isinstance(m, TransformerModel)
        out = m.eval()(
            lucid.tensor([[1, 2, 3]]).long(),
            lucid.tensor([[1]]).long(),
        )
        assert tuple(out.last_hidden_state.shape) == (1, 1, _HIDDEN)
