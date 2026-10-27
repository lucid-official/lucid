"""Unit tests for GPT-2 (Phase 4 third concrete model).

Key things to verify beyond the GPT-1 set:
    - pre-LN block structure (LN comes *before* sublayer)
    - final ``ln_f`` LayerNorm exists
    - residual init scaling (``c_proj`` weights are visibly smaller in std)
"""

import pytest

import lucid
from lucid.models import (
    GPT2Config,
    GPT2ForSequenceClassification,
    GPT2LMHeadModel,
    GPT2Model,
    create_model,
    is_model,
)

_VOCAB = 64
_HIDDEN = 32
_LAYERS = 4
_HEADS = 4
_INTER = 64
_MAX_POS = 16


def _tiny_config(**overrides: object) -> GPT2Config:
    base = {
        "vocab_size": _VOCAB,
        "hidden_size": _HIDDEN,
        "num_hidden_layers": _LAYERS,
        "num_attention_heads": _HEADS,
        "intermediate_size": _INTER,
        "max_position_embeddings": _MAX_POS,
        "num_labels": 3,
    }
    base.update(overrides)
    return GPT2Config(**base)  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────


class TestGPT2Config:
    def test_paper_defaults_small(self) -> None:
        cfg = GPT2Config()
        assert cfg.vocab_size == 50_257
        assert cfg.hidden_size == 768
        assert cfg.num_hidden_layers == 12
        assert cfg.max_position_embeddings == 1024
        assert cfg.bos_token_id == 50_256
        assert cfg.eos_token_id == 50_256
        assert cfg.layer_norm_eps == 1e-5
        assert cfg.scale_residual_init is True

    def test_num_labels_invariant(self) -> None:
        with pytest.raises(ValueError, match="num_labels"):
            GPT2Config(num_labels=0)


# ─────────────────────────────────────────────────────────────────────────────
# Architecture sanity — pre-LN, ln_f, c_proj init scaling
# ─────────────────────────────────────────────────────────────────────────────


class TestGPT2Architecture:
    def test_has_final_ln_f(self) -> None:
        m = GPT2Model(_tiny_config()).eval()
        # Final LayerNorm must be present at the trunk top.
        assert hasattr(m, "ln_f")
        # Forward must produce normalised output at that LayerNorm — quick
        # numerical sanity: variance across the hidden dim is close to 1.
        ids = lucid.tensor([[1, 2, 3, 4]]).long()
        h = m(ids).last_hidden_state
        var = float(((h - h.mean(dim=-1, keepdim=True)) ** 2).mean().item())
        assert 0.1 < var < 5.0, f"ln_f output looks unnormalised (var={var})"

    def test_pre_ln_block_ordering(self) -> None:
        """In a pre-LN block, ``ln_1`` is declared before ``attn`` and
        ``ln_2`` before ``mlp`` — guards against future refactors silently
        flipping back to post-LN."""
        m = GPT2Model(_tiny_config()).eval()
        block = m.h[0]
        attrs = list(block.__dict__["_modules"].keys())
        assert attrs.index("ln_1") < attrs.index("attn")
        assert attrs.index("ln_2") < attrs.index("mlp")

    def test_residual_init_scaling(self) -> None:
        """``c_proj`` weights have std proportional to ``1 / sqrt(2N)`` — so a
        deeper config must produce *smaller* std than a shallow one."""
        shallow = GPT2Model(_tiny_config(num_hidden_layers=2)).eval()
        deep = GPT2Model(_tiny_config(num_hidden_layers=8)).eval()

        def _proj_std(m: GPT2Model) -> float:
            w = m.h[0].attn.c_proj.weight
            mean = float(w.mean().item())
            return float(((w - mean) ** 2).mean().item()) ** 0.5

        assert _proj_std(deep) < _proj_std(shallow)


# ─────────────────────────────────────────────────────────────────────────────
# Forward / heads
# ─────────────────────────────────────────────────────────────────────────────


class TestGPT2Forward:
    def test_trunk_shape(self) -> None:
        m = GPT2Model(_tiny_config()).eval()
        ids = lucid.tensor([[1, 2, 3, 4, 5, 6]]).long()
        out = m(ids)
        assert tuple(out.last_hidden_state.shape) == (1, 6, _HIDDEN)

    def test_seq_too_long_raises(self) -> None:
        m = GPT2Model(_tiny_config(max_position_embeddings=4)).eval()
        with pytest.raises(ValueError, match="max_position_embeddings"):
            m(lucid.tensor([[1, 2, 3, 4, 5]]).long())


class TestGPT2LMHead:
    def test_logits_and_shift_loss(self) -> None:
        cfg = _tiny_config()
        m = GPT2LMHeadModel(cfg).eval()
        ids = lucid.tensor([[1, 2, 3, 4, 5, 6]]).long()
        out = m(ids, labels=ids)
        assert tuple(out.logits.shape) == (1, 6, _VOCAB)
        assert out.loss is not None

    def test_tied(self) -> None:
        m = GPT2LMHeadModel(_tiny_config()).eval()
        assert m.lm_head.weight is m.transformer.wte.weight


class TestGPT2Generate:
    def test_greedy(self) -> None:
        m = GPT2LMHeadModel(_tiny_config()).eval()
        out = m.generate(lucid.tensor([[1, 2, 3]]).long(), max_length=7, do_sample=False)
        assert tuple(out.shape) == (1, 7)
        assert [int(out[0, t].item()) for t in range(3)] == [1, 2, 3]

    def test_sampling(self) -> None:
        m = GPT2LMHeadModel(_tiny_config()).eval()
        out = m.generate(
            lucid.tensor([[1, 2, 3]]).long(), max_length=7, do_sample=True, temperature=0.7, top_k=8
        )
        assert tuple(out.shape) == (1, 7)


class TestGPT2Cls:
    def test_logits_with_mask(self) -> None:
        cfg = _tiny_config()
        m = GPT2ForSequenceClassification(cfg).eval()
        ids = lucid.tensor([[1, 2, 3, 4, 5, 6]]).long()
        mask = lucid.tensor([[1, 1, 1, 0, 0, 0]]).long()
        out = m(ids, attention_mask=mask, labels=lucid.tensor([1]).long())
        assert tuple(out.logits.shape) == (1, cfg.num_labels)


# ─────────────────────────────────────────────────────────────────────────────
# Causal mask leak — identical guard to GPT-1
# ─────────────────────────────────────────────────────────────────────────────


class TestGPT2CausalMask:
    def test_future_tokens_do_not_leak(self) -> None:
        m = GPT2Model(_tiny_config()).eval()
        ids_a = lucid.tensor([[1, 2, 3, 4, 5, 6]]).long()
        ids_b = lucid.tensor([[1, 2, 3, 40, 50, 60]]).long()
        h_a = m(ids_a).last_hidden_state
        h_b = m(ids_b).last_hidden_state
        diff = float(((h_a[:, :3, :] - h_b[:, :3, :]) ** 2).sum().item())
        assert diff < 1e-6, f"Causal mask leaks: prefix diff = {diff}"


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────


class TestGPT2Registry:
    @pytest.mark.parametrize(
        "name",
        [
            "gpt2_small",
            "gpt2_medium",
            "gpt2_large",
            "gpt2_xlarge",
            "gpt2_small_lm",
            "gpt2_medium_lm",
            "gpt2_large_lm",
            "gpt2_xlarge_lm",
            "gpt2_small_cls",
        ],
    )
    def test_registered(self, name: str) -> None:
        assert is_model(name)

    def test_factory_with_override(self) -> None:
        m = create_model(
            "gpt2_small",
            vocab_size=_VOCAB,
            hidden_size=_HIDDEN,
            num_hidden_layers=_LAYERS,
            num_attention_heads=_HEADS,
            intermediate_size=_INTER,
            max_position_embeddings=_MAX_POS,
        )
        assert isinstance(m, GPT2Model)
        out = m.eval()(lucid.tensor([[1, 2, 3]]).long())
        assert tuple(out.last_hidden_state.shape) == (1, 3, _HIDDEN)


from lucid.models import GPT2DoubleHeadsModel


class TestGPT2DoubleHeadsModel:
    def test_shapes_and_losses(self) -> None:
        m = GPT2DoubleHeadsModel(_tiny_config()).eval()
        ids = lucid.tensor([[[1, 2, 3, 4], [5, 6, 7, 8]]]).long()  # (N=1, C=2, L=4)
        mc = lucid.tensor([[3, 3]]).long()
        out = m(ids, mc, labels=ids, mc_labels=lucid.tensor([0]).long())
        assert tuple(out.lm_logits.shape) == (1, 2, 4, _VOCAB)
        assert tuple(out.mc_logits.shape) == (1, 2)
        assert out.loss is not None


# (enum_name, slug, num_classes, task) — eight shipped GPT-2 checkpoints:
# four trunks (num_classes = hidden_size) and four causal-LM heads
# (num_classes = vocab_size).  All WebText, byte-level BPE 50 257 vocab.
_SHIPPED = (
    ("GPT2SmallWeights", "gpt2-small", 768, "base"),
    ("GPT2MediumWeights", "gpt2-medium", 1024, "base"),
    ("GPT2LargeWeights", "gpt2-large", 1280, "base"),
    ("GPT2XLargeWeights", "gpt2-xlarge", 1600, "base"),
    ("GPT2SmallLMWeights", "gpt2-small-lm", 50_257, "lm"),
    ("GPT2MediumLMWeights", "gpt2-medium-lm", 50_257, "lm"),
    ("GPT2LargeLMWeights", "gpt2-large-lm", 50_257, "lm"),
    ("GPT2XLargeLMWeights", "gpt2-xlarge-lm", 50_257, "lm"),
)
_TAG = "WEBTEXT"


class TestGPT2WeightsEnums:
    """Static contract of the per-variant Weights enums — no network."""

    @staticmethod
    def _enum(name: str) -> type:
        import lucid.models.weights as weights_ns

        return getattr(weights_ns, name)

    def test_default_aliases(self) -> None:
        for enum_name, *_rest in _SHIPPED:
            cls = self._enum(enum_name)
            assert cls.DEFAULT is cls[_TAG]

    def test_entry_fields(self) -> None:
        for enum_name, slug, num_classes, _task in _SHIPPED:
            member = self._enum(enum_name)[_TAG]
            e = member.entry
            assert e.num_classes == num_classes
            assert len(e.sha256) == 64
            assert f"lucid-dl/{slug}" in e.url
            assert f"/{_TAG}/" in e.url
            assert member.meta["tag"] == _TAG
            assert member.meta["license"] == "mit"

    def test_registered_for_factories(self) -> None:
        from lucid.weights import weights_for

        for enum_name, slug, *_rest in _SHIPPED:
            factory = slug.replace("-", "_")
            resolved = weights_for(factory)
            assert resolved is not None
            assert resolved.__name__ == enum_name


class TestGPT2EncoderPretrainedTransfer:
    """``gpt2_small_cls`` loads the pretrained ``gpt2_small`` trunk into
    ``model.transformer`` (classifier head random).  Verify the encoder
    checkpoint's key layout is identical to the ``.transformer`` submodule —
    so ``load_weight_entry(model.transformer, entry)`` succeeds with
    ``strict=True`` at full scale — without network access.
    """

    def test_encoder_state_loads_into_transformer_submodule(self) -> None:
        cfg = _tiny_config()
        enc = GPT2Model(cfg)
        head = GPT2ForSequenceClassification(cfg)
        result = head.transformer.load_state_dict(enc.state_dict(), strict=True)
        assert not list(getattr(result, "missing_keys", []) or [])
        assert not list(getattr(result, "unexpected_keys", []) or [])

    def test_factory_exposes_weights_kwarg(self) -> None:
        import importlib
        import inspect

        mod = importlib.import_module("lucid.models.text.gpt2._pretrained")
        assert "weights" in inspect.signature(mod.gpt2_small_cls).parameters
