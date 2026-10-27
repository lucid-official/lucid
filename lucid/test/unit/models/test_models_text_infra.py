"""Unit tests for the text-domain Phase 4 base layer.

Covers:
- LanguageModelConfig validation
- CausalLMMixin greedy + sampling
- get/set_input_embeddings on every text task wrapper, and the tied heads

(Positional-encoding tests — RoPE / sinusoidal PE — live in
``test/unit/nn/test_nn_positional.py`` since those primitives moved to
:mod:`lucid.nn` in 2026-05.)
"""

import importlib
import inspect
from typing import Any

import pytest

import lucid
import lucid.nn as nn
from lucid.models import CausalLMOutput, LanguageModelConfig, PretrainedModel
from lucid.models._mixins import CausalLMMixin
from lucid.models._tasks import TaskModel

# ─────────────────────────────────────────────────────────────────────────────
# LanguageModelConfig
# ─────────────────────────────────────────────────────────────────────────────


class TestLanguageModelConfig:
    def test_defaults(self) -> None:
        cfg = LanguageModelConfig()
        assert cfg.vocab_size == 30_522
        assert cfg.hidden_size == 768
        assert cfg.num_attention_heads == 12
        assert cfg.hidden_size % cfg.num_attention_heads == 0

    def test_head_divisibility_violation(self) -> None:
        with pytest.raises(ValueError, match="divisible"):
            LanguageModelConfig(hidden_size=10, num_attention_heads=3)

    def test_negative_vocab(self) -> None:
        with pytest.raises(ValueError, match="vocab_size"):
            LanguageModelConfig(vocab_size=0)

    def test_dropout_bounds(self) -> None:
        with pytest.raises(ValueError, match="hidden_dropout"):
            LanguageModelConfig(hidden_dropout=1.5)
        with pytest.raises(ValueError, match="attention_dropout"):
            LanguageModelConfig(attention_dropout=-0.1)

    def test_layer_norm_eps_positive(self) -> None:
        with pytest.raises(ValueError, match="layer_norm_eps"):
            LanguageModelConfig(layer_norm_eps=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# CausalLMMixin
# ─────────────────────────────────────────────────────────────────────────────


class _DeterministicLM(CausalLMMixin, nn.Module):
    """Test fixture: always predicts token (last_token + 1) % vocab."""

    def __init__(self, vocab: int = 7, eos: int | None = None) -> None:
        super().__init__()
        self.vocab = vocab
        self.config = type("C", (), {"eos_token_id": eos, "pad_token_id": 0, "vocab_size": vocab})()

    def forward(self, input_ids: lucid.Tensor) -> CausalLMOutput:
        B = int(input_ids.shape[0])
        T = int(input_ids.shape[1])
        rows: list[list[list[float]]] = []
        for b in range(B):
            seq_logits: list[list[float]] = []
            for t in range(T):
                tok = int(input_ids[b, t].item())
                row = [0.0] * self.vocab
                row[(tok + 1) % self.vocab] = 10.0
                seq_logits.append(row)
            rows.append(seq_logits)
        logits = lucid.tensor(rows)
        return CausalLMOutput(logits=logits, loss=None)


class TestCausalLMMixin:
    def test_greedy_extends_correctly(self) -> None:
        m = _DeterministicLM(vocab=7)
        prompt = lucid.tensor([[1, 2]]).long()
        out = m.generate(prompt, max_length=5)
        assert tuple(out.shape) == (1, 5)
        # Each step picks (last + 1) % 7 → 1, 2, 3, 4, 5
        assert [int(out[0, t].item()) for t in range(5)] == [1, 2, 3, 4, 5]

    def test_max_new_tokens_takes_precedence(self) -> None:
        m = _DeterministicLM(vocab=7)
        prompt = lucid.tensor([[1, 2]]).long()
        out = m.generate(prompt, max_length=100, max_new_tokens=2)
        assert tuple(out.shape) == (1, 4)

    def test_eos_stops_and_pads(self) -> None:
        # eos_token_id = 4 → after producing 4, should pad with 0.
        m = _DeterministicLM(vocab=7, eos=4)
        prompt = lucid.tensor([[1, 2]]).long()
        out = m.generate(prompt, max_length=7, eos_token_id=4)
        toks = [int(out[0, t].item()) for t in range(int(out.shape[1]))]
        # 1, 2, 3, 4 then padded to length 7 with pad_token_id=0
        assert toks[:4] == [1, 2, 3, 4]
        assert all(t == 0 for t in toks[4:])

    def test_sampling_runs(self) -> None:
        m = _DeterministicLM(vocab=7)
        prompt = lucid.tensor([[1, 2]]).long()
        out = m.generate(prompt, max_length=5, do_sample=True, temperature=0.7, top_k=3, top_p=0.9)
        assert tuple(out.shape) == (1, 5)

    def test_input_must_be_2d(self) -> None:
        m = _DeterministicLM(vocab=7)
        with pytest.raises(ValueError, match="2-D"):
            m.generate(lucid.tensor([1, 2, 3]).long())


class TestSamplingFilters:
    """The vectorized on-device sampling primitives (no per-element CPU loops)."""

    def test_top_k_keeps_exactly_k(self) -> None:
        from lucid.models._sampling import _top_k_filter

        logits = lucid.tensor([[1.0, 5.0, 2.0, 4.0, 3.0]])
        out = _top_k_filter(logits, 2)  # keep 5.0, 4.0
        kept = [v > -1e8 for v in out[0].numpy().tolist()]
        assert kept == [False, True, False, True, False]

    def test_top_k_ge_vocab_is_identity(self) -> None:
        from lucid.models._sampling import _top_k_filter

        logits = lucid.tensor([[1.0, 2.0, 3.0]])
        out = _top_k_filter(logits, 5)
        assert out[0].numpy().tolist() == [1.0, 2.0, 3.0]

    def test_top_p_keeps_dominant_token(self) -> None:
        from lucid.models._sampling import _top_p_filter

        # softmax mass concentrates on index 1; a tight p keeps only it.
        logits = lucid.tensor([[0.0, 10.0, 0.1, 0.2]])
        out = _top_p_filter(logits, 0.5)
        kept = [v > -1e8 for v in out[0].numpy().tolist()]
        assert kept[1] is True  # argmax always kept
        assert sum(kept) == 1  # nucleus is just the dominant token

    def test_repetition_penalty_lowers_seen(self) -> None:
        from lucid.models._sampling import _apply_repetition_penalty

        logits = lucid.tensor([[2.0, 2.0, 2.0, 2.0]])
        prefix = lucid.tensor([[1, 3]]).long()  # tokens 1, 3 already generated
        out = _apply_repetition_penalty(logits, prefix, 2.0)
        row = out[0].numpy().tolist()
        assert row[0] == 2.0 and row[2] == 2.0  # unseen unchanged
        assert row[1] == 1.0 and row[3] == 1.0  # seen positive logit halved

    def test_multinomial_in_range_and_deterministic(self) -> None:
        from lucid.models._sampling import _multinomial_one

        probs = lucid.tensor([[0.1, 0.2, 0.3, 0.4]])
        lucid.manual_seed(0)
        a = _multinomial_one(probs, device="cpu")
        lucid.manual_seed(0)
        b = _multinomial_one(probs, device="cpu")
        assert int(a[0].item()) == int(b[0].item())  # same RNG → same draw
        assert 0 <= int(a[0].item()) < 4  # valid index


# ─────────────────────────────────────────────────────────────────────────────
# Embedding accessors on task wrappers
# ─────────────────────────────────────────────────────────────────────────────

_TEXT_FAMILIES = ("gpt", "gpt2", "bert", "roformer", "transformer")
_V = 50  # tiny vocabulary
_H = 16  # tiny hidden size


def _tiny_config(config_cls: Any, **overrides: object) -> Any:
    """The family's own config, shrunk to one small layer."""
    small: dict[str, object] = {
        "vocab_size": _V,
        "hidden_size": _H,
        "num_hidden_layers": 1,
        "num_decoder_layers": 1,
        "num_attention_heads": 2,
        "intermediate_size": 32,
        "max_position_embeddings": 16,
        **overrides,
    }
    names = set(inspect.signature(config_cls).parameters)
    return config_cls(**{k: v for k, v in small.items() if k in names})


def _wrappers() -> list[type[PretrainedModel]]:
    """Every head-bearing model class the text families export."""
    found: list[type[PretrainedModel]] = []
    for family in _TEXT_FAMILIES:
        module = importlib.import_module(f"lucid.models.text.{family}")
        for name in module.__all__:
            obj = getattr(module, name)
            if isinstance(obj, type) and issubclass(obj, TaskModel):
                found.append(obj)
    return found


_WRAPPERS = _wrappers()


def _build(wrapper: type[PretrainedModel], **overrides: object) -> Any:
    return wrapper(_tiny_config(wrapper.config_class, **overrides)).eval()


def _trunk(model: Any) -> Any:
    return getattr(model, type(model).base_model_prefix)


def _tied_head(model: Any) -> Any:
    """The output projection that shares the table, or ``None``."""
    if hasattr(model, "lm_head"):
        return model.lm_head
    predictions = getattr(getattr(model, "cls", None), "predictions", None)
    return None if predictions is None else predictions.decoder


class TestTaskWrapperEmbeddings:
    """A task wrapper exposes the table its trunk owns.

    Every text wrapper used to report ``None`` and refuse the swap, although
    each one owns a table: only the trunks had the accessors.
    """

    def test_the_sweep_sees_every_family(self) -> None:
        # Guards the parametrised tests below against matching nothing.
        families = {w.__module__.split(".")[3] for w in _WRAPPERS}
        assert families == set(_TEXT_FAMILIES)
        assert len(_WRAPPERS) >= 21

    @pytest.mark.parametrize("wrapper", _WRAPPERS, ids=lambda w: w.__name__)
    def test_reports_the_trunk_table(self, wrapper: type[PretrainedModel]) -> None:
        model = _build(wrapper)
        table = model.get_input_embeddings()
        assert isinstance(table, nn.Embedding)
        assert table is _trunk(model).get_input_embeddings()

    @pytest.mark.parametrize("wrapper", _WRAPPERS, ids=lambda w: w.__name__)
    def test_swaps_the_trunk_table(self, wrapper: type[PretrainedModel]) -> None:
        model = _build(wrapper)
        new = nn.Embedding(_V, _H)
        model.set_input_embeddings(new)
        assert model.get_input_embeddings() is new
        assert _trunk(model).get_input_embeddings() is new

    @pytest.mark.parametrize("wrapper", _WRAPPERS, ids=lambda w: w.__name__)
    def test_a_tied_head_follows_the_new_table(self, wrapper: type[PretrainedModel]) -> None:
        # Shared source/target tables, so the seq2seq head -- tied to the
        # target side -- is tied to the table being replaced as well.
        model = _build(wrapper, share_embeddings=True)
        head = _tied_head(model)
        new = nn.Embedding(_V + 10, _H)
        model.set_input_embeddings(new)
        if head is None:
            # No head reads the table, so nothing but the table itself may
            # hold the new weight — a classifier that quietly shared it would
            # have been checked by nothing when this was a skip.
            sharing = [
                n
                for n, m in model.named_modules()
                if m is not new and getattr(m, "weight", None) is new.weight
            ]
            assert not sharing, f"{wrapper.__name__}: {sharing} share the table"
            return
        assert head.weight is new.weight
        assert head.out_features == _V + 10

    @pytest.mark.parametrize(
        "family, name",
        [
            ("gpt", "GPTLMHeadModel"),
            ("gpt2", "GPT2LMHeadModel"),
            ("bert", "BERTForMaskedLM"),
            ("roformer", "RoFormerForMaskedLM"),
        ],
    )
    def test_logits_widen_to_a_larger_table(self, family: str, name: str) -> None:
        module = importlib.import_module(f"lucid.models.text.{family}")
        model = _build(getattr(module, name))
        model.set_input_embeddings(nn.Embedding(_V + 10, _H))
        ids = lucid.tensor([[_V + 5, 1, 2]]).long()  # only the new table has it
        with lucid.no_grad():
            logits = model(ids).logits
        assert tuple(logits.shape) == (1, 3, _V + 10)

    def test_a_tied_bias_keeps_its_entries(self) -> None:
        from lucid.models.text.bert import BERTForMaskedLM

        model = _build(BERTForMaskedLM)
        head = model.cls.predictions
        head.bias = nn.Parameter(lucid.arange(_V).float())
        model.set_input_embeddings(nn.Embedding(_V + 10, _H))
        assert head.bias.tolist() == [float(i) for i in range(_V)] + [0.0] * 10

    def test_an_untied_head_is_left_alone(self) -> None:
        from lucid.models.text.gpt import GPTLMHeadModel

        model = _build(GPTLMHeadModel, tie_word_embeddings=False)
        before = model.lm_head.weight
        new = nn.Embedding(_V, _H)
        model.set_input_embeddings(new)
        assert model.get_input_embeddings() is new
        assert model.lm_head.weight is before

    def test_seq2seq_head_stays_on_a_separate_target_table(self) -> None:
        from lucid.models.text.transformer import TransformerForSeq2SeqLM

        model = _build(TransformerForSeq2SeqLM, share_embeddings=False)
        target = model.transformer.tgt_tok_emb
        model.set_input_embeddings(nn.Embedding(_V, _H))
        # Only the source side is replaced; the head decodes into the target
        # vocabulary and keeps sharing that table.
        assert model.transformer.tgt_tok_emb is target
        assert model.lm_head.weight is target.weight

    def test_shared_tables_are_replaced_together(self) -> None:
        from lucid.models.text.transformer import TransformerForSeq2SeqLM

        model = _build(TransformerForSeq2SeqLM, share_embeddings=True)
        new = nn.Embedding(_V, _H)
        model.set_input_embeddings(new)
        assert model.transformer.tgt_tok_emb is new
        assert model.lm_head.weight is new.weight

    def test_an_encoder_only_trunk_keeps_its_alias(self) -> None:
        from lucid.models.text.transformer import TransformerForSequenceClassification

        model = _build(TransformerForSequenceClassification)
        new = nn.Embedding(_V, _H)
        model.set_input_embeddings(new)
        # No decoder is built, so the target name aliases the source table;
        # leaving it on the old table would keep that table in the checkpoint.
        assert model.transformer.tgt_tok_emb is new

    def test_a_non_embedding_is_refused(self) -> None:
        from lucid.models.text.bert import BERTForSequenceClassification

        model = _build(BERTForSequenceClassification)
        with pytest.raises(TypeError, match="nn.Embedding"):
            model.set_input_embeddings(nn.Linear(_H, _H))

    def test_a_wrapper_without_a_table_still_refuses(self) -> None:
        from lucid.models.vision.lenet import lenet_5_cls

        model = lenet_5_cls()
        assert model.get_input_embeddings() is None
        with pytest.raises(NotImplementedError, match="LeNetForImageClassification"):
            model.set_input_embeddings(nn.Embedding(_V, _H))
