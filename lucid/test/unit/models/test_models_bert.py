"""Unit tests for BERT (text family, Phase 4 first concrete model).

Uses tiny config (vocab=64, hidden=32, layers=2, heads=4) so the full suite
runs in well under a second even on Metal.
"""

import pytest

import lucid
from lucid.models import (
    BERTConfig,
    BERTForMaskedLM,
    BERTForQuestionAnswering,
    BERTForSequenceClassification,
    BERTForTokenClassification,
    BERTModel,
    create_model,
    is_model,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_VOCAB = 64
_HIDDEN = 32
_LAYERS = 2
_HEADS = 4
_INTER = 64
_MAX_POS = 32


def _tiny_config(**overrides: object) -> BERTConfig:
    base = {
        "vocab_size": _VOCAB,
        "hidden_size": _HIDDEN,
        "num_hidden_layers": _LAYERS,
        "num_attention_heads": _HEADS,
        "intermediate_size": _INTER,
        "max_position_embeddings": _MAX_POS,
        "type_vocab_size": 2,
        "num_labels": 3,
    }
    base.update(overrides)
    return BERTConfig(**base)  # type: ignore[arg-type]


def _ids(B: int = 2, T: int = 8) -> tuple[lucid.Tensor, lucid.Tensor, lucid.Tensor]:
    ids = lucid.tensor([[i % _VOCAB for i in range(T)] for _ in range(B)]).long()
    attn = lucid.tensor([[1] * T for _ in range(B)]).long()
    tt = lucid.tensor([[0] * T for _ in range(B)]).long()
    return ids, attn, tt


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────


class TestBERTConfig:
    def test_defaults_match_bert_base(self) -> None:
        cfg = BERTConfig()
        assert cfg.hidden_size == 768
        assert cfg.num_hidden_layers == 12
        assert cfg.num_attention_heads == 12
        assert cfg.intermediate_size == 3072
        assert cfg.type_vocab_size == 2
        assert cfg.model_type == "bert"

    def test_type_vocab_size_invariant(self) -> None:
        with pytest.raises(ValueError, match="type_vocab_size"):
            BERTConfig(type_vocab_size=0)

    def test_num_labels_invariant(self) -> None:
        with pytest.raises(ValueError, match="num_labels"):
            BERTConfig(num_labels=0)

    def test_classifier_dropout_range(self) -> None:
        with pytest.raises(ValueError, match="classifier_dropout"):
            BERTConfig(classifier_dropout=1.5)

    def test_head_divisibility_inherited(self) -> None:
        with pytest.raises(ValueError, match="divisible"):
            BERTConfig(hidden_size=10, num_attention_heads=3)


# ─────────────────────────────────────────────────────────────────────────────
# Forward shapes
# ─────────────────────────────────────────────────────────────────────────────


class TestBERTModelForward:
    def test_bare_encoder(self) -> None:
        cfg = _tiny_config()
        m = BERTModel(cfg).eval()
        ids, attn, tt = _ids(B=2, T=8)
        out = m(ids, attn, tt)
        assert tuple(out.last_hidden_state.shape) == (2, 8, _HIDDEN)
        assert tuple(out.pooler_output.shape) == (2, _HIDDEN)

    def test_attention_mask_optional(self) -> None:
        cfg = _tiny_config()
        m = BERTModel(cfg).eval()
        ids, _, _ = _ids(B=1, T=6)
        out = m(ids)  # no mask, no token_type_ids
        assert tuple(out.last_hidden_state.shape) == (1, 6, _HIDDEN)

    def test_input_embeddings_accessor(self) -> None:
        m = BERTModel(_tiny_config()).eval()
        emb = m.get_input_embeddings()
        assert emb is m.embeddings.word_embeddings


class TestBERTForMaskedLM:
    def test_logits_shape_and_loss(self) -> None:
        cfg = _tiny_config()
        m = BERTForMaskedLM(cfg).eval()
        ids, attn, tt = _ids(B=2, T=8)
        labels = lucid.tensor([[3, -100, 5, -100, -100, -100, -100, -100]] * 2).long()
        out = m(ids, attn, tt, labels=labels)
        assert tuple(out.logits.shape) == (2, 8, _VOCAB)
        assert out.loss is not None
        assert float(out.loss.item()) > 0.0

    def test_decoder_weight_is_tied(self) -> None:
        m = BERTForMaskedLM(_tiny_config()).eval()
        assert m.cls.predictions.decoder.weight is m.bert.embeddings.word_embeddings.weight

    def test_untied_when_disabled(self) -> None:
        cfg = _tiny_config(tie_word_embeddings=False)
        m = BERTForMaskedLM(cfg).eval()
        assert m.cls.predictions.decoder.weight is not m.bert.embeddings.word_embeddings.weight


class TestBERTForSequenceClassification:
    def test_logits_and_loss(self) -> None:
        cfg = _tiny_config()
        m = BERTForSequenceClassification(cfg).eval()
        ids, attn, tt = _ids(B=2, T=8)
        labels = lucid.tensor([1, 2]).long()
        out = m(ids, attn, tt, labels=labels)
        assert tuple(out.logits.shape) == (2, cfg.num_labels)
        assert out.loss is not None


class TestBERTForTokenClassification:
    def test_logits_shape(self) -> None:
        cfg = _tiny_config()
        m = BERTForTokenClassification(cfg).eval()
        ids, attn, tt = _ids(B=1, T=6)
        out = m(ids, attn, tt)
        assert tuple(out.logits.shape) == (1, 6, cfg.num_labels)


class TestBERTForQuestionAnswering:
    def test_logits_and_span_loss(self) -> None:
        cfg = _tiny_config()
        m = BERTForQuestionAnswering(cfg).eval()
        ids, attn, tt = _ids(B=2, T=8)
        starts = lucid.tensor([1, 3]).long()
        ends = lucid.tensor([4, 5]).long()
        out = m(ids, attn, tt, start_positions=starts, end_positions=ends)
        # Start and end scores arrive as named fields rather than fused into
        # a trailing size-2 axis the caller has to remember the order of.
        assert tuple(out.start_logits.shape) == (2, 8)
        assert tuple(out.end_logits.shape) == (2, 8)
        assert out.loss is not None

    def test_a_batch_with_no_answer_in_any_window_trains_to_zero(self) -> None:
        # Every span outside the window is clamped to the ignored sentinel;
        # the zero loss must still accept backward(), or one such batch
        # stops a SQuAD run.
        cfg = _tiny_config()
        m = BERTForQuestionAnswering(cfg)
        ids, attn, tt = _ids(B=2, T=8)
        outside = lucid.tensor([20, 30]).long()
        out = m(ids, attn, tt, start_positions=outside, end_positions=outside)
        assert out.loss is not None and float(out.loss.item()) == 0.0
        out.loss.backward()
        grad = m.qa_outputs.weight.grad
        assert grad is not None and float(grad.abs().sum().item()) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Sequence-classification head loads the pretrained encoder into .bert (random
# head); QA / token-cls heads ship *full* fine-tuned checkpoints (TestBERTTaskWeights)
# ─────────────────────────────────────────────────────────────────────────────


class TestBERTEncoderPretrainedTransfer:
    """``bert_*_cls`` loads the pretrained encoder into ``model.bert`` and leaves
    the classifier random (no canonical GLUE checkpoint).  These verify the
    encoder checkpoint's key layout is identical to the head's ``.bert``
    submodule — i.e. ``load_weight_entry(model.bert, entry)`` succeeds with
    ``strict=True`` at full scale — without any network access.  (QA / NER no
    longer use this path; they load full fine-tuned models — see
    :class:`TestBERTTaskWeights`.)
    """

    @pytest.mark.parametrize(
        "head_cls",
        [
            BERTForSequenceClassification,
            BERTForTokenClassification,
            BERTForQuestionAnswering,
        ],
    )
    def test_encoder_state_loads_into_bert_submodule(self, head_cls: type) -> None:
        cfg = _tiny_config()
        enc = BERTModel(cfg)
        head = head_cls(cfg)
        result = head.bert.load_state_dict(enc.state_dict(), strict=True)
        assert not list(getattr(result, "missing_keys", []) or [])
        assert not list(getattr(result, "unexpected_keys", []) or [])

    def test_transfer_copies_encoder_values_and_leaves_head_random(self) -> None:
        cfg = _tiny_config()
        enc = BERTModel(cfg)
        head = BERTForSequenceClassification(cfg)
        # snapshot the head's random init as a scalar checksum (Tensors have no
        # .copy(); the encoder load must not touch the classifier).
        head_checksum_before = float(head.classifier.weight.abs().sum().item())
        head.bert.load_state_dict(enc.state_dict(), strict=True)
        # an encoder param now equals the source ...
        key = "embeddings.word_embeddings.weight"
        a = head.bert.state_dict()[key]
        b = enc.state_dict()[key]
        assert float((a - b).abs().sum().item()) == 0.0
        # ... and the classifier head is untouched (still its random init).
        head_checksum_after = float(head.classifier.weight.abs().sum().item())
        assert head_checksum_after == head_checksum_before

    def test_cls_factory_pretrained_false_no_download(self) -> None:
        m = create_model(
            "bert_base_cls",
            pretrained=False,
            vocab_size=_VOCAB,
            hidden_size=_HIDDEN,
            num_hidden_layers=_LAYERS,
            num_attention_heads=_HEADS,
            intermediate_size=_INTER,
            max_position_embeddings=_MAX_POS,
        )
        assert isinstance(m, BERTForSequenceClassification)

    @pytest.mark.parametrize(
        "factory_name",
        ["bert_base_cls", "bert_large_cls", "bert_base_token_cls", "bert_base_qa"],
    )
    def test_factory_exposes_weights_kwarg(self, factory_name: str) -> None:
        import importlib
        import inspect

        mod = importlib.import_module("lucid.models.text.bert._pretrained")
        fn = getattr(mod, factory_name)
        assert "weights" in inspect.signature(fn).parameters


# ─────────────────────────────────────────────────────────────────────────────
# Fine-tuned task checkpoints — SQuAD (base + official large WWM) + CoNLL NER
# ─────────────────────────────────────────────────────────────────────────────


class TestBERTTaskWeights:
    """The QA + NER factories ship *full* fine-tuned checkpoints on canonical
    benchmarks.  Network-free: enum/registry shape + the NER cased-config switch.
    """

    def _enum(self, name: str) -> type:
        import lucid.models.weights as weights_ns

        return getattr(weights_ns, name)

    @pytest.mark.parametrize(
        ("enum_name", "tag", "slug", "num_classes", "license_id"),
        [
            ("BERTBaseQAWeights", "SQUAD_V1", "bert-base-squad", 2, "mit"),
            ("BERTLargeQAWeights", "SQUAD_V1", "bert-large-squad", 2, "apache-2.0"),
            ("BERTBaseNERWeights", "CONLL2003", "bert-base-ner", 9, "mit"),
        ],
    )
    def test_entry_fields(
        self,
        enum_name: str,
        tag: str,
        slug: str,
        num_classes: int,
        license_id: str,
    ) -> None:
        cls = self._enum(enum_name)
        assert cls.DEFAULT is cls[tag]
        e = cls[tag].entry
        assert e.num_classes == num_classes
        assert len(e.sha256) == 64 and set(e.sha256) != {"0"}
        assert f"lucid-dl/{slug}" in e.url
        assert f"/{tag}/" in e.url
        assert e.meta["license"] == license_id

    @pytest.mark.parametrize(
        ("factory", "enum_name"),
        [
            ("bert_base_qa", "BERTBaseQAWeights"),
            ("bert_large_qa", "BERTLargeQAWeights"),
            ("bert_base_token_cls", "BERTBaseNERWeights"),
        ],
    )
    def test_registered_for_factories(self, factory: str, enum_name: str) -> None:
        from lucid.weights import weights_for

        resolved = weights_for(factory)
        assert resolved is not None
        assert resolved.__name__ == enum_name

    def test_bert_large_qa_registered(self) -> None:
        assert is_model("bert_large_qa")

    def test_ner_random_init_is_uncased_base(self) -> None:
        # No weights/pretrained → plain uncased _CFG_BASE (vocab 30522, 2 labels).
        m = create_model(
            "bert_base_token_cls",
            pretrained=False,
            vocab_size=_VOCAB,
            hidden_size=_HIDDEN,
            num_hidden_layers=_LAYERS,
            num_attention_heads=_HEADS,
            intermediate_size=_INTER,
            max_position_embeddings=_MAX_POS,
        )
        assert isinstance(m, BERTForTokenClassification)
        assert m.config.vocab_size == _VOCAB

    def test_ner_cased_config_shape_matches_checkpoint(self) -> None:
        # The CoNLL checkpoint is cased (vocab 28 996, 9 BIO labels); the factory
        # builds that exact shape for pretrained loads.
        from lucid.models.text.bert._pretrained import _CFG_NER_CONLL

        assert _CFG_NER_CONLL.vocab_size == 28_996
        assert _CFG_NER_CONLL.num_labels == 9
        m = BERTForTokenClassification(_CFG_NER_CONLL)
        emb = m.bert.embeddings.word_embeddings.weight
        assert tuple(emb.shape) == (28_996, 768)
        assert tuple(m.classifier.weight.shape) == (9, 768)


# ─────────────────────────────────────────────────────────────────────────────
# Registry — factory dispatch
# ─────────────────────────────────────────────────────────────────────────────


class TestBERTRegistry:
    @pytest.mark.parametrize(
        "name",
        [
            "bert_tiny",
            "bert_mini",
            "bert_small",
            "bert_medium",
            "bert_base",
            "bert_large",
        ],
    )
    def test_base_variants_registered(self, name: str) -> None:
        assert is_model(name)

    @pytest.mark.parametrize(
        "name",
        [
            "bert_base_mlm",
            "bert_large_mlm",
            "bert_base_cls",
            "bert_large_cls",
            "bert_base_token_cls",
            "bert_base_qa",
            "bert_large_qa",
        ],
    )
    def test_task_heads_registered(self, name: str) -> None:
        assert is_model(name)

    def test_tiny_factory_instantiates_with_override(self) -> None:
        # Override to tiny dims so we don't allocate a real BERT-Tiny (~4M params)
        # in CI just to verify the factory plumbing.
        m = create_model(
            "bert_tiny",
            vocab_size=_VOCAB,
            hidden_size=_HIDDEN,
            num_hidden_layers=_LAYERS,
            num_attention_heads=_HEADS,
            intermediate_size=_INTER,
            max_position_embeddings=_MAX_POS,
        )
        assert isinstance(m, BERTModel)
        ids, _, _ = _ids(B=1, T=4)
        out = m.eval()(ids)
        assert tuple(out.last_hidden_state.shape) == (1, 4, _HIDDEN)


# ─────────────────────────────────────────────────────────────────────────────
# Extra task wrappers — PreTraining / NSP / CausalLM
# ─────────────────────────────────────────────────────────────────────────────


from lucid.models import (
    BERTForCausalLM,
    BERTForNextSentencePrediction,
    BERTForPreTraining,
)


class TestBERTForPreTraining:
    def test_combined_loss(self) -> None:
        cfg = _tiny_config()
        m = BERTForPreTraining(cfg).eval()
        ids, attn, tt = _ids(B=2, T=8)
        mlm_labels = lucid.tensor([[3, -100, 5, -100, -100, -100, -100, -100]] * 2).long()
        nsp_labels = lucid.tensor([1, 0]).long()
        out = m(ids, attn, tt, labels=mlm_labels, next_sentence_label=nsp_labels)
        assert tuple(out.prediction_logits.shape) == (2, 8, _VOCAB)
        assert tuple(out.seq_relationship_logits.shape) == (2, 2)
        assert out.loss is not None and out.mlm_loss is not None and out.nsp_loss is not None

    def test_mlm_only(self) -> None:
        m = BERTForPreTraining(_tiny_config()).eval()
        ids, attn, tt = _ids(B=1, T=6)
        out = m(ids, attn, tt, labels=ids)
        assert out.mlm_loss is not None
        assert out.nsp_loss is None
        assert out.loss is not None  # = mlm_loss

    def test_tied_decoder(self) -> None:
        m = BERTForPreTraining(_tiny_config()).eval()
        assert m.cls.predictions.decoder.weight is m.bert.embeddings.word_embeddings.weight


class TestBERTForNextSentencePrediction:
    def test_logits_and_loss(self) -> None:
        m = BERTForNextSentencePrediction(_tiny_config()).eval()
        ids, attn, tt = _ids(B=2, T=8)
        out = m(ids, attn, tt, labels=lucid.tensor([1, 0]).long())
        assert tuple(out.logits.shape) == (2, 2)
        assert out.loss is not None


class TestBERTForCausalLM:
    def test_logits_and_shift_loss(self) -> None:
        m = BERTForCausalLM(_tiny_config()).eval()
        ids, attn, tt = _ids(B=1, T=8)
        out = m(ids, attn, tt, labels=ids)
        assert tuple(out.logits.shape) == (1, 8, _VOCAB)
        assert out.loss is not None

    def test_causal_mask_prevents_leak(self) -> None:
        m = BERTForCausalLM(_tiny_config()).eval()
        # Every id must stay inside ``_VOCAB``; ids 40..80 used to run past the
        # embedding table, and the out-of-bounds read made this test flaky.
        ids_a = lucid.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]).long()
        ids_b = lucid.tensor([[1, 2, 3, 40, 50, 60, 61, 62]]).long()
        h_a = m(ids_a).logits
        h_b = m(ids_b).logits
        diff = float(((h_a[:, :3, :] - h_b[:, :3, :]) ** 2).sum().item())
        assert diff < 1e-6, f"Causal mask leaks in BERTForCausalLM: diff = {diff}"


# (factory, slug, hidden, num_params, task) — the eight shipped BERT
# checkpoints: six base encoders (Turc miniatures + Devlin base/large) and two
# masked-LM heads.  All Wikipedia + BookCorpus, uncased, 30 522-token vocab.
_SHIPPED = (
    ("BERTTinyWeights", "bert-tiny", 128, 4_385_920, "base"),
    ("BERTMiniWeights", "bert-mini", 256, 11_171_328, "base"),
    ("BERTSmallWeights", "bert-small", 512, 28_763_648, "base"),
    ("BERTMediumWeights", "bert-medium", 512, 41_373_184, "base"),
    ("BERTBaseWeights", "bert-base", 768, 109_482_240, "base"),
    ("BERTLargeWeights", "bert-large", 1024, 335_141_888, "base"),
    ("BERTBaseMLMWeights", "bert-base-mlm", 30_522, 109_514_298, "mlm"),
    ("BERTLargeMLMWeights", "bert-large-mlm", 30_522, 335_174_586, "mlm"),
)
_TAG = "WIKIPEDIA_BOOKSCORPUS"


class TestBERTWeightsEnums:
    """Static contract of the per-variant Weights enums — no network."""

    @staticmethod
    def _enum(name: str) -> type:
        import lucid.models.weights as weights_ns

        return getattr(weights_ns, name)

    def test_default_aliases(self) -> None:
        # Each enum's DEFAULT points at its single shipped tag member.
        for enum_name, *_rest in _SHIPPED:
            cls = self._enum(enum_name)
            assert cls.DEFAULT is cls[_TAG]

    def test_entry_fields(self) -> None:
        for enum_name, slug, num_classes, _nparams, _task in _SHIPPED:
            member = self._enum(enum_name)[_TAG]
            e = member.entry
            assert e.num_classes == num_classes
            assert len(e.sha256) == 64
            assert f"lucid-dl/{slug}" in e.url
            assert f"/{_TAG}/" in e.url
            assert member.meta["tag"] == _TAG
            assert member.meta["license"] == "apache-2.0"

    def test_registered_for_factories(self) -> None:
        # The enum is discoverable from its factory name via lucid.weights.
        from lucid.weights import weights_for

        for enum_name, slug, *_rest in _SHIPPED:
            factory = slug.replace("-", "_")
            resolved = weights_for(factory)
            assert resolved is not None
            assert resolved.__name__ == enum_name


class TestBERTHiddenStates:
    """``output_hidden_states`` is opt-in and must not perturb the output."""

    @staticmethod
    def _model():
        from lucid.models.text.bert._config import BERTConfig
        from lucid.models.text.bert._model import BERTModel

        return BERTModel(
            BERTConfig(
                num_hidden_layers=3,
                hidden_size=32,
                num_attention_heads=2,
                intermediate_size=64,
                vocab_size=100,
            )
        ).eval()

    def test_off_by_default(self) -> None:
        m = self._model()
        assert m(lucid.tensor([[1, 2, 3, 4]])).hidden_states is None

    def test_collects_embedding_plus_every_layer(self) -> None:
        m = self._model()
        out = m(lucid.tensor([[1, 2, 3, 4]]), output_hidden_states=True)
        assert out.hidden_states is not None
        # The reference puts the embedding output first, so it is layers + 1.
        assert len(out.hidden_states) == 4
        assert all(tuple(h.shape) == (1, 4, 32) for h in out.hidden_states)

    def test_last_state_is_the_returned_one(self) -> None:
        m = self._model()
        out = m(lucid.tensor([[1, 2, 3, 4]]), output_hidden_states=True)
        assert out.hidden_states is not None
        delta = (out.hidden_states[-1] - out.last_hidden_state).abs().max()
        assert float(delta.item()) == 0.0

    def test_first_state_is_the_embedding_not_a_copy(self) -> None:
        """A stack that returned the same tensor N+1 times would also pass
        the length check, so pin that the ends actually differ."""
        m = self._model()
        out = m(lucid.tensor([[1, 2, 3, 4]]), output_hidden_states=True)
        assert out.hidden_states is not None
        spread = (out.hidden_states[0] - out.hidden_states[-1]).abs().max()
        assert float(spread.item()) > 0.0

    def test_flag_does_not_change_the_result(self) -> None:
        m = self._model()
        ids = lucid.tensor([[1, 2, 3, 4]])
        off = m(ids).last_hidden_state
        on = m(ids, output_hidden_states=True).last_hidden_state
        assert float((off - on).abs().max().item()) == 0.0


class TestBERTAttentionWeights:
    """``output_attentions`` / ``head_mask`` and the unfused path they need."""

    @staticmethod
    def _model():
        from lucid.models.text.bert._config import BERTConfig
        from lucid.models.text.bert._model import BERTModel

        return BERTModel(
            BERTConfig(
                num_hidden_layers=3,
                hidden_size=32,
                num_attention_heads=4,
                intermediate_size=64,
                vocab_size=100,
            )
        ).eval()

    def test_off_by_default(self) -> None:
        assert self._model()(lucid.tensor([[1, 2, 3, 4]])).attentions is None

    def test_one_matrix_per_layer_and_rows_are_a_distribution(self) -> None:
        out = self._model()(lucid.tensor([[1, 2, 3, 4]]), output_attentions=True)
        assert out.attentions is not None
        assert len(out.attentions) == 3
        assert tuple(out.attentions[0].shape) == (1, 4, 4, 4)
        # Post-softmax, every query row sums to 1.
        assert abs(float(out.attentions[0].sum(dim=-1).mean().item()) - 1.0) < 1e-5

    def test_unfused_path_matches_the_fused_one(self) -> None:
        """Asking for the weights must not change the answer."""
        m = self._model()
        ids = lucid.tensor([[1, 2, 3, 4]])
        fused = m(ids).last_hidden_state
        unfused = m(ids, output_attentions=True).last_hidden_state
        assert float((fused - unfused).abs().max().item()) < 1e-5

    def test_all_ones_head_mask_is_a_no_op(self) -> None:
        m = self._model()
        ids = lucid.tensor([[1, 2, 3, 4]])
        base = m(ids).last_hidden_state
        masked = m(ids, head_mask=lucid.ones(4)).last_hidden_state
        assert float((base - masked).abs().max().item()) < 1e-5

    def test_zeroed_head_contributes_no_weight(self) -> None:
        """A masked head must be gone, not merely down-weighted."""
        m = self._model()
        out = m(
            lucid.tensor([[1, 2, 3, 4]]),
            head_mask=lucid.tensor([1.0, 0.0, 1.0, 1.0]),
            output_attentions=True,
        )
        assert out.attentions is not None
        per_head = out.attentions[0].sum(dim=(0, 2, 3)).tolist()
        assert per_head[1] == 0.0
        assert all(v > 0.0 for i, v in enumerate(per_head) if i != 1)

    def test_masking_every_head_changes_the_output(self) -> None:
        m = self._model()
        ids = lucid.tensor([[1, 2, 3, 4]])
        base = m(ids).last_hidden_state
        killed = m(ids, head_mask=lucid.zeros(4)).last_hidden_state
        assert float((base - killed).abs().max().item()) > 0.0
