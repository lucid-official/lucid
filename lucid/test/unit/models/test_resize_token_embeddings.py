"""``resize_token_embeddings``, ``get_output_embeddings`` and a config that follows the table (LCD-1111).

A model's vocabulary lives in three places: the table that looks token ids up, the head that
scores them (tied to the table, or its own), and the config's ``vocab_size``.
``set_input_embeddings`` swapped the table and re-tied the head but left the config at the old
size, so a model with a 120-row table and head described itself as 100 -- and a
``save_pretrained`` copy could not load back.  There was also no way to *grow* a vocabulary while
keeping the rows a checkpoint trained.

The invariant is owned by ``PretrainedModel`` (a trunk declares ``_input_embeddings``, a head
``_output_embeddings``, a family its initialiser); this file holds every registered text class
to it, so a new family has to join the table below before it can land:

* the table, the head, the head's own bias and every config in the model move together, which a
  model rebuilt from the updated config proves by having the same ``state_dict`` shapes;
* the rows a model had are the rows it keeps, and the new ones come from the family's own
  initialiser rather than the framework default;
* the result trains and decodes -- a train step reaches the new rows, and ``generate`` runs on a
  decoder-only and an encoder-decoder model.
"""

import dataclasses
from pathlib import Path

import pytest

import lucid
import lucid.nn as nn
from lucid.models import AutoModelForLanguageModeling, PretrainedModel, create_model
from lucid.models._registry import _REGISTRY
from lucid.models.text.bert import BERTForMaskedLM
from lucid.models.text.t5 import T5ForSeq2SeqLM
from lucid.models.text.transformer import TransformerConfig, TransformerForSeq2SeqLM
from lucid.test._default_device import metal_available

_V, _H = 40, 16
_SHARED = dict(
    vocab_size=_V, hidden_size=_H, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32
)
# family -> config overrides that shrink one of its registered factories (H11: a config
# override is how a test gets a small model, the factory is the family's own).
_FAMILIES: dict[str, dict[str, object]] = {
    "gpt": dict(_SHARED, max_position_embeddings=16),
    "gpt2": dict(_SHARED, max_position_embeddings=16),
    "bert": dict(_SHARED, max_position_embeddings=16),
    "roformer": dict(_SHARED, max_position_embeddings=16),
    "t5": dict(_SHARED, num_decoder_layers=1, head_dim=8),
    "transformer": dict(_SHARED, num_decoder_layers=1, max_position_embeddings=16),
}
# The config field that is the standard deviation a family draws its embeddings from.
_INIT_STD = {
    "gpt": "initializer_range",
    "gpt2": "initializer_range",
    "bert": "initializer_range",
    "roformer": "initializer_range",
    "t5": "initializer_factor",
}


def _classes() -> list[tuple[type[PretrainedModel], str]]:
    """Every registered model class of the table's families, with the first factory that builds it."""
    chosen: dict[type[PretrainedModel], str] = {}
    for name, entry in sorted(_REGISTRY.items()):
        if entry.family in _FAMILIES and entry.model_class is not None:
            chosen.setdefault(entry.model_class, name)
    return sorted(chosen.items(), key=lambda kv: kv[0].__name__)


_CLASSES = _classes()


def _build(factory: str, **overrides: object) -> PretrainedModel:
    family = _REGISTRY[factory].family
    return create_model(factory, **{**_FAMILIES[family], **overrides}).eval()


def _scores(model: PretrainedModel, ids: lucid.Tensor) -> lucid.Tensor:
    """The vocabulary logits of a model that has a head, called the way its class is."""
    if isinstance(model, T5ForSeq2SeqLM):
        out = model(ids, labels=ids)
    elif isinstance(model, TransformerForSeq2SeqLM):
        out = model(ids, decoder_input_ids=ids)
    else:
        out = model(ids)
    logits = out.get("logits", out.get("prediction_logits"))
    assert logits is not None
    return logits


def _vocab_linears(model: PretrainedModel, vocab: int) -> list[nn.Module]:
    """The ``Linear`` layers that score ``vocab`` tokens."""
    return [
        m for _, m in model.named_modules() if isinstance(m, nn.Linear) and m.out_features == vocab
    ]


def _shapes(model: nn.Module) -> dict[str, tuple[int, ...]]:
    return {name: tuple(t.shape) for name, t in model.state_dict().items()}


def _ids(top: int) -> lucid.Tensor:
    return lucid.tensor([[1, 2, top - 1, 3]], dtype=lucid.int64)


# ── the registry decides which models this has to hold for ─────────────────────


def test_the_table_names_every_family_with_a_vocabulary() -> None:
    """A family whose config has a ``vocab_size`` has to be in the table, or be named here."""
    with_vocab = {
        entry.family
        for entry in _REGISTRY.values()
        if entry.model_class is not None
        and entry.model_class.config_class is not None
        and "vocab_size" in {f.name for f in dataclasses.fields(entry.model_class.config_class)}
    }
    assert with_vocab == set(_FAMILIES) | {"clip"}  # CLIP: see the xfail below
    assert len(_CLASSES) >= 25


@pytest.mark.parametrize(("cls", "factory"), _CLASSES, ids=lambda v: getattr(v, "__name__", v))
def test_a_model_reports_the_table_and_the_head_it_has(
    cls: type[PretrainedModel], factory: str
) -> None:
    model = _build(factory)
    table = model.get_input_embeddings()
    assert isinstance(table, nn.Embedding) and table.num_embeddings == _V
    head = model.get_output_embeddings()
    # Whatever scores the vocabulary is the head -- so a family that adds an LM head and forgets
    # to declare it (``lucid.peft`` finds the head through this) fails here, not in a user's model.
    assert _vocab_linears(model, _V) == ([] if head is None else [head])
    if head is not None:
        assert isinstance(head, nn.Linear) and head.weight is table.weight


# ── resizing ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("tie", [True, False], ids=["tied", "untied"])
@pytest.mark.parametrize("new_n", [_V + 9, _V - 5], ids=["grow", "shrink"])
@pytest.mark.parametrize(("cls", "factory"), _CLASSES, ids=lambda v: getattr(v, "__name__", v))
def test_resize_moves_the_table_the_head_and_every_config_together(
    cls: type[PretrainedModel], factory: str, new_n: int, tie: bool
) -> None:
    model = _build(factory, tie_word_embeddings=tie)
    old = model.get_input_embeddings()
    assert isinstance(old, nn.Embedding)
    before = old.weight.tolist()
    head = model.get_output_embeddings()
    head_before = None if head is None else head.weight.tolist()

    table = model.resize_token_embeddings(new_n)

    assert table is model.get_input_embeddings() and table is not old
    assert tuple(table.weight.shape) == (new_n, _H) and table.padding_idx == old.padding_idx
    keep = min(_V, new_n)
    assert table.weight[:keep].tolist() == before[:keep]
    assert tuple(old.weight.shape) == (_V, _H)  # the table it replaced is not touched

    # Every config the model holds -- a wrapper's and its trunk's -- says the new size.
    assert {m.config.vocab_size for m in model._config_chain()} == {new_n}

    head = model.get_output_embeddings()
    if head is not None:
        assert tuple(head.weight.shape) == (new_n, _H)
        assert (head.weight is table.weight) is tie
        if not tie:
            assert head_before is not None and head.weight[:keep].tolist() == head_before[:keep]

    # What the config now says is what the model is: a model rebuilt from it has these shapes,
    # which is what lets a saved copy load back.
    assert _shapes(model) == _shapes(type(model)(model.config))

    if head is not None:
        logits = _scores(model, _ids(new_n))
        assert int(logits.shape[-1]) == new_n


@pytest.mark.parametrize(("cls", "factory"), _CLASSES, ids=lambda v: getattr(v, "__name__", v))
def test_a_size_it_already_has_is_a_no_op(cls: type[PretrainedModel], factory: str) -> None:
    model = _build(factory)
    table = model.get_input_embeddings()
    config = model.config
    assert model.resize_token_embeddings(_V) is table
    assert model.get_input_embeddings() is table and model.config is config


@pytest.mark.parametrize("family", sorted(_INIT_STD))
def test_new_rows_come_from_the_familys_own_initialiser(family: str) -> None:
    """Not the framework default: a family whose standard deviation is 0.5 draws 0.5."""
    knob = _INIT_STD[family]
    factory = next(
        n for n, e in sorted(_REGISTRY.items()) if e.family == family and e.task == "base"
    )
    wide = 0.5 if knob == "initializer_range" else 3.0
    model = _build(factory, **{knob: wide})
    old = model.get_input_embeddings()
    assert isinstance(old, nn.Embedding)
    table = model.resize_token_embeddings(_V + 400, mean_resizing=False)
    new_rows = table.weight[_V:]
    assert abs(float(new_rows.std().item()) / wide - 1.0) < 0.1
    assert abs(float(new_rows.mean().item())) < 0.1 * wide
    # The rows it had are the rows it trained, untouched by that draw.
    assert table.weight[:_V].tolist() == old.weight.tolist()


def test_the_new_rows_of_an_untied_head_are_drawn_like_the_tables() -> None:
    model = _build("gpt_lm", tie_word_embeddings=False, initializer_range=0.5)
    model.resize_token_embeddings(_V + 400, mean_resizing=False)
    head = model.get_output_embeddings()
    assert head is not None
    assert abs(float(head.weight[_V:].std().item()) / 0.5 - 1.0) < 0.1


def test_a_bert_head_keeps_the_biases_it_has_and_zeroes_the_new_ones() -> None:
    """Zeros for a tied head's bias; for an untied head's, when it is not drawn (below)."""
    for tie in (True, False):
        model = _build("bert_base_mlm", tie_word_embeddings=tie)
        assert isinstance(model, BERTForMaskedLM)
        model.cls.predictions.bias = nn.Parameter(lucid.arange(_V).float())
        model.resize_token_embeddings(_V + 6, mean_resizing=False)
        assert model.cls.predictions.bias.tolist() == [float(i) for i in range(_V)] + [0.0] * 6
        model.resize_token_embeddings(_V - 3)
        assert model.cls.predictions.bias.tolist() == [float(i) for i in range(_V - 3)]


def test_a_frozen_table_stays_frozen() -> None:
    model = _build("gpt_lm")
    table = model.get_input_embeddings()
    assert table is not None
    table.weight.requires_grad = False
    new = model.resize_token_embeddings(_V + 3)
    assert new.weight.requires_grad is False


@pytest.mark.parametrize("device", ["cpu", "metal"])
@pytest.mark.parametrize("dtype", [lucid.float32, lucid.float16, lucid.bfloat16], ids=str)
def test_the_new_table_and_head_are_where_the_model_is(device: str, dtype: lucid.dtype) -> None:
    if device == "metal" and not metal_available():
        pytest.skip("Metal device not available on this host")
    model = _build("gpt_lm").to(dtype).to(device)
    table = model.resize_token_embeddings(_V + 5)
    for tensor in (table.weight, model.lm_head.weight):
        assert tensor.dtype == dtype and tensor.device.type == device
    ids = _ids(_V + 5).to(device)
    logits = model(ids).logits
    assert logits.dtype == dtype and tuple(logits.shape) == (1, 4, _V + 5)
    # Moved after the resize, the model carries the new table with it.
    again = model.to("cpu").resize_token_embeddings(_V + 7)
    assert again.weight.device.type == "cpu"


# ── what a resized model does ──────────────────────────────────────────────────


def test_a_resized_model_trains_and_the_new_rows_receive_the_update() -> None:
    lucid.manual_seed(0)
    model = _build("gpt_lm").train()
    table = model.resize_token_embeddings(_V + 6)
    ids = _ids(_V + 6)  # the last id is a row that did not exist
    before = table.weight[_V + 5].tolist()
    optimizer = lucid.optim.SGD(model.parameters(), lr=0.5)
    loss = model(ids, labels=ids).loss
    assert loss is not None and bool(lucid.isfinite(loss).item())
    optimizer.zero_grad()
    loss.backward()
    assert table.weight.grad is not None and tuple(table.weight.grad.shape) == (_V + 6, _H)
    optimizer.step()
    assert table.weight[_V + 5].tolist() != before


def test_a_decoder_only_model_generates_after_a_resize() -> None:
    model = _build("gpt_lm")
    model.resize_token_embeddings(_V + 6)
    prompt = lucid.tensor([[1, 2, _V + 5]], dtype=lucid.int64)
    for cache in ("dynamic", "static"):
        out = model.generate(prompt, max_new_tokens=4, cache_implementation=cache)
        assert tuple(out.shape) == (1, 7)
        assert int(out.max().item()) < _V + 6
    # And a sampler may land on the new rows: the head scores them.
    sampled = model.generate(prompt, max_new_tokens=4, do_sample=True, top_k=_V + 6)
    assert int(sampled.max().item()) < _V + 6


@pytest.mark.parametrize("factory", ["t5_small_seq2seq", "transformer_base_seq2seq"])
def test_an_encoder_decoder_model_generates_after_a_resize(factory: str) -> None:
    model = _build(factory)
    model.resize_token_embeddings(_V + 6)
    source = lucid.tensor([[4, 5, _V + 5]], dtype=lucid.int64)
    out = model.generate(source, max_new_tokens=4)
    assert int(out.shape[0]) == 1 and int(out.max().item()) < _V + 6
    assert int(_scores(model, _ids(_V + 6)).shape[-1]) == _V + 6


@pytest.mark.parametrize(
    "factory", ["gpt_lm", "bert_base_mlm", "t5_small_seq2seq", "transformer_base_seq2seq"]
)
@pytest.mark.parametrize("tie", [True, False], ids=["tied", "untied"])
def test_a_resized_model_saves_and_loads_back_as_itself(
    factory: str, tie: bool, tmp_path: Path
) -> None:
    lucid.manual_seed(0)
    model = _build(factory, tie_word_embeddings=tie)
    model.resize_token_embeddings(_V + 7)
    model.save_pretrained(str(tmp_path))
    ids = _ids(_V + 7)
    for loaded in (
        type(model).from_pretrained(str(tmp_path)),
        AutoModelForLanguageModeling.from_pretrained(str(tmp_path)),
    ):
        loaded.eval()
        assert loaded.config == model.config
        assert _scores(loaded, ids).tolist() == _scores(model, ids).tolist()


# ── a model with two vocabularies ─────────────────────────────────────────────


def _translation(**overrides: object) -> TransformerForSeq2SeqLM:
    config = TransformerConfig(
        **{**_FAMILIES["transformer"], "decoder_vocab_size": _V + 4, **overrides}
    )
    return TransformerForSeq2SeqLM(config).eval()


_TARGET = pytest.mark.parametrize(
    "decoder_vocab_size", [_V + 4, None], ids=["explicit-target", "implicit-target"]
)


@_TARGET
@pytest.mark.parametrize("tie", [True, False], ids=["tied", "untied"])
def test_separate_source_and_target_vocabularies_resize_the_source_only(
    tie: bool, decoder_vocab_size: int | None
) -> None:
    """With no ``decoder_vocab_size`` the target is "as big as the source", until the source moves."""
    target_rows = _V if decoder_vocab_size is None else decoder_vocab_size
    model = _translation(
        share_embeddings=False, tie_word_embeddings=tie, decoder_vocab_size=decoder_vocab_size
    )
    target = model.transformer.tgt_tok_emb
    head_before = model.lm_head
    model.resize_token_embeddings(_V + 9)
    assert tuple(model.transformer.src_tok_emb.weight.shape) == (_V + 9, _H)
    assert model.transformer.tgt_tok_emb is target
    assert model.lm_head is head_before and tuple(model.lm_head.weight.shape) == (target_rows, _H)
    assert (model.config.vocab_size, model.config.decoder_vocab_size) == (_V + 9, target_rows)
    assert _shapes(model) == _shapes(TransformerForSeq2SeqLM(model.config))


@_TARGET
@pytest.mark.parametrize("tie", [True, False], ids=["tied", "untied"])
def test_swapping_in_a_source_table_leaves_the_target_vocabulary_where_it_was(
    tie: bool, decoder_vocab_size: int | None
) -> None:
    target_rows = _V if decoder_vocab_size is None else decoder_vocab_size
    model = _translation(
        share_embeddings=False, tie_word_embeddings=tie, decoder_vocab_size=decoder_vocab_size
    )
    model.set_input_embeddings(nn.Embedding(_V + 20, _H))
    assert (model.config.vocab_size, model.config.decoder_vocab_size) == (_V + 20, target_rows)
    assert _shapes(model) == _shapes(TransformerForSeq2SeqLM(model.config))


@_TARGET
@pytest.mark.parametrize("tie", [True, False], ids=["tied", "untied"])
@pytest.mark.parametrize("how", ["resize", "swap"])
def test_a_translation_model_with_two_vocabularies_saves_and_loads_back(
    tie: bool, decoder_vocab_size: int | None, how: str, tmp_path: Path
) -> None:
    lucid.manual_seed(0)
    model = _translation(
        share_embeddings=False, tie_word_embeddings=tie, decoder_vocab_size=decoder_vocab_size
    )
    if how == "resize":
        model.resize_token_embeddings(_V + 9)
    else:
        model.set_input_embeddings(nn.Embedding(_V + 9, _H))
    model.save_pretrained(str(tmp_path))
    loaded = TransformerForSeq2SeqLM.from_pretrained(str(tmp_path)).eval()
    assert loaded.config == model.config
    source = lucid.tensor([[1, 2, _V + 8, 3]], dtype=lucid.int64)  # an id only the new table has
    target = lucid.tensor([[1, 2, 3, 4]], dtype=lucid.int64)
    expected = model(source, decoder_input_ids=target).logits
    assert loaded(source, decoder_input_ids=target).logits.tolist() == expected.tolist()


def test_a_shared_vocabulary_resizes_both_sides_and_the_config_that_names_both() -> None:
    model = _translation(share_embeddings=True, decoder_vocab_size=_V)
    model.resize_token_embeddings(_V + 3)
    assert model.transformer.tgt_tok_emb is model.transformer.src_tok_emb
    assert (model.config.vocab_size, model.config.decoder_vocab_size) == (_V + 3, _V + 3)
    assert tuple(model.lm_head.weight.shape) == (_V + 3, _H)


# ── set_input_embeddings ───────────────────────────────────────────────────────


@pytest.mark.parametrize(("cls", "factory"), _CLASSES, ids=lambda v: getattr(v, "__name__", v))
def test_swapping_a_table_in_keeps_the_config_in_step(
    cls: type[PretrainedModel], factory: str
) -> None:
    model = _build(factory)
    new = nn.Embedding(_V + 10, _H)
    model.set_input_embeddings(new)
    assert model.get_input_embeddings() is new
    assert {m.config.vocab_size for m in model._config_chain()} == {_V + 10}
    assert _shapes(model) == _shapes(type(model)(model.config))


def test_a_config_that_refuses_the_size_leaves_the_model_as_it_was() -> None:
    model = _build("gpt_lm")
    table, config, head = model.get_input_embeddings(), model.config, model.lm_head
    with pytest.raises(ValueError, match="vocab_size"):
        model.set_input_embeddings(nn.Embedding(0, _H))
    assert model.get_input_embeddings() is table and model.config is config
    assert model.transformer.config is config and model.lm_head is head
    assert model.lm_head.weight is table.weight


@pytest.mark.parametrize("bad", [0, -1, 2.5, True, "40"], ids=repr)
def test_a_size_that_is_not_a_positive_int_is_refused_and_nothing_changes(bad: object) -> None:
    model = _build("gpt_lm")
    table, config = model.get_input_embeddings(), model.config
    with pytest.raises((TypeError, ValueError), match="new_num_tokens"):
        model.resize_token_embeddings(bad)
    assert model.get_input_embeddings() is table and model.config is config


@pytest.mark.parametrize("name", ["lenet_5", "lenet_5_cls", "resnet_18"])
def test_a_model_with_no_vocabulary_refuses_to_resize(name: str) -> None:
    model = create_model(name)
    assert model.get_input_embeddings() is None and model.get_output_embeddings() is None
    with pytest.raises(NotImplementedError, match="resize_token_embeddings"):
        model.resize_token_embeddings(10)


# ── families this file cannot yet hold ─────────────────────────────────────────


@pytest.mark.xfail(
    strict=True,
    reason="CLIP token table follows after LCD-1107 lands: clip/_model.py is claimed by that "
    "branch, and CLIPModel needs `_input_embeddings = 'textual.token_embedding'` there.  Remove "
    "this mark with it; `[EOS]` is found as the argmax of the ids, so ids added past it break "
    "the pooling, which the docstring has to say.",
)
def test_clip_resizes_its_token_table() -> None:
    model = create_model(
        "clip_vit_base_32",
        image_size=32,
        patch_size=16,
        vision_layers=1,
        vision_width=32,
        vision_heads=2,
        context_length=8,
        vocab_size=64,
        text_width=32,
        text_heads=2,
        text_layers=1,
        embed_dim=16,
    )
    table = model.resize_token_embeddings(70)
    assert tuple(table.weight.shape) == (70, 32) and model.config.vocab_size == 70
