"""``resize_token_embeddings(new_num_tokens, pad_to_multiple_of, mean_resizing)`` (LCD-1111).

The reference's signature, and its defaults: ``new_num_tokens=None`` keeps the size,
``pad_to_multiple_of`` rounds it up, and ``mean_resizing=True`` draws each new row from a normal
with the old rows' mean and covariance (the covariance scaled by 1e-9, and just the mean where it
is not positive definite), a head's own bias from a normal with the old entries' mean and spread.
``mean_resizing=False`` is the family's own initialiser, which ``test_resize_token_embeddings.py``
holds every family to.  The draws follow ``lucid.manual_seed``.
"""

import pytest

import lucid
import lucid.nn as nn
from lucid.models import PretrainedModel, create_model
from lucid.models._embeddings import resized_projection, resized_vocab_size
from lucid.models.text.bert import BERTForMaskedLM

_V, _H = 40, 16
_TINY = dict(
    vocab_size=_V,
    hidden_size=_H,
    num_hidden_layers=1,
    num_attention_heads=2,
    intermediate_size=32,
    max_position_embeddings=16,
)


def _lm(**overrides: object) -> PretrainedModel:
    return create_model("gpt_lm", **{**_TINY, **overrides}).eval()


def _deviation_from_mean(rows: lucid.Tensor, old: lucid.Tensor) -> float:
    """The largest distance of any entry of ``rows`` from the mean of ``old``'s rows."""
    return float((rows - old.mean(dim=0)).abs().max().item())


# ── mean_resizing ──────────────────────────────────────────────────────────────


def test_new_rows_sit_a_hair_from_the_mean_of_the_old_ones() -> None:
    """A positive definite covariance (40 rows, 16 dimensions), scaled by 1e-9: near the mean, not on it."""
    model = _lm(initializer_range=0.5)
    old = model.get_input_embeddings().weight.detach()
    new = model.resize_token_embeddings(_V + 9).weight[_V:]
    deviation = _deviation_from_mean(new, old)
    assert 0.0 < deviation < 1e-3  # an std of 0.5 would put them ~0.5 away
    assert _deviation_from_mean(old, old) > 0.1


def test_new_rows_are_the_mean_where_the_covariance_is_not_positive_definite() -> None:
    """Ten rows span at most nine of sixteen dimensions: no distribution, so the mean itself."""
    model = _lm(vocab_size=10)
    old = model.get_input_embeddings().weight.detach()
    new = model.resize_token_embeddings(15).weight[10:]
    assert _deviation_from_mean(new, old) < 1e-7
    assert new[0].tolist() == new[4].tolist()


def test_mean_resizing_false_is_the_familys_initialiser_not_the_mean() -> None:
    model = _lm(initializer_range=0.5)
    old = model.get_input_embeddings().weight.detach()
    new = model.resize_token_embeddings(_V + 400, mean_resizing=False).weight[_V:]
    assert _deviation_from_mean(new, old) > 1.0  # N(0, 0.5^2) over 6400 entries
    assert abs(float(new.std().item()) / 0.5 - 1.0) < 0.1


def test_mean_resizing_is_the_default() -> None:
    lucid.manual_seed(0)
    explicit, default = _lm(), _lm()
    default.load_state_dict(explicit.state_dict())
    lucid.manual_seed(5)
    a = explicit.resize_token_embeddings(_V + 3, mean_resizing=True).weight.tolist()
    lucid.manual_seed(5)
    b = default.resize_token_embeddings(_V + 3).weight.tolist()
    assert a == b


def test_the_draw_follows_the_seed() -> None:
    def resized(seed: int) -> list[list[float]]:
        lucid.manual_seed(0)
        model = _lm()
        lucid.manual_seed(seed)
        return model.resize_token_embeddings(_V + 5).weight[_V:].tolist()

    assert resized(1) == resized(1)
    assert resized(1) != resized(2)


def test_shrinking_draws_nothing() -> None:
    lucid.manual_seed(0)
    model = _lm()
    before = model.get_input_embeddings().weight.tolist()
    after = model.resize_token_embeddings(_V - 6).weight.tolist()
    assert after == before[: _V - 6]


def test_an_untied_head_is_drawn_from_its_own_rows() -> None:
    model = _lm(tie_word_embeddings=False)
    head = model.get_output_embeddings()
    assert head is not None
    old = head.weight.detach()
    model.resize_token_embeddings(_V + 9)
    new = model.get_output_embeddings().weight
    assert tuple(new.shape) == (_V + 9, _H)
    assert 0.0 < _deviation_from_mean(new[_V:], old) < 1e-3
    assert new[:_V].tolist() == old.tolist()


def test_a_tied_head_gets_no_draw_of_its_own() -> None:
    model = _lm()
    table = model.resize_token_embeddings(_V + 9)
    assert model.get_output_embeddings().weight is table.weight


@pytest.mark.parametrize("tie", [True, False], ids=["tied", "untied"])
def test_a_bert_biass_new_entries_are_zero_if_tied_and_drawn_if_it_is_the_heads_own(
    tie: bool,
) -> None:
    model = create_model("bert_base_mlm", **_TINY, tie_word_embeddings=tie).eval()
    assert isinstance(model, BERTForMaskedLM)
    model.cls.predictions.bias = nn.Parameter(lucid.arange(_V).float())
    model.resize_token_embeddings(_V + 400)
    bias = model.cls.predictions.bias
    assert bias[:_V].tolist() == [float(i) for i in range(_V)]
    new = bias[_V:]
    if tie:
        assert new.tolist() == [0.0] * 400
        return
    old = lucid.arange(_V).float()
    assert abs(float(new.mean().item()) - float(old.mean().item())) < 3.0  # 11.7 / sqrt(400)
    assert abs(float(new.std().item()) / float(old.std().item()) - 1.0) < 0.2


def test_a_linear_heads_own_bias_is_drawn_like_the_sibling_bias() -> None:
    lucid.manual_seed(0)
    old = nn.Linear(4, 30)
    old.bias = nn.Parameter(lucid.arange(30).float())
    new = resized_projection(old, 430, lambda table: None, mean_resizing=True)
    assert new.bias[:30].tolist() == old.bias.tolist()
    tail = new.bias[30:]
    assert abs(float(tail.mean().item()) - 14.5) < 3.0
    assert abs(float(tail.std().item()) / float(old.bias.std().item()) - 1.0) < 0.2


@pytest.mark.parametrize("dtype", [lucid.float16, lucid.bfloat16], ids=str)
def test_the_statistics_are_taken_in_float32_and_the_rows_keep_the_tables_dtype(
    dtype: lucid.dtype,
) -> None:
    model = _lm().to(dtype)
    table = model.resize_token_embeddings(_V + 4)
    assert table.weight.dtype == dtype
    new = table.weight[_V:].float()
    assert bool(lucid.isfinite(new).all().item())


# ── new_num_tokens and pad_to_multiple_of ──────────────────────────────────────


@pytest.mark.parametrize(
    ("requested", "pad", "size"),
    [
        (None, None, _V),
        (None, 8, _V),
        (None, 64, 64),
        (50, None, 50),
        (50, 8, 56),
        (56, 8, 56),
        (41, 1, 41),
        (3, 4, 4),
    ],
)
def test_the_size_a_resize_asks_for(requested: int | None, pad: int | None, size: int) -> None:
    assert resized_vocab_size(_V, requested, pad) == size


def test_a_resize_with_no_size_keeps_the_table_and_the_config() -> None:
    model = _lm()
    table, config = model.get_input_embeddings(), model.config
    assert model.resize_token_embeddings() is table
    assert model.resize_token_embeddings(None) is table
    assert model.resize_token_embeddings(None, None, False) is table
    assert model.config is config


def test_padding_rounds_up_both_the_requested_and_the_current_size() -> None:
    model = _lm()
    table = model.resize_token_embeddings(50, pad_to_multiple_of=8)
    assert tuple(table.weight.shape) == (56, _H) and model.config.vocab_size == 56
    assert model.lm_head.weight is table.weight  # the tied head follows the padded table
    again = model.resize_token_embeddings(pad_to_multiple_of=64)
    assert tuple(again.weight.shape) == (64, _H) and model.config.vocab_size == 64


def test_the_arguments_may_be_given_by_position() -> None:
    model = _lm()
    table = model.resize_token_embeddings(50, 8, False)
    assert int(table.weight.shape[0]) == 56


@pytest.mark.parametrize("bad", [0, -8, 2.5, True, "8"], ids=repr)
def test_a_pad_multiple_that_is_not_a_positive_int_is_refused_and_nothing_changes(
    bad: object,
) -> None:
    model = _lm()
    table, config = model.get_input_embeddings(), model.config
    with pytest.raises(ValueError, match="multiple of .*not a positive integer"):
        model.resize_token_embeddings(50, pad_to_multiple_of=bad)
    assert model.get_input_embeddings() is table and model.config is config


@pytest.mark.parametrize("bad", [0, -1, 2.5, True, "40"], ids=repr)
def test_a_size_that_is_not_a_positive_int_is_still_refused_with_a_pad(bad: object) -> None:
    with pytest.raises((TypeError, ValueError), match="new_num_tokens"):
        resized_vocab_size(_V, bad, 8)
