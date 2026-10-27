"""The tokenizers, and what each does with a token it has never seen.

``utils/tokenizer/_base.py`` sat at 71.4%.  The shared machinery — the
id/token conversions, the batch helpers, the padding and truncation in
``__call__`` — was largely dark.

The part worth pinning is not the happy path.  It is that the family
answers the same question three different ways, each deliberately and
each documented, and one of those answers is silence.
"""

import pytest

import lucid.utils.tokenizer as tokenizers
from lucid.utils.tokenizer import SpecialTokens

WORDS = ["[PAD]", "[UNK]", "the", "quick", "brown", "fox", "lazy", "dog", "a", "and"]
VOCAB = {word: index for index, word in enumerate(WORDS)}
SPECIALS = SpecialTokens(pad="[PAD]", unk="[UNK]")


def _word():
    return tokenizers.WordTokenizer(VOCAB, special_tokens=SPECIALS)


# ── the round trip ────────────────────────────────────────────────────────────


def test_a_known_sentence_round_trips():
    tokenizer = _word()
    ids = tokenizer.encode("the quick dog")
    assert ids == [2, 3, 7]
    assert tokenizer.decode(ids) == "the quick dog"


def test_every_id_is_inside_the_vocabulary():
    tokenizer = _word()
    assert tokenizer.vocab_size == len(WORDS)
    ids = tokenizer.encode("the quick brown fox and the lazy dog")
    assert all(0 <= i < tokenizer.vocab_size for i in ids)


def test_get_vocab_returns_what_it_was_given():
    assert _word().get_vocab() == VOCAB


def test_id_to_token_inverts_the_vocabulary():
    tokenizer = _word()
    for token, index in tokenizer.get_vocab().items():
        assert tokenizer.id_to_token(index) == token


def test_an_id_past_the_end_is_none_rather_than_a_crash():
    assert _word().id_to_token(10**6) is None


def test_the_conversions_accept_one_or_many():
    tokenizer = _word()
    assert tokenizer.convert_tokens_to_ids("dog") == 7
    assert tokenizer.convert_tokens_to_ids(["the", "dog"]) == [2, 7]
    assert tokenizer.convert_ids_to_tokens(7) == "dog"
    assert tokenizer.convert_ids_to_tokens([2, 7]) == ["the", "dog"]


def test_the_special_ids_resolve_and_the_unset_ones_stay_none():
    tokenizer = _word()
    assert tokenizer.pad_token_id == 0
    assert tokenizer.unk_token_id == 1
    assert tokenizer.bos_token_id is None
    assert set(tokenizer.all_special_ids) == {0, 1}


def test_the_batch_helpers_are_the_single_ones_mapped():
    tokenizer = _word()
    encoded = tokenizer.encode_batch(["the dog", "a quick fox"])
    assert [len(row) for row in encoded] == [2, 3]
    assert tokenizer.decode_batch(encoded) == ["the dog", "a quick fox"]


# ── the three answers to an unknown token ─────────────────────────────────────


def test_an_unknown_word_becomes_unk_when_one_is_configured():
    tokenizer = _word()
    ids = tokenizer.encode("the zzzz dog")
    assert len(ids) == 3
    assert ids[1] == VOCAB["[UNK]"]


def test_a_word_tokenizer_without_unk_refuses_the_unknown_word():
    """The loud answer, and the one that names the word."""
    with pytest.raises(ValueError, match="OOV"):
        tokenizers.WordTokenizer(VOCAB).encode("the zzzz dog")


def test_a_whitespace_tokenizer_without_unk_drops_it_silently():
    """The quiet answer.  Documented — ``_whitespace.py`` says "OOV words
    are silently dropped" — and pinned here because the same family
    answers this three different ways, so anyone reasoning from one of
    them is wrong about the others.

    The sharp edge is what it does with an *empty* vocabulary: every
    token is unknown, so every sentence encodes to nothing at all.
    """
    tokenizer = tokenizers.WhitespaceTokenizer({"the": 0, "dog": 1})
    assert tokenizer.encode("the zzzz dog") == [0, 1]  # the middle word is gone

    empty = tokenizers.WhitespaceTokenizer()
    assert empty.vocab_size == 0
    assert empty.encode("the quick dog") == []
    assert empty.decode(empty.encode("the quick dog")) == ""


def test_a_char_tokenizer_with_no_vocabulary_encodes_to_nothing():
    """Same contract as whitespace, and the same edge: a tokenizer built
    without a vocabulary turns every text into an empty sequence, and a
    model then trains on nothing while every shape stays plausible."""
    empty = tokenizers.CharTokenizer()
    assert empty.vocab_size == 0
    assert empty.encode("hello") == []


def test_a_char_tokenizer_with_a_vocabulary_round_trips():
    tokenizer = tokenizers.CharTokenizer({c: i for i, c in enumerate("helo")})
    assert tokenizer.decode(tokenizer.encode("hello")) == "hello"


# ── the byte tokenizer needs no vocabulary at all ─────────────────────────────


@pytest.mark.parametrize("text", ["hello", "héllo", "日本", "café ☕", ""])
def test_the_byte_tokenizer_round_trips_any_text(text):
    """It has no out-of-vocabulary case by construction, which is the
    whole reason to reach for it."""
    tokenizer = tokenizers.ByteTokenizer()
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_the_byte_tokenizer_covers_every_byte():
    tokenizer = tokenizers.ByteTokenizer()
    assert tokenizer.vocab_size >= 256
    ascii_text = "".join(chr(i) for i in range(1, 128))
    assert tokenizer.decode(tokenizer.encode(ascii_text)) == ascii_text


def test_a_multi_byte_character_costs_more_than_one_id():
    tokenizer = tokenizers.ByteTokenizer()
    assert len(tokenizer.encode("é")) == 2  # two UTF-8 bytes
    assert len(tokenizer.encode("日")) == 3


# ── __call__: padding, masking, truncation ────────────────────────────────────


def test_calling_the_tokenizer_returns_input_ids():
    result = _word()("the quick dog")
    assert "input_ids" in result
    assert result["input_ids"] == [2, 3, 7]


def test_calling_it_on_a_batch_returns_a_list_of_rows():
    result = _word()(["the dog", "a quick fox"])
    assert isinstance(result["input_ids"][0], list)
    assert [len(r) for r in result["input_ids"]] == [2, 3]


def test_padding_squares_the_batch_and_masks_what_it_added():
    """The mask is the half that matters — padded positions must be
    marked, or attention reads the padding as content."""
    result = _word()(["the dog", "the quick brown fox"], padding=True)
    ids = result["input_ids"]
    assert len({len(row) for row in ids}) == 1
    assert ids[0][-1] == _word().pad_token_id
    assert result["attention_mask"][0] == [1, 1, 0, 0]
    assert result["attention_mask"][1] == [1, 1, 1, 1]


def test_padding_without_a_pad_token_is_refused():
    """Refused rather than padded with a zero that means a real word."""
    with pytest.raises(ValueError, match="pad"):
        tokenizers.WordTokenizer(VOCAB)(["a", "the quick fox"], padding=True)


def test_truncation_caps_the_length():
    result = _word()(["the quick brown fox and the lazy dog"], truncation=True, max_length=3)
    assert all(len(row) <= 3 for row in result["input_ids"])


def test_truncation_keeps_the_front():
    result = _word()(["the quick brown fox"], truncation=True, max_length=2)
    assert result["input_ids"][0] == [2, 3]


def test_padding_and_truncation_together_give_exactly_max_length():
    result = _word()(
        ["the dog", "the quick brown fox and the lazy dog"],
        padding=True,
        truncation=True,
        max_length=3,
    )
    assert [len(row) for row in result["input_ids"]] == [3, 3]
