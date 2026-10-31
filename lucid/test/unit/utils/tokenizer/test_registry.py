"""Tokenizers by name: a class statement registers, a name or a directory resolves.

Nothing is downloaded, so what a name can do is name a class — the vocabulary comes in the
keyword arguments or from a directory.
"""

import json
from pathlib import Path
from typing import override

import pytest

import lucid.models  # noqa: F401  # the zoo's tokenizers register as the families are imported
from lucid.models.text.bert import BERTTokenizer, BERTTokenizerFast
from lucid.models.text.gpt2 import GPT2Tokenizer
from lucid.models.text.roformer import RoFormerTokenizer
from lucid.utils.tokenizer import Tokenizer, WordPieceTokenizer
from lucid.utils.tokenizer._registry import _Registry, resolve_tokenizer

_VOCAB = {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "[PAD]": 3, "[MASK]": 4, "a": 5, "b": 6}


class _Bare(Tokenizer):
    """A tokenizer with no way to be read from a directory."""

    @override
    def _encode_one(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    @override
    def _decode_one(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)

    @property
    @override
    def vocab_size(self) -> int:
        return 256

    @property
    @override
    def algo(self) -> str:
        return "bare"


class ShoutTokenizer(Tokenizer, names=("shout", "Loud_Name")):
    """A user tokenizer registered by its class statement alone."""

    @override
    def _encode_one(self, text: str) -> list[int]:
        return [ord(c) for c in text.upper()]

    @override
    def _decode_one(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)

    @property
    @override
    def vocab_size(self) -> int:
        return 256

    @property
    @override
    def algo(self) -> str:
        return "shout"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("bert", BERTTokenizer),
        ("BERT", BERTTokenizer),
        ("BERTTokenizer", BERTTokenizer),
        ("bert-tokenizer-fast", BERTTokenizerFast),
        ("roformer", RoFormerTokenizer),
        ("GPT2", GPT2Tokenizer),
        ("wordpiece_tokenizer", WordPieceTokenizer),
    ],
)
def test_the_zoo_names_resolve_to_their_classes(name: str, expected: type[Tokenizer]) -> None:
    assert _Registry.lookup(name) is expected


def test_a_class_statement_registers_the_class_and_its_aliases() -> None:
    for name in ("ShoutTokenizer", "shout", "loud name", "LOUD-NAME"):
        assert _Registry.lookup(name) is ShoutTokenizer
    assert resolve_tokenizer("shout").encode("ab", add_special_tokens=False) == [65, 66]


def test_a_private_class_is_not_registered() -> None:
    assert "bare" not in _Registry.names()
    with pytest.raises(ValueError, match="no tokenizer is registered as '_Bare'"):
        _Registry.lookup("_Bare")


def test_a_name_builds_the_tokenizer_from_its_keyword_arguments() -> None:
    tokenizer = resolve_tokenizer("bert", vocab=_VOCAB)
    assert isinstance(tokenizer, BERTTokenizer)
    assert tokenizer.encode("a b") == [1, 5, 6, 2]  # [CLS] a b [SEP]


def test_an_unknown_name_lists_the_registered_ones() -> None:
    with pytest.raises(ValueError, match="registered: .*bert.*roformer") as raised:
        resolve_tokenizer("bert-base-uncased")
    assert "without a download" in str(raised.value)


def test_a_name_without_what_the_class_needs_says_so() -> None:
    with pytest.raises(ValueError, match="'gpt2' cannot be built.*ships no vocabulary"):
        resolve_tokenizer("gpt2")
    with pytest.raises(ValueError, match="unexpected keyword argument 'nonsense'"):
        resolve_tokenizer("bert", vocab=_VOCAB, nonsense=1)


class TestDirectory:
    def _saved(self, directory: Path, tokenizer_class: str | None) -> None:
        BERTTokenizer(_VOCAB).save(str(directory))
        if tokenizer_class is not None:
            (directory / "tokenizer_config.json").write_text(
                json.dumps({"tokenizer_class": tokenizer_class}), encoding="utf-8"
            )

    def test_a_directory_is_read_by_the_class_its_config_names(self, tmp_path: Path) -> None:
        self._saved(tmp_path, "BertTokenizer")
        tokenizer = resolve_tokenizer(str(tmp_path))
        assert isinstance(tokenizer, WordPieceTokenizer)
        assert tokenizer.get_vocab() == _VOCAB

    def test_keyword_arguments_go_to_the_class_that_reads_it(self, tmp_path: Path) -> None:
        self._saved(tmp_path, "BERTTokenizer")
        short = resolve_tokenizer(str(tmp_path), max_chars_per_word=3)
        assert isinstance(short, BERTTokenizer)
        assert short.encode("aaaa", add_special_tokens=False) == [0]  # too long a word is [UNK]

    def test_a_directory_must_name_its_class(self, tmp_path: Path) -> None:
        self._saved(tmp_path, None)
        with pytest.raises(ValueError, match="no tokenizer_config.json"):
            resolve_tokenizer(str(tmp_path))
        (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        with pytest.raises(TypeError, match="tokenizer_class must be a string"):
            resolve_tokenizer(str(tmp_path))

    def test_a_class_that_cannot_be_read_from_a_directory_is_named(self, tmp_path: Path) -> None:
        (tmp_path / "tokenizer_config.json").write_text(
            json.dumps({"tokenizer_class": "ShoutTokenizer"}), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="ShoutTokenizer cannot be read from a directory"):
            resolve_tokenizer(str(tmp_path))
