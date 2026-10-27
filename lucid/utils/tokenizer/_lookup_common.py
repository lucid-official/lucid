"""Internal helpers shared between the simple lookup-based
tokenizers (Char / Whitespace / Word / Regex).

Each of those tokenizers has the same persistence + special-token
loading boilerplate; factoring it here keeps the user-facing
modules focused on the algorithm.

Not part of the public API — names are underscore-prefixed at the
module level too (just to make it obvious in stack traces).
"""

from collections.abc import Callable, Iterable
import json
import os
from typing import TYPE_CHECKING, Self, override

from lucid._C import engine as _C_engine
from lucid.utils.tokenizer._base import _CANONICAL_SPECIAL_SLOTS, SpecialTokens, Tokenizer
from lucid.utils.tokenizer._bpe import _special_tokens_from_map


def _save_vocab_txt(vocab: dict[str, int], path: str) -> None:
    """Write a ``vocab.txt`` file: one token per line, line index = id.

    Same format BERT / WordPiece use; suitable for any lookup-style
    tokenizer.  Tokens are written in id-ascending order; any "holes"
    in the id range (rare) are filled with empty lines so id ↔ line
    mapping survives the round-trip.
    """
    if not vocab:
        with open(path, "w", encoding="utf-8") as f:
            pass
        return
    max_id = max(vocab.values())
    inv = {v: k for k, v in vocab.items()}
    with open(path, "w", encoding="utf-8") as f:
        for i in range(max_id + 1):
            f.write(inv.get(i, "") + "\n")


def _load_vocab_txt(path: str) -> dict[str, int]:
    """Inverse of :func:`_save_vocab_txt`.  Empty lines are skipped
    (the corresponding ids are simply absent from the returned dict)."""
    out: dict[str, int] = {}
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            tok = line.rstrip("\n")
            if tok:
                out[tok] = i
    return out


def _load_special_tokens_map(directory: str) -> SpecialTokens | None:
    """Read ``special_tokens_map.json`` from ``directory`` if present."""
    sp_path = os.path.join(directory, "special_tokens_map.json")
    if not os.path.isfile(sp_path):
        return None
    with open(sp_path, encoding="utf-8") as f:
        sp = json.load(f)
    return _special_tokens_from_map(sp)


def _load_vocab_txt_and_specials(
    directory: str, special_tokens: SpecialTokens | None, caller: str
) -> tuple[dict[str, int], SpecialTokens | None]:
    """Load ``vocab.txt`` + optional ``special_tokens_map.json`` from ``directory``."""
    vocab_path = os.path.join(directory, "vocab.txt")
    if not os.path.isfile(vocab_path):
        raise FileNotFoundError(f"{caller}.from_file: vocab.txt not found in {directory}")
    vocab = _load_vocab_txt(vocab_path)
    st = special_tokens or _load_special_tokens_map(directory)
    return vocab, st


def _build_insertion_vocab(
    corpus: Iterable[str], split_fn: Callable[[str], Iterable[str]], vocab_size: int
) -> dict[str, int]:
    """Collect unique chunks in encounter order up to ``vocab_size``."""
    vocab: dict[str, int] = {}
    next_id = 0
    for doc in corpus:
        for chunk in split_fn(doc):
            if chunk not in vocab:
                vocab[chunk] = next_id
                next_id += 1
                if next_id >= vocab_size:
                    break
        if next_id >= vocab_size:
            break
    return vocab


class _VocabTokenizer(Tokenizer):
    """Shared Python-vocab state + persistence for lookup tokenizers."""

    _algo_name: str = ""
    _model_type: str = ""
    _vocab: dict[str, int]
    _id_to_token: dict[int, str]

    if TYPE_CHECKING:

        def __init__(
            self,
            vocab: dict[str, int] | None = None,
            *,
            special_tokens: SpecialTokens | None = None,
        ) -> None: ...

    def _set_vocab(self, vocab: dict[str, int] | None) -> None:
        self._vocab = dict(vocab) if vocab else {}
        self._id_to_token = {v: k for k, v in self._vocab.items()}

    def _init_vocab_tokenizer(
        self, vocab: dict[str, int] | None = None, *, special_tokens: SpecialTokens | None = None
    ) -> None:
        self._set_vocab(vocab)
        super().__init__(special_tokens=special_tokens)

    @override
    @property
    def vocab_size(self) -> int:
        """Total number of tokens currently registered in the vocab."""
        return len(self._vocab)

    @override
    @property
    def algo(self) -> str:
        """Algorithm identifier."""
        return self._algo_name

    @override
    def get_vocab(self) -> dict[str, int]:
        """Return a shallow copy of the token → id map."""
        return dict(self._vocab)

    @override
    def id_to_token(self, token_id: int) -> str | None:
        """Return the surface string for ``token_id`` or ``None``."""
        return self._id_to_token.get(token_id)

    def _train_insertion_vocab(
        self, corpus: Iterable[str], split_fn: Callable[[str], Iterable[str]], vocab_size: int
    ) -> None:
        self._set_vocab(_build_insertion_vocab(corpus, split_fn, vocab_size))
        self._refresh_special_ids()

    @override
    def save(self, directory: str) -> None:
        """Persist as ``vocab.txt`` + unified ``tokenizer.json``."""
        os.makedirs(directory, exist_ok=True)
        _save_vocab_txt(self._vocab, os.path.join(directory, "vocab.txt"))
        super().save(directory)

    @override
    def _save_extras(self) -> dict[str, object]:
        """Emit the unified-format ``model`` block."""
        return {"model": {"type": self._model_type, "vocab": self._vocab}}

    @classmethod
    def from_file(cls, directory: str, *, special_tokens: SpecialTokens | None = None) -> Self:
        """Load from a directory containing ``vocab.txt``."""
        vocab, st = _load_vocab_txt_and_specials(directory, special_tokens, cls.__name__)
        return cls(vocab=vocab, special_tokens=st)

    from_pretrained = from_file


class _FastVocabTokenizer(_VocabTokenizer):
    """Shared C++-backed lookup tokenizer state and persistence."""

    _cpp: _C_engine.LookupTokenizer
    _sync_cpp_specials: bool = False

    def _init_fast_vocab_tokenizer(
        self,
        cpp_obj: _C_engine.LookupTokenizer,
        vocab: dict[str, int] | None = None,
        *,
        special_tokens: SpecialTokens | None = None,
    ) -> None:
        self._cpp = cpp_obj
        self._init_vocab_tokenizer(vocab, special_tokens=special_tokens)
        if self._sync_cpp_specials:
            self._sync_specials_to_cpp()

    def _sync_specials_to_cpp(self) -> None:
        """Mirror :attr:`_special_ids` into the C++ tokenizer registry."""
        st = _C_engine.utils.tokenizer.SpecialTokens()
        for name in _CANONICAL_SPECIAL_SLOTS:
            tid = self._special_ids.get(name)
            if tid is not None:
                setattr(st, name, tid)
        self._cpp.special_tokens = st

    @override
    @property
    def vocab_size(self) -> int:
        """Total number of tokens in the live C++ vocab."""
        return self._cpp.vocab_size()

    @override
    def _encode_one(self, text: str) -> list[int]:
        """Delegate to the C++ encode hot path."""
        return list(self._cpp.encode(text))

    @override
    def _decode_one(self, ids: list[int]) -> str:
        """Delegate to the C++ decode path."""
        return self._cpp.decode(list(ids))

    def _train_cpp(self, corpus: Iterable[str], vocab_size: int) -> None:
        self._cpp.train(list(corpus), vocab_size)
        self._set_vocab(dict(self._cpp.get_vocab()))
        self._refresh_special_ids()
        if self._sync_cpp_specials:
            self._sync_specials_to_cpp()
