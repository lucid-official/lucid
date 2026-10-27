"""WhitespaceTokenizer — one token per whitespace-delimited word.

The simplest "word-level" tokenizer.  Vocab is the set of distinct
whitespace-separated tokens encountered during training.  OOV words
are silently dropped (use :class:`WordTokenizer` instead if you
want explicit UNK fallback).

Decode joins ids with a single space — original whitespace
(multiple spaces, tabs, newlines) is not preserved.

Useful as a baseline / debugging tool; production text models almost
always use a subword tokenizer (BPE / WordPiece / Unigram) for OOV
robustness + better compression.
"""

from typing import Iterable, override

from lucid._C import engine as _C_engine

from lucid.utils.tokenizer._base import SpecialTokens
from lucid.utils.tokenizer._lookup_common import _FastVocabTokenizer, _VocabTokenizer


def _whitespace_split(text: str) -> list[str]:
    """Built-in Python ``str.split()`` — splits on Unicode whitespace
    and drops empty chunks.  Matches the C++ side's behaviour
    (which uses the same ASCII whitespace set; Unicode whitespace is
    treated as opaque chars by the C++ but in practice ``str.split()``
    is the same for the corpora these tokenizers are used on)."""
    return text.split()


class WhitespaceTokenizer(_VocabTokenizer):
    """Reference (pure-Python) whitespace-split tokenizer.

    Parameters
    ----------
    vocab : dict[str, int], optional
        Pre-built vocab.  Empty = empty tokenizer; call :meth:`train`
        first.
    special_tokens : SpecialTokens, optional
        Special-token registry.

    Examples
    --------
    >>> from lucid.utils.tokenizer import WhitespaceTokenizer
    >>> tok = WhitespaceTokenizer()
    >>> tok.train(["the quick brown fox", "the lazy dog sleeps"],
    ...           vocab_size=64)
    >>> tok.encode("the quick dog")
    [0, 1, 5]
    >>> tok.decode(tok.encode("the quick dog"))
    'the quick dog'
    """

    _algo_name = "whitespace"
    _model_type = "Whitespace"

    def __init__(
        self, vocab: dict[str, int] | None = None, *, special_tokens: SpecialTokens | None = None
    ) -> None:
        r"""Construct a pure-Python whitespace tokenizer.

        Parameters
        ----------
        vocab : dict[str, int] or None, optional
            Pre-built vocab.  ``None`` or empty = empty tokenizer;
            call :meth:`train` to populate.
        special_tokens : SpecialTokens or None, optional, keyword-only
            Special-token registry.
        """
        self._init_vocab_tokenizer(vocab, special_tokens=special_tokens)

    @override
    def _encode_one(self, text: str) -> list[int]:
        """Whitespace-split + vocab lookup; OOV words are silently dropped."""
        out: list[int] = []
        for w in _whitespace_split(text):
            tid = self._vocab.get(w)
            if tid is not None:
                out.append(tid)
            # OOV silently dropped — that's the WhitespaceTokenizer
            # contract.  Use WordTokenizer for explicit UNK fallback.
        return out

    @override
    def _decode_one(self, ids: list[int]) -> str:
        """Join known token surface forms with single spaces."""
        return " ".join(self._id_to_token[i] for i in ids if i in self._id_to_token)

    def train(self, corpus: Iterable[str], *, vocab_size: int = 30_000) -> None:
        """Collect unique whitespace-split words in insertion order.

        Parameters
        ----------
        corpus : iterable of str
            Each item is one document.  Generators are consumed once.
        vocab_size : int, default 30 000
            Maximum number of distinct words to retain; training stops
            as soon as the cap is hit.
        """
        self._train_insertion_vocab(corpus, _whitespace_split, vocab_size)


class WhitespaceTokenizerFast(_FastVocabTokenizer):
    """C++-backed whitespace tokenizer; bit-identical to
    :class:`WhitespaceTokenizer`.

    Same vocab format + OOV-dropping contract as
    :class:`WhitespaceTokenizer`; the encode loop runs in C++ via
    the engine ``WhitespaceTokenizer`` binding.

    Parameters
    ----------
    vocab : dict[str, int], optional
        Pre-built word → id map.  Empty = call :meth:`train` first.
    special_tokens : SpecialTokens, optional
        Special-token registry.

    Examples
    --------
    >>> from lucid.utils.tokenizer import WhitespaceTokenizerFast
    >>> tok = WhitespaceTokenizerFast()
    >>> tok.train(["the quick brown fox", "the lazy dog sleeps"],
    ...           vocab_size=64)
    >>> tok.encode("the quick dog")
    [0, 1, 5]
    """

    _algo_name = "whitespace"
    _model_type = "Whitespace"

    def __init__(
        self, vocab: dict[str, int] | None = None, *, special_tokens: SpecialTokens | None = None
    ) -> None:
        r"""Construct a C++-backed whitespace tokenizer.

        Parameters
        ----------
        vocab : dict[str, int] or None, optional
            Pre-built word → id map.  ``None`` or empty constructs
            an empty C++ backend; call :meth:`train` to populate.
        special_tokens : SpecialTokens or None, optional, keyword-only
            Special-token registry.
        """
        v = dict(vocab) if vocab else {}
        cpp = (
            _C_engine.utils.tokenizer.WhitespaceTokenizer(v)
            if v
            else _C_engine.utils.tokenizer.WhitespaceTokenizer()
        )
        self._init_fast_vocab_tokenizer(cpp, v, special_tokens=special_tokens)

    def train(self, corpus: Iterable[str], *, vocab_size: int = 30_000) -> None:
        r"""Collect unique whitespace-split words via the C++ trainer.

        Parameters
        ----------
        corpus : iterable of str
            Documents to scan.  Materialised into a list before
            crossing into C++.
        vocab_size : int, default 30 000
            Maximum number of distinct words to retain.

        Notes
        -----
        Mutates `_vocab` / `_id_to_token` in place after the C++
        side finishes; re-resolves special-token ids against the
        new vocab.
        """
        self._train_cpp(corpus, vocab_size)
