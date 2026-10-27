"""CharTokenizer — one Unicode codepoint = one token.

Vocab is learned from the corpus (every distinct codepoint gets a
fresh id in insertion order).  Sequences are about as long as
:class:`ByteTokenizer`'s for ASCII text but ~4× shorter for
non-ASCII (one id per codepoint rather than per byte).

Trade-off vs :class:`ByteTokenizer`:
* **Char** — needs training; OOV (codepoint not in vocab) → UNK.
* **Byte** — no training; no OOV.

Use Char when you want a closed vocab tied to the training corpus
(language modelling on a fixed alphabet, character RNNs); use Byte
when you need an open-vocab universal fallback.
"""

from typing import Iterable, override

from lucid._C import engine as _C_engine

from lucid.utils.tokenizer._base import SpecialTokens
from lucid.utils.tokenizer._lookup_common import _FastVocabTokenizer, _VocabTokenizer


def _split_codepoints(text: str) -> list[str]:
    """Split Python ``str`` into per-codepoint chunks.

    Python's ``str`` is already codepoint-indexed, so this is a
    cheap list comprehension; matches the C++ ``CharTokenizer``'s
    UTF-8 codepoint splitter byte-for-byte.
    """
    return list(text)


class CharTokenizer(_VocabTokenizer):
    """Reference (pure-Python) character-level tokenizer.

    Parameters
    ----------
    vocab : dict[str, int], optional
        Pre-built vocab; if omitted, an empty tokenizer is
        constructed (call :meth:`train` before encoding).
    special_tokens : SpecialTokens, optional
        Special-token registry.  Configuring ``unk`` enables OOV
        fallback (otherwise unknown codepoints are dropped).

    Examples
    --------
    >>> from lucid.utils.tokenizer import CharTokenizer
    >>> tok = CharTokenizer()
    >>> tok.train(["the quick brown fox", "the lazy dog sleeps"],
    ...           vocab_size=64)
    >>> tok.encode("the quick dog")[:5]
    [0, 1, 2, 3, 4]
    >>> tok.decode(tok.encode("the quick dog"))
    'the quick dog'
    """

    _algo_name = "char"
    _model_type = "Char"

    def __init__(
        self, vocab: dict[str, int] | None = None, *, special_tokens: SpecialTokens | None = None
    ) -> None:
        r"""Construct a pure-Python character tokenizer.

        Parameters
        ----------
        vocab : dict[str, int] or None, optional
            Pre-built codepoint → id map; ``None`` or empty means
            empty tokenizer (call :meth:`train` to populate).
        special_tokens : SpecialTokens or None, optional, keyword-only
            Special-token registry; configure ``unk`` for OOV
            fallback during encode.
        """
        self._init_vocab_tokenizer(vocab, special_tokens=special_tokens)

    @override
    def _encode_one(self, text: str) -> list[int]:
        """Per-codepoint vocab lookup with optional UNK fallback."""
        unk = self.unk_token_id
        out: list[int] = []
        for ch in _split_codepoints(text):
            tid = self._vocab.get(ch)
            if tid is not None:
                out.append(tid)
            elif unk is not None:
                out.append(unk)
        return out

    @override
    def _decode_one(self, ids: list[int]) -> str:
        """Concatenate codepoint surface forms for known ids."""
        return "".join(self._id_to_token[i] for i in ids if i in self._id_to_token)

    def train(self, corpus: Iterable[str], *, vocab_size: int = 1000) -> None:
        """Build vocab by collecting unique codepoints in insertion order.

        Stops when ``vocab_size`` distinct codepoints have been
        observed.

        Parameters
        ----------
        corpus : iterable of str
            Each item is one document.  Generators are consumed once.
        vocab_size : int, default 1000
            Maximum number of distinct codepoints to retain.
        """
        self._train_insertion_vocab(corpus, _split_codepoints, vocab_size)


class CharTokenizerFast(_FastVocabTokenizer):
    """C++-backed character tokenizer; bit-identical to :class:`CharTokenizer`.

    Same vocab format + semantics as :class:`CharTokenizer`; the
    per-codepoint encode loop runs in C++ via
    :class:`lucid._C.engine.utils.tokenizer.CharTokenizer`.

    Parameters
    ----------
    vocab : dict[str, int], optional
        Pre-built codepoint → id map.  If omitted the tokenizer is
        constructed empty (call :meth:`train` before encoding).
    special_tokens : SpecialTokens, optional
        Special-token registry; configure ``unk`` for OOV fallback.

    Examples
    --------
    >>> from lucid.utils.tokenizer import CharTokenizerFast
    >>> tok = CharTokenizerFast()
    >>> tok.train(["the quick brown fox"], vocab_size=64)
    >>> tok.decode(tok.encode("the quick"))
    'the quick'
    """

    _algo_name = "char"
    _model_type = "Char"

    def __init__(
        self, vocab: dict[str, int] | None = None, *, special_tokens: SpecialTokens | None = None
    ) -> None:
        r"""Construct a C++-backed character tokenizer.

        Parameters
        ----------
        vocab : dict[str, int] or None, optional
            Pre-built codepoint → id map.  ``None`` or empty
            constructs the C++ backend with an empty vocab; call
            :meth:`train` to populate.
        special_tokens : SpecialTokens or None, optional, keyword-only
            Special-token registry.

        Notes
        -----
        Caches the Python-side vocab so :meth:`get_vocab` and
        :meth:`id_to_token` avoid a binding round-trip; both halves
        are kept in sync by :meth:`train`.
        """
        v = dict(vocab) if vocab else {}
        cpp = (
            _C_engine.utils.tokenizer.CharTokenizer(v)
            if v
            else _C_engine.utils.tokenizer.CharTokenizer()
        )
        self._init_fast_vocab_tokenizer(cpp, v, special_tokens=special_tokens)

    def train(self, corpus: Iterable[str], *, vocab_size: int = 1000) -> None:
        """Train in C++; mirror the resulting vocab back into Python state.

        Parameters
        ----------
        corpus : iterable of str
            Materialised into a list before hand-off to C++.
        vocab_size : int, default 1000
            Maximum number of distinct codepoints to retain.
        """
        self._train_cpp(corpus, vocab_size)
