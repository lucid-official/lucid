"""WordTokenizer — whitespace split + explicit UNK on OOV.

This file holds **both flavours** of the word-level tokenizer:

* :class:`WordTokenizer` — pure-Python reference implementation.
  Trivial to read / debug; suitable for any environment without
  the C++ engine available.

* :class:`WordTokenizerFast` — C++-backed via
  :class:`lucid._C.engine.utils.tokenizer.WordTokenizer`.  Same
  whitespace-split + dict-lookup semantics, but the hot loop runs
  in C++.  Use this in production where encode throughput matters.

Both flavours share the chunking rule of
:class:`~lucid.utils.tokenizer.WhitespaceTokenizer` (split on
runs of ASCII whitespace) but differ in OOV handling: instead of
silently dropping out-of-vocab words, they emit the configured
``unk`` special token.  When ``special_tokens.unk`` is ``None``
and an OOV word is hit, encoding raises :class:`ValueError` — use
:class:`~lucid.utils.tokenizer.WhitespaceTokenizer` if you want
silent drop semantics instead.

On-disk format
--------------
Both flavours write a single ``vocab.txt`` (one token per line,
line number = id) plus the shared ``tokenizer.json`` /
``special_tokens_map.json`` files produced by the base class.

Use case
--------
Common for classical NLP / language modelling baselines where an
explicit OOV signal in the id sequence is preferable to silent
information loss.
"""

from typing import Iterable, override

from lucid._C import engine as _C_engine

from lucid.utils.tokenizer._base import SpecialTokens
from lucid.utils.tokenizer._lookup_common import _FastVocabTokenizer, _VocabTokenizer
from lucid.utils.tokenizer._whitespace import _whitespace_split


class WordTokenizer(_VocabTokenizer):
    r"""Reference (pure-Python) word-level tokenizer with UNK fallback.

    Splits the input on ASCII whitespace runs, looks each word up
    in the vocab, and either emits the configured ``unk`` token or
    raises :class:`ValueError` on out-of-vocab inputs.

    For production / latency-sensitive use, prefer the matching
    :class:`WordTokenizerFast` — same algorithm, same vocab format,
    same OOV-error semantics, but the hot loop runs in C++.

    Parameters
    ----------
    vocab : dict[str, int], optional
        Pre-built token-string → id map.  If omitted, the
        tokenizer starts empty and :meth:`train` must be called
        before encoding.
    special_tokens : SpecialTokens, optional
        Must include a non-None ``unk`` if the caller plans to
        encode text containing OOV words.  Without it, OOV at
        encode time raises :class:`ValueError`.

    Notes
    -----
    :meth:`train` does **not** pre-seed an UNK token.  The caller
    is responsible for adding it to both the vocab and the
    :class:`SpecialTokens` registry before encoding OOV text.

    Examples
    --------
    >>> tok = WordTokenizer(special_tokens=SpecialTokens(unk="<unk>"))
    >>> tok.train(["the quick brown fox", "<unk>"])
    >>> tok.encode("the slow fox")  # 'slow' → unk
    [0, 4, 3]

    See Also
    --------
    WordTokenizerFast : C++-backed sibling with the same surface API.
    """

    _algo_name = "word"
    _model_type = "Word"

    def __init__(
        self, vocab: dict[str, int] | None = None, *, special_tokens: SpecialTokens | None = None
    ) -> None:
        r"""Construct a pure-Python word tokenizer.

        Parameters
        ----------
        vocab : dict[str, int] or None, optional
            Pre-built word → id map; ``None`` or empty leaves the
            tokenizer empty (call :meth:`train` to populate).
        special_tokens : SpecialTokens or None, optional, keyword-only
            Special-token registry; configure ``unk`` for OOV
            fallback at encode time.
        """
        self._init_vocab_tokenizer(vocab, special_tokens=special_tokens)

    @override
    def _encode_one(self, text: str) -> list[int]:
        """Algorithm-specific encode: whitespace-split + dict lookup."""
        out: list[int] = []
        unk = self.unk_token_id
        for w in _whitespace_split(text):
            tid = self._vocab.get(w)
            if tid is not None:
                out.append(tid)
            elif unk is not None:
                out.append(unk)
            else:
                raise ValueError(
                    f"WordTokenizer.encode: OOV word {w!r} encountered and "
                    f"there is no id to put in its place. What is missing is "
                    f"the token in the vocabulary, not the setting: "
                    f"``special_tokens.unk`` names it and ``train`` does not "
                    f"reserve it, so seed the vocabulary with it — "
                    f"``WordTokenizer(vocab={{'<unk>': 0}}, "
                    f"special_tokens=SpecialTokens(unk='<unk>'))`` — or "
                    f"encode text the vocabulary covers."
                )
        return out

    @override
    def _decode_one(self, ids: list[int]) -> str:
        """Algorithm-specific decode: join known ids with a single space."""
        return " ".join(self._id_to_token[i] for i in ids if i in self._id_to_token)

    def train(self, corpus: Iterable[str], *, vocab_size: int = 30_000) -> None:
        """Collect unique words in insertion order until ``vocab_size``.

        Same algorithm as :meth:`WhitespaceTokenizer.train` — pure
        insertion-order vocab build with a hard cap.  No frequency
        cut-off is applied.

        Training does **not** pre-seed an UNK token; the caller is
        responsible for adding it to the vocab + the special-token
        registry before encoding OOV text.

        Parameters
        ----------
        corpus : iterable of str
            Each item is one document.  Generators are consumed
            exactly once.
        vocab_size : int, default 30 000
            Hard cap on the number of distinct tokens kept.
        """
        self._train_insertion_vocab(corpus, _whitespace_split, vocab_size)


class WordTokenizerFast(_FastVocabTokenizer):
    r"""C++-backed word tokenizer with UNK fallback.

    Identical surface API and OOV-error semantics to
    :class:`WordTokenizer`; the whitespace scan + dict lookup run
    in C++ via :class:`lucid._C.engine.utils.tokenizer.WordTokenizer`.

    Parameters
    ----------
    vocab : dict[str, int], optional
        Pre-built token-string → id map.  Forwarded to the C++
        backend at construction; if omitted, the C++ tokenizer is
        constructed with an empty vocab and :meth:`train` must be
        called before encoding.
    special_tokens : SpecialTokens, optional
        Must include a non-None ``unk`` to allow OOV input; the id
        is mirrored into the C++ registry so the C++ OOV fall-back
        path agrees with the Python-side OOV check.

    Notes
    -----
    The C++ ``LookupTokenizer.encode`` silently substitutes UNK for
    OOV words when one is configured, or drops them otherwise.  To
    match the Python reference's explicit error for unconfigured
    UNK, :meth:`_encode_one` performs an OOV pre-check before
    delegating.

    See Also
    --------
    WordTokenizer : Pure-Python reference sibling.

    Examples
    --------
    >>> from lucid.utils.tokenizer import WordTokenizerFast
    >>> tok = WordTokenizerFast()
    >>> tok.train(["the quick brown fox", "the lazy dog sleeps"],
    ...           vocab_size=64)
    >>> tok.decode(tok.encode("the quick dog"))
    'the quick dog'
    """

    _algo_name = "word"
    _model_type = "Word"
    _sync_cpp_specials = True

    def __init__(
        self, vocab: dict[str, int] | None = None, *, special_tokens: SpecialTokens | None = None
    ) -> None:
        r"""Construct a C++-backed word tokenizer.

        Parameters
        ----------
        vocab : dict[str, int] or None, optional
            Pre-built word → id map.  ``None`` or empty constructs
            the C++ backend with an empty vocab.
        special_tokens : SpecialTokens or None, optional, keyword-only
            Special-token registry; mirrored into the C++ backend
            via :meth:`_sync_specials_to_cpp` so the C++ encode
            fall-back path sees the same UNK id.
        """
        v = dict(vocab) if vocab else {}
        cpp = (
            _C_engine.utils.tokenizer.WordTokenizer(v)
            if v
            else _C_engine.utils.tokenizer.WordTokenizer()
        )
        self._init_fast_vocab_tokenizer(cpp, v, special_tokens=special_tokens)

    @override
    def _encode_one(self, text: str) -> list[int]:
        """C++ encode path with a Python-side OOV pre-check."""
        # The C++ tokenizer falls back to UNK silently or drops the
        # token if UNK is unset.  To match the Python reference's
        # explicit error, check for OOV in Python before delegating.
        if self.unk_token_id is None:
            for w in _whitespace_split(text):
                if w not in self._vocab:
                    raise ValueError(
                        f"WordTokenizerFast.encode: OOV word {w!r} "
                        f"encountered and there is no id to put in its "
                        f"place. What is missing is the token in the "
                        f"vocabulary, not the setting: "
                        f"``special_tokens.unk`` names it and ``train`` "
                        f"does not reserve it, so seed the vocabulary with "
                        f"it — ``WordTokenizerFast(vocab={{'<unk>': 0}}, "
                        f"special_tokens=SpecialTokens(unk='<unk>'))`` — or "
                        f"encode text the vocabulary covers."
                    )
        return list(self._cpp.encode(text))

    def train(self, corpus: Iterable[str], *, vocab_size: int = 30_000) -> None:
        """Re-train in C++; refresh Python-side caches afterwards.

        Parameters
        ----------
        corpus : iterable of str
            Materialised into a ``list`` before being handed to the
            C++ binding.
        vocab_size : int, default 30 000
            Hard cap on the number of distinct tokens kept.
        """
        self._train_cpp(corpus, vocab_size)
