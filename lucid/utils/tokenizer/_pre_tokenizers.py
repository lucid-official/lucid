"""Pre-tokenizers — split the normalised text into algorithm-ready chunks.

Every BPE / WordPiece / Unigram implementation expects its
``_encode_one`` hook to receive **one chunk** at a time (typically
one word).  The pre-tokenizer chain is what produces those chunks
from the raw post-normalised string; it sits between the
:class:`~lucid.utils.tokenizer._normalizers.Normalizer` step and
the algorithm-specific encoder.

The standard chains map to algorithm families:

* :class:`WhitespaceSplit` — split on Unicode whitespace + drop
  empties.  Used by classical BPE (without byte-level remapping)
  and by the SentencePiece pre-stage of LLaMA / T5.
* :class:`WhitespacePunctuationSplit` — split on whitespace AND
  separate every punctuation character as its own chunk.  Standard
  for WordPiece (BERT / DistilBERT / RoFormer / ALBERT) — matches
  the reference framework's ``BasicTokenizer`` after normalisation.
* :class:`ByteLevel` — re-map every UTF-8 byte to a printable
  Unicode character (the GPT-2 byte-to-Unicode table), then split
  before every ``Ġ``-prefixed token (``Ġ`` = the original space
  byte under the mapping).  Used by GPT-2 / RoBERTa / BART
  byte-level BPE.  Guarantees the BPE algorithm never sees raw
  bytes or whitespace.

**Output contract.** Every pre-tokenizer returns
``list[tuple[str, tuple[int, int]]]`` where each tuple is
``(chunk, (start, end))`` — the chunk text plus its codepoint
offsets into the input string.  Offsets feed HF-compatible
``return_offsets_mapping`` for span-level downstream tasks (NER,
extractive QA, etc.).
"""

import re
import unicodedata
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import override

from lucid.utils.tokenizer._config import (
    WHITE_SPACE,
    as_list,
    as_object,
    get_bool,
    get_str,
    pattern_from_config,
    type_of,
)

# Public type alias for pre-tokenizer outputs.
Chunk = tuple[str, tuple[int, int]]


class PreTokenizer(ABC):
    """Abstract base — every pre-tokenizer implements :meth:`pre_tokenize`.

    Subclasses are callable: ``pre_tokenizer(text)`` is sugar for
    ``pre_tokenizer.pre_tokenize(text)``.

    See Also
    --------
    WhitespaceSplit, WhitespacePunctuationSplit, ByteLevel
    """

    @abstractmethod
    def pre_tokenize(self, text: str) -> list[Chunk]:
        """Split ``text`` into ``(chunk, (start, end))`` tuples.

        Parameters
        ----------
        text : str
            Post-normalised input string.

        Returns
        -------
        list of (str, (int, int))
            One tuple per chunk.  Offsets are codepoint indices into
            ``text`` (Python ``str`` indexing is codepoint-based —
            the convention used by every HF-compatible tokenizer for
            ``return_offsets_mapping``).
        """

    def __call__(self, text: str) -> list[Chunk]:
        """Sugar for :meth:`pre_tokenize`."""
        return self.pre_tokenize(text)

    def _pre_tokenize_at(self, text: str, offset: int) -> list[Chunk]:
        """Pre-tokenize a piece that starts at ``offset`` in the full input.

        The default shifts :meth:`pre_tokenize`'s offsets; a pre-tokenizer
        whose output depends on *where* the piece sits — :class:`Metaspace`
        with ``prepend_scheme="first"`` marks only the piece at offset 0 —
        overrides this.  :class:`Sequence` and the Unigram pipeline call it
        so that position survives being split up first.
        """
        return [(c, (s + offset, e + offset)) for c, (s, e) in self.pre_tokenize(text)]


class WhitespaceSplit(PreTokenizer):
    """Split on Unicode whitespace, dropping empty chunks.

    The default for classical BPE pre-tokenization (LLaMA, T5
    after the SentencePiece pre-stage, GPT-3-style word BPE, ...).
    Uses Python's :meth:`str.isspace` so all Unicode whitespace
    categories count (regular space, tab, newline, NBSP, etc.).

    See Also
    --------
    WhitespacePunctuationSplit : Also splits on punctuation.
    ByteLevel : Byte-level GPT-2 / RoBERTa pre-tokenizer.
    """

    @override
    def pre_tokenize(self, text: str) -> list[Chunk]:
        """Split ``text`` on whitespace runs, dropping empty chunks."""
        out: list[Chunk] = []
        i = 0
        n = len(text)
        while i < n:
            # Skip leading whitespace.
            while i < n and text[i].isspace():
                i += 1
            if i >= n:
                break
            start = i
            # Consume word.
            while i < n and not text[i].isspace():
                i += 1
            out.append((text[start:i], (start, i)))
        return out


class WhitespacePunctuationSplit(PreTokenizer):
    """Split on whitespace AND emit each punctuation character as its
    own chunk.

    The standard pre-tokenizer for WordPiece (BERT / RoFormer /
    ALBERT / DistilBERT) — matches the canonical ``BasicTokenizer``
    after normalisation, ensuring punctuation is never merged with
    adjacent word characters during sub-word splitting.

    Notes
    -----
    Punctuation detection follows the BERT convention: ASCII
    punctuation ranges (``!``-``/``, ``:``-``@``, ``[``-`````,
    ``{``-``~``) plus every Unicode ``P*`` category — see
    :func:`_is_punctuation`.
    """

    @override
    def pre_tokenize(self, text: str) -> list[Chunk]:
        """Walk ``text`` emitting word / punctuation chunks separately."""
        out: list[Chunk] = []
        i = 0
        n = len(text)
        while i < n:
            # Skip whitespace.
            while i < n and text[i].isspace():
                i += 1
            if i >= n:
                break
            ch = text[i]
            if _is_punctuation(ch):
                # Punctuation is always its own chunk.
                out.append((ch, (i, i + 1)))
                i += 1
            else:
                start = i
                while i < n and not text[i].isspace() and not _is_punctuation(text[i]):
                    i += 1
                out.append((text[start:i], (start, i)))
        return out


class ByteLevel(PreTokenizer):
    r"""GPT-2 byte-level pre-tokenizer.

    Re-maps every UTF-8 byte to a printable Unicode codepoint via
    the canonical GPT-2 byte-to-Unicode table, then chunks the
    result along word / digit / punctuation boundaries (each chunk
    may absorb a single leading space, which after the byte mapping
    appears as the famous ``Ġ`` prefix — ``Ġ`` = ``chr(0x100 + 0)``,
    the image of byte ``0x20`` under the table).

    **Byte-to-Unicode mapping.** Bytes that are already printable
    ASCII / Latin-1 (``0x21``-``0x7E`` + ``0xA1``-``0xAC`` +
    ``0xAE``-``0xFF``) map to themselves; every other byte
    (``0x00``-``0x20``, ``0x7F``-``0xA0``, ``0xAD``) is shifted
    into the ``0x100``+ block where it lands on a printable,
    non-whitespace codepoint.  This guarantees that the BPE
    algorithm downstream sees no whitespace and no control
    characters — making "byte-level BPE" identical to "BPE over
    the mapped Unicode string".

    The mapping is bijective on bytes ``0x00``-``0xFF``, so a full
    inverse (:meth:`decode_bytes`) reconstructs the original byte
    sequence exactly.  This is what lets GPT-2 / RoBERTa / BART
    losslessly round-trip arbitrary bytes (including emoji, control
    characters, invalid UTF-8) through their tokenizers.

    Parameters
    ----------
    add_prefix_space : bool, default ``False``
        Whether to prepend a space to the input so the first word
        also gets a ``Ġ`` prefix after byte-encoding.  GPT-2 default
        is ``False``; RoBERTa / BART default is ``True``.

    Notes
    -----
    The byte-to-Unicode table is built lazily on first use and
    cached on the class — construction cost is paid once per process.

    See Also
    --------
    WhitespaceSplit : Simpler whitespace-only chunker for classical BPE.
    """

    # Compiled once per process — the table is small (256 entries).
    _byte_encoder: dict[int, str] = {}
    _byte_decoder: dict[str, int] = {}

    def __init__(self, *, add_prefix_space: bool = False) -> None:
        r"""Record the prefix-space flag and warm the byte tables.

        Parameters
        ----------
        add_prefix_space : bool, default False, keyword-only
            Prepend a space before pre-tokenizing so the first word
            also receives the ``Ġ`` word-start marker.  GPT-2 uses
            ``False``, RoBERTa / BART use ``True``.

        Notes
        -----
        Lazily builds the class-level byte-encoder / byte-decoder
        tables on first instantiation; subsequent calls reuse the
        cached maps.
        """
        self._add_prefix_space = add_prefix_space
        if not ByteLevel._byte_encoder:
            ByteLevel._build_byte_tables_()

    @classmethod
    def _build_byte_tables_(cls) -> None:
        """Build the GPT-2 byte-to-unicode mapping (lazy, once).

        Bytes 0x21–0x7E + 0xA1–0xAC + 0xAE–0xFF map to themselves;
        every other byte (0–0x20, 0x7F–0xA0, 0xAD) gets shifted up
        by 0x100 into the Unicode range so the result is always
        printable + non-whitespace.
        """
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        cls._byte_encoder = {b: chr(c) for b, c in zip(bs, cs, strict=True)}
        cls._byte_decoder = {chr(c): b for b, c in zip(bs, cs, strict=True)}

    @classmethod
    def encode_bytes(cls, raw: bytes) -> str:
        """Re-map every byte in ``raw`` to its GPT-2 printable form."""
        if not cls._byte_encoder:
            cls._build_byte_tables_()
        return "".join(cls._byte_encoder[b] for b in raw)

    @classmethod
    def decode_bytes(cls, encoded: str) -> bytes:
        """Inverse of :meth:`encode_bytes`."""
        if not cls._byte_decoder:
            cls._build_byte_tables_()
        return bytes(cls._byte_decoder[c] for c in encoded)

    @override
    def pre_tokenize(self, text: str) -> list[Chunk]:
        """Chunk ``text`` along word/digit/punctuation boundaries
        and byte-encode each chunk."""
        # GPT-2 splits on the regex:
        #   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+
        # We approximate without ``regex`` (which isn't a stdlib
        # module): walk the string char-by-char + emit chunks at
        # word / digit / punctuation boundaries.  Each chunk
        # absorbs a leading space if one preceded it (this is what
        # the ``Ġ`` prefix represents post-byte-encode).
        if self._add_prefix_space and (not text or not text[0].isspace()):
            text = " " + text
        out: list[Chunk] = []
        i = 0
        n = len(text)
        while i < n:
            start = i
            # Optional leading space (matches GPT-2's `` ?`` capture).
            if text[i] == " ":
                i += 1
                if i >= n:
                    out.append((text[start:i], (start, i)))
                    break
            ch = text[i]
            if ch.isalpha():
                while i < n and text[i].isalpha():
                    i += 1
            elif ch.isdigit():
                while i < n and text[i].isdigit():
                    i += 1
            elif ch.isspace():
                # Consecutive whitespace (after the optional leading
                # space) — emit each as its own chunk.
                while i < n and text[i].isspace():
                    i += 1
            else:
                # Punctuation / symbols — consume a run.
                while i < n and not text[i].isalnum() and not text[i].isspace():
                    i += 1
            if i > start:
                chunk_text = text[start:i]
                # Byte-encode the chunk for the algorithm's consumption.
                encoded = ByteLevel.encode_bytes(chunk_text.encode("utf-8"))
                out.append((encoded, (start, i)))
        return out


def _is_punctuation(ch: str) -> bool:
    """Mirror BERT's ``_is_punctuation``: ASCII punctuation + every
    Unicode ``P*`` category."""
    cp = ord(ch)
    if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    return unicodedata.category(ch).startswith("P")


# ── Hugging Face pipeline pre-tokenizers ────────────────────────────
#
# The classes below exist to reproduce a ``tokenizer.json``
# ``pre_tokenizer`` block exactly.  Their splitting follows the Hugging
# Face contract: a pattern yields alternating (span, is-match) runs over
# the text, and a *behaviour* decides what to do with the matched runs.
# Empty chunks are always dropped, as Hugging Face drops them.

_WHITE_SPACE = frozenset(WHITE_SPACE)

#: Split behaviours, in the snake-case spelling the Python API uses.
_BEHAVIORS = ("removed", "isolated", "merged_with_previous", "merged_with_next", "contiguous")

#: ``tokenizer.json`` spells behaviours in CamelCase.
_BEHAVIOR_FROM_CONFIG = {
    "Removed": "removed",
    "Isolated": "isolated",
    "MergedWithPrevious": "merged_with_previous",
    "MergedWithNext": "merged_with_next",
    "Contiguous": "contiguous",
}

_Run = tuple[int, int, bool]


def _char_runs(text: str, predicate: Callable[[str], bool]) -> list[_Run]:
    """Split ``text`` into runs, one match per character ``predicate`` accepts.

    Consecutive matching characters are *separate* matches, which is what
    makes ``contiguous`` and ``merged_with_next`` differ.
    """
    if not text:
        return [(0, 0, False)]
    runs: list[_Run] = []
    last = 0
    for i, ch in enumerate(text):
        if predicate(ch):
            if last < i:
                runs.append((last, i, False))
            runs.append((i, i + 1, True))
            last = i + 1
    if last < len(text):
        runs.append((last, len(text), False))
    return runs


def _regex_runs(text: str, pattern: re.Pattern[str]) -> list[_Run]:
    """Split ``text`` into runs of non-matches and ``pattern`` matches."""
    if not text:
        return [(0, 0, False)]
    runs: list[_Run] = []
    prev = 0
    for m in pattern.finditer(text):
        if prev != m.start():
            runs.append((prev, m.start(), False))
        runs.append((m.start(), m.end(), True))
        prev = m.end()
    if prev != len(text):
        runs.append((prev, len(text), False))
    return runs


def _apply_behavior(runs: list[_Run], behavior: str) -> list[tuple[int, int]]:
    """Turn runs into kept spans according to ``behavior``.

    The merge rules are Hugging Face's, including the one that is easy to
    get wrong: under ``merged_with_next`` a *run* of delimiters keeps all
    but its last as standalone pieces, so ``"a  b"`` split on spaces gives
    ``["a", " ", " b"]`` rather than ``["a", "  b"]``.
    """
    spans: list[tuple[int, int]] = []
    if behavior == "removed":
        spans = [(s, e) for s, e, m in runs if not m]
    elif behavior == "isolated":
        spans = [(s, e) for s, e, _ in runs]
    elif behavior == "merged_with_previous":
        previous = False
        for s, e, m in runs:
            if m and not previous and spans:
                spans[-1] = (spans[-1][0], e)
            else:
                spans.append((s, e))
            previous = m
    elif behavior == "merged_with_next":
        previous = False
        for s, e, m in reversed(runs):
            if m and not previous and spans:
                spans[-1] = (s, spans[-1][1])
            else:
                spans.append((s, e))
            previous = m
        spans.reverse()
    elif behavior == "contiguous":
        previous = False
        for s, e, m in runs:
            if m == previous and spans:
                spans[-1] = (spans[-1][0], e)
            else:
                spans.append((s, e))
            previous = m
    else:
        raise ValueError(f"unknown split behavior {behavior!r}; expected one of {_BEHAVIORS}")
    return [(s, e) for s, e in spans if e > s]


def _check_behavior(behavior: str) -> str:
    """Validate a snake-case behaviour name."""
    if behavior not in _BEHAVIORS:
        raise ValueError(f"unknown split behavior {behavior!r}; expected one of {_BEHAVIORS}")
    return behavior


class Metaspace(PreTokenizer):
    r"""SentencePiece-style pre-tokenizer: spaces become ``▁``.

    Replaces every U+0020 space with the replacement character, optionally
    prepends one, and splits so each ``▁`` starts a new chunk.  Only the
    ASCII space is replaced — tabs and newlines are left for the model —
    which is what distinguishes this from :class:`SentencePiecePreTokenizer
    <lucid.utils.tokenizer._unigram.SentencePiecePreTokenizer>`, Lucid's own
    default for Unigram vocabularies it trains itself.

    Parameters
    ----------
    replacement : str, default "▁"
        The marker character.
    prepend_scheme : {"always", "first", "never"}, default "always"
        When to prepend the marker to a piece that does not start with one:
        every piece, only the piece at the very start of the input, or
        never.
    split : bool, default True
        Split so each marker starts a chunk.  ``False`` keeps the whole
        piece as one chunk.

    Notes
    -----
    Consecutive markers split as Hugging Face splits them: every marker in
    a run but the last is a chunk of its own, and the last one leads the
    following word — ``"a  b"`` becomes ``["▁a", "▁", "▁b"]``.

    Examples
    --------
    >>> from lucid.utils.tokenizer._pre_tokenizers import Metaspace
    >>> [c for c, _ in Metaspace().pre_tokenize("hello  world")]
    ['▁hello', '▁', '▁world']
    >>> [c for c, _ in Metaspace(prepend_scheme="never").pre_tokenize("hi you")]
    ['hi', '▁you']
    """

    def __init__(
        self, replacement: str = "▁", prepend_scheme: str = "always", split: bool = True
    ) -> None:
        r"""Record the marker, the prepend scheme and the split flag.

        Parameters
        ----------
        replacement : str, default "▁"
            The marker character.
        prepend_scheme : {"always", "first", "never"}, default "always"
            When to prepend the marker.
        split : bool, default True
            Split on markers.
        """
        if len(replacement) != 1:
            raise ValueError(f"Metaspace: replacement must be one character, got {replacement!r}")
        if prepend_scheme not in ("always", "first", "never"):
            raise ValueError(f"Metaspace: unknown prepend_scheme {prepend_scheme!r}")
        self._replacement = replacement
        self._prepend_scheme = prepend_scheme
        self._split = split

    @override
    def pre_tokenize(self, text: str) -> list[Chunk]:
        """Replace spaces, prepend the marker, and split on it.

        Parameters
        ----------
        text : str
            Post-normalised input string.

        Returns
        -------
        list of (str, (int, int))
            Marked chunks.  The offsets index ``text``; a prepended marker
            occupies no characters of it.
        """
        return self._pre_tokenize_at(text, 0)

    @override
    def _pre_tokenize_at(self, text: str, offset: int) -> list[Chunk]:
        """Pre-tokenize a piece starting at ``offset`` in the full input."""
        # Prepending to nothing would conjure a token out of an empty
        # piece; Hugging Face's prepend is a no-op there too.
        if not text:
            return []
        marker = self._replacement
        body = text.replace(" ", marker)
        prepend = self._prepend_scheme == "always" or (
            self._prepend_scheme == "first" and offset == 0
        )
        shift = 0
        if prepend and not body.startswith(marker):
            body = marker + body
            shift = 1

        def origin(i: int) -> int:
            return offset + max(0, i - shift)

        if not self._split:
            return [(body, (offset, offset + len(text)))]
        runs = _char_runs(body, lambda ch: ch == marker)
        return [
            (body[s:e], (origin(s), origin(e)))
            for s, e in _apply_behavior(runs, "merged_with_next")
        ]


class Split(PreTokenizer):
    r"""Split on a literal or a regular expression.

    Parameters
    ----------
    pattern : str or re.Pattern
        A literal delimiter, or a compiled expression.
    behavior : {"removed", "isolated", "merged_with_previous", \
"merged_with_next", "contiguous"}
        What happens to the matched text: dropped, kept as its own chunk,
        attached to the chunk before or after it, or kept with adjacent
        matches fused into one chunk.
    invert : bool, default False
        Treat the matches as the content and everything between them as
        the delimiters.

    Examples
    --------
    >>> from lucid.utils.tokenizer._pre_tokenizers import Split
    >>> [c for c, _ in Split(" ", "merged_with_next").pre_tokenize("a  b ")]
    ['a', ' ', ' b', ' ']
    >>> import re
    >>> [c for c, _ in Split(re.compile(r"\d+"), "isolated").pre_tokenize("ab12c")]
    ['ab', '12', 'c']
    """

    def __init__(self, pattern: str | re.Pattern[str], behavior: str, invert: bool = False) -> None:
        r"""Compile the pattern and record the behaviour.

        Parameters
        ----------
        pattern : str or re.Pattern
            Literal delimiter or compiled expression.
        behavior : str
            One of the five split behaviours.
        invert : bool, default False
            Swap matches and non-matches.
        """
        self._source = pattern
        self._pattern = re.compile(re.escape(pattern)) if isinstance(pattern, str) else pattern
        self._behavior = _check_behavior(behavior)
        self._invert = invert

    @override
    def pre_tokenize(self, text: str) -> list[Chunk]:
        """Split ``text`` per the pattern and behaviour.

        Parameters
        ----------
        text : str
            Post-normalised input string.

        Returns
        -------
        list of (str, (int, int))
            The kept chunks.
        """
        runs = _regex_runs(text, self._pattern)
        if self._invert:
            runs = [(s, e, not m) for s, e, m in runs]
        return [(text[s:e], (s, e)) for s, e in _apply_behavior(runs, self._behavior)]


class Digits(PreTokenizer):
    r"""Separate numbers from the text around them.

    Parameters
    ----------
    individual_digits : bool, default False
        Split every digit into its own chunk instead of keeping runs of
        digits together.

    Examples
    --------
    >>> from lucid.utils.tokenizer._pre_tokenizers import Digits
    >>> [c for c, _ in Digits().pre_tokenize("ab123c")]
    ['ab', '123', 'c']
    >>> [c for c, _ in Digits(individual_digits=True).pre_tokenize("a12")]
    ['a', '1', '2']
    """

    def __init__(self, individual_digits: bool = False) -> None:
        r"""Record whether digits are split one by one.

        Parameters
        ----------
        individual_digits : bool, default False
            One chunk per digit when ``True``.
        """
        self._individual = individual_digits

    @override
    def pre_tokenize(self, text: str) -> list[Chunk]:
        """Split ``text`` at number boundaries.

        Parameters
        ----------
        text : str
            Post-normalised input string.

        Returns
        -------
        list of (str, (int, int))
            Number and non-number chunks.
        """
        # Any Unicode number — the ``N*`` categories — as Hugging Face's
        # ``char::is_numeric`` counts them, not only ASCII 0-9.
        runs = _char_runs(text, lambda ch: unicodedata.category(ch)[0] == "N")
        behavior = "isolated" if self._individual else "contiguous"
        return [(text[s:e], (s, e)) for s, e in _apply_behavior(runs, behavior)]


class Punctuation(PreTokenizer):
    r"""Separate punctuation from the text around it.

    Parameters
    ----------
    behavior : str, default "isolated"
        One of the five split behaviours, applied to each punctuation
        character.

    Examples
    --------
    >>> from lucid.utils.tokenizer._pre_tokenizers import Punctuation
    >>> [c for c, _ in Punctuation().pre_tokenize("hi, you!")]
    ['hi', ',', ' you', '!']
    """

    def __init__(self, behavior: str = "isolated") -> None:
        r"""Record the behaviour.

        Parameters
        ----------
        behavior : str, default "isolated"
            How punctuation characters are kept.
        """
        self._behavior = _check_behavior(behavior)

    @override
    def pre_tokenize(self, text: str) -> list[Chunk]:
        """Split ``text`` at punctuation characters.

        Parameters
        ----------
        text : str
            Post-normalised input string.

        Returns
        -------
        list of (str, (int, int))
            The kept chunks.
        """
        runs = _char_runs(text, _is_punctuation)
        return [(text[s:e], (s, e)) for s, e in _apply_behavior(runs, self._behavior)]


def _is_word_char(ch: str) -> bool:
    """Oniguruma's Unicode ``\\w``: letters, marks, decimal digits, ``_``-like."""
    cat = unicodedata.category(ch)
    return cat[0] in ("L", "M") or cat in ("Nd", "Pc")


class Whitespace(PreTokenizer):
    r"""Split into word runs and punctuation runs, dropping whitespace.

    Reproduces Hugging Face's ``Whitespace`` pre-tokenizer, the expression
    ``\w+|[^\w\s]+`` with Oniguruma's Unicode classes — under which
    combining marks count as word characters, unlike Python's ``\w``.

    Examples
    --------
    >>> from lucid.utils.tokenizer._pre_tokenizers import Whitespace
    >>> [c for c, _ in Whitespace().pre_tokenize("hello, world!!")]
    ['hello', ',', 'world', '!!']
    """

    @override
    def pre_tokenize(self, text: str) -> list[Chunk]:
        """Emit maximal word runs and maximal symbol runs.

        Parameters
        ----------
        text : str
            Post-normalised input string.

        Returns
        -------
        list of (str, (int, int))
            Word and symbol chunks.
        """
        out: list[Chunk] = []
        i, n = 0, len(text)
        while i < n:
            ch = text[i]
            if ch in _WHITE_SPACE:
                i += 1
                continue
            start = i
            if _is_word_char(ch):
                while i < n and _is_word_char(text[i]):
                    i += 1
            else:
                while i < n and not _is_word_char(text[i]) and text[i] not in _WHITE_SPACE:
                    i += 1
            out.append((text[start:i], (start, i)))
        return out


class Sequence(PreTokenizer):
    r"""Apply several pre-tokenizers in order, each to every chunk so far.

    An empty sequence passes its input through as a single chunk, which is
    what a ``tokenizer.json`` with ``"pre_tokenizer": null`` means.

    Parameters
    ----------
    pre_tokenizers : list of PreTokenizer
        Applied left to right.

    Examples
    --------
    T5's pipeline: split on whitespace, then mark each word.

    >>> from lucid.utils.tokenizer._pre_tokenizers import (
    ...     Metaspace, Sequence, WhitespaceSplit,
    ... )
    >>> t5 = Sequence([WhitespaceSplit(), Metaspace()])
    >>> [c for c, _ in t5.pre_tokenize("the  quick fox")]
    ['▁the', '▁quick', '▁fox']
    """

    def __init__(self, pre_tokenizers: list[PreTokenizer]) -> None:
        r"""Snapshot the pre-tokenizers.

        Parameters
        ----------
        pre_tokenizers : list of PreTokenizer
            Applied left to right.
        """
        self._pre_tokenizers = list(pre_tokenizers)

    @override
    def pre_tokenize(self, text: str) -> list[Chunk]:
        """Thread ``text`` through every pre-tokenizer.

        Parameters
        ----------
        text : str
            Post-normalised input string.

        Returns
        -------
        list of (str, (int, int))
            The last stage's chunks.
        """
        return self._pre_tokenize_at(text, 0)

    @override
    def _pre_tokenize_at(self, text: str, offset: int) -> list[Chunk]:
        """Thread a piece at ``offset`` through every pre-tokenizer."""
        chunks: list[Chunk] = [(text, (offset, offset + len(text)))] if text else []
        for stage in self._pre_tokenizers:
            chunks = [
                sub
                for chunk, (start, _) in chunks
                for sub in stage._pre_tokenize_at(chunk, start)
                if sub[0]
            ]
        return chunks


def prepend_scheme_from_config(config: dict[str, object], where: str) -> str:
    """Resolve a Metaspace block's prepend scheme, legacy keys included.

    Parameters
    ----------
    config : dict of str to object
        A ``Metaspace`` pre-tokenizer or decoder block.
    where : str
        What was being read, for the error message.

    Returns
    -------
    str
        ``"always"``, ``"first"`` or ``"never"``.

    Notes
    -----
    Older files carry ``add_prefix_space`` instead of ``prepend_scheme``
    (T5's does).  ``true`` there means ``"always"`` and ``false`` means
    ``"never"``; a file carrying both must agree with itself, as Hugging
    Face requires.
    """
    scheme = get_str(config, "prepend_scheme", where, "always")
    if scheme not in ("always", "first", "never"):
        raise ValueError(f"{where}: unknown prepend_scheme {scheme!r}")
    add_prefix_space = config.get("add_prefix_space")
    if add_prefix_space is False:
        return "never"
    if add_prefix_space is True and scheme == "never":
        raise ValueError(f"{where}: add_prefix_space=true contradicts prepend_scheme='never'")
    return scheme


def pre_tokenizer_from_config(config: object) -> PreTokenizer:
    """Build a pre-tokenizer from a ``tokenizer.json`` ``pre_tokenizer`` block.

    Parameters
    ----------
    config : dict or None
        The block.  ``None`` — the JSON ``null`` — builds a pass-through
        that returns the whole input as one chunk.

    Returns
    -------
    PreTokenizer
        The pre-tokenizer the block describes.

    Raises
    ------
    ValueError
        For an unsupported ``type`` or a malformed block.

    Examples
    --------
    >>> from lucid.utils.tokenizer._pre_tokenizers import (
    ...     pre_tokenizer_from_config,
    ... )
    >>> t5 = pre_tokenizer_from_config({
    ...     "type": "Sequence",
    ...     "pretokenizers": [
    ...         {"type": "WhitespaceSplit"},
    ...         {"type": "Metaspace", "replacement": "▁", "add_prefix_space": True},
    ...     ],
    ... })
    >>> [c for c, _ in t5.pre_tokenize("hi  there")]
    ['▁hi', '▁there']
    """
    if config is None:
        return Sequence([])
    where = "pre_tokenizer"
    cfg = as_object(config, where)
    kind = type_of(cfg, where)
    if kind == "Sequence":
        return Sequence(
            [
                pre_tokenizer_from_config(p)
                for p in as_list(cfg.get("pretokenizers"), f"{where}.pretokenizers")
            ]
        )
    if kind == "WhitespaceSplit":
        return WhitespaceSplit()
    if kind == "Whitespace":
        return Whitespace()
    if kind == "BertPreTokenizer":
        return WhitespacePunctuationSplit()
    if kind == "Metaspace":
        return Metaspace(
            replacement=get_str(cfg, "replacement", where, "▁"),
            prepend_scheme=prepend_scheme_from_config(cfg, where),
            split=get_bool(cfg, "split", where, True),
        )
    if kind == "Split":
        raw_behavior = get_str(cfg, "behavior", where, "")
        if raw_behavior not in _BEHAVIOR_FROM_CONFIG:
            raise ValueError(f"{where}: unknown Split behavior {raw_behavior!r}")
        return Split(
            pattern_from_config(cfg.get("pattern"), f"{where}.pattern"),
            _BEHAVIOR_FROM_CONFIG[raw_behavior],
            invert=get_bool(cfg, "invert", where, False),
        )
    if kind == "Digits":
        return Digits(individual_digits=get_bool(cfg, "individual_digits", where, False))
    if kind == "Punctuation":
        raw_behavior = get_str(cfg, "behavior", where, "Isolated")
        if raw_behavior not in _BEHAVIOR_FROM_CONFIG:
            raise ValueError(f"{where}: unknown Punctuation behavior {raw_behavior!r}")
        return Punctuation(_BEHAVIOR_FROM_CONFIG[raw_behavior])
    if kind == "ByteLevel":
        if not get_bool(cfg, "use_regex", where, True):
            raise ValueError(
                f"{where}: ByteLevel with use_regex=false is not supported; "
                f"Lucid's ByteLevel always applies the GPT-2 split"
            )
        return ByteLevel(add_prefix_space=get_bool(cfg, "add_prefix_space", where, True))
    raise ValueError(
        f"{where}: unsupported type {kind!r}; supported are Sequence, "
        f"WhitespaceSplit, Whitespace, BertPreTokenizer, Metaspace, Split, "
        f"Digits, Punctuation and ByteLevel"
    )
