"""Text-normalisation primitives composable into a normalizer chain.

A :class:`Normalizer` is any callable ``str -> str`` applied to the
raw input text **before** pre-tokenization.  The standard chain for
HF-style tokenizers is

    Sequence([NFD(), Lowercase(), StripAccents()])  # BERT-uncased
    Sequence([NFC()])                                # GPT-2
    Sequence([NFKC(), Replace(...)])                 # LLaMA / Mistral

so the API mirrors HF's: each primitive is its own class so config
serialisation (algo name + args) round-trips cleanly through
``tokenizer.json``.

Pure Python — these run once per encode call on the surface string
and aren't the hot path (encoding dominates).  No C++ counterpart.
"""

import base64
import re
import struct
import unicodedata
from abc import ABC, abstractmethod
from typing import override

from lucid.utils.tokenizer._config import (
    WHITE_SPACE as _WHITE_SPACE,
    as_list,
    as_object,
    get_bool,
    get_str,
    pattern_from_config,
    type_of,
)


class Normalizer(ABC):
    """Abstract base — every concrete normalizer implements
    :meth:`normalize`.

    Composable via :class:`Sequence` for the canonical chain
    semantics (apply primitives left-to-right).
    """

    @abstractmethod
    def normalize(self, text: str) -> str:
        """Return the normalised form of ``text``."""

    def __call__(self, text: str) -> str:
        r"""Convenience callable form — delegates to :meth:`normalize`.

        Parameters
        ----------
        text : str
            Raw input string.

        Returns
        -------
        str
            Normalised form, identical to ``self.normalize(text)``.
        """
        return self.normalize(text)


class Sequence(Normalizer):
    """Apply a series of normalizers in order.

    The standard HF normalizer chain for any tokenizer can be
    expressed as ``Sequence([N1, N2, ...])``.
    """

    def __init__(self, normalizers: list[Normalizer]) -> None:
        r"""Snapshot ``normalizers`` into an internal list.

        Parameters
        ----------
        normalizers : list of Normalizer
            Primitives applied left-to-right by :meth:`normalize`.
            The list is shallow-copied so later mutations of the
            caller's list don't affect this sequence.
        """
        self._normalizers = list(normalizers)

    @override
    def normalize(self, text: str) -> str:
        r"""Apply every wrapped normalizer in order.

        Parameters
        ----------
        text : str
            Raw input string.

        Returns
        -------
        str
            Output of the final wrapped normalizer.  Equivalent to
            ``reduce(lambda t, n: n(t), normalizers, text)``.
        """
        for n in self._normalizers:
            text = n.normalize(text)
        return text

    @override
    def __repr__(self) -> str:
        r"""Return a developer-readable representation listing the
        wrapped primitives in apply order.

        Returns
        -------
        str
            String of the form ``Sequence([NFD(), Lowercase(), ...])``.
        """
        return f"Sequence({self._normalizers!r})"


class NFC(Normalizer):
    """Unicode NFC (Canonical Composition) — used by GPT-2 and friends.

    Combines decomposed characters back into their composed forms
    (e.g. ``a + ◌́  → á``).  No-op on most ASCII text.
    """

    @override
    def normalize(self, text: str) -> str:
        r"""Apply Unicode NFC normalisation.

        Parameters
        ----------
        text : str
            Raw input string.

        Returns
        -------
        str
            ``text`` with all canonical-decomposed sequences
            recomposed into their canonical-composed equivalents.
        """
        return unicodedata.normalize("NFC", text)


class NFD(Normalizer):
    """Unicode NFD (Canonical Decomposition).

    Splits composed characters into base + combining marks (the
    inverse of NFC).  Used by BERT-uncased as a precursor to
    :class:`StripAccents`.
    """

    @override
    def normalize(self, text: str) -> str:
        r"""Apply Unicode NFD normalisation.

        Parameters
        ----------
        text : str
            Raw input string.

        Returns
        -------
        str
            ``text`` with composed characters split into base + combining
            marks.  Inverse of :class:`NFC`; typically chained before
            :class:`StripAccents`.
        """
        return unicodedata.normalize("NFD", text)


class NFKC(Normalizer):
    """Unicode NFKC (Compatibility Composition) — used by LLaMA /
    Mistral.

    Like NFC, but additionally folds compatibility characters
    (e.g. fullwidth digits → ASCII digits).
    """

    @override
    def normalize(self, text: str) -> str:
        r"""Apply Unicode NFKC normalisation.

        Parameters
        ----------
        text : str
            Raw input string.

        Returns
        -------
        str
            ``text`` with canonical-composed output plus compatibility
            characters (ligatures, fullwidth digits, ...) folded to
            their canonical equivalents.
        """
        return unicodedata.normalize("NFKC", text)


class NFKD(Normalizer):
    r"""Unicode NFKD (Compatibility Decomposition) — splits composed
    characters AND folds compatibility forms.

    Like :class:`NFD`, but additionally decomposes compatibility
    characters (ligatures, super-/sub-scripts, fullwidth digits, …)
    into their canonical components.  Commonly chained before
    :class:`StripAccents` for the most aggressive normalisation
    pipelines.

    Notes
    -----
    NFKD is the most lossy normalisation form — ``½`` becomes
    ``1⁄2``, ``ﬁ`` becomes ``f + i``, fullwidth digits collapse to
    ASCII, etc.  Use :class:`NFD` instead if you only want canonical
    decomposition without the compatibility fold.
    """

    @override
    def normalize(self, text: str) -> str:
        r"""Apply Unicode NFKD normalisation.

        Parameters
        ----------
        text : str
            Raw input string.

        Returns
        -------
        str
            ``text`` decomposed canonically AND folded for compatibility
            — the most lossy of the four standard Unicode forms.
        """
        return unicodedata.normalize("NFKD", text)


class Lowercase(Normalizer):
    r"""Lowercase every character via :meth:`str.lower`.

    Used by ``bert-base-uncased`` and any other uncased checkpoint;
    typically chained after :class:`NFD` + :class:`StripAccents` so
    accented uppercase letters fold to their unaccented lowercase
    forms in one pass.

    Notes
    -----
    Locale-independent — uses Python's default Unicode case folding,
    which matches HF's reference implementation but differs from
    locale-specific folds (e.g. Turkish dotless-I).
    """

    @override
    def normalize(self, text: str) -> str:
        r"""Lowercase every character via :meth:`str.lower`.

        Parameters
        ----------
        text : str
            Raw input string.

        Returns
        -------
        str
            Locale-independent Unicode lowercase of ``text``.
        """
        return text.lower()


class StripAccents(Normalizer):
    """Drop every combining mark.

    Must be preceded by :class:`NFD` (or :class:`NFKD`) so accents
    have been split off the base characters; otherwise composed
    characters pass through unchanged.
    """

    @override
    def normalize(self, text: str) -> str:
        r"""Drop every combining mark.

        Parameters
        ----------
        text : str
            Input string, typically already :class:`NFD`-decomposed.

        Returns
        -------
        str
            ``text`` with all combining-mark characters removed.
            Pre-composed characters pass through unchanged — chain
            after :class:`NFD` / :class:`NFKD` for the standard
            accent-stripping pipeline.

        Notes
        -----
        A mark is anything in General Category ``M`` — nonspacing,
        spacing and enclosing — which is what Hugging Face's
        ``StripAccents`` removes.  Testing the canonical combining class
        instead, as this once did, keeps every mark whose class is 0: Thai
        vowel signs, most Indic vowel signs, enclosing circles.
        """
        return "".join(c for c in text if unicodedata.category(c)[0] != "M")


class Strip(Normalizer):
    """Strip leading / trailing whitespace.

    Parameters
    ----------
    left : bool, default True
        Strip leading whitespace.
    right : bool, default True
        Strip trailing whitespace.
    """

    def __init__(self, left: bool = True, right: bool = True) -> None:
        r"""Record which sides to strip.

        Parameters
        ----------
        left : bool, default True
            Strip leading whitespace.
        right : bool, default True
            Strip trailing whitespace.
        """
        self._left = left
        self._right = right

    @override
    def normalize(self, text: str) -> str:
        r"""Strip whitespace per the configured sides.

        Parameters
        ----------
        text : str
            Raw input string.

        Returns
        -------
        str
            ``text`` with leading/trailing whitespace removed per
            ``left`` / ``right`` flags.  If both are ``False`` the
            string is returned unchanged.
        """
        # Unicode White_Space, as Hugging Face strips — ``str.strip()`` with
        # no argument would also eat U+001C..U+001F, which are not spaces.
        if self._left:
            text = text.lstrip(_WHITE_SPACE)
        if self._right:
            text = text.rstrip(_WHITE_SPACE)
        return text


class Replace(Normalizer):
    r"""Substitute a literal substring or a regular-expression match.

    Parameters
    ----------
    pattern : str or re.Pattern
        A literal substring, or a compiled expression — T5's
        ``tokenizer.json`` collapses space runs with ``" {2,}"``.
    replacement : str
        Replacement string, inserted verbatim (no group references).

    Examples
    --------
    >>> import re
    >>> from lucid.utils.tokenizer._normalizers import Replace
    >>> Replace(re.compile(" {2,}"), " ")("a   b")
    'a b'
    >>> Replace("’", "'")("it’s")
    "it's"
    """

    def __init__(self, pattern: str | re.Pattern[str], replacement: str) -> None:
        r"""Record the substitution pair.

        Parameters
        ----------
        pattern : str or re.Pattern
            Literal substring or compiled expression.
        replacement : str
            Replacement string.
        """
        self._pattern = pattern
        self._replacement = replacement

    @override
    def normalize(self, text: str) -> str:
        r"""Replace every occurrence of ``pattern`` with ``replacement``.

        Parameters
        ----------
        text : str
            Raw input string.

        Returns
        -------
        str
            ``text`` with every match replaced.
        """
        if isinstance(self._pattern, str):
            return text.replace(self._pattern, self._replacement)
        # A function replacement keeps ``replacement`` literal; a string
        # one would read any backslash in it as a group reference.
        replacement = self._replacement
        return self._pattern.sub(lambda _m: replacement, text)


class Prepend(Normalizer):
    r"""Prepend a fixed string to non-empty input.

    Parameters
    ----------
    prepend : str
        The prefix — ``"▁"`` in Llama-style SentencePiece pipelines.

    Examples
    --------
    >>> from lucid.utils.tokenizer._normalizers import Prepend
    >>> Prepend("▁")("hi")
    '▁hi'
    >>> Prepend("▁")("")
    ''
    """

    def __init__(self, prepend: str) -> None:
        r"""Record the prefix.

        Parameters
        ----------
        prepend : str
            The prefix.
        """
        self._prepend = prepend

    @override
    def normalize(self, text: str) -> str:
        r"""Prepend the prefix unless ``text`` is empty.

        Parameters
        ----------
        text : str
            Raw input string.

        Returns
        -------
        str
            ``prepend + text``, or ``""`` for empty input — a prefix on
            nothing would turn an empty document into a token.
        """
        return self._prepend + text if text else text


class Nmt(Normalizer):
    r"""SentencePiece's NMT clean-up: drop control characters, map
    separators to spaces.

    Examples
    --------
    >>> from lucid.utils.tokenizer._normalizers import Nmt
    >>> Nmt()("a\x01b\tc")
    'ab c'
    """

    @override
    def normalize(self, text: str) -> str:
        r"""Apply the NMT character rules.

        Parameters
        ----------
        text : str
            Raw input string.

        Returns
        -------
        str
            ``text`` with C0 controls (except the ones below) and DEL / two
            C1 controls removed, and tab, newline, form feed, carriage
            return, ogham space, the zero-width and direction marks, the
            line / paragraph separators, ``▁``, the BOM and U+FFFD mapped
            to a space.
        """
        out: list[str] = []
        for ch in text:
            cp = ord(ch)
            if 0x01 <= cp <= 0x08 or cp == 0x0B or 0x0E <= cp <= 0x1F or cp in (0x7F, 0x8F, 0x9F):
                continue
            if (
                cp in (0x09, 0x0A, 0x0C, 0x0D, 0x1680, 0x2028, 0x2029, 0x2581)
                or 0x200B <= cp <= 0x200F
                or cp in (0xFEFF, 0xFFFD)
            ):
                out.append(" ")
            else:
                out.append(ch)
        return "".join(out)


class BERTNormalizer(Normalizer):
    """Composite normalizer matching BERT's standard pipeline.

    Parameters
    ----------
    lowercase : bool, default True
        Lowercase the text last (matches BERT-uncased).
    strip_accents : bool, default True
        NFD-decompose and drop nonspacing marks (``Mn``).  Cased
        checkpoints set this ``False`` and then no decomposition happens
        at all.
    clean_text : bool, default True
        Replace control characters with spaces + collapse
        whitespace runs into single spaces.  Matches BERT's
        ``BasicTokenizer._clean_text``.
    handle_chinese_chars : bool, default True
        Wrap every CJK ideograph with spaces so the
        whitespace pre-tokenizer treats them as standalone tokens
        (matches BERT-Chinese / Multilingual-BERT behaviour).
    """

    def __init__(
        self,
        *,
        lowercase: bool = True,
        strip_accents: bool = True,
        clean_text: bool = True,
        handle_chinese_chars: bool = True,
    ) -> None:
        r"""Record the BERT-style normalisation flags.

        Parameters
        ----------
        lowercase : bool, default True
            Lowercase the output (BERT-uncased semantics).
        strip_accents : bool, default True
            Drop combining marks after NFD decomposition.
        clean_text : bool, default True
            Replace control characters with spaces + collapse runs.
        handle_chinese_chars : bool, default True
            Wrap every CJK ideograph with spaces so each is a
            standalone whitespace-split token downstream.
        """
        self._lowercase = lowercase
        self._strip_accents = strip_accents
        self._clean_text = clean_text
        self._handle_chinese_chars = handle_chinese_chars

    @override
    def normalize(self, text: str) -> str:
        r"""Run the configured BERT normalisation pipeline.

        Parameters
        ----------
        text : str
            Raw input string.

        Returns
        -------
        str
            ``text`` after (optionally) control-char cleaning, CJK
            spacing, accent stripping (NFD plus ``Mn`` removal), and
            lowercasing — applied in that fixed order.

        Notes
        -----
        Order matches Google's reference BERT ``BasicTokenizer``
        pipeline; toggling individual flags reproduces the four
        canonical BERT variants (uncased, cased, multilingual, ...).
        """
        if self._clean_text:
            text = self._do_clean_text(text)
        if self._handle_chinese_chars:
            text = self._do_handle_chinese_chars(text)
        if self._strip_accents:
            # Decomposing belongs to accent stripping and nowhere else.  A
            # cased checkpoint's vocabulary holds composed forms (``é``), so
            # decomposing without stripping split every accented word into
            # a base letter plus an orphaned mark.  Only ``Mn`` goes, as in
            # Google's ``_run_strip_accents`` and Hugging Face's
            # ``BertNormalizer`` — spacing marks stay.
            text = unicodedata.normalize("NFD", text)
            text = "".join(c for c in text if unicodedata.category(c) != "Mn")
        if self._lowercase:
            text = text.lower()
        return text

    @staticmethod
    def _do_clean_text(text: str) -> str:
        """Replace control characters with spaces; collapse other
        whitespace into single spaces."""
        out: list[str] = []
        for ch in text:
            cp = ord(ch)
            if cp == 0 or cp == 0xFFFD or _is_control(ch):
                continue
            if _is_whitespace(ch):
                out.append(" ")
            else:
                out.append(ch)
        return "".join(out)

    @staticmethod
    def _do_handle_chinese_chars(text: str) -> str:
        """Surround every CJK ideograph with spaces."""
        out: list[str] = []
        for ch in text:
            cp = ord(ch)
            if _is_cjk(cp):
                out.append(" ")
                out.append(ch)
                out.append(" ")
            else:
                out.append(ch)
        return "".join(out)


def _is_whitespace(ch: str) -> bool:
    """Mirror BERT's ``_is_whitespace``: spaces + tabs + line breaks."""
    if ch in (" ", "\t", "\n", "\r"):
        return True
    return unicodedata.category(ch) == "Zs"


def _is_control(ch: str) -> bool:
    """Mirror BERT's ``_is_control``: control chars excluding standard
    whitespace."""
    if ch in ("\t", "\n", "\r"):
        return False
    return unicodedata.category(ch).startswith("C")


def _is_cjk(cp: int) -> bool:
    """CJK Unicode ranges per BERT's ``_is_chinese_char``."""
    return (
        (0x4E00 <= cp <= 0x9FFF)
        or (0x3400 <= cp <= 0x4DBF)
        or (0x20000 <= cp <= 0x2A6DF)
        or (0x2A700 <= cp <= 0x2B73F)
        or (0x2B740 <= cp <= 0x2B81F)
        or (0x2B820 <= cp <= 0x2CEAF)
        or (0xF900 <= cp <= 0xFAFF)
        or (0x2F800 <= cp <= 0x2FA1F)
    )


# ── SentencePiece precompiled character map ─────────────────────────

#: Grapheme_Cluster_Break=Prepend (Unicode 16).  These attach to the
#: character *after* them.
_GCB_PREPEND = frozenset(
    [*range(0x0600, 0x0606), 0x06DD, 0x070F, 0x0890, 0x0891, 0x08E2, 0x0D4E]
    + [0x110BD, 0x110CD, 0x111C2, 0x111C3, 0x1193F, 0x11941, 0x11A3A]
    + [*range(0x11A84, 0x11A8A), 0x11D46, 0x11F02]
)


def _gcb(ch: str) -> str:
    """Grapheme_Cluster_Break class of ``ch``, as far as :class:`Precompiled`
    needs it.

    The property is not in :mod:`unicodedata`, so it is rebuilt from
    general categories plus the handful of code points the categories get
    wrong.  Every mark (``Mn`` / ``Me`` / ``Mc``) is folded into ``Extend``
    because a spacing mark and an extending one join the preceding
    character alike; the distinction only matters for rules this helper
    does not need (see :func:`_graphemes`).
    """
    cp = ord(ch)
    if ch == "\r":
        return "CR"
    if ch == "\n":
        return "LF"
    if cp in _GCB_PREPEND:
        return "Prepend"
    if (
        cp in (0x200C, 0x200D, 0x0E33, 0x0EB3)
        or 0xFF9E <= cp <= 0xFF9F
        or 0x1F3FB <= cp <= 0x1F3FF
        or 0xE0020 <= cp <= 0xE007F
    ):
        # ZWNJ / ZWJ, the Thai and Lao AM vowels (spacing marks by
        # exception), halfwidth kana voicing marks, emoji skin-tone
        # modifiers and tag characters — all joining, none category M.
        return "Extend"
    cat = unicodedata.category(ch)
    if cat[0] == "M":
        return "Extend"
    if cat in ("Cc", "Cf", "Zl", "Zp"):
        return "Control"
    if 0x1100 <= cp <= 0x115F or 0xA960 <= cp <= 0xA97C:
        return "L"
    if 0x1160 <= cp <= 0x11A7 or 0xD7B0 <= cp <= 0xD7C6:
        return "V"
    if 0x11A8 <= cp <= 0x11FF or 0xD7CB <= cp <= 0xD7FB:
        return "T"
    if 0xAC00 <= cp <= 0xD7A3:
        return "LV" if (cp - 0xAC00) % 28 == 0 else "LVT"
    return "Other"


def _graphemes(text: str) -> list[str]:
    """Split ``text`` into extended grapheme clusters (UAX #29), closely
    enough for :class:`Precompiled`.

    Implemented: CR LF, control breaks, Hangul syllable sequences, joining
    marks / ZWJ, and Prepend.  Omitted: regional-indicator pairing, emoji
    ZWJ sequences and Indic conjuncts.  Each omitted rule only ever joins
    characters into a cluster of six UTF-8 bytes or more, and
    :class:`Precompiled` treats such a cluster character by character — so
    whether it is one cluster or several cannot change the output.
    """
    out: list[str] = []
    start = 0
    prev = ""
    for i, ch in enumerate(text):
        cls = _gcb(ch)
        if i > 0:
            if prev == "CR" and cls == "LF":
                join = True
            elif prev in ("Control", "CR", "LF") or cls in ("Control", "CR", "LF"):
                join = False
            elif prev == "L" and cls in ("L", "V", "LV", "LVT"):
                join = True
            elif prev in ("LV", "V") and cls in ("V", "T"):
                join = True
            elif prev in ("LVT", "T") and cls == "T":
                join = True
            else:
                join = cls == "Extend" or prev == "Prepend"
            if not join:
                out.append(text[start:i])
                start = i
        prev = cls
    if text:
        out.append(text[start:])
    return out


class Precompiled(Normalizer):
    r"""SentencePiece's compiled normalization rules (``precompiled_charsmap``).

    A SentencePiece model ships its normalization — NFKC plus the NMT
    clean-up rules for ``nmt_nfkc`` models such as T5 — compiled into a
    blob: a Darts double-array trie over UTF-8 byte strings whose leaves
    point into a pool of NUL-terminated replacement strings.  This class
    reads that blob and applies it the way Hugging Face ``tokenizers``
    does, so a converted checkpoint normalises to the same text here.

    Parameters
    ----------
    precompiled_charsmap : bytes
        The raw blob: a little-endian ``uint32`` trie size in bytes, the
        trie (``uint32`` units), then the replacement pool.  An empty blob
        is the identity.

    Raises
    ------
    ValueError
        If the blob is truncated or its trie size overruns it.

    Notes
    -----
    **Blob layout.**  Each trie unit is a ``uint32``: bit 8 flags a leaf
    below this node, bits 0-7 (plus bit 31, set on value units so they can
    never match a label) are the label, bits 10-30 the offset to the
    children — shifted left by 8 more when bit 9 is set — and a value unit
    holds its value in bits 0-30.  A lookup walks the input's bytes from
    unit 0, XOR-ing the offset and the byte to find each child; every leaf
    passed on the way is a match whose value is the replacement's position
    in the pool.

    **Which match, over what.**  Hugging Face applies the map one grapheme
    cluster at a time.  A cluster shorter than six bytes is looked up
    whole and, if any prefix of it matches, is replaced by the
    *shortest* match's replacement; otherwise — or for longer clusters —
    each character is looked up on its own.  SentencePiece itself scans
    bytes for the *longest* match instead, and the two disagree on a
    cluster whose first character has a mapping of its own: ``"Ａ"`` plus a
    combining acute is ``"A"`` here and ``"Á"`` in SentencePiece.  Hugging
    Face's reading is the one reproduced, because it is the one every
    ``tokenizer.json`` consumer — and the models evaluated through them —
    actually sees.

    Examples
    --------
    An empty map changes nothing:

    >>> from lucid.utils.tokenizer._normalizers import Precompiled
    >>> Precompiled(b"")("Ｆｕｌｌ")
    'Ｆｕｌｌ'

    Loaded from a T5 ``tokenizer.json`` it folds full-width forms and
    ideographic spaces as NFKC would::

        norm = normalizer_from_config(tokenizer_json["normalizer"])
        norm("ｈｅｌｌｏ　ｗｏｒｌｄ")   # 'hello world'
    """

    # Bounded memo of per-cluster results — text repeats the same few
    # hundred characters, and a trie walk per character in Python is the
    # whole cost of this normalizer.
    _CACHE_LIMIT = 1 << 16

    def __init__(self, precompiled_charsmap: bytes) -> None:
        r"""Parse the blob into trie units and the replacement pool.

        Parameters
        ----------
        precompiled_charsmap : bytes
            The raw ``precompiled_charsmap`` blob.
        """
        self._units: tuple[int, ...] = ()
        self._pool = b""
        self._cache: dict[str, str] = {}
        blob = bytes(precompiled_charsmap)
        if not blob:
            return
        if len(blob) <= 4:
            raise ValueError("Precompiled: charsmap blob is truncated")
        (trie_size,) = struct.unpack_from("<I", blob, 0)
        if trie_size > len(blob) - 4 or trie_size % 4 != 0:
            raise ValueError(
                f"Precompiled: trie size {trie_size} does not fit a " f"{len(blob)}-byte blob"
            )
        self._units = struct.unpack_from(f"<{trie_size // 4}I", blob, 4)
        self._pool = blob[4 + trie_size :]

    @classmethod
    def from_base64(cls, encoded: str) -> Precompiled:
        r"""Build from the base64 string a ``tokenizer.json`` stores.

        Parameters
        ----------
        encoded : str
            Base64 of the blob.

        Returns
        -------
        Precompiled
            The normalizer.
        """
        return cls(base64.b64decode(encoded))

    def _shortest_match(self, piece: str) -> str | None:
        """Replacement for the shortest trie key prefixing ``piece``."""
        units = self._units
        n_units = len(units)
        pos = 0
        unit = units[0]
        pos ^= (unit >> 10) << ((unit & (1 << 9)) >> 6)
        for byte in piece.encode("utf-8"):
            # A NUL byte ends the key in Hugging Face's walk as well.
            if byte == 0:
                return None
            pos ^= byte
            if pos >= n_units:
                return None
            unit = units[pos]
            if (unit & ((1 << 31) | 0xFF)) != byte:
                return None
            pos ^= (unit >> 10) << ((unit & (1 << 9)) >> 6)
            if (unit >> 8) & 1:
                if pos >= n_units:
                    return None
                start = units[pos] & ((1 << 31) - 1)
                end = self._pool.find(b"\0", start)
                if end < 0:
                    end = len(self._pool)
                return self._pool[start:end].decode("utf-8", errors="replace")
        return None

    def _normalize_cluster(self, cluster: str) -> str:
        """Apply the map to one grapheme cluster."""
        if len(cluster.encode("utf-8")) < 6:
            whole = self._shortest_match(cluster)
            if whole is not None:
                return whole
        out: list[str] = []
        for ch in cluster:
            mapped = self._shortest_match(ch)
            out.append(ch if mapped is None else mapped)
        return "".join(out)

    @override
    def normalize(self, text: str) -> str:
        r"""Apply the compiled map, one grapheme cluster at a time.

        Parameters
        ----------
        text : str
            Raw input string.

        Returns
        -------
        str
            The normalised text; ``text`` itself when the map is empty.
        """
        if not self._units:
            return text
        cache = self._cache
        out: list[str] = []
        for cluster in _graphemes(text):
            done = cache.get(cluster)
            if done is None:
                done = self._normalize_cluster(cluster)
                if len(cache) >= self._CACHE_LIMIT:
                    cache.clear()
                cache[cluster] = done
            out.append(done)
        return "".join(out)


def normalizer_from_config(config: object) -> Normalizer | None:
    """Build a normalizer from a ``tokenizer.json`` ``normalizer`` block.

    Parameters
    ----------
    config : dict or None
        The block, or ``None`` for the JSON ``null``.

    Returns
    -------
    Normalizer or None
        The normalizer, or ``None`` when the block is ``null`` — the text
        then reaches the pre-tokenizer untouched.

    Raises
    ------
    ValueError
        For an unsupported ``type`` or a malformed block.

    Examples
    --------
    UMT5's chain collapses runs of spaces:

    >>> from lucid.utils.tokenizer._normalizers import normalizer_from_config
    >>> umt5 = normalizer_from_config({
    ...     "type": "Replace",
    ...     "pattern": {"Regex": " {2,}"},
    ...     "content": " ",
    ... })
    >>> umt5("a    b")
    'a b'
    >>> normalizer_from_config(None) is None
    True
    """
    if config is None:
        return None
    where = "normalizer"
    cfg = as_object(config, where)
    kind = type_of(cfg, where)
    simple: dict[str, type[Normalizer]] = {
        "NFC": NFC,
        "NFD": NFD,
        "NFKC": NFKC,
        "NFKD": NFKD,
        "Lowercase": Lowercase,
        "StripAccents": StripAccents,
        "Nmt": Nmt,
    }
    if kind in simple:
        return simple[kind]()
    if kind == "Sequence":
        parts: list[Normalizer] = []
        for raw in as_list(cfg.get("normalizers"), f"{where}.normalizers"):
            inner = normalizer_from_config(raw)
            if inner is not None:
                parts.append(inner)
        return Sequence(parts)
    if kind == "Strip":
        return Strip(
            left=get_bool(cfg, "strip_left", where, True),
            right=get_bool(cfg, "strip_right", where, True),
        )
    if kind == "BertNormalizer":
        lowercase = get_bool(cfg, "lowercase", where, True)
        return BERTNormalizer(
            lowercase=lowercase,
            # ``null`` means "follow lowercase" — uncased checkpoints strip
            # accents, cased ones keep them.
            strip_accents=get_bool(cfg, "strip_accents", where, lowercase),
            clean_text=get_bool(cfg, "clean_text", where, True),
            handle_chinese_chars=get_bool(cfg, "handle_chinese_chars", where, True),
        )
    if kind == "Prepend":
        return Prepend(get_str(cfg, "prepend", where, ""))
    if kind == "Replace":
        return Replace(
            pattern_from_config(cfg.get("pattern"), f"{where}.pattern"),
            get_str(cfg, "content", where, ""),
        )
    if kind == "Precompiled":
        return Precompiled.from_base64(get_str(cfg, "precompiled_charsmap", where, ""))
    raise ValueError(
        f"{where}: unsupported type {kind!r}; supported are Sequence, NFC, NFD, "
        f"NFKC, NFKD, Lowercase, StripAccents, Strip, BertNormalizer, Prepend, "
        f"Replace, Nmt and Precompiled"
    )
