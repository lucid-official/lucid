"""Added tokens — vocabulary entries matched in the text before anything else.

A ``tokenizer.json`` lists its special tokens (``</s>``, ``<pad>``,
``<extra_id_0>`` …) under ``added_tokens``, and Hugging Face cuts them out
of the input *before* normalization and pre-tokenization run.  That order
is the whole point: a literal ``"</s>"`` typed into the text must become
id 1, not be normalized, split on its punctuation, and spelled out as
``▁</``, ``s``, ``>``.  The rest of the text is then processed piece by
piece around the cut-out tokens.

Two passes, as Hugging Face does them.  Tokens marked ``normalized: false``
— every special token in practice — are matched against the raw text.
Tokens marked ``normalized: true`` are matched afterwards against each
remaining piece's *normalized* form, with their own content normalized the
same way, so a lowercasing pipeline still finds them.
"""

import re
from dataclasses import dataclass

from lucid.utils.tokenizer._config import (
    WHITE_SPACE,
    as_list,
    as_object,
    get_bool,
    get_str,
)
from lucid.utils.tokenizer._normalizers import Normalizer

#: One piece of split text: the text, the added token's id when the piece
#: *is* an added token (``None`` otherwise), and where the piece starts in
#: the string that was split.
Segment = tuple[str, int | None, int]


@dataclass(frozen=True, slots=True)
class AddedToken:
    r"""One ``added_tokens`` entry of a ``tokenizer.json``.

    Parameters
    ----------
    content : str
        The token's surface, matched literally.
    id : int
        The id it encodes to.
    special : bool, default False
        Whether it is a special token — registered as such and dropped by
        ``decode(skip_special_tokens=True)``.
    single_word : bool, default False
        Match only when not glued to a word character on either side.
    lstrip : bool, default False
        The match also swallows whitespace to its left.
    rstrip : bool, default False
        The match also swallows whitespace to its right.
    normalized : bool, default False
        Match against normalized text instead of the raw input.
    """

    content: str
    id: int
    special: bool = False
    single_word: bool = False
    lstrip: bool = False
    rstrip: bool = False
    normalized: bool = False

    def to_config(self) -> dict[str, object]:
        """Return the ``added_tokens`` entry for ``tokenizer.json``.

        Returns
        -------
        dict of str to object
            The Hugging Face schema for one added token.
        """
        return {
            "id": self.id,
            "content": self.content,
            "single_word": self.single_word,
            "lstrip": self.lstrip,
            "rstrip": self.rstrip,
            "normalized": self.normalized,
            "special": self.special,
        }


def added_tokens_from_config(value: object) -> list[AddedToken]:
    """Parse a ``tokenizer.json`` ``added_tokens`` array.

    Parameters
    ----------
    value : list or None
        The array; ``None`` means there are none.

    Returns
    -------
    list of AddedToken
        One entry per array element, in file order.

    Raises
    ------
    ValueError
        If an entry lacks an integer ``id`` or a string ``content``.
    """
    if value is None:
        return []
    out: list[AddedToken] = []
    for raw in as_list(value, "added_tokens"):
        entry = as_object(raw, "added_tokens")
        tid = entry.get("id")
        if not isinstance(tid, int) or isinstance(tid, bool):
            raise TypeError(f"added_tokens: entry without an integer id: {entry!r}")
        where = f"added_tokens[{tid}]"
        content = get_str(entry, "content", where, "")
        if not content:
            raise ValueError(f"{where}: entry without content")
        special = get_bool(entry, "special", where, False)
        out.append(
            AddedToken(
                content=content,
                id=tid,
                special=special,
                single_word=get_bool(entry, "single_word", where, False),
                lstrip=get_bool(entry, "lstrip", where, False),
                rstrip=get_bool(entry, "rstrip", where, False),
                # Hugging Face's default when the key is absent: special
                # tokens are matched raw, ordinary added tokens normalized.
                normalized=get_bool(entry, "normalized", where, not special),
            )
        )
    return out


def _is_word_char(ch: str) -> bool:
    """What ``single_word`` refuses to be glued to."""
    return ch.isalnum() or ch == "_"


class _Matcher:
    """Leftmost-longest matching of a fixed set of token surfaces."""

    def __init__(self, by_pattern: dict[str, AddedToken]) -> None:
        self._by_pattern = by_pattern
        # Longest first: at any one position the first alternative that
        # matches is then the longest, and ``finditer`` already scans left
        # to right — together that is leftmost-longest, the matching
        # Hugging Face's Aho-Corasick automaton is configured for.
        ordered = sorted((p for p in by_pattern if p), key=len, reverse=True)
        self._regex = re.compile("|".join(map(re.escape, ordered))) if ordered else None

    def split(self, text: str) -> list[Segment]:
        """Cut every added token out of ``text``."""
        if self._regex is None or not text:
            return [(text, None, 0)]
        segments: list[Segment] = []
        consumed = 0
        for m in self._regex.finditer(text):
            start, stop = m.start(), m.end()
            token = self._by_pattern[m.group()]
            if token.single_word:
                clear_left = start == 0 or not _is_word_char(text[start - 1])
                clear_right = stop == len(text) or not _is_word_char(text[stop])
                if not (clear_left and clear_right):
                    continue
            if token.lstrip:
                # Whitespace a previous match already swallowed stays with it.
                start = max(len(text[:start].rstrip(WHITE_SPACE)), consumed)
            if token.rstrip:
                tail = text[stop:]
                stop += len(tail) - len(tail.lstrip(WHITE_SPACE))
            if consumed < start:
                segments.append((text[consumed:start], None, consumed))
            segments.append((text[start:stop], token.id, start))
            consumed = stop
        if consumed < len(text):
            segments.append((text[consumed:], None, consumed))
        return segments


class AddedVocabulary:
    r"""The added tokens of one tokenizer, ready to cut out of text.

    Parameters
    ----------
    tokens : list of AddedToken
        The entries, typically from :func:`added_tokens_from_config`.
    normalizer : Normalizer, optional
        The tokenizer's normalizer, applied to the content of tokens marked
        ``normalized`` so they are matched in the space they will be
        compared in.

    Examples
    --------
    >>> from lucid.utils.tokenizer._added_tokens import (
    ...     AddedToken, AddedVocabulary,
    ... )
    >>> vocab = AddedVocabulary([AddedToken("</s>", 1, special=True)])
    >>> vocab.split_raw("hello</s>world")
    [('hello', None, 0), ('</s>', 1, 5), ('world', None, 9)]
    """

    def __init__(self, tokens: list[AddedToken], normalizer: Normalizer | None = None) -> None:
        r"""Build the two matchers.

        Parameters
        ----------
        tokens : list of AddedToken
            The entries.
        normalizer : Normalizer, optional
            Applied to the content of ``normalized`` tokens.
        """
        self._tokens = list(tokens)
        raw: dict[str, AddedToken] = {}
        normalized: dict[str, AddedToken] = {}
        for token in self._tokens:
            if token.normalized:
                key = normalizer(token.content) if normalizer is not None else token.content
                normalized[key] = token
            else:
                raw[token.content] = token
        self._raw = _Matcher(raw)
        self._normalized = _Matcher(normalized)
        self._id_to_content = {t.id: t.content for t in self._tokens}

    @property
    def tokens(self) -> list[AddedToken]:
        """The entries, in the order given.

        Returns
        -------
        list of AddedToken
            A copy of the entry list.
        """
        return list(self._tokens)

    def content_for(self, token_id: int) -> str | None:
        """Surface of the added token with ``token_id``, if there is one.

        Parameters
        ----------
        token_id : int
            An id.

        Returns
        -------
        str or None
            The surface, or ``None`` when no added token has that id.
        """
        return self._id_to_content.get(token_id)

    def split_raw(self, text: str) -> list[Segment]:
        """Cut the raw-matched tokens out of the input.

        Parameters
        ----------
        text : str
            The raw input.

        Returns
        -------
        list of (str, int or None, int)
            Pieces in order; token pieces carry their id.
        """
        return self._raw.split(text)

    def split_normalized(self, text: str) -> list[Segment]:
        """Cut the normalized-matched tokens out of one normalized piece.

        Parameters
        ----------
        text : str
            A piece of the input after normalization.

        Returns
        -------
        list of (str, int or None, int)
            Pieces in order; token pieces carry their id.
        """
        return self._normalized.split(text)
