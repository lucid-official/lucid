"""BERT family tokenizer wrappers — WordPiece with BERT defaults.

BERT (Devlin et al., 2018) ships a WordPiece vocabulary together
with a canonical set of special tokens that every BERT checkpoint
relies on:

* ``[UNK]`` — unknown / OOV fallback
* ``[CLS]`` — first-position classification head
* ``[SEP]`` — segment-pair separator (also used as EOS in some heads)
* ``[PAD]`` — padding
* ``[MASK]`` — masked-LM placeholder

Both :class:`BERTTokenizer` (pure-Python) and
:class:`BERTTokenizerFast` (C++-backed) subclass the matching
algorithm classes in :mod:`lucid.utils.tokenizer._wordpiece` and
register the five tokens above by default.  The defaults match
``bert-base-uncased`` so any HF ``vocab.txt`` checkpoint loads via
:meth:`~lucid.utils.tokenizer.WordPieceTokenizer.from_pretrained`
without modification.

Two flavours, bit-identical encode output:

* :class:`BERTTokenizer` — the easy-to-debug reference path.
* :class:`BERTTokenizerFast` — the production path; the greedy
  longest-match loop runs in C++ via the engine-side
  ``WordPiece`` binding.

Lower-casing
------------
The default normalizer is :class:`~lucid.utils.tokenizer.normalizers.
BERTNormalizer` with ``lowercase=True``, matching the uncased BERT
family.  Set ``do_lower_case=False`` (or pass an explicit
``normalizer``) to load cased checkpoints such as
``bert-base-cased``.
"""

from lucid.utils.tokenizer._base import SpecialTokens
from lucid.utils.tokenizer._normalizers import BERTNormalizer, Normalizer
from lucid.utils.tokenizer._pre_tokenizers import PreTokenizer, WhitespacePunctuationSplit
from lucid.utils.tokenizer._wordpiece import WordPieceTokenizer, WordPieceTokenizerFast


def _bert_special_tokens() -> SpecialTokens:
    """Return BERT's canonical 5-slot special-token registry.

    Returns
    -------
    SpecialTokens
        Registry with ``unk=[UNK]``, ``pad=[PAD]``, ``cls=[CLS]``,
        ``sep=[SEP]``, ``mask=[MASK]``.  Used as the default for
        both :class:`BERTTokenizer` and :class:`BERTTokenizerFast`
        when the caller does not pass an explicit ``special_tokens``.
    """
    return SpecialTokens(unk="[UNK]", pad="[PAD]", cls="[CLS]", sep="[SEP]", mask="[MASK]")


class BERTTokenizer(WordPieceTokenizer):
    r"""BERT tokenizer — pure-Python reference.

    A thin convenience subclass of
    :class:`~lucid.utils.tokenizer.WordPieceTokenizer` with BERT's
    canonical ``[UNK]/[CLS]/[SEP]/[PAD]/[MASK]`` registry, the
    :class:`~lucid.utils.tokenizer.normalizers.BERTNormalizer`
    (lowercase + accent-strip + Chinese-char split + clean-text), and
    :class:`~lucid.utils.tokenizer.pre_tokenizers.WhitespacePunctuationSplit`
    pre-tokenizer baked in as defaults.

    Loads any published HF BERT-family ``vocab.txt`` checkpoint
    unchanged.

    Parameters
    ----------
    vocab : dict[str, int]
        BERT-style vocab.  Continuation subwords prefixed with ``##``.
        Pass ``{}`` and call :meth:`~WordPieceTokenizer.train` to
        train a fresh vocab from a corpus.
    unk_token : str, default ``"[UNK]"``
        Token emitted when no valid longest-match prefix is found.
        Must be present in ``vocab`` for the id to materialise.
    continuing_prefix : str, default ``"##"``
        Marker for non-initial subwords (BERT convention).
    max_chars_per_word : int, default 100
        Words longer than this skip the longest-match loop and emit
        the ``unk_token`` directly (mirrors BERT ``BasicTokenizer``).
    do_lower_case : bool, default ``True``
        Forwarded to :class:`BERTNormalizer` as both ``lowercase`` and
        ``strip_accents``.  Set to ``False`` for cased checkpoints
        (``bert-base-cased``, multilingual models), which keep their
        accents as well as their case.  Ignored if ``normalizer`` is
        given explicitly.
    normalizer : Normalizer, optional
        Override the default :class:`BERTNormalizer`.
    pre_tokenizer : PreTokenizer, optional
        Override the default :class:`WhitespacePunctuationSplit`.
    special_tokens : SpecialTokens, optional
        Override the default BERT registry.

    Examples
    --------
    Train a small BERT-style tokenizer from a corpus, then round-trip:

    >>> from lucid.models.text.bert import BERTTokenizer
    >>> tok = BERTTokenizer(vocab={})
    >>> tok.train(["hello world", "hello bert"] * 8, vocab_size=60)
    >>> ids = tok.encode("Hello World", add_special_tokens=False)
    >>> # default do_lower_case=True → "Hello" and "hello" share ids.
    >>> tok.decode(ids, skip_special_tokens=False)
    'hello world'

    Encoding with ``add_special_tokens=True`` (the default) frames the
    sequence as ``[CLS] … [SEP]``, the input every BERT head was trained
    on; ``[CLS]`` is the position the pooler reads.

    >>> vocab = {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "hello": 3}
    >>> BERTTokenizer(vocab=vocab).encode("hello")
    [1, 3, 2]

    See Also
    --------
    BERTTokenizerFast : C++-backed variant with identical output.
    """

    # [CLS] A [SEP] — and [CLS] A [SEP] B [SEP] for a pair.  The registry
    # has always named both tokens; without this the base framed with BOS
    # and EOS, which BERT does not define, so nothing was added at all and
    # the pooler read the first word instead of [CLS].
    _special_layout = "cls_sep"

    def __init__(
        self,
        vocab: dict[str, int],
        *,
        unk_token: str = "[UNK]",
        continuing_prefix: str = "##",
        max_chars_per_word: int = 100,
        do_lower_case: bool = True,
        normalizer: Normalizer | None = None,
        pre_tokenizer: PreTokenizer | None = None,
        special_tokens: SpecialTokens | None = None,
    ) -> None:
        super().__init__(
            vocab,
            unk_token=unk_token,
            continuing_prefix=continuing_prefix,
            max_chars_per_word=max_chars_per_word,
            normalizer=(
                normalizer
                if normalizer is not None
                else BERTNormalizer(lowercase=do_lower_case, strip_accents=do_lower_case)
            ),
            pre_tokenizer=(
                pre_tokenizer if pre_tokenizer is not None else WhitespacePunctuationSplit()
            ),
            special_tokens=(
                special_tokens if special_tokens is not None else _bert_special_tokens()
            ),
        )


class BERTTokenizerFast(WordPieceTokenizerFast):
    """BERT tokenizer — C++-backed.

    Bit-identical to :class:`BERTTokenizer`: same vocab format, same
    special-token registry, same encode output.  The greedy
    longest-match loop runs in C++ via the engine-side ``WordPiece``
    binding, typically 20-30x faster than the pure-Python flavour on
    large corpora.

    Constructor parameters mirror :class:`BERTTokenizer` exactly —
    see that class for the full reference.

    Examples
    --------
    >>> from lucid.models.text.bert._tokenizer import BERTTokenizerFast
    >>> vocab = {"[UNK]": 0, "he": 1, "##llo": 2}
    >>> BERTTokenizerFast(vocab=vocab).encode("hello")
    [1, 2]

    The same ids the plain tokenizer produces. The speed is in the
    lookup, and a pair that segmented differently would be a bug no type
    can catch.  No ``[CLS]`` / ``[SEP]`` appear only because this toy
    vocabulary has neither; with them it frames as :class:`BERTTokenizer`
    does.
    """

    _special_layout = "cls_sep"

    def __init__(
        self,
        vocab: dict[str, int],
        *,
        unk_token: str = "[UNK]",
        continuing_prefix: str = "##",
        max_chars_per_word: int = 100,
        do_lower_case: bool = True,
        normalizer: Normalizer | None = None,
        pre_tokenizer: PreTokenizer | None = None,
        special_tokens: SpecialTokens | None = None,
    ) -> None:
        super().__init__(
            vocab,
            unk_token=unk_token,
            continuing_prefix=continuing_prefix,
            max_chars_per_word=max_chars_per_word,
            normalizer=(
                normalizer
                if normalizer is not None
                else BERTNormalizer(lowercase=do_lower_case, strip_accents=do_lower_case)
            ),
            pre_tokenizer=(
                pre_tokenizer if pre_tokenizer is not None else WhitespacePunctuationSplit()
            ),
            special_tokens=(
                special_tokens if special_tokens is not None else _bert_special_tokens()
            ),
        )
