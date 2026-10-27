"""GPT-1 family tokenizer wrappers — lowercased word-level BPE.

GPT-1 (Radford et al., 2018) §4.1 uses "a bytepair encoding (BPE)
vocabulary with 40,000 merges", built over *words*: the text is cleaned,
pre-tokenised, lowercased, and each word is split into characters with the
final one carrying an end-of-word marker ``</w>``.  Byte-level BPE — the
``Ġ``-prefixed scheme these wrappers used to inherit — arrived with GPT-2
and is a different tokenizer entirely.

Divergence, deliberate: the paper cleans text with **ftfy** and
pre-tokenises with **spaCy**, and H4 forbids both inside ``lucid/``.  The
reference implementation already provides for their absence — its
tokenizer "uses SpaCy tokenizer and ftfy for pre-BPE tokenization if they
are installed, fallback to BERT's ``BasicTokenizer`` if not" — and that
fallback is what is built here, out of pieces Lucid already has:
:class:`~lucid.utils.tokenizer._pre_tokenizers.WhitespacePunctuationSplit`
for the BasicTokenizer role and
:class:`~lucid.utils.tokenizer._normalizers.Lowercase` for the casing.  So ids match the published vocabulary wherever the fallback and
spaCy agree on word boundaries, and diverge where they do not — mostly
contractions and unusual punctuation runs.

Loads a GPT-1 ``vocab.json`` + ``merges.txt`` pair unmodified.
"""

from typing import override

from lucid.utils.tokenizer._base import SpecialTokens
from lucid.utils.tokenizer._bpe import BPETokenizer
from lucid.utils.tokenizer._normalizers import Lowercase, NFC, Normalizer, Sequence
from lucid.utils.tokenizer._pre_tokenizers import PreTokenizer, WhitespacePunctuationSplit

__all__ = ["GPTTokenizer", "GPTTokenizerFast", "END_OF_WORD"]

END_OF_WORD = "</w>"
"""GPT-1's end-of-word marker, glued to a word's final character."""


class GPTTokenizer(BPETokenizer):
    r"""GPT-1 tokenizer — lowercased word-level BPE with ``</w>``.

    The end-of-word marker is what distinguishes this from generic BPE:
    a word is seeded as ``list(word[:-1]) + [word[-1] + "</w>"]``, so the
    merge table can tell ``"in"`` inside *inside* from ``"in</w>"`` as a
    whole word.  Without it a subword and a complete word share an id and
    the vocabulary silently means two things.

    Parameters
    ----------
    vocab : dict[str, int]
        Token-string → id map from GPT-1's ``vocab.json``.  Entries ending
        in ``</w>`` are whole-word forms.
    merges : list[tuple[str, str]]
        Ordered BPE merges; index = rank, lower applied first.
    normalizer : Normalizer, optional
        Defaults to ``NFC`` then ``Lowercase`` — §4.1 lowercases.
    pre_tokenizer : PreTokenizer, optional
        Defaults to :class:`WhitespacePunctuationSplit`, the
        ``BasicTokenizer`` role in the reference's no-spaCy fallback.
    special_tokens : SpecialTokens, optional
        GPT-1 trained without bos/eos; pass them only if downstream code
        needs them.  They are registered, never inserted by ``encode``.

    Examples
    --------
    >>> from lucid.models.text.gpt import GPTTokenizer
    >>> vocab = {"l": 0, "o": 1, "o</w>": 2, "lo</w>": 3}
    >>> tok = GPTTokenizer(vocab=vocab, merges=[("l", "o</w>")])
    >>> tok.encode("lo")
    [3]
    >>> tok.decode(tok.encode("lo"))
    'lo'
    """

    # No automatic framing, whatever the registry names.  GPT-1's start,
    # delimiter and extract tokens belong to the task-specific input
    # transformations of §3.3, which place them per task; a blanket BOS/EOS
    # from ``encode`` would be a framing the paper never used.
    _special_layout = "none"

    def __init__(
        self,
        vocab: dict[str, int],
        merges: list[tuple[str, str]],
        *,
        normalizer: Normalizer | None = None,
        pre_tokenizer: PreTokenizer | None = None,
        special_tokens: SpecialTokens | None = None,
    ) -> None:
        super().__init__(
            vocab,
            merges,
            normalizer=normalizer or Sequence([NFC(), Lowercase()]),
            pre_tokenizer=pre_tokenizer or WhitespacePunctuationSplit(),
            special_tokens=special_tokens,
        )

    @override
    def _encode_chunk(self, chunk: str) -> list[int]:
        """Seed with GPT-1's end-of-word symbol, then merge as usual.

        Only the seeding differs from :class:`BPETokenizer`; the merge
        loop is inherited, so the two stay in step if it ever changes.
        """
        if not chunk:
            return []
        symbols = list(chunk[:-1]) + [chunk[-1] + END_OF_WORD]
        ids: list[int] = []
        unk_id = self.unk_token_id
        for sym in symbols:
            tid = self._vocab.get(sym)
            if tid is not None:
                ids.append(tid)
            elif unk_id is not None:
                ids.append(unk_id)
        return self._merge_ids(ids)

    @override
    def _decode_one(self, ids: list[int]) -> str:
        """Join surfaces, turning each ``</w>`` back into a space.

        The marker *is* the word boundary, so dropping it without
        substituting a space would run every word together.
        """
        out: list[str] = []
        for i in ids:
            tok = self._id_to_token.get(i)
            if tok is not None:
                out.append(tok)
        return "".join(out).replace(END_OF_WORD, " ").strip()


class GPTTokenizerFast(GPTTokenizer):
    """GPT-1 tokenizer — same results, same Python merge loop.

    There is no C++ acceleration for this scheme.  The engine's ``BPE``
    seeds per codepoint, which was measured directly: encoding
    ``"hello</w>"`` against a vocabulary containing ``"o</w>"`` returns
    ``['he', 'l', 'l', 'o', '<', '/', 'w', '>']`` — the marker is torn
    into four symbols.  Expressing GPT-1's end-of-word symbol there needs
    an engine change, so this subclass exists to keep the name working and
    the output correct, not because it is faster.

    Byte-level families (GPT-2, RoBERTa) do have a genuine fast path; see
    :class:`~lucid.utils.tokenizer.ByteLevelBPETokenizerFast`.

    Examples
    --------
    >>> from lucid.models.text.gpt._tokenizer import GPTTokenizerFast
    >>> vocab = {"l": 0, "o": 1, "o</w>": 2, "lo</w>": 3}
    >>> tok = GPTTokenizerFast(vocab=vocab, merges=[("l", "o</w>")])
    >>> tok.encode("lo")
    [3]

    One id: the merge applies and the pair collapses into the vocabulary
    entry that covers both.
    """
