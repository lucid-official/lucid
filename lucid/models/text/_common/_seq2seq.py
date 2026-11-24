"""The decode loop the encoder-decoder text families share.

Transformer and T5 decode the same way once the encoder has run: seed
the target with a start token, ask the decoder for the next-token logits,
pick a token through the shared sampling primitives, and stop at the
length cap or when every row has emitted end-of-sequence.  What differs
is how a family turns a target prefix into logits, which it hands in as
a callable.
"""

from collections.abc import Callable

import lucid
from lucid._tensor.tensor import Tensor
from lucid.models._sampling import _SamplingParams, _select_next_ondevice
from lucid.utils.cache import DynamicCache, EncoderDecoderCache, StaticCache

__all__ = ["NextTokenLogits", "seq2seq_generate"]

#: ``(tokens, cache, [cache_position]) -> (B, vocab)`` logits for the last target position.
#: With a cache ``tokens`` is the newest ``(B, 1)`` column and the cache
#: holds everything before it; without one it is the whole prefix.
type NextTokenLogits = Callable[..., Tensor]


def seq2seq_generate(
    next_token_logits: NextTokenLogits,
    *,
    batch: int,
    device: str,
    start_token_id: int,
    max_length: int,
    use_cache: bool,
    sampling: _SamplingParams,
    cache_implementation: str = "dynamic",
    max_cache_len: int | None = None,
) -> Tensor:
    """Unroll a decoder from ``start_token_id`` until ``max_length`` tokens.

    Parameters
    ----------
    next_token_logits : NextTokenLogits
        The family's decoder and head over the already-encoded source.
    batch : int
        Number of rows being decoded.
    device : str
        Device the token rows live on.
    start_token_id : int
        First target token of every row.
    max_length : int
        Cap on the returned length, the start token included.
    use_cache : bool
        Decode one new token per step against an
        :class:`~lucid.utils.cache.EncoderDecoderCache` instead of
        re-reading the whole prefix.
    sampling : _SamplingParams
        Greedy or stochastic selection, end-of-sequence and padding ids.
    cache_implementation : str, optional, keyword-only, default="dynamic"
        ``"dynamic"`` uses :class:`~lucid.utils.cache.DynamicCache` for
        decoder self-attention; ``"static"`` uses a pre-allocated
        :class:`~lucid.utils.cache.StaticCache`; ``"compiled"`` pairs a
        self-attention :class:`~lucid.utils.cache.StaticCache` with
        :func:`~lucid.compile._entry.decode_step.compiled_decode_step` so
        each decode step runs through a single cached executable.
    max_cache_len : int or None, optional, keyword-only
        Self-attention buffer capacity for ``"static"`` or ``"compiled"``;
        defaults to ``max_length``.

    Returns
    -------
    Tensor
        ``(batch, T_out)`` int64 target tokens, ``T_out <= max_length``.
        A row that emitted end-of-sequence is padded with
        ``sampling.pad_token_id`` while the others finish.
    """
    finished = lucid.zeros((batch,), device=device).bool()
    out_tokens = [lucid.tensor([start_token_id] * batch, device=device).long()]
    cache: EncoderDecoderCache | None = None
    compiled_step = None
    if use_cache:
        self_cache = (
            StaticCache(max_cache_len=(max_cache_len if max_cache_len is not None else max_length))
            if cache_implementation in ("static", "compiled")
            else DynamicCache()
        )
        cache = EncoderDecoderCache(self_cache, DynamicCache())
        if cache_implementation == "compiled":
            from lucid.compile._entry.decode_step import compiled_decode_step

            compiled_step = compiled_decode_step(
                lambda tok, cp: next_token_logits(tok, cache, cp), cache
            )
    for pos in range(max_length - 1):
        # The cache already holds every column but the newest.
        tokens = (
            out_tokens[-1].reshape(batch, 1)
            if cache is not None
            else lucid.stack(out_tokens, dim=1)
        )
        if compiled_step is not None:
            cp = lucid.tensor([pos], device=device).long()
            logits = compiled_step(tokens, cp)
        else:
            logits = next_token_logits(tokens, cache)
        finished = _select_next_ondevice(logits, out_tokens, finished, sampling)
        if sampling.eos_token_id is not None and bool(lucid.all(finished).item()):
            break
    return lucid.stack(out_tokens, dim=1).long()
