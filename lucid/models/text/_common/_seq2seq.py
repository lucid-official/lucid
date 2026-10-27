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
from lucid.utils.cache import DynamicCache, EncoderDecoderCache

__all__ = ["NextTokenLogits", "seq2seq_generate"]

#: ``(tokens, cache) -> (B, vocab)`` logits for the last target position.
#: With a cache ``tokens`` is the newest ``(B, 1)`` column and the cache
#: holds everything before it; without one it is the whole prefix.
type NextTokenLogits = Callable[[Tensor, EncoderDecoderCache | None], Tensor]


def seq2seq_generate(
    next_token_logits: NextTokenLogits,
    *,
    batch: int,
    device: str,
    start_token_id: int,
    max_length: int,
    use_cache: bool,
    sampling: _SamplingParams,
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

    Returns
    -------
    Tensor
        ``(batch, T_out)`` int64 target tokens, ``T_out <= max_length``.
        A row that emitted end-of-sequence is padded with
        ``sampling.pad_token_id`` while the others finish.
    """
    finished = lucid.zeros((batch,), device=device).bool()
    out_tokens = [lucid.tensor([start_token_id] * batch, device=device).long()]
    cache = EncoderDecoderCache(DynamicCache(), DynamicCache()) if use_cache else None
    for _ in range(max_length - 1):
        # The cache already holds every column but the newest.
        tokens = (
            out_tokens[-1].reshape(batch, 1)
            if cache is not None
            else lucid.stack(out_tokens, dim=1)
        )
        logits = next_token_logits(tokens, cache)
        finished = _select_next_ondevice(logits, out_tokens, finished, sampling)
        if sampling.eos_token_id is not None and bool(lucid.all(finished).item()):
            break
    return lucid.stack(out_tokens, dim=1).long()
