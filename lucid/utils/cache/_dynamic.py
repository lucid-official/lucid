"""``DynamicCache`` — the default growing key/value cache.

Mirrors the reference ``DynamicCache``: same class name, same ``key_cache`` /
``value_cache`` / ``_seen_tokens`` attributes, same ``update`` /
``get_seq_length`` / ``to_legacy_cache`` / ``from_legacy_cache`` surface, so
existing generation code ports verbatim.  Pure ``lucid`` core ops only
(``lucid.cat`` along the sequence dimension); no engine work required.
"""

from typing import TYPE_CHECKING, Iterator, Self, final, override

import lucid
from lucid.utils.cache._base import Cache

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

# Per-layer key/value tensors are laid out ``(B, num_heads, T, head_dim)`` in
# every Lucid attention, so the sequence axis is the second-to-last (dim -2).
_SEQ_DIM = -2


@final
class DynamicCache(Cache):
    """A cache that grows its key/value tensors by concatenation each step.

    Holds one key tensor and one value tensor per layer in ``key_cache`` and
    ``value_cache`` (see the Attributes section).  The
    first :meth:`update` for a layer seeds the
    entry; subsequent updates concatenate the new token(s) along the sequence
    dimension.  This is the eager default for autoregressive decoding.

    Attributes
    ----------
    key_cache : list[Tensor]
        Per-layer accumulated keys, each ``(B, num_heads, T, head_dim)``.
    value_cache : list[Tensor]
        Per-layer accumulated values, same shapes.
    _seen_tokens : int
        Running count of tokens the cache has processed (tracked from
        ``layer_idx == 0``), exposed via :attr:`seen_tokens`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.utils.cache import DynamicCache
    >>> cache = DynamicCache()
    >>> k = lucid.zeros(1, 2, 3, 4)   # (B, H, T, D)
    >>> v = lucid.zeros(1, 2, 3, 4)
    >>> ck, cv = cache.update(k, v, layer_idx=0)
    >>> cache.get_seq_length()
    3
    """

    def __init__(self) -> None:
        """Create an empty cache with no layers populated."""
        self._seen_tokens: int = 0
        self.key_cache: list[Tensor] = []
        self.value_cache: list[Tensor] = []

    @property
    def seen_tokens(self) -> int:
        """Total number of tokens processed across all :meth:`update` calls."""
        return self._seen_tokens

    def __len__(self) -> int:
        """Number of layers currently held in the cache."""
        return len(self.key_cache)

    def __getitem__(self, layer_idx: int) -> tuple[Tensor, Tensor]:
        """Return the ``(keys, values)`` pair cached for ``layer_idx``."""
        if 0 <= layer_idx < len(self.key_cache):
            return self.key_cache[layer_idx], self.value_cache[layer_idx]
        raise KeyError(f"Cache has {len(self.key_cache)} layers; no entry for layer {layer_idx}.")

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        """Iterate ``(keys, values)`` pairs in layer order."""
        for layer_idx in range(len(self.key_cache)):
            yield self.key_cache[layer_idx], self.value_cache[layer_idx]

    @override
    def update(
        self,
        key_states: Tensor,
        value_states: Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, object] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Concatenate ``key_states`` / ``value_states`` onto ``layer_idx`` and
        return the grown key/value tensors.

        The first call for a layer seeds the entry; later calls
        :func:`lucid.cat` the new token(s) onto the sequence dimension.

        Parameters
        ----------
        key_states : Tensor
            New key projections of shape ``(B, num_heads, T_new, head_dim)``.
        value_states : Tensor
            New value projections of the same shape.
        layer_idx : int
            Index of the transformer layer this update belongs to.
        cache_kwargs : dict or None, optional
            Accepted for interface parity; unused by the dynamic cache.

        Returns
        -------
        tuple of (Tensor, Tensor)
            The accumulated ``(keys, values)`` for ``layer_idx``, each of shape
            ``(B, num_heads, T_total, head_dim)``.
        """
        if layer_idx == 0:
            self._seen_tokens += int(key_states.shape[_SEQ_DIM])

        if len(self.key_cache) <= layer_idx:
            # First time we see this layer — seed the entry.  Layers are
            # expected to update in order (0, 1, 2, ...), matching the trunk's
            # forward loop.
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = lucid.cat(
                [self.key_cache[layer_idx], key_states], dim=_SEQ_DIM
            )
            self.value_cache[layer_idx] = lucid.cat(
                [self.value_cache[layer_idx], value_states], dim=_SEQ_DIM
            )
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    @override
    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Number of cached tokens for ``layer_idx`` (0 if not yet seeded),
        read directly from the accumulated tensor's sequence dimension."""
        if len(self.key_cache) <= layer_idx:
            return 0
        return int(self.key_cache[layer_idx].shape[_SEQ_DIM])

    @override
    def get_max_cache_shape(self) -> int | None:
        """Always ``None`` — the dynamic cache is unbounded and grows on demand."""
        return None

    @override
    def reorder_cache(self, beam_idx: Tensor) -> None:
        """Reindex every layer's keys/values along the batch dimension with
        ``beam_idx`` (in place) to keep beams aligned after a beam permutation."""
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx)
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx)

    def reset(self) -> None:
        """Empty the cache in place so the same object can be reused for a
        fresh sequence (clears every layer and the token counter)."""
        self._seen_tokens = 0
        self.key_cache = []
        self.value_cache = []

    def crop(self, max_length: int) -> None:
        """Truncate the cache to at most ``max_length`` tokens along the
        sequence dimension (in place).

        A negative ``max_length`` drops that many tokens off the end.  Useful
        for rolling the cache back (e.g. rejecting speculative tokens).  A
        ``max_length`` at or above the current length is a no-op.
        """
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)
        if self.get_seq_length() <= max_length:
            return
        self._seen_tokens = max_length
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][:, :, :max_length, :]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][:, :, :max_length, :]

    def batch_repeat_interleave(self, repeats: int) -> None:
        """Repeat every batch row ``repeats`` times along the batch dimension
        (in place) — e.g. to fan one prompt out into ``repeats`` samples."""
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx].repeat_interleave(repeats, dim=0)
            self.value_cache[layer_idx] = self.value_cache[layer_idx].repeat_interleave(
                repeats, dim=0
            )

    def batch_select_indices(self, indices: Tensor) -> None:
        """Keep only the batch rows named by ``indices`` along the batch
        dimension (in place)."""
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, indices)
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, indices)

    def to_legacy_cache(self) -> tuple[tuple[Tensor, Tensor], ...]:
        """Convert to the legacy ``tuple(tuple(key, value), ...)`` format.

        Returns one ``(key, value)`` pair per layer, in layer order — the
        representation older decoder ``forward`` signatures consumed before
        :class:`Cache` objects existed.
        """
        return tuple(
            (self.key_cache[layer_idx], self.value_cache[layer_idx])
            for layer_idx in range(len(self.key_cache))
        )

    @classmethod
    def from_legacy_cache(
        cls, past_key_values: tuple[tuple[Tensor, Tensor], ...] | None = None
    ) -> Self:
        """Build a :class:`DynamicCache` from the legacy tuple format.

        ``past_key_values`` is a per-layer tuple of ``(key, value)`` tensors;
        ``None`` yields an empty cache.
        """
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache
