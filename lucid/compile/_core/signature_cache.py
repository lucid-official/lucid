"""
lucid.compile._core.signature_cache — the executables one live wrapper keeps.

An ``MPSGraphExecutable`` keeps the intermediate buffers MPSGraph planned for it
from its first run until it is released, so every signature a long-lived wrapper
has compiled pins memory in proportion to its tensors (``obsidian`` note
``engine-mpsgraph-executable-pins-intermediates``).  The process-wide session cache
(``ExecutableCache``) bounds only the references it holds; a live
:class:`~lucid.compile._entry.module.CompiledModule` kept every signature's
executable in a plain dict, and so did ``make_step``, ``fused_step`` and the decode
step.  :class:`SignatureCache` is that dict, bounded: one owner for the policy, used
by each of them.

Policy: least recently used out, on an entry cap and on a byte cap, the entry just
stored spared.  A dropped signature costs a recompile on its next call and nothing
else — an entry still running holds its own reference.  The byte count is
:func:`_C_engine.compile.retained_bytes_estimate` of the graph each entry was built
from, the number the session cache counts, so the layers agree.  An executable the
session cache also holds stays pinned until that cache lets go: the two caps bound
the union, and a wrapper's cap bounds what the session cache's eviction cannot.
"""

import os
import re
import sys
import threading
from collections.abc import Callable, Hashable, ItemsView, Iterator, MutableMapping, ValuesView
from dataclasses import dataclass
from typing import Protocol, overload, override

from lucid._C import engine as _C_engine
from lucid.compile._core.fallback import _narrate

__all__ = ["HasRetainedBytes", "SignatureCache", "SignatureLimits", "signature_limits"]

ENTRIES_ENV = "LUCID_COMPILE_MAX_SIGNATURES"
BYTES_ENV = "LUCID_COMPILE_MAX_SIGNATURE_BYTES"

#: Signatures a wrapper keeps by default.  A graph's fixed cost (Metal and MPSGraph state
#: the byte estimate does not see) measured 1.1 MB for a 3-op MLP, 1.7 MB for a 6-conv
#: net and 1.9 MB for a 3-layer transformer encoder, so 256 bounds it near half a
#: gigabyte; a training or inference loop has a handful of signatures.
DEFAULT_MAX_ENTRIES = 256

_UINT64_MAX = 2**64 - 1
#: Digits of ``_UINT64_MAX``.  ``int()`` refuses a longer string past 4300 digits with a
#: ``ValueError`` of its own, so the length is checked first: an unusable value must
#: reach the fallback and its one-time message, whatever its length.
_MAX_DIGITS = len(str(_UINT64_MAX))
_COUNT = re.compile(r"[0-9]+")
_BYTES = re.compile(r"([0-9]+)([kKmMgG]?)")
_SHIFT = {"": 0, "k": 10, "m": 20, "g": 30}

#: Variables already reported as unusable: said once per process, as the engine's are.
_REPORTED: set[str] = set()

#: What ``pop`` was given no default for (``None`` is a legitimate default).
_MISSING = object()


class HasRetainedBytes(Protocol):
    """What :class:`SignatureCache` asks of an entry: the bytes its executables pin.

    ``retained`` is the engine's ``retained_bytes_estimate`` of the graph the entry was
    built from, fixed when the entry is made.  A rough estimate, neither an upper nor a
    lower bound, that the cache sums against its byte cap.
    """

    @property
    def retained(self) -> int: ...


@dataclass(frozen=True, slots=True)
class SignatureLimits:
    """The caps a wrapper starts with; ``0`` means no cap on that count.

    ``entries`` is a number of signatures and ``bytes`` an estimate of the memory their
    executables keep pinned, as :func:`signature_limits` read them from the environment.
    """

    entries: int
    bytes: int


def _parse_count(text: str) -> int | None:
    """A whole number of entries; ``None`` for anything else (no sign, no blanks)."""
    if _COUNT.fullmatch(text) is None or len(text) > _MAX_DIGITS:
        return None
    value = int(text)
    return value if value <= _UINT64_MAX else None


def _parse_bytes(text: str) -> int | None:
    """A whole number of bytes, or of KiB / MiB / GiB with a ``k`` / ``m`` / ``g`` suffix.

    The engine's reader of ``LUCID_COMPILE_MAX_CACHE_BYTES`` takes exactly this: a blank or
    a ``-`` is no number, and trailing text is not ignored.
    """
    found = _BYTES.fullmatch(text)
    if found is None or len(found.group(1)) > _MAX_DIGITS:
        return None
    value = int(found.group(1)) << _SHIFT[found.group(2).lower()]
    return value if value <= _UINT64_MAX else None


def _from_environment(
    name: str, parse: Callable[[str], int | None], fallback: int, want: str
) -> int:
    """``name``'s value, or ``fallback`` — said once on stderr when the value is unusable."""
    text = os.environ.get(name)
    if text is None:
        return fallback
    value = parse(text)
    if value is not None:
        return value
    if name not in _REPORTED:
        _REPORTED.add(name)
        print(
            f'lucid: ignoring {name}="{text}" (want {want}; 0 turns the cap off); using {fallback}',
            file=sys.stderr,
        )
    return fallback


def signature_limits() -> SignatureLimits:
    """The caps a new wrapper starts with, read now from the environment.

    ``LUCID_COMPILE_MAX_SIGNATURES`` is a number of entries (default
    :data:`DEFAULT_MAX_ENTRIES`); ``LUCID_COMPILE_MAX_SIGNATURE_BYTES`` a number of
    bytes with an optional ``k`` / ``m`` / ``g`` suffix (default an eighth of the
    machine's memory, the session cache's).  ``0`` turns a cap off.  An unusable value
    is replaced by the default and said once on stderr.
    """
    return SignatureLimits(
        entries=_from_environment(
            ENTRIES_ENV, _parse_count, DEFAULT_MAX_ENTRIES, "a whole number of entries"
        ),
        bytes=_from_environment(
            BYTES_ENV,
            _parse_bytes,
            _C_engine.compile.default_cache_bytes(),
            "a whole number of bytes, optionally with a k, m or g suffix",
        ),
    )


@dataclass(slots=True)
class _Slot[V]:
    """An entry and when it was last used (a tick of the cache's own counter)."""

    value: V
    tick: int


class SignatureCache[K: Hashable, V: HasRetainedBytes](MutableMapping[K, V]):
    """A mapping from call signature to compiled entry that forgets the least recently used.

    Reads like the ``dict`` it replaces — ``len``, ``in``, ``items``, ``pop``,
    ``clear`` — with two differences.  Storing an entry evicts the oldest ones while the
    cache is over its entry cap or its byte cap (:attr:`retained_bytes`, the sum of the
    entries' ``retained``), never the one just stored, however large it is.  And
    :meth:`lookup` is the read that counts as a use: ``get`` and ``[]`` do not, so
    introspection (``cache_info``, ``timing``, a dump) cannot keep an entry alive.

    Safe to share between threads, each call answering as the ``dict`` it replaced would
    (a store that evicts cannot make a ``pop`` or a listing raise): ``pop`` is one step
    under the lock, and ``items`` / ``values`` / iteration are snapshots, not live views.

    Parameters
    ----------
    max_entries, max_bytes : int, optional
        The caps; ``0`` means none.  Taken from :func:`signature_limits` when omitted.
        A wrapper with a cap of its own (``fused_step`` keeps 8 plans) passes it.
    """

    def __init__(self, max_entries: int | None = None, max_bytes: int | None = None) -> None:
        limits = signature_limits()
        self._max_entries = limits.entries if max_entries is None else max_entries
        self._max_bytes = limits.bytes if max_bytes is None else max_bytes
        self._slots: dict[K, _Slot[V]] = {}
        # A plain int, so the cache pickles and copies like the dict it replaced; two
        # lookups racing for a tick may share one, which orders nothing wrongly enough to matter.
        self._clock = 0
        # Mutations only: ``lookup`` is one dict read and one attribute write.
        self._lock = threading.Lock()

    @override
    def __getstate__(self) -> dict[str, object]:
        return {name: value for name, value in self.__dict__.items() if name != "_lock"}

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()

    @property
    def max_entries(self) -> int:
        """The entry cap (``0``: none)."""
        return self._max_entries

    @property
    def max_bytes(self) -> int:
        """The byte cap (``0``: none)."""
        return self._max_bytes

    @property
    def retained_bytes(self) -> int:
        """The estimated bytes the stored entries' executables keep pinned."""
        with self._lock:
            return self._retained()

    def lookup(self, key: K) -> V | None:
        """The entry for ``key`` — now the most recently used — or ``None``."""
        slot = self._slots.get(key)
        if slot is None:
            return None
        self._clock += 1
        slot.tick = self._clock
        return slot.value

    @override
    def __getitem__(self, key: K) -> V:
        return self._slots[key].value

    @override
    def __setitem__(self, key: K, value: V) -> None:
        with self._lock:
            self._clock += 1
            self._slots[key] = _Slot(value, self._clock)
            while len(self._slots) > 1 and self._over_cap():
                # Not by tick alone: a lookup racing for one may share the new entry's.
                oldest = min(
                    (k for k in self._slots if k != key), key=lambda k: self._slots[k].tick
                )
                del self._slots[oldest]
                _narrate(
                    "a wrapper's signature cache is over its cap: dropped its oldest executable"
                )

    @override
    def __delitem__(self, key: K) -> None:
        with self._lock:
            del self._slots[key]

    @overload
    def pop(self, key: K, /) -> V: ...

    @overload
    def pop(self, key: K, /, default: V) -> V: ...

    @overload
    def pop[T](self, key: K, /, default: T) -> V | T: ...

    @override
    def pop(self, key: K, /, default: object = _MISSING) -> object:
        """Remove ``key`` and return its entry, else ``default``; one step, so atomic.

        ``MutableMapping.pop`` reads and then deletes, and a store that evicts ``key``
        in between makes the delete raise.
        """
        with self._lock:
            slot = self._slots.pop(key, None)
        if slot is not None:
            return slot.value
        if default is _MISSING:
            raise KeyError(key)
        return default

    @override
    def __iter__(self) -> Iterator[K]:
        with self._lock:
            return iter(tuple(self._slots))

    @override
    def items(self) -> ItemsView[K, V]:
        """A snapshot of the ``(key, entry)`` pairs: no use, and safe against eviction."""
        return self._snapshot().items()

    @override
    def values(self) -> ValuesView[V]:
        """A snapshot of the entries: no use, and safe against eviction."""
        return self._snapshot().values()

    @override
    def __len__(self) -> int:
        return len(self._slots)

    @override
    def __contains__(self, key: object) -> bool:
        return key in self._slots

    @override
    def clear(self) -> None:
        """Drop every entry."""
        with self._lock:
            self._slots.clear()

    def _snapshot(self) -> dict[K, V]:
        with self._lock:
            return {key: slot.value for key, slot in self._slots.items()}

    def _retained(self) -> int:
        """:attr:`retained_bytes`, for a caller that holds the lock."""
        return sum(slot.value.retained for slot in self._slots.values())

    def _over_cap(self) -> bool:
        if self._max_entries and len(self._slots) > self._max_entries:
            return True
        return bool(self._max_bytes) and self._retained() > self._max_bytes
