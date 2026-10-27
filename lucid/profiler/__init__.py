"""
lucid.profiler: performance profiling utilities.

Wraps the C++ engine's Profiler, OpEvent, and MemoryStats classes.

Usage:
    with lucid.profiler.profile() as prof:
        loss = model(x)
        loss.backward()

    for event in prof.events():
        print(event.name, event.time_ns)
"""

from contextlib import contextmanager
from typing import Iterator, override, TYPE_CHECKING
from lucid._C import engine as _C_engine


class OpEvent:
    r"""A single recorded event captured by the profiler.

    An ``OpEvent`` is one op the engine dispatched, or backward work: a
    whole backward pass (``"backward"``) or one node of it, named after the
    node (``"MulBackward"``, ``"AccumulateGrad"``) — which :attr:`backward`
    tells apart.  Each carries a name, wall-clock duration, shape, an FLOP
    estimate and the net memory delta.  Profilers expose lists of these
    events so that per-op cost can be aggregated, exported, and visualised.

    Parameters
    ----------
    impl : object
        Underlying engine ``OpEvent`` handle returned from the C++
        profiler.

    Notes
    -----
    Durations are reported in nanoseconds at the source; helper
    properties expose the same value in microseconds and milliseconds
    for convenience.

    Examples
    --------
    >>> import lucid
    >>> with lucid.profiler.profile() as prof:
    ...     y = lucid.randn(64, 64) @ lucid.randn(64, 64)
    >>> [ev.name for ev in prof.events()]
    ['randn', 'randn', 'matmul']
    >>> all(ev.time_us >= 0.0 for ev in prof.events())   # timings vary run to run
    True
    """

    def __init__(self, impl: _C_engine.OpEvent) -> None:
        """Initialise the instance.  See the class docstring for parameter semantics."""
        self._impl = impl

    @property
    def name(self) -> str:
        """Operation name (e.g. 'matmul', 'add')."""
        return str(self._impl.name)

    @property
    def time_ns(self) -> int:
        """Wall-clock duration of the operation in nanoseconds."""
        return int(self._impl.time_ns)

    @property
    def time_us(self) -> float:
        """Wall-clock duration in microseconds."""
        return self.time_ns / 1_000.0

    @property
    def time_ms(self) -> float:
        """Wall-clock duration in milliseconds."""
        return self.time_ns / 1_000_000.0

    @property
    def shape(self) -> list[int]:
        """Output tensor shape."""
        return list(self._impl.shape)

    @property
    def flops(self) -> int:
        """Estimated FLOPs for this operation."""
        return int(self._impl.flops)

    @property
    def memory_delta_bytes(self) -> int:
        """Net memory allocated by this operation in bytes."""
        return int(self._impl.memory_delta_bytes)

    @property
    def backward(self) -> bool:
        """Whether the event is backward work — a backward pass or one node of
        it — rather than a dispatched op.  A node's shape is its gradient's
        element count: an eager gradient carries no shape of its own."""
        return bool(self._impl.backward)

    @override
    def __repr__(self) -> str:
        """Return a developer-facing string representation of the instance."""
        return f"OpEvent(name={self.name!r}, time_us={self.time_us:.3f}, " f"shape={self.shape})"


class ProfileSummary:
    r"""Aggregated timing summary for a single op name across many calls.

    Produced by :meth:`Profiler.key_averages`, which buckets the recorded
    :class:`OpEvent` list by op name and exposes total time, average
    time, call count, and cumulative FLOP estimate. This is the natural
    object for a "top-k by self-time" performance report.

    Parameters
    ----------
    name : str
        Op name shared by all events in the summary.
    events : list of OpEvent
        Recorded events bucketed under ``name``.

    Notes
    -----
    The average is computed as

    .. math::

        \bar{t} = \frac{1}{N} \sum_{i=1}^{N} t_i,

    where :math:`N` is the call count and :math:`t_i` is the per-call
    duration in microseconds.

    Examples
    --------
    >>> import lucid
    >>> with lucid.profiler.profile() as prof:
    ...     for _ in range(10):
    ...         y = lucid.randn(64, 64) @ lucid.randn(64, 64)
    >>> sorted((s.name, s.count) for s in prof.key_averages())
    [('matmul', 10), ('randn', 20)]
    """

    def __init__(self, name: str, events: list[OpEvent]) -> None:
        """Initialise the instance.  See the class docstring for parameter semantics."""
        self.name = name
        self._events = events

    @property
    def count(self) -> int:
        """Number of times this operation was called."""
        return len(self._events)

    @property
    def total_time_us(self) -> float:
        """Total time in microseconds across all calls."""
        return sum(e.time_us for e in self._events)

    @property
    def avg_time_us(self) -> float:
        """Average time in microseconds per call."""
        return self.total_time_us / max(1, self.count)

    @property
    def total_flops(self) -> int:
        """Total FLOPs across all calls."""
        return sum(e.flops for e in self._events)

    @override
    def __repr__(self) -> str:
        """Return a developer-facing string representation of the instance."""
        return (
            f"ProfileSummary(name={self.name!r}, count={self.count}, "
            f"avg_us={self.avg_time_us:.3f}, total_flops={self.total_flops})"
        )


class MemoryStats:
    r"""Snapshot of allocator state at a single point in time.

    Wraps the engine's memory tracker so user code can correlate peak
    memory usage with specific phases of training or inference. Combined
    with :class:`Profiler` events, this is sufficient to diagnose memory
    regressions and validate the effect of activation checkpointing.

    Parameters
    ----------
    impl : object
        Underlying engine ``MemoryStats`` handle.

    Notes
    -----
    Reported quantities are byte counts. The ratio
    :math:`\text{peak} / \text{current}` is a coarse indicator of
    fragmentation overhead.

    Examples
    --------
    >>> import lucid
    >>> with lucid.profiler.profile() as prof:
    ...     _ = lucid.randn(1024, 1024)
    >>> stats = prof.memory_stats()
    >>> stats is None or stats.peak_bytes > 0     # byte counts vary per process
    True
    """

    def __init__(self, impl: _C_engine.MemoryStats) -> None:
        """Initialise the instance.  See the class docstring for parameter semantics."""
        self._impl = impl

    @property
    def current_bytes(self) -> int:
        """Currently allocated bytes."""
        return int(self._impl.current_bytes)

    @property
    def peak_bytes(self) -> int:
        """Peak allocated bytes since last reset."""
        return int(self._impl.peak_bytes)

    @property
    def alloc_count(self) -> int:
        """Number of allocations."""
        return int(self._impl.alloc_count)

    @property
    def free_count(self) -> int:
        """Number of deallocations."""
        return int(self._impl.free_count)

    @override
    def __repr__(self) -> str:
        """Return a developer-facing string representation of the instance."""
        return (
            f"MemoryStats(current={self.current_bytes / 1024:.1f}KB, "
            f"peak={self.peak_bytes / 1024:.1f}KB)"
        )


class Profiler:
    r"""Record op-level timing and memory traces from the C++ engine.

    Drives the engine's profiling hook so that every dispatched op
    becomes an :class:`OpEvent`. Suitable for both interactive use
    (``with`` block) and programmatic control via explicit
    :meth:`start` / :meth:`stop`. Recorded events can be exported as a
    Chrome / Perfetto trace for visual inspection.

    Parameters
    ----------
    with_memory : bool, optional
        If ``True`` (default), also capture allocator stats via
        :class:`MemoryStats`.

    Notes
    -----
    Total recorded time can be summarised per op as

    .. math::

        T_{\text{op}} = \sum_{i : \text{name}(e_i) = \text{op}} t_i,

    via :meth:`key_averages`. Only one :class:`Profiler` may be active
    at a time; entering a nested instance overwrites the engine's
    current profiler pointer until the outer context exits.

    Examples
    --------
    >>> import lucid
    >>> with lucid.profiler.profile() as prof:
    ...     y = lucid.randn(64, 64) @ lucid.randn(64, 64)
    >>> top = prof.key_averages()[:5]
    """

    def __init__(self, with_memory: bool = True) -> None:
        """Initialise the instance.  See the class docstring for parameter semantics."""
        self._impl = _C_engine.Profiler()
        self._with_memory = with_memory
        self._active = False

    def start(self) -> None:
        """Start recording operations."""
        _C_engine.set_current_profiler(self._impl)
        self._impl.start()
        self._active = True

    def stop(self) -> None:
        """Stop recording operations."""
        self._impl.stop()
        _C_engine.set_current_profiler(None)
        self._active = False

    def __enter__(self) -> Profiler:
        """Enter the context.  Returns self so the value can be bound via ``with ... as``."""
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        """Exit the context, restoring any state that was modified on entry."""
        self.stop()

    def events(self) -> list[OpEvent]:
        """Return all recorded OpEvents."""
        return [OpEvent(e) for e in self._impl.events]

    def key_averages(self) -> list[ProfileSummary]:
        """Return per-operation-name summaries sorted by total time."""
        from collections import defaultdict

        by_name: dict[str, list[OpEvent]] = defaultdict(list)
        for ev in self.events():
            by_name[ev.name].append(ev)
        summaries = [ProfileSummary(name, evs) for name, evs in by_name.items()]
        return sorted(summaries, key=lambda s: -s.total_time_us)

    def memory_stats(self) -> MemoryStats | None:
        """Return memory stats if available."""
        if not self._with_memory:
            return None
        return MemoryStats(_C_engine.memory_stats(_C_engine.Device.CPU))

    def export_chrome_trace(self, path: str) -> None:
        """Export events as a Chrome trace JSON file.

        Open with chrome://tracing or https://ui.perfetto.dev.
        """
        import json

        trace_events = []
        for i, ev in enumerate(self.events()):
            trace_events.append(
                {
                    "name": ev.name,
                    "ph": "X",  # complete event
                    "ts": i * 100,  # synthetic timestamp (µs)
                    "dur": ev.time_us,
                    "pid": 0,
                    "tid": 0,
                    "args": {"shape": str(ev.shape), "flops": ev.flops},
                }
            )
        with open(path, "w") as f:
            json.dump({"traceEvents": trace_events}, f)

    def clear(self) -> None:
        """Clear all recorded events."""
        self._impl.clear()


@contextmanager
def profile(
    activities: list[str] | None = None,
    with_memory: bool = True,
) -> Iterator[Profiler]:
    r"""Context manager that records ops executed within the ``with`` block.

    The canonical entry point for ad-hoc profiling: enter the context,
    run any Lucid computation, exit, then inspect ``prof.events()`` or
    ``prof.key_averages()``. The ``activities`` parameter is accepted
    for parity with the conventional profiling API but is not
    interpreted.

    Parameters
    ----------
    activities : list of str, optional
        Currently ignored; reserved for future per-device filtering.
    with_memory : bool, optional
        Include memory statistics in the recorded trace. Default
        ``True``.

    Yields
    ------
    Profiler
        Active profiler instance whose events list grows as ops are
        executed inside the block.

    Notes
    -----
    Internally instantiates a :class:`Profiler` and routes its lifecycle
    through ``__enter__`` / ``__exit__``. Equivalent to::

        prof = Profiler(with_memory=with_memory)
        prof.start()
        try: ...
        finally: prof.stop()

    Examples
    --------
    >>> import lucid
    >>> with lucid.profiler.profile() as prof:
    ...     out = lucid.randn(128, 128) @ lucid.randn(128, 128)
    >>> sorted((row.name, row.count, row.total_flops) for row in prof.key_averages())
    [('matmul', 1, 4194304), ('randn', 2, 0)]
    """
    del activities  # reference-signature parity
    prof = Profiler(with_memory=with_memory)
    with prof:
        yield prof


@contextmanager
def record_function(name: str) -> Iterator[None]:
    r"""Annotate an enclosing region with a name visible to the profiler.

    Lets user code carve the trace into semantically meaningful regions
    (e.g. ``"encoder"``, ``"decoder"``, ``"loss"``) alongside the
    automatic per-op events. The hook is a no-op when no profiler is
    active and when the engine does not yet support named regions, so
    its presence is always safe.

    Parameters
    ----------
    name : str
        Label to associate with the enclosed block in the trace.

    Yields
    ------
    None
        The context manager does not bind a value.

    Notes
    -----
    Intended placement is around coarse-grained phases — invoking it on
    every micro-op would inflate trace size without aiding analysis.

    Examples
    --------
    >>> import lucid
    >>> with lucid.profiler.profile() as prof:
    ...     with lucid.profiler.record_function("matmul-region"):
    ...         y = lucid.randn(64, 64) @ lucid.randn(64, 64)
    """
    # If a profiler is active, we can't easily inject named regions without
    # engine support. This is a best-effort no-op that preserves the API.
    del name
    yield


__all__ = [
    "Profiler",
    "OpEvent",
    "ProfileSummary",
    "MemoryStats",
    "profile",
    "record_function",
]
