"""
DataLoader and default_collate.

NumPy is imported lazily — the dataloader is one of the H4 bridge
boundaries (external data ingest), so an ``np.ndarray`` input is
expected, but ``import lucid.utils.data`` itself stays numpy-free.
"""

import itertools
import multiprocessing as _mp
import multiprocessing.queues as _mpq
from multiprocessing.context import BaseContext
from multiprocessing.process import BaseProcess
import pickle
import queue
import random
import time
import traceback
from dataclasses import dataclass
from typing import Callable, Iterator, Protocol, cast, final, override

from lucid._tensor.tensor import Tensor
from lucid._factories.converters import tensor as _tensor_fn
from lucid._factories.random import Generator, manual_seed as _manual_seed
from lucid import stack
from lucid.utils.data.dataset import Dataset, IterableDataset
from lucid.utils.data._rng import _as_generator, _draw_seed
from lucid.utils.data._worker import WorkerInfo, _set_worker_info, get_worker_info
from lucid.utils.data.sampler import (
    Sampler,
    SequentialSampler,
    RandomSampler,
    BatchSampler,
)

# Sentinel pushed to index queues to signal workers to shut down.
_SHUTDOWN = None

# How long a blocked wait runs before it checks that the other side is
# still alive — the main process checks its workers, a worker its parent
# and whether it has been told to stop.
_STATUS_INTERVAL = 1.0
_WORKER_POLL_INTERVAL = 0.25

# How long a shutdown waits for workers to exit before terminating them.
_SHUTDOWN_GRACE = 5.0


# ── collation ─────────────────────────────────────────────────────────────────


# dtype kinds that become a Tensor: bool, signed / unsigned int, float, complex.
# String, bytes, object, datetime and void arrays stay as they are.
_NUMERIC_KINDS = "biufc"


def _is_numpy_numeric(obj: object) -> bool:
    """True for a numeric NumPy array or NumPy scalar (``np.int64(3)``).

    Decided from the type's module and the value's ``dtype`` alone, so a
    batch of pure Lucid tensors, Python scalars or strings never triggers
    a numpy import.  ``np.float64`` subclasses ``float`` and must be
    caught here, before the Python-scalar branch claims it.
    """
    if type(obj).__module__ != "numpy":
        return False
    kind = getattr(getattr(obj, "dtype", None), "kind", None)
    return isinstance(kind, str) and kind in _NUMERIC_KINDS


def default_convert(data: object) -> object:
    r"""Recursively convert a single sample's elements to Lucid Tensors.

    The inverse-flavour partner of :func:`default_collate`:
    :func:`default_collate` stacks a *batch list* into batched tensors,
    while :func:`default_convert` walks a *single sample* (possibly
    nested in lists / tuples / dicts / named tuples) and turns leaf
    ndarrays / Python scalars into :class:`Tensor` leaves.  Existing
    Tensors / strings / bytes pass through unchanged.

    Parameters
    ----------
    data : object
        A sample, possibly nested.  Supported leaf types are
        :class:`Tensor`, ``np.ndarray``, NumPy scalars (``np.int64``,
        ``np.float32``, ...), ``int``, ``float``, ``bool``, ``str``, and
        ``bytes``.  Container types ``dict``, ``list``,
        ``tuple``, and ``NamedTuple`` are walked recursively and their
        original type is preserved.

    Returns
    -------
    object
        Same nested structure as ``data`` with leaf ndarrays / scalars
        promoted to :class:`Tensor`.  A NumPy scalar becomes a 0-d
        :class:`Tensor` of its own dtype.

    Notes
    -----
    Used by :class:`DataLoader` when a dataset returns numpy arrays per
    sample but the user wants Tensor leaves *before* collation.  Numpy
    is imported lazily — this is one of the H4 bridge boundaries
    (external data ingest).

    Examples
    --------
    >>> default_convert({"x": 1.5, "y": [1, 2]})
    {'x': tensor(1.5), 'y': [tensor(1, dtype=lucid.int64), tensor(2, dtype=lucid.int64)]}
    """
    if isinstance(data, Tensor):
        return data
    if _is_numpy_numeric(data):
        import numpy as np

        # ``asarray`` turns a numpy scalar into a 0-d array, which keeps its
        # dtype and its zero rank; the bare scalar converts to neither.
        return _tensor_fn(np.asarray(data))
    if isinstance(data, (str, bytes)):
        return data
    if isinstance(data, dict):
        return {k: default_convert(v) for k, v in data.items()}
    if isinstance(data, tuple) and hasattr(data, "_fields"):
        return type(data)(*(default_convert(v) for v in data))  # type: ignore[arg-type]
    if isinstance(data, (list, tuple)):
        converted = [default_convert(v) for v in data]
        return type(data)(converted) if isinstance(data, tuple) else converted
    if isinstance(data, (int, float, bool)):
        return _tensor_fn(data)
    return data


def collate(
    batch: list[object],
    *,
    collate_fn_map: dict[type, Callable[..., object]] | None = None,
) -> Tensor | list[object] | dict[str, object] | tuple[object, ...]:
    r"""Composable collate dispatcher with user-overridable type handlers.

    Same dispatch behaviour as :func:`default_collate` plus an optional
    ``collate_fn_map`` that overrides handling for specific element
    types.  Use this when a custom record type needs special-case
    batching while everything else (tensors, dicts, namedtuples, ...)
    should follow the default rules.

    Parameters
    ----------
    batch : list of object
        Samples emitted by a dataset for a single mini-batch.
    collate_fn_map : dict, optional
        Mapping from element type to a callable
        ``fn(batch, *, collate_fn_map=...)``.  The first
        ``isinstance(batch[0], t)`` match wins; on miss the function
        falls back to :func:`default_collate`'s dispatch chain.

    Returns
    -------
    Tensor or list or dict or tuple
        Collated batch.  Structure mirrors ``default_collate``'s output.

    Notes
    -----
    When ``collate_fn_map is None`` this is exactly
    :func:`default_collate`.  The recursive forwarding of
    ``collate_fn_map`` lets nested containers reuse the same overrides.

    Examples
    --------
    >>> import lucid
    >>> from dataclasses import dataclass
    >>> from lucid.utils.data import DataLoader
    >>> from lucid.utils.data.dataloader import collate
    >>> @dataclass
    ... class MyRecord:
    ...     text: str
    ...     score: float
    >>> def my_record_collate(batch, *, collate_fn_map=None):
    ...     return [r.text for r in batch], lucid.tensor([r.score for r in batch])
    >>> ds = [MyRecord("a", 0.5), MyRecord("b", 1.5), MyRecord("c", 2.5)]
    >>> loader = DataLoader(ds, batch_size=3, collate_fn=lambda b: collate(
    ...     b, collate_fn_map={MyRecord: my_record_collate}))
    >>> texts, scores = next(iter(loader))
    >>> texts, scores.tolist()
    (['a', 'b', 'c'], [0.5, 1.5, 2.5])
    """
    elem = batch[0]
    if collate_fn_map is not None:
        for t, fn in collate_fn_map.items():
            if isinstance(elem, t):
                return fn(batch, collate_fn_map=collate_fn_map)  # type: ignore[return-value]
    return default_collate(batch)


def default_collate(
    batch: list[object],
) -> Tensor | list[object] | dict[str, object] | tuple[object, ...]:
    r"""Collate a list of samples into a batched tensor or nested structure.

    The default ``collate_fn`` used by :class:`DataLoader`.  Walks the
    first element of ``batch`` to decide how to combine the remaining
    elements, preserving the original nested container structure.

    Parameters
    ----------
    batch : list of object
        Samples (one per dataset item) to combine into a single batch.
        Every element must share the same structure / type as
        ``batch[0]``.

    Returns
    -------
    Tensor or list or dict or tuple
        * :class:`Tensor` leaves → stacked along a new leading axis.
        * ``np.ndarray`` leaves → stacked then wrapped as :class:`Tensor`
          (numpy bridge — :class:`DataLoader` is an H4 carve-out).
        * NumPy scalar leaves (``np.int64``, ``np.float32``, ...) → 1-D
          :class:`Tensor` of their dtype.
        * ``int`` / ``float`` leaves → 1-D :class:`Tensor`.
        * ``str`` / ``bytes`` leaves → kept as a Python list.
        * ``dict`` / ``list`` / ``tuple`` / ``NamedTuple`` containers
          → walked recursively; original container type is preserved.

    Notes
    -----
    The recursion is structural: a batch of ``{"x": Tensor, "y": int}``
    becomes ``{"x": stacked_Tensor, "y": 1d_Tensor}`` of the same dict
    shape.  Heterogeneous batches (different keys / shapes) are not
    supported — callers must normalise upstream.

    Examples
    --------
    >>> import lucid
    >>> from lucid.utils.data.dataloader import default_collate
    >>> t1, t2, t3 = lucid.zeros(4), lucid.ones(4), lucid.ones(4)
    >>> xs, ys = default_collate([(t1, 0), (t2, 1), (t3, 2)])
    >>> xs.shape, ys.tolist()
    ((3, 4), [0, 1, 2])
    """
    elem = batch[0]

    if isinstance(elem, Tensor):
        return stack(batch, 0)  # type: ignore[arg-type]

    if _is_numpy_numeric(elem):
        # User opted into a numpy bridge by handing us an ndarray or a
        # numpy scalar (``y[i]`` on a label array); stacking keeps its dtype.
        import numpy as np

        return Tensor(np.stack(batch, axis=0))  # type: ignore[arg-type]

    if isinstance(elem, (int, float)):
        # Build a 1-D Tensor from a Python list — no numpy stack needed.
        return _tensor_fn(list(batch))

    if isinstance(elem, (str, bytes)):
        return batch

    if isinstance(elem, dict):
        return {key: default_collate([d[key] for d in batch]) for key in elem}

    if isinstance(elem, tuple) and hasattr(elem, "_fields"):
        return type(elem)(*(default_collate([d[i] for d in batch]) for i in range(len(elem))))

    if isinstance(elem, (list, tuple)):
        collated = [default_collate([d[i] for d in batch]) for i in range(len(elem))]
        return type(elem)(collated) if isinstance(elem, tuple) else collated  # type: ignore[return-value]

    return batch


# ── fetching ──────────────────────────────────────────────────────────────────
# The same fetchers run in the main process (``num_workers=0``) and in every
# worker, so a loader yields the same batches whatever ``num_workers`` is.
#
# A fetcher raises ``StopIteration`` for one reason only: an iterable
# dataset's stream is spent.  The same exception escaping user code — a
# ``__getitem__``, a ``collate_fn`` — would otherwise end the epoch early
# and silently, or (in a worker) retire the worker with its share of the
# data, so it becomes a ``RuntimeError`` here, as PEP 479 does for
# generators.


def _escaped_stop(source: str) -> RuntimeError:
    return RuntimeError(
        f"{source} raised StopIteration; a DataLoader would read that as the "
        "end of the data and drop the rest of the epoch without a word."
    )


@dataclass(frozen=True, slots=True)
class _FetchSpec:
    """What a fetcher needs from its loader; small and picklable for workers."""

    iterable: bool
    auto_collation: bool
    collate_fn: Callable[..., object]
    batch_size: int
    drop_last: bool
    use_getitems: bool

    def make(self, dataset: object) -> _MapFetcher | _IterableFetcher:
        if self.iterable:
            return _IterableFetcher(dataset, self)
        return _MapFetcher(dataset, self)


@final
class _MapFetcher:
    """Turn one entry of the index sampler into a batch (or a sample)."""

    def __init__(self, dataset: object, spec: _FetchSpec) -> None:
        self._dataset = cast(Dataset, dataset)
        self._auto = spec.auto_collation
        self._collate_fn = spec.collate_fn
        # 3.2.0: the optional vectorised batch-fetch protocol.  A dataset
        # that implements ``__getitems__(indices) -> already-batched`` owns
        # its own collation — but only when the caller did not ask for a
        # different one.  The fast path *is* a collation, so taking it with
        # a user-supplied ``collate_fn`` in hand would silently discard it.
        self._getitems: Callable[[list[int]], object] | None = (
            getattr(dataset, "__getitems__", None) if spec.use_getitems else None
        )

    def fetch(self, index: object) -> object:
        try:
            if not self._auto:
                # ``batch_size=None``: one sample per step, through
                # ``collate_fn`` (``default_convert`` unless overridden).
                return self._collate_fn(self._dataset[cast(int, index)])
            indices = cast(list[int], index)
            if self._getitems is not None:
                return self._getitems(indices)
            return self._collate_fn([self._dataset[i] for i in indices])
        except StopIteration as exc:
            raise _escaped_stop("the dataset or collate_fn") from exc


@final
class _IterableFetcher:
    """Pull the next batch (or sample) off an iterable dataset's iterator.

    Raises ``StopIteration`` once the stream is spent — and keeps raising
    it, without touching the iterator again, so a stream that would
    restart on a further ``next`` cannot leak a second pass.
    """

    def __init__(self, dataset: object, spec: _FetchSpec) -> None:
        self._iter: Iterator[object] = iter(cast(IterableDataset, dataset))
        self._auto = spec.auto_collation
        self._collate_fn = spec.collate_fn
        self._batch_size = spec.batch_size
        self._drop_last = spec.drop_last
        self._ended = False

    def fetch(self, index: object = None) -> object:
        del index  # _BaseDatasetFetcher.fetch(index) interface parity
        if self._ended:
            raise StopIteration
        if not self._auto:
            try:
                item: object = next(self._iter)
            except StopIteration:
                self._ended = True
                raise
            return self._collate(item)
        batch: list[object] = []
        while len(batch) < self._batch_size:
            try:
                batch.append(next(self._iter))
            except StopIteration:
                self._ended = True
                break
        if not batch or (self._drop_last and len(batch) < self._batch_size):
            self._ended = True
            raise StopIteration
        return self._collate(batch)

    def _collate(self, data: object) -> object:
        try:
            return self._collate_fn(data)
        except StopIteration as exc:
            raise _escaped_stop("collate_fn") from exc


# ── worker ⇄ main-process messages ────────────────────────────────────────────


class _KeyErrorMessage(str):
    """A ``KeyError`` prints its argument with ``repr``, which would fold a
    multi-line worker traceback into one escaped line."""

    @override
    def __repr__(self) -> str:
        return str(self)


class _WorkerError:
    """An exception raised in a worker, in a form that survives the pipe.

    The exception object itself may not pickle (or may pickle without its
    traceback), so the worker sends its type and the formatted traceback,
    and the main process raises a fresh exception of the same type whose
    message carries the worker's traceback.  A type that cannot be sent
    or rebuilt from one message falls back to ``RuntimeError``.
    """

    def __init__(self, exc: BaseException, where: str) -> None:
        try:
            self.type_bytes: bytes | None = pickle.dumps(type(exc))
        except pickle.PicklingError:  # a local class, say
            self.type_bytes = None
        trace = "".join(traceback.format_exception(exc))
        self.message = f"Caught {type(exc).__name__} {where}.\nOriginal {trace}"

    def to_exception(self) -> BaseException:
        """The exception to raise in the main process for this one."""
        exc_type: object = None
        if self.type_bytes is not None:
            try:
                exc_type = pickle.loads(self.type_bytes)
            except (ImportError, AttributeError, pickle.UnpicklingError):
                exc_type = None  # its module or class is not importable here
        message: str = self.message
        if exc_type is KeyError:
            message = _KeyErrorMessage(message)
        if isinstance(exc_type, type) and issubclass(exc_type, StopIteration):
            # Raised from ``__next__`` it would end the epoch, not report.
            exc_type = RuntimeError
        if isinstance(exc_type, type) and issubclass(exc_type, BaseException):
            try:
                return exc_type(message)
            except TypeError:
                return RuntimeError(message)  # needs more than a message
        return RuntimeError(message)


class _IterableEnd:
    """A worker's reply once its copy of an iterable dataset is spent."""


class _ResultQueue(_mpq.Queue):  # type: ignore[type-arg]
    """The shared result queue, reporting a batch it cannot send.

    ``Queue.put`` pickles in a background feeder thread.  When a batch does
    not pickle, the stock queue prints the error and drops the batch, and
    the main process then waits for it forever.  This sends the error in
    the batch's place — through the hook ``concurrent.futures`` overrides
    for the same reason.
    """

    def _on_queue_feeder_error(self, e: Exception, obj: object) -> None:
        if isinstance(obj, tuple) and len(obj) == 3 and not isinstance(obj[2], _WorkerError):
            info = get_worker_info()
            worker = info.id if info is not None else "?"
            where = f"while sending a batch from DataLoader worker process {worker}"
            self.put((obj[0], obj[1], _WorkerError(e, where)))
        else:
            traceback.print_exception(e)  # the stock queue's behaviour


class _Flag(Protocol):
    def is_set(self) -> bool: ...


class _SharedInt(Protocol):
    value: int


# ── worker process entry point ────────────────────────────────────────────────
# Must be a top-level function so `spawn` can pickle it.


def _init_worker(
    worker_id: int,
    num_workers: int,
    dataset: object,
    worker_init_fn: Callable[[int], object] | None,
    base_seed: int,
) -> _WorkerError | None:
    """Everything a worker sets up before its first task, in this order.

    The one place per-worker randomness is decided: every RNG a dataset
    or transform may draw from — Lucid's generator (and through it every
    ``lucid.utils.transforms`` augmentation), Python's ``random``, NumPy's
    legacy global RNG — is seeded with ``base_seed + worker_id``.  The
    main process draws ``base_seed`` once per iterator, so the streams
    differ between workers and between epochs, and reproduce under
    :func:`lucid.manual_seed` or the loader's ``generator``.  Then
    :func:`get_worker_info` is published, and only then does
    ``worker_init_fn`` run, so it can read both.

    Returns the ``worker_init_fn`` failure, to be reported for every task.
    """
    seed = base_seed + worker_id
    random.seed(seed)
    _manual_seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ImportError:
        pass
    _set_worker_info(
        WorkerInfo(
            id=worker_id,
            num_workers=num_workers,
            seed=seed,
            dataset=cast(Dataset, dataset),
        )
    )
    try:
        if worker_init_fn is not None:
            worker_init_fn(worker_id)
    except Exception as exc:  # noqa: BLE001 — reported for every task
        return _WorkerError(exc, f"in DataLoader worker process {worker_id}")
    return None


def _worker_loop(
    worker_id: int,
    num_workers: int,
    dataset: object,
    index_queue: _mpq.Queue[object],
    result_queue: _ResultQueue,
    done_event: _Flag,
    current_epoch: _SharedInt,
    spec: _FetchSpec,
    worker_init_fn: Callable[[int], object] | None,
    base_seed: int,
) -> None:
    """Worker process: pull tasks, fetch data, push ``(epoch, seq, result)``."""
    init_error = _init_worker(worker_id, num_workers, dataset, worker_init_fn, base_seed)
    parent = _mp.parent_process()
    fetcher: _MapFetcher | _IterableFetcher | None = None
    fetcher_epoch = -1
    orphaned = False
    try:
        while True:
            try:
                msg = index_queue.get(timeout=_WORKER_POLL_INTERVAL)
            except queue.Empty:
                # An orphaned worker would wait here forever.  And the
                # ``_SHUTDOWN`` message can be lost: when the iterator is
                # freed by the garbage collector, its queues' own
                # finalizers may already have stopped their feeder threads,
                # so the stop event is the signal that always arrives.
                if done_event.is_set():
                    break
                if parent is not None and not parent.is_alive():
                    orphaned = True
                    break
                continue
            if msg is _SHUTDOWN:
                break
            epoch, seq, index = cast(tuple[int, int, object], msg)
            # Shutting down, or a task the main process has abandoned (the
            # epoch it belonged to was broken out of): its result would
            # only be thrown away.
            if done_event.is_set() or epoch != current_epoch.value:
                continue
            data: object
            if init_error is not None:
                data = init_error
            else:
                try:
                    if fetcher is None or fetcher_epoch != epoch:
                        # An iterable dataset restarts its stream each epoch.
                        fetcher = spec.make(dataset)
                        fetcher_epoch = epoch
                    data = fetcher.fetch(index)
                except StopIteration as exc:
                    data = (
                        _IterableEnd()
                        if spec.iterable
                        else _WorkerError(exc, f"in DataLoader worker process {worker_id}")
                    )
                except Exception as exc:  # noqa: BLE001 — re-raised in main
                    data = _WorkerError(exc, f"in DataLoader worker process {worker_id}")
            result_queue.put((epoch, seq, data))
            del data
    except KeyboardInterrupt:
        # Ctrl-C reaches the whole process group; the main process reports it.
        pass
    if orphaned or done_event.is_set():
        # Nobody will read what is still buffered.  Flushing it would block
        # the exit for good: the feeder thread waits in ``send_bytes`` on a
        # full pipe whose read end every worker still holds open.
        result_queue.cancel_join_thread()
    result_queue.close()


# ── single-process iterator ───────────────────────────────────────────────────


@final
class _SingleProcessDataLoaderIter:
    def __init__(self, loader: DataLoader) -> None:
        spec = loader._fetch_spec()
        self._fetcher = spec.make(loader.dataset)
        self._index_iter: Iterator[object] = (
            itertools.repeat(None)
            if loader._iterable_style
            else iter(cast(Sampler, loader._index_sampler))
        )

    def __iter__(self) -> _SingleProcessDataLoaderIter:
        return self

    def __next__(self) -> Tensor | tuple[Tensor, ...]:
        index = next(self._index_iter)
        return cast(Tensor | tuple[Tensor, ...], self._fetcher.fetch(index))


# ── multi-process iterator ────────────────────────────────────────────────────


@final
class _MultiProcessDataLoaderIter:
    """Multi-worker iterator with prefetching and ordered delivery.

    Design:
    - Each worker owns one index queue; tasks go round-robin to the
      workers that still have data (an iterable dataset's copy may run
      out on one worker before another).
    - Workers push ``(epoch, seq, result)`` onto one shared result queue;
      the main process reorders by ``seq`` and yields in sampler order.
    - At most ``num_workers * prefetch_factor`` tasks are in flight.
    - Every task and result carries the epoch it belongs to.  A
      persistent pool reused after a ``break`` skips, and the main process
      discards, whatever the abandoned epoch still had in flight — so the
      next epoch neither repeats nor loses samples.
    - A blocked wait wakes every ``_STATUS_INTERVAL`` seconds to check the
      workers: one that died raises instead of hanging the loader.
    - An exception in a worker is re-raised here with the worker's
      traceback; the pool survives it, so the next epoch runs.
    """

    def __init__(self, loader: DataLoader, base_seed: int) -> None:
        # Nothing to shut down until the pool exists (``__del__`` runs even
        # when this constructor fails part-way).
        self._shutdown = True
        self._num_workers: int = loader.num_workers
        self._window: int = loader.num_workers * (loader.prefetch_factor or 2)
        self._persistent: bool = loader.persistent_workers
        self._timeout: float = loader.timeout
        self._iterable: bool = loader._iterable_style

        ctx = _mp_context(loader.multiprocessing_context)
        self._epoch = 0
        self._result_queue = _ResultQueue(ctx=ctx)
        self._done_event = ctx.Event()
        self._current_epoch: _SharedInt = ctx.Value("q", 0, lock=False)
        self._index_queues: list[_mpq.Queue[object]] = []
        self._workers: list[BaseProcess] = []
        self._shutdown = False
        spec = loader._fetch_spec()
        try:
            for wid in range(self._num_workers):
                index_queue: _mpq.Queue[object] = ctx.Queue()
                # A task still buffered when the main process exits is moot.
                index_queue.cancel_join_thread()
                # Every concrete context defines ``Process``; the base class
                # the type stubs describe does not.
                worker = ctx.Process(  # type: ignore[attr-defined]
                    target=_worker_loop,
                    args=(
                        wid,
                        self._num_workers,
                        loader.dataset,
                        index_queue,
                        self._result_queue,
                        self._done_event,
                        self._current_epoch,
                        spec,
                        loader.worker_init_fn,
                        base_seed,
                    ),
                    daemon=True,
                )
                worker.start()
                self._index_queues.append(index_queue)
                self._workers.append(worker)
        except BaseException:
            self._shutdown_workers()
            raise
        self._begin_epoch(loader)

    # ── epoch bookkeeping ─────────────────────────────────────────────────────

    def _begin_epoch(self, loader: DataLoader) -> None:
        self._index_iter: Iterator[object] = (
            itertools.repeat(None) if self._iterable else iter(cast(Sampler, loader._index_sampler))
        )
        self._index_exhausted = False
        self._send_idx = 0  # next task to dispatch
        self._rcvd_idx = 0  # next task to yield
        self._task_worker: dict[int, int] = {}  # dispatched, not yet received
        self._reorder: dict[int, object] = {}  # received, not yet yielded
        self._active = [True] * self._num_workers
        self._worker_cycle = itertools.cycle(range(self._num_workers))
        self._fill()

    def _reset(self, loader: DataLoader) -> None:
        """Start the next epoch on the same (persistent) pool."""
        self._epoch += 1
        # Before the first new task is sent, so a worker that reads one
        # already sees the new epoch and skips the old epoch's leftovers.
        self._current_epoch.value = self._epoch
        self._begin_epoch(loader)

    def _dispatch(self) -> bool:
        if self._index_exhausted:
            return False
        for _ in range(self._num_workers):
            worker_id = next(self._worker_cycle)
            if self._active[worker_id]:
                break
        else:
            return False  # every worker's stream is spent
        try:
            index = next(self._index_iter)
        except StopIteration:
            self._index_exhausted = True
            return False
        self._index_queues[worker_id].put((self._epoch, self._send_idx, index))
        self._task_worker[self._send_idx] = worker_id
        self._send_idx += 1
        return True

    def _fill(self) -> None:
        while self._send_idx - self._rcvd_idx < self._window and self._dispatch():
            pass

    # ── receiving ─────────────────────────────────────────────────────────────

    def _check_workers(self) -> None:
        dead = [w for w in self._workers if not w.is_alive()]
        if not dead:
            return
        which = ", ".join(f"pid {w.pid} (exit code {w.exitcode})" for w in dead)
        self._shutdown_workers()
        raise RuntimeError(
            f"DataLoader worker(s) {which} exited unexpectedly.  A negative "
            "exit code is the signal that ended it — -9 is usually the "
            "system running out of memory."
        )

    def _get(self) -> tuple[int, int, object]:
        deadline = None if self._timeout <= 0 else time.monotonic() + self._timeout
        while True:
            wait = _STATUS_INTERVAL
            if deadline is not None:
                wait = min(wait, max(deadline - time.monotonic(), 0.0))
            try:
                return cast(tuple[int, int, object], self._result_queue.get(timeout=wait))
            except queue.Empty:
                pass
            except Exception:
                # A non-timeout failure (a result that does not unpickle
                # here, a closed queue) — shut the pool down, then surface
                # the real error.
                self._shutdown_workers()
                raise
            self._check_workers()
            if deadline is not None and time.monotonic() >= deadline:
                self._shutdown_workers()
                raise RuntimeError(
                    f"DataLoader timed out after {self._timeout} seconds waiting "
                    "for a batch from its workers.  Increase timeout or reduce "
                    "batch size."
                )

    def _receive(self) -> None:
        epoch, seq, data = self._get()
        if epoch != self._epoch:
            return  # a batch from an epoch that was broken out of
        worker_id = self._task_worker.pop(seq, None)
        if isinstance(data, _IterableEnd) and worker_id is not None:
            self._active[worker_id] = False
        self._reorder[seq] = data

    # ── shutdown ──────────────────────────────────────────────────────────────

    def _shutdown_workers(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        try:
            # Workers poll this between tasks, so it stops them even when
            # the ``_SHUTDOWN`` message below never reaches them.
            self._done_event.set()
            for index_queue in self._index_queues:
                try:
                    index_queue.put(_SHUTDOWN)
                except (ValueError, RuntimeError):
                    # Already closed, or at interpreter exit where its feeder
                    # thread cannot start; the stop event reaches the worker.
                    continue
            deadline = time.monotonic() + _SHUTDOWN_GRACE
            for w in self._workers:
                w.join(timeout=max(deadline - time.monotonic(), 0.0))
        finally:
            # Whatever interrupted the graceful path (Ctrl-C included), no
            # worker outlives the shutdown.
            for w in self._workers:
                if w.is_alive():
                    w.terminate()
                    w.join(timeout=1.0)
                if w.is_alive():
                    w.kill()
                    w.join(timeout=1.0)
            for index_queue in self._index_queues:
                index_queue.close()
            self._result_queue.close()

    # ── iteration ─────────────────────────────────────────────────────────────

    def __iter__(self) -> _MultiProcessDataLoaderIter:
        return self

    def __next__(self) -> Tensor | tuple[Tensor, ...]:
        if self._shutdown:
            raise StopIteration
        while True:
            self._fill()
            if self._rcvd_idx == self._send_idx:
                # Nothing in flight and nothing left to send.
                if not self._persistent:
                    self._shutdown_workers()
                raise StopIteration
            if self._rcvd_idx not in self._reorder:
                self._receive()
                continue
            data = self._reorder.pop(self._rcvd_idx)
            self._rcvd_idx += 1
            if isinstance(data, _IterableEnd):
                continue
            if isinstance(data, _WorkerError):
                # The pool survives: a caller that catches this may go on
                # to the next batch, and a persistent loader to the next
                # epoch.
                raise data.to_exception()
            return cast(Tensor | tuple[Tensor, ...], data)

    def __del__(self) -> None:
        try:
            self._shutdown_workers()
        except (OSError, ValueError, RuntimeError):
            # Pipes or process handles already released, or interpreter
            # exit; anything else reaches ``sys.unraisablehook``.
            return


# ── argument checks ───────────────────────────────────────────────────────────


def _require_count(name: str, value: object, *, minimum: int, optional: bool = False) -> None:
    """``value`` is an ``int`` (not a ``bool``) ``>= minimum`` — or ``None``."""
    if optional and value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        allowed = f"an integer >= {minimum}" + (" or None" if optional else "")
        raise ValueError(f"{name} must be {allowed}, got {value!r}")


def _mp_context(context: object) -> BaseContext:
    """The worker start context: ``None`` → ``spawn``, a start-method name, or
    a ``multiprocessing`` context object.

    ``spawn`` by default: ``fork`` is unsafe on macOS once Metal, Accelerate
    or any other thread-owning framework is loaded.
    """
    if context is None:
        return _mp.get_context("spawn")
    if isinstance(context, str):
        return _mp.get_context(context)  # ValueError for an unknown method
    if isinstance(context, BaseContext):
        return context
    raise TypeError(
        "multiprocessing_context must be a start-method name or a "
        f"multiprocessing context, got {type(context).__name__}"
    )


# ── DataLoader ────────────────────────────────────────────────────────────────


class DataLoader:
    r"""Combine a dataset with a sampler to provide iteration over mini-batches.

    Wraps a :class:`Dataset` to provide batching, optional shuffling,
    parallel data loading via worker processes, and customisable
    collation.  Iteration yields one collated batch per step until the
    underlying sampler is exhausted.

    Parameters
    ----------
    dataset : Dataset
        Dataset to load data from.  May be either map-style
        (:class:`Dataset`) or iterable-style (:class:`IterableDataset`).
    batch_size : int or None, default=1
        Number of samples per batch.  Ignored when ``batch_sampler`` is
        provided.  ``None`` turns automatic batching off: each step
        yields one sample, passed through ``collate_fn``
        (:func:`default_convert` by default).
    shuffle : bool, optional
        If ``True``, the default sampler is :class:`RandomSampler`;
        otherwise :class:`SequentialSampler`.  Mutually exclusive with
        ``sampler``.
    sampler : Sampler, optional
        Custom per-sample index sampler.  Mutually exclusive with
        ``shuffle``.
    batch_sampler : Sampler, optional
        Custom batch sampler yielding lists of indices.  Mutually
        exclusive with ``batch_size`` / ``shuffle`` / ``sampler`` /
        ``drop_last``.
    num_workers : int, default=0
        Worker processes for parallel data loading.  ``0`` runs
        single-process in the main thread; ``> 0`` spawns a worker pool.
        With an :class:`IterableDataset` every worker iterates its own
        copy of the dataset — shard it with :func:`get_worker_info`, or
        each sample arrives ``num_workers`` times.
    collate_fn : callable, optional
        Merge a list of samples into a batch (default:
        :func:`default_collate`, or :func:`default_convert` when
        ``batch_size=None``).
    drop_last : bool, default=False
        If ``True``, drop the trailing batch when the dataset length is
        not divisible by ``batch_size``.
    timeout : float, default=0.0
        Seconds to wait for a worker to deliver a batch before raising
        ``RuntimeError``.  ``0`` waits indefinitely — for a live worker:
        a worker that dies raises either way.
    worker_init_fn : callable, optional
        Called as ``worker_init_fn(worker_id)`` in each worker process,
        after the worker's random generators are seeded and
        :func:`get_worker_info` is available.
    multiprocessing_context : str or context, optional
        Start method (or context) for the workers; default ``"spawn"``.
    generator : lucid.Generator or int, optional
        Generator the loader's randomness is drawn from: the default
        :class:`RandomSampler`'s order and the workers' base seed.
        ``None`` (default) uses the global generator, so
        :func:`lucid.manual_seed` reproduces both.
    prefetch_factor : int, optional
        Batches pre-loaded per worker (default ``2`` when
        ``num_workers > 0``).  Higher values trade memory for throughput.
    persistent_workers : bool, default=False
        Keep worker processes alive between epochs to avoid repeated
        process-startup overhead.  Requires ``num_workers > 0``.
    pin_memory : bool, default=False
        Accepted for compatibility with code written for discrete-memory
        accelerators; a no-op here, since Apple Silicon's CPU and GPU
        share one unified memory.

    Notes
    -----
    Worker processes communicate via ``multiprocessing.Queue``: each
    worker owns one index queue, all workers share a single result
    queue, and the main process reorders results back into sampler
    order before yielding.  Sequence numbers ensure deterministic
    delivery regardless of completion order across workers.

    **Randomness in workers.**  Each iterator draws one ``base_seed``
    from ``generator`` (or the global generator), and worker ``i`` seeds
    Lucid's generator, Python's :mod:`random` and — when installed —
    NumPy's legacy global RNG with ``base_seed + i`` before
    ``worker_init_fn`` runs.  Random augmentations therefore differ
    between workers and between epochs, and the whole run reproduces
    under :func:`lucid.manual_seed`.  A persistent pool is seeded once
    and its streams carry on from epoch to epoch.

    An exception raised in a worker is re-raised in the main process, as
    the same exception type where possible, with the worker's traceback
    in its message.

    Examples
    --------
    >>> import lucid
    >>> from lucid.utils.data import DataLoader, TensorDataset
    >>> dataset = TensorDataset(lucid.randn(100, 8), lucid.randint(0, 10, (100,)))
    >>> dl = DataLoader(dataset, batch_size=32, shuffle=True)  # num_workers=4 to parallelise
    >>> for batch in dl:
    ...     x, y = batch
    >>> x.shape, y.shape                     # the last, short batch: 100 = 3 * 32 + 4
    ((4, 8), (4,))

    One sample per step, without automatic batching:

    >>> x, y = next(iter(DataLoader(dataset, batch_size=None)))
    >>> x.shape, y.shape
    ((8,), ())
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int | None = 1,
        shuffle: bool | None = None,
        sampler: Sampler | None = None,
        batch_sampler: Sampler | None = None,
        num_workers: int = 0,
        collate_fn: Callable[..., object] | None = None,
        drop_last: bool = False,
        timeout: float = 0.0,
        worker_init_fn: Callable[..., object] | None = None,
        multiprocessing_context: object = None,
        generator: Generator | int | None = None,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        *,
        pin_memory: bool = False,
    ) -> None:
        """Configure a ``DataLoader``; see the class docstring for parameter
        semantics.

        Notes
        -----
        ``sampler`` / ``batch_sampler`` / ``shuffle`` / ``batch_size`` /
        ``drop_last`` interact: passing ``batch_sampler`` precludes the
        other four; passing ``sampler`` precludes ``shuffle``. When no
        sampler is supplied, a :class:`SequentialSampler` (``shuffle=False``)
        or :class:`RandomSampler` (``shuffle=True``) is constructed
        automatically. ``persistent_workers`` requires ``num_workers > 0``;
        ``batch_size=None`` precludes ``drop_last``.

        Raises
        ------
        ValueError
            On any of the above mutual-exclusion violations; a
            ``batch_size`` or ``prefetch_factor`` that is not a positive
            integer (or ``None``); a ``num_workers`` that is not an integer
            ``>= 0``; a negative or NaN ``timeout``; ``prefetch_factor``,
            ``multiprocessing_context`` or ``persistent_workers`` given
            with ``num_workers=0``; or an unknown start-method name.
        TypeError
            If ``timeout`` is not a number, ``collate_fn`` or
            ``worker_init_fn`` is not callable, ``multiprocessing_context``
            is neither a name nor a context, or ``generator`` is neither a
            :class:`lucid.Generator`, a seed, nor ``None`` (a ``bool`` is
            not a seed).
        """
        # The boundary: every argument is checked here, with an error naming
        # it, before anything is stored or a worker exists.
        _require_count("batch_size", batch_size, minimum=1, optional=True)
        _require_count("num_workers", num_workers, minimum=0)
        _require_count("prefetch_factor", prefetch_factor, minimum=1, optional=True)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise TypeError(f"timeout must be a number of seconds, got {type(timeout).__name__}")
        if not timeout >= 0:  # NaN included
            raise ValueError(f"timeout must be >= 0, got {timeout}")
        for name, fn in (
            ("collate_fn", collate_fn),
            ("worker_init_fn", worker_init_fn),
        ):
            if fn is not None and not callable(fn):
                raise TypeError(f"{name} must be callable, got {type(fn).__name__}")
        if num_workers == 0:
            for name, given in (
                ("prefetch_factor", prefetch_factor is not None),
                ("multiprocessing_context", multiprocessing_context is not None),
                ("persistent_workers", bool(persistent_workers)),
            ):
                if given:
                    raise ValueError(
                        f"{name} only applies to worker processes; it needs " "num_workers > 0."
                    )
        _mp_context(multiprocessing_context)  # unknown or wrong type: refused now
        if batch_size is None and drop_last:
            raise ValueError(
                "batch_size=None turns automatic batching off, so there is no "
                "short last batch for drop_last to drop."
            )
        generator = _as_generator(generator, "DataLoader")

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.multiprocessing_context = multiprocessing_context
        # Shared by the default RandomSampler and the workers' base seed, so
        # one generator reproduces the order and the augmentations together.
        self.generator: Generator | None = generator
        # Match reference framework: prefetch_factor=None when num_workers=0, else default 2
        if prefetch_factor is None:
            self.prefetch_factor = 2 if num_workers > 0 else None
        else:
            self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        # Unified memory: there is no pageable/pinned distinction to act on.
        self.pin_memory = pin_memory

        # An iterable-style dataset has no length and no indices, so there
        # is nothing for a sampler to sample.
        self.batch_sampler: Sampler | None
        self.sampler: Sampler | None
        self._persistent_iter: _MultiProcessDataLoaderIter | None = None
        self._iterable_style = isinstance(dataset, IterableDataset)
        if self._iterable_style:
            if shuffle:
                raise ValueError(
                    "shuffle is not meaningful for an IterableDataset — it "
                    "has no indices to permute.  Shuffle inside the "
                    "dataset's __iter__, or use a map-style Dataset."
                )
            if sampler is not None or batch_sampler is not None:
                raise ValueError(
                    "sampler and batch_sampler do not apply to an "
                    "IterableDataset; it decides its own order."
                )
            self.batch_sampler = None
            self.sampler = None
        elif batch_sampler is not None:
            if batch_size != 1 or shuffle or sampler is not None or drop_last:
                raise ValueError(
                    "batch_sampler is mutually exclusive with "
                    "batch_size, shuffle, sampler, and drop_last."
                )
            self.batch_sampler = batch_sampler
            self.batch_size = None
            self.sampler = sampler
        else:
            if sampler is not None and shuffle:
                raise ValueError("sampler and shuffle are mutually exclusive.")
            if sampler is None:
                sampler = (
                    RandomSampler(dataset, generator=self.generator)
                    if shuffle  # None and False both → SequentialSampler
                    else SequentialSampler(dataset)
                )
            self.batch_sampler = (
                BatchSampler(sampler, batch_size, drop_last) if batch_size is not None else None
            )
            self.sampler = sampler

        if collate_fn is None:
            collate_fn = default_collate if self._auto_collation else default_convert
        self.collate_fn: Callable[..., object] = collate_fn

    @property
    def _auto_collation(self) -> bool:
        """Whether samples are grouped into batches before ``collate_fn``."""
        if self._iterable_style:
            return self.batch_size is not None
        return self.batch_sampler is not None

    @property
    def _index_sampler(self) -> Sampler | None:
        """What one step draws: a batch of indices, or one index."""
        return self.batch_sampler if self.batch_sampler is not None else self.sampler

    def _fetch_spec(self) -> _FetchSpec:
        auto = self._auto_collation
        return _FetchSpec(
            iterable=self._iterable_style,
            auto_collation=auto,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size or 1,
            drop_last=self.drop_last,
            use_getitems=(not self._iterable_style and auto and self.collate_fn is default_collate),
        )

    def __iter__(self) -> Iterator[Tensor | tuple[Tensor, ...]]:
        """Return an iterator over one full pass of collated mini-batches.

        Dispatches to either the single-process iterator (``num_workers ==
        0``) or the multi-process iterator. When ``persistent_workers`` is
        enabled the multi-process worker pool survives between epochs and
        the same iterator is reset for each pass; otherwise workers are
        spawned per pass and shut down when it ends or the iterator is
        dropped.

        Returns
        -------
        Iterator
            Yields the output of ``collate_fn`` applied to each sampled
            batch of dataset items.
        """
        it = self._persistent_iter
        if it is not None and not it._shutdown:
            # A persistent pool keeps the streams it was seeded with; they
            # carry on into the new epoch rather than restart.
            it._reset(self)
            return it
        # The one place a base seed is drawn: once per new iterator, from
        # ``generator`` (default: the global generator).  Drawn with no
        # workers to use it too, so a seed gives the same shuffle whatever
        # ``num_workers`` is.
        base_seed = _draw_seed(self.generator)
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        # A first epoch, or a pool shut down by a timeout or a dead worker:
        # start a fresh one rather than feed dead queues.
        it = _MultiProcessDataLoaderIter(self, base_seed)
        if self.persistent_workers:
            self._persistent_iter = it
        return it

    def __len__(self) -> int:
        """Return the number of steps per epoch (``len`` of the index sampler).

        An iterable-style dataset has no length to divide, so neither has
        the loader over it.  ``TypeError`` rather than a guess: a wrong
        ``len`` silently truncates a progress bar, a learning-rate
        schedule, or an epoch.
        """
        index_sampler = self._index_sampler
        if self._iterable_style or index_sampler is None:
            raise TypeError(
                "this DataLoader wraps an IterableDataset, which has no "
                "length — iterate it instead of asking how long it is."
            )
        return len(index_sampler)
