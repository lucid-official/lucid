"""
lucid.compile._entry.decode_step — compiled single-token decode for StaticCache.

``compiled_decode_step(forward_fn, static_cache)`` returns a ``step(input_ids,
cache_position)`` callable that compiles the decode forward ONCE into a reused
:class:`MPSGraphExecutable` and rolls the cache forward in place.

Why it compiles where :class:`DynamicCache` does not
----------------------------------------------------
``DynamicCache`` grows its K/V by concatenation, so the tensor shape changes
every token and ``lucid.compile``'s shape-keyed signature forces a recompile per
step.  :class:`StaticCache` writes each token into a fixed
``(B, H, max_cache_len, D)`` buffer at a runtime ``cache_position`` via the
traceable :func:`lucid.index_copy`, so the signature is constant and one
executable serves every step (the ``cache_position`` is a fixed-shape ``(T,)``
int64 runtime feed, not a Python int — an int would be baked into the signature
and recompile every step).

Buffer write-back
-----------------
The ``index_copy`` write produces a NEW buffer tensor inside the executable; it
is surfaced as an explicit output (alongside the model output) and ``copy_``'d
back into the live ``StaticCache`` buffer after each run — the same shape as the
BatchNorm running-stats write-back, but forward-only (generation is ``no_grad``).
Because the live buffer is also the executable's pinned feed, the next step reads
the rolled-forward value.

Under grad mode a step that reads a tensor requiring grad — the model's
parameters included — runs ``forward_fn`` eagerly, as a :func:`lucid.compile`
call does: the executable records no autograd graph (LCD-618).
"""

import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Protocol, cast, final

from lucid._C import engine as _C_engine
from lucid.compile._core.signature import TraceReads, tree_requires_grad

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor
    from lucid.nn.module import Module
    from lucid.utils.cache import StaticCache

__all__ = ["compiled_decode_step", "is_compiled_decode_tracing"]

# Thread-local flag set while ``compiled_decode_step`` is tracing the decode
# graph.  A StaticCache-aware attention (GPT-2, …) checks it to attend over the
# FULL fixed-shape ``max_cache_len`` buffer (so the signature is constant and the
# decode compiles once) instead of its eager ``read_len``-narrowed view (which
# grows each step and would recompile).  Off → eager → the fast narrowed path.
_tls = threading.local()


def is_compiled_decode_tracing() -> bool:
    """Return ``True`` while ``compiled_decode_step`` is tracing the decode graph.

    A StaticCache-aware attention checks this to attend over the full fixed-shape
    ``max_cache_len`` buffer (so the decode compiles once) instead of its eager
    ``read_len``-narrowed view (which grows each step).  See the module docstring.
    """
    return bool(getattr(_tls, "tracing", False))


class _ExecutableLike(Protocol):
    """Surface of the pybind ``CompiledExecutable`` consumed here."""

    input_ids: list[int]
    output_ids: list[int]
    num_inputs: int


@final
@dataclass(slots=True)
class _DecodeEntry:
    """One compiled decode executable + the plan to invoke / roll forward.

    The cache buffers are fed PER-CALL (not pinned) and the executable's updated
    buffers are rebound onto ``static_cache`` — a pointer swap, no data copy.  So
    the only per-step write-back cost is a Python rebind, not ``copy_``-ing every
    ``(B,H,max_cache_len,D)`` buffer back (24 dispatches ≈ 0.6 ms on a 12-layer
    model).
    """

    exe: object
    external_feeds: dict[int, object]
    # Per ``exe.input_ids[i]``: 0 → input_ids arg, 1 → cache_position arg,
    # ("buf", slot) → the current cache buffer at flat slot (2*layer + {0:key,
    # 1:value}), None → a pinned param/constant in ``external_feeds``.
    input_source: tuple[object, ...]
    # Number of trailing executable outputs that are updated cache buffers
    # (== 2 * n_layers); they follow the model output at outs[1:].
    n_buffers: int = 0
    n_hits: int = 0
    compile_ms: float = 0.0
    last_run_ms: float = 0.0
    # What it reads besides the call's tensors and the cache buffers: under
    # grad mode one requiring grad sends the call eager (LCD-618).
    reads: TraceReads = TraceReads()


def _needs_gradients(
    arguments: tuple[Tensor, Tensor],
    static_cache: StaticCache,
    entry: _DecodeEntry | None,
    adapter: Module,
) -> bool:
    """Whether, grad mode on, the eager step would record an autograd graph.

    It does when a tensor it reads requires grad: an argument, a cache buffer,
    or one the executable reads besides them.  Before the first trace those
    are not known, and the tree's parameters and buffers stand in for them —
    ``adapter`` names ``forward_fn``'s modules.
    """
    if not _C_engine.grad_enabled():
        return False
    buffers = (*static_cache.key_cache, *static_cache.value_cache)
    if any(t.requires_grad for t in (*arguments, *buffers)):
        return True
    if entry is not None:
        return entry.reads.require_grad()
    return tree_requires_grad(adapter)


def compiled_decode_step(
    forward_fn: Callable[[Tensor, Tensor], Tensor],
    static_cache: StaticCache,
) -> Callable[[Tensor, Tensor], Tensor]:
    """Return a ``step(input_ids, cache_position)`` running one compiled decode.

    Parameters
    ----------
    forward_fn : callable
        ``forward_fn(input_ids, cache_position) -> output`` — one decode forward
        that reads/writes ``static_cache`` internally (e.g. a model's
        single-token path).  Must return a single Tensor (the decode logits /
        hidden state).
    static_cache : StaticCache
        The cache whose per-layer buffers ``forward_fn`` writes via
        :func:`lucid.index_copy`.  Its buffers are rolled forward in place.

    Returns
    -------
    callable
        ``step(input_ids, cache_position) -> Tensor``.  ``input_ids`` is the new
        token(s); ``cache_position`` is a fixed-shape ``(T,)`` int64 Tensor of
        absolute write indices.  On a signature miss it (re)traces + compiles;
        otherwise it is a pure run + buffer write-back.  Falls back to a plain
        eager ``forward_fn`` call if the decode graph cannot be compiled, and
        under grad mode when the step needs gradients (see the module notes).
    """
    from lucid._dispatch import _unwrap, _wrap
    from lucid._tensor.tensor import Tensor
    from lucid.autograd._grad_mode import no_grad
    from lucid.compile import _CallableModule, _tracing
    from lucid.compile._core.attention_probe import maybe_probe_for_graph
    from lucid.compile._core.signature import ParamFingerprint

    cache: dict[Any, _DecodeEntry] = {}
    eager_sigs: set[Any] = set()
    adapter = _CallableModule(forward_fn)  # names forward_fn's modules
    fingerprint = ParamFingerprint(adapter, cache.clear)

    def _restore(snap: tuple[Any, ...]) -> None:
        """Undo every cache mutation the trace caused (buffer rebinds + counters),
        so the real run starts from the pre-trace state."""
        orig_key, orig_val, orig_cumlen, orig_seen = snap
        static_cache.key_cache[:] = orig_key
        static_cache.value_cache[:] = orig_val
        static_cache._cumulative_length[:] = orig_cumlen
        static_cache._seen_tokens = orig_seen

    def _compile_for(input_ids: Tensor, cache_position: Tensor) -> _DecodeEntry | None:
        """Trace + compile one decode step for this signature."""
        t0 = time.perf_counter()

        # Snapshot ALL cache state before tracing.  forward_fn's ``update`` rebinds
        # key_cache[i]/value_cache[i] to the index_copy write outputs AND advances
        # the length counters — both must be rolled back so the live buffers stay
        # the pinned feeds and the real run isn't double-counted.
        n_layers = len(static_cache.key_cache)
        snap = (
            list(static_cache.key_cache),
            list(static_cache.value_cache),
            list(static_cache._cumulative_length),
            static_cache._seen_tokens,
        )
        orig_key, orig_val = snap[0], snap[1]
        call = (input_ids, cache_position)
        _tls.tracing = True
        try:
            with no_grad(), _tracing(watch=call, module=adapter) as tracer:
                out = forward_fn(*call)
        finally:
            _tls.tracing = False

        graph = tracer.graph
        if not graph.ops or not isinstance(out, Tensor):
            _restore(snap)
            return None

        maybe_probe_for_graph(graph)
        ext = tracer.external_feeds

        # The decode output + every cache buffer's post-write id become the
        # executable's explicit outputs (output first, then key/value per layer,
        # flat slot 2*i + {0:key, 1:value}).  Also map each ORIGINAL buffer's feed
        # impl id → its flat slot so the run feeds the current buffer and rebinds
        # the matching output.
        out_id = tracer.lookup_id(_unwrap(out))
        if out_id is None:
            _restore(snap)
            return None
        write_ids: list[int] = [int(out_id)]
        buf_slot: dict[int, int] = {}
        ok = True
        for i in range(n_layers):
            for slot_kv, (buf_after, buf_before) in enumerate(
                (
                    (static_cache.key_cache[i], orig_key[i]),
                    (static_cache.value_cache[i], orig_val[i]),
                )
            ):
                wid = tracer.lookup_id(_unwrap(buf_after))
                if wid is None:
                    ok = False
                    break
                write_ids.append(int(wid))
                buf_slot[id(_unwrap(buf_before))] = 2 * i + slot_kv
            if not ok:
                break

        _restore(snap)
        if not ok:
            return None

        try:
            exe = _C_engine.compile.compile_or_cached(
                graph, dict(ext), False, [], write_ids
            )
        except RuntimeError:
            return None
        if exe is None:
            return None

        # Feed plan: input_ids → 0, cache_position → 1, a cache buffer →
        # ("buf", slot) (fed per-call, rebound after), else → None (pinned param).
        ids_impl = id(_unwrap(input_ids))
        cp_impl = id(_unwrap(cache_position))
        input_source: list[object] = []
        for tid in exe.input_ids:
            impl = ext.get(tid)
            if impl is None:
                return None
            iid = id(impl)
            if iid == ids_impl:
                input_source.append(0)
            elif iid == cp_impl:
                input_source.append(1)
            elif iid in buf_slot:
                input_source.append(("buf", buf_slot[iid]))
            else:
                input_source.append(None)

        return _DecodeEntry(
            exe=exe,
            external_feeds=dict(ext),
            input_source=tuple(input_source),
            n_buffers=2 * n_layers,
            compile_ms=(time.perf_counter() - t0) * 1000.0,
            reads=TraceReads.of(tracer, exe.input_ids, input_source),
        )

    def _run(entry: _DecodeEntry, input_ids: Tensor, cache_position: Tensor) -> Tensor:
        """Run one cached decode step + roll the cache forward (rebind, no copy)."""
        exe = cast(_ExecutableLike, entry.exe)
        ids_impl, cp_impl = _unwrap(input_ids), _unwrap(cache_position)
        feed_impls: list[object] = []
        for tid, src in zip(exe.input_ids, entry.input_source, strict=True):
            if src == 0:
                feed_impls.append(ids_impl)
            elif src == 1:
                feed_impls.append(cp_impl)
            elif isinstance(src, tuple):  # ("buf", slot) — current cache buffer
                layer, kv = divmod(src[1], 2)
                buf = (
                    static_cache.key_cache[layer]
                    if kv == 0
                    else static_cache.value_cache[layer]
                )
                feed_impls.append(_unwrap(buf))
            else:  # pinned param / constant
                feed_impls.append(entry.external_feeds[tid])

        t0 = time.perf_counter()
        outs = _C_engine.compile.run_executable(entry.exe, feed_impls)
        entry.last_run_ms = (time.perf_counter() - t0) * 1000.0
        entry.n_hits += 1

        n_expected = 1 + entry.n_buffers
        if len(outs) != n_expected:
            raise RuntimeError(
                f"compiled_decode_step: executable returned {len(outs)} outputs, "
                f"expected {n_expected} (1 output + {entry.n_buffers} cache buffers)"
            )
        # Roll the cache forward by REBINDING each buffer onto the executable's
        # updated output (a pointer swap) — the next step feeds it.  No copy_.
        for slot in range(entry.n_buffers):
            layer, kv = divmod(slot, 2)
            new_buf = _wrap(cast(Any, outs[1 + slot]))
            if kv == 0:
                static_cache.key_cache[layer] = new_buf
            else:
                static_cache.value_cache[layer] = new_buf
        # Keep the length counter coherent for get_seq_length() queries (the
        # compiled run drives positions off cache_position, not the counter).
        t_new = int(input_ids.shape[1])
        for layer in range(len(static_cache._cumulative_length)):
            static_cache._cumulative_length[layer] += t_new
        static_cache._seen_tokens += t_new
        return _wrap(cast(Any, outs[0]))

    def step(input_ids: Tensor, cache_position: Tensor) -> Tensor:
        """Run one compiled decode step (or eager fallback)."""
        # The cache buffers must be pre-allocated (the prefill step does this) so
        # they are persistent external feeds rather than in-graph ``zeros``
        # re-created each run.  Until then, run eager — this allocates the cache
        # WITHOUT recording the signature as eager-only (so the next, allocated,
        # call still attempts to compile).
        if len(static_cache.key_cache) == 0:
            return forward_fn(input_ids, cache_position)

        try:  # ``key_of`` drops what read replaced tensors
            key = fingerprint.key_of((input_ids, cache_position), {}, dynamic=False)
        except TypeError, AttributeError:
            key = None

        if key is not None and key in eager_sigs:
            return forward_fn(input_ids, cache_position)

        call = (input_ids, cache_position)
        entry = cache.get(key) if key is not None else None
        if entry is None and not _needs_gradients(call, static_cache, None, adapter):
            before, entry = fingerprint.outside_slots(), _compile_for(*call)
            fingerprint.settle(() if entry is None else entry.reads.fed, before)
            if key is not None:
                key = fingerprint.rekey(key, call, {})
                if entry is None:
                    eager_sigs.add(key)
                else:
                    cache[key] = entry
        # A trace may show it reads a tensor requiring grad; it stays cached
        # for the steps made under ``no_grad``.
        if entry is None or _needs_gradients(call, static_cache, entry, adapter):
            return forward_fn(input_ids, cache_position)
        return _run(entry, input_ids, cache_position)

    step.cache = cache  # type: ignore[attr-defined]
    return step
