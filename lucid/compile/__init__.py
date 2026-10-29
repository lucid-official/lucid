"""
lucid.compile — graph-capture compile path for Apple Silicon.

Lowers a Lucid model's forward pass (and optionally its backward +
optimizer-update graph) into a single :class:`MPSGraphExecutable`
that is cached per input signature.  Subsequent calls with the same
signature reuse the cached executable, skipping Python dispatch
entirely for every op inside the captured region.

Surface
-------

* :func:`compile` — the top-level entry.  Wraps either an
  :class:`nn.Module` or a plain callable; returns a
  :class:`CompiledModule`.  Also usable as a decorator
  (``@lucid.compile`` or ``@lucid.compile(dynamic=False)``).
* :class:`CompiledModule` — module-shaped wrapper exposing
  ``cache_info()`` / ``timing()`` / ``clear_cache()`` plus
  ``model``-delegated APIs (``parameters`` / ``state_dict`` / ``to``
  / ``train`` / ``eval``).
* :func:`compile_optimizer` — wraps the *update* step of an optimizer
  into a single cached executable (8 optimizers supported; see
  :mod:`lucid.compile._optim.compiler`).  Forward + backward stay eager.
* :func:`fused_step` — single executable covering forward + loss +
  backward (via MPSGraph autodiff) + optimizer update via the
  ghost-grad placeholder mechanism.
* :func:`compiled_step` — convenience wrapper combining
  :func:`compile` + :func:`make_step` for one-line training loops.
* :func:`compiled_decode_step` — one cached executable per decode step of an
  autoregressive model over a :class:`StaticCache`; what
  ``generate(cache_implementation="compiled")`` uses (LCD-660).
* :func:`_tracing` — low-level context manager that installs a
  thread-local :class:`Tracer`; used internally by every entry above.
  Public-but-underscored so external callers know they're touching
  an unstable surface.

The MPSGraph builder + executable cache + per-op emitters live in
:file:`lucid/_C/compile/`.  The fallback policy is in :func:`compile`'s
notes: a call that needs gradients, or that the compiler cannot lower,
runs eagerly.
"""

import sys
import types
from contextlib import contextmanager
from collections.abc import Iterable
from typing import TYPE_CHECKING, Callable, Iterator, Protocol, cast, final, overload, override

from lucid._C import engine as _C_engine
from lucid._tensor import Tensor
from lucid._types import _ModuleOutput
from lucid.nn.module import Module as _Module

if TYPE_CHECKING:
    # The Tracer class lives in the `_C.engine.compile` sub-module.  The
    # bare attribute reference at runtime works under Python 3.14's lazy
    # annotation evaluation (PEP 649); we only need the import for the
    # static type checker.
    #
    # The trailing imports below mirror the names served by the runtime
    # ``__getattr__`` lazy loader: they exist purely so static tools
    # (type checkers, Griffe-based doc generation) see the full
    # ``__all__`` surface.  Runtime resolution still flows through
    # ``__getattr__`` to keep the heavy ``lucid.nn`` / ``lucid.autograd``
    # imports out of package init.
    from lucid._C.engine.compile import Tracer
    from lucid.compile._entry.module import CompiledModule
    from lucid.compile._debug.diagnose import DiagnosisReport, OpInfo, diagnose
    from lucid.compile._entry.decode_step import compiled_decode_step
    from lucid.compile._entry.function import compiled_step
    from lucid.compile._entry.fused_step import fused_step
    from lucid.compile._optim.compiler import compile_optimizer
    from lucid.compile._entry.step import make_step
    from lucid.nn.module import Module

__all__ = [
    "_tracing",
    "compile",
    "compiled_step",
    "compiled_decode_step",
    "compile_optimizer",
    "fused_step",
    "CompiledModule",
    "make_step",
    "save_compiled",
    "load_compiled",
    "diagnose",
    "DiagnosisReport",
    "OpInfo",
]


def save_compiled(cm: object, path: str) -> bool:
    """Serialise a compiled forward graph to disk (AOT export).

    Writes two files at ``path``: ``<path>.mpsgraphpackage`` (Apple's
    native MPSGraphExecutable archive, macOS 14+) and ``<path>.meta``
    (Lucid's I/O plan + dtype / shape / ABI metadata).  A
    :class:`CompiledModule` may hold multiple cached executables (one
    per input signature); this entry point serialises the most recently
    compiled one that has run — sufficient for AOT deployment where the
    production input shape is fixed.

    The on-disk artifact is identical to the per-process cache used
    when ``LUCID_COMPILE_DISK_CACHE=1`` is set — so calls
    ``save_compiled`` produces by hand are loadable by either
    :func:`load_compiled` or the runtime cache machinery.

    .. warning::

       Saving always succeeds; **loading is what is restricted**.  A
       reloaded MPSGraph package does not report the feed order it
       expects, and that order only survives the round trip while every
       feed shares one shape and dtype.  An executable whose feeds
       differ — which includes every :class:`~lucid.nn.Module` with
       parameters, since the weights and the activations rarely share a
       shape — is refused by :func:`load_compiled` rather than bound
       wrongly.  Until that is resolved, AOT export is usable for
       traced *functions* over uniform inputs, not for saving a trained
       model's forward pass.  Weights are not in the package either way:
       use :func:`lucid.save` / ``save_pretrained`` for those.

    Parameters
    ----------
    cm : CompiledModule
        A model wrapper returned by :func:`lucid.compile`, called at least
        once on Metal under :class:`lucid.no_grad` (or
        :func:`lucid.inference_mode`): a call that needs gradients runs
        eagerly and compiles nothing to save.
    path : str
        Filesystem prefix.  Two files are written:
        ``f"{path}.mpsgraphpackage"`` and ``f"{path}.meta"``.

    Returns
    -------
    bool
        ``True`` on success.  Raises :class:`RuntimeError` if no
        executable of ``cm`` has run yet.

    Examples
    --------
    >>> import lucid
    >>> import lucid.compile as lc
    >>> cm = lc.compile(lambda a, b: a * b + 1.0)
    >>> a = lucid.randn(2, 8).to("metal")
    >>> b = lucid.randn(2, 8).to("metal")
    >>> _ = cm(a, b)                             # populate cache
    >>> lc.save_compiled(cm, "/tmp/my_graph")    # writes .mpsgraphpackage + .meta
    True

    See Also
    --------
    :func:`load_compiled` — read back the artifact written here.
    :class:`CompiledModule` — the wrapper holding the cache being saved.
    """
    from lucid._C import engine as _C_engine

    # An entry is cached before its first run, which MPSGraph may still
    # refuse; one that has run is known to work.  Newest first.
    cache = getattr(cm, "_cache", None) or {}
    ran = [entry for entry in reversed(cache.values()) if entry.n_hits]
    if not ran:
        raise RuntimeError(
            "save_compiled: no executable of this CompiledModule has run yet. "
            "Call it once on the target input, on Metal, under lucid.no_grad() "
            "(or lucid.inference_mode()) before saving: a call that needs "
            "gradients - grad mode on and a tensor it reads, such as a "
            "trainable parameter, requiring grad - runs eagerly and compiles "
            "nothing."
        )
    return bool(_C_engine.compile.save_executable(ran[0].exe, path))


def load_compiled(path: str) -> object:
    """Load a previously-saved compiled executable into a thin wrapper.

    Returns an object exposing ``__call__(*args)`` that runs the saved
    executable directly.  **The executable expects every feed in
    feed order** — that includes both model parameters and runtime
    inputs.  This is the raw artifact level: there is no automatic
    parameter re-binding, because the saved package does not carry the
    parameter *values* (only the graph structure + I/O ids).

    For a smoother AOT workflow that re-binds parameters from a live
    model, build a thin caller-side helper around ``load_compiled``
    that stitches ``model.parameters()`` onto the executable's feed
    list before invocation.

    Parameters
    ----------
    path : str
        Filesystem prefix matching a previous :func:`save_compiled`
        call.  Expects both ``<path>.mpsgraphpackage`` and
        ``<path>.meta`` to exist.

    Returns
    -------
    object
        A callable wrapper.  Attributes:

        * ``num_inputs``: total feed count the executable expects.
        * ``input_ids``: ordered trace ids (debug / introspection).
        * ``__call__(*feeds)``: run with feeds in ``input_ids`` order.

    Raises
    ------
    FileNotFoundError
        Either ``.mpsgraphpackage`` or ``.meta`` is missing.
    RuntimeError
        The ``.meta`` sidecar is corrupted, the SDK ABI mismatches,
        or the saved executable references engine internals that no
        longer exist.

    Examples
    --------
    >>> import lucid
    >>> import lucid.compile as lc
    >>> cm = lc.compile(lambda a, b: a * b + 1.0)
    >>> a = lucid.randn(2, 8).to("metal")
    >>> b = lucid.randn(2, 8).to("metal")
    >>> _ = cm(a, b)                              # populate cache
    >>> lc.save_compiled(cm, "/tmp/my_graph")
    True
    >>> step = lc.load_compiled("/tmp/my_graph")
    >>> step.num_inputs                           # a and b, in first-read order
    2
    >>> out = step(a, b)
    >>> bool(lucid.allclose(out.to("cpu"), (a * b + 1.0).to("cpu")))
    True

    See Also
    --------
    :func:`save_compiled` — corresponding writer.
    """
    import os

    from lucid._C import engine as _C_engine
    from lucid._dispatch import _unwrap, _wrap

    pkg_path = f"{path}.mpsgraphpackage"
    meta_path = f"{path}.meta"
    if not os.path.exists(pkg_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"load_compiled: missing artifact at {path!r} — expected both "
            f"{pkg_path!r} and {meta_path!r}."
        )

    # Engine-level executable handle — its attributes are pybind11
    # objects, untyped at the Python boundary.  Narrow via a small
    # Protocol so the wrapper class stays type-checked.
    class _ExecutableHandle(Protocol):
        num_inputs: int
        input_ids: list[int]
        output_ids: list[int]
        input_shapes: list[list[int]]
        input_dtypes: list[str]

    exe_raw = _C_engine.compile.load_executable(path)
    if exe_raw is None:
        raise RuntimeError(
            f"load_compiled: failed to deserialise {path!r} — likely an "
            "ABI / format-version mismatch.  Recompile the model from "
            "source."
        )
    exe = cast(_ExecutableHandle, exe_raw)

    @final
    class _LoadedExecutable:
        """Thin wrapper that runs the deserialised executable."""

        def __init__(self, exe: _ExecutableHandle) -> None:
            self._exe = exe
            self.num_inputs = exe.num_inputs
            self.input_ids = list(exe.input_ids)
            self.output_ids = list(exe.output_ids)
            self.input_shapes = [list(sh) for sh in exe.input_shapes]
            self.input_dtypes = list(exe.input_dtypes)

        def feed_contract(self) -> str:
            """One line per feed slot: what shape and dtype it expects."""
            return "\n".join(
                f"  slot {i}: shape {tuple(sh)} dtype {dt}"
                for i, (sh, dt) in enumerate(zip(self.input_shapes, self.input_dtypes, strict=True))
            )

        def __call__(self, *args: object) -> object:
            if len(args) != self.num_inputs:
                raise ValueError(
                    f"load_compiled wrapper: expected {self.num_inputs} feeds "
                    f"in trace order, got {len(args)}.  The executable wants:\n"
                    f"{self.feed_contract()}\n"
                    "Feeds are the trace's external tensors in the order they "
                    "were first read — parameters and runtime inputs "
                    "interleaved, not parameters-then-inputs."
                )
            feeds = [_unwrap(cast(Tensor, a)) for a in args]
            outs = _C_engine.compile.run_executable(self._exe, feeds)
            wrapped = [_wrap(o) for o in outs]
            if not wrapped:
                return None
            if len(wrapped) == 1:
                return wrapped[0]
            return tuple(wrapped)

    return _LoadedExecutable(exe)


def __getattr__(name: str) -> object:
    """Lazy attribute loader for the public surface listed in ``__all__``.

    The user-facing names (``compiled_step``, ``CompiledModule``,
    ``make_step``, ``compile_optimizer``, ``fused_step``) live in
    submodules that pull :mod:`lucid.nn` / :mod:`lucid.autograd` at
    import time.  Importing them eagerly here would cycle through
    every op family during ``lucid.compile`` package init — slow and
    fragile.  The lazy path defers each import until first access.

    Parameters
    ----------
    name : str
        Attribute name requested via ``lucid.compile.<name>``.

    Returns
    -------
    object
        The resolved attribute (typically a callable or class).

    Raises
    ------
    AttributeError
        If ``name`` is not in the lazy table above.
    """
    # Lazy-load attributes that pull ``lucid.nn`` / ``lucid.autograd``
    # to avoid forcing those imports during ``lucid.compile`` package
    # init (the tracer scaffold above must stay importable before
    # nn/autograd have built their own surfaces).
    if name == "compiled_step":
        from lucid.compile._entry.function import compiled_step

        return compiled_step
    if name == "compiled_decode_step":
        from lucid.compile._entry.decode_step import compiled_decode_step

        return compiled_decode_step
    if name == "CompiledModule":
        from lucid.compile._entry.module import CompiledModule

        return CompiledModule
    if name == "make_step":
        from lucid.compile._entry.step import make_step

        return make_step
    if name == "compile_optimizer":
        from lucid.compile._optim.compiler import compile_optimizer

        return compile_optimizer
    if name == "fused_step":
        from lucid.compile._entry.fused_step import fused_step

        return fused_step
    if name == "diagnose":
        from lucid.compile._debug.diagnose import diagnose

        return diagnose
    if name == "DiagnosisReport":
        from lucid.compile._debug.diagnose import DiagnosisReport

        return DiagnosisReport
    if name == "OpInfo":
        from lucid.compile._debug.diagnose import OpInfo

        return OpInfo
    raise AttributeError(f"module 'lucid.compile' has no attribute {name!r}")


def _refuse_if_already_recording() -> None:
    """Raise if a :class:`Tracer` is already installed on this thread.

    Guards :func:`_tracing` against reentrancy (LCD-626): opening a second
    tracer while one is active would replace it, and :func:`_tracing`'s
    ``finally`` detaches to ``None`` unconditionally on exit rather than
    restoring whatever was there — dropping the outer recording.  See
    :func:`_tracing`'s ``Raises`` section for the entry points this reaches
    and the one that does not (:class:`CompiledModule`, which checks
    :func:`lucid.compile._core.trace_context.recording_active` itself and
    runs its wrapped model inline instead of reaching here).
    """
    if _C_engine.compile.current_tracer() is not None:
        raise RuntimeError(
            "lucid.compile._tracing: a trace is already recording on this "
            "thread. This entry point (make_step / fused_step / "
            "compiled_step / compile_optimizer / decode- or segment-step "
            "tracing) cannot be entered while already inside another "
            "trace — it would replace the active Tracer and drop it on "
            "exit instead of restoring it. A lucid.compile(...)-wrapped "
            "callable may be called from inside another trace: it runs "
            "its wrapped model inline, eagerly, under the outer trace."
        )


@contextmanager
def _tracing(
    watch: Iterable[object] = (), module: _Module | None = None, *, fold_constants: bool = False
) -> Iterator[Tracer]:
    """Install a fresh :class:`Tracer` for the calling thread.

    Every op dispatched inside the ``with`` block enters its
    :class:`OpScopeFull` constructor and forwards its header (name +
    output meta) into the active tracer.  On exit the tracer is detached
    and its recorded :class:`TraceGraph` becomes a frozen view that can
    still be inspected via ``tracer.graph``.

    Parameters
    ----------
    watch : iterable of object, optional
        The traced call's arguments, tensors nested in lists, tuples and
        dicts included. A host read of one of them (``.item()``,
        ``.tolist()``, ``float(t)``…) marks the recording unsupported even
        before an op has consumed it: the value read is the call's data,
        and whatever the host computes from it would be baked into every
        later call.
    module : Module or None, optional
        The traced model, whose parameters and buffers are watched the
        same way: a compiled executable feeds them on every run, so a
        value read from one would go stale once it changes (an optimizer
        step, ``load_state_dict``). So are those of the modules a wrapped
        function is bound to, closes over or reads as a global. Core ML
        export leaves it out — its weights are constants of the package
        anyway.
    fold_constants : bool, optional, keyword-only, default=False
        Leave out of the recording every op that reads nothing the trace
        sees, so a constant built on the spot (``lucid.tensor([...])
        .to(x.device)``, ``lucid.arange(n).float()``) reaches the first
        recorded op as one feed instead of as ops — on another device,
        those made the trace mixed-device and the call eager. Only for a
        caller whose ``watch`` and ``module`` cover every live tensor the
        call reads: anything left unwatched would be folded into its
        trace-time value. Only an op whose every input the trace made itself
        is left out; a tensor from outside the trace is always a feed.

    Yields
    ------
    Tracer
        The freshly-installed tracer.  Useful both for inspection during
        the trace (``len(tracer.graph)`` grows monotonically) and after
        (the final recorded DAG).

    Raises
    ------
    RuntimeError
        A :class:`Tracer` is already installed on this thread (LCD-626)
        — see :func:`_refuse_if_already_recording`.

    Notes
    -----
    Phase 1.1 only records op headers — input-id wiring and MPSGraph
    emission arrive in Phase 1.2.  The context manager today is the
    smoke-test surface that proves the :class:`OpScopeFull` hook is
    correctly integrated; it is intentionally a leading underscore name
    so callers do not depend on the partial Week-1 surface.

    Examples
    --------
    Smoke-test pattern::

        import lucid
        import lucid.nn as nn

        model = nn.Linear(8, 4)
        x = lucid.zeros((2, 8))

        with lucid.compile._tracing() as tracer:
            _ = model(x)

        for node in tracer.graph.ops:
            print(node)
    """
    from lucid.compile._core.signature import leaf_tensors, tree_tensors

    _refuse_if_already_recording()
    tracer = _C_engine.compile.Tracer()
    tracer.fold_constants = fold_constants
    state = () if module is None else tree_tensors(module)
    for argument in leaf_tensors((watch, state), {}):
        tracer.watch(argument._impl)
    _C_engine.compile.set_current_tracer(tracer)
    try:
        yield tracer
    finally:
        _C_engine.compile.set_current_tracer(None)
        # LUCID_COMPILE_DEBUG=1 writes the captured TraceGraph to a
        # temp-dir JSON file for offline inspection (Phase 1.2 builder
        # + parity tooling will consume the same format).  No-op when
        # the env var is unset.
        from lucid.compile._debug.trace_dump import dump_to_path_if_debug_enabled

        dump_to_path_if_debug_enabled(tracer.graph)


@contextmanager
def _host_check() -> Iterator[None]:
    """Run a validation read outside the active trace.

    A read that only decides whether to raise — an index bounds check, a
    "weights must be non-negative" guard — feeds nothing into the result,
    so it is not the data dependency a host read inside a trace otherwise
    is (see :func:`_tracing`'s ``watch``). Inside this block no tracer is
    installed: the read does not mark the recording unsupported, and the
    operations computing the checked value are not recorded either. The
    check itself runs once, on the traced call's values; a compiled
    program or an exported package does not repeat it.

    Only for reads whose value cannot reach the result. Anything computed
    from a value read here is a constant to the trace.

    Examples
    --------
    >>> import lucid
    >>> def scale(x):
    ...     with lucid.compile._host_check():
    ...         if bool((x < 0).any()):
    ...             raise ValueError("scale: x must be non-negative")
    ...     return x * 2
    """
    from lucid._engine_scope import untraced

    with untraced():
        yield


type _CompileOptions = dict[str, str | int | bool | Callable[..., object]]


@overload
def compile[**P, R](
    model: Callable[P, R],
    *,
    fullgraph: bool = ...,
    dynamic: bool | None = ...,
    backend: str | Callable[..., object] | None = ...,
    mode: str | None = ...,
    options: _CompileOptions | None = ...,
    disable: bool = ...,
) -> CompiledModule[P, R]: ...


@overload
def compile(
    model: None = None,
    *,
    fullgraph: bool = ...,
    dynamic: bool | None = ...,
    backend: str | Callable[..., object] | None = ...,
    mode: str | None = ...,
    options: _CompileOptions | None = ...,
    disable: bool = ...,
) -> Callable[[Callable[..., object]], object]: ...


def compile[**P, R](
    model: Callable[P, R] | None = None,
    *,
    fullgraph: bool = False,
    dynamic: bool | None = None,
    backend: str | Callable[..., object] | None = None,
    mode: str | None = None,
    options: _CompileOptions | None = None,
    disable: bool = False,
) -> CompiledModule[P, R] | Callable[[Callable[..., object]], object]:
    """Wrap ``model`` so calls are routed through cached MPSGraph executables.

    Accepts three call styles:

    * **Module wrapping** — ``lucid.compile(model)`` returns a
      :class:`CompiledModule` that delegates parameter walks /
      ``state_dict`` / device moves to ``model`` while routing
      ``__call__`` through an executable cache.
    * **Plain callable wrapping** — ``lucid.compile(fn)`` where ``fn``
      is a regular function or any callable.  Returns a thin wrapper
      whose ``__call__`` traces the function's body on the first
      invocation with a new input signature and caches an executable;
      it has no parameters of its own — pure tensor-in / tensor-out.
    * **Decorator usage** — ``@lucid.compile`` on a function (or
      ``@lucid.compile(dynamic=True)`` factory form) — identical to
      the plain-callable form above but lets the user opt-in inline.

    Parameters
    ----------
    model : nn.Module or callable, optional
        Either an :class:`nn.Module` instance (preferred when the
        compiled unit carries learnable parameters) or any callable
        whose signature is ``(*tensor_args) -> Tensor | tuple | dict``.
        When ``None``, returns a decorator factory.
    fullgraph : bool, optional
        Compatibility flag with the reference signature.  Default ``False``.
    dynamic : bool or None, optional
        Treat the leading (batch) axis as symbolic so calls differing only
        in batch size share **one** compiled executable (no recompile per
        batch).  Default ``None`` (treated as ``False``).
    backend : str or callable or None, optional
        Compatibility flag with the reference signature.  Default ``None``.
    mode : str or None, optional
        Compatibility flag with the reference signature.  Default ``None``.
    options : dict or None, optional
        Compatibility flag with the reference signature.  Default ``None``.
    disable : bool, optional
        When ``True``, makes ``compile`` a no-op returning ``model``
        unchanged.  Default ``False``.

    Returns
    -------
    CompiledModule
        A wrapper exposing the cache + run path.  When ``model`` was a
        Module the wrapper re-exposes the inner module's
        ``parameters`` / ``state_dict`` / training mode; when it was a
        plain callable those return empty sequences.

    Notes
    -----
    * A call the compiler cannot lower, or one that needs gradients (see
      :class:`CompiledModule`), runs eagerly with a one-time ``UserWarning``
      naming why; ``LUCID_COMPILE_VERBOSE=1`` reports every occurrence.
    * Attention lowers to a matmul-softmax-matmul decomposition, which
      materialises the score matrix; eager Metal's fused kernel never
      does.  A call whose attention would write more than 512 MiB of
      scores therefore runs eagerly, with a one-time notice — on an M4 Max
      a diffusion-transformer block at 18 720 keys took 120.8 ms compiled
      and 87.5 ms eager, and MPSGraph's own fused attention
      (``LUCID_COMPILE_FUSED_SDPA=1``) did not close the gap.
      ``LUCID_COMPILE_LONG_ATTENTION=compile`` keeps such calls compiled.

    Examples
    --------
    Module form::

        compiled = lucid.compile(nn.Linear(8, 4).to('metal'))
        with lucid.no_grad():
            y = compiled(x)

    Callable form::

        @lucid.compile
        def f(x, y):
            return (x @ y.T).relu()

        z = f(a, b)

    Factory decorator with kwargs::

        @lucid.compile(dynamic=False)
        def attention(q, k, v):
            return F.scaled_dot_product_attention(q, k, v)

    See Also
    --------
    CompiledModule : the wrapper class returned by this function.
    compile_optimizer : fused forward + backward + optimizer step
        for an even tighter training loop.
    """
    _ = (fullgraph, backend, mode, options)
    return _compile_model(model, dyn=bool(dynamic), disable=disable)


def _compile_model[**P, R](
    model: Callable[P, R] | None, *, dyn: bool, disable: bool
) -> CompiledModule[P, R] | Callable[[Callable[..., object]], object]:
    if model is None:
        return _compile_decorator_factory(dynamic=dyn, disable=disable)
    if disable:
        return cast("CompiledModule[P, R]", model)
    from lucid.compile._entry.module import CompiledModule as _CM

    if isinstance(model, _Module):
        return cast(_CM[P, R], _CM(model, dynamic=dyn))
    if callable(model):
        return cast(_CM[P, R], _CM(_CallableModule(model), dynamic=dyn))
    raise TypeError(
        f"lucid.compile: model must be an nn.Module or a callable, " f"got {type(model).__name__}"
    )


@final
class _CallableModule(_Module):
    """Internal adapter: wraps a plain callable as an :class:`nn.Module`.

    Subclasses :class:`Module` so :class:`CompiledModule`'s static type
    is ``Module`` everywhere (no ``object`` fallback).  Holds zero
    parameters / buffers / submodules; :meth:`forward` simply invokes
    the wrapped callable.  All Module hooks (training-mode flag,
    pre/post forward hooks, state_dict, to/train/eval) come from
    Module's default implementations.
    """

    _fn: object

    def __init__(self, fn: object) -> None:
        """Wrap ``fn`` so it satisfies the :class:`nn.Module` contract.

        Parameters
        ----------
        fn : callable
            The plain callable the user passed to :func:`compile`.
            Stored unwrapped; invoked verbatim on every forward.
        """
        super().__init__()
        # Stored via object.__setattr__ to bypass Module.__setattr__'s
        # Parameter / Tensor / Module-routing — ``fn`` is none of those.
        object.__setattr__(self, "_fn", fn)

    @override
    def forward(self, *args: object, **kwargs: object) -> _ModuleOutput:
        """Invoke the wrapped callable and return its result as a Module output.

        The wrapped callable's runtime return type isn't statically
        known; the :data:`_ModuleOutput` cast at the boundary is the
        contract :class:`CompiledModule` expects.  A mismatch (e.g. a
        callable returning a ``dict``) would surface at runtime as a
        downstream ``isinstance`` failure rather than silently corrupt
        the cache key.
        """
        fn = cast("Callable[..., object]", self._fn)
        out = fn(*args, **kwargs)
        # A list is as good as a tuple — the return-spec walk repacks either —
        # and ``split`` / ``chunk`` / ``unbind`` / ``meshgrid`` return lists,
        # so ``lucid.compile(lambda x: x.split(2))`` used to raise here.
        if not isinstance(out, Tensor) and not (
            isinstance(out, (tuple, list)) and all(isinstance(t, Tensor) for t in out)
        ):
            raise TypeError(
                "lucid.compile: wrapped callable must return a Tensor or a "
                f"tuple or list of Tensors, got {type(out).__name__}"
            )
        return cast(_ModuleOutput, out)

    @override
    def __repr__(self) -> str:
        """Diagnostic repr including the wrapped callable's qualname."""
        name = getattr(self._fn, "__qualname__", repr(self._fn))
        return f"<_CallableModule wrapping {name}>"


def _compile_decorator_factory(
    *, dynamic: bool = False, disable: bool = False
) -> Callable[[Callable[..., object]], object]:
    """Return a decorator that applies :func:`compile` with the given options."""

    def _decorator(target: object) -> object:
        """Apply :func:`compile` to ``target`` with the captured ``dynamic`` flag."""
        return compile(cast(Callable[..., object], target), dynamic=dynamic, disable=disable)

    return _decorator


@final
class _CallableCompileModule(types.ModuleType):
    """Module subclass so ``lucid.compile(...)`` is callable."""

    __call__ = staticmethod(compile)


sys.modules[__name__].__class__ = _CallableCompileModule
