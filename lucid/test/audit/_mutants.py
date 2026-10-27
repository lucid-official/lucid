"""Proving the axes can go red.

A green axis is not evidence.  It becomes evidence only once the axis has
been shown to produce a red light for the defect it claims to catch —
otherwise "0 defects" and "0 questions asked" print identically, and this
tool's whole output is the first of those two sentences.

The method is mutation.  Each axis gets a symbol that is *deliberately
broken in exactly the way that axis exists to notice*, and the axis has
to report it:

    grad        an op whose analytic gradient is not its derivative
    nonfinite   an op that swallows a NaN
    edge        an op that invents data from an empty input
    device      an op that answers differently on the two backends
    serialize   a round trip that loses values
    hook        a handle whose ``remove`` does not remove

Run it::

    lucid-audit --self-check

An axis with no mutant is reported as **unproven** rather than quietly
omitted — the same rule the sweep applies to itself.  Unproven is not a
failure; it is the honest statement that nobody has shown this axis can
fail, and it is the work queue for the next person who wants to trust it.

The mutants live here rather than in a test because the question they
answer is one the tool should be able to answer *about itself* on demand,
on the machine it is running on.  A test that only runs in CI proves the
axes could fail somewhere else.
"""

import contextlib
import subprocess
import sys
from typing import TYPE_CHECKING, Any

import numpy as np

import lucid
from lucid.test.audit import _axes, _probe, _surface
from lucid.test.audit._result import Status

if TYPE_CHECKING:
    from collections.abc import Iterator


class Mutant:
    """One deliberate defect, and the axis that must notice it."""

    __slots__ = (
        "axis",
        "name",
        "why",
        "kind",
        "subsystem",
        "build",
        "patch",
        "qualname",
        "isolated_exit",
    )

    def __init__(
        self,
        axis: str,
        name: str,
        why: str,
        build: Any,
        kind: str = "op",
        subsystem: str = "lucid",
        patch: Any = None,
        qualname: str = "",
        isolated_exit: int | None = None,
    ) -> None:
        self.axis = axis
        self.name = name
        self.why = why
        self.build = build
        self.kind = kind
        self.subsystem = subsystem
        #: The spelling the axis dispatches on.  A patched mutant has to
        #: keep the real symbol's name — the state, hook, metadata,
        #: weights and quant axes all decide what to do by looking at it,
        #: so a synthetic name makes the axis decline the very defect it
        #: was handed.
        self.qualname = qualname
        #: A context manager that installs the defect into the real
        #: package, for axes that dispatch on a symbol's *name* and so
        #: cannot be handed a synthetic one.
        self.patch = patch
        self.isolated_exit = isolated_exit


# ── mutants that need no patching: a broken callable is enough ───────────────


def _swallows_nan(x: Any) -> Any:
    return lucid.where(lucid.isnan(x), lucid.zeros_like(x), x)


def _terminates_process(*args: object, **kwargs: object) -> None:
    # Only invoked in the self-check's dedicated subprocess. A deterministic
    # abnormal exit exercises fatal-call detection without creating a core dump.
    import os

    os._exit(73)


def _invents_data(x: Any) -> Any:
    if int(np.prod(tuple(x.shape))) == 0:
        return lucid.ones((3,))
    return x


def _degenerate_on_rank(x: Any) -> Any:
    # Accepts a rank it cannot actually handle and answers with an empty
    # tensor instead of refusing.  That is the shape the rank axis exists
    # for: the caller gets no exception, just a result with nothing in it.
    if x.ndim <= 1:
        return lucid.zeros((0,))
    return x


def _device_dependent(x: Any) -> Any:
    return x + (1.0 if "metal" in str(x.device) else 0.0)


def _wrong_gradient(x: Any) -> Any:
    # Forward is x**2 and the backward pass reports x, not 2x.  Detaching
    # one factor is the shortest way to say "the value is right and the
    # derivative is not", which is the defect class this framework has
    # produced most often.
    return x * x.detach()


def _wrong_second_derivative(x: Any) -> Any:
    return x * x * x.detach()


def _drops_the_graph(x: Any) -> Any:
    # The right values, detached from the input: a loss through it trains
    # nothing.  ``masked_select``, ``F.ctc_loss`` and five ``linalg`` ops
    # did exactly this, and the axis filed them under "unsupported".
    return (x * x).detach()


def _unstable_at_scale(x: Any) -> Any:
    return lucid.where(lucid.abs(x) > 1e3, x * float("nan"), x)


def _narrows_dtype(shape: Any, dtype: Any = None) -> Any:
    # Ignores the dtype it was asked for.  A factory that quietly hands
    # back single precision is the silent-wrong-answer shape.
    del dtype
    return lucid.zeros(shape, dtype=lucid.float32)


class _NoGradientModule(lucid.nn.Module):
    """Forwards, and no parameter ever receives a gradient."""

    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.weight = lucid.nn.Parameter(_probe.as_f32(_probe.sample("moderate", (width, width))))

    def forward(self, x: Any) -> Any:
        return x @ self.weight.detach()


class _StuckOptimizer(lucid.optim.SGD):
    """Accepts a step and moves nothing."""

    def step(self, closure: Any = None) -> Any:
        return closure() if closure is not None else None


def _wrong_under_broadcast(a: Any, b: Any) -> Any:
    """Elementwise on equal shapes, doubled once anything broadcasts.

    The mutant that showed the broadcast axis was comparing shapes and
    nothing else — all nine directions passed while every broadcast
    answer was twice what it should be.
    """
    if tuple(a.shape) == tuple(b.shape):
        return a * b
    return (a * b) * 2.0


def _randmutant(x: Any) -> Any:
    """Named like a sampler, and deaf to the seed."""
    return lucid.tensor(np.random.rand(*tuple(x.shape)))  # noqa: NPY002


class _LyingDataset:
    """``len()`` says 99 and the ninety-ninth item does not exist."""

    def __len__(self) -> int:
        return 99

    def __getitem__(self, index: int) -> tuple[Any, int]:
        if index >= 6:
            raise IndexError(index)
        return lucid.tensor(np.zeros(2, np.float32)), index % 2


class _ReprRaises:
    """Constructs, and cannot be printed."""

    def __repr__(self) -> str:
        raise RuntimeError("this class cannot describe itself")


class _MutatesItsInput:
    """An augmentation that writes through to the caller's image."""

    def __call__(self, image: Any) -> Any:
        image += 1.0
        return image


# ── mutants that have to be installed into the package ───────────────────────


@contextlib.contextmanager
def _patched(owner: Any, name: str, value: Any) -> Iterator[None]:
    """Install ``value`` at ``owner.name`` and put the original back."""
    missing = object()
    before = getattr(owner, name, missing)
    setattr(owner, name, value)
    try:
        yield
    finally:
        if before is missing:
            with contextlib.suppress(AttributeError):
                delattr(owner, name)
        else:
            setattr(owner, name, before)


@contextlib.contextmanager
def _deaf_setter() -> Iterator[None]:
    """``set_num_threads`` accepts the value and the getter never sees it."""
    with _patched(lucid, "set_num_threads", lambda _n: None):
        yield


@contextlib.contextmanager
def _seed_that_does_nothing() -> Iterator[None]:
    with _patched(lucid, "manual_seed", lambda _n: None):
        yield


@contextlib.contextmanager
def _asymmetric_promotion() -> Iterator[None]:
    real = lucid.promote_types

    def broken(a: Any, b: Any) -> Any:
        # Commutative for everything except one pair, which is how a real
        # promotion table goes wrong: one missing entry, not all of them.
        if str(a) == "lucid.int32" and str(b) == "lucid.float32":
            return a
        return real(a, b)

    with _patched(lucid, "promote_types", broken):
        yield


@contextlib.contextmanager
def _leaky_hook_handle() -> Iterator[None]:
    real = lucid.nn.register_module_forward_hook

    class _Handle:
        def __init__(self, inner: Any) -> None:
            self._inner = inner

        def remove(self) -> None:
            return None  # ...and the hook stays installed

    issued: list[_Handle] = []

    def register(fn: Any) -> _Handle:
        handle = _Handle(real(fn))
        issued.append(handle)
        return handle

    with _patched(lucid.nn, "register_module_forward_hook", register):
        try:
            yield
        finally:
            # The defect belongs to the mutant, not to the process.  Left
            # installed, the hook kept every later forward's output alive —
            # the step-cost budget, run after the audit, saw fewer frees and
            # a higher peak in every workload.
            for handle in issued:
                handle._inner.remove()


@contextlib.contextmanager
def _lossy_round_trip() -> Iterator[None]:
    real = lucid.load

    def broken(path: str, *args: Any, **kwargs: Any) -> Any:
        restored = real(path, *args, **kwargs)
        return restored + 1.0 if hasattr(restored, "shape") else restored

    with _patched(lucid, "load", broken):
        yield


@contextlib.contextmanager
def _compiler_that_disagrees() -> Iterator[None]:
    real = lucid.compile.compile

    def broken(fn: Any, *args: Any, **kwargs: Any) -> Any:
        compiled = real(fn, *args, **kwargs)
        return lambda *a, **k: compiled(*a, **k) + 1.0

    with _patched(lucid.compile, "compile", broken):
        yield


@contextlib.contextmanager
def _slice_that_takes_the_wrong_stride() -> Iterator[None]:
    """``x[..., ::2]`` comes back holding ``x[..., 1::2]``.

    The failure a *materialised* view can actually have: the right shape
    and the wrong elements.  The layout axis builds its strided operand by
    interleaving a −7 sentinel into the odd columns and slicing the even
    ones back out, so a slice that takes the wrong stride returns the
    sentinel — which is exactly what that axis now checks for, and why it
    is no longer unprovable on an engine with no lazy views.
    """
    real = lucid.Tensor.__getitem__

    def broken(self: Any, key: Any) -> Any:
        if (
            isinstance(key, tuple)
            and key
            and isinstance(key[-1], slice)
            and key[-1] == slice(None, None, 2)
        ):
            return real(self, key[:-1] + (slice(1, None, 2),))
        return real(self, key)

    with _patched(lucid.Tensor, "__getitem__", broken):
        yield


@contextlib.contextmanager
def _prune_that_misses_its_target() -> Iterator[None]:
    real = lucid.nn.utils.prune.l1_unstructured

    def broken(module: Any, name: str = "weight", amount: float = 0.5) -> Any:
        return real(module, name, amount=amount / 2.0)

    with _patched(lucid.nn.utils.prune, "l1_unstructured", broken):
        yield


@contextlib.contextmanager
def _registry_that_forgets() -> Iterator[None]:
    with _patched(lucid.weights, "weights_for", lambda _name: None):
        yield


@contextlib.contextmanager
def _entry_points_that_disagree() -> Iterator[None]:
    """The free function and the method compute different things."""
    import lucid._tensor.tensor as tensor_module

    with _patched(lucid, "mutant_entry", lambda x: x * 2.0):
        with _patched(tensor_module.Tensor, "mutant_entry", lambda self: self * 3.0):
            yield


@contextlib.contextmanager
def _functional_grad_that_is_wrong() -> Iterator[None]:
    real = lucid.func.grad

    def broken(fn: Any, *args: Any, **kwargs: Any) -> Any:
        inner = real(fn, *args, **kwargs)
        return lambda *a, **k: inner(*a, **k) * 0.5

    with _patched(lucid.func, "grad", broken):
        yield


@contextlib.contextmanager
def _tableau_that_is_inconsistent() -> Iterator[None]:
    real = lucid.diffeq.RK4

    # ``ButcherTableau`` validates its own weights — "b must sum to 1" —
    # so the mutant cannot be constructed inconsistent.  That is a good
    # property of the class and it means the defect has to be installed
    # after the fact.
    broken = lucid.diffeq.ButcherTableau(
        a=real.a, b=real.b, c=real.c, order=real.order, name="mutant"
    )
    object.__setattr__(broken, "b", tuple(w * 2.0 for w in real.b))
    with _patched(lucid.diffeq, "MUTANT_TABLEAU", broken):
        yield


@contextlib.contextmanager
def _scheduler_that_goes_nan() -> Iterator[None]:
    real = lucid.optim.lr_scheduler.StepLR

    class _Broken(real):  # type: ignore[misc, valid-type]
        def get_lr(self) -> Any:
            return [float("nan") for _ in self.optimizer.param_groups]

    with _patched(lucid.optim.lr_scheduler, "StepLR", _Broken):
        yield


@contextlib.contextmanager
def _tokenizer_that_loses_text() -> Iterator[None]:
    real = lucid.utils.tokenizer.CharTokenizer

    class _Broken(real):  # type: ignore[misc, valid-type]
        def decode(self, *args: Any, **kwargs: Any) -> Any:
            text = super().decode(*args, **kwargs)
            return text[:-1] if isinstance(text, str) and text else text

    with _patched(lucid.utils.tokenizer, "CharTokenizer", _Broken):
        yield


@contextlib.contextmanager
def _distribution_with_an_impossible_sample() -> Iterator[None]:
    real = lucid.distributions.Normal

    class _Broken(real):  # type: ignore[misc, valid-type]
        def log_prob(self, value: Any) -> Any:
            return super().log_prob(value) * float("inf")

    with _patched(lucid.distributions, "Normal", _Broken):
        yield


@contextlib.contextmanager
def _create_graph_that_disagrees() -> Iterator[None]:
    """``grad(create_graph=True)`` returns something ``backward()`` does not.

    The defect class this axis exists for: ``prod`` / ``max`` / ``min``
    once returned the incoming seed under ``create_graph=True`` while
    ``backward()`` was correct, so the two routes to one derivative
    disagreed and only a comparison between them could see it.
    """
    real = lucid.autograd.grad

    def broken(*args: Any, **kwargs: Any) -> Any:
        out = real(*args, **kwargs)
        if not kwargs.get("create_graph"):
            return out
        return tuple(g * 0.5 if g is not None else g for g in out)

    with _patched(lucid.autograd, "grad", broken):
        yield


@contextlib.contextmanager
def _observer_that_ignores_its_input() -> Iterator[None]:
    real = lucid.quantization.MinMaxObserver

    class _Deaf(real):  # type: ignore[misc, valid-type]
        def calculate_qparams(self) -> Any:
            return (_probe.as_f32(np.array(0.01)), _probe.as_int(np.array(0)))

    with _patched(lucid.quantization, "MinMaxObserver", _Deaf):
        yield


#: One entry per axis that has a mutant.  Order follows ``ALL_AXES`` so a
#: reader can see the gaps.
MUTANTS: tuple[Mutant, ...] = (
    Mutant(
        "smoke",
        "fatal_callable",
        "invoking a callable terminates its isolated process",
        lambda: _terminates_process,
        isolated_exit=73,
    ),
    Mutant(
        "extreme",
        "softmax_not_normalized",
        "saturated softmax returns unnormalized probabilities",
        lambda: (lambda x: lucid.ones_like(x)),
        qualname="lucid.softmax",
    ),
    Mutant(
        "constant",
        "invalid_version",
        "an exported version is not a version string",
        lambda: "not-a-version",
        kind="value",
        qualname="lucid.__version__",
    ),
    Mutant("nonfinite", "swallows_nan", "returns 0 where its input was NaN", lambda: _swallows_nan),
    Mutant(
        "edge", "invents_data", "an empty input produces a non-empty output", lambda: _invents_data
    ),
    Mutant(
        "rank",
        "degenerate_on_rank",
        "a rank it cannot handle is answered with an empty tensor, not refused",
        lambda: _degenerate_on_rank,
    ),
    Mutant(
        "dtype",
        "ignores_dtype_argument",
        "a factory that hands back float32 whatever it was asked for",
        lambda: _narrows_dtype,
    ),
    Mutant(
        "device",
        "device_dependent",
        "answers differently on cpu and on metal",
        lambda: _device_dependent,
    ),
    Mutant(
        "stability",
        "unstable_at_scale",
        "returns NaN once the input passes 1e3",
        lambda: _unstable_at_scale,
    ),
    Mutant(
        "grad",
        "wrong_gradient",
        "forward is x**2 and the backward pass reports x",
        lambda: _wrong_gradient,
    ),
    Mutant(
        "grad",
        "drops_the_graph",
        "the output is right and carries no graph back to the input",
        lambda: _drops_the_graph,
    ),
    Mutant(
        "grad2",
        "wrong_second_derivative",
        "the second derivative is short by a factor",
        lambda: _wrong_second_derivative,
    ),
    Mutant(
        "module",
        "no_gradient_reaches_the_weight",
        "forwards, and no parameter ever receives a gradient",
        lambda: _NoGradientModule,
        kind="module",
        subsystem="nn",
    ),
    Mutant(
        "optim",
        "stuck_optimizer",
        "accepts a step and moves nothing",
        lambda: _StuckOptimizer,
        kind="optim",
        subsystem="optim",
    ),
    Mutant(
        "state",
        "deaf_setter",
        "set_num_threads accepts a value its getter never sees",
        lambda: lucid.get_num_threads,
        patch=_deaf_setter,
        qualname="lucid.get_num_threads",
    ),
    Mutant(
        "state",
        "seed_that_does_nothing",
        "manual_seed does not make a draw reproducible",
        lambda: lucid.manual_seed,
        patch=_seed_that_does_nothing,
        qualname="lucid.manual_seed",
    ),
    Mutant(
        "metadata",
        "asymmetric_promotion",
        "promote_types(int32, float32) disagrees with the reverse",
        lambda: lucid.promote_types,
        patch=_asymmetric_promotion,
        qualname="lucid.promote_types",
    ),
    Mutant(
        "hook",
        "leaky_handle",
        "handle.remove() leaves the hook installed",
        lambda: lucid.nn.register_module_forward_hook,
        patch=_leaky_hook_handle,
        qualname="nn.register_module_forward_hook",
    ),
    Mutant(
        "serialize",
        "lossy_round_trip",
        "load returns something other than what save wrote",
        lambda: lucid.save,
        kind="serialize",
        subsystem="serialization",
        patch=_lossy_round_trip,
        qualname="lucid.save",
    ),
    Mutant(
        "compiled",
        "compiler_disagrees",
        "the compiled function does not match the eager one",
        lambda: lucid.compile.compile,
        kind="compiled",
        subsystem="compile",
        patch=_compiler_that_disagrees,
        qualname="lucid.compile.compile",
    ),
    Mutant(
        "layout",
        "slice_takes_the_wrong_stride",
        "x[..., ::2] comes back holding the interleaved sentinel",
        lambda: lucid.Tensor.__getitem__,
        patch=_slice_that_takes_the_wrong_stride,
        qualname="lucid.Tensor.__getitem__",
    ),
    Mutant(
        "nnutils",
        "prune_misses_its_target",
        "amount=0.5 zeroes a quarter of the weights",
        lambda: lucid.nn.utils.prune.l1_unstructured,
        patch=_prune_that_misses_its_target,
        qualname="lucid.nn.utils.prune.l1_unstructured",
    ),
    Mutant(
        "weights",
        "registry_forgets",
        "weights_for cannot find what register_weights just registered",
        lambda: lucid.weights.weights_for,
        kind="util",
        subsystem="weights",
        patch=_registry_that_forgets,
        qualname="lucid.weights.weights_for",
    ),
    Mutant(
        "entry",
        "entry_points_disagree",
        "the free function doubles and the method triples",
        lambda: lucid.mutant_entry,
        patch=_entry_points_that_disagree,
        qualname="lucid.mutant_entry",
    ),
    Mutant(
        "broadcast",
        "wrong_under_broadcast",
        "correct shape, and twice the value once anything broadcasts",
        lambda: _wrong_under_broadcast,
    ),
    Mutant(
        "determinism",
        "randmutant",
        "named like a sampler and deaf to the seed",
        lambda: _randmutant,
        qualname="lucid.randmutant",
    ),
    Mutant(
        "data",
        "lying_dataset",
        "len() says 99 and the ninety-ninth item does not exist",
        lambda: _LyingDataset,
        kind="data",
        subsystem="utils.data",
        qualname="lucid.utils.data.LyingDataset",
    ),
    Mutant(
        "contract",
        "repr_raises",
        "constructs, and cannot be printed",
        lambda: _ReprRaises,
        kind="class",
        qualname="lucid.ReprRaises",
    ),
    Mutant(
        "transform",
        "mutates_its_input",
        "an augmentation that writes through to the caller's image",
        lambda: _MutatesItsInput,
        kind="transform",
        subsystem="utils.transforms",
        qualname="lucid.utils.transforms.MutatesItsInput",
    ),
    Mutant(
        "functional",
        "wrong_functional_grad",
        "func.grad returns half the derivative",
        lambda: lucid.func.grad,
        patch=_functional_grad_that_is_wrong,
        qualname="lucid.func.grad",
    ),
    Mutant(
        "scheduler",
        "scheduler_goes_nan",
        "the learning rate becomes NaN",
        lambda: lucid.optim.lr_scheduler.StepLR,
        kind="scheduler",
        subsystem="optim.lr_scheduler",
        patch=_scheduler_that_goes_nan,
        qualname="lucid.optim.StepLR",
    ),
    Mutant(
        "tokenizer",
        "tokenizer_loses_text",
        "decode drops the last character of what encode was given",
        lambda: lucid.utils.tokenizer.CharTokenizer,
        kind="tokenizer",
        subsystem="utils.tokenizer",
        patch=_tokenizer_that_loses_text,
        qualname="lucid.utils.tokenizer.CharTokenizer",
    ),
    Mutant(
        "distribution",
        "impossible_sample",
        "log_prob is not finite on the distribution's own samples",
        lambda: lucid.distributions.Normal,
        kind="distribution",
        subsystem="distributions",
        patch=_distribution_with_an_impossible_sample,
        qualname="lucid.distributions.Normal",
    ),
    Mutant(
        "diffeq",
        "inconsistent_tableau",
        "the Runge-Kutta weights sum to 2, not 1",
        lambda: lucid.diffeq.MUTANT_TABLEAU,
        kind="diffeq",
        subsystem="diffeq",
        patch=_tableau_that_is_inconsistent,
        qualname="lucid.diffeq.MUTANT_TABLEAU",
    ),
    Mutant(
        "creategraph",
        "create_graph_disagrees",
        "grad(create_graph=True) returns half of what backward() does",
        lambda: _wrong_gradient,
        patch=_create_graph_that_disagrees,
        qualname="lucid.creategraph_mutant",
    ),
    Mutant(
        "quant",
        "deaf_observer",
        "an observer whose scale is the same for every range",
        lambda: lucid.quantization.MinMaxObserver,
        kind="quant",
        subsystem="quantization",
        patch=_observer_that_ignores_its_input,
        qualname="lucid.quantization.MinMaxObserver",
    ),
)


class Verdict:
    """What one mutant proved, or failed to."""

    __slots__ = ("axis", "name", "why", "caught", "status", "detail")

    def __init__(
        self, axis: str, name: str, why: str, caught: bool, status: str, detail: str
    ) -> None:
        self.axis = axis
        self.name = name
        self.why = why
        self.caught = caught
        self.status = status
        self.detail = detail


def _symbol_for(mutant: Mutant) -> _surface.Symbol:
    obj = mutant.build()
    if mutant.qualname:
        return _surface.Symbol(mutant.qualname, mutant.subsystem, mutant.kind, obj)
    prefix = {"nn": "nn.", "tensor": "Tensor."}.get(mutant.subsystem, "lucid.")
    return _surface.Symbol(f"{prefix}{mutant.name}", mutant.subsystem, mutant.kind, obj)


def verify(ctx: _axes.Context | None = None) -> list[Verdict]:
    """Run every mutant against the axis that must catch it."""
    context = ctx if ctx is not None else _axes.Context()
    out: list[Verdict] = []
    for mutant in MUTANTS:
        if mutant.isolated_exit is not None:
            code = (
                "import sys; from lucid.test.audit import _axes, _mutants; "
                "m = next(m for m in _mutants.MUTANTS if m.name == sys.argv[1]); "
                "a = _axes.axis_by_name(m.axis); s = _mutants._symbol_for(m); "
                "assert a.applies(s); a.run(s, _axes.Context())"
            )
            try:
                child = subprocess.run(
                    [sys.executable, "-c", code, mutant.name], capture_output=True, timeout=30
                )
                caught = child.returncode == mutant.isolated_exit
                out.append(
                    Verdict(
                        mutant.axis,
                        mutant.name,
                        mutant.why,
                        caught,
                        "isolated process exit",
                        f"exit {child.returncode}; expected deliberate exit {mutant.isolated_exit}",
                    )
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                out.append(
                    Verdict(mutant.axis, mutant.name, mutant.why, False, "harness error", str(exc))
                )
            continue
        axis = _axes.axis_by_name(mutant.axis)
        if axis is None:
            out.append(Verdict(mutant.axis, mutant.name, mutant.why, False, "no such axis", ""))
            continue
        guard = mutant.patch() if mutant.patch else contextlib.nullcontext()
        try:
            with guard, _probe.preserved_globals():
                symbol = _symbol_for(mutant)
                if not axis.applies(symbol):
                    out.append(
                        Verdict(
                            mutant.axis,
                            mutant.name,
                            mutant.why,
                            False,
                            "not applicable",
                            "the axis does not consider this symbol its business",
                        )
                    )
                    continue
                finding = axis.run(symbol, context)
        except Exception as exc:  # noqa: BLE001 - a mutant must not take the run down
            out.append(
                Verdict(
                    mutant.axis,
                    mutant.name,
                    mutant.why,
                    False,
                    "harness error",
                    f"{type(exc).__name__}: {str(exc)[:70]}",
                )
            )
            continue
        # A mutant is caught when the axis refuses to call it a pass.
        # FAIL is the usual answer; VACUOUS is also a refusal — it says
        # the check did not establish what it set out to — and an axis
        # that downgrades to it has still noticed.
        caught = finding.status in (Status.FAIL, Status.ERROR, Status.VACUOUS)
        out.append(
            Verdict(
                mutant.axis, mutant.name, mutant.why, caught, finding.status.value, finding.detail
            )
        )
    return out


#: Why an axis has no mutant.  Naming the reason is the difference
#: between "nobody has got to it" and "it cannot be done" — the second is
#: a finding about the framework and belongs in the report, not in a
#: backlog.
UNPROVEN_REASONS: dict[str, str] = {}


def unproven_axes() -> list[str]:
    """Axes no mutant exercises — reported, never omitted."""
    covered = {mutant.axis for mutant in MUTANTS}
    return [axis.name for axis in _axes.ALL_AXES if axis.name not in covered]


__all__ = ["MUTANTS", "Mutant", "UNPROVEN_REASONS", "Verdict", "unproven_axes", "verify"]
