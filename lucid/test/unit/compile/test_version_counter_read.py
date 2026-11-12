"""A forward that decides on ``Tensor._version`` is refused, not replayed (LCD-986).

A trace runs the forward's Python once, so a branch on a tensor's mutation counter
(``if self.w._version != self._stamp: rebuild``) is taken once, at the trace, and the
executable replays that branch for every later call.  After an in-place change of the
parameter (an optimizer step) the compiled call kept answering from the cache the trace
saw: 4.15 to 10.26 off eager on the toys here.

The owner is the ``Tensor._version`` property: a read while a trace is active marks it
unsupported (:func:`lucid._engine_scope.refuse_trace`, the one place that verdict is
given), every compile entry point refuses it, and the call runs eagerly.  The engine's own
readers (``compile._core.buffer_writes``, ``autograd.function``) read ``TensorImpl.version``
directly and are not refused.

Each pattern runs call, call, an in-place change, call, call against an eager twin.  What a
read is not (a read outside any trace, an engine read) must not refuse, and the scenario
must catch the replay when the refusal is taken away.

What ``StateContract`` documents as a limit is pinned at the end: a branch on ``.grad`` or
``requires_grad`` is not in the key, so a compiled call replays the branch of its trace.
"""

import warnings
from collections.abc import Callable, Iterator
from typing import Final

import pytest

import lucid
import lucid.nn as nn
from lucid._C import engine as _C_engine
from lucid.compile._core.fallback import CompileFallbackWarning
from lucid.compile._core.module_state import StateContract
from lucid.compile._core.trace_context import recording
from lucid.test._fixtures.devices import metal_available
from lucid.test.unit.compile._helpers import COMPILE_DEVICE

pytestmark = [
    pytest.mark.skipif(not metal_available(), reason="compiled executables run on Metal"),
    pytest.mark.usefixtures("inference"),
]

Tensor = lucid.Tensor
WIDTH: Final = 4
TOLERANCE: Final = 1e-4

#: Why the read is refused, as the fallback notice says it.
VERSION_NOTICE: Final = "version counter was read while tracing"


def _scale_of(w: Tensor) -> Tensor:
    return w.sum(0).abs() + 1.0


class _Base(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(lucid.randn(WIDTH, WIDTH))


class _BufferRebuild(_Base):
    """A buffer rebuilt from the parameter when its counter moved: the card's toy."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("scale", lucid.ones(WIDTH), persistent=False)
        self._stamp = -1

    def forward(self, x: Tensor) -> Tensor:
        if self.w._version != self._stamp:
            self.scale = _scale_of(self.w)
            self._stamp = self.w._version
        return x @ self.w * self.scale


class _PlainAttributeCache(_Base):
    """The same cache kept in a plain tensor attribute instead of a buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = lucid.ones(WIDTH, device=COMPILE_DEVICE)
        self._stamp = -1

    def forward(self, x: Tensor) -> Tensor:
        if self.w._version != self._stamp:
            self.scale = _scale_of(self.w)
            self._stamp = self.w._version
        return x @ self.w * self.scale


class _BranchOnly(_Base):
    """Nothing is kept: the counter alone picks the branch."""

    def forward(self, x: Tensor) -> Tensor:
        y = x @ self.w
        return y * 2.0 if self.w._version > 0 else y


class _HostRead(_Base):
    """The positive control: a host read decides, which the tracer has always refused."""

    def __init__(self) -> None:
        super().__init__()
        self.limit = float(self.w.sum()) + 1.0

    def forward(self, x: Tensor) -> Tensor:
        y = x @ self.w
        return y * 2.0 if float(self.w.sum()) > self.limit else y


type _Model = Callable[[], nn.Module]

PATTERNS: Final[dict[str, _Model]] = {
    "buffer_rebuild": _BufferRebuild,
    "plain_attribute_cache": _PlainAttributeCache,
    "branch_only": _BranchOnly,
}


class _Run:
    """What one model did over the scenario: its four outputs, its cache, what it said."""

    def __init__(
        self, outputs: list[Tensor], info: dict[str, object] | None, said: list[str]
    ) -> None:
        self.outputs = outputs
        self.info = info
        self.said = said


def _run(make: _Model, *, compiled: bool) -> _Run:
    """Call twice, change the parameter in place the way an optimizer step does, call twice."""
    lucid.manual_seed(0)
    model = make().to(COMPILE_DEVICE)
    x = lucid.randn(3, WIDTH).to(COMPILE_DEVICE)
    wrapped = lucid.compile(model) if compiled else None
    fn = model if wrapped is None else wrapped
    outputs: list[Tensor] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        outputs += [fn(x), fn(x)]
        model.w.add_(lucid.ones(WIDTH, WIDTH, device=COMPILE_DEVICE) * 0.5)
        outputs += [fn(x), fn(x)]
    said = [str(w.message) for w in caught if issubclass(w.category, CompileFallbackWarning)]
    return _Run(outputs, None if wrapped is None else wrapped.cache_info(), said)


def _gaps(make: _Model) -> list[float]:
    """The largest difference from the eager twin after each of the four calls."""
    eager, compiled = _run(make, compiled=False), _run(make, compiled=True)
    return [
        float((a - b).abs().max().item())
        for a, b in zip(eager.outputs, compiled.outputs, strict=True)
    ]


@pytest.mark.parametrize("name", PATTERNS)
def test_a_version_read_is_refused_and_the_call_runs_eagerly(name: str) -> None:
    run = _run(PATTERNS[name], compiled=True)
    assert run.info is not None
    assert run.info["eager_only"] and run.info["entries"] == 0, run.info
    assert any(VERSION_NOTICE in why for why in run.said), run.said


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(
            "buffer_rebuild",
            marks=pytest.mark.xfail(
                strict=True,
                reason="LCD-990: the refused trace puts the rebuilt buffer back but keeps its "
                "stamp, so the eager rerun skips the rebuild (see "
                "test_a_refused_trace_leaves_the_buffer_and_its_stamp_agreeing)",
            ),
        ),
        "plain_attribute_cache",
        "branch_only",
    ],
)
def test_the_refused_call_equals_eager_after_every_call(name: str) -> None:
    assert max(_gaps(PATTERNS[name])) <= TOLERANCE


def test_a_host_read_is_still_refused() -> None:
    """The control: the same scenario, decided by a host read, was and stays eager and right."""
    run = _run(_HostRead, compiled=True)
    assert run.info is not None and run.info["eager_only"] and run.info["entries"] == 0
    assert any("host" in why for why in run.said), run.said
    assert max(_gaps(_HostRead)) <= TOLERANCE


def test_the_scenario_catches_a_replayed_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the refusal the compiled call keeps the trace-time branch: the guard has teeth."""
    monkeypatch.setattr(Tensor, "_version", property(lambda self: self._impl.version))
    run = _run(_BranchOnly, compiled=True)
    assert run.info is not None and run.info["entries"] == 1 and not run.info["eager_only"]
    assert max(_gaps(_BranchOnly)) > 1.0


# -- LCD-990: what a refused trace leaves behind -----------------------------------------


class _Stamped(nn.Module):
    """A buffer rebuilt once, guarded by a plain stamp; the call is refused before or after it.

    ``refuse`` is what the trace is refused by (a host read, or the version read this card
    refuses) and ``write`` how the rebuild lands (rebinding the buffer, or writing into it).
    """

    def __init__(self, refuse: str, write: str) -> None:
        super().__init__()
        self.refuse, self.write = refuse, write
        self.w = nn.Parameter(lucid.randn(WIDTH, WIDTH))
        self.register_buffer("scale", lucid.ones(WIDTH), persistent=False)
        self._stamp = -1

    def forward(self, x: Tensor) -> Tensor:
        if self.refuse == "host_read":
            float(self.w.sum())
        else:
            self.w._version
        if self._stamp != 0:
            if self.write == "rebind":
                self.scale = _scale_of(self.w)
            else:
                self.scale.copy_(_scale_of(self.w))
            self._stamp = 0
        return x @ self.w * self.scale


@pytest.mark.xfail(
    strict=True,
    reason="LCD-990: a refused CompiledModule trace puts a rebuilt buffer back (it is not in "
    "__dict__, so SlotWatch does not restore the stamp) and the eager rerun, which finds the "
    "stamp moved, skips the rebuild and reads the buffer as it was before the trace",
)
@pytest.mark.parametrize("write", ["rebind", "copy_"])
@pytest.mark.parametrize("refuse", ["host_read", "version_read"])
def test_a_refused_trace_leaves_the_buffer_and_its_stamp_agreeing(refuse: str, write: str) -> None:
    """The first call of a refused module equals eager's, buffer and stamp alike (33.73 off today)."""

    def build() -> tuple[_Stamped, Tensor]:
        lucid.manual_seed(0)
        return _Stamped(refuse, write).to(COMPILE_DEVICE), lucid.randn(3, WIDTH).to(COMPILE_DEVICE)

    twin, x = build()
    model, _ = build()
    compiled = lucid.compile(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CompileFallbackWarning)
        got = compiled(x)
    want = twin(x)
    assert model._stamp == twin._stamp
    assert float((model.scale - twin.scale).abs().max().item()) <= TOLERANCE
    assert float((got - want).abs().max().item()) <= TOLERANCE


# -- what must not be refused ----------------------------------------------------------


@pytest.fixture
def traced() -> Iterator[_C_engine.compile.Tracer]:
    with recording() as tracer:
        yield tracer


def test_a_version_read_outside_a_trace_refuses_nothing() -> None:
    lucid.manual_seed(0)
    model = nn.Linear(WIDTH, WIDTH).to(COMPILE_DEVICE)
    x = lucid.randn(3, WIDTH).to(COMPILE_DEVICE)
    compiled = lucid.compile(model)
    with warnings.catch_warnings():
        warnings.simplefilter("error", CompileFallbackWarning)
        compiled(x)
        before = model.weight._version
        model.weight.add_(0.5)
        assert model.weight._version == before + 1
        compiled(x)
    info = compiled.cache_info()
    assert info["entries"] == 1 and not info["eager_only"], info


def test_the_engines_own_version_reads_do_not_refuse(traced: _C_engine.compile.Tracer) -> None:
    """``buffer_writes`` and ``autograd.function`` read ``TensorImpl.version``: that stays free."""
    t = lucid.randn(3)
    assert t._impl.version == 0
    assert not traced.graph.unsupported
    assert t._version == 0
    assert VERSION_NOTICE in str(traced.graph.unsupported)


def test_a_batch_norm_in_training_still_compiles() -> None:
    """Its running statistics are carried writes, which read the counter themselves."""
    lucid.manual_seed(0)
    model = nn.BatchNorm1d(WIDTH).to(COMPILE_DEVICE).train()
    x = lucid.randn(5, WIDTH).to(COMPILE_DEVICE)
    compiled = lucid.compile(model)
    with warnings.catch_warnings():
        warnings.simplefilter("error", CompileFallbackWarning)
        compiled(x)
        compiled(x)
    info = compiled.cache_info()
    assert info["entries"] == 1 and not info["eager_only"], info


# -- the refusal has one owner ----------------------------------------------------------


def test_refuse_trace_does_nothing_outside_a_trace_and_says_why_inside_one() -> None:
    from lucid._engine_scope import refuse_trace

    refuse_trace("outside")
    with recording() as tracer:
        refuse_trace("inside")
        assert "inside" in str(tracer.graph.unsupported)


def test_no_function_asks_for_the_tracer_to_mark_it_but_refuse_trace() -> None:
    """A refusal is ``refuse_trace(why)``: a function that reads ``current_tracer()`` and then
    calls ``mark_unsupported`` has made another copy of it (there were three before LCD-986)."""
    import ast
    from pathlib import Path

    root = Path(lucid.__file__).parent
    copies: list[str] = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        if rel.parts[0] == "test" or rel.as_posix() == "_engine_scope.py":
            continue
        source = path.read_text(encoding="utf-8")
        if "current_tracer" not in source or "mark_unsupported" not in source:
            continue
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            called = {
                call.func.attr
                for call in ast.walk(node)
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
            }
            if {"current_tracer", "mark_unsupported"} <= called:
                copies.append(f"{rel}:{node.lineno} {node.name}")
    assert not copies, copies


# -- the documented limit of the key ----------------------------------------------------


class _GradBranch(_Base):
    def forward(self, x: Tensor) -> Tensor:
        y = x @ self.w
        return y * 2.0 if self.w.grad is None else y


class _RequiresGradBranch(_Base):
    def forward(self, x: Tensor) -> Tensor:
        y = x @ self.w
        return y * 2.0 if self.w.requires_grad else y


def _flip_grad(model: _Base) -> None:
    model.w.grad = lucid.zeros(WIDTH, WIDTH, device=COMPILE_DEVICE)


def _flip_requires_grad(model: _Base) -> None:
    model.w.requires_grad_(False)


@pytest.mark.parametrize(
    ("make", "flip"),
    [(_GradBranch, _flip_grad), (_RequiresGradBranch, _flip_requires_grad)],
    ids=["grad_is_none", "requires_grad"],
)
def test_a_branch_on_grad_metadata_replays_the_branch_of_the_trace(
    make: Callable[[], _Base], flip: Callable[[_Base], None]
) -> None:
    """The limit ``StateContract`` documents: nothing keys ``.grad`` or ``requires_grad`` as branch state.

    The compiled call is a cache hit that replays the trace's ``y * 2``, where eager, after the
    flip, answers ``y``.  Nothing in the tree branches this way; when this starts failing, the key
    covers it, and the docstring is to be rewritten.
    """
    lucid.manual_seed(0)
    model = make().to(COMPILE_DEVICE)
    x = lucid.randn(3, WIDTH).to(COMPILE_DEVICE)
    compiled = lucid.compile(model)
    with warnings.catch_warnings():
        warnings.simplefilter("error", CompileFallbackWarning)
        first = compiled(x)
        flip(model)
        replayed = compiled(x)
    eager = model(x)
    assert float((replayed - first).abs().max().item()) <= TOLERANCE
    assert float((replayed - eager).abs().max().item()) > 1.0
    info = compiled.cache_info()
    assert info["entries"] == 1 and not info["eager_only"], info


def test_the_state_contract_documents_the_limit() -> None:
    doc = StateContract.__doc__ or ""
    for words in ("grad is None", "requires_grad", "_version"):
        assert words in doc, words
