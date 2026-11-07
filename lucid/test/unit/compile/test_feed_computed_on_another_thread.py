"""A value another thread computed during the call follows its inputs (LCD-904).

The tracer is per thread, so what a worker thread computes (``x * 2.0``) is no op
of the recording.  Its result is a tensor the trace never saw and that did not
exist before the call; an op on the tracing thread that reads it took it for an
outside input and pinned it at the first call's bytes, so every compiled call
after the first answered with the first call's value.  A grad node would have
given it away, but under ``no_grad`` (or when nothing it read requires grad)
there is none.

Each case computes the value one way (a worker thread, a pool, a worker whose
result goes into a module attribute), or builds a control: a tensor made before
the call, on any thread, or on the tracing thread.  It runs through four entry
points over four calls with a different input each, and every answer is compared
with a twin running eagerly.  The later calls are what tell the bug from a
correct answer: the first agrees either way.  A control keeps its executable.
"""

import threading
import warnings
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

import pytest

import lucid
import lucid.nn as nn
import lucid.optim as optim
from lucid._C import engine as _C_engine
from lucid.compile._core import fallback
from lucid.test.unit.compile._helpers import COMPILE_DEVICE


def _metal_ok() -> bool:
    try:
        lucid.zeros((1,)).to(COMPILE_DEVICE)
    except RuntimeError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")

Tensor = lucid.Tensor

WIDTH = 4
#: The scale of the input on each call; a compiled call that kept the first
#: call's value answers 1.0's on every later one.
SCALES = (1.0, 5.0, -2.0, 3.0)

_POOL = ThreadPoolExecutor(max_workers=2)


@pytest.fixture(autouse=True)
def _fresh_session(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(fallback, "_NOTICED", set())
    _C_engine.compile.session_cache_clear()
    yield


def _on_a_thread(fn: Callable[[], Tensor]) -> Tensor:
    out: list[Tensor] = []
    worker = threading.Thread(target=lambda: out.append(fn()))
    worker.start()
    worker.join()
    return out[0]


def _worker(net: nn.Module, x: Tensor) -> Tensor:
    return _on_a_thread(lambda: x * 2.0)


def _pool(net: nn.Module, x: Tensor) -> Tensor:
    return _POOL.submit(lambda: x * 2.0).result()


def _stored(net: nn.Module, x: Tensor) -> Tensor:
    # Computed on a worker, kept on the module, read back by the forward.
    net.cache = _on_a_thread(lambda: x * 2.0)
    return net.cache


def _worker_chain(net: nn.Module, x: Tensor) -> Tensor:
    # A worker's value that went through more ops on the tracing thread first.
    return _on_a_thread(lambda: x + 1.0) * 2.0 - 2.0


def _before(net: nn.Module, x: Tensor) -> Tensor:
    # Made before the call, on the main thread: an input, live.
    return net.before


def _before_on_a_thread(net: nn.Module, x: Tensor) -> Tensor:
    # Made before the call, on another thread: still outside the call.
    return net.before_threaded


def _on_the_tracing_thread(net: nn.Module, x: Tensor) -> Tensor:
    return x * 2.0


def _made_by_the_call(net: nn.Module, x: Tensor) -> Tensor:
    return lucid.ones(WIDTH, device=COMPILE_DEVICE) * 2.0 * x


#: name -> (how the forward gets the value, whether another thread computed it
#: during the call).
SOURCES: dict[str, tuple[Callable[[nn.Module, Tensor], Tensor], bool]] = {
    "worker thread": (_worker, True),
    "thread pool": (_pool, True),
    "stored in a module attribute": (_stored, True),
    "worker value used by more ops": (_worker_chain, True),
    "made before the call": (_before, False),
    "made before the call on a thread": (_before_on_a_thread, False),
    "made on the tracing thread": (_on_the_tracing_thread, False),
    "made by the call": (_made_by_the_call, False),
}


class _Net(nn.Module):
    """``lin(source(x) * 3)``: a layer, so that a training step has a parameter."""

    def __init__(self, source: str) -> None:
        super().__init__()
        lucid.manual_seed(0)
        self.lin = nn.Linear(WIDTH, 2)
        self.source = SOURCES[source][0]
        self.before = lucid.ones(WIDTH, device=COMPILE_DEVICE) * 2.0
        self.before_threaded = _on_a_thread(
            lambda: lucid.ones(WIDTH, device=COMPILE_DEVICE) * 2.0
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.lin(self.source(self, x) * 3.0)


def _net(source: str) -> _Net:
    return _Net(source).to(COMPILE_DEVICE)


def _floats(t: Tensor) -> list[float]:
    return [float(v) for v in t.reshape(-1).tolist()]


def _loss(out: Tensor) -> Tensor:
    return (out * out).mean()


def _x(scale: float) -> Tensor:
    return lucid.ones(3, WIDTH, device=COMPILE_DEVICE) * scale


class _Seen(NamedTuple):
    answers: list[list[float]]
    #: Every call ran an executable, none fell back to eager.
    compiled: bool = False


def _eager_module(net: _Net) -> _Seen:
    with lucid.no_grad():
        return _Seen([_floats(net(_x(s))) for s in SCALES])


def _module(net: _Net) -> _Seen:
    compiled = lucid.compile(net)
    with lucid.no_grad():
        answers = [_floats(compiled(_x(s))) for s in SCALES]
    info = compiled.cache_info()
    return _Seen(answers, info["entries"] == 1 and not info["eager_only"])


def _eager_module_grad(net: _Net) -> _Seen:
    return _Seen([_floats(net(_x(s))) for s in SCALES])


def _module_grad(net: _Net) -> _Seen:
    compiled = lucid.compile(net)
    return _Seen([_floats(compiled(_x(s))) for s in SCALES])


def _eager_step(net: _Net) -> _Seen:
    seen = _Seen([])
    for s in SCALES:
        net.zero_grad()
        loss = _loss(net(_x(s)))
        loss.backward()
        seen.answers.append([float(loss.item()), *_floats(net.lin.weight.grad)])
    return seen


def _step(net: _Net) -> _Seen:
    step = lucid.compile.make_step(net, _loss)
    answers = []
    for s in SCALES:
        net.zero_grad()
        loss = step(_x(s))
        loss.backward()
        answers.append([float(loss.item()), *_floats(net.lin.weight.grad)])
    return _Seen(answers, len(step.cache) == 1 and not step.eager_only.snapshot())


def _eager_fused(net: _Net) -> _Seen:
    opt = optim.SGD(net.parameters(), lr=1e-4)
    seen = _Seen([])
    for s in SCALES:
        opt.zero_grad()
        loss = _loss(net(_x(s)))
        loss.backward()
        opt.step()
        seen.answers.append([float(loss.item()), *_floats(net.lin.weight)])
    return seen


def _fused(net: _Net) -> _Seen:
    step = lucid.compile.fused_step(net, _loss, optim.SGD(net.parameters(), lr=1e-4))
    answers = []
    for s in SCALES:
        loss = step(_x(s))
        answers.append([float(loss.item()), *_floats(net.lin.weight)])
    return _Seen(answers, not step.cache_info()["eager_only"])


#: name -> (the compiled call, its eager twin, whether a control keeps its
#: executable there).  A grad-mode ``lucid.compile`` runs eagerly by design.
ENTRIES: dict[str, tuple[Callable[[_Net], _Seen], Callable[[_Net], _Seen], bool]] = {
    "compile no_grad": (_module, _eager_module, True),
    "compile grad mode": (_module_grad, _eager_module_grad, False),
    "make_step": (_step, _eager_step, True),
    "fused_step": (_fused, _eager_fused, True),
}


def _close(got: list[float], want: list[float]) -> bool:
    return len(got) == len(want) and all(
        abs(g - w) <= 1e-4 * (1.0 + abs(w)) for g, w in zip(got, want, strict=True)
    )


def _agree(got: _Seen, want: _Seen) -> None:
    for i in range(len(SCALES)):
        assert _close(got.answers[i], want.answers[i]), f"call {i}: {got.answers} vs {want.answers}"


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("source", SOURCES)
def test_a_value_computed_on_another_thread_follows_its_inputs(source: str, entry: str) -> None:
    run, run_eager, keeps = ENTRIES[entry]
    got = run(_net(source))
    _agree(got, run_eager(_net(source)))
    if keeps and not SOURCES[source][1]:
        assert got.compiled, f"{source}: {entry} fell back to eager"


def test_the_repro_of_the_report() -> None:
    """The case as filed: a worker thread's ``x * 2.0`` read by ``* 3.0``."""

    class M(nn.Module):
        def forward(self, x: Tensor) -> Tensor:
            return _on_a_thread(lambda: x * 2.0) * 3.0

    compiled = lucid.compile(M())
    with lucid.no_grad():
        for v in (1.0, -2.0, -9.0):
            x = lucid.tensor([v, v], device=COMPILE_DEVICE)
            assert _floats(compiled(x)) == [v * 6.0, v * 6.0]


def test_it_says_why_it_runs_eagerly() -> None:
    class M(nn.Module):
        def forward(self, x: Tensor) -> Tensor:
            return _on_a_thread(lambda: x * 2.0) * 3.0

    compiled = lucid.compile(M())
    x = lucid.ones(WIDTH, device=COMPILE_DEVICE)
    with lucid.no_grad(), pytest.warns(fallback.CompileFallbackWarning, match="another thread"):
        compiled(x)
    assert compiled.cache_info()["eager_only"]


def test_a_tensor_made_before_the_call_stays_compiled() -> None:
    before = _on_a_thread(lambda: lucid.ones(WIDTH, device=COMPILE_DEVICE) * 2.0)
    compiled = lucid.compile(lambda x: x * before)
    with lucid.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("error", fallback.CompileFallbackWarning)
        for s in SCALES:
            assert _floats(compiled(_x(s))) == _floats(_x(s) * before)
    info = compiled.cache_info()
    assert info["entries"] == 1 and not info["eager_only"], info


def test_only_the_compile_entries_watch_other_threads() -> None:
    """Core ML export (``_tracing``) holds every outside tensor as a constant."""
    from lucid.compile._core.trace_context import recording

    with lucid.compile._tracing() as export_tracer:
        assert not export_tracer.refuse_off_thread_values
    with recording() as compile_tracer:
        assert compile_tracer.refuse_off_thread_values
    assert not compile_tracer.refuse_off_thread_values


def test_a_recording_refuses_what_a_thread_made_after_it_opened_only() -> None:
    """The window is the recording's: an older one's tensors are inputs of a newer one."""
    from lucid.compile._core.trace_context import recording

    opened, release = threading.Event(), threading.Event()

    def lead() -> None:
        with recording():
            opened.set()
            release.wait(timeout=30)

    leader = threading.Thread(target=lead)
    leader.start()
    try:
        assert opened.wait(timeout=30)
        # Made while the leader's recording is open, before this one opens.
        earlier = lucid.ones(WIDTH, device=COMPILE_DEVICE) * 2.0
        with lucid.no_grad(), recording() as tracer:
            _ = earlier * 3.0
            assert not tracer.graph.unsupported
            later = _on_a_thread(lambda: lucid.ones(WIDTH, device=COMPILE_DEVICE) * 2.0)
            _ = later * 3.0
        assert "another thread" in str(tracer.graph.unsupported)
    finally:
        release.set()
        leader.join()


def test_a_value_the_tracing_thread_made_inside_a_detached_scope_stays_an_input() -> None:
    """A detached scope is still the tracing thread: what it makes is not another thread's."""
    from lucid.compile._core.trace_context import recording
    from lucid._engine_scope import untraced

    with lucid.no_grad(), recording() as tracer:
        with untraced():
            made = lucid.ones(WIDTH, device=COMPILE_DEVICE) * 2.0
        _ = made * 3.0
    assert "another thread" not in str(tracer.graph.unsupported)
