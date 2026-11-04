"""A compiled call's in-place write ends where eager ends it (LCD-853, LCD-854, LCD-855).

The tracer used to ask what the written *tensor* was.  Never seen, or a feed,
meant "from outside": kept, put back, carried to the same tensor after every
run.  An op's output meant "made here": nothing to do.  Neither answer looks at
the buffer the write lands in, and two silent wrong answers followed:

* A tensor the call made with no op to its name (``lucid.tensor([...],
  device="metal")``, an ndarray, ``arange`` on Metal, ``shared_tensor``) looked
  never seen, so each compiled call started from the last one's result.
* A view the call took of a buffer from outside looked like an op's output, so
  the write landed in the buffer once, at trace time, and never again.  (A
  buffer that is the whole of its bytes hid this: the tracer re-derives such a
  member through the view.)

Now a tensor is told by how it reached the trace, and a write by the bytes it
lands in.  A tensor the call made starts over on every call; a write landing in
bytes something from outside reads is put back after the trace and carried on
every run, through the very view that wrote it.

Every case compares three calls with different inputs, and the buffers after
each one, against a twin model running eagerly.  A later call is what tells
these bugs from a correct answer: the first call agrees either way.
"""

from collections.abc import Callable, Iterator
from typing import NamedTuple

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid._C import engine as _C_engine
from lucid._dispatch import _wrap
from lucid.compile import _tracing
from lucid.compile._core import fallback
from lucid.test.unit.compile._helpers import COMPILE_DEVICE


def _metal_ok() -> bool:
    try:
        lucid.zeros((1,)).to(COMPILE_DEVICE)
    except RuntimeError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")

SHAPE = (3, 4)
ROWS = [[1.0, 2.0, 3.0, 4.0]] * 3
SCALES = (1.0, 5.0, -2.0)

Tensor = lucid.Tensor
Body = Callable[[nn.Module, Tensor], Tensor]
Make = Callable[[], Tensor]
Setup = Callable[[nn.Module], None]


@pytest.fixture(autouse=True)
def _fresh_session(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(fallback, "_NOTICED", set())
    _C_engine.compile.session_cache_clear()
    yield


def _inputs() -> list[Tensor]:
    return [lucid.ones(*SHAPE, device=COMPILE_DEVICE) * scale for scale in SCALES]


class _Model(nn.Module):
    """``body``, followed by a layer so that a training step has a parameter."""

    def __init__(self, body: Body, setup: Setup | None) -> None:
        super().__init__()
        self.body = body
        self.lin = nn.Linear(SHAPE[-1], 2)
        if setup is not None:
            setup(self)

    def forward(self, x: Tensor) -> Tensor:
        return self.lin(self.body(self, x))


def _twin(body: Body, setup: Setup | None) -> _Model:
    lucid.manual_seed(0)
    return _Model(body, setup).to(COMPILE_DEVICE)


def _loss(out: Tensor) -> Tensor:
    return (out * out).mean()


def _state(model: nn.Module) -> dict[str, list[complex]]:
    return {name: buf.tolist() for name, buf in model.named_buffers()}


class _Seen(NamedTuple):
    answers: list[list[float]]
    states: list[dict[str, list[complex]]]
    compiled: bool


Run = Callable[[_Model], _Seen]


def _floats(t: Tensor) -> list[float]:
    return [float(v) for v in t.reshape(-1).tolist()]


def _run_eager_module(model: _Model) -> _Seen:
    seen = _Seen([], [], False)
    with lucid.no_grad():
        for x in _inputs():
            seen.answers.append(_floats(model(x)))
            seen.states.append(_state(model))
    return seen


def _run_module(model: _Model) -> _Seen:
    compiled = lucid.compile(model)
    answers: list[list[float]] = []
    states: list[dict[str, list[complex]]] = []
    with lucid.no_grad():
        for x in _inputs():
            answers.append(_floats(compiled(x)))
            states.append(_state(model))
    info = compiled.cache_info()
    return _Seen(answers, states, info["entries"] == 1 and not info["eager_only"])


def _run_eager_step(model: _Model) -> _Seen:
    seen = _Seen([], [], False)
    for x in _inputs():
        loss = _loss(model(x))
        loss.backward()
        seen.answers.append([float(loss.item()), *_floats(model.lin.weight.grad)])
        seen.states.append(_state(model))
    return seen


def _run_step(model: _Model) -> _Seen:
    step = lucid.compile.make_step(model, _loss)
    answers: list[list[float]] = []
    states: list[dict[str, list[complex]]] = []
    for x in _inputs():
        loss = step(x)
        loss.backward()
        answers.append([float(loss.item()), *_floats(model.lin.weight.grad)])
        states.append(_state(model))
    return _Seen(answers, states, len(step.cache) == 1 and not step.eager_only.snapshot())


def _run_eager_fused(model: _Model) -> _Seen:
    opt = optim.SGD(model.parameters(), lr=1e-4)
    seen = _Seen([], [], False)
    for x in _inputs():
        opt.zero_grad()
        loss = _loss(model(x))
        loss.backward()
        opt.step()
        seen.answers.append([float(loss.item()), *_floats(model.lin.weight)])
        seen.states.append(_state(model))
    return seen


def _run_fused(model: _Model) -> _Seen:
    step = lucid.compile.fused_step(model, _loss, optim.SGD(model.parameters(), lr=1e-4))
    answers: list[list[float]] = []
    states: list[dict[str, list[complex]]] = []
    for x in _inputs():
        loss = step(x)
        answers.append([float(loss.item()), *_floats(model.lin.weight)])
        states.append(_state(model))
    return _Seen(answers, states, not step.cache_info()["eager_only"])


#: name -> (the compiled call, its eager twin).  ``module`` traces with folding
#: on, the two steps with folding off.
ENTRIES: dict[str, tuple[Run, Run]] = {
    "module": (_run_module, _run_eager_module),
    "make_step": (_run_step, _run_eager_step),
    "fused_step": (_run_fused, _run_eager_fused),
}


def _close(got: list[float], want: list[float]) -> bool:
    return len(got) == len(want) and all(
        abs(g - w) <= 1e-4 * (1.0 + abs(w)) for g, w in zip(got, want, strict=True)
    )


def _agrees(body: Body, entry: str, setup: Setup | None = None) -> _Seen:
    """Run ``body`` compiled and eagerly over three calls; return what compiled saw."""
    run, run_eager = ENTRIES[entry]
    got = run(_twin(body, setup))
    want = run_eager(_twin(body, setup))
    for i in range(len(SCALES)):
        assert _close(got.answers[i], want.answers[i]), f"call {i}: {got.answers} vs {want.answers}"
        assert got.states[i] == want.states[i], f"call {i}: {got.states} vs {want.states}"
    return got


# -- the call made it: it starts over on every call --------------------------


def _ndarray() -> object:
    return pytest.importorskip("numpy").array(ROWS, dtype="float32")


#: Every way a call makes a tensor from nothing, on Metal.
FACTORIES: dict[str, Make] = {
    "from_bytes": lambda: lucid.tensor(ROWS, device=COMPILE_DEVICE),
    "as_tensor": lambda: lucid.as_tensor(ROWS, device=COMPILE_DEVICE),
    "numpy-ctor": lambda: lucid.tensor(_ndarray(), device=COMPILE_DEVICE),
    "arange": lambda: lucid.arange(12.0, device=COMPILE_DEVICE).reshape(*SHAPE),
    "eye": lambda: lucid.eye(*SHAPE, device=COMPILE_DEVICE),
    "linspace": lambda: lucid.linspace(0.0, 1.0, 12, device=COMPILE_DEVICE).reshape(*SHAPE),
    "empty": lambda: lucid.empty(*SHAPE, device=COMPILE_DEVICE),
    "zeros": lambda: lucid.zeros(*SHAPE, device=COMPILE_DEVICE),
    "ones": lambda: lucid.ones(*SHAPE, device=COMPILE_DEVICE),
    "full": lambda: lucid.full(SHAPE, 2.0, device=COMPILE_DEVICE),
    "shared_tensor": lambda: lucid.metal.shared_tensor(SHAPE).to(COMPILE_DEVICE),
}

#: The same, made elsewhere and moved.
MOVED: dict[str, Make] = {
    "literal": lambda: lucid.tensor(ROWS).to(COMPILE_DEVICE),
    "from_numpy": lambda: lucid.from_numpy(_ndarray()).to(COMPILE_DEVICE),
    "zeros": lambda: lucid.zeros(*SHAPE).to(COMPILE_DEVICE),
    "arange": lambda: lucid.arange(12.0).reshape(*SHAPE).to(COMPILE_DEVICE),
}


def _add_input(y: Tensor, x: Tensor) -> None:
    y.add_(x)


def _through_a_slice(y: Tensor, x: Tensor) -> None:
    y[1:].add_(x[1:])


def _setitem(y: Tensor, x: Tensor) -> None:
    y[:2] = x[:2] * 3.0


#: Each leaves the tensor depending on the call's input, so a result that
#: carries over from an earlier call cannot pass for a right one.
MADE_WRITES: dict[str, Callable[[Tensor, Tensor], None]] = {
    "add_": _add_input,
    "through-a-slice": _through_a_slice,
    "setitem": _setitem,
}


def _writes_what_it_made(make: Make, write: Callable[[Tensor, Tensor], None]) -> Body:
    def body(_model: nn.Module, x: Tensor) -> Tensor:
        y = make()
        write(y, x)
        return y * 2.0

    return body


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("write", MADE_WRITES)
@pytest.mark.parametrize("factory", FACTORIES)
def test_a_tensor_the_call_made_starts_over_on_every_call(
    factory: str, write: str, entry: str
) -> None:
    seen = _agrees(_writes_what_it_made(FACTORIES[factory], MADE_WRITES[write]), entry)
    # The backward's lowering refuses ``empty`` wherever it appears, read or written.
    if not (factory == "empty" and entry != "module"):
        assert seen.compiled, f"{factory} / {write} no longer compiles under {entry}"


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("factory", MOVED)
def test_a_tensor_the_call_made_and_moved_starts_over_on_every_call(
    factory: str, entry: str
) -> None:
    # A step records the host's ops, which run it eager; the answer is what counts.
    _agrees(_writes_what_it_made(MOVED[factory], _add_input), entry)


@pytest.mark.parametrize("fold", [True, False], ids=["fold-on", "fold-off"])
@pytest.mark.parametrize("factory", FACTORIES)
def test_a_write_to_what_the_call_made_is_owed_to_nobody(factory: str, fold: bool) -> None:
    """Whatever folding does with the recording: the tensor is the call's, the
    feed reading it keeps the value from before the write, and there is nothing
    to carry back."""
    x = lucid.ones(*SHAPE, device=COMPILE_DEVICE)
    make = FACTORIES[factory]
    with lucid.no_grad(), _tracing(watch=[x], fold_constants=fold) as tracer:
        y = make()
        before = y.tolist()
        y.add_(x)
    assert list(tracer.outside_writes()) == []
    feeds = [f for f in tracer.external_feeds.values() if f is not x._impl]
    assert all(f is not y._impl for f in feeds)
    for feed in feeds:
        # ``arange`` and ``linspace`` are read through a reshape, so flat.
        assert _wrap(feed).reshape(*SHAPE).tolist() == before


def test_a_shared_tensor_written_through_an_alias_it_made_is_the_calls_own() -> None:
    x = lucid.ones(*SHAPE, device=COMPILE_DEVICE)
    with lucid.no_grad(), _tracing(watch=[x]) as tracer:
        host = lucid.metal.shared_tensor(SHAPE)
        alias = host.to(COMPILE_DEVICE)
        alias.add_(x)
    assert tracer.made_here(host._impl) and tracer.made_here(alias._impl)
    assert list(tracer.outside_writes()) == []


def _literal_after_an_op(_model: nn.Module, x: Tensor) -> Tensor:
    a = x * 2
    c = lucid.tensor([1.0, 2.0, 3.0, 4.0]).to(x.device)
    return a + c


def test_a_literal_after_an_op_leaves_that_op_recorded() -> None:
    """Making a tensor from host data opens no op, so the op before it is not
    the literal's to hold out: it used to vanish from the recording, and the
    whole call fell back to eager when the emitter found its result missing."""
    x = lucid.ones(*SHAPE, device=COMPILE_DEVICE)
    with lucid.no_grad(), _tracing(watch=[x], fold_constants=True) as tracer:
        _literal_after_an_op(nn.Module(), x)
    assert [op.name for op in tracer.graph.ops] == ["full", "mul", "add"]


@pytest.mark.parametrize("entry", ENTRIES)
def test_a_literal_after_an_op_keeps_the_call_compiled(entry: str) -> None:
    if entry == "module":
        seen = _agrees(_literal_after_an_op, entry)
        assert seen.compiled
    else:
        # A step records the literal's host ops: eager, and the same answer.
        _agrees(_literal_after_an_op, entry)


def test_a_literal_on_metal_after_an_op_keeps_every_entry_compiled() -> None:
    def body(_model: nn.Module, x: Tensor) -> Tensor:
        a = x * 2
        c = lucid.tensor([1.0, 2.0, 3.0, 4.0], device=x.device)
        return a + c

    for entry in ENTRIES:
        assert _agrees(body, entry).compiled, entry


def test_a_draw_written_in_place_is_drawn_again_each_call() -> None:
    """A draw is a feed, and the call's own: nothing to carry back onto it."""
    x = lucid.ones(4, device=COMPILE_DEVICE)

    def fn(x: Tensor) -> Tensor:
        r = lucid.randn(4, device=COMPILE_DEVICE)
        r.add_(x)
        return r

    with lucid.no_grad(), _tracing(watch=[x]) as tracer:
        fn(x)
    assert list(tracer.outside_writes()) == []
    compiled = lucid.compile(fn)
    with lucid.no_grad():
        first, second = compiled(x).tolist(), compiled(x).tolist()
    assert first != second
    assert abs(sum(second) / len(second)) < 10.0  # not accumulating the writes


# -- a write through a view reaches the buffer ---------------------------------


def _plain(model: nn.Module) -> None:
    model.register_buffer("b", lucid.zeros(*SHAPE, device=COMPILE_DEVICE))


def _a_run_of_a_larger_buffer(model: nn.Module) -> None:
    model.register_buffer("b", lucid.zeros(5, 4, device=COMPILE_DEVICE)[1:4])


def _strided_part_of_a_larger_buffer(model: nn.Module) -> None:
    model.register_buffer("b", lucid.zeros(4, 5, device=COMPILE_DEVICE)[1:, 1:])


BUFFERS: dict[str, Setup] = {
    "plain": _plain,
    "run-of-larger": _a_run_of_a_larger_buffer,
    "strided-part": _strided_part_of_a_larger_buffer,
}

VIEWS: dict[str, Callable[[Tensor], Tensor]] = {
    "slice": lambda b: b[1:],
    "index": lambda b: b[1],
    "column": lambda b: b[:, 1],
    "mT": lambda b: b.mT,
    "diagonal": lambda b: lucid.diagonal(b),
    "view-of-a-view": lambda b: b[1:][0],
}


def _writes_through(view: Callable[[Tensor], Tensor]) -> Body:
    def body(model: nn.Module, x: Tensor) -> Tensor:
        view(model.b).add_(x.mean())
        return x * 1.0

    return body


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("view", VIEWS)
@pytest.mark.parametrize("buffer", BUFFERS)
def test_a_write_through_a_view_reaches_the_buffer(buffer: str, view: str, entry: str) -> None:
    seen = _agrees(_writes_through(VIEWS[view]), entry, BUFFERS[buffer])
    assert seen.compiled, f"{buffer} / {view} no longer compiles under {entry}"


def _complex_buffer(model: nn.Module) -> None:
    model.register_buffer("b", lucid.zeros(4, dtype=lucid.complex64, device=COMPILE_DEVICE))


COMPLEX_PARTS: dict[str, Callable[[Tensor], Tensor]] = {
    "real": lambda b: b.real,
    "imag": lambda b: b.imag,
    "slice-of-real": lambda b: b.real[1:],
}


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("part", COMPLEX_PARTS)
def test_a_write_to_a_part_of_a_complex_buffer_reaches_it(part: str, entry: str) -> None:
    _agrees(_writes_through(COMPLEX_PARTS[part]), entry, _complex_buffer)


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("buffer", BUFFERS)
def test_a_write_then_a_read_of_the_buffer_starts_each_call_from_the_right_bytes(
    buffer: str, entry: str
) -> None:
    """The read falls back to eager (the recording has the value from before the
    write), and eager runs the write itself: the trace-time write has to be gone
    by then, or the first call counts it twice."""

    def body(model: nn.Module, x: Tensor) -> Tensor:
        model.b[1:].add_(x.mean())
        return x * model.b.sum()

    seen = _agrees(body, entry, BUFFERS[buffer])
    if buffer != "plain":  # the plain buffer's value follows the view's write
        assert not seen.compiled


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("buffer", BUFFERS)
def test_a_value_read_through_the_written_view_needs_no_fallback(buffer: str, entry: str) -> None:
    def body(model: nn.Module, x: Tensor) -> Tensor:
        row = model.b[1:]
        row.add_(x[1:].mean())
        return x * row.sum()

    assert _agrees(body, entry, BUFFERS[buffer]).compiled


# -- views the call holds ------------------------------------------------------


def _with_a_held_view(model: nn.Module) -> None:
    model.register_buffer("b", lucid.zeros(5, 4, device=COMPILE_DEVICE)[1:4])
    model.held = model.b[:2]


@pytest.mark.parametrize("entry", ENTRIES)
def test_a_held_view_read_only_before_a_write_stays_compiled(entry: str) -> None:
    """LCD-837's fallback covers a read *after* the write.  Refusing every view
    a write leaves stale would send this correct case eager for nothing."""

    def body(model: nn.Module, x: Tensor) -> Tensor:
        a = x * model.held.sum()
        model.b[1:].add_(x.mean())
        return a

    assert _agrees(body, entry, _with_a_held_view).compiled


@pytest.mark.parametrize("entry", ENTRIES)
def test_a_held_view_written_directly_reaches_the_buffer(entry: str) -> None:
    def body(model: nn.Module, x: Tensor) -> Tensor:
        model.held.add_(x.mean())
        return x * 1.0

    assert _agrees(body, entry, _with_a_held_view).compiled


@pytest.mark.parametrize("entry", ENTRIES)
def test_a_held_view_read_after_a_write_through_another_answers_as_eager(entry: str) -> None:
    def body(model: nn.Module, x: Tensor) -> Tensor:
        model.b[1:].add_(x.mean())
        return x * model.held.sum()

    assert not _agrees(body, entry, _with_a_held_view).compiled


@pytest.mark.parametrize("entry", ENTRIES)
def test_disjoint_views_written_in_one_call_stay_compiled(entry: str) -> None:
    def body(model: nn.Module, x: Tensor) -> Tensor:
        head, tail = model.b[:1], model.b[2:]
        head.add_(x.mean())
        tail.add_(x.max())
        return x * 1.0

    assert _agrees(body, entry, _a_run_of_a_larger_buffer).compiled


def _second_reads_the_first(head: Tensor, mid: Tensor, x: Tensor) -> None:
    head.add_(x.mean())
    mid.mul_(2.0)


def _second_only_writes(head: Tensor, mid: Tensor, x: Tensor) -> None:
    head.add_(x.mean())
    mid.fill_(2.0)


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("write", [_second_reads_the_first, _second_only_writes])
def test_overlapping_views_written_in_one_call_end_as_eager_ends(
    write: Callable[[Tensor, Tensor, Tensor], None], entry: str
) -> None:
    """Neither is the whole buffer, and carrying ``head`` back would undo ``mid``
    where they meet: the call runs eager."""

    def body(model: nn.Module, x: Tensor) -> Tensor:
        write(model.b[:2], model.b[1:3], x)
        return x * 1.0

    assert not _agrees(body, entry, _a_run_of_a_larger_buffer).compiled


@pytest.mark.parametrize("entry", ENTRIES)
def test_a_buffer_that_is_its_whole_buffer_is_carried_once(entry: str) -> None:
    """Its value already holds the write, re-derived by the family: the views
    are not carried besides it, and it does not matter that the second write
    leaves the first view stale.  It compiled before, and has to keep compiling."""

    def body(model: nn.Module, x: Tensor) -> Tensor:
        _second_only_writes(model.b[:2], model.b[1:3], x)
        return x * 1.0

    assert _agrees(body, entry, _plain).compiled


def test_a_view_of_a_whole_buffer_is_not_carried_beside_it() -> None:
    model = _twin(_writes_through(VIEWS["slice"]), _plain)
    x = lucid.ones(*SHAPE, device=COMPILE_DEVICE)
    with lucid.no_grad(), _tracing(watch=[x], module=model) as tracer:
        model(x)
    assert [impl for _tid, impl in tracer.outside_writes()] == [model.b._impl]
    tracer.restore_outside_writes()


def test_a_view_of_part_of_a_buffer_is_carried_itself() -> None:
    model = _twin(lambda m, x: _write_and_keep(m, x), _a_run_of_a_larger_buffer)
    x = lucid.ones(*SHAPE, device=COMPILE_DEVICE)
    with lucid.no_grad(), _tracing(watch=[x], module=model) as tracer:
        model(x)
    written = [impl for _tid, impl in tracer.outside_writes()]
    assert written == [model.kept._impl]
    tracer.restore_outside_writes()
    assert model.b.tolist() == [[0.0] * 4] * 3


def _write_and_keep(model: nn.Module, x: Tensor) -> Tensor:
    model.kept = model.b[1:]
    model.kept.add_(x.mean())
    return x * 1.0


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("buffer", ["plain", "run-of-larger"])
def test_the_whole_tensor_written_after_a_view_of_it_ends_with_its_own_value(
    buffer: str, entry: str
) -> None:
    """The later copy of the buffer already holds the view's write: putting it
    back has to leave the buffer from before both.  A buffer that is part of its
    bytes runs eager, as the copy leaves the view stale and nothing else holds
    the view's write."""

    def body(model: nn.Module, x: Tensor) -> Tensor:
        model.b[1:].add_(x.mean())
        model.b.copy_(x * 2.0)
        return x * 1.0

    assert _agrees(body, entry, BUFFERS[buffer]).compiled is (buffer == "plain")


class _Segments(nn.Module):
    """A first segment that writes through a view of a buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("b", lucid.zeros(5, 4, device=COMPILE_DEVICE)[1:4])
        self.lin = nn.Linear(SHAPE[-1], SHAPE[-1])

    def forward(self, x: Tensor) -> Tensor:
        self.b[1:].add_(x.mean())
        return self.lin(x)


def _two_segments() -> nn.Module:
    lucid.manual_seed(0)
    return nn.Sequential(_Segments(), nn.Linear(SHAPE[-1], 2)).to(COMPILE_DEVICE)


def test_a_step_that_cannot_write_back_runs_eager_and_writes_once_per_call() -> None:
    """Segments have no way to carry a write, so a view's counts like a direct one."""
    model, ref = _two_segments(), _two_segments()
    step = lucid.compile.make_step(model, _loss, segments=2)
    for x in _inputs():
        got = step(x)
        got.backward()
        want = _loss(ref(x))
        want.backward()
        assert _close([float(got.item())], [float(want.item())])
        assert model[0].b.tolist() == ref[0].b.tolist()


def test_a_step_with_no_eager_path_refuses_and_leaves_the_buffer_alone() -> None:
    model = _two_segments()
    with pytest.raises(RuntimeError, match="wrote in place"):
        lucid.compile.compiled_step(model, _inputs()[0], _loss)
    assert model[0].b.tolist() == [[0.0] * 4] * 3


# -- what the call computed itself owes nobody ---------------------------------


@pytest.mark.parametrize("grad", [False, True], ids=["no-grad", "grad"])
@pytest.mark.parametrize("write", MADE_WRITES)
def test_a_write_through_a_view_of_what_the_call_computed_is_owed_to_nobody(
    write: str, grad: bool
) -> None:
    x = lucid.ones(*SHAPE, device=COMPILE_DEVICE, requires_grad=grad)
    with lucid.enable_grad() if grad else lucid.no_grad():
        with _tracing(watch=[x]) as tracer:
            MADE_WRITES[write](x * 2.0, x)
    assert list(tracer.outside_writes()) == []


# -- a view of an argument is the argument --------------------------------------


def _through_a_view_of_the_argument(_model: nn.Module, x: Tensor) -> Tensor:
    x[:1].add_(1.0)
    return x * 2.0


def _one_call(entry: str, model: _Model, *, compiled: bool) -> Callable[[Tensor], Tensor]:
    """What ``entry`` runs per call, the same model's eager step when not ``compiled``."""
    if entry == "module":
        return lucid.compile(model) if compiled else model
    opt = optim.SGD(model.parameters(), lr=1e-4)
    if entry == "fused_step" and compiled:
        return lucid.compile.fused_step(model, _loss, opt)
    if entry == "make_step" and compiled:
        return lucid.compile.make_step(model, _loss)

    def eager(x: Tensor) -> Tensor:
        opt.zero_grad()
        loss = _loss(model(x))
        loss.backward()
        if entry == "fused_step":
            opt.step()
        return loss

    return eager


def _one_argument(entry: str, body: Body, x: Tensor, *, compiled: bool) -> list[list[float]]:
    """``body`` on ``x`` three times; what each call answered and left in ``x``."""
    call = _one_call(entry, _twin(body, None), compiled=compiled)
    seen: list[list[float]] = []
    with lucid.no_grad() if entry == "module" else lucid.enable_grad():
        for _ in range(3):
            seen.append(_floats(call(x)))
            seen.append(_floats(x))
    return seen


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("arg", ["whole", "part"])
def test_a_write_through_a_view_of_an_argument_runs_eager(entry: str, arg: str) -> None:
    """The next call brings another tensor, so a compiled call cannot carry the
    write: it has to land in whichever tensor this call was given."""

    def make() -> Tensor:
        if arg == "whole":
            return lucid.zeros(*SHAPE, device=COMPILE_DEVICE)
        return lucid.zeros(5, 4, device=COMPILE_DEVICE)[1:4]

    got = _one_argument(entry, _through_a_view_of_the_argument, make(), compiled=True)
    want = _one_argument(entry, _through_a_view_of_the_argument, make(), compiled=False)
    for g, w in zip(got, want, strict=True):
        assert _close(g, w), f"{got} vs {want}"


# -- a Metal shared buffer under two names --------------------------------------


class _SharedAlias(nn.Module):
    """Two aliases of one shared buffer, which share no view family."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("m", lucid.metal.shared_tensor(SHAPE).to(COMPILE_DEVICE))
        self.alias = self.m.to("cpu").to(COMPILE_DEVICE)


class _WritesThenReadsAlias(_SharedAlias):
    def forward(self, x: Tensor) -> Tensor:
        self.m.add_(x.mean())
        return self.alias * 1.0


class _ReadsAliasThenWrites(_SharedAlias):
    def forward(self, x: Tensor) -> Tensor:
        before = self.alias * 1.0
        self.m.add_(x.mean())
        return before


class _WritesAliasThenReadsBuffer(_SharedAlias):
    def forward(self, x: Tensor) -> Tensor:
        self.alias.add_(x.mean())
        return self.m * 1.0


class _WritesTheBufferThenReadsIt(_SharedAlias):
    def forward(self, x: Tensor) -> Tensor:
        self.m.add_(x.mean())
        return self.m * 1.0


@pytest.mark.parametrize(
    ("kind", "compiles"),
    [
        (_WritesThenReadsAlias, False),
        (_WritesAliasThenReadsBuffer, False),
        (_ReadsAliasThenWrites, True),
        (_WritesTheBufferThenReadsIt, True),
    ],
)
def test_a_shared_buffer_under_another_name_reads_what_the_call_wrote(
    kind: type[_SharedAlias], compiles: bool
) -> None:
    def history(model: _SharedAlias, run: Callable[[Tensor], Tensor]) -> list[list[float]]:
        with lucid.no_grad():
            return [_floats(run(x)) + _floats(model.m) for x in _inputs()]

    eager_model = kind()
    want = history(eager_model, eager_model)
    model = kind()
    compiled = lucid.compile(model)
    got = history(model, compiled)
    assert got == want
    assert (not compiled.cache_info()["eager_only"]) is compiles


# -- the loss sees a tensor the call made, too -----------------------------------


def test_a_made_tensor_written_in_a_loss_starts_over() -> None:
    def loss(out: Tensor) -> Tensor:
        target = lucid.tensor([1.0, 2.0], device=COMPILE_DEVICE)
        target.add_(out.mean().detach())
        return F.mse_loss(out, target.expand(*out.shape))

    def body(_model: nn.Module, x: Tensor) -> Tensor:
        return x

    model, ref = _twin(body, None), _twin(body, None)
    step = lucid.compile.make_step(model, loss)
    for x in _inputs():
        got = step(x)
        want = loss(ref(x))
        assert abs(float(got.item()) - float(want.item())) <= 1e-4 * (1.0 + abs(float(want.item())))
