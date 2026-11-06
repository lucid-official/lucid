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

import functools
from collections.abc import Callable, Iterator
from typing import NamedTuple, cast

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
SCALES = (1.0, 5.0, -2.0, 3.0)

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
    #: Executables, and whether any signature ran eager; for a compiled run.
    entries: int = 0
    eager: bool = False


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
    eager = bool(info["eager_only"])
    return _Seen(answers, states, info["entries"] == 1 and not eager, info["entries"], eager)


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
    eager = bool(step.eager_only.snapshot())
    return _Seen(answers, states, len(step.cache) == 1 and not eager, len(step.cache), eager)


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
    info = step.cache_info()
    eager = bool(info["eager_only"])
    return _Seen(answers, states, not eager, len(info["keys"]), eager)


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
    "logspace": lambda: lucid.logspace(0.0, 1.0, 12, device=COMPILE_DEVICE).reshape(*SHAPE),
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
    # The backward's lowering refuses these two wherever they appear, read or written.
    if not (factory in ("empty", "logspace") and entry != "module"):
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


# -- the call made it, and something goes on holding it -------------------------


def _a_box(model: nn.Module) -> None:
    model.box = {}


def _kept_in_a_box(make: Make) -> Body:
    def body(model: nn.Module, x: Tensor) -> Tensor:
        if "h" not in model.box:
            model.box["h"] = make()
        model.box["h"].add_(x.mean())
        return x * model.box["h"].sum()

    return body


def _a_view_kept_in_a_box(make: Make) -> Body:
    def body(model: nn.Module, x: Tensor) -> Tensor:
        if "h" not in model.box:
            model.box["h"] = make()[1:]
        model.box["h"].add_(x.mean())
        return x * model.box["h"].sum()

    return body


def _an_empty_buffer(model: nn.Module) -> None:
    model.register_buffer("h", None)


def _kept_as_a_buffer(make: Make) -> Body:
    def body(model: nn.Module, x: Tensor) -> Tensor:
        if model.h is None:
            model.h = make()
        model.h.add_(x.mean())
        return x * model.h.sum()

    return body


#: Each makes the tensor once, on the first call, and goes on from it: eager
#: accumulates, and a compiled call that started it over would not.
KEPT: dict[str, tuple[Callable[[Make], Body], Setup]] = {
    "in-a-box": (_kept_in_a_box, _a_box),
    "view-in-a-box": (_a_view_kept_in_a_box, _a_box),
    "as-a-buffer": (_kept_as_a_buffer, _an_empty_buffer),
}


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("kept", KEPT)
@pytest.mark.parametrize("factory", [*FACTORIES])
def test_a_tensor_the_call_made_and_holds_goes_on_from_the_last_call(
    factory: str, kept: str, entry: str
) -> None:
    """The tensor looks like the call's own, but nothing about it says the next
    call starts over: Python holds it past the call, and the write reads the
    input.  So the call runs eager, and eager is the first to write it, so the
    trace's write has to be gone."""
    wrap, setup = KEPT[kept]
    seen = _agrees(wrap(FACTORIES[factory]), entry, setup)
    assert not seen.compiled


def _adds_a_constant_to_what_it_keeps(model: nn.Module, x: Tensor) -> Tensor:
    if "h" not in model.box:
        model.box["h"] = lucid.zeros(SHAPE[-1], device=COMPILE_DEVICE)
    model.box["h"].add_(1.0)
    return x * model.box["h"]


@pytest.mark.parametrize("entry", ENTRIES)
def test_a_constant_added_to_a_tensor_the_model_keeps_goes_on_from_the_last_call(
    entry: str,
) -> None:
    """The first call leaves the tensor as a constant table; the model now holds a
    tensor it did not before, so the next call is traced again and finds it from
    outside."""
    _agrees(_adds_a_constant_to_what_it_keeps, entry, _a_box)


def _adds_a_constant_to_what_a_closure_keeps() -> Body:
    box: dict[str, Tensor] = {}

    def body(_model: nn.Module, x: Tensor) -> Tensor:
        if "h" not in box:
            box["h"] = lucid.zeros(SHAPE[-1], device=COMPILE_DEVICE)
        box["h"].add_(1.0)
        return x * box["h"]

    return body


@pytest.mark.xfail(
    strict=True,
    reason="LCD-864: a constant added to a lazily kept tensor on every call reads as a "
    "constant table, so the compiled call holds its trace-time value",
)
def test_a_constant_added_to_a_tensor_a_closure_keeps_goes_on_from_the_last_call() -> None:
    got = _run_module(_twin(_adds_a_constant_to_what_a_closure_keeps(), None))
    want = _run_eager_module(_twin(_adds_a_constant_to_what_a_closure_keeps(), None))
    assert all(_close(g, w) for g, w in zip(got.answers, want.answers, strict=True))


@pytest.mark.parametrize("factory", MOVED)
def test_a_tensor_the_call_made_moved_and_holds_goes_on_from_the_last_call(factory: str) -> None:
    """With folding off a move is an op, and its result an op's output like ``x * 2``:
    told apart from what the call made, so left to the lazy-attribute issue."""
    _agrees(_kept_in_a_box(MOVED[factory]), "module", _a_box)


def _a_lazy_mask(model: nn.Module) -> None:
    model.lazy = None


def _builds_its_mask_once(model: nn.Module, x: Tensor) -> Tensor:
    if model.lazy is None:
        mask = lucid.zeros(*SHAPE, device=COMPILE_DEVICE)
        mask[0, 0] = 1.0
        mask[2] = 3.0
        model.lazy = mask
    return x * model.lazy


@pytest.mark.parametrize("entry", ENTRIES)
def test_a_mask_built_once_and_kept_holds_its_values_on_every_call(entry: str) -> None:
    """Nothing writes the mask after the first call, so putting it back for the
    next one to write would leave it empty for good.  Its writes are constants,
    so the call stays compiled: one executable before the mask exists and one
    after, as the model's tensor attributes are part of its key."""
    seen = _agrees(_builds_its_mask_once, entry, _a_lazy_mask)
    assert not seen.eager and seen.entries == 2


def _a_cache_slot(model: nn.Module) -> None:
    model.register_buffer("scale", lucid.full((SHAPE[-1],), 2.0, device=COMPILE_DEVICE))
    model.cache = None


def _fills_a_cache_from_its_state(model: nn.Module, x: Tensor) -> Tensor:
    if model.cache is None:
        model.cache = lucid.zeros(SHAPE[-1], device=COMPILE_DEVICE)
        model.cache.copy_(model.scale * 3.0)
    return x * model.cache


@pytest.mark.parametrize("entry", ENTRIES)
def test_a_cache_filled_once_from_the_model_s_state_holds_its_values(entry: str) -> None:
    """The write reads the model's state, so the trace cannot call it a constant.
    Run again eagerly with the cache put back, the call would find it built and
    answer from the empty one (and a training step would step on that answer for
    good); so the trace-time call stands as the call."""
    _agrees(_fills_a_cache_from_its_state, entry, _a_cache_slot)


class _TracksItsStatistics(nn.Module):
    """A cache filled once from the model's state, beside a batch norm in training."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("scale", lucid.full((SHAPE[-1],), 2.0, device=COMPILE_DEVICE))
        self.cache: Tensor | None = None
        self.norm = nn.BatchNorm1d(SHAPE[-1])
        self.lin = nn.Linear(SHAPE[-1], 2)

    def forward(self, x: Tensor) -> Tensor:
        if self.cache is None:
            self.cache = lucid.zeros(SHAPE[-1], device=COMPILE_DEVICE)
            self.cache.copy_(self.scale * 3.0)
        # Rows that differ, or the norm answers zeros from any scale of them.
        rows = lucid.arange(float(SHAPE[0]), device=COMPILE_DEVICE).reshape(SHAPE[0], 1)
        return self.lin(self.norm((x + rows) * self.cache))


@pytest.mark.xfail(
    strict=True,
    reason="a traced call leaves batch norm's statistics alone, so it cannot stand as the "
    "call; the eager rerun answers its first call from the cache put back (to be filed)",
)
@pytest.mark.parametrize("entry", ["make_step", "fused_step"])
def test_a_cache_filled_beside_a_batch_norm_holds_its_values(entry: str) -> None:
    run, run_eager = ENTRIES[entry]

    def build() -> _Model:
        lucid.manual_seed(0)
        return cast(_Model, _TracksItsStatistics().to(COMPILE_DEVICE))

    got, want = run(build()), run_eager(build())
    for i in range(len(SCALES)):
        assert _close(got.answers[i], want.answers[i]), f"call {i}"


def _rebuilds_its_total_every_call(model: nn.Module, x: Tensor) -> Tensor:
    model.total = lucid.zeros(SHAPE[-1], device=COMPILE_DEVICE)
    model.total.add_(x.mean())
    return x * model.total.sum()


@pytest.mark.parametrize("entry", ENTRIES)
def test_a_tensor_made_anew_every_call_and_kept_starts_over_every_call(entry: str) -> None:
    """Held after the call like a cached one, but every call makes another: taking
    it for state that goes on from the last call would add up the calls.  From
    inside one trace the two look the same, so the call runs eager."""
    assert not _agrees(_rebuilds_its_total_every_call, entry, None).compiled


def _fills_a_local(model: nn.Module, x: Tensor) -> Tensor:
    y = lucid.zeros(*SHAPE, device=COMPILE_DEVICE)
    y[1] = x[1] * 2.0
    return x + y


@pytest.mark.parametrize("entry", ENTRIES)
def test_a_tensor_the_call_makes_fills_and_drops_stays_compiled(entry: str) -> None:
    assert _agrees(_fills_a_local, entry, None).compiled


def _a_dropout_into_what_it_keeps(model: nn.Module, x: Tensor) -> Tensor:
    dropped = F.dropout(x, p=0.5, training=True)
    if "h" not in model.box:
        model.box["h"] = lucid.zeros(SHAPE[-1], device=COMPILE_DEVICE)
    model.box["h"].add_(dropped.mean())
    return dropped * model.box["h"].sum()


def _draws(body: Body, run: Run) -> _Seen:
    lucid.manual_seed(7)
    model = _Model(body, _a_box).to(COMPILE_DEVICE)
    return run(model)


def test_the_draws_of_a_call_that_runs_eager_for_a_kept_write_are_eager_s() -> None:
    """The trace draws, and the eager call that follows has to draw the same: the
    compile attempt puts the streams back.  Every later call is eager too."""
    got = _draws(_a_dropout_into_what_it_keeps, _run_module)
    want = _draws(_a_dropout_into_what_it_keeps, _run_eager_module)
    assert not got.compiled
    for i, (g, w) in enumerate(zip(got.answers, want.answers, strict=True)):
        assert _close(g, w), f"call {i}"


_CACHE_ROW = [0.0, 2.0, 0.0, -1.0]


@functools.lru_cache(maxsize=4)
def _a_cached_table(n: int, device: str) -> Tensor:
    table = lucid.zeros(n, device=device)
    table[1] = _CACHE_ROW[1]
    table[3] = _CACHE_ROW[3]
    return table


def _reads_a_cached_table(_model: nn.Module, x: Tensor) -> Tensor:
    return x * _a_cached_table(SHAPE[-1], COMPILE_DEVICE)


@pytest.mark.parametrize("entry", ENTRIES)
def test_a_table_a_cache_keeps_from_a_trace_is_whole_for_every_later_call(entry: str) -> None:
    """A module-level cache built inside the trace keeps the tensor for every
    later call in the process, compiled or not.  Twins share the cache, so they
    cannot tell a broken one; the table's values can."""
    _a_cached_table.cache_clear()
    assert _agrees(_reads_a_cached_table, entry, None).compiled
    assert _a_cached_table(SHAPE[-1], COMPILE_DEVICE).tolist() == _CACHE_ROW
    with lucid.no_grad():
        x = lucid.ones(*SHAPE, device=COMPILE_DEVICE)
        assert (x * _a_cached_table(SHAPE[-1], COMPILE_DEVICE)).tolist()[0] == _CACHE_ROW
    _a_cached_table.cache_clear()


def test_a_compiled_bicubic_resize_leaves_every_later_resize_right() -> None:
    """``interpolate`` caches the matrix it builds inside the trace.  Put back for
    an eager rerun that never rewrote it, every later bicubic resize in the
    process answered zeros, eager included."""
    from lucid.nn.functional import _interpolate

    lucid.manual_seed(0)
    x = lucid.randn(1, 2, 17, 13, device=COMPILE_DEVICE)
    size = (68, 85)
    _interpolate._dense_matrix.cache_clear()
    with lucid.no_grad():
        want = F.interpolate(x, size=size, mode="bicubic")
        _interpolate._dense_matrix.cache_clear()
        compiled = lucid.compile(lambda a: F.interpolate(a, size=size, mode="bicubic"))
        got = compiled(x)
        after = F.interpolate(x, size=size, mode="bicubic")
        again = compiled(x * 2.0)
    assert float(want.abs().mean().item()) > 0.1
    assert _close(_floats(got), _floats(want))
    assert _close(_floats(after), _floats(want))
    assert _close(_floats(again), _floats(want * 2.0))
    info = compiled.cache_info()
    assert info["entries"] == 1 and not info["eager_only"]
    _interpolate._dense_matrix.cache_clear()


def _returns_it(y: Tensor) -> object:
    return y


def _returns_it_in_a_tuple(y: Tensor) -> object:
    return (y, y * 2.0)


def _returns_a_view_of_it(y: Tensor) -> object:
    return y[1:]


def _leaves(value: object) -> list[list[float]]:
    return [_floats(value)] if isinstance(value, Tensor) else [_floats(t) for t in value]


@pytest.mark.parametrize("result", [_returns_it, _returns_it_in_a_tuple, _returns_a_view_of_it])
@pytest.mark.parametrize("factory", FACTORIES)
def test_a_tensor_the_call_made_and_returns_stays_compiled(
    factory: str, result: Callable[[Tensor], object]
) -> None:
    """The result is on the caller's hands by the time anything could write it
    again, and the next call hands back another."""
    make = FACTORIES[factory]

    def fn(x: Tensor) -> object:
        y = make()
        y.add_(x)
        return result(y)

    compiled = lucid.compile(fn)
    with lucid.no_grad():
        for x in _inputs():
            got, want = _leaves(compiled(x)), _leaves(fn(x))
            assert all(_close(g, w) for g, w in zip(got, want, strict=True)), f"{got} vs {want}"
    info = compiled.cache_info()
    assert info["entries"] == 1 and not info["eager_only"]


class _WritesAParameterIntoALiteral(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(lucid.ones(*SHAPE))

    def forward(self, x: Tensor) -> Tensor:
        y = lucid.tensor(ROWS, device=COMPILE_DEVICE)
        y.add_(self.w * x)
        return y


def test_a_made_tensor_in_the_graph_is_not_held_by_the_graph() -> None:
    """Autograd nodes hold what they saved; that is not a holder anyone can read."""

    def build() -> nn.Module:
        lucid.manual_seed(0)
        return _WritesAParameterIntoALiteral().to(COMPILE_DEVICE)

    model, ref = build(), build()
    step = lucid.compile.make_step(model, _loss)
    for x in _inputs():
        got = step(x)
        got.backward()
        want = _loss(ref(x))
        want.backward()
        assert _close([float(got.item()), *_floats(model.w.grad)], [
            float(want.item()),
            *_floats(ref.w.grad),
        ])
    assert len(step.cache) == 1 and not step.eager_only.snapshot()


# -- a write through a view whose elements overlap ------------------------------


def _a_vector(model: nn.Module) -> None:
    model.register_buffer("vec", lucid.zeros(4, device=COMPILE_DEVICE))


def _part_of_a_vector(model: nn.Module) -> None:
    model.register_buffer("vec", lucid.zeros(5, device=COMPILE_DEVICE)[1:])


OVERLAPPING: dict[str, Callable[[Tensor], Tensor]] = {
    "expand": lambda v: v[0:1].expand(4),
    "unfold": lambda v: v.unfold(0, 2, 1),
}


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("view", OVERLAPPING)
@pytest.mark.parametrize("buffer", [_a_vector, _part_of_a_vector], ids=["whole", "part"])
def test_a_write_through_overlapping_elements_ends_as_eager_ends(
    buffer: Setup, view: str, entry: str
) -> None:
    """``copy_from`` refuses such a view, so it can neither be carried back nor
    put back.  The buffer it reads can."""
    make = OVERLAPPING[view]

    def body(model: nn.Module, x: Tensor) -> Tensor:
        a = x * model.vec.sum()
        make(model.vec).fill_(2.0)
        return a

    _agrees(body, entry, buffer)


# -- what one write costs each call ----------------------------------------------


def _a_big_buffer(model: nn.Module) -> None:
    model.register_buffer("big", lucid.zeros(64, 4, device=COMPILE_DEVICE))


def _one_setitem(model: nn.Module, m: Tensor) -> None:
    model.big[0] = m


def _three_setitems(model: nn.Module, m: Tensor) -> None:
    for i in range(3):
        model.big[i] = m


def _slice_iadd(model: nn.Module, m: Tensor) -> None:
    model.big[0:10] += m


def _whole_slice_add(model: nn.Module, m: Tensor) -> None:
    for _ in range(8):
        model.big[:].add_(m)


def _many_setitems(model: nn.Module, m: Tensor) -> None:
    for i in range(300):
        model.big[i % 64] = m


TEMPORARIES: dict[str, Callable[[nn.Module, Tensor], None]] = {
    "one-setitem": _one_setitem,
    "three-setitems": _three_setitems,
    "slice-iadd": _slice_iadd,
    "whole-slice-add": _whole_slice_add,
    "many-setitems": _many_setitems,
}


@pytest.mark.parametrize("write", TEMPORARIES)
def test_a_buffer_costs_one_copy_and_one_output_however_many_temporaries_reach_it(
    write: str,
) -> None:
    """Every ``buf[i] = v`` goes through a view and a whole-buffer reshape.  Each
    would be a copy at the trace and an output and a full copy on every run: the
    cost suite cannot see that, as it counts the eager path."""
    body = TEMPORARIES[write]

    def forward(model: nn.Module, x: Tensor) -> Tensor:
        body(model, x.mean())
        return x * 1.0

    model = _twin(forward, _a_big_buffer)
    x = lucid.ones(*SHAPE, device=COMPILE_DEVICE)
    with lucid.no_grad(), _tracing(watch=[x], module=model) as tracer:
        forward(model, x)
    assert tracer.outside_write_count() == 1
    assert [impl for _tid, impl in tracer.outside_writes()] == [model.big._impl]
    tracer.restore_outside_writes()
    assert _agrees(forward, "module", _a_big_buffer).compiled


def test_the_compiled_paths_keep_no_copy_of_what_the_trace_wrote() -> None:
    model = _twin(lambda m, x: _write_and_keep(m, x), _plain)
    x = lucid.ones(*SHAPE, device=COMPILE_DEVICE)
    with lucid.no_grad(), _tracing(watch=[x], module=model) as tracer:
        model(x)
    assert not tracer.keep_written_values
    tracer.restore_outside_writes()
    held = {id(impl) for impl in tracer.retained_values.values()}
    assert id(model.b._impl) in held  # the tensor itself, not a copy of its value


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


def _shared_names(model: nn.Module) -> None:
    model.register_buffer("m", lucid.metal.shared_tensor(SHAPE).to(COMPILE_DEVICE))
    model.alias = model.m.to("cpu").to(COMPILE_DEVICE)
    model.row = model.alias[1]
    model.part = model.m[1]


def _shared_add(model: nn.Module, x: Tensor) -> None:
    model.m.add_(x.mean())


def _shared_copy(model: nn.Module, x: Tensor) -> None:
    model.m.copy_(x)


def _shared_setitem(model: nn.Module, x: Tensor) -> None:
    model.m[:] = x


def _shared_through_a_view(model: nn.Module, x: Tensor) -> None:
    model.part.add_(x.mean())


SHARED_WRITES: dict[str, Callable[[nn.Module, Tensor], None]] = {
    "add_": _shared_add,
    "copy_": _shared_copy,
    "setitem": _shared_setitem,
    "through-a-view": _shared_through_a_view,
}

SHARED_READS: dict[str, Callable[[nn.Module], Tensor]] = {
    "alias": lambda model: model.alias * 1.0,
    "view-of-the-alias": lambda model: model.row * 1.0,
}


def test_the_names_are_one_buffer() -> None:
    """Without this, the cases below would be blind."""
    model = _twin(lambda m, x: x, _shared_names)
    model.m.fill_(3.0)
    assert model.alias.tolist()[1][0] == 3.0
    assert model.row.tolist()[0] == 3.0


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("read", SHARED_READS)
@pytest.mark.parametrize("write", SHARED_WRITES)
def test_a_shared_buffer_under_any_name_reads_what_the_call_wrote(
    write: str, read: str, entry: str
) -> None:
    """A view holds the buffer as much as the tensor it came from does, in a
    family of its own: the write reaches it, and no stale view says so."""
    writes, reads = SHARED_WRITES[write], SHARED_READS[read]

    def body(model: nn.Module, x: Tensor) -> Tensor:
        writes(model, x)
        return reads(model)

    assert not _agrees(body, entry, _shared_names).compiled


def _writer_outliving_its_base(model: nn.Module) -> None:
    base = lucid.metal.shared_tensor(SHAPE).to(COMPILE_DEVICE)
    model.alias = base.to("cpu").to(COMPILE_DEVICE)
    model.part = base[1]


@pytest.mark.parametrize("entry", ENTRIES)
def test_a_write_through_a_view_whose_base_is_gone_is_seen_by_the_alias(entry: str) -> None:
    def body(model: nn.Module, x: Tensor) -> Tensor:
        model.part.add_(x.mean())
        return model.alias * 1.0

    _agrees(body, entry, _writer_outliving_its_base)


@pytest.mark.parametrize("entry", ENTRIES)
def test_a_name_read_only_before_the_write_stays_compiled(entry: str) -> None:
    def body(model: nn.Module, x: Tensor) -> Tensor:
        before = model.row * 1.0
        model.m.add_(x.mean())
        return before

    assert _agrees(body, entry, _shared_names).compiled


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
