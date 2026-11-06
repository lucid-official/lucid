"""A tensor a traced call makes and something goes on holding (LCD-853).

The trace sees the tensor made, so it takes it for the call's own, to start
over on every call; whoever holds it expects each call to go on from the last,
or to find it built.  The entry points answer two ways:

* ``lucid.compile``, ``make_step``, a segmented ``make_step`` and
  ``fused_step`` let the trace-time call stand as the call when such a tensor
  is written from anything but constants, and run the calls after it eagerly.
  A segmented step lets the trace of the segment that wrote it stand for that
  segment and runs the segments after it eagerly.  It used to put back only a
  write that read the tensor, for an eager rerun that a tensor built on the
  first call only was never written by again, and compiled an overwrite, so a
  buffer filled from the first batch was filled again from every batch
  (LCD-876).
* ``compiled_step`` traces every call afresh, and the trace's writes are that
  call's: the first call is right whatever it writes, and from the second the
  tensor is the model's state, a write to which it refuses as to any buffer.

Every case runs four calls on inputs that differ in sign and size against a
twin running eagerly, and compares the answers and the held tensor after each.
"""

import functools
from collections.abc import Callable, Iterator

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
SHAPE = (3, 4)
#: (scale, offset) per call: masks and positions read off the input differ
#: from call to call, the last two all of one sign in their first column.
CALLS = ((1.0, 0.0), (-0.5, 0.2), (0.5, -0.6), (0.25, 0.4))


@pytest.fixture(autouse=True)
def _fresh_session(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(fallback, "_NOTICED", set())
    _C_engine.compile.session_cache_clear()
    yield


def _inputs() -> list[Tensor]:
    base = (lucid.arange(12.0, device=COMPILE_DEVICE).reshape(*SHAPE) - 5.0) / 6.0
    return [base * scale + offset for scale, offset in CALLS]


def _made() -> Tensor:
    return lucid.ones(*SHAPE, device=COMPILE_DEVICE)


def _floats(t: Tensor) -> list[float]:
    return [float(v) for v in t.reshape(-1).tolist()]


# -- where the tensor lives between calls ---------------------------------------


class _Holder:
    """Holds the tensor, made inside the first call's trace."""

    def attach(self, owner: nn.Module) -> None:
        """Give ``owner`` what the holder keeps in it, if anything."""

    def fetch(self, owner: nn.Module) -> tuple[Tensor, bool]:
        """The tensor, and whether this call made it."""
        raise NotImplementedError

    def peek(self, owner: nn.Module) -> Tensor:
        """The tensor, without making it."""
        raise NotImplementedError


class _InABox(_Holder):
    """A dict on the module."""

    def attach(self, owner: nn.Module) -> None:
        owner.box = {}

    def fetch(self, owner: nn.Module) -> tuple[Tensor, bool]:
        new = "t" not in owner.box
        if new:
            owner.box["t"] = _made()
        return owner.box["t"], new

    def peek(self, owner: nn.Module) -> Tensor:
        return owner.box["t"]


class _AsABuffer(_Holder):
    """A buffer registered empty and filled by the first call."""

    def attach(self, owner: nn.Module) -> None:
        owner.register_buffer("t", None)

    def fetch(self, owner: nn.Module) -> tuple[Tensor, bool]:
        new = owner.t is None
        if new:
            owner.t = _made()
        return owner.t, new

    def peek(self, owner: nn.Module) -> Tensor:
        return owner.t


class _InACell(_Holder):
    """A closure's cell."""

    def __init__(self) -> None:
        held: Tensor | None = None

        def fetch() -> tuple[Tensor, bool]:
            nonlocal held
            new = held is None
            if held is None:
                held = _made()
            return held, new

        def peek() -> Tensor:
            assert held is not None
            return held

        self._fetch, self._peek = fetch, peek

    def fetch(self, owner: nn.Module) -> tuple[Tensor, bool]:
        return self._fetch()

    def peek(self, owner: nn.Module) -> Tensor:
        return self._peek()


class _InAGenerator(_Holder):
    """A suspended generator's frame."""

    def __init__(self) -> None:
        def frames() -> Iterator[tuple[Tensor, bool]]:
            held = _made()
            yield held, True
            while True:
                yield held, False

        self._frames = frames()

    def fetch(self, owner: nn.Module) -> tuple[Tensor, bool]:
        return next(self._frames)

    def peek(self, owner: nn.Module) -> Tensor:
        frame = self._frames.gi_frame
        assert frame is not None
        return frame.f_locals["held"]


class _InAnLruCache(_Holder):
    """An ``lru_cache`` of the function that makes it."""

    def __init__(self) -> None:
        self._table = functools.lru_cache(maxsize=1)(_made)

    def fetch(self, owner: nn.Module) -> tuple[Tensor, bool]:
        new = self._table.cache_info().currsize == 0
        return self._table(), new

    def peek(self, owner: nn.Module) -> Tensor:
        return self._table()


class _InAListInATuple(_Holder):
    """A list inside a tuple."""

    def __init__(self) -> None:
        self._slots: tuple[list[Tensor]] = ([],)

    def fetch(self, owner: nn.Module) -> tuple[Tensor, bool]:
        new = not self._slots[0]
        if new:
            self._slots[0].append(_made())
        return self._slots[0][0], new

    def peek(self, owner: nn.Module) -> Tensor:
        return self._slots[0][0]


class _AViewOfIt(_Holder):
    """A view of a larger tensor the call made; the view is all that is kept."""

    def __init__(self) -> None:
        self._box: dict[str, Tensor] = {}

    def fetch(self, owner: nn.Module) -> tuple[Tensor, bool]:
        new = "t" not in self._box
        if new:
            self._box["t"] = lucid.ones(SHAPE[0] + 1, SHAPE[1], device=COMPILE_DEVICE)[1:]
        return self._box["t"], new

    def peek(self, owner: nn.Module) -> Tensor:
        return self._box["t"]


HOLDERS: dict[str, Callable[[], _Holder]] = {
    "box": _InABox,
    "buffer": _AsABuffer,
    "closure-cell": _InACell,
    "generator-frame": _InAGenerator,
    "lru_cache": _InAnLruCache,
    "list-in-a-tuple": _InAListInATuple,
    "held-view": _AViewOfIt,
}


# -- what each call writes into it ----------------------------------------------

Write = Callable[[nn.Module, Tensor, Tensor, bool], None]


def _add(_o: nn.Module, t: Tensor, x: Tensor, _new: bool) -> None:
    t.add_(x)


def _mul(_o: nn.Module, t: Tensor, x: Tensor, _new: bool) -> None:
    t.mul_(x)


def _copy_of_itself(_o: nn.Module, t: Tensor, x: Tensor, _new: bool) -> None:
    t.copy_(t * 2.0 + x)


def _slice_add(_o: nn.Module, t: Tensor, x: Tensor, _new: bool) -> None:
    t[1:].add_(x[1:])


def _add_then_setitem(_o: nn.Module, t: Tensor, x: Tensor, _new: bool) -> None:
    t.add_(x.mean())
    t[0] = 5.0


def _masked_fill(_o: nn.Module, t: Tensor, x: Tensor, _new: bool) -> None:
    t.masked_fill_(x > 0.0, 7.0)


def _scatter(_o: nn.Module, t: Tensor, x: Tensor, _new: bool) -> None:
    t.scatter_(1, (x > 0.0).long() * 3, lucid.full(SHAPE, 2.0, device=COMPILE_DEVICE))


def _index_put(_o: nn.Module, t: Tensor, x: Tensor, _new: bool) -> None:
    rows = (x[:, 0] > 0.0).long() * 2
    t.index_put_((rows,), lucid.full((SHAPE[-1],), 3.0, device=COMPILE_DEVICE))


def _copy_of_the_input(_o: nn.Module, t: Tensor, x: Tensor, _new: bool) -> None:
    t.copy_(x * 2.0)


def _copy_of_the_state(o: nn.Module, t: Tensor, _x: Tensor, _new: bool) -> None:
    t.copy_(o.scale * 3.0)


def _setitem(_o: nn.Module, t: Tensor, x: Tensor, _new: bool) -> None:
    t[:2] = x[:2] * 3.0


def _first_batch(_o: nn.Module, t: Tensor, x: Tensor, new: bool) -> None:
    if new:
        t.copy_(x * 2.0)


def _first_batch_add(_o: nn.Module, t: Tensor, x: Tensor, new: bool) -> None:
    if new:
        t.add_(x)


def _first_batch_row(_o: nn.Module, t: Tensor, x: Tensor, new: bool) -> None:
    if new:
        t[1] = x[1]


def _first_from_the_state(o: nn.Module, t: Tensor, _x: Tensor, new: bool) -> None:
    if new:
        t.copy_(o.scale * 3.0)


def _first_a_constant(_o: nn.Module, t: Tensor, _x: Tensor, new: bool) -> None:
    if new:
        t[0] = 2.0


def _a_constant(_o: nn.Module, t: Tensor, _x: Tensor, _new: bool) -> None:
    t[0] = 2.0


#: Every call, reading what the tensor held: the reruns count.  The last four
#: hide the read behind a partial overwrite, whose positions are read off the
#: input or leave the rest of an earlier value in place.
READS_IT: dict[str, Write] = {
    "add_": _add,
    "mul_": _mul,
    "copy_-of-itself": _copy_of_itself,
    "slice-add_": _slice_add,
    "add_-then-setitem": _add_then_setitem,
    "masked_fill_": _masked_fill,
    "scatter_": _scatter,
    "index_put_": _index_put,
}

#: Every call, overwriting all of it (or what a constant position picks).
OVERWRITES_IT: dict[str, Write] = {
    "copy_-of-the-input": _copy_of_the_input,
    "copy_-of-the-state": _copy_of_the_state,
    "setitem": _setitem,
}

#: The first call only: a buffer filled from the first batch (``mu``), a cache
#: filled from the model's state, a constant mask.
FIRST_CALL: dict[str, Write] = {
    "first-batch": _first_batch,
    "first-batch-add_": _first_batch_add,
    "first-batch-row": _first_batch_row,
    "first-from-the-state": _first_from_the_state,
    "first-a-constant": _first_a_constant,
}

WRITES: dict[str, Write] = {**READS_IT, **OVERWRITES_IT, **FIRST_CALL, "a-constant": _a_constant}


class _Keeps(nn.Module):
    """Writes the held tensor, and answers from it."""

    def __init__(self, write: str, holder: str) -> None:
        super().__init__()
        self.register_buffer("scale", lucid.full((SHAPE[-1],), 2.0, device=COMPILE_DEVICE))
        # A parameter of its own, so that a segmented step compiles this part.
        self.gain = nn.Parameter(lucid.ones(SHAPE[-1], device=COMPILE_DEVICE))
        self.holder = HOLDERS[holder]()
        self.holder.attach(self)
        self.write = WRITES[write]

    def forward(self, x: Tensor) -> Tensor:
        t, new = self.holder.fetch(self)
        self.write(self, t, x, new)
        return x * t * self.gain


def _model(write: str, holder: str) -> nn.Sequential:
    lucid.manual_seed(0)
    return nn.Sequential(_Keeps(write, holder), nn.Linear(SHAPE[-1], 2)).to(COMPILE_DEVICE)


def _kept(model: nn.Sequential, at: int = 0) -> list[float]:
    keeps = model[at]
    assert isinstance(keeps, _Keeps)
    return _floats(keeps.holder.peek(keeps))


def _loss(out: Tensor) -> Tensor:
    return (out * out).mean()


def _grads(model: nn.Module) -> list[float]:
    return [v for p in model.parameters() if p.grad is not None for v in _floats(p.grad)]


def _weights(model: nn.Module) -> list[float]:
    return [v for p in model.parameters() for v in _floats(p)]


# -- the entry points, each beside its eager twin -------------------------------

Seen = list[list[float]]


def _module(model: nn.Sequential, compiled: bool) -> Seen:
    run = lucid.compile(model) if compiled else model
    with lucid.no_grad():
        return [[*_floats(run(x)), *_kept(model)] for x in _inputs()]


def _with_grads(model: nn.Sequential, call: Callable[[Tensor], Tensor]) -> Seen:
    seen: Seen = []
    for x in _inputs():
        loss = call(x)
        if loss.requires_grad:
            loss.backward()
        seen.append([float(loss.item()), *_grads(model), *_kept(model)])
    return seen


def _compiled_step(model: nn.Sequential, compiled: bool) -> Seen:
    if compiled:
        return _with_grads(model, lambda x: lucid.compile.compiled_step(model, x, _loss))
    return _with_grads(model, lambda x: _loss(model(x)))


def _make_step(segments: int) -> Callable[[nn.Sequential, bool], Seen]:
    def run(model: nn.Sequential, compiled: bool) -> Seen:
        if compiled:
            return _with_grads(model, lucid.compile.make_step(model, _loss, segments=segments))
        return _with_grads(model, lambda x: _loss(model(x)))

    return run


def _fused_step(model: nn.Sequential, compiled: bool) -> Seen:
    opt = optim.SGD(model.parameters(), lr=1e-2)
    if compiled:
        step = lucid.compile.fused_step(model, _loss, opt)
        return [[float(step(x).item()), *_weights(model), *_kept(model)] for x in _inputs()]
    seen: Seen = []
    for x in _inputs():
        opt.zero_grad()
        loss = _loss(model(x))
        loss.backward()
        opt.step()
        seen.append([float(loss.item()), *_weights(model), *_kept(model)])
    return seen


#: Can let the trace-time call stand as the call.
STANDING: dict[str, Callable[[nn.Sequential, bool], Seen]] = {
    "module": _module,
    "make_step": _make_step(1),
    "make_step-segments": _make_step(2),
    "fused_step": _fused_step,
}


def _close(got: list[float], want: list[float]) -> bool:
    return len(got) == len(want) and all(
        abs(g - w) <= 1e-4 * (1.0 + abs(w)) for g, w in zip(got, want, strict=True)
    )


def _ends_as_eager_ends(
    run: Callable[[nn.Sequential, bool], Seen], write: str, holder: str
) -> None:
    got = run(_model(write, holder), True)
    want = run(_model(write, holder), False)
    for i, (g, w) in enumerate(zip(got, want, strict=True)):
        assert _close(g, w), f"call {i}: {g} vs {w}"


@pytest.mark.parametrize("entry", STANDING)
@pytest.mark.parametrize("holder", HOLDERS)
@pytest.mark.parametrize("write", WRITES)
def test_a_held_tensor_ends_each_call_as_eager_ends(write: str, holder: str, entry: str) -> None:
    """Whatever writes it and wherever it is held: any write from the call's
    inputs or the model's state makes the trace stand as the call."""
    _ends_as_eager_ends(STANDING[entry], write, holder)


class _KeepsBehindATrunk(_Keeps):
    """Writes from its input's values: the input requires grad, and a tensor kept
    across calls must not keep the graph of the call that wrote it."""

    def forward(self, x: Tensor) -> Tensor:
        t, new = self.holder.fetch(self)
        self.write(self, t, x.detach(), new)
        return x * t * self.gain


def _in_a_later_segment(write: str) -> nn.Sequential:
    lucid.manual_seed(0)
    return nn.Sequential(
        nn.Linear(SHAPE[-1], SHAPE[-1]),
        _KeepsBehindATrunk(write, "closure-cell"),
        nn.Linear(SHAPE[-1], 2),
    ).to(COMPILE_DEVICE)


@pytest.mark.parametrize("write", ["first-batch", "add_", "copy_-of-the-input", "a-constant"])
def test_a_segment_after_a_compiled_one_stands_for_its_own_call(write: str) -> None:
    """The first segment runs compiled; the second's trace stands for it, and the
    third runs eagerly on that trace's output.  The loss is backpropagated
    through all three, into the first segment's executable."""
    model, twin = _in_a_later_segment(write), _in_a_later_segment(write)
    step = lucid.compile.make_step(model, _loss, segments=3)
    for x in _inputs():
        got, want = step(x), _loss(twin(x))
        got.backward()
        want.backward()
        assert _close(
            [float(got.item()), *_grads(model), *_kept(model, 1)],
            [float(want.item()), *_grads(twin), *_kept(twin, 1)],
        )
    assert [len(cache) for cache in step.cache] == [1, 0, 0]
    assert step.eager_only["flag"]


def _one_call(model: nn.Sequential, x: Tensor, *, compiled: bool) -> list[float]:
    loss = lucid.compile.compiled_step(model, x, _loss) if compiled else _loss(model(x))
    if loss.requires_grad:
        loss.backward()
    return [float(loss.item()), *_grads(model), *_kept(model)]


@pytest.mark.parametrize("write", [*READS_IT, *OVERWRITES_IT, "a-constant"])
def test_a_compiled_step_keeps_the_first_write_and_refuses_the_next(write: str) -> None:
    """``compiled_step`` traces every call, and the trace's writes are the call's:
    the first is right, a read behind a partial overwrite too.  The next call's
    trace finds the tensor from outside and refuses a write to it, as to any
    buffer, loudly."""
    model, twin = _model(write, "closure-cell"), _model(write, "closure-cell")
    first, second = _inputs()[:2]
    assert _close(_one_call(model, first, compiled=True), _one_call(twin, first, compiled=False))
    with pytest.raises(RuntimeError, match="did not create"):
        lucid.compile.compiled_step(model, second, _loss)


@pytest.mark.parametrize("write", FIRST_CALL)
def test_a_compiled_step_ends_a_tensor_written_on_the_first_call_as_eager_ends(write: str) -> None:
    """After the first call the tensor is the model's state, read and not written:
    a buffer filled from the first batch keeps that batch (no LCD-876 here)."""
    _ends_as_eager_ends(_compiled_step, write, "closure-cell")
