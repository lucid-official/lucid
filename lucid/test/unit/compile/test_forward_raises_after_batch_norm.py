"""A forward that raises leaves the batch norm statistics where eager does (LCD-888).

A batch norm updates its running statistics inside its own call, so an eager
forward that raises *after* it has already moved ``running_mean`` /
``running_var`` and ``num_batches_tracked`` by one batch, and one that raises
*before* it has moved nothing.  Traced, the kernel hands the new statistics to
the executable instead of writing them, and the entry point writes them back
only after a run that succeeded: a trace-time call that raised left the buffers
at their values from before the call, one batch behind the eager twin.

The fix is one helper every entry point's trace goes through
(:func:`completes_on_raise`).  Each case is one entry point, one kind of norm and one place
for the forward to raise, run beside an eager twin: the exception must reach
the caller unchanged, the buffers must match after it, and the calls that
follow (which retrace, the raising call having cached nothing) must neither
repeat nor skip the update.
"""

from collections.abc import Callable, Iterator
from typing import NamedTuple, override

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._C import engine as _C_engine
from lucid.compile._core.buffer_writes import completes_on_raise
from lucid.compile._core.trace_context import recording
from lucid.compile._entry.decode_step import compiled_decode_step
from lucid.test.unit.compile._helpers import COMPILE_DEVICE
from lucid.utils.cache import StaticCache

Tensor = lucid.Tensor

_WIDTH = 4
_SHAPE = (6, _WIDTH)
_MESSAGE = "forward failed on purpose"


def _metal_ok() -> bool:
    try:
        lucid.zeros((1,)).to(COMPILE_DEVICE)
    except RuntimeError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")


@pytest.fixture(autouse=True)
def _fresh_session() -> Iterator[None]:
    _C_engine.compile.session_cache_clear()
    yield


class _Boom(ValueError):
    """What the forward raises."""


class _BoomRuntime(RuntimeError):
    """The same, as a ``RuntimeError``: ``fused_step`` answers one with an eager rerun,
    which updates the statistics itself, and the others let it through."""


RAISES: dict[str, type[Exception]] = {"value": _Boom, "runtime": _BoomRuntime}


class _Stage(nn.Module):
    """The norm under test, and the point in it where the forward may raise."""

    def __init__(self, norm: str) -> None:
        super().__init__()
        self.norm = norm
        self.raise_at = "never"
        self.raises: type[Exception] = _Boom
        # Not the defaults, so a statistic left alone shows.
        self.register_buffer("rm", lucid.arange(float(_WIDTH)) * 0.1)
        self.register_buffer("rv", lucid.arange(float(_WIDTH)) * 0.2 + 1.0)
        self.bn = nn.BatchNorm1d(_WIDTH) if norm == "module" else nn.Identity()

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        if self.raise_at == "before":
            raise self.raises(_MESSAGE)
        if self.norm == "module":
            y = self.bn(x)
        else:
            y = F.batch_norm(x, self.rm, self.rv, training=True, momentum=0.3)
        if self.raise_at == "after":
            raise self.raises(_MESSAGE)
        return y


def _model(norm: str) -> nn.Sequential:
    lucid.manual_seed(0)
    return nn.Sequential(
        nn.Linear(_WIDTH, _WIDTH), _Stage(norm), nn.Linear(_WIDTH, _WIDTH), nn.Linear(_WIDTH, 2)
    ).to(COMPILE_DEVICE)


def _stage(model: nn.Sequential) -> _Stage:
    stage = model[1]
    assert isinstance(stage, _Stage)
    return stage


def _inputs() -> list[Tensor]:
    lucid.manual_seed(1)
    base = lucid.randn(*_SHAPE, device=COMPILE_DEVICE)
    return [base * scale + offset for scale, offset in ((1.0, 0.0), (2.0, 1.0), (0.5, -2.0))]


def _floats(t: Tensor) -> list[float]:
    return [float(v) for v in t.detach().reshape(-1).tolist()]


def _statistics(model: nn.Sequential) -> list[float]:
    stage = _stage(model)
    values = [*_floats(stage.rm), *_floats(stage.rv)]
    if stage.norm == "module":
        bn = stage.bn
        values += [*_floats(bn.running_mean), *_floats(bn.running_var)]
        values += [float(bn.num_batches_tracked.item())]
    return values


def _close(got: list[float], want: list[float]) -> bool:
    return len(got) == len(want) and all(
        abs(g - w) <= 1e-4 * (1.0 + abs(w)) for g, w in zip(got, want, strict=True)
    )


def _loss(out: Tensor) -> Tensor:
    return out.square().mean()


class _Entry(NamedTuple):
    """How one entry point makes a call, and what it needs of the model."""

    #: A callable taking an input, and a model to run it on.
    build: Callable[[nn.Sequential], Callable[[Tensor], Tensor]]
    #: Whether the call runs under ``no_grad`` (the module entry compiles there).
    no_grad: bool = False
    #: Whether the call also steps an SGD optimizer, as the eager twin must.
    steps: bool = False


def _module_entry(model: nn.Sequential) -> Callable[[Tensor], Tensor]:
    return lucid.compile(model)


def _make_step(model: nn.Sequential) -> Callable[[Tensor], Tensor]:
    return lucid.compile.make_step(model, _loss)


def _segmented(segments: int) -> Callable[[nn.Sequential], Callable[[Tensor], Tensor]]:
    def build(model: nn.Sequential) -> Callable[[Tensor], Tensor]:
        return lucid.compile.make_step(model, _loss, segments=segments)

    return build


def _fused_step(model: nn.Sequential) -> Callable[[Tensor], Tensor]:
    return lucid.compile.fused_step(model, _loss, lucid.optim.SGD(model.parameters(), lr=1e-3))


def _compiled_step(model: nn.Sequential) -> Callable[[Tensor], Tensor]:
    return lambda x: lucid.compile.compiled_step(model, x, _loss)


ENTRIES: dict[str, _Entry] = {
    "module": _Entry(_module_entry, no_grad=True),
    "make_step": _Entry(_make_step),
    "make_step-segments-2": _Entry(_segmented(2)),
    "make_step-segments-3": _Entry(_segmented(3)),
    "fused_step": _Entry(_fused_step, steps=True),
    "compiled_step": _Entry(_compiled_step),
}


def _call(step: Callable[[Tensor], Tensor], x: Tensor, entry: _Entry) -> Tensor | Exception:
    """One call, answering what it raised instead of raising it."""
    try:
        if entry.no_grad:
            with lucid.no_grad():
                return step(x)
        return step(x)
    except Exception as why:  # the exception is the thing under test
        return why


def _eager_twin(model: nn.Sequential, entry: _Entry) -> Callable[[Tensor], Tensor | Exception]:
    """The same calls made eagerly, answering what they raise."""
    opt = lucid.optim.SGD(model.parameters(), lr=1e-3) if entry.steps else None

    def call(x: Tensor) -> Tensor | Exception:
        try:
            if entry.no_grad:
                with lucid.no_grad():
                    return model(x)
            if opt is None:
                return _loss(model(x))
            opt.zero_grad()
            loss = _loss(model(x))
            loss.backward()
            opt.step()
            return loss.detach()
        except Exception as why:
            return why

    return call


def _make_twins(entry_name: str, norm: str) -> tuple[nn.Sequential, nn.Sequential, _Entry]:
    entry = ENTRIES[entry_name]
    model, twin = _model(norm), _model(norm)
    twin.load_state_dict(model.state_dict())
    return model, twin, entry


@pytest.mark.parametrize("kind", RAISES)
@pytest.mark.parametrize("raise_at", ["after", "before"])
@pytest.mark.parametrize("norm", ["module", "functional"])
@pytest.mark.parametrize("entry_name", ENTRIES)
def test_a_forward_that_raises_leaves_the_statistics_as_eager_does(
    entry_name: str, norm: str, raise_at: str, kind: str
) -> None:
    """The exception is the forward's own, and the buffers are eager's after it."""
    model, twin, entry = _make_twins(entry_name, norm)
    step, eager = entry.build(model), _eager_twin(twin, entry)
    x = _inputs()[0]
    _stage(model).raise_at = _stage(twin).raise_at = raise_at
    _stage(model).raises = _stage(twin).raises = RAISES[kind]

    got = _call(step, x, entry)
    want = eager(x)

    assert isinstance(want, RAISES[kind])
    assert type(got) is type(want) and str(got) == str(want) == _MESSAGE, got
    assert _close(
        _statistics(model), _statistics(twin)
    ), f"{entry_name}/{norm}/{raise_at}/{kind}: {_statistics(model)} vs eager {_statistics(twin)}"

    # The raising call cached nothing: the calls after it retrace, and must
    # neither repeat the update the exception path made nor skip theirs.
    _stage(model).raise_at = _stage(twin).raise_at = "never"
    for x in _inputs()[1:]:
        got_next = _call(step, x, entry)
        want_next = eager(x)
        assert isinstance(got_next, Tensor) and isinstance(want_next, Tensor)
        assert _close(_floats(got_next), _floats(want_next))
        assert _close(_statistics(model), _statistics(twin)), (
            f"{entry_name}/{norm}/{raise_at}/{kind}, call after: "
            f"{_statistics(model)} vs eager {_statistics(twin)}"
        )


class _Tally(nn.Module):
    """A stage that writes a buffer in place and then raises."""

    def __init__(self, raises: type[Exception]) -> None:
        super().__init__()
        self.raises = raises
        self.register_buffer("tally", lucid.zeros(1))

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        self.tally.add_(1.0)
        raise self.raises(_MESSAGE)


def _tally_model(raises: type[Exception]) -> nn.Sequential:
    lucid.manual_seed(0)
    return nn.Sequential(
        nn.Linear(_WIDTH, _WIDTH), _Tally(raises), nn.Linear(_WIDTH, _WIDTH), nn.Linear(_WIDTH, 2)
    ).to(COMPILE_DEVICE)


def _tally_of(model: nn.Sequential) -> Tensor:
    stage = model[1]
    assert isinstance(stage, _Tally)
    return stage.tally


@pytest.mark.parametrize("kind", RAISES)
@pytest.mark.parametrize("entry_name", ENTRIES)
def test_a_forward_that_raises_keeps_the_write_it_made_before(entry_name: str, kind: str) -> None:
    """An in-place write to a buffer before the raise has happened, as in eager:
    ``fused_step`` put every outside write back on any exception, so a forward
    that raised left the buffer where it was before the call."""
    entry = ENTRIES[entry_name]
    model, twin = _tally_model(RAISES[kind]), _tally_model(RAISES[kind])
    got = _call(entry.build(model), _inputs()[0], entry)
    want = _eager_twin(twin, entry)(_inputs()[0])
    assert isinstance(want, RAISES[kind])
    assert type(got) is type(want) and str(got) == str(want) == _MESSAGE, got
    assert _floats(_tally_of(model)) == _floats(_tally_of(twin)) == [1.0]


def test_the_trace_does_not_complete_a_call_a_retry_will_run_again() -> None:
    """A hook error is the entry's cue to run the call eagerly, which updates the
    statistics itself: the trace must leave them alone then (once, not twice)."""
    model = _model("functional")
    stage = _stage(model)
    before = _statistics(model)
    x = _inputs()[0]
    with pytest.raises(RuntimeError, match="register_hook"):
        with lucid.no_grad():
            with recording(watch=(x,), module=model) as tracer, completes_on_raise(tracer, model):
                stage(x)
                raise RuntimeError("register_hook: not a tensor that requires grad")
    assert _statistics(model) == before


# -- compiled_decode_step, whose forward takes the cache position -------------------


class _Decoder:
    """A decode body: the norm under test, a place to raise, then the cache write."""

    def __init__(self, norm: str, raise_at: str, raises: type[Exception]) -> None:
        self.cache = StaticCache(8)
        self.stage = _Stage(norm).to(COMPILE_DEVICE)
        self.stage.raise_at = raise_at
        self.stage.raises = raises
        self.stage.train()

    def forward_fn(self) -> Callable[[Tensor, Tensor], Tensor]:
        """The step's body, with the norm held in a closure cell: how the step finds
        a module to read its counter from."""
        stage, cache = self.stage, self.cache

        def forward(hidden: Tensor, cache_position: Tensor) -> Tensor:
            key = (stage(hidden).mean(0) * 1.0).reshape(1, 1, 1, _WIDTH)
            keys, _ = cache.update(key, key, 0, {"cache_position": cache_position})
            return keys * 1.0

        return forward

    def statistics(self) -> list[float]:
        values = [*_floats(self.stage.rm), *_floats(self.stage.rv)]
        if self.stage.norm == "module":
            bn = self.stage.bn
            values += [*_floats(bn.running_mean), *_floats(bn.running_var)]
            values += [float(bn.num_batches_tracked.item())]
        return values


def _decode_call(forward: Callable[[Tensor, Tensor], Tensor], pos: int) -> Tensor | Exception:
    hidden = lucid.arange(3 * _WIDTH, device=COMPILE_DEVICE).reshape(3, _WIDTH) * 0.1 * (pos + 1.0)
    position = lucid.tensor([pos], device=COMPILE_DEVICE).long()
    try:
        with lucid.no_grad():
            return forward(hidden, position)
    except Exception as why:
        return why


@pytest.mark.parametrize("kind", RAISES)
@pytest.mark.parametrize("raise_at", ["after", "before"])
@pytest.mark.parametrize("norm", ["module", "functional"])
def test_a_decode_forward_that_raises_leaves_the_statistics_as_eager_does(
    norm: str, raise_at: str, kind: str
) -> None:
    """The decode step lets the exception through and moves the statistics as eager does.

    The first call runs eagerly (the cache is still empty), so it is made clean:
    the call that raises is the second, the one the step traces.
    """
    lucid.manual_seed(0)
    rig = _Decoder(norm, "never", RAISES[kind])
    lucid.manual_seed(0)
    twin = _Decoder(norm, "never", RAISES[kind])
    step, eager = compiled_decode_step(rig.forward_fn(), rig.cache), twin.forward_fn()
    assert isinstance(_decode_call(step, 0), Tensor)
    assert isinstance(_decode_call(eager, 0), Tensor)

    rig.stage.raise_at = twin.stage.raise_at = raise_at
    got, want = _decode_call(step, 1), _decode_call(eager, 1)
    assert isinstance(want, RAISES[kind])
    assert type(got) is type(want) and str(got) == str(want) == _MESSAGE, got
    assert _close(rig.statistics(), twin.statistics()), (rig.statistics(), twin.statistics())

    rig.stage.raise_at = twin.stage.raise_at = "never"
    for pos in range(1, 4):
        got_next, want_next = _decode_call(step, pos), _decode_call(eager, pos)
        assert isinstance(got_next, Tensor) and isinstance(want_next, Tensor)
        assert _close(_floats(got_next), _floats(want_next))
        assert _close(rig.statistics(), twin.statistics()), (rig.statistics(), twin.statistics())
