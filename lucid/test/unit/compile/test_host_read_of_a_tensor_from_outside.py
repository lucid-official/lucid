"""A host read of a tensor from outside the call follows that tensor (LCD-910).

``x * clo.item()`` hands the tensor's value to Python, where the recording cannot
follow what is done with it.  The tracer refused a read of a value it knew (an
argument, an op's output, the model's state) and left alone a tensor it had not
seen, taking "not seen" for "a constant made on the spot".  A tensor from outside
the call is not that: it is an input, live, and a compiled call replays no Python,
so every call after the first answered with the value the first one read.

The call's own constant stays readable — ``lucid.tensor([2.0]).item()`` is the
same on every call — and the model's parameters and buffers were already refused
(the trace watches them), so those cases only pin the rule down.

Each case holds the source one way (a closure, a global, a plain attribute, a
dict item, a buffer, a parameter, or nothing: made by the call), changes it between
four calls, and compares every answer with a twin running eagerly.  A later call
is what tells the bug from a correct answer: the first agrees either way.
"""

import warnings
from collections.abc import Callable, Iterator
from typing import NamedTuple

import pytest

import lucid
import lucid.nn as nn
import lucid.optim as optim
from lucid._C import engine as _C_engine
from lucid.compile._core import fallback
from lucid.compile._entry.decode_step import compiled_decode_step
from lucid.test.unit.compile._helpers import COMPILE_DEVICE
from lucid.utils.cache import StaticCache


def _metal_ok() -> bool:
    try:
        lucid.zeros((1,)).to(COMPILE_DEVICE)
    except RuntimeError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")

Tensor = lucid.Tensor

WIDTH = 4
#: What the source holds on each call; a compiled call that kept the first
#: call's value answers 1.0's on every later one.
VALUES = (1.0, 5.0, -2.0, 3.0)

_GLOBAL: dict[str, Tensor] = {}


@pytest.fixture(autouse=True)
def _fresh_session(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(fallback, "_NOTICED", set())
    _C_engine.compile.session_cache_clear()
    yield
    _GLOBAL.clear()


def _src() -> Tensor:
    return lucid.zeros(1, device=COMPILE_DEVICE)


class _Held(NamedTuple):
    """One way to hold the source: what the forward reads, and how a test changes it."""

    read: Callable[[], float]
    change: Callable[[float], None]


def _fill(t: Tensor) -> Callable[[float], None]:
    def change(v: float) -> None:
        with lucid.no_grad():
            t.fill_(v)

    return change


def _closure(net: nn.Module) -> _Held:
    clo = _src()
    return _Held(lambda: clo.item(), _fill(clo))


def _global(net: nn.Module) -> _Held:
    _GLOBAL["src"] = _src()
    return _Held(lambda: _GLOBAL["src"].item(), _fill(_GLOBAL["src"]))


def _attribute(net: nn.Module) -> _Held:
    holder = _Plain()
    return _Held(lambda: holder.src.item(), _fill(holder.src))


def _dict_item(net: nn.Module) -> _Held:
    table = {"k": _src()}
    return _Held(lambda: table["k"].item(), _fill(table["k"]))


def _buffer(net: nn.Module) -> _Held:
    net.register_buffer("src", _src())
    return _Held(lambda: net.get_buffer("src").item(), _fill(net.get_buffer("src")))


def _parameter(net: nn.Module) -> _Held:
    net.src = nn.Parameter(_src(), requires_grad=False)
    return _Held(lambda: net.src.item(), _fill(net.src))


def _made(net: nn.Module) -> _Held:
    # Built from Python numbers inside the call: a constant, never an input.
    return _Held(lambda: lucid.tensor([2.0], device=COMPILE_DEVICE).item(), lambda v: None)


class _Plain:
    """A plain object holding a tensor: not a module, so nothing registers it."""

    def __init__(self) -> None:
        self.src = _src()


#: name -> (how it is held, whether it comes from outside the call).
HOLDERS: dict[str, tuple[Callable[[nn.Module], _Held], bool]] = {
    "closure": (_closure, True),
    "global": (_global, True),
    "plain attribute": (_attribute, True),
    "dict item": (_dict_item, True),
    "module buffer": (_buffer, True),
    "parameter": (_parameter, True),
    "made by the call": (_made, False),
}


class _Net(nn.Module):
    """``lin(x * read())``: a layer, so that a training step has a parameter."""

    def __init__(self, holder: str) -> None:
        super().__init__()
        lucid.manual_seed(0)
        self.lin = nn.Linear(WIDTH, 2)
        self.held = HOLDERS[holder][0](self)

    def forward(self, x: Tensor) -> Tensor:
        return self.lin(x * self.held.read())


def _net(holder: str) -> _Net:
    return _Net(holder).to(COMPILE_DEVICE)


def _floats(t: Tensor) -> list[float]:
    return [float(v) for v in t.reshape(-1).tolist()]


def _loss(out: Tensor) -> Tensor:
    return (out * out).mean()


def _x() -> Tensor:
    return lucid.ones(3, WIDTH, device=COMPILE_DEVICE)


class _Seen(NamedTuple):
    answers: list[list[float]]
    #: Every call ran an executable, none fell back to eager.
    compiled: bool = False


def _eager_module(net: _Net) -> _Seen:
    seen = _Seen([])
    with lucid.no_grad():
        for v in VALUES:
            net.held.change(v)
            seen.answers.append(_floats(net(_x())))
    return seen


def _module(net: _Net) -> _Seen:
    compiled = lucid.compile(net)
    answers = []
    with lucid.no_grad():
        for v in VALUES:
            net.held.change(v)
            answers.append(_floats(compiled(_x())))
    info = compiled.cache_info()
    return _Seen(answers, info["entries"] == 1 and not info["eager_only"])


def _eager_step(net: _Net) -> _Seen:
    seen = _Seen([])
    for v in VALUES:
        net.held.change(v)
        net.zero_grad()
        loss = _loss(net(_x()))
        loss.backward()
        seen.answers.append([float(loss.item()), *_floats(net.lin.weight.grad)])
    return seen


def _step(net: _Net) -> _Seen:
    step = lucid.compile.make_step(net, _loss)
    answers = []
    for v in VALUES:
        net.held.change(v)
        net.zero_grad()
        loss = step(_x())
        loss.backward()
        answers.append([float(loss.item()), *_floats(net.lin.weight.grad)])
    return _Seen(answers, len(step.cache) == 1 and not step.eager_only.snapshot())


def _eager_fused(net: _Net) -> _Seen:
    opt = optim.SGD(net.parameters(), lr=1e-4)
    seen = _Seen([])
    for v in VALUES:
        net.held.change(v)
        opt.zero_grad()
        loss = _loss(net(_x()))
        loss.backward()
        opt.step()
        seen.answers.append([float(loss.item()), *_floats(net.lin.weight)])
    return seen


def _fused(net: _Net) -> _Seen:
    step = lucid.compile.fused_step(net, _loss, optim.SGD(net.parameters(), lr=1e-4))
    answers = []
    for v in VALUES:
        net.held.change(v)
        loss = step(_x())
        answers.append([float(loss.item()), *_floats(net.lin.weight)])
    info = step.cache_info()
    return _Seen(answers, not info["eager_only"])


class _Decode:
    """A decode step reading the source: the key it caches is ``mean(hidden) * read()``."""

    def __init__(self, holder: str) -> None:
        self.cache = StaticCache(8)
        self.held = HOLDERS[holder][0](nn.Module())

    def forward(self, hidden: Tensor, cache_position: Tensor) -> Tensor:
        key = (hidden * self.held.read()).mean(0).reshape(1, 1, 1, WIDTH)
        keys, _ = self.cache.update(key, key, 0, {"cache_position": cache_position})
        return keys * 1.0


def _decode(holder: str, compiled: bool) -> _Seen:
    rig = _Decode(holder)
    step = compiled_decode_step(rig.forward, rig.cache)
    forward = step if compiled else rig.forward
    answers = []
    with lucid.no_grad():
        for pos, v in enumerate(VALUES):
            rig.held.change(v)
            hidden = lucid.ones(3, WIDTH, device=COMPILE_DEVICE) * (pos + 1.0)
            cp = lucid.tensor([pos], device=COMPILE_DEVICE).long()
            answers.append(_floats(forward(hidden, cp)))
    return _Seen(answers, compiled and len(step.cache) == 1)


#: name -> (the compiled call, its eager twin).
ENTRIES: dict[str, tuple[Callable[[_Net], _Seen], Callable[[_Net], _Seen]]] = {
    "module": (_module, _eager_module),
    "make_step": (_step, _eager_step),
    "fused_step": (_fused, _eager_fused),
}


def _close(got: list[float], want: list[float]) -> bool:
    return len(got) == len(want) and all(
        abs(g - w) <= 1e-4 * (1.0 + abs(w)) for g, w in zip(got, want, strict=True)
    )


def _agree(got: _Seen, want: _Seen) -> None:
    for i in range(len(VALUES)):
        assert _close(got.answers[i], want.answers[i]), f"call {i}: {got.answers} vs {want.answers}"


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("holder", HOLDERS)
def test_a_host_read_follows_its_tensor_and_stays_compiled_only_if_the_call_made_it(
    holder: str, entry: str
) -> None:
    run, run_eager = ENTRIES[entry]
    got = run(_net(holder))
    _agree(got, run_eager(_net(holder)))
    if not HOLDERS[holder][1]:
        assert got.compiled, f"{holder}: {entry} fell back to eager"


@pytest.mark.parametrize("holder", [h for h in HOLDERS if h not in ("module buffer", "parameter")])
def test_a_decode_step_reads_its_tensor_afresh(holder: str) -> None:
    got = _decode(holder, compiled=True)
    _agree(got, _decode(holder, compiled=False))
    if not HOLDERS[holder][1]:
        assert got.compiled, f"{holder}: the decode step fell back to eager"


def test_the_repro_of_the_report() -> None:
    """The case as filed: a closure-held tensor read by ``item()``."""
    clo = lucid.zeros(1, device=COMPILE_DEVICE)

    def fn(x: Tensor) -> Tensor:
        return x * clo.item()

    compiled = lucid.compile(fn)
    x = lucid.ones(WIDTH, device=COMPILE_DEVICE)
    with lucid.no_grad():
        assert _floats(compiled(x)) == [0.0] * WIDTH
        clo.fill_(5.0)
        assert _floats(compiled(x)) == _floats(fn(x)) == [5.0] * WIDTH


def test_an_outside_read_says_why_it_runs_eagerly() -> None:
    clo = lucid.zeros(1, device=COMPILE_DEVICE)
    compiled = lucid.compile(lambda x: x * clo.item())
    x = lucid.ones(WIDTH, device=COMPILE_DEVICE)
    with lucid.no_grad(), pytest.warns(fallback.CompileFallbackWarning, match="outside the call"):
        compiled(x)
    assert compiled.cache_info()["eager_only"]


def test_a_read_of_a_tensor_the_call_made_stays_compiled() -> None:
    def fn(x: Tensor) -> Tensor:
        return x * lucid.tensor([2.0, 3.0], device=COMPILE_DEVICE).tolist()[1]

    compiled = lucid.compile(fn)
    with lucid.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("error", fallback.CompileFallbackWarning)
        for scale in VALUES:
            x = lucid.ones(WIDTH, device=COMPILE_DEVICE) * scale
            assert _floats(compiled(x)) == _floats(fn(x))
    info = compiled.cache_info()
    assert info["entries"] == 1 and not info["eager_only"], info
