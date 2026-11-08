"""A tensor derived from one the call never read follows it on every call (LCD-861).

``to_shared``, a cross-device move, a fused kernel and a custom Metal kernel make
a tensor outside the op dispatcher, and tell the tracer through
``Tracer::on_derived``.  When the source was one the trace had not seen, the
tracer took it for a constant, and so the result too: nothing was recorded.  A
later op that read the result took it as a feed pinned at the value it held at
trace time, so a result that is a *copy* of the source answered with the first
call's bytes on every call, whatever the source held since.

The source is a tensor the call did not make, so what the derive does with its
bytes decides.  A label or view of them (a detach, a relabel of a shared buffer,
a reshape) reads the bytes as they are, and a feed of it is live.  A copy, or a
value computed from them, holds the values of the moment: the trace records it
and the source becomes a feed.

Each case holds the source on the model (a plain attribute, a buffer), changes it
between four calls, and compares every answer with a twin running eagerly.  A
later call is what tells the bug from a correct answer: the first agrees either
way.
"""

from collections.abc import Callable, Iterator
from typing import NamedTuple

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid._C import engine as _C_engine
from lucid.test.unit.compile._helpers import COMPILE_DEVICE


def _metal_ok() -> bool:
    try:
        lucid.zeros((1,)).to(COMPILE_DEVICE)
    except RuntimeError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")

WIDTH = 4
SHAPE = (3, WIDTH)
#: What the source holds on each call; a compiled call that kept the first
#: call's copy answers 1.0's on every later one.
VALUES = (1.0, 5.0, -2.0, 3.0)

Tensor = lucid.Tensor
Derive = Callable[[Tensor], Tensor]
Source = Callable[[], Tensor]

_DOUBLE = """
#include <metal_stdlib>
using namespace metal;
kernel void twice(device const float* x [[buffer(0)]],
                  device float* y [[buffer(1)]],
                  uint gid [[thread_position_in_grid]]) { y[gid] = 2.0f * x[gid]; }
"""


@pytest.fixture(autouse=True)
def _fresh_session() -> Iterator[None]:
    _C_engine.compile.session_cache_clear()
    _OUTSIDE["weight"] = lucid.eye(WIDTH, device=COMPILE_DEVICE) * 2.0
    yield


class _Net(nn.Module):
    """``x * derive(source)``, then a layer so that a training step has a parameter."""

    def __init__(self, derive: Derive, source: Source, as_buffer: bool) -> None:
        super().__init__()
        self.derive = derive
        self.lin = nn.Linear(WIDTH, 2)
        if as_buffer:
            self.register_buffer("src", source())
        else:
            self.src = source()

    def forward(self, x: Tensor) -> Tensor:
        return self.lin(x * self.derive(self.src))


def _twin(derive: Derive, source: Source, as_buffer: bool) -> _Net:
    lucid.manual_seed(0)
    return _Net(derive, source, as_buffer).to(COMPILE_DEVICE)


def _hold(net: _Net, value: float) -> None:
    net.src.fill_(value)


def _floats(t: Tensor) -> list[float]:
    return [float(v) for v in t.reshape(-1).tolist()]


def _loss(out: Tensor) -> Tensor:
    return (out * out).mean()


def _x() -> Tensor:
    return lucid.ones(*SHAPE, device=COMPILE_DEVICE)


class _Seen(NamedTuple):
    answers: list[list[float]]
    #: Every executable ran, none fell back to eager.
    compiled: bool = False


def _eager_module(net: _Net) -> _Seen:
    seen = _Seen([])
    with lucid.no_grad():
        for v in VALUES:
            _hold(net, v)
            seen.answers.append(_floats(net(_x())))
    return seen


def _module(net: _Net) -> _Seen:
    compiled = lucid.compile(net)
    answers = []
    with lucid.no_grad():
        for v in VALUES:
            _hold(net, v)
            answers.append(_floats(compiled(_x())))
    info = compiled.cache_info()
    return _Seen(answers, info["entries"] == 1 and not info["eager_only"])


def _eager_step(net: _Net) -> _Seen:
    seen = _Seen([])
    for v in VALUES:
        _hold(net, v)
        loss = _loss(net(_x()))
        loss.backward()
        seen.answers.append([float(loss.item()), *_floats(net.lin.weight.grad)])
    return seen


def _step(net: _Net) -> _Seen:
    step = lucid.compile.make_step(net, _loss)
    answers = []
    for v in VALUES:
        _hold(net, v)
        loss = step(_x())
        loss.backward()
        answers.append([float(loss.item()), *_floats(net.lin.weight.grad)])
    return _Seen(answers, len(step.cache) == 1 and not step.eager_only.snapshot())


def _eager_fused(net: _Net) -> _Seen:
    opt = optim.SGD(net.parameters(), lr=1e-4)
    seen = _Seen([])
    for v in VALUES:
        _hold(net, v)
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
        _hold(net, v)
        loss = step(_x())
        answers.append([float(loss.item()), *_floats(net.lin.weight)])
    info = step.cache_info()
    return _Seen(answers, not info["eager_only"])


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


def _agrees(derive: Derive, source: Source, entry: str, as_buffer: bool) -> _Seen:
    run, run_eager = ENTRIES[entry]
    got = run(_twin(derive, source, as_buffer))
    want = run_eager(_twin(derive, source, as_buffer))
    for i in range(len(VALUES)):
        assert _close(got.answers[i], want.answers[i]), f"call {i}: {got.answers} vs {want.answers}"
    return got


def _metal() -> Tensor:
    return lucid.zeros(WIDTH, device=COMPILE_DEVICE)


def _cpu() -> Tensor:
    return lucid.zeros(WIDTH)


def _shared() -> Tensor:
    return lucid.metal.to_shared(_metal())


def _fused_linear(src: Tensor) -> Tensor:
    """``relu(src @ (2 I).T + src)``: a fused kernel reading the source twice and a constant."""
    weight = lucid.eye(WIDTH, device=COMPILE_DEVICE) * 2.0
    return F.fused_linear_relu(src, weight, src)


#: A weight made before any trace, so the fused kernel's every operand is from outside.
_OUTSIDE: dict[str, Tensor] = {}


def _fused_linear_outside(src: Tensor) -> Tensor:
    return F.fused_linear_relu(src, _OUTSIDE["weight"], src)


def _custom_kernel(src: Tensor) -> Tensor:
    out = lucid.metal.run_kernel(_DOUBLE, "twice", [src], (WIDTH,), grid=(WIDTH, 1, 1))
    return out.to(COMPILE_DEVICE)


#: A derive that copies or computes: its result holds the source's values of the
#: moment.  Each is (a source, what is made of it).
COPIES: dict[str, tuple[Source, Derive]] = {
    "to_shared": (_metal, lambda s: lucid.metal.to_shared(s)),
    "to_shared of a cpu tensor": (_cpu, lambda s: lucid.metal.to_shared(s).to(COMPILE_DEVICE)),
    "cpu to metal": (_cpu, lambda s: s.to(COMPILE_DEVICE)),
    "metal through cpu": (_metal, lambda s: s.to("cpu").to(COMPILE_DEVICE)),
    "fused kernel": (_metal, _fused_linear),
    "fused kernel, all operands outside": (_metal, _fused_linear_outside),
    "custom kernel": (_metal, _custom_kernel),
}

#: A derive that only labels or views the source's bytes: a feed of it is live.
LABELS: dict[str, tuple[Source, Derive]] = {
    "detach": (_metal, lambda s: s.detach()),
    "detach of a cpu tensor": (_cpu, lambda s: s.detach().to(COMPILE_DEVICE)),
    "reshape": (_metal, lambda s: s.reshape(2, 2).reshape(WIDTH)),
    "shared buffer relabelled": (_shared, lambda s: s.to("cpu").to(COMPILE_DEVICE)),
    "shared already": (_shared, lambda s: lucid.metal.to_shared(s)),
    "plain read": (_metal, lambda s: s),
}

HOLDERS = {"attribute": False, "buffer": True}

#: A source left on the host stays there as an attribute, so a Metal trace that
#: reads it is mixed-device and always runs eagerly; as a buffer, ``.to`` moves it.
_HOST_SOURCES = frozenset(
    {"to_shared of a cpu tensor", "cpu to metal", "metal through cpu", "detach of a cpu tensor"}
)
#: As a buffer, ``Module.to`` moves the source (and leaves a shared one's buffer).
_ATTRIBUTE_ONLY = _HOST_SOURCES | {"shared buffer relabelled"}


def _on_the_host(kind: str, holder: str) -> bool:
    if kind in _ATTRIBUTE_ONLY and holder == "buffer":
        pytest.skip("Module.to moves a buffer to the model's device")
    return kind in _HOST_SOURCES


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("holder", HOLDERS)
@pytest.mark.parametrize("kind", COPIES)
def test_a_copy_of_an_unseen_tensor_follows_it(kind: str, holder: str, entry: str) -> None:
    source, derive = COPIES[kind]
    _on_the_host(kind, holder)
    _agrees(derive, source, entry, HOLDERS[holder])


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("holder", HOLDERS)
@pytest.mark.parametrize("kind", LABELS)
def test_a_label_of_an_unseen_tensor_follows_it_and_stays_compiled(
    kind: str, holder: str, entry: str
) -> None:
    source, derive = LABELS[kind]
    host = _on_the_host(kind, holder)
    seen = _agrees(derive, source, entry, HOLDERS[holder])
    assert host or seen.compiled, f"{kind}: {entry} fell back to eager"


def test_the_repro_of_the_report() -> None:
    """The case as filed: a closure-held tensor, copied by ``to_shared``."""
    clo = lucid.zeros(WIDTH, device=COMPILE_DEVICE)

    def fn(x: Tensor) -> Tensor:
        return x * lucid.metal.to_shared(clo)

    compiled = lucid.compile(fn)
    x = lucid.ones(WIDTH, device=COMPILE_DEVICE)
    with lucid.no_grad():
        assert _floats(compiled(x)) == [0.0] * WIDTH
        clo.fill_(5.0)
        assert _floats(compiled(x)) == _floats(fn(x)) == [5.0] * WIDTH


def test_a_copy_of_a_tensor_the_call_made_stays_a_constant() -> None:
    """The source is the call's own: nothing about it can differ between calls."""

    def fn(x: Tensor) -> Tensor:
        table = lucid.metal.to_shared(lucid.arange(float(WIDTH), device=COMPILE_DEVICE))
        return x * table

    compiled = lucid.compile(fn)
    with lucid.no_grad():
        for scale in VALUES:
            x = lucid.ones(WIDTH, device=COMPILE_DEVICE) * scale
            assert _floats(compiled(x)) == _floats(fn(x))
    info = compiled.cache_info()
    assert info["entries"] == 1 and not info["eager_only"], info
