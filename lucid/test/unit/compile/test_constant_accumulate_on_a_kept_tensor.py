"""A kept tensor a call writes from constants alone (LCD-864).

``Tracer::refuse_kept_writes`` lets a tensor the call made and Python keeps
stay compiled when every write to it is a constant: a table built once and
cached holds the same values however often a compiled call rebuilds it.  That
is true of a write that *overwrites* (``fill_``, ``copy_``, a setitem), and
false of one that *accumulates*: ``h.add_(1.0)`` reads ``h``, so a compiled
call that rebuilds ``h`` from its factory starts it over every call.  Eager
goes on from the last call, and the compiled answers froze at the first one.

The other operands being constants says nothing about whether the write reads
the tensor's own value.  Every case runs five calls beside an eager twin; the
writes that read the tensor must agree with eager (by running eager), the ones
that only overwrite must stay compiled on one executable.
"""

from collections.abc import Callable
from typing import NamedTuple

import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid.test.unit.compile._helpers import COMPILE_DEVICE


def _metal_ok() -> bool:
    try:
        lucid.zeros((1,)).to(COMPILE_DEVICE)
    except RuntimeError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")

Tensor = lucid.Tensor
N = 4
CALLS = 5


@pytest.fixture(autouse=True)
def _fresh_session() -> None:
    _C_engine.compile.session_cache_clear()


def _zeros() -> Tensor:
    return lucid.zeros(N, device=COMPILE_DEVICE)


def _ones() -> Tensor:
    return lucid.ones(N, device=COMPILE_DEVICE)


def _const(value: float) -> Tensor:
    return lucid.full((N,), value, device=COMPILE_DEVICE)


def _index() -> Tensor:
    return lucid.tensor([0, 2], device=COMPILE_DEVICE)


class _Case(NamedTuple):
    #: Makes the kept tensor, once, on the first call.
    make: Callable[[], Tensor]
    #: The write, run on the kept tensor.
    write: Callable[[Tensor], object]
    #: Whether it runs on every call, or only inside the build.
    every_call: bool = True


#: Writes that read the kept tensor: each call goes on from the last one.
ACCUMULATING: dict[str, _Case] = {
    "add_": _Case(_zeros, lambda h: h.add_(1.0)),
    "sub_": _Case(_zeros, lambda h: h.sub_(0.5)),
    "mul_": _Case(_ones, lambda h: h.mul_(1.5)),
    "div_": _Case(_ones, lambda h: h.div_(2.0)),
    "addcmul_": _Case(_zeros, lambda h: h.addcmul_(_ones(), _const(2.0), value=0.5)),
    "clamp_": _Case(_zeros, lambda h: h.add_(1.0).clamp_(max=3.0)),
    "add_of-a-tensor": _Case(_zeros, lambda h: h.add_(_const(2.0))),
    "through-a-view": _Case(_zeros, lambda h: h[1:].add_(1.0)),
    "index_add_": _Case(
        _zeros, lambda h: h.index_add_(0, _index(), lucid.ones(2, device=COMPILE_DEVICE))
    ),
    "scatter_add_": _Case(
        _zeros, lambda h: h.scatter_add_(0, _index(), lucid.ones(2, device=COMPILE_DEVICE))
    ),
    "copy_-of-itself-plus-a-constant": _Case(_zeros, lambda h: h.copy_(h + 1.0)),
}

#: Writes that only overwrite: the same values however often they run.
OVERWRITING: dict[str, _Case] = {
    "fill_-once": _Case(_zeros, lambda h: h.fill_(2.0), every_call=False),
    "fill_-every-call": _Case(_zeros, lambda h: h.fill_(2.0)),
    "zero_-every-call": _Case(_ones, lambda h: h.zero_()),
    "copy_-once": _Case(_zeros, lambda h: h.copy_(_const(3.0)), every_call=False),
    "copy_-every-call": _Case(_zeros, lambda h: h.copy_(_const(3.0))),
    "setitem-once": _Case(_zeros, lambda h: h.__setitem__(1, 5.0), every_call=False),
    "setitem-every-call": _Case(_zeros, lambda h: h.__setitem__(1, 5.0)),
}

#: Cases the tracer cannot tell from an accumulate, so they run eager: an
#: overwrite through a view (the tracer does not follow a view write to the
#: tensor's value) and a table built by accumulating inside the build (no
#: difference from the repro inside one trace).  A missed compile, never a wrong
#: answer: pinned to agree with eager, not to compile.
CONSERVATIVE: dict[str, _Case] = {
    "view-fill_-once": _Case(_zeros, lambda h: h[1:].fill_(2.0), every_call=False),
    "view-fill_-every-call": _Case(_zeros, lambda h: h[1:].fill_(2.0)),
    "add_-once": _Case(_zeros, lambda h: h.add_(1.0), every_call=False),
}


def _body(case: _Case) -> Callable[[Tensor], Tensor]:
    box: dict[str, Tensor] = {}

    def fn(x: Tensor) -> Tensor:
        if "h" not in box:
            box["h"] = case.make()
            if not case.every_call:
                case.write(box["h"])
        if case.every_call:
            case.write(box["h"])
        return x * box["h"].sum()

    return fn


def _inputs() -> list[Tensor]:
    return [lucid.ones(N, device=COMPILE_DEVICE) * (i + 1.0) for i in range(CALLS)]


def _answers(fn: Callable[[Tensor], Tensor]) -> list[list[float]]:
    with lucid.no_grad():
        return [[float(v) for v in fn(x).tolist()] for x in _inputs()]


def _compare(case: _Case) -> dict[str, object]:
    want = _answers(_body(case))
    compiled = lucid.compile(_body(case))
    got = _answers(compiled)
    assert got == want, f"compiled {got} vs eager {want}"
    return compiled.cache_info()


@pytest.mark.parametrize("name", ACCUMULATING)
def test_a_kept_tensor_a_constant_accumulates_into_goes_on_from_the_last_call(name: str) -> None:
    info = _compare(ACCUMULATING[name])
    assert info["eager_only"], "a write that reads the kept tensor cannot be compiled"


@pytest.mark.parametrize("name", OVERWRITING)
def test_a_kept_tensor_a_constant_overwrites_stays_compiled(name: str) -> None:
    info = _compare(OVERWRITING[name])
    assert not info["eager_only"], f"fell back to eager: {info}"
    assert info["entries"] == 1


@pytest.mark.parametrize("name", CONSERVATIVE)
def test_a_write_the_tracer_cannot_tell_from_an_accumulate_agrees_with_eager(name: str) -> None:
    _compare(CONSERVATIVE[name])


def test_a_tensor_built_once_from_a_constant_chain_stays_compiled() -> None:
    """The table the exemption exists for: made with a setitem, never rewritten."""
    box: dict[str, Tensor] = {}

    def fn(x: Tensor) -> Tensor:
        if "mask" not in box:
            mask = lucid.zeros(N, device=COMPILE_DEVICE)
            mask[0] = 1.0
            mask[2] = 3.0
            box["mask"] = mask
        return x * box["mask"]

    want = _answers(fn)
    box.clear()
    compiled = lucid.compile(fn)
    assert _answers(compiled) == want
    info = compiled.cache_info()
    assert not info["eager_only"] and info["entries"] == 1


def test_the_repro_from_the_issue() -> None:
    box: dict[str, Tensor] = {}

    def f(x: Tensor) -> Tensor:
        if "h" not in box:
            box["h"] = lucid.zeros(3, device=COMPILE_DEVICE)
        box["h"].add_(1.0)
        return x * box["h"].sum()

    x = lucid.ones(3, device=COMPILE_DEVICE)
    compiled = lucid.compile(f)
    with lucid.no_grad():
        got = [float(compiled(x).tolist()[0]) for _ in range(4)]
    assert got == [3.0, 6.0, 9.0, 12.0]
