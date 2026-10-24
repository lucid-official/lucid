"""Code that changes a gradient writes into it, so a held ``.grad`` sees it (LCD-546).

The reference's ``p.grad`` is one tensor, shared by everything that holds
it: a hook, a gradient-accumulation buffer, a loop that kept ``g = p.grad``.
``zero_grad(set_to_none=False)``, ``clip_grad_norm_``, ``clip_grad_value_``
and ``GradScaler.unscale_`` change that tensor in place.  Lucid installed a
new gradient instead, so ``g`` kept the old values while ``p.grad`` moved on.
``LBFGS.zero_grad`` also ignored ``set_to_none`` and always dropped the
gradient.  All of these now write through ``lucid.autograd._grad_inplace``.

Every ``p.grad`` read here builds a new Python wrapper.  "The same tensor"
therefore means the same engine tensor, ``g._impl is p.grad._impl``: a write
through either reaches the other.
"""

from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import lucid
import lucid.nn as nn
import lucid.optim as optim

type Reset = Callable[[Any, Any, bool], None]


def _optimizer(ns: Any, p: Any, set_to_none: bool) -> None:
    ns.optim.SGD([p], lr=0.1).zero_grad(set_to_none=set_to_none)


def _module(ns: Any, p: Any, set_to_none: bool) -> None:
    m = ns.nn.Module()
    m.p = p
    m.zero_grad(set_to_none=set_to_none)


def _lbfgs(ns: Any, p: Any, set_to_none: bool) -> None:
    ns.optim.LBFGS([p], lr=1.0).zero_grad(set_to_none=set_to_none)


_RESETS: dict[str, Reset] = {
    "Optimizer.zero_grad": _optimizer,
    "Module.zero_grad": _module,
    "LBFGS.zero_grad": _lbfgs,
}


def _same(a: lucid.Tensor, b: lucid.Tensor | None) -> bool:
    return b is not None and a._impl is b._impl


# ── zero_grad: {optimizer, module, LBFGS} × set_to_none × device ─────────────


@pytest.mark.parametrize("reset", _RESETS.values(), ids=_RESETS.keys())
def test_zero_grad_zeroes_the_held_gradient(reset: Reset, device: str) -> None:
    p = nn.Parameter(lucid.ones(3, device=device))
    (p * 2).sum().backward()
    g = p.grad
    assert g is not None
    before = g._version
    reset(lucid, p, False)
    assert g.tolist() == [0.0, 0.0, 0.0]
    assert _same(g, p.grad)
    assert g._version > before
    # The zeroed buffer stays the gradient: the next backward adds into it.
    (p * 3).sum().backward()
    assert g.tolist() == [3.0, 3.0, 3.0]
    assert _same(g, p.grad)


@pytest.mark.parametrize("reset", _RESETS.values(), ids=_RESETS.keys())
def test_set_to_none_drops_the_gradient_and_leaves_the_held_one(
    reset: Reset, device: str
) -> None:
    p = nn.Parameter(lucid.ones(3, device=device))
    (p * 2).sum().backward()
    g = p.grad
    assert g is not None
    reset(lucid, p, True)
    assert p.grad is None
    assert g.tolist() == [2.0, 2.0, 2.0]


@pytest.mark.parametrize("reset", _RESETS.values(), ids=_RESETS.keys())
def test_a_graph_gradient_is_detached_and_zeroed(reset: Reset, device: str) -> None:
    p = nn.Parameter(lucid.ones(3, device=device))
    (p * p).sum().backward(create_graph=True)
    g = p.grad
    assert g is not None and g.grad_fn is not None
    reset(lucid, p, False)
    assert g.tolist() == [0.0, 0.0, 0.0]
    assert g.grad_fn is None and not g.requires_grad
    assert p.grad is not None and p.grad.grad_fn is None
    assert p.grad.tolist() == [0.0, 0.0, 0.0]


@pytest.mark.parametrize("make", [optim.SGD, optim.Adam], ids=["SGD", "Adam"])
def test_a_step_after_zeroing_a_graph_gradient_uses_the_next_gradient(
    make: Callable[..., optim.Optimizer], device: str
) -> None:
    """The gradient goes back to where the engine optimizers read it."""
    p = nn.Parameter(lucid.ones(3, device=device))
    o = make([p], lr=0.1)
    (p * p).sum().backward(create_graph=True)
    o.zero_grad(set_to_none=False)
    (p * 3).sum().backward()
    o.step()
    q = nn.Parameter(lucid.ones(3, device=device))
    oq = make([q], lr=0.1)
    (q * 3).sum().backward()
    oq.step()
    assert p.tolist() == q.tolist() != [1.0, 1.0, 1.0]


@pytest.mark.xfail(
    strict=True,
    reason="LCD-546 FOUND: a graph-mode gradient lives apart from the slot "
    "the engine optimizers read, so zeroing hands p.grad a new tensor",
)
def test_a_zeroed_graph_gradient_stays_the_gradient(device: str) -> None:
    p = nn.Parameter(lucid.ones(3, device=device))
    (p * p).sum().backward(create_graph=True)
    g = p.grad
    assert g is not None
    optim.SGD([p], lr=0.1).zero_grad(set_to_none=False)
    assert _same(g, p.grad)


def test_an_assigned_gradient_that_requires_grad_stops_requiring_it(
    device: str,
) -> None:
    p = nn.Parameter(lucid.ones(3, device=device))
    w = lucid.full((3,), 2.0, device=device, requires_grad=True)
    p.grad = w
    optim.SGD([p], lr=0.1).zero_grad(set_to_none=False)
    assert w.tolist() == [0.0, 0.0, 0.0] and not w.requires_grad
    assert p.grad is not None and not p.grad.requires_grad
    assert p.grad.tolist() == [0.0, 0.0, 0.0]


# ── the other writers: clip_grad_norm_, clip_grad_value_, GradScaler ────────


def _clip_norm(ns: Any, p: Any) -> None:
    ns.nn.utils.clip_grad_norm_([p], 1.0)


def _clip_value(ns: Any, p: Any) -> None:
    ns.nn.utils.clip_grad_value_([p], 0.5)


def _unscale(ns: Any, p: Any) -> None:
    # The gradient is taken as one of a loss scaled by 4; ``scale`` is still
    # called first, since the reference's scaler refuses an unscale before it.
    scaler = (
        ns.amp.GradScaler(init_scale=4.0)
        if ns is lucid
        else ns.amp.GradScaler("cpu", init_scale=4.0)
    )
    scaler.scale(p.detach().sum())
    scaler.unscale_(ns.optim.SGD([p], lr=0.1))


_WRITERS: dict[str, Callable[[Any, Any], None]] = {
    "clip_grad_norm_": _clip_norm,
    "clip_grad_value_": _clip_value,
    "GradScaler.unscale_": _unscale,
}


def _write(
    ns: Any, write: Callable[[Any, Any], None], device: str
) -> tuple[list[float], bool, int]:
    """The held gradient after ``write``: values, still the gradient, versions moved."""
    kwargs = {"device": device} if ns is lucid else {}
    p = ns.nn.Parameter(ns.ones(3, **kwargs))
    (p * 8).sum().backward()
    g = p.grad
    before = g._version
    write(ns, p)
    same = _same(g, p.grad) if ns is lucid else g is p.grad
    return g.tolist(), same, g._version - before


@pytest.mark.parametrize("write", _WRITERS.values(), ids=_WRITERS.keys())
def test_the_writers_change_the_held_gradient(
    write: Callable[[Any, Any], None], device: str
) -> None:
    values, same, moved = _write(lucid, write, device)
    assert values != [8.0, 8.0, 8.0] and same and moved > 0


@pytest.mark.parity
@pytest.mark.parametrize("write", _WRITERS.values(), ids=_WRITERS.keys())
def test_the_writers_match_the_reference(
    write: Callable[[Any, Any], None], device: str, ref: ModuleType
) -> None:
    # Versions are left out: the reference's fused unscale does not move one.
    values, same, _ = _write(lucid, write, device)
    ref_values, ref_same, _ = _write(ref, write, "cpu")
    assert values == pytest.approx(ref_values) and same == ref_same


@pytest.mark.parity
@pytest.mark.parametrize("reset", _RESETS.values(), ids=_RESETS.keys())
@pytest.mark.parametrize("set_to_none", [False, True])
def test_zero_grad_matches_the_reference(
    reset: Reset, set_to_none: bool, device: str, ref: ModuleType
) -> None:
    def run(ns: Any, **kwargs: str) -> tuple[list[float], bool, bool]:
        p = ns.nn.Parameter(ns.ones(3, **kwargs))
        (p * 2).sum().backward()
        g = p.grad
        reset(ns, p, set_to_none)
        same = _same(g, p.grad) if ns is lucid else g is p.grad
        return g.tolist(), same, p.grad is None

    assert run(lucid, device=device) == run(ref)


def test_unscaling_still_finds_an_inf_per_device(device: str) -> None:
    p = nn.Parameter(lucid.ones(3, device=device))
    o = optim.SGD([p], lr=0.1)
    scaler = lucid.amp.GradScaler(init_scale=4.0)
    p.grad = lucid.tensor([1.0, float("inf"), 2.0], device=device)
    g = p.grad
    assert g is not None
    scaler.unscale_(o)
    assert g.tolist()[0] == 0.25 and g.tolist()[2] == 0.5
    with pytest.raises(RuntimeError, match="already been called"):
        scaler.unscale_(o)
    scaler.step(o)
    assert p.tolist() == [1.0, 1.0, 1.0]
