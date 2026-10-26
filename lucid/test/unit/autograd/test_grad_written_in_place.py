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
    assert _same(g, p.grad)
    (p * 3).sum().backward()
    assert g.tolist() == [3.0, 3.0, 3.0]
    assert _same(g, p.grad)


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


def _graph_gradient(p: nn.Parameter) -> None:
    (p * p).sum().backward(create_graph=True)


def _assigned_tracked(p: nn.Parameter) -> None:
    p.grad = lucid.full((3,), 2.0, device=p.device, requires_grad=True)


# Both kinds keep the gradient apart from the slot the optimizers read.
_HELD_APART: dict[str, Callable[[nn.Parameter], None]] = {
    "create_graph": _graph_gradient,
    "assigned_requires_grad": _assigned_tracked,
}


def test_detaching_the_gradient_detaches_it(device: str) -> None:
    p = nn.Parameter(lucid.ones(3, device=device))
    (p * p).sum().backward(create_graph=True)
    g = p.grad
    assert g is not None
    v = g.view_as(g)
    ver_before = g._version
    g.detach_()
    assert g._version == ver_before
    assert _same(g, p.grad)
    assert p.grad is not None and p.grad.grad_fn is None
    assert not p.grad.requires_grad and not g.requires_grad
    assert v.grad_fn is not None

    # A subsequent non-graph backward accumulates into the detached grad in place.
    (p * 3.0).sum().backward()
    assert g._version == ver_before + 1
    assert v._version == ver_before + 1
    assert p.grad.tolist() == [5.0, 5.0, 5.0]
    assert v.tolist() == [5.0, 5.0, 5.0]


def test_detach_in_place_detaches_every_holder_of_the_same_impl(
    device: str,
) -> None:
    x = lucid.tensor([1.0, 2.0, 3.0], device=device, requires_grad=True)
    y = x * 2.0
    holder = lucid.Tensor.__new_from_impl__(y._impl)
    v = y[1:]
    v_holder = lucid.Tensor.__new_from_impl__(v._impl)

    ver_v = v._version
    v.detach_()
    assert v._version == ver_v
    assert v.grad_fn is None and not v.requires_grad and v.is_leaf
    assert v_holder.grad_fn is None and not v_holder.requires_grad and v_holder.is_leaf

    ver_y = y._version
    holder.detach_()
    assert y._version == ver_y
    assert y.grad_fn is None and not y.requires_grad and y.is_leaf
    assert holder.grad_fn is None and not holder.requires_grad and holder.is_leaf



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
    assert _same(w, p.grad)


@pytest.mark.parametrize("reset", _RESETS.values(), ids=_RESETS.keys())
@pytest.mark.parametrize("kind", _HELD_APART.values(), ids=_HELD_APART.keys())
def test_a_held_apart_gradient_is_replaced_by_zeros(
    kind: Callable[[nn.Parameter], None], reset: Reset, device: str
) -> None:
    p = nn.Parameter(lucid.ones(3, device=device))
    kind(p)
    reset(lucid, p, False)
    assert p.grad is not None and p.grad.tolist() == [0.0, 0.0, 0.0]
    assert p.grad.grad_fn is None and not p.grad.requires_grad


# The engine keeps a parameter's first graph-mode gradient as it arrives, so
# it can be another tensor.  Zeroing must leave that tensor alone.  Each case
# builds the gradient and returns the tensor that must survive.
def _shares_with_another_gradient(p: nn.Parameter) -> lucid.Tensor:
    q = nn.Parameter(lucid.ones(3, device=p.device))
    (p + q).sum().backward(create_graph=True)
    assert q.grad is not None
    return q.grad


def _is_the_backward_seed(p: nn.Parameter) -> lucid.Tensor:
    seed = lucid.full((3,), 5.0, device=p.device)
    p.clone().backward(seed, create_graph=True)
    return seed


def _is_a_parameter_seed(p: nn.Parameter) -> lucid.Tensor:
    w = nn.Parameter(lucid.full((3,), 7.0, device=p.device))
    p.view(3).backward(w, create_graph=True)
    return w


def _is_a_view_of_a_leaf(p: nn.Parameter) -> lucid.Tensor:
    big = lucid.full((6,), 3.0, device=p.device, requires_grad=True)
    p.grad = big[0:3]
    return big


_OTHERS: dict[str, Callable[[nn.Parameter], lucid.Tensor]] = {
    "another_parameters_gradient": _shares_with_another_gradient,
    "the_gradient_argument": _is_the_backward_seed,
    "a_parameter_as_the_seed": _is_a_parameter_seed,
    "a_view_of_a_leaf": _is_a_view_of_a_leaf,
}


@pytest.mark.parametrize("reset", _RESETS.values(), ids=_RESETS.keys())
@pytest.mark.parametrize("make", _OTHERS.values(), ids=_OTHERS.keys())
def test_zeroing_leaves_the_tensor_a_gradient_came_from(
    make: Callable[[nn.Parameter], lucid.Tensor], reset: Reset, device: str
) -> None:
    p = nn.Parameter(lucid.ones(3, device=device))
    other = make(p)
    before = other.tolist()
    reset(lucid, p, False)
    assert other.tolist() == before
    assert p.grad is not None and p.grad.tolist() == [0.0, 0.0, 0.0]


# ── the other writers: clip_grad_norm_, clip_grad_value_, GradScaler, maximize ─


def _clip_norm(ns: Any, p: Any) -> None:
    ns.nn.utils.clip_grad_norm_([p], 1.0)


def _clip_value(ns: Any, p: Any) -> None:
    ns.nn.utils.clip_grad_value_([p], 0.5)


def _scaler(ns: Any) -> Any:
    # ``scale`` is called first: the reference's scaler refuses an unscale
    # before it.  The gradients are taken as those of a loss scaled by 4.
    scaler = (
        ns.amp.GradScaler(init_scale=4.0)
        if ns is lucid
        else ns.amp.GradScaler("cpu", init_scale=4.0)
    )
    scaler.scale(ns.ones(()))
    return scaler


def _unscale(ns: Any, p: Any) -> None:
    _scaler(ns).unscale_(ns.optim.SGD([p], lr=0.1))


def _maximize(ns: Any, p: Any) -> None:
    ns.optim.SGD([p], lr=0.1, maximize=True).step()


# Each writer, and what it leaves in a gradient of eights.
_WRITERS = [
    pytest.param(_clip_norm, [3**-0.5] * 3, id="clip_grad_norm_"),
    pytest.param(_clip_value, [0.5] * 3, id="clip_grad_value_"),
    pytest.param(_unscale, [2.0] * 3, id="GradScaler.unscale_"),
    pytest.param(_maximize, [8.0] * 3, id="maximize_step"),
]


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


@pytest.mark.parametrize(("write", "expected"), _WRITERS)
def test_the_writers_change_the_held_gradient(
    write: Callable[[Any, Any], None], expected: list[float], device: str
) -> None:
    values, same, moved = _write(lucid, write, device)
    assert values == pytest.approx(expected) and same
    assert moved > 0 or expected == [8.0] * 3


@pytest.mark.parity
@pytest.mark.parametrize(("write", "expected"), _WRITERS)
def test_the_writers_match_the_reference(
    write: Callable[[Any, Any], None],
    expected: list[float],
    device: str,
    ref: ModuleType,
) -> None:
    # Versions are left out: the reference's fused unscale does not move one.
    values, same, _ = _write(lucid, write, device)
    ref_values, ref_same, _ = _write(ref, write, "cpu")
    assert ref_values == pytest.approx(expected)
    assert values == pytest.approx(ref_values) and same == ref_same


# ── a gradient held apart from the slot: the next step must still read it ───


def _clip_norm_step(p: nn.Parameter, o: optim.Optimizer) -> None:
    nn.utils.clip_grad_norm_([p], 1.0)
    o.step()


def _clip_value_step(p: nn.Parameter, o: optim.Optimizer) -> None:
    nn.utils.clip_grad_value_([p], 0.5)
    o.step()


def _unscale_step(p: nn.Parameter, o: optim.Optimizer) -> None:
    scaler = _scaler(lucid)
    scaler.unscale_(o)
    scaler.step(o)
    scaler.update()


# Each writer then a step, and where SGD(lr=0.1) takes [1, 2, 3] from a
# gradient of [2, 4, 6]: main's values before LCD-546.
_STEPS = [
    pytest.param(_clip_norm_step, [0.97327, 1.94655, 2.91982], id="clip_grad_norm_"),
    pytest.param(_clip_value_step, [0.95, 1.95, 2.95], id="clip_grad_value_"),
    pytest.param(_unscale_step, [0.95, 1.9, 2.85], id="GradScaler.unscale_"),
]
_OPTIMIZERS = {"SGD": optim.SGD, "Adam": optim.Adam}


def _step_after(
    write: Callable[[nn.Parameter, optim.Optimizer], None],
    make: Callable[..., optim.Optimizer],
    device: str,
    create_graph: bool,
    *,
    in_no_grad: bool = True,
) -> list[float]:
    p = nn.Parameter(lucid.tensor([1.0, 2.0, 3.0], device=device))
    o = make([p], lr=0.1)
    (p * p).sum().backward(create_graph=create_graph)
    if in_no_grad:
        with lucid.no_grad():
            write(p, o)
    else:
        write(p, o)
    values: list[float] = p.tolist()
    return values


@pytest.mark.parametrize("in_no_grad", [True, False], ids=["no_grad", "grad_mode"])
@pytest.mark.parametrize("make", _OPTIMIZERS.values(), ids=_OPTIMIZERS.keys())
@pytest.mark.parametrize(("write", "sgd_expected"), _STEPS)
def test_a_step_reads_what_a_writer_left_in_a_graph_gradient(
    write: Callable[[nn.Parameter, optim.Optimizer], None],
    sgd_expected: list[float],
    make: Callable[..., optim.Optimizer],
    in_no_grad: bool,
    device: str,
) -> None:
    """After ``backward(create_graph=True)`` the step moves as after a first-order one."""
    moved = _step_after(write, make, device, create_graph=True, in_no_grad=in_no_grad)
    assert moved == pytest.approx(
        _step_after(write, make, device, create_graph=False, in_no_grad=in_no_grad)
    )
    assert moved != pytest.approx([1.0, 2.0, 3.0])
    if make is optim.SGD:
        assert moved == pytest.approx(sgd_expected, abs=1e-5)


@pytest.mark.parametrize("make", _OPTIMIZERS.values(), ids=_OPTIMIZERS.keys())
def test_optimizer_step_moves_parameter_with_create_graph_gradient(
    make: Callable[..., optim.Optimizer], device: str
) -> None:
    p = nn.Parameter(lucid.tensor([1.0, 2.0, 3.0], device=device))
    o = make([p], lr=0.1)
    (p * p).sum().backward(create_graph=True)
    g = p.grad
    assert g is not None and g.requires_grad
    o.step()
    assert _same(g, p.grad)
    q = nn.Parameter(lucid.tensor([1.0, 2.0, 3.0], device=device))
    oq = make([q], lr=0.1)
    (q * q).sum().backward(create_graph=False)
    oq.step()
    assert p.tolist() == pytest.approx(q.tolist(), abs=1e-5)


def _zero_grad(p: nn.Parameter) -> None:
    optim.SGD([p], lr=0.1).zero_grad(set_to_none=False)


def _no_grad(write: Callable[[Any, Any], None]) -> Callable[[nn.Parameter], None]:
    def run(p: nn.Parameter) -> None:
        with lucid.no_grad():
            write(lucid, p)

    return run


_HELD_APART_WRITERS: dict[str, Callable[[nn.Parameter], None]] = {
    "zero_grad": _zero_grad,
    "clip_grad_norm_": _no_grad(_clip_norm),
    "clip_grad_value_": _no_grad(_clip_value),
    "GradScaler.unscale_": _no_grad(_unscale),
}


@pytest.mark.parametrize(
    "write", _HELD_APART_WRITERS.values(), ids=_HELD_APART_WRITERS.keys()
)
@pytest.mark.parametrize("kind", _HELD_APART.values(), ids=_HELD_APART.keys())
def test_a_written_held_apart_gradient_stays_the_gradient(
    kind: Callable[[nn.Parameter], None],
    write: Callable[[nn.Parameter], None],
    device: str,
) -> None:
    p = nn.Parameter(lucid.tensor([1.0, 2.0, 3.0], device=device))
    kind(p)
    g = p.grad
    assert g is not None
    write(p)
    assert _same(g, p.grad)


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
