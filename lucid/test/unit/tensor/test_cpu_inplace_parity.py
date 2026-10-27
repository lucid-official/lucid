"""In-place writes on the CPU agree with the reference and with Metal.

Two CPU behaviours had drifted from both (the Metal side is pinned in
``test_metal_view_writes.py``):

* A write into an expanded view went through.  ``lucid.ones(1, 3)
  .expand(2, 3).add_(1.0)`` raised nothing: the check for elements that share
  a byte guarded only the write into a view whose base was still alive, and
  with the base gone the op handed the view a dense buffer of its own.
* A write into a gradient did not reach it.  ``p.grad.mul_(s)`` was refused
  as storage "shared with a view" — the slot ``.grad`` reads counted as one
  more holder of the buffer — while ``p.grad.neg_()`` and
  ``p.grad.clamp_(...)`` took a buffer of their own and left the gradient as
  it was.  Gradient clipping and loss unscaling write exactly this way.

Every case runs on each available device and is held to the same answer, but
one that Metal still gets wrong and is marked so: a tensor read from ``.grad``
before a write keeps its old values there.  The refusals the gradient write
keeps — a tensor saved for backward, a NumPy array over the buffer — are the
engine's own and are pinned on the CPU.
"""

from collections.abc import Callable

import numpy as np
import pytest

import lucid
import lucid.nn as nn

# ── expanded views ───────────────────────────────────────────────────────────

_EXPANDED_WRITES: dict[str, Callable[[lucid.Tensor], object]] = {
    "add_": lambda e: e.add_(1.0),
    "sub_ tensor": lambda e: e.sub_(lucid.ones(2, 3).to(e.device)),
    "mul_": lambda e: e.mul_(2.0),
    "div_": lambda e: e.div_(2.0),
    "pow_": lambda e: e.pow_(2.0),
    "neg_": lambda e: e.neg_(),
    "exp_": lambda e: e.exp_(),
    "relu_": lambda e: e.relu_(),
    "clamp_": lambda e: e.clamp_(0.0, 0.5),
    "+=": lambda e: e.__iadd__(1.0),
    "*=": lambda e: e.__imul__(2.0),
    "copy_": lambda e: e.copy_(lucid.zeros(2, 3).to(e.device)),
}


@pytest.mark.parametrize("base_alive", [True, False], ids=["base alive", "base gone"])
@pytest.mark.parametrize("name", list(_EXPANDED_WRITES))
def test_a_write_into_an_expanded_view_is_refused(device: str, name: str, base_alive: bool) -> None:
    base = lucid.tensor([[1.0, 2.0, 3.0]]).to(device)
    expanded = base.expand(2, 3)
    if not base_alive:
        del base
    with pytest.raises(RuntimeError, match="overlap"):
        _EXPANDED_WRITES[name](expanded)
    assert expanded.tolist() == [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    if base_alive:
        assert base.tolist() == [[1.0, 2.0, 3.0]]


def test_overlapping_and_leaf_views_are_refused(device: str) -> None:
    expanded = lucid.ones(1, 3).to(device).expand(2, 3)
    with pytest.raises(RuntimeError, match="overlap"):
        expanded.add_(1.0)
    leaf = lucid.ones(3).to(device).requires_grad_()
    with pytest.raises(RuntimeError, match="leaf tensor that requires grad"):
        leaf[0].mul_(2.0)
    with lucid.no_grad():
        leaf[0].mul_(2.0)
    assert leaf.tolist() == [2.0, 1.0, 1.0]


# ── gradients ────────────────────────────────────────────────────────────────


def _with_grad(device: str) -> lucid.Tensor:
    """A leaf whose gradient is ``[2, 4, 6]``."""
    x = lucid.tensor([1.0, 2.0, 3.0]).to(device).requires_grad_()
    (x * x).sum().backward()
    return x


def _augmented(x: lucid.Tensor) -> None:
    g = x.grad
    g += 1.0


_GRAD_WRITES: dict[str, tuple[Callable[[lucid.Tensor], object], list[float]]] = {
    "mul_": (lambda x: x.grad.mul_(10.0), [20.0, 40.0, 60.0]),
    "div_": (lambda x: x.grad.div_(2.0), [1.0, 2.0, 3.0]),
    "sub_ tensor": (lambda x: x.grad.sub_(lucid.ones(3).to(x.device)), [1.0, 3.0, 5.0]),
    "neg_": (lambda x: x.grad.neg_(), [-2.0, -4.0, -6.0]),
    "clamp_": (lambda x: x.grad.clamp_(3.0, 5.0), [3.0, 4.0, 5.0]),
    "augmented": (_augmented, [3.0, 5.0, 7.0]),
    "itself": (lambda x: x.grad.mul_(x.grad), [4.0, 16.0, 36.0]),
    "a view": (lambda x: x.grad.view(-1).mul_(2.0), [4.0, 8.0, 12.0]),
    "a slice": (lambda x: x.grad[1:].mul_(0.5), [2.0, 2.0, 3.0]),
    "detach()": (lambda x: x.grad.detach().mul_(3.0), [6.0, 12.0, 18.0]),
    "zero_": (lambda x: x.grad.zero_(), [0.0, 0.0, 0.0]),
}


@pytest.mark.parametrize("name", list(_GRAD_WRITES))
def test_an_in_place_write_lands_in_the_gradient(device: str, name: str) -> None:
    write, expected = _GRAD_WRITES[name]
    x = _with_grad(device)
    write(x)
    assert x.grad.tolist() == expected


def test_data_detach_and_grad_write_through(device: str) -> None:
    p = lucid.arange(4.0).to(device).requires_grad_()
    p.data.mul_(10.0)
    assert p.tolist() == [0.0, 10.0, 20.0, 30.0]
    q = lucid.arange(4.0).to(device)
    q.detach().add_(100.0)
    assert q.tolist() == [100.0, 101.0, 102.0, 103.0]
    g = lucid.ones(3).to(device).requires_grad_()
    (g * 2.0).sum().backward()
    g.grad.mul_(10.0)
    assert g.grad.tolist() == [20.0, 20.0, 20.0]
    g.grad.zero_()
    assert g.grad.tolist() == [0.0, 0.0, 0.0]


def test_a_gradient_read_earlier_does_not_stop_the_write(device: str) -> None:
    # ``g = p.grad`` kept alive is the same gradient, not a second holder of
    # its buffer, so ``p.grad.mul_`` still writes — and ``g`` reads the result.
    x = _with_grad(device)
    held = x.grad
    x.grad.mul_(2.0)
    assert x.grad.tolist() == [4.0, 8.0, 12.0]
    held.add_(1.0)
    assert x.grad.tolist() == [5.0, 9.0, 13.0]


def test_backward_accumulates_into_the_written_gradient(device: str) -> None:
    x = _with_grad(device)
    x.grad.mul_(0.5)
    (x * x).sum().backward()
    assert x.grad.tolist() == [3.0, 6.0, 9.0]


def test_clipping_by_hand_then_stepping_moves_by_the_clipped_gradient(device: str) -> None:
    lucid.manual_seed(0)
    layer = nn.Linear(4, 3).to(device)
    x = lucid.randn(8, 4).to(device)
    (layer(x) ** 2).mean().backward()
    params = list(layer.parameters())
    before = [p.numpy().copy() for p in params]
    grads = [p.grad.numpy().copy() for p in params]
    total = float(np.sqrt(sum(float((g**2).sum()) for g in grads)))
    coef = 0.5 / total  # clip to a norm of 0.5
    for p in params:
        p.grad.mul_(coef)
    for p, g in zip(params, grads):
        np.testing.assert_allclose(p.grad.numpy(), g * coef, rtol=1e-5, atol=1e-7)
    lucid.optim.SGD(params, lr=0.1).step()
    for p, b, g in zip(params, before, grads):
        np.testing.assert_allclose(p.numpy(), b - 0.1 * g * coef, rtol=1e-5, atol=1e-7)


# ── what a gradient write still refuses (the engine's own, so the CPU) ───────


def test_a_gradient_saved_for_backward_is_not_written() -> None:
    # The product saved the gradient for ``w``'s backward; writing the new
    # values under it would change what that backward reads.
    x = _with_grad("cpu")
    w = lucid.ones(3, requires_grad=True)
    y = (w * x.grad).sum()
    with pytest.raises(RuntimeError, match="shares storage"):
        x.grad.mul_(2.0)
    with pytest.raises(RuntimeError, match="gradient"):
        x.grad.neg_()
    assert x.grad.tolist() == [2.0, 4.0, 6.0]
    y.backward()
    assert w.grad.tolist() == [2.0, 4.0, 6.0]


def test_a_gradient_that_is_another_gradient_is_not_written(device: str) -> None:
    """A hook that returns ``p.grad`` gives ``x`` a copy of it.

    The hook runs inside backward now, before the gradient is accumulated,
    and the engine copies what a hook hands back — as the reference's
    accumulator clones it — so ``x.grad`` is a buffer of its own and a
    write to it leaves ``p.grad`` alone.  ``q.grad = p.grad`` shares the
    buffer instead, so ``q.grad.mul_(s)`` and ``q.grad.neg_()`` write
    through to both ``q.grad`` and ``p.grad``, matching ``zero_`` and the
    reference.
    """
    p = _with_grad(device)
    x = lucid.tensor([1.0, 1.0, 1.0], device=device, requires_grad=True)
    x.register_hook(lambda g: p.grad)
    (x * 1.0).sum().backward()
    x.grad.mul_(10.0)
    assert x.grad.tolist() == [20.0, 40.0, 60.0]
    assert p.grad.tolist() == [2.0, 4.0, 6.0]
    q = lucid.zeros(3, device=device, requires_grad=True)
    q.grad = p.grad
    q.grad.mul_(10.0)
    assert q.grad.tolist() == [20.0, 40.0, 60.0]
    assert p.grad.tolist() == [20.0, 40.0, 60.0]
    q.grad.neg_()
    assert q.grad.tolist() == [-20.0, -40.0, -60.0]
    assert p.grad.tolist() == [-20.0, -40.0, -60.0]


def test_a_numpy_array_over_a_gradient_stops_the_write() -> None:
    x = _with_grad("cpu")
    arr = x.grad.numpy()
    with pytest.raises(RuntimeError, match="gradient"):
        x.grad.mul_(2.0)
    assert arr.tolist() == [2.0, 4.0, 6.0]
    del arr
    x.grad.mul_(2.0)
    assert x.grad.tolist() == [4.0, 8.0, 12.0]


def test_a_gradient_let_go_of_is_an_ordinary_tensor() -> None:
    # Once ``.grad`` is cleared, a tensor read from it earlier no longer
    # stands for anything, and a write to it is an ordinary write.
    x = _with_grad("cpu")
    old = x.grad
    x.grad = None
    old.mul_(2.0)
    assert old.tolist() == [4.0, 8.0, 12.0]
    assert x.grad is None
