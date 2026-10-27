"""Graph-mode derivatives through an in-place write, against the reference.

The in-place op's own node read its destination as the write left it when
only the other operand required grad, so ``grad(..., create_graph=True)``
returned ``buf * v`` where the reference returns ``buf`` for
``buf.mul_(v)``.  The oracle-free checks are in
``lucid/test/unit/autograd/test_create_graph_inplace.py``, with the cases
where the two part ways: a saved output written in place (the reference
refuses; metal reads the saved values) and a write into an input the node
did not save (the reference allows it; Lucid refuses, for now).
"""

from collections.abc import Callable
from types import ModuleType

import numpy as np
import pytest

import lucid
import lucid.autograd

pytestmark = pytest.mark.parity

B = [1.0, 2.0, 3.0]
V = [2.0, 3.0, 4.0]

# A derivative that cancels to zero keeps the rounding of the O(1) terms that
# cancel: the reference's second derivative of ``div_ then mul_`` is -2^-23
# where Lucid's is exactly 0.  The defect guarded here is off by O(1).
ATOL = 4 * float(np.finfo(np.float32).eps)


def _derivatives(
    autograd: ModuleType,
    buf: object,
    v: object,
    write: Callable[[object, object], object],
) -> tuple[np.ndarray, np.ndarray]:
    """``d(sum buf)/dv`` after ``write(buf, v)``, and its own derivative —
    zero when the first carries no dependence on ``v``."""
    write(buf, v)
    (g,) = autograd.grad(buf.sum(), v, create_graph=True)
    first = np.asarray(g.detach().tolist(), dtype=np.float64)
    if not g.requires_grad:
        return first, np.zeros_like(first)
    (h,) = autograd.grad(g.sum(), v, allow_unused=True)
    second = np.zeros_like(first) if h is None else np.asarray(h.tolist(), dtype=np.float64)
    return first, second


WRITES: dict[str, Callable[[object, object], object]] = {
    "mul_": lambda b, v: b.mul_(v),
    "add_": lambda b, v: b.add_(v),
    "sub_": lambda b, v: b.sub_(v),
    "div_": lambda b, v: b.div_(v),
    "mul_ twice": lambda b, v: (b.mul_(v), b.mul_(v)),
    "div_ then mul_": lambda b, v: (b.div_(v), b.mul_(v * v)),
}


@pytest.mark.parametrize("write", list(WRITES))
def test_a_destination_without_grad(write: str, device: str, ref: ModuleType) -> None:
    got = _derivatives(
        lucid.autograd,
        lucid.tensor(B, device=device),
        lucid.tensor(V, requires_grad=True, device=device),
        WRITES[write],
    )
    want = _derivatives(
        ref.autograd,
        ref.tensor(B),
        ref.tensor(V, requires_grad=True),
        WRITES[write],
    )
    for g, w in zip(got, want, strict=True):
        np.testing.assert_allclose(g, w, rtol=1e-5, atol=ATOL)


@pytest.mark.parametrize("op", ["mul_", "add_", "sub_", "div_"])
def test_a_write_into_a_gradient(op: str, device: str, ref: ModuleType) -> None:
    # ``p.grad.<op>(w)`` with ``w`` requiring grad: the write lands in the
    # gradient, which then differentiates with respect to ``w``.
    def run(mod: ModuleType, p: object, w: object) -> tuple[list[float], list[float], bool]:
        (p * 2.0).sum().backward()
        g = p.grad
        getattr(g, op)(w)
        (h,) = mod.autograd.grad(g.sum(), w)
        return p.grad.tolist(), h.tolist(), bool(g.requires_grad)

    got = run(
        lucid,
        lucid.tensor([1.0, 2.0, 3.0], requires_grad=True, device=device),
        lucid.tensor(V, requires_grad=True, device=device),
    )
    want = run(
        ref,
        ref.tensor([1.0, 2.0, 3.0], requires_grad=True),
        ref.tensor(V, requires_grad=True),
    )
    np.testing.assert_allclose(got[0], want[0], rtol=1e-6)
    np.testing.assert_allclose(got[1], want[1], rtol=1e-6)
    assert got[2] == want[2]


def test_the_operand_is_the_destination(device: str, ref: ModuleType) -> None:
    # The reference refuses ``y.mul_(y)`` at backward — its node saved ``y``
    # and the write moved it.  Lucid's node reads the ``y`` it was handed,
    # so the write differentiates as ``y * y`` does, which is the oracle.
    def run(
        mod: ModuleType, x: object, square: Callable[[object], object]
    ) -> tuple[np.ndarray, np.ndarray]:
        y = square(x * 1.0)
        (g,) = mod.autograd.grad(y.sum(), x, create_graph=True)
        (h,) = mod.autograd.grad(g.sum(), x)
        return np.asarray(g.detach().tolist()), np.asarray(h.tolist())

    got = run(
        lucid,
        lucid.tensor([1.0, 2.0], requires_grad=True, device=device),
        lambda y: y.mul_(y),
    )
    want = run(ref, ref.tensor([1.0, 2.0], requires_grad=True), lambda y: y * y)
    for g, w in zip(got, want):
        np.testing.assert_allclose(g, w, rtol=1e-5)
