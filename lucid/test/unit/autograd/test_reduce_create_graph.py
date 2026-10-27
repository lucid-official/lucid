"""Regression tests: prod / max / min lost their rule under create_graph.

Found 2026-08-02 by comparing ``autograd.grad`` against ``backward()`` for
every op.  ``x.prod().backward()`` gave the right gradient;
``autograd.grad(x.prod(), [x], create_graph=True)`` gave ``[1,1,1,1,1,1]``
— the incoming seed, broadcast to every position, with no error.

``ReduceKernel::apply_for_graph`` broadcasts the upstream gradient back
over the reduced axes — which is the **sum** rule — and then calls
``Derived::scale_graph_grad`` to apply the per-op factor.  Only
``SumBackward`` (identity) and ``MeanBackward`` (1/n) overrode it, so
``prod``, ``max`` and ``min`` silently inherited sum's rule.  Twelve other
ops without a graph formula raise a clear "not yet supported"; these three
did not, which is what made them dangerous rather than merely limited.

``max`` and ``min`` are the ones that matter in practice: they appear in
attention, pooling and every stabilised softmax, so any double-backward
through them — a gradient penalty, a meta-learning inner loop — was
silently wrong.

The invariant asserted is that ``autograd.grad`` agrees with
``backward()``.  ``backward()`` is the reference because it is the path
every model in the zoo trains through.
"""

import numpy as np
import pytest

import lucid
import lucid.autograd

_BASE = np.array([[0.4, 1.1, 0.7], [1.6, 0.9, 1.3]], dtype=np.float64)
_SEED = np.array([0.7, -1.3, 0.4, 2.1, -0.6, 1.5], dtype=np.float64)


def _f64(a):
    return lucid.tensor(np.asarray(a, dtype=np.float64), dtype=lucid.float64)


def _loss(fn, x):
    flat = fn(x).reshape(-1)
    n = int(flat.shape[0])
    return (flat * _f64(_SEED[:n])).sum()


_REDUCTIONS = [
    ("sum", lambda t: t.sum()),
    ("mean", lambda t: t.mean()),
    ("prod", lambda t: t.prod()),
    ("max", lambda t: t.max()),
    ("min", lambda t: t.min()),
]


@pytest.mark.parametrize("name,fn", _REDUCTIONS)
@pytest.mark.parametrize("create_graph", [False, True])
def test_grad_agrees_with_backward(name, fn, create_graph):
    """The defect: create_graph=True returned the seed for prod / max / min."""
    reference = _f64(_BASE).requires_grad_(True)
    _loss(fn, reference).backward()

    probe = _f64(_BASE).requires_grad_(True)
    (got,) = lucid.autograd.grad(_loss(fn, probe), [probe], create_graph=create_graph)
    assert np.allclose(got.numpy(), reference.grad.numpy(), atol=1e-12), name


def test_the_seed_is_not_the_answer() -> None:
    """Guard the instrument.

    Every check above compares two entry points.  Pin the numbers for one
    case so that a future regression to "pass the seed through" cannot
    hide by breaking both: prod's gradient is ``seed * prod(x) / x_i``,
    which is not the seed.
    """
    x = _f64(_BASE).requires_grad_(True)
    (g,) = lucid.autograd.grad(_loss(lambda t: t.prod(), x), [x], create_graph=True)
    want = _SEED[0] * _BASE.prod() / _BASE.reshape(-1)
    assert np.allclose(g.numpy().reshape(-1), want)
    assert not np.allclose(g.numpy().reshape(-1), _SEED[0])


def test_max_routes_the_gradient_only_to_the_maximum() -> None:
    """Sum's rule would put the gradient everywhere; max's puts it in one place."""
    x = _f64(_BASE).requires_grad_(True)
    (g,) = lucid.autograd.grad(_loss(lambda t: t.max(), x), [x], create_graph=True)
    flat = g.numpy().reshape(-1)
    argmax = int(np.argmax(_BASE))
    assert np.count_nonzero(flat) == 1
    assert np.isclose(flat[argmax], _SEED[0])


def test_max_splits_a_tie_the_same_way_as_backward() -> None:
    """Both modes use an equality mask, so tied maxima each get the gradient."""
    tied = np.array([[2.0, 1.0], [2.0, 0.5]], dtype=np.float64)
    reference = _f64(tied).requires_grad_(True)
    reference.max().backward()

    probe = _f64(tied).requires_grad_(True)
    (got,) = lucid.autograd.grad(probe.max(), [probe], create_graph=True)
    assert np.allclose(got.numpy(), reference.grad.numpy())
    assert np.count_nonzero(got.numpy()) == 2


def test_prod_second_derivative_matches_finite_differences() -> None:
    """What the graph mode is *for* — and it was unreachable before.

    ``max`` and ``min`` are not included: their gradient is piecewise
    constant, so the true second derivative is zero and the engine
    reports the unreachable-input case, exactly as ``sum`` and ``mean``
    already did.
    """
    weight = np.array([1.3, -0.8, 0.5, 1.9, -0.4, 0.7])

    def directional(arr):
        x = _f64(arr).requires_grad_(True)
        (g,) = lucid.autograd.grad(_loss(lambda t: t.prod(), x), [x], create_graph=True)
        return x, (g.reshape(-1) * _f64(weight)).sum()

    x, scalar = directional(_BASE)
    (second,) = lucid.autograd.grad(scalar, [x])

    step = 1e-6
    flat = _BASE.reshape(-1)
    fd = np.empty(6)
    for i in range(6):
        plus, minus = flat.copy(), flat.copy()
        plus[i] += step
        minus[i] -= step
        fd[i] = (
            float(directional(plus.reshape(2, 3))[1]) - float(directional(minus.reshape(2, 3))[1])
        ) / (2 * step)

    analytic = second.numpy().reshape(-1)
    scale = max(np.abs(analytic).max(), np.abs(fd).max(), 1e-12)
    assert np.abs(analytic - fd).max() / scale < 1e-6


@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("axis", [0, 1])
def test_axis_reductions_agree_too(keepdims, axis):
    """The broadcast-back path differs per axis and per keepdims."""
    for fn in (
        lambda t: t.prod(dim=axis, keepdim=keepdims),
        lambda t: t.max(dim=axis, keepdim=keepdims).values,
        lambda t: t.min(dim=axis, keepdim=keepdims).values,
    ):
        reference = _f64(_BASE).requires_grad_(True)
        _loss(fn, reference).backward()
        probe = _f64(_BASE).requires_grad_(True)
        (got,) = lucid.autograd.grad(_loss(fn, probe), [probe], create_graph=True)
        assert np.allclose(got.numpy(), reference.grad.numpy(), atol=1e-12)
