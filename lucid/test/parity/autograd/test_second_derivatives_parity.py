"""Graph-mode derivatives of the eleven newly covered ops, against the reference.

An op needs ``grad_formula_impl`` for ``grad(create_graph=True)`` to work
— an eager ``grad_formula`` computes a gradient but not a differentiable
one.  Eight activations and three comparisons had only the eager form and
refused the second derivative outright.

The reference cannot differentiate some of its own backward kernels
(``derivative for aten::hardsigmoid_backward is not implemented``), so it
is the arbiter for the *first* derivative only; the second is checked
against a central difference of the reference's gradient.

This file holds every check that needs the reference; the ones that need
no oracle stay in ``lucid/test/unit/autograd/test_second_derivatives.py``
so the fast tier runs them.
"""

import numpy as np
import pytest

import lucid
import lucid.autograd
import lucid.nn.functional as F

pytestmark = pytest.mark.parity

X = np.array([-4.0, -3.5, -1.0, -0.25, 0.25, 1.0, 3.5, 4.0])

ACTIVATIONS = [
    ("leaky_relu", F.leaky_relu, "leaky_relu"),
    ("elu", F.elu, "elu"),
    ("selu", F.selu, "selu"),
    ("mish", F.mish, "mish"),
    ("hardsigmoid", F.hardsigmoid, "hardsigmoid"),
    ("hardswish", F.hardswish, "hardswish"),
    ("relu6", F.relu6, "relu6"),
]


def _first(fn, values=X):
    x = lucid.tensor(values.copy(), requires_grad=True)
    (g,) = lucid.autograd.grad(fn(x).sum(), [x], create_graph=True)
    return x, g, np.asarray(g.numpy())


def _second(x, g):
    x.grad = None
    g.sum().backward()
    return np.zeros(x.shape) if x.grad is None else np.asarray(x.grad.numpy())


# ── the activations ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("name,fn,ref_name", ACTIVATIONS)
def test_first_derivative_matches_the_reference(name, fn, ref_name, ref) -> None:
    _, _, got = _first(fn)
    r = ref.from_numpy(X.copy()).requires_grad_(True)
    (rg,) = ref.autograd.grad(getattr(ref.nn.functional, ref_name)(r).sum(), [r], create_graph=True)
    assert np.allclose(got, np.asarray(rg.tolist()), atol=1e-6)


@pytest.mark.parametrize("name,fn,ref_name", ACTIVATIONS)
def test_second_derivative_matches_a_finite_difference(name, fn, ref_name, ref) -> None:
    """Of the *reference's* first derivative, so the check does not lean
    on the implementation it is checking."""
    x, g, _ = _first(fn)
    got = _second(x, g)

    def ref_first_at(values, index):
        r = ref.from_numpy(values).requires_grad_(True)
        (rg,) = ref.autograd.grad(getattr(ref.nn.functional, ref_name)(r).sum(), [r])
        return np.asarray(rg.tolist())[index]

    step = 1e-4
    numeric = np.empty_like(X)
    for i in range(X.size):
        up, down = X.copy(), X.copy()
        up[i] += step
        down[i] -= step
        numeric[i] = (ref_first_at(up, i) - ref_first_at(down, i)) / (2 * step)
    assert np.allclose(got, numeric, atol=2e-3), (got, numeric)


# ── the comparisons ───────────────────────────────────────────────────────────

A = np.array([-2.0, -0.5, 0.5, 1.5, 3.0])
B = np.array([-1.0, 0.5, -0.5, 2.5, 1.0])


@pytest.mark.parametrize(
    "name,lf,rf", [("maximum", lucid.maximum, "maximum"), ("minimum", lucid.minimum, "minimum")]
)
@pytest.mark.parametrize("wrt", ["a", "b"])
def test_comparison_first_derivative(name, lf, rf, wrt, ref) -> None:
    a = lucid.tensor(A.copy(), requires_grad=True)
    b = lucid.tensor(B.copy(), requires_grad=True)
    (g,) = lucid.autograd.grad(lf(a, b).sum(), [a if wrt == "a" else b], create_graph=True)
    ra = ref.from_numpy(A.copy()).requires_grad_(True)
    rb = ref.from_numpy(B.copy()).requires_grad_(True)
    (rg,) = ref.autograd.grad(
        getattr(ref, rf)(ra, rb).sum(), [ra if wrt == "a" else rb], create_graph=True
    )
    assert np.allclose(np.asarray(g.numpy()), np.asarray(rg.tolist()), atol=1e-8)


def test_clip_first_derivative(ref) -> None:
    x = lucid.tensor(A.copy(), requires_grad=True)
    (g,) = lucid.autograd.grad(lucid.clip(x, -1.0, 2.0).sum(), [x], create_graph=True)
    r = ref.from_numpy(A.copy()).requires_grad_(True)
    (rg,) = ref.autograd.grad(ref.clip(r, -1.0, 2.0).sum(), [r], create_graph=True)
    assert np.allclose(np.asarray(g.numpy()), np.asarray(rg.tolist()), atol=1e-8)


# ── gather, and the loss path behind it ───────────────────────────────────────


def test_gather_is_differentiable_twice(ref) -> None:
    """The adjoint of a gather is a scatter-add: each output element came
    from one input position, so the gradient goes back there."""
    values = np.arange(1.0, 13.0).reshape(3, 4)
    indices = np.array([[0, 2, 1, 3], [3, 1, 0, 0], [2, 2, 2, 1]])

    x = lucid.tensor(values.copy(), requires_grad=True)
    idx = lucid.tensor(indices, dtype=lucid.int32)
    (g,) = lucid.autograd.grad((lucid.gather(x, 1, idx) ** 2).sum(), [x], create_graph=True)

    r = ref.from_numpy(values.copy()).requires_grad_(True)
    (rg,) = ref.autograd.grad(
        (ref.gather(r, 1, ref.from_numpy(indices).long()) ** 2).sum(), [r], create_graph=True
    )
    assert np.allclose(np.asarray(g.numpy()), np.asarray(rg.tolist()))


@pytest.mark.parametrize("name", ["cross_entropy", "nll_loss"])
def test_the_classification_losses_reach_create_graph(name, ref) -> None:
    """What sixteen symbols were actually blocked on — these are training
    paths, not corners."""
    logits = np.random.default_rng(0).standard_normal((4, 5))
    target = np.array([0, 3, 1, 4])

    a = lucid.tensor(logits.copy(), requires_grad=True)
    tgt = lucid.tensor(target, dtype=lucid.int32)
    if name == "cross_entropy":
        loss = F.cross_entropy(a, tgt)
    else:
        loss = F.nll_loss(lucid.log(F.softmax(a, dim=1)), tgt)
    (g,) = lucid.autograd.grad(loss, [a], create_graph=True)

    ra = ref.from_numpy(logits.copy()).requires_grad_(True)
    rtgt = ref.from_numpy(target).long()
    if name == "cross_entropy":
        ref_loss = ref.nn.functional.cross_entropy(ra, rtgt)
    else:
        ref_loss = ref.nn.functional.nll_loss(ref.log_softmax(ra, dim=1), rtgt)
    (rg,) = ref.autograd.grad(ref_loss, [ra], create_graph=True)
    assert np.allclose(np.asarray(g.numpy()), np.asarray(rg.tolist()), atol=1e-6)
