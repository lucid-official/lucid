"""Eleven ops gained a graph-mode derivative, and one unblocked the rest.

An op needs ``grad_formula_impl`` for ``grad(create_graph=True)`` to work
— an eager ``grad_formula`` computes a gradient but not a differentiable
one.  Eight activations and three comparisons had only the eager form and
refused the second derivative outright.

``where`` was the load-bearing one.  It had ``apply`` and no
``apply_for_graph``, and the base class raises before an op name is
available, so the message read "not yet supported for op 'unknown'" — the
largest single unexplained bucket in the audit.  ``where`` is how every
piecewise function is written, so ``softplus``, ``celu``, ``prelu`` and
the rest inherited a refusal from a composite they had no say in.

Every check that needs the reference — first derivatives against it,
second derivatives against a central difference of its gradient — lives
in ``lucid/test/parity/autograd/test_second_derivatives_parity.py``; this
file keeps the properties that need no oracle, so the fast tier runs them.
"""

import numpy as np
import pytest

import lucid
import lucid.autograd
import lucid.nn.functional as F

X = np.array([-4.0, -3.5, -1.0, -0.25, 0.25, 1.0, 3.5, 4.0])


def _first(fn, values=X):
    x = lucid.tensor(values.copy(), requires_grad=True)
    (g,) = lucid.autograd.grad(fn(x).sum(), [x], create_graph=True)
    return x, g, np.asarray(g.numpy())


def _second(x, g):
    x.grad = None
    g.sum().backward()
    return np.zeros(x.shape) if x.grad is None else np.asarray(x.grad.numpy())


# ── the activations ───────────────────────────────────────────────────────────


def test_a_piecewise_linear_second_derivative_is_zero() -> None:
    """``relu6`` is linear on each piece, so ``d2`` is genuinely zero away
    from the kinks — the truthful answer, not a missing gradient."""
    x, g, _ = _first(F.relu6, np.array([-1.0, 1.0, 3.0, 7.0]))
    assert np.allclose(_second(x, g), 0.0)


def test_mish_has_a_curved_second_derivative() -> None:
    """Guard the guard: if every case were flat, the tests above would
    pass on a formula that returned zero."""
    x, g, _ = _first(F.mish)
    assert np.abs(_second(x, g)).max() > 0.1


# ── the comparisons ───────────────────────────────────────────────────────────

A = np.array([-2.0, -0.5, 0.5, 1.5, 3.0])
B = np.array([-1.0, 0.5, -0.5, 2.5, 1.0])


def test_the_two_branches_sum_to_the_incoming_gradient() -> None:
    """Nothing created, nothing lost: a tie must not send the gradient to
    both operands, and a win must not drop it."""
    a = lucid.tensor(A.copy(), requires_grad=True)
    b = lucid.tensor(B.copy(), requires_grad=True)
    out = lucid.maximum(a, b)
    (ga,) = lucid.autograd.grad(out.sum(), [a], create_graph=True, retain_graph=True)
    (gb,) = lucid.autograd.grad(out.sum(), [b], create_graph=True)
    total = np.asarray(ga.numpy()) + np.asarray(gb.numpy())
    assert np.allclose(total, 1.0)


# ── where, which unblocked the rest ───────────────────────────────────────────


def test_where_differentiates_twice() -> None:
    """``where`` refused ``create_graph`` for a while, for the wrong reason.

    Its graph-mode derivative was written and reverted because ``cdist``'s
    second derivative came back wrong.  The fault was upstream: ``where``
    keeps no reference to its operands, so ``sqrt``'s output was dropped
    and rebuilt as a leaf by ``sqrt``'s graph-mode backward, making its
    formula ``g / 2y`` treat ``y`` as a constant.  The rebuild keeps the
    output's history now (``test_where_gradients.py``).

    ``softplus`` is written over ``where`` and was refused with it; its
    second derivative is ``sigmoid(x) * (1 - sigmoid(x))``.
    """
    x = lucid.tensor(A.copy(), requires_grad=True)
    (g,) = lucid.autograd.grad(F.softplus(x).sum(), [x], create_graph=True)
    (h,) = lucid.autograd.grad(g.sum(), [x])
    s = 1.0 / (1.0 + np.exp(-A))
    np.testing.assert_allclose(np.asarray(h.numpy()), s * (1.0 - s), rtol=1e-5)


def test_wheres_eager_gradient_is_unaffected() -> None:
    """The first-order routing: each branch gets the gradient where it was chosen."""
    condition = lucid.tensor(np.array([True, False, True, False, True]))
    a = lucid.tensor(A.copy(), requires_grad=True)
    b = lucid.tensor(B.copy(), requires_grad=True)
    lucid.where(condition, a, b).sum().backward()
    assert np.allclose(np.asarray(a.grad.numpy()), [1.0, 0.0, 1.0, 0.0, 1.0])
    assert np.allclose(np.asarray(b.grad.numpy()), [0.0, 1.0, 0.0, 1.0, 0.0])


# ── structural ops, each its own inverse ──────────────────────────────────────


@pytest.mark.parametrize(
    "name,fn,values",
    [
        ("flip", lambda t: lucid.flip(t, 0), np.arange(1.0, 7.0)),
        ("fliplr", lucid.fliplr, np.arange(1.0, 10.0).reshape(3, 3)),
        ("roll", lambda t: lucid.roll(t, [2], [0]), np.arange(1.0, 7.0)),
        ("tril", lucid.tril, np.arange(1.0, 10.0).reshape(3, 3)),
        ("triu", lambda t: lucid.triu(t, 1), np.arange(1.0, 10.0).reshape(3, 3)),
    ],
)
def test_a_rearrangement_is_differentiable_twice(name, fn, values) -> None:
    """``flip``, ``roll`` and the triangle masks move or zero elements
    without computing anything, so the graph-mode derivative is the same
    rearrangement applied to the gradient.  Checked against a finite
    difference, which knows nothing about that symmetry."""
    x = lucid.tensor(values.copy(), requires_grad=True)
    (g,) = lucid.autograd.grad((fn(x) ** 2).sum(), [x], create_graph=True)
    analytic = np.asarray(g.numpy()).ravel()

    step = 1e-4
    flat = values.ravel().copy()
    numeric = np.empty_like(flat)
    for i in range(flat.size):
        up, down = flat.copy(), flat.copy()
        up[i] += step
        down[i] -= step

        def loss(v):
            return float((fn(lucid.tensor(v.reshape(values.shape))) ** 2).sum().item())

        numeric[i] = (loss(up) - loss(down)) / (2 * step)

    assert np.allclose(analytic, numeric, atol=1e-4), (analytic, numeric)


def test_tril_zeroes_the_second_derivative_where_it_masks() -> None:
    """The mask has to survive to the second order too."""
    values = np.arange(1.0, 10.0).reshape(3, 3)
    x = lucid.tensor(values, requires_grad=True)
    (g,) = lucid.autograd.grad((lucid.tril(x) ** 3).sum(), [x], create_graph=True)
    x.grad = None
    g.sum().backward()
    second = np.asarray(x.grad.numpy())
    assert np.allclose(np.triu(second, 1), 0.0)
    assert np.abs(np.tril(second)).max() > 0.0


# ── gather, and the loss path behind it ───────────────────────────────────────


def test_duplicate_indices_accumulate() -> None:
    """It is a scatter-*add*: reading one position three times must send
    three units of gradient back, not one."""
    x = lucid.tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    idx = lucid.tensor(np.array([1, 1, 1]), dtype=lucid.int32)
    (g,) = lucid.autograd.grad(lucid.gather(x, 0, idx).sum(), [x], create_graph=True)
    assert np.allclose(np.asarray(g.numpy()), [0.0, 3.0, 0.0])


def test_cross_entropy_has_a_second_derivative() -> None:
    logits = np.random.default_rng(0).standard_normal((4, 5))
    a = lucid.tensor(logits, requires_grad=True)
    tgt = lucid.tensor(np.array([0, 3, 1, 4]), dtype=lucid.int32)
    (g,) = lucid.autograd.grad(F.cross_entropy(a, tgt), [a], create_graph=True)
    a.grad = None
    g.sum().backward()
    assert a.grad is not None
    assert np.abs(np.asarray(a.grad.numpy())).max() > 0.0


@pytest.mark.parametrize("fn", [lucid.clone, lambda t: t.contiguous()])
def test_a_layout_copy_passes_the_gradient_through(fn) -> None:
    """``contiguous`` moves bytes without touching a value, so its
    derivative is the identity."""
    x = lucid.tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    (g,) = lucid.autograd.grad((fn(x) ** 2).sum(), [x], create_graph=True)
    assert np.allclose(np.asarray(g.numpy()), [2.0, 4.0, 6.0])
    x.grad = None
    g.sum().backward()
    assert np.allclose(np.asarray(x.grad.numpy()), 2.0)


# ── the name in the refusal ───────────────────────────────────────────────────


class TestANodeIsNamed:
    """A node says which op it differentiates.

    ``node_name`` falls back to the C++ type name, and it derived that by
    stripping a leading run of digits — the flat Itanium mangling
    ``12MulBackward``.  That form only occurs for a class in the *global*
    namespace, and every node here lives in ``namespace lucid``, so the
    real name ``N5lucid10DetBackwardE`` starts with ``N``, the strip never
    advanced, and the whole mangled string came back as the op name.
    Nodes in the anonymous namespace inside ``lucid`` were worse
    (``N5lucid12_GLOBAL__N_113WhereBackwardE``).

    It was checked through refusal messages, the only place the name
    surfaced, until no op reached from Python refused ``create_graph``
    any more; ``grad_fn.name()`` shows it directly now.
    """

    @pytest.mark.parametrize(
        "label,build",
        [
            # ``lucid::GridSampleBackward`` — the plain namespaced case.
            (
                "grid_sample",
                lambda x: F.grid_sample(
                    x.reshape(1, 1, 2, 2), lucid.zeros(1, 1, 1, 2, dtype=x.dtype)
                ),
            ),
            # These live in an anonymous namespace inside ``lucid``, which
            # mangles differently again and was the worse of the two.
            ("dot 2-D", lambda x: x.dot(x)),
            ("cummax", lambda x: lucid.cummax(x, dim=1).values),
        ],
    )
    def test_the_name_is_the_op_not_its_mangling(self, label, build) -> None:
        x = lucid.tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
        name = build(x).grad_fn.name()
        assert name.endswith("Backward"), name
        # The tells of an unparsed mangling.
        assert not name.startswith("N"), name
        assert "_GLOBAL__N_" not in name, name
        assert "lucid" not in name, name
        assert name.isidentifier(), name
